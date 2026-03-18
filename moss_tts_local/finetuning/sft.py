# Copyright 2026 OpenMOSS team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import importlib.util
import json
import math
import shutil
import sys
import time
from contextlib import contextmanager, nullcontext
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedType, enable_fsdp_ram_efficient_loading, set_seed
from accelerate.utils.dataclasses import DistributedDataParallelKwargs
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, get_scheduler
from transformers.utils import cached_file

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moss_tts_local.finetuning.common import load_jsonl_spec, normalize_audio_path_list
from moss_tts_local.finetuning.dataset import MossTTSSFTDataset
from moss_tts_local.modeling_moss_tts import MossTTSDelayModel
from moss_tts_local.processing_moss_tts import MossTTSDelayProcessor


SCHEDULER_CHOICES = (
    "linear",
    "cosine",
    "cosine_with_restarts",
    "polynomial",
    "constant",
    "constant_with_warmup",
    "inverse_sqrt",
)

SUPPORT_FILES = (
    REPO_ROOT / "moss_tts_local" / "__init__.py",
    REPO_ROOT / "moss_tts_local" / "configuration_moss_tts.py",
    REPO_ROOT / "moss_tts_local" / "modeling_moss_tts.py",
    REPO_ROOT / "moss_tts_local" / "processing_moss_tts.py",
    REPO_ROOT / "moss_tts_local" / "inference_utils.py",
)

INFERENCE_ASSET_FILES = (
    "__init__.py",
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)

BASE_PROCESSOR_CONFIG = {
    "processor_class": "MossTTSDelayProcessor",
    "auto_map": {
        "AutoProcessor": "processing_moss_tts.MossTTSDelayProcessor",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supervised finetuning for MossTTSLocal-family tasks."
    )
    parser.add_argument("--model-path", type=str, default="OpenMOSS-Team/MOSS-TTS-Local-Transformer")
    parser.add_argument("--codec-path", type=str, default="OpenMOSS-Team/MOSS-Audio-Tokenizer")
    parser.add_argument(
        "--train-jsonl",
        type=str,
        required=True,
        help="Supports a single JSONL, a directory, a glob, or a comma-separated list of JSONL files.",
    )
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler-type", type=str, default="linear", choices=SCHEDULER_CHOICES)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--mixed-precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--attn-implementation", type=str, default="auto")
    parser.add_argument("--audio-tokenizer-device", type=str, default=None)
    parser.add_argument("--n-vq", type=int, default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--channelwise-loss-weight",
        type=str,
        default="1,32",
        help=(
            "Comma-separated loss weights. Use either n_vq+1 values "
            "(text_head,vq0,...,vqN) or two values "
            "(text_weight,total_audio_weight). When two values are given, "
            "the total audio weight is evenly distributed across all audio heads."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def configure_torch_backends() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)


def resolve_torch_dtype(mixed_precision: str) -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def resolve_attn_implementation(requested: str, dtype: torch.dtype) -> str:
    if requested != "auto":
        return requested
    if not torch.cuda.is_available():
        return "eager"
    if (
        importlib.util.find_spec("flash_attn") is not None
        and dtype in {torch.float16, torch.bfloat16}
    ):
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return "flash_attention_2"
    return "sdpa"


def format_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float) -> str:
    return str(timedelta(seconds=max(0, int(seconds))))


def resolve_warmup_steps(args: argparse.Namespace, num_training_steps: int) -> int:
    if args.warmup_steps > 0:
        return args.warmup_steps
    if args.warmup_ratio > 0:
        return math.ceil(num_training_steps * args.warmup_ratio)
    return 0


def parse_channelwise_loss_weight(spec: Optional[str], n_heads: int) -> Optional[List[float]]:
    if spec is None:
        return None

    values = [float(item.strip()) for item in spec.split(",") if item.strip()]
    if not values:
        return None
    if len(values) == n_heads:
        resolved = values
    elif len(values) == 2 and n_heads > 1:
        text_weight, total_audio_weight = values
        per_audio_weight = total_audio_weight / max(1, n_heads - 1)
        resolved = [text_weight] + [per_audio_weight] * (n_heads - 1)
    else:
        raise ValueError(
            f"`channelwise_loss_weight` expects either {n_heads} values or 2 values "
            f"(text_weight,total_audio_weight), got {len(values)}."
        )
    if sum(resolved) <= 0:
        raise ValueError("`channelwise_loss_weight` must sum to a positive value.")
    return resolved


def validate_args(args: argparse.Namespace) -> None:
    if args.per_device_batch_size <= 0:
        raise ValueError("`per_device_batch_size` must be > 0.")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("`gradient_accumulation_steps` must be > 0.")
    if args.learning_rate <= 0:
        raise ValueError("`learning_rate` must be > 0.")
    if args.weight_decay < 0:
        raise ValueError("`weight_decay` must be >= 0.")
    if args.warmup_steps < 0:
        raise ValueError("`warmup_steps` must be >= 0.")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("`warmup_ratio` must be in [0, 1).")
    if args.num_epochs <= 0:
        raise ValueError("`num_epochs` must be > 0.")
    if args.max_train_steps is not None and args.max_train_steps <= 0:
        raise ValueError("`max_train_steps` must be > 0 when provided.")
    if args.max_grad_norm < 0:
        raise ValueError("`max_grad_norm` must be >= 0.")
    if args.logging_steps <= 0:
        raise ValueError("`logging_steps` must be > 0.")
    if args.num_workers < 0:
        raise ValueError("`num_workers` must be >= 0.")
    if args.n_vq is not None and args.n_vq <= 0:
        raise ValueError("`n_vq` must be > 0 when provided.")


def processor_needs_audio_tokenizer(records: List[Dict[str, Any]]) -> bool:
    for record in records:
        ref_audio = normalize_audio_path_list(record.get("ref_audio"), "ref_audio")
        if record.get("ref_audio_codes") is None and ref_audio is not None:
            return True
        if record.get("reference_audio_codes") is None:
            reference = normalize_audio_path_list(record.get("reference"), "reference", allow_none=True)
            if reference is not None and any(item is not None for item in reference):
                return True
            reference_audio = normalize_audio_path_list(record.get("reference_audio"), "reference_audio")
            if reference_audio is not None:
                return True
    return False


def build_processor(
    model_path: str,
    codec_path: str,
    need_audio_tokenizer: bool,
    audio_tokenizer_device: Optional[str],
    default_audio_tokenizer_device: str,
):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = MossTTSDelayProcessor(
        tokenizer=tokenizer,
        audio_tokenizer=None,
        model_config=config,
    )

    if need_audio_tokenizer:
        processor = MossTTSDelayProcessor.from_pretrained(
            model_path,
            codec_path=codec_path,
        )
        device = audio_tokenizer_device or default_audio_tokenizer_device
        processor.audio_tokenizer = processor.audio_tokenizer.to(device)

    return processor


@contextmanager
def processor_init_context(accelerator: Accelerator):
    if accelerator.distributed_type != DistributedType.DEEPSPEED:
        yield
        return

    plugin = accelerator.state.deepspeed_plugin
    if plugin is None or not plugin.is_zero3_init_enabled():
        yield
        return

    import deepspeed

    with plugin.zero3_init_context_manager(enable=False):
        deepspeed.zero.partition_parameters.shutdown_init_context()
        try:
            yield
        finally:
            deepspeed.zero.partition_parameters.restore_init_context()


def model_init_context(accelerator: Accelerator):
    if accelerator.distributed_type == DistributedType.FSDP:
        enable_fsdp_ram_efficient_loading()

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        plugin = accelerator.state.deepspeed_plugin
        if plugin is not None and plugin.is_zero3_init_enabled():
            return plugin.zero3_init_context_manager(enable=True)

    return nullcontext()


def copy_support_files(output_dir: Path) -> None:
    for src in SUPPORT_FILES:
        if src.exists():
            shutil.copy2(src, output_dir / src.name)


def resolve_inference_asset(model_path: str, filename: str) -> Optional[Path]:
    model_path_obj = Path(model_path)
    if model_path_obj.is_dir():
        candidate = model_path_obj / filename
        return candidate if candidate.exists() else None

    try:
        resolved = cached_file(
            model_path,
            filename,
            _raise_exceptions_for_missing_entries=False,
        )
    except OSError:
        return None

    if resolved is None:
        return None
    return Path(resolved)


def copy_inference_assets(model_path: str, codec_path: str, output_dir: Path) -> None:
    for filename in INFERENCE_ASSET_FILES:
        src = resolve_inference_asset(model_path, filename)
        if src is not None and src.exists():
            shutil.copy2(src, output_dir / filename)

    processor_config = dict(BASE_PROCESSOR_CONFIG)
    processor_config["audio_tokenizer_name_or_path"] = codec_path
    with open(output_dir / "processor_config.json", "w", encoding="utf-8") as f:
        json.dump(processor_config, f, indent=2, ensure_ascii=False)


def save_checkpoint(
    accelerator: Accelerator,
    model: MossTTSDelayModel,
    model_path: str,
    codec_path: str,
    output_dir: Path,
    train_args: Dict[str, Any],
) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    state_dict = accelerator.get_state_dict(model)
    unwrapped_model = accelerator.unwrap_model(model)

    unwrapped_model.save_pretrained(
        output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=state_dict,
        safe_serialization=True,
    )
    if accelerator.is_main_process:
        copy_support_files(output_dir)
        copy_inference_assets(model_path, codec_path, output_dir)
        with open(output_dir / "finetune_args.json", "w", encoding="utf-8") as f:
            json.dump(train_args, f, indent=2, ensure_ascii=False)
    accelerator.wait_for_everyone()


def main() -> None:
    args = parse_args()
    validate_args(args)
    configure_torch_backends()
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[ddp_kwargs],
    )
    set_seed(args.seed, device_specific=True)
    global_micro_batch_size = args.per_device_batch_size * accelerator.num_processes
    global_batch_size = global_micro_batch_size * args.gradient_accumulation_steps
    global_batch_formula = (
        f"{args.per_device_batch_size} (per_device_batch_size) x "
        f"{accelerator.num_processes} (num_processes) x "
        f"{args.gradient_accumulation_steps} (gradient_accumulation_steps) = "
        f"{global_batch_size}"
    )
    accelerator.print(f"[{format_timestamp()}] [sft] global_batch_size={global_batch_formula}")

    train_paths, records = load_jsonl_spec(args.train_jsonl)
    if not records:
        raise ValueError(f"No records found in {args.train_jsonl}.")
    accelerator.print(
        f"[{format_timestamp()}] [sft] distributed_type={accelerator.distributed_type} "
        f"num_processes={accelerator.num_processes} "
        f"train_files={len(train_paths)} train_records={len(records)}"
    )

    need_audio_tokenizer = processor_needs_audio_tokenizer(records)
    if need_audio_tokenizer:
        accelerator.print(
            f"[{format_timestamp()}] [sft] found records without precomputed reference audio codes; "
            "keeping audio_tokenizer for on-the-fly reference encoding."
        )
    with processor_init_context(accelerator):
        processor = build_processor(
            model_path=args.model_path,
            codec_path=args.codec_path,
            need_audio_tokenizer=need_audio_tokenizer,
            audio_tokenizer_device=args.audio_tokenizer_device,
            default_audio_tokenizer_device=str(accelerator.device),
        )

    dataset = MossTTSSFTDataset(
        records=records,
        processor=processor,
        n_vq=args.n_vq,
    )

    if getattr(processor, "audio_tokenizer", None) is not None and not need_audio_tokenizer:
        processor.audio_tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model_dtype = resolve_torch_dtype(args.mixed_precision)
    attn_implementation = resolve_attn_implementation(args.attn_implementation, model_dtype)

    with model_init_context(accelerator):
        model = MossTTSDelayModel.from_pretrained(
            args.model_path,
            torch_dtype=model_dtype,
            attn_implementation=attn_implementation,
        )
    resolved_channelwise_loss_weight = parse_channelwise_loss_weight(
        args.channelwise_loss_weight,
        model.config.n_vq + 1,
    )
    accelerator.print(
        f"[{format_timestamp()}] [sft] resolved channelwise_loss_weight="
        f"{resolved_channelwise_loss_weight}"
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    train_dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
    )

    micro_batches_per_epoch = math.ceil(len(records) / global_micro_batch_size)
    update_steps_per_epoch = math.ceil(micro_batches_per_epoch / args.gradient_accumulation_steps)
    max_train_steps = args.max_train_steps or (args.num_epochs * update_steps_per_epoch)
    warmup_steps = resolve_warmup_steps(args, max_train_steps)
    accelerator.print(
        f"[{format_timestamp()}] [sft] scheduler={args.lr_scheduler_type} "
        f"warmup_steps={warmup_steps} "
        f"micro_batches_per_epoch={micro_batches_per_epoch} "
        f"optimizer_steps_per_epoch={update_steps_per_epoch} "
        f"max_train_steps={max_train_steps}"
    )
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    output_root = Path(args.output_dir)
    if accelerator.is_main_process:
        output_root.mkdir(parents=True, exist_ok=True)

    global_step = 0
    completed_epochs = 0
    last_log_time = time.perf_counter()
    last_logged_step = 0
    train_args_to_save = vars(args).copy()
    train_args_to_save["global_batch_size"] = global_batch_size
    train_args_to_save["global_batch_size_formula"] = global_batch_formula
    train_args_to_save["micro_batches_per_epoch"] = micro_batches_per_epoch
    train_args_to_save["optimizer_steps_per_epoch"] = update_steps_per_epoch
    train_args_to_save["resolved_warmup_steps"] = warmup_steps
    train_args_to_save["resolved_channelwise_loss_weight"] = resolved_channelwise_loss_weight

    for epoch in range(args.num_epochs):
        model.train()
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    channelwise_loss_weight=resolved_channelwise_loss_weight,
                )
                loss = outputs.loss
                accelerator.backward(loss)

                if accelerator.sync_gradients and args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                if global_step % args.logging_steps == 0:
                    now = time.perf_counter()
                    steps_since_last_log = max(global_step - last_logged_step, 1)
                    elapsed = max(now - last_log_time, 1e-12)
                    last_log_time = now
                    last_logged_step = global_step
                    step_time = elapsed / steps_since_last_log
                    steps_per_sec = steps_since_last_log / elapsed
                    samples_per_sec = (global_batch_size * steps_since_last_log) / elapsed
                    eta_seconds = max(max_train_steps - global_step, 0) / steps_per_sec
                    logged_loss = accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                    accelerator.print(
                        f"[{format_timestamp()}] "
                        f"epoch={epoch} step={global_step}/{max_train_steps} "
                        f"loss={logged_loss:.4f} "
                        f"lr={lr_scheduler.get_last_lr()[0]:.2e} "
                        f"step_time={step_time:.2f}s "
                        f"steps_per_sec={steps_per_sec:.3f} "
                        f"samples_per_sec={samples_per_sec:.2f} "
                        f"eta={format_duration(eta_seconds)}"
                    )

                if global_step >= max_train_steps:
                    break

        checkpoint_dir = output_root / f"checkpoint-epoch-{epoch}"
        save_checkpoint(
            accelerator=accelerator,
            model=model,
            model_path=args.model_path,
            codec_path=args.codec_path,
            output_dir=checkpoint_dir,
            train_args=train_args_to_save,
        )
        completed_epochs = epoch + 1

        if global_step >= max_train_steps:
            break

    accelerator.print(
        f"[{format_timestamp()}] Finished training: "
        f"global_step={global_step}, saved_epochs={completed_epochs}, output_dir={output_root}"
    )


if __name__ == "__main__":
    main()
