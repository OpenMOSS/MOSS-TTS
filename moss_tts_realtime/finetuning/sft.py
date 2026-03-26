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
import re
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
from transformers import AutoTokenizer, get_scheduler
from transformers.utils import cached_file

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moss_tts_realtime.finetuning.common import load_jsonl, normalize_audio_path_list, resolve_jsonl_paths
from moss_tts_realtime.finetuning.dataset import MossTTSRealtimeSFTDataset
from moss_tts_realtime.mossttsrealtime.modeling_mossttsrealtime import MossTTSRealtime
from moss_tts_realtime.mossttsrealtime.processing_mossttsrealtime import MossTTSRealtimeProcessor


SCHEDULER_CHOICES = (
    "linear",
    "cosine",
    "cosine_with_restarts",
    "polynomial",
    "constant",
    "constant_with_warmup",
    "inverse_sqrt",
)

# The entire mossttsrealtime package is copied into each checkpoint directory
# so the checkpoint is self-contained and loadable without the full repository.
MOSSTTSREALTIME_PACKAGE = REPO_ROOT / "moss_tts_realtime" / "mossttsrealtime"

INFERENCE_ASSET_FILES = (
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)

BASE_PROCESSOR_CONFIG = {
    "processor_class": "MossTTSRealtimeProcessor",
    "auto_map": {
        "AutoProcessor": "mossttsrealtime.processing_mossttsrealtime.MossTTSRealtimeProcessor",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supervised finetuning for MossTTSRealtime."
    )
    parser.add_argument("--model-path", type=str, default="OpenMOSS-Team/MOSS-TTS-Realtime")
    parser.add_argument(
        "--codec-path", type=str, default="OpenMOSS-Team/MOSS-Audio-Tokenizer",
        help="Codec model path. Not loaded during training; stored in finetune_args.json for traceability.",
    )
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
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="If set, log metrics to Weights & Biases (main process only). Requires: pip install wandb",
    )
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument(
        "--wandb-tags",
        type=str,
        default=None,
        help="Comma-separated tags for the W&B run.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--mixed-precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--attn-implementation", type=str, default="auto")
    parser.add_argument(
        "--n-vq", type=int, default=None,
        help="Number of RVQ codebooks to use (default: all 16).",
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


def validate_records(records: List[Dict[str, Any]]) -> None:
    """Ensure every record has the required fields (conversations format)."""
    for i, record in enumerate(records):
        if "conversations" not in record:
            raise ValueError(
                f"Record {i} is missing `conversations`.\n"
                f"  Record keys: {list(record.keys())}"
            )
        convs = record["conversations"]
        if not isinstance(convs, list) or len(convs) == 0:
            raise ValueError(f"Record {i}: `conversations` must be a non-empty list.")
        for j, turn in enumerate(convs):
            if "audio_codes" not in turn:
                raise ValueError(
                    f"Record {i}, turn {j} is missing `audio_codes`. "
                    "Run prepare_data.py first."
                )
        if record.get("ref_wav") is not None and record.get("ref_audio_codes") is None:
            raise ValueError(
                f"Record {i} has `ref_wav` but no `ref_audio_codes`. "
                "Run prepare_data.py first."
            )


def build_processor(model_path: str) -> MossTTSRealtimeProcessor:
    """Load tokenizer from model_path and wrap in MossTTSRealtimeProcessor."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return MossTTSRealtimeProcessor(tokenizer=tokenizer)


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
    """Copy the entire mossttsrealtime Python package into output_dir/mossttsrealtime/.

    This makes the checkpoint self-contained: users can load it by prepending
    ``output_dir`` to ``sys.path``.
    """
    dst = output_dir / "mossttsrealtime"
    dst.mkdir(parents=True, exist_ok=True)
    for src in MOSSTTSREALTIME_PACKAGE.glob("*.py"):
        shutil.copy2(src, dst / src.name)


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


def copy_inference_assets(model_path: str, output_dir: Path) -> None:
    """Copy tokenizer files and write processor_config.json."""
    for filename in INFERENCE_ASSET_FILES:
        src = resolve_inference_asset(model_path, filename)
        if src is not None and src.exists():
            shutil.copy2(src, output_dir / filename)

    with open(output_dir / "processor_config.json", "w", encoding="utf-8") as f:
        json.dump(BASE_PROCESSOR_CONFIG, f, indent=2, ensure_ascii=False)


def save_checkpoint(
    accelerator: Accelerator,
    model: MossTTSRealtime,
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
        copy_inference_assets(model_path, output_dir)
        with open(output_dir / "finetune_args.json", "w", encoding="utf-8") as f:
            json.dump(train_args, f, indent=2, ensure_ascii=False)
    accelerator.wait_for_everyone()


def shard_paths_for_rank(paths: List[Path], world_size: int, rank: int) -> tuple[List[Path], bool]:
    if world_size <= 1:
        return paths, False

    shard_pattern = re.compile(r"\.rank(\d+)-of-(\d+)\.jsonl$")
    parsed: List[tuple[Path, int, int]] = []
    for path in paths:
        match = shard_pattern.search(path.name)
        if match is None:
            return paths, False
        shard_rank = int(match.group(1))
        shard_world_size = int(match.group(2))
        parsed.append((path, shard_rank, shard_world_size))

    shard_world_sizes = {item[2] for item in parsed}
    if len(shard_world_sizes) != 1:
        return paths, False

    selected = [path for path, shard_rank, _ in parsed if shard_rank % world_size == rank]
    if not selected:
        raise ValueError(
            f"No shard assigned for rank={rank} world_size={world_size}. "
            "Please check --train-jsonl shard files and distributed config."
        )
    return selected, True


def load_jsonl_for_rank(
    spec: str,
    world_size: int,
    rank: int,
) -> tuple[List[Path], List[Dict[str, Any]], List[Path], bool]:
    all_paths = resolve_jsonl_paths(spec)
    rank_paths, using_pre_sharded_files = shard_paths_for_rank(
        all_paths, world_size=world_size, rank=rank
    )
    records: List[Dict[str, Any]] = []
    for path in rank_paths:
        records.extend(load_jsonl(path))
    return all_paths, records, rank_paths, using_pre_sharded_files


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

    train_paths, records, local_train_paths, using_pre_sharded_files = load_jsonl_for_rank(
        args.train_jsonl,
        world_size=accelerator.num_processes,
        rank=accelerator.process_index,
    )
    if not records:
        raise ValueError(f"No records found in {args.train_jsonl}.")
    accelerator.print(
        f"[{format_timestamp()}] [sft] distributed_type={accelerator.distributed_type} "
        f"num_processes={accelerator.num_processes} "
        f"using_pre_sharded_files={using_pre_sharded_files} "
        f"train_files={len(train_paths)} local_train_files={len(local_train_paths)} "
        f"local_train_records={len(records)}"
    )

    # Fail fast if any record is missing pre-computed audio codes.
    # Run prepare_data.py first to encode all audio files.
    validate_records(records)

    with processor_init_context(accelerator):
        processor = build_processor(model_path=args.model_path)

    dataset = MossTTSRealtimeSFTDataset(
        records=records,
        processor=processor,
        n_vq=args.n_vq,
    )

    model_dtype = resolve_torch_dtype(args.mixed_precision)
    attn_implementation = resolve_attn_implementation(args.attn_implementation, model_dtype)

    with model_init_context(accelerator):
        model = MossTTSRealtime.from_pretrained(
            args.model_path,
            torch_dtype=model_dtype,
            attn_implementation=attn_implementation,
        )

    model.language_model.embed_tokens.weight.requires_grad = False

    accelerator.print(
        f"[{format_timestamp()}] [sft] attn_implementation={attn_implementation} "
        f"model_dtype={model_dtype} rvq={model.config.rvq}"
    )


    train_dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        # In pre-sharded mode the dataloader is not wrapped by Accelerate, so
        # there is no DistributedSampler to pad shorter shards.  Drop the last
        # incomplete batch to guarantee every rank runs the same number of
        # steps and avoids a DDP allreduce hang when shard sizes differ by one.
        drop_last=using_pre_sharded_files,
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

    if using_pre_sharded_files:
        # Inputs are already split by rank; do not divide by world_size again.
        micro_batches_per_epoch = math.ceil(len(records) / args.per_device_batch_size)
    else:
        # A full/global dataset is loaded per process and then sharded by Accelerate.
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

    if using_pre_sharded_files:
        model, optimizer, lr_scheduler = accelerator.prepare(
            model,
            optimizer,
            lr_scheduler,
        )
    else:
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model,
            optimizer,
            train_dataloader,
            lr_scheduler,
        )

    output_root = Path(args.output_dir)
    if accelerator.is_main_process:
        output_root.mkdir(parents=True, exist_ok=True)

    train_args_to_save = vars(args).copy()
    train_args_to_save["global_batch_size"] = global_batch_size
    train_args_to_save["global_batch_size_formula"] = global_batch_formula
    train_args_to_save["micro_batches_per_epoch"] = micro_batches_per_epoch
    train_args_to_save["optimizer_steps_per_epoch"] = update_steps_per_epoch
    train_args_to_save["resolved_warmup_steps"] = warmup_steps

    wandb_module = None
    if args.wandb_project and accelerator.is_main_process:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "wandb is not installed. Install with: pip install wandb"
            ) from exc
        init_kwargs: Dict[str, Any] = {
            "project": args.wandb_project,
            "config": train_args_to_save,
        }
        if args.wandb_entity:
            init_kwargs["entity"] = args.wandb_entity
        if args.wandb_run_name:
            init_kwargs["name"] = args.wandb_run_name
        if args.wandb_tags:
            init_kwargs["tags"] = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        wandb.init(**init_kwargs)
        wandb_module = wandb

    try:
        _training_loop(
            accelerator,
            args,
            model,
            train_dataloader,
            optimizer,
            lr_scheduler,
            max_train_steps,
            global_batch_size,
            output_root,
            train_args_to_save,
            wandb_module,
        )
    finally:
        if wandb_module is not None:
            wandb_module.finish()


def _training_loop(
    accelerator: Accelerator,
    args: argparse.Namespace,
    model: MossTTSRealtime,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    max_train_steps: int,
    global_batch_size: int,
    output_root: Path,
    train_args_to_save: Dict[str, Any],
    wandb_module: Optional[Any],
) -> None:
    global_step = 0
    completed_epochs = 0
    last_log_time = time.perf_counter()
    last_logged_step = 0

    for epoch in range(args.num_epochs):
        model.train()
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                # input_ids      : [B, T, 17]  (1 text + 16 audio channels)
                # attention_mask : [B, T]       (bool, False at right-padded positions)
                # labels         : [B, T, 17]   (-100 at non-audio / padded positions)
                # Loss is returned directly from local_transformer; no channelwise
                # weighting is applied (the backbone has no separate LM head).
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
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
                    lr_val = lr_scheduler.get_last_lr()[0]
                    accelerator.print(
                        f"[{format_timestamp()}] "
                        f"epoch={epoch} step={global_step}/{max_train_steps} "
                        f"loss={logged_loss:.4f} "
                        f"lr={lr_val:.2e} "
                        f"step_time={step_time:.2f}s "
                        f"steps_per_sec={steps_per_sec:.3f} "
                        f"samples_per_sec={samples_per_sec:.2f} "
                        f"eta={format_duration(eta_seconds)}"
                    )
                    if wandb_module is not None:
                        wandb_module.log(
                            {
                                "train/loss": logged_loss,
                                "train/lr": lr_val,
                                "train/step_time": step_time,
                                "train/steps_per_sec": steps_per_sec,
                                "train/samples_per_sec": samples_per_sec,
                                "train/epoch": epoch,
                            },
                            step=global_step,
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
