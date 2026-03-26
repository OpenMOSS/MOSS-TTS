#!/usr/bin/env python3
"""LoRA finetuning for MOSS-TTS Norwegian data."""

import argparse
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import wandb
from accelerate import Accelerator
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset
from transformers import AutoModel, AutoTokenizer

from dataset import MossTTSDataset, collate_fn

MODEL_ID = "OpenMOSS-Team/MOSS-TTS"
MODEL_REVISION = "0c8df9988ab61071cdb06fe40b2bdc3132ac3b7e"

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
LM_HEAD_COUNT = 33


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MOSS-TTS with LoRA adapters")
    parser.add_argument(
        "--manifest-train",
        default="/root/moss-tts-norwegian/moss_tts_train.jsonl",
    )
    parser.add_argument(
        "--manifest-val",
        default="/root/moss-tts-norwegian/moss_tts_val.jsonl",
    )
    parser.add_argument(
        "--tokenized-dir",
        default="/root/moss-tts-norwegian/tokenized",
    )
    parser.add_argument(
        "--output-dir",
        default="/root/moss-tts-norwegian/checkpoints",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=250)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--wandb-project", default="moss-tts-norwegian")
    parser.add_argument("--wandb-name", default="lora-r8-no-v1")
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--resume-adapter-path", default="")
    parser.add_argument(
        "--trainable-lora-modules",
        choices=("all", "mlp", "mlp_plus_o"),
        default="all",
    )
    parser.add_argument(
        "--lm-heads-mode",
        choices=("none", "audio", "all"),
        default="none",
    )
    return parser.parse_args()


def load_model_with_attention_fallback(
    accelerator: Accelerator,
) -> Tuple[torch.nn.Module, str]:
    common_kwargs = {
        "trust_remote_code": True,
        "revision": MODEL_REVISION,
        "dtype": torch.bfloat16,
    }

    # Try flash_attention_2 first, then sdpa, then default
    for attn_impl in ("flash_attention_2", "sdpa", None):
        try:
            kwargs = dict(common_kwargs)
            if attn_impl is not None:
                kwargs["attn_implementation"] = attn_impl
            model = AutoModel.from_pretrained(MODEL_ID, **kwargs)
            return model, attn_impl or "default"
        except Exception as exc:
            if accelerator.is_main_process:
                label = attn_impl or "default"
                print(f"{label} loading failed ({exc}), trying next...")

    raise RuntimeError("Failed to load model with any attention implementation")


def load_tokenizer_if_available(accelerator: Accelerator) -> Optional[AutoTokenizer]:
    try:
        return AutoTokenizer.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            revision=MODEL_REVISION,
        )
    except Exception as exc:
        if accelerator.is_main_process:
            print(f"Tokenizer load failed (not required for this training loop): {exc}")
        return None


def build_lora_target_modules(lm_heads_mode: str) -> List[str]:
    target_modules = list(LORA_TARGET_MODULES)

    if lm_heads_mode == "audio":
        target_modules.extend([f"lm_heads.{idx}" for idx in range(1, LM_HEAD_COUNT)])
    elif lm_heads_mode == "all":
        target_modules.extend([f"lm_heads.{idx}" for idx in range(LM_HEAD_COUNT)])

    return target_modules


def apply_lora_to_language_backbone(
    model: torch.nn.Module,
    args: argparse.Namespace,
) -> Tuple[torch.nn.Module, Dict[str, int]]:
    for param in model.parameters():
        param.requires_grad = False

    target_modules = build_lora_target_modules(args.lm_heads_mode)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # Monkey-patch get_input_embeddings for PEFT compatibility.
    # PEFT calls model.get_input_embeddings() with no args during setup to find
    # the nn.Embedding layer. But MossTTSDelayModel.get_input_embeddings(input_ids)
    # requires input_ids. We temporarily patch it, let PEFT do its setup, then restore.
    original_get_input_embeddings = type(model).get_input_embeddings
    type(model).get_input_embeddings = lambda self, input_ids=None: (
        original_get_input_embeddings(self, input_ids)
        if input_ids is not None
        else self.language_model.get_input_embeddings()
    )
    if args.resume_adapter_path:
        model = PeftModel.from_pretrained(
            model,
            args.resume_adapter_path,
            is_trainable=True,
        )
    else:
        model = get_peft_model(model, lora_config)

    # Monkey-patch forward to prevent output_hidden_states duplication.
    # PEFT passes output_hidden_states via kwargs, but MossTTSDelayModel.forward()
    # doesn't have it as a named parameter, so it lands in **kwargs. The model then
    # passes output_hidden_states=True explicitly to self.language_model(**kwargs),
    # causing 'got multiple values for keyword argument'. Fix: wrap forward to pop it.
    _original_forward = type(
        model.get_base_model() if hasattr(model, "get_base_model") else model
    ).forward

    def _patched_forward(
        self, *args, output_hidden_states=None, return_dict=None, **kwargs
    ):
        return _original_forward(self, *args, **kwargs)

    base_cls = type(
        model.get_base_model() if hasattr(model, "get_base_model") else model
    )
    base_cls.forward = _patched_forward

    allowed_fragments = ["language_model.layers."]
    if args.lm_heads_mode in {"audio", "all"}:
        allowed_fragments.append("lm_heads.")

    module_substrings = {
        "all": tuple(LORA_TARGET_MODULES),
        "mlp": ("gate_proj", "up_proj", "down_proj"),
        "mlp_plus_o": ("gate_proj", "up_proj", "down_proj", "o_proj"),
    }
    allowed_lora_modules = module_substrings[args.trainable_lora_modules]

    for name, param in model.named_parameters():
        is_lora_param = "lora_" in name
        in_allowed_scope = any(fragment in name for fragment in allowed_fragments)
        in_allowed_lora_module = any(key in name for key in allowed_lora_modules)
        param.requires_grad = (
            is_lora_param and in_allowed_scope and in_allowed_lora_module
        )

        if args.lm_heads_mode == "audio" and "lm_heads.0." in name:
            param.requires_grad = False

    for name, param in model.named_parameters():
        if "emb_ext" in name:
            param.requires_grad = False

    trainable = {
        name: param.numel()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    if not trainable:
        raise RuntimeError(
            "No trainable LoRA parameters found. "
            "Confirm that target modules exist in language_model.model.layers.*"
        )

    head_trainable = [name for name in trainable if "lm_heads." in name]
    if args.lm_heads_mode == "none" and head_trainable:
        raise RuntimeError(
            f"Unexpected lm_heads LoRA params found: {head_trainable[:3]}"
        )
    if args.lm_heads_mode in {"audio", "all"} and not head_trainable:
        raise RuntimeError(
            "lm_heads-mode enabled but no lm_heads LoRA params are trainable"
        )
    if args.lm_heads_mode == "audio" and any(
        "lm_heads.0." in name for name in head_trainable
    ):
        raise RuntimeError("lm_heads.0 must remain frozen in --lm-heads-mode audio")

    if not any(any(key in name for key in allowed_lora_modules) for name in trainable):
        raise RuntimeError(
            "No trainable LoRA parameters matched --trainable-lora-modules="
            f"{args.trainable_lora_modules}"
        )

    return model, trainable


def enable_gradient_checkpointing(model: torch.nn.Module) -> None:
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    language_model = getattr(base_model, "language_model", None)

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    elif hasattr(base_model, "enable_input_require_grads"):
        base_model.enable_input_require_grads()

    if language_model is not None and hasattr(
        language_model, "gradient_checkpointing_enable"
    ):
        language_model.gradient_checkpointing_enable()
    elif hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()

    for cfg_obj in (
        getattr(model, "config", None),
        getattr(base_model, "config", None),
        getattr(language_model, "config", None) if language_model is not None else None,
    ):
        if cfg_obj is not None and hasattr(cfg_obj, "use_cache"):
            cfg_obj.use_cache = False


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    warmup_steps = max(0, warmup_steps)
    total_steps = max(1, total_steps)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        if total_steps <= warmup_steps:
            return 1.0
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def _to_tensor(value: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if torch.is_tensor(value):
        tensor = value
    else:
        tensor = torch.as_tensor(value)
    return tensor.to(device=device, dtype=dtype, non_blocking=True)


def prepare_batch(
    batch: Dict[str, Any],
    device: torch.device,
    default_channelwise_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(batch, dict):
        raise TypeError(f"Expected dict batch from collate_fn, got {type(batch)}")

    for required_key in ("input_ids", "attention_mask", "labels"):
        if required_key not in batch:
            raise KeyError(f"Batch missing required key: {required_key}")

    input_ids = _to_tensor(batch["input_ids"], device=device, dtype=torch.long)
    attention_mask = _to_tensor(
        batch["attention_mask"], device=device, dtype=torch.bool
    )
    labels = _to_tensor(batch["labels"], device=device, dtype=torch.long)

    if "channelwise_loss_weight" in batch:
        channelwise_loss_weight = _to_tensor(
            batch["channelwise_loss_weight"],
            device=device,
            dtype=torch.float32,
        )
    else:
        channelwise_loss_weight = default_channelwise_weight

    if channelwise_loss_weight.ndim == 2:
        channelwise_loss_weight = channelwise_loss_weight[0]
    if channelwise_loss_weight.ndim != 1:
        channelwise_loss_weight = channelwise_loss_weight.reshape(-1)
    if channelwise_loss_weight.numel() != 33:
        raise ValueError(
            "channelwise_loss_weight must have 33 entries, "
            f"got shape {tuple(channelwise_loss_weight.shape)}"
        )

    return input_ids, attention_mask, labels, channelwise_loss_weight


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    accelerator: Accelerator,
    default_channelwise_weight: torch.Tensor,
) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for batch in val_loader:
        input_ids, attention_mask, labels, channelwise_loss_weight = prepare_batch(
            batch,
            device=accelerator.device,
            default_channelwise_weight=default_channelwise_weight,
        )
        with accelerator.autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                channelwise_loss_weight=channelwise_loss_weight,
            )

        loss = outputs.loss.detach().float()
        gathered = accelerator.gather_for_metrics(loss.unsqueeze(0))
        total_loss += gathered.mean().item()
        total_batches += 1

    model.train()
    if total_batches == 0:
        return float("nan")
    return total_loss / total_batches


def save_adapter_checkpoint(
    model: torch.nn.Module,
    output_dir: str,
    tag: str,
    accelerator: Accelerator,
) -> None:
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return

    ckpt_dir = Path(output_dir) / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(str(ckpt_dir))


def lora_grad_stats(model: torch.nn.Module) -> Tuple[float, bool]:
    total_norm_sq = 0.0
    has_nonzero_grad = False

    for name, param in model.named_parameters():
        if "lora_" not in name or param.grad is None:
            continue
        grad = param.grad.detach().float()
        grad_norm = grad.norm(2).item()
        total_norm_sq += grad_norm * grad_norm
        if not has_nonzero_grad and torch.count_nonzero(grad).item() > 0:
            has_nonzero_grad = True

    return math.sqrt(total_norm_sq), has_nonzero_grad


def main() -> None:
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    accelerator = Accelerator(mixed_precision="bf16")

    if accelerator.is_main_process:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    tokenizer = load_tokenizer_if_available(accelerator)
    model, attn_impl = load_model_with_attention_fallback(accelerator)
    model, trainable_map = apply_lora_to_language_backbone(model, args)
    enable_gradient_checkpointing(model)

    train_dataset = MossTTSDataset(
        manifest_path=args.manifest_train,
        tokenized_dir=args.tokenized_dir,
        max_seq_len=args.max_seq_len,
    )
    val_dataset = MossTTSDataset(
        manifest_path=args.manifest_val,
        tokenized_dir=args.tokenized_dir,
        max_seq_len=args.max_seq_len,
    )

    if args.smoke_test:
        train_limit = min(128, len(train_dataset))
        val_limit = min(128, len(val_dataset))
        train_dataset = Subset(train_dataset, range(train_limit))
        val_dataset = Subset(val_dataset, range(val_limit))
        args.warmup_steps = min(args.warmup_steps, 5)

    if len(train_dataset) == 0:
        raise RuntimeError("Training dataset is empty")
    if len(val_dataset) == 0:
        raise RuntimeError("Validation dataset is empty")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_fn,
        drop_last=False,
    )

    num_update_steps_per_epoch = max(1, math.ceil(len(train_loader) / args.grad_accum))
    max_train_steps = args.epochs * num_update_steps_per_epoch
    if args.max_train_steps > 0:
        max_train_steps = args.max_train_steps
    if args.smoke_test:
        max_train_steps = min(max_train_steps, 20)

    if max_train_steps <= 0:
        raise RuntimeError("max_train_steps resolved to zero")

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = build_lr_scheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=max_train_steps,
    )

    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        val_loader,
        lr_scheduler,
    )

    default_channelwise_weight = torch.ones(
        33, dtype=torch.float32, device=accelerator.device
    )

    wandb_run = None
    if accelerator.is_main_process:
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config={
                "model_id": MODEL_ID,
                "revision": MODEL_REVISION,
                "attn_implementation": attn_impl,
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "max_grad_norm": args.max_grad_norm,
                "warmup_steps": args.warmup_steps,
                "num_epochs": args.epochs,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.grad_accum,
                "save_steps": args.save_steps,
                "eval_steps": args.eval_steps,
                "log_steps": args.log_steps,
                "max_seq_len": args.max_seq_len,
                "smoke_test": args.smoke_test,
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "max_train_steps": max_train_steps,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "lm_heads_mode": args.lm_heads_mode,
                "resume_adapter_path": args.resume_adapter_path,
                "trainable_lora_modules": args.trainable_lora_modules,
            },
        )

        total_trainable = sum(trainable_map.values())
        print(f"Using device: {accelerator.device}")
        print(f"Attention implementation: {attn_impl}")
        print(f"Tokenizer loaded: {tokenizer is not None}")
        print(
            f"Trainable LoRA tensors: {len(trainable_map)}, parameters: {total_trainable}"
        )
        for name in list(trainable_map.keys())[:6]:
            print(f"  trainable: {name}")
        if len(trainable_map) > 6:
            print("  ...")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    start_time = time.time()
    global_step = 0
    micro_step = 0
    accum_loss_sum = 0.0
    log_loss_sum = 0.0
    log_count = 0
    smoke_losses = []
    saw_nonzero_lora_grad = False

    for epoch in range(args.epochs):
        if global_step >= max_train_steps:
            break

        for batch in train_loader:
            input_ids, attention_mask, labels, channelwise_loss_weight = prepare_batch(
                batch,
                device=accelerator.device,
                default_channelwise_weight=default_channelwise_weight,
            )

            with accelerator.autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    channelwise_loss_weight=channelwise_loss_weight,
                )
                loss = outputs.loss

            scaled_loss = loss / args.grad_accum
            accelerator.backward(scaled_loss)

            accum_loss_sum += loss.detach().float().item()
            micro_step += 1

            if micro_step % args.grad_accum != 0:
                continue

            grad_norm_raw = accelerator.clip_grad_norm_(
                model.parameters(), args.max_grad_norm
            )
            lora_grad_norm, has_nonzero_grad = lora_grad_stats(model)
            saw_nonzero_lora_grad = saw_nonzero_lora_grad or has_nonzero_grad

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            step_loss = accum_loss_sum / args.grad_accum
            accum_loss_sum = 0.0
            log_loss_sum += step_loss
            log_count += 1
            smoke_losses.append(step_loss)

            grad_norm = (
                grad_norm_raw.item()
                if torch.is_tensor(grad_norm_raw)
                else float(grad_norm_raw)
            )

            if global_step % args.log_steps == 0:
                mean_loss = log_loss_sum / max(1, log_count)
                current_lr = lr_scheduler.get_last_lr()[0]
                if accelerator.is_main_process:
                    print(
                        f"step={global_step} epoch={epoch + 1} "
                        f"loss={mean_loss:.6f} lr={current_lr:.2e} "
                        f"grad_norm={grad_norm:.4f} lora_grad_norm={lora_grad_norm:.4f}"
                    )
                    if wandb_run is not None:
                        wandb.log(
                            {
                                "train/loss": mean_loss,
                                "train/learning_rate": current_lr,
                                "train/grad_norm": grad_norm,
                                "train/lora_grad_norm": lora_grad_norm,
                                "train/epoch": epoch + 1,
                            },
                            step=global_step,
                        )
                log_loss_sum = 0.0
                log_count = 0

            if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                val_loss = evaluate(
                    model=model,
                    val_loader=val_loader,
                    accelerator=accelerator,
                    default_channelwise_weight=default_channelwise_weight,
                )
                if accelerator.is_main_process:
                    print(f"step={global_step} val_loss={val_loss:.6f}")
                    if wandb_run is not None:
                        wandb.log({"val/loss": val_loss}, step=global_step)

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                save_adapter_checkpoint(
                    model=model,
                    output_dir=args.output_dir,
                    tag=f"step_{global_step}",
                    accelerator=accelerator,
                )

            if global_step >= max_train_steps:
                break

    save_adapter_checkpoint(
        model=model,
        output_dir=args.output_dir,
        tag="final",
        accelerator=accelerator,
    )

    elapsed = time.time() - start_time

    if args.smoke_test:
        if not saw_nonzero_lora_grad:
            raise RuntimeError("Smoke test failed: LoRA gradients are all zero")
        if len(smoke_losses) < 2:
            raise RuntimeError(
                "Smoke test failed: fewer than 2 optimizer steps completed"
            )

        window = min(3, len(smoke_losses))
        first_loss = sum(smoke_losses[:window]) / window
        last_loss = sum(smoke_losses[-window:]) / window
        if last_loss > first_loss * 1.1:
            raise RuntimeError(
                "Smoke test failed: loss increased significantly "
                f"(first={first_loss:.6f}, last={last_loss:.6f}, ratio={last_loss / first_loss:.4f})"
            )
        elif last_loss >= first_loss:
            if accelerator.is_main_process:
                print(
                    f"Note: loss did not decrease in {len(smoke_losses)} steps "
                    f"(first={first_loss:.6f}, last={last_loss:.6f}). "
                    f"This is expected for short smoke tests on 8B models."
                )

        if torch.cuda.is_available() and accelerator.is_main_process:
            peak_allocated = torch.cuda.max_memory_allocated() / (1024**3)
            peak_reserved = torch.cuda.max_memory_reserved() / (1024**3)
            current_allocated = torch.cuda.memory_allocated() / (1024**3)
            print(
                "Smoke test VRAM "
                f"current_allocated={current_allocated:.2f}GB "
                f"peak_allocated={peak_allocated:.2f}GB "
                f"peak_reserved={peak_reserved:.2f}GB"
            )

    if accelerator.is_main_process:
        print(f"Training finished at step {global_step} in {elapsed:.1f}s")

    if wandb_run is not None and accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()
