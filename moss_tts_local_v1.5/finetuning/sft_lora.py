"""LoRA finetuning entry point for MOSS-TTS Local Transformer v1.5.

This module deliberately reuses the data pipeline, supervised loss, gradient
checkpointing, and training loop from ``sft.py``.  It only adds PEFT adapters,
an optimizer selector, and adapter-only checkpoint saving.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

import torch

import sft


GLOBAL_LORA_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
LOCAL_LORA_MODULES = (
    "c_attn",
    "c_proj",
    "fc_in",
    "fc_out",
)


def parse_lora_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    group = parser.add_argument_group("LoRA options")
    group.add_argument("--lora-r", type=int, default=8)
    group.add_argument("--lora-alpha", type=int, default=16)
    group.add_argument("--lora-dropout", type=float, default=0.05)
    group.add_argument(
        "--lora-resume-adapter",
        type=str,
        default=None,
        help="Existing PEFT adapter directory to load and continue training.",
    )
    group.add_argument(
        "--optimizer",
        choices=["adamw", "adamw_8bit", "paged_adamw_8bit"],
        default="adamw",
    )
    group.add_argument(
        "--modules-to-save",
        type=str,
        default=None,
        help="Optional comma-separated non-LoRA modules to train and save.",
    )
    group.add_argument("-h", "--help", action="store_true", dest="lora_help")
    return parser.parse_known_args(argv)


def validate_lora_args(args: argparse.Namespace) -> None:
    if args.lora_r <= 0:
        raise ValueError("`lora_r` must be > 0.")
    if args.lora_alpha <= 0:
        raise ValueError("`lora_alpha` must be > 0.")
    if not 0.0 <= args.lora_dropout < 1.0:
        raise ValueError("`lora_dropout` must be in [0, 1).")


def comma_separated(value: str | None) -> list[str] | None:
    if value is None:
        return None
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("A comma-separated module list cannot be empty.")
    return values


def resolve_target_modules() -> list[str]:
    """Always adapt both the global Qwen3 and local GPT-2 stacks."""
    return list(GLOBAL_LORA_MODULES + LOCAL_LORA_MODULES)


def target_is_in_scope(name: str, target: str) -> bool:
    is_global = name.startswith("transformer.")
    is_local = name.startswith("local_transformer.")
    if target in GLOBAL_LORA_MODULES:
        return is_global
    if target in LOCAL_LORA_MODULES:
        return is_local
    return False


def matching_linear_modules(
    model: torch.nn.Module,
    targets: Iterable[str],
) -> Dict[str, list[str]]:
    matches = {target: [] for target in targets}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        suffix = name.rsplit(".", 1)[-1]
        if suffix in matches and target_is_in_scope(name, suffix):
            matches[suffix].append(name)
    return matches


def exact_target_module_names(
    model: torch.nn.Module,
    targets: Iterable[str],
) -> list[str]:
    """Resolve suffix selectors to exact paths before handing them to PEFT.

    PEFT normally treats a list of target modules as suffix matches.  Resolving
    those suffixes here prevents a local GPT-2 selector such as ``c_proj`` from
    accidentally affecting a future module outside ``local_transformer``.
    """
    matches = matching_linear_modules(model, targets)
    missing = [target for target, names in matches.items() if not names]
    if missing:
        raise ValueError(
            "LoRA target modules did not match nn.Linear layers: " + ", ".join(missing)
        )
    return sorted({name for names in matches.values() for name in names})


def attach_lora(model: torch.nn.Module, args: argparse.Namespace):
    try:
        from peft import LoraConfig, PeftModel, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "LoRA training requires PEFT. Install it with `pip install peft`."
        ) from exc

    if args.lora_resume_adapter:
        adapter_path = Path(args.lora_resume_adapter)
        if not adapter_path.is_dir():
            raise FileNotFoundError(f"LoRA adapter directory not found: {adapter_path}")
        return PeftModel.from_pretrained(model, adapter_path, is_trainable=True)

    targets = resolve_target_modules()
    exact_targets = exact_target_module_names(model, targets)

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=exact_targets,
        modules_to_save=comma_separated(args.modules_to_save),
    )
    return get_peft_model(model, config)


def unwrap_peft_model(model):
    unwrapped = model
    while hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module
    if hasattr(unwrapped, "get_base_model"):
        return unwrapped.get_base_model()
    return unwrapped


def trainable_parameters(parameters: Iterable[torch.nn.Parameter]) -> list[torch.nn.Parameter]:
    selected = [parameter for parameter in parameters if parameter.requires_grad]
    if not selected:
        raise RuntimeError("No trainable LoRA parameters were found.")
    return selected


def optimizer_factory(
    optimizer_name: str,
    torch_adamw,
    bnb_adamw8bit,
    bnb_paged_adamw8bit,
):
    optimizer_classes = {
        "adamw": torch_adamw,
        "adamw_8bit": bnb_adamw8bit,
        "paged_adamw_8bit": bnb_paged_adamw8bit,
    }

    def build(parameters, *args, **kwargs):
        kwargs.pop("foreach", None)
        if optimizer_name == "adamw":
            return optimizer_classes[optimizer_name](trainable_parameters(parameters), *args, **kwargs)
        return optimizer_classes[optimizer_name](trainable_parameters(parameters), *args, **kwargs)

    return build


def save_lora_checkpoint(
    *,
    accelerator,
    model,
    model_path: str,
    codec_path: str,
    output_dir: Path,
    train_args: Dict[str, Any],
    global_step: int,
    epoch: int,
) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(output_dir, safe_serialization=True)
        sft.copy_support_files(output_dir)
        sft.copy_inference_assets(model_path, codec_path, output_dir)
        metadata = dict(train_args)
        metadata.update(
            saved_global_step=int(global_step),
            saved_epoch=int(epoch),
            saved_at=sft.format_timestamp(),
            checkpoint_type="peft_lora_adapter",
            base_model_name_or_path=model_path,
        )
        with open(output_dir / "finetune_args.json", "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)
    accelerator.wait_for_everyone()


def print_help() -> None:
    print(
        "LoRA-specific options:\n"
        "  --lora-r INT                         default: 8\n"
        "  --lora-alpha INT                     default: 16\n"
        "  --lora-dropout FLOAT                 default: 0.05\n"
        "  --lora-resume-adapter DIR             continue an adapter\n"
        "  --optimizer {adamw,adamw_8bit,paged_adamw_8bit}  default: adamw\n"
        "  --modules-to-save CSV                 train/save extra modules\n"
    )


def main() -> None:
    lora_args, base_argv = parse_lora_args(sys.argv[1:])
    validate_lora_args(lora_args)
    if lora_args.lora_help:
        print_help()
        if not base_argv or any(arg in {"-h", "--help"} for arg in base_argv):
            return

    # Keep sft.py's parser authoritative for all ordinary SFT arguments.
    original_argv = sys.argv
    sys.argv = [sys.argv[0], *base_argv]

    original_parse_args = sft.parse_args
    original_load_model = sft.load_training_model
    original_unwrap = sft.unwrap_training_model
    original_save = sft.save_checkpoint
    original_torch_adamw = sft.AdamW

    try:
        import bitsandbytes as bnb
    except ImportError as exc:
        if lora_args.optimizer != "adamw":
            raise ImportError(
                f"{lora_args.optimizer} requires bitsandbytes. Install it with "
                "`pip install bitsandbytes`."
            ) from exc
        bnb = None

    bnb_adamw8bit = None if bnb is None else bnb.optim.AdamW8bit
    bnb_paged_adamw8bit = None if bnb is None else bnb.optim.PagedAdamW8bit
    build_optimizer = optimizer_factory(
        lora_args.optimizer,
        original_torch_adamw,
        bnb_adamw8bit,
        bnb_paged_adamw8bit,
    )

    def parse_args_with_lora() -> argparse.Namespace:
        args = original_parse_args()
        for key, value in vars(lora_args).items():
            if key != "lora_help":
                setattr(args, key, value)
        return args

    def load_model_with_lora(**kwargs):
        model = original_load_model(**kwargs)
        model = attach_lora(model, lora_args)
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in model.parameters())
        print(
            f"[{sft.format_timestamp()}] [lora] trainable={trainable:,} "
            f"total={total:,} ratio={100.0 * trainable / total:.4f}% "
            f"targets={resolve_target_modules()}"
        )
        return model

    sft.parse_args = parse_args_with_lora
    sft.load_training_model = load_model_with_lora
    sft.unwrap_training_model = unwrap_peft_model
    sft.save_checkpoint = save_lora_checkpoint
    sft.AdamW = build_optimizer
    # Also intercept a local sft.py variant that imports bitsandbytes and uses AdamW8bit directly.
    # The stock sft.py uses torch AdamW and has no `bnb` attribute, so only patch when it exists.
    patch_sft_bnb = bnb is not None and hasattr(sft, "bnb")
    if patch_sft_bnb:
        sft.bnb.optim.AdamW8bit = build_optimizer

    try:
        sft.main()
    finally:
        sys.argv = original_argv
        sft.parse_args = original_parse_args
        sft.load_training_model = original_load_model
        sft.unwrap_training_model = original_unwrap
        sft.save_checkpoint = original_save
        sft.AdamW = original_torch_adamw
        if patch_sft_bnb:
            sft.bnb.optim.AdamW8bit = bnb_adamw8bit


if __name__ == "__main__":
    main()
