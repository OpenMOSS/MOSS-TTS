from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from types import TracebackType
from typing import Protocol, cast

import safetensors.torch as safetensors_torch
import torch
from safetensors import safe_open

MAIN_MODEL_PREFIX = "language_model."
FUSED_MAIN_MODEL_PREFIX = "model."
CODEC_PREFIX = "codec_model."
INDEX_FILENAME = "model.safetensors.index.json"
SINGLE_FILE_FILENAME = "model.safetensors"
MAX_SHARD_SIZE = 5 * 1024**3
PROCESSOR_SOURCE_FILENAME = "processing_moss_tts_delay_with_codec.py"
PROCESSOR_CONFIG_FILENAME = "processor_config.json"
TOKENIZER_CONFIG_FILENAME = "tokenizer_config.json"
PROCESSOR_CLASS_NAME = "MossTTSDelayWithCodecProcessor"
PROCESSOR_MODULE_NAME = "processing_moss_tts_delay_with_codec"
PROCESSOR_AUTO_PROCESSOR = f"{PROCESSOR_MODULE_NAME}.{PROCESSOR_CLASS_NAME}"
REQUIRED_INPUT_FILES = ("config.json",)
REQUIRED_TOKENIZER_FILES = (
    "tokenizer.json",
    TOKENIZER_CONFIG_FILENAME,
    "special_tokens_map.json",
    "chat_template.jinja",
)
OPTIONAL_TOKENIZER_FILES = (
    "added_tokens.json",
    "merges.txt",
    "vocab.json",
)
PROCESSOR_TOKEN_CONFIG_FIELDS = (
    ("audio_start_token", "audio_start_token_id"),
    ("audio_end_token", "audio_end_token_id"),
    ("audio_user_slot_token", "audio_user_slot_token_id"),
    ("audio_assistant_gen_slot_token", "audio_assistant_gen_slot_token_id"),
    ("audio_assistant_delay_slot_token", "audio_assistant_delay_slot_token_id"),
)
JSONDict = dict[str, object]
SaveFileFn = Callable[[dict[str, torch.Tensor], str], None]


class SafeOpenReader(Protocol):
    def __enter__(self) -> "SafeOpenReader": ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...

    def keys(self) -> Iterable[str]: ...

    def get_tensor(self, key: str) -> torch.Tensor: ...


def _open_safe_reader(path: Path) -> SafeOpenReader:
    return cast(
        SafeOpenReader,
        cast(object, safe_open(str(path), framework="pt")),
    )


_SAVE_FILE = cast(SaveFileFn, getattr(safetensors_torch, "save_file"))


def save_file(tensors: dict[str, torch.Tensor], filename: str) -> None:
    _SAVE_FILE(tensors, filename)


def load_json(path: Path) -> JSONDict:
    with path.open(encoding="utf-8") as handle:
        raw_data = cast(object, json.load(handle))

    if not isinstance(raw_data, dict):
        raise ValueError(f"Expected JSON object in {path}")

    raw_keys = cast(Iterable[object], raw_data.keys())
    if not all(isinstance(key, str) for key in raw_keys):
        raise ValueError(f"Expected JSON object with string keys in {path}")

    return cast(JSONDict, raw_data)


def write_sorted_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        _ = handle.write("\n")


def require_json_object(value: object, context: str) -> JSONDict:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object")

    raw_keys = cast(Iterable[object], value.keys())
    if not all(isinstance(key, str) for key in raw_keys):
        raise ValueError(f"{context} must have string keys")

    return cast(JSONDict, value)


def require_json_int(value: object, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{context} must be an int")
    return value


def require_json_array(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a JSON array")
    return cast(list[object], value)


def load_added_token_content_by_id(tokenizer_json_path: Path) -> dict[int, str]:
    tokenizer_payload = load_json(tokenizer_json_path)
    added_tokens = require_json_array(
        tokenizer_payload.get("added_tokens"),
        f"{tokenizer_json_path} field 'added_tokens'",
    )

    content_by_id: dict[int, str] = {}
    for index, raw_added_token in enumerate(added_tokens):
        added_token = require_json_object(
            raw_added_token,
            f"{tokenizer_json_path} field 'added_tokens[{index}]'",
        )
        token_id = require_json_int(
            added_token.get("id"),
            f"{tokenizer_json_path} field 'added_tokens[{index}].id'",
        )
        token_content = added_token.get("content")
        if not isinstance(token_content, str):
            raise ValueError(
                f"{tokenizer_json_path} field 'added_tokens[{index}].content' must be a string"
            )
        if token_id in content_by_id:
            raise ValueError(
                f"{tokenizer_json_path} field 'added_tokens' contains duplicate id {token_id}"
            )
        content_by_id[token_id] = token_content

    return content_by_id


def resolve_processor_token_strings(
    fused_config: JSONDict, tokenizer_json_path: Path
) -> dict[str, str]:
    token_content_by_id = load_added_token_content_by_id(tokenizer_json_path)
    resolved_tokens: dict[str, str] = {}

    for token_field, token_id_field in PROCESSOR_TOKEN_CONFIG_FIELDS:
        token_id = require_json_int(
            fused_config.get(token_id_field),
            f"fused config field '{token_id_field}'",
        )
        token_content = token_content_by_id.get(token_id)
        if token_content is None:
            raise ValueError(
                f"{tokenizer_json_path} field 'added_tokens' is missing id {token_id} required by fused config field '{token_id_field}'"
            )
        resolved_tokens[token_field] = token_content

    return resolved_tokens


def require_weight_map(index: JSONDict, context: str) -> dict[str, str]:
    raw_weight_map = require_json_object(
        index.get("weight_map"), f"{context} field 'weight_map'"
    )

    weight_map: dict[str, str] = {}
    raw_items = cast(Iterable[tuple[object, object]], raw_weight_map.items())
    for tensor_name, shard_name in raw_items:
        if not isinstance(tensor_name, str) or not isinstance(shard_name, str):
            raise ValueError(
                f"{context} field 'weight_map' must map strings to strings"
            )
        weight_map[tensor_name] = shard_name

    return weight_map


def load_source_index(model_dir: Path) -> JSONDict:
    index_path = model_dir / INDEX_FILENAME
    if index_path.exists():
        return load_json(index_path)

    single_file_path = model_dir / SINGLE_FILE_FILENAME
    if single_file_path.exists():
        reader = _open_safe_reader(single_file_path)
        with reader as handle:
            keys = handle.keys()
            return {
                "metadata": {},
                "weight_map": {key: SINGLE_FILE_FILENAME for key in keys},
            }

    raise FileNotFoundError(f"No safetensors files found in {model_dir}")


def group_weight_map_by_shard(weight_map: dict[str, str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}

    for tensor_name, shard_name in weight_map.items():
        grouped.setdefault(shard_name, []).append(tensor_name)

    return {shard_name: sorted(grouped[shard_name]) for shard_name in sorted(grouped)}


def build_prefixed_codec_keys(
    codec_weight_map: dict[str, str],
    main_weight_map: dict[str, str],
    prefix: str,
) -> dict[str, str]:
    prefixed_weight_map: dict[str, str] = {}
    main_keys = set(main_weight_map)

    for tensor_name in sorted(codec_weight_map):
        prefixed_name = f"{prefix}{tensor_name}"
        if prefixed_name in main_keys:
            raise ValueError(
                f"Prefixed codec tensor name collision with main checkpoint: '{prefixed_name}'"
            )
        prefixed_weight_map[prefixed_name] = codec_weight_map[tensor_name]

    return prefixed_weight_map


def remap_main_tensor_name(tensor_name: str) -> str:
    if tensor_name.startswith(MAIN_MODEL_PREFIX):
        return FUSED_MAIN_MODEL_PREFIX + tensor_name[len(MAIN_MODEL_PREFIX) :]
    return tensor_name


def build_remapped_main_weight_map(main_weight_map: dict[str, str]) -> dict[str, str]:
    remapped_weight_map: dict[str, str] = {}
    source_names_by_output_name: dict[str, str] = {}

    for tensor_name in sorted(main_weight_map):
        remapped_name = remap_main_tensor_name(tensor_name)
        previous_name = source_names_by_output_name.get(remapped_name)
        if previous_name is not None:
            raise ValueError(
                f"Main tensor name collision after remap: '{previous_name}' and '{tensor_name}' both map to '{remapped_name}'"
            )
        source_names_by_output_name[remapped_name] = tensor_name
        remapped_weight_map[remapped_name] = main_weight_map[tensor_name]

    return remapped_weight_map


def _iter_shard_tensors(
    model_dir: Path,
    shard_groups: dict[str, list[str]],
    output_name_builder: Callable[[str], str],
) -> Iterator[tuple[str, torch.Tensor]]:
    for shard_name in sorted(shard_groups):
        reader = _open_safe_reader(model_dir / shard_name)
        with reader as handle:
            for tensor_name in sorted(shard_groups[shard_name]):
                yield output_name_builder(tensor_name), handle.get_tensor(tensor_name)


def iter_merged_tensors(
    main_dir: Path,
    main_groups: dict[str, list[str]],
    codec_dir: Path,
    codec_groups: dict[str, list[str]],
    prefix: str,
) -> Iterator[tuple[str, torch.Tensor]]:
    yield from _iter_shard_tensors(main_dir, main_groups, remap_main_tensor_name)
    yield from _iter_shard_tensors(
        codec_dir,
        codec_groups,
        lambda tensor_name: f"{prefix}{tensor_name}",
    )


def write_merged_shards(
    output_dir: Path,
    merged_tensors: Iterable[tuple[str, torch.Tensor]],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_idx = 0
    current_shard_size = 0
    current_tensors: dict[str, torch.Tensor] = {}
    merged_weight_map: dict[str, str] = {}
    saved_shards: list[tuple[str, tuple[str, ...]]] = []

    def flush_shard() -> None:
        nonlocal shard_idx, current_shard_size, current_tensors

        if not current_tensors:
            return

        shard_idx += 1
        shard_name = f"model-{shard_idx:05d}-of-PLACEHOLDER.safetensors"
        save_file(current_tensors, str(output_dir / shard_name))

        tensor_names = tuple(current_tensors)
        for tensor_name in tensor_names:
            merged_weight_map[tensor_name] = shard_name
        saved_shards.append((shard_name, tensor_names))

        current_tensors = {}
        current_shard_size = 0

    for tensor_name, tensor in merged_tensors:
        tensor_bytes = tensor.nelement() * tensor.element_size()
        if current_shard_size + tensor_bytes > MAX_SHARD_SIZE and current_tensors:
            flush_shard()

        current_tensors[tensor_name] = tensor
        current_shard_size += tensor_bytes

    flush_shard()

    total_shards = len(saved_shards)
    for shard_number, (old_name, tensor_names) in enumerate(saved_shards, start=1):
        new_name = f"model-{shard_number:05d}-of-{total_shards:05d}.safetensors"
        old_path = output_dir / old_name
        new_path = output_dir / new_name
        if old_name != new_name:
            _ = old_path.rename(new_path)

        for tensor_name in tensor_names:
            merged_weight_map[tensor_name] = new_name

    return merged_weight_map


def write_merged_index(output_dir: Path, merged_weight_map: dict[str, str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_names = sorted(set(merged_weight_map.values()))
    total_size = sum(
        (output_dir / shard_name).stat().st_size for shard_name in shard_names
    )
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": {
            tensor_name: merged_weight_map[tensor_name]
            for tensor_name in sorted(merged_weight_map)
        },
    }

    write_sorted_json(output_dir / INDEX_FILENAME, index)


def verify_fused_artifact(
    output_dir: Path,
    main_index: JSONDict,
    codec_index: JSONDict,
    prefix: str,
) -> dict[str, int]:
    config_path = output_dir / "config.json"
    output_config = load_json(config_path)

    codec_config = require_json_object(
        output_config.get("codec_config"),
        f"{config_path} field 'codec_config'",
    )
    if "auto_map" in output_config:
        raise ValueError(
            f"Fused config still contains top-level 'auto_map': {config_path}"
        )
    if "auto_map" in codec_config:
        raise ValueError(
            f"Fused config codec_config still contains 'auto_map': {config_path}"
        )

    index_path = output_dir / INDEX_FILENAME
    output_index = load_json(index_path)
    if set(output_index) != {"metadata", "weight_map"}:
        raise ValueError(
            f"Fused index must contain only 'metadata' and 'weight_map': {index_path}"
        )

    metadata = require_json_object(
        output_index.get("metadata"), f"{index_path} field 'metadata'"
    )
    if set(metadata) != {"total_size"}:
        raise ValueError(
            f"Fused index metadata must contain only 'total_size': {index_path}"
        )

    total_size = metadata.get("total_size")
    if not isinstance(total_size, int):
        raise ValueError(
            f"Fused index metadata.total_size must be an int: {index_path}"
        )

    typed_main_index = require_json_object(cast(object, main_index), "main_index")
    typed_codec_index = require_json_object(cast(object, codec_index), "codec_index")
    main_weight_map = build_remapped_main_weight_map(
        require_weight_map(typed_main_index, "main_index")
    )
    codec_weight_map = require_weight_map(typed_codec_index, "codec_index")
    output_weight_map = require_weight_map(output_index, f"fused index {index_path}")

    expected_codec_keys = {f"{prefix}{tensor_name}" for tensor_name in codec_weight_map}
    missing_prefixed_codec_keys = sorted(
        tensor_name
        for tensor_name in expected_codec_keys
        if tensor_name not in output_weight_map
    )
    if missing_prefixed_codec_keys:
        raise ValueError(
            f"Fused index is missing prefixed codec tensor '{missing_prefixed_codec_keys[0]}'"
        )

    unprefixed_codec_keys = sorted(
        tensor_name
        for tensor_name in codec_weight_map
        if tensor_name in output_weight_map
    )
    if unprefixed_codec_keys:
        raise ValueError(
            f"Fused index contains codec tensor without prefix '{prefix}': {unprefixed_codec_keys[0]}"
        )

    expected_weight_count = len(main_weight_map) + len(codec_weight_map)
    if len(output_weight_map) != expected_weight_count:
        raise ValueError(
            f"Fused index weight count mismatch: expected {expected_weight_count}, found {len(output_weight_map)}"
        )

    expected_keys = set(main_weight_map) | expected_codec_keys
    actual_keys = set(output_weight_map)
    if actual_keys != expected_keys:
        missing_keys = sorted(expected_keys - actual_keys)
        unexpected_keys = sorted(actual_keys - expected_keys)
        details: list[str] = []
        if missing_keys:
            details.append(f"missing key '{missing_keys[0]}'")
        if unexpected_keys:
            details.append(f"unexpected key '{unexpected_keys[0]}'")
        raise ValueError(
            f"Fused index keys do not match expected merged keys: {', '.join(details)}"
        )

    shard_groups = group_weight_map_by_shard(output_weight_map)
    actual_total_size = 0
    for shard_name in shard_groups:
        shard_path = Path(shard_name)
        if shard_path.is_absolute():
            raise ValueError(f"Fused index uses absolute shard path: {shard_name}")

        full_shard_path = output_dir / shard_name
        if not full_shard_path.is_file():
            raise FileNotFoundError(
                f"Fused index references missing shard: {full_shard_path}"
            )

        actual_total_size += full_shard_path.stat().st_size

        reader = _open_safe_reader(full_shard_path)
        with reader as handle:
            for tensor_name in shard_groups[shard_name]:
                try:
                    _ = handle.get_tensor(tensor_name)
                except Exception as exc:
                    raise ValueError(
                        f"Failed to read indexed tensor '{tensor_name}' from shard '{shard_name}': {exc}"
                    ) from exc

    if actual_total_size != total_size:
        raise ValueError(
            f"Fused index metadata.total_size mismatch: expected {actual_total_size}, found {total_size}"
        )

    return {
        "main_weight_count": len(main_weight_map),
        "codec_weight_count": len(codec_weight_map),
        "merged_weight_count": len(output_weight_map),
        "output_shard_count": len(shard_groups),
    }


def collect_tokenizer_asset_filenames(model_dir: Path) -> list[str]:
    missing_required = [
        filename
        for filename in REQUIRED_TOKENIZER_FILES
        if not (model_dir / filename).is_file()
    ]
    if missing_required:
        missing_names = ", ".join(missing_required)
        raise FileNotFoundError(
            f"Missing required tokenizer files in {model_dir}: {missing_names}"
        )

    return [
        filename
        for filename in REQUIRED_TOKENIZER_FILES + OPTIONAL_TOKENIZER_FILES
        if (model_dir / filename).is_file()
    ]


def copy_tokenizer_assets(
    model_dir: Path,
    output_dir: Path,
    tokenizer_filenames: Iterable[str] | None = None,
) -> list[str]:
    filenames_to_copy = (
        list(tokenizer_filenames)
        if tokenizer_filenames is not None
        else collect_tokenizer_asset_filenames(model_dir)
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    copied_files: list[str] = []
    for filename in filenames_to_copy:
        source_path = model_dir / filename
        if not source_path.is_file():
            continue
        _ = shutil.copy2(source_path, output_dir / filename)
        copied_files.append(filename)

    return copied_files


def local_processor_source_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent
        / "moss_tts_delay"
        / PROCESSOR_SOURCE_FILENAME
    )


def copy_processor_source(output_dir: Path) -> None:
    source_path = local_processor_source_path()
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing processor source file: {source_path}")
    _ = shutil.copy2(source_path, output_dir / PROCESSOR_SOURCE_FILENAME)


def build_processor_config(fused_config: JSONDict, tokenizer_json_path: Path) -> JSONDict:
    codec_config = require_json_object(
        fused_config.get("codec_config"), "fused config field 'codec_config'"
    )
    language_config = require_json_object(
        fused_config.get("language_config"), "fused config field 'language_config'"
    )
    resolved_token_strings = resolve_processor_token_strings(
        fused_config, tokenizer_json_path
    )

    return {
        "AudioToken_PlaceHolder": "<|Audio|>",
        "audio_start_token": resolved_token_strings["audio_start_token"],
        "audio_end_token": resolved_token_strings["audio_end_token"],
        "audio_pad_code": require_json_int(
            fused_config.get("audio_pad_code"),
            "fused config field 'audio_pad_code'",
        ),
        "audio_user_slot_token": resolved_token_strings["audio_user_slot_token"],
        "auto_map": {"AutoProcessor": PROCESSOR_AUTO_PROCESSOR},
        "downsample_rate": require_json_int(
            codec_config.get("downsample_rate"),
            "fused config field 'codec_config.downsample_rate'",
        ),
        "audio_assistant_delay_slot_token": resolved_token_strings[
            "audio_assistant_delay_slot_token"
        ],
        "audio_assistant_gen_slot_token": resolved_token_strings[
            "audio_assistant_gen_slot_token"
        ],
        "n_vq": require_json_int(fused_config.get("n_vq"), "fused config field 'n_vq'"),
        "processor_class": PROCESSOR_CLASS_NAME,
        "shift": True,
        "text_pad_code": require_json_int(
            language_config.get("pad_token_id"),
            "fused config field 'language_config.pad_token_id'",
        ),
    }


def write_processor_config(output_dir: Path, fused_config: JSONDict) -> None:
    write_sorted_json(
        output_dir / PROCESSOR_CONFIG_FILENAME,
        build_processor_config(fused_config, output_dir / "tokenizer.json"),
    )


def patch_tokenizer_config_processor_class(
    tokenizer_config_path: Path, processor_class: str
) -> None:
    tokenizer_config = load_json(tokenizer_config_path)
    updated_tokenizer_config = copy.deepcopy(tokenizer_config)
    updated_tokenizer_config["processor_class"] = processor_class

    raw_text = tokenizer_config_path.read_text(encoding="utf-8")
    patched_text, replacements = re.subn(
        r'("processor_class"\s*:\s*)"[^"\n]*"',
        lambda match: f'{match.group(1)}"{processor_class}"',
        raw_text,
        count=1,
    )
    if replacements != 1:
        raise ValueError(
            f"Failed to patch 'processor_class' in {tokenizer_config_path}"
        )

    _ = tokenizer_config_path.write_text(patched_text, encoding="utf-8")

    reloaded_tokenizer_config = load_json(tokenizer_config_path)
    if reloaded_tokenizer_config != updated_tokenizer_config:
        raise ValueError(
            f"Patched tokenizer config changed fields beyond 'processor_class': {tokenizer_config_path}"
        )


def cleanup_partial_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)


def build_merged_config(main_config: JSONDict, codec_config: JSONDict) -> JSONDict:
    if "codec_config" in main_config:
        raise ValueError(
            "Main config already contains top-level 'codec_config'; refusing duplicate codec_config merge"
        )

    merged_config = copy.deepcopy(main_config)
    _ = merged_config.pop("auto_map", None)
    merged_config["architectures"] = ["MossTTSDelayWithCodec"]
    merged_config["model_type"] = "moss_tts_delay_with_codec"

    if "audio_end_token_id" not in merged_config:
        raise ValueError(
            "Main config field 'audio_end_token_id' is required for fused eos_token propagation"
        )
    if "dtype" not in merged_config:
        raise ValueError(
            "Main config field 'dtype' is required for fused language_config dtype propagation"
        )
    language_config = require_json_object(
        merged_config.get("language_config"), "main config field 'language_config'"
    )
    language_config["eos_token_id"] = merged_config["audio_end_token_id"]
    language_config["dtype"] = merged_config["dtype"]

    cleaned_codec_config = copy.deepcopy(codec_config)
    _ = cleaned_codec_config.pop("auto_map", None)
    merged_config["codec_config"] = cleaned_codec_config

    return merged_config


def write_merged_config(output_dir: Path, merged_config: JSONDict) -> None:
    write_sorted_json(output_dir / "config.json", merged_config)


def verify_processor_artifacts(output_dir: Path, fused_config: JSONDict) -> None:
    source_processor_path = local_processor_source_path()
    output_processor_path = output_dir / PROCESSOR_SOURCE_FILENAME
    if not output_processor_path.is_file():
        raise FileNotFoundError(
            f"Fused output is missing processor source: {output_processor_path}"
        )

    if output_processor_path.read_bytes() != source_processor_path.read_bytes():
        raise ValueError(
            f"Fused processor source does not match local source: {output_processor_path}"
        )

    processor_config_path = output_dir / PROCESSOR_CONFIG_FILENAME
    processor_config = load_json(processor_config_path)
    expected_processor_config = build_processor_config(
        fused_config, output_dir / "tokenizer.json"
    )
    if processor_config != expected_processor_config:
        raise ValueError(
            f"Fused processor config does not match expected fused-config values: {processor_config_path}"
        )

    tokenizer_config_path = output_dir / TOKENIZER_CONFIG_FILENAME
    tokenizer_config = load_json(tokenizer_config_path)
    if tokenizer_config.get("processor_class") != PROCESSOR_CLASS_NAME:
        raise ValueError(
            f"Fused tokenizer config has unexpected processor_class: {tokenizer_config_path}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate inputs for fusing a MOSS-TTS checkpoint with a codec checkpoint."
    )
    _ = parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the main model checkpoint directory.",
    )
    _ = parser.add_argument(
        "--codec-model-path",
        required=True,
        help="Path to the codec model checkpoint directory.",
    )
    _ = parser.add_argument(
        "--save-path",
        required=True,
        help="Path to the fused output directory. Existing directories require --overwrite or interactive confirmation.",
    )
    _ = parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing save directory without prompting.",
    )
    return parser.parse_args()


def _normalize_path(value: str, arg_name: str) -> Path:
    if not value:
        raise ValueError(f"{arg_name} is empty")
    return Path(value).expanduser()


def _require_namespace_str(
    args: argparse.Namespace, attr_name: str, arg_name: str
) -> str:
    value = getattr(args, attr_name, None)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{arg_name} is empty")
    return value


def _require_namespace_path(args: argparse.Namespace, attr_name: str) -> Path:
    value = getattr(args, attr_name, None)
    if not isinstance(value, Path):
        raise ValueError(f"{attr_name} was not normalized to Path")
    return value


def _require_namespace_bool(args: argparse.Namespace, attr_name: str) -> bool:
    value = getattr(args, attr_name, None)
    if not isinstance(value, bool):
        raise ValueError(f"{attr_name} was not normalized to bool")
    return value


def _validate_source_dir(arg_name: str, model_dir: Path) -> None:
    if not model_dir.exists():
        raise FileNotFoundError(f"{arg_name}: {model_dir} does not exist")
    if not model_dir.is_dir():
        raise ValueError(f"{arg_name}: {model_dir} is not a directory")

    for required_name in REQUIRED_INPUT_FILES:
        required_path = model_dir / required_name
        if not required_path.is_file():
            raise FileNotFoundError(
                f"{arg_name}: missing required input file '{required_name}' in {model_dir}"
            )

    index_path = model_dir / INDEX_FILENAME
    single_file_path = model_dir / SINGLE_FILE_FILENAME
    if not index_path.is_file() and not single_file_path.is_file():
        raise FileNotFoundError(
            f"{arg_name}: expected '{INDEX_FILENAME}' or '{SINGLE_FILE_FILENAME}' in {model_dir}"
        )


def _prompt_overwrite_confirmation(save_path: Path) -> bool:
    if not sys.stdin.isatty():
        raise FileExistsError(
            f"--save-path already exists: {save_path}. Re-run with --overwrite to replace it."
        )

    response = input(
        f"--save-path already exists: {save_path}. Overwrite it? [y/N]: "
    ).strip()
    return response.lower() in {"y", "yes"}


def _resolve_overwrite_choice(save_path: Path, overwrite: bool) -> bool:
    if not save_path.exists():
        return False

    if not save_path.is_dir():
        raise FileExistsError(
            f"--save-path already exists and is not a directory: {save_path}"
        )

    if overwrite:
        return True

    if _prompt_overwrite_confirmation(save_path):
        return True

    raise FileExistsError(
        f"Refusing to overwrite existing --save-path: {save_path}. Re-run with --overwrite to replace it."
    )


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    args.model_path = _normalize_path(
        _require_namespace_str(args, "model_path", "--model-path"),
        "--model-path",
    )
    args.codec_model_path = _normalize_path(
        _require_namespace_str(args, "codec_model_path", "--codec-model-path"),
        "--codec-model-path",
    )
    args.save_path = _normalize_path(
        _require_namespace_str(args, "save_path", "--save-path"),
        "--save-path",
    )

    _validate_source_dir("--model-path", args.model_path)
    _validate_source_dir("--codec-model-path", args.codec_model_path)

    args.overwrite = _resolve_overwrite_choice(
        args.save_path,
        _require_namespace_bool(args, "overwrite"),
    )

    return args


def main() -> None:
    args = parse_args()
    args = validate_args(args)
    model_path = _require_namespace_path(args, "model_path")
    codec_model_path = _require_namespace_path(args, "codec_model_path")
    output_dir = _require_namespace_path(args, "save_path")
    overwrite = _require_namespace_bool(args, "overwrite")

    main_config = load_json(model_path / "config.json")
    codec_config = load_json(codec_model_path / "config.json")
    main_index = load_source_index(model_path)
    codec_index = load_source_index(codec_model_path)
    tokenizer_filenames = collect_tokenizer_asset_filenames(model_path)
    merged_config = build_merged_config(main_config, codec_config)

    if overwrite and output_dir.exists():
        cleanup_partial_output_dir(output_dir)

    output_dir.mkdir(parents=True, exist_ok=False)

    copied_tokenizer_files = copy_tokenizer_assets(
        model_path,
        output_dir,
        tokenizer_filenames,
    )
    copy_processor_source(output_dir)
    write_processor_config(output_dir, merged_config)
    patch_tokenizer_config_processor_class(
        output_dir / TOKENIZER_CONFIG_FILENAME,
        PROCESSOR_CLASS_NAME,
    )

    write_merged_config(output_dir, merged_config)

    main_weight_map = require_weight_map(main_index, "main_index")
    remapped_main_weight_map = build_remapped_main_weight_map(main_weight_map)
    codec_weight_map = require_weight_map(codec_index, "codec_index")
    main_groups = group_weight_map_by_shard(main_weight_map)
    codec_groups = group_weight_map_by_shard(codec_weight_map)

    expected_codec_weight_map = build_prefixed_codec_keys(
        codec_weight_map,
        remapped_main_weight_map,
        CODEC_PREFIX,
    )

    merged_weight_map = write_merged_shards(
        output_dir,
        iter_merged_tensors(
            model_path,
            main_groups,
            codec_model_path,
            codec_groups,
            CODEC_PREFIX,
        ),
    )

    expected_merged_keys = set(remapped_main_weight_map) | set(expected_codec_weight_map)
    actual_merged_keys = set(merged_weight_map)
    if actual_merged_keys != expected_merged_keys:
        missing_keys = sorted(expected_merged_keys - actual_merged_keys)
        unexpected_keys = sorted(actual_merged_keys - expected_merged_keys)
        details: list[str] = []
        if missing_keys:
            details.append(f"missing key '{missing_keys[0]}'")
        if unexpected_keys:
            details.append(f"unexpected key '{unexpected_keys[0]}'")
        raise ValueError(
            f"Merged shard output keys do not match expected keys: {', '.join(details)}"
        )

    write_merged_index(output_dir, merged_weight_map)
    verification = verify_fused_artifact(output_dir, main_index, codec_index, CODEC_PREFIX)
    verify_processor_artifacts(output_dir, merged_config)

    print(
        f"Fused checkpoint ready: tokenizer_files={len(copied_tokenizer_files)} main_weights={verification['main_weight_count']} codec_weights={verification['codec_weight_count']} output_shards={verification['output_shard_count']}"
    )


if __name__ == "__main__":
    main()
