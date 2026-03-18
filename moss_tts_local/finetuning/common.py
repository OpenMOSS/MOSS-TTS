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
import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dump_jsonl(records: Iterable[Dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_jsonl_paths(spec: str | Sequence[str]) -> List[Path]:
    if isinstance(spec, (list, tuple)):
        tokens = [str(item).strip() for item in spec if str(item).strip()]
    else:
        tokens = [item.strip() for item in str(spec).split(",") if item.strip()]

    paths: List[Path] = []
    for token in tokens:
        if any(ch in token for ch in "*?[]"):
            matches = [Path(match) for match in sorted(glob.glob(token))]
            paths.extend(match for match in matches if match.suffix == ".jsonl")
            continue

        path = Path(token)
        if path.is_dir():
            paths.extend(sorted(p for p in path.iterdir() if p.suffix == ".jsonl"))
            continue

        paths.append(path)

    deduped: List[Path] = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)

    if not deduped:
        raise ValueError(f"No JSONL files found for input spec: {spec}")
    return deduped


def load_jsonl_spec(spec: str | Sequence[str]) -> tuple[List[Path], List[Dict[str, Any]]]:
    paths = resolve_jsonl_paths(spec)
    records: List[Dict[str, Any]] = []
    for path in paths:
        records.extend(load_jsonl(path))
    return paths, records


def resolve_shard_spec(
    num_shards: Optional[int],
    shard_rank: Optional[int],
    default_num_shards: Optional[int] = None,
    default_shard_rank: Optional[int] = None,
) -> tuple[int, int]:
    world_size = int(num_shards or default_num_shards or os.environ.get("WORLD_SIZE", 1))
    rank = int(shard_rank if shard_rank is not None else default_shard_rank or os.environ.get("RANK", 0))
    if world_size < 1:
        raise ValueError(f"`num_shards` must be >= 1, got {world_size}.")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"`shard_rank` must satisfy 0 <= rank < world_size, got rank={rank}, world_size={world_size}.")
    return world_size, rank


def select_rank_shard(records: Sequence[Dict[str, Any]], num_shards: int, shard_rank: int) -> List[Dict[str, Any]]:
    return [record for index, record in enumerate(records) if index % num_shards == shard_rank]


def shard_output_path(path: str | Path, shard_rank: int, num_shards: int) -> Path:
    output_path = Path(path)
    suffix = "".join(output_path.suffixes) or ".jsonl"
    stem = output_path.name[: -len(suffix)] if suffix else output_path.name
    shard_name = f"{stem}.rank{shard_rank:05d}-of-{num_shards:05d}{suffix}"
    return output_path.with_name(shard_name)


def normalize_audio_path_list(
    value: Any,
    field_name: str,
    allow_none: bool = False,
) -> Optional[List[Optional[str]]]:
    if value in (None, "", []):
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        if allow_none:
            if not all(item is None or isinstance(item, str) for item in value):
                raise ValueError(f"`{field_name}` must be a string, null, or a list containing strings/nulls.")
        elif not all(isinstance(item, str) for item in value):
            raise ValueError(f"`{field_name}` must be a string or a list of strings.")
        return value
    raise TypeError(f"Unsupported `{field_name}` type: {type(value)}")
