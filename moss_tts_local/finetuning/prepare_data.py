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
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from accelerate import Accelerator
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moss_tts_local.finetuning.common import (
    dump_jsonl,
    load_jsonl,
    normalize_audio_path_list,
    resolve_shard_spec,
    select_rank_shard,
    shard_output_path,
)
from moss_tts_local.processing_moss_tts import MossTTSDelayProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare MOSS-TTS finetuning JSONL by extracting target audio codes."
    )
    parser.add_argument("--model-path", type=str, default="OpenMOSS-Team/MOSS-TTS-Local-Transformer")
    parser.add_argument("--codec-path", type=str, default="OpenMOSS-Team/MOSS-Audio-Tokenizer")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Audio tokenizer device. Use `auto` to follow the current Accelerate rank device.",
    )
    parser.add_argument("--input-jsonl", type=str, required=True)
    parser.add_argument("--output-jsonl", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-vq", type=int, default=None)
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Advanced override. Normally inferred from Accelerate world size.",
    )
    parser.add_argument(
        "--shard-rank",
        type=int,
        default=None,
        help="Advanced override. Normally inferred from Accelerate rank.",
    )
    parser.add_argument(
        "--skip-reference-audio-codes",
        dest="encode_reference_audio",
        action="store_false",
        help="Skip pre-encoding `ref_audio` / `reference` / `reference_audio`. Default is to encode them.",
    )
    parser.add_argument(
        "--encode-reference-audio",
        dest="encode_reference_audio",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-encode-reference-audio",
        dest="encode_reference_audio",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--encode-ref-audio",
        dest="encode_reference_audio",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--save-shard-suffix",
        action="store_true",
        help="Advanced override. Save output.rankXXXXX-of-YYYYY.jsonl even in manual sharding mode.",
    )
    parser.set_defaults(encode_reference_audio=True)
    return parser.parse_args()


def batch_encode(
    processor: MossTTSDelayProcessor,
    paths: List[str],
    batch_size: int,
    n_vq: Optional[int],
    desc: str,
) -> List[torch.Tensor]:
    all_codes: List[torch.Tensor] = []
    for start in tqdm(range(0, len(paths), batch_size), desc=desc):
        batch_paths = paths[start : start + batch_size]
        all_codes.extend(processor.encode_audios_from_path(batch_paths, n_vq=n_vq))
    return all_codes


def collect_paths(records: List[Dict[str, Any]], field_name: str) -> List[str]:
    paths: List[str] = []
    for record in records:
        values = normalize_audio_path_list(record.get(field_name), field_name, allow_none=(field_name == "reference"))
        if values is not None:
            paths.extend(value for value in values if value is not None)
    return list(dict.fromkeys(paths))


def collect_reference_paths(records: List[Dict[str, Any]]) -> List[str]:
    unique_paths: List[str] = []
    for field_name in ("ref_audio", "reference_audio", "reference"):
        unique_paths.extend(collect_paths(records, field_name))
    return list(dict.fromkeys(unique_paths))


def attach_reference_audio_codes(
    records: List[Dict[str, Any]],
    path_to_codes: Dict[str, List[List[int]]],
) -> None:
    for record in records:
        ref_audio = normalize_audio_path_list(record.get("ref_audio"), "ref_audio")
        if ref_audio is not None:
            if len(ref_audio) != 1:
                raise ValueError("`ref_audio` only supports a single path.")
            record["ref_audio_codes"] = path_to_codes[ref_audio[0]]

        reference_audio = normalize_audio_path_list(record.get("reference_audio"), "reference_audio")
        if reference_audio is not None:
            record["reference_audio_codes"] = [path_to_codes[path] for path in reference_audio]

        reference = normalize_audio_path_list(record.get("reference"), "reference", allow_none=True)
        if reference is not None:
            record["reference_audio_codes"] = [
                None if path is None else path_to_codes[path]
                for path in reference
            ]


def main() -> None:
    args = parse_args()
    accelerator = Accelerator()
    device = str(accelerator.device) if args.device == "auto" else args.device

    all_records = load_jsonl(args.input_jsonl)
    world_size, rank = resolve_shard_spec(
        args.num_shards,
        args.shard_rank,
        default_num_shards=accelerator.num_processes,
        default_shard_rank=accelerator.process_index,
    )
    records = select_rank_shard(all_records, world_size, rank)
    if not records:
        raise ValueError(
            f"No records found for shard rank={rank} / world_size={world_size} in {args.input_jsonl}."
        )

    processor = MossTTSDelayProcessor.from_pretrained(
        args.model_path,
        codec_path=args.codec_path,
    )
    processor.audio_tokenizer = processor.audio_tokenizer.to(device)

    target_audio_paths = []
    for index, record in enumerate(records):
        audio_path = record.get("audio")
        if not isinstance(audio_path, str) or not audio_path:
            raise ValueError(f"Record {index} is missing a valid `audio` field.")
        target_audio_paths.append(audio_path)

    target_audio_codes = batch_encode(
        processor=processor,
        paths=target_audio_paths,
        batch_size=args.batch_size,
        n_vq=args.n_vq,
        desc="Encoding target audio",
    )

    for record, codes in zip(records, target_audio_codes):
        record["audio_codes"] = codes.tolist()

    if args.encode_reference_audio:
        unique_reference_paths = collect_reference_paths(records)
        if unique_reference_paths:
            reference_codes = batch_encode(
                processor=processor,
                paths=unique_reference_paths,
                batch_size=args.batch_size,
                n_vq=args.n_vq,
                desc="Encoding reference audio",
            )
            reference_code_map = {
                path: codes.tolist()
                for path, codes in zip(unique_reference_paths, reference_codes)
            }
            attach_reference_audio_codes(records, reference_code_map)

    output_path = args.output_jsonl
    if world_size > 1 or args.save_shard_suffix:
        output_path = str(shard_output_path(args.output_jsonl, rank, world_size))
    dump_jsonl(records, output_path)
    accelerator.print(
        f"[prepare_data] rank={rank}/{world_size} input_records={len(all_records)} "
        f"local_records={len(records)} device={device} output={output_path}"
    )


if __name__ == "__main__":
    main()
