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
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torchaudio
from accelerate import Accelerator
from transformers import AutoModel
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moss_tts_realtime.finetuning.common import (
    dump_jsonl,
    load_jsonl,
    normalize_audio_path_list,
    resolve_shard_spec,
    select_rank_shard,
    shard_output_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare MOSS-TTS-Realtime finetuning JSONL by pre-encoding audio to RVQ codes."
    )
    parser.add_argument("--codec-path", type=str, default="OpenMOSS-Team/MOSS-Audio-Tokenizer")
    parser.add_argument(
        "--codec-sample-rate",
        type=int,
        default=24000,
        help="Target sample rate expected by the codec. Default: 24000.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Codec device. Use `auto` to follow the current Accelerate rank device.",
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


def load_codec(codec_path: str, device: str):
    """Load the MOSS-Audio-Tokenizer codec model onto the given device."""
    codec = AutoModel.from_pretrained(codec_path, trust_remote_code=True).eval()
    return codec.to(device)


def _load_and_resample(path: str, target_sr: int, device: torch.device | str) -> torch.Tensor:
    """Load a single audio file as a mono waveform tensor resampled to `target_sr`."""
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)
    return wav.squeeze(0).to(device)


@torch.no_grad()
def _encode_batch_with_codec(
    codec,
    wav_list: List[torch.Tensor],
    n_vq: Optional[int],
) -> List[torch.Tensor]:
    """Encode a batch of mono waveform tensors to a list of [T, NQ] int64 CPU tensors.

    Handles both the newer ``batch_encode`` API and the ``encode`` API
    with explicit padding, mirroring the encoding logic used in inference.
    """
    if hasattr(codec, "batch_encode"):
        enc = codec.batch_encode(wav_list, num_quantizers=n_vq)
        audio_codes = enc.audio_codes          # [NQ, B, T]
        audio_codes_lengths = enc.audio_codes_lengths  # [B]
    else:
        device = next(codec.parameters()).device
        max_len = max(int(w.shape[-1]) for w in wav_list)
        input_values = torch.zeros(len(wav_list), 1, max_len, device=device, dtype=torch.float32)
        padding_mask = torch.zeros(len(wav_list), max_len, device=device, dtype=torch.bool)
        for i, w in enumerate(wav_list):
            t_len = int(w.shape[-1])
            input_values[i, 0, :t_len] = w
            padding_mask[i, :t_len] = True
        enc = codec.encode(
            input_values,
            padding_mask=padding_mask,
            num_quantizers=n_vq,
            return_dict=True,
        )
        audio_codes = enc.audio_codes
        audio_codes_lengths = enc.audio_codes_lengths

    codes_list: List[torch.Tensor] = []
    for i in range(int(audio_codes.shape[1])):
        length_i = int(audio_codes_lengths[i].item())
        codes_i = (
            audio_codes[:, i, :length_i]
            .transpose(0, 1)
            .contiguous()
            .to(torch.long)
            .cpu()
        )
        codes_list.append(codes_i)
    return codes_list


def batch_encode(
    codec,
    paths: List[str],
    batch_size: int,
    n_vq: Optional[int],
    desc: str,
    sample_rate: int = 24000,
) -> List[torch.Tensor]:
    """Encode a list of audio file paths to RVQ codes in batches."""
    device = next(codec.parameters()).device
    all_codes: List[torch.Tensor] = []
    for start in tqdm(range(0, len(paths), batch_size), desc=desc):
        batch_paths = paths[start : start + batch_size]
        wav_list = [_load_and_resample(p, sample_rate, device) for p in batch_paths]
        all_codes.extend(_encode_batch_with_codec(codec, wav_list, n_vq))
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


logger = logging.getLogger(__name__)


def _filter_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Skip records that contain no assistant turn."""
    filtered = []
    for i, record in enumerate(records):
        convs = record.get("conversations", [])
        has_assistant = any(t.get("role") == "assistant" for t in convs)
        if not has_assistant:
            record_id = record.get("id", f"index={i}")
            roles = [t.get("role", "?") for t in convs]
            logger.warning(
                "[prepare_data] Skipping record %s: no assistant turn "
                "(roles=%s)", record_id, roles,
            )
            continue
        filtered.append(record)
    return filtered


def _collect_all_wav_paths(records: List[Dict[str, Any]]) -> List[str]:
    """Collect all unique wav paths from conversations and ref_wav fields."""
    seen = {}
    for record in records:
        ref_wav = record.get("ref_wav")
        if isinstance(ref_wav, str) and ref_wav:
            seen.setdefault(ref_wav, None)
        for turn in record.get("conversations", []):
            wav = turn.get("wav")
            if isinstance(wav, str) and wav:
                seen.setdefault(wav, None)
    return list(seen.keys())


def _attach_codes(
    records: List[Dict[str, Any]],
    path_to_codes: Dict[str, List[List[int]]],
) -> None:
    """Attach audio_codes to each conversation turn and ref_audio_codes to the record."""
    for record in records:
        ref_wav = record.get("ref_wav")
        if isinstance(ref_wav, str) and ref_wav and ref_wav in path_to_codes:
            record["ref_audio_codes"] = path_to_codes[ref_wav]

        for turn in record.get("conversations", []):
            wav = turn.get("wav")
            if isinstance(wav, str) and wav and wav in path_to_codes:
                turn["audio_codes"] = path_to_codes[wav]


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

    n_before = len(records)
    records = _filter_records(records)
    n_skipped = n_before - len(records)
    if n_skipped:
        accelerator.print(
            f"[prepare_data] Skipped {n_skipped}/{n_before} records "
            f"(no assistant turn in conversations)."
        )
    if not records:
        raise ValueError("All records were skipped — no assistant turns found.")

    codec = load_codec(args.codec_path, device)

    unique_paths = _collect_all_wav_paths(records)
    if unique_paths:
        accelerator.print(
            f"[prepare_data] Encoding {len(unique_paths)} unique wav files "
            f"from {len(records)} records ..."
        )
        coded = batch_encode(
            codec=codec,
            paths=unique_paths,
            batch_size=args.batch_size,
            n_vq=args.n_vq,
            desc="Encoding audio",
            sample_rate=args.codec_sample_rate,
        )
        path_to_codes = {
            p: c.tolist() for p, c in zip(unique_paths, coded)
        }
        _attach_codes(records, path_to_codes)

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

