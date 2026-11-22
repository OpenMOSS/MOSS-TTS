"""SRT-driven dubbing with MOSS-TTS-Local-Transformer-v1.5 voice cloning.

Inputs:
  * source audio (any format librosa can read)
  * source SRT  — transcript of the source audio, with timings
  * target SRT  — translated subtitles (timings usually identical to the source SRT)

Pipeline:
  1. Cut the source audio into per-line reference clips using the source SRT.
  2. For every line of the target SRT, pick the matching source clip (by time overlap)
     and clone it while synthesizing the translated text. Each subtitle line is
     generated as one utterance (no chunking: the SRT already keeps lines short).
     By default the v1.5 duration control (`tokens`, 12.5 frames/s) asks the model
     for audio about as long as the subtitle slot.

     --mode selects how the source clip is used:
       clone (default)     clip in the user message's Reference(s); source text unused
       continuation        clip codes as assistant prefix, text = source text + target text
       continuation_clone  continuation + the same clip also in Reference(s)
                           (closest to VoxCPM's hi-fi mode)
     The continuation modes need the source SRT text to match the clip word for word.
  3. Trim silence, fit each generated clip into its subtitle slot (speed up / pad),
     and place it on a timeline as long as the source audio.

Outputs (in --out-dir):
  refs/0001.wav ...      reference clips cut from the source audio
  segments/<mode>/0001.wav ...  raw synthesized lines (reused on re-run unless --overwrite)
  dubbed_<mode>.wav             the assembled dubbing track
  manifest_<mode>.json          per-line details (reference used, durations, speed factor)
  Outputs are kept per mode, so several modes can share one --out-dir for A/B listening.

Example:
  python scripts/dubbing.py --audio input.wav --src-srt zh.srt --tgt-srt en.srt --out-dir dub_out --language English
  python scripts/dubbing.py ... --mode continuation_clone
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
FRAME_RATE = 12.5  # MOSS-TTS-Local-Transformer-v1.5 codec frames per second

# --------------------------------------------------------------------------- #
# SRT parsing
# --------------------------------------------------------------------------- #

_TIME_RE = re.compile(
    r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)"
)
_TAG_RE = re.compile(r"<[^>]+>|\{\\[^}]*\}")  # <i>..</i>, {\an8} etc.


@dataclass
class Cue:
    index: int
    start: float
    end: float
    text: str


def _to_sec(h, m, s, ms) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")[:3]) / 1000.0


def parse_srt(path: str) -> list[Cue]:
    """Line-based SRT parser; tolerant of BOM, mixed line endings and missing blank lines."""
    with open(path, "r", encoding="utf-8-sig") as f:
        lines = re.split(r"\r\n|\r|\n", f.read())
    raw: list[tuple[float, float, list[str]]] = []
    for line in lines:
        m = _TIME_RE.search(line)
        if m:
            g = m.groups()
            raw.append((_to_sec(*g[:4]), _to_sec(*g[4:]), []))
        elif raw and line.strip():
            raw[-1][2].append(line.strip())
    cues: list[Cue] = []
    for start, end, text_lines in raw:
        # The last line before the next timing line may be the next cue's index number.
        if text_lines and text_lines[-1].isdigit():
            text_lines = text_lines[:-1]
        text = " ".join(_TAG_RE.sub("", l).strip() for l in text_lines).strip()
        if text:
            cues.append(Cue(0, start, end, text))
    cues.sort(key=lambda c: c.start)
    for i, c in enumerate(cues, 1):
        c.index = i
    return cues


# --------------------------------------------------------------------------- #
# Matching target cues to source cues
# --------------------------------------------------------------------------- #


def match_source(tgt: Cue, src_cues: list[Cue]) -> int:
    """Index into src_cues of the cue overlapping tgt most (nearest midpoint if none overlap)."""
    best, best_ov = -1, 0.0
    for i, s in enumerate(src_cues):
        ov = min(tgt.end, s.end) - max(tgt.start, s.start)
        if ov > best_ov:
            best, best_ov = i, ov
    if best >= 0:
        return best
    mid = (tgt.start + tgt.end) / 2
    return min(range(len(src_cues)), key=lambda i: abs((src_cues[i].start + src_cues[i].end) / 2 - mid))


def usable_ref(idx: int, src_cues: list[Cue], min_sec: float) -> int:
    """If the matched clip is too short for good cloning, fall back to the nearest long-enough one."""
    if src_cues[idx].end - src_cues[idx].start >= min_sec:
        return idx
    candidates = [i for i, c in enumerate(src_cues) if c.end - c.start >= min_sec]
    if not candidates:
        return idx
    return min(candidates, key=lambda i: abs(i - idx))


# --------------------------------------------------------------------------- #
# Audio helpers
# --------------------------------------------------------------------------- #


MODES = ("clone", "continuation", "continuation_clone")


def join_prompt(prompt: str, text: str) -> str:
    """Source text + target text for continuation; a space only between two ASCII boundaries."""
    prompt, text = prompt.strip(), text.strip()
    if not prompt:
        return text
    sep = " " if prompt[-1].isascii() and text[:1].isascii() else ""
    return prompt + sep + text


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def fit_to_slot(wav: np.ndarray, sr: int, slot: float, max_speed: float) -> tuple[np.ndarray, float]:
    """Speed up (pitch-preserving) up to max_speed so wav fits in `slot` seconds."""
    dur = len(wav) / sr
    if slot <= 0 or dur <= slot:
        return wav, 1.0
    rate = min(dur / slot, max_speed)
    if rate > 1.01:
        wav = librosa.effects.time_stretch(wav, rate=rate)
    return wav, rate


def fade_out(wav: np.ndarray, sr: int, ms: float = 30.0) -> np.ndarray:
    n = min(len(wav), int(sr * ms / 1000))
    if n > 0:
        wav = wav.copy()
        wav[-n:] *= np.linspace(1.0, 0.0, n, dtype=wav.dtype)
    return wav


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser(description="SRT-based dubbing with MOSS-TTS-Local-Transformer-v1.5 cloning")
    ap.add_argument("--audio", required=True, help="Source audio file")
    ap.add_argument("--src-srt", required=True, help="SRT transcript of the source audio")
    ap.add_argument("--tgt-srt", required=True, help="Translated SRT to be dubbed")
    ap.add_argument("--out-dir", default="dubbing_out")
    ap.add_argument("--model-dir", default="OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5")
    ap.add_argument("--codec-dir", default="OpenMOSS-Team/MOSS-Audio-Tokenizer-v2")
    ap.add_argument("--lora-dir", default=None, help="Optional PEFT adapter (finetuning/sft_lora.py)")
    ap.add_argument("--mode", choices=MODES, default="clone",
                    help="clone: Reference(s) only (default); continuation: clip as prefix + source text; "
                         "continuation_clone: both (VoxCPM hi-fi style)")
    ap.add_argument("--continuation-tokens", choices=["new", "total"], default="new",
                    help="Continuation modes: `tokens` covers only the new audio (new, like the v1.5 app) "
                         "or prefix + new audio (total). Check generated_sec in the manifest to see which fits")
    ap.add_argument("--language", default="auto",
                    help="Target language tag, e.g. English / Chinese (auto omits the tag)")
    ap.add_argument("--device", default=None, help="Default: CUDA if available, otherwise CPU")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default=None)
    ap.add_argument("--attn-implementation", choices=["auto", "sdpa", "eager", "flash_attention_2"],
                    default="auto")
    ap.add_argument("--codec-offload", action=argparse.BooleanOptionalAction, default=True,
                    help="Move the audio tokenizer to CPU while the TTS model generates (default: enabled)")
    ap.add_argument("--duration-control", action=argparse.BooleanOptionalAction, default=True,
                    help="Ask the model for audio as long as the subtitle slot via `tokens` (default: enabled)")
    ap.add_argument("--duration-scale", type=float, default=1.0,
                    help="Multiply the requested duration (e.g. 0.95 to leave a little headroom)")
    ap.add_argument("--max-new-tokens", type=int, default=1500, help="Frame cap per line (12.5 frames = 1 s)")
    ap.add_argument("--temperature", type=float, default=1.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=25)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pad", type=float, default=0.05, help="Seconds of context added around each reference cut")
    ap.add_argument("--min-ref-sec", type=float, default=1.5,
                    help="References shorter than this borrow the nearest longer source line")
    ap.add_argument("--max-speed", type=float, default=1.35,
                    help="Max speed-up to fit a line into its slot (1.0 disables stretching)")
    ap.add_argument("--allow-overflow", action="store_true",
                    help="Let lines run into the gap before the next subtitle instead of being cut")
    ap.add_argument("--no-loudness-match", action="store_true", help="Don't match each line's RMS to its reference")
    ap.add_argument("--mix-original", type=float, default=0.0,
                    help="Also write dubbed_<mode>_mix.wav with the original audio mixed in at this gain (e.g. 0.15)")
    ap.add_argument("--overwrite", action="store_true", help="Re-synthesize lines that already exist")
    ap.add_argument("--only", type=str, default=None, help="Only (re)generate these target line numbers, e.g. 3,7,12")
    args = ap.parse_args()

    # Subtitle text is printed to the console; avoid UnicodeEncodeError on cp1252/gbk consoles.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except AttributeError:
            pass

    src_cues = parse_srt(args.src_srt)
    tgt_cues = parse_srt(args.tgt_srt)
    if not src_cues or not tgt_cues:
        sys.exit("Empty or unparsable SRT file.")
    print(f"Source cues: {len(src_cues)}  Target cues: {len(tgt_cues)}  Mode: {args.mode}", file=sys.stderr)

    ref_dir = os.path.join(args.out_dir, "refs")
    seg_dir = os.path.join(args.out_dir, "segments", args.mode)
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)

    # 1. Cut reference clips --------------------------------------------------
    src_wav, src_sr = librosa.load(args.audio, sr=None, mono=True)
    total_sec = len(src_wav) / src_sr
    ref_paths, ref_rms = [], []
    for c in src_cues:
        a = max(0, int((c.start - args.pad) * src_sr))
        b = min(len(src_wav), int((c.end + args.pad) * src_sr))
        clip = src_wav[a:b]
        p = os.path.join(ref_dir, f"{c.index:04d}.wav")
        sf.write(p, clip, src_sr)
        ref_paths.append(p)
        ref_rms.append(rms(clip))
    print(f"Wrote {len(ref_paths)} reference clips to {ref_dir}", file=sys.stderr)

    # Decide per target line which reference it uses and whether it must be generated.
    only = {int(x) for x in args.only.split(",")} if args.only else None
    plan = []
    for t in tgt_cues:
        si = usable_ref(match_source(t, src_cues), src_cues, args.min_ref_sec)
        seg_path = os.path.join(seg_dir, f"{t.index:04d}.wav")
        if only is not None:
            need_gen = t.index in only or not os.path.exists(seg_path)
        else:
            need_gen = args.overwrite or not os.path.exists(seg_path)
        plan.append((t, si, seg_path, need_gen))

    # 2. Load model (only if something has to be generated) --------------------
    runtime = None
    sr = None
    if any(p[3] for p in plan):
        import torch
        from transformers import set_seed

        sys.path.insert(0, str(ROOT / "moss_tts_local_v1.5"))
        from streaming import build_continuation_clone_batch, load_runtime

        torch.backends.cuda.enable_cudnn_sdp(False)
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = args.dtype or ("bf16" if torch.device(device).type == "cuda" else "fp32")
        set_seed(args.seed)
        runtime = load_runtime(model_dir=args.model_dir, lora_dir=args.lora_dir, codec_dir=args.codec_dir,
                               device=device, codec_device=device, dtype=dtype,
                               codec_weight_dtype="bf16", codec_compute_dtype="bf16",
                               attn_implementation=args.attn_implementation,
                               codec_offload=args.codec_offload, warmup=False)
        sr = int(runtime.sample_rate)
        processor = runtime.processor
        codec = processor.audio_tokenizer
        codec_device = runtime.codec_device

        @contextmanager
        def codec_on_device():
            if args.codec_offload and codec_device.type == "cuda":
                codec.to(codec_device)
            try:
                yield
            finally:
                if args.codec_offload and codec_device.type == "cuda":
                    codec.to("cpu")
                    torch.cuda.empty_cache()

        # Encode every needed reference once, up front, so the codec can stay off the GPU
        # while the language model generates.
        needed_refs = sorted({si for _, si, _, need in plan if need})
        ref_codes: dict[int, "torch.Tensor"] = {}
        with torch.inference_mode(), codec_on_device():
            for si in needed_refs:
                ref_codes[si] = processor.encode_audios_from_path(ref_paths[si])[0]
        print(f"Encoded {len(ref_codes)} reference clips", file=sys.stderr)
        language = None if args.language.lower() == "auto" else args.language
    else:
        sr = sf.info(plan[0][2]).samplerate
        print("All segments exist; reassembling without loading the model.", file=sys.stderr)

    timeline = np.zeros(int(np.ceil(max(total_sec, tgt_cues[-1].end + 1.0) * sr)), dtype=np.float32)
    manifest = []

    # 3. Synthesize + place ---------------------------------------------------
    for k, (t, si, seg_path, need_gen) in enumerate(plan):
        s = src_cues[si]
        slot = t.end - t.start
        continuation = args.mode != "clone"
        model_text = join_prompt(s.text, t.text) if continuation else t.text
        tokens = None
        if args.duration_control and slot > 0:
            tokens = max(1, int(round(slot * args.duration_scale * FRAME_RATE)))

        if need_gen:
            if tokens is not None and continuation and args.continuation_tokens == "total":
                tokens += int(ref_codes[si].shape[0])
            print(f"[{t.index}/{len(tgt_cues)}] ref#{s.index} tokens={tokens} | {model_text}", file=sys.stderr)
            with torch.inference_mode():
                ref = ref_codes[si]
                # In the continuation modes the prefix is decoded with the new audio and
                # trimmed off again by processor.decode.
                if args.mode == "continuation_clone":
                    user = processor.build_user_message(text=model_text, tokens=tokens, language=language)
                    batch = build_continuation_clone_batch(processor, user, ref)
                elif args.mode == "continuation":
                    conversation = [
                        processor.build_user_message(text=model_text, tokens=tokens, language=language),
                        processor.build_assistant_message(audio_codes_list=[ref]),
                    ]
                    batch = processor([conversation], mode="continuation")
                else:
                    conversation = [processor.build_user_message(
                        text=model_text, reference=[ref], tokens=tokens, language=language)]
                    batch = processor([conversation], mode="generation")
                outputs = runtime.model.generate(
                    input_ids=batch["input_ids"].to(runtime.device),
                    attention_mask=batch["attention_mask"].to(runtime.device),
                    max_new_tokens=args.max_new_tokens, do_sample=True,
                    audio_temperature=args.temperature, audio_top_p=args.top_p, audio_top_k=args.top_k,
                    audio_repetition_penalty=args.repetition_penalty)
                with codec_on_device():
                    messages = processor.decode(outputs)
            if not messages or messages[0] is None or not messages[0].audio_codes_list:
                print(f"  ! line {t.index}: no audio generated, skipped", file=sys.stderr)
                continue
            audio = messages[0].audio_codes_list[0].detach().float().cpu().numpy()
            wav = audio.mean(axis=0) if audio.ndim == 2 else audio.reshape(-1)
            del outputs, messages, batch
            if wav.size == 0:
                print(f"  ! line {t.index}: empty audio, skipped", file=sys.stderr)
                continue
            sf.write(seg_path, wav, sr)
        else:
            wav, _ = librosa.load(seg_path, sr=sr, mono=True)

        wav = np.asarray(wav, dtype=np.float32)
        raw_dur = len(wav) / sr
        trimmed, _ = librosa.effects.trim(wav, top_db=40)
        if trimmed.size:
            wav = trimmed

        if not args.no_loudness_match and ref_rms[si] > 0 and rms(wav) > 0:
            wav = wav * (ref_rms[si] / rms(wav))

        next_start = plan[k + 1][0].start if k + 1 < len(plan) else timeline.size / sr
        hard_limit = (next_start - t.start) if args.allow_overflow else slot
        wav, speed = fit_to_slot(wav, sr, slot, args.max_speed)
        truncated = False
        max_n = int(max(hard_limit, slot) * sr)
        if len(wav) > max_n:
            wav = fade_out(wav[:max_n], sr)
            truncated = True

        a = int(t.start * sr)
        b = min(a + len(wav), timeline.size)
        timeline[a:b] += wav[: b - a]

        manifest.append({
            "mode": args.mode,
            "target": asdict(t),
            "model_text": model_text,
            "reference": {"index": s.index, "text": s.text, "path": ref_paths[si]},
            "requested_tokens": tokens,
            "generated_sec": round(raw_dur, 3),
            "placed_sec": round(len(wav) / sr, 3),
            "slot_sec": round(slot, 3),
            "speed": round(speed, 3),
            "truncated": truncated,
        })
        if truncated:
            print(f"  ! line {t.index} truncated (consider shortening the translation or --allow-overflow)",
                  file=sys.stderr)

    # 4. Write outputs ----------------------------------------------------------
    peak = float(np.max(np.abs(timeline))) if timeline.size else 0.0
    if peak > 0.99:
        timeline *= 0.99 / peak
    out_path = os.path.join(args.out_dir, f"dubbed_{args.mode}.wav")
    sf.write(out_path, timeline, sr)
    print(f"Dubbed track: {out_path}", file=sys.stderr)

    if args.mix_original > 0:
        orig = librosa.resample(src_wav, orig_sr=src_sr, target_sr=sr) if src_sr != sr else src_wav
        mix = timeline.copy()
        n = min(len(orig), len(mix))
        mix[:n] += args.mix_original * orig[:n]
        peak = float(np.max(np.abs(mix)))
        if peak > 0.99:
            mix *= 0.99 / peak
        mix_path = os.path.join(args.out_dir, f"dubbed_{args.mode}_mix.wav")
        sf.write(mix_path, mix, sr)
        print(f"Mixed track:  {mix_path}", file=sys.stderr)

    with open(os.path.join(args.out_dir, f"manifest_{args.mode}.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    n_trunc = sum(m["truncated"] for m in manifest)
    n_fast = sum(m["speed"] > 1.01 for m in manifest)
    print(f"Done. {len(manifest)} lines, {n_fast} sped up, {n_trunc} truncated.", file=sys.stderr)


if __name__ == "__main__":
    main()
