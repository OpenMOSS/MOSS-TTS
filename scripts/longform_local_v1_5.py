"""Long-form synthesis for MOSS-TTS-Local-Transformer-v1.5 by codec-level continuation.

Modes (--mode):
  b             Plan B. Chunk k continues from prefix = ref codes + gap + codes(k-1);
                text = ref_text + text(k-1) + text(k). Official continuation path.
  continuation  Control group. Prefix = codes(k-1) only; text = text(k-1) + text(k).
  clone         Control group. Every chunk is an independent voice clone (no prefix).

The first chunk uses --first-mode: "clone" (Reference = ref audio) or
"continuation" (prefix = ref codes, text = ref_text + text(1)).

Low-VRAM layout (fits 8 GB cards):
  1. Codec on GPU only to encode the reference, then back to CPU.
  2. TTS model generates every chunk as codec codes (codec stays on CPU).
     --gpu-memory caps the GPU share; the remaining layers are CPU-offloaded
     by accelerate. --quant int8/nf4 (bitsandbytes) is the faster alternative.
  3. The TTS model is released, then the codec decodes each chunk and the
     concatenated codes (combined.wav) on GPU. ASR / speaker metrics run on CPU.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import re
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from clis.tts_chunking import iter_text_chunks  # noqa: E402

SENTENCE_END = re.compile(r"[。？！，、；：…—～.?!,;:~\-\"'\)\]\}\>”’）》】〉]\s*$")
QUANT_SKIP_MODULES = ["text_lm_head", "local_text_lm_head", "audio_lm_heads", "audio_embeddings", "local_transformer"]


def ensure_terminal_punctuation(text: str) -> str:
    text = text.strip()
    if not text or SENTENCE_END.search(text):
        return text
    return text + ("。" if re.search(r"[一-鿿]", text) else ".")


def plan_chunks(text: str, max_chars: int) -> list[str]:
    return [chunk for chunk in iter_text_chunks(text, max_chars) if chunk.strip()]


def chunk_strategy(index: int, mode: str, first_mode: str) -> str:
    """Return how chunk `index` (0-based) is generated: clone, ref_continuation, b, continuation."""
    if index == 0:
        return "clone" if first_mode == "clone" else "ref_continuation"
    if mode == "clone":
        return "clone"
    return mode


def prompt_text(strategy: str, ref_text: str, prev_text: str, text: str) -> str:
    """Prefix transcript + new text, joined exactly as the official examples do."""
    if strategy == "clone":
        return text
    if strategy == "ref_continuation":
        return ref_text + text
    if strategy == "b":
        return ref_text + prev_text + text
    if strategy == "continuation":
        return prev_text + text
    raise ValueError(f"Unknown strategy: {strategy}")


def normalize_for_cer(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(ch for ch in text if not (unicodedata.category(ch)[0] in "PZS" or ch.isspace()))


def cer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize_for_cer(reference), normalize_for_cer(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        current = [i]
        for j, h in enumerate(hyp, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h)))
        previous = current
    return previous[-1] / len(ref)


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-txt", type=Path, required=True)
    parser.add_argument("--reference-audio", type=Path, required=True)
    parser.add_argument("--reference-text", default=None, help="Exact transcript of the reference audio")
    parser.add_argument("--reference-text-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["b", "continuation", "clone"], default="b")
    parser.add_argument("--first-mode", choices=["clone", "continuation"], default="clone")
    parser.add_argument("--chunk-chars", type=positive_int, default=80,
                        help="Max characters of NEW text per chunk (default 80)")
    parser.add_argument("--gap-ms", type=int, default=300,
                        help="Silence between ref codes and previous-chunk codes in the B prefix")
    parser.add_argument("--model-dir", default="OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5")
    parser.add_argument("--codec-dir", default="OpenMOSS-Team/MOSS-Audio-Tokenizer-v2")
    parser.add_argument("--language", default="Chinese", help="Use auto to omit the language tag")
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--attn-implementation", choices=["auto", "sdpa", "eager", "flash_attention_2"], default="sdpa")
    parser.add_argument("--quant", choices=["none", "int8", "nf4"], default="none",
                        help="bitsandbytes quantization of the TTS backbone (needs `pip install bitsandbytes`)")
    parser.add_argument("--gpu-memory", default="6GiB",
                        help="GPU budget for TTS weights when --quant none; the rest is CPU-offloaded")
    parser.add_argument("--cpu-memory", default="48GiB")
    parser.add_argument("--max-new-tokens", type=positive_int, default=1500, help="Frame budget per chunk (12.5/s)")
    parser.add_argument("--audio-temperature", type=float, default=1.7)
    parser.add_argument("--audio-top-p", type=float, default=0.8)
    parser.add_argument("--audio-top-k", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42, help="Chunk k uses seed + k")
    parser.add_argument("--asr-check", action="store_true", help="Transcribe each chunk and report CER (faster-whisper)")
    parser.add_argument("--asr-model", default="large-v3-turbo")
    parser.add_argument("--asr-device", default="cpu")
    parser.add_argument("--speaker-sim", action="store_true",
                        help="Report cosine similarity to the reference (WavLM x-vector, runs on CPU)")
    parser.add_argument("--sv-model", default="microsoft/wavlm-base-plus-sv")
    parser.add_argument("--dry-run", action="store_true", help="Only write the chunk plan")
    args = parser.parse_args()
    if not args.reference_audio.is_file():
        parser.error(f"Reference audio not found: {args.reference_audio}")
    if args.reference_text and args.reference_text_file:
        parser.error("Use either --reference-text or --reference-text-file")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory is not empty; choose a new --output-dir")
    if args.gap_ms < 0:
        parser.error("--gap-ms must be >= 0")
    return args


def load_tts_model(args, torch, dtype, attn):
    from transformers import AutoModel
    kwargs = dict(trust_remote_code=True, dtype=dtype, attn_implementation=attn,
                  local_files_only=Path(args.model_dir).exists())
    device = torch.device(args.device)
    if args.quant != "none":
        from transformers import BitsAndBytesConfig
        if args.quant == "int8":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True, llm_int8_skip_modules=QUANT_SKIP_MODULES)
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
                llm_int8_skip_modules=QUANT_SKIP_MODULES)
        kwargs["device_map"] = {"": device.index or 0}
    elif device.type == "cuda":
        # accelerate keeps as many decoder layers as fit on the GPU and streams the rest from CPU.
        kwargs["device_map"] = "auto"
        kwargs["max_memory"] = {device.index or 0: args.gpu_memory, "cpu": args.cpu_memory}
    model = AutoModel.from_pretrained(args.model_dir, **kwargs)
    if "device_map" not in kwargs:
        model.to(device)
    model.eval()
    placement = getattr(model, "hf_device_map", None)
    if placement:
        offloaded = sorted({k for k, v in placement.items() if str(v) in {"cpu", "disk"}})
        print(f"[INFO] TTS modules offloaded to CPU: {len(offloaded)}"
              + (f" (e.g. {offloaded[0]} ... {offloaded[-1]})" if offloaded else ""), flush=True)
    if device.type == "cuda":
        print(f"[INFO] GPU allocated after TTS load: {torch.cuda.memory_allocated(device) / 2**30:.2f} GiB", flush=True)
    return model


def main() -> None:
    args = parse_args()
    try:
        text = args.input_txt.read_text(encoding=args.encoding)
    except (OSError, UnicodeError, LookupError) as exc:
        raise SystemExit(str(exc))
    chunks = plan_chunks(text, args.chunk_chars)
    if not chunks:
        raise SystemExit("Input TXT contains no non-blank text")
    strategies = [chunk_strategy(i, args.mode, args.first_mode) for i in range(len(chunks))]
    needs_ref_text = any(s in {"b", "ref_continuation"} for s in strategies)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"options": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "chunks": [{"index": i + 1, "strategy": s, "text": c, "status": "pending"}
                           for i, (c, s) in enumerate(zip(chunks, strategies))]}

    def save_manifest():
        (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    save_manifest()
    print(f"Planned {len(chunks)} chunks (mode={args.mode}, first={args.first_mode}) in {out}", flush=True)
    if args.dry_run:
        return

    import torch
    import torchaudio
    from transformers import AutoProcessor, set_seed

    sys.path.insert(0, str(ROOT / "moss_tts_local_v1.5"))
    from streaming import _resolve_dtype, resolve_attn_implementation

    language = None if args.language.lower() == "auto" else args.language
    ref_text = ""
    if needs_ref_text:
        if args.reference_text_file:
            ref_text = args.reference_text_file.read_text(encoding="utf-8-sig")
        elif args.reference_text:
            ref_text = args.reference_text
        else:
            from clis.tts_chunking import transcribe_reference
            ref_text = transcribe_reference(args.reference_audio, args.asr_model, args.asr_device, args.language)
            print(f"[WARN] Reference transcript from ASR, please verify: {ref_text}", flush=True)
        ref_text = ensure_terminal_punctuation(ref_text)
        if not ref_text:
            raise SystemExit("Reference transcript is empty")
        (out / "reference_text.txt").write_text(ref_text, encoding="utf-8")
        manifest["reference_text"] = ref_text
        save_manifest()

    torch.backends.cuda.enable_cudnn_sdp(False)
    device = torch.device(args.device)
    dtype = _resolve_dtype({"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}[args.dtype])
    attn = resolve_attn_implementation(args.attn_implementation, device, dtype)

    # ---- Stage 1: codec encodes the reference (TTS model not loaded yet) ----
    processor = AutoProcessor.from_pretrained(
        args.model_dir, trust_remote_code=True, codec_path=args.codec_dir,
        codec_weight_dtype="bf16", codec_compute_dtype="bf16", codec_attention_implementation=attn)
    codec = processor.audio_tokenizer
    if hasattr(codec, "set_attention_implementation"):
        codec.set_attention_implementation(attn)
    if hasattr(codec, "set_compute_dtype"):
        codec.set_compute_dtype("bf16")
    codec.eval()
    n_vq = int(processor.model_config.n_vq)
    sr = int(getattr(codec, "sampling_rate", processor.model_config.sampling_rate))
    frame_rate = sr / float(getattr(codec, "downsample_rate", 3840))

    with torch.inference_mode():
        codec.to(device)
        ref_codes = processor.encode_audios_from_path(str(args.reference_audio.resolve()), n_vq=n_vq)[0]
        gap_samples = round(sr * args.gap_ms / 1000)
        gap_codes = (processor.encode_audios_from_wav(torch.zeros(2, gap_samples), sr, n_vq=n_vq)[0]
                     if gap_samples >= sr / frame_rate else ref_codes[:0])
        codec.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    manifest.update(reference_frames=int(ref_codes.shape[0]), gap_frames=int(gap_codes.shape[0]))
    save_manifest()

    # ---- Stage 2: TTS model generates codes for every chunk ----
    model = load_tts_model(args, torch, dtype, attn)
    audio_pad = int(model.config.audio_pad_token_id)
    all_codes, prev_codes, prev_text = [], None, ""
    with torch.inference_mode():
        for i, (chunk, strategy) in enumerate(zip(chunks, strategies)):
            entry = manifest["chunks"][i]
            print(f"[{i + 1}/{len(chunks)}] {strategy}: {chunk.strip()[:40]}", flush=True)
            started = time.monotonic()
            full_text = prompt_text(strategy, ref_text, prev_text, chunk)
            if strategy == "clone":
                conversation = [processor.build_user_message(text=full_text, reference=[ref_codes], language=language)]
                proc_mode = "generation"
            else:
                prefix = {"ref_continuation": ref_codes,
                          "b": torch.cat([ref_codes, gap_codes, prev_codes], dim=0),
                          "continuation": prev_codes}[strategy]
                conversation = [processor.build_user_message(text=full_text, language=language),
                                processor.build_assistant_message(audio_codes_list=[prefix])]
                proc_mode = "continuation"
            batch = processor(conversation, mode=proc_mode, n_vq=n_vq)
            set_seed(args.seed + i)
            start_length, ids = model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                max_new_tokens=args.max_new_tokens, do_sample=True,
                audio_temperature=args.audio_temperature, audio_top_p=args.audio_top_p,
                audio_top_k=args.audio_top_k, audio_repetition_penalty=1.0)[0]
            # ids starts at the last <audio_start>; the first start_length frames are the prefix.
            new_rows = ids[1 + int(start_length):, 1:]
            codes = new_rows[~new_rows.eq(audio_pad).all(dim=1)].contiguous().cpu()
            if codes.shape[0] == 0:
                entry.update(status="failed", error="no audio generated")
                save_manifest()
                raise RuntimeError(f"Chunk {i + 1}: no audio generated")
            truncated = int(new_rows.shape[0]) >= args.max_new_tokens
            torch.save(codes, out / f"{i + 1:04d}.pt")
            new_chars = len(normalize_for_cer(chunk))
            entry.update(status="generated", prefix_frames=int(start_length), frames=int(codes.shape[0]),
                         seconds=round(codes.shape[0] / frame_rate, 2), truncated=truncated,
                         frames_per_char=round(codes.shape[0] / max(new_chars, 1), 2),
                         gen_seconds=round(time.monotonic() - started, 2))
            print(f"  {entry['seconds']}s audio in {entry['gen_seconds']}s, frames/char={entry['frames_per_char']}"
                  + ("  [WARN] hit --max-new-tokens, chunk may be cut" if truncated else ""), flush=True)
            save_manifest()
            all_codes.append(codes)
            prev_codes, prev_text = codes, chunk

    # ---- Stage 3: release TTS model, decode on GPU, metrics on CPU ----
    del model, batch, ids, new_rows
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    asr = None
    if args.asr_check:
        from clis.tts_chunking import transcribe_reference as asr
    sv = None
    if args.speaker_sim:
        from transformers import AutoFeatureExtractor, WavLMForXVector
        sv_fe = AutoFeatureExtractor.from_pretrained(args.sv_model)
        sv_model = WavLMForXVector.from_pretrained(args.sv_model).eval()

        def sv(wav, rate):  # wav: [channels, samples]
            mono = torchaudio.functional.resample(wav.mean(0), rate, 16000).numpy()
            inputs = sv_fe(mono, sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                emb = sv_model(**inputs).embeddings
            return torch.nn.functional.normalize(emb, dim=-1)[0]

        ref_emb = sv(*torchaudio.load(str(args.reference_audio)))

    with torch.inference_mode():
        codec.to(device)
        for i, codes in enumerate(all_codes):
            entry = manifest["chunks"][i]
            wav = processor.decode_audio_codes([codes])[0].float().cpu()
            wav_path = out / f"{i + 1:04d}.wav"
            torchaudio.save(str(wav_path), wav, sr)
            entry["status"] = "done"
            if asr:
                try:
                    hyp = asr(wav_path, args.asr_model, args.asr_device, args.language)
                    entry.update(asr_text=hyp, cer=round(cer(chunks[i], hyp), 4))
                except Exception as exc:  # noqa: BLE001 - metric only
                    entry.update(asr_error=str(exc))
            if sv:
                entry["speaker_sim"] = round(float(torch.dot(sv(wav, sr), ref_emb)), 4)
            print(f"[decode {i + 1}/{len(all_codes)}] " + ", ".join(
                f"{k}={entry[k]}" for k in ("seconds", "cer", "speaker_sim") if k in entry), flush=True)
            save_manifest()

        combined_codes = torch.cat(all_codes, dim=0)
        torch.save(combined_codes, out / "combined.pt")
        torchaudio.save(str(out / "combined.wav"), processor.decode_audio_codes([combined_codes])[0].float().cpu(), sr)
        codec.to("cpu")
    manifest["combined_seconds"] = round(combined_codes.shape[0] / frame_rate, 2)
    save_manifest()
    print(f"Done: {out / 'combined.wav'} ({manifest['combined_seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
