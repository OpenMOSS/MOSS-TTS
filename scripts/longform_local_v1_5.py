"""Long-form synthesis for MOSS-TTS-Local-Transformer-v1.5 by codec-level continuation.

Modes (--mode):
  b             Plan B. Chunk k continues from prefix = ref codes + gap + codes(k-1);
                text = ref_text + text(k-1) + text(k). Official continuation path.
  continuation  Control group. Prefix = codes(k-1) only; text = text(k-1) + text(k).
  clone         Control group. Every chunk is an independent voice clone (no prefix).

The first chunk is a voice clone when --reference-audio is given, otherwise plain TTS
with no reference (for a single-speaker LoRA, whose voice is in the weights).
Modes b and clone need --reference-audio; continuation works with or without it.
--lora-dir loads a PEFT adapter trained by moss_tts_local_v1.5/finetuning/sft_lora.py.

Chunk size: --chunk-seconds (default 20) packs whole sentences by weighted characters
(OmniVoice's per-script weights), starting from a default speaking rate and recalibrating
from the measured rate after every chunk.

Low-VRAM layout (fits 8 GB cards):
  1. Codec on GPU only to encode the reference, then back to CPU.
  2. TTS model generates every chunk as codec codes (codec stays on CPU).
     The GPU share is free VRAM at load time minus a reserve for generation; layers
     that do not fit are CPU-offloaded by accelerate. --quant int8/nf4 (bitsandbytes)
     is the faster alternative on small cards.
  3. The TTS model is released, then the codec decodes each chunk and the
     concatenated codes (combined.wav) on GPU. ASR / speaker metrics run on CPU.
"""

from __future__ import annotations

import argparse
from collections import deque
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


SENTENCE_BOUNDARY = re.compile(
    r"[.!?]+[\"'\u201d\u2019)\]]*(?=\s)"  # English end + closing quotes, before whitespace
    r"|[\u3002\uff01\uff1f\u2026]+[\u201d\u2019\u300d\u300f\uff09]*"  # CJK end + closing quotes
    r"|\n+"
)

ABBREVIATIONS = {"mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "jr.", "sr.", "vs.", "etc.",
                 "e.g.", "i.e.", "no.", "mt.", "ft.", "inc.", "ltd.", "co.", "u.s.", "a.m.", "p.m."}

def split_sentences(text: str) -> list[str]:
    """Split after sentence-final punctuation or newlines, keeping every character."""
    pieces, start = [], 0
    for match in SENTENCE_BOUNDARY.finditer(text):
        last_token = text[:match.end()].split()[-1] if text[:match.end()].split() else ""
        if last_token.lower() in ABBREVIATIONS or re.fullmatch(r"[A-Z]\.", last_token):
            continue  # "Mr. Smith", "J. K. Rowling": not a sentence end
        pieces.append(text[start:match.end()])
        start = match.end()
    if start < len(text):
        pieces.append(text[start:])
    return [piece for piece in pieces if piece]


SENTENCE_FINAL = re.compile(r"[.!?。！？…][\"'”’)\]」』）]*$")
CLAUSE_FINAL = re.compile(r"[,;:\u2014\uff0c\u3001\uff1b\uff1a][\"'\u201d\u2019)\]\u300d\u300f\uff09]*$")
CJK_TAIL = re.compile(r"[　-鿿＀-￯]$")


def add_pause_markers(text: str, sentence_pause: float, paragraph_pause: float, is_last: bool,
                      clause_pause: float = 0.0) -> str:
    """Insert v1.5 `[pause X.Ys]` markers after sentence-final punctuation.

    A sentence followed by a newline gets `paragraph_pause`, otherwise `sentence_pause`.
    The final sentence of the whole text (is_last) gets none. A chunk that ends on a
    comma / semicolon / colon / dash (an over-long sentence split mid-way) gets
    `clause_pause` at its end only; commas inside a chunk are left to the model.
    """
    pieces = split_sentences(text)
    out = []
    for index, piece in enumerate(pieces):
        body = piece.rstrip()
        trailing = piece[len(body):]
        rest = "".join(pieces[index + 1:])
        chunk_end = not rest.strip()
        if not body or (is_last and chunk_end):
            out.append(piece)
            continue
        if SENTENCE_FINAL.search(body):
            paragraph = "\n" in trailing or rest.lstrip(" \t").startswith("\n")
            pause = paragraph_pause if paragraph else sentence_pause
        elif chunk_end and CLAUSE_FINAL.search(body):
            pause = clause_pause
        else:
            pause = 0.0
        if pause <= 0:
            out.append(piece)
            continue
        sep = "" if CJK_TAIL.search(body) else " "
        out.append(f"{body}{sep}[pause {pause:.1f}s]{trailing}")
    return "".join(out)


# Character weights from OmniVoice's RuleDurationEstimator (k2-fsa/OmniVoice, utils/duration.py):
# roughly "how long a character takes to say", so one seconds budget works across languages.
_SCRIPT_WEIGHTS = (
    ((0x4E00, 0x9FFF), 3.0), ((0x3400, 0x4DBF), 3.0), ((0xF900, 0xFAFF), 3.0),  # CJK ideographs
    ((0x3040, 0x30FF), 2.2), ((0x31F0, 0x31FF), 2.2),                          # kana
    ((0xAC00, 0xD7AF), 2.5), ((0x1100, 0x11FF), 2.5), ((0x3130, 0x318F), 2.5),  # hangul
    ((0x0900, 0x0DFF), 1.8), ((0x1000, 0x109F), 1.8), ((0x1780, 0x17FF), 1.8),  # indic, myanmar, khmer
    ((0x0E00, 0x0EFF), 1.5), ((0x0600, 0x06FF), 1.5), ((0x0590, 0x05FF), 1.5),  # thai/lao, arabic, hebrew
    ((0x1200, 0x139F), 3.0), ((0xA000, 0xA4CF), 3.0),                          # ethiopic, yi
)
# OmniVoice's no-reference fallback: "Nice to meet you." (14.1 weight) spoken in 25 frames at 25 Hz.
DEFAULT_WEIGHT_PER_SECOND = 14.1


def char_weight(ch: str) -> float:
    if ch.isascii() and ch.isalpha():
        return 1.0
    if ch == " ":
        return 0.2
    category = unicodedata.category(ch)
    if category[0] == "M":
        return 0.0
    if category[0] in "PS":
        return 0.5
    if category[0] == "Z" or ch.isspace():
        return 0.2
    if category[0] == "N":
        return 3.5
    code = ord(ch)
    for (low, high), weight in _SCRIPT_WEIGHTS:
        if low <= code <= high:
            return weight
    return 1.0  # latin extended, cyrillic, greek, ... and anything else


def text_weight(text: str) -> float:
    return sum(char_weight(ch) for ch in text)


def next_chunk(queue, budget: float, measure=text_weight) -> str:
    """Pop whole sentences from `queue` (a deque of split_sentences pieces) while
    measure(chunk) <= budget. Only a sentence that alone exceeds the budget is split
    (comma / word boundary); its remainder goes back to the front of the queue.
    Whitespace-only pieces (newlines) that follow are attached to the chunk."""
    current = ""
    while queue:
        sentence = queue[0]
        if not current.strip() and sentence.strip() and measure(sentence) > budget:
            queue.popleft()
            per_char = measure(sentence) / max(len(sentence), 1)
            max_chars = max(1, int(budget / per_char)) if per_char > 0 else len(sentence)
            pieces = list(iter_text_chunks(sentence, max_chars))
            current += pieces[0]
            rest = "".join(pieces[1:])
            if rest:
                queue.appendleft(rest)
            break
        if current.strip() and measure(current + sentence) > budget:
            break
        current += queue.popleft()
    while queue and not queue[0].strip():
        current += queue.popleft()
    return current


def plan_chunks(text: str, max_chars: float, measure=text_weight) -> list[str]:
    """Pack whole sentences up to a budget (weighted characters by default; pass measure=len for plain characters);
    chunks never end mid-sentence unless a single sentence is over budget."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    queue, chunks = deque(split_sentences(text)), []
    while queue:
        chunk = next_chunk(queue, max_chars, measure)
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def chunk_strategy(index: int, mode: str, has_reference: bool) -> str:
    """Return how chunk `index` (0-based) is generated: clone, direct, b, continuation."""
    if index == 0:
        return "clone" if has_reference else "direct"
    if mode == "clone":
        return "clone"
    return mode


def prompt_text(strategy: str, ref_text: str, prev_text: str, text: str) -> str:
    """Prefix transcript + new text, joined exactly as the official examples do."""
    if strategy in {"clone", "direct"}:
        return text
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
    parser.add_argument("--reference-audio", type=Path, default=None,
                        help="Voice reference; omit to synthesize with the model/LoRA voice only")
    parser.add_argument("--reference-text", default=None, help="Exact transcript of the reference audio")
    parser.add_argument("--reference-text-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["b", "continuation", "clone"], default="b")
    parser.add_argument("--lora-dir", default=None,
                        help="PEFT LoRA adapter directory (e.g. output/.../checkpoint-last); needs `pip install peft`")
    parser.add_argument("--chunk-seconds", type=float, default=20.0,
                        help="Target seconds of NEW audio per chunk (default 20). Chunk size is estimated from weighted characters and recalibrated from the measured speaking rate after every chunk")
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
    parser.add_argument("--max-new-tokens", type=positive_int, default=1500, help="Frame budget per chunk (12.5/s)")
    parser.add_argument("--audio-temperature", type=float, default=1.7)
    parser.add_argument("--audio-top-p", type=float, default=0.8)
    parser.add_argument("--audio-top-k", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42, help="Chunk k uses seed + k")
    parser.add_argument("--sentence-pause", type=float, default=0.0,
                        help="Insert [pause Xs] after each sentence (seconds; 0 = off)")
    parser.add_argument("--paragraph-pause", type=float, default=0.0,
                        help="Pause after a sentence that ends a paragraph (seconds; 0 = off)")
    parser.add_argument("--clause-pause", type=float, default=0.0,
                        help="Pause when a chunk ends on a comma/semicolon/colon/dash (seconds; 0 = off)")
    parser.add_argument("--asr-check", action="store_true", help="Transcribe each chunk and report CER (faster-whisper)")
    parser.add_argument("--asr-model", default="large-v3-turbo")
    parser.add_argument("--asr-device", default="cpu")
    parser.add_argument("--speaker-sim", action="store_true",
                        help="Report cosine similarity to the reference (WavLM x-vector, runs on CPU)")
    parser.add_argument("--sv-model", default="microsoft/wavlm-base-plus-sv")
    parser.add_argument("--dry-run", action="store_true", help="Only write the chunk plan")
    args = parser.parse_args()
    if args.reference_audio is None:
        if args.mode in {"b", "clone"}:
            parser.error(f"--mode {args.mode} needs --reference-audio")
        if args.speaker_sim:
            parser.error("--speaker-sim needs --reference-audio to compare against")
    elif not args.reference_audio.is_file():
        parser.error(f"Reference audio not found: {args.reference_audio}")
    if args.reference_text and args.reference_text_file:
        parser.error("Use either --reference-text or --reference-text-file")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory is not empty; choose a new --output-dir")
    if args.gap_ms < 0:
        parser.error("--gap-ms must be >= 0")
    if args.chunk_seconds <= 0:
        parser.error("--chunk-seconds must be > 0")
    if min(args.sentence_pause, args.paragraph_pause, args.clause_pause) < 0:
        parser.error("--sentence-pause / --paragraph-pause / --clause-pause must be >= 0")
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
        free, total = torch.cuda.mem_get_info(device)
        # Keep room for activations / KV cache during generation (small: < 1k frames per chunk).
        reserve = max(1.5 * 2**30, 0.08 * total)
        gpu_budget = max(int(free - reserve), 2**30)
        print(f"[INFO] GPU: {free / 2**30:.1f} GiB free of {total / 2**30:.1f} GiB, "
              f"budget for TTS weights {gpu_budget / 2**30:.1f} GiB", flush=True)
        import psutil  # accelerate needs a CPU entry, otherwise layers that do not fit go to disk
        cpu_budget = int(psutil.virtual_memory().available * 0.9)
        kwargs["max_memory"] = {device.index or 0: gpu_budget, "cpu": cpu_budget}
    model = AutoModel.from_pretrained(args.model_dir, **kwargs)
    if args.lora_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_dir, is_trainable=False)
        if args.quant == "none":
            model = model.merge_and_unload()  # same default as the official app
        print(f"[INFO] LoRA loaded from {args.lora_dir} (merged={args.quant == 'none'})", flush=True)
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
    if not text.strip():
        raise SystemExit("Input TXT contains no non-blank text")
    has_reference = args.reference_audio is not None
    needs_ref_text = args.mode == "b"

    def mark(chunk: str, is_last: bool) -> str:
        # Text actually sent to the model (with pause markers); CER is still scored on the plain text.
        return add_pause_markers(chunk, args.sentence_pause, args.paragraph_pause, is_last, args.clause_pause)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"options": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "chunks": []}

    def save_manifest():
        (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Chunking: {args.chunk_seconds:g}s per chunk (weighted chars, recalibrated per chunk); mode={args.mode}, reference={'yes' if has_reference else 'no'}; output {out}", flush=True)
    if args.dry_run:
        # Preview only: the real plan adapts to the measured speaking rate during generation.
        preview = plan_chunks(text, args.chunk_seconds * DEFAULT_WEIGHT_PER_SECOND)
        for i, chunk in enumerate(preview):
            entry = {"index": i + 1, "strategy": chunk_strategy(i, args.mode, has_reference), "text": chunk,
                     "model_text": mark(chunk, i == len(preview) - 1), "status": "planned",
                     "chars": len(chunk), "weight": round(text_weight(chunk), 1),
                     "estimated_seconds": round(text_weight(chunk) / DEFAULT_WEIGHT_PER_SECOND, 1)}
            manifest["chunks"].append(entry)
        save_manifest()
        print(f"Dry run: {len(preview)} chunks planned at the default speaking rate "
              "(actual chunks adapt during generation)", flush=True)
        return
    save_manifest()

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

    ref_codes = None
    gap_codes = torch.zeros((0, n_vq), dtype=torch.long)
    if has_reference:
        with torch.inference_mode():
            codec.to(device)
            ref_codes = processor.encode_audios_from_path(str(args.reference_audio.resolve()), n_vq=n_vq)[0]
            gap_samples = round(sr * args.gap_ms / 1000)
            if args.mode == "b" and gap_samples >= sr / frame_rate:
                gap_codes = processor.encode_audios_from_wav(torch.zeros(2, gap_samples), sr, n_vq=n_vq)[0]
            codec.to("cpu")
    elif not args.lora_dir:
        print("[WARN] No --reference-audio and no --lora-dir: the base model will pick an arbitrary voice", flush=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    manifest.update(reference_frames=0 if ref_codes is None else int(ref_codes.shape[0]),
                    gap_frames=int(gap_codes.shape[0]))
    save_manifest()

    # ---- Stage 2: TTS model generates codes for every chunk ----
    model = load_tts_model(args, torch, dtype, attn)
    audio_pad = int(model.config.audio_pad_token_id)
    all_codes, prev_codes, prev_text = [], None, ""
    chunks: list[str] = []
    queue = deque(split_sentences(text))
    rate_weight = rate_seconds = 0.0  # measured speaking rate, in weighted chars per second
    with torch.inference_mode():
        while queue:
            rate = rate_weight / rate_seconds if rate_seconds > 0 else DEFAULT_WEIGHT_PER_SECOND
            budget = args.chunk_seconds * rate
            chunk = next_chunk(queue, budget)
            if not chunk.strip():
                continue
            i = len(chunks)
            chunks.append(chunk)
            strategy = chunk_strategy(i, args.mode, has_reference)
            model_text = mark(chunk, not any(piece.strip() for piece in queue))
            entry = {"index": i + 1, "strategy": strategy, "text": chunk, "model_text": model_text,
                     "status": "pending", "chars": len(chunk), "weight": round(text_weight(chunk), 1),
                     "budget": round(budget, 1), "rate_used": round(rate, 2)}
            manifest["chunks"].append(entry)
            print(f"[{i + 1}] {strategy}: {len(chunk)} chars, rate={rate:.2f} w/s, budget={budget:.0f} w"
                  f" | {chunk.strip()[:40]}", flush=True)
            started = time.monotonic()
            # The processor encodes text verbatim; official scripts strip it, so never start with a space.
            full_text = prompt_text(strategy, ref_text, prev_text, model_text).strip()
            if strategy in {"clone", "direct"}:
                reference = [ref_codes] if strategy == "clone" else None
                conversation = [processor.build_user_message(text=full_text, reference=reference, language=language)]
                proc_mode = "generation"
            else:
                prefix = (torch.cat([ref_codes, gap_codes, prev_codes], dim=0)
                          if strategy == "b" else prev_codes)
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
            if not truncated:  # a cut-off chunk would understate the speaking rate
                rate_weight += text_weight(chunk)
                rate_seconds += codes.shape[0] / frame_rate
            save_manifest()
            all_codes.append(codes)
            prev_codes, prev_text = codes, model_text

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
