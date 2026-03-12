from __future__ import annotations

import argparse
import time
import wave
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torchaudio
from transformers import AutoTokenizer, AutoModel

from mossttsrealtime.modeling_mossttsrealtime import MossTTSRealtime
from mossttsrealtime.processing_mossttsrealtime import MossTTSRealtimeProcessor
from mossttsrealtime.streaming_mossttsrealtime import (
    AudioStreamDecoder,
    MossTTSRealtimeInference,
    MossTTSRealtimeStreamingSession,
)

SAMPLE_RATE = 24000


def _extract_codes(encode_result):
    return encode_result["audio_codes"]


def _load_audio(path: Path, target_sample_rate: int = 24000) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if sr != target_sample_rate:
        wav = torchaudio.functional.resample(wav, sr, target_sample_rate)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav


def _load_codec(codec_path: str, device: torch.device):
    codec = AutoModel.from_pretrained(codec_path, trust_remote_code=True).eval()
    return codec.to(device)


def _sanitize_tokens(
    tokens: torch.Tensor,
    codebook_size: int,
    audio_eos_token: int,
) -> tuple[torch.Tensor, bool]:
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
    if tokens.numel() == 0:
        return tokens, False
    eos_rows = (tokens[:, 0] == audio_eos_token).nonzero(as_tuple=False)
    invalid_rows = ((tokens < 0) | (tokens >= codebook_size)).any(dim=1)
    stop_idx = None
    if eos_rows.numel() > 0:
        stop_idx = int(eos_rows[0].item())
    if invalid_rows.any():
        invalid_idx = int(invalid_rows.nonzero(as_tuple=False)[0].item())
        stop_idx = invalid_idx if stop_idx is None else min(stop_idx, invalid_idx)
    if stop_idx is not None:
        tokens = tokens[:stop_idx]
        return tokens, True
    return tokens, False


def decode_audio_frames(
    audio_frames: list[torch.Tensor],
    decoder: AudioStreamDecoder,
    codebook_size: int,
    audio_eos_token: int,
) -> Iterator[np.ndarray]:
    for frame in audio_frames:
        tokens = frame
        if tokens.dim() == 3:
            tokens = tokens[0]
        if tokens.dim() != 2:
            raise ValueError(f"Expected [T, C] audio tokens, got {tuple(tokens.shape)}")
        tokens, _ = _sanitize_tokens(tokens, codebook_size, audio_eos_token)
        if tokens.numel() == 0:
            continue
        decoder.push_tokens(tokens.detach())
        for wav in decoder.audio_chunks():
            if wav.numel() == 0:
                continue
            yield wav.detach().cpu().numpy().reshape(-1)


def flush_decoder(decoder: AudioStreamDecoder) -> Iterator[np.ndarray]:
    """与 gradio AudioFrameDecoder.flush 完全一致。"""
    final_chunk = decoder.flush()
    if final_chunk is not None and final_chunk.numel() > 0:
        yield final_chunk.detach().cpu().numpy().reshape(-1)



def fake_llm_text_stream(
    text: str,
    chunk_chars: int = 1,
    delay_s: float = 0.0,
) -> Iterator[str]:
    """
    Simulate streaming text deltas from an LLM.
    Each iteration yields `chunk_chars` characters with a delay of `delay_s` seconds.
    In real-world usage, this can be replaced with streaming responses from models such as OpenAI or vLLM.
    """
    if not text:
        return
    step = max(1, chunk_chars)
    for idx in range(0, len(text), step):
        if delay_s > 0 and idx > 0:
            time.sleep(delay_s)
        yield text[idx : idx + step]



def write_wav(out_path: Path, sample_rate: int, chunks: Iterator[np.ndarray]) -> None:
    all_chunks: list[np.ndarray] = []
    for chunk in chunks:
        all_chunks.append(chunk.astype(np.float32).reshape(-1))

    if not all_chunks:
        raise RuntimeError("No audio chunks produced.")

    audio = np.concatenate(all_chunks)
    # float32 → int16 PCM
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype(np.int16)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm16.tobytes())



# Core: Streaming generation: text delta → push_text → audio
def run_streaming_tts(
    session: MossTTSRealtimeStreamingSession,
    codec,
    decoder: AudioStreamDecoder,
    text_deltas: Iterator[str],
) -> Iterator[np.ndarray]:
    """
    Receives streaming text deltas, feeds them into the TTS via `session.push_text()`,
    and produces playable WAV chunks in real time.

    The pipeline matches the Gradio demo:
    codec.streaming → push_text → decode_frames → end_text → drain → flush

    Args:
        session: A streaming session that has been initialized and `reset_turn` has been called.
        codec: The audio codec (used for streaming context).
        decoder: An `AudioStreamDecoder` instance.
        text_deltas: An iterator of text deltas (simulating LLM streaming output).
    """
    codebook_size = int(getattr(codec, "codebook_size", 1024))
    audio_eos_token = int(getattr(session.inferencer, "audio_eos_token", 1026))

    with codec.streaming(batch_size=1):
        for delta in text_deltas:
            print(delta, end="", flush=True)
            audio_frames = session.push_text(delta)
            yield from decode_audio_frames(
                audio_frames, decoder, codebook_size, audio_eos_token
            )

        audio_frames = session.end_text()
        yield from decode_audio_frames(
            audio_frames, decoder, codebook_size, audio_eos_token
        )

        while True:
            audio_frames = session.drain(max_steps=1)
            if not audio_frames:
                break
            yield from decode_audio_frames(
                audio_frames, decoder, codebook_size, audio_eos_token
            )
            if session.inferencer.is_finished:
                break

        yield from flush_decoder(decoder)


def main():
    p = argparse.ArgumentParser(
        description="Simulated LLM streaming text → TTS streaming audio。"
    )
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--codec_path", type=str, required=True)
    p.add_argument("--prompt_wav", type=str, required=True)
    p.add_argument("--out_wav", type=str, default="out_streaming.wav")

    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--attn_implementation", type=str, default="sdpa")
    p.add_argument("--sample_rate", type=int, default=SAMPLE_RATE)
    p.add_argument("--chunk_duration", type=float, default=0.24)

    p.add_argument("--decode_chunk_frames", type=int, default=3)
    p.add_argument("--decode_overlap_frames", type=int, default=0)

    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.6)
    p.add_argument("--top_k", type=int, default=30)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    p.add_argument("--repetition_window", type=int, default=50)
    p.add_argument("--no_sample", action="store_true")
    p.add_argument("--max_length", type=int, default=3000)
    p.add_argument("--seed", type=int, default=None)

    # 模拟 LLM streaming 参数
    p.add_argument("--delta_chunk_chars", type=int, default=1, help="Number of characters to output at each delta (1 = verbatim)")
    p.add_argument("--delta_delay_s", type=float, default=0.0, help="Simulated delay in seconds between deltas, let 0 = no delay")

    p.add_argument(
        "--assistant_text",
        type=str,
        default=(
            "This is a long piece of text used to simulate streaming output from a large language model."
            " We split it into many small text deltas and feed them into the TTS incrementally."
            " Once the TTS accumulates enough tokens, it begins prefill and starts generating audio."
            " After that, as more text arrives, it continues generating audio and outputs playable WAV chunks in real time."
        ),
    )

    args = p.parse_args()
    if args.repetition_window is not None and int(args.repetition_window) <= 0:
        args.repetition_window = None

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    device = torch.device(args.device)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path))
    processor = MossTTSRealtimeProcessor(tokenizer)

    model = MossTTSRealtime.from_pretrained(
        str(args.model_path),
        attn_implementation=args.attn_implementation,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    codec = _load_codec(args.codec_path, device)

    with torch.inference_mode():
        prompt_audio = _load_audio(Path(args.prompt_wav), target_sample_rate=args.sample_rate)
        prompt_result = codec.encode(prompt_audio.unsqueeze(0).to(device), chunk_duration=args.chunk_duration)
        prompt_tokens = _extract_codes(prompt_result).cpu().numpy().squeeze(1)

    inferencer = MossTTSRealtimeInference(model, tokenizer, max_length=args.max_length)
    inferencer.reset_generation_state(keep_cache=False)

    session = MossTTSRealtimeStreamingSession(
        inferencer,
        processor,
        codec=codec,
        codec_sample_rate=args.sample_rate,
        codec_encode_kwargs={"chunk_duration": args.chunk_duration},
        prefill_text_len=processor.delay_tokens_len,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=not args.no_sample,
        repetition_penalty=args.repetition_penalty,
        repetition_window=args.repetition_window,
    )
    session.set_voice_prompt_tokens(prompt_tokens)

    # ── Build input_ids without the user turn: system_prompt + assistant prefix ──
    system_prompt = processor.make_ensemble(prompt_tokens)
    assistant_prefix_ids = tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n")
    assistant_prefix = np.full(
        (len(assistant_prefix_ids), system_prompt.shape[1]),
        fill_value=processor.audio_channel_pad,
        dtype=np.int64,
    )
    assistant_prefix[:, 0] = assistant_prefix_ids
    input_ids = np.concatenate([system_prompt, assistant_prefix], axis=0)

    session.reset_turn(
        input_ids=input_ids,
        include_system_prompt=False,
        reset_cache=True,
    )

    decoder = AudioStreamDecoder(
        codec,
        chunk_frames=args.decode_chunk_frames,
        overlap_frames=args.decode_overlap_frames,
        decode_kwargs={"chunk_duration": -1},
        device=device,
    )

    print(f"[INFO] Simulate LLM streaming output, each time {args.delta_chunk_chars} tokens"
          f", interval {args.delta_delay_s}s")
    print(f"[INFO] assistant_text len({len(args.assistant_text)} ): "
          f"{args.assistant_text[:60]}...")

    text_deltas = fake_llm_text_stream(
        args.assistant_text,
        chunk_chars=args.delta_chunk_chars,
        delay_s=args.delta_delay_s,
    )

    wav_chunks = run_streaming_tts(
        session=session,
        codec=codec,
        decoder=decoder,
        text_deltas=text_deltas,
    )

    out_path = Path(args.out_wav).expanduser()
    write_wav(out_path, args.sample_rate, wav_chunks)
    print(f"\n[OK] Write complete: {out_path}")


if __name__ == "__main__":
    main()


#   python3 example_llm_stream_to_tts.py \
#     --model_path OpenMOSS-Team/MOSS-TTS-Realtime \
#     --codec_path OpenMOSS-Team/MOSS-Audio-Tokenizer \
#     --prompt_wav ./audio/prompt_audio1.mp3 
