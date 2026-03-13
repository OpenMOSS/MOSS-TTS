import argparse
import base64
import functools
import json
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import gradio as gr
import numpy as np

import torch
import torchaudio
import torch._dynamo
from transformers import AutoModel, AutoTokenizer
from mossttsrealtime import MossTTSRealtime, MossTTSRealtimeProcessor
from mossttsrealtime.streaming_mossttsrealtime import (
    AudioStreamDecoder,
    MossTTSRealtimeInference,
    MossTTSRealtimeStreamingSession,
)

torch._dynamo.config.cache_size_limit = 64

APP_DIR = Path(__file__).resolve().parent
AUDIO_DIR = APP_DIR / "audio"
SAMPLE_RATE = 24000

CODEC_MODEL_PATH = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
MODEL_PATH = "OpenMOSS-Team/MOSS-TTS-Realtime"
TOKENIZER_PATH = "OpenMOSS-Team/MOSS-TTS-Realtime"

PROMPT_WAV = AUDIO_DIR / "prompt_audio1.mp3"
USER_WAV = AUDIO_DIR / "user1.wav"

WARMUP_POLL_INTERVAL_SECONDS = 0.5
DEFAULT_REPETITION_WINDOW = 50
WARMUP_STEP_TOKENS = DEFAULT_REPETITION_WINDOW + 1
WARMUP_USER_TEXT = "Hello!"
WARMUP_BASE_ASSISTANT_TEXT = (
    "This startup warmup request primes the streaming text to speech path "
    "so the first real user request avoids the cold compile stall."
)


def _apply_seed(seed: int | None) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_audio(path: Path, target_sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if sr != target_sample_rate:
        wav = torchaudio.functional.resample(wav, sr, target_sample_rate)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav


def _load_codec(device: torch.device, codec_model_path: str):
    codec = AutoModel.from_pretrained(codec_model_path, trust_remote_code=True).eval()
    return codec.to(device)


def _extract_codes(encode_result):
    if isinstance(encode_result, dict):
        codes = encode_result["audio_codes"]

    elif isinstance(encode_result, (list, tuple)) and encode_result:
        codes = encode_result[0]
    else:
        codes = encode_result

    if isinstance(codes, np.ndarray):
        codes = torch.from_numpy(codes)

    if isinstance(codes, torch.Tensor) and codes.dim() == 3:
        if codes.shape[1] == 1:
            codes = codes[:, 0, :]
        elif codes.shape[0] == 1:
            codes = codes[0]
        else:
            raise ValueError(f"Unsupported 3D audio code shape: {tuple(codes.shape)}")

    return codes


@dataclass(frozen=True)
class BackendPaths:
    model_path: str
    tokenizer_path: str
    codec_model_path: str
    device_str: str
    attn_impl: str


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    repetition_window: int
    do_sample: bool
    max_length: int
    seed: int | None


@dataclass(frozen=True)
class StreamingConfig:
    text_chunk_tokens: int
    input_delay: float
    decode_chunk_frames: int
    decode_overlap_frames: int
    chunk_duration: float
    prebuffer_seconds: float
    buffer_threshold_seconds: float = 0.0


@dataclass(frozen=True)
class StreamingRequest:
    user_text: str
    assistant_text: str
    prompt_audio: str | None
    user_audio: str | None
    use_default_prompt: bool
    use_default_user: bool
    generation: GenerationConfig
    streaming: StreamingConfig
    backend: BackendPaths


@dataclass(frozen=True)
class StreamEvent:
    message: str
    audio: tuple[int, np.ndarray] | None = None


@dataclass(frozen=True)
class WarmupSnapshot:
    state: str
    progress: float
    message: str
    detail: str | None = None
    error: str | None = None

    @property
    def ready(self) -> bool:
        return self.state == "ready"

    @property
    def failed(self) -> bool:
        return self.state == "failed"


class TokenChunkStream:
    def __init__(
        self,
        tokens: Sequence[int],
        chunk_size: int,
    ):
        self._tokens = list(tokens)
        self._chunk_size = int(chunk_size)

    def __iter__(self) -> Iterator[list[int]]:
        if not self._tokens:
            return
        step = len(self._tokens) if self._chunk_size <= 0 else self._chunk_size
        for idx in range(0, len(self._tokens), step):
            yield self._tokens[idx : idx + step]


class BufferedAudioTracker:
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.start_time: float | None = None
        self.samples_emitted = 0

    def add_chunk(self, chunk: np.ndarray) -> None:
        if chunk.size == 0:
            return
        if self.start_time is None:
            self.start_time = time.monotonic()
        self.samples_emitted += int(chunk.size)

    def buffered_seconds(self) -> float:
        if self.start_time is None:
            return 0.0
        elapsed = time.monotonic() - self.start_time
        buffered = self.samples_emitted / self.sample_rate - elapsed
        return max(0.0, buffered)


class AudioFrameDecoder:
    def __init__(
        self,
        decoder: AudioStreamDecoder,
        codebook_size: int,
        audio_eos_token: int,
    ):
        self.decoder = decoder
        self.codebook_size = codebook_size
        self.audio_eos_token = audio_eos_token

    def decode_frames(self, audio_frames: list[torch.Tensor]) -> Iterator[np.ndarray]:
        for frame in audio_frames:
            tokens = frame
            if tokens.dim() == 3:
                tokens = tokens[0]
            if tokens.dim() != 2:
                raise ValueError(f"Expected [T, C] audio tokens, got {tuple(tokens.shape)}")
            tokens, stop = _sanitize_tokens(tokens, self.codebook_size, self.audio_eos_token)
            if tokens.numel() == 0:
                if stop:
                    break
                continue
            self.decoder.push_tokens(tokens.detach())
            for wav in self.decoder.audio_chunks():
                if wav.numel() == 0:
                    continue
                yield wav.detach().cpu().numpy().reshape(-1)
            if stop:
                break

    def flush(self) -> Iterator[np.ndarray]:
        final_chunk = self.decoder.flush()
        if final_chunk is not None and final_chunk.numel() > 0:
            yield final_chunk.detach().cpu().numpy().reshape(-1)


class StreamAudioEmitter:
    def __init__(self, sample_rate: int, prebuffer_seconds: float):
        self.sample_rate = sample_rate
        self._buffer_tracker = BufferedAudioTracker(sample_rate)
        self._prebuffer_target = max(0.0, float(prebuffer_seconds))
        self._prebuffering = self._prebuffer_target > 0.0
        self._pending_chunks: list[np.ndarray] = []
        self._pending_samples = 0
        self.chunk_count = 0
        self.has_audio = False

    def wait_for_capacity(self, threshold_seconds: float) -> None:
        _maybe_wait_for_buffer(self._buffer_tracker, threshold_seconds)

    def emit_many(self, chunks: Iterator[np.ndarray], message_prefix: str) -> Iterator[StreamEvent]:
        for chunk in chunks:
            yield from self.emit(chunk, message_prefix)

    def emit(self, chunk: np.ndarray, message_prefix: str) -> Iterator[StreamEvent]:
        chunk = np.asarray(chunk).reshape(-1)
        if chunk.size == 0:
            return
        if self._prebuffering:
            self._pending_chunks.append(chunk)
            self._pending_samples += int(chunk.size)
            if (self._pending_samples / self.sample_rate) < self._prebuffer_target:
                return
            self._prebuffering = False
            pending_chunks = self._pending_chunks
            self._pending_chunks = []
            self._pending_samples = 0
            for pending in pending_chunks:
                yield self._make_event(pending, message_prefix)
            return
        yield self._make_event(chunk, message_prefix)

    def flush(self, message_prefix: str) -> Iterator[StreamEvent]:
        if not self._prebuffering or not self._pending_chunks:
            self._prebuffering = False
            return
        self._prebuffering = False
        pending_chunks = self._pending_chunks
        self._pending_chunks = []
        self._pending_samples = 0
        for chunk in pending_chunks:
            yield self._make_event(chunk, message_prefix)

    def _make_event(self, chunk: np.ndarray, message_prefix: str) -> StreamEvent:
        self.chunk_count += 1
        self.has_audio = True
        self._buffer_tracker.add_chunk(chunk)
        return StreamEvent(
            message=f"{message_prefix} chunk {self.chunk_count}",
            audio=(self.sample_rate, chunk),
        )


def _maybe_wait_for_buffer(buffer_tracker: BufferedAudioTracker, threshold_seconds: float) -> None:
    if threshold_seconds <= 0:
        return
    while buffer_tracker.buffered_seconds() > threshold_seconds:
        time.sleep(0.01)


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


def _build_streaming_session(
    model: MossTTSRealtime,
    tokenizer,
    processor: MossTTSRealtimeProcessor,
    codec,
    *,
    max_length: int,
    chunk_duration: float,
    temperature: float,
    top_p: float,
    top_k: int,
    do_sample: bool,
    repetition_penalty: float,
    repetition_window: int,
) -> tuple[MossTTSRealtimeStreamingSession, MossTTSRealtimeInference]:
    inferencer = MossTTSRealtimeInference(model, tokenizer, max_length=max_length)
    inferencer.reset_generation_state(keep_cache=False)
    session = MossTTSRealtimeStreamingSession(
        inferencer,
        processor,
        codec=codec,
        codec_sample_rate=SAMPLE_RATE,
        codec_encode_kwargs={"chunk_duration": chunk_duration},
        prefill_text_len=processor.delay_tokens_len,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        repetition_window=repetition_window,
    )
    return session, inferencer


def _build_frame_decoder(
    codec,
    inferencer: MossTTSRealtimeInference,
    device: torch.device,
    *,
    chunk_frames: int,
    overlap_frames: int,
) -> AudioFrameDecoder:
    decoder = AudioStreamDecoder(
        codec,
        chunk_frames=chunk_frames,
        overlap_frames=overlap_frames,
        decode_kwargs={"chunk_duration": -1},
        device=device,
    )
    return AudioFrameDecoder(
        decoder,
        int(getattr(codec, "codebook_size", 1024)),
        int(getattr(inferencer, "audio_eos_token", 1026)),
    )


def _normalize_seed(value: float | int | None) -> int | None:
    if value is None:
        return None
    seed = int(value)
    return None if seed == 0 else seed


def _format_completion_status(
    chunk_count: int,
    sample_rate: int,
    full_audio: np.ndarray,
    started_at: float,
    first_chunk_time: float | None,
) -> str:
    elapsed = time.monotonic() - started_at
    audio_seconds = float(full_audio.size) / float(sample_rate) if full_audio.size > 0 else 0.0
    rtf = (elapsed / audio_seconds) if audio_seconds > 0 else float("inf")
    parts = [
        "Done",
        f"chunks={chunk_count}",
        f"audio={audio_seconds:.2f}s",
        f"elapsed={elapsed:.2f}s",
        f"RTF={rtf:.3f}" if np.isfinite(rtf) else "RTF=inf",
    ]
    if first_chunk_time is not None:
        parts.append(f"TTFB={(first_chunk_time - started_at) * 1000.0:.0f}ms")
    return " | ".join(parts)


@functools.lru_cache(maxsize=1)
def _load_backend(
    model_path: str,
    tokenizer_path: str,
    codec_model_path: str,
    device_str: str,
    attn_impl: str,
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the MossTTSRealtime streaming demo.")

    device = torch.device(device_str)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    processor = MossTTSRealtimeProcessor(tokenizer)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if attn_impl and attn_impl.lower() not in {"none", ""}:
        model = MossTTSRealtime.from_pretrained(model_path, attn_implementation=attn_impl, torch_dtype=dtype).to(device)
        if (
            attn_impl.lower() == "flash_attention_2"
            and hasattr(model, "language_model")
            and hasattr(model.language_model, "config")
        ):
            model.language_model.config.attn_implementation = "flash_attention_2"
    else:
        model = MossTTSRealtime.from_pretrained(model_path, torch_dtype=dtype).to(device)
    model.eval()

    codec = _load_codec(device, codec_model_path)
    return model, tokenizer, processor, codec, device


def _resolve_audio_path(audio_path: str | None, use_default: bool, default_path: str | Path) -> Path | None:
    if audio_path:
        return Path(audio_path).expanduser()
    if use_default:
        return Path(default_path).expanduser()
    return None


class StreamingTTSDemo:
    def __init__(self, audio_token_cache_size: int = 8):
        self._audio_token_cache_size = max(1, int(audio_token_cache_size))
        self._audio_token_cache: OrderedDict[tuple[str, int, float], np.ndarray] = OrderedDict()

    def get_or_load_backend(self, backend: BackendPaths):
        return _load_backend(
            backend.model_path,
            backend.tokenizer_path,
            backend.codec_model_path,
            backend.device_str,
            backend.attn_impl,
        )

    def _validate_request(self, request: StreamingRequest) -> tuple[Path | None, Path | None]:
        if not request.user_text.strip():
            raise ValueError("user_text is required.")
        if not request.assistant_text.strip():
            raise ValueError("assistant_text is required.")
        if request.streaming.text_chunk_tokens <= 0:
            raise ValueError("text_chunk_tokens must be greater than 0.")
        if request.streaming.decode_chunk_frames <= 0:
            raise ValueError("decode_chunk_frames must be greater than 0.")
        if request.streaming.chunk_duration <= 0:
            raise ValueError("chunk_duration must be greater than 0.")

        prompt_path = _resolve_audio_path(request.prompt_audio, request.use_default_prompt, PROMPT_WAV)
        user_path = _resolve_audio_path(request.user_audio, request.use_default_user, USER_WAV)

        if prompt_path is not None and not prompt_path.exists():
            raise FileNotFoundError(f"Prompt wav not found: {prompt_path}")
        if user_path is not None and not user_path.exists():
            raise FileNotFoundError(f"User wav not found: {user_path}")

        return prompt_path, user_path

    def _encode_audio_tokens(
        self,
        path: Path,
        codec,
        device: torch.device,
        chunk_duration: float,
    ) -> np.ndarray:
        resolved_path = path.expanduser().resolve()
        cache_key = (str(resolved_path), int(resolved_path.stat().st_mtime_ns), float(chunk_duration))
        cached_tokens = self._audio_token_cache.get(cache_key)
        if cached_tokens is not None:
            self._audio_token_cache.move_to_end(cache_key)
            return cached_tokens

        with torch.inference_mode():
            audio_tensor = _load_audio(resolved_path)
            waveform = audio_tensor.to(device)
            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(0)
            encode_result = codec.encode(waveform, chunk_duration=chunk_duration)

        tokens = _extract_codes(encode_result)
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.detach().cpu().numpy()
        else:
            tokens = np.asarray(tokens)

        self._audio_token_cache[cache_key] = tokens
        self._audio_token_cache.move_to_end(cache_key)
        while len(self._audio_token_cache) > self._audio_token_cache_size:
            self._audio_token_cache.popitem(last=False)

        return tokens

    @staticmethod
    def _build_text_only_turn_input(
        processor: MossTTSRealtimeProcessor,
        user_text: str,
        prompt_tokens: np.ndarray | None,
    ) -> np.ndarray:
        system_prompt = processor.make_ensemble(prompt_tokens)
        user_prompt_text = "<|im_end|>\n<|im_start|>user\n" + user_text + "<|im_end|>\n<|im_start|>assistant\n"
        user_prompt_tokens = processor.tokenizer(user_prompt_text)["input_ids"]
        user_prompt = np.full(
            shape=(len(user_prompt_tokens), processor.channels + 1),
            fill_value=processor.audio_channel_pad,
            dtype=np.int64,
        )
        user_prompt[:, 0] = np.asarray(user_prompt_tokens, dtype=np.int64)
        return np.concatenate([system_prompt, user_prompt], axis=0)

    def _prepare_session_turn(
        self,
        session: MossTTSRealtimeStreamingSession,
        processor: MossTTSRealtimeProcessor,
        user_text: str,
        prompt_tokens: np.ndarray | None,
        user_tokens: np.ndarray | None,
    ) -> str | None:
        if user_tokens is None:
            turn_input_ids = self._build_text_only_turn_input(processor, user_text, prompt_tokens)
            session.reset_turn(input_ids=turn_input_ids, include_system_prompt=True, reset_cache=True)
            return "No user audio provided, running text-only turn."

        session.reset_turn(
            user_text=user_text,
            user_audio_tokens=user_tokens,
            include_system_prompt=True,
            reset_cache=True,
        )
        return None

    def run_stream(self, request: StreamingRequest) -> Iterator[StreamEvent]:
        prompt_path, user_path = self._validate_request(request)
        model, tokenizer, processor, codec, device = self.get_or_load_backend(request.backend)
        _apply_seed(request.generation.seed)

        prompt_tokens = (
            self._encode_audio_tokens(
                prompt_path,
                codec,
                device,
                chunk_duration=request.streaming.chunk_duration,
            )
            if prompt_path is not None
            else None
        )
        user_tokens = (
            self._encode_audio_tokens(
                user_path,
                codec,
                device,
                chunk_duration=request.streaming.chunk_duration,
            )
            if user_path is not None
            else None
        )

        session, inferencer = _build_streaming_session(
            model,
            tokenizer,
            processor,
            codec,
            max_length=request.generation.max_length,
            chunk_duration=request.streaming.chunk_duration,
            temperature=request.generation.temperature,
            top_p=request.generation.top_p,
            top_k=request.generation.top_k,
            do_sample=request.generation.do_sample,
            repetition_penalty=request.generation.repetition_penalty,
            repetition_window=request.generation.repetition_window,
        )
        if prompt_tokens is not None:
            session.set_voice_prompt_tokens(prompt_tokens)
        else:
            session.clear_voice_prompt()

        turn_message = self._prepare_session_turn(
            session,
            processor,
            request.user_text,
            prompt_tokens,
            user_tokens,
        )
        if turn_message:
            yield StreamEvent(message=turn_message)

        frame_decoder = _build_frame_decoder(
            codec,
            inferencer,
            device,
            chunk_frames=request.streaming.decode_chunk_frames,
            overlap_frames=request.streaming.decode_overlap_frames,
        )

        text_tokens = tokenizer.encode(request.assistant_text, add_special_tokens=False)
        if not text_tokens:
            raise RuntimeError("Assistant text tokenization returned no tokens.")

        token_stream = TokenChunkStream(text_tokens, request.streaming.text_chunk_tokens)
        audio_emitter = StreamAudioEmitter(SAMPLE_RATE, request.streaming.prebuffer_seconds)

        with codec.streaming(batch_size=1):
            for token_chunk in token_stream:
                audio_emitter.wait_for_capacity(request.streaming.buffer_threshold_seconds)
                audio_frames = session.push_text_tokens(token_chunk)
                yield from audio_emitter.emit_many(frame_decoder.decode_frames(audio_frames), "Streaming")
                if request.streaming.input_delay > 0:
                    time.sleep(request.streaming.input_delay)

            final_frames = session.end_text()
            yield from audio_emitter.emit_many(frame_decoder.decode_frames(final_frames), "Finalizing")

            while True:
                drain_frames = session.drain(max_steps=1)
                if not drain_frames:
                    break
                yield from audio_emitter.emit_many(frame_decoder.decode_frames(drain_frames), "Finalizing")
                if session.inferencer.is_finished:
                    break

            yield from audio_emitter.emit_many(frame_decoder.flush(), "Final")
            yield from audio_emitter.flush("Final")

        if not audio_emitter.has_audio:
            raise RuntimeError("No audio waveform chunks decoded from streaming inference.")

        yield StreamEvent(message="Streaming complete.")


class WarmupManager:
    def __init__(self, tts_demo: "StreamingTTSDemo", backend: BackendPaths):
        self.tts_demo = tts_demo
        self.backend = backend
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._started = False
        self._state = "pending"
        self._progress = 0.0
        self._message = "Waiting for startup warmup."
        self._detail = "The app warms the streaming path before the first real request."
        self._error: str | None = None

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            self._thread = threading.Thread(target=self._run, name="tts-startup-warmup", daemon=True)
            self._thread.start()

    def snapshot(self) -> WarmupSnapshot:
        with self._lock:
            return WarmupSnapshot(
                state=self._state,
                progress=self._progress,
                message=self._message,
                detail=self._detail,
                error=self._error,
            )

    def _set_state(
        self,
        *,
        state: str | None = None,
        progress: float | None = None,
        message: str | None = None,
        detail: str | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            if state is not None:
                self._state = state
            if progress is not None:
                self._progress = max(0.0, min(1.0, float(progress)))
            if message is not None:
                self._message = message
            if detail is not None:
                self._detail = detail
            self._error = error

    @staticmethod
    def _consume_audio(chunks: Iterator[np.ndarray]) -> None:
        for _chunk in chunks:
            pass

    @staticmethod
    def _ensure_warmup_text(tokenizer, minimum_tokens: int) -> tuple[str, list[int]]:
        text = WARMUP_BASE_ASSISTANT_TEXT
        tokens = tokenizer.encode(text, add_special_tokens=False)
        while len(tokens) < minimum_tokens:
            text = f"{text} {WARMUP_BASE_ASSISTANT_TEXT}"
            tokens = tokenizer.encode(text, add_special_tokens=False)
        return text, tokens

    @staticmethod
    def _warmup_step_detail(step_idx: int, total_steps: int) -> str:
        if step_idx == 1:
            return "First incremental step is compiling the cold streaming path."
        if step_idx == 2:
            return "Second incremental step is warming the next steady-state path."
        if step_idx == DEFAULT_REPETITION_WINDOW:
            return "Warming the first full repetition-window step."
        if step_idx == WARMUP_STEP_TOKENS:
            return "Confirming the post-window steady-state step."
        return f"Warming token step {step_idx}/{total_steps}."

    def _run(self) -> None:
        try:
            self._set_state(
                state="running",
                progress=0.02,
                message="Starting startup warmup.",
                detail="Preparing backend state for the first real request.",
                error=None,
            )

            self._set_state(
                progress=0.08,
                message="Loading backend.",
                detail="Model, tokenizer, codec, and CUDA runtime are warming up.",
                error=None,
            )
            model, tokenizer, processor, codec, device = self.tts_demo.get_or_load_backend(self.backend)

            self._set_state(
                progress=0.32,
                message="Preparing streaming session.",
                detail="Building a text-only warmup turn and its decoder.",
                error=None,
            )
            session, inferencer = _build_streaming_session(
                model,
                tokenizer,
                processor,
                codec,
                max_length=256,
                chunk_duration=0.24,
                temperature=0.8,
                top_p=0.6,
                top_k=30,
                do_sample=True,
                repetition_penalty=1.1,
                repetition_window=DEFAULT_REPETITION_WINDOW,
            )
            session.clear_voice_prompt()
            session.reset_turn(
                input_ids=self.tts_demo._build_text_only_turn_input(processor, WARMUP_USER_TEXT, None),
                include_system_prompt=True,
                reset_cache=True,
            )

            frame_decoder = _build_frame_decoder(
                codec,
                inferencer,
                device,
                chunk_frames=WARMUP_STEP_TOKENS,
                overlap_frames=0,
            )

            _, warmup_tokens = self._ensure_warmup_text(
                tokenizer,
                processor.delay_tokens_len + WARMUP_STEP_TOKENS,
            )

            with codec.streaming(batch_size=1):
                self._set_state(
                    progress=0.45,
                    message="Running prefill.",
                    detail="Building the first KV cache and warming the backbone path.",
                    error=None,
                )
                prefill_frames = session.push_text_tokens(warmup_tokens[: processor.delay_tokens_len])
                self._consume_audio(frame_decoder.decode_frames(prefill_frames))

                step_tokens = warmup_tokens[
                    processor.delay_tokens_len : processor.delay_tokens_len + WARMUP_STEP_TOKENS
                ]
                total_steps = max(1, len(step_tokens))
                for idx, token in enumerate(step_tokens, start=1):
                    self._set_state(
                        progress=0.55 + 0.25 * (idx - 1) / total_steps,
                        message="Compiling first streaming steps.",
                        detail=self._warmup_step_detail(idx, total_steps),
                        error=None,
                    )
                    step_frames = session.push_text_tokens([token])
                    self._consume_audio(frame_decoder.decode_frames(step_frames))

                self._set_state(
                    progress=0.86,
                    message="Warming finalization path.",
                    detail="Priming end-text, drain, and decoder flush before user traffic.",
                    error=None,
                )
                final_frames = session.end_text()
                self._consume_audio(frame_decoder.decode_frames(final_frames))
                drain_frames = session.drain(max_steps=1)
                self._consume_audio(frame_decoder.decode_frames(drain_frames))
                self._consume_audio(frame_decoder.flush())

            self._set_state(
                state="ready",
                progress=1.0,
                message="Warmup complete.",
                detail="The first real request should avoid the cold-start stall.",
                error=None,
            )
        except Exception as exc:
            self._set_state(
                state="failed",
                progress=1.0,
                message="Warmup failed.",
                detail="The app did not finish startup warmup.",
                error=str(exc),
            )
            print(f"[MossTTSRealtime][warmup-error] {exc}", file=sys.stderr, flush=True)


def _warmup_button_update(snapshot: WarmupSnapshot):
    if snapshot.ready:
        return gr.update(value="Generate", interactive=True)
    if snapshot.failed:
        return gr.update(value="Warmup Failed", interactive=False)
    return gr.update(value="Warming Up...", interactive=False)


def _warmup_gate_message(snapshot: WarmupSnapshot) -> str:
    progress_pct = int(round(max(0.0, min(1.0, snapshot.progress)) * 100.0))
    if snapshot.failed:
        return f"Warmup failed: {snapshot.error or snapshot.message}"
    return f"Warmup in progress ({progress_pct}%): {snapshot.message}"


def _status_from_snapshot(snapshot: WarmupSnapshot) -> str:
    return "Ready." if snapshot.ready else _warmup_gate_message(snapshot)


def _warmup_status_update(snapshot: WarmupSnapshot):
    return gr.update(value=_status_from_snapshot(snapshot))


def _warmup_timer_update(snapshot: WarmupSnapshot):
    return gr.update(active=not (snapshot.ready or snapshot.failed))


def _encode_chunk(sr: int, chunk: np.ndarray, idx: int) -> str:
    if chunk.dtype != np.float32:
        chunk = chunk.astype(np.float32)
    if chunk.ndim != 1:
        chunk = chunk.reshape(-1)
    payload = {
        "sr": int(sr),
        "idx": int(idx),
        "data": base64.b64encode(chunk.tobytes()).decode("ascii"),
    }
    return json.dumps(payload)


def _build_request(
    args: argparse.Namespace,
    *,
    user_text: str | None,
    assistant_text: str | None,
    prompt_audio: str | None,
    user_audio: str | None,
    use_default_prompt: bool,
    use_default_user: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    repetition_window: int,
    do_sample: bool,
    max_length: int,
    seed: float | int | None,
    text_chunk_tokens: int,
    input_delay: float,
    decode_chunk_frames: int,
    decode_overlap_frames: int,
    chunk_duration: float,
    prebuffer_seconds: float,
) -> StreamingRequest:
    return StreamingRequest(
        user_text=str(user_text or "Hello!"),
        assistant_text=str(assistant_text or ""),
        prompt_audio=prompt_audio,
        user_audio=user_audio,
        use_default_prompt=use_default_prompt,
        use_default_user=use_default_user,
        generation=GenerationConfig(
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            repetition_penalty=float(repetition_penalty),
            repetition_window=int(repetition_window),
            do_sample=bool(do_sample),
            max_length=int(max_length),
            seed=_normalize_seed(seed),
        ),
        streaming=StreamingConfig(
            text_chunk_tokens=int(text_chunk_tokens),
            input_delay=float(input_delay),
            decode_chunk_frames=int(decode_chunk_frames),
            decode_overlap_frames=int(decode_overlap_frames),
            chunk_duration=float(chunk_duration),
            prebuffer_seconds=float(prebuffer_seconds),
        ),
        backend=BackendPaths(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            codec_model_path=args.codec_model_path,
            device_str=args.device,
            attn_impl=args.attn_implementation,
        ),
    )


STREAM_PLAYER_HTML = """
<style>
#pcm_stream {
  position: absolute !important;
  left: -9999px !important;
  width: 1px !important;
  height: 1px !important;
  opacity: 0 !important;
  pointer-events: none !important;
}
#pcm_stream textarea, #pcm_stream input {
  width: 1px !important;
  height: 1px !important;
  opacity: 0 !important;
}
</style>
"""

STREAM_PLAYER_JS = r"""
const elemId = "pcm_stream";
if (window.__pcm_streaming_inited__) {
  return;
}
window.__pcm_streaming_inited__ = true;

let audioCtx = null;
let nextTime = 0;
let lastIdx = -1;
let lastValue = "";
let boundField = null;
let usingSetterHook = false;
const FADE_MS = 6;
const MIN_BUFFER_SEC = 0.25;

function initAudio(sr) {
  if (audioCtx && audioCtx.sampleRate !== sr) {
    audioCtx.close();
    audioCtx = null;
  }
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: sr });
    nextTime = audioCtx.currentTime;
  }
  if (audioCtx.state === "suspended") {
    audioCtx.resume();
  }
}

function decodeBase64ToFloat32(base64) {
  const binary = atob(base64);
  const len = binary.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Float32Array(bytes.buffer);
}

function playChunk(samples, sr, idx) {
  initAudio(sr);
  const buffer = audioCtx.createBuffer(1, samples.length, sr);
  buffer.copyToChannel(samples, 0);
  const source = audioCtx.createBufferSource();
  source.buffer = buffer;
  const gain = audioCtx.createGain();
  source.connect(gain);
  gain.connect(audioCtx.destination);
  const now = audioCtx.currentTime;
  if (nextTime < now + MIN_BUFFER_SEC) {
    nextTime = now + MIN_BUFFER_SEC;
  }
  const startTime = Math.max(now, nextTime);
  const endTime = startTime + buffer.duration;
  const fade = Math.min(FADE_MS / 1000.0, buffer.duration / 4);
  gain.gain.setValueAtTime(0.0, startTime);
  gain.gain.linearRampToValueAtTime(1.0, startTime + fade);
  gain.gain.setValueAtTime(1.0, Math.max(startTime + fade, endTime - fade));
  gain.gain.linearRampToValueAtTime(0.0, endTime);
  source.start(startTime);
  nextTime = endTime;
}

function handlePayload(text) {
  if (!text) return;
  let payload;
  try {
    payload = JSON.parse(text);
  } catch (e) {
    return;
  }
  if (Array.isArray(payload)) {
    for (const item of payload) {
      handlePayloadObject(item);
    }
    return;
  }
  handlePayloadObject(payload);
}

function handlePayloadObject(payload) {
  if (!payload) return;
  if (payload.reset) {
    lastIdx = -1;
    lastValue = "";
    if (audioCtx) {
      audioCtx.close();
      audioCtx = null;
    }
    return;
  }
  const idx = payload.idx ?? 0;
  if (idx <= lastIdx) return;
  lastIdx = idx;
  const sr = payload.sr || 24000;
  const samples = decodeBase64ToFloat32(payload.data);
  playChunk(samples, sr, idx);
}

function hookField(field) {
  if (!field || field === boundField) return;
  boundField = field;
  const proto = field.tagName === "TEXTAREA" ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
  const desc = Object.getOwnPropertyDescriptor(proto, "value");
  if (!desc || !desc.get || !desc.set) {
    usingSetterHook = false;
    return;
  }
  usingSetterHook = true;
  const nativeGet = desc.get;
  const nativeSet = desc.set;
  Object.defineProperty(field, "value", {
    configurable: true,
    get() {
      return nativeGet.call(field);
    },
    set(v) {
      nativeSet.call(field, v);
      if (v && v !== lastValue) {
        lastValue = v;
        handlePayload(v);
      }
    },
  });

  const initial = field.value;
  if (initial && initial !== lastValue) {
    lastValue = initial;
    handlePayload(initial);
  }
}

function pollField() {
  const field = document.querySelector(`#${elemId} textarea, #${elemId} input`);
  if (!field) {
    boundField = null;
    usingSetterHook = false;
    setTimeout(pollField, 300);
    return;
  }
  if (field !== boundField) {
    hookField(field);
  }
  setTimeout(pollField, 300);
}

function pollValue() {
  if (usingSetterHook) {
    setTimeout(pollValue, 500);
    return;
  }
  const field = document.querySelector(`#${elemId} textarea, #${elemId} input`);
  if (!field) {
    setTimeout(pollValue, 300);
    return;
  }
  const value = field.value;
  if (value && value !== lastValue) {
    lastValue = value;
    handlePayload(value);
  }
  setTimeout(pollValue, 40);
}

function tryUnlockAudio() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  if (audioCtx.state === "suspended") {
    audioCtx.resume();
  }
}

document.addEventListener("click", (event) => {
  const btn = event.target.closest("#tts_generate");
  if (btn) {
    tryUnlockAudio();
  }
});

pollField();
pollValue();
"""


def _build_demo(
    args: argparse.Namespace,
    tts_demo: StreamingTTSDemo,
    warmup_manager: WarmupManager,
):
    initial_warmup_snapshot = warmup_manager.snapshot()
    with gr.Blocks(title="MossTTSRealtime") as demo:
        gr.Markdown("MossTTSRealtime demo")
        gr.Markdown("Note: The first run may take a while to load the model.")
        gr.HTML(STREAM_PLAYER_HTML, js_on_load=STREAM_PLAYER_JS)

        with gr.Row():
            with gr.Column():
                prompt_audio = gr.Audio(label="Prompt WAV (optional)", type="filepath")
                assistant_text = gr.Textbox(label="Assistant Text", lines=3)
                user_audio = gr.Audio(label="User WAV (optional)", type="filepath")
                user_text = gr.Textbox(label="User Text (optional)", lines=3)
                use_default_prompt = gr.Checkbox(label="Use Default Prompt WAV (fallback)", value=False)
                use_default_user = gr.Checkbox(label="Use Default User WAV (fallback)", value=False)

                with gr.Accordion("Generation Options", open=False):
                    temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="Temperature")
                    top_p = gr.Slider(0.1, 1.0, value=0.6, step=0.05, label="Top P")
                    top_k = gr.Slider(1, 100, value=30, step=1, label="Top K")
                    repetition_penalty = gr.Slider(1.0, 2.0, value=1.1, step=0.05, label="Repetition Penalty")
                    repetition_window = gr.Slider(
                        1, 200, value=DEFAULT_REPETITION_WINDOW, step=1, label="Repetition Window"
                    )
                    do_sample = gr.Checkbox(label="Do Sample", value=True)
                    max_length = gr.Slider(100, 10000, value=2000, step=10, label="Max Length")
                    seed = gr.Number(value=0, precision=0, label="Seed (0 for random)")

                with gr.Accordion("Streaming Options", open=False):
                    stream_text_chunk_tokens = gr.Slider(1, 64, value=12, step=1, label="Text Chunk Tokens")
                    stream_input_delay = gr.Slider(0.0, 0.5, value=0.0, step=0.05, label="Input Delay (s)")
                    stream_decode_chunk_frames = gr.Slider(1, 20, value=6, step=1, label="Decode Chunk Frames")
                    stream_decode_overlap_frames = gr.Slider(0, 10, value=0, step=1, label="Decode Overlap Frames")
                    chunk_duration = gr.Slider(0.08, 4.0, value=0.96, step=0.08, label="Codec Chunk Duration (s)")
                    stream_prebuffer_seconds = gr.Slider(0.0, 20.0, value=0.0, step=0.05, label="Initial Buffer (s)")

                run_btn = gr.Button(
                    "Generate" if initial_warmup_snapshot.ready else "Warming Up...",
                    elem_id="tts_generate",
                    interactive=initial_warmup_snapshot.ready,
                )

            with gr.Column():
                stream_data = gr.Textbox(label="PCM Stream (JSON)", elem_id="pcm_stream", interactive=False, lines=6)
                output_audio = gr.Audio(label="Final Audio", type="numpy")
                initial_status = _status_from_snapshot(initial_warmup_snapshot)
                status = gr.Textbox(label="Status", lines=3, value=initial_status)

        warmup_timer = gr.Timer(value=WARMUP_POLL_INTERVAL_SECONDS, active=True)

        def _poll_warmup_state():
            snapshot = warmup_manager.snapshot()
            return (
                _warmup_button_update(snapshot),
                _warmup_status_update(snapshot),
                _warmup_timer_update(snapshot),
            )

        def _on_generate(
            user_text_value,
            assistant_text_value,
            prompt_audio_value,
            user_audio_value,
            use_default_prompt_value,
            use_default_user_value,
            temperature_value,
            top_p_value,
            top_k_value,
            repetition_penalty_value,
            repetition_window_value,
            do_sample_value,
            max_length_value,
            seed_value,
            stream_text_chunk_tokens_value,
            stream_input_delay_value,
            stream_decode_chunk_frames_value,
            stream_decode_overlap_frames_value,
            chunk_duration_value,
            stream_prebuffer_seconds_value,
        ):
            warmup_snapshot = warmup_manager.snapshot()
            if not warmup_snapshot.ready:
                yield json.dumps({"reset": True}), gr.update(value=None), _warmup_gate_message(warmup_snapshot)
                return
            try:
                started_at = time.monotonic()
                full_chunks: list[np.ndarray] = []
                first_chunk_time: float | None = None
                sample_rate = SAMPLE_RATE
                yield json.dumps({"reset": True}), gr.update(value=None), "Started"

                request = _build_request(
                    args,
                    user_text=user_text_value,
                    assistant_text=assistant_text_value,
                    prompt_audio=prompt_audio_value,
                    user_audio=user_audio_value,
                    use_default_prompt=bool(use_default_prompt_value),
                    use_default_user=bool(use_default_user_value),
                    temperature=float(temperature_value),
                    top_p=float(top_p_value),
                    top_k=int(top_k_value),
                    repetition_penalty=float(repetition_penalty_value),
                    repetition_window=int(repetition_window_value),
                    do_sample=bool(do_sample_value),
                    max_length=int(max_length_value),
                    seed=seed_value,
                    text_chunk_tokens=int(stream_text_chunk_tokens_value),
                    input_delay=float(stream_input_delay_value),
                    decode_chunk_frames=int(stream_decode_chunk_frames_value),
                    decode_overlap_frames=int(stream_decode_overlap_frames_value),
                    chunk_duration=float(chunk_duration_value),
                    prebuffer_seconds=float(stream_prebuffer_seconds_value),
                )

                for event in tts_demo.run_stream(request):
                    if event.audio is None:
                        yield gr.update(), gr.update(), event.message
                        continue

                    sr, chunk = event.audio
                    chunk = np.asarray(chunk).reshape(-1)
                    if chunk.size == 0:
                        continue
                    full_chunks.append(chunk)
                    sample_rate = sr
                    idx = len(full_chunks)
                    if first_chunk_time is None:
                        first_chunk_time = time.monotonic()
                    payload = _encode_chunk(sr, chunk, idx)
                    ttfb_ms = (first_chunk_time - started_at) * 1000.0 if first_chunk_time is not None else float("nan")
                    status_msg = f"{event.message} | chunks={idx} | ttfb={ttfb_ms:.0f}ms"
                    yield payload, gr.update(), status_msg

                if full_chunks:
                    full_audio = np.concatenate(full_chunks)
                    done_msg = _format_completion_status(
                        len(full_chunks),
                        sample_rate,
                        full_audio,
                        started_at,
                        first_chunk_time,
                    )
                    yield gr.update(), (sample_rate, full_audio), done_msg
                else:
                    yield gr.update(), gr.update(), "Done | no audio chunks emitted"
            except Exception as exc:
                import traceback
                traceback.print_exc()
                yield gr.update(), gr.update(), f"Error: {exc}"

        run_btn.click(
            _on_generate,
            inputs=[
                user_text,
                assistant_text,
                prompt_audio,
                user_audio,
                use_default_prompt,
                use_default_user,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                repetition_window,
                do_sample,
                max_length,
                seed,
                stream_text_chunk_tokens,
                stream_input_delay,
                stream_decode_chunk_frames,
                stream_decode_overlap_frames,
                chunk_duration,
                stream_prebuffer_seconds,
            ],
            outputs=[stream_data, output_audio, status],
        )
        demo.load(
            _poll_warmup_state,
            outputs=[run_btn, status, warmup_timer],
            queue=False,
            show_progress="hidden",
        )
        warmup_timer.tick(
            _poll_warmup_state,
            outputs=[run_btn, status, warmup_timer],
            queue=False,
            show_progress="hidden",
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="MossTTSRealtime streaming TTS Gradio demo")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--tokenizer_path", type=str, default=TOKENIZER_PATH)
    parser.add_argument("--codec_model_path", type=str, default=CODEC_MODEL_PATH)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa", # It is recommended to use sdpa, which will also use torch.compile
        choices=["sdpa", "flash_attention_2", "eager", "none"],
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18082)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    tts_demo = StreamingTTSDemo()
    warmup_manager = WarmupManager(
        tts_demo,
        BackendPaths(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            codec_model_path=args.codec_model_path,
            device_str=args.device,
            attn_impl=args.attn_implementation,
        ),
    )
    warmup_manager.start()
    demo = _build_demo(args, tts_demo, warmup_manager)
    demo.queue(max_size=10, default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
