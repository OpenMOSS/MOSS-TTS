import argparse
import contextlib
import functools
import hashlib
import os
import queue
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import torch
import torchaudio
from transformers import AutoModel, AutoTokenizer

from mossttsrealtime import MossTTSRealtime, MossTTSRealtimeProcessor
from mossttsrealtime.streaming_mossttsrealtime import (
    AudioStreamDecoder,
    MossTTSRealtimeInference,
    MossTTSRealtimeStreamingSession,
)

try:
    import soxr
except Exception:
    soxr = None


SAMPLE_RATE = 24000
DEFAULT_PROMPT_WAV = "./audio/prompt_audio1.mp3"
DEFAULT_USER_WAV = ""
DEFAULT_AUDIO_PROMPTS_DIR = "./audio"
TARGET_SR = int(os.getenv("MOSS_TTS_TARGET_SR", "24000"))
MODEL_PATH = os.getenv("MOSS_TTS_MODEL_PATH", "OpenMOSS-Team/MOSS-TTS-Realtime")
TOKENIZER_PATH = os.getenv("MOSS_TTS_TOKENIZER_PATH", "OpenMOSS-Team/MOSS-TTS-Realtime")
CODEC_MODEL_PATH = os.getenv("MOSS_TTS_CODEC_MODEL_PATH", "OpenMOSS-Team/MOSS-Audio-Tokenizer")

DEVICE = os.getenv("MOSS_TTS_DEVICE", "cuda:0")
ATTN_IMPL = os.getenv("MOSS_TTS_ATTN_IMPL", "sdpa")
RESAMPLED_AUDIO_DIR = Path(
    os.getenv("MOSS_TTS_RESAMPLED_AUDIO_DIR", "./tmp")
).expanduser()
AUDIO_PROMPTS_DIR = Path(
    os.getenv("MOSS_TTS_AUDIO_PROMPTS_DIR", DEFAULT_AUDIO_PROMPTS_DIR)
).expanduser()


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
    repetition_window: Optional[int]
    do_sample: bool
    max_length: int
    seed: Optional[int]


@dataclass(frozen=True)
class StreamingConfig:
    decode_chunk_frames: int
    decode_overlap_frames: int
    chunk_duration: float
    buffer_threshold_seconds: float = 0.0


@dataclass(frozen=True)
class StreamingCallbacks:
    on_audio_stream_start: Optional[Callable[[], None]] = None
    on_audio_stream_stop: Optional[Callable[[], None]] = None


class BufferedAudioTracker:
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.start_time: Optional[float] = None
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
        callbacks: Optional[StreamingCallbacks] = None,
    ):
        self.decoder = decoder
        self.codebook_size = codebook_size
        self.audio_eos_token = audio_eos_token
        self.callbacks = callbacks or StreamingCallbacks()
        self._started = False
        self._finished = False

    def _mark_started(self) -> None:
        if self._started:
            return
        self._started = True
        if self.callbacks.on_audio_stream_start:
            self.callbacks.on_audio_stream_start()

    def finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        if self._started and self.callbacks.on_audio_stream_stop:
            self.callbacks.on_audio_stream_stop()

    def decode_frames(self, audio_frames: list[torch.Tensor]) -> Iterator[np.ndarray]:
        for frame in audio_frames:
            tokens = frame
            if tokens.dim() == 3:
                tokens = tokens[0]
            if tokens.dim() != 2:
                raise ValueError(f"Expected [T, C] audio tokens, got {tuple(tokens.shape)}")
            tokens, _ = _sanitize_tokens(tokens, self.codebook_size, self.audio_eos_token)
            if tokens.numel() == 0:
                continue
            self.decoder.push_tokens(tokens.detach())
            for wav in self.decoder.audio_chunks():
                if wav.numel() == 0:
                    continue
                self._mark_started()
                yield wav.detach().cpu().numpy().reshape(-1)

    def flush(self) -> Iterator[np.ndarray]:
        final_chunk = self.decoder.flush()
        if final_chunk is not None and final_chunk.numel() > 0:
            self._mark_started()
            yield final_chunk.detach().cpu().numpy().reshape(-1)
        self.finish()


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


def _load_audio(path: Path, target_sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if sr != target_sample_rate:
        wav = torchaudio.functional.resample(wav, sr, target_sample_rate)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav


def _resampled_audio_cache_path(src: Path, target_sample_rate: int) -> Path:
    src = src.expanduser().resolve()
    mtime_ns = int(src.stat().st_mtime_ns)
    key = f"{src}:{mtime_ns}:{int(target_sample_rate)}".encode("utf-8", errors="ignore")
    digest = hashlib.sha1(key).hexdigest()[:12]
    return RESAMPLED_AUDIO_DIR / f"{src.stem}.sr{int(target_sample_rate)}.{digest}.wav"


def _ensure_audio_resampled_file(src: Path, target_sample_rate: int = SAMPLE_RATE) -> Path:
    src = src.expanduser().resolve()
    try:
        info = torchaudio.info(str(src))
        src_sr = int(info.sample_rate)
    except Exception:
        src_sr = -1

    src_is_wav = src.suffix.lower() == ".wav"
    if src_is_wav and src_sr == int(target_sample_rate):
        return src

    out = _resampled_audio_cache_path(src, int(target_sample_rate))
    if out.exists():
        return out

    RESAMPLED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    wav, sr = torchaudio.load(src)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if int(sr) != int(target_sample_rate):
        wav = torchaudio.functional.resample(wav, int(sr), int(target_sample_rate))
    wav = wav.detach().cpu().to(torch.float32)
    torchaudio.save(str(out), wav, sample_rate=int(target_sample_rate))
    return out


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


@functools.lru_cache(maxsize=1)
def _load_backend(
    model_path: str,
    tokenizer_path: str,
    codec_model_path: str,
    device_str: str,
    attn_impl: str,
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for MossTTSRealtime streaming inference.")

    device = torch.device(device_str)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    processor = MossTTSRealtimeProcessor(tokenizer)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if attn_impl and attn_impl.lower() not in {"none", ""}:
        model = MossTTSRealtime.from_pretrained(
            model_path,
            attn_implementation=attn_impl,
            torch_dtype=dtype,
        ).to(device)
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


def _resolve_audio_path(
    audio_path: Optional[str],
    use_default: bool,
    default_path: str,
) -> Optional[Path]:
    if audio_path:
        p = Path(audio_path).expanduser()
        if not p.is_absolute() and not p.exists():
            candidate = (AUDIO_PROMPTS_DIR / p).expanduser()
            if candidate.exists():
                return candidate
        return p
    if use_default and default_path.strip():
        return Path(default_path).expanduser()
    return None


class StreamingTTSDemo:
    def __init__(self, audio_token_cache_size: int = 8):
        self._audio_token_cache_size = max(1, int(audio_token_cache_size))
        self._audio_token_cache: OrderedDict[
            tuple[str, int, float], np.ndarray
        ] = OrderedDict()

    def get_or_load_backend(self, backend: BackendPaths):
        return _load_backend(
            backend.model_path,
            backend.tokenizer_path,
            backend.codec_model_path,
            backend.device_str,
            backend.attn_impl,
        )

    def _validate_paths(
        self,
        prompt_audio: Optional[str],
        user_audio: Optional[str],
        use_default_prompt: bool = False,
        use_default_user: bool = False,
    ) -> tuple[Optional[Path], Optional[Path]]:
        prompt_path = _resolve_audio_path(prompt_audio, use_default_prompt, DEFAULT_PROMPT_WAV)
        user_path = _resolve_audio_path(user_audio, use_default_user, DEFAULT_USER_WAV)

        if prompt_path is not None and not prompt_path.exists():
            raise FileNotFoundError(f"Prompt audio not found: {prompt_path}")
        if user_path is not None and not user_path.exists():
            raise FileNotFoundError(f"User audio not found: {user_path}")

        return prompt_path, user_path

    def _encode_audio_tokens(
        self,
        path: Path,
        codec,
        device: torch.device,
        chunk_duration: float,
        *,
        ensure_prompt_sample_rate: bool = False,
    ) -> np.ndarray:
        resolved_path = path.expanduser().resolve()
        if ensure_prompt_sample_rate:
            resolved_path = _ensure_audio_resampled_file(
                resolved_path,
                target_sample_rate=SAMPLE_RATE,
            )

        cache_key = (
            str(resolved_path),
            int(resolved_path.stat().st_mtime_ns),
            float(chunk_duration),
        )
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


demo = StreamingTTSDemo()


@contextlib.asynccontextmanager
async def _lifespan(application: FastAPI):
    backend = BackendPaths(
        model_path=MODEL_PATH,
        tokenizer_path=TOKENIZER_PATH,
        codec_model_path=CODEC_MODEL_PATH,
        device_str=DEVICE,
        attn_impl=ATTN_IMPL,
    )
    print("[warmup] Loading backend ...", flush=True)
    try:
        demo.get_or_load_backend(backend)
        print("[warmup] Backend loaded.", flush=True)
    except Exception as e:
        print(f"[warmup] Failed: {e}", flush=True)
    yield


app = FastAPI(title="MOSS TTS Audio Streaming", lifespan=_lifespan)


def float32_to_pcm16_bytes(x: np.ndarray) -> bytes:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    i16 = np.clip(x * 32768.0, -32768, 32767).astype(np.int16)
    return i16.tobytes()


def resample_if_needed(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return x
    if soxr is None:
        src_idx = np.arange(x.shape[0], dtype=np.float32)
        dst_len = max(1, int(round(x.shape[0] * dst_sr / src_sr)))
        dst_idx = np.linspace(0, max(0, x.shape[0] - 1), dst_len, dtype=np.float32)
        return np.interp(dst_idx, src_idx, x).astype(np.float32)
    return soxr.resample(x, src_sr, dst_sr).astype(np.float32)


@dataclass
class SessionRuntime:
    session_id: str
    backend: BackendPaths
    generation: GenerationConfig
    streaming: StreamingConfig

    lock: threading.RLock = field(default_factory=threading.RLock)
    command_queue: "queue.Queue[dict]" = field(default_factory=queue.Queue)
    audio_queue: "queue.Queue[Optional[bytes]]" = field(default_factory=queue.Queue)
    worker_thread: Optional[threading.Thread] = None

    turn_active: bool = False
    turn_finished: bool = False
    last_access_time: float = field(default_factory=time.time)
    pending_audio_queue: Optional["queue.Queue[Optional[bytes]]"] = None

    model: object = None
    tokenizer: object = None
    processor: object = None
    codec: object = None
    device: object = None

    inferencer: object = None
    tts_session: object = None
    decoder: object = None
    frame_decoder: object = None
    buffer_tracker: object = None
    codec_stream_cm: object = None

    prompt_tokens: Optional[np.ndarray] = None
    user_tokens: Optional[np.ndarray] = None

    def touch(self):
        self.last_access_time = time.time()


class SessionManager:
    def __init__(self):
        self._sessions: dict[str, SessionRuntime] = {}
        self._lock = threading.RLock()

    def _make_backend(self) -> BackendPaths:
        return BackendPaths(
            model_path=MODEL_PATH,
            tokenizer_path=TOKENIZER_PATH,
            codec_model_path=CODEC_MODEL_PATH,
            device_str=DEVICE,
            attn_impl=ATTN_IMPL,
        )

    def _make_generation(self) -> GenerationConfig:
        return GenerationConfig(
            temperature=0.8,
            top_p=0.6,
            top_k=30,
            repetition_penalty=1.1,
            repetition_window=50,
            do_sample=True,
            max_length=10000,
            seed=None,
        )

    def _make_streaming(self) -> StreamingConfig:
        return StreamingConfig(
            decode_chunk_frames=6,
            decode_overlap_frames=0,
            chunk_duration=0.96,
            buffer_threshold_seconds=0.0,
        )

    def get_or_create(self, session_id: str) -> SessionRuntime:
        with self._lock:
            if session_id not in self._sessions:
                sess = SessionRuntime(
                    session_id=session_id,
                    backend=self._make_backend(),
                    generation=self._make_generation(),
                    streaming=self._make_streaming(),
                )
                self._sessions[session_id] = sess
                self._start_worker(sess)
            sess = self._sessions[session_id]
            sess.touch()
            return sess

    def get(self, session_id: str) -> Optional[SessionRuntime]:
        with self._lock:
            sess = self._sessions.get(session_id)
            if sess is not None:
                sess.touch()
            return sess

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def _start_worker(self, sess: SessionRuntime) -> None:
        t = threading.Thread(
            target=self._session_worker,
            args=(sess,),
            name=f"tts-session-{sess.session_id}",
            daemon=True,
        )
        sess.worker_thread = t
        t.start()

    def _session_worker(self, sess: SessionRuntime) -> None:
        try:
            model, tokenizer, processor, codec, device = demo.get_or_load_backend(sess.backend)
            sess.model = model
            sess.tokenizer = tokenizer
            sess.processor = processor
            sess.codec = codec
            sess.device = device
        except Exception as e:
            print(f"[session_worker_error] backend load failed: {type(e).__name__}: {e}", flush=True)
            self._safe_put_audio_end(sess)
            return

        while True:
            cmd = sess.command_queue.get()
            cmd_type = cmd.get("type")

            if cmd_type == "shutdown":
                self._safe_put_audio_end(sess)
                break

            try:
                if cmd_type == "start_turn":
                    self._handle_start_turn(sess, cmd)
                elif cmd_type == "push_text":
                    self._handle_push_text(sess, cmd)
                elif cmd_type == "finish_turn":
                    self._handle_finish_turn(sess)
            except Exception as e:
                print(f"[session_worker_error] cmd={cmd_type}: {type(e).__name__}: {e}", flush=True)
                self._safe_put_audio_end(sess)
                sess.turn_active = False

    def _safe_put_audio_end(self, sess: SessionRuntime) -> None:
        try:
            sess.audio_queue.put_nowait(None)
        except Exception:
            pass
        pending_q = sess.pending_audio_queue
        if pending_q is not None and pending_q is not sess.audio_queue:
            try:
                pending_q.put_nowait(None)
            except Exception:
                pass

    def _reset_audio_queue_for_new_turn(self, sess: SessionRuntime) -> None:
        # Deprecated: draining the shared audio_queue can drop the `None` sentinel
        # used to end an existing /audio stream, causing the client to hang.
        # We now swap in a fresh queue per turn and keep the old one for any
        # in-flight audio stream generator.
        pass

    def _emit_audio_chunks(
        self,
        sess: SessionRuntime,
        chunks: Iterator[np.ndarray],
    ) -> None:
        for chunk in chunks:
            if chunk.size == 0:
                continue
            _maybe_wait_for_buffer(
                sess.buffer_tracker,
                sess.streaming.buffer_threshold_seconds,
            )
            sess.buffer_tracker.add_chunk(chunk)
            x = np.asarray(chunk, dtype=np.float32).reshape(-1)
            x = resample_if_needed(x, SAMPLE_RATE, TARGET_SR)
            pcm = float32_to_pcm16_bytes(x)
            sess.audio_queue.put(pcm)

    def _handle_start_turn(self, sess: SessionRuntime, cmd: dict) -> None:
        with sess.lock:
            turn_audio_queue = cmd.get("audio_queue")
            if turn_audio_queue is None:
                turn_audio_queue = queue.Queue()
            sess.audio_queue = turn_audio_queue
            sess.pending_audio_queue = None

            user_text: str = (cmd.get("user_text") or "").strip()
            assistant_text: str = cmd.get("assistant_text") or ""
            prompt_audio: Optional[str] = cmd.get("prompt_audio")
            user_audio: Optional[str] = cmd.get("user_audio")

            prompt_path, user_path = demo._validate_paths(
                prompt_audio=prompt_audio,
                user_audio=user_audio,
                use_default_prompt=False,
                use_default_user=False,
            )

            sess.prompt_tokens = None
            sess.user_tokens = None

            if prompt_path is not None:
                sess.prompt_tokens = demo._encode_audio_tokens(
                    prompt_path,
                    sess.codec,
                    sess.device,
                    chunk_duration=sess.streaming.chunk_duration,
                    ensure_prompt_sample_rate=True,
                )

            if user_path is not None:
                sess.user_tokens = demo._encode_audio_tokens(
                    user_path,
                    sess.codec,
                    sess.device,
                    chunk_duration=sess.streaming.chunk_duration,
                    ensure_prompt_sample_rate=False,
                )

            if sess.generation.seed is not None:
                torch.manual_seed(sess.generation.seed)
                torch.cuda.manual_seed_all(sess.generation.seed)

            inferencer = MossTTSRealtimeInference(
                sess.model,
                sess.tokenizer,
                max_length=sess.generation.max_length,
            )
            inferencer.reset_generation_state(keep_cache=False)

            tts_session = MossTTSRealtimeStreamingSession(
                inferencer,
                sess.processor,
                codec=sess.codec,
                codec_sample_rate=SAMPLE_RATE,
                codec_encode_kwargs={"chunk_duration": sess.streaming.chunk_duration},
                prefill_text_len=sess.processor.delay_tokens_len,
                temperature=sess.generation.temperature,
                top_p=sess.generation.top_p,
                top_k=sess.generation.top_k,
                do_sample=sess.generation.do_sample,
                repetition_penalty=sess.generation.repetition_penalty,
                repetition_window=sess.generation.repetition_window,
            )

            if sess.prompt_tokens is not None:
                tts_session.set_voice_prompt_tokens(sess.prompt_tokens)
            else:
                tts_session.clear_voice_prompt()

            decoder = AudioStreamDecoder(
                sess.codec,
                chunk_frames=sess.streaming.decode_chunk_frames,
                overlap_frames=sess.streaming.decode_overlap_frames,
                initial_chunk_frames=1,
                device=sess.device,
            )
            codebook_size = int(getattr(sess.codec, "codebook_size", 1024))
            audio_eos_token = int(getattr(inferencer, "audio_eos_token", 1026))
            frame_decoder = AudioFrameDecoder(
                decoder, codebook_size, audio_eos_token, StreamingCallbacks()
            )
            buffer_tracker = BufferedAudioTracker(SAMPLE_RATE)

            # Start a single codec.streaming context for the whole turn (matches app.py).
            # Re-entering codec.streaming repeatedly across deltas can cause audible drift.
            if sess.codec_stream_cm is not None:
                try:
                    sess.codec_stream_cm.__exit__(None, None, None)
                except Exception:
                    pass
                sess.codec_stream_cm = None
            try:
                sess.codec_stream_cm = sess.codec.streaming(batch_size=1)
                sess.codec_stream_cm.__enter__()
            except Exception:
                sess.codec_stream_cm = None
                raise

            sess.inferencer = inferencer
            sess.tts_session = tts_session
            sess.decoder = decoder
            sess.frame_decoder = frame_decoder
            sess.buffer_tracker = buffer_tracker

            # Prefer speech input when user audio tokens are available.
            if sess.user_tokens is not None:
                tts_session.reset_turn(
                    user_text=user_text,
                    user_audio_tokens=sess.user_tokens,
                    include_system_prompt=True,
                    reset_cache=True,
                )

                # If assistant text is already available at start, push its
                # tokens directly so generation can begin immediately.
                if assistant_text:
                    tokens = sess.tokenizer.encode(
                        assistant_text,
                        add_special_tokens=False,
                    )
                    if tokens:
                        audio_frames = tts_session.push_text_tokens(tokens)
                        self._emit_audio_chunks(
                            sess,
                            frame_decoder.decode_frames(audio_frames),
                        )

            else:
                # Otherwise, construct a text-only turn input.
                system_prompt = sess.processor.make_ensemble(sess.prompt_tokens)

                user_prompt_text = (
                    "<|im_end|>\n<|im_start|>user\n"
                    + (user_text or "")
                    + "<|im_end|>\n<|im_start|>assistant\n"
                )
                user_prompt_tokens = sess.processor.tokenizer(user_prompt_text)[
                    "input_ids"
                ]
                user_prompt = np.full(
                    shape=(len(user_prompt_tokens), sess.processor.channels + 1),
                    fill_value=sess.processor.audio_channel_pad,
                    dtype=np.int64,
                )
                user_prompt[:, 0] = np.asarray(user_prompt_tokens, dtype=np.int64)
                turn_input_ids = np.concatenate([system_prompt, user_prompt], axis=0)

                tts_session.reset_turn(
                    input_ids=turn_input_ids,
                    include_system_prompt=True,
                    reset_cache=True,
                )

                # If assistant text is already available at start, push its
                # tokens directly so generation can begin immediately.
                if assistant_text:
                    tokens = sess.tokenizer.encode(
                        assistant_text,
                        add_special_tokens=False,
                    )
                    if tokens:
                        audio_frames = tts_session.push_text_tokens(tokens)
                        self._emit_audio_chunks(
                            sess,
                            frame_decoder.decode_frames(audio_frames),
                        )

            sess.turn_active = True
            sess.turn_finished = False

    def _handle_push_text(self, sess: SessionRuntime, cmd: dict) -> None:
        with sess.lock:
            if not sess.turn_active or sess.tts_session is None:
                print("[session_worker_warning] push_text ignored: no active turn", flush=True)
                return

            text: str = cmd["text"]
            if not text:
                return
            tokens = sess.tokenizer.encode(text, add_special_tokens=False)
            if not tokens:
                return
            audio_frames = sess.tts_session.push_text_tokens(tokens)
            self._emit_audio_chunks(
                sess,
                sess.frame_decoder.decode_frames(audio_frames),
            )

    def _handle_finish_turn(self, sess: SessionRuntime) -> None:
        with sess.lock:
            if not sess.turn_active or sess.tts_session is None:
                self._safe_put_audio_end(sess)
                return
            audio_frames = sess.tts_session.end_text()
            self._emit_audio_chunks(
                sess,
                sess.frame_decoder.decode_frames(audio_frames),
            )

            while True:
                audio_frames = sess.tts_session.drain(max_steps=1)
                if not audio_frames:
                    break
                self._emit_audio_chunks(
                    sess,
                    sess.frame_decoder.decode_frames(audio_frames),
                )
                if sess.tts_session.inferencer.is_finished:
                    break

            self._emit_audio_chunks(sess, sess.frame_decoder.flush())

            # Close codec.streaming context for this turn.
            if sess.codec_stream_cm is not None:
                try:
                    sess.codec_stream_cm.__exit__(None, None, None)
                except Exception:
                    pass
                sess.codec_stream_cm = None

            sess.turn_active = False
            sess.turn_finished = True
            sess.audio_queue.put(None)


session_manager = SessionManager()


class SessionStartReq(BaseModel):
    session_id: str
    user_text: Optional[str] = None
    assistant_text: Optional[str] = None
    prompt_audio: Optional[str] = None
    user_audio: Optional[str] = None
    new_turn: bool = True


class SessionPushReq(BaseModel):
    session_id: str
    text: str
    is_final: bool = False


class SessionCloseReq(BaseModel):
    session_id: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "target_sr": TARGET_SR,
        "model_path": MODEL_PATH,
        "tokenizer_path": TOKENIZER_PATH,
        "codec_model_path": CODEC_MODEL_PATH,
        "device": DEVICE,
        "attn_impl": ATTN_IMPL,
    }


@app.post("/tts/session/start")
def tts_session_start(req: SessionStartReq):
    if not req.session_id.strip():
        raise HTTPException(status_code=400, detail="session_id is required")
    if not req.new_turn:
        raise HTTPException(status_code=400, detail="start endpoint requires new_turn=true")

    sess = session_manager.get_or_create(req.session_id)

    try:
        next_audio_queue: "queue.Queue[Optional[bytes]]" = queue.Queue()

        # If a previous turn is still active for this session_id, auto-finish it
        # when caller starts a new turn (new_turn=True).
        if req.new_turn and sess.turn_active:
            sess.command_queue.put({"type": "finish_turn"})

        with sess.lock:
            sess.pending_audio_queue = next_audio_queue

        sess.command_queue.put(
            {
                "type": "start_turn",
                "user_text": req.user_text,
                "assistant_text": req.assistant_text,
                "prompt_audio": req.prompt_audio,
                "user_audio": req.user_audio,
                "audio_queue": next_audio_queue,
            }
        )
        return {
            "ok": True,
            "session_id": req.session_id,
            "message": "turn started",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts/session/push")
def tts_session_push(req: SessionPushReq):
    sess = session_manager.get(req.session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="session not found")

    try:
        if req.text:
            sess.command_queue.put(
                {
                    "type": "push_text",
                    "text": req.text,
                }
            )
        if req.is_final:
            sess.command_queue.put({"type": "finish_turn"})

        return {
            "ok": True,
            "session_id": req.session_id,
            "accepted_text_len": len(req.text or ""),
            "is_final": req.is_final,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tts/session/{session_id}/audio")
def tts_session_audio(session_id: str):
    sess = session_manager.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="session not found")

    def gen():
        # Capture the pending turn queue if start has already been accepted but
        # the worker has not yet initialized the turn. This avoids attaching the
        # client to the previous turn's queue and then hanging forever.
        with sess.lock:
            q = sess.pending_audio_queue or sess.audio_queue
        while True:
            item = q.get()
            if item is None:
                break
            yield item

    return StreamingResponse(
        gen(),
        media_type="application/octet-stream",
        headers={
            "X-Audio-Codec": "pcm_s16le",
            "X-Audio-Sample-Rate": str(TARGET_SR),
            "X-Audio-Channels": "1",
            "X-Session-Id": session_id,
        },
    )


@app.post("/tts/session/close")
def tts_session_close(req: SessionCloseReq):
    sess = session_manager.get(req.session_id)
    if sess is not None:
        try:
            sess.command_queue.put({"type": "shutdown"})
        except Exception:
            pass
    session_manager.delete(req.session_id)
    return {"ok": True, "session_id": req.session_id, "message": "session closed"}

def main() -> None:
    global TARGET_SR, MODEL_PATH, TOKENIZER_PATH, CODEC_MODEL_PATH, DEVICE, ATTN_IMPL

    parser = argparse.ArgumentParser(description="MOSS TTS split request/audio server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8083)
    parser.add_argument("--target_sr", type=int, default=TARGET_SR)
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--tokenizer_path", type=str, default=TOKENIZER_PATH)
    parser.add_argument("--codec_model_path", type=str, default=CODEC_MODEL_PATH)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--attn_impl", type=str, default=ATTN_IMPL)
    args = parser.parse_args()

    TARGET_SR = int(args.target_sr)
    MODEL_PATH = args.model_path
    TOKENIZER_PATH = args.tokenizer_path
    CODEC_MODEL_PATH = args.codec_model_path
    DEVICE = args.device
    ATTN_IMPL = args.attn_impl

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
