import os
import argparse
import base64
import functools
import json
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Sequence

import gradio as gr
import numpy as np

import torch
import torchaudio

from transformers import AutoTokenizer, AutoModel
from mossttsrealtime import MossTTSRealtime, MossTTSRealtimeProcessor
from mossttsrealtime.streaming_mossttsrealtime import (
    AudioStreamDecoder,
    MossTTSRealtimeInference,
    MossTTSRealtimeStreamingSession,
)
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64

SAMPLE_RATE = 24000
CODEC_MODEL_PATH = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
MODEL_PATH = (
    "/OpenMOSS-Team/MOSS-TTS-Realtime"
)
TOKENIZER_PATH = "OpenMOSS-Team/MOSS-TTS-Realtime"
PROMPT_WAV = "./audio/prompt_audio1.mp3"
USER_WAV = "./audio/user1.wav"



def _path_or_env(env_name: str, default_path: str | None = None) -> Path | None:
    value = os.getenv(env_name)
    if value:
        return Path(value)
    if default_path is not None:
        candidate = Path(default_path)
        if candidate.exists():
            return candidate
    return None


def _resolve_path(path_value: str | None, env_name: str, default_path: str | None) -> Path | None:
    if path_value:
        return Path(path_value).expanduser()
    return _path_or_env(env_name, default_path)


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
class StreamingCallbacks:
    on_text_stream_start: Callable[[], None] | None = None
    on_text_stream_stop: Callable[[], None] | None = None
    on_audio_stream_start: Callable[[], None] | None = None
    on_audio_stream_stop: Callable[[], None] | None = None


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


class TokenChunkStream:
    def __init__(
        self,
        tokens: Sequence[int],
        chunk_size: int,
        callbacks: StreamingCallbacks | None = None,
    ):
        self._tokens = list(tokens)
        self._chunk_size = int(chunk_size)
        self._callbacks = callbacks

    def __iter__(self) -> Iterator[list[int]]:
        if not self._tokens:
            return
        started = False
        try:
            step = len(self._tokens) if self._chunk_size <= 0 else self._chunk_size
            for idx in range(0, len(self._tokens), step):
                if not started:
                    started = True
                    if self._callbacks and self._callbacks.on_text_stream_start:
                        self._callbacks.on_text_stream_start()
                yield self._tokens[idx : idx + step]
        finally:
            if started and self._callbacks and self._callbacks.on_text_stream_stop:
                self._callbacks.on_text_stream_stop()


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
        callbacks: StreamingCallbacks | None = None,
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
        if hasattr(model, "language_model") and hasattr(model.language_model, "config"):
            model.language_model.config.attn_implementation = "flash_attention_2"
    else:
        model = MossTTSRealtime.from_pretrained(model_path, torch_dtype=dtype).to(device)
    model.eval()

    codec = _load_codec(device, codec_model_path)
    return model, tokenizer, processor, codec, device


def _resolve_audio_path(audio_path: str | None, use_default: bool, default_path: str) -> Path | None:
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
            raise ValueError("assistant_text is required.")
        if not request.assistant_text.strip():
            raise ValueError("assistant_text is required.")

        prompt_path = _resolve_audio_path(request.prompt_audio, request.use_default_prompt, PROMPT_WAV)
        user_path = _resolve_audio_path(request.user_audio, request.use_default_user, USER_WAV)

        if prompt_path is not None and not prompt_path.exists():
            raise FileNotFoundError(f"Prompt wav not found: {prompt_path}")
        if user_path is not None and not user_path.exists():
            raise FileNotFoundError(f"User wav not found: {user_path}")

        return prompt_path, user_path

    def _encode_audio_tokens(self, path: Path, codec, device: torch.device, chunk_duration: float) -> np.ndarray:
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

    def run_stream(self, request: StreamingRequest) -> Iterator[StreamEvent]:
        prompt_path, user_path = self._validate_request(request)
        model, tokenizer, processor, codec, device = self.get_or_load_backend(request.backend)

        if request.generation.seed is not None:
            torch.manual_seed(request.generation.seed)
            torch.cuda.manual_seed_all(request.generation.seed)

        prompt_tokens: np.ndarray | None = None
        user_tokens: np.ndarray | None = None
        if prompt_path is not None:
            prompt_tokens = self._encode_audio_tokens(
                prompt_path, codec, device, chunk_duration=request.streaming.chunk_duration
            )
        if user_path is not None:
            user_tokens = self._encode_audio_tokens(user_path, codec, device, chunk_duration=request.streaming.chunk_duration)

        inferencer = MossTTSRealtimeInference(model, tokenizer, max_length=request.generation.max_length)
        inferencer.reset_generation_state(keep_cache=False)
        session = MossTTSRealtimeStreamingSession(
            inferencer,
            processor,
            codec=codec,
            codec_sample_rate=SAMPLE_RATE,
            codec_encode_kwargs={"chunk_duration": request.streaming.chunk_duration},
            prefill_text_len=processor.delay_tokens_len,
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

        if user_tokens is None:
            turn_input_ids = self._build_text_only_turn_input(processor, request.user_text, prompt_tokens)
            session.reset_turn(input_ids=turn_input_ids, include_system_prompt=True, reset_cache=True)
            yield StreamEvent(message="No user audio provided, running text-only turn.")
        else:
            session.reset_turn(
                user_text=request.user_text,
                user_audio_tokens=user_tokens,
                include_system_prompt=True,
                reset_cache=True,
            )

        decoder = AudioStreamDecoder(
            codec,
            chunk_frames=request.streaming.decode_chunk_frames,
            overlap_frames=request.streaming.decode_overlap_frames,
            decode_kwargs={"chunk_duration": -1},
            device=device,
        )

        codebook_size = int(getattr(codec, "codebook_size", 1024))
        audio_eos_token = int(getattr(inferencer, "audio_eos_token", 1026))
        text_tokens = tokenizer.encode(request.assistant_text, add_special_tokens=False)
        if not text_tokens:
            raise RuntimeError("Assistant text tokenization returned no tokens.")

        callbacks = StreamingCallbacks()
        token_stream = TokenChunkStream(text_tokens, request.streaming.text_chunk_tokens, callbacks)
        frame_decoder = AudioFrameDecoder(decoder, codebook_size, audio_eos_token, callbacks)
        buffer_tracker = BufferedAudioTracker(SAMPLE_RATE)

        chunk_index = 0
        has_audio = False
        prebuffer_target = max(0.0, float(request.streaming.prebuffer_seconds))
        prebuffering = prebuffer_target > 0.0
        prebuffer_chunks: list[np.ndarray] = []
        prebuffer_samples = 0

        def _emit_buffered(chunks: list[np.ndarray], prefix: str):
            nonlocal chunk_index, has_audio
            for buffered in chunks:
                chunk_index += 1
                has_audio = True
                buffer_tracker.add_chunk(buffered)
                yield StreamEvent(message=f"{prefix} chunk {chunk_index}", audio=(SAMPLE_RATE, buffered))

        def _emit_chunk(chunk: np.ndarray, prefix: str):
            nonlocal prebuffering, prebuffer_samples, prebuffer_chunks
            if prebuffering:
                prebuffer_chunks.append(chunk)
                prebuffer_samples += int(chunk.size)
                buffered_seconds = prebuffer_samples / SAMPLE_RATE
                if buffered_seconds < prebuffer_target:
                    return
                prebuffering = False
                yield from _emit_buffered(prebuffer_chunks, prefix)
                prebuffer_chunks = []
                prebuffer_samples = 0
                return
            yield from _emit_buffered([chunk], prefix)

        def _flush_prebuffer(prefix: str):
            nonlocal prebuffering, prebuffer_samples, prebuffer_chunks
            if not prebuffering or not prebuffer_chunks:
                prebuffering = False
                return
            prebuffering = False
            yield from _emit_buffered(prebuffer_chunks, prefix)
            prebuffer_chunks = []
            prebuffer_samples = 0

        def _emit_chunks(chunks: Iterator[np.ndarray], prefix: str):
            for chunk in chunks:
                if chunk.size == 0:
                    continue
                yield from _emit_chunk(chunk, prefix)

        with codec.streaming(batch_size=1):
            for token_chunk in token_stream:
                _maybe_wait_for_buffer(buffer_tracker, request.streaming.buffer_threshold_seconds)
                audio_frames = session.push_text_tokens(token_chunk)
                for event in _emit_chunks(frame_decoder.decode_frames(audio_frames), "Streaming"):
                    yield event
                if request.streaming.input_delay > 0:
                    time.sleep(request.streaming.input_delay)

            audio_frames = session.end_text()
            for event in _emit_chunks(frame_decoder.decode_frames(audio_frames), "Finalizing"):
                yield event

            while True:
                audio_frames = session.drain(max_steps=1)
                if not audio_frames:
                    break
                for event in _emit_chunks(frame_decoder.decode_frames(audio_frames), "Finalizing"):
                    yield event
                if session.inferencer.is_finished:
                    break

            for event in _emit_chunks(frame_decoder.flush(), "Final"):
                yield event

            for event in _flush_prebuffer("Final"):
                yield event

        if not has_audio:
            raise RuntimeError("No audio waveform chunks decoded from streaming inference.")

        yield StreamEvent(message="Streaming complete.")


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
<div id="pcm-stream-status" style="font-size: 12px; color: #555;">
  Live playback uses Web Audio API. Click Generate to unlock audio. First load may take a while.
</div>
<div id="pcm-stream-meta" style="font-size: 12px; color: #333; margin: 6px 0;">
  <div>Now Playing Chunk: <span id="pcm-stream-playing">-</span></div>
  <div>Last Yielded Chunk: <span id="pcm-stream-yielded">-</span></div>
</div>
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

const statusEl = document.getElementById("pcm-stream-status");
const playingEl = document.getElementById("pcm-stream-playing");
const yieldedEl = document.getElementById("pcm-stream-yielded");
function setStatus(msg) {
  if (statusEl) {
    statusEl.textContent = msg;
  }
}
function setPlaying(idx) {
  if (playingEl) {
    playingEl.textContent = `${idx}`;
  }
}
function setYielded(idx) {
  if (yieldedEl) {
    yieldedEl.textContent = `${idx}`;
  }
}

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
  setPlaying(idx);
  setStatus(`Streaming... (chunk ${idx})`);
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
    setPlaying("-");
    setYielded("-");
    setStatus("Live playback uses Web Audio API. Click Generate to unlock audio. First load may take a while.");
    return;
  }
  const idx = payload.idx ?? 0;
  if (idx <= lastIdx) return;
  lastIdx = idx;
  const sr = payload.sr || 24000;
  const samples = decodeBase64ToFloat32(payload.data);
  setYielded(idx);
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


def _build_demo(args: argparse.Namespace):
    tts_demo = StreamingTTSDemo()

    with gr.Blocks(title="MossTTSRealtime") as demo:
        gr.Markdown("MossTTSRealtime demo")
        gr.Markdown("Note: The first run may take a while to load the model.")
        gr.HTML(STREAM_PLAYER_HTML, js_on_load=STREAM_PLAYER_JS)

        with gr.Row():
            with gr.Column():
                user_text = gr.Textbox(label="User Text(optional)", lines=2)
                assistant_text = gr.Textbox(label="Assistant Text", lines=6)
                prompt_audio = gr.Audio(label="Prompt WAV (optional)", type="filepath")
                user_audio = gr.Audio(label="User WAV (optional)", type="filepath")
                use_default_prompt = gr.Checkbox(label="Use Default Prompt WAV (fallback)", value=False)
                use_default_user = gr.Checkbox(label="Use Default User WAV (fallback)", value=False)

                with gr.Accordion("Generation Options", open=False):
                    temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="Temperature")
                    top_p = gr.Slider(0.1, 1.0, value=0.6, step=0.05, label="Top P")
                    top_k = gr.Slider(1, 100, value=30, step=1, label="Top K")
                    repetition_penalty = gr.Slider(1.0, 2.0, value=1.1, step=0.05, label="Repetition Penalty")
                    repetition_window = gr.Slider(1, 200, value=50, step=1, label="Repetition Window")
                    do_sample = gr.Checkbox(label="Do Sample", value=True)
                    max_length = gr.Slider(100, 10000, value=2000, step=10, label="Max Length")
                    seed = gr.Number(value=0, precision=0, label="Seed (0 for random)")

                with gr.Accordion("Streaming Options", open=False):
                    stream_text_chunk_tokens = gr.Slider(1, 64, value=12, step=1, label="Text Chunk Tokens")
                    stream_input_delay = gr.Slider(0.0, 0.5, value=0.0, step=0.05, label="Input Delay (s)")
                    stream_decode_chunk_frames = gr.Slider(0, 20, value=12, step=1, label="Decode Chunk Frames")
                    stream_decode_overlap_frames = gr.Slider(0, 10, value=0, step=1, label="Decode Overlap Frames")
                    chunk_duration = gr.Slider(0.0, 1.0, value=0.24, step=0.01, label="Codec Chunk Duration (s)")
                    stream_prebuffer_seconds = gr.Slider(0.0, 20.0, value=0.0, step=0.05, label="Initial Buffer (s)")

                run_btn = gr.Button("Generate", elem_id="tts_generate")

            with gr.Column():
                stream_data = gr.Textbox(label="PCM Stream (JSON)", elem_id="pcm_stream", interactive=False, lines=6)
                output_audio = gr.Audio(label="Final Audio", type="numpy")
                status = gr.Textbox(label="Status", lines=3)

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
            try:
                started_at = time.monotonic()
                seed = None if seed_value is None else int(seed_value)
                if seed == 0:
                    seed = None
                full_chunks: list[np.ndarray] = []
                first_chunk_time: float | None = None
                sample_rate = SAMPLE_RATE
                stream_reset = json.dumps({"reset": True})
                yield stream_reset, gr.update(value=None), "Started"

                request = StreamingRequest(
                    user_text=str(user_text_value or "Hello!"),
                    assistant_text=str(assistant_text_value or ""),
                    prompt_audio=prompt_audio_value,
                    user_audio=user_audio_value,
                    use_default_prompt=bool(use_default_prompt_value),
                    use_default_user=bool(use_default_user_value),
                    generation=GenerationConfig(
                        temperature=float(temperature_value),
                        top_p=float(top_p_value),
                        top_k=int(top_k_value),
                        repetition_penalty=float(repetition_penalty_value),
                        repetition_window=int(repetition_window_value),
                        do_sample=bool(do_sample_value),
                        max_length=int(max_length_value),
                        seed=seed,
                    ),
                    streaming=StreamingConfig(
                        text_chunk_tokens=int(stream_text_chunk_tokens_value),
                        input_delay=float(stream_input_delay_value),
                        decode_chunk_frames=int(stream_decode_chunk_frames_value),
                        decode_overlap_frames=int(stream_decode_overlap_frames_value),
                        chunk_duration=float(chunk_duration_value),
                        prebuffer_seconds=float(stream_prebuffer_seconds_value),
                    ),
                    backend=BackendPaths(
                        model_path=args.model_path,
                        tokenizer_path=args.tokenizer_path,
                        codec_model_path=args.codec_model_path,
                        device_str=args.device,
                        attn_impl=args.attn_implementation,
                    ),
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
                    elapsed = time.monotonic() - started_at
                    audio_seconds = float(full_audio.size) / float(sample_rate) if full_audio.size > 0 else 0.0
                    rtf = (elapsed / audio_seconds) if audio_seconds > 0 else float("inf")
                    done_msg = (
                        f"Done | chunks={len(full_chunks)} | audio={audio_seconds:.2f}s | "
                        f"elapsed={elapsed:.2f}s | RTF={rtf:.3f}"
                    )
                    if first_chunk_time is not None:
                        done_msg += f" | TTFB={(first_chunk_time - started_at) * 1000.0:.0f}ms"
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

    return demo


def main():
    parser = argparse.ArgumentParser(description="MossTTSRealtime streaming TTS Gradio demo")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--tokenizer_path", type=str, default=TOKENIZER_PATH)
    parser.add_argument("--codec_model_path",type=str,default=None,)
    parser.add_argument("--codec_root", type=str, default=None)
    parser.add_argument("--codec_config", type=str, default=None)
    parser.add_argument("--codec_ckpt", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    codec_model_path = _resolve_path(args.codec_model_path, "MossTTSRealtime_CODEC_MODEL_PATH", CODEC_MODEL_PATH)
    if codec_model_path is None or not codec_model_path.exists():
        raise FileNotFoundError(
            "Codec model path not found. Set --codec_model_path or env MossTTSRealtime_CODEC_MODEL_PATH."
        )
    args.codec_model_path = str(codec_model_path)

    demo = _build_demo(args)
    demo.queue(max_size=10, default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
