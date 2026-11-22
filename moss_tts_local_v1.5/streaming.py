# coding=utf-8
"""Streaming inference helpers for MOSS-TTS Local Transformer v1.5.

The helper uses the Hugging Face local-transformer checkpoint together with
MOSS-Audio-Tokenizer-v2 and yields decoded audio chunks while generation is
still running.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, Optional

import torch
import torchaudio
from transformers import AutoModel, AutoProcessor


DEFAULT_MODEL_DIR = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
DEFAULT_CODEC_DIR = "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"
DEFAULT_OUTPUT_DIR = Path("outputs/moss_tts_local_v1_5_streaming")

TEXT_TO_AUDIO_TOKENS_PER_CHAR = {
    "zh": 3.098411951313033,
    "cmn": 3.098411951313033,
    "yue": 3.098411951313033,
    "chinese": 3.098411951313033,
    "english": 0.8673376262755219,
    "en": 0.8673376262755219,
    "french": 0.9,
    "fr": 0.9,
    "japanese": 2.2,
    "ja": 2.2,
    "korean": 1.8,
    "ko": 1.8,
}


@dataclass
class StreamingRuntime:
    model: Any
    processor: Any
    device: torch.device
    tts_device: torch.device
    codec_device: torch.device
    dtype: torch.dtype
    model_dir: str | Path
    lora_dir: str | Path | None
    lora_merged: bool
    codec_dir: str | Path
    sample_rate: int
    frame_rate: float
    n_vq: int
    attn_implementation: str
    codec_weight_dtype: str
    codec_compute_dtype: str
    codec_offload: bool = True


@dataclass
class StreamingRequest:
    text: str
    mode: str = "continuation"
    prompt_text: str = ""
    prompt_audio_path: Optional[str] = None
    language: str = ""
    tokens_control: bool = False
    tokens: int = 0
    max_new_frames: int = 7500
    do_sample: bool = True
    temperature: float = 1.2
    top_p: float = 1.0
    top_k: int = 25
    repetition_penalty: float = 1.0
    text_temperature: float = 1.0
    text_top_p: float = 1.0
    text_top_k: int = 50
    seed: Optional[int] = None
    codec_chunk_frames: int = 8
    prompt_audio_filename: Optional[str] = None


def _generated_audio_filename(request: StreamingRequest) -> str:
    reference = request.prompt_audio_filename or request.prompt_audio_path
    if not reference:
        return "generated.wav"
    # Handle client paths from either Windows or Unix without using their directories.
    basename = reference.replace("\\", "/").rsplit("/", 1)[-1]
    stem = Path(basename).stem
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f\x7f]', "_", stem).strip(" .")[:80].rstrip(" .")
    return f"{stem}_generated.wav" if stem else "generated.wav"


@dataclass
class StreamingEvent:
    type: str
    data: dict[str, Any] = field(default_factory=dict)


def _resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    normalized = str(dtype or "bfloat16").strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype!r}")


def resolve_attn_implementation(
    requested: str | None,
    device: torch.device,
    dtype: torch.dtype,
) -> str:
    requested_norm = str(requested or "flash_attention_2").strip().lower()
    if requested_norm in {"", "auto"}:
        requested_norm = "flash_attention_2"

    if requested_norm == "flash_attention_2":
        can_try_flash = (
            device.type == "cuda"
            and importlib.util.find_spec("flash_attn") is not None
            and dtype in {torch.float16, torch.bfloat16}
        )
        if can_try_flash:
            try:
                major, _ = torch.cuda.get_device_capability(device)
                if major >= 8:
                    return "flash_attention_2"
            except Exception:
                pass
        return "sdpa" if device.type == "cuda" else "eager"

    if requested_norm in {"sdpa", "eager"}:
        return requested_norm

    raise ValueError(
        "attn_implementation must be one of "
        "['auto', 'eager', 'flash_attention_2', 'sdpa'], "
        f"got {requested!r}."
    )


def _has_cjk(text: str) -> bool:
    return any("\u3400" <= ch <= "\u9fff" for ch in text or "")


def normalize_language_for_tokens(language: str, text: str) -> str:
    normalized = str(language or "").strip().lower().replace("_", "-")
    if normalized:
        if normalized.startswith("zh") or normalized in {"cmn", "yue", "chinese"}:
            return "zh"
        if normalized.startswith("en") or normalized == "english":
            return "en"
        if normalized.startswith("fr") or normalized == "french":
            return "fr"
        if normalized.startswith("ja") or normalized == "japanese":
            return "ja"
        if normalized.startswith("ko") or normalized == "korean":
            return "ko"
    return "zh" if _has_cjk(text) else "en"


def estimate_tokens(text: str, language: str = "") -> int:
    key = normalize_language_for_tokens(language, text)
    ratio = TEXT_TO_AUDIO_TOKENS_PER_CHAR.get(key, TEXT_TO_AUDIO_TOKENS_PER_CHAR["en"])
    return max(1, int(round(len(text or "") * float(ratio))))


def _move_batch_to_device(batch: Any, device: torch.device) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for key, value in dict(batch).items():
        if torch.is_tensor(value):
            result[key] = value.to(device)
        else:
            result[key] = value
    return result


def load_runtime(
    *,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    lora_dir: str | Path | None = None,
    merge_lora: bool = True,
    codec_dir: str | Path = DEFAULT_CODEC_DIR,
    device: str | torch.device = "cuda",
    tts_device: str | torch.device | None = None,
    codec_device: str | torch.device | None = None,
    dtype: str | torch.dtype = "bfloat16",
    attn_implementation: str = "flash_attention_2",
    codec_weight_dtype: str = "fp32",
    codec_compute_dtype: str = "bf16",
    codec_offload: bool = True,
    warmup: bool = True,
) -> StreamingRuntime:
    model_ref = str(model_dir)
    lora_ref = str(lora_dir).strip() if lora_dir is not None else ""
    codec_ref = str(codec_dir)
    local_model_path = Path(model_ref)
    local_files_only = local_model_path.exists()
    resolved_tts_device = torch.device(tts_device if tts_device is not None else device)
    resolved_codec_device = torch.device(codec_device if codec_device is not None else resolved_tts_device)
    if (resolved_tts_device.type == "cuda" or resolved_codec_device.type == "cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    resolved_dtype = _resolve_dtype(dtype)
    resolved_attn_implementation = resolve_attn_implementation(
        attn_implementation,
        resolved_tts_device,
        resolved_dtype,
    )

    processor = AutoProcessor.from_pretrained(
        model_ref,
        trust_remote_code=True,
        codec_path=codec_ref,
        codec_weight_dtype=codec_weight_dtype,
        codec_compute_dtype=codec_compute_dtype,
        codec_attention_implementation=resolved_attn_implementation,
    )
    model = AutoModel.from_pretrained(
        model_ref,
        trust_remote_code=True,
        dtype=resolved_dtype,
        local_files_only=local_files_only,
        attn_implementation=resolved_attn_implementation,
    )
    lora_merged = False
    if lora_ref:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError(
                "LoRA inference requires PEFT. Install it with `python -m pip install peft`."
            ) from exc

        model = PeftModel.from_pretrained(
            model,
            lora_ref,
            is_trainable=False,
        )
        if merge_lora:
            model = model.merge_and_unload()
            lora_merged = True

    model.to(resolved_tts_device)
    model.eval()

    audio_tokenizer = processor.audio_tokenizer
    if hasattr(audio_tokenizer, "set_attention_implementation"):
        audio_tokenizer.set_attention_implementation(resolved_attn_implementation)
    if hasattr(audio_tokenizer, "set_compute_dtype"):
        audio_tokenizer.set_compute_dtype(codec_compute_dtype)
    if hasattr(audio_tokenizer, "to"):
        # Keep the codec on CPU while idle so the large TTS model has the GPU
        # to itself.  The streaming worker moves it back for the short decode
        # window and returns it to CPU when the request finishes.
        audio_tokenizer.to("cpu" if codec_offload and resolved_codec_device.type == "cuda" else resolved_codec_device)
    if hasattr(audio_tokenizer, "eval"):
        audio_tokenizer.eval()

    sample_rate = int(getattr(audio_tokenizer, "sampling_rate", getattr(model.config, "sampling_rate", 48000)))
    downsample_rate = int(getattr(audio_tokenizer, "downsample_rate", 3840))
    n_vq = int(getattr(model.config, "n_vq", 12))
    runtime = StreamingRuntime(
        model=model,
        processor=processor,
        device=resolved_tts_device,
        tts_device=resolved_tts_device,
        codec_device=resolved_codec_device,
        dtype=resolved_dtype,
        model_dir=model_ref,
        lora_dir=lora_ref or None,
        lora_merged=lora_merged,
        codec_dir=codec_ref,
        sample_rate=sample_rate,
        frame_rate=float(sample_rate) / float(downsample_rate),
        n_vq=n_vq,
        attn_implementation=resolved_attn_implementation,
        codec_weight_dtype=str(codec_weight_dtype),
        codec_compute_dtype=str(codec_compute_dtype),
        codec_offload=bool(codec_offload),
    )
    if warmup:
        warmup_streaming_runtime(runtime)
    return runtime


def _build_prompt_fields(request: StreamingRequest) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "instruction": None,
        "quality": None,
        "sound_event": None,
        "ambient_sound": None,
        "language": request.language.strip() or None,
    }
    if request.tokens_control:
        tokens = int(request.tokens)
        if tokens <= 0:
            tokens = estimate_tokens(request.text, request.language)
        fields["tokens"] = tokens
    return fields


def _apply_repetition_penalty_with_seen_mask(
    scores: torch.Tensor,
    seen_mask: Optional[torch.Tensor],
    penalty: float,
) -> torch.Tensor:
    if seen_mask is None or float(penalty) == 1.0:
        return scores
    penalized = torch.where(scores < 0, scores * float(penalty), scores / float(penalty))
    return torch.where(seen_mask, penalized, scores)


def _sample_next_token_topk_subspace(
    *,
    model: Any,
    logits: torch.Tensor,
    do_sample: bool,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    seen_mask: Optional[torch.Tensor] = None,
    repetition_penalty: float = 1.0,
) -> torch.LongTensor:
    scores = logits.float()
    scores = _apply_repetition_penalty_with_seen_mask(scores, seen_mask, repetition_penalty)
    if not do_sample:
        return torch.argmax(scores, dim=-1)
    if float(temperature) <= 0:
        raise ValueError("temperature must be positive when do_sample=True.")

    vocab_size = int(scores.shape[-1])
    k = int(top_k) if top_k is not None else 0
    if k > 0 and k < vocab_size:
        top_values, top_indices = torch.topk(scores, k, dim=-1, sorted=True)
        top_values = top_values / float(temperature)
        if top_p is not None and 0.0 < float(top_p) < 1.0:
            top_probs = torch.softmax(top_values, dim=-1)
            cumulative_probs = top_probs.cumsum(dim=-1)
            remove_mask = cumulative_probs > float(top_p)
            remove_mask[..., 1:] = remove_mask[..., :-1].clone()
            remove_mask[..., 0] = False
            top_values = top_values.masked_fill(remove_mask, -torch.inf)
        probs = torch.softmax(top_values, dim=-1)
        sampled_offsets = torch.multinomial(probs, num_samples=1)
        return top_indices.gather(dim=-1, index=sampled_offsets).squeeze(-1)

    return model._sample_next_token(
        logits=scores,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        previous_token_ids=None,
        repetition_penalty=1.0,
    )


_USER_MESSAGE_KEYS = ("text", "reference", "instruction", "tokens", "quality", "sound_event", "ambient_sound", "language")


def build_continuation_clone_batch(
    processor: Any,
    user_message: dict[str, Any],
    prompt_audio_codes: torch.Tensor,
    n_vq: Optional[int] = None,
) -> dict[str, torch.Tensor]:
    """Continuation + Clone: a continuation prompt whose user turn also carries the prompt audio in Reference(s).

    The processor's own continuation path always renders Reference(s) as "None", and the
    processor is loaded from the model repo (trust_remote_code), so the sequence is assembled
    here: the voice-clone prompt (user turn with the reference, ending at the assistant
    <audio_start>) followed by the prompt audio as assistant audio rows -- exactly what
    continuation mode builds, plus the reference. start_length and decode() behave as in
    plain continuation (the prefix is trimmed from the decoded audio).
    """
    fields = {key: user_message.get(key) for key in _USER_MESSAGE_KEYS}
    fields["reference"] = [prompt_audio_codes]
    message = processor.build_user_message(**fields)
    clone = processor([message], mode="generation", n_vq=n_vq)["input_ids"][0]
    prefix = processor._build_audio_rows(
        prompt_audio_codes.to(device=clone.device, dtype=torch.long),
        int(processor.model_config.audio_assistant_slot_token_id),
    )
    return dict(processor._pad([torch.cat([clone, prefix], dim=0)]))


def build_processor_inputs(
    runtime: StreamingRuntime,
    request: StreamingRequest,
) -> tuple[dict[str, torch.Tensor], Optional[torch.Tensor], str]:
    processor = runtime.processor
    mode = str(request.mode or "continuation").strip().lower()
    if mode not in {"continuation", "voice_clone", "continuation_clone"}:
        raise ValueError("mode must be continuation, voice_clone or continuation_clone.")

    prompt_audio_path = (request.prompt_audio_path or "").strip() or None
    prompt_text = request.prompt_text or ""
    if mode == "voice_clone" and not prompt_audio_path:
        mode = "continuation"
    prompt_fields = _build_prompt_fields(request)
    user_kwargs = dict(
        text=request.text,
        instruction=prompt_fields.get("instruction"),
        tokens=prompt_fields.get("tokens"),
        quality=prompt_fields.get("quality"),
        sound_event=prompt_fields.get("sound_event"),
        ambient_sound=prompt_fields.get("ambient_sound"),
        language=prompt_fields.get("language"),
    )

    prompt_audio_codes: Optional[torch.Tensor] = None
    if mode == "voice_clone":
        conversation = [
            processor.build_user_message(
                reference=[prompt_audio_path],
                **user_kwargs,
            )
        ]
        processor_mode = "generation"
    elif prompt_audio_path:
        prompt_audio_codes = processor.encode_audios_from_path(prompt_audio_path, n_vq=runtime.n_vq)[0]
        continuation_text = prompt_text + request.text if prompt_text.strip() else request.text
        user_message = processor.build_user_message(
            text=continuation_text,
            instruction=prompt_fields.get("instruction"),
            tokens=prompt_fields.get("tokens"),
            quality=prompt_fields.get("quality"),
            sound_event=prompt_fields.get("sound_event"),
            ambient_sound=prompt_fields.get("ambient_sound"),
            language=prompt_fields.get("language"),
        )
        if mode == "continuation_clone":
            batch = build_continuation_clone_batch(processor, user_message, prompt_audio_codes, n_vq=runtime.n_vq)
            return _move_batch_to_device(batch, runtime.tts_device), prompt_audio_codes, "continuation"
        conversation = [
            user_message,
            processor.build_assistant_message(audio_codes_list=[prompt_audio_codes]),
        ]
        processor_mode = "continuation"
    else:
        # No-prompt continuation degenerates to direct TTS generation. This is
        # trained for TACv5 and avoids forcing users to provide a reference.
        conversation = [processor.build_user_message(**user_kwargs)]
        processor_mode = "generation"

    batch = processor(conversation, mode=processor_mode, n_vq=runtime.n_vq)
    return _move_batch_to_device(batch, runtime.tts_device), prompt_audio_codes, processor_mode


@torch.inference_mode()
def iter_generate_frames(
    model: Any,
    *,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor],
    max_new_frames: int,
    do_sample: bool,
    text_temperature: float,
    text_top_p: float,
    text_top_k: int,
    audio_temperature: float,
    audio_top_p: float,
    audio_top_k: int,
    audio_repetition_penalty: float,
    use_kv_cache: bool = True,
) -> Generator[StreamingEvent, None, None]:
    n_vq = int(model.config.n_vq)
    max_new_frames = int(max_new_frames)
    model._resolve_fixed_nq(n_vq_for_inference=n_vq)

    if input_ids.ndim == 2:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.ndim != 3:
        raise ValueError(f"Expected input_ids with 3 dims, got {tuple(input_ids.shape)}.")
    if input_ids.shape[-1] != n_vq + 1:
        raise ValueError(
            f"Expected {n_vq + 1} channels from config.n_vq, got {input_ids.shape[-1]}."
        )
    if attention_mask is None:
        attention_mask = torch.ones(input_ids.shape[:2], dtype=torch.bool, device=input_ids.device)
    elif attention_mask.ndim == 1:
        attention_mask = attention_mask.unsqueeze(0)
    attention_mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)

    batch_size = int(input_ids.shape[0])
    if batch_size != 1:
        raise ValueError("Streaming helper currently supports batch_size=1.")

    current_input_ids = input_ids
    current_attention_mask = attention_mask
    current_model_input_ids = current_input_ids
    generated_audio_history = torch.empty(
        (batch_size, max(0, max_new_frames), n_vq),
        dtype=torch.long,
        device=input_ids.device,
    )
    use_fast_audio_sampling = (
        n_vq >= 32
        and bool(do_sample)
        and audio_top_k is not None
        and int(audio_top_k) > 0
    )
    seen_audio_token_masks: Optional[list[torch.Tensor]] = None
    if use_fast_audio_sampling and float(audio_repetition_penalty) != 1.0:
        seen_audio_token_masks = [
            torch.zeros(
                (batch_size, int(model.config.audio_codebook_sizes[channel_index])),
                dtype=torch.bool,
                device=input_ids.device,
            )
            for channel_index in range(n_vq)
        ]
    generated_frame_count = 0
    finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
    past_key_values = None
    local_dtype = model.local_transformer.ln_f.weight.dtype

    for step in range(max_new_frames):
        global_inputs_embeds = model._build_inputs_embeds(current_model_input_ids)
        global_outputs = model.transformer(
            input_ids=None,
            past_key_values=past_key_values,
            attention_mask=current_attention_mask,
            position_ids=None,
            inputs_embeds=global_inputs_embeds,
            use_cache=use_kv_cache,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            cu_seqlens=None,
            num_sequences=None,
        )
        global_hidden_states = global_outputs.last_hidden_state[:, -1, :]
        local_global_hidden_states = model._global_hidden_to_local(global_hidden_states).to(dtype=local_dtype)

        local_prefix_hidden_states, local_prefix_past_key_values = model._decode_local_hidden_states_with_cache(
            local_global_hidden_states.unsqueeze(1)
        )
        local_hidden_states = local_prefix_hidden_states[:, -1, :]
        next_text_tokens = model._sample_next_assistant_text_token(
            local_hidden_states=local_hidden_states,
            do_sample=do_sample,
            temperature=text_temperature,
            top_k=text_top_k,
            top_p=text_top_p,
        )
        should_continue = next_text_tokens.eq(int(model.config.audio_assistant_slot_token_id)) & ~finished
        finished = finished | next_text_tokens.eq(int(model.config.audio_end_token_id))
        if not bool(should_continue.any().item()):
            break

        next_frame_tokens = []
        for channel_index in range(n_vq):
            channel_logits = model.audio_lm_heads[channel_index](local_hidden_states)
            if use_fast_audio_sampling:
                channel_token = _sample_next_token_topk_subspace(
                    model=model,
                    logits=channel_logits,
                    do_sample=do_sample,
                    temperature=audio_temperature,
                    top_k=audio_top_k,
                    top_p=audio_top_p,
                    seen_mask=(
                        None
                        if seen_audio_token_masks is None
                        else seen_audio_token_masks[channel_index]
                    ),
                    repetition_penalty=audio_repetition_penalty,
                )
            else:
                channel_token = model._sample_next_token(
                    logits=channel_logits,
                    do_sample=do_sample,
                    temperature=audio_temperature,
                    top_k=audio_top_k,
                    top_p=audio_top_p,
                    previous_token_ids=(
                        None
                        if generated_frame_count <= 0
                        else generated_audio_history[:, :generated_frame_count, channel_index]
                    ),
                    repetition_penalty=audio_repetition_penalty,
                )
            next_frame_tokens.append(channel_token)
            if seen_audio_token_masks is not None:
                seen_audio_token_masks[channel_index].scatter_(1, channel_token.unsqueeze(-1), True)
            if channel_index + 1 < n_vq:
                current_local_input = model.audio_embeddings[channel_index](channel_token).to(dtype=local_dtype)
                local_token_hidden_states, local_prefix_past_key_values = model._decode_local_hidden_states_with_cache(
                    current_local_input.unsqueeze(1),
                    past_key_values=local_prefix_past_key_values,
                )
                local_hidden_states = local_token_hidden_states[:, -1, :]

        next_frame = torch.stack(next_frame_tokens, dim=-1)
        next_frame = next_frame.masked_fill(~should_continue.unsqueeze(-1), int(model.config.audio_pad_token_id))
        generated_audio_history[:, generated_frame_count, :] = next_frame
        generated_frame_count += 1
        yield StreamingEvent(
            type="frame",
            data={
                "frame_index": step,
                "audio_token_ids": next_frame.detach().clone(),
                "finished": bool(finished.item()),
            },
        )

        next_row = model._build_generation_row(
            batch_size=batch_size,
            device=input_ids.device,
            audio_token_ids=next_frame,
        )
        if bool((~should_continue).any().item()):
            next_row[~should_continue, 0, 0] = int(model.config.pad_token_id)
            next_row[~should_continue, 0, 1:] = int(model.config.audio_pad_token_id)

        current_input_ids = torch.cat([current_input_ids, next_row], dim=1)
        current_attention_mask = torch.cat([current_attention_mask, should_continue.unsqueeze(1)], dim=1)
        if use_kv_cache:
            current_model_input_ids = next_row
            past_key_values = global_outputs.past_key_values
        else:
            current_model_input_ids = current_input_ids

    if generated_frame_count > 0:
        audio_token_ids = generated_audio_history[0, :generated_frame_count, :].detach().clone()
    else:
        audio_token_ids = torch.empty(
            (0, n_vq),
            dtype=torch.long,
            device=input_ids.device,
        )
    yield StreamingEvent(type="final_tokens", data={"audio_token_ids": audio_token_ids})


class StatefulCodecDecoder:
    def __init__(self, audio_tokenizer: Any, *, n_vq: int) -> None:
        self.audio_tokenizer = audio_tokenizer
        self.n_vq = int(n_vq)
        self._ctx = None

    @property
    def device(self) -> torch.device:
        try:
            return next(self.audio_tokenizer.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def __enter__(self) -> "StatefulCodecDecoder":
        self._ctx = self.audio_tokenizer.streaming(batch_size=1)
        self._ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._ctx is not None:
            self._ctx.__exit__(exc_type, exc, tb)
            self._ctx = None

    @torch.inference_mode()
    def decode_codes(self, codes: torch.LongTensor) -> torch.Tensor:
        if codes.numel() == 0:
            return torch.empty((2, 0), dtype=torch.float32, device=self.device)
        if codes.ndim != 2 or int(codes.shape[1]) != self.n_vq:
            raise ValueError(f"Expected codes shape [T, {self.n_vq}], got {tuple(codes.shape)}.")
        codes_qbt = codes.transpose(0, 1).contiguous().unsqueeze(1).to(device=self.device, dtype=torch.long)
        codes_lengths = torch.tensor([codes_qbt.shape[-1]], device=self.device, dtype=torch.long)
        active_mask = torch.tensor([codes_qbt.shape[-1] > 0], device=self.device, dtype=torch.bool)
        self.audio_tokenizer._set_streaming_exec_mask(active_mask)
        decoded = self.audio_tokenizer._decode_frame(codes_qbt, codes_lengths)
        if decoded.audio is None or decoded.audio_lengths is None:
            raise RuntimeError("audio tokenizer did not return audio/audio_lengths.")
        audio_length = int(decoded.audio_lengths[0].item())
        if audio_length <= 0:
            return torch.empty(
                (int(getattr(self.audio_tokenizer, "number_channels", 2)), 0),
                dtype=torch.float32,
                device=self.device,
            )
        return decoded.audio[0, :, :audio_length].detach().to(torch.float32)


@torch.inference_mode()
def warmup_streaming_runtime(runtime: StreamingRuntime, *, codec_frames: int = 4) -> None:
    """Pre-run small TTS and codec steps so the first user stream is hot."""
    codec = runtime.processor.audio_tokenizer
    if runtime.codec_offload and runtime.codec_device.type == "cuda":
        codec.to(runtime.codec_device)
    try:
        request = StreamingRequest(
            text="这是一个流式预热。", language="Chinese", max_new_frames=2, do_sample=False
        )
        batch, _, _ = build_processor_inputs(runtime, request)
        for event in iter_generate_frames(
            runtime.model,
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            max_new_frames=2,
            do_sample=False,
            text_temperature=1.0,
            text_top_p=1.0,
            text_top_k=1,
            audio_temperature=1.0,
            audio_top_p=1.0,
            audio_top_k=1,
            audio_repetition_penalty=1.0,
            use_kv_cache=True,
        ):
            if event.type == "frame":
                break
        if runtime.tts_device.type == "cuda":
            torch.cuda.synchronize(runtime.tts_device)
    except Exception as exc:  # noqa: BLE001
        print(f"[moss_tts_local_v1.5] TTS warmup skipped: {exc}")

    try:
        frames = max(1, int(codec_frames))
        dummy_codes = torch.zeros((frames, runtime.n_vq), dtype=torch.long)
        with StatefulCodecDecoder(runtime.processor.audio_tokenizer, n_vq=runtime.n_vq) as decoder:
            _ = decoder.decode_codes(dummy_codes.to(device=decoder.device, dtype=torch.long))
        if runtime.codec_device.type == "cuda":
            torch.cuda.synchronize(runtime.codec_device)
    except Exception as exc:  # noqa: BLE001
        print(f"[moss_tts_local_v1.5] codec warmup skipped: {exc}")
    finally:
        if runtime.codec_offload and runtime.codec_device.type == "cuda":
            codec.to("cpu")
            torch.cuda.empty_cache()


def _decode_budget_from_stream_state(
    *,
    lead_seconds: float,
    default_budget: int = 0,
    first_decode_submitted: bool = False,
    first_audio_emitted: bool = False,
    decode_queue_depth: int = 0,
    decode_chunks_submitted: int = 0,
    n_vq: int = 12,
) -> int:
    if int(default_budget) > 0:
        return max(1, int(default_budget))

    if int(n_vq) >= 32:
        # RVQ32 TTS generation has 32 local autoregressive steps per frame.
        # Decoding 1-2 frames at a time creates too many codec calls and can
        # starve the generator thread. Keep automatic streaming chunks larger
        # for RVQ32 only; lower-depth models keep the lower-latency ladder below.
        if not first_decode_submitted:
            return 4
        if not first_audio_emitted:
            return 4
        if lead_seconds < 0.64:
            return 4
        if lead_seconds < 1.0:
            return 6
        return 8

    # Lower-depth models can decode sooner than RVQ32, but a one-frame
    # first packet is only 80 ms at 12.5 fps and tends to underrun browser
    # playback. Start with four frames so the first PCM packet carries about
    # 320 ms of audio; after audio is flowing, switch back to the lead ladder.
    if not first_decode_submitted:
        return 4

    budget = _smooth_decode_budget_from_lead(lead_seconds)

    if not first_audio_emitted:
        return 4

    # Once audio is flowing, use lead as a smooth latency/throughput knob:
    # low or negative lead decodes short chunks to avoid audible gaps; healthy
    # lead decodes longer chunks to reduce codec call overhead.
    if decode_queue_depth > 0:
        return min(8, max(4, budget))
    return budget


def _smooth_decode_budget_from_lead(lead_seconds: float) -> int:
    if lead_seconds <= 0.0:
        return 1
    if lead_seconds < 0.16:
        return 2
    if lead_seconds < 0.32:
        return 3
    if lead_seconds < 0.64:
        return 4
    if lead_seconds < 1.0:
        return 6
    return 8


def _decode_budget_from_lead(lead_seconds: float, default_budget: int = 0) -> int:
    return _decode_budget_from_stream_state(
        lead_seconds=lead_seconds,
        default_budget=default_budget,
        first_decode_submitted=True,
        first_audio_emitted=True,
    )


def _save_waveform(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), waveform.detach().cpu().to(torch.float32), sample_rate)


def synthesize_stream(
    runtime: StreamingRuntime,
    request: StreamingRequest,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Generator[StreamingEvent, None, None]:
    if request.seed is not None and int(request.seed) >= 0:
        torch.manual_seed(int(request.seed))
        if runtime.tts_device.type == "cuda" or runtime.codec_device.type == "cuda":
            torch.cuda.manual_seed_all(int(request.seed))

    batch, prompt_audio_codes, processor_mode = build_processor_inputs(runtime, request)
    output_dir = Path(output_dir)
    run_id = f"{int(time.time() * 1000)}_{abs(hash((request.text, request.mode))) % 1000000:06d}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    yield StreamingEvent(
        type="metadata",
        data={
            "run_id": run_id,
            "run_dir": str(run_dir),
            "sample_rate": runtime.sample_rate,
            "frame_rate": runtime.frame_rate,
            "n_vq": runtime.n_vq,
            "tts_device": str(runtime.tts_device),
            "codec_device": str(runtime.codec_device),
            "attn_implementation": runtime.attn_implementation,
            "processor_mode": processor_mode,
            "tokens": _build_prompt_fields(request).get("tokens"),
        },
    )

    pending_frames: list[torch.LongTensor] = []
    emitted_segments: list[torch.Tensor] = []
    generated_frames = 0
    emitted_audio_seconds = 0.0
    generation_start = time.perf_counter()
    first_decode_submitted = False
    decode_chunks_submitted = 0
    first_audio_emitted_at: Optional[float] = None
    generated_frames_at_first_audio = 0
    final_audio_token_ids: Optional[torch.Tensor] = None
    decode_input_queue: queue.Queue[Optional[dict[str, Any]]] = queue.Queue(maxsize=8)
    decode_output_queue: queue.Queue[dict[str, Any]] = queue.Queue()

    def _codec_worker() -> None:
        codec = runtime.processor.audio_tokenizer
        try:
            if runtime.codec_offload and runtime.codec_device.type == "cuda":
                codec.to(runtime.codec_device)
            with StatefulCodecDecoder(codec, n_vq=runtime.n_vq) as decoder:
                if prompt_audio_codes is not None and prompt_audio_codes.numel() > 0:
                    _ = decoder.decode_codes(prompt_audio_codes.to(device=decoder.device, dtype=torch.long))
                while True:
                    item = decode_input_queue.get()
                    if item is None:
                        break
                    codes = item["codes"]
                    audio = decoder.decode_codes(codes.to(device=decoder.device, dtype=torch.long))
                    decode_output_queue.put(
                        {
                            "type": "audio",
                            "waveform": audio.detach().cpu(),
                            "generated_frames": int(item["generated_frames"]),
                            "lead_seconds": float(item.get("lead_seconds", 0.0)),
                            "chunk_frames": int(codes.shape[0]),
                        }
                    )
        except BaseException as exc:  # noqa: BLE001
            decode_output_queue.put({"type": "error", "error": repr(exc)})
        finally:
            if runtime.codec_offload and runtime.codec_device.type == "cuda":
                codec.to("cpu")
                torch.cuda.empty_cache()
            decode_output_queue.put({"type": "done"})

    decoder_thread = threading.Thread(target=_codec_worker, name="moss-tts-local-v1.5-codec-worker", daemon=True)
    decoder_thread.start()
    decoder_done = False

    def _generation_lead_seconds(*, now: Optional[float] = None) -> float:
        current_time = time.perf_counter() if now is None else now
        if first_audio_emitted_at is not None:
            post_first_elapsed_seconds = max(0.0, current_time - first_audio_emitted_at)
            post_first_generated_audio_seconds = max(
                0.0,
                (generated_frames - generated_frames_at_first_audio) / float(runtime.frame_rate),
            )
            return post_first_generated_audio_seconds - post_first_elapsed_seconds
        elapsed_seconds = max(0.0, current_time - generation_start)
        generated_audio_seconds = generated_frames / float(runtime.frame_rate)
        return generated_audio_seconds - emitted_audio_seconds - elapsed_seconds

    def _stream_metrics(*, now: Optional[float] = None) -> dict[str, Any]:
        current_time = time.perf_counter() if now is None else now
        elapsed_seconds = max(0.0, current_time - generation_start)
        generated_audio_seconds = generated_frames / float(runtime.frame_rate)
        generation_realtime_factor = (
            generated_audio_seconds / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
        )
        post_first_generation_realtime_factor: Optional[float] = None
        post_first_generated_audio_seconds: Optional[float] = None
        playback_lead_seconds: Optional[float] = None
        first_audio_latency_seconds: Optional[float] = None
        if first_audio_emitted_at is not None:
            first_audio_latency_seconds = max(0.0, first_audio_emitted_at - generation_start)
            post_first_elapsed_seconds = max(0.0, current_time - first_audio_emitted_at)
            post_first_generated_audio_seconds = max(
                0.0,
                (generated_frames - generated_frames_at_first_audio) / float(runtime.frame_rate),
            )
            post_first_generation_realtime_factor = (
                post_first_generated_audio_seconds / post_first_elapsed_seconds
                if post_first_elapsed_seconds > 0.0
                else 0.0
            )
            playback_lead_seconds = emitted_audio_seconds - post_first_elapsed_seconds
        return {
            "generated_audio_seconds": generated_audio_seconds,
            "generation_lead_seconds": _generation_lead_seconds(now=current_time),
            "generation_elapsed_seconds": elapsed_seconds,
            "generation_realtime_factor": generation_realtime_factor,
            "post_first_generated_audio_seconds": post_first_generated_audio_seconds,
            "post_first_generation_realtime_factor": post_first_generation_realtime_factor,
            "playback_lead_seconds": playback_lead_seconds,
            "first_audio_latency_seconds": first_audio_latency_seconds,
            "decode_chunks_submitted": decode_chunks_submitted,
            "decode_queue_depth": decode_input_queue.qsize(),
            "pending_decode_frames": len(pending_frames),
        }

    def _drain_decoder_outputs(*, block: bool = False) -> list[StreamingEvent]:
        nonlocal emitted_audio_seconds, decoder_done, first_audio_emitted_at, generated_frames_at_first_audio
        drained: list[StreamingEvent] = []
        while True:
            try:
                item = decode_output_queue.get(block=block and not drained, timeout=0.05 if block and not drained else 0.0)
            except queue.Empty:
                break
            if item.get("type") == "done":
                decoder_done = True
                continue
            if item.get("type") == "error":
                raise RuntimeError(f"codec worker failed: {item.get('error')}")
            audio = item["waveform"]
            emitted_segments.append(audio)
            now = time.perf_counter()
            if first_audio_emitted_at is None:
                first_audio_emitted_at = now
                generated_frames_at_first_audio = int(item.get("generated_frames", generated_frames))
            emitted_audio_seconds += int(audio.shape[-1]) / float(runtime.sample_rate)
            metrics = _stream_metrics(now=now)
            drained.append(
                StreamingEvent(
                    type="audio",
                    data={
                        "waveform": audio,
                        "sample_rate": runtime.sample_rate,
                        "generated_frames": int(item.get("generated_frames", generated_frames)),
                        "emitted_audio_seconds": emitted_audio_seconds,
                        "lead_seconds": float(item.get("lead_seconds", 0.0)),
                        "chunk_frames": int(item.get("chunk_frames", 0)),
                        "tts_device": str(runtime.tts_device),
                        "codec_device": str(runtime.codec_device),
                        **metrics,
                    },
                )
            )
        return drained

    try:
        for event in iter_generate_frames(
            runtime.model,
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            max_new_frames=int(request.max_new_frames),
            do_sample=bool(request.do_sample),
            text_temperature=float(request.text_temperature),
            text_top_p=float(request.text_top_p),
            text_top_k=int(request.text_top_k),
            audio_temperature=float(request.temperature),
            audio_top_p=float(request.top_p),
            audio_top_k=int(request.top_k),
            audio_repetition_penalty=float(request.repetition_penalty),
            use_kv_cache=True,
        ):
            for audio_event in _drain_decoder_outputs(block=False):
                yield audio_event
            if event.type == "frame":
                frame = event.data["audio_token_ids"][0].detach().cpu().clone()
                pending_frames.append(frame)
                generated_frames += 1
                now = time.perf_counter()
                lead_seconds = _generation_lead_seconds(now=now)
                decode_budget = _decode_budget_from_stream_state(
                    lead_seconds=lead_seconds,
                    default_budget=int(request.codec_chunk_frames),
                    first_decode_submitted=first_decode_submitted,
                    first_audio_emitted=first_audio_emitted_at is not None,
                    decode_queue_depth=decode_input_queue.qsize(),
                    decode_chunks_submitted=decode_chunks_submitted,
                    n_vq=runtime.n_vq,
                )
                if len(pending_frames) >= decode_budget:
                    codes = torch.stack(pending_frames, dim=0).contiguous()
                    pending_frames.clear()
                    first_decode_submitted = True
                    decode_chunks_submitted += 1
                    decode_input_queue.put(
                        {
                            "codes": codes,
                            "generated_frames": generated_frames,
                            "lead_seconds": lead_seconds,
                        }
                    )
                for audio_event in _drain_decoder_outputs(block=False):
                    yield audio_event
                yield StreamingEvent(
                    type="progress",
                    data={
                        "generated_frames": generated_frames,
                        "emitted_audio_seconds": emitted_audio_seconds,
                        "tts_device": str(runtime.tts_device),
                        "codec_device": str(runtime.codec_device),
                        **_stream_metrics(),
                    },
                )
            elif event.type == "final_tokens":
                final_audio_token_ids = event.data["audio_token_ids"].detach().cpu()

        if pending_frames:
            codes = torch.stack(pending_frames, dim=0).contiguous()
            pending_frames.clear()
            first_decode_submitted = True
            decode_chunks_submitted += 1
            decode_input_queue.put(
                {
                    "codes": codes,
                    "generated_frames": generated_frames,
                    "lead_seconds": 0.0,
                }
            )
    finally:
        decode_input_queue.put(None)

    while not decoder_done or not decode_output_queue.empty():
        for audio_event in _drain_decoder_outputs(block=True):
            yield audio_event
        if decoder_done:
            break
    decoder_thread.join(timeout=5.0)

    final_audio = (
        torch.cat(emitted_segments, dim=-1)
        if emitted_segments
        else torch.empty((2, 0), dtype=torch.float32)
    )
    audio_path = run_dir / _generated_audio_filename(request)
    tokens_path = run_dir / "audio_tokens.pt"
    meta_path = run_dir / "metadata.json"
    _save_waveform(audio_path, final_audio, runtime.sample_rate)
    if final_audio_token_ids is None:
        final_audio_token_ids = torch.empty((0, runtime.n_vq), dtype=torch.long)
    torch.save(final_audio_token_ids, tokens_path)
    final_metrics = _stream_metrics()
    metadata = {
        "run_id": run_id,
        "mode": request.mode,
        "processor_mode": processor_mode,
        "text": request.text,
        "prompt_text": request.prompt_text,
        "prompt_audio_path": request.prompt_audio_path,
        "prompt_audio_filename": request.prompt_audio_filename,
        "language": request.language,
        "tokens_control": request.tokens_control,
        "tokens": _build_prompt_fields(request).get("tokens"),
        "max_new_frames": request.max_new_frames,
        "generated_frames": int(final_audio_token_ids.shape[0]),
        "sample_rate": runtime.sample_rate,
        "duration_seconds": final_audio.shape[-1] / float(runtime.sample_rate),
        "first_audio_latency_seconds": final_metrics["first_audio_latency_seconds"],
        "generation_realtime_factor": final_metrics["generation_realtime_factor"],
        "post_first_generation_realtime_factor": final_metrics["post_first_generation_realtime_factor"],
        "generation_lead_seconds": final_metrics["generation_lead_seconds"],
        "decode_chunks_submitted": final_metrics["decode_chunks_submitted"],
        "audio_path": str(audio_path),
        "tokens_path": str(tokens_path),
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    yield StreamingEvent(
        type="result",
        data={
            "waveform": final_audio,
            "sample_rate": runtime.sample_rate,
            "audio_path": str(audio_path),
            "tokens_path": str(tokens_path),
            "metadata_path": str(meta_path),
            "metadata": metadata,
            "audio_token_ids": final_audio_token_ids,
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MOSS-TTS Local Transformer v1.5 streaming inference runner.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--lora-dir", default="", help="Optional PEFT LoRA adapter directory or Hugging Face repo ID.")
    parser.add_argument(
        "--merge-lora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Merge the LoRA adapter into the base model after loading (default: enabled).",
    )
    parser.add_argument("--codec-dir", default=str(DEFAULT_CODEC_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--text", default="这是一个流式推理测试。")
    parser.add_argument("--mode", default="continuation", choices=["continuation", "voice_clone", "continuation_clone"])
    parser.add_argument("--prompt-text", default="")
    parser.add_argument("--prompt-audio-path", default="")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--tokens-control", action="store_true")
    parser.add_argument("--tokens", type=int, default=0)
    parser.add_argument("--max-new-frames", type=int, default=8)
    parser.add_argument("--device", default="cuda", help="Legacy default device used when --tts-device is omitted.")
    parser.add_argument("--tts-device", default="", help="Device for the TTS model, e.g. cuda:0.")
    parser.add_argument("--codec-device", default="", help="Device for the codec, e.g. cuda:1. Defaults to TTS device.")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
    )
    parser.add_argument(
        "--codec-weight-dtype",
        default="fp32",
        choices=["bf16", "bfloat16", "fp32", "float32"],
        help="Codec encoder/decoder parameter dtype. Defaults to fp32; pass bf16 to reduce memory. The quantizer stays fp32.",
    )
    parser.add_argument(
        "--codec-compute-dtype",
        default="bf16",
        choices=["bf16", "fp32"],
        help="Codec non-quantizer autocast compute dtype.",
    )
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--text-temperature", type=float, default=1.0)
    parser.add_argument("--text-top-p", type=float, default=1.0)
    parser.add_argument("--text-top-k", type=int, default=50)
    parser.add_argument(
        "--codec-chunk-frames",
        type=int,
        default=8,
        help="Codec decode chunk size. 0 uses adaptive streaming scheduling.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    runtime = load_runtime(
        model_dir=args.model_dir,
        lora_dir=args.lora_dir or None,
        merge_lora=args.merge_lora,
        codec_dir=args.codec_dir,
        device=args.device,
        tts_device=args.tts_device or args.device,
        codec_device=args.codec_device or args.tts_device or args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        codec_weight_dtype=args.codec_weight_dtype,
        codec_compute_dtype=args.codec_compute_dtype,
    )
    request = StreamingRequest(
        text=args.text,
        mode=args.mode,
        prompt_text=args.prompt_text,
        prompt_audio_path=args.prompt_audio_path or None,
        language=args.language,
        tokens_control=bool(args.tokens_control),
        tokens=int(args.tokens),
        max_new_frames=int(args.max_new_frames),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        repetition_penalty=float(args.repetition_penalty),
        text_temperature=float(args.text_temperature),
        text_top_p=float(args.text_top_p),
        text_top_k=int(args.text_top_k),
        codec_chunk_frames=int(args.codec_chunk_frames),
        seed=int(args.seed),
    )
    last_result = None
    for event in synthesize_stream(runtime, request, output_dir=args.output_dir):
        if event.type == "metadata":
            print(json.dumps(event.data, ensure_ascii=False))
        elif event.type == "progress":
            print(json.dumps(event.data, ensure_ascii=False))
        elif event.type == "audio":
            print(
                json.dumps(
                    {
                        "audio_chunk_samples": int(event.data["waveform"].shape[-1]),
                        "generated_frames": event.data["generated_frames"],
                        "chunk_frames": event.data.get("chunk_frames"),
                        "first_audio_latency_seconds": event.data.get("first_audio_latency_seconds"),
                        "generation_realtime_factor": event.data.get("generation_realtime_factor"),
                        "post_first_generation_realtime_factor": event.data.get(
                            "post_first_generation_realtime_factor"
                        ),
                    },
                    ensure_ascii=False,
                )
            )
        elif event.type == "result":
            last_result = event.data
    if last_result is not None:
        print(json.dumps(last_result["metadata"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
