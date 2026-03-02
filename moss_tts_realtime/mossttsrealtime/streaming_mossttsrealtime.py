# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"""Streaming inference utilities for MossTTSRealtime."""

from __future__ import annotations

import contextlib
import re
from typing import Iterable, Iterator, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from transformers.cache_utils import DynamicCache, StaticCache
from transformers.utils import is_torchaudio_available, requires_backends
from transformers.utils.import_utils import requires

if is_torchaudio_available():
    import torchaudio


@requires(backends=("torch",))
class MossTTSRealtimeInference:
    """Step-wise inference wrapper for MossTTSRealtime.
    This class mirrors the non-streaming inference logic but exposes a
    prefill/step/finish API for streaming usage.
    """

    def __init__(
        self,
        model,
        tokenizer,
        max_length: int = 1000,
        channels: int = 16,
        audio_channel_pad: int = 1024,
        audio_bos_token: int = 1025,
        audio_eos_token: int = 1026,
        text_pad_id: int = 151655,
        aud_pad_id: int = 151654,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.channels = channels
        self.audio_channel_pad = audio_channel_pad
        self.audio_bos_token = audio_bos_token
        self.audio_eos_token = audio_eos_token
        self.text_pad_id = text_pad_id
        self.aud_pad_id = aud_pad_id

        self.past_key_values = None
        self.attention_mask = None
        self._generated_tokens: List[torch.Tensor] = []
        self._is_stopping = None
        self._last_audio_tokens = None
        self._step_idx = 0
        attn_impl = ""
        for cfg in (
            getattr(getattr(self.model, "local_transformer", None), "config", None),
            getattr(getattr(self.model, "config", None), "local_config", None),
            getattr(self.model, "config", None),
        ):
            if cfg is None:
                continue
            for name in ("_attn_implementation", "attn_implementation"):
                candidate = getattr(cfg, name, None)
                if isinstance(candidate, str) and candidate.strip():
                    attn_impl = candidate.strip().lower()
                    break
            if attn_impl:
                break
        self._use_dynamic_local_cache = attn_impl == "flash_attention_2"
        self._should_compile_local_transformer = not self._use_dynamic_local_cache
        self._compiled_local_transformer = None

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def is_finished(self) -> bool:
        return self._is_stopping is not None and bool(self._is_stopping.all())

    def _build_local_past_key_values(self):
        if self._use_dynamic_local_cache:
            return DynamicCache()
        return StaticCache(config=self.model.local_transformer.config, max_cache_len=self.channels)

    def _get_local_transformer_runner(self):
        if not self._should_compile_local_transformer:
            return self._generate_local_transformer_impl
        if self._compiled_local_transformer is None:
            self._compiled_local_transformer = torch.compile(self._generate_local_transformer_impl, fullgraph=True)
        return self._compiled_local_transformer

    def reset_generation_state(self, keep_cache: bool = True):
        if not keep_cache:
            self.past_key_values = None
            self.attention_mask = None
        # Keep the mask when reusing cache so it stays aligned with past_key_values.
        # This allows concatenation with the next turn prefill mask.
        self._generated_tokens = []
        self._is_stopping = None
        self._last_audio_tokens = None
        self._step_idx = 0

    def _normalize_input_ids(self, input_ids):
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.detach().cpu().numpy()
        if isinstance(input_ids, np.ndarray):
            if input_ids.ndim == 2:
                return [input_ids]
            if input_ids.ndim == 3:
                return [input_ids[i] for i in range(input_ids.shape[0])]
        if isinstance(input_ids, (list, tuple)):
            return [np.array(item) for item in input_ids]
        raise ValueError("input_ids must be a list/array/tensor of shape [T, C] or [B, T, C].")

    def _normalize_text_prefix(self, text_prefix_ids, batch_size: int) -> list[list[int]]:
        if text_prefix_ids is None:
            raise ValueError("text_prefix_ids must be provided for prefill.")
        if isinstance(text_prefix_ids, torch.Tensor):
            text_prefix_ids = text_prefix_ids.detach().cpu().tolist()
        if isinstance(text_prefix_ids, np.ndarray):
            text_prefix_ids = text_prefix_ids.tolist()
        if isinstance(text_prefix_ids, list):
            if len(text_prefix_ids) == 0:
                return [[] for _ in range(batch_size)]
            if isinstance(text_prefix_ids[0], (int, np.integer)):
                return [list(text_prefix_ids)]
            if len(text_prefix_ids) == 1 and batch_size > 1:
                return [list(text_prefix_ids[0]) for _ in range(batch_size)]
            if len(text_prefix_ids) != batch_size:
                raise ValueError(
                    f"text_prefix_ids batch size mismatch: got {len(text_prefix_ids)}, expected {batch_size}."
                )
            return [list(item) for item in text_prefix_ids]
        raise ValueError("text_prefix_ids must be list-like or tensor-like.")

    @torch.inference_mode()
    def prefill(
        self,
        input_ids,
        text_prefix_ids,
        max_prefill_len: Optional[int] = None,
        past_key_values=None,
        device: Optional[torch.device] = None,
        temperature: float = 0.8,
        top_p: float = 0.6,
        top_k: int = 30,
        do_sample: bool = True,
        repetition_penalty: Optional[float] = 1.1,
        repetition_window: Optional[int] = 50,
    ) -> torch.Tensor:
        if device is None:
            device = self.device

        if past_key_values is not None:
            self.past_key_values = past_key_values

        input_ids_list = self._normalize_input_ids(input_ids)
        batch_size = len(input_ids_list)
        text_prefix_list = self._normalize_text_prefix(text_prefix_ids, batch_size)

        concat_inputs_id_list = []
        for i in range(batch_size):
            prefix = text_prefix_list[i]
            if max_prefill_len is not None:
                prefix = prefix[:max_prefill_len]
            if len(prefix) == 0:
                raise ValueError("Prefill requires at least one text token.")

            text_seg = np.full((len(prefix), self.channels + 1), self.audio_channel_pad, dtype=np.int64)
            text_seg[:, 0] = np.array(prefix, dtype=np.int64)
            text_seg[len(prefix) - 1, 1] = self.audio_bos_token
            concat_inputs_id = np.concatenate([input_ids_list[i], text_seg], axis=0)
            concat_inputs_id_list.append(concat_inputs_id)

        attention_masks = [np.ones(ids.shape[0], dtype=np.bool_) for ids in concat_inputs_id_list]
        max_len = max(ids.shape[0] for ids in concat_inputs_id_list)
        padded_input_ids, padded_attns = [], []
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.text_pad_id

        for ids, attn in zip(concat_inputs_id_list, attention_masks):
            pad_len = max_len - ids.shape[0]
            input_pad = np.full((pad_len, self.channels + 1), self.audio_channel_pad, dtype=np.int64)
            input_pad[:, 0] = pad_token_id
            padded_input_ids.append(np.concatenate([input_pad, ids]))
            attn_pad = np.zeros(pad_len, dtype=np.bool_)
            padded_attns.append(np.concatenate([attn_pad, attn]))

        current_input_ids = torch.from_numpy(np.stack(padded_input_ids)).to(device)
        current_attention_mask = torch.from_numpy(np.stack(padded_attns)).to(device)

        # For multi-turn continuation, concatenate the cached mask and the current prefill mask.
        if self.attention_mask is not None and self.past_key_values is not None:
            current_attention_mask = torch.cat([self.attention_mask, current_attention_mask], dim=-1)

        outputs = self.model(
            input_ids=current_input_ids,
            attention_mask=current_attention_mask,
            past_key_values=self.past_key_values,
            use_cache=True,
            return_dict=True,
        )
        self.past_key_values = outputs.past_key_values
        self.attention_mask = current_attention_mask

        backbone_hidden_states = outputs.last_hidden_state[:, -1:, :]
        audio_tokens = self.generate_local_transformer(
            hidden_states=backbone_hidden_states,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
            generated_tokens=None,
            gen_step=0,
        )

        self._generated_tokens = [audio_tokens]
        self._last_audio_tokens = audio_tokens
        self._is_stopping = audio_tokens[:, 0] == self.audio_eos_token
        self._step_idx = 1
        return audio_tokens

    @torch.inference_mode()
    def step(
        self,
        text_token: Optional[Iterable[int] | torch.Tensor | int],
        temperature: float = 0.8,
        top_p: float = 0.6,
        top_k: int = 30,
        do_sample: bool = True,
        repetition_penalty: Optional[float] = 1.1,
        repetition_window: Optional[int] = 50,
    ) -> torch.Tensor:
        if self._last_audio_tokens is None or self.attention_mask is None:
            raise ValueError("You must call prefill() before step().")
        if self.is_finished:
            return self._last_audio_tokens

        batch_size = self._last_audio_tokens.shape[0]
        if text_token is None:
            text_tokens = [self.text_pad_id] * batch_size
        elif isinstance(text_token, torch.Tensor):
            text_tokens = text_token.detach().cpu().tolist()
        elif isinstance(text_token, (list, tuple, np.ndarray)):
            text_tokens = list(text_token)
        else:
            text_tokens = [int(text_token)]

        if len(text_tokens) != batch_size:
            raise ValueError(f"text_token batch size mismatch: got {len(text_tokens)}, expected {batch_size}.")

        device = self._last_audio_tokens.device
        text_t = torch.tensor(text_tokens, device=device, dtype=torch.long)
        step_ids = torch.cat([text_t[:, None, None], self._last_audio_tokens.unsqueeze(1)], dim=2)
        self.attention_mask = torch.cat([self.attention_mask, (~self._is_stopping).unsqueeze(-1)], dim=-1)

        outputs = self.model(
            input_ids=step_ids,
            attention_mask=self.attention_mask,
            past_key_values=self.past_key_values,
            use_cache=True,
            return_dict=True,
        )
        self.past_key_values = outputs.past_key_values
        backbone_hidden_states = outputs.last_hidden_state[:, -1:, :]

        history = torch.stack(self._generated_tokens, dim=1) if self._generated_tokens else None
        audio_tokens = self.generate_local_transformer(
            hidden_states=backbone_hidden_states,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
            generated_tokens=history,
            gen_step=self._step_idx,
        )

        self._generated_tokens.append(audio_tokens)
        self._last_audio_tokens = audio_tokens
        self._is_stopping |= audio_tokens[:, 0] == self.audio_eos_token
        self._step_idx += 1
        return audio_tokens

    @torch.inference_mode()
    def finish(
        self,
        max_steps: Optional[int] = None,
        temperature: float = 0.8,
        top_p: float = 0.6,
        top_k: int = 30,
        do_sample: bool = True,
        repetition_penalty: Optional[float] = 1.1,
        repetition_window: Optional[int] = 50,
    ) -> list[torch.Tensor]:
        outputs = []
        steps_left = max_steps if max_steps is not None else self.max_length
        while steps_left > 0 and not self.is_finished:
            outputs.append(
                self.step(
                    text_token=None,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty,
                    repetition_window=repetition_window,
                )
            )
            steps_left -= 1
        return outputs

    def generate_local_transformer(
        self,
        hidden_states: torch.Tensor,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
        repetition_penalty: Optional[float],
        repetition_window: Optional[int],
        generated_tokens: Optional[torch.Tensor],
        gen_step: int,
    ) -> torch.Tensor:
        runner = self._get_local_transformer_runner()
        return runner(
            hidden_states=hidden_states,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
            generated_tokens=generated_tokens,
            gen_step=gen_step,
        )

    def _generate_local_transformer_impl(
        self,
        hidden_states: torch.Tensor,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
        repetition_penalty: Optional[float],
        repetition_window: Optional[int],
        generated_tokens: Optional[torch.Tensor],
        gen_step: int,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        device = hidden_states.device
        local_inputs = hidden_states.reshape(-1, 1, self.model.config.local_config.hidden_size)
        output_token = torch.empty(batch_size, self.channels, dtype=torch.long, device=device)

        past_key_values = self._build_local_past_key_values()
        local_token = None

        cache_pos_t = torch.zeros(1, dtype=torch.long, device=device)

        for i in range(self.channels):
            cache_pos_t.fill_(i)

            local_outputs = self.model.local_transformer(
                input_ids=local_token,
                inputs_embeds=local_inputs,
                past_key_values=past_key_values,
                cache_position=cache_pos_t,
                codebook_idx=i,
                use_cache=True,
                logits_to_keep=1,
            )
            logits = local_outputs.logits

            if repetition_penalty and repetition_penalty != 1.0 and generated_tokens is not None:
                logits = self.apply_repetition_penalty(
                    scores=logits,
                    history_tokens=generated_tokens[:, :gen_step, i],
                    penalty=float(repetition_penalty),
                    repetition_window=repetition_window,
                )

            local_token = self.sample_token(
                logits=logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
            )
            output_token[:, i] = local_token.squeeze(-1)

            if i == 0:
                local_inputs = None
        return output_token

    def apply_repetition_penalty(
        self,
        scores: torch.Tensor,
        history_tokens: torch.Tensor,
        penalty: float = 1.1,
        repetition_window: Optional[int] = None,
    ):
        scores_ = scores[:, 0, :]
        ht = history_tokens

        if repetition_window is not None and repetition_window > 0:
            ht = ht[:, -repetition_window:]

        cur = scores_.gather(1, ht)
        new = torch.where(cur < 0, cur * penalty, cur / penalty)
        scores_.scatter_(1, ht, new)
        return scores_

    def sample_token(self, logits, temperature, top_p=0.6, top_k=30, do_sample=True):
        if not do_sample or temperature == 0:
            return torch.argmax(logits, dim=-1)
        logits = logits / temperature
        original_shape = logits.shape
        vocab_size = original_shape[-1]
        reshaped_logits = logits.reshape(-1, vocab_size)

        if top_k is not None:
            reshaped_logits = self.apply_top_k(reshaped_logits, top_k)

        if top_p is not None:
            reshaped_logits = self.apply_top_p(reshaped_logits, top_p)

        probs = F.softmax(reshaped_logits, dim=-1)
        next_tokens_flat = torch.multinomial(probs, num_samples=1)

        output_shape = original_shape[:-1]
        return next_tokens_flat.view(output_shape)

    def apply_top_k(self, logits, top_k, filter_value=float("-inf"), min_tokens_to_keep: int = 1):
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")
        batch_size, vocab_size = logits.shape
        top_k = max(top_k, min_tokens_to_keep)
        top_k = min(top_k, vocab_size)
        indices_to_remove = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        return logits.masked_fill(logits < indices_to_remove, filter_value)

    def apply_top_p(self, logits, top_p, filter_value=float("-inf"), min_tokens_to_keep: int = 1):
        top_p = float(top_p)
        if top_p < 0 or top_p > 1.0:
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")

        sorted_logits, sorted_indices = torch.sort(logits, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
        sorted_indices_to_remove[..., -min_tokens_to_keep:] = 0
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter(1, sorted_indices, sorted_indices_to_remove)
        logits_processed = logits.masked_fill(indices_to_remove, filter_value)
        return logits_processed


@requires(backends=("torch",))
class MossTTSRealtimeStreamingSession:
    """Manage text-to-audio streaming for a single conversation."""

    _split_pattern = re.compile(
        r"[。！？!?\.\u2026]\s*"  # sentence boundaries: 。！？ ! ? . …
        r"|[,，;；:：\u2014\u2013\-]\s*"  # short pauses: , ， ; ； : ： — – -
        r"|\)\s*|\]\s*"  # closing brackets: ) ]
        r"|\n"
    )

    def __init__(
        self,
        inferencer: MossTTSRealtimeInference,
        processor,
        codec=None,
        codec_sample_rate: int = 24000,
        codec_encode_kwargs: Optional[dict] = None,
        prefill_text_len: int = 12,
        text_buffer_size: int = 32,
        min_text_chunk_chars: int = 8,
        temperature: float = 0.8,
        top_p: float = 0.6,
        top_k: int = 30,
        do_sample: bool = True,
        repetition_penalty: Optional[float] = 1.1,
        repetition_window: Optional[int] = 50,
    ):
        self.inferencer = inferencer
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.codec = codec
        self.codec_sample_rate = codec_sample_rate
        self.codec_encode_kwargs = codec_encode_kwargs or {}

        self.prefill_text_len = prefill_text_len
        self.text_buffer_size = text_buffer_size
        self.min_text_chunk_chars = min_text_chunk_chars

        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.do_sample = do_sample
        self.repetition_penalty = repetition_penalty
        self.repetition_window = repetition_window

        self._voice_prompt_tokens = None
        self._turn_input_ids = None
        self._turn_idx = 0

        self._text_cache = ""
        self._pending_tokens: list[int] = []
        self._prefilled = False
        self._text_ended = False

    def set_voice_prompt_tokens(self, audio_tokens: np.ndarray):
        self._voice_prompt_tokens = audio_tokens

    def set_voice_prompt(self, audio, sample_rate: Optional[int] = None):
        """Set voice prompt from either audio tokens or waveform.
        If `audio` is a 2D array whose shape matches the codebook channels, it is
        treated as audio tokens. Otherwise a codec is required to encode waveform
        prompts into tokens.
        """
        if isinstance(audio, np.ndarray) and audio.ndim == 2:
            if self.processor.channels in audio.shape:
                self._voice_prompt_tokens = audio
                return
        if isinstance(audio, torch.Tensor) and audio.dim() == 2:
            if self.processor.channels in audio.shape:
                self._voice_prompt_tokens = audio.detach().cpu().numpy()
                return

        if self.codec is None:
            raise ValueError("codec is required to encode waveform prompts.")

        waveform = audio
        if isinstance(audio, (str, bytes)):
            requires_backends(self, ["torchaudio"])
            wav, sr = torchaudio.load(audio)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            waveform = wav.squeeze(0)
            sample_rate = sr

        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)
        if not isinstance(waveform, torch.Tensor):
            raise ValueError("Unsupported audio type for voice prompt.")

        if sample_rate is not None and sample_rate != self.codec_sample_rate:
            requires_backends(self, ["torchaudio"])
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.codec_sample_rate)

        waveform = waveform.to(self.inferencer.device)
        encode_out = self.codec.encode([waveform], **self.codec_encode_kwargs)
        if isinstance(encode_out, dict):
            if "codes_list" in encode_out:
                tokens = encode_out["codes_list"][0]
            elif "audio_codes" in encode_out:
                tokens = encode_out["audio_codes"][0]
            else:
                raise ValueError("codec.encode output missing audio codes.")
        else:
            tokens = encode_out
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.detach().cpu().numpy()
        self._voice_prompt_tokens = tokens

    def clear_voice_prompt(self):
        self._voice_prompt_tokens = None

    def reset_turn(
        self,
        user_text: Optional[str] = None,
        user_audio_tokens: Optional[np.ndarray] = None,
        input_ids: Optional[np.ndarray] = None,
        include_system_prompt: Optional[bool] = None,
        reset_cache: bool = False,
    ):
        if include_system_prompt is None:
            include_system_prompt = self._turn_idx == 0

        if input_ids is None:
            if user_text is None or user_audio_tokens is None:
                raise ValueError("user_text and user_audio_tokens are required when input_ids is not provided.")
            user_prompt = self.processor.make_user_prompt(user_text, user_audio_tokens)
            if include_system_prompt:
                system_prompt = self.processor.make_ensemble(self._voice_prompt_tokens)
                input_ids = np.concatenate([system_prompt, user_prompt], axis=0)
            else:
                input_ids = user_prompt

        self._turn_input_ids = input_ids
        self._turn_idx += 1

        self._text_cache = ""
        self._pending_tokens = []
        self._prefilled = False
        self._text_ended = False

        self.inferencer.reset_generation_state(keep_cache=not reset_cache)

    def push_text_tokens(self, tokens: Iterable[int]) -> list[torch.Tensor]:
        self._pending_tokens.extend([int(t) for t in tokens])
        return self._drain_pending_tokens()

    def push_text(self, text_fragment: str) -> list[torch.Tensor]:
        self._text_cache += text_fragment
        segments = self._extract_text_segments(force=False)
        for segment in segments:
            self._pending_tokens.extend(self._tokenize(segment))
        return self._drain_pending_tokens()

    def end_text(self) -> list[torch.Tensor]:
        self._text_ended = True
        if self._text_cache:
            self._pending_tokens.extend(self._tokenize(self._text_cache))
            self._text_cache = ""
        return self._drain_pending_tokens()

    def drain(self, max_steps: Optional[int] = None) -> list[torch.Tensor]:
        if not self._prefilled:
            return []
        return self.inferencer.finish(
            max_steps=max_steps,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            do_sample=self.do_sample,
            repetition_penalty=self.repetition_penalty,
            repetition_window=self.repetition_window,
        )

    def _tokenize(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _extract_text_segments(self, force: bool) -> list[str]:
        segments = []
        if force:
            if self._text_cache:
                segments.append(self._text_cache)
                self._text_cache = ""
            return segments

        while self._text_cache:
            cut_idx = None
            if len(self._text_cache) >= self.min_text_chunk_chars:
                matches = list(self._split_pattern.finditer(self._text_cache))
                for match in matches:
                    if match.end() >= self.min_text_chunk_chars:
                        cut_idx = match.end()
                        break
            if cut_idx is None and len(self._text_cache) >= self.text_buffer_size:
                whitespace_idx = self._text_cache.rfind(" ")
                if whitespace_idx != -1:
                    cut_idx = whitespace_idx + 1
            if cut_idx is None:
                break
            segments.append(self._text_cache[:cut_idx])
            self._text_cache = self._text_cache[cut_idx:]
        return segments

    def _prefill_if_needed(self) -> list[torch.Tensor]:
        if self._prefilled:
            return []
        if not self._pending_tokens and not self._text_ended:
            return []
        if len(self._pending_tokens) < self.prefill_text_len and not self._text_ended:
            return []
        if self._turn_input_ids is None:
            raise ValueError("reset_turn must be called before streaming text.")

        if self._text_ended:
            prefill_len = len(self._pending_tokens)
        else:
            prefill_len = min(len(self._pending_tokens), self.prefill_text_len)

        if prefill_len == 0:
            return []

        prefix_tokens = [self._pending_tokens.pop(0) for _ in range(prefill_len)]
        audio_tokens = self.inferencer.prefill(
            input_ids=[self._turn_input_ids],
            text_prefix_ids=[prefix_tokens],
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            do_sample=self.do_sample,
            repetition_penalty=None,
            repetition_window=self.repetition_window,
        )
        self._prefilled = True
        return [audio_tokens]

    def _drain_pending_tokens(self) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        outputs.extend(self._prefill_if_needed())
        if not self._prefilled:
            return outputs

        while self._pending_tokens and not self.inferencer.is_finished:
            token = self._pending_tokens.pop(0)
            outputs.append(
                self.inferencer.step(
                    token,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    do_sample=self.do_sample,
                    repetition_penalty=self.repetition_penalty,
                    repetition_window=self.repetition_window,
                )
            )
        return outputs


@requires(backends=("torch",))
class AudioStreamDecoder:
    """Decode audio tokens into waveform chunks with optional crossfade."""

    def __init__(
        self,
        codec,
        chunk_frames: int = 40,
        overlap_frames: int = 4,
        decode_kwargs: Optional[dict] = None,
        device: Optional[torch.device] = None,
    ):
        self.codec = codec
        self.chunk_frames = chunk_frames
        self.overlap_frames = overlap_frames
        self.decode_kwargs = decode_kwargs or {}
        self.device = device

        self._buffer: list[torch.Tensor] = []
        self._buffer_len = 0
        self._prev_tail: Optional[torch.Tensor] = None

    def push_tokens(self, audio_tokens: np.ndarray | torch.Tensor):
        if isinstance(audio_tokens, np.ndarray):
            audio_tokens = torch.from_numpy(audio_tokens)
        if audio_tokens.dim() != 2:
            raise ValueError(f"Expected [T, C] audio tokens, got {tuple(audio_tokens.shape)}")
        self._buffer.append(audio_tokens)
        self._buffer_len += audio_tokens.shape[0]

    def audio_chunks(self) -> Iterable[torch.Tensor]:
        while self._buffer_len >= self.chunk_frames:
            chunk_tokens = self._consume_frames(self.chunk_frames)
            wav = self._decode(chunk_tokens, chunk_duration=0.32)
            yield self._apply_crossfade(wav)

    def flush(self) -> Optional[torch.Tensor]:
        if self._buffer_len == 0:
            return None
        chunk_tokens = self._consume_frames(self._buffer_len)
        wav = self._decode(chunk_tokens)
        return self._apply_crossfade(wav, final_chunk=True)

    def _consume_frames(self, num_frames: int) -> torch.Tensor:
        frames = []
        remaining = num_frames
        while remaining > 0 and self._buffer:
            head = self._buffer[0]
            if head.shape[0] <= remaining:
                frames.append(head)
                remaining -= head.shape[0]
                self._buffer.pop(0)
            else:
                frames.append(head[:remaining])
                self._buffer[0] = head[remaining:]
                remaining = 0
        self._buffer_len -= num_frames - remaining
        return torch.cat(frames, dim=0)

    def _decode(self, tokens: torch.Tensor, chunk_duration: float = 0.32) -> torch.Tensor:
        device = self.device
        if device is None:
            if hasattr(self.codec, "device"):
                device = self.codec.device
            else:
                try:
                    device = next(self.codec.parameters()).device
                except Exception:
                    device = None
        if device is not None:
            tokens = tokens.to(device)
        tokens_t = tokens.permute(1, 0)
        # allow callers to override decode settings (e.g. chunk_duration=-1 to disable internal streaming)
        decode_kwargs = dict(self.decode_kwargs) if self.decode_kwargs else {}
        if "chunk_duration" in decode_kwargs:
            override = decode_kwargs.pop("chunk_duration")
            if override is None:
                chunk_duration_arg = None
            else:
                try:
                    override_f = float(override)
                except Exception:
                    override_f = None
                chunk_duration_arg = None if override_f is None or override_f <= 0 else override_f
        else:
            chunk_duration_arg = chunk_duration

        decoded = self.codec.decode(tokens_t, chunk_duration=chunk_duration_arg, **decode_kwargs)
        if isinstance(decoded, dict):
            wav = decoded["audio"][0]
        else:
            wav = decoded
        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav)
        if wav.dim() > 1:
            wav = wav.squeeze(0)
        return wav

    def _apply_crossfade(self, wav: torch.Tensor, final_chunk: bool = False) -> torch.Tensor:
        if self.overlap_frames <= 0:
            return wav
        if self._prev_tail is None:
            self._prev_tail = wav[-self._overlap_samples(wav) :].clone() if not final_chunk else None
            return wav

        overlap = self._overlap_samples(wav)
        if overlap == 0:
            return wav

        prev_tail = self._prev_tail
        if prev_tail.numel() < overlap:
            overlap = prev_tail.numel()
        if overlap == 0:
            return wav

        fade_out = torch.linspace(1.0, 0.0, overlap, device=wav.device)
        fade_in = 1.0 - fade_out
        cross = prev_tail[-overlap:] * fade_out + wav[:overlap] * fade_in
        merged = torch.cat([prev_tail[:-overlap], cross, wav[overlap:]], dim=-1)

        self._prev_tail = None if final_chunk else wav[-overlap:].clone()
        return merged

    def _overlap_samples(self, wav: torch.Tensor) -> int:
        if self.chunk_frames <= 0:
            return 0
        return int(wav.numel() * (self.overlap_frames / self.chunk_frames))


class TextDeltaTokenizer:
    """
    Convert LLM streaming text (delta) into “incremental token IDs”.
    Notes:
    - The input is a delta that is progressively appended to the same string
    (consistent with the common delta output behavior in vLLM).
    - Each time, re-encode the *full text* with the tokenizer, then take only
    the newly added token IDs.
    - This guarantees that tokenization is consistent with the final complete
    text, avoiding boundary mismatches caused by tokenizing partial segments.
    """

    def __init__(self, tokenizer, *, hold_back: int = 3):
        self.tokenizer = tokenizer
        self.hold_back = max(0, int(hold_back))
        self._text = ""
        self._all_ids: list[int] = []
        self._emitted_count: int = 0

    @property
    def text(self) -> str:
        return self._text

    @property
    def token_ids(self) -> list[int]:
        return list(self._all_ids)

    def push_delta(self, delta: str) -> list[int]:
        """Append a text delta and return newly stable token ids (may be empty)."""
        if not delta:
            return []
        self._text += str(delta)
        self._all_ids = self.tokenizer.encode(self._text, add_special_tokens=False)
        # Keep the tail un-emitted because the latest tokens can still change.
        stable_count = max(self._emitted_count, len(self._all_ids) - self.hold_back)
        new_ids = self._all_ids[self._emitted_count : stable_count]
        self._emitted_count = stable_count
        return new_ids

    def flush(self) -> list[int]:
        """Emit all remaining token ids at end of stream."""
        self._all_ids = self.tokenizer.encode(self._text, add_special_tokens=False)
        remaining = self._all_ids[self._emitted_count :]
        self._emitted_count = len(self._all_ids)
        return remaining


def _sanitize_audio_tokens(
    tokens: torch.Tensor,
    *,
    codebook_size: int,
    audio_eos_token: int,
) -> tuple[torch.Tensor, bool]:
    """Trim rows after EOS/invalid tokens and return whether decoding should stop."""
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
        return tokens[:stop_idx], True
    return tokens, False


def _maybe_codec_streaming(codec, *, batch_size: int):
    if codec is None or not hasattr(codec, "streaming"):
        return contextlib.nullcontext()
    return codec.streaming(batch_size=batch_size)


@requires(backends=("torch",))
class MossTTSRealtimeTextStreamBridge:
    """
    Bridge: external LLM streaming text (delta) -> TTS streaming audio chunks.
    Usage overview:
    - First configure `MossTTSRealtimeStreamingSession` (especially `prefill_text_len=12`).
    - Provide an `AudioStreamDecoder`, then continuously feed the LLM delta text via
    `push_text_delta()`.
    - Once the accumulated token count reaches `prefill_text_len`, the session will
    start generating audio tokens; the bridge will immediately decode them into WAV
    chunks and yield them.
    """

    def __init__(
        self,
        session: MossTTSRealtimeStreamingSession,
        decoder: AudioStreamDecoder,
        *,
        codebook_size: Optional[int] = None,
        audio_eos_token: Optional[int] = None,
        batch_size: int = 1,
    ):
        self.session = session
        self.decoder = decoder
        self.batch_size = int(batch_size)

        if codebook_size is None:
            codebook_size = int(getattr(getattr(session, "codec", None), "codebook_size", 1024))
        if audio_eos_token is None:
            audio_eos_token = int(getattr(session.inferencer, "audio_eos_token", 1026))

        self.codebook_size = int(codebook_size)
        self.audio_eos_token = int(audio_eos_token)

    def push_text_delta(self, delta: str) -> Iterator[torch.Tensor]:
        """
        Push a chunk of incremental text output from the LLM and return newly generated WAV chunks.
        Internally, this directly calls `session.push_text()`, which segments the text
        based on punctuation/length and then tokenizes the *entire segment* at once,
        avoiding the prefix instability issues of incremental BPE tokenization.
        """
        audio_frames = self.session.push_text(delta)
        yield from self._decode_audio_frames(audio_frames)

    def push_text_tokens(self, token_ids: Sequence[int]) -> Iterator[torch.Tensor]:
        """Push token ids directly (for sources that stream token ids)."""
        if not token_ids:
            return
        audio_frames = self.session.push_text_tokens(token_ids)
        yield from self._decode_audio_frames(audio_frames)

    def finish(self, *, drain_step: int = 1) -> Iterator[torch.Tensor]:
        """Mark text stream end and emit all remaining audio chunks (including flush)."""
        audio_frames = self.session.end_text()
        yield from self._decode_audio_frames(audio_frames)

        while True:
            more_frames = self.session.drain(max_steps=drain_step)
            if not more_frames:
                break
            yield from self._decode_audio_frames(more_frames)
            if self.session.inferencer.is_finished:
                break

        final = self.decoder.flush()
        if final is not None and final.numel() > 0:
            yield final.detach().cpu()

    def stream_from_text_deltas(self, deltas: Iterable[str], *, drain_step: int = 1) -> Iterator[torch.Tensor]:
        """Consume a full delta iterator and continuously yield waveform chunks."""
        with _maybe_codec_streaming(getattr(self.session, "codec", None), batch_size=self.batch_size):
            for delta in deltas:
                yield from self.push_text_delta(delta)
            yield from self.finish(drain_step=drain_step)

    def _decode_audio_frames(self, audio_frames: list[torch.Tensor]) -> Iterator[torch.Tensor]:
        for frame in audio_frames:
            tokens = frame
            if tokens.dim() == 3:
                tokens = tokens[0]
            if tokens.dim() != 2:
                raise ValueError(f"Expected [B, C] or [1, C] audio tokens, got {tuple(tokens.shape)}")
            if tokens.shape[0] != 1:
                raise ValueError(
                    f"This bridge currently supports batch_size=1 for decoding, got batch={tokens.shape[0]}."
                )

            tokens, stop = _sanitize_audio_tokens(
                tokens,
                codebook_size=self.codebook_size,
                audio_eos_token=self.audio_eos_token,
            )
            if tokens.numel() == 0:
                if stop:
                    break
                continue

            self.decoder.push_tokens(tokens.detach())
            for wav in self.decoder.audio_chunks():
                if wav.numel() == 0:
                    continue
                yield wav.detach().cpu()
            if stop:
                break


__all__ = [
    "AudioStreamDecoder",
    "MossTTSRealtimeInference",
    "MossTTSRealtimeStreamingSession",
    "MossTTSRealtimeTextStreamBridge",
    "TextDeltaTokenizer",
]
