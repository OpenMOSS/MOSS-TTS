# Copyright 2026 OpenMOSS and the HuggingFace Inc. team. All rights reserved.
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
"""Processing utilities for MossTTSRealtime."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from transformers.processing_utils import ProcessorMixin


class MossTTSRealtimeProcessor(ProcessorMixin):
    """Builds MossTTSRealtime prompt inputs with text and audio codebooks.
    This processor focuses on preparing the mixed text/audio token layout expected by MossTTSRealtime.
    It does not perform audio encoding/decoding by itself.
    """

    attributes = ["tokenizer"]
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        tokenizer,
        audio_pad_token: str = "<|audio_pad|>",
        text_pad_token: str = "<|text_pad|>",
        tts_system_prompt: Optional[str] = None,
        channels: int = 16,
        audio_channel_pad: int = 1024,
        audio_bos_token: int = 1025,
        audio_eos_token: int = 1026,
        delay_tokens_len: int = 12,
    ):
        super().__init__(tokenizer=tokenizer)
        self.audio_pad_token = audio_pad_token
        self.text_pad_token = text_pad_token
        self.channels = channels
        self.audio_channel_pad = audio_channel_pad
        self.audio_bos_token = audio_bos_token
        self.audio_eos_token = audio_eos_token
        self.delay_tokens_len = delay_tokens_len

        self.audio_pad_token_id = self._convert_token_to_id(audio_pad_token)
        self.text_pad_token_id = self._convert_token_to_id(text_pad_token)

        if tts_system_prompt is None:
            tts_system_prompt = (
                "<|im_start|>system\n"
                "You are a highly expressive text-to-speech (TTS) engine developed by Mosi Intelligence. \n"
                "You possess natural language understanding, emotional modeling, and multi-style speech generation "
                "capabilities, allowing you to generate the corresponding speech based on the text given in the assistant."
                "<|im_end|>\n"
            )
        self.tts_system_prompt = tts_system_prompt

    def _convert_token_to_id(self, token: str) -> int:
        if hasattr(self.tokenizer, "convert_tokens_to_ids"):
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id is not None and token_id != self.tokenizer.unk_token_id:
                return int(token_id)
        token_ids = self.tokenizer.encode(token, add_special_tokens=False)
        if not token_ids:
            raise ValueError(f"Token '{token}' could not be converted to an id.")
        if len(token_ids) != 1:
            raise ValueError(f"Token '{token}' maps to multiple ids: {token_ids}")
        return int(token_ids[0])

    def make_voice_clone_prompt(self, prompt_audio_tokens_len: int) -> str:
        padded_audio_prompt = f"{self.audio_pad_token * prompt_audio_tokens_len}"
        voice_clone = (
            "<|im_start|>context\n"
            "The assistant section should be synthesized using the following voice timbre:"
            f"{padded_audio_prompt}"
            "<|im_end|>\n"
        )
        return voice_clone

    def _normalize_audio_tokens(self, audio_tokens: np.ndarray | Iterable) -> np.ndarray:
        tokens = np.array(audio_tokens)
        if tokens.ndim != 2:
            raise ValueError(f"Expected 2D audio tokens, got shape {tokens.shape}")
        # Accept [channels, T] or [T, channels], and slice to expected channels if needed.
        if tokens.shape[0] == self.channels:
            tokens = tokens.T
        elif tokens.shape[1] == self.channels:
            tokens = tokens
        elif tokens.shape[0] > self.channels and tokens.shape[1] != self.channels:
            tokens = tokens[: self.channels, :].T
        elif tokens.shape[1] > self.channels and tokens.shape[0] != self.channels:
            tokens = tokens[:, : self.channels]
        if tokens.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got shape {tokens.shape}")
        return tokens

    def make_ensemble(self, prompt_audio_tokens: Optional[np.ndarray] = None) -> np.ndarray:
        if prompt_audio_tokens is not None:
            prompt_audio_tokens = self._normalize_audio_tokens(prompt_audio_tokens)
            prompt_audio_tokens = prompt_audio_tokens[:, : self.channels]
            system_prompt_text = f"{self.tts_system_prompt}" + f"{self.make_voice_clone_prompt(prompt_audio_tokens.shape[0])}"
        else:
            system_prompt_text = f"{self.tts_system_prompt}"

        system_prompt_tokens = self.tokenizer(system_prompt_text)["input_ids"]
        system_prompt_tokens_full = np.full(
            shape=(len(system_prompt_tokens), self.channels + 1), fill_value=self.audio_channel_pad, dtype=np.int64
        )
        system_prompt_tokens_full[:, 0] = system_prompt_tokens

        if prompt_audio_tokens is not None:
            system_prompt_tokens = np.array(system_prompt_tokens)
            indices = np.where(system_prompt_tokens == self.audio_pad_token_id)[0]
            if indices.size == 0:
                raise ValueError("No <|audio_pad|> tokens found in the system prompt.")
            prompt_audio_start_pos, prompt_audio_end_pos = indices[0], indices[-1]
            system_prompt_tokens_full[prompt_audio_start_pos : prompt_audio_end_pos + 1, 1:] = prompt_audio_tokens

        return system_prompt_tokens_full

    def make_user_prompt(self, text: str, audio_tokens: np.ndarray) -> np.ndarray:
        prefill_temp = "<|im_end|>\n<|im_start|>user\n"
        text_tokens = self.tokenizer(text)["input_ids"]
        text_start_pos = len(self.tokenizer.encode(prefill_temp))
        token = self._normalize_audio_tokens(audio_tokens)

        text_len = len(text_tokens)
        audio_len = token.shape[0]

        if text_len >= self.delay_tokens_len:
            padded_text_len = audio_len + self.delay_tokens_len - text_len + 1
            cur_input_id_ch1 = prefill_temp + text + "<|text_pad|>" * padded_text_len
            assistant_tokens_ch1 = self.tokenizer(cur_input_id_ch1)["input_ids"]
            cur_input_id = np.full(
                shape=(len(assistant_tokens_ch1), self.channels + 1),
                fill_value=self.audio_channel_pad,
                dtype=np.int64,
            )
            cur_input_id[:, 0] = assistant_tokens_ch1
            cur_input_id[
                text_start_pos + self.delay_tokens_len : text_start_pos + self.delay_tokens_len + audio_len, 1:
            ] = token
            cur_input_id[text_start_pos + self.delay_tokens_len - 1, 1] = self.audio_bos_token
            cur_input_id[text_start_pos + self.delay_tokens_len + audio_len, 1] = self.audio_eos_token
        else:
            padded_text_len = audio_len + 1
            cur_input_id_ch1 = prefill_temp + text + "<|text_pad|>" * padded_text_len
            assistant_tokens_ch1 = self.tokenizer(cur_input_id_ch1)["input_ids"]
            cur_input_id = np.full(
                shape=(len(assistant_tokens_ch1), self.channels + 1),
                fill_value=self.audio_channel_pad,
                dtype=np.int64,
            )
            cur_input_id[:, 0] = assistant_tokens_ch1
            cur_input_id[-(audio_len + 1) : -1, 1:] = token
            cur_input_id[-(audio_len + 2), 1] = self.audio_bos_token
            cur_input_id[-1, 1] = self.audio_eos_token

        begin_of_response = self.tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n")
        begin_of_response_full = np.full(
            shape=(len(begin_of_response), self.channels + 1), fill_value=self.audio_channel_pad, dtype=np.int64
        )
        begin_of_response_full[:, 0] = begin_of_response

        input_ids = np.concatenate([cur_input_id, begin_of_response_full], axis=0)
        return input_ids


__all__ = ["MossTTSRealtimeProcessor"]