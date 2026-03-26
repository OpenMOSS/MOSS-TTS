# Copyright 2026 OpenMOSS team. All rights reserved.
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
"""Dataset for MossTTSRealtime supervised fine-tuning.

Record format (single-turn and multi-turn share the same structure)::

    {
        "conversations": [
            {"role": "user",      "text": "...", "audio_codes": [[...]]},
            {"role": "assistant", "text": "...", "audio_codes": [[...]]},
            ...
        ],
        "ref_audio_codes": [[...]]   # optional, for voice cloning
    }

Labels are set only for assistant turns.  BOS (1025) is never in labels;
EOS (1026) is kept as a valid label on channel 1.
If the last turn is a user turn it is skipped (no assistant response to train).
"""
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

AUDIO_CHANNELS = 16


def _normalize_audio_codes(value, field_name):
    codes = np.asarray(value, dtype=np.int64)
    if codes.ndim != 2:
        raise ValueError(
            f"`{field_name}` must be 2-D (T, n_vq) or (n_vq, T), "
            f"got shape {codes.shape}."
        )
    if codes.shape[0] == AUDIO_CHANNELS and codes.shape[1] != AUDIO_CHANNELS:
        codes = codes.T
    return codes


def _pad_or_trim_channels(codes, pad_value):
    n = codes.shape[1]
    if n == AUDIO_CHANNELS:
        return codes
    if n < AUDIO_CHANNELS:
        pad = np.full((codes.shape[0], AUDIO_CHANNELS - n), pad_value, dtype=np.int64)
        return np.concatenate([codes, pad], axis=1)
    return codes[:, :AUDIO_CHANNELS]


class MossTTSRealtimeSFTDataset(Dataset):
    """Teacher-forcing dataset for MossTTSRealtime supervised fine-tuning.

    All records use the ``conversations`` format.  Labels are only set for
    assistant turns; BOS is masked to -100.
    """

    def __init__(self, records, processor, n_vq=None):
        self.records = list(records)
        self.processor = processor
        self.n_vq = n_vq
        self.channels = processor.channels
        self.audio_channel_pad = processor.audio_channel_pad
        self.audio_bos_token = processor.audio_bos_token
        self.audio_eos_token = processor.audio_eos_token
        self.delay_tokens_len = processor.delay_tokens_len
        self.tokenizer = processor.tokenizer

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self._pack_record(self.records[index])

    # ------------------------------------------------------------------
    # Audio code helpers
    # ------------------------------------------------------------------

    def _norm_token(self, codes):
        codes = np.asarray(codes, dtype=np.int64)
        if codes.ndim != 2:
            raise ValueError(f"audio_codes must be 2-D, got shape {codes.shape}.")
        if codes.shape[0] == self.channels and codes.shape[1] != self.channels:
            codes = codes.T
        return _pad_or_trim_channels(codes, self.audio_channel_pad)

    def _get_ref_audio_codes(self, record):
        if record.get("ref_audio_codes") is not None:
            codes = _normalize_audio_codes(record["ref_audio_codes"], "ref_audio_codes")
            return _pad_or_trim_channels(codes, self.audio_channel_pad)
        if record.get("ref_wav") is not None:
            raise ValueError(
                "Record has `ref_wav` but no `ref_audio_codes`. "
                "Run prepare_data.py first."
            )
        return None

    # ------------------------------------------------------------------
    # Build one turn's input_ids + labels
    # ------------------------------------------------------------------

    def _build_turn(self, text, token, prefill_temp, is_assistant):
        """Build input_ids [T_turn, 17] and labels [T_turn, 17] for one turn.

        Labels are set only when ``is_assistant`` is True.  BOS (1025) is
        never included in labels; EOS (1026) is included on channel 1 only.
        """
        text_tokens = self.tokenizer(text)["input_ids"]
        text_start_pos = len(self.tokenizer.encode(prefill_temp))
        text_len = len(text_tokens)
        audio_len = token.shape[0]

        if text_len >= self.delay_tokens_len:
            padded_text_len = audio_len + self.delay_tokens_len - text_len + 1
            cur_input_id_ch1 = prefill_temp + text + "<|text_pad|>" * padded_text_len
            ch1_ids = self.tokenizer(cur_input_id_ch1)["input_ids"]

            cur_input_id = np.full(
                (len(ch1_ids), self.channels + 1),
                fill_value=self.audio_channel_pad, dtype=np.int64,
            )
            cur_input_id[:, 0] = ch1_ids

            audio_start = text_start_pos + self.delay_tokens_len
            cur_input_id[audio_start: audio_start + audio_len, 1:] = token
            cur_input_id[audio_start - 1, 1] = self.audio_bos_token
            cur_input_id[audio_start + audio_len, 1] = self.audio_eos_token

            cur_label = np.full_like(cur_input_id, -100)
            if is_assistant:
                cur_label[audio_start: audio_start + audio_len, 1:] = token
                cur_label[audio_start + audio_len, 1] = self.audio_eos_token
        else:
            padded_text_len = audio_len + 1
            cur_input_id_ch1 = prefill_temp + text + "<|text_pad|>" * padded_text_len
            ch1_ids = self.tokenizer(cur_input_id_ch1)["input_ids"]

            cur_input_id = np.full(
                (len(ch1_ids), self.channels + 1),
                fill_value=self.audio_channel_pad, dtype=np.int64,
            )
            cur_input_id[:, 0] = ch1_ids

            cur_input_id[-(audio_len + 1): -1, 1:] = token
            cur_input_id[-(audio_len + 2), 1] = self.audio_bos_token
            cur_input_id[-1, 1] = self.audio_eos_token

            cur_label = np.full_like(cur_input_id, -100)
            if is_assistant:
                cur_label[-(audio_len + 1): -1, 1:] = token
                cur_label[-1, 1] = self.audio_eos_token

        return cur_input_id, cur_label

    # ------------------------------------------------------------------
    # Record packing
    # ------------------------------------------------------------------

    def _pack_record(self, record):
        conversations = record["conversations"]
        ref_audio_codes = self._get_ref_audio_codes(record)

        ensemble = self.processor.make_ensemble(ref_audio_codes)
        sys_label = np.full_like(ensemble, -100)

        inputs_list = [ensemble]
        labels_list = [sys_label]

        for i, turn in enumerate(conversations):
            role = turn["role"]
            text = turn["text"]
            is_last_turn = (i == len(conversations) - 1)

            if role == "user" and is_last_turn:
                break

            audio_codes = self._norm_token(turn["audio_codes"])

            if i == 0:
                prefill_temp = "<|im_start|>" + role + "\n"
            else:
                prefill_temp = "<|im_end|>\n<|im_start|>" + role + "\n"

            is_assistant = (role == "assistant")
            turn_input, turn_label = self._build_turn(
                text, audio_codes, prefill_temp, is_assistant=is_assistant,
            )
            inputs_list.append(turn_input)
            labels_list.append(turn_label)

        input_ids = np.concatenate(inputs_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)

        return {
            "input_ids": torch.from_numpy(input_ids).long(),
            "labels": torch.from_numpy(labels).long(),
        }

    # ------------------------------------------------------------------
    # Collation
    # ------------------------------------------------------------------

    def collate_fn(self, batch):
        max_len = max(item["input_ids"].shape[0] for item in batch)
        B = len(batch)

        tok = self.processor.tokenizer
        text_pad_id = (
            getattr(tok, "pad_token_id", None)
            or getattr(tok, "eos_token_id", None)
            or 0
        )

        input_ids_padded = torch.full(
            (B, max_len, self.channels + 1),
            fill_value=self.audio_channel_pad, dtype=torch.long,
        )
        labels_padded = torch.full(
            (B, max_len, self.channels + 1), fill_value=-100, dtype=torch.long,
        )
        attention_mask = torch.zeros(B, max_len, dtype=torch.bool)

        for i, item in enumerate(batch):
            T = item["input_ids"].shape[0]
            input_ids_padded[i, :T] = item["input_ids"]
            input_ids_padded[i, T:, 0] = text_pad_id
            labels_padded[i, :T] = item["labels"]
            attention_mask[i, :T] = True

        return {
            "input_ids": input_ids_padded.contiguous(),
            "attention_mask": attention_mask.contiguous(),
            "labels": labels_padded.contiguous(),
        }
