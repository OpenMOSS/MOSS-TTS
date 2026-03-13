"""
Processor class for Asteroid fused with Codec.
"""

from typing import Union

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput


class MossTTSDelayWithCodecProcessor(ProcessorMixin):
    attributes = ["tokenizer"]
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        tokenizer=None,
        chat_template=None,
        AudioToken_PlaceHolder="<|Audio|>",
        audio_assistant_delay_slot_token="<|audio_assistant_delay_slot|>",
        audio_assistant_gen_slot_token="<|audio_assistant_gen_slot|>",
        audio_end_token="<|audio_end|>",
        audio_start_token="<|audio_start|>",
        audio_user_slot_token="<|audio_user_slot|>",
        audio_pad_code: int = 1024,
        text_pad_code: int = 151643,
        n_vq: int = 32,
        shift: bool = True,
        downsample_rate: int = 1920,
    ):
        super().__init__(tokenizer, chat_template=chat_template)
        self.audio_user_slot_token = (
            tokenizer.audio_user_slot_token
            if hasattr(tokenizer, "audio_user_slot_token")
            else audio_user_slot_token
        )
        self.audio_start_token = (
            tokenizer.audio_start_token
            if hasattr(tokenizer, "audio_start_token")
            else audio_start_token
        )
        self.audio_end_token = (
            tokenizer.audio_end_token
            if hasattr(tokenizer, "audio_end_token")
            else audio_end_token
        )
        self.AudioToken_PlaceHolder = AudioToken_PlaceHolder
        self.audio_assistant_gen_slot_token = audio_assistant_gen_slot_token
        self.audio_assistant_delay_slot_token = audio_assistant_delay_slot_token

        self.audio_user_slot_token_id = tokenizer.convert_tokens_to_ids(
            self.audio_user_slot_token
        )
        self.audio_start_token_id = tokenizer.convert_tokens_to_ids(
            self.audio_start_token
        )
        self.audio_end_token_id = tokenizer.convert_tokens_to_ids(self.audio_end_token)
        self.audio_assistant_gen_slot_token_id = tokenizer.convert_tokens_to_ids(
            self.audio_assistant_gen_slot_token
        )
        self.audio_assistant_delay_slot_token_id = tokenizer.convert_tokens_to_ids(
            self.audio_assistant_delay_slot_token
        )

        self.audio_pad_code = audio_pad_code
        self.text_pad_code = text_pad_code
        self.n_vq = n_vq
        self.shift = shift
        self.downsample_rate = downsample_rate

    def __call__(
        self,
        text: Union[
            TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]
        ] = None,
        padding=True,
        return_tensors="pt",
        audios: Union[np.ndarray, list[np.ndarray]] = None,
        **kwargs,
    ) -> BatchFeature:
        if text is None:
            raise ValueError("You need to specify `text` input to process.")
        elif isinstance(text, str):
            text = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
        elif (
            isinstance(text, list)
            and all(isinstance(t, str) for t in text)
            and len(text) == 1
        ):
            text = f"<|im_start|>user\n{text[0]}<|im_end|>\n<|im_start|>assistant\n"
        else:
            raise ValueError("`text` input must be a string or list of strings (=1).")

        self._parsed_cache = {"text": [], "audio": [], "sign": []}
        _parsed_text_list = text.split(self.AudioToken_PlaceHolder)

        if audios is not None:
            if isinstance(audios, np.ndarray):
                audio_pad_length = [audios.shape[-1] // self.downsample_rate]
                audios = [audios]
            elif (
                isinstance(audios, list)
                and len(audios) == 1
                and isinstance(audios[0], np.ndarray)
            ):
                audio_pad_length = [
                    audio.shape[-1] // self.downsample_rate for audio in audios
                ]
            elif isinstance(audios, list) and all(
                isinstance(a, np.ndarray) for a in audios
            ):
                raise NotImplementedError("Multi references not implemented yet.")
                audio_pad_length = [
                    audio.shape[-1] // self.downsample_rate for audio in audios
                ]
            else:
                raise ValueError(
                    "`audio` input must be a numpy array or list of numpy arrays."
                )

            audios = [self.loudness_normalize(audio) for audio in audios]

            for i, parsed_audio in enumerate(audios):
                parsed_text_left = _parsed_text_list[i] + self.audio_start_token
                _parsed_text_list[i + 1] = (
                    self.audio_end_token + _parsed_text_list[i + 1]
                )
                parsed_text_tokens_left = torch.tensor(
                    self.tokenizer.encode(parsed_text_left), dtype=torch.long
                )
                self._parsed_cache["text"].extend(
                    [
                        parsed_text_tokens_left,
                        torch.tensor(
                            self.tokenizer.encode(
                                self.audio_user_slot_token * audio_pad_length[i]
                            ),
                            dtype=torch.long,
                        ),
                    ]
                )
                parsed_audio_tokens_left = torch.full(
                    (parsed_text_tokens_left.shape[0], self.n_vq),
                    self.audio_pad_code,
                    dtype=torch.long,
                )
                parsed_audio_tokens_pad = torch.full(
                    (audio_pad_length[i], self.n_vq),
                    self.audio_pad_code,
                    dtype=torch.long,
                )
                parsed_audio_tokens_left[:, 0] = self.audio_pad_code
                self._parsed_cache["audio"].extend(
                    [parsed_audio_tokens_left, parsed_audio_tokens_pad]
                )
                self._parsed_cache["sign"].extend([False, True])

        parsed_text_tokens_right = torch.tensor(
            self.tokenizer.encode(_parsed_text_list[-1]), dtype=torch.long
        )
        self._parsed_cache["text"].append(parsed_text_tokens_right)
        parsed_audio_tokens_right = torch.full(
            (parsed_text_tokens_right.shape[0], self.n_vq),
            self.audio_pad_code,
            dtype=torch.long,
        )
        parsed_audio_tokens_right[:, 0] = self.audio_pad_code
        self._parsed_cache["audio"].append(parsed_audio_tokens_right)
        self._parsed_cache["sign"].append(False)
        unified_tokens = []

        for i, (text_tokens, audio_tokens, sign) in enumerate(
            zip(
                self._parsed_cache["text"],
                self._parsed_cache["audio"],
                self._parsed_cache["sign"],
            )
        ):
            if sign:
                text_tokens = torch.cat(
                    [
                        text_tokens,
                        torch.full(
                            (audio_tokens.shape[1] - 1,),
                            self.audio_user_slot_token_id,
                            dtype=torch.long,
                        ),
                    ]
                )
                audio_tokens = self.apply_delay_pattern(
                    audio_tokens,
                    ch0_pad_id=self.audio_pad_code,
                    pad_id=self.audio_pad_code,
                )
            cur_unified_tokens = torch.cat(
                [text_tokens.unsqueeze(1), audio_tokens], dim=1
            )
            unified_tokens.append(cur_unified_tokens)
        unified_tokens = torch.cat(unified_tokens, dim=0)

        if audios is None:
            inputs = {
                "input_ids": unified_tokens,
            }
        else:
            inputs = {
                "input_ids": unified_tokens,
                "audio_features": torch.as_tensor(audios[0], dtype=torch.float32),
            }

        return BatchFeature(data={**inputs}, tensor_type=return_tensors)

    @staticmethod
    def apply_delay_pattern(
        tokens: torch.Tensor, ch0_pad_id: int, pad_id: int
    ) -> torch.Tensor:
        delayed_tokens = torch.full(
            (tokens.shape[0] + tokens.shape[1] - 1, tokens.shape[1]),
            pad_id,
            dtype=torch.long,
        )
        delayed_tokens[:, 0] = torch.cat(
            [
                tokens[:, 0],
                torch.full((tokens.shape[1] - 1,), ch0_pad_id, dtype=torch.long),
            ]
        )
        for i in range(1, tokens.shape[1]):
            delayed_tokens[i : i + tokens.shape[0], i] = tokens[:, i]
        return delayed_tokens

    @staticmethod
    def loudness_normalize(
        wav: np.ndarray,
        target_dbfs: float = -20,
        gain_range: tuple[float, float] = (-3.0, 3.0),
    ) -> np.ndarray:
        wav = np.asarray(wav, dtype=np.float32)
        if wav.size == 0:
            return wav
        rms = np.sqrt(np.mean(wav * wav))
        current_dbfs = 20.0 * np.log10(rms + 1e-9)
        gain = np.clip(target_dbfs - current_dbfs, gain_range[0], gain_range[1])
        factor = 10.0 ** (gain / 20.0)
        return wav * factor


__all__ = ["MossTTSDelayWithCodecProcessor"]
