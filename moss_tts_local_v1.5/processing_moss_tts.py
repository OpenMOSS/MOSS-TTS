# coding=utf-8
"""Processor for the MOSS-TTS-Local-Transformer-v1.5 HuggingFace release."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import torch
import torchaudio
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    BatchFeature,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    logging,
)

from .configuration_moss_tts import MossTTSLocalConfig

logger = logging.get_logger(__name__)

AUDIO_PLACEHOLDER = "<|audio|>"
USER_ROLE_PREFIX = "user\n"
USER_TEMPLATE_REFERENCE_PREFIX = (
    "<user_inst>\n"
    "- Reference(s):\n"
)
USER_TEMPLATE_AFTER_REFERENCE_SUFFIX = (
    "\n"
    "- Text:\n"
)
USER_TEMPLATE_SUFFIX = "\n</user_inst>"
ASSISTANT_TURN_PREFIX = "\n"
ASSISTANT_ROLE_PREFIX = "assistant\n"
USER_MESSAGE_FIELDS = (
    "text",
    "reference",
    "instruction",
    "tokens",
    "quality",
    "sound_event",
    "ambient_sound",
    "language",
)


def _normalize_template_value(value: Any) -> str:
    if value is None:
        return "None"
    resolved = str(value).strip()
    return resolved or "None"


def _render_user_prompt_after_reference(
    language_code: object | None = None,
    prompt_fields: Optional[Dict[str, Any]] = None,
) -> str:
    fields = dict(prompt_fields or {})
    return (
        "\n- Instruction:\n"
        + _normalize_template_value(fields.get("instruction"))
        + "\n- Tokens:\n"
        + _normalize_template_value(fields.get("tokens"))
        + "\n- Quality:\n"
        + _normalize_template_value(fields.get("quality"))
        + "\n- Sound Event:\n"
        + _normalize_template_value(fields.get("sound_event"))
        + "\n- Ambient Sound:\n"
        + _normalize_template_value(fields.get("ambient_sound"))
        + "\n- Language:\n"
        + _normalize_template_value(fields.get("language", language_code))
        + USER_TEMPLATE_AFTER_REFERENCE_SUFFIX
    )


@dataclass
class Message:
    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError


@dataclass
class UserMessage(Message):
    text: Optional[str] = None
    reference: Optional[List[Optional[Union[str, os.PathLike, torch.Tensor]]]] = None
    instruction: Optional[str] = None
    tokens: Optional[int] = None
    quality: Optional[str] = None
    sound_event: Optional[str] = None
    ambient_sound: Optional[str] = None
    language: Optional[str] = None

    def __post_init__(self) -> None:
        template = """<user_inst>
- Reference(s):
{reference}
- Instruction:
{instruction}
- Tokens:
{tokens}
- Quality:
{quality}
- Sound Event:
{sound_event}
- Ambient Sound:
{ambient_sound}
- Language:
{language}
- Text:
{text}
</user_inst>"""

        audio_codes_list: list[Union[str, os.PathLike, torch.Tensor]] = []
        if self.reference is None:
            reference = "None"
        else:
            reference_items: list[str] = []
            for speaker_idx, speaker_reference in enumerate(self.reference):
                if speaker_reference is None:
                    continue
                # Keep raw audio placeholders directly under "- Reference(s):".
                # Speaker labels such as "[S1]:" change the token sequence and
                # can affect voice-clone conditioning.
                reference_items.append(AUDIO_PLACEHOLDER)
                audio_codes_list.append(speaker_reference)
            reference = "\n".join(reference_items) if reference_items else "None"

        self._content = (
            template.replace("{reference}", str(reference))
            .replace("{instruction}", str(self.instruction))
            .replace("{tokens}", str(self.tokens))
            .replace("{quality}", str(self.quality))
            .replace("{sound_event}", str(self.sound_event))
            .replace("{ambient_sound}", str(self.ambient_sound))
            .replace("{language}", str(self.language))
            .replace("{text}", str(self.text))
        )
        self._audio_codes_list = audio_codes_list

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": "user",
            "content": self._content,
            "audio_codes_list": self._audio_codes_list,
            "text": self.text,
            "instruction": self.instruction,
            "tokens": self.tokens,
            "quality": self.quality,
            "sound_event": self.sound_event,
            "ambient_sound": self.ambient_sound,
            "language": self.language,
        }


@dataclass
class AssistantMessage(Message):
    audio_codes_list: List[Union[str, os.PathLike, torch.Tensor]]
    content: str = AUDIO_PLACEHOLDER

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": "assistant",
            "content": self.content,
            "audio_codes_list": self.audio_codes_list,
        }


class MossTTSLocalProcessor(ProcessorMixin):
    # Required constructor attributes; the codec is managed locally.
    attributes = ["tokenizer"]
    tokenizer_class = "AutoTokenizer"
    audio_tokenizer_class = "AutoModel"

    tokenizer: PreTrainedTokenizerBase
    audio_tokenizer: Any

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        audio_tokenizer: Any = None,
        model_config: Optional[MossTTSLocalConfig] = None,
        **kwargs,
    ) -> None:
        super().__init__(tokenizer=tokenizer, **kwargs)
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.model_config = model_config or MossTTSLocalConfig()

        def _id_to_token(token_id: int) -> str:
            token = tokenizer.convert_ids_to_tokens(int(token_id))
            if isinstance(token, list):
                return token[0] if token else ""
            return cast(str, token)

        self.audio_user_slot_token = _id_to_token(self.model_config.audio_user_slot_token_id)
        self.audio_assistant_slot_token = _id_to_token(self.model_config.audio_assistant_slot_token_id)
        self.audio_start_token = _id_to_token(self.model_config.audio_start_token_id)
        self.audio_end_token = _id_to_token(self.model_config.audio_end_token_id)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        trust_remote_code = kwargs.pop("trust_remote_code", True)
        kwargs.pop("_from_auto", None)
        codec_path = kwargs.pop("codec_path", None)
        codec_weight_dtype = kwargs.pop("codec_weight_dtype", "fp32")
        codec_compute_dtype = kwargs.pop("codec_compute_dtype", None)
        codec_attention_implementation = kwargs.pop("codec_attention_implementation", None)

        codec_kwargs: Dict[str, Any] = {}
        if codec_weight_dtype is not None:
            # Default to fp32 codec weights; callers can pass codec_weight_dtype="bf16" to reduce memory.
            codec_kwargs["codec_weight_dtype"] = codec_weight_dtype

        model_ref = Path(str(pretrained_model_name_or_path))
        model_ref_or_name = model_ref if model_ref.exists() else pretrained_model_name_or_path
        model_config = cast(
            MossTTSLocalConfig,
            AutoConfig.from_pretrained(
                model_ref_or_name,
                *args,
                trust_remote_code=trust_remote_code,
                **kwargs,
            ),
        )

        if codec_path is None:
            try:
                processor_dict, _ = cls.get_processor_dict(
                    pretrained_model_name_or_path,
                    **dict(kwargs),
                )
                codec_path = processor_dict.get("audio_tokenizer_name_or_path")
                audio_tokenizer_dict = processor_dict.get("audio_tokenizer", {})
                if isinstance(audio_tokenizer_dict, dict):
                    codec_path = audio_tokenizer_dict.get("audio_tokenizer_name_or_path") or codec_path
            except Exception:
                codec_path = None
        if codec_path is None:
            codec_path = getattr(model_config, "audio_tokenizer_name_or_path", None)
        if codec_path is None:
            codec_path = "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"

        tokenizer = AutoTokenizer.from_pretrained(
            model_ref_or_name,
            *args,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        audio_tokenizer = AutoModel.from_pretrained(
            codec_path,
            trust_remote_code=trust_remote_code,
            **kwargs,
            **codec_kwargs,
        )
        if codec_compute_dtype is not None and hasattr(audio_tokenizer, "set_compute_dtype"):
            audio_tokenizer.set_compute_dtype(codec_compute_dtype)
        if codec_attention_implementation is not None and hasattr(
            audio_tokenizer, "set_attention_implementation"
        ):
            audio_tokenizer.set_attention_implementation(codec_attention_implementation)
        return cls(
            tokenizer=tokenizer,
            audio_tokenizer=audio_tokenizer,
            model_config=model_config,
            **kwargs,
        )

    @staticmethod
    def build_user_message(
        text: Optional[str] = None,
        reference: Optional[List[Optional[Union[str, os.PathLike, torch.Tensor]]]] = None,
        instruction: Optional[str] = None,
        tokens: Optional[int] = None,
        quality: Optional[str] = None,
        sound_event: Optional[str] = None,
        ambient_sound: Optional[str] = None,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        if reference is not None and not isinstance(reference, list):
            reference = [reference]
        return UserMessage(
            text=text,
            reference=reference,
            instruction=instruction,
            tokens=tokens,
            quality=quality,
            sound_event=sound_event,
            ambient_sound=ambient_sound,
            language=language,
        ).to_dict()

    @staticmethod
    def build_assistant_message(
        audio_codes_list: List[Union[str, os.PathLike, torch.Tensor]],
        content: str = AUDIO_PLACEHOLDER,
    ) -> Dict[str, Any]:
        return AssistantMessage(audio_codes_list=audio_codes_list, content=content).to_dict()

    def _assert_fixed_nq(self, n_vq: Optional[int]) -> int:
        config_nq = int(self.model_config.n_vq)
        if n_vq is not None and int(n_vq) != config_nq:
            raise ValueError(
                "This MOSS-TTS-Local-Transformer-v1.5 release uses the RVQ depth stored in the model config. "
                f"Expected n_vq={config_nq}, got {int(n_vq)}."
            )
        return config_nq

    def _encode_text(self, text: str) -> list[int]:
        try:
            return list(self.tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            return list(self.tokenizer.encode(text))

    def _build_text_rows(self, token_ids: Sequence[int], *, device: Optional[torch.device] = None) -> torch.Tensor:
        rows = torch.full(
            (len(token_ids), int(self.model_config.n_vq) + 1),
            int(self.model_config.audio_pad_token_id),
            dtype=torch.long,
            device=device,
        )
        if token_ids:
            rows[:, 0] = torch.tensor([int(token_id) for token_id in token_ids], dtype=torch.long, device=rows.device)
        return rows

    def _build_audio_rows(self, audio_tokens: torch.Tensor, slot_token_id: int) -> torch.Tensor:
        rows = torch.full(
            (int(audio_tokens.shape[0]), int(self.model_config.n_vq) + 1),
            int(self.model_config.audio_pad_token_id),
            dtype=torch.long,
            device=audio_tokens.device,
        )
        if rows.shape[0] > 0:
            rows[:, 0] = int(slot_token_id)
            rows[:, 1:] = audio_tokens.to(dtype=torch.long)
        return rows

    def _user_prompt_prefix_ids(self) -> list[int]:
        return (
            [int(self.model_config.im_start_token_id)]
            + self._encode_text(USER_ROLE_PREFIX)
            + self._encode_text(USER_TEMPLATE_REFERENCE_PREFIX)
        )

    def _user_prompt_after_reference_ids(
        self,
        language_code: object | None,
        prompt_fields: Optional[Dict[str, Any]],
    ) -> list[int]:
        return self._encode_text(
            _render_user_prompt_after_reference(
                language_code=language_code,
                prompt_fields=prompt_fields,
            )
        )

    def _assistant_prompt_prefix_ids(self) -> list[int]:
        return (
            self._encode_text(USER_TEMPLATE_SUFFIX)
            + [int(self.model_config.im_end_token_id)]
            + self._encode_text(ASSISTANT_TURN_PREFIX)
            + [int(self.model_config.im_start_token_id)]
            + self._encode_text(ASSISTANT_ROLE_PREFIX)
        )

    def _prompt_fields_from_user_message(self, message: Dict[str, Any]) -> dict[str, Any]:
        fields = {}
        for key in ("instruction", "tokens", "quality", "sound_event", "ambient_sound"):
            if key in message and message.get(key) is not None:
                fields[key] = message.get(key)
        if "language" in message and message.get("language") is not None:
            fields["language"] = message.get("language")
        return fields

    def _build_generation_or_voice_clone_codes(
        self,
        message: Dict[str, Any],
        n_vq: int,
    ) -> torch.Tensor:
        if "text" not in message:
            raise ValueError("Direct MOSS-TTS-Local-Transformer-v1.5 generation requires messages built by build_user_message(...).")
        text = "" if message.get("text") is None else str(message.get("text"))
        prompt_fields = self._prompt_fields_from_user_message(message)
        language_code = message.get("language")
        audio_codes_list = self._resolve_audio_items(message.get("audio_codes_list", []), n_vq)
        text_token_ids = self._encode_text(text)

        if audio_codes_list:
            parts: list[torch.Tensor] = [self._build_text_rows(
                self._user_prompt_prefix_ids(),
                device=audio_codes_list[0].device,
            )]
            for reference_codes in audio_codes_list:
                parts.append(self._build_text_rows([int(self.model_config.audio_start_token_id)], device=reference_codes.device))
                parts.append(self._build_audio_rows(reference_codes, int(self.model_config.audio_user_slot_token_id)))
                parts.append(self._build_text_rows([int(self.model_config.audio_end_token_id)], device=reference_codes.device))
            parts.append(
                self._build_text_rows(
                    self._user_prompt_after_reference_ids(language_code, prompt_fields)
                    + text_token_ids
                    + self._assistant_prompt_prefix_ids()
                    + [int(self.model_config.audio_start_token_id)],
                    device=audio_codes_list[0].device,
                )
            )
            return torch.cat(parts, dim=0)

        prompt_token_ids = (
            self._user_prompt_prefix_ids()
            + self._encode_text("None")
            + self._user_prompt_after_reference_ids(language_code, prompt_fields)
            + text_token_ids
            + self._assistant_prompt_prefix_ids()
            + [int(self.model_config.audio_start_token_id)]
        )
        return self._build_text_rows(prompt_token_ids)

    def _build_continuation_codes(
        self,
        conversation: list[Dict[str, Any]],
        n_vq: int,
    ) -> torch.Tensor:
        if len(conversation) < 2:
            raise ValueError("continuation mode requires a user message followed by an assistant audio message.")
        user_message = conversation[-2]
        assistant_message = conversation[-1]
        if user_message.get("role") != "user" or assistant_message.get("role") != "assistant":
            raise ValueError("continuation mode requires the last two messages to be user, assistant.")
        if "text" not in user_message:
            raise ValueError("Direct MOSS-TTS-Local-Transformer-v1.5 continuation requires user messages built by build_user_message(...).")

        text = "" if user_message.get("text") is None else str(user_message.get("text"))
        prompt_fields = self._prompt_fields_from_user_message(user_message)
        language_code = user_message.get("language")
        prompt_token_ids = (
            self._user_prompt_prefix_ids()
            + self._encode_text("None")
            + self._user_prompt_after_reference_ids(language_code, prompt_fields)
            + self._encode_text(text)
            + self._assistant_prompt_prefix_ids()
            + [int(self.model_config.audio_start_token_id)]
        )
        audio_codes_list = self._resolve_audio_items(assistant_message.get("audio_codes_list", []), n_vq)
        if not audio_codes_list:
            return self._build_text_rows(prompt_token_ids)
        if len(audio_codes_list) != 1:
            raise ValueError("The MOSS-TTS-Local-Transformer-v1.5 continuation path expects exactly one prompt audio item.")
        prompt_audio_codes = audio_codes_list[0]
        return torch.cat(
            [
                self._build_text_rows(prompt_token_ids, device=prompt_audio_codes.device),
                self._build_audio_rows(prompt_audio_codes, int(self.model_config.audio_assistant_slot_token_id)),
            ],
            dim=0,
        )

    def _try_build_direct_codes(
        self,
        conversation: list[Dict[str, Any]],
        mode: str,
        n_vq: int,
    ) -> Optional[torch.Tensor]:
        if mode == "generation" and len(conversation) == 1 and conversation[-1].get("role") == "user":
            if "text" in conversation[-1]:
                return self._build_generation_or_voice_clone_codes(conversation[-1], n_vq)
            return None
        if mode == "continuation" and len(conversation) >= 2:
            if "text" in conversation[-2]:
                return self._build_continuation_codes(conversation, n_vq)
            return None
        return None

    def __call__(self, *args, **kwargs) -> BatchFeature:
        conversations = args[0] if args else kwargs.pop("conversations")
        mode: str = kwargs.pop("mode", "generation")
        apply_chat_template: bool = kwargs.pop("apply_chat_template", True)
        n_vq = self._assert_fixed_nq(kwargs.pop("n_vq", None))

        kwargs.pop("return_tensors", None)
        kwargs.pop("padding", None)
        kwargs.pop("truncation", None)

        if mode not in {"generation", "continuation", "computing_loss"}:
            raise ValueError(f"Unsupported mode: {mode}")
        if isinstance(conversations, (Message, dict)):
            conversations = [conversations]
        elif isinstance(conversations, list) and conversations and all(
            isinstance(item, (Message, dict)) for item in conversations
        ):
            conversations = [conversations]

        input_ids_list: list[torch.Tensor] = []
        for conversation in conversations:
            if isinstance(conversation, (Message, dict)):
                conversation = [conversation]
            conversation = [self._normalize_message(message) for message in conversation]

            if (mode == "generation") ^ (conversation[-1]["role"] == "user"):
                raise ValueError("generation mode must end with a user message.")
            if mode == "continuation" and conversation[-1]["role"] != "assistant":
                raise ValueError("continuation mode must end with an assistant message.")

            direct_codes = self._try_build_direct_codes(conversation, mode, n_vq)
            if direct_codes is not None:
                input_ids_list.append(direct_codes)
                continue

            unified_parts = []
            for message_idx, message in enumerate(conversation):
                content = str(message["content"])
                if apply_chat_template:
                    add_generation_prompt = mode == "generation" and message_idx == len(conversation) - 1
                    try:
                        content = self.tokenizer.apply_chat_template(
                            [{"role": message["role"], "content": content}],
                            add_generation_prompt=add_generation_prompt,
                            tokenize=False,
                        )
                    except Exception:
                        logger.warning("apply_chat_template failed; falling back to raw message content.")

                raw_audio_items = message.get("audio_codes_list", [])
                audio_codes_list = self._resolve_audio_items(raw_audio_items, n_vq)
                unified_parts.append(
                    self._get_unified_codes(
                        role=message["role"],
                        content=content,
                        audio_codes_list=audio_codes_list,
                        truncation=(mode == "continuation"),
                    )
                )

            unified_codes = torch.cat(unified_parts, dim=0)
            if mode == "generation":
                audio_start_row = torch.full(
                    (1, n_vq + 1),
                    int(self.model_config.audio_pad_token_id),
                    dtype=unified_codes.dtype,
                    device=unified_codes.device,
                )
                audio_start_row[:, 0] = int(self.model_config.audio_start_token_id)
                unified_codes = torch.cat([unified_codes, audio_start_row], dim=0)
            input_ids_list.append(unified_codes)

        return BatchFeature(data=self._pad(input_ids_list))

    def _normalize_message(self, message: Union[Message, Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(message, Message):
            return message.to_dict()
        if not isinstance(message, dict):
            raise TypeError("Each message must be a Message or dict.")
        if "content" in message and "audio_codes_list" in message:
            return message
        role = message.get("role")
        if role == "user":
            return self.build_user_message(**{key: message.get(key) for key in USER_MESSAGE_FIELDS})
        if role == "assistant":
            return self.build_assistant_message(
                audio_codes_list=message.get("audio_codes_list", []),
                content=message.get("content", AUDIO_PLACEHOLDER),
            )
        raise ValueError(f"Unsupported role: {role}")

    def _resolve_audio_items(
        self,
        raw_audio_items: list[Any],
        n_vq: int,
    ) -> list[torch.Tensor]:
        if not raw_audio_items:
            return []
        resolved: list[Optional[torch.Tensor]] = [None] * len(raw_audio_items)
        paths: list[str] = []
        path_positions: list[int] = []
        for index, item in enumerate(raw_audio_items):
            if isinstance(item, torch.Tensor):
                if item.ndim != 2 or int(item.shape[1]) != n_vq:
                    raise ValueError(f"audio code tensor must have shape [T, {n_vq}], got {tuple(item.shape)}.")
                resolved[index] = item.to(dtype=torch.long).cpu()
            elif isinstance(item, (str, os.PathLike)):
                paths.append(str(item))
                path_positions.append(index)
            else:
                raise TypeError("Audio items must be tensors or path-like values.")
        if paths:
            encoded = self.encode_audios_from_path(paths, n_vq=n_vq)
            for position, codes in zip(path_positions, encoded):
                resolved[position] = codes
        return [cast(torch.Tensor, item) for item in resolved]

    def _pad(self, input_ids_list: list[torch.Tensor]) -> Dict[str, torch.Tensor]:
        device = input_ids_list[0].device
        lengths = torch.tensor([item.shape[0] for item in input_ids_list], device=device)
        padded = torch.nn.utils.rnn.pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=int(self.model_config.audio_pad_token_id),
            padding_side="left",
        )
        left_pad_mask = (padded.shape[1] - lengths).unsqueeze(1) > torch.arange(
            padded.shape[1],
            device=device,
        ).unsqueeze(0)
        padded[..., 0][left_pad_mask] = int(self.model_config.pad_token_id)
        attention_mask = torch.zeros(padded.shape[:2], dtype=torch.bool, device=device)
        attention_mask[~left_pad_mask] = True
        return {"input_ids": padded, "attention_mask": attention_mask}

    @staticmethod
    def _replace_audio_placeholders(
        content: str,
        lengths: list[int],
        slot_token: str,
        audio_start_token: str,
        audio_end_token: str,
    ) -> str:
        placeholder_count = content.count(AUDIO_PLACEHOLDER)
        if placeholder_count != len(lengths):
            raise ValueError(
                f"Number of {AUDIO_PLACEHOLDER} ({placeholder_count}) does not match "
                f"audio item count ({len(lengths)})."
            )
        lengths_iter = iter(lengths)

        def replacer(_: re.Match) -> str:
            length = int(next(lengths_iter))
            if length <= 0:
                return f"{audio_start_token}{audio_end_token}"
            return f"{audio_start_token}{slot_token * length}{audio_end_token}"

        return re.sub(re.escape(AUDIO_PLACEHOLDER), replacer, content)

    def _get_unified_codes(
        self,
        role: str,
        content: str,
        audio_codes_list: list[torch.Tensor],
        truncation: bool,
    ) -> torch.Tensor:
        n_vq = int(self.model_config.n_vq)
        slot_token = self.audio_user_slot_token if role == "user" else self.audio_assistant_slot_token
        content = self._replace_audio_placeholders(
            content=content,
            lengths=[int(codes.shape[0]) for codes in audio_codes_list],
            slot_token=slot_token,
            audio_start_token=self.audio_start_token,
            audio_end_token=self.audio_end_token,
        )
        text_codes = torch.tensor(
            self.tokenizer.encode(content),
            dtype=torch.long,
            device=audio_codes_list[0].device if audio_codes_list else None,
        )

        audio_start_indices = torch.where(text_codes == int(self.model_config.audio_start_token_id))[0]
        audio_end_indices = torch.where(text_codes == int(self.model_config.audio_end_token_id))[0]
        if len(audio_start_indices) != len(audio_codes_list) or len(audio_end_indices) != len(audio_codes_list):
            raise ValueError("Audio placeholders do not match the encoded audio spans.")

        if not audio_codes_list:
            audio_codes = torch.full(
                (len(text_codes), n_vq),
                int(self.model_config.audio_pad_token_id),
                dtype=torch.long,
                device=text_codes.device,
            )
        else:
            pieces: list[torch.Tensor] = []
            prefix_idx = 0
            for start_t, end_t, codes in zip(audio_start_indices, audio_end_indices, audio_codes_list):
                start_idx = int(start_t.item())
                end_idx = int(end_t.item())
                pad_before = torch.full(
                    (start_idx - prefix_idx + 1, n_vq),
                    int(self.model_config.audio_pad_token_id),
                    dtype=torch.long,
                    device=codes.device,
                )
                pieces.extend([pad_before, codes.to(dtype=torch.long)])
                prefix_idx = end_idx
            if truncation:
                trailing = torch.zeros(
                    (0, n_vq),
                    dtype=torch.long,
                    device=audio_codes_list[0].device,
                )
            else:
                last_end = int(audio_end_indices[-1].item())
                trailing = torch.full(
                    (len(text_codes) - last_end, n_vq),
                    int(self.model_config.audio_pad_token_id),
                    dtype=torch.long,
                    device=audio_codes_list[0].device,
                )
            pieces.append(trailing)
            audio_codes = torch.cat(pieces, dim=0)

        if text_codes.shape[0] != audio_codes.shape[0]:
            min_len = min(text_codes.shape[0], audio_codes.shape[0])
            text_codes = text_codes[:min_len]
            audio_codes = audio_codes[:min_len]
        return torch.cat([text_codes.unsqueeze(1), audio_codes], dim=1)

    def _parse_text_codes(self, start_length: int, text_codes: torch.LongTensor) -> str:
        text = cast(str, self.tokenizer.decode(text_codes))
        prefix = cast(str, self.tokenizer.decode(text_codes[:start_length]))
        text = text[len(prefix):]
        audio_pattern = re.compile(
            rf"(?:{re.escape(self.audio_start_token)})?"
            rf"(?:{re.escape(self.audio_assistant_slot_token)})*"
            rf"{re.escape(self.audio_end_token)}"
        )
        return audio_pattern.sub(
            lambda match: AUDIO_PLACEHOLDER if self.audio_assistant_slot_token in match.group(0) else "",
            text,
        )

    def _parse_audio_codes(
        self,
        start_length: int,
        audio_codes: torch.LongTensor,
        *,
        return_stereo: bool = True,
    ) -> list[torch.Tensor]:
        is_pad = audio_codes.eq(int(self.model_config.audio_pad_token_id)).all(dim=1)
        non_pad = ~is_pad
        if not bool(non_pad.any().item()):
            return []
        idx = torch.nonzero(non_pad).squeeze(1)
        breaks = torch.where(idx[1:] != idx[:-1] + 1)[0] + 1
        segment_indices = [idx] if breaks.numel() == 0 else list(torch.tensor_split(idx, breaks.cpu().tolist()))
        code_segments = [audio_codes[segment] for segment in segment_indices]
        decoded = self.decode_audio_codes(code_segments, return_stereo=return_stereo)

        if start_length > 0 and code_segments and decoded:
            first_code_length = int(code_segments[0].shape[0])
            if first_code_length > 0:
                trim_ratio = max(0.0, min(float(start_length) / float(first_code_length), 1.0))
                if trim_ratio >= 1.0:
                    decoded = decoded[1:]
                elif trim_ratio > 0.0:
                    trim_samples = int(decoded[0].shape[-1] * trim_ratio)
                    decoded[0] = decoded[0][..., trim_samples:]
        return decoded

    def decode(self, output: Any, *, return_stereo: bool = True) -> list[Optional[AssistantMessage]]:
        generated_messages: list[Optional[AssistantMessage]] = []
        for start_length, generation_ids in output:
            content = self._parse_text_codes(int(start_length), generation_ids[:, 0])
            audio_codes_list = self._parse_audio_codes(
                int(start_length),
                generation_ids[:, 1:],
                return_stereo=return_stereo,
            )
            if content == "":
                generated_messages.append(None)
            else:
                generated_messages.append(
                    AssistantMessage(
                        content=content,
                        audio_codes_list=cast(list[Union[str, torch.Tensor]], audio_codes_list),
                    )
                )
        return generated_messages

    @staticmethod
    def loudness_normalize(
        wav: torch.Tensor,
        target_dbfs: float = -20.0,
        gain_range: tuple[float, float] = (-3.0, 3.0),
    ) -> torch.Tensor:
        wav = wav.to(torch.float32)
        if wav.numel() == 0:
            return wav
        current_dbfs = 10.0 * torch.log10(torch.mean(wav**2) + 1e-9)
        gain = max(gain_range[0], min(float(target_dbfs - current_dbfs), gain_range[1]))
        return wav * (10.0 ** (gain / 20.0))

    def _get_audio_tokenizer_device(self) -> torch.device:
        audio_tokenizer = getattr(self, "audio_tokenizer", None)
        if audio_tokenizer is None:
            raise RuntimeError("audio_tokenizer is not set.")
        try:
            return next(audio_tokenizer.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def encode_audios_from_wav(
        self,
        wav_list: Union[torch.Tensor, list[torch.Tensor]],
        sampling_rate: int,
        n_vq: Optional[int] = None,
    ) -> list[torch.Tensor]:
        n_vq = self._assert_fixed_nq(n_vq)
        if self.audio_tokenizer is None:
            raise RuntimeError("audio_tokenizer is not set.")
        if isinstance(wav_list, torch.Tensor):
            wav_list = [wav_list]
        target_sr = int(self.model_config.sampling_rate)
        device = self._get_audio_tokenizer_device()
        prepared = []
        for wav in wav_list:
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            if wav.shape[0] == 1:
                wav = wav.repeat(2, 1)
            elif wav.shape[0] > 2:
                wav = wav[:2]
            if int(sampling_rate) != target_sr:
                wav = torchaudio.functional.resample(wav, int(sampling_rate), target_sr)
            prepared.append(self.loudness_normalize(wav).to(device))

        if hasattr(self.audio_tokenizer, "batch_encode"):
            encoded = self.audio_tokenizer.batch_encode(prepared, num_quantizers=n_vq)
            audio_codes = encoded.audio_codes
            audio_lengths = encoded.audio_codes_lengths
        else:
            max_len = max(int(wav.shape[-1]) for wav in prepared)
            input_values = torch.zeros(len(prepared), 1, max_len, dtype=torch.float32, device=device)
            padding_mask = torch.zeros(len(prepared), max_len, dtype=torch.bool, device=device)
            for index, wav in enumerate(prepared):
                input_values[index, 0, : wav.shape[-1]] = wav
                padding_mask[index, : wav.shape[-1]] = True
            encoded = self.audio_tokenizer.encode(
                input_values,
                padding_mask=padding_mask,
                num_quantizers=n_vq,
                return_dict=True,
            )
            audio_codes = encoded.audio_codes
            audio_lengths = encoded.audio_codes_lengths

        if audio_codes is None or audio_lengths is None:
            raise RuntimeError("audio_tokenizer did not return audio_codes/audio_codes_lengths.")
        result = []
        for index in range(int(audio_codes.shape[1])):
            length = int(audio_lengths[index].item())
            result.append(audio_codes[:, index, :length].transpose(0, 1).contiguous().cpu().long())
        return result

    def encode_audios_from_path(
        self,
        wav_path_list: Union[str, os.PathLike, list[Union[str, os.PathLike]]],
        n_vq: Optional[int] = None,
    ) -> list[torch.Tensor]:
        if isinstance(wav_path_list, (str, os.PathLike)):
            wav_path_list = [wav_path_list]
        wavs = []
        target_sr = int(self.model_config.sampling_rate)
        for wav_path in wav_path_list:
            wav, sr = torchaudio.load(str(wav_path))
            if int(sr) != target_sr:
                wav = torchaudio.functional.resample(wav, int(sr), target_sr)
            wavs.append(wav)
        return self.encode_audios_from_wav(wavs, target_sr, n_vq=n_vq)

    def decode_audio_codes(
        self,
        audio_tokens_list: Union[torch.Tensor, list[torch.Tensor]],
        *,
        return_stereo: bool = True,
    ) -> list[torch.Tensor]:
        if self.audio_tokenizer is None:
            raise RuntimeError("audio_tokenizer is not set.")
        if isinstance(audio_tokens_list, torch.Tensor):
            audio_tokens_list = [audio_tokens_list]
        if not audio_tokens_list:
            return []

        n_vq = int(self.model_config.n_vq)
        device = self._get_audio_tokenizer_device()
        codes_list = [
            codes[:, :n_vq].transpose(0, 1).contiguous().to(device=device, dtype=torch.long)
            for codes in audio_tokens_list
        ]
        max_len = max(int(codes.shape[1]) for codes in codes_list)
        audio_codes = torch.zeros(n_vq, len(codes_list), max_len, device=device, dtype=torch.long)
        padding_mask = torch.zeros(len(codes_list), max_len, device=device, dtype=torch.bool)
        for index, codes in enumerate(codes_list):
            length = int(codes.shape[1])
            audio_codes[:, index, :length] = codes
            padding_mask[index, :length] = True

        decoded = self.audio_tokenizer.decode(
            audio_codes,
            padding_mask=padding_mask,
            num_quantizers=n_vq,
            return_dict=True,
            chunk_duration=8,
        )
        audio = decoded.audio
        audio_lengths = decoded.audio_lengths
        if audio is None or audio_lengths is None:
            raise RuntimeError("audio_tokenizer.decode did not return audio/audio_lengths.")
        wavs = []
        for index in range(int(audio.shape[0])):
            length = int(audio_lengths[index].item())
            wav = audio[index, :, :length].contiguous().cpu().to(torch.float32)
            if not return_stereo:
                if wav.shape[0] == 1:
                    wav = wav.squeeze(0)
                else:
                    wav = wav.mean(dim=0)
            wavs.append(wav)
        return wavs
