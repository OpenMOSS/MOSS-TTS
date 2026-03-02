import torch
import torchaudio
import torch.nn.functional as F
from typing import Optional, List, Union, Any
from transformers import AutoTokenizer
import numpy as np
from mossttsrealtime.modeling_mossttsrealtime import MossTTSRealtime
from transformers.cache_utils import DynamicCache, StaticCache


class MossTTSRealtimeProcessor():
    def __init__(
        self,
        tokenizer: Optional[AutoTokenizer] = None,
        tokenizer_path: str = "/inspire/hdd/project/embodied-multimodality/public/ywzhao/TTS/voiceagent/MOSSTrainer/tts_processor",
        audio_pad="<|audio_pad|>", # 151654
        text_pad="<|text_pad|>", # 151655
    ):
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(tokenizer_path)
        self.audio_pad = self.tokenizer.encode(audio_pad)
        self.text_pad = self.tokenizer.encode(text_pad)
        self.ttsbase_system_prompt = """<|im_start|>system
You are a highly expressive text-to-speech (TTS) engine developed by Mosi Intelligence. 
You possess natural language understanding, emotional modeling, and multi-style speech generation capabilities, allowing you to generate the corresponding speech based on the text given in the assistant.<|im_end|>\n"""
        self.channels = 16
        self.audio_pad_tokens = 1024
        self.delay_tokens_len = 12 
    
    def make_voice_clone_prompt(self, prompt_audio_tokens_len):
        padded_audio_prompt = f"{'<|audio_pad|>' * prompt_audio_tokens_len}"
        voice_clone = f"""<|im_start|>context\nThe assistant section should be synthesized using the following voice timbre:{padded_audio_prompt}<|im_end|>\n"""
        return voice_clone
    
    def make_ensemble(self, prompt_audio_tokens=None):
        if prompt_audio_tokens is not None:
            prompt_audio_tokens = np.array(prompt_audio_tokens)
            prompt_audio_tokens = prompt_audio_tokens[:16,:]
            prompt_audio_tokens = np.transpose(prompt_audio_tokens)
            system_prompt_text = f"{self.ttsbase_system_prompt}" + f"{self.make_voice_clone_prompt(prompt_audio_tokens.shape[0])}"
        else:
            system_prompt_text = f"{self.ttsbase_system_prompt}"

        system_prompt_tokens = self.tokenizer(system_prompt_text)["input_ids"]
        system_prompt_tokens_full = np.full(shape=(len(system_prompt_tokens), self.channels + 1), fill_value=1024)
        system_prompt_tokens_full[:, 0] = system_prompt_tokens

        if prompt_audio_tokens is not None:
            system_prompt_tokens = np.array(system_prompt_tokens)
            indices = np.where(system_prompt_tokens == 151654)[0]
            assert indices.size > 0
            prompt_audio_start_pos, prompt_audio_end_pos = indices[0], indices[-1]
            system_prompt_tokens_full[prompt_audio_start_pos : prompt_audio_end_pos + 1, 1:] = prompt_audio_tokens

        begin_of_response = self.tokenizer.encode("<|im_start|>assistant\n")
        begin_of_response_full = np.full(shape=(len(begin_of_response), self.channels + 1), fill_value=1024)
        begin_of_response_full[:, 0] = begin_of_response

        input_ids = np.concatenate([system_prompt_tokens_full, begin_of_response_full], axis=0)
        return input_ids


class MossTTSRealtimeInference:
    def __init__(
        self,
        model: MossTTSRealtime,
        tokenizer: AutoTokenizer,
        max_length: int = 1000,
        *,
        processor: Optional[Any] = None,
        codec: Optional[Any] = None,
        codec_sample_rate: int = 24000,
        codec_encode_kwargs: Optional[dict] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.aud_pad_id = 151654
        self.text_pad_id = 151655
        self.audio_channel_pad = 1024
        self.bos_audio_id = 1025
        self.eos_audio_id = 1026
        self.channels = 16
        self.processor = processor if processor is not None else MossTTSRealtimeProcessor(tokenizer=self.tokenizer)
        self.codec = codec
        self.codec_sample_rate = int(codec_sample_rate)
        self.codec_encode_kwargs = codec_encode_kwargs or {"chunk_duration": 8}
        self._use_dynamic_local_cache = self.model.config._attn_implementation == "flash_attention_2"

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _build_local_past_key_values(self):
        if self._use_dynamic_local_cache:
            return DynamicCache()
        return StaticCache(config=self.model.local_transformer.config, max_cache_len=self.channels)

    def _load_audio(self, audio_path: str, target_sample_rate: int) -> torch.Tensor:
        wav, sr = torchaudio.load(audio_path)
        if sr != target_sample_rate:
            wav = torchaudio.functional.resample(wav, sr, target_sample_rate)
        if wav.ndim == 2 and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav

    def _extract_codec_codes(self, encode_result: Any) -> torch.Tensor:
        if isinstance(encode_result, dict):
            if "audio_codes" in encode_result:
                return encode_result["audio_codes"]
        if hasattr(encode_result, "audio_codes"):
            return encode_result.audio_codes
        raise ValueError("Unsupported codec.encode() result: expected dict/object with 'audio_codes'.")

    def _encode_reference_audio(self, reference_audio_path: str, device: torch.device) -> np.ndarray:
        if self.codec is None:
            raise ValueError("MossTTSRealtimeInference.codec is None. Pass codec=... when constructing MossTTSRealtimeInference.")
        wav = self._load_audio(reference_audio_path, target_sample_rate=self.codec_sample_rate)
        wav_tensor = wav.unsqueeze(0).to(device)

        encode_result = self.codec.encode(wav_tensor)
        codes = self._extract_codec_codes(encode_result)
        return codes.detach().cpu().numpy()

    def _normalize_batch_inputs(
        self,
        text: Union[str, List[str]],
        reference_audio_path: Optional[Union[str, List[str]]],
    ) -> tuple[List[str], List[Optional[str]]]:
        texts = [text] if isinstance(text, str) else list(text)
        if reference_audio_path is None:
            paths: List[Optional[str]] = [None for _ in range(len(texts))]
            return texts, paths
        if isinstance(reference_audio_path, str):
            paths = [reference_audio_path for _ in range(len(texts))]
            return texts, paths
        paths = [p for p in reference_audio_path]
        if len(paths) == 1 and len(texts) > 1:
            paths = [paths[0] for _ in range(len(texts))]
        if len(texts) == 1 and len(paths) > 1:
            texts = [texts[0] for _ in range(len(paths))]
        if len(texts) != len(paths):
            raise ValueError(f"Batch size mismatch: got {len(texts)} texts but {len(paths)} reference paths.")
        return texts, paths

    def _build_prefill_batch(
        self,
        input_ids_list: List[np.ndarray],
        text_ids_list: List[List[int]],
        text_lengths_list: List[int],
        prefill_max_text: int = 12,
    ):
        text_cur_idx = [min(int(L), prefill_max_text) for L in text_lengths_list]
        batch_size = len(input_ids_list)
        C = self.channels
        concat_list = []
        for i in range(batch_size):
            cur_len = text_cur_idx[i]
            seg = np.full((cur_len, C + 1), self.audio_channel_pad, dtype=np.int64)
            seg[:, 0] = np.asarray(text_ids_list[i][:cur_len], dtype=np.int64)
            seg[cur_len - 1, 1] = self.bos_audio_id

            concat_list.append(np.concatenate([input_ids_list[i], seg], axis=0))

        lengths = [x.shape[0] for x in concat_list]
        max_len = max(lengths)

        padded_ids = []
        padded_attn = []
        for arr, L in zip(concat_list, lengths):
            pad_len = max_len - L
            pad = np.full((pad_len, C + 1), self.audio_channel_pad, dtype=np.int64)
            pad[:, 0] = self.tokenizer.pad_token_id

            padded_ids.append(np.concatenate([pad, arr], axis=0))

            attn = np.ones(L, dtype=np.int64)
            attn_pad = np.zeros(pad_len, dtype=np.int64)
            padded_attn.append(np.concatenate([attn_pad, attn], axis=0))

        input_ids_tensor = torch.from_numpy(np.stack(padded_ids)).to(device=self.device, dtype=torch.long)
        attn_mask_tensor = torch.from_numpy(np.stack(padded_attn)).to(device=self.device, dtype=torch.long)
        return input_ids_tensor, attn_mask_tensor, text_cur_idx

    def _next_text_tokens(self, text_ids_tensors: List[torch.Tensor], cur_idx: List[int], lengths: List[int]):
        out = []
        batch_size = len(text_ids_tensors)
        for i in range(batch_size):
            if cur_idx[i] < lengths[i]:
                out.append(text_ids_tensors[i][cur_idx[i]])
            else:
                out.append(text_ids_tensors[i].new_tensor(self.text_pad_id))
            cur_idx[i] += 1
        return torch.stack(out, dim=0)

    @torch.inference_mode()
    def _generate_from_ids(
        self,
        input_ids: List[np.ndarray], 
        text_ids: List[List[int]], 
        text_lengths: List[int],
        max_length: int,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
        repetition_penalty: float,
        repetition_window: Optional[int],
        device: torch.device,
    ):
        batch_size = len(input_ids)
        C = self.channels

        past_key_values = None
        is_stopping = torch.zeros(batch_size, dtype=torch.bool, device=device)

        current_input_ids, attn_mask, text_cur_idx = self._build_prefill_batch(
            input_ids_list=input_ids,
            text_ids_list=text_ids,
            text_lengths_list=text_lengths,
            prefill_max_text=12,
        )

        outputs = self.model(
            input_ids=current_input_ids,
            attention_mask=attn_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        backbone_h = outputs.last_hidden_state[:, -1:, :]

        audio_tokens = self.generate_local_transformer(
            hidden_states=backbone_h,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            repetition_penalty=None,
            repetition_window=repetition_window,
            generated_tokens=None,
            gen_step=0,
        )

        generated_tokens = torch.full(
            (batch_size, max_length + 1, C),
            self.audio_channel_pad,
            dtype=audio_tokens.dtype,
            device=device,
        )
        generated_tokens[:, 0, :] = audio_tokens
        gen_step = 1

        text_ids_tensors = [torch.tensor(t, device=device, dtype=torch.long) for t in text_ids]

        for _ in range(max_length):
            text_t = self._next_text_tokens(text_ids_tensors, text_cur_idx, text_lengths)
            step_ids = torch.cat([text_t[:, None, None], audio_tokens[:, None, :]], dim=2)
            attn_mask = torch.cat([attn_mask, (~is_stopping).to(attn_mask.dtype)[:, None]], dim=1)
            outputs = self.model(
                input_ids=step_ids,
                attention_mask=attn_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            backbone_h = outputs.last_hidden_state[:, -1:, :]

            audio_tokens = self.generate_local_transformer(
                hidden_states=backbone_h,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                repetition_window=repetition_window,
                generated_tokens=generated_tokens,
                gen_step=gen_step,
            )

            is_stopping |= (audio_tokens[:, 0] == self.eos_audio_id)

            if gen_step < max_length + 1:
                generated_tokens[:, gen_step, :] = audio_tokens
                gen_step += 1
            else:
                break

            if bool(is_stopping.all()):
                break

        effective_len = gen_step

        token_list = []
        for i in range(batch_size):
            sample = generated_tokens[i, :effective_len, :] 
            eos_pos = (sample[:, 0] == self.eos_audio_id).nonzero(as_tuple=True)[0]
            if eos_pos.numel() > 0:
                sample = sample[: int(eos_pos[0].item())]
            token_list.append(sample.cpu().numpy())

        return token_list

    @torch.inference_mode()
    def generate(
        self,
        text: Union[str, List[str]],
        reference_audio_path: Optional[Union[str, List[str]]] = None,
        *,
        max_length: Optional[int] = None,
        temperature: float = 0.8,
        top_p: float = 0.6,
        top_k: int = 30,
        do_sample: bool = True,
        repetition_penalty: float = 1.1,
        repetition_window: Optional[int] = 50,
        device: Optional[torch.device] = None,
    ):
        if device is None:
            device = self.device
        if max_length is None:
            max_length = self.max_length

        texts, ref_paths = self._normalize_batch_inputs(text, reference_audio_path)

        input_ids = []
        for p in ref_paths:
            if p is None:
                input_ids.append(self.processor.make_ensemble(prompt_audio_tokens=None))
            else:
                prompt_codes = self._encode_reference_audio(p, device=device)
                input_ids.append(self.processor.make_ensemble(prompt_codes.squeeze(1)))

        text_ids = self.tokenizer(texts)["input_ids"]
        if isinstance(text_ids, (list, tuple)) and len(texts) == 1 and text_ids and isinstance(text_ids[0], int):
            text_ids = [list(text_ids)]
        text_lengths = [len(item) for item in text_ids]

        return self._generate_from_ids(
            input_ids=input_ids,
            text_ids=text_ids,
            text_lengths=text_lengths,
            max_length=int(max_length),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            do_sample=bool(do_sample),
            repetition_penalty=float(repetition_penalty),
            repetition_window=repetition_window,
            device=device,
        )

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
        batch_size = hidden_states.shape[0]
        device = hidden_states.device
        local_inputs = hidden_states.reshape(-1, 1, self.model.config.local_config.hidden_size)
        output_token = torch.empty(batch_size, self.channels, dtype=torch.long, device=device)

        past_key_values = self._build_local_past_key_values()
        local_token = None

        cache_pos_t = torch.zeros(1, dtype=torch.long, device=device)

        for i in range(self.channels):
            cache_pos_t.fill_(i)

            local_outputs =  self.model.local_transformer(
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

    def apply_repetition_penalty(
        self,
        scores: torch.Tensor, 
        history_tokens: torch.Tensor,
        penalty: float = 1.1,
        repetition_window: Optional[int] = None,
    ):
        scores_ = scores[:, 0, :]
        B, V = scores_.shape
        ht = history_tokens

        if repetition_window is not None and repetition_window > 0:
            ht = ht[:, -repetition_window:]  

        cur = scores_.gather(1, ht)
        new = torch.where(cur < 0, cur * penalty, cur / penalty)
        scores_.scatter_(1, ht, new)
        return scores_

    def apply_top_k(self, logits, top_k, filter_value=float('-inf'), min_tokens_to_keep: int = 1):
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")
        batch_size, vocab_size = logits.shape
        top_k = max(top_k, min_tokens_to_keep)
        top_k = min(top_k, vocab_size)
        indices_to_remove = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        return logits.masked_fill(logits < indices_to_remove, filter_value)

    def apply_top_p(self, logits, top_p, filter_value=float('-inf'), min_tokens_to_keep: int = 1):
        top_p = float(top_p)
        if top_p < 0 or top_p > 1.0:
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")
        
        sorted_logits, sorted_indices = torch.sort(logits, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        
        sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
        sorted_indices_to_remove[..., - min_tokens_to_keep:] = 0
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter(1, sorted_indices, sorted_indices_to_remove)
        logits_processed = logits.masked_fill(indices_to_remove, filter_value)
        return logits_processed
