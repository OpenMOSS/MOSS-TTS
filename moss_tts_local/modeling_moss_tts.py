import os
import copy
import torch
import torch.nn as nn
import logging
import sys

from tqdm import tqdm
from dataclasses import dataclass
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils import ModelOutput
from transformers.cache_utils import Cache
from typing import Optional, List, Tuple, Union
from transformers.loss.loss_utils import ForCausalLMLoss
from transformers import PreTrainedModel, GenerationMixin
from transformers.generation.streamers import BaseStreamer
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model, Qwen3Attention, eager_attention_forward
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessorList, RepetitionPenaltyLogitsProcessor, TopKLogitsWarper, TopPLogitsWarper, TemperatureLogitsWarper
from transformers.masking_utils import create_causal_mask

from .inference_utils import find_last_equal_C
from .configuration_moss_tts import MossTTSDelayConfig

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MossTTSRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., dim]
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.weight


class MossTTSMLP(nn.Module):
    """
    HF-style MLP adapter equivalent to Megatron's SwiGLU FFN:
      in:  input_size
      mid: ffn_hidden_size
      out: output_size

    Computes:
      y = down( silu(gate(x)) * up(x) )

    Optionally includes a pre-norm on input (common in Megatron blocks).
    """
    def __init__(
        self,
        input_size: int,
        ffn_hidden_size: int,
        output_size: int,
        bias: bool = False,
        prenorm: bool = False,
        norm_eps: float = 1e-6,
        use_rmsnorm: bool = True,
    ):
        super().__init__()

        self.prenorm = prenorm
        if prenorm:
            if use_rmsnorm:
                self.norm = MossTTSRMSNorm(input_size, eps=norm_eps)
            else:
                self.norm = nn.LayerNorm(input_size, eps=norm_eps)
        else:
            self.norm = None

        # SwiGLU uses two projections to ffn_hidden_size: gate and up
        self.gate_proj = nn.Linear(input_size, ffn_hidden_size, bias=bias)
        self.up_proj   = nn.Linear(input_size, ffn_hidden_size, bias=bias)

        # down projection to output_size (note: output can differ from input)
        self.down_proj = nn.Linear(ffn_hidden_size, output_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm is not None:
            x = self.norm(x)

        gate = self.gate_proj(x)
        up   = self.up_proj(x)
        h = F.silu(gate) * up
        y = self.down_proj(h)
        return y

def moss_tts_masked_embedding(embedding: nn.Embedding,
                     input_ids: torch.LongTensor,
                     ignore_index: int = -100) -> torch.Tensor:
    """
    对 input_ids 中 != ignore_index 的位置做 embedding，ignore_index 的位置输出全 0 向量。

    Args:
        embedding: 一个 nn.Embedding 层
        input_ids: 任意形状的 LongTensor，里面允许出现 ignore_index
        ignore_index: 需要被忽略的位置标记（默认 -100）

    Returns:
        embeddings: 形状为 (*input_ids.shape, embedding.embedding_dim) 的张量
    """
    # mask: True 表示需要正常 embedding，False 表示输出 0
    mask = (input_ids != ignore_index)  # shape: [...]

    # 为了避免 -100 这种非法 index 传进 embedding，这里先临时替换掉
    safe_ids = input_ids.clone()
    safe_ids[~mask] = 0

    # 正常过 embedding
    out = embedding(safe_ids)  # shape: [..., dim]

    # 把 ignore_index 对应的位置置 0
    out[~mask] = 0.0

    return out

class MossTTSAttentionWithoutPositionalEmbedding(Qwen3Attention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: MossTTSDelayConfig, layer_idx: int):
        super().__init__(config, layer_idx)


    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        assert past_key_value is None

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                print(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            is_causal=True,
            attention_mask=None,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class MossTTSLocalTransformer(Qwen3Model):
    def __init__(self, config: MossTTSDelayConfig):
        super().__init__(config)
        del self.rotary_emb
        del self.embed_tokens
        for layer_idx in range(config.num_hidden_layers):
            self.layers[layer_idx].self_attn = MossTTSAttentionWithoutPositionalEmbedding(config, layer_idx)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs,
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        use_cache = False
        assert not use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            print(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            assert False
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # causal_mask = self._update_causal_mask( # ???
        #     attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        # )
        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        causal_mask = create_causal_mask(**mask_kwargs)


        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        # position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=None,
                past_key_value=None,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=None,
                position_embeddings=None,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        
@dataclass
class MosiTTSOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    loss_all: Optional[Tuple[torch.FloatTensor]] = None
    logits_all: Optional[Tuple[torch.FloatTensor]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class MossTTSGenerateDecoderOnlyOutput(ModelOutput):
    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None


class CustomMixin(GenerationMixin): # TODO 待检查正确性
    def _sample(
        self,
        input_ids: torch.LongTensor, # (B, T, 1+Nq)
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[MossTTSGenerateDecoderOnlyOutput, torch.LongTensor]:
        # 提取配置参数
        # assert False
        speech_pad_idx = self.config.audio_pad_code
        device = input_ids.device
        eos_token_id = generation_config.eos_token_id
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # 初始化输出元组
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # 初始化跟踪变量
        batch_size, cur_len, channels = input_ids.shape  # channels = 8
        input_ids_length = cur_len
        # assert batch_size == 1
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device) # (B, )
        base_length = input_ids.shape[1]
        model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)
        # model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        # 定义logits processor
        if generation_config.do_samples is not None:
            do_samples = generation_config.do_samples
            realprocessor = [LogitsProcessorList() for _ in range(channels)]
            for i, layer_config in enumerate(generation_config.layers):
                if not do_samples[i]:
                    continue
                if layer_config.get("repetition_penalty") is not None and i != 0: # 文本层不用重复惩罚
                    realprocessor[i].append(RepetitionPenaltyLogitsProcessor(penalty=layer_config.get("repetition_penalty")))
                if layer_config.get("temperature") is not None:
                    realprocessor[i].append(TemperatureLogitsWarper(temperature=layer_config.get("temperature")))
                if layer_config.get("top_k") is not None:
                    realprocessor[i].append(TopKLogitsWarper(top_k=layer_config.get("top_k")))
                if layer_config.get("top_p") is not None:
                    realprocessor[i].append(TopPLogitsWarper(top_p=layer_config.get("top_p")))
        else:
            assert False
            do_samples = [do_sample for _ in range(channels)]
            realprocessor = [logits_processor for _ in range(channels)]
        
        pbar = tqdm()
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
             # 准备模型输入
            pbar.update()
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})
            # 前向传递
            outputs = self(**model_inputs, n_vq_for_inference=generation_config.n_vq_for_inference, return_dict=True, output_hidden_states=True)
            model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs)

            if synced_gpus and this_peer_finished:
                continue

            global_trm_output_hidden_states = outputs.hidden_states[-1][:, -1, :] # (B, D)
            dtype = global_trm_output_hidden_states.dtype

            local_trm_dim = self.local_transformer_config.hidden_size
            local_transformer_inputs = torch.zeros(batch_size, 0, local_trm_dim).to(device).to(dtype) # (B, 0 <= t <= Nq, D), 维护当前 local trm 的输入
            current_local_transformer_input = self.speech_embedding_to_local_mlp(global_trm_output_hidden_states) # (B, D) 维护当前 timestamp 的 local trm 的输入，

            next_tokens = [] # 1+Nq * (B, )
            # n_vq_for_inference = int(os.environ['N_VQ_FOR_INFERENCE'])
            n_vq_for_inference = generation_config.n_vq_for_inference
            for layer_index in range(min(channels, 1 + n_vq_for_inference)):
                local_transformer_inputs = torch.cat([local_transformer_inputs, current_local_transformer_input.unsqueeze(1)], dim=1) # (B, t, D)
                local_transformer_outputs = self.local_transformer(
                    input_ids=None,
                    attention_mask=None,
                    inputs_embeds=local_transformer_inputs # (B, t=1+Nq, D)
                )[0] # (B, t=1+Nq, D)
                local_transformer_outputs = self.layer_norm_before_lm_heads[layer_index](
                    self.local_to_speech_embedding_mlps[layer_index](local_transformer_outputs) # (B, t=1+Nq, D)
                ) # (B, t=1+Nq, D)

                next_token_logit = self.lm_heads[layer_index](local_transformer_outputs[:, -1, :]) # (B, V)
                if layer_index != 0:
                    next_token_logit[:, speech_pad_idx] = -torch.inf
                next_token_score = realprocessor[layer_index](input_ids[..., layer_index], next_token_logit) # (B, V)

                if do_samples[layer_index]:
                    channel_ntk = torch.multinomial(nn.functional.softmax(next_token_score, dim=-1), num_samples=1).squeeze(1) # (B, )
                else:
                    channel_ntk = torch.argmax(next_token_score, dim=-1) # (B, )

                next_tokens.append(channel_ntk) # 1+Nq * (B, )
                current_local_transformer_input = self.model.embedding_list[layer_index](channel_ntk) # (B, D)
                current_local_transformer_input = self.speech_embedding_to_local_mlp(current_local_transformer_input) # (B, D)

            for layer_index in range(1 + n_vq_for_inference, channels):
                next_tokens.append(torch.zeros((batch_size, )).to(torch.int).to(device))
            next_tokens = torch.stack(next_tokens, dim=-1)  # (B, 1+Nq)

            if has_eos_stopping_criteria:
                for i in range(channels):
                    pddp = eos_token_id if i == 0 else speech_pad_idx
                    next_tokens[:, i] = next_tokens[:, i] * unfinished_sequences + pddp * (1 - unfinished_sequences)
            
            input_ids = torch.cat([input_ids, next_tokens[:, None, :]], dim=1) # (B, T, 1+Nq)
            if streamer is not None:
                streamer.put(next_tokens[:, 0].cpu())

            stopping = stopping_criteria(input_ids[..., 0], scores)
            # stopping = stopping_criteria(input_ids[..., 0], scores)
            unfinished_sequences = unfinished_sequences & ~stopping
            this_peer_finished = unfinished_sequences.max() == 0

            if return_dict_in_generate:
                if output_scores:
                    assert False
                    scores += (next_token_scores,)
                if output_logits:
                    assert False
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (outputs.attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (outputs.hidden_states,)

            cur_len += 1
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            return MossTTSGenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            start_indices = find_last_equal_C(input_ids[..., 0], self.config.audio_start_token_id)
            start_lengths = input_ids_length - start_indices - 1 # voice clone 下是 0，续写情况下是 prompt 音频的长度，不包括 audio_start_token
            output = []
            for start_idx, start_length, cur_generation_ids in zip(start_indices, start_lengths, input_ids):
                output.append((start_length, cur_generation_ids[start_idx:]))
            
            return output


class MosiTTSPretrainedModel(PreTrainedModel):
    config_class = MossTTSDelayConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True


class MosiTTSModel(MosiTTSPretrainedModel):
    def __init__(self, config: MossTTSDelayConfig):
        super().__init__(config)
        self.text_pad_idx = config.pad_token_id
        self.speech_pad_idx = config.audio_pad_code
        self.embedding_list = nn.ModuleList([])
        self.embedding_list.append(nn.Embedding(config.vocab_size, config.hidden_size, self.text_pad_idx))
        self.channels = 1 + config.n_vq
        for _ in range(1, self.channels):
            self.embedding_list.append(nn.Embedding(config.audio_vocab_size + 1, config.hidden_size, self.speech_pad_idx))

        self.language_model = Qwen3Model(config.language_config)
        self.post_init()
        self.language_model.embed_tokens.requires_grad_(False)

    def get_input_embeddings(self):
        return self.embedding_list[0]

    def set_input_embeddings(self, value: nn.Embedding):
        self.embedding_list[0] = value

    def _prepare_multi_modal_inputs(
        self,
        input_ids: torch.LongTensor,
        n_vq_for_inference: Optional[int] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        """
        Prepares multi-modal embeddings from input_ids of shape (batch_size, channels, sequence_length).
        For channel 0: text + speech tokens, for channels 1 to channels-1: speech tokens padded with speech_pad_token.
        """
        batch_size, seq_length, channels = input_ids.shape
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {channels}")

        if n_vq_for_inference is None:
            n_vq_for_inference = self.channels - 1
        n_vq_for_inference = max(1, min(self.channels - 1, int(n_vq_for_inference)))

        inputs_embeds = torch.zeros(batch_size, seq_length, self.config.hidden_size, device=input_ids.device, dtype=self.embedding_list[0].weight.dtype)
        for i in range(min(channels, 1 + n_vq_for_inference)):
            embed_layer = self.embedding_list[i]
            channel_input = input_ids[...,i]
            inputs_embeds += embed_layer(channel_input)

        return inputs_embeds # (B, T, D)

    def forward(
        self,
        input_ids: torch.LongTensor = None,  # Shape: (batch_size, channels, sequence_length)
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if input_ids is not None:
            inputs_embeds = self._prepare_multi_modal_inputs(input_ids, **kwargs) # (B, T, D)

        outputs = self.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        return outputs


class MossTTSDelayModel(MosiTTSPretrainedModel, CustomMixin):
    _tied_weights_keys = None
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: MossTTSDelayConfig):
        super().__init__(config)
        self.model = MosiTTSModel(config)
        self.channels = 1 + config.n_vq
        self.weights = [1 for _ in range(self.channels)]
        self.vocab_size = config.vocab_size

        local_transformer_config = copy.deepcopy(config.language_config)
        local_transformer_config.num_hidden_layers = config.local_num_layers
        local_transformer_config.hidden_size = config.local_hidden_size
        local_transformer_config.intermediate_size = config.local_ffn_hidden_size
        self.local_transformer_config = local_transformer_config
        self.local_transformer = MossTTSLocalTransformer(self.local_transformer_config)

        self.speech_embedding_to_local_mlp = MossTTSMLP(
            input_size=config.hidden_size,
            ffn_hidden_size=config.additional_mlp_ffn_hidden_size,
            output_size=config.local_hidden_size
        )
        self.local_to_speech_embedding_mlps = nn.ModuleList([
            MossTTSMLP(
                input_size=config.local_hidden_size,
                ffn_hidden_size=config.additional_mlp_ffn_hidden_size,
                output_size=config.hidden_size
            )
            for _ in range(self.channels)
        ])

        self.layer_norm_before_lm_heads = nn.ModuleList([
            MossTTSRMSNorm(config.hidden_size)
            for _ in range(self.channels)
        ])

        self.lm_heads = nn.ModuleList([])
        self.lm_heads.append(nn.Linear(config.hidden_size, config.vocab_size, bias=False))
        for _ in range(1, self.channels):
            self.lm_heads.append(nn.Linear(config.hidden_size, 1 + config.audio_vocab_size, bias=False))
        self.post_init()
        self._freeze_unused_qwen_embeddings()
        self.register_load_state_dict_post_hook(self._freeze_unused_qwen_embeddings_post_load)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model = super().from_pretrained(*args, **kwargs)
        model._freeze_unused_qwen_embeddings()
        return model

    def get_input_embeddings(self):
        return self.model.embedding_list[0]

    def _freeze_unused_qwen_embeddings(self) -> None:
        self.model.language_model.embed_tokens.requires_grad_(False)

    def _freeze_unused_qwen_embeddings_post_load(self, module, incompatible_keys) -> None:
        module._freeze_unused_qwen_embeddings()

    def can_generate(self):
        return True

    def _build_generation_config(
        self,
        generation_config: Optional[GenerationConfig] = None,
        max_new_tokens: Optional[int] = None,
        text_temperature: Optional[float] = None,
        text_top_p: Optional[float] = None,
        text_top_k: Optional[int] = None,
        text_repetition_penalty: Optional[float] = None,
        audio_temperature: Optional[float] = None,
        audio_top_p: Optional[float] = None,
        audio_top_k: Optional[int] = None,
        audio_repetition_penalty: Optional[float] = None,
        n_vq_for_inference: Optional[int] = None,
    ) -> GenerationConfig:
        config = copy.deepcopy(generation_config or self.generation_config)

        text_temperature = 1.5 if text_temperature is None else float(text_temperature)
        text_top_p = 1.0 if text_top_p is None else float(text_top_p)
        text_top_k = 50 if text_top_k is None else int(text_top_k)
        text_repetition_penalty = 1.0 if text_repetition_penalty is None else float(text_repetition_penalty)
        audio_temperature = 1.0 if audio_temperature is None else float(audio_temperature)
        audio_top_p = 0.95 if audio_top_p is None else float(audio_top_p)
        audio_top_k = 50 if audio_top_k is None else int(audio_top_k)
        audio_repetition_penalty = 1.1 if audio_repetition_penalty is None else float(audio_repetition_penalty)

        text_do_sample = text_temperature > 0
        if not text_do_sample:
            text_temperature = 1.0
        audio_do_sample = audio_temperature > 0
        if not audio_do_sample:
            audio_temperature = 1.0

        if max_new_tokens is not None:
            config.max_new_tokens = int(max_new_tokens)
        elif getattr(config, "max_new_tokens", None) is None:
            config.max_new_tokens = 100000 # about 2.2 hours , can be overridden by user input, you can set to a smaller value for faster generation during debugging

        if getattr(config, "pad_token_id", None) is None:
            config.pad_token_id = self.config.pad_token_id
        config.eos_token_id = self.config.audio_end_token_id
        config.use_cache = True
        config.do_sample = text_do_sample or audio_do_sample

        resolved_n_vq = self.channels - 1 if n_vq_for_inference is None else int(n_vq_for_inference)
        resolved_n_vq = max(1, min(self.channels - 1, resolved_n_vq))
        config.n_vq_for_inference = resolved_n_vq
        config.do_samples = [text_do_sample] + [audio_do_sample] * (self.channels - 1)
        config.layers = [
            {
                "repetition_penalty": text_repetition_penalty,
                "temperature": text_temperature,
                "top_p": text_top_p,
                "top_k": text_top_k,
            }
        ] + [
            {
                "repetition_penalty": audio_repetition_penalty,
                "temperature": audio_temperature,
                "top_p": audio_top_p,
                "top_k": audio_top_k,
            }
            for _ in range(self.channels - 1)
        ]
        return config

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        max_new_tokens: Optional[int] = None,
        text_temperature: Optional[float] = None,
        text_top_p: Optional[float] = None,
        text_top_k: Optional[int] = None,
        text_repetition_penalty: Optional[int] = None,
        audio_temperature: Optional[float] = None,
        audio_top_p: Optional[float] = None,
        audio_top_k: Optional[int] = None,
        audio_repetition_penalty: Optional[float] = None,
        n_vq_for_inference: Optional[int] = None,
        **kwargs,
    ):
        resolved_generation_config = self._build_generation_config(
            generation_config=generation_config,
            max_new_tokens=max_new_tokens,
            text_temperature=text_temperature,
            text_top_p=text_top_p,
            text_top_k=text_top_k,
            text_repetition_penalty=text_repetition_penalty,
            audio_temperature=audio_temperature,
            audio_top_p=audio_top_p,
            audio_top_k=audio_top_k,
            audio_repetition_penalty=audio_repetition_penalty,
            n_vq_for_inference=n_vq_for_inference,
        )
        return super().generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=resolved_generation_config,
            **kwargs,
        )

    # def tie_weights(self):
    #     ...
        # for i in range(self.config.channels):
        #     self._tie_or_clone_weights(self.lm_heads[i], self.model.embedding_list[i])

    def set_input_embeddings(self, value):
        self.model.embedding_list[0] = value

    def get_output_embeddings(self):
        return self.lm_heads[0]

    def set_output_embeddings(self, new_embeddings):
        self.lm_heads[0] = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def set_weights(self, weights):
        self.weights = weights

    def _prepare_shifted_audio_inputs(self, label_ids): # (B, T, 1 + Nq) 可能有 -100
        text_and_audio_label_embed_list = [] # Nq * (1, T, B, D)
        for i in range(self.channels - 1):
            text_and_audio_label_embed_list.append(
                moss_tts_masked_embedding(self.model.embedding_list[i], label_ids[:, :, i]).unsqueeze(0).transpose(1, 2) # (B, T) -> (B, T, D) -> (1, B, T, D) -> (1, T, B, D)
            ) # (1, T, B, D)
        audio_label_embeds = torch.stack(text_and_audio_label_embed_list, dim=0) # (Nq, 1, T, B, D)
        audio_label_embeds = audio_label_embeds.contiguous()[:, 0, :, :, :].transpose(1, 2) # (Nq, B, T, D)
        return audio_label_embeds # (Nq, B, T, D)

    def forward(
        self,
        input_ids: torch.LongTensor = None, # (B, T, 1 + Nq)
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None, # (B, T, 1 + Nq), TODO labels 为 input_ids shift 一位的结果
        channelwise_loss_weight: Optional[List[float]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, MosiTTSOutputWithPast]:
        device = input_ids.device if not input_ids is None else inputs_embeds.device
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids, # (B, T, 1 + Nq)
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        if labels is not None:
            local_transformer_inputs_from_global = outputs[0].unsqueeze(0) # (1, B, T, D)
            D_global= local_transformer_inputs_from_global.shape[-1]
            local_transformer_inputs_from_speech_embeddings = self._prepare_shifted_audio_inputs(labels) # (B, T, 1 + Nq) -> (Nq, B, T, D)
            local_transformer_input_hidden_states = torch.cat([local_transformer_inputs_from_global, local_transformer_inputs_from_speech_embeddings], dim=0).contiguous() # (1 + Nq, B, T, D)
            local_transformer_input_hidden_states = self.speech_embedding_to_local_mlp(local_transformer_input_hidden_states) # (1 + Nq, B, T, D)
            N_channels, B, T, D_local = local_transformer_input_hidden_states.shape
            local_transformer_input_hidden_states = local_transformer_input_hidden_states.permute(1, 2, 0, 3) # (B, T, 1 + Nq, D)
            local_transformer_input_hidden_states = local_transformer_input_hidden_states.reshape(B * T, N_channels, D_local) # (batch_size=B * T, time=1+Nq, D)
            local_transformer_output_hidden_states = self.local_transformer( # TODO 没有开位置编码
                input_ids=None,
                attention_mask=None,
                inputs_embeds=local_transformer_input_hidden_states # (batch_size=B * T, time=1+Nq, D)
            )[0] # (batch_size=B * T, time=1+Nq, D)
            after_lm_head_mlp_hidden_states = [] # Nq+1 * (B*T, D) TODO ???
            for i in range(self.channels):
                after_lm_head_mlp_hidden_states.append(
                    self.layer_norm_before_lm_heads[i](
                        self.local_to_speech_embedding_mlps[i](
                            local_transformer_output_hidden_states[:, i, :] # (B*T, D)
                        )
                    )
                )  # Nq+1 * (B*T, D)

            after_lm_head_mlp_hidden_states = torch.stack(after_lm_head_mlp_hidden_states, dim=0)  # (1 + Nq, B*T, D)
            after_lm_head_mlp_hidden_states = after_lm_head_mlp_hidden_states.reshape(N_channels, B, T, D_global) # (1 + Nq, B, T, D)
            logits_all = [lm_head(h_i) for lm_head, h_i in zip(self.lm_heads, after_lm_head_mlp_hidden_states)] # 1+Nq * (B, T, V)

            loss_all = torch.empty(self.channels, device=device) # (1 + Nq)

            for i in range(self.channels):
                vocab_size = self.config.vocab_size if i == 0 else (1 + self.config.audio_vocab_size)
                loss_all[i] = ForCausalLMLoss(logits_all[i], labels[..., i], vocab_size, shift_labels=labels[..., i]) # (B, T, V), (B, T) => (1, )
            effective_weights = channelwise_loss_weight if channelwise_loss_weight is not None else self.weights
            if len(effective_weights) != self.channels:
                raise ValueError(
                    f"`channelwise_loss_weight` length {len(effective_weights)} != {self.channels}."
                )
            total_weight = float(sum(effective_weights))
            if total_weight <= 0:
                raise ValueError("`channelwise_loss_weight` must sum to a positive value.")
            normalized_weights = [float(weight_i) / total_weight for weight_i in effective_weights] # (1+Nq, )

            total_loss = 0
            for w, loss in zip(normalized_weights, loss_all):
                total_loss += w * loss
        else:
            total_loss = None
            loss_all = None
            logits_all = [None] * self.channels

        if not return_dict:
            output = (logits_all,) + outputs[1:]
            return (total_loss, loss_all) + output if total_loss is not None else output

        return MosiTTSOutputWithPast(
            loss=total_loss,
            logits=logits_all[0],
            loss_all=loss_all,
            logits_all=logits_all, # 1+Nq * (B, T, V)
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states, # L * (B, T, D)
            attentions=outputs.attentions,
        )
