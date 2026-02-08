# coding=utf-8
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
""" Modeling classes for MossTTSDelay. """

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.cache_utils import Cache
from transformers.models.qwen3 import Qwen3Model
from transformers import initialization as init

from .configuration_moss_tts import MossTTSDelayConfig
from .inference_utils import sample_token, find_last_equal_C

try:
    from .processing_moss_tts import UserMessage, AssistantMessage, MossTTSDelayProcessor
except Exception:
    UserMessage = None
    AssistantMessage = None
    MossTTSDelayProcessor = None

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MossTTSDelayConfig"


@dataclass
class MossTTSDelayOutputWithPast(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Weighted sum of channel losses.
        all_sum_losses (`torch.FloatTensor` of shape `(batch_size, n_vq + 1)`, *optional*):
            Sum of losses for each sample and each channel before averaging.
        all_token_nums (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Number of non-masked tokens per sample.
        sample_losses (`torch.FloatTensor` of shape `(batch_size,)`, *optional*):
            Loss per sample.
        channel_losses (`torch.FloatTensor` of shape `(n_vq + 1,)`, *optional*):
            Loss per channel (text head + vq heads).
        logits (`List[torch.FloatTensor]`, *optional*):
            List of prediction scores from each head.
        past_key_values (`Cache`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed):
            Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer).
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed):
            Tuple of torch.FloatTensor (one for each layer) of the attention weights.
    """
    loss: Optional[torch.FloatTensor] = None
    all_sum_losses: Optional[torch.FloatTensor] = None
    all_token_nums: Optional[torch.LongTensor] = None
    sample_losses: Optional[torch.FloatTensor] = None
    channel_losses: Optional[torch.FloatTensor] = None
    logits: Optional[List[torch.FloatTensor]] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None




class MossTTSDelayPreTrainedModel(PreTrainedModel):
    config_class = MossTTSDelayConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True

    def _init_weights(self, module):
        """
        Transformers 5.0+ safe init:
        - MUST use transformers.initialization helpers
        - MUST respect param._is_hf_initialized to avoid overwriting ckpt-loaded params
        """
        # Let HF handle its standard modules first (LayerNorm, Linear, Embedding, etc.)
        super()._init_weights(module)

        # Pick a std consistent with HF conventions
        # Prefer model/text config initializer_range if present.
        std = None
        if hasattr(self.config, "initializer_range"):
            std = self.config.initializer_range
        elif hasattr(self.config, "language_config") and hasattr(self.config.language_config, "initializer_range"):
            std = self.config.language_config.initializer_range
        else:
            std = 0.02

        # Initialize extra audio embeddings
        if isinstance(module, nn.Embedding):
            # Only touch our extra embeddings (avoid double touching LM's embeddings if not desired)
            # If you prefer, you can skip this check and rely on super()._init_weights for all embeddings.
            if getattr(module, "num_embeddings", None) == self.config.audio_vocab_size + 1:
                init.normal_(module.weight, mean=0.0, std=std)
                # If you later set padding_idx, you must explicitly zero it (and respect _is_hf_initialized!)
                # init.zeros_ will internally check param flags, but slicing needs manual care.

        # Initialize multi-head projections you added
        if isinstance(module, nn.Linear):
            # For your lm_heads, super()._init_weights already covers typical Linear.
            # This block is only needed if you have custom Linear variants later.
            pass



MOSSTTS_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MossTTSDelayConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The MossTTSDelay Model architecture tailored for Text-to-Speech generation with multi-head VQ prediction.",
    MOSSTTS_START_DOCSTRING,
)
class MossTTSDelayModel(MossTTSDelayPreTrainedModel):
    UserMessage = UserMessage
    AssistantMessage = AssistantMessage
    Processor = MossTTSDelayProcessor

    def __init__(self, config: MossTTSDelayConfig):
        super().__init__(config)
        self.config = config

        config.language_config.torch_dtype = config.torch_dtype
        
        self.language_model = Qwen3Model(config.language_config)

        # Audio VQ Embeddings (Extra channels)
        # Note: input_ids[..., 0] uses Qwen's embedding. 
        # input_ids[..., 1:] use these extensions.
        self.emb_ext = nn.ModuleList()
        for vq_idx in range(self.config.n_vq):
            # Add +1 for potential padding/special tokens logic if strictly required by upstream data prep
            self.emb_ext.append(
                nn.Embedding(self.config.audio_vocab_size + 1, config.language_config.hidden_size, padding_idx=None)
            )

        # Multi-Head Prediction Layers
        # Head 0: Main language head
        # Head 1..N: Audio VQ heads
        self.lm_heads = nn.ModuleList([
            nn.Linear(config.language_config.hidden_size, config.language_config.vocab_size, bias=False)
        ])
        for vq_idx in range(self.config.n_vq):
            self.lm_heads.append(
                nn.Linear(config.language_config.hidden_size, self.config.audio_vocab_size + 1, bias=False)
            )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        Computes the combined embeddings from text and multiple audio VQ channels.
        
        Args:
            input_ids: Shape (Batch, Seq_Len, 1 + n_vq)
        """
        # Base Text/Content Embedding
        # input_ids[..., 0] is standard text or semantic tokens
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids[..., 0])

        # Add VQ Embeddings
        for i, embed_layer in enumerate(self.emb_ext):
            # i corresponds to channel i+1 in input_ids
            # We assume the data pipeline ensures indices are within range
            inputs_embeds = inputs_embeds + embed_layer(input_ids[..., i + 1])
        
        return inputs_embeds

    def set_input_embeddings(self, value):
        self.language_model.embed_tokens = value

    def get_output_embeddings(self):
        # Returning a list of heads might break some HF utilities expecting a single head.
        # However, for custom models, this is acceptable.
        return self.lm_heads

    @add_start_docstrings_to_model_forward(MOSSTTS_START_DOCSTRING)
    @replace_return_docstrings(output_type=MossTTSDelayOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        hidden_out_layers: Optional[List[int]] = None,
        channelwise_loss_weight: Optional[List[float]] = None,
        **kwargs,
    ) -> Union[Tuple, MossTTSDelayOutputWithPast]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length, 1 + n_vq)`):
                Indices of input sequence tokens in the vocabulary. 
                Dimension 2 contains: [Text/Semantics, VQ_0, VQ_1, ..., VQ_N].
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length, 1 + n_vq)`, *optional*):
                Labels for computing the masked language modeling loss.
            channelwise_loss_weight (`List[float]`, *optional*):
                Manual weights for summing losses across different heads (Text vs Audio channels).

        Returns:
        """

        if len(input_ids.shape) != 3 or input_ids.shape[-1] != self.config.n_vq + 1:
            raise ValueError("`Input_ids`'s shape should be exactly (batch_size, sequence_length, 1 + n_vq).")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        # 1. Prepare Embeddings
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings(input_ids)

        # 2. Backbone Forward
        # Qwen3Model outputs standard CausalLMOutputWithPast or similar
        outputs = self.language_model(
            input_ids=None, # Passed via inputs_embeds
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True, # Always need hidden states for multi-head projection
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        # 3. Handle specific layer outputs if requested (Delay Pattern often requires features from specific layers)
        last_hidden_state = outputs.last_hidden_state
        if hidden_out_layers is None:
            # Default to using the last layer for all heads
            # In some architectures (like MusicGen), different codebooks come from different transformer layers.
            # Here we default to the final layer as per original code behavior [-1] * (n + 1).
            hidden_states_for_heads = [last_hidden_state] * (len(self.lm_heads))
        else:
            # If hidden_out_layers is provided (e.g. [-1, -2, -3...]), fetch them from all_hidden_states
            # Note: outputs.hidden_states includes embedding output at index 0 usually.
            all_hs = outputs.hidden_states
            hidden_states_for_heads = [all_hs[idx] for idx in hidden_out_layers]

        # 4. Project to Logits (Multi-Head)
        layer_logits = []
        for i, (hs, head) in enumerate(zip(hidden_states_for_heads, self.lm_heads)):
            logits = head(hs)
            # Original code logic: Mask the last token index for audio heads (indices > 0)
            # This implies the vocab size is (N+1) but the model shouldn't predict the (N+1)-th token 
            # (perhaps reserved for padding in the input but invalid for prediction).
            if i > 0:
                logits[..., -1] = float("-inf")
            layer_logits.append(logits)

        # 5. Loss Calculation
        loss = None
        all_sum_losses = None
        all_token_nums = None
        sample_losses = None
        channel_losses = None

        if labels is not None:
            # Ensure labels match input shape rank (B, S, C)
            if labels.dim() != 3:
                raise ValueError(f"Labels must have rank 3 (B, S, C), got {labels.shape}")

            batch_size = labels.size(0)
            n_heads = len(layer_logits)
            
            # Container for per-sample, per-channel losses
            # Shape: [Batch, n_heads]
            all_sum_losses_list = [] 
            
            # Count valid tokens (not -100) per sample. 
            # Note: Assuming mask is consistent across channels or we take sum over dim 1 (seq)
            # Usually strict masking means checking one channel or all. 
            # Original code: torch.sum(labels != -100, dim=1) -> [B, C]
            all_token_nums = torch.sum(labels != -100, dim=1)

            for i, logits in enumerate(layer_logits):
                # logits: [B, S, V]
                # cur_labels: [B, S]
                cur_labels = labels[..., i]
                
                # Flatten for CrossEntropy
                # logits: [B*S, V], labels: [B*S]
                loss_fct = CrossEntropyLoss(reduction='none')
                vocab_size = logits.size(-1)
                
                reshaped_logits = logits.view(-1, vocab_size)
                reshaped_labels = cur_labels.contiguous().view(-1)
                
                # Calculate loss per token
                per_token_loss = loss_fct(reshaped_logits, reshaped_labels)
                
                # Reshape back to [B, S] and sum over Sequence dimension to get per-sample loss
                per_token_loss = per_token_loss.view(batch_size, -1)
                per_sample_loss = torch.sum(per_token_loss, dim=-1) # [B]
                
                all_sum_losses_list.append(per_sample_loss)

            # Stack to [B, n_heads]
            all_sum_losses = torch.stack(all_sum_losses_list, dim=1)

            # Weighted Loss Aggregation
            if channelwise_loss_weight is not None:
                if len(channelwise_loss_weight) != n_heads:
                    raise ValueError(f"channelwise_loss_weight length {len(channelwise_loss_weight)} != {n_heads}")
                
                w_tensor = torch.tensor(channelwise_loss_weight, device=all_sum_losses.device, dtype=all_sum_losses.dtype)
                
                # Sample losses: Weighted sum over channels per sample / Total weight
                # Normalize by token count per channel
                # Avoid division by zero with epsilon or mask
                token_counts_safe = all_token_nums.float().clamp(min=1.0)
                
                normalized_losses = all_sum_losses / token_counts_safe
                sample_losses = (normalized_losses * w_tensor).sum(dim=1) / w_tensor.sum()
                
                # Channel losses: Sum over batch / Sum tokens over batch
                total_loss_per_channel = all_sum_losses.sum(dim=0)
                total_tokens_per_channel = all_token_nums.sum(dim=0).float().clamp(min=1.0)
                channel_losses = total_loss_per_channel / total_tokens_per_channel
                
                # Final scalar loss
                loss = (channel_losses * w_tensor).sum() / w_tensor.sum()
            else:
                # Default average if no weights provided
                total_tokens = all_token_nums.sum().float().clamp(min=1.0)
                loss = all_sum_losses.sum() / total_tokens
                channel_losses = all_sum_losses.sum(dim=0) / all_token_nums.sum(dim=0).clamp(min=1.0)

        return MossTTSDelayOutputWithPast(
            loss=loss,
            all_sum_losses=all_sum_losses,
            all_token_nums=all_token_nums,
            sample_losses=sample_losses,
            channel_losses=channel_losses,
            logits=layer_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 1000,
        text_temperature: float = 1.5,
        text_top_p: float = 1.0,
        text_top_k: int = 50,
        audio_temperature: float = 1.5,
        audio_top_p: float = 0.8,
        audio_top_k: int = 50,
        audio_repetition_penalty: float = 1.0,
    ):
        if text_temperature > 0:
            text_do_sample = True
        else:
            text_temperature = 1
            text_do_sample = False
        if audio_temperature > 0:
            audio_do_sample = True
        else:
            audio_temperature = 1
            audio_do_sample = False
            
        past_key_values = None
        device = input_ids.device
        current_input_ids = input_ids
        current_attention_mask = attention_mask
        batch_size, seq_len, n_vq = input_ids.shape
        n_vq -= 1
        
        generation_ids = input_ids[:]
        is_stopping = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # 三个阶段: 1. 非 audio; 2. audio not delay; 3. audio delay
        audio_lengths = torch.zeros(batch_size, dtype=torch.int64, device=device) # 0 的时候表示阶段1;
        torch_int64_max = torch.iinfo(torch.int64).max
        delayed_lengths = torch.full((batch_size,), torch_int64_max, dtype=torch.int64, device=device) # 最大值的时候表示阶段2;
        
        # 考虑 continuation 时 audio_start 已经在 input_ids 中的情况;
        # NOTE 注意我们目前不考虑任何输入已经开始 delay 的情况;
        # 需要同时考虑 continuation 和直接生成的情况;
        is_continuation = (input_ids[:, -1, 0] == self.config.audio_start_token_id) | (input_ids[:, -1, 0] == self.config.audio_assistant_gen_slot_token_id)
        audio_start_indices = find_last_equal_C(input_ids[..., 0], self.config.audio_start_token_id)
        audio_start_mask = is_continuation & (audio_start_indices != -1)
        audio_lengths[audio_start_mask] = seq_len - audio_start_indices[audio_start_mask]
        
        is_audio = audio_start_mask.clone()
        
        pre_exclude_mask0 = torch.tensor([self.config.pad_token_id, self.config.audio_assistant_gen_slot_token_id, self.config.audio_assistant_delay_slot_token_id, self.config.audio_end_token_id], device=device)
        pre_exclude_mask1 = torch.ones(self.config.language_config.vocab_size, device=device).bool()
        pre_exclude_mask1[[self.config.audio_assistant_gen_slot_token_id, self.config.audio_assistant_delay_slot_token_id]] = False
        
        
        # 注意 time_step 未必表示对于实际对话时，当前输出token的位置，因为有续写的情况;
        for time_step in tqdm(range(max_new_tokens), desc=f"Generating bs{batch_size} ..."):
            outputs = self(
                input_ids=current_input_ids,
                attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            
            next_token_logits = [logit[:, -1, :] / text_temperature if logit_idx == 0 else logit[:, -1, :] / audio_temperature for logit_idx, logit in enumerate(outputs.logits)] # List, len=n_vq+1, [batch_size, 1, vocab_size]; 
            next_token_logits[0] = next_token_logits[0].clone()
            # 1. 先处理 text token;
            next_text_token = torch.full((batch_size,), self.config.pad_token_id, device=device)
            # 第二个 audio_assistant_delay_slot_token_id 和 audio_end 是不需要采样的，audio_start, 每一个 audio_assistant_gen_slot_token_ids 和第一个 audio_assistant_delay_slot_token_id 是需要采样的;
            next_text_token[~is_stopping & (delayed_lengths < n_vq)] = self.config.audio_assistant_delay_slot_token_id
            is_audio_eos = ~is_stopping & (delayed_lengths == n_vq)
            next_text_token[is_audio_eos] = self.config.audio_end_token_id
            is_audio[is_audio_eos] = False
            sampling_text_mask = ~is_stopping & (delayed_lengths > n_vq)
            next_token_logits[0][~is_audio] = next_token_logits[0][~is_audio].index_fill(-1, pre_exclude_mask0, float('-inf'))
            next_token_logits[0][is_audio] = next_token_logits[0][is_audio].masked_fill(pre_exclude_mask1, float('-inf'))
            if time_step == 0:
                next_token_logits[0][..., 151662] = float('-inf')
            if time_step <= n_vq:
                next_token_logits[0][..., self.config.im_end_token_id] = float('-inf')
                
            # 文本层不使用重复惩罚;
            next_text_token[sampling_text_mask] = sample_token(
                logits=next_token_logits[0][sampling_text_mask],
                top_p=text_top_p,
                top_k=text_top_k,
                do_sample=text_do_sample
            )
            is_audio[next_text_token == self.config.audio_start_token_id] = True
            # 只存在一种停止逻辑，即 next_text_token = <|im_end|>;
            is_stopping[next_text_token == self.config.im_end_token_id] = True

            # 2. 再处理 audio tokens;
            # audio_start 和 audio_end 之外的内容直接pad，默认是 pad，我们只需要填充有值的部分即可;
            next_audio_tokens = torch.full((batch_size, n_vq), self.config.audio_pad_code, device=device)
            
            # 需要考虑的是与 audio_start 的距离;
            # 先查看是否是pad的情况; true 表示有值;
            pre_audio_mask = audio_lengths.unsqueeze(1) > torch.arange(n_vq, dtype=int, device=device).expand(batch_size, n_vq)
            post_audio_mask = torch.arange(n_vq, dtype=int, device=device).expand(batch_size, n_vq) > delayed_lengths.unsqueeze(1) - 1
            post_audio_mask[delayed_lengths == torch_int64_max] = True
            sampling_audio_mask = pre_audio_mask & post_audio_mask
            next_audio_tokens[~sampling_audio_mask] = self.config.audio_pad_code
            
            if sampling_audio_mask.sum() > 0:
                audio_logits = torch.stack(next_token_logits[1:], dim=1)[sampling_audio_mask] # torch.stack -> [batch_size, n_vq - 1, vocab_size]
                audio_logits[..., self.config.audio_pad_code] = float('-inf')
                next_audio_tokens[sampling_audio_mask] = sample_token(
                    logits=audio_logits,
                    prev_tokens=generation_ids[:, :, 1:],
                    repetition_penalty=audio_repetition_penalty,
                    top_p=audio_top_p, 
                    top_k=audio_top_k, 
                    do_sample=audio_do_sample
                )
                
            # 这里显示的是下一个时间步时可以直接使用的 audio_lengths 和 delayed_lengths 的状态;
            # audio_lengths[(next_text_token == self.audio_start_token_id) & (audio_lengths > 0)] += 1
            # audio_lengths[(next_text_token == self.audio_start_token_id) | (next_text_token == self.audio_assistant_gen_slot_token_id)] += 1
            audio_lengths[(next_text_token == self.config.audio_start_token_id) | (next_text_token == self.config.audio_assistant_gen_slot_token_id) | (next_text_token == self.config.audio_assistant_delay_slot_token_id)] += 1
            audio_lengths[next_text_token == self.config.audio_end_token_id] = 0
            delayed_lengths[(delayed_lengths == torch_int64_max) & (next_text_token == self.config.audio_assistant_delay_slot_token_id)] = 0
            delayed_lengths[delayed_lengths != torch_int64_max] += 1
            delayed_lengths[delayed_lengths > n_vq] = torch_int64_max
            
            current_input_ids = torch.cat([next_text_token[:, None, None], next_audio_tokens[:, None, :]], dim=2) # [batch_size, 1, n_vq + 1]
            current_attention_mask = torch.cat([current_attention_mask, (~is_stopping).unsqueeze(-1)], dim=-1)
            generation_ids = torch.cat([generation_ids, current_input_ids], dim=1) # [batch_size, seq_len, n_vq + 1]
            
            if is_stopping.sum() == batch_size:
                break
                
        start_indices = find_last_equal_C(input_ids[..., 0], self.config.im_start_token_id) + 3
        start_lengths = seq_len - start_indices

        output = []
        for start_idx, start_length, cur_generation_ids in zip(start_indices, start_lengths, generation_ids):
            output.append((start_length, cur_generation_ids[start_idx:]))
        
        return output
