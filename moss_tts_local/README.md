# Architecture: Global Latent + Local Transformer (MossTTSLocal)

This document details the **MossTTSLocal** architecture, a flagship variant of the MOSS-TTS family. It utilizes a hierarchical autoregressive framework consisting of a **Global Transformer (Temporal)** and a **Local Transformer (Depth)** to achieve high-fidelity, controllable speech synthesis.

---

## 1. Overview: Temporal + Depth Modeling

Unlike the "Delay-Pattern" architecture which shifts codebooks across time steps, the **MossTTSLocal** architecture performs **Time-Synchronous Generation**. In each time step $t$, the model predicts the entire sequence of Residual Vector Quantization (RVQ) tokens for that specific frame.

### Key Components
*   **Global Transformer (Temporal Backbone):** Responsible for long-range dependencies, linguistic context, and global style/prosody modeling.
*   **Local Transformer (Depth Transformer):** A lightweight module that models the coarse-to-fine dependencies between RVQ codebook layers within a single time step.
*   **Cat (Causal Audio Tokenizer):** The underlying discrete interface that provides high-fidelity audio compression and reconstruction at 24kHz.

---

## 2. Technical Specifications


| Feature | Specification |
| :--- | :--- |
| **Backbone Model** | Initialized from **Qwen3-1.7B** |
| **Depth Transformer** | 4 Transformer blocks (Hidden: 1536, FFN: 8960) |
| **Audio Tokenizer** | **Cat** (Causal Audio Tokenizer) |
| **Sampling Rate** | 24,000 Hz |
| **Frame Rate** | 12.5 Hz (1s ≈ 12.5 tokens/blocks) |
| **Codebooks** | 32 RVQ layers (10-bit each) |
| **Generation Mode** | Purely Autoregressive (AR) |

---

## 3. Core Mechanism: Progressive Sequence Dropout

A distinguishing feature of MossTTSLocal is its support for **Variable Bitrate Synthesis** without architectural changes. This is achieved through **Progressive Sequence Dropout** during training.

### Training Strategy
*   With probability $p$, the model randomly samples a prefix length $K \in \{1, \dots, N_q - 1\}$.
*   RVQ layers beyond $K$ are discarded, forcing the model to learn conditional generation under varying bitrates.
*   This removes the train-test mismatch when decoding at reduced depths.

### Inference Control
During inference, users can explicitly control the synthesis bitrate by selecting a depth $K_{infer}$:
*   **Low Bitrate (e.g., $K=4$):** Faster generation, lower bandwidth, suitable for previews.
*   **High Bitrate (e.g., $K=32$):** Maximum fidelity and SOTA reconstruction quality.

---

## 4. Prediction Topology

Generation follows a nested autoregressive logic:
1.  **Step-by-Step (Temporal):** Step $t$ depends on all previous steps $<t$.
2.  **Layer-by-Layer (Depth):** Within step $t$, RVQ layer $k$ depends on the global latent $g_t$ and all preceding layers $1 \dots k-1$.

**Block Layout:**
At each step $t$, the model emits a block $Y[t]$:
$Y[t] = [ \text{Audio Token Layer}_1, \text{Audio Token Layer}_2, \dots, \text{Audio Token Layer}_K ]$
There is **no cross-step token shifting**; all tokens in the block correspond to the same physical audio frame.

---

## 5. Evaluation & Performance

According to the `moss_tts_model_card.md`, the **MossTTSLocal-1.7B** achieves state-of-the-art results on zero-shot TTS benchmarks:

| Metric | Result (Seed-TTS-Eval) |
| :--- | :--- |
| **EN SIM (Speaker Similarity)** | **0.7342** (Highest among open-source models) |
| **ZH SIM (Speaker Similarity)** | **0.7882** (Highest among open-source models) |
| **EN WER (Word Error Rate)** | **1.85%** |
| **ZH CER (Char Error Rate)** | **1.20%** |

**Conclusion:** MossTTSLocal excels in objective reconstruction metrics and speaker identity preservation, making it the ideal choice for research, quality-critical applications, and bitrate-sensitive environments.

---

## 6. Architecture Comparison

| Aspect | MossTTSLocal (Architecture B) | MossTTSDelay (Architecture A) |
| :--- | :--- | :--- |
| **Structure** | Temporal + Depth Transformers | Single Transformer (Multi-head) |
| **Scheduling** | Per-step Synchronous Blocks | Delay-Pattern Scheduling |
| **Bitrate** | Variable (via Dropout) | Fixed (usually full depth) |
| **Streaming** | Naturally streaming-friendly | Requires alignment buffering |
| **Best For** | Research, Quality Benchmarks | Production, Long-form Stability |

---

## 7. Quick Start: Generation & Continuation

The usage snippets below show direct generation (with/without cloning) and continuation for MossTTSLocal. Run them as-is to try the model end to end.

### 7.1 Direct Generation & Voice Cloning

```python
import os
from pathlib import Path
import torch
import torchaudio
from transformers import AutoModel, AutoProcessor, GenerationConfig
# Disable the broken cuDNN SDPA backend
torch.backends.cuda.enable_cudnn_sdp(False)
# Keep these enabled as fallbacks
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

class DelayGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layers = kwargs.get("layers", [{} for _ in range(32)])
        self.do_samples = kwargs.get("do_samples", None)
        self.n_vq_for_inference = 32
        
def initial_config(tokenizer, model_name_or_path):
    generation_config = DelayGenerationConfig.from_pretrained(model_name_or_path)
    generation_config.pad_token_id = tokenizer.pad_token_id
    generation_config.eos_token_id = 151653
    generation_config.max_new_tokens = 1000000
    generation_config.temperature = 1.0
    generation_config.top_p = 0.95
    generation_config.top_k = 100
    generation_config.repetition_penalty = 1.1
    generation_config.use_cache = True
    generation_config.do_sample = False
    return generation_config




pretrained_model_name_or_path = "OpenMOSS-Team/MOSS-TTS"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

processor = AutoProcessor.from_pretrained(
    pretrained_model_name_or_path,
    trust_remote_code=True,
)
processor.audio_tokenizer = processor.audio_tokenizer.to(device)

text_1 = """亲爱的你，
你好呀。

今天，我想用最认真、最温柔的声音，对你说一些重要的话。
这些话，像一颗小小的星星，希望能在你的心里慢慢发光。

首先，我想祝你——
每天都能平平安安、快快乐乐。

希望你早上醒来的时候，
窗外有光，屋子里很安静，
你的心是轻轻的，没有着急，也没有害怕。
"""
text_2 = """We stand on the threshold of the AI era.
Artificial intelligence is no longer just a concept in laboratories, but is entering every industry, every creative endeavor, and every decision. It has learned to see, hear, speak, and think, and is beginning to become an extension of human capabilities. AI is not about replacing humans, but about amplifying human creativity, making knowledge more equitable, more efficient, and allowing imagination to reach further. A new era, jointly shaped by humans and intelligent systems, has arrived."""
text_3 = "nin2 hao3，qing3 wen4 nin2 lai2 zi4 na3 zuo4 cheng2 shi4？"
text_4 = "nin2 hao3，qing4 wen3 nin2 lai2 zi4 na4 zuo3 cheng4 shi3？"
text_5 = "您好，请问您来自哪 zuo4 cheng2 shi4？"
text_6 = "/həloʊ, meɪ aɪ æsk wɪtʃ sɪti juː ɑːr frʌm?/"

ref_audio_1 = "https://speech-demo.oss-cn-shanghai.aliyuncs.com/moss_tts_demo/tts_readme_demo/reference_zh.wav"
ref_audio_2 = "https://speech-demo.oss-cn-shanghai.aliyuncs.com/moss_tts_demo/tts_readme_demo/reference_en.m4a"

conversations = [
    # Direct TTS (no reference)
    [
        processor.build_user_message(text=text_1)
    ],
    [
        processor.build_user_message(text=text_2)
    ],
    # Pinyin or IPA input
    [
        processor.build_user_message(text=text_3)
    ],
    [
        processor.build_user_message(text=text_4)
    ],
    [
        processor.build_user_message(text=text_5)
    ],
    [
        processor.build_user_message(text=text_6)
    ],
    # Voice cloning (with reference)
    [
        processor.build_user_message(text=text_1, reference=[ref_audio_1])
    ],
    [
        processor.build_user_message(text=text_2, reference=[ref_audio_2])
    ],
]



model = AutoModel.from_pretrained(
    pretrained_model_name_or_path,
    trust_remote_code=True,
    attn_implementation="sdpa",
    torch_dtype=dtype,
).to(device)
model.eval()

generation_config = initial_config(processor.tokenizer, pretrained_model_name_or_path)
generation_config.n_vq_for_inference = model.channels - 1
generation_config.do_samples = [True] * model.channels
generation_config.layers = [
    {
        "repetition_penalty": 1.0, 
        "temperature": 1.5, 
        "top_p": 1.0, 
        "top_k": 50
    }
] + [ 
    {
        "repetition_penalty": 1.1, 
        "temperature": 1.0, 
        "top_p": 0.95,
        "top_k": 50
    }
] * (model.channels - 1) 

batch_size = 1

messages = []
save_dir = Path(f"inference_root_moss_tts_local_transformer_generation")
save_dir.mkdir(exist_ok=True, parents=True)
sample_idx = 0
with torch.no_grad():
    for start in range(0, len(conversations), batch_size):
        batch_conversations = conversations[start : start + batch_size]
        batch = processor(batch_conversations, mode="generation")
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config
        )

        for message in processor.decode(outputs):
            audio = message.audio_codes_list[0]
            out_path = save_dir / f"sample{sample_idx}.wav"
            sample_idx += 1
            torchaudio.save(out_path, audio.unsqueeze(0), processor.model_config.sampling_rate)
```

### 7.2 Continuation + Prefix Audio

```python
import os
from pathlib import Path
import torch
import torchaudio
from transformers import AutoModel, AutoProcessor, GenerationConfig
# Disable the broken cuDNN SDPA backend
torch.backends.cuda.enable_cudnn_sdp(False)
# Keep these enabled as fallbacks
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

class DelayGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layers = kwargs.get("layers", [{} for _ in range(32)])
        self.do_samples = kwargs.get("do_samples", None)
        self.n_vq_for_inference = 32
        
def initial_config(tokenizer, model_name_or_path):
    generation_config = DelayGenerationConfig.from_pretrained(model_name_or_path)
    generation_config.pad_token_id = tokenizer.pad_token_id
    generation_config.eos_token_id = 151653
    generation_config.max_new_tokens = 1000000
    generation_config.temperature = 1.0
    generation_config.top_p = 0.95
    generation_config.top_k = 100
    generation_config.repetition_penalty = 1.1
    generation_config.use_cache = True
    generation_config.do_sample = False
    return generation_config


pretrained_model_name_or_path = "OpenMOSS-Team/MOSS-TTS"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

processor = AutoProcessor.from_pretrained(
    pretrained_model_name_or_path,
    trust_remote_code=True,
)
processor.audio_tokenizer = processor.audio_tokenizer.to(device)

text_1 = """亲爱的你，
你好呀。

今天，我想用最认真、最温柔的声音，对你说一些重要的话。
这些话，像一颗小小的星星，希望能在你的心里慢慢发光。

首先，我想祝你——
每天都能平平安安、快快乐乐。

希望你早上醒来的时候，
窗外有光，屋子里很安静，
你的心是轻轻的，没有着急，也没有害怕。
"""

ref_text_1 = "太阳系八大行星之一。"
ref_audio_1 = "https://speech-demo.oss-cn-shanghai.aliyuncs.com/moss_tts_demo/tts_readme_demo/reference_zh.wav"

conversations = [
    # Continuatoin only
    [
        processor.build_user_message(text=ref_text_1 + text_1),
        processor.build_assistant_message(audio_codes_list=[ref_audio_1])
    ],
]

model = AutoModel.from_pretrained(
    pretrained_model_name_or_path,
    trust_remote_code=True,
    attn_implementation="sdpa",
    torch_dtype=dtype,
).to(device)
model.eval()

generation_config = initial_config(processor.tokenizer, pretrained_model_name_or_path)
generation_config.n_vq_for_inference = model.channels - 1
generation_config.do_samples = [True] * model.channels
generation_config.layers = [
    {
        "repetition_penalty": 1.0, 
        "temperature": 1.5, 
        "top_p": 1.0, 
        "top_k": 50
    }
] + [ 
    {
        "repetition_penalty": 1.1, 
        "temperature": 1.0, 
        "top_p": 0.95,
        "top_k": 50
    }
] * (model.channels - 1) 


batch_size = 1

messages = []
save_dir = Path("inference_root_moss_tts_local_transformer_continuation")
save_dir.mkdir(exist_ok=True, parents=True)
sample_idx = 0
with torch.no_grad():
    for start in range(0, len(conversations), batch_size):
        batch_conversations = conversations[start : start + batch_size]
        batch = processor(batch_conversations, mode="continuation")
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config
        )

        for message in processor.decode(outputs):
            audio = message.audio_codes_list[0]
            out_path = save_dir / f"sample{sample_idx}.wav"
            sample_idx += 1
            torchaudio.save(out_path, audio.unsqueeze(0), processor.model_config.sampling_rate)
```
