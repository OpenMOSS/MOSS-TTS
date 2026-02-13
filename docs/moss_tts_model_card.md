# MOSS-TTS Model Card

**MOSS-TTS** is a next-generation, production-grade TTS foundation model focused on **voice cloning**, **ultra-long stable speech generation**, **token-level duration control**, **multilingual & code-switched synthesis**, and **fine-grained Pinyin/phoneme-level pronunciation control**. It is built on a clean autoregressive discrete-token recipe that emphasizes high-quality audio tokenization, large-scale diverse pre-training data, and efficient discrete token modeling.



## 1. Overview

### 1.1 TTS Family Positioning
MOSS-TTS is the **flagship base model** in our open-source **TTS Family**. It is designed as a production-ready synthesis backbone that can serve as the primary high-quality engine for scalable voice applications, and as a strong research baseline for controllable TTS and discrete audio token modeling.

**Design goals**
- **Production readiness**: robust voice cloning with stable, on-brand speaker identity at scale
- **Controllability**: duration and pronunciation controls that integrate into real workflows
- **Long-form stability**: consistent identity and delivery for extended narration
- **Multilingual coverage**: multilingual and code-switched synthesis as first-class capabilities



### 1.2 Key Capabilities

MOSS-TTS delivers state-of-the-art quality while providing the fine-grained controllability and long-form stability required for production-grade voice applications, from zero-shot cloning and hour-long narration to token- and phoneme-level control across multilingual and code-switched speech.

* **State-of-the-art evaluation performance** — top-tier objective and subjective results across standard TTS benchmarks and in-house human preference testing, validating both fidelity and naturalness.
* **Zero-shot Voice Cloning (Voice Clone)** — clone a target speaker’s timbre (and part of speaking style) from short reference audio, without speaker-specific fine-tuning.
* **Ultra-long Speech Generation (up to 1 hour)** — support continuous long-form speech generation for up to one hour in a single run, designed for extended narration and long-session content creation.
* **Token-level Duration Control** — control pacing, rhythm, pauses, and speaking rate at token resolution for precise alignment and expressive delivery.
* **Phoneme-level Pronunciation Control** — supports:

  * pure **Pinyin** input
  * pure **IPA** phoneme input
  * mixed **Chinese / English / Pinyin / IPA** input in any combination
* **Multilingual support** — high-quality multilingual synthesis with robust generalization across languages and accents.
* **Code-switching** — natural mixed-language generation within a single utterance (e.g., Chinese–English), with smooth transitions, consistent speaker identity, and pronunciation-aware rendering on both sides of the switch.



### 1.3 Model Architecture

MOSS-TTS includes **two complementary architectures**, both trained and released to explore different performance/latency tradeoffs and to support downstream research.

**Architecture A: Delay Pattern (MossTTSDelay)**
- Single Transformer backbone with **(n_vq + 1) heads**.
- Uses **delay scheduling** for multi-codebook audio tokens.
- Strong long-context stability, efficient inference, and production-friendly behavior.

**Architecture B: Global Latent + Local Transformer (MossTTSLocal)**
- Backbone produces a **global latent** per time step.
- A lightweight **Local Transformer** emits a token block per step.
- **Streaming-friendly** with simpler alignment (no delay scheduling).

**Why train both?**
- **Exploration of architectural potential** and validation across multiple generation paradigms.
- **Different tradeoffs**: Delay pattern tends to be faster and more stable for long-form synthesis; Local is smaller and excels on objective benchmarks.
- **Open-source value**: two strong baselines for research, ablation, and downstream innovation.

For full details, see:
- **`moss_tts_delay/README.md`**
- **`moss_tts_local/README.md`**



### 1.4 Released Models

| Model | Description |
|---|---|
| **MossTTSDelay-8B** | **Recommended for production**. Faster inference, stronger long-context stability, and robust voice cloning quality. Best for large-scale deployment and long-form narration. |
| **MossTTSLocal-1.7B** | **Recommended for evaluation and research**. Smaller model size with SOTA objective metrics. Great for quick experiments, ablations, and academic studies. |

**Recommended decoding hyperparameters (per model)**

| Model | audio_temperature | audio_top_p | audio_top_k | audio_repetition_penalty |
|---|---:|---:|---:|---:|
| **MossTTSDelay-8B** | 1.7 | 0.8 | 25 | 1.0 |
| **MossTTSLocal-1.7B** | 1.0 | 0.95 | 50 | 1.1 |




## 2. Quick Start

> Tip: For production usage, prioritize **MossTTSDelay-8B**. The examples below use this model; **MossTTSLocal-1.7B** supports the same API, and a practical walkthrough is available in [moss_tts_local/README.md](../moss_tts_local/README.md).

MOSS-TTS provides a convenient `generate` interface for rapid usage. The examples below cover:
1. Direct generation (Chinese / English / Pinyin / IPA)
2. Voice cloning
3. Duration control

```python
from pathlib import Path
import importlib.util
import torch
import torchaudio
from transformers import AutoModel, AutoProcessor
# Disable the broken cuDNN SDPA backend
torch.backends.cuda.enable_cudnn_sdp(False)
# Keep these enabled as fallbacks
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)


pretrained_model_name_or_path = "OpenMOSS-Team/MOSS-TTS"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

def resolve_attn_implementation() -> str:
    # Prefer FlashAttention 2 when package + device conditions are met.
    if (
        device == "cuda"
        and importlib.util.find_spec("flash_attn") is not None
        and dtype in {torch.float16, torch.bfloat16}
    ):
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return "flash_attention_2"

    # CUDA fallback: use PyTorch SDPA kernels.
    if device == "cuda":
        return "sdpa"

    # CPU fallback.
    return "eager"


attn_implementation = resolve_attn_implementation()
print(f"[INFO] Using attn_implementation={attn_implementation}")

processor = AutoProcessor.from_pretrained(
    pretrained_model_name_or_path,
    trust_remote_code=True,
)
processor.audio_tokenizer = processor.audio_tokenizer.to(device)

text_1 = "亲爱的你，\n你好呀。\n\n今天，我想用最认真、最温柔的声音，对你说一些重要的话。\n这些话，像一颗小小的星星，希望能在你的心里慢慢发光。\n\n首先，我想祝你——\n每天都能平平安安、快快乐乐。\n\n希望你早上醒来的时候，\n窗外有光，屋子里很安静，\n你的心是轻轻的，没有着急，也没有害怕。\n\n希望你吃饭的时候胃口很好，\n走路的时候脚步稳稳，\n晚上睡觉的时候，能做一个又一个甜甜的梦。\n\n我希望你能一直保持好奇心。\n对世界充满问题，\n对天空、星星、花草、书本和故事感兴趣。\n当你问“为什么”的时候，\n希望总有人愿意认真地听你说话。\n\n我也希望你学会温柔。\n温柔地对待朋友，\n温柔地对待小动物，\n也温柔地对待自己。\n\n如果有一天你犯了错，\n请不要太快责怪自己，\n因为每一个认真成长的人，\n都会在路上慢慢学会更好的方法。\n\n愿你拥有勇气。\n当你站在陌生的地方时，\n当你第一次举手发言时，\n当你遇到困难、感到害怕的时候，\n希望你能轻轻地告诉自己：\n“我可以试一试。”\n\n就算没有一次成功，也没有关系。\n失败不是坏事，\n它只是告诉你，你正在努力。\n\n我希望你学会分享快乐。\n把开心的事情告诉别人，\n把笑声送给身边的人，\n因为快乐被分享的时候，\n会变得更大、更亮。\n\n如果有一天你感到难过，\n我希望你知道——\n难过并不丢脸，\n哭泣也不是软弱。\n\n愿你能找到一个安全的地方，\n慢慢把心里的话说出来，\n然后再一次抬起头，看见希望。\n\n我还希望你能拥有梦想。\n这个梦想也许很大，\n也许很小，\n也许现在还说不清楚。\n\n没关系。\n梦想会和你一起长大，\n在时间里慢慢变得清楚。\n\n最后，我想送你一个最最重要的祝福：\n\n愿你被世界温柔对待，\n也愿你成为一个温柔的人。\n\n愿你的每一天，\n都值得被记住，\n都值得被珍惜。\n\n亲爱的你，\n请记住，\n你是独一无二的，\n你已经很棒了，\n而你的未来，\n一定会慢慢变得闪闪发光。\n\n祝你健康、勇敢、幸福，\n祝你永远带着笑容向前走。"
text_2 = "We stand on the threshold of the AI era.\nArtificial intelligence is no longer just a concept in laboratories, but is entering every industry, every creative endeavor, and every decision. It has learned to see, hear, speak, and think, and is beginning to become an extension of human capabilities. AI is not about replacing humans, but about amplifying human creativity, making knowledge more equitable, more efficient, and allowing imagination to reach further. A new era, jointly shaped by humans and intelligent systems, has arrived."
text_3 = "nin2 hao3，qing3 wen4 nin2 lai2 zi4 na3 zuo4 cheng2 shi4？"
text_4 = "nin2 hao3，qing4 wen3 nin2 lai2 zi4 na4 zuo3 cheng4 shi3？"
text_5 = "您好，请问您来自哪 zuo4 cheng2 shi4？"
text_6 = "/həloʊ, meɪ aɪ æsk wɪtʃ sɪti juː ɑːr frʌm?/"

# Use audio from ./assets/audio to avoid downloading from the cloud.
ref_audio_1 = "https://speech-demo.oss-cn-shanghai.aliyuncs.com/moss_tts_demo/tts_readme_demo/reference_zh.wav"
ref_audio_2 = "https://speech-demo.oss-cn-shanghai.aliyuncs.com/moss_tts_demo/tts_readme_demo/reference_en.m4a"

conversations = [
    # Direct TTS (no reference)
    [processor.build_user_message(text=text_1)],
    [processor.build_user_message(text=text_2)],
    # Pinyin or IPA input
    [processor.build_user_message(text=text_3)],
    [processor.build_user_message(text=text_4)],
    [processor.build_user_message(text=text_5)],
    [processor.build_user_message(text=text_6)],
    # Voice cloning (with reference)
    [processor.build_user_message(text=text_1, reference=[ref_audio_1])],
    [processor.build_user_message(text=text_2, reference=[ref_audio_2])],
    # Duration control
    [processor.build_user_message(text=text_2, tokens=325)],
    [processor.build_user_message(text=text_2, tokens=600)],
]

model = AutoModel.from_pretrained(
    pretrained_model_name_or_path,
    trust_remote_code=True,
    attn_implementation=attn_implementation,
    torch_dtype=dtype,
).to(device)
model.eval()

batch_size = 1

save_dir = Path("inference_root")
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
            max_new_tokens=4096,
        )

        for message in processor.decode(outputs):
            audio = message.audio_codes_list[0]
            out_path = save_dir / f"sample{sample_idx}.wav"
            sample_idx += 1
            torchaudio.save(out_path, audio.unsqueeze(0), processor.model_config.sampling_rate)

```

### Continuation + Voice Cloning (Prefix Audio + Text)

MOSS-TTS supports continuation-based cloning: provide a prefix audio clip in the assistant message, and make sure the **prefix transcript** is included in the text. The model continues in the same speaker identity and style.

```python
from pathlib import Path
import importlib.util
import torch
import torchaudio
from transformers import AutoModel, AutoProcessor
# Disable the broken cuDNN SDPA backend
torch.backends.cuda.enable_cudnn_sdp(False)
# Keep these enabled as fallbacks
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)


pretrained_model_name_or_path = "OpenMOSS-Team/MOSS-TTS"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

def resolve_attn_implementation() -> str:
    # Prefer FlashAttention 2 when package + device conditions are met.
    if (
        device == "cuda"
        and importlib.util.find_spec("flash_attn") is not None
        and dtype in {torch.float16, torch.bfloat16}
    ):
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return "flash_attention_2"

    # CUDA fallback: use PyTorch SDPA kernels.
    if device == "cuda":
        return "sdpa"

    # CPU fallback.
    return "eager"


attn_implementation = resolve_attn_implementation()
print(f"[INFO] Using attn_implementation={attn_implementation}")

processor = AutoProcessor.from_pretrained(
    pretrained_model_name_or_path,
    trust_remote_code=True
)
processor.audio_tokenizer = processor.audio_tokenizer.to(device)

text_1 = "亲爱的你，\n你好呀。\n\n今天，我想用最认真、最温柔的声音，对你说一些重要的话。\n这些话，像一颗小小的星星，希望能在你的心里慢慢发光。\n\n首先，我想祝你——\n每天都能平平安安、快快乐乐。\n\n希望你早上醒来的时候，\n窗外有光，屋子里很安静，\n你的心是轻轻的，没有着急，也没有害怕。\n\n希望你吃饭的时候胃口很好，\n走路的时候脚步稳稳，\n晚上睡觉的时候，能做一个又一个甜甜的梦。\n\n我希望你能一直保持好奇心。\n对世界充满问题，\n对天空、星星、花草、书本和故事感兴趣。\n当你问“为什么”的时候，\n希望总有人愿意认真地听你说话。\n\n我也希望你学会温柔。\n温柔地对待朋友，\n温柔地对待小动物，\n也温柔地对待自己。\n\n如果有一天你犯了错，\n请不要太快责怪自己，\n因为每一个认真成长的人，\n都会在路上慢慢学会更好的方法。\n\n愿你拥有勇气。\n当你站在陌生的地方时，\n当你第一次举手发言时，\n当你遇到困难、感到害怕的时候，\n希望你能轻轻地告诉自己：\n“我可以试一试。”\n\n就算没有一次成功，也没有关系。\n失败不是坏事，\n它只是告诉你，你正在努力。\n\n我希望你学会分享快乐。\n把开心的事情告诉别人，\n把笑声送给身边的人，\n因为快乐被分享的时候，\n会变得更大、更亮。\n\n如果有一天你感到难过，\n我希望你知道——\n难过并不丢脸，\n哭泣也不是软弱。\n\n愿你能找到一个安全的地方，\n慢慢把心里的话说出来，\n然后再一次抬起头，看见希望。\n\n我还希望你能拥有梦想。\n这个梦想也许很大，\n也许很小，\n也许现在还说不清楚。\n\n没关系。\n梦想会和你一起长大，\n在时间里慢慢变得清楚。\n\n最后，我想送你一个最最重要的祝福：\n\n愿你被世界温柔对待，\n也愿你成为一个温柔的人。\n\n愿你的每一天，\n都值得被记住，\n都值得被珍惜。\n\n亲爱的你，\n请记住，\n你是独一无二的，\n你已经很棒了，\n而你的未来，\n一定会慢慢变得闪闪发光。\n\n祝你健康、勇敢、幸福，\n祝你永远带着笑容向前走。"
text_2 = "We stand on the threshold of the AI era.\nArtificial intelligence is no longer just a concept in laboratories, but is entering every industry, every creative endeavor, and every decision. It has learned to see, hear, speak, and think, and is beginning to become an extension of human capabilities. AI is not about replacing humans, but about amplifying human creativity, making knowledge more equitable, more efficient, and allowing imagination to reach further. A new era, jointly shaped by humans and intelligent systems, has arrived."
ref_text_1 = "太阳系八大行星之一。"
ref_text_2 = "But I really can't complain about not having a normal college experience to you."
ref_audio_1 = "https://speech-demo.oss-cn-shanghai.aliyuncs.com/moss_tts_demo/tts_readme_demo/reference_zh.wav"
ref_audio_2 = "https://speech-demo.oss-cn-shanghai.aliyuncs.com/moss_tts_demo/tts_readme_demo/reference_en.m4a"

conversations = [
    # Continuatoin only
    [
        processor.build_user_message(text=ref_text_1 + text_1),
        processor.build_assistant_message(audio_codes_list=[ref_audio_1])
    ],
    # Continuation with voice cloning
    [
        processor.build_user_message(text=ref_text_2 + text_2, reference=[ref_audio_2]),
        processor.build_assistant_message(audio_codes_list=[ref_audio_2])
    ],
]

model = AutoModel.from_pretrained(
    pretrained_model_name_or_path,
    trust_remote_code=True,
    attn_implementation=attn_implementation,
    torch_dtype=dtype,
).to(device)
model.eval()

batch_size = 1

save_dir = Path("inference_root")
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
            max_new_tokens=4096,
        )

        for message in processor.decode(outputs):
            audio = message.audio_codes_list[0]
            out_path = save_dir / f"sample{sample_idx}.wav"
            sample_idx += 1
            torchaudio.save(out_path, audio.unsqueeze(0), processor.model_config.sampling_rate)

```



### Input Types

**UserMessage**

| Field | Type | Required | Description |
|---|---|---:|---|
| `text` | `str` | Yes | Text to synthesize. Supports Chinese, English, German, French, Spanish, Japanese, Korean, etc. Can mix raw text with Pinyin or IPA for pronunciation control. |
| `reference` | `List[str]` | No | Reference audio for voice cloning. For current MOSS-TTS, **one audio** is expected in the list. |
| `tokens` | `int` | No | Expected number of audio tokens. **1s ≈ 12.5 tokens**. |

**AssistantMessage**

| Field | Type | Required | Description |
|---|---|---:|---|
| `audio_codes_list` | `List[str]` | Only for continuation | Prefix audio for continuation-based cloning. Use audio file paths or URLs. |



### Generation Hyperparameters

| Parameter | Type | Default | Description |
|---|---|---:|---|
| `max_new_tokens` | `int` | — | Controls total generated audio tokens. Use duration rule: **1s ≈ 12.5 tokens**. |
| `audio_temperature` | `float` | 1.7 | Higher values increase variation; lower values stabilize prosody. |
| `audio_top_p` | `float` | 0.8 | Nucleus sampling cutoff. Lower values are more conservative. |
| `audio_top_k` | `int` | 25 | Top-K sampling. Lower values tighten sampling space. |
| `audio_repetition_penalty` | `float` | 1.0 | >1.0 discourages repeating patterns. |

> Note: MOSS-TTS is a pretrained base model and is **sensitive to decoding hyperparameters**. See **Released Models** for recommended defaults.



### Pinyin Input

Use tone-numbered Pinyin such as `ni3 hao3 wo3 men1`. You can convert Chinese text with [pypinyin](https://github.com/mozillazg/python-pinyin), then adjust tones for pronunciation control.

```python
import re
from pypinyin import pinyin, Style

CN_PUNCT = r"，。！？；：、（）“”‘’"


def fix_punctuation_spacing(s: str) -> str:
    s = re.sub(rf"\s+([{CN_PUNCT}])", r"\1", s)
    s = re.sub(rf"([{CN_PUNCT}])\s+", r"\1", s)
    return s


def zh_to_pinyin_tone3(text: str, strict: bool = True) -> str:
    result = pinyin(
        text,
        style=Style.TONE3,
        heteronym=False,
        strict=strict,
        errors="default",
    )

    s = " ".join(item[0] for item in result)
    return fix_punctuation_spacing(s)

text = zh_to_pinyin_tone3("您好，请问您来自哪座城市？")
print(text)

# Expected: nin2 hao3，qing3 wen4 nin2 lai2 zi4 na3 zuo4 cheng2 shi4？
# Try: nin2 hao3，qing4 wen3 nin2 lai2 zi4 na4 zuo3 cheng4 shi3？
```



### IPA Input

Use `/.../` to wrap IPA sequences so they are distinct from normal text. You can use [DeepPhonemizer](https://github.com/spring-media/DeepPhonemizer) to convert English paragraphs or words into IPA sequences.

```python
from dp.phonemizer import Phonemizer

# Download a phonemizer checkpoint from https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/DeepPhonemizer/en_us_cmudict_ipa_forward.pt
model_path = "<path-to-phonemizer-checkpoint>"
phonemizer = Phonemizer.from_checkpoint(model_path)

english_texts = "Hello, may I ask which city you are from?"
phoneme_outputs = phonemizer(
    english_texts,
    lang="en_us",
    batch_size=8
)
model_input_text = f"/{phoneme_outputs}/"
print(model_input_text)

# Expected: /həloʊ, meɪ aɪ æsk wɪtʃ sɪti juː ɑːr frʌm?/
```



## 3. Evaluation
MOSS-TTS achieved state-of-the-art results on the open-source zero-shot TTS benchmark Seed-TTS-eval, not only surpassing all open-source models but also rivaling the most powerful closed-source models.

| Model | Params | Open-source | EN WER (%) ↓ | EN SIM (%) ↑ | ZH CER (%) ↓ | ZH SIM (%) ↑ |
|---|---:|:---:|---:|---:|---:|---:|
| DiTAR | 0.6B | ❌ | 1.69 | 73.5 | 1.02 | 75.3 |
| FishAudio-S1 | 4B | ❌ | 1.72 | 62.57 | 1.22 | 72.1 |
| Seed-TTS |  | ❌ | 2.25 | 76.2 | 1.12 | 79.6 |
| MiniMax-Speech |  | ❌ | 1.65 | 69.2 | 0.83 | 78.3 |
|  |  |  |  |  |  |  |
| CosyVoice | 0.3B | ✅ | 4.29 | 60.9 | 3.63 | 72.3 |
| CosyVoice2 | 0.5B | ✅ | 3.09 | 65.9 | 1.38 | 75.7 |
| CosyVoice3 | 0.5B | ✅ | 2.02 | 71.8 | 1.16 | 78 |
| CosyVoice3 | 1.5B | ✅ | 2.22 | 72 | 1.12 | 78.1 |
| F5-TTS | 0.3B | ✅ | 2 | 67 | 1.53 | 76 |
| SparkTTS | 0.5B | ✅ | 3.14 | 57.3 | 1.54 | 66 |
| FireRedTTS | 0.5B | ✅ | 3.82 | 46 | 1.51 | 63.5 |
| FireRedTTS-2 | 1.5B | ✅ | 1.95 | 66.5 | 1.14 | 73.6 |
| Qwen2.5-Omni | 7B | ✅ | 2.72 | 63.2 | 1.7 | 75.2 |
| FishAudio-S1-mini | 0.5B | ✅ | 1.94 | 55 | 1.18 | 68.5 |
| IndexTTS2 | 1.5B | ✅ | 2.23 | 70.6 | 1.03 | 76.5 |
| VibeVoice | 1.5B | ✅ | 3.04 | 68.9 | 1.16 | 74.4 |
| HiggsAudio-v2 | 3B | ✅ | 2.44 | 67.7 | 1.5 | 74 |
| VoxCPM | 0.5B | ✅ | 1.85 | 72.9 | **0.93** | 77.2 |
| Qwen3-TTS | 0.6B | ✅ | 1.68 | 70.39 | 1.23 | 76.4 |
| Qwen3-TTS | 1.7B | ✅ | **1.5** | 71.45 | 1.33 | 76.72 |
|  |  |  |  |  |  |  |
| MossTTSDelay | 8B | ✅ | 1.79 | 71.46 | 1.32 | 77.05 |
| MossTTSLocal | 1.7B | ✅ | 1.85 | **73.42** | 1.2 | **78.82** |
