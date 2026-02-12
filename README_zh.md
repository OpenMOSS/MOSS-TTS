# MOSS-TTS 家族



<br>

<p align="center" style="display:flex; justify-content:center; align-items:center; gap:24px;">
  <img src="./assets/OpenMOSS_Logo.png" height="80" style="display:block; transform: translateY(0px);" />
  <img src="./assets/mosi-logo.png" height="50" style="display:block; transform: translateY(-8px);" />
</p>



<div align="center">
  <a href="https://huggingface.co/collections/OpenMOSS-Team/moss-tts"><img src="https://img.shields.io/badge/Huggingface-Models-orange?logo=huggingface&amp"></a>
  <a href="https://modelscope.cn/collections/OpenMOSS-Team/MOSS-TTS"><img src="https://img.shields.io/badge/ModelScope-Models-lightgrey?logo=modelscope&amp"></a>
  <a href="https://mosi.cn/#models"><img src="https://img.shields.io/badge/Blog-View-blue?logo=internet-explorer&amp"></a>
  <a href="https://github.com/OpenMOSS/MOSS-TTS"><img src="https://img.shields.io/badge/Arxiv-Coming%20soon-red?logo=arxiv&amp"></a>

  <a href="https://studio.mosi.cn"><img src="https://img.shields.io/badge/AIStudio-Try-green?logo=internet-explorer&amp"></a>
  <a href="https://studio.mosi.cn/docs/moss-tts"><img src="https://img.shields.io/badge/API-Docs-00A3FF?logo=fastapi&amp"></a>
  <a href="https://x.com/Open_MOSS"><img src="https://img.shields.io/badge/Twitter-Follow-black?logo=x&amp"></a>
  <a href="https://discord.gg/fvm5TaWjU3"><img src="https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&amp"></a>
  
</div>


[English](README.md) | [简体中文](README_zh.md)


MOSS‑TTS 家族是由 [MOSI.AI](https://mosi.cn/#hero) 与 [OpenMOSS 团队](https://www.open-moss.com/) 推出的开源 **语音与声音生成模型家族**。该系列面向 **高保真**、**高表现力** 与 **复杂真实场景** 设计，覆盖稳定长文本语音、多说话人对话、音色/角色设计、环境音效以及实时流式 TTS 等能力。

<a id="news"></a>
## 新闻
* 2026.2.10：🎉🎉🎉 我们已发布 [MOSS-TTS Family](https://huggingface.co/collections/OpenMOSS-Team/moss-tts)。更多详情请查看我们的 [Blog](https://mosi.cn/#models)！

## 演示

<div align="center">
  <video src="https://gist.github.com/user-attachments/assets/fdce9f66-20ec-45e8-9615-89606ae2fbe8" width="70%" poster=""> </video>
</div>


## 目录

- [介绍](#introduction)
- [模型架构](#architecture)
- [已发布模型](#released-models)
- [支持的语言](#supported-languages)
- [快速开始](#quickstart)
  - [环境准备](#environment-setup)
  - [（可选）安装 FlashAttention 2](#optional-install-flashattention-2)
  - [基础用法](#moss-tts-basic-usage)
- [评测](#evaluation)
  - [MOSS-TTS 评测](#eval-moss-tts)
  - [MOSS-TTSD 评测](#eval-moss-ttsd)
  - [MOSS-VoiceGenerator 评测](#eval-moss-voicegenerator)
- [语音编解码器](#audio-tokenizer)
  - [介绍](#audio-tokenizer-intro)
  - [模型权重](#model-weights)
  - [重建质量客观评测](#重建质量客观评测)


<a id="introduction"></a>
## 介绍

<p align="center">
  <img src="./assets/moss_tts_family.jpeg" width="85%" />
</p>

当一段音频需要 **听起来像真实的人类**、**准确发音**、**在不同内容间切换说话风格**、**稳定持续数十分钟**，并且 **支持对话、角色扮演与实时交互** 时，单一 TTS 模型往往不足以胜任。**MOSS‑TTS 家族**将工作流拆分为 5 个可独立使用、亦可组合成完整管线的量产级模型。

- **MOSS‑TTS**：MOSS‑TTS 是家族中的旗舰量产级 TTS 基础模型，**核心能力是高保真以及最优性能的零样本语音克隆**，支持**长文本长语音生成**、**拼音、音标与时长精细控制**，以及**多语种/中英混合合成**。它可作为大规模旁白、配音和语音产品的核心底座。
- **MOSS‑TTSD**：MOSS‑TTSD 是对话语音生成模型，用于生成高表现力、多说话人、超长连续对话的音频。本次我们更新了全新的**v1.0版本**，相比于0.7版本，它在音色相似度，说话人切换准确率，词错误率等**客观指标上取得了业界最优的性能**，在竞技场主观评测中，也**战胜了豆包、Gemini2.5-pro**等顶尖闭源模型。
- **MOSS‑VoiceGenerator**：MOSS‑VoiceGenerator 是开源音色设计模型，可从文本风格指令直接生成多样的说话人音色或风格，**无需参考音频**。它统一音色设计、风格控制与内容合成，可独立创作，也可作为下游 TTS 的音色设计层。模型性能在**竞技场评分上超过了其余等顶尖音色设计模型**。
- **MOSS‑TTS‑Realtime**：MOSS‑TTS‑Realtime 是面向实时语音智能体的多轮上下文感知实时 TTS 模型。它结合多轮对话中的文本与历史语音信号进行低时延增量合成，使多轮回复保持连贯、自然且音色一致。**非常适合搭配文本模型构建低时延语音智能体**。
- **MOSS‑SoundEffect**：MOSS‑SoundEffect 是面向内容制作的**音效生成**模型，具备广泛类别覆盖与可控时长能力。它能根据文本指令生成自然环境、城市场景、生物、人类动作与类音乐片段等音频，适用于影视、游戏、交互体验和数据合成。

<a id="architecture"></a>
## 模型架构

我们在统一训练/评测框架下将 **MossTTSDelay** 与 **MossTTSLocal** 作为互补基线：**Delay** 更强调长上下文稳定性、推理速度与工程可用性，**Local** 更强调轻量灵活和面向流式场景的客观指标表现。二者共同提供可复现、可对比的落地与研究参考。

**MossTTSRealtime** 不是第三个对比基线，而是面向语音智能体的能力型设计。它同时利用历史文本与用户语音声学信息建模多轮上下文，以低时延流式合成保持回复连贯和音色一致。


| 架构  | 核心机制 | 架构细节 |
|---|---|---|
| `MossTTSDelay` |  多头并行 RVQ 预测，结合延迟模式调度 | [![Arch Details](https://img.shields.io/badge/Model%20Card-View-blue?logo=markdown)](moss_tts_delay/README.md) |
| `MossTTSLocal` | 基于深度 Transformer 的时间同步 RVQ 模块 | [![Arch Details](https://img.shields.io/badge/Model%20Card-View-blue?logo=markdown)](moss_tts_local/README.md) |
| `MossTTSRealtime` | 用于实时合成的分层文本-音频输入 | [![Arch Details](https://img.shields.io/badge/Model%20Card-View-blue?logo=markdown)](moss_tts_realtime/README.md) |

<a id="released-models"></a>
## 模型概览

| Model | Architecture | Size | Model Card | Hugging Face | ModelScope |
|---|---|---:|---|---|---|
| **MOSS-TTS** | `MossTTSDelay` | 8B | [![Model Card](https://img.shields.io/badge/Model%20Card-View-blue?logo=markdown)](docs/moss_tts_model_card.md) | [![Hugging Face](https://img.shields.io/badge/Huggingface-Model-orange?logo=huggingface)](https://huggingface.co/OpenMOSS-Team/MOSS-TTS) | [![ModelScope](https://img.shields.io/badge/ModelScope-Model-lightgrey?logo=modelscope)](https://modelscope.cn/models/openmoss/MOSS-TTS) |
|  | `MossTTSLocal` | 1.7B | [![Model Card](https://img.shields.io/badge/Model%20Card-View-blue?logo=markdown)](docs/moss_tts_model_card.md) | [![Hugging Face](https://img.shields.io/badge/Huggingface-Model-orange?logo=huggingface)](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer) | [![ModelScope](https://img.shields.io/badge/ModelScope-Model-lightgrey?logo=modelscope)](https://modelscope.cn/models/openmoss/MOSS-TTS-Local-Transformer) |
| **MOSS‑TTSD‑V1.0** | `MossTTSDelay` | 8B | [![Model Card](https://img.shields.io/badge/Model%20Card-View-blue?logo=markdown)](docs/moss_ttsd_model_card.md) | [![Hugging Face](https://img.shields.io/badge/Huggingface-Model-orange?logo=huggingface)](https://huggingface.co/OpenMOSS-Team/MOSS-TTSD-v1.0) | [![ModelScope](https://img.shields.io/badge/ModelScope-Model-lightgrey?logo=modelscope)](https://modelscope.cn/models/openmoss/MOSS-TTSD-v1.0) |
| **MOSS‑VoiceGenerator** | `MossTTSDelay` | 1.7B | [![Model Card](https://img.shields.io/badge/Model%20Card-View-blue?logo=markdown)](docs/moss_voice_generator_model_card.md) | [![Hugging Face](https://img.shields.io/badge/Huggingface-Model-orange?logo=huggingface)](https://huggingface.co/OpenMOSS-Team/MOSS-VoiceGenerator) | [![ModelScope](https://img.shields.io/badge/ModelScope-Model-lightgrey?logo=modelscope)](https://modelscope.cn/models/openmoss/MOSS-VoiceGenerator) |
| **MOSS‑SoundEffect** | `MossTTSDelay` | 8B | [![Model Card](https://img.shields.io/badge/Model%20Card-View-blue?logo=markdown)](docs/moss_sound_effect_model_card.md) | [![Hugging Face](https://img.shields.io/badge/Huggingface-Model-orange?logo=huggingface)](https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect) | [![ModelScope](https://img.shields.io/badge/ModelScope-Model-lightgrey?logo=modelscope)](https://modelscope.cn/models/openmoss/MOSS-SoundEffect) |
| **MOSS‑TTS‑Realtime** | `MossTTSRealtime` | 1.7B | [![Model Card](https://img.shields.io/badge/Model%20Card-View-blue?logo=markdown)](docs/moss_tts_realtime_model_card.md) | [![Hugging Face](https://img.shields.io/badge/Huggingface-Model-orange?logo=huggingface)](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Realtime) | [![ModelScope](https://img.shields.io/badge/ModelScope-Model-lightgrey?logo=modelscope)](https://modelscope.cn/models/openmoss/MOSS-TTS-Realtime) |

<a id="supported-languages"></a>

## 支持的语言

MOSS-TT、MOSS-TTSD 和 MOSS-TTS-Realtime 目前支持 **20 种语言**：

| Language | Code | Flag | Language | Code | Flag | Language | Code | Flag |
|---|---|---|---|---|---|---|---|---|
| 中文 | zh | 🇨🇳 | 英语 | en | 🇺🇸 | 德语 | de | 🇩🇪 |
| 西班牙语 | es | 🇪🇸 | 法语 | fr | 🇫🇷 | 日语 | ja | 🇯🇵 |
| 意大利语 | it | 🇮🇹 | 希伯来语 | he | 🇮🇱 | 韩语 | ko | 🇰🇷 |
| 俄语 | ru | 🇷🇺 | 波斯语（法尔西语） | fa | 🇮🇷 | 阿拉伯语 | ar | 🇸🇦 |
| 波兰语 | pl | 🇵🇱 | 葡萄牙语 | pt | 🇵🇹 | 捷克语 | cs | 🇨🇿 |
| 丹麦语 | da | 🇩🇰 | 瑞典语 | sv | 🇸🇪 | 匈牙利语 | hu | 🇭🇺 |
| 希腊语 | el | 🇬🇷 | 土耳其语 | tr | 🇹🇷 |  |  |  |



<a id="quickstart"></a>
## 快速开始

<a id="environment-setup"></a>
### 环境准备

建议使用干净的 Python 环境。

```bash
conda create -n moss-tts python=3.12 -y
conda activate moss-tts
```

安装全部依赖：

```bash
git clone https://github.com/OpenMOSS/MOSS-TTS.git
cd MOSS-TTS
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e .
```
<a id="optional-install-flashattention-2"></a>
#### （可选）安装 FlashAttention 2

如果你的硬件支持，可以安装 FlashAttention 2 以提升速度并降低显存占用。

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e ".[flash-attn]"
```

如果机器内存较小、CPU 核数较多，可以限制并行编译数：

```bash
MAX_JOBS=4 pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e ".[flash-attn]"
```

说明：
- 依赖统一在 `pyproject.toml` 中管理，当前固定了 `torch==2.9.1+cu128` 和 `torchaudio==2.9.1+cu128`。
- 如果 FlashAttention 2 编译失败，可以跳过，直接使用默认 attention 后端。
- FlashAttention 2 仅支持部分 GPU，通常搭配 `torch.float16` 或 `torch.bfloat16` 使用。


<a id="moss-tts-basic-usage"></a>
### MOSS‑TTS 基础用法

如果你更希望使用 Gradio 界面，我们为 4 个主模型提供了对应脚本：

| Model | Script | 
|---|---|
| MOSS-TTS | [clis/moss_tts_app.py](clis/moss_tts_app.py) |
| MOSS-TTSD | [clis/moss_ttsd_app.py](clis/moss_ttsd_app.py) | 
| MOSS-VoiceGenerator | [clis/moss_voice_generator_app.py](clis/moss_voice_generator_app.py) | 
| MOSS-SoundEffect | [clis/moss_sound_effect_app.py](clis/moss_sound_effect_app.py) |

MOSS-TTS-Realtime 的 Gradio demo 请直接参考 [MOSS-TTS-Realtime Model Card](docs/moss_tts_realtime_model_card.md)

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

各模型的完整使用方式请参考对应的 model card。


<a id="evaluation"></a>
## 评测

本节总结 MOSS‑TTS 与 MOSS‑VoiceGenerator 的 **家族级评测亮点**。完整细节请参见各模型的 model card。

<a id="eval-moss-tts"></a>
### MOSS‑TTS 评测
MOSS‑TTS 在开源零样本 TTS 基准 `Seed‑TTS‑eval` 上取得当前最佳结果，超越所有开源模型，并与主流闭源系统相当。

| Model | Params | Open‑source | EN WER (%) ↓ | EN SIM (%) ↑ | ZH CER (%) ↓ | ZH SIM (%) ↑ |
|---|---:|:---:|---:|---:|---:|---:|
| DiTAR | 0.6B | ❌ | 1.69 | 73.5 | 1.02 | 75.3 |
| FishAudio‑S1 | 4B | ❌ | 1.72 | 62.57 | 1.22 | 72.1 |
| Seed‑TTS |  | ❌ | 2.25 | 76.2 | 1.12 | 79.6 |
| MiniMax‑Speech |  | ❌ | 1.65 | 69.2 | 0.83 | 78.3 |
|  |  |  |  |  |  |  |
| CosyVoice | 0.3B | ✅ | 4.29 | 60.9 | 3.63 | 72.3 |
| CosyVoice2 | 0.5B | ✅ | 3.09 | 65.9 | 1.38 | 75.7 |
| CosyVoice3 | 0.5B | ✅ | 2.02 | 71.8 | 1.16 | 78 |
| CosyVoice3 | 1.5B | ✅ | 2.22 | 72 | 1.12 | 78.1 |
| F5‑TTS | 0.3B | ✅ | 2 | 67 | 1.53 | 76 |
| SparkTTS | 0.5B | ✅ | 3.14 | 57.3 | 1.54 | 66 |
| FireRedTTS | 0.5B | ✅ | 3.82 | 46 | 1.51 | 63.5 |
| FireRedTTS‑2 | 1.5B | ✅ | 1.95 | 66.5 | 1.14 | 73.6 |
| Qwen2.5‑Omni | 7B | ✅ | 2.72 | 63.2 | 1.7 | 75.2 |
| FishAudio‑S1‑mini | 0.5B | ✅ | 1.94 | 55 | 1.18 | 68.5 |
| IndexTTS2 | 1.5B | ✅ | 2.23 | 70.6 | 1.03 | 76.5 |
| VibeVoice | 1.5B | ✅ | 3.04 | 68.9 | 1.16 | 74.4 |
| HiggsAudio‑v2 | 3B | ✅ | 2.44 | 67.7 | 1.5 | 74 |
| VoxCPM | 0.5B | ✅ | 1.85 | 72.9 | **0.93** | 77.2 |
| Qwen3‑TTS | 0.6B | ✅ | 1.68 | 70.39 | 1.23 | 76.4 |
| Qwen3‑TTS | 1.7B | ✅ | **1.5** | 71.45 | 1.33 | 76.72 |
|  |  |  |  |  |  |  |
| **MossTTSDelay** | **8B** | ✅ | 1.79 | 71.46 | 1.32 | 77.05 |
| **MossTTSLocal** | **1.7B** | ✅ | 1.85 | **73.42** | 1.2 | **78.82** |

<a id="eval-moss-ttsd"></a>
### MOSS‑TTSD 评测
#### 客观评测
我们使用三个客观指标来评估 MOSS‑TTSD-v1.0 的性能：说话人切换准确性（ACC）、说话人相似度（SIM）和词错误率（WER）。我们对比了 MOSS‑TTSD-v1.0 与多个开源模型和闭源模型的性能，结果如下，MOSS-TTSD-v1.0 均取得了最优或次优性能。

| Model | ZH - SIM | ZH - ACC | ZH - WER | EN - SIM | EN - ACC | EN - WER |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Comparison with Open-Source Models** | | | | | | |
| **MOSS-TTSD-v1.0** | **0.7949** | **0.9587** | **0.0485** | **0.7326** | **0.9626** | 0.0988 |
| MOSS-TTSD-v0.7 | 0.7423 | 0.9391 | 0.0517 | 0.6743 | 0.9266 | 0.1612 |
| Vibevoice 7B | 0.7590 | 0.9222 | 0.0570 | 0.7140 | 0.9554 | **0.0946** |
| Vibevoice 1.5 B | 0.7415 | 0.8798 | 0.0818 | 0.6961 | 0.9353 | 0.1133 |
| FireRedTTS2 | 0.7383 | 0.9022 | 0.0768 | - | - | - |
| Higgs Audio V2 | - | - | - | 0.6860 | 0.9025 | 0.2131 |
| **Comparison with Proprietary Models** | | | | | | |
| **MOSS-TTSD-v1.0 (elevenlabs_voice)** | **0.8165** | **0.9736** | 0.0391 | **0.7304** | **0.9565** | 0.1005 |
| Eleven V3 | 0.6970 | 0.9653 | **0.0363** | 0.6730 | 0.9498 | **0.0824** |
| | | | | | | |
| **MOSS-TTSD-v1.0 (gemini_voice)** | - | - | - | **0.7893** | **0.9655** | 0.0984 |
| gemini-2.5-pro-preview-tts | - | - | - | 0.6786 | 0.9537 | **0.0859** |
| gemini-2.5-flash-preview-tts | - | - | - | 0.7194 | 0.9511 | 0.0871 |
| | | | | | | |
| **MOSS-TTSD-v1.0 (doubao_voice)** | **0.8226** | **0.9630** | 0.0571 | - | - | - |
| Doubao_Podcast | 0.8034 | 0.9606 | **0.0472** | - | - | - |

#### 主观评测
对于开源模型，标注者会从说话人归因准确性、音色相似度、韵律与整体质量等维度对每个样本对进行评分。遵循 LMSYS Chatbot Arena 的方法，我们计算各维度的 Elo 评分与置信区间。
![alt text](assets/VS_Open-Source_Models.jpg)

对于闭源模型，标注者只需在每个样本对中选择整体更偏好的一项，并据此计算胜率。
![alt text](assets/VS_Proprietary_Models.png)


<a id="eval-moss-voicegenerator"></a>
### MOSS‑VoiceGenerator 主观评测
MOSS‑VoiceGenerator 在 **整体偏好**、**指令遵循** 与 **自然度** 上表现出强主观偏好。

<p align="center">
  <img src="./assets/moss_voice_generator_winrate.png" width="70%" />
</p>

<a id="audio-tokenizer"></a>
## 语音编解码器

<a id="audio-tokenizer-intro"></a>
### 介绍
**MOSS-Audio-Tokenizer** 是 MOSS‑TTS 家族的统一离散音频接口，基于 **Cat**（**C**ausal **A**udio **T**okenizer with **T**ransformer）架构——一个 16 亿参数、完全由 Causal Transformer 块构建的“无 CNN”同构音频 tokenizer。

- **统一离散桥接**：为 MOSS‑TTS、MOSS‑TTSD、MOSS‑VoiceGenerator、MOSS‑SoundEffect 与 MOSS‑TTS‑Realtime 提供共享骨干，使家族内音频表示一致。
- **极致压缩与高保真**：将 24kHz 原始音频压缩到 12.5Hz 的极低帧率；采用 32 层残差向量量化（RVQ），支持从 0.125kbps 到 4kbps 的可变码率高保真重建。
- **超大规模通用音频训练**：从零训练，使用 300 万小时多样化数据（语音、音效与音乐），在开源音频 tokenizer 中达到 SOTA 级重建效果。
- **原生流式设计**：纯 Causal Transformer 架构专为可扩展性与低时延流式推理而设计，支持实时生产流程。

如需了解更多配置、进阶用法与评测指标，请访问 [MOSS-Audio-Tokenizer 仓库](https://github.com/OpenMOSS/MOSS-Audio-Tokenizer)。

<p align="center">
  <img src="./assets/arch_moss_audio_tokenizer.png" alt="MOSS Audio Tokenizer 架构示意" width="100%" />
  MOSS Audio Tokenizer 架构图
</p>

<a id="model-weights"></a>
### 模型权重

| Model | Hugging Face | ModelScope |
|:-----:|:------------:|:----------:|
| **MOSS-Audio-Tokenizer** | [![Hugging Face](https://img.shields.io/badge/Huggingface-Model-orange?logo=huggingface)](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer) | [![ModelScope](https://img.shields.io/badge/ModelScope-Model-lightgrey?logo=modelscope)](https://modelscope.cn/models/openmoss/MOSS-Audio-Tokenizer) |

### 重建质量客观评测

我们在 LibriSpeech test-clean 子集上，对比 **MOSS Audio Tokenizer** 与多个开源音频 tokenizer 的 SIM、STOI、PESQ-NB、PESQ-WB 指标，并通过调节 RVQ 码本数量来控制码率。MOSS Audio Tokenizer 在 0–4 kbps 的比特率上的重建质量领先其他开源音频 tokenizer。

<p align="center">
  <img src="./assets/evaluation_moss_audio_tokenizer.png" alt="LibriSpeech objective metrics for audio tokenizers" width="90%" />
</p>


## 证书

MOSS-TTS 家族中的模型使用 Apache License 2.0 许可证。

## 引用

```bibtex
```
## 星标历史数据

[![Star History Chart](https://api.star-history.com/svg?repos=OpenMOSS/MOSS-TTS&type=date&legend=top-left)](https://www.star-history.com/#OpenMOSS/MOSS-TTS&type=date&legend=top-left)
