# MOSS-TTS Family

<br>

<p align="center">
  <img src="./assets/OpenMOSS_Logo.png" height="70" align="middle" />
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="./assets/mosi-logo.png" height="50" align="middle" />
</p>




<div align="center">
  <a href="https://clawhub.ai/luogao2333/moss-tts-voice"><img src="https://img.shields.io/badge/🦞_OpenClaw-Skills-8A2BE2" alt="OpenClaw"></a>
  <a href="https://huggingface.co/collections/OpenMOSS-Team/moss-tts"><img src="https://img.shields.io/badge/Huggingface-Models-orange?logo=huggingface&amp"></a>
  <a href="https://modelscope.cn/collections/OpenMOSS-Team/MOSS-TTS"><img src="https://img.shields.io/badge/ModelScope-Models-lightgrey?logo=modelscope&amp"></a>
  <a href="https://mosi.cn/#models"><img src="https://img.shields.io/badge/Blog-View-blue?logo=internet-explorer&amp"></a>
  <a href="https://github.com/OpenMOSS/MOSS-TTS"><img src="https://img.shields.io/badge/Arxiv-Coming%20soon-red?logo=arxiv&amp"></a>

  <a href="https://studio.mosi.cn"><img src="https://img.shields.io/badge/AIStudio-Try-green?logo=internet-explorer&amp"></a>
  <a href="https://studio.mosi.cn/docs/moss-tts"><img src="https://img.shields.io/badge/API-Docs-00A3FF?logo=fastapi&amp"></a>
  <a href="https://x.com/Open_MOSS"><img src="https://img.shields.io/badge/Twitter-Follow-black?logo=x&amp"></a>
  <a href="https://discord.gg/fvm5TaWjU3"><img src="https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&amp"></a>
  <a href="./assets/wechat.jpg"><img src="https://img.shields.io/badge/WeChat-Join-07C160?logo=wechat&amp;logoColor=white" alt="WeChat"></a>
</div>


[English](README.md) | [简体中文](README_zh.md)


MOSS‑TTS Family is an open‑source **speech and sound generation model family** from [MOSI.AI](https://mosi.cn/#hero) and the [OpenMOSS team](https://www.open-moss.com/). It is designed for **high‑fidelity**, **high‑expressiveness**, and **complex real‑world scenarios**, covering stable long‑form speech, multi‑speaker dialogue, voice/character design, environmental sound effects, and real‑time streaming TTS.

## News
* 2026.3.4: 🎉 We add MOSS-TTS skills in [ClawHub](https://clawhub.ai) of 🦞 OpenClaw: [feishu-voice-tts](https://clawhub.ai/helloeveryworlds/feishu-voice-tts) and [moss-tts-voice](https://clawhub.ai/luogao2333/moss-tts-voice).
* 2026.2.10: 🎉🎉🎉 We have released [MOSS-TTS Family](https://huggingface.co/collections/OpenMOSS-Team/moss-tts). Check our [Blog](https://mosi.cn/#models) for more details! Our **Huggingface Space** is here: [MOSS-TTS](https://huggingface.co/spaces/OpenMOSS-Team/MOSS-TTS), [MOSS-TTSD-v1.0](https://huggingface.co/spaces/OpenMOSS-Team/MOSS-TTSD-v1.0), [MOSS-VoiceGenerator](https://huggingface.co/spaces/OpenMOSS-Team/MOSS-VoiceGenerator).


## Demo

<div align="center">
  <video src="https://gist.github.com/user-attachments/assets/fdce9f66-20ec-45e8-9615-89606ae2fbe8" width="70%" poster=""> </video>
</div>

## Contents

- [Introduction](#introduction)
- [Model Architecture](#model-architecture)
- [Released Models](#released-models)
- [Supported Languages](#supported-languages)
- [Quickstart](#quickstart)
  - [OpenClaw API Skills](#openclaw-api-skills)
  - [Environment Setup](#environment-setup)
  - [(Optional) Install FlashAttention 2](#optional-install-flashattention-2)
  - [MOSS-TTS Basic Usage](#moss-tts-basic-usage)
- [Evaluation](#evaluation)
  - [MOSS-TTS](#moss-tts-seed-tts-eval)
  - [MOSS-TTSD](#moss-ttsd-subjective--ttsd-eval)
  - [MOSS-VoiceGenerator](#moss-voicegenerator-subjective)
- [MOSS-Audio-Tokenizer](#moss-audio-tokenizer)
  - [Introduction](#mat-intro)
  - [Model Weights](#model-weights)
  - [Objective Reconstruction Evaluation](#objective-reconstruction-evaluation)


## Introduction

<p align="center">
  <img src="./assets/moss_tts_family.jpeg" width="85%" />
</p>

When a single piece of audio needs to **sound like a real person**, **pronounce every word accurately**, **switch speaking styles across content**, **remain stable over tens of minutes**, and **support dialogue, role‑play, and real‑time interaction**, a single TTS model is often not enough. The **MOSS‑TTS Family** breaks the workflow into five production‑ready models that can be used independently or composed into a complete pipeline.

- **MOSS‑TTS**: The flagship production model featuring high fidelity and optimal zero-shot voice cloning. It supports **long-speech generation**, **fine-grained control over Pinyin, phonemes, and duration**, as well as **multilingual/code-switched synthesis**.
- **MOSS‑TTSD**: A spoken dialogue generation model for expressive, multi-speaker, and ultra-long dialogues. The new **v1.0 version** achieves **industry-leading performance on objective metrics** and **outperformed top closed-source models like Doubao and Gemini 2.5-pro** in subjective evaluations. You can visit the [MOSS-TTSD repository](https://github.com/OpenMOSS/MOSS-TTSD) for details.
- **MOSS‑VoiceGenerator**: An open-source voice design model capable of generating diverse voices and styles directly from text prompts, **without any reference speech**. It unifies voice design, style control, and synthesis, functioning independently or as a design layer for downstream TTS. Its performance **surpasses other top-tier voice design models in arena ratings**.
- **MOSS‑TTS‑Realtime**: A multi-turn context-aware model for real-time voice agents. It uses incremental synthesis to ensure natural and coherent replies, making it **ideal for building low-latency voice agents when paired with text models**.
- **MOSS‑SoundEffect**: A content creation model specialized in **sound effect generation** with wide category coverage and controllable duration. It generates audio for natural environments, urban scenes, biological sounds, human actions, and musical fragments, suitable for film, games, and interactive experiences.


## Model Architecture

We train **MossTTSDelay** and **MossTTSLocal** as complementary baselines under one training/evaluation setup: **Delay** emphasizes long-context stability, inference speed, and production readiness, while **Local** emphasizes lightweight flexibility and strong objective performance for streaming-oriented systems. Together they provide reproducible references for deployment and research.

**MossTTSRealtime** is not a third comparison baseline; it is a capability-driven design for voice agents. By modeling multi-turn context from both prior text and user acoustics, it delivers low-latency streaming speech that stays coherent and voice-consistent across turns.


| Architecture  | Core Mechanism | Arch Details |
|---|---|---|
| `MossTTSDelay` |  Multi‑head parallel RVQ prediction with delay‑pattern scheduling | [![Arch Details](https://img.shields.io/badge/Model%20Card-View-blue?logo=markdown)](moss_tts_delay/README.md) |
| `MossTTSLocal` | Time‑synchronous RVQ blocks with a depth transformer | [![Arch Details](https://img.shields.io/badge/Model%20Card-View-blue?logo=markdown)](moss_tts_local/README.md) |
| `MossTTSRealtime` | Hierarchical text–audio inputs for realtime synthesis | [![Arch Details](https://img.shields.io/badge/Model%20Card-View-blue?logo=markdown)](moss_tts_realtime/README.md) |

## Released Models


| Model | Architecture | Size | Model Card | Hugging Face | ModelScope |
|---|---|---:|---|---|---|
| **MOSS-TTS** | `MossTTSDelay` | 8B | [![Model Card](https://img.shields.io/badge/Model%20Card-View-blue?logo=markdown)](docs/moss_tts_model_card.md) | [![Hugging Face](https://img.shields.io/badge/Huggingface-Model-orange?logo=huggingface)](https://huggingface.co/OpenMOSS-Team/MOSS-TTS) | [![ModelScope](https://img.shields.io/badge/ModelScope-Model-lightgrey?logo=modelscope)](https://modelscope.cn/models/openmoss/MOSS-TTS) |
|  | `MossTTSLocal` | 1.7B | [![Model Card](https://img.shields.io/badge/Model%20Card-View-blue?logo=markdown)](docs/moss_tts_model_card.md) | [![Hugging Face](https://img.shields.io/badge/Huggingface-Model-orange?logo=huggingface)](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer) | [![ModelScope](https://img.shields.io/badge/ModelScope-Model-lightgrey?logo=modelscope)](https://modelscope.cn/models/openmoss/MOSS-TTS-Local-Transformer) |
| **MOSS‑TTSD‑V1.0** | `MossTTSDelay` | 8B | [![Model Card](https://img.shields.io/badge/Model%20Card-View-blue?logo=markdown)](docs/moss_ttsd_model_card.md) | [![Hugging Face](https://img.shields.io/badge/Huggingface-Model-orange?logo=huggingface)](https://huggingface.co/OpenMOSS-Team/MOSS-TTSD-v1.0) | [![ModelScope](https://img.shields.io/badge/ModelScope-Model-lightgrey?logo=modelscope)](https://modelscope.cn/models/openmoss/MOSS-TTSD-v1.0) |
| **MOSS‑VoiceGenerator** | `MossTTSDelay` | 1.7B | [![Model Card](https://img.shields.io/badge/Model%20Card-View-blue?logo=markdown)](docs/moss_voice_generator_model_card.md) | [![Hugging Face](https://img.shields.io/badge/Huggingface-Model-orange?logo=huggingface)](https://huggingface.co/OpenMOSS-Team/MOSS-VoiceGenerator) | [![ModelScope](https://img.shields.io/badge/ModelScope-Model-lightgrey?logo=modelscope)](https://modelscope.cn/models/openmoss/MOSS-VoiceGenerator) |
| **MOSS‑SoundEffect** | `MossTTSDelay` | 8B | [![Model Card](https://img.shields.io/badge/Model%20Card-View-blue?logo=markdown)](docs/moss_sound_effect_model_card.md) | [![Hugging Face](https://img.shields.io/badge/Huggingface-Model-orange?logo=huggingface)](https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect) | [![ModelScope](https://img.shields.io/badge/ModelScope-Model-lightgrey?logo=modelscope)](https://modelscope.cn/models/openmoss/MOSS-SoundEffect) |
| **MOSS‑TTS‑Realtime** | `MossTTSRealtime` | 1.7B | [![Model Card](https://img.shields.io/badge/Model%20Card-View-blue?logo=markdown)](docs/moss_tts_realtime_model_card.md) | [![Hugging Face](https://img.shields.io/badge/Huggingface-Model-orange?logo=huggingface)](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Realtime) | [![ModelScope](https://img.shields.io/badge/ModelScope-Model-lightgrey?logo=modelscope)](https://modelscope.cn/models/openmoss/MOSS-TTS-Realtime) |

## Supported Languages

MOSS-TTS, MOSS-TTSD and MOSS-TTS-Realtime currently supports **20 languages**:

| Language | Code | Flag | Language | Code | Flag | Language | Code | Flag |
|---|---|---|---|---|---|---|---|---|
| Chinese | zh | 🇨🇳 | English | en | 🇺🇸 | German | de | 🇩🇪 |
| Spanish | es | 🇪🇸 | French | fr | 🇫🇷 | Japanese | ja | 🇯🇵 |
| Italian | it | 🇮🇹 | Hungarian | hu | 🇭🇺 | Korean | ko | 🇰🇷 |
| Russian | ru | 🇷🇺 | Persian (Farsi) | fa | 🇮🇷 | Arabic | ar | 🇸🇦 |
| Polish | pl | 🇵🇱 | Portuguese | pt | 🇵🇹 | Czech | cs | 🇨🇿 |
| Danish | da | 🇩🇰 | Swedish | sv | 🇸🇪 | | | |
| Greek | el | 🇬🇷 | Turkish | tr | 🇹🇷 |  |  |  |


## Quickstart

### OpenClaw API Skills

We add MOSS-TTS skills in [ClawHub](https://clawhub.ai) of 🦞 OpenClaw. You can get your API key from [MOSI AI Studio](https://studio.mosi.cn).

| Skill | Description | Install |
|---|---|---|
| [`feishu-voice-tts`](https://clawhub.ai/helloeveryworlds/feishu-voice-tts) | Send voice messages in Feishu | `clawhub install feishu-voice-tts` |
| [`moss-tts-voice`](https://clawhub.ai/luogao2333/moss-tts-voice) | Call MOSS-TTS API to generate speech | `clawhub install moss-tts-voice` |

### Environment Setup

We recommend a clean, isolated Python environment with **Transformers 5.0.0** to avoid dependency conflicts.

#### Using Conda

```bash
conda create -n moss-tts python=3.12 -y
conda activate moss-tts
```

Install all required dependencies:

```bash
git clone https://github.com/OpenMOSS/MOSS-TTS.git
cd MOSS-TTS
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e .
```

#### Using `uv`

```bash
# Install uv first: https://docs.astral.sh/uv/getting-started/installation/
git clone https://github.com/OpenMOSS/MOSS-TTS.git
cd MOSS-TTS
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install --torch-backend cu128 -e .
```

#### (Optional) Install FlashAttention 2

For better speed and lower GPU memory usage, you can install FlashAttention 2 if your hardware supports it.

If you use Conda/pip:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e ".[flash-attn]"
```

If your machine has limited RAM and many CPU cores, you can cap build parallelism:

```bash
MAX_JOBS=4 pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e ".[flash-attn]"
```

If you use `uv`:

```bash
uv pip install --torch-backend cu128 -e ".[flash-attn]"
```

If your machine has limited RAM and many CPU cores, you can cap build parallelism:

```bash
MAX_JOBS=4 uv pip install --torch-backend cu128 -e ".[flash-attn]"
```

Notes:
- Dependencies are managed in `pyproject.toml`, which currently pins `torch==2.9.1+cu128` and `torchaudio==2.9.1+cu128`.
- In `uv`, `--torch-backend cu128` lets uv fetch compatible PyTorch CUDA wheels and resolve the rest from PyPI with the default safe index strategy.
- If you need another backend, replace `cu128` with your target (for example, `cpu`, `cu126`).
- If FlashAttention 2 fails to build on your machine, you can skip it and use the default attention backend.
- FlashAttention 2 is only available on supported GPUs and is typically used with `torch.float16` or `torch.bfloat16`.


<a id="moss-tts-basic-usage"></a>
### MOSS‑TTS Basic Usage

If you prefer Gradio demos, we provide 4 scripts for the main models:

| Model | Script | Run |
|---|---|---|
| MOSS-TTS | [clis/moss_tts_app.py](clis/moss_tts_app.py) |
| MOSS-TTSD | [clis/moss_ttsd_app.py](clis/moss_ttsd_app.py) | 
| MOSS-VoiceGenerator | [clis/moss_voice_generator_app.py](clis/moss_voice_generator_app.py) |
| MOSS-SoundEffect | [clis/moss_sound_effect_app.py](clis/moss_sound_effect_app.py) | 

For the MOSS-TTS-Realtime Gradio demo, please refer to [MOSS-TTS-Realtime Model Card](docs/moss_tts_realtime_model_card.md)

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

For each model’s full usage, please refer to its corresponding model card.


## Evaluation

This section summarizes the **family‑level evaluation highlights** for MOSS‑TTS and MOSS‑VoiceGenerator. For full details, see each model’s model card.

### MOSS‑TTS
MOSS‑TTS achieved state‑of‑the‑art results on the open‑source zero‑shot TTS benchmark `Seed‑TTS‑eval`, surpassing all open‑source models and rivaling leading closed‑source systems.

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
| GLM-TTS | 1.5B | ✅ | 2.23 | 67.2 | 1.03 | 76.1 |
| GLM-TTS-RL | 1.5B | ✅ | 1.91 | 68.1 | 0.89 | 76.4 |
|  |  |  |  |  |  |  |
| **MossTTSDelay** | **8B** | ✅ | 1.79 | 71.46 | 1.32 | 77.05 |
| **MossTTSLocal** | **1.7B** | ✅ | 1.85 | **73.42** | 1.2 | **78.82** |

### MOSS‑TTSD

#### Objective Evaluation
We evaluate MOSS‑TTSD-v1.0 using three objective metrics: Speaker Attribution Accuracy (ACC), Speaker Similarity (SIM), and Word Error Rate (WER). Benchmarked against multiple open-source and closed-source models, the results show that MOSS‑TTSD-v1.0 consistently achieves either the best or second-best performance.

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

#### Subjective Evaluation
For open-source models, annotators are asked to score each sample pair in terms of speaker attribution accuracy, voice similarity, prosody, and overall quality. Following the methodology of the LMSYS Chatbot Arena, we compute Elo ratings and confidence intervals for each dimension.
![alt text](assets/VS_Open-Source_Models.jpg)

For closed-source models, annotators are only asked to choose the overall preferred one in each pair, and we compute the win rate accordingly.
![alt text](assets/VS_Proprietary_Models.png)


### MOSS‑VoiceGenerator
MOSS‑VoiceGenerator demonstrates strong subjective preference across **overall preference**, **instruction following**, and **naturalness**.

<p align="center">
  <img src="./assets/moss_voice_generator_winrate.png" width="70%" />
</p>

## MOSS-Audio-Tokenizer

<a id="mat-intro"></a>
### Introduction
**MOSS-Audio-Tokenizer** serves as the unified discrete audio interface for the entire MOSS-TTS Family. It is based on the **Cat** (**C**ausal **A**udio **T**okenizer with **T**ransformer) architecture—a 1.6-billion-parameter, "CNN-free" homogeneous audio tokenizer built entirely from Causal Transformer blocks.

- **Unified Discrete Bridge**: It acts as the shared backbone for MOSS-TTS, MOSS-TTSD, MOSS-VoiceGenerator, MOSS-SoundEffect, and MOSS-TTS-Realtime, providing a consistent audio representation across the family.
- **Extreme Compression & High Fidelity**: It compresses 24kHz raw audio into a remarkably low frame rate of 12.5Hz. Utilizing a 32-layer Residual Vector Quantizer (RVQ), it supports high-fidelity reconstruction across variable bitrates from 0.125kbps to 4kbps.
- **Massive-Scale General Audio Training**: Trained from scratch on 3 million hours of diverse data (speech, sound effects, and music), the model achieves state-of-the-art reconstruction among open source audio tokenizers.
- **Native Streaming Design**: The pure Causal Transformer architecture is specifically designed for scalability and low-latency streaming inference, enabling real-time production workflows.

To learn more about setup, advanced usage, and evaluation metrics, please visit the [MOSS-Audio-Tokenizer Repository](https://github.com/OpenMOSS/MOSS-Audio-Tokenizer)

<p align="center">
  <img src="./assets/arch_moss_audio_tokenizer.png" alt="MOSS Audio Tokenizer architecture" width="100%" />
  Architecture of MOSS Audio Tokenizer
</p>

### Model Weights

| Model | Hugging Face | ModelScope |
|:-----:|:------------:|:----------:|
| **MOSS-Audio-Tokenizer** | [![Hugging Face](https://img.shields.io/badge/Huggingface-Model-orange?logo=huggingface)](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer) | [![ModelScope](https://img.shields.io/badge/ModelScope-Model-lightgrey?logo=modelscope)](https://modelscope.cn/models/openmoss/MOSS-Audio-Tokenizer) |

### Objective Reconstruction Evaluation

We compare **MOSS Audio Tokenizer** with open-source audio tokenizers on the LibriSpeech test-clean subset using SIM, STOI, PESQ-NB, and PESQ-WB. Bitrate is controlled by varying the number of RVQ codebooks during decoding, and MOSS Audio Tokenizer leads reconstruction quality among open-source audio tokenizers at comparable 0–4 kbps bitrates.

<p align="center">
  <img src="./assets/evaluation_moss_audio_tokenizer.png" alt="LibriSpeech objective metrics for audio tokenizers" width="90%" />
</p>

## LICENSE

Models in MOSS-TTS Family are licensed under the Apache License 2.0.

## Citation

```bibtex
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=OpenMOSS/MOSS-TTS&type=date&legend=top-left)](https://www.star-history.com/#OpenMOSS/MOSS-TTS&type=date&legend=top-left)
