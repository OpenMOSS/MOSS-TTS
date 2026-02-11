# MOSS-SoundEffect Model Card

**MOSS-SoundEffect** is the **environment sound & sound effect generation model** in the **MOSS‑TTS Family**. It generates ambient soundscapes and concrete sound effects directly from text descriptions, and is designed to complement speech content with immersive context in production workflows.


## 1. Overview

### 1.1 TTS Family Positioning

MOSS-SoundEffect is designed as an audio generation backbone for creating high-fidelity environmental and action sounds from text, serving both scalable content pipelines and a strong research baseline for controllable audio generation.

**Design goals**
* **Coverage & richness**: broad sound taxonomy with layered ambience and realistic texture
* **Composability**: easy integration into creative pipelines (games/film/tools) and synthetic data generation setups


### 1.2 Key Capabilities
MOSS‑SoundEffect focuses on **contextual audio completion** beyond speech, enabling creators and systems to enrich scenes with believable acoustic environments and action‑level cues.

**What it can generate**
- **Natural environments**: e.g., “fresh snow crunching under footsteps.”
- **Urban environments**: e.g., “a sports car roaring past on the highway.”
- **Animals & creatures**: e.g., “early morning park with birds chirping in a quiet atmosphere.”
- **Human actions**: e.g., “clear footsteps echoing on concrete at a steady rhythm.”

**Why it matters**
- Completes **scene immersion** for narrative content, film/TV, documentaries, games, and podcasts.
- Supports **voice agents** and interactive systems that need ambient context, not just speech.
- Acts as the **sound‑design layer** of the MOSS‑TTS Family’s end‑to‑end workflow.



### 1.3 Model Architecture
**MOSS-SoundEffect** employs the **MossTTSDelay** architecture (see [moss_tts_delay/README.md](../moss_tts_delay/README.md)), reusing the same discrete token generation backbone for audio synthesis. A text prompt (optionally with simple control tags such as **duration**) is tokenized and fed into the Delay-pattern autoregressive model to predict **RVQ audio tokens** over time. The generated tokens are then decoded by the audio tokenizer/vocoder to produce high-fidelity sound effects, enabling consistent quality and controllable length across diverse SFX categories.



### 1.4 Released Models
**Recommended decoding hyperparameters**
| Model | audio_temperature | audio_top_p | audio_top_k | audio_repetition_penalty |
|---|---:|---:|---:|---:|
| **MOSS-SoundEffect** | 1.5 | 0.6 | 50 | 1.2 |


## 2. Quick Start



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


pretrained_model_name_or_path = "OpenMOSS-Team/MOSS-SoundEffect"
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

text_1 = "雷声隆隆，雨声淅沥。"
text_2 = "清晰脚步声在水泥地面回响，节奏稳定。"

conversations = [
    [processor.build_user_message(ambient_sound=text_1)],
    [processor.build_user_message(ambient_sound=text_2)]
]

model = AutoModel.from_pretrained(
    pretrained_model_name_or_path,
    trust_remote_code=True,
    # If FlashAttention 2 is installed, you can set attn_implementation="flash_attention_2"
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

### Input Types

**UserMessage**
| Field | Type | Required | Description |
|---|---|---:|---|
| `ambient_sound` | `str` | Yes | Description of environment sound & sound effect |
| `tokens` | `int` | No | Expected number of audio tokens. **1s ≈ 12.5 tokens**. |
