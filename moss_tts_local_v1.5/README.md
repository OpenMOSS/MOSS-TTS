# MOSS-TTS-Local-Transformer-v1.5

This folder contains the public remote-code implementation and realtime
streaming examples for `OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5`.

MOSS-TTS-Local-Transformer-v1.5 uses the `MossTTSLocal` architecture with
time-synchronous RVQ frame generation and `OpenMOSS-Team/MOSS-Audio-Tokenizer-v2` as the audio tokenizer
for 48 kHz stereo audio encoding and decoding.

<p align="center">
  <img src="../assets/archi_local.png" width="60%" />
</p>

---

## Model

| Item | Value |
|---|---|
| **Checkpoint** | `OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5` |
| **Architecture** | `MossTTSLocal` |
| **Audio tokenizer** | `OpenMOSS-Team/MOSS-Audio-Tokenizer-v2` |
| **Audio format** | 48 kHz stereo |
| **Backbone Model** | Initialized from **Qwen3-4B** |
| **Depth Transformer** | 1 Transformer blocks (Hidden: 2560, FFN: 9728) |
| **Frame Rate** | 12.5 Hz (1s ≈ 12.5 tokens/blocks) |
| **Codebooks** | 12 RVQ layers (10-bit each) |
| **Generation Mode** | Purely Autoregressive (AR) |

> **VRAM note (measured on an NVIDIA T4, 16 GB):** this checkpoint fits comfortably on a 16 GB GPU — measured peak VRAM ≈13.4 GB (≈16.5% headroom), vs. the flagship 8B `MOSS-TTS-v1.5`/`MOSS-TTS 1.0` checkpoint, which OOMs on 16 GB GPUs just loading the bf16 weights (≈15.5 GB). This makes `MOSS-TTS-Local-Transformer-v1.5` the recommended checkpoint for 16 GB-class GPUs such as the T4.

## Batch Inference

```python
from pathlib import Path
from tqdm import tqdm
import importlib.util

import torch
import torchaudio
from transformers import AutoModel, AutoProcessor

# Disable the broken cuDNN SDPA backend on some CUDA/PyTorch combinations.
torch.backends.cuda.enable_cudnn_sdp(False)
# Keep these enabled as fallbacks.
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

pretrained_model_name_or_path = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32
# Hardware note: this default is fine on Ampere+ GPUs (A100/H100, compute capability >= 8).
# On Turing GPUs (e.g. NVIDIA T4, compute capability 7.5), prefer torch.float16 instead --
# bf16 inputs are rejected by PyTorch's SDPA EFFICIENT_ATTENTION kernel on that architecture
# ("RuntimeError: No available kernel"), silently forcing a fallback to the slower MATH backend.


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

text_zh = "亲爱的你，愿你的每一天都值得被记住，也值得被珍惜。"
text_en = "We stand on the threshold of the AI era, where intelligence becomes an extension of human creativity."
text_fr = "Bonjour, je voudrais essayer une voix francaise naturelle et stable."
text_pause = "我今天学习了一首中国的古诗，它的名字是[pause 3.2s]静夜思！"

# Use remote demo audio to avoid requiring local assets.
ref_audio_zh = "https://speech-demo.oss-cn-shanghai.aliyuncs.com/moss_tts_demo/tts_readme_demo/reference_zh.wav"
ref_audio_en = "https://speech-demo.oss-cn-shanghai.aliyuncs.com/moss_tts_demo/tts_readme_demo/reference_en.m4a"

conversations = [
    # Direct TTS. Language tags are recommended in v1.5 when the language is known.
    [processor.build_user_message(text=text_zh, language="Chinese")],
    [processor.build_user_message(text=text_en, language="English")],
    [processor.build_user_message(text=text_fr, language="French")],
    # Explicit pause control. Use [pause X.Ys], such as [pause 3.2s].
    [processor.build_user_message(text=text_pause, language="Chinese")],
    # Voice cloning with a reference audio.
    [processor.build_user_message(text=text_zh, reference=[ref_audio_zh], language="Chinese")],
    [processor.build_user_message(text=text_en, reference=[ref_audio_en], language="English")],
    # Duration control. At 12.5 frames per second, 125 frames is about 10 seconds.
    [processor.build_user_message(text=text_en, tokens=125, language="English")],
]

model = AutoModel.from_pretrained(
    pretrained_model_name_or_path,
    trust_remote_code=True,
    attn_implementation=attn_implementation,
    torch_dtype=dtype,
).to(device)
model.eval()

batch_size = 1
save_dir = Path("inference_root_moss_tts_local_v1_5")
save_dir.mkdir(exist_ok=True, parents=True)
sample_idx = 0

with torch.no_grad():
    for start in tqdm(range(0, len(conversations), batch_size)):
        batch_conversations = conversations[start : start + batch_size]
        batch = processor(batch_conversations, mode="generation")
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=4096,
            do_sample=True,
            audio_temperature=1.7,
            audio_top_p=0.8,
            audio_top_k=25,
            audio_repetition_penalty=1.0,
        )

        for message in processor.decode(outputs):
            if message is None:
                continue
            audio = message.audio_codes_list[0]
            out_path = save_dir / f"sample{sample_idx}.wav"
            sample_idx += 1
            # MOSS-TTS Local v1.5 codec returns stereo audio as [channels, samples].
            # Save the two-channel tensor directly.
            torchaudio.save(str(out_path), audio, processor.model_config.sampling_rate)
```

## Realtime Streaming Decode

Launch the realtime browser app. It uses FastAPI plus Web Audio playback so
decoded PCM chunks can be played while generation is still running:

```bash
MODEL_DIR=OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
CODEC_DIR=OpenMOSS-Team/MOSS-Audio-Tokenizer-v2 \
TTS_DEVICE=cuda:0 \
CODEC_DEVICE=cuda:1 \
CODEC_WEIGHT_DTYPE=fp32 \
bash moss_tts_local_v1.5/run_streaming_app.sh
```

The app defaults to `flash_attention_2`. If FlashAttention 2 is not available
in the current environment, runtime loading falls back to `sdpa` on CUDA.
The codec encoder/decoder weights default to `fp32`; pass
`--codec-weight-dtype bf16` or set `CODEC_WEIGHT_DTYPE=bf16` to reduce memory.
For Continuation and Continuation + Clone modes, provide the reference audio
transcript in the separate Reference Audio Transcript field.

The streaming demo defaults are:

| Parameter | Default |
|---|---:|
| Audio Temperature | 1.7 |
| Audio Top P | 0.8 |
| Audio Top K | 25 |
| Audio Repetition Penalty | 1.0 |
| Codec Chunk Frames | 0 (auto, UI range 0-32) |
| Initial Playback Delay | 0.08 s |

Generated audio, token dumps, and metadata are written under
`outputs/moss_tts_local_v1_5_streaming/` by default.
