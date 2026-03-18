# Fine-Tuning MossTTSLocal

This directory provides a complete finetuning workflow built on the `MossTTSLocal` architecture:

- `prepare_data.py`: pre-extract target audio `audio_codes`, with rank-sharded output support
- `dataset.py`: pack `text / instruction / ambient_sound / reference` and related fields into teacher-forcing samples
- `sft.py`: supports single-GPU, data parallel training, and optional FSDP / DeepSpeed ZeRO-3 sharded training
- `convert_seed_tts_eval_to_jsonl.py`: convert `seed-tts-eval` folders into training JSONL
- `run_train.sh`: one-click launcher

## 1. Install

Install training dependencies first:

```bash
git clone https://github.com/OpenMOSS/MOSS-TTS.git
cd MOSS-TTS
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e ".[torch-runtime,finetune]"
```

If your environment supports FlashAttention 2, you can also follow the installation notes in the root README.

If you plan to use **DeepSpeed ZeRO-3**, install the extra dependency group as well:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e ".[torch-runtime,finetune-deepspeed]"
```

## 2. Input JSONL format

All tasks share the same basic idea:

- `audio`: target training audio path; `prepare_data.py` will encode it into `audio_codes`
- all other fields are mapped directly into `processor.build_user_message(...)`

### 2.1 MOSS-TTS

#### Plain `text, speech` pairs

This format does not require reference audio and is supported directly:

```jsonl
{"audio":"./data/utt0001.wav","text":"Actually, I noticed that I am very sensitive to other people's emotions.","language":"en"}
{"audio":"./data/utt0002.wav","text":"She said she would be here by noon.","language":"en"}
```

#### Voice cloning / reference-conditioned training

```jsonl
{"audio":"./data/utt0001.wav","text":"Actually, I noticed that I am very sensitive to other people's emotions.","ref_audio":"./data/ref.wav","language":"en"}
{"audio":"./data/utt0002.wav","text":"She said she would be here by noon.","ref_audio":"./data/ref.wav","language":"en"}
```

### 2.2 MOSS-TTSD

MOSS-TTSD shares the same `prepare_data.py / sft.py` pipeline as MOSS-TTS, and the format can stay the same.  
The only difference is that `reference` may be a multi-speaker list, and list elements may be `null`, meaning that speaker has no cloning reference:

```jsonl
{
  "audio":"./data/dialog_target.wav",
  "text":"[S1] This is the prefix from speaker one. [S2] This is the prefix from speaker two. [S1] Now continue the next turn.",
  "reference":["./data/s1_ref.wav", null],
  "language":"en"
}
```

Notes:

- `prepare_data.py` always encodes `audio`
- by default it also encodes any reference audio found in `reference` / `ref_audio` / `reference_audio`
- `null` entries inside `reference` are preserved as `None` during training and will not be encoded incorrectly
- no extra `prompt_audio` field is required; autoregressive continuation is already learned through standard teacher-forcing training

### 2.3 MOSS-SoundEffect

MOSS-SoundEffect uses the same pipeline, with `ambient_sound` as the user-side field:

```jsonl
{"audio":"./data/rain.wav","ambient_sound":"Rolling thunder with steady rainfall."}
{"audio":"./data/footsteps.wav","ambient_sound":"Clear footsteps echoing on concrete at a steady rhythm.","tokens":160}
```

### 2.4 MOSS-VoiceGenerator

MOSS-VoiceGenerator also shares the same training flow, using `text + instruction`:

```jsonl
{"audio":"./data/old_man.wav","text":"My old back is really giving me trouble these days.","instruction":"A tired, hoarse elderly voice complaining slowly with a faint groan."}
{"audio":"./data/tavern.wav","text":"Hey there, stranger!","instruction":"Hearty, jovial tavern owner's voice, loud and welcoming with a slightly gruff tone."}
```

## 3. Prepare data

### 3.1 Single process

```bash
python moss_tts_local/finetuning/prepare_data.py \
    --model-path OpenMOSS-Team/MOSS-TTS-Local-Transformer \
    --codec-path OpenMOSS-Team/MOSS-Audio-Tokenizer \
    --device auto \
    --input-jsonl train_raw.jsonl \
    --output-jsonl train_with_codes.jsonl
```

By default, `prepare_data.py` pre-encodes reference audio as well. If you only want target audio codes, disable it explicitly:

```bash
python moss_tts_local/finetuning/prepare_data.py \
    --model-path OpenMOSS-Team/MOSS-TTS-Local-Transformer \
    --codec-path OpenMOSS-Team/MOSS-Audio-Tokenizer \
    --device auto \
    --input-jsonl train_raw.jsonl \
    --output-jsonl train_with_codes.jsonl \
    --skip-reference-audio-codes
```

### 3.2 Multi-node / multi-GPU parallel preprocessing

`prepare_data.py` now follows the `accelerate launch` multi-process model directly.  
For example, with 2 nodes and 16 GPUs in total, the dataset is split into 16 shards and each rank writes one shard:

```bash
accelerate launch --num_processes 16 moss_tts_local/finetuning/prepare_data.py \
    --model-path OpenMOSS-Team/MOSS-TTS-Local-Transformer \
    --codec-path OpenMOSS-Team/MOSS-Audio-Tokenizer \
    --device auto \
    --input-jsonl train_raw.jsonl \
    --output-jsonl prepared/train_with_codes.jsonl
```

The output will look like:

- `prepared/train_with_codes.rank00000-of-00016.jsonl`
- `prepared/train_with_codes.rank00001-of-00016.jsonl`
- ...
- `prepared/train_with_codes.rank00015-of-00016.jsonl`

During training, `sft.py` can read:

- a single JSONL
- a directory
- a glob such as `prepared/train_with_codes.rank*.jsonl`
- or a comma-separated list of files

If your platform already injects distributed communication environment variables, `accelerate launch` will reuse them directly, so you usually do not need to write `torchrun`-style communication arguments yourself.

## 4. Train

### 4.1 Single-GPU baseline

```bash
accelerate launch moss_tts_local/finetuning/sft.py \
    --model-path OpenMOSS-Team/MOSS-TTS-Local-Transformer \
    --train-jsonl train_with_codes.jsonl \
    --output-dir output/moss_tts_local_sft \
    --per-device-batch-size 1 \
    --gradient-accumulation-steps 8 \
    --learning-rate 1e-5 \
    --warmup-ratio 0.03 \
    --num-epochs 3 \
    --mixed-precision bf16 \
    --channelwise-loss-weight 1,32 \
    --gradient-checkpointing
```

### 4.2 Data parallel

For single-node 8-GPU data parallel training, you can use:

```bash
accelerate launch \
    --config_file moss_tts_local/finetuning/configs/accelerate_ddp_8gpu.yaml \
    moss_tts_local/finetuning/sft.py \
    --model-path OpenMOSS-Team/MOSS-TTS-Local-Transformer \
    --train-jsonl 'prepared/train_with_codes.rank*.jsonl' \
    --output-dir output/moss_tts_local_sft_ddp \
    --per-device-batch-size 1 \
    --gradient-accumulation-steps 4 \
    --mixed-precision bf16 \
    --channelwise-loss-weight 1,32 \
    --gradient-checkpointing
```

### 4.3 Optional parameter-sharded training

For the 1.7B `MOSS-TTS-Local-Transformer` model, single-node DDP is usually enough. If you still want parameter sharding, the following approaches are supported:

- **FSDP**: shard parameters, gradients, and optimizer states across ranks
- **DeepSpeed ZeRO-3**: fully shard parameters, gradients, and optimizer states; better suited for larger models and multi-node setups

#### FSDP

```bash
accelerate launch \
    --config_file moss_tts_local/finetuning/configs/accelerate_fsdp_1.7b.yaml \
    moss_tts_local/finetuning/sft.py \
    --model-path OpenMOSS-Team/MOSS-TTS-Local-Transformer \
    --train-jsonl 'prepared/train_with_codes.rank*.jsonl' \
    --output-dir output/moss_tts_local_sft_fsdp \
    --per-device-batch-size 1 \
    --gradient-accumulation-steps 4 \
    --mixed-precision bf16 \
    --channelwise-loss-weight 1,32 \
    --gradient-checkpointing
```

#### DeepSpeed ZeRO-3

```bash
accelerate launch \
    --config_file moss_tts_local/finetuning/configs/accelerate_zero3_1.7b.yaml \
    moss_tts_local/finetuning/sft.py \
    --model-path OpenMOSS-Team/MOSS-TTS-Local-Transformer \
    --train-jsonl 'prepared/train_with_codes.rank*.jsonl' \
    --output-dir output/moss_tts_local_sft_zero3 \
    --per-device-batch-size 1 \
    --gradient-accumulation-steps 4 \
    --mixed-precision bf16 \
    --channelwise-loss-weight 1,32 \
    --gradient-checkpointing
```

ZeRO-3 requires the `deepspeed` package. If you only use DDP or FSDP, you do not need it.

### 4.4 Common tunable hyperparameters

`sft.py` now exposes the common training hyperparameters directly:

- optimizer: `--learning-rate`, `--weight-decay`, `--adam-beta1`, `--adam-beta2`, `--adam-eps`
- LR schedule: `--lr-scheduler-type`, `--warmup-steps`, `--warmup-ratio`
- stability: `--max-grad-norm`, `--gradient-checkpointing`, `--mixed-precision`
- RVQ multi-head loss weighting: `--channelwise-loss-weight`

`--channelwise-loss-weight` supports two forms:

- `n_vq + 1` values: `text_head,vq0,...,vqN`
- two values: `text_weight,total_audio_weight`

The default is `1,32`, which means the text head and each individual audio head have equal weight.

Training logs now print:

- timestamped log prefixes
- `global_batch_size` and its formula
- `step_time`
- `steps_per_sec`
- `samples_per_sec`
- `eta`

### 4.5 Multi-node training

Update the following fields in the config file for your cluster:

- `num_machines`
- `num_processes`
- `machine_rank`
- `main_process_ip`
- `main_process_port`

For example, for 2 nodes and 16 GPUs:

- node 0: `machine_rank: 0`
- node 1: `machine_rank: 1`
- `num_machines: 2`
- `num_processes: 16`

The training command itself can stay unchanged.

## 5. Quick inference test

Each checkpoint saved by `sft.py` now contains model config, runtime Python files, tokenizer files, and processor metadata, so you can call `from_pretrained` directly on that checkpoint directory:

```python
from pathlib import Path
import importlib.util
import torch
import torchaudio
from transformers import AutoProcessor

from moss_tts_local.modeling_moss_tts import MossTTSDelayModel

torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)


def resolve_attn_implementation(device: str, dtype: torch.dtype) -> str:
    if (
        device == "cuda"
        and importlib.util.find_spec("flash_attn") is not None
        and dtype in {torch.float16, torch.bfloat16}
    ):
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return "flash_attention_2"
    if device == "cuda":
        return "sdpa"
    return "eager"


model_path = "output/moss_tts_local_sft/checkpoint-epoch-2"
reference_audio = "./assets/audio/reference_en_0.mp3"
text = "This is a quick finetuning smoke test for MOSS-TTS Local."

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32
attn_implementation = resolve_attn_implementation(device, dtype)

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
processor.audio_tokenizer = processor.audio_tokenizer.to(device)

model = MossTTSDelayModel.from_pretrained(
    model_path,
    torch_dtype=dtype,
    attn_implementation=attn_implementation,
).to(device)
model.eval()

conversation = [[
    processor.build_user_message(
        text=text,
        reference=[reference_audio],
    )
]]

batch = processor(conversation, mode="generation")
outputs = model.generate(
    input_ids=batch["input_ids"].to(device),
    attention_mask=batch["attention_mask"].to(device),
    max_new_tokens=4096,
)

message = processor.decode(outputs)[0]
audio = message.audio_codes_list[0]
Path("demo_outputs").mkdir(parents=True, exist_ok=True)
torchaudio.save("demo_outputs/finetuned_sample.wav", audio.unsqueeze(0), processor.model_config.sampling_rate)
```

## 6. One-click launcher

Run directly:

```bash
bash moss_tts_local/finetuning/run_train.sh
```

Common environment variables:

- `RAW_JSONL`: raw training JSONL
- `PREPARED_JSONL`: output file from `prepare_data.py`
- `TRAIN_JSONL`: optional; training input, which can be a single file, directory, or glob. If unset, it is inferred automatically from `PREPARED_JSONL`
- `OUTPUT_DIR`: training output directory
- `ACCELERATE_CONFIG_FILE`: optional; DDP / FSDP / ZeRO-3 config file
- `SKIP_PREPARE`: set to `1` to skip preprocessing and train directly from existing `TRAIN_JSONL` / `PREPARED_JSONL`
- `PREP_EXTRA_ARGS_STR`: extra arguments passed to `prepare_data.py`
- `PREP_ACCELERATE_ARGS_STR`: if you want preprocessing to also launch through `accelerate`, set this, for example `--num_processes 16` or `--config_file moss_tts_local/finetuning/configs/accelerate_ddp_8gpu.yaml`
- `TRAIN_EXTRA_ARGS_STR`: extra arguments passed to `sft.py`

For example, to launch with ZeRO-3:

```bash
RAW_JSONL=train_raw.jsonl \
PREPARED_JSONL=prepared/train_with_codes.jsonl \
OUTPUT_DIR=output/moss_tts_local_sft_zero3 \
ACCELERATE_CONFIG_FILE=moss_tts_local/finetuning/configs/accelerate_zero3_1.7b.yaml \
PREP_ACCELERATE_ARGS_STR='--config_file moss_tts_local/finetuning/configs/accelerate_ddp_8gpu.yaml' \
PREP_EXTRA_ARGS_STR='' \
TRAIN_EXTRA_ARGS_STR='--per-device-batch-size 1 --gradient-accumulation-steps 4 --num-epochs 3 --warmup-ratio 0.03 --mixed-precision bf16 --channelwise-loss-weight 1,32 --gradient-checkpointing' \
bash moss_tts_local/finetuning/run_train.sh
```

## 7. Additional task format notes

The remaining tasks do not require a separate trainer. You only need to switch the JSONL fields:

- **MOSS-TTS**: use `text`, optionally `ref_audio`
- **MOSS-TTSD**: use `text + reference`, where `reference` supports `null`
- **MOSS-SoundEffect**: use `ambient_sound`
- **MOSS-VoiceGenerator**: use `text + instruction`

Shared fields:

- `audio`: required target audio
- `language`, `tokens`, `quality`, `sound_event`, `ambient_sound`, `instruction`: fill them as needed by the task

Shared scripts:

- use `prepare_data.py` for data preparation
- use `sft.py` for training
- `train-jsonl` supports a single file, directory, glob, or multi-file list
