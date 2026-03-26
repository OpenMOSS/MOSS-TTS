# Fine-Tuning MOSS-TTS-Realtime

This directory provides a complete finetuning workflow built on the `MOSS-TTS-Realtime` architecture:

- `prepare_data.py`: pre-extract target audio `audio_codes`, with rank-sharded output support
- `dataset.py`: assemble preprocessed data into a trainable format
- `sft.py`: supports single-GPU, data parallel training, and optional FSDP / DeepSpeed ZeRO-3 sharded training
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

All data uses a unified `conversations` multi-turn dialogue format. Each record contains a `conversations` list, where each element represents one dialogue turn with `role` (`user` or `assistant`), `text` (text content), and `wav` (audio file path). An optional `ref_wav` field specifies the reference audio for voice cloning.

### 2.1 Single-turn data

Single-turn data contains only one assistant turn, functioning the same as standard TTS and voice cloning. If no reference audio is available, the `ref_wav` field can be omitted.

```jsonl
{"id": "000001", "ref_wav": "./data/ref0.wav", "conversations": [{"role": "assistant", "text": "Actually, I noticed that I am very sensitive to other people's emotions.", "wav": "./data/utt0001.wav"}]}
{"id": "000002", "ref_wav": "./data/ref1.wav", "conversations": [{"role": "assistant", "text": "She said she would be here by noon.", "wav": "./data/utt0002.wav"}]}
```

### 2.2 Multi-turn data

In multi-turn data, `user` turns represent the user's voice interaction with the VoiceAgent, and `assistant` turns represent the speech synthesized by MOSS-TTS-Realtime, which should share the same speaker as `ref_wav`. All assistant turns must be from the same speaker, while user turns can be from different speakers.
Single-turn and multi-turn data can be mixed together for training to maintain both single-turn and multi-turn capabilities.

```jsonl
{"id": "000003", "ref_wav": "./data/ref0.wav", "conversations": [{"role": "user", "text": "Hey, I just landed in Paris. I have about six hours before my next flight. Any ideas?", "wav": "./data/user_utt0001.wav"}, {"role": "assistant", "text": "Nice, welcome to Paris! Six hours is actually perfect for a short city walk. Are you traveling light, or do you have luggage with you?", "wav": "./data/assistant_utt0001.wav"}, {"role": "user", "text": "Just a backpack. I don't want anything too rushed.", "wav": "./data/user_utt0002.wav"}, {"role": "assistant", "text": "Got it. In that case, I'd suggest starting near the Seine. You could walk from Notre-Dame to the Louvre, grab a coffee.", "wav": "./data/assistant_utt0002.wav"}]}
```

## 3. Prepare data

### 3.1 Single process

```bash
python moss_tts_realtime/finetuning/prepare_data.py \
    --codec-path OpenMOSS-Team/MOSS-Audio-Tokenizer \
    --device auto \
    --input-jsonl train_raw.jsonl \
    --output-jsonl train_with_codes.jsonl
```

By default, `prepare_data.py` pre-encodes reference audio as well. If you only want target audio codes, disable it explicitly:

```bash
python moss_tts_realtime/finetuning/prepare_data.py \
    --codec-path OpenMOSS-Team/MOSS-Audio-Tokenizer \
    --device auto \
    --input-jsonl train_raw.jsonl \
    --output-jsonl train_with_codes.jsonl \
    --skip-reference-audio-codes
```

### 3.2 Multi-node / multi-GPU parallel preprocessing

`prepare_data.py` follows the `accelerate launch` multi-process model directly.  
For example, with 2 nodes and 16 GPUs in total, the dataset is split into 16 shards and each rank writes one shard:

```bash
accelerate launch --num_processes 16 moss_tts_realtime/finetuning/prepare_data.py \
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
accelerate launch moss_tts_realtime/finetuning/sft.py \
    --model-path OpenMOSS-Team/MOSS-TTS-Realtime \
    --codec-path OpenMOSS-Team/MOSS-Audio-Tokenizer \
    --train-jsonl train_with_codes.jsonl \
    --output-dir output/moss_tts_realtime_sft \
    --per-device-batch-size 1 \
    --gradient-accumulation-steps 8 \
    --learning-rate 1e-5 \
    --warmup-ratio 0.03 \
    --num-epochs 3 \
    --mixed-precision bf16
```

### 4.2 Data parallel

For single-node 8-GPU data parallel training, you can use:

```bash
accelerate launch \
    --config_file moss_tts_realtime/finetuning/configs/accelerate_ddp_8gpu.yaml \
    moss_tts_realtime/finetuning/sft.py \
    --model-path OpenMOSS-Team/MOSS-TTS-Realtime \
    --codec-path OpenMOSS-Team/MOSS-Audio-Tokenizer \
    --train-jsonl 'prepared/train_with_codes.rank*.jsonl' \
    --output-dir output/moss_tts_realtime_sft_ddp \
    --per-device-batch-size 1 \
    --gradient-accumulation-steps 4 \
    --mixed-precision bf16
```

### 4.3 Optional parameter-sharded training

For the 1.7B `OpenMOSS-Team/MOSS-TTS-Realtime` model, single-node DDP is usually enough. If you still want parameter sharding, the following approaches are supported:

- **FSDP**: shard parameters, gradients, and optimizer states across ranks
- **DeepSpeed ZeRO-3**: fully shard parameters, gradients, and optimizer states; better suited for larger models and multi-node setups

#### FSDP

```bash
accelerate launch \
    --config_file moss_tts_realtime/finetuning/configs/accelerate_fsdp_1.7b.yaml \
    moss_tts_realtime/finetuning/sft.py \
    --model-path OpenMOSS-Team/MOSS-TTS-Realtime \
    --codec-path OpenMOSS-Team/MOSS-Audio-Tokenizer \
    --train-jsonl 'prepared/train_with_codes.rank*.jsonl' \
    --output-dir output/moss_tts_realtime_sft_fsdp \
    --per-device-batch-size 1 \
    --gradient-accumulation-steps 4 \
    --mixed-precision bf16
```

#### DeepSpeed ZeRO-3

```bash
accelerate launch \
    --config_file moss_tts_realtime/finetuning/configs/accelerate_zero3_1.7b.yaml \
    moss_tts_realtime/finetuning/sft.py \
    --model-path OpenMOSS-Team/MOSS-TTS-Realtime \
    --codec-path OpenMOSS-Team/MOSS-Audio-Tokenizer \
    --train-jsonl 'prepared/train_with_codes.rank*.jsonl' \
    --output-dir output/moss_tts_realtime_sft_zero3 \
    --per-device-batch-size 1 \
    --gradient-accumulation-steps 4 \
    --mixed-precision bf16
```

ZeRO-3 requires the `deepspeed` package. If you only use DDP or FSDP, you do not need it.

### 4.4 Common tunable hyperparameters

`sft.py` now exposes the common training hyperparameters directly:

- Optimizer: `--learning-rate`, `--weight-decay`, `--adam-beta1`, `--adam-beta2`, `--adam-eps`
- LR schedule: `--lr-scheduler-type`, `--warmup-steps`, `--warmup-ratio`
- Stability: `--max-grad-norm`, `--mixed-precision`

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

## 5. One-click launcher

Run directly:

```bash
bash moss_tts_realtime/finetuning/run_train.sh
```

Common environment variables:

- `RAW_JSONL`: raw training JSONL
- `PREPARED_JSONL`: output file from `prepare_data.py`
- `TRAIN_JSONL`: optional; training input, which can be a single file, directory, or glob. If unset, it is inferred automatically from `PREPARED_JSONL`
- `OUTPUT_DIR`: training output directory
- `ACCELERATE_CONFIG_FILE`: optional; DDP / FSDP / ZeRO-3 config file
- `SKIP_PREPARE`: set to `1` to skip preprocessing and train directly from existing `TRAIN_JSONL` / `PREPARED_JSONL`
- `PREP_EXTRA_ARGS_STR`: extra arguments passed to `prepare_data.py`
- `PREP_ACCELERATE_ARGS_STR`: if you want preprocessing to also launch through `accelerate`, set this, for example `--num_processes 16` or `--config_file moss_tts_realtime/finetuning/configs/accelerate_ddp_8gpu.yaml`
- `TRAIN_EXTRA_ARGS_STR`: extra arguments passed to `sft.py`

For example, to launch with ZeRO-3:

```bash
RAW_JSONL=train_raw.jsonl \
PREPARED_JSONL=prepared/train_with_codes.jsonl \
OUTPUT_DIR=output/moss_tts_realtime_sft_zero3 \
ACCELERATE_CONFIG_FILE=moss_tts_realtime/finetuning/configs/accelerate_zero3_1.7b.yaml \
PREP_ACCELERATE_ARGS_STR='--config_file moss_tts_realtime/finetuning/configs/accelerate_ddp_8gpu.yaml' \
PREP_EXTRA_ARGS_STR='' \
TRAIN_EXTRA_ARGS_STR='--per-device-batch-size 1 --gradient-accumulation-steps 4 --num-epochs 3 --warmup-ratio 0.03 --mixed-precision bf16' \
bash moss_tts_realtime/finetuning/run_train.sh
```
