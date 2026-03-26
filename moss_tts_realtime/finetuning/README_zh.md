# MOSS-TTS-Realtime 微调教程

本目录提供基于 `MOSS-TTS-Realtime` 架构的完整微调流程：

- `prepare_data.py`: 预提取训练目标音频的 `audio_codes`，支持按 rank 切分数据并分别保存结果
- `dataset.py`: 将预处理好的数据组装为可训练的格式
- `sft.py`: 支持单卡、数据并行，以及可选的 FSDP / DeepSpeed ZeRO-3 分片训练
- `run_train.sh`: 一键启动脚本

## 1. 环境准备

先安装训练依赖：

```bash
git clone https://github.com/OpenMOSS/MOSS-TTS.git
cd MOSS-TTS
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e ".[torch-runtime,finetune]"
```

如果你的环境支持 FlashAttention 2，也可以继续沿用根目录 README 里的安装方式。

如果你准备使用 **DeepSpeed ZeRO-3**，请额外安装：

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e ".[torch-runtime,finetune-deepspeed]"
```

## 2. 输入 JSONL 格式

所有数据统一使用 `conversations` 多轮对话格式。每条记录包含一个 `conversations` 列表，列表中的每个元素为一个对话轮次，包含 `role`（`user` 或 `assistant`）、`text`（文本内容）和 `wav`（音频路径）。可选的 `ref_wav` 字段用于指定参考音频（voice cloning）。

### 2.1 单轮数据

单轮数据构造中只有一个 assistant 部分，与一般 TTS 和 Voice clone 的功能相同。如果没有参考音频则可以删去 `ref_wav` 字段。

```jsonl
{"id": "000001", "ref_wav": "./data/ref0.wav", "conversations": [{"role": "assistant", "text": "Actually, I noticed that I am very sensitive to other people's emotions.", "wav": "./data/utt0001.wav"}]}
{"id": "000002", "ref_wav": "./data/ref1.wav", "conversations": [{"role": "assistant", "text": "She said she would be here by noon.", "wav": "./data/utt0002.wav"}]}
```

### 2.2 多轮数据

在多轮数据的构造中，`user` 部分为用户与 VoiceAgent 交互的语音内容，`assistant` 部分为 MOSS-TTS-Realtime 所合成的语音，需要与 `ref_wav` 为相同说话人。所有轮次的 assistant 必须为相同说话人，user 部分可以不同。
单轮数据和多轮数据可以一起进行混合训练，以同时保证单轮和多轮能力。

```jsonl
{"id": "000003", "ref_wav": "./data/ref0.wav", "conversations": [{"role": "user", "text": "Hey, I just landed in Paris. I have about six hours before my next flight. Any ideas?", "wav": "./data/user_utt0001.wav"}, {"role": "assistant", "text": "Nice, welcome to Paris! Six hours is actually perfect for a short city walk. Are you traveling light, or do you have luggage with you?", "wav": "./data/assistant_utt0001.wav"}, {"role": "user", "text": "Just a backpack. I don't want anything too rushed.", "wav": "./data/user_utt0002.wav"}, {"role": "assistant", "text": "Got it. In that case, I'd suggest starting near the Seine. You could walk from Notre-Dame to the Louvre, grab a coffee.", "wav": "./data/assistant_utt0002.wav"}]}
```

## 3. 预处理数据

### 3.1 单进程

```bash
python moss_tts_realtime/finetuning/prepare_data.py \
    --codec-path OpenMOSS-Team/MOSS-Audio-Tokenizer \
    --device auto \
    --input-jsonl train_raw.jsonl \
    --output-jsonl train_with_codes.jsonl
```

默认情况下，`prepare_data.py` 会自动预编码参考音频；如果你只想编码目标音频，可以显式关闭：

```bash
python moss_tts_realtime/finetuning/prepare_data.py \
    --codec-path OpenMOSS-Team/MOSS-Audio-Tokenizer \
    --device auto \
    --input-jsonl train_raw.jsonl \
    --output-jsonl train_with_codes.jsonl \
    --skip-reference-audio-codes
```

### 3.2 多机多卡并行编码

`prepare_data.py` 直接按 `accelerate launch` 的多进程语义切分数据。  
例如 2 台节点、16 张卡，总共切 16 份，每个 rank 单独输出一个 shard：

```bash
accelerate launch --num_processes 16 moss_tts_realtime/finetuning/prepare_data.py \
    --codec-path OpenMOSS-Team/MOSS-Audio-Tokenizer \
    --device auto \
    --input-jsonl train_raw.jsonl \
    --output-jsonl prepared/train_with_codes.jsonl
```

输出会类似：

- `prepared/train_with_codes.rank00000-of-00016.jsonl`
- `prepared/train_with_codes.rank00001-of-00016.jsonl`
- ...
- `prepared/train_with_codes.rank00015-of-00016.jsonl`

后续训练阶段，`sft.py` 可以直接读取：

- 单个 JSONL
- 一个目录
- 一个 glob，例如 `prepared/train_with_codes.rank*.jsonl`
- 或逗号分隔的多个文件

如果你的平台已经自动注入了多机通信环境变量，`accelerate launch` 会直接复用这些信息；通常不再需要手动写 `torchrun` 风格的通信参数。

## 4. 启动训练

### 4.1 单卡基线

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

### 4.2 数据并行

单机 8 卡数据并行可直接使用模板：

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

### 4.3 可选的参数分片训练

对于 1.7B 的 `OpenMOSS-Team/MOSS-TTS-Realtime` 模型，单机 DDP 通常已经足够。如果你仍然希望做参数分片训练，也支持下面两种方式：

- **FSDP**: 参数、梯度、优化器状态按 rank 分片
- **DeepSpeed ZeRO-3**: 参数、梯度、优化器状态全分片，适合更大的模型和多机场景

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

ZeRO-3 需要 `deepspeed` 包；如果只使用 DDP 或 FSDP，则不需要额外安装它。

### 4.4 常用可调超参数

`sft.py` 现在把常见训练超参数都直接开放出来了：

- 优化器：`--learning-rate`、`--weight-decay`、`--adam-beta1`、`--adam-beta2`、`--adam-eps`
- 学习率调度：`--lr-scheduler-type`、`--warmup-steps`、`--warmup-ratio`
- 稳定性相关：`--max-grad-norm`、`--mixed-precision`

训练日志现在会直接打印：

- 带时间戳的日志前缀
- `global_batch_size` 及其计算公式
- `step_time`
- `steps_per_sec`
- `samples_per_sec`
- `eta`

### 4.5 多机训练

将配置文件里的以下字段改成你的集群值即可：

- `num_machines`
- `num_processes`
- `machine_rank`
- `main_process_ip`
- `main_process_port`

例如 2 节点 16 卡，可以在两台机器分别设置：

- 节点 0: `machine_rank: 0`
- 节点 1: `machine_rank: 1`
- `num_machines: 2`
- `num_processes: 16`

其余训练命令保持不变。

## 5. 一键启动脚本

直接运行：

```bash
bash moss_tts_realtime/finetuning/run_train.sh
```

常用环境变量：

- `RAW_JSONL`: 原始训练 JSONL
- `PREPARED_JSONL`: `prepare_data.py` 的输出文件
- `TRAIN_JSONL`: 可选；训练输入，可以是单文件、目录或 glob。如果未设置，会根据 `PREPARED_JSONL` 自动推断
- `OUTPUT_DIR`: 训练输出目录
- `ACCELERATE_CONFIG_FILE`: 可选；DDP / FSDP / ZeRO-3 配置文件
- `SKIP_PREPARE`: 设为 `1` 跳过预处理，直接使用已有的 `TRAIN_JSONL` / `PREPARED_JSONL` 训练
- `PREP_EXTRA_ARGS_STR`: 传给 `prepare_data.py` 的额外参数
- `PREP_ACCELERATE_ARGS_STR`: 如果希望预处理也通过 `accelerate` 启动，设置此变量，例如 `--num_processes 16` 或 `--config_file moss_tts_realtime/finetuning/configs/accelerate_ddp_8gpu.yaml`
- `TRAIN_EXTRA_ARGS_STR`: 传给 `sft.py` 的额外参数

例如，使用 ZeRO-3 启动：

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
