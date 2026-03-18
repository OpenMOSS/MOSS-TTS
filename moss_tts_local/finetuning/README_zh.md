# MossTTSLocal 微调教程

本目录提供基于 `MossTTSLocal` 架构的完整微调流程：

- `prepare_data.py`: 预提取训练目标音频的 `audio_codes`，支持按 rank 切分数据并分别保存结果
- `dataset.py`: 将 `text / instruction / ambient_sound / reference` 等字段统一打包成 teacher-forcing 样本
- `sft.py`: 支持单卡、数据并行，以及可选的 FSDP / DeepSpeed ZeRO-3 分片训练
- `convert_seed_tts_eval_to_jsonl.py`: 将 `seed-tts-eval` 目录改写成训练 JSONL
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

所有任务共享一套基础思路：

- `audio`: 目标训练音频路径，`prepare_data.py` 会把它编码为 `audio_codes`
- 其余字段直接映射到 `processor.build_user_message(...)`

### 2.1 MOSS-TTS

#### 纯 `text, speech` pair

这种格式不需要参考音频，已经直接支持：

```jsonl
{"audio":"./data/utt0001.wav","text":"其实我真的有发现，我是一个特别善于观察别人情绪的人。","language":"zh"}
{"audio":"./data/utt0002.wav","text":"She said she would be here by noon.","language":"en"}
```

#### 音色克隆 / 参考音频条件训练

```jsonl
{"audio":"./data/utt0001.wav","text":"其实我真的有发现，我是一个特别善于观察别人情绪的人。","ref_audio":"./data/ref.wav","language":"zh"}
{"audio":"./data/utt0002.wav","text":"She said she would be here by noon.","ref_audio":"./data/ref.wav","language":"en"}
```

### 2.2 MOSS-TTSD

MOSS-TTSD 和 MOSS-TTS 共用同一套 `prepare_data.py / sft.py`，格式也可以保持一致。  
区别只在于 `reference` 可以是多说话人列表，且列表中的元素允许为 `null`，表示该说话人没有参考音频：

```jsonl
{
  "audio":"./data/dialog_target.wav",
  "text":"[S1] 这是说话人一的前缀。 [S2] 这是说话人二的前缀。 [S1] 下面开始新一轮对话。",
  "reference":["./data/s1_ref.wav", null],
  "language":"zh"
}
```

说明：

- `prepare_data.py` 会编码 `audio`
- 默认会额外编码 `reference` / `ref_audio` / `reference_audio` 中出现的参考音频
- `reference` 中的 `null` 会在训练时保留为 `None`，不会被错误编码
- 不需要额外的 `prompt_audio` 字段；自回归续写能力本身就包含在标准 teacher-forcing 训练里

### 2.3 MOSS-SoundEffect

MOSS-SoundEffect 同样共享这套流程，只需要把用户侧字段换成 `ambient_sound`：

```jsonl
{"audio":"./data/rain.wav","ambient_sound":"雷声隆隆，雨声淅沥。"}
{"audio":"./data/footsteps.wav","ambient_sound":"清晰脚步声在水泥地面回响，节奏稳定。","tokens":160}
```

### 2.4 MOSS-VoiceGenerator

MOSS-VoiceGenerator 共享同一训练流，只需要使用 `text + instruction`：

```jsonl
{"audio":"./data/old_man.wav","text":"哎呀，我的老腰啊，这年纪大了就是不行了。","instruction":"疲惫沙哑的老年声音缓慢抱怨，带有轻微呻吟。"}
{"audio":"./data/tavern.wav","text":"Hey there, stranger!","instruction":"Hearty, jovial tavern owner's voice, loud and welcoming with a slightly gruff tone."}
```

## 3. 预处理数据

### 3.1 单进程

```bash
python moss_tts_local/finetuning/prepare_data.py \
    --model-path OpenMOSS-Team/MOSS-TTS-Local-Transformer \
    --codec-path OpenMOSS-Team/MOSS-Audio-Tokenizer \
    --device auto \
    --input-jsonl train_raw.jsonl \
    --output-jsonl train_with_codes.jsonl
```

默认情况下，`prepare_data.py` 会自动预编码参考音频；如果你只想编码目标音频，可以显式关闭：

```bash
python moss_tts_local/finetuning/prepare_data.py \
    --model-path OpenMOSS-Team/MOSS-TTS-Local-Transformer \
    --codec-path OpenMOSS-Team/MOSS-Audio-Tokenizer \
    --device auto \
    --input-jsonl train_raw.jsonl \
    --output-jsonl train_with_codes.jsonl \
    --skip-reference-audio-codes
```

### 3.2 多机多卡并行编码

`prepare_data.py` 现在直接按 `accelerate launch` 的多进程语义切分数据。  
例如 2 台节点、16 张卡，总共切 16 份，每个 rank 单独输出一个 shard：

```bash
accelerate launch --num_processes 16 moss_tts_local/finetuning/prepare_data.py \
    --model-path OpenMOSS-Team/MOSS-TTS-Local-Transformer \
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

### 4.2 数据并行

单机 8 卡数据并行可直接使用模板：

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

### 4.3 可选的参数分片训练

对于 1.7B 的 `MOSS-TTS-Local-Transformer` 模型，单机 DDP 通常已经足够。如果你仍然希望做参数分片训练，也支持下面两种方式：

- **FSDP**: 参数、梯度、优化器状态按 rank 分片
- **DeepSpeed ZeRO-3**: 参数、梯度、优化器状态全分片，适合更大的模型和多机场景

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

ZeRO-3 需要 `deepspeed` 包；如果只使用 DDP 或 FSDP，则不需要额外安装它。

### 4.4 常用可调超参数

`sft.py` 现在把常见训练超参数都直接开放出来了：

- 优化器：`--learning-rate`、`--weight-decay`、`--adam-beta1`、`--adam-beta2`、`--adam-eps`
- 学习率调度：`--lr-scheduler-type`、`--warmup-steps`、`--warmup-ratio`
- 稳定性相关：`--max-grad-norm`、`--gradient-checkpointing`、`--mixed-precision`
- RVQ 多头 loss 加权：`--channelwise-loss-weight`

`--channelwise-loss-weight` 支持两种写法：

- `n_vq + 1` 个值：`text_head,vq0,...,vqN`
- 两个值：`text_weight,total_audio_weight`

默认值是 `1,32`。这表示文本头与音频的每一个头的权重都相等。

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

## 5. 快速推理验证

`sft.py` 保存的每个 checkpoint 目录现在都会附带模型配置、运行时 Python 文件、tokenizer 文件和 processor 元数据，因此可以直接对这个目录调用 `from_pretrained`：

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
reference_audio = "./assets/audio/reference_zh_0.wav"
text = "今天我们继续把 MOSS-TTS Local 的微调流程跑通。"

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

## 6. 一键脚本

直接使用：

```bash
bash moss_tts_local/finetuning/run_train.sh
```

常见环境变量：

- `RAW_JSONL`: 原始训练 JSONL
- `PREPARED_JSONL`: `prepare_data.py` 输出文件
- `TRAIN_JSONL`: 可选；训练输入，可以是单文件、目录或 glob。默认会自动从 `PREPARED_JSONL` 推断
- `OUTPUT_DIR`: 训练输出目录
- `ACCELERATE_CONFIG_FILE`: 可选，填 DDP / FSDP / ZeRO-3 配置
- `SKIP_PREPARE`: 设为 `1` 时跳过预处理，直接用现有的 `TRAIN_JSONL` / `PREPARED_JSONL` 进入训练
- `PREP_EXTRA_ARGS_STR`: 额外传给 `prepare_data.py`
- `PREP_ACCELERATE_ARGS_STR`: 如果你希望预处理也通过 `accelerate launch` 并行启动，可在脚本里设置这组参数，例如 `--num_processes 16` 或 `--config_file moss_tts_local/finetuning/configs/accelerate_ddp_8gpu.yaml`
- `TRAIN_EXTRA_ARGS_STR`: 额外传给 `sft.py`

例如用 ZeRO-3 一键启动：

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

## 7. 任务格式补充说明

其余任务不需要重新实现训练器，直接在 JSONL 中切换字段即可：

- **MOSS-TTS**: 使用 `text`，可选 `ref_audio`
- **MOSS-TTSD**: 使用 `text + reference`，其中 `reference` 支持 `null`
- **MOSS-SoundEffect**: 使用 `ambient_sound`
- **MOSS-VoiceGenerator**: 使用 `text + instruction`

共享字段：

- `audio`: 必填，目标音频
- `language`, `tokens`, `quality`, `sound_event`, `ambient_sound`, `instruction`: 按任务需要填写

共享脚本：

- 数据准备统一使用 `prepare_data.py`
- 训练统一使用 `sft.py`
- `train-jsonl` 支持单文件、目录、glob、多文件列表
