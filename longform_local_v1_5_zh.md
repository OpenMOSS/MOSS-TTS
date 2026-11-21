# MOSS-TTS Local v1.5 长篇生成与单人 LoRA 微调

本文档整理 `MOSS-TTS-Local-Transformer-v1.5`（双 AR 架构）的长篇生成完整流程：环境安装、预训练模型长篇生成、单人 LoRA 微调、LoRA 长篇生成，以及输出指标与常见问题。所有命令均为 Windows PowerShell，在仓库根目录执行；反引号 `` ` `` 续行时必须是行尾最后一个字符。

相关文件：

| 文件 | 作用 |
|---|---|
| `scripts/longform_local_v1_5.py` | 长篇生成：codec 层续写、最后统一解码、逐段指标 |
| `scripts/merge_ljspeech.py` | 把连续编号的短切片合并为训练样本 |
| `moss_tts_local_v1.5/finetuning/prepare_data.py` | 预编码训练音频为 codec codes |
| `moss_tts_local_v1.5/finetuning/sft_lora.py` | LoRA 训练 |

---

## 1. 环境安装

```powershell
# 创建并激活虚拟环境（已有 .venv 可跳过创建）
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip

# torch（CUDA 12.8）+ LoRA 微调 + 长篇推理依赖
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e ".[torch-runtime,finetune-lora,longform]"

# 验证 GPU（RTX 50 系列需在 arch list 中看到 sm_120）
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0)); print(torch.cuda.get_arch_list())"

# 验证关键包
python -c "import transformers, accelerate, peft, torchaudio; print(transformers.__version__, accelerate.__version__, peft.__version__, torchaudio.__version__)"
```

| 依赖组 | 内容 | 用途 |
|---|---|---|
| `torch-runtime` | torch / torchaudio（cu128）、transformers 5.0.0、accelerate | 基础运行 |
| `finetune-lora` | peft、tensorboard | `sft_lora.py` 训练 |
| `longform` | faster-whisper、bitsandbytes | 长篇脚本的 `--asr-check`、`--quant` |

说明：

- 不需要 flash-attn：长篇脚本默认 `sdpa`，训练脚本缺少 flash-attn 时也会回退到 sdpa。
- 单卡训练不需要 `finetune`（wandb）和 `finetune-deepspeed`。
- 模型首次运行时从 Hugging Face 自动下载；下载慢可先设置镜像：`$env:HF_ENDPOINT = "https://hf-mirror.com"`。

---

## 2. 长篇生成原理

`longform_local_v1_5.py` 把长文本切成若干段，逐段生成，并在 **codec codes 层面**接续：

- 上一段生成的 codes 和原文直接作为下一段的续写前缀，不解码再编码，也不需要 ASR。
- 所有段的 codes 拼接后**统一解码**为 `combined.wav`，没有波形拼接的接缝。
- 续写走官方 continuation 路径：`text = 前缀原文 + 本段原文`，前缀音频放在 assistant 消息里。

### 2.1 三种模式

| `--mode` | 第 k 段（k ≥ 2）的输入 | 需要参考音频 | 特点 |
|---|---|---|---|
| `b`（默认） | 前缀 = 参考 codes + 静音 + 第 k−1 段 codes；文本 = 参考转写 + 上一段原文 + 本段原文 | 是，且需参考转写 | 参考音频每段都在前缀里，用于锚定音色 |
| `continuation` | 前缀 = 第 k−1 段 codes；文本 = 上一段原文 + 本段原文 | 否 | 衔接最自然；无参考锚定，可能逐段漂移；LoRA 推荐 |
| `clone` | 每段独立克隆，无前缀 | 是 | 音色有参考兜底，但段间不连贯，每段从零起步，较易崩坏 |

**第 1 段**由是否提供 `--reference-audio` 决定：提供则用参考音频克隆，不提供则直接生成（适合 LoRA，音色在权重里）。

### 2.2 分段长度

- 只在句末（`. ! ?`、`。！？…`、换行）切段；单句超过上限时才退回逗号或单词边界。能识别 `Mr.`、`Dr.`、`e.g.`、`J. K.` 等缩写。
- **`--chunk-seconds`（默认 20）**：按目标秒数分段，适用于所有语言，不需要参考转写。
  - 用加权字数估算时长，权重沿用 OmniVoice：拉丁/西里尔字母 1.0、汉字 3.0、假名 2.2、韩文 2.5、空格 0.2、标点 0.5、数字 3.5 等。
  - 第 1 段按默认语速（每秒 14.1 个加权字数，OmniVoice 的无参考默认值）估算；之后每生成一段，就用"已生成的总加权字数 ÷ 总秒数"校准语速，再决定下一段装多少句。被截断的段不计入。
  - 停顿标记产生的时长也计入，因此每段总时长（含停顿）贴近目标值。
  - 只能控制上限：按整句装段，实际每段通常为目标的 60%–100%；第 1 段可能偏离较多（默认语速与实际不同）。
- 每段的实际时长、使用的语速与预算记录在 `manifest.json` 的 `seconds`、`rate_used`、`budget`。
- 建议每段 15–25 秒。LoRA 推理时续写总长约为段长的 2 倍，不要超出训练样本的最大时长（见 4.3）。
- 参考音频建议 8–15 秒，B 方案需提供准确转写（末尾带标点，缺失时自动补）。

### 2.3 显存与加载方式

脚本分三阶段运行，任何时刻 GPU 上只有一个大模型：

1. codec 编码参考音频（TTS 模型尚未加载）。
2. TTS 模型逐段生成 codes（codec 在 CPU）。
3. 释放 TTS 模型，codec 在 GPU 上逐段解码并生成 `combined.wav`；ASR 与说话人相似度在 CPU 上计算。

显存自动分配：加载 TTS 模型前读取当前空闲显存，减去留给生成过程的余量（至少 1.5 GiB，或总显存的 8%），剩余部分作为权重放置上限。显存足够（如 RTX 5090）时模型全部在 GPU；不够（如 8 GB 显卡）时，放不下的层自动 offload 到 CPU。启动日志会打印空闲显存与预算。

| 显卡 | 建议参数 |
|---|---|
| 8 GB | 无需设置（部分层自动 offload，较慢）；或 `--quant int8` / `--quant nf4`（更快，需 bitsandbytes） |
| 32 GB（如 RTX 5090） | 无需设置，模型全部在 GPU |

预算按启动时的空闲显存计算，GPU 上同时运行其他程序会让更多层被 offload。启动日志中 `TTS modules offloaded to CPU: 0` 表示没有 offload。

---

## 3. 预训练模型长篇生成

以英文、RTX 5090 为例。三组对比实验：

```powershell
# B 方案
python scripts\longform_local_v1_5.py `
  --input-txt long.txt `
  --reference-audio ref.wav `
  --reference-text "Exact transcript of the reference audio." `
  --output-dir outputs\lf_b `
  --mode b `
  --language English `
  --chunk-seconds 20 `
  --asr-check `
  --speaker-sim

# 对照组：纯续写（第 1 段用参考音频克隆）
python scripts\longform_local_v1_5.py `
  --input-txt long.txt `
  --reference-audio ref.wav `
  --output-dir outputs\lf_cont `
  --mode continuation `
  --language English `
  --chunk-seconds 20 `
  --asr-check `
  --speaker-sim

# 对照组：每段独立克隆
python scripts\longform_local_v1_5.py `
  --input-txt long.txt `
  --reference-audio ref.wav `
  --output-dir outputs\lf_clone `
  --mode clone `
  --language English `
  --chunk-seconds 20 `
  --asr-check `
  --speaker-sim
```

- `--language` 默认 `Chinese`，英文必须显式传 `English`；它同时决定模型语言标签和 ASR 语言。
- 输出目录必须为空。正式生成前可在命令末尾加 `--dry-run`，只写出切分计划到 `manifest.json`。
- 对比时保持各组 `--language`、`--chunk-seconds`、`--seed`、采样参数、`--asr-device` 一致；同一台机器上跑完三组更可比。

---

## 4. 单人 LoRA 微调

### 4.1 录音与切片要求

- 只用同一说话人；干净、无背景音乐、少混响；话筒与环境一致。
- 时长至少 30 分钟，1–3 小时更好；风格与目标用途一致（例如讲道、有声书叙述）。
- 从连续录音中按句切片，**切点落在静音中并保留首尾静音**，这样合并后句间停顿接近原始录音。
- 不要对每个切片单独做音量归一化，否则合并后句间会有音量跳变。
- 切片文件名末尾需为连续递增编号（如 `clip_1.wav`、`clip_2.wav`），合并脚本靠它判断连续性。
- 转写需逐字对应音频并**包含标点**，模型从标点学习停顿。

### 4.2 清单格式

UTF-8 文本，`音频路径|转写文本`，每行一条：

```text
clip_1.wav|In the beginning was the Word,
clip_2.wav|and the Word was with God.
```

相对路径默认相对于清单所在目录下的 `wavs` 文件夹；其他位置用 `--audio-root` 指定。

### 4.3 合并为训练样本

推理时每次输入的音频总长约为"上一段 + 本段"（`--chunk-seconds 20` 时约 40 秒；B 方案再加参考约 50 秒）。训练样本覆盖到 60 秒即可，主力放在 20–50 秒：

```powershell
# 先 dry-run 检查合并统计
python scripts\merge_ljspeech.py `
  --input "data\metadata.txt" `
  --output-dir "data\merged_en" `
  --text-joiner " " `
  --language English `
  --max-seconds 60 `
  --target-dist "10-20:0.15,20-35:0.35,35-50:0.35,50-60:0.15" `
  --seed 7 `
  --dry-run

# 确认后去掉 --dry-run 正式合并（输出目录必须尚不存在）
python scripts\merge_ljspeech.py `
  --input "data\metadata.txt" `
  --output-dir "data\merged_en" `
  --text-joiner " " `
  --language English `
  --max-seconds 60 `
  --target-dist "10-20:0.15,20-35:0.35,35-50:0.35,50-60:0.15" `
  --seed 7
```

- 英文必须加 `--text-joiner " " --language English`（默认分别是空字符串和 `Chinese`）。
- dry-run 输出中 `singletons` 接近切片总数，说明编号不连续、没有合并起来，需检查文件名。
- 清单中找不到的音频会被静默跳过，应核对"选中条数/总条数"。
- 合并后抽听几条，确认句间停顿自然、没有音量跳变。
- 若推理时改了 `--chunk-seconds`，按"上一段 + 本段（B 方案加参考）"的总时长 × 1.2 调整 `--max-seconds`。

合并输出：`data\merged_en\wavs\*.wav`、`train_raw.jsonl`（训练输入，音频为绝对路径）、`metadata.txt`、`sources.jsonl`（每条样本的来源切片）。

训练 JSONL 不含 `ref_audio`：长篇推理（续写、B 方案）时 Reference 为 None，训练格式与之一致。

### 4.4 预编码

```powershell
python moss_tts_local_v1.5\finetuning\prepare_data.py `
  --model-path OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 `
  --codec-path OpenMOSS-Team/MOSS-Audio-Tokenizer-v2 `
  --codec-weight-dtype fp32 `
  --codec-compute-dtype bf16 `
  --device auto `
  --input-jsonl data\merged_en\train_raw.jsonl `
  --output-jsonl data\merged_en\train_with_codes.jsonl
```

### 4.5 训练

需要大显存显卡（如 RTX 5090）；8 GB 显卡无法训练。

```powershell
accelerate launch --num_processes 1 --num_machines 1 --mixed_precision bf16 --dynamo_backend no `
  moss_tts_local_v1.5\finetuning\sft_lora.py `
  --model-path OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 `
  --train-jsonl data\merged_en\train_with_codes.jsonl `
  --output-dir output\moss_tts_local_v1_5_lora_en `
  --per-device-batch-size 1 `
  --gradient-accumulation-steps 4 `
  --learning-rate 1e-4 `
  --warmup-ratio 0.03 `
  --num-epochs 15 `
  --save-every-epochs 1 `
  --mixed-precision bf16 `
  --gradient-checkpointing `
  --gradient-checkpointing-scope all `
  --lora-r 16 `
  --lora-alpha 32 `
  --lora-dropout 0.05
```

- `accelerate launch` 后的四个参数只用于消除默认值警告，可省略。
- 按优化步数而不是轮数规划：步数 ≈ 样本条数 ÷ `--gradient-accumulation-steps` × `--num-epochs`（batch size 为 1）。单人 LoRA 以 500–800 步为目标。例如 1.5 小时数据合并后约 160 条（以 dry-run 打印的输出组数为准），梯度累积 4 时每轮约 40 步，15 轮约 600 步。官方示例的 3 轮、梯度累积 8 只有约 60 步，明显不足。
- 数据更多或更少时，按上式调整 `--num-epochs`；15 轮后音色仍不够像，可继续加轮数或把 `--lora-r` 调到 32。
- 每个 checkpoint 只保存 PEFT adapter，基础模型仍由 `--model-path` 指定。
- 选 checkpoint 不要只看 loss，也不要直接用最后一轮：取第 3、6、9、12、15 轮的 checkpoint，用第 5 节命令生成同一段文本试听。音色不像 → 继续训练；出现电音、复读、读错或语气僵硬 → 过拟合，回退到前一个 checkpoint，或把学习率降到 `5e-5`。

---

## 5. LoRA 长篇生成（最终生成命令）

不提供参考音频，第 1 段直接用 LoRA 音色生成，之后逐段续写，并在句末插入显式停顿标记：

```powershell
python scripts\longform_local_v1_5.py `
  --input-txt long.txt `
  --output-dir outputs\lf_final `
  --mode continuation `
  --lora-dir output\moss_tts_local_v1_5_lora_en\checkpoint-last `
  --language English `
  --chunk-seconds 20 `
  --sentence-pause 0.4 `
  --paragraph-pause 0.8 `
  --clause-pause 0.2 `
  --asr-check
```

- `--lora-dir` 换成试听选出的最佳 checkpoint，不一定是 `checkpoint-last`。
- 每次生成换一个新的空输出目录；最终成品是 `combined.wav`，逐段 wav 与 `manifest.json` 用于排查。
- `--asr-check` 用于检查读错的段，确认参数稳定后可去掉以加快生成。
- `--lora-dir` 默认合并进基础模型（与官方 App 一致）；同时使用 `--quant` 时以 PEFT adapter 形式挂载、不合并。
- 对照组：加 `--reference-audio ref.wav --reference-text "..." --mode b`，观察 LoRA 再加参考锚定是否更好。
- 提供 `--reference-audio` 会让第 1 段改为克隆；因此纯 LoRA 组无法计算 `--speaker-sim`（该参数要求提供参考音频）。

### 5.1 停顿参数

v1.5 支持 `[pause X.Ys]` 显式停顿。脚本按位置自动插入，三个参数默认均为 0（关闭）：

| 位置 | 参数 | 建议值（秒） |
|---|---|---|
| 句末（`. ! ?`、`。！？…`） | `--sentence-pause` | 0.3–0.6 |
| 段落结尾的那句 | `--paragraph-pause` | 0.7–1.2 |
| 段尾停在逗号、分号、冒号、破折号（含中文 `，、；：`），即超长句被迫切开时 | `--clause-pause` | 0.15–0.25 |
| 句内逗号 | 不加 | 由模型按标点自然停顿 |
| 全文最后一句、`Mr.` 等缩写之后 | 不加 | — |

示例：一段以 `One, two. Three, four,` 结束时，送入模型的文本为 `One, two. [pause 0.4s] Three, four, [pause 0.2s]`。实际送入模型的文本记录在 `manifest.json` 的 `model_text`，CER 仍按原文计算。

### 5.2 试听后的调整

| 听到的问题 | 调整 |
|---|---|
| 句间停顿仍偏短 | `--sentence-pause` 调到 0.5–0.6 |
| 停顿过长、拖沓 | `--sentence-pause` 降到 0.3 |
| 段落之间区分不明显 | `--paragraph-pause` 调到 1.0–1.2 |
| 停顿标记被念出来、停顿处有杂音 | LoRA 未保留停顿标记能力：去掉三个停顿参数，改为在合并训练数据时补足句间停顿后重新训练 |
| 个别段读错、复读 | 查看该段 `cer`、`frames_per_char`，换 `--seed` 重新生成 |

---

## 6. 输出与指标

输出目录内容：

| 文件 | 内容 |
|---|---|
| `0001.wav` / `0001.pt` … | 每段音频与 codes（逐段单独解码，便于试听） |
| `combined.wav` / `combined.pt` | 全部 codes 拼接后统一解码的整篇音频 |
| `manifest.json` | 参数、切分计划、每段指标 |
| `reference_text.txt` | B 方案使用的参考转写 |

`manifest.json` 中每段的字段：

| 字段 | 含义 | 用法 |
|---|---|---|
| `strategy` | `clone` / `direct` / `b` / `continuation` | 该段实际生成方式 |
| `seconds` | 时长 | 检查分段是否贴近目标 |
| `chars` / `weight` | 字符数 / 加权字数 | 分段大小 |
| `rate_used` / `budget` | 该段使用的语速（加权字数/秒）与加权字数上限（`--chunk-seconds`） | 查看语速校准过程 |
| `frames_per_char` | 每字符帧数 | 明显偏高 → 拖长或复读；偏低 → 吞字 |
| `truncated` | 是否达到 `--max-new-tokens` | `true` 时该段可能被截断 |
| `prefix_frames` | 前缀帧数 | 检查前缀长度 |
| `cer` / `asr_text` | 字错率与识别文本（`--asr-check`） | 定位漏读、错读 |
| `speaker_sim` | 与参考音频的相似度（`--speaker-sim`） | 按段号看曲线是否下降（音色漂移） |

---

## 7. 常见问题

**`sft_lora.py` 报 `AttributeError: module 'sft' has no attribute 'bnb'`**
安装了 bitsandbytes 时，旧版 `sft_lora.py` 会替换 `sft.bnb`，而官方 `sft.py` 并未导入 bitsandbytes。已修复为仅在 `sft` 存在 `bnb` 时才替换；请使用仓库中的新版 `sft_lora.py`。

**clone 模式崩坏多、有电音**
clone 每段从零起步，无前缀约束，失败概率高；`--audio-temperature 1.7` 偏高时高层 RVQ 容易跑偏。可尝试 `--audio-temperature 1.2` 或减小 `--chunk-seconds`；长篇建议用 `b` 或 `continuation`。

**从某一段开始声音越来越小**
续写会继承前缀的响度，而生成的前缀 codes 不经过响度归一化，形成逐段下降的正反馈。LoRA 能减轻但不能消除。可行的修正：前缀响度归一化（解码 → 归一 → 再编码作前缀）、逐段响度检测并重生成、最终整篇响度归一化（如 ACX：−23 至 −18 dB RMS，峰值 ≤ −3 dB）。

**语速偏赶、句间停顿不明显**
多半来自训练数据：切片时裁掉了首尾静音，合并后句子紧贴，LoRA 学到"句间不停顿"。推理侧可用 v1.5 的显式停顿标记补救：`--sentence-pause 0.4 --paragraph-pause 0.8` 会在每个句末插入 `[pause 0.4s]`，段落结尾插入 `[pause 0.8s]`（全文最后一句不加）；超长句被迫在逗号、分号、冒号、破折号处切段时，可用 `--clause-pause 0.2` 只在该段末尾补一个短停顿，句内逗号不加标记、由模型自行停顿。实际送入模型的文本记录在 `manifest.json` 的 `model_text`，CER 仍按原文计算。根治需在合并训练数据时补足句间停顿后重新训练。

**生成很慢**
检查启动日志的 offload 数量。若有层被 offload，可关闭其他占用显存的程序后重跑（预算按当时的空闲显存计算），或使用 `--quant int8`。

**`accelerate launch` 提示若干参数使用默认值**
仅为警告，混合精度由脚本的 `--mixed-precision bf16` 控制；可按 4.5 节在 `accelerate launch` 后显式传参消除。
