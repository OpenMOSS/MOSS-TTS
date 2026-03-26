# Norwegian LoRA for MOSS-TTS

A community-contributed LoRA adapter fine-tuned on Norwegian speech data.

**Contributor:** [Martin Bergo](https://x.com/martinbergo) at [Tosee](https://tosee.no/)

## Dataset

The dataset is available at: [NbAiLab/NST](https://huggingface.co/datasets/NbAiLab/NST)

## Training Configuration

| Parameter | Value |
|---|---|
| Base model | `OpenMOSS-Team/MOSS-TTS` |
| LoRA target modules | `mlp` (gate_proj, up_proj, down_proj) |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Learning rate | 2e-6 |
| Warmup steps | 100 |
| Max train steps | 30,000 |
| Weight decay | 0.01 |
| Max grad norm | 0.5 |

## Model Weights

The released LoRA adapter weights are available on Hugging Face:

[ToSee-Norway/MOSS-TTS-Norwegian-LoRA](https://huggingface.co/ToSee-Norway/MOSS-TTS-Norwegian-LoRA)

```bash
huggingface-cli download ToSee-Norway/MOSS-TTS-Norwegian-LoRA --local-dir weights/MOSS-TTS-Norwegian-LoRA
```

## Files

- `train_lora.py` — LoRA fine-tuning script for MOSS-TTS on Norwegian data.
- `run_train.sh` — Example launch script used for the released adapter.

## Usage

```bash
python train_lora.py \
  --manifest-train /path/to/moss_tts_train.jsonl \
  --manifest-val /path/to/moss_tts_val.jsonl \
  --tokenized-dir /path/to/tokenized \
  --output-dir /path/to/checkpoints \
  --trainable-lora-modules mlp \
  --lora-r 16 \
  --lora-alpha 32 \
  --lr 2e-6 \
  --max-train-steps 30000
```

## Citation

If you use this adapter, please cite the contributor and their company:

- **Martin Bergo** — [https://x.com/martinbergo](https://x.com/martinbergo)
- **Tosee** — [https://tosee.no/](https://tosee.no/)

And cite the MOSS-TTS paper:

```bibtex
@misc{gong2026mossttstechnicalreport,
      title={MOSS-TTS Technical Report},
      author={Yitian Gong and Botian Jiang and Yiwei Zhao and Yucheng Yuan and Kuangwei Chen and Yaozhou Jiang and Cheng Chang and Dong Hong and Mingshu Chen and Ruixiao Li and Yiyang Zhang and Yang Gao and Hanfu Chen and Ke Chen and Songlin Wang and Xiaogui Yang and Yuqian Zhang and Kexin Huang and ZhengYuan Lin and Kang Yu and Ziqi Chen and Jin Wang and Zhaoye Fei and Qinyuan Cheng and Shimin Li and Xipeng Qiu},
      year={2026},
      eprint={2603.18090},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2603.18090},
}
```
