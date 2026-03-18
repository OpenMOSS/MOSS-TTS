#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-OpenMOSS-Team/MOSS-TTS-Local-Transformer}"
CODEC_PATH="${CODEC_PATH:-OpenMOSS-Team/MOSS-Audio-Tokenizer}"

RAW_JSONL="${RAW_JSONL:-train_raw.jsonl}"
PREPARED_JSONL="${PREPARED_JSONL:-train_with_codes.jsonl}"
TRAIN_JSONL="${TRAIN_JSONL:-}"
OUTPUT_DIR="${OUTPUT_DIR:-output/moss_tts_local_sft}"

PREP_DEVICE="${PREP_DEVICE:-auto}"
ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"

PREP_ACCELERATE_ARGS_STR="${PREP_ACCELERATE_ARGS_STR:-}"
PREP_EXTRA_ARGS_STR="${PREP_EXTRA_ARGS_STR:-}"
TRAIN_EXTRA_ARGS_STR="${TRAIN_EXTRA_ARGS_STR:---per-device-batch-size 1 --gradient-accumulation-steps 8 --learning-rate 1e-5 --num-epochs 3 --mixed-precision bf16 --channelwise-loss-weight 1,32 --gradient-checkpointing}"

PREP_ACCELERATE_ARGS=()
PREP_EXTRA_ARGS=()
TRAIN_EXTRA_ARGS=()

if [[ -n "${PREP_ACCELERATE_ARGS_STR}" ]]; then
  read -r -a PREP_ACCELERATE_ARGS <<< "${PREP_ACCELERATE_ARGS_STR}"
fi
if [[ -n "${PREP_EXTRA_ARGS_STR}" ]]; then
  read -r -a PREP_EXTRA_ARGS <<< "${PREP_EXTRA_ARGS_STR}"
fi
if [[ -n "${TRAIN_EXTRA_ARGS_STR}" ]]; then
  read -r -a TRAIN_EXTRA_ARGS <<< "${TRAIN_EXTRA_ARGS_STR}"
fi

derive_shard_glob() {
  local path="$1"
  if [[ "$path" == *.jsonl ]]; then
    printf '%s\n' "${path%.jsonl}.rank*.jsonl"
  else
    printf '%s\n' "${path}.rank*"
  fi
}

if [[ -z "${TRAIN_JSONL}" ]]; then
  TRAIN_JSONL="${PREPARED_JSONL}"
  if [[ -n "${PREP_ACCELERATE_ARGS_STR}" ]]; then
    TRAIN_JSONL="$(derive_shard_glob "${PREPARED_JSONL}")"
  elif [[ ! -e "${PREPARED_JSONL}" ]]; then
    SHARD_GLOB="$(derive_shard_glob "${PREPARED_JSONL}")"
    if compgen -G "${SHARD_GLOB}" > /dev/null; then
      TRAIN_JSONL="${SHARD_GLOB}"
    fi
  fi
fi

if [[ "${SKIP_PREPARE}" != "1" ]]; then
  if [[ -n "${PREP_ACCELERATE_ARGS_STR}" ]]; then
    accelerate launch "${PREP_ACCELERATE_ARGS[@]}" moss_tts_local/finetuning/prepare_data.py \
      --model-path "${MODEL_PATH}" \
      --codec-path "${CODEC_PATH}" \
      --device "${PREP_DEVICE}" \
      --input-jsonl "${RAW_JSONL}" \
      --output-jsonl "${PREPARED_JSONL}" \
      "${PREP_EXTRA_ARGS[@]}"
  else
    python moss_tts_local/finetuning/prepare_data.py \
      --model-path "${MODEL_PATH}" \
      --codec-path "${CODEC_PATH}" \
      --device "${PREP_DEVICE}" \
      --input-jsonl "${RAW_JSONL}" \
      --output-jsonl "${PREPARED_JSONL}" \
      "${PREP_EXTRA_ARGS[@]}"
  fi
fi

if [[ -n "${ACCELERATE_CONFIG_FILE}" ]]; then
  accelerate launch --config_file "${ACCELERATE_CONFIG_FILE}" moss_tts_local/finetuning/sft.py \
    --model-path "${MODEL_PATH}" \
    --codec-path "${CODEC_PATH}" \
    --train-jsonl "${TRAIN_JSONL}" \
    --output-dir "${OUTPUT_DIR}" \
    "${TRAIN_EXTRA_ARGS[@]}"
else
  accelerate launch moss_tts_local/finetuning/sft.py \
    --model-path "${MODEL_PATH}" \
    --codec-path "${CODEC_PATH}" \
    --train-jsonl "${TRAIN_JSONL}" \
    --output-dir "${OUTPUT_DIR}" \
    "${TRAIN_EXTRA_ARGS[@]}"
fi
