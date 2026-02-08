# Copyright 2026 OpenMOSS team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import torch
import torchaudio
from ..core.modeling.processing_moss_tts import MossTTSDelayProcessor
from ..core.modeling.modeling_moss_tts import MossTTSDelayModel

# !!! Enable the following lines only when using SDPA !!!
torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_flash_sdp(True) # Keep these enabled as fallbacks
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

pretrained_model_name_or_path = os.environ.get("MOSS_TTS_DELAY_CHECKPOINT_PATH", "OpenMOSS-Team/MOSS-TTS")
codec_path = os.environ.get("MOSS_AUDIO_TOKENIZER_PATH", "OpenMOSS-Team/MOSS-Audio-Tokenizer")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

prompt_wav_zh = "examples/prompt_audio/prompt_zh.wav"
prompt_wav_en = "examples/prompt_audio/prompt_en.wav"
if not prompt_wav_zh or not prompt_wav_en:
    raise ValueError(
        "Missing prompt wav paths. Set MOSS_TTS_PROMPT_WAV_ZH and MOSS_TTS_PROMPT_WAV_EN."
    )

processor = MossTTSDelayProcessor.from_pretrained(
    pretrained_model_name_or_path,
    trust_remote_code=True,
    codec_path=codec_path,
)
if getattr(processor, "audio_tokenizer", None) is not None:
    processor.audio_tokenizer = processor.audio_tokenizer.to(device)


conversations = [
    [
        processor.build_user_message(
            text="而起因竟是人气动画片瑞克和莫迪中。他大声呼唤妈妈，声音穿过重重岩层，传入三娘耳中。",
        ),
        processor.build_assistant_message(audio_codes_list=[prompt_wav_zh]),
    ],
    [
        processor.build_user_message(
            text="The practitioners then placed their hands through holes in the screen. Some of these include the sambar, wild boar, South China rabbit and badger.",
        ),
        processor.build_assistant_message(audio_codes_list=[prompt_wav_en]),
    ],
]

batch = processor(conversations, mode="continuation")
input_ids = batch["input_ids"].to(device)
attention_mask = batch["attention_mask"].to(device)

model = MossTTSDelayModel.from_pretrained(
    pretrained_model_name_or_path,
    trust_remote_code=True,
    attn_implementation="sdpa",
    torch_dtype=dtype,
).to(device)
model.eval()

outputs = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=2000,
)

messages = processor.decode(outputs)

os.makedirs("inference_root", exist_ok=True)
for sample_idx, message in enumerate(messages):
    if message is None:
        continue
    for seg_idx, audio in enumerate(message.audio_codes_list):
        # audio is a waveform tensor after decode_audio_codes
        out_path = os.path.join(
            "inference_root", f"continuation_sample{sample_idx}_seg{seg_idx}.wav"
        )
        torchaudio.save(
            out_path, audio.unsqueeze(0), processor.model_config.sampling_rate
        )
