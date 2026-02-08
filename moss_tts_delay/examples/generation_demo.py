import os
from pathlib import Path
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

processor = MossTTSDelayProcessor.from_pretrained(
    pretrained_model_name_or_path,
    trust_remote_code=True,
    codec_path=codec_path,
)
if getattr(processor, "audio_tokenizer", None) is not None:
    processor.audio_tokenizer = processor.audio_tokenizer.to(device)


# ====== Batch demo (2 direct TTS + 2 voice-clone examples) ======
text_tts_1 = """我们正站在 AI 时代的门槛上。
人工智能不再只是实验室里的概念，而是正在进入每一个行业、每一次创作、每一种决策之中。它学会了看、听、说、思考，也开始成为人类能力的延伸。AI 并不是取代人，而是放大人的创造力，让知识更平等、效率更高、想象力走得更远。一个由人类与智能系统共同塑造的新时代，已经到来。"""
text_tts_2 = """亲爱的你，
你好呀。

今天，我想用最认真、最温柔的声音，对你说一些重要的话。
这些话，像一颗小小的星星，希望能在你的心里慢慢发光。

首先，我想祝你——
每天都能平平安安、快快乐乐。

希望你早上醒来的时候，
窗外有光，屋子里很安静，
你的心是轻轻的，没有着急，也没有害怕。

希望你吃饭的时候胃口很好，
走路的时候脚步稳稳，
晚上睡觉的时候，能做一个又一个甜甜的梦。

我希望你能一直保持好奇心。
对世界充满问题，
对天空、星星、花草、书本和故事感兴趣。
当你问“为什么”的时候，
希望总有人愿意认真地听你说话。

我也希望你学会温柔。
温柔地对待朋友，
温柔地对待小动物，
也温柔地对待自己。

如果有一天你犯了错，
请不要太快责怪自己，
因为每一个认真成长的人，
都会在路上慢慢学会更好的方法。

愿你拥有勇气。
当你站在陌生的地方时，
当你第一次举手发言时，
当你遇到困难、感到害怕的时候，
希望你能轻轻地告诉自己：
“我可以试一试。”

就算没有一次成功，也没有关系。
失败不是坏事，
它只是告诉你，你正在努力。

我希望你学会分享快乐。
把开心的事情告诉别人，
把笑声送给身边的人，
因为快乐被分享的时候，
会变得更大、更亮。

如果有一天你感到难过，
我希望你知道——
难过并不丢脸，
哭泣也不是软弱。

愿你能找到一个安全的地方，
慢慢把心里的话说出来，
然后再一次抬起头，看见希望。

我还希望你能拥有梦想。
这个梦想也许很大，
也许很小，
也许现在还说不清楚。

没关系。
梦想会和你一起长大，
在时间里慢慢变得清楚。

最后，我想送你一个最最重要的祝福：

愿你被世界温柔对待，
也愿你成为一个温柔的人。

愿你的每一天，
都值得被记住，
都值得被珍惜。

亲爱的你，
请记住，
你是独一无二的，
你已经很棒了，
而你的未来，
一定会慢慢变得闪闪发光。

祝你健康、勇敢、幸福，
祝你永远带着笑容向前走。"""

ref_clone_1 = "examples/reference_audio/ref_clone_1.wav"
ref_clone_2 = "examples/reference_audio/ref_clone_2.wav"

conversations = [
    # Direct TTS (no reference)
    [
        processor.build_user_message(
            text=text_tts_1,
        )
    ],
    [
        processor.build_user_message(
            text=text_tts_2,
        )
    ],
    # Voice cloning (with reference)
    [
        processor.build_user_message(
            text=text_tts_1,
            reference=[ref_clone_1],
        )
    ],
    [
        processor.build_user_message(
            text=text_tts_1,
            reference=[ref_clone_2],
        )
    ],
    [
        processor.build_user_message(
            text=text_tts_2,
            reference=[ref_clone_1],
        )
    ],
    [
        processor.build_user_message(
            text=text_tts_2,
            reference=[ref_clone_2],
        )
    ],
]

batch = processor(conversations, mode="generation")
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
            "inference_root", f"sample{sample_idx}_seg{seg_idx}.wav"
        )
        torchaudio.save(
            out_path, audio.unsqueeze(0), processor.model_config.sampling_rate
        )
