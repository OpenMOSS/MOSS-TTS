import importlib.util
import torch
import torchaudio
import torch.nn.functional as F
from typing import Any
from transformers import AutoTokenizer
from mossttsrealtime.modeling_mossttsrealtime import MossTTSRealtime
from inferencer import MossTTSRealtimeInference
from transformers import AutoModel

MAX_CHANNELS = 16
CODEC_SAMPLE_RATE = 24000

def main(model_path, codec_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

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
    
    # Because the local transformer module uses StaticCache, which causes errors when using the FlashAttention implementation. If FlashAttention is required, replace StaticCache with DynamicCache.
    model = MossTTSRealtime.from_pretrained(model_path, attn_implementation=attn_implementation, torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    codec = AutoModel.from_pretrained(codec_path, trust_remote_code=True).eval()
    codec = codec.to(device)

    inferencer = MossTTSRealtimeInference(model, tokenizer, max_length=5000, codec=codec, codec_sample_rate=CODEC_SAMPLE_RATE, codec_encode_kwargs={"chunk_duration": 8})
    
    text = ["Welcome to the world of MOSS TTS Realtime. Experience how text transforms into smooth, human-like speech in real time.", "MOSS TTS Realtime is a context-aware multi-turn streaming TTS, a speech generation foundation model designed for voice agents."]
    reference_audio_path = ["./audio/prompt_audio.mp3", "./audio/prompt_audio1.mp3"]

    result = inferencer.generate(
        text=text,
        reference_audio_path=reference_audio_path,
        temperature=0.8,
        top_p = 0.6,
        top_k = 30,
        repetition_penalty = 1.1,
        repetition_window = 50,
        device = device,
    )

    for i, generated_tokens, in enumerate[Any](result):
        output = torch.tensor(generated_tokens).to(device)
        decode_result = codec.decode(output.permute(1, 0), chunk_duration=8)
        wav = decode_result["audio"][0].cpu().detach()

        if wav.ndim == 1:
            wav = wav.unsqueeze(0)

        torchaudio.save(f'{i}.wav', wav, CODEC_SAMPLE_RATE)


if __name__ == "__main__":
    model_path = "OpenMOSS-Team/MOSS-TTS-Realtime"
    codec_path = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
    main(model_path, codec_path) 
