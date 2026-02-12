import argparse
import functools
import importlib.util
import re
import time
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torchaudio
from transformers import AutoModel, AutoProcessor

# Disable the broken cuDNN SDPA backend
torch.backends.cuda.enable_cudnn_sdp(False)
# Keep these enabled as fallbacks
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

MODEL_PATH = "OpenMOSS-Team/MOSS-TTSD-v1.0"
CODEC_MODEL_PATH = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
DEFAULT_ATTN_IMPLEMENTATION = "auto"
DEFAULT_MAX_NEW_TOKENS = 2000
MIN_SPEAKERS = 2
MAX_SPEAKERS = 5


def resolve_attn_implementation(requested: str, device: torch.device, dtype: torch.dtype) -> str | None:
    requested_norm = (requested or "").strip().lower()

    if requested_norm in {"none"}:
        return None

    if requested_norm not in {"", "auto"}:
        return requested

    # Prefer FlashAttention 2 when package + device conditions are met.
    if (
        device.type == "cuda"
        and importlib.util.find_spec("flash_attn") is not None
        and dtype in {torch.float16, torch.bfloat16}
    ):
        major, _ = torch.cuda.get_device_capability(device)
        if major >= 8:
            return "flash_attention_2"

    # CUDA fallback: use PyTorch SDPA kernels.
    if device.type == "cuda":
        return "sdpa"

    # CPU fallback.
    return "eager"


@functools.lru_cache(maxsize=1)
def load_backend(model_path: str, codec_path: str, device_str: str, attn_implementation: str):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    resolved_attn_implementation = resolve_attn_implementation(
        requested=attn_implementation,
        device=device,
        dtype=dtype,
    )

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        codec_path=codec_path,
    )
    if hasattr(processor, "audio_tokenizer"):
        processor.audio_tokenizer = processor.audio_tokenizer.to(device)
        processor.audio_tokenizer.eval()

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }
    if resolved_attn_implementation:
        model_kwargs["attn_implementation"] = resolved_attn_implementation

    model = AutoModel.from_pretrained(model_path, **model_kwargs).to(device)
    model.eval()

    sample_rate = int(getattr(processor.model_config, "sampling_rate", 24000))
    return model, processor, device, sample_rate


def _load_audio(audio_path: str, target_sample_rate: int) -> torch.Tensor:
    path = Path(audio_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Reference audio not found: {path}")

    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sample_rate:
        wav = torchaudio.functional.resample(wav, sr, target_sample_rate)
    return wav


def _validate_dialogue_text(dialogue_text: str, speaker_count: int) -> str:
    text = (dialogue_text or "").strip()
    if not text:
        raise ValueError("Please enter dialogue text.")

    tags = re.findall(r"\[S(\d+)\]", text)
    if not tags:
        raise ValueError("Dialogue must include speaker tags like [S1], [S2], ...")

    max_tag = max(int(t) for t in tags)
    if max_tag > speaker_count:
        raise ValueError(
            f"Dialogue contains [S{max_tag}], but speaker count is set to {speaker_count}."
        )
    return text


def update_speaker_panels(speaker_count: int):
    count = int(speaker_count)
    count = max(MIN_SPEAKERS, min(MAX_SPEAKERS, count))
    return [gr.update(visible=(idx < count)) for idx in range(MAX_SPEAKERS)]


def build_conversation(
    dialogue_text: str,
    reference_list_with_none: list[str | None],
    prompt_audio: torch.Tensor | None,
    processor,
):
    user_message = processor.build_user_message(
        text=dialogue_text,
        reference=reference_list_with_none,
    )

    if prompt_audio is None:
        return [[user_message]], "generation", "Generation (No Clone)"

    return (
        [
            [
                user_message,
                processor.build_assistant_message(audio_codes_list=[prompt_audio]),
            ],
        ],
        "continuation",
        "Continuation + Clone",
    )


def run_inference(speaker_count: int, *all_inputs):
    speaker_count = int(speaker_count)
    speaker_count = max(MIN_SPEAKERS, min(MAX_SPEAKERS, speaker_count))

    reference_audio_values = all_inputs[:MAX_SPEAKERS]
    dialogue_text = all_inputs[MAX_SPEAKERS]
    temperature, top_p, top_k, repetition_penalty, max_new_tokens, model_path, codec_path, device, attn_implementation = all_inputs[
        MAX_SPEAKERS + 1 :
    ]

    started_at = time.monotonic()
    model, processor, torch_device, sample_rate = load_backend(
        model_path=str(model_path),
        codec_path=str(codec_path),
        device_str=str(device),
        attn_implementation=str(attn_implementation),
    )

    normalized_dialogue = _validate_dialogue_text(str(dialogue_text or ""), speaker_count)

    # Build reference list with explicit None placeholders, e.g. [wav1, None, wav3]
    reference_list_with_none: list[str | None] = []
    cloned_speakers: list[int] = []
    clone_wavs: list[torch.Tensor] = []
    for idx in range(speaker_count):
        ref_audio = reference_audio_values[idx]
        if ref_audio:
            ref_audio_path = str(ref_audio)
            reference_list_with_none.append(ref_audio_path)
            cloned_speakers.append(idx + 1)
            clone_wavs.append(_load_audio(ref_audio_path, sample_rate))
        else:
            reference_list_with_none.append(None)

    prompt_audio = None
    if clone_wavs:
        concat_prompt_wav = torch.cat(clone_wavs, dim=-1)
        prompt_audio = processor.encode_audios_from_wav([concat_prompt_wav], sampling_rate=sample_rate)[0]

    conversations, mode, mode_name = build_conversation(
        dialogue_text=normalized_dialogue,
        reference_list_with_none=reference_list_with_none,
        prompt_audio=prompt_audio,
        processor=processor,
    )

    batch = processor(conversations, mode=mode)
    input_ids = batch["input_ids"].to(torch_device)
    attention_mask = batch["attention_mask"].to(torch_device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(max_new_tokens),
            audio_temperature=float(temperature),
            audio_top_p=float(top_p),
            audio_top_k=int(top_k),
            audio_repetition_penalty=float(repetition_penalty),
        )

    messages = processor.decode(outputs)
    if not messages or messages[0] is None:
        raise RuntimeError("The model did not return a decodable audio result.")

    audio = messages[0].audio_codes_list[0]
    if isinstance(audio, torch.Tensor):
        audio_np = audio.detach().float().cpu().numpy()
    else:
        audio_np = np.asarray(audio, dtype=np.float32)

    if audio_np.ndim > 1:
        audio_np = audio_np.reshape(-1)
    audio_np = audio_np.astype(np.float32, copy=False)

    clone_summary = "none" if not cloned_speakers else ",".join([f"S{i}" for i in cloned_speakers])
    elapsed = time.monotonic() - started_at
    status = (
        f"Done | mode={mode_name} | speakers={speaker_count} | cloned={clone_summary} | elapsed={elapsed:.2f}s | "
        f"max_new_tokens={int(max_new_tokens)}, "
        f"audio_temperature={float(temperature):.2f}, audio_top_p={float(top_p):.2f}, "
        f"audio_top_k={int(top_k)}, audio_repetition_penalty={float(repetition_penalty):.2f}"
    )
    return (sample_rate, audio_np), status


def build_demo(args: argparse.Namespace):
    custom_css = """
    :root {
      --bg: #f6f7f8;
      --panel: #ffffff;
      --ink: #111418;
      --muted: #4d5562;
      --line: #e5e7eb;
      --accent: #0f766e;
    }
    .gradio-container {
      background: linear-gradient(180deg, #f7f8fa 0%, #f3f5f7 100%);
      color: var(--ink);
    }
    .app-card {
      border: 1px solid var(--line);
      border-radius: 16px;
      background: var(--panel);
      padding: 14px;
    }
    .app-title {
      font-size: 22px;
      font-weight: 700;
      margin-bottom: 6px;
      letter-spacing: 0.2px;
    }
    .app-subtitle {
      color: var(--muted);
      font-size: 14px;
      margin-bottom: 8px;
    }
    #output_panel {
      overflow: hidden !important;
    }
    #output_audio {
      padding-bottom: 24px;
      margin-bottom: 0;
      overflow: hidden !important;
    }
    #output_audio > .wrap,
    #output_audio .wrap,
    #output_audio .audio-container,
    #output_audio .block {
      overflow: hidden !important;
    }
    #output_audio .audio-container {
      padding-bottom: 10px;
      min-height: 96px;
    }
    #output_audio_spacer {
      height: 12px;
    }
    #output_status {
      margin-top: 0;
    }
    #run-btn {
      background: var(--accent);
      border: none;
    }
    """

    with gr.Blocks(title="MOSS-TTSD Demo", css=custom_css) as demo:
        gr.Markdown(
            """
            <div class="app-card">
              <div class="app-title">MOSS-TTSD</div>
              <div class="app-subtitle">Multi-speaker dialogue synthesis with optional per-speaker voice cloning.</div>
            </div>
            """
        )

        speaker_panels: list[gr.Group] = []
        speaker_refs = []

        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                speaker_count = gr.Slider(
                    minimum=MIN_SPEAKERS,
                    maximum=MAX_SPEAKERS,
                    step=1,
                    value=2,
                    label="Speaker Count",
                    info="Default 2 speakers. Minimum 2, maximum 5.",
                )

                gr.Markdown("### Voice Cloning (Optional, placed first)")
                gr.Markdown(
                    "Upload reference audio only for speakers you want to clone. "
                    "For example with 3 speakers, you can set S1/S3 and leave S2 empty."
                )

                for idx in range(1, MAX_SPEAKERS + 1):
                    with gr.Group(visible=idx <= 2) as panel:
                        speaker_ref = gr.Audio(
                            label=f"S{idx} Reference Audio (Optional)",
                            type="filepath",
                        )
                    speaker_panels.append(panel)
                    speaker_refs.append(speaker_ref)

                gr.Markdown("### Multi-turn Dialogue")
                dialogue_text = gr.Textbox(
                    label="Dialogue Text",
                    lines=12,
                    placeholder=(
                        "Use explicit tags in a single box, e.g.\n"
                        "[S1] Hello.\n"
                        "[S2] Hi, how are you?\n"
                        "[S1] Great, let's continue."
                    ),
                )
                gr.Markdown(
                    "When any reference audio is used, the model runs in continuation+clone mode. "
                    "Please include the prompt transcript in this dialogue box when needed."
                )

                with gr.Accordion("Sampling Parameters (Audio)", open=True):
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=3.0,
                        step=0.05,
                        value=1.1,
                        label="temperature",
                    )
                    top_p = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        step=0.01,
                        value=0.9,
                        label="top_p",
                    )
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=200,
                        step=1,
                        value=50,
                        label="top_k",
                    )
                    repetition_penalty = gr.Slider(
                        minimum=0.8,
                        maximum=2.0,
                        step=0.05,
                        value=1.1,
                        label="repetition_penalty",
                    )
                    max_new_tokens = gr.Slider(
                        minimum=256,
                        maximum=8192,
                        step=128,
                        value=DEFAULT_MAX_NEW_TOKENS,
                        label="max_new_tokens",
                    )

                run_btn = gr.Button("Generate Dialogue Audio", variant="primary", elem_id="run-btn")

            with gr.Column(scale=2, elem_id="output_panel"):
                output_audio = gr.Audio(label="Output Audio", type="numpy", elem_id="output_audio")
                gr.HTML("", elem_id="output_audio_spacer")
                status = gr.Textbox(label="Status", lines=4, interactive=False, elem_id="output_status")

        speaker_count.change(
            fn=update_speaker_panels,
            inputs=[speaker_count],
            outputs=speaker_panels,
        )

        run_btn.click(
            fn=lambda speaker_count, *inputs: run_inference(
                speaker_count,
                *inputs,
                args.model_path,
                args.codec_path,
                args.device,
                args.attn_implementation,
            ),
            inputs=[
                speaker_count,
                *speaker_refs,
                dialogue_text,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                max_new_tokens,
            ],
            outputs=[output_audio, status],
        )
    return demo


def main():
    parser = argparse.ArgumentParser(description="MOSS-TTSD Gradio Demo")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--codec_path", type=str, default=CODEC_MODEL_PATH)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--attn_implementation", type=str, default=DEFAULT_ATTN_IMPLEMENTATION)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7863)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    runtime_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    runtime_dtype = torch.bfloat16 if runtime_device.type == "cuda" else torch.float32
    args.attn_implementation = resolve_attn_implementation(
        requested=args.attn_implementation,
        device=runtime_device,
        dtype=runtime_dtype,
    ) or "none"
    print(f"[INFO] Using attn_implementation={args.attn_implementation}", flush=True)

    preload_started_at = time.monotonic()
    print(
        f"[Startup] Preloading backend: model={args.model_path}, codec={args.codec_path}, "
        f"device={args.device}, attn={args.attn_implementation}",
        flush=True,
    )
    load_backend(
        model_path=args.model_path,
        codec_path=args.codec_path,
        device_str=args.device,
        attn_implementation=args.attn_implementation,
    )
    print(
        f"[Startup] Backend preload finished in {time.monotonic() - preload_started_at:.2f}s",
        flush=True,
    )

    demo = build_demo(args)
    demo.queue(max_size=16, default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
