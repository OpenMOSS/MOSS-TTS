import argparse
import functools
import importlib.util
import re
import time

import gradio as gr
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

# Disable the broken cuDNN SDPA backend
torch.backends.cuda.enable_cudnn_sdp(False)
# Keep these enabled as fallbacks
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

MODEL_PATH = "OpenMOSS-Team/MOSS-TTS"
DEFAULT_ATTN_IMPLEMENTATION = "auto"
DEFAULT_MAX_NEW_TOKENS = 4096
CONTINUATION_NOTICE = (
    "Continuation mode is active. Make sure the reference audio transcript is prepended to the input text."
)

MODE_CLONE = "Clone"
MODE_CONTINUE = "Continuation"
MODE_CONTINUE_CLONE = "Continuation + Clone"
ZH_TOKENS_PER_CHAR = 3.098411951313033
EN_TOKENS_PER_CHAR = 0.8673376262755219


@functools.lru_cache(maxsize=1)
def load_backend(model_path: str, device_str: str, attn_implementation: str):
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
    )
    if hasattr(processor, "audio_tokenizer"):
        processor.audio_tokenizer = processor.audio_tokenizer.to(device)

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


def detect_text_language(text: str) -> str:
    zh_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    en_chars = len(re.findall(r"[A-Za-z]", text))
    if zh_chars == 0 and en_chars == 0:
        return "en"
    return "zh" if zh_chars >= en_chars else "en"


def supports_duration_control(mode_with_reference: str) -> bool:
    return mode_with_reference not in {MODE_CONTINUE, MODE_CONTINUE_CLONE}


def estimate_duration_tokens(text: str) -> tuple[str, int, int, int]:
    normalized = text or ""
    effective_len = max(len(normalized), 1)
    language = detect_text_language(normalized)
    factor = ZH_TOKENS_PER_CHAR if language == "zh" else EN_TOKENS_PER_CHAR
    default_tokens = max(1, int(effective_len * factor))
    min_tokens = max(1, int(default_tokens * 0.5))
    max_tokens = max(min_tokens, int(default_tokens * 1.5))
    return language, default_tokens, min_tokens, max_tokens


def update_duration_controls(
    enabled: bool,
    text: str,
    current_tokens: float | int | None,
    mode_with_reference: str,
):
    if not supports_duration_control(mode_with_reference):
        return (
            gr.update(visible=False),
            "Duration control is disabled for Continuation modes.",
            gr.update(value=False, interactive=False),
        )

    checkbox_update = gr.update(interactive=True)
    if not enabled:
        return gr.update(visible=False), "Duration control is disabled.", checkbox_update

    language, default_tokens, min_tokens, max_tokens = estimate_duration_tokens(text)
    # Slider is initialized with value=1 as a placeholder; treat it as "unset"
    # so first-time estimation uses the computed default instead of clamping to min.
    if current_tokens is None or int(current_tokens) == 1:
        slider_value = default_tokens
    else:
        slider_value = int(current_tokens)
        slider_value = max(min_tokens, min(max_tokens, slider_value))

    language_label = "Chinese" if language == "zh" else "English"
    hint = (
        f"Duration control enabled | detected language: {language_label} | "
        f"default={default_tokens}, range=[{min_tokens}, {max_tokens}]"
    )
    return (
        gr.update(
            visible=True,
            minimum=min_tokens,
            maximum=max_tokens,
            value=slider_value,
            step=1,
        ),
        hint,
        checkbox_update,
    )


def build_conversation(
    text: str,
    reference_audio: str | None,
    mode_with_reference: str,
    expected_tokens: int | None,
    processor,
):
    text = (text or "").strip()
    if not text:
        raise ValueError("Please enter text to synthesize.")

    user_kwargs = {"text": text}
    if expected_tokens is not None:
        user_kwargs["tokens"] = int(expected_tokens)

    if not reference_audio:
        conversations = [[processor.build_user_message(**user_kwargs)]]
        return conversations, "generation", "Direct Generation"

    if mode_with_reference == MODE_CLONE:
        clone_kwargs = dict(user_kwargs)
        clone_kwargs["reference"] = [reference_audio]
        conversations = [[processor.build_user_message(**clone_kwargs)]]
        return conversations, "generation", MODE_CLONE

    if mode_with_reference == MODE_CONTINUE:
        conversations = [
            [
                processor.build_user_message(**user_kwargs),
                processor.build_assistant_message(audio_codes_list=[reference_audio]),
            ]
        ]
        return conversations, "continuation", MODE_CONTINUE

    continue_clone_kwargs = dict(user_kwargs)
    continue_clone_kwargs["reference"] = [reference_audio]
    conversations = [
        [
            processor.build_user_message(**continue_clone_kwargs),
            processor.build_assistant_message(audio_codes_list=[reference_audio]),
        ]
    ]
    return conversations, "continuation", MODE_CONTINUE_CLONE


def render_mode_hint(reference_audio: str | None, mode_with_reference: str):
    if not reference_audio:
        return "Current mode: **Direct Generation** (no reference audio uploaded)"
    if mode_with_reference == MODE_CLONE:
        return "Current mode: **Clone** (speaker timbre will be cloned from the reference audio)"
    return f"Current mode: **{mode_with_reference}**  \n> {CONTINUATION_NOTICE}"


def run_inference(
    text: str,
    reference_audio: str | None,
    mode_with_reference: str,
    duration_control_enabled: bool,
    duration_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    model_path: str,
    device: str,
    attn_implementation: str,
    max_new_tokens: int,
):
    started_at = time.monotonic()
    model, processor, torch_device, sample_rate = load_backend(
        model_path=model_path,
        device_str=device,
        attn_implementation=attn_implementation,
    )
    duration_enabled = bool(duration_control_enabled and supports_duration_control(mode_with_reference))
    expected_tokens = int(duration_tokens) if duration_enabled else None
    conversations, mode, mode_name = build_conversation(
        text=text,
        reference_audio=reference_audio,
        mode_with_reference=mode_with_reference,
        expected_tokens=expected_tokens,
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

    elapsed = time.monotonic() - started_at
    status = (
        f"Done | mode: {mode_name} | elapsed: {elapsed:.2f}s | "
        f"max_new_tokens={int(max_new_tokens)}, "
        f"expected_tokens={expected_tokens if expected_tokens is not None else 'off'}, "
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
    #output_audio {
      padding-bottom: 12px;
      margin-bottom: 8px;
      overflow: hidden !important;
    }
    #output_audio > .wrap {
      overflow: hidden !important;
    }
    #output_audio audio {
      margin-bottom: 6px;
    }
    #run-btn {
      background: var(--accent);
      border: none;
    }
    """

    with gr.Blocks(title="MOSS-TTS Demo", css=custom_css) as demo:
        gr.Markdown(
            """
            <div class="app-card">
              <div class="app-title">MOSS-TTS</div>
              <div class="app-subtitle">Minimal UI: Direct Generation, Clone, Continuation, Continuation + Clone</div>
            </div>
            """
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                text = gr.Textbox(
                    label="Text",
                    lines=9,
                    placeholder="Enter text to synthesize. In continuation modes, prepend the reference audio transcript.",
                )
                reference_audio = gr.Audio(
                    label="Reference Audio (Optional)",
                    type="filepath",
                )
                mode_with_reference = gr.Radio(
                    choices=[MODE_CLONE, MODE_CONTINUE, MODE_CONTINUE_CLONE],
                    value=MODE_CLONE,
                    label="Mode with Reference Audio",
                    info="If no reference audio is uploaded, Direct Generation will be used automatically.",
                )
                mode_hint = gr.Markdown(render_mode_hint(None, MODE_CLONE))
                duration_control_enabled = gr.Checkbox(
                    value=False,
                    label="Enable Duration Control (Expected Audio Tokens)",
                )
                duration_tokens = gr.Slider(
                    minimum=1,
                    maximum=1,
                    step=1,
                    value=1,
                    label="expected_tokens",
                    visible=False,
                )
                duration_hint = gr.Markdown("Duration control is disabled.")

                with gr.Accordion("Sampling Parameters (Audio)", open=True):
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=3.0,
                        step=0.05,
                        value=1.7,
                        label="temperature",
                    )
                    top_p = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        step=0.01,
                        value=0.8,
                        label="top_p",
                    )
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=200,
                        step=1,
                        value=25,
                        label="top_k",
                    )
                    repetition_penalty = gr.Slider(
                        minimum=0.8,
                        maximum=2.0,
                        step=0.05,
                        value=1.0,
                        label="repetition_penalty",
                    )
                    max_new_tokens = gr.Slider(
                        minimum=256,
                        maximum=8192,
                        step=128,
                        value=DEFAULT_MAX_NEW_TOKENS,
                        label="max_new_tokens",
                    )

                run_btn = gr.Button("Generate Speech", variant="primary", elem_id="run-btn")

            with gr.Column(scale=2):
                output_audio = gr.Audio(label="Output Audio", type="numpy", elem_id="output_audio")
                status = gr.Textbox(label="Status", lines=4, interactive=False)

        reference_audio.change(
            fn=render_mode_hint,
            inputs=[reference_audio, mode_with_reference],
            outputs=[mode_hint],
        )
        mode_with_reference.change(
            fn=render_mode_hint,
            inputs=[reference_audio, mode_with_reference],
            outputs=[mode_hint],
        )
        duration_control_enabled.change(
            fn=update_duration_controls,
            inputs=[duration_control_enabled, text, duration_tokens, mode_with_reference],
            outputs=[duration_tokens, duration_hint, duration_control_enabled],
        )
        text.change(
            fn=update_duration_controls,
            inputs=[duration_control_enabled, text, duration_tokens, mode_with_reference],
            outputs=[duration_tokens, duration_hint, duration_control_enabled],
        )
        mode_with_reference.change(
            fn=update_duration_controls,
            inputs=[duration_control_enabled, text, duration_tokens, mode_with_reference],
            outputs=[duration_tokens, duration_hint, duration_control_enabled],
        )

        run_btn.click(
            fn=lambda text, reference_audio, mode_with_reference, duration_control_enabled, duration_tokens, temperature, top_p, top_k, repetition_penalty, max_new_tokens: run_inference(
                text=text,
                reference_audio=reference_audio,
                mode_with_reference=mode_with_reference,
                duration_control_enabled=duration_control_enabled,
                duration_tokens=duration_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                model_path=args.model_path,
                device=args.device,
                attn_implementation=args.attn_implementation,
                max_new_tokens=max_new_tokens,
            ),
            inputs=[
                text,
                reference_audio,
                mode_with_reference,
                duration_control_enabled,
                duration_tokens,
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
    parser = argparse.ArgumentParser(description="MossTTS Gradio Demo")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--attn_implementation", type=str, default=DEFAULT_ATTN_IMPLEMENTATION)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
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

    # Preload model/processor at startup to avoid first-request cold start latency.
    preload_started_at = time.monotonic()
    print(
        f"[Startup] Preloading backend: model={args.model_path}, device={args.device}, attn={args.attn_implementation}",
        flush=True,
    )
    load_backend(
        model_path=args.model_path,
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
