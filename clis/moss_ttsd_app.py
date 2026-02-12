import argparse
import functools
import importlib.util
import re
import time
from pathlib import Path
from typing import Optional

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
MIN_SPEAKERS = 1
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


def _resample_wav(wav: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if int(orig_sr) == int(target_sr):
        return wav
    new_num_samples = int(round(wav.shape[-1] * float(target_sr) / float(orig_sr)))
    if new_num_samples <= 0:
        raise ValueError(f"Invalid resample length from {orig_sr}Hz to {target_sr}Hz.")
    return torch.nn.functional.interpolate(
        wav.unsqueeze(0),
        size=new_num_samples,
        mode="linear",
        align_corners=False,
    ).squeeze(0)


def _load_audio(audio_path: str) -> tuple[torch.Tensor, int]:
    path = Path(audio_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Reference audio not found: {path}")

    wav, sr = torchaudio.load(str(path))
    if wav.numel() == 0:
        raise ValueError(f"Reference audio is empty: {path}")

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    return wav, int(sr)


def normalize_text(text: str) -> str:
    text = re.sub(r"\[(\d+)\]", r"[S\1]", text)
    remove_chars = "【】《》（）『』「」" '"-_“”～~‘’'

    segments = re.split(r"(?=\[S\d+\])", text.replace("\n", " "))
    processed_parts = []
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue

        matched = re.match(r"^(\[S\d+\])\s*(.*)", seg)
        tag, content = matched.groups() if matched else ("", seg)

        content = re.sub(f"[{re.escape(remove_chars)}]", "", content)
        content = re.sub(r"哈{2,}", "[笑]", content)
        content = re.sub(r"\b(ha(\s*ha)+)\b", "[laugh]", content, flags=re.IGNORECASE)

        content = content.replace("——", "，")
        content = content.replace("……", "，")
        content = content.replace("...", "，")
        content = content.replace("⸺", "，")
        content = content.replace("―", "，")
        content = content.replace("—", "，")
        content = content.replace("…", "，")

        internal_punct_map = str.maketrans(
            {"；": "，", ";": ",", "：": "，", ":": ",", "、": "，"}
        )
        content = content.translate(internal_punct_map)
        content = content.strip()
        content = re.sub(r"([，。？！,.?!])[，。？！,.?!]+", r"\1", content)

        if len(content) > 1:
            last_ch = "。" if content[-1] == "，" else ("." if content[-1] == "," else content[-1])
            body = content[:-1].replace("。", "，")
            content = body + last_ch

        processed_parts.append({"tag": tag, "content": content})

    if not processed_parts:
        return ""

    merged_lines = []
    current_tag = processed_parts[0]["tag"]
    current_content = [processed_parts[0]["content"]]
    for part in processed_parts[1:]:
        if part["tag"] == current_tag and current_tag:
            current_content.append(part["content"])
        else:
            merged_lines.append(f"{current_tag}{''.join(current_content)}".strip())
            current_tag = part["tag"]
            current_content = [part["content"]]
    merged_lines.append(f"{current_tag}{''.join(current_content)}".strip())

    return "".join(merged_lines).replace("‘", "'").replace("’", "'")


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


def _merge_consecutive_speaker_tags(text: str) -> str:
    segments = re.split(r"(?=\[S\d+\])", text)
    if not segments:
        return text

    merged_parts = []
    current_tag = None
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        matched = re.match(r"^(\[S\d+\])\s*(.*)", seg, re.DOTALL)
        if not matched:
            merged_parts.append(seg)
            continue
        tag, content = matched.groups()
        if tag == current_tag:
            merged_parts.append(content)
        else:
            current_tag = tag
            merged_parts.append(f"{tag}{content}")
    return "".join(merged_parts)


def _normalize_prompt_text(prompt_text: str, speaker_id: int) -> str:
    text = (prompt_text or "").strip()
    if not text:
        raise ValueError(f"S{speaker_id} prompt text is empty.")

    expected_tag = f"[S{speaker_id}]"
    if not text.lstrip().startswith(expected_tag):
        text = f"{expected_tag} {text}"
    return text


def _build_prefixed_text(
    dialogue_text: str,
    prompt_text_map: dict[int, str],
    cloned_speakers: list[int],
) -> str:
    prompt_prefix = "".join([prompt_text_map[speaker_id] for speaker_id in cloned_speakers])
    return _merge_consecutive_speaker_tags(prompt_prefix + dialogue_text)


def _encode_reference_audio_codes(
    processor,
    clone_wavs: list[torch.Tensor],
    cloned_speakers: list[int],
    speaker_count: int,
    sample_rate: int,
) -> list[Optional[torch.Tensor]]:
    encoded_list = processor.encode_audios_from_wav(clone_wavs, sampling_rate=sample_rate)
    reference_audio_codes: list[Optional[torch.Tensor]] = [None for _ in range(speaker_count)]
    for speaker_id, audio_codes in zip(cloned_speakers, encoded_list):
        reference_audio_codes[speaker_id - 1] = audio_codes
    return reference_audio_codes


def build_conversation(
    dialogue_text: str,
    reference_audio_codes: list[Optional[torch.Tensor]],
    prompt_audio: torch.Tensor | None,
    processor,
):
    if prompt_audio is None:
        return [[processor.build_user_message(text=dialogue_text)]], "generation", "Generation"

    user_message = processor.build_user_message(
        text=dialogue_text,
        reference=reference_audio_codes,
    )
    return (
        [
            [
                user_message,
                processor.build_assistant_message(audio_codes_list=[prompt_audio]),
            ],
        ],
        "continuation",
        "voice_clone_and_continuation",
    )


def run_inference(speaker_count: int, *all_inputs):
    speaker_count = int(speaker_count)
    speaker_count = max(MIN_SPEAKERS, min(MAX_SPEAKERS, speaker_count))

    reference_audio_values = all_inputs[:MAX_SPEAKERS]
    prompt_text_values = all_inputs[MAX_SPEAKERS : 2 * MAX_SPEAKERS]
    dialogue_text = all_inputs[2 * MAX_SPEAKERS]
    text_normalize, sample_rate_normalize, temperature, top_p, top_k, repetition_penalty, max_new_tokens, model_path, codec_path, device, attn_implementation = all_inputs[
        2 * MAX_SPEAKERS + 1 :
    ]

    started_at = time.monotonic()
    model, processor, torch_device, sample_rate = load_backend(
        model_path=str(model_path),
        codec_path=str(codec_path),
        device_str=str(device),
        attn_implementation=str(attn_implementation),
    )

    text_normalize = bool(text_normalize)
    sample_rate_normalize = bool(sample_rate_normalize)

    normalized_dialogue = str(dialogue_text or "").strip()
    if text_normalize:
        normalized_dialogue = normalize_text(normalized_dialogue)
    normalized_dialogue = _validate_dialogue_text(normalized_dialogue, speaker_count)

    cloned_speakers: list[int] = []
    loaded_clone_wavs: list[tuple[torch.Tensor, int]] = []
    prompt_text_map: dict[int, str] = {}
    for idx in range(speaker_count):
        ref_audio = reference_audio_values[idx]
        prompt_text = str(prompt_text_values[idx] or "").strip()

        has_reference = bool(ref_audio)
        has_prompt_text = bool(prompt_text)
        if has_reference != has_prompt_text:
            raise ValueError(
                f"S{idx + 1} must provide both reference audio and prompt text together."
            )

        if has_reference:
            speaker_id = idx + 1
            ref_audio_path = str(ref_audio)
            cloned_speakers.append(speaker_id)
            loaded_clone_wavs.append(_load_audio(ref_audio_path))
            prompt_text_map[speaker_id] = _normalize_prompt_text(prompt_text, speaker_id)

    prompt_audio: Optional[torch.Tensor] = None
    reference_audio_codes: list[Optional[torch.Tensor]] = []
    conversation_text = normalized_dialogue
    if cloned_speakers:
        conversation_text = _build_prefixed_text(
            dialogue_text=normalized_dialogue,
            prompt_text_map=prompt_text_map,
            cloned_speakers=cloned_speakers,
        )
        if text_normalize:
            conversation_text = normalize_text(conversation_text)
        conversation_text = _validate_dialogue_text(conversation_text, speaker_count)

        if sample_rate_normalize:
            min_sr = min(sr for _, sr in loaded_clone_wavs)
        else:
            min_sr = None

        clone_wavs: list[torch.Tensor] = []
        for wav, orig_sr in loaded_clone_wavs:
            processed_wav = wav
            current_sr = int(orig_sr)
            if min_sr is not None:
                processed_wav = _resample_wav(processed_wav, current_sr, int(min_sr))
                current_sr = int(min_sr)
            processed_wav = _resample_wav(processed_wav, current_sr, sample_rate)
            clone_wavs.append(processed_wav)

        reference_audio_codes = _encode_reference_audio_codes(
            processor=processor,
            clone_wavs=clone_wavs,
            cloned_speakers=cloned_speakers,
            speaker_count=speaker_count,
            sample_rate=sample_rate,
        )
        concat_prompt_wav = torch.cat(clone_wavs, dim=-1)
        prompt_audio = processor.encode_audios_from_wav([concat_prompt_wav], sampling_rate=sample_rate)[0]

    conversations, mode, mode_name = build_conversation(
        dialogue_text=conversation_text,
        reference_audio_codes=reference_audio_codes,
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
        f"text_normalize={text_normalize}, sample_rate_normalize={sample_rate_normalize} | "
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
        speaker_prompts = []

        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                speaker_count = gr.Slider(
                    minimum=MIN_SPEAKERS,
                    maximum=MAX_SPEAKERS,
                    step=1,
                    value=2,
                    label="Speaker Count",
                    info="Default 2 speakers. Minimum 1, maximum 5.",
                )

                gr.Markdown("### Voice Cloning (Optional, placed first)")
                gr.Markdown(
                    "If you provide reference audio for a speaker, you must also provide that speaker's prompt text. "
                    "Prompt text may omit [Sx]; the app will auto-prepend it."
                )

                for idx in range(1, MAX_SPEAKERS + 1):
                    with gr.Group(visible=idx <= 2) as panel:
                        speaker_ref = gr.Audio(
                            label=f"S{idx} Reference Audio (Optional)",
                            type="filepath",
                        )
                        speaker_prompt = gr.Textbox(
                            label=f"S{idx} Prompt Text (Required with reference audio)",
                            lines=2,
                            placeholder=f"Example: [S{idx}] This is a prompt line for S{idx}.",
                        )
                    speaker_panels.append(panel)
                    speaker_refs.append(speaker_ref)
                    speaker_prompts.append(speaker_prompt)

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
                    "Without any reference audio, the model runs in generation mode. "
                    "Once any reference audio is provided, the model switches to voice-clone continuation mode."
                )

                with gr.Accordion("Sampling Parameters (Audio)", open=True):
                    gr.Markdown(
                        "- `text_normalize`: Normalize input text (**recommended to always enable**).\n"
                        "- `sample_rate_normalize`: Resample prompt audios to the lowest sample rate before encoding "
                        "(**recommended when using 2 or more speakers**)."
                    )
                    text_normalize = gr.Checkbox(
                        value=True,
                        label="text_normalize",
                    )
                    sample_rate_normalize = gr.Checkbox(
                        value=False,
                        label="sample_rate_normalize",
                    )
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
                *speaker_prompts,
                dialogue_text,
                text_normalize,
                sample_rate_normalize,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                max_new_tokens,
            ],
            outputs=[output_audio, status],
        )
    return demo


def main() -> None:
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
    demo.queue(default_concurrency_limit=2).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()