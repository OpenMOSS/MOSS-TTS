import threading
import uuid
import wave
import time
from typing import Iterator

import requests

BASE_URL = "http://127.0.0.1:8083"

OUT_PCM_PATH = "out_streaming_split.pcm"
OUT_WAV_PATH = "out_streaming_split.wav"

FULL_TEXT = """Welcome to the world of MOSS TTS Realtime. Experience how text transforms into smooth, human-like speech in real time. MOSS TTS Realtime is a context-aware multi-turn streaming TTS, a speech generation foundation model designed for voice agents."""
PROMPT_AUDIO = "./audio/prompt_audio.mp3"

DELTA_CHUNK_CHARS = 50

DELTA_DELAY_S = 0.1

START_TIMEOUT = 60
PUSH_TIMEOUT = 60
CLOSE_TIMEOUT = 30
AUDIO_TIMEOUT = 600


class AudioReaderState:
    def __init__(self) -> None:
        self.error: Exception | None = None
        self.first_chunk_time: float | None = None
        self.total_bytes: int = 0
        self.sample_rate: int = 24000


def iter_text_deltas(text: str, chunk_chars: int = 20) -> Iterator[str]:
    text = text or ""
    step = max(1, int(chunk_chars))
    for idx in range(0, len(text), step):
        yield text[idx : idx + step]


def audio_reader(
    session_id: str,
    out_pcm_path: str,
    out_wav_path: str,
    state: AudioReaderState,
) -> None:
    url = f"{BASE_URL}/tts/session/{session_id}/audio"
    print(f"[audio] connect -> {url}")

    try:
        with requests.get(url, stream=True, timeout=AUDIO_TIMEOUT) as resp:
            resp.raise_for_status()

            sr = int(resp.headers.get("X-Audio-Sample-Rate", "24000"))
            ch = int(resp.headers.get("X-Audio-Channels", "1"))
            codec = resp.headers.get("X-Audio-Codec", "pcm_s16le")

            print(f"[audio] headers: sr={sr}, ch={ch}, codec={codec}")

            if ch != 1:
                raise RuntimeError(f"Expected mono audio, got channels={ch}")
            if codec.lower() != "pcm_s16le":
                raise RuntimeError(f"Expected pcm_s16le, got codec={codec}")

            with open(out_pcm_path, "wb") as f_pcm, wave.open(out_wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # int16
                wf.setframerate(sr)

                chunk_count = 0
                byte_count = 0
                for chunk in resp.iter_content(chunk_size=4096):
                    if not chunk:
                        continue
                    if chunk_count == 0:
                        state.first_chunk_time = time.monotonic()
                    chunk_count += 1
                    byte_count += len(chunk)
                    f_pcm.write(chunk)
                    wf.writeframesraw(chunk)

                state.total_bytes = byte_count
                state.sample_rate = sr

            print(f"[audio] done, chunks={chunk_count}, bytes={byte_count}")
            print(f"[audio] saved pcm -> {out_pcm_path}")
            print(f"[audio] saved wav -> {out_wav_path}")
    except Exception as exc:
        state.error = exc
        raise


def start_session(
    session_id: str,
    prompt_audio: str | None = None,
    assistant_text: str | None = None,
    user_text: str | None = None,
    user_audio: str | None = None,
) -> dict:
    payload = {
        "session_id": session_id,
        "assistant_text": assistant_text,
        "user_text": user_text,
        "prompt_audio": prompt_audio,
        "user_audio": user_audio,
        "new_turn": True,
    }

    print(f"[start] payload={payload}")
    resp = requests.post(
        f"{BASE_URL}/tts/session/start",
        json=payload,
        timeout=START_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    print(f"[start] resp={data}")
    return data


def push_text(session_id: str, text: str, is_final: bool) -> dict:
    payload = {
        "session_id": session_id,
        "text": text,
        "is_final": is_final,
    }
    resp = requests.post(
        f"{BASE_URL}/tts/session/push",
        json=payload,
        timeout=PUSH_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def close_session(session_id: str) -> dict:
    payload = {"session_id": session_id}
    resp = requests.post(
        f"{BASE_URL}/tts/session/close",
        json=payload,
        timeout=CLOSE_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    session_id = str(uuid.uuid4())
    print(f"[main] SESSION_ID={session_id}")
    audio_state = AudioReaderState()

    deltas = list(iter_text_deltas(FULL_TEXT, chunk_chars=DELTA_CHUNK_CHARS))
    if not deltas:
        raise RuntimeError("No text to send. FULL_TEXT is empty.")

    first_delta = deltas[0]
    remaining_deltas = deltas[1:]

    print(f"[start] first_delta={first_delta!r}")
    start_session(
        session_id=session_id,
        prompt_audio=PROMPT_AUDIO,
        assistant_text=first_delta,
        user_text=None,
        user_audio=None,
    )

    audio_thread = threading.Thread(
        target=audio_reader,
        args=(session_id, OUT_PCM_PATH, OUT_WAV_PATH, audio_state),
        daemon=True,
    )
    audio_thread.start()

    for i, delta in enumerate(remaining_deltas):
        is_final = i == len(remaining_deltas) - 1
        idx = i + 2
        print(f"[push #{idx}] is_final={is_final}, text={delta!r}")
        data = push_text(session_id=session_id, text=delta, is_final=is_final)
        print(f"[push #{idx}] resp={data}")
        if DELTA_DELAY_S > 0 and not is_final:
            time.sleep(DELTA_DELAY_S)

    if not remaining_deltas:
        push_text(session_id=session_id, text="", is_final=True)

    audio_thread.join()
    if audio_state.error is not None:
        raise RuntimeError(f"audio reader failed: {audio_state.error}") from audio_state.error

    try:
        data = close_session(session_id)
        print(f"[close] resp={data}")
    except Exception as e:
        print(f"[close] warning: {e}")

    print("[main] done")


if __name__ == "__main__":
    main()

