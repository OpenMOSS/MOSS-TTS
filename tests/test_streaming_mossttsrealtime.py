from moss_tts_realtime.mossttsrealtime.streaming_mossttsrealtime import (
    MossTTSRealtimeStreamingSession,
)


def make_session(text_cache, text_buffer_size=32, min_text_chunk_chars=8):
    # The text-segmentation logic only reads the buffer size and the chunk
    # threshold; the audio inferencer and processor are irrelevant here.
    session = object.__new__(MossTTSRealtimeStreamingSession)
    session._text_cache = text_cache
    session.text_buffer_size = text_buffer_size
    session.min_text_chunk_chars = min_text_chunk_chars
    return session


def test_buffer_over_limit_is_flushed_without_spaces_or_punctuation():
    text = "这是一段没有任何标点也没有空格的中文文本用来验证缓冲区上限到达后能够强制切段继续朗读"
    assert len(text) > 32
    session = make_session(text)

    segments = session._extract_text_segments(force=False)

    assert segments == [text[:32]]
    assert session._text_cache == text[32:]


def test_buffer_over_limit_still_prefers_whitespace_cut():
    # The last whitespace wins over the size limit: the flush cuts at the
    # space (index 11), not mid-word at text_buffer_size.
    session = make_session("a" * 10 + " " + "b" * 30)

    segments = session._extract_text_segments(force=False)

    assert segments == ["a" * 10 + " "]
    assert session._text_cache == "b" * 30


def test_punctuation_cut_is_unchanged():
    session = make_session("一二三四五六七八九十。后续的文本内容继续")

    segments = session._extract_text_segments(force=False)

    assert segments == ["一二三四五六七八九十。"]
    assert session._text_cache == "后续的文本内容继续"


def test_short_buffer_without_split_point_is_kept():
    text = "不足一段的中文文本"
    session = make_session(text)

    segments = session._extract_text_segments(force=False)

    assert segments == []
    assert session._text_cache == text
