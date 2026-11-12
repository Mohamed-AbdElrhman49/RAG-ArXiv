"""Groq generator.

اتصلح هنا bug من كود الـ notebook: `str(chunk.choices[0].delta.content)` كانت بتحوّل None لنص "None"
(آخر chunk في الـ stream بيبقى content=None)، فكانت كلمة None بتظهر في آخر كل إجابة.
"""
import re
import time
from typing import Iterator


class GenerationError(Exception):
    """فشل التوليد (رد فاضي أو خطأ من الـ API)."""


class RateLimited(GenerationError):
    """429 من Groq."""


OPEN, CLOSE = "<think>", "</think>"
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S)


def strip_think(text: str) -> str:
    text = _THINK_BLOCK.sub("", text)
    if OPEN in text:  # تفكير اتقطع قبل ما يقفل (max tokens خلص)
        text = text.split(OPEN)[0]
    return text.strip()


def _overlap(buf: str, tag: str) -> int:
    """أطول suffix في buf هو prefix من tag (tag ممكن يكون جاي مقسوم على chunkين)."""
    for k in range(min(len(tag) - 1, len(buf)), 0, -1):
        if buf.endswith(tag[:k]):
            return k
    return 0


class ThinkFilter:
    """بيشيل <think>...</think> من الـ stream حتى لو التاج اتقسم بين chunkين."""

    def __init__(self):
        self.buf, self.inside = "", False

    def feed(self, s: str) -> str:
        self.buf += s
        out = []
        while True:
            if self.inside:
                i = self.buf.find(CLOSE)
                if i == -1:
                    k = _overlap(self.buf, CLOSE)
                    self.buf = self.buf[len(self.buf) - k:] if k else ""
                    break
                self.buf, self.inside = self.buf[i + len(CLOSE):], False
            else:
                i = self.buf.find(OPEN)
                if i == -1:
                    k = _overlap(self.buf, OPEN)
                    cut = len(self.buf) - k
                    out.append(self.buf[:cut])
                    self.buf = self.buf[cut:]
                    break
                out.append(self.buf[:i])
                self.buf, self.inside = self.buf[i + len(OPEN):], True
        return "".join(out)

    def flush(self) -> str:
        rest = "" if self.inside else self.buf
        self.buf = ""
        return rest


def _content(chunk) -> str:
    if not getattr(chunk, "choices", None):
        return ""
    return getattr(chunk.choices[0].delta, "content", None) or ""


class GroqGenerator:
    def __init__(self, api_key: str, model: str, temperature: float = 0.3,
                 reasoning_effort: str | None = None, retries: int = 2):
        if not api_key:
            raise RuntimeError("GROQ_API_KEY مش متظبط. حطه في .env أو كـ environment variable")
        from groq import Groq

        self.client = Groq(api_key=api_key)
        self.model, self.temperature = model, temperature
        self.reasoning_effort, self.retries = reasoning_effort, retries

    def _kwargs(self, messages, max_tokens, stream):
        kw = dict(model=self.model, messages=messages, temperature=self.temperature,
                  max_completion_tokens=max_tokens, stream=stream)
        if self.reasoning_effort:
            kw["reasoning_effort"] = self.reasoning_effort
        return kw

    @staticmethod
    def _wrap(e: Exception) -> GenerationError:
        if type(e).__name__ == "RateLimitError" or "429" in str(e):
            return RateLimited(str(e)[:300])
        return GenerationError(str(e)[:300])

    def chat(self, messages: list[dict], max_tokens: int = 1500) -> str:
        for attempt in range(self.retries + 1):
            try:
                resp = self.client.chat.completions.create(**self._kwargs(messages, max_tokens, False))
                text = strip_think(resp.choices[0].message.content or "")
                if not text:
                    raise GenerationError(
                        "الموديل رجّع رد فاضي (غالباً التفكير خلّص كل الـ tokens). زوّد MAX_NEW_TOKENS أو قلّل reasoning effort")
                return text
            except GenerationError:
                raise
            except Exception as e:
                err = self._wrap(e)
                if isinstance(err, RateLimited) and attempt < self.retries:
                    time.sleep(4 * 2 ** attempt)
                    continue
                raise err from e
        raise GenerationError("unreachable")

    def stream(self, messages: list[dict], max_tokens: int = 1500) -> Iterator[str]:
        flt = ThinkFilter()
        try:
            for chunk in self.client.chat.completions.create(**self._kwargs(messages, max_tokens, True)):
                piece = flt.feed(_content(chunk))
                if piece:
                    yield piece
        except Exception as e:
            raise self._wrap(e) from e
        tail = flt.flush()
        if tail:
            yield tail
