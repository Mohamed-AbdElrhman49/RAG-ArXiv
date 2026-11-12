import pytest

from app.rag.generator import ThinkFilter, _content, strip_think
from app.rag.pipeline import NO_ANSWER, RagPipeline, build_messages, check_citations, is_refusal


# ---- <think> stripping ----
def test_strip_think_block():
    assert strip_think("<think>hmm\nlong</think>The answer [1].") == "The answer [1]."


def test_strip_think_unclosed_returns_empty():
    assert strip_think("<think>still thinking...") == ""


def _stream(chunks):
    f, out = ThinkFilter(), ""
    for c in chunks:
        out += f.feed(c)
    return out + f.flush()


def test_think_filter_split_tags():
    assert _stream(["<th", "ink>secret</thi", "nk>Hello", " world [1]"]) == "Hello world [1]"


def test_think_filter_no_think():
    assert _stream(["plain ", "text"]) == "plain text"


def test_think_filter_partial_lt_is_not_lost():
    assert _stream(["a < b ", "and c"]) == "a < b and c"


def test_think_filter_unclosed_think_is_dropped():
    assert _stream(["ok <think>never ends"]) == "ok "


# ---- الـ bug الأصلي: content=None بقت "None" ----
class _D:  # delta
    def __init__(self, c): self.content = c
class _C:  # choice
    def __init__(self, c): self.delta = _D(c)
class _Chunk:
    def __init__(self, c, empty=False): self.choices = [] if empty else [_C(c)]


def test_none_content_is_empty_string_not_None_text():
    assert _content(_Chunk(None)) == ""
    assert _content(_Chunk("hi")) == "hi"
    assert _content(_Chunk(None, empty=True)) == ""


# ---- citations ----
def test_check_citations():
    assert check_citations("A [1]. B [2][3]. C [7].", 3) == ([1, 2, 3], [7])
    assert check_citations("A [1, 2].", 3) == ([1, 2], [])
    assert check_citations("no cites", 3) == ([], [])


def test_refusal_detection():
    assert is_refusal(NO_ANSWER) and is_refusal(NO_ANSWER.rstrip("."))
    assert not is_refusal("It uses a reranker [1].")


def test_build_messages_numbers_sources():
    res = [{"title": "T", "arxiv_id": "1", "page": 2, "section": "S", "text": "body", "chunk_id": "c"}]
    msgs = build_messages("q?", res)
    assert "[1] T (arXiv:1, p.2, section: S)" in msgs[1]["content"] and msgs[0]["role"] == "system"


# ---- pipeline كامل بـ generator وهمي ----
class FakeGen:
    def __init__(self, text): self.text = text
    def chat(self, messages, max_tokens): return self.text
    def stream(self, messages, max_tokens): yield from self.text.split(" ")


def test_pipeline_answer_flags(retriever):
    p = RagPipeline(retriever, FakeGen("Reranking raised MRR [1]. Also [9]."), top_k=2)
    out = p.answer("cross-encoder reranker MRR")
    assert out["invalid_citations"] == [9] and out["uncited"] is False
    assert [c["n"] for c in out["citations"]] == [1] and out["citations"][0]["url"].startswith("https://arxiv.org/abs/")


def test_pipeline_uncited_and_refusal(retriever):
    assert RagPipeline(retriever, FakeGen("No cites here."), top_k=2).answer("reranker")["uncited"] is True
    out = RagPipeline(retriever, FakeGen(NO_ANSWER), top_k=2).answer("chocolate cake")
    assert out["refused"] and not out["uncited"]


def test_pipeline_stream_events(retriever):
    evs = list(RagPipeline(retriever, FakeGen("a b [1]"), top_k=2).stream("reranker"))
    assert [e for e, _ in evs][0] == "sources" and evs[-1][0] == "done"
    assert "".join(d["text"] for e, d in evs if e == "token") == "ab[1]"  # FakeGen بيطلّع كلمات من غير مسافات
    assert evs[-1][1]["answer"] == "ab[1]"
