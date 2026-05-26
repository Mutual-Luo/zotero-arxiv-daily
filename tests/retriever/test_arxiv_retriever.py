"""Tests for ArxivRetriever."""

import time
from types import SimpleNamespace

import feedparser

from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever, _run_with_hard_timeout
import zotero_arxiv_daily.retriever.arxiv_retriever as arxiv_retriever
from zotero_arxiv_daily.protocol import Paper


def _sleep_and_return(value: str, delay_seconds: float) -> str:
    time.sleep(delay_seconds)
    return value


def _raise_runtime_error() -> None:
    raise RuntimeError("boom")


def test_arxiv_retriever_rss_first_uses_rss_metadata(config, mock_feedparser, monkeypatch):
    config.source.arxiv.retrieval_strategy = "rss_first"

    new_entries = [
        e for e in mock_feedparser.entries
        if e.get("arxiv_announce_type", "new") == "new"
    ]

    class FakeClient:
        def __init__(self, **kw):
            pass

        def results(self, search):
            raise AssertionError("arXiv API should not be called in rss_first mode")

    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", FakeClient)
    monkeypatch.setattr(
        arxiv_retriever,
        "extract_full_text",
        lambda paper: (_ for _ in ()).throw(AssertionError("full text should be delayed until after reranking")),
    )

    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()

    assert len(papers) == len(new_entries)
    assert set(p.title for p in papers) == set(e.title for e in new_entries)
    assert all(p.full_text is None for p in papers)
    assert all("Abstract:" not in p.abstract for p in papers)


def test_arxiv_retriever_legacy_api(config, mock_feedparser, monkeypatch):
    config.source.arxiv.retrieval_strategy = "legacy_api"
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    # The RSS fixture gives us paper IDs.  After feedparser, the code calls
    # arxiv.Client().results(search) which makes real HTTP requests.  We mock
    # the arxiv Client so the test stays offline.
    new_entries = [
        e for e in mock_feedparser.entries
        if e.get("arxiv_announce_type", "new") == "new"
    ]
    paper_ids = [e.id.removeprefix("oai:arXiv.org:") for e in new_entries]

    # Build fake ArxivResult-like objects matching each RSS entry
    fake_results = []
    for entry in new_entries:
        pid = entry.id.removeprefix("oai:arXiv.org:")
        fake_results.append(SimpleNamespace(
            title=entry.title,
            authors=[SimpleNamespace(name="Test Author")],
            summary="Test abstract",
            pdf_url=f"https://arxiv.org/pdf/{pid}",
            entry_id=f"https://arxiv.org/abs/{pid}",
            source_url=lambda pid=pid: f"https://arxiv.org/e-print/{pid}",
        ))

    class FakeClient:
        def __init__(self, **kw):
            pass
        def results(self, search):
            return iter(fake_results)

    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", FakeClient)

    # Skip file downloads in convert_to_paper
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_html", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_pdf", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_tar", lambda paper: None)

    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()

    assert len(papers) == len(new_entries)
    assert set(p.title for p in papers) == set(e.title for e in new_entries)


def test_arxiv_rss_first_fetches_selected_full_text(config, monkeypatch):
    config.source.arxiv.retrieval_strategy = "rss_first"
    calls = []

    def fake_extract_full_text(paper):
        calls.append(paper.title)
        return f"full text for {paper.title}"

    monkeypatch.setattr(arxiv_retriever, "extract_full_text", fake_extract_full_text)
    monkeypatch.setattr(arxiv_retriever, "sleep", lambda _: None)

    retriever = ArxivRetriever(config)
    selected = [
        Paper(source="arxiv", title="Selected A", authors=[], abstract="a", url="https://arxiv.org/abs/1", pdf_url="https://arxiv.org/pdf/1"),
        Paper(source="biorxiv", title="Other Source", authors=[], abstract="b", url="https://example.com", pdf_url=None),
        Paper(source="arxiv", title="Selected B", authors=[], abstract="c", url="https://arxiv.org/abs/2", pdf_url="https://arxiv.org/pdf/2"),
    ]

    prepared = retriever.prepare_selected_papers(selected)

    assert prepared is selected
    assert calls == ["Selected A", "Selected B"]
    assert selected[0].full_text == "full text for Selected A"
    assert selected[1].full_text is None
    assert selected[2].full_text == "full text for Selected B"


def test_run_with_hard_timeout_returns_value():
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 0.01), timeout=1, operation="test op", paper_title="paper"
    )
    assert result == "done"


def test_run_with_hard_timeout_returns_none_on_timeout(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 1.0), timeout=0.01, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "timed out" in warnings[0]


def test_run_with_hard_timeout_returns_none_on_failure(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _raise_runtime_error, (), timeout=1, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "boom" in warnings[0]
