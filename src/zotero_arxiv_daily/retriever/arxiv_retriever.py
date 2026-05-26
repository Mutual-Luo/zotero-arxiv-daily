from .base import BaseRetriever, register_retriever
import arxiv
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar
from tempfile import TemporaryDirectory
import feedparser
from tqdm import tqdm
import multiprocessing
import os
from queue import Empty
from time import sleep
from typing import Any, Callable, TypeVar
from loguru import logger
import requests
import re
from dataclasses import dataclass

T = TypeVar("T")

DOWNLOAD_TIMEOUT = (10, 60)
PDF_EXTRACT_TIMEOUT = 180
TAR_EXTRACT_TIMEOUT = 180
ARXIV_RETRIEVAL_STRATEGIES = {"rss_first", "legacy_api"}


@dataclass
class RssArxivAuthor:
    name: str


@dataclass
class RssArxivResult:
    paper_id: str
    title: str
    authors: list[RssArxivAuthor]
    summary: str
    pdf_url: str
    entry_id: str

    def source_url(self) -> str:
        return f"https://arxiv.org/e-print/{self.paper_id}"


def _download_file(url: str, path: str) -> None:
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def _run_in_subprocess(
    result_queue: Any,
    func: Callable[..., T | None],
    args: tuple[Any, ...],
) -> None:
    try:
        result_queue.put(("ok", func(*args)))
    except Exception as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_with_hard_timeout(
    func: Callable[..., T | None],
    args: tuple[Any, ...],
    *,
    timeout: float,
    operation: str,
    paper_title: str,
) -> T | None:
    start_methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in start_methods else start_methods[0])
    result_queue = context.Queue()
    process = context.Process(target=_run_in_subprocess, args=(result_queue, func, args))
    process.start()

    try:
        status, payload = result_queue.get(timeout=timeout)
    except Empty:
        if process.is_alive():
            process.kill()
        process.join(5)
        result_queue.close()
        result_queue.join_thread()
        logger.warning(f"{operation} timed out for {paper_title} after {timeout} seconds")
        return None

    process.join(5)
    result_queue.close()
    result_queue.join_thread()

    if status == "ok":
        return payload

    logger.warning(f"{operation} failed for {paper_title}: {payload}")
    return None


def _extract_text_from_pdf_worker(pdf_url: str) -> str:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.pdf")
        _download_file(pdf_url, path)
        return extract_markdown_from_pdf(path)


def _extract_text_from_html_worker(html_url: str) -> str | None:
    import trafilatura

    downloaded = trafilatura.fetch_url(html_url)
    if downloaded is None:
        raise ValueError(f"Failed to download HTML from {html_url}")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text:
        raise ValueError(f"No text extracted from {html_url}")
    return text


def _extract_text_from_tar_worker(source_url: str, paper_id: str, paper_title: str | None = None) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.tar.gz")
        _download_file(source_url, path)
        file_contents = extract_tex_code_from_tar(path, paper_id, paper_title=paper_title)
        if not file_contents or "all" not in file_contents:
            raise ValueError("Main tex file not found.")
        return file_contents["all"]


def _clean_rss_summary(summary: str) -> str:
    if "Abstract:" in summary:
        summary = summary.split("Abstract:", 1)[1]
    return re.sub(r"\s+", " ", summary).strip()


def _get_rss_authors(entry: Any) -> list[RssArxivAuthor]:
    authors = []
    for author in entry.get("authors", []) or []:
        name = author.get("name") if hasattr(author, "get") else getattr(author, "name", None)
        if name:
            authors.append(RssArxivAuthor(name=name))

    if authors:
        return authors

    creator = entry.get("dc_creator") or entry.get("author") or entry.get("creator") or ""
    names = [name.strip() for name in creator.split(",") if name.strip()]
    return [RssArxivAuthor(name=name) for name in names]


def _rss_entry_to_result(entry: Any) -> RssArxivResult:
    paper_id = entry.id.removeprefix("oai:arXiv.org:")
    entry_id = entry.get("link") or f"https://arxiv.org/abs/{paper_id}"
    return RssArxivResult(
        paper_id=paper_id,
        title=entry.title,
        authors=_get_rss_authors(entry),
        summary=_clean_rss_summary(entry.get("summary", "")),
        pdf_url=f"https://arxiv.org/pdf/{paper_id}",
        entry_id=entry_id,
    )


def _paper_title(paper: Any) -> str:
    return getattr(paper, "title", str(paper))


def _paper_entry_id(paper: Any) -> str | None:
    return getattr(paper, "entry_id", None) or getattr(paper, "url", None)


def _paper_pdf_url(paper: Any) -> str | None:
    return getattr(paper, "pdf_url", None)


def _paper_id_from_url(url: str | None) -> str | None:
    if not url:
        return None
    paper_id = url.rstrip("/").split("/")[-1]
    if paper_id.endswith(".pdf"):
        paper_id = paper_id[:-4]
    return paper_id or None


def _paper_id(paper: Any) -> str | None:
    return _paper_id_from_url(_paper_pdf_url(paper)) or _paper_id_from_url(_paper_entry_id(paper))


def _paper_source_url(paper: Any) -> str | None:
    source_url = getattr(paper, "source_url", None)
    if callable(source_url):
        return source_url()
    paper_id = _paper_id(paper)
    if paper_id is None:
        return None
    return f"https://arxiv.org/e-print/{paper_id}"


def extract_full_text(paper: Any) -> str | None:
    full_text = extract_text_from_tar(paper)
    if full_text is None:
        full_text = extract_text_from_html(paper)
    if full_text is None:
        full_text = extract_text_from_pdf(paper)
    return full_text


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")
        self.retrieval_strategy = self.config.source.arxiv.get("retrieval_strategy", "rss_first")
        if self.retrieval_strategy not in ARXIV_RETRIEVAL_STRATEGIES:
            raise ValueError(
                f"source.arxiv.retrieval_strategy must be one of {sorted(ARXIV_RETRIEVAL_STRATEGIES)}, "
                f"got {self.retrieval_strategy!r}."
            )

    def conversion_delay_seconds(self) -> float:
        return 0 if self.retrieval_strategy == "rss_first" else 1

    def _retrieve_raw_papers(self) -> list[ArxivResult | RssArxivResult]:
        query = '+'.join(self.config.source.arxiv.category)
        include_cross_list = self.config.source.arxiv.get("include_cross_list", False)
        # Get the latest paper from arxiv rss feed
        feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
        if 'Feed error for query' in feed.feed.title:
            raise Exception(f"Invalid ARXIV_QUERY: {query}.")
        raw_papers = []
        allowed_announce_types = {"new", "cross"} if include_cross_list else {"new"}
        entries = [
            i
            for i in feed.entries
            if i.get("arxiv_announce_type", "new") in allowed_announce_types
        ]
        if self.config.executor.debug:
            entries = entries[:10]

        if self.retrieval_strategy == "rss_first":
            return [_rss_entry_to_result(entry) for entry in entries]

        client = arxiv.Client(num_retries=10, delay_seconds=10)
        all_paper_ids = [i.id.removeprefix("oai:arXiv.org:") for i in entries]

        # Get full information of each paper from arxiv api
        bar = tqdm(total=len(all_paper_ids))
        max_batch_retries = 5
        batch_retry_delay = 30
        for i in range(0, len(all_paper_ids), 20):
            search = arxiv.Search(id_list=all_paper_ids[i:i + 20])
            for attempt in range(max_batch_retries):
                try:
                    batch = list(client.results(search))
                    bar.update(len(batch))
                    raw_papers.extend(batch)
                    break
                except arxiv.HTTPError as exc:
                    if exc.status == 429 and attempt < max_batch_retries - 1:
                        wait = batch_retry_delay * (attempt + 1)
                        logger.warning(f"arXiv API 429 on batch {i // 20}, retry {attempt + 1}/{max_batch_retries} in {wait}s")
                        sleep(wait)
                    else:
                        raise
            if i + 20 < len(all_paper_ids):
                sleep(3)
        bar.close()

        return raw_papers

    def convert_to_paper(self, raw_paper: ArxivResult | RssArxivResult) -> Paper:
        title = raw_paper.title
        authors = [a.name for a in raw_paper.authors]
        abstract = raw_paper.summary
        pdf_url = raw_paper.pdf_url
        full_text = extract_full_text(raw_paper) if self.retrieval_strategy == "legacy_api" else None
        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=raw_paper.entry_id,
            pdf_url=pdf_url,
            full_text=full_text,
        )

    def prepare_selected_papers(self, papers: list[Paper]) -> list[Paper]:
        if self.retrieval_strategy != "rss_first":
            return papers

        selected = [paper for paper in papers if paper.source == self.name and paper.full_text is None]
        if not selected:
            return papers

        logger.info(f"Fetching arXiv full text for {len(selected)} selected papers after reranking")
        for paper in tqdm(selected, desc="Fetching arXiv full text"):
            paper.full_text = extract_full_text(paper)
            sleep(1)
        return papers


def extract_text_from_html(paper: Any) -> str | None:
    entry_id = _paper_entry_id(paper)
    if entry_id is None:
        logger.warning(f"No arXiv entry URL available for {_paper_title(paper)}")
        return None
    html_url = entry_id.replace("/abs/", "/html/")
    try:
        return _extract_text_from_html_worker(html_url)
    except Exception as exc:
        logger.warning(f"HTML extraction failed for {_paper_title(paper)}: {exc}")
        return None


def extract_text_from_pdf(paper: Any) -> str | None:
    pdf_url = _paper_pdf_url(paper)
    if pdf_url is None:
        logger.warning(f"No PDF URL available for {_paper_title(paper)}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_pdf_worker,
        (pdf_url,),
        timeout=PDF_EXTRACT_TIMEOUT,
        operation="PDF extraction",
        paper_title=_paper_title(paper),
    )


def extract_text_from_tar(paper: Any) -> str | None:
    source_url = _paper_source_url(paper)
    if source_url is None:
        logger.warning(f"No source URL available for {_paper_title(paper)}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_tar_worker,
        (source_url, _paper_id(paper) or _paper_entry_id(paper) or _paper_title(paper), _paper_title(paper)),
        timeout=TAR_EXTRACT_TIMEOUT,
        operation="Tar extraction",
        paper_title=_paper_title(paper),
    )
