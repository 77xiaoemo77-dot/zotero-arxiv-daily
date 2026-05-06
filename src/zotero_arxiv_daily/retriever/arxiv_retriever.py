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
from typing import Any, Callable, TypeVar
from loguru import logger
import requests
import time
import random

T = TypeVar("T")

DOWNLOAD_TIMEOUT = (10, 60)
HTML_EXTRACT_TIMEOUT = 60
PDF_EXTRACT_TIMEOUT = 180
TAR_EXTRACT_TIMEOUT = 180


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


def _extract_text_from_tar_worker(source_url: str, paper_id: str) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.tar.gz")
        _download_file(source_url, path)
        file_contents = extract_tex_code_from_tar(path, paper_id)
        if not file_contents or "all" not in file_contents:
            raise ValueError("Main tex file not found.")
        return file_contents["all"]


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")

    def _retrieve_raw_papers(self) -> list[ArxivResult]:
        # Be conservative with arXiv API to reduce 429/503 failures.
        client = arxiv.Client(
            page_size=20,
            num_retries=5,
            delay_seconds=10,
        )

        query = "+".join(self.config.source.arxiv.category)
        include_cross_list = self.config.source.arxiv.get("include_cross_list", False)

        # Get the latest papers from arXiv RSS feed.
        feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")

        if getattr(feed, "bozo", False):
            logger.warning(f"RSS parsing warning for ARXIV_QUERY={query}: {feed.get('bozo_exception')}")

        feed_title = getattr(feed.feed, "title", "")
        if "Feed error for query" in feed_title:
            raise Exception(f"Invalid ARXIV_QUERY: {query}.")

        raw_papers: list[ArxivResult] = []
        allowed_announce_types = {"new", "cross"} if include_cross_list else {"new"}

        all_paper_ids = [
            entry.id.removeprefix("oai:arXiv.org:")
            for entry in feed.entries
            if entry.get("arxiv_announce_type", "new") in allowed_announce_types
        ]

        # Deduplicate while preserving order. This helps when cross-listed papers appear repeatedly.
        all_paper_ids = list(dict.fromkeys(all_paper_ids))

        if self.config.executor.debug:
            all_paper_ids = all_paper_ids[:10]

        # Important: apply max_paper_num before querying the arXiv API,
        # not only when rendering the final email.
        max_paper_num = self.config.executor.get("max_paper_num", None)
        if max_paper_num is not None:
            all_paper_ids = all_paper_ids[: int(max_paper_num)]

        if not all_paper_ids:
            logger.info("No arXiv papers found for the configured categories.")
            return []

        # Get full metadata for each paper from arXiv API.
        batch_size = 20
        bar = tqdm(total=len(all_paper_ids))

        try:
            for i in range(0, len(all_paper_ids), batch_size):
                current_ids = all_paper_ids[i : i + batch_size]
                search = arxiv.Search(id_list=current_ids)

                batch: list[ArxivResult] = []
                for attempt in range(8):
                    try:
                        batch = list(client.results(search))
                        break

                    except arxiv.HTTPError as exc:
                        msg = str(exc)
                        if "429" in msg or "503" in msg:
                            sleep_s = min(300, 15 * (2 ** attempt)) + random.uniform(0, 5)
                            logger.warning(
                                f"arXiv API rate-limited or unavailable for batch "
                                f"{i // batch_size + 1}: {exc}. Sleeping {sleep_s:.1f}s "
                                f"before retry {attempt + 1}/8."
                            )
                            time.sleep(sleep_s)
                            continue

                        raise

                    except Exception as exc:
                        logger.warning(
                            f"arXiv API failed for batch {i // batch_size + 1}: "
                            f"{type(exc).__name__}: {exc}. Skipping this batch."
                        )
                        break

                else:
                    logger.warning(
                        f"Skipping arXiv batch {i // batch_size + 1} after repeated 429/503 errors."
                    )

                raw_papers.extend(batch)
                bar.update(len(current_ids))

        finally:
            bar.close()

        return raw_papers

    def convert_to_paper(self, raw_paper: ArxivResult) -> Paper:
        title = raw_paper.title
        authors = [author.name for author in raw_paper.authors]
        abstract = raw_paper.summary
        pdf_url = raw_paper.pdf_url

        # Preferred order:
        # 1. arXiv HTML, if available;
        # 2. PDF extraction;
        # 3. TeX source extraction.
        full_text = extract_text_from_html(raw_paper)
        if full_text is None:
            full_text = extract_text_from_pdf(raw_paper)
        if full_text is None:
            full_text = extract_text_from_tar(raw_paper)

        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=raw_paper.entry_id,
            pdf_url=pdf_url,
            full_text=full_text,
        )


def extract_text_from_html(paper: ArxivResult) -> str | None:
    html_url = paper.entry_id.replace("/abs/", "/html/")
    return _run_with_hard_timeout(
        _extract_text_from_html_worker,
        (html_url,),
        timeout=HTML_EXTRACT_TIMEOUT,
        operation="HTML extraction",
        paper_title=paper.title,
    )


def extract_text_from_pdf(paper: ArxivResult) -> str | None:
    if paper.pdf_url is None:
        logger.warning(f"No PDF URL available for {paper.title}")
        return None

    return _run_with_hard_timeout(
        _extract_text_from_pdf_worker,
        (paper.pdf_url,),
        timeout=PDF_EXTRACT_TIMEOUT,
        operation="PDF extraction",
        paper_title=paper.title,
    )


def extract_text_from_tar(paper: ArxivResult) -> str | None:
    source_url = paper.source_url()
    if source_url is None:
        logger.warning(f"No source URL available for {paper.title}")
        return None

    return _run_with_hard_timeout(
        _extract_text_from_tar_worker,
        (source_url, paper.entry_id),
        timeout=TAR_EXTRACT_TIMEOUT,
        operation="Tar extraction",
        paper_title=paper.title,
    )
