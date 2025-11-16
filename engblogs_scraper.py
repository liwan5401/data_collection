#!/usr/bin/env python3
"""Scrape engblogs.dev cards and the linked articles, then export to CSV.

The script walks every paginated listing page on https://www.engblogs.dev,
collects the card metadata (title, publisher, publish date, summary, URL),
downloads each linked article, extracts the readable text, and writes the
results to a CSV file. The extraction relies on trafilatura for robustness
across the myriad company blogs referenced by engblogs.dev.

Dependencies:
    pip install requests beautifulsoup4 trafilatura

Typical usage:
    python3 data_collection/engblogs_scraper.py --output engblogs.csv
"""

from __future__ import annotations

import argparse
import csv
import html
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter

try:
    from ftfy import fix_text as _fix_unicode_text
except ImportError:  # pragma: no cover - optional dependency
    _fix_unicode_text = None

try:
    import trafilatura
except ImportError as exc:  # pragma: no cover - makes failure explicit for the user
    raise SystemExit(
        "trafilatura is required. Install with `pip install trafilatura`."
    ) from exc


LOGGER = logging.getLogger("engblogs")
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)  # Ordinary desktop UA keeps the responses consistent.
DATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b")


def clean_text(value: str | None) -> str:
    """Collapse whitespace inside strings."""
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def clean_article_text(text: Optional[str]) -> str:
    """Remove HTML artifacts, decode entities, and normalize whitespace."""
    if not text:
        return ""

    text = (
        html.unescape(text)
        .replace("\xa0", " ")
        .replace("Â", " ")
        .strip()
    )
    if _fix_unicode_text:
        text = _fix_unicode_text(text)
    if "<" in text and ">" in text:
        # When trafilatura returns HTML fragments we strip tags defensively.
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text("\n")

    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def strip_unwanted_blocks(raw_html: str) -> str:
    """Remove code/script/style blocks that shouldn't appear in article text."""
    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup.find_all(["script", "style", "noscript"]):
        tag.decompose()
    for tag in soup.find_all(["pre", "code", "samp", "kbd"]):
        tag.decompose()
    return str(soup)


def response_text(response: requests.Response) -> str:
    """Return response text using apparent encoding when servers omit headers."""
    if not response.encoding:
        response.encoding = response.apparent_encoding or "utf-8"
    return response.text


def normalize_date(raw: str) -> str:
    """Return ISO date when possible, otherwise the raw string."""
    raw = raw.strip()
    for fmt in ("%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    return raw


@dataclass
class ArticleRecord:
    """Structured representation of a single card on engblogs.dev."""

    title: str
    publisher: str
    publish_date: str
    summary: str
    url: str
    detail_content: str = ""
    list_page: int = 0

    def as_csv_row(self) -> dict[str, str]:
        return {
            "title": self.title,
            "publisher": self.publisher,
            "publish_date": self.publish_date,
            "summary": self.summary,
            "detail_content": self.detail_content,
            "source_url": self.url,
            "engblogs_page": str(self.list_page),
        }


class EngBlogsScraper:
    def __init__(
        self,
        *,
        start_page: int = 1,
        end_page: int = 733,
        base_url: str = "https://www.engblogs.dev",
        concurrency: int = 6,
        listing_timeout: float = 20.0,
        detail_timeout: float = 25.0,
        detail_retries: int = 3,
        retry_backoff: float = 1.5,
    ) -> None:
        if start_page < 1:
            raise ValueError("start_page must be >= 1")
        if end_page < start_page:
            raise ValueError("end_page must be >= start_page")

        self.start_page = start_page
        self.end_page = end_page
        self.base_url = base_url.rstrip("/")
        self.concurrency = max(1, concurrency)
        self.listing_timeout = max(5.0, listing_timeout)
        self.detail_timeout = max(5.0, detail_timeout)
        self.detail_retries = max(1, detail_retries)
        self.retry_backoff = max(0.5, retry_backoff)

        pool_size = max(10, self.concurrency * 2)

        def build_session() -> requests.Session:
            session = requests.Session()
            session.headers.update({"User-Agent": USER_AGENT})
            adapter = HTTPAdapter(pool_maxsize=pool_size)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            return session

        self.listing_session = build_session()
        self.detail_session = build_session()

    def collect_articles(self) -> List[ArticleRecord]:
        """Download and parse every listing page in the given range."""
        articles: List[ArticleRecord] = []
        seen_urls: set[str] = set()

        for page in range(self.start_page, self.end_page + 1):
            try:
                html = self._fetch_listing_html(page)
            except requests.RequestException as exc:
                LOGGER.error("Failed to fetch listing page %s: %s", page, exc)
                continue

            parsed = self._parse_listing(html, page)
            if not parsed:
                LOGGER.warning(
                    "Listing page %s returned zero cards; stopping early.", page
                )
                break

            inserted = 0
            for record in parsed:
                if record.url in seen_urls:
                    continue
                seen_urls.add(record.url)
                articles.append(record)
                inserted += 1

            LOGGER.info(
                "Page %s: parsed %s cards (%s new, %s total)",
                page,
                len(parsed),
                inserted,
                len(articles),
            )
        return articles

    def _fetch_listing_html(self, page: int) -> str:
        params = {"page": page}
        response = self.listing_session.get(
            f"{self.base_url}/", params=params, timeout=self.listing_timeout
        )
        response.raise_for_status()
        return response_text(response)

    def _parse_listing(self, html: str, page: int) -> List[ArticleRecord]:
        soup = BeautifulSoup(html, "html.parser")
        anchors = soup.select("main a[target='_blank'][href]")
        records: List[ArticleRecord] = []

        for anchor in anchors:
            href = anchor.get("href") or ""
            url = urljoin(f"{self.base_url}/", href)
            title = clean_text(anchor.find("h3").get_text(strip=True) if anchor.find("h3") else "")
            summary_tag = anchor.find("p")
            summary = clean_text(summary_tag.get_text(" ", strip=True) if summary_tag else "")
            publisher = self._extract_publisher(anchor)
            publish_date = self._extract_publish_date(anchor)

            if not (title and summary and publisher and publish_date):
                LOGGER.debug(
                    "Skipping incomplete card on page %s (%s)",
                    page,
                    url,
                )
                continue

            records.append(
                ArticleRecord(
                    title=title,
                    publisher=publisher,
                    publish_date=publish_date,
                    summary=summary,
                    url=url,
                    list_page=page,
                )
            )

        return records

    def _extract_publisher(self, anchor: Tag) -> str:
        badge_span = anchor.select_one("div.flex span")
        if badge_span:
            text = clean_text(badge_span.get_text())
            if text and not DATE_RE.fullmatch(text):
                return text

        logo = anchor.select_one("div.flex img[alt]")
        if logo and logo.get("alt"):
            return clean_text(logo["alt"])

        for span in anchor.find_all("span"):
            text = clean_text(span.get_text())
            if text and not DATE_RE.fullmatch(text):
                return text
        return ""

    def _extract_publish_date(self, anchor: Tag) -> str:
        for candidate in anchor.find_all(string=DATE_RE):
            match = DATE_RE.search(candidate)
            if match:
                return normalize_date(match.group())
        return ""

    def fetch_all_details(self, articles: List[ArticleRecord]) -> None:
        if not articles:
            LOGGER.warning("No articles to enrich with details.")
            return

        total = len(articles)
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            future_to_article = {
                executor.submit(self._fetch_single_detail, article): article
                for article in articles
            }

            for completed, future in enumerate(as_completed(future_to_article), 1):
                future.result()
                if completed % 25 == 0 or completed == total:
                    LOGGER.info(
                        "Fetched article bodies for %s/%s entries",
                        completed,
                        total,
                    )

    def _candidate_urls(self, original_url: str) -> List[str]:
        parsed = urlparse(original_url)
        hosts: List[str] = [parsed.netloc]

        if parsed.netloc.startswith("cdn."):
            hosts.append(parsed.netloc[len("cdn.") :])
        if parsed.netloc.startswith("www."):
            hosts.append(parsed.netloc[len("www.") :])
        elif parsed.netloc:
            hosts.append(f"www.{parsed.netloc}")

        candidates: List[str] = []
        seen: set[str] = set()
        for host in hosts:
            if not host:
                continue
            rebuilt = urlunparse(parsed._replace(netloc=host))
            if rebuilt not in seen:
                seen.add(rebuilt)
                candidates.append(rebuilt)
        return candidates

    def _fetch_single_detail(self, article: ArticleRecord) -> None:
        candidate_urls = self._candidate_urls(article.url)
        last_error: Optional[Exception] = None

        for url_index, url in enumerate(candidate_urls, 1):
            for attempt in range(1, self.detail_retries + 1):
                try:
                    response = self.detail_session.get(url, timeout=self.detail_timeout)
                    response.raise_for_status()
                    article.detail_content = self._extract_article_body(
                        response_text(response), response.url
                    )
                    if url_index > 1:
                        LOGGER.debug(
                            "Fetched detail for %s via fallback URL %s",
                            article.url,
                            url,
                        )
                    if not article.detail_content:
                        LOGGER.warning(
                            "Empty detail content after extraction for %s",
                            article.url,
                        )
                    return
                except requests.RequestException as exc:
                    last_error = exc
                    wait = min(self.retry_backoff * attempt, 8.0)
                    LOGGER.warning(
                        "Detail fetch failed (%s/%s) for %s (candidate %s/%s): %s",
                        attempt,
                        self.detail_retries,
                        article.url,
                        url_index,
                        len(candidate_urls),
                        exc,
                    )
                    if attempt < self.detail_retries:
                        time.sleep(wait)
                except Exception as exc:  # pragma: no cover - extraction edge cases
                    last_error = exc
                    LOGGER.exception("Unexpected error parsing %s: %s", article.url, exc)
                    break

        if last_error:
            LOGGER.error("Giving up on %s: %s", article.url, last_error)
        article.detail_content = ""

    def _extract_article_body(self, html: str, url: str) -> str:
        sanitized_html = strip_unwanted_blocks(html)
        text = trafilatura.extract(
            sanitized_html,
            url=url,
            include_comments=False,
            include_images=False,
            favor_precision=True,
        )
        cleaned = clean_article_text(text)
        if cleaned:
            return cleaned

        # Fallback: best-effort paragraph concatenation.
        soup = BeautifulSoup(sanitized_html, "html.parser")
        paragraphs = [
            clean_text(p.get_text(" ", strip=True))
            for p in soup.find_all("p")
        ]
        filtered = [p for p in paragraphs if len(p.split()) >= 3]
        return clean_article_text("\n\n".join(filtered))


def write_csv(path: Path, rows: Iterable[ArticleRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "title",
        "publisher",
        "publish_date",
        "summary",
        "detail_content",
        "source_url",
        "engblogs_page",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_csv_row())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download engblogs.dev listings, fetch article bodies, and export to CSV."
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=1,
        help="First page (1-indexed) to fetch. Default: 1",
    )
    parser.add_argument(
        "--end-page",
        type=int,
        default=733,
        help="Last page (inclusive) to fetch. Default: 733",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=6,
        help="Number of concurrent article downloads. Default: 6",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("engblogs_articles.csv"),
        help="CSV output path.",
    )
    parser.add_argument(
        "--detail-timeout",
        type=float,
        default=25.0,
        help="Timeout (seconds) for downloading article detail pages.",
    )
    parser.add_argument(
        "--detail-retries",
        type=int,
        default=3,
        help="Retry attempts for article detail requests.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ...).",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    scraper = EngBlogsScraper(
        start_page=args.start_page,
        end_page=args.end_page,
        concurrency=args.concurrency,
        detail_timeout=args.detail_timeout,
        detail_retries=args.detail_retries,
    )

    LOGGER.info(
        "Starting engblogs scrape from page %s to %s",
        args.start_page,
        args.end_page,
    )
    articles = scraper.collect_articles()
    LOGGER.info("Collected %s articles from listings.", len(articles))

    scraper.fetch_all_details(articles)
    write_csv(args.output, articles)
    LOGGER.info("Saved %s rows to %s", len(articles), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
