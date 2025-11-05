#!/usr/bin/env python3
"""Scan RSS/Atom feeds from an OPML file and capture entries containing target keywords."""

from __future__ import annotations

import argparse
import csv
import html
import logging
import re
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple
from xml.etree import ElementTree as ET

import requests

try:
    import feedparser  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    feedparser = None  # type: ignore

try:
    from bs4 import BeautifulSoup  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    BeautifulSoup = None  # type: ignore

LOGGER = logging.getLogger("rss_keyword_scanner")

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
DEFAULT_OUTPUT = "output/rss_keyword_matches.csv"
DEFAULT_KEYWORDS = ["ai", "future"]
DEFAULT_DELAY = 0.5
DEFAULT_TIMEOUT = 20
DEFAULT_MAX_ITEMS = 50
DEFAULT_ENCODING = "utf-8"

CSV_FIELDS = [
    "feed_title",
    "feed_xml_url",
    "feed_html_url",
    "entry_title",
    "entry_author",
    "entry_url",
    "published",
    "categories",
    "matched_keywords",
    "article_text",
]


@dataclass
class FeedOutline:
    title: str
    xml_url: str
    html_url: Optional[str] = None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch feeds listed in an OPML file and export entries containing specified keywords."
    )
    parser.add_argument("--opml", required=True, help="Path to the OPML file that lists RSS/Atom feeds.")
    parser.add_argument(
        "--keywords",
        nargs="*",
        default=DEFAULT_KEYWORDS,
        help="Keywords to search for (default: %(default)s).",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="CSV output path.")
    parser.add_argument("--encoding", default=DEFAULT_ENCODING, help="CSV encoding (default: %(default)s).")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="Delay between feed requests in seconds.")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="HTTP timeout per request in seconds.")
    parser.add_argument(
        "--max-items",
        type=int,
        default=DEFAULT_MAX_ITEMS,
        help="Limit the number of items per feed to inspect (default: %(default)s).",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO).")
    return parser.parse_args(argv)


def load_opml(path: Path) -> List[FeedOutline]:
    if not path.exists():
        raise FileNotFoundError(f"OPML file not found: {path}")
    tree = ET.parse(path)
    root = tree.getroot()
    feeds: OrderedDict[str, FeedOutline] = OrderedDict()

    for outline in root.findall(".//outline"):
        xml_url = outline.attrib.get("xmlUrl")
        if not xml_url:
            continue
        title = outline.attrib.get("title") or outline.attrib.get("text") or xml_url
        html_url = outline.attrib.get("htmlUrl")
        if xml_url not in feeds:
            feeds[xml_url] = FeedOutline(title=title.strip(), xml_url=xml_url.strip(), html_url=html_url)
    return list(feeds.values())


def normalize_keyword(keyword: str) -> str:
    return keyword.lower().strip()


def matched_keywords(text: str, keywords: Iterable[str]) -> List[str]:
    lowered = text.lower()
    hits = []
    for kw in keywords:
        if kw in lowered:
            hits.append(kw)
    return hits


def to_plain_text(raw: str) -> str:
    if not raw:
        return ""
    if BeautifulSoup:
        try:
            soup = BeautifulSoup(raw, "html.parser")
            return soup.get_text(" ", strip=True)
        except Exception:  # pragma: no cover - fallback on parser issues
            pass
    no_tags = re.sub(r"<[^>]+>", " ", raw)
    unescaped = html.unescape(no_tags)
    return " ".join(unescaped.split())


def fetch_feed(
    session: requests.Session,
    feed: FeedOutline,
    *,
    timeout: float,
) -> Optional[Mapping[str, object]]:
    try:
        response = session.get(feed.xml_url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        LOGGER.warning("Failed to fetch %s: %s", feed.xml_url, exc)
        return None

    if feedparser:
        parsed = feedparser.parse(response.content)
        if parsed.bozo:
            LOGGER.warning("Feed parser raised an issue for %s: %s", feed.xml_url, parsed.bozo_exception)
        return parsed

    # Minimal fallback parser for basic RSS/Atom.
    try:
        content_tree = ET.fromstring(response.content)
    except ET.ParseError as exc:
        LOGGER.warning("Failed to parse XML for %s: %s", feed.xml_url, exc)
        return None

    entries = []
    if content_tree.tag.endswith("feed"):
        entries = content_tree.findall(".//{http://www.w3.org/2005/Atom}entry")
    else:
        entries = content_tree.findall(".//item")

    return {
        "feed": {"title": feed.title},
        "entries": entries,
        "fallback": True,
    }


def iter_entries(parsed_feed: Mapping[str, object], max_items: int) -> Iterator[Mapping[str, object]]:
    if feedparser and not parsed_feed.get("fallback"):
        entries = parsed_feed.get("entries", []) or []
        for entry in entries[:max_items]:
            yield entry
        return

    # Fallback path for ElementTree-based parsing.
    entries = parsed_feed.get("entries", []) or []
    for element in entries[:max_items]:
        if not isinstance(element, ET.Element):
            continue
        entry: dict = {
            "title": element.findtext("title") or "",
            "link": element.findtext("link") or "",
            "published": element.findtext("pubDate") or element.findtext("updated") or "",
            "summary": element.findtext("description") or element.findtext("summary") or "",
        }
        entry["content"] = [{"value": entry["summary"]}]
        yield entry


def entry_text(entry: Mapping[str, object]) -> str:
    parts: List[str] = []
    title = entry.get("title")
    if isinstance(title, str):
        parts.append(to_plain_text(title))

    summary = entry.get("summary")
    if isinstance(summary, str):
        parts.append(to_plain_text(summary))

    # Feedparser-specific content field.
    contents = entry.get("content")
    if isinstance(contents, list):
        for content in contents:
            if isinstance(content, Mapping):
                value = content.get("value")
                if isinstance(value, str):
                    parts.append(to_plain_text(value))
    summary_detail = entry.get("summary_detail")
    if isinstance(summary_detail, Mapping):
        value = summary_detail.get("value")
        if isinstance(value, str):
            parts.append(to_plain_text(value))
    return "\n\n".join(filter(None, parts))


def entry_author(entry: Mapping[str, object]) -> str:
    authors = entry.get("authors")
    if isinstance(authors, list):
        names = []
        for author in authors:
            if isinstance(author, Mapping):
                name = author.get("name")
                if isinstance(name, str) and name.strip():
                    names.append(name.strip())
        if names:
            return ", ".join(names)
    author = entry.get("author")
    if isinstance(author, str):
        return author.strip()
    return ""


def entry_categories(entry: Mapping[str, object]) -> str:
    categories = entry.get("tags") or entry.get("categories") or []
    names: List[str] = []
    if isinstance(categories, list):
        for category in categories:
            if isinstance(category, Mapping):
                term = category.get("term") or category.get("label")
                if isinstance(term, str) and term.strip():
                    names.append(term.strip())
            elif isinstance(category, str) and category.strip():
                names.append(category.strip())
    return ", ".join(dict.fromkeys(names))


def entry_link(entry: Mapping[str, object]) -> str:
    link = entry.get("link")
    if isinstance(link, str) and link.strip():
        return link.strip()
    if isinstance(entry.get("links"), list):
        for candidate in entry["links"]:
            if isinstance(candidate, Mapping):
                href = candidate.get("href")
                if isinstance(href, str) and href.strip():
                    return href.strip()
    return ""


def entry_published(entry: Mapping[str, object]) -> str:
    for field in ("published", "updated", "created"):
        value = entry.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def sanitize_for_csv(value: str) -> str:
    # Remove or escape characters that commonly cause CSV formatting issues.
    cleaned = value.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    cleaned = cleaned.replace("\ufeff", "").replace("\u2028", " ").replace("\u2029", " ")
    return " ".join(cleaned.split())


def ensure_csv(path: Path, *, encoding: str = DEFAULT_ENCODING) -> Tuple[csv.DictWriter, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    fp = path.open("a", encoding=encoding, newline="")
    writer = csv.DictWriter(fp, fieldnames=CSV_FIELDS)
    if not file_exists or path.stat().st_size == 0:
        writer.writeheader()
        fp.flush()
    return writer, fp


def process_feeds(
    feeds: Sequence[FeedOutline],
    keywords: Sequence[str],
    *,
    output: Path,
    delay: float,
    timeout: float,
    max_items: int,
    encoding: str,
) -> int:
    normalized_keywords = [normalize_keyword(kw) for kw in keywords if kw.strip()]
    if not normalized_keywords:
        LOGGER.error("No valid keywords provided.")
        return 0

    writer, fp = ensure_csv(output, encoding=encoding)
    matched_count = 0
    session = requests.Session()
    session.headers.update({"User-Agent": DEFAULT_USER_AGENT})

    try:
        for index, feed in enumerate(feeds, start=1):
            LOGGER.info("Processing feed %s/%s: %s", index, len(feeds), feed.title)
            parsed = fetch_feed(session, feed, timeout=timeout)
            if not parsed:
                continue
            for entry in iter_entries(parsed, max_items):
                plain_text = entry_text(entry)
                if not plain_text:
                    continue
                hits = matched_keywords(plain_text, normalized_keywords)
                if not hits:
                    continue
                matched_count += 1
                row = {
                    "feed_title": sanitize_for_csv(feed.title),
                    "feed_xml_url": feed.xml_url,
                    "feed_html_url": feed.html_url or "",
                    "entry_title": sanitize_for_csv(entry.get("title", "")),
                    "entry_author": sanitize_for_csv(entry_author(entry)),
                    "entry_url": entry_link(entry),
                    "published": sanitize_for_csv(entry_published(entry)),
                    "categories": sanitize_for_csv(entry_categories(entry)),
                    "matched_keywords": ", ".join(sorted(set(hits))),
                    "article_text": sanitize_for_csv(plain_text),
                }
                writer.writerow(row)
                fp.flush()

            if delay > 0:
                time.sleep(delay)
    finally:
        fp.close()
        session.close()
    return matched_count


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    try:
        feeds = load_opml(Path(args.opml))
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Failed to load OPML: %s", exc, exc_info=True)
        return 1

    if not feeds:
        LOGGER.warning("No feeds found in %s.", args.opml)
        return 0

    LOGGER.info("Loaded %s unique feed(s) from %s", len(feeds), args.opml)
    matches = process_feeds(
        feeds,
        args.keywords,
        output=Path(args.output).expanduser(),
        delay=max(args.delay, 0.0),
        timeout=args.timeout,
        max_items=args.max_items,
        encoding=args.encoding,
    )
    LOGGER.info("Finished. Found %s matching entries.", matches)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
