#!/usr/bin/env python3
"""Fetch PressReader search results and article content, then export to CSV."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import IO, Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

import requests

LOGGER = logging.getLogger("pressreader")

DEFAULT_BASE_URL = "https://ingress.pressreader.com/services"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
DEFAULT_PAGE_SIZE = 20
DEFAULT_MAX_RESULTS = 100
DEFAULT_DELAY = 0.75
DEFAULT_JITTER = 0.35
DEFAULT_PAGE_PAUSE = 1.0
DEFAULT_BATCH_PAUSE = 1.0
DEFAULT_DETAIL_PAUSE = 1.2
ARTICLE_BATCH_SIZE = 20
CSV_FIELDNAMES = [
    "search_text",
    "highlights",
    "region_id",
    "profile_id",
    "issue_id",
    "issue_title",
    "issue_date",
    "issue_country",
    "issue_cid",
    "page",
    "page_name",
    "section",
    "title",
    "subtitle",
    "hyphenated_title",
    "language",
    "authors",
    "byline",
    "is_top_article",
    "rank",
    "rate",
    "similars_count",
    "current_text_length",
    "total_text_length",
    "classification_tags",
    "related_titles",
    "images",
    "article_text",
]


def load_config(path: Optional[Path]) -> Dict[str, Any]:
    if not path:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Config file {path} is empty.")
    data = json.loads(content)
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a JSON object.")
    return data


def ensure_parent_dir(path: Path) -> None:
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


class PressReaderClient:
    def __init__(
        self,
        auth_token: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        delay: float = DEFAULT_DELAY,
        jitter: float = DEFAULT_JITTER,
        timeout: float = 30.0,
        user_agent: str = DEFAULT_USER_AGENT,
        session: Optional[requests.Session] = None,
    ) -> None:
        if not auth_token:
            raise ValueError("auth_token must be provided.")
        self.base_url = base_url.rstrip("/")
        self.delay = max(delay, 0.0)
        self.jitter = max(jitter, 0.0)
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {auth_token}",
                "Accept": "*/*",
                "Accept-Encoding": "gzip, deflate, br, zstd",
                "Accept-Language": "en-US,en;q=0.9",
                "Origin": "https://www.pressreader.com",
                "Referer": "https://www.pressreader.com/",
                "User-Agent": user_agent,
            }
        )

    def _sleep(self) -> None:
        if self.delay <= 0.0:
            return
        jitter = random.uniform(0.0, self.jitter) if self.jitter > 0 else 0.0
        time.sleep(self.delay + jitter)

    def _get(self, path: str, params: Mapping[str, Any]) -> Any:
        self._sleep()
        url = f"{self.base_url}/{path.lstrip('/')}"
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json() if response.headers.get("Content-Type", "").startswith("application/json") else response.text

    def get_count(self, params: Mapping[str, Any]) -> int:
        result = self._get("fts/GetCount", params)
        if isinstance(result, int):
            return result
        if isinstance(result, str):
            try:
                return int(result.strip())
            except ValueError:
                pass
        raise ValueError(f"Unexpected GetCount response: {result!r}")

    def search_articles(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        data = self._get("search/GetArticles", params)
        if not isinstance(data, dict):
            raise ValueError(f"Unexpected GetArticles response: {data!r}")
        return data

    def get_items(self, article_ids: Sequence[str]) -> Dict[str, Any]:
        if not article_ids:
            return {"Articles": []}
        params = {
            "articles": ",".join(article_ids),
            "pages": "",
            "socialInfoArticles": "",
            "comment": "LatestByAll",
            "options": 1,
            "viewType": "search",
            "IsHyphenated": "true",
        }
        data = self._get("articles/GetItems", params)
        if not isinstance(data, dict):
            raise ValueError(f"Unexpected GetItems response: {data!r}")
        return data

    def get_article(self, article_id: str) -> Dict[str, Any]:
        params = {
            "key": article_id,
            "viewType": "search",
            "IsHyphenated": "true",
        }
        data = self._get("articles/GetArticle", params)
        if not isinstance(data, dict):
            raise ValueError(f"Unexpected GetArticle response: {data!r}")
        return data


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download article data from ingress.pressreader.com and save to CSV."
    )
    parser.add_argument("--config", help="Optional JSON config file.", default=None)
    parser.add_argument("--auth-token", help="PressReader bearer token.", default=None)
    parser.add_argument("--search-text", help="Search keywords.", default=None)
    parser.add_argument("--languages", help="Comma separated language codes.", default=None)
    parser.add_argument("--countries", help="Comma separated country codes.", default=None)
    parser.add_argument("--categories", help="Comma separated category IDs.", default=None)
    parser.add_argument("--newspapers", help="Comma separated newspaper IDs.", default=None)
    parser.add_argument("--search-author", help="Author to filter by.", default=None)
    parser.add_argument("--search-in", default=None, help="Search scope (e.g., ALL).")
    parser.add_argument("--search-for", default=None, help="Search target (e.g., All).")
    parser.add_argument("--order-by", default=None, help="Sort order (e.g., Relevance).")
    parser.add_argument("--range", dest="range_", default=None, help="Date range mode (e.g., Range).")
    parser.add_argument("--start-date", help="Start date YYYY-MM-DD.", default=None)
    parser.add_argument("--stop-date", help="Stop date YYYY-MM-DD.", default=None)
    parser.add_argument("--page-size", type=int, default=None, help="Number of hits per request.")
    parser.add_argument("--max-results", type=int, default=None, help="Maximum articles to fetch.")
    parser.add_argument("--delay", type=float, default=None, help="Base delay between requests.")
    parser.add_argument("--jitter", type=float, default=None, help="Additional random delay.")
    parser.add_argument("--page-pause", type=float, default=None, help="Pause in seconds between paginated search calls.")
    parser.add_argument("--batch-pause", type=float, default=None, help="Pause in seconds between metadata batches.")
    parser.add_argument("--detail-pause", type=float, default=None, help="Pause in seconds between full article fetches.")
    parser.add_argument("--output", default=None, help="CSV output path.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    parser.add_argument("--timeout", type=float, default=None, help="HTTP timeout (seconds).")
    return parser.parse_args(argv)


def resolve_option(
    key: str,
    *,
    args: Mapping[str, Any],
    config: Mapping[str, Any],
    default: Any,
    config_key: Optional[str] = None,
) -> Any:
    cli_value = args.get(key)
    if cli_value not in (None, ""):
        return cli_value
    cfg_key = config_key or key
    if cfg_key in config and config[cfg_key] not in (None, ""):
        return config[cfg_key]
    return default


def build_base_params(options: Mapping[str, Any]) -> Dict[str, Any]:
    params = {
        "searchText": options["search_text"],
        "searchAuthor": options.get("search_author", ""),
        "languages": options.get("languages", ""),
        "newspapers": options.get("newspapers", ""),
        "countries": options.get("countries", ""),
        "categories": options.get("categories", ""),
        "range": options.get("range", "Range"),
        "searchIn": options.get("search_in", "ALL"),
        "startDate": options.get("start_date", ""),
        "stopDate": options.get("stop_date", ""),
        "orderBy": options.get("order_by", "Relevance"),
        "hideSame": options.get("hide_same", 0),
        "searchFor": options.get("search_for", "All"),
    }
    params["pageNumber"] = 0
    params["pageSize"] = options.get("page_size", DEFAULT_PAGE_SIZE)
    return params


def chunked(seq: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for idx in range(0, len(seq), size):
        yield seq[idx : idx + size]


def normalize_text(text: str) -> str:
    return text.replace("\u00ad", "").replace("\xa0", " ").strip()


def article_text(blocks: Iterable[Mapping[str, Any]]) -> str:
    if not blocks:
        return ""
    texts: List[str] = []
    for block in blocks:
        if block.get("Role") != "text":
            continue
        value = block.get("Text")
        if isinstance(value, str):
            value = normalize_text(value)
            if value:
                texts.append(value)
    return "\n\n".join(texts)


def join_list(items: Optional[Iterable[str]], *, sep: str = "; ") -> str:
    if not items:
        return ""
    cleaned = [normalize_text(str(item)) for item in items if item is not None]
    return sep.join(filter(None, cleaned))


def collect_articles(
    client: PressReaderClient,
    options: Mapping[str, Any],
) -> Dict[str, Any]:
    base_params = build_base_params(options)

    count_params = dict(base_params)
    count_params["pageSize"] = 1
    LOGGER.info("Requesting total count...")
    total_count = client.get_count(count_params)
    LOGGER.info("Total matching articles reported: %s", total_count)

    remaining = options.get("max_results", DEFAULT_MAX_RESULTS)
    row_number = 0
    collected: List[Dict[str, Any]] = []
    highlights: List[str] = []

    page_pause = float(options.get("page_pause", 0.0) or 0.0)

    while remaining > 0:
        page_size = min(options.get("page_size", DEFAULT_PAGE_SIZE), remaining)
        page_params = dict(base_params)
        page_params["rowNumber"] = row_number
        page_params["pageSize"] = page_size
        LOGGER.debug("Fetching articles rowNumber=%s pageSize=%s", row_number, page_size)
        page = client.search_articles(page_params)
        items = page.get("Items") or []
        LOGGER.info("Received %s items for rowNumber=%s", len(items), row_number)
        if not items:
            break
        collected.extend(items)
        row_number += len(items)
        remaining -= len(items)
        if highlights == [] and isinstance(page.get("Highlights"), list):
            highlights = [normalize_text(str(h)) for h in page["Highlights"] if h]
        if len(items) < page_size:
            break
        if page_pause > 0:
            LOGGER.debug("Sleeping %.2fs between search pages.", page_pause)
            time.sleep(page_pause)

    limited = collected[: options.get("max_results", DEFAULT_MAX_RESULTS)]
    LOGGER.info("Collected %s search hits.", len(limited))
    return {
        "total_count": total_count,
        "search_items": limited,
        "highlights": highlights,
    }


def enrich_with_metadata(
    client: PressReaderClient,
    search_items: Sequence[Mapping[str, Any]],
    *,
    batch_pause: float = 0.0,
) -> Dict[str, Dict[str, Any]]:
    id_map: Dict[str, Dict[str, Any]] = {}
    ids = [str(item["RegionId"]) for item in search_items if item.get("RegionId")]
    LOGGER.info("Fetching metadata for %s articles...", len(ids))
    for batch in chunked(ids, ARTICLE_BATCH_SIZE):
        LOGGER.debug("Requesting metadata batch of size %s", len(batch))
        try:
            data = client.get_items(batch)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status in {401, 403}:
                LOGGER.warning(
                    "Skipping metadata batch (size=%s) due to HTTP %s.",
                    len(batch),
                    status,
                )
                continue
            LOGGER.error(
                "HTTP error retrieving metadata batch (size=%s): %s",
                len(batch),
                exc,
                exc_info=True,
            )
            continue
        except requests.RequestException as exc:
            LOGGER.error(
                "Request error retrieving metadata batch (size=%s): %s",
                len(batch),
                exc,
                exc_info=True,
            )
            continue
        for article in data.get("Articles", []):
            article_id = str(article.get("Id"))
            if article_id:
                id_map[article_id] = article
        if batch_pause > 0:
            pause = float(batch_pause)
            LOGGER.debug("Sleeping %.2fs between metadata batches.", pause)
            time.sleep(pause)
    LOGGER.info("Retrieved metadata for %s articles.", len(id_map))
    return id_map


def fetch_article_detail(
    client: PressReaderClient,
    article_id: str,
    *,
    detail_pause: float = 0.0,
) -> Optional[Dict[str, Any]]:
    LOGGER.debug("Fetching full article %s", article_id)
    try:
        detail = client.get_article(article_id)
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status in {401, 403}:
            LOGGER.warning("Skipping article %s due to HTTP %s.", article_id, status)
            return None
        LOGGER.error("HTTP error retrieving article %s: %s", article_id, exc, exc_info=True)
        return None
    except requests.RequestException as exc:
        LOGGER.error("Request error retrieving article %s: %s", article_id, exc, exc_info=True)
        return None
    if detail_pause > 0:
        LOGGER.debug("Sleeping %.2fs after article %s.", detail_pause, article_id)
        time.sleep(detail_pause)
    return detail


def build_row(
    item: Mapping[str, Any],
    meta: Mapping[str, Any],
    detail: Optional[Mapping[str, Any]],
    *,
    search_text: str,
    highlights: str,
) -> Dict[str, Any]:
    article_id = str(item.get("RegionId"))
    issue = meta.get("Issue") or {}
    classification = meta.get("Classification") or {}
    tags = [
        tag.get("DisplayName")
        for tag in (classification.get("Tags") or [])
        if isinstance(tag, Mapping) and tag.get("DisplayName")
    ]
    images_source = meta.get("Images") or []
    images = [img.get("Url") for img in images_source if isinstance(img, Mapping) and img.get("Url")]
    related_titles = [
        rel.get("Title")
        for rel in (meta.get("Related") or [])
        if isinstance(rel, Mapping) and rel.get("Title")
    ]
    bylines = [
        bl.get("Text")
        for bl in (meta.get("Bylines") or [])
        if isinstance(bl, Mapping) and bl.get("Text")
    ]
    article_body = article_text(detail.get("Blocks", [])) if detail else ""
    total_text_length = None
    if detail and detail.get("TotalTextLength") is not None:
        total_text_length = detail.get("TotalTextLength")
    elif meta.get("TotalTextLength") is not None:
        total_text_length = meta.get("TotalTextLength")

    return {
        "search_text": search_text,
        "highlights": highlights,
        "region_id": article_id,
        "profile_id": item.get("ProfileId", ""),
        "issue_id": issue.get("Id", ""),
        "issue_title": issue.get("Title", ""),
        "issue_date": issue.get("ShortDateString") or issue.get("Date", ""),
        "issue_country": issue.get("CountryCode", ""),
        "issue_cid": issue.get("CID", ""),
        "page": meta.get("Page", ""),
        "page_name": meta.get("PageName", ""),
        "section": meta.get("Section", ""),
        "title": meta.get("Title", ""),
        "subtitle": meta.get("Subtitle", ""),
        "hyphenated_title": meta.get("HyphenatedTitle", ""),
        "language": meta.get("Language", ""),
        "authors": join_list(bylines),
        "byline": meta.get("Byline", ""),
        "is_top_article": item.get("IsTopArticle", ""),
        "rank": item.get("Rank", ""),
        "rate": meta.get("Rate", ""),
        "similars_count": meta.get("SimilarsCount", ""),
        "current_text_length": meta.get("CurrentTextLength", ""),
        "total_text_length": total_text_length or "",
        "classification_tags": join_list(tags),
        "related_titles": join_list(related_titles),
        "images": join_list(images),
        "article_text": article_body,
    }


def load_existing_rows(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    if not path.exists():
        return [], {}
    try:
        with path.open("r", encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp)
            rows = [row for row in reader]
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to read existing CSV %s: %s", path, exc)
        return [], {}
    row_map: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        region_id = (row.get("region_id") or "").strip()
        if region_id:
            row_map[region_id] = row
    return rows, row_map


def open_csv_writer(path: Path, fieldnames: Sequence[str]) -> Tuple[csv.DictWriter, IO[str]]:
    ensure_parent_dir(path)
    file_has_data = path.exists() and path.stat().st_size > 0
    fp = path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(fp, fieldnames=fieldnames)
    if not file_has_data:
        writer.writeheader()
        fp.flush()
    return writer, fp


def rewrite_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")

    config_path = Path(args.config).expanduser() if args.config else None
    if not config_path and Path("pressreader_config.json").exists():
        config_path = Path("pressreader_config.json")
    config = load_config(config_path) if config_path else {}

    options = {
        "auth_token": resolve_option("auth_token", args=vars(args), config=config, default=None),
        "search_text": resolve_option("search_text", args=vars(args), config=config, default=None),
        "languages": resolve_option("languages", args=vars(args), config=config, default="en"),
        "countries": resolve_option("countries", args=vars(args), config=config, default=""),
        "categories": resolve_option("categories", args=vars(args), config=config, default=""),
        "newspapers": resolve_option("newspapers", args=vars(args), config=config, default=""),
        "search_author": resolve_option("search_author", args=vars(args), config=config, default=""),
        "search_in": resolve_option("search_in", args=vars(args), config=config, default="ALL"),
        "search_for": resolve_option("search_for", args=vars(args), config=config, default="All"),
        "order_by": resolve_option("order_by", args=vars(args), config=config, default="Relevance"),
        "range": resolve_option("range_", args=vars(args), config=config, config_key="range", default="Range"),
        "start_date": resolve_option("start_date", args=vars(args), config=config, default=""),
        "stop_date": resolve_option("stop_date", args=vars(args), config=config, default=""),
        "page_size": int(resolve_option("page_size", args=vars(args), config=config, default=DEFAULT_PAGE_SIZE)),
        "max_results": int(resolve_option("max_results", args=vars(args), config=config, default=DEFAULT_MAX_RESULTS)),
        "output": resolve_option("output", args=vars(args), config=config, default="output/pressreader_articles.csv"),
        "hide_same": int(config.get("hide_same", 0)),
        "page_pause": float(resolve_option("page_pause", args=vars(args), config=config, default=DEFAULT_PAGE_PAUSE)),
        "batch_pause": float(resolve_option("batch_pause", args=vars(args), config=config, default=DEFAULT_BATCH_PAUSE)),
        "detail_pause": float(resolve_option("detail_pause", args=vars(args), config=config, default=DEFAULT_DETAIL_PAUSE)),
    }

    auth_token = options["auth_token"]
    if not auth_token:
        LOGGER.error("An auth token is required (--auth-token or config).")
        return 2
    if not options["search_text"]:
        LOGGER.error("A search query is required (--search-text or config).")
        return 2

    delay = resolve_option("delay", args=vars(args), config=config, default=DEFAULT_DELAY)
    jitter = resolve_option("jitter", args=vars(args), config=config, default=DEFAULT_JITTER)
    timeout = resolve_option("timeout", args=vars(args), config=config, default=30.0)

    client = PressReaderClient(
        auth_token,
        delay=float(delay),
        jitter=float(jitter),
        timeout=float(timeout),
        user_agent=config.get("user_agent", DEFAULT_USER_AGENT),
    )

    output_path = Path(options["output"]).expanduser()
    existing_rows, existing_row_map = load_existing_rows(output_path)
    completed_ids = {
        region_id
        for region_id, row in existing_row_map.items()
        if (row.get("article_text") or "").strip()
    }
    LOGGER.info(
        "Loaded %s rows (%s with article detail) from %s",
        len(existing_rows),
        len(completed_ids),
        output_path,
    )

    try:
        search_data = collect_articles(client, options)
        search_items = search_data["search_items"]
        highlights = join_list(search_data.get("highlights"), sep="|")
        search_text = options.get("search_text", "")

        new_items = [
            item
            for item in search_items
            if str(item.get("RegionId")) and str(item.get("RegionId")) not in completed_ids
        ]
        if not new_items:
            LOGGER.info("No new articles to process. Exiting.")
            return 0

        metadata = enrich_with_metadata(
            client,
            new_items,
            batch_pause=options.get("batch_pause", 0.0),
        )

        writer, fp = open_csv_writer(output_path, CSV_FIELDNAMES)
        processed_new = 0
        updated_existing = 0
        need_rewrite = False
        processed_ids: Set[str] = set()
        try:
            for item in new_items:
                article_id = str(item.get("RegionId"))
                if not article_id:
                    continue
                if article_id in processed_ids:
                    continue
                processed_ids.add(article_id)
                meta = metadata.get(article_id)
                if not meta:
                    LOGGER.warning("Skipping article %s due to missing metadata.", article_id)
                    continue
                detail = fetch_article_detail(
                    client,
                    article_id,
                    detail_pause=options.get("detail_pause", 0.0),
                )
                if detail is None:
                    LOGGER.warning("Article %s has no detail data; writing metadata only.", article_id)
                row = build_row(
                    item,
                    meta,
                    detail,
                    search_text=search_text,
                    highlights=highlights,
                )
                if article_id in existing_row_map and (existing_row_map[article_id].get("article_text") or "").strip() == "":
                    existing_row_map[article_id].update(row)
                    need_rewrite = True
                    updated_existing += 1
                    if row.get("article_text", "").strip():
                        completed_ids.add(article_id)
                elif article_id in existing_row_map:
                    # Row already complete but requested again (e.g., metadata refreshed).
                    existing_row_map[article_id].update(row)
                    need_rewrite = True
                    updated_existing += 1
                    if row.get("article_text", "").strip():
                        completed_ids.add(article_id)
                else:
                    writer.writerow(row)
                    fp.flush()
                    existing_rows.append(row)
                    existing_row_map[article_id] = row
                    processed_new += 1
                    if row.get("article_text", "").strip():
                        completed_ids.add(article_id)
        finally:
            fp.close()

        if need_rewrite:
            rewrite_csv(output_path, existing_rows)

        LOGGER.info(
            "Appended %s new articles and updated %s existing articles in %s",
            processed_new,
            updated_existing,
            output_path,
        )
    except requests.HTTPError as exc:
        LOGGER.error("HTTP error: %s", exc, exc_info=True)
        return 1
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.error("Unhandled error: %s", exc, exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
