#!/usr/bin/env python3
"""Fill missing PressReader article_text by refetching details with a fresh token."""

from __future__ import annotations

import argparse
import csv
import logging
import re
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

from pressreader_scraper import PressReaderClient, article_text, load_config


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refetch PressReader article bodies for rows with article_url but empty article_text."
    )
    parser.add_argument("--config", default="pressreader_config.json", help="Path to config JSON with auth_token.")
    parser.add_argument("--auth-token", default=None, help="Optional PressReader auth token to override config.")
    parser.add_argument("--input", required=True, help="Input CSV path (pressreader output).")
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (defaults to overwriting the input file).",
    )
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between detail requests.")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout for PressReader requests.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO).")
    return parser.parse_args(argv)


def extract_article_id(row: Dict[str, str]) -> Optional[str]:
    region_id = (row.get("region_id") or "").strip()
    if region_id:
        return region_id
    url = (row.get("article_url") or "").strip()
    if not url:
        return None
    match = re.search(r"/article/(\d+)", url)
    if match:
        return match.group(1)
    return None


def fetch_article_body(client: PressReaderClient, article_id: str, timeout_pause: float) -> Optional[str]:
    try:
        detail = client.get_article(article_id)
    except Exception as exc:  # noqa: BLE001
        logging.warning("Failed to fetch article %s: %s", article_id, exc)
        return None
    if timeout_pause > 0:
        time.sleep(timeout_pause)
    body = article_text(detail.get("Blocks", [])) if isinstance(detail, dict) else ""
    return body or None


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    config_path = Path(args.config)
    config = load_config(config_path) if config_path.exists() else {}
    auth_token = args.auth_token or config.get("auth_token")
    if not auth_token:
        logging.error("Missing auth token. Provide --auth-token or set auth_token in config.")
        return 1

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path

    client = PressReaderClient(
        auth_token,
        delay=max(args.delay, 0.0),
        timeout=args.timeout,
    )

    temp_path = output_path if args.output else input_path.with_suffix(input_path.suffix + ".tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8", newline="") as inp, temp_path.open(
        "w", encoding="utf-8", newline=""
    ) as out_fp:
        reader = csv.DictReader(inp)
        fieldnames = reader.fieldnames
        if not fieldnames:
            logging.error("Input CSV %s has no headers.", input_path)
            return 1
        writer = csv.DictWriter(out_fp, fieldnames=fieldnames)
        writer.writeheader()

        updated = 0
        skipped = 0
        for row in reader:
            article_text_value = (row.get("article_text") or "").strip()
            url = (row.get("article_url") or "").strip()
            if not article_text_value and url:
                article_id = extract_article_id(row)
                if article_id:
                    body = fetch_article_body(client, article_id, args.delay)
                    if body:
                        row["article_text"] = body
                        updated += 1
                    else:
                        skipped += 1
                else:
                    skipped += 1
            writer.writerow(row)
            out_fp.flush()

    if args.output:
        final_path = output_path
    else:
        temp_path.replace(input_path)
        final_path = input_path
    logging.info(
        "Completed fill operation. Updated %s rows, %s rows could not be filled. Output: %s",
        updated,
        skipped,
        final_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
