#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DISCLAIMER:
This script is provided for educational purposes only. LinkedIn's terms of service prohibit
scraping or any form of automated data collection. Using this script to scrape LinkedIn's data
is against their terms of service and can result in your account being banned.
Use this script at your own risk. The author is not responsible for any misuse of this script.
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode, quote_plus

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

from bs4 import BeautifulSoup as bs

from job_extraction import JobInsightExtractor

# ---------------------------------------------------------------------------------------
# Function to load cookies from a Netscape-format cookies.txt file into Selenium's browser
# ---------------------------------------------------------------------------------------
def load_cookies(browser, file_path: Path) -> None:
    if not file_path.exists():
        raise FileNotFoundError(f"Cookies file not found: {file_path}")

    content = file_path.read_text(encoding='utf-8').lstrip()
    if not content:
        raise ValueError(f"Cookies file {file_path} is empty.")

    # JSON export (e.g. EditThisCookie or Chrome dev-tools)
    if content.startswith('['):
        cookies = json.loads(content)
        added = 0
        for raw_cookie in cookies:
            name = raw_cookie.get('name')
            value = raw_cookie.get('value')
            if not name or value is None:
                continue

            cookie = {
                'name': name,
                'value': value,
                'path': raw_cookie.get('path', '/'),
            }

            domain = raw_cookie.get('domain')
            if domain:
                cookie['domain'] = domain

            expiry = raw_cookie.get('expirationDate') or raw_cookie.get('expiry')
            if expiry:
                try:
                    cookie['expiry'] = int(expiry)
                except (TypeError, ValueError):
                    pass

            secure = raw_cookie.get('secure')
            if secure is not None:
                cookie['secure'] = bool(secure)

            http_only = raw_cookie.get('httpOnly')
            if http_only is not None:
                cookie['httpOnly'] = bool(http_only)

            browser.add_cookie(cookie)
            added += 1

        if not added:
            raise ValueError(f"No valid cookies found in JSON file {file_path}.")
        return

    # Netscape format
    added = 0
    for line in content.splitlines():
        if line.startswith('#') or not line.strip():
            continue

        fields = line.strip().split('\t')
        if len(fields) != 7:
            continue

        domain, flag, path, secure, expiration, name, value = fields

        cookie = {
            'name': name,
            'value': value,
            'domain': domain,
            'path': path,
            'secure': secure.lower() == 'true',
        }

        if expiration.isdigit():
            cookie['expiry'] = int(expiration)

        browser.add_cookie(cookie)
        added += 1

    if not added:
        raise ValueError(f"No valid cookies parsed from Netscape file {file_path}.")

# ---------------------------------------------------------------------------------------
# Helper function to convert abbreviated reaction/comment strings (e.g., "1K") to integers
# ---------------------------------------------------------------------------------------
def convert_abbreviated_to_number(s):
    s = s.upper().strip()
    if 'K' in s:
        return int(float(s.replace('K', '')) * 1000)
    elif 'M' in s:
        return int(float(s.replace('M', '')) * 1000000)
    else:
        # If it's just a normal number or empty, attempt to parse it
        try:
            return int(s)
        except ValueError:
            return 0


def build_content_search_url(
    keywords: str,
    sort_order: str = "recent",
    time_filter: Optional[str] = None,
) -> str:
    """
    Build a LinkedIn search URL that returns public posts containing the given keywords.

    Parameters
    ----------
    keywords:
        Search terms exactly as you would type them in LinkedIn's search box.
    sort_order:
        Either "recent" (default) or "relevance". LinkedIn translates this into sortBy params.
    time_filter:
        Optional LinkedIn time filter code (e.g. "r86400" = past 24 hours, "r604800" = past week).
        Leave as None to include all time ranges.
    """
    params = {
        "keywords": keywords,
        "origin": "GLOBAL_SEARCH_HEADER",
    }
    if sort_order == "recent":
        params["sortBy"] = "date"
    elif sort_order == "relevance":
        params["sortBy"] = "relevance"

    if time_filter:
        params["f_TPR"] = time_filter

    return f"https://www.linkedin.com/search/results/content/?{urlencode(params, quote_via=quote_plus)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect LinkedIn posts from a profile feed or keyword search result page."
    )
    parser.add_argument(
        "--config",
        help="Optional path to a JSON configuration file (default: ./scrape_config.json if present).",
    )
    parser.add_argument(
        "--target-url",
        default=None,
        help=(
            "Full LinkedIn URL to scrape (profile activity feed or search results). "
            "If omitted, you must pass --keywords to construct a search URL automatically."
        ),
    )
    parser.add_argument(
        "--keywords",
        default=None,
        help=(
            "Keyword phrase to search for in LinkedIn posts (e.g. \"AI engineer\"). "
            "When supplied, the script ignores --target-url and builds a content search URL."
        ),
    )
    parser.add_argument(
        "--keyword-sort",
        choices=("recent", "relevance"),
        default=None,
        help="Sort order to use when --keywords is provided (default: recent).",
    )
    parser.add_argument(
        "--keyword-time-filter",
        default=None,
        help=(
            "LinkedIn time filter code for keyword search (e.g. r86400=past 24h, r604800=past week). "
            "Only used when --keywords is supplied."
        ),
    )
    parser.add_argument(
        "--cookies",
        default=None,
        help="Path to a Netscape-format cookies.txt file exported after logging into LinkedIn.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Destination CSV file (default: user_posts_extended.csv).",
    )
    parser.add_argument(
        "--max-posts",
        type=int,
        default=None,
        help="Total number of unique posts to collect before stopping (default: 20).",
    )
    parser.add_argument(
        "--scroll-attempts",
        type=int,
        default=None,
        help="Maximum number of times to scroll the page looking for new posts (default: 40).",
    )
    parser.add_argument(
        "--max-no-new-posts",
        type=int,
        default=None,
        help="Stop after this many consecutive scrolls without discovering new posts (default: 3).",
    )
    parser.add_argument(
        "--scroll-pause",
        type=float,
        default=None,
        help="Seconds to wait after each scroll before re-parsing the page (default: 4).",
    )
    parser.add_argument(
        "--chrome-binary",
        default=None,
        help=(
            "Absolute path to the Chrome binary. "
            "For the downloaded Chrome for Testing build this is typically "
            "\"/Users/<you>/apify/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing\"."
        ),
    )
    parser.add_argument(
        "--chromedriver",
        default=None,
        help=(
            "Absolute path to the chromedriver executable. "
            "If omitted, Selenium searches your PATH. Make sure the driver version matches Chrome."
        ),
    )
    parser.add_argument(
        "--show-browser",
        action="store_true",
        help="Run Chrome with a visible window instead of headless mode.",
    )
    parser.add_argument(
        "--disable-gpu",
        action="store_true",
        help="Add the --disable-gpu flag to Chrome (useful on some environments).",
    )
    parser.add_argument(
        "--llm-mode",
        choices=("none", "local", "heuristic", "openai"),
        default=None,
        help="Post-processing mode for extracting job insights from post content.",
    )
    parser.add_argument(
        "--openai-api-key",
        default=None,
        help="OpenAI API key used when --llm-mode=openai. Defaults to OPENAI_API_KEY env var.",
    )
    parser.add_argument(
        "--openai-model",
        default=None,
        help="OpenAI model for extraction when using --llm-mode=openai (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--openai-base-url",
        default=None,
        help="Base URL for OpenAI-compatible API (default: https://api.openai.com/v1).",
    )
    parser.add_argument(
        "--openai-batch-size",
        type=int,
        default=None,
        help="Maximum number of posts per OpenAI request batch (default: 5).",
    )
    parser.add_argument(
        "--openai-rpm",
        type=float,
        default=None,
        help="Soft requests-per-minute limit when calling OpenAI (default: 8).",
    )
    parser.add_argument(
        "--openai-max-retries",
        type=int,
        default=None,
        help="Maximum retry attempts for OpenAI calls (default: 5).",
    )
    parser.add_argument(
        "--openai-backoff-seconds",
        type=float,
        default=None,
        help="Initial backoff in seconds for OpenAI retries (default: 3).",
    )
    return parser.parse_args()


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def apply_config(args: argparse.Namespace) -> argparse.Namespace:
    config_path = Path(args.config).expanduser() if args.config else BASE_DIR / "scrape_config.json"
    config: Dict[str, Any] = {}
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            print(f"[*] Loaded configuration from {config_path}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in config file {config_path}: {exc}") from exc
    elif args.config:
        raise FileNotFoundError(f"Specified config file not found: {config_path}")

    def get_value(attr: str, default: Any = None, cast=None):
        current = getattr(args, attr, None)
        if current is not None:
            return current
        if attr in config and config[attr] is not None:
            value = config[attr]
        else:
            value = default
        if cast and value is not None:
            try:
                value = cast(value)
            except (TypeError, ValueError):
                raise ValueError(f"Invalid value for {attr}: {value!r}")
        return value

    args.target_url = get_value("target_url")
    args.keywords = get_value("keywords")
    args.keyword_sort = get_value("keyword_sort", default="recent")
    args.keyword_time_filter = get_value("keyword_time_filter")
    args.cookies = get_value("cookies", cast=lambda v: str(v))
    args.output = get_value("output", default="user_posts_extended.csv")
    args.max_posts = get_value("max_posts", default=20, cast=int)
    args.scroll_attempts = get_value("scroll_attempts", default=40, cast=int)
    args.max_no_new_posts = get_value("max_no_new_posts", default=3, cast=int)
    args.scroll_pause = get_value("scroll_pause", default=4.0, cast=float)
    args.chrome_binary = get_value("chrome_binary")
    args.chromedriver = get_value("chromedriver")

    if args.show_browser is False and "show_browser" in config:
        args.show_browser = _coerce_bool(config["show_browser"])
    if args.disable_gpu is False and "disable_gpu" in config:
        args.disable_gpu = _coerce_bool(config["disable_gpu"])

    args.llm_mode = (get_value("llm_mode", default="none") or "none").lower()
    args.openai_api_key = get_value("openai_api_key") or os.environ.get("OPENAI_API_KEY")
    args.openai_model = get_value("openai_model", default="gpt-4o-mini")
    args.openai_base_url = get_value("openai_base_url", default="https://api.openai.com/v1")
    args.openai_batch_size = get_value("openai_batch_size", default=5, cast=int)
    args.openai_rpm = get_value("openai_rpm", default=8.0, cast=float)
    args.openai_max_retries = get_value("openai_max_retries", default=5, cast=int)
    args.openai_backoff_seconds = get_value("openai_backoff_seconds", default=3.0, cast=float)

    return args

# ---------------------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DEFAULT_CHROME_BINARY = Path("/Applications/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing")
DEFAULT_CHROMEDRIVER = None


def main():
    args = parse_args()
    args = apply_config(args)

    if not args.cookies:
        raise SystemExit("Cookies file path must be provided via --cookies or config file.")

    if args.keywords:
        target_url = build_content_search_url(
            keywords=args.keywords,
            sort_order=args.keyword_sort,
            time_filter=args.keyword_time_filter,
        )
        print(f"[*] Generated LinkedIn search URL for keywords '{args.keywords}': {target_url}")
    else:
        target_url = args.target_url

    if not target_url:
        raise SystemExit("You must provide either --target-url or --keywords.")

    cookies_path = Path(args.cookies).expanduser()
    csv_path = Path(args.output).expanduser()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    max_posts = args.max_posts
    max_scroll_attempts = args.scroll_attempts
    max_no_new_posts = args.max_no_new_posts
    load_pause_time = max(args.scroll_pause, 1.0)

    chrome_binary = Path(args.chrome_binary).expanduser() if args.chrome_binary else DEFAULT_CHROME_BINARY
    if chrome_binary and not chrome_binary.exists():
        print(f"[!] Chrome binary not found at {chrome_binary}. Falling back to system Chrome.")
        chrome_binary = None

    chromedriver_path = Path(args.chromedriver).expanduser() if args.chromedriver else DEFAULT_CHROMEDRIVER
    if chromedriver_path and not chromedriver_path.exists():
        print(f"[!] chromedriver not found at {chromedriver_path}. Selenium will look on PATH instead.")
        chromedriver_path = None

    chrome_options = Options()
    if not args.show_browser:
        chrome_options.add_argument('--headless=new')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_argument('--disable-infobars')
    if args.disable_gpu:
        chrome_options.add_argument('--disable-gpu')
    if chrome_binary:
        chrome_options.binary_location = str(chrome_binary)

    if chromedriver_path:
        service = Service(str(chromedriver_path))
        browser = webdriver.Chrome(service=service, options=chrome_options)
    else:
        browser = webdriver.Chrome(options=chrome_options)

    browser.set_window_size(1920, 1080)
    
    # --------------------------------
    # Log in by loading cookies
    # --------------------------------
    print(f"[*] Going to LinkedIn home page and loading cookies from {cookies_path} ...")
    browser.get('https://www.linkedin.com/')
    time.sleep(2)
    
    # Load cookies
    load_cookies(browser, cookies_path)
    
    # Refresh to apply cookies
    browser.refresh()
    print("[*] Cookies loaded; refreshing page to apply them...")
    
    # Ensure page is loaded
    try:
        WebDriverWait(browser, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "#global-nav"))
        )
        print("[*] Successfully logged in (navigation bar found).")
    except TimeoutException:
        print("[!] Navigation bar not found after applying cookies. Exiting.")
        browser.quit()
        return
    
    # --------------------------------
    # Navigate to the desired profile
    # --------------------------------
    print(f"[*] Navigating to {target_url} ...")
    browser.get(target_url)
    time.sleep(5)  # Let the page load
    
    records: List[Dict[str, Any]] = []
    # Use a set to track post IDs to avoid duplicates
    unique_post_ids = set()
    post_count = 0
    
    # Scroll parameters
    scroll_attempts = 0
    no_new_posts_count = 0

    print("[*] Starting to scroll and collect post data...")
    while (
        post_count < max_posts
        and scroll_attempts < max_scroll_attempts
        and no_new_posts_count < max_no_new_posts
    ):
        
        soup = bs(browser.page_source, "html.parser")
        
        # Each post is generally in an element: <div class="feed-shared-update-v2 ..."/>
        post_wrappers = soup.find_all("div", {"class": "feed-shared-update-v2"})
        
        new_posts_in_this_pass = 0  # track how many brand-new posts we discovered in this pass
        
        for pw in post_wrappers:
            # ---
            # 1) Post ID & Post URL
            # ---
            post_id = None
            post_url = None
            
            detail_link_tag = pw.find("a", {"class": "update-components-mini-update-v2__link-to-details-page"})
            if detail_link_tag and detail_link_tag.get("href"):
                post_url = detail_link_tag["href"].strip()
                if "urn:li:activity:" in post_url:
                    part = post_url.split("urn:li:activity:")[-1].replace("/", "")
                    post_id = part
            
            # Also check data-urn
            if not post_id:
                data_urn = pw.get("data-urn", "")
                if "urn:li:activity:" in data_urn:
                    post_id = data_urn.split("urn:li:activity:")[-1]
            
            # If we still can't find ID, skip
            if not post_id:
                continue
            
            # If we already have this post in our set, skip it
            if post_id in unique_post_ids:
                continue
            
            # Mark it as new
            unique_post_ids.add(post_id)
            new_posts_in_this_pass += 1

            # Convert relative URL to absolute
            if post_url and post_url.startswith("/feed/update/"):
                post_url = "https://www.linkedin.com" + post_url
            
            # ---
            # 2) Post Author name, profile link, job title, posted time
            # ---
            author_name = None
            author_profile_link = None
            author_jobtitle = None
            post_time = None
            
            actor_container = pw.find("div", {"class": "update-components-actor__container"})
            if actor_container:
                # Author name
                name_tag = actor_container.find("span", {"class": "update-components-actor__title"})
                if name_tag:
                    inner_span = name_tag.find("span", {"dir": "ltr"})
                    if inner_span:
                        author_name = inner_span.get_text(strip=True)
                
                # Profile link
                actor_link = actor_container.find("a", {"class": "update-components-actor__meta-link"})
                if actor_link and actor_link.get("href"):
                    author_profile_link = actor_link["href"].strip()
                    if author_profile_link.startswith("/in/"):
                        author_profile_link = "https://www.linkedin.com" + author_profile_link
                
                # Job title
                jobtitle_tag = actor_container.find("span", {"class": "update-components-actor__description"})
                if jobtitle_tag:
                    author_jobtitle = jobtitle_tag.get_text(strip=True)
                
                # Time posted
                time_tag = actor_container.find("span", {"class": "update-components-actor__sub-description"})
                if time_tag:
                    post_time = time_tag.get_text(strip=True)
            
            # ---
            # 3) Post content
            # ---
            post_content = None
            content_div = pw.find("div", {"class": "update-components-text"})
            if content_div:
                post_content = content_div.get_text(separator="\n", strip=True)
            
            # ---
            # 4) Reactions, Comments, Impressions
            # ---
            post_reactions = 0
            post_comments = 0
            post_impressions = 0
            
            social_counts_div = pw.find("div", {"class": "social-details-social-counts"})
            if social_counts_div:
                # Reactions
                reaction_item = social_counts_div.find("li", {"class": "social-details-social-counts__reactions"})
                if reaction_item:
                    button_tag = reaction_item.find("button")
                    if button_tag and button_tag.has_attr("aria-label"):
                        raw_reactions = button_tag["aria-label"].split(" ")[0]
                        post_reactions = convert_abbreviated_to_number(raw_reactions)
                
                # Comments
                comment_item = social_counts_div.find("li", {"class": "social-details-social-counts__comments"})
                if comment_item:
                    cbutton_tag = comment_item.find("button")
                    if cbutton_tag and cbutton_tag.has_attr("aria-label"):
                        raw_comments = cbutton_tag["aria-label"].split(" ")[0]
                        post_comments = convert_abbreviated_to_number(raw_comments)
            
            # Impressions
            impressions_span = pw.find("span", {"class": "analytics-entry-point"})
            if impressions_span:
                possible_text = impressions_span.get_text(strip=True)
                if "impressions" in possible_text.lower():
                    raw_impressions = possible_text.lower().replace("impressions", "").strip()
                    raw_impressions = raw_impressions.split(" ")[0]
                    post_impressions = convert_abbreviated_to_number(raw_impressions)
            
            date_collected = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            
            # Print for debugging
            print(f"[+] Found new Post ID {post_id}. So far we have {post_count + 1} unique posts.")
            print(f"    URL: {post_url}")
            print(f"    Author: {author_name} | {author_profile_link}")
            snippet = (post_content or "")[:70]
            ellipsis = "..." if len(post_content or "") > 70 else ""
            print(f"    Content snippet: {snippet}{ellipsis}")
            
            records.append({
                "Post_ID": post_id or "",
                "Post_URL": post_url or "",
                "Post_Author_Name": author_name or "",
                "Post_Author_Profile": author_profile_link or "",
                "Post_Author_JobTitle": author_jobtitle or "",
                "Post_Time": post_time or "",
                "Post_Content": post_content or "",
                "Post_Reactions": post_reactions,
                "Post_Comments": post_comments,
                "Post_Impressions": post_impressions,
                "Date_Collected": date_collected,
            })

            # Increase final count
            post_count += 1
            if post_count >= max_posts:
                break
        
        # If we found no new posts in this pass, increment no_new_posts_count
        # otherwise reset it
        if new_posts_in_this_pass == 0:
            no_new_posts_count += 1
        else:
            no_new_posts_count = 0
        
        # Scroll further only if we haven't reached MAX_POSTS
        if post_count < max_posts:
            print("[*] Scrolling to load more posts...")
            browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(load_pause_time)
            scroll_attempts += 1
    
    print(f"[*] Finished after collecting {post_count} unique posts.")
    print("[*] Closing browser.")
    browser.quit()

    if records:
        llm_mode = (args.llm_mode or "none").lower()
        if llm_mode == "heuristic":
            llm_mode = "local"

        if llm_mode == "openai" and not args.openai_api_key:
            raise ValueError("OpenAI API key is required when llm_mode is 'openai'. Provide it via config or --openai-api-key.")

        if llm_mode == "none":
            insights = [
                {"job_titles": [], "responsibilities": [], "skills": [], "requirements": []}
                for _ in records
            ]
        else:
            extractor = JobInsightExtractor(
                mode=llm_mode,
                openai_api_key=args.openai_api_key,
                openai_model=args.openai_model,
                openai_base_url=args.openai_base_url,
                openai_requests_per_minute=args.openai_rpm,
                openai_batch_size=args.openai_batch_size,
                openai_max_retries=args.openai_max_retries,
                openai_backoff_seconds=args.openai_backoff_seconds,
            )
            contents = [record["Post_Content"] for record in records]
            insights = extractor.extract_many(contents)
    else:
        insights = []

    headers = [
        "Post_ID",
        "Post_URL",
        "Post_Author_Name",
        "Post_Author_Profile",
        "Post_Author_JobTitle",
        "Post_Time",
        "Post_Content",
        "Post_Reactions",
        "Post_Comments",
        "Post_Impressions",
        "Date_Collected",
        "Job_Titles",
        "Job_Responsibilities",
        "Job_Skills",
        "Job_Requirements",
    ]

    with csv_path.open(mode='w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        empty_insight = {"job_titles": [], "responsibilities": [], "skills": [], "requirements": []}
        for idx, record in enumerate(records):
            insight = insights[idx] if idx < len(insights) else empty_insight
            writer.writerow([
                record["Post_ID"],
                record["Post_URL"],
                record["Post_Author_Name"],
                record["Post_Author_Profile"],
                record["Post_Author_JobTitle"],
                record["Post_Time"],
                record["Post_Content"],
                record["Post_Reactions"],
                record["Post_Comments"],
                record["Post_Impressions"],
                record["Date_Collected"],
                "; ".join(insight.get("job_titles", [])),
                "; ".join(insight.get("responsibilities", [])),
                "; ".join(insight.get("skills", [])),
                "; ".join(insight.get("requirements", [])),
            ])

    print(f"[*] Data saved to {csv_path}")

# ---------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
