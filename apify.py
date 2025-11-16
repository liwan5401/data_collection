"""Command-line helper for querying LinkedIn jobs via Apify."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import requests
except ImportError:
    requests = None  # type: ignore
from apify_client import ApifyClient

DEFAULT_ACTOR_ID = "BHzefUZlZRKWxkTck"
DEFAULT_TITLE_QUERY = (
    '(("AI" OR "artificial intelligence" OR "machine learning" OR '
    '"deep learning" OR "generative AI" OR "AI strategy" OR "AI governance"))'
)
DEFAULT_LOCATION = "United States"
DEFAULT_ROWS = 50
DEFAULT_TABLE_LIMIT = 20
DEFAULT_CSV_PATH = "jobs.csv"
DEFAULT_OPENAI_MODEL = "gpt-5o-mini"
DEFAULT_OPENAI_RPM = 10
DEFAULT_OPENAI_MAX_RETRIES = 5
DEFAULT_OPENAI_BACKOFF_SECONDS = 3.0
DEFAULT_OPENAI_BATCH_SIZE = 5
RESPONSIBILITY_HEADINGS = (
    "responsibilities",
    "key responsibilities",
    "responsibility",
    "what you'll do",
    "what you will do",
    "what you do",
    "your role",
    "岗位职责",
)
SKILL_HEADINGS = (
    "skills",
    "required skills",
    "desired skills",
    "skills & qualifications",
    "skill requirements",
    "岗位技能",
)
REQUIREMENT_HEADINGS = (
    "requirements",
    "qualifications",
    "must have",
    "nice to have",
    "experience",
    "岗位要求",
)

SECTION_MAP = {
    "responsibilities": RESPONSIBILITY_HEADINGS,
    "skills": SKILL_HEADINGS,
    "requirements": REQUIREMENT_HEADINGS,
}

SKILL_PREFIXES = (
    "experience with",
    "experience in",
    "experience using",
    "experience working with",
    "experience working in",
    "knowledge of",
    "knowledge in",
    "knowledge with",
    "proficiency in",
    "proficiency with",
    "proficiency using",
    "expertise in",
    "expertise with",
    "familiarity with",
    "familiarity in",
    "ability to",
    "capability to",
    "responsible for",
    "strong knowledge of",
    "strong understanding of",
    "hands-on experience with",
    "hands-on experience in",
)

SKILL_SUFFIXES = (
    " skills",
    " skill",
    " expertise",
)

MAX_SKILL_WORDS = 8
MAX_SKILL_CHARS = 80
MAX_DESCRIPTION_CHARS = 4000

LOCAL_SKILL_PHRASES = (
    "machine learning",
    "deep learning",
    "data science",
    "data analysis",
    "statistical modeling",
    "predictive analytics",
    "natural language processing",
    "computer vision",
    "cloud computing",
    "distributed systems",
    "big data",
    "project management",
    "product management",
    "stakeholder management",
    "requirements gathering",
    "user research",
    "business intelligence",
    "performance optimization",
    "process automation",
    "risk management",
    "quality assurance",
    "change management",
    "test automation",
    "continuous integration",
    "continuous delivery",
    "model deployment",
    "microservices architecture",
    "data visualization",
    "feature engineering",
    "security compliance",
    "incident management",
    "capacity planning",
    "workflow automation",
    "talent development",
    "go-to-market strategy",
    "customer success",
    "vendor management",
)

LOCAL_SKILL_SINGLE_WORDS = {
    "python": "Python",
    "java": "Java",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "golang": "Go",
    "go": "Go",
    "sql": "SQL",
    "nosql": "NoSQL",
    "hadoop": "Hadoop",
    "spark": "Spark",
    "kafka": "Kafka",
    "airflow": "Airflow",
    "aws": "AWS",
    "azure": "Azure",
    "gcp": "GCP",
    "docker": "Docker",
    "kubernetes": "Kubernetes",
    "terraform": "Terraform",
    "ansible": "Ansible",
    "linux": "Linux",
    "git": "Git",
    "jira": "Jira",
    "figma": "Figma",
    "photoshop": "Photoshop",
    "tableau": "Tableau",
    "powerbi": "Power BI",
    "excel": "Excel",
    "snowflake": "Snowflake",
    "redshift": "Redshift",
    "postgresql": "PostgreSQL",
    "mysql": "MySQL",
    "oracle": "Oracle",
    "sap": "SAP",
    "salesforce": "Salesforce",
    "servicenow": "ServiceNow",
    "scala": "Scala",
    "r": "R",
    "matlab": "MATLAB",
    "pytorch": "PyTorch",
    "tensorflow": "TensorFlow",
    "keras": "Keras",
    "scikitlearn": "scikit-learn",
    "sparkml": "Spark ML",
    "html": "HTML",
    "css": "CSS",
    "rest": "REST",
    "graphql": "GraphQL",
    "ci": "CI",
    "cd": "CD",
    "mlops": "MLOps",
    "devops": "DevOps",
    "nlp": "NLP",
    "ai": "AI",
    "ml": "ML",
    "ui": "UI",
    "ux": "UX",
}

SkillTask = Tuple[str, List[str]]


@dataclass
class JobRecord:
    search_query: str
    title: str
    company: str
    location: str
    published_at: str
    url: str
    responsibilities: List[str]
    skills: List[str]
    requirements: List[str]
    description: str


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search LinkedIn jobs via Apify and export structured results."
    )
    parser.add_argument(
        "--config",
        help="Path to a JSON configuration file with credentials and search parameters.",
    )
    parser.add_argument(
        "--title",
        dest="titles",
        action="append",
        help=(
            "Job title search query (boolean expressions allowed). Can be supplied multiple times. "
            "If omitted, a future-of-work themed query is used."
        ),
    )
    parser.add_argument(
        "--title-file",
        help="Path to a text file containing additional title queries (one per line).",
    )
    parser.add_argument(
        "--location",
        default=DEFAULT_LOCATION,
        help="Location filter for the Apify actor (e.g. 'United States').",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=DEFAULT_ROWS,
        help=f"Maximum number of rows requested from Apify per title query (default: {DEFAULT_ROWS}).",
    )
    parser.add_argument(
        "--company-name",
        dest="company_names",
        action="append",
        default=None,
        help="Company name filter (can be supplied multiple times).",
    )
    parser.add_argument(
        "--company-id",
        dest="company_ids",
        action="append",
        default=None,
        help="Company ID filter (repeatable). Provide numeric LinkedIn company IDs.",
    )
    parser.add_argument(
        "--published-after",
        dest="published_after",
        default="",
        help="Only fetch jobs published after the given ISO timestamp.",
    )
    parser.add_argument(
        "--work-type",
        choices=("1", "2", "3"),
        help="LinkedIn work arrangement filter (1=On-site, 2=Hybrid, 3=Remote).",
    )
    parser.add_argument(
        "--contract-type",
        choices=("F", "P", "C", "T", "I", "V"),
        help="LinkedIn contract type filter (F=Full-time, P=Part-time, etc.).",
    )
    parser.add_argument(
        "--experience-level",
        choices=("1", "2", "3", "4", "5"),
        help="LinkedIn experience level filter (1=Internship, 5=Director).",
    )
    parser.add_argument(
        "--actor-id",
        default=DEFAULT_ACTOR_ID,
        help=f"Apify actor ID to invoke (default: {DEFAULT_ACTOR_ID}).",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("APIFY_TOKEN"),
        help="Apify API token. Defaults to the APIFY_TOKEN environment variable.",
    )
    parser.add_argument(
        "--csv-path",
        default=DEFAULT_CSV_PATH,
        help=f"Path for the CSV export (default: {DEFAULT_CSV_PATH}).",
    )
    parser.add_argument(
        "--table-limit",
        type=int,
        default=DEFAULT_TABLE_LIMIT,
        help=(
            "Maximum number of rows to display in the console table "
            f"(default: {DEFAULT_TABLE_LIMIT})."
        ),
    )
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="Disable Apify proxy usage when invoking the actor.",
    )
    parser.add_argument(
        "--proxy-group",
        dest="proxy_groups",
        action="append",
        default=None,
        help="Apify proxy group (repeatable). Ignored if --no-proxy is set.",
    )
    parser.add_argument(
        "--proxy-json",
        help="Raw JSON string with proxy configuration. Overrides --no-proxy/--proxy-group.",
    )
    parser.add_argument(
        "--skill-extractor",
        choices=("heuristic", "local", "openai"),
        default="heuristic",
        help=(
            "Algorithm used to derive skills from job descriptions. "
            "'heuristic' uses rule-based parsing, 'openai' leverages an OpenAI chat model."
        ),
    )
    parser.add_argument(
        "--openai-api-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="API key for OpenAI skill extraction. Defaults to the OPENAI_API_KEY environment variable.",
    )
    parser.add_argument(
        "--openai-model",
        default=os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        help=f"OpenAI model for skill extraction when --skill-extractor=openai (default: {DEFAULT_OPENAI_MODEL}).",
    )
    parser.add_argument(
        "--openai-base-url",
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="Base URL for OpenAI-compatible API when --skill-extractor=openai.",
    )
    parser.add_argument(
        "--openai-requests-per-minute",
        type=float,
        default=DEFAULT_OPENAI_RPM,
        help=(
            "Soft rate limit for OpenAI calls (requests per minute). "
            f"Set to 0 to disable throttling (default: {DEFAULT_OPENAI_RPM})."
        ),
    )
    parser.add_argument(
        "--openai-max-retries",
        type=int,
        default=DEFAULT_OPENAI_MAX_RETRIES,
        help=(
            "Maximum retry attempts for OpenAI skill extraction on transient errors "
            f"(default: {DEFAULT_OPENAI_MAX_RETRIES})."
        ),
    )
    parser.add_argument(
        "--openai-backoff-seconds",
        type=float,
        default=DEFAULT_OPENAI_BACKOFF_SECONDS,
        help=(
            "Initial backoff (seconds) when retrying OpenAI calls after rate limits. "
            f"Backoff grows exponentially (default: {DEFAULT_OPENAI_BACKOFF_SECONDS})."
        ),
    )
    parser.add_argument(
        "--openai-batch-size",
        type=int,
        default=DEFAULT_OPENAI_BATCH_SIZE,
        help=(
            "Number of job descriptions to include in each OpenAI request when extracting skills "
            f"(default: {DEFAULT_OPENAI_BATCH_SIZE})."
        ),
    )
    return parser.parse_args(argv)


def build_run_input(args: argparse.Namespace, title_query: str, proxy_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    rows = max(1, min(args.rows, 1000))
    if rows != args.rows:
        logging.warning("Row count %s clamped to %s to match API limits.", args.rows, rows)

    run_input: Dict[str, Any] = {
        "title": title_query,
        "location": args.location,
        "companyName": args.company_names or [],
        "companyId": args.company_ids or [],
        "publishedAt": args.published_after,
        "rows": rows,
    }

    if args.work_type:
        run_input["workType"] = args.work_type
    if args.contract_type:
        run_input["contractType"] = args.contract_type
    if args.experience_level:
        run_input["experienceLevel"] = args.experience_level

    if proxy_config is not None:
        run_input["proxy"] = proxy_config

    return run_input


def fetch_jobs(client: ApifyClient, actor_id: str, run_input: Dict[str, Any]) -> List[Dict[str, Any]]:
    
    print("========== fetch_jobs ===========")
    print(run_input)
    
    run = client.actor(actor_id).call(run_input=run_input)
    dataset_client = client.dataset(run["defaultDatasetId"])
    return list(dataset_client.iterate_items())


def resolve_title_queries(args: argparse.Namespace) -> List[str]:
    queries: List[str] = []

    if args.titles:
        for raw in args.titles:
            if raw:
                cleaned = raw.strip()
                if cleaned:
                    queries.append(cleaned)

    if args.title_file:
        title_path = Path(args.title_file)
        if not title_path.exists():
            raise FileNotFoundError(f"Title file not found: {title_path}")
        content = title_path.read_text(encoding="utf-8")
        for line in content.splitlines():
            cleaned = line.strip()
            if cleaned and not cleaned.startswith("#"):
                queries.append(cleaned)

    if not queries:
        queries.append(DEFAULT_TITLE_QUERY)

    deduped: List[str] = []
    seen = set()
    for query in queries:
        key = query.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(query)

    return deduped


def load_config(path: Optional[str]) -> Dict[str, Any]:
    """Load configuration JSON from disk. Defaults to config.json if no path is provided."""
    if path:
        config_path = Path(path)
    else:
        config_path = Path("config.json")

    if not config_path.exists():
        if path:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        return {}

    try:
        content = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise OSError(f"Failed to read config file {config_path}: {exc}") from exc

    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in config file {config_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Config file {config_path} must contain a JSON object.")

    return data


def _to_string_list(value: Any, *, field_name: str) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        result: List[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                result.append(text)
        return result
    raise ValueError(f"Expected a string or sequence for '{field_name}', got {type(value).__name__}.")


def apply_config_to_args(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    if not config:
        return

    def set_if_present(attr: str, keys: Sequence[str], transform: Optional[Callable[[Any], Any]] = None) -> None:
        for key in keys:
            if key in config and config[key] is not None:
                value = config[key]
                if transform:
                    value = transform(value)
                setattr(args, attr, value)
                return

    set_if_present("token", ("apify_token", "token"))
    set_if_present("actor_id", ("actor_id",))
    set_if_present("location", ("location",))
    set_if_present("rows", ("rows",), lambda v: int(v))
    set_if_present("published_after", ("published_after",))
    set_if_present("work_type", ("work_type",))
    set_if_present("contract_type", ("contract_type",))
    set_if_present("experience_level", ("experience_level",))
    set_if_present("skill_extractor", ("skill_extractor",))
    set_if_present("openai_api_key", ("openai_api_key",))
    set_if_present("openai_model", ("openai_model",))
    set_if_present("openai_base_url", ("openai_base_url",))
    set_if_present("openai_requests_per_minute", ("openai_requests_per_minute",), float)
    set_if_present("openai_max_retries", ("openai_max_retries",), int)
    set_if_present("openai_backoff_seconds", ("openai_backoff_seconds",), float)
    set_if_present("openai_batch_size", ("openai_batch_size",), int)
    set_if_present("csv_path", ("csv_path",))
    set_if_present("table_limit", ("table_limit",), lambda v: int(v))

    if "titles" in config and config["titles"] is not None:
        args.titles = _to_string_list(config["titles"], field_name="titles")

    if "title_file" in config and config["title_file"]:
        logging.warning("Config field 'title_file' is deprecated; prefer using 'titles'.")
        args.title_file = str(config["title_file"])

    if "company_names" in config and config["company_names"] is not None:
        args.company_names = _to_string_list(config["company_names"], field_name="company_names")

    if "company_ids" in config and config["company_ids"] is not None:
        args.company_ids = _to_string_list(config["company_ids"], field_name="company_ids")

    if "proxy_groups" in config and config["proxy_groups"] is not None:
        args.proxy_groups = _to_string_list(config["proxy_groups"], field_name="proxy_groups")

    if "proxy" in config and config["proxy"] is not None:
        if not isinstance(config["proxy"], dict):
            raise ValueError("Config field 'proxy' must be an object.")
        setattr(args, "proxy_config", config["proxy"])

    if "proxy_json" in config and config["proxy_json"] is not None:
        args.proxy_json = config["proxy_json"]

    if "no_proxy" in config and config["no_proxy"] is not None:
        args.no_proxy = bool(config["no_proxy"])


def resolve_proxy_config(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    config_override = getattr(args, "proxy_config", None)
    if config_override is not None:
        return config_override

    if args.proxy_json:
        try:
            parsed = json.loads(args.proxy_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid proxy JSON provided: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("Proxy JSON must evaluate to an object/dict.")
        return parsed

    if args.no_proxy:
        return None

    proxy_groups = args.proxy_groups if args.proxy_groups else ["RESIDENTIAL"]
    return {
        "useApifyProxy": True,
        "apifyProxyGroups": proxy_groups,
    }


def normalize_job_entry(
    item: Dict[str, Any],
    search_query: str,
) -> JobRecord:
    description = (
        item.get("description")
        or item.get("jobDescription")
        or item.get("jobDescriptionText")
        or ""
    )
    sections = extract_description_sections(description)

    heuristic_skills = normalize_skill_phrases(sections["skills"])
    sections["skills"] = heuristic_skills

    return JobRecord(
        search_query=search_query,
        title=(item.get("title") or "").strip(),
        company=(item.get("companyName") or item.get("company") or "").strip(),
        location=(item.get("location") or item.get("companyLocation") or "").strip(),
        published_at=(item.get("publishedAt") or item.get("listedAt") or ""),
        url=(item.get("jobUrl") or item.get("url") or item.get("link") or ""),
        responsibilities=sections["responsibilities"],
        skills=sections["skills"],
        requirements=sections["requirements"],
        description=description.strip(),
    )


def extract_description_sections(description: str) -> Dict[str, List[str]]:
    """Pull out structured responsibilities, skills, and requirement snippets."""
    sections: Dict[str, List[str]] = {name: [] for name in SECTION_MAP}
    if not description:
        return sections

    text = re.sub(r"\r\n?", "\n", description)
    lines = [line.strip() for line in text.split("\n")]

    current: Optional[str] = None
    for raw_line in lines:
        line = raw_line.strip(" -*•\t")
        if not line:
            current = None
            continue

        normalized = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", " ", line).strip().lower()

        matched_section = None
        for section_name, headings in SECTION_MAP.items():
            if any(
                normalized.startswith(head.lower())
                or normalized == head.lower()
                for head in headings
            ):
                matched_section = section_name
                break

        if matched_section:
            current = matched_section
            continue

        if current:
            sections[current].append(line)
            continue

        if "responsib" in normalized:
            sections["responsibilities"].append(line)
        elif any(keyword in normalized for keyword in ("skill", "competenc", "能力")):
            sections["skills"].append(line)
        elif any(keyword in normalized for keyword in ("require", "qualif", "experience", "要求")):
            sections["requirements"].append(line)

    for name, values in sections.items():
        deduped: List[str] = []
        seen = set()
        for value in values:
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(value)
        sections[name] = deduped

    return sections


def create_skill_extractor(args: argparse.Namespace) -> Optional[Callable[[List[SkillTask]], List[List[str]]]]:
    if args.skill_extractor == "heuristic":
        def extractor(tasks: List[SkillTask]) -> List[List[str]]:
            return [normalize_skill_phrases(fallback) for _, fallback in tasks]

        return extractor

    if args.skill_extractor == "local":
        return extract_skills_locally

    if args.skill_extractor == "openai":
        api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing OpenAI API key. Provide --openai-api-key or set OPENAI_API_KEY when "
                "using --skill-extractor=openai."
            )
        if requests is None:
            raise ValueError(
                "The 'requests' package is required for OpenAI skill extraction. Install it via 'pip install requests'."
            )
        base_url = (args.openai_base_url or "https://api.openai.com/v1").rstrip("/")
        model = args.openai_model or DEFAULT_OPENAI_MODEL
        requests_per_minute = max(0.0, float(args.openai_requests_per_minute or 0))
        max_retries = max(1, int(args.openai_max_retries or DEFAULT_OPENAI_MAX_RETRIES))
        backoff_seconds = max(0.0, float(args.openai_backoff_seconds or DEFAULT_OPENAI_BACKOFF_SECONDS))
        batch_size = max(1, int(args.openai_batch_size or DEFAULT_OPENAI_BATCH_SIZE))

        rate_limiter = SimpleRateLimiter(requests_per_minute) if requests_per_minute > 0 else None

        def extractor(tasks: List[SkillTask]) -> List[List[str]]:
            return extract_skills_with_openai_batch(
                tasks=tasks,
                api_key=api_key,
                model=model,
                base_url=base_url,
                rate_limiter=rate_limiter,
                max_retries=max_retries,
                backoff_seconds=backoff_seconds,
                batch_size=batch_size,
            )

        return extractor

    raise ValueError(f"Unsupported skill extractor: {args.skill_extractor}")


def extract_skills_with_openai_batch(
    tasks: Sequence[SkillTask],
    api_key: str,
    model: str,
    base_url: str,
    rate_limiter: Optional[SimpleRateLimiter],
    max_retries: int,
    backoff_seconds: float,
    batch_size: int,
) -> List[List[str]]:
    if not tasks:
        return []

    if requests is None:
        logging.warning("Requests library unavailable; cannot call OpenAI. Falling back to heuristics.")
        return [normalize_skill_phrases(fallback) for _, fallback in tasks]

    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    aggregated_results: List[List[str]] = []
    for start in range(0, len(tasks), batch_size):
        chunk = list(tasks[start : start + batch_size])
        chunk_results = _call_openai_chunk(
            chunk=chunk,
            url=url,
            headers=headers,
            model=model,
            rate_limiter=rate_limiter,
            max_retries=max_retries,
            backoff_seconds=backoff_seconds,
        )
        aggregated_results.extend(chunk_results)

    return aggregated_results


def _call_openai_chunk(
    chunk: Sequence[SkillTask],
    url: str,
    headers: Dict[str, str],
    model: str,
    rate_limiter: Optional[SimpleRateLimiter],
    max_retries: int,
    backoff_seconds: float,
) -> List[List[str]]:
    fallback_results = [normalize_skill_phrases(fallback) for _, fallback in chunk]

    descriptions: List[str] = []
    for index, (description, _) in enumerate(chunk, start=1):
        text = (description or "").strip()
        if not text:
            text = "(description not provided)"
        if len(text) > MAX_DESCRIPTION_CHARS:
            text = text[: MAX_DESCRIPTION_CHARS] + "..."
        descriptions.append(f"{index}. {text}")

    prompt = (
        "For each numbered job description below, list the distinct skills a candidate must possess. "
        "Respond with a JSON array where each element corresponds to the matching job description order. "
        "Each element must be an array of concise skill strings. Use an empty array if no skills are present. "
        "Avoid explanations and keep skill names short."
    )
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "You extract skill inventories for recruiters."},
            {"role": "user", "content": f"{prompt}\n\n" + "\n\n".join(descriptions)},
        ],
    }

    attempt = 0
    while attempt < max_retries:
        attempt += 1
        if rate_limiter:
            rate_limiter.wait()
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=90)
        except requests.RequestException as exc:
            logging.warning("OpenAI request failed (attempt %d/%d): %s", attempt, max_retries, exc)
            if attempt >= max_retries:
                return fallback_results
            _sleep_with_backoff(attempt, backoff_seconds)
            continue

        status = response.status_code
        if status == 429 or status >= 500:
            logging.warning(
                "OpenAI rate/server limit (status %s) on attempt %d/%d.",
                status,
                attempt,
                max_retries,
            )
            if attempt >= max_retries:
                return fallback_results
            _sleep_with_backoff(attempt, backoff_seconds)
            continue

        if status >= 400:
            logging.warning("OpenAI request returned status %s: %s", status, response.text[:200])
            return fallback_results

        try:
            body = response.json()
        except json.JSONDecodeError as exc:
            logging.warning("Failed to decode OpenAI response JSON: %s", exc)
            return fallback_results

        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            logging.warning("Unexpected OpenAI response format: %s", exc)
            return fallback_results

        parsed = parse_skill_batches(content, len(chunk))
        if not parsed:
            logging.warning("OpenAI response could not be parsed; falling back to heuristics.")
            return fallback_results

        normalized_results = [
            normalize_skill_phrases(skills) for skills in parsed
        ]
        return normalized_results

    return fallback_results


def parse_skill_batches(raw_content: str, expected_count: int) -> Optional[List[List[str]]]:
    if not raw_content:
        return None

    def _load_json(candidate: str) -> Optional[Any]:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    data = _load_json(raw_content)
    if data is None:
        match = re.search(r"\[[\s\S]+\]", raw_content)
        if match:
            data = _load_json(match.group(0))

    if not isinstance(data, list) or len(data) != expected_count:
        return None

    results: List[List[str]] = []
    for element in data:
        if isinstance(element, list):
            skills = [str(item).strip() for item in element if str(item).strip()]
            results.append(skills)
        elif isinstance(element, str) and element.strip():
            results.append([element.strip()])
        else:
            results.append([])

    return results


def parse_skill_list(raw_content: str) -> List[str]:
    """Best-effort parsing of JSON or bullet-form skill lists."""
    if not raw_content:
        return []

    candidates: List[str] = []

    def normalize_sequence(seq: Iterable[str]) -> List[str]:
        normalized: List[str] = []
        seen = set()
        for value in seq:
            item = (value or "").strip(" -*•\t\r\n")
            if not item:
                continue
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(item)
        return normalized

    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError:
        match = re.search(r"\[[^\]]+\]", raw_content, flags=re.S)
        parsed = None
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                parsed = None

    if isinstance(parsed, dict):
        for key in ("skills", "skill_list", "list"):
            value = parsed.get(key)
            if isinstance(value, list):
                candidates.extend(value)
    elif isinstance(parsed, list):
        candidates.extend(parsed)

    if not candidates:
        lines = re.split(r"[\n,;•\-]+", raw_content)
        candidates.extend(lines)

    return normalize_sequence(candidates)


def clean_skill_phrase(text: str) -> Optional[str]:
    if not text:
        return None
    cleaned = text.strip(" -•\t\r\n.:;")
    if not cleaned:
        return None
    lowered = cleaned.lower()
    for prefix in SKILL_PREFIXES:
        if lowered.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip(" -•\t\r\n.:;")
            lowered = cleaned.lower()
            break
    for suffix in SKILL_SUFFIXES:
        if lowered.endswith(suffix):
            cleaned = cleaned[: -len(suffix)].strip(" -•\t\r\n.:;")
            lowered = cleaned.lower()
    if not cleaned:
        return None
    if len(cleaned) > MAX_SKILL_CHARS:
        return None
    if len(cleaned.split()) > MAX_SKILL_WORDS:
        return None
    return cleaned


def normalize_skill_phrases(skills: List[str]) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for raw in skills:
        if not raw:
            continue
        segments = re.split(r"[,/;•\u2022\r\n]+", raw)
        for segment in segments:
            candidate = clean_skill_phrase(segment)
            if not candidate:
                continue
            key = candidate.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(candidate)
    return normalized


def extract_skills_locally(tasks: Sequence[SkillTask]) -> List[List[str]]:
    results: List[List[str]] = []
    for description, fallback in tasks:
        text = description or ""
        text_lower = text.lower()
        candidates: List[str] = list(fallback)

        for phrase in LOCAL_SKILL_PHRASES:
            if phrase in text_lower:
                candidates.append(phrase)

        token_pattern = re.compile(r"\b[A-Za-z][A-Za-z0-9\+#\.\-/]{1,}\b")
        for token in token_pattern.findall(text):
            stripped = token.strip("-./")
            if not stripped:
                continue
            lower = stripped.lower()
            if lower in LOCAL_SKILL_SINGLE_WORDS:
                candidates.append(LOCAL_SKILL_SINGLE_WORDS[lower])
            elif re.fullmatch(r"[A-Z0-9]{2,5}", stripped):
                candidates.append(stripped)

        results.append(normalize_skill_phrases(candidates))
    return results


class SimpleRateLimiter:
    def __init__(self, requests_per_minute: float):
        self.min_interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self._last_invocation: float = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        now = time.monotonic()
        elapsed = now - self._last_invocation
        pause = self.min_interval - elapsed
        if pause > 0:
            time.sleep(pause)
            now = time.monotonic()
        self._last_invocation = now


def _sleep_with_backoff(attempt: int, base_delay: float) -> None:
    if base_delay <= 0:
        return
    delay = base_delay * (2 ** (attempt - 1))
    capped_delay = min(delay, 30.0)
    time.sleep(capped_delay)


def format_section_for_console(items: List[str], max_items: int, max_chars: int) -> str:
    if not items:
        return "-"
    joined = "; ".join(items[:max_items])
    if len(joined) > max_chars:
        return joined[: max_chars - 1] + "…"
    return joined


def deduplicate_records(records: List[JobRecord]) -> List[JobRecord]:
    deduped: List[JobRecord] = []
    seen = set()
    for record in records:
        key = (
            (record.url or "").lower(),
            (record.title or "").lower(),
            (record.company or "").lower(),
            (record.location or "").lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def print_jobs_table(records: List[JobRecord], limit: int) -> None:
    if not records:
        print("No matching jobs found.")
        return

    headers = ["Query", "Title", "Company", "Location", "Published", "Responsibilities", "Skills"]
    max_widths = [36, 40, 28, 24, 20, 60, 60]

    rows: List[List[str]] = []
    for record in records[:limit]:
        rows.append(
            [
                record.search_query or "-",
                record.title or "-",
                record.company or "-",
                record.location or "-",
                (record.published_at[:19] if record.published_at else "-"),
                format_section_for_console(record.responsibilities, 3, max_widths[4]),
                format_section_for_console(record.skills, 3, max_widths[5]),
            ]
        )

    widths = []
    for col_idx, header in enumerate(headers):
        column_values = [header] + [row[col_idx] for row in rows]
        width = min(max(len(value) for value in column_values), max_widths[col_idx])
        widths.append(width)

    def format_row(values: List[str]) -> str:
        return " | ".join(value[:widths[idx]].ljust(widths[idx]) for idx, value in enumerate(values))

    separator = "-+-".join("-" * width for width in widths)
    print(format_row(headers))
    print(separator)
    for row in rows:
        print(format_row(row))

    if len(records) > limit:
        print(f"... {len(records) - limit} more rows not shown")


def write_jobs_csv(records: List[JobRecord], csv_path: str) -> Path:
    destination = Path(csv_path)
    if destination.parent and not destination.parent.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "search_query",
        "title",
        "company",
        "location",
        "published_at",
        "url",
        "responsibilities",
        "skills",
        "requirements",
        "description",
    ]

    with destination.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "search_query": record.search_query,
                    "title": record.title,
                    "company": record.company,
                    "location": record.location,
                    "published_at": record.published_at,
                    "url": record.url,
                    "responsibilities": "; ".join(record.responsibilities),
                    "skills": "; ".join(record.skills),
                    "requirements": "; ".join(record.requirements),
                    "description": record.description,
                }
            )

    return destination


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        config_data = load_config(args.config)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Failed to load configuration: {exc}", file=sys.stderr)
        return 1

    try:
        apply_config_to_args(args, config_data)
    except ValueError as exc:
        print(f"Invalid configuration value: {exc}", file=sys.stderr)
        return 1

    if not args.token:
        print("Missing Apify token. Pass --token, set APIFY_TOKEN, or provide 'apify_token' in the config file.", file=sys.stderr)
        return 1

    try:
        title_queries = resolve_title_queries(args)
    except Exception as exc:
        print(f"Failed to prepare title queries: {exc}", file=sys.stderr)
        return 2

    try:
        skill_extractor = create_skill_extractor(args)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 3

    try:
        proxy_config = resolve_proxy_config(args)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 4

    client = ApifyClient(args.token)

    aggregated_records: List[JobRecord] = []
    total_raw = 0

    for query in title_queries:
        run_input = build_run_input(args, query, proxy_config)
        try:
            raw_items = fetch_jobs(client, args.actor_id, run_input)
        except Exception as exc:
            logging.error("Failed to fetch jobs from Apify for query '%s': %s", query, exc)
            continue

        total_raw += len(raw_items)
        filtered_items = raw_items
        normalized = [
            normalize_job_entry(item, query) for item in filtered_items
        ]
        if skill_extractor and normalized:
            tasks: List[Tuple[str, List[str]]] = [
                (record.description, record.skills) for record in normalized
            ]
            try:
                batched_skills = skill_extractor(tasks)
            except Exception as exc:
                logging.warning("Skill extraction batch failed for query '%s': %s", query, exc)
                batched_skills = []
            if batched_skills and len(batched_skills) == len(normalized):
                for record, skills in zip(normalized, batched_skills):
                    if skills:
                        record.skills = skills
            elif batched_skills:
                logging.warning(
                    "Skill extractor returned %d results for %d records on query '%s'; ignoring batch output.",
                    len(batched_skills),
                    len(normalized),
                    query,
                )
        aggregated_records.extend(normalized)
        logging.info(
            "Query '%s': %d raw items processed.",
            query,
            len(raw_items),
        )

    if not aggregated_records:
        print("No matching jobs found across the provided queries.")
        return 0

    normalized_records = deduplicate_records(aggregated_records)
    logging.info(
        "Aggregated %d unique records from %d queries (raw items fetched: %d).",
        len(normalized_records),
        len(title_queries),
        total_raw,
    )

    print_jobs_table(normalized_records, args.table_limit)

    try:
        destination = write_jobs_csv(normalized_records, args.csv_path)
    except OSError as exc:
        print(f"Failed to write CSV output: {exc}", file=sys.stderr)
        return 5

    print(f"CSV export saved to {destination}")
    return 0


if __name__ == "__main__":
    main()
