"""Shared utilities for extracting structured job insights from free-form text."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import requests


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

TITLE_KEYWORDS = (
    "job title",
    "title",
    "position",
    "role",
    "opportunity",
    "opening",
    "hiring",
    "we're hiring",
    "seeking",
    "looking for",
    "vacancy",
)

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

MAX_SKILL_WORDS = 8
MAX_SKILL_CHARS = 80
MAX_DESCRIPTION_CHARS = 4000


def normalize_skill_phrases(skills: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for raw in skills:
        if not raw:
            continue
        segments = re.split(r"[,/;•\u2022\r\n]+", raw)
        for segment in segments:
            cleaned = clean_skill_phrase(segment)
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(cleaned)
    return normalized


def clean_skill_phrase(text: str) -> Optional[str]:
    if not text:
        return None
    cleaned = text.strip(" -•\t\r\n.:;")
    if not cleaned:
        return None
    lowered = cleaned.lower()
    for prefix in SKILL_PREFIXES:
        if lowered.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip(" -•\t\r\n.:;")
            lowered = cleaned.lower()
            break
    for suffix in SKILL_SUFFIXES:
        if lowered.endswith(suffix):
            cleaned = cleaned[:-len(suffix)].strip(" -•\t\r\n.:;")
            lowered = cleaned.lower()
    if not cleaned:
        return None
    if len(cleaned) > MAX_SKILL_CHARS or len(cleaned.split()) > MAX_SKILL_WORDS:
        return None
    return cleaned


def extract_description_sections(description: str) -> Dict[str, List[str]]:
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


def extract_skills_locally(description: str, fallback: Iterable[str]) -> List[str]:
    candidates: List[str] = list(fallback)
    text_lower = (description or "").lower()

    for phrase in LOCAL_SKILL_PHRASES:
        if phrase in text_lower:
            candidates.append(phrase)

    for hashtag in re.findall(r"#([A-Za-z0-9][A-Za-z0-9_\-]+)", description or ""):
        candidates.append(hashtag.replace('_', ' ').replace('-', ' '))

    token_pattern = re.compile(r"\b[A-Za-z][A-Za-z0-9\+#\.\-/]{1,}\b")
    for token in token_pattern.findall(description or ""):
        stripped = token.strip("-./")
        if not stripped:
            continue
        lower = stripped.lower()
        if lower in LOCAL_SKILL_SINGLE_WORDS:
            candidates.append(LOCAL_SKILL_SINGLE_WORDS[lower])
        elif re.fullmatch(r"[A-Z0-9]{2,5}", stripped):
            candidates.append(stripped)

    return normalize_skill_phrases(candidates)


def guess_job_titles(description: str) -> List[str]:
    lines = [line.strip() for line in re.sub(r"\r\n?", "\n", description or "").split("\n")]
    titles: List[str] = []
    for line in lines:
        if not line or len(line) > 120:
            continue
        lowered = line.lower()
        if any(keyword in lowered for keyword in TITLE_KEYWORDS):
            titles.append(line.strip(":- "))
        else:
            match = re.search(
                r"\b([A-Z][A-Za-z0-9\-&/]+(?:\s+[A-Z][A-Za-z0-9\-&/]+){0,5}(Engineer|Specialist|Manager|Director|Lead|Designer|Developer|Consultant|Analyst|Architect|Scientist|Executive|Officer))\b",
                line,
            )
            if match:
                titles.append(match.group(1))
    # Deduplicate
    seen = set()
    result = []
    for title in titles:
        key = title.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(title)
    return result


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


def empty_insight() -> Dict[str, List[str]]:
    return {
        "job_titles": [],
        "responsibilities": [],
        "skills": [],
        "requirements": [],
    }


def parse_openai_job_batch(raw_content: str, expected_count: int) -> Optional[List[Dict[str, List[str]]]]:
    if not raw_content:
        return None

    def _try_load(candidate: str) -> Optional[Any]:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    data = _try_load(raw_content)
    if data is None:
        match = re.search(r"\[[\s\S]+\]", raw_content)
        if match:
            data = _try_load(match.group(0))

    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        return None

    if len(data) != expected_count:
        if len(data) == 1 and expected_count > 1:
            single = data[0]
            if not isinstance(single, dict):
                return None
            padded = [single] + [empty_insight() for _ in range(expected_count - 1)]
            data = padded
        elif len(data) == 1 and expected_count == 1:
            pass
        else:
            return None

    cleaned: List[Dict[str, List[str]]] = []
    for element in data:
        if not isinstance(element, dict):
            return None
        entry: Dict[str, List[str]] = {}
        for key in ("job_titles", "responsibilities", "skills", "requirements"):
            value = element.get(key, [])
            if isinstance(value, list):
                entry[key] = [str(item).strip() for item in value if str(item).strip()]
            elif isinstance(value, str) and value.strip():
                entry[key] = [value.strip()]
            else:
                entry[key] = []
        cleaned.append(entry)
    return cleaned


def _call_openai_job_chunk(
    chunk: Sequence[str],
    url: str,
    headers: Dict[str, str],
    model: str,
    rate_limiter: Optional[SimpleRateLimiter],
    max_retries: int,
    backoff_seconds: float,
) -> Optional[List[Dict[str, List[str]]]]:
    if not chunk:
        return []

    prompt_lines = [
        "Extract structured job information from each LinkedIn post below.",
        "For every CONTENT block return four arrays: job_titles, responsibilities, skills, requirements.",
        "Infer information when reasonable. Treat explicit titles (e.g. ‘Senior Cloud & AI Platform Specialist’) or clear hiring phrases as job_titles.",
        "Responsibilities should be short action phrases (e.g. 'unlock the full potential of Azure and AI').",
        "Skills include technologies, tools, methodologies, or hashtags that imply competencies (convert hashtags into readable phrases).",
        "Requirements capture qualifications or traits implied as necessary (e.g. 'curiosity', 'collaboration').",
        "If a category is not present leave the array empty. Keep responses concise.",
        "Respond strictly with a JSON array of length N where element i corresponds to CONTENT i.",
    ]
    payload = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "You are an assistant that extracts structured job insights from text."},
            {
                "role": "user",
                "content": "\n".join(prompt_lines + [
                    "",
                    *[f"CONTENT {idx + 1}:\n{text.strip()[:MAX_DESCRIPTION_CHARS]}" for idx, text in enumerate(chunk)],
                ]),
            },
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
            print(f"[!] OpenAI request failed (attempt {attempt}/{max_retries}): {exc}")
            if attempt >= max_retries:
                return None
            _sleep_with_backoff(attempt, backoff_seconds)
            continue

        status = response.status_code
        if status == 429 or status >= 500:
            print(f"[!] OpenAI rate/server limit (status {status}) on attempt {attempt}/{max_retries}.")
            if attempt >= max_retries:
                return None
            _sleep_with_backoff(attempt, backoff_seconds)
            continue

        if status >= 400:
            print(f"[!] OpenAI request returned status {status}: {response.text[:200]}")
            return None

        try:
            body = response.json()
        except json.JSONDecodeError as exc:
            print(f"[!] Failed to decode OpenAI response JSON: {exc}")
            return None

        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            print(f"[!] Unexpected OpenAI response format: {exc}")
            return None

        parsed = parse_openai_job_batch(content, len(chunk))
        if parsed is None:
            print("[!] OpenAI response could not be parsed as expected JSON structure.")
            return None
        return parsed

    return None


@dataclass
class JobInsightExtractor:
    mode: str = "local"  # 'local', 'openai', or 'none'
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_requests_per_minute: float = 8.0
    openai_batch_size: int = 5
    openai_max_retries: int = 5
    openai_backoff_seconds: float = 3.0

    def extract_many(self, texts: Sequence[str]) -> List[Dict[str, List[str]]]:
        if not texts:
            return []

        if self.mode in (None, "", "none"):
            return [self._empty_result() for _ in texts]

        if self.mode == "openai":
            if not self.openai_api_key:
                raise ValueError("OpenAI API key is required when mode is 'openai'.")
            return self._extract_with_openai(texts)

        # Default to local heuristic extraction
        return [self._extract_local(text) for text in texts]

    def _empty_result(self) -> Dict[str, List[str]]:
        return {
            "job_titles": [],
            "responsibilities": [],
            "skills": [],
            "requirements": [],
        }

    def _extract_local(self, description: str) -> Dict[str, List[str]]:
        sections = extract_description_sections(description or "")
        skills = extract_skills_locally(description or "", sections["skills"])
        return {
            "job_titles": guess_job_titles(description or ""),
            "responsibilities": sections["responsibilities"],
            "skills": skills,
            "requirements": sections["requirements"],
        }

    def _extract_with_openai(self, texts: Sequence[str]) -> List[Dict[str, List[str]]]:
        url = f"{self.openai_base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json",
        }
        rate_limiter = SimpleRateLimiter(self.openai_requests_per_minute)
        results: List[Dict[str, List[str]]] = []

        for start in range(0, len(texts), self.openai_batch_size):
            chunk = list(texts[start:start + self.openai_batch_size])
            llm_result = _call_openai_job_chunk(
                chunk=chunk,
                url=url,
                headers=headers,
                model=self.openai_model,
                rate_limiter=rate_limiter,
                max_retries=self.openai_max_retries,
                backoff_seconds=self.openai_backoff_seconds,
            )
            if llm_result is None:
                llm_result = [self._extract_local(text) for text in chunk]
            else:
                patched: List[Dict[str, List[str]]] = []
                for text, insight in zip(chunk, llm_result):
                    if not any(insight.get(key) for key in ("job_titles", "responsibilities", "skills", "requirements")):
                        patched.append(self._extract_local(text))
                    else:
                        patched.append({
                            "job_titles": insight.get("job_titles", []),
                            "responsibilities": insight.get("responsibilities", []),
                            "skills": insight.get("skills", []),
                            "requirements": insight.get("requirements", []),
                        })
                llm_result = patched
            results.extend(llm_result)

        return results[:len(texts)]
