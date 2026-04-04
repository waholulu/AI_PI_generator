import os
from typing import Any, Dict, List, Optional

import requests

try:
    import pyalex
except ImportError:  # pragma: no cover - exercised in dependency-light test envs
    pyalex = None


def configure_openalex() -> None:
    """Apply OpenAlex credentials and retry policy from environment."""
    if pyalex is None:
        return

    api_key = os.getenv("OPENALEX_API_KEY")
    if api_key:
        pyalex.config.api_key = api_key

    email = os.getenv("OPENALEX_EMAIL")
    if email:
        pyalex.config.email = email

    pyalex.config.max_retries = 3
    pyalex.config.retry_backoff_factor = 0.5
    pyalex.config.retry_http_codes = [429, 500, 503]


def reconstruct_abstract(work: Dict[str, Any]) -> str:
    """Rebuild abstract text from inverted index when needed."""
    abstract = work.get("abstract")
    if abstract:
        return abstract

    inverted_index = work.get("abstract_inverted_index") or {}
    if not inverted_index:
        return ""

    token_positions: List[tuple[int, str]] = []
    for token, positions in inverted_index.items():
        for position in positions:
            if isinstance(position, int):
                token_positions.append((position, token))

    if not token_positions:
        return ""

    token_positions.sort(key=lambda item: item[0])
    return " ".join(token for _, token in token_positions)


def extract_authors(work: Dict[str, Any]) -> List[Dict[str, Any]]:
    authors = []
    for authorship in work.get("authorships", []):
        author_info = authorship.get("author") or {}
        institutions = authorship.get("institutions", [])
        authors.append(
            {
                "name": author_info.get("display_name", "Unknown"),
                "author_id": author_info.get("id"),
                "orcid": author_info.get("orcid"),
                "institutions": [
                    {
                        "name": institution.get("display_name"),
                        "id": institution.get("id"),
                        "country_code": institution.get("country_code"),
                    }
                    for institution in institutions
                ],
            }
        )
    return authors


def extract_concepts(
    work: Dict[str, Any],
    *,
    max_items: Optional[int] = None,
    max_level: Optional[int] = None,
) -> List[Dict[str, Any]]:
    concepts = []
    for concept in work.get("concepts", []):
        level = concept.get("level")
        if max_level is not None and isinstance(level, int) and level > max_level:
            continue
        concepts.append(
            {
                "id": concept.get("id"),
                "name": concept.get("display_name"),
                "level": level,
                "score": concept.get("score"),
            }
        )

    if max_items is not None:
        return concepts[:max_items]
    return concepts


def _collect_download_urls(work: Dict[str, Any]) -> List[str]:
    urls: List[str] = []
    candidates = [
        work.get("best_oa_location"),
        work.get("primary_location"),
        *(work.get("locations") or []),
    ]

    for location in candidates:
        if not isinstance(location, dict):
            continue
        for key in ("pdf_url", "landing_page_url"):
            url = location.get(key)
            if url and url not in urls:
                urls.append(url)

    oa_url = (work.get("open_access") or {}).get("oa_url")
    if oa_url and oa_url not in urls:
        urls.append(oa_url)

    return urls


def extract_work_metadata(work: Dict[str, Any]) -> Dict[str, Any]:
    openalex_id = work.get("id", "")
    primary_location = work.get("primary_location") or {}
    source = primary_location.get("source") or {}
    open_access = work.get("open_access") or {}
    authors = extract_authors(work)
    author_names = [author.get("name") for author in authors if author.get("name")]

    return {
        "title": work.get("display_name") or work.get("title", ""),
        "openalex_id": openalex_id,
        "paperId": openalex_id.replace("https://openalex.org/", "") if openalex_id else "",
        "doi": work.get("doi"),
        "year": work.get("publication_year"),
        "publication_date": work.get("publication_date"),
        "type": work.get("type"),
        "language": work.get("language"),
        "citationCount": work.get("cited_by_count", 0),
        "authors": authors,
        "author_names": author_names,
        "first_author": author_names[0] if author_names else "Unknown",
        "journal": source.get("display_name"),
        "journal_id": source.get("id"),
        "host_organization": source.get("host_organization_name"),
        "biblio": work.get("biblio") or {},
        "abstract": reconstruct_abstract(work),
        "concepts": extract_concepts(work),
        "broad_concepts": extract_concepts(work, max_items=5, max_level=1),
        "isOpenAccess": open_access.get("is_oa", False),
        "oa_status": open_access.get("oa_status"),
        "oa_url": open_access.get("oa_url"),
        "best_oa_location": work.get("best_oa_location") or {},
        "landing_page_url": primary_location.get("landing_page_url"),
        "pdf_url": primary_location.get("pdf_url"),
        "candidate_download_urls": _collect_download_urls(work),
        "has_fulltext": work.get("has_fulltext"),
        "referenced_works": work.get("referenced_works", []),
        "referenced_works_count": work.get("referenced_works_count", 0),
        "is_retracted": work.get("is_retracted", False),
    }


def search_openalex(
    query: str,
    limit: int = 20,
    from_year: int | None = None,
    *,
    from_publication_date: str | None = None,
    to_publication_date: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Search OpenAlex for works matching *query*, sorted by citation count.

    If *from_year* is provided, only works published on or after that year
    are returned (uses OpenAlex ``from_publication_date`` filter).

    Returns a list of metadata dicts (via extract_work_metadata).  If pyalex
    is not installed or the call fails, returns an empty list.
    """
    from agents.logging_config import get_logger
    _logger = get_logger(__name__)

    try:
        from pyalex import Works as _Works
    except ImportError:
        _logger.warning("OpenAlex search skipped: pyalex is not installed.")
        return []

    _logger.info("Querying OpenAlex for: %s (from_year=%s)", query, from_year)
    try:
        chain = _Works().search(query)
        filters: Dict[str, str] = {}
        if from_year is not None:
            filters["from_publication_date"] = f"{from_year}-01-01"
        if from_publication_date:
            filters["from_publication_date"] = from_publication_date
        if to_publication_date:
            filters["to_publication_date"] = to_publication_date
        if filters:
            chain = chain.filter(**filters)
        works = chain.sort(cited_by_count="desc").get(per_page=limit)
        return [extract_work_metadata(w) for w in works[:limit]]
    except Exception as e:
        _logger.error("OpenAlex API Error: %s", e)
        return []


def multi_search_openalex(
    query_pool: List[str],
    *,
    per_query_limit: int = 20,
    final_limit: int | None = None,
    from_year: int | None = None,
    cache_namespace: str = "openalex",
    cache_prefix: str = "openalex_query",
    from_publication_date: str | None = None,
    to_publication_date: str | None = None,
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Execute one OpenAlex search per query, deduplicate by openalex_id,
    and return (merged_results, query_hits).

    If *from_year* is provided, only works published on or after that year
    are returned.  If *final_limit* is set, stop collecting once enough
    unique papers are gathered and truncate the result.
    """
    import os as _os
    from agents.cache_utils import build_cache_key, load_json_cache, save_json_cache
    from agents.logging_config import get_logger
    _logger = get_logger(__name__)

    cache_hours = float(_os.getenv("OPENALEX_CACHE_HOURS", "48"))
    seen_ids: set[str] = set()
    merged: List[Dict[str, Any]] = []
    query_hits: Dict[str, int] = {}

    for query in query_pool:
        cache_key = build_cache_key(
            cache_prefix,
            {
                "query": query,
                "limit": per_query_limit,
                "from_year": from_year,
                "from_publication_date": from_publication_date,
                "to_publication_date": to_publication_date,
            },
        )
        cached = load_json_cache(cache_namespace, cache_key, max_age_hours=cache_hours)
        try:
            if isinstance(cached, list):
                results = cached
            else:
                results = search_openalex(
                    query,
                    limit=per_query_limit,
                    from_year=from_year,
                    from_publication_date=from_publication_date,
                    to_publication_date=to_publication_date,
                )
                save_json_cache(cache_namespace, cache_key, results)
        except Exception as exc:
            _logger.warning("Query '%s' failed: %s", query, exc)
            query_hits[query] = 0
            continue

        query_hits[query] = len(results)

        for paper in results:
            oa_id = (
                paper.get("openalex_id")
                or paper.get("paperId")
                or paper.get("doi")
                or paper.get("title", "")
            )
            if oa_id and oa_id in seen_ids:
                continue
            if oa_id:
                seen_ids.add(oa_id)
            merged.append(paper)

        if final_limit and len(merged) >= final_limit * 3:
            break

    if final_limit:
        merged = merged[:final_limit]

    return merged, query_hits


def download_pdf(url: str, target_path: str, timeout: int = 20) -> Dict[str, Any]:
    """
    Download a PDF when the URL points to one.

    Returns a status dict so callers can persist the outcome.
    """
    if not url:
        return {"status": "pdf_unavailable", "path": None, "reason": "missing_url"}

    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
    except Exception as exc:
        return {"status": "pdf_download_failed", "path": None, "reason": str(exc)}

    content_type = (response.headers.get("content-type") or "").lower()
    is_pdf = "pdf" in content_type or url.lower().endswith(".pdf")
    if not is_pdf:
        response.close()
        return {
            "status": "pdf_not_direct",
            "path": None,
            "reason": f"content_type={content_type or 'unknown'}",
        }

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "wb") as file_obj:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file_obj.write(chunk)
    response.close()

    return {"status": "pdf_downloaded", "path": target_path, "reason": None}
