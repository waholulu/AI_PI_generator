"""
arXiv search utilities for Auto-PI.

Wraps the `arxiv` Python SDK to return paper metadata in the same schema
used by openalex_utils so the literature agent can merge results uniformly.

Required package: arxiv>=2.4.0  (already in pyproject.toml)
"""

from __future__ import annotations

import time
from datetime import date
from typing import Any, Dict, List, Optional

from agents.logging_config import get_logger

logger = get_logger(__name__)


def _to_openalex_schema(result: Any) -> Dict[str, Any]:
    """Convert an arxiv.Result object to the OpenAlex paper dict schema."""
    authors = [{"name": str(a)} for a in (result.authors or [])]

    # Prefer the DOI-based URL, fall back to the PDF link
    candidate_urls: List[str] = []
    pdf_url: Optional[str] = None
    if result.pdf_url:
        candidate_urls.append(result.pdf_url)
        pdf_url = result.pdf_url

    # Primary ID: use DOI when present, otherwise the arXiv short-id
    doi: Optional[str] = result.doi or None
    arxiv_id: str = result.get_short_id() if hasattr(result, "get_short_id") else str(result.entry_id).split("/")[-1]
    paper_id = f"arxiv:{arxiv_id}"

    year: Optional[int] = None
    pub_date: Optional[str] = None
    if result.published:
        year = result.published.year
        pub_date = result.published.date().isoformat()

    # Flatten categories into concept-like objects
    concepts = [{"display_name": c} for c in (result.categories or [])]

    abstract = (result.summary or "").replace("\n", " ").strip()

    return {
        "paperId": paper_id,
        "openalex_id": None,  # arXiv has no OpenAlex ID natively
        "arxiv_id": arxiv_id,
        "title": (result.title or "").strip(),
        "authors": authors,
        "author_details": authors,
        "year": year,
        "publication_date": pub_date,
        "type": "preprint",
        "language": "en",
        "doi": doi,
        "journal": "arXiv",
        "journal_id": None,
        "host_organization": "arXiv",
        "abstract": abstract,
        "citationCount": 0,  # arXiv SDK does not expose citation counts
        "isOpenAccess": True,
        "oa_status": "green",
        "oa_url": result.pdf_url,
        "landing_page_url": str(result.entry_id),
        "pdf_url": pdf_url,
        "candidate_download_urls": candidate_urls,
        "biblio": {},
        "concepts": concepts,
        "referenced_works": [],
        "referenced_works_count": 0,
        "is_retracted": False,
        "source": "arxiv",
    }


def search_arxiv(
    query: str,
    max_results: int = 10,
    from_date: Optional[date] = None,
) -> List[Dict[str, Any]]:
    """Search arXiv and return papers in the shared OpenAlex-compatible schema.

    Args:
        query: Free-text search string (arXiv supports boolean operators).
        max_results: Maximum number of results to return.
        from_date: If provided, only return papers published on or after this date.

    Returns:
        List of paper dicts conforming to the openalex_utils metadata schema.
    """
    try:
        import arxiv  # type: ignore[import]
    except ImportError:
        logger.warning("arxiv package not installed; skipping arXiv search.")
        return []

    try:
        client = arxiv.Client(num_retries=2, delay_seconds=3.0)
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        results = list(client.results(search))
    except Exception as exc:
        logger.warning("arXiv search failed for query '%s': %s", query, exc)
        return []

    papers: List[Dict[str, Any]] = []
    for r in results:
        try:
            paper = _to_openalex_schema(r)
            # Date filter (arXiv API doesn't support server-side date filtering in all modes)
            if from_date and r.published and r.published.date() < from_date:
                continue
            papers.append(paper)
        except Exception as exc:
            logger.debug("Failed to parse arXiv result: %s", exc)

    logger.info("arXiv search '%s': %d results returned.", query[:60], len(papers))
    return papers


def multi_search_arxiv(
    queries: List[str],
    per_query_limit: int = 5,
    final_limit: int = 10,
    from_date: Optional[date] = None,
    inter_query_delay: float = 1.0,
) -> List[Dict[str, Any]]:
    """Run multiple arXiv queries and deduplicate by arxiv_id / DOI.

    Args:
        queries: List of search strings.
        per_query_limit: Max results per query.
        final_limit: Total results cap after dedup.
        from_date: Earliest publication date filter.
        inter_query_delay: Seconds to sleep between queries (be polite).

    Returns:
        Deduplicated list of papers capped at final_limit.
    """
    seen: set[str] = set()
    merged: List[Dict[str, Any]] = []

    for i, query in enumerate(queries):
        if i > 0:
            time.sleep(inter_query_delay)
        papers = search_arxiv(query, max_results=per_query_limit, from_date=from_date)
        for p in papers:
            key = p.get("arxiv_id") or p.get("doi") or p.get("title", "")
            if key and key not in seen:
                seen.add(key)
                merged.append(p)
        if len(merged) >= final_limit:
            break

    return merged[:final_limit]
