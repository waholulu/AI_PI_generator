import json
import os
from collections import Counter
from typing import Any, Dict, List

from agents import settings
from agents.cache_utils import build_cache_key, load_json_cache, save_json_cache
from agents.logging_config import get_logger
from agents.openalex_utils import configure_openalex, multi_search_openalex
from agents.keyword_planner import KeywordPlanner

logger = get_logger(__name__)


class FieldScannerAgent:
    def __init__(self):
        configure_openalex()
        self._planner = KeywordPlanner()

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Executes the field scan to generate constraints for ideation."""
        domain = state.get("domain_input", "Urban Green Space and Inequality")
        logger.info("--- Module 0: Field Scanner for '%s' ---", domain)

        per_query_limit = int(os.getenv("OPENALEX_QUERY_REWRITE_PER_QUERY_LIMIT", "20"))
        plan_cache_hours = float(os.getenv("KEYWORD_PLAN_CACHE_HOURS", "24"))
        plan_cache_key = build_cache_key("field_scan_keyword_plan", {"domain": domain})
        keyword_plan = load_json_cache("keyword_plans", plan_cache_key, max_age_hours=plan_cache_hours)
        if not isinstance(keyword_plan, dict):
            keyword_plan = self._planner.plan(domain)
            save_json_cache("keyword_plans", plan_cache_key, keyword_plan)
        query_pool: List[str] = keyword_plan.get("query_pool") or [domain]
        used_fallback: bool = keyword_plan.get("used_fallback", True)

        logger.info(
            "Query pool (%d queries, fallback=%s): %s",
            len(query_pool), used_fallback, query_pool,
        )

        try:
            top_results_raw, query_hits = multi_search_openalex(
                query_pool,
                per_query_limit=per_query_limit,
                cache_prefix="field_scan_openalex_query",
            )
            # Convert full metadata to field-scan view with concept names
            top_results = []
            for m in top_results_raw:
                top_results.append({
                    "title": m["title"],
                    "citations": m["citationCount"],
                    "year": m["year"],
                    "author": m["first_author"],
                    "authors": m["author_names"],
                    "concepts": [
                        c.get("name") for c in m.get("broad_concepts", []) if c.get("name")
                    ],
                    "concept_details": m.get("broad_concepts", []),
                    "openalex_id": m["openalex_id"],
                    "doi": m["doi"],
                    "type": m["type"],
                    "journal": m["journal"],
                    "publication_date": m["publication_date"],
                    "is_open_access": m["isOpenAccess"],
                    "oa_status": m["oa_status"],
                    "oa_url": m["oa_url"],
                    "landing_page_url": m["landing_page_url"],
                    "pdf_url": m["pdf_url"],
                    "candidate_download_urls": m["candidate_download_urls"],
                    "referenced_works_count": m["referenced_works_count"],
                    "is_retracted": m["is_retracted"],
                })
        except Exception as e:
            logger.error("Failed to scan OpenAlex: %s", e)
            raise RuntimeError(f"Field scanner OpenAlex search failed: {e}") from e

        concept_counter: Counter[str] = Counter()
        for work in top_results:
            for concept in work.get("concepts", []) or []:
                if isinstance(concept, str):
                    concept_counter[concept] += 1

        high_traction_keywords: List[str] = [
            name for name, _ in concept_counter.most_common(15)
        ]

        scan_status = "full" if top_results else "empty"

        scan_results = {
            "domain_scanned": domain,
            "meta": {"scan_status": scan_status},
            "search_strategy": {
                "query_pool": query_pool,
                "queries_executed": len(query_pool),
                "used_fallback": used_fallback,
                "primary_domains": keyword_plan.get("primary_domains", []),
                "methods": keyword_plan.get("methods", []),
                "topics": keyword_plan.get("topics", []),
            },
            "query_hits": query_hits,
            "openalex_traction": {"top_results": top_results},
            "keywords": {
                "raw_query": domain,
                "high_traction": high_traction_keywords,
            },
        }

        field_scan_path = settings.field_scan_path()
        with open(field_scan_path, "w", encoding="utf-8") as f:
            json.dump(scan_results, f, indent=2, ensure_ascii=False)

        logger.info(
            "Field scan complete. %d unique works collected. Saved to %s",
            len(top_results), field_scan_path,
        )

        return {
            "field_scan_path": field_scan_path,
            "execution_status": "ideation",
        }


def field_scanner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    agent = FieldScannerAgent()
    return agent.run(state)


def summarize_field_scan(scan_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a compact summary of the field scan for downstream modules.
    """
    summary: Dict[str, Any] = {}

    keywords = (scan_data.get("keywords") or {}).get("high_traction") or []
    summary["high_traction_keywords"] = keywords

    top_results = (scan_data.get("openalex_traction") or {}).get("top_results") or []
    top_papers = []
    year_counter: Counter[int] = Counter()
    concept_counter: Counter[str] = Counter()

    for work in top_results:
        year = work.get("year")
        if isinstance(year, int):
            year_counter[year] += 1

        concepts = work.get("concepts") or []
        for concept in concepts:
            if isinstance(concept, str):
                concept_counter[concept] += 1

        top_papers.append(
            {
                "title": work.get("title"),
                "year": year,
                "citations": work.get("citations"),
                "concepts": concepts[:3] if isinstance(concepts, list) else [],
                "doi": work.get("doi"),
            }
        )

    summary["top_papers"] = top_papers[:10]
    summary["year_distribution"] = {
        str(year): count for year, count in sorted(year_counter.items())
    }

    crowded_concepts: List[str] = [
        name for name, count in concept_counter.items() if count >= 5
    ]
    summary["crowded_concepts"] = sorted(crowded_concepts)

    search_strategy = scan_data.get("search_strategy") or {}
    if search_strategy:
        summary["search_strategy"] = {
            "query_pool": search_strategy.get("query_pool", []),
            "used_fallback": search_strategy.get("used_fallback", True),
            "queries_executed": search_strategy.get("queries_executed", 1),
        }

    summary["scan_status"] = (
        (scan_data.get("meta") or {}).get("scan_status") or "unknown"
    )
    summary["domain_scanned"] = scan_data.get("domain_scanned")

    return summary
