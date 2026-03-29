import json
import os
from collections import Counter
from typing import Any, Dict, List

from tenacity import retry, stop_after_attempt, wait_exponential

from agents.openalex_utils import configure_openalex, extract_work_metadata
from agents.keyword_planner import KeywordPlanner

try:
    from pyalex import Works
except ImportError:  # pragma: no cover - exercised in dependency-light test envs
    Works = None


class FieldScannerAgent:
    def __init__(self):
        configure_openalex()
        self._planner = KeywordPlanner()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=30))
    def _search_openalex(self, query: str, limit: int = 20) -> Dict[str, Any]:
        """Queries OpenAlex for top cited works matching the given domain query."""
        print(f"Querying OpenAlex for: {query}")
        if Works is None:
            print("OpenAlex search skipped: pyalex is not installed.")
            return {"organic_results": [], "total_results": 0}
        try:
            works = (
                Works()
                .search(query)
                .sort(cited_by_count="desc")
                .get(per_page=limit)
            )
            results = []
            for work in works[:limit]:
                metadata = extract_work_metadata(work)
                results.append({
                    "title": metadata["title"],
                    "citations": metadata["citationCount"],
                    "year": metadata["year"],
                    "author": metadata["first_author"],
                    "authors": metadata["author_names"],
                    "concepts": [
                        concept.get("name")
                        for concept in metadata["broad_concepts"]
                        if concept.get("name")
                    ],
                    "concept_details": metadata["broad_concepts"],
                    "openalex_id": metadata["openalex_id"],
                    "doi": metadata["doi"],
                    "type": metadata["type"],
                    "journal": metadata["journal"],
                    "publication_date": metadata["publication_date"],
                    "is_open_access": metadata["isOpenAccess"],
                    "oa_status": metadata["oa_status"],
                    "oa_url": metadata["oa_url"],
                    "landing_page_url": metadata["landing_page_url"],
                    "pdf_url": metadata["pdf_url"],
                    "candidate_download_urls": metadata["candidate_download_urls"],
                    "referenced_works_count": metadata["referenced_works_count"],
                    "is_retracted": metadata["is_retracted"],
                })
            return {
                "organic_results": results,
                "total_results": len(results),
            }
        except Exception as e:
            print(f"OpenAlex search error: {e}")
            return {"organic_results": [], "total_results": 0}

    def _multi_search(
        self,
        query_pool: List[str],
        per_query_limit: int = 20,
    ) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        Execute one OpenAlex search per query in *query_pool*, deduplicate by
        openalex_id, and return (merged_results, query_hits).

        Each result gains two extra fields:
          - matched_query  : the query string that first retrieved it
          - topic_label    : placeholder; callers may enrich this
        """
        seen_ids: set[str] = set()
        merged: List[Dict[str, Any]] = []
        query_hits: Dict[str, int] = {}

        for query in query_pool:
            try:
                data = self._search_openalex(query, limit=per_query_limit)
            except Exception as exc:
                print(f"[FieldScanner] Query '{query}' failed: {exc}")
                query_hits[query] = 0
                continue

            results = data.get("organic_results") or []
            query_hits[query] = len(results)

            for result in results:
                oa_id = result.get("openalex_id") or result.get("doi") or result.get("title", "")
                if oa_id and oa_id in seen_ids:
                    continue
                if oa_id:
                    seen_ids.add(oa_id)
                result.setdefault("matched_query", query)
                merged.append(result)

        return merged, query_hits

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Executes the field scan to generate constraints for ideation."""
        domain = state.get("domain_input", "Urban Green Space and Inequality")
        print(f"--- Module 0: Field Scanner for '{domain}' ---")

        # --- Step 1: Build query pool via KeywordPlanner ----------------
        per_query_limit = int(os.getenv("OPENALEX_QUERY_REWRITE_PER_QUERY_LIMIT", "20"))

        keyword_plan = self._planner.plan(domain)
        query_pool: List[str] = keyword_plan.get("query_pool") or [domain]
        used_fallback: bool = keyword_plan.get("used_fallback", True)

        print(
            f"[FieldScanner] Query pool ({len(query_pool)} queries, "
            f"fallback={used_fallback}): {query_pool}"
        )

        # --- Step 2: Multi-query search + dedup -------------------------
        try:
            top_results, query_hits = self._multi_search(query_pool, per_query_limit=per_query_limit)
        except Exception as e:
            print(f"Warning: Failed to scan OpenAlex ({e}).")
            top_results = []
            query_hits = {q: 0 for q in query_pool}

        # --- Step 3: Concept statistics (high-traction keywords) --------
        concept_counter: Counter[str] = Counter()
        for work in top_results:
            for concept in work.get("concepts", []) or []:
                if isinstance(concept, str):
                    concept_counter[concept] += 1

        high_traction_keywords: List[str] = [
            name for name, _ in concept_counter.most_common(15)
        ]

        scan_status = "full" if top_results else "empty"

        # --- Step 4: Assemble and persist output -----------------------
        scan_results = {
            "domain_scanned": domain,
            "meta": {
                "scan_status": scan_status,
            },
            "search_strategy": {
                "query_pool": query_pool,
                "queries_executed": len(query_pool),
                "used_fallback": used_fallback,
                "primary_domains": keyword_plan.get("primary_domains", []),
                "methods": keyword_plan.get("methods", []),
                "topics": keyword_plan.get("topics", []),
            },
            "query_hits": query_hits,
            "openalex_traction": {
                "top_results": top_results,
            },
            "keywords": {
                "raw_query": domain,
                "high_traction": high_traction_keywords,
            },
        }

        os.makedirs("output", exist_ok=True)
        field_scan_path = "output/field_scan.json"

        with open(field_scan_path, "w", encoding="utf-8") as f:
            json.dump(scan_results, f, indent=2, ensure_ascii=False)

        print(f"Field scan complete. {len(top_results)} unique works collected. Saved to {field_scan_path}")

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

    The full field_scan.json can be large; this keeps only the most
    salient traction signals to reduce prompt noise and enable reuse.
    """
    summary: Dict[str, Any] = {}

    # High-traction keywords (already precomputed in scan_results)
    keywords = (scan_data.get("keywords") or {}).get("high_traction") or []
    summary["high_traction_keywords"] = keywords

    # Top papers with minimal but useful metadata
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

    # Only keep the first 10 for brevity
    summary["top_papers"] = top_papers[:10]

    # Year distribution
    summary["year_distribution"] = {
        str(year): count for year, count in sorted(year_counter.items())
    }

    # Crowded concepts (frequent concepts that may signal crowded subareas)
    crowded_concepts: List[str] = [
        name for name, count in concept_counter.items() if count >= 5
    ]
    summary["crowded_concepts"] = sorted(crowded_concepts)

    # Search strategy summary (new, optional – won't break old callers)
    search_strategy = scan_data.get("search_strategy") or {}
    if search_strategy:
        summary["search_strategy"] = {
            "query_pool": search_strategy.get("query_pool", []),
            "used_fallback": search_strategy.get("used_fallback", True),
            "queries_executed": search_strategy.get("queries_executed", 1),
        }

    # Basic meta
    summary["scan_status"] = (
        (scan_data.get("meta") or {}).get("scan_status") or "unknown"
    )
    summary["domain_scanned"] = scan_data.get("domain_scanned")

    return summary
