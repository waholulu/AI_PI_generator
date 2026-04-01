import json
import os
from typing import Any, Dict, List

from tenacity import retry, stop_after_attempt, wait_exponential

from agents import settings
from agents.cache_utils import build_cache_key, load_json_cache, save_json_cache
from agents.logging_config import get_logger
from agents.openalex_utils import (
    configure_openalex,
    download_pdf,
    extract_work_metadata,
)
from agents.keyword_planner import KeywordPlanner

logger = get_logger(__name__)

try:
    from pyalex import Works
except ImportError:  # pragma: no cover - exercised in dependency-light test envs
    Works = None


class LiteratureHarvester:
    def __init__(self):
        configure_openalex()
        self.output_dir = str(settings.literature_dir())
        self.cards_dir = str(settings.literature_cards_dir())
        self.pdfs_dir = str(settings.literature_pdfs_dir())
        self.references_bib = settings.references_bib_path()
        self.index_json = settings.literature_index_path()
        self._planner = KeywordPlanner()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=30))
    def search_openalex(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Searches OpenAlex for works matching the query."""
        logger.info("Querying OpenAlex for: %s", query)
        if Works is None:
            logger.warning("OpenAlex search skipped: pyalex is not installed.")
            return []
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
                if not metadata["paperId"]:
                    metadata["paperId"] = f"oa_{len(results)}"
                results.append(metadata)
            return results
        except Exception as e:
            logger.error("OpenAlex API Error: %s", e)
            return []

    def _multi_search(
        self,
        query_pool: List[str],
        final_limit: int = 3,
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        """
        Run search_openalex for each query in *query_pool*, deduplicate by
        openalex_id, and return up to *final_limit* papers.
        """
        per_query_limit = int(os.getenv("OPENALEX_QUERY_REWRITE_PER_QUERY_LIMIT", "5"))
        seen_ids: set[str] = set()
        merged: List[Dict[str, Any]] = []
        queries_used: List[str] = []
        cache_hours = float(os.getenv("OPENALEX_CACHE_HOURS", "48"))

        for query in query_pool:
            cache_key = build_cache_key(
                "literature_openalex_query",
                {"query": query, "limit": per_query_limit},
            )
            try:
                cached = load_json_cache("openalex", cache_key, max_age_hours=cache_hours)
                if isinstance(cached, list):
                    results = cached
                else:
                    results = self.search_openalex(query, limit=per_query_limit)
                    save_json_cache("openalex", cache_key, results)
            except Exception as exc:
                logger.warning("Query '%s' failed: %s", query, exc)
                continue

            if not results:
                continue

            queries_used.append(query)
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

            if len(merged) >= final_limit * 3:
                break

        return merged[:final_limit], queries_used

    def save_paper_info(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Generates evidence card and updates indices."""
        title = str(paper.get("title") or "Unknown")
        year = str(paper.get("year") or "Unknown")
        paper_id = paper.get("paperId", "unknown")

        safe_title = "".join([c for c in title if c.isalnum()]).lower()[:15]
        citation_key = f"paper_{year}_{safe_title}"

        pdf_result = {"status": "pdf_unavailable", "path": None, "reason": "missing_url"}
        candidate_urls = paper.get("candidate_download_urls", [])
        for url in candidate_urls:
            pdf_result = download_pdf(url, os.path.join(self.pdfs_dir, f"{citation_key}.pdf"))
            if pdf_result["status"] == "pdf_downloaded":
                break

        card = {
            "citation_key": citation_key,
            "paperId": paper_id,
            "openalex_id": paper.get("openalex_id"),
            "title": title,
            "authors": [a.get("name") for a in paper.get("authors", [])],
            "author_details": paper.get("authors", []),
            "year": year,
            "publication_date": paper.get("publication_date"),
            "type": paper.get("type"),
            "language": paper.get("language"),
            "doi": paper.get("doi"),
            "journal": paper.get("journal"),
            "journal_id": paper.get("journal_id"),
            "host_organization": paper.get("host_organization"),
            "abstract": paper.get("abstract", ""),
            "citationCount": paper.get("citationCount", 0),
            "isOpenAccess": paper.get("isOpenAccess", False),
            "oa_status": paper.get("oa_status"),
            "oa_url": paper.get("oa_url"),
            "landing_page_url": paper.get("landing_page_url"),
            "pdf_url": paper.get("pdf_url"),
            "candidate_download_urls": candidate_urls,
            "biblio": paper.get("biblio", {}),
            "concepts": paper.get("concepts", []),
            "referenced_works": paper.get("referenced_works", []),
            "referenced_works_count": paper.get("referenced_works_count", 0),
            "is_retracted": paper.get("is_retracted", False),
            "pdf_status": pdf_result["status"],
            "pdf_path": pdf_result["path"],
            "pdf_reason": pdf_result["reason"],
        }

        card_path = os.path.join(self.cards_dir, f"{citation_key}.json")
        with open(card_path, "w", encoding="utf-8") as f:
            json.dump(card, f, indent=2, ensure_ascii=False)

        return card

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Executes the literature harvesting."""
        logger.info("--- Module 2: Literature Harvester ---")

        plan_path = state.get("current_plan_path", settings.research_plan_path())
        context_path = state.get("research_context_path", settings.research_context_path())
        try:
            with open(plan_path, "r", encoding="utf-8") as f:
                plan = json.load(f)
        except Exception as e:
            logger.error("Failed to load plan: %s", e)
            return state

        context: Dict[str, Any] = {}
        if os.path.exists(context_path):
            try:
                with open(context_path, "r", encoding="utf-8") as f:
                    context = json.load(f)
            except Exception as e:
                logger.warning("Failed to load research context: %s", e)

        extra_parts: List[str] = []
        selected = context.get("selected_topic", {}) if isinstance(context, dict) else {}
        title = selected.get("title")
        if isinstance(title, str) and title:
            extra_parts.append(f"Selected topic: {title}")

        plan_ess = context.get("plan_essentials", {}) if isinstance(context, dict) else {}
        outcomes = plan_ess.get("outcomes", [])
        for outcome in outcomes[:2]:
            if isinstance(outcome, dict):
                var = outcome.get("variable") or outcome.get("name")
                if isinstance(var, str) and var:
                    extra_parts.append(f"Outcome: {var}")
            elif isinstance(outcome, str) and outcome:
                extra_parts.append(f"Outcome: {outcome}")

        extra_context = "; ".join(extra_parts)

        domain_for_planner = (
            title
            or " ".join(
                kw for kw in (plan.get("keywords") or [])[:3] if isinstance(kw, str) and kw
            )
            or context.get("domain", "geoai")
            or "geoai"
        )

        final_limit = int(os.getenv("LITERATURE_FINAL_LIMIT", "3"))
        plan_cache_hours = float(os.getenv("KEYWORD_PLAN_CACHE_HOURS", "24"))
        plan_cache_key = build_cache_key(
            "literature_keyword_plan",
            {"domain_for_planner": domain_for_planner, "extra_context": extra_context},
        )
        keyword_plan = load_json_cache("keyword_plans", plan_cache_key, max_age_hours=plan_cache_hours)
        if not isinstance(keyword_plan, dict):
            keyword_plan = self._planner.plan(domain_for_planner, extra_context=extra_context)
            save_json_cache("keyword_plans", plan_cache_key, keyword_plan)
        query_pool: List[str] = keyword_plan.get("query_pool") or [domain_for_planner]
        used_fallback: bool = keyword_plan.get("used_fallback", True)

        logger.info(
            "Query pool (%d queries, fallback=%s): %s",
            len(query_pool), used_fallback, query_pool,
        )

        papers, queries_used = self._multi_search(query_pool, final_limit=final_limit)

        inventory = []
        for paper in papers:
            card = self.save_paper_info(paper)
            inventory.append(card)

        with open(self.index_json, "w", encoding="utf-8") as f:
            json.dump(inventory, f, indent=2, ensure_ascii=False)

        if os.path.exists(context_path):
            try:
                with open(context_path, "r", encoding="utf-8") as f:
                    ctx = json.load(f)
            except Exception as e:
                logger.warning("Failed to reload research context for literature summary: %s", e)
                ctx = {}

            if isinstance(ctx, dict):
                ctx["literature_summary"] = {
                    "query_used": queries_used[0] if queries_used else domain_for_planner,
                    "queries_used": queries_used,
                    "query_pool": query_pool,
                    "used_fallback": used_fallback,
                    "paper_count": len(inventory),
                    "citation_keys": [item.get("citation_key") for item in inventory],
                }
                try:
                    with open(context_path, "w", encoding="utf-8") as f:
                        json.dump(ctx, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    logger.warning("Failed to write literature summary into research context: %s", e)

        with open(self.references_bib, "w", encoding="utf-8") as f:
            f.write("% Generated BibTeX\n")
            for item in inventory:
                authors = " and ".join(item.get("authors", []))
                f.write(
                    f"@article{{{item['citation_key']},\n"
                    f"  title={{{item['title']}}},\n"
                    f"  author={{{authors}}},\n"
                    f"  journal={{{item.get('journal') or ''}}},\n"
                    f"  year={{{item['year']}}},\n"
                    f"  doi={{{item.get('doi') or ''}}}\n"
                    f"}}\n\n"
                )

        logger.info("Literature harvested. %d items saved.", len(inventory))
        return {
            "execution_status": "drafting",
            "literature_inventory_path": self.index_json,
        }


def literature_node(state: Dict[str, Any]) -> Dict[str, Any]:
    harvester = LiteratureHarvester()
    return harvester.run(state)
