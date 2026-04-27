import json
import os
from typing import Any, Dict, List

from agents import settings
from agents.cache_utils import build_cache_key, load_json_cache, save_json_cache
from agents.logging_config import get_logger
from agents.openalex_utils import (
    configure_openalex,
    download_pdf,
    multi_search_openalex,
)
from agents.arxiv_utils import multi_search_arxiv
from models.research_plan_schema import ResearchPlan

logger = get_logger(__name__)


class LiteratureHarvester:
    def __init__(self):
        configure_openalex()
        self.output_dir = str(settings.literature_dir())
        self.cards_dir = str(settings.literature_cards_dir())
        self.pdfs_dir = str(settings.literature_pdfs_dir())
        self.references_bib = settings.references_bib_path()
        self.index_json = settings.literature_index_path()

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
        try:
            with open(plan_path, "r", encoding="utf-8") as f:
                plan = ResearchPlan.model_validate(json.load(f))
        except Exception as e:
            logger.error("Failed to load plan: %s", e)
            return state

        domain_for_planner = plan.project_title or "geoai"
        query_pool: List[str] = list(plan.literature_queries or [])
        if not query_pool:
            query_pool = [
                f"{plan.research_question} {plan.geography}".strip(),
                f"{plan.exposure.name} {plan.outcome.name} {plan.identification.primary_method}".strip(),
                f"{plan.exposure.name} {plan.outcome.name} open data".strip(),
            ]

        final_limit = int(os.getenv("LITERATURE_FINAL_LIMIT", "3"))
        used_fallback: bool = not bool(plan.literature_queries)

        logger.info(
            "Query pool (%d queries, fallback=%s): %s",
            len(query_pool), used_fallback, query_pool,
        )

        per_query_limit = int(os.getenv("OPENALEX_QUERY_REWRITE_PER_QUERY_LIMIT", "5"))
        papers, query_hits = multi_search_openalex(
            query_pool,
            per_query_limit=per_query_limit,
            final_limit=final_limit,
            cache_prefix="literature_openalex_query",
        )
        queries_used = [q for q, hits in query_hits.items() if hits > 0]

        # ── arXiv supplement ─────────────────────────────────────────────
        arxiv_enabled = os.getenv("ARXIV_SEARCH_ENABLED", "true").lower() != "false"
        arxiv_limit = int(os.getenv("ARXIV_FINAL_LIMIT", str(max(final_limit, 3))))
        arxiv_papers: List[Dict[str, Any]] = []
        if arxiv_enabled:
            try:
                arxiv_papers = multi_search_arxiv(
                    query_pool,
                    per_query_limit=per_query_limit,
                    final_limit=arxiv_limit,
                )
                logger.info("arXiv supplement: %d papers fetched.", len(arxiv_papers))
            except Exception as exc:
                logger.warning("arXiv search failed; skipping supplement: %s", exc)

        # Merge OpenAlex + arXiv, dedup by normalised title
        def _title_key(p: Dict[str, Any]) -> str:
            return "".join(c for c in (p.get("title") or "").lower() if c.isalnum())

        seen_titles: set[str] = {_title_key(p) for p in papers}
        for ap in arxiv_papers:
            key = _title_key(ap)
            if key and key not in seen_titles:
                seen_titles.add(key)
                papers.append(ap)

        # Respect the final_limit across both sources
        papers = papers[:final_limit + arxiv_limit]

        plan_terms = {
            term.lower()
            for term in [plan.exposure.name, plan.outcome.name, plan.research_question]
            if isinstance(term, str) and term
        }
        inventory = []
        for paper in papers:
            card = self.save_paper_info(paper)
            source_db = paper.get("source") or ("arxiv" if str(paper.get("paperId", "")).startswith("arxiv:") else "openalex")
            matched_query = ""
            title_abstract = f"{card.get('title', '')} {card.get('abstract', '')}".lower()
            overlap_score = 0
            for query in query_pool:
                query_terms = [q for q in query.lower().split() if len(q) > 2]
                score = sum(1 for q in query_terms if q in title_abstract)
                if score > overlap_score:
                    overlap_score = score
                    matched_query = query
            relevance = "low"
            if overlap_score >= 3 or sum(1 for t in plan_terms if t in title_abstract) >= 2:
                relevance = "high"
            elif overlap_score >= 1:
                relevance = "medium"
            card["matched_query"] = matched_query or (query_pool[0] if query_pool else "")
            card["relevance_to_plan"] = relevance
            card["source_database"] = source_db
            card_path = os.path.join(self.cards_dir, f"{card['citation_key']}.json")
            with open(card_path, "w", encoding="utf-8") as f:
                json.dump(card, f, indent=2, ensure_ascii=False)
            inventory.append(card)

        with open(self.index_json, "w", encoding="utf-8") as f:
            json.dump(inventory, f, indent=2, ensure_ascii=False)

        context_path = state.get("research_context_path", settings.research_context_path())
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
