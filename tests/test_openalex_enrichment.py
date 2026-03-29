import json
import os
import tempfile
from unittest.mock import patch

from agents.literature_agent import LiteratureHarvester
from agents.openalex_utils import extract_work_metadata, reconstruct_abstract


def test_openalex_enrichment_flow():
    sample_work = {
        "id": "https://openalex.org/W1234567890",
        "display_name": "Urban AI for Climate Adaptation",
        "doi": "https://doi.org/10.1234/example",
        "publication_year": 2024,
        "publication_date": "2024-03-01",
        "type": "article",
        "language": "en",
        "cited_by_count": 42,
        "authorships": [
            {
                "author": {
                    "display_name": "Alice Zhang",
                    "id": "https://openalex.org/A1",
                    "orcid": "https://orcid.org/0000-0001",
                },
                "institutions": [
                    {
                        "display_name": "Example University",
                        "id": "https://openalex.org/I1",
                        "country_code": "US",
                    }
                ],
            }
        ],
        "abstract_inverted_index": {
            "Urban": [0],
            "AI": [1],
            "supports": [2],
            "adaptation": [3],
        },
        "open_access": {
            "is_oa": True,
            "oa_status": "gold",
            "oa_url": "https://example.org/oa",
        },
        "primary_location": {
            "landing_page_url": "https://example.org/landing",
            "pdf_url": "https://example.org/paper.pdf",
            "source": {
                "display_name": "Journal of Urban AI",
                "id": "https://openalex.org/S1",
                "host_organization_name": "Example Press",
            },
        },
        "best_oa_location": {
            "pdf_url": "https://example.org/best.pdf",
        },
        "locations": [
            {
                "landing_page_url": "https://mirror.example.org/landing",
            }
        ],
        "biblio": {"volume": "12", "issue": "3", "first_page": "1", "last_page": "18"},
        "concepts": [
            {"id": "C1", "display_name": "Artificial intelligence", "level": 0, "score": 0.9},
            {"id": "C2", "display_name": "Urban planning", "level": 1, "score": 0.8},
            {"id": "C3", "display_name": "Climate adaptation", "level": 2, "score": 0.7},
        ],
        "referenced_works": ["https://openalex.org/W2", "https://openalex.org/W3"],
        "referenced_works_count": 2,
        "is_retracted": False,
        "has_fulltext": True,
    }

    assert reconstruct_abstract(sample_work) == "Urban AI supports adaptation"

    metadata = extract_work_metadata(sample_work)
    assert metadata["title"] == "Urban AI for Climate Adaptation"
    assert metadata["paperId"] == "W1234567890"
    assert metadata["first_author"] == "Alice Zhang"
    assert metadata["journal"] == "Journal of Urban AI"
    assert metadata["pdf_url"] == "https://example.org/paper.pdf"
    assert "https://example.org/best.pdf" in metadata["candidate_download_urls"]
    assert metadata["concepts"][0]["name"] == "Artificial intelligence"

    with tempfile.TemporaryDirectory() as tmp_dir:
        harvester = LiteratureHarvester()
        harvester.output_dir = tmp_dir
        harvester.cards_dir = os.path.join(tmp_dir, "cards")
        harvester.pdfs_dir = os.path.join(tmp_dir, "pdfs")
        os.makedirs(harvester.cards_dir, exist_ok=True)
        os.makedirs(harvester.pdfs_dir, exist_ok=True)

        expected_pdf_path = os.path.join(harvester.pdfs_dir, "paper_2024_urbanaiforclima.pdf")
        fake_download = {
            "status": "pdf_downloaded",
            "path": expected_pdf_path,
            "reason": None,
        }

        with patch("agents.literature_agent.download_pdf", return_value=fake_download):
            card = harvester.save_paper_info(metadata)

        assert card["citation_key"] == "paper_2024_urbanaiforclima"
        assert card["doi"] == "https://doi.org/10.1234/example"
        assert card["journal"] == "Journal of Urban AI"
        assert card["pdf_status"] == "pdf_downloaded"
        assert card["pdf_path"] == expected_pdf_path
        assert card["authors"] == ["Alice Zhang"]

        card_path = os.path.join(harvester.cards_dir, f"{card['citation_key']}.json")
        assert os.path.exists(card_path)

        with open(card_path, "r", encoding="utf-8") as file_obj:
            saved_card = json.load(file_obj)

        assert saved_card["oa_status"] == "gold"
        assert saved_card["referenced_works_count"] == 2
        assert saved_card["concepts"][2]["name"] == "Climate adaptation"


if __name__ == "__main__":
    test_openalex_enrichment_flow()
    print("test_openalex_enrichment_flow passed!")
