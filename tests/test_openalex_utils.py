import os
from typing import Any, Dict, List

import pytest

from agents import openalex_utils


def test_configure_openalex_applies_env_to_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """configure_openalex should push env settings into pyalex.config when available."""
    class _Config:
        api_key: str | None = None
        email: str | None = None
        max_retries: int = 0
        retry_backoff_factor: float = 0.0
        retry_http_codes: List[int] = []

    class _PyAlex:
        config = _Config()

    monkeypatch.setattr(openalex_utils, "pyalex", _PyAlex())  # type: ignore[arg-type]
    monkeypatch.setenv("OPENALEX_API_KEY", "test-key")
    monkeypatch.setenv("OPENALEX_EMAIL", "user@example.com")

    openalex_utils.configure_openalex()

    cfg = openalex_utils.pyalex.config  # type: ignore[assignment]
    assert cfg.api_key == "test-key"
    assert cfg.email == "user@example.com"
    assert cfg.max_retries == 3
    assert cfg.retry_backoff_factor == 0.5
    assert cfg.retry_http_codes == [429, 500, 503]


def test_reconstruct_abstract_uses_index_when_no_plain_text() -> None:
    work: Dict[str, Any] = {
        "abstract_inverted_index": {
            "GeoAI": [1],
            "Urban": [0],
            "planning": [2],
        }
    }
    text = openalex_utils.reconstruct_abstract(work)
    assert text == "Urban GeoAI planning"


def test_reconstruct_abstract_returns_existing_abstract_verbatim() -> None:
    work = {"abstract": "Existing abstract", "abstract_inverted_index": {"ignored": [0]}}
    assert openalex_utils.reconstruct_abstract(work) == "Existing abstract"


def test_extract_concepts_filters_by_level_and_limit() -> None:
    work = {
        "concepts": [
            {"id": "C1", "display_name": "Level0", "level": 0, "score": 0.9},
            {"id": "C2", "display_name": "Level1", "level": 1, "score": 0.8},
            {"id": "C3", "display_name": "Level2", "level": 2, "score": 0.7},
        ]
    }
    broad = openalex_utils.extract_concepts(work, max_items=2, max_level=1)
    assert len(broad) == 2
    assert all(item["level"] <= 1 for item in broad if isinstance(item["level"], int))


def test_extract_work_metadata_builds_expected_fields() -> None:
    work: Dict[str, Any] = {
        "id": "https://openalex.org/W1",
        "display_name": "Test Work",
        "doi": "10.1234/test",
        "publication_year": 2024,
        "publication_date": "2024-01-01",
        "type": "article",
        "language": "en",
        "cited_by_count": 5,
        "authorships": [
            {
                "author": {"display_name": "Author One", "id": "https://openalex.org/A1", "orcid": None},
                "institutions": [],
            }
        ],
        "open_access": {"is_oa": True, "oa_status": "gold", "oa_url": "https://example.org/oa"},
        "primary_location": {
            "landing_page_url": "https://example.org/landing",
            "pdf_url": "https://example.org/paper.pdf",
            "source": {
                "display_name": "Journal Test",
                "id": "https://openalex.org/S1",
                "host_organization_name": "Host Org",
            },
        },
        "concepts": [],
        "referenced_works": [],
        "referenced_works_count": 0,
        "is_retracted": False,
    }

    meta = openalex_utils.extract_work_metadata(work)
    assert meta["title"] == "Test Work"
    assert meta["paperId"] == "W1"
    assert meta["first_author"] == "Author One"
    assert meta["journal"] == "Journal Test"
    assert meta["pdf_url"] == "https://example.org/paper.pdf"
    assert meta["isOpenAccess"] is True


class _FakeResponse:
    def __init__(self, status_code: int, headers: Dict[str, str], body: bytes) -> None:
        self.status_code = status_code
        self.headers = headers
        self._body = body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size: int = 8192):
        yield self._body

    def close(self) -> None:  # pragma: no cover - trivial
        pass


def test_download_pdf_success(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    def _fake_get(url: str, timeout: int, stream: bool):
        assert url.endswith(".pdf")
        return _FakeResponse(
            200,
            headers={"content-type": "application/pdf"},
            body=b"%PDF-1.4 data",
        )

    monkeypatch.setattr(openalex_utils.requests, "get", _fake_get)  # type: ignore[arg-type]

    target = tmp_path / "paper.pdf"
    result = openalex_utils.download_pdf("https://example.org/paper.pdf", str(target))
    assert result["status"] == "pdf_downloaded"
    assert os.path.exists(target)


def test_download_pdf_non_pdf_content_type(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    def _fake_get(url: str, timeout: int, stream: bool):
        return _FakeResponse(
            200,
            headers={"content-type": "text/html"},
            body=b"<html>Not a PDF</html>",
        )

    monkeypatch.setattr(openalex_utils.requests, "get", _fake_get)  # type: ignore[arg-type]

    target = tmp_path / "paper.pdf"
    result = openalex_utils.download_pdf("https://example.org/page", str(target))
    assert result["status"] == "pdf_not_direct"
    assert result["path"] is None
    assert not target.exists()


def test_download_pdf_handles_http_error(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    def _fake_get(url: str, timeout: int, stream: bool):
        return _FakeResponse(
            500,
            headers={"content-type": "application/pdf"},
            body=b"error",
        )

    monkeypatch.setattr(openalex_utils.requests, "get", _fake_get)  # type: ignore[arg-type]

    target = tmp_path / "paper.pdf"
    result = openalex_utils.download_pdf("https://example.org/fail.pdf", str(target))
    assert result["status"] == "pdf_download_failed"
    assert result["path"] is None
    assert not target.exists()


def test_download_pdf_missing_url_returns_unavailable(tmp_path) -> None:
    target = tmp_path / "paper.pdf"
    result = openalex_utils.download_pdf("", str(target))
    assert result["status"] == "pdf_unavailable"
    assert result["path"] is None
    assert not target.exists()

