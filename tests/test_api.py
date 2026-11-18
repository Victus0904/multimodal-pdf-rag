"""API tests that need no Gemini key: the PDF parser and the RAG calls are stubbed."""

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# `unstructured` is heavy (OCR, layout models); stub the parser module before import.
stub = types.ModuleType("document_parser")
stub.parse_pdf = lambda path: []
stub.save_processed_metadata = lambda elements: None
sys.modules.setdefault("document_parser", stub)

from fastapi.testclient import TestClient  # noqa: E402

import multimodal_api  # noqa: E402
import multimodal_rag  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(multimodal_api, "UPLOAD_DIR", tmp_path)
    return TestClient(multimodal_api.app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ingest_rejects_non_pdf(client):
    r = client.post("/ingest", files={"pdf": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_ingest_sanitises_filename(client, tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    seen = {}

    def fake_ingest(path, settings, debug):
        seen["path"] = Path(path)
        return {"records_indexed": 0}

    monkeypatch.setattr(multimodal_api, "ingest_pdf", fake_ingest)
    r = client.post("/ingest", files={"pdf": ("../../evil.pdf", b"%PDF-1.4", "application/pdf")})
    assert r.status_code == 200
    assert seen["path"].parent == tmp_path and seen["path"].name == "evil.pdf"


def test_ask_validates_input(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "hi", "top_k": 99}).status_code == 422


def test_ask_without_key_returns_500(client, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    r = client.post("/ask", json={"question": "What is in the table?"})
    assert r.status_code == 500
    assert "GEMINI_API_KEY" in r.json()["detail"]


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "256")
    s = multimodal_rag.Settings.from_env()
    assert s.embedding_dimensions == 256 and s.gemini_api_key == "k"
