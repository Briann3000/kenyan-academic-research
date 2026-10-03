"""Tests for OpenAlex client rate limiting, retries, and pagination."""

import pytest
from unittest.mock import patch, MagicMock
import requests

from src.ingestion.openalex_client import OpenAlexClient


def test_client_init():
    client = OpenAlexClient(base_url="https://api.openalex.org", max_req_per_sec=2.0)
    assert client.base_url == "https://api.openalex.org"
    assert client.min_interval == 0.5


@patch("requests.Session.get")
def test_client_get_success(mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"results": [{"id": "W123", "title": "Test Paper"}]}
    mock_get.return_value = mock_resp

    client = OpenAlexClient()
    data = client.get("works", params={"filter": "country_code:KE"})

    assert "results" in data
    assert data["results"][0]["id"] == "W123"
    assert mock_get.call_count == 1


@patch("requests.Session.get")
def test_client_retry_on_429(mock_get):
    # Mock 429 followed by 200
    mock_429 = MagicMock()
    mock_429.status_code = 429

    mock_200 = MagicMock()
    mock_200.status_code = 200
    mock_200.json.return_value = {"results": []}

    mock_get.side_effect = [mock_429, mock_200]

    client = OpenAlexClient(max_retries=3, backoff_factor=0.01)
    data = client.get("works")

    assert data == {"results": []}
    assert mock_get.call_count == 2


@patch("requests.Session.get")
def test_client_pagination(mock_get):
    page1 = MagicMock()
    page1.status_code = 200
    page1.json.return_value = {
        "meta": {"next_cursor": "cursor_page_2"},
        "results": [{"id": f"W{i}"} for i in range(5)]
    }

    page2 = MagicMock()
    page2.status_code = 200
    page2.json.return_value = {
        "meta": {"next_cursor": None},
        "results": [{"id": f"W{i}"} for i in range(5, 8)]
    }

    mock_get.side_effect = [page1, page2]

    client = OpenAlexClient()
    records = list(client.paginate("works", max_records=10, per_page=5))

    assert len(records) == 8
    assert records[0]["id"] == "W0"
    assert records[-1]["id"] == "W7"
