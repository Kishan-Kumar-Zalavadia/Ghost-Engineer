"""Tests for POST /webhooks/gitlab."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient, ASGITransport

from backend.main import app

VALID_TOKEN = "test-secret"

# Minimal valid push payload
PUSH_PAYLOAD = {
    "object_kind": "push",
    "event_name": "push",
    "before": "abc123" + "0" * 34,
    "after": "def456" + "0" * 34,
    "ref": "refs/heads/main",
    "user_name": "jane",
    "user_email": "jane@example.com",
    "user_username": "jane",
    "project": {
        "id": 1,
        "name": "my-repo",
        "path_with_namespace": "acme/my-repo",
        "web_url": "https://gitlab.com/acme/my-repo",
        "default_branch": "main",
    },
    "commits": [
        {
            "id": "a" * 40,
            "message": "fix: something\n",
            "title": "fix: something",
            "timestamp": "2024-01-15T12:00:00+00:00",
            "url": "https://gitlab.com/acme/my-repo/-/commit/" + "a" * 40,
            "author": {"name": "Jane", "email": "jane@example.com"},
            "added": ["new_file.py"],
            "modified": ["existing.py"],
            "removed": [],
        }
    ],
    "total_commits_count": 1,
}

MR_PAYLOAD = {
    "object_kind": "merge_request",
    "event_type": "merge_request",
    "user": {"id": 42, "name": "Jane", "username": "jane"},
    "project": {
        "id": 1,
        "name": "my-repo",
        "path_with_namespace": "acme/my-repo",
        "web_url": "https://gitlab.com/acme/my-repo",
    },
    "object_attributes": {
        "id": 101,
        "iid": 5,
        "title": "Add feature X",
        "state": "opened",
        "action": "open",
        "source_branch": "feature/x",
        "target_branch": "main",
        "url": "https://gitlab.com/acme/my-repo/-/merge_requests/5",
        "author_id": 42,
    },
}


def _mock_app_deps():
    """Patch DB and config so tests run without real infrastructure."""
    return [
        patch("backend.main.database.connect", new_callable=AsyncMock),
        patch("backend.main.database.disconnect", new_callable=AsyncMock),
        patch("backend.webhooks.router.settings") as mock_settings,
    ]


@pytest.mark.asyncio
async def test_push_event_accepted():
    with patch("backend.main.database.connect", new_callable=AsyncMock), \
         patch("backend.main.database.disconnect", new_callable=AsyncMock), \
         patch("backend.webhooks.router.settings") as mock_cfg, \
         patch("backend.webhooks.processors.database.get_collection") as mock_col:

        mock_cfg.gitlab_webhook_token = VALID_TOKEN
        col = MagicMock()
        col.update_one = AsyncMock()
        mock_col.return_value = col

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/webhooks/gitlab",
                content=json.dumps(PUSH_PAYLOAD),
                headers={
                    "X-Gitlab-Token": VALID_TOKEN,
                    "X-Gitlab-Event": "Push Hook",
                    "Content-Type": "application/json",
                },
            )

    assert response.status_code == 200
    assert response.json()["status"] == "accepted"


@pytest.mark.asyncio
async def test_merge_request_event_accepted():
    with patch("backend.main.database.connect", new_callable=AsyncMock), \
         patch("backend.main.database.disconnect", new_callable=AsyncMock), \
         patch("backend.webhooks.router.settings") as mock_cfg:

        mock_cfg.gitlab_webhook_token = VALID_TOKEN

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/webhooks/gitlab",
                content=json.dumps(MR_PAYLOAD),
                headers={
                    "X-Gitlab-Token": VALID_TOKEN,
                    "X-Gitlab-Event": "Merge Request Hook",
                    "Content-Type": "application/json",
                },
            )

    assert response.status_code == 200
    assert response.json()["status"] == "accepted"


@pytest.mark.asyncio
async def test_invalid_token_rejected():
    with patch("backend.main.database.connect", new_callable=AsyncMock), \
         patch("backend.main.database.disconnect", new_callable=AsyncMock), \
         patch("backend.webhooks.router.settings") as mock_cfg:

        mock_cfg.gitlab_webhook_token = VALID_TOKEN

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/webhooks/gitlab",
                content=json.dumps(PUSH_PAYLOAD),
                headers={
                    "X-Gitlab-Token": "wrong-token",
                    "X-Gitlab-Event": "Push Hook",
                    "Content-Type": "application/json",
                },
            )

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_unsupported_event_ignored():
    with patch("backend.main.database.connect", new_callable=AsyncMock), \
         patch("backend.main.database.disconnect", new_callable=AsyncMock), \
         patch("backend.webhooks.router.settings") as mock_cfg:

        mock_cfg.gitlab_webhook_token = VALID_TOKEN

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/webhooks/gitlab",
                content=json.dumps({"object_kind": "pipeline"}),
                headers={
                    "X-Gitlab-Token": VALID_TOKEN,
                    "X-Gitlab-Event": "Pipeline Hook",
                    "Content-Type": "application/json",
                },
            )

    assert response.status_code == 200
    assert response.json()["status"] == "ignored"
