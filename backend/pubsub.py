"""Google Cloud Pub/Sub integration for GhostEngineer.

Decouples webhook receipt from processing — GitLab always gets a 200 within
milliseconds, and the real work (diff fetch, Gemini calls, MongoDB writes)
happens when Pub/Sub delivers the message to the push subscriber endpoint.

When ``GOOGLE_CLOUD_PROJECT`` and ``PUBSUB_TOPIC`` are not configured
(local development), ``publish_webhook_event`` is a no-op and the caller
should fall back to FastAPI BackgroundTasks.

Public API
----------
is_configured()                    → bool
publish_webhook_event(...)         → bool
decode_pubsub_message(body)        → dict | None
route_pubsub_event(event)          → Awaitable[dict]
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from datetime import datetime, timezone
from typing import Any

from backend.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy publisher singleton
# ---------------------------------------------------------------------------

_publisher: Any = None
_topic_path: str = ""


def _get_publisher() -> tuple[Any, str]:
    global _publisher, _topic_path
    if _publisher is None:
        from google.cloud import pubsub_v1  # type: ignore
        _publisher = pubsub_v1.PublisherClient()
        _topic_path = _publisher.topic_path(
            settings.google_cloud_project,
            settings.pubsub_topic,
        )
        logger.info("[pubsub] publisher initialised | topic=%s", _topic_path)
    return _publisher, _topic_path


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def is_configured() -> bool:
    """Return True if Pub/Sub is enabled (Google Cloud project is set)."""
    return bool(settings.google_cloud_project and settings.pubsub_topic)


async def publish_webhook_event(
    event_kind: str,
    project_id: int,
    payload: dict,
) -> bool:
    """
    Publish a GitLab webhook event to the Pub/Sub topic.

    Parameters
    ----------
    event_kind:
        ``"push"``, ``"tag_push"``, or ``"merge_request"``.
    project_id:
        GitLab numeric project ID (extracted from the payload before calling).
    payload:
        The raw webhook payload dict.

    Returns
    -------
    bool  True if the message was published successfully.
    """
    if not is_configured():
        return False

    message_body = {
        "event_kind": event_kind,
        "project_id": project_id,
        "payload": payload,
        "received_at": datetime.now(timezone.utc).isoformat(),
    }
    data = json.dumps(message_body, default=str).encode("utf-8")

    def _sync_publish() -> str:
        pub, topic = _get_publisher()
        future = pub.publish(
            topic,
            data=data,
            event_kind=event_kind,
            project_id=str(project_id),
        )
        return future.result(timeout=10)

    try:
        msg_id = await asyncio.to_thread(_sync_publish)
        logger.info(
            "[pubsub] published | event_kind=%s project_id=%s msg_id=%s",
            event_kind, project_id, msg_id,
        )
        return True
    except Exception as exc:
        logger.error(
            "[pubsub] publish failed | event_kind=%s project_id=%s error=%s",
            event_kind, project_id, exc,
        )
        return False


def decode_pubsub_message(body: dict) -> dict | None:
    """
    Decode a Pub/Sub push-delivery request body into our event dict.

    Pub/Sub push format::

        {
          "message": {
            "data": "<base64-encoded JSON>",
            "messageId": "...",
            "publishTime": "..."
          },
          "subscription": "projects/.../subscriptions/..."
        }

    Returns None if the message cannot be decoded.
    """
    try:
        message = body.get("message", {})
        data_b64 = message.get("data", "")
        if not data_b64:
            logger.warning("[pubsub] received message with empty data field")
            return None
        raw = base64.b64decode(data_b64).decode("utf-8")
        return json.loads(raw)
    except Exception as exc:
        logger.error("[pubsub] failed to decode push message: %s", exc)
        return None


async def route_pubsub_event(event: dict) -> dict:
    """
    Route a decoded Pub/Sub event to the correct orchestrator function.

    Called by the ``POST /pubsub/receive`` endpoint after decoding the message.

    Parameters
    ----------
    event:
        The decoded event dict produced by ``decode_pubsub_message``.

    Returns
    -------
    dict  The orchestrator result (used only for logging; Pub/Sub ignores it).
    """
    # Import here to avoid circular imports at module load time
    from backend.modules import orchestrator

    event_kind: str = event.get("event_kind", "")
    project_id: int = event.get("project_id", 0)
    payload: dict = event.get("payload", {})

    logger.info(
        "[pubsub] routing event | kind=%s project_id=%s",
        event_kind, project_id,
    )

    if event_kind in ("push", "tag_push"):
        return await orchestrator.process_new_commit(payload)
    elif event_kind == "merge_request":
        return await orchestrator.process_new_merge_request(payload)
    else:
        logger.warning("[pubsub] unknown event_kind=%r — ignored", event_kind)
        return {"status": "ignored", "event_kind": event_kind}
