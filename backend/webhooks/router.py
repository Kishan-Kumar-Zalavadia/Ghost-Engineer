"""GitLab webhook router.

POST /webhooks/gitlab
  - Verifies X-Gitlab-Token against GITLAB_WEBHOOK_TOKEN env var
  - Always returns 200 immediately so GitLab never times out
  - When Google Cloud Pub/Sub is configured: publishes event to
    the 'ghost-engineer-events' topic; the push subscriber handles processing
  - When Pub/Sub is not configured (local dev): dispatches to background
    processors via FastAPI BackgroundTasks
"""

import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request, status
from pydantic import ValidationError

from backend.config import settings
from backend.webhooks.models import MergeRequestEvent, PushEvent
from backend.webhooks.processors import process_merge_request_event, process_push_event

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["webhooks"])

_SUPPORTED_EVENTS = {"push", "tag_push", "merge_request"}


def _verify_token(x_gitlab_token: str = Header(...)) -> None:
    """Dependency: reject requests with a wrong or missing secret token."""
    if x_gitlab_token != settings.gitlab_webhook_token:
        logger.warning("[webhook] invalid X-Gitlab-Token received")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )


@router.post(
    "/gitlab",
    status_code=status.HTTP_200_OK,
    summary="GitLab webhook receiver",
)
async def gitlab_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_gitlab_token: str = Header(...),
    x_gitlab_event: str = Header(...),
) -> dict[str, str]:
    # 1. Authenticate
    _verify_token(x_gitlab_token)

    # 2. Read raw body
    raw: dict[str, Any] = await request.json()
    event_kind = raw.get("object_kind", "")
    project_id: int = raw.get("project", {}).get("id", settings.gitlab_project_id)

    logger.info(
        "[webhook] received | x_gitlab_event=%r object_kind=%r project_id=%s",
        x_gitlab_event, event_kind, project_id,
    )

    if event_kind not in _SUPPORTED_EVENTS:
        logger.info("[webhook] unsupported event_kind=%r — acknowledged", event_kind)
        return {"status": "ignored", "reason": f"event '{event_kind}' not handled"}

    # 3. Try Pub/Sub first (production path — ensures immediate 200 to GitLab)
    from backend import pubsub as pubsub_module  # late import avoids circular dep

    if pubsub_module.is_configured():
        published = await pubsub_module.publish_webhook_event(
            event_kind=event_kind,
            project_id=project_id,
            payload=raw,
        )
        if published:
            logger.info(
                "[webhook] event published to Pub/Sub | kind=%s project=%s",
                event_kind, project_id,
            )
            return {"status": "accepted", "via": "pubsub"}
        else:
            logger.warning(
                "[webhook] Pub/Sub publish failed, falling back to BackgroundTasks"
            )

    # 4. Fallback: BackgroundTasks (local dev / Pub/Sub unavailable)
    if event_kind in ("push", "tag_push"):
        try:
            event = PushEvent.model_validate(raw)
        except ValidationError:
            logger.exception("[webhook] failed to parse push payload")
            return {"status": "ignored", "reason": "payload parse error"}
        background_tasks.add_task(process_push_event, event)

    elif event_kind == "merge_request":
        try:
            event = MergeRequestEvent.model_validate(raw)
        except ValidationError:
            logger.exception("[webhook] failed to parse merge_request payload")
            return {"status": "ignored", "reason": "payload parse error"}
        background_tasks.add_task(process_merge_request_event, event)

    logger.info("[webhook] event dispatched to BackgroundTasks | kind=%s", event_kind)
    return {"status": "accepted", "via": "background_tasks"}
