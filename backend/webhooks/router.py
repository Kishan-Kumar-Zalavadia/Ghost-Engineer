"""GitLab webhook router.

POST /webhooks/gitlab
  - Verifies X-Gitlab-Token against GITLAB_WEBHOOK_TOKEN env var
  - Dispatches push / merge_request events to background processors
  - Always returns 200 within the current request; processing is non-blocking
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
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


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
    # --- 1. Authenticate ---
    _verify_token(x_gitlab_token)

    # --- 2. Read raw body (needed for future HMAC if desired) ---
    raw: dict[str, Any] = await request.json()

    event_kind = raw.get("object_kind", "")
    logger.info("[webhook] received event x_gitlab_event=%r object_kind=%r", x_gitlab_event, event_kind)

    # --- 3. Normalise event type ---
    if event_kind not in _SUPPORTED_EVENTS:
        logger.info("[webhook] unsupported event_kind=%r — acknowledged and ignored", event_kind)
        return {"status": "ignored", "reason": f"event '{event_kind}' not handled"}

    # --- 4. Dispatch to background — response returns immediately ---
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

    logger.debug("[webhook] event dispatched to background | kind=%s", event_kind)
    return {"status": "accepted"}
