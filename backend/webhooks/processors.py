"""Background processors for GitLab webhook events.

Each processor is a plain async function so it can be handed off to
FastAPI's BackgroundTasks without blocking the HTTP response.
"""

import logging

from backend.modules import orchestrator
from backend.webhooks.models import MergeRequestEvent, PushEvent

logger = logging.getLogger(__name__)


async def process_push_event(event: PushEvent) -> None:
    """Route a push event through the orchestrator for full analysis."""
    if event.is_deletion:
        logger.info(
            "[push] branch deleted | project=%s branch=%s user=%s",
            event.project.path_with_namespace,
            event.branch,
            event.user_name,
        )
        return

    logger.info(
        "[push] project=%s branch=%s commits=%d user=%s",
        event.project.path_with_namespace,
        event.branch,
        event.total_commits_count,
        event.user_name,
    )

    if not event.commits:
        logger.debug("[push] no commit objects in payload, skipping")
        return

    result = await orchestrator.process_new_commit(event.model_dump(mode="json"))
    logger.info(
        "[push] orchestrator processed %d commits | branch=%s",
        result.get("commits_processed", 0),
        result.get("branch"),
    )


async def process_merge_request_event(event: MergeRequestEvent) -> None:
    """Route an MR event through the orchestrator for full analysis."""
    attrs = event.object_attributes
    logger.info(
        "[merge_request] action=%s state=%s iid=%s title=%r project=%s user=%s",
        attrs.action,
        attrs.state,
        attrs.iid,
        attrs.title,
        event.project.path_with_namespace,
        event.user.username,
    )

    result = await orchestrator.process_new_merge_request(event.model_dump(mode="json"))
    logger.info(
        "[merge_request] orchestrator result | risk=%.1f findings=%d ghost_comment=%s",
        result.get("risk_score", 0),
        result.get("total_findings", 0),
        result.get("ghost_comment_prepared", False),
    )
