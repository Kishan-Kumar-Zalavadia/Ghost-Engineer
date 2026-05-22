"""Background processors for GitLab webhook events.

Each processor is a plain async function so it can be handed off to
FastAPI's BackgroundTasks without blocking the HTTP response.
"""

import logging
from datetime import datetime, timezone

from backend import database
from backend.webhooks.models import MergeRequestEvent, PushEvent

logger = logging.getLogger(__name__)


async def process_push_event(event: PushEvent) -> None:
    """Persist every commit from a push event into the 'commits' collection."""
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
        logger.debug("[push] no commit objects in payload, skipping DB write")
        return

    collection = database.get_collection("commits")
    docs = [
        {
            "commit_hash": commit.id,
            "message": commit.message,
            "title": commit.title,
            "timestamp": commit.timestamp,
            "url": commit.url,
            "author_name": commit.author.name,
            "author_email": commit.author.email,
            "branch": event.branch,
            "project_id": event.project.id,
            "project_name": event.project.path_with_namespace,
            "files_added": commit.added,
            "files_modified": commit.modified,
            "files_removed": commit.removed,
            "pushed_by": event.user_name,
            "created_at": datetime.now(timezone.utc),
        }
        for commit in event.commits
    ]

    # updateOne with upsert so re-delivered webhooks are idempotent
    for doc in docs:
        try:
            await collection.update_one(
                {"commit_hash": doc["commit_hash"]},
                {"$setOnInsert": doc},
                upsert=True,
            )
            logger.debug("[push] upserted commit %s", doc["commit_hash"])
        except Exception:
            logger.exception("[push] failed to upsert commit %s", doc["commit_hash"])


async def process_merge_request_event(event: MergeRequestEvent) -> None:
    """Log MR events; extend this to persist or trigger agent actions."""
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
    logger.debug(
        "[merge_request] %s -> %s | url=%s",
        attrs.source_branch,
        attrs.target_branch,
        attrs.url,
    )
