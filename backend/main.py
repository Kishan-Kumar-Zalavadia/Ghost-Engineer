from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from bson import ObjectId
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend import database
from backend.config import settings
from backend.modules import orchestrator, proactive_ghost
from backend.webhooks.router import router as webhooks_router


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class GhostFeedback(BaseModel):
    action_id: str
    was_helpful: bool
    reason: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    await database.connect()
    yield
    await database.disconnect()


app = FastAPI(
    title="Ghost Engineer API",
    version="0.1.0",
    description="AI-powered engineering assistant",
    lifespan=lifespan,
)

app.include_router(webhooks_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    db_ok = await database.ping()
    return {
        "status": "ok" if db_ok else "degraded",
        "env": settings.app_env,
        "database": "connected" if db_ok else "unreachable",
    }


@app.post("/initialize", status_code=status.HTTP_200_OK, tags=["orchestrator"])
async def initialize_repository(project_id: int):
    """
    Trigger a full initial scan for a GitLab project.

    Fetches all commits and MRs, runs pattern detection on every commit,
    and builds developer profiles for every unique author found.
    """
    try:
        result = await orchestrator.initialize_repository(project_id)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )
    return result


@app.get("/status", tags=["orchestrator"])
async def get_status():
    """
    Return current counts from MongoDB:
    - total commits analyzed
    - total patterns found
    - developers profiled
    - ghost actions taken
    """
    commits_col = database.get_collection("commits")
    profiles_col = database.get_collection("developer_profiles")
    ghost_actions_col = database.get_collection("ghost_actions")

    total_commits = await commits_col.count_documents({})

    # Commits that have gone through pattern detection
    commits_analyzed = await commits_col.count_documents(
        {"patterns_detected": {"$exists": True}}
    )

    # Total individual findings across all analyzed commits
    pipeline = [
        {"$match": {"patterns_detected.findings": {"$exists": True}}},
        {"$project": {"finding_count": {"$size": "$patterns_detected.findings"}}},
        {"$group": {"_id": None, "total": {"$sum": "$finding_count"}}},
    ]
    patterns_found = 0
    async for doc in commits_col.aggregate(pipeline):
        patterns_found = doc.get("total", 0)

    developers_profiled = await profiles_col.count_documents({})
    ghost_actions_taken = await ghost_actions_col.count_documents({})

    return {
        "total_commits": total_commits,
        "commits_analyzed": commits_analyzed,
        "patterns_found": patterns_found,
        "developers_profiled": developers_profiled,
        "ghost_actions_taken": ghost_actions_taken,
    }


# ---------------------------------------------------------------------------
# Ghost endpoints
# ---------------------------------------------------------------------------

@app.post("/ghost/scan-now", tags=["ghost"])
async def ghost_scan_now(project_id: int):
    """
    Manually trigger a proactive Ghost scan for the given project.

    Runs security → duplicate code → documentation checks in priority order.
    Respects the 4-hour cooldown — if a proactive MR was already opened
    recently the scan is skipped and the response will say so.
    """
    try:
        result = await proactive_ghost.run_scheduled_proactive_scan(project_id)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )
    return result


@app.get("/ghost/actions", tags=["ghost"])
async def get_ghost_actions(limit: int = 20):
    """
    Return the most recent Ghost actions from MongoDB.

    Each record includes: action_type, timestamp, MR / issue URL,
    developer, findings_count, severity_max, and outcome fields
    (was_accepted / was_merged).
    """
    col = database.get_collection("ghost_actions")
    cursor = col.find(
        {},
        {
            "action_type": 1,
            "timestamp": 1,
            "project_id": 1,
            "developer": 1,
            "mr_url": 1,
            "issue_url": 1,
            "gitlab_url": 1,
            "pattern_type": 1,
            "findings_count": 1,
            "severity_max": 1,
            "was_accepted": 1,
            "was_merged": 1,
            "files_affected": 1,
            "mr_title": 1,
            "issue_title": 1,
        },
        sort=[("timestamp", -1)],
        limit=max(1, min(limit, 100)),
    )

    actions = []
    async for doc in cursor:
        doc["id"] = str(doc.pop("_id"))
        # Normalise the URL field regardless of which key was used
        doc["url"] = doc.get("mr_url") or doc.get("issue_url") or doc.get("gitlab_url") or ""
        actions.append(doc)

    return {"count": len(actions), "actions": actions}


@app.post("/ghost/feedback", tags=["ghost"])
async def ghost_feedback(feedback: GhostFeedback):
    """
    Record human feedback on a Ghost action.

    Updates the ``was_accepted`` (for comments/issues) or ``was_merged``
    (for MRs) field in MongoDB so Ghost can track which of its actions
    were useful and which were rejected.

    Body fields
    -----------
    action_id   : The ``id`` string from GET /ghost/actions
    was_helpful : True = accepted / merged, False = dismissed / rejected
    reason      : Optional free-text explanation
    """
    try:
        oid = ObjectId(feedback.action_id)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid action_id: {feedback.action_id!r}",
        )

    col = database.get_collection("ghost_actions")

    # Determine which outcome field to update based on the action type
    existing = await col.find_one({"_id": oid}, {"action_type": 1})
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ghost action {feedback.action_id} not found",
        )

    action_type: str = existing.get("action_type", "")
    update_fields: dict = {
        "was_accepted": feedback.was_helpful,
        "feedback_reason": feedback.reason,
        "feedback_recorded_at": datetime.now(timezone.utc),
    }
    # MRs track was_merged instead of was_accepted
    if action_type in ("proactive_mr",):
        update_fields["was_merged"] = feedback.was_helpful

    result = await col.update_one({"_id": oid}, {"$set": update_fields})

    if result.matched_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ghost action {feedback.action_id} not found",
        )

    return {
        "updated": True,
        "action_id": feedback.action_id,
        "action_type": action_type,
        "was_helpful": feedback.was_helpful,
        "reason": feedback.reason,
    }
