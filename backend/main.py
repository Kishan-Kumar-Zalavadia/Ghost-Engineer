from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from backend import database
from backend.config import settings
from backend.modules import orchestrator
from backend.webhooks.router import router as webhooks_router


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
