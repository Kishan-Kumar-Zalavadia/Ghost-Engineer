"""GitLab repository reader module.

Fetches complete commit and merge request history from a GitLab project
and stores everything in MongoDB.

Usage:
    import asyncio
    from backend.modules.gitlab_reader import run_initial_repository_scan
    asyncio.run(run_initial_repository_scan(project_id=123))
"""

import logging
from datetime import datetime, timezone
from typing import Any

import gitlab
from gitlab.exceptions import GitlabError

from backend import database
from backend.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GitLab client
# ---------------------------------------------------------------------------

def _get_gitlab_client() -> gitlab.Gitlab:
    """Return an authenticated python-gitlab client."""
    gl = gitlab.Gitlab(
        url=settings.gitlab_url,
        private_token=settings.gitlab_token,
    )
    gl.auth()
    return gl


# ---------------------------------------------------------------------------
# 1. Fetch commit history
# ---------------------------------------------------------------------------

def fetch_all_commits(project_id: int) -> list[dict[str, Any]]:
    """Fetch every commit from the project, including diffs.

    Returns a list of commit dicts ready for MongoDB insertion.
    """
    gl = _get_gitlab_client()
    project = gl.projects.get(project_id)

    logger.info("[gitlab_reader] fetching commits for project_id=%s name=%s", project_id, project.name)

    commits_data: list[dict[str, Any]] = []

    # python-gitlab paginates automatically with as_list=False (lazy iterator)
    for i, commit in enumerate(project.commits.list(all=True, iterator=True), start=1):
        try:
            # Fetch the diff for this commit
            diff_list = commit.diff(get_all=True)
            files_changed: list[str] = []
            diff_content: list[dict] = []
            lines_added = 0
            lines_removed = 0

            for diff_item in diff_list:
                files_changed.append(diff_item.get("new_path") or diff_item.get("old_path", ""))
                raw_diff = diff_item.get("diff", "")
                diff_content.append({
                    "old_path": diff_item.get("old_path"),
                    "new_path": diff_item.get("new_path"),
                    "diff": raw_diff,
                    "new_file": diff_item.get("new_file", False),
                    "deleted_file": diff_item.get("deleted_file", False),
                    "renamed_file": diff_item.get("renamed_file", False),
                })
                for line in raw_diff.splitlines():
                    if line.startswith("+") and not line.startswith("+++"):
                        lines_added += 1
                    elif line.startswith("-") and not line.startswith("---"):
                        lines_removed += 1

            commits_data.append({
                "commit_hash": commit.id,
                "short_id": commit.short_id,
                "title": commit.title,
                "message": commit.message,
                "author_name": commit.author_name,
                "author_email": commit.author_email,
                "authored_date": _parse_dt(commit.authored_date),
                "committer_name": commit.committer_name,
                "committer_email": commit.committer_email,
                "committed_date": _parse_dt(commit.committed_date),
                "project_id": project_id,
                "project_name": project.path_with_namespace,
                "files_changed": files_changed,
                "diff_content": diff_content,
                "lines_added": lines_added,
                "lines_removed": lines_removed,
                "web_url": commit.web_url,
                "created_at": datetime.now(timezone.utc),
            })

            if i % 50 == 0:
                logger.info("[gitlab_reader] fetched %d commits so far ...", i)

        except GitlabError as exc:
            logger.warning("[gitlab_reader] could not fetch diff for commit %s: %s", commit.id, exc)

    logger.info("[gitlab_reader] total commits fetched: %d", len(commits_data))
    return commits_data


# ---------------------------------------------------------------------------
# 2. Fetch merge request history
# ---------------------------------------------------------------------------

def fetch_all_merge_requests(project_id: int) -> list[dict[str, Any]]:
    """Fetch all MRs (open, merged, closed) with comments and resolution time."""
    gl = _get_gitlab_client()
    project = gl.projects.get(project_id)

    logger.info("[gitlab_reader] fetching merge requests for project_id=%s", project_id)

    mrs_data: list[dict[str, Any]] = []

    for mr in project.mergerequests.list(state="all", all=True, iterator=True):
        try:
            # Fetch all notes/comments on this MR
            notes: list[dict] = []
            for note in mr.notes.list(all=True, iterator=True):
                notes.append({
                    "note_id": note.id,
                    "author": note.author.get("username") if note.author else None,
                    "body": note.body,
                    "created_at": _parse_dt(note.created_at),
                    "system": note.system,  # True = auto system note, False = human comment
                })

            # Calculate time open before resolution
            created_at = _parse_dt(mr.created_at)
            resolved_at = _parse_dt(mr.merged_at or mr.closed_at)
            time_open_seconds: int | None = None
            if created_at and resolved_at:
                time_open_seconds = int((resolved_at - created_at).total_seconds())

            mrs_data.append({
                "mr_id": mr.id,
                "iid": mr.iid,
                "title": mr.title,
                "description": mr.description,
                "state": mr.state,                  # opened / merged / closed
                "author": mr.author.get("username") if mr.author else None,
                "author_name": mr.author.get("name") if mr.author else None,
                "source_branch": mr.source_branch,
                "target_branch": mr.target_branch,
                "project_id": project_id,
                "project_name": project.path_with_namespace,
                "created_at": created_at,
                "merged_at": _parse_dt(mr.merged_at),
                "closed_at": _parse_dt(mr.closed_at),
                "time_open_seconds": time_open_seconds,
                "was_merged": mr.state == "merged",
                "merged_by": mr.merged_by.get("username") if mr.merged_by else None,
                "comments": notes,
                "comment_count": len(notes),
                "web_url": mr.web_url,
                "scanned_at": datetime.now(timezone.utc),
            })

        except GitlabError as exc:
            logger.warning("[gitlab_reader] could not fetch MR %s: %s", mr.iid, exc)

    logger.info("[gitlab_reader] total MRs fetched: %d", len(mrs_data))
    return mrs_data


# ---------------------------------------------------------------------------
# 3. Store commits to MongoDB
# ---------------------------------------------------------------------------

async def store_commits_to_mongodb(commits: list[dict[str, Any]]) -> int:
    """Upsert commits into the 'commits' collection. Returns count of new inserts."""
    if not commits:
        return 0

    collection = database.get_collection("commits")
    new_count = 0

    for i, doc in enumerate(commits, start=1):
        try:
            result = await collection.update_one(
                {"commit_hash": doc["commit_hash"]},
                {"$setOnInsert": doc},
                upsert=True,
            )
            if result.upserted_id:
                new_count += 1
        except Exception:
            logger.exception("[gitlab_reader] failed to upsert commit %s", doc.get("commit_hash"))

        if i % 50 == 0:
            logger.info("[gitlab_reader] stored %d / %d commits ...", i, len(commits))

    logger.info("[gitlab_reader] commits stored: %d new, %d already existed",
                new_count, len(commits) - new_count)
    return new_count


async def store_merge_requests_to_mongodb(mrs: list[dict[str, Any]]) -> int:
    """Upsert merge requests into the 'merge_requests' collection."""
    if not mrs:
        return 0

    collection = database.get_collection("merge_requests")
    new_count = 0

    for doc in mrs:
        try:
            result = await collection.update_one(
                {"mr_id": doc["mr_id"], "project_id": doc["project_id"]},
                {"$set": doc},
                upsert=True,
            )
            if result.upserted_id:
                new_count += 1
        except Exception:
            logger.exception("[gitlab_reader] failed to upsert MR %s", doc.get("mr_id"))

    logger.info("[gitlab_reader] MRs stored: %d new / updated, total %d", new_count, len(mrs))
    return new_count


# ---------------------------------------------------------------------------
# 4. Initial scan
# ---------------------------------------------------------------------------

async def run_initial_repository_scan(project_id: int) -> dict[str, Any]:
    """Run a full scan of commits and MRs, store to MongoDB, return summary."""
    logger.info("[gitlab_reader] ===== starting initial repository scan project_id=%s =====", project_id)

    # --- Commits ---
    commits = fetch_all_commits(project_id)
    new_commits = await store_commits_to_mongodb(commits)

    # --- Merge Requests ---
    mrs = fetch_all_merge_requests(project_id)
    await store_merge_requests_to_mongodb(mrs)

    # --- Build summary ---
    summary = _build_summary(commits, mrs, new_commits)

    logger.info(
        "[gitlab_reader] scan complete | commits=%d MRs=%d authors=%d date_range=%s → %s",
        summary["total_commits_scanned"],
        summary["total_mrs_scanned"],
        len(summary["unique_authors"]),
        summary["earliest_commit"],
        summary["latest_commit"],
    )
    return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_dt(value: str | None) -> datetime | None:
    """Parse an ISO-8601 string from GitLab into a timezone-aware datetime."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, AttributeError):
        return None


def _build_summary(
    commits: list[dict],
    mrs: list[dict],
    new_commits: int,
) -> dict[str, Any]:
    dates = [c["authored_date"] for c in commits if c.get("authored_date")]
    authors = {
        f"{c['author_name']} <{c['author_email']}>"
        for c in commits
        if c.get("author_email")
    }
    return {
        "total_commits_scanned": len(commits),
        "new_commits_inserted": new_commits,
        "total_mrs_scanned": len(mrs),
        "unique_authors": sorted(authors),
        "earliest_commit": min(dates).isoformat() if dates else None,
        "latest_commit": max(dates).isoformat() if dates else None,
        "mrs_merged": sum(1 for mr in mrs if mr["was_merged"]),
        "mrs_closed": sum(1 for mr in mrs if mr["state"] == "closed"),
        "mrs_open": sum(1 for mr in mrs if mr["state"] == "opened"),
    }
