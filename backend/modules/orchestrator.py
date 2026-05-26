"""Orchestrator – the brain that connects all Ghost Engineer modules.

Public API
----------
initialize_repository(project_id)     – called once when a repo is connected
process_new_commit(webhook_payload)    – called on every push webhook
process_new_merge_request(webhook_payload) – called on every MR webhook
run_scheduled_scan(project_id)         – called every 4 hours by Cloud Scheduler
"""

import logging
from datetime import datetime, timezone
from typing import Any

import gitlab as gitlab_lib

from backend import database
from backend.config import settings
from backend.modules import developer_profiler, ghost_actions, ghost_brain, gitlab_reader, pattern_detector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gitlab_client() -> gitlab_lib.Gitlab:
    gl = gitlab_lib.Gitlab(settings.gitlab_url, private_token=settings.gitlab_token)
    gl.auth()
    return gl


async def _fetch_commit_diff(project_id: int, commit_sha: str) -> str:
    """Fetch raw unified diff for a single commit via the GitLab API."""
    try:
        gl = _gitlab_client()
        project = gl.projects.get(project_id)
        diffs = project.commits.get(commit_sha).diff()
        return "\n".join(d.get("diff", "") for d in diffs if isinstance(d, dict))
    except Exception as exc:
        logger.warning("Could not fetch diff for commit %s: %s", commit_sha, exc)
        return ""


async def _fetch_mr_diff(project_id: int, mr_iid: int) -> str:
    """Fetch combined diff for a merge request via the GitLab API."""
    try:
        gl = _gitlab_client()
        project = gl.projects.get(project_id)
        mr = project.mergerequests.get(mr_iid)
        changes = mr.changes()
        diffs = changes.get("changes", [])
        return "\n".join(d.get("diff", "") for d in diffs if isinstance(d, dict))
    except Exception as exc:
        logger.warning("Could not fetch MR diff for MR!%s: %s", mr_iid, exc)
        return ""


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Function 1 – Initialize Repository
# ---------------------------------------------------------------------------

async def initialize_repository(project_id: int) -> dict:
    """
    Called once when a new repo is connected.

    Steps
    -----
    1. Run a full GitLab scan (all commits + MRs) via gitlab_reader.
    2. Run pattern detection on every stored commit.
    3. Build a developer profile for every unique author found.

    Returns a structured initialization summary.
    """
    logger.info("[orchestrator] initializing project_id=%s", project_id)

    # Step 1 – scan GitLab and persist raw data
    scan_summary = await gitlab_reader.run_initial_repository_scan(project_id)

    # Step 2 – pattern detection over every stored commit
    commits_col = database.get_collection("commits")
    total_in_db = await commits_col.count_documents({})
    logger.info("[orchestrator] commits in MongoDB after scan: %d", total_in_db)

    cursor = commits_col.find(
        {},  # initial scan; project_id may not be stored by gitlab_reader yet
        {"commit_hash": 1, "diff": 1, "diff_content": 1, "author_email": 1, "author_name": 1, "created_at": 1},
    )

    pattern_results: list[dict] = []
    authors_seen: set[str] = set()

    async for commit in cursor:
        sha: str = commit.get("commit_hash", "")

        # gitlab_reader stores diffs as diff_content (list of dicts); flatten to string
        raw_diff = commit.get("diff_content") or commit.get("diff") or []
        if isinstance(raw_diff, list):
            diff = "\n".join(
                d.get("diff", "") for d in raw_diff if isinstance(d, dict)
            )
        else:
            diff = raw_diff or ""

        author_email: str = commit.get("author_email", "")
        author_name: str = commit.get("author_name", "")
        timestamp: datetime = _parse_timestamp(commit.get("created_at"))

        logger.info(
            "[orchestrator] processing commit %s | author=%s | diff_chars=%d",
            sha[:8], author_email or author_name, len(diff),
        )

        try:
            result = await pattern_detector.analyze_commit(
                commit_sha=sha,
                diff_content=diff,
                author=author_email or author_name,
                timestamp=timestamp,
            )
            logger.info(
                "[orchestrator] pattern result for %s | risk_score=%s | findings=%d",
                sha[:8],
                result.get("risk_score"),
                len(result.get("findings", [])),
            )
            pattern_results.append(result)
        except Exception as exc:
            logger.error("[orchestrator] pattern detection failed for %s: %s", sha, exc)

        identifier = author_email or author_name
        if identifier:
            authors_seen.add(identifier)

    # Step 3 – build developer profiles
    profile_results: list[dict] = []
    for author in authors_seen:
        try:
            profile = await developer_profiler.build_developer_profile(author)
            profile_results.append({"author": author, "status": "ok"})
        except Exception as exc:
            logger.error("[orchestrator] profile build failed for %s: %s", author, exc)
            profile_results.append({"author": author, "status": "error", "reason": str(exc)})

    high_risk = sum(1 for r in pattern_results if r.get("risk_score", 0) > 3)

    return {
        "project_id": project_id,
        "scan_summary": scan_summary,
        "pattern_detection": {
            "commits_analyzed": len(pattern_results),
            "high_risk_commits": high_risk,
        },
        "developer_profiles": {
            "unique_authors": len(authors_seen),
            "profiles_built": sum(1 for p in profile_results if p["status"] == "ok"),
            "errors": sum(1 for p in profile_results if p["status"] == "error"),
        },
        "initialized_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Function 2 – Process New Commit (push webhook)
# ---------------------------------------------------------------------------

async def process_new_commit(webhook_payload: dict) -> dict:
    """
    Called every time a push webhook fires.

    Steps (per commit in the push)
    --------------------------------
    1. Extract commit metadata from the payload.
    2. Fetch the diff from GitLab API (diffs are not included in webhook).
    3. Run pattern detection on the diff.
    4. Incrementally update the developer profile.
    5. Fetch developer context for the AI agent.
    6. If risk_score > 3 → queue a ghost response (placeholder).
    7. Persist enriched commit document to MongoDB.

    Returns a summary of all commits processed.
    """
    project: dict = webhook_payload.get("project", {})
    project_id: int = project.get("id", settings.gitlab_project_id)
    branch: str = (
        webhook_payload.get("ref", "")
        .removeprefix("refs/heads/")
        .removeprefix("refs/tags/")
    )
    pushed_by: str = (
        webhook_payload.get("user_username")
        or webhook_payload.get("user_email")
        or webhook_payload.get("user_name", "unknown")
    )

    raw_commits: list[dict] = webhook_payload.get("commits", [])
    results: list[dict] = []
    commits_col = database.get_collection("commits")

    for raw_commit in raw_commits:
        sha: str = raw_commit.get("id", "")
        author: dict = raw_commit.get("author", {})
        author_email: str = author.get("email", "")
        author_name: str = author.get("name", pushed_by)
        gitlab_username: str = (
            webhook_payload.get("user_username") or author_email or author_name
        )
        timestamp: datetime = _parse_timestamp(raw_commit.get("timestamp"))

        # Step 2 – fetch diff (not part of the webhook payload)
        diff_content = await _fetch_commit_diff(project_id, sha)

        # Step 3 – pattern detection
        pattern_result: dict = {
            "commit_sha": sha, "risk_score": 0, "total_findings": 0, "findings": []
        }
        try:
            pattern_result = await pattern_detector.analyze_commit(
                commit_sha=sha,
                diff_content=diff_content,
                author=author_email or author_name,
                timestamp=timestamp,
            )
        except Exception as exc:
            logger.error("[orchestrator] pattern detection error for %s: %s", sha, exc)

        risk_score: float = pattern_result.get("risk_score", 0)

        # Step 4 – incremental developer profile update
        try:
            await developer_profiler.update_developer_profile(
                gitlab_username,
                {
                    "commit_hash": sha,
                    "risk_score": risk_score,
                    "author_email": author_email,
                    "author_name": author_name,
                    "timestamp": timestamp,
                    "diff": diff_content,
                },
            )
        except Exception as exc:
            logger.error("[orchestrator] profile update error for %s: %s", gitlab_username, exc)

        # Step 5 – developer context for AI agent
        developer_context: dict = {}
        try:
            developer_context = await developer_profiler.get_developer_context(
                gitlab_username, timestamp
            )
        except Exception as exc:
            logger.error("[orchestrator] get_developer_context error for %s: %s", gitlab_username, exc)

        # Step 6 – ghost response trigger
        ghost_triggered = False
        ghost_comment_text: str = ""
        findings: list[dict] = pattern_result.get("findings", [])
        if risk_score > 3:
            ghost_triggered = True
            logger.info(
                "[orchestrator] HIGH RISK commit %s (score=%.1f) by %s – triggering ghost actions",
                sha, risk_score, gitlab_username,
            )
            # Enrich developer_context with commit SHA so ghost_actions can use it
            developer_context["last_commit_sha"] = sha
            developer_context["gitlab_username"] = gitlab_username
            try:
                action = await ghost_actions.post_commit_comment(
                    project_id=project_id,
                    commit_sha=sha,
                    findings=findings,
                    developer_context=developer_context,
                )
                ghost_comment_text = action.get("comment_text", "")
            except Exception as exc:
                logger.error("[orchestrator] ghost_actions.post_commit_comment failed for %s: %s", sha, exc)

            # Create security issues for any CRITICAL findings
            critical = [f for f in findings if f.get("severity") == "CRITICAL"]
            for crit_finding in critical:
                try:
                    await ghost_actions.create_security_issue(
                        project_id=project_id,
                        finding=crit_finding,
                        commit_sha=sha,
                        developer=gitlab_username,
                    )
                except Exception as exc:
                    logger.error("[orchestrator] create_security_issue failed for %s: %s", sha, exc)

        # Step 7 – persist enriched document
        doc = {
            "commit_hash": sha,
            "project_id": project_id,
            "branch": branch,
            "message": raw_commit.get("message", ""),
            "title": raw_commit.get("title", ""),
            "author_name": author_name,
            "author_email": author_email,
            "gitlab_username": gitlab_username,
            "timestamp": timestamp,
            "url": raw_commit.get("url", ""),
            "files_added": raw_commit.get("added", []),
            "files_modified": raw_commit.get("modified", []),
            "files_removed": raw_commit.get("removed", []),
            "diff": diff_content,
            "pushed_by": pushed_by,
            "patterns_detected": pattern_result,
            "developer_context_snapshot": developer_context,
            "ghost_triggered": ghost_triggered,
            "ghost_comment": ghost_comment_text or None,
            "processed_at": datetime.now(timezone.utc),
        }
        try:
            await commits_col.update_one(
                {"commit_hash": sha},
                {"$set": doc},
                upsert=True,
            )
        except Exception as exc:
            logger.error("[orchestrator] MongoDB write failed for %s: %s", sha, exc)

        results.append({
            "commit_sha": sha,
            "risk_score": risk_score,
            "total_findings": pattern_result.get("total_findings", 0),
            "ghost_triggered": ghost_triggered,
            "response_style": developer_context.get("response_style", "educational"),
        })

    return {
        "project_id": project_id,
        "branch": branch,
        "pushed_by": pushed_by,
        "commits_processed": len(results),
        "results": results,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Function 3 – Process New Merge Request
# ---------------------------------------------------------------------------

async def process_new_merge_request(webhook_payload: dict) -> dict:
    """
    Called every time an MR webhook fires.

    Steps
    -----
    1. Extract MR metadata from the payload.
    2. Fetch the full MR diff from the GitLab API.
    3. Run pattern detection across the entire diff.
    4. Fetch developer context for the author.
    5. If any CRITICAL or HIGH findings → prepare a ghost comment.
    6. Persist the MR analysis document to MongoDB.

    Returns findings summary.
    """
    project: dict = webhook_payload.get("project", {})
    project_id: int = project.get("id", settings.gitlab_project_id)

    mr_attrs: dict = webhook_payload.get("object_attributes", {})
    mr_iid: int = mr_attrs.get("iid", 0)
    mr_id: int = mr_attrs.get("id", 0)
    mr_title: str = mr_attrs.get("title", "")
    mr_state: str = mr_attrs.get("state", "")
    mr_action: str = mr_attrs.get("action", "")
    source_branch: str = mr_attrs.get("source_branch", "")
    target_branch: str = mr_attrs.get("target_branch", "")
    mr_url: str = mr_attrs.get("url", "")

    user: dict = webhook_payload.get("user", {})
    gitlab_username: str = user.get("username", user.get("name", "unknown"))

    # Step 2 – fetch full MR diff
    diff_content = await _fetch_mr_diff(project_id, mr_iid)

    # Step 3 – pattern detection over the entire diff
    pattern_result: dict = {
        "commit_sha": f"mr-{mr_iid}", "risk_score": 0, "total_findings": 0, "findings": []
    }
    try:
        pattern_result = await pattern_detector.analyze_commit(
            commit_sha=f"mr-{mr_iid}",
            diff_content=diff_content,
            author=gitlab_username,
            timestamp=datetime.now(timezone.utc),
        )
    except Exception as exc:
        logger.error("[orchestrator] pattern detection error for MR!%s: %s", mr_iid, exc)

    # Step 4 – developer context
    developer_context: dict = {}
    try:
        developer_context = await developer_profiler.get_developer_context(
            gitlab_username, datetime.now(timezone.utc)
        )
    except Exception as exc:
        logger.error("[orchestrator] get_developer_context error for %s: %s", gitlab_username, exc)

    # Step 5 – post ghost comment for CRITICAL / HIGH findings
    findings: list[dict] = pattern_result.get("findings", [])
    critical_high = [f for f in findings if f.get("severity") in ("CRITICAL", "HIGH")]

    ghost_comment: dict | None = None
    if critical_high:
        logger.info(
            "[orchestrator] MR!%s has %d critical/high findings – posting ghost comment",
            mr_iid, len(critical_high),
        )
        developer_context["gitlab_username"] = gitlab_username
        try:
            ghost_comment = await ghost_actions.post_mr_comment(
                project_id=project_id,
                mr_iid=mr_iid,
                findings=critical_high,
                developer_context=developer_context,
                diff_content=diff_content,
            )
        except Exception as exc:
            logger.error("[orchestrator] ghost_actions.post_mr_comment failed for MR!%s: %s", mr_iid, exc)

        # Create security issues for any CRITICAL findings in the MR
        critical_only = [f for f in critical_high if f.get("severity") == "CRITICAL"]
        for crit_finding in critical_only:
            try:
                await ghost_actions.create_security_issue(
                    project_id=project_id,
                    finding=crit_finding,
                    commit_sha=f"mr-{mr_iid}",
                    developer=gitlab_username,
                )
            except Exception as exc:
                logger.error("[orchestrator] create_security_issue failed for MR!%s: %s", mr_iid, exc)

    # Step 6 – persist MR analysis
    mrs_col = database.get_collection("merge_requests")
    try:
        await mrs_col.update_one(
            {"mr_id": mr_id, "project_id": project_id},
            {
                "$set": {
                    "mr_id": mr_id,
                    "mr_iid": mr_iid,
                    "project_id": project_id,
                    "title": mr_title,
                    "state": mr_state,
                    "action": mr_action,
                    "source_branch": source_branch,
                    "target_branch": target_branch,
                    "url": mr_url,
                    "author_username": gitlab_username,
                    "patterns_detected": pattern_result,
                    "developer_context_snapshot": developer_context,
                    "ghost_comment_prepared": ghost_comment is not None,
                    "ghost_comment": ghost_comment,
                    "analyzed_at": datetime.now(timezone.utc),
                }
            },
            upsert=True,
        )
    except Exception as exc:
        logger.error("[orchestrator] MongoDB write failed for MR!%s: %s", mr_iid, exc)

    return {
        "project_id": project_id,
        "mr_iid": mr_iid,
        "mr_title": mr_title,
        "mr_action": mr_action,
        "risk_score": pattern_result.get("risk_score", 0),
        "total_findings": pattern_result.get("total_findings", 0),
        "critical_high_findings": len(critical_high),
        "ghost_comment_prepared": ghost_comment is not None,
        "response_style": developer_context.get("response_style", "educational"),
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Function 4 – Run Scheduled Scan
# ---------------------------------------------------------------------------

async def run_scheduled_scan(project_id: int) -> dict:
    """
    Called every 4 hours by Cloud Scheduler.

    Steps
    -----
    1. Find patterns that have appeared 3+ times without Ghost having flagged them.
    2. Flag commits with hardcoded-secrets findings as CVE / credential risk candidates.
    3. Collect dead-code (unused-import) findings.
    4. Queue proactive MR generation for the top 3 unflagged patterns.
    5. Refresh all developer profiles for the project.

    Returns a scan summary.
    """
    logger.info("[orchestrator] scheduled scan started for project_id=%s", project_id)

    commits_col = database.get_collection("commits")
    profiles_col = database.get_collection("developer_profiles")
    ghost_actions_col = database.get_collection("ghost_actions")

    # Step 1 – aggregate repeated patterns not yet actioned by Ghost
    pipeline = [
        {
            "$match": {
                "project_id": project_id,
                "patterns_detected.findings": {"$exists": True, "$ne": []},
            }
        },
        {"$unwind": "$patterns_detected.findings"},
        {
            "$group": {
                "_id": "$patterns_detected.findings.detector_name",
                "count": {"$sum": 1},
                "latest_commit": {"$last": "$commit_hash"},
                "authors": {"$addToSet": "$author_email"},
            }
        },
        {"$match": {"count": {"$gte": 3}}},
        {"$sort": {"count": -1}},
    ]

    unflagged_patterns: list[dict] = []
    async for doc in commits_col.aggregate(pipeline):
        detector_name = doc["_id"]
        already_flagged = await ghost_actions_col.find_one(
            {
                "project_id": project_id,
                "pattern_name": detector_name,
                "status": "completed",
            }
        )
        if not already_flagged:
            unflagged_patterns.append(
                {
                    "detector_name": detector_name,
                    "occurrences": doc["count"],
                    "latest_commit": doc["latest_commit"],
                    "affected_authors": doc["authors"],
                }
            )

    # Step 2 – credential / CVE risk: commits with hardcoded secrets
    cve_candidates: list[dict] = []
    async for doc in commits_col.find(
        {
            "project_id": project_id,
            "patterns_detected.findings": {
                "$elemMatch": {"detector_name": "detect_hardcoded_secrets"}
            },
        },
        {"commit_hash": 1, "author_email": 1, "timestamp": 1},
        limit=20,
    ):
        cve_candidates.append(
            {"commit_hash": doc["commit_hash"], "author": doc.get("author_email")}
        )

    # Step 3 – dead code: commits with unused-import findings
    dead_code_commits: list[dict] = []
    async for doc in commits_col.find(
        {
            "project_id": project_id,
            "patterns_detected.findings": {
                "$elemMatch": {"detector_name": "detect_dead_imports"}
            },
        },
        {"commit_hash": 1, "author_email": 1},
        limit=20,
    ):
        dead_code_commits.append(
            {"commit_hash": doc["commit_hash"], "author": doc.get("author_email")}
        )

    # Step 4 – open proactive MRs for top 3 unflagged patterns
    # Only fire if Ghost hasn't already opened one for this pattern recently.
    proactive_mrs_opened: list[str] = []
    for pattern in unflagged_patterns[:3]:
        detector_name = pattern["detector_name"]
        already_opened = await ghost_actions_col.find_one(
            {
                "project_id": project_id,
                "action_type": "proactive_mr",
                "pattern_type": detector_name,
                "was_merged": {"$ne": True},
            }
        )
        if already_opened:
            logger.info(
                "[orchestrator] proactive MR already open for pattern=%s, skipping",
                detector_name,
            )
            continue

        # Collect the affected files from recent commits that triggered this pattern
        affected_files: list[str] = []
        async for doc in commits_col.find(
            {
                "project_id": project_id,
                "patterns_detected.findings": {"$elemMatch": {"detector_name": detector_name}},
            },
            {"files_modified": 1, "files_added": 1},
            limit=10,
        ):
            affected_files.extend(doc.get("files_modified", []))
            affected_files.extend(doc.get("files_added", []))
        # Deduplicate and cap
        affected_files = list(dict.fromkeys(affected_files))[:10]

        # Ghost opens the MR with an empty proposed_file_changes list —
        # the description will explain the pattern and the team applies the fix.
        # A future enhancement will auto-generate the patch using Gemini.
        try:
            mr_url = await ghost_actions.open_proactive_mr(
                project_id=project_id,
                pattern_type=detector_name,
                affected_files=affected_files,
                proposed_file_changes=[],   # description-only MR for now
            )
            proactive_mrs_opened.append(mr_url)
            logger.info("[orchestrator] proactive MR opened: %s", mr_url)
        except Exception as exc:
            logger.error(
                "[orchestrator] open_proactive_mr failed for pattern=%s: %s",
                detector_name, exc,
            )

    # Step 5 – refresh developer profiles and send weekly coaching reports
    profiles_updated = 0
    coaching_reports_sent = 0
    async for profile in profiles_col.find(
        {},  # profiles are not scoped to project_id yet
        {"email": 1, "gitlab_username": 1},
    ):
        identifier = profile.get("gitlab_username") or profile.get("email", "")
        if not identifier:
            continue
        try:
            updated_profile = await developer_profiler.build_developer_profile(identifier)
            profiles_updated += 1

            # Send a weekly coaching report if one hasn't been sent this week
            username = profile.get("gitlab_username") or ""
            if username:
                from datetime import timedelta
                one_week_ago = datetime.now(timezone.utc) - timedelta(days=7)
                recent_report = await ghost_actions_col.find_one(
                    {
                        "action_type": "coaching_report",
                        "developer": username,
                        "timestamp": {"$gte": one_week_ago},
                    }
                )
                if not recent_report:
                    try:
                        await ghost_actions.send_weekly_coaching_report(
                            project_id=project_id,
                            developer_username=username,
                            developer_profile=updated_profile,
                        )
                        coaching_reports_sent += 1
                    except Exception as exc:
                        logger.error(
                            "[orchestrator] coaching report failed for %s: %s",
                            username, exc,
                        )
        except Exception as exc:
            logger.error(
                "[orchestrator] profile refresh failed for %s: %s", identifier, exc
            )

    return {
        "project_id": project_id,
        "unflagged_patterns_found": len(unflagged_patterns),
        "top_unflagged": unflagged_patterns[:5],
        "cve_candidates": len(cve_candidates),
        "dead_code_findings": len(dead_code_commits),
        "proactive_mrs_opened": len(proactive_mrs_opened),
        "proactive_mr_urls": proactive_mrs_opened,
        "developer_profiles_updated": profiles_updated,
        "coaching_reports_sent": coaching_reports_sent,
        "scanned_at": datetime.now(timezone.utc).isoformat(),
    }
