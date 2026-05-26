"""Ghost Action Engine — GitLab operations for GhostEngineer.

This module is Ghost's hands. It posts comments, opens MRs, creates issues,
and sends private coaching reports through the GitLab API.  Every action is
persisted to MongoDB so the orchestrator can track what Ghost has done.

Public API
----------
post_mr_comment(project_id, mr_iid, findings, developer_context, diff_content)
open_proactive_mr(project_id, pattern_type, affected_files, proposed_file_changes)
post_commit_comment(project_id, commit_sha, findings, developer_context)
create_security_issue(project_id, finding, commit_sha, developer)
send_weekly_coaching_report(project_id, developer_username, developer_profile)
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Any

import gitlab
from gitlab.exceptions import GitlabError

from backend import database
from backend.config import settings
from backend.modules import ghost_brain

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Severity ordering — used to pick the worst finding for issue titles
# ---------------------------------------------------------------------------

_SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "WARNING": 3, "LOW": 4}

_GHOST_LABEL = "ghost-engineer"
_SECURITY_LABEL = "security"


# ---------------------------------------------------------------------------
# GitLab client (sync — wrapped in asyncio.to_thread where needed)
# ---------------------------------------------------------------------------

def _gl_client() -> gitlab.Gitlab:
    gl = gitlab.Gitlab(url=settings.gitlab_url, private_token=settings.gitlab_token)
    gl.auth()
    return gl


# ---------------------------------------------------------------------------
# MongoDB helpers
# ---------------------------------------------------------------------------

async def _store_ghost_action(doc: dict) -> str:
    """Persist a ghost action record and return its inserted id as a string."""
    col = database.get_collection("ghost_actions")
    doc.setdefault("timestamp", datetime.now(timezone.utc))
    result = await col.insert_one(doc)
    return str(result.inserted_id)


async def _get_first_commit_date(project_id: int) -> str:
    """Return the earliest commit date for this project as a human-readable string."""
    col = database.get_collection("commits")
    doc = await col.find_one(
        {"project_id": project_id},
        sort=[("authored_date", 1)],
        projection={"authored_date": 1, "created_at": 1},
    )
    if not doc:
        return "the beginning"
    ts = doc.get("authored_date") or doc.get("created_at")
    if isinstance(ts, datetime):
        return ts.strftime("%b %Y")
    return "the beginning"


async def _get_pattern_occurrence_count(project_id: int, detector_name: str) -> int:
    """Count how many commits have triggered a specific detector."""
    col = database.get_collection("commits")
    count = await col.count_documents(
        {
            "project_id": project_id,
            "patterns_detected.findings": {
                "$elemMatch": {"detector_name": detector_name}
            },
        }
    )
    return count


async def _build_historical_patterns(
    project_id: int,
    findings: list[dict],
    limit: int = 5,
) -> list[dict]:
    """
    For each unique detector in *findings*, pull recent past occurrences
    from MongoDB so Ghost can cite real commit SHAs.
    """
    detector_names = {f.get("detector_name") for f in findings if f.get("detector_name")}
    if not detector_names:
        return []

    col = database.get_collection("commits")
    historical: list[dict] = []

    for detector in detector_names:
        cursor = col.find(
            {
                "project_id": project_id,
                "patterns_detected.findings": {
                    "$elemMatch": {"detector_name": detector}
                },
            },
            {"commit_hash": 1, "author_email": 1, "authored_date": 1, "created_at": 1},
            sort=[("authored_date", -1)],
            limit=limit,
        )
        async for doc in cursor:
            ts = doc.get("authored_date") or doc.get("created_at")
            date_str = ts.strftime("%Y-%m-%d") if isinstance(ts, datetime) else ""
            historical.append(
                {
                    "detector_name": detector,
                    "commit_sha": doc.get("commit_hash", ""),
                    "author": doc.get("author_email", ""),
                    "date": date_str,
                }
            )

    return historical


def _worst_severity(findings: list[dict]) -> str:
    """Return the highest severity label from a list of findings."""
    if not findings:
        return "LOW"
    return min(
        (f.get("severity", "LOW") for f in findings),
        key=lambda s: _SEVERITY_ORDER.get(s, 4),
    )


def _sanitize_branch_name(name: str) -> str:
    """Make a string safe to use as a GitLab branch name."""
    name = name.lower()
    name = re.sub(r"[^a-z0-9_-]", "-", name)
    name = re.sub(r"-{2,}", "-", name)
    return name.strip("-")[:60]


# ---------------------------------------------------------------------------
# Ghost comment footer
# ---------------------------------------------------------------------------

async def _build_signature(project_id: int, detector_names: list[str]) -> str:
    """Build Ghost's signature footer for any comment."""
    first_date = await _get_first_commit_date(project_id)

    counts: list[str] = []
    for name in detector_names[:2]:          # cap at 2 to keep footer short
        n = await _get_pattern_occurrence_count(project_id, name)
        if n > 0:
            label = name.replace("detect_", "").replace("_", " ")
            counts.append(f"{n}× {label}")

    pattern_note = (
        f"This pattern has appeared {', '.join(counts)} in this repo."
        if counts
        else "I've been tracking patterns across this entire codebase."
    )

    return (
        f"\n\n---\n"
        f"*🔍 GhostEngineer — I've been watching this codebase since {first_date}. "
        f"{pattern_note}*"
    )


# ---------------------------------------------------------------------------
# Function 1 — Post MR comment
# ---------------------------------------------------------------------------

async def post_mr_comment(
    project_id: int,
    mr_iid: int,
    findings: list[dict],
    developer_context: dict,
    diff_content: str,
) -> dict:
    """
    Generate a Ghost review comment and post it to a GitLab MR.

    Steps
    -----
    1. Build historical context from MongoDB.
    2. Call ghost_brain.generate_mr_comment() for the Gemini-authored body.
    3. Append Ghost's signature footer.
    4. Post to GitLab via MR notes API.
    5. Persist action record to MongoDB.

    Returns
    -------
    dict  The stored ghost_action record (including gitlab_url).
    """
    logger.info("[ghost_actions] posting MR comment | project=%s mr_iid=%s", project_id, mr_iid)

    # Step 1 — historical context for Ghost's memory
    historical = await _build_historical_patterns(project_id, findings)

    # Step 2 — Gemini-generated body
    commit_sha = developer_context.get("last_commit_sha", f"mr-{mr_iid}")
    body_text = await ghost_brain.generate_mr_comment(
        findings=findings,
        developer_context=developer_context,
        commit_sha=commit_sha,
        diff_content=diff_content,
        historical_patterns=historical,
    )

    # Step 3 — signature footer
    detector_names = list({f.get("detector_name", "") for f in findings if f.get("detector_name")})
    signature = await _build_signature(project_id, detector_names)
    full_comment = body_text + signature

    # Step 4 — post to GitLab (blocking API call → thread)
    note_url = ""
    try:
        def _post() -> str:
            gl = _gl_client()
            project = gl.projects.get(project_id)
            mr = project.mergerequests.get(mr_iid)
            note = mr.notes.create({"body": full_comment})
            return getattr(note, "web_url", "") or mr.web_url

        note_url = await asyncio.to_thread(_post)
        logger.info("[ghost_actions] MR comment posted | url=%s", note_url)
    except GitlabError as exc:
        logger.error("[ghost_actions] GitLab error posting MR comment: %s", exc)
    except Exception as exc:
        logger.error("[ghost_actions] unexpected error posting MR comment: %s", exc)

    # Step 5 — persist action record
    severity_max = _worst_severity(findings)
    developer = (
        developer_context.get("gitlab_username")
        or developer_context.get("email")
        or "unknown"
    )
    action_record = {
        "action_type": "mr_comment",
        "project_id": project_id,
        "mr_iid": mr_iid,
        "developer": developer,
        "findings_count": len(findings),
        "severity_max": severity_max,
        "comment_text": full_comment,
        "gitlab_url": note_url,
        "timestamp": datetime.now(timezone.utc),
        "was_accepted": None,   # updated later when Ghost tracks reactions
    }
    action_id = await _store_ghost_action(action_record)
    action_record["_id"] = action_id

    logger.info("[ghost_actions] MR comment action stored | id=%s", action_id)
    return action_record


# ---------------------------------------------------------------------------
# Function 2 — Open proactive MR
# ---------------------------------------------------------------------------

async def open_proactive_mr(
    project_id: int,
    pattern_type: str,
    affected_files: list[str],
    proposed_file_changes: list[dict],
) -> str:
    """
    Ghost opens a merge request that nobody asked for.

    Ghost creates a branch, commits the fixes, writes the MR description using
    Gemini, and opens the MR — all autonomously.

    Parameters
    ----------
    project_id:
        GitLab numeric project ID.
    pattern_type:
        The detector name that triggered this MR (e.g. 'detect_race_conditions').
    affected_files:
        File paths involved, used for the MR description context.
    proposed_file_changes:
        List of dicts each with keys: file_path (str), content (str),
        action ('create' | 'update' | 'delete').

    Returns
    -------
    str  The URL of the created MR.
    """
    logger.info(
        "[ghost_actions] opening proactive MR | project=%s pattern=%s files=%d",
        project_id, pattern_type, len(affected_files),
    )

    ts_slug = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    safe_pattern = _sanitize_branch_name(pattern_type.replace("detect_", ""))
    branch_name = f"ghost/fix-{safe_pattern}-{ts_slug}"

    # Step 1 — get project metadata (sync in thread)
    def _get_project_meta() -> dict:
        gl = _gl_client()
        proj = gl.projects.get(project_id)
        return {
            "default_branch": proj.default_branch or "main",
            "name": getattr(proj, "path_with_namespace", str(project_id)),
        }

    meta = await asyncio.to_thread(_get_project_meta)
    default_branch: str = meta["default_branch"]

    # Step 2 — create branch
    def _create_branch() -> None:
        gl = _gl_client()
        proj = gl.projects.get(project_id)
        proj.branches.create({"branch": branch_name, "ref": default_branch})
        logger.info("[ghost_actions] branch created: %s", branch_name)

    try:
        await asyncio.to_thread(_create_branch)
    except GitlabError as exc:
        logger.error("[ghost_actions] failed to create branch %s: %s", branch_name, exc)
        raise

    # Step 3 — commit file changes onto the new branch
    commit_message = (
        f"Ghost: fix {pattern_type.replace('detect_', '').replace('_', ' ')} "
        f"pattern across {len(affected_files)} file(s)"
    )

    def _commit_changes() -> None:
        gl = _gl_client()
        proj = gl.projects.get(project_id)
        actions = [
            {
                "action": change.get("action", "update"),
                "file_path": change["file_path"],
                "content": change["content"],
            }
            for change in proposed_file_changes
            if change.get("file_path") and change.get("content") is not None
        ]
        if actions:
            proj.commits.create(
                {
                    "branch": branch_name,
                    "commit_message": commit_message,
                    "author_name": "GhostEngineer",
                    "author_email": "ghost@ghost-engineer.dev",
                    "actions": actions,
                }
            )
            logger.info("[ghost_actions] committed %d file change(s) to %s", len(actions), branch_name)

    try:
        await asyncio.to_thread(_commit_changes)
    except GitlabError as exc:
        logger.error("[ghost_actions] failed to commit changes to %s: %s", branch_name, exc)
        raise

    # Step 4 — build historical context and generate MR description
    historical_col = database.get_collection("commits")
    occurrences = await _get_pattern_occurrence_count(project_id, pattern_type)

    # Grab first and latest commit SHAs for this pattern
    first_doc = await historical_col.find_one(
        {
            "project_id": project_id,
            "patterns_detected.findings": {"$elemMatch": {"detector_name": pattern_type}},
        },
        sort=[("authored_date", 1)],
        projection={"commit_hash": 1, "authored_date": 1},
    )
    latest_doc = await historical_col.find_one(
        {
            "project_id": project_id,
            "patterns_detected.findings": {"$elemMatch": {"detector_name": pattern_type}},
        },
        sort=[("authored_date", -1)],
        projection={"commit_hash": 1, "author_email": 1},
    )

    # Unique affected authors
    affected_authors_cursor = historical_col.distinct(
        "author_email",
        {
            "project_id": project_id,
            "patterns_detected.findings": {"$elemMatch": {"detector_name": pattern_type}},
        },
    )
    affected_authors: list[str] = await affected_authors_cursor

    historical_context = {
        "occurrences": occurrences,
        "first_seen_sha": first_doc.get("commit_hash", "") if first_doc else "",
        "latest_sha": latest_doc.get("commit_hash", "") if latest_doc else "",
        "affected_authors": affected_authors,
        "pattern_description": pattern_type.replace("detect_", "").replace("_", " "),
    }

    # Representative before/after from the first proposed change
    before_snippet = ""
    after_snippet = ""
    if proposed_file_changes:
        first_change = proposed_file_changes[0]
        after_snippet = (first_change.get("content", ""))[:600]

    mr_description = await ghost_brain.generate_proactive_mr_description(
        pattern_type=pattern_type,
        affected_files=affected_files,
        proposed_changes={
            "summary": commit_message,
            "before": before_snippet,
            "after": after_snippet,
        },
        historical_context=historical_context,
    )

    # Append signature
    signature = await _build_signature(project_id, [pattern_type])
    full_description = mr_description + signature

    # Step 5 — create the MR
    pattern_label = pattern_type.replace("detect_", "").replace("_", " ").title()
    mr_title = f"🔍 Ghost: fix {pattern_label} ({occurrences} occurrences)"

    def _create_mr() -> str:
        gl = _gl_client()
        proj = gl.projects.get(project_id)
        mr = proj.mergerequests.create(
            {
                "source_branch": branch_name,
                "target_branch": default_branch,
                "title": mr_title,
                "description": full_description,
                "labels": [_GHOST_LABEL, "automated"],
                "remove_source_branch": True,
            }
        )
        logger.info("[ghost_actions] MR created | url=%s", mr.web_url)
        return mr.web_url

    mr_url = ""
    try:
        mr_url = await asyncio.to_thread(_create_mr)
    except GitlabError as exc:
        logger.error("[ghost_actions] failed to create MR from %s: %s", branch_name, exc)
        raise

    # Step 6 — persist action record
    await _store_ghost_action(
        {
            "action_type": "proactive_mr",
            "project_id": project_id,
            "pattern_type": pattern_type,
            "branch_name": branch_name,
            "files_affected": affected_files,
            "files_changed_count": len(proposed_file_changes),
            "mr_title": mr_title,
            "mr_url": mr_url,
            "historical_occurrences": occurrences,
            "timestamp": datetime.now(timezone.utc),
            "was_merged": None,     # updated when MR webhook fires
        }
    )

    logger.info("[ghost_actions] proactive MR opened | url=%s", mr_url)
    return mr_url


# ---------------------------------------------------------------------------
# Function 3 — Post commit comment
# ---------------------------------------------------------------------------

async def post_commit_comment(
    project_id: int,
    commit_sha: str,
    findings: list[dict],
    developer_context: dict,
) -> dict:
    """
    Post a Ghost review comment directly on a commit (when no MR exists yet).

    Uses the GitLab Commits notes API so the comment appears on the commit
    itself rather than any MR.

    Returns
    -------
    dict  The stored ghost_action record.
    """
    logger.info(
        "[ghost_actions] posting commit comment | project=%s sha=%s",
        project_id, commit_sha[:8],
    )

    # Fetch the commit's diff from MongoDB so Ghost has context
    col = database.get_collection("commits")
    doc = await col.find_one({"commit_hash": commit_sha}, {"diff": 1, "diff_content": 1})
    diff_content = ""
    if doc:
        if isinstance(doc.get("diff"), str):
            diff_content = doc["diff"]
        elif isinstance(doc.get("diff_content"), list):
            diff_content = "\n".join(
                item.get("diff", "") for item in doc["diff_content"] if isinstance(item, dict)
            )

    historical = await _build_historical_patterns(project_id, findings)

    body_text = await ghost_brain.generate_mr_comment(
        findings=findings,
        developer_context=developer_context,
        commit_sha=commit_sha,
        diff_content=diff_content,
        historical_patterns=historical,
    )

    detector_names = list({f.get("detector_name", "") for f in findings if f.get("detector_name")})
    signature = await _build_signature(project_id, detector_names)
    full_comment = body_text + signature

    # Post via GitLab Commits API
    note_url = ""
    try:
        def _post() -> str:
            gl = _gl_client()
            proj = gl.projects.get(project_id)
            commit = proj.commits.get(commit_sha)
            note = commit.comments.create({"note": full_comment})
            return commit.web_url

        note_url = await asyncio.to_thread(_post)
        logger.info("[ghost_actions] commit comment posted | sha=%s", commit_sha[:8])
    except GitlabError as exc:
        logger.error("[ghost_actions] GitLab error posting commit comment: %s", exc)
    except Exception as exc:
        logger.error("[ghost_actions] unexpected error posting commit comment: %s", exc)

    severity_max = _worst_severity(findings)
    developer = (
        developer_context.get("gitlab_username")
        or developer_context.get("email")
        or "unknown"
    )
    action_record = {
        "action_type": "commit_comment",
        "project_id": project_id,
        "commit_sha": commit_sha,
        "developer": developer,
        "findings_count": len(findings),
        "severity_max": severity_max,
        "comment_text": full_comment,
        "gitlab_url": note_url,
        "timestamp": datetime.now(timezone.utc),
        "was_accepted": None,
    }
    action_id = await _store_ghost_action(action_record)
    action_record["_id"] = action_id

    logger.info("[ghost_actions] commit comment action stored | id=%s", action_id)
    return action_record


# ---------------------------------------------------------------------------
# Function 4 — Create security issue (CRITICAL findings only)
# ---------------------------------------------------------------------------

async def create_security_issue(
    project_id: int,
    finding: dict,
    commit_sha: str,
    developer: str,
) -> dict:
    """
    Create a confidential GitLab issue for a CRITICAL security finding.

    The issue is assigned to the commit author and labelled 'security' and
    'ghost-engineer' so it surfaces in security dashboards.

    Parameters
    ----------
    project_id:
        GitLab numeric project ID.
    finding:
        A single Finding dict from pattern_detector (severity should be CRITICAL).
    commit_sha:
        The commit that introduced the issue.
    developer:
        GitLab username of the commit author (used for assignment).

    Returns
    -------
    dict  The stored ghost_action record including issue_url.
    """
    short_sha = commit_sha[:8] if len(commit_sha) >= 8 else commit_sha
    detector = finding.get("detector_name", "unknown")
    description_text = finding.get("description", "Security issue detected")
    suggested_fix = finding.get("suggested_fix", "")
    snippet = finding.get("snippet", "")
    line_no = finding.get("line_number")

    title = f"🚨 Security: {description_text[:80]}"

    location_detail = f"Line {line_no}" if line_no else "See diff"
    snippet_block = f"\n```\n{snippet[:500]}\n```" if snippet else ""
    fix_block = f"\n\n**Suggested fix:** {suggested_fix}" if suggested_fix else ""

    body = f"""\
## Security Issue Detected by GhostEngineer

**Commit:** `{commit_sha}`
**Detector:** `{detector}`
**Location:** {location_detail}
**Author:** @{developer}

### What was found

{description_text}
{snippet_block}
{fix_block}

### Why this matters

This pattern was flagged as CRITICAL because it can expose credentials, \
enable unauthorised access, or introduce a vulnerability that can be exploited \
without requiring further user interaction.

### Immediate next step

1. Rotate any credentials or secrets that may have been exposed.
2. Audit git history to check whether this commit reached a public branch.
3. Apply the suggested fix and request an expedited review.

---
*🔍 Opened automatically by GhostEngineer — commit `{short_sha}`*
"""

    issue_url = ""
    assignee_id: int | None = None

    def _resolve_user_and_create(body: str, title: str) -> tuple[str, int | None]:
        gl = _gl_client()
        proj = gl.projects.get(project_id)

        # Resolve GitLab user ID for the developer
        uid: int | None = None
        try:
            users = gl.users.list(username=developer)
            if users:
                uid = users[0].id
        except GitlabError:
            pass

        issue_payload: dict[str, Any] = {
            "title": title,
            "description": body,
            "labels": [_SECURITY_LABEL, _GHOST_LABEL],
            "confidential": True,
        }
        if uid:
            issue_payload["assignee_ids"] = [uid]

        issue = proj.issues.create(issue_payload)
        logger.info("[ghost_actions] security issue created | url=%s", issue.web_url)
        return issue.web_url, uid

    try:
        issue_url, assignee_id = await asyncio.to_thread(
            _resolve_user_and_create, body, title
        )
    except GitlabError as exc:
        logger.error("[ghost_actions] GitLab error creating security issue: %s", exc)
    except Exception as exc:
        logger.error("[ghost_actions] unexpected error creating security issue: %s", exc)

    action_record = {
        "action_type": "security_issue",
        "project_id": project_id,
        "commit_sha": commit_sha,
        "developer": developer,
        "detector_name": detector,
        "severity": finding.get("severity", "CRITICAL"),
        "issue_title": title,
        "issue_url": issue_url,
        "assignee_gitlab_id": assignee_id,
        "timestamp": datetime.now(timezone.utc),
        "was_accepted": None,
    }
    action_id = await _store_ghost_action(action_record)
    action_record["_id"] = action_id

    logger.info("[ghost_actions] security issue action stored | id=%s", action_id)
    return action_record


# ---------------------------------------------------------------------------
# Function 5 — Send weekly coaching report
# ---------------------------------------------------------------------------

async def send_weekly_coaching_report(
    project_id: int,
    developer_username: str,
    developer_profile: dict,
) -> dict:
    """
    Generate a private engineering coaching report and deliver it as a
    confidential GitLab issue assigned only to the developer.

    Steps
    -----
    1. Pull the developer's recent commits from MongoDB.
    2. Call ghost_brain.generate_developer_coaching_report() for the content.
    3. Create a confidential GitLab issue assigned to the developer.
    4. Persist the ghost_action record to MongoDB.

    Returns
    -------
    dict  The stored ghost_action record including issue_url.
    """
    logger.info(
        "[ghost_actions] generating coaching report | developer=%s project=%s",
        developer_username, project_id,
    )

    # Step 1 — pull recent commits for this developer
    col = database.get_collection("commits")
    recent_commits: list[dict] = []
    cursor = col.find(
        {
            "project_id": project_id,
            "$or": [
                {"gitlab_username": developer_username},
                {"author_email": developer_profile.get("email", "__none__")},
            ],
        },
        {
            "commit_hash": 1,
            "timestamp": 1,
            "authored_date": 1,
            "patterns_detected.risk_score": 1,
            "patterns_detected.findings": 1,
        },
        sort=[("authored_date", -1)],
        limit=30,
    )
    async for doc in cursor:
        recent_commits.append(
            {
                "commit_hash": doc.get("commit_hash", ""),
                "timestamp": doc.get("authored_date") or doc.get("timestamp"),
                "risk_score": doc.get("patterns_detected", {}).get("risk_score", 0),
            }
        )

    # Step 2 — generate report via Gemini
    report_text = await ghost_brain.generate_developer_coaching_report(
        developer_profile=developer_profile,
        recent_commits=recent_commits,
        period="last_30_days",
    )

    # Step 3 — create the confidential GitLab issue
    week_str = datetime.now(timezone.utc).strftime("%b %d, %Y")
    issue_title = f"👻 Your Weekly Ghost Report — {week_str}"

    footer = (
        "\n\n---\n"
        "*This report is confidential and visible only to you. "
        "GhostEngineer generates it from your commit history — "
        "no one else has reviewed it.*"
    )
    full_body = report_text + footer

    issue_url = ""
    assignee_id: int | None = None

    def _create_coaching_issue() -> tuple[str, int | None]:
        gl = _gl_client()
        proj = gl.projects.get(project_id)

        uid: int | None = None
        try:
            users = gl.users.list(username=developer_username)
            if users:
                uid = users[0].id
        except GitlabError:
            pass

        payload: dict[str, Any] = {
            "title": issue_title,
            "description": full_body,
            "labels": [_GHOST_LABEL, "coaching"],
            "confidential": True,
        }
        if uid:
            payload["assignee_ids"] = [uid]

        issue = proj.issues.create(payload)
        logger.info(
            "[ghost_actions] coaching issue created | developer=%s url=%s",
            developer_username, issue.web_url,
        )
        return issue.web_url, uid

    try:
        issue_url, assignee_id = await asyncio.to_thread(_create_coaching_issue)
    except GitlabError as exc:
        logger.error(
            "[ghost_actions] GitLab error creating coaching issue for %s: %s",
            developer_username, exc,
        )
    except Exception as exc:
        logger.error(
            "[ghost_actions] unexpected error creating coaching issue for %s: %s",
            developer_username, exc,
        )

    # Step 4 — persist action record
    action_record = {
        "action_type": "coaching_report",
        "project_id": project_id,
        "developer": developer_username,
        "issue_title": issue_title,
        "issue_url": issue_url,
        "assignee_gitlab_id": assignee_id,
        "commits_analysed": len(recent_commits),
        "report_period": "last_30_days",
        "timestamp": datetime.now(timezone.utc),
        "was_accepted": None,
    }
    action_id = await _store_ghost_action(action_record)
    action_record["_id"] = action_id

    logger.info(
        "[ghost_actions] coaching report action stored | id=%s developer=%s",
        action_id, developer_username,
    )
    return action_record
