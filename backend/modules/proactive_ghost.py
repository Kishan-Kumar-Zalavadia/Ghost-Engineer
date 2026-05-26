"""Proactive Ghost Scanner — autonomous code improvement engine.

Runs on a schedule and finds things to fix WITHOUT being triggered by a
commit.  This is what makes Ghost feel truly autonomous: it opens MRs
nobody asked for, fixes what it finds, and moves on.

Public API
----------
find_duplicate_code_opportunities(project_id) → dict | None
find_security_quick_wins(project_id)          → list[dict]
find_documentation_gaps(project_id)           → dict | None
run_scheduled_proactive_scan(project_id)      → dict
"""

from __future__ import annotations

import ast
import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import gitlab
from gitlab.exceptions import GitlabError

from backend import database
from backend.config import settings
from backend.modules import ghost_actions, ghost_brain, pattern_detector

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROACTIVE_MR_COOLDOWN_HOURS = 4   # min gap between proactive MRs
_MIN_COPY_PASTE_OCCURRENCES = 2    # fire duplicate scan after N findings
_MIN_FILE_MODIFICATIONS = 3        # file must be touched this many times for doc scan
_MAX_SECURITY_FIXES_PER_RUN = 3    # cap per run to avoid MR spam

# Actual detector_name values stored by pattern_detector (no "detect_" prefix)
_DN_SECRETS = "hardcoded_secrets"
_DN_COPY_PASTE = "copy_paste"

# Secret patterns mirrors from pattern_detector — used to validate current files
_SECRET_PATTERNS_RAW: list[tuple[str, re.Pattern]] = [
    ("api_key assignment",     re.compile(r"""api_key\s*=\s*['"][^'"]{8,}['"]""", re.IGNORECASE)),
    ("password assignment",    re.compile(r"""password\s*=\s*['"][^'"]{4,}['"]""", re.IGNORECASE)),
    ("secret assignment",      re.compile(r"""secret\s*=\s*['"][^'"]{8,}['"]""", re.IGNORECASE)),
    ("AWS access key",         re.compile(r"""AKIA[0-9A-Z]{16}""")),
    ("token assignment",       re.compile(r"""token\s*=\s*['"][a-zA-Z0-9]{20,}['"]""", re.IGNORECASE)),
    ("db connection string",   re.compile(r"""(postgres|mysql|mongodb):\/\/[^:]+:[^@]+@""", re.IGNORECASE)),
    ("generic key literal",    re.compile(r"""(api_key|secret_key|private_key)\s*=\s*['"][^'"]{6,}['"]""", re.IGNORECASE)),
]


# ---------------------------------------------------------------------------
# GitLab helpers
# ---------------------------------------------------------------------------

def _gl_client() -> gitlab.Gitlab:
    gl = gitlab.Gitlab(url=settings.gitlab_url, private_token=settings.gitlab_token)
    gl.auth()
    return gl


def _fetch_file_content(project_id: int, file_path: str, ref: str = "HEAD") -> str | None:
    """Synchronous helper — fetch a file's decoded content from GitLab."""
    try:
        gl = _gl_client()
        proj = gl.projects.get(project_id)
        f = proj.files.get(file_path=file_path, ref=ref)
        return f.decode().decode("utf-8", errors="replace")
    except GitlabError as exc:
        logger.debug("Could not fetch %s@%s: %s", file_path, ref, exc)
        return None
    except Exception as exc:
        logger.warning("Unexpected error fetching %s: %s", file_path, exc)
        return None


def _get_default_branch(project_id: int) -> str:
    try:
        gl = _gl_client()
        proj = gl.projects.get(project_id)
        return proj.default_branch or "main"
    except Exception:
        return "main"


# ---------------------------------------------------------------------------
# Rate-limit guard
# ---------------------------------------------------------------------------

async def _recent_proactive_mr_exists(project_id: int) -> bool:
    """Return True if a proactive MR was opened within the cooldown window."""
    col = database.get_collection("ghost_actions")
    cutoff = datetime.now(timezone.utc) - timedelta(hours=_PROACTIVE_MR_COOLDOWN_HOURS)
    doc = await col.find_one(
        {
            "project_id": project_id,
            "action_type": "proactive_mr",
            "timestamp": {"$gte": cutoff},
        }
    )
    return doc is not None


async def _already_addressed(project_id: int, pattern_type: str) -> bool:
    """True if Ghost already has an open (not merged) proactive MR for this pattern."""
    col = database.get_collection("ghost_actions")
    doc = await col.find_one(
        {
            "project_id": project_id,
            "action_type": "proactive_mr",
            "pattern_type": pattern_type,
            "was_merged": {"$ne": True},
        }
    )
    return doc is not None


# ---------------------------------------------------------------------------
# Function 1 — Find duplicate code opportunities
# ---------------------------------------------------------------------------

async def find_duplicate_code_opportunities(project_id: int) -> dict | None:
    """
    Find the most duplicated code block in the project and open a proactive MR
    to extract it into a shared utility.

    Steps
    -----
    1. Query MongoDB for commits where the copy_paste detector fired 2+ times.
    2. Identify which file appears most often across those findings.
    3. Fetch the current file content from GitLab.
    4. Ask Gemini to extract the duplicated function into a shared utility module.
    5. Generate updated versions of affected files that import the shared utility.
    6. Open one proactive MR with all the changes.

    Returns the opportunity dict on success, None if nothing actionable found.
    """
    logger.info("[proactive] scanning for duplicate code | project=%s", project_id)

    if await _already_addressed(project_id, _DN_COPY_PASTE):
        logger.info("[proactive] copy_paste MR already open, skipping")
        return None

    commits_col = database.get_collection("commits")

    # Step 1 — find commits with copy_paste findings
    pipeline = [
        {
            "$match": {
                "project_id": project_id,
                "patterns_detected.findings": {
                    "$elemMatch": {"detector_name": _DN_COPY_PASTE}
                },
            }
        },
        # Flatten the files lists
        {
            "$project": {
                "commit_hash": 1,
                "files": {
                    "$concatArrays": [
                        {"$ifNull": ["$files_modified", []]},
                        {"$ifNull": ["$files_added", []]},
                    ]
                },
                "copy_paste_findings": {
                    "$filter": {
                        "input": {"$ifNull": ["$patterns_detected.findings", []]},
                        "as": "f",
                        "cond": {"$eq": ["$$f.detector_name", _DN_COPY_PASTE]},
                    }
                },
            }
        },
        {"$unwind": "$files"},
        {
            "$group": {
                "_id": "$files",
                "finding_count": {"$sum": {"$size": "$copy_paste_findings"}},
                "commit_count": {"$sum": 1},
                "sample_snippet": {"$first": {"$arrayElemAt": ["$copy_paste_findings.snippet", 0]}},
            }
        },
        {"$match": {"finding_count": {"$gte": _MIN_COPY_PASTE_OCCURRENCES}}},
        {"$sort": {"finding_count": -1}},
        {"$limit": 1},
    ]

    top_file: dict | None = None
    async for doc in commits_col.aggregate(pipeline):
        top_file = doc

    if not top_file:
        logger.info("[proactive] no copy_paste candidates found")
        return None

    file_path: str = top_file["_id"]
    finding_count: int = top_file["finding_count"]
    sample_snippet: str = top_file.get("sample_snippet") or ""

    # Only process Python files for now
    if not file_path.endswith(".py"):
        logger.info("[proactive] copy_paste candidate %s is not Python, skipping", file_path)
        return None

    logger.info("[proactive] duplicate code candidate: %s (%d findings)", file_path, finding_count)

    # Step 3 — fetch current file content
    default_branch = await asyncio.to_thread(_get_default_branch, project_id)
    current_content = await asyncio.to_thread(_fetch_file_content, project_id, file_path, default_branch)
    if not current_content:
        logger.warning("[proactive] could not fetch %s, skipping", file_path)
        return None

    # Step 4 — use Gemini to extract the duplicated function into a shared utility
    dir_path = "/".join(file_path.split("/")[:-1]) or "."
    utils_path = (f"{dir_path}/ghost_utils.py" if dir_path != "." else "ghost_utils.py")

    snippet_context = f"\nThe following duplicated snippet was detected:\n{sample_snippet[:800]}" if sample_snippet else ""

    utility_code = await ghost_brain.generate_code_fix(
        task=(
            f"Extract the most-duplicated function or logic block from this file into a "
            f"new shared utility module at `{utils_path}`. "
            f"The utility module should contain only the extracted function(s) with "
            f"proper docstrings. Return ONLY the content of the new utility file."
            f"{snippet_context}"
        ),
        original_code=current_content,
        context=f"This function appears in {finding_count} detected duplicate instances across the project.",
    )

    # Step 5 — generate updated version of the original file that imports from utils
    updated_original = await ghost_brain.generate_code_fix(
        task=(
            f"Update this file to import the extracted function(s) from `{utils_path}` "
            f"instead of defining them locally. Remove the duplicate function body. "
            f"Add the appropriate import at the top of the file."
        ),
        original_code=current_content,
        context=f"The extracted utility is now in `{utils_path}`.",
    )

    proposed_changes = [
        {"action": "create", "file_path": utils_path, "content": utility_code},
        {"action": "update", "file_path": file_path, "content": updated_original},
    ]

    # Step 6 — open the proactive MR
    try:
        mr_url = await ghost_actions.open_proactive_mr(
            project_id=project_id,
            pattern_type=_DN_COPY_PASTE,
            affected_files=[file_path, utils_path],
            proposed_file_changes=proposed_changes,
        )
    except Exception as exc:
        logger.error("[proactive] open_proactive_mr failed for copy_paste: %s", exc)
        return None

    result = {
        "opportunity": "duplicate_code",
        "file": file_path,
        "utility_created": utils_path,
        "finding_count": finding_count,
        "mr_url": mr_url,
    }
    logger.info("[proactive] duplicate code MR opened: %s", mr_url)
    return result


# ---------------------------------------------------------------------------
# Function 2 — Find security quick wins
# ---------------------------------------------------------------------------

async def find_security_quick_wins(project_id: int) -> list[dict]:
    """
    Find hardcoded secrets that are still present in the codebase and open
    a proactive MR replacing each one with an environment variable reference.

    Steps
    -----
    1. Query MongoDB for commits with hardcoded_secrets findings that have
       no ghost action on record yet.
    2. For each affected file, fetch the current content from GitLab.
    3. Re-run secret detection on the live file content.
    4. If a secret is still there, ask Gemini to replace it with os.environ.
    5. Open one proactive MR per file (capped at _MAX_SECURITY_FIXES_PER_RUN).

    Returns a list of result dicts for each file addressed.
    """
    logger.info("[proactive] scanning for security quick wins | project=%s", project_id)

    commits_col = database.get_collection("commits")
    ghost_col = database.get_collection("ghost_actions")
    default_branch = await asyncio.to_thread(_get_default_branch, project_id)

    # Step 1 — find commits with unaddressed secret findings
    # Collect unique files that have triggered this detector
    file_set: dict[str, dict] = {}   # file_path → {commit_hash, snippet}

    cursor = commits_col.find(
        {
            "project_id": project_id,
            "patterns_detected.findings": {
                "$elemMatch": {"detector_name": _DN_SECRETS}
            },
        },
        {
            "commit_hash": 1,
            "files_modified": 1,
            "files_added": 1,
            "patterns_detected.findings": 1,
        },
        limit=50,
    )

    async for doc in cursor:
        sha = doc.get("commit_hash", "")
        files = (doc.get("files_modified") or []) + (doc.get("files_added") or [])
        findings_list = doc.get("patterns_detected", {}).get("findings", [])
        secret_findings = [f for f in findings_list if f.get("detector_name") == _DN_SECRETS]
        snippet = secret_findings[0].get("snippet", "") if secret_findings else ""

        for fp in files:
            if fp.endswith(".py") and fp not in file_set:
                file_set[fp] = {"commit_hash": sha, "snippet": snippet}

    if not file_set:
        logger.info("[proactive] no hardcoded_secrets candidates found")
        return []

    results: list[dict] = []

    for file_path, meta in list(file_set.items())[:_MAX_SECURITY_FIXES_PER_RUN]:
        # Check if Ghost already has a proactive MR open for this exact file
        already = await ghost_col.find_one(
            {
                "project_id": project_id,
                "action_type": "proactive_mr",
                "pattern_type": _DN_SECRETS,
                "files_affected": file_path,
                "was_merged": {"$ne": True},
            }
        )
        if already:
            logger.info("[proactive] security MR already open for %s, skipping", file_path)
            continue

        # Step 2 — fetch current file content
        current_content = await asyncio.to_thread(
            _fetch_file_content, project_id, file_path, default_branch
        )
        if not current_content:
            continue

        # Step 3 — check if secret still present in live file
        still_has_secret = any(
            pat.search(line)
            for label, pat in _SECRET_PATTERNS_RAW
            for line in current_content.splitlines()
        )
        if not still_has_secret:
            logger.info("[proactive] secret already removed from %s, skipping", file_path)
            continue

        logger.info("[proactive] secret still present in %s — generating fix", file_path)

        # Step 4 — ask Gemini to replace the hardcoded value with os.environ
        snippet_hint = f"\nKnown offending snippet: {meta['snippet'][:300]}" if meta["snippet"] else ""
        fixed_content = await ghost_brain.generate_code_fix(
            task=(
                "Replace all hardcoded secrets, API keys, passwords, and connection strings "
                "in this file with `os.environ.get('VAR_NAME', '')` references. "
                "Choose descriptive, UPPER_SNAKE_CASE variable names derived from context "
                "(e.g. STRIPE_API_KEY, DATABASE_PASSWORD). "
                "Add `import os` at the top if not already present. "
                "Add a comment above each replaced line: "
                "# TODO: set VAR_NAME in your .env or secrets manager"
                f"{snippet_hint}"
            ),
            original_code=current_content,
            context=(
                f"Commit {meta['commit_hash'][:8]} introduced this secret. "
                "The goal is to make this file safe to commit."
            ),
        )

        if not fixed_content or fixed_content == current_content:
            logger.warning("[proactive] Gemini returned unchanged content for %s", file_path)
            continue

        # Step 5 — open proactive MR
        try:
            mr_url = await ghost_actions.open_proactive_mr(
                project_id=project_id,
                pattern_type=_DN_SECRETS,
                affected_files=[file_path],
                proposed_file_changes=[
                    {"action": "update", "file_path": file_path, "content": fixed_content}
                ],
            )
            results.append({
                "opportunity": "security_quick_win",
                "file": file_path,
                "commit_sha": meta["commit_hash"],
                "mr_url": mr_url,
            })
            logger.info("[proactive] security fix MR opened for %s: %s", file_path, mr_url)
        except Exception as exc:
            logger.error("[proactive] open_proactive_mr failed for %s: %s", file_path, exc)

    return results


# ---------------------------------------------------------------------------
# Function 3 — Find documentation gaps
# ---------------------------------------------------------------------------

def _functions_missing_docstrings(source: str) -> list[str]:
    """
    Parse Python source and return names of public functions / methods that
    have no docstring.  Private functions (starting with `_`) are skipped.
    """
    missing: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return missing

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_"):
            continue
        # A docstring is the first statement if it's a string constant
        has_docstring = (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        )
        if not has_docstring:
            missing.append(node.name)

    return missing


async def find_documentation_gaps(project_id: int) -> dict | None:
    """
    Find the most-modified Python file that still lacks docstrings on its
    public functions, then open a proactive MR adding them.

    Steps
    -----
    1. Query MongoDB: which Python files have been modified 3+ times?
    2. For the top candidate, fetch current content from GitLab.
    3. Use the ast module to identify public functions without docstrings.
    4. If any found, ask Gemini to add Google-style docstrings throughout.
    5. Open one proactive MR with the documented file.

    Returns the result dict on success, None if nothing actionable found.
    """
    logger.info("[proactive] scanning for documentation gaps | project=%s", project_id)

    commits_col = database.get_collection("commits")

    # Step 1 — find the most frequently modified Python file
    pipeline = [
        {"$match": {"project_id": project_id}},
        {
            "$project": {
                "files": {
                    "$concatArrays": [
                        {"$ifNull": ["$files_modified", []]},
                        {"$ifNull": ["$files_added", []]},
                    ]
                }
            }
        },
        {"$unwind": "$files"},
        {"$match": {"files": re.compile(r"\.py$")}},
        {"$group": {"_id": "$files", "modification_count": {"$sum": 1}}},
        {"$match": {"modification_count": {"$gte": _MIN_FILE_MODIFICATIONS}}},
        {"$sort": {"modification_count": -1}},
        {"$limit": 5},
    ]

    candidates: list[tuple[str, int]] = []
    async for doc in commits_col.aggregate(pipeline):
        candidates.append((doc["_id"], doc["modification_count"]))

    if not candidates:
        logger.info("[proactive] no high-churn Python files found")
        return None

    default_branch = await asyncio.to_thread(_get_default_branch, project_id)

    # Step 2 — find the first candidate that still has missing docstrings
    target_file: str = ""
    target_content: str = ""
    missing_functions: list[str] = []
    modification_count: int = 0

    for file_path, mod_count in candidates:
        content = await asyncio.to_thread(
            _fetch_file_content, project_id, file_path, default_branch
        )
        if not content:
            continue

        # Step 3 — detect missing docstrings
        missing = _functions_missing_docstrings(content)
        if missing:
            target_file = file_path
            target_content = content
            missing_functions = missing
            modification_count = mod_count
            break

    if not target_file:
        logger.info("[proactive] all high-churn files are fully documented")
        return None

    # Check we haven't already opened a doc MR for this file
    ghost_col = database.get_collection("ghost_actions")
    already = await ghost_col.find_one(
        {
            "project_id": project_id,
            "action_type": "proactive_mr",
            "pattern_type": "documentation_gaps",
            "files_affected": target_file,
            "was_merged": {"$ne": True},
        }
    )
    if already:
        logger.info("[proactive] doc MR already open for %s, skipping", target_file)
        return None

    logger.info(
        "[proactive] documentation gaps in %s — %d functions missing docstrings: %s",
        target_file, len(missing_functions), missing_functions,
    )

    # Step 4 — ask Gemini to add docstrings
    documented_content = await ghost_brain.generate_code_fix(
        task=(
            f"Add Google-style docstrings to the following public functions that are "
            f"currently undocumented: {', '.join(f'`{n}`' for n in missing_functions)}. "
            "Each docstring should describe: what the function does (1 sentence), "
            "its parameters (Args section), and what it returns (Returns section). "
            "Do NOT add docstrings to private functions (those starting with `_`). "
            "Do NOT change any logic, just add the docstrings."
        ),
        original_code=target_content,
        context=(
            f"This file has been modified {modification_count} times — it is a core "
            "module that benefits most from good documentation."
        ),
    )

    if not documented_content or documented_content == target_content:
        logger.warning("[proactive] Gemini returned unchanged content for %s", target_file)
        return None

    # Step 5 — open proactive MR
    try:
        mr_url = await ghost_actions.open_proactive_mr(
            project_id=project_id,
            pattern_type="documentation_gaps",
            affected_files=[target_file],
            proposed_file_changes=[
                {"action": "update", "file_path": target_file, "content": documented_content}
            ],
        )
    except Exception as exc:
        logger.error("[proactive] open_proactive_mr failed for documentation_gaps: %s", exc)
        return None

    result = {
        "opportunity": "documentation_gaps",
        "file": target_file,
        "modification_count": modification_count,
        "functions_documented": missing_functions,
        "mr_url": mr_url,
    }
    logger.info("[proactive] documentation MR opened: %s", mr_url)
    return result


# ---------------------------------------------------------------------------
# Function 4 — Master scheduled proactive scan
# ---------------------------------------------------------------------------

async def run_scheduled_proactive_scan(project_id: int) -> dict:
    """
    Master function called every 4 hours by Cloud Scheduler.

    Priority order
    --------------
    1. Security quick wins  — credentials in live code
    2. Duplicate code       — extract shared utilities
    3. Documentation gaps   — add docstrings to churned files

    Only the highest-priority opportunity that returns something actionable
    is acted on per run.  If a proactive MR was already opened within the
    last 4 hours, the whole scan is skipped to avoid overwhelming the team.

    Returns
    -------
    dict  Scan summary including what was scanned, found, and done.
    """
    started_at = datetime.now(timezone.utc)
    logger.info(
        "[proactive] ===== scheduled proactive scan started | project=%s =====",
        project_id,
    )

    summary: dict[str, Any] = {
        "project_id": project_id,
        "started_at": started_at.isoformat(),
        "skipped": False,
        "skip_reason": None,
        "scanned": [],
        "action_taken": None,
        "result": None,
    }

    # Step 1 — rate-limit guard
    if await _recent_proactive_mr_exists(project_id):
        logger.info(
            "[proactive] proactive MR opened within last %dh — scan skipped",
            _PROACTIVE_MR_COOLDOWN_HOURS,
        )
        summary["skipped"] = True
        summary["skip_reason"] = f"proactive MR already opened within last {_PROACTIVE_MR_COOLDOWN_HOURS}h"
        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        return summary

    # Step 2 — priority: security → duplicate → documentation
    # Security quick wins
    summary["scanned"].append("security_quick_wins")
    try:
        security_results = await find_security_quick_wins(project_id)
    except Exception as exc:
        logger.error("[proactive] find_security_quick_wins failed: %s", exc)
        security_results = []

    if security_results:
        summary["action_taken"] = "security_quick_wins"
        summary["result"] = security_results
        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        logger.info(
            "[proactive] scan complete — security action taken on %d file(s)",
            len(security_results),
        )
        return summary

    logger.info("[proactive] security scan: nothing actionable, trying duplicate code")

    # Duplicate code opportunities
    summary["scanned"].append("duplicate_code")
    try:
        dup_result = await find_duplicate_code_opportunities(project_id)
    except Exception as exc:
        logger.error("[proactive] find_duplicate_code_opportunities failed: %s", exc)
        dup_result = None

    if dup_result:
        summary["action_taken"] = "duplicate_code"
        summary["result"] = dup_result
        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        logger.info("[proactive] scan complete — duplicate code MR opened")
        return summary

    logger.info("[proactive] duplicate scan: nothing actionable, trying documentation")

    # Documentation gaps
    summary["scanned"].append("documentation_gaps")
    try:
        doc_result = await find_documentation_gaps(project_id)
    except Exception as exc:
        logger.error("[proactive] find_documentation_gaps failed: %s", exc)
        doc_result = None

    if doc_result:
        summary["action_taken"] = "documentation_gaps"
        summary["result"] = doc_result
        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        logger.info("[proactive] scan complete — documentation MR opened")
        return summary

    # Nothing actionable found
    summary["action_taken"] = None
    summary["result"] = None
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    logger.info(
        "[proactive] scan complete — all %d checks ran, nothing actionable found",
        len(summary["scanned"]),
    )
    return summary
