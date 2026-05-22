"""
Pattern detection engine — analyzes commit diffs for bad coding patterns.

This module exposes 8 independent detectors plus a master ``analyze_commit``
coroutine that runs all of them, scores the result, and persists findings to
MongoDB.

Detectors
---------
1. detect_hardcoded_secrets      — API keys, passwords, AWS creds, conn strings
2. detect_missing_error_handling — bare network/IO calls without try/except
3. detect_large_functions        — functions over 50 / 100 / 200 lines
4. detect_todo_bombs             — TODO / FIXME / HACK markers in new code
5. detect_copy_paste             — near-duplicate blocks via SequenceMatcher
6. detect_missing_validation     — FastAPI endpoints bypassing Pydantic models
7. detect_race_conditions        — check-then-act and shared-state patterns
8. detect_dead_imports           — imported names never referenced in the diff

All detectors accept a raw unified-diff string (as produced by ``git diff``)
and only inspect lines that start with ``+`` (i.e. newly added lines).
"""

from __future__ import annotations

import ast
import difflib
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

from backend import database

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Severity weights used for risk-score computation
# ---------------------------------------------------------------------------

SEVERITY_WEIGHTS: dict[str, float] = {
    "CRITICAL": 3.0,
    "HIGH": 2.0,
    "MEDIUM": 1.0,
    "WARNING": 0.5,
    "LOW": 0.5,
}


# ---------------------------------------------------------------------------
# Finding dataclass
# ---------------------------------------------------------------------------


@dataclass
class Finding:
    """Represents a single pattern-detection result.

    Attributes
    ----------
    detector_name:
        Short identifier for the detector that produced this finding.
    severity:
        One of ``CRITICAL``, ``HIGH``, ``MEDIUM``, ``LOW``, ``WARNING``.
    line_number:
        1-based line number within the diff where the issue was found,
        or ``None`` when a line number cannot be determined.
    description:
        Human-readable explanation of the problem.
    suggested_fix:
        Actionable recommendation to resolve the problem.
    snippet:
        The offending line or code fragment (may be empty).
    """

    detector_name: str
    severity: str
    line_number: int | None
    description: str
    suggested_fix: str
    snippet: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _added_lines(diff_content: str) -> list[tuple[int, str]]:
    """Return ``(diff_line_number, content)`` pairs for every added line.

    Added lines are those that start with ``+`` but are *not* the ``+++``
    file-header lines produced by ``git diff``.
    """
    result: list[tuple[int, str]] = []
    for lineno, raw in enumerate(diff_content.splitlines(), start=1):
        if raw.startswith("+") and not raw.startswith("+++"):
            result.append((lineno, raw[1:]))  # strip the leading "+"
    return result


def _context_lines(diff_content: str) -> list[tuple[int, str]]:
    """Return all non-header lines (added + context, not removed) with numbers."""
    result: list[tuple[int, str]] = []
    for lineno, raw in enumerate(diff_content.splitlines(), start=1):
        if not raw.startswith("-") and not raw.startswith("@@") and not raw.startswith("\\"):
            result.append((lineno, raw.lstrip("+ ")))
    return result


# ---------------------------------------------------------------------------
# DETECTOR 1 — Hardcoded Secrets
# ---------------------------------------------------------------------------

_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("api_key assignment", re.compile(r"""api_key\s*=\s*['"][^'"]{8,}['"]""", re.IGNORECASE)),
    ("password assignment", re.compile(r"""password\s*=\s*['"][^'"]{4,}['"]""", re.IGNORECASE)),
    ("secret assignment", re.compile(r"""secret\s*=\s*['"][^'"]{8,}['"]""", re.IGNORECASE)),
    ("AWS access key", re.compile(r"""AKIA[0-9A-Z]{16}""")),
    ("token assignment", re.compile(r"""token\s*=\s*['"][a-zA-Z0-9]{20,}['"]""", re.IGNORECASE)),
    (
        "database connection string",
        re.compile(r"""(postgres|mysql|mongodb):\/\/[^:]+:[^@]+@""", re.IGNORECASE),
    ),
]


def detect_hardcoded_secrets(diff_content: str) -> list[Finding]:
    """Scan diff for hardcoded credentials and connection strings.

    Only lines starting with ``+`` in the diff are inspected so that
    pre-existing (unchanged or removed) secrets are not re-reported.

    Parameters
    ----------
    diff_content:
        Raw unified diff string.

    Returns
    -------
    list[Finding]
        One ``Finding`` per matched line per pattern, with severity
        ``CRITICAL``.
    """
    findings: list[Finding] = []
    try:
        for diff_lineno, line in _added_lines(diff_content):
            for label, pattern in _SECRET_PATTERNS:
                match = pattern.search(line)
                if match:
                    findings.append(
                        Finding(
                            detector_name="hardcoded_secrets",
                            severity="CRITICAL",
                            line_number=diff_lineno,
                            description=(
                                f"Potential hardcoded secret detected ({label}). "
                                "Secrets committed to version control can be scraped "
                                "by automated scanners within minutes."
                            ),
                            suggested_fix=(
                                "Remove the secret from source code. Store it in an "
                                "environment variable or a secrets manager (e.g. "
                                "AWS Secrets Manager, HashiCorp Vault, Doppler) and "
                                "reference it via os.environ or your config layer."
                            ),
                            snippet=line.strip(),
                        )
                    )
    except Exception:
        logger.exception("[hardcoded_secrets] unexpected error during detection")
    return findings


# ---------------------------------------------------------------------------
# DETECTOR 2 — Missing Error Handling
# ---------------------------------------------------------------------------

_PY_NETWORK_RE = re.compile(r"""requests\.(get|post|put|delete|patch|head)\s*\(""")
_PY_OPEN_RE = re.compile(r"""\bopen\s*\(""")
_PY_AWAIT_RE = re.compile(r"""\bawait\s+\w""")
_JS_FETCH_RE = re.compile(r"""\bfetch\s*\(""")
_JS_AXIOS_RE = re.compile(r"""\baxios\.""")
_JS_CATCH_RE = re.compile(r"""\.catch\s*\(|\.catch\(""")
_JS_TRY_RE = re.compile(r"""\btry\s*\{""")
_PY_TRY_RE = re.compile(r"""^\s*try\s*:""")
_PY_EXCEPT_RE = re.compile(r"""^\s*except[\s(:]""")


def _has_try_nearby(lines: list[str], index: int, window: int = 10) -> bool:
    """Return True if a ``try:`` or ``except`` appears within *window* lines."""
    start = max(0, index - window)
    end = min(len(lines), index + window + 1)
    for ln in lines[start:end]:
        if _PY_TRY_RE.search(ln) or _PY_EXCEPT_RE.search(ln):
            return True
    return False


def detect_missing_error_handling(
    diff_content: str, language: str = "python"
) -> list[Finding]:
    """Detect network and I/O calls that lack surrounding error handling.

    For Python, flags ``requests.*`` HTTP calls and ``open(`` file operations
    that have no ``try``/``except`` block in the surrounding 10-line window,
    and ``await`` calls in async functions that contain no ``try``/``except``.

    For JavaScript, flags ``fetch(`` and ``axios.`` calls that have no
    ``.catch(`` or ``try {`` nearby.

    Parameters
    ----------
    diff_content:
        Raw unified diff string.
    language:
        ``"python"`` (default) or ``"javascript"`` / ``"js"``.

    Returns
    -------
    list[Finding]
        Findings with severity ``HIGH``.
    """
    findings: list[Finding] = []
    try:
        added = _added_lines(diff_content)
        added_text = [ln for _, ln in added]
        lang = language.lower()

        if lang == "python":
            for i, (diff_lineno, line) in enumerate(added):
                # requests.* calls
                if _PY_NETWORK_RE.search(line):
                    if not _has_try_nearby(added_text, i):
                        findings.append(
                            Finding(
                                detector_name="missing_error_handling",
                                severity="HIGH",
                                line_number=diff_lineno,
                                description=(
                                    "HTTP call via `requests` is not inside a "
                                    "try/except block. Network failures, timeouts, "
                                    "and non-2xx responses will raise unhandled "
                                    "exceptions."
                                ),
                                suggested_fix=(
                                    "Wrap the call in try/except "
                                    "(requests.exceptions.RequestException) and "
                                    "handle connection errors, timeouts, and HTTP "
                                    "error responses explicitly."
                                ),
                                snippet=line.strip(),
                            )
                        )

                # open( file calls
                if _PY_OPEN_RE.search(line):
                    if not _has_try_nearby(added_text, i):
                        findings.append(
                            Finding(
                                detector_name="missing_error_handling",
                                severity="HIGH",
                                line_number=diff_lineno,
                                description=(
                                    "`open(` file operation is not inside a "
                                    "try/except block. The file may not exist, "
                                    "permissions may be denied, or the disk could "
                                    "be full."
                                ),
                                suggested_fix=(
                                    "Wrap file operations in try/except OSError (or "
                                    "use a context manager with try/except) to handle "
                                    "FileNotFoundError, PermissionError, and IOError."
                                ),
                                snippet=line.strip(),
                            )
                        )

                # await calls — check if the enclosing function has try/except
                if _PY_AWAIT_RE.search(line):
                    if not _has_try_nearby(added_text, i, window=20):
                        findings.append(
                            Finding(
                                detector_name="missing_error_handling",
                                severity="HIGH",
                                line_number=diff_lineno,
                                description=(
                                    "`await` expression is used inside an async "
                                    "function without a visible try/except block. "
                                    "Awaited coroutines may raise and crash the task."
                                ),
                                suggested_fix=(
                                    "Surround `await` calls with try/except and "
                                    "handle expected exception types. Consider "
                                    "asyncio.shield for operations that must not be "
                                    "cancelled."
                                ),
                                snippet=line.strip(),
                            )
                        )

        elif lang in ("javascript", "js", "typescript", "ts"):
            for i, (diff_lineno, line) in enumerate(added):
                if _JS_FETCH_RE.search(line) or _JS_AXIOS_RE.search(line):
                    window_lines = [ln for _, ln in added[max(0, i - 5) : i + 6]]
                    has_catch = any(_JS_CATCH_RE.search(wl) for wl in window_lines)
                    has_try = any(_JS_TRY_RE.search(wl) for wl in window_lines)
                    if not has_catch and not has_try:
                        label = "fetch" if _JS_FETCH_RE.search(line) else "axios"
                        findings.append(
                            Finding(
                                detector_name="missing_error_handling",
                                severity="HIGH",
                                line_number=diff_lineno,
                                description=(
                                    f"`{label}` network call has no `.catch()` "
                                    "handler or enclosing try/catch block."
                                ),
                                suggested_fix=(
                                    f"Chain `.catch(err => ...)` on the `{label}` "
                                    "promise, or wrap the call in a try/catch block "
                                    "inside an async function."
                                ),
                                snippet=line.strip(),
                            )
                        )
    except Exception:
        logger.exception("[missing_error_handling] unexpected error during detection")
    return findings


# ---------------------------------------------------------------------------
# DETECTOR 3 — Large Functions
# ---------------------------------------------------------------------------

_PY_FUNC_RE = re.compile(r"""^(\s*)(async\s+)?def\s+(\w+)\s*\(""")
_JS_FUNC_RE = re.compile(
    r"""^(\s*)(function\s+\w+\s*\(|(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?(?:\([^)]*\)|\w+)\s*=>)"""
)


def detect_large_functions(diff_content: str) -> list[Finding]:
    """Detect overly large functions added in the diff.

    Counts lines for each ``def``/``async def`` (Python) or ``function``/
    arrow function (JavaScript) found in the added lines.  Thresholds:

    * 50+ lines → ``WARNING``
    * 100+ lines → ``HIGH``
    * 200+ lines → ``CRITICAL``

    Parameters
    ----------
    diff_content:
        Raw unified diff string.

    Returns
    -------
    list[Finding]
        One finding per oversized function.
    """
    findings: list[Finding] = []
    try:
        added = _added_lines(diff_content)
        if not added:
            return findings

        # Detect function start positions and their indentation levels
        func_starts: list[tuple[int, int, str, int]] = []
        # (index_in_added, diff_lineno, func_name, indent_len)
        for i, (diff_lineno, line) in enumerate(added):
            m_py = _PY_FUNC_RE.match(line)
            if m_py:
                indent = len(m_py.group(1))
                name = m_py.group(3)
                func_starts.append((i, diff_lineno, name, indent))
                continue
            m_js = _JS_FUNC_RE.match(line)
            if m_js:
                indent = len(m_js.group(1))
                # Extract a best-effort name from the match
                raw = m_js.group(2).strip()
                name = raw.split("(")[0].split("=")[0].strip().split()[-1]
                func_starts.append((i, diff_lineno, name, indent))

        for j, (start_idx, diff_lineno, name, indent) in enumerate(func_starts):
            # Find end: next function at same or shallower indent, or end of list
            end_idx = len(added)
            for k in range(j + 1, len(func_starts)):
                next_indent = func_starts[k][3]
                if next_indent <= indent:
                    end_idx = func_starts[k][0]
                    break

            line_count = end_idx - start_idx

            if line_count >= 200:
                severity = "CRITICAL"
            elif line_count >= 100:
                severity = "HIGH"
            elif line_count >= 50:
                severity = "WARNING"
            else:
                continue

            findings.append(
                Finding(
                    detector_name="large_functions",
                    severity=severity,
                    line_number=diff_lineno,
                    description=(
                        f"Function `{name}` is {line_count} lines long. "
                        "Large functions are harder to test, understand, and maintain."
                    ),
                    suggested_fix=(
                        "Break the function into smaller, single-responsibility "
                        "functions of 20–40 lines each. Extract cohesive logic "
                        "blocks into well-named helpers."
                    ),
                    snippet=added[start_idx][1].strip(),
                )
            )
    except Exception:
        logger.exception("[large_functions] unexpected error during detection")
    return findings


# ---------------------------------------------------------------------------
# DETECTOR 4 — TODO Bombs
# ---------------------------------------------------------------------------

_TODO_RE = re.compile(
    r"""(?:#|//|/\*).*?\b(TODO|FIXME|HACK|XXX|TEMP)\b(.*)""", re.IGNORECASE
)


def detect_todo_bombs(
    diff_content: str,
    commit_timestamp: Optional[datetime] = None,
) -> list[Finding]:
    """Find TODO / FIXME / HACK / XXX / TEMP markers in added lines.

    If *commit_timestamp* is provided and is more than 30 days in the past
    the severity is ``HIGH`` (stale technical debt), otherwise ``LOW``.

    Parameters
    ----------
    diff_content:
        Raw unified diff string.
    commit_timestamp:
        Optional timestamp of the commit being analysed.  Used to determine
        whether the TODO is stale (>30 days old).

    Returns
    -------
    list[Finding]
        One finding per TODO marker found in added lines.
    """
    findings: list[Finding] = []
    try:
        stale = False
        if commit_timestamp is not None:
            now = datetime.now(timezone.utc)
            ts = commit_timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            stale = (now - ts).days > 30

        for diff_lineno, line in _added_lines(diff_content):
            m = _TODO_RE.search(line)
            if m:
                keyword = m.group(1).upper()
                comment_text = (m.group(0)).strip()
                severity = "HIGH" if stale else "LOW"
                findings.append(
                    Finding(
                        detector_name="todo_bombs",
                        severity=severity,
                        line_number=diff_lineno,
                        description=(
                            f"`{keyword}` comment added to codebase"
                            + (" (stale — commit is >30 days old)" if stale else "")
                            + f": {comment_text}"
                        ),
                        suggested_fix=(
                            f"Create a tracked issue for this `{keyword}` and link "
                            "it in the comment, or resolve it before merging. "
                            "Unresolved markers accumulate as technical debt."
                        ),
                        snippet=line.strip(),
                    )
                )
    except Exception:
        logger.exception("[todo_bombs] unexpected error during detection")
    return findings


# ---------------------------------------------------------------------------
# DETECTOR 5 — Copy-Paste Code
# ---------------------------------------------------------------------------


def _extract_added_blocks(diff_content: str, min_lines: int = 10) -> list[list[str]]:
    """Extract contiguous runs of added lines that are at least *min_lines* long."""
    added = _added_lines(diff_content)
    if not added:
        return []

    blocks: list[list[str]] = []
    current: list[str] = []
    prev_lineno = -2

    for diff_lineno, line in added:
        if diff_lineno == prev_lineno + 1:
            current.append(line)
        else:
            if len(current) >= min_lines:
                blocks.append(current)
            current = [line]
        prev_lineno = diff_lineno

    if len(current) >= min_lines:
        blocks.append(current)

    return blocks


def detect_copy_paste(
    diff_content: str,
    existing_samples: list[str],
) -> list[Finding]:
    """Detect blocks of added code that are near-duplicates of existing samples.

    Uses ``difflib.SequenceMatcher`` with a similarity threshold of 0.85.
    Only blocks of 10+ consecutive added lines are compared.

    Parameters
    ----------
    diff_content:
        Raw unified diff string.
    existing_samples:
        List of code strings (e.g. previously seen function bodies) to
        compare against.

    Returns
    -------
    list[Finding]
        Findings with severity ``MEDIUM``.
    """
    findings: list[Finding] = []
    try:
        blocks = _extract_added_blocks(diff_content, min_lines=10)
        if not blocks or not existing_samples:
            return findings

        for block in blocks:
            block_text = "\n".join(block)
            for sample in existing_samples:
                ratio = difflib.SequenceMatcher(
                    None, block_text, sample, autojunk=False
                ).ratio()
                if ratio >= 0.85:
                    pct = int(ratio * 100)
                    findings.append(
                        Finding(
                            detector_name="copy_paste",
                            severity="MEDIUM",
                            line_number=None,
                            description=(
                                f"Added block of {len(block)} lines is {pct}% "
                                "similar to an existing code sample. "
                                "Copy-pasted code creates maintenance burden and "
                                "spreads bugs."
                            ),
                            suggested_fix=(
                                "Extract the duplicated logic into a shared helper "
                                "function or module and call it from both sites."
                            ),
                            snippet=block[0].strip(),
                        )
                    )
                    break  # one finding per block is enough
    except Exception:
        logger.exception("[copy_paste] unexpected error during detection")
    return findings


# ---------------------------------------------------------------------------
# DETECTOR 6 — Missing Input Validation
# ---------------------------------------------------------------------------

_ROUTE_DECORATOR_RE = re.compile(
    r"""@(?:app|router)\.(post|put|patch|delete|get)\s*\(""", re.IGNORECASE
)
_RAW_REQUEST_RE = re.compile(
    r"""\brequest\.(json|body|form|data)\s*\(""", re.IGNORECASE
)
_REQUEST_PARAM_RE = re.compile(r"""\brequest\s*:\s*Request\b""")
_COLLECTION_WRITE_RE = re.compile(
    r"""\bcollection\.(insert_one|insert_many|update_one|update_many|replace_one)\s*\("""
)
_PYDANTIC_PARAM_RE = re.compile(
    r""":\s*[A-Z]\w+Model\b|:\s*[A-Z]\w+Schema\b|:\s*[A-Z]\w+Request\b|:\s*[A-Z]\w+\b(?!.*Request\b)"""
)


def detect_missing_validation(diff_content: str) -> list[Finding]:
    """Find FastAPI endpoints or MongoDB writes that skip input validation.

    Flags:
    * ``@app.post`` / ``@router.post`` / ``@app.put`` decorator followed within
      5 lines by a function that takes ``request: Request`` instead of a Pydantic
      model.
    * Direct ``request.json()`` / ``request.body()`` usage.
    * ``collection.insert`` / ``collection.update`` calls where the data appears
      to come directly from a raw request.

    Parameters
    ----------
    diff_content:
        Raw unified diff string.

    Returns
    -------
    list[Finding]
        Findings with severity ``HIGH``.
    """
    findings: list[Finding] = []
    try:
        added = _added_lines(diff_content)
        added_text = [ln for _, ln in added]

        for i, (diff_lineno, line) in enumerate(added):
            # Raw request.json() / request.body() usage
            if _RAW_REQUEST_RE.search(line):
                findings.append(
                    Finding(
                        detector_name="missing_validation",
                        severity="HIGH",
                        line_number=diff_lineno,
                        description=(
                            "Raw `request.json()` / `request.body()` used without "
                            "passing data through a Pydantic model. Untrusted input "
                            "reaches the application without schema validation."
                        ),
                        suggested_fix=(
                            "Define a Pydantic `BaseModel` for the request body and "
                            "declare it as a typed function parameter. FastAPI will "
                            "validate and parse it automatically."
                        ),
                        snippet=line.strip(),
                    )
                )

            # Route decorator followed by request: Request parameter
            if _ROUTE_DECORATOR_RE.search(line):
                lookahead = added_text[i + 1 : i + 6]
                for la_line in lookahead:
                    if _REQUEST_PARAM_RE.search(la_line):
                        findings.append(
                            Finding(
                                detector_name="missing_validation",
                                severity="HIGH",
                                line_number=diff_lineno,
                                description=(
                                    "Route handler receives `request: Request` "
                                    "directly instead of a typed Pydantic model. "
                                    "Input is not validated or sanitised."
                                ),
                                suggested_fix=(
                                    "Replace the `Request` parameter with a Pydantic "
                                    "`BaseModel` subclass that describes the expected "
                                    "body schema."
                                ),
                                snippet=line.strip(),
                            )
                        )
                        break

            # collection.insert/update with request data nearby
            if _COLLECTION_WRITE_RE.search(line):
                nearby = added_text[max(0, i - 5) : i + 1]
                has_raw_request = any(_RAW_REQUEST_RE.search(nl) for nl in nearby)
                if has_raw_request:
                    findings.append(
                        Finding(
                            detector_name="missing_validation",
                            severity="HIGH",
                            line_number=diff_lineno,
                            description=(
                                "MongoDB write operation appears to use data taken "
                                "directly from a raw request object without "
                                "Pydantic validation."
                            ),
                            suggested_fix=(
                                "Parse and validate the request body with a Pydantic "
                                "model before passing it to the database layer. Call "
                                "`model.model_dump()` to get a clean dict."
                            ),
                            snippet=line.strip(),
                        )
                    )
    except Exception:
        logger.exception("[missing_validation] unexpected error during detection")
    return findings


# ---------------------------------------------------------------------------
# DETECTOR 7 — Race Conditions
# ---------------------------------------------------------------------------

_ASYNC_DEF_RE = re.compile(r"""^\s*async\s+def\s+\w+""")
_AWAIT_RE = re.compile(r"""\bawait\s+""")
_ASYNCIO_SLEEP_RE = re.compile(r"""\bawait\s+asyncio\.sleep\s*\(""")
_IF_VAR_RE = re.compile(r"""^\s*if\s+(\w+)\s*""")
_ASSIGN_RE = re.compile(r"""^\s*(\w+)\s*(?:\+=|-=|\*=|/=|=(?!=))""")
_GLOBAL_RE = re.compile(r"""^\s*global\s+(\w+)""")


def detect_race_conditions(
    diff_content: str, language: str = "python"
) -> list[Finding]:
    """Detect potential race conditions introduced by the diff.

    Looks for:
    * A global/shared variable that is read (``if x:``) then written
      (``x =``) inside the same async block without an asyncio lock
      (check-then-act pattern).
    * ``asyncio.sleep`` inside a block that also modifies shared state.
    * Multiple ``await`` calls that modify the same variable name.

    Parameters
    ----------
    diff_content:
        Raw unified diff string.
    language:
        ``"python"`` (default).  JavaScript support is planned.

    Returns
    -------
    list[Finding]
        Findings with severity ``CRITICAL``.
    """
    findings: list[Finding] = []
    try:
        lang = language.lower()
        if lang != "python":
            return findings

        added = _added_lines(diff_content)
        added_text = [ln for _, ln in added]

        # Track globals declared in the diff
        global_vars: set[str] = set()
        for _, ln in added:
            gm = _GLOBAL_RE.match(ln)
            if gm:
                for gv in gm.group(1).split(","):
                    global_vars.add(gv.strip())

        # Find async function boundaries in added lines
        async_blocks: list[tuple[int, int]] = []  # (start_idx, end_idx) in added list
        i = 0
        while i < len(added):
            if _ASYNC_DEF_RE.match(added[i][1]):
                start = i
                base_indent = len(added[i][1]) - len(added[i][1].lstrip())
                end = len(added)
                for j in range(i + 1, len(added)):
                    line_content = added[j][1]
                    if line_content.strip() == "":
                        continue
                    line_indent = len(line_content) - len(line_content.lstrip())
                    # Next function/class at same or shallower level ends block
                    if line_indent <= base_indent and re.match(
                        r"""\s*(def |async def |class )""", line_content
                    ):
                        end = j
                        break
                async_blocks.append((start, end))
                i = end
            else:
                i += 1

        for block_start, block_end in async_blocks:
            block_lines = added_text[block_start:block_end]
            block_linenos = [ln for ln, _ in added[block_start:block_end]]

            # --- Check-then-act ---
            read_vars: dict[str, int] = {}  # var_name -> line_index
            for idx, line in enumerate(block_lines):
                m_if = _IF_VAR_RE.match(line)
                if m_if:
                    read_vars[m_if.group(1)] = idx

            for idx, line in enumerate(block_lines):
                m_assign = _ASSIGN_RE.match(line)
                if m_assign:
                    var_name = m_assign.group(1)
                    if var_name in read_vars and read_vars[var_name] < idx:
                        diff_lineno = (
                            block_linenos[idx] if idx < len(block_linenos) else None
                        )
                        findings.append(
                            Finding(
                                detector_name="race_conditions",
                                severity="CRITICAL",
                                line_number=diff_lineno,
                                description=(
                                    f"Check-then-act race condition on variable "
                                    f"`{var_name}` inside async function. Variable is "
                                    "read in a conditional and then written without an "
                                    "asyncio.Lock, creating a TOCTOU window."
                                ),
                                suggested_fix=(
                                    f"Protect the read-modify-write sequence for "
                                    f"`{var_name}` with an `asyncio.Lock` (or "
                                    "`asyncio.Semaphore`). Acquire the lock before the "
                                    "check and release it after the write."
                                ),
                                snippet=line.strip(),
                            )
                        )

            # --- asyncio.sleep with shared-state modification ---
            has_sleep = any(_ASYNCIO_SLEEP_RE.search(bl) for bl in block_lines)
            if has_sleep:
                for idx, line in enumerate(block_lines):
                    m_assign = _ASSIGN_RE.match(line)
                    if m_assign:
                        var_name = m_assign.group(1)
                        if var_name in global_vars or any(
                            re.search(rf"""\bglobal\s+.*\b{re.escape(var_name)}\b""", bl)
                            for bl in block_lines
                        ):
                            diff_lineno = (
                                block_linenos[idx] if idx < len(block_linenos) else None
                            )
                            findings.append(
                                Finding(
                                    detector_name="race_conditions",
                                    severity="CRITICAL",
                                    line_number=diff_lineno,
                                    description=(
                                        f"Shared state variable `{var_name}` is "
                                        "modified in an async function that also calls "
                                        "`asyncio.sleep`. Other coroutines may observe "
                                        "inconsistent state during the sleep window."
                                    ),
                                    suggested_fix=(
                                        "Use an `asyncio.Lock` around modifications to "
                                        f"`{var_name}`, or redesign to avoid shared "
                                        "mutable state in async contexts."
                                    ),
                                    snippet=line.strip(),
                                )
                            )

            # --- Multiple awaits modifying the same variable ---
            await_writes: dict[str, list[int]] = {}
            for idx, line in enumerate(block_lines):
                if _AWAIT_RE.search(line):
                    m_assign = _ASSIGN_RE.match(line)
                    if m_assign:
                        var_name = m_assign.group(1)
                        await_writes.setdefault(var_name, []).append(idx)

            for var_name, indices in await_writes.items():
                if len(indices) >= 2:
                    diff_lineno = (
                        block_linenos[indices[0]]
                        if indices[0] < len(block_linenos)
                        else None
                    )
                    findings.append(
                        Finding(
                            detector_name="race_conditions",
                            severity="CRITICAL",
                            line_number=diff_lineno,
                            description=(
                                f"Variable `{var_name}` is written by multiple "
                                "`await` expressions in the same async function "
                                "without synchronisation. Interleaved coroutine "
                                "scheduling can produce lost updates."
                            ),
                            suggested_fix=(
                                f"Collect all awaited results first, then assign "
                                f"to `{var_name}` once, or use an `asyncio.Lock` to "
                                "serialise access."
                            ),
                            snippet=block_lines[indices[0]].strip(),
                        )
                    )

    except Exception:
        logger.exception("[race_conditions] unexpected error during detection")
    return findings


# ---------------------------------------------------------------------------
# DETECTOR 8 — Dead Imports
# ---------------------------------------------------------------------------

_IMPORT_RE = re.compile(r"""^\s*import\s+(\S+)(?:\s+as\s+(\S+))?""")
_FROM_IMPORT_RE = re.compile(
    r"""^\s*from\s+\S+\s+import\s+(.+)"""
)


def _parse_imported_names(line: str) -> list[str]:
    """Return all top-level names imported by a single import line."""
    m_import = _IMPORT_RE.match(line)
    if m_import:
        # `import foo as bar` — the usable name is the alias
        alias = m_import.group(2)
        module = m_import.group(1).split(".")[0]
        return [alias if alias else module]

    m_from = _FROM_IMPORT_RE.match(line)
    if m_from:
        names_part = m_from.group(1)
        # Handle `from x import (a, b, c)` and `from x import a, b as c`
        names_part = names_part.strip().strip("()")
        names: list[str] = []
        for part in names_part.split(","):
            part = part.strip()
            if not part:
                continue
            tokens = part.split()
            if len(tokens) >= 3 and tokens[1].lower() == "as":
                names.append(tokens[2])
            else:
                names.append(tokens[0])
        return names

    return []


def detect_dead_imports(diff_content: str) -> list[Finding]:
    """Find imported names that are never used in the added lines of the diff.

    Extracts every ``import`` and ``from … import`` statement from added lines,
    then checks whether the imported name appears anywhere else in the added
    lines as an identifier.

    Parameters
    ----------
    diff_content:
        Raw unified diff string.

    Returns
    -------
    list[Finding]
        Findings with severity ``LOW``.
    """
    findings: list[Finding] = []
    try:
        added = _added_lines(diff_content)
        # Separate import lines from usage lines
        import_lines: list[tuple[int, str, list[str]]] = []
        # (diff_lineno, raw_line, [imported_names])
        non_import_text: str = ""

        for diff_lineno, line in added:
            names = _parse_imported_names(line)
            if names:
                import_lines.append((diff_lineno, line, names))
            else:
                non_import_text += " " + line

        # Build a combined string of all added non-import lines for usage search
        # Also include the import lines themselves so `import os; os.path.join` works
        all_added_text = " ".join(ln for _, ln in added)

        for diff_lineno, raw_line, names in import_lines:
            for name in names:
                if not name or name == "*":
                    continue
                # Look for the name used as an identifier outside the import line
                # itself — use word-boundary regex
                pattern = re.compile(rf"""\b{re.escape(name)}\b""")
                # Remove the import line itself from the search target
                search_target = all_added_text.replace(raw_line, "", 1)
                if not pattern.search(search_target):
                    findings.append(
                        Finding(
                            detector_name="dead_imports",
                            severity="LOW",
                            line_number=diff_lineno,
                            description=(
                                f"`{name}` is imported but never used in the "
                                "added lines of this diff. Dead imports add noise "
                                "and slow down module loading."
                            ),
                            suggested_fix=(
                                f"Remove the import of `{name}`, or use it. "
                                "Run `ruff check --select F401` to catch all "
                                "unused imports automatically."
                            ),
                            snippet=raw_line.strip(),
                        )
                    )
    except Exception:
        logger.exception("[dead_imports] unexpected error during detection")
    return findings


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------


async def analyze_commit(
    commit_sha: str,
    diff_content: str,
    author: str,
    timestamp: datetime,
    language: str = "python",
    existing_samples: list[str] | None = None,
) -> dict:
    """Run all 8 pattern detectors on a commit diff and persist results.

    Each detector is run independently; a crash in one detector is logged and
    ignored so the remaining detectors always run.

    The ``risk_score`` is computed as::

        score = sum(SEVERITY_WEIGHTS[f.severity] for f in findings)
        risk_score = min(score, 10.0)

    Results are upserted into the MongoDB ``commits`` collection under the
    ``patterns_detected`` field, keyed by ``commit_sha``.

    Parameters
    ----------
    commit_sha:
        The full SHA of the commit being analysed.
    diff_content:
        Raw unified diff string (concatenated across all changed files).
    author:
        Committer name / email for logging purposes.
    timestamp:
        Authoring timestamp of the commit.
    language:
        Primary language of the diff — used by detectors that are
        language-sensitive.  Defaults to ``"python"``.
    existing_samples:
        Optional list of existing code strings for the copy-paste detector.
        Pass ``None`` or an empty list to skip copy-paste detection.

    Returns
    -------
    dict
        Keys: ``commit_sha``, ``risk_score``, ``total_findings``,
        ``findings`` (list of Finding dicts), ``analyzed_at``.
    """
    logger.info(
        "[analyze_commit] starting analysis sha=%s author=%s", commit_sha, author
    )

    all_findings: list[Finding] = []

    # ---- Run each detector, catching individual failures ----
    detectors = [
        ("hardcoded_secrets", lambda: detect_hardcoded_secrets(diff_content)),
        (
            "missing_error_handling",
            lambda: detect_missing_error_handling(diff_content, language),
        ),
        ("large_functions", lambda: detect_large_functions(diff_content)),
        (
            "todo_bombs",
            lambda: detect_todo_bombs(diff_content, timestamp),
        ),
        (
            "copy_paste",
            lambda: detect_copy_paste(diff_content, existing_samples or []),
        ),
        ("missing_validation", lambda: detect_missing_validation(diff_content)),
        (
            "race_conditions",
            lambda: detect_race_conditions(diff_content, language),
        ),
        ("dead_imports", lambda: detect_dead_imports(diff_content)),
    ]

    for name, detector_fn in detectors:
        try:
            results = detector_fn()
            all_findings.extend(results)
            logger.debug(
                "[analyze_commit] %s produced %d finding(s)", name, len(results)
            )
        except Exception:
            logger.exception(
                "[analyze_commit] detector '%s' raised an unhandled exception", name
            )

    # ---- Compute risk score ----
    raw_score = sum(SEVERITY_WEIGHTS.get(f.severity, 0) for f in all_findings)
    risk_score = round(min(raw_score, 10.0), 2)

    analyzed_at = datetime.now(timezone.utc)
    findings_dicts = [asdict(f) for f in all_findings]

    result = {
        "commit_sha": commit_sha,
        "risk_score": risk_score,
        "total_findings": len(all_findings),
        "findings": findings_dicts,
        "analyzed_at": analyzed_at,
    }

    # ---- Persist to MongoDB ----
    try:
        collection = database.get_collection("commits")
        await collection.update_one(
            {"commit_hash": commit_sha},
            {
                "$set": {
                    "patterns_detected": {
                        "risk_score": risk_score,
                        "total_findings": len(all_findings),
                        "findings": findings_dicts,
                        "analyzed_at": analyzed_at,
                        "author": author,
                    }
                }
            },
            upsert=True,
        )
        logger.info(
            "[analyze_commit] persisted findings sha=%s risk_score=%.2f findings=%d",
            commit_sha,
            risk_score,
            len(all_findings),
        )
    except Exception:
        logger.exception(
            "[analyze_commit] failed to persist findings for sha=%s", commit_sha
        )

    return result
