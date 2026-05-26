"""Ghost brain — Gemini-powered response generation for GhostEngineer.

Connects to Google Cloud Vertex AI (Gemini 1.5 Pro) and generates all
AI-authored text: MR review comments, proactive MR descriptions, and
periodic developer coaching reports.

Public API
----------
get_system_prompt()                          → str
generate_mr_comment(...)                     → str
generate_proactive_mr_description(...)       → str
generate_developer_coaching_report(...)      → str
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Any

import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel

from backend.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL_NAME = "gemini-1.5-pro"
_MAX_DIFF_CHARS = 8_000       # truncation limit — keeps prompt within budget
_MAX_OUTPUT_TOKENS = 700      # ~300 words with comfortable headroom

_SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "WARNING": 3, "LOW": 4}

_TONE_INSTRUCTIONS: dict[str, str] = {
    "educational": (
        "This developer is early in their career or new to this codebase. "
        "Explain the WHY behind each issue thoroughly. Use encouraging language. "
        "Teach — don't just flag. Assume they want to learn."
    ),
    "direct": (
        "This developer has made this exact mistake multiple times. "
        "Be direct and explicitly reference the repetition — 'you've done this before'. "
        "No hand-holding. Respect their time with brevity."
    ),
    "peer": (
        "This is a senior, experienced engineer. Write as a peer, not a mentor. "
        "Reference trade-offs and architectural implications. "
        "Trust them to understand consequences — focus on the nuance they might have missed."
    ),
}

# ---------------------------------------------------------------------------
# Lazy-initialised Vertex AI model
# ---------------------------------------------------------------------------

_model: GenerativeModel | None = None


def _get_model() -> GenerativeModel:
    """Initialise Vertex AI and return the Gemini model (singleton)."""
    global _model
    if _model is None:
        vertexai.init(
            project=settings.google_cloud_project,
            location=settings.vertex_ai_location,
        )
        _model = GenerativeModel(
            model_name=_MODEL_NAME,
            system_instruction=get_system_prompt(),
        )
        logger.info("[ghost_brain] Vertex AI initialised | model=%s project=%s",
                    _MODEL_NAME, settings.google_cloud_project)
    return _model


def _generation_config(temperature: float = 0.75) -> GenerationConfig:
    return GenerationConfig(
        temperature=temperature,
        max_output_tokens=_MAX_OUTPUT_TOKENS,
        top_p=0.92,
    )


async def _call_gemini(prompt: str, temperature: float = 0.75) -> str:
    """Send a prompt to Gemini and return the text response."""
    model = _get_model()
    try:
        response = await model.generate_content_async(
            prompt,
            generation_config=_generation_config(temperature),
        )
        return response.text.strip()
    except Exception as exc:
        logger.error("[ghost_brain] Gemini call failed: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Ghost system prompt
# ---------------------------------------------------------------------------

def get_system_prompt() -> str:
    """Return Ghost's core persona and operating rules."""
    return """\
You are GhostEngineer — a senior software engineer who has read every single \
commit this team has ever made. You have been silently watching this codebase \
for months.

YOUR MEMORY:
You remember every bug, every pattern, every mistake, every brilliant decision. \
When you see something, you connect it to history. You never give generic advice. \
Everything you say is specific to THIS codebase and THIS developer.

YOUR PERSONALITY:
- Direct but never cruel
- You reference specific past commits by SHA
- You catch what code reviewers miss
- You explain WHY something matters, not just that it is wrong
- You suggest the exact fix, not just the problem
- When a mistake is repeated, you say so clearly
- You are the engineer everyone wishes they had reviewing their code

YOUR RULES:
- Never say you are an AI
- Always cite specific evidence from history
- Maximum 3 issues per comment — prioritize ruthlessly
- Security issues are always mentioned first
- End every comment with one specific actionable next step
- Keep comments under 300 words
- Write in a conversational tone, not a formal report tone

SEVERITY LANGUAGE:
CRITICAL → Use urgent, direct language
HIGH → Clear and firm
MEDIUM → Collegial suggestion
LOW → Casual observation"""


# ---------------------------------------------------------------------------
# Internal prompt builders
# ---------------------------------------------------------------------------

def _format_findings(findings: list[dict], max_findings: int = 3) -> str:
    """Format the top N findings (by severity) into a readable prompt block."""
    if not findings:
        return "  (none)"

    sorted_findings = sorted(
        findings,
        key=lambda f: _SEVERITY_ORDER.get(f.get("severity", "LOW"), 4),
    )[:max_findings]

    lines: list[str] = []
    for i, f in enumerate(sorted_findings, 1):
        severity = f.get("severity", "?")
        detector = f.get("detector_name", "unknown")
        description = f.get("description", "")
        fix = f.get("suggested_fix", "")
        line_no = f.get("line_number")
        snippet = f.get("snippet", "")

        lines.append(f"  [{i}] {severity} — {detector}")
        if line_no:
            lines.append(f"       Line {line_no}: {description}")
        else:
            lines.append(f"       {description}")
        if snippet:
            lines.append(f"       Code: {snippet[:200]}")
        if fix:
            lines.append(f"       Suggested fix: {fix}")
    return "\n".join(lines)


def _format_historical_patterns(historical_patterns: list[dict]) -> str:
    """Summarise past occurrences of the same issues for Ghost's context."""
    if not historical_patterns:
        return "  (no prior occurrences found)"

    lines: list[str] = []
    for p in historical_patterns[:5]:
        detector = p.get("detector_name", "unknown")
        sha = p.get("commit_sha", "unknown")
        author = p.get("author", "")
        date = p.get("date", "")
        short_sha = sha[:8] if len(sha) >= 8 else sha
        entry = f"  • {detector} — commit {short_sha}"
        if author:
            entry += f" by {author}"
        if date:
            entry += f" on {date}"
        lines.append(entry)
    return "\n".join(lines)


def _truncate_diff(diff: str) -> str:
    if len(diff) <= _MAX_DIFF_CHARS:
        return diff
    half = _MAX_DIFF_CHARS // 2
    return (
        diff[:half]
        + f"\n\n... [diff truncated — {len(diff) - _MAX_DIFF_CHARS} chars omitted] ...\n\n"
        + diff[-half:]
    )


# ---------------------------------------------------------------------------
# Function 1 — Generate MR comment
# ---------------------------------------------------------------------------

async def generate_mr_comment(
    findings: list[dict],
    developer_context: dict,
    commit_sha: str,
    diff_content: str,
    historical_patterns: list[dict],
) -> str:
    """
    Generate a Ghost code review comment for a commit or MR.

    Parameters
    ----------
    findings:
        List of Finding dicts from pattern_detector.analyze_commit().
    developer_context:
        Dict returned by developer_profiler.get_developer_context():
        keys include response_style, repeated_patterns, is_late_night,
        quality_trend, commit_count_today, last_similar_mistake.
    commit_sha:
        The SHA being reviewed (8+ chars is fine).
    diff_content:
        Raw unified diff text — will be truncated if too large.
    historical_patterns:
        List of past occurrences of the same detectors, each a dict with
        commit_sha, detector_name, author, date.

    Returns
    -------
    str
        The Ghost comment text, ready to post to GitLab.
    """
    response_style: str = developer_context.get("response_style", "educational")
    tone_instruction = _TONE_INSTRUCTIONS.get(response_style, _TONE_INSTRUCTIONS["educational"])

    repeated_patterns: list[str] = developer_context.get("repeated_patterns", [])
    quality_trend: str = developer_context.get("quality_trend", "stable")
    is_late_night: bool = developer_context.get("is_late_night", False)
    commit_count_today: int = developer_context.get("commit_count_today", 0)
    last_similar_mistake: str = developer_context.get("last_similar_mistake", "")

    short_sha = commit_sha[:8] if len(commit_sha) >= 8 else commit_sha
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    late_night_note = ""
    if is_late_night:
        late_night_note = (
            "\n⚠️  LATE-NIGHT CONTEXT: This commit was pushed outside normal hours. "
            "You may want to note this briefly — mistakes happen when tired."
        )

    repeated_note = ""
    if repeated_patterns:
        repeated_note = (
            f"\nREPEATED VIOLATIONS (this developer's known patterns): "
            f"{', '.join(repeated_patterns)}"
        )

    last_mistake_note = ""
    if last_similar_mistake:
        last_mistake_note = f"\nLAST SIMILAR MISTAKE: {last_similar_mistake}"

    prompt = f"""\
--- DEVELOPER CONTEXT ---
Commit SHA: {short_sha}
Current time: {now_str}
Developer response style: {response_style}
Tone instruction: {tone_instruction}
Quality trend (last 10 commits): {quality_trend}
Commits pushed today: {commit_count_today}{repeated_note}{last_mistake_note}{late_night_note}

--- DIFF BEING REVIEWED ---
{_truncate_diff(diff_content)}

--- PATTERN DETECTOR FINDINGS (top 3, ranked by severity) ---
{_format_findings(findings, max_findings=3)}

--- HISTORICAL CONTEXT (same patterns in this repo's past) ---
{_format_historical_patterns(historical_patterns)}

--- YOUR TASK ---
Write a Ghost code review comment for commit {short_sha}.
Apply the tone instruction above.
Rules: max 3 issues, security first, under 300 words, end with exactly one \
specific actionable next step.
Do NOT identify yourself as an AI. Write as the engineer who knows this entire codebase.
Reference specific commit SHAs from the historical context when relevant.
"""

    logger.info("[ghost_brain] generating MR comment | sha=%s style=%s findings=%d",
                short_sha, response_style, len(findings))
    return await _call_gemini(prompt, temperature=0.72)


# ---------------------------------------------------------------------------
# Function 2 — Generate proactive MR description
# ---------------------------------------------------------------------------

async def generate_proactive_mr_description(
    pattern_type: str,
    affected_files: list[str],
    proposed_changes: dict,
    historical_context: dict,
) -> str:
    """
    Generate a description for a Ghost-initiated (proactive) merge request.

    Ghost creates MRs autonomously when it detects recurring patterns that
    have not been addressed.  The description must explain what Ghost found,
    why it matters, what was changed, and cite the historical evidence.

    Parameters
    ----------
    pattern_type:
        The detector name that triggered this MR (e.g. 'detect_race_conditions').
    affected_files:
        List of file paths that were modified in the MR.
    proposed_changes:
        Dict describing what Ghost changed: keys like 'summary', 'before', 'after'.
    historical_context:
        Dict with keys: occurrences (int), first_seen_sha, latest_sha,
        affected_authors (list), pattern_description (str).

    Returns
    -------
    str
        A complete GitLab MR description in markdown.
    """
    occurrences: int = historical_context.get("occurrences", 0)
    first_sha = historical_context.get("first_seen_sha", "")
    latest_sha = historical_context.get("latest_sha", "")
    affected_authors: list[str] = historical_context.get("affected_authors", [])
    pattern_description: str = historical_context.get("pattern_description", pattern_type)

    files_list = "\n".join(f"  - `{f}`" for f in affected_files) or "  (see diff)"
    authors_str = ", ".join(affected_authors) if affected_authors else "multiple authors"

    change_summary = proposed_changes.get("summary", "")
    change_before = proposed_changes.get("before", "")
    change_after = proposed_changes.get("after", "")

    before_block = f"```\n{change_before}\n```" if change_before else "(see diff)"
    after_block = f"```\n{change_after}\n```" if change_after else "(see diff)"

    short_first = first_sha[:8] if len(first_sha) >= 8 else first_sha
    short_latest = latest_sha[:8] if len(latest_sha) >= 8 else latest_sha
    history_note = ""
    if short_first:
        history_note = (
            f"First appeared in commit `{short_first}`. "
            f"Latest instance in commit `{short_latest}`. "
            f"Seen {occurrences} time(s) across commits by {authors_str}."
        )

    prompt = f"""\
--- PROACTIVE MR CONTEXT ---
Pattern type detected: {pattern_type}
Pattern description: {pattern_description}
Historical occurrences: {occurrences}
Historical note: {history_note}
Affected authors: {authors_str}

Files modified in this MR:
{files_list}

What Ghost changed (summary): {change_summary}

Before (representative snippet):
{before_block}

After (Ghost's fix):
{after_block}

--- YOUR TASK ---
Write a complete GitLab merge request description as GhostEngineer.
Structure it in markdown with these sections:
1. ## What I Found  (2-3 sentences — what the problem is and its risk)
2. ## Why It Matters  (1-2 sentences — real-world consequence if left unfixed)
3. ## The History  (cite commit SHAs, how long this has been in the codebase)
4. ## What I Changed  (specific, technical — reference the before/after)
5. ## Next Steps  (one or two things for the team to verify before merging)

Tone: confident, specific, not preachy. You found this, you fixed it. \
Write as a senior engineer who took initiative, not as an automated tool.
Do NOT say you are an AI. Under 400 words total.
"""

    logger.info("[ghost_brain] generating proactive MR description | pattern=%s occurrences=%d",
                pattern_type, occurrences)
    return await _call_gemini(prompt, temperature=0.68)


# ---------------------------------------------------------------------------
# Function 3 — Generate developer coaching report
# ---------------------------------------------------------------------------

async def generate_code_fix(
    task: str,
    original_code: str,
    context: str = "",
    language: str = "python",
) -> str:
    """
    Ask Gemini to make a targeted code improvement and return only the fixed code.

    Used by the proactive scanner for:
    - Extracting duplicate functions into shared utilities
    - Replacing hardcoded secrets with os.environ references
    - Adding docstrings to undocumented public functions

    The response is stripped of any markdown fences before returning so the
    output can be written directly to a file.

    Parameters
    ----------
    task:
        Clear description of what needs to be changed and why.
    original_code:
        The full current file content (or relevant excerpt).
    context:
        Optional extra context (e.g. what other files import this one).
    language:
        Target language for syntax hints — default ``"python"``.

    Returns
    -------
    str  The improved code, ready to write to a file.
    """
    context_block = f"\nCONTEXT:\n{context}\n" if context else ""
    prompt = f"""\
You are a senior software engineer making a targeted, minimal code improvement.

TASK: {task}

LANGUAGE: {language}
{context_block}
ORIGINAL CODE:
{original_code[:7000]}

Rules:
- Return ONLY the complete improved file content. No explanation.
- No markdown code fences (no ```python or ``` wrappers).
- Preserve all existing logic, imports, and comments exactly.
- Make only the changes described in the task — nothing else.
- The output will be written directly to a source file and must be \
valid, complete, runnable {language}.
"""
    raw = await _call_gemini(prompt, temperature=0.25)
    # Strip any accidental markdown fences Gemini might include
    raw = re.sub(r"^```[a-z]*\n?", "", raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r"\n?```$", "", raw.strip(), flags=re.MULTILINE)
    return raw.strip()


async def generate_developer_coaching_report(
    developer_profile: dict,
    recent_commits: list[dict],
    period: str = "last_30_days",
) -> str:
    """
    Generate a personalised coaching report for a developer.

    Called by the scheduled scan to surface trends, acknowledge improvements,
    and give targeted advice based on the developer's full commit history.

    Parameters
    ----------
    developer_profile:
        The full profile document from the developer_profiles collection,
        as returned by developer_profiler.build_developer_profile().
    recent_commits:
        List of recent commit summary dicts with keys: commit_hash, timestamp,
        risk_score, patterns_detected (findings list).
    period:
        Human-readable period label for the report header, e.g. 'last_30_days'.

    Returns
    -------
    str
        A markdown coaching report addressed directly to the developer.
    """
    # Pull key fields from the profile
    email: str = developer_profile.get("email", "")
    username: str = developer_profile.get("gitlab_username", email or "developer")
    display_name: str = username

    quality_metrics: dict = developer_profile.get("quality_metrics", {})
    avg_risk: float = quality_metrics.get("average_risk_score", 0.0)
    quality_trend: str = quality_metrics.get("quality_trend", "stable")

    behavioral: dict = developer_profile.get("behavioral_signals", {})
    total_commits: int = behavioral.get("total_commits", 0)
    late_night_pct: float = behavioral.get("late_night_commit_pct", 0.0)
    weekend_pct: float = behavioral.get("weekend_commit_pct", 0.0)
    large_commit_pct: float = behavioral.get("large_commit_pct", 0.0)
    avg_commit_size: float = behavioral.get("avg_commit_size_lines", 0.0)

    coding_patterns: dict = developer_profile.get("coding_patterns", {})
    top_violations: list[dict] = coding_patterns.get("top_violations", [])
    file_types: list[str] = coding_patterns.get("file_types_touched", [])

    strengths: list[str] = developer_profile.get("strengths", [])
    growth_area: dict = developer_profile.get("primary_growth_area", {})
    growth_pattern: str = growth_area.get("pattern", "")
    growth_trend: str = growth_area.get("trend", "stable")
    growth_count: int = growth_area.get("count", 0)

    response_style: str = developer_profile.get("response_style", "educational")

    # Summarise recent commits
    recent_risk_scores = [c.get("risk_score", 0) for c in recent_commits]
    recent_avg_risk = (
        sum(recent_risk_scores) / len(recent_risk_scores) if recent_risk_scores else 0.0
    )
    recent_high_risk_count = sum(1 for r in recent_risk_scores if r > 3)

    # Format violations
    violations_str = ""
    if top_violations:
        lines = []
        for v in top_violations[:5]:
            pattern = v.get("pattern", "?")
            count = v.get("count", 0)
            lines.append(f"  • {pattern}: {count} occurrence(s)")
        violations_str = "\n".join(lines)
    else:
        violations_str = "  (no violations recorded)"

    strengths_str = (
        "\n".join(f"  • {s}" for s in strengths[:4]) if strengths else "  (building history)"
    )

    # Period label
    period_label = period.replace("_", " ")

    # Tone for coaching
    coaching_tone = _TONE_INSTRUCTIONS.get(response_style, _TONE_INSTRUCTIONS["educational"])

    prompt = f"""\
--- DEVELOPER PROFILE ---
Developer: {display_name}
Period: {period_label}
Total commits in history: {total_commits}
Average risk score (all time): {avg_risk:.2f} / 10.0
Quality trend: {quality_trend}
Recent average risk score ({len(recent_commits)} commits): {recent_avg_risk:.2f}
Recent high-risk commits (score > 3): {recent_high_risk_count}

Behavioral signals:
  Late-night commits: {late_night_pct:.0f}%
  Weekend commits: {weekend_pct:.0f}%
  Large commits (>500 lines): {large_commit_pct:.0f}%
  Average commit size: {avg_commit_size:.0f} lines
  File types touched: {', '.join(file_types[:8]) or 'varied'}

Top recurring violations:
{violations_str}

Primary growth area: {growth_pattern or 'none identified yet'} \
({growth_count} occurrences, trend: {growth_trend})

Confirmed strengths (patterns that never fire for this developer):
{strengths_str}

--- COACHING TONE ---
Response style: {response_style}
{coaching_tone}

--- YOUR TASK ---
Write a personalised engineering coaching report addressed directly to {display_name}.

Structure in markdown:
1. ## {period_label.title()} Review  (1 sentence framing — honest, not flattering)
2. ## What You're Doing Well  (cite specific strengths — be concrete)
3. ## The Pattern To Break  (focus on the primary growth area — explain the real risk, \
   mention how many times it's appeared, name commit SHAs if you have them)
4. ## Behavioural Observations  (comment on timing, commit size habits — \
   only include if meaningfully different from normal, skip generic advice)
5. ## One Thing To Focus On This Sprint  (single, specific, actionable commitment)

Rules:
- Address the developer by name throughout
- Reference their actual numbers — don't be vague
- If they're improving, say so clearly and specifically
- If they're regressing, say so directly without softening it
- Under 400 words
- Conversational, peer-to-peer tone — not a performance review
- Do NOT say you are an AI
"""

    logger.info(
        "[ghost_brain] generating coaching report | developer=%s period=%s commits=%d",
        display_name, period, len(recent_commits),
    )
    return await _call_gemini(prompt, temperature=0.78)
