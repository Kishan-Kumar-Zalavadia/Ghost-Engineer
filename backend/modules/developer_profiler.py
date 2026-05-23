"""Developer profile builder.

Reads commit history from MongoDB and builds a rich behavioral profile for
each developer — coding patterns, quality trends, timing signals, strengths,
and growth areas.  Profiles are stored in the ``developer_profiles`` collection
and incrementally updated on each new commit.

Public API
----------
build_developer_profile(gitlab_username)
    Full rebuild from all commits in MongoDB.

update_developer_profile(gitlab_username, new_commit_data)
    Incremental update — does not re-scan all commits.

get_developer_context(gitlab_username, current_time)
    Lightweight context object consumed by the AI comment agent.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

from backend import database

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LATE_NIGHT_START = 21   # 9 PM
_LATE_NIGHT_END = 5      # 5 AM

_RESPONSE_STYLE_EDUCATIONAL_MAX_COMMITS = 10
_RESPONSE_STYLE_PEER_MIN_COMMITS = 100
_RESPONSE_STYLE_PEER_MAX_AVG_RISK = 4.0
_REPEATED_VIOLATION_THRESHOLD = 3   # same pattern N+ times → "direct" style

_TREND_WINDOW = 10       # last N commits for trend calculation
_QUALITY_STABLE_DELTA = 0.5   # avg risk change < this = stable

# Severity weights — mirrored from pattern_detector for scoring
_SEVERITY_WEIGHTS: dict[str, float] = {
    "CRITICAL": 3.0,
    "HIGH": 2.0,
    "MEDIUM": 1.0,
    "WARNING": 0.5,
    "LOW": 0.5,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _hour_of(dt: datetime | None) -> int | None:
    if dt is None:
        return None
    return _ensure_utc(dt).hour  # type: ignore[union-attr]


def _is_late_night(hour: int) -> bool:
    return hour >= _LATE_NIGHT_START or hour < _LATE_NIGHT_END


def _is_weekend(dt: datetime | None) -> bool:
    if dt is None:
        return False
    return _ensure_utc(dt).weekday() >= 5  # type: ignore[union-attr]


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _quality_trend(risk_scores: list[float]) -> str:
    """Compute trend from a list of risk scores ordered oldest → newest."""
    if len(risk_scores) < 2:
        return "stable"
    half = len(risk_scores) // 2
    older_avg = _avg(risk_scores[:half])
    newer_avg = _avg(risk_scores[half:])
    delta = newer_avg - older_avg
    if delta > _QUALITY_STABLE_DELTA:
        return "declining"
    if delta < -_QUALITY_STABLE_DELTA:
        return "improving"
    return "stable"


def _extension(filename: str) -> str:
    parts = filename.rsplit(".", 1)
    return f".{parts[1]}" if len(parts) == 2 else "unknown"


# ---------------------------------------------------------------------------
# Core computations (pure — work on a list of commit dicts)
# ---------------------------------------------------------------------------

def _compute_coding_patterns(commits: list[dict]) -> dict[str, Any]:
    """Aggregate pattern violations across commits."""
    pattern_counter: Counter = Counter()
    pattern_recurrence: dict[str, list[str]] = defaultdict(list)  # pattern → [sha, ...]
    file_type_counter: Counter = Counter()

    for commit in commits:
        sha = commit.get("commit_hash", "")
        pd = commit.get("patterns_detected") or {}
        for finding in pd.get("findings", []):
            name = finding.get("detector_name", "unknown")
            pattern_counter[name] += 1
            pattern_recurrence[name].append(sha)

        # File types
        for fname in (
            commit.get("files_changed", [])
            + commit.get("files_added", [])
            + commit.get("files_modified", [])
        ):
            file_type_counter[_extension(str(fname))] += 1

    top_violations = [
        {"pattern": name, "count": count}
        for name, count in pattern_counter.most_common(10)
    ]

    # Recurrence rate: % of commits that triggered each pattern
    total = max(len(commits), 1)
    recurrence_rates = {
        name: round(len(shas) / total * 100, 1)
        for name, shas in pattern_recurrence.items()
    }

    return {
        "top_violations": top_violations,
        "pattern_counts": dict(pattern_counter),
        "recurrence_rates": recurrence_rates,
        "file_types_touched": dict(file_type_counter.most_common(20)),
    }


def _compute_quality_metrics(commits: list[dict]) -> dict[str, Any]:
    """Risk scores, trends, and best/worst times of day."""
    # Sort oldest → newest for trend
    sorted_commits = sorted(
        commits,
        key=lambda c: _ensure_utc(c.get("authored_date") or c.get("committed_date")) or datetime.min.replace(tzinfo=timezone.utc),
    )

    risk_scores_by_time: list[float] = []
    risk_by_hour: dict[int, list[float]] = defaultdict(list)
    risk_by_dow: dict[int, list[float]] = defaultdict(list)  # 0=Mon

    for commit in sorted_commits:
        pd = commit.get("patterns_detected") or {}
        risk = pd.get("risk_score")
        if risk is None:
            continue
        risk = float(risk)
        risk_scores_by_time.append(risk)

        dt = _ensure_utc(commit.get("authored_date") or commit.get("committed_date"))
        if dt:
            risk_by_hour[dt.hour].append(risk)
            risk_by_dow[dt.weekday()].append(risk)

    avg_risk = round(_avg(risk_scores_by_time), 2)
    trend = _quality_trend(risk_scores_by_time[-_TREND_WINDOW:])

    # Best / worst hour
    hour_avgs = {h: _avg(scores) for h, scores in risk_by_hour.items()}
    best_hour = min(hour_avgs, key=hour_avgs.get) if hour_avgs else None   # type: ignore[arg-type]
    worst_hour = max(hour_avgs, key=hour_avgs.get) if hour_avgs else None  # type: ignore[arg-type]

    dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_avgs = {dow_names[d]: round(_avg(s), 2) for d, s in risk_by_dow.items()}

    return {
        "average_risk_score": avg_risk,
        "quality_trend": trend,
        "risk_scores_over_time": risk_scores_by_time,
        "best_hour_of_day": best_hour,
        "worst_hour_of_day": worst_hour,
        "risk_by_hour": {h: round(_avg(s), 2) for h, s in risk_by_hour.items()},
        "risk_by_day_of_week": dow_avgs,
    }


def _compute_behavioral_signals(commits: list[dict]) -> dict[str, Any]:
    """Commit size, frequency, late-night/weekend stats."""
    lines_changed: list[int] = []
    late_night_count = 0
    weekend_count = 0
    large_commit_count = 0   # >500 lines
    small_commit_count = 0   # <=50 lines

    for commit in commits:
        added = commit.get("lines_added", 0) or 0
        removed = commit.get("lines_removed", 0) or 0
        total_lines = int(added) + int(removed)
        lines_changed.append(total_lines)

        if total_lines > 500:
            large_commit_count += 1
        if total_lines <= 50:
            small_commit_count += 1

        dt = _ensure_utc(commit.get("authored_date") or commit.get("committed_date"))
        if dt:
            if _is_late_night(dt.hour):
                late_night_count += 1
            if _is_weekend(dt):
                weekend_count += 1

    total = max(len(commits), 1)
    avg_size = round(_avg([float(x) for x in lines_changed]), 1)

    return {
        "total_commits": len(commits),
        "average_commit_size_lines": avg_size,
        "large_commit_percentage": round(large_commit_count / total * 100, 1),
        "small_focused_commit_percentage": round(small_commit_count / total * 100, 1),
        "late_night_commit_percentage": round(late_night_count / total * 100, 1),
        "weekend_commit_percentage": round(weekend_count / total * 100, 1),
    }


def _compute_strengths_and_growth(
    commits: list[dict],
    pattern_counts: dict[str, int],
    all_detector_names: list[str],
) -> dict[str, Any]:
    """Find consistent strengths and the single most repeated growth area."""
    # Strengths: detectors that NEVER fired
    strengths = [d for d in all_detector_names if pattern_counts.get(d, 0) == 0]

    # Areas where risk is consistently low (bottom 25% of avg risk per detector type)
    # — approximated by detectors with count == 0
    low_risk_areas = strengths  # same set for now

    # Growth area: most repeated pattern
    growth_area: dict[str, Any] = {}
    if pattern_counts:
        worst_pattern = max(pattern_counts, key=pattern_counts.get)  # type: ignore[arg-type]
        count = pattern_counts[worst_pattern]

        # Trend for this specific pattern — is it appearing in recent commits?
        sorted_commits = sorted(
            commits,
            key=lambda c: _ensure_utc(c.get("authored_date") or c.get("committed_date")) or datetime.min.replace(tzinfo=timezone.utc),
        )
        half = max(len(sorted_commits) // 2, 1)
        older_count = sum(
            1 for c in sorted_commits[:half]
            if any(
                f.get("detector_name") == worst_pattern
                for f in (c.get("patterns_detected") or {}).get("findings", [])
            )
        )
        newer_count = sum(
            1 for c in sorted_commits[half:]
            if any(
                f.get("detector_name") == worst_pattern
                for f in (c.get("patterns_detected") or {}).get("findings", [])
            )
        )
        pattern_trend = (
            "getting_worse" if newer_count > older_count
            else "improving" if newer_count < older_count
            else "stable"
        )

        growth_area = {
            "pattern": worst_pattern,
            "total_occurrences": count,
            "trend": pattern_trend,
        }

    return {
        "strengths": strengths,
        "low_risk_areas": low_risk_areas,
        "primary_growth_area": growth_area,
    }


def _determine_response_style(
    total_commits: int,
    avg_risk: float,
    pattern_counts: dict[str, int],
) -> str:
    """Decide how the AI agent should communicate with this developer."""
    if total_commits < _RESPONSE_STYLE_EDUCATIONAL_MAX_COMMITS:
        return "educational"

    max_violations = max(pattern_counts.values(), default=0)
    if max_violations >= _REPEATED_VIOLATION_THRESHOLD:
        return "direct"

    if (
        total_commits >= _RESPONSE_STYLE_PEER_MIN_COMMITS
        and avg_risk <= _RESPONSE_STYLE_PEER_MAX_AVG_RISK
    ):
        return "peer"

    return "educational"


# ---------------------------------------------------------------------------
# FUNCTION 1 — Full profile build
# ---------------------------------------------------------------------------

async def build_developer_profile(gitlab_username: str) -> dict[str, Any]:
    """Build (or rebuild) a complete behavioral profile from MongoDB commits.

    Reads every commit attributed to *gitlab_username*, computes all metrics,
    and upserts the result into ``developer_profiles``.

    Parameters
    ----------
    gitlab_username:
        The GitLab username (matched against ``author_email`` or
        ``pushed_by`` fields in the commits collection).

    Returns
    -------
    dict
        The full profile document that was stored.
    """
    logger.info("[developer_profiler] building profile for %s", gitlab_username)

    commits = await _fetch_commits_for_user(gitlab_username)
    if not commits:
        logger.warning("[developer_profiler] no commits found for %s", gitlab_username)
        profile = _empty_profile(gitlab_username)
        await _upsert_profile(gitlab_username, profile)
        return profile

    all_detector_names = [
        "hardcoded_secrets",
        "missing_error_handling",
        "large_functions",
        "todo_bombs",
        "copy_paste",
        "missing_validation",
        "race_conditions",
        "dead_imports",
    ]

    coding_patterns = _compute_coding_patterns(commits)
    quality_metrics = _compute_quality_metrics(commits)
    behavioral = _compute_behavioral_signals(commits)
    strengths_growth = _compute_strengths_and_growth(
        commits,
        coding_patterns["pattern_counts"],
        all_detector_names,
    )
    response_style = _determine_response_style(
        behavioral["total_commits"],
        quality_metrics["average_risk_score"],
        coding_patterns["pattern_counts"],
    )

    profile: dict[str, Any] = {
        "gitlab_username": gitlab_username,
        "coding_patterns": coding_patterns,
        "quality_metrics": quality_metrics,
        "behavioral_signals": behavioral,
        "strengths": strengths_growth["strengths"],
        "low_risk_areas": strengths_growth["low_risk_areas"],
        "primary_growth_area": strengths_growth["primary_growth_area"],
        "response_style": response_style,
        "updated_at": _utc_now(),
        "commit_count": len(commits),
    }

    await _upsert_profile(gitlab_username, profile)
    logger.info(
        "[developer_profiler] profile built for %s — commits=%d risk=%.2f trend=%s style=%s",
        gitlab_username,
        len(commits),
        quality_metrics["average_risk_score"],
        quality_metrics["quality_trend"],
        response_style,
    )
    return profile


# ---------------------------------------------------------------------------
# FUNCTION 2 — Incremental update
# ---------------------------------------------------------------------------

async def update_developer_profile(
    gitlab_username: str,
    new_commit_data: dict[str, Any],
) -> dict[str, Any]:
    """Incrementally update a developer profile with one new commit.

    Avoids a full MongoDB scan.  Fetches the existing profile, merges in the
    new commit's metrics, and re-upserts.  Falls back to a full rebuild if no
    profile exists yet.

    Parameters
    ----------
    gitlab_username:
        The developer's GitLab username.
    new_commit_data:
        A commit document as stored in the ``commits`` collection, including
        an optional ``patterns_detected`` sub-document.

    Returns
    -------
    dict
        The updated profile document.
    """
    logger.info("[developer_profiler] incremental update for %s", gitlab_username)

    collection = database.get_collection("developer_profiles")
    existing = await collection.find_one({"gitlab_username": gitlab_username})

    if existing is None:
        logger.info(
            "[developer_profiler] no existing profile for %s — running full build",
            gitlab_username,
        )
        return await build_developer_profile(gitlab_username)

    # --- Merge new commit into behavioral signals ---
    behavioral = existing.get("behavioral_signals", {})
    prev_total = behavioral.get("total_commits", 0)
    new_total = prev_total + 1

    added = int(new_commit_data.get("lines_added", 0) or 0)
    removed = int(new_commit_data.get("lines_removed", 0) or 0)
    new_size = added + removed

    prev_avg_size = behavioral.get("average_commit_size_lines", 0.0)
    new_avg_size = round(
        (prev_avg_size * prev_total + new_size) / new_total, 1
    )

    # Recalculate late-night / weekend percentages with running average
    dt = _ensure_utc(
        new_commit_data.get("authored_date") or new_commit_data.get("committed_date")
    )
    late_night_pct = behavioral.get("late_night_commit_percentage", 0.0)
    weekend_pct = behavioral.get("weekend_commit_percentage", 0.0)

    if dt:
        new_late = 100.0 if _is_late_night(dt.hour) else 0.0
        new_weekend = 100.0 if _is_weekend(dt) else 0.0
        late_night_pct = round(
            (late_night_pct * prev_total + new_late) / new_total, 1
        )
        weekend_pct = round(
            (weekend_pct * prev_total + new_weekend) / new_total, 1
        )

    large_pct = behavioral.get("large_commit_percentage", 0.0)
    small_pct = behavioral.get("small_focused_commit_percentage", 0.0)
    if new_size > 500:
        large_pct = round((large_pct * prev_total + 100) / new_total, 1)
    if new_size <= 50:
        small_pct = round((small_pct * prev_total + 100) / new_total, 1)

    behavioral.update({
        "total_commits": new_total,
        "average_commit_size_lines": new_avg_size,
        "late_night_commit_percentage": late_night_pct,
        "weekend_commit_percentage": weekend_pct,
        "large_commit_percentage": large_pct,
        "small_focused_commit_percentage": small_pct,
    })

    # --- Merge pattern counts ---
    coding_patterns = existing.get("coding_patterns", {})
    pattern_counts = coding_patterns.get("pattern_counts", {})

    pd = new_commit_data.get("patterns_detected") or {}
    new_risk = float(pd.get("risk_score", 0.0))
    for finding in pd.get("findings", []):
        name = finding.get("detector_name", "unknown")
        pattern_counts[name] = pattern_counts.get(name, 0) + 1

    coding_patterns["pattern_counts"] = pattern_counts
    # Rebuild top_violations ranking
    coding_patterns["top_violations"] = [
        {"pattern": n, "count": c}
        for n, c in sorted(pattern_counts.items(), key=lambda x: -x[1])[:10]
    ]

    # --- Update risk score running average ---
    quality_metrics = existing.get("quality_metrics", {})
    prev_avg_risk = quality_metrics.get("average_risk_score", 0.0)
    new_avg_risk = round(
        (prev_avg_risk * prev_total + new_risk) / new_total, 2
    )
    # Append to risk-over-time list (keep last 200)
    rot = quality_metrics.get("risk_scores_over_time", [])
    rot.append(new_risk)
    rot = rot[-200:]
    quality_metrics["average_risk_score"] = new_avg_risk
    quality_metrics["risk_scores_over_time"] = rot
    quality_metrics["quality_trend"] = _quality_trend(rot[-_TREND_WINDOW:])

    response_style = _determine_response_style(new_total, new_avg_risk, pattern_counts)

    updated_profile: dict[str, Any] = {
        **existing,
        "behavioral_signals": behavioral,
        "coding_patterns": coding_patterns,
        "quality_metrics": quality_metrics,
        "response_style": response_style,
        "commit_count": new_total,
        "updated_at": _utc_now(),
    }
    updated_profile.pop("_id", None)

    await _upsert_profile(gitlab_username, updated_profile)
    logger.info(
        "[developer_profiler] incremental update done for %s — total_commits=%d",
        gitlab_username,
        new_total,
    )
    return updated_profile


# ---------------------------------------------------------------------------
# FUNCTION 3 — Agent context object
# ---------------------------------------------------------------------------

async def get_developer_context(
    gitlab_username: str,
    current_time: datetime | None = None,
) -> dict[str, Any]:
    """Return a lightweight context dict consumed by the AI comment agent.

    Parameters
    ----------
    gitlab_username:
        The developer's GitLab username.
    current_time:
        The time of the event being processed.  Defaults to UTC now.

    Returns
    -------
    dict with keys:
        repeated_patterns, is_late_night, commit_count_today,
        quality_trend, last_similar_mistake, response_style
    """
    if current_time is None:
        current_time = _utc_now()
    ct = _ensure_utc(current_time)

    collection = database.get_collection("developer_profiles")
    profile = await collection.find_one({"gitlab_username": gitlab_username})

    if profile is None:
        logger.warning(
            "[developer_profiler] no profile found for %s — returning default context",
            gitlab_username,
        )
        return _default_context(gitlab_username, ct)

    # Repeated patterns (appeared more than once)
    pattern_counts: dict[str, int] = (
        profile.get("coding_patterns", {}).get("pattern_counts", {})
    )
    repeated_patterns = [
        {"pattern": name, "count": count}
        for name, count in sorted(pattern_counts.items(), key=lambda x: -x[1])
        if count > 1
    ]

    # Is this commit late-night?
    is_late_night = _is_late_night(ct.hour)  # type: ignore[arg-type]

    # Commits today (fetch from commits collection for this user)
    commits_today = await _count_commits_today(gitlab_username, ct)  # type: ignore[arg-type]

    quality_trend = (
        profile.get("quality_metrics", {}).get("quality_trend", "stable")
    )

    # Last similar mistake — most common growth-area pattern, most recent commit with it
    growth_area = profile.get("primary_growth_area", {})
    last_similar: dict[str, Any] = {}
    if growth_area.get("pattern"):
        last_similar = await _find_last_mistake(
            gitlab_username, growth_area["pattern"]
        )

    response_style = profile.get("response_style", "educational")

    return {
        "gitlab_username": gitlab_username,
        "repeated_patterns": repeated_patterns,
        "is_late_night": is_late_night,
        "commit_count_today": commits_today,
        "quality_trend": quality_trend,
        "last_similar_mistake": last_similar,
        "response_style": response_style,
        "profile_updated_at": profile.get("updated_at"),
    }


# ---------------------------------------------------------------------------
# Private MongoDB helpers
# ---------------------------------------------------------------------------

async def _fetch_commits_for_user(gitlab_username: str) -> list[dict]:
    """Fetch all commits in MongoDB for this developer."""
    collection = database.get_collection("commits")
    cursor = collection.find(
        {
            "$or": [
                {"author_email": {"$regex": gitlab_username, "$options": "i"}},
                {"author_name": {"$regex": gitlab_username, "$options": "i"}},
                {"pushed_by": gitlab_username},
                {"user_username": gitlab_username},
            ]
        }
    )
    commits = await cursor.to_list(length=None)
    logger.debug(
        "[developer_profiler] found %d commits for %s", len(commits), gitlab_username
    )
    return commits


async def _count_commits_today(gitlab_username: str, now: datetime) -> int:
    """Count commits by this user on the same calendar day as *now*."""
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    collection = database.get_collection("commits")
    count = await collection.count_documents(
        {
            "$or": [
                {"author_email": {"$regex": gitlab_username, "$options": "i"}},
                {"pushed_by": gitlab_username},
            ],
            "$or": [  # noqa: F601 — second $or overrides first; intentional for date filter
                {"authored_date": {"$gte": start_of_day}},
                {"committed_date": {"$gte": start_of_day}},
                {"timestamp": {"$gte": start_of_day}},
            ],
        }
    )
    return int(count)


async def _find_last_mistake(
    gitlab_username: str, pattern_name: str
) -> dict[str, Any]:
    """Find the most recent commit by this user that triggered *pattern_name*."""
    collection = database.get_collection("commits")
    commit = await collection.find_one(
        {
            "$or": [
                {"author_email": {"$regex": gitlab_username, "$options": "i"}},
                {"pushed_by": gitlab_username},
            ],
            "patterns_detected.findings": {
                "$elemMatch": {"detector_name": pattern_name}
            },
        },
        sort=[("authored_date", -1)],
    )
    if not commit:
        return {}

    # Extract the specific finding description
    description = ""
    for f in (commit.get("patterns_detected") or {}).get("findings", []):
        if f.get("detector_name") == pattern_name:
            description = f.get("description", "")
            break

    return {
        "sha": commit.get("commit_hash", ""),
        "date": (commit.get("authored_date") or commit.get("committed_date")),
        "description": description,
        "pattern": pattern_name,
    }


async def _upsert_profile(gitlab_username: str, profile: dict[str, Any]) -> None:
    """Upsert the profile document into ``developer_profiles``."""
    collection = database.get_collection("developer_profiles")
    profile_to_store = {k: v for k, v in profile.items() if k != "_id"}
    try:
        await collection.update_one(
            {"gitlab_username": gitlab_username},
            {"$set": profile_to_store},
            upsert=True,
        )
    except Exception:
        logger.exception(
            "[developer_profiler] failed to upsert profile for %s", gitlab_username
        )


def _empty_profile(gitlab_username: str) -> dict[str, Any]:
    return {
        "gitlab_username": gitlab_username,
        "coding_patterns": {
            "top_violations": [],
            "pattern_counts": {},
            "recurrence_rates": {},
            "file_types_touched": {},
        },
        "quality_metrics": {
            "average_risk_score": 0.0,
            "quality_trend": "stable",
            "risk_scores_over_time": [],
            "best_hour_of_day": None,
            "worst_hour_of_day": None,
            "risk_by_hour": {},
            "risk_by_day_of_week": {},
        },
        "behavioral_signals": {
            "total_commits": 0,
            "average_commit_size_lines": 0.0,
            "large_commit_percentage": 0.0,
            "small_focused_commit_percentage": 0.0,
            "late_night_commit_percentage": 0.0,
            "weekend_commit_percentage": 0.0,
        },
        "strengths": [],
        "low_risk_areas": [],
        "primary_growth_area": {},
        "response_style": "educational",
        "updated_at": _utc_now(),
        "commit_count": 0,
    }


def _default_context(
    gitlab_username: str, current_time: datetime
) -> dict[str, Any]:
    return {
        "gitlab_username": gitlab_username,
        "repeated_patterns": [],
        "is_late_night": _is_late_night(current_time.hour),
        "commit_count_today": 0,
        "quality_trend": "stable",
        "last_similar_mistake": {},
        "response_style": "educational",
        "profile_updated_at": None,
    }
