"""
Comprehensive unit tests for backend.modules.pattern_detector.

All detectors are tested against real input strings — no mocking of the
detectors themselves.  The ``TestAnalyzeCommit`` suite mocks only the
MongoDB layer (``database.get_collection``).

Run with:
    pytest tests/test_pattern_detector.py -v
"""

from __future__ import annotations

import textwrap
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.modules.pattern_detector import (
    Finding,
    analyze_commit,
    detect_copy_paste,
    detect_dead_imports,
    detect_hardcoded_secrets,
    detect_large_functions,
    detect_missing_error_handling,
    detect_missing_validation,
    detect_race_conditions,
    detect_todo_bombs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _diff(*lines: str) -> str:
    """Build a minimal unified-diff string from the provided lines.

    Each element of *lines* should start with ``+``, ``-``, or a space.
    A fake ``+++`` header is prepended so the detector's header-stripping
    logic is exercised.
    """
    header = "+++ b/fake_file.py\n--- a/fake_file.py\n"
    return header + "\n".join(lines)


def _added(*lines: str) -> str:
    """Wrap lines as added (``+`` prefix) in a minimal diff."""
    return _diff(*[f"+{ln}" for ln in lines])


# ---------------------------------------------------------------------------
# DETECTOR 1 — Hardcoded Secrets
# ---------------------------------------------------------------------------


class TestDetectHardcodedSecrets:
    """Tests for detect_hardcoded_secrets."""

    def test_detects_api_key(self) -> None:
        diff = _added('api_key = "supersecretkey123"')
        findings = detect_hardcoded_secrets(diff)
        assert len(findings) >= 1
        assert any(f.severity == "CRITICAL" for f in findings)
        assert any("api_key" in f.description.lower() or "secret" in f.description.lower() for f in findings)

    def test_detects_api_key_single_quotes(self) -> None:
        diff = _added("api_key = 'anothersecret99!'")
        findings = detect_hardcoded_secrets(diff)
        assert len(findings) >= 1
        assert findings[0].severity == "CRITICAL"

    def test_detects_aws_key(self) -> None:
        diff = _added("aws_access_key = 'AKIAIOSFODNN7EXAMPLE'")
        findings = detect_hardcoded_secrets(diff)
        assert len(findings) >= 1
        assert any(f.severity == "CRITICAL" for f in findings)

    def test_detects_aws_key_inline(self) -> None:
        diff = _added("key = AKIAIOSFODNN7EXAMPLE")
        findings = detect_hardcoded_secrets(diff)
        assert len(findings) >= 1

    def test_detects_password(self) -> None:
        diff = _added('password = "hunter2"')
        findings = detect_hardcoded_secrets(diff)
        assert len(findings) >= 1
        assert findings[0].severity == "CRITICAL"

    def test_detects_password_case_insensitive(self) -> None:
        diff = _added('PASSWORD = "S3cr3tP@ss"')
        findings = detect_hardcoded_secrets(diff)
        assert len(findings) >= 1

    def test_detects_secret_assignment(self) -> None:
        diff = _added('secret = "my_very_long_secret_value"')
        findings = detect_hardcoded_secrets(diff)
        assert len(findings) >= 1
        assert findings[0].severity == "CRITICAL"

    def test_detects_connection_string(self) -> None:
        diff = _added('db_url = "postgres://admin:password@localhost:5432/mydb"')
        findings = detect_hardcoded_secrets(diff)
        assert len(findings) >= 1
        assert findings[0].severity == "CRITICAL"

    def test_detects_mongodb_connection_string(self) -> None:
        diff = _added('uri = "mongodb://user:pass@cluster0.mongodb.net/mydb"')
        findings = detect_hardcoded_secrets(diff)
        assert len(findings) >= 1

    def test_detects_token(self) -> None:
        diff = _added('token = "abcdefghijklmnopqrstu1234"')
        findings = detect_hardcoded_secrets(diff)
        assert len(findings) >= 1
        assert findings[0].severity == "CRITICAL"

    def test_ignores_clean_code(self) -> None:
        diff = _added(
            "x = 1",
            "name = 'Alice'",
            "def greet(name): return f'Hello {name}'",
        )
        findings = detect_hardcoded_secrets(diff)
        assert findings == []

    def test_ignores_short_password(self) -> None:
        # password value is only 3 chars — below the 4-char threshold
        diff = _added('password = "abc"')
        findings = detect_hardcoded_secrets(diff)
        assert findings == []

    def test_only_flags_added_lines(self) -> None:
        """Lines with a ``-`` prefix (removals) must NOT be flagged."""
        diff = (
            "+++ b/fake.py\n"
            "--- a/fake.py\n"
            '-api_key = "oldsecretkey123"\n'
            ' # unchanged context line\n'
        )
        findings = detect_hardcoded_secrets(diff)
        assert findings == []

    def test_finding_contains_snippet(self) -> None:
        diff = _added('api_key = "supersecretkey123"')
        findings = detect_hardcoded_secrets(diff)
        assert len(findings) >= 1
        assert "api_key" in findings[0].snippet

    def test_finding_has_suggested_fix(self) -> None:
        diff = _added('password = "securepassword"')
        findings = detect_hardcoded_secrets(diff)
        assert len(findings) >= 1
        assert len(findings[0].suggested_fix) > 0

    def test_detector_name(self) -> None:
        diff = _added('api_key = "supersecretkey123"')
        findings = detect_hardcoded_secrets(diff)
        assert all(f.detector_name == "hardcoded_secrets" for f in findings)


# ---------------------------------------------------------------------------
# DETECTOR 2 — Missing Error Handling
# ---------------------------------------------------------------------------


class TestDetectMissingErrorHandling:
    """Tests for detect_missing_error_handling."""

    def test_detects_requests_get_without_try(self) -> None:
        diff = _added(
            "def fetch_data():",
            "    resp = requests.get('https://example.com/api')",
            "    return resp.json()",
        )
        findings = detect_missing_error_handling(diff, language="python")
        assert len(findings) >= 1
        assert findings[0].severity == "HIGH"
        assert "requests" in findings[0].description

    def test_detects_requests_post_without_try(self) -> None:
        diff = _added(
            "def send_data(payload):",
            "    resp = requests.post('https://example.com', json=payload)",
            "    return resp.status_code",
        )
        findings = detect_missing_error_handling(diff, language="python")
        assert len(findings) >= 1
        assert findings[0].severity == "HIGH"

    def test_ignores_requests_inside_try(self) -> None:
        diff = _added(
            "def fetch_data():",
            "    try:",
            "        resp = requests.get('https://example.com/api')",
            "        return resp.json()",
            "    except requests.exceptions.RequestException:",
            "        return None",
        )
        findings = detect_missing_error_handling(diff, language="python")
        # No finding because try/except is present
        assert all(
            "requests" not in f.description for f in findings
        ), f"Unexpected findings: {findings}"

    def test_detects_open_without_try(self) -> None:
        diff = _added(
            "def read_config():",
            "    f = open('/etc/config.txt', 'r')",
            "    return f.read()",
        )
        findings = detect_missing_error_handling(diff, language="python")
        assert len(findings) >= 1
        assert any("open" in f.description for f in findings)
        assert all(f.severity == "HIGH" for f in findings)

    def test_ignores_open_inside_try(self) -> None:
        diff = _added(
            "def read_config():",
            "    try:",
            "        f = open('/etc/config.txt', 'r')",
            "        return f.read()",
            "    except OSError:",
            "        return ''",
        )
        findings = detect_missing_error_handling(diff, language="python")
        assert all("open" not in f.description for f in findings)

    def test_detects_await_without_try(self) -> None:
        diff = _added(
            "async def call_service():",
            "    result = await some_service.do_work()",
            "    return result",
        )
        findings = detect_missing_error_handling(diff, language="python")
        assert len(findings) >= 1
        assert any("await" in f.description for f in findings)

    def test_detector_name_is_correct(self) -> None:
        diff = _added(
            "def bad():",
            "    r = requests.get('http://example.com')",
        )
        findings = detect_missing_error_handling(diff, language="python")
        assert all(f.detector_name == "missing_error_handling" for f in findings)

    def test_ignores_non_network_code(self) -> None:
        diff = _added(
            "def add(a, b):",
            "    return a + b",
        )
        findings = detect_missing_error_handling(diff, language="python")
        assert findings == []

    def test_javascript_fetch_without_catch(self) -> None:
        diff = _added(
            "function loadData() {",
            "    fetch('https://api.example.com/data')",
            "        .then(r => r.json())",
            "        .then(data => console.log(data));",
            "}",
        )
        findings = detect_missing_error_handling(diff, language="javascript")
        assert len(findings) >= 1
        assert findings[0].severity == "HIGH"

    def test_javascript_fetch_with_catch_ignored(self) -> None:
        diff = _added(
            "fetch('https://api.example.com/data')",
            "    .then(r => r.json())",
            "    .catch(err => console.error(err));",
        )
        findings = detect_missing_error_handling(diff, language="javascript")
        assert findings == []


# ---------------------------------------------------------------------------
# DETECTOR 3 — Large Functions
# ---------------------------------------------------------------------------


def _make_large_function_diff(n_body_lines: int, func_name: str = "big_func") -> str:
    """Return a diff containing a Python function with *n_body_lines* body lines."""
    lines = [f"def {func_name}(x):"]
    for i in range(n_body_lines):
        lines.append(f"    x = x + {i}  # line {i}")
    lines.append("    return x")
    return _added(*lines)


class TestDetectLargeFunctions:
    """Tests for detect_large_functions."""

    def test_detects_function_over_50_lines(self) -> None:
        diff = _make_large_function_diff(55)
        findings = detect_large_functions(diff)
        assert len(findings) >= 1
        assert findings[0].severity == "WARNING"
        assert "big_func" in findings[0].description
        assert findings[0].detector_name == "large_functions"

    def test_detects_function_over_100_lines(self) -> None:
        diff = _make_large_function_diff(105)
        findings = detect_large_functions(diff)
        assert len(findings) >= 1
        assert findings[0].severity == "HIGH"
        assert "big_func" in findings[0].description

    def test_detects_function_over_200_lines(self) -> None:
        diff = _make_large_function_diff(205)
        findings = detect_large_functions(diff)
        assert len(findings) >= 1
        assert findings[0].severity == "CRITICAL"

    def test_ignores_small_functions(self) -> None:
        diff = _make_large_function_diff(10)
        findings = detect_large_functions(diff)
        assert findings == []

    def test_ignores_function_just_below_threshold(self) -> None:
        # _make_large_function_diff(n) adds a `def` line + n body lines + a
        # `return` line, so n=47 gives a 49-line function — just under the
        # 50-line WARNING threshold.
        diff = _make_large_function_diff(47)
        findings = detect_large_functions(diff)
        assert findings == []

    def test_function_name_in_description(self) -> None:
        diff = _make_large_function_diff(60, func_name="process_orders")
        findings = detect_large_functions(diff)
        assert len(findings) >= 1
        assert "process_orders" in findings[0].description

    def test_line_count_in_description(self) -> None:
        diff = _make_large_function_diff(60)
        findings = detect_large_functions(diff)
        assert len(findings) >= 1
        # Description should mention the line count (as a number string)
        assert any(char.isdigit() for char in findings[0].description)

    def test_detects_async_function(self) -> None:
        lines = ["async def big_async(x):"]
        for i in range(55):
            lines.append(f"    x = x + {i}")
        lines.append("    return x")
        diff = _added(*lines)
        findings = detect_large_functions(diff)
        assert len(findings) >= 1
        assert "big_async" in findings[0].description

    def test_empty_diff_returns_empty(self) -> None:
        findings = detect_large_functions("")
        assert findings == []


# ---------------------------------------------------------------------------
# DETECTOR 4 — TODO Bombs
# ---------------------------------------------------------------------------


class TestDetectTodoBombs:
    """Tests for detect_todo_bombs."""

    def test_detects_todo(self) -> None:
        diff = _added("    # TODO: refactor this later")
        findings = detect_todo_bombs(diff)
        assert len(findings) == 1
        assert "TODO" in findings[0].description
        assert findings[0].detector_name == "todo_bombs"

    def test_detects_fixme(self) -> None:
        diff = _added("    # FIXME: this crashes on large input")
        findings = detect_todo_bombs(diff)
        assert len(findings) == 1
        assert "FIXME" in findings[0].description

    def test_detects_hack(self) -> None:
        diff = _added("    # HACK: workaround for broken API")
        findings = detect_todo_bombs(diff)
        assert len(findings) == 1

    def test_detects_xxx(self) -> None:
        diff = _added("    # XXX: this is terrible but it works")
        findings = detect_todo_bombs(diff)
        assert len(findings) == 1

    def test_detects_temp(self) -> None:
        diff = _added("    # TEMP: remove before release")
        findings = detect_todo_bombs(diff)
        assert len(findings) == 1

    def test_severity_low_for_fresh_commit(self) -> None:
        fresh_ts = datetime.now(timezone.utc) - timedelta(days=5)
        diff = _added("    # TODO: clean up")
        findings = detect_todo_bombs(diff, commit_timestamp=fresh_ts)
        assert findings[0].severity == "LOW"

    def test_severity_high_for_stale_commit(self) -> None:
        stale_ts = datetime.now(timezone.utc) - timedelta(days=45)
        diff = _added("    # TODO: clean up")
        findings = detect_todo_bombs(diff, commit_timestamp=stale_ts)
        assert findings[0].severity == "HIGH"

    def test_ignores_removed_lines(self) -> None:
        """Lines with ``-`` prefix (removals) must NOT be flagged."""
        diff = (
            "+++ b/fake.py\n"
            "--- a/fake.py\n"
            "-    # TODO: old comment being removed\n"
        )
        findings = detect_todo_bombs(diff)
        assert findings == []

    def test_ignores_clean_code(self) -> None:
        diff = _added(
            "def process():",
            "    return 42",
        )
        findings = detect_todo_bombs(diff)
        assert findings == []

    def test_case_insensitive_detection(self) -> None:
        diff = _added("    # todo: lowercase marker")
        findings = detect_todo_bombs(diff)
        assert len(findings) == 1

    def test_multiple_todos_return_multiple_findings(self) -> None:
        diff = _added(
            "    # TODO: first issue",
            "    x = 1",
            "    # FIXME: second issue",
        )
        findings = detect_todo_bombs(diff)
        assert len(findings) == 2

    def test_snippet_contains_comment(self) -> None:
        diff = _added("    # TODO: fix the memory leak")
        findings = detect_todo_bombs(diff)
        assert "TODO" in findings[0].snippet


# ---------------------------------------------------------------------------
# DETECTOR 5 — Copy-Paste Code
# ---------------------------------------------------------------------------


def _make_block(n: int = 12, prefix: str = "line") -> list[str]:
    """Return a list of *n* unique-ish code lines."""
    return [f"    result_{i} = process_{prefix}({i})" for i in range(n)]


class TestDetectCopyPaste:
    """Tests for detect_copy_paste."""

    def test_detects_similar_block(self) -> None:
        block = _make_block(12, "alpha")
        existing_sample = "\n".join(block)
        # Slight variation: change 1 line
        block_copy = block[:]
        block_copy[5] = "    result_5 = process_beta(5)"
        diff = _added(*block_copy)
        findings = detect_copy_paste(diff, existing_samples=[existing_sample])
        assert len(findings) >= 1
        assert findings[0].severity == "MEDIUM"
        assert findings[0].detector_name == "copy_paste"

    def test_detects_exact_duplicate(self) -> None:
        block = _make_block(15)
        existing_sample = "\n".join(block)
        diff = _added(*block)
        findings = detect_copy_paste(diff, existing_samples=[existing_sample])
        assert len(findings) >= 1

    def test_ignores_dissimilar_code(self) -> None:
        block_a = _make_block(12, "alpha")
        block_b = [f"    totally_different_thing_{i}()" for i in range(12)]
        existing_sample = "\n".join(block_a)
        diff = _added(*block_b)
        findings = detect_copy_paste(diff, existing_samples=[existing_sample])
        assert findings == []

    def test_ignores_short_blocks(self) -> None:
        """Blocks shorter than 10 lines should not be checked."""
        block = _make_block(5)
        existing_sample = "\n".join(block)
        diff = _added(*block)
        findings = detect_copy_paste(diff, existing_samples=[existing_sample])
        assert findings == []

    def test_empty_samples_returns_empty(self) -> None:
        diff = _added(*_make_block(15))
        findings = detect_copy_paste(diff, existing_samples=[])
        assert findings == []

    def test_finding_has_snippet(self) -> None:
        block = _make_block(12)
        existing_sample = "\n".join(block)
        diff = _added(*block)
        findings = detect_copy_paste(diff, existing_samples=[existing_sample])
        assert len(findings) >= 1
        assert findings[0].snippet != ""


# ---------------------------------------------------------------------------
# DETECTOR 6 — Missing Input Validation
# ---------------------------------------------------------------------------


class TestDetectMissingValidation:
    """Tests for detect_missing_validation."""

    def test_detects_raw_request_json(self) -> None:
        diff = _added(
            "@router.post('/items')",
            "async def create_item(request: Request):",
            "    data = await request.json()",
            "    await collection.insert_one(data)",
        )
        findings = detect_missing_validation(diff)
        assert len(findings) >= 1
        assert any(f.severity == "HIGH" for f in findings)
        assert any(f.detector_name == "missing_validation" for f in findings)

    def test_detects_request_body(self) -> None:
        diff = _added(
            "async def handler(request: Request):",
            "    body = await request.body()",
            "    return body",
        )
        findings = detect_missing_validation(diff)
        assert len(findings) >= 1
        assert any("request" in f.description.lower() for f in findings)

    def test_detects_route_with_raw_request_param(self) -> None:
        diff = _added(
            "@app.post('/users')",
            "async def create_user(request: Request):",
            "    return {'status': 'ok'}",
        )
        findings = detect_missing_validation(diff)
        assert len(findings) >= 1
        assert any("Request" in f.description for f in findings)

    def test_detects_collection_write_after_raw_request(self) -> None:
        diff = _added(
            "async def save(request: Request):",
            "    data = await request.json()",
            "    await collection.insert_one(data)",
        )
        findings = detect_missing_validation(diff)
        assert len(findings) >= 1
        # At minimum the raw request.json() should be flagged
        assert any("request" in f.description.lower() for f in findings)

    def test_ignores_pydantic_model(self) -> None:
        diff = _added(
            "@router.post('/items')",
            "async def create_item(item: ItemModel):",
            "    await collection.insert_one(item.model_dump())",
            "    return item",
        )
        findings = detect_missing_validation(diff)
        # No raw request.json() or request: Request, so no validation findings
        assert all("request" not in f.description.lower() for f in findings)

    def test_ignores_get_endpoint_without_body(self) -> None:
        diff = _added(
            "@router.get('/items')",
            "async def list_items():",
            "    return []",
        )
        findings = detect_missing_validation(diff)
        assert findings == []

    def test_severity_is_high(self) -> None:
        diff = _added(
            "    data = await request.json()",
        )
        findings = detect_missing_validation(diff)
        assert all(f.severity == "HIGH" for f in findings)


# ---------------------------------------------------------------------------
# DETECTOR 7 — Race Conditions
# ---------------------------------------------------------------------------


class TestDetectRaceConditions:
    """Tests for detect_race_conditions."""

    def test_detects_check_then_act(self) -> None:
        diff = _added(
            "async def update_counter():",
            "    if counter > 0:",
            "        counter = counter - 1",
            "    return counter",
        )
        findings = detect_race_conditions(diff, language="python")
        assert len(findings) >= 1
        assert findings[0].severity == "CRITICAL"
        assert findings[0].detector_name == "race_conditions"
        assert "counter" in findings[0].description

    def test_detects_shared_state_in_async_with_sleep(self) -> None:
        diff = _added(
            "global shared_counter",
            "",
            "async def increment():",
            "    global shared_counter",
            "    await asyncio.sleep(0.1)",
            "    shared_counter = shared_counter + 1",
        )
        findings = detect_race_conditions(diff, language="python")
        assert len(findings) >= 1
        assert any("sleep" in f.description.lower() or "shared" in f.description.lower()
                   for f in findings)
        assert all(f.severity == "CRITICAL" for f in findings)

    def test_detects_multiple_await_writes_to_same_var(self) -> None:
        diff = _added(
            "async def fetch_and_store():",
            "    result = await get_data_a()",
            "    result = await get_data_b()",
            "    return result",
        )
        findings = detect_race_conditions(diff, language="python")
        assert len(findings) >= 1
        assert any("result" in f.description for f in findings)

    def test_ignores_safe_async_function(self) -> None:
        diff = _added(
            "async def safe_fetch():",
            "    data = await some_api_call()",
            "    return data",
        )
        findings = detect_race_conditions(diff, language="python")
        # No check-then-act, no shared state, single await write
        assert findings == []

    def test_ignores_non_python_language(self) -> None:
        diff = _added(
            "async function update() {",
            "    if (counter > 0) { counter--; }",
            "}",
        )
        # JavaScript language — detector returns empty for non-python
        findings = detect_race_conditions(diff, language="javascript")
        assert findings == []

    def test_finding_has_suggested_fix(self) -> None:
        diff = _added(
            "async def update_counter():",
            "    if counter > 0:",
            "        counter = counter - 1",
        )
        findings = detect_race_conditions(diff, language="python")
        assert len(findings) >= 1
        assert "lock" in findings[0].suggested_fix.lower()


# ---------------------------------------------------------------------------
# DETECTOR 8 — Dead Imports
# ---------------------------------------------------------------------------


class TestDetectDeadImports:
    """Tests for detect_dead_imports."""

    def test_detects_unused_import(self) -> None:
        diff = _added(
            "import os",
            "import json",
            "",
            "def greet():",
            "    return json.dumps({'hello': 'world'})",
        )
        findings = detect_dead_imports(diff)
        assert len(findings) >= 1
        assert any("os" in f.description for f in findings)
        assert all(f.severity == "LOW" for f in findings)
        assert all(f.detector_name == "dead_imports" for f in findings)

    def test_detects_unused_from_import(self) -> None:
        diff = _added(
            "from datetime import datetime, timedelta",
            "",
            "def now():",
            "    return datetime.now()",
        )
        findings = detect_dead_imports(diff)
        assert len(findings) >= 1
        assert any("timedelta" in f.description for f in findings)

    def test_ignores_used_import(self) -> None:
        diff = _added(
            "import os",
            "",
            "def get_path():",
            "    return os.getcwd()",
        )
        findings = detect_dead_imports(diff)
        assert all("os" not in f.description for f in findings)

    def test_ignores_used_from_import(self) -> None:
        diff = _added(
            "from pathlib import Path",
            "",
            "def resolve(p):",
            "    return Path(p).resolve()",
        )
        findings = detect_dead_imports(diff)
        assert all("Path" not in f.description for f in findings)

    def test_detects_aliased_import_unused(self) -> None:
        diff = _added(
            "import numpy as np",
            "",
            "def compute():",
            "    return 42",
        )
        findings = detect_dead_imports(diff)
        assert len(findings) >= 1
        assert any("np" in f.description for f in findings)

    def test_ignores_aliased_import_used(self) -> None:
        diff = _added(
            "import numpy as np",
            "",
            "def compute(x):",
            "    return np.sqrt(x)",
        )
        findings = detect_dead_imports(diff)
        assert all("np" not in f.description for f in findings)

    def test_empty_diff_returns_empty(self) -> None:
        findings = detect_dead_imports("")
        assert findings == []

    def test_finding_has_low_severity(self) -> None:
        diff = _added("import sys", "def run(): return 0")
        findings = detect_dead_imports(diff)
        assert all(f.severity == "LOW" for f in findings)

    def test_no_imports_returns_empty(self) -> None:
        diff = _added(
            "def add(a, b):",
            "    return a + b",
        )
        findings = detect_dead_imports(diff)
        assert findings == []


# ---------------------------------------------------------------------------
# Master function — TestAnalyzeCommit
# ---------------------------------------------------------------------------


def _make_mock_collection() -> MagicMock:
    """Return a mock MongoDB collection with async update_one."""
    collection = MagicMock()
    collection.update_one = AsyncMock(return_value=MagicMock(upserted_id="fake_id"))
    return collection


class TestAnalyzeCommit:
    """Integration tests for analyze_commit."""

    @pytest.mark.asyncio
    async def test_returns_risk_score(self) -> None:
        """analyze_commit should return a dict with the expected keys."""
        diff = _added(
            'api_key = "supersecretkey123"',
            "import os",
            "def simple(): return 1",
        )
        mock_collection = _make_mock_collection()

        with patch("backend.modules.pattern_detector.database") as mock_db:
            mock_db.get_collection.return_value = mock_collection
            result = await analyze_commit(
                commit_sha="abc123def456",
                diff_content=diff,
                author="dev@example.com",
                timestamp=datetime.now(timezone.utc),
                language="python",
            )

        assert "commit_sha" in result
        assert result["commit_sha"] == "abc123def456"
        assert "risk_score" in result
        assert isinstance(result["risk_score"], float)
        assert 0.0 <= result["risk_score"] <= 10.0
        assert "total_findings" in result
        assert "findings" in result
        assert isinstance(result["findings"], list)
        assert "analyzed_at" in result

    @pytest.mark.asyncio
    async def test_risk_score_capped_at_10(self) -> None:
        """Even with many critical findings the risk_score must not exceed 10."""
        # Generate a diff with many hardcoded secrets (many CRITICAL findings)
        lines = [f'api_key_{i} = "supersecretkey{i:03d}xyz"' for i in range(20)]
        diff = _added(*lines)
        mock_collection = _make_mock_collection()

        with patch("backend.modules.pattern_detector.database") as mock_db:
            mock_db.get_collection.return_value = mock_collection
            result = await analyze_commit(
                commit_sha="zzz999",
                diff_content=diff,
                author="bad_dev@example.com",
                timestamp=datetime.now(timezone.utc),
            )

        assert result["risk_score"] <= 10.0

    @pytest.mark.asyncio
    async def test_clean_diff_has_zero_score(self) -> None:
        """A clean, simple diff should produce 0 findings and a 0 risk score."""
        diff = _added(
            "def add(a: int, b: int) -> int:",
            "    return a + b",
        )
        mock_collection = _make_mock_collection()

        with patch("backend.modules.pattern_detector.database") as mock_db:
            mock_db.get_collection.return_value = mock_collection
            result = await analyze_commit(
                commit_sha="clean001",
                diff_content=diff,
                author="clean_dev@example.com",
                timestamp=datetime.now(timezone.utc),
            )

        assert result["risk_score"] == 0.0
        assert result["total_findings"] == 0

    @pytest.mark.asyncio
    async def test_findings_list_contains_finding_dicts(self) -> None:
        """Each entry in findings should be a dict with the Finding fields."""
        diff = _added('password = "hunter2"')
        mock_collection = _make_mock_collection()

        with patch("backend.modules.pattern_detector.database") as mock_db:
            mock_db.get_collection.return_value = mock_collection
            result = await analyze_commit(
                commit_sha="pwd001",
                diff_content=diff,
                author="dev@example.com",
                timestamp=datetime.now(timezone.utc),
            )

        assert len(result["findings"]) >= 1
        finding = result["findings"][0]
        assert "detector_name" in finding
        assert "severity" in finding
        assert "description" in finding
        assert "suggested_fix" in finding

    @pytest.mark.asyncio
    async def test_database_upsert_called(self) -> None:
        """analyze_commit must call collection.update_one with upsert=True."""
        diff = _added("x = 1")
        mock_collection = _make_mock_collection()

        with patch("backend.modules.pattern_detector.database") as mock_db:
            mock_db.get_collection.return_value = mock_collection
            await analyze_commit(
                commit_sha="db_test_sha",
                diff_content=diff,
                author="dev@example.com",
                timestamp=datetime.now(timezone.utc),
            )

        mock_collection.update_one.assert_called_once()
        call_kwargs = mock_collection.update_one.call_args
        # Third positional argument or keyword: upsert=True
        assert call_kwargs.kwargs.get("upsert") is True or (
            len(call_kwargs.args) >= 3 and call_kwargs.args[2] is True
        ) or call_kwargs.kwargs.get("upsert", False)

    @pytest.mark.asyncio
    async def test_database_error_does_not_crash(self) -> None:
        """If the MongoDB upsert raises, analyze_commit should still return a result."""
        diff = _added("x = 1")
        mock_collection = MagicMock()
        mock_collection.update_one = AsyncMock(side_effect=RuntimeError("DB down"))

        with patch("backend.modules.pattern_detector.database") as mock_db:
            mock_db.get_collection.return_value = mock_collection
            result = await analyze_commit(
                commit_sha="db_err_sha",
                diff_content=diff,
                author="dev@example.com",
                timestamp=datetime.now(timezone.utc),
            )

        assert "risk_score" in result
        assert result["commit_sha"] == "db_err_sha"

    @pytest.mark.asyncio
    async def test_total_findings_matches_findings_list(self) -> None:
        diff = _added(
            'api_key = "supersecretkey123"',
            "    # TODO: remove this",
            "import os",
        )
        mock_collection = _make_mock_collection()

        with patch("backend.modules.pattern_detector.database") as mock_db:
            mock_db.get_collection.return_value = mock_collection
            result = await analyze_commit(
                commit_sha="count_check",
                diff_content=diff,
                author="dev@example.com",
                timestamp=datetime.now(timezone.utc),
            )

        assert result["total_findings"] == len(result["findings"])

    @pytest.mark.asyncio
    async def test_analyzed_at_is_datetime(self) -> None:
        diff = _added("x = 1")
        mock_collection = _make_mock_collection()

        with patch("backend.modules.pattern_detector.database") as mock_db:
            mock_db.get_collection.return_value = mock_collection
            result = await analyze_commit(
                commit_sha="ts_check",
                diff_content=diff,
                author="dev@example.com",
                timestamp=datetime.now(timezone.utc),
            )

        assert isinstance(result["analyzed_at"], datetime)

    @pytest.mark.asyncio
    async def test_risk_score_reflects_severity_weights(self) -> None:
        """A single CRITICAL finding should produce risk_score >= 3.0."""
        diff = _added('api_key = "supersecretkey_very_long"')
        mock_collection = _make_mock_collection()

        with patch("backend.modules.pattern_detector.database") as mock_db:
            mock_db.get_collection.return_value = mock_collection
            result = await analyze_commit(
                commit_sha="weight_check",
                diff_content=diff,
                author="dev@example.com",
                timestamp=datetime.now(timezone.utc),
            )

        assert result["risk_score"] >= 3.0
