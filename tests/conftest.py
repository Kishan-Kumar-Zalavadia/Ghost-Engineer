"""
Pytest configuration and shared fixtures for the Ghost-Engineer test suite.

Sets required environment variables before any project modules are imported so
that pydantic-settings does not raise validation errors when config.Settings is
instantiated during collection.
"""

import os

# ---------------------------------------------------------------------------
# Environment bootstrap — must happen before any backend.* import
# ---------------------------------------------------------------------------

# Provide a valid integer for gitlab_project_id so pydantic-settings can
# parse the Settings model even when the real .env has a placeholder value.
os.environ.setdefault("GITLAB_PROJECT_ID", "0")
os.environ.setdefault("GITLAB_TOKEN", "test-token")
os.environ.setdefault("GITLAB_WEBHOOK_TOKEN", "test-webhook-token")
os.environ.setdefault("MONGODB_URI", "mongodb://localhost:27017")
