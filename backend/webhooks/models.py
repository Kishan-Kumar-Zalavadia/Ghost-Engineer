"""Pydantic models for GitLab webhook payloads.

Only the fields we actually use are declared; extra fields are ignored.
"""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

class GitLabProject(BaseModel):
    id: int
    name: str
    path_with_namespace: str
    web_url: str
    default_branch: Optional[str] = None

    model_config = {"extra": "ignore"}


class GitLabAuthor(BaseModel):
    name: str
    email: str

    model_config = {"extra": "ignore"}


# ---------------------------------------------------------------------------
# Push event
# ---------------------------------------------------------------------------

class GitLabCommit(BaseModel):
    id: str                          # SHA
    message: str
    title: str
    timestamp: datetime
    url: str
    author: GitLabAuthor
    added: list[str] = Field(default_factory=list)
    modified: list[str] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)

    model_config = {"extra": "ignore"}


class PushEvent(BaseModel):
    object_kind: Literal["push", "tag_push"]
    event_name: str
    before: str                      # SHA before push
    after: str                       # SHA after push (0000... = branch deleted)
    ref: str                         # e.g. refs/heads/main
    user_name: str
    user_email: Optional[str] = None
    user_username: Optional[str] = None
    project: GitLabProject
    commits: list[GitLabCommit] = Field(default_factory=list)
    total_commits_count: int = 0

    model_config = {"extra": "ignore"}

    @property
    def branch(self) -> str:
        return self.ref.removeprefix("refs/heads/").removeprefix("refs/tags/")

    @property
    def is_deletion(self) -> bool:
        return self.after == "0" * 40


# ---------------------------------------------------------------------------
# Merge request event
# ---------------------------------------------------------------------------

class MRAttributes(BaseModel):
    id: int
    iid: int
    title: str
    state: str                       # opened / closed / locked / merged
    action: Optional[str] = None     # open / close / reopen / update / merge …
    description: Optional[str] = None
    source_branch: str
    target_branch: str
    url: str
    author_id: int
    merge_status: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = {"extra": "ignore"}


class MRUser(BaseModel):
    id: int
    name: str
    username: str

    model_config = {"extra": "ignore"}


class MergeRequestEvent(BaseModel):
    object_kind: Literal["merge_request"]
    event_type: str
    user: MRUser
    project: GitLabProject
    object_attributes: MRAttributes

    model_config = {"extra": "ignore"}
