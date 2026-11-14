# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Deployment documents composed from a profile and local launcher settings.

The server owns the document's meaning; the launcher only chooses values it can
justify from the card. A card's preset fixes the model's own context and output
capacity, so a launcher setting may only narrow what the package declares: the
server rejects a request that does not fit, and a launch should not surprise a
user with a package that cannot prepare.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from loom.launch.errors import LaunchError
from loom.launch.profile import Profile

#: The server's own default when a deployment does not choose a value.
DEFAULT_MAX_TOKENS = 512
#: A launch never reserves more headroom than this for additional cached state.
DEFAULT_CACHE_TARGET_BYTES = 1 << 30


@dataclass(frozen=True)
class ContextSettings:
    """The capacities one launch asks the text service to provision."""

    context_length: int | None = None
    max_tokens: int | None = None
    rows: int | None = None
    max_sessions: int | None = None
    prefill_capacity: int | None = None
    checkpoint_capacity: int | None = None
    session_cache_capacity: int | None = None
    prefix_cache_capacity: int | None = None
    cache_target_bytes: int | None = None
    draft_depth: int | None = None
    continuation_epochs: int | None = None
    pending_requests: int | None = None

    def describe(self) -> dict:
        return {
            "context_length": self.context_length,
            "max_tokens": self.max_tokens,
            "rows": self.rows,
            "max_sessions": self.max_sessions,
            "prefill_capacity": self.prefill_capacity,
            "checkpoint_capacity": self.checkpoint_capacity,
            "session_cache_capacity": self.session_cache_capacity,
            "prefix_cache_capacity": self.prefix_cache_capacity,
            "cache_target_bytes": self.cache_target_bytes,
            "draft_depth": self.draft_depth,
            "continuation_epochs": self.continuation_epochs,
            "pending_requests": self.pending_requests,
        }

    def definition(self) -> dict:
        """The `definition` object a deployment document carries."""
        definition: dict = {"kind": "text"}
        native_names = {
            "rows": "rows",
            "max_sessions": "max_sessions",
            "context_length": "context_capacity",
            "prefill_capacity": "prefill_capacity",
            "checkpoint_capacity": "checkpoint_capacity",
            "session_cache_capacity": "session_cache_capacity",
            "prefix_cache_capacity": "prefix_cache_capacity",
            "cache_target_bytes": "cache_target_bytes",
            "max_tokens": "max_tokens",
            "draft_depth": "draft_depth",
            "continuation_epochs": "continuation_epochs",
            "pending_requests": "pending_requests",
        }
        for field_name, native_name in native_names.items():
            value = getattr(self, field_name)
            if value is not None:
                definition[native_name] = value
        return definition


def context_settings(
    profile: Profile,
    *,
    context_length: int | None = None,
    max_tokens: int | None = None,
    parallel: int | None = None,
    draft_depth: int | None = None,
    checkpoint_capacity: int | None = None,
    session_cache_capacity: int | None = None,
    prefix_cache_capacity: int | None = None,
    cache_target_bytes: int | None = None,
    prefill_capacity: int | None = None,
    continuation_epochs: int | None = None,
    pending_requests: int | None = None,
) -> ContextSettings:
    """Resolves one launch's capacities against the profile's own limits."""
    execution = profile.execution
    declared_max_tokens = None
    declared_rows = None
    declared_prefill = None
    declared_draft = None
    if execution is not None:
        declared_max_tokens = execution.max_tokens
        declared_rows = execution.rows
        declared_prefill = execution.prefill_capacity
        declared_draft = execution.draft_depth
    # The preset's served context is the default; the model's native context
    # is the ceiling a request may widen it to.
    served_context = profile.context_length or None
    native_context = profile.native_context_length or served_context
    resolved_context = context_length if context_length is not None else served_context
    if resolved_context is None:
        resolved_context = 16384
    if resolved_context < 1:
        raise LaunchError("context length must be a positive token count")
    if native_context and resolved_context > native_context:
        raise LaunchError(
            f"context {resolved_context} exceeds the model's {native_context} tokens; "
            "the package cannot serve more than it was prepared for"
        )
    resolved_max_tokens = max_tokens if max_tokens is not None else declared_max_tokens
    if resolved_max_tokens is None:
        resolved_max_tokens = DEFAULT_MAX_TOKENS
    if resolved_max_tokens < 1:
        raise LaunchError("max tokens must be a positive token count")
    if resolved_max_tokens >= resolved_context:
        raise LaunchError(
            f"max tokens {resolved_max_tokens} leaves no room in a "
            f"{resolved_context}-token context"
        )
    resolved_rows = declared_rows
    resolved_sessions = None
    if parallel is not None:
        if parallel < 1:
            raise LaunchError("parallel requests must be at least one")
        resolved_rows = parallel
        resolved_sessions = parallel
    resolved_prefill = (
        prefill_capacity if prefill_capacity is not None else declared_prefill
    )
    resolved_draft = draft_depth if draft_depth is not None else declared_draft
    if resolved_draft is not None and not 0 <= resolved_draft <= 7:
        raise LaunchError("draft depth must be between 0 and 7")
    return ContextSettings(
        context_length=resolved_context,
        max_tokens=resolved_max_tokens,
        rows=resolved_rows,
        max_sessions=resolved_sessions,
        prefill_capacity=resolved_prefill,
        checkpoint_capacity=checkpoint_capacity,
        session_cache_capacity=session_cache_capacity,
        prefix_cache_capacity=prefix_cache_capacity,
        cache_target_bytes=cache_target_bytes,
        draft_depth=resolved_draft,
        continuation_epochs=continuation_epochs,
        pending_requests=pending_requests,
    )


def deployment_document(
    profile: Profile,
    card_path: str | Path,
    settings: ContextSettings,
    *,
    name: str | None = None,
    preset: str | None = None,
) -> dict:
    """Composes the deployment document one launch registers."""
    document: dict = {
        "name": name or profile.slug,
        "definition": settings.definition(),
    }
    model: dict = {"path": _absolute(card_path)}
    selected = preset or (profile.execution.preset if profile.execution else None)
    if selected:
        model["preset"] = selected
    document["model"] = model
    return document


def _absolute(path: str | Path) -> str:
    """An absolute path, because the server resolves a relative one elsewhere.

    `loom-serve` reads `model.path` relative to the deployment document's own
    directory, so a relative path in a run directory can never be what the caller
    meant, however the caller spelled it.
    """
    return os.path.abspath(os.path.expanduser(str(path)))


def write_deployment(path: str | Path, document: dict) -> Path:
    """Writes a deployment document next to a run's logs."""
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path
