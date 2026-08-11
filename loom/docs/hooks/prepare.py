# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Assembles authored and generated Loom documentation for MkDocs."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any

DOCS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
AUTHORED_SOURCE_ROOT = DOCS_ROOT / "src"
PYTHON_SOURCE_ROOT = REPO_ROOT / "loom" / "py"
DEFAULT_WORK_ROOT = REPO_ROOT / "build" / "loom-docs"

if str(PYTHON_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_SOURCE_ROOT))

from loom.gen.docs.reference import (  # noqa: E402
    generate_reference_files,
    write_reference_files,
)


def _work_root() -> Path:
    configured_root = Path(os.environ.get("LOOM_DOCS_WORK_DIR", DEFAULT_WORK_ROOT))
    if not configured_root.is_absolute():
        configured_root = REPO_ROOT / configured_root
    work_root = configured_root.resolve()
    protected_roots = {
        Path("/").resolve(),
        REPO_ROOT.resolve(),
        DOCS_ROOT.resolve(),
        AUTHORED_SOURCE_ROOT.resolve(),
    }
    if work_root in protected_roots:
        raise ValueError(
            "LOOM_DOCS_WORK_DIR must name a dedicated generated-output directory"
        )
    return work_root


def _staged_source_root() -> Path:
    return _work_root() / "mkdocs-source"


def _prepare_staged_source(staged_source_root: Path) -> None:
    if staged_source_root.exists():
        shutil.rmtree(staged_source_root)
    shutil.copytree(AUTHORED_SOURCE_ROOT, staged_source_root)
    files = generate_reference_files()
    write_reference_files(staged_source_root / "reference", files)


def on_config(config: Any) -> Any:
    """Points MkDocs at an isolated, generated source tree."""

    staged_source_root = _staged_source_root()
    staged_source_root.mkdir(parents=True, exist_ok=True)
    config["docs_dir"] = str(staged_source_root)
    return config


def on_pre_build(config: Any) -> None:
    """Rebuilds the complete source tree before every MkDocs build."""

    _prepare_staged_source(Path(config["docs_dir"]))


def on_page_context(context: Any, page: Any, config: Any, nav: Any) -> Any:
    """Hides edit links for build-generated reference pages."""

    del config, nav
    if page.file.src_path.startswith(
        (
            "reference/attributes/",
            "reference/dialects/",
            "reference/encodings/",
            "reference/types/",
        )
    ):
        page.edit_url = None
    return context
