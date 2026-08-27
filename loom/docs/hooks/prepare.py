# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Assembles authored and generated Loom documentation for MkDocs."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

DOCS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
AUTHORED_SOURCE_ROOT = DOCS_ROOT / "src"
EXAMPLE_SOURCE_ROOT = DOCS_ROOT / "examples"
PYTHON_SOURCE_ROOT = REPO_ROOT / "loom" / "py"
DEFAULT_WORK_ROOT = REPO_ROOT / "build" / "loom-docs"
C_API_GENERATOR = REPO_ROOT / "loom" / "binding" / "c" / "doc" / "generate.sh"
DOC_GENERATOR_NAME = ".doc-generate.sh"

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


def _remove_generated_directory(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"generated documentation path must not be a symlink: {path}")
    if path.exists():
        shutil.rmtree(path)


def _generate_c_api(work_root: Path) -> Path:
    output_root = work_root / "c-api"
    _remove_generated_directory(output_root)
    environment = os.environ.copy()
    python_bin_dir = str(Path(sys.executable).parent)
    environment["PATH"] = python_bin_dir + os.pathsep + environment.get("PATH", "")
    subprocess.run(
        [
            str(C_API_GENERATOR),
            "--check",
            f"--output={output_root}",
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
    )
    html_root = output_root / "html"
    if not (html_root / "index.html").is_file():
        raise FileNotFoundError("Doxygen did not produce the Loom C API index")
    return html_root


def _generate_example_outputs(work_root: Path) -> Path:
    output_root = work_root / "examples"
    _remove_generated_directory(output_root)
    output_root.mkdir(parents=True)

    environment = os.environ.copy()
    python_bin_dir = str(Path(sys.executable).parent)
    environment["PATH"] = python_bin_dir + os.pathsep + environment.get("PATH", "")
    for generator in sorted(EXAMPLE_SOURCE_ROOT.rglob(DOC_GENERATOR_NAME)):
        relative_package = generator.parent.relative_to(EXAMPLE_SOURCE_ROOT)
        package_output_root = output_root / relative_package
        package_output_root.mkdir(parents=True)
        subprocess.run(
            [str(generator), str(package_output_root)],
            cwd=REPO_ROOT,
            env=environment,
            check=True,
        )
    return output_root


def _prepare_staged_source(
    staged_source_root: Path,
    c_api_html_root: Path,
    generated_example_root: Path,
) -> None:
    _remove_generated_directory(staged_source_root)
    shutil.copytree(AUTHORED_SOURCE_ROOT, staged_source_root)
    files = generate_reference_files()
    write_reference_files(staged_source_root / "reference", files)
    shutil.copytree(
        c_api_html_root,
        staged_source_root / "reference" / "c-api" / "generated",
    )
    shutil.copytree(
        generated_example_root,
        staged_source_root / "generated" / "examples",
    )


def on_config(config: Any) -> Any:
    """Points MkDocs at an isolated, generated source tree."""

    staged_source_root = _staged_source_root()
    config["docs_dir"] = str(staged_source_root)
    snippet_config = config.get("mdx_configs", {}).get("pymdownx.snippets")
    if snippet_config is not None:
        base_paths = snippet_config["base_path"]
        if str(staged_source_root) not in base_paths:
            base_paths.append(str(staged_source_root))
    return config


def on_pre_build(config: Any) -> None:
    """Rebuilds the complete source tree before every MkDocs build."""

    work_root = _work_root()
    c_api_html_root = _generate_c_api(work_root)
    generated_example_root = _generate_example_outputs(work_root)
    _prepare_staged_source(
        Path(config["docs_dir"]), c_api_html_root, generated_example_root
    )


def on_serve(server: Any, config: Any, builder: Any) -> Any:
    """Watches canonical inputs without watching generated staging output."""

    del builder
    server.unwatch(str(config["docs_dir"]))
    return server


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
