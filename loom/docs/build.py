#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Builds and serves the Loom programming guide."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = REPO_ROOT / "loom" / "docs"
MKDOCS_CONFIG = DOCS_ROOT / "mkdocs.yml"
DEFAULT_SITE_DIR = REPO_ROOT / "build" / "loom-docs" / "site"


def _repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _run_mkdocs(arguments: list[str]) -> None:
    command = [
        sys.executable,
        "-m",
        "mkdocs",
        *arguments,
        "--config-file",
        str(MKDOCS_CONFIG),
        "--strict",
    ]
    environment = os.environ.copy()
    python_paths = [str(DOCS_ROOT)]
    if configured_python_path := environment.get("PYTHONPATH"):
        python_paths.append(configured_python_path)
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)


def _build(args: argparse.Namespace) -> None:
    site_dir = _repo_path(args.site_dir).resolve()
    _run_mkdocs(["build", "--site-dir", str(site_dir)])
    print(f"Loom programming guide: {site_dir / 'index.html'}")


def _serve(args: argparse.Namespace) -> None:
    _run_mkdocs(["serve", "--dev-addr", args.address])


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or serve the Loom programming guide."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build the static site.")
    build_parser.add_argument(
        "--site-dir",
        type=Path,
        default=DEFAULT_SITE_DIR,
        help=f"Output directory. Defaults to {DEFAULT_SITE_DIR}.",
    )
    build_parser.set_defaults(handler=_build)

    serve_parser = subparsers.add_parser(
        "serve", help="Serve the site with live reload."
    )
    serve_parser.add_argument(
        "--address",
        default="127.0.0.1:8000",
        help="Development server address. Defaults to 127.0.0.1:8000.",
    )
    serve_parser.set_defaults(handler=_serve)

    return parser.parse_args()


def main() -> None:
    args = _parse_arguments()
    args.handler(args)


if __name__ == "__main__":
    main()
