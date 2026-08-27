# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

from loom.gen import bootstrap


def test_import_does_not_require_repository_layout(tmp_path: Path) -> None:
    module_path = tmp_path / "runfiles" / "_main" / "loom" / "py" / "loom" / "gen" / "bootstrap.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_bytes(Path(bootstrap.__file__).read_bytes())

    spec = importlib.util.spec_from_file_location("materialized_loom_bootstrap", module_path)
    assert spec is not None
    assert spec.loader is not None
    materialized_bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(materialized_bootstrap)

    try:
        materialized_bootstrap.find_repo_root(module_path)
    except RuntimeError:
        pass
    else:
        raise AssertionError("materialized runfiles unexpectedly contained a Loom repository")


def test_repository_packages_precede_ambient_imports(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    script_path = repo_root / "loom" / "py" / "loom" / "gen" / "run.py"
    script_path.parent.mkdir(parents=True)
    script_path.touch()
    (repo_root / "loom" / "src" / "loom").mkdir(parents=True)

    ambient_packages = tmp_path / "ambient-packages"
    ambient_packages.mkdir()
    runtime_py = str(repo_root / "loom" / "py")
    repo_root_string = str(repo_root)
    runtime_probe = repo_root / "loom" / "py" / "loom_bootstrap_runtime_probe.py"
    repository_probe = repo_root / "loom_bootstrap_repository_probe.py"
    runtime_probe.write_text("", encoding="utf-8")
    repository_probe.write_text("", encoding="utf-8")
    (ambient_packages / runtime_probe.name).write_text("", encoding="utf-8")
    (ambient_packages / repository_probe.name).write_text("", encoding="utf-8")

    original_path = list(sys.path)
    try:
        sys.path[:] = [
            str(ambient_packages),
            runtime_py,
            "other-packages",
            runtime_py,
            *original_path,
        ]
        importlib.invalidate_caches()

        result = bootstrap.ensure_repository_packages_on_path(script_path)
        repeated_result = bootstrap.ensure_repository_packages_on_path(script_path)

        assert result == repo_root
        assert repeated_result == repo_root
        assert sys.path[:4] == [
            runtime_py,
            repo_root_string,
            str(ambient_packages),
            "other-packages",
        ]
        assert sys.path.count(runtime_py) == 1
        assert sys.path.count(repo_root_string) == 1
        runtime_spec = importlib.util.find_spec(runtime_probe.stem)
        repository_spec = importlib.util.find_spec(repository_probe.stem)
        assert runtime_spec is not None
        assert runtime_spec.origin is not None
        assert Path(runtime_spec.origin) == runtime_probe
        assert repository_spec is not None
        assert repository_spec.origin is not None
        assert Path(repository_spec.origin) == repository_probe
    finally:
        sys.path[:] = original_path
        importlib.invalidate_caches()
