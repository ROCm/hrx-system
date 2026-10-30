# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Real filesystem tests for the declared source view."""

import tempfile
from pathlib import Path

import pytest

from loom.tools.source_inventory import inventory


def test_membership_does_not_depend_on_contents_or_build_registration():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        sources = root / "loom/nested package"
        sources.mkdir(parents=True)
        source = sources / "unregistered.loom"
        source.write_text("first", encoding="utf-8")
        first = inventory(root)
        assert first["sources"] == ["loom/nested package/unregistered.loom"]
        assert first["directories"] == {
            "loom": {"nested package": True},
            "loom/nested package": {"unregistered.loom": False},
        }
        source.write_text("different contents", encoding="utf-8")
        assert inventory(root) == first
        consumer = sources / "consumer.loom-test"
        consumer.touch()
        assert inventory(root)["sources"] == [
            "loom/nested package/consumer.loom-test",
            "loom/nested package/unregistered.loom",
        ]
        consumer.unlink()
        assert inventory(root) == first
        source.rename(sources / "renamed.loom")
        assert inventory(root)["sources"] == ["loom/nested package/renamed.loom"]


def test_hidden_scratch_and_other_source_types_are_not_inputs():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        scratch = root / "loom/.notes"
        scratch.mkdir(parents=True)
        (scratch / "broken.loom").touch()
        (root / "loom/.hidden.loom").touch()
        (root / "loom/source.c").touch()
        assert inventory(root)["sources"] == []
        assert inventory(root)["directories"] == {"loom": {"source.c": False}}


def test_missing_source_root_fails():
    with tempfile.TemporaryDirectory() as temporary, pytest.raises(FileNotFoundError):
        inventory(Path(temporary))


def test_check_families_preserve_source_and_build_policy_coverage():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        paths = [
            "loom/unregistered.loom-test",
            "loom/src/loom/format/testdata/invalid.loom",
            "loom/src/loom/test/corpus/authoring/example.loom",
            "loom/src/loom/target/example/BUILD.bazel",
            "loom/src/loom/target/example/CMakeLists.txt",
            "loom/src/loom/target/example/source.c",
            "loom/src/loom/target/example/source.cc",
            "loom/src/loom/target/example/source.h",
            "loom/src/loom/target/example/source.txt",
            "loom/not_repository_policy.c",
        ]
        for path in paths:
            source = root / path
            source.parent.mkdir(parents=True, exist_ok=True)
            source.touch()
        result = inventory(root)
        assert result["sources"] == sorted(paths[:3])
        assert result["format_sources"] == sorted([paths[0], paths[2]])
        assert result["policy_sources"] == sorted(paths[2:8])
        (root / paths[3]).write_text("changed build policy", encoding="utf-8")
        assert inventory(root) == result
        (root / paths[3]).unlink()
        assert inventory(root)["policy_sources"] == sorted([paths[2], *paths[4:8]])


@pytest.mark.parametrize(
    "path",
    [
        "loom/src/loom/serve/runtime/program/testdata/compilation/invalid.loom",
        "loom/src/loom/tools/loom-benchmark/testdata/duplicate_symbol.loom",
    ],
)
def test_negative_fixtures_retain_lint_coverage_without_canonical_formatting(path):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / path
        source.parent.mkdir(parents=True)
        source.touch()
        result = inventory(root)
        assert result["sources"] == [path]
        assert result["format_sources"] == []


def test_directory_removal_and_file_directory_transitions_change_membership():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "loom").mkdir()
        node = root / "loom/node"
        node.touch()
        first = inventory(root)
        assert first["directories"] == {"loom": {"node": False}}
        node.unlink()
        node.mkdir()
        source = node / "source.loom"
        source.touch()
        directory = inventory(root)
        assert directory["sources"] == ["loom/node/source.loom"]
        assert directory["directories"] == {
            "loom": {"node": True},
            "loom/node": {"source.loom": False},
        }
        source.unlink()
        node.rmdir()
        assert inventory(root)["directories"] == {"loom": {}}
        node.touch()
        assert inventory(root) == first


def test_symlink_directory_cannot_hide_coverage_or_create_a_cycle():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "loom").mkdir()
        link = root / "loom/cycle"
        link.symlink_to(root / "loom", target_is_directory=True)
        with pytest.raises(ValueError, match="directory must not be a symlink"):
            inventory(root)


def test_missing_source_symlink_fails():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "loom").mkdir()
        (root / "loom/missing.loom").symlink_to(root / "absent")
        with pytest.raises(ValueError, match="source must be a regular file"):
            inventory(root)
