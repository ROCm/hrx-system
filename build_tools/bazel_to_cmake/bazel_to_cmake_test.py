#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Tests for bazel_to_cmake.py."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bazel_to_cmake


class BazelToCMakeTest(unittest.TestCase):
    def test_build_tree_tracks_inputs_and_retires_only_generated_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            package = source / "package"
            package.mkdir(parents=True)
            declaration = package / "BUILD.bazel"
            declaration.write_text(
                'load(":settings.bzl", "OPTIONS")\n'
                'cc_library(name="value", defines=OPTIONS)\n',
                encoding="utf-8",
            )
            settings = package / "settings.bzl"
            settings.write_text('OPTIONS = ["VALUE=1"]\n', encoding="utf-8")
            output = root / "build/.iree/cmake"
            with (
                mock.patch.object(bazel_to_cmake, "repo_root", str(source)),
                mock.patch.object(bazel_to_cmake, "repo_cfg", SimpleNamespace()),
            ):

                def generate():
                    bazel_to_cmake.generate_build_tree(output, ["package"])

                generate()
                generated = output / "package/package.cmake"
                self.assertFalse((package / "CMakeLists.txt").exists())
                self.assertIn("VALUE=1", generated.read_text())
                manifest = output / "inputs.cmake"
                self.assertIn(settings.as_posix(), manifest.read_text())
                self.assertIn(declaration.as_posix(), manifest.read_text())
                timestamps = (generated.stat().st_mtime_ns, manifest.stat().st_mtime_ns)
                generate()
                self.assertEqual(
                    (generated.stat().st_mtime_ns, manifest.stat().st_mtime_ns),
                    timestamps,
                )
                settings.write_text('OPTIONS = ["VALUE=2"]\n', encoding="utf-8")
                generate()
                self.assertIn("VALUE=2", generated.read_text())
                generated.unlink()
                generate()
                self.assertTrue(generated.exists())
                declaration.unlink()
                generate()
                self.assertFalse(generated.exists())
                # The generator owns its output, never the authored input.
                self.assertTrue(settings.exists())

    def test_authored_packages_and_skips_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "source", root / "build/.iree/cmake"
            source.mkdir()
            (source / "BUILD.bazel").write_text("package()\n")
            with (
                mock.patch.object(bazel_to_cmake, "repo_root", str(source)),
                mock.patch.object(bazel_to_cmake, "repo_cfg", SimpleNamespace()),
            ):
                bazel_to_cmake.generate_build_tree(output, ["."])
                generated = output / "CMakeLists.txt"
                generated.write_bytes(generated.read_bytes().replace(b"\n", b"\r\n"))
                bazel_to_cmake.generate_build_tree(output, ["."])
                self.assertNotIn(b"\r\n", generated.read_bytes())
                authored = source / "CMakeLists.txt"
                authored.write_text("# Authored package.\n")
                bazel_to_cmake.generate_build_tree(output, ["."])
                self.assertFalse(generated.exists())
                self.assertEqual(authored.read_text(), "# Authored package.\n")
                authored.unlink()
                (source / "BUILD.bazel").write_text("# bazel-to-cmake: skip\n")
                bazel_to_cmake.generate_build_tree(output, ["."])
                self.assertFalse(generated.exists())

    def test_output_cannot_overwrite_source_or_an_unowned_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "BUILD.bazel").write_text("package()\n")
            unowned = root / "unowned"
            unowned.mkdir()
            authored = unowned / "CMakeLists.txt"
            authored.write_text("# Authored package.\n")
            with mock.patch.object(bazel_to_cmake, "repo_root", str(source)):
                for output in (root, source, unowned):
                    with self.subTest(output=output), self.assertRaises(ValueError):
                        bazel_to_cmake.generate_build_tree(output, ["."])
            self.assertEqual(authored.read_text(), "# Authored package.\n")

    def test_traversal_reaches_authored_packages_without_replacing_them(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "source", root / "build/.iree/cmake"
            authored = source / "group/nested/authored/CMakeLists.txt"
            authored.parent.mkdir(parents=True)
            authored.write_text("project(authored)\n")
            with (
                mock.patch.object(bazel_to_cmake, "repo_root", str(source)),
                mock.patch.object(bazel_to_cmake, "repo_cfg", SimpleNamespace()),
            ):
                bazel_to_cmake.generate_build_tree(output, ["group"])
                self.assertEqual(
                    {
                        path.relative_to(output)
                        for path in output.rglob("CMakeLists.txt")
                    },
                    {Path("group/CMakeLists.txt"), Path("group/nested/CMakeLists.txt")},
                )
                self.assertEqual(authored.read_text(), "project(authored)\n")
                self.assertIn(
                    'include("${IREE_GENERATED_CMAKE_DIR}/group/nested/package.cmake")',
                    (output / "group/package.cmake").read_text(),
                )
                self.assertIn(
                    'iree_add_subdirectory("authored")',
                    (output / "group/nested/package.cmake").read_text(),
                )
                # A new authored parent takes over traversal and retires the
                # generated entry; its contents are never replaced.
                parent = source / "group/nested/CMakeLists.txt"
                parent.write_text("add_subdirectory(authored)\n")
                bazel_to_cmake.generate_build_tree(output, ["group"])
                self.assertFalse((output / "group/nested/CMakeLists.txt").exists())
                self.assertFalse((output / "group/nested/package.cmake").exists())
                self.assertEqual(parent.read_text(), "add_subdirectory(authored)\n")
                self.assertIn(
                    'iree_add_subdirectory("nested")',
                    (output / "group/package.cmake").read_text(),
                )

    def test_traversal_reaches_authored_directory_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "source", root / "build/.iree/cmake"
            source.mkdir()
            (source / "BUILD.bazel").write_text("package()\n")
            native = root / "native"
            native.mkdir()
            (native / "CMakeLists.txt").write_text("project(native)\n")
            try:
                (source / "linked").symlink_to(native, target_is_directory=True)
            except OSError as error:
                self.skipTest(f"Directory symlinks are unavailable: {error}")
            with (
                mock.patch.object(bazel_to_cmake, "repo_root", str(source)),
                mock.patch.object(bazel_to_cmake, "repo_cfg", SimpleNamespace()),
            ):
                bazel_to_cmake.generate_build_tree(output, ["."])
            self.assertIn(
                'iree_add_subdirectory("linked")',
                (output / "package.cmake").read_text(),
            )
            self.assertFalse((output / "linked/CMakeLists.txt").exists())

    def test_failed_first_conversion_can_be_repaired(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "source", root / "build/.iree/cmake"
            child = source / "child"
            child.mkdir(parents=True)
            (source / "BUILD.bazel").write_text("package()\n")
            declaration = child / "BUILD.bazel"
            declaration.write_text("unknown_rule()\n")
            with (
                mock.patch.object(bazel_to_cmake, "repo_root", str(source)),
                mock.patch.object(bazel_to_cmake, "repo_cfg", SimpleNamespace()),
            ):
                with self.assertRaisesRegex(RuntimeError, "unknown_rule"):
                    bazel_to_cmake.generate_build_tree(output, ["."])
                self.assertFalse(output.exists())
                declaration.write_text("package()\n")
                bazel_to_cmake.generate_build_tree(output, ["."])
                self.assertTrue((output / "child/CMakeLists.txt").exists())


if __name__ == "__main__":
    unittest.main()
