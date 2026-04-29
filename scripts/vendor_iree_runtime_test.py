#!/usr/bin/env python3
# Copyright 2026 The HRX Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("vendor_iree_runtime.py")
SPEC = importlib.util.spec_from_file_location("vendor_iree_runtime", SCRIPT_PATH)
vendor = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["vendor_iree_runtime"] = vendor
SPEC.loader.exec_module(vendor)


def run(args: list[str], cwd: Path) -> str:
    result = subprocess.run(
        args,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def git(cwd: Path, *args: str) -> str:
    return run(["git", *args], cwd)


def commit_all(repo: Path, message: str) -> str:
    git(repo, "add", ".")
    git(
        repo,
        "-c",
        "user.name=HRX Test",
        "-c",
        "user.email=hrx-test@example.com",
        "commit",
        "-m",
        message,
    )
    return git(repo, "rev-parse", "HEAD")


class VendorIreeRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.iree = self.root / "iree"
        self.flatcc = self.root / "flatcc"
        self.vendor_dir = self.root / "out" / "iree-runtime"
        self.patch_dir = self.root / "patches"
        self.patch_dir.mkdir()
        self._init_flatcc()
        self._init_iree()
        self.old_paths = vendor.VENDOR_PATHS
        self.old_submodules = vendor.VENDORED_SUBMODULES
        self.old_git_deps = vendor.VENDORED_GIT_DEPS
        self.old_stub_cmake_dirs = vendor.STUB_CMAKE_DIRS
        vendor.VENDOR_PATHS = (
            "CMakeLists.txt",
            "runtime",
            "build_tools/cmake",
        )
        vendor.VENDORED_SUBMODULES = ("third_party/flatcc",)
        vendor.VENDORED_GIT_DEPS = ()
        vendor.STUB_CMAKE_DIRS = ("tools",)

    def tearDown(self) -> None:
        vendor.VENDOR_PATHS = self.old_paths
        vendor.VENDORED_SUBMODULES = self.old_submodules
        vendor.VENDORED_GIT_DEPS = self.old_git_deps
        vendor.STUB_CMAKE_DIRS = self.old_stub_cmake_dirs
        self.temp.cleanup()

    def _init_flatcc(self) -> None:
        self.flatcc.mkdir()
        git(self.flatcc, "init")
        (self.flatcc / "include").mkdir()
        (self.flatcc / "include" / "flatcc.h").write_text("flatcc\n")
        (self.flatcc / "flatcc_test.cc").write_text("test\n")
        self.flatcc_commit = commit_all(self.flatcc, "flatcc")

    def _init_iree(self) -> None:
        self.iree.mkdir()
        git(self.iree, "init")
        (self.iree / "CMakeLists.txt").write_text("cmake\n")
        (self.iree / "runtime").mkdir()
        (self.iree / "runtime" / "runtime.c").write_text("runtime\n")
        (self.iree / "runtime" / "runtime_test.cc").write_text("test\n")
        (self.iree / "runtime" / "testdata").mkdir()
        (self.iree / "runtime" / "testdata" / "data.bin").write_text("data\n")
        (self.iree / "build_tools" / "cmake").mkdir(parents=True)
        (self.iree / "build_tools" / "cmake" / "iree.cmake").write_text("helper\n")
        git(
            self.iree,
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            str(self.flatcc),
            "third_party/flatcc",
        )
        self.iree_commit = commit_all(self.iree, "iree")

    def test_update_imports_filtered_tree_and_metadata(self) -> None:
        vendor.generate_vendor_tree(
            iree_repo=self.iree,
            ref=self.iree_commit,
            destination=self.vendor_dir,
            patch_dir=self.patch_dir,
        )

        self.assertTrue((self.vendor_dir / "runtime" / "runtime.c").exists())
        self.assertTrue((self.vendor_dir / "runtime" / "runtime_test.cc").exists())
        self.assertTrue((self.vendor_dir / "runtime" / "testdata").exists())
        self.assertTrue(
            (self.vendor_dir / "third_party" / "flatcc" / "include" / "flatcc.h")
            .exists()
        )
        self.assertTrue(
            (self.vendor_dir / "third_party" / "flatcc" / "flatcc_test.cc").exists()
        )
        self.assertFalse(
            (self.vendor_dir / "third_party" / "flatcc" / ".git").exists()
        )
        self.assertTrue((self.vendor_dir / "tools" / "CMakeLists.txt").exists())
        self.assertFalse((self.vendor_dir / "HRX_VENDOR.json").exists())
        metadata = vendor.metadata_path(self.vendor_dir).read_text()
        self.assertIn(self.iree_commit, metadata)
        self.assertIn(self.flatcc_commit, metadata)
        self.assertIn("hsa-runtime64", metadata)

    def test_check_detects_drift(self) -> None:
        vendor.generate_vendor_tree(
            iree_repo=self.iree,
            ref=self.iree_commit,
            destination=self.vendor_dir,
            patch_dir=self.patch_dir,
        )
        (self.vendor_dir / "runtime" / "runtime.c").write_text("changed\n")

        differences = []
        with tempfile.TemporaryDirectory() as td:
            generated = Path(td) / "generated"
            vendor.generate_vendor_tree(
                iree_repo=self.iree,
                ref=self.iree_commit,
                destination=generated,
                patch_dir=self.patch_dir,
            )
            differences = vendor.compare_trees(generated, self.vendor_dir)
        self.assertIn("changed: runtime/runtime.c", differences)

    def test_patch_stack_applies_after_import(self) -> None:
        (self.patch_dir / "0001-change-runtime.patch").write_text(
            """diff --git a/runtime/runtime.c b/runtime/runtime.c
index c4f72b6..03cfd74 100644
--- a/runtime/runtime.c
+++ b/runtime/runtime.c
@@ -1 +1 @@
-runtime
+patched
"""
        )

        vendor.generate_vendor_tree(
            iree_repo=self.iree,
            ref=self.iree_commit,
            destination=self.vendor_dir,
            patch_dir=self.patch_dir,
        )

        self.assertEqual(
            "patched\n", (self.vendor_dir / "runtime" / "runtime.c").read_text()
        )

    def test_dumped_patches_apply_as_commits(self) -> None:
        hrx = self.root / "hrx"
        vendor_dir = hrx / "third_party" / "iree-runtime"
        patch_dir = hrx / "scripts" / "iree-runtime-patches"
        (vendor_dir / "runtime").mkdir(parents=True)
        patch_dir.mkdir(parents=True)
        git(hrx, "init")
        git(hrx, "config", "user.name", "HRX Test")
        git(hrx, "config", "user.email", "hrx-test@example.com")
        (vendor_dir / "runtime" / "runtime.c").write_text("runtime\n")
        pristine = commit_all(hrx, "Vendor pristine IREE runtime test")

        (vendor_dir / "runtime" / "runtime.c").write_text("patched\n")
        commit_all(hrx, "Teach runtime about tests")

        old_hrx_repo = vendor.hrx_repo
        vendor.hrx_repo = lambda: hrx
        try:
            vendor.dump_patches(
                argparse.Namespace(
                    vendor_dir=vendor_dir,
                    patch_dir=patch_dir,
                    diffbase=pristine,
                )
            )
            patches = vendor.patch_files(patch_dir)
            self.assertEqual(1, len(patches))
            self.assertIn("Teach runtime about tests", patches[0].read_text())
            saved_patch_dir = self.root / "saved-patches"
            saved_patch_dir.mkdir()
            for patch in patches:
                (saved_patch_dir / patch.name).write_text(patch.read_text())

            git(hrx, "reset", "--hard", pristine)
            run(["git", "clean", "-fd", "scripts"], hrx)
            vendor.apply_patch_queue(
                argparse.Namespace(vendor_dir=vendor_dir, patch_dir=saved_patch_dir)
            )
            self.assertEqual(
                "patched\n", (vendor_dir / "runtime" / "runtime.c").read_text()
            )
            self.assertNotEqual(pristine, git(hrx, "rev-parse", "HEAD"))
            self.assertEqual(
                "Teach runtime about tests",
                git(hrx, "show", "-s", "--format=%s", "HEAD"),
            )
        finally:
            vendor.hrx_repo = old_hrx_repo

    def test_submodule_must_match_superproject_commit(self) -> None:
        (self.flatcc / "new_file").write_text("new\n")
        new_commit = commit_all(self.flatcc, "move flatcc")
        git(self.iree / "third_party" / "flatcc", "fetch", str(self.flatcc), new_commit)
        git(self.iree / "third_party" / "flatcc", "checkout", new_commit)

        with self.assertRaises(vendor.VendorError):
            vendor.generate_vendor_tree(
                iree_repo=self.iree,
                ref=self.iree_commit,
                destination=self.vendor_dir,
                patch_dir=self.patch_dir,
            )


if __name__ == "__main__":
    unittest.main()
