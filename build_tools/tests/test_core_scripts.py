# Copyright 2026 The HRX Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from fetch_rocm_artifacts import S3Object, select_available, wanted_artifacts
from hrx_build_tools import copy_tree_contents, flatten_therock_artifact, sha256_file


class CoreScriptTest(unittest.TestCase):
    def test_core_wanted_artifacts_excludes_upstream_hip(self):
        wanted = wanted_artifacts("core")
        self.assertIn("core-runtime_dev_generic", wanted)
        self.assertIn("amd-llvm_run_generic", wanted)
        self.assertIn("aqlprofile_run_generic", wanted)
        self.assertNotIn("amd-llvm_dev_generic", wanted)
        self.assertNotIn("core-hip_lib_generic", wanted)

    def test_select_available_prefers_zst_and_reports_missing(self):
        prefix = "123-linux/"
        available = [
            S3Object(prefix + "base_lib_generic.tar.xz", 10, "old"),
            S3Object(prefix + "base_lib_generic.tar.zst", 8, "new"),
            S3Object(prefix + "base_lib_generic.tar.zst.sha256sum", 1, "new"),
        ]
        selected, missing = select_available(
            available, prefix, ["base_lib_generic", "core-runtime_lib_generic"]
        )
        self.assertEqual(
            [obj.key for obj in selected], [prefix + "base_lib_generic.tar.zst"]
        )
        self.assertEqual(missing, ["core-runtime_lib_generic"])

    def test_flatten_therock_artifact_archive(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            archive = root / "base_lib_generic.tar.zst"
            output = root / "out"
            self._write_artifact_archive(
                archive,
                {
                    "artifact_manifest.txt": b"project/build/stage\n",
                    "project/build/stage/lib/libexample.so": b"example",
                    "project/build/stage/include/ignored.h": b"header",
                },
            )
            relroots = flatten_therock_artifact(archive, output)
            self.assertEqual(relroots, {"project/build/stage"})
            self.assertEqual(
                (output / "lib" / "libexample.so").read_bytes(), b"example"
            )
            self.assertEqual((output / "include" / "ignored.h").read_bytes(), b"header")

    def test_sha256_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "data"
            path.write_bytes(b"abc")
            self.assertEqual(
                sha256_file(path),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )

    def test_copy_tree_contents_can_skip_names(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "src"
            dst = root / "dst"
            (src / "lib").mkdir(parents=True)
            (src / ".download_cache").mkdir()
            (src / "lib" / "libexample.so").write_text("example")
            (src / ".download_cache" / "archive.tar.xz").write_text("cache")
            copy_tree_contents(src, dst, skip_names={".download_cache"})
            self.assertTrue((dst / "lib" / "libexample.so").exists())
            self.assertFalse((dst / ".download_cache").exists())

    def _write_artifact_archive(self, path: Path, files: dict[str, bytes]) -> None:
        import zstandard

        raw = io.BytesIO()
        with tarfile.open(fileobj=raw, mode="w") as tf:
            for name, data in files.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                info.mode = 0o644
                tf.addfile(info, io.BytesIO(data))
        path.write_bytes(zstandard.ZstdCompressor().compress(raw.getvalue()))


if __name__ == "__main__":
    unittest.main()
