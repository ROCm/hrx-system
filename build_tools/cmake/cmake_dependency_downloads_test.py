# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from build_tools.cmake.test_environment import configured_cmake_arguments

CMAKE_COMMAND = os.environ["IREE_TEST_CMAKE_COMMAND"]
FETCH_HELPERS = Path(__file__).with_name("iree_third_party_helpers.cmake")


class CMakeDependencyDownloadsTest(unittest.TestCase):
    def test_github_urls_gain_independent_fallbacks(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            script = Path(temporary_dir) / "fallbacks.cmake"
            script.write_text(
                f'''include("{FETCH_HELPERS.as_posix()}")
set(input_urls
  "https://github.com/google/benchmark/archive/refs/tags/v1.9.5.tar.gz"
  "https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz"
  "https://github.com/facebook/zstd/releases/download/v1.5.7/zstd-1.5.7.tar.gz"
  "https://raw.githubusercontent.com/torvalds/linux/revision/include/uapi/linux/io_uring.h"
  "https://raw.githubusercontent.com/amd/xdna-driver/revision/include/uapi/drm/amdxdna_accel.h"
  "https://example.com/source.tar.gz"
)
_iree_dependency_expand_download_urls(actual_urls ${{input_urls}})
set(expected_urls
  "https://github.com/google/benchmark/archive/refs/tags/v1.9.5.tar.gz"
  "https://codeload.github.com/google/benchmark/tar.gz/refs/tags/v1.9.5"
  "https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz"
  "https://codeload.github.com/google/googletest/tar.gz/refs/tags/v1.17.0"
  "https://mirror.bazel.build/github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz"
  "https://github.com/facebook/zstd/releases/download/v1.5.7/zstd-1.5.7.tar.gz"
  "https://mirror.bazel.build/github.com/facebook/zstd/releases/download/v1.5.7/zstd-1.5.7.tar.gz"
  "https://raw.githubusercontent.com/torvalds/linux/revision/include/uapi/linux/io_uring.h"
  "https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/include/uapi/linux/io_uring.h?id=revision"
  "https://raw.githubusercontent.com/amd/xdna-driver/revision/include/uapi/drm/amdxdna_accel.h"
  "https://cdn.jsdelivr.net/gh/amd/xdna-driver@revision/include/uapi/drm/amdxdna_accel.h"
  "https://example.com/source.tar.gz"
)
if(NOT "${{actual_urls}}" STREQUAL "${{expected_urls}}")
  message(FATAL_ERROR
    "download URL expansion mismatch:\nactual: ${{actual_urls}}\nexpected: ${{expected_urls}}")
endif()
''',
                encoding="utf-8",
            )

            result = subprocess.run(
                [CMAKE_COMMAND, "-P", str(script)],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )

            self.assertEqual(result.returncode, 0, result.stdout)

    def test_locked_file_tries_transport_and_hash_failures(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            correct = root / "correct.txt"
            correct.write_text("correct\n", encoding="utf-8")
            wrong = root / "wrong.txt"
            wrong.write_text("wrong\n", encoding="utf-8")
            missing = root / "missing.txt"
            digest = hashlib.sha256(correct.read_bytes()).hexdigest()
            (root / "CMakeLists.txt").write_text(
                f'''cmake_minimum_required(VERSION 3.26)
project(dependency_download_test NONE)
include("{FETCH_HELPERS.as_posix()}")
set(IREE_DEP_SAMPLE_KIND "http_file")
set(IREE_DEP_SAMPLE_URLS
  "{missing.as_uri()}"
  "{wrong.as_uri()}"
  "{correct.as_uri()}"
)
set(IREE_DEP_SAMPLE_SHA256 "{digest}")
set(IREE_DEP_SAMPLE_DOWNLOADED_FILE_PATH "data/value.txt")
iree_populate_locked_file(sample output_file)
file(READ "${{output_file}}" output_value)
if(NOT output_value STREQUAL "correct\n")
  message(FATAL_ERROR "downloaded unexpected value: ${{output_value}}")
endif()
''',
                encoding="utf-8",
            )
            build = root / "build"

            for corrupt_cached_file in (False, True):
                if corrupt_cached_file:
                    cached = build / "_deps/sample-src/file/data/value.txt"
                    cached.write_text("stale\n", encoding="utf-8")
                result = subprocess.run(
                    [
                        CMAKE_COMMAND,
                        "-S",
                        str(root),
                        "-B",
                        str(build),
                        *configured_cmake_arguments(),
                    ],
                    check=False,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                self.assertEqual(result.returncode, 0, result.stdout)
                downloaded = build / "_deps/sample-src/file/data/value.txt"
                self.assertEqual(downloaded.read_text(encoding="utf-8"), "correct\n")
                self.assertFalse(downloaded.with_suffix(".txt.download").exists())


if __name__ == "__main__":
    unittest.main()
