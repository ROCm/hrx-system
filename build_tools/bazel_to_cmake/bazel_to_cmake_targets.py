# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import re
from typing import Dict, List


class TargetConverter:
    def __init__(self, repo_map: Dict[str, str]):
        self._explicit_target_mapping = {}
        self._repo_map = repo_map

        iree_repo = self._repo_alias("@iree")
        self._update_target_mappings(
            {
                # Internal utilities to emulate various binary/library options.
                f"{iree_repo}//build_tools:pthreads": [],
                f"{iree_repo}//build_tools:dl": ["${CMAKE_DL_LIBS}"],
                f"{iree_repo}//build_tools:rt": [],
                # HIP
                "@hip_api_headers//:headers": [
                    "iree::third_party::hip_api_headers",
                ],
                # NCCL
                "@nccl//:headers": [
                    "nccl::headers",
                ],
                # RCCL
                "@rccl//:headers": [
                    "iree::third_party::rccl_headers",
                ],
                # Tracy.
                "@tracy_client//:runtime": ["tracy_client::runtime"],
                # Misc single targets
                "//third_party:google_benchmark": [
                    "iree::third_party::google_benchmark"
                ],
                "//third_party:google_benchmark_main": [
                    "iree::third_party::google_benchmark_main"
                ],
                "//third_party:flatcc": ["iree-flatcc-cli"],
                "//third_party:flatcc_compiler": ["iree-flatcc-cli"],
                "//third_party:flatcc_parsing": ["iree::third_party::flatcc_parsing"],
                "//third_party:flatcc_runtime": ["iree::third_party::flatcc_runtime"],
                "//third_party:google_test": ["iree::third_party::google_test"],
                "//third_party:google_test_main": [
                    "iree::third_party::google_test_main"
                ],
                "@spirv_cross//:spirv_cross_lib": ["spirv-cross-msl"],
                "//third_party:hsa_runtime_headers": [
                    "iree::third_party::hsa_runtime_headers"
                ],
                "//third_party:aqlprofile_sdk_headers": [
                    "iree::third_party::aqlprofile_sdk_headers"
                ],
                "//third_party:hip_api_headers": ["iree::third_party::hip_api_headers"],
                "//third_party:libbacktrace": ["${IREE_LIBBACKTRACE_TARGET}"],
                "//third_party:rccl_headers": ["iree::third_party::rccl_headers"],
                "//third_party:spirv_as": ["iree::third_party::spirv_as"],
                "//third_party:spirv_dis": ["iree::third_party::spirv_dis"],
                "//third_party:vulkan_headers": ["iree::third_party::vulkan_headers"],
                "//third_party:catch2": ["iree::third_party::catch2"],
                "@webgpu_headers": [],
                # py_binary targets have no CMake equivalent.
                # This is the only target bazel needs to execute the lit tests.
                ":python_with_numpy": [],
            }
        )

        self._initialize()

    def _initialize(self):
        pass

    def _repo_alias(self, repo_name: str) -> str:
        """Returns the prefix of a repo (i.e. '@iree') given the repo map."""
        return self._repo_map.get(repo_name, repo_name)

    def _normalize_target_repo_alias(self, target: str) -> str:
        iree_repo = self._repo_alias("@iree")
        if target.startswith("@iree//") and iree_repo != "@iree":
            target_body = target[len("@iree//") :]
            if iree_repo:
                return f"{iree_repo}//{target_body}"
            return f"//{target_body}"
        return target

    def _update_target_mappings(self, mappings: Dict[str, List[str]]):
        self._explicit_target_mapping.update(mappings)

    def _convert_iree_cuda_target(self, target):
        # Convert like:
        #   @iree_cuda//:libdevice_embedded -> iree_cuda::libdevice_embedded
        label = target.rsplit(":")[-1]
        return [f"iree_cuda::{label}"]

    def _convert_to_cmake_path(self, bazel_path_fragment: str) -> str:
        cmake_path = bazel_path_fragment
        # Bazel `//iree/base`     -> CMake `iree::base`
        # Bazel `//iree/base:foo` -> CMake `iree::base::foo`
        if cmake_path.startswith("//"):
            cmake_path = cmake_path[len("//") :]
        cmake_path = cmake_path.replace(":", "::")  # iree/base::foo or ::foo
        cmake_path = cmake_path.replace("/", "::")  # iree::base
        return cmake_path

    def _repo_local_package(self, target: str) -> str:
        iree_repo = self._repo_alias("@iree")
        repo_prefix = f"{iree_repo}//"
        if target.startswith(repo_prefix):
            label_body = target[len(repo_prefix) :]
        elif target.startswith("//"):
            label_body = target[len("//") :]
        else:
            label_body = target
        return label_body.split(":", 1)[0]

    def _is_removed_compiler_package(self, target: str) -> bool:
        package = self._repo_local_package(target)
        return (
            package == "compiler"
            or package.startswith("compiler/")
            or package == "llvm-external-projects"
            or package.startswith("llvm-external-projects/")
        )

    def convert_target(self, target):
        """Converts a Bazel target to a list of CMake targets.

        IREE targets are expected to follow a standard form between Bazel and CMake
        that facilitates conversion. External targets *may* have their own patterns,
        or they may be purely special cases.

        Multiple target in Bazel may map to a single target in CMake and a Bazel
        target may map to multiple CMake targets.

        Returns:
          A list of converted targets if it was successfully converted.

        Raises:
          KeyError: No conversion was found for the target.
        """
        target = self._normalize_target_repo_alias(target)
        iree_repo = self._repo_alias("@iree")
        if target in self._explicit_target_mapping:
            return self._explicit_target_mapping[target]
        if target.startswith("@iree_cuda//"):
            return self._convert_iree_cuda_target(target)
        # pip dependencies don't exist in CMake (system Python is used).
        if target.startswith("@pip//"):
            return []
        if target.startswith(f"{iree_repo}//"):
            return self._convert_iree_repo_target(target)
        if target.startswith("@"):
            raise KeyError(f"No conversion found for target '{target}'")

        # Pass through package-relative targets
        #   :target_name
        #   file_name.txt
        if target.startswith(":") or (":" not in target and not target.startswith("/")):
            # Bazel data deps can reference generated files by package-relative
            # labels. Treat labels that look like file names as files instead of
            # CMake target aliases so `:foo.so` becomes `foo.so`, not `::foo.so`.
            if target.startswith(":") and "." in target.rsplit("/", 1)[-1]:
                return [target[1:]]
            return [self._convert_to_cmake_path(target)]

        return self._convert_unmatched_target(target)

    def _convert_iree_repo_target(self, target):
        iree_repo = self._repo_alias("@iree")
        if self._is_removed_compiler_package(target):
            raise ValueError(f"No target matching for removed compiler label {target}")

        # IREE root paths map to package names based on explicit rules.
        #   * runtime/src/iree/ creates its own root path by trimming down to
        #     just "iree"
        #   * tools/ uses an empty root, for binary target names like "iree-run-module"
        #   * other top level directories add back an 'iree' prefix
        # If changing these, make the corresponding change in iree_macros.cmake
        # (iree_package_ns function).

        # Map //runtime/src/iree/(.*) -> iree::\1
        m = re.match(f"^{iree_repo}//runtime/src/iree/(.+)", target)
        if m:
            return ["iree::" + self._convert_to_cmake_path(m.group(1))]

        # Map root tool aliases to the runtime tool namespace.
        m = re.match(f"^{iree_repo}//tools[|:](.+)", target)
        if m:
            return ["iree::tools::" + self._convert_to_cmake_path(m.group(1))]

        return self._convert_unmatched_target(target)

    def _convert_unmatched_target(self, target: str) -> str:
        """Converts unmatched targets in a repo specific way."""
        raise ValueError(f"No target matching for {target}")
