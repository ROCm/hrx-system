// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef HRX_CTS_CORE_AMDGPU_TEST_DATA_HPP_
#define HRX_CTS_CORE_AMDGPU_TEST_DATA_HPP_

#include <cstddef>
#include <string>
#include <utility>

#include "build_tools/amdgpu/target_map.h"

namespace hrx_cts {

// One embedded fixture compatible with an AMDGPU architecture.
template <typename File>
struct AmdgpuTestImage {
  // Embedded file containing the code-object bytes.
  const File* file = nullptr;
  // Canonical AMDGPU target key implemented by the code object.
  std::string target_key;
};

template <typename File>
inline const File* FindAmdgpuTestImageForTarget(
    const std::string& target_key, const std::string& filename_prefix,
    const File* files, size_t file_count) {
  char fragment[64] = {};
  if (!iree_amdgpu_target_label_fragment(target_key.c_str(), fragment,
                                         sizeof(fragment))) {
    return nullptr;
  }
  const std::string filename = filename_prefix + fragment + ".so";
  for (size_t i = 0; i < file_count; ++i) {
    if (filename == files[i].name) return &files[i];
  }
  return nullptr;
}

// Finds the exact or generic code object selected for |architecture|.
template <typename File>
inline AmdgpuTestImage<File> FindAmdgpuTestImage(
    std::string architecture, const std::string& filename_prefix,
    const File* files, size_t file_count) {
  const size_t feature_position = architecture.find(':');
  if (feature_position != std::string::npos) {
    architecture.resize(feature_position);
  }
  if (const File* file = FindAmdgpuTestImageForTarget(
          architecture, filename_prefix, files, file_count)) {
    return AmdgpuTestImage<File>{file, architecture};
  }
  const char* code_object_target =
      iree_amdgpu_code_object_target_for_exact(architecture.c_str());
  if (code_object_target && architecture != code_object_target) {
    if (const File* file = FindAmdgpuTestImageForTarget(
            code_object_target, filename_prefix, files, file_count)) {
      return AmdgpuTestImage<File>{file, code_object_target};
    }
  }
  return {};
}

}  // namespace hrx_cts

#endif  // HRX_CTS_CORE_AMDGPU_TEST_DATA_HPP_
