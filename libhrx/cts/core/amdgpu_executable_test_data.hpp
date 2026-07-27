// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef HRX_CTS_CORE_AMDGPU_EXECUTABLE_TEST_DATA_HPP_
#define HRX_CTS_CORE_AMDGPU_EXECUTABLE_TEST_DATA_HPP_

#include <string>

#include "build_tools/amdgpu/target_map.h"
#include "libhrx/cts/amdgpu_executable_test_kernels.h"

namespace hrx_cts {

// One embedded HSACO fixture compatible with an AMDGPU architecture.
struct AmdgpuExecutableTestImage {
  // Embedded file containing the HSACO bytes.
  const iree_file_toc_t* file = nullptr;
  // Canonical AMDGPU target key implemented by the HSACO.
  std::string target_key;
};

inline const iree_file_toc_t* FindAmdgpuExecutableTestImageForTarget(
    const std::string& target_key) {
  char fragment[64] = {};
  if (!iree_amdgpu_target_label_fragment(target_key.c_str(), fragment,
                                         sizeof(fragment))) {
    return nullptr;
  }
  const std::string filename =
      std::string("hrx_cts_executable_kernel_") + fragment + ".so";
  const iree_file_toc_t* toc = hrx_cts_amdgpu_executable_test_kernels_create();
  for (size_t i = 0; i < hrx_cts_amdgpu_executable_test_kernels_size(); ++i) {
    if (filename == toc[i].name) return &toc[i];
  }
  return nullptr;
}

// Finds the exact or generic code object selected for |architecture|.
inline AmdgpuExecutableTestImage FindAmdgpuExecutableTestImage(
    std::string architecture) {
  const size_t feature_position = architecture.find(':');
  if (feature_position != std::string::npos) {
    architecture.resize(feature_position);
  }
  if (const iree_file_toc_t* file =
          FindAmdgpuExecutableTestImageForTarget(architecture)) {
    return AmdgpuExecutableTestImage{file, architecture};
  }
  const char* code_object_target =
      iree_amdgpu_code_object_target_for_exact(architecture.c_str());
  if (code_object_target && architecture != code_object_target) {
    if (const iree_file_toc_t* file =
            FindAmdgpuExecutableTestImageForTarget(code_object_target)) {
      return AmdgpuExecutableTestImage{file, code_object_target};
    }
  }
  return {};
}

}  // namespace hrx_cts

#endif  // HRX_CTS_CORE_AMDGPU_EXECUTABLE_TEST_DATA_HPP_
