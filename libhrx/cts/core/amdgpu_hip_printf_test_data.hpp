// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef HRX_CTS_CORE_AMDGPU_HIP_PRINTF_TEST_DATA_HPP_
#define HRX_CTS_CORE_AMDGPU_HIP_PRINTF_TEST_DATA_HPP_

#include "libhrx/cts/amdgpu_hip_printf_test_kernels.h"
#include "libhrx/cts/core/amdgpu_test_data.hpp"

namespace hrx_cts {

using AmdgpuHipPrintfTestImage = AmdgpuTestImage<iree_file_toc_t>;

inline AmdgpuHipPrintfTestImage FindAmdgpuHipPrintfTestImage(
    std::string architecture) {
  const iree_file_toc_t* toc = hrx_cts_amdgpu_hip_printf_test_kernels_create();
  return FindAmdgpuTestImage(std::move(architecture),
                             "hrx_cts_hip_printf_kernel_", toc,
                             hrx_cts_amdgpu_hip_printf_test_kernels_size());
}

}  // namespace hrx_cts

#endif  // HRX_CTS_CORE_AMDGPU_HIP_PRINTF_TEST_DATA_HPP_
