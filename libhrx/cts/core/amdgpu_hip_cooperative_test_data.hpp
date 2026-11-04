// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef HRX_CTS_CORE_AMDGPU_HIP_COOPERATIVE_TEST_DATA_HPP_
#define HRX_CTS_CORE_AMDGPU_HIP_COOPERATIVE_TEST_DATA_HPP_

#include "iree/hal/drivers/amd/gpu/image/cooperative_images.h"
#include "libhrx/cts/core/amdgpu_test_data.hpp"

namespace hrx_cts {

using AmdgpuHipCooperativeTestImage = AmdgpuTestImage<iree_file_toc_t>;

inline AmdgpuHipCooperativeTestImage FindAmdgpuHipCooperativeTestImage(
    std::string architecture) {
  const iree_file_toc_t* toc = iree_hal_amd_gpu_cooperative_images_create();
  return FindAmdgpuTestImage(std::move(architecture), "amd_gpu_cooperative_",
                             toc, iree_hal_amd_gpu_cooperative_images_size());
}

}  // namespace hrx_cts

#endif  // HRX_CTS_CORE_AMDGPU_HIP_COOPERATIVE_TEST_DATA_HPP_
