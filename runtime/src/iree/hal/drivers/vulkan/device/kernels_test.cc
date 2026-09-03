// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/device/kernels.h"

#include <cstddef>
#include <cstring>
#include <string>

#include "iree/hal/drivers/vulkan/device/kernels_generated.h"
#include "iree/testing/gtest.h"

namespace {

const iree_file_toc_t* FindFile(const iree_file_toc_t* toc, size_t count,
                                const char* file_name) {
  for (size_t i = 0; i < count; ++i) {
    if (std::strcmp(toc[i].name, file_name) == 0) return &toc[i];
  }
  return nullptr;
}

TEST(KernelsTest, CheckedModulesMatchCanonicalAssembly) {
  const iree_file_toc_t* checked_toc = iree_hal_vulkan_device_kernels_create();
  const iree_file_toc_t* generated_toc =
      iree_hal_vulkan_device_kernels_generated_create();
  const char* const file_names[] = {"atomic_32", "atomic_64", "copy_unaligned",
                                    "fill_unaligned", "update_unaligned"};
  for (const char* file_name : file_names) {
    const std::string checked_name = std::string(file_name) + ".spv";
    const std::string generated_name =
        std::string(file_name) + "_generated.spv";
    const iree_file_toc_t* checked =
        FindFile(checked_toc, iree_hal_vulkan_device_kernels_size(),
                 checked_name.c_str());
    const iree_file_toc_t* generated =
        FindFile(generated_toc, iree_hal_vulkan_device_kernels_generated_size(),
                 generated_name.c_str());
    ASSERT_NE(checked, nullptr) << checked_name;
    ASSERT_NE(generated, nullptr) << generated_name;
    ASSERT_EQ(checked->size, generated->size) << file_name;
    EXPECT_EQ(std::memcmp(checked->data, generated->data, checked->size), 0)
        << file_name;
  }
}

}  // namespace
