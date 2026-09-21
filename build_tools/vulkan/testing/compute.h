// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILD_TOOLS_VULKAN_TESTING_COMPUTE_H_
#define IREE_BUILD_TOOLS_VULKAN_TESTING_COMPUTE_H_

#include <cstddef>

#include "build_tools/vulkan/testing/context.h"

namespace iree::vulkan::testing {

// Owns a descriptor-free compute pipeline. The caller supplies SPIR-V and its
// push-constant ABI; this helper performs no reflection or argument packing.
class Compute {
 public:
  Compute() = default;
  ~Compute();
  Compute(const Compute&) = delete;
  Compute& operator=(const Compute&) = delete;

  ::testing::AssertionResult Initialize(
      const Device& device, size_t code_size, const uint32_t* code,
      const char* entry_point, uint32_t push_constant_range_count,
      const VkPushConstantRange* push_constant_ranges,
      const VkSpecializationInfo* specialization = nullptr);
  VkPipeline handle() const { return pipeline_; }
  VkPipelineLayout layout() const { return layout_; }

 private:
  // Borrowed parent, alive through pipeline teardown.
  const Device* device_ = nullptr;
  // Owned layout with zero descriptor sets and explicit push-constant ranges.
  VkPipelineLayout layout_ = VK_NULL_HANDLE;
  // Owned pipeline, destroyed before its layout.
  VkPipeline pipeline_ = VK_NULL_HANDLE;
};

}  // namespace iree::vulkan::testing

#endif  // IREE_BUILD_TOOLS_VULKAN_TESTING_COMPUTE_H_
