// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILD_TOOLS_VULKAN_TESTING_SUBMISSION_H_
#define IREE_BUILD_TOOLS_VULKAN_TESTING_SUBMISSION_H_

#include "build_tools/vulkan/testing/context.h"

namespace iree::vulkan::testing {

// Owns reusable command storage and a completion fence for one explicit queue.
// The caller records commands and barriers between Begin and SubmitAndWait.
// Each successful submission completes before command storage can be reused.
class Submission {
 public:
  Submission() = default;
  ~Submission();
  Submission(const Submission&) = delete;
  Submission& operator=(const Submission&) = delete;

  ::testing::AssertionResult Initialize(const Device& device,
                                        uint32_t queue_family_index,
                                        uint32_t queue_index);
  ::testing::AssertionResult Begin();
  ::testing::AssertionResult SubmitAndWait();
  VkCommandBuffer command_buffer() const { return command_buffer_; }

 private:
  // Borrowed parent, alive through command storage teardown.
  const Device* device_ = nullptr;
  // Borrowed native queue; callers serialize any other host access to it.
  VkQueue queue_ = VK_NULL_HANDLE;
  // Owned pool and backing command storage.
  VkCommandPool pool_ = VK_NULL_HANDLE;
  // Primary command buffer owned by pool_.
  VkCommandBuffer command_buffer_ = VK_NULL_HANDLE;
  // Owned fence reset before each submission.
  VkFence fence_ = VK_NULL_HANDLE;
  // An accepted submission whose completion has not been established. An
  // unrecoverable wait failure stops the process before reachable memory frees.
  bool pending_ = false;
};

}  // namespace iree::vulkan::testing

#endif  // IREE_BUILD_TOOLS_VULKAN_TESTING_SUBMISSION_H_
