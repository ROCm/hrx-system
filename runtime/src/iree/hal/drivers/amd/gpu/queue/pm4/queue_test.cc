// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/queue.h"

#include "iree/testing/gtest.h"

namespace {

amdf_queue_family_info_t CompleteFamily() {
  return {
      .command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
      .format_version = AMDF_GPU_PM4_QUEUE_FORMAT_VERSION_1,
      .format_features = AMDF_GPU_PM4_FORMAT_FEATURE_ACQUIRE_MEM_GCR,
      .roles = AMDF_QUEUE_ROLE_COMPUTE | AMDF_QUEUE_ROLE_TRANSFER |
               AMDF_QUEUE_ROLE_ATOMIC | AMDF_QUEUE_ROLE_CACHE_CONTROL,
      .cache_operations = AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM |
                          AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM,
      .cache_transition_kinds = AMDF_CACHE_TRANSITION_KINDS_GLOBAL,
      .atomic_capabilities =
          {
              .wait_conditions_32 = AMDF_ATOMIC_WAIT_CONDITION_EQUAL,
              .operations_without_dispatch_32 =
                  AMDF_ATOMIC_OPERATION_STORE | AMDF_ATOMIC_OPERATION_WAIT,
          },
  };
}

TEST(QueueFamilyTest,
     PublicationComesFromProtocolAndNativeProducerCapabilities) {
  auto family = CompleteFamily();
  family.publication_modes = AMDF_QUEUE_PUBLICATION_MODE_KERNEL;
  EXPECT_EQ(iree_hal_amd_gpu_pm4_queue_publication_modes(&family),
            AMDF_QUEUE_PUBLICATION_MODE_KERNEL);
  family.publication_modes |= AMDF_QUEUE_PUBLICATION_MODE_USER;
  EXPECT_EQ(iree_hal_amd_gpu_pm4_queue_publication_modes(&family),
            AMDF_QUEUE_PUBLICATION_MODE_KERNEL);
  family.user_queue_capabilities = AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER;
  EXPECT_EQ(iree_hal_amd_gpu_pm4_queue_publication_modes(&family),
            family.publication_modes);
  family.publication_modes = AMDF_QUEUE_PUBLICATION_MODE_USER;
  EXPECT_EQ(iree_hal_amd_gpu_pm4_queue_publication_modes(&family),
            AMDF_QUEUE_PUBLICATION_MODE_USER);
  family.user_queue_capabilities = 0;
  EXPECT_EQ(iree_hal_amd_gpu_pm4_queue_publication_modes(&family), 0u);
}

TEST(QueueFamilyTest, AutomaticSelectionRequiresTheCompleteExecutionContract) {
  for (uint32_t missing = 0; missing < 11; ++missing) {
    SCOPED_TRACE(missing);
    auto family = CompleteFamily();
    family.publication_modes = AMDF_QUEUE_PUBLICATION_MODE_KERNEL;
    switch (missing) {
      case 0:
        family.command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_AQL;
        break;
      case 1:
        family.format_version = 0;
        break;
      case 2:
        family.roles &= ~AMDF_QUEUE_ROLE_COMPUTE;
        break;
      case 3:
        family.roles &= ~AMDF_QUEUE_ROLE_TRANSFER;
        break;
      case 4:
        family.roles &= ~AMDF_QUEUE_ROLE_ATOMIC;
        break;
      case 5:
        family.roles &= ~AMDF_QUEUE_ROLE_CACHE_CONTROL;
        break;
      case 6:
        family.format_features = 0;
        break;
      case 7:
        family.cache_operations = AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM;
        break;
      case 8:
        family.cache_transition_kinds = 0;
        break;
      case 9:
        family.atomic_capabilities.operations_without_dispatch_32 =
            AMDF_ATOMIC_OPERATION_STORE;
        break;
      case 10:
        family.atomic_capabilities.wait_conditions_32 = 0;
        break;
    }
    EXPECT_EQ(iree_hal_amd_gpu_pm4_queue_publication_modes(&family), 0u);
  }
}

}  // namespace
