// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/aql/queue.h"

#include "iree/testing/gtest.h"

namespace {

amdf_queue_family_info_t CompleteFamily() {
  return {
      .command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_AQL,
      .publication_modes = AMDF_QUEUE_PUBLICATION_MODE_USER,
      .format_version = AMDF_GPU_AQL_QUEUE_FORMAT_VERSION_1,
      .format_features = AMDF_GPU_AQL_FORMAT_FEATURE_SCRATCH_SERVICE,
      .roles = AMDF_QUEUE_ROLE_COMPUTE,
      .user_queue_capabilities = AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER |
                                 AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER,
  };
}

TEST(QueueFamilyTest, RequiresDevicePublicationAndNativeScratch) {
  for (uint32_t missing = 0; missing < 7; ++missing) {
    SCOPED_TRACE(missing);
    auto family = CompleteFamily();
    switch (missing) {
      case 0:
        family.command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_PM4;
        break;
      case 1:
        family.format_version = 0;
        break;
      case 2:
        family.format_features = 0;
        break;
      case 3:
        family.roles = AMDF_QUEUE_ROLE_TRANSFER;
        break;
      case 4:
        family.publication_modes = AMDF_QUEUE_PUBLICATION_MODE_KERNEL;
        break;
      case 5:
        family.user_queue_capabilities =
            AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER;
        break;
      case 6:
        family.user_queue_capabilities =
            AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER;
        break;
    }
    EXPECT_EQ(iree_hal_amd_gpu_aql_queue_publication_modes(&family), 0u);
  }
}

TEST(QueueFamilyTest, SelectsOnlyTheSupportedPublicationMode) {
  auto family = CompleteFamily();
  EXPECT_EQ(iree_hal_amd_gpu_aql_queue_publication_modes(&family),
            AMDF_QUEUE_PUBLICATION_MODE_USER);
  family.publication_modes |= AMDF_QUEUE_PUBLICATION_MODE_KERNEL;
  EXPECT_EQ(iree_hal_amd_gpu_aql_queue_publication_modes(&family),
            AMDF_QUEUE_PUBLICATION_MODE_USER);
}

}  // namespace
