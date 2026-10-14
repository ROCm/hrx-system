// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {
namespace {

class AmdgpuSemaphoreTest : public CtsTestBase<> {};

TEST_P(AmdgpuSemaphoreTest, HostProgressRequiresSystemPublication) {
  // Host network progress must be able to reject private device edges even
  // though their logical timeline can be queried from the CPU. Completion of
  // an AGENT-scope edge does not publish the producer's payload to the host.
  Ref<iree_hal_semaphore_t> private_edge;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device_, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEVICE_LOCAL, private_edge.out()));
  EXPECT_EQ(
      iree_hal_device_query_semaphore_compatibility(device_, private_edge),
      IREE_HAL_SEMAPHORE_COMPATIBILITY_DEVICE_ONLY);

  const iree_hal_semaphore_flags_t publication_flags[] = {
      IREE_HAL_SEMAPHORE_FLAG_NONE,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT,
      IREE_HAL_SEMAPHORE_FLAG_DEVICE_LOCAL |
          IREE_HAL_SEMAPHORE_FLAG_HOST_INTERRUPT,
      IREE_HAL_SEMAPHORE_FLAG_DEVICE_LOCAL | IREE_HAL_SEMAPHORE_FLAG_EXPORTABLE,
      IREE_HAL_SEMAPHORE_FLAG_DEVICE_LOCAL |
          IREE_HAL_SEMAPHORE_FLAG_EXPORTABLE_TIMEPOINTS,
  };
  for (iree_hal_semaphore_flags_t flags : publication_flags) {
    SCOPED_TRACE(flags);
    Ref<iree_hal_semaphore_t> public_edge;
    IREE_ASSERT_OK(iree_hal_semaphore_create(device_,
                                             IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
                                             0, flags, public_edge.out()));
    EXPECT_EQ(
        iree_hal_device_query_semaphore_compatibility(device_, public_edge),
        IREE_HAL_SEMAPHORE_COMPATIBILITY_ALL);
  }
}

CTS_REGISTER_TEST_SUITE(AmdgpuSemaphoreTest);

}  // namespace
}  // namespace iree::hal::cts
