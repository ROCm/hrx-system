// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/semaphore.h"

#include <cstring>

#include "iree/async/proactor.h"
#include "iree/async/proactor_platform.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class SemaphoreTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &proactor_));
  }

  void TearDown() override { iree_async_proactor_release(proactor_); }

  iree_hal_semaphore_t* CreateSemaphore(uint64_t initial_value) {
    iree_hal_semaphore_t* semaphore = nullptr;
    IREE_CHECK_OK(iree_hal_amdxdna_semaphore_create(
        proactor_, IREE_HAL_QUEUE_AFFINITY_ANY, initial_value,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, iree_allocator_system(), &semaphore));
    return semaphore;
  }

  iree_async_proactor_t* proactor_ = nullptr;
};

TEST_F(SemaphoreTest, QuerySignalAndWait) {
  iree_hal_semaphore_t* semaphore = CreateSemaphore(3);

  uint64_t value = 0;
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphore, &value));
  EXPECT_EQ(value, 3ull);

  IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphore, 7, /*frontier=*/nullptr));
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphore, &value));
  EXPECT_EQ(value, 7ull);
  IREE_ASSERT_OK(iree_hal_semaphore_wait(semaphore, 7, iree_immediate_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_semaphore_release(semaphore);
}

TEST_F(SemaphoreTest, RejectsDuplicateAndLowerSignals) {
  iree_hal_semaphore_t* semaphore = CreateSemaphore(3);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphore, 7, /*frontier=*/nullptr));

  iree_status_t status =
      iree_hal_semaphore_signal(semaphore, 7, /*frontier=*/nullptr);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_free(status);

  status = iree_hal_semaphore_signal(semaphore, 6, /*frontier=*/nullptr);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_free(status);

  iree_hal_semaphore_release(semaphore);
}

TEST_F(SemaphoreTest, NativeTimepointImportExportUnimplemented) {
  iree_hal_semaphore_t* semaphore = CreateSemaphore(0);

  iree_hal_external_timepoint_t external_timepoint;
  memset(&external_timepoint, 0, sizeof(external_timepoint));
  iree_status_t status = iree_hal_semaphore_import_timepoint(
      semaphore, 1, IREE_HAL_QUEUE_AFFINITY_ANY, external_timepoint);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_UNIMPLEMENTED);
  iree_status_free(status);

  iree_hal_external_timepoint_t exported_timepoint;
  status = iree_hal_semaphore_export_timepoint(
      semaphore, 1, IREE_HAL_QUEUE_AFFINITY_ANY,
      IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_NONE,
      IREE_HAL_EXTERNAL_TIMEPOINT_FLAG_NONE, &exported_timepoint);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_UNIMPLEMENTED);
  iree_status_free(status);
  EXPECT_EQ(exported_timepoint.type, IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_NONE);

  iree_hal_semaphore_release(semaphore);
}

}  // namespace
