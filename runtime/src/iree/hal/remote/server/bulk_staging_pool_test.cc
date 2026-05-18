// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/server/bulk_staging_pool.h"

#include <cstddef>
#include <cstdint>

#include "iree/async/util/proactor_pool.h"
#include "iree/base/threading/numa.h"
#include "iree/hal/drivers/local_task/registration/driver_module.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

typedef struct test_slot_storage_t {
  // Test marker value preserved while the slot is acquired.
  uint64_t marker;
} test_slot_storage_t;

iree_status_t RegisterLocalTaskDriver() {
  iree_status_t status = iree_hal_local_task_driver_module_register(
      iree_hal_driver_registry_default());
  if (iree_status_is_already_exists(status)) {
    iree_status_free(status);
    status = iree_ok_status();
  }
  return status;
}

class BulkStagingPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(RegisterLocalTaskDriver());
    IREE_ASSERT_OK(iree_hal_driver_registry_try_create(
        iree_hal_driver_registry_default(),
        iree_make_cstring_view("local-task"), iree_allocator_system(),
        &driver_));
    IREE_ASSERT_OK(CreateDevice(&device_));
  }

  void TearDown() override {
    iree_hal_device_release(device_);
    iree_hal_driver_release(driver_);
  }

  iree_status_t CreateDevice(iree_hal_device_t** out_device) {
    *out_device = nullptr;
    iree_async_proactor_pool_t* proactor_pool = nullptr;
    iree_status_t status = iree_async_proactor_pool_create(
        iree_numa_node_count(), /*node_ids=*/nullptr,
        iree_async_proactor_pool_options_default(), iree_allocator_system(),
        &proactor_pool);
    iree_hal_device_t* device = nullptr;
    if (iree_status_is_ok(status)) {
      iree_hal_device_create_params_t create_params =
          iree_hal_device_create_params_default();
      create_params.proactor_pool = proactor_pool;
      status = iree_hal_driver_create_default_device(
          driver_, &create_params, iree_allocator_system(), &device);
    }
    iree_async_proactor_pool_release(proactor_pool);
    if (iree_status_is_ok(status)) {
      *out_device = device;
    } else {
      iree_hal_device_release(device);
    }
    return status;
  }

  iree_status_t AllocatePool(
      iree_host_size_t slot_count,
      iree_hal_remote_server_bulk_staging_pool_t** out_pool) {
    iree_hal_remote_server_bulk_staging_pool_options_t options =
        iree_hal_remote_server_bulk_staging_pool_options_default();
    options.slot_count = slot_count;
    options.slot_length = 256;
    options.user_storage_size = sizeof(test_slot_storage_t);
    options.user_storage_alignment = alignof(test_slot_storage_t);
    return iree_hal_remote_server_bulk_staging_pool_create(
        &options, iree_allocator_system(), out_pool);
  }

  iree_hal_driver_t* driver_ = nullptr;
  iree_hal_device_t* device_ = nullptr;
};

TEST_F(BulkStagingPoolTest, AcquiresAndReusesFixedSlots) {
  iree_hal_remote_server_bulk_staging_pool_t* pool = nullptr;
  IREE_ASSERT_OK(AllocatePool(/*slot_count=*/2, &pool));

  EXPECT_EQ(iree_hal_remote_server_bulk_staging_pool_capacity(pool), 2);
  EXPECT_EQ(iree_hal_remote_server_bulk_staging_pool_count(pool), 0);

  iree_hal_remote_server_bulk_staging_slot_t* first_slot = nullptr;
  IREE_ASSERT_OK(iree_hal_remote_server_bulk_staging_pool_try_acquire(
      pool, device_, &first_slot));
  ASSERT_NE(first_slot, nullptr);
  EXPECT_EQ(iree_hal_remote_server_bulk_staging_pool_count(pool), 1);
  EXPECT_EQ(iree_hal_remote_server_bulk_staging_slot_device(first_slot),
            device_);
  EXPECT_NE(iree_hal_remote_server_bulk_staging_slot_file(first_slot), nullptr);
  EXPECT_NE(iree_hal_remote_server_bulk_staging_slot_semaphore(first_slot),
            nullptr);

  iree_byte_span_t contents =
      iree_hal_remote_server_bulk_staging_slot_contents(first_slot);
  ASSERT_EQ(contents.data_length, 256);
  contents.data[0] = 0xCD;
  iree_byte_span_t user_storage =
      iree_hal_remote_server_bulk_staging_slot_user_storage(first_slot);
  ASSERT_EQ(user_storage.data_length, sizeof(test_slot_storage_t));
  auto* storage = reinterpret_cast<test_slot_storage_t*>(user_storage.data);
  storage->marker = 0xABCD1234u;

  iree_hal_remote_server_bulk_staging_slot_t* second_slot = nullptr;
  IREE_ASSERT_OK(iree_hal_remote_server_bulk_staging_pool_try_acquire(
      pool, device_, &second_slot));
  ASSERT_NE(second_slot, nullptr);
  EXPECT_EQ(iree_hal_remote_server_bulk_staging_pool_count(pool), 2);

  iree_hal_remote_server_bulk_staging_slot_t* exhausted_slot = nullptr;
  IREE_ASSERT_OK(iree_hal_remote_server_bulk_staging_pool_try_acquire(
      pool, device_, &exhausted_slot));
  EXPECT_EQ(exhausted_slot, nullptr);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_hal_remote_server_bulk_staging_pool_acquire(
                            pool, device_, &exhausted_slot));

  iree_hal_remote_server_bulk_staging_slot_release(first_slot,
                                                   /*last_signal_value=*/7);
  EXPECT_EQ(iree_hal_remote_server_bulk_staging_pool_count(pool), 1);

  iree_hal_remote_server_bulk_staging_slot_t* reused_slot = nullptr;
  IREE_ASSERT_OK(iree_hal_remote_server_bulk_staging_pool_acquire(
      pool, device_, &reused_slot));
  ASSERT_EQ(reused_slot, first_slot);
  EXPECT_EQ(
      iree_hal_remote_server_bulk_staging_slot_last_signal_value(reused_slot),
      7);
  EXPECT_EQ(
      iree_hal_remote_server_bulk_staging_slot_next_signal_value(reused_slot),
      8);

  iree_hal_remote_server_bulk_staging_slot_release(second_slot,
                                                   /*last_signal_value=*/3);
  iree_hal_remote_server_bulk_staging_slot_release(reused_slot,
                                                   /*last_signal_value=*/8);
  EXPECT_EQ(iree_hal_remote_server_bulk_staging_pool_count(pool), 0);
  iree_hal_remote_server_bulk_staging_pool_release(pool);
}

TEST_F(BulkStagingPoolTest, RebindsSlotsWhenDeviceChanges) {
  iree_hal_device_t* second_device = nullptr;
  IREE_ASSERT_OK(CreateDevice(&second_device));

  iree_hal_remote_server_bulk_staging_pool_t* pool = nullptr;
  IREE_ASSERT_OK(AllocatePool(/*slot_count=*/1, &pool));

  iree_hal_remote_server_bulk_staging_slot_t* slot = nullptr;
  IREE_ASSERT_OK(
      iree_hal_remote_server_bulk_staging_pool_acquire(pool, device_, &slot));
  ASSERT_NE(slot, nullptr);
  EXPECT_EQ(iree_hal_remote_server_bulk_staging_slot_device(slot), device_);
  iree_hal_remote_server_bulk_staging_slot_release(slot,
                                                   /*last_signal_value=*/9);

  slot = nullptr;
  IREE_ASSERT_OK(iree_hal_remote_server_bulk_staging_pool_acquire(
      pool, second_device, &slot));
  ASSERT_NE(slot, nullptr);
  EXPECT_EQ(iree_hal_remote_server_bulk_staging_slot_device(slot),
            second_device);
  EXPECT_EQ(iree_hal_remote_server_bulk_staging_slot_last_signal_value(slot),
            0);

  iree_hal_remote_server_bulk_staging_slot_release(slot,
                                                   /*last_signal_value=*/1);
  iree_hal_remote_server_bulk_staging_pool_release(pool);
  iree_hal_device_release(second_device);
}

typedef struct callback_state_t {
  // True once the staging callback has run.
  bool callback_called;

  // Signal value expected by the staging callback.
  uint64_t expected_signal_value;
} callback_state_t;

static void ReleaseSlotCallback(
    void* user_data, iree_hal_remote_server_bulk_staging_slot_t* slot,
    uint64_t signal_value, iree_status_t status) {
  auto* state = reinterpret_cast<callback_state_t*>(user_data);
  IREE_EXPECT_OK(status);
  EXPECT_EQ(signal_value, state->expected_signal_value);
  EXPECT_EQ(iree_hal_remote_server_bulk_staging_pool_count(
                iree_hal_remote_server_bulk_staging_slot_pool(slot)),
            1);
  state->callback_called = true;
  iree_hal_remote_server_bulk_staging_slot_release(slot, signal_value);
}

TEST_F(BulkStagingPoolTest, TimepointCallbackRetainsPoolUntilFired) {
  iree_hal_remote_server_bulk_staging_pool_t* pool = nullptr;
  IREE_ASSERT_OK(AllocatePool(/*slot_count=*/1, &pool));

  iree_hal_remote_server_bulk_staging_slot_t* slot = nullptr;
  IREE_ASSERT_OK(
      iree_hal_remote_server_bulk_staging_pool_acquire(pool, device_, &slot));
  ASSERT_NE(slot, nullptr);

  iree_hal_semaphore_t* semaphore =
      iree_hal_remote_server_bulk_staging_slot_semaphore(slot);
  iree_hal_semaphore_retain(semaphore);

  callback_state_t callback_state = {};
  callback_state.callback_called = false;
  callback_state.expected_signal_value = 1;
  IREE_ASSERT_OK(iree_hal_remote_server_bulk_staging_slot_acquire_timepoint(
      slot, callback_state.expected_signal_value, ReleaseSlotCallback,
      &callback_state));

  iree_hal_remote_server_bulk_staging_pool_release(pool);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(
      semaphore, callback_state.expected_signal_value, /*frontier=*/nullptr));
  EXPECT_TRUE(callback_state.callback_called);

  iree_hal_semaphore_release(semaphore);
}

}  // namespace
