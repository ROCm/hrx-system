// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "common/device.h"
#include "common/init.h"
#include "hrx_internal.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct RejectingRegistryAllocator {
  // Native devices already published when binding storage is requested.
  int activated_device_count = 0;

  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* params, void** inout_pointer) {
    if (command == IREE_ALLOCATOR_COMMAND_FREE) {
      const auto allocator = iree_allocator_system();
      return allocator.ctl(allocator.self, command, params, inout_pointer);
    }
    auto* allocator = static_cast<RejectingRegistryAllocator*>(self);
    IREE_RETURN_IF_ERROR(hrx_status_to_iree(
        hrx_gpu_device_count(&allocator->activated_device_count)));
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "caller cannot allocate the binding registry");
  }
};

class StreamingInitializationTest : public testing::Test {
 protected:
  void SetUp() override {
    hrx_status_t status = hrx_cpu_initialize(0);
    cpu_initialized_ = hrx_status_is_ok(status);
    IREE_ASSERT_OK(hrx_status_to_iree(status));
  }

  void TearDown() override {
    iree_hal_streaming_cleanup_global();
    if (cpu_initialized_) {
      IREE_EXPECT_OK(hrx_status_to_iree(hrx_cpu_shutdown()));
    }
  }

  static iree_status_t SelectFirstDevice(void* self,
                                         iree_string_view_t driver_name,
                                         iree_allocator_t host_allocator,
                                         iree_host_size_t* device_count,
                                         iree_hal_device_info_t* device_infos) {
    ++*static_cast<int*>(self);
    *device_count = *device_count ? 1 : 0;
    return iree_ok_status();
  }

  // CPU lifetime retained across GPU initialization failures and shutdown.
  bool cpu_initialized_ = false;
};

TEST_F(StreamingInitializationTest,
       NativeActivationRollbackKeepsRetryPossible) {
  auto* proactor_pool = hrx_runtime_proactor_pool();
  ASSERT_NE(proactor_pool, nullptr);
  {
    RejectingRegistryAllocator allocator;
    const hrx_gpu_initialize_options_t options = {};
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        iree_hal_streaming_init_global(
            &options,
            iree_allocator_t{&allocator, RejectingRegistryAllocator::Control}));
    ASSERT_GT(allocator.activated_device_count, 0);
    EXPECT_EQ(iree_hal_streaming_device_registry(), nullptr);
    int count = 0;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                          hrx_status_to_iree(hrx_gpu_device_count(&count)));
    EXPECT_EQ(hrx_runtime_proactor_pool(), proactor_pool);
  }

  // The failed caller's allocator and selection state need not survive retry.
  // Each generation supplies its own callback and releases all device state.
  for (int generation = 0; generation < 3; ++generation) {
    SCOPED_TRACE(generation);
    int selection_count = 0;
    const hrx_gpu_initialize_options_t options = {
        .device_selector =
            {
                .self = &selection_count,
                .select = SelectFirstDevice,
            },
    };
    IREE_ASSERT_OK(
        iree_hal_streaming_init_global(&options, iree_allocator_system()));
    EXPECT_EQ(selection_count, 1);
    int count = 0;
    IREE_EXPECT_OK(hrx_status_to_iree(hrx_gpu_device_count(&count)));
    EXPECT_EQ(count, 1);
    iree_hal_streaming_cleanup_global();
    EXPECT_EQ(iree_hal_streaming_device_registry(), nullptr);
    EXPECT_EQ(hrx_runtime_proactor_pool(), proactor_pool);
    IREE_EXPECT_OK(hrx_status_to_iree(hrx_cpu_device_count(&count)));
    EXPECT_GT(count, 0);
  }
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_cpu_shutdown()));
  cpu_initialized_ = false;
  EXPECT_EQ(hrx_runtime_proactor_pool(), nullptr);
}

}  // namespace
