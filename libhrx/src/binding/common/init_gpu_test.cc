// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "common/device.h"
#include "common/init.h"
#include "hrx_internal.h"
#include "hrx_runtime_cxx.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct FailingFrontendAllocator {
  // Fail before activation, or at the first allocation after group creation.
  bool after_activation = false;
  // Number of allocations refused at the selected construction boundary.
  size_t failure_count = 0;

  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* params, void** inout_pointer) {
    auto* allocator = static_cast<FailingFrontendAllocator*>(self);
    auto* registry = iree_hal_streaming_device_registry();
    if (command != IREE_ALLOCATOR_COMMAND_FREE &&
        (!allocator->after_activation || (registry && registry->group))) {
      ++allocator->failure_count;
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "caller cannot allocate frontend metadata");
    }
    const auto system_allocator = iree_allocator_system();
    return system_allocator.ctl(system_allocator.self, command, params,
                                inout_pointer);
  }
};

class StreamingInitializationTest : public testing::Test {
 protected:
  void SetUp() override {
    const hrx_runtime_options_t options = {
        .event_sink =
            {
                .fn = RecordEvent,
                .user_data = &native_event_count_,
            },
    };
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_runtime_create(
        &options, hrx_host_allocator_system(), runtime_.for_output())));
    const hrx_device_create_params_t params = {.uri = "task://"};
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_device_group_create(runtime_, 1, &params, group_.for_output())));
  }

  void TearDown() override { iree_hal_streaming_cleanup_global(); }

  static void RecordEvent(void* self, const hrx_device_event_t* event) {
    if (event->type == HRX_DEVICE_EVENT_TYPE_USER) {
      ++*static_cast<size_t*>(self);
    }
  }

  static iree_status_t SelectFirstDevice(void* self,
                                         iree_string_view_t driver_name,
                                         iree_allocator_t host_allocator,
                                         iree_host_size_t* device_count,
                                         iree_hal_device_info_t* device_infos) {
    ++*static_cast<size_t*>(self);
    *device_count = *device_count ? 1 : 0;
    return iree_ok_status();
  }

  // Native owners survive every frontend construction and teardown below.
  hrx::runtime::runtime_ptr runtime_;
  // Independent task device and its execution resources.
  hrx::runtime::device_group_ptr group_;
  // Application events delivered only through the native runtime's sink.
  size_t native_event_count_ = 0;
  // Compatibility events delivered only through the frontend's sink.
  size_t frontend_event_count_ = 0;
};

TEST_F(StreamingInitializationTest, FailureAndTeardownPreserveNativeRuntime) {
  hrx_device_t native_device = hrx_device_group_get_device(group_, 0);
  auto* native_proactors = hrx_device_proactor_pool(native_device);
  size_t selection_count = 0;
  const iree_hal_streaming_init_options_t options = {
      .runtime =
          {
              .event_sink =
                  {
                      .fn = RecordEvent,
                      .user_data = &frontend_event_count_,
                  },
          },
      .devices =
          {
              .device_selector =
                  {
                      .self = &selection_count,
                      .select = SelectFirstDevice,
                  },
          },
  };
  {
    FailingFrontendAllocator allocator;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        iree_hal_streaming_init_global(
            &options,
            iree_allocator_t{&allocator, FailingFrontendAllocator::Control}));
    EXPECT_EQ(allocator.failure_count, 1u);
    EXPECT_EQ(selection_count, 0u);
    EXPECT_EQ(iree_hal_streaming_device_registry(), nullptr);
  }
  {
    // Native activation succeeds, then compatibility metadata allocation fails.
    // Cleanup must finish before the failed caller's allocator goes away.
    FailingFrontendAllocator allocator = {.after_activation = true};
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        iree_hal_streaming_init_global(
            &options,
            iree_allocator_t{&allocator, FailingFrontendAllocator::Control}));
    EXPECT_EQ(allocator.failure_count, 1u);
    EXPECT_EQ(selection_count, 1u);
    EXPECT_EQ(iree_hal_streaming_device_registry(), nullptr);
  }

  // Retry supplies a new allocator and owns its own successful GPU lifetime.
  IREE_ASSERT_OK(
      iree_hal_streaming_init_global(&options, iree_allocator_system()));
  auto* registry = iree_hal_streaming_device_registry();
  ASSERT_NE(registry, nullptr);
  EXPECT_EQ(selection_count, 2u);
  EXPECT_EQ(registry->device_count, 1u);
  EXPECT_NE(registry->runtime, runtime_.get());
  EXPECT_NE(registry->group, group_.get());
  EXPECT_NE(hrx_device_proactor_pool(registry->devices[0].hrx_device),
            native_proactors);
  EXPECT_EQ(hrx_device_get_device_group(registry->devices[0].hrx_device),
            registry->group);

  iree_hal_device_event_t event = iree_hal_device_event_default();
  event.type = IREE_HAL_DEVICE_EVENT_TYPE_USER;
  iree_hal_device_event_sink_publish(
      hrx_device_hal_event_sink(registry->devices[0].hrx_device), &event);
  EXPECT_EQ(frontend_event_count_, 1u);
  EXPECT_EQ(native_event_count_, 0u);
  iree_hal_streaming_cleanup_global();
  EXPECT_EQ(iree_hal_streaming_device_registry(), nullptr);
  EXPECT_EQ(hrx_device_proactor_pool(native_device), native_proactors);
  iree_hal_device_event_sink_publish(hrx_device_hal_event_sink(native_device),
                                     &event);
  EXPECT_EQ(frontend_event_count_, 1u);
  EXPECT_EQ(native_event_count_, 1u);

  // Native execution and host-visible results remain usable after HIP teardown.
  hrx::runtime::buffer_ptr buffer;
  const hrx_buffer_params_t buffer_params = {
      .type = HRX_MEMORY_TYPE_HOST_LOCAL,
      .access = HRX_MEMORY_ACCESS_ALL,
      .usage = HRX_BUFFER_USAGE_DEFAULT,
  };
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_allocator_allocate_buffer(hrx_device_allocator(native_device),
                                    buffer_params, 256, buffer.for_output())));
  hrx::runtime::semaphore_ptr semaphore;
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_semaphore_create(native_device, 0, semaphore.for_output())));
  hrx_semaphore_t signal = semaphore;
  uint64_t value = 1;
  const hrx_semaphore_list_t signals = {&signal, &value, 1};
  const uint32_t pattern = 0x12345678;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_queue_fill(
      hrx_device_get_queue(native_device, 0, 0), nullptr, &signals, buffer, 0,
      256, &pattern, sizeof(pattern), nullptr)));
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_semaphore_wait(semaphore, value, UINT64_MAX)));
  void* data = nullptr;
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_buffer_map(buffer, HRX_MAP_READ, 0, 256, &data)));
  for (size_t i = 0; i < 64; ++i) {
    EXPECT_EQ(static_cast<uint32_t*>(data)[i], pattern);
  }
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_buffer_unmap(buffer)));
}

}  // namespace
