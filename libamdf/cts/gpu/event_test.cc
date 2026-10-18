// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <cstring>
#include <future>
#include <thread>

#include "libamdf/cts/gpu/gpu_device_fixture.h"
#include "libamdf/cts/util/wait.h"

namespace {

amdf_gpu_event_create_info_t EventCreateInfo() {
  return {
      .type = AMDF_STRUCTURE_TYPE_GPU_EVENT_CREATE_INFO,
      .structure_size = sizeof(amdf_gpu_event_create_info_t),
  };
}

class GpuEventAvailabilityTest : public GpuDeviceFixture {};

TEST_F(GpuEventAvailabilityTest, CreationMatchesAdvertisedService) {
  const auto create_info = EventCreateInfo();
  amdf_gpu_event_t* event = nullptr;
  const auto status = gpu_api_->event_create(device_, &create_info, &event);
  if ((features_ & AMDF_GPU_DEVICE_FEATURE_NATIVE_EVENTS) != 0) {
    EXPECT_EQ(status, AMDF_STATUS_OK);
    EXPECT_NE(event, nullptr);
  } else {
    EXPECT_EQ(status, amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED));
    EXPECT_EQ(event, nullptr);
  }
  if (event) {
    EXPECT_EQ(gpu_api_->event_destroy(event), AMDF_STATUS_OK);
  }
}

class GpuEventTest : public GpuDeviceFixture {
 protected:
  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(GpuDeviceFixture::SetUp());
    if (IsSkipped()) {
      return;
    }
    ASSERT_GE(gpu_api_->extension_version, AMDF_GPU_EXTENSION_VERSION_3);
    const auto create_info = EventCreateInfo();
    const auto status = gpu_api_->event_create(device_, &create_info, &event_);
    if ((features_ & AMDF_GPU_DEVICE_FEATURE_NATIVE_EVENTS) == 0) {
      ASSERT_EQ(status, amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED));
      ASSERT_EQ(event_, nullptr);
      GTEST_SKIP() << "device has no qualified native GPU event service";
    }
    ASSERT_EQ(status, AMDF_STATUS_OK) << "domain=" << amdf_status_domain(status)
                                      << " code=" << amdf_status_code(status);
    ASSERT_NE(event_, nullptr);
  }

  void TearDown() override {
    if (event_) {
      EXPECT_EQ(gpu_api_->event_destroy(event_), AMDF_STATUS_OK);
    }
  }

  // Test-owned native event; all producers/waiters join before teardown.
  amdf_gpu_event_t* event_ = nullptr;
};

TEST_F(GpuEventTest, ReportsStableProducerDomainAndOperands) {
  amdf_gpu_event_info_t first = {
      .type = AMDF_STRUCTURE_TYPE_GPU_EVENT_INFO,
      .structure_size = sizeof(first),
  };
  amdf_gpu_event_info_t second = first;
  ASSERT_EQ(gpu_api_->event_query_info(event_, &first), AMDF_STATUS_OK);
  ASSERT_EQ(gpu_api_->event_query_info(event_, &second), AMDF_STATUS_OK);
  EXPECT_EQ(std::memcmp(&first, &second, sizeof(first)), 0);
  amdf_gpu_device_info_t device_info = {
      .type = AMDF_STRUCTURE_TYPE_GPU_DEVICE_INFO,
      .structure_size = sizeof(device_info),
  };
  ASSERT_EQ(gpu_api_->device_query_info(device_, &device_info), AMDF_STATUS_OK);
  EXPECT_TRUE(amdf_device_id_is_equal(&first.device_id, &device_info.id));
  EXPECT_EQ(first.reset_epoch, device_info.reset_epoch);
  EXPECT_NE(first.format, AMDF_GPU_EVENT_FORMAT_NONE);
  EXPECT_NE(first.format_version, 0u);
  EXPECT_NE(first.publication_modes, 0u);
  if (first.format == AMDF_GPU_EVENT_FORMAT_KFD_SIGNAL) {
    EXPECT_EQ(first.format_version, 1u);
    EXPECT_EQ(first.mailbox_address % 8, 0u);
    EXPECT_NE(first.trigger_data, 0u);
    EXPECT_NE(first.interrupt_context, 0u);
  }
}

TEST_F(GpuEventTest, CoalescesEarlyHostSignals) {
  for (uint32_t i = 0; i < 8; ++i) {
    ASSERT_EQ(gpu_api_->event_signal(event_), AMDF_STATUS_OK);
  }
  ASSERT_EQ(gpu_api_->event_wait(event_, amdf::cts::kWaitTimeoutNs),
            AMDF_STATUS_OK);
  EXPECT_EQ(gpu_api_->event_wait(event_, 0),
            amdf_make_api_status(AMDF_STATUS_CODE_DEADLINE_EXCEEDED));
  ASSERT_EQ(gpu_api_->event_signal(event_), AMDF_STATUS_OK);
  EXPECT_EQ(gpu_api_->event_wait(event_, 0), AMDF_STATUS_OK);
}

TEST_F(GpuEventTest, EmptyFiniteWaitExpiresWithoutConsumingLaterSignal) {
  EXPECT_EQ(gpu_api_->event_wait(event_, 1),
            amdf_make_api_status(AMDF_STATUS_CODE_DEADLINE_EXCEEDED));
  ASSERT_EQ(gpu_api_->event_signal(event_), AMDF_STATUS_OK);
  EXPECT_EQ(gpu_api_->event_wait(event_, amdf::cts::kWaitTimeoutNs),
            AMDF_STATUS_OK);
}

TEST_F(GpuEventTest, HostSignalRacesNativeWaitEntry) {
  std::promise<void> started;
  auto ready = started.get_future();
  amdf_status_t wait_status = AMDF_STATUS_OK;
  std::thread waiter([&] {
    started.set_value();
    wait_status = gpu_api_->event_wait(event_, amdf::cts::kWaitTimeoutNs);
  });
  ready.get();
  EXPECT_EQ(gpu_api_->event_signal(event_), AMDF_STATUS_OK);
  waiter.join();
  EXPECT_EQ(wait_status, AMDF_STATUS_OK);
  EXPECT_EQ(gpu_api_->event_wait(event_, 0),
            amdf_make_api_status(AMDF_STATUS_CODE_DEADLINE_EXCEEDED));
}

TEST_F(GpuEventTest, RecreationDoesNotInheritPendingNotification) {
  const auto create_info = EventCreateInfo();
  for (uint32_t i = 0; i < 8; ++i) {
    ASSERT_EQ(gpu_api_->event_signal(event_), AMDF_STATUS_OK);
    ASSERT_EQ(gpu_api_->event_destroy(event_), AMDF_STATUS_OK);
    event_ = nullptr;
    ASSERT_EQ(gpu_api_->event_create(device_, &create_info, &event_),
              AMDF_STATUS_OK);
    EXPECT_EQ(gpu_api_->event_wait(event_, 0),
              amdf_make_api_status(AMDF_STATUS_CODE_DEADLINE_EXCEEDED));
  }
}

TEST_F(GpuEventTest, LiveEventKeepsDeviceBusyWithoutConsumingTheEvent) {
  EXPECT_EQ(api_->device_destroy(device_),
            amdf_make_api_status(AMDF_STATUS_CODE_BUSY));
  ASSERT_EQ(gpu_api_->event_signal(event_), AMDF_STATUS_OK);
  EXPECT_EQ(gpu_api_->event_wait(event_, amdf::cts::kWaitTimeoutNs),
            AMDF_STATUS_OK);
}

TEST_F(GpuEventTest, InvalidArgumentsPreserveOutputs) {
  const auto invalid = amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  auto create_info = EventCreateInfo();
  auto* const sentinel = reinterpret_cast<amdf_gpu_event_t*>(uintptr_t{1});
  amdf_gpu_event_t* output = sentinel;
  EXPECT_EQ(gpu_api_->event_create(nullptr, &create_info, &output), invalid);
  EXPECT_EQ(gpu_api_->event_create(device_, nullptr, &output), invalid);
  EXPECT_EQ(gpu_api_->event_create(device_, &create_info, nullptr), invalid);
  create_info.reserved = 1;
  EXPECT_EQ(gpu_api_->event_create(device_, &create_info, &output), invalid);
  EXPECT_EQ(output, sentinel);
  amdf_gpu_event_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_GPU_EVENT_INFO,
      .structure_size = sizeof(info),
      .trigger_data = 0x12345678,
  };
  const auto original = info;
  EXPECT_EQ(gpu_api_->event_query_info(nullptr, &info), invalid);
  EXPECT_EQ(std::memcmp(&info, &original, sizeof(info)), 0);
  EXPECT_EQ(gpu_api_->event_query_info(event_, nullptr), invalid);
  EXPECT_EQ(gpu_api_->event_signal(nullptr), invalid);
  EXPECT_EQ(gpu_api_->event_wait(nullptr, 0), invalid);
  EXPECT_EQ(gpu_api_->event_destroy(nullptr), invalid);
}

// Complete separately owned instance flow, including its native signal backing.
struct EventContext {
  // API tables borrowed from the loaded CTS provider.
  const amdf_api_t* api;
  // GPU extension borrowed from the same provider.
  const amdf_gpu_api_t* gpu;
  // Independent instance owning its native connection state.
  amdf_instance_t* instance = nullptr;
  // Endpoint owned by this instance, borrowed by device.
  amdf_endpoint_t* endpoint = nullptr;
  // Device borrowed by event.
  amdf_device_t* device = nullptr;
  // Native signal ID owned until the test's host operations end.
  amdf_gpu_event_t* event = nullptr;

  ~EventContext() { Release(); }

  // A failed native release retains every parent that the child still borrows.
  void Release() {
    if (event) {
      ASSERT_EQ(gpu->event_destroy(event), AMDF_STATUS_OK);
      event = nullptr;
    }
    if (device) {
      ASSERT_EQ(api->device_destroy(device), AMDF_STATUS_OK);
      device = nullptr;
    }
    if (endpoint) {
      ASSERT_EQ(api->endpoint_close(endpoint), AMDF_STATUS_OK);
      endpoint = nullptr;
    }
    if (instance) {
      ASSERT_EQ(api->instance_destroy(instance), AMDF_STATUS_OK);
      instance = nullptr;
    }
  }

  amdf_status_t Initialize(amdf_endpoint_id_t id) {
    const amdf_instance_create_info_t instance_info = {
        .type = AMDF_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .structure_size = sizeof(instance_info),
        .native_lifetime = AMDF_NATIVE_LIFETIME_INSTANCE,
    };
    auto status = api->instance_create(&instance_info, &instance);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    status = api->endpoint_open(instance, &id, &endpoint);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    const amdf_gpu_device_create_info_t device_info = {
        .type = AMDF_STRUCTURE_TYPE_GPU_DEVICE_CREATE_INFO,
        .structure_size = sizeof(device_info),
    };
    status = gpu->device_create(endpoint, &device_info, &device);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    const auto event_info = EventCreateInfo();
    return gpu->event_create(device, &event_info, &event);
  }
};

TEST_F(GpuEventTest, IndependentInstanceContextsIsolateAndReclaimEvents) {
  amdf_endpoint_info_t endpoint_info = {
      .type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO,
      .structure_size = sizeof(endpoint_info),
  };
  ASSERT_EQ(api_->endpoint_query_info(endpoint_, &endpoint_info),
            AMDF_STATUS_OK);
  // Repeatedly create two simultaneous contexts, then fully release each pair.
  // Neither native event identity nor backing is shared with the cached device.
  for (uint32_t epoch = 0; epoch < 3; ++epoch) {
    EventContext first = {api_, gpu_api_};
    EventContext second = {api_, gpu_api_};
    auto status = first.Initialize(endpoint_info.id);
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
      GTEST_SKIP() << "independent native instance lifetime is unavailable";
    }
    ASSERT_EQ(status, AMDF_STATUS_OK);
    ASSERT_EQ(second.Initialize(endpoint_info.id), AMDF_STATUS_OK);
    ASSERT_EQ(gpu_api_->event_signal(first.event), AMDF_STATUS_OK);
    EXPECT_EQ(gpu_api_->event_wait(second.event, 0),
              amdf_make_api_status(AMDF_STATUS_CODE_DEADLINE_EXCEEDED));
    EXPECT_EQ(gpu_api_->event_wait(first.event, amdf::cts::kWaitTimeoutNs),
              AMDF_STATUS_OK);
    ASSERT_EQ(gpu_api_->event_signal(second.event), AMDF_STATUS_OK);
    EXPECT_EQ(gpu_api_->event_wait(second.event, amdf::cts::kWaitTimeoutNs),
              AMDF_STATUS_OK);
  }
}

}  // namespace
