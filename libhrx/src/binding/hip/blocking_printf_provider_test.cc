// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/blocking_printf_provider.h"

#include <array>
#include <cstring>
#include <string>
#include <vector>

#include "binding/hip/blocking_printf_protocol.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

enum FragmentFlags : uint64_t {
  kFragmentBegin = UINT64_C(1) << 0,
  kFragmentEnd = UINT64_C(1) << 1,
};

struct ProviderRecorder {
  // Number of typed printf events received.
  int event_count = 0;

  // Stream carried by the last typed printf event.
  iree_hal_device_printf_stream_t stream =
      IREE_HAL_DEVICE_PRINTF_STREAM_DEFAULT;

  // Physical-device ordinal carried by the last event.
  uint32_t physical_device_ordinal = UINT32_MAX;

  // Complete printf text copied during the event callback.
  std::string text;

  // Number of logical-device failures received.
  int error_count = 0;

  // Status code consumed from the last logical-device failure.
  iree_status_code_t error_code = IREE_STATUS_OK;
};

static void CaptureEvent(void* user_data,
                         const iree_hal_device_event_t* event) {
  ProviderRecorder* recorder = (ProviderRecorder*)user_data;
  EXPECT_EQ(event->type, IREE_HAL_DEVICE_EVENT_TYPE_PRINTF);
  EXPECT_EQ(event->severity, IREE_HAL_DEVICE_EVENT_SEVERITY_INFO);
  EXPECT_TRUE(iree_string_view_equal(event->source.driver_id, IREE_SV("hip")));
  EXPECT_EQ(event->payload.data_length, sizeof(iree_hal_device_printf_event_t));

  iree_hal_device_printf_event_t printf_event;
  memcpy(&printf_event, event->payload.data, sizeof(printf_event));
  EXPECT_EQ(printf_event.record_length, sizeof(printf_event));
  EXPECT_EQ(printf_event.abi_version,
            IREE_HAL_DEVICE_PRINTF_EVENT_ABI_VERSION_0);
  EXPECT_EQ(printf_event.arguments.data_length, 0u);
  ++recorder->event_count;
  recorder->stream = printf_event.stream;
  recorder->physical_device_ordinal = event->source.physical_device_ordinal;
  recorder->text.assign(printf_event.text.data, printf_event.text.size);
}

static void CaptureError(void* user_data, iree_status_t status) {
  ProviderRecorder* recorder = (ProviderRecorder*)user_data;
  ++recorder->error_count;
  recorder->error_code = iree_status_code(status);
  iree_status_free(status);
}

static uint64_t MakeDescriptor(uint64_t flags, iree_host_size_t length) {
  return flags | ((uint64_t)length << 5);
}

TEST(BlockingPrintfProviderTest, PublishesTypedEventsAndDeviceFailures) {
  ProviderRecorder recorder;
  const iree_hal_device_event_sink_t event_sink = {
      /*.fn=*/CaptureEvent,
      /*.user_data=*/&recorder,
  };
  iree_hip_blocking_printf_provider_t provider;
  iree_hip_blocking_printf_provider_initialize(
      event_sink, iree_allocator_system(), &provider);

  const iree_hal_device_create_params_extension_t* base_extension =
      iree_hip_blocking_printf_provider_device_extension(&provider);
  ASSERT_NE(base_extension, nullptr);
  EXPECT_EQ(base_extension->type,
            IREE_HAL_DEVICE_CREATE_PARAMS_EXTENSION_TYPE_HOSTCALL_PROVIDER);
  const iree_hal_hostcall_provider_extension_t* extension =
      (const iree_hal_hostcall_provider_extension_t*)base_extension;

  const iree_hal_hostcall_provider_device_info_t device_info = {
      /*.physical_device_ordinal=*/3,
      /*.execution_unit_count=*/1,
      /*.maximum_resident_subgroup_count=*/1,
  };
  iree_hal_hostcall_provider_requirements_t requirements;
  IREE_ASSERT_OK(extension->provider.query_requirements(
      extension->provider.user_data, &device_info, &requirements));
  ASSERT_GT(requirements.allocation_size, 0u);
  ASSERT_TRUE(
      iree_host_size_is_power_of_two(requirements.allocation_alignment));
  EXPECT_EQ(requirements.notification_type,
            IREE_HAL_HOSTCALL_NOTIFICATION_TYPE_HSA_SIGNAL);

  std::vector<uint64_t> storage(
      (requirements.allocation_size + sizeof(uint64_t) - 1) / sizeof(uint64_t));
  ASSERT_EQ((uintptr_t)storage.data() & (requirements.allocation_alignment - 1),
            0u);
  const uint64_t device_address = (uint64_t)(uintptr_t)storage.data();
  const uint64_t notification_token = UINT64_C(0x12345678);
  const iree_hal_hostcall_notification_t notification = {
      /*.type=*/IREE_HAL_HOSTCALL_NOTIFICATION_TYPE_HSA_SIGNAL,
      /*.reserved=*/0,
      /*.token=*/notification_token,
  };
  const iree_hal_hostcall_error_callback_t error_callback = {
      /*.fn=*/CaptureError,
      /*.user_data=*/&recorder,
  };
  void* context = nullptr;
  IREE_ASSERT_OK(extension->provider.initialize(
      extension->provider.user_data, &device_info,
      iree_make_byte_span(storage.data(), requirements.allocation_size),
      device_address, notification, error_callback, &context));
  ASSERT_NE(context, nullptr);

  iree_hip_hostcall_buffer_header_t* buffer_header =
      (iree_hip_hostcall_buffer_header_t*)storage.data();
  EXPECT_EQ(buffer_header->doorbell, notification_token);
  iree_hip_hostcall_packet_header_t* packet_headers =
      (iree_hip_hostcall_packet_header_t*)(uintptr_t)buffer_header->headers;
  iree_hip_hostcall_packet_payload_t* packet_payloads =
      (iree_hip_hostcall_packet_payload_t*)(uintptr_t)buffer_header->payloads;

  const uint32_t packet_index = 1;
  iree_hip_hostcall_packet_header_t* packet_header =
      &packet_headers[packet_index];
  packet_header->next = 0;
  packet_header->activemask = 1;
  packet_header->service = IREE_HIP_HOSTCALL_SERVICE_PRINTF;
  iree_atomic_store(&packet_header->control,
                    IREE_HIP_HOSTCALL_PACKET_CONTROL_READY,
                    iree_memory_order_relaxed);
  const char format[] = "value=%d\n";
  std::array<uint64_t, 4> message = {
      /*stream=*/IREE_HIP_BLOCKING_PRINTF_STREAM_STDERR, 0, 0, 42};
  memcpy(&message[1], format, sizeof(format));
  uint64_t* payload = packet_payloads[packet_index].slots[0];
  payload[0] = MakeDescriptor(kFragmentBegin | kFragmentEnd, message.size());
  memcpy(payload + 1, message.data(), sizeof(message));
  iree_atomic_store(&buffer_header->ready_stack, packet_index,
                    iree_memory_order_release);

  extension->provider.service(context);

  EXPECT_EQ(recorder.event_count, 1);
  EXPECT_EQ(recorder.stream, IREE_HAL_DEVICE_PRINTF_STREAM_STDERR);
  EXPECT_EQ(recorder.physical_device_ordinal, 3u);
  EXPECT_EQ(recorder.text, "value=42\n");
  EXPECT_EQ(payload[0], 9u);
  EXPECT_EQ(
      iree_atomic_load(&packet_header->control, iree_memory_order_acquire), 0u);

  packet_header->next = 0;
  packet_header->activemask = 1;
  packet_header->service = UINT32_MAX;
  iree_atomic_store(&packet_header->control,
                    IREE_HIP_HOSTCALL_PACKET_CONTROL_READY,
                    iree_memory_order_relaxed);
  iree_atomic_store(&buffer_header->ready_stack, packet_index,
                    iree_memory_order_release);

  extension->provider.service(context);

  EXPECT_EQ(recorder.error_count, 1);
  EXPECT_EQ(recorder.error_code, IREE_STATUS_UNIMPLEMENTED);
  EXPECT_EQ(payload[0], UINT64_MAX);
  EXPECT_EQ(
      iree_atomic_load(&packet_header->control, iree_memory_order_acquire), 0u);

  extension->provider.deinitialize(context);
}

}  // namespace
