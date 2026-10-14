// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/context.h"

#include <array>
#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// The dependency exposes the native borrowed-owner contract. Only dependency
// calls fail: allocation, rollback and captured facts use the production owner.
struct NativeState {
  // Next native acquisition/query call, starting at one.
  uint32_t call_count = 0;
  // Call to reject, or zero for successful native construction.
  uint32_t fail_call = 0;
  // Number of physical-local scopes advertised by this endpoint.
  uint32_t scope_count = 2;
  // Target identity deliberately independent of its queue protocols.
  uint32_t gfx_major = 11;
  // Whether the dependency currently owns an endpoint.
  bool endpoint_live = false;
  // Whether the dependency currently owns a device borrowing the endpoint.
  bool device_live = false;
  // Whether the dependency currently owns an event borrowing the device.
  bool event_live = false;
  // Endpoint-owned identities used by borrowed scope handles.
  std::array<int, 2> scopes = {};
  // Native destruction order, checked independently of acquisition failures.
  std::vector<std::string> released;
  // Native teardown boundary to reject, or empty for successful cleanup.
  std::string fail_release;
  // Original native failures delivered through the device's event sink.
  std::vector<std::string> reports;

  amdf_status_t NextCall() {
    return ++call_count == fail_call
               ? amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED)
               : AMDF_STATUS_OK;
  }
};

amdf_status_t AMDF_CALL OpenEndpoint(amdf_instance_t* instance,
                                     const amdf_endpoint_id_t* id,
                                     amdf_endpoint_t** out_endpoint) {
  auto* state = reinterpret_cast<NativeState*>(instance);
  amdf_status_t status = state->NextCall();
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  EXPECT_EQ(id->words[0], 37u);
  EXPECT_FALSE(state->endpoint_live);
  state->endpoint_live = true;
  *out_endpoint = reinterpret_cast<amdf_endpoint_t*>(state);
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL QueryEndpoint(amdf_endpoint_t* endpoint,
                                      amdf_endpoint_info_t* out_info) {
  auto* state = reinterpret_cast<NativeState*>(endpoint);
  amdf_status_t status = state->NextCall();
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  EXPECT_EQ(out_info->type, AMDF_STRUCTURE_TYPE_ENDPOINT_INFO);
  out_info->queue_family_count = 3;
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL QueryTarget(amdf_endpoint_t* endpoint,
                                    amdf_gpu_endpoint_info_t* out_info) {
  auto* state = reinterpret_cast<NativeState*>(endpoint);
  amdf_status_t status = state->NextCall();
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  EXPECT_EQ(out_info->type, AMDF_STRUCTURE_TYPE_GPU_ENDPOINT_INFO);
  out_info->gfx_ip.major = state->gfx_major;
  out_info->compute.compute_unit_count = 12;
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL QueryFamily(amdf_endpoint_t* endpoint, uint32_t ordinal,
                                    amdf_queue_family_info_t* out_info) {
  auto* state = reinterpret_cast<NativeState*>(endpoint);
  amdf_status_t status = state->NextCall();
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  EXPECT_EQ(out_info->type, AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO);
  out_info->ordinal = ordinal;
  out_info->command_type = ordinal == 1 ? AMDF_QUEUE_COMMAND_TYPE_GPU_AQL
                                        : AMDF_QUEUE_COMMAND_TYPE_GPU_PM4;
  out_info->publication_modes = ordinal == 2
                                    ? AMDF_QUEUE_PUBLICATION_MODE_KERNEL
                                    : AMDF_QUEUE_PUBLICATION_MODE_USER;
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL CreateDevice(amdf_endpoint_t* endpoint,
                                     const amdf_gpu_device_create_info_t* info,
                                     amdf_device_t** out_device) {
  auto* state = reinterpret_cast<NativeState*>(endpoint);
  amdf_status_t status = state->NextCall();
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  EXPECT_EQ(info->type, AMDF_STRUCTURE_TYPE_GPU_DEVICE_CREATE_INFO);
  EXPECT_TRUE(state->endpoint_live);
  state->device_live = true;
  *out_device = reinterpret_cast<amdf_device_t*>(state);
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL QueryDevice(amdf_device_t* device,
                                    amdf_gpu_device_info_t* out_info) {
  auto* state = reinterpret_cast<NativeState*>(device);
  amdf_status_t status = state->NextCall();
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  EXPECT_EQ(out_info->type, AMDF_STRUCTURE_TYPE_GPU_DEVICE_INFO);
  out_info->reset_epoch = 91;
  out_info->features = AMDF_GPU_DEVICE_FEATURE_NATIVE_EVENTS;
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL EnumerateScopes(amdf_device_t* device,
                                        uint32_t capacity,
                                        amdf_memory_scope_t** scopes,
                                        uint32_t* out_count) {
  auto* state = reinterpret_cast<NativeState*>(device);
  amdf_status_t status = state->NextCall();
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  *out_count = state->scope_count;
  for (uint32_t i = 0; i < capacity && i < state->scope_count; ++i) {
    scopes[i] = reinterpret_cast<amdf_memory_scope_t*>(&state->scopes[i]);
  }
  return capacity < state->scope_count
             ? amdf_make_api_status(AMDF_STATUS_CODE_BUFFER_TOO_SMALL)
             : AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL CreateEvent(amdf_device_t* device,
                                    const amdf_gpu_event_create_info_t* info,
                                    amdf_gpu_event_t** out_event) {
  auto* state = reinterpret_cast<NativeState*>(device);
  amdf_status_t status = state->NextCall();
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  EXPECT_EQ(info->type, AMDF_STRUCTURE_TYPE_GPU_EVENT_CREATE_INFO);
  EXPECT_TRUE(state->device_live);
  state->event_live = true;
  *out_event = reinterpret_cast<amdf_gpu_event_t*>(state);
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL QueryEvent(amdf_gpu_event_t* event,
                                   amdf_gpu_event_info_t* out_info) {
  auto* state = reinterpret_cast<NativeState*>(event);
  amdf_status_t status = state->NextCall();
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  EXPECT_EQ(out_info->type, AMDF_STRUCTURE_TYPE_GPU_EVENT_INFO);
  out_info->format = AMDF_GPU_EVENT_FORMAT_KFD_SIGNAL;
  out_info->format_version = 1;
  out_info->mailbox_address = 0x10000;
  out_info->trigger_data = 73;
  out_info->reset_epoch = 91;
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL DestroyEvent(amdf_gpu_event_t* event) {
  auto* state = reinterpret_cast<NativeState*>(event);
  EXPECT_TRUE(state->event_live);
  EXPECT_TRUE(state->device_live);
  if (state->fail_release == "event") {
    return amdf_make_api_status(AMDF_STATUS_CODE_PERMISSION_DENIED);
  }
  state->event_live = false;
  state->released.push_back("event");
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL DestroyDevice(amdf_device_t* device) {
  auto* state = reinterpret_cast<NativeState*>(device);
  EXPECT_FALSE(state->event_live);
  EXPECT_TRUE(state->device_live);
  EXPECT_TRUE(state->endpoint_live);
  if (state->fail_release == "device") {
    return amdf_make_api_status(AMDF_STATUS_CODE_PERMISSION_DENIED);
  }
  state->device_live = false;
  state->released.push_back("device");
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL CloseEndpoint(amdf_endpoint_t* endpoint) {
  auto* state = reinterpret_cast<NativeState*>(endpoint);
  EXPECT_FALSE(state->device_live);
  EXPECT_TRUE(state->endpoint_live);
  if (state->fail_release == "endpoint") {
    return amdf_make_api_status(AMDF_STATUS_CODE_PERMISSION_DENIED);
  }
  state->endpoint_live = false;
  state->released.push_back("endpoint");
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL QueryExtension(amdf_extension_id_t id,
                                       uint32_t minimum_version,
                                       uint32_t maximum_version,
                                       const void** out_api) {
  static const amdf_gpu_api_t gpu = {
      .structure_size = sizeof(gpu),
      .extension_version = AMDF_GPU_EXTENSION_VERSION_2,
      .endpoint_query_info = QueryTarget,
      .device_create = CreateDevice,
      .device_query_info = QueryDevice,
      .event_create = CreateEvent,
      .event_query_info = QueryEvent,
      .event_destroy = DestroyEvent,
  };
  EXPECT_EQ(id, AMDF_EXTENSION_GPU);
  EXPECT_EQ(minimum_version, AMDF_GPU_EXTENSION_VERSION_2);
  EXPECT_GE(maximum_version, minimum_version);
  *out_api = &gpu;
  return AMDF_STATUS_OK;
}

class ContextTest : public ::testing::Test {
 protected:
  void SetUp() override {
    api_.query_extension = QueryExtension;
    api_.endpoint_open = OpenEndpoint;
    api_.endpoint_query_info = QueryEndpoint;
    api_.endpoint_query_queue_family_info = QueryFamily;
    api_.endpoint_close = CloseEndpoint;
    api_.device_destroy = DestroyDevice;
    api_.device_enumerate_memory_scopes = EnumerateScopes;
    iree_atomic_ref_count_init(&fabric_.ref_count);
    fabric_.api = &api_;
    fabric_.instance = reinterpret_cast<amdf_instance_t*>(&native_);
  }

  void TearDown() override {
    EXPECT_FALSE(native_.endpoint_live);
    EXPECT_FALSE(native_.device_live);
    EXPECT_FALSE(native_.event_live);
    EXPECT_EQ(iree_atomic_ref_count_load(&fabric_.ref_count), 1);
  }

  iree_status_t Allocate(iree_hal_amd_gpu_context_t** out_context) {
    const amdf_endpoint_id_t id = {{37, 0}};
    const iree_hal_device_event_sink_t sink = {
        .fn =
            [](void* user_data, const iree_hal_device_event_t* event) {
              auto* state = static_cast<NativeState*>(user_data);
              EXPECT_EQ(event->type, IREE_HAL_DEVICE_EVENT_TYPE_DRIVER_FAILURE);
              const auto* payload = reinterpret_cast<
                  const iree_hal_device_driver_failure_event_t*>(
                  event->payload.data);
              EXPECT_EQ(
                  payload->backend_result_code,
                  amdf_make_api_status(AMDF_STATUS_CODE_PERMISSION_DENIED));
              state->reports.emplace_back(payload->message.data,
                                          payload->message.size);
            },
        .user_data = &native_,
    };
    return iree_hal_amd_gpu_context_allocate(
        &fabric_, &id, sink, iree_allocator_system(), out_context);
  }

  // Dependency-owned identities and construction failure schedule.
  NativeState native_;
  // Native entry points supplied to the real context constructor.
  amdf_api_t api_ = {};
  // Existing fabric reference retained independently of the context.
  iree_hal_amd_fabric_t fabric_ = {};
};

TEST_F(ContextTest, CapturesFamiliesIndependentlyOfIsa) {
  for (uint32_t gfx_major : {9u, 11u, 12u}) {
    native_.gfx_major = gfx_major;
    iree_hal_amd_gpu_context_t* context = nullptr;
    IREE_ASSERT_OK(Allocate(&context));
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(iree_atomic_ref_count_load(&fabric_.ref_count), 2);
    EXPECT_EQ(context->target_info.gfx_ip.major, gfx_major);
    EXPECT_EQ(context->target_info.compute.compute_unit_count, 12u);
    EXPECT_EQ(context->device_info.reset_epoch, 91u);
    EXPECT_EQ(context->event_info.mailbox_address, 0x10000u);
    EXPECT_EQ(context->event_info.trigger_data, 73u);
    ASSERT_EQ(context->endpoint_info.queue_family_count, 3u);
    EXPECT_EQ(context->families[0].command_type,
              AMDF_QUEUE_COMMAND_TYPE_GPU_PM4);
    EXPECT_EQ(context->families[1].command_type,
              AMDF_QUEUE_COMMAND_TYPE_GPU_AQL);
    EXPECT_EQ(context->families[2].publication_modes,
              AMDF_QUEUE_PUBLICATION_MODE_KERNEL);
    ASSERT_EQ(context->memory_scope_count, 2u);
    for (uint32_t i = 0; i < context->memory_scope_count; ++i) {
      EXPECT_EQ(context->memory_scopes[i],
                reinterpret_cast<amdf_memory_scope_t*>(&native_.scopes[i]));
    }
    iree_hal_amd_gpu_context_free(context);
    EXPECT_EQ(native_.released,
              (std::vector<std::string>{"event", "device", "endpoint"}));
    native_.released.clear();
  }
}

TEST_F(ContextTest, NoPhysicalLocalStorageStillHasSystemStorage) {
  native_.scope_count = 0;
  iree_hal_amd_gpu_context_t* context = nullptr;
  IREE_ASSERT_OK(Allocate(&context));
  EXPECT_EQ(context->memory_scope_count, 0u);
  EXPECT_EQ(context->memory_scopes, nullptr);
  EXPECT_EQ(context->fabric, &fabric_);
  iree_hal_amd_gpu_context_free(context);
}

TEST_F(ContextTest, EveryNativeConstructionFailureUnwindsAcquiredOwners) {
  iree_hal_amd_gpu_context_t* context = nullptr;
  IREE_ASSERT_OK(Allocate(&context));
  const uint32_t call_count = native_.call_count;
  iree_hal_amd_gpu_context_free(context);
  for (uint32_t failure = 1; failure <= call_count; ++failure) {
    SCOPED_TRACE(failure);
    native_.call_count = 0;
    native_.fail_call = failure;
    context = nullptr;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, Allocate(&context));
    EXPECT_EQ(context, nullptr);
    EXPECT_FALSE(native_.endpoint_live);
    EXPECT_FALSE(native_.device_live);
    EXPECT_FALSE(native_.event_live);
    EXPECT_EQ(iree_atomic_ref_count_load(&fabric_.ref_count), 1);
  }
}

TEST_F(ContextTest, FailedCleanupPreservesBorrowedParentsAndReportsFailure) {
  for (const char* boundary : {"event", "device", "endpoint"}) {
    SCOPED_TRACE(boundary);
    iree_hal_amd_gpu_context_t* context = nullptr;
    IREE_ASSERT_OK(Allocate(&context));
    native_.fail_release = boundary;
    iree_hal_amd_gpu_context_free(context);
    EXPECT_EQ(native_.event_live, native_.fail_release == "event");
    EXPECT_EQ(native_.device_live, native_.fail_release != "endpoint");
    EXPECT_TRUE(native_.endpoint_live);
    EXPECT_EQ(iree_atomic_ref_count_load(&fabric_.ref_count), 2);
    ASSERT_EQ(native_.reports.size(), 1u);
    const std::string operation =
        native_.fail_release == "event"    ? "gpu.event_destroy"
        : native_.fail_release == "device" ? "device_destroy"
                                           : "endpoint_close";
    EXPECT_EQ(native_.reports[0], operation);

    // The dependency now permits teardown. The saved, preserved owner remains
    // available to the test, and already-consumed children must not be retried.
    native_.fail_release.clear();
    iree_hal_amd_gpu_context_free(context);
    EXPECT_EQ(native_.released,
              (std::vector<std::string>{"event", "device", "endpoint"}));
    native_.released.clear();
    native_.reports.clear();
  }
}

}  // namespace
