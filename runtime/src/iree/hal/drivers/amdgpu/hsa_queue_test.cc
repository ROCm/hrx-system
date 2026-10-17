// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/hsa_queue.h"

#include <cstdint>
#include <initializer_list>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

#if IREE_HAL_AMDGPU_HAVE_HSA_AMD_QUEUE_CREATE && !IREE_HAL_AMDGPU_LIBHSA_STATIC

enum class QueueEvent : uint8_t {
  kNone = 0,
  kDescriptorCreate,
  kStableCreate,
  kSetPriority,
  kSetMask,
  kDestroy,
};

struct QueueCreateState {
  hsa_queue_t queue = {};
  hsa_amd_queue_create_desc_t descriptor = {};
  QueueEvent events[5] = {};
  uint32_t event_count = 0;
  uint32_t stable_packet_count = 0;
  hsa_queue_type32_t stable_type = 0;
  void (*stable_callback)(hsa_status_t, hsa_queue_t*, void*) = nullptr;
  void* stable_callback_data = nullptr;
  uint32_t stable_private_segment_size = 0;
  uint32_t stable_group_segment_size = 0;
  hsa_amd_queue_priority_t applied_priority = HSA_AMD_QUEUE_PRIORITY_NORMAL;
  uint32_t applied_mask_bit_count = 0;
  uint32_t applied_mask = 0;
  hsa_status_t descriptor_create_status = HSA_STATUS_SUCCESS;
  hsa_status_t stable_create_status = HSA_STATUS_SUCCESS;
  hsa_status_t priority_status = HSA_STATUS_SUCCESS;
  hsa_status_t mask_status = HSA_STATUS_SUCCESS;
  bool descriptor_returns_queue = true;
  bool stable_returns_queue = true;
  uint32_t callback_cookie = 0;
};

static void RecordEvent(QueueCreateState* state, QueueEvent event) {
  IREE_ASSERT(state->event_count < IREE_ARRAYSIZE(state->events));
  state->events[state->event_count++] = event;
}

static void ExpectEvents(const QueueCreateState& state,
                         std::initializer_list<QueueEvent> expected) {
  ASSERT_EQ(state.event_count, expected.size());
  uint32_t i = 0;
  for (QueueEvent event : expected) {
    EXPECT_EQ(state.events[i++], event);
  }
}

static QueueCreateState* QueueCreateStateFromAgent(hsa_agent_t agent) {
  return reinterpret_cast<QueueCreateState*>(
      static_cast<uintptr_t>(agent.handle));
}

static QueueCreateState* QueueCreateStateFromQueue(const hsa_queue_t* queue) {
  return reinterpret_cast<QueueCreateState*>(static_cast<uintptr_t>(queue->id));
}

static void FakeErrorCallback(hsa_status_t, hsa_queue_t*, void*) {}

static hsa_status_t HSA_API FakeDescriptorQueueCreate(
    hsa_agent_t agent, hsa_amd_queue_create_desc_t* descriptors,
    uint32_t descriptor_count) {
  QueueCreateState* state = QueueCreateStateFromAgent(agent);
  if (!state || !descriptors || descriptor_count != 1) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  RecordEvent(state, QueueEvent::kDescriptorCreate);
  state->descriptor = descriptors[0];
  if (state->descriptor_returns_queue) {
    state->queue.id = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(state));
    descriptors[0].queue = &state->queue;
  }
  return state->descriptor_create_status;
}

static hsa_status_t HSA_API FakeStableQueueCreate(
    hsa_agent_t agent, uint32_t packet_count, hsa_queue_type32_t type,
    void (*callback)(hsa_status_t status, hsa_queue_t* source, void* data),
    void* callback_data, uint32_t private_segment_size,
    uint32_t group_segment_size, hsa_queue_t** out_queue) {
  QueueCreateState* state = QueueCreateStateFromAgent(agent);
  if (!state || !out_queue) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  RecordEvent(state, QueueEvent::kStableCreate);
  state->stable_packet_count = packet_count;
  state->stable_type = type;
  state->stable_callback = callback;
  state->stable_callback_data = callback_data;
  state->stable_private_segment_size = private_segment_size;
  state->stable_group_segment_size = group_segment_size;
  if (state->stable_returns_queue) {
    state->queue.id = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(state));
    *out_queue = &state->queue;
  }
  return state->stable_create_status;
}

static hsa_status_t HSA_API
FakeQueueSetPriority(hsa_queue_t* queue, hsa_amd_queue_priority_t priority) {
  QueueCreateState* state = QueueCreateStateFromQueue(queue);
  RecordEvent(state, QueueEvent::kSetPriority);
  state->applied_priority = priority;
  return state->priority_status;
}

static hsa_status_t HSA_API FakeQueueSetMask(const hsa_queue_t* queue,
                                             uint32_t mask_bit_count,
                                             const uint32_t* mask) {
  QueueCreateState* state = QueueCreateStateFromQueue(queue);
  RecordEvent(state, QueueEvent::kSetMask);
  state->applied_mask_bit_count = mask_bit_count;
  state->applied_mask = mask ? mask[0] : 0;
  return state->mask_status;
}

static hsa_status_t HSA_API FakeQueueDestroy(hsa_queue_t* queue) {
  QueueCreateState* state = QueueCreateStateFromQueue(queue);
  RecordEvent(state, QueueEvent::kDestroy);
  return HSA_STATUS_SUCCESS;
}

static iree_hal_amdgpu_libhsa_t QueueCreateLibhsa(bool has_descriptor_create) {
  return {
      .hsa_queue_create = FakeStableQueueCreate,
      .hsa_amd_queue_create =
          has_descriptor_create ? FakeDescriptorQueueCreate : nullptr,
      .hsa_queue_destroy = FakeQueueDestroy,
      .hsa_amd_queue_cu_set_mask = FakeQueueSetMask,
      .hsa_amd_queue_set_priority = FakeQueueSetPriority,
  };
}

static iree_hal_amdgpu_hsa_queue_params_t QueueCreateParams(
    QueueCreateState* state, const iree_hal_amdgpu_libhsa_t* libhsa,
    const uint32_t* mask) {
  const hsa_agent_t agent = {
      .handle = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(state)),
  };
  return {
      .libhsa = libhsa,
      .agent = agent,
      .packet_count = 64,
      .type = HSA_QUEUE_TYPE_MULTI,
      .priority = HSA_AMD_QUEUE_PRIORITY_HIGH,
      .compute_unit_mask_bit_count = mask ? 32u : 0u,
      .compute_unit_mask = mask,
      .error_callback = FakeErrorCallback,
      .error_callback_data = &state->callback_cookie,
  };
}

static hsa_queue_t* OutputSentinel() {
  return reinterpret_cast<hsa_queue_t*>(uintptr_t{0x1234});
}

TEST(HsaQueueTest, SelectsDescriptorFactory) {
  QueueCreateState state;
  iree_hal_amdgpu_libhsa_t libhsa = QueueCreateLibhsa(true);
  const uint32_t mask = 0xA5A5A5A5u;
  const iree_hal_amdgpu_hsa_queue_params_t params =
      QueueCreateParams(&state, &libhsa, &mask);

  hsa_queue_t* queue = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_hsa_queue_allocate(&params, &queue));

  EXPECT_EQ(queue, &state.queue);
  EXPECT_EQ(state.descriptor.version, HSA_AMD_QUEUE_CREATE_DESC_VERSION);
  EXPECT_EQ(state.descriptor.flags, HSA_AMD_QUEUE_CREATE_SYSTEM_MEM);
  EXPECT_EQ(state.descriptor.engine_type, HSA_AMD_QUEUE_ENGINE_COMPUTE);
  EXPECT_EQ(state.descriptor.queue_size_bytes,
            params.packet_count * sizeof(hsa_kernel_dispatch_packet_t));
  EXPECT_EQ(state.descriptor.priority, params.priority);
  EXPECT_EQ(state.descriptor.callback, params.error_callback);
  EXPECT_EQ(state.descriptor.callback_data, params.error_callback_data);
  EXPECT_EQ(state.descriptor.engine.compute.type, params.type);
  EXPECT_EQ(state.descriptor.engine.compute.private_segment_size,
            HSA_AMD_PRIVATE_SEGMENT_SIZE_DEFAULT);
  EXPECT_EQ(state.descriptor.engine.compute.cu_mask, nullptr);
  EXPECT_EQ(state.descriptor.engine.compute.cu_mask_count, 0u);
  EXPECT_EQ(state.applied_mask_bit_count, 32u);
  EXPECT_EQ(state.applied_mask, mask);
  ExpectEvents(state, {QueueEvent::kDescriptorCreate, QueueEvent::kSetMask});

  iree_hal_amdgpu_hsa_queue_free(&libhsa, queue);
  ExpectEvents(state, {QueueEvent::kDescriptorCreate, QueueEvent::kSetMask,
                       QueueEvent::kDestroy});
}

TEST(HsaQueueTest, SelectsStableFactoryAndReplaysProperties) {
  QueueCreateState state;
  iree_hal_amdgpu_libhsa_t libhsa = QueueCreateLibhsa(false);
  const uint32_t mask = 0x5A5A5A5Au;
  const iree_hal_amdgpu_hsa_queue_params_t params =
      QueueCreateParams(&state, &libhsa, &mask);

  hsa_queue_t* queue = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_hsa_queue_allocate(&params, &queue));

  EXPECT_EQ(queue, &state.queue);
  EXPECT_EQ(state.stable_packet_count, params.packet_count);
  EXPECT_EQ(state.stable_type, params.type);
  EXPECT_EQ(state.stable_callback, params.error_callback);
  EXPECT_EQ(state.stable_callback_data, params.error_callback_data);
  EXPECT_EQ(state.stable_private_segment_size, UINT32_MAX);
  EXPECT_EQ(state.stable_group_segment_size, UINT32_MAX);
  EXPECT_EQ(state.applied_priority, params.priority);
  EXPECT_EQ(state.applied_mask_bit_count, 32u);
  EXPECT_EQ(state.applied_mask, mask);
  ExpectEvents(state, {QueueEvent::kStableCreate, QueueEvent::kSetPriority,
                       QueueEvent::kSetMask});

  iree_hal_amdgpu_hsa_queue_free(&libhsa, queue);
  ExpectEvents(state, {QueueEvent::kStableCreate, QueueEvent::kSetPriority,
                       QueueEvent::kSetMask, QueueEvent::kDestroy});
}

TEST(HsaQueueTest, DescriptorCapacityIsByteBound) {
  QueueCreateState state;
  iree_hal_amdgpu_libhsa_t libhsa = QueueCreateLibhsa(true);
  iree_hal_amdgpu_hsa_queue_params_t params =
      QueueCreateParams(&state, &libhsa, nullptr);
  params.packet_count = UINT32_MAX / sizeof(hsa_kernel_dispatch_packet_t) + 1u;

  hsa_queue_t* queue = OutputSentinel();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_hal_amdgpu_hsa_queue_allocate(&params, &queue));
  EXPECT_EQ(queue, OutputSentinel());
  ExpectEvents(state, {});
}

TEST(HsaQueueTest, StableCapacityUsesPackets) {
  QueueCreateState state;
  iree_hal_amdgpu_libhsa_t libhsa = QueueCreateLibhsa(false);
  iree_hal_amdgpu_hsa_queue_params_t params =
      QueueCreateParams(&state, &libhsa, nullptr);
  params.packet_count = UINT32_MAX / sizeof(hsa_kernel_dispatch_packet_t) + 1u;

  hsa_queue_t* queue = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_hsa_queue_allocate(&params, &queue));
  EXPECT_EQ(queue, &state.queue);
  EXPECT_EQ(state.stable_packet_count, params.packet_count);
  ExpectEvents(state, {QueueEvent::kStableCreate, QueueEvent::kSetPriority});

  iree_hal_amdgpu_hsa_queue_free(&libhsa, queue);
}

TEST(HsaQueueTest, RejectsDescriptorSuccessWithoutQueue) {
  QueueCreateState state;
  state.descriptor_returns_queue = false;
  iree_hal_amdgpu_libhsa_t libhsa = QueueCreateLibhsa(true);
  const iree_hal_amdgpu_hsa_queue_params_t params =
      QueueCreateParams(&state, &libhsa, nullptr);

  hsa_queue_t* queue = OutputSentinel();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL,
                        iree_hal_amdgpu_hsa_queue_allocate(&params, &queue));
  EXPECT_EQ(queue, OutputSentinel());
  ExpectEvents(state, {QueueEvent::kDescriptorCreate});
}

TEST(HsaQueueTest, RejectsStableSuccessWithoutQueue) {
  QueueCreateState state;
  state.stable_returns_queue = false;
  iree_hal_amdgpu_libhsa_t libhsa = QueueCreateLibhsa(false);
  const iree_hal_amdgpu_hsa_queue_params_t params =
      QueueCreateParams(&state, &libhsa, nullptr);

  hsa_queue_t* queue = OutputSentinel();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL,
                        iree_hal_amdgpu_hsa_queue_allocate(&params, &queue));
  EXPECT_EQ(queue, OutputSentinel());
  ExpectEvents(state, {QueueEvent::kStableCreate});
}

TEST(HsaQueueTest, StablePriorityFailureDestroysUnpublishedQueue) {
  QueueCreateState state;
  state.priority_status = HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  iree_hal_amdgpu_libhsa_t libhsa = QueueCreateLibhsa(false);
  const iree_hal_amdgpu_hsa_queue_params_t params =
      QueueCreateParams(&state, &libhsa, nullptr);

  hsa_queue_t* queue = OutputSentinel();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_hal_amdgpu_hsa_queue_allocate(&params, &queue));
  EXPECT_EQ(queue, OutputSentinel());
  ExpectEvents(state, {QueueEvent::kStableCreate, QueueEvent::kSetPriority,
                       QueueEvent::kDestroy});
}

TEST(HsaQueueTest, StableMaskFailureDestroysConfiguredUnpublishedQueue) {
  QueueCreateState state;
  state.mask_status = HSA_STATUS_ERROR_INVALID_ARGUMENT;
  iree_hal_amdgpu_libhsa_t libhsa = QueueCreateLibhsa(false);
  const uint32_t mask = 0xFFFFFFFFu;
  const iree_hal_amdgpu_hsa_queue_params_t params =
      QueueCreateParams(&state, &libhsa, &mask);

  hsa_queue_t* queue = OutputSentinel();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amdgpu_hsa_queue_allocate(&params, &queue));
  EXPECT_EQ(queue, OutputSentinel());
  ExpectEvents(state, {QueueEvent::kStableCreate, QueueEvent::kSetPriority,
                       QueueEvent::kSetMask, QueueEvent::kDestroy});
}

TEST(HsaQueueTest, DescriptorFailureDestroysReturnedQueue) {
  QueueCreateState state;
  state.descriptor_create_status = HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  iree_hal_amdgpu_libhsa_t libhsa = QueueCreateLibhsa(true);
  const iree_hal_amdgpu_hsa_queue_params_t params =
      QueueCreateParams(&state, &libhsa, nullptr);

  hsa_queue_t* queue = OutputSentinel();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_hal_amdgpu_hsa_queue_allocate(&params, &queue));
  EXPECT_EQ(queue, OutputSentinel());
  ExpectEvents(state, {QueueEvent::kDescriptorCreate, QueueEvent::kDestroy});
}

TEST(HsaQueueTest, CooperativeStableCreationSkipsQueueMutation) {
  QueueCreateState state;
  iree_hal_amdgpu_libhsa_t libhsa = QueueCreateLibhsa(false);
  const uint32_t mask = 0xFFFFFFFFu;
  iree_hal_amdgpu_hsa_queue_params_t params =
      QueueCreateParams(&state, &libhsa, &mask);
  params.type = HSA_QUEUE_TYPE_COOPERATIVE;
  params.priority = HSA_AMD_QUEUE_PRIORITY_NORMAL;

  hsa_queue_t* queue = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_hsa_queue_allocate(&params, &queue));
  EXPECT_EQ(queue, &state.queue);
  EXPECT_EQ(state.stable_type, HSA_QUEUE_TYPE_COOPERATIVE);
  ExpectEvents(state, {QueueEvent::kStableCreate});

  iree_hal_amdgpu_hsa_queue_free(&libhsa, queue);
  ExpectEvents(state, {QueueEvent::kStableCreate, QueueEvent::kDestroy});
}

#endif  // IREE_HAL_AMDGPU_HAVE_HSA_AMD_QUEUE_CREATE &&
        // !IREE_HAL_AMDGPU_LIBHSA_STATIC

}  // namespace
}  // namespace iree::hal::amdgpu
