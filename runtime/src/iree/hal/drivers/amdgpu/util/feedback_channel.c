// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/feedback_channel.h"

#include "iree/base/internal/math.h"
#include "iree/hal/drivers/amdgpu/device/support/common.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"

static iree_status_t iree_hal_amdgpu_feedback_channel_validate_params(
    const iree_hal_amdgpu_feedback_channel_params_t* params) {
  IREE_ASSERT_ARGUMENT(params);
  if (IREE_UNLIKELY(!params->libhsa)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU feedback channel requires libhsa");
  }
  if (IREE_UNLIKELY(!params->device_agent.handle)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU feedback channel requires a device agent");
  }
  if (IREE_UNLIKELY(!params->control_memory_pool.handle)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU feedback channel requires a control memory pool");
  }
  if (IREE_UNLIKELY(!params->ring_memory_pool.handle)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU feedback channel requires a ring memory pool");
  }
  if (IREE_UNLIKELY(!params->topology)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU feedback channel requires a topology");
  }
  return iree_ok_status();
}

static iree_device_size_t iree_hal_amdgpu_feedback_channel_capacity(
    iree_device_size_t minimum_capacity) {
  iree_device_size_t capacity = minimum_capacity;
  if (capacity < IREE_HAL_AMDGPU_FEEDBACK_DEFAULT_RING_CAPACITY) {
    capacity = IREE_HAL_AMDGPU_FEEDBACK_DEFAULT_RING_CAPACITY;
  }
  if (!iree_device_size_is_power_of_two(capacity)) {
    capacity = iree_device_size_next_power_of_two(capacity);
  }
  return capacity;
}

iree_status_t iree_hal_amdgpu_feedback_channel_initialize(
    const iree_hal_amdgpu_feedback_channel_params_t* params,
    iree_hal_amdgpu_feedback_channel_t* out_channel) {
  IREE_ASSERT_ARGUMENT(out_channel);
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_channel, 0, sizeof(*out_channel));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_feedback_channel_validate_params(params));

  out_channel->libhsa = params->libhsa;

  const iree_device_size_t capacity =
      iree_hal_amdgpu_feedback_channel_capacity(params->minimum_capacity);
  if (IREE_UNLIKELY(!iree_device_size_is_power_of_two(capacity) ||
                    capacity > IREE_DEVICE_SIZE_MAX / 3)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                         "AMDGPU feedback ring capacity is too large: %" PRIu64,
                         (uint64_t)capacity));
  }

  iree_status_t status =
      iree_hal_amdgpu_vmem_ringbuffer_initialize_with_topology(
          params->libhsa, params->device_agent, params->ring_memory_pool,
          IREE_HAL_AMDGPU_VMEM_MEMORY_TYPE_PINNED_HOST, capacity,
          params->topology, IREE_HAL_AMDGPU_ACCESS_MODE_SHARED,
          &out_channel->ringbuffer);

  if (iree_status_is_ok(status) &&
      !iree_device_size_is_power_of_two(out_channel->ringbuffer.capacity)) {
    status = iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU feedback VMM ring capacity must be a power of two: %" PRIu64,
        (uint64_t)out_channel->ringbuffer.capacity);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hsa_amd_memory_pool_allocate(
        IREE_LIBHSA(params->libhsa), params->control_memory_pool,
        sizeof(*out_channel->control), HSA_AMD_MEMORY_POOL_STANDARD_FLAG,
        (void**)&out_channel->control);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hsa_amd_agents_allow_access(
        IREE_LIBHSA(params->libhsa), /*num_agents=*/1, &params->device_agent,
        /*flags=*/NULL, out_channel->control);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hsa_amd_signal_create(
        IREE_LIBHSA(params->libhsa), /*initial_value=*/0,
        /*num_consumers=*/0, /*consumers=*/NULL, /*attributes=*/0,
        &out_channel->notify_signal);
  }

  if (iree_status_is_ok(status)) {
    memset(out_channel->control, 0, sizeof(*out_channel->control));
    out_channel->control->record_length = sizeof(*out_channel->control);
    out_channel->control->abi_version =
        IREE_HAL_AMDGPU_FEEDBACK_CHANNEL_ABI_VERSION_0;
    out_channel->control->flags = IREE_HAL_AMDGPU_FEEDBACK_CONFIG_FLAG_ENABLED;
    out_channel->control->ring_base =
        (uint64_t)(uintptr_t)out_channel->ringbuffer.ring_base_ptr;
    out_channel->control->ring_capacity = out_channel->ringbuffer.capacity;

    out_channel->config = (iree_hal_amdgpu_feedback_config_t){
        .record_length = sizeof(out_channel->config),
        .abi_version = IREE_HAL_AMDGPU_FEEDBACK_CONFIG_ABI_VERSION_0,
        .flags = IREE_HAL_AMDGPU_FEEDBACK_CONFIG_FLAG_ENABLED,
        .reserved0 = 0,
        .channel_base = (uint64_t)(uintptr_t)out_channel->control,
        .notify_signal = out_channel->notify_signal,
        .source_context = 0,
        .reserved = {0},
    };
  } else {
    iree_hal_amdgpu_feedback_channel_deinitialize(out_channel);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_amdgpu_feedback_channel_deinitialize(
    iree_hal_amdgpu_feedback_channel_t* channel) {
  if (!channel || !channel->libhsa) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!iree_hsa_signal_is_null(channel->notify_signal)) {
    iree_hal_amdgpu_hsa_cleanup_assert_success(
        iree_hsa_signal_destroy_raw(channel->libhsa, channel->notify_signal));
  }
  if (channel->control) {
    iree_hal_amdgpu_hsa_cleanup_assert_success(
        iree_hsa_amd_memory_pool_free_raw(channel->libhsa, channel->control));
  }
  iree_hal_amdgpu_vmem_ringbuffer_deinitialize(channel->libhsa,
                                               &channel->ringbuffer);
  memset(channel, 0, sizeof(*channel));

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_amdgpu_feedback_channel_drain(
    iree_hal_amdgpu_feedback_channel_t* channel,
    iree_host_size_t max_packet_count,
    iree_hal_amdgpu_feedback_drain_fn_t callback, void* user_data,
    iree_host_size_t* out_packet_count) {
  IREE_ASSERT_ARGUMENT(channel);
  IREE_ASSERT_ARGUMENT(callback);
  IREE_ASSERT_ARGUMENT(out_packet_count);
  *out_packet_count = 0;

  iree_hal_amdgpu_feedback_channel_header_t* control = channel->control;
  if (IREE_UNLIKELY(!control)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU feedback channel is not initialized");
  }

  const uint64_t ring_capacity = control->ring_capacity;
  const uint64_t ring_mask = ring_capacity - 1u;
  if (IREE_UNLIKELY(ring_capacity == 0 ||
                    !iree_device_size_is_power_of_two(ring_capacity))) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU feedback ring capacity must be a non-zero power of two");
  }

  iree_amdgpu_scoped_atomic_uint64_t* reservation_head =
      (iree_amdgpu_scoped_atomic_uint64_t*)&control->reservation_head;
  iree_amdgpu_scoped_atomic_uint64_t* read_tail =
      (iree_amdgpu_scoped_atomic_uint64_t*)&control->read_tail;
  uint64_t read_position = iree_amdgpu_scoped_atomic_load(
      read_tail, iree_amdgpu_memory_order_acquire,
      iree_amdgpu_memory_scope_system);
  const uint64_t reserved_position = iree_amdgpu_scoped_atomic_load(
      reservation_head, iree_amdgpu_memory_order_acquire,
      iree_amdgpu_memory_scope_system);

  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) && *out_packet_count < max_packet_count &&
         read_position < reserved_position) {
    iree_hal_amdgpu_feedback_packet_t* packet =
        (iree_hal_amdgpu_feedback_packet_t*)(uintptr_t)(control->ring_base +
                                                        (read_position &
                                                         ring_mask));
    const iree_hal_amdgpu_feedback_packet_state_t state =
        iree_amdgpu_scoped_atomic_load(
            (iree_amdgpu_scoped_atomic_uint32_t*)&packet->state,
            iree_amdgpu_memory_order_acquire, iree_amdgpu_memory_scope_system);
    if (state != IREE_HAL_AMDGPU_FEEDBACK_PACKET_STATE_READY ||
        packet->sequence != read_position) {
      break;
    }

    const uint32_t packet_length = packet->record_length;
    if (IREE_UNLIKELY(packet->header_length < sizeof(*packet) ||
                      packet_length < packet->header_length ||
                      packet_length > ring_capacity ||
                      (packet_length &
                       (IREE_HAL_AMDGPU_FEEDBACK_PACKET_ALIGNMENT - 1u)) !=
                          0)) {
      status = iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "invalid AMDGPU feedback packet at ring position %" PRIu64
          ": header_length=%u, record_length=%u, ring_capacity=%" PRIu64,
          read_position, packet->header_length, packet_length, ring_capacity);
      break;
    }

    status = callback(packet, user_data);
    if (iree_status_is_ok(status)) {
      read_position += packet_length;
      iree_amdgpu_scoped_atomic_store(read_tail, read_position,
                                      iree_amdgpu_memory_order_release,
                                      iree_amdgpu_memory_scope_system);
      ++*out_packet_count;
    }
  }

  return status;
}
