// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Device-side feedback channel helpers built on top of the ABI types.

#ifndef IREE_HAL_DRIVERS_AMDGPU_DEVICE_SUPPORT_FEEDBACK_H_
#define IREE_HAL_DRIVERS_AMDGPU_DEVICE_SUPPORT_FEEDBACK_H_

#include "iree/hal/drivers/amdgpu/abi/feedback.h"  // IWYU pragma: export
#include "iree/hal/drivers/amdgpu/device/support/kernel.h"
#include "iree/hal/drivers/amdgpu/device/support/signal.h"

// Returns the payload storage following |packet|.
static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE void*
iree_hal_amdgpu_feedback_packet_payload(
    iree_hal_amdgpu_feedback_packet_t* packet) {
  return (uint8_t*)packet + packet->header_length;
}

// Returns the immutable payload storage following |packet|.
static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE const void*
iree_hal_amdgpu_feedback_packet_const_payload(
    const iree_hal_amdgpu_feedback_packet_t* packet) {
  return (const uint8_t*)packet + packet->header_length;
}

// Attempts to reserve packet storage in |config| for |payload_length| bytes.
//
// Returns false if feedback is disabled, the payload is too large, or the ring
// lacks sufficient free space. On success |out_packet| receives writable packet
// storage and the caller must fill the payload before calling
// |iree_hal_amdgpu_feedback_publish|.
static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE bool
iree_hal_amdgpu_feedback_try_reserve(
    const iree_hal_amdgpu_feedback_config_t* IREE_AMDGPU_RESTRICT config,
    iree_hal_amdgpu_feedback_packet_kind_t kind,
    iree_hal_amdgpu_feedback_packet_flags_t flags, size_t payload_length,
    iree_hal_amdgpu_feedback_packet_t** out_packet) {
  if (!out_packet) return false;
  *out_packet = NULL;
  if (!config ||
      !IREE_AMDGPU_ANY_BIT_SET(config->flags,
                               IREE_HAL_AMDGPU_FEEDBACK_CONFIG_FLAG_ENABLED)) {
    return false;
  }

  iree_hal_amdgpu_feedback_channel_header_t* channel =
      (iree_hal_amdgpu_feedback_channel_header_t*)(uintptr_t)
          config->channel_base;
  if (!channel || channel->record_length < sizeof(*channel) ||
      channel->abi_version != IREE_HAL_AMDGPU_FEEDBACK_CHANNEL_ABI_VERSION_0) {
    return false;
  }
  if (payload_length > IREE_HAL_AMDGPU_FEEDBACK_PACKET_MAX_PAYLOAD_LENGTH) {
    return false;
  }

  const uint64_t ring_capacity = channel->ring_capacity;
  if (ring_capacity == 0 || (ring_capacity & (ring_capacity - 1u)) != 0) {
    return false;
  }

  const size_t packet_length =
      iree_hal_amdgpu_feedback_packet_length(payload_length);
  if (packet_length > ring_capacity || packet_length > UINT32_MAX) {
    return false;
  }

  iree_amdgpu_scoped_atomic_uint64_t* reservation_head =
      (iree_amdgpu_scoped_atomic_uint64_t*)&channel->reservation_head;
  iree_amdgpu_scoped_atomic_uint64_t* read_tail =
      (iree_amdgpu_scoped_atomic_uint64_t*)&channel->read_tail;

  while (true) {
    const uint64_t head = iree_amdgpu_scoped_atomic_load(
        reservation_head, iree_amdgpu_memory_order_relaxed,
        iree_amdgpu_memory_scope_system);
    const uint64_t tail = iree_amdgpu_scoped_atomic_load(
        read_tail, iree_amdgpu_memory_order_acquire,
        iree_amdgpu_memory_scope_system);
    const uint64_t used_capacity = head - tail;
    if (used_capacity > ring_capacity ||
        ring_capacity - used_capacity < packet_length) {
      iree_amdgpu_scoped_atomic_fetch_add(
          (iree_amdgpu_scoped_atomic_uint64_t*)&channel->dropped_packet_count,
          1, iree_amdgpu_memory_order_relaxed, iree_amdgpu_memory_scope_system);
      return false;
    }

    const uint64_t next_head = head + packet_length;
    uint64_t expected_head = head;
    if (!iree_amdgpu_scoped_atomic_compare_exchange_strong(
            reservation_head, &expected_head, next_head,
            iree_amdgpu_memory_order_acq_rel, iree_amdgpu_memory_order_relaxed,
            iree_amdgpu_memory_scope_system)) {
      iree_amdgpu_yield();
      continue;
    }

    const uint64_t ring_offset = head & (ring_capacity - 1u);
    iree_hal_amdgpu_feedback_packet_t* packet =
        (iree_hal_amdgpu_feedback_packet_t*)(uintptr_t)(channel->ring_base +
                                                        ring_offset);
    iree_amdgpu_scoped_atomic_store(
        (iree_amdgpu_scoped_atomic_uint32_t*)&packet->state,
        IREE_HAL_AMDGPU_FEEDBACK_PACKET_STATE_RESERVED,
        iree_amdgpu_memory_order_relaxed, iree_amdgpu_memory_scope_system);
    packet->record_length = (uint32_t)packet_length;
    packet->header_length = sizeof(*packet);
    packet->kind = kind;
    packet->flags = flags;
    packet->sequence = head;
#if defined(IREE_AMDGPU_TARGET_DEVICE)
    packet->source_dispatch_ptr =
        (uint64_t)(uintptr_t)iree_amdgcn_dispatch_ptr();
    packet->source_workgroup_id_x = iree_hal_amdgpu_device_group_id_x();
    packet->source_workitem_id_x = iree_hal_amdgpu_device_local_id_x();
#else
    packet->source_dispatch_ptr = 0;
    packet->source_workgroup_id_x = 0;
    packet->source_workitem_id_x = 0;
#endif  // IREE_AMDGPU_TARGET_DEVICE
    packet->source_context = config->source_context;
    packet->reserved[0] = 0;
    packet->reserved[1] = 0;
    *out_packet = packet;
    return true;
  }
}

// Publishes a reserved packet and wakes the host feedback service.
static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE void
iree_hal_amdgpu_feedback_publish(
    const iree_hal_amdgpu_feedback_config_t* IREE_AMDGPU_RESTRICT config,
    iree_hal_amdgpu_feedback_packet_t* IREE_AMDGPU_RESTRICT packet) {
  iree_amdgpu_scoped_atomic_store(
      (iree_amdgpu_scoped_atomic_uint32_t*)&packet->state,
      IREE_HAL_AMDGPU_FEEDBACK_PACKET_STATE_READY,
      iree_amdgpu_memory_order_release, iree_amdgpu_memory_scope_system);
#if defined(IREE_AMDGPU_TARGET_DEVICE)
  if (!iree_hsa_signal_is_null(config->notify_signal)) {
    iree_hsa_signal_add(config->notify_signal, 1,
                        iree_amdgpu_memory_order_release);
  }
#else
  (void)config;
#endif  // IREE_AMDGPU_TARGET_DEVICE
}

#endif  // IREE_HAL_DRIVERS_AMDGPU_DEVICE_SUPPORT_FEEDBACK_H_
