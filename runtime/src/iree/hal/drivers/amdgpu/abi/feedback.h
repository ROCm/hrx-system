// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Device-visible feedback channel ABI shared by AMDGPU host code and
// instrumented device code.

#ifndef IREE_HAL_DRIVERS_AMDGPU_ABI_FEEDBACK_H_
#define IREE_HAL_DRIVERS_AMDGPU_ABI_FEEDBACK_H_

#include "iree/hal/drivers/amdgpu/abi/signal.h"

// Name of the executable global containing |iree_hal_amdgpu_feedback_config_t|.
#define IREE_HAL_AMDGPU_FEEDBACK_CONFIG_GLOBAL_NAME "iree_feedback_config"

// ABI version for |iree_hal_amdgpu_feedback_config_t|.
#define IREE_HAL_AMDGPU_FEEDBACK_CONFIG_ABI_VERSION_0 0u

// ABI version for |iree_hal_amdgpu_feedback_channel_header_t|.
#define IREE_HAL_AMDGPU_FEEDBACK_CHANNEL_ABI_VERSION_0 0u

// Required byte alignment for all packets in the feedback ring.
#define IREE_HAL_AMDGPU_FEEDBACK_PACKET_ALIGNMENT 64u

// Maximum payload byte length accepted by the device-side reservation helper.
#define IREE_HAL_AMDGPU_FEEDBACK_PACKET_MAX_PAYLOAD_LENGTH (16u * 1024u)

// Default feedback ring capacity in bytes.
#define IREE_HAL_AMDGPU_FEEDBACK_DEFAULT_RING_CAPACITY (2u * 1024u * 1024u)

// Bitfield specifying properties of the feedback configuration.
typedef uint32_t iree_hal_amdgpu_feedback_config_flags_t;
enum iree_hal_amdgpu_feedback_config_flag_bits_t {
  IREE_HAL_AMDGPU_FEEDBACK_CONFIG_FLAG_NONE = 0u,
  // Device feedback packets are accepted by the owning physical device.
  IREE_HAL_AMDGPU_FEEDBACK_CONFIG_FLAG_ENABLED = 1u << 0,
};

// Feedback packet state written by device producers and read by the host
// service thread.
typedef uint32_t iree_hal_amdgpu_feedback_packet_state_t;
enum iree_hal_amdgpu_feedback_packet_state_bits_t {
  // Packet storage has not been published for host consumption.
  IREE_HAL_AMDGPU_FEEDBACK_PACKET_STATE_RESERVED = 0u,
  // Packet header and payload are ready for host consumption.
  IREE_HAL_AMDGPU_FEEDBACK_PACKET_STATE_READY = 1u,
};

// Feedback packet kind. Higher-level facilities own their payload schema.
typedef uint16_t iree_hal_amdgpu_feedback_packet_kind_t;
enum iree_hal_amdgpu_feedback_packet_kind_bits_t {
  // Reserved packet kind.
  IREE_HAL_AMDGPU_FEEDBACK_PACKET_KIND_NONE = 0u,
  // Address-sanitizer diagnostic packet.
  IREE_HAL_AMDGPU_FEEDBACK_PACKET_KIND_ASAN = 1u,
  // Formatted device printf packet.
  IREE_HAL_AMDGPU_FEEDBACK_PACKET_KIND_PRINTF = 2u,
  // Device-to-host service call packet.
  IREE_HAL_AMDGPU_FEEDBACK_PACKET_KIND_HOST_CALL = 3u,
  // Thread-sanitizer diagnostic packet.
  IREE_HAL_AMDGPU_FEEDBACK_PACKET_KIND_TSAN = 4u,
  // First packet kind reserved for user-defined packet schemas.
  IREE_HAL_AMDGPU_FEEDBACK_PACKET_KIND_USER = 0x8000u,
};

// Bitfield specifying packet-level properties.
typedef uint32_t iree_hal_amdgpu_feedback_packet_flags_t;
enum iree_hal_amdgpu_feedback_packet_flag_bits_t {
  IREE_HAL_AMDGPU_FEEDBACK_PACKET_FLAG_NONE = 0u,
  // The device producer does not expect a host response.
  IREE_HAL_AMDGPU_FEEDBACK_PACKET_FLAG_ASYNC = 1u << 0,
};

// Runtime-published feedback configuration read by instrumented device code.
typedef struct IREE_AMDGPU_ALIGNAS(8) iree_hal_amdgpu_feedback_config_t {
  // Size of this record in bytes for forward-compatible parsing.
  uint32_t record_length;
  // ABI version of this record layout.
  uint32_t abi_version;
  // Flags describing enabled feedback facilities.
  iree_hal_amdgpu_feedback_config_flags_t flags;
  // Reserved padding for 8-byte alignment. Must be zero.
  uint32_t reserved0;
  // Device-visible pointer to |iree_hal_amdgpu_feedback_channel_header_t|.
  uint64_t channel_base;
  // Host interrupt signal used by device producers after publishing packets.
  iree_hsa_signal_t notify_signal;
  // Opaque host source-context value copied into feedback packets.
  //
  // Device producers must treat this as an uninterpreted value. Host feedback
  // handling resolves it back to executable-owned source metadata while the
  // submitting executable is still retained by the retiring work.
  uint64_t source_context;
  // Reserved for future feedback configuration state. Must be zero.
  uint64_t reserved[3];
} iree_hal_amdgpu_feedback_config_t;
IREE_AMDGPU_STATIC_ASSERT(sizeof(iree_hal_amdgpu_feedback_config_t) == 64,
                          "feedback config size is part of the device ABI");
IREE_AMDGPU_STATIC_ASSERT(
    IREE_AMDGPU_OFFSETOF(iree_hal_amdgpu_feedback_config_t, source_context) ==
        32,
    "feedback config source context offset is part of the device ABI");

// Device-visible control block for one feedback ring.
typedef struct IREE_AMDGPU_ALIGNAS(64)
    iree_hal_amdgpu_feedback_channel_header_t {
  // Size of this record in bytes for forward-compatible parsing.
  uint32_t record_length;
  // ABI version of this record layout.
  uint32_t abi_version;
  // Flags describing enabled feedback channel behavior.
  iree_hal_amdgpu_feedback_config_flags_t flags;
  // Reserved padding for 8-byte alignment. Must be zero.
  uint32_t reserved0;
  // Device-visible base pointer of the packet storage ring.
  uint64_t ring_base;
  // Packet storage ring capacity in bytes. Must be a power of two.
  uint64_t ring_capacity;
  // Monotonic byte position of the first unconsumed packet.
  volatile uint64_t read_tail;
  // Monotonic byte position one past the last reserved packet byte.
  volatile uint64_t reservation_head;
  // Number of packets dropped by device producers due to reservation failure.
  volatile uint64_t dropped_packet_count;
  // Reserved for future channel state. Must be zero.
  uint64_t reserved[1];
} iree_hal_amdgpu_feedback_channel_header_t;
IREE_AMDGPU_STATIC_ASSERT(
    sizeof(iree_hal_amdgpu_feedback_channel_header_t) == 64,
    "feedback channel header size is part of the device ABI");

// Header preceding every variable-size feedback packet.
typedef struct IREE_AMDGPU_ALIGNAS(64) iree_hal_amdgpu_feedback_packet_t {
  // Total packet byte length including this header and payload padding.
  uint32_t record_length;
  // Byte length of this packet header.
  uint16_t header_length;
  // Packet schema discriminator.
  iree_hal_amdgpu_feedback_packet_kind_t kind;
  // Flags describing packet-level behavior.
  iree_hal_amdgpu_feedback_packet_flags_t flags;
  // Publication state of this packet.
  volatile iree_hal_amdgpu_feedback_packet_state_t state;
  // Absolute ring byte position assigned by reservation.
  uint64_t sequence;
  // Device-visible dispatch packet pointer captured at reservation time.
  uint64_t source_dispatch_ptr;
  // X dimension workgroup id captured at reservation time.
  uint32_t source_workgroup_id_x;
  // X dimension workitem id captured at reservation time.
  uint32_t source_workitem_id_x;
  // Opaque host source-context value copied from the feedback config.
  uint64_t source_context;
  // Reserved for future packet fields. Must be zero.
  uint64_t reserved[2];
} iree_hal_amdgpu_feedback_packet_t;
IREE_AMDGPU_STATIC_ASSERT(sizeof(iree_hal_amdgpu_feedback_packet_t) == 64,
                          "feedback packet header size is part of the ABI");
IREE_AMDGPU_STATIC_ASSERT(
    IREE_AMDGPU_OFFSETOF(iree_hal_amdgpu_feedback_packet_t, source_context) ==
        40,
    "feedback packet source context offset is part of the device ABI");

// Rounds |value| up to the required feedback packet alignment.
static inline size_t iree_hal_amdgpu_feedback_align_packet_length(
    size_t value) {
  return (value + (IREE_HAL_AMDGPU_FEEDBACK_PACKET_ALIGNMENT - 1u)) &
         ~(size_t)(IREE_HAL_AMDGPU_FEEDBACK_PACKET_ALIGNMENT - 1u);
}

// Returns the total packet byte length required for |payload_length|.
static inline size_t iree_hal_amdgpu_feedback_packet_length(
    size_t payload_length) {
  return iree_hal_amdgpu_feedback_align_packet_length(
      sizeof(iree_hal_amdgpu_feedback_packet_t) + payload_length);
}

#endif  // IREE_HAL_DRIVERS_AMDGPU_ABI_FEEDBACK_H_
