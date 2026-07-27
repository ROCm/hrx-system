// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU kernel feedback ABI constants used by Loom code generation.
//
// This header intentionally mirrors only the device-visible layout facts needed
// by the compiler. It must not include runtime/src/iree/hal/drivers/amdgpu/abi
// headers because their host side includes HSA headers, while the Loom compiler
// should stay independent from HSA except in execution/tooling code.

#ifndef LOOM_TARGET_ARCH_AMDGPU_ABI_FEEDBACK_H_
#define LOOM_TARGET_ARCH_AMDGPU_ABI_FEEDBACK_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_AMDGPU_FEEDBACK_CONFIG_ABI_VERSION 0u
#define LOOM_AMDGPU_FEEDBACK_CHANNEL_ABI_VERSION 0u

#define LOOM_AMDGPU_FEEDBACK_PACKET_ALIGNMENT 64u
#define LOOM_AMDGPU_FEEDBACK_PACKET_MAX_PAYLOAD_LENGTH (16u * 1024u)

// Bitset of loom_amdgpu_feedback_config_flag_bits_e values.
typedef uint32_t loom_amdgpu_feedback_config_flags_t;

enum loom_amdgpu_feedback_config_flag_bits_e {
  // Device feedback packets are disabled.
  LOOM_AMDGPU_FEEDBACK_CONFIG_FLAG_NONE = 0u,
  // Device feedback packets are accepted by the owning physical device.
  LOOM_AMDGPU_FEEDBACK_CONFIG_FLAG_ENABLED = 1u << 0,
};

// Feedback packet state written by device producers and read by host drains.
typedef uint32_t loom_amdgpu_feedback_packet_state_t;

enum loom_amdgpu_feedback_packet_state_e {
  // Packet storage has not been published for host consumption.
  LOOM_AMDGPU_FEEDBACK_PACKET_STATE_RESERVED = 0u,
  // Packet header and payload are ready for host consumption.
  LOOM_AMDGPU_FEEDBACK_PACKET_STATE_READY = 1u,
};

// Feedback packet schema discriminator.
typedef uint16_t loom_amdgpu_feedback_packet_kind_t;

enum loom_amdgpu_feedback_packet_kind_e {
  // Reserved packet kind.
  LOOM_AMDGPU_FEEDBACK_PACKET_KIND_NONE = 0u,
  // Address-sanitizer diagnostic packet.
  LOOM_AMDGPU_FEEDBACK_PACKET_KIND_ASAN = 1u,
  // Formatted device printf packet.
  LOOM_AMDGPU_FEEDBACK_PACKET_KIND_PRINTF = 2u,
  // Device-to-host service call packet.
  LOOM_AMDGPU_FEEDBACK_PACKET_KIND_HOST_CALL = 3u,
  // Thread-sanitizer diagnostic packet.
  LOOM_AMDGPU_FEEDBACK_PACKET_KIND_TSAN = 4u,
  // First packet kind reserved for user-defined packet schemas.
  LOOM_AMDGPU_FEEDBACK_PACKET_KIND_USER = 0x8000u,
};

// Bitset of loom_amdgpu_feedback_packet_flag_bits_e values.
typedef uint32_t loom_amdgpu_feedback_packet_flags_t;

enum loom_amdgpu_feedback_packet_flag_bits_e {
  // No packet-level flags are set.
  LOOM_AMDGPU_FEEDBACK_PACKET_FLAG_NONE = 0u,
  // The device producer does not expect a host response.
  LOOM_AMDGPU_FEEDBACK_PACKET_FLAG_ASYNC = 1u << 0,
};

enum loom_amdgpu_feedback_config_layout_e {
  LOOM_AMDGPU_FEEDBACK_CONFIG_BYTE_LENGTH = 64u,
  LOOM_AMDGPU_FEEDBACK_CONFIG_RECORD_LENGTH_OFFSET = 0u,
  LOOM_AMDGPU_FEEDBACK_CONFIG_ABI_VERSION_OFFSET = 4u,
  LOOM_AMDGPU_FEEDBACK_CONFIG_FLAGS_OFFSET = 8u,
  LOOM_AMDGPU_FEEDBACK_CONFIG_CHANNEL_BASE_OFFSET = 16u,
  LOOM_AMDGPU_FEEDBACK_CONFIG_NOTIFY_SIGNAL_OFFSET = 24u,
  LOOM_AMDGPU_FEEDBACK_CONFIG_SOURCE_CONTEXT_OFFSET = 32u,
};

enum loom_amdgpu_feedback_channel_layout_e {
  LOOM_AMDGPU_FEEDBACK_CHANNEL_BYTE_LENGTH = 64u,
  LOOM_AMDGPU_FEEDBACK_CHANNEL_RECORD_LENGTH_OFFSET = 0u,
  LOOM_AMDGPU_FEEDBACK_CHANNEL_ABI_VERSION_OFFSET = 4u,
  LOOM_AMDGPU_FEEDBACK_CHANNEL_FLAGS_OFFSET = 8u,
  LOOM_AMDGPU_FEEDBACK_CHANNEL_RING_BASE_OFFSET = 16u,
  LOOM_AMDGPU_FEEDBACK_CHANNEL_RING_CAPACITY_OFFSET = 24u,
  LOOM_AMDGPU_FEEDBACK_CHANNEL_READ_TAIL_OFFSET = 32u,
  LOOM_AMDGPU_FEEDBACK_CHANNEL_RESERVATION_HEAD_OFFSET = 40u,
  LOOM_AMDGPU_FEEDBACK_CHANNEL_DROPPED_PACKET_COUNT_OFFSET = 48u,
};

enum loom_amdgpu_feedback_packet_layout_e {
  LOOM_AMDGPU_FEEDBACK_PACKET_BYTE_LENGTH = 64u,
  LOOM_AMDGPU_FEEDBACK_PACKET_RECORD_LENGTH_OFFSET = 0u,
  LOOM_AMDGPU_FEEDBACK_PACKET_HEADER_LENGTH_OFFSET = 4u,
  LOOM_AMDGPU_FEEDBACK_PACKET_KIND_OFFSET = 6u,
  LOOM_AMDGPU_FEEDBACK_PACKET_FLAGS_OFFSET = 8u,
  LOOM_AMDGPU_FEEDBACK_PACKET_STATE_OFFSET = 12u,
  LOOM_AMDGPU_FEEDBACK_PACKET_SEQUENCE_OFFSET = 16u,
  LOOM_AMDGPU_FEEDBACK_PACKET_SOURCE_DISPATCH_PTR_OFFSET = 24u,
  LOOM_AMDGPU_FEEDBACK_PACKET_SOURCE_WORKGROUP_ID_X_OFFSET = 32u,
  LOOM_AMDGPU_FEEDBACK_PACKET_SOURCE_WORKITEM_ID_X_OFFSET = 36u,
  LOOM_AMDGPU_FEEDBACK_PACKET_SOURCE_CONTEXT_OFFSET = 40u,
  LOOM_AMDGPU_FEEDBACK_PACKET_RESERVED_ARRAY_0_OFFSET = 48u,
  LOOM_AMDGPU_FEEDBACK_PACKET_RESERVED_ARRAY_1_OFFSET = 56u,
};

// Rounds |value| up to the required feedback packet alignment.
static inline uint64_t loom_amdgpu_feedback_align_packet_length(
    uint64_t value) {
  return (value + (LOOM_AMDGPU_FEEDBACK_PACKET_ALIGNMENT - 1u)) &
         ~(uint64_t)(LOOM_AMDGPU_FEEDBACK_PACKET_ALIGNMENT - 1u);
}

// Returns the total packet byte length required for |payload_length|.
static inline uint64_t loom_amdgpu_feedback_packet_length(
    uint64_t payload_length) {
  return loom_amdgpu_feedback_align_packet_length(
      LOOM_AMDGPU_FEEDBACK_PACKET_BYTE_LENGTH + payload_length);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_ABI_FEEDBACK_H_
