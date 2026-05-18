// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// HAL remote protocol: profiling bulk transfer payloads.
//
// Server-side profiling uses the bulk channel to relay HAL profile sink
// callbacks to the client-owned sink. Each profile callback is serialized as
// one server-originated bulk transfer. The transfer payload carries both the
// callback metadata and callback bytes so the client can create receive state
// from the bulk START frame without depending on a separately ordered control
// notification.

#ifndef IREE_HAL_REMOTE_PROTOCOL_PROFILE_H_
#define IREE_HAL_REMOTE_PROTOCOL_PROFILE_H_

#include "iree/hal/remote/protocol/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Profile sink callback encoded by a bulk transfer.
typedef uint8_t iree_hal_remote_profile_callback_type_t;
enum iree_hal_remote_profile_callback_type_e {
  IREE_HAL_REMOTE_PROFILE_CALLBACK_TYPE_BEGIN_SESSION = 0x01,
  IREE_HAL_REMOTE_PROFILE_CALLBACK_TYPE_WRITE_CHUNK = 0x02,
  IREE_HAL_REMOTE_PROFILE_CALLBACK_TYPE_END_SESSION = 0x03,
};

// Header at the start of every profile bulk transfer payload.
//
// The transfer payload layout is:
//   iree_hal_remote_profile_transfer_header_t header
//   char content_type[content_type_length]  (padded to 8-byte alignment)
//   char name[name_length]  (padded to 8-byte alignment)
//   uint8_t payload[payload_length]
//
// For BEGIN_SESSION and END_SESSION transfers |payload_length| is usually 0.
// For END_SESSION transfers |session_status_code| carries the producer-side
// session status code. For other transfer kinds it must be IREE_STATUS_OK.
typedef struct iree_hal_remote_profile_transfer_header_t {
  // Profile callback sequence number within the active remote session.
  uint64_t sequence;
  // Process-local profiling session identifier from the server HAL.
  uint64_t session_id;
  // Producer-defined stream identifier within |session_id|.
  uint64_t stream_id;
  // Producer-defined event identifier associated with this chunk, or 0.
  uint64_t event_id;
  // Session-local executable identifier associated with this chunk, or 0.
  uint64_t executable_id;
  // Session-local command-buffer identifier associated with this chunk, or 0.
  uint64_t command_buffer_id;
  // iree_hal_profile_chunk_flags_t bits.
  uint64_t chunk_flags;
  // Number of producer-side records omitted from this chunk stream.
  uint64_t dropped_record_count;
  // Byte length of the flattened profile payload following metadata strings.
  uint64_t payload_length;
  // iree_status_code_t for END_SESSION transfers; otherwise 0.
  uint32_t session_status_code;
  // Physical device ordinal associated with this callback, or UINT32_MAX.
  uint32_t physical_device_ordinal;
  // Queue ordinal associated with this callback, or UINT32_MAX.
  uint32_t queue_ordinal;
  // Byte length of the content type string following this header.
  uint16_t content_type_length;
  // Byte length of the human-readable stream/artifact name string.
  uint16_t name_length;
  // iree_hal_remote_profile_callback_type_t.
  uint8_t callback_type;
  // Must be 0.
  uint8_t reserved[7];
} iree_hal_remote_profile_transfer_header_t;
static_assert(sizeof(iree_hal_remote_profile_transfer_header_t) == 96, "");
static_assert(offsetof(iree_hal_remote_profile_transfer_header_t, sequence) ==
                  0,
              "");
static_assert(offsetof(iree_hal_remote_profile_transfer_header_t,
                       payload_length) == 64,
              "");
static_assert(offsetof(iree_hal_remote_profile_transfer_header_t,
                       session_status_code) == 72,
              "");
static_assert(offsetof(iree_hal_remote_profile_transfer_header_t,
                       callback_type) == 88,
              "");

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_PROTOCOL_PROFILE_H_
