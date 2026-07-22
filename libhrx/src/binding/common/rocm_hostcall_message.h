// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_STREAMING_ROCM_HOSTCALL_MESSAGE_H_
#define IREE_HAL_STREAMING_ROCM_HOSTCALL_MESSAGE_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Maximum payload retained for one unterminated device message.
#define IREE_HAL_STREAMING_HOSTCALL_MAX_MESSAGE_BYTES (1024u * 1024u)
// Maximum aggregate allocation retained by one message table.
#define IREE_HAL_STREAMING_HOSTCALL_MAX_MESSAGE_TABLE_BYTES \
  (64u * 1024u * 1024u)

typedef struct iree_hal_streaming_hostcall_message_t {
  // Accumulated qword payload for one in-flight device-library message.
  uint64_t* data;
  // Number of qwords currently accumulated in |data|.
  iree_host_size_t count;
  // Capacity of |data| in qwords.
  iree_host_size_t capacity;
  // Next reusable message index when this entry is not live.
  iree_host_size_t next_free;
  // True while this message ID is active.
  bool live;
} iree_hal_streaming_hostcall_message_t;

typedef struct iree_hal_streaming_hostcall_message_table_t {
  // Indexed message table. Device descriptors carry the table index.
  iree_hal_streaming_hostcall_message_t* messages;
  // Number of entries in |messages|.
  iree_host_size_t count;
  // Capacity of |messages|.
  iree_host_size_t capacity;
  // Head of the reusable message index list, or IREE_HOST_SIZE_MAX.
  iree_host_size_t free_head;
  // Total bytes allocated for all message payloads.
  iree_host_size_t allocated_payload_bytes;
  // Allocator used for table and message payload storage.
  iree_allocator_t host_allocator;
} iree_hal_streaming_hostcall_message_table_t;

void iree_hal_streaming_hostcall_message_table_initialize(
    iree_allocator_t host_allocator,
    iree_hal_streaming_hostcall_message_table_t* out_table);

void iree_hal_streaming_hostcall_message_table_deinitialize(
    iree_hal_streaming_hostcall_message_table_t* table);

// Processes one device-library printf message fragment in-place. Returns false
// and writes -1 to the result descriptor when the fragment is malformed or
// exceeds the bounded host allocation budget.
bool iree_hal_streaming_hostcall_message_handle_printf(
    iree_hal_streaming_hostcall_message_table_t* table, uint64_t* payload);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_STREAMING_ROCM_HOSTCALL_MESSAGE_H_
