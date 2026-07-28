// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HRX_BINDING_HIP_HOSTCALL_MESSAGE_H_
#define HRX_BINDING_HIP_HOSTCALL_MESSAGE_H_

#include <stdint.h>

#include "binding/hip/hostcall_packet.h"
#include "iree/base/api.h"
#include "iree/base/internal/arena.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Lifecycle state for one device-library message table entry.
typedef enum iree_hip_hostcall_message_state_e {
  // Entry is available for reuse.
  IREE_HIP_HOSTCALL_MESSAGE_STATE_FREE = 0,
  // Entry accepts continuation fragments.
  IREE_HIP_HOSTCALL_MESSAGE_STATE_ACTIVE = 1,
  // Entry owns a completed message borrowed by the caller.
  IREE_HIP_HOSTCALL_MESSAGE_STATE_COMPLETE = 2,
} iree_hip_hostcall_message_state_t;

// Accumulation state for one device-library message.
typedef struct iree_hip_hostcall_message_t {
  // Oldest pooled payload block.
  iree_arena_block_t* block_head;
  // Newest pooled payload block.
  iree_arena_block_t* block_tail;
  // Number of qwords accumulated across the block chain.
  iree_host_size_t count;
  // Next reusable message index while this entry is free.
  iree_host_size_t next_free;
  // Current entry lifecycle state.
  iree_hip_hostcall_message_state_t state;
} iree_hip_hostcall_message_t;

// Thread-confined table of fragmented HIP device-library messages.
typedef struct iree_hip_hostcall_message_table_t {
  // Indexed message table. Device descriptors carry the table index.
  iree_hip_hostcall_message_t* messages;
  // Number of entries in |messages|.
  iree_host_size_t count;
  // Capacity of |messages|.
  iree_host_size_t capacity;
  // Head of the reusable message index list, or IREE_HOST_SIZE_MAX.
  iree_host_size_t free_head;
  // Shared fixed-size storage for fragmented message payloads.
  iree_arena_block_pool_t block_pool;
  // Allocator used for message table storage.
  iree_allocator_t host_allocator;
} iree_hip_hostcall_message_table_t;

// Outcome of consuming one valid message fragment.
typedef enum iree_hip_hostcall_message_result_type_e {
  // The device must submit another fragment with the returned descriptor.
  IREE_HIP_HOSTCALL_MESSAGE_RESULT_CONTINUE = 0,
  // The complete message is borrowed by the caller until explicitly released.
  IREE_HIP_HOSTCALL_MESSAGE_RESULT_COMPLETE = 1,
} iree_hip_hostcall_message_result_type_t;

// Result produced by one message fragment.
typedef struct iree_hip_hostcall_message_result_t {
  // Whether this result continues or completes a message.
  iree_hip_hostcall_message_result_type_t type;
  // Descriptor returned to the device for a continued message.
  uint64_t continuation_descriptor;
  // Stable table identifier for a complete borrowed message.
  iree_host_size_t message_id;
  // Number of qwords in the complete message.
  iree_host_size_t count;
} iree_hip_hostcall_message_result_t;

// Initializes |out_table|.
//
// The allocator must remain valid until the table is deinitialized. The table
// is thread-confined and requires external serialization if shared.
void iree_hip_hostcall_message_table_initialize(
    iree_allocator_t host_allocator,
    iree_hip_hostcall_message_table_t* out_table);

void iree_hip_hostcall_message_table_deinitialize(
    iree_hip_hostcall_message_table_t* table);

// Consumes one ROCm device-library message fragment.
//
// The first qword is the stack/tag fragment descriptor and the remaining seven
// qwords are fragment data. A continued result contains the descriptor the
// caller must return to the device. A complete result borrows table-owned data
// until iree_hip_hostcall_message_release is called with its |message_id|.
//
// On failure any live message referenced by a continuation fragment is
// discarded, no result owns storage, and the status describes malformed device
// input or allocation failure.
iree_status_t iree_hip_hostcall_message_consume_fragment(
    iree_hip_hostcall_message_table_t* table,
    const uint64_t fragment[IREE_HIP_HOSTCALL_PACKET_SLOT_QWORD_COUNT],
    iree_hip_hostcall_message_result_t* out_result);

// Copies one complete message into |target|.
//
// |target| must have exactly |count * sizeof(uint64_t)| bytes, where |count| is
// the value returned by the completing fragment. The message remains owned by
// |table| until explicitly released.
void iree_hip_hostcall_message_copy(
    const iree_hip_hostcall_message_table_t* table, iree_host_size_t message_id,
    iree_byte_span_t target);

// Releases one complete message and makes its identifier reusable.
void iree_hip_hostcall_message_release(iree_hip_hostcall_message_table_t* table,
                                       iree_host_size_t message_id);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // HRX_BINDING_HIP_HOSTCALL_MESSAGE_H_
