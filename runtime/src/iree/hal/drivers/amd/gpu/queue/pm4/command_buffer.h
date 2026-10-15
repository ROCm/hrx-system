// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_COMMAND_BUFFER_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_COMMAND_BUFFER_H_

#include "iree/hal/drivers/amd/gpu/queue/pm4/barrier.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates an immutable native recording for an exact PM4 family. Recording
// prepares packets, argument templates, and address relocations without needing
// committed buffer addresses. UNRETAINED controls direct-resource retention;
// each invocation independently owns its captured table and publication cursor.
// No native allocation or additional worker belongs to the recording.
iree_status_t iree_hal_amd_gpu_pm4_command_buffer_create(
    const iree_hal_queue_family_t* family, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t categories, iree_host_size_t binding_capacity,
    uint32_t maximum_group_segment_byte_length,
    const iree_hal_amd_gpu_pm4_barrier_profile_t* profile,
    iree_allocator_t host_allocator, iree_hal_command_buffer_t** out_buffer);

typedef struct iree_hal_amd_gpu_pm4_record_block_t
    iree_hal_amd_gpu_pm4_record_block_t;

// Exclusive publication position into a retained immutable recording. Records
// are dense within blocks; crossing a block consumes one linked-list pointer.
typedef struct iree_hal_amd_gpu_pm4_command_cursor_t {
  // Current block, or NULL after the complete program has been published.
  const iree_hal_amd_gpu_pm4_record_block_t* block;
  // Byte offset of the next record from the beginning of the block payload.
  uint32_t offset;
  // Published bytes of a DMA record spanning multiple native pieces.
  uint64_t transfer_offset;
} iree_hal_amd_gpu_pm4_command_cursor_t;

// Captured, qualified binding-table entry. Native addresses remain unread until
// dependencies admit publication, including queue-ordered allocation
// commitment.
typedef struct iree_hal_amd_gpu_pm4_command_binding_t {
  // Owned buffer unless invocation lifetime was explicitly borrowed.
  iree_hal_buffer_t* buffer;
  // Byte offset from the logical beginning of buffer.
  iree_device_size_t offset;
  // Captured logical extent, including resolution of WHOLE_BUFFER.
  iree_device_size_t length;
  // Prepared native slots for the queue (0) and shader program (1) sites.
  iree_hal_buffer_native_binding_slot_t slots[2];
} iree_hal_amd_gpu_pm4_command_binding_t;

// Invocation state embedded in the owning submission capture. Its table storage
// immediately follows the capture's common arrays, not the immutable recording.
typedef struct iree_hal_amd_gpu_pm4_command_invocation_t {
  // Retained recording, independent of caller ownership and other invocations.
  iree_hal_command_buffer_t* command_buffer;
  // Captured table indexed by the recording's indirect binding slots.
  iree_hal_amd_gpu_pm4_command_binding_t* bindings;
  // Exclusively owned unpublished suffix; retirement never modifies it.
  iree_hal_amd_gpu_pm4_command_cursor_t cursor;
  // First publication owns external wait packets and a single-record budget.
  bool first;
  // Binding buffers are borrowed; table entries themselves are always copied.
  bool borrow_bindings;
} iree_hal_amd_gpu_pm4_command_invocation_t;

// Initializes an invocation in caller-owned capture storage. The table has
// passed public HAL submission validation. storage contains binding_count
// command_binding_t entries. Failure leaves a deinitializable prefix;
// successful initialization retains the recording and every used, non-borrowed
// buffer.
iree_status_t iree_hal_amd_gpu_pm4_command_invocation_initialize(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t bindings, void* storage,
    iree_hal_queue_execute_flags_t flags,
    iree_hal_amd_gpu_pm4_command_invocation_t* out_invocation);

// Releases resources after every published prefix has completed, or before any
// publication. Failure of an unpublished suffix does not prove native
// completion.
void iree_hal_amd_gpu_pm4_command_invocation_deinitialize(
    iree_hal_amd_gpu_pm4_command_invocation_t* invocation);

// Minimum requirement to progress the next record, or {0, 1} for an empty
// suffix. Dispatches need their complete arguments; transfers can use a smaller
// piece. This constant-time query lets the transport wrap or grow its argument
// FIFO without inspecting the rest of the program.
typedef struct iree_hal_amd_gpu_pm4_command_arguments_t {
  // Minimum native argument/staging bytes needed by the next record.
  uint32_t length;
  // Required power-of-two GPU address alignment.
  uint32_t alignment;
} iree_hal_amd_gpu_pm4_command_arguments_t;

iree_hal_amd_gpu_pm4_command_arguments_t
iree_hal_amd_gpu_pm4_command_next_arguments(
    iree_hal_amd_gpu_pm4_command_cursor_t cursor);

enum { IREE_HAL_AMD_GPU_PM4_COMMAND_MAX_WORDS = 1024 };

// One bounded, invocation-private native prefix. The transport commits next
// only after reserving its ring and completion tail. Preparing a candidate
// never mutates the recording, advances the invocation, or allocates native
// memory.
typedef struct iree_hal_amd_gpu_pm4_command_chunk_t {
  // Complete native packet prefix with addresses relocated for this invocation.
  uint32_t words[IREE_HAL_AMD_GPU_PM4_COMMAND_MAX_WORDS];
  // Initialized native extent in DWORDs.
  uint32_t word_count;
  // Argument bytes written, including alignment holes between records.
  uint32_t argument_length;
  // Number of dispatches, used to advance the queue's native dispatch IDs.
  uint32_t dispatch_count;
  // DMA in this prefix still needs joining before staging and buffers retire.
  bool pending_dma;
  // Unpublished suffix, potentially partway through a native DMA span.
  iree_hal_amd_gpu_pm4_command_cursor_t next;
} iree_hal_amd_gpu_pm4_command_chunk_t;

// Copies admitted native templates, relocates captured addresses, and emits
// bounded native DMA spans. No public HAL operation is replayed.
// word_capacity is at most COMMAND_MAX_WORDS. Available argument bytes are a
// contiguous FIFO window whose address satisfies next_arguments().alignment.
// A zero-word result means capacity is unavailable or the program is empty;
// next.block distinguishes these cases. Failures are uncommitted native backing
// or invalid externally supplied native-address alignment, never ABI
// rediscovery.
iree_status_t iree_hal_amd_gpu_pm4_command_prepare_chunk(
    const iree_hal_amd_gpu_pm4_command_invocation_t* invocation,
    uint32_t word_capacity, iree_byte_span_t arguments,
    uint64_t argument_address, uint64_t dispatch_id,
    iree_hal_amd_gpu_pm4_command_chunk_t* out_chunk);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_COMMAND_BUFFER_H_
