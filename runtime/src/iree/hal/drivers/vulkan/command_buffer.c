// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/command_buffer.h"

#include <stddef.h>
#include <string.h>

#include "iree/base/internal/arena.h"
#include "iree/hal/drivers/vulkan/barrier.h"
#include "iree/hal/drivers/vulkan/buffer.h"
#include "iree/hal/drivers/vulkan/executable.h"
#include "iree/hal/drivers/vulkan/sparse_buffer.h"
#include "iree/hal/utils/resource_set.h"

typedef enum iree_hal_vulkan_command_type_e {
  IREE_HAL_VULKAN_COMMAND_TYPE_FILL_BUFFER = 0,
  IREE_HAL_VULKAN_COMMAND_TYPE_UPDATE_BUFFER = 1,
  IREE_HAL_VULKAN_COMMAND_TYPE_COPY_BUFFER = 2,
  IREE_HAL_VULKAN_COMMAND_TYPE_DISPATCH = 3,
  IREE_HAL_VULKAN_COMMAND_TYPE_EXECUTION_BARRIER = 4,
  IREE_HAL_VULKAN_COMMAND_TYPE_ATOMIC = 5,
  IREE_HAL_VULKAN_COMMAND_TYPE_BEGIN_DEBUG_GROUP = 6,
  IREE_HAL_VULKAN_COMMAND_TYPE_END_DEBUG_GROUP = 7,
} iree_hal_vulkan_command_type_t;

typedef struct iree_hal_vulkan_command_t {
  // Total byte length of this command record including trailing payload bytes.
  iree_host_size_t record_length;

  // Recorded operation selector.
  iree_hal_vulkan_command_type_t type;
} iree_hal_vulkan_command_t;

typedef struct iree_hal_vulkan_command_fill_buffer_t {
  // Target buffer reference captured during recording.
  iree_hal_buffer_ref_t target_ref;

  // Fill pattern bytes captured during recording.
  uint8_t pattern[4];

  // Number of bytes in pattern.
  iree_host_size_t pattern_length;

  // HAL fill flags captured during recording.
  iree_hal_fill_flags_t flags;

  // Native implementation selected while recording.
  iree_hal_vulkan_transfer_strategy_t strategy;
} iree_hal_vulkan_command_fill_buffer_t;

typedef struct iree_hal_vulkan_command_update_buffer_t {
  // Source bytes copied from the recording caller into the command record.
  void* source_data;

  // Target buffer reference captured during recording.
  iree_hal_buffer_ref_t target_ref;

  // Number of bytes copied into source_data.
  iree_host_size_t source_data_length;

  // HAL update flags captured during recording.
  iree_hal_update_flags_t flags;

  // Native implementation selected while recording.
  iree_hal_vulkan_transfer_strategy_t strategy;
} iree_hal_vulkan_command_update_buffer_t;

typedef struct iree_hal_vulkan_command_copy_buffer_t {
  // Source buffer reference captured during recording.
  iree_hal_buffer_ref_t source_ref;

  // Target buffer reference captured during recording.
  iree_hal_buffer_ref_t target_ref;

  // HAL copy flags captured during recording.
  iree_hal_copy_flags_t flags;

  // Native implementation selected while recording.
  iree_hal_vulkan_transfer_strategy_t strategy;
} iree_hal_vulkan_command_copy_buffer_t;

typedef struct iree_hal_vulkan_command_dispatch_t {
  // Executable captured during recording.
  // Retained by resource_set unless command buffer mode is UNRETAINED.
  iree_hal_executable_t* executable;

  // Pipeline metadata borrowed from executable and stable while retained.
  const iree_hal_vulkan_pipeline_t* pipeline;

  // Executable function ordinal captured during recording.
  iree_hal_executable_function_t function_ordinal;

  // Dispatch workgroup configuration captured during recording.
  iree_hal_dispatch_config_t config;

  // Number of trailing push constant bytes copied after the fixed payload.
  iree_host_size_t constants_data_length;

  // Number of trailing buffer references copied after push constants.
  iree_host_size_t binding_count;

  // HAL dispatch flags captured during recording.
  iree_hal_dispatch_flags_t flags;
} iree_hal_vulkan_command_dispatch_t;

// Capture-time state mapping logical transfer dependencies to their concrete
// Vulkan pipeline stages.
typedef struct iree_hal_vulkan_command_transfer_barrier_t {
  // Source and target stage expansions required by this dependency.
  iree_hal_vulkan_barrier_flags_t flags;

  // Next dependency awaiting a future compute transfer target, or NULL.
  struct iree_hal_vulkan_command_transfer_barrier_t* next_pending;
} iree_hal_vulkan_command_transfer_barrier_t;

typedef struct iree_hal_vulkan_command_execution_barrier_t {
  // Source HAL execution stages captured during recording.
  iree_hal_execution_stage_t source_stage_mask;

  // Target HAL execution stages captured during recording.
  iree_hal_execution_stage_t target_stage_mask;

  // HAL barrier flags captured during recording.
  iree_hal_execution_barrier_flags_t flags;

  // Number of memory barriers represented by the native barrier.
  iree_host_size_t memory_barrier_count;

  // Number of buffer barriers represented by the native barrier.
  iree_host_size_t buffer_barrier_count;

  // Concrete transfer stages participating in this dependency.
  iree_hal_vulkan_command_transfer_barrier_t transfer_barrier;
} iree_hal_vulkan_command_execution_barrier_t;

typedef struct iree_hal_vulkan_command_atomic_t {
  // Source HAL execution stages captured during recording.
  iree_hal_execution_stage_t source_stage_mask;

  // Target HAL execution stages captured during recording.
  iree_hal_execution_stage_t target_stage_mask;

  // Target buffer reference captured during recording.
  iree_hal_buffer_ref_t target_ref;

  // Validated operation encoded for the built-in shader.
  iree_hal_vulkan_atomic_params_t params;

  // Concrete transfer stages participating in the atomic dependencies.
  iree_hal_vulkan_command_transfer_barrier_t transfer_barrier;
} iree_hal_vulkan_command_atomic_t;

typedef struct iree_hal_vulkan_command_begin_debug_group_t {
  // Debug label color captured during recording.
  iree_hal_label_color_t label_color;
} iree_hal_vulkan_command_begin_debug_group_t;

typedef struct iree_hal_vulkan_command_block_t {
  // Next command block in recording order.
  struct iree_hal_vulkan_command_block_t* next;

  // Number of bytes populated in the block data segment.
  iree_host_size_t data_length;

  // Capacity in bytes of the block data segment.
  iree_host_size_t capacity;
} iree_hal_vulkan_command_block_t;

typedef enum iree_hal_vulkan_command_buffer_state_e {
  IREE_HAL_VULKAN_COMMAND_BUFFER_STATE_INITIAL = 0,
  IREE_HAL_VULKAN_COMMAND_BUFFER_STATE_RECORDING = 1,
  IREE_HAL_VULKAN_COMMAND_BUFFER_STATE_ENDED = 2,
} iree_hal_vulkan_command_buffer_state_t;

typedef struct iree_hal_vulkan_command_buffer_t {
  iree_hal_command_buffer_t base;

  // Host allocator used for command-buffer object and command array storage.
  iree_allocator_t host_allocator;

  // Device-owned atomic pipelines borrowed for record-time validation.
  const iree_hal_vulkan_atomic_pipelines_t* atomic_pipelines;

  // Arena used for variable-length command payload storage.
  iree_arena_allocator_t payload_arena;

  // Current recording lifecycle state.
  iree_hal_vulkan_command_buffer_state_t state;

  // Retained resources referenced by recorded commands.
  // NULL only when the command buffer mode is UNRETAINED.
  iree_hal_resource_set_t* resource_set;

  // First command block in recording order.
  iree_hal_vulkan_command_block_t* command_block_head;

  // Last command block in recording order.
  iree_hal_vulkan_command_block_t* command_block_tail;

  // Next writable byte in command_block_tail's data segment.
  uint8_t* command_block_next;

  // One byte past command_block_tail's writable data segment.
  uint8_t* command_block_end;

  // Number of commands recorded across all command blocks.
  iree_host_size_t command_count;

  // Number of dispatch commands recorded.
  iree_host_size_t dispatch_count;

  // Native Vulkan queue capabilities required by recorded commands.
  VkQueueFlags required_queue_flags;

  // Whether a compute transfer has already been recorded.
  bool has_compute_transfer;

  // Whether any transfer must resolve its dispatch shape at issue time.
  bool requires_per_issue_recording;

  // Dependencies awaiting a later compute transfer in their target scope.
  iree_hal_vulkan_command_transfer_barrier_t* pending_compute_transfer_barriers;

  // Host-published BDA byte length required to replay recorded commands.
  iree_device_size_t bda_publication_length;
} iree_hal_vulkan_command_buffer_t;

typedef struct iree_hal_vulkan_command_buffer_bda_recording_state_t {
  // Host-visible span reserved for all BDA replay data.
  iree_byte_span_t host_span;

  // Device address corresponding to host_span.data.
  VkDeviceAddress device_address;

  // Next unallocated byte in host_span.
  iree_host_size_t byte_offset;

  // Whether the HOST_WRITE to COMPUTE_SHADER publication barrier was emitted.
  bool barrier_recorded;
} iree_hal_vulkan_command_buffer_bda_recording_state_t;

typedef struct iree_hal_vulkan_command_buffer_iterator_t {
  // Command block currently being traversed.
  const iree_hal_vulkan_command_block_t* block;

  // Byte offset of the next command within block.
  iree_host_size_t block_offset;

  // Index of the next command across the full command buffer.
  iree_host_size_t command_index;
} iree_hal_vulkan_command_buffer_iterator_t;

static const iree_hal_command_buffer_vtable_t
    iree_hal_vulkan_command_buffer_vtable;

static void iree_hal_vulkan_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer);

enum {
  IREE_HAL_VULKAN_COMMAND_BUFFER_INLINE_DESCRIPTOR_SET_CAPACITY = 8,
  IREE_HAL_VULKAN_COMMAND_BUFFER_INLINE_DESCRIPTOR_BINDING_CAPACITY = 16,
  IREE_HAL_VULKAN_COMMAND_BUFFER_MINIMUM_COMMAND_BLOCK_CAPACITY =
      64 * (sizeof(iree_hal_vulkan_command_t) +
            sizeof(iree_hal_vulkan_command_dispatch_t)),
};

static iree_host_size_t iree_hal_vulkan_command_buffer_payload_offset(void) {
  return iree_host_align(sizeof(iree_hal_vulkan_command_t), iree_max_align_t);
}

static iree_status_t iree_hal_vulkan_command_buffer_record_length(
    iree_host_size_t payload_length, iree_host_size_t* out_record_length,
    iree_host_size_t* out_payload_offset) {
  iree_host_size_t payload_offset = 0;
  payload_offset = iree_hal_vulkan_command_buffer_payload_offset();
  iree_host_size_t record_length = 0;
  if (!iree_host_size_checked_add(payload_offset, payload_length,
                                  &record_length) ||
      !iree_host_size_checked_align(record_length, iree_max_align_t,
                                    &record_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan command record size overflows");
  }
  *out_record_length = record_length;
  *out_payload_offset = payload_offset;
  return iree_ok_status();
}

static iree_host_size_t iree_hal_vulkan_command_buffer_block_header_length(
    void) {
  return iree_host_align(sizeof(iree_hal_vulkan_command_block_t),
                         iree_max_align_t);
}

static uint8_t* iree_hal_vulkan_command_buffer_block_data(
    iree_hal_vulkan_command_block_t* block) {
  return (uint8_t*)block + iree_hal_vulkan_command_buffer_block_header_length();
}

static const uint8_t* iree_hal_vulkan_command_buffer_const_block_data(
    const iree_hal_vulkan_command_block_t* block) {
  return (const uint8_t*)block +
         iree_hal_vulkan_command_buffer_block_header_length();
}

static const void* iree_hal_vulkan_command_buffer_const_command_payload(
    const iree_hal_vulkan_command_t* command) {
  return (const uint8_t*)command +
         iree_hal_vulkan_command_buffer_payload_offset();
}

static const iree_hal_vulkan_command_fill_buffer_t*
iree_hal_vulkan_command_fill_buffer_payload(
    const iree_hal_vulkan_command_t* command) {
  return (const iree_hal_vulkan_command_fill_buffer_t*)
      iree_hal_vulkan_command_buffer_const_command_payload(command);
}

static const iree_hal_vulkan_command_update_buffer_t*
iree_hal_vulkan_command_update_buffer_payload(
    const iree_hal_vulkan_command_t* command) {
  return (const iree_hal_vulkan_command_update_buffer_t*)
      iree_hal_vulkan_command_buffer_const_command_payload(command);
}

static const iree_hal_vulkan_command_copy_buffer_t*
iree_hal_vulkan_command_copy_buffer_payload(
    const iree_hal_vulkan_command_t* command) {
  return (const iree_hal_vulkan_command_copy_buffer_t*)
      iree_hal_vulkan_command_buffer_const_command_payload(command);
}

static const iree_hal_vulkan_command_dispatch_t*
iree_hal_vulkan_command_dispatch_payload(
    const iree_hal_vulkan_command_t* command) {
  return (const iree_hal_vulkan_command_dispatch_t*)
      iree_hal_vulkan_command_buffer_const_command_payload(command);
}

static const void* iree_hal_vulkan_command_dispatch_constants_data(
    const iree_hal_vulkan_command_dispatch_t* dispatch) {
  if (dispatch->constants_data_length == 0) return NULL;
  return (const uint8_t*)dispatch +
         iree_host_align(sizeof(*dispatch), iree_alignof(uint32_t));
}

static const iree_hal_buffer_ref_t* iree_hal_vulkan_command_dispatch_bindings(
    const iree_hal_vulkan_command_dispatch_t* dispatch) {
  if (dispatch->binding_count == 0) return NULL;
  const iree_host_size_t constants_offset =
      iree_host_align(sizeof(*dispatch), iree_alignof(uint32_t));
  const iree_host_size_t bindings_offset =
      iree_host_align(constants_offset + dispatch->constants_data_length,
                      iree_alignof(iree_hal_buffer_ref_t));
  return (const iree_hal_buffer_ref_t*)((const uint8_t*)dispatch +
                                        bindings_offset);
}

static const iree_hal_vulkan_command_execution_barrier_t*
iree_hal_vulkan_command_execution_barrier_payload(
    const iree_hal_vulkan_command_t* command) {
  return (const iree_hal_vulkan_command_execution_barrier_t*)
      iree_hal_vulkan_command_buffer_const_command_payload(command);
}

static const iree_hal_vulkan_command_atomic_t*
iree_hal_vulkan_command_atomic_payload(
    const iree_hal_vulkan_command_t* command) {
  return (const iree_hal_vulkan_command_atomic_t*)
      iree_hal_vulkan_command_buffer_const_command_payload(command);
}

static const iree_hal_vulkan_command_begin_debug_group_t*
iree_hal_vulkan_command_begin_debug_group_payload(
    const iree_hal_vulkan_command_t* command) {
  return (const iree_hal_vulkan_command_begin_debug_group_t*)
      iree_hal_vulkan_command_buffer_const_command_payload(command);
}

static const char* iree_hal_vulkan_command_begin_debug_group_label(
    const iree_hal_vulkan_command_begin_debug_group_t* begin_debug_group) {
  return (const char*)begin_debug_group + sizeof(*begin_debug_group);
}

static iree_hal_vulkan_command_buffer_t* iree_hal_vulkan_command_buffer_cast(
    iree_hal_command_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_vulkan_command_buffer_vtable);
  return (iree_hal_vulkan_command_buffer_t*)base_value;
}

static bool iree_hal_vulkan_command_buffer_validates(
    const iree_hal_vulkan_command_buffer_t* command_buffer) {
#if IREE_HAL_COMMAND_BUFFER_VALIDATION_ENABLE
  return !iree_any_bit_set(command_buffer->base.mode,
                           IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED);
#else
  (void)command_buffer;
  return false;
#endif  // IREE_HAL_COMMAND_BUFFER_VALIDATION_ENABLE
}

static iree_status_t iree_hal_vulkan_command_buffer_ensure_command_capacity(
    iree_hal_vulkan_command_buffer_t* command_buffer,
    iree_host_size_t payload_length, iree_host_size_t* out_record_length,
    iree_host_size_t* out_payload_offset) {
  if (command_buffer->command_count == IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan command buffer command count exceeds "
                            "host size");
  }
  iree_host_size_t record_length = 0;
  iree_host_size_t payload_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_record_length(
      payload_length, &record_length, &payload_offset));

  if (!command_buffer->command_block_tail ||
      record_length > (iree_host_size_t)(command_buffer->command_block_end -
                                         command_buffer->command_block_next)) {
    const iree_host_size_t header_length =
        iree_hal_vulkan_command_buffer_block_header_length();
    iree_host_size_t block_pool_capacity =
        IREE_HAL_VULKAN_COMMAND_BUFFER_MINIMUM_COMMAND_BLOCK_CAPACITY;
    const iree_host_size_t aligned_usable_block_size =
        command_buffer->payload_arena.block_pool->usable_block_size &
        ~(iree_host_size_t)(iree_max_align_t - 1);
    if (aligned_usable_block_size > header_length) {
      block_pool_capacity = aligned_usable_block_size - header_length;
    }
    const iree_host_size_t capacity =
        iree_max(record_length, block_pool_capacity);
    iree_host_size_t block_size = 0;
    if (!iree_host_size_checked_add(header_length, capacity, &block_size)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "Vulkan command block size overflows");
    }
    iree_hal_vulkan_command_block_t* block = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate(&command_buffer->payload_arena,
                                             block_size, (void**)&block));
    block->next = NULL;
    block->data_length = 0;
    block->capacity = capacity;
    if (command_buffer->command_block_tail) {
      command_buffer->command_block_tail->next = block;
    } else {
      command_buffer->command_block_head = block;
    }
    command_buffer->command_block_tail = block;
    command_buffer->command_block_next =
        iree_hal_vulkan_command_buffer_block_data(block);
    command_buffer->command_block_end =
        command_buffer->command_block_next + capacity;
  }
  *out_record_length = record_length;
  *out_payload_offset = payload_offset;
  return iree_ok_status();
}

static void iree_hal_vulkan_command_buffer_append_command(
    iree_hal_vulkan_command_buffer_t* command_buffer,
    iree_hal_vulkan_command_type_t type, iree_host_size_t record_length,
    iree_host_size_t payload_offset, iree_hal_vulkan_command_t** out_command,
    void** out_payload) {
  iree_hal_vulkan_command_block_t* block = command_buffer->command_block_tail;
  IREE_ASSERT(record_length <=
              (iree_host_size_t)(command_buffer->command_block_end -
                                 command_buffer->command_block_next));
  uint8_t* const record = command_buffer->command_block_next;
  iree_hal_vulkan_command_t* command = (iree_hal_vulkan_command_t*)record;
  command->record_length = record_length;
  command->type = type;
  command_buffer->command_block_next += record_length;
  block->data_length += record_length;
  command_buffer->command_count = command_buffer->command_count + 1;
  if (out_command) *out_command = command;
  if (out_payload) *out_payload = record + payload_offset;
}

static iree_hal_vulkan_command_buffer_iterator_t
iree_hal_vulkan_command_buffer_iterator(
    const iree_hal_vulkan_command_buffer_t* command_buffer) {
  iree_hal_vulkan_command_buffer_iterator_t iterator = {
      .block = command_buffer->command_block_head,
      .block_offset = 0,
      .command_index = 0,
  };
  return iterator;
}

static bool iree_hal_vulkan_command_buffer_iterator_next(
    iree_hal_vulkan_command_buffer_iterator_t* iterator,
    const iree_hal_vulkan_command_t** out_command,
    iree_host_size_t* out_command_index) {
  while (iterator->block &&
         iterator->block_offset >= iterator->block->data_length) {
    iterator->block = iterator->block->next;
    iterator->block_offset = 0;
  }
  if (!iterator->block) return false;
  const iree_hal_vulkan_command_t* command =
      (const iree_hal_vulkan_command_t*)(iree_hal_vulkan_command_buffer_const_block_data(
                                             iterator->block) +
                                         iterator->block_offset);
  IREE_ASSERT(command->record_length != 0);
  iterator->block_offset += command->record_length;
  *out_command = command;
  if (out_command_index) *out_command_index = iterator->command_index;
  ++iterator->command_index;
  return true;
}

static iree_status_t iree_hal_vulkan_command_buffer_validate_recording_state(
    iree_hal_vulkan_command_buffer_t* command_buffer,
    iree_string_view_t command_name) {
  if (command_buffer->state != IREE_HAL_VULKAN_COMMAND_BUFFER_STATE_RECORDING) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan command buffer %.*s requires recording state",
        (int)command_name.size, command_name.data);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_command_buffer_retain_resource(
    iree_hal_vulkan_command_buffer_t* command_buffer, void* resource) {
  if (!resource || !command_buffer->resource_set) {
    return iree_ok_status();
  }
  return iree_hal_resource_set_insert(command_buffer->resource_set, 1,
                                      &resource);
}

static iree_status_t iree_hal_vulkan_command_buffer_track_buffer_ref(
    iree_hal_vulkan_command_buffer_t* command_buffer,
    iree_hal_buffer_ref_t buffer_ref) {
  if (buffer_ref.buffer) {
    return iree_hal_vulkan_command_buffer_retain_resource(command_buffer,
                                                          buffer_ref.buffer);
  }
  command_buffer->base.binding_count =
      iree_max(command_buffer->base.binding_count, buffer_ref.buffer_slot + 1);
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_command_buffer_track_buffer_refs(
    iree_hal_vulkan_command_buffer_t* command_buffer, iree_host_size_t count,
    const iree_hal_buffer_ref_t* buffer_refs) {
  if (!command_buffer->resource_set &&
      command_buffer->base.binding_count ==
          command_buffer->base.binding_capacity) {
    return iree_ok_status();
  }
  bool has_static_refs = false;
  uint32_t binding_count = command_buffer->base.binding_count;
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (buffer_refs[i].buffer) {
      has_static_refs = true;
    } else {
      binding_count = iree_max(binding_count, buffer_refs[i].buffer_slot + 1);
    }
  }
  command_buffer->base.binding_count = binding_count;
  if (has_static_refs && command_buffer->resource_set) {
    IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert_strided(
        command_buffer->resource_set, count, buffer_refs,
        offsetof(iree_hal_buffer_ref_t, buffer), sizeof(buffer_refs[0])));
  }
  return iree_ok_status();
}

static iree_hal_vulkan_transfer_strategy_t
iree_hal_vulkan_command_buffer_fill_update_strategy_for_ref(
    iree_hal_buffer_ref_t buffer_ref) {
  if (buffer_ref.length == 0) {
    return IREE_HAL_VULKAN_TRANSFER_STRATEGY_NATIVE;
  }
  if (!buffer_ref.buffer) {
    return buffer_ref.length == IREE_HAL_WHOLE_BUFFER
               ? IREE_HAL_VULKAN_TRANSFER_STRATEGY_PER_ISSUE_COMPUTE
               : IREE_HAL_VULKAN_TRANSFER_STRATEGY_INDIRECT_COMPUTE;
  }
  switch (iree_hal_vulkan_buffer_range_dword_alignment(
      buffer_ref.buffer, buffer_ref.offset, buffer_ref.length)) {
    case IREE_HAL_VULKAN_BUFFER_RANGE_ALIGNMENT_ALIGNED:
      return IREE_HAL_VULKAN_TRANSFER_STRATEGY_NATIVE;
    case IREE_HAL_VULKAN_BUFFER_RANGE_ALIGNMENT_UNALIGNED:
      return IREE_HAL_VULKAN_TRANSFER_STRATEGY_EDGE_PATCHED_NATIVE;
    case IREE_HAL_VULKAN_BUFFER_RANGE_ALIGNMENT_UNKNOWN:
      return IREE_HAL_VULKAN_TRANSFER_STRATEGY_PER_ISSUE_COMPUTE;
  }
  return IREE_HAL_VULKAN_TRANSFER_STRATEGY_INDIRECT_COMPUTE;
}

static iree_hal_vulkan_transfer_strategy_t
iree_hal_vulkan_command_buffer_copy_strategy_for_refs(
    iree_hal_buffer_ref_t source_ref, iree_hal_buffer_ref_t target_ref) {
  if (source_ref.length == 0 && target_ref.length == 0) {
    return IREE_HAL_VULKAN_TRANSFER_STRATEGY_NATIVE;
  }
  if (!source_ref.buffer || !target_ref.buffer) {
    return source_ref.length == IREE_HAL_WHOLE_BUFFER
               ? IREE_HAL_VULKAN_TRANSFER_STRATEGY_PER_ISSUE_COMPUTE
               : IREE_HAL_VULKAN_TRANSFER_STRATEGY_INDIRECT_COMPUTE;
  }
  const iree_hal_vulkan_buffer_range_alignment_t source_alignment =
      iree_hal_vulkan_buffer_range_dword_alignment(
          source_ref.buffer, source_ref.offset, source_ref.length);
  const iree_hal_vulkan_buffer_range_alignment_t target_alignment =
      iree_hal_vulkan_buffer_range_dword_alignment(
          target_ref.buffer, target_ref.offset, target_ref.length);
  if (source_alignment == IREE_HAL_VULKAN_BUFFER_RANGE_ALIGNMENT_ALIGNED &&
      target_alignment == IREE_HAL_VULKAN_BUFFER_RANGE_ALIGNMENT_ALIGNED) {
    return IREE_HAL_VULKAN_TRANSFER_STRATEGY_NATIVE;
  }
  if (source_alignment == IREE_HAL_VULKAN_BUFFER_RANGE_ALIGNMENT_UNKNOWN ||
      target_alignment == IREE_HAL_VULKAN_BUFFER_RANGE_ALIGNMENT_UNKNOWN) {
    return IREE_HAL_VULKAN_TRANSFER_STRATEGY_PER_ISSUE_COMPUTE;
  }
  return IREE_HAL_VULKAN_TRANSFER_STRATEGY_COMPUTE;
}

static VkQueueFlags iree_hal_vulkan_command_buffer_transfer_queue_flags(
    iree_hal_vulkan_transfer_strategy_t strategy) {
  return strategy == IREE_HAL_VULKAN_TRANSFER_STRATEGY_NATIVE
             ? VK_QUEUE_TRANSFER_BIT
             : VK_QUEUE_COMPUTE_BIT;
}

static void iree_hal_vulkan_command_buffer_capture_transfer_barrier(
    iree_hal_vulkan_command_buffer_t* command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_vulkan_command_transfer_barrier_t* transfer_barrier) {
  memset(transfer_barrier, 0, sizeof(*transfer_barrier));
  if (command_buffer->has_compute_transfer &&
      iree_any_bit_set(source_stage_mask, IREE_HAL_EXECUTION_STAGE_TRANSFER)) {
    transfer_barrier->flags |=
        IREE_HAL_VULKAN_BARRIER_FLAG_COMPUTE_TRANSFER_SOURCE;
  }
  if (iree_any_bit_set(target_stage_mask, IREE_HAL_EXECUTION_STAGE_TRANSFER)) {
    transfer_barrier->next_pending =
        command_buffer->pending_compute_transfer_barriers;
    command_buffer->pending_compute_transfer_barriers = transfer_barrier;
  }
}

static void iree_hal_vulkan_command_buffer_capture_compute_transfer(
    iree_hal_vulkan_command_buffer_t* command_buffer) {
  iree_hal_vulkan_command_transfer_barrier_t* transfer_barrier =
      command_buffer->pending_compute_transfer_barriers;
  while (transfer_barrier) {
    iree_hal_vulkan_command_transfer_barrier_t* next =
        transfer_barrier->next_pending;
    transfer_barrier->flags |=
        IREE_HAL_VULKAN_BARRIER_FLAG_COMPUTE_TRANSFER_TARGET;
    transfer_barrier->next_pending = NULL;
    transfer_barrier = next;
  }
  command_buffer->pending_compute_transfer_barriers = NULL;
  command_buffer->has_compute_transfer = true;
}

static void iree_hal_vulkan_command_buffer_require_transfer_strategy(
    iree_hal_vulkan_command_buffer_t* command_buffer,
    iree_hal_vulkan_transfer_strategy_t strategy) {
  if (strategy != IREE_HAL_VULKAN_TRANSFER_STRATEGY_NATIVE) {
    iree_hal_vulkan_command_buffer_capture_compute_transfer(command_buffer);
  }
  command_buffer->required_queue_flags |=
      iree_hal_vulkan_command_buffer_transfer_queue_flags(strategy);
  if (strategy == IREE_HAL_VULKAN_TRANSFER_STRATEGY_PER_ISSUE_COMPUTE) {
    command_buffer->requires_per_issue_recording = true;
  }
}

static iree_status_t iree_hal_vulkan_command_buffer_resolve_buffer_ref(
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_buffer_ref_t buffer_ref, iree_string_view_t usage,
    iree_hal_buffer_ref_t* out_resolved_ref) {
  IREE_RETURN_IF_ERROR(iree_hal_buffer_binding_table_resolve_ref(
      binding_table, buffer_ref, out_resolved_ref));
  if (!out_resolved_ref->buffer) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan command buffer %.*s buffer reference resolved to NULL",
        (int)usage.size, usage.data);
  }
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(out_resolved_ref->buffer),
      IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE));
  return iree_hal_buffer_calculate_range(
      /*base_offset=*/0, iree_hal_buffer_byte_length(out_resolved_ref->buffer),
      out_resolved_ref->offset, out_resolved_ref->length,
      &out_resolved_ref->offset, &out_resolved_ref->length);
}

static iree_status_t iree_hal_vulkan_command_buffer_resolve_native_buffer_ref(
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_buffer_ref_t buffer_ref, iree_string_view_t usage,
    VkBuffer* out_handle, VkDeviceSize* out_offset, VkDeviceSize* out_length) {
  iree_hal_buffer_ref_t resolved_ref;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_resolve_buffer_ref(
      binding_table, buffer_ref, usage, &resolved_ref));
  iree_hal_buffer_t* backing_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_buffer_resolve_backing(
      resolved_ref.buffer, &backing_buffer));
  iree_hal_buffer_t* allocated_buffer =
      iree_hal_buffer_allocated_buffer(backing_buffer);
  if (!iree_hal_vulkan_buffer_isa(allocated_buffer) &&
      !iree_hal_vulkan_sparse_buffer_isa(allocated_buffer)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan command buffer %.*s buffer reference is not backed by the "
        "Vulkan HAL",
        (int)usage.size, usage.data);
  }

  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkBuffer handle = VK_NULL_HANDLE;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_buffer_handle(backing_buffer, &memory, &handle));
  (void)memory;

  iree_device_size_t absolute_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_buffer_resolve_backing_offset(
      resolved_ref.buffer, backing_buffer, resolved_ref.offset,
      &absolute_offset));

  *out_handle = handle;
  *out_offset = (VkDeviceSize)absolute_offset;
  *out_length = (VkDeviceSize)resolved_ref.length;
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_expand_fill_pattern(
    const uint8_t* pattern, iree_host_size_t pattern_length,
    uint32_t* out_expanded_pattern) {
  *out_expanded_pattern = 0;
  switch (pattern_length) {
    case sizeof(uint8_t):
      *out_expanded_pattern = pattern[0];
      *out_expanded_pattern |= *out_expanded_pattern << 8;
      *out_expanded_pattern |= *out_expanded_pattern << 16;
      break;
    case sizeof(uint16_t): {
      uint16_t pattern16 = 0;
      memcpy(&pattern16, pattern, sizeof(pattern16));
      *out_expanded_pattern = pattern16;
      *out_expanded_pattern |= *out_expanded_pattern << 16;
      break;
    }
    case sizeof(uint32_t):
      memcpy(out_expanded_pattern, pattern, sizeof(*out_expanded_pattern));
      break;
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Vulkan fill pattern length must be 1, 2, or 4 bytes (got %" PRIhsz
          ")",
          pattern_length);
  }
  return iree_ok_status();
}

static iree_status_t
iree_hal_vulkan_command_buffer_validate_indirect_parameters_ref(
    iree_hal_vulkan_command_buffer_t* command_buffer,
    iree_hal_buffer_ref_t workgroup_count_ref) {
  const iree_device_size_t workgroup_count_length = sizeof(uint32_t[3]);
  if ((workgroup_count_ref.offset % sizeof(uint32_t)) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan indirect workgroup parameter offset must be 4-byte aligned");
  }
  if (workgroup_count_ref.length != IREE_HAL_WHOLE_BUFFER &&
      workgroup_count_ref.length < workgroup_count_length) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan indirect workgroup parameter buffer must contain at least "
        "uint32_t[3]");
  }
  if (!workgroup_count_ref.buffer) return iree_ok_status();

  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(workgroup_count_ref.buffer),
      IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(workgroup_count_ref.buffer),
      IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(workgroup_count_ref.buffer),
      IREE_HAL_MEMORY_ACCESS_READ));
  return iree_hal_buffer_validate_range(workgroup_count_ref.buffer,
                                        workgroup_count_ref.offset,
                                        workgroup_count_length);
}

iree_status_t iree_hal_vulkan_command_buffer_create(
    iree_hal_allocator_t* device_allocator, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    const iree_hal_vulkan_atomic_pipelines_t* atomic_pipelines,
    iree_arena_block_pool_t* command_buffer_block_pool,
    iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(device_allocator);
  IREE_ASSERT_ARGUMENT(atomic_pipelines);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = NULL;
  if (IREE_UNLIKELY(!command_buffer_block_pool)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Vulkan command-buffer block pool is "
                            "required");
  }
  if (IREE_UNLIKELY(binding_capacity > UINT32_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan command-buffer binding capacity exceeds uint32_t");
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_host_size_t validation_state_size =
      iree_hal_command_buffer_validation_state_size(mode, binding_capacity);
  iree_hal_vulkan_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator,
                                sizeof(*command_buffer) + validation_state_size,
                                (void**)&command_buffer));
  memset(command_buffer, 0, sizeof(*command_buffer) + validation_state_size);

  iree_hal_command_buffer_initialize(
      device_allocator, mode, command_categories, queue_affinity,
      binding_capacity, (uint8_t*)command_buffer + sizeof(*command_buffer),
      &iree_hal_vulkan_command_buffer_vtable, &command_buffer->base);
  command_buffer->host_allocator = host_allocator;
  command_buffer->atomic_pipelines = atomic_pipelines;
  iree_arena_initialize(command_buffer_block_pool,
                        &command_buffer->payload_arena);
  command_buffer->state = IREE_HAL_VULKAN_COMMAND_BUFFER_STATE_INITIAL;
  iree_status_t status = iree_ok_status();
  if (!iree_all_bits_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED)) {
    status = iree_hal_resource_set_allocate(command_buffer_block_pool,
                                            &command_buffer->resource_set);
  }
  if (iree_status_is_ok(status)) {
    *out_command_buffer = &command_buffer->base;
  } else {
    iree_hal_command_buffer_destroy(&command_buffer->base);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

bool iree_hal_vulkan_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer) {
  return iree_hal_resource_is(&command_buffer->resource,
                              &iree_hal_vulkan_command_buffer_vtable);
}

bool iree_hal_vulkan_command_buffer_is_empty(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_vulkan_command_buffer_t* command_buffer =
      iree_hal_vulkan_command_buffer_cast(base_command_buffer);
  return command_buffer->command_count == 0;
}

bool iree_hal_vulkan_command_buffer_has_native_commands(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_vulkan_command_buffer_t* command_buffer =
      iree_hal_vulkan_command_buffer_cast(base_command_buffer);
  return command_buffer->command_count != 0;
}

bool iree_hal_vulkan_command_buffer_requires_per_issue_recording(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_vulkan_command_buffer_t* command_buffer =
      iree_hal_vulkan_command_buffer_cast(base_command_buffer);
  return command_buffer->requires_per_issue_recording;
}

iree_host_size_t iree_hal_vulkan_command_buffer_dispatch_count(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_vulkan_command_buffer_t* command_buffer =
      iree_hal_vulkan_command_buffer_cast(base_command_buffer);
  return command_buffer->dispatch_count;
}

static iree_hal_profile_command_operation_type_t
iree_hal_vulkan_command_buffer_profile_operation_type(
    const iree_hal_vulkan_command_t* command) {
  switch (command->type) {
    case IREE_HAL_VULKAN_COMMAND_TYPE_FILL_BUFFER:
      return IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_FILL;
    case IREE_HAL_VULKAN_COMMAND_TYPE_UPDATE_BUFFER:
      return IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_UPDATE;
    case IREE_HAL_VULKAN_COMMAND_TYPE_COPY_BUFFER:
      return IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_COPY;
    case IREE_HAL_VULKAN_COMMAND_TYPE_DISPATCH:
      return IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_DISPATCH;
    case IREE_HAL_VULKAN_COMMAND_TYPE_EXECUTION_BARRIER:
      return IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_BARRIER;
    case IREE_HAL_VULKAN_COMMAND_TYPE_ATOMIC: {
      const iree_hal_vulkan_command_atomic_t* atomic =
          iree_hal_vulkan_command_atomic_payload(command);
      switch (atomic->params.operation) {
        case IREE_HAL_VULKAN_ATOMIC_OPERATION_WAIT_EQUAL:
        case IREE_HAL_VULKAN_ATOMIC_OPERATION_WAIT_NOT_EQUAL:
        case IREE_HAL_VULKAN_ATOMIC_OPERATION_WAIT_UNSIGNED_GREATER_EQUAL:
          return IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_ATOMIC_WAIT;
        case IREE_HAL_VULKAN_ATOMIC_OPERATION_STORE:
          return IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_ATOMIC_STORE;
        case IREE_HAL_VULKAN_ATOMIC_OPERATION_RMW_ADD:
        case IREE_HAL_VULKAN_ATOMIC_OPERATION_RMW_SUBTRACT:
        case IREE_HAL_VULKAN_ATOMIC_OPERATION_RMW_AND:
        case IREE_HAL_VULKAN_ATOMIC_OPERATION_RMW_OR:
        case IREE_HAL_VULKAN_ATOMIC_OPERATION_RMW_XOR:
          return IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_ATOMIC_RMW;
        default:
          IREE_ASSERT_UNREACHABLE("atomic parameters must be validated");
          return IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_NONE;
      }
    }
    case IREE_HAL_VULKAN_COMMAND_TYPE_BEGIN_DEBUG_GROUP:
    case IREE_HAL_VULKAN_COMMAND_TYPE_END_DEBUG_GROUP:
      return IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_PROFILE_MARKER;
  }
  return IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_NONE;
}

static iree_hal_profile_command_operation_flags_t
iree_hal_vulkan_command_buffer_profile_binding_flags(
    iree_hal_buffer_ref_t buffer_ref) {
  return buffer_ref.buffer
             ? IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_STATIC_BINDINGS
             : IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_DYNAMIC_BINDINGS;
}

static void iree_hal_vulkan_command_buffer_profile_ref(
    iree_hal_buffer_ref_t buffer_ref, uint32_t* out_binding_ordinal,
    uint64_t* out_offset, uint64_t* out_length) {
  *out_binding_ordinal =
      buffer_ref.buffer ? UINT32_MAX : buffer_ref.buffer_slot;
  *out_offset = buffer_ref.offset;
  *out_length =
      buffer_ref.length == IREE_HAL_WHOLE_BUFFER ? 0 : buffer_ref.length;
}

static iree_status_t iree_hal_vulkan_command_buffer_profile_operation(
    const iree_hal_vulkan_command_t* command, uint64_t command_buffer_id,
    uint32_t command_index,
    iree_hal_profile_command_operation_record_t* out_record) {
  iree_hal_profile_command_operation_record_t record =
      iree_hal_profile_command_operation_record_default();
  record.type = iree_hal_vulkan_command_buffer_profile_operation_type(command);
  record.command_buffer_id = command_buffer_id;
  record.command_index = command_index;

  switch (command->type) {
    case IREE_HAL_VULKAN_COMMAND_TYPE_FILL_BUFFER: {
      const iree_hal_vulkan_command_fill_buffer_t* fill_buffer =
          iree_hal_vulkan_command_fill_buffer_payload(command);
      record.flags |= iree_hal_vulkan_command_buffer_profile_binding_flags(
          fill_buffer->target_ref);
      iree_hal_vulkan_command_buffer_profile_ref(
          fill_buffer->target_ref, &record.target_ordinal,
          &record.target_offset, &record.length);
      break;
    }
    case IREE_HAL_VULKAN_COMMAND_TYPE_UPDATE_BUFFER: {
      const iree_hal_vulkan_command_update_buffer_t* update_buffer =
          iree_hal_vulkan_command_update_buffer_payload(command);
      record.flags |= iree_hal_vulkan_command_buffer_profile_binding_flags(
          update_buffer->target_ref);
      iree_hal_vulkan_command_buffer_profile_ref(
          update_buffer->target_ref, &record.target_ordinal,
          &record.target_offset, &record.length);
      record.length = update_buffer->source_data_length;
      break;
    }
    case IREE_HAL_VULKAN_COMMAND_TYPE_COPY_BUFFER: {
      const iree_hal_vulkan_command_copy_buffer_t* copy_buffer =
          iree_hal_vulkan_command_copy_buffer_payload(command);
      record.flags |= iree_hal_vulkan_command_buffer_profile_binding_flags(
          copy_buffer->source_ref);
      record.flags |= iree_hal_vulkan_command_buffer_profile_binding_flags(
          copy_buffer->target_ref);
      iree_hal_vulkan_command_buffer_profile_ref(
          copy_buffer->source_ref, &record.source_ordinal,
          &record.source_offset, &record.length);
      iree_hal_vulkan_command_buffer_profile_ref(
          copy_buffer->target_ref, &record.target_ordinal,
          &record.target_offset, &record.length);
      break;
    }
    case IREE_HAL_VULKAN_COMMAND_TYPE_DISPATCH: {
      const iree_hal_vulkan_command_dispatch_t* dispatch =
          iree_hal_vulkan_command_dispatch_payload(command);
      const iree_hal_vulkan_pipeline_t* pipeline = NULL;
      IREE_RETURN_IF_ERROR(iree_hal_vulkan_executable_lookup_pipeline(
          dispatch->executable, dispatch->function_ordinal, &pipeline));
      record.executable_id =
          iree_hal_vulkan_executable_profile_id(dispatch->executable);
      record.function_ordinal =
          iree_hal_executable_function_index(dispatch->function_ordinal);
      if (dispatch->binding_count > UINT32_MAX) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "Vulkan command buffer profile binding count exceeds uint32_t");
      }
      record.binding_count = (uint32_t)dispatch->binding_count;
      memcpy(record.workgroup_size, pipeline->workgroup_size,
             sizeof(record.workgroup_size));
      if (iree_hal_dispatch_uses_indirect_parameters(dispatch->flags)) {
        record.flags |=
            IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_INDIRECT_PARAMETERS;
        record.flags |= iree_hal_vulkan_command_buffer_profile_binding_flags(
            dispatch->config.workgroup_count_ref);
      } else {
        memcpy(record.workgroup_count, dispatch->config.workgroup_count,
               sizeof(record.workgroup_count));
      }
      const iree_hal_buffer_ref_t* bindings =
          iree_hal_vulkan_command_dispatch_bindings(dispatch);
      for (iree_host_size_t i = 0; i < dispatch->binding_count; ++i) {
        record.flags |=
            iree_hal_vulkan_command_buffer_profile_binding_flags(bindings[i]);
      }
      break;
    }
    case IREE_HAL_VULKAN_COMMAND_TYPE_EXECUTION_BARRIER:
      record.flags |= IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_EXECUTION_BARRIER;
      break;
    case IREE_HAL_VULKAN_COMMAND_TYPE_ATOMIC: {
      const iree_hal_vulkan_command_atomic_t* atomic =
          iree_hal_vulkan_command_atomic_payload(command);
      record.flags |= iree_hal_vulkan_command_buffer_profile_binding_flags(
          atomic->target_ref);
      iree_hal_vulkan_command_buffer_profile_ref(
          atomic->target_ref, &record.target_ordinal, &record.target_offset,
          &record.length);
      break;
    }
    case IREE_HAL_VULKAN_COMMAND_TYPE_BEGIN_DEBUG_GROUP:
    case IREE_HAL_VULKAN_COMMAND_TYPE_END_DEBUG_GROUP:
      break;
  }

  *out_record = record;
  return iree_ok_status();
}

iree_status_t iree_hal_vulkan_command_buffer_record_profile_metadata(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_local_profile_recorder_t* profile_recorder,
    iree_hal_local_profile_queue_scope_t scope, uint64_t command_buffer_id) {
  if (!iree_hal_local_profile_recorder_is_enabled(
          profile_recorder,
          IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA)) {
    return iree_ok_status();
  }

  iree_hal_vulkan_command_buffer_t* command_buffer =
      iree_hal_vulkan_command_buffer_cast(base_command_buffer);
  if (command_buffer->state != IREE_HAL_VULKAN_COMMAND_BUFFER_STATE_ENDED) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan command buffer profile metadata requires ended state");
  }
  iree_hal_vulkan_command_buffer_iterator_t iterator =
      iree_hal_vulkan_command_buffer_iterator(command_buffer);
  const iree_hal_vulkan_command_t* command = NULL;
  while (iree_hal_vulkan_command_buffer_iterator_next(
      &iterator, &command, /*out_command_index=*/NULL)) {
    if (command->type != IREE_HAL_VULKAN_COMMAND_TYPE_DISPATCH) continue;
    const iree_hal_vulkan_command_dispatch_t* dispatch =
        iree_hal_vulkan_command_dispatch_payload(command);
    IREE_RETURN_IF_ERROR(
        iree_hal_local_profile_recorder_record_executable_with_id(
            profile_recorder, dispatch->executable,
            iree_hal_vulkan_executable_profile_id(dispatch->executable)));
  }

  if (command_buffer_id == 0) return iree_ok_status();
  iree_hal_profile_command_buffer_record_t command_buffer_record =
      iree_hal_profile_command_buffer_record_default();
  command_buffer_record.command_buffer_id = command_buffer_id;
  command_buffer_record.mode = command_buffer->base.mode;
  command_buffer_record.command_categories =
      command_buffer->base.allowed_categories;
  command_buffer_record.queue_affinity = command_buffer->base.queue_affinity;
  command_buffer_record.physical_device_ordinal = scope.physical_device_ordinal;

  iree_hal_profile_command_operation_record_t* operation_records = NULL;
  iree_status_t status = iree_ok_status();
  if (command_buffer->command_count != 0) {
    if (command_buffer->command_count > UINT32_MAX) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "Vulkan command buffer profile operation count exceeds uint32_t");
    }
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc_array(
          command_buffer->host_allocator, command_buffer->command_count,
          sizeof(operation_records[0]), (void**)&operation_records);
    }
    iterator = iree_hal_vulkan_command_buffer_iterator(command_buffer);
    iree_host_size_t command_index = 0;
    while (iree_status_is_ok(status) &&
           iree_hal_vulkan_command_buffer_iterator_next(&iterator, &command,
                                                        &command_index)) {
      status = iree_hal_vulkan_command_buffer_profile_operation(
          command, command_buffer_id, (uint32_t)command_index,
          &operation_records[command_index]);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_local_profile_recorder_record_command_buffer(
        profile_recorder, &command_buffer_record, command_buffer->command_count,
        operation_records);
  }
  iree_allocator_free(command_buffer->host_allocator, operation_records);
  return status;
}

static bool iree_hal_vulkan_command_buffer_profile_filter_matches_dispatch(
    iree_hal_local_profile_recorder_t* profile_recorder,
    iree_hal_local_profile_queue_scope_t scope, uint64_t command_buffer_id,
    uint32_t command_index, const iree_hal_vulkan_pipeline_t* pipeline) {
  const iree_hal_device_profiling_options_t* options =
      iree_hal_local_profile_recorder_options(profile_recorder);
  if (!options) return false;
  const iree_hal_profile_capture_filter_t* filter = &options->capture_filter;
  if (!iree_hal_profile_capture_filter_matches_location(
          filter, command_buffer_id, command_index,
          scope.physical_device_ordinal, scope.queue_ordinal)) {
    return false;
  }
  if (iree_any_bit_set(
          filter->flags,
          IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_EXECUTABLE_FUNCTION_PATTERN) &&
      !iree_string_view_match_pattern(pipeline->name,
                                      filter->executable_function_pattern)) {
    return false;
  }
  return true;
}

iree_status_t iree_hal_vulkan_command_buffer_count_profiled_dispatches(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_local_profile_recorder_t* profile_recorder,
    iree_hal_local_profile_queue_scope_t scope, uint64_t command_buffer_id,
    uint32_t* out_dispatch_count) {
  *out_dispatch_count = 0;
  if (!iree_hal_local_profile_recorder_is_enabled(
          profile_recorder, IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS)) {
    return iree_ok_status();
  }

  iree_hal_vulkan_command_buffer_t* command_buffer =
      iree_hal_vulkan_command_buffer_cast(base_command_buffer);
  if (command_buffer->state != IREE_HAL_VULKAN_COMMAND_BUFFER_STATE_ENDED) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan dispatch profile counting requires ended command buffer");
  }

  const bool is_command_buffer_dispatch = command_buffer_id != 0;
  uint32_t dispatch_count = 0;
  iree_hal_vulkan_command_buffer_iterator_t iterator =
      iree_hal_vulkan_command_buffer_iterator(command_buffer);
  const iree_hal_vulkan_command_t* command = NULL;
  iree_host_size_t command_index = 0;
  while (iree_hal_vulkan_command_buffer_iterator_next(&iterator, &command,
                                                      &command_index)) {
    if (command->type != IREE_HAL_VULKAN_COMMAND_TYPE_DISPATCH) continue;
    const iree_hal_vulkan_command_dispatch_t* dispatch =
        iree_hal_vulkan_command_dispatch_payload(command);
    if (is_command_buffer_dispatch && command_index > UINT32_MAX) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "Vulkan dispatch profile command index exceeds uint32_t");
    }
    const uint32_t profile_command_index =
        is_command_buffer_dispatch ? (uint32_t)command_index : UINT32_MAX;
    if (!iree_hal_vulkan_command_buffer_profile_filter_matches_dispatch(
            profile_recorder, scope, command_buffer_id, profile_command_index,
            dispatch->pipeline)) {
      continue;
    }
    if (dispatch_count == UINT32_MAX) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "Vulkan dispatch profile dispatch count exceeds uint32_t");
    }
    dispatch_count = dispatch_count + 1;
  }

  *out_dispatch_count = dispatch_count;
  return iree_ok_status();
}

iree_status_t iree_hal_vulkan_command_buffer_append_dispatch_profile_events(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_local_profile_recorder_t* profile_recorder,
    iree_hal_local_profile_queue_scope_t scope, uint64_t submission_id,
    uint64_t command_buffer_id, const uint64_t* dispatch_ticks,
    iree_host_size_t dispatch_count) {
  if (!iree_hal_local_profile_recorder_is_enabled(
          profile_recorder, IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS)) {
    return iree_ok_status();
  }
  if (IREE_UNLIKELY(dispatch_count != 0 && !dispatch_ticks)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan dispatch profile timestamp storage is required");
  }

  iree_hal_vulkan_command_buffer_t* command_buffer =
      iree_hal_vulkan_command_buffer_cast(base_command_buffer);
  if (command_buffer->state != IREE_HAL_VULKAN_COMMAND_BUFFER_STATE_ENDED) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan dispatch profile events require ended command buffer");
  }

  const bool is_command_buffer_dispatch = command_buffer_id != 0;
  iree_host_size_t dispatch_ordinal = 0;
  iree_hal_vulkan_command_buffer_iterator_t iterator =
      iree_hal_vulkan_command_buffer_iterator(command_buffer);
  const iree_hal_vulkan_command_t* command = NULL;
  iree_host_size_t command_index = 0;
  while (iree_hal_vulkan_command_buffer_iterator_next(&iterator, &command,
                                                      &command_index)) {
    if (command->type != IREE_HAL_VULKAN_COMMAND_TYPE_DISPATCH) continue;
    const iree_hal_vulkan_command_dispatch_t* dispatch =
        iree_hal_vulkan_command_dispatch_payload(command);
    const iree_hal_vulkan_pipeline_t* pipeline = dispatch->pipeline;
    if (is_command_buffer_dispatch && command_index > UINT32_MAX) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "Vulkan dispatch profile command index exceeds uint32_t");
    }
    const uint32_t profile_command_index =
        is_command_buffer_dispatch ? (uint32_t)command_index : UINT32_MAX;
    if (!iree_hal_vulkan_command_buffer_profile_filter_matches_dispatch(
            profile_recorder, scope, command_buffer_id, profile_command_index,
            pipeline)) {
      continue;
    }
    if (dispatch_ordinal >= dispatch_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "Vulkan dispatch profile timestamp count is smaller than the "
          "profiled dispatch count");
    }
    if (pipeline->workgroup_size[0] == 0) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "Vulkan dispatch profiling requires static SPIR-V LocalSize "
          "metadata");
    }

    const uint64_t* ticks = &dispatch_ticks[dispatch_ordinal * 2];
    if (ticks[1] < ticks[0]) {
      return iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "Vulkan dispatch profiling timestamp range is not monotonic");
    }

    iree_hal_local_profile_dispatch_event_info_t event_info =
        iree_hal_local_profile_dispatch_event_info_default();
    event_info.scope = scope;
    event_info.submission_id = submission_id;
    event_info.command_buffer_id = command_buffer_id;
    event_info.executable_id =
        iree_hal_vulkan_executable_profile_id(dispatch->executable);
    event_info.command_index = profile_command_index;
    event_info.function_ordinal =
        iree_hal_executable_function_index(dispatch->function_ordinal);
    if (is_command_buffer_dispatch) {
      event_info.flags |= IREE_HAL_PROFILE_DISPATCH_EVENT_FLAG_COMMAND_BUFFER;
    }
    if (iree_hal_dispatch_uses_indirect_parameters(dispatch->flags)) {
      event_info.flags |=
          IREE_HAL_PROFILE_DISPATCH_EVENT_FLAG_INDIRECT_PARAMETERS;
    } else {
      memcpy(event_info.workgroup_count, dispatch->config.workgroup_count,
             sizeof(event_info.workgroup_count));
    }
    memcpy(event_info.workgroup_size, pipeline->workgroup_size,
           sizeof(event_info.workgroup_size));
    event_info.start_tick = ticks[0];
    event_info.end_tick = ticks[1];
    IREE_RETURN_IF_ERROR(iree_hal_local_profile_recorder_append_dispatch_event(
        profile_recorder, &event_info, /*out_event_id=*/NULL));
    ++dispatch_ordinal;
  }

  if (dispatch_ordinal != dispatch_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan dispatch profile timestamp count exceeds profiled dispatch "
        "count");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_command_buffer_add_bda_publication_bytes(
    iree_hal_vulkan_command_buffer_t* command_buffer,
    iree_device_size_t byte_length) {
  iree_device_size_t publication_length = 0;
  if (!iree_device_size_checked_add(command_buffer->bda_publication_length,
                                    byte_length, &publication_length)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan command buffer BDA publication length overflows");
  }
  command_buffer->bda_publication_length = publication_length;
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_command_buffer_dispatch_payload_layout(
    iree_const_byte_span_t constants, iree_hal_buffer_ref_list_t bindings,
    iree_host_size_t* out_payload_length,
    iree_host_size_t* out_constants_offset,
    iree_host_size_t* out_bindings_offset) {
  *out_payload_length = 0;
  *out_constants_offset = 0;
  *out_bindings_offset = 0;

  iree_host_size_t bindings_offset = 0;
  iree_host_size_t constants_offset = 0;
  if (!iree_host_size_checked_align(sizeof(iree_hal_vulkan_command_dispatch_t),
                                    iree_alignof(uint32_t),
                                    &constants_offset) ||
      !iree_host_size_checked_add(constants_offset, constants.data_length,
                                  &bindings_offset) ||
      !iree_host_size_checked_align(bindings_offset,
                                    iree_alignof(iree_hal_buffer_ref_t),
                                    &bindings_offset)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan dispatch command constants payload size overflows");
  }
  iree_host_size_t binding_byte_length = 0;
  if (!iree_host_size_checked_mul(bindings.count, sizeof(bindings.values[0]),
                                  &binding_byte_length)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan dispatch command binding payload size overflows");
  }
  iree_host_size_t payload_length = 0;
  if (!iree_host_size_checked_add(bindings_offset, binding_byte_length,
                                  &payload_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan dispatch command payload size overflows");
  }
  *out_payload_length = payload_length;
  *out_constants_offset = constants_offset;
  *out_bindings_offset = bindings_offset;
  return iree_ok_status();
}

iree_status_t iree_hal_vulkan_command_buffer_native_bda_publication_length(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_device_size_t* out_publication_length) {
  IREE_ASSERT_ARGUMENT(base_command_buffer);
  IREE_ASSERT_ARGUMENT(out_publication_length);
  iree_hal_vulkan_command_buffer_t* command_buffer =
      iree_hal_vulkan_command_buffer_cast(base_command_buffer);
  *out_publication_length = command_buffer->bda_publication_length;
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_command_buffer_resolve_bda_binding_slot(
    iree_hal_buffer_binding_table_t binding_table, uint32_t buffer_slot,
    iree_hal_vulkan_command_buffer_bda_binding_cache_t* bda_binding_cache,
    VkDeviceAddress* out_device_address, iree_device_size_t* out_length) {
  *out_device_address = 0;
  *out_length = 0;

  if (IREE_UNLIKELY(buffer_slot >= binding_table.count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "BDA binding slot %u out of range of binding "
                            "table with capacity %" PRIhsz,
                            buffer_slot, binding_table.count);
  }

  iree_hal_vulkan_command_buffer_bda_binding_slot_t* cached_slot = NULL;
  if (bda_binding_cache && buffer_slot < bda_binding_cache->slot_count) {
    cached_slot = &bda_binding_cache->slots[buffer_slot];
    if (cached_slot->device_address != 0) {
      *out_device_address = cached_slot->device_address;
      *out_length = cached_slot->length;
      return iree_ok_status();
    }
  }

  iree_hal_buffer_ref_t resolved_ref;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_binding_table_resolve_ref(
      binding_table,
      iree_hal_make_indirect_buffer_ref(buffer_slot, /*offset=*/0,
                                        IREE_HAL_WHOLE_BUFFER),
      &resolved_ref));
  if (!resolved_ref.buffer || resolved_ref.length == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan BDA binding table slot %u resolved to an empty buffer range",
        buffer_slot);
  }

  VkDeviceAddress buffer_address = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_buffer_device_address(
      resolved_ref.buffer, &buffer_address));
  if (buffer_address == 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan BDA binding table slot %u buffer has no device address",
        buffer_slot);
  }
  if (resolved_ref.offset > UINT64_MAX - buffer_address) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan BDA binding table slot %u device address overflows",
        buffer_slot);
  }
  const VkDeviceAddress device_address = buffer_address + resolved_ref.offset;
  if (resolved_ref.length > UINT64_MAX - device_address) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan BDA binding table slot %u device range overflows", buffer_slot);
  }

  if (cached_slot) {
    cached_slot->device_address = device_address;
    cached_slot->length = resolved_ref.length;
  }
  *out_device_address = device_address;
  *out_length = resolved_ref.length;
  return iree_ok_status();
}

iree_status_t iree_hal_vulkan_command_buffer_resolve_bda_binding_table_slot(
    iree_hal_buffer_binding_table_t binding_table, uint32_t buffer_slot,
    iree_hal_vulkan_command_buffer_bda_binding_cache_t* bda_binding_cache,
    iree_hal_vulkan_command_buffer_bda_binding_slot_t* out_slot) {
  IREE_ASSERT_ARGUMENT(out_slot);
  *out_slot = (iree_hal_vulkan_command_buffer_bda_binding_slot_t){0};
  return iree_hal_vulkan_command_buffer_resolve_bda_binding_slot(
      binding_table, buffer_slot, bda_binding_cache, &out_slot->device_address,
      &out_slot->length);
}

static iree_status_t iree_hal_vulkan_command_buffer_resolve_bda_ref(
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_buffer_ref_t buffer_ref, iree_string_view_t usage,
    iree_hal_vulkan_command_buffer_bda_binding_cache_t* bda_binding_cache,
    VkDeviceAddress* out_device_address,
    iree_device_size_t* out_resolved_length) {
  *out_device_address = 0;
  *out_resolved_length = 0;

  VkDeviceAddress device_address = 0;
  iree_device_size_t resolved_length = 0;
  if (!buffer_ref.buffer && bda_binding_cache) {
    VkDeviceAddress slot_device_address = 0;
    iree_device_size_t slot_length = 0;
    IREE_RETURN_IF_ERROR(
        iree_hal_vulkan_command_buffer_resolve_bda_binding_slot(
            binding_table, buffer_ref.buffer_slot, bda_binding_cache,
            &slot_device_address, &slot_length));
    iree_device_size_t resolved_offset = 0;
    IREE_RETURN_IF_ERROR(iree_hal_buffer_calculate_range(
        /*binding_offset=*/0, slot_length, buffer_ref.offset, buffer_ref.length,
        &resolved_offset, &resolved_length));
    if (resolved_offset > UINT64_MAX - slot_device_address) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "Vulkan command buffer %.*s device address overflows",
          (int)usage.size, usage.data);
    }
    device_address = slot_device_address + resolved_offset;
  } else {
    iree_hal_buffer_ref_t resolved_ref;
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_resolve_buffer_ref(
        binding_table, buffer_ref, usage, &resolved_ref));
    resolved_length = resolved_ref.length;
    if (resolved_ref.length == 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Vulkan command buffer %.*s resolved to an empty buffer range",
          (int)usage.size, usage.data);
    }

    VkDeviceAddress buffer_address = 0;
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_buffer_device_address(
        resolved_ref.buffer, &buffer_address));
    if (buffer_address == 0) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "Vulkan command buffer %.*s buffer has no device address",
          (int)usage.size, usage.data);
    }
    if (resolved_ref.offset > UINT64_MAX - buffer_address) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "Vulkan command buffer %.*s device address overflows",
          (int)usage.size, usage.data);
    }
    device_address = buffer_address + resolved_ref.offset;
  }
  if (resolved_length == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan command buffer %.*s resolved to an empty buffer range",
        (int)usage.size, usage.data);
  }
  if (resolved_length > UINT64_MAX - device_address) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan command buffer %.*s device range overflows",
                            (int)usage.size, usage.data);
  }

  *out_device_address = device_address;
  *out_resolved_length = resolved_length;
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_command_buffer_resolve_bda_binding(
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_buffer_ref_t buffer_ref, iree_host_size_t binding_ordinal,
    const iree_hal_vulkan_pipeline_t* pipeline,
    iree_hal_vulkan_command_buffer_bda_binding_cache_t* bda_binding_cache,
    VkDeviceAddress* out_device_address) {
  VkDeviceAddress device_address = 0;
  iree_device_size_t resolved_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_resolve_bda_ref(
      binding_table, buffer_ref, IREE_SV("BDA dispatch binding"),
      bda_binding_cache, &device_address, &resolved_length));
  if (binding_ordinal < pipeline->bda.binding_requirement_count) {
    const iree_hal_vulkan_bda_binding_requirement_t* requirement =
        &pipeline->bda.binding_requirements[binding_ordinal];
    if (requirement->minimum_alignment > 1 &&
        (device_address & (requirement->minimum_alignment - 1)) != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Vulkan dispatch binding %" PRIhsz
                              " device address 0x%" PRIx64
                              " does not satisfy BDA alignment %u",
                              binding_ordinal, (uint64_t)device_address,
                              requirement->minimum_alignment);
    }
    if (resolved_length < requirement->minimum_length) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "Vulkan dispatch binding %" PRIhsz " has length %" PRIdsz
          " but BDA pipeline requires at least %" PRIu64 " bytes",
          binding_ordinal, resolved_length, requirement->minimum_length);
    }
  }
  *out_device_address = device_address;
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_command_buffer_allocate_bda_publication(
    iree_hal_vulkan_command_buffer_bda_recording_state_t* bda_recording_state,
    iree_device_size_t byte_length, iree_byte_span_t* out_host_span,
    VkDeviceAddress* out_device_address) {
  *out_host_span = iree_byte_span_empty();
  *out_device_address = 0;
  if (byte_length == 0) return iree_ok_status();
  if (byte_length > IREE_HOST_SIZE_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan BDA replay publication exceeds host addressable size");
  }
  if (IREE_UNLIKELY(!bda_recording_state ||
                    !bda_recording_state->host_span.data)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Vulkan BDA replay requires publication storage");
  }

  const iree_host_size_t host_length = (iree_host_size_t)byte_length;
  const iree_host_size_t byte_offset = bda_recording_state->byte_offset;
  if (IREE_UNLIKELY(byte_offset > bda_recording_state->host_span.data_length ||
                    host_length > bda_recording_state->host_span.data_length -
                                      byte_offset)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan BDA replay data exceeds publication storage");
  }
  if (IREE_UNLIKELY(
          byte_offset > UINT64_MAX - bda_recording_state->device_address ||
          host_length > UINT64_MAX - (bda_recording_state->device_address +
                                      byte_offset))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan BDA replay publication device address range overflows");
  }

  *out_host_span = iree_make_byte_span(
      bda_recording_state->host_span.data + byte_offset, host_length);
  *out_device_address = bda_recording_state->device_address + byte_offset;
  bda_recording_state->byte_offset = byte_offset + host_length;
  return iree_ok_status();
}

static void iree_hal_vulkan_command_buffer_record_bda_publication_barrier(
    const iree_hal_vulkan_device_syms_t* syms,
    VkCommandBuffer native_command_buffer,
    iree_hal_vulkan_command_buffer_bda_recording_state_t* bda_recording_state) {
  if (bda_recording_state->barrier_recorded) return;
  const iree_hal_vulkan_barrier_t barrier = {
      .flags = IREE_HAL_VULKAN_BARRIER_FLAG_NONE,
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_HOST,
      .source_access_mask = VK_ACCESS_2_HOST_WRITE_BIT,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .target_access_mask = VK_ACCESS_2_SHADER_READ_BIT,
  };
  iree_hal_vulkan_barrier_record(syms, native_command_buffer, &barrier);
  bda_recording_state->barrier_recorded = true;
}

static iree_status_t iree_hal_vulkan_command_buffer_publish_fill_parameters(
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_vulkan_command_fill_buffer_t* fill_buffer,
    iree_hal_vulkan_command_buffer_bda_recording_state_t* bda_recording_state,
    iree_hal_vulkan_command_buffer_bda_binding_cache_t* bda_binding_cache,
    VkDeviceAddress* out_publication_address) {
  *out_publication_address = 0;
  VkDeviceAddress target_address = 0;
  iree_device_size_t target_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_resolve_bda_ref(
      binding_table, fill_buffer->target_ref, IREE_SV("fill target"),
      bda_binding_cache, &target_address, &target_length));
  if (target_length != fill_buffer->target_ref.length) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan indirect fill resolved a dynamic command range");
  }

  iree_device_size_t publication_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_byte_transfer_fill_publication_length(
      target_length, &publication_length));
  iree_byte_span_t publication_span = iree_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_allocate_bda_publication(
      bda_recording_state, publication_length, &publication_span,
      out_publication_address));
  return iree_hal_vulkan_byte_transfer_publish_fill(
      target_address, target_length, publication_span);
}

static iree_status_t
iree_hal_vulkan_command_buffer_publish_copy_parameters_for_addresses(
    VkDeviceAddress source_address, VkDeviceAddress target_address,
    iree_device_size_t length,
    iree_hal_vulkan_command_buffer_bda_recording_state_t* bda_recording_state,
    VkDeviceAddress* out_publication_address) {
  *out_publication_address = 0;
  iree_device_size_t publication_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_byte_transfer_copy_publication_length(
      length, &publication_length));
  iree_byte_span_t publication_span = iree_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_allocate_bda_publication(
      bda_recording_state, publication_length, &publication_span,
      out_publication_address));
  return iree_hal_vulkan_byte_transfer_publish_copy(
      source_address, target_address, length, publication_span);
}

static iree_status_t iree_hal_vulkan_command_buffer_publish_copy_parameters(
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_vulkan_command_copy_buffer_t* copy_buffer,
    iree_hal_vulkan_command_buffer_bda_recording_state_t* bda_recording_state,
    iree_hal_vulkan_command_buffer_bda_binding_cache_t* bda_binding_cache,
    VkDeviceAddress* out_publication_address) {
  VkDeviceAddress source_address = 0;
  iree_device_size_t source_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_resolve_bda_ref(
      binding_table, copy_buffer->source_ref, IREE_SV("copy source"),
      bda_binding_cache, &source_address, &source_length));
  VkDeviceAddress target_address = 0;
  iree_device_size_t target_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_resolve_bda_ref(
      binding_table, copy_buffer->target_ref, IREE_SV("copy target"),
      bda_binding_cache, &target_address, &target_length));
  if (source_length != target_length ||
      source_length != copy_buffer->source_ref.length) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan indirect copy resolved a dynamic command range");
  }
  return iree_hal_vulkan_command_buffer_publish_copy_parameters_for_addresses(
      source_address, target_address, source_length, bda_recording_state,
      out_publication_address);
}

static iree_status_t iree_hal_vulkan_command_buffer_record_copy_native_refs(
    const iree_hal_vulkan_device_syms_t* syms,
    const iree_hal_vulkan_builtins_t* builtins,
    VkCommandBuffer native_command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_buffer_ref_t source_ref, iree_string_view_t source_usage,
    iree_hal_buffer_ref_t target_ref, iree_string_view_t target_usage,
    iree_hal_vulkan_transfer_strategy_t strategy,
    iree_hal_vulkan_command_buffer_bda_recording_state_t* bda_recording_state,
    iree_hal_vulkan_command_buffer_bda_binding_cache_t* bda_binding_cache) {
  iree_hal_buffer_ref_t resolved_source_ref;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_binding_table_resolve_ref(
      binding_table, source_ref, &resolved_source_ref));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(resolved_source_ref.buffer),
      IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE));
  iree_hal_buffer_ref_t resolved_target_ref;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_binding_table_resolve_ref(
      binding_table, target_ref, &resolved_target_ref));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(resolved_target_ref.buffer),
      IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET));
  if (resolved_source_ref.length != resolved_target_ref.length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "resolved Vulkan command buffer copy spans differ "
                            "(source_length=%" PRIu64 ", target_length=%" PRIu64
                            ")",
                            (uint64_t)resolved_source_ref.length,
                            (uint64_t)resolved_target_ref.length);
  }
  if (resolved_source_ref.length == 0) return iree_ok_status();
  if (iree_hal_buffer_test_overlap(
          resolved_source_ref.buffer, resolved_source_ref.offset,
          resolved_source_ref.length, resolved_target_ref.buffer,
          resolved_target_ref.offset,
          resolved_target_ref.length) != IREE_HAL_BUFFER_OVERLAP_DISJOINT) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "resolved Vulkan command buffer copy ranges overlap");
  }

  if (iree_hal_vulkan_transfer_strategy_uses_compute(strategy)) {
    if (!builtins) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "Vulkan compute copy requires built-in pipelines");
    }
    VkDeviceAddress source_address = 0;
    iree_device_size_t source_length = 0;
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_resolve_bda_ref(
        binding_table, source_ref, source_usage, bda_binding_cache,
        &source_address, &source_length));
    VkDeviceAddress target_address = 0;
    iree_device_size_t target_length = 0;
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_resolve_bda_ref(
        binding_table, target_ref, target_usage, bda_binding_cache,
        &target_address, &target_length));
    if (source_length != target_length) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "resolved Vulkan command buffer copy spans differ "
          "(source_length=%" PRIu64 ", target_length=%" PRIu64 ")",
          (uint64_t)source_length, (uint64_t)target_length);
    }
    if (strategy == IREE_HAL_VULKAN_TRANSFER_STRATEGY_INDIRECT_COMPUTE) {
      VkDeviceAddress publication_address = 0;
      IREE_RETURN_IF_ERROR(
          iree_hal_vulkan_command_buffer_publish_copy_parameters_for_addresses(
              source_address, target_address, source_length,
              bda_recording_state, &publication_address));
      iree_hal_vulkan_command_buffer_record_bda_publication_barrier(
          syms, native_command_buffer, bda_recording_state);
      return iree_hal_vulkan_byte_transfer_record_copy_indirect(
          &builtins->byte_transfer_pipelines, native_command_buffer,
          publication_address, source_length);
    }
    return iree_hal_vulkan_byte_transfer_record_copy(
        &builtins->byte_transfer_pipelines, native_command_buffer,
        source_address, target_address, source_length);
  }
  if (strategy != IREE_HAL_VULKAN_TRANSFER_STRATEGY_NATIVE) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "invalid Vulkan copy transfer strategy %u",
                            (uint32_t)strategy);
  }

  VkBuffer source_handle = VK_NULL_HANDLE;
  VkDeviceSize source_offset = 0;
  VkDeviceSize source_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_resolve_native_buffer_ref(
      binding_table, source_ref, source_usage, &source_handle, &source_offset,
      &source_length));
  VkBuffer target_handle = VK_NULL_HANDLE;
  VkDeviceSize target_offset = 0;
  VkDeviceSize target_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_resolve_native_buffer_ref(
      binding_table, target_ref, target_usage, &target_handle, &target_offset,
      &target_length));
  if (source_length != target_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "resolved Vulkan command buffer copy spans differ "
                            "(source_length=%" PRIu64 ", target_length=%" PRIu64
                            ")",
                            (uint64_t)source_length, (uint64_t)target_length);
  }
  if (source_length == 0) return iree_ok_status();
  if ((source_offset | target_offset | source_length) &
      (sizeof(uint32_t) - 1)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan native copy lost its recorded dword-alignment invariant");
  }

  VkBufferCopy copy_region = {
      .srcOffset = source_offset,
      .dstOffset = target_offset,
      .size = source_length,
  };
  iree_vkCmdCopyBuffer(IREE_VULKAN_DEVICE(syms), native_command_buffer,
                       source_handle, target_handle, /*regionCount=*/1,
                       &copy_region);
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_command_buffer_record_fill_native(
    const iree_hal_vulkan_device_syms_t* syms,
    const iree_hal_vulkan_builtins_t* builtins,
    VkCommandBuffer native_command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_vulkan_command_t* command,
    iree_hal_vulkan_command_buffer_bda_recording_state_t* bda_recording_state,
    iree_hal_vulkan_command_buffer_bda_binding_cache_t* bda_binding_cache) {
  const iree_hal_vulkan_command_fill_buffer_t* fill_buffer =
      iree_hal_vulkan_command_fill_buffer_payload(command);
  iree_hal_buffer_ref_t resolved_target_ref;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_binding_table_resolve_ref(
      binding_table, fill_buffer->target_ref, &resolved_target_ref));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(resolved_target_ref.buffer),
      IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET));
  if (resolved_target_ref.length == 0) return iree_ok_status();

  if (iree_hal_vulkan_transfer_strategy_uses_compute(fill_buffer->strategy)) {
    if (!builtins) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "Vulkan compute fill requires built-in pipelines");
    }
    if (fill_buffer->strategy ==
        IREE_HAL_VULKAN_TRANSFER_STRATEGY_INDIRECT_COMPUTE) {
      VkDeviceAddress publication_address = 0;
      IREE_RETURN_IF_ERROR(
          iree_hal_vulkan_command_buffer_publish_fill_parameters(
              binding_table, fill_buffer, bda_recording_state,
              bda_binding_cache, &publication_address));
      iree_hal_vulkan_command_buffer_record_bda_publication_barrier(
          syms, native_command_buffer, bda_recording_state);
      return iree_hal_vulkan_byte_transfer_record_fill_indirect(
          &builtins->byte_transfer_pipelines, native_command_buffer,
          publication_address, fill_buffer->target_ref.length,
          fill_buffer->pattern, fill_buffer->pattern_length);
    }
    VkDeviceAddress target_address = 0;
    iree_device_size_t target_length = 0;
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_resolve_bda_ref(
        binding_table, fill_buffer->target_ref, IREE_SV("fill target"),
        bda_binding_cache, &target_address, &target_length));
    return iree_hal_vulkan_byte_transfer_record_fill(
        &builtins->byte_transfer_pipelines, native_command_buffer,
        target_address, target_length, fill_buffer->pattern,
        fill_buffer->pattern_length);
  }

  VkBuffer target_handle = VK_NULL_HANDLE;
  VkDeviceSize target_offset = 0;
  VkDeviceSize target_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_resolve_native_buffer_ref(
      binding_table, fill_buffer->target_ref, IREE_SV("fill target"),
      &target_handle, &target_offset, &target_length));
  if (target_length == 0) return iree_ok_status();
  const bool is_dword_aligned = target_offset % sizeof(uint32_t) == 0 &&
                                target_length % sizeof(uint32_t) == 0;
  if (fill_buffer->strategy ==
      IREE_HAL_VULKAN_TRANSFER_STRATEGY_EDGE_PATCHED_NATIVE) {
    if (is_dword_aligned) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "Vulkan edge-patched fill lost its recorded alignment invariant");
    }
    if (!builtins) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "Vulkan unaligned native fill requires built-in pipelines");
    }
    VkDeviceAddress target_address = 0;
    iree_device_size_t addressable_length = 0;
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_resolve_bda_ref(
        binding_table, fill_buffer->target_ref, IREE_SV("fill target"),
        bda_binding_cache, &target_address, &addressable_length));
    if (addressable_length != target_length) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "resolved Vulkan command buffer fill spans differ "
          "(native_length=%" PRIu64 ", BDA_length=%" PRIdsz ")",
          (uint64_t)target_length, addressable_length);
    }
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_builtins_record_fill_edges(
        builtins, native_command_buffer, target_address, target_length,
        fill_buffer->pattern, fill_buffer->pattern_length));
    if (target_length > UINT64_MAX - target_offset) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "Vulkan native fill range overflows");
    }
    const VkDeviceSize target_end = target_offset + target_length;
    const VkDeviceSize aligned_target_offset =
        iree_device_align(target_offset, sizeof(uint32_t));
    const VkDeviceSize aligned_target_end =
        target_end & ~(VkDeviceSize)(sizeof(uint32_t) - 1);
    if (aligned_target_offset >= aligned_target_end) {
      return iree_ok_status();
    }
    target_offset = aligned_target_offset;
    target_length = aligned_target_end - aligned_target_offset;
  } else if (fill_buffer->strategy ==
             IREE_HAL_VULKAN_TRANSFER_STRATEGY_NATIVE) {
    if (!is_dword_aligned) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "Vulkan native fill lost its recorded dword-alignment invariant");
    }
  } else {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "invalid Vulkan fill transfer strategy %u",
                            (uint32_t)fill_buffer->strategy);
  }

  uint32_t pattern = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_expand_fill_pattern(
      fill_buffer->pattern, fill_buffer->pattern_length, &pattern));
  iree_vkCmdFillBuffer(IREE_VULKAN_DEVICE(syms), native_command_buffer,
                       target_handle, target_offset, target_length, pattern);
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_command_buffer_update_publication_length(
    iree_hal_vulkan_transfer_strategy_t strategy,
    iree_host_size_t source_data_length,
    iree_device_size_t* out_publication_length) {
  *out_publication_length = 0;
  if (!iree_hal_vulkan_transfer_strategy_uses_compute(strategy)) {
    return iree_ok_status();
  }
  if (!iree_device_size_checked_align((iree_device_size_t)source_data_length,
                                      sizeof(uint64_t),
                                      out_publication_length)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan command buffer update publication length overflows");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_command_buffer_publish_update_source(
    const iree_hal_vulkan_command_update_buffer_t* update_buffer,
    iree_hal_vulkan_command_buffer_bda_recording_state_t* bda_recording_state,
    VkDeviceAddress* out_source_address) {
  *out_source_address = 0;
  iree_device_size_t publication_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_update_publication_length(
      update_buffer->strategy, update_buffer->source_data_length,
      &publication_length));
  if (publication_length == 0) return iree_ok_status();

  iree_byte_span_t publication_span = iree_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_allocate_bda_publication(
      bda_recording_state, publication_length, &publication_span,
      out_source_address));
  if ((*out_source_address & (sizeof(uint32_t) - 1)) != 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Vulkan update publication address 0x%" PRIx64
                            " is not dword aligned",
                            (uint64_t)*out_source_address);
  }
  memcpy(publication_span.data, update_buffer->source_data,
         update_buffer->source_data_length);
  memset(publication_span.data + update_buffer->source_data_length, 0,
         publication_span.data_length - update_buffer->source_data_length);
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_command_buffer_record_update_chunks(
    const iree_hal_vulkan_device_syms_t* syms,
    VkCommandBuffer native_command_buffer, VkBuffer target_handle,
    VkDeviceSize target_offset, VkDeviceSize length, const uint8_t* source_data,
    iree_host_size_t source_data_offset) {
  if (length == 0) return iree_ok_status();
  if (length > (VkDeviceSize)(IREE_HOST_SIZE_MAX - source_data_offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan native update source offset overflows");
  }

  const VkDeviceSize max_update_length = 65536;
  VkDeviceSize remaining_length = length;
  VkDeviceSize update_offset = target_offset;
  iree_host_size_t update_source_offset = source_data_offset;
  while (remaining_length != 0) {
    const VkDeviceSize chunk_length =
        iree_min(remaining_length, max_update_length);
    iree_vkCmdUpdateBuffer(IREE_VULKAN_DEVICE(syms), native_command_buffer,
                           target_handle, update_offset, chunk_length,
                           source_data + update_source_offset);
    update_offset += chunk_length;
    update_source_offset += (iree_host_size_t)chunk_length;
    remaining_length -= chunk_length;
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_command_buffer_record_update_native(
    const iree_hal_vulkan_device_syms_t* syms,
    const iree_hal_vulkan_builtins_t* builtins,
    VkCommandBuffer native_command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_vulkan_command_t* command,
    iree_hal_vulkan_command_buffer_bda_recording_state_t* bda_recording_state,
    iree_hal_vulkan_command_buffer_bda_binding_cache_t* bda_binding_cache) {
  const iree_hal_vulkan_command_update_buffer_t* update_buffer =
      iree_hal_vulkan_command_update_buffer_payload(command);
  iree_hal_buffer_ref_t resolved_target_ref;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_binding_table_resolve_ref(
      binding_table, update_buffer->target_ref, &resolved_target_ref));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(resolved_target_ref.buffer),
      IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET));
  if (resolved_target_ref.length == 0) return iree_ok_status();
  if (resolved_target_ref.length != update_buffer->source_data_length) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "resolved Vulkan command buffer update span differs from captured "
        "source data (target_length=%" PRIdsz ", source_length=%" PRIhsz ")",
        resolved_target_ref.length, update_buffer->source_data_length);
  }

  if (iree_hal_vulkan_transfer_strategy_uses_compute(update_buffer->strategy)) {
    if (!builtins) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "Vulkan compute update requires built-in pipelines");
    }
    VkDeviceAddress target_address = 0;
    iree_device_size_t target_length = 0;
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_resolve_bda_ref(
        binding_table, update_buffer->target_ref, IREE_SV("update target"),
        bda_binding_cache, &target_address, &target_length));
    if (target_length != update_buffer->source_data_length) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "resolved Vulkan compute update span differs from captured source "
          "data (target_length=%" PRIu64 ", source_length=%" PRIhsz ")",
          (uint64_t)target_length, update_buffer->source_data_length);
    }
    VkDeviceAddress source_address = 0;
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_publish_update_source(
        update_buffer, bda_recording_state, &source_address));
    if (update_buffer->strategy ==
        IREE_HAL_VULKAN_TRANSFER_STRATEGY_INDIRECT_COMPUTE) {
      VkDeviceAddress publication_address = 0;
      IREE_RETURN_IF_ERROR(
          iree_hal_vulkan_command_buffer_publish_copy_parameters_for_addresses(
              source_address, target_address, target_length,
              bda_recording_state, &publication_address));
      iree_hal_vulkan_command_buffer_record_bda_publication_barrier(
          syms, native_command_buffer, bda_recording_state);
      return iree_hal_vulkan_byte_transfer_record_copy_indirect(
          &builtins->byte_transfer_pipelines, native_command_buffer,
          publication_address, target_length);
    }
    iree_hal_vulkan_command_buffer_record_bda_publication_barrier(
        syms, native_command_buffer, bda_recording_state);
    return iree_hal_vulkan_byte_transfer_record_copy(
        &builtins->byte_transfer_pipelines, native_command_buffer,
        source_address, target_address, target_length);
  }

  VkBuffer target_handle = VK_NULL_HANDLE;
  VkDeviceSize target_offset = 0;
  VkDeviceSize target_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_resolve_native_buffer_ref(
      binding_table, update_buffer->target_ref, IREE_SV("update target"),
      &target_handle, &target_offset, &target_length));
  if (target_length == 0) return iree_ok_status();
  if (target_length != update_buffer->source_data_length) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "resolved Vulkan command buffer update span differs from captured "
        "source data (target_length=%" PRIu64 ", source_length=%" PRIhsz ")",
        (uint64_t)target_length, update_buffer->source_data_length);
  }

  VkDeviceSize update_offset = target_offset;
  VkDeviceSize update_length = target_length;
  iree_host_size_t source_data_offset = 0;
  const bool is_dword_aligned = target_offset % sizeof(uint32_t) == 0 &&
                                target_length % sizeof(uint32_t) == 0;
  if (update_buffer->strategy ==
      IREE_HAL_VULKAN_TRANSFER_STRATEGY_EDGE_PATCHED_NATIVE) {
    if (is_dword_aligned) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "Vulkan edge-patched update lost its recorded alignment invariant");
    }
    if (!builtins) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "Vulkan unaligned native update requires built-in pipelines");
    }
    VkDeviceAddress target_address = 0;
    iree_device_size_t addressable_length = 0;
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_resolve_bda_ref(
        binding_table, update_buffer->target_ref, IREE_SV("update target"),
        bda_binding_cache, &target_address, &addressable_length));
    if (addressable_length != target_length) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "resolved Vulkan command buffer update spans differ "
          "(native_length=%" PRIu64 ", BDA_length=%" PRIdsz ")",
          (uint64_t)target_length, addressable_length);
    }
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_builtins_record_update_edges(
        builtins, native_command_buffer, target_address, target_length,
        update_buffer->source_data, update_buffer->source_data_length));
    if (target_length > UINT64_MAX - target_offset) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "Vulkan native update range overflows");
    }
    const VkDeviceSize target_end = target_offset + target_length;
    const VkDeviceSize aligned_target_offset =
        iree_device_align(target_offset, sizeof(uint32_t));
    const VkDeviceSize aligned_target_end =
        target_end & ~(VkDeviceSize)(sizeof(uint32_t) - 1);
    if (aligned_target_offset >= aligned_target_end) {
      update_length = 0;
    } else {
      update_offset = aligned_target_offset;
      update_length = aligned_target_end - aligned_target_offset;
      source_data_offset =
          (iree_host_size_t)(aligned_target_offset - target_offset);
    }
  } else if (update_buffer->strategy ==
             IREE_HAL_VULKAN_TRANSFER_STRATEGY_NATIVE) {
    if (!is_dword_aligned) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "Vulkan native update lost its recorded dword-alignment invariant");
    }
  } else {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "invalid Vulkan update transfer strategy %u",
                            (uint32_t)update_buffer->strategy);
  }

  return iree_hal_vulkan_command_buffer_record_update_chunks(
      syms, native_command_buffer, target_handle, update_offset, update_length,
      update_buffer->source_data, source_data_offset);
}

static iree_status_t iree_hal_vulkan_command_buffer_record_copy_native(
    const iree_hal_vulkan_device_syms_t* syms,
    const iree_hal_vulkan_builtins_t* builtins,
    VkCommandBuffer native_command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_vulkan_command_t* command,
    iree_hal_vulkan_command_buffer_bda_recording_state_t* bda_recording_state,
    iree_hal_vulkan_command_buffer_bda_binding_cache_t* bda_binding_cache) {
  const iree_hal_vulkan_command_copy_buffer_t* copy_buffer =
      iree_hal_vulkan_command_copy_buffer_payload(command);
  return iree_hal_vulkan_command_buffer_record_copy_native_refs(
      syms, builtins, native_command_buffer, binding_table,
      copy_buffer->source_ref, IREE_SV("copy source"), copy_buffer->target_ref,
      IREE_SV("copy target"), copy_buffer->strategy, bda_recording_state,
      bda_binding_cache);
}

VkQueueFlags iree_hal_vulkan_command_buffer_required_queue_flags(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_vulkan_command_buffer_t* command_buffer =
      iree_hal_vulkan_command_buffer_cast(base_command_buffer);
  return command_buffer->required_queue_flags;
}

static void iree_hal_vulkan_command_buffer_record_execution_barrier_native(
    const iree_hal_vulkan_device_syms_t* syms,
    VkCommandBuffer native_command_buffer,
    const iree_hal_vulkan_command_t* command) {
  const iree_hal_vulkan_command_execution_barrier_t* execution_barrier =
      iree_hal_vulkan_command_execution_barrier_payload(command);
  // HAL barriers without memory or buffer payload are execution dependencies
  // only. Memory visibility is requested by the barrier payloads.
  const bool has_memory_visibility =
      execution_barrier->memory_barrier_count != 0 ||
      execution_barrier->buffer_barrier_count != 0 ||
      execution_barrier->flags != IREE_HAL_EXECUTION_BARRIER_FLAG_NONE;
  const bool acquire_system_scope =
      iree_any_bit_set(execution_barrier->flags,
                       IREE_HAL_EXECUTION_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE) ||
      iree_any_bit_set(execution_barrier->source_stage_mask,
                       IREE_HAL_EXECUTION_STAGE_HOST);
  const bool release_system_scope =
      iree_any_bit_set(execution_barrier->flags,
                       IREE_HAL_EXECUTION_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE) ||
      iree_any_bit_set(execution_barrier->target_stage_mask,
                       IREE_HAL_EXECUTION_STAGE_HOST);
  iree_hal_vulkan_barrier_t barrier = {
      .flags = execution_barrier->transfer_barrier.flags,
      .source_stage_mask = execution_barrier->source_stage_mask,
      .source_access_mask =
          has_memory_visibility ? VK_ACCESS_2_MEMORY_WRITE_BIT : 0,
      .target_stage_mask = execution_barrier->target_stage_mask,
      .target_access_mask =
          has_memory_visibility
              ? VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT
              : 0,
  };
  if (acquire_system_scope) {
    barrier.source_stage_mask |= IREE_HAL_EXECUTION_STAGE_HOST;
    barrier.source_access_mask |= VK_ACCESS_2_HOST_WRITE_BIT;
  }
  if (release_system_scope) {
    barrier.target_stage_mask |= IREE_HAL_EXECUTION_STAGE_HOST;
    barrier.target_access_mask |=
        VK_ACCESS_2_HOST_READ_BIT | VK_ACCESS_2_HOST_WRITE_BIT;
  }
  iree_hal_vulkan_barrier_record(syms, native_command_buffer, &barrier);
}

static iree_status_t
iree_hal_vulkan_command_buffer_resolve_indirect_parameters_buffer(
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_buffer_ref_t workgroup_count_ref, VkBuffer* out_handle,
    VkDeviceSize* out_offset) {
  VkBuffer parameter_handle = VK_NULL_HANDLE;
  VkDeviceSize parameter_offset = 0;
  VkDeviceSize parameter_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_resolve_native_buffer_ref(
      binding_table, workgroup_count_ref, IREE_SV("indirect parameters"),
      &parameter_handle, &parameter_offset, &parameter_length));
  if (parameter_offset % sizeof(uint32_t) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan indirect workgroup parameter offset must be 4-byte aligned");
  }
  if (parameter_length < sizeof(uint32_t[3])) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan indirect workgroup parameter buffer must contain at least "
        "uint32_t[3]");
  }

  *out_handle = parameter_handle;
  *out_offset = parameter_offset;
  return iree_ok_status();
}

static iree_status_t
iree_hal_vulkan_command_buffer_record_dispatch_issue_native(
    const iree_hal_vulkan_device_syms_t* syms,
    VkCommandBuffer native_command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_vulkan_command_t* command,
    const iree_hal_vulkan_command_buffer_dispatch_profile_marker_t*
        profile_marker) {
  const iree_hal_vulkan_command_dispatch_t* dispatch =
      iree_hal_vulkan_command_dispatch_payload(command);
  iree_status_t status = iree_ok_status();
  if (profile_marker && profile_marker->query_pool) {
    iree_vkCmdWriteTimestamp2(IREE_VULKAN_DEVICE(syms), native_command_buffer,
                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                              profile_marker->query_pool,
                              profile_marker->start_query);
  }
  if (iree_hal_dispatch_uses_indirect_parameters(dispatch->flags)) {
    VkBuffer parameter_handle = VK_NULL_HANDLE;
    VkDeviceSize parameter_offset = 0;
    status = iree_hal_vulkan_command_buffer_resolve_indirect_parameters_buffer(
        binding_table, dispatch->config.workgroup_count_ref, &parameter_handle,
        &parameter_offset);
    if (iree_status_is_ok(status)) {
      iree_vkCmdDispatchIndirect(IREE_VULKAN_DEVICE(syms),
                                 native_command_buffer, parameter_handle,
                                 parameter_offset);
    }
  } else {
    iree_vkCmdDispatch(IREE_VULKAN_DEVICE(syms), native_command_buffer,
                       dispatch->config.workgroup_count[0],
                       dispatch->config.workgroup_count[1],
                       dispatch->config.workgroup_count[2]);
  }
  if (iree_status_is_ok(status) && profile_marker &&
      profile_marker->query_pool) {
    iree_vkCmdWriteTimestamp2(IREE_VULKAN_DEVICE(syms), native_command_buffer,
                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                              profile_marker->query_pool,
                              profile_marker->end_query);
  }
  return status;
}

static iree_status_t iree_hal_vulkan_command_buffer_publish_bda_dispatch_table(
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_vulkan_command_t* command,
    const iree_hal_vulkan_pipeline_t* pipeline,
    iree_hal_vulkan_command_buffer_bda_recording_state_t* bda_recording_state,
    iree_hal_vulkan_command_buffer_bda_binding_cache_t* bda_binding_cache,
    VkDeviceAddress* out_binding_table_address) {
  *out_binding_table_address = 0;
  const iree_hal_vulkan_command_dispatch_t* dispatch =
      iree_hal_vulkan_command_dispatch_payload(command);
  const iree_host_size_t binding_count = pipeline->bda.binding_count_known
                                             ? pipeline->binding_count
                                             : dispatch->binding_count;
  iree_device_size_t binding_table_length = 0;
  if (!iree_device_size_checked_mul((iree_device_size_t)binding_count,
                                    sizeof(uint64_t), &binding_table_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan BDA binding table length overflows");
  }

  iree_byte_span_t binding_table_span = iree_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_allocate_bda_publication(
      bda_recording_state, binding_table_length, &binding_table_span,
      out_binding_table_address));
  if (binding_count == 0) return iree_ok_status();

  const iree_hal_buffer_ref_t* bindings =
      iree_hal_vulkan_command_dispatch_bindings(dispatch);
  uint64_t* published_binding_table = (uint64_t*)binding_table_span.data;
  for (iree_host_size_t i = 0; i < binding_count; ++i) {
    VkDeviceAddress device_address = 0;
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_resolve_bda_binding(
        binding_table, bindings[i], i, pipeline, bda_binding_cache,
        &device_address));
    published_binding_table[i] = device_address;
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_command_buffer_resolve_atomic_target(
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_vulkan_command_atomic_t* atomic,
    iree_hal_vulkan_command_buffer_bda_binding_cache_t* bda_binding_cache,
    VkDeviceAddress* out_target_address) {
  iree_device_size_t target_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_resolve_bda_ref(
      binding_table, atomic->target_ref, IREE_SV("atomic target"),
      bda_binding_cache, out_target_address, &target_length));
  const iree_device_size_t required_length =
      iree_hal_atomic_width_byte_count(atomic->params.width);
  if (target_length < required_length) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan command buffer atomic target has length %" PRIdsz " but %" PRIu8
        "-bit operation requires %" PRIdsz " bytes",
        target_length, atomic->params.width, required_length);
  }
  return iree_hal_vulkan_atomic_validate_target_address(*out_target_address,
                                                        atomic->params.width);
}

static iree_status_t iree_hal_vulkan_command_buffer_publish_atomic_target(
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_vulkan_command_atomic_t* atomic,
    iree_hal_vulkan_command_buffer_bda_recording_state_t* bda_recording_state,
    iree_hal_vulkan_command_buffer_bda_binding_cache_t* bda_binding_cache,
    VkDeviceAddress* out_publication_address) {
  iree_byte_span_t publication_span = iree_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_allocate_bda_publication(
      bda_recording_state, sizeof(uint64_t), &publication_span,
      out_publication_address));
  if ((*out_publication_address & (sizeof(uint64_t) - 1)) != 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan atomic target publication address 0x%" PRIx64
        " is not naturally aligned",
        (uint64_t)*out_publication_address);
  }

  VkDeviceAddress target_address = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_resolve_atomic_target(
      binding_table, atomic, bda_binding_cache, &target_address));
  memcpy(publication_span.data, &target_address, sizeof(target_address));
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_command_buffer_validate_bda_publication(
    const iree_hal_vulkan_command_buffer_t* command_buffer,
    const iree_hal_vulkan_command_buffer_bda_publication_t* bda_publication) {
  if (command_buffer->bda_publication_length != 0 && !bda_publication) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Vulkan BDA publication storage is required");
  }
  if (bda_publication && bda_publication->host_span.data_length !=
                             command_buffer->bda_publication_length) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Vulkan BDA publication storage has %" PRIhsz
                            " bytes but command buffer requires %" PRIhsz
                            " bytes",
                            bda_publication->host_span.data_length,
                            command_buffer->bda_publication_length);
  }
  return iree_ok_status();
}

iree_status_t iree_hal_vulkan_command_buffer_publish_bda_replay_data(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_vulkan_command_buffer_bda_publication_t* bda_publication,
    iree_hal_vulkan_command_buffer_bda_binding_cache_t* bda_binding_cache) {
  IREE_ASSERT_ARGUMENT(base_command_buffer);
  iree_hal_vulkan_command_buffer_t* command_buffer =
      iree_hal_vulkan_command_buffer_cast(base_command_buffer);
  if (command_buffer->state != IREE_HAL_VULKAN_COMMAND_BUFFER_STATE_ENDED) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan BDA publication requires ended command buffer");
  }
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_validate_bda_publication(
      command_buffer, bda_publication));

  iree_hal_vulkan_command_buffer_bda_recording_state_t bda_recording_state = {
      .host_span =
          bda_publication ? bda_publication->host_span : iree_byte_span_empty(),
      .device_address = bda_publication ? bda_publication->device_address : 0,
  };

  iree_status_t status = iree_ok_status();
  iree_hal_vulkan_command_buffer_iterator_t iterator =
      iree_hal_vulkan_command_buffer_iterator(command_buffer);
  const iree_hal_vulkan_command_t* command = NULL;
  while (iree_status_is_ok(status) &&
         iree_hal_vulkan_command_buffer_iterator_next(
             &iterator, &command, /*out_command_index=*/NULL)) {
    switch (command->type) {
      case IREE_HAL_VULKAN_COMMAND_TYPE_FILL_BUFFER: {
        const iree_hal_vulkan_command_fill_buffer_t* fill_buffer =
            iree_hal_vulkan_command_fill_buffer_payload(command);
        if (fill_buffer->strategy ==
            IREE_HAL_VULKAN_TRANSFER_STRATEGY_INDIRECT_COMPUTE) {
          VkDeviceAddress publication_address = 0;
          status = iree_hal_vulkan_command_buffer_publish_fill_parameters(
              binding_table, fill_buffer, &bda_recording_state,
              bda_binding_cache, &publication_address);
        }
        break;
      }
      case IREE_HAL_VULKAN_COMMAND_TYPE_UPDATE_BUFFER: {
        const iree_hal_vulkan_command_update_buffer_t* update_buffer =
            iree_hal_vulkan_command_update_buffer_payload(command);
        VkDeviceAddress source_address = 0;
        status = iree_hal_vulkan_command_buffer_publish_update_source(
            update_buffer, &bda_recording_state, &source_address);
        if (iree_status_is_ok(status) &&
            update_buffer->strategy ==
                IREE_HAL_VULKAN_TRANSFER_STRATEGY_INDIRECT_COMPUTE) {
          VkDeviceAddress target_address = 0;
          iree_device_size_t target_length = 0;
          status = iree_hal_vulkan_command_buffer_resolve_bda_ref(
              binding_table, update_buffer->target_ref,
              IREE_SV("update target"), bda_binding_cache, &target_address,
              &target_length);
          if (iree_status_is_ok(status) &&
              target_length != update_buffer->source_data_length) {
            status = iree_make_status(
                IREE_STATUS_FAILED_PRECONDITION,
                "Vulkan indirect update resolved a dynamic command range");
          }
          if (iree_status_is_ok(status)) {
            VkDeviceAddress publication_address = 0;
            status =
                iree_hal_vulkan_command_buffer_publish_copy_parameters_for_addresses(
                    source_address, target_address, target_length,
                    &bda_recording_state, &publication_address);
          }
        }
        break;
      }
      case IREE_HAL_VULKAN_COMMAND_TYPE_COPY_BUFFER: {
        const iree_hal_vulkan_command_copy_buffer_t* copy_buffer =
            iree_hal_vulkan_command_copy_buffer_payload(command);
        if (copy_buffer->strategy ==
            IREE_HAL_VULKAN_TRANSFER_STRATEGY_INDIRECT_COMPUTE) {
          VkDeviceAddress publication_address = 0;
          status = iree_hal_vulkan_command_buffer_publish_copy_parameters(
              binding_table, copy_buffer, &bda_recording_state,
              bda_binding_cache, &publication_address);
        }
        break;
      }
      case IREE_HAL_VULKAN_COMMAND_TYPE_DISPATCH: {
        const iree_hal_vulkan_command_dispatch_t* dispatch =
            iree_hal_vulkan_command_dispatch_payload(command);
        VkDeviceAddress binding_table_address = 0;
        status = iree_hal_vulkan_command_buffer_publish_bda_dispatch_table(
            binding_table, command, dispatch->pipeline, &bda_recording_state,
            bda_binding_cache, &binding_table_address);
        break;
      }
      case IREE_HAL_VULKAN_COMMAND_TYPE_ATOMIC: {
        const iree_hal_vulkan_command_atomic_t* atomic =
            iree_hal_vulkan_command_atomic_payload(command);
        if (!atomic->target_ref.buffer) {
          VkDeviceAddress publication_address = 0;
          status = iree_hal_vulkan_command_buffer_publish_atomic_target(
              binding_table, atomic, &bda_recording_state, bda_binding_cache,
              &publication_address);
        }
        break;
      }
      default:
        break;
    }
  }
  if (iree_status_is_ok(status) && bda_recording_state.byte_offset !=
                                       command_buffer->bda_publication_length) {
    status =
        iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                         "Vulkan BDA publication populated %" PRIhsz
                         " bytes but command buffer requires %" PRIhsz " bytes",
                         bda_recording_state.byte_offset,
                         command_buffer->bda_publication_length);
  }
  return status;
}

static iree_status_t iree_hal_vulkan_command_buffer_record_atomic_native(
    const iree_hal_vulkan_device_syms_t* syms,
    const iree_hal_vulkan_atomic_pipelines_t* pipelines,
    VkCommandBuffer native_command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_vulkan_command_t* command,
    iree_hal_vulkan_command_buffer_bda_recording_state_t* bda_recording_state,
    iree_hal_vulkan_command_buffer_bda_binding_cache_t* bda_binding_cache) {
  const iree_hal_vulkan_command_atomic_t* atomic =
      iree_hal_vulkan_command_atomic_payload(command);

  VkDeviceAddress target_address = 0;
  iree_hal_vulkan_atomic_record_flags_t record_flags =
      IREE_HAL_VULKAN_ATOMIC_RECORD_FLAG_NONE;
  if (atomic->target_ref.buffer) {
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_resolve_atomic_target(
        binding_table, atomic, bda_binding_cache, &target_address));
  } else {
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_publish_atomic_target(
        binding_table, atomic, bda_recording_state, bda_binding_cache,
        &target_address));
    record_flags |= IREE_HAL_VULKAN_ATOMIC_RECORD_FLAG_INDIRECT_TARGET;
  }

  if (!atomic->target_ref.buffer) {
    iree_hal_vulkan_command_buffer_record_bda_publication_barrier(
        syms, native_command_buffer, bda_recording_state);
  }
  iree_hal_vulkan_atomic_record_command(
      syms, pipelines, native_command_buffer, atomic->source_stage_mask,
      atomic->target_stage_mask, target_address, record_flags,
      atomic->transfer_barrier.flags, atomic->params);
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_command_buffer_record_dispatch_bda_native(
    const iree_hal_vulkan_device_syms_t* syms,
    VkCommandBuffer native_command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_vulkan_command_t* command,
    const iree_hal_vulkan_pipeline_t* pipeline,
    iree_hal_vulkan_command_buffer_bda_recording_state_t* bda_recording_state,
    iree_hal_vulkan_command_buffer_bda_binding_cache_t* bda_binding_cache,
    const iree_hal_vulkan_command_buffer_dispatch_profile_marker_t*
        profile_marker) {
  const iree_hal_vulkan_command_dispatch_t* dispatch =
      iree_hal_vulkan_command_dispatch_payload(command);
  VkDeviceAddress binding_table_address = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_command_buffer_publish_bda_dispatch_table(
          binding_table, command, pipeline, bda_recording_state,
          bda_binding_cache, &binding_table_address));
  if (binding_table_address != 0) {
    iree_hal_vulkan_command_buffer_record_bda_publication_barrier(
        syms, native_command_buffer, bda_recording_state);
  }

  iree_hal_vulkan_bda_dispatch_root_v1_t root = {
      .binding_table_address = binding_table_address,
      .constants_address = 0,
      .binding_base = 0,
      .constant_base = 0,
      .flags = 0,
      .reserved0 = 0,
  };
  iree_vkCmdPushConstants(IREE_VULKAN_DEVICE(syms), native_command_buffer,
                          pipeline->layout, VK_SHADER_STAGE_COMPUTE_BIT,
                          pipeline->bda.root_push_constant_offset, sizeof(root),
                          &root);
  if (dispatch->constants_data_length != 0) {
    iree_vkCmdPushConstants(
        IREE_VULKAN_DEVICE(syms), native_command_buffer, pipeline->layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        pipeline->bda.constant_push_constant_offset,
        (uint32_t)dispatch->constants_data_length,
        iree_hal_vulkan_command_dispatch_constants_data(dispatch));
  }
  iree_vkCmdBindPipeline(IREE_VULKAN_DEVICE(syms), native_command_buffer,
                         VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->handle);
  return iree_hal_vulkan_command_buffer_record_dispatch_issue_native(
      syms, native_command_buffer, binding_table, command, profile_marker);
}

static iree_status_t iree_hal_vulkan_command_buffer_record_dispatch_native(
    const iree_hal_vulkan_device_syms_t* syms,
    VkCommandBuffer native_command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_vulkan_command_t* command,
    iree_hal_vulkan_command_buffer_bda_recording_state_t* bda_recording_state,
    iree_hal_vulkan_command_buffer_bda_binding_cache_t* bda_binding_cache,
    const iree_hal_vulkan_command_buffer_dispatch_profile_marker_t*
        profile_marker) {
  const iree_hal_vulkan_command_dispatch_t* dispatch =
      iree_hal_vulkan_command_dispatch_payload(command);
  const iree_hal_vulkan_pipeline_t* pipeline = dispatch->pipeline;
  return iree_hal_vulkan_command_buffer_record_dispatch_bda_native(
      syms, native_command_buffer, binding_table, command, pipeline,
      bda_recording_state, bda_binding_cache, profile_marker);
}

iree_status_t iree_hal_vulkan_command_buffer_record_native(
    iree_hal_command_buffer_t* base_command_buffer,
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    const iree_hal_vulkan_debug_utils_t* debug_utils,
    const iree_hal_vulkan_builtins_t* builtins,
    VkCommandBuffer native_command_buffer,
    VkCommandBufferUsageFlags usage_flags,
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_vulkan_command_buffer_bda_publication_t* bda_publication,
    iree_hal_vulkan_command_buffer_bda_binding_cache_t* bda_binding_cache,
    const iree_hal_vulkan_command_buffer_profile_marker_t* profile_marker,
    iree_allocator_t host_allocator) {
  IREE_ASSERT_ARGUMENT(syms);
  IREE_ASSERT_ARGUMENT(debug_utils);
  IREE_ASSERT_ARGUMENT(native_command_buffer);

  iree_hal_vulkan_command_buffer_t* command_buffer =
      iree_hal_vulkan_command_buffer_cast(base_command_buffer);
  if (command_buffer->state != IREE_HAL_VULKAN_COMMAND_BUFFER_STATE_ENDED) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Vulkan native command buffer recording requires "
                            "ended state");
  }
  if (command_buffer->command_count == 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_validate_bda_publication(
      command_buffer, bda_publication));

  iree_hal_vulkan_command_buffer_bda_recording_state_t bda_recording_state = {
      .host_span =
          bda_publication ? bda_publication->host_span : iree_byte_span_empty(),
      .device_address = bda_publication ? bda_publication->device_address : 0,
  };
  iree_status_t status = iree_ok_status();
  VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = usage_flags,
  };
  status = iree_vkBeginCommandBuffer(IREE_VULKAN_DEVICE(syms),
                                     native_command_buffer, &begin_info);
  if (iree_status_is_ok(status) && profile_marker &&
      profile_marker->query_pool && profile_marker->query_count != 0) {
    iree_vkCmdResetQueryPool(IREE_VULKAN_DEVICE(syms), native_command_buffer,
                             profile_marker->query_pool,
                             profile_marker->first_query,
                             profile_marker->query_count);
    if (profile_marker->queue_start_query !=
        IREE_HAL_VULKAN_PROFILE_QUERY_ABSENT) {
      iree_vkCmdWriteTimestamp2(IREE_VULKAN_DEVICE(syms), native_command_buffer,
                                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                                profile_marker->query_pool,
                                profile_marker->queue_start_query);
    }
  }
  uint32_t dispatch_query_ordinal = 0;
  iree_hal_vulkan_command_buffer_iterator_t iterator =
      iree_hal_vulkan_command_buffer_iterator(command_buffer);
  const iree_hal_vulkan_command_t* command = NULL;
  iree_host_size_t command_index = 0;
  while (iree_status_is_ok(status) &&
         iree_hal_vulkan_command_buffer_iterator_next(&iterator, &command,
                                                      &command_index)) {
    switch (command->type) {
      case IREE_HAL_VULKAN_COMMAND_TYPE_FILL_BUFFER:
        status = iree_hal_vulkan_command_buffer_record_fill_native(
            syms, builtins, native_command_buffer, binding_table, command,
            &bda_recording_state, bda_binding_cache);
        break;
      case IREE_HAL_VULKAN_COMMAND_TYPE_UPDATE_BUFFER:
        status = iree_hal_vulkan_command_buffer_record_update_native(
            syms, builtins, native_command_buffer, binding_table, command,
            &bda_recording_state, bda_binding_cache);
        break;
      case IREE_HAL_VULKAN_COMMAND_TYPE_COPY_BUFFER:
        status = iree_hal_vulkan_command_buffer_record_copy_native(
            syms, builtins, native_command_buffer, binding_table, command,
            &bda_recording_state, bda_binding_cache);
        break;
      case IREE_HAL_VULKAN_COMMAND_TYPE_DISPATCH: {
        iree_hal_vulkan_command_buffer_dispatch_profile_marker_t
            dispatch_profile_marker = {0};
        const iree_hal_vulkan_command_buffer_dispatch_profile_marker_t*
            dispatch_profile_marker_ptr = NULL;
        if (profile_marker && profile_marker->query_pool &&
            profile_marker->dispatch_base_query !=
                IREE_HAL_VULKAN_PROFILE_QUERY_ABSENT) {
          const iree_hal_vulkan_command_dispatch_t* dispatch =
              iree_hal_vulkan_command_dispatch_payload(command);
          if (profile_marker->command_buffer_id != 0 &&
              command_index > UINT32_MAX) {
            status = iree_make_status(
                IREE_STATUS_OUT_OF_RANGE,
                "Vulkan dispatch profile command index exceeds uint32_t");
            break;
          }
          const uint32_t profile_command_index =
              profile_marker->command_buffer_id != 0 ? (uint32_t)command_index
                                                     : UINT32_MAX;
          const bool profile_dispatch =
              iree_hal_vulkan_command_buffer_profile_filter_matches_dispatch(
                  profile_marker->recorder, profile_marker->scope,
                  profile_marker->command_buffer_id, profile_command_index,
                  dispatch->pipeline);
          if (profile_dispatch) {
            if (dispatch_query_ordinal >=
                profile_marker->dispatch_query_count) {
              status = iree_make_status(
                  IREE_STATUS_OUT_OF_RANGE,
                  "Vulkan dispatch profile query count is smaller than the "
                  "profiled dispatch count");
              break;
            }
            const uint32_t query_index = profile_marker->dispatch_base_query +
                                         dispatch_query_ordinal * 2;
            dispatch_profile_marker =
                (iree_hal_vulkan_command_buffer_dispatch_profile_marker_t){
                    .query_pool = profile_marker->query_pool,
                    .start_query = query_index,
                    .end_query = query_index + 1,
                };
            dispatch_profile_marker_ptr = &dispatch_profile_marker;
            dispatch_query_ordinal = dispatch_query_ordinal + 1;
          }
        }
        status = iree_hal_vulkan_command_buffer_record_dispatch_native(
            syms, native_command_buffer, binding_table, command,
            &bda_recording_state, bda_binding_cache,
            dispatch_profile_marker_ptr);
        break;
      }
      case IREE_HAL_VULKAN_COMMAND_TYPE_EXECUTION_BARRIER:
        iree_hal_vulkan_command_buffer_record_execution_barrier_native(
            syms, native_command_buffer, command);
        break;
      case IREE_HAL_VULKAN_COMMAND_TYPE_ATOMIC:
        status = iree_hal_vulkan_command_buffer_record_atomic_native(
            syms, &builtins->atomic_pipelines, native_command_buffer,
            binding_table, command, &bda_recording_state, bda_binding_cache);
        break;
      case IREE_HAL_VULKAN_COMMAND_TYPE_BEGIN_DEBUG_GROUP: {
        const iree_hal_vulkan_command_begin_debug_group_t* begin_debug_group =
            iree_hal_vulkan_command_begin_debug_group_payload(command);
        iree_hal_vulkan_debug_utils_begin_command_label(
            debug_utils, syms, native_command_buffer,
            iree_hal_vulkan_command_begin_debug_group_label(begin_debug_group),
            begin_debug_group->label_color);
        break;
      }
      case IREE_HAL_VULKAN_COMMAND_TYPE_END_DEBUG_GROUP:
        iree_hal_vulkan_debug_utils_end_command_label(debug_utils, syms,
                                                      native_command_buffer);
        break;
      default:
        status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                  "Vulkan command buffer contains non-native "
                                  "command kind %u",
                                  (uint32_t)command->type);
        break;
    }
  }
  if (iree_status_is_ok(status) && bda_recording_state.byte_offset !=
                                       command_buffer->bda_publication_length) {
    status =
        iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                         "Vulkan BDA recording populated %" PRIhsz
                         " bytes but command buffer requires %" PRIhsz " bytes",
                         bda_recording_state.byte_offset,
                         command_buffer->bda_publication_length);
  }
  if (iree_status_is_ok(status) && profile_marker &&
      profile_marker->query_pool &&
      profile_marker->dispatch_base_query !=
          IREE_HAL_VULKAN_PROFILE_QUERY_ABSENT &&
      dispatch_query_ordinal != profile_marker->dispatch_query_count) {
    status = iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan dispatch profile query count exceeds profiled dispatch count");
  }
  if (iree_status_is_ok(status) && profile_marker &&
      profile_marker->query_pool &&
      profile_marker->queue_end_query != IREE_HAL_VULKAN_PROFILE_QUERY_ABSENT) {
    iree_vkCmdWriteTimestamp2(IREE_VULKAN_DEVICE(syms), native_command_buffer,
                              VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                              profile_marker->query_pool,
                              profile_marker->queue_end_query);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vkEndCommandBuffer(IREE_VULKAN_DEVICE(syms),
                                     native_command_buffer);
  }

  return status;
}

static void iree_hal_vulkan_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_vulkan_command_buffer_t* command_buffer =
      iree_hal_vulkan_command_buffer_cast(base_command_buffer);
  iree_allocator_t host_allocator = command_buffer->host_allocator;
  iree_hal_resource_set_free(command_buffer->resource_set);
  iree_arena_deinitialize(&command_buffer->payload_arena);
  iree_allocator_free(host_allocator, command_buffer);
}

static iree_status_t iree_hal_vulkan_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_vulkan_command_buffer_t* command_buffer =
      iree_hal_vulkan_command_buffer_cast(base_command_buffer);
  if (command_buffer->state != IREE_HAL_VULKAN_COMMAND_BUFFER_STATE_INITIAL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Vulkan command buffer begin requires initial "
                            "state");
  }
  command_buffer->state = IREE_HAL_VULKAN_COMMAND_BUFFER_STATE_RECORDING;
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_vulkan_command_buffer_t* command_buffer =
      iree_hal_vulkan_command_buffer_cast(base_command_buffer);
  if (command_buffer->state != IREE_HAL_VULKAN_COMMAND_BUFFER_STATE_RECORDING) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Vulkan command buffer end requires recording "
                            "state");
  }
  iree_hal_resource_set_freeze(command_buffer->resource_set);
  command_buffer->state = IREE_HAL_VULKAN_COMMAND_BUFFER_STATE_ENDED;
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t* base_command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color,
    const iree_hal_label_location_t* location) {
  (void)location;
  iree_hal_vulkan_command_buffer_t* command_buffer =
      iree_hal_vulkan_command_buffer_cast(base_command_buffer);
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_validate_recording_state(
      command_buffer, IREE_SV("begin_debug_group")));

  iree_host_size_t payload_length = 0;
  if (!iree_host_size_checked_add(
          sizeof(iree_hal_vulkan_command_begin_debug_group_t), label.size,
          &payload_length) ||
      !iree_host_size_checked_add(payload_length, 1, &payload_length)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan command buffer debug-group label size overflows");
  }
  iree_host_size_t record_length = 0;
  iree_host_size_t payload_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_ensure_command_capacity(
      command_buffer, payload_length, &record_length, &payload_offset));

  void* payload = NULL;
  iree_hal_vulkan_command_buffer_append_command(
      command_buffer, IREE_HAL_VULKAN_COMMAND_TYPE_BEGIN_DEBUG_GROUP,
      record_length, payload_offset, /*out_command=*/NULL, &payload);
  iree_hal_vulkan_command_begin_debug_group_t* begin_debug_group =
      (iree_hal_vulkan_command_begin_debug_group_t*)payload;
  begin_debug_group->label_color = label_color;
  char* label_data =
      (char*)payload + sizeof(iree_hal_vulkan_command_begin_debug_group_t);
  if (label.size != 0) {
    memcpy(label_data, label.data, label.size);
  }
  label_data[label.size] = '\0';
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_command_buffer_end_debug_group(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_vulkan_command_buffer_t* command_buffer =
      iree_hal_vulkan_command_buffer_cast(base_command_buffer);
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_validate_recording_state(
      command_buffer, IREE_SV("end_debug_group")));

  iree_host_size_t record_length = 0;
  iree_host_size_t payload_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_ensure_command_capacity(
      command_buffer, /*payload_length=*/0, &record_length, &payload_offset));
  iree_hal_vulkan_command_buffer_append_command(
      command_buffer, IREE_HAL_VULKAN_COMMAND_TYPE_END_DEBUG_GROUP,
      record_length, payload_offset, /*out_command=*/NULL,
      /*out_payload=*/NULL);
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  (void)memory_barriers;
  (void)buffer_barriers;
  iree_hal_vulkan_command_buffer_t* command_buffer =
      iree_hal_vulkan_command_buffer_cast(base_command_buffer);
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_validate_recording_state(
      command_buffer, IREE_SV("execution_barrier")));
  const iree_hal_execution_barrier_flags_t supported_flags =
      IREE_HAL_EXECUTION_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE |
      IREE_HAL_EXECUTION_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE;
  if (flags & ~supported_flags) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported Vulkan command buffer execution "
                            "barrier flags: 0x%016" PRIx64,
                            flags & ~supported_flags);
  }
  if (source_stage_mask == 0 && target_stage_mask == 0 &&
      memory_barrier_count == 0 && buffer_barrier_count == 0) {
    return iree_ok_status();
  }

  iree_host_size_t record_length = 0;
  iree_host_size_t payload_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_ensure_command_capacity(
      command_buffer, sizeof(iree_hal_vulkan_command_execution_barrier_t),
      &record_length, &payload_offset));
  void* payload = NULL;
  iree_hal_vulkan_command_buffer_append_command(
      command_buffer, IREE_HAL_VULKAN_COMMAND_TYPE_EXECUTION_BARRIER,
      record_length, payload_offset, /*out_command=*/NULL, &payload);
  iree_hal_vulkan_command_execution_barrier_t* execution_barrier =
      (iree_hal_vulkan_command_execution_barrier_t*)payload;
  execution_barrier->source_stage_mask = source_stage_mask;
  execution_barrier->target_stage_mask = target_stage_mask;
  execution_barrier->flags = flags;
  execution_barrier->memory_barrier_count = memory_barrier_count;
  execution_barrier->buffer_barrier_count = buffer_barrier_count;
  iree_hal_vulkan_command_buffer_capture_transfer_barrier(
      command_buffer, source_stage_mask, target_stage_mask,
      &execution_barrier->transfer_barrier);
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_command_buffer_record_atomic(
    iree_hal_vulkan_command_buffer_t* command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_buffer_ref_t target_ref, iree_hal_vulkan_atomic_params_t params) {
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_validate_recording_state(
      command_buffer, IREE_SV("atomic operation")));
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_atomic_validate(
      command_buffer->atomic_pipelines, params));
  if (iree_any_bit_set(source_stage_mask | target_stage_mask,
                       IREE_HAL_EXECUTION_STAGE_HOST)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Vulkan built-in atomics do not support host execution stages");
  }

  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_track_buffer_ref(
      command_buffer, target_ref));
  iree_host_size_t record_length = 0;
  iree_host_size_t payload_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_ensure_command_capacity(
      command_buffer, sizeof(iree_hal_vulkan_command_atomic_t), &record_length,
      &payload_offset));
  if (!target_ref.buffer) {
    IREE_RETURN_IF_ERROR(
        iree_hal_vulkan_command_buffer_add_bda_publication_bytes(
            command_buffer, sizeof(uint64_t)));
  }

  void* payload = NULL;
  iree_hal_vulkan_command_buffer_append_command(
      command_buffer, IREE_HAL_VULKAN_COMMAND_TYPE_ATOMIC, record_length,
      payload_offset, /*out_command=*/NULL, &payload);
  iree_hal_vulkan_command_atomic_t* atomic =
      (iree_hal_vulkan_command_atomic_t*)payload;
  atomic->source_stage_mask = source_stage_mask;
  atomic->target_stage_mask = target_stage_mask;
  atomic->target_ref = target_ref;
  atomic->params = params;
  iree_hal_vulkan_command_buffer_capture_transfer_barrier(
      command_buffer, source_stage_mask, target_stage_mask,
      &atomic->transfer_barrier);
  command_buffer->required_queue_flags |= VK_QUEUE_COMPUTE_BIT;
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_command_buffer_atomic_wait(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_buffer_ref_t target_ref, iree_hal_atomic_wait_params_t params) {
  return iree_hal_vulkan_command_buffer_record_atomic(
      iree_hal_vulkan_command_buffer_cast(base_command_buffer),
      source_stage_mask, target_stage_mask, target_ref,
      iree_hal_vulkan_atomic_params_from_wait(params));
}

static iree_status_t iree_hal_vulkan_command_buffer_atomic_store(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_buffer_ref_t target_ref, iree_hal_atomic_store_params_t params) {
  return iree_hal_vulkan_command_buffer_record_atomic(
      iree_hal_vulkan_command_buffer_cast(base_command_buffer),
      source_stage_mask, target_stage_mask, target_ref,
      iree_hal_vulkan_atomic_params_from_store(params));
}

static iree_status_t iree_hal_vulkan_command_buffer_atomic_rmw(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_buffer_ref_t target_ref, iree_hal_atomic_rmw_params_t params) {
  return iree_hal_vulkan_command_buffer_record_atomic(
      iree_hal_vulkan_command_buffer_cast(base_command_buffer),
      source_stage_mask, target_stage_mask, target_ref,
      iree_hal_vulkan_atomic_params_from_rmw(params));
}

static iree_status_t iree_hal_vulkan_command_buffer_advise_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t buffer_ref, iree_hal_memory_advise_flags_t flags,
    uint64_t arg0, uint64_t arg1) {
  (void)base_command_buffer;
  (void)buffer_ref;
  (void)flags;
  (void)arg0;
  (void)arg1;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "Vulkan command buffer memory advice is unsupported");
}

static iree_status_t iree_hal_vulkan_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t target_ref, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  iree_hal_vulkan_command_buffer_t* command_buffer =
      iree_hal_vulkan_command_buffer_cast(base_command_buffer);
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_validate_recording_state(
      command_buffer, IREE_SV("fill_buffer")));
  if (flags != IREE_HAL_FILL_FLAG_NONE) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unsupported Vulkan command buffer fill flags: 0x%" PRIx64, flags);
  }
  if (!pattern) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Vulkan command buffer fill pattern is NULL");
  }
  if (pattern_length != 1 && pattern_length != 2 && pattern_length != 4) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan command buffer fill pattern length must be 1, 2, or 4 bytes "
        "(got %" PRIhsz ")",
        pattern_length);
  }

  const iree_hal_vulkan_transfer_strategy_t strategy =
      iree_hal_vulkan_command_buffer_fill_update_strategy_for_ref(target_ref);

  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_track_buffer_ref(
      command_buffer, target_ref));
  iree_host_size_t record_length = 0;
  iree_host_size_t payload_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_ensure_command_capacity(
      command_buffer, sizeof(iree_hal_vulkan_command_fill_buffer_t),
      &record_length, &payload_offset));
  if (strategy == IREE_HAL_VULKAN_TRANSFER_STRATEGY_INDIRECT_COMPUTE) {
    iree_device_size_t publication_length = 0;
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_byte_transfer_fill_publication_length(
        target_ref.length, &publication_length));
    IREE_RETURN_IF_ERROR(
        iree_hal_vulkan_command_buffer_add_bda_publication_bytes(
            command_buffer, publication_length));
  }

  void* payload = NULL;
  iree_hal_vulkan_command_buffer_append_command(
      command_buffer, IREE_HAL_VULKAN_COMMAND_TYPE_FILL_BUFFER, record_length,
      payload_offset, /*out_command=*/NULL, &payload);
  iree_hal_vulkan_command_fill_buffer_t* fill_buffer =
      (iree_hal_vulkan_command_fill_buffer_t*)payload;
  fill_buffer->target_ref = target_ref;
  memset(fill_buffer->pattern, 0, sizeof(fill_buffer->pattern));
  memcpy(fill_buffer->pattern, pattern, pattern_length);
  fill_buffer->pattern_length = pattern_length;
  fill_buffer->flags = flags;
  fill_buffer->strategy = strategy;
  iree_hal_vulkan_command_buffer_require_transfer_strategy(command_buffer,
                                                           strategy);
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_ref_t target_ref,
    iree_hal_update_flags_t flags) {
  iree_hal_vulkan_command_buffer_t* command_buffer =
      iree_hal_vulkan_command_buffer_cast(base_command_buffer);
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_validate_recording_state(
      command_buffer, IREE_SV("update_buffer")));
  if (flags != IREE_HAL_UPDATE_FLAG_NONE) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unsupported Vulkan command buffer update flags: 0x%" PRIx64, flags);
  }
  if (!source_buffer) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Vulkan command buffer update source is NULL");
  }
  if (target_ref.length == IREE_HAL_WHOLE_BUFFER) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan command buffer updates require an explicit target length");
  }
  if (target_ref.length > IREE_HOST_SIZE_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan command buffer update target length exceeds host size");
  }
  if (source_offset >
      IREE_HOST_SIZE_MAX - (iree_host_size_t)target_ref.length) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan command buffer update source range exceeds host size");
  }

  const iree_hal_vulkan_transfer_strategy_t strategy =
      iree_hal_vulkan_command_buffer_fill_update_strategy_for_ref(target_ref);

  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_track_buffer_ref(
      command_buffer, target_ref));
  iree_host_size_t record_length = 0;
  iree_host_size_t payload_offset = 0;
  iree_host_size_t payload_length = 0;
  if (!iree_host_size_checked_add(
          sizeof(iree_hal_vulkan_command_update_buffer_t),
          (iree_host_size_t)target_ref.length, &payload_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan update command record size overflows");
  }
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_ensure_command_capacity(
      command_buffer, payload_length, &record_length, &payload_offset));
  if (iree_hal_vulkan_transfer_strategy_uses_compute(strategy)) {
    iree_device_size_t publication_length = 0;
    IREE_RETURN_IF_ERROR(
        iree_hal_vulkan_command_buffer_update_publication_length(
            strategy, (iree_host_size_t)target_ref.length,
            &publication_length));
    IREE_RETURN_IF_ERROR(
        iree_hal_vulkan_command_buffer_add_bda_publication_bytes(
            command_buffer, publication_length));
  }
  if (strategy == IREE_HAL_VULKAN_TRANSFER_STRATEGY_INDIRECT_COMPUTE) {
    iree_device_size_t publication_length = 0;
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_byte_transfer_copy_publication_length(
        target_ref.length, &publication_length));
    IREE_RETURN_IF_ERROR(
        iree_hal_vulkan_command_buffer_add_bda_publication_bytes(
            command_buffer, publication_length));
  }

  void* payload = NULL;
  iree_hal_vulkan_command_buffer_append_command(
      command_buffer, IREE_HAL_VULKAN_COMMAND_TYPE_UPDATE_BUFFER, record_length,
      payload_offset, /*out_command=*/NULL, &payload);
  iree_hal_vulkan_command_update_buffer_t* update_buffer =
      (iree_hal_vulkan_command_update_buffer_t*)payload;
  void* source_data =
      (uint8_t*)payload + sizeof(iree_hal_vulkan_command_update_buffer_t);
  if (target_ref.length != 0) {
    memcpy(source_data, (const uint8_t*)source_buffer + source_offset,
           (iree_host_size_t)target_ref.length);
  }
  update_buffer->target_ref = target_ref;
  update_buffer->source_data = source_data;
  update_buffer->source_data_length = (iree_host_size_t)target_ref.length;
  update_buffer->flags = flags;
  update_buffer->strategy = strategy;
  iree_hal_vulkan_command_buffer_require_transfer_strategy(command_buffer,
                                                           strategy);
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t source_ref, iree_hal_buffer_ref_t target_ref,
    iree_hal_copy_flags_t flags) {
  iree_hal_vulkan_command_buffer_t* command_buffer =
      iree_hal_vulkan_command_buffer_cast(base_command_buffer);
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_validate_recording_state(
      command_buffer, IREE_SV("copy_buffer")));
  if (flags != IREE_HAL_COPY_FLAG_NONE) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unsupported Vulkan command buffer copy flags: 0x%" PRIx64, flags);
  }
  if (source_ref.length != target_ref.length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "copy spans between source and target must match "
                            "(source_length=%" PRIdsz ", target_length=%" PRIdsz
                            ")",
                            source_ref.length, target_ref.length);
  }

  const iree_hal_buffer_ref_t buffer_refs[2] = {source_ref, target_ref};
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_track_buffer_refs(
      command_buffer, IREE_ARRAYSIZE(buffer_refs), buffer_refs));
  iree_host_size_t record_length = 0;
  iree_host_size_t payload_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_ensure_command_capacity(
      command_buffer, sizeof(iree_hal_vulkan_command_copy_buffer_t),
      &record_length, &payload_offset));
  const iree_hal_vulkan_transfer_strategy_t strategy =
      iree_hal_vulkan_command_buffer_copy_strategy_for_refs(source_ref,
                                                            target_ref);
  if (strategy == IREE_HAL_VULKAN_TRANSFER_STRATEGY_INDIRECT_COMPUTE) {
    iree_device_size_t publication_length = 0;
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_byte_transfer_copy_publication_length(
        source_ref.length, &publication_length));
    IREE_RETURN_IF_ERROR(
        iree_hal_vulkan_command_buffer_add_bda_publication_bytes(
            command_buffer, publication_length));
  }
  void* payload = NULL;
  iree_hal_vulkan_command_buffer_append_command(
      command_buffer, IREE_HAL_VULKAN_COMMAND_TYPE_COPY_BUFFER, record_length,
      payload_offset, /*out_command=*/NULL, &payload);
  iree_hal_vulkan_command_copy_buffer_t* copy_buffer =
      (iree_hal_vulkan_command_copy_buffer_t*)payload;
  copy_buffer->source_ref = source_ref;
  copy_buffer->target_ref = target_ref;
  copy_buffer->flags = flags;
  copy_buffer->strategy = strategy;
  iree_hal_vulkan_command_buffer_require_transfer_strategy(command_buffer,
                                                           strategy);
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_command_buffer_collective(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_channel_t* channel,
    iree_hal_collective_op_t op, uint32_t param, iree_hal_buffer_ref_t send_ref,
    iree_hal_buffer_ref_t recv_ref, iree_device_size_t element_count) {
  (void)base_command_buffer;
  (void)channel;
  (void)op;
  (void)param;
  (void)send_ref;
  (void)recv_ref;
  (void)element_count;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "Vulkan command buffer collectives are unsupported");
}

static iree_status_t iree_hal_vulkan_command_buffer_validate_dispatch_bda(
    const iree_hal_vulkan_pipeline_t* pipeline,
    iree_const_byte_span_t constants, iree_hal_buffer_ref_list_t bindings) {
  return iree_hal_vulkan_pipeline_validate_bda_dispatch(
      pipeline, constants, bindings.count,
      IREE_SV("Vulkan command buffer dispatch"));
}

static iree_status_t iree_hal_vulkan_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable,
    iree_hal_executable_function_t function_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  iree_hal_vulkan_command_buffer_t* command_buffer =
      iree_hal_vulkan_command_buffer_cast(base_command_buffer);
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_validate_recording_state(
      command_buffer, IREE_SV("dispatch")));
  if (iree_hal_dispatch_uses_indirect_arguments(flags) ||
      iree_any_bit_set(flags, IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Vulkan command buffer custom dispatch arguments are unsupported");
  }
  const iree_hal_dispatch_flags_t supported_flags =
      IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS |
      IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS |
      IREE_HAL_DISPATCH_FLAG_ALLOW_INLINE_EXECUTION |
      IREE_HAL_DISPATCH_FLAG_BORROW_RESOURCE_LIFETIMES;
  if (iree_any_bit_set(flags, ~supported_flags)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unsupported Vulkan command buffer dispatch flags: 0x%" PRIx64, flags);
  }
  const iree_hal_dispatch_flags_t indirect_parameter_flags =
      flags & (IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS |
               IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS);
  if (indirect_parameter_flags ==
      (IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS |
       IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan dispatch cannot use both static and dynamic indirect "
        "workgroup parameters");
  }
  if (config.workgroup_size[0] != 0 || config.workgroup_size[1] != 0 ||
      config.workgroup_size[2] != 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Vulkan command buffer dispatch workgroup size overrides are "
        "unsupported");
  }
  if (config.dynamic_workgroup_local_memory != 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Vulkan command buffer dispatch dynamic workgroup local memory is "
        "unsupported");
  }
  if (constants.data_length > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan command buffer dispatch constants exceed Vulkan limit %u",
        UINT32_MAX);
  }
  if (bindings.count != 0 && !bindings.values) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan command buffer dispatch binding storage is NULL");
  }
  if (!iree_hal_vulkan_executable_isa(executable)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan command buffer dispatch executable is not a Vulkan executable");
  }

  const iree_hal_vulkan_pipeline_t* pipeline = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_executable_lookup_pipeline(
      executable, function_ordinal, &pipeline));

  if (iree_hal_vulkan_command_buffer_validates(command_buffer)) {
    if (constants.data_length % sizeof(uint32_t) != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Vulkan command buffer dispatch constants must be 4-byte aligned");
    }
    if (constants.data_length > pipeline->constant_byte_length) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "Vulkan command buffer dispatch provides %" PRIhsz
                              " constant bytes but pipeline accepts at most %u",
                              constants.data_length,
                              pipeline->constant_byte_length);
    }
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_validate_dispatch_bda(
        pipeline, constants, bindings));
    if (iree_hal_dispatch_uses_indirect_parameters(flags)) {
      IREE_RETURN_IF_ERROR(
          iree_hal_vulkan_command_buffer_validate_indirect_parameters_ref(
              command_buffer, config.workgroup_count_ref));
    }
  }

  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_retain_resource(
      command_buffer, executable));
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_track_buffer_refs(
      command_buffer, bindings.count, bindings.values));
  if (iree_hal_dispatch_uses_indirect_parameters(flags)) {
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_track_buffer_ref(
        command_buffer, config.workgroup_count_ref));
  }

  iree_host_size_t payload_length = 0;
  iree_host_size_t constants_offset = 0;
  iree_host_size_t bindings_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_dispatch_payload_layout(
      constants, bindings, &payload_length, &constants_offset,
      &bindings_offset));
  iree_host_size_t record_length = 0;
  iree_host_size_t payload_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_ensure_command_capacity(
      command_buffer, payload_length, &record_length, &payload_offset));

  iree_status_t status = iree_ok_status();
  if (command_buffer->dispatch_count == IREE_HOST_SIZE_MAX) {
    status = iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan command buffer dispatch count exceeds host size");
  }
  if (iree_status_is_ok(status)) {
    const iree_host_size_t binding_count = pipeline->bda.binding_count_known
                                               ? pipeline->binding_count
                                               : bindings.count;
    iree_device_size_t binding_table_length = 0;
    if (!iree_device_size_checked_mul((iree_device_size_t)binding_count,
                                      sizeof(uint64_t),
                                      &binding_table_length)) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "Vulkan command buffer BDA binding table length overflows");
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_vulkan_command_buffer_add_bda_publication_bytes(
          command_buffer, binding_table_length);
    }
  }

  if (iree_status_is_ok(status)) {
    void* payload = NULL;
    iree_hal_vulkan_command_buffer_append_command(
        command_buffer, IREE_HAL_VULKAN_COMMAND_TYPE_DISPATCH, record_length,
        payload_offset, /*out_command=*/NULL, &payload);
    iree_hal_vulkan_command_dispatch_t* dispatch =
        (iree_hal_vulkan_command_dispatch_t*)payload;
    if (constants.data_length != 0) {
      void* constants_data = (uint8_t*)payload + constants_offset;
      memcpy(constants_data, constants.data, constants.data_length);
    }
    if (bindings.count != 0) {
      iree_hal_buffer_ref_t* binding_refs =
          (iree_hal_buffer_ref_t*)((uint8_t*)payload + bindings_offset);
      memcpy(binding_refs, bindings.values,
             bindings.count * sizeof(binding_refs[0]));
    }
    dispatch->executable = executable;
    dispatch->pipeline = pipeline;
    dispatch->function_ordinal = function_ordinal;
    dispatch->config = config;
    dispatch->constants_data_length = constants.data_length;
    dispatch->binding_count = bindings.count;
    dispatch->flags = flags;
    command_buffer->dispatch_count = command_buffer->dispatch_count + 1;
    command_buffer->required_queue_flags |= VK_QUEUE_COMPUTE_BIT;
  }
  return status;
}

static const iree_hal_command_buffer_vtable_t
    iree_hal_vulkan_command_buffer_vtable = {
        .destroy = iree_hal_vulkan_command_buffer_destroy,
        .begin = iree_hal_vulkan_command_buffer_begin,
        .end = iree_hal_vulkan_command_buffer_end,
        .begin_debug_group = iree_hal_vulkan_command_buffer_begin_debug_group,
        .end_debug_group = iree_hal_vulkan_command_buffer_end_debug_group,
        .execution_barrier = iree_hal_vulkan_command_buffer_execution_barrier,
        .atomic_wait = iree_hal_vulkan_command_buffer_atomic_wait,
        .atomic_store = iree_hal_vulkan_command_buffer_atomic_store,
        .atomic_rmw = iree_hal_vulkan_command_buffer_atomic_rmw,
        .advise_buffer = iree_hal_vulkan_command_buffer_advise_buffer,
        .fill_buffer = iree_hal_vulkan_command_buffer_fill_buffer,
        .update_buffer = iree_hal_vulkan_command_buffer_update_buffer,
        .copy_buffer = iree_hal_vulkan_command_buffer_copy_buffer,
        .collective = iree_hal_vulkan_command_buffer_collective,
        .dispatch = iree_hal_vulkan_command_buffer_dispatch,
};
