// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/command_buffer.h"

#include "iree/base/internal/arena.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/commands.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/dispatch.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/transfer.h"
#include "iree/hal/utils/resource_set.h"

enum {
  IREE_HAL_AMD_GPU_PM4_RECORD_BLOCK_SIZE = 16384,
  IREE_HAL_AMD_GPU_PM4_RECORD_DISPATCH = 1u << 0,
  IREE_HAL_AMD_GPU_PM4_RECORD_TRANSFER = 1u << 1,
  IREE_HAL_AMD_GPU_PM4_RECORD_JOIN_DMA = 1u << 2,
  IREE_HAL_AMD_GPU_PM4_SITE_QUEUE = 0,
  IREE_HAL_AMD_GPU_PM4_SITE_PROGRAM = 1,
  IREE_HAL_AMD_GPU_PM4_PATCH_ARGUMENT = 1u << 31,
};

// Only this allocation boundary has a link. Native records occupy its dense
// payload in order; oversized argument templates receive one larger block.
struct iree_hal_amd_gpu_pm4_record_block_t {
  // Following immutable block, or NULL at the end of the recording.
  iree_hal_amd_gpu_pm4_record_block_t* next;
  // Initialized payload extent, in bytes.
  uint32_t length;
  // Allocated payload capacity, in bytes.
  uint32_t capacity;
};

// Followed by patches, native words, alignment padding, and argument bytes.
typedef struct iree_hal_amd_gpu_pm4_record_t {
  // Complete eight-byte-aligned extent to the following record.
  uint32_t byte_length;
  // Native argument template extent, or zero for externally supplied arguments.
  uint32_t argument_length;
  // Required power-of-two native argument alignment.
  uint32_t argument_alignment;
  // Initialized native command extent in DWORDs.
  uint16_t word_count;
  // Number of captured address relocations preceding the native words.
  uint16_t patch_count;
  // Native kernarg user-data pair, or UINT16_MAX when absent/external.
  uint16_t kernarg_word_offset;
  // Native dispatch-ID user-data pair, or UINT16_MAX when absent.
  uint16_t dispatch_id_word_offset;
  // Byte offset from this header to arguments or a native DMA span recipe.
  uint32_t payload_offset;
  // Native publication properties; no HAL operation is replayed.
  uint32_t flags;
  // Relocations for device-populated, invocation-private launch geometry.
  struct {
    // First DMA packet, relative to this record's native command words.
    uint16_t word_offset;
    // Consecutive seven-DWORD copies; zero requires no argument fixups.
    uint16_t copy_count;
  } indirect;
} iree_hal_amd_gpu_pm4_record_t;
static_assert(sizeof(iree_hal_amd_gpu_pm4_record_t) == 32,
              "record headers occupy half a cache line");

typedef struct iree_hal_amd_gpu_pm4_patch_t {
  // Direct recorded buffer, or NULL for an invocation binding-table slot.
  iree_hal_buffer_t* buffer;
  // Byte offset relative to the recorded buffer or invocation binding.
  uint64_t offset;
  // Exactly one source is active, selected by buffer presence.
  union {
    // Qualified device-address slot for a direct buffer.
    iree_hal_buffer_native_binding_slot_t slot;
    // Invocation binding index for an indirect buffer.
    uint32_t binding;
  } source;
  // Byte destination; PATCH_ARGUMENT selects argument instead of command bytes.
  uint32_t target_offset;
  // Required power-of-two address alignment, one for ordinary shader bindings.
  uint32_t alignment;
  // Queue or program site selecting the captured native address slot.
  uint32_t site;
} iree_hal_amd_gpu_pm4_patch_t;
static_assert(sizeof(iree_hal_amd_gpu_pm4_patch_t) == 32,
              "two address relocations occupy one cache line");

typedef struct iree_hal_amd_gpu_pm4_binding_requirement_t {
  // Smallest table extent covering any indirect native kernarg use.
  uint64_t minimum_length;
  // Required queue/program sites; zero marks an unused binding-table hole.
  uint32_t sites;
} iree_hal_amd_gpu_pm4_binding_requirement_t;

// Native DMA span with one target relocation, or source then target for copies.
// UPDATE bytes immediately follow this fixed header. The logical extent may be
// WHOLE_BUFFER and is then resolved from the invocation's captured range.
typedef struct iree_hal_amd_gpu_pm4_record_transfer_t {
  // Logical byte extent, independent of firmware and publication chunk sizes.
  uint64_t length;
  // Four repeating fill bytes, unused for copies and updates.
  uint32_t pattern;
  // Admitted native DMA family: fill, copy, or captured update.
  iree_hal_transfer_operation_type_t type;
} iree_hal_amd_gpu_pm4_record_transfer_t;

typedef struct iree_hal_amd_gpu_pm4_command_buffer_t {
  // Public resource identity and optional recording validation.
  iree_hal_command_buffer_t base;
  // Allocator for native-template blocks and the resource-set block pool.
  iree_allocator_t host_allocator;
  // Device-local LDS capacity used only while admitting recorded dispatches.
  uint32_t maximum_group_segment_byte_length;
  // Copied physical cache encoding; no topology survives construction.
  iree_hal_amd_gpu_pm4_barrier_profile_t profile;
  // First immutable native-template block, or NULL for an empty recording.
  iree_hal_amd_gpu_pm4_record_block_t* head;
  // Current append block, never accessed by an invocation.
  iree_hal_amd_gpu_pm4_record_block_t* tail;
  // Consecutive barriers are combined before the next native command.
  iree_hal_amd_gpu_pm4_barrier_t pending_barrier;
  // Inline slot requirements established once while recording.
  iree_hal_amd_gpu_pm4_binding_requirement_t* requirements;
  // Blocks backing direct-resource ownership, separate from native templates.
  iree_arena_block_pool_t resource_blocks;
  // Retained direct buffers/executables, or NULL for UNRETAINED recording.
  iree_hal_resource_set_t* resources;
} iree_hal_amd_gpu_pm4_command_buffer_t;

static const iree_hal_command_buffer_vtable_t
    iree_hal_amd_gpu_pm4_command_buffer_vtable;

static void iree_hal_amd_gpu_pm4_command_buffer_destroy(
    iree_hal_command_buffer_t* base) {
  iree_hal_amd_gpu_pm4_command_buffer_t* buffer = (void*)base;
  iree_hal_resource_set_free(buffer->resources);
  iree_arena_block_pool_deinitialize(&buffer->resource_blocks);
  iree_hal_amd_gpu_pm4_record_block_t* block = buffer->head;
  while (block) {
    iree_hal_amd_gpu_pm4_record_block_t* next = block->next;
    iree_allocator_free(buffer->host_allocator, block);
    block = next;
  }
  iree_allocator_free(buffer->host_allocator, buffer);
}

iree_status_t iree_hal_amd_gpu_pm4_command_buffer_create(
    const iree_hal_queue_family_t* family, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t categories, iree_host_size_t binding_capacity,
    uint32_t maximum_group_segment_byte_length,
    const iree_hal_amd_gpu_pm4_barrier_profile_t* profile,
    iree_allocator_t host_allocator, iree_hal_command_buffer_t** out_buffer) {
  *out_buffer = NULL;
  iree_host_size_t size = 0, requirements_offset = 0, validation_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_amd_gpu_pm4_command_buffer_t), &size,
      IREE_STRUCT_FIELD(binding_capacity,
                        iree_hal_amd_gpu_pm4_binding_requirement_t,
                        &requirements_offset),
      IREE_STRUCT_FIELD(
          iree_hal_command_buffer_validation_state_size(mode, binding_capacity),
          uint8_t, &validation_offset)));
  iree_hal_amd_gpu_pm4_command_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, size, (void**)&buffer));
  iree_hal_command_buffer_initialize(
      family, mode, categories, binding_capacity,
      (uint8_t*)buffer + validation_offset,
      &iree_hal_amd_gpu_pm4_command_buffer_vtable, &buffer->base);
  buffer->host_allocator = host_allocator;
  buffer->maximum_group_segment_byte_length = maximum_group_segment_byte_length;
  buffer->profile = *profile;
  buffer->requirements = (void*)((uint8_t*)buffer + requirements_offset);
  iree_arena_block_pool_initialize(4096, host_allocator,
                                   &buffer->resource_blocks);
  iree_status_t status = iree_ok_status();
  if (!iree_any_bit_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED)) {
    status = iree_hal_resource_set_allocate(&buffer->resource_blocks,
                                            &buffer->resources);
  }
  if (iree_status_is_ok(status)) {
    *out_buffer = &buffer->base;
  } else {
    iree_hal_command_buffer_release(&buffer->base);
  }
  return status;
}

// Native-interface admission is separate from address commitment. The public
// boundary validates ranges, usages, and permissions; this backend additionally
// needs an actual prepared native site address slot, including for legacy
// buffers that can satisfy generic HAL validation without a native contract.
static iree_status_t iree_hal_amd_gpu_pm4_command_qualify_binding(
    const iree_hal_queue_family_t* family, iree_hal_buffer_t* buffer,
    uint32_t site, iree_hal_buffer_native_binding_slot_t* out_slot) {
  const iree_hal_memory_contract_t* contract = buffer->memory.contract;
  const uint64_t scope_id = (uint64_t)family->memory.queue_scope_id + site;
  if (!contract || !family->memory.domain ||
      contract->domain != family->memory.domain ||
      scope_id >= contract->scope_count) {
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "recorded buffer has no native site contract");
  }
  const uint16_t index =
      contract->scopes[scope_id]
          .bindings[IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS];
  if (index == IREE_HAL_BUFFER_NATIVE_BINDING_INDEX_NONE) {
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "recorded buffer has no prepared GPU address slot");
  }
  *out_slot = (iree_hal_buffer_native_binding_slot_t){
      .index = index, .type = IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS};
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_pm4_command_buffer_append(
    iree_hal_amd_gpu_pm4_command_buffer_t* buffer, uint16_t word_count,
    uint16_t patch_count, uint32_t argument_length, uint32_t argument_alignment,
    iree_hal_amd_gpu_pm4_record_t** out_record) {
  const uint32_t argument_offset = iree_host_align(
      sizeof(iree_hal_amd_gpu_pm4_record_t) +
          patch_count * sizeof(iree_hal_amd_gpu_pm4_patch_t) + word_count * 4,
      8);
  const uint32_t byte_length =
      iree_host_align(argument_offset + argument_length, 8);
  iree_hal_amd_gpu_pm4_record_block_t* block = buffer->tail;
  if (!block || byte_length > block->capacity - block->length) {
    const uint32_t capacity =
        iree_max(byte_length, IREE_HAL_AMD_GPU_PM4_RECORD_BLOCK_SIZE);
    block = NULL;
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
        buffer->host_allocator, sizeof(*block) + capacity, (void**)&block));
    *block = (iree_hal_amd_gpu_pm4_record_block_t){.capacity = capacity};
    if (buffer->tail) {
      buffer->tail->next = block;
    } else {
      buffer->head = block;
    }
    buffer->tail = block;
  }
  iree_hal_amd_gpu_pm4_record_t* record =
      (void*)((uint8_t*)(block + 1) + block->length);
  *record = (iree_hal_amd_gpu_pm4_record_t){
      .byte_length = byte_length,
      .argument_length = argument_length,
      .argument_alignment = argument_alignment,
      .word_count = word_count,
      .patch_count = patch_count,
      .kernarg_word_offset = UINT16_MAX,
      .dispatch_id_word_offset = UINT16_MAX,
      .payload_offset = argument_offset,
      .flags = (buffer->pending_barrier.execution &
                IREE_HAL_AMD_GPU_PM4_BARRIER_EXECUTION_DMA)
                   ? IREE_HAL_AMD_GPU_PM4_RECORD_JOIN_DMA
                   : 0,
  };
  block->length += byte_length;
  *out_record = record;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_pm4_command_buffer_begin(
    iree_hal_command_buffer_t* base) {
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_pm4_command_buffer_end(
    iree_hal_command_buffer_t* base) {
  iree_hal_amd_gpu_pm4_command_buffer_t* buffer = (void*)base;
  const uint32_t word_count =
      iree_hal_amd_gpu_pm4_barrier_word_count(buffer->pending_barrier);
  if (word_count) {
    iree_hal_amd_gpu_pm4_record_t* record = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_buffer_append(
        buffer, (uint16_t)word_count, 0, 0, 1, &record));
    iree_hal_amd_gpu_pm4_barrier_emit((uint32_t*)(record + 1),
                                      buffer->pending_barrier);
    buffer->pending_barrier = (iree_hal_amd_gpu_pm4_barrier_t){0};
  }
  if (buffer->resources) {
    iree_hal_resource_set_freeze(buffer->resources);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_pm4_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t* base, iree_string_view_t label,
    iree_hal_label_color_t color, const iree_hal_label_location_t* location) {
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_pm4_command_buffer_end_debug_group(
    iree_hal_command_buffer_t* base) {
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_pm4_command_buffer_barrier(
    iree_hal_command_buffer_t* base, const iree_hal_barrier_t* barrier) {
  iree_hal_amd_gpu_pm4_command_buffer_t* buffer = (void*)base;
  const iree_hal_barrier_list_t list = {.count = 1, .values = barrier};
  const iree_hal_amd_gpu_pm4_barrier_t prepared =
      iree_hal_amd_gpu_pm4_barrier_prepare(&buffer->profile, &list);
  buffer->pending_barrier.cache_control |= prepared.cache_control;
  buffer->pending_barrier.execution |= prepared.execution;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_pm4_command_buffer_atomic_wait(
    iree_hal_command_buffer_t* base, iree_hal_execution_stage_t source_stage,
    iree_hal_execution_stage_t target_stage, iree_hal_buffer_ref_t target,
    iree_hal_atomic_wait_params_t params) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "PM4 recorded atomic waits are not admitted");
}

static iree_status_t iree_hal_amd_gpu_pm4_command_buffer_atomic_store(
    iree_hal_command_buffer_t* base, iree_hal_execution_stage_t source_stage,
    iree_hal_execution_stage_t target_stage, iree_hal_buffer_ref_t target,
    iree_hal_atomic_store_params_t params) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "PM4 recorded atomic stores are not admitted");
}

static iree_status_t iree_hal_amd_gpu_pm4_command_buffer_atomic_rmw(
    iree_hal_command_buffer_t* base, iree_hal_execution_stage_t source_stage,
    iree_hal_execution_stage_t target_stage, iree_hal_buffer_ref_t target,
    iree_hal_atomic_rmw_params_t params) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "PM4 recorded atomic RMWs are not admitted");
}

static iree_status_t iree_hal_amd_gpu_pm4_command_buffer_advise_buffer(
    iree_hal_command_buffer_t* base, iree_hal_buffer_ref_t reference,
    iree_hal_memory_advise_flags_t flags, uint64_t arg0, uint64_t arg1) {
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_pm4_command_buffer_capture_binding(
    iree_hal_amd_gpu_pm4_command_buffer_t* buffer,
    iree_hal_buffer_ref_t reference, uint32_t site, uint32_t minimum_length,
    iree_hal_amd_gpu_pm4_patch_t* patch) {
  patch->buffer = reference.buffer;
  patch->offset = reference.offset;
  patch->site = site;
  if (minimum_length && reference.length != IREE_HAL_WHOLE_BUFFER &&
      reference.length < minimum_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "recorded arguments do not cover the native ABI");
  }
  if (reference.buffer) {
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_qualify_binding(
        buffer->base.queue_family, reference.buffer, site,
        &patch->source.slot));
    if (minimum_length &&
        iree_hal_buffer_byte_length(reference.buffer) - reference.offset <
            minimum_length) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "recorded argument buffer is too short");
    }
    if (buffer->resources) {
      IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert(buffer->resources, 1,
                                                        &reference.buffer));
    }
  } else {
    iree_hal_amd_gpu_pm4_binding_requirement_t* requirement =
        &buffer->requirements[reference.buffer_slot];
    patch->source.binding = reference.buffer_slot;
    requirement->sites |= 1u << site;
    if (minimum_length) {
      uint64_t required_length = 0;
      if (!iree_checked_add_u64(reference.offset, minimum_length,
                                &required_length)) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "recorded argument range overflows");
      }
      requirement->minimum_length =
          iree_max(requirement->minimum_length, required_length);
    }
    buffer->base.binding_count =
        iree_max(buffer->base.binding_count, reference.buffer_slot + 1);
  }
  return iree_ok_status();
}

static void iree_hal_amd_gpu_pm4_command_buffer_rollback(
    iree_hal_amd_gpu_pm4_command_buffer_t* buffer,
    iree_hal_amd_gpu_pm4_record_block_t* previous_tail,
    const iree_hal_amd_gpu_pm4_record_t* record) {
  // No invocation can observe a partially initialized native record. Any
  // already retained direct resources remain in the recording's resource set.
  buffer->tail->length -= record->byte_length;
  if (buffer->tail != previous_tail) {
    iree_allocator_free(buffer->host_allocator, buffer->tail);
    buffer->tail = previous_tail;
    if (previous_tail) {
      previous_tail->next = NULL;
    } else {
      buffer->head = NULL;
    }
  }
}

static iree_status_t iree_hal_amd_gpu_pm4_command_buffer_transfer(
    iree_hal_amd_gpu_pm4_command_buffer_t* buffer,
    iree_hal_transfer_operation_type_t type, iree_hal_buffer_ref_t source,
    iree_hal_buffer_ref_t target, const void* source_bytes, uint32_t pattern) {
  const uint16_t patch_count =
      type == IREE_HAL_TRANSFER_OPERATION_TYPE_COPY ? 2 : 1;
  const uint32_t update_length = type == IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE
                                     ? (uint32_t)target.length
                                     : 0;
  const uint32_t barrier_words =
      iree_hal_amd_gpu_pm4_barrier_word_count(buffer->pending_barrier);
  iree_hal_amd_gpu_pm4_record_t* record = NULL;
  iree_hal_amd_gpu_pm4_record_block_t* previous_tail = buffer->tail;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_buffer_append(
      buffer, (uint16_t)barrier_words, patch_count,
      sizeof(iree_hal_amd_gpu_pm4_record_transfer_t) + update_length, 1,
      &record));
  record->flags |= IREE_HAL_AMD_GPU_PM4_RECORD_TRANSFER;
  // Even one available staging byte permits progress. Larger windows batch
  // updates and byte-fill edges without forcing a full-span native allocation.
  record->argument_length = type != IREE_HAL_TRANSFER_OPERATION_TYPE_COPY;
  iree_hal_amd_gpu_pm4_patch_t* patches = (void*)(record + 1);
  memset(patches, 0, patch_count * sizeof(*patches));
  for (uint16_t i = 0; i < patch_count; ++i) {
    patches[i].alignment = 1;
  }
  iree_status_t status = iree_ok_status();
  if (patch_count == 2) {
    status = iree_hal_amd_gpu_pm4_command_buffer_capture_binding(
        buffer, source, IREE_HAL_AMD_GPU_PM4_SITE_QUEUE, 0, &patches[0]);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_gpu_pm4_command_buffer_capture_binding(
        buffer, target, IREE_HAL_AMD_GPU_PM4_SITE_QUEUE, 0,
        &patches[patch_count - 1]);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_amd_gpu_pm4_barrier_emit((uint32_t*)(patches + patch_count),
                                      buffer->pending_barrier);
    iree_hal_amd_gpu_pm4_record_transfer_t* transfer =
        (void*)((uint8_t*)record + record->payload_offset);
    *transfer = (iree_hal_amd_gpu_pm4_record_transfer_t){
        .length = target.length,
        .pattern = pattern,
        .type = type,
    };
    if (update_length) {
      memcpy(transfer + 1, source_bytes, update_length);
    }
    buffer->pending_barrier = (iree_hal_amd_gpu_pm4_barrier_t){0};
  } else {
    iree_hal_amd_gpu_pm4_command_buffer_rollback(buffer, previous_tail, record);
  }
  return status;
}

static iree_status_t iree_hal_amd_gpu_pm4_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base, iree_hal_buffer_ref_t target,
    const void* pattern, iree_host_size_t pattern_length,
    iree_hal_fill_flags_t flags) {
  uint32_t value = 0;
  memcpy(&value, pattern, pattern_length);
  if (pattern_length == 1) {
    value *= 0x01010101u;
  } else if (pattern_length == 2) {
    value |= value << 16;
  }
  return iree_hal_amd_gpu_pm4_command_buffer_transfer(
      (void*)base, IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
      (iree_hal_buffer_ref_t){0}, target, NULL, value);
}

static iree_status_t iree_hal_amd_gpu_pm4_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base, const void* source,
    iree_host_size_t source_offset, iree_hal_buffer_ref_t target,
    iree_hal_update_flags_t flags) {
  return iree_hal_amd_gpu_pm4_command_buffer_transfer(
      (void*)base, IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE,
      (iree_hal_buffer_ref_t){0}, target,
      (const uint8_t*)source + source_offset, 0);
}

static iree_status_t iree_hal_amd_gpu_pm4_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base, iree_hal_buffer_ref_t source,
    iree_hal_buffer_ref_t target, iree_hal_copy_flags_t flags) {
  return iree_hal_amd_gpu_pm4_command_buffer_transfer(
      (void*)base, IREE_HAL_TRANSFER_OPERATION_TYPE_COPY, source, target, NULL,
      0);
}

static iree_status_t iree_hal_amd_gpu_pm4_command_buffer_dispatch(
    iree_hal_command_buffer_t* base, iree_hal_executable_t* executable,
    iree_hal_executable_function_t token, iree_hal_dispatch_config_t config,
    iree_const_byte_span_t constants, iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags) {
  iree_hal_amd_gpu_pm4_command_buffer_t* buffer = (void*)base;
  const iree_hal_amd_gpu_pm4_function_t* function = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_gpu_executable_lookup_pm4(executable, token, &function));
  iree_hal_amd_gpu_pm4_dispatch_t dispatch;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_dispatch_prepare(
      function, buffer->maximum_group_segment_byte_length, config, constants,
      bindings.count, flags, &dispatch));
  const bool indirect_parameters =
      iree_hal_dispatch_uses_indirect_parameters(flags);
  if (!indirect_parameters &&
      (!dispatch.workgroup_count[0] || !dispatch.workgroup_count[1] ||
       !dispatch.workgroup_count[2])) {
    return iree_ok_status();
  }
  if (buffer->resources) {
    IREE_RETURN_IF_ERROR(
        iree_hal_resource_set_insert(buffer->resources, 1, &executable));
  }
  const bool indirect = iree_hal_dispatch_uses_indirect_arguments(flags);
  const uint16_t binding_patch_count =
      indirect && function->launch.user_data.kernarg_pointer == UINT8_MAX
          ? 0
          : (uint16_t)bindings.count;
  const uint32_t patch_count = binding_patch_count + indirect_parameters;
  if (patch_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "recorded dispatch has too many native references");
  }
  const uint32_t barrier_words =
      iree_hal_amd_gpu_pm4_barrier_word_count(buffer->pending_barrier);
  const uint32_t launch_words = function->launch.user_data_count
                                    ? 24 + function->launch.user_data_count
                                    : 22;
  const uint16_t indirect_copy_count =
      indirect_parameters && !indirect ? function->kernarg->indirect_copy_count
                                       : 0;
  const uint32_t dispatch_words =
      indirect_parameters ? iree_hal_amd_gpu_pm4_dispatch_indirect_word_count(
                                indirect_copy_count)
                          : 5;
  const uint32_t word_count = barrier_words + launch_words + dispatch_words;
  iree_hal_amd_gpu_pm4_record_t* record = NULL;
  iree_hal_amd_gpu_pm4_record_block_t* previous_tail = buffer->tail;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_buffer_append(
      buffer, (uint16_t)word_count, (uint16_t)patch_count,
      indirect ? 0 : function->kernarg->byte_length,
      indirect ? 1 : function->kernarg->alignment, &record));
  record->flags |= IREE_HAL_AMD_GPU_PM4_RECORD_DISPATCH;
  record->indirect.word_offset = (uint16_t)(barrier_words + launch_words);
  record->indirect.copy_count = indirect_copy_count;
  if (indirect_copy_count) {
    record->flags |= IREE_HAL_AMD_GPU_PM4_RECORD_JOIN_DMA;
  }
  iree_hal_amd_gpu_pm4_patch_t* patches = (void*)(record + 1);
  uint16_t kernarg_word = iree_hal_amd_gpu_pm4_launch_user_data_word_offset(
      function->launch.user_data.kernarg_pointer);
  if (kernarg_word != UINT16_MAX) {
    kernarg_word += barrier_words;
    if (!indirect) {
      record->kernarg_word_offset = kernarg_word;
    }
  }
  record->dispatch_id_word_offset =
      iree_hal_amd_gpu_pm4_launch_user_data_word_offset(
          function->launch.user_data.dispatch_id);
  if (record->dispatch_id_word_offset != UINT16_MAX) {
    record->dispatch_id_word_offset += barrier_words;
  }
  const uint16_t* binding_offsets =
      iree_hal_amd_gpu_kernarg_layout_binding_offsets(function->kernarg);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < binding_patch_count && iree_status_is_ok(status); ++i) {
    patches[i] = (iree_hal_amd_gpu_pm4_patch_t){
        .target_offset =
            indirect ? (uint32_t)kernarg_word * 4
                     : IREE_HAL_AMD_GPU_PM4_PATCH_ARGUMENT | binding_offsets[i],
        .alignment = indirect ? function->kernarg->alignment : 1,
    };
    status = iree_hal_amd_gpu_pm4_command_buffer_capture_binding(
        buffer, bindings.values[i], IREE_HAL_AMD_GPU_PM4_SITE_PROGRAM,
        indirect ? function->kernarg->byte_length : 0, &patches[i]);
  }
  if (indirect_parameters && iree_status_is_ok(status)) {
    iree_hal_amd_gpu_pm4_patch_t* patch = &patches[binding_patch_count];
    *patch = (iree_hal_amd_gpu_pm4_patch_t){
        .target_offset = (word_count - 3) * 4,
        .alignment = 4,
    };
    status = iree_hal_amd_gpu_pm4_command_buffer_capture_binding(
        buffer, config.workgroup_count_ref, IREE_HAL_AMD_GPU_PM4_SITE_QUEUE, 12,
        patch);
  }
  if (iree_status_is_ok(status)) {
    uint32_t* words = (void*)(patches + patch_count);
    words = iree_hal_amd_gpu_pm4_barrier_emit(words, buffer->pending_barrier);
    const iree_hal_amd_gpu_pm4_launch_inputs_t inputs = {0};
    words = iree_hal_amd_gpu_pm4_launch_emit(
        words, &function->launch, dispatch.workgroup_size,
        dispatch.partial_size, dispatch.group_segment_byte_length, &inputs);
    const uint32_t initiator =
        function->launch.dispatch_initiator |
        (dispatch.partial_size[0] ? UINT32_C(1) << 6 : 0);
    if (indirect_parameters) {
      iree_hal_amd_gpu_pm4_dispatch_emit_indirect(
          words, function->kernarg, indirect_copy_count, 0, 0,
          buffer->profile.command_processor_bypasses_l2
              ? buffer->profile.acquire
              : buffer->profile.shader,
          initiator);
    } else {
      iree_hal_amd_gpu_pm4_emit_dispatch_direct(words, dispatch.workgroup_count,
                                                initiator);
    }
    if (!indirect) {
      iree_hal_amd_gpu_pm4_dispatch_emplace_arguments(
          function, &dispatch, constants, NULL, flags,
          (uint8_t*)record + record->payload_offset);
    }
    buffer->pending_barrier = (iree_hal_amd_gpu_pm4_barrier_t){0};
  } else {
    iree_hal_amd_gpu_pm4_command_buffer_rollback(buffer, previous_tail, record);
  }
  return status;
}

iree_status_t iree_hal_amd_gpu_pm4_command_invocation_initialize(
    iree_hal_command_buffer_t* base, iree_hal_buffer_binding_table_t bindings,
    void* storage, iree_hal_queue_execute_flags_t flags,
    iree_hal_amd_gpu_pm4_command_invocation_t* out_invocation) {
  iree_hal_amd_gpu_pm4_command_buffer_t* buffer = (void*)base;
  *out_invocation = (iree_hal_amd_gpu_pm4_command_invocation_t){
      .command_buffer = base,
      .bindings = storage,
      .cursor = {.block = buffer->head},
      .first = true,
      .borrow_bindings = iree_any_bit_set(
          flags, IREE_HAL_QUEUE_EXECUTE_FLAG_BORROW_BINDING_TABLE_LIFETIME),
  };
  iree_hal_command_buffer_retain(base);
  if (base->binding_count) {
    memset(
        storage, 0,
        base->binding_count * sizeof(iree_hal_amd_gpu_pm4_command_binding_t));
  }
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < base->binding_count && iree_status_is_ok(status);
       ++i) {
    const iree_hal_amd_gpu_pm4_binding_requirement_t requirement =
        buffer->requirements[i];
    if (!requirement.sites) {
      continue;
    }
    const iree_hal_buffer_binding_t binding = bindings.bindings[i];
    const uint64_t length =
        binding.length == IREE_HAL_WHOLE_BUFFER
            ? iree_hal_buffer_byte_length(binding.buffer) - binding.offset
            : binding.length;
    if (length < requirement.minimum_length) {
      status =
          iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                           "binding %u does not cover native arguments", i);
    } else {
      iree_hal_amd_gpu_pm4_command_binding_t* captured =
          &out_invocation->bindings[i];
      for (uint32_t site = 0; site < 2 && iree_status_is_ok(status); ++site) {
        if (requirement.sites & (1u << site)) {
          status = iree_hal_amd_gpu_pm4_command_qualify_binding(
              base->queue_family, binding.buffer, site, &captured->slots[site]);
        }
      }
      if (iree_status_is_ok(status)) {
        captured->buffer = binding.buffer;
        captured->offset = binding.offset;
        captured->length = length;
        if (!out_invocation->borrow_bindings) {
          iree_hal_buffer_retain(binding.buffer);
        }
      }
    }
  }
  return status;
}

void iree_hal_amd_gpu_pm4_command_invocation_deinitialize(
    iree_hal_amd_gpu_pm4_command_invocation_t* invocation) {
  if (!invocation->borrow_bindings) {
    for (uint32_t i = 0; i < invocation->command_buffer->binding_count; ++i) {
      iree_hal_buffer_release(invocation->bindings[i].buffer);
    }
  }
  iree_hal_command_buffer_release(invocation->command_buffer);
}

static const iree_hal_amd_gpu_pm4_record_t*
iree_hal_amd_gpu_pm4_command_cursor_record(
    iree_hal_amd_gpu_pm4_command_cursor_t cursor) {
  return (const void*)((const uint8_t*)(cursor.block + 1) + cursor.offset);
}

iree_hal_amd_gpu_pm4_command_arguments_t
iree_hal_amd_gpu_pm4_command_next_arguments(
    iree_hal_amd_gpu_pm4_command_cursor_t cursor) {
  if (!cursor.block) {
    return (iree_hal_amd_gpu_pm4_command_arguments_t){.alignment = 1};
  }
  const iree_hal_amd_gpu_pm4_record_t* record =
      iree_hal_amd_gpu_pm4_command_cursor_record(cursor);
  return (iree_hal_amd_gpu_pm4_command_arguments_t){
      .length = record->argument_length,
      .alignment = record->argument_alignment};
}

static void iree_hal_amd_gpu_pm4_command_advance(
    const iree_hal_amd_gpu_pm4_record_t* record,
    iree_hal_amd_gpu_pm4_command_cursor_t* cursor) {
  cursor->transfer_offset = 0;
  cursor->offset += record->byte_length;
  if (cursor->offset == cursor->block->length) {
    cursor->block = cursor->block->next;
    cursor->offset = 0;
  }
}

static uint64_t iree_hal_amd_gpu_pm4_command_remaining_length(
    const iree_hal_amd_gpu_pm4_command_invocation_t* invocation,
    const iree_hal_amd_gpu_pm4_patch_t* patch) {
  const uint64_t length =
      patch->buffer ? iree_hal_buffer_byte_length(patch->buffer)
                    : invocation->bindings[patch->source.binding].length;
  return length - patch->offset;
}

static iree_status_t iree_hal_amd_gpu_pm4_command_resolve_address(
    const iree_hal_amd_gpu_pm4_command_invocation_t* invocation,
    const iree_hal_amd_gpu_pm4_patch_t* patch, uint64_t* out_address) {
  iree_hal_buffer_t* buffer = patch->buffer;
  iree_hal_buffer_native_binding_slot_t slot = patch->source.slot;
  uint64_t offset = patch->offset;
  if (!buffer) {
    const iree_hal_amd_gpu_pm4_command_binding_t* binding =
        &invocation->bindings[patch->source.binding];
    buffer = binding->buffer;
    slot = binding->slots[patch->site];
    offset += binding->offset;
  }
  uint64_t address =
      iree_hal_buffer_native_binding(buffer, slot).device_address;
  if (!address) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "recorded buffer has no committed GPU address");
  }
  if (!iree_checked_add_u64(address, offset, &address) ||
      (address & (patch->alignment - 1))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "recorded GPU address overflows or is misaligned");
  }
  *out_address = address;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_pm4_command_prepare_transfer(
    const iree_hal_amd_gpu_pm4_command_invocation_t* invocation,
    const iree_hal_amd_gpu_pm4_record_t* record, uint32_t word_capacity,
    iree_byte_span_t arguments, uint64_t argument_address,
    iree_hal_amd_gpu_pm4_command_chunk_t* chunk) {
  const uint32_t prefix_words =
      chunk->next.transfer_offset ? 0 : record->word_count;
  if (prefix_words + 7 > word_capacity - chunk->word_count) {
    return iree_ok_status();
  }
  const iree_hal_amd_gpu_pm4_patch_t* patches = (const void*)(record + 1);
  const iree_hal_amd_gpu_pm4_patch_t* target =
      &patches[record->patch_count - 1];
  const iree_hal_amd_gpu_pm4_record_transfer_t* transfer =
      (const void*)((const uint8_t*)record + record->payload_offset);
  iree_hal_amd_gpu_pm4_transfer_span_t span = {
      .source_bytes = (const uint8_t*)(transfer + 1),
      .length = transfer->length,
      .pattern = transfer->pattern,
      .type = transfer->type,
  };
  if (span.length == IREE_HAL_WHOLE_BUFFER) {
    span.length =
        iree_hal_amd_gpu_pm4_command_remaining_length(invocation, target);
    if (transfer->type == IREE_HAL_TRANSFER_OPERATION_TYPE_COPY &&
        span.length != iree_hal_amd_gpu_pm4_command_remaining_length(
                           invocation, &patches[0])) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "recorded copy remaining ranges differ");
    }
  }
  uint32_t* words = chunk->words + chunk->word_count;
  if (!span.length) {
    // A late-bound empty range consumes one bounded record, without touching
    // backing memory or scanning ahead to find a nonempty command.
    memcpy(words, patches + record->patch_count, prefix_words * 4);
    iree_hal_amd_gpu_pm4_emit_nop(words + prefix_words, 2);
    chunk->word_count += prefix_words + 2;
    if (record->flags & IREE_HAL_AMD_GPU_PM4_RECORD_JOIN_DMA) {
      chunk->pending_dma = false;
    }
    iree_hal_amd_gpu_pm4_command_advance(record, &chunk->next);
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_resolve_address(
      invocation, target, &span.target_address));
  if (transfer->type == IREE_HAL_TRANSFER_OPERATION_TYPE_COPY) {
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_resolve_address(
        invocation, &patches[0], &span.source_address));
  }
  // Resolve each range once per bounded publication, then feed only native
  // addresses and the immutable span to the DMA writer.
  uint32_t pending_prefix = prefix_words;
  do {
    const iree_hal_amd_gpu_pm4_transfer_packet_t packet =
        iree_hal_amd_gpu_pm4_transfer_emit_packet(
            &span, chunk->next.transfer_offset,
            iree_make_byte_span(
                arguments.data ? arguments.data + chunk->argument_length : NULL,
                arguments.data_length - chunk->argument_length),
            argument_address + chunk->argument_length,
            chunk->words + chunk->word_count + pending_prefix);
    if (!packet.length) {
      break;
    }
    if (pending_prefix) {
      memcpy(chunk->words + chunk->word_count, patches + record->patch_count,
             pending_prefix * 4);
    }
    chunk->word_count += pending_prefix + 7;
    pending_prefix = 0;
    chunk->argument_length += packet.staging_length;
    chunk->pending_dma = true;
    chunk->next.transfer_offset += packet.length;
    if (chunk->next.transfer_offset == span.length) {
      iree_hal_amd_gpu_pm4_command_advance(record, &chunk->next);
      break;
    }
  } while (!invocation->first && chunk->word_count + 7 <= word_capacity);
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_pm4_command_prepare_chunk(
    const iree_hal_amd_gpu_pm4_command_invocation_t* invocation,
    uint32_t word_capacity, iree_byte_span_t arguments,
    uint64_t argument_address, uint64_t dispatch_id,
    iree_hal_amd_gpu_pm4_command_chunk_t* out_chunk) {
  // Do not clear the 4 KiB command array: every returned word is overwritten.
  out_chunk->word_count = out_chunk->argument_length = 0;
  out_chunk->dispatch_count = 0;
  out_chunk->pending_dma = false;
  out_chunk->next = invocation->cursor;
  iree_status_t status = iree_ok_status();
  do {
    if (!out_chunk->next.block) {
      break;
    }
    const iree_hal_amd_gpu_pm4_record_t* record =
        iree_hal_amd_gpu_pm4_command_cursor_record(out_chunk->next);
    if (record->flags & IREE_HAL_AMD_GPU_PM4_RECORD_TRANSFER) {
      const uint32_t previous_word_count = out_chunk->word_count;
      status = iree_hal_amd_gpu_pm4_command_prepare_transfer(
          invocation, record, word_capacity, arguments, argument_address,
          out_chunk);
      if (out_chunk->word_count == previous_word_count) {
        break;
      }
      continue;
    }
    const uint64_t argument_offset =
        iree_align_uint64(argument_address + out_chunk->argument_length,
                          record->argument_alignment) -
        argument_address;
    if (record->word_count > word_capacity - out_chunk->word_count ||
        (record->argument_length &&
         argument_offset + record->argument_length > arguments.data_length)) {
      break;
    }
    const iree_hal_amd_gpu_pm4_patch_t* patches = (const void*)(record + 1);
    uint32_t* words = out_chunk->words + out_chunk->word_count;
    memcpy(words, patches + record->patch_count, record->word_count * 4);
    uint8_t* target_arguments =
        record->argument_length ? arguments.data + argument_offset : NULL;
    if (record->argument_length) {
      memcpy(target_arguments, (const uint8_t*)record + record->payload_offset,
             record->argument_length);
      out_chunk->argument_length =
          (uint32_t)argument_offset + record->argument_length;
    }
    for (uint16_t i = 0; i < record->patch_count && iree_status_is_ok(status);
         ++i) {
      const iree_hal_amd_gpu_pm4_patch_t* patch = &patches[i];
      uint64_t address = 0;
      status = iree_hal_amd_gpu_pm4_command_resolve_address(invocation, patch,
                                                            &address);
      if (iree_status_is_ok(status)) {
        uint8_t* target =
            patch->target_offset & IREE_HAL_AMD_GPU_PM4_PATCH_ARGUMENT
                ? target_arguments
                : (uint8_t*)words;
        iree_unaligned_store_le_u64(
            target +
                (patch->target_offset & ~IREE_HAL_AMD_GPU_PM4_PATCH_ARGUMENT),
            address);
      }
    }
    if (iree_status_is_ok(status)) {
      if (record->indirect.copy_count) {
        const uint64_t parameters_address =
            iree_unaligned_load_le_u64(words + record->word_count - 3);
        uint32_t* copy = words + record->indirect.word_offset;
        for (uint16_t i = 0; i < record->indirect.copy_count; ++i, copy += 7) {
          iree_unaligned_store_le_u64(
              copy + 2,
              parameters_address + iree_unaligned_load_le_u64(copy + 2));
          iree_unaligned_store_le_u64(copy + 4,
                                      argument_address + argument_offset +
                                          iree_unaligned_load_le_u64(copy + 4));
        }
      }
      if (record->kernarg_word_offset != UINT16_MAX) {
        iree_unaligned_store_le_u64(words + record->kernarg_word_offset,
                                    argument_address + argument_offset);
      }
      if (record->dispatch_id_word_offset != UINT16_MAX) {
        iree_unaligned_store_le_u64(words + record->dispatch_id_word_offset,
                                    dispatch_id + out_chunk->dispatch_count);
      }
      out_chunk->dispatch_count +=
          !!(record->flags & IREE_HAL_AMD_GPU_PM4_RECORD_DISPATCH);
      out_chunk->word_count += record->word_count;
      if (record->flags & IREE_HAL_AMD_GPU_PM4_RECORD_JOIN_DMA) {
        out_chunk->pending_dma = false;
      }
      iree_hal_amd_gpu_pm4_command_advance(record, &out_chunk->next);
    }
  } while (!invocation->first && iree_status_is_ok(status));
  return status;
}

static const iree_hal_command_buffer_vtable_t
    iree_hal_amd_gpu_pm4_command_buffer_vtable = {
        .destroy = iree_hal_amd_gpu_pm4_command_buffer_destroy,
        .begin = iree_hal_amd_gpu_pm4_command_buffer_begin,
        .end = iree_hal_amd_gpu_pm4_command_buffer_end,
        .begin_debug_group =
            iree_hal_amd_gpu_pm4_command_buffer_begin_debug_group,
        .end_debug_group = iree_hal_amd_gpu_pm4_command_buffer_end_debug_group,
        .barrier = iree_hal_amd_gpu_pm4_command_buffer_barrier,
        .atomic_wait = iree_hal_amd_gpu_pm4_command_buffer_atomic_wait,
        .atomic_store = iree_hal_amd_gpu_pm4_command_buffer_atomic_store,
        .atomic_rmw = iree_hal_amd_gpu_pm4_command_buffer_atomic_rmw,
        .advise_buffer = iree_hal_amd_gpu_pm4_command_buffer_advise_buffer,
        .fill_buffer = iree_hal_amd_gpu_pm4_command_buffer_fill_buffer,
        .update_buffer = iree_hal_amd_gpu_pm4_command_buffer_update_buffer,
        .copy_buffer = iree_hal_amd_gpu_pm4_command_buffer_copy_buffer,
        .dispatch = iree_hal_amd_gpu_pm4_command_buffer_dispatch,
};
