// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/command_buffer.h"

#include "iree/base/internal/arena.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/commands.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/dispatch.h"
#include "iree/hal/utils/resource_set.h"

enum {
  IREE_HAL_AMD_GPU_PM4_RECORD_BLOCK_SIZE = 16384,
  IREE_HAL_AMD_GPU_PM4_RECORD_DISPATCH = 1u << 0,
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
  // Byte offset from this header to the argument template.
  uint32_t argument_offset;
  // Native publication properties; no HAL operation is replayed.
  uint32_t flags;
  // Zero padding preserving eight-byte alignment of the following patches.
  uint32_t reserved;
} iree_hal_amd_gpu_pm4_record_t;
static_assert(sizeof(iree_hal_amd_gpu_pm4_record_t) == 32,
              "record headers occupy half a cache line");

typedef struct iree_hal_amd_gpu_pm4_patch_t {
  // Direct recorded buffer, or NULL for an invocation binding-table slot.
  iree_hal_buffer_t* buffer;
  // Byte offset relative to the recorded buffer or invocation binding.
  uint64_t offset;
  // Invocation binding index, unused for a direct buffer.
  uint32_t binding;
  // Byte destination; PATCH_ARGUMENT selects argument instead of command bytes.
  uint32_t target_offset;
  // Required power-of-two address alignment, one for ordinary shader bindings.
  uint32_t alignment;
  // Qualified device-address slot for a direct buffer.
  iree_hal_buffer_native_binding_slot_t slot;
} iree_hal_amd_gpu_pm4_patch_t;
static_assert(sizeof(iree_hal_amd_gpu_pm4_patch_t) == 32,
              "two address relocations occupy one cache line");

typedef struct iree_hal_amd_gpu_pm4_binding_requirement_t {
  // Smallest table extent covering any indirect native kernarg use.
  uint64_t minimum_length;
  // The slot is used; holes in a binding table are never dereferenced.
  bool used;
} iree_hal_amd_gpu_pm4_binding_requirement_t;

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
// needs an actual prepared program-site address slot, including for legacy
// buffers that can satisfy generic HAL validation without a native contract.
static iree_status_t iree_hal_amd_gpu_pm4_command_qualify_binding(
    const iree_hal_queue_family_t* family, iree_hal_buffer_t* buffer,
    iree_hal_buffer_native_binding_slot_t* out_slot) {
  const iree_hal_memory_contract_t* contract = buffer->memory.contract;
  const uint64_t scope_id = (uint64_t)family->memory.queue_scope_id + 1;
  if (!contract || !family->memory.domain ||
      contract->domain != family->memory.domain ||
      scope_id >= contract->scope_count) {
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "recorded buffer has no program-site contract");
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
      .argument_offset = argument_offset,
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

static iree_status_t iree_hal_amd_gpu_pm4_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base, iree_hal_buffer_ref_t target,
    const void* pattern, iree_host_size_t pattern_length,
    iree_hal_fill_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "PM4 recorded fills are not admitted");
}

static iree_status_t iree_hal_amd_gpu_pm4_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base, const void* source,
    iree_host_size_t source_offset, iree_hal_buffer_ref_t target,
    iree_hal_update_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "PM4 recorded updates are not admitted");
}

static iree_status_t iree_hal_amd_gpu_pm4_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base, iree_hal_buffer_ref_t source,
    iree_hal_buffer_ref_t target, iree_hal_copy_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "PM4 recorded copies are not admitted");
}

static iree_status_t iree_hal_amd_gpu_pm4_command_buffer_capture_binding(
    iree_hal_amd_gpu_pm4_command_buffer_t* buffer,
    iree_hal_buffer_ref_t reference, uint32_t minimum_length,
    iree_hal_amd_gpu_pm4_patch_t* patch) {
  patch->buffer = reference.buffer;
  patch->offset = reference.offset;
  patch->binding = reference.buffer_slot;
  if (minimum_length && reference.length != IREE_HAL_WHOLE_BUFFER &&
      reference.length < minimum_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "recorded arguments do not cover the native ABI");
  }
  if (reference.buffer) {
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_qualify_binding(
        buffer->base.queue_family, reference.buffer, &patch->slot));
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
    requirement->used = true;
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
  if (!dispatch.workgroup_count[0] || !dispatch.workgroup_count[1] ||
      !dispatch.workgroup_count[2]) {
    return iree_ok_status();
  }
  if (buffer->resources) {
    IREE_RETURN_IF_ERROR(
        iree_hal_resource_set_insert(buffer->resources, 1, &executable));
  }
  const bool indirect = iree_hal_dispatch_uses_indirect_arguments(flags);
  const uint16_t patch_count =
      indirect && function->launch.user_data.kernarg_pointer == UINT8_MAX
          ? 0
          : (uint16_t)bindings.count;
  const uint32_t barrier_words =
      iree_hal_amd_gpu_pm4_barrier_word_count(buffer->pending_barrier);
  const uint32_t launch_words = function->launch.user_data_count
                                    ? 24 + function->launch.user_data_count
                                    : 22;
  iree_hal_amd_gpu_pm4_record_t* record = NULL;
  iree_hal_amd_gpu_pm4_record_block_t* previous_tail = buffer->tail;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_buffer_append(
      buffer, (uint16_t)(barrier_words + launch_words + 5), patch_count,
      indirect ? 0 : function->kernarg->byte_length,
      indirect ? 1 : function->kernarg->alignment, &record));
  record->flags = IREE_HAL_AMD_GPU_PM4_RECORD_DISPATCH;
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
  for (iree_host_size_t i = 0; i < patch_count && iree_status_is_ok(status);
       ++i) {
    patches[i] = (iree_hal_amd_gpu_pm4_patch_t){
        .target_offset =
            indirect ? (uint32_t)kernarg_word * 4
                     : IREE_HAL_AMD_GPU_PM4_PATCH_ARGUMENT | binding_offsets[i],
        .alignment = indirect ? function->kernarg->alignment : 1,
    };
    status = iree_hal_amd_gpu_pm4_command_buffer_capture_binding(
        buffer, bindings.values[i],
        indirect ? function->kernarg->byte_length : 0, &patches[i]);
  }
  if (iree_status_is_ok(status)) {
    uint32_t* words = (void*)(patches + patch_count);
    words = iree_hal_amd_gpu_pm4_barrier_emit(words, buffer->pending_barrier);
    const iree_hal_amd_gpu_pm4_launch_inputs_t inputs = {0};
    words = iree_hal_amd_gpu_pm4_launch_emit(
        words, &function->launch, dispatch.workgroup_size,
        dispatch.partial_size, dispatch.group_segment_byte_length, &inputs);
    iree_hal_amd_gpu_pm4_emit_dispatch_direct(
        words, dispatch.workgroup_count,
        function->launch.dispatch_initiator |
            (dispatch.partial_size[0] ? UINT32_C(1) << 6 : 0));
    if (!indirect) {
      iree_hal_amd_gpu_pm4_dispatch_emplace_arguments(
          function, &dispatch, constants, NULL, flags,
          (uint8_t*)record + record->argument_offset);
    }
    buffer->pending_barrier = (iree_hal_amd_gpu_pm4_barrier_t){0};
  } else {
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
    if (!requirement.used) {
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
      status = iree_hal_amd_gpu_pm4_command_qualify_binding(
          base->queue_family, binding.buffer, &captured->slot);
      if (iree_status_is_ok(status)) {
        captured->buffer = binding.buffer;
        captured->offset = binding.offset;
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

iree_status_t iree_hal_amd_gpu_pm4_command_prepare_chunk(
    const iree_hal_amd_gpu_pm4_command_invocation_t* invocation,
    uint32_t word_capacity, iree_byte_span_t arguments,
    uint64_t argument_address, uint64_t dispatch_id,
    iree_hal_amd_gpu_pm4_command_chunk_t* out_chunk) {
  // Do not clear the 4 KiB command array: every returned word is overwritten.
  out_chunk->word_count = out_chunk->argument_length = 0;
  out_chunk->dispatch_count = 0;
  out_chunk->next = invocation->cursor;
  iree_status_t status = iree_ok_status();
  do {
    if (!out_chunk->next.block) {
      break;
    }
    const iree_hal_amd_gpu_pm4_record_t* record =
        iree_hal_amd_gpu_pm4_command_cursor_record(out_chunk->next);
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
      memcpy(target_arguments, (const uint8_t*)record + record->argument_offset,
             record->argument_length);
      out_chunk->argument_length =
          (uint32_t)argument_offset + record->argument_length;
    }
    for (uint16_t i = 0; i < record->patch_count && iree_status_is_ok(status);
         ++i) {
      const iree_hal_amd_gpu_pm4_patch_t* patch = &patches[i];
      iree_hal_buffer_t* buffer = patch->buffer;
      iree_hal_buffer_native_binding_slot_t slot = patch->slot;
      uint64_t offset = patch->offset;
      if (!buffer) {
        const iree_hal_amd_gpu_pm4_command_binding_t* binding =
            &invocation->bindings[patch->binding];
        buffer = binding->buffer;
        slot = binding->slot;
        offset += binding->offset;
      }
      uint64_t address =
          iree_hal_buffer_native_binding(buffer, slot).device_address;
      if (!address) {
        status =
            iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                             "recorded buffer has no committed GPU address");
      } else if (!iree_checked_add_u64(address, offset, &address) ||
                 (address & (patch->alignment - 1))) {
        status =
            iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                             "recorded GPU address overflows or is misaligned");
      } else {
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
      out_chunk->next.offset += record->byte_length;
      if (out_chunk->next.offset == out_chunk->next.block->length) {
        out_chunk->next.block = out_chunk->next.block->next;
        out_chunk->next.offset = 0;
      }
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
