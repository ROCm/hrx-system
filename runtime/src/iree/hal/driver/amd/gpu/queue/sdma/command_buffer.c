// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/sdma/command_buffer.h"

#include "iree/base/internal/arena.h"
#include "iree/hal/driver/amd/gpu/queue/sdma/recording.h"
#include "iree/hal/utils/resource_set.h"

enum {
  IREE_HAL_AMD_GPU_SDMA_VALUE_PRIVATE = 0,
  IREE_HAL_AMD_GPU_SDMA_VALUE_FIRST_INPUT = 1,
};

typedef struct iree_hal_amd_gpu_sdma_command_buffer_t {
  // Public recording identity and validation state.
  iree_hal_command_buffer_t base;
  // Allocator owning this object and its sealed host program.
  iree_allocator_t host_allocator;
  // Cold append storage, released after successful sealing.
  iree_hal_amd_gpu_sdma_recording_t recording;
  // Immutable native program shared by every invocation.
  iree_hal_amd_gpu_sdma_program_t program;
  // Device-owned placement policy borrowed by sealing and invocation growth.
  iree_hal_amd_gpu_program_pool_t* pool;
  // Captured update/pattern bytes and private timestamp storage. Empty when
  // the recording needs no native data. Leases outlive every partial span.
  iree_hal_amd_gpu_program_t storage;
  // Deduplicated input values and aggregate indirect binding requirements.
  iree_hal_amd_gpu_command_inputs_t inputs;
  // Selected native visibility carrier, captured during family construction.
  iree_hal_amd_gpu_sdma_barrier_profile_t barriers;
  // Adjacent public boundaries combined before the next native operation.
  iree_hal_amd_gpu_sdma_barrier_t pending_barrier;
  // First failed mutation; prevents execution of a partial program.
  iree_status_t recording_status;
  // Private 32-byte timestamp cell, or UINT32_MAX before its first use.
  uint32_t timestamp_offset;
  // Complete invocation host storage, established at sealing.
  iree_host_size_t invocation_size;
  // Dense value-vector displacement in invocation host storage.
  iree_host_size_t values_offset;
  // Number of recorded payload operations for profiling attribution.
  iree_host_size_t operation_count;
  // Blocks backing retained direct resource references.
  iree_arena_block_pool_t resource_blocks;
  // Direct resources, or NULL for explicitly unretained recordings.
  iree_hal_resource_set_t* resources;
} iree_hal_amd_gpu_sdma_command_buffer_t;

static const iree_hal_command_buffer_vtable_t
    iree_hal_amd_gpu_sdma_command_buffer_vtable;

static void iree_hal_amd_gpu_sdma_command_buffer_destroy(
    iree_hal_command_buffer_t* base) {
  iree_hal_amd_gpu_sdma_command_buffer_t* buffer = (void*)base;
  const iree_allocator_t allocator = buffer->host_allocator;
  iree_status_free(buffer->recording_status);
  iree_hal_amd_gpu_sdma_recording_deinitialize(&buffer->recording);
  iree_hal_amd_gpu_program_deinitialize(&buffer->storage);
  iree_allocator_free(allocator, buffer->program.storage);
  iree_hal_resource_set_free(buffer->resources);
  iree_arena_block_pool_deinitialize(&buffer->resource_blocks);
  iree_hal_amd_gpu_command_inputs_deinitialize(&buffer->inputs);
  iree_allocator_free(allocator, buffer);
}

iree_status_t iree_hal_amd_gpu_sdma_command_buffer_create(
    const iree_hal_queue_family_t* family,
    iree_hal_amd_gpu_program_pool_t* pool,
    amdf_queue_format_features_t features,
    iree_hal_amd_gpu_sdma_barrier_profile_t barriers,
    iree_hal_command_buffer_mode_t mode, iree_hal_command_category_t categories,
    iree_host_size_t binding_capacity, iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_buffer) {
  *out_buffer = NULL;
  if (binding_capacity >= UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "SDMA binding table exceeds its index space");
  }
  iree_host_size_t size = 0, requirements_offset = 0, validation_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_amd_gpu_sdma_command_buffer_t), &size,
      IREE_STRUCT_FIELD(binding_capacity + 1,
                        iree_hal_amd_gpu_binding_requirement_t,
                        &requirements_offset),
      IREE_STRUCT_FIELD(
          iree_hal_command_buffer_validation_state_size(mode, binding_capacity),
          uint8_t, &validation_offset)));
  iree_hal_amd_gpu_sdma_command_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, size, (void**)&buffer));
  iree_hal_command_buffer_initialize(
      family, mode, categories, binding_capacity,
      (uint8_t*)buffer + validation_offset,
      &iree_hal_amd_gpu_sdma_command_buffer_vtable, &buffer->base);
  buffer->host_allocator = host_allocator;
  buffer->pool = pool;
  buffer->barriers = barriers;
  buffer->timestamp_offset = UINT32_MAX;
  iree_hal_amd_gpu_sdma_recording_initialize(features, host_allocator,
                                             &buffer->recording);
  iree_arena_block_pool_initialize(4096, host_allocator,
                                   &buffer->resource_blocks);
  iree_hal_amd_gpu_command_inputs_initialize(
      &buffer->base, IREE_HAL_AMD_GPU_SDMA_VALUE_FIRST_INPUT, NULL,
      (void*)((uint8_t*)buffer + requirements_offset), host_allocator,
      &buffer->inputs);
  iree_status_t status = iree_ok_status();
  if (!iree_any_bit_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED)) {
    status = iree_hal_resource_set_allocate(&buffer->resource_blocks,
                                            &buffer->resources);
  }
  if (iree_status_is_ok(status)) {
    buffer->inputs.resources = buffer->resources;
    *out_buffer = &buffer->base;
  } else {
    iree_hal_command_buffer_release(&buffer->base);
  }
  return status;
}

const iree_hal_amd_gpu_sdma_program_t*
iree_hal_amd_gpu_sdma_command_buffer_program(iree_hal_command_buffer_t* base) {
  return &((iree_hal_amd_gpu_sdma_command_buffer_t*)base)->program;
}

iree_hal_amd_gpu_program_t* iree_hal_amd_gpu_sdma_command_buffer_storage(
    iree_hal_command_buffer_t* base) {
  iree_hal_amd_gpu_sdma_command_buffer_t* buffer = (void*)base;
  return buffer->program.data.data_length ? &buffer->storage : NULL;
}

iree_host_size_t iree_hal_amd_gpu_sdma_command_buffer_invocation_size(
    iree_hal_command_buffer_t* base) {
  return ((iree_hal_amd_gpu_sdma_command_buffer_t*)base)->invocation_size;
}

iree_host_size_t iree_hal_amd_gpu_sdma_command_buffer_operation_count(
    iree_hal_command_buffer_t* base) {
  return ((iree_hal_amd_gpu_sdma_command_buffer_t*)base)->operation_count;
}

static iree_status_t iree_hal_amd_gpu_sdma_command_status(
    iree_hal_amd_gpu_sdma_command_buffer_t* buffer, iree_status_t status) {
  if (!iree_status_is_ok(status) &&
      iree_status_is_ok(buffer->recording_status)) {
    buffer->recording_status = iree_status_clone(status);
  }
  return status;
}

static iree_status_t iree_hal_amd_gpu_sdma_command_flush_barrier(
    iree_hal_amd_gpu_sdma_command_buffer_t* buffer) {
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_sdma_recording_barrier(
      &buffer->recording, buffer->barriers, buffer->pending_barrier));
  buffer->pending_barrier = 0;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_sdma_command_begin(
    iree_hal_command_buffer_t* base) {
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_sdma_command_seal(
    iree_hal_amd_gpu_sdma_command_buffer_t* buffer) {
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_sdma_command_flush_barrier(buffer));
  iree_host_size_t bindings_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &buffer->invocation_size,
      IREE_STRUCT_FIELD(buffer->base.binding_count,
                        iree_hal_amd_gpu_command_binding_t, &bindings_offset),
      IREE_STRUCT_FIELD(buffer->inputs.count + 1, uint64_t,
                        &buffer->values_offset)));
  iree_hal_amd_gpu_command_inputs_seal(&buffer->inputs);
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_sdma_recording_seal(&buffer->recording,
                                                            &buffer->program));
  if (buffer->program.data.data_length) {
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_program_initialize(
        buffer->pool, 64, (uint32_t)buffer->program.data.data_length, 0,
        &buffer->storage));
    memcpy(buffer->storage.contents.data, buffer->program.data.data,
           buffer->program.data.data_length);
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_program_seal(&buffer->storage));
  }
  if (buffer->resources) {
    iree_hal_resource_set_freeze(buffer->resources);
  }
  iree_hal_amd_gpu_sdma_recording_deinitialize(&buffer->recording);
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_sdma_command_end(
    iree_hal_command_buffer_t* base) {
  iree_hal_amd_gpu_sdma_command_buffer_t* buffer = (void*)base;
  IREE_RETURN_IF_ERROR(iree_status_clone(buffer->recording_status));
  return iree_hal_amd_gpu_sdma_command_status(
      buffer, iree_hal_amd_gpu_sdma_command_seal(buffer));
}

static iree_status_t iree_hal_amd_gpu_sdma_command_begin_debug_group(
    iree_hal_command_buffer_t* base, iree_string_view_t label,
    iree_hal_label_color_t color, const iree_hal_label_location_t* location) {
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_sdma_command_end_debug_group(
    iree_hal_command_buffer_t* base) {
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_sdma_command_barrier(
    iree_hal_command_buffer_t* base, const iree_hal_barrier_t* barrier) {
  iree_hal_amd_gpu_sdma_command_buffer_t* buffer = (void*)base;
  IREE_RETURN_IF_ERROR(iree_status_clone(buffer->recording_status));
  const iree_hal_barrier_list_t list = {.count = 1, .values = barrier};
  buffer->pending_barrier |= iree_hal_amd_gpu_sdma_barrier_prepare(&list);
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_sdma_command_transfer(
    iree_hal_amd_gpu_sdma_command_buffer_t* buffer,
    iree_hal_transfer_operation_type_t type, iree_hal_buffer_ref_t source,
    iree_hal_buffer_ref_t target, const void* source_bytes, uint32_t pattern) {
  const bool copy = type == IREE_HAL_TRANSFER_OPERATION_TYPE_COPY;
  const bool update = type == IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE;
  const bool whole = target.length == IREE_HAL_WHOLE_BUFFER;
  if (copy && whole) {
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_command_inputs_equal_lengths(
        &buffer->inputs, source, target));
  }
  uint64_t length = target.length;
  if (whole && target.buffer) {
    length = iree_hal_buffer_byte_length(target.buffer) - target.offset;
  }
  if (!length) {
    return iree_ok_status();
  }
  iree_hal_amd_gpu_sdma_program_range_t range = {
      .source_value = IREE_HAL_AMD_GPU_SDMA_VALUE_PRIVATE,
      .length_value = UINT32_MAX,
      .pattern = pattern,
      .source_offset = source.offset,
      .target_offset = target.offset,
      .length = length,
  };
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_command_inputs_add_binding(
      &buffer->inputs, target, 0, 0, 1, whole, &range.target_value));
  if (copy) {
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_command_inputs_add_binding(
        &buffer->inputs, source, 0, 0, 1, whole, &range.source_value));
  } else {
    if (update && length > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "SDMA captured update exceeds its data section");
    }
    uint32_t data_offset = 0;
    void* data = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_recording_data(
        &buffer->recording.storage, update ? (uint32_t)length : 8, 4,
        &data_offset, &data));
    if (update) {
      memcpy(data, source_bytes, (size_t)length);
    } else {
      iree_unaligned_store_le_u32(data, pattern);
      iree_unaligned_store_le_u32((uint8_t*)data + 4, pattern);
    }
    range.source_offset = data_offset;
  }
  if (length == IREE_HAL_WHOLE_BUFFER) {
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_command_inputs_add_length(
        &buffer->inputs, target.buffer_slot, &range.length_value));
    range.length = 0 - target.offset;
  }
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_sdma_command_flush_barrier(buffer));
  if ((copy || update) && length != IREE_HAL_WHOLE_BUFFER) {
    iree_status_t status = iree_ok_status();
    for (uint64_t offset = 0; offset < length && iree_status_is_ok(status);) {
      const uint32_t part = (uint32_t)iree_min(
          length - offset, IREE_HAL_AMD_GPU_SDMA_MAX_PACKET_LENGTH);
      uint32_t command_offset = 0;
      uint32_t* words = NULL;
      status = iree_hal_amd_gpu_sdma_recording_commands(
          &buffer->recording, 7, &command_offset, &words);
      if (iree_status_is_ok(status)) {
        iree_hal_amd_gpu_sdma_emit_copy(words, buffer->recording.features, 0, 0,
                                        part);
        status = iree_hal_amd_gpu_sdma_recording_pointer(
            &buffer->recording, command_offset + 12, range.source_value,
            range.source_offset + offset);
      }
      if (iree_status_is_ok(status)) {
        status = iree_hal_amd_gpu_sdma_recording_pointer(
            &buffer->recording, command_offset + 20, range.target_value,
            range.target_offset + offset);
      }
      offset += part;
    }
    IREE_RETURN_IF_ERROR(status);
  } else {
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_sdma_recording_range(
        &buffer->recording,
        copy ? IREE_HAL_AMD_GPU_SDMA_PROGRAM_COPY
             : IREE_HAL_AMD_GPU_SDMA_PROGRAM_FILL,
        range));
  }
  ++buffer->operation_count;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_sdma_command_fill_buffer(
    iree_hal_command_buffer_t* base, iree_hal_buffer_ref_t target,
    const void* pattern, iree_host_size_t pattern_length,
    iree_hal_fill_flags_t flags) {
  iree_hal_amd_gpu_sdma_command_buffer_t* buffer = (void*)base;
  IREE_RETURN_IF_ERROR(iree_status_clone(buffer->recording_status));
  uint32_t value = 0;
  memcpy(&value, pattern, pattern_length);
  if (pattern_length == 1) {
    value *= 0x01010101u;
  }
  if (pattern_length == 2) {
    value |= value << 16;
  }
  return iree_hal_amd_gpu_sdma_command_status(
      buffer, iree_hal_amd_gpu_sdma_command_transfer(
                  buffer, IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
                  (iree_hal_buffer_ref_t){0}, target, NULL, value));
}

static iree_status_t iree_hal_amd_gpu_sdma_command_update_buffer(
    iree_hal_command_buffer_t* base, const void* source,
    iree_host_size_t source_offset, iree_hal_buffer_ref_t target,
    iree_hal_update_flags_t flags) {
  iree_hal_amd_gpu_sdma_command_buffer_t* buffer = (void*)base;
  IREE_RETURN_IF_ERROR(iree_status_clone(buffer->recording_status));
  return iree_hal_amd_gpu_sdma_command_status(
      buffer, iree_hal_amd_gpu_sdma_command_transfer(
                  buffer, IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE,
                  (iree_hal_buffer_ref_t){0}, target,
                  (const uint8_t*)source + source_offset, 0));
}

static iree_status_t iree_hal_amd_gpu_sdma_command_copy_buffer(
    iree_hal_command_buffer_t* base, iree_hal_buffer_ref_t source,
    iree_hal_buffer_ref_t target, iree_hal_copy_flags_t flags) {
  iree_hal_amd_gpu_sdma_command_buffer_t* buffer = (void*)base;
  IREE_RETURN_IF_ERROR(iree_status_clone(buffer->recording_status));
  return iree_hal_amd_gpu_sdma_command_status(
      buffer, iree_hal_amd_gpu_sdma_command_transfer(
                  buffer, IREE_HAL_TRANSFER_OPERATION_TYPE_COPY, source, target,
                  NULL, 0));
}

static iree_status_t iree_hal_amd_gpu_sdma_command_sample(
    iree_hal_amd_gpu_sdma_command_buffer_t* buffer,
    iree_hal_buffer_ref_t target) {
  uint32_t target_value = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_command_inputs_add_binding(
      &buffer->inputs, target, 0, 8, 8, false, &target_value));
  if (buffer->timestamp_offset == UINT32_MAX) {
    void* data = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_recording_data(
        &buffer->recording.storage, 32, 32, &buffer->timestamp_offset, &data));
  }
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_sdma_command_flush_barrier(buffer));
  uint32_t offset = 0;
  uint32_t* words = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_sdma_recording_commands(
      &buffer->recording, 11, &offset, &words));
  words = iree_hal_amd_gpu_sdma_emit_timestamp(words,
                                               buffer->recording.features, 0);
  *words++ = 0;
  iree_hal_amd_gpu_sdma_emit_copy(words, buffer->recording.features, 0, 0, 8);
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_sdma_recording_pointer(
      &buffer->recording, offset + 4, IREE_HAL_AMD_GPU_SDMA_VALUE_PRIVATE,
      buffer->timestamp_offset));
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_sdma_recording_pointer(
      &buffer->recording, offset + 28, IREE_HAL_AMD_GPU_SDMA_VALUE_PRIVATE,
      buffer->timestamp_offset));
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_sdma_recording_pointer(
      &buffer->recording, offset + 36, target_value, target.offset));
  ++buffer->operation_count;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_sdma_command_timestamp(
    iree_hal_command_buffer_t* base, iree_hal_execution_stage_t source_stage,
    iree_hal_execution_stage_t target_stage, iree_hal_buffer_ref_t target,
    iree_hal_timestamp_flags_t flags) {
  iree_hal_amd_gpu_sdma_command_buffer_t* buffer = (void*)base;
  IREE_RETURN_IF_ERROR(iree_status_clone(buffer->recording_status));
  return iree_hal_amd_gpu_sdma_command_status(
      buffer, iree_hal_amd_gpu_sdma_command_sample(buffer, target));
}

static iree_status_t iree_hal_amd_gpu_sdma_command_advise_buffer(
    iree_hal_command_buffer_t* base, iree_hal_buffer_ref_t reference,
    iree_hal_memory_advise_flags_t flags, uint64_t arg0, uint64_t arg1) {
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_sdma_command_atomic_wait(
    iree_hal_command_buffer_t* base, iree_hal_execution_stage_t source_stage,
    iree_hal_execution_stage_t target_stage, iree_hal_buffer_ref_t target,
    iree_hal_atomic_wait_params_t params) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "SDMA family does not advertise public atomic waits");
}
static iree_status_t iree_hal_amd_gpu_sdma_command_atomic_store(
    iree_hal_command_buffer_t* base, iree_hal_execution_stage_t source_stage,
    iree_hal_execution_stage_t target_stage, iree_hal_buffer_ref_t target,
    iree_hal_atomic_store_params_t params) {
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "SDMA family does not advertise public atomic stores");
}
static iree_status_t iree_hal_amd_gpu_sdma_command_atomic_rmw(
    iree_hal_command_buffer_t* base, iree_hal_execution_stage_t source_stage,
    iree_hal_execution_stage_t target_stage, iree_hal_buffer_ref_t target,
    iree_hal_atomic_rmw_params_t params) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "SDMA family does not advertise public atomic RMW");
}
static iree_status_t iree_hal_amd_gpu_sdma_command_dispatch(
    iree_hal_command_buffer_t* base, iree_hal_executable_t* executable,
    iree_hal_executable_function_t function, iree_hal_dispatch_config_t config,
    iree_const_byte_span_t constants, iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "SDMA transfer queues cannot dispatch programs");
}

iree_status_t iree_hal_amd_gpu_sdma_command_invocation_initialize(
    iree_hal_command_buffer_t* base, iree_hal_buffer_binding_table_t bindings,
    iree_hal_queue_execute_flags_t flags, void* storage,
    iree_hal_amd_gpu_sdma_command_invocation_t* out_invocation) {
  iree_hal_amd_gpu_sdma_command_buffer_t* buffer = (void*)base;
  *out_invocation = (iree_hal_amd_gpu_sdma_command_invocation_t){
      .command_buffer = base,
      .bindings = storage,
      .values = (void*)((uint8_t*)storage + buffer->values_offset),
      .borrow_bindings = iree_any_bit_set(
          flags, IREE_HAL_QUEUE_EXECUTE_FLAG_BORROW_BINDING_TABLE_LIFETIME),
  };
  iree_hal_command_buffer_retain(base);
  if (!iree_status_is_ok(buffer->recording_status)) {
    if (base->binding_count) {
      memset(storage, 0,
             base->binding_count * sizeof(*out_invocation->bindings));
    }
    return iree_status_clone(buffer->recording_status);
  }
  return iree_hal_amd_gpu_command_inputs_capture(
      &buffer->inputs, bindings, flags, out_invocation->bindings);
}

void iree_hal_amd_gpu_sdma_command_invocation_deinitialize(
    iree_hal_amd_gpu_sdma_command_invocation_t* invocation) {
  if (!invocation->borrow_bindings) {
    for (uint32_t i = 0; i < invocation->command_buffer->binding_count; ++i) {
      iree_hal_buffer_release(invocation->bindings[i].buffer);
    }
  }
  iree_hal_command_buffer_release(invocation->command_buffer);
}

iree_status_t iree_hal_amd_gpu_sdma_command_invocation_resolve(
    const iree_hal_amd_gpu_sdma_command_invocation_t* invocation,
    uint32_t queue_ordinal, uint64_t private_address) {
  const iree_hal_amd_gpu_sdma_command_buffer_t* buffer =
      (const void*)invocation->command_buffer;
  invocation->values[IREE_HAL_AMD_GPU_SDMA_VALUE_PRIVATE] = private_address;
  return iree_hal_amd_gpu_command_inputs_write(
      &buffer->inputs, invocation->bindings, queue_ordinal,
      invocation->values + IREE_HAL_AMD_GPU_SDMA_VALUE_FIRST_INPUT);
}

static const iree_hal_command_buffer_vtable_t
    iree_hal_amd_gpu_sdma_command_buffer_vtable = {
        .destroy = iree_hal_amd_gpu_sdma_command_buffer_destroy,
        .begin = iree_hal_amd_gpu_sdma_command_begin,
        .end = iree_hal_amd_gpu_sdma_command_end,
        .begin_debug_group = iree_hal_amd_gpu_sdma_command_begin_debug_group,
        .end_debug_group = iree_hal_amd_gpu_sdma_command_end_debug_group,
        .barrier = iree_hal_amd_gpu_sdma_command_barrier,
        .timestamp = iree_hal_amd_gpu_sdma_command_timestamp,
        .atomic_wait = iree_hal_amd_gpu_sdma_command_atomic_wait,
        .atomic_store = iree_hal_amd_gpu_sdma_command_atomic_store,
        .atomic_rmw = iree_hal_amd_gpu_sdma_command_atomic_rmw,
        .advise_buffer = iree_hal_amd_gpu_sdma_command_advise_buffer,
        .fill_buffer = iree_hal_amd_gpu_sdma_command_fill_buffer,
        .update_buffer = iree_hal_amd_gpu_sdma_command_update_buffer,
        .copy_buffer = iree_hal_amd_gpu_sdma_command_copy_buffer,
        .dispatch = iree_hal_amd_gpu_sdma_command_dispatch,
};
