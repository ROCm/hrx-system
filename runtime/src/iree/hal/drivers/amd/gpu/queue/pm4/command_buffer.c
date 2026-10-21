// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/command_buffer.h"

#include "iree/base/internal/arena.h"
#include "iree/hal/drivers/amd/gpu/executable.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/commands.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/dispatch.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/recording.h"
#include "iree/hal/drivers/amd/gpu/queue/program.h"
#include "iree/hal/utils/resource_set.h"

enum {
  IREE_HAL_AMD_GPU_PM4_VALUE_INSTANCE = 0,
  IREE_HAL_AMD_GPU_PM4_VALUE_DISPATCH_ID = 1,
  IREE_HAL_AMD_GPU_PM4_VALUE_CONTINUATION = 2,
  IREE_HAL_AMD_GPU_PM4_VALUE_FIRST_INPUT = 3,
  IREE_HAL_AMD_GPU_PM4_VALUE_PROFILE_ADDRESS = 3,
  IREE_HAL_AMD_GPU_PM4_VALUE_PROFILE_HEADER = 4,
  IREE_HAL_AMD_GPU_PM4_SITE_QUEUE = 0,
  IREE_HAL_AMD_GPU_PM4_SITE_PROGRAM = 1,
};

typedef struct iree_hal_amd_gpu_pm4_command_buffer_t {
  // Public resource identity and optional external recording validation.
  iree_hal_command_buffer_t base;
  // Allocator for this object, inputs, and temporary recording bytes.
  iree_allocator_t host_allocator;
  // Borrowed native owner, dominated by the application-held family device.
  iree_hal_amd_gpu_pm4_program_pool_t* pool;
  // Independently retained resident execution facts after end.
  iree_hal_amd_gpu_pm4_command_program_t* program;
  // Temporary native bytes and fixups, released after sealing.
  iree_hal_amd_gpu_pm4_recording_t recording;
  // Deduplicated external values and aggregate binding requirements.
  iree_hal_amd_gpu_command_inputs_t inputs;
  // Device-local LDS capacity for public dispatch admission.
  uint32_t maximum_group_segment_byte_length;
  // Native per-wave private capacity; zero when the service is unavailable.
  uint32_t maximum_scratch_wave_byte_length;
  // Captured physical cache behavior; no topology is retained.
  iree_hal_amd_gpu_pm4_barrier_profile_t profile;
  // First failed native recording mutation; prevents sealing partial packets.
  iree_status_t recording_status;
  // Adjacent public barriers are combined before the next native command.
  iree_hal_amd_gpu_pm4_barrier_t pending_barrier;
  // Copies recorded since the last join; drained before the program exits.
  bool pending_dma;
  // Blocks backing direct-resource ownership.
  iree_arena_block_pool_t resource_blocks;
  // Retained buffers/executables, or NULL for explicitly unretained recordings.
  iree_hal_resource_set_t* resources;
} iree_hal_amd_gpu_pm4_command_buffer_t;

// Profile-capable recordings reserve a sample base and packet header. Ordinary
// recordings have no profiling inputs or packets.
static uint32_t iree_hal_amd_gpu_pm4_command_first_input(
    const iree_hal_amd_gpu_pm4_command_buffer_t* buffer) {
  return IREE_HAL_AMD_GPU_PM4_VALUE_FIRST_INPUT +
         (buffer->program->profile ? 2 : 0);
}

void iree_hal_amd_gpu_pm4_command_program_retain(
    iree_hal_amd_gpu_pm4_command_program_t* program) {
  if (!program) {
    return;
  }
  iree_atomic_ref_count_inc(&program->ref_count);
}

void iree_hal_amd_gpu_pm4_command_program_release(
    iree_hal_amd_gpu_pm4_command_program_t* program) {
  if (!program) {
    return;
  }
  if (iree_atomic_ref_count_dec(&program->ref_count) != 1) {
    return;
  }
  const iree_allocator_t host_allocator = program->host_allocator;
  iree_hal_amd_gpu_profile_commands_release(program->profile);
  iree_hal_amd_gpu_program_deinitialize(&program->resident);
  iree_allocator_free(host_allocator, program);
}

static const iree_hal_command_buffer_vtable_t
    iree_hal_amd_gpu_pm4_command_buffer_vtable;

static void iree_hal_amd_gpu_pm4_command_buffer_destroy(
    iree_hal_command_buffer_t* base) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_amd_gpu_pm4_command_buffer_t* buffer = (void*)base;
  const iree_allocator_t host_allocator = buffer->host_allocator;
  iree_hal_amd_gpu_pm4_command_program_t* program = buffer->program;
  // Last recording ownership proves every invocation has joined. Drop caller
  // resources now so an immediately following pool release sees no live
  // children. Cold growth independently retains only the resident program.
  iree_status_free(buffer->recording_status);
  iree_hal_amd_gpu_recording_deinitialize(&buffer->recording.storage);
  iree_hal_resource_set_free(buffer->resources);
  iree_arena_block_pool_deinitialize(&buffer->resource_blocks);
  iree_hal_amd_gpu_command_inputs_deinitialize(&buffer->inputs);
  iree_allocator_free(host_allocator, buffer);
  iree_hal_amd_gpu_pm4_command_program_release(program);
  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_amd_gpu_pm4_command_buffer_create(
    const iree_hal_queue_family_t* family,
    iree_hal_amd_gpu_pm4_program_pool_t* pool,
    iree_hal_command_buffer_mode_t mode, iree_hal_command_category_t categories,
    iree_host_size_t binding_capacity,
    uint32_t maximum_group_segment_byte_length,
    uint32_t maximum_scratch_wave_byte_length,
    const iree_hal_amd_gpu_pm4_barrier_profile_t* profile,
    iree_allocator_t host_allocator, iree_hal_command_buffer_t** out_buffer) {
  *out_buffer = NULL;
  if (binding_capacity >= UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "resident binding table exceeds its index space");
  }
  iree_host_size_t size = 0, requirements_offset = 0, validation_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_amd_gpu_pm4_command_buffer_t), &size,
      IREE_STRUCT_FIELD(binding_capacity + 1,
                        iree_hal_amd_gpu_binding_requirement_t,
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
  buffer->pool = pool;
  buffer->maximum_group_segment_byte_length = maximum_group_segment_byte_length;
  buffer->maximum_scratch_wave_byte_length = maximum_scratch_wave_byte_length;
  buffer->profile = *profile;
  iree_hal_amd_gpu_command_inputs_initialize(
      &buffer->base, 0, NULL, (void*)((uint8_t*)buffer + requirements_offset),
      host_allocator, &buffer->inputs);
  iree_hal_amd_gpu_pm4_recording_initialize(
      host_allocator, IREE_HAL_AMD_GPU_PM4_VALUE_INSTANCE, &buffer->recording);
  iree_arena_block_pool_initialize(4096, host_allocator,
                                   &buffer->resource_blocks);
  buffer->program = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*buffer->program), (void**)&buffer->program);
  if (iree_status_is_ok(status)) {
    *buffer->program = (iree_hal_amd_gpu_pm4_command_program_t){
        .host_allocator = host_allocator,
    };
    iree_atomic_ref_count_init(&buffer->program->ref_count);
  }
  if (iree_status_is_ok(status) &&
      iree_any_bit_set(
          mode, IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA |
                    IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_DISPATCH_METADATA)) {
    iree_hal_profile_command_buffer_record_t record =
        iree_hal_profile_command_buffer_record_default();
    record.command_buffer_id =
        iree_hal_command_buffer_profile_id(&buffer->base);
    record.mode = mode;
    record.command_categories = categories;
    record.queue_family_ordinal = family->ordinal;
    record.physical_device_ordinal = 0;
    status = iree_hal_amd_gpu_profile_commands_create(
        &record, host_allocator, &buffer->program->profile);
  }
  if (iree_status_is_ok(status) &&
      !iree_any_bit_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED)) {
    status = iree_hal_resource_set_allocate(&buffer->resource_blocks,
                                            &buffer->resources);
  }
  if (iree_status_is_ok(status)) {
    buffer->inputs.value_base =
        iree_hal_amd_gpu_pm4_command_first_input(buffer);
    buffer->inputs.resources = buffer->resources;
    *out_buffer = &buffer->base;
  } else {
    iree_hal_command_buffer_release(&buffer->base);
  }
  return status;
}

iree_host_size_t iree_hal_amd_gpu_pm4_command_buffer_operation_count(
    iree_hal_command_buffer_t* base) {
  return ((iree_hal_amd_gpu_pm4_command_buffer_t*)base)
      ->program->operation_count;
}

static iree_status_t iree_hal_amd_gpu_pm4_command_flush_barrier(
    iree_hal_amd_gpu_pm4_command_buffer_t* buffer) {
  // Primary publications join their own CP DMA before entering this program.
  // Only copies recorded since the last join can leave a DMA dependency here.
  if (!buffer->pending_dma) {
    buffer->pending_barrier.execution &=
        ~IREE_HAL_AMD_GPU_PM4_BARRIER_EXECUTION_DMA;
  }
  const uint32_t count =
      iree_hal_amd_gpu_pm4_barrier_word_count(buffer->pending_barrier);
  if (!count) {
    return iree_ok_status();
  }
  uint32_t offset = 0;
  uint32_t* words = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_recording_commands(
      &buffer->recording, count, &offset, &words));
  iree_hal_amd_gpu_pm4_barrier_emit(words, buffer->pending_barrier);
  if (buffer->pending_barrier.execution &
      IREE_HAL_AMD_GPU_PM4_BARRIER_EXECUTION_DMA) {
    buffer->pending_dma = false;
  }
  buffer->pending_barrier = (iree_hal_amd_gpu_pm4_barrier_t){0};
  ++buffer->program->operation_count;
  return iree_ok_status();
}

// Synchronous recording owns native acquisition, upload initialization, and
// sealing on its caller, independently of queued asynchronous slab batches.
static iree_status_t iree_hal_amd_gpu_pm4_command_seal(
    iree_hal_amd_gpu_pm4_command_buffer_t* buffer, uint32_t length) {
  iree_hal_amd_gpu_program_t* program = &buffer->program->resident;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_program_initialize(
      &buffer->pool->storage, buffer->recording.storage.alignment, length,
      iree_hal_amd_gpu_recording_fixup_count(&buffer->recording.storage),
      program));
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_amd_gpu_recording_write(
      &buffer->recording.storage, program->contents.data,
      (void*)(program->contents.data + program->fixup_offset));
  iree_status_t status = iree_hal_amd_gpu_program_seal(program);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Failed append/seal operations invalidate the native recording. Preserve the
// original error instead of allowing an end/submission to consume a partially
// initialized packet or a partially acquired native allocation.
static iree_status_t iree_hal_amd_gpu_pm4_command_record_status(
    iree_hal_amd_gpu_pm4_command_buffer_t* buffer, iree_status_t status) {
  if (!iree_status_is_ok(status) &&
      iree_status_is_ok(buffer->recording_status)) {
    buffer->recording_status = iree_status_clone(status);
  }
  return status;
}

static iree_status_t iree_hal_amd_gpu_pm4_command_buffer_begin(
    iree_hal_command_buffer_t* base) {
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_pm4_command_end(
    iree_hal_command_buffer_t* base) {
  iree_hal_amd_gpu_pm4_command_buffer_t* buffer = (void*)base;
  if (buffer->pending_dma) {
    buffer->pending_barrier.execution |=
        IREE_HAL_AMD_GPU_PM4_BARRIER_EXECUTION_DMA;
  }
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_flush_barrier(buffer));
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_recording_end(
      &buffer->recording, IREE_HAL_AMD_GPU_PM4_VALUE_CONTINUATION,
      buffer->pool->chain));
  buffer->program->entry = (iree_hal_amd_gpu_pm4_program_block_t){
      .offset = 0,
      .word_count = buffer->recording.entry_word_count,
  };
  const uint64_t length =
      iree_align_uint64(buffer->recording.storage.commands.length,
                        buffer->recording.storage.alignment) +
      buffer->recording.storage.data.length;
  if (length > UINT32_MAX - 15) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "resident command/data placement exceeds 4 GiB");
  }
  buffer->program->value_count =
      buffer->inputs.count + iree_hal_amd_gpu_pm4_command_first_input(buffer);
  buffer->program->argument_length =
      sizeof(iree_hal_amd_gpu_pm4_program_arguments_t) +
      buffer->program->value_count * 8;
  iree_hal_amd_gpu_command_inputs_seal(&buffer->inputs);
  iree_status_t status =
      iree_hal_amd_gpu_pm4_command_seal(buffer, (uint32_t)length);
  if (iree_status_is_ok(status)) {
    if (buffer->resources) {
      iree_hal_resource_set_freeze(buffer->resources);
    }
    iree_hal_amd_gpu_recording_deinitialize(&buffer->recording.storage);
  }
  return status;
}

static iree_status_t iree_hal_amd_gpu_pm4_command_buffer_end(
    iree_hal_command_buffer_t* base) {
  iree_hal_amd_gpu_pm4_command_buffer_t* buffer = (void*)base;
  IREE_RETURN_IF_ERROR(iree_status_clone(buffer->recording_status));
  return iree_hal_amd_gpu_pm4_command_record_status(
      buffer, iree_hal_amd_gpu_pm4_command_end(base));
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

static iree_status_t iree_hal_amd_gpu_pm4_command_buffer_advise_buffer(
    iree_hal_command_buffer_t* base, iree_hal_buffer_ref_t reference,
    iree_hal_memory_advise_flags_t flags, uint64_t arg0, uint64_t arg1) {
  return iree_ok_status();
}

// Appends one address replacement without retaining a host record for the
// operation. Both halves read the same dense input and overwrite complete
// words.
static iree_status_t iree_hal_amd_gpu_pm4_command_pointer(
    iree_hal_amd_gpu_pm4_command_buffer_t* buffer, bool data, uint32_t offset,
    uint32_t value, uint64_t displacement, bool private_data) {
  return iree_hal_amd_gpu_recording_fixup(
      &buffer->recording.storage, offset, value, displacement, 0,
      (data ? IREE_HAL_AMD_GPU_RECORDING_FIXUP_TARGET_DATA : 0) |
          (private_data ? IREE_HAL_AMD_GPU_RECORDING_FIXUP_ADDEND_DATA : 0) |
          IREE_HAL_AMD_GPU_RECORDING_FIXUP_WIDE);
}

static iree_status_t iree_hal_amd_gpu_pm4_command_atomic(
    iree_hal_amd_gpu_pm4_command_buffer_t* buffer,
    iree_hal_execution_stage_t source_stage,
    iree_hal_execution_stage_t target_stage, iree_hal_buffer_ref_t target,
    iree_hal_amd_gpu_control_t command) {
  uint32_t value = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_command_inputs_add_atomic(
      &buffer->inputs, target, command, &value));
  iree_hal_amd_gpu_pm4_barrier_t after = {0};
  iree_hal_amd_gpu_pm4_barrier_atomic(&buffer->profile, command.flags,
                                      source_stage, &buffer->pending_barrier,
                                      &after);
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_flush_barrier(buffer));
  uint32_t offset = 0;
  uint32_t* words = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_recording_commands(
      &buffer->recording, iree_hal_amd_gpu_control_word_count(command), &offset,
      &words));
  iree_hal_amd_gpu_control_emit(words, 0, command);
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_pointer(
      buffer, false,
      offset + 4 * iree_hal_amd_gpu_control_address_word(command), value,
      target.offset, false));
  buffer->pending_barrier = after;
  ++buffer->program->operation_count;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_pm4_command_buffer_atomic_wait(
    iree_hal_command_buffer_t* base, iree_hal_execution_stage_t source_stage,
    iree_hal_execution_stage_t target_stage, iree_hal_buffer_ref_t target,
    iree_hal_atomic_wait_params_t params) {
  iree_hal_amd_gpu_pm4_command_buffer_t* buffer = (void*)base;
  IREE_RETURN_IF_ERROR(iree_status_clone(buffer->recording_status));
  return iree_hal_amd_gpu_pm4_command_record_status(
      buffer, iree_hal_amd_gpu_pm4_command_atomic(
                  buffer, source_stage, target_stage, target,
                  iree_hal_amd_gpu_control_wait(params)));
}

static iree_status_t iree_hal_amd_gpu_pm4_command_buffer_atomic_store(
    iree_hal_command_buffer_t* base, iree_hal_execution_stage_t source_stage,
    iree_hal_execution_stage_t target_stage, iree_hal_buffer_ref_t target,
    iree_hal_atomic_store_params_t params) {
  iree_hal_amd_gpu_pm4_command_buffer_t* buffer = (void*)base;
  IREE_RETURN_IF_ERROR(iree_status_clone(buffer->recording_status));
  return iree_hal_amd_gpu_pm4_command_record_status(
      buffer, iree_hal_amd_gpu_pm4_command_atomic(
                  buffer, source_stage, target_stage, target,
                  iree_hal_amd_gpu_control_store(params)));
}

static iree_status_t iree_hal_amd_gpu_pm4_command_buffer_atomic_rmw(
    iree_hal_command_buffer_t* base, iree_hal_execution_stage_t source_stage,
    iree_hal_execution_stage_t target_stage, iree_hal_buffer_ref_t target,
    iree_hal_atomic_rmw_params_t params) {
  iree_hal_amd_gpu_pm4_command_buffer_t* buffer = (void*)base;
  IREE_RETURN_IF_ERROR(iree_status_clone(buffer->recording_status));
  return iree_hal_amd_gpu_pm4_command_record_status(
      buffer, iree_hal_amd_gpu_pm4_command_atomic(
                  buffer, source_stage, target_stage, target,
                  iree_hal_amd_gpu_control_rmw(params)));
}

static iree_status_t iree_hal_amd_gpu_pm4_command_transfer(
    iree_hal_amd_gpu_pm4_command_buffer_t* buffer,
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
  // Byte-aligned copies use CP DMA. A direct fill can use its immediate form
  // only when prepared backing proves alignment without reading an address.
  const bool aligned_fill =
      !copy && !update && target.buffer && target.buffer->memory.backing &&
      target.buffer->memory.backing->allocation_alignment >= 4 &&
      !((target.buffer->memory.offset + target.offset) & 3) && !(length & 3);
  const bool native_dma =
      length != IREE_HAL_WHOLE_BUFFER && (copy || update || aligned_fill);
  const uint32_t site = native_dma ? IREE_HAL_AMD_GPU_PM4_SITE_QUEUE
                                   : IREE_HAL_AMD_GPU_PM4_SITE_PROGRAM;
  uint32_t source_value = IREE_HAL_AMD_GPU_PM4_VALUE_INSTANCE, target_value = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_command_inputs_add_binding(
      &buffer->inputs, target, site, 0, 1, whole, &target_value));
  if (copy) {
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_command_inputs_add_binding(
        &buffer->inputs, source, site, 0, 1, whole, &source_value));
  }
  uint32_t update_offset = 0;
  if (update) {
    void* bytes = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_recording_data(
        &buffer->recording.storage, (uint32_t)length, 1, &update_offset,
        &bytes));
    memcpy(bytes, source_bytes, (iree_host_size_t)length);
  }
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_flush_barrier(buffer));
  if (native_dma) {
    iree_status_t status = iree_ok_status();
    uint64_t position = 0;
    while (position < length && iree_status_is_ok(status)) {
      const uint32_t part =
          (uint32_t)iree_min(length - position, UINT64_C(0x03fffffc));
      uint32_t command_offset = 0;
      uint32_t* words = NULL;
      status = iree_hal_amd_gpu_pm4_recording_commands(&buffer->recording, 7,
                                                       &command_offset, &words);
      if (!iree_status_is_ok(status)) {
        break;
      }
      if (aligned_fill) {
        iree_hal_amd_gpu_pm4_emit_dma_fill(words, 0, part, pattern);
      } else {
        iree_hal_amd_gpu_pm4_emit_dma_copy(words, 0, 0, part);
        status = iree_hal_amd_gpu_pm4_command_pointer(
            buffer, false, command_offset + 8, source_value,
            (update ? update_offset : source.offset) + position, update);
      }
      if (iree_status_is_ok(status)) {
        status = iree_hal_amd_gpu_pm4_command_pointer(
            buffer, false, command_offset + 16, target_value,
            target.offset + position, false);
      }
      position += part;
    }
    IREE_RETURN_IF_ERROR(status);
    buffer->pending_dma = true;
  } else {
    uint32_t argument_offset = 0;
    iree_hal_amd_gpu_blit_arguments_t* arguments = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_recording_data(
        &buffer->recording.storage, sizeof(*arguments), 8, &argument_offset,
        (void**)&arguments));
    *arguments = (iree_hal_amd_gpu_blit_arguments_t){
        .length = length == IREE_HAL_WHOLE_BUFFER ? 0 : length,
        .pattern = pattern,
        .fill = !copy,
    };
    if (copy) {
      IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_pointer(
          buffer, true, argument_offset, source_value, source.offset, false));
    }
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_pointer(
        buffer, true, argument_offset + 8, target_value, target.offset, false));
    if (length == IREE_HAL_WHOLE_BUFFER) {
      uint32_t length_value = 0;
      IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_command_inputs_add_length(
          &buffer->inputs, target.buffer_slot, &length_value));
      IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_pointer(
          buffer, true, argument_offset + 16, length_value, 0 - target.offset,
          false));
    }
    uint32_t command_offset = 0;
    uint32_t* words = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_recording_commands(
        &buffer->recording, 31, &command_offset, &words));
    iree_hal_amd_gpu_pm4_blit_emit(words, &buffer->pool->blit, 0);
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_pointer(
        buffer, false, command_offset + 24 * 4,
        IREE_HAL_AMD_GPU_PM4_VALUE_INSTANCE, argument_offset, true));
  }
  ++buffer->program->operation_count;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_pm4_command_buffer_transfer(
    iree_hal_amd_gpu_pm4_command_buffer_t* buffer,
    iree_hal_transfer_operation_type_t type, iree_hal_buffer_ref_t source,
    iree_hal_buffer_ref_t target, const void* source_bytes, uint32_t pattern) {
  IREE_RETURN_IF_ERROR(iree_status_clone(buffer->recording_status));
  return iree_hal_amd_gpu_pm4_command_record_status(
      buffer, iree_hal_amd_gpu_pm4_command_transfer(
                  buffer, type, source, target, source_bytes, pattern));
}

static iree_status_t iree_hal_amd_gpu_pm4_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base, iree_hal_buffer_ref_t target,
    const void* pattern, iree_host_size_t pattern_length,
    iree_hal_fill_flags_t flags) {
  uint32_t value = 0;
  memcpy(&value, pattern, pattern_length);
  if (pattern_length == 1) {
    value *= 0x01010101u;
  }
  if (pattern_length == 2) {
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

static iree_status_t iree_hal_amd_gpu_pm4_command_image(
    iree_hal_amd_gpu_pm4_command_buffer_t* buffer, const uint64_t* addresses,
    bool data, uint32_t offset, uint64_t address, uint32_t shift) {
  uint32_t value = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_command_inputs_add_image(
      &buffer->inputs, addresses, &value));
  return iree_hal_amd_gpu_recording_fixup(
      &buffer->recording.storage, offset, value, address - addresses[0], shift,
      (data ? IREE_HAL_AMD_GPU_RECORDING_FIXUP_TARGET_DATA : 0) |
          IREE_HAL_AMD_GPU_RECORDING_FIXUP_WIDE);
}

static iree_status_t iree_hal_amd_gpu_pm4_command_dispatch(
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
      function, buffer->maximum_group_segment_byte_length,
      buffer->maximum_scratch_wave_byte_length, config, constants,
      bindings.count, flags, &dispatch));
  const bool indirect_parameters =
      iree_hal_dispatch_uses_indirect_parameters(flags);
  const bool indirect = iree_hal_dispatch_uses_indirect_arguments(flags);
  if (!indirect_parameters && (!dispatch.geometry.workgroup_count[0] ||
                               !dispatch.geometry.workgroup_count[1] ||
                               !dispatch.geometry.workgroup_count[2])) {
    return iree_ok_status();
  }
  if (buffer->resources) {
    IREE_RETURN_IF_ERROR(
        iree_hal_resource_set_insert(buffer->resources, 1, &executable));
  }
  const bool prepare_packet = function->packet && indirect_parameters;
  if (prepare_packet) {
    buffer->pending_barrier.cache_control |=
        buffer->profile.command_processor_bypasses_l2 ? buffer->profile.acquire
                                                      : buffer->profile.shader;
  }
  if (function->launch.requirements &
      IREE_HAL_AMD_GPU_PM4_LAUNCH_REQUIREMENT_TSAN) {
    // Instrumented dispatches reuse one queue-local shadow arena. The next
    // generation must not touch that arena until prior shaders have finished.
    buffer->pending_barrier.execution |=
        IREE_HAL_AMD_GPU_PM4_BARRIER_EXECUTION_COMPUTE;
    buffer->pending_barrier.cache_control |= buffer->profile.shader;
  }
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_flush_barrier(buffer));
  const iree_hal_amd_gpu_pm4_dispatch_storage_t storage =
      iree_hal_amd_gpu_pm4_dispatch_storage(function, flags);
  uint32_t argument_offset = 0;
  void* arguments = NULL;
  if (storage.length) {
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_recording_data(
        &buffer->recording.storage, storage.length, storage.alignment,
        &argument_offset, &arguments));
    iree_hal_amd_gpu_pm4_dispatch_emplace_arguments(
        function, &dispatch, constants, NULL, flags, arguments);
  }
  const uint32_t prepare_words =
      prepare_packet ? IREE_HAL_AMD_GPU_PM4_DISPATCH_PREPARE_WORDS : 0;
  const uint32_t launch_words = function->launch.user_data_count
                                    ? 24 + function->launch.user_data_count
                                    : 22;
  const uint16_t copy_count = indirect_parameters && !indirect
                                  ? function->base.kernarg->indirect_copy_count
                                  : 0;
  const uint32_t dispatch_words =
      indirect_parameters
          ? iree_hal_amd_gpu_pm4_dispatch_indirect_word_count(copy_count)
          : 5;
  const uint32_t profile_words = buffer->program->profile ? 8 : 0;
  const uint32_t word_count =
      prepare_words + launch_words + dispatch_words + 2 * profile_words;
  uint32_t command_offset = 0;
  uint32_t* words = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_recording_commands(
      &buffer->recording, word_count, &command_offset, &words));
  uint32_t* cursor = words;
  if (prepare_packet) {
    cursor = iree_hal_amd_gpu_pm4_dispatch_emit_prepare(
        cursor, function->packet->prepare, 0, buffer->profile.shader);
  }
  const uint32_t launch_offset = command_offset + prepare_words * 4;
  const iree_hal_amd_gpu_pm4_launch_inputs_t inputs = {
      .private_segment_size =
          (function->base.metadata->private_segment_fixed_size + 3u) & ~3u};
  cursor = iree_hal_amd_gpu_pm4_launch_emit(
      cursor, &function->launch, dispatch.geometry.workgroup_size,
      dispatch.geometry.partial_size,
      dispatch.geometry.group_segment_byte_length, &inputs);
  const uint32_t initiator =
      function->launch.dispatch_initiator |
      (dispatch.geometry.partial_size[0] ? UINT32_C(1) << 6 : 0);
  const uint32_t profile_begin_offset =
      command_offset + (uint32_t)(cursor - words) * 4;
  if (buffer->program->profile) {
    cursor = iree_hal_amd_gpu_pm4_emit_compute_flush(cursor);
    cursor = iree_hal_amd_gpu_pm4_emit_clock(cursor, 0);
  }
  if (indirect_parameters) {
    iree_hal_amd_gpu_pm4_dispatch_emit_indirect(
        cursor, function->base.kernarg, copy_count, 0, 0,
        buffer->profile.command_processor_bypasses_l2 ? buffer->profile.acquire
                                                      : buffer->profile.shader,
        initiator);
  } else {
    iree_hal_amd_gpu_pm4_emit_dispatch_direct(
        cursor, dispatch.geometry.workgroup_count, initiator);
  }
  if (buffer->program->profile) {
    const uint32_t profile_end_offset = command_offset + (word_count - 8) * 4;
    uint32_t* end =
        iree_hal_amd_gpu_pm4_emit_compute_flush(cursor + dispatch_words);
    iree_hal_amd_gpu_pm4_emit_clock(end, 0);
    const uint64_t sample_offset = (uint64_t)buffer->program->dispatch_count *
                                   sizeof(iree_hal_amd_gpu_profile_sample_t);
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_recording_fixup(
        &buffer->recording.storage, profile_begin_offset,
        IREE_HAL_AMD_GPU_PM4_VALUE_PROFILE_HEADER, 0, 0, 0));
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_recording_fixup(
        &buffer->recording.storage, profile_end_offset,
        IREE_HAL_AMD_GPU_PM4_VALUE_PROFILE_HEADER, 0, 0, 0));
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_pointer(
        buffer, false, profile_begin_offset + 6 * 4,
        IREE_HAL_AMD_GPU_PM4_VALUE_PROFILE_ADDRESS, sample_offset, false));
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_pointer(
        buffer, false, profile_end_offset + 6 * 4,
        IREE_HAL_AMD_GPU_PM4_VALUE_PROFILE_ADDRESS, sample_offset + 8, false));
    iree_hal_amd_gpu_profile_dispatch_t capture;
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_executable_profile_dispatch(
        executable, token, &dispatch.geometry, flags, &capture));
    capture.event.flags |= IREE_HAL_PROFILE_DISPATCH_EVENT_FLAG_COMMAND_BUFFER;
    capture.event.command_buffer_id = iree_hal_command_buffer_profile_id(base);
    capture.event.command_index = buffer->program->operation_count;
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_profile_commands_append(
        buffer->program->profile, capture));
  }
  const uint16_t kernarg_word =
      iree_hal_amd_gpu_pm4_launch_user_data_word_offset(
          function->launch.user_data.kernarg_pointer);
  const uint32_t packet_offset = argument_offset + storage.packet_offset;
  uint32_t kernarg_value = IREE_HAL_AMD_GPU_PM4_VALUE_INSTANCE;
  uint64_t kernarg_displacement = argument_offset;
  if (indirect && (kernarg_word != UINT16_MAX || function->packet)) {
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_command_inputs_add_binding(
        &buffer->inputs, bindings.values[0], IREE_HAL_AMD_GPU_PM4_SITE_PROGRAM,
        function->base.kernarg->byte_length, function->base.kernarg->alignment,
        false, &kernarg_value));
    kernarg_displacement = bindings.values[0].offset;
  } else if (!indirect) {
    const uint16_t* offsets =
        iree_hal_amd_gpu_kernarg_layout_binding_offsets(function->base.kernarg);
    iree_status_t status = iree_ok_status();
    for (iree_host_size_t i = 0;
         i < bindings.count && iree_status_is_ok(status); ++i) {
      uint32_t value = 0;
      status = iree_hal_amd_gpu_command_inputs_add_binding(
          &buffer->inputs, bindings.values[i],
          IREE_HAL_AMD_GPU_PM4_SITE_PROGRAM, 0, 1, false, &value);
      if (iree_status_is_ok(status)) {
        status = iree_hal_amd_gpu_pm4_command_pointer(
            buffer, true, argument_offset + offsets[i], value,
            bindings.values[i].offset, false);
      }
    }
    IREE_RETURN_IF_ERROR(status);
  }
  if (kernarg_word != UINT16_MAX) {
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_pointer(
        buffer, false, launch_offset + kernarg_word * 4, kernarg_value,
        kernarg_displacement, !indirect));
  }
  const uint16_t dispatch_id_word =
      iree_hal_amd_gpu_pm4_launch_user_data_word_offset(
          function->launch.user_data.dispatch_id);
  if (dispatch_id_word != UINT16_MAX) {
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_pointer(
        buffer, false, launch_offset + dispatch_id_word * 4,
        IREE_HAL_AMD_GPU_PM4_VALUE_DISPATCH_ID, buffer->program->dispatch_count,
        false));
  }
  if (function->packet) {
    const uint16_t packet_word =
        iree_hal_amd_gpu_pm4_launch_user_data_word_offset(
            function->launch.user_data.dispatch_pointer);
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_pointer(
        buffer, false, launch_offset + packet_word * 4,
        IREE_HAL_AMD_GPU_PM4_VALUE_INSTANCE, packet_offset, true));
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_pointer(
        buffer, true,
        packet_offset +
            offsetof(iree_hal_amd_gpu_pm4_dispatch_packet_t, kernarg_address),
        kernarg_value, kernarg_displacement, !indirect));
  }
  if (indirect_parameters) {
    uint32_t parameters = 0;
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_command_inputs_add_binding(
        &buffer->inputs, config.workgroup_count_ref,
        IREE_HAL_AMD_GPU_PM4_SITE_QUEUE, 12, 4, false, &parameters));
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_pointer(
        buffer, false, command_offset + (word_count - profile_words - 3) * 4,
        parameters, config.workgroup_count_ref.offset, false));
    iree_status_t status = iree_ok_status();
    for (uint16_t i = 0; i < copy_count && iree_status_is_ok(status); ++i) {
      uint32_t* copy = cursor + i * 7;
      const uint32_t offset =
          command_offset +
          (prepare_words + launch_words + profile_words + i * 7) * 4;
      status = iree_hal_amd_gpu_pm4_command_pointer(
          buffer, false, offset + 8, parameters,
          config.workgroup_count_ref.offset +
              iree_unaligned_load_le_u64(copy + 2),
          false);
      if (iree_status_is_ok(status)) {
        status = iree_hal_amd_gpu_pm4_command_pointer(
            buffer, false, offset + 16, IREE_HAL_AMD_GPU_PM4_VALUE_INSTANCE,
            argument_offset + iree_unaligned_load_le_u64(copy + 4), true);
      }
    }
    IREE_RETURN_IF_ERROR(status);
    if (copy_count) {
      buffer->pending_dma = false;
    }
    if (prepare_packet) {
      IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_pointer(
          buffer, true, packet_offset + 64, parameters,
          config.workgroup_count_ref.offset, false));
      IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_pointer(
          buffer, true, packet_offset + 72, IREE_HAL_AMD_GPU_PM4_VALUE_INSTANCE,
          packet_offset, true));
      IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_pointer(
          buffer, false, command_offset + 24 * 4,
          IREE_HAL_AMD_GPU_PM4_VALUE_INSTANCE, packet_offset + 64, true));
    }
  }
  const uint64_t* image_addresses =
      iree_hal_amd_gpu_executable_queue_image_addresses(executable);
  if (image_addresses) {
    const uint64_t address = ((uint64_t)function->launch.program[0] << 8) |
                             ((uint64_t)function->launch.program[1] << 40);
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_image(
        buffer, image_addresses, false, launch_offset + 8, address, 8));
    if (function->packet) {
      IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_image(
          buffer, image_addresses, true,
          packet_offset +
              offsetof(iree_hal_amd_gpu_pm4_dispatch_packet_t, kernel_object),
          function->packet->kernel_object, 0));
      if (prepare_packet) {
        const iree_hal_amd_gpu_pm4_launch_t* prepare =
            function->packet->prepare;
        const uint64_t prepare_address = ((uint64_t)prepare->program[0] << 8) |
                                         ((uint64_t)prepare->program[1] << 40);
        IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_command_image(
            buffer, image_addresses, false, command_offset + 8, prepare_address,
            8));
      }
    }
  }
  buffer->program->scratch_wave_byte_length =
      iree_max(buffer->program->scratch_wave_byte_length,
               dispatch.scratch_wave_byte_length);
  ++buffer->program->dispatch_count;
  ++buffer->program->operation_count;
  return iree_ok_status();
}
static iree_status_t iree_hal_amd_gpu_pm4_command_buffer_dispatch(
    iree_hal_command_buffer_t* base, iree_hal_executable_t* executable,
    iree_hal_executable_function_t token, iree_hal_dispatch_config_t config,
    iree_const_byte_span_t constants, iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags) {
  iree_hal_amd_gpu_pm4_command_buffer_t* buffer = (void*)base;
  IREE_RETURN_IF_ERROR(iree_status_clone(buffer->recording_status));
  return iree_hal_amd_gpu_pm4_command_record_status(
      buffer, iree_hal_amd_gpu_pm4_command_dispatch(
                  base, executable, token, config, constants, bindings, flags));
}

iree_status_t iree_hal_amd_gpu_pm4_command_invocation_initialize(
    iree_hal_command_buffer_t* base, iree_hal_buffer_binding_table_t bindings,
    void* storage, iree_hal_queue_execute_flags_t flags,
    iree_hal_amd_gpu_pm4_command_invocation_t* out_invocation) {
  iree_hal_amd_gpu_pm4_command_buffer_t* buffer = (void*)base;
  *out_invocation = (iree_hal_amd_gpu_pm4_command_invocation_t){
      .command_buffer = base,
      .program = buffer->program,
      .bindings = storage,
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

void iree_hal_amd_gpu_pm4_command_invocation_deinitialize(
    iree_hal_amd_gpu_pm4_command_invocation_t* invocation) {
  if (invocation->instance) {
    if (invocation->published) {
      iree_hal_amd_gpu_program_release(&invocation->program->resident,
                                       invocation->instance);
    } else {
      iree_hal_amd_gpu_program_cancel(&invocation->program->resident,
                                      invocation->instance);
    }
  }
  if (!invocation->borrow_bindings) {
    for (uint32_t i = 0; i < invocation->command_buffer->binding_count; ++i) {
      iree_hal_buffer_release(invocation->bindings[i].buffer);
    }
  }
  iree_hal_command_buffer_release(invocation->command_buffer);
}

iree_status_t iree_hal_amd_gpu_pm4_command_invocation_write(
    const iree_hal_amd_gpu_pm4_command_invocation_t* invocation,
    uint32_t queue_ordinal, uint64_t dispatch_id, uint64_t continuation,
    uint64_t profile_address, void* upload) {
  const iree_hal_amd_gpu_pm4_command_buffer_t* buffer =
      (const void*)invocation->command_buffer;
  uint64_t* values =
      (void*)((iree_hal_amd_gpu_pm4_program_arguments_t*)upload + 1);
  values[IREE_HAL_AMD_GPU_PM4_VALUE_INSTANCE] = invocation->instance->address;
  values[IREE_HAL_AMD_GPU_PM4_VALUE_DISPATCH_ID] = dispatch_id;
  values[IREE_HAL_AMD_GPU_PM4_VALUE_CONTINUATION] = continuation;
  if (buffer->program->profile) {
    values[IREE_HAL_AMD_GPU_PM4_VALUE_PROFILE_ADDRESS] = profile_address;
    // Each disabled eight-word NOP consumes the compute join and clock read.
    values[IREE_HAL_AMD_GPU_PM4_VALUE_PROFILE_HEADER] =
        profile_address ? UINT32_C(0xc0004600) : UINT32_C(0xc0061000);
  }
  return iree_hal_amd_gpu_command_inputs_write(
      &buffer->inputs, invocation->bindings, queue_ordinal,
      values + iree_hal_amd_gpu_pm4_command_first_input(buffer));
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
