// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/aql/command_buffer.h"

#include "iree/base/internal/arena.h"
#include "iree/hal/drivers/amd/gpu/executable.h"
#include "iree/hal/drivers/amd/gpu/queue/aql/barrier.h"
#include "iree/hal/drivers/amd/gpu/queue/aql/transfer.h"
#include "iree/hal/drivers/amd/gpu/queue/recording.h"
#include "iree/hal/utils/resource_set.h"

typedef struct iree_hal_amd_gpu_aql_command_buffer_t {
  // Public identity and external recording validation.
  iree_hal_command_buffer_t base;
  // Allocator for this object and recording-only metadata.
  iree_allocator_t host_allocator;
  // Device-owned placement and immutable descriptor kernels.
  iree_hal_amd_gpu_aql_program_pool_t* pool;
  // Immutable native execution facts after sealing.
  iree_hal_amd_gpu_aql_command_program_t program;
  // Append-only native bytes/private data/fixups, released after sealing.
  iree_hal_amd_gpu_recording_t recording;
  // Dense unique invocation values and aggregated binding requirements.
  iree_hal_amd_gpu_command_inputs_t inputs;
  // Allocated host block descriptor capacity, grown only while recording.
  uint32_t block_capacity;
  // Maximum packets per block from the family's native ring geometry.
  uint32_t maximum_packet_count;
  // Physical LDS capacity used at the public dispatch boundary.
  uint32_t maximum_group_segment_byte_length;
  // Adjacent explicit execution/cache barriers, combined before emission.
  uint32_t pending_barrier;
  // Last payload's complete header in stable host recording storage. Indirect
  // dispatches name their patch arguments' desired header, never INVALID.
  uint32_t* last_header;
  // The last payload begins after all earlier payloads have completed, so its
  // release fence can cover their writes without an additional native packet.
  bool last_ordered;
  // First failed mutation, preventing partial native programs from sealing.
  iree_status_t recording_status;
  // Blocks backing optional direct-resource retention.
  iree_arena_block_pool_t resource_blocks;
  // Retained direct resources, or NULL for UNRETAINED recordings.
  iree_hal_resource_set_t* resources;
} iree_hal_amd_gpu_aql_command_buffer_t;

static const iree_hal_command_buffer_vtable_t
    iree_hal_amd_gpu_aql_command_buffer_vtable;

static void iree_hal_amd_gpu_aql_command_buffer_destroy(
    iree_hal_command_buffer_t* base) {
  iree_hal_amd_gpu_aql_command_buffer_t* buffer = (void*)base;
  iree_hal_device_t* device = base->queue_family->device;
  iree_status_free(buffer->recording_status);
  iree_hal_amd_gpu_recording_deinitialize(&buffer->recording);
  iree_hal_amd_gpu_profile_commands_release(buffer->program.profile);
  iree_hal_amd_gpu_command_inputs_deinitialize(&buffer->inputs);
  iree_hal_resource_set_free(buffer->resources);
  iree_arena_block_pool_deinitialize(&buffer->resource_blocks);
  iree_hal_amd_gpu_program_deinitialize(&buffer->program.resident);
  iree_allocator_free(buffer->host_allocator, buffer->program.blocks);
  iree_allocator_free(buffer->host_allocator, buffer);
  iree_hal_device_release(device);
}

iree_status_t iree_hal_amd_gpu_aql_command_buffer_create(
    const iree_hal_queue_family_t* family,
    iree_hal_amd_gpu_aql_program_pool_t* pool,
    iree_hal_command_buffer_mode_t mode, iree_hal_command_category_t categories,
    iree_host_size_t binding_capacity, uint32_t maximum_packet_count,
    uint32_t maximum_group_segment_byte_length, iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_buffer) {
  *out_buffer = NULL;
  if (binding_capacity >= UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "resident binding table exceeds its index space");
  }
  iree_host_size_t size = 0, requirements_offset = 0, validation_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_amd_gpu_aql_command_buffer_t), &size,
      IREE_STRUCT_FIELD(binding_capacity + 1,
                        iree_hal_amd_gpu_binding_requirement_t,
                        &requirements_offset),
      IREE_STRUCT_FIELD(
          iree_hal_command_buffer_validation_state_size(mode, binding_capacity),
          uint8_t, &validation_offset)));
  iree_hal_amd_gpu_aql_command_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, size, (void**)&buffer));
  iree_hal_command_buffer_initialize(
      family, mode, categories, binding_capacity,
      (uint8_t*)buffer + validation_offset,
      &iree_hal_amd_gpu_aql_command_buffer_vtable, &buffer->base);
  iree_hal_device_retain(family->device);
  buffer->host_allocator = host_allocator;
  buffer->pool = pool;
  buffer->maximum_packet_count = maximum_packet_count;
  buffer->maximum_group_segment_byte_length = maximum_group_segment_byte_length;
  iree_hal_amd_gpu_recording_initialize(host_allocator, &buffer->recording);
  iree_hal_amd_gpu_command_inputs_initialize(
      &buffer->base, 1, NULL, (void*)((uint8_t*)buffer + requirements_offset),
      host_allocator, &buffer->inputs);
  iree_arena_block_pool_initialize(4096, host_allocator,
                                   &buffer->resource_blocks);
  iree_status_t status = iree_ok_status();
  if (iree_any_bit_set(
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
    status = iree_hal_amd_gpu_profile_commands_create(&record, host_allocator,
                                                      &buffer->program.profile);
  }
  if (iree_status_is_ok(status) &&
      !(mode & IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED)) {
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

iree_hal_amd_gpu_aql_command_program_t*
iree_hal_amd_gpu_aql_command_buffer_program(iree_hal_command_buffer_t* base) {
  return &((iree_hal_amd_gpu_aql_command_buffer_t*)base)->program;
}

// Appends complete native packets without splitting an indirect pair. The
// frozen block table is the only packet-layout metadata read at submission.
static iree_status_t iree_hal_amd_gpu_aql_command_packets(
    iree_hal_amd_gpu_aql_command_buffer_t* buffer, uint32_t count,
    uint32_t* out_offset, iree_hal_amd_gpu_aql_packet_t** out_packets) {
  iree_hal_amd_gpu_aql_command_program_t* program = &buffer->program;
  if (!program->block_count ||
      program->blocks[program->block_count - 1].packet_count + count >
          buffer->maximum_packet_count) {
    if (program->block_count == buffer->block_capacity) {
      const uint32_t capacity = iree_max(buffer->block_capacity * 2, 4u);
      IREE_RETURN_IF_ERROR(iree_allocator_realloc(
          buffer->host_allocator, capacity * sizeof(*program->blocks),
          (void**)&program->blocks));
      buffer->block_capacity = capacity;
    }
    program->blocks[program->block_count++] =
        (iree_hal_amd_gpu_aql_program_block_t){
            .offset = buffer->recording.commands.length,
            .dispatches = {.first = program->dispatch_count},
        };
  }
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_recording_commands(
      &buffer->recording, count * 64, 64, out_offset, (void**)out_packets));
  program->blocks[program->block_count - 1].packet_count += count;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_aql_command_flush_barrier(
    iree_hal_amd_gpu_aql_command_buffer_t* buffer) {
  if (!buffer->pending_barrier) {
    return iree_ok_status();
  }
  uint32_t offset = 0;
  iree_hal_amd_gpu_aql_packet_t* packet = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_gpu_aql_command_packets(buffer, 1, &offset, &packet));
  *packet = iree_hal_amd_gpu_aql_packet_barrier(
      IREE_HAL_AMD_GPU_AQL_HEADER_BARRIER_AND | buffer->pending_barrier, 0,
      NULL, 0);
  buffer->pending_barrier = 0;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_aql_command_record_status(
    iree_hal_amd_gpu_aql_command_buffer_t* buffer, iree_status_t status) {
  if (!iree_status_is_ok(status) &&
      iree_status_is_ok(buffer->recording_status)) {
    buffer->recording_status = iree_status_clone(status);
  }
  return status;
}

static iree_status_t iree_hal_amd_gpu_aql_command_buffer_begin(
    iree_hal_command_buffer_t* base) {
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_aql_command_end(
    iree_hal_amd_gpu_aql_command_buffer_t* buffer) {
  // Each native chunk ends with an ordered completion packet. A trailing
  // execution-only dependency needs no second packet before that boundary.
  if (buffer->pending_barrier == IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED) {
    buffer->pending_barrier = 0;
  }
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_command_flush_barrier(buffer));
  const uint64_t length = iree_align_uint64(buffer->recording.commands.length,
                                            buffer->recording.alignment) +
                          buffer->recording.data.length;
  if (length > UINT32_MAX - 15) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "resident command/data placement exceeds 4 GiB");
  }
  buffer->program.value_count = 1 + buffer->inputs.count;
  if (length) {
    iree_hal_amd_gpu_program_t* program = &buffer->program.resident;
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_program_initialize(
        &buffer->pool->storage, buffer->recording.alignment, (uint32_t)length,
        iree_hal_amd_gpu_recording_fixup_count(&buffer->recording), program));
    iree_hal_amd_gpu_recording_write(
        &buffer->recording, program->contents.data,
        (void*)(program->contents.data + program->fixup_offset));
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_program_seal(program));
  }
  iree_hal_amd_gpu_command_inputs_seal(&buffer->inputs);
  if (buffer->resources) {
    iree_hal_resource_set_freeze(buffer->resources);
  }
  iree_hal_amd_gpu_recording_deinitialize(&buffer->recording);
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_aql_command_buffer_end(
    iree_hal_command_buffer_t* base) {
  iree_hal_amd_gpu_aql_command_buffer_t* buffer = (void*)base;
  IREE_RETURN_IF_ERROR(iree_status_clone(buffer->recording_status));
  return iree_hal_amd_gpu_aql_command_record_status(
      buffer, iree_hal_amd_gpu_aql_command_end(buffer));
}

static iree_status_t iree_hal_amd_gpu_aql_command_buffer_barrier(
    iree_hal_command_buffer_t* base, const iree_hal_barrier_t* barrier) {
  iree_hal_amd_gpu_aql_command_buffer_t* buffer = (void*)base;
  const iree_hal_barrier_list_t list = {.count = 1, .values = barrier};
  uint32_t prepared = iree_hal_amd_gpu_aql_barrier_prepare(&list);
  const uint32_t release_mask = 3u << 11;
  if (buffer->last_ordered) {
    // This is a host template, not the hardware ring. A dominating payload
    // releases all prior writes; unordered cohorts keep a standalone barrier.
    const uint32_t previous = *buffer->last_header;
    *buffer->last_header =
        (previous & ~release_mask) |
        iree_max(previous & release_mask, prepared & release_mask);
    prepared &= ~release_mask;
  }
  buffer->pending_barrier =
      iree_hal_amd_gpu_aql_barrier_join(buffer->pending_barrier, prepared);
  ++buffer->program.operation_count;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_aql_command_pointer(
    iree_hal_amd_gpu_aql_command_buffer_t* buffer, uint32_t offset,
    uint32_t value, uint64_t addend,
    iree_hal_amd_gpu_recording_fixup_flags_t flags) {
  return iree_hal_amd_gpu_recording_fixup(
      &buffer->recording, offset, value, addend, 0,
      flags | IREE_HAL_AMD_GPU_RECORDING_FIXUP_WIDE);
}

// A producer release that could not fold into a dominating payload needs its
// own barrier. Other effects become part of the next complete payload header.
static iree_status_t iree_hal_amd_gpu_aql_command_header(
    iree_hal_amd_gpu_aql_command_buffer_t* buffer, uint32_t* out_header) {
  uint32_t header = buffer->pending_barrier;
  if (header & (3u << 11)) {
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_command_flush_barrier(buffer));
    header = IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED;
  }
  buffer->pending_barrier = 0;
  *out_header = header;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_aql_command_dispatch(
    iree_hal_amd_gpu_aql_command_buffer_t* buffer,
    iree_hal_executable_t* executable, iree_hal_executable_function_t token,
    iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  const iree_hal_amd_gpu_aql_function_t* function = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_executable_lookup_aql_for_queue(
      executable, token, 0, &function));
  iree_hal_amd_gpu_dispatch_geometry_t geometry;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_dispatch_prepare(
      function, buffer->maximum_group_segment_byte_length, config, constants,
      bindings.count, flags, &geometry));
  const bool indirect = iree_hal_dispatch_uses_indirect_parameters(flags);
  const bool external_arguments =
      iree_hal_dispatch_uses_indirect_arguments(flags);
  if (!indirect &&
      (!geometry.workgroup_count[0] || !geometry.workgroup_count[1] ||
       !geometry.workgroup_count[2])) {
    return iree_ok_status();
  }
  if (buffer->resources) {
    IREE_RETURN_IF_ERROR(
        iree_hal_resource_set_insert(buffer->resources, 1, &executable));
  }
  uint32_t header = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_command_header(buffer, &header));
  if (indirect) {
    header = iree_hal_amd_gpu_aql_barrier_join(
        header, IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_SYSTEM);
  }
  const iree_hal_amd_gpu_kernarg_layout_t* layout = function->base.kernarg;
  uint32_t argument_offset = 0;
  if (!external_arguments && layout->byte_length) {
    void* arguments = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_recording_data(
        &buffer->recording, layout->byte_length, layout->alignment,
        &argument_offset, &arguments));
    iree_hal_amd_gpu_dispatch_emplace_arguments(
        &function->base, &geometry, constants, NULL, flags, arguments);
  }
  uint32_t argument_value = 0;
  uint64_t argument_addend = argument_offset;
  if (external_arguments) {
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_command_inputs_add_binding(
        &buffer->inputs, bindings.values[0], 1, layout->byte_length,
        layout->alignment, false, &argument_value));
    argument_addend = bindings.values[0].offset;
  } else {
    const uint16_t* offsets =
        iree_hal_amd_gpu_kernarg_layout_binding_offsets(layout);
    iree_status_t status = iree_ok_status();
    for (iree_host_size_t i = 0;
         i < bindings.count && iree_status_is_ok(status); ++i) {
      uint32_t value = 0;
      status = iree_hal_amd_gpu_command_inputs_add_binding(
          &buffer->inputs, bindings.values[i], 1, 0, 1, false, &value);
      if (iree_status_is_ok(status)) {
        status = iree_hal_amd_gpu_aql_command_pointer(
            buffer, argument_offset + offsets[i], value,
            bindings.values[i].offset,
            IREE_HAL_AMD_GPU_RECORDING_FIXUP_TARGET_DATA);
      }
    }
    IREE_RETURN_IF_ERROR(status);
  }
  uint32_t offset = 0;
  iree_hal_amd_gpu_aql_packet_t* packets = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_command_packets(
      buffer, indirect ? 2 : 1, &offset, &packets));
  const uint32_t target_offset = offset + (indirect ? 64 : 0);
  iree_hal_amd_gpu_aql_packet_t* target = packets + indirect;
  *target = iree_hal_amd_gpu_aql_dispatch_packet(function, &geometry, header, 0,
                                                 0, flags);
  // An indirect producer is itself ordered before publishing its target.
  buffer->last_ordered =
      indirect || !buffer->last_header ||
      (target->words[0] & IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED) != 0;
  buffer->last_header = &target->words[0];
  const iree_hal_amd_gpu_recording_fixup_flags_t argument_flags =
      external_arguments ? 0 : IREE_HAL_AMD_GPU_RECORDING_FIXUP_ADDEND_DATA;
  if (layout->byte_length) {
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_command_pointer(
        buffer, target_offset + 40, argument_value, argument_addend,
        argument_flags));
  }
  const uint64_t* images =
      iree_hal_amd_gpu_executable_queue_image_addresses(executable);
  if (images) {
    uint32_t value = 0;
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_command_inputs_add_image(
        &buffer->inputs, images, &value));
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_command_pointer(
        buffer, target_offset + 32, value, function->kernel_object - images[0],
        0));
  }
  if (indirect) {
    uint32_t counts_value = 0;
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_command_inputs_add_binding(
        &buffer->inputs, config.workgroup_count_ref, 1, 12, 4, false,
        &counts_value));
    uint32_t patch_offset = 0;
    iree_hal_amd_gpu_aql_indirect_arguments_t* patch = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_recording_data(
        &buffer->recording, sizeof(*patch), 8, &patch_offset, (void**)&patch));
    iree_hal_amd_gpu_aql_indirect_prepare(layout, &geometry, flags, 0, 0, 0,
                                          target->words[0], patch);
    buffer->last_header = &patch->header;
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_command_pointer(
        buffer, patch_offset + 8, counts_value,
        config.workgroup_count_ref.offset,
        IREE_HAL_AMD_GPU_RECORDING_FIXUP_TARGET_DATA));
    if (!external_arguments && layout->indirect_copy_count) {
      IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_command_pointer(
          buffer, patch_offset + 16, argument_value, argument_addend,
          IREE_HAL_AMD_GPU_RECORDING_FIXUP_TARGET_DATA | argument_flags));
    }
    packets[0] =
        iree_hal_amd_gpu_aql_indirect_packet(buffer->pool->indirect, 0);
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_command_pointer(
        buffer, offset + 40, 0, patch_offset,
        IREE_HAL_AMD_GPU_RECORDING_FIXUP_ADDEND_DATA));
    target->words[0] = IREE_HAL_AMD_GPU_AQL_HEADER_INVALID;
  }
  iree_hal_amd_gpu_aql_program_block_t* block =
      &buffer->program.blocks[buffer->program.block_count - 1];
  if (buffer->program.profile) {
    iree_hal_amd_gpu_profile_dispatch_t capture;
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_executable_profile_dispatch(
        executable, token, &geometry, flags, &capture));
    capture.event.flags |= IREE_HAL_PROFILE_DISPATCH_EVENT_FLAG_COMMAND_BUFFER;
    capture.event.command_buffer_id =
        iree_hal_command_buffer_profile_id(&buffer->base);
    capture.event.command_index = buffer->program.operation_count;
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_profile_commands_append(
        buffer->program.profile, capture));
    // Private template operand, replaced before a native header is published.
    target->words[14] = block->dispatches.count + 1;
  }
  ++block->dispatches.count;
  ++buffer->program.dispatch_count;
  ++buffer->program.operation_count;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_aql_command_buffer_dispatch(
    iree_hal_command_buffer_t* base, iree_hal_executable_t* executable,
    iree_hal_executable_function_t token, iree_hal_dispatch_config_t config,
    iree_const_byte_span_t constants, iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags) {
  iree_hal_amd_gpu_aql_command_buffer_t* buffer = (void*)base;
  IREE_RETURN_IF_ERROR(iree_status_clone(buffer->recording_status));
  return iree_hal_amd_gpu_aql_command_record_status(
      buffer,
      iree_hal_amd_gpu_aql_command_dispatch(buffer, executable, token, config,
                                            constants, bindings, flags));
}

static iree_status_t iree_hal_amd_gpu_aql_command_transfer(
    iree_hal_amd_gpu_aql_command_buffer_t* buffer,
    iree_hal_transfer_operation_type_t type, iree_hal_buffer_ref_t source,
    iree_hal_buffer_ref_t target, const void* source_bytes, uint32_t pattern) {
  const bool copy = type == IREE_HAL_TRANSFER_OPERATION_TYPE_COPY;
  const bool update = type == IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE;
  const bool whole = target.length == IREE_HAL_WHOLE_BUFFER;
  if (copy && whole) {
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_command_inputs_equal_lengths(
        &buffer->inputs, source, target));
  }
  const uint64_t length =
      whole && target.buffer
          ? iree_hal_buffer_byte_length(target.buffer) - target.offset
          : target.length;
  if (!length) {
    return iree_ok_status();
  }
  uint32_t header = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_command_header(buffer, &header));
  uint32_t target_value = 0, source_value = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_command_inputs_add_binding(
      &buffer->inputs, target, 1, 0, 1, whole, &target_value));
  if (copy) {
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_command_inputs_add_binding(
        &buffer->inputs, source, 1, 0, 1, whole, &source_value));
  }
  uint32_t update_offset = 0;
  if (update) {
    void* bytes = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_recording_data(
        &buffer->recording, (uint32_t)length, 1, &update_offset, &bytes));
    memcpy(bytes, source_bytes, (iree_host_size_t)length);
  }
  uint32_t argument_offset = 0;
  iree_hal_amd_gpu_blit_arguments_t* arguments = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_gpu_recording_data(&buffer->recording, sizeof(*arguments), 8,
                                      &argument_offset, (void**)&arguments));
  *arguments = (iree_hal_amd_gpu_blit_arguments_t){
      .length = length == IREE_HAL_WHOLE_BUFFER ? 0 : length,
      .pattern = pattern,
      .fill = !copy && !update};
  if (copy || update) {
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_command_pointer(
        buffer, argument_offset, source_value,
        update ? update_offset : source.offset,
        IREE_HAL_AMD_GPU_RECORDING_FIXUP_TARGET_DATA |
            (update ? IREE_HAL_AMD_GPU_RECORDING_FIXUP_ADDEND_DATA : 0)));
  }
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_command_pointer(
      buffer, argument_offset + 8, target_value, target.offset,
      IREE_HAL_AMD_GPU_RECORDING_FIXUP_TARGET_DATA));
  if (length == IREE_HAL_WHOLE_BUFFER) {
    uint32_t value = 0;
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_command_inputs_add_length(
        &buffer->inputs, target.buffer_slot, &value));
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_command_pointer(
        buffer, argument_offset + 16, value, 0 - target.offset,
        IREE_HAL_AMD_GPU_RECORDING_FIXUP_TARGET_DATA));
  }
  uint32_t offset = 0;
  iree_hal_amd_gpu_aql_packet_t* packet = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_gpu_aql_command_packets(buffer, 1, &offset, &packet));
  *packet = iree_hal_amd_gpu_aql_transfer_packet(buffer->pool->blit, 0, length,
                                                 header);
  buffer->last_ordered = !buffer->last_header ||
                         (header & IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED) != 0;
  buffer->last_header = &packet->words[0];
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_command_pointer(
      buffer, offset + 40, 0, argument_offset,
      IREE_HAL_AMD_GPU_RECORDING_FIXUP_ADDEND_DATA));
  ++buffer->program.operation_count;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_aql_command_buffer_transfer(
    iree_hal_amd_gpu_aql_command_buffer_t* buffer,
    iree_hal_transfer_operation_type_t type, iree_hal_buffer_ref_t source,
    iree_hal_buffer_ref_t target, const void* source_bytes, uint32_t pattern) {
  IREE_RETURN_IF_ERROR(iree_status_clone(buffer->recording_status));
  return iree_hal_amd_gpu_aql_command_record_status(
      buffer, iree_hal_amd_gpu_aql_command_transfer(
                  buffer, type, source, target, source_bytes, pattern));
}

static iree_status_t iree_hal_amd_gpu_aql_command_buffer_fill_buffer(
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
  return iree_hal_amd_gpu_aql_command_buffer_transfer(
      (void*)base, IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
      (iree_hal_buffer_ref_t){0}, target, NULL, value);
}

static iree_status_t iree_hal_amd_gpu_aql_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base, const void* source,
    iree_host_size_t source_offset, iree_hal_buffer_ref_t target,
    iree_hal_update_flags_t flags) {
  return iree_hal_amd_gpu_aql_command_buffer_transfer(
      (void*)base, IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE,
      (iree_hal_buffer_ref_t){0}, target,
      (const uint8_t*)source + source_offset, 0);
}

static iree_status_t iree_hal_amd_gpu_aql_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base, iree_hal_buffer_ref_t source,
    iree_hal_buffer_ref_t target, iree_hal_copy_flags_t flags) {
  return iree_hal_amd_gpu_aql_command_buffer_transfer(
      (void*)base, IREE_HAL_TRANSFER_OPERATION_TYPE_COPY, source, target, NULL,
      0);
}

iree_status_t iree_hal_amd_gpu_aql_command_invocation_initialize(
    iree_hal_command_buffer_t* base, iree_hal_buffer_binding_table_t bindings,
    void* storage, iree_hal_queue_execute_flags_t flags,
    iree_hal_amd_gpu_aql_command_invocation_t* out_invocation) {
  iree_hal_amd_gpu_aql_command_buffer_t* buffer = (void*)base;
  *out_invocation = (iree_hal_amd_gpu_aql_command_invocation_t){
      .command_buffer = base,
      .program = &buffer->program,
      .bindings = storage,
      .borrow_bindings =
          (flags & IREE_HAL_QUEUE_EXECUTE_FLAG_BORROW_BINDING_TABLE_LIFETIME) !=
          0};
  iree_atomic_ref_count_init(&out_invocation->ref_count);
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

void iree_hal_amd_gpu_aql_command_invocation_deinitialize(
    iree_hal_amd_gpu_aql_command_invocation_t* invocation) {
  if (!invocation->command_buffer) {
    return;
  }
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
  invocation->command_buffer = NULL;
}

iree_status_t iree_hal_amd_gpu_aql_command_invocation_write(
    const iree_hal_amd_gpu_aql_command_invocation_t* invocation,
    uint32_t queue_ordinal, uint64_t* values) {
  const iree_hal_amd_gpu_aql_command_buffer_t* buffer =
      (const void*)invocation->command_buffer;
  values[0] = invocation->instance->address;
  return iree_hal_amd_gpu_command_inputs_write(
      &buffer->inputs, invocation->bindings, queue_ordinal, values + 1);
}

static iree_status_t iree_hal_amd_gpu_aql_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t* base, iree_string_view_t label,
    iree_hal_label_color_t color, const iree_hal_label_location_t* location) {
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_aql_command_buffer_end_debug_group(
    iree_hal_command_buffer_t* base) {
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_aql_command_buffer_atomic_wait(
    iree_hal_command_buffer_t* base, iree_hal_execution_stage_t source_stage,
    iree_hal_execution_stage_t target_stage, iree_hal_buffer_ref_t target,
    iree_hal_atomic_wait_params_t params) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AQL recorded atomic waits are not admitted");
}

static iree_status_t iree_hal_amd_gpu_aql_command_buffer_atomic_store(
    iree_hal_command_buffer_t* base, iree_hal_execution_stage_t source_stage,
    iree_hal_execution_stage_t target_stage, iree_hal_buffer_ref_t target,
    iree_hal_atomic_store_params_t params) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AQL recorded atomic stores are not admitted");
}

static iree_status_t iree_hal_amd_gpu_aql_command_buffer_atomic_rmw(
    iree_hal_command_buffer_t* base, iree_hal_execution_stage_t source_stage,
    iree_hal_execution_stage_t target_stage, iree_hal_buffer_ref_t target,
    iree_hal_atomic_rmw_params_t params) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AQL recorded atomic RMWs are not admitted");
}

static iree_status_t iree_hal_amd_gpu_aql_command_buffer_advise_buffer(
    iree_hal_command_buffer_t* base, iree_hal_buffer_ref_t reference,
    iree_hal_memory_advise_flags_t flags, uint64_t arg0, uint64_t arg1) {
  return iree_ok_status();
}

static const iree_hal_command_buffer_vtable_t
    iree_hal_amd_gpu_aql_command_buffer_vtable = {
        .destroy = iree_hal_amd_gpu_aql_command_buffer_destroy,
        .begin = iree_hal_amd_gpu_aql_command_buffer_begin,
        .end = iree_hal_amd_gpu_aql_command_buffer_end,
        .begin_debug_group =
            iree_hal_amd_gpu_aql_command_buffer_begin_debug_group,
        .end_debug_group = iree_hal_amd_gpu_aql_command_buffer_end_debug_group,
        .barrier = iree_hal_amd_gpu_aql_command_buffer_barrier,
        .atomic_wait = iree_hal_amd_gpu_aql_command_buffer_atomic_wait,
        .atomic_store = iree_hal_amd_gpu_aql_command_buffer_atomic_store,
        .atomic_rmw = iree_hal_amd_gpu_aql_command_buffer_atomic_rmw,
        .advise_buffer = iree_hal_amd_gpu_aql_command_buffer_advise_buffer,
        .fill_buffer = iree_hal_amd_gpu_aql_command_buffer_fill_buffer,
        .update_buffer = iree_hal_amd_gpu_aql_command_buffer_update_buffer,
        .copy_buffer = iree_hal_amd_gpu_aql_command_buffer_copy_buffer,
        .dispatch = iree_hal_amd_gpu_aql_command_buffer_dispatch,
};
