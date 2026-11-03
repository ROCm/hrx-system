// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/hal_command_buffer.h"

#include "iree/base/alignment.h"
#include "loom/target/arch/cmd/format.h"

// The portable vocabulary has its own format contract. Keep the direct HAL
// projection checked here instead of translating every bit while recording.
#define LOOM_CMD_CHECK_STAGE(name)                             \
  static_assert((uint32_t)LOOM_CMD_PROGRAM_STAGE_##name ==     \
                    (uint32_t)IREE_HAL_EXECUTION_STAGE_##name, \
                "portable and HAL execution stages agree")
LOOM_CMD_CHECK_STAGE(COMMAND_ISSUE);
LOOM_CMD_CHECK_STAGE(COMMAND_PROCESS);
LOOM_CMD_CHECK_STAGE(DISPATCH);
LOOM_CMD_CHECK_STAGE(TRANSFER);
LOOM_CMD_CHECK_STAGE(COMMAND_RETIRE);
LOOM_CMD_CHECK_STAGE(HOST);
LOOM_CMD_CHECK_STAGE(ATOMIC);
#undef LOOM_CMD_CHECK_STAGE
#define LOOM_CMD_CHECK_ACCESS(name)                         \
  static_assert((uint32_t)LOOM_CMD_PROGRAM_ACCESS_##name == \
                    (uint32_t)IREE_HAL_ACCESS_SCOPE_##name, \
                "portable and HAL access scopes agree")
LOOM_CMD_CHECK_ACCESS(INDIRECT_COMMAND_READ);
LOOM_CMD_CHECK_ACCESS(CONSTANT_READ);
LOOM_CMD_CHECK_ACCESS(DISPATCH_READ);
LOOM_CMD_CHECK_ACCESS(DISPATCH_WRITE);
LOOM_CMD_CHECK_ACCESS(TRANSFER_READ);
LOOM_CMD_CHECK_ACCESS(TRANSFER_WRITE);
LOOM_CMD_CHECK_ACCESS(HOST_READ);
LOOM_CMD_CHECK_ACCESS(HOST_WRITE);
LOOM_CMD_CHECK_ACCESS(MEMORY_READ);
LOOM_CMD_CHECK_ACCESS(MEMORY_WRITE);
LOOM_CMD_CHECK_ACCESS(ATOMIC_READ);
LOOM_CMD_CHECK_ACCESS(ATOMIC_WRITE);
#undef LOOM_CMD_CHECK_ACCESS
static_assert((uint64_t)LOOM_CMD_PROGRAM_BARRIER_FLAG_ACQUIRE_SYSTEM ==
                  (uint64_t)IREE_HAL_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE,
              "portable and HAL acquire flags agree");
static_assert((uint64_t)LOOM_CMD_PROGRAM_BARRIER_FLAG_RELEASE_SYSTEM ==
                  (uint64_t)IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
              "portable and HAL release flags agree");

// Reused for every explicit dependency boundary during cold-path recording.
typedef struct loom_cmd_program_hal_barriers_t {
  // Owned allocation, with capacity for the largest dependency slice.
  iree_hal_buffer_barrier_t* buffers;
  // Global dependencies in the same allocation, following the buffer array.
  iree_hal_memory_barrier_t* memory;
} loom_cmd_program_hal_barriers_t;

// Cold-path scratch for translating one entry's logical arguments to HAL slots.
typedef struct loom_cmd_program_hal_layout_t {
  // Reflected logical parameter and physical argument counts.
  iree_hal_executable_function_info_t info;
  // Owned positional parameter descriptors.
  iree_hal_executable_function_parameter_t* parameters;
  // Owned reusable recording scratch, not issue-time argument storage.
  iree_hal_buffer_ref_t* bindings;
  // Owned reusable recording scratch including zeroed ABI padding.
  uint8_t* constants;
} loom_cmd_program_hal_layout_t;

static iree_status_t loom_cmd_program_hal_buffer_ref(
    loom_cmd_program_buffer_ref_t ref,
    const iree_hal_buffer_binding_t* fixed_bindings,
    iree_hal_buffer_ref_t* out_ref) {
  if (ref.role != LOOM_CMD_PROGRAM_BUFFER_ROLE_FIXED) {
    *out_ref = iree_hal_make_indirect_buffer_ref(
        ref.root_index, ref.byte_offset, ref.byte_length);
    return iree_ok_status();
  }

  const iree_hal_buffer_binding_t binding = fixed_bindings[ref.root_index];
  if (!binding.buffer) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "fixed command root requires a buffer");
  }
  const bool is_whole_buffer = binding.length == IREE_HAL_WHOLE_BUFFER;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_range(
      binding.buffer, binding.offset, is_whole_buffer ? 0 : binding.length));
  const iree_device_size_t binding_length =
      is_whole_buffer
          ? iree_hal_buffer_byte_length(binding.buffer) - binding.offset
          : binding.length;
  if (ref.byte_offset > binding_length ||
      (ref.byte_length != IREE_HAL_WHOLE_BUFFER &&
       ref.byte_length > binding_length - ref.byte_offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command range [%" PRIu64 ", %" PRIu64
                            ") exceeds fixed root length %" PRIu64,
                            ref.byte_offset,
                            ref.byte_length == IREE_HAL_WHOLE_BUFFER
                                ? (uint64_t)binding_length
                                : ref.byte_offset + ref.byte_length,
                            (uint64_t)binding_length);
  }
  iree_device_size_t resolved_offset = 0;
  if (!iree_device_size_checked_add(binding.offset, ref.byte_offset,
                                    &resolved_offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "fixed command root offset overflows device size");
  }
  const iree_device_size_t resolved_length =
      ref.byte_length == IREE_HAL_WHOLE_BUFFER
          ? binding_length - ref.byte_offset
          : ref.byte_length;
  *out_ref = iree_hal_make_buffer_ref(binding.buffer, resolved_offset,
                                      resolved_length);
  return iree_ok_status();
}

static loom_cmd_program_buffer_ref_t loom_cmd_program_hal_read_buffer_ref(
    const uint8_t* data) {
  return (loom_cmd_program_buffer_ref_t){
      .role = iree_unaligned_load_le_u32(
          data + LOOM_CMD_PROGRAM_BUFFER_REF_ROLE_OFFSET),
      .root_index = iree_unaligned_load_le_u32(
          data + LOOM_CMD_PROGRAM_BUFFER_REF_ROOT_INDEX_OFFSET),
      .byte_offset = iree_unaligned_load_le_u64(
          data + LOOM_CMD_PROGRAM_BUFFER_REF_BYTE_OFFSET_OFFSET),
      .byte_length = iree_unaligned_load_le_u64(
          data + LOOM_CMD_PROGRAM_BUFFER_REF_BYTE_LENGTH_OFFSET),
  };
}

static iree_status_t loom_cmd_program_hal_pack_scalar(
    loom_cmd_program_argument_kind_t kind, const uint8_t** cursor,
    const iree_hal_executable_function_parameter_t* parameter,
    uint8_t* constants) {
  if (parameter->type != IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "scalar command argument requires a HAL constant");
  }
  const uint32_t size = parameter->size;
  const bool is_index = kind == LOOM_CMD_PROGRAM_ARGUMENT_KIND_INDEX;
  const bool is_offset = kind == LOOM_CMD_PROGRAM_ARGUMENT_KIND_OFFSET;
  if (!is_index && !is_offset) {
    const uint32_t expected_size =
        1u << (kind - LOOM_CMD_PROGRAM_ARGUMENT_KIND_B8);
    if (size != expected_size) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "scalar width mismatch: command %u, kernel %u",
                              expected_size, size);
    }
    memcpy(constants + parameter->offset, *cursor, size);
    *cursor += size;
    return iree_ok_status();
  }
  if (size != 4 && size != 8) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "index/offset requires a 32- or 64-bit constant");
  }
  const uint64_t value = iree_unaligned_load_le_u64(*cursor);
  if (size == 4 && ((is_index && ((int64_t)value < INT32_MIN ||
                                  (int64_t)value > INT32_MAX)) ||
                    (is_offset && value > UINT32_MAX))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command index/offset does not fit kernel width");
  }
  if (size == 4) {
    iree_unaligned_store_le_u32(constants + parameter->offset, (uint32_t)value);
  } else {
    iree_unaligned_store_le_u64(constants + parameter->offset, value);
  }
  *cursor += sizeof(uint64_t);
  return iree_ok_status();
}

static iree_status_t loom_cmd_program_hal_record_dispatch(
    const loom_cmd_program_t* program,
    const loom_cmd_program_command_t* command,
    const iree_hal_buffer_binding_t* fixed_bindings,
    const loom_cmd_program_hal_entry_t* entries,
    loom_cmd_program_hal_layout_t* layouts,
    iree_hal_command_buffer_t* command_buffer) {
  const loom_cmd_program_command_kind_t kind =
      loom_cmd_program_command_kind_base(command->kind);
  const bool is_direct = kind == LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT;
  const uint32_t entry_index =
      is_direct ? command->payload.dispatch_direct.entry_index
                : command->payload.dispatch_indirect.entry_index;
  const loom_cmd_program_hal_entry_t* entry = &entries[entry_index];
  loom_cmd_program_hal_layout_t* layout = &layouts[entry_index];
  const loom_cmd_program_entry_schema_t schema =
      loom_cmd_program_entry_schema_at(program, command->argument_schema_index);
  if (schema.argument_count != layout->info.parameter_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command/kernel argument count mismatch: %u vs %u",
                            schema.argument_count,
                            layout->info.parameter_count);
  }
  const uint8_t* cursor =
      loom_cmd_program_command_argument_data(program, command).data;
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < schema.argument_count && iree_status_is_ok(status);
       ++i) {
    const loom_cmd_program_argument_kind_t argument_kind =
        loom_cmd_program_entry_schema_kind_at(program, &schema, i);
    const iree_hal_executable_function_parameter_t* parameter =
        &layout->parameters[i];
    if (argument_kind == LOOM_CMD_PROGRAM_ARGUMENT_KIND_BUFFER) {
      if (parameter->type !=
          IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BINDING) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "buffer command argument requires a HAL "
                                  "binding, not a raw pointer");
      } else {
        status = loom_cmd_program_hal_buffer_ref(
            loom_cmd_program_hal_read_buffer_ref(cursor), fixed_bindings,
            &layout->bindings[parameter->offset]);
        cursor += LOOM_CMD_PROGRAM_BUFFER_REF_SIZE;
      }
    } else {
      status = loom_cmd_program_hal_pack_scalar(argument_kind, &cursor,
                                                parameter, layout->constants);
    }
  }
  if (iree_status_is_ok(status)) {
    iree_hal_dispatch_config_t config = {0};
    iree_hal_dispatch_flags_t flags = IREE_HAL_DISPATCH_FLAG_NONE;
    if (is_direct) {
      config = iree_hal_make_static_dispatch_config(
          command->payload.dispatch_direct.workgroup_count_x,
          command->payload.dispatch_direct.workgroup_count_y,
          command->payload.dispatch_direct.workgroup_count_z);
    } else {
      status = loom_cmd_program_hal_buffer_ref(
          loom_cmd_program_buffer_ref_at(
              program,
              command->payload.dispatch_indirect.workgroup_count_buffer_ref),
          fixed_bindings, &config.workgroup_count_ref);
      flags = kind == LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_STATIC
                  ? IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS
                  : IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS;
    }
    if (iree_status_is_ok(status)) {
      const iree_hal_buffer_ref_list_t bindings = {layout->info.binding_count,
                                                   layout->bindings};
      status = iree_hal_command_buffer_dispatch(
          command_buffer, entry->executable, entry->function, config,
          iree_make_const_byte_span(layout->constants,
                                    layout->info.constant_byte_length),
          bindings, flags);
    }
  }
  return status;
}

static iree_status_t loom_cmd_program_hal_record_barrier(
    const loom_cmd_program_t* program,
    const loom_cmd_program_command_t* command,
    const iree_hal_buffer_binding_t* fixed_bindings,
    loom_cmd_program_hal_barriers_t scratch,
    iree_hal_command_buffer_t* command_buffer) {
  iree_hal_barrier_t barrier = {
      .source_stage_mask = command->payload.barrier.source_stages,
      .target_stage_mask = command->payload.barrier.target_stages,
      .flags = command->payload.barrier.flags,
      .memory_barriers = scratch.memory,
      .buffer_barriers = scratch.buffers,
  };
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < command->payload.barrier.dependency_count &&
                       iree_status_is_ok(status);
       ++i) {
    const loom_cmd_program_memory_dependency_t dependency =
        loom_cmd_program_memory_dependency_at(
            program, command->payload.barrier.first_dependency + i);
    if (dependency.buffer_ref == LOOM_CMD_PROGRAM_MEMORY_DEPENDENCY_GLOBAL) {
      scratch.memory[barrier.memory_barrier_count++] =
          (iree_hal_memory_barrier_t){
              .source_scope = dependency.source_scope,
              .target_scope = dependency.target_scope,
          };
    } else {
      iree_hal_buffer_barrier_t* buffer =
          &scratch.buffers[barrier.buffer_barrier_count++];
      *buffer = (iree_hal_buffer_barrier_t){
          .source_scope = dependency.source_scope,
          .target_scope = dependency.target_scope,
      };
      status = loom_cmd_program_hal_buffer_ref(
          loom_cmd_program_buffer_ref_at(program, dependency.buffer_ref),
          fixed_bindings, &buffer->buffer_ref);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_barrier(command_buffer, &barrier);
  }
  return status;
}

static iree_status_t loom_cmd_program_hal_record_commands(
    const loom_cmd_program_t* program, loom_cmd_program_command_range_t range,
    const iree_hal_buffer_binding_t* fixed_bindings,
    const loom_cmd_program_hal_entry_t* entries,
    loom_cmd_program_hal_layout_t* layouts,
    loom_cmd_program_hal_barriers_t barriers,
    iree_hal_command_buffer_t* command_buffer) {
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < range.command_count && iree_status_is_ok(status);
       ++i) {
    const loom_cmd_program_command_t command =
        loom_cmd_program_command_at(program, range.first_command + i);
    if (loom_cmd_program_command_kind_has_implicit_execution_barrier(
            command.kind)) {
      const iree_hal_barrier_t execution = {
          .source_stage_mask = LOOM_CMD_PROGRAM_STAGE_WORK,
          .target_stage_mask = LOOM_CMD_PROGRAM_STAGE_WORK,
      };
      status = iree_hal_command_buffer_barrier(command_buffer, &execution);
      if (!iree_status_is_ok(status)) {
        break;
      }
    }
    switch (loom_cmd_program_command_kind_base(command.kind)) {
      case LOOM_CMD_PROGRAM_COMMAND_KIND_BARRIER_EXECUTION:
        break;
      case LOOM_CMD_PROGRAM_COMMAND_KIND_BARRIER:
        status = loom_cmd_program_hal_record_barrier(
            program, &command, fixed_bindings, barriers, command_buffer);
        break;
      case LOOM_CMD_PROGRAM_COMMAND_KIND_TIMESTAMP: {
        iree_hal_buffer_ref_t target = {0};
        status = loom_cmd_program_hal_buffer_ref(
            loom_cmd_program_buffer_ref_at(
                program, command.payload.timestamp.target_buffer_ref),
            fixed_bindings, &target);
        if (iree_status_is_ok(status)) {
          // Wave edges already order capture. Zero masks preserve concurrent
          // siblings; delivery belongs to memory dependencies or completion.
          status = iree_hal_command_buffer_timestamp(
              command_buffer, 0, 0, target, IREE_HAL_TIMESTAMP_FLAG_NONE);
        }
        break;
      }
      case LOOM_CMD_PROGRAM_COMMAND_KIND_FILL: {
        iree_hal_buffer_ref_t target = {0};
        status = loom_cmd_program_hal_buffer_ref(
            loom_cmd_program_buffer_ref_at(
                program, command.payload.fill.target_buffer_ref),
            fixed_bindings, &target);
        uint8_t pattern[4];
        iree_unaligned_store_le_u32(pattern, command.payload.fill.pattern);
        if (iree_status_is_ok(status)) {
          status = iree_hal_command_buffer_fill_buffer(
              command_buffer, target, pattern,
              command.payload.fill.pattern_length, IREE_HAL_FILL_FLAG_NONE);
        }
        break;
      }
      case LOOM_CMD_PROGRAM_COMMAND_KIND_COPY: {
        iree_hal_buffer_ref_t source = {0};
        iree_hal_buffer_ref_t target = {0};
        status = loom_cmd_program_hal_buffer_ref(
            loom_cmd_program_buffer_ref_at(
                program, command.payload.copy.source_buffer_ref),
            fixed_bindings, &source);
        if (iree_status_is_ok(status)) {
          status = loom_cmd_program_hal_buffer_ref(
              loom_cmd_program_buffer_ref_at(
                  program, command.payload.copy.target_buffer_ref),
              fixed_bindings, &target);
        }
        if (iree_status_is_ok(status)) {
          status = iree_hal_command_buffer_copy_buffer(
              command_buffer, source, target, IREE_HAL_COPY_FLAG_NONE);
        }
        break;
      }
      default:
        status = loom_cmd_program_hal_record_dispatch(program, &command,
                                                      fixed_bindings, entries,
                                                      layouts, command_buffer);
        break;
    }
  }
  return status;
}

static iree_status_t loom_cmd_program_hal_record(
    const loom_cmd_program_t* program, loom_cmd_program_command_range_t range,
    const iree_hal_buffer_binding_t* fixed_bindings,
    const loom_cmd_program_hal_entry_t* entries,
    loom_cmd_program_hal_layout_t* layouts,
    loom_cmd_program_hal_barriers_t barriers,
    iree_hal_command_buffer_t* command_buffer) {
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_begin(command_buffer));
  // A program or selected range exchanges payloads with staging and other
  // invocations. Its semaphore edges order execution; the recording owns
  // visibility across that external boundary, including indirect count reads.
  const iree_hal_execution_stage_t stages = LOOM_CMD_PROGRAM_STAGE_WORK;
  const iree_hal_barrier_t acquire = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
      .target_stage_mask = stages,
      .flags = IREE_HAL_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE,
  };
  IREE_RETURN_IF_ERROR(
      iree_hal_command_buffer_barrier(command_buffer, &acquire));
  iree_status_t status = loom_cmd_program_hal_record_commands(
      program, range, fixed_bindings, entries, layouts, barriers,
      command_buffer);
  if (iree_status_is_ok(status)) {
    const iree_hal_barrier_t release = {
        .source_stage_mask = stages,
        .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
        .flags = IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
    };
    status = iree_hal_command_buffer_barrier(command_buffer, &release);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_end(command_buffer);
  }
  return status;
}

iree_status_t loom_cmd_program_hal_command_buffer_create(
    const iree_hal_queue_family_t* queue_family,
    iree_hal_command_buffer_mode_t mode, const loom_cmd_program_t* program,
    const loom_cmd_program_command_range_t* range,
    iree_host_size_t fixed_buffer_count,
    const iree_hal_buffer_binding_t* fixed_bindings,
    iree_host_size_t entry_count, const loom_cmd_program_hal_entry_t* entries,
    iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer) {
  if (fixed_buffer_count != program->requirements.fixed_buffer_count ||
      entry_count != program->requirements.entry_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "command resource counts do not match requirements");
  }
  loom_cmd_program_hal_layout_t* layouts = NULL;
  if (entry_count) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        host_allocator, entry_count * sizeof(*layouts), (void**)&layouts));
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < entry_count && iree_status_is_ok(status);
       ++i) {
    loom_cmd_program_hal_layout_t* layout = &layouts[i];
    status = iree_hal_executable_function_info(
        entries[i].executable, entries[i].function, &layout->info);
    if (iree_status_is_ok(status) && layout->info.parameter_count) {
      status = iree_allocator_malloc(
          host_allocator,
          layout->info.parameter_count * sizeof(*layout->parameters),
          (void**)&layout->parameters);
    }
    if (iree_status_is_ok(status) && layout->info.parameter_count) {
      status = iree_hal_executable_function_parameters(
          entries[i].executable, entries[i].function,
          layout->info.parameter_count, layout->parameters);
    }
    if (iree_status_is_ok(status) && layout->info.binding_count) {
      status = iree_allocator_malloc(
          host_allocator,
          layout->info.binding_count * sizeof(*layout->bindings),
          (void**)&layout->bindings);
    }
    if (iree_status_is_ok(status) && layout->info.constant_byte_length) {
      status = iree_allocator_malloc(host_allocator,
                                     layout->info.constant_byte_length,
                                     (void**)&layout->constants);
    }
  }
  iree_hal_command_buffer_t* command_buffer = NULL;
  loom_cmd_program_hal_barriers_t barriers = {0};
  if (iree_status_is_ok(status) && program->maximum_barrier_dependency_count) {
    status = iree_allocator_malloc_array(
        host_allocator, program->maximum_barrier_dependency_count,
        sizeof(*barriers.buffers) + sizeof(*barriers.memory),
        (void**)&barriers.buffers);
    if (iree_status_is_ok(status)) {
      barriers.memory =
          (iree_hal_memory_barrier_t*)(barriers.buffers +
                                       program
                                           ->maximum_barrier_dependency_count);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_create(
        queue_family, mode,
        IREE_HAL_COMMAND_CATEGORY_DISPATCH | IREE_HAL_COMMAND_CATEGORY_TRANSFER,
        program->requirements.rebindable_binding_count, &command_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = loom_cmd_program_hal_record(
        program, range ? *range : loom_cmd_program_command_range_all(program),
        fixed_bindings, entries, layouts, barriers, command_buffer);
  }
  iree_allocator_free(host_allocator, barriers.buffers);
  for (iree_host_size_t i = 0; i < entry_count; ++i) {
    iree_allocator_free(host_allocator, layouts[i].constants);
    iree_allocator_free(host_allocator, layouts[i].bindings);
    iree_allocator_free(host_allocator, layouts[i].parameters);
  }
  iree_allocator_free(host_allocator, layouts);
  if (iree_status_is_ok(status)) {
    *out_command_buffer = command_buffer;
  } else {
    iree_hal_command_buffer_release(command_buffer);
  }
  return status;
}
