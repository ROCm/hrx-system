// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/runtime/command.h"

#include "iree/base/alignment.h"
#include "loom/target/arch/cmd/format.h"

// Cold-path scratch for translating one entry's logical arguments to HAL slots.
typedef struct loom_serve_command_layout_t {
  // Reflected logical parameter and physical argument counts.
  iree_hal_executable_function_info_t info;
  // Owned positional parameter descriptors.
  iree_hal_executable_function_parameter_t* parameters;
  // Owned reusable recording scratch, not issue-time argument storage.
  iree_hal_buffer_ref_t* bindings;
  // Owned reusable recording scratch including zeroed ABI padding.
  uint8_t* constants;
} loom_serve_command_layout_t;

static iree_hal_buffer_ref_t loom_serve_command_buffer_ref(
    loom_cmd_program_buffer_ref_t ref,
    iree_hal_buffer_t* const* fixed_buffers) {
  return ref.role == LOOM_CMD_PROGRAM_BUFFER_ROLE_FIXED
             ? iree_hal_make_buffer_ref(fixed_buffers[ref.root_index],
                                        ref.byte_offset, ref.byte_length)
             : iree_hal_make_indirect_buffer_ref(
                   ref.root_index, ref.byte_offset, ref.byte_length);
}

static loom_cmd_program_buffer_ref_t loom_serve_command_read_buffer_ref(
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

static iree_status_t loom_serve_command_pack_scalar(
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

static iree_status_t loom_serve_command_record_dispatch(
    const loom_cmd_program_t* program,
    const loom_cmd_program_command_t* command,
    iree_hal_buffer_t* const* fixed_buffers,
    const loom_serve_command_entry_t* entries,
    loom_serve_command_layout_t* layouts,
    iree_hal_command_buffer_t* command_buffer) {
  const loom_cmd_program_command_kind_t kind =
      loom_cmd_program_command_kind_base(command->kind);
  const bool is_direct = kind == LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT;
  const uint32_t entry_index =
      is_direct ? command->payload.dispatch_direct.entry_index
                : command->payload.dispatch_indirect.entry_index;
  const loom_serve_command_entry_t* entry = &entries[entry_index];
  loom_serve_command_layout_t* layout = &layouts[entry_index];
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
        layout->bindings[parameter->offset] = loom_serve_command_buffer_ref(
            loom_serve_command_read_buffer_ref(cursor), fixed_buffers);
        cursor += LOOM_CMD_PROGRAM_BUFFER_REF_SIZE;
      }
    } else {
      status = loom_serve_command_pack_scalar(argument_kind, &cursor, parameter,
                                              layout->constants);
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
      config.workgroup_count_ref = loom_serve_command_buffer_ref(
          loom_cmd_program_buffer_ref_at(
              program,
              command->payload.dispatch_indirect.workgroup_count_buffer_ref),
          fixed_buffers);
      flags = kind == LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_STATIC
                  ? IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS
                  : IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS;
    }
    const iree_hal_buffer_ref_list_t bindings = {layout->info.binding_count,
                                                 layout->bindings};
    status = iree_hal_command_buffer_dispatch(
        command_buffer, entry->executable, entry->function, config,
        iree_make_const_byte_span(layout->constants,
                                  layout->info.constant_byte_length),
        bindings, flags);
  }
  return status;
}

static iree_hal_execution_stage_t loom_serve_command_wave_stages(
    const loom_cmd_program_t* program, loom_cmd_program_command_range_t range) {
  const iree_hal_execution_stage_t stages =
      IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_TRANSFER;
  // A leading direct command may share its wave with indirect consumers.
  for (uint32_t i = 0; i < range.command_count; ++i) {
    const loom_cmd_program_command_kind_t kind =
        loom_cmd_program_command_kind_base(
            loom_cmd_program_command_at(program, range.first_command + i).kind);
    if (kind == LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_STATIC ||
        kind == LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_DYNAMIC) {
      return stages | IREE_HAL_EXECUTION_STAGE_COMMAND_PROCESS;
    }
  }
  return stages;
}

static iree_status_t loom_serve_command_record_wave(
    const loom_cmd_program_t* program, loom_cmd_program_command_range_t range,
    iree_hal_buffer_t* const* fixed_buffers,
    const loom_serve_command_entry_t* entries,
    loom_serve_command_layout_t* layouts,
    iree_hal_command_buffer_t* command_buffer) {
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < range.command_count && iree_status_is_ok(status);
       ++i) {
    const loom_cmd_program_command_t command =
        loom_cmd_program_command_at(program, range.first_command + i);
    switch (loom_cmd_program_command_kind_base(command.kind)) {
      case LOOM_CMD_PROGRAM_COMMAND_KIND_BARRIER_EXECUTION:
        break;
      case LOOM_CMD_PROGRAM_COMMAND_KIND_FILL: {
        const iree_hal_buffer_ref_t target = loom_serve_command_buffer_ref(
            loom_cmd_program_buffer_ref_at(
                program, command.payload.fill.target_buffer_ref),
            fixed_buffers);
        uint8_t pattern[4];
        iree_unaligned_store_le_u32(pattern, command.payload.fill.pattern);
        status = iree_hal_command_buffer_fill_buffer(
            command_buffer, target, pattern,
            command.payload.fill.pattern_length, IREE_HAL_FILL_FLAG_NONE);
        break;
      }
      case LOOM_CMD_PROGRAM_COMMAND_KIND_COPY:
        status = iree_hal_command_buffer_copy_buffer(
            command_buffer,
            loom_serve_command_buffer_ref(
                loom_cmd_program_buffer_ref_at(
                    program, command.payload.copy.source_buffer_ref),
                fixed_buffers),
            loom_serve_command_buffer_ref(
                loom_cmd_program_buffer_ref_at(
                    program, command.payload.copy.target_buffer_ref),
                fixed_buffers),
            IREE_HAL_COPY_FLAG_NONE);
        break;
      default:
        status = loom_serve_command_record_dispatch(
            program, &command, fixed_buffers, entries, layouts, command_buffer);
        break;
    }
  }
  return status;
}

static iree_status_t loom_serve_command_record(
    const loom_cmd_program_t* program, iree_hal_buffer_t* const* fixed_buffers,
    const loom_serve_command_entry_t* entries,
    loom_serve_command_layout_t* layouts,
    iree_hal_command_buffer_t* command_buffer) {
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_begin(command_buffer));
  loom_cmd_program_barrier_wave_iterator_t iterator;
  loom_cmd_program_barrier_wave_iterator_initialize(program, &iterator);
  loom_cmd_program_barrier_wave_t wave;
  iree_hal_execution_stage_t source_stages =
      IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_TRANSFER;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         loom_cmd_program_barrier_wave_iterator_next(&iterator, &wave)) {
    const iree_hal_execution_stage_t target_stages =
        loom_serve_command_wave_stages(program, wave.commands);
    if (wave.ordinal != 0) {
      iree_hal_memory_barrier_t barrier = {
          .source_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE |
                          IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE,
          .target_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_READ |
                          IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE |
                          IREE_HAL_ACCESS_SCOPE_TRANSFER_READ |
                          IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE,
      };
      if (iree_any_bit_set(target_stages,
                           IREE_HAL_EXECUTION_STAGE_COMMAND_PROCESS)) {
        barrier.target_scope |= IREE_HAL_ACCESS_SCOPE_INDIRECT_COMMAND_READ;
      }
      // Count publication targets command processing, not shader reads. The
      // predecessor stage also orders indirect reads before count reuse.
      status = iree_hal_command_buffer_execution_barrier(
          command_buffer, source_stages, target_stages,
          IREE_HAL_EXECUTION_BARRIER_FLAG_NONE, 1, &barrier, 0, NULL);
    }
    if (iree_status_is_ok(status)) {
      status =
          loom_serve_command_record_wave(program, wave.commands, fixed_buffers,
                                         entries, layouts, command_buffer);
    }
    source_stages = target_stages;
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_end(command_buffer);
  }
  return status;
}

iree_status_t loom_serve_command_create(
    const iree_hal_queue_family_t* queue_family,
    iree_hal_command_buffer_mode_t mode, const loom_cmd_program_t* program,
    iree_host_size_t fixed_buffer_count,
    iree_hal_buffer_t* const* fixed_buffers, iree_host_size_t entry_count,
    const loom_serve_command_entry_t* entries, iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer) {
  if (fixed_buffer_count != program->requirements.fixed_buffer_count ||
      entry_count != program->requirements.entry_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "command resource counts do not match requirements");
  }
  loom_serve_command_layout_t* layouts = NULL;
  if (entry_count) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        host_allocator, entry_count * sizeof(*layouts), (void**)&layouts));
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < entry_count && iree_status_is_ok(status);
       ++i) {
    loom_serve_command_layout_t* layout = &layouts[i];
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
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_create(
        queue_family, mode,
        IREE_HAL_COMMAND_CATEGORY_DISPATCH | IREE_HAL_COMMAND_CATEGORY_TRANSFER,
        program->requirements.rebindable_binding_count, &command_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_command_record(program, fixed_buffers, entries, layouts,
                                       command_buffer);
  }
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
