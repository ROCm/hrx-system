// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target/cmd/hal.h"

#include "iree/base/alignment.h"
#include "iree/base/internal/atomics.h"
#include "loom/binding/c/target/cmd/program.h"
#include "loom/target/arch/cmd/hal/recording.h"
#include "loomc/iree.h"

struct loomc_cmd_hal_program_t {
  // Atomic reference count for shared ownership.
  iree_atomic_ref_count_t ref_count;

  // Allocator used for this object.
  loomc_allocator_t allocator;

  // Recorded reusable command buffer owned by this object.
  iree_hal_command_buffer_t* command_buffer;

  // Exact number of issue-time binding-table slots.
  loomc_host_size_t binding_count;
};

static loomc_status_t loomc_cmd_hal_validate_options(
    const loom_cmd_program_t* program,
    const loomc_cmd_hal_program_options_t* options) {
  if (options == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "command HAL program options must not be NULL");
  }
  if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
      options->type != LOOMC_STRUCTURE_TYPE_CMD_HAL_PROGRAM_OPTIONS) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "command HAL program options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "command HAL program options structure_size is too small");
  }
  if (options->next != NULL) {
    return loomc_make_status(
        LOOMC_STATUS_UNIMPLEMENTED,
        "command HAL program option extensions are not supported");
  }
  if (options->fixed_buffer_count != program->requirements.fixed_buffer_count) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "command HAL fixed-buffer count does not match the program ABI");
  }
  if (options->fixed_buffer_count != 0 && options->fixed_buffers == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "command HAL fixed_buffer_count is non-zero but fixed_buffers is NULL");
  }
  for (loomc_host_size_t i = 0; i < options->fixed_buffer_count; ++i) {
    if (options->fixed_buffers[i].buffer == NULL) {
      return loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "command HAL fixed buffers must be direct buffer ranges");
    }
  }
  if (options->executable_count != program->requirements.executable_count) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "command HAL executable count does not match the program ABI");
  }
  if (options->executable_count != 0 && options->executables == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "command HAL executable_count is non-zero but executables is NULL");
  }
  for (loomc_host_size_t i = 0; i < options->executable_count; ++i) {
    if (options->executables[i] == NULL) {
      return loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "command HAL executable table contains a NULL entry");
    }
  }
  return loomc_ok_status();
}

static iree_hal_command_category_t loomc_cmd_hal_program_categories(
    const loom_cmd_program_t* program) {
  iree_hal_command_category_t categories = 0;
  for (uint32_t i = 0; i < program->commands.count; ++i) {
    const loom_cmd_program_command_t command =
        loom_cmd_program_command_at(program, i);
    switch (loom_cmd_program_command_kind_base(command.kind)) {
      case LOOM_CMD_PROGRAM_COMMAND_KIND_FILL:
      case LOOM_CMD_PROGRAM_COMMAND_KIND_COPY:
        categories |= IREE_HAL_COMMAND_CATEGORY_TRANSFER;
        break;
      case LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT:
      case LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_STATIC:
      case LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_DYNAMIC:
        categories |= IREE_HAL_COMMAND_CATEGORY_DISPATCH;
        break;
      case LOOM_CMD_PROGRAM_COMMAND_KIND_BARRIER_EXECUTION:
        break;
      default:
        IREE_ASSERT_UNREACHABLE("validated command kind");
        break;
    }
  }
  return categories != 0 ? categories : IREE_HAL_COMMAND_CATEGORY_ANY;
}

static iree_status_t loomc_cmd_hal_resolve_entries(
    const loom_cmd_program_package_t* package,
    const loom_cmd_program_package_export_t* package_export,
    const loomc_cmd_hal_program_options_t* options,
    iree_allocator_t host_allocator, uint8_t** out_storage,
    loom_cmd_hal_entry_t** out_entries) {
  *out_storage = NULL;
  *out_entries = NULL;

  iree_host_size_t storage_size = 0;
  iree_host_size_t entries_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &storage_size,
      IREE_STRUCT_FIELD_ALIGNED(
          package_export->entry_count, loom_cmd_hal_entry_t,
          iree_alignof(loom_cmd_hal_entry_t), &entries_offset)));

  uint8_t* storage = NULL;
  iree_status_t status = iree_ok_status();
  if (storage_size != 0) {
    status = iree_allocator_malloc_uninitialized(host_allocator, storage_size,
                                                 (void**)&storage);
  }
  loom_cmd_hal_entry_t* entries =
      storage != NULL ? (loom_cmd_hal_entry_t*)(storage + entries_offset)
                      : NULL;
  iree_host_size_t parameter_count = 0;
  for (uint32_t i = 0;
       i < package_export->entry_count && iree_status_is_ok(status); ++i) {
    const loom_cmd_program_package_entry_t package_entry =
        loom_cmd_program_package_export_entry_at(package, package_export, i);
    iree_hal_executable_t* executable =
        options->executables[package_entry.executable_index];
    entries[i].executable_index = package_entry.executable_index;
    status = iree_hal_executable_lookup_function_by_name(
        executable, package_entry.name, &entries[i].function);
    if (iree_status_is_ok(status)) {
      status = iree_hal_executable_function_info(
          executable, entries[i].function, &entries[i].info);
    }
    if (iree_status_is_ok(status) &&
        !iree_host_size_checked_add(parameter_count,
                                    entries[i].info.parameter_count,
                                    &parameter_count)) {
      status =
          iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                           "command HAL executable parameter count overflow");
    }
  }

  iree_host_size_t parameters_offset = 0;
  if (iree_status_is_ok(status)) {
    storage_size = 0;
    status = IREE_STRUCT_LAYOUT(
        0, &storage_size,
        IREE_STRUCT_FIELD_ALIGNED(
            package_export->entry_count, loom_cmd_hal_entry_t,
            iree_alignof(loom_cmd_hal_entry_t), &entries_offset),
        IREE_STRUCT_FIELD_ALIGNED(
            parameter_count, iree_hal_executable_function_parameter_t,
            iree_alignof(iree_hal_executable_function_parameter_t),
            &parameters_offset));
  }
  if (iree_status_is_ok(status) && storage_size != 0) {
    status =
        iree_allocator_realloc(host_allocator, storage_size, (void**)&storage);
  }
  if (iree_status_is_ok(status)) {
    entries = storage != NULL
                  ? (loom_cmd_hal_entry_t*)(storage + entries_offset)
                  : NULL;
    iree_hal_executable_function_parameter_t* parameters =
        storage != NULL
            ? (iree_hal_executable_function_parameter_t*)(storage +
                                                          parameters_offset)
            : NULL;
    iree_host_size_t parameter_offset = 0;
    for (uint32_t i = 0;
         i < package_export->entry_count && iree_status_is_ok(status); ++i) {
      const loom_cmd_program_package_entry_t package_entry =
          loom_cmd_program_package_export_entry_at(package, package_export, i);
      const iree_host_size_t entry_parameter_count =
          entries[i].info.parameter_count;
      entries[i].parameters =
          entry_parameter_count != 0 ? parameters + parameter_offset : NULL;
      if (entry_parameter_count != 0) {
        iree_hal_executable_function_parameter_t* entry_parameters =
            parameters + parameter_offset;
        status = iree_hal_executable_function_parameters(
            options->executables[package_entry.executable_index],
            entries[i].function, entry_parameter_count, entry_parameters);
      }
      parameter_offset += entry_parameter_count;
    }
  }

  if (iree_status_is_ok(status)) {
    *out_storage = storage;
    *out_entries = entries;
  } else {
    iree_allocator_free(host_allocator, storage);
  }
  return status;
}

loomc_status_t loomc_cmd_hal_program_create(
    const loomc_cmd_program_package_t* package,
    loomc_cmd_program_export_t program_export, iree_hal_device_t* device,
    const loomc_cmd_hal_program_options_t* options, loomc_allocator_t allocator,
    loomc_cmd_hal_program_t** out_program) {
  if (out_program == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_program must not be NULL");
  }
  *out_program = NULL;
  if (device == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "command HAL program requires a device");
  }
  if (!loomc_allocator_is_valid(allocator)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "command HAL program allocator is invalid");
  }
  const loom_cmd_program_package_export_t* package_export =
      loomc_cmd_program_package_resolve_export(package, program_export);
  if (package_export == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "command program export token is invalid");
  }
  LOOMC_RETURN_IF_ERROR(
      loomc_cmd_hal_validate_options(&package_export->program, options));

  const loom_cmd_program_package_t* parsed_package =
      loomc_cmd_program_package_parsed(package);
  const iree_allocator_t host_allocator = iree_allocator_from_loomc(allocator);
  uint8_t* entry_storage = NULL;
  loom_cmd_hal_entry_t* entries = NULL;
  iree_status_t status =
      loomc_cmd_hal_resolve_entries(parsed_package, package_export, options,
                                    host_allocator, &entry_storage, &entries);

  iree_hal_command_buffer_t* command_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_create(
        device, options->command_buffer_mode,
        loomc_cmd_hal_program_categories(&package_export->program),
        options->queue_affinity,
        package_export->program.requirements.rebindable_binding_count,
        &command_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_begin(command_buffer);
  }
  if (iree_status_is_ok(status)) {
    const loom_cmd_hal_recording_inputs_t recording_inputs = {
        .fixed_buffers = options->fixed_buffers,
        .executables = options->executables,
        .entries = entries,
    };
    status =
        loom_cmd_hal_record_program(&package_export->program, &recording_inputs,
                                    command_buffer, host_allocator);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_end(command_buffer);
  }
  iree_allocator_free(host_allocator, entry_storage);

  loomc_cmd_hal_program_t* program = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_uninitialized(
        host_allocator, sizeof(*program), (void**)&program);
  }
  if (iree_status_is_ok(status)) {
    iree_atomic_ref_count_init(&program->ref_count);
    program->allocator = allocator;
    program->command_buffer = command_buffer;
    program->binding_count =
        package_export->program.requirements.rebindable_binding_count;
    *out_program = program;
  } else {
    iree_hal_command_buffer_release(command_buffer);
  }
  return loomc_status_from_iree(status);
}

void loomc_cmd_hal_program_retain(loomc_cmd_hal_program_t* program) {
  if (program == NULL) return;
  iree_atomic_ref_count_inc(&program->ref_count);
}

void loomc_cmd_hal_program_release(loomc_cmd_hal_program_t* program) {
  if (program == NULL) return;
  if (iree_atomic_ref_count_dec(&program->ref_count) == 1) {
    const loomc_allocator_t allocator = program->allocator;
    iree_hal_command_buffer_release(program->command_buffer);
    loomc_allocator_free(allocator, program);
  }
}

iree_hal_command_buffer_t* loomc_cmd_hal_program_command_buffer(
    const loomc_cmd_hal_program_t* program) {
  return program != NULL ? program->command_buffer : NULL;
}

loomc_host_size_t loomc_cmd_hal_program_binding_count(
    const loomc_cmd_hal_program_t* program) {
  return program != NULL ? program->binding_count : 0;
}
