// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/reporting/format_command.h"

#include "loom/util/json.h"

static iree_status_t loom_report_command_buffer_json(
    const loom_target_compile_report_command_buffer_t* buffer,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("role"),
      loom_target_compile_report_command_buffer_role_name(buffer->role)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("root_index"), buffer->root_index));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("byte_offset"), buffer->byte_offset));
  if (buffer->byte_length == UINT64_MAX) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_write_null_field(&object, IREE_SV("byte_length")));
  } else {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("byte_length"), buffer->byte_length));
  }
  return loom_json_object_end(&object);
}

static iree_status_t loom_report_command_storage_json(
    const loom_target_compile_report_command_storage_t* storage,
    loom_output_stream_t* stream) {
  if (storage->root_index == UINT32_MAX) {
    return loom_output_stream_write(stream, IREE_SV("null"));
  }
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("root_index"), storage->root_index));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("required_byte_length"), storage->required_byte_length));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("minimum_alignment"), storage->minimum_alignment));
  return loom_json_object_end(&object);
}

static iree_status_t loom_report_command_json(
    const loom_target_compile_report_command_t* command,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("kind"),
      loom_target_compile_report_command_kind_name(command->kind)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
      &object, IREE_SV("barrier_before"),
      (command->flags & LOOM_TARGET_COMPILE_REPORT_COMMAND_BARRIER_BEFORE) !=
          0));
  if (!iree_string_view_is_empty(command->entry_name)) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("entry"), command->entry_name));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("argument_count"), command->argument_count));
  }
  switch (command->kind) {
    case LOOM_TARGET_COMPILE_REPORT_COMMAND_DISPATCH_DIRECT: {
      IREE_RETURN_IF_ERROR(
          loom_json_object_begin_field(&object, IREE_SV("workgroups")));
      loom_json_array_writer_t array;
      IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
      for (uint32_t axis = 0; axis < 3; ++axis) {
        IREE_RETURN_IF_ERROR(loom_json_array_write_uint64_element(
            &array, command->payload.workgroup_counts[axis]));
      }
      IREE_RETURN_IF_ERROR(loom_json_array_end(&array));
      break;
    }
    case LOOM_TARGET_COMPILE_REPORT_COMMAND_DISPATCH_INDIRECT_STATIC:
    case LOOM_TARGET_COMPILE_REPORT_COMMAND_DISPATCH_INDIRECT_DYNAMIC: {
      IREE_RETURN_IF_ERROR(
          loom_json_object_begin_field(&object, IREE_SV("count_buffer")));
      IREE_RETURN_IF_ERROR(loom_report_command_buffer_json(
          &command->payload.indirect_counts, stream));
      break;
    }
    case LOOM_TARGET_COMPILE_REPORT_COMMAND_FILL: {
      IREE_RETURN_IF_ERROR(
          loom_json_object_begin_field(&object, IREE_SV("target")));
      IREE_RETURN_IF_ERROR(loom_report_command_buffer_json(
          &command->payload.fill.target, stream));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &object, IREE_SV("pattern"), command->payload.fill.pattern));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &object, IREE_SV("pattern_byte_count"),
          command->payload.fill.pattern_byte_count));
      break;
    }
    case LOOM_TARGET_COMPILE_REPORT_COMMAND_COPY: {
      IREE_RETURN_IF_ERROR(
          loom_json_object_begin_field(&object, IREE_SV("source")));
      IREE_RETURN_IF_ERROR(loom_report_command_buffer_json(
          &command->payload.copy.source, stream));
      IREE_RETURN_IF_ERROR(
          loom_json_object_begin_field(&object, IREE_SV("target")));
      IREE_RETURN_IF_ERROR(loom_report_command_buffer_json(
          &command->payload.copy.target, stream));
      break;
    }
    case LOOM_TARGET_COMPILE_REPORT_COMMAND_TIMESTAMP: {
      IREE_RETURN_IF_ERROR(
          loom_json_object_begin_field(&object, IREE_SV("target")));
      IREE_RETURN_IF_ERROR(
          loom_report_command_buffer_json(&command->payload.timestamp, stream));
      break;
    }
    case LOOM_TARGET_COMPILE_REPORT_COMMAND_BARRIER:
      break;
    case LOOM_TARGET_COMPILE_REPORT_COMMAND_KIND_COUNT:
      IREE_ASSERT_UNREACHABLE("command kind is a diagnostic operation");
  }
  return loom_json_object_end(&object);
}

static iree_status_t loom_report_command_plan_json(
    const loom_target_compile_report_command_plan_t* plan,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("root"), plan->root_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("command_count"), plan->command_count));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("command_counts")));
  loom_json_object_writer_t counts;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &counts));
  for (uint32_t i = 0; i < LOOM_TARGET_COMPILE_REPORT_COMMAND_KIND_COUNT; ++i) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &counts, loom_target_compile_report_command_kind_name(i),
        plan->command_kind_counts[i]));
  }
  IREE_RETURN_IF_ERROR(loom_json_object_end(&counts));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("barrier_before_count"), plan->barrier_before_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("fixed_buffer_count"), plan->fixed_buffer_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("rebindable_binding_count"),
      plan->rebindable_binding_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("executable_count"), plan->executable_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("entry_count"), plan->entry_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("binding_count"), plan->binding_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("parameter_root_count"), plan->parameter_root_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("parameter_count"), plan->parameter_count));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("transient")));
  IREE_RETURN_IF_ERROR(
      loom_report_command_storage_json(&plan->transient, stream));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("launch_counts")));
  IREE_RETURN_IF_ERROR(
      loom_report_command_storage_json(&plan->launch_counts, stream));
  if (mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    if (plan->commands || plan->command_count == 0) {
      IREE_RETURN_IF_ERROR(
          loom_json_object_begin_field(&object, IREE_SV("commands")));
      loom_json_array_writer_t array;
      IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
      for (uint32_t i = 0; i < plan->command_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
        IREE_RETURN_IF_ERROR(
            loom_report_command_json(&plan->commands[i], stream));
      }
      IREE_RETURN_IF_ERROR(loom_json_array_end(&array));
    }
    if (plan->bindings || plan->binding_count == 0) {
      IREE_RETURN_IF_ERROR(
          loom_json_object_begin_field(&object, IREE_SV("bindings")));
      loom_json_array_writer_t array;
      IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
      for (uint32_t i = 0; i < plan->binding_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
        loom_json_object_writer_t binding;
        IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &binding));
        IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
            &binding, IREE_SV("role"),
            loom_target_compile_report_command_buffer_role_name(
                plan->bindings[i].role)));
        IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
            &binding, IREE_SV("root_index"), plan->bindings[i].root_index));
        IREE_RETURN_IF_ERROR(loom_json_object_end(&binding));
      }
      IREE_RETURN_IF_ERROR(loom_json_array_end(&array));
    }
    if (plan->parameter_roots || plan->parameter_root_count == 0) {
      IREE_RETURN_IF_ERROR(
          loom_json_object_begin_field(&object, IREE_SV("parameter_roots")));
      loom_json_array_writer_t array;
      IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
      for (uint32_t i = 0; i < plan->parameter_root_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
        IREE_RETURN_IF_ERROR(loom_report_command_storage_json(
            &plan->parameter_roots[i], stream));
      }
      IREE_RETURN_IF_ERROR(loom_json_array_end(&array));
    }
    if (plan->parameters || plan->parameter_count == 0) {
      IREE_RETURN_IF_ERROR(
          loom_json_object_begin_field(&object, IREE_SV("parameters")));
      loom_json_array_writer_t array;
      IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
      for (uint32_t i = 0; i < plan->parameter_count; ++i) {
        const loom_target_compile_report_command_parameter_t* parameter =
            &plan->parameters[i];
        IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
        loom_json_object_writer_t row;
        IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &row));
        IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
            &row, IREE_SV("key"), parameter->key));
        IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
            &row, IREE_SV("root_index"), parameter->root_index));
        IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
            &row, IREE_SV("byte_offset"), parameter->byte_offset));
        IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
            &row, IREE_SV("byte_length"), parameter->byte_length));
        IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
            &row, IREE_SV("minimum_alignment"), parameter->minimum_alignment));
        IREE_RETURN_IF_ERROR(loom_json_object_end(&row));
      }
      IREE_RETURN_IF_ERROR(loom_json_array_end(&array));
    }
  }
  return loom_json_object_end(&object);
}

iree_status_t loom_target_compile_report_format_commands_json(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
  for (const loom_target_compile_report_vec_t* block =
           report->command_plans.head;
       block; block = block->next) {
    const loom_target_compile_report_command_plan_t* plans =
        loom_target_compile_report_vec_const_rows(block);
    for (iree_host_size_t i = 0; i < block->count; ++i) {
      IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
      IREE_RETURN_IF_ERROR(
          loom_report_command_plan_json(&plans[i], mode, stream));
    }
  }
  return loom_json_array_end(&array);
}

static iree_status_t loom_report_command_buffer_text(
    const loom_target_compile_report_command_buffer_t* buffer,
    iree_string_builder_t* builder) {
  const iree_string_view_t role =
      loom_target_compile_report_command_buffer_role_name(buffer->role);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "%.*s[%" PRIu32 "]+%" PRIu64 ":", (int)role.size, role.data,
      buffer->root_index, buffer->byte_offset));
  if (buffer->byte_length == UINT64_MAX) {
    return iree_string_builder_append_cstring(builder, "remaining");
  }
  return iree_string_builder_append_format(builder, "%" PRIu64,
                                           buffer->byte_length);
}

static iree_status_t loom_report_command_storage_text(
    iree_string_view_t name,
    const loom_target_compile_report_command_storage_t* storage,
    iree_string_builder_t* builder) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "  %.*s: ", (int)name.size, name.data));
  if (storage->root_index == UINT32_MAX) {
    return iree_string_builder_append_cstring(builder, "none\n");
  }
  return iree_string_builder_append_format(
      builder, "root[%" PRIu32 "] bytes=%" PRIu64 " alignment=%" PRIu64 "\n",
      storage->root_index, storage->required_byte_length,
      storage->minimum_alignment);
}

static iree_status_t loom_report_command_text(
    const loom_target_compile_report_command_t* command, uint32_t ordinal,
    iree_string_builder_t* builder) {
  const iree_string_view_t kind =
      loom_target_compile_report_command_kind_name(command->kind);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "  command[%" PRIu32 "]: %.*s%s", ordinal, (int)kind.size,
      kind.data,
      command->flags & LOOM_TARGET_COMPILE_REPORT_COMMAND_BARRIER_BEFORE
          ? " barrier_before"
          : ""));
  if (!iree_string_view_is_empty(command->entry_name)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, " @%.*s arguments=%" PRIu32, (int)command->entry_name.size,
        command->entry_name.data, command->argument_count));
  }
  switch (command->kind) {
    case LOOM_TARGET_COMPILE_REPORT_COMMAND_DISPATCH_DIRECT: {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, " workgroups=[%" PRIu32 ", %" PRIu32 ", %" PRIu32 "]",
          command->payload.workgroup_counts[0],
          command->payload.workgroup_counts[1],
          command->payload.workgroup_counts[2]));
      break;
    }
    case LOOM_TARGET_COMPILE_REPORT_COMMAND_DISPATCH_INDIRECT_STATIC:
    case LOOM_TARGET_COMPILE_REPORT_COMMAND_DISPATCH_INDIRECT_DYNAMIC: {
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(builder, " counts="));
      IREE_RETURN_IF_ERROR(loom_report_command_buffer_text(
          &command->payload.indirect_counts, builder));
      break;
    }
    case LOOM_TARGET_COMPILE_REPORT_COMMAND_FILL: {
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(builder, " target="));
      IREE_RETURN_IF_ERROR(loom_report_command_buffer_text(
          &command->payload.fill.target, builder));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, " pattern=0x%08" PRIx32 " pattern_bytes=%" PRIu32,
          command->payload.fill.pattern,
          command->payload.fill.pattern_byte_count));
      break;
    }
    case LOOM_TARGET_COMPILE_REPORT_COMMAND_COPY: {
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(builder, " source="));
      IREE_RETURN_IF_ERROR(loom_report_command_buffer_text(
          &command->payload.copy.source, builder));
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(builder, " target="));
      IREE_RETURN_IF_ERROR(loom_report_command_buffer_text(
          &command->payload.copy.target, builder));
      break;
    }
    case LOOM_TARGET_COMPILE_REPORT_COMMAND_TIMESTAMP: {
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(builder, " target="));
      IREE_RETURN_IF_ERROR(loom_report_command_buffer_text(
          &command->payload.timestamp, builder));
      break;
    }
    case LOOM_TARGET_COMPILE_REPORT_COMMAND_BARRIER:
      break;
    case LOOM_TARGET_COMPILE_REPORT_COMMAND_KIND_COUNT:
      IREE_ASSERT_UNREACHABLE("command kind is a diagnostic operation");
  }
  return iree_string_builder_append_cstring(builder, "\n");
}

iree_status_t loom_target_compile_report_format_commands_text(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    iree_string_builder_t* builder) {
  iree_host_size_t ordinal = 0;
  for (const loom_target_compile_report_vec_t* block =
           report->command_plans.head;
       block; block = block->next) {
    const loom_target_compile_report_command_plan_t* plans =
        loom_target_compile_report_vec_const_rows(block);
    for (iree_host_size_t i = 0; i < block->count; ++i, ++ordinal) {
      const loom_target_compile_report_command_plan_t* plan = &plans[i];
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "command_program[%" PRIhsz "]: @%.*s\n  commands: %" PRIu32 "\n",
          ordinal, (int)plan->root_name.size, plan->root_name.data,
          plan->command_count));
      for (uint32_t kind = 0;
           kind < LOOM_TARGET_COMPILE_REPORT_COMMAND_KIND_COUNT; ++kind) {
        if (plan->command_kind_counts[kind] == 0) {
          continue;
        }
        const iree_string_view_t name =
            loom_target_compile_report_command_kind_name(kind);
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, "    %.*s: %" PRIu32 "\n", (int)name.size, name.data,
            plan->command_kind_counts[kind]));
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "  barrier_before_count: %" PRIu32 "\n  resources: fixed=%" PRIu32
          " rebindable=%" PRIu32 " executables=%" PRIu32 " entries=%" PRIu32
          "\n"
          "  placement: bindings=%" PRIu32 " parameter_roots=%" PRIu32
          " parameters=%" PRIu32 "\n",
          plan->barrier_before_count, plan->fixed_buffer_count,
          plan->rebindable_binding_count, plan->executable_count,
          plan->entry_count, plan->binding_count, plan->parameter_root_count,
          plan->parameter_count));
      IREE_RETURN_IF_ERROR(loom_report_command_storage_text(
          IREE_SV("transient"), &plan->transient, builder));
      IREE_RETURN_IF_ERROR(loom_report_command_storage_text(
          IREE_SV("launch_counts"), &plan->launch_counts, builder));
      if (mode != LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
        continue;
      }
      for (uint32_t j = 0; plan->commands && j < plan->command_count; ++j) {
        IREE_RETURN_IF_ERROR(
            loom_report_command_text(&plan->commands[j], j, builder));
      }
      for (uint32_t j = 0; plan->bindings && j < plan->binding_count; ++j) {
        const iree_string_view_t role =
            loom_target_compile_report_command_buffer_role_name(
                plan->bindings[j].role);
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, "  binding[%" PRIu32 "]: %.*s[%" PRIu32 "]\n", j,
            (int)role.size, role.data, plan->bindings[j].root_index));
      }
      for (uint32_t j = 0;
           plan->parameter_roots && j < plan->parameter_root_count; ++j) {
        IREE_RETURN_IF_ERROR(loom_report_command_storage_text(
            IREE_SV("parameter_root"), &plan->parameter_roots[j], builder));
      }
      for (uint32_t j = 0; plan->parameters && j < plan->parameter_count; ++j) {
        const loom_target_compile_report_command_parameter_t* parameter =
            &plan->parameters[j];
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder,
            "  parameter[%" PRIu32 "]: %.*s fixed[%" PRIu32 "]+%" PRIu64
            ":%" PRIu64 " alignment=%" PRIu64 "\n",
            j, (int)parameter->key.size, parameter->key.data,
            parameter->root_index, parameter->byte_offset,
            parameter->byte_length, parameter->minimum_alignment));
      }
    }
  }
  return iree_ok_status();
}
