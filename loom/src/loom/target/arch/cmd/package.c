// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/package.h"

#include <inttypes.h>
#include <string.h>

#include "iree/base/alignment.h"
#include "loom/target/arch/cmd/package_format.h"

static void loom_cmd_program_package_bind(
    iree_const_byte_span_t data, uint32_t export_count, uint32_t entry_count,
    uint32_t string_length, uint32_t payload_length,
    const loom_cmd_program_package_format_layout_t* layout,
    loom_cmd_program_package_t* out_package) {
  *out_package = (loom_cmd_program_package_t){
      .storage = data,
      .export_table = data.data + layout->export_offset,
      .export_count = export_count,
      .entry_table = data.data + layout->entry_offset,
      .entry_count = entry_count,
      .strings = iree_make_const_byte_span(data.data + layout->string_offset,
                                           string_length),
      .payloads = iree_make_const_byte_span(data.data + layout->payload_offset,
                                            payload_length),
  };
}

static loom_cmd_program_package_export_t
loom_cmd_program_package_bind_export_at(
    const loom_cmd_program_package_t* package, uint32_t export_index,
    iree_const_byte_span_t* out_program_data) {
  IREE_ASSERT_ARGUMENT(package);
  IREE_ASSERT_LT(export_index, package->export_count);
  const uint8_t* record = package->export_table +
                          export_index * LOOM_CMD_PROGRAM_PACKAGE_EXPORT_SIZE;
  const uint32_t name_offset = iree_unaligned_load_le_u32(
      record + LOOM_CMD_PROGRAM_PACKAGE_EXPORT_NAME_OFFSET_OFFSET);
  const uint32_t name_length = iree_unaligned_load_le_u32(
      record + LOOM_CMD_PROGRAM_PACKAGE_EXPORT_NAME_LENGTH_OFFSET);
  const uint32_t program_offset = iree_unaligned_load_le_u32(
      record + LOOM_CMD_PROGRAM_PACKAGE_EXPORT_PROGRAM_OFFSET_OFFSET);
  const uint32_t program_length = iree_unaligned_load_le_u32(
      record + LOOM_CMD_PROGRAM_PACKAGE_EXPORT_PROGRAM_LENGTH_OFFSET);
  IREE_ASSERT_LE(name_offset, package->strings.data_length);
  IREE_ASSERT_LE(name_length, package->strings.data_length - name_offset);
  IREE_ASSERT_LE(program_offset, package->storage.data_length);
  IREE_ASSERT_LE(program_length, package->storage.data_length - program_offset);
  *out_program_data = iree_make_const_byte_span(
      package->storage.data + program_offset, program_length);
  return (loom_cmd_program_package_export_t){
      .name = iree_make_string_view(
          (const char*)package->strings.data + name_offset, name_length),
      .first_entry = iree_unaligned_load_le_u32(
          record + LOOM_CMD_PROGRAM_PACKAGE_EXPORT_FIRST_ENTRY_OFFSET),
      .entry_count = iree_unaligned_load_le_u32(
          record + LOOM_CMD_PROGRAM_PACKAGE_EXPORT_ENTRY_COUNT_OFFSET),
  };
}

loom_cmd_program_package_export_t loom_cmd_program_package_export_at(
    const loom_cmd_program_package_t* package, uint32_t export_index) {
  iree_const_byte_span_t program_data = iree_const_byte_span_empty();
  loom_cmd_program_package_export_t program_export =
      loom_cmd_program_package_bind_export_at(package, export_index,
                                              &program_data);
  loom_cmd_program_bind_verified(program_data, &program_export.program);
  return program_export;
}

bool loom_cmd_program_package_lookup_export(
    const loom_cmd_program_package_t* package, iree_string_view_t name,
    loom_cmd_program_package_export_t* out_export) {
  IREE_ASSERT_ARGUMENT(package);
  IREE_ASSERT_ARGUMENT(out_export);
  *out_export = (loom_cmd_program_package_export_t){0};
  for (uint32_t i = 0; i < package->export_count; ++i) {
    const loom_cmd_program_package_export_t program_export =
        loom_cmd_program_package_export_at(package, i);
    if (iree_string_view_equal(program_export.name, name)) {
      *out_export = program_export;
      return true;
    }
  }
  return false;
}

loom_cmd_program_package_entry_t loom_cmd_program_package_export_entry_at(
    const loom_cmd_program_package_t* package,
    const loom_cmd_program_package_export_t* program_export,
    uint32_t entry_index) {
  IREE_ASSERT_ARGUMENT(package);
  IREE_ASSERT_ARGUMENT(program_export);
  IREE_ASSERT_LT(entry_index, program_export->entry_count);
  const uint32_t global_entry_index = program_export->first_entry + entry_index;
  IREE_ASSERT_LT(global_entry_index, package->entry_count);
  const uint8_t* record =
      package->entry_table +
      global_entry_index * LOOM_CMD_PROGRAM_PACKAGE_ENTRY_SIZE;
  const uint32_t name_offset = iree_unaligned_load_le_u32(
      record + LOOM_CMD_PROGRAM_PACKAGE_ENTRY_NAME_OFFSET_OFFSET);
  const uint32_t name_length = iree_unaligned_load_le_u32(
      record + LOOM_CMD_PROGRAM_PACKAGE_ENTRY_NAME_LENGTH_OFFSET);
  IREE_ASSERT_LE(name_offset, package->strings.data_length);
  IREE_ASSERT_LE(name_length, package->strings.data_length - name_offset);
  return (loom_cmd_program_package_entry_t){
      .executable_index = iree_unaligned_load_le_u32(
          record + LOOM_CMD_PROGRAM_PACKAGE_ENTRY_EXECUTABLE_INDEX_OFFSET),
      .name = iree_make_string_view(
          (const char*)package->strings.data + name_offset, name_length),
  };
}

static bool loom_cmd_program_package_command_dispatch_pair(
    const loom_cmd_program_command_t* command, uint32_t* out_executable_index,
    uint32_t* out_entry_index) {
  switch (loom_cmd_program_command_kind_base(command->kind)) {
    case LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT:
      *out_executable_index = command->payload.dispatch_direct.executable_index;
      *out_entry_index = command->payload.dispatch_direct.entry_index;
      return true;
    case LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_STATIC:
    case LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_DYNAMIC:
      *out_executable_index =
          command->payload.dispatch_indirect.executable_index;
      *out_entry_index = command->payload.dispatch_indirect.entry_index;
      return true;
    default:
      return false;
  }
}

static iree_status_t loom_cmd_program_package_validate_source_dispatches(
    const loom_cmd_program_package_source_export_t* program_export) {
  for (uint32_t i = 0; i < program_export->program->commands.count; ++i) {
    const loom_cmd_program_command_t command =
        loom_cmd_program_command_at(program_export->program, i);
    uint32_t executable_index = 0;
    uint32_t entry_index = 0;
    if (!loom_cmd_program_package_command_dispatch_pair(
            &command, &executable_index, &entry_index)) {
      continue;
    }
    if (program_export->entries[entry_index].executable_index !=
        executable_index) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "command package export '%.*s' dispatch %" PRIu32
          " pairs executable %" PRIu32 " with entry %" PRIu32
          " provided by executable %" PRIu32,
          (int)program_export->name.size, program_export->name.data, i,
          executable_index, entry_index,
          program_export->entries[entry_index].executable_index);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_cmd_program_package_analyze_sources(
    const loom_cmd_program_package_source_export_t* exports,
    iree_host_size_t export_count, uint32_t* out_entry_count,
    uint32_t* out_string_length, uint32_t* out_payload_length,
    loom_cmd_program_package_format_layout_t* out_layout) {
  if (exports == NULL || export_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command package requires at least one export");
  }
  if (export_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command package export count exceeds u32");
  }

  iree_host_size_t entry_count = 0;
  iree_host_size_t string_length = 0;
  iree_host_size_t payload_length = 0;
  for (iree_host_size_t i = 0; i < export_count; ++i) {
    const loom_cmd_program_package_source_export_t* program_export =
        &exports[i];
    if (program_export->name.data == NULL ||
        iree_string_view_is_empty(program_export->name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "command package export %" PRIhsz " has no name",
                              i);
    }
    for (iree_host_size_t j = 0; j < i; ++j) {
      if (iree_string_view_equal(program_export->name, exports[j].name)) {
        return iree_make_status(
            IREE_STATUS_ALREADY_EXISTS,
            "command package export name '%.*s' is duplicated",
            (int)program_export->name.size, program_export->name.data);
      }
    }
    if (program_export->program == NULL ||
        program_export->program->storage.data == NULL ||
        program_export->program->storage.data_length == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "command package export '%.*s' has no program",
                              (int)program_export->name.size,
                              program_export->name.data);
    }
    if (program_export->entry_count !=
        program_export->program->requirements.entry_count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "command package export '%.*s' has %" PRIhsz
          " entries but its program requires %" PRIu32,
          (int)program_export->name.size, program_export->name.data,
          program_export->entry_count,
          program_export->program->requirements.entry_count);
    }
    if (program_export->entry_count != 0 && program_export->entries == NULL) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "command package export '%.*s' has no entry table",
          (int)program_export->name.size, program_export->name.data);
    }
    if (!iree_host_size_checked_add(string_length, program_export->name.size,
                                    &string_length) ||
        !iree_host_size_checked_add(
            payload_length, program_export->program->storage.data_length,
            &payload_length) ||
        !iree_host_size_checked_add(entry_count, program_export->entry_count,
                                    &entry_count)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "command package contents are too large");
    }
    for (iree_host_size_t j = 0; j < program_export->entry_count; ++j) {
      const loom_cmd_program_package_source_entry_t* entry =
          &program_export->entries[j];
      if (entry->name.data == NULL || iree_string_view_is_empty(entry->name)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "command package export '%.*s' entry %" PRIhsz " has no name",
            (int)program_export->name.size, program_export->name.data, j);
      }
      if (entry->executable_index >=
          program_export->program->requirements.executable_count) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "command package export '%.*s' entry %" PRIhsz
            " selects executable %" PRIu32 " outside the requirement table",
            (int)program_export->name.size, program_export->name.data, j,
            entry->executable_index);
      }
      if (!iree_host_size_checked_add(string_length, entry->name.size,
                                      &string_length)) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "command package strings are too large");
      }
    }
    IREE_RETURN_IF_ERROR(
        loom_cmd_program_package_validate_source_dispatches(program_export));
  }

  if (entry_count > UINT32_MAX || string_length > UINT32_MAX ||
      payload_length > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command package contents exceed u32");
  }
  *out_entry_count = (uint32_t)entry_count;
  *out_string_length = (uint32_t)string_length;
  *out_payload_length = (uint32_t)payload_length;
  return loom_cmd_program_package_format_calculate_layout(
      (uint32_t)export_count, (uint32_t)entry_count, (uint32_t)string_length,
      (uint32_t)payload_length, out_layout);
}

iree_status_t loom_cmd_program_package_build(
    const loom_cmd_program_package_source_export_t* exports,
    iree_host_size_t export_count, iree_allocator_t allocator,
    iree_byte_span_t* out_data, loom_cmd_program_package_t* out_package) {
  IREE_ASSERT_ARGUMENT(out_data);
  IREE_ASSERT_ARGUMENT(out_package);
  *out_data = iree_byte_span_empty();
  *out_package = (loom_cmd_program_package_t){0};

  uint32_t entry_count = 0;
  uint32_t string_length = 0;
  uint32_t payload_length = 0;
  loom_cmd_program_package_format_layout_t layout = {0};
  IREE_RETURN_IF_ERROR(loom_cmd_program_package_analyze_sources(
      exports, export_count, &entry_count, &string_length, &payload_length,
      &layout));

  uint8_t* data = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array_uninitialized(
      allocator, layout.total_length, 1, (void**)&data));
  memcpy(data + LOOM_CMD_PROGRAM_PACKAGE_HEADER_MAGIC_OFFSET,
         LOOM_CMD_PROGRAM_PACKAGE_FORMAT_MAGIC,
         LOOM_CMD_PROGRAM_PACKAGE_FORMAT_MAGIC_LENGTH);
  iree_unaligned_store_le_u16(
      data + LOOM_CMD_PROGRAM_PACKAGE_HEADER_VERSION_OFFSET,
      LOOM_CMD_PROGRAM_PACKAGE_FORMAT_VERSION);
  iree_unaligned_store_le_u16(
      data + LOOM_CMD_PROGRAM_PACKAGE_HEADER_SIZE_OFFSET,
      LOOM_CMD_PROGRAM_PACKAGE_HEADER_SIZE);
  iree_unaligned_store_le_u32(
      data + LOOM_CMD_PROGRAM_PACKAGE_HEADER_TOTAL_LENGTH_OFFSET,
      layout.total_length);
  iree_unaligned_store_le_u32(
      data + LOOM_CMD_PROGRAM_PACKAGE_HEADER_EXPORT_COUNT_OFFSET,
      (uint32_t)export_count);
  iree_unaligned_store_le_u32(
      data + LOOM_CMD_PROGRAM_PACKAGE_HEADER_ENTRY_COUNT_OFFSET, entry_count);
  iree_unaligned_store_le_u32(
      data + LOOM_CMD_PROGRAM_PACKAGE_HEADER_EXPORT_TABLE_OFFSET,
      layout.export_offset);
  iree_unaligned_store_le_u32(
      data + LOOM_CMD_PROGRAM_PACKAGE_HEADER_ENTRY_TABLE_OFFSET,
      layout.entry_offset);
  iree_unaligned_store_le_u32(
      data + LOOM_CMD_PROGRAM_PACKAGE_HEADER_STRING_TABLE_OFFSET,
      layout.string_offset);
  iree_unaligned_store_le_u32(
      data + LOOM_CMD_PROGRAM_PACKAGE_HEADER_STRING_TABLE_LENGTH_OFFSET,
      string_length);
  iree_unaligned_store_le_u32(
      data + LOOM_CMD_PROGRAM_PACKAGE_HEADER_PAYLOAD_OFFSET,
      layout.payload_offset);
  iree_unaligned_store_le_u32(
      data + LOOM_CMD_PROGRAM_PACKAGE_HEADER_PAYLOAD_LENGTH_OFFSET,
      payload_length);

  uint8_t* export_record = data + layout.export_offset;
  uint8_t* entry_record = data + layout.entry_offset;
  uint32_t next_entry = 0;
  uint32_t next_string = 0;
  uint32_t next_payload = layout.payload_offset;
  for (iree_host_size_t i = 0; i < export_count; ++i) {
    const loom_cmd_program_package_source_export_t* program_export =
        &exports[i];
    iree_unaligned_store_le_u32(
        export_record + LOOM_CMD_PROGRAM_PACKAGE_EXPORT_NAME_OFFSET_OFFSET,
        next_string);
    iree_unaligned_store_le_u32(
        export_record + LOOM_CMD_PROGRAM_PACKAGE_EXPORT_NAME_LENGTH_OFFSET,
        (uint32_t)program_export->name.size);
    iree_unaligned_store_le_u32(
        export_record + LOOM_CMD_PROGRAM_PACKAGE_EXPORT_FIRST_ENTRY_OFFSET,
        next_entry);
    iree_unaligned_store_le_u32(
        export_record + LOOM_CMD_PROGRAM_PACKAGE_EXPORT_ENTRY_COUNT_OFFSET,
        (uint32_t)program_export->entry_count);
    iree_unaligned_store_le_u32(
        export_record + LOOM_CMD_PROGRAM_PACKAGE_EXPORT_PROGRAM_OFFSET_OFFSET,
        next_payload);
    iree_unaligned_store_le_u32(
        export_record + LOOM_CMD_PROGRAM_PACKAGE_EXPORT_PROGRAM_LENGTH_OFFSET,
        (uint32_t)program_export->program->storage.data_length);
    memcpy(data + layout.string_offset + next_string, program_export->name.data,
           program_export->name.size);
    next_string += (uint32_t)program_export->name.size;
    memcpy(data + next_payload, program_export->program->storage.data,
           program_export->program->storage.data_length);
    next_payload += (uint32_t)program_export->program->storage.data_length;
    for (iree_host_size_t j = 0; j < program_export->entry_count; ++j) {
      const loom_cmd_program_package_source_entry_t* entry =
          &program_export->entries[j];
      iree_unaligned_store_le_u32(
          entry_record + LOOM_CMD_PROGRAM_PACKAGE_ENTRY_EXECUTABLE_INDEX_OFFSET,
          entry->executable_index);
      iree_unaligned_store_le_u32(
          entry_record + LOOM_CMD_PROGRAM_PACKAGE_ENTRY_NAME_OFFSET_OFFSET,
          next_string);
      iree_unaligned_store_le_u32(
          entry_record + LOOM_CMD_PROGRAM_PACKAGE_ENTRY_NAME_LENGTH_OFFSET,
          (uint32_t)entry->name.size);
      memcpy(data + layout.string_offset + next_string, entry->name.data,
             entry->name.size);
      next_string += (uint32_t)entry->name.size;
      ++next_entry;
      entry_record += LOOM_CMD_PROGRAM_PACKAGE_ENTRY_SIZE;
    }
    export_record += LOOM_CMD_PROGRAM_PACKAGE_EXPORT_SIZE;
  }

  *out_data = iree_make_byte_span(data, layout.total_length);
  loom_cmd_program_package_bind(
      iree_make_const_byte_span(data, layout.total_length),
      (uint32_t)export_count, entry_count, string_length, payload_length,
      &layout, out_package);
  return iree_ok_status();
}

static iree_status_t loom_cmd_program_package_validate_contents(
    const loom_cmd_program_package_format_layout_t* layout,
    loom_cmd_program_package_t* package) {
  uint32_t expected_entry = 0;
  uint32_t expected_string = 0;
  uint32_t expected_payload = layout->payload_offset;
  for (uint32_t i = 0; i < package->export_count; ++i) {
    const uint8_t* record =
        package->export_table + i * LOOM_CMD_PROGRAM_PACKAGE_EXPORT_SIZE;
    const uint32_t name_offset = iree_unaligned_load_le_u32(
        record + LOOM_CMD_PROGRAM_PACKAGE_EXPORT_NAME_OFFSET_OFFSET);
    const uint32_t name_length = iree_unaligned_load_le_u32(
        record + LOOM_CMD_PROGRAM_PACKAGE_EXPORT_NAME_LENGTH_OFFSET);
    const uint32_t first_entry = iree_unaligned_load_le_u32(
        record + LOOM_CMD_PROGRAM_PACKAGE_EXPORT_FIRST_ENTRY_OFFSET);
    const uint32_t entry_count = iree_unaligned_load_le_u32(
        record + LOOM_CMD_PROGRAM_PACKAGE_EXPORT_ENTRY_COUNT_OFFSET);
    const uint32_t program_offset = iree_unaligned_load_le_u32(
        record + LOOM_CMD_PROGRAM_PACKAGE_EXPORT_PROGRAM_OFFSET_OFFSET);
    const uint32_t program_length = iree_unaligned_load_le_u32(
        record + LOOM_CMD_PROGRAM_PACKAGE_EXPORT_PROGRAM_LENGTH_OFFSET);

    if (name_length == 0 || name_offset != expected_string ||
        name_length > package->strings.data_length - expected_string) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "command package export %" PRIu32 " has a noncanonical name", i);
    }
    expected_string += name_length;
    if (first_entry != expected_entry ||
        entry_count > package->entry_count - expected_entry) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "command package export %" PRIu32
                              " has a noncanonical entry range",
                              i);
    }
    if (program_length == 0 || program_offset != expected_payload ||
        program_length > package->storage.data_length - expected_payload) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "command package export %" PRIu32
                              " has a noncanonical command payload",
                              i);
    }
    expected_payload += program_length;
    iree_const_byte_span_t program_data = iree_const_byte_span_empty();
    const loom_cmd_program_package_export_t program_export =
        loom_cmd_program_package_bind_export_at(package, i, &program_data);
    for (uint32_t j = 0; j < i; ++j) {
      const loom_cmd_program_package_export_t previous_export =
          loom_cmd_program_package_export_at(package, j);
      if (iree_string_view_equal(program_export.name, previous_export.name)) {
        return iree_make_status(
            IREE_STATUS_ALREADY_EXISTS,
            "command package export name '%.*s' is duplicated",
            (int)program_export.name.size, program_export.name.data);
      }
    }

    loom_cmd_program_t program = {0};
    IREE_RETURN_IF_ERROR(loom_cmd_program_parse(program_data, &program));
    if (entry_count != program.requirements.entry_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "command package export '%.*s' has %" PRIu32
                              " entries but its program requires %" PRIu32,
                              (int)program_export.name.size,
                              program_export.name.data, entry_count,
                              program.requirements.entry_count);
    }
    for (uint32_t j = 0; j < entry_count; ++j) {
      const uint8_t* entry_record =
          package->entry_table +
          expected_entry * LOOM_CMD_PROGRAM_PACKAGE_ENTRY_SIZE;
      const uint32_t executable_index = iree_unaligned_load_le_u32(
          entry_record +
          LOOM_CMD_PROGRAM_PACKAGE_ENTRY_EXECUTABLE_INDEX_OFFSET);
      const uint32_t entry_name_offset = iree_unaligned_load_le_u32(
          entry_record + LOOM_CMD_PROGRAM_PACKAGE_ENTRY_NAME_OFFSET_OFFSET);
      const uint32_t entry_name_length = iree_unaligned_load_le_u32(
          entry_record + LOOM_CMD_PROGRAM_PACKAGE_ENTRY_NAME_LENGTH_OFFSET);
      if (entry_name_length == 0 || entry_name_offset != expected_string ||
          entry_name_length > package->strings.data_length - expected_string) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "command package export '%.*s' entry %" PRIu32
                                " has a noncanonical name",
                                (int)program_export.name.size,
                                program_export.name.data, j);
      }
      if (executable_index >= program.requirements.executable_count) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "command package export '%.*s' entry %" PRIu32
                                " selects executable %" PRIu32
                                " outside the requirement table",
                                (int)program_export.name.size,
                                program_export.name.data, j, executable_index);
      }
      expected_string += entry_name_length;
      ++expected_entry;
    }
    for (uint32_t j = 0; j < program.commands.count; ++j) {
      const loom_cmd_program_command_t command =
          loom_cmd_program_command_at(&program, j);
      uint32_t executable_index = 0;
      uint32_t entry_index = 0;
      if (!loom_cmd_program_package_command_dispatch_pair(
              &command, &executable_index, &entry_index)) {
        continue;
      }
      const loom_cmd_program_package_entry_t entry =
          loom_cmd_program_package_export_entry_at(package, &program_export,
                                                   entry_index);
      if (entry.executable_index != executable_index) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "command package export '%.*s' dispatch %" PRIu32
            " pairs executable %" PRIu32 " with entry %" PRIu32
            " provided by executable %" PRIu32,
            (int)program_export.name.size, program_export.name.data, j,
            executable_index, entry_index, entry.executable_index);
      }
    }
  }

  if (expected_entry != package->entry_count ||
      expected_string != package->strings.data_length ||
      expected_payload != package->storage.data_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command package has unreferenced trailing data");
  }
  return iree_ok_status();
}

iree_status_t loom_cmd_program_package_parse(
    iree_const_byte_span_t data, loom_cmd_program_package_t* out_package) {
  IREE_ASSERT_ARGUMENT(out_package);
  *out_package = (loom_cmd_program_package_t){0};
  if (data.data == NULL ||
      data.data_length < LOOM_CMD_PROGRAM_PACKAGE_HEADER_SIZE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command package is shorter than its header");
  }
  if (memcmp(data.data, LOOM_CMD_PROGRAM_PACKAGE_FORMAT_MAGIC,
             LOOM_CMD_PROGRAM_PACKAGE_FORMAT_MAGIC_LENGTH) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command package has invalid magic bytes");
  }
  const uint16_t version = iree_unaligned_load_le_u16(
      data.data + LOOM_CMD_PROGRAM_PACKAGE_HEADER_VERSION_OFFSET);
  if (version != LOOM_CMD_PROGRAM_PACKAGE_FORMAT_VERSION) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "command package format version %" PRIu16 " is unsupported", version);
  }
  const uint16_t header_size = iree_unaligned_load_le_u16(
      data.data + LOOM_CMD_PROGRAM_PACKAGE_HEADER_SIZE_OFFSET);
  if (header_size != LOOM_CMD_PROGRAM_PACKAGE_HEADER_SIZE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command package header size %" PRIu16
                            " is not canonical",
                            header_size);
  }
  const uint32_t total_length = iree_unaligned_load_le_u32(
      data.data + LOOM_CMD_PROGRAM_PACKAGE_HEADER_TOTAL_LENGTH_OFFSET);
  if (total_length != data.data_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command package declares %" PRIu32
                            " bytes but received %" PRIhsz,
                            total_length, data.data_length);
  }
  const uint32_t export_count = iree_unaligned_load_le_u32(
      data.data + LOOM_CMD_PROGRAM_PACKAGE_HEADER_EXPORT_COUNT_OFFSET);
  const uint32_t entry_count = iree_unaligned_load_le_u32(
      data.data + LOOM_CMD_PROGRAM_PACKAGE_HEADER_ENTRY_COUNT_OFFSET);
  const uint32_t string_length = iree_unaligned_load_le_u32(
      data.data + LOOM_CMD_PROGRAM_PACKAGE_HEADER_STRING_TABLE_LENGTH_OFFSET);
  const uint32_t payload_length = iree_unaligned_load_le_u32(
      data.data + LOOM_CMD_PROGRAM_PACKAGE_HEADER_PAYLOAD_LENGTH_OFFSET);
  if (export_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command package has no exports");
  }

  loom_cmd_program_package_format_layout_t layout = {0};
  IREE_RETURN_IF_ERROR(loom_cmd_program_package_format_calculate_layout(
      export_count, entry_count, string_length, payload_length, &layout));
  if (layout.total_length != total_length ||
      iree_unaligned_load_le_u32(
          data.data + LOOM_CMD_PROGRAM_PACKAGE_HEADER_EXPORT_TABLE_OFFSET) !=
          layout.export_offset ||
      iree_unaligned_load_le_u32(
          data.data + LOOM_CMD_PROGRAM_PACKAGE_HEADER_ENTRY_TABLE_OFFSET) !=
          layout.entry_offset ||
      iree_unaligned_load_le_u32(
          data.data + LOOM_CMD_PROGRAM_PACKAGE_HEADER_STRING_TABLE_OFFSET) !=
          layout.string_offset ||
      iree_unaligned_load_le_u32(
          data.data + LOOM_CMD_PROGRAM_PACKAGE_HEADER_PAYLOAD_OFFSET) !=
          layout.payload_offset) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command package table layout is not canonical");
  }

  loom_cmd_program_package_bind(data, export_count, entry_count, string_length,
                                payload_length, &layout, out_package);
  const iree_status_t status =
      loom_cmd_program_package_validate_contents(&layout, out_package);
  if (!iree_status_is_ok(status)) {
    *out_package = (loom_cmd_program_package_t){0};
  }
  return status;
}
