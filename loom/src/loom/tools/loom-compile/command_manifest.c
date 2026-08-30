// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-compile/command_manifest.h"

#include "loom/util/json.h"

// Current loom-compile command manifest schema version.
#define LOOM_COMPILE_COMMAND_MANIFEST_SCHEMA_VERSION 2

// Storage for canonical ordinal-derived artifact filenames.
#define LOOM_COMPILE_COMMAND_FILENAME_CAPACITY 64

iree_status_t loom_compile_command_manifest_format_program_filename(
    iree_host_size_t program_ordinal, iree_host_size_t buffer_capacity,
    char* buffer, iree_string_view_t* out_filename) {
  *out_filename = iree_string_view_empty();
  const int length = iree_snprintf(
      buffer, buffer_capacity, "program-%" PRIhsz ".loomcmd", program_ordinal);
  if (length < 0 || (iree_host_size_t)length >= buffer_capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "command artifact filename is too large");
  }
  *out_filename = iree_make_string_view(buffer, (iree_host_size_t)length);
  return iree_ok_status();
}

iree_status_t loom_compile_command_manifest_format_kernel_request_filename(
    iree_host_size_t entry_ordinal, iree_host_size_t buffer_capacity,
    char* buffer, iree_string_view_t* out_filename) {
  *out_filename = iree_string_view_empty();
  const int length = iree_snprintf(buffer, buffer_capacity,
                                   "kernel-%" PRIhsz ".loombc", entry_ordinal);
  if (length < 0 || (iree_host_size_t)length >= buffer_capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "kernel request filename is too large");
  }
  *out_filename = iree_make_string_view(buffer, (iree_host_size_t)length);
  return iree_ok_status();
}

static iree_status_t loom_compile_command_manifest_write_program(
    const loom_cmd_program_artifact_t* program,
    iree_host_size_t program_ordinal, loom_output_stream_t* stream) {
  char filename_storage[LOOM_COMPILE_COMMAND_FILENAME_CAPACITY];
  iree_string_view_t filename = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_compile_command_manifest_format_program_filename(
      program_ordinal, sizeof(filename_storage), filename_storage, &filename));

  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("symbol"), program->symbol));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("artifact"), filename));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("byte_length"),
      iree_byte_sequence_length(program->data)));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("entry_requirements")));
  loom_json_array_writer_t entry_requirements;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &entry_requirements));
  for (uint32_t i = 0; i < program->entry_requirement_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_json_array_write_uint32_element(
        &entry_requirements, program->entry_requirement_indices[i]));
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&entry_requirements));
  return loom_json_object_end(&object);
}

iree_status_t loom_compile_command_manifest_write(
    const loom_cmd_program_artifact_set_t* artifact_set,
    const uint32_t* source_requirement_indices,
    iree_host_size_t source_requirement_count, loom_output_stream_t* stream) {
  loom_json_object_writer_t root;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &root));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &root, IREE_SV("schema_version"),
      LOOM_COMPILE_COMMAND_MANIFEST_SCHEMA_VERSION));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &root, IREE_SV("format"), IREE_SV(LOOM_COMPILE_COMMAND_MANIFEST_FORMAT)));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&root, IREE_SV("programs")));
  loom_json_array_writer_t programs;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &programs));
  for (iree_host_size_t i = 0; i < artifact_set->programs.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&programs));
    IREE_RETURN_IF_ERROR(loom_compile_command_manifest_write_program(
        &artifact_set->programs.values[i], i, stream));
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&programs));
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&root, IREE_SV("entries")));
  loom_json_array_writer_t entries;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &entries));
  iree_host_size_t source_requirement_cursor = 0;
  for (iree_host_size_t i = 0; i < artifact_set->entries.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&entries));
    loom_json_object_writer_t entry;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &entry));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &entry, IREE_SV("symbol"), artifact_set->entries.values[i].symbol));
    if (source_requirement_cursor < source_requirement_count &&
        source_requirement_indices[source_requirement_cursor] == i) {
      char filename_storage[LOOM_COMPILE_COMMAND_FILENAME_CAPACITY];
      iree_string_view_t filename = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(
          loom_compile_command_manifest_format_kernel_request_filename(
              i, sizeof(filename_storage), filename_storage, &filename));
      IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
          &entry, IREE_SV("source_request"), filename));
      ++source_requirement_cursor;
    }
    IREE_RETURN_IF_ERROR(loom_json_object_end(&entry));
  }
  IREE_ASSERT_EQ(source_requirement_cursor, source_requirement_count);
  IREE_RETURN_IF_ERROR(loom_json_array_end(&entries));
  return loom_json_object_end(&root);
}
