// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/artifact_set.h"

#include <string.h>

#include "loom/ir/module.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/target/arch/cmd/lower/serialize.h"
#include "loom/util/json.h"

static iree_string_view_t loom_cmd_program_artifact_set_symbol_name(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref) {
  IREE_ASSERT(loom_symbol_ref_is_valid(symbol_ref));
  IREE_ASSERT_EQ(symbol_ref.module_id, 0u);
  IREE_ASSERT_LT(symbol_ref.symbol_id, module->symbols.count);
  const loom_string_id_t name_id =
      module->symbols.entries[symbol_ref.symbol_id].name_id;
  IREE_ASSERT_LT(name_id, module->strings.count);
  return module->strings.entries[name_id];
}

static iree_string_view_t loom_cmd_program_artifact_set_copy_string(
    iree_string_view_t source, char** storage_cursor) {
  char* target = *storage_cursor;
  memcpy(target, source.data, source.size);
  *storage_cursor += source.size;
  return iree_make_string_view(target, source.size);
}

iree_status_t loom_cmd_program_artifact_set_build(
    const loom_cmd_program_plan_t* plan,
    loom_cmd_program_artifact_set_t* out_artifact_set,
    iree_allocator_t host_allocator) {
  if (plan == NULL || plan->root_module == NULL || plan->root_count == 0 ||
      out_artifact_set == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "a prepared command program plan is required");
  }
  *out_artifact_set = (loom_cmd_program_artifact_set_t){0};

  iree_host_size_t string_storage_size = 0;
  iree_host_size_t entry_projection_count = 0;
  for (iree_host_size_t i = 0; i < plan->root_count; ++i) {
    const loom_func_like_t root =
        loom_func_like_cast(plan->root_module, plan->roots[i].function_op);
    IREE_ASSERT(loom_func_like_isa(root));
    const iree_string_view_t name = loom_cmd_program_artifact_set_symbol_name(
        plan->root_module, loom_func_like_callee(root));
    if (!iree_host_size_checked_add(string_storage_size, name.size,
                                    &string_storage_size) ||
        !iree_host_size_checked_add(entry_projection_count,
                                    plan->roots[i].entry_requirement_count,
                                    &entry_projection_count)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "command artifact metadata is too large");
    }
  }
  for (iree_host_size_t i = 0; i < plan->entry_requirement_count; ++i) {
    const loom_op_t* declaration_op =
        plan->entry_requirements[i].declaration_op;
    IREE_ASSERT(loom_kernel_entry_decl_isa(declaration_op));
    const iree_string_view_t name = loom_cmd_program_artifact_set_symbol_name(
        plan->root_module, loom_kernel_entry_decl_callee(declaration_op));
    if (!iree_host_size_checked_add(string_storage_size, name.size,
                                    &string_storage_size)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "command artifact symbol storage is too large");
    }
  }

  loom_cmd_program_artifact_set_t artifact_set = {
      .host_allocator = host_allocator,
  };
  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, plan->root_count, sizeof(*artifact_set.programs.values),
      (void**)&artifact_set.programs.values);
  if (iree_status_is_ok(status)) {
    memset(artifact_set.programs.values, 0,
           plan->root_count * sizeof(*artifact_set.programs.values));
    artifact_set.programs.count = plan->root_count;
  }
  if (iree_status_is_ok(status) && plan->entry_requirement_count != 0) {
    status = iree_allocator_malloc_array(host_allocator,
                                         plan->entry_requirement_count,
                                         sizeof(*artifact_set.entries.values),
                                         (void**)&artifact_set.entries.values);
    if (iree_status_is_ok(status)) {
      artifact_set.entries.count = plan->entry_requirement_count;
    }
  }
  if (iree_status_is_ok(status) && entry_projection_count != 0) {
    status = iree_allocator_malloc_array(
        host_allocator, entry_projection_count,
        sizeof(*artifact_set.entry_requirement_index_storage),
        (void**)&artifact_set.entry_requirement_index_storage);
  }
  if (iree_status_is_ok(status) && string_storage_size != 0) {
    status = iree_allocator_malloc(host_allocator, string_storage_size,
                                   (void**)&artifact_set.string_storage);
  }

  char* string_cursor = artifact_set.string_storage;
  uint32_t* entry_projection_cursor =
      artifact_set.entry_requirement_index_storage;
  for (iree_host_size_t i = 0;
       i < plan->entry_requirement_count && iree_status_is_ok(status); ++i) {
    const loom_op_t* declaration_op =
        plan->entry_requirements[i].declaration_op;
    const iree_string_view_t name = loom_cmd_program_artifact_set_symbol_name(
        plan->root_module, loom_kernel_entry_decl_callee(declaration_op));
    artifact_set.entries.values[i].symbol =
        loom_cmd_program_artifact_set_copy_string(name, &string_cursor);
    artifact_set.entries.values[i].has_source_request =
        plan->entry_requirements[i].has_source_request;
  }
  for (iree_host_size_t i = 0;
       i < plan->root_count && iree_status_is_ok(status); ++i) {
    const loom_cmd_program_root_t* root = &plan->roots[i];
    loom_cmd_program_artifact_t* artifact = &artifact_set.programs.values[i];
    const loom_func_like_t root_function =
        loom_func_like_cast(plan->root_module, root->function_op);
    const iree_string_view_t name = loom_cmd_program_artifact_set_symbol_name(
        plan->root_module, loom_func_like_callee(root_function));
    artifact->symbol =
        loom_cmd_program_artifact_set_copy_string(name, &string_cursor);
    artifact->entry_requirement_indices = entry_projection_cursor;
    artifact->entry_requirement_count = root->entry_requirement_count;
    if (root->entry_requirement_count != 0) {
      memcpy(entry_projection_cursor, root->entry_requirement_indices,
             root->entry_requirement_count * sizeof(*entry_projection_cursor));
      entry_projection_cursor += root->entry_requirement_count;
    }
    status = loom_cmd_program_plan_serialize_root(plan, i, &artifact->data,
                                                  host_allocator);
  }

  if (iree_status_is_ok(status)) {
    *out_artifact_set = artifact_set;
  } else {
    loom_cmd_program_artifact_set_deinitialize(&artifact_set);
  }
  return status;
}

void loom_cmd_program_artifact_set_deinitialize(
    loom_cmd_program_artifact_set_t* artifact_set) {
  if (artifact_set == NULL) {
    return;
  }
  for (iree_host_size_t i = 0; i < artifact_set->programs.count; ++i) {
    iree_allocator_free(artifact_set->host_allocator,
                        artifact_set->programs.values[i].data.data);
  }
  iree_allocator_free(artifact_set->host_allocator,
                      artifact_set->entry_requirement_index_storage);
  iree_allocator_free(artifact_set->host_allocator,
                      artifact_set->string_storage);
  iree_allocator_free(artifact_set->host_allocator,
                      artifact_set->entries.values);
  iree_allocator_free(artifact_set->host_allocator,
                      artifact_set->programs.values);
  *artifact_set = (loom_cmd_program_artifact_set_t){0};
}

iree_status_t loom_cmd_program_artifact_format_filename(
    iree_host_size_t program_ordinal, iree_host_size_t buffer_capacity,
    char* buffer, iree_string_view_t* out_filename) {
  if (buffer == NULL || out_filename == NULL || buffer_capacity == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command artifact filename storage is required");
  }
  *out_filename = iree_string_view_empty();
  const int length = iree_snprintf(
      buffer, buffer_capacity,
      "program-%" PRIhsz LOOM_CMD_PROGRAM_ARTIFACT_EXTENSION, program_ordinal);
  if (length < 0 || (iree_host_size_t)length >= buffer_capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "command artifact filename is too large");
  }
  *out_filename = iree_make_string_view(buffer, (iree_host_size_t)length);
  return iree_ok_status();
}

iree_status_t loom_cmd_kernel_request_format_filename(
    iree_host_size_t entry_ordinal, iree_host_size_t buffer_capacity,
    char* buffer, iree_string_view_t* out_filename) {
  if (buffer == NULL || out_filename == NULL || buffer_capacity == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel request filename storage is required");
  }
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

static iree_status_t loom_cmd_program_artifact_set_format_program_json(
    const loom_cmd_program_artifact_t* program,
    iree_host_size_t program_ordinal, loom_output_stream_t* stream) {
  char filename_storage[LOOM_CMD_PROGRAM_ARTIFACT_FILENAME_CAPACITY];
  iree_string_view_t filename = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_cmd_program_artifact_format_filename(
      program_ordinal, sizeof(filename_storage), filename_storage, &filename));

  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("symbol"), program->symbol));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("artifact"), filename));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("byte_length"), program->data.data_length));
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

iree_status_t loom_cmd_program_artifact_set_format_manifest_json(
    const loom_cmd_program_artifact_set_t* artifact_set,
    loom_output_stream_t* stream) {
  if (artifact_set == NULL || stream == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command artifact set and stream are required");
  }

  loom_json_object_writer_t root;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &root));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &root, IREE_SV("schema_version"),
      LOOM_CMD_PROGRAM_ARTIFACT_SET_SCHEMA_VERSION));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &root, IREE_SV("format"),
      IREE_SV(LOOM_CMD_PROGRAM_ARTIFACT_SET_MANIFEST_FORMAT)));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&root, IREE_SV("programs")));
  loom_json_array_writer_t programs;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &programs));
  for (iree_host_size_t i = 0; i < artifact_set->programs.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&programs));
    IREE_RETURN_IF_ERROR(loom_cmd_program_artifact_set_format_program_json(
        &artifact_set->programs.values[i], i, stream));
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&programs));
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&root, IREE_SV("entries")));
  loom_json_array_writer_t entries;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &entries));
  for (iree_host_size_t i = 0; i < artifact_set->entries.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&entries));
    loom_json_object_writer_t entry;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &entry));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &entry, IREE_SV("symbol"), artifact_set->entries.values[i].symbol));
    if (artifact_set->entries.values[i].has_source_request) {
      char filename_storage[LOOM_CMD_KERNEL_REQUEST_FILENAME_CAPACITY];
      iree_string_view_t filename = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(loom_cmd_kernel_request_format_filename(
          i, sizeof(filename_storage), filename_storage, &filename));
      IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
          &entry, IREE_SV("source_request"), filename));
    }
    IREE_RETURN_IF_ERROR(loom_json_object_end(&entry));
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&entries));
  return loom_json_object_end(&root);
}
