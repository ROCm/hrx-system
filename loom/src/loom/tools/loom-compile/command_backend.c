// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-compile/command_backend.h"

#include "iree/io/stdio_stream.h"
#include "loom/codegen/low/repr.h"
#include "loom/format/bytecode/writer.h"
#include "loom/format/low_repr.h"
#include "loom/link/module_index.h"
#include "loom/ops/command/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/pass/builtin_registry.h"
#include "loom/target/arch/cmd/artifact_builder.h"
#include "loom/target/arch/cmd/artifact_set.h"
#include "loom/target/entry_selection.h"
#include "loom/tooling/io/file.h"
#include "loom/util/json.h"
#include "loom/util/stream.h"

// Current loom-compile command manifest schema version.
#define LOOM_COMPILE_COMMAND_MANIFEST_SCHEMA_VERSION 2

// Storage for canonical ordinal-derived artifact filenames.
#define LOOM_COMPILE_COMMAND_FILENAME_CAPACITY 64

static iree_status_t loom_compile_command_backend_format_program_filename(
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

static iree_status_t
loom_compile_command_backend_format_kernel_request_filename(
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

static iree_status_t loom_compile_command_backend_format_program_json(
    const loom_cmd_program_artifact_t* program,
    iree_host_size_t program_ordinal, loom_output_stream_t* stream) {
  char filename_storage[LOOM_COMPILE_COMMAND_FILENAME_CAPACITY];
  iree_string_view_t filename = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_compile_command_backend_format_program_filename(
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

static iree_status_t loom_compile_command_backend_format_manifest_json(
    const loom_cmd_program_artifact_set_t* artifact_set,
    loom_output_stream_t* stream) {
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
    IREE_RETURN_IF_ERROR(loom_compile_command_backend_format_program_json(
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
      char filename_storage[LOOM_COMPILE_COMMAND_FILENAME_CAPACITY];
      iree_string_view_t filename = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(
          loom_compile_command_backend_format_kernel_request_filename(
              i, sizeof(filename_storage), filename_storage, &filename));
      IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
          &entry, IREE_SV("source_request"), filename));
    }
    IREE_RETURN_IF_ERROR(loom_json_object_end(&entry));
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&entries));
  return loom_json_object_end(&root);
}

static bool loom_compile_command_backend_is_root(const loom_op_t* op) {
  return loom_command_program_def_isa(op) &&
         (loom_command_program_def_visibility(op) ==
              LOOM_COMMAND_VISIBILITY_PUBLIC ||
          loom_command_program_def_retain(op) == LOOM_COMMAND_RETAIN_RETAIN);
}

static iree_host_size_t loom_compile_command_backend_count_roots(
    loom_module_t* module) {
  iree_host_size_t root_count = 0;
  loom_op_t* op = NULL;
  loom_block_for_each_op(loom_module_block(module), op) {
    if (loom_compile_command_backend_is_root(op)) {
      ++root_count;
    }
  }
  return root_count;
}

static iree_status_t loom_compile_command_backend_require_directory(
    iree_string_view_t path, iree_allocator_t host_allocator) {
  IREE_RETURN_IF_ERROR(
      loom_tooling_create_directory_if_needed(path, host_allocator));
  bool is_directory = false;
  IREE_RETURN_IF_ERROR(
      loom_tooling_file_path_is_directory(path, host_allocator, &is_directory));
  if (!is_directory) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "artifact path '%.*s' is not a directory",
                            (int)path.size, path.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_compile_command_backend_index_roots(
    const loom_link_module_index_module_t* indexed_module,
    loom_module_t* module, iree_host_size_t root_count,
    iree_arena_allocator_t* scratch_arena,
    iree_host_size_t** out_root_symbol_ordinals) {
  *out_root_symbol_ordinals = NULL;
  iree_host_size_t* root_symbol_ordinals = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, root_count, sizeof(*root_symbol_ordinals),
      (void**)&root_symbol_ordinals));

  iree_host_size_t root_ordinal = 0;
  loom_op_t* op = NULL;
  loom_block_for_each_op(loom_module_block(module), op) {
    if (!loom_compile_command_backend_is_root(op)) {
      continue;
    }
    const loom_symbol_ref_t callee = loom_command_program_def_callee(op);
    IREE_ASSERT(loom_symbol_ref_is_valid(callee));
    IREE_ASSERT_EQ(callee.module_id, 0u);
    IREE_ASSERT_LT(callee.symbol_id, indexed_module->symbol_count);
    root_symbol_ordinals[root_ordinal++] =
        indexed_module->symbol_start_ordinal + callee.symbol_id;
  }
  IREE_ASSERT_EQ(root_ordinal, root_count);
  *out_root_symbol_ordinals = root_symbol_ordinals;
  return iree_ok_status();
}

static iree_status_t loom_compile_command_backend_write_programs(
    const loom_cmd_program_artifact_set_t* artifact_set,
    iree_string_view_t artifact_directory, iree_allocator_t host_allocator) {
  IREE_RETURN_IF_ERROR(loom_compile_command_backend_require_directory(
      artifact_directory, host_allocator));

  for (iree_host_size_t i = 0; i < artifact_set->programs.count; ++i) {
    char filename_storage[LOOM_COMPILE_COMMAND_FILENAME_CAPACITY];
    iree_string_view_t filename = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_compile_command_backend_format_program_filename(
        i, sizeof(filename_storage), filename_storage, &filename));
    char* artifact_path_storage = NULL;
    IREE_RETURN_IF_ERROR(loom_tooling_file_path_join(
        artifact_directory, filename, host_allocator, &artifact_path_storage));
    const iree_string_view_t artifact_path =
        iree_make_cstring_view(artifact_path_storage);
    const iree_byte_span_t data = artifact_set->programs.values[i].data;
    iree_status_t status = loom_tooling_write_output_file(
        artifact_path,
        iree_make_string_view((const char*)data.data, data.data_length),
        host_allocator);
    iree_allocator_free(host_allocator, artifact_path_storage);
    IREE_RETURN_IF_ERROR(status);
  }
  return iree_ok_status();
}

static iree_status_t loom_compile_command_backend_write_manifest(
    const loom_cmd_program_artifact_set_t* artifact_set,
    iree_string_view_t manifest_path, iree_allocator_t host_allocator) {
  loom_tooling_output_stream_t output;
  IREE_RETURN_IF_ERROR(
      loom_tooling_output_stream_open(manifest_path, host_allocator, &output));
  iree_status_t status = loom_compile_command_backend_format_manifest_json(
      artifact_set, &output.stream);
  if (iree_status_is_ok(status)) {
    status = loom_output_stream_write_cstring(&output.stream, "\n");
  }
  return iree_status_join(status, loom_tooling_output_stream_close(&output));
}

typedef struct loom_compile_kernel_request_writer_t {
  // Destination directory for ordinary Loom bytecode request modules.
  iree_string_view_t directory;

  // Stable Low descriptor codec shared with the input session.
  loom_low_repr_environment_t low_repr_environment;

  // Arena block pool used by the streaming bytecode writer.
  iree_arena_block_pool_t* block_pool;

  // Host allocator for paths and file streams.
  iree_allocator_t host_allocator;
} loom_compile_kernel_request_writer_t;

static iree_status_t loom_compile_command_backend_write_kernel_request(
    void* user_data, loom_cmd_program_kernel_request_t request) {
  loom_compile_kernel_request_writer_t* writer =
      (loom_compile_kernel_request_writer_t*)user_data;
  char filename_storage[LOOM_COMPILE_COMMAND_FILENAME_CAPACITY];
  iree_string_view_t filename = iree_string_view_empty();
  iree_status_t status =
      loom_compile_command_backend_format_kernel_request_filename(
          request.entry_requirement_index, sizeof(filename_storage),
          filename_storage, &filename);

  char* path_storage = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_tooling_file_path_join(writer->directory, filename,
                                         writer->host_allocator, &path_storage);
  }
  iree_io_stream_t* stream = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_io_stdio_stream_open(
        IREE_IO_STDIO_STREAM_MODE_WRITE | IREE_IO_STDIO_STREAM_MODE_DISCARD,
        iree_make_cstring_view(path_storage), writer->host_allocator, &stream);
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_write_module(
        request.source.product.module, stream,
        &(loom_bytecode_write_options_t){
            .producer = IREE_SV("loom-compile"),
            .location_mode = LOOM_BYTECODE_LOCATION_MODE_SOURCE_LOCATIONS,
            .low_repr_environment = writer->low_repr_environment,
        },
        writer->block_pool);
  }
  iree_io_stream_release(stream);
  iree_allocator_free(writer->host_allocator, path_storage);
  loom_kernel_class_product_deinitialize(&request.source.product);
  return status;
}

iree_status_t loom_compile_command_backend_emit(
    loom_run_session_t* session, loom_run_module_t* run_module,
    const loom_compile_command_backend_options_t* options, bool* out_emitted,
    iree_allocator_t host_allocator) {
  if (session == NULL || run_module == NULL || run_module->module == NULL ||
      options == NULL || out_emitted == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command backend inputs must be present");
  }
  *out_emitted = false;
  if (iree_string_view_is_empty(options->artifact_directory) ||
      loom_tooling_file_path_is_stdio(options->artifact_directory)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "command backend requires a filesystem artifact directory");
  }
  if (!iree_string_view_is_empty(options->kernel_request_directory) &&
      loom_tooling_file_path_is_stdio(options->kernel_request_directory)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "kernel request artifacts require a filesystem directory");
  }

  const iree_host_size_t root_count =
      loom_compile_command_backend_count_roots(run_module->module);
  if (root_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "command backend requires at least one retained command program root");
  }

  if (!iree_string_view_is_empty(options->kernel_request_directory)) {
    IREE_RETURN_IF_ERROR(loom_compile_command_backend_require_directory(
        options->kernel_request_directory, host_allocator));
  }

  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(loom_run_session_block_pool(session), &scratch_arena);
  loom_link_module_index_t* index = NULL;
  iree_status_t status = loom_link_module_index_allocate(
      run_module->module->context, loom_run_session_block_pool(session),
      host_allocator, &index);
  iree_host_size_t provider_ordinal = 0;
  if (iree_status_is_ok(status)) {
    status = loom_link_module_index_add_materialized(
        index, run_module->module,
        &(loom_link_module_index_add_options_t){
            .provider_name = IREE_SV("loom-compile-command"),
        },
        &provider_ordinal);
  }

  const loom_link_module_index_module_t* indexed_module = NULL;
  if (iree_status_is_ok(status)) {
    const loom_link_module_index_provider_t* provider =
        loom_link_module_index_provider_at(index, provider_ordinal);
    IREE_ASSERT(provider != NULL);
    IREE_ASSERT_EQ(provider->module_count, 1u);
    indexed_module =
        loom_link_module_index_module_at(index, provider->module_start_ordinal);
    IREE_ASSERT(indexed_module != NULL);
  }

  iree_host_size_t* root_symbol_ordinals = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_compile_command_backend_index_roots(
        indexed_module, run_module->module, root_count, &scratch_arena,
        &root_symbol_ordinals);
  }

  const loom_target_entry_options_t diagnostic_options = {
      .diagnostic_sink = options->diagnostic_sink,
      .source_resolver = options->source_resolver,
      .max_errors = options->max_errors,
  };
  loom_target_entry_diagnostic_emitter_t diagnostic_emitter = {0};
  loom_target_entry_diagnostic_emitter_initialize(
      run_module->module, &diagnostic_options, LOOM_EMITTER_PASS,
      &diagnostic_emitter);
  bool plan_valid = false;
  loom_low_repr_environment_t low_repr_environment = {0};
  loom_low_repr_environment_initialize(
      &loom_run_session_low_descriptor_registry(session)->registry,
      &low_repr_environment);
  const loom_link_plan_materialization_environment_t
      materialization_environment = {
          .context = run_module->module->context,
          .block_pool = loom_run_session_block_pool(session),
          .low_repr_environment = low_repr_environment,
          .allocator = host_allocator,
      };
  loom_compile_kernel_request_writer_t kernel_request_writer = {
      .directory = options->kernel_request_directory,
      .low_repr_environment = low_repr_environment,
      .block_pool = loom_run_session_block_pool(session),
      .host_allocator = host_allocator,
  };
  loom_cmd_program_plan_index_options_t plan_options;
  loom_cmd_program_plan_index_options_initialize(&plan_options);
  if (!iree_string_view_is_empty(options->kernel_request_directory)) {
    plan_options.kernel_request_sink = (loom_cmd_program_kernel_request_sink_t){
        .publish = loom_compile_command_backend_write_kernel_request,
        .user_data = &kernel_request_writer,
    };
  }
  loom_cmd_program_artifact_set_t artifact_set = {0};
  if (iree_status_is_ok(status)) {
    status = loom_cmd_program_artifact_set_build_from_index(
        index, root_symbol_ordinals, root_count,
        &(loom_cmd_program_artifact_builder_options_t){
            .plan_options = plan_options.kernel_request_sink.publish != NULL
                                ? &plan_options
                                : NULL,
            .pass_registry = loom_pass_builtin_registry(),
            .diagnostic_emitter =
                loom_target_entry_emitter(&diagnostic_emitter),
            .materialization_environment = &materialization_environment,
        },
        &scratch_arena, &plan_valid, &artifact_set, host_allocator);
  }
  if (iree_status_is_ok(status) && !plan_valid &&
      diagnostic_emitter.error_count == 0) {
    status = iree_make_status(
        IREE_STATUS_INTERNAL,
        "command program preparation failed without a diagnostic");
  }

  if (iree_status_is_ok(status) && plan_valid) {
    status = loom_compile_command_backend_write_programs(
        &artifact_set, options->artifact_directory, host_allocator);
  }
  if (iree_status_is_ok(status) && plan_valid) {
    status = loom_compile_command_backend_write_manifest(
        &artifact_set, options->manifest_path, host_allocator);
  }
  if (iree_status_is_ok(status) && plan_valid) {
    *out_emitted = true;
  }

  loom_cmd_program_artifact_set_deinitialize(&artifact_set);
  loom_link_module_index_free(index);
  iree_arena_deinitialize(&scratch_arena);
  return status;
}
