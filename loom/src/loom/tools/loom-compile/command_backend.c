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
#include "loom/pass/builtin_registry.h"
#include "loom/target/arch/cmd/artifact_builder.h"
#include "loom/target/arch/cmd/artifact_set.h"
#include "loom/target/entry_selection.h"
#include "loom/tooling/io/file.h"
#include "loom/tools/loom-compile/command_manifest.h"
#include "loom/transforms/kernel/kernel_request_producer.h"
#include "loom/util/stream.h"

// Storage for canonical ordinal-derived artifact filenames.
#define LOOM_COMPILE_COMMAND_FILENAME_CAPACITY 64

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

// Selects command roots from source roles retained by the module index. The
// compact indexed-symbol scan replaces source-IR walks and allocates the result
// once at its maximum possible size.
static iree_status_t loom_compile_command_backend_collect_roots(
    const loom_link_module_index_t* index,
    const loom_link_module_index_module_t* indexed_module,
    iree_arena_allocator_t* scratch_arena,
    iree_host_size_t** out_root_symbol_ordinals,
    iree_host_size_t* out_root_count) {
  *out_root_symbol_ordinals = NULL;
  *out_root_count = 0;
  if (indexed_module->symbol_count == 0) {
    return iree_ok_status();
  }

  iree_host_size_t* root_symbol_ordinals = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, indexed_module->symbol_count,
      sizeof(*root_symbol_ordinals), (void**)&root_symbol_ordinals));

  iree_host_size_t root_count = 0;
  for (iree_host_size_t i = 0; i < indexed_module->symbol_count; ++i) {
    const iree_host_size_t symbol_ordinal =
        indexed_module->symbol_start_ordinal + i;
    const loom_link_module_index_symbol_t* symbol =
        loom_link_module_index_symbol_at(index, symbol_ordinal);
    IREE_ASSERT(symbol != NULL);
    if (!iree_any_bit_set(symbol->facets.schema.interfaces,
                          LOOM_SYMBOL_INTERFACE_COMMAND_PROGRAM) ||
        !iree_any_bit_set(symbol->flags,
                          LOOM_LINK_SYMBOL_FLAG_CONCRETE_DEFINITION) ||
        !iree_any_bit_set(symbol->flags, LOOM_LINK_SYMBOL_FLAG_PUBLIC |
                                             LOOM_LINK_SYMBOL_FLAG_RETAIN)) {
      continue;
    }
    root_symbol_ordinals[root_count++] = symbol_ordinal;
  }
  *out_root_symbol_ordinals = root_symbol_ordinals;
  *out_root_count = root_count;
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
    IREE_RETURN_IF_ERROR(loom_compile_command_manifest_format_program_filename(
        i, sizeof(filename_storage), filename_storage, &filename));
    char* artifact_path_storage = NULL;
    IREE_RETURN_IF_ERROR(loom_tooling_file_path_join(
        artifact_directory, filename, host_allocator, &artifact_path_storage));
    const iree_string_view_t artifact_path =
        iree_make_cstring_view(artifact_path_storage);
    iree_status_t status = loom_tooling_write_output_byte_sequence(
        artifact_path, artifact_set->programs.values[i].data, host_allocator);
    iree_allocator_free(host_allocator, artifact_path_storage);
    IREE_RETURN_IF_ERROR(status);
  }
  return iree_ok_status();
}

static iree_status_t loom_compile_command_backend_write_manifest(
    const loom_cmd_program_artifact_set_t* artifact_set,
    const uint32_t* source_requirement_indices,
    iree_host_size_t source_requirement_count, iree_string_view_t manifest_path,
    iree_allocator_t host_allocator) {
  loom_tooling_output_stream_t output;
  IREE_RETURN_IF_ERROR(
      loom_tooling_output_stream_open(manifest_path, host_allocator, &output));
  iree_status_t status = loom_compile_command_manifest_write(
      artifact_set, source_requirement_indices, source_requirement_count,
      &output.stream);
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

  // Requirements whose request files were written during this operation.
  struct {
    // Sorted plan-local requirement ordinal table.
    uint32_t* values;

    // Number of entries in |values|.
    iree_host_size_t count;

    // Number of allocated entries in |values|.
    iree_host_size_t capacity;
  } source_requirements;
} loom_compile_kernel_request_writer_t;

static iree_status_t loom_compile_kernel_request_writer_record_requirement(
    loom_compile_kernel_request_writer_t* writer, uint32_t requirement_index) {
  IREE_ASSERT(writer->source_requirements.count == 0 ||
              writer->source_requirements
                      .values[writer->source_requirements.count - 1] <
                  requirement_index);
  if (writer->source_requirements.count ==
      writer->source_requirements.capacity) {
    iree_host_size_t new_capacity = 8;
    if (writer->source_requirements.capacity != 0 &&
        !iree_host_size_checked_add(writer->source_requirements.capacity,
                                    writer->source_requirements.capacity,
                                    &new_capacity)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "kernel request table is too large");
    }
    IREE_RETURN_IF_ERROR(iree_allocator_realloc_array(
        writer->host_allocator, new_capacity,
        sizeof(*writer->source_requirements.values),
        (void**)&writer->source_requirements.values));
    writer->source_requirements.capacity = new_capacity;
  }
  writer->source_requirements.values[writer->source_requirements.count++] =
      requirement_index;
  return iree_ok_status();
}

static iree_status_t loom_compile_command_backend_write_kernel_request(
    void* user_data, const loom_cmd_program_kernel_request_t* request) {
  loom_compile_kernel_request_writer_t* writer =
      (loom_compile_kernel_request_writer_t*)user_data;
  loom_kernel_class_product_t kernel_product = {0};
  iree_status_t status = loom_kernel_request_materialize(
      request->kernel_request, writer->block_pool, writer->host_allocator,
      &kernel_product);
  char filename_storage[LOOM_COMPILE_COMMAND_FILENAME_CAPACITY];
  iree_string_view_t filename = iree_string_view_empty();
  if (iree_status_is_ok(status)) {
    status = loom_compile_command_manifest_format_kernel_request_filename(
        request->entry_requirement_index, sizeof(filename_storage),
        filename_storage, &filename);
  }
  if (iree_status_is_ok(status)) {
    status = loom_compile_kernel_request_writer_record_requirement(
        writer, request->entry_requirement_index);
  }

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
        kernel_product.module, stream,
        &(loom_bytecode_write_options_t){
            .producer = IREE_SV("loom-compile"),
            .location_mode = LOOM_BYTECODE_LOCATION_MODE_SOURCE_LOCATIONS,
            .low_repr_environment = writer->low_repr_environment,
        },
        writer->block_pool);
  }
  iree_io_stream_release(stream);
  iree_allocator_free(writer->host_allocator, path_storage);
  loom_kernel_class_product_deinitialize(&kernel_product);
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
  iree_host_size_t root_count = 0;
  if (iree_status_is_ok(status)) {
    status = loom_compile_command_backend_collect_roots(
        index, indexed_module, &scratch_arena, &root_symbol_ordinals,
        &root_count);
  }
  if (iree_status_is_ok(status) && root_count == 0) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "command backend requires at least one retained command program root");
  }
  if (iree_status_is_ok(status) &&
      !iree_string_view_is_empty(options->kernel_request_directory)) {
    status = loom_compile_command_backend_require_directory(
        options->kernel_request_directory, host_allocator);
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
        &artifact_set, kernel_request_writer.source_requirements.values,
        kernel_request_writer.source_requirements.count, options->manifest_path,
        host_allocator);
  }
  if (iree_status_is_ok(status) && plan_valid) {
    *out_emitted = true;
  }

  loom_cmd_program_artifact_set_deinitialize(&artifact_set);
  iree_allocator_free(host_allocator,
                      kernel_request_writer.source_requirements.values);
  loom_link_module_index_free(index);
  iree_arena_deinitialize(&scratch_arena);
  return status;
}
