// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-compile/command_backend.h"

#include "loom/link/module_index.h"
#include "loom/ops/command/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/pass/builtin_registry.h"
#include "loom/target/arch/cmd/artifact_set.h"
#include "loom/target/arch/cmd/lower/program_plan_index.h"
#include "loom/target/entry_selection.h"
#include "loom/tooling/io/file.h"

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
  IREE_RETURN_IF_ERROR(loom_tooling_create_directory_if_needed(
      artifact_directory, host_allocator));
  bool is_directory = false;
  IREE_RETURN_IF_ERROR(loom_tooling_file_path_is_directory(
      artifact_directory, host_allocator, &is_directory));
  if (!is_directory) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "command artifact path '%.*s' is not a directory",
                            (int)artifact_directory.size,
                            artifact_directory.data);
  }

  for (iree_host_size_t i = 0; i < artifact_set->programs.count; ++i) {
    char filename_storage[LOOM_CMD_PROGRAM_ARTIFACT_FILENAME_CAPACITY];
    iree_string_view_t filename = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_cmd_program_artifact_format_filename(
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
  iree_status_t status = loom_cmd_program_artifact_set_format_manifest_json(
      artifact_set, &output.stream);
  if (iree_status_is_ok(status)) {
    status = loom_output_stream_write_cstring(&output.stream, "\n");
  }
  return iree_status_join(status, loom_tooling_output_stream_close(&output));
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

  const iree_host_size_t root_count =
      loom_compile_command_backend_count_roots(run_module->module);
  if (root_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "command backend requires at least one retained command program root");
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
  loom_cmd_program_plan_t plan = {0};
  bool plan_valid = false;
  if (iree_status_is_ok(status)) {
    status = loom_cmd_program_plan_prepare_index(
        index, root_symbol_ordinals, root_count, loom_pass_builtin_registry(),
        loom_target_entry_emitter(&diagnostic_emitter),
        loom_run_session_block_pool(session), &scratch_arena, &plan_valid,
        &plan, host_allocator);
  }
  if (iree_status_is_ok(status) && !plan_valid &&
      diagnostic_emitter.error_count == 0) {
    status = iree_make_status(
        IREE_STATUS_INTERNAL,
        "command program preparation failed without a diagnostic");
  }

  loom_cmd_program_artifact_set_t artifact_set = {0};
  if (iree_status_is_ok(status) && plan_valid) {
    status = loom_cmd_program_artifact_set_build(&plan, &artifact_set,
                                                 host_allocator);
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
  loom_cmd_program_plan_deinitialize(&plan);
  loom_link_module_index_free(index);
  iree_arena_deinitialize(&scratch_arena);
  return status;
}
