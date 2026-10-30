// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// loom-link: imports sources and merges or links bytecode modules through
// metadata-first planning and materialization.

#include <stdio.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/tooling/flags.h"
#include "loom/config/text_binding.h"
#include "loom/testing/target_environment.h"
#include "loom/tooling/cli/help.h"
#include "loom/tooling/cli/loomc_options.h"
#include "loom/tooling/cli/loomc_result.h"
#include "loom/tooling/config/input.h"
#include "loom/tooling/input/flags.h"
#include "loom/tooling/input/loomc_configured.h"
#include "loom/tooling/io/file.h"
#include "loom/tooling/io/module_format.h"
#include "loom/util/stream.h"
#include "loomc/iree.h"
#include "loomc/loomc.h"
#include "loomc/target/configured_target.h"

IREE_FLAG(string, mode, "auto",
          "Planning mode: auto, merge, or link. Auto selects "
          "link mode when roots, root libraries, or exported input symbols "
          "are requested.");
IREE_FLAG(string, from, "auto",
          "Input encoding for every input: auto, text, bc, or bytecode. "
          "Source language is selected separately with --input-format.");
IREE_FLAG(string, to, "text", "Output format: text, bc, or bytecode.");
IREE_FLAG(string, output, "-",
          "Output path. Use '-' or the empty string for stdout.");
IREE_FLAG(bool, cleanup, true,
          "Normalize imported C++ source modules before linking.");
IREE_FLAG_LIST(string, root,
               "Root symbol to materialize in link mode. Repeat for "
               "multiple roots.");
IREE_FLAG_LIST(string, library,
               "Library contributing exported exact definitions and template "
               "implementations. Repeat for multiple libraries.");
IREE_FLAG_LIST_NAMED(
    string, root_library, "root-library",
    "Direct library whose exported symbols are link roots. Repeat for "
    "multiple libraries.");
IREE_FLAG_LIST_NAMED(
    string, transitive_library, "transitive-library",
    "Transitive library available to linking and strict dependency audits, "
    "but not declared as a direct dependency. Repeat for multiple libraries.");
IREE_FLAG_NAMED(bool, strict_deps, "strict-deps", false,
                "Require every exact dependency of merged inputs to be owned "
                "locally or by a direct --library.");
IREE_FLAG_NAMED(string, dependency_report, "dependency-report", "",
                "Write a strict dependency analysis report as JSON. Implies "
                "--strict-deps.");
IREE_FLAG_NAMED(
    string, dependency_component, "dependency-component", "",
    "Component name used in strict dependency diagnostics and reports. "
    "Defaults to the first primary input path.");
IREE_FLAG_LIST(string, config,
               "Compile/link-time config binding. Repeat as "
               "--config=key=value. Bindings specialize each linked analysis "
               "module before template selection; unused bindings are "
               "ignored.");
IREE_FLAG_LIST_NAMED(
    string, config_file, "config-file",
    "JSON/JSONC config object file. Repeat for multiple files. Nested object "
    "keys are flattened with '.' separators.");
IREE_FLAG(string, target, "",
          "Optional homogeneous target in family:selector form, such as "
          "amdgpu:gfx11-generic. The target specializes each linked analysis "
          "module before template selection.");
IREE_FLAG_NAMED(bool, include_input_exports, "include-input-exports", false,
                "In link mode, add exported input symbols as roots.");
IREE_FLAG_NAMED(
    bool, include_input_tests, "include-input-tests", false,
    "In link mode, add test/benchmark-only input symbols as roots; with "
    "--strip-check, retain their direct callable subjects instead.");
IREE_FLAG_NAMED(bool, strip_check, "strip-check", false,
                "Strip test/benchmark-only symbols before output.");
IREE_FLAG_NAMED(
    bool, require_resolved_config, "require-resolved-config", false,
    "Require all config.decl symbols to be materialized before output.");
IREE_FLAG_NAMED(bool, allow_unresolved, "allow-unresolved", false,
                "Preserve unresolved symbols for a later link instead of "
                "rejecting them.");
IREE_FLAG_NAMED(bool, print_config_schema, "print-config-schema", false,
                "Print config schema JSON instead of linked Loom IR.");
IREE_FLAG_NAMED(bool, print_plan, "print-plan", false,
                "Print the planner's selected symbols instead of linked "
                "output.");
IREE_FLAG_NAMED(bool, list_symbols, "list-symbols", false,
                "Print indexed input symbols instead of linked output.");
IREE_FLAG(bool, verify, true,
          "Verify the linked output module before printing.");

typedef enum loom_link_cli_mode_e {
  LOOM_LINK_CLI_MODE_AUTO = 0,
  LOOM_LINK_CLI_MODE_MERGE = 1,
  LOOM_LINK_CLI_MODE_LINK = 2,
} loom_link_cli_mode_t;

typedef enum loom_link_cli_input_class_e {
  // Primary source jointly owned by the output module.
  LOOM_LINK_CLI_INPUT_PRIMARY = 0,
  // Direct library whose exports are link roots.
  LOOM_LINK_CLI_INPUT_ROOT_LIBRARY = 1,
  // Direct library available for dependency resolution.
  LOOM_LINK_CLI_INPUT_DIRECT_LIBRARY = 2,
  // Transitive library available only for dependency resolution.
  LOOM_LINK_CLI_INPUT_TRANSITIVE_LIBRARY = 3,
} loom_link_cli_input_class_t;

typedef struct loom_link_cli_input_t {
  // Physical input path used for file access and importer include lookup.
  iree_string_view_t path;
  // Linkage and dependency classification assigned by the command line.
  loom_link_cli_input_class_t input_class;
} loom_link_cli_input_t;

typedef struct loom_link_cli_index_t {
  // Frozen index owning immutable bytecode sources and module snapshots.
  loomc_link_index_t* module_index;
  // Library providers declared as direct dependencies.
  struct {
    // Allocator-owned provider ordinals in command-line order.
    loomc_host_size_t* values;
    // Number of provider ordinals in values.
    loomc_host_size_t count;
  } direct_providers;
  // Providers whose exports are link roots.
  struct {
    // Allocator-owned provider ordinals in command-line order.
    loomc_host_size_t* values;
    // Number of provider ordinals in values.
    loomc_host_size_t count;
  } root_providers;
} loom_link_cli_index_t;

static const char* loom_link_cli_role_name(loomc_link_provider_role_t role) {
  switch (role) {
    case LOOMC_LINK_PROVIDER_ROLE_INPUT:
      return "input";
    case LOOMC_LINK_PROVIDER_ROLE_LIBRARY:
      return "library";
  }
  return "unknown";
}

static loomc_link_provider_role_t loom_link_cli_input_role(
    const loom_link_cli_input_t* input) {
  return input->input_class == LOOM_LINK_CLI_INPUT_PRIMARY
             ? LOOMC_LINK_PROVIDER_ROLE_INPUT
             : LOOMC_LINK_PROVIDER_ROLE_LIBRARY;
}

static bool loom_link_cli_input_is_direct_dependency(
    const loom_link_cli_input_t* input) {
  return input->input_class == LOOM_LINK_CLI_INPUT_ROOT_LIBRARY ||
         input->input_class == LOOM_LINK_CLI_INPUT_DIRECT_LIBRARY;
}

static bool loom_link_cli_input_exports_are_roots(
    const loom_link_cli_input_t* input) {
  return input->input_class == LOOM_LINK_CLI_INPUT_ROOT_LIBRARY;
}

static const char* loom_link_cli_identity_name(
    loomc_link_symbol_identity_t identity) {
  switch (identity) {
    case LOOMC_LINK_SYMBOL_IDENTITY_PRIVATE:
      return "private";
    case LOOMC_LINK_SYMBOL_IDENTITY_GLOBAL:
      return "global";
  }
  return "unknown";
}

static iree_status_t loom_link_cli_parse_mode(iree_string_view_t value,
                                              loom_link_cli_mode_t* out_mode) {
  if (iree_string_view_equal(value, IREE_SV("auto"))) {
    *out_mode = LOOM_LINK_CLI_MODE_AUTO;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("merge"))) {
    *out_mode = LOOM_LINK_CLI_MODE_MERGE;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("link"))) {
    *out_mode = LOOM_LINK_CLI_MODE_LINK;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "unsupported link mode '%.*s'; expected auto, merge, or link",
      (int)value.size, value.data);
}

static iree_status_t loom_link_cli_resolve_plan_mode(
    loom_link_cli_mode_t cli_mode, const iree_flag_string_list_t roots,
    iree_host_size_t root_library_count, bool include_input_exports,
    bool include_input_tests, loomc_link_mode_t* out_mode) {
  if (cli_mode == LOOM_LINK_CLI_MODE_AUTO) {
    *out_mode = (roots.count > 0 || root_library_count > 0 ||
                 include_input_exports || include_input_tests)
                    ? LOOMC_LINK_MODE_LINK
                    : LOOMC_LINK_MODE_MERGE;
    return iree_ok_status();
  }
  if (cli_mode == LOOM_LINK_CLI_MODE_MERGE) {
    if (roots.count > 0 || root_library_count > 0 || include_input_exports ||
        include_input_tests) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "merge mode does not accept --root, --root-library, or "
          "--include-input-exports/--include-input-tests");
    }
    *out_mode = LOOMC_LINK_MODE_MERGE;
    return iree_ok_status();
  }
  *out_mode = LOOMC_LINK_MODE_LINK;
  return iree_ok_status();
}

static iree_status_t loom_link_cli_append_config_flags(
    loom_config_text_binding_set_t* config_set) {
  iree_flag_string_list_t assignments = FLAG_config_list();
  for (iree_host_size_t i = 0; i < assignments.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_tooling_config_text_binding_set_append_assignment(
        config_set, assignments.values[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_link_cli_append_config_files(
    loom_config_text_binding_set_t* config_set, iree_allocator_t allocator) {
  iree_flag_string_list_t paths = FLAG_config_file_list();
  for (iree_host_size_t i = 0; i < paths.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_tooling_config_text_binding_set_append_json_file(
        config_set, paths.values[i], allocator));
  }
  return iree_ok_status();
}

static loomc_status_t loom_link_cli_create_target_environment(
    loomc_allocator_t allocator, loomc_target_environment_t** out_environment) {
  loomc_target_environment_t* inputs[2] = {NULL, NULL};
  loomc_status_t status =
      loomc_target_environment_create_configured_target(allocator, &inputs[0]);
  if (loomc_status_is_ok(status)) {
    status = loomc_target_environment_create_test_ir(allocator, &inputs[1]);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_target_environment_create(IREE_ARRAYSIZE(inputs), inputs,
                                             allocator, out_environment);
  }
  loomc_target_environment_release(inputs[1]);
  loomc_target_environment_release(inputs[0]);
  return status;
}

static iree_status_t loom_link_cli_check_result(const loomc_result_t* result,
                                                const char* failure_message) {
  bool succeeded = false;
  IREE_RETURN_IF_ERROR(
      loom_tooling_cli_print_loomc_result(stderr, result, &succeeded));
  return succeeded ? iree_ok_status()
                   : iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "%s",
                                      failure_message);
}

static iree_status_t loom_link_cli_prepare_pass_program(
    loomc_context_t* context, const char* pipeline,
    loomc_allocator_t compiler_allocator,
    loomc_pass_program_t** out_pass_program) {
  loomc_result_t* result = NULL;
  iree_status_t status =
      iree_status_from_loomc(loomc_pass_program_create_from_pipeline_text(
          context, loomc_make_cstring_view(pipeline), NULL, compiler_allocator,
          out_pass_program, &result));
  if (iree_status_is_ok(status)) {
    status =
        loom_link_cli_check_result(result, "cleanup pipeline creation failed");
  }
  loomc_result_release(result);
  return status;
}

static void loom_link_cli_index_deinitialize(loom_link_cli_index_t* index,
                                             iree_allocator_t allocator) {
  loomc_link_index_release(index->module_index);
  iree_allocator_free(allocator, index->root_providers.values);
  iree_allocator_free(allocator, index->direct_providers.values);
  *index = (loom_link_cli_index_t){0};
}

static void LOOMC_API_PTR
loom_link_cli_release_file(void* user_data, loomc_byte_span_t contents) {
  (void)contents;
  iree_io_file_contents_free((iree_io_file_contents_t*)user_data);
}

static iree_status_t loom_link_cli_add_input(
    const loom_link_cli_input_t* input, loom_module_format_t requested_format,
    loomc_context_t* context, loomc_workspace_t* workspace,
    bool cleanup_sources, loomc_pass_program_t** inout_source_cleanup_program,
    loomc_link_index_builder_t* builder, iree_arena_block_pool_t* block_pool,
    iree_allocator_t allocator, loomc_link_index_source_slot_t* out_slot) {
  const iree_string_view_t filename =
      loom_tooling_file_path_is_stdio(input->path) ? IREE_SV("<stdin>")
                                                   : input->path;
  const loomc_allocator_t compiler_allocator =
      loomc_allocator_from_iree(allocator);
  const loomc_link_index_source_options_t provider_options = {
      .provider_name = loomc_string_view_from_iree(filename),
      .role = loom_link_cli_input_role(input),
  };
  iree_io_file_contents_t* contents = NULL;
  IREE_RETURN_IF_ERROR(
      loom_tooling_read_input_file(input->path, allocator, &contents));
  const loom_module_format_t format =
      requested_format == LOOM_MODULE_FORMAT_AUTO
          ? loom_module_format_detect_input(contents->const_buffer)
          : requested_format;
  loomc_source_t* source = NULL;
  loomc_module_t* module = NULL;
  loomc_result_t* result = NULL;
  iree_status_t status = iree_ok_status();
  if (format == LOOM_MODULE_FORMAT_BYTECODE) {
    // Retain the file buffer without decoding bodies or copying its bytes.
    const loomc_source_options_t source_options = {
        .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
        .structure_size = sizeof(source_options),
        .format = LOOMC_SOURCE_FORMAT_BYTECODE,
        .identifier = loomc_string_view_from_iree(filename),
        .contents = loomc_byte_span_from_iree(contents->const_buffer),
        .storage = LOOMC_SOURCE_STORAGE_EXTERNAL,
        .release = loom_link_cli_release_file,
        .release_user_data = contents,
    };
    status = iree_status_from_loomc(
        loomc_source_create(&source_options, compiler_allocator, &source));
    if (iree_status_is_ok(status)) {
      contents = NULL;
      status = iree_status_from_loomc(loomc_link_index_builder_add_source(
          builder, source, &provider_options, out_slot));
    }
  } else {
    loom_tooling_loomc_input_options_t admission_options = {
        .input = loom_input_options_from_flags(),
        .path = filename,
        .source = loom_tooling_file_contents_string_view(contents),
        .import = loom_configured_input_loomc_importer(),
    };
    const loom_input_format_t* selected_format = NULL;
    status = loom_input_format_select(loom_configured_input_loomc_formats(),
                                      admission_options.input.format, filename,
                                      &selected_format, NULL);
    if (iree_status_is_ok(status)) {
      admission_options.input.format = selected_format->name;
      status = loom_tooling_input_admit_loomc_module(
          &admission_options, context, workspace, block_pool, &module, &result,
          allocator);
    }
    if (iree_status_is_ok(status)) {
      status = loom_link_cli_check_result(result, "input has source errors");
    }
    const bool cleanup_source =
        cleanup_sources && selected_format &&
        iree_string_view_equal(selected_format->name, IREE_SV("cxx"));
    if (iree_status_is_ok(status) && cleanup_source &&
        !*inout_source_cleanup_program) {
      status = loom_link_cli_prepare_pass_program(
          context, "combine,cse,dce,location-dce", compiler_allocator,
          inout_source_cleanup_program);
    }
    if (iree_status_is_ok(status) && cleanup_source) {
      loomc_result_release(result);
      result = NULL;
      status = iree_status_from_loomc(
          loomc_pass_program_run(*inout_source_cleanup_program, workspace,
                                 module, NULL, compiler_allocator, &result));
      if (iree_status_is_ok(status)) {
        status =
            loom_link_cli_check_result(result, "C++ source cleanup failed");
      }
    }
    if (iree_status_is_ok(status)) {
      status = iree_status_from_loomc(loomc_link_index_builder_add_module(
          builder, module, &provider_options, out_slot));
    }
  }
  loomc_result_release(result);
  loomc_module_release(module);
  loomc_source_release(source);
  iree_io_file_contents_free(contents);
  return status;
}

static iree_status_t loom_link_cli_collect_inputs(
    int argc, char** argv, iree_allocator_t allocator,
    loom_link_cli_input_t** out_inputs, iree_host_size_t* out_input_count) {
  *out_inputs = NULL;
  *out_input_count = 0;

  const iree_flag_string_list_t root_libraries = FLAG_root_library_list();
  const iree_flag_string_list_t libraries = FLAG_library_list();
  const iree_flag_string_list_t transitive_libraries =
      FLAG_transitive_library_list();
  const iree_host_size_t primary_count =
      argc < 2 && root_libraries.count == 0 && libraries.count == 0 &&
              transitive_libraries.count == 0
          ? 1
          : (iree_host_size_t)(argc - 1);
  iree_host_size_t input_count = 0;
  if (!iree_host_size_checked_add(primary_count, root_libraries.count,
                                  &input_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "input count overflow");
  }
  if (!iree_host_size_checked_add(input_count, libraries.count, &input_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "input count overflow");
  }
  if (!iree_host_size_checked_add(input_count, transitive_libraries.count,
                                  &input_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "input count overflow");
  }

  loom_link_cli_input_t* inputs = NULL;
  if (input_count > 0) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        allocator, input_count, sizeof(*inputs), (void**)&inputs));
    memset(inputs, 0, input_count * sizeof(*inputs));
  }

  iree_status_t status = iree_ok_status();
  iree_host_size_t input_ordinal = 0;
  for (iree_host_size_t i = 0; i < primary_count && iree_status_is_ok(status);
       ++i) {
    iree_string_view_t path = argc < 2 ? iree_string_view_empty()
                                       : iree_make_cstring_view(argv[i + 1]);
    if (input_count > 1 && loom_tooling_file_path_is_stdio(path)) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "'-' stdin input must be the only input path in this invocation");
      break;
    }
    inputs[input_ordinal++] =
        (loom_link_cli_input_t){path, LOOM_LINK_CLI_INPUT_PRIMARY};
  }
  for (iree_host_size_t i = 0;
       i < root_libraries.count && iree_status_is_ok(status); ++i) {
    const iree_string_view_t path = root_libraries.values[i];
    if (loom_tooling_file_path_is_stdio(path)) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "root library input requires a filesystem path, not stdin");
      break;
    }
    inputs[input_ordinal++] =
        (loom_link_cli_input_t){path, LOOM_LINK_CLI_INPUT_ROOT_LIBRARY};
  }
  for (iree_host_size_t i = 0; i < libraries.count && iree_status_is_ok(status);
       ++i) {
    iree_string_view_t path = libraries.values[i];
    if (loom_tooling_file_path_is_stdio(path)) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "library input requires a filesystem path, not stdin");
      break;
    }
    inputs[input_ordinal++] =
        (loom_link_cli_input_t){path, LOOM_LINK_CLI_INPUT_DIRECT_LIBRARY};
  }
  for (iree_host_size_t i = 0;
       i < transitive_libraries.count && iree_status_is_ok(status); ++i) {
    iree_string_view_t path = transitive_libraries.values[i];
    if (loom_tooling_file_path_is_stdio(path)) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "transitive library input requires a filesystem path, not stdin");
      break;
    }
    inputs[input_ordinal++] =
        (loom_link_cli_input_t){path, LOOM_LINK_CLI_INPUT_TRANSITIVE_LIBRARY};
  }

  if (!iree_status_is_ok(status)) {
    iree_allocator_free(allocator, inputs);
    return status;
  }
  *out_inputs = inputs;
  *out_input_count = input_count;
  return iree_ok_status();
}

static iree_status_t loom_link_cli_build_index(
    const loom_link_cli_input_t* inputs, iree_host_size_t input_count,
    loom_module_format_t input_format, loomc_context_t* context,
    loomc_workspace_t* workspace, bool cleanup_sources,
    loomc_pass_program_t** inout_source_cleanup_program,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_link_cli_index_t* out_index) {
  *out_index = (loom_link_cli_index_t){0};
  loomc_link_index_builder_t* builder = NULL;
  IREE_RETURN_IF_ERROR(iree_status_from_loomc(loomc_link_index_builder_allocate(
      context, NULL, loomc_allocator_from_iree(allocator), &builder)));
  iree_host_size_t direct_count = 0;
  iree_host_size_t root_count = 0;
  for (iree_host_size_t i = 0; i < input_count; ++i) {
    direct_count += loom_link_cli_input_is_direct_dependency(&inputs[i]);
    root_count += loom_link_cli_input_exports_are_roots(&inputs[i]);
  }
  iree_status_t status = iree_ok_status();
  if (direct_count) {
    status = iree_allocator_malloc_array(
        allocator, direct_count, sizeof(*out_index->direct_providers.values),
        (void**)&out_index->direct_providers.values);
  }
  if (iree_status_is_ok(status) && root_count) {
    status = iree_allocator_malloc_array(
        allocator, root_count, sizeof(*out_index->root_providers.values),
        (void**)&out_index->root_providers.values);
  }
  for (iree_host_size_t i = 0; i < input_count && iree_status_is_ok(status);
       ++i) {
    loomc_link_index_source_slot_t slot = {0};
    status = loom_link_cli_add_input(
        &inputs[i], input_format, context, workspace, cleanup_sources,
        inout_source_cleanup_program, builder, block_pool, allocator, &slot);
    if (iree_status_is_ok(status) &&
        loom_link_cli_input_is_direct_dependency(&inputs[i])) {
      out_index->direct_providers.values[out_index->direct_providers.count++] =
          slot.ordinal;
    }
    if (iree_status_is_ok(status) &&
        loom_link_cli_input_exports_are_roots(&inputs[i])) {
      out_index->root_providers.values[out_index->root_providers.count++] =
          slot.ordinal;
    }
  }
  loomc_result_t* result = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_link_index_builder_finish(
        builder, &out_index->module_index, &result));
  }
  if (iree_status_is_ok(status)) {
    status = loom_link_cli_check_result(result, "input indexing failed");
  }
  loomc_result_release(result);
  loomc_link_index_builder_free(builder);
  if (!iree_status_is_ok(status)) {
    loom_link_cli_index_deinitialize(out_index, allocator);
  }
  return status;
}

static iree_status_t loom_link_cli_write_report(
    const loomc_artifact_t* artifact, iree_string_view_t path, bool newline,
    iree_allocator_t allocator) {
  loom_tooling_output_stream_t output = {0};
  IREE_RETURN_IF_ERROR(
      loom_tooling_output_stream_open(path, allocator, &output));
  iree_status_t status = iree_status_from_loomc(
      loomc_artifact_write_to_file(artifact, output.file));
  if (iree_status_is_ok(status) && newline) {
    status = loom_output_stream_write_char(&output.stream, '\n');
  }
  return iree_status_join(status, loom_tooling_output_stream_close(&output));
}

static iree_status_t loom_link_cli_analyze_dependencies(
    const loom_link_cli_index_t* index, iree_string_view_t component_name,
    iree_string_view_t report_path, loomc_workspace_t* workspace,
    iree_allocator_t allocator) {
  const bool report_requested = !iree_string_view_is_empty(report_path);
  const loomc_link_dependency_analysis_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_LINK_DEPENDENCY_ANALYSIS_OPTIONS,
      .structure_size = sizeof(options),
      .direct_provider_ordinals = index->direct_providers.values,
      .direct_provider_count = index->direct_providers.count,
      .component_name = loomc_string_view_from_iree(component_name),
      .artifact_flags = report_requested
                            ? LOOMC_LINK_DEPENDENCY_ARTIFACT_FLAG_REPORT_JSON
                            : 0,
  };
  loomc_result_t* result = NULL;
  iree_status_t status = iree_status_from_loomc(loomc_link_analyze_dependencies(
      index->module_index, workspace, &options, &result));
  if (iree_status_is_ok(status) && report_requested) {
    status = loom_link_cli_write_report(loomc_result_artifact_at(result, 0),
                                        report_path, true, allocator);
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_link_cli_check_result(result, "strict dependency analysis failed");
  }
  loomc_result_release(result);
  return status;
}

static iree_status_t loom_link_cli_append_flag(iree_string_builder_t* builder,
                                               bool* needs_separator,
                                               iree_string_view_t flag_name) {
  if (*needs_separator) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
  }
  *needs_separator = true;
  return iree_string_builder_append_string(builder, flag_name);
}

static iree_status_t loom_link_cli_append_symbol_flags(
    iree_string_builder_t* builder, loomc_link_symbol_flags_t flags) {
  bool needs_separator = false;
  if (iree_any_bit_set(flags, LOOMC_LINK_SYMBOL_FLAG_PUBLIC)) {
    IREE_RETURN_IF_ERROR(loom_link_cli_append_flag(builder, &needs_separator,
                                                   IREE_SV("public")));
  }
  if (iree_any_bit_set(flags, LOOMC_LINK_SYMBOL_FLAG_IMPORT)) {
    IREE_RETURN_IF_ERROR(loom_link_cli_append_flag(builder, &needs_separator,
                                                   IREE_SV("import")));
  }
  if (iree_any_bit_set(flags, LOOMC_LINK_SYMBOL_FLAG_EXPORT)) {
    IREE_RETURN_IF_ERROR(loom_link_cli_append_flag(builder, &needs_separator,
                                                   IREE_SV("export")));
  }
  if (iree_any_bit_set(flags, LOOMC_LINK_SYMBOL_FLAG_DECLARATION)) {
    IREE_RETURN_IF_ERROR(loom_link_cli_append_flag(builder, &needs_separator,
                                                   IREE_SV("declaration")));
  }
  if (iree_any_bit_set(flags, LOOMC_LINK_SYMBOL_FLAG_CONCRETE_DEFINITION)) {
    IREE_RETURN_IF_ERROR(loom_link_cli_append_flag(builder, &needs_separator,
                                                   IREE_SV("definition")));
  }
  if (iree_any_bit_set(flags, LOOMC_LINK_SYMBOL_FLAG_CONFIG)) {
    IREE_RETURN_IF_ERROR(loom_link_cli_append_flag(builder, &needs_separator,
                                                   IREE_SV("config")));
  }
  if (iree_any_bit_set(flags, LOOMC_LINK_SYMBOL_FLAG_TEST_ONLY)) {
    IREE_RETURN_IF_ERROR(loom_link_cli_append_flag(builder, &needs_separator,
                                                   IREE_SV("test-only")));
  }
  if (!needs_separator) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "-"));
  }
  return iree_ok_status();
}

static iree_status_t loom_link_cli_print_symbol_list(
    const loomc_link_index_t* index, iree_string_view_t output_path,
    iree_allocator_t allocator) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(allocator, &builder);

  iree_status_t status = iree_string_builder_append_format(
      &builder, "symbols %" PRIhsz "\n", loomc_link_index_symbol_count(index));
  for (iree_host_size_t i = 0;
       i < loomc_link_index_symbol_count(index) && iree_status_is_ok(status);
       ++i) {
    loomc_link_index_symbol_t symbol;
    loomc_link_index_symbol_at(index, i, &symbol);
    loomc_link_index_provider_t provider;
    loomc_link_index_provider_at(index, symbol.provider_ordinal, &provider);
    loomc_link_index_module_t module;
    loomc_link_index_module_at(index, symbol.module_ordinal, &module);
    status = iree_string_builder_append_format(
        &builder,
        "[%" PRIhsz
        "] @%.*s role=%s provider=%.*s module=%.*s identity=%s "
        "flags=",
        i, (int)symbol.name.size, symbol.name.data,
        loom_link_cli_role_name(provider.role), (int)provider.name.size,
        provider.name.data, (int)module.name.size, module.name.data,
        loom_link_cli_identity_name(symbol.identity));
    if (iree_status_is_ok(status)) {
      status = loom_link_cli_append_symbol_flags(&builder, symbol.flags);
    }
    if (iree_status_is_ok(status)) {
      status = iree_string_builder_append_cstring(&builder, "\n");
    }
  }

  if (iree_status_is_ok(status)) {
    status = loom_tooling_write_output_file(
        output_path, iree_string_builder_view(&builder), allocator);
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

static iree_status_t loom_link_cli_write_module_output(
    iree_string_view_t path, const loomc_module_t* module,
    loom_module_format_t format, iree_allocator_t allocator) {
  const loomc_module_serialize_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS,
      .structure_size = sizeof(options),
      .format = format == LOOM_MODULE_FORMAT_BYTECODE
                    ? LOOMC_SOURCE_FORMAT_BYTECODE
                    : LOOMC_SOURCE_FORMAT_TEXT,
  };
  loomc_source_t* source = NULL;
  IREE_RETURN_IF_ERROR(iree_status_from_loomc(loomc_module_serialize_to_source(
      module, &options, loomc_allocator_from_iree(allocator), &source)));
  const loomc_byte_span_t contents = loomc_source_contents(source);
  iree_status_t status = loom_tooling_write_output_file(
      path,
      iree_make_string_view((const char*)contents.data, contents.data_length),
      allocator);
  loomc_source_release(source);
  return status;
}

static void loom_link_cli_print_agents_markdown(FILE* stream) {
  fprintf(
      stream,
      "## loom-link\n"
      "\n"
      "`loom-link` imports source and combines bytecode modules, applies "
      "config\n"
      "bindings to materialized modules, and selects the symbols that should "
      "be\n"
      "kept in a merged library or reachable runtime artifact.\n"
      "\n"
      "### Common flows\n"
      "\n"
      "```shell\n"
      "loom-link root.loom providers.loom --output=linked.loom\n"
      "loom-link root.loom --library=providers.loombc --root=@entry \\\n"
      "  --to=bc --output=entry.loombc\n"
      "loom-link --root-library=app.loombc \\\n"
      "  --transitive-library=motifs.loombc --output=app.loom\n"
      "loom-link root.loom --library=providers.loom --root=@entry "
      "--print-plan\n"
      "loom-link library.loom --mode=merge --strip-check --to=bc \\\n"
      "  --output=runtime-library.loombc\n"
      "loom-link component.loom --library=direct.loombc \\\n"
      "  --transitive-library=indirect.loombc --mode=merge \\\n"
      "  --dependency-report=component.dependencies.json\n"
      "loom-link root.loom --print-config-schema\n"
      "```\n"
      "\n"
      "### Inputs and libraries\n"
      "\n"
      "Positional inputs jointly form the direct source module. A\n"
      "`--library=path` names a separate library. Merge mode leaves library\n"
      "symbols out of the result; link mode may select their exported exact\n"
      "definitions and template implementations. A `--root-library=path` is\n"
      "a direct library whose exported symbols seed link mode without making\n"
      "its private symbols visible to other providers. A "
      "`--transitive-library` is\n"
      "equally available to link mode, but does not own a direct dependency\n"
      "during strict dependency analysis. `--from=auto|text|bc` "
      "controls\n"
      "input encoding and `--to=text|bc` controls output encoding. Source\n"
      "languages are selected by filename or `--input-format=loom|cxx`.\n"
      "Optional importers must be enabled in the build. Provider options use\n"
      "`--input-options=\"cxx:std=c++20 I=include\"`; relative headers "
      "resolve\n"
      "from the physical input path. `--source-prefix-map=old=new` changes\n"
      "diagnostic filenames without changing header lookup.\n"
      "\n"
      "### Merge and link\n"
      "\n"
      "`--mode=merge` preserves every non-stripped primary-input symbol in\n"
      "input order. `--mode=link` keeps explicit `--root=@symbol` values,\n"
      "exports from each `--root-library`, optional\n"
      "`--include-input-exports`, test-only symbols selected by\n"
      "`--include-input-tests`, and reachable dependencies. The input-test\n"
      "policy does not select tests contributed by libraries.\n"
      "A public library definition selected only as a dependency becomes\n"
      "private in the output; library visibility does not implicitly "
      "re-export\n"
      "it through the requester. Select it as a root when it is also an "
      "output.\n"
      "C++ source modules receive the same target-aware normalization as\n"
      "`loom-import-cxx`; `--cleanup=false` preserves their imported form.\n"
      "`--strip-check` removes symbols marked as test/benchmark-only from\n"
      "runtime artifacts.\n"
      "\n"
      "### Dependency hygiene\n"
      "\n"
      "`--strict-deps` checks the exact symbol dependencies authored by the\n"
      "merged positional inputs before materialization. Each dependency must\n"
      "be defined locally or exported by a direct `--library`; a compatible\n"
      "definition found only in a `--transitive-library` is diagnosed as a\n"
      "missing direct dependency. Open template families may remain "
      "unresolved\n"
      "for later specialization. `--dependency-report=path` implies the check\n"
      "and writes its schema-versioned JSON result even when the check fails.\n"
      "`--dependency-component=name` supplies external build provenance for\n"
      "diagnostics and the report without embedding it in Loom IR.\n"
      "\n"
      "### Provider debugging\n"
      "\n"
      "```shell\n"
      "loom-link root.loom --library=providers.loom --list-symbols\n"
      "loom-link root.loom --library=providers.loom --root=@entry "
      "--print-plan\n"
      "loom-link root.loom --library=providers.loom --root=@entry \\\n"
      "  --config=model.hidden_size=4096 \\\n"
      "  --target=amdgpu:gfx11-generic \\\n"
      "  --require-resolved-config\n"
      "loom-link root.loom --library=provider-a.loom --root=@entry \\\n"
      "  --allow-unresolved --to=bc --output=partial.loombc\n"
      "```\n"
      "\n"
      "`--list-symbols` shows the indexed providers. `--print-plan` shows why\n"
      "each symbol is live in the completed link plan. Config\n"
      "bindings are applied to the composed analysis module before each "
      "dependency and template-selection step. `--target` applies "
      "structured target facts at the same boundary, so target and shape "
      "predicates can prune unreachable provider templates.\n"
      "Templates whose choice depends on later caller or target facts retain\n"
      "their viable providers for compilation, even in a closed link.\n"
      "`--allow-unresolved` preserves unresolved declarations whose libraries\n"
      "were not supplied so the output can be linked again.\n");
}

int main(int argc, char** argv) {
  iree_flags_set_usage(
      "loom-link",
      "Imports source and links bytecode modules into one Loom module.\n"
      "\n"
      "Usage:\n"
      "  loom-link [--mode=merge|link] [--from=auto|text|bc] "
      "[--to=text|bc] [--target=family:selector] "
      "[--output=file] [file...]\n"
      "  loom-link model.loom --library=kernels.loombc --root=@entry "
      "--to=bc --output=model.loombc\n"
      "  loom-link --root-library=app.loombc --library=kernels.loombc "
      "--to=bc --output=app.loombc\n"
      "  loom-link --agents_md\n"
      "\n"
      "Input defaults to stdin only when no primary inputs or libraries are "
      "provided. Positional inputs jointly own private definitions; "
      "--root-library, --library, and --transitive-library inputs remain "
      "separate in merge mode and contribute exported definitions in link "
      "mode.\n"
      "Merge mode keeps every non-stripped primary-input symbol in stable "
      "input order. Link "
      "mode keeps explicit roots, root-library exports, or exported input "
      "symbols and their reachable dependencies.\n"
      "Use --strip-check to remove symbols marked as test/benchmark-only from "
      "runtime artifacts. Use --allow-unresolved to emit a reusable partial "
      "artifact. Use --strict-deps when producing a relocatable library to "
      "enforce direct dependency ownership before materialization.\n");
  for (int i = 1; i < argc; ++i) {
    if (loom_tooling_cli_is_agents_markdown_arg(argv[i])) {
      loom_link_cli_print_agents_markdown(stdout);
      return 0;
    }
  }
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);

  loom_tooling_cli_set_default_help_filter();
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_allocator_t allocator = iree_allocator_system();
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(32 * 1024, allocator, &block_pool);

  const loomc_allocator_t compiler_allocator =
      loomc_allocator_from_iree(allocator);
  loomc_target_environment_t* target_environment = NULL;
  loomc_target_profile_t* target_profile = NULL;
  loomc_context_t* context = NULL;
  loomc_workspace_t* workspace = NULL;
  loomc_linker_t* linker = NULL;
  loomc_module_t* linked_module = NULL;
  loomc_pass_program_t* source_cleanup_program = NULL;
  loomc_result_t* result = NULL;
  loom_config_text_binding_set_t config_set;
  loom_config_text_binding_set_initialize(allocator, &config_set);
  loomc_config_binding_t* config_bindings = NULL;
  loomc_config_options_t config_options = {0};
  loomc_string_view_t* root_symbols = NULL;
  loom_link_cli_input_t* inputs = NULL;
  iree_host_size_t input_count = 0;
  loom_link_cli_index_t link_index = {0};

  loom_module_format_t input_format = LOOM_MODULE_FORMAT_AUTO;
  loom_module_format_t output_format = LOOM_MODULE_FORMAT_TEXT;
  loom_link_cli_mode_t cli_mode = LOOM_LINK_CLI_MODE_AUTO;
  loomc_link_mode_t plan_mode = LOOMC_LINK_MODE_MERGE;
  const iree_flag_string_list_t roots = FLAG_root_list();
  const iree_string_view_t dependency_report_path =
      iree_make_cstring_view(FLAG_dependency_report);
  const iree_string_view_t requested_dependency_component =
      iree_make_cstring_view(FLAG_dependency_component);
  const bool dependency_analysis_requested =
      FLAG_strict_deps || !iree_string_view_is_empty(dependency_report_path);

  iree_status_t status = loom_module_format_parse(
      iree_make_cstring_view(FLAG_from), /*allow_auto=*/true, &input_format);
  if (iree_status_is_ok(status)) {
    status = loom_module_format_parse(iree_make_cstring_view(FLAG_to),
                                      /*allow_auto=*/false, &output_format);
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_link_cli_parse_mode(iree_make_cstring_view(FLAG_mode), &cli_mode);
  }
  if (iree_status_is_ok(status)) {
    status = loom_link_cli_resolve_plan_mode(
        cli_mode, roots, FLAG_root_library_list().count,
        FLAG_include_input_exports, FLAG_include_input_tests, &plan_mode);
  }
  if (iree_status_is_ok(status) && dependency_analysis_requested &&
      plan_mode != LOOMC_LINK_MODE_MERGE) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--strict-deps and --dependency-report require merge mode");
  }
  if (iree_status_is_ok(status) && !dependency_analysis_requested &&
      !iree_string_view_is_empty(requested_dependency_component)) {
    status =
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "--dependency-component requires --strict-deps or "
                         "--dependency-report");
  }
  if (iree_status_is_ok(status) && FLAG_print_config_schema &&
      (FLAG_print_plan || FLAG_list_symbols)) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--print-config-schema cannot be combined with --print-plan or "
        "--list-symbols");
  }
  if (iree_status_is_ok(status) && FLAG_print_plan && FLAG_list_symbols) {
    status =
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "--print-plan cannot be combined with --list-symbols");
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loom_link_cli_create_target_environment(
        compiler_allocator, &target_environment));
  }
  const iree_string_view_t target =
      iree_string_view_trim(iree_make_cstring_view(FLAG_target));
  if (iree_status_is_ok(status) && !iree_string_view_is_empty(target)) {
    status = iree_status_from_loomc(loomc_target_profile_select(
        target_environment, loomc_string_view_from_iree(target),
        compiler_allocator, &target_profile));
  }
  if (iree_status_is_ok(status)) {
    const loomc_context_target_options_t target_options = {
        .type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
        .structure_size = sizeof(target_options),
        .target_environment = target_environment,
    };
    const loomc_context_options_t context_options = {
        .type = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
        .structure_size = sizeof(context_options),
        .next = &target_options,
    };
    status = iree_status_from_loomc(
        loomc_context_create(&context_options, compiler_allocator, &context));
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(
        loomc_workspace_create(NULL, compiler_allocator, &workspace));
  }
  if (iree_status_is_ok(status)) {
    status = loom_link_cli_append_config_files(&config_set, allocator);
  }
  if (iree_status_is_ok(status)) {
    status = loom_link_cli_append_config_flags(&config_set);
  }
  if (iree_status_is_ok(status)) {
    status = loom_tooling_cli_make_loomc_config_options(
        &config_set, allocator, &config_bindings, &config_options);
    // Linking can produce relocatable modules with unresolved config. The
    // explicit final-resolution check runs after symbol selection below.
    config_options.flags = 0;
  }
  if (iree_status_is_ok(status)) {
    status = loom_link_cli_collect_inputs(argc, argv, allocator, &inputs,
                                          &input_count);
  }
  if (iree_status_is_ok(status)) {
    status = loom_link_cli_build_index(
        inputs, input_count, input_format, context, workspace, FLAG_cleanup,
        &source_cleanup_program, &block_pool, allocator, &link_index);
  }
  if (iree_status_is_ok(status) && dependency_analysis_requested) {
    iree_string_view_t component_name = requested_dependency_component;
    if (iree_string_view_is_empty(component_name) && input_count &&
        inputs[0].input_class == LOOM_LINK_CLI_INPUT_PRIMARY) {
      component_name = loom_tooling_file_path_is_stdio(inputs[0].path)
                           ? IREE_SV("<stdin>")
                           : inputs[0].path;
    }
    status = loom_link_cli_analyze_dependencies(&link_index, component_name,
                                                dependency_report_path,
                                                workspace, allocator);
  }
  if (iree_status_is_ok(status) && FLAG_list_symbols) {
    status = loom_link_cli_print_symbol_list(
        link_index.module_index, iree_make_cstring_view(FLAG_output),
        allocator);
  }
  if (iree_status_is_ok(status) && !FLAG_list_symbols) {
    status = iree_status_from_loomc(
        loomc_linker_create(context, NULL, compiler_allocator, &linker));
  }
  if (iree_status_is_ok(status) && !FLAG_list_symbols && roots.count) {
    status = iree_allocator_malloc_array(
        allocator, roots.count, sizeof(*root_symbols), (void**)&root_symbols);
    if (iree_status_is_ok(status)) {
      for (iree_host_size_t i = 0; i < roots.count; ++i) {
        root_symbols[i] = loomc_string_view_from_iree(roots.values[i]);
      }
    }
  }
  if (iree_status_is_ok(status) && !FLAG_list_symbols) {
    const loomc_link_options_t options = {
        .type = LOOMC_STRUCTURE_TYPE_LINK_OPTIONS,
        .structure_size = sizeof(options),
        .link_index = link_index.module_index,
        .mode = plan_mode,
        .root_symbols = root_symbols,
        .root_symbol_count = roots.count,
        .flags =
            (FLAG_include_input_exports ? LOOMC_LINK_FLAG_INCLUDE_INPUT_EXPORTS
                                        : 0) |
            (FLAG_include_input_tests ? LOOMC_LINK_FLAG_INCLUDE_INPUT_TESTS
                                      : 0) |
            (FLAG_strip_check ? LOOMC_LINK_FLAG_STRIP_TEST_SYMBOLS : 0) |
            (FLAG_allow_unresolved ? LOOMC_LINK_FLAG_ALLOW_UNRESOLVED_SYMBOLS
                                   : 0),
        .config = config_options,
        .root_provider_ordinals = link_index.root_providers.values,
        .root_provider_count = link_index.root_providers.count,
        .target_profile = target_profile,
        .artifact_flags =
            FLAG_print_plan ? LOOMC_LINK_ARTIFACT_FLAG_PLAN_TEXT : 0,
    };
    status = iree_status_from_loomc(loomc_link_module(
        linker, workspace, &options, &linked_module, &result));
    if (iree_status_is_ok(status)) {
      status = loom_link_cli_check_result(result, "linking failed");
    }
    if (iree_status_is_ok(status) && FLAG_print_plan) {
      status = loom_link_cli_write_report(loomc_result_artifact_at(result, 0),
                                          iree_make_cstring_view(FLAG_output),
                                          false, allocator);
    }
    loomc_result_release(result);
    result = NULL;
  }
  if (iree_status_is_ok(status) && linked_module && !FLAG_print_plan &&
      FLAG_require_resolved_config) {
    const loomc_module_config_options_t options = {
        .type = LOOMC_STRUCTURE_TYPE_MODULE_CONFIG_OPTIONS,
        .structure_size = sizeof(options),
        .config = {.flags = LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED},
    };
    status = iree_status_from_loomc(loomc_module_apply_config(
        linked_module, workspace, &options, compiler_allocator, &result));
    if (iree_status_is_ok(status)) {
      status = loom_link_cli_check_result(result, "unresolved configuration");
    }
    loomc_result_release(result);
    result = NULL;
  }
  if (iree_status_is_ok(status) && linked_module && !FLAG_print_plan &&
      FLAG_verify) {
    const loomc_module_verify_options_t options = {
        .type = LOOMC_STRUCTURE_TYPE_MODULE_VERIFY_OPTIONS,
        .structure_size = sizeof(options),
        .max_errors = 100,
        .scope = LOOMC_MODULE_VERIFY_SCOPE_STRUCTURAL,
    };
    status = iree_status_from_loomc(loomc_module_verify(
        linked_module, &options, compiler_allocator, &result));
    if (iree_status_is_ok(status)) {
      status = loom_link_cli_check_result(result,
                                          "linked module verification failed");
    }
    loomc_result_release(result);
    result = NULL;
  }
  if (iree_status_is_ok(status) && linked_module && !FLAG_print_plan) {
    if (FLAG_print_config_schema) {
      status = iree_status_from_loomc(loomc_module_query_config_schema(
          linked_module, compiler_allocator, &result));
      if (iree_status_is_ok(status)) {
        status =
            loom_link_cli_check_result(result, "config schema query failed");
      }
      if (iree_status_is_ok(status)) {
        status = loom_link_cli_write_report(loomc_result_artifact_at(result, 0),
                                            iree_make_cstring_view(FLAG_output),
                                            true, allocator);
      }
    } else {
      status = loom_link_cli_write_module_output(
          iree_make_cstring_view(FLAG_output), linked_module, output_format,
          allocator);
    }
  }

  int exit_code = 0;
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = 1;
  }
  loomc_result_release(result);
  loomc_pass_program_release(source_cleanup_program);
  loomc_module_release(linked_module);
  loomc_linker_release(linker);
  loom_link_cli_index_deinitialize(&link_index, allocator);
  iree_allocator_free(allocator, inputs);
  iree_allocator_free(allocator, root_symbols);
  iree_allocator_free(allocator, config_bindings);
  loom_config_text_binding_set_deinitialize(&config_set);
  loomc_workspace_release(workspace);
  loomc_context_release(context);
  loomc_target_profile_release(target_profile);
  loomc_target_environment_release(target_environment);
  iree_arena_block_pool_deinitialize(&block_pool);
  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
