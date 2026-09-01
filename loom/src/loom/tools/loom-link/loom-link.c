// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// loom-link: merges or links Loom text and bytecode modules through
// metadata-first planning and materialization.

#include <stdio.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/tooling/flags.h"
#include "iree/io/stream.h"
#include "iree/io/vec_stream.h"
#include "loom/codegen/low/repr.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/error/diagnostic.h"
#include "loom/format/bytecode/writer.h"
#include "loom/format/text/parser.h"
#include "loom/format/text/printer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/link/dependency_analysis.h"
#include "loom/link/dependency_report.h"
#include "loom/link/index_materializer.h"
#include "loom/link/module_index.h"
#include "loom/link/planner.h"
#include "loom/target/configured/provider.h"
#include "loom/target/entry_selection.h"
#include "loom/target/module_specialization.h"
#include "loom/target/provider.h"
#include "loom/tooling/cli/help.h"
#include "loom/tooling/config/config.h"
#include "loom/tooling/context/context.h"
#include "loom/tooling/io/file.h"
#include "loom/tools/loom-format/convert.h"
#include "loom/util/stream.h"
#include "loom/verify/verify.h"

#ifndef LOOM_LINK_HAVE_AMDGPU
#define LOOM_LINK_HAVE_AMDGPU 0
#endif  // LOOM_LINK_HAVE_AMDGPU

#if LOOM_LINK_HAVE_AMDGPU
#include "loom/target/arch/amdgpu/artifact_key.h"
#include "loom/target/arch/amdgpu/profile.h"
#endif  // LOOM_LINK_HAVE_AMDGPU

IREE_FLAG(string, mode, "auto",
          "Planning mode: auto, merge, or link. Auto selects "
          "link mode when roots, root libraries, or exported input symbols "
          "are requested.");
IREE_FLAG(string, from, "auto",
          "Input format for every input: auto, text, bc, or bytecode.");
IREE_FLAG(string, to, "text", "Output format: text, bc, or bytecode.");
IREE_FLAG(string, output, "-",
          "Output path. Use '-' or the empty string for stdout.");
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
IREE_FLAG_NAMED(
    string, target_profile, "target-profile", "",
    "Optional homogeneous target profile in family:selector form, such as "
    "amdgpu:gfx11-generic. The profile specializes each linked analysis "
    "module before template selection.");
IREE_FLAG_NAMED(bool, include_input_exports, "include-input-exports", false,
                "In link mode, add exported input symbols as roots.");
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
  // Diagnostic filename used for parser and bytecode diagnostics.
  iree_string_view_t filename;
  // Linkage and dependency classification assigned by the command line.
  loom_link_cli_input_class_t input_class;
  // Detected or forced external input format.
  loom_module_format_t format;
  // File contents kept alive while bytecode metadata borrows from it.
  iree_io_file_contents_t* contents;
  // Materialized module owned by a text input; NULL for bytecode.
  loom_module_t* materialized_module;
  // Source table entry for text diagnostics.
  loom_source_entry_t source_entry;
  // True when source_entry is valid.
  bool has_source_entry;
} loom_link_cli_input_t;

typedef struct loom_link_cli_index_t {
  // Provider-backed module index.
  loom_link_module_index_t* module_index;
  // Library providers declared as direct dependencies.
  struct {
    // Allocator-owned provider ordinals in command-line order.
    iree_host_size_t* values;
    // Number of provider ordinals in values.
    iree_host_size_t count;
  } direct_providers;
  // Providers whose exports are link roots.
  struct {
    // Allocator-owned provider ordinals in command-line order.
    iree_host_size_t* values;
    // Number of provider ordinals in values.
    iree_host_size_t count;
  } root_providers;
} loom_link_cli_index_t;

typedef struct loom_link_cli_prepare_state_t {
  // Compile-time configuration applied before each selection query.
  const loom_tooling_config_set_t* config_set;
  // Configured target environment used to project the selected profile.
  const loom_target_environment_t* target_environment;
  // Homogeneous target profile applied to every kernel entry, if any.
  const loom_target_profile_t* target_profile;
} loom_link_cli_prepare_state_t;

typedef struct loom_link_cli_target_profile_storage_t {
  // Selected target-neutral profile borrowing family-specific storage below.
  const loom_target_profile_t* profile;
#if LOOM_LINK_HAVE_AMDGPU
  // AMDGPU profile storage valid for the lifetime of the link invocation.
  loom_amdgpu_target_profile_t amdgpu;
#endif  // LOOM_LINK_HAVE_AMDGPU
} loom_link_cli_target_profile_storage_t;

static const char* loom_link_cli_mode_name(loom_link_plan_mode_t mode) {
  switch (mode) {
    case LOOM_LINK_PLAN_MERGE:
      return "merge";
    case LOOM_LINK_PLAN_LINK:
      return "link";
  }
  return "unknown";
}

static const char* loom_link_cli_role_name(loom_link_provider_role_t role) {
  switch (role) {
    case LOOM_LINK_PROVIDER_ROLE_INPUT:
      return "input";
    case LOOM_LINK_PROVIDER_ROLE_LIBRARY:
      return "library";
  }
  return "unknown";
}

static loom_link_provider_role_t loom_link_cli_input_role(
    const loom_link_cli_input_t* input) {
  return input->input_class == LOOM_LINK_CLI_INPUT_PRIMARY
             ? LOOM_LINK_PROVIDER_ROLE_INPUT
             : LOOM_LINK_PROVIDER_ROLE_LIBRARY;
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
    loom_link_symbol_identity_t identity) {
  switch (identity) {
    case LOOM_LINK_SYMBOL_IDENTITY_PRIVATE:
      return "private";
    case LOOM_LINK_SYMBOL_IDENTITY_GLOBAL:
      return "global";
  }
  return "unknown";
}

static const char* loom_link_cli_reason_name(
    loom_link_plan_live_reason_t reason) {
  switch (reason) {
    case LOOM_LINK_PLAN_LIVE_MERGE:
      return "merge";
    case LOOM_LINK_PLAN_LIVE_ROOT:
      return "root";
    case LOOM_LINK_PLAN_LIVE_DEPENDENCY:
      return "dependency";
    case LOOM_LINK_PLAN_LIVE_PROVIDER:
      return "provider";
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
    loom_link_plan_mode_t* out_mode) {
  if (cli_mode == LOOM_LINK_CLI_MODE_AUTO) {
    *out_mode =
        (roots.count > 0 || root_library_count > 0 || include_input_exports)
            ? LOOM_LINK_PLAN_LINK
            : LOOM_LINK_PLAN_MERGE;
    return iree_ok_status();
  }
  if (cli_mode == LOOM_LINK_CLI_MODE_MERGE) {
    if (roots.count > 0 || root_library_count > 0 || include_input_exports) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "merge mode does not accept --root, --root-library, or "
          "--include-input-exports");
    }
    *out_mode = LOOM_LINK_PLAN_MERGE;
    return iree_ok_status();
  }
  *out_mode = LOOM_LINK_PLAN_LINK;
  return iree_ok_status();
}

static iree_status_t loom_link_cli_append_config_flags(
    loom_tooling_config_set_t* config_set) {
  iree_flag_string_list_t assignments = FLAG_config_list();
  for (iree_host_size_t i = 0; i < assignments.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_tooling_config_set_append_assignment(
        config_set, assignments.values[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_link_cli_append_config_files(
    loom_tooling_config_set_t* config_set, iree_allocator_t allocator) {
  iree_flag_string_list_t paths = FLAG_config_file_list();
  for (iree_host_size_t i = 0; i < paths.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_tooling_config_set_append_json_file(
        config_set, paths.values[i], allocator));
  }
  return iree_ok_status();
}

static iree_status_t loom_link_cli_materialize_config(
    loom_module_t* module, const loom_tooling_config_set_t* config_set,
    iree_arena_block_pool_t* block_pool) {
  loom_tooling_config_materialize_options_t options;
  loom_tooling_config_materialize_options_initialize(&options);
  options.config_set = config_set;
  return loom_tooling_config_materialize_module(module, &options, block_pool,
                                                NULL);
}

static iree_status_t loom_link_cli_select_target_profile(
    loom_link_cli_target_profile_storage_t* out_storage) {
  *out_storage = (loom_link_cli_target_profile_storage_t){0};

  const iree_string_view_t specification =
      iree_string_view_trim(iree_make_cstring_view(FLAG_target_profile));
  if (iree_string_view_is_empty(specification)) {
    return iree_ok_status();
  }
  iree_string_view_t family = iree_string_view_empty();
  iree_string_view_t selector = iree_string_view_empty();
  if (iree_string_view_split(specification, ':', &family, &selector) < 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--target-profile='%.*s' must use family:selector syntax",
        (int)specification.size, specification.data);
  }
  family = iree_string_view_trim(family);
  selector = iree_string_view_trim(selector);
  if (iree_string_view_is_empty(family) ||
      iree_string_view_is_empty(selector)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--target-profile='%.*s' must name a non-empty family and selector",
        (int)specification.size, specification.data);
  }

#if LOOM_LINK_HAVE_AMDGPU
  if (iree_string_view_equal(family, IREE_SV("amdgpu"))) {
    loom_amdgpu_target_identity_t identity = {0};
    IREE_RETURN_IF_ERROR(loom_amdgpu_artifact_key_parse(selector, &identity));
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_target_profile_initialize(&identity, &out_storage->amdgpu));
    out_storage->profile = &out_storage->amdgpu.base;
    return iree_ok_status();
  }
#endif  // LOOM_LINK_HAVE_AMDGPU

  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "target profile family '%.*s' is not available in this loom-link "
      "binary",
      (int)family.size, family.data);
}

static loom_diagnostic_sink_t loom_link_cli_materialization_diagnostic_sink(
    void* user_data, const loom_link_module_index_provider_t* provider) {
  (void)user_data;
  (void)provider;
  return (loom_diagnostic_sink_t){.fn = loom_diagnostic_stderr_sink};
}

static iree_status_t loom_link_cli_prepare_linked_module(
    void* user_data, iree_arena_block_pool_t* block_pool,
    iree_allocator_t allocator, loom_module_t** inout_module) {
  loom_link_cli_prepare_state_t* state =
      (loom_link_cli_prepare_state_t*)user_data;
  IREE_RETURN_IF_ERROR(loom_link_cli_materialize_config(
      *inout_module, state->config_set, block_pool));
  if (state->target_profile == NULL) {
    return iree_ok_status();
  }

  const loom_target_entry_options_t diagnostic_options = {
      .diagnostic_sink = {.fn = loom_diagnostic_stderr_sink},
  };
  loom_target_entry_diagnostic_emitter_t diagnostic_emitter;
  loom_target_entry_diagnostic_emitter_initialize(
      *inout_module, &diagnostic_options, LOOM_EMITTER_PASS,
      &diagnostic_emitter);
  uint32_t error_count = 0;
  IREE_RETURN_IF_ERROR(loom_target_specialize_module_kernel_entries(
      state->target_environment, state->target_profile,
      loom_target_entry_emitter(&diagnostic_emitter), block_pool, allocator,
      inout_module, &error_count));
  if (error_count != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target profile specialization failed with %u error%s", error_count,
        error_count == 1 ? "" : "s");
  }
  return iree_ok_status();
}

static void loom_link_cli_input_deinitialize(loom_link_cli_input_t* input) {
  if (!input) {
    return;
  }
  loom_module_free(input->materialized_module);
  iree_io_file_contents_free(input->contents);
  *input = (loom_link_cli_input_t){0};
}

static void loom_link_cli_inputs_deinitialize(loom_link_cli_input_t* inputs,
                                              iree_host_size_t input_count,
                                              iree_allocator_t allocator) {
  if (!inputs) {
    return;
  }
  for (iree_host_size_t i = 0; i < input_count; ++i) {
    loom_link_cli_input_deinitialize(&inputs[i]);
  }
  iree_allocator_free(allocator, inputs);
}

static void loom_link_cli_index_deinitialize(loom_link_cli_index_t* index,
                                             iree_allocator_t allocator) {
  if (!index) {
    return;
  }
  iree_allocator_free(allocator, index->root_providers.values);
  iree_allocator_free(allocator, index->direct_providers.values);
  loom_link_module_index_free(index->module_index);
  *index = (loom_link_cli_index_t){0};
}

static iree_status_t loom_link_cli_read_input(
    iree_string_view_t path, loom_link_cli_input_class_t input_class,
    loom_module_format_t requested_format, loom_context_t* context,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_link_cli_input_t* out_input) {
  *out_input = (loom_link_cli_input_t){
      .filename =
          loom_tooling_file_path_is_stdio(path) ? IREE_SV("<stdin>") : path,
      .input_class = input_class,
  };

  IREE_RETURN_IF_ERROR(
      loom_tooling_read_input_file(path, allocator, &out_input->contents));
  loom_module_format_t format = requested_format;
  if (format == LOOM_MODULE_FORMAT_AUTO) {
    format = loom_module_format_detect_input(out_input->contents->const_buffer);
  }
  out_input->format = format;

  if (format == LOOM_MODULE_FORMAT_BYTECODE) {
    return iree_ok_status();
  }
  if (format != LOOM_MODULE_FORMAT_TEXT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported input format '%s'",
                            loom_module_format_name(format));
  }

  loom_text_parse_options_t parse_options = {
      .diagnostic_sink = {.fn = loom_diagnostic_stderr_sink},
      .max_errors = 20,
  };
  loom_target_low_descriptor_registry_t low_registry = {0};
  IREE_RETURN_IF_ERROR(
      loom_target_environment_initialize_low_descriptor_registry(
          loom_configured_target_environment(), &low_registry));
  loom_low_descriptor_text_asm_environment_initialize(
      &low_registry.registry, &parse_options.low_asm_environment);
  loom_module_t* module = NULL;
  iree_string_view_t source =
      loom_tooling_file_contents_string_view(out_input->contents);
  IREE_RETURN_IF_ERROR(loom_text_parse(source, out_input->filename, context,
                                       block_pool, &parse_options, &module));
  if (!module) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT, "failed to parse text input '%.*s'",
        (int)out_input->filename.size, out_input->filename.data);
  }

  out_input->materialized_module = module;

  loom_source_id_t source_id = LOOM_SOURCE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_register_source(
      out_input->materialized_module, out_input->filename, &source_id));
  out_input->source_entry = (loom_source_entry_t){
      .source_id = source_id,
      .source = source,
      .filename = out_input->filename,
  };
  out_input->has_source_entry = true;
  return iree_ok_status();
}

static iree_status_t loom_link_cli_read_inputs(
    int argc, char** argv, loom_module_format_t requested_format,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    iree_allocator_t allocator, loom_link_cli_input_t** out_inputs,
    iree_host_size_t* out_input_count) {
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
    status = loom_link_cli_read_input(path, LOOM_LINK_CLI_INPUT_PRIMARY,
                                      requested_format, context, block_pool,
                                      allocator, &inputs[input_ordinal++]);
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
    status = loom_link_cli_read_input(path, LOOM_LINK_CLI_INPUT_ROOT_LIBRARY,
                                      requested_format, context, block_pool,
                                      allocator, &inputs[input_ordinal++]);
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
    status = loom_link_cli_read_input(path, LOOM_LINK_CLI_INPUT_DIRECT_LIBRARY,
                                      requested_format, context, block_pool,
                                      allocator, &inputs[input_ordinal++]);
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
    status = loom_link_cli_read_input(
        path, LOOM_LINK_CLI_INPUT_TRANSITIVE_LIBRARY, requested_format, context,
        block_pool, allocator, &inputs[input_ordinal++]);
  }

  if (!iree_status_is_ok(status)) {
    loom_link_cli_inputs_deinitialize(inputs, input_count, allocator);
    return status;
  }
  *out_inputs = inputs;
  *out_input_count = input_count;
  return iree_ok_status();
}

static iree_status_t loom_link_cli_build_index(
    loom_link_cli_input_t* inputs, iree_host_size_t input_count,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    iree_allocator_t allocator, loom_link_cli_index_t* out_index) {
  *out_index = (loom_link_cli_index_t){0};
  loom_link_module_index_t* index = NULL;
  IREE_RETURN_IF_ERROR(
      loom_link_module_index_allocate(context, block_pool, allocator, &index));

  iree_host_size_t direct_provider_count = 0;
  iree_host_size_t root_provider_count = 0;
  for (iree_host_size_t i = 0; i < input_count; ++i) {
    if (loom_link_cli_input_is_direct_dependency(&inputs[i])) {
      ++direct_provider_count;
    }
    if (loom_link_cli_input_exports_are_roots(&inputs[i])) {
      ++root_provider_count;
    }
  }
  iree_host_size_t* direct_provider_ordinals = NULL;
  iree_host_size_t* root_provider_ordinals = NULL;
  iree_status_t status = iree_ok_status();
  if (direct_provider_count != 0) {
    status = iree_allocator_malloc_array(allocator, direct_provider_count,
                                         sizeof(*direct_provider_ordinals),
                                         (void**)&direct_provider_ordinals);
  }
  if (iree_status_is_ok(status) && root_provider_count != 0) {
    status = iree_allocator_malloc_array(allocator, root_provider_count,
                                         sizeof(*root_provider_ordinals),
                                         (void**)&root_provider_ordinals);
  }

  iree_host_size_t direct_provider_ordinal = 0;
  iree_host_size_t root_provider_ordinal = 0;
  for (iree_host_size_t i = 0; i < input_count && iree_status_is_ok(status);
       ++i) {
    loom_link_cli_input_t* input = &inputs[i];
    loom_link_module_index_add_options_t options = {
        .provider_name = input->filename,
        .role = loom_link_cli_input_role(input),
    };
    iree_host_size_t provider_ordinal = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
    if (input->materialized_module != NULL) {
      status = loom_link_module_index_add_materialized(
          index, input->materialized_module, &options, &provider_ordinal);
    } else {
      loom_bytecode_index_options_t index_options = {
          .diagnostic_sink = {.fn = loom_diagnostic_stderr_sink},
      };
      status = loom_link_module_index_add_bytecode(
          index, input->contents->const_buffer, input->filename, &index_options,
          &options, &provider_ordinal);
    }
    if (iree_status_is_ok(status) &&
        loom_link_cli_input_is_direct_dependency(input)) {
      direct_provider_ordinals[direct_provider_ordinal++] = provider_ordinal;
    }
    if (iree_status_is_ok(status) &&
        loom_link_cli_input_exports_are_roots(input)) {
      root_provider_ordinals[root_provider_ordinal++] = provider_ordinal;
    }
  }

  if (!iree_status_is_ok(status)) {
    iree_allocator_free(allocator, root_provider_ordinals);
    iree_allocator_free(allocator, direct_provider_ordinals);
    loom_link_module_index_free(index);
  } else {
    out_index->module_index = index;
    out_index->direct_providers.values = direct_provider_ordinals;
    out_index->direct_providers.count = direct_provider_count;
    out_index->root_providers.values = root_provider_ordinals;
    out_index->root_providers.count = root_provider_count;
  }
  return status;
}

static iree_status_t loom_link_cli_write_dependency_report(
    const loom_link_dependency_analysis_t* analysis,
    iree_string_view_t component_name, iree_string_view_t path,
    iree_allocator_t allocator) {
  loom_tooling_output_stream_t output;
  IREE_RETURN_IF_ERROR(
      loom_tooling_output_stream_open(path, allocator, &output));
  iree_status_t status = loom_link_dependency_format_json(
      analysis, component_name, &output.stream);
  if (iree_status_is_ok(status)) {
    status = loom_output_stream_write_char(&output.stream, '\n');
  }
  return iree_status_join(status, loom_tooling_output_stream_close(&output));
}

static iree_status_t loom_link_cli_print_dependency_diagnostics(
    const loom_link_dependency_analysis_t* analysis,
    iree_string_view_t component_name) {
  loom_output_stream_t stream;
  loom_output_stream_for_file(stderr, &stream);
  for (iree_host_size_t i = 0; i < analysis->requirements.count; ++i) {
    const loom_link_dependency_requirement_t* requirement =
        &analysis->requirements.values[i];
    if (loom_link_dependency_requirement_satisfied(requirement)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, "loom-link: "));
    IREE_RETURN_IF_ERROR(loom_output_stream_write(
        &stream, loom_link_dependency_diagnostic_code(requirement)));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ": "));
    IREE_RETURN_IF_ERROR(loom_link_dependency_format_diagnostic(
        analysis, requirement, component_name, &stream));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(&stream, '\n'));
  }
  return iree_ok_status();
}

static iree_status_t loom_link_cli_analyze_dependencies(
    const loom_link_cli_index_t* index, iree_string_view_t component_name,
    iree_string_view_t report_path, iree_arena_block_pool_t* block_pool,
    iree_allocator_t allocator, bool* out_succeeded) {
  *out_succeeded = false;
  iree_arena_allocator_t arena;
  iree_arena_initialize(block_pool, &arena);
  const loom_link_dependency_analysis_options_t options = {
      .direct_provider_ordinals = index->direct_providers.values,
      .direct_provider_count = index->direct_providers.count,
  };
  loom_link_dependency_analysis_t analysis = {0};
  iree_status_t status = loom_link_dependency_analyze(
      index->module_index, &options, block_pool, &arena, allocator, &analysis);
  bool succeeded = false;
  if (iree_status_is_ok(status)) {
    succeeded = loom_link_dependency_analysis_succeeded(&analysis);
  }
  if (iree_status_is_ok(status) && !iree_string_view_is_empty(report_path)) {
    status = loom_link_cli_write_dependency_report(&analysis, component_name,
                                                   report_path, allocator);
  }
  if (iree_status_is_ok(status) && !succeeded) {
    status =
        loom_link_cli_print_dependency_diagnostics(&analysis, component_name);
  }
  if (iree_status_is_ok(status)) {
    *out_succeeded = succeeded;
  }
  iree_arena_deinitialize(&arena);
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
    iree_string_builder_t* builder, loom_link_symbol_flags_t flags) {
  bool needs_separator = false;
  if (iree_any_bit_set(flags, LOOM_LINK_SYMBOL_FLAG_PUBLIC)) {
    IREE_RETURN_IF_ERROR(loom_link_cli_append_flag(builder, &needs_separator,
                                                   IREE_SV("public")));
  }
  if (iree_any_bit_set(flags, LOOM_LINK_SYMBOL_FLAG_IMPORT)) {
    IREE_RETURN_IF_ERROR(loom_link_cli_append_flag(builder, &needs_separator,
                                                   IREE_SV("import")));
  }
  if (iree_any_bit_set(flags, LOOM_LINK_SYMBOL_FLAG_EXPORT)) {
    IREE_RETURN_IF_ERROR(loom_link_cli_append_flag(builder, &needs_separator,
                                                   IREE_SV("export")));
  }
  if (iree_any_bit_set(flags, LOOM_LINK_SYMBOL_FLAG_DECLARATION)) {
    IREE_RETURN_IF_ERROR(loom_link_cli_append_flag(builder, &needs_separator,
                                                   IREE_SV("declaration")));
  }
  if (iree_any_bit_set(flags, LOOM_LINK_SYMBOL_FLAG_CONCRETE_DEFINITION)) {
    IREE_RETURN_IF_ERROR(loom_link_cli_append_flag(builder, &needs_separator,
                                                   IREE_SV("definition")));
  }
  if (iree_any_bit_set(flags, LOOM_LINK_SYMBOL_FLAG_CONFIG)) {
    IREE_RETURN_IF_ERROR(loom_link_cli_append_flag(builder, &needs_separator,
                                                   IREE_SV("config")));
  }
  if (iree_any_bit_set(flags, LOOM_LINK_SYMBOL_FLAG_TEST_ONLY)) {
    IREE_RETURN_IF_ERROR(loom_link_cli_append_flag(builder, &needs_separator,
                                                   IREE_SV("test-only")));
  }
  if (!needs_separator) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "-"));
  }
  return iree_ok_status();
}

static iree_status_t loom_link_cli_print_symbol_list(
    const loom_link_module_index_t* index, iree_string_view_t output_path,
    iree_allocator_t allocator) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(allocator, &builder);

  iree_status_t status = iree_string_builder_append_format(
      &builder, "symbols %" PRIhsz "\n",
      loom_link_module_index_symbol_count(index));
  for (iree_host_size_t i = 0; i < loom_link_module_index_symbol_count(index) &&
                               iree_status_is_ok(status);
       ++i) {
    const loom_link_module_index_symbol_t* symbol =
        loom_link_module_index_symbol_at(index, i);
    const loom_link_module_index_provider_t* provider =
        loom_link_module_index_symbol_provider(index, symbol);
    const loom_link_module_index_module_t* module =
        loom_link_module_index_symbol_module(index, symbol);
    status = iree_string_builder_append_format(
        &builder,
        "[%" PRIhsz
        "] @%.*s role=%s provider=%.*s module=%.*s identity=%s "
        "flags=",
        i, (int)symbol->name.size, symbol->name.data,
        loom_link_cli_role_name(provider ? provider->role
                                         : LOOM_LINK_PROVIDER_ROLE_INPUT),
        provider ? (int)provider->name.size : 0,
        provider ? provider->name.data : "",
        module ? (int)module->name.size : 0, module ? module->name.data : "",
        loom_link_cli_identity_name(symbol->identity));
    if (iree_status_is_ok(status)) {
      status = loom_link_cli_append_symbol_flags(&builder, symbol->flags);
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

static iree_status_t loom_link_cli_print_plan(const loom_link_plan_t* plan,
                                              loom_link_plan_mode_t mode,
                                              iree_string_view_t output_path,
                                              iree_allocator_t allocator) {
  const loom_link_module_index_t* index = loom_link_plan_index(plan);
  iree_string_builder_t builder;
  iree_string_builder_initialize(allocator, &builder);

  iree_status_t status = iree_string_builder_append_format(
      &builder, "plan mode=%s symbols=%" PRIhsz "\n",
      loom_link_cli_mode_name(mode), loom_link_plan_symbol_count(plan));
  for (iree_host_size_t i = 0;
       i < loom_link_plan_symbol_count(plan) && iree_status_is_ok(status);
       ++i) {
    const loom_link_plan_symbol_t* planned_symbol =
        loom_link_plan_symbol_at(plan, i);
    const loom_link_module_index_symbol_t* symbol =
        loom_link_module_index_symbol_at(index, planned_symbol->symbol_ordinal);
    const loom_link_module_index_provider_t* provider =
        loom_link_module_index_symbol_provider(index, symbol);
    const loom_link_module_index_module_t* module =
        loom_link_module_index_symbol_module(index, symbol);
    status = iree_string_builder_append_format(
        &builder, "[%" PRIhsz "] %s @%.*s provider=%.*s module=%.*s cause=", i,
        loom_link_cli_reason_name(planned_symbol->reason),
        (int)symbol->name.size, symbol->name.data,
        provider ? (int)provider->name.size : 0,
        provider ? provider->name.data : "",
        module ? (int)module->name.size : 0, module ? module->name.data : "");
    if (iree_status_is_ok(status)) {
      if (planned_symbol->cause_ordinal ==
          LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL) {
        status = iree_string_builder_append_cstring(&builder, "-");
      } else {
        status = iree_string_builder_append_format(
            &builder, "%" PRIhsz, planned_symbol->cause_ordinal);
      }
    }
    if (iree_status_is_ok(status)) {
      status = iree_string_builder_append_cstring(&builder, " flags=");
    }
    if (iree_status_is_ok(status)) {
      status = loom_link_cli_append_symbol_flags(&builder, symbol->flags);
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

static iree_status_t loom_link_cli_verify_output(
    const loom_source_entry_t* source_entries, iree_host_size_t source_count,
    loom_module_t* module) {
  loom_source_table_resolver_t source_resolver = {
      .entries = source_entries,
      .count = source_count,
  };
  loom_verify_options_t verify_options = {
      .sink = {.fn = loom_diagnostic_stderr_sink},
      .max_errors = 100,
      .source_resolver = {.fn = loom_source_table_resolve,
                          .user_data = &source_resolver},
  };
  loom_verify_result_t verify_result = {0};
  IREE_RETURN_IF_ERROR(
      loom_verify_module(module, &verify_options, &verify_result));
  if (verify_result.error_count > 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "linked module verification failed with %u error%s",
                            verify_result.error_count,
                            verify_result.error_count == 1 ? "" : "s");
  }
  return iree_ok_status();
}

static iree_status_t loom_link_cli_collect_source_entries(
    const loom_link_cli_input_t* inputs, iree_host_size_t input_count,
    loom_source_entry_t** out_source_entries,
    iree_host_size_t* out_source_count, iree_allocator_t allocator) {
  *out_source_entries = NULL;
  *out_source_count = 0;
  iree_host_size_t source_count = 0;
  for (iree_host_size_t i = 0; i < input_count; ++i) {
    if (inputs[i].has_source_entry) {
      ++source_count;
    }
  }
  if (source_count == 0) {
    return iree_ok_status();
  }

  loom_source_entry_t* source_entries = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(allocator, source_count,
                                                   sizeof(*source_entries),
                                                   (void**)&source_entries));
  iree_host_size_t source_ordinal = 0;
  for (iree_host_size_t i = 0; i < input_count; ++i) {
    if (inputs[i].has_source_entry) {
      source_entries[source_ordinal++] = inputs[i].source_entry;
    }
  }
  *out_source_entries = source_entries;
  *out_source_count = source_count;
  return iree_ok_status();
}

static iree_status_t loom_link_cli_write_text_output(
    const loom_module_t* module, loom_format_output_t* out_output,
    iree_allocator_t allocator) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(allocator, &builder);

  iree_status_t status = loom_text_print_module_to_builder(
      module, &builder, LOOM_TEXT_PRINT_DEFAULT);
  if (iree_status_is_ok(status)) {
    out_output->length = iree_string_builder_size(&builder);
    out_output->data = (uint8_t*)iree_string_builder_take_storage(&builder);
  }

  iree_string_builder_deinitialize(&builder);
  return status;
}

static iree_status_t loom_link_cli_write_bytecode_output(
    const loom_module_t* module, iree_arena_block_pool_t* block_pool,
    loom_format_output_t* out_output, iree_allocator_t allocator) {
  iree_io_stream_t* stream = NULL;
  iree_status_t status = iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE |
          IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_RESIZABLE,
      4096, allocator, &stream);

  loom_bytecode_write_options_t write_options = {
      .producer = IREE_SV("loom-link"),
      .location_mode = LOOM_BYTECODE_LOCATION_MODE_SOURCE_LOCATIONS,
  };
  loom_target_low_descriptor_registry_t low_registry = {0};
  if (iree_status_is_ok(status)) {
    status = loom_target_environment_initialize_low_descriptor_registry(
        loom_configured_target_environment(), &low_registry);
  }
  loom_low_repr_environment_initialize(&low_registry.registry,
                                       &write_options.low_repr_environment);
  if (iree_status_is_ok(status)) {
    status =
        loom_bytecode_write_module(module, stream, &write_options, block_pool);
  }

  iree_io_stream_pos_t stream_length = 0;
  if (iree_status_is_ok(status)) {
    stream_length = iree_io_stream_length(stream);
    if (stream_length < 0 || stream_length > IREE_HOST_SIZE_MAX) {
      status =
          iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                           "bytecode output length is not host-addressable");
    }
  }

  if (iree_status_is_ok(status)) {
    out_output->length = (iree_host_size_t)stream_length;
    if (out_output->length > 0) {
      status = iree_allocator_malloc(allocator, out_output->length,
                                     (void**)&out_output->data);
    }
  }
  if (iree_status_is_ok(status) && out_output->length > 0) {
    status = iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0);
  }
  if (iree_status_is_ok(status) && out_output->length > 0) {
    status =
        iree_io_stream_read(stream, out_output->length, out_output->data, NULL);
  }

  iree_io_stream_release(stream);
  if (!iree_status_is_ok(status)) {
    loom_format_output_deinitialize(out_output, allocator);
  }
  return status;
}

static iree_status_t loom_link_cli_write_module_output(
    iree_string_view_t path, const loom_module_t* module,
    loom_module_format_t output_format, iree_arena_block_pool_t* block_pool,
    iree_allocator_t allocator) {
  loom_format_output_t output = {0};
  iree_status_t status = iree_ok_status();
  switch (output_format) {
    case LOOM_MODULE_FORMAT_TEXT:
      status = loom_link_cli_write_text_output(module, &output, allocator);
      break;
    case LOOM_MODULE_FORMAT_BYTECODE:
      status = loom_link_cli_write_bytecode_output(module, block_pool, &output,
                                                   allocator);
      break;
    case LOOM_MODULE_FORMAT_AUTO:
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "output format must be explicit");
      break;
  }
  if (iree_status_is_ok(status)) {
    status = loom_tooling_write_output_file(
        path, iree_make_string_view((const char*)output.data, output.length),
        allocator);
  }
  loom_format_output_deinitialize(&output, allocator);
  return status;
}

static iree_status_t loom_link_cli_print_config_schema(
    iree_string_view_t path, const loom_module_t* module,
    iree_allocator_t allocator) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(allocator, &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  iree_status_t status =
      loom_tooling_config_format_schema_json(module, &stream);
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_string(&builder, IREE_SV("\n"));
  }
  if (iree_status_is_ok(status)) {
    status = loom_tooling_write_output_file(
        path, iree_string_builder_view(&builder), allocator);
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

static void loom_link_cli_print_agents_markdown(FILE* stream) {
  fprintf(
      stream,
      "## loom-link\n"
      "\n"
      "`loom-link` combines Loom text and bytecode modules, applies config\n"
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
      "input decoding and `--to=text|bc` controls output encoding.\n"
      "\n"
      "### Merge and link\n"
      "\n"
      "`--mode=merge` preserves every non-stripped primary-input symbol in\n"
      "input order. `--mode=link` keeps explicit `--root=@symbol` values,\n"
      "exports from each `--root-library`, optional\n"
      "`--include-input-exports`, and reachable dependencies.\n"
      "A public library definition selected only as a dependency becomes\n"
      "private in the output; library visibility does not implicitly "
      "re-export\n"
      "it through the requester. Select it as a root when it is also an "
      "output.\n"
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
      "  --target-profile=amdgpu:gfx11-generic \\\n"
      "  --require-resolved-config\n"
      "loom-link root.loom --library=provider-a.loom --root=@entry \\\n"
      "  --allow-unresolved --to=bc --output=partial.loombc\n"
      "```\n"
      "\n"
      "`--list-symbols` shows the indexed providers. `--print-plan` shows why\n"
      "each symbol is live before streaming modules into the linker. Config\n"
      "bindings are applied to the composed analysis module before each "
      "dependency and template-selection step. `--target-profile` applies "
      "structured target facts at the same boundary, so target and shape "
      "predicates can prune unreachable provider templates.\n"
      "`--allow-unresolved` preserves unresolved declarations whose libraries\n"
      "were not supplied so the output can be linked again.\n");
}

int main(int argc, char** argv) {
  iree_flags_set_usage(
      "loom-link",
      "Links Loom text and bytecode modules into one module.\n"
      "\n"
      "Usage:\n"
      "  loom-link [--mode=merge|link] [--from=auto|text|bc] "
      "[--to=text|bc] [--target-profile=family:selector] "
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

  loom_context_t context = {0};
  bool context_initialized = false;
  loom_tooling_config_set_t config_set;
  loom_tooling_config_set_initialize(allocator, &config_set);
  loom_link_cli_target_profile_storage_t target_profile = {0};
  loom_link_cli_input_t* inputs = NULL;
  iree_host_size_t input_count = 0;
  loom_link_cli_index_t link_index = {0};
  loom_link_index_materialization_t materialization = {0};
  loom_source_entry_t* source_entries = NULL;
  iree_host_size_t source_count = 0;

  loom_module_format_t input_format = LOOM_MODULE_FORMAT_AUTO;
  loom_module_format_t output_format = LOOM_MODULE_FORMAT_TEXT;
  loom_link_cli_mode_t cli_mode = LOOM_LINK_CLI_MODE_AUTO;
  loom_link_plan_mode_t plan_mode = LOOM_LINK_PLAN_MERGE;
  const iree_flag_string_list_t roots = FLAG_root_list();
  const iree_string_view_t dependency_report_path =
      iree_make_cstring_view(FLAG_dependency_report);
  const iree_string_view_t requested_dependency_component =
      iree_make_cstring_view(FLAG_dependency_component);
  const bool dependency_analysis_requested =
      FLAG_strict_deps || !iree_string_view_is_empty(dependency_report_path);
  bool dependency_analysis_succeeded = true;

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
        FLAG_include_input_exports, &plan_mode);
  }
  if (iree_status_is_ok(status) && dependency_analysis_requested &&
      plan_mode != LOOM_LINK_PLAN_MERGE) {
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
    status = loom_link_cli_select_target_profile(&target_profile);
  }
  if (iree_status_is_ok(status)) {
    loom_context_initialize(allocator, &context);
    context_initialized = true;
    status =
        loom_tooling_context_register_tool_dialects_with_target_environment(
            loom_configured_target_environment(), &context);
  }
  if (iree_status_is_ok(status)) {
    status = loom_context_finalize(&context);
  }
  if (iree_status_is_ok(status)) {
    status = loom_link_cli_append_config_files(&config_set, allocator);
  }
  if (iree_status_is_ok(status)) {
    status = loom_link_cli_append_config_flags(&config_set);
  }
  if (iree_status_is_ok(status)) {
    status = loom_link_cli_read_inputs(argc, argv, input_format, &context,
                                       &block_pool, allocator, &inputs,
                                       &input_count);
  }
  if (iree_status_is_ok(status)) {
    status = loom_link_cli_build_index(inputs, input_count, &context,
                                       &block_pool, allocator, &link_index);
  }
  if (iree_status_is_ok(status) && dependency_analysis_requested) {
    iree_string_view_t component_name = requested_dependency_component;
    if (iree_string_view_is_empty(component_name)) {
      for (iree_host_size_t i = 0; i < input_count; ++i) {
        if (inputs[i].input_class == LOOM_LINK_CLI_INPUT_PRIMARY) {
          component_name = inputs[i].filename;
          break;
        }
      }
    }
    status = loom_link_cli_analyze_dependencies(
        &link_index, component_name, dependency_report_path, &block_pool,
        allocator, &dependency_analysis_succeeded);
  }
  if (iree_status_is_ok(status) && dependency_analysis_succeeded &&
      FLAG_list_symbols) {
    status = loom_link_cli_print_symbol_list(
        link_index.module_index, iree_make_cstring_view(FLAG_output),
        allocator);
  }

  loom_link_plan_options_t plan_options = {
      .mode = plan_mode,
      .root_symbols = roots,
      .include_input_exports = FLAG_include_input_exports,
      .root_providers =
          {
              .count = link_index.root_providers.count,
              .values = link_index.root_providers.values,
          },
      .unresolved_policy = FLAG_allow_unresolved
                               ? LOOM_LINK_PLAN_UNRESOLVED_ALLOW
                               : LOOM_LINK_PLAN_UNRESOLVED_ERROR,
      .test_symbol_policy = FLAG_strip_check ? LOOM_LINK_PLAN_TEST_SYMBOL_STRIP
                                             : LOOM_LINK_PLAN_TEST_SYMBOL_KEEP,
  };
  if (iree_status_is_ok(status) && dependency_analysis_succeeded &&
      !FLAG_list_symbols) {
    loom_target_low_descriptor_registry_t low_registry = {0};
    status = loom_target_environment_initialize_low_descriptor_registry(
        loom_configured_target_environment(), &low_registry);
    loom_low_repr_environment_t low_repr_environment = {0};
    if (iree_status_is_ok(status)) {
      loom_low_repr_environment_initialize(&low_registry.registry,
                                           &low_repr_environment);
    }
    loom_link_cli_prepare_state_t prepare_state = {
        .config_set = &config_set,
        .target_environment = loom_configured_target_environment(),
        .target_profile = target_profile.profile,
    };
    const loom_link_plan_materialization_environment_t environment = {
        .context = &context,
        .block_pool = &block_pool,
        .low_repr_environment = low_repr_environment,
        .diagnostic_sink = loom_link_cli_materialization_diagnostic_sink,
        .prepare_module = loom_link_cli_prepare_linked_module,
        .user_data = &prepare_state,
        .allocator = allocator,
    };
    if (iree_status_is_ok(status)) {
      status = loom_link_index_materialize(link_index.module_index,
                                           &plan_options, &environment,
                                           IREE_SV("linked"), &materialization);
    }
  }
  if (iree_status_is_ok(status) && dependency_analysis_succeeded &&
      FLAG_print_plan) {
    status = loom_link_cli_print_plan(materialization.plan, plan_mode,
                                      iree_make_cstring_view(FLAG_output),
                                      allocator);
  }
  loom_module_t* linked_module =
      FLAG_print_plan ? NULL : materialization.product.module;
  if (iree_status_is_ok(status) && linked_module != NULL) {
    status = loom_link_cli_collect_source_entries(
        inputs, input_count, &source_entries, &source_count, allocator);
  }
  if (iree_status_is_ok(status) && linked_module &&
      FLAG_require_resolved_config) {
    status = loom_tooling_config_require_resolved_module(linked_module, NULL);
  }
  if (iree_status_is_ok(status) && linked_module && FLAG_verify) {
    status = loom_link_cli_verify_output(source_entries, source_count,
                                         linked_module);
  }
  if (iree_status_is_ok(status) && linked_module) {
    if (FLAG_print_config_schema) {
      status = loom_link_cli_print_config_schema(
          iree_make_cstring_view(FLAG_output), linked_module, allocator);
    } else {
      status = loom_link_cli_write_module_output(
          iree_make_cstring_view(FLAG_output), linked_module, output_format,
          &block_pool, allocator);
    }
  }

  int exit_code = 0;
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = 1;
  } else if (!dependency_analysis_succeeded) {
    exit_code = 1;
  }

  iree_allocator_free(allocator, source_entries);
  loom_link_index_materialization_deinitialize(&materialization);
  loom_link_cli_index_deinitialize(&link_index, allocator);
  loom_link_cli_inputs_deinitialize(inputs, input_count, allocator);
  loom_tooling_config_set_deinitialize(&config_set);
  if (context_initialized) {
    loom_context_deinitialize(&context);
  }
  iree_arena_block_pool_deinitialize(&block_pool);

  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
