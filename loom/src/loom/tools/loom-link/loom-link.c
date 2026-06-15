// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// loom-link: links Loom text and bytecode modules through the provider index
// and planner before streaming selected modules into the incremental linker.

#include <stdio.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/tooling/flags.h"
#include "iree/io/stream.h"
#include "iree/io/vec_stream.h"
#include "loom/error/diagnostic.h"
#include "loom/format/bytecode/reader.h"
#include "loom/format/bytecode/writer.h"
#include "loom/format/text/parser.h"
#include "loom/format/text/printer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/link/linker.h"
#include "loom/link/module_index.h"
#include "loom/link/planner.h"
#include "loom/target/configured/provider.h"
#include "loom/tooling/cli/help.h"
#include "loom/tooling/config/config.h"
#include "loom/tooling/context/context.h"
#include "loom/tooling/io/file.h"
#include "loom/tools/loom-format/convert.h"
#include "loom/util/stream.h"
#include "loom/verify/verify.h"

IREE_FLAG(string, mode, "auto",
          "Planning mode: auto, archive, link, or selective. Auto selects "
          "link mode when roots or exported roots are requested.");
IREE_FLAG(string, from, "auto",
          "Input format for every input: auto, text, bc, or bytecode.");
IREE_FLAG(string, to, "text", "Output format: text, bc, or bytecode.");
IREE_FLAG(string, output, "-",
          "Output path. Use '-' or the empty string for stdout.");
IREE_FLAG_LIST(string, root,
               "Root symbol to materialize in link/selective mode. Repeat for "
               "multiple roots.");
IREE_FLAG_LIST(string, library,
               "Library input searched after primary inputs. Repeat for "
               "multiple libraries.");
IREE_FLAG_LIST(string, config,
               "Compile/link-time config binding. Repeat as "
               "--config=key=value. Bindings specialize each materialized "
               "input module before linking; unused bindings are ignored.");
IREE_FLAG_LIST_NAMED(
    string, config_file, "config-file",
    "JSON/JSONC config object file. Repeat for multiple files. Nested object "
    "keys are flattened with '.' separators.");
IREE_FLAG_NAMED(bool, include_exported_roots, "include-exported-roots", false,
                "In link/selective mode, add exported symbols as roots.");
IREE_FLAG_NAMED(bool, strip_check, "strip-check", false,
                "Strip check.case and check.benchmark symbols before output.");
IREE_FLAG_NAMED(
    bool, require_resolved_config, "require-resolved-config", false,
    "Require all config.decl symbols to be materialized before output.");
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
  LOOM_LINK_CLI_MODE_ARCHIVE = 1,
  LOOM_LINK_CLI_MODE_SELECTIVE = 2,
} loom_link_cli_mode_t;

typedef struct loom_link_cli_input_t {
  // Diagnostic filename used for parser and bytecode diagnostics.
  iree_string_view_t filename;
  // Provider role assigned to this input.
  loom_link_provider_role_t role;
  // Detected or forced external input format.
  loom_module_format_t format;
  // File contents kept alive while bytecode metadata borrows from it.
  iree_io_file_contents_t* contents;
  // Materialized modules owned by this input.
  loom_module_t** materialized_modules;
  // Number of entries in materialized_modules.
  iree_host_size_t materialized_module_count;
  // Source table entry for text diagnostics.
  loom_source_entry_t source_entry;
  // True when source_entry is valid.
  bool has_source_entry;
} loom_link_cli_input_t;

typedef struct loom_link_cli_provider_state_t {
  // Input ordinal that produced this provider.
  iree_host_size_t input_ordinal;
} loom_link_cli_provider_state_t;

typedef struct loom_link_cli_index_t {
  // Provider-backed symbol index.
  loom_link_module_index_t* index;
  // Provider states parallel to index provider ordinals.
  loom_link_cli_provider_state_t* provider_states;
  // Number of valid provider state entries.
  iree_host_size_t provider_state_count;
  // Allocated provider state capacity.
  iree_host_size_t provider_state_capacity;
} loom_link_cli_index_t;

typedef struct loom_link_cli_module_roots_t {
  // Borrowed root symbol names selected for one indexed module.
  iree_string_view_t* values;
  // Number of root names.
  iree_host_size_t count;
  // Allocated root value capacity.
  iree_host_size_t capacity;
  // True after this indexed module has been streamed into the linker.
  bool linked;
} loom_link_cli_module_roots_t;

typedef struct loom_link_cli_plan_materializer_t {
  // Provider-backed symbol index used by the active plan.
  const loom_link_cli_index_t* cli_index;
  // Input records that own lazily materialized bytecode modules.
  loom_link_cli_input_t* inputs;
  // Number of input records.
  iree_host_size_t input_count;
  // Config bindings applied to every materialized module.
  const loom_tooling_config_set_t* config_set;
  // Context shared by all modules in the active link.
  loom_context_t* context;
  // Block pool used by bytecode materialization.
  iree_arena_block_pool_t* block_pool;
  // Host allocator for module object ownership.
  iree_allocator_t allocator;
} loom_link_cli_plan_materializer_t;

static const char* loom_link_cli_mode_name(loom_link_plan_mode_t mode) {
  switch (mode) {
    case LOOM_LINK_PLAN_ARCHIVE:
      return "archive";
    case LOOM_LINK_PLAN_SELECTIVE:
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
    case LOOM_LINK_PLAN_LIVE_ARCHIVE:
      return "archive";
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
  if (iree_string_view_equal(value, IREE_SV("archive"))) {
    *out_mode = LOOM_LINK_CLI_MODE_ARCHIVE;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("link")) ||
      iree_string_view_equal(value, IREE_SV("selective"))) {
    *out_mode = LOOM_LINK_CLI_MODE_SELECTIVE;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "unsupported link mode '%.*s'; expected auto, archive, link, or "
      "selective",
      (int)value.size, value.data);
}

static iree_status_t loom_link_cli_resolve_plan_mode(
    loom_link_cli_mode_t cli_mode, const iree_flag_string_list_t roots,
    bool include_exported_roots, loom_link_plan_mode_t* out_mode) {
  if (cli_mode == LOOM_LINK_CLI_MODE_AUTO) {
    *out_mode = (roots.count > 0 || include_exported_roots)
                    ? LOOM_LINK_PLAN_SELECTIVE
                    : LOOM_LINK_PLAN_ARCHIVE;
    return iree_ok_status();
  }
  if (cli_mode == LOOM_LINK_CLI_MODE_ARCHIVE) {
    if (roots.count > 0 || include_exported_roots) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "archive mode does not accept --root or --include-exported-roots");
    }
    *out_mode = LOOM_LINK_PLAN_ARCHIVE;
    return iree_ok_status();
  }
  *out_mode = LOOM_LINK_PLAN_SELECTIVE;
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

static void loom_link_cli_input_deinitialize(loom_link_cli_input_t* input,
                                             iree_allocator_t allocator) {
  if (!input) {
    return;
  }
  for (iree_host_size_t i = 0; i < input->materialized_module_count; ++i) {
    loom_module_free(input->materialized_modules[i]);
  }
  iree_allocator_free(allocator, input->materialized_modules);
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
    loom_link_cli_input_deinitialize(&inputs[i], allocator);
  }
  iree_allocator_free(allocator, inputs);
}

static iree_status_t loom_link_cli_read_input(
    iree_string_view_t path, loom_link_provider_role_t role,
    loom_module_format_t requested_format, loom_context_t* context,
    iree_arena_block_pool_t* block_pool,
    const loom_tooling_config_set_t* config_set, iree_allocator_t allocator,
    loom_link_cli_input_t* out_input) {
  *out_input = (loom_link_cli_input_t){
      .filename =
          loom_tooling_file_path_is_stdio(path) ? IREE_SV("<stdin>") : path,
      .role = role,
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

  iree_status_t status =
      loom_link_cli_materialize_config(module, config_set, block_pool);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(allocator,
                                   sizeof(*out_input->materialized_modules),
                                   (void**)&out_input->materialized_modules);
  }
  if (iree_status_is_ok(status)) {
    out_input->materialized_modules[0] = module;
    out_input->materialized_module_count = 1;
    module = NULL;
  }
  loom_module_free(module);
  if (!iree_status_is_ok(status)) {
    return status;
  }

  loom_source_id_t source_id = LOOM_SOURCE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_register_source(
      out_input->materialized_modules[0], out_input->filename, &source_id));
  out_input->source_entry = (loom_source_entry_t){
      .source_id = source_id,
      .source = source,
      .filename = out_input->filename,
  };
  out_input->has_source_entry = true;
  return iree_ok_status();
}

static iree_status_t loom_link_cli_materialize_bytecode_module(
    const loom_link_cli_input_t* input, iree_host_size_t module_ordinal,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    const loom_tooling_config_set_t* config_set, iree_allocator_t allocator,
    loom_module_t** out_module) {
  *out_module = NULL;
  if (module_ordinal > UINT16_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "bytecode module ordinal %" PRIhsz " is not encodable", module_ordinal);
  }

  loom_bytecode_read_options_t read_options = {
      .diagnostic_sink = {.fn = loom_diagnostic_stderr_sink},
      .verify_module = false,
      .verify_max_errors = 0,
  };
  loom_bytecode_read_result_t read_result = {0};
  loom_module_t* module = NULL;
  IREE_RETURN_IF_ERROR(loom_bytecode_read_module_ordinal(
      input->contents->const_buffer, input->filename, context, block_pool,
      (uint16_t)module_ordinal, &read_options, &read_result, &module,
      allocator));
  if (read_result.error_count > 0 || !module) {
    loom_module_free(module);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "failed to read bytecode module %" PRIhsz " from '%.*s'",
        module_ordinal, (int)input->filename.size, input->filename.data);
  }

  iree_status_t status =
      loom_link_cli_materialize_config(module, config_set, block_pool);
  if (!iree_status_is_ok(status)) {
    loom_module_free(module);
    return status;
  }
  *out_module = module;
  return iree_ok_status();
}

static iree_status_t loom_link_cli_read_inputs(
    int argc, char** argv, loom_module_format_t requested_format,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    const loom_tooling_config_set_t* config_set, iree_allocator_t allocator,
    loom_link_cli_input_t** out_inputs, iree_host_size_t* out_input_count) {
  *out_inputs = NULL;
  *out_input_count = 0;

  iree_flag_string_list_t libraries = FLAG_library_list();
  const iree_host_size_t primary_count =
      argc < 2 && libraries.count == 0 ? 1 : (iree_host_size_t)(argc - 1);
  iree_host_size_t input_count = 0;
  if (!iree_host_size_checked_add(primary_count, libraries.count,
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
    status = loom_link_cli_read_input(
        path, LOOM_LINK_PROVIDER_ROLE_INPUT, requested_format, context,
        block_pool, config_set, allocator, &inputs[input_ordinal++]);
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
    status = loom_link_cli_read_input(
        path, LOOM_LINK_PROVIDER_ROLE_LIBRARY, requested_format, context,
        block_pool, config_set, allocator, &inputs[input_ordinal++]);
  }

  if (!iree_status_is_ok(status)) {
    loom_link_cli_inputs_deinitialize(inputs, input_count, allocator);
    return status;
  }
  *out_inputs = inputs;
  *out_input_count = input_count;
  return iree_ok_status();
}

static void loom_link_cli_index_deinitialize(loom_link_cli_index_t* cli_index,
                                             iree_allocator_t allocator) {
  if (!cli_index) {
    return;
  }
  loom_link_module_index_free(cli_index->index);
  iree_allocator_free(allocator, cli_index->provider_states);
  *cli_index = (loom_link_cli_index_t){0};
}

static iree_status_t loom_link_cli_reserve_provider_states(
    loom_link_cli_index_t* cli_index, iree_host_size_t count,
    iree_allocator_t allocator) {
  if (count <= cli_index->provider_state_capacity) {
    return iree_ok_status();
  }
  return iree_allocator_grow_array(
      allocator, count, sizeof(*cli_index->provider_states),
      &cli_index->provider_state_capacity, (void**)&cli_index->provider_states);
}

static iree_status_t loom_link_cli_set_provider_state(
    loom_link_cli_index_t* cli_index, iree_host_size_t provider_ordinal,
    iree_host_size_t input_ordinal, iree_allocator_t allocator) {
  IREE_RETURN_IF_ERROR(loom_link_cli_reserve_provider_states(
      cli_index, provider_ordinal + 1, allocator));
  cli_index->provider_states[provider_ordinal] =
      (loom_link_cli_provider_state_t){
          .input_ordinal = input_ordinal,
      };
  if (provider_ordinal >= cli_index->provider_state_count) {
    cli_index->provider_state_count = provider_ordinal + 1;
  }
  return iree_ok_status();
}

static iree_status_t loom_link_cli_materialized_provider_name(
    const loom_link_cli_input_t* input, iree_host_size_t module_ordinal,
    iree_string_builder_t* scratch_builder, iree_string_view_t* out_name) {
  if (input->materialized_module_count <= 1) {
    *out_name = input->filename;
    return iree_ok_status();
  }
  iree_string_builder_reset(scratch_builder);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      scratch_builder, "%.*s#%" PRIhsz, (int)input->filename.size,
      input->filename.data, module_ordinal));
  *out_name = iree_string_builder_view(scratch_builder);
  return iree_ok_status();
}

static iree_status_t loom_link_cli_build_index(
    loom_link_cli_input_t* inputs, iree_host_size_t input_count,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    iree_allocator_t allocator, loom_link_cli_index_t* out_cli_index) {
  *out_cli_index = (loom_link_cli_index_t){0};
  IREE_RETURN_IF_ERROR(loom_link_module_index_create(
      context, block_pool, allocator, &out_cli_index->index));

  iree_string_builder_t provider_name_builder;
  iree_string_builder_initialize(allocator, &provider_name_builder);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < input_count && iree_status_is_ok(status);
       ++i) {
    loom_link_cli_input_t* input = &inputs[i];
    if (input->materialized_module_count > 0) {
      for (iree_host_size_t j = 0;
           j < input->materialized_module_count && iree_status_is_ok(status);
           ++j) {
        iree_string_view_t provider_name = iree_string_view_empty();
        status = loom_link_cli_materialized_provider_name(
            input, j, &provider_name_builder, &provider_name);
        iree_host_size_t provider_ordinal =
            LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
        if (iree_status_is_ok(status)) {
          loom_link_module_index_add_options_t options = {
              .provider_name = provider_name,
              .role = input->role,
          };
          status = loom_link_module_index_add_materialized(
              out_cli_index->index, input->materialized_modules[j], &options,
              &provider_ordinal);
        }
        if (iree_status_is_ok(status)) {
          status = loom_link_cli_set_provider_state(
              out_cli_index, provider_ordinal, i, allocator);
        }
      }
      continue;
    }

    loom_bytecode_read_options_t read_options = {
        .diagnostic_sink = {.fn = loom_diagnostic_stderr_sink},
    };
    loom_link_module_index_add_options_t options = {
        .provider_name = input->filename,
        .role = input->role,
    };
    iree_host_size_t provider_ordinal = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
    status = loom_link_module_index_add_bytecode(
        out_cli_index->index, input->contents->const_buffer, input->filename,
        &read_options, &options, &provider_ordinal);
    if (iree_status_is_ok(status)) {
      status = loom_link_cli_set_provider_state(out_cli_index, provider_ordinal,
                                                i, allocator);
    }
  }

  iree_string_builder_deinitialize(&provider_name_builder);
  if (!iree_status_is_ok(status)) {
    loom_link_cli_index_deinitialize(out_cli_index, allocator);
  }
  return status;
}

static iree_status_t loom_link_cli_ensure_materialized_module_slots(
    loom_link_cli_input_t* input, iree_host_size_t module_count,
    iree_allocator_t allocator) {
  if (input->materialized_module_count > 0) {
    if (input->materialized_module_count != module_count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "materialized module slot count changed from %" PRIhsz " to %" PRIhsz,
          input->materialized_module_count, module_count);
    }
    return iree_ok_status();
  }
  if (module_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      allocator, module_count, sizeof(*input->materialized_modules),
      (void**)&input->materialized_modules));
  memset(input->materialized_modules, 0,
         module_count * sizeof(*input->materialized_modules));
  input->materialized_module_count = module_count;
  return iree_ok_status();
}

static iree_status_t loom_link_cli_plan_materialize_module(
    void* user_data, const loom_link_module_index_t* index,
    const loom_link_module_index_module_t* indexed_module,
    const loom_module_t** out_module) {
  *out_module = NULL;
  loom_link_cli_plan_materializer_t* materializer =
      (loom_link_cli_plan_materializer_t*)user_data;
  const loom_link_cli_index_t* cli_index = materializer->cli_index;
  if (index != cli_index->index) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "planner materializer index mismatch");
  }
  if (indexed_module->provider_ordinal >= cli_index->provider_state_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "provider state for module '%.*s' is missing",
                            (int)indexed_module->name.size,
                            indexed_module->name.data);
  }

  const loom_link_cli_provider_state_t* provider_state =
      &cli_index->provider_states[indexed_module->provider_ordinal];
  if (provider_state->input_ordinal >= materializer->input_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "provider state for module '%.*s' is invalid",
                            (int)indexed_module->name.size,
                            indexed_module->name.data);
  }
  loom_link_cli_input_t* input =
      &materializer->inputs[provider_state->input_ordinal];
  if (input->format != LOOM_MODULE_FORMAT_BYTECODE) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "module '%.*s' is not a bytecode-backed planning input",
        (int)indexed_module->name.size, indexed_module->name.data);
  }

  const loom_link_module_index_provider_t* provider =
      loom_link_module_index_provider_at(index,
                                         indexed_module->provider_ordinal);
  if (!provider) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT, "provider for module '%.*s' is missing",
        (int)indexed_module->name.size, indexed_module->name.data);
  }
  IREE_RETURN_IF_ERROR(loom_link_cli_ensure_materialized_module_slots(
      input, provider->module_count, materializer->allocator));
  if (indexed_module->provider_module_ordinal >=
      input->materialized_module_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "module ordinal %" PRIhsz
        " for '%.*s' is outside provider count %" PRIhsz,
        indexed_module->provider_module_ordinal, (int)indexed_module->name.size,
        indexed_module->name.data, input->materialized_module_count);
  }

  loom_module_t** materialized_module =
      &input->materialized_modules[indexed_module->provider_module_ordinal];
  if (!*materialized_module) {
    IREE_RETURN_IF_ERROR(loom_link_cli_materialize_bytecode_module(
        input, indexed_module->provider_module_ordinal, materializer->context,
        materializer->block_pool, materializer->config_set,
        materializer->allocator, materialized_module));
  }
  *out_module = *materialized_module;
  return iree_ok_status();
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
  if (iree_any_bit_set(flags, LOOM_LINK_SYMBOL_FLAG_HAS_BODY)) {
    IREE_RETURN_IF_ERROR(
        loom_link_cli_append_flag(builder, &needs_separator, IREE_SV("body")));
  }
  if (iree_any_bit_set(flags, LOOM_LINK_SYMBOL_FLAG_CONFIG)) {
    IREE_RETURN_IF_ERROR(loom_link_cli_append_flag(builder, &needs_separator,
                                                   IREE_SV("config")));
  }
  if (iree_any_bit_set(flags, LOOM_LINK_SYMBOL_FLAG_CHECK_CASE)) {
    IREE_RETURN_IF_ERROR(loom_link_cli_append_flag(builder, &needs_separator,
                                                   IREE_SV("check.case")));
  }
  if (iree_any_bit_set(flags, LOOM_LINK_SYMBOL_FLAG_CHECK_BENCHMARK)) {
    IREE_RETURN_IF_ERROR(loom_link_cli_append_flag(builder, &needs_separator,
                                                   IREE_SV("check.benchmark")));
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

static void loom_link_cli_module_roots_deinitialize(
    loom_link_cli_module_roots_t* roots, iree_host_size_t module_count,
    iree_allocator_t allocator) {
  if (!roots) {
    return;
  }
  for (iree_host_size_t i = 0; i < module_count; ++i) {
    iree_allocator_free(allocator, roots[i].values);
  }
  iree_allocator_free(allocator, roots);
}

static iree_status_t loom_link_cli_module_roots_append(
    loom_link_cli_module_roots_t* roots, iree_string_view_t value,
    iree_allocator_t allocator) {
  for (iree_host_size_t i = 0; i < roots->count; ++i) {
    if (iree_string_view_equal(roots->values[i], value)) {
      return iree_ok_status();
    }
  }
  IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
      allocator, roots->count + 1, sizeof(*roots->values), &roots->capacity,
      (void**)&roots->values));
  roots->values[roots->count++] = value;
  return iree_ok_status();
}

static iree_status_t loom_link_cli_build_module_roots(
    const loom_link_plan_t* plan, loom_link_cli_module_roots_t** out_roots,
    iree_host_size_t* out_module_count, iree_allocator_t allocator) {
  const loom_link_module_index_t* index = loom_link_plan_index(plan);
  const iree_host_size_t module_count =
      loom_link_module_index_module_count(index);
  loom_link_cli_module_roots_t* roots = NULL;
  if (module_count > 0) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        allocator, module_count, sizeof(*roots), (void**)&roots));
    memset(roots, 0, module_count * sizeof(*roots));
  }

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < loom_link_plan_symbol_count(plan) && iree_status_is_ok(status);
       ++i) {
    const loom_link_plan_symbol_t* planned_symbol =
        loom_link_plan_symbol_at(plan, i);
    const loom_link_module_index_symbol_t* symbol =
        loom_link_module_index_symbol_at(index, planned_symbol->symbol_ordinal);
    const loom_link_module_index_module_t* module =
        loom_link_module_index_symbol_module(index, symbol);
    if (!module || module->ordinal >= module_count) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "planned symbol record is stale");
      break;
    }
    status = loom_link_cli_module_roots_append(&roots[module->ordinal],
                                               symbol->name, allocator);
  }

  if (!iree_status_is_ok(status)) {
    loom_link_cli_module_roots_deinitialize(roots, module_count, allocator);
    return status;
  }
  *out_roots = roots;
  *out_module_count = module_count;
  return iree_ok_status();
}

static iree_status_t loom_link_cli_link_index_module(
    const loom_link_cli_index_t* cli_index, loom_link_cli_input_t* inputs,
    iree_host_size_t input_count, const loom_tooling_config_set_t* config_set,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    const loom_link_module_index_module_t* indexed_module,
    const loom_link_cli_module_roots_t* module_roots, bool link_full_module,
    loom_linker_t* linker, iree_allocator_t allocator) {
  if (indexed_module->provider_ordinal >= cli_index->provider_state_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "provider state for module '%.*s' is missing",
                            (int)indexed_module->name.size,
                            indexed_module->name.data);
  }
  const loom_link_cli_provider_state_t* provider_state =
      &cli_index->provider_states[indexed_module->provider_ordinal];
  if (provider_state->input_ordinal >= input_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "provider state for module '%.*s' is invalid",
                            (int)indexed_module->name.size,
                            indexed_module->name.data);
  }
  loom_link_cli_input_t* input = &inputs[provider_state->input_ordinal];

  const loom_module_t* source_module = indexed_module->materialized_module;
  loom_module_t* transient_module = NULL;
  loom_module_t** retained_module_slot = NULL;
  if (!source_module && indexed_module->provider_module_ordinal <
                            input->materialized_module_count) {
    retained_module_slot =
        &input->materialized_modules[indexed_module->provider_module_ordinal];
    source_module = *retained_module_slot;
  }
  if (!source_module) {
    IREE_RETURN_IF_ERROR(loom_link_cli_materialize_bytecode_module(
        input, indexed_module->provider_module_ordinal, context, block_pool,
        config_set, allocator, &transient_module));
    source_module = transient_module;
  }

  const loom_linker_add_options_t add_options = {
      .root_symbols =
          {
              .count = link_full_module ? 0 : module_roots->count,
              .values = link_full_module ? NULL : module_roots->values,
          },
  };
  iree_status_t status =
      loom_linker_add_module(linker, source_module, &add_options);
  loom_module_free(transient_module);
  if (retained_module_slot) {
    loom_module_free(*retained_module_slot);
    *retained_module_slot = NULL;
  }
  return status;
}

static iree_status_t loom_link_cli_link_plan(
    const loom_link_cli_index_t* cli_index, const loom_link_plan_t* plan,
    loom_link_plan_mode_t mode, bool strip_check, loom_link_cli_input_t* inputs,
    iree_host_size_t input_count, const loom_tooling_config_set_t* config_set,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    iree_allocator_t allocator, loom_module_t** out_module) {
  *out_module = NULL;

  loom_linker_t* linker = NULL;
  IREE_RETURN_IF_ERROR(loom_linker_create(
      context, &(loom_linker_options_t){.module_name = IREE_SV("linked")},
      block_pool, allocator, &linker));

  loom_link_cli_module_roots_t* module_roots = NULL;
  iree_host_size_t module_root_count = 0;
  iree_status_t status = loom_link_cli_build_module_roots(
      plan, &module_roots, &module_root_count, allocator);

  const bool archive_full_modules =
      mode == LOOM_LINK_PLAN_ARCHIVE && !strip_check;
  if (iree_status_is_ok(status) && archive_full_modules) {
    for (iree_host_size_t i = 0;
         i < module_root_count && iree_status_is_ok(status); ++i) {
      const loom_link_module_index_module_t* indexed_module =
          loom_link_module_index_module_at(cli_index->index, i);
      status = loom_link_cli_link_index_module(
          cli_index, inputs, input_count, config_set, context, block_pool,
          indexed_module, &module_roots[i], /*link_full_module=*/true, linker,
          allocator);
    }
  }
  if (iree_status_is_ok(status) && !archive_full_modules) {
    for (iree_host_size_t i = 0;
         i < loom_link_plan_symbol_count(plan) && iree_status_is_ok(status);
         ++i) {
      const loom_link_plan_symbol_t* planned_symbol =
          loom_link_plan_symbol_at(plan, i);
      const loom_link_module_index_symbol_t* symbol =
          loom_link_module_index_symbol_at(cli_index->index,
                                           planned_symbol->symbol_ordinal);
      const loom_link_module_index_module_t* indexed_module =
          loom_link_module_index_symbol_module(cli_index->index, symbol);
      loom_link_cli_module_roots_t* roots =
          &module_roots[indexed_module->ordinal];
      if (roots->linked) {
        continue;
      }
      roots->linked = true;
      status = loom_link_cli_link_index_module(
          cli_index, inputs, input_count, config_set, context, block_pool,
          indexed_module, roots, /*link_full_module=*/false, linker, allocator);
    }
  }

  if (iree_status_is_ok(status)) {
    status = loom_linker_finish(linker, out_module);
  }
  loom_link_cli_module_roots_deinitialize(module_roots, module_root_count,
                                          allocator);
  loom_linker_free(linker);
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
      "kept for an archive or selective runtime artifact.\n"
      "\n"
      "### Common flows\n"
      "\n"
      "```shell\n"
      "loom-link root.loom providers.loom --output=linked.loom\n"
      "loom-link root.loom --library=providers.loombc --root=@entry \\\n"
      "  --to=bc --output=entry.loombc\n"
      "loom-link root.loom --library=providers.loom --root=@entry "
      "--print-plan\n"
      "loom-link library.loom --mode=archive --strip-check --to=bc \\\n"
      "  --output=runtime-library.loombc\n"
      "loom-link root.loom --print-config-schema\n"
      "```\n"
      "\n"
      "### Inputs and libraries\n"
      "\n"
      "Positional inputs are primary modules. `--library=path` adds provider\n"
      "modules searched after primary inputs. `--from=auto|text|bc` controls\n"
      "input decoding and `--to=text|bc` controls output encoding.\n"
      "\n"
      "### Archive and selective linking\n"
      "\n"
      "`--mode=archive` preserves every non-stripped symbol in input order.\n"
      "`--mode=link` or `--mode=selective` keeps explicit `--root=@symbol`\n"
      "values, optional `--include-exported-roots`, and reachable "
      "dependencies.\n"
      "`--strip-check` removes `check.case` and `check.benchmark` records "
      "from\n"
      "runtime artifacts.\n"
      "\n"
      "### Provider debugging\n"
      "\n"
      "```shell\n"
      "loom-link root.loom --library=providers.loom --list-symbols\n"
      "loom-link root.loom --library=providers.loom --root=@entry "
      "--print-plan\n"
      "loom-link root.loom --library=providers.loom --root=@entry \\\n"
      "  --config=model.hidden_size=4096 --require-resolved-config\n"
      "```\n"
      "\n"
      "`--list-symbols` shows the indexed providers. `--print-plan` shows why\n"
      "each symbol is live before streaming modules into the linker. Config\n"
      "bindings are applied to each materialized input before dependency "
      "walking,\n"
      "so target and shape predicates can prune unreachable provider "
      "templates.\n");
}

int main(int argc, char** argv) {
  iree_flags_set_usage(
      "loom-link",
      "Links Loom text and bytecode modules into one module.\n"
      "\n"
      "Usage:\n"
      "  loom-link [--mode=archive|link] [--from=auto|text|bc] "
      "[--to=text|bc] [--output=file] [file...]\n"
      "  loom-link model.loom --library=kernels.loombc --root=@entry "
      "--to=bc --output=model.loombc\n"
      "  loom-link --agents_md\n"
      "\n"
      "Input defaults to stdin only when no primary inputs or libraries are "
      "provided. Positional inputs are searched before --library inputs.\n"
      "Archive mode keeps every non-stripped symbol in stable input order. "
      "Link "
      "mode keeps explicit roots or exported roots and their reachable "
      "dependencies.\n"
      "Use --strip-check to remove check.case and check.benchmark symbols from "
      "runtime artifacts.\n");
  for (int i = 1; i < argc; ++i) {
    if (loom_tooling_cli_is_agents_markdown_arg(argv[i])) {
      loom_link_cli_print_agents_markdown(stdout);
      return 0;
    }
  }
  loom_tooling_cli_set_default_help_filter();
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_allocator_t allocator = iree_allocator_system();
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(32 * 1024, allocator, &block_pool);

  loom_context_t context = {0};
  bool context_initialized = false;
  loom_tooling_config_set_t config_set;
  loom_tooling_config_set_initialize(allocator, &config_set);
  loom_link_cli_input_t* inputs = NULL;
  iree_host_size_t input_count = 0;
  loom_link_cli_index_t cli_index = {0};
  loom_link_plan_t* plan = NULL;
  loom_module_t* linked_module = NULL;
  loom_source_entry_t* source_entries = NULL;
  iree_host_size_t source_count = 0;

  loom_module_format_t input_format = LOOM_MODULE_FORMAT_AUTO;
  loom_module_format_t output_format = LOOM_MODULE_FORMAT_TEXT;
  loom_link_cli_mode_t cli_mode = LOOM_LINK_CLI_MODE_AUTO;
  loom_link_plan_mode_t plan_mode = LOOM_LINK_PLAN_ARCHIVE;
  const iree_flag_string_list_t roots = FLAG_root_list();

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
        cli_mode, roots, FLAG_include_exported_roots, &plan_mode);
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
                                       &block_pool, &config_set, allocator,
                                       &inputs, &input_count);
  }
  if (iree_status_is_ok(status)) {
    status = loom_link_cli_build_index(inputs, input_count, &context,
                                       &block_pool, allocator, &cli_index);
  }
  if (iree_status_is_ok(status) && FLAG_list_symbols) {
    status = loom_link_cli_print_symbol_list(
        cli_index.index, iree_make_cstring_view(FLAG_output), allocator);
  }

  loom_link_cli_plan_materializer_t plan_materializer = {
      .cli_index = &cli_index,
      .inputs = inputs,
      .input_count = input_count,
      .config_set = &config_set,
      .context = &context,
      .block_pool = &block_pool,
      .allocator = allocator,
  };
  loom_link_plan_options_t plan_options = {
      .mode = plan_mode,
      .root_symbols = roots,
      .include_exported_roots = FLAG_include_exported_roots,
      .check_policy = FLAG_strip_check ? LOOM_LINK_PLAN_CHECK_STRIP
                                       : LOOM_LINK_PLAN_CHECK_KEEP,
      .materialize_module = loom_link_cli_plan_materialize_module,
      .materialize_module_user_data = &plan_materializer,
  };
  if (iree_status_is_ok(status) && !FLAG_list_symbols) {
    status = loom_link_plan_build(cli_index.index, &plan_options, &block_pool,
                                  allocator, &plan);
  }
  if (iree_status_is_ok(status) && FLAG_print_plan) {
    status = loom_link_cli_print_plan(
        plan, plan_mode, iree_make_cstring_view(FLAG_output), allocator);
  }
  if (iree_status_is_ok(status) && !FLAG_list_symbols && !FLAG_print_plan) {
    status = loom_link_cli_link_plan(
        &cli_index, plan, plan_mode, FLAG_strip_check, inputs, input_count,
        &config_set, &context, &block_pool, allocator, &linked_module);
  }
  if (iree_status_is_ok(status) && linked_module) {
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
  }

  iree_allocator_free(allocator, source_entries);
  loom_module_free(linked_module);
  loom_link_plan_free(plan);
  loom_link_cli_index_deinitialize(&cli_index, allocator);
  loom_link_cli_inputs_deinitialize(inputs, input_count, allocator);
  loom_tooling_config_set_deinitialize(&config_set);
  if (context_initialized) {
    loom_context_deinitialize(&context);
  }
  iree_arena_block_pool_deinitialize(&block_pool);
  return exit_code;
}
