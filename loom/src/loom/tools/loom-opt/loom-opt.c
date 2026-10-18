// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// loom-opt: executes Loom pass pipelines and prints the transformed module.

#include <inttypes.h>
#include <stdio.h>

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "loom/config/text_binding.h"
#include "loom/pass/registry.h"
#include "loom/pass/report.h"
#include "loom/tooling/cli/help.h"
#include "loom/tooling/cli/loomc_diagnostic_json.h"
#include "loom/tooling/cli/loomc_options.h"
#include "loom/tooling/cli/loomc_result.h"
#include "loom/tooling/config/input.h"
#include "loom/tooling/io/file.h"
#include "loom/tooling/pass/loomc_trace.h"
#include "loom/tooling/pass/trace_cli.h"
#include "loom/util/json.h"
#include "loom/util/stream.h"
#include "loomc/interop.h"
#include "loomc/iree.h"
#include "loomc/module_config.h"
#include "loomc/target/configured_ir.h"

IREE_FLAG(string, to, "text", "Output format: text, bc, or bytecode.");
IREE_FLAG(string, output, "-",
          "Output path. Use '-' or the empty string for stdout.");
IREE_FLAG(string, pipeline, "",
          "Pass pipeline to execute. Use 'source-low', 'prepared-low', or "
          "'default' for shared compile pipeline stages, '@symbol' for a "
          "module-local pass.pipeline, or empty to run no pipeline.");
IREE_FLAG_LIST(string, pass,
               "Pass pipeline entry to append. Repeat for multiple passes.");
IREE_FLAG_LIST(
    string, config,
    "Compile/link-time config binding. Repeat as --config=key=value. Bindings "
    "not referenced by the loaded module are ignored.");
IREE_FLAG_LIST_NAMED(
    string, config_file, "config-file",
    "JSON/JSONC config object file. Repeat for multiple files. Nested object "
    "keys are flattened with '.' separators.");
IREE_FLAG_NAMED(
    bool, require_resolved_config, "require-resolved-config", false,
    "Require all config.decl symbols to be materialized before output.");
IREE_FLAG_NAMED(bool, print_config_schema, "print-config-schema", false,
                "Print config schema JSON instead of Loom IR.");
IREE_FLAG(bool, verify, true,
          "Run explicit checks before and after passes. Pass execution always "
          "requires verified input.");
IREE_FLAG_NAMED(bool, list_passes, "list-passes", false,
                "Print registered passes and exit.");
IREE_FLAG_NAMED(string, pass_help, "pass-help", "",
                "Print detailed help for one pass and exit.");
IREE_FLAG_NAMED(string, pass_report, "pass-report", "",
                "Pass execution report format. Use 'json' or empty/'none'.");
IREE_FLAG_NAMED(string, pass_reproducer, "pass-reproducer", "",
                "Path to a one-file pass failure reproducer to write on "
                "failure.");
IREE_FLAG_NAMED(string, diagnostic_format, "diagnostic-format", "text",
                "Diagnostic output format. Use 'text' or 'json'.");

typedef enum loom_opt_pass_report_mode_e {
  LOOM_OPT_PASS_REPORT_NONE = 0,
  LOOM_OPT_PASS_REPORT_JSON = 1,
} loom_opt_pass_report_mode_t;

typedef enum loom_opt_diagnostic_format_e {
  LOOM_OPT_DIAGNOSTIC_FORMAT_TEXT = 0,
  LOOM_OPT_DIAGNOSTIC_FORMAT_JSON = 1,
} loom_opt_diagnostic_format_t;

typedef enum loom_opt_pipeline_kind_e {
  LOOM_OPT_PIPELINE_NONE = 0,
  LOOM_OPT_PIPELINE_MODULE_SYMBOL = 1,
  LOOM_OPT_PIPELINE_COMMAND_LINE = 2,
  LOOM_OPT_PIPELINE_SOURCE_LOW = 3,
  LOOM_OPT_PIPELINE_PREPARED_LOW = 4,
} loom_opt_pipeline_kind_t;

enum {
  LOOM_OPT_BLOCK_POOL_BLOCK_SIZE = 128 * 1024,
};

static iree_status_t loom_opt_present_loomc_result(
    loom_opt_diagnostic_format_t format, const loomc_result_t* result,
    bool* out_succeeded) {
  if (format == LOOM_OPT_DIAGNOSTIC_FORMAT_TEXT) {
    return loom_tooling_cli_print_loomc_result(stderr, result, out_succeeded);
  }
  if (result == NULL) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "compiler operation returned no result");
  }
  loom_output_stream_t stream;
  loom_output_stream_for_file(stderr, &stream);
  for (loomc_host_size_t i = 0; i < loomc_result_diagnostic_count(result);
       ++i) {
    const loomc_diagnostic_t* diagnostic =
        loomc_result_diagnostic_at(result, i);
    if (diagnostic != NULL) {
      IREE_RETURN_IF_ERROR(
          loom_tooling_cli_write_loomc_diagnostic_json(&stream, diagnostic));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(&stream, '\n'));
    }
  }
  *out_succeeded = loomc_result_succeeded(result);
  return iree_ok_status();
}

static const char* loom_opt_pass_kind_name(loom_pass_kind_t kind) {
  switch (kind) {
    case LOOM_PASS_MODULE:
      return "module";
    case LOOM_PASS_FUNCTION:
      return "func";
    default:
      return "unknown";
  }
}

static iree_status_t loom_opt_parse_pass_report_mode(
    iree_string_view_t value, loom_opt_pass_report_mode_t* out_mode) {
  value = iree_string_view_trim(value);
  if (iree_string_view_is_empty(value) ||
      iree_string_view_equal(value, IREE_SV("none"))) {
    *out_mode = LOOM_OPT_PASS_REPORT_NONE;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("json"))) {
    *out_mode = LOOM_OPT_PASS_REPORT_JSON;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unsupported --pass-report mode '%.*s'",
                          (int)value.size, value.data);
}

static iree_status_t loom_opt_parse_diagnostic_format(
    iree_string_view_t value, loom_opt_diagnostic_format_t* out_format) {
  value = iree_string_view_trim(value);
  if (iree_string_view_is_empty(value) ||
      iree_string_view_equal(value, IREE_SV("text"))) {
    *out_format = LOOM_OPT_DIAGNOSTIC_FORMAT_TEXT;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("json"))) {
    *out_format = LOOM_OPT_DIAGNOSTIC_FORMAT_JSON;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unsupported --diagnostic-format mode '%.*s'",
                          (int)value.size, value.data);
}

static iree_status_t loom_opt_parse_pipeline_kind(
    iree_string_view_t pipeline, bool has_pass_list,
    loom_opt_pipeline_kind_t* out_kind) {
  pipeline = iree_string_view_trim(pipeline);
  const bool has_pipeline = !iree_string_view_is_empty(pipeline);
  if (has_pipeline && has_pass_list) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--pipeline and --pass cannot be combined");
  }
  if (!has_pipeline && !has_pass_list) {
    *out_kind = LOOM_OPT_PIPELINE_NONE;
    return iree_ok_status();
  }
  if (has_pass_list) {
    *out_kind = LOOM_OPT_PIPELINE_COMMAND_LINE;
    return iree_ok_status();
  }
  if (iree_string_view_starts_with_char(pipeline, '@')) {
    *out_kind = LOOM_OPT_PIPELINE_MODULE_SYMBOL;
    return iree_ok_status();
  }
  if (iree_string_view_equal(pipeline, IREE_SV("source-low"))) {
    *out_kind = LOOM_OPT_PIPELINE_SOURCE_LOW;
    return iree_ok_status();
  }
  if (iree_string_view_equal(pipeline, IREE_SV("prepared-low")) ||
      iree_string_view_equal(pipeline, IREE_SV("default"))) {
    *out_kind = LOOM_OPT_PIPELINE_PREPARED_LOW;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "unsupported --pipeline '%.*s'; use '@symbol', 'source-low', "
      "'prepared-low', or 'default'",
      (int)pipeline.size, pipeline.data);
}

static iree_status_t loom_opt_verify_module(
    loomc_module_t* module, loom_opt_diagnostic_format_t diagnostic_format,
    iree_allocator_t allocator, bool* out_verified) {
  const loomc_module_verify_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_MODULE_VERIFY_OPTIONS,
      .structure_size = sizeof(options),
      .max_errors = 100,
  };
  loomc_result_t* result = NULL;
  iree_status_t status = iree_status_from_loomc(loomc_module_verify(
      module, &options, loomc_allocator_from_iree(allocator), &result));
  if (iree_status_is_ok(status)) {
    status =
        loom_opt_present_loomc_result(diagnostic_format, result, out_verified);
  }
  loomc_result_release(result);
  return status;
}

static iree_status_t loom_opt_join_pass_list(iree_flag_string_list_t passes,
                                             iree_string_builder_t* builder) {
  for (iree_host_size_t i = 0; i < passes.count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_string(builder, IREE_SV(",")));
    }
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_string(builder, passes.values[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_opt_append_shell_quoted(
    iree_string_builder_t* builder, iree_string_view_t value) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "'"));
  iree_host_size_t segment_start = 0;
  for (iree_host_size_t i = 0; i < value.size; ++i) {
    if (value.data[i] != '\'') {
      continue;
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
        builder,
        iree_make_string_view(value.data + segment_start, i - segment_start)));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(builder, "'\"'\"'"));
    segment_start = i + 1;
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
      builder, iree_make_string_view(value.data + segment_start,
                                     value.size - segment_start)));
  return iree_string_builder_append_cstring(builder, "'");
}

static iree_status_t loom_opt_append_reproducer_run_line(
    iree_string_builder_t* builder) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "// RUN: loom-opt --verify=%s", FLAG_verify ? "true" : "false"));
  iree_string_view_t pass_report =
      iree_string_view_trim(iree_make_cstring_view(FLAG_pass_report));
  if (!iree_string_view_is_empty(pass_report)) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(builder, " --pass-report="));
    IREE_RETURN_IF_ERROR(loom_opt_append_shell_quoted(builder, pass_report));
  }

  iree_string_view_t pipeline_symbol =
      iree_string_view_trim(iree_make_cstring_view(FLAG_pipeline));
  if (!iree_string_view_is_empty(pipeline_symbol)) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(builder, " --pipeline="));
    IREE_RETURN_IF_ERROR(
        loom_opt_append_shell_quoted(builder, pipeline_symbol));
  } else {
    iree_flag_string_list_t passes = FLAG_pass_list();
    for (iree_host_size_t i = 0; i < passes.count; ++i) {
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(builder, " --pass="));
      IREE_RETURN_IF_ERROR(
          loom_opt_append_shell_quoted(builder, passes.values[i]));
    }
  }
  return iree_string_builder_append_cstring(builder, " %s\n");
}

static iree_status_t loom_opt_format_pass_selection_json(
    loom_output_stream_t* stream) {
  iree_string_view_t pipeline_symbol =
      iree_string_view_trim(iree_make_cstring_view(FLAG_pipeline));
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  if (!iree_string_view_is_empty(pipeline_symbol)) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("kind"), IREE_SV("pipeline")));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("pipeline"), pipeline_symbol));
    return loom_json_object_end(&object);
  }

  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("kind"), IREE_SV("pass-list")));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("passes")));
  loom_json_array_writer_t passes_array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &passes_array));
  iree_flag_string_list_t passes = FLAG_pass_list();
  for (iree_host_size_t i = 0; i < passes.count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_json_array_write_string_element(&passes_array, passes.values[i]));
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&passes_array));
  return loom_json_object_end(&object);
}

static iree_status_t loom_opt_append_commented_block(
    iree_string_builder_t* builder, iree_string_view_t text) {
  bool at_line_start = true;
  iree_host_size_t segment_start = 0;
  for (iree_host_size_t i = 0; i < text.size; ++i) {
    if (at_line_start) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "// "));
      at_line_start = false;
      segment_start = i;
    }
    if (text.data[i] != '\n') {
      continue;
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
        builder, iree_make_string_view(text.data + segment_start,
                                       i - segment_start + 1)));
    at_line_start = true;
    segment_start = i + 1;
  }
  if (segment_start < text.size) {
    if (at_line_start) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "// "));
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
        builder, iree_make_string_view(text.data + segment_start,
                                       text.size - segment_start)));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
  }
  return iree_ok_status();
}

static iree_status_t loom_opt_append_reproducer_metadata(
    iree_string_builder_t* builder, const loom_pass_registry_t* registry,
    iree_status_code_t failure_status_code, iree_allocator_t allocator) {
  iree_string_builder_t metadata_builder;
  iree_string_builder_initialize(allocator, &metadata_builder);
  loom_output_stream_t metadata_stream;
  loom_output_stream_for_builder(&metadata_builder, &metadata_stream);

  loom_json_object_writer_t metadata_object;
  iree_status_t status =
      loom_json_object_begin(&metadata_stream, &metadata_object);
  if (iree_status_is_ok(status)) {
    status =
        loom_json_object_begin_field(&metadata_object, IREE_SV("selection"));
  }
  if (iree_status_is_ok(status)) {
    status = loom_opt_format_pass_selection_json(&metadata_stream);
  }
  if (iree_status_is_ok(status)) {
    status = loom_json_object_begin_field(&metadata_object, IREE_SV("status"));
  }
  if (iree_status_is_ok(status)) {
    status = loom_json_write_status_object(
        &metadata_stream, failure_status_code, iree_string_view_empty());
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_json_object_begin_field(&metadata_object, IREE_SV("registry"));
  }
  if (iree_status_is_ok(status)) {
    status = loom_pass_report_format_registry_json(registry, &metadata_stream);
  }
  if (iree_status_is_ok(status)) {
    status = loom_json_object_end(&metadata_object);
  }
  if (iree_status_is_ok(status)) {
    status = loom_output_stream_write_char(&metadata_stream, '\n');
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(builder,
                                                "// Pass failure metadata:\n");
  }
  if (iree_status_is_ok(status)) {
    status = loom_opt_append_commented_block(
        builder, iree_string_builder_view(&metadata_builder));
  }

  iree_string_builder_deinitialize(&metadata_builder);
  return status;
}

static iree_status_t loom_opt_write_pass_reproducer(
    iree_string_view_t path, iree_string_view_t filename,
    const loomc_source_t* input_snapshot, const loom_pass_registry_t* registry,
    iree_status_code_t failure_status_code, iree_allocator_t allocator) {
  path = iree_string_view_trim(path);
  if (iree_string_view_is_empty(path)) {
    return iree_ok_status();
  }
  const loomc_byte_span_t contents = loomc_source_contents(input_snapshot);
  const iree_string_view_t source =
      iree_make_string_view((const char*)contents.data, contents.data_length);

  iree_string_builder_t builder;
  iree_string_builder_initialize(allocator, &builder);
  iree_status_t status = iree_string_builder_append_cstring(
      &builder,
      "// Reproducer generated by loom-opt after pass pipeline failure.\n");
  if (iree_status_is_ok(status)) {
    status =
        iree_string_builder_append_format(&builder, "// Original input: %.*s\n",
                                          (int)filename.size, filename.data);
  }
  if (iree_status_is_ok(status)) {
    status = loom_opt_append_reproducer_metadata(
        &builder, registry, failure_status_code, allocator);
  }

  if (iree_status_is_ok(status)) {
    status = loom_opt_append_reproducer_run_line(&builder);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(&builder, "\n");
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_string(&builder, source);
  }
  if (iree_status_is_ok(status) &&
      (source.size == 0 || source.data[source.size - 1] != '\n')) {
    status = iree_string_builder_append_cstring(&builder, "\n");
  }
  if (iree_status_is_ok(status)) {
    status = loom_tooling_write_output_file(
        path, iree_string_builder_view(&builder), allocator);
  }
  iree_string_builder_deinitialize(&builder);
  if (!iree_status_is_ok(status)) {
    return status;
  }

  if (fprintf(stderr, "pass pipeline reproducer written to %.*s\n",
              (int)path.size, path.data) < 0) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "failed to write pass reproducer path");
  }
  return fflush(stderr) == 0
             ? iree_ok_status()
             : iree_make_status(IREE_STATUS_DATA_LOSS,
                                "failed to flush pass reproducer path");
}

static iree_status_t loom_opt_prepare_pass_program(
    loomc_context_t* context, loomc_module_t* module,
    loom_opt_pipeline_kind_t pipeline_kind, iree_allocator_t allocator,
    loomc_pass_program_t** out_program, loomc_result_t** out_result) {
  const loomc_allocator_t loom_allocator = loomc_allocator_from_iree(allocator);
  if (pipeline_kind == LOOM_OPT_PIPELINE_MODULE_SYMBOL) {
    return iree_status_from_loomc(loomc_pass_program_create_from_module_symbol(
        module,
        loomc_string_view_from_iree(
            iree_string_view_trim(iree_make_cstring_view(FLAG_pipeline))),
        /*options=*/NULL, loom_allocator, out_program, out_result));
  }
  if (pipeline_kind == LOOM_OPT_PIPELINE_SOURCE_LOW ||
      pipeline_kind == LOOM_OPT_PIPELINE_PREPARED_LOW) {
    const loomc_target_pipeline_options_t options = {
        .type = LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS,
        .structure_size = sizeof(options),
        .identifier = loomc_make_cstring_view(
            pipeline_kind == LOOM_OPT_PIPELINE_SOURCE_LOW
                ? "__loom_opt_source_low"
                : "__loom_opt_prepared_low"),
        .kind = pipeline_kind == LOOM_OPT_PIPELINE_SOURCE_LOW
                    ? LOOMC_TARGET_PIPELINE_KIND_SOURCE_LOW
                    : LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW,
    };
    return iree_status_from_loomc(
        loomc_pass_program_create_from_target_pipeline(
            context, &options, loom_allocator, out_program, out_result));
  }
  iree_string_builder_t builder;
  iree_string_builder_initialize(allocator, &builder);
  iree_status_t status = loom_opt_join_pass_list(FLAG_pass_list(), &builder);
  if (iree_status_is_ok(status)) {
    status =
        iree_status_from_loomc(loomc_pass_program_create_from_pipeline_text(
            context,
            loomc_string_view_from_iree(iree_string_builder_view(&builder)),
            /*options=*/NULL, loom_allocator, out_program, out_result));
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

static iree_status_t loom_opt_append_config_flags(
    loom_config_text_binding_set_t* config_set) {
  iree_flag_string_list_t assignments = FLAG_config_list();
  for (iree_host_size_t i = 0; i < assignments.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_tooling_config_text_binding_set_append_assignment(
        config_set, assignments.values[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_opt_append_config_files(
    loom_config_text_binding_set_t* config_set, iree_allocator_t allocator) {
  iree_flag_string_list_t paths = FLAG_config_file_list();
  for (iree_host_size_t i = 0; i < paths.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_tooling_config_text_binding_set_append_json_file(
        config_set, paths.values[i], allocator));
  }
  return iree_ok_status();
}

static iree_status_t loom_opt_write_pass_report(
    loom_opt_pass_report_mode_t mode, const loom_pass_report_t* report) {
  switch (mode) {
    case LOOM_OPT_PASS_REPORT_NONE:
      return iree_ok_status();
    case LOOM_OPT_PASS_REPORT_JSON: {
      loom_output_stream_t stream;
      loom_output_stream_for_file(stderr, &stream);
      IREE_RETURN_IF_ERROR(loom_pass_report_format_json(report, &stream));
      return fflush(stderr) == 0
                 ? iree_ok_status()
                 : iree_make_status(IREE_STATUS_DATA_LOSS,
                                    "failed to flush pass report");
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported pass report mode");
  }
}

static iree_status_t loom_opt_parse_output_format(
    iree_string_view_t value, loomc_source_format_t* out_format) {
  if (iree_string_view_equal(value, IREE_SV("text"))) {
    *out_format = LOOMC_SOURCE_FORMAT_TEXT;
  } else if (iree_string_view_equal(value, IREE_SV("bc")) ||
             iree_string_view_equal(value, IREE_SV("bytecode"))) {
    *out_format = LOOMC_SOURCE_FORMAT_BYTECODE;
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unsupported --to format '%.*s'; use text or bytecode", (int)value.size,
        value.data);
  }
  if (FLAG_print_config_schema && *out_format != LOOMC_SOURCE_FORMAT_TEXT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--print-config-schema requires --to=text");
  }
  return iree_ok_status();
}

static iree_status_t loom_opt_print_module(iree_string_view_t output_path,
                                           loomc_module_t* module,
                                           loomc_source_format_t format,
                                           iree_allocator_t allocator) {
  const loomc_module_serialize_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS,
      .structure_size = sizeof(options),
      .format = format,
      .text_presentation = LOOMC_MODULE_TEXT_PRESENTATION_GENERIC,
  };
  loomc_source_t* source = NULL;
  IREE_RETURN_IF_ERROR(iree_status_from_loomc(loomc_module_serialize_to_source(
      module, &options, loomc_allocator_from_iree(allocator), &source)));
  loom_tooling_output_stream_t output = {0};
  iree_status_t status =
      loom_tooling_output_stream_open(output_path, allocator, &output);
  if (iree_status_is_ok(status)) {
    const loomc_byte_span_t contents = loomc_source_contents(source);
    status = loom_output_stream_write(
        &output.stream, iree_make_string_view((const char*)contents.data,
                                              contents.data_length));
  }
  if (iree_status_is_ok(status) && format == LOOMC_SOURCE_FORMAT_TEXT) {
    status = loom_output_stream_write_char(&output.stream, '\n');
  }
  status = iree_status_join(status, loom_tooling_output_stream_close(&output));
  loomc_source_release(source);
  return status;
}

static iree_status_t loom_opt_print_config_schema(
    iree_string_view_t output_path, const loomc_artifact_t* artifact,
    iree_allocator_t allocator) {
  loom_tooling_output_stream_t output = {0};
  iree_status_t status =
      loom_tooling_output_stream_open(output_path, allocator, &output);
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(
        loomc_artifact_write_to_file(artifact, output.file));
  }
  if (iree_status_is_ok(status)) {
    status = loom_output_stream_write_char(&output.stream, '\n');
  }
  return iree_status_join(status, loom_tooling_output_stream_close(&output));
}

static iree_status_t loom_opt_print_pass_list(
    const loom_pass_registry_t* registry) {
  for (iree_host_size_t i = 0; i < registry->descriptor_count; ++i) {
    const loom_pass_descriptor_t* descriptor =
        loom_pass_registry_at(registry, i);
    const loom_pass_info_t* info = descriptor->info();
    if (!info) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "pass descriptor '%.*s' returned no info",
                              (int)descriptor->key.size, descriptor->key.data);
    }
    fprintf(stdout, "%.*s\t%s\t%.*s\n", (int)descriptor->key.size,
            descriptor->key.data, loom_opt_pass_kind_name(info->kind),
            (int)info->description.size, info->description.data);
  }
  return fflush(stdout) == 0 ? iree_ok_status()
                             : iree_make_status(IREE_STATUS_DATA_LOSS,
                                                "failed to flush stdout");
}

static void loom_opt_print_pass_option_schema(
    const loom_pass_option_schema_t* schema) {
  fprintf(stdout, "    %.*s: ", (int)schema->name.size, schema->name.data);
  switch (schema->kind) {
    case LOOM_PASS_OPTION_SCHEMA_STRING:
      fprintf(stdout, "string");
      break;
    case LOOM_PASS_OPTION_SCHEMA_UINT32:
      fprintf(stdout, "uint32 [%" PRIu32 "..%" PRIu32 "]",
              schema->minimum_uint32, schema->maximum_uint32);
      break;
    case LOOM_PASS_OPTION_SCHEMA_ENUM:
      fprintf(stdout, "enum");
      if (schema->enum_value_count > 0) {
        fprintf(stdout, " {");
        for (uint16_t i = 0; i < schema->enum_value_count; ++i) {
          if (i > 0) {
            fprintf(stdout, ", ");
          }
          fprintf(stdout, "%.*s", (int)schema->enum_values[i].value.size,
                  schema->enum_values[i].value.data);
        }
        fprintf(stdout, "}");
      }
      break;
    default:
      fprintf(stdout, "unknown");
      break;
  }
  fprintf(stdout, "\n");
}

static iree_status_t loom_opt_print_pass_help(
    const loom_pass_registry_t* registry, iree_string_view_t key) {
  const loom_pass_descriptor_t* descriptor =
      loom_pass_registry_lookup(registry, key);
  if (!descriptor) {
    return iree_make_status(IREE_STATUS_NOT_FOUND, "unknown pass '%.*s'",
                            (int)key.size, key.data);
  }
  const loom_pass_info_t* info = descriptor->info();
  if (!info) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass descriptor '%.*s' returned no info",
                            (int)descriptor->key.size, descriptor->key.data);
  }

  fprintf(stdout, "%.*s\n", (int)descriptor->key.size, descriptor->key.data);
  fprintf(stdout, "  kind: %s\n", loom_opt_pass_kind_name(info->kind));
  fprintf(stdout, "  description: %.*s\n", (int)info->description.size,
          info->description.data);
  if (!loom_pass_descriptor_is_available(descriptor)) {
    fprintf(stdout, "  unavailable: %.*s\n",
            (int)descriptor->unavailable_reason.size,
            descriptor->unavailable_reason.data);
  }
  if (descriptor->option_schema_count > 0) {
    fprintf(stdout, "  options:\n");
    for (uint16_t i = 0; i < descriptor->option_schema_count; ++i) {
      loom_opt_print_pass_option_schema(&descriptor->option_schema[i]);
    }
  }
  return fflush(stdout) == 0 ? iree_ok_status()
                             : iree_make_status(IREE_STATUS_DATA_LOSS,
                                                "failed to flush stdout");
}

static void loom_opt_print_agents_markdown(FILE* stream) {
  fprintf(
      stream,
      "## loom-opt\n"
      "\n"
      "`loom-opt` parses Loom IR, optionally materializes config values,\n"
      "executes pass pipelines, and prints transformed Loom IR. Use it when\n"
      "authoring `.loom` files, reducing a compiler issue, or checking what a\n"
      "shared compile stage does before `loom-compile` emits an artifact.\n"
      "\n"
      "### Common flows\n"
      "\n"
      "```shell\n"
      "loom-opt input.loom --pass=canonicalize --pass=cse --pass=dce\n"
      "loom-opt input.loom --pipeline=source-low --output=source-low.loom\n"
      "loom-opt input.loom --pipeline=prepared-low --output=prepared-low.loom\n"
      "loom-opt input.loom --pipeline=@my_pipeline --config=shape.m=4096\n"
      "loom-opt input.loom --print-config-schema\n"
      "loom-opt --list-passes\n"
      "loom-opt --pass-help=pipeline-scf-for\n"
      "loom-opt --pass-help=unroll-scf-for\n"
      "```\n"
      "\n"
      "### Pass selection\n"
      "\n"
      "`--pass=<name>{key=value}` appends a pass to a shallow command-line\n"
      "pipeline. Repeat it to build a pipeline. `--pipeline=source-low`,\n"
      "`--pipeline=prepared-low`, and `--pipeline=default` run the shared\n"
      "compile stages. `--pipeline=@symbol` runs an authored `pass.pipeline`\n"
      "from the input module.\n"
      "\n"
      "### Config specialization\n"
      "\n"
      "`--config=key=value` and `--config-file=file.jsonc` bind `config.decl`\n"
      "values before passes run. `--require-resolved-config` rejects final\n"
      "output that still contains unresolved config declarations.\n"
      "\n"
      "### Inspect authored loop schedules\n"
      "\n"
      "`scf.for ... pipeline(%%depth) unroll(%%factor)` requests ordinary\n"
      "read-ahead before unrolling; either clause also works alone. Keep the\n"
      "logical loop; reusable motifs take per-instance template arguments or\n"
      "calculate policies from specialized arguments and target properties.\n"
      "The default compiler pipeline consumes both policies. Global config\n"
      "overrides are useful for experiment sweeps, but couple all instances\n"
      "when used inside a motif library. This config-driven experiment\n"
      "isolates the source transformation:\n"
      "\n"
      "```shell\n"
      "loom-opt read-ahead.loom --config=read_ahead.depth=3 "
      "\\\n"
      "  --config=read_ahead.unroll=2 --pass=pipeline-scf-for "
      "\\\n"
      "  --pass=unroll-scf-for --output=scheduled.loom\n"
      "```\n"
      "\n"
      "`loom-compile --compile-report=details` retains the applied source\n"
      "schedule alongside final target resources. The "
      "[loop-tuning walkthrough](https://rocm.github.io/hrx-system/loom/"
      "workflows/tune-loop-schedules/)\n"
      "includes checked examples and serial controls.\n"
      "\n"
      "### Diagnostics and reports\n"
      "\n"
      "```shell\n"
      "loom-opt input.loom --pipeline=source-low --pass-report=json \\\n"
      "  --diagnostic-format=json --output=/tmp/out.loom\n"
      "loom-opt input.loom --pipeline=source-low --dump-ir-after-all \\\n"
      "  --dump-ir-format=jsonl --dump-ir-output=/tmp/trace.jsonl\n"
      "loom-opt input.loom --pass=unroll-scf-for --pass-report=json "
      "2>report.json\n"
      "jq '.invocations[] | {pass, anchor, changed, details}' report.json\n"
      "jq -r 'select(.pass == \"source-to-low\") | .ir' /tmp/trace.jsonl\n"
      "```\n"
      "\n"
      "`--pass-report=json` is the structured channel for pass decisions and\n"
      "per-pass facts. `--diagnostic-format=json` emits compiler diagnostics "
      "as\n"
      "JSONL. `--dump-ir-before*` and `--dump-ir-after*` capture intermediate "
      "IR\n"
      "for bisection. Each trace event snapshots the whole module, even when "
      "the\n"
      "pass itself is anchored on one function. Prefer "
      "`--dump-ir-format=jsonl`\n"
      "for agent ingestion so `jq -r '.ir'` can extract a replayable "
      "module. Stage filters use `pipeline-text` for --pass and\n"
      "`pipeline-symbol` for --pipeline=@name.\n");
}

int main(int argc, char** argv) {
  iree_flags_set_usage(
      "loom-opt",
      "Executes Loom pass pipelines and prints the transformed module.\n"
      "\n"
      "Usage:\n"
      "  loom-opt [--pipeline=@name] [--output=file] [file]\n"
      "  loom-opt --pipeline=source-low [file]\n"
      "  loom-opt --pass=canonicalize --pass=cse --pass=dce [file]\n"
      "  cat module.loom | loom-opt --pass=symbol-dce\n"
      "  loom-opt --list-passes\n"
      "  loom-opt --pass-help=canonicalize\n"
      "  loom-opt --agents_md\n"
      "\n"
      "Input defaults to stdin when no file is provided. Output defaults to "
      "stdout. Use --to=bytecode for binary output; text is the default.\n"
      "Use --pipeline=source-low or --pipeline=prepared-low/default to run "
      "shared compile pipeline stages, --pipeline=@name to execute a named "
      "pass.pipeline symbol from the input module, or repeat --pass for a "
      "shallow command-line pipeline backed by the C pass registry.\n"
      "Repeat --config=key=value to materialize compile/link-time config "
      "symbols before passes run. Unused config bindings are ignored.\n"
      "Use --config-file=path to load a JSON/JSONC config object. Files and "
      "--config values share one config set and duplicate keys are rejected.\n"
      "Use --require-resolved-config for final outputs that must not retain "
      "config.decl symbols.\n"
      "Use --print-config-schema to print config schema JSON instead of Loom "
      "IR.\n"
      "Use --pass-report=json to print a structured execution report to "
      "stderr.\n"
      "Use --diagnostic-format=json to print structured diagnostic JSONL to "
      "stderr.\n" LOOM_TOOLING_PASS_TRACE_USAGE
      "Use --pass-reproducer=file to capture a rerunnable failure file.\n");
  for (int i = 1; i < argc; ++i) {
    if (loom_tooling_cli_is_agents_markdown_arg(argv[i])) {
      loom_opt_print_agents_markdown(stdout);
      return 0;
    }
  }
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);

  loom_tooling_cli_set_default_help_filter();
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_allocator_t allocator = iree_allocator_system();
  loomc_allocator_t loom_allocator = loomc_allocator_from_iree(allocator);

  iree_io_file_contents_t* contents = NULL;
  loomc_target_environment_t* public_target_environment = NULL;
  loomc_context_t* public_context = NULL;
  loomc_workspace_t* public_workspace = NULL;
  loomc_source_t* public_source = NULL;
  loomc_source_t* reproducer_input = NULL;
  loomc_module_t* public_module = NULL;
  loomc_result_t* public_result = NULL;
  loomc_pass_program_t* pass_program = NULL;
  loom_config_text_binding_set_t config_set;
  loom_config_text_binding_set_initialize(allocator, &config_set);
  loom_pass_registry_storage_t pass_registry_storage = {0};
  const loom_pass_registry_t* pass_registry =
      loom_pass_registry_storage_registry(&pass_registry_storage);
  iree_string_view_t source = iree_string_view_empty();
  loom_opt_pass_report_mode_t pass_report_mode = LOOM_OPT_PASS_REPORT_NONE;
  loom_opt_diagnostic_format_t diagnostic_format =
      LOOM_OPT_DIAGNOSTIC_FORMAT_TEXT;
  loom_opt_pipeline_kind_t pipeline_kind = LOOM_OPT_PIPELINE_NONE;
  loomc_source_format_t output_format = LOOMC_SOURCE_FORMAT_TEXT;
  loom_pass_report_t pass_report = {0};
  bool pass_report_initialized = false;
  loom_tooling_loomc_pass_trace_t pass_trace = {0};
  bool pipeline_attempted = false;
  iree_status_code_t pipeline_status_code = IREE_STATUS_OK;
  bool module_ready = true;

  // The optimizer accepts target Low before artifact selection and therefore
  // needs every configured architecture provider, including portable cmd.core.
  // The configured compiler package contains only artifact-capable providers.
  iree_status_t status =
      iree_status_from_loomc(loomc_target_environment_create_configured_ir(
          loom_allocator, &public_target_environment));
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(
        loomc_target_environment_get_native_pass_registry(
            public_target_environment, &pass_registry_storage));
  }
  if (iree_status_is_ok(status)) {
    status = loom_opt_parse_pass_report_mode(
        iree_make_cstring_view(FLAG_pass_report), &pass_report_mode);
  }
  if (iree_status_is_ok(status)) {
    status = loom_opt_parse_diagnostic_format(
        iree_make_cstring_view(FLAG_diagnostic_format), &diagnostic_format);
  }
  if (iree_status_is_ok(status) && FLAG_list_passes) {
    status = loom_opt_print_pass_list(pass_registry);
  }
  iree_string_view_t pass_help =
      iree_string_view_trim(iree_make_cstring_view(FLAG_pass_help));
  if (iree_status_is_ok(status) && !iree_string_view_is_empty(pass_help)) {
    status = loom_opt_print_pass_help(pass_registry, pass_help);
  }

  bool metadata_only =
      FLAG_list_passes || !iree_string_view_is_empty(pass_help);
  if (iree_status_is_ok(status) && !metadata_only && argc > 2) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "loom-opt accepts at most one input file or '-' for stdin; got %d "
        "inputs",
        argc - 1);
  }

  if (iree_status_is_ok(status) && !metadata_only) {
    status = loom_opt_parse_pipeline_kind(iree_make_cstring_view(FLAG_pipeline),
                                          FLAG_pass_list().count > 0,
                                          &pipeline_kind);
  }
  if (iree_status_is_ok(status) && !metadata_only) {
    status = loom_opt_parse_output_format(iree_make_cstring_view(FLAG_to),
                                          &output_format);
  }

  if (iree_status_is_ok(status) && !metadata_only) {
    const loom_tooling_output_path_t output_paths[] = {
        {
            .active = true,
            .flag_name = IREE_SV("--output"),
            .path = iree_make_cstring_view(FLAG_output),
        },
        loom_tooling_pass_trace_output_path_from_flags(),
    };
    status = loom_tooling_output_paths_validate_exclusive_stdout(
        output_paths, IREE_ARRAYSIZE(output_paths));
  }

  if (iree_status_is_ok(status) && !metadata_only) {
    status = loom_opt_append_config_files(&config_set, allocator);
  }
  if (iree_status_is_ok(status) && !metadata_only) {
    status = loom_opt_append_config_flags(&config_set);
  }
  if (iree_status_is_ok(status) && !metadata_only) {
    const loomc_context_target_options_t target_options = {
        .type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
        .structure_size = sizeof(target_options),
        .target_environment = public_target_environment,
    };
    const loomc_context_options_t context_options = {
        .type = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
        .structure_size = sizeof(context_options),
        .next = &target_options,
    };
    status = iree_status_from_loomc(loomc_context_create(
        &context_options, loom_allocator, &public_context));
  }
  if (iree_status_is_ok(status) && !metadata_only) {
    const loomc_workspace_options_t workspace_options = {
        .type = LOOMC_STRUCTURE_TYPE_WORKSPACE_OPTIONS,
        .structure_size = sizeof(workspace_options),
        .block_size = LOOM_OPT_BLOCK_POOL_BLOCK_SIZE,
    };
    status = iree_status_from_loomc(loomc_workspace_create(
        &workspace_options, loom_allocator, &public_workspace));
  }
  if (iree_status_is_ok(status) && !metadata_only &&
      pass_report_mode != LOOM_OPT_PASS_REPORT_NONE) {
    loom_pass_report_initialize(allocator, &pass_report);
    pass_report_initialized = true;
  }

  iree_string_view_t input_path =
      argc < 2 ? iree_string_view_empty() : iree_make_cstring_view(argv[1]);
  iree_string_view_t filename =
      (argc < 2 ||
       iree_string_view_equal(input_path, iree_make_cstring_view("-")))
          ? iree_make_cstring_view("<stdin>")
          : input_path;

  if (iree_status_is_ok(status) && !metadata_only) {
    status = loom_tooling_read_input_file(input_path, allocator, &contents);
    if (iree_status_is_ok(status)) {
      source = loom_tooling_file_contents_string_view(contents);
    }
  }
  if (iree_status_is_ok(status) && !metadata_only) {
    const loomc_source_options_t source_options = {
        .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
        .structure_size = sizeof(source_options),
        .format = LOOMC_SOURCE_FORMAT_UNKNOWN,
        .identifier = loomc_string_view_from_iree(filename),
        .contents = loomc_byte_span_from_iree(
            iree_make_const_byte_span(source.data, source.size)),
        .storage = LOOMC_SOURCE_STORAGE_BORROWED,
    };
    status = iree_status_from_loomc(
        loomc_source_create(&source_options, loom_allocator, &public_source));
  }
  if (iree_status_is_ok(status) && !metadata_only) {
    status = iree_status_from_loomc(loomc_module_deserialize_from_source(
        public_context, public_workspace, public_source,
        /*options=*/NULL, loom_allocator, &public_module, &public_result));
  }
  if (iree_status_is_ok(status) && !metadata_only) {
    bool input_succeeded = false;
    status = loom_opt_present_loomc_result(diagnostic_format, public_result,
                                           &input_succeeded);
    module_ready = input_succeeded;
  }
  loomc_result_release(public_result);
  public_result = NULL;
  loomc_source_release(public_source);
  public_source = NULL;
  if (iree_status_is_ok(status) && !metadata_only && module_ready &&
      FLAG_verify) {
    status = loom_opt_verify_module(public_module, diagnostic_format, allocator,
                                    &module_ready);
  }
  if (iree_status_is_ok(status) && !metadata_only && module_ready &&
      config_set.binding_count > 0) {
    loomc_config_binding_t* bindings = NULL;
    loomc_module_config_options_t options = {
        .type = LOOMC_STRUCTURE_TYPE_MODULE_CONFIG_OPTIONS,
        .structure_size = sizeof(options),
    };
    status = loom_tooling_cli_make_loomc_config_options(
        &config_set, allocator, &bindings, &options.config);
    // Passes may remove unused declarations before the final resolution check.
    options.config.flags = 0;
    if (iree_status_is_ok(status)) {
      status = iree_status_from_loomc(
          loomc_module_apply_config(public_module, public_workspace, &options,
                                    loom_allocator, &public_result));
    }
    if (iree_status_is_ok(status)) {
      status = loom_opt_present_loomc_result(diagnostic_format, public_result,
                                             &module_ready);
    }
    loomc_result_release(public_result);
    public_result = NULL;
    iree_allocator_free(allocator, bindings);
  }
  if (iree_status_is_ok(status) && !metadata_only && module_ready &&
      !iree_string_view_is_empty(iree_string_view_trim(
          iree_make_cstring_view(FLAG_pass_reproducer)))) {
    // Capture the exact pass input before execution can mutate it. Applied
    // configuration is already in the IR, and bytecode becomes rerunnable text.
    status = iree_status_from_loomc(loomc_module_serialize_text_to_source(
        public_module, /*options=*/NULL, loom_allocator, &reproducer_input));
  }
  if (iree_status_is_ok(status) && !metadata_only && module_ready) {
    status = loom_tooling_loomc_pass_trace_open_from_flags(
        &(loom_tooling_pass_trace_open_options_t){
            .tool_name = IREE_SV("loom-opt"),
            .input_path = filename,
        },
        allocator, &pass_trace);
  }
  if (iree_status_is_ok(status) && !metadata_only && module_ready &&
      pipeline_kind != LOOM_OPT_PIPELINE_NONE) {
    pipeline_attempted = true;
    status = loom_opt_prepare_pass_program(public_context, public_module,
                                           pipeline_kind, allocator,
                                           &pass_program, &public_result);
    pipeline_status_code = iree_status_code(status);
    if (iree_status_is_ok(status)) {
      pipeline_status_code =
          (iree_status_code_t)loomc_result_status_code(public_result);
      status = loom_opt_present_loomc_result(diagnostic_format, public_result,
                                             &module_ready);
    }
    loomc_result_release(public_result);
    public_result = NULL;
  }
  if (iree_status_is_ok(status) && module_ready && pass_program) {
    const loomc_native_pass_report_options_t report_options = {
        .type = LOOMC_STRUCTURE_TYPE_NATIVE_PASS_REPORT_OPTIONS,
        .structure_size = sizeof(report_options),
        .next = pass_trace.output.enabled ? &pass_trace.options : NULL,
        .report = &pass_report,
    };
    const loomc_pass_run_options_t run_options = {
        .type = LOOMC_STRUCTURE_TYPE_PASS_RUN_OPTIONS,
        .structure_size = sizeof(run_options),
        .next = pass_report_initialized     ? (const void*)&report_options
                : pass_trace.output.enabled ? (const void*)&pass_trace.options
                                            : NULL,
    };
    status = iree_status_from_loomc(
        loomc_pass_program_run(pass_program, public_workspace, public_module,
                               &run_options, loom_allocator, &public_result));
    pipeline_status_code = iree_status_code(status);
    if (iree_status_is_ok(status)) {
      pipeline_status_code =
          (iree_status_code_t)loomc_result_status_code(public_result);
      status = loom_opt_present_loomc_result(diagnostic_format, public_result,
                                             &module_ready);
    }
    loomc_result_release(public_result);
    public_result = NULL;
  }
  if (pipeline_attempted && pipeline_status_code != IREE_STATUS_OK) {
    status = iree_status_join(
        status,
        loom_opt_write_pass_reproducer(
            iree_make_cstring_view(FLAG_pass_reproducer), filename,
            reproducer_input, pass_registry, pipeline_status_code, allocator));
  }
  if (pass_report_initialized && pipeline_attempted) {
    status = iree_status_join(
        status, loom_opt_write_pass_report(pass_report_mode, &pass_report));
  }
  if (iree_status_is_ok(status) && !metadata_only && module_ready &&
      FLAG_verify) {
    status = loom_opt_verify_module(public_module, diagnostic_format, allocator,
                                    &module_ready);
    if (iree_status_is_ok(status) && !module_ready && pipeline_attempted) {
      status = iree_status_join(
          status, loom_opt_write_pass_reproducer(
                      iree_make_cstring_view(FLAG_pass_reproducer), filename,
                      reproducer_input, pass_registry,
                      IREE_STATUS_FAILED_PRECONDITION, allocator));
    }
  }
  if (iree_status_is_ok(status) && !metadata_only && module_ready &&
      FLAG_require_resolved_config) {
    const loomc_module_config_options_t options = {
        .type = LOOMC_STRUCTURE_TYPE_MODULE_CONFIG_OPTIONS,
        .structure_size = sizeof(options),
        .config = {.flags = LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED},
    };
    status = iree_status_from_loomc(
        loomc_module_apply_config(public_module, public_workspace, &options,
                                  loom_allocator, &public_result));
    if (iree_status_is_ok(status)) {
      status = loom_opt_present_loomc_result(diagnostic_format, public_result,
                                             &module_ready);
    }
    loomc_result_release(public_result);
    public_result = NULL;
  }
  if (iree_status_is_ok(status) && !metadata_only && module_ready) {
    if (FLAG_print_config_schema) {
      status = iree_status_from_loomc(loomc_module_query_config_schema(
          public_module, loom_allocator, &public_result));
      if (iree_status_is_ok(status)) {
        status = loom_opt_present_loomc_result(diagnostic_format, public_result,
                                               &module_ready);
      }
      if (iree_status_is_ok(status) && module_ready) {
        status = loom_opt_print_config_schema(
            iree_make_cstring_view(FLAG_output),
            loomc_result_artifact_at(public_result, 0), allocator);
      }
    } else {
      status = loom_opt_print_module(iree_make_cstring_view(FLAG_output),
                                     public_module, output_format, allocator);
    }
  }
  status = iree_status_join(status,
                            loom_tooling_loomc_pass_trace_close(&pass_trace));

  bool had_error = !module_ready;
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    had_error = true;
  }

  if (pass_report_initialized) {
    loom_pass_report_deinitialize(&pass_report);
  }
  loom_config_text_binding_set_deinitialize(&config_set);
  loomc_result_release(public_result);
  loomc_source_release(reproducer_input);
  loomc_pass_program_release(pass_program);
  loomc_module_release(public_module);
  loomc_source_release(public_source);
  iree_io_file_contents_free(contents);
  loomc_workspace_release(public_workspace);
  loomc_context_release(public_context);
  loomc_target_environment_release(public_target_environment);

  const int exit_code = had_error ? 1 : 0;
  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
