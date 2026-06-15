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
#include "loom/codegen/low/pipeline/pass_environment.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/codegen/low/verify.h"
#include "loom/error/diagnostic.h"
#include "loom/error/json_sink.h"
#include "loom/format/text/printer.h"
#include "loom/ir/module.h"
#include "loom/pass/builtin_registry.h"
#include "loom/pass/pipeline.h"
#include "loom/pass/registry.h"
#include "loom/pass/report.h"
#include "loom/pass/tooling.h"
#include "loom/target/configured/provider.h"
#include "loom/target/pipeline.h"
#include "loom/target/predicate.h"
#include "loom/target/provider.h"
#include "loom/tooling/cli/help.h"
#include "loom/tooling/config/config.h"
#include "loom/tooling/context/context.h"
#include "loom/tooling/execution/session.h"
#include "loom/tooling/io/file.h"
#include "loom/tooling/pass/trace_cli.h"
#include "loom/util/json.h"
#include "loom/util/stream.h"
#include "loom/verify/verify.h"

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
          "Verify the module before and after executing passes.");
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
IREE_FLAG_NAMED(string, low_asm_descriptor_set, "low-asm-descriptor-set", "",
                "Descriptor-set key used when printing low asm regions.");

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

typedef struct loom_opt_diagnostic_emitter_t {
  // Module containing the op referenced by emitted diagnostics.
  const loom_module_t* module;
  // Source resolver for source-backed operation locations.
  loom_source_resolver_t source_resolver;
  // Destination for materialized low verifier diagnostics.
  loom_diagnostic_sink_t diagnostic_sink;
  // Subsystem identity to store in materialized diagnostics.
  loom_emitter_t emitter;
} loom_opt_diagnostic_emitter_t;

typedef struct loom_opt_diagnostic_sink_t {
  // Selected diagnostic output format.
  loom_opt_diagnostic_format_t format;
  // Parsed module used for full type rendering after parse succeeds.
  const loom_run_module_t* run_module;
  // Printer context used to render target-owned register and storage types.
  loom_low_descriptor_text_print_context_t type_print_context;
  // Stream used when structured diagnostics are requested.
  loom_output_stream_t json_stream;
} loom_opt_diagnostic_sink_t;

static iree_status_t loom_opt_format_diagnostic_type(
    loom_type_t type, void* user_data, loom_output_stream_t* stream) {
  const loom_opt_diagnostic_sink_t* sink =
      (const loom_opt_diagnostic_sink_t*)user_data;
  const loom_module_t* module =
      sink && sink->run_module ? sink->run_module->module : NULL;
  if (sink && sink->type_print_context.options.low_asm_environment.vtable) {
    return loom_text_print_type_with_options(type, module, stream,
                                             &sink->type_print_context.options);
  }
  if (module) {
    return loom_text_print_type(type, module, stream);
  }
  return loom_type_format_minimal(type, NULL, stream);
}

static iree_status_t loom_opt_diagnostic_sink(
    void* user_data, const loom_diagnostic_t* diagnostic) {
  loom_opt_diagnostic_sink_t* sink = (loom_opt_diagnostic_sink_t*)user_data;
  const loom_type_formatter_t type_formatter = {
      .fn = loom_opt_format_diagnostic_type,
      .user_data = sink,
  };
  switch (sink->format) {
    case LOOM_OPT_DIAGNOSTIC_FORMAT_TEXT: {
      loom_output_stream_t stream;
      loom_output_stream_for_file(stderr, &stream);
      const loom_diagnostic_format_options_t format_options = {
          .type_formatter = type_formatter,
      };
      return loom_diagnostic_format_with_options(diagnostic, &format_options,
                                                 &stream);
    }
    case LOOM_OPT_DIAGNOSTIC_FORMAT_JSON:
      return loom_diagnostic_json_write_object(&sink->json_stream, diagnostic,
                                               type_formatter);
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported diagnostic format");
  }
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

static iree_string_view_t loom_opt_pipeline_kind_stage_name(
    loom_opt_pipeline_kind_t pipeline_kind) {
  switch (pipeline_kind) {
    case LOOM_OPT_PIPELINE_MODULE_SYMBOL:
      return IREE_SV("module-pipeline");
    case LOOM_OPT_PIPELINE_COMMAND_LINE:
      return IREE_SV("command-line");
    case LOOM_OPT_PIPELINE_SOURCE_LOW:
      return IREE_SV("source-low");
    case LOOM_OPT_PIPELINE_PREPARED_LOW:
      return IREE_SV("prepared-low");
    default:
      return IREE_SV("none");
  }
}

static iree_status_t loom_opt_register_context(void* user_data,
                                               loom_context_t* context) {
  const loom_target_environment_t* target_environment =
      (const loom_target_environment_t*)user_data;
  return loom_tooling_context_register_tool_dialects_with_target_environment(
      target_environment, context);
}

static iree_status_t loom_opt_initialize_low_descriptor_registry(
    void* user_data, loom_target_low_descriptor_registry_t* out_registry) {
  const loom_target_environment_t* target_environment =
      (const loom_target_environment_t*)user_data;
  return loom_target_environment_initialize_low_descriptor_registry(
      target_environment, out_registry);
}

static bool loom_opt_resolve_emission_location(
    const loom_opt_diagnostic_emitter_t* emitter, const loom_op_t* op,
    loom_source_range_t* out_source_location) {
  if (!emitter || !emitter->module || !op) {
    return false;
  }
  if (!loom_source_resolve(emitter->source_resolver, emitter->module,
                           op->location, out_source_location)) {
    return false;
  }
  if (out_source_location->provenance ==
          LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE &&
      out_source_location->source.size > 0) {
    out_source_location->provenance = LOOM_SOURCE_PROVENANCE_EXACT_SOURCE;
  }
  return true;
}

static iree_host_size_t loom_opt_collect_related_locations(
    const loom_opt_diagnostic_emitter_t* emitter,
    const loom_diagnostic_related_op_t* related_ops,
    iree_host_size_t related_op_count,
    loom_diagnostic_related_location_t* out_related_locations,
    iree_host_size_t* out_omitted_count) {
  *out_omitted_count = 0;
  if (!related_ops || related_op_count == 0) {
    return 0;
  }
  iree_host_size_t related_location_count = 0;
  for (iree_host_size_t i = 0; i < related_op_count; ++i) {
    loom_source_range_t source_location = {
        .provenance = LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE,
    };
    if (!loom_opt_resolve_emission_location(emitter, related_ops[i].op,
                                            &source_location)) {
      continue;
    }
    if (related_location_count >= LOOM_DIAGNOSTIC_MAX_RELATED_LOCATIONS) {
      ++*out_omitted_count;
      continue;
    }
    out_related_locations[related_location_count++] =
        (loom_diagnostic_related_location_t){
            .label = related_ops[i].label,
            .source_location = source_location,
        };
  }
  return related_location_count;
}

static iree_status_t loom_opt_diagnostic_emitter_emit(
    void* user_data, const loom_diagnostic_emission_t* emission) {
  loom_opt_diagnostic_emitter_t* emitter =
      (loom_opt_diagnostic_emitter_t*)user_data;
  if (!emitter || !emission || !emission->error) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "diagnostic emitter requires an emission");
  }

  loom_diagnostic_t diagnostic = {
      .severity = emission->error->severity,
      .error = emission->error,
      .params = emission->params,
      .param_count = emission->param_count,
      .emitter = emitter->emitter,
      .origin = {.provenance = LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE},
      .source_location = {.provenance =
                              LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE},
  };

  loom_diagnostic_related_location_t
      related_locations[LOOM_DIAGNOSTIC_MAX_RELATED_LOCATIONS];
  diagnostic.related_location_count = loom_opt_collect_related_locations(
      emitter, emission->related_ops, emission->related_op_count,
      related_locations, &diagnostic.related_location_omitted_count);
  if (diagnostic.related_location_count > 0) {
    diagnostic.related_locations = related_locations;
  }

  if (loom_opt_resolve_emission_location(emitter, emission->op,
                                         &diagnostic.source_location)) {
    diagnostic.origin = diagnostic.source_location;
  }
  return loom_diagnostic_emit(&emitter->diagnostic_sink, &diagnostic);
}

static iree_status_t loom_opt_verify_module(
    const loom_target_low_descriptor_registry_t* low_registry,
    loom_run_module_t* run_module, loom_diagnostic_sink_t diagnostic_sink) {
  loom_module_t* module = run_module->module;
  loom_source_resolver_t source_resolver =
      loom_run_module_source_resolver(run_module);

  loom_verify_options_t verify_options = {
      .sink = diagnostic_sink,
      .max_errors = 100,
      .source_resolver = source_resolver,
  };
  loom_verify_result_t verify_result = {0};
  IREE_RETURN_IF_ERROR(
      loom_verify_module(module, &verify_options, &verify_result));
  if (verify_result.error_count > 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "module verification failed with %" PRIu32 " error%s",
        verify_result.error_count, verify_result.error_count == 1 ? "" : "s");
  }

  loom_opt_diagnostic_emitter_t low_emitter = {
      .module = module,
      .source_resolver = source_resolver,
      .diagnostic_sink = diagnostic_sink,
      .emitter = LOOM_EMITTER_VERIFIER,
  };
  loom_low_verify_options_t low_verify_options = {
      .descriptor_registry = &low_registry->registry,
      .emitter = {.fn = loom_opt_diagnostic_emitter_emit,
                  .user_data = &low_emitter},
      .max_errors = 100,
  };
  loom_low_verify_result_t low_verify_result = {0};
  loom_low_verify_scratch_t low_verify_scratch =
      loom_low_verify_scratch_for_module(module);
  IREE_RETURN_IF_ERROR(loom_low_verify_module(
      module, &low_verify_options, &low_verify_scratch, &low_verify_result));
  if (low_verify_result.error_count > 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low verification failed with %" PRIu32 " error%s",
                            low_verify_result.error_count,
                            low_verify_result.error_count == 1 ? "" : "s");
  }
  return iree_ok_status();
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
    iree_string_builder_t* builder, bool use_synthetic_pipeline) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "// RUN: loom-opt --verify=%s", FLAG_verify ? "true" : "false"));
  iree_flag_string_list_t config_assignments = FLAG_config_list();
  for (iree_host_size_t i = 0; i < config_assignments.count; ++i) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(builder, " --config="));
    IREE_RETURN_IF_ERROR(
        loom_opt_append_shell_quoted(builder, config_assignments.values[i]));
  }
  iree_string_view_t low_asm_descriptor_set = iree_string_view_trim(
      iree_make_cstring_view(FLAG_low_asm_descriptor_set));
  if (!iree_string_view_is_empty(low_asm_descriptor_set)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
        builder, " --low-asm-descriptor-set="));
    IREE_RETURN_IF_ERROR(
        loom_opt_append_shell_quoted(builder, low_asm_descriptor_set));
  }
  iree_string_view_t pass_report =
      iree_string_view_trim(iree_make_cstring_view(FLAG_pass_report));
  if (!iree_string_view_is_empty(pass_report)) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(builder, " --pass-report="));
    IREE_RETURN_IF_ERROR(loom_opt_append_shell_quoted(builder, pass_report));
  }

  iree_string_view_t pipeline_symbol =
      iree_string_view_trim(iree_make_cstring_view(FLAG_pipeline));
  if (use_synthetic_pipeline) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
        builder, " --pipeline='@__command_line'"));
  } else if (!iree_string_view_is_empty(pipeline_symbol)) {
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

static bool loom_opt_pass_key_is_printable(iree_string_view_t key) {
  if (key.size == 0) return false;
  char first = key.data[0];
  bool valid_start = (first >= 'a' && first <= 'z') ||
                     (first >= 'A' && first <= 'Z') || first == '_' ||
                     first == '$';
  if (!valid_start) return false;
  for (iree_host_size_t i = 1; i < key.size; ++i) {
    char c = key.data[i];
    bool valid = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                 (c >= '0' && c <= '9') || c == '_' || c == '$' || c == '-' ||
                 c == '.';
    if (!valid) return false;
  }
  return true;
}

static iree_status_t loom_opt_append_reproducer_pass_run(
    iree_string_builder_t* builder, iree_string_view_t indent,
    const loom_pass_pipeline_entry_spec_t* spec) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(builder, indent));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "pass.run<"));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(builder, spec->name));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ">"));
  if (!iree_string_view_is_empty(spec->options)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, " {"));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_string(builder, spec->options));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "}"));
  }
  return iree_string_builder_append_cstring(builder, "\n");
}

static iree_status_t loom_opt_reproducer_function_group_open(
    iree_string_builder_t* builder, bool* in_function_group) {
  if (*in_function_group) {
    return iree_ok_status();
  }
  *in_function_group = true;
  return iree_string_builder_append_cstring(builder,
                                            "  pass.for<func> pipeline {\n");
}

static iree_status_t loom_opt_reproducer_function_group_close(
    iree_string_builder_t* builder, bool* in_function_group) {
  if (!*in_function_group) {
    return iree_ok_status();
  }
  *in_function_group = false;
  return iree_string_builder_append_cstring(builder, "    pass.yield\n  }\n");
}

static iree_status_t loom_opt_build_reproducer_synthetic_pipeline(
    iree_string_builder_t* builder, const loom_pass_registry_t* registry,
    bool* out_use_synthetic_pipeline, iree_allocator_t allocator) {
  *out_use_synthetic_pipeline = false;
  iree_string_view_t pipeline_symbol =
      iree_string_view_trim(iree_make_cstring_view(FLAG_pipeline));
  iree_flag_string_list_t passes = FLAG_pass_list();
  if (!iree_string_view_is_empty(pipeline_symbol) || passes.count == 0) {
    return iree_ok_status();
  }

  iree_string_builder_t pipeline_builder;
  iree_string_builder_initialize(allocator, &pipeline_builder);
  iree_status_t status = iree_string_builder_append_cstring(
      &pipeline_builder, "pass.pipeline<module> @__command_line {\n");

  iree_string_builder_t pass_list_builder;
  iree_string_builder_initialize(allocator, &pass_list_builder);
  if (iree_status_is_ok(status)) {
    status = loom_opt_join_pass_list(passes, &pass_list_builder);
  }

  iree_string_view_t remaining = iree_string_builder_view(&pass_list_builder);
  bool in_function_group = false;
  while (iree_status_is_ok(status)) {
    loom_pass_pipeline_entry_spec_t spec = {0};
    bool has_entry = false;
    status = loom_pass_pipeline_consume_entry(&remaining, &spec, &has_entry);
    if (!iree_status_is_ok(status) || !has_entry) break;
    if (!loom_opt_pass_key_is_printable(spec.name)) {
      iree_string_builder_reset(&pipeline_builder);
      status = iree_ok_status();
      break;
    }

    const loom_pass_descriptor_t* descriptor = NULL;
    status = loom_pass_registry_lookup(registry, spec.name, &descriptor);
    if (!iree_status_is_ok(status)) break;
    const loom_pass_info_t* info = descriptor ? descriptor->info() : NULL;
    if (info && info->kind == LOOM_PASS_FUNCTION) {
      status = loom_opt_reproducer_function_group_open(&pipeline_builder,
                                                       &in_function_group);
      if (iree_status_is_ok(status)) {
        status = loom_opt_append_reproducer_pass_run(&pipeline_builder,
                                                     IREE_SV("    "), &spec);
      }
    } else {
      status = loom_opt_reproducer_function_group_close(&pipeline_builder,
                                                        &in_function_group);
      if (iree_status_is_ok(status)) {
        status = loom_opt_append_reproducer_pass_run(&pipeline_builder,
                                                     IREE_SV("  "), &spec);
      }
    }
  }

  status = iree_status_join(status, loom_opt_reproducer_function_group_close(
                                        &pipeline_builder, &in_function_group));
  if (iree_status_is_ok(status) &&
      iree_string_builder_size(&pipeline_builder)) {
    status = iree_string_builder_append_cstring(&pipeline_builder,
                                                "  pass.yield\n}\n");
  }
  if (iree_status_is_ok(status) &&
      iree_string_builder_size(&pipeline_builder)) {
    status = iree_string_builder_append_string(
        builder, iree_string_builder_view(&pipeline_builder));
  }
  if (iree_status_is_ok(status) &&
      iree_string_builder_size(&pipeline_builder)) {
    *out_use_synthetic_pipeline = true;
  }

  iree_string_builder_deinitialize(&pass_list_builder);
  iree_string_builder_deinitialize(&pipeline_builder);
  return status;
}

static iree_status_t loom_opt_format_pass_selection_json(
    loom_output_stream_t* stream) {
  iree_string_view_t pipeline_symbol =
      iree_string_view_trim(iree_make_cstring_view(FLAG_pipeline));
  if (!iree_string_view_is_empty(pipeline_symbol)) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, "{\"kind\":\"pipeline\","));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, "\"pipeline\":"));
    IREE_RETURN_IF_ERROR(
        loom_json_write_escaped_string(stream, pipeline_symbol));
    return loom_output_stream_write_cstring(stream, "}");
  }

  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "{\"kind\":\"pass-list\","));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "\"passes\":["));
  iree_flag_string_list_t passes = FLAG_pass_list();
  for (iree_host_size_t i = 0; i < passes.count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, i == 0 ? "" : ","));
    IREE_RETURN_IF_ERROR(
        loom_json_write_escaped_string(stream, passes.values[i]));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "]}"));
  return iree_ok_status();
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

  iree_status_t status = loom_output_stream_write_cstring(
      &metadata_stream, "{\n  \"selection\": ");
  if (iree_status_is_ok(status)) {
    status = loom_opt_format_pass_selection_json(&metadata_stream);
  }
  if (iree_status_is_ok(status)) {
    status = loom_output_stream_write_cstring(&metadata_stream,
                                              ",\n  \"failure_status\": ");
  }
  if (iree_status_is_ok(status)) {
    status = loom_json_write_escaped_cstring(
        &metadata_stream, iree_status_code_string(failure_status_code));
  }
  if (iree_status_is_ok(status)) {
    status = loom_output_stream_write_cstring(&metadata_stream,
                                              ",\n  \"registry\": ");
  }
  if (iree_status_is_ok(status)) {
    status = loom_pass_report_format_registry_json(registry, &metadata_stream);
  }
  if (iree_status_is_ok(status)) {
    status = loom_output_stream_write_cstring(&metadata_stream, "}\n");
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
    iree_string_view_t source, const loom_pass_registry_t* registry,
    iree_status_code_t failure_status_code, iree_allocator_t allocator) {
  path = iree_string_view_trim(path);
  if (iree_string_view_is_empty(path)) {
    return iree_ok_status();
  }

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

  iree_string_builder_t synthetic_pipeline_builder;
  iree_string_builder_initialize(allocator, &synthetic_pipeline_builder);
  bool use_synthetic_pipeline = false;
  if (iree_status_is_ok(status)) {
    status = loom_opt_build_reproducer_synthetic_pipeline(
        &synthetic_pipeline_builder, registry, &use_synthetic_pipeline,
        allocator);
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_opt_append_reproducer_run_line(&builder, use_synthetic_pipeline);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(&builder, "\n");
  }
  if (iree_status_is_ok(status) && use_synthetic_pipeline) {
    status = iree_string_builder_append_string(
        &builder, iree_string_builder_view(&synthetic_pipeline_builder));
    if (iree_status_is_ok(status)) {
      status = iree_string_builder_append_cstring(&builder, "\n");
    }
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
  iree_string_builder_deinitialize(&synthetic_pipeline_builder);
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

static iree_status_t loom_opt_run_shared_compile_pipeline(
    loom_module_t* module, loom_opt_pipeline_kind_t pipeline_kind,
    const loom_target_environment_t* target_environment,
    const loom_pass_tool_run_options_t* run_options,
    loom_pass_run_result_t* out_result) {
  *out_result = (loom_pass_run_result_t){0};

  loom_module_t* pipeline_module = NULL;
  iree_status_t status = loom_module_allocate(
      module->context, IREE_SV("__loom_opt_compile_pipeline"),
      run_options->block_pool, NULL, module->allocator, &pipeline_module);
  loom_op_t* pipeline_op = NULL;
  const loom_target_pipeline_options_t target_pipeline_options = {0};
  if (iree_status_is_ok(status)) {
    switch (pipeline_kind) {
      case LOOM_OPT_PIPELINE_SOURCE_LOW:
        status = loom_target_pipeline_build_to_source_low(
            pipeline_module, IREE_SV("__loom_opt_source_low"),
            &target_pipeline_options, target_environment,
            run_options->environment, &pipeline_op);
        break;
      case LOOM_OPT_PIPELINE_PREPARED_LOW:
        status = loom_target_pipeline_build_to_prepared_low(
            pipeline_module, IREE_SV("__loom_opt_prepared_low"),
            &target_pipeline_options, target_environment,
            run_options->environment, &pipeline_op);
        break;
      default:
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "pipeline kind is not a compile stage");
        break;
    }
  }
  if (iree_status_is_ok(status) && pipeline_op == NULL) {
    status = iree_make_status(IREE_STATUS_INTERNAL,
                              "compile pipeline builder produced no op");
  }
  if (iree_status_is_ok(status)) {
    status = loom_pass_tool_run_pipeline_module_op(
        module, pipeline_module, pipeline_op, run_options, out_result);
  }
  if (pipeline_module != NULL) {
    loom_module_free(pipeline_module);
  }
  return status;
}

static iree_status_t loom_opt_run_passes(
    const loom_target_low_descriptor_registry_t* low_registry,
    const loom_target_environment_t* target_environment,
    const loom_pass_registry_t* pass_registry,
    iree_arena_block_pool_t* block_pool, loom_run_module_t* run_module,
    loom_diagnostic_sink_t diagnostic_sink, loom_pass_report_t* report,
    const loom_pass_trace_options_t* trace_options, bool* out_execution_started,
    loom_pass_run_result_t* out_result, iree_allocator_t allocator) {
  loom_module_t* module = run_module->module;
  *out_execution_started = false;
  *out_result = (loom_pass_run_result_t){0};
  iree_flag_string_list_t passes = FLAG_pass_list();
  iree_string_view_t pipeline_symbol =
      iree_string_view_trim(iree_make_cstring_view(FLAG_pipeline));
  loom_opt_pipeline_kind_t pipeline_kind = LOOM_OPT_PIPELINE_NONE;
  IREE_RETURN_IF_ERROR(loom_opt_parse_pipeline_kind(
      pipeline_symbol, passes.count > 0, &pipeline_kind));
  if (pipeline_kind == LOOM_OPT_PIPELINE_NONE) {
    return iree_ok_status();
  }

  loom_source_resolver_t source_resolver =
      loom_run_module_source_resolver(run_module);
  loom_opt_diagnostic_emitter_t pass_emitter = {
      .module = module,
      .source_resolver = source_resolver,
      .diagnostic_sink = diagnostic_sink,
      .emitter = LOOM_EMITTER_PASS,
  };

  loom_low_lower_policy_registry_t low_lower_policy_registry = {0};
  IREE_RETURN_IF_ERROR(
      loom_target_environment_initialize_low_lower_policy_registry(
          target_environment, &low_lower_policy_registry));
  loom_target_math_policy_registry_t math_policy_registry = {0};
  IREE_RETURN_IF_ERROR(loom_target_environment_initialize_math_policy_registry(
      target_environment, &math_policy_registry));
  const loom_target_low_legality_provider_list_t low_legality_provider_list =
      loom_target_environment_low_legality_provider_list(target_environment);
  const loom_target_legalizer_provider_list_t legalizer_provider_list =
      loom_target_environment_legalizer_provider_list(target_environment);
  loom_low_pass_environment_storage_t low_pass_environment_storage;
  loom_target_pass_predicate_provider_storage_t predicate_storage;
  loom_target_pass_predicate_provider_storage_initialize(block_pool,
                                                         &predicate_storage);
  loom_pass_trace_options_t run_trace_options = {0};
  loom_pass_trace_t trace = {0};
  loom_pass_trace_t* trace_ptr = NULL;
  if (loom_pass_trace_options_is_enabled(trace_options)) {
    run_trace_options = *trace_options;
    run_trace_options.stage = loom_opt_pipeline_kind_stage_name(pipeline_kind);
    loom_pass_trace_initialize(&run_trace_options, &trace);
    trace_ptr = &trace;
  }
  loom_pass_tool_run_options_t run_options = {
      .registry = pass_registry,
      .environment = loom_low_pass_environment_storage_initialize(
          &low_registry->registry, &low_lower_policy_registry,
          &low_legality_provider_list, &legalizer_provider_list,
          &math_policy_registry, /*compile_report=*/NULL,
          loom_target_selection_empty(), loom_symbol_ref_null(),
          &low_pass_environment_storage),
      .predicate_provider =
          loom_target_pass_predicate_provider(&predicate_storage),
      .block_pool = block_pool,
      .diagnostic_emitter = {.fn = loom_opt_diagnostic_emitter_emit,
                             .user_data = &pass_emitter},
      .report = report,
      .trace = trace_ptr,
  };

  if (pipeline_kind == LOOM_OPT_PIPELINE_MODULE_SYMBOL) {
    *out_execution_started = true;
    return loom_pass_tool_run_pipeline_symbol(module, pipeline_symbol,
                                              &run_options, out_result);
  }
  if (pipeline_kind == LOOM_OPT_PIPELINE_SOURCE_LOW ||
      pipeline_kind == LOOM_OPT_PIPELINE_PREPARED_LOW) {
    *out_execution_started = true;
    return loom_opt_run_shared_compile_pipeline(
        module, pipeline_kind, target_environment, &run_options, out_result);
  }

  iree_string_builder_t pipeline_builder;
  iree_string_builder_initialize(allocator, &pipeline_builder);
  iree_status_t status = loom_opt_join_pass_list(passes, &pipeline_builder);
  if (iree_status_is_ok(status)) {
    *out_execution_started = true;
    status = loom_pass_tool_run_flat_pipeline(
        module, iree_string_builder_view(&pipeline_builder), &run_options,
        out_result);
  }
  iree_string_builder_deinitialize(&pipeline_builder);
  return status;
}

static iree_status_t loom_opt_append_config_flags(
    loom_tooling_config_set_t* config_set) {
  iree_flag_string_list_t assignments = FLAG_config_list();
  for (iree_host_size_t i = 0; i < assignments.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_tooling_config_set_append_assignment(
        config_set, assignments.values[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_opt_append_config_files(
    loom_tooling_config_set_t* config_set, iree_allocator_t allocator) {
  iree_flag_string_list_t paths = FLAG_config_file_list();
  for (iree_host_size_t i = 0; i < paths.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_tooling_config_set_append_json_file(
        config_set, paths.values[i], allocator));
  }
  return iree_ok_status();
}

static iree_status_t loom_opt_materialize_config_set(
    loom_module_t* module, const loom_tooling_config_set_t* config_set,
    iree_arena_block_pool_t* block_pool,
    loom_tooling_config_materialize_result_t* out_result) {
  *out_result = (loom_tooling_config_materialize_result_t){0};
  loom_tooling_config_materialize_options_t options;
  loom_tooling_config_materialize_options_initialize(&options);
  options.config_set = config_set;
  return loom_tooling_config_materialize_module(module, &options, block_pool,
                                                out_result);
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

static iree_status_t loom_opt_print_module(
    iree_string_view_t output_path,
    const loom_target_low_descriptor_registry_t* low_registry,
    const loom_module_t* module, iree_allocator_t allocator) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(allocator, &builder);

  loom_text_print_options_t print_options = {
      .flags = LOOM_TEXT_PRINT_DEFAULT,
      .low_asm_descriptor_set_key =
          iree_make_cstring_view(FLAG_low_asm_descriptor_set),
  };
  loom_low_descriptor_text_asm_environment_initialize(
      &low_registry->registry, &print_options.low_asm_environment);
  iree_status_t status = loom_text_print_module_to_builder_with_options(
      module, &builder, &print_options);
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_string(&builder, IREE_SV("\n"));
  }
  if (iree_status_is_ok(status)) {
    status = loom_tooling_write_output_file(
        output_path, iree_string_builder_view(&builder), allocator);
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

static iree_status_t loom_opt_print_config_schema(
    iree_string_view_t output_path, const loom_module_t* module,
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
        output_path, iree_string_builder_view(&builder), allocator);
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

static iree_status_t loom_opt_print_pass_list(
    const loom_pass_registry_t* registry) {
  for (iree_host_size_t i = 0; i < registry->descriptor_count; ++i) {
    const loom_pass_descriptor_t* descriptor = &registry->descriptors[i];
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
  const loom_pass_descriptor_t* descriptor = NULL;
  IREE_RETURN_IF_ERROR(loom_pass_registry_lookup(registry, key, &descriptor));
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
      "### Diagnostics and reports\n"
      "\n"
      "```shell\n"
      "loom-opt input.loom --pipeline=source-low --pass-report=json \\\n"
      "  --diagnostic-format=json --output=/tmp/out.loom\n"
      "loom-opt input.loom --pipeline=source-low --dump-ir-after-all \\\n"
      "  --dump-ir-format=jsonl --dump-ir-output=/tmp/trace.jsonl\n"
      "loom-opt input.loom --pass=unroll-scf-for --pass-report=json "
      "2>report.json\n"
      "jq '.passes[] | {pass, op, changed, details}' report.json\n"
      "jq 'select(.pass == \"source-to-low\") | .ir' /tmp/trace.jsonl\n"
      "```\n"
      "\n"
      "`--pass-report=json` is the structured channel for pass decisions and\n"
      "per-pass facts. `--diagnostic-format=json` emits compiler diagnostics "
      "as\n"
      "JSONL. `--dump-ir-before*` and `--dump-ir-after*` capture intermediate "
      "IR\n"
      "for bisection; prefer `--dump-ir-format=jsonl` for agent ingestion.\n");
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
      "stdout.\n"
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
  loom_tooling_cli_set_default_help_filter();
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_allocator_t allocator = iree_allocator_system();

  iree_io_file_contents_t* contents = NULL;
  loom_run_session_t run_session = {0};
  loom_run_module_t run_module = {0};
  loom_tooling_config_set_t config_set;
  loom_tooling_config_set_initialize(allocator, &config_set);
  const loom_target_environment_t* target_environment =
      loom_configured_target_environment();
  loom_pass_registry_storage_t pass_registry_storage = {0};
  const loom_pass_registry_t* pass_registry = NULL;
  iree_string_view_t source = iree_string_view_empty();
  loom_opt_pass_report_mode_t pass_report_mode = LOOM_OPT_PASS_REPORT_NONE;
  loom_opt_diagnostic_format_t diagnostic_format =
      LOOM_OPT_DIAGNOSTIC_FORMAT_TEXT;
  loom_opt_diagnostic_sink_t diagnostic_sink_state = {
      .run_module = &run_module,
  };
  loom_diagnostic_sink_t diagnostic_sink = {
      .fn = loom_opt_diagnostic_sink,
      .user_data = &diagnostic_sink_state,
  };
  loom_pass_report_t pass_report = {0};
  bool pass_report_initialized = false;
  loom_tooling_pass_trace_t pass_trace = {0};
  bool pass_execution_started = false;
  loom_tooling_config_materialize_result_t config_materialize_result = {0};
  loom_pass_run_result_t pass_run_result = {0};
  iree_status_t pass_pipeline_status = iree_ok_status();

  const loom_pass_registry_t* pass_registries[] = {
      loom_pass_builtin_registry(),
      loom_target_environment_pass_registry(target_environment),
  };
  iree_status_t status = loom_pass_registry_storage_initialize_from_registries(
      pass_registries, IREE_ARRAYSIZE(pass_registries), &pass_registry_storage);
  if (iree_status_is_ok(status)) {
    pass_registry = loom_pass_registry_storage_registry(&pass_registry_storage);
  }
  if (iree_status_is_ok(status)) {
    status = loom_opt_parse_pass_report_mode(
        iree_make_cstring_view(FLAG_pass_report), &pass_report_mode);
  }
  if (iree_status_is_ok(status)) {
    status = loom_opt_parse_diagnostic_format(
        iree_make_cstring_view(FLAG_diagnostic_format), &diagnostic_format);
  }
  if (iree_status_is_ok(status)) {
    diagnostic_sink_state.format = diagnostic_format;
    if (diagnostic_format == LOOM_OPT_DIAGNOSTIC_FORMAT_JSON) {
      loom_output_stream_for_file(stderr, &diagnostic_sink_state.json_stream);
    }
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
    status = loom_opt_append_config_files(&config_set, allocator);
  }
  if (iree_status_is_ok(status) && !metadata_only) {
    status = loom_opt_append_config_flags(&config_set);
  }
  if (iree_status_is_ok(status) && !metadata_only) {
    loom_run_session_options_t session_options = {0};
    loom_run_session_options_initialize(&session_options);
    session_options.host_allocator = allocator;
    session_options.register_context = (loom_run_register_context_callback_t){
        .fn = loom_opt_register_context,
        .user_data = (void*)target_environment,
    };
    session_options.initialize_low_descriptor_registry =
        (loom_run_initialize_low_descriptor_registry_callback_t){
            .fn = loom_opt_initialize_low_descriptor_registry,
            .user_data = (void*)target_environment,
        };
    status = loom_run_session_initialize(&session_options, &run_session);
    if (iree_status_is_ok(status)) {
      loom_low_descriptor_text_print_context_initialize(
          &loom_run_session_low_descriptor_registry(&run_session)->registry,
          &diagnostic_sink_state.type_print_context);
    }
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
    loom_run_module_parse_options_t parse_options = {0};
    loom_run_module_parse_options_initialize(&parse_options);
    parse_options.filename = filename;
    parse_options.source = source;
    parse_options.diagnostic_sink = diagnostic_sink;
    status = loom_run_module_parse(&run_session, &parse_options, &run_module);
  }
  if (iree_status_is_ok(status) && !metadata_only && FLAG_verify) {
    status = loom_opt_verify_module(
        loom_run_session_low_descriptor_registry(&run_session), &run_module,
        diagnostic_sink);
  }
  if (iree_status_is_ok(status) && !metadata_only) {
    status = loom_opt_materialize_config_set(
        run_module.module, &config_set,
        loom_run_session_block_pool(&run_session), &config_materialize_result);
  }
  if (iree_status_is_ok(status) && !metadata_only && FLAG_verify &&
      config_materialize_result.materialized_count > 0) {
    status = loom_opt_verify_module(
        loom_run_session_low_descriptor_registry(&run_session), &run_module,
        diagnostic_sink);
  }
  if (iree_status_is_ok(status) && !metadata_only) {
    const loom_tooling_pass_trace_stdout_conflict_t stdout_conflicts[] = {
        {
            .active = true,
            .flag_name = IREE_SV("--output"),
            .path = iree_make_cstring_view(FLAG_output),
        },
    };
    status = loom_tooling_pass_trace_open_from_flags(
        &(loom_tooling_pass_trace_open_options_t){
            .tool_name = IREE_SV("loom-opt"),
            .input_path = filename,
            .low_asm_descriptor_set_key =
                iree_make_cstring_view(FLAG_low_asm_descriptor_set),
            .stdout_conflicts = stdout_conflicts,
            .stdout_conflict_count = IREE_ARRAYSIZE(stdout_conflicts),
        },
        allocator, &pass_trace);
    if (iree_status_is_ok(status) && pass_trace.enabled) {
      loom_low_descriptor_text_asm_environment_initialize(
          &loom_run_session_low_descriptor_registry(&run_session)->registry,
          &pass_trace.pass_options.print_options.low_asm_environment);
    }
  }
  if (iree_status_is_ok(status) && !metadata_only) {
    pass_pipeline_status = loom_opt_run_passes(
        loom_run_session_low_descriptor_registry(&run_session),
        target_environment, pass_registry,
        loom_run_session_block_pool(&run_session), &run_module, diagnostic_sink,
        pass_report_initialized ? &pass_report : NULL,
        loom_tooling_pass_trace_options(&pass_trace), &pass_execution_started,
        &pass_run_result, allocator);
    status = pass_pipeline_status;
  }
  bool pass_pipeline_failed = !iree_status_is_ok(pass_pipeline_status) ||
                              pass_run_result.error_count != 0;
  if (!metadata_only && pass_execution_started && pass_pipeline_failed) {
    status = iree_status_join(
        status, loom_opt_write_pass_reproducer(
                    iree_make_cstring_view(FLAG_pass_reproducer), filename,
                    source, pass_registry,
                    !iree_status_is_ok(pass_pipeline_status)
                        ? iree_status_code(pass_pipeline_status)
                        : IREE_STATUS_FAILED_PRECONDITION,
                    allocator));
  }
  if (!metadata_only && pass_report_initialized && pass_execution_started) {
    iree_status_t report_status =
        loom_opt_write_pass_report(pass_report_mode, &pass_report);
    status = iree_status_join(status, report_status);
  }
  if (iree_status_is_ok(status) && pass_run_result.error_count == 0 &&
      !metadata_only && FLAG_verify) {
    status = loom_opt_verify_module(
        loom_run_session_low_descriptor_registry(&run_session), &run_module,
        diagnostic_sink);
    if (!iree_status_is_ok(status) && pass_execution_started &&
        iree_status_is_ok(pass_pipeline_status)) {
      iree_status_code_t status_code = iree_status_code(status);
      status = iree_status_join(
          status, loom_opt_write_pass_reproducer(
                      iree_make_cstring_view(FLAG_pass_reproducer), filename,
                      source, pass_registry, status_code, allocator));
    }
  }
  if (iree_status_is_ok(status) && pass_run_result.error_count == 0 &&
      !metadata_only && FLAG_require_resolved_config) {
    status =
        loom_tooling_config_require_resolved_module(run_module.module, NULL);
  }
  if (iree_status_is_ok(status) && pass_run_result.error_count == 0 &&
      !metadata_only) {
    if (FLAG_print_config_schema) {
      status = loom_opt_print_config_schema(iree_make_cstring_view(FLAG_output),
                                            run_module.module, allocator);
    } else {
      status = loom_opt_print_module(
          iree_make_cstring_view(FLAG_output),
          loom_run_session_low_descriptor_registry(&run_session),
          run_module.module, allocator);
    }
  }
  status = iree_status_join(status, loom_tooling_pass_trace_close(&pass_trace));

  bool had_error = pass_run_result.error_count != 0;
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    had_error = true;
  }

  if (pass_report_initialized) {
    loom_pass_report_deinitialize(&pass_report);
  }
  loom_tooling_config_set_deinitialize(&config_set);
  loom_run_module_deinitialize(&run_module);
  iree_io_file_contents_free(contents);
  loom_run_session_deinitialize(&run_session);
  return had_error ? 1 : 0;
}
