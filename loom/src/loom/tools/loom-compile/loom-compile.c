// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// loom-compile: compiles a Loom module to a runtime artifact.

#include <stdio.h>

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "loom/config/text_binding.h"
#include "loom/sanitizer/options.h"
#include "loom/tooling/cli/help.h"
#include "loom/tooling/cli/loomc_options.h"
#include "loom/tooling/cli/loomc_result.h"
#include "loom/tooling/config/input.h"
#include "loom/tooling/io/file.h"
#include "loom/tooling/io/source_path.h"
#include "loom/tooling/pass/trace_cli.h"
#include "loom/util/stream.h"
#include "loomc/iree.h"
#include "loomc/loomc.h"
#include "loomc/target/configured.h"

IREE_FLAG(string, format, "",
          "Optional exact artifact format, such as 'amdgpu-hsaco', "
          "'spirv', or 'wasm-binary'. Omit this to "
          "select the canonical format for the selected entries and target.");
IREE_FLAG(string, target, "",
          "Optional compilation target in family:selector form, such as "
          "'amdgpu:gfx11-generic' or 'spirv:vulkan1.3+bda'. Selects the exact "
          "profile for kernel entries or the public/retained functions of a "
          "module and their callees. Authored targets remain compatibility "
          "requirements; target-free source needs no target attributes.");
IREE_FLAG_LIST(
    string, root,
    "Root symbol to materialize before compilation. Repeat for "
    "multiple roots. Roots must have one homogeneous entry category. "
    "When omitted, the module must have at most one category of "
    "default entries; mixed categories require explicit roots. "
    "Command-program roots require the LoomC command-program "
    "transaction.");
IREE_FLAG_LIST_NAMED(
    string, exclude_root, "exclude-root",
    "Member of the selected default root set to omit before target "
    "specialization and dependency materialization. Repeat for multiple "
    "roots. Entry category inference occurs before exclusions are applied. "
    "Cannot be combined with --root.");
IREE_FLAG(string, pipeline, "default",
          "Pass pipeline to run before artifact emission. Use 'default' or "
          "empty for the selected format's default compile pipeline. 'none' "
          "disables all compiler transformations and passes the input directly "
          "to the selected emitter. Use '@symbol' to run a module-local "
          "pass.pipeline or a comma-separated pass list such as "
          "'canonicalize,cse'.");
IREE_FLAG(string, sanitizer, "none",
          "Sanitizer checks to insert in the default target pipeline: none, "
          "all, or a '|'-separated set of access, value, and operation.");
IREE_FLAG_NAMED(string, sanitizer_reporting, "sanitizer-reporting", "default",
                "Sanitizer assertion failure reporting mode in the default "
                "target pipeline: default, trap, or report-only.");
IREE_FLAG_LIST(
    string, config,
    "Compile-time config binding. Repeat as --config=key=value. Bindings not "
    "referenced by the loaded module are ignored.");
IREE_FLAG_LIST_NAMED(
    string, config_file, "config-file",
    "JSON/JSONC config object file. Repeat for multiple files. Nested object "
    "keys are flattened with '.' separators.");
IREE_FLAG(string, output, "-",
          "Output path for the selected single-file kernel or module format.");
IREE_FLAG_NAMED(
    string, compile_report, "compile-report", "",
    "Optional compile report output. Use 'summary'/'details' for structured "
    "JSON, 'text-summary'/'text-details' for human-readable text, or "
    "empty/'none'. Inspect structured reports with loom-compile-report.");
IREE_FLAG_NAMED(
    string, artifact_manifest, "artifact-manifest", "",
    "Optional emitted artifact manifest sidecar. Use 'summary', 'details', "
    "'analysis', or empty/'none'.");
IREE_FLAG_NAMED(
    string, emit_artifact_manifest, "emit-artifact-manifest", "",
    "Optional output path for --artifact-manifest JSON. Empty derives from the "
    "artifact output path by appending '.manifest.json'.");
IREE_FLAG_NAMED(string, compile_report_output, "compile-report-output",
                "stderr",
                "Output path for --compile-report. Use 'stderr', 'stdout'/'-', "
                "or a file path.");
IREE_FLAG_LIST_NAMED(
    string, source_prefix_map, "source-prefix-map",
    "Remap source paths in diagnostics and serialized locations. Repeat as\n"
    "--source-prefix-map=old=new; entries are applied in reverse order so the\n"
    "last matching map wins. Use old= to strip a prefix.");

static iree_string_view_t loom_compile_input_path(int argc, char** argv) {
  return argc < 2 ? iree_string_view_empty() : iree_make_cstring_view(argv[1]);
}

static iree_string_view_t loom_compile_input_identifier(
    iree_string_view_t input_path) {
  return loom_tooling_file_path_is_stdio(input_path) ? IREE_SV("<stdin>")
                                                     : input_path;
}

static iree_status_t loom_compile_parse_report_options(
    loomc_compile_report_options_t* out_options, bool* out_enabled) {
  *out_options = (loomc_compile_report_options_t){
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_REPORT_OPTIONS,
      .structure_size = sizeof(*out_options),
  };
  *out_enabled = false;
  const iree_string_view_t value =
      iree_string_view_trim(iree_make_cstring_view(FLAG_compile_report));
  if (iree_string_view_is_empty(value) ||
      iree_string_view_equal(value, IREE_SV("none"))) {
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("summary")) ||
      iree_string_view_equal(value, IREE_SV("json")) ||
      iree_string_view_equal(value, IREE_SV("json-summary"))) {
    out_options->mode = LOOMC_COMPILE_REPORT_MODE_SUMMARY;
    out_options->format = LOOMC_COMPILE_REPORT_FORMAT_JSON;
  } else if (iree_string_view_equal(value, IREE_SV("details")) ||
             iree_string_view_equal(value, IREE_SV("json-details"))) {
    out_options->mode = LOOMC_COMPILE_REPORT_MODE_DETAILS;
    out_options->format = LOOMC_COMPILE_REPORT_FORMAT_JSON;
  } else if (iree_string_view_equal(value, IREE_SV("text")) ||
             iree_string_view_equal(value, IREE_SV("text-summary"))) {
    out_options->mode = LOOMC_COMPILE_REPORT_MODE_SUMMARY;
    out_options->format = LOOMC_COMPILE_REPORT_FORMAT_TEXT;
  } else if (iree_string_view_equal(value, IREE_SV("text-details"))) {
    out_options->mode = LOOMC_COMPILE_REPORT_MODE_DETAILS;
    out_options->format = LOOMC_COMPILE_REPORT_FORMAT_TEXT;
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unsupported compile report request '%.*s'; expected 'none', "
        "'summary', 'details', 'json', 'json-summary', 'json-details', "
        "'text', 'text-summary', or 'text-details'",
        (int)value.size, value.data);
  }
  *out_enabled = true;
  return iree_ok_status();
}

static iree_status_t loom_compile_make_artifact_manifest_path(
    iree_string_view_t artifact_path, iree_allocator_t allocator,
    iree_string_view_t* out_path, char** out_path_storage) {
  *out_path = iree_string_view_empty();
  *out_path_storage = NULL;
  if (loom_tooling_output_path_is_stdout(artifact_path)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--artifact-manifest requires --emit-artifact-manifest when the "
        "selected artifact output writes to stdout");
  }
  iree_string_builder_t builder;
  iree_string_builder_initialize(allocator, &builder);
  iree_status_t status =
      iree_string_builder_append_string(&builder, artifact_path);
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(&builder, ".manifest.json");
  }
  if (iree_status_is_ok(status)) {
    const iree_host_size_t path_length = iree_string_builder_size(&builder);
    *out_path_storage = iree_string_builder_take_storage(&builder);
    *out_path = iree_make_string_view(*out_path_storage, path_length);
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

static iree_status_t loom_compile_parse_manifest_options(
    iree_allocator_t allocator, loomc_artifact_manifest_options_t* out_options,
    bool* out_enabled, iree_string_view_t* out_output_path,
    char** out_output_path_storage) {
  *out_options = (loomc_artifact_manifest_options_t){
      .type = LOOMC_STRUCTURE_TYPE_ARTIFACT_MANIFEST_OPTIONS,
      .structure_size = sizeof(*out_options),
  };
  *out_enabled = false;
  *out_output_path = iree_string_view_empty();
  *out_output_path_storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_status_from_loomc(loomc_artifact_manifest_mode_parse(
          loomc_make_cstring_view(FLAG_artifact_manifest),
          &out_options->mode)));
  const iree_string_view_t explicit_output_path =
      iree_make_cstring_view(FLAG_emit_artifact_manifest);
  if (out_options->mode == LOOMC_ARTIFACT_MANIFEST_MODE_NONE) {
    if (!iree_string_view_is_empty(explicit_output_path)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "--emit-artifact-manifest requires --artifact-manifest");
    }
    return iree_ok_status();
  }
  *out_enabled = true;
  if (iree_string_view_is_empty(explicit_output_path)) {
    IREE_RETURN_IF_ERROR(loom_compile_make_artifact_manifest_path(
        iree_make_cstring_view(FLAG_output), allocator, out_output_path,
        out_output_path_storage));
  } else {
    *out_output_path = explicit_output_path;
  }
  if (!loom_tooling_output_path_is_stdout(*out_output_path)) {
    out_options->identifier = loomc_string_view_from_iree(*out_output_path);
  }
  return iree_ok_status();
}

static iree_status_t loom_compile_parse_sanitizer_options(
    loomc_sanitizer_options_t* out_options, bool* out_enabled) {
  loom_sanitizer_options_t internal_options = {0};
  IREE_RETURN_IF_ERROR(loom_sanitizer_options_parse_checks(
      iree_make_cstring_view(FLAG_sanitizer), IREE_SV("--sanitizer"),
      &internal_options));
  IREE_RETURN_IF_ERROR(loom_sanitizer_reporting_mode_parse(
      iree_make_cstring_view(FLAG_sanitizer_reporting),
      IREE_SV("--sanitizer-reporting"), &internal_options.reporting_mode));
  loom_tooling_cli_make_loomc_sanitizer_options(&internal_options, out_options);
  *out_enabled =
      internal_options.checks != 0 ||
      internal_options.reporting_mode != LOOM_SANITIZER_REPORTING_MODE_DEFAULT;
  return iree_ok_status();
}

static iree_status_t loom_compile_append_config_flags(
    loom_config_text_binding_set_t* config_set) {
  const iree_flag_string_list_t assignments = FLAG_config_list();
  for (iree_host_size_t i = 0; i < assignments.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_tooling_config_text_binding_set_append_assignment(
        config_set, assignments.values[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_compile_append_config_files(
    loom_config_text_binding_set_t* config_set, iree_allocator_t allocator) {
  const iree_flag_string_list_t paths = FLAG_config_file_list();
  for (iree_host_size_t i = 0; i < paths.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_tooling_config_text_binding_set_append_json_file(
        config_set, paths.values[i], allocator));
  }
  return iree_ok_status();
}

static iree_status_t loom_compile_make_string_array(
    iree_string_view_list_t values, iree_allocator_t allocator,
    loomc_string_view_t** out_values) {
  *out_values = NULL;
  if (values.count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      allocator, values.count * sizeof(**out_values), (void**)out_values));
  for (iree_host_size_t i = 0; i < values.count; ++i) {
    (*out_values)[i] = loomc_string_view_from_iree(values.values[i]);
  }
  return iree_ok_status();
}

static iree_status_t loom_compile_load_source(
    int argc, char** argv,
    const loom_tooling_source_path_options_t* source_path_options,
    iree_allocator_t allocator, char** out_identifier_storage,
    loomc_source_t** out_source) {
  *out_identifier_storage = NULL;
  *out_source = NULL;
  if (argc > 2) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "loom-compile accepts at most one input file or '-' for stdin; got %d "
        "inputs",
        argc - 1);
  }
  const iree_string_view_t input_path = loom_compile_input_path(argc, argv);
  iree_string_view_t identifier = loom_compile_input_identifier(input_path);
  if (!loom_tooling_file_path_is_stdio(input_path)) {
    IREE_RETURN_IF_ERROR(loom_tooling_source_path_remap(
        identifier, source_path_options, allocator, &identifier,
        out_identifier_storage));
  }
  const loomc_source_load_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_SOURCE_LOAD_OPTIONS,
      .structure_size = sizeof(options),
      .format = LOOMC_SOURCE_FORMAT_UNKNOWN,
      .identifier = loomc_string_view_from_iree(identifier),
  };
  if (loom_tooling_file_path_is_stdio(input_path)) {
    return iree_status_from_loomc(loomc_source_create_from_file(
        stdin, &options, loomc_allocator_from_iree(allocator), out_source));
  }
  return iree_status_from_loomc(loomc_source_create_from_path(
      loomc_string_view_from_iree(input_path), &options,
      loomc_allocator_from_iree(allocator), out_source));
}

static loomc_status_t loom_compile_trace_write(void* user_data,
                                               loomc_string_view_t fragment) {
  return loomc_status_from_iree(loom_output_stream_write(
      (loom_output_stream_t*)user_data, iree_string_view_from_loomc(fragment)));
}

static loomc_status_t loom_compile_trace_open_artifact(
    void* user_data, const loomc_pass_trace_event_t* event,
    loomc_pass_trace_artifact_t* out_artifact) {
  iree_string_view_t point = iree_string_view_empty();
  switch (event->point) {
    case LOOMC_PASS_TRACE_POINT_BEFORE:
      point = IREE_SV("before");
      break;
    case LOOMC_PASS_TRACE_POINT_AFTER:
      point = IREE_SV("after");
      break;
    default:
      return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                               "pass trace point is invalid");
  }
  loom_output_stream_t* stream = NULL;
  iree_string_view_t reference = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      loomc_status_from_iree(loom_tooling_pass_trace_open_artifact(
          (loom_tooling_pass_trace_t*)user_data, event->event_ordinal, point,
          iree_string_view_from_loomc(event->pass_key), &stream, &reference)));
  *out_artifact = (loomc_pass_trace_artifact_t){
      .reference = loomc_string_view_from_iree(reference),
      .sink =
          {
              .write = loom_compile_trace_write,
              .user_data = stream,
          },
  };
  return loomc_ok_status();
}

static loomc_status_t loom_compile_trace_close_artifact(
    void* user_data, loomc_pass_trace_artifact_t* artifact) {
  (void)artifact;
  return loomc_status_from_iree(loom_tooling_pass_trace_close_artifact(
      (loom_tooling_pass_trace_t*)user_data));
}

static void loom_compile_initialize_trace_options(
    loom_tooling_pass_trace_t* trace, const loomc_string_view_t* before_filters,
    const loomc_string_view_t* after_filters,
    loomc_pass_trace_options_t* out_options) {
  *out_options = (loomc_pass_trace_options_t){
      .type = LOOMC_STRUCTURE_TYPE_PASS_TRACE_OPTIONS,
      .structure_size = sizeof(*out_options),
      .format = trace->format == LOOM_TOOLING_PASS_TRACE_FORMAT_JSONL
                    ? LOOMC_PASS_TRACE_FORMAT_JSONL
                    : LOOMC_PASS_TRACE_FORMAT_TEXT,
      .flags = (trace->dump_before_all ? LOOMC_PASS_TRACE_FLAG_BEFORE_ALL : 0) |
               (trace->dump_after_all ? LOOMC_PASS_TRACE_FLAG_AFTER_ALL : 0),
      .tool_name = loomc_string_view_from_iree(trace->tool_name),
      .input_identifier = loomc_string_view_from_iree(trace->input_path),
      .before_filters = before_filters,
      .before_filter_count = trace->dump_before.count,
      .after_filters = after_filters,
      .after_filter_count = trace->dump_after.count,
      .sink =
          {
              .write = loom_compile_trace_write,
              .user_data = &trace->output.stream,
          },
  };
  if (loom_tooling_pass_trace_has_artifact_sink(trace)) {
    out_options->artifact_sink = (loomc_pass_trace_artifact_sink_t){
        .open = loom_compile_trace_open_artifact,
        .close = loom_compile_trace_close_artifact,
        .user_data = trace,
    };
  }
}

typedef struct loom_compile_artifacts_t {
  // Primary target executable or loadable module.
  const loomc_artifact_t* primary;
  // Optional artifact manifest sidecar.
  const loomc_artifact_t* manifest;
  // Optional compile report sidecar.
  const loomc_artifact_t* report;
} loom_compile_artifacts_t;

static iree_status_t loom_compile_select_artifacts(
    const loomc_result_t* result, loom_compile_artifacts_t* out_artifacts) {
  *out_artifacts = (loom_compile_artifacts_t){0};
  for (loomc_host_size_t i = 0; i < loomc_result_artifact_count(result); ++i) {
    const loomc_artifact_t* artifact = loomc_result_artifact_at(result, i);
    const bool is_manifest = loomc_string_view_equal(
        artifact->format,
        loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_ARTIFACT_MANIFEST_JSON));
    const bool is_report =
        loomc_string_view_equal(
            artifact->format, loomc_make_cstring_view(
                                  LOOMC_ARTIFACT_FORMAT_COMPILE_REPORT_JSON)) ||
        loomc_string_view_equal(
            artifact->format,
            loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_COMPILE_REPORT_TEXT));
    const loomc_artifact_t** slot = NULL;
    if (is_manifest) {
      slot = &out_artifacts->manifest;
    } else if (is_report) {
      slot = &out_artifacts->report;
    } else if (artifact->kind == LOOMC_ARTIFACT_KIND_EXECUTABLE) {
      slot = &out_artifacts->primary;
    }
    if (slot && *slot) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "compiler returned duplicate output artifacts");
    }
    if (slot) {
      *slot = artifact;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_compile_write_artifact(
    const loomc_artifact_t* artifact, iree_string_view_t output_path,
    iree_allocator_t allocator) {
  if (!artifact) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "compiler did not return a requested artifact");
  }
  FILE* file = NULL;
  if (iree_string_view_equal(output_path, IREE_SV("stderr"))) {
    file = stderr;
  } else if (loom_tooling_output_path_is_stdout(output_path)) {
    file = stdout;
  }
  if (!file) {
    return iree_status_from_loomc(loomc_artifact_write_to_path(
        artifact, loomc_string_view_from_iree(output_path),
        loomc_allocator_from_iree(allocator)));
  }
  IREE_RETURN_IF_ERROR(
      iree_status_from_loomc(loomc_artifact_write_to_file(artifact, file)));
  if (fflush(file) != 0) {
    return iree_make_status(IREE_STATUS_UNKNOWN,
                            "failed to flush artifact output");
  }
  return iree_ok_status();
}

static void loom_compile_print_agents_markdown(FILE* stream) {
  static const char* const kLines[] = {
      "## loom-compile",
      "",
      "`loom-compile` specializes Loom text or bytecode and emits an offline "
      "artifact.",
      "Run `loom-compile --help` for the complete command-line reference.",
      "",
      "### Common commands",
      "",
      "```shell",
      "# Compile a targetless kernel for a portable GFX11 profile.",
      "loom-compile kernel.loom \\",
      "  --target=amdgpu:gfx11-generic --output=kernel.hsaco",
      "",
      "# Select one catalog root, exact target, and compile-time value.",
      "loom-compile catalog.loombc --root=@entry \\",
      "  --target=amdgpu:gfx1151 --config=model.hidden_size=4096 \\",
      "  --output=entry.hsaco",
      "",
      "# Emit SPIR-V or WebAssembly from source that declares its target.",
      "loom-compile kernel.loom --format=spirv --output=kernel.spv",
      "loom-compile functions.loom --format=wasm-binary "
      "--output=functions.wasm",
      "```",
      "",
      "### Inspect a compilation",
      "",
      "```shell",
      "loom-compile kernel.loom \\",
      "  --target=amdgpu:gfx11-generic --output=kernel.hsaco \\",
      "  --artifact-manifest=summary \\",
      "  --emit-artifact-manifest=manifest.json \\",
      "  --compile-report=summary --compile-report-output=report.json",
      "loom-compile-report show report.json",
      "loom-compile-report diff baseline.json report.json",
      "loom-compile-report suggest report.json",
      "```",
      "",
      "### Documentation",
      "",
      "- [From source to artifacts](https://rocm.github.io/hrx-system/loom/"
      "getting-started/source-to-artifacts/)",
      "- [Compile artifacts](https://rocm.github.io/hrx-system/loom/workflows/"
      "compile-artifacts/)",
      "- [Read compile reports](https://rocm.github.io/hrx-system/loom/"
      "workflows/compile-reports/)",
      "- [Tune loop schedules](https://rocm.github.io/hrx-system/loom/"
      "workflows/tune-loop-schedules/)",
      "- [Workflow index](https://rocm.github.io/hrx-system/loom/workflows/)",
      "- [Language guide](https://rocm.github.io/hrx-system/loom/guide/)",
  };
  _Static_assert(IREE_ARRAYSIZE(kLines) <= 100,
                 "--agents_md must remain at most 100 lines");
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kLines); ++i) {
    fprintf(stream, "%s\n", kLines[i]);
  }
}

int main(int argc, char** argv) {
  iree_flags_set_usage(
      "loom-compile",
      "Compiles a Loom module to a runtime artifact.\n"
      "\n"
      "Usage:\n"
      "  loom-compile [file.loom] --format=amdgpu-hsaco "
      "--target=amdgpu:gfx11-generic --output=kernel.hsaco\n"
      "  loom-compile --agents_md\n"
      "\n"
      "Repeat --config=key=value to materialize compile-time config symbols "
      "before the pass pipeline. Use --config-file=path for a JSON/JSONC "
      "object such as {\"model36\":{\"model\":{\"hidden_size\":4096}}}. "
      "Files and direct bindings share one config set and duplicate keys are "
      "rejected.\n"
      "Use --agents_md to print common commands and documentation "
      "links.\n" LOOM_TOOLING_PASS_TRACE_USAGE);
  for (int i = 1; i < argc; ++i) {
    if (loom_tooling_cli_is_agents_markdown_arg(argv[i])) {
      loom_compile_print_agents_markdown(stdout);
      return 0;
    }
  }
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);

  loom_tooling_cli_set_default_help_filter();
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  const iree_allocator_t allocator = iree_allocator_system();
  const loomc_allocator_t loom_allocator = loomc_allocator_from_iree(allocator);
  const loom_tooling_source_path_options_t source_path_options = {
      .prefix_maps = FLAG_source_prefix_map_list(),
  };
  iree_status_t status = iree_ok_status();
  int exit_code = 0;

  loom_config_text_binding_set_t config_set;
  loom_config_text_binding_set_initialize(allocator, &config_set);
  loomc_config_binding_t* config_bindings = NULL;
  loomc_config_options_t config_options = {0};
  loomc_string_view_t* roots = NULL;
  loomc_string_view_t* excluded_roots = NULL;
  loomc_string_view_t* before_filters = NULL;
  loomc_string_view_t* after_filters = NULL;
  char* input_identifier_storage = NULL;
  char* manifest_output_path_storage = NULL;
  iree_string_view_t manifest_output_path = iree_string_view_empty();

  loomc_target_environment_t* target_environment = NULL;
  loomc_context_t* context = NULL;
  loomc_workspace_t* workspace = NULL;
  loomc_compiler_t* compiler = NULL;
  loomc_source_t* source = NULL;
  loomc_module_t* module = NULL;
  loomc_target_profile_t* target_profile = NULL;
  loomc_pass_program_t* pass_program = NULL;
  loomc_result_t* result = NULL;
  loom_tooling_pass_trace_t pass_trace = {0};

  loomc_compile_report_options_t report_options = {0};
  loomc_artifact_manifest_options_t manifest_options = {0};
  loomc_sanitizer_options_t sanitizer_options = {0};
  loomc_pass_trace_options_t trace_options = {0};
  bool report_enabled = false;
  bool manifest_enabled = false;
  bool sanitizer_enabled = false;

  if (iree_status_is_ok(status)) {
    status =
        loom_compile_parse_report_options(&report_options, &report_enabled);
  }
  if (iree_status_is_ok(status)) {
    status = loom_compile_parse_manifest_options(
        allocator, &manifest_options, &manifest_enabled, &manifest_output_path,
        &manifest_output_path_storage);
  }
  if (iree_status_is_ok(status)) {
    status = loom_compile_parse_sanitizer_options(&sanitizer_options,
                                                  &sanitizer_enabled);
  }
  const iree_string_view_t pipeline =
      iree_string_view_trim(iree_make_cstring_view(FLAG_pipeline));
  if (iree_status_is_ok(status) && sanitizer_enabled &&
      !loom_tooling_cli_pipeline_uses_default(pipeline)) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--sanitizer and --sanitizer-reporting require --pipeline=default");
  }
  if (iree_status_is_ok(status)) {
    const loom_tooling_output_path_t output_paths[] = {
        {
            .active = true,
            .flag_name = IREE_SV("--output"),
            .path = iree_make_cstring_view(FLAG_output),
        },
        {
            .active = manifest_enabled,
            .flag_name = IREE_SV("--emit-artifact-manifest"),
            .path = manifest_output_path,
        },
        {
            .active = report_enabled,
            .flag_name = IREE_SV("--compile-report-output"),
            .path = iree_make_cstring_view(FLAG_compile_report_output),
        },
        loom_tooling_pass_trace_output_path_from_flags(),
    };
    status = loom_tooling_output_paths_validate_exclusive_stdout(
        output_paths, IREE_ARRAYSIZE(output_paths));
  }
  if (iree_status_is_ok(status)) {
    status = loom_compile_append_config_files(&config_set, allocator);
  }
  if (iree_status_is_ok(status)) {
    status = loom_compile_append_config_flags(&config_set);
  }
  if (iree_status_is_ok(status)) {
    status = loom_tooling_cli_make_loomc_config_options(
        &config_set, allocator, &config_bindings, &config_options);
  }
  const iree_flag_string_list_t root_list = FLAG_root_list();
  const iree_flag_string_list_t excluded_root_list = FLAG_exclude_root_list();
  if (iree_status_is_ok(status)) {
    status = loom_compile_make_string_array(root_list, allocator, &roots);
  }
  if (iree_status_is_ok(status)) {
    status = loom_compile_make_string_array(excluded_root_list, allocator,
                                            &excluded_roots);
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_target_environment_create_configured(
        loom_allocator, &target_environment));
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
        loomc_context_create(&context_options, loom_allocator, &context));
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(
        loomc_workspace_create(NULL, loom_allocator, &workspace));
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(
        loomc_compiler_create(context, NULL, loom_allocator, &compiler));
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_compile_load_source(argc, argv, &source_path_options, allocator,
                                 &input_identifier_storage, &source);
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_module_deserialize_from_source(
        context, workspace, source, NULL, loom_allocator, &module, &result));
  }
  if (iree_status_is_ok(status)) {
    bool parse_succeeded = false;
    status =
        loom_tooling_cli_print_loomc_result(stderr, result, &parse_succeeded);
    if (iree_status_is_ok(status) && !parse_succeeded) {
      exit_code = 1;
    }
    loomc_result_release(result);
    result = NULL;
  }
  const iree_string_view_t target =
      iree_string_view_trim(iree_make_cstring_view(FLAG_target));
  if (iree_status_is_ok(status) && exit_code == 0 &&
      !iree_string_view_is_empty(target)) {
    status = iree_status_from_loomc(loomc_target_profile_select(
        target_environment, loomc_string_view_from_iree(target), loom_allocator,
        &target_profile));
  }
  if (iree_status_is_ok(status) && exit_code == 0) {
    status = loom_tooling_cli_prepare_loomc_pass_program(
        context, module, iree_make_cstring_view(FLAG_pipeline),
        IREE_SV("loom-compile pipeline"), &pass_program, &result, allocator);
  }
  if (iree_status_is_ok(status) && result) {
    bool preparation_succeeded = false;
    status = loom_tooling_cli_print_loomc_result(stderr, result,
                                                 &preparation_succeeded);
    if (iree_status_is_ok(status) && !preparation_succeeded) {
      exit_code = 1;
    }
    loomc_result_release(result);
    result = NULL;
  }
  if (iree_status_is_ok(status) && exit_code == 0) {
    status = loom_tooling_pass_trace_open_from_flags(
        &(loom_tooling_pass_trace_open_options_t){
            .tool_name = IREE_SV("loom-compile"),
            .input_path =
                iree_string_view_from_loomc(loomc_source_identifier(source)),
        },
        allocator, &pass_trace);
  }
  if (iree_status_is_ok(status) && exit_code == 0 && pass_trace.enabled) {
    status = loom_compile_make_string_array(pass_trace.dump_before, allocator,
                                            &before_filters);
  }
  if (iree_status_is_ok(status) && exit_code == 0 && pass_trace.enabled) {
    status = loom_compile_make_string_array(pass_trace.dump_after, allocator,
                                            &after_filters);
  }
  if (iree_status_is_ok(status) && exit_code == 0 && pass_trace.enabled) {
    loom_compile_initialize_trace_options(&pass_trace, before_filters,
                                          after_filters, &trace_options);
  }

  if (report_enabled) {
    const iree_string_view_t report_output_path =
        iree_make_cstring_view(FLAG_compile_report_output);
    if (!loom_tooling_output_path_is_stdout(report_output_path) &&
        !iree_string_view_equal(report_output_path, IREE_SV("stderr"))) {
      report_options.identifier =
          loomc_string_view_from_iree(report_output_path);
    }
  }
  manifest_options.next = report_enabled ? &report_options : NULL;
  loomc_emit_options_t emit_options = {
      .type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
      .structure_size = sizeof(emit_options),
      .next = manifest_enabled ? (const void*)&manifest_options
              : report_enabled ? (const void*)&report_options
                               : NULL,
      .artifact_format = loomc_make_cstring_view(FLAG_format),
      .artifact_flags = LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY,
  };
  const iree_string_view_t output_path = iree_make_cstring_view(FLAG_output);
  if (!loom_tooling_output_path_is_stdout(output_path)) {
    emit_options.identifier = loomc_string_view_from_iree(output_path);
  }
  trace_options.next = sanitizer_enabled ? &sanitizer_options : NULL;
  loomc_compile_artifact_options_t compile_options = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_ARTIFACT_OPTIONS,
      .structure_size = sizeof(compile_options),
      .next = pass_trace.enabled  ? (const void*)&trace_options
              : sanitizer_enabled ? (const void*)&sanitizer_options
                                  : NULL,
      .roots = roots,
      .root_count = root_list.count,
      .excluded_roots = excluded_roots,
      .excluded_root_count = excluded_root_list.count,
      .target_profile = target_profile,
      .config = &config_options,
      .emit_options = &emit_options,
  };
  if (iree_status_is_ok(status) && exit_code == 0) {
    status = iree_status_from_loomc(
        loomc_compile_artifact(compiler, workspace, pass_program, module,
                               &compile_options, loom_allocator, &result));
  }
  status = iree_status_join(status, loom_tooling_pass_trace_close(&pass_trace));

  if (iree_status_is_ok(status) && exit_code == 0) {
    bool compile_succeeded = false;
    status =
        loom_tooling_cli_print_loomc_result(stderr, result, &compile_succeeded);
    loom_compile_artifacts_t artifacts = {0};
    if (iree_status_is_ok(status)) {
      status = loom_compile_select_artifacts(result, &artifacts);
    }
    if (iree_status_is_ok(status) && report_enabled) {
      status = loom_compile_write_artifact(
          artifacts.report, iree_make_cstring_view(FLAG_compile_report_output),
          allocator);
    }
    if (iree_status_is_ok(status) && compile_succeeded) {
      status = loom_compile_write_artifact(artifacts.primary, output_path,
                                           allocator);
    }
    if (iree_status_is_ok(status) && compile_succeeded && manifest_enabled) {
      status = loom_compile_write_artifact(artifacts.manifest,
                                           manifest_output_path, allocator);
    }
    if (!compile_succeeded) {
      exit_code = 1;
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = 1;
  }

  loomc_result_release(result);
  loomc_pass_program_release(pass_program);
  loomc_target_profile_release(target_profile);
  loomc_module_release(module);
  loomc_source_release(source);
  loomc_compiler_release(compiler);
  loomc_workspace_release(workspace);
  loomc_context_release(context);
  loomc_target_environment_release(target_environment);
  iree_allocator_free(allocator, after_filters);
  iree_allocator_free(allocator, before_filters);
  iree_allocator_free(allocator, excluded_roots);
  iree_allocator_free(allocator, roots);
  iree_allocator_free(allocator, config_bindings);
  loom_config_text_binding_set_deinitialize(&config_set);
  iree_allocator_free(allocator, manifest_output_path_storage);
  iree_allocator_free(allocator, input_identifier_storage);

  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
