// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// loom-compile: compiles a Loom module to a runtime artifact.

#include <stdio.h>

#include "iree/base/api.h"
#include "iree/base/byte_sequence.h"
#include "iree/base/tooling/flags.h"
#include "iree/io/file_contents.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/error/diagnostic.h"
#include "loom/format/text/printer.h"
#include "loom/ir/module.h"
#include "loom/link/linker.h"
#include "loom/ops/op_defs.h"
#include "loom/sanitizer/options.h"
#include "loom/target/arch/cmd/artifact_set.h"
#include "loom/target/entry_selection.h"
#include "loom/target/module_specialization.h"
#include "loom/target/reporting/artifact_manifest.h"
#include "loom/tooling/cli/help.h"
#include "loom/tooling/compile/artifact.h"
#include "loom/tooling/compile/configured.h"
#include "loom/tooling/compile/pipeline.h"
#include "loom/tooling/compile/report_capture.h"
#include "loom/tooling/compile/request.h"
#include "loom/tooling/compile/request_flags.h"
#include "loom/tooling/config/config.h"
#include "loom/tooling/context/context.h"
#include "loom/tooling/execution/session.h"
#include "loom/tooling/io/file.h"
#include "loom/tooling/io/source_path.h"
#include "loom/tooling/pass/trace_cli.h"
#include "loom/tools/loom-compile/command_backend.h"
#include "loom/tools/loom-compile/command_manifest.h"

typedef struct loom_compile_diagnostic_sink_t {
  // Parsed module used for full type rendering.
  const loom_run_module_t* run_module;
  // Printer context used to render target-owned register and storage types.
  loom_low_descriptor_text_print_context_t type_print_context;
  // Optional compile report capture receiving canonical diagnostic JSON.
  loom_compile_report_capture_t* compile_report_capture;
} loom_compile_diagnostic_sink_t;

static iree_status_t loom_compile_format_diagnostic_type(
    loom_type_t type, void* user_data, loom_output_stream_t* stream) {
  const loom_compile_diagnostic_sink_t* sink =
      (const loom_compile_diagnostic_sink_t*)user_data;
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

static iree_status_t loom_compile_diagnostic_sink(
    void* user_data, const loom_diagnostic_t* diagnostic) {
  loom_compile_diagnostic_sink_t* sink =
      (loom_compile_diagnostic_sink_t*)user_data;
  loom_output_stream_t stream;
  loom_output_stream_for_file(stderr, &stream);
  const loom_diagnostic_format_options_t format_options = {
      .type_formatter =
          {
              .fn = loom_compile_format_diagnostic_type,
              .user_data = user_data,
          },
  };
  iree_status_t status =
      loom_diagnostic_format_with_options(diagnostic, &format_options, &stream);
  if (iree_status_is_ok(status)) {
    loom_compile_report_capture_t* compile_report_capture =
        sink ? sink->compile_report_capture : NULL;
    status = loom_compile_report_capture_record_diagnostic(
        compile_report_capture, diagnostic, format_options.type_formatter);
  }
  return status;
}

IREE_FLAG_LIST(string, root,
               "Root symbol to materialize before compilation. Repeat for "
               "multiple roots. Roots must infer one homogeneous product. "
               "When omitted, --product selects its canonical roots; without "
               "either, public or retained command programs take precedence, "
               "then kernel entries, then the whole module.");
IREE_FLAG(string, pipeline, "default",
          "Pass pipeline to run before artifact emission. Use 'default' or "
          "empty for the selected format's default compile pipeline. 'none' "
          "disables "
          "all compiler transformations and passes the input directly to the "
          "selected emitter. Use '@symbol' to run a module-local pass.pipeline "
          "or a comma-separated pass list such as "
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
          "Output path for the selected format. For 'loom-command' this is "
          "the command artifact-set manifest; single-file kernel and module "
          "formats write their artifact directly.");
IREE_FLAG_NAMED(
    string, emit_target_artifact, "emit-target-artifact", "",
    "Optional output path for a target-native artifact produced beside the "
    "primary runtime artifact, such as AMDGPU HSACO.");
IREE_FLAG_NAMED(
    string, emit_command_artifacts, "emit-command-artifacts", "",
    "Directory receiving portable roots for the 'loom-command' format. The "
    "primary --output is the artifact-set manifest.");
IREE_FLAG_NAMED(
    string, emit_kernel_requests, "emit-kernel-requests", "",
    "Optional directory receiving independently compilable Loom bytecode "
    "kernel requests for the 'loom-command' format.");
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
                "Output path for --compile-report. Use 'stderr', '-', or a "
                "file path.");
IREE_FLAG_LIST_NAMED(
    string, source_prefix_map, "source-prefix-map",
    "Remap source paths in diagnostics and serialized locations. Repeat as\n"
    "--source-prefix-map=old=new; entries are applied in reverse order so the\n"
    "last matching map wins. Use old= to strip a prefix.");

static iree_status_t loom_compile_register_context(void* user_data,
                                                   loom_context_t* context) {
  const loom_target_environment_t* target_environment =
      (const loom_target_environment_t*)user_data;
  return loom_tooling_context_register_tool_dialects_with_target_environment(
      target_environment, context);
}

static iree_status_t loom_compile_initialize_low_descriptor_registry(
    void* user_data, loom_target_low_descriptor_registry_t* out_registry) {
  const loom_target_environment_t* target_environment =
      (const loom_target_environment_t*)user_data;
  return loom_target_environment_initialize_low_descriptor_registry(
      target_environment, out_registry);
}

static iree_status_t loom_compile_initialize_session(
    const loom_target_environment_t* target_environment,
    iree_allocator_t allocator, loom_run_session_t* out_session) {
  loom_run_session_options_t session_options = {0};
  loom_run_session_options_initialize(&session_options);
  session_options.host_allocator = allocator;
  session_options.register_context = (loom_run_register_context_callback_t){
      .fn = loom_compile_register_context,
      .user_data = (void*)target_environment,
  };
  session_options.initialize_low_descriptor_registry =
      (loom_run_initialize_low_descriptor_registry_callback_t){
          .fn = loom_compile_initialize_low_descriptor_registry,
          .user_data = (void*)target_environment,
      };
  return loom_run_session_initialize(&session_options, out_session);
}

static iree_string_view_t loom_compile_input_path(int argc, char** argv) {
  return argc < 2 ? iree_string_view_empty() : iree_make_cstring_view(argv[1]);
}

static iree_string_view_t loom_compile_input_filename(
    iree_string_view_t input_path) {
  return (iree_string_view_is_empty(input_path) ||
          iree_string_view_equal(input_path, IREE_SV("-")))
             ? IREE_SV("<stdin>")
             : input_path;
}

static iree_status_t loom_compile_parse_input_module(
    int argc, char** argv, loom_run_session_t* session,
    const loom_tooling_source_path_options_t* source_path_options,
    iree_allocator_t allocator, iree_io_file_contents_t** out_contents,
    char** out_filename_storage, loom_run_module_t* out_run_module) {
  *out_filename_storage = NULL;
  if (argc > 2) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "loom-compile accepts at most one input file or '-' for stdin; got %d "
        "inputs",
        argc - 1);
  }

  const iree_string_view_t input_path = loom_compile_input_path(argc, argv);
  iree_string_view_t filename = loom_compile_input_filename(input_path);
  IREE_RETURN_IF_ERROR(
      loom_tooling_read_input_file(input_path, allocator, out_contents));
  if (!loom_tooling_file_path_is_stdio(input_path)) {
    IREE_RETURN_IF_ERROR(
        loom_tooling_source_path_remap(filename, source_path_options, allocator,
                                       &filename, out_filename_storage));
  }

  loom_run_module_parse_options_t parse_options = {0};
  loom_run_module_parse_options_initialize(&parse_options);
  parse_options.filename = filename;
  parse_options.source = loom_tooling_file_contents_string_view(*out_contents);
  return loom_run_module_parse(session, &parse_options, out_run_module);
}

static iree_status_t loom_compile_append_config_flags(
    loom_tooling_config_set_t* config_set) {
  iree_flag_string_list_t assignments = FLAG_config_list();
  for (iree_host_size_t i = 0; i < assignments.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_tooling_config_set_append_assignment(
        config_set, assignments.values[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_compile_append_config_files(
    loom_tooling_config_set_t* config_set, iree_allocator_t allocator) {
  iree_flag_string_list_t paths = FLAG_config_file_list();
  for (iree_host_size_t i = 0; i < paths.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_tooling_config_set_append_json_file(
        config_set, paths.values[i], allocator));
  }
  return iree_ok_status();
}

static iree_status_t loom_compile_materialize_config_set(
    loom_run_session_t* session, loom_run_module_t* run_module,
    const loom_tooling_config_set_t* config_set) {
  loom_tooling_config_materialize_options_t options;
  loom_tooling_config_materialize_options_initialize(&options);
  options.config_set = config_set;
  return loom_tooling_config_materialize_module(
      run_module->module, &options, loom_run_session_block_pool(session), NULL);
}

static bool loom_compile_is_concrete_kernel_root(const loom_symbol_t* symbol) {
  return symbol->defining_op != NULL &&
         loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_KERNEL_ENTRY) &&
         !loom_symbol_definition_is_declaration(symbol->definition);
}

static iree_status_t loom_compile_select_roots(
    loom_run_session_t* session, loom_run_module_t* run_module,
    const loom_compile_request_t* request, iree_allocator_t allocator) {
  iree_string_view_list_t roots = request->roots;
  iree_string_view_t* implicit_root_values = NULL;
  if (roots.count == 0 && request->product == LOOM_COMPILE_PRODUCT_KERNEL) {
    for (iree_host_size_t i = 0; i < run_module->module->symbols.count; ++i) {
      const loom_symbol_t* symbol = &run_module->module->symbols.entries[i];
      if (loom_compile_is_concrete_kernel_root(symbol)) {
        ++roots.count;
      }
    }
    if (roots.count == 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "kernel product requires at least one kernel entry");
    }
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        allocator, roots.count, sizeof(*implicit_root_values),
        (void**)&implicit_root_values));
    iree_host_size_t root_ordinal = 0;
    for (iree_host_size_t i = 0; i < run_module->module->symbols.count; ++i) {
      const loom_symbol_t* symbol = &run_module->module->symbols.entries[i];
      if (!loom_compile_is_concrete_kernel_root(symbol)) {
        continue;
      }
      implicit_root_values[root_ordinal++] =
          run_module->module->strings.entries[symbol->name_id];
    }
    roots.values = implicit_root_values;
  }
  if (roots.count == 0) {
    return iree_ok_status();
  }

  const loom_module_t* const source_modules[] = {run_module->module};
  iree_string_view_t module_name = iree_string_view_empty();
  if (run_module->module->name_id < run_module->module->strings.count) {
    module_name =
        run_module->module->strings.entries[run_module->module->name_id];
  }
  loom_module_t* linked_module = NULL;
  iree_status_t status = loom_link_materialized_modules(
      source_modules, IREE_ARRAYSIZE(source_modules),
      &(loom_link_options_t){
          .module_name = module_name,
          .root_symbols = roots,
      },
      loom_run_session_block_pool(session), allocator, &linked_module);
  iree_allocator_free(allocator, implicit_root_values);
  if (!iree_status_is_ok(status)) {
    return status;
  }

  loom_module_free(run_module->module);
  run_module->module = linked_module;
  return iree_ok_status();
}

static iree_status_t loom_compile_report_options_initialize(
    loom_compile_report_capture_options_t* out_options) {
  loom_compile_report_capture_options_initialize(out_options);
  IREE_RETURN_IF_ERROR(loom_compile_report_capture_options_parse_request(
      iree_make_cstring_view(FLAG_compile_report), out_options));
  return iree_ok_status();
}

static iree_status_t loom_compile_sanitizer_options_initialize(
    loom_sanitizer_options_t* out_options) {
  IREE_RETURN_IF_ERROR(loom_sanitizer_options_parse_checks(
      iree_make_cstring_view(FLAG_sanitizer), IREE_SV("--sanitizer"),
      out_options));
  return loom_sanitizer_reporting_mode_parse(
      iree_make_cstring_view(FLAG_sanitizer_reporting),
      IREE_SV("--sanitizer-reporting"), &out_options->reporting_mode);
}

static iree_status_t loom_compile_make_artifact_manifest_path(
    iree_string_view_t artifact_path, iree_allocator_t allocator,
    iree_string_view_t* out_path, char** out_path_storage) {
  *out_path = iree_string_view_empty();
  *out_path_storage = NULL;
  if (loom_tooling_file_path_is_stdio(artifact_path)) {
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
    char* path_storage = iree_string_builder_take_storage(&builder);
    *out_path = iree_make_string_view(path_storage, path_length);
    *out_path_storage = path_storage;
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

static iree_status_t loom_compile_artifact_manifest_options_initialize(
    loom_compile_artifact_manifest_options_t* out_options,
    bool has_artifact_provider, iree_allocator_t allocator,
    iree_string_view_t* out_output_path, char** out_output_path_storage) {
  *out_options = (loom_compile_artifact_manifest_options_t){0};
  *out_output_path = iree_string_view_empty();
  *out_output_path_storage = NULL;
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_mode_parse(
      iree_make_cstring_view(FLAG_artifact_manifest), &out_options->mode));

  const iree_string_view_t explicit_output_path =
      iree_make_cstring_view(FLAG_emit_artifact_manifest);
  if (out_options->mode == LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE) {
    if (!iree_string_view_is_empty(explicit_output_path)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "--emit-artifact-manifest requires --artifact-manifest");
    }
    return iree_ok_status();
  }
  if (!has_artifact_provider) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--artifact-manifest is only valid for loadable kernel formats");
  }

  const iree_string_view_t target_artifact_path =
      iree_make_cstring_view(FLAG_emit_target_artifact);
  const iree_string_view_t primary_artifact_path =
      iree_make_cstring_view(FLAG_output);
  out_options->artifact_name = !iree_string_view_is_empty(target_artifact_path)
                                   ? target_artifact_path
                                   : primary_artifact_path;
  if (iree_string_view_is_empty(explicit_output_path)) {
    IREE_RETURN_IF_ERROR(loom_compile_make_artifact_manifest_path(
        out_options->artifact_name, allocator, out_output_path,
        out_output_path_storage));
  } else {
    *out_output_path = explicit_output_path;
  }
  out_options->identifier = *out_output_path;
  return iree_ok_status();
}

static iree_status_t loom_compile_run_pass_pipeline(
    const loom_target_environment_t* target_environment,
    loom_run_session_t* session, loom_run_module_t* run_module,
    loom_compile_default_pipeline_t default_pipeline,
    const loom_compile_options_t* compile_options,
    loom_compile_report_capture_t* compile_report_capture,
    const loom_pass_trace_options_t* trace_options,
    loom_compile_pipeline_result_t* out_result) {
  loom_compile_pipeline_options_t pipeline_options = {0};
  loom_compile_pipeline_options_initialize(&pipeline_options);
  pipeline_options.pipeline = iree_make_cstring_view(FLAG_pipeline);
  pipeline_options.default_pipeline = default_pipeline;
  pipeline_options.target_pipeline_options =
      compile_options->target_pipeline_options;
  pipeline_options.target_environment = target_environment;
  pipeline_options.low_descriptor_registry =
      loom_run_session_low_descriptor_registry(session);
  loom_compile_diagnostic_sink_t diagnostic_sink = {
      .run_module = run_module,
      .compile_report_capture = compile_report_capture,
  };
  loom_low_descriptor_text_print_context_initialize(
      &pipeline_options.low_descriptor_registry->registry,
      &diagnostic_sink.type_print_context);
  pipeline_options.diagnostic_sink = (loom_diagnostic_sink_t){
      .fn = loom_compile_diagnostic_sink,
      .user_data = &diagnostic_sink,
  };
  pipeline_options.source_resolver =
      loom_run_module_source_resolver(run_module);
  pipeline_options.max_errors = compile_options->max_errors;
  pipeline_options.report = compile_options->report;
  pipeline_options.trace_options = trace_options;

  return loom_compile_run_pipeline(run_module->module, &pipeline_options,
                                   loom_run_session_block_pool(session),
                                   out_result);
}

static iree_status_t loom_compile_write_bytes(iree_string_view_t path,
                                              iree_const_byte_span_t contents,
                                              iree_allocator_t allocator) {
  return loom_tooling_write_output_file(
      path,
      iree_make_string_view((const char*)contents.data, contents.data_length),
      allocator);
}

static iree_status_t loom_compile_write_optional_target_artifact(
    const loom_artifact_t* artifact, iree_allocator_t allocator) {
  const iree_string_view_t path =
      iree_make_cstring_view(FLAG_emit_target_artifact);
  if (iree_string_view_is_empty(path)) {
    return iree_ok_status();
  }
  if (artifact->target_artifact_data == NULL ||
      iree_byte_sequence_length(artifact->target_artifact_data) == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "selected artifact provider produced no "
                            "target-native artifact");
  }
  return loom_tooling_write_output_byte_sequence(
      path, artifact->target_artifact_data, allocator);
}

static iree_status_t loom_compile_write_optional_artifact_manifest(
    const loom_artifact_t* artifact, iree_string_view_t path,
    iree_allocator_t allocator) {
  if (iree_string_view_is_empty(path)) {
    return iree_ok_status();
  }
  const loom_target_emit_sidecar_artifact_t* manifest = NULL;
  for (iree_host_size_t i = 0; i < artifact->sidecar_count; ++i) {
    const loom_target_emit_sidecar_artifact_t* sidecar = &artifact->sidecars[i];
    if (sidecar->kind !=
        LOOM_TARGET_EMIT_SIDECAR_ARTIFACT_KIND_ARTIFACT_MANIFEST) {
      continue;
    }
    if (manifest != NULL) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "selected artifact provider produced multiple artifact "
          "manifests");
    }
    manifest = sidecar;
  }
  if (manifest == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "selected artifact provider produced no "
                            "artifact manifest");
  }
  return loom_tooling_write_output_byte_sequence(path, manifest->contents,
                                                 allocator);
}

static iree_status_t loom_compile_write_report(
    const loom_compile_report_capture_t* compile_report_capture,
    iree_string_view_t artifact_manifest_output_path,
    iree_allocator_t allocator) {
  if (!loom_compile_report_capture_is_enabled(compile_report_capture)) {
    return iree_ok_status();
  }
  const iree_string_view_t path =
      iree_make_cstring_view(FLAG_compile_report_output);
  if (iree_string_view_is_empty(path) ||
      iree_string_view_equal(path, IREE_SV("stderr"))) {
    loom_output_stream_t stream;
    loom_output_stream_for_file(stderr, &stream);
    IREE_RETURN_IF_ERROR(loom_compile_report_capture_write_output(
        compile_report_capture, &stream, allocator));
    return fflush(stderr) == 0
               ? iree_ok_status()
               : iree_make_status(IREE_STATUS_DATA_LOSS,
                                  "failed to flush compile report stderr");
  }
  if (loom_tooling_file_path_is_stdio(path)) {
    const iree_string_view_t target_artifact_path =
        iree_make_cstring_view(FLAG_emit_target_artifact);
    if (loom_tooling_file_path_is_stdio(iree_make_cstring_view(FLAG_output)) ||
        (!iree_string_view_is_empty(target_artifact_path) &&
         loom_tooling_file_path_is_stdio(target_artifact_path)) ||
        (!iree_string_view_is_empty(artifact_manifest_output_path) &&
         loom_tooling_file_path_is_stdio(artifact_manifest_output_path))) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "--compile-report-output=- cannot share stdout with --output=-, "
          "--emit-target-artifact=-, or --emit-artifact-manifest=-");
    }
    loom_output_stream_t stream;
    loom_output_stream_for_file(stdout, &stream);
    IREE_RETURN_IF_ERROR(loom_compile_report_capture_write_output(
        compile_report_capture, &stream, allocator));
    return fflush(stdout) == 0
               ? iree_ok_status()
               : iree_make_status(IREE_STATUS_DATA_LOSS,
                                  "failed to flush compile report stdout");
  }

  loom_tooling_output_stream_t output;
  IREE_RETURN_IF_ERROR(
      loom_tooling_output_stream_open(path, allocator, &output));
  iree_status_t status = loom_compile_report_capture_write_output(
      compile_report_capture, &output.stream, allocator);
  return iree_status_join(status, loom_tooling_output_stream_close(&output));
}

static void loom_compile_record_terminal_report_status(
    loom_target_compile_report_t* report, iree_status_code_t status_code,
    int exit_code) {
  if (report == NULL) {
    return;
  }
  if (status_code != IREE_STATUS_OK) {
    loom_target_compile_report_record_status(report, status_code);
  } else if (exit_code != 0) {
    loom_target_compile_report_record_status(report,
                                             IREE_STATUS_FAILED_PRECONDITION);
  }
}

static iree_status_t loom_compile_emit_command(
    loom_run_session_t* session, loom_run_module_t* run_module,
    const loom_compile_options_t* compile_options,
    loom_compile_report_capture_t* compile_report_capture,
    iree_allocator_t allocator, bool* out_emitted) {
  loom_compile_diagnostic_sink_t diagnostic_sink = {
      .run_module = run_module,
      .compile_report_capture = compile_report_capture,
  };
  loom_low_descriptor_text_print_context_initialize(
      &loom_run_session_low_descriptor_registry(session)->registry,
      &diagnostic_sink.type_print_context);

  if (compile_options->report != NULL) {
    compile_options->report->artifact_kind =
        LOOM_TARGET_COMPILE_ARTIFACT_KIND_TARGET_ARTIFACT;
    compile_options->report->backend_name = IREE_SV("command");
    compile_options->report->artifact_format =
        IREE_SV(LOOM_COMPILE_COMMAND_MANIFEST_FORMAT);
  }
  return loom_compile_command_backend_emit(
      session, run_module,
      &(loom_compile_command_backend_options_t){
          .artifact_directory =
              iree_make_cstring_view(FLAG_emit_command_artifacts),
          .kernel_request_directory =
              iree_make_cstring_view(FLAG_emit_kernel_requests),
          .manifest_path = iree_make_cstring_view(FLAG_output),
          .diagnostic_sink =
              {
                  .fn = loom_compile_diagnostic_sink,
                  .user_data = &diagnostic_sink,
              },
          .source_resolver = compile_options->source_resolver,
          .max_errors = compile_options->max_errors,
      },
      out_emitted, allocator);
}

static iree_status_t loom_compile_emit_artifact(
    const loom_artifact_provider_t* artifact_provider,
    const loom_artifact_target_t* artifact_target,
    loom_run_module_t* run_module,
    const loom_compile_options_t* compile_options,
    const loom_compile_report_capture_t* compile_report_capture,
    iree_string_view_t artifact_manifest_output_path,
    iree_allocator_t allocator, bool* out_emitted, bool* out_report_written) {
  *out_emitted = false;
  *out_report_written = false;
  const iree_string_view_t output_path = iree_make_cstring_view(FLAG_output);
  const iree_string_view_t target_artifact_path =
      iree_make_cstring_view(FLAG_emit_target_artifact);
  if (!iree_string_view_is_empty(target_artifact_path) &&
      loom_tooling_file_path_is_stdio(output_path) &&
      loom_tooling_file_path_is_stdio(target_artifact_path)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--output and --emit-target-artifact cannot both write to stdout");
  }
  if (!iree_string_view_is_empty(artifact_manifest_output_path) &&
      loom_tooling_file_path_is_stdio(output_path) &&
      loom_tooling_file_path_is_stdio(artifact_manifest_output_path)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--output and --emit-artifact-manifest cannot both write to stdout");
  }
  if (!iree_string_view_is_empty(target_artifact_path) &&
      !iree_string_view_is_empty(artifact_manifest_output_path) &&
      loom_tooling_file_path_is_stdio(target_artifact_path) &&
      loom_tooling_file_path_is_stdio(artifact_manifest_output_path)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--emit-target-artifact and "
                            "--emit-artifact-manifest cannot both write to "
                            "stdout");
  }

  loom_artifact_candidate_t candidate = {0};
  iree_status_t status =
      artifact_target != NULL
          ? loom_artifact_candidate_emit_target(
                artifact_provider, artifact_target, run_module->module,
                compile_options, allocator, &candidate)
          : loom_artifact_candidate_emit_module_target(
                artifact_provider, run_module->module, compile_options,
                allocator, &candidate);
  if (iree_status_is_ok(status) && candidate.compiled) {
    status = loom_tooling_write_output_byte_sequence(
        output_path, candidate.artifact.executable_data, allocator);
  }
  if (iree_status_is_ok(status) && candidate.compiled) {
    status = loom_compile_write_optional_target_artifact(&candidate.artifact,
                                                         allocator);
  }
  if (iree_status_is_ok(status) && candidate.compiled) {
    status = loom_compile_write_optional_artifact_manifest(
        &candidate.artifact, artifact_manifest_output_path, allocator);
  }
  if (iree_status_is_ok(status)) {
    *out_emitted = candidate.compiled;
  }
  if (loom_compile_report_capture_is_enabled(compile_report_capture)) {
    loom_compile_record_terminal_report_status(compile_options->report,
                                               iree_status_code(status),
                                               candidate.compiled ? 0 : 1);
    status = iree_status_join(
        status,
        loom_compile_write_report(compile_report_capture,
                                  artifact_manifest_output_path, allocator));
    *out_report_written = true;
  }
  loom_artifact_candidate_deinitialize(&candidate);
  return status;
}

static iree_string_view_t loom_compile_target_artifact_identifier(
    iree_string_view_t output_path, const loom_target_emitter_t* emitter) {
  return loom_tooling_file_path_is_stdio(output_path)
             ? emitter->default_identifier
             : output_path;
}

static iree_status_t loom_compile_emit_target(
    const loom_target_environment_t* target_environment,
    loom_run_session_t* session, const loom_target_emitter_t* target_emitter,
    loom_run_module_t* run_module,
    const loom_compile_options_t* compile_options, iree_allocator_t allocator,
    bool* out_emitted) {
  *out_emitted = false;
  if (target_emitter == NULL || target_emitter->emit == NULL) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "selected target emitter is incomplete");
  }
  if (!iree_string_view_is_empty(
          iree_make_cstring_view(FLAG_emit_target_artifact))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--emit-target-artifact is only valid for loadable kernel formats");
  }
  if (compile_options->artifact_manifest.mode !=
      LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--artifact-manifest is only valid for loadable kernel formats");
  }

  loom_target_entry_options_t target_options = {
      .function_versions = compile_options->function_versions,
      .diagnostic_sink = compile_options->diagnostic_sink,
      .source_resolver = compile_options->source_resolver,
      .max_errors = compile_options->max_errors,
  };
  loom_target_entry_diagnostic_emitter_t diagnostic_emitter = {0};
  loom_target_entry_diagnostic_emitter_initialize(
      run_module->module, &target_options, LOOM_EMITTER_VERIFIER,
      &diagnostic_emitter);

  iree_arena_allocator_t arena;
  iree_arena_initialize(loom_run_session_block_pool(session), &arena);
  loom_target_emit_artifact_t artifact = {0};
  const iree_string_view_t output_path = iree_make_cstring_view(FLAG_output);
  const iree_string_view_t identifier =
      loom_compile_target_artifact_identifier(output_path, target_emitter);
  if (compile_options->report != NULL) {
    compile_options->report->artifact_kind =
        LOOM_TARGET_COMPILE_ARTIFACT_KIND_TARGET_ARTIFACT;
    compile_options->report->backend_name = target_emitter->name;
    compile_options->report->artifact_format = loom_target_artifact_format_name(
        target_emitter->target_artifact_format);
  }
  const loom_target_emit_request_t request = {
      .target_environment = target_environment,
      .low_descriptor_registry =
          &loom_run_session_low_descriptor_registry(session)->registry,
      .module = run_module->module,
      .function_versions = compile_options->function_versions,
      .identifier = identifier,
      .compile_report = compile_options->report,
      .diagnostic_emitter = loom_target_entry_emitter(&diagnostic_emitter),
      .scratch_arena = &arena,
      .allocator = allocator,
  };

  iree_status_t status = target_emitter->emit(&request, &artifact);
  if (compile_options->report != NULL) {
    loom_target_compile_report_record_status(compile_options->report,
                                             iree_status_code(status));
    if (artifact.contents != NULL) {
      loom_target_compile_report_record_artifact_size(
          compile_options->report,
          iree_byte_sequence_length(artifact.contents));
    }
  }
  if (iree_status_is_ok(status) && diagnostic_emitter.error_count == 0 &&
      artifact.contents != NULL &&
      iree_byte_sequence_length(artifact.contents) != 0) {
    status = loom_tooling_write_output_byte_sequence(
        output_path, artifact.contents, allocator);
    if (iree_status_is_ok(status)) {
      *out_emitted = true;
    }
  }
  loom_target_emit_artifact_release(&artifact);
  iree_arena_deinitialize(&arena);
  return status;
}

static iree_status_t loom_compile_validate_request_flags(
    const loom_compile_request_t* request) {
  const iree_string_view_t command_artifact_directory =
      iree_make_cstring_view(FLAG_emit_command_artifacts);
  const iree_string_view_t kernel_request_directory =
      iree_make_cstring_view(FLAG_emit_kernel_requests);
  if (!loom_compile_request_is_command(request)) {
    if (!iree_string_view_is_empty(command_artifact_directory)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "--emit-command-artifacts requires product 'command'");
    }
    if (!iree_string_view_is_empty(kernel_request_directory)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--emit-kernel-requests requires "
                              "product 'command'");
    }
    return iree_ok_status();
  }

  if (iree_string_view_is_empty(command_artifact_directory) ||
      loom_tooling_file_path_is_stdio(command_artifact_directory)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "format 'loom-command' requires "
                            "--emit-command-artifacts=<directory>");
  }
  if (!iree_string_view_is_empty(
          iree_make_cstring_view(FLAG_emit_target_artifact))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--emit-target-artifact is only valid for loadable kernel formats");
  }
  if (!iree_string_view_is_empty(kernel_request_directory) &&
      loom_tooling_file_path_is_stdio(kernel_request_directory)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--emit-kernel-requests requires a filesystem directory");
  }
  return iree_ok_status();
}

static iree_status_t loom_compile_specialize_explicit_target(
    const loom_target_environment_t* target_environment,
    loom_run_session_t* session, loom_run_module_t* run_module,
    const loom_compile_target_t* target,
    loom_compile_report_capture_t* compile_report_capture,
    iree_allocator_t allocator) {
  loom_compile_diagnostic_sink_t compile_diagnostic_sink = {
      .run_module = run_module,
      .compile_report_capture = compile_report_capture,
  };
  loom_low_descriptor_text_print_context_initialize(
      &loom_run_session_low_descriptor_registry(session)->registry,
      &compile_diagnostic_sink.type_print_context);
  const loom_target_entry_options_t diagnostic_options = {
      .diagnostic_sink =
          {
              .fn = loom_compile_diagnostic_sink,
              .user_data = &compile_diagnostic_sink,
          },
      .source_resolver = loom_run_module_source_resolver(run_module),
  };
  loom_target_entry_diagnostic_emitter_t diagnostic_emitter;
  loom_target_entry_diagnostic_emitter_initialize(
      run_module->module, &diagnostic_options, LOOM_EMITTER_PASS,
      &diagnostic_emitter);

  uint32_t error_count = 0;
  IREE_RETURN_IF_ERROR(loom_target_specialize_module_kernel_entries(
      target_environment, target->target_profile,
      loom_target_entry_emitter(&diagnostic_emitter),
      loom_run_session_block_pool(session), allocator, &run_module->module,
      &error_count));
  if (error_count != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target profile specialization failed with %u error%s", error_count,
        error_count == 1 ? "" : "s");
  }
  return iree_ok_status();
}

static void loom_compile_print_agents_markdown(FILE* stream) {
  fprintf(
      stream,
      "## loom-compile\n"
      "\n"
      "`loom-compile` specializes Loom text or bytecode and emits an offline\n"
      "artifact. Selected roots infer the product, the target selects the "
      "machine or API contract, and the format names the output encoding.\n"
      "\n"
      "### Select the product\n"
      "\n"
      "```shell\n"
      "loom-compile program.loom --output=manifest.json \\\n"
      "  --emit-command-artifacts=commands/ \\\n"
      "  --emit-kernel-requests=kernel-requests/\n"
      "loom-compile kernel.loom \\\n"
      "  --target=amdgpu:gfx11-generic --output=kernel.hsaco\n"
      "loom-compile catalog.loombc --root=@entry \\\n"
      "  --target=amdgpu:gfx1151 --output=entry.hsaco\n"
      "loom-compile kernel.loom --format=llvmir-text --output=kernel.ll\n"
      "```\n"
      "\n"
      "Command roots select `--format=loom-command` by default and emit "
      "portable command programs and their shared\n"
      "entry-requirement manifest. `--emit-kernel-requests` additionally "
      "emits\n"
      "one ordinary Loom bytecode request per reachable semantic kernel "
      "class.\n"
      "Kernel roots select the canonical format for their target. "
      "`--format` can request an exact alternate or diagnostic format. "
      "With no explicit roots, `--product` selects that product's canonical "
      "roots. "
      "`--root=@symbol`\n"
      "selects entries from a catalog; without it, the tool compiles every\n"
      "relevant exported entry and its dependency closure.\n"
      "\n"
      "A generic target such as `gfx11-generic` preserves family portability.\n"
      "An exact target such as `gfx1151` permits device specialization. "
      "Authored\n"
      "target requirements remain constraints and incompatible specialization "
      "fails.\n"
      "`--config=key=value` and `--config-file=file.jsonc` bind\n"
      "`config.decl` values before compilation.\n"
      "\n"
      "### Pipeline contract\n"
      "\n"
      "The default pipeline is the maintained source-to-artifact path. Use it "
      "for\n"
      "normal source, correctness acceptance, and production artifacts.\n"
      "\n"
      "```shell\n"
      "loom-compile prepared-low.loom --format=amdgpu-hsaco \\\n"
      "  --pipeline=none --output=oracle.hsaco\n"
      "```\n"
      "\n"
      "`--pipeline=none` disables all compiler transformations. The input is\n"
      "passed directly to the selected emitter and must already satisfy its "
      "complete\n"
      "input contract. This is a supported assembly path for prepared Low and "
      "other\n"
      "already lowered forms; success does not show that source survives the "
      "default\n"
      "pipeline. Named pipelines and explicit pass lists replace the default "
      "pipeline\n"
      "with the deliberately selected transformation sequence.\n"
      "\n"
      "### Describe the artifact and compilation\n"
      "\n"
      "```shell\n"
      "loom-compile kernel.loom --format=amdgpu-hsaco \\\n"
      "  --target=amdgpu:gfx11-generic --output=kernel.hsaco \\\n"
      "  --artifact-manifest=summary --emit-artifact-manifest=manifest.json "
      "\\\n"
      "  --compile-report=summary --compile-report-output=report.json\n"
      "loom-compile-report show report.json\n"
      "loom-compile-report diff baseline.json report.json\n"
      "loom-compile-report suggest report.json\n"
      "jq '.functions[] | {name, target, workgroup_size}' manifest.json\n"
      "jq '{function,lowered,target_export,"
      "code_bytes:.emission.code_byte_count,"
      "spills:.allocation.materialized_spill_storage_count}' report.json\n"
      "jq '.entries.rows[]? | {function,source_function,target_export_symbol,"
      "code_byte_count,allocation_spill_count}' report.json\n"
      "```\n"
      "\n"
      "The artifact manifest describes what was emitted. The compile report "
      "describes\n"
      "how the compiler produced it. Start with summary mode and\n"
      "`loom-compile-report show`; request details or raw fields only after "
      "that\n"
      "view identifies a concrete scheduling, allocation, memory, or "
      "legalization\n"
      "question. Version-zero reports are same-checkout diagnostics rather "
      "than a\n"
      "compatibility format.\n"
      "\n"
      "Detailed compilation and report workflows live in\n"
      "`loom/docs/src/workflows/compile-artifacts.md` and\n"
      "`loom/docs/src/workflows/compile-reports.md`. Scenario-indexed raw\n"
      "queries live in\n"
      "`loom/docs/src/workflows/compile-report-queries.md`. Native-to-Loom\n"
      "reconstruction lives in\n"
      "`loom/docs/src/workflows/oracles/native-schedule.md`.\n");
}

int main(int argc, char** argv) {
  iree_flags_set_usage(
      "loom-compile",
      "Compiles a Loom module to a runtime artifact.\n"
      "\n"
      "Usage:\n"
      "  loom-compile [file.loom] --product=command "
      "--output=commands.json --emit-command-artifacts=commands/ "
      "[--emit-kernel-requests=kernels/]\n"
      "  loom-compile [file.loom] --product=kernel "
      "--format=amdgpu-hsaco "
      "--target=amdgpu:gfx11-generic --output=kernel.hsaco\n"
      "  loom-compile --agents_md\n"
      "\n"
      "Repeat --config=key=value to materialize compile-time config symbols "
      "before the pass pipeline. Use --config-file=path for a JSON/JSONC "
      "object such as {\"model36\":{\"model\":{\"hidden_size\":4096}}}. "
      "Files and direct bindings share one config set and duplicate keys are "
      "rejected.\n"
      "Use --agents_md to print agent-facing workflow "
      "guidance.\n" LOOM_TOOLING_PASS_TRACE_USAGE);
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

  iree_allocator_t allocator = iree_allocator_system();
  const loom_tooling_source_path_options_t source_path_options = {
      .prefix_maps = FLAG_source_prefix_map_list(),
  };
  const loom_tooling_compile_environment_t* compile_environment =
      loom_tooling_configured_compile_environment();
  iree_status_t status = iree_ok_status();
  loom_tooling_config_set_t config_set;
  loom_tooling_config_set_initialize(allocator, &config_set);

  iree_io_file_contents_t* contents = NULL;
  char* input_filename_storage = NULL;
  loom_run_session_t session = {0};
  loom_run_module_t run_module = {0};
  loom_compile_report_capture_options_t compile_report_options = {0};
  loom_compile_report_capture_t compile_report_capture = {0};
  loom_tooling_pass_trace_t pass_trace = {0};
  loom_compile_request_t request = {0};
  loom_compile_artifact_manifest_options_t artifact_manifest_options = {0};
  iree_string_view_t artifact_manifest_output_path = iree_string_view_empty();
  char* artifact_manifest_output_path_storage = NULL;
  bool emitted = false;
  bool report_written = false;
  int exit_code = 0;

  if (iree_status_is_ok(status)) {
    status = loom_compile_initialize_session(
        compile_environment->target_environment, allocator, &session);
  }
  if (iree_status_is_ok(status)) {
    status = loom_compile_append_config_files(&config_set, allocator);
  }
  if (iree_status_is_ok(status)) {
    status = loom_compile_append_config_flags(&config_set);
  }
  if (iree_status_is_ok(status)) {
    status = loom_compile_parse_input_module(
        argc, argv, &session, &source_path_options, allocator, &contents,
        &input_filename_storage, &run_module);
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_compile_materialize_config_set(&session, &run_module, &config_set);
  }
  if (iree_status_is_ok(status)) {
    status = loom_compile_report_options_initialize(&compile_report_options);
  }
  if (iree_status_is_ok(status)) {
    status = loom_compile_report_capture_initialize(
        &compile_report_options, allocator, &compile_report_capture);
  }
  if (iree_status_is_ok(status)) {
    status = loom_compile_report_capture_record_materialized_config(
        &compile_report_capture, run_module.module, &config_set);
  }
  if (iree_status_is_ok(status)) {
    const loom_compile_request_options_t request_options =
        loom_compile_request_options_from_flags(FLAG_root_list());
    status = loom_compile_request_resolve(
        run_module.module, &request_options,
        compile_environment->artifact_provider_registry,
        compile_environment->target_environment,
        /*inferred_target_fact_type=*/NULL, &request);
    if (iree_status_is_ok(status)) {
      status = loom_compile_validate_request_flags(&request);
    }
  }
  if (iree_status_is_ok(status) &&
      request.explicit_target.target_profile != NULL) {
    status = loom_compile_specialize_explicit_target(
        compile_environment->target_environment, &session, &run_module,
        &request.explicit_target, &compile_report_capture, allocator);
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_compile_select_roots(&session, &run_module, &request, allocator);
  }
  if (iree_status_is_ok(status)) {
    status = loom_compile_artifact_manifest_options_initialize(
        &artifact_manifest_options,
        loom_compile_request_artifact_provider(&request) != NULL, allocator,
        &artifact_manifest_output_path, &artifact_manifest_output_path_storage);
  }

  loom_compile_options_t compile_options = {0};
  loom_compile_options_initialize(&compile_options);
  loom_compile_pipeline_result_t pipeline_result = {0};
  compile_options.artifact_manifest = artifact_manifest_options;
  const loom_artifact_provider_t* artifact_provider =
      loom_compile_request_artifact_provider(&request);
  if (artifact_provider != NULL) {
    compile_options.target_pipeline_options =
        artifact_provider->default_pipeline_options;
  }
  if (iree_status_is_ok(status)) {
    status = loom_compile_sanitizer_options_initialize(
        &compile_options.target_pipeline_options.sanitizer);
  }
  if (iree_status_is_ok(status)) {
    compile_options.source_resolver =
        loom_run_module_source_resolver(&run_module);
    loom_compile_report_capture_configure_compile_options(
        &compile_report_capture, &compile_options);
  }
  if (iree_status_is_ok(status)) {
    const iree_string_view_t target_artifact_path =
        iree_make_cstring_view(FLAG_emit_target_artifact);
    const loom_tooling_pass_trace_stdout_conflict_t stdout_conflicts[] = {
        {
            .active = true,
            .flag_name = IREE_SV("--output"),
            .path = iree_make_cstring_view(FLAG_output),
        },
        {
            .active = !iree_string_view_is_empty(target_artifact_path),
            .flag_name = IREE_SV("--emit-target-artifact"),
            .path = target_artifact_path,
        },
        {
            .active = !iree_string_view_is_empty(artifact_manifest_output_path),
            .flag_name = IREE_SV("--emit-artifact-manifest"),
            .path = artifact_manifest_output_path,
        },
        {
            .active = loom_compile_report_capture_options_is_enabled(
                &compile_report_options),
            .flag_name = IREE_SV("--compile-report-output"),
            .path = iree_make_cstring_view(FLAG_compile_report_output),
        },
    };
    status = loom_tooling_pass_trace_open_from_flags(
        &(loom_tooling_pass_trace_open_options_t){
            .tool_name = IREE_SV("loom-compile"),
            .input_path = run_module.filename,
            .stdout_conflicts = stdout_conflicts,
            .stdout_conflict_count = IREE_ARRAYSIZE(stdout_conflicts),
        },
        allocator, &pass_trace);
    if (iree_status_is_ok(status) && pass_trace.enabled) {
      loom_low_descriptor_text_asm_environment_initialize(
          &loom_run_session_low_descriptor_registry(&session)->registry,
          &pass_trace.pass_options.print_options.low_asm_environment);
    }
  }
  if (iree_status_is_ok(status)) {
    // Command compilation owns selective source preparation after indexing.
    // Eagerly expanding the whole input here would erase unresolved kernel
    // decisions before command planning can classify launch sites and emit
    // independent kernel requests. Explicit user pipelines remain honored.
    const bool run_pipeline = !loom_compile_request_is_command(&request) ||
                              !loom_compile_pipeline_is_default(
                                  iree_make_cstring_view(FLAG_pipeline));
    if (run_pipeline) {
      status = loom_compile_run_pass_pipeline(
          compile_environment->target_environment, &session, &run_module,
          LOOM_COMPILE_DEFAULT_PIPELINE_PREPARED_LOW, &compile_options,
          &compile_report_capture, loom_tooling_pass_trace_options(&pass_trace),
          &pipeline_result);
    }
    status =
        iree_status_join(status, loom_tooling_pass_trace_close(&pass_trace));
    if (iree_status_is_ok(status) && pipeline_result.pass.error_count != 0) {
      if (compile_options.report != NULL) {
        loom_target_compile_report_record_status(
            compile_options.report, IREE_STATUS_FAILED_PRECONDITION);
      }
      exit_code = 1;
    }
    compile_options.function_versions = &pipeline_result.function_versions.list;
  }
  if (iree_status_is_ok(status) && exit_code == 0 &&
      !loom_compile_request_is_command(&request)) {
    status =
        loom_tooling_config_require_resolved_module(run_module.module, NULL);
  }

  if (iree_status_is_ok(status) && exit_code == 0) {
    const loom_artifact_target_t explicit_artifact_target =
        loom_compile_target_artifact_target(&request.explicit_target);
    switch (request.producer.kind) {
      case LOOM_COMPILE_PRODUCER_ARTIFACT:
        status = loom_compile_emit_artifact(
            request.producer.value.artifact_provider,
            request.explicit_target.target_profile != NULL
                ? &explicit_artifact_target
                : NULL,
            &run_module, &compile_options, &compile_report_capture,
            artifact_manifest_output_path, allocator, &emitted,
            &report_written);
        break;
      case LOOM_COMPILE_PRODUCER_TARGET_EMITTER:
        status = loom_compile_emit_target(
            compile_environment->target_environment, &session,
            request.producer.value.target_emitter, &run_module,
            &compile_options, allocator, &emitted);
        break;
      case LOOM_COMPILE_PRODUCER_COMMAND:
        status = loom_compile_emit_command(
            &session, &run_module, &compile_options, &compile_report_capture,
            allocator, &emitted);
        break;
      case LOOM_COMPILE_PRODUCER_INVALID:
        status = iree_make_status(IREE_STATUS_INTERNAL,
                                  "compile request has no producer");
        break;
    }
  }
  if (iree_status_is_ok(status) && exit_code == 0 && !emitted) {
    if (compile_options.report != NULL) {
      loom_target_compile_report_record_status(compile_options.report,
                                               IREE_STATUS_FAILED_PRECONDITION);
    }
    exit_code = 1;
  }
  status = iree_status_join(status, loom_tooling_pass_trace_close(&pass_trace));
  if (!report_written) {
    loom_compile_record_terminal_report_status(
        compile_options.report, iree_status_code(status), exit_code);
    status = iree_status_join(
        status,
        loom_compile_write_report(&compile_report_capture,
                                  artifact_manifest_output_path, allocator));
  }
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = 1;
  }

  loom_compile_pipeline_result_deinitialize(&pipeline_result);
  loom_compile_report_capture_deinitialize(&compile_report_capture);
  loom_run_module_deinitialize(&run_module);
  iree_allocator_free(allocator, input_filename_storage);
  iree_io_file_contents_free(contents);
  loom_tooling_config_set_deinitialize(&config_set);
  loom_run_session_deinitialize(&session);
  iree_allocator_free(allocator, artifact_manifest_output_path_storage);

  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
