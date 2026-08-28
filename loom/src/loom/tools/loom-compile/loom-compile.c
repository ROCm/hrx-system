// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// loom-compile: compiles a Loom module to a runtime artifact.

#include <stdio.h>

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/io/byte_sequence.h"
#include "iree/io/file_contents.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/error/diagnostic.h"
#include "loom/format/text/printer.h"
#include "loom/ir/module.h"
#include "loom/link/linker.h"
#include "loom/ops/op_defs.h"
#include "loom/sanitizer/options.h"
#include "loom/target/arch/cmd/artifact_set.h"
#include "loom/target/arch/cmd/provider.h"
#include "loom/target/entry_selection.h"
#include "loom/target/module_specialization.h"
#include "loom/target/reporting/artifact_manifest.h"
#include "loom/tooling/cli/help.h"
#include "loom/tooling/compile/pipeline.h"
#include "loom/tooling/config/config.h"
#include "loom/tooling/context/context.h"
#include "loom/tooling/execution/compile_report_capture.h"
#include "loom/tooling/execution/execution_provider.h"
#include "loom/tooling/execution/hal/artifact.h"
#include "loom/tooling/execution/hal/candidate.h"
#include "loom/tooling/execution/session.h"
#include "loom/tooling/io/file.h"
#include "loom/tooling/io/source_path.h"
#include "loom/tooling/pass/trace_cli.h"
#include "loom/tools/loom-compile/command_backend.h"

#ifndef LOOM_COMPILE_HAVE_AMDGPU
#define LOOM_COMPILE_HAVE_AMDGPU 0
#endif  // LOOM_COMPILE_HAVE_AMDGPU
#ifndef LOOM_COMPILE_HAVE_IREE_VM
#define LOOM_COMPILE_HAVE_IREE_VM 0
#endif  // LOOM_COMPILE_HAVE_IREE_VM
#ifndef LOOM_COMPILE_HAVE_SPIRV_VULKAN
#define LOOM_COMPILE_HAVE_SPIRV_VULKAN 0
#endif  // LOOM_COMPILE_HAVE_SPIRV_VULKAN
#ifndef LOOM_COMPILE_HAVE_LLVMIR
#define LOOM_COMPILE_HAVE_LLVMIR 0
#endif  // LOOM_COMPILE_HAVE_LLVMIR
#ifndef LOOM_COMPILE_HAVE_LLVMIR_AMDGPU_TARGET_ENV
#define LOOM_COMPILE_HAVE_LLVMIR_AMDGPU_TARGET_ENV 0
#endif  // LOOM_COMPILE_HAVE_LLVMIR_AMDGPU_TARGET_ENV
#ifndef LOOM_COMPILE_HAVE_LLVMIR_X86_TARGET_ENV
#define LOOM_COMPILE_HAVE_LLVMIR_X86_TARGET_ENV 0
#endif  // LOOM_COMPILE_HAVE_LLVMIR_X86_TARGET_ENV

typedef struct loom_compile_diagnostic_sink_t {
  // Parsed module used for full type rendering.
  const loom_run_module_t* run_module;
  // Printer context used to render target-owned register and storage types.
  loom_low_descriptor_text_print_context_t type_print_context;
  // Optional compile report capture receiving canonical diagnostic JSON.
  loom_run_compile_report_capture_t* compile_report_capture;
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
    loom_run_compile_report_capture_t* compile_report_capture =
        sink ? sink->compile_report_capture : NULL;
    status = loom_run_compile_report_capture_record_diagnostic(
        compile_report_capture, diagnostic, format_options.type_formatter);
  }
  return status;
}

#define LOOM_COMPILE_HAVE_ANY_HAL_ARTIFACT_PROVIDER \
  (LOOM_COMPILE_HAVE_AMDGPU || LOOM_COMPILE_HAVE_SPIRV_VULKAN)

#if LOOM_COMPILE_HAVE_AMDGPU
#include "loom/target/arch/amdgpu/provider.h"
#include "loom/tooling/target/amdgpu/artifact_provider.h"
#endif  // LOOM_COMPILE_HAVE_AMDGPU
#if LOOM_COMPILE_HAVE_IREE_VM
#include "loom/tooling/execution/ireevm/candidate.h"
#include "loom/tooling/execution/ireevm/provider.h"
#endif  // LOOM_COMPILE_HAVE_IREE_VM
#if LOOM_COMPILE_HAVE_SPIRV_VULKAN
#include "loom/target/arch/spirv/provider.h"
#include "loom/tooling/target/spirv/artifact_provider.h"
#endif  // LOOM_COMPILE_HAVE_SPIRV_VULKAN
#if LOOM_COMPILE_HAVE_LLVMIR
#include "loom/target/arch/llvmir/provider.h"
#include "loom/target/emit/llvmir/artifact_emitter.h"
#if LOOM_COMPILE_HAVE_LLVMIR_AMDGPU_TARGET_ENV
#include "loom/target/emit/llvmir/amdgpu/target_env.h"
#endif  // LOOM_COMPILE_HAVE_LLVMIR_AMDGPU_TARGET_ENV
#if LOOM_COMPILE_HAVE_LLVMIR_X86_TARGET_ENV
#include "loom/target/emit/llvmir/x86/target_env.h"
#endif  // LOOM_COMPILE_HAVE_LLVMIR_X86_TARGET_ENV
#endif  // LOOM_COMPILE_HAVE_LLVMIR

IREE_FLAG(string, backend, "vm",
          "Compilation backend to emit, such as 'command', 'vm', or a "
          "linked native backend.");
IREE_FLAG_NAMED(string, module_name, "module-name", "loom",
                "Module name to store in VM bytecode archives.");
IREE_FLAG(string, target, "",
          "Optional HAL backend target key, such as 'gfx11-generic' or "
          "'gfx1151'. When present, every materialized HAL kernel entry is "
          "specialized to that exact provider-owned profile before the pass "
          "pipeline. Authored targets remain compatibility requirements.");
IREE_FLAG_LIST(string, root,
               "Root symbol to materialize before compilation. Repeat for "
               "multiple roots. When omitted, HAL backends compile every "
               "kernel entry and its dependency closure, the command backend "
               "emits every public or retained command program, and other "
               "backends compile the full input module.");
IREE_FLAG(string, pipeline, "default",
          "Pass pipeline to run before artifact emission. Use 'default' or "
          "empty for the backend's default compile pipeline, 'none' to "
          "disable pass execution, '@symbol' to run a module-local "
          "pass.pipeline, or a comma-separated pass list such as "
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
          "Output path for the primary runtime artifact. For VM this is the VM "
          "bytecode archive; for command this is the command artifact-set "
          "manifest; for HAL this is the executable artifact passed to the "
          "selected HAL loader.");
IREE_FLAG_NAMED(
    string, emit_target_artifact, "emit-target-artifact", "",
    "Optional output path for a target-native artifact produced beside the "
    "primary runtime artifact, such as AMDGPU HSACO.");
IREE_FLAG_NAMED(
    string, emit_command_artifacts, "emit-command-artifacts", "",
    "Directory receiving portable root artifacts for --backend=command. The "
    "primary --output is the artifact-set manifest.");
IREE_FLAG_NAMED(
    string, emit_kernel_requests, "emit-kernel-requests", "",
    "Optional directory receiving independently compilable Loom bytecode "
    "kernel requests for --backend=command.");
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

#if LOOM_COMPILE_HAVE_AMDGPU
static const loom_run_execution_provider_t kLoomCompileAmdgpuProvider = {
    .name = IREE_SVL("amdgpu"),
    .target_provider = &loom_amdgpu_target_provider,
};
#endif  // LOOM_COMPILE_HAVE_AMDGPU

static const loom_run_execution_provider_t kLoomCompileCommandProvider = {
    .name = IREE_SVL("command"),
    .target_provider = &loom_cmd_target_provider,
};

#if LOOM_COMPILE_HAVE_SPIRV_VULKAN
static const loom_run_execution_provider_t kLoomCompileSpirvProvider = {
    .name = IREE_SVL("spirv"),
    .target_provider = &loom_spirv_target_provider,
};
#endif  // LOOM_COMPILE_HAVE_SPIRV_VULKAN

#if LOOM_COMPILE_HAVE_LLVMIR
static const loom_run_execution_provider_t kLoomCompileLlvmirProvider = {
    .name = IREE_SVL("llvmir"),
    .target_provider = &loom_llvmir_target_provider,
};

static const loom_run_execution_provider_t
    kLoomCompileLlvmirArtifactEmitterProvider = {
        .name = IREE_SVL("llvmir-artifacts"),
        .target_provider = &loom_llvmir_artifact_emitter_provider,
};
#endif  // LOOM_COMPILE_HAVE_LLVMIR

static const loom_run_execution_provider_t* const kLoomCompileProviders[] = {
    &kLoomCompileCommandProvider,
#if LOOM_COMPILE_HAVE_AMDGPU
    &kLoomCompileAmdgpuProvider,
#endif  // LOOM_COMPILE_HAVE_AMDGPU
#if LOOM_COMPILE_HAVE_IREE_VM
    &loom_ireevm_execution_provider,
#endif  // LOOM_COMPILE_HAVE_IREE_VM
#if LOOM_COMPILE_HAVE_SPIRV_VULKAN
    &kLoomCompileSpirvProvider,
#endif  // LOOM_COMPILE_HAVE_SPIRV_VULKAN
#if LOOM_COMPILE_HAVE_LLVMIR
    // Register the target executor and the artifact-only emitter.
    &kLoomCompileLlvmirProvider,
    &kLoomCompileLlvmirArtifactEmitterProvider,
#endif  // LOOM_COMPILE_HAVE_LLVMIR
};

static const loom_run_execution_provider_set_t kLoomCompileProviderSet = {
    .providers = kLoomCompileProviders,
    .provider_count = IREE_ARRAYSIZE(kLoomCompileProviders),
};

#if LOOM_COMPILE_HAVE_ANY_HAL_ARTIFACT_PROVIDER
static const loom_run_hal_artifact_provider_t* const
    kLoomCompileHalArtifactProviders[] = {
#if LOOM_COMPILE_HAVE_AMDGPU
        &loom_amdgpu_hal_artifact_provider,
#endif  // LOOM_COMPILE_HAVE_AMDGPU
#if LOOM_COMPILE_HAVE_SPIRV_VULKAN
        &loom_spirv_vulkan_hal_artifact_provider,
#endif  // LOOM_COMPILE_HAVE_SPIRV_VULKAN
};
#endif  // LOOM_COMPILE_HAVE_ANY_HAL_ARTIFACT_PROVIDER

static const loom_run_hal_artifact_provider_registry_t
    kLoomCompileHalArtifactProviderRegistry = {
#if LOOM_COMPILE_HAVE_ANY_HAL_ARTIFACT_PROVIDER
        .providers = kLoomCompileHalArtifactProviders,
        .provider_count = IREE_ARRAYSIZE(kLoomCompileHalArtifactProviders),
#else
        .providers = NULL,
        .provider_count = 0,
#endif  // LOOM_COMPILE_HAVE_ANY_HAL_ARTIFACT_PROVIDER
};

typedef struct loom_compile_backend_t {
  // Selected HAL artifact provider, if |--backend| names a HAL path.
  const loom_run_hal_artifact_provider_t* hal_artifact_provider;
  // Selected target-owned emitter, if |--backend| names an emitter format.
  const loom_target_emitter_t* target_emitter;
  // True when |--backend| names the VM archive emitter.
  bool is_vm_backend;
  // True when |--backend| names portable command artifact emission.
  bool is_command_backend;
} loom_compile_backend_t;

static iree_status_t loom_compile_register_context(void* user_data,
                                                   loom_context_t* context) {
  loom_run_execution_environment_t* environment =
      (loom_run_execution_environment_t*)user_data;
  return loom_tooling_context_register_tool_dialects_with_target_environment(
      loom_run_execution_environment_target_environment(environment), context);
}

static iree_status_t loom_compile_initialize_session(
    loom_run_execution_environment_t* environment, iree_allocator_t allocator,
    loom_run_session_t* out_session) {
  loom_run_session_options_t session_options = {0};
  loom_run_session_options_initialize(&session_options);
  session_options.host_allocator = allocator;
  session_options.register_context = (loom_run_register_context_callback_t){
      .fn = loom_compile_register_context,
      .user_data = environment,
  };
  session_options.initialize_low_descriptor_registry =
      loom_run_execution_environment_low_descriptor_registry_callback(
          environment);
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

static iree_status_t loom_compile_select_roots(
    loom_run_session_t* session, loom_run_module_t* run_module,
    const loom_run_hal_artifact_provider_t* hal_artifact_provider,
    iree_allocator_t allocator) {
  iree_string_view_list_t roots = FLAG_root_list();
  iree_string_view_t* implicit_root_values = NULL;
  if (roots.count == 0 && hal_artifact_provider != NULL) {
    loom_op_t* op = NULL;
    loom_block_for_each_op(loom_module_block(run_module->module), op) {
      if (loom_func_like_is_kernel_entry(
              loom_func_like_cast(run_module->module, op))) {
        ++roots.count;
      }
    }
    if (roots.count == 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "HAL backend '%.*s' requires at least one kernel entry",
          (int)hal_artifact_provider->name.size,
          hal_artifact_provider->name.data);
    }
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        allocator, roots.count, sizeof(*implicit_root_values),
        (void**)&implicit_root_values));
    iree_host_size_t root_ordinal = 0;
    loom_block_for_each_op(loom_module_block(run_module->module), op) {
      const loom_func_like_t function =
          loom_func_like_cast(run_module->module, op);
      if (!loom_func_like_is_kernel_entry(function)) {
        continue;
      }
      const loom_symbol_ref_t function_ref = loom_func_like_callee(function);
      const loom_symbol_t* function_symbol =
          &run_module->module->symbols.entries[function_ref.symbol_id];
      implicit_root_values[root_ordinal++] =
          run_module->module->strings.entries[function_symbol->name_id];
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
    loom_run_compile_report_capture_options_t* out_options) {
  loom_run_compile_report_capture_options_initialize(out_options);
  IREE_RETURN_IF_ERROR(loom_run_compile_report_capture_options_parse_request(
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
    loom_run_candidate_artifact_manifest_options_t* out_options,
    bool is_hal_backend, iree_allocator_t allocator,
    iree_string_view_t* out_output_path, char** out_output_path_storage) {
  *out_options = (loom_run_candidate_artifact_manifest_options_t){0};
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
  if (!is_hal_backend) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--artifact-manifest is only valid for HAL artifact providers");
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
    const loom_run_execution_environment_t* environment,
    loom_run_session_t* session, loom_run_module_t* run_module,
    loom_compile_default_pipeline_t default_pipeline,
    const loom_run_candidate_compile_options_t* compile_options,
    const loom_pass_trace_options_t* trace_options,
    loom_compile_pipeline_result_t* out_result) {
  loom_compile_pipeline_options_t pipeline_options = {0};
  loom_compile_pipeline_options_initialize(&pipeline_options);
  pipeline_options.pipeline = iree_make_cstring_view(FLAG_pipeline);
  pipeline_options.default_pipeline = default_pipeline;
  pipeline_options.target_pipeline_options =
      compile_options->target_pipeline_options;
  pipeline_options.target_environment =
      loom_run_execution_environment_target_environment(environment);
  pipeline_options.low_descriptor_registry =
      loom_run_session_low_descriptor_registry(session);
  loom_compile_diagnostic_sink_t diagnostic_sink = {
      .run_module = run_module,
      .compile_report_capture = compile_options->report_capture,
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
    const loom_run_hal_artifact_t* artifact, iree_allocator_t allocator) {
  const iree_string_view_t path =
      iree_make_cstring_view(FLAG_emit_target_artifact);
  if (iree_string_view_is_empty(path)) {
    return iree_ok_status();
  }
  if (artifact->target_artifact_data == NULL ||
      iree_io_byte_sequence_length(artifact->target_artifact_data) == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "selected HAL artifact provider produced no "
                            "target-native artifact");
  }
  return loom_tooling_write_output_byte_sequence(
      path, artifact->target_artifact_data, allocator);
}

static iree_status_t loom_compile_write_optional_artifact_manifest(
    const loom_run_hal_artifact_t* artifact, iree_string_view_t path,
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
          "selected HAL artifact provider produced multiple artifact "
          "manifests");
    }
    manifest = sidecar;
  }
  if (manifest == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "selected HAL artifact provider produced no "
                            "artifact manifest");
  }
  return loom_tooling_write_output_byte_sequence(path, manifest->contents,
                                                 allocator);
}

static iree_status_t loom_compile_write_report(
    const loom_run_compile_report_capture_t* compile_report_capture,
    iree_string_view_t artifact_manifest_output_path,
    iree_allocator_t allocator) {
  if (!loom_run_compile_report_capture_is_enabled(compile_report_capture)) {
    return iree_ok_status();
  }
  const iree_string_view_t path =
      iree_make_cstring_view(FLAG_compile_report_output);
  if (iree_string_view_is_empty(path) ||
      iree_string_view_equal(path, IREE_SV("stderr"))) {
    loom_output_stream_t stream;
    loom_output_stream_for_file(stderr, &stream);
    IREE_RETURN_IF_ERROR(loom_run_compile_report_capture_write_output(
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
    IREE_RETURN_IF_ERROR(loom_run_compile_report_capture_write_output(
        compile_report_capture, &stream, allocator));
    return fflush(stdout) == 0
               ? iree_ok_status()
               : iree_make_status(IREE_STATUS_DATA_LOSS,
                                  "failed to flush compile report stdout");
  }

  loom_tooling_output_stream_t output;
  IREE_RETURN_IF_ERROR(
      loom_tooling_output_stream_open(path, allocator, &output));
  iree_status_t status = loom_run_compile_report_capture_write_output(
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
    const loom_run_candidate_compile_options_t* compile_options,
    iree_allocator_t allocator, bool* out_emitted) {
  loom_compile_diagnostic_sink_t diagnostic_sink = {
      .run_module = run_module,
      .compile_report_capture = compile_options->report_capture,
  };
  loom_low_descriptor_text_print_context_initialize(
      &loom_run_session_low_descriptor_registry(session)->registry,
      &diagnostic_sink.type_print_context);

  if (compile_options->report != NULL) {
    compile_options->report->artifact_kind =
        LOOM_TARGET_COMPILE_ARTIFACT_KIND_TARGET_ARTIFACT;
    compile_options->report->backend_name = IREE_SV("command");
    compile_options->report->artifact_format =
        IREE_SV(LOOM_CMD_PROGRAM_ARTIFACT_SET_MANIFEST_FORMAT);
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

static iree_status_t loom_compile_emit_hal(
    const loom_run_hal_artifact_provider_t* artifact_provider,
    const loom_run_hal_device_target_t* hal_target,
    loom_run_module_t* run_module,
    const loom_run_candidate_compile_options_t* compile_options,
    const loom_run_compile_report_capture_t* compile_report_capture,
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

  loom_run_hal_candidate_t candidate = {0};
  iree_status_t status = hal_target != NULL
                             ? loom_run_hal_candidate_emit_target(
                                   artifact_provider, hal_target, run_module,
                                   compile_options, allocator, &candidate)
                             : loom_run_hal_candidate_emit_module_target(
                                   artifact_provider, run_module,
                                   compile_options, allocator, &candidate);
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
  if (loom_run_compile_report_capture_is_enabled(compile_report_capture)) {
    loom_compile_record_terminal_report_status(compile_options->report,
                                               iree_status_code(status),
                                               candidate.compiled ? 0 : 1);
    status = iree_status_join(
        status,
        loom_compile_write_report(compile_report_capture,
                                  artifact_manifest_output_path, allocator));
    *out_report_written = true;
  }
  loom_run_hal_candidate_deinitialize(&candidate);
  return status;
}

static iree_status_t loom_compile_emit_vm(
    loom_run_module_t* run_module,
    const loom_run_candidate_compile_options_t* compile_options,
    iree_allocator_t allocator, bool* out_emitted) {
  *out_emitted = false;
  if (!iree_string_view_is_empty(
          iree_make_cstring_view(FLAG_emit_target_artifact))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--emit-target-artifact is only valid for HAL artifact providers");
  }
  if (compile_options->artifact_manifest.mode !=
      LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--artifact-manifest is only valid for HAL artifact providers");
  }
#if LOOM_COMPILE_HAVE_IREE_VM
  loom_ireevm_run_candidate_t candidate = {0};
  iree_status_t status = loom_ireevm_run_candidate_emit(
      run_module, compile_options, allocator, &candidate);
  if (iree_status_is_ok(status) && candidate.emitted) {
    status = loom_compile_write_bytes(
        iree_make_cstring_view(FLAG_output),
        iree_make_const_byte_span(candidate.archive.data,
                                  candidate.archive.data_length),
        allocator);
  }
  if (iree_status_is_ok(status)) {
    *out_emitted = candidate.emitted;
  }
  loom_ireevm_run_candidate_deinitialize(&candidate);
  return status;
#else
  (void)run_module;
  (void)compile_options;
  (void)allocator;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "loom-compile was built without VM emission");
#endif  // LOOM_COMPILE_HAVE_IREE_VM
}

static iree_string_view_t loom_compile_target_artifact_identifier(
    iree_string_view_t output_path, const loom_target_emitter_t* emitter) {
  return loom_tooling_file_path_is_stdio(output_path)
             ? emitter->default_identifier
             : output_path;
}

#if LOOM_COMPILE_HAVE_LLVMIR
static bool loom_compile_target_emitter_is_llvmir(
    const loom_target_emitter_t* target_emitter) {
  switch (target_emitter->target_artifact_format) {
    case LOOM_TARGET_ARTIFACT_FORMAT_LLVMIR_TEXT:
    case LOOM_TARGET_ARTIFACT_FORMAT_LLVMIR_BITCODE:
      return true;
    default:
      return false;
  }
}
#endif  // LOOM_COMPILE_HAVE_LLVMIR

static iree_status_t loom_compile_emit_target(
    const loom_run_execution_environment_t* environment,
    loom_run_session_t* session, const loom_target_emitter_t* target_emitter,
    loom_run_module_t* run_module,
    const loom_run_candidate_compile_options_t* compile_options,
    iree_allocator_t allocator, bool* out_emitted) {
  *out_emitted = false;
  if (target_emitter == NULL || target_emitter->emit == NULL) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "selected target emitter is incomplete");
  }
  if (!iree_string_view_is_empty(
          iree_make_cstring_view(FLAG_emit_target_artifact))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--emit-target-artifact is only valid for HAL artifact providers");
  }
  if (compile_options->artifact_manifest.mode !=
      LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--artifact-manifest is only valid for HAL artifact providers");
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
#if LOOM_COMPILE_HAVE_LLVMIR
  const loom_llvmir_target_profile_provider_t*
      llvmir_target_profile_providers[] = {
#if LOOM_COMPILE_HAVE_LLVMIR_X86_TARGET_ENV
          loom_llvmir_x86_target_profile_provider(),
#endif  // LOOM_COMPILE_HAVE_LLVMIR_X86_TARGET_ENV
#if LOOM_COMPILE_HAVE_LLVMIR_AMDGPU_TARGET_ENV
          loom_llvmir_amdgpu_target_profile_provider(),
#endif  // LOOM_COMPILE_HAVE_LLVMIR_AMDGPU_TARGET_ENV
          NULL,
      };
  const loom_llvmir_target_profile_registry_t llvmir_target_profile_registry = {
      .providers = llvmir_target_profile_providers,
      .provider_count = IREE_ARRAYSIZE(llvmir_target_profile_providers) - 1,
  };
  loom_llvmir_artifact_emitter_options_t llvmir_options = {0};
  const void* option_chain = NULL;
  if (loom_compile_target_emitter_is_llvmir(target_emitter)) {
    loom_llvmir_artifact_emitter_options_initialize(&llvmir_options);
    llvmir_options.target_profile_registry = &llvmir_target_profile_registry;
    option_chain = &llvmir_options;
  }
#else
  const void* option_chain = NULL;
#endif  // LOOM_COMPILE_HAVE_LLVMIR
  if (compile_options->report != NULL) {
    compile_options->report->artifact_kind =
        LOOM_TARGET_COMPILE_ARTIFACT_KIND_TARGET_ARTIFACT;
    compile_options->report->backend_name = target_emitter->name;
    compile_options->report->artifact_format = loom_target_artifact_format_name(
        target_emitter->target_artifact_format);
  }
  const loom_target_emit_request_t request = {
      .target_environment =
          loom_run_execution_environment_target_environment(environment),
      .low_descriptor_registry =
          &loom_run_session_low_descriptor_registry(session)->registry,
      .module = run_module->module,
      .function_versions = compile_options->function_versions,
      .option_chain = option_chain,
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
          iree_io_byte_sequence_length(artifact.contents));
    }
  }
  if (iree_status_is_ok(status) && diagnostic_emitter.error_count == 0 &&
      artifact.contents != NULL &&
      iree_io_byte_sequence_length(artifact.contents) != 0) {
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

static iree_status_t loom_compile_append_backend_name(
    iree_string_builder_t* builder, iree_string_view_t name,
    bool* inout_needs_separator) {
  if (*inout_needs_separator) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ", "));
  }
  *inout_needs_separator = true;
  return iree_string_builder_append_string(builder, name);
}

static iree_status_t loom_compile_append_hal_backend_names(
    const loom_run_hal_artifact_provider_registry_t* artifact_provider_registry,
    iree_string_builder_t* builder, bool* inout_needs_separator) {
  for (iree_host_size_t i = 0; i < artifact_provider_registry->provider_count;
       ++i) {
    IREE_RETURN_IF_ERROR(loom_compile_append_backend_name(
        builder, artifact_provider_registry->providers[i]->name,
        inout_needs_separator));
  }
  return iree_ok_status();
}

static iree_status_t loom_compile_append_target_emitter_names(
    const loom_target_environment_t* target_environment,
    iree_string_builder_t* builder, bool* inout_needs_separator) {
  const loom_target_emitter_list_t emitters =
      loom_target_environment_emitter_list(target_environment);
  for (iree_host_size_t i = 0; i < emitters.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_compile_append_backend_name(
        builder, emitters.values[i]->public_artifact_format,
        inout_needs_separator));
  }
  return iree_ok_status();
}

static iree_status_t loom_compile_make_unknown_backend_status(
    iree_string_view_t backend_name,
    const loom_run_hal_artifact_provider_registry_t* artifact_provider_registry,
    const loom_target_environment_t* target_environment,
    iree_allocator_t allocator) {
  iree_string_builder_t backend_names;
  iree_string_builder_initialize(allocator, &backend_names);
  bool needs_separator = false;
  iree_status_t status = loom_compile_append_backend_name(
      &backend_names, IREE_SV("command"), &needs_separator);
  if (iree_status_is_ok(status)) {
    status = loom_compile_append_backend_name(&backend_names, IREE_SV("vm"),
                                              &needs_separator);
  }
  if (iree_status_is_ok(status)) {
    status = loom_compile_append_hal_backend_names(
        artifact_provider_registry, &backend_names, &needs_separator);
  }
  if (iree_status_is_ok(status)) {
    status = loom_compile_append_target_emitter_names(
        target_environment, &backend_names, &needs_separator);
  }
  if (!iree_status_is_ok(status)) {
    iree_string_builder_deinitialize(&backend_names);
    return status;
  }
  status = iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "unknown --backend='%.*s'; expected registered backend in [%.*s]",
      (int)backend_name.size, backend_name.data,
      (int)iree_string_builder_size(&backend_names),
      iree_string_builder_buffer(&backend_names));
  iree_string_builder_deinitialize(&backend_names);
  return status;
}

static iree_status_t loom_compile_select_backend(
    iree_string_view_t backend_name,
    const loom_run_hal_artifact_provider_registry_t* artifact_provider_registry,
    const loom_target_environment_t* target_environment,
    iree_allocator_t allocator, loom_compile_backend_t* out_backend) {
  *out_backend = (loom_compile_backend_t){0};
  const loom_run_hal_artifact_provider_t* artifact_provider =
      loom_run_hal_artifact_provider_registry_lookup(artifact_provider_registry,
                                                     backend_name);
  if (artifact_provider != NULL) {
    out_backend->hal_artifact_provider = artifact_provider;
    return iree_ok_status();
  }
  if (iree_string_view_equal(backend_name, IREE_SV("command"))) {
    out_backend->is_command_backend = true;
    return iree_ok_status();
  }
  const loom_target_emitter_list_t emitters =
      loom_target_environment_emitter_list(target_environment);
  for (iree_host_size_t i = 0; i < emitters.count; ++i) {
    const loom_target_emitter_t* emitter = emitters.values[i];
    if (!iree_string_view_equal(emitter->public_artifact_format,
                                backend_name)) {
      continue;
    }
    if (out_backend->target_emitter != NULL) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "target environment contains duplicate emitter artifact formats");
    }
    out_backend->target_emitter = emitter;
  }
  if (out_backend->target_emitter != NULL) {
    return iree_ok_status();
  }
  if (iree_string_view_equal(backend_name, IREE_SV("vm"))) {
    out_backend->is_vm_backend = true;
    return iree_ok_status();
  }
  return loom_compile_make_unknown_backend_status(
      backend_name, artifact_provider_registry, target_environment, allocator);
}

static iree_status_t loom_compile_validate_backend_flags(
    const loom_compile_backend_t* backend) {
  const iree_string_view_t command_artifact_directory =
      iree_make_cstring_view(FLAG_emit_command_artifacts);
  const iree_string_view_t kernel_request_directory =
      iree_make_cstring_view(FLAG_emit_kernel_requests);
  if (!backend->is_command_backend) {
    if (!iree_string_view_is_empty(command_artifact_directory)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "--emit-command-artifacts requires --backend=command");
    }
    if (!iree_string_view_is_empty(kernel_request_directory)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--emit-kernel-requests requires "
                              "--backend=command");
    }
    return iree_ok_status();
  }

  if (iree_string_view_is_empty(command_artifact_directory) ||
      loom_tooling_file_path_is_stdio(command_artifact_directory)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--backend=command requires --emit-command-artifacts=<directory>");
  }
  if (!iree_string_view_is_empty(
          iree_make_cstring_view(FLAG_emit_target_artifact))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--emit-target-artifact is only valid for HAL artifact providers");
  }
  if (!iree_string_view_is_empty(kernel_request_directory) &&
      loom_tooling_file_path_is_stdio(kernel_request_directory)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--emit-kernel-requests requires a filesystem directory");
  }
  return iree_ok_status();
}

static iree_status_t loom_compile_select_explicit_hal_target(
    const loom_run_hal_artifact_provider_t* artifact_provider,
    iree_allocator_t allocator, bool* out_target_selected,
    loom_run_hal_device_target_t* out_target) {
  *out_target_selected = false;
  *out_target = (loom_run_hal_device_target_t){0};

  const iree_string_view_t target_key =
      iree_string_view_trim(iree_make_cstring_view(FLAG_target));
  if (iree_string_view_is_empty(target_key)) {
    return iree_ok_status();
  }
  if (artifact_provider == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--target is only valid for HAL artifact backends");
  }
  IREE_RETURN_IF_ERROR(loom_run_hal_artifact_provider_select_target_key(
      artifact_provider, target_key, allocator, out_target));
  *out_target_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_compile_specialize_explicit_hal_target(
    const loom_run_execution_environment_t* environment,
    loom_run_session_t* session, loom_run_module_t* run_module,
    const loom_run_hal_device_target_t* target,
    loom_run_compile_report_capture_t* compile_report_capture,
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
      loom_run_execution_environment_target_environment(environment),
      target->target_profile, loom_target_entry_emitter(&diagnostic_emitter),
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

static iree_status_t loom_compile_require_hal_kernel_targets(
    const loom_run_hal_artifact_provider_t* artifact_provider,
    const loom_run_hal_device_target_t* explicit_target,
    loom_module_t* module) {
  if (artifact_provider == NULL || explicit_target != NULL) {
    return iree_ok_status();
  }
  uint32_t unselected_root_count = 0;
  if (module == NULL || module->body == NULL ||
      module->body->block_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HAL target validation requires a module with a body block");
  }
  loom_op_t* op = NULL;
  loom_block_for_each_op(loom_module_block(module), op) {
    const loom_func_like_t function = loom_func_like_cast(module, op);
    if (loom_func_like_is_kernel_entry(function) &&
        !loom_symbol_ref_is_valid(loom_func_like_target(function))) {
      ++unselected_root_count;
    }
  }
  if (unselected_root_count == 0) {
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "HAL backend '%.*s' requires --target= when %u kernel.def root%s omit "
      "target(...) attrs",
      (int)artifact_provider->name.size, artifact_provider->name.data,
      (unsigned)unselected_root_count, unselected_root_count == 1 ? "" : "s");
}

static void loom_compile_print_agents_markdown(FILE* stream) {
  fprintf(
      stream,
      "## loom-compile\n"
      "\n"
      "`loom-compile` turns Loom text or bytecode into runtime artifacts. It "
      "is\n"
      "the offline path for producing portable command programs, VM bytecode "
      "archives,\n"
      "and loader-ready HAL executables such as AMDGPU HSACO.\n"
      "\n"
      "### Common flows\n"
      "\n"
      "```shell\n"
      "loom-compile kernel.loom --backend=vm --output=kernel.vmfb\n"
      "loom-compile model.loom --backend=command --output=commands.json \\\n"
      "  --emit-command-artifacts=commands/ \\\n"
      "  --emit-kernel-requests=kernel-requests/\n"
      "loom-compile kernel.loom --backend=amdgpu-hal "
      "--target=gfx11-generic \\\n"
      "  --output=kernel.hsaco\n"
      "loom-compile catalog.loom --root=@entry --backend=amdgpu-hal \\\n"
      "  --target=gfx11-generic --output=entry.hsaco\n"
      "loom-compile kernel.loom --backend=llvmir-text --output=kernel.ll\n"
      "loom-compile kernel.loom --backend=llvmir-bitcode --output=kernel.bc\n"
      "loom-compile kernel.loombc --backend=amdgpu-hal "
      "--target=gfx11-generic \\\n"
      "  --artifact-manifest=summary --emit-artifact-manifest=kernel.json\n"
      "loom-compile kernel.loom --backend=vm --pipeline=none\n"
      "loom-compile kernel.loom --backend=vm --pipeline=@my_pipeline\n"
      "```\n"
      "\n"
      "### Backend and target specialization\n"
      "\n"
      "`--backend=command` emits one portable `.loomcmd` file per public or "
      "retained command program into `--emit-command-artifacts`, and writes "
      "their shared entry-requirement manifest to `--output`. Repeated "
      "`--root=@symbol` values select an explicit subset; omitting roots emits "
      "all exported command programs. `--emit-kernel-requests` additionally "
      "streams one ordinary Loom bytecode module per reachable semantic kernel "
      "class. The manifest maps source-backed entries to those files; bodyless "
      "external entries remain binding requirements without source access.\n"
      "`--backend=vm` emits a VM bytecode archive. HAL/native backends such as "
      "`--backend=amdgpu-hal` write the selected provider's loader-ready\n"
      "executable bytes to `--output`. `--emit-target-artifact=path` "
      "additionally\n"
      "writes the provider's target-native artifact when requested. The two\n"
      "representations may be identical; AMDGPU currently uses HSACO for "
      "both.\n"
      "Use a generic target such\n"
      "as `--target=gfx11-generic` for portable code, or an exact target such\n"
      "as `--target=gfx1151` to specialize every materialized HAL kernel "
      "entry\n"
      "before compilation. Authored targets are retained as compatibility\n"
      "requirements, so a generic `gfx11-generic` kernel can specialize to\n"
      "`gfx1151` while an incompatible family fails. `--target` is required\n"
      "only when a materialized HAL kernel entry has no authored target.\n"
      "When `--root` is omitted, a HAL backend compiles every kernel entry "
      "and\n"
      "its dependency closure; authoring-only check programs and unrelated "
      "host\n"
      "code are not part of the executable library.\n"
      "Target-owned emitters such as `--backend=llvmir-text` and\n"
      "`--backend=llvmir-bitcode` write the artifact directly to `--output`.\n"
      "Use repeated `--root=@symbol` values to compile selected entries from "
      "a\n"
      "catalog-like module without first producing a linked bytecode file.\n"
      "\n"
      "### Config specialization\n"
      "\n"
      "`--config=key=value` and `--config-file=file.jsonc` bind `config.decl`\n"
      "values before the compile pipeline. Use this for JIT shape, layout, "
      "and\n"
      "provider-selection parameters that should specialize the artifact.\n"
      "\n"
      "### Reports, manifests, and IR traces\n"
      "\n"
      "```shell\n"
      "loom-compile kernel.loom --backend=amdgpu-hal "
      "--target=gfx11-generic \\\n"
      "  --compile-report=summary --compile-report-output=report.json \\\n"
      "  --artifact-manifest=details --emit-artifact-manifest=manifest.json "
      "\\\n"
      "  --output=kernel.hsaco\n"
      "loom-compile-report show report.json\n"
      "loom-compile-report diff baseline.json report.json --format=json\n"
      "loom-compile-report suggest report.json --format=json\n"
      "loom-compile kernel.loom --backend=amdgpu-hal "
      "--target=gfx11-generic \\\n"
      "  --dump-ir-after-all --dump-ir-format=jsonl "
      "--dump-ir-output=trace.jsonl\n"
      "jq '.functions[] | {name, target, workgroup_size}' manifest.json\n"
      "jq '{function, lowered, target_export, "
      "code_byte_count:.emission.code_byte_count, "
      "source_packets:(.economics.memory.source_low.packet_count // null), "
      "source_unknown_dynamic_packets:"
      "(.economics.memory.source_low.unknown_dynamic_packet_count // null), "
      "source_issued_bytes:"
      "(.economics.memory.source_low.dispatch_issued.total_bytes // null), "
      "source_logical_bytes:"
      "(.economics.memory.source_low.dispatch_source.total_bytes // null), "
      "source_dynamic_bytes:"
      "(.economics.memory.source_low.dynamic_source_byte_count // null), "
      "source_read_dynamic_bytes:"
      "(.economics.memory.source_low.dynamic_read_byte_count // null), "
      "source_write_dynamic_bytes:"
      "(.economics.memory.source_low.dynamic_write_byte_count // null), "
      "source_unique_bytes:"
      "(.economics.memory.source_low.interval_envelope.unique_byte_count // "
      "null), "
      "source_read_unique_bytes:"
      "(.economics.memory.source_low.read_interval_envelope."
      "unique_byte_count // null), "
      "source_write_unique_bytes:"
      "(.economics.memory.source_low.write_interval_envelope."
      "unique_byte_count // null), "
      "low_dynamic_issued_bytes:"
      "(.economics.memory.dispatch_issued.total_bytes // null), "
      "valu:.static_instruction_mix.vector_alu_count, "
      "matrix:.static_instruction_mix.matrix_count, "
      "wmma:.static_instruction_mix.wmma_count, "
      "mfma:.static_instruction_mix.mfma_count, "
      "planned_spills:.allocation.spill_count, "
      "planned_spill_slots:.allocation.spill_plan_count, "
      "final_vgprs:.target_resources.vector.final.register_count, "
      "scheduled_vgpr_pressure:"
      ".target_resources.vector.scheduled_pressure.peak_live_units, "
      "materialized_spill_storage:"
      ".allocation.materialized_spill_storage_count, "
      "materialized_spill_stores:"
      ".allocation.materialized_spill_store_count, "
      "materialized_reloads:.allocation.materialized_reload_count, "
      "private:.memory.private_bytes}, "
      "(.entries.rows[]? | {function, source_function, "
      "target_export_symbol, code_byte_count, "
      "low_dynamic_issued_bytes:"
      "(.economics.memory.dispatch_issued.total_bytes // null), "
      "valu:.static_instruction_mix.vector_alu_count, "
      "matrix:.static_instruction_mix.matrix_count, "
      "wmma:.static_instruction_mix.wmma_count, "
      "mfma:.static_instruction_mix.mfma_count, "
      "planned_spills:.allocation_spill_count, "
      "planned_spill_slots:.allocation_spill_plan_count, "
      "final_vgprs:.target_resources.vector.final.register_count, "
      "scheduled_vgpr_pressure:"
      ".target_resources.vector.scheduled_pressure.peak_live_units, "
      "materialized_spill_storage:"
      ".allocation_materialized_spill_storage_count, "
      "materialized_spill_stores:"
      ".allocation_materialized_spill_store_count, "
      "materialized_reloads:.allocation_materialized_reload_count, "
      "private:.private_memory_bytes})' "
      "report.json\n"
      "jq --arg f branchy 'select(.function == $f or .lowered == $f), "
      "(.entries.rows[]? | select(.function == $f or .source_function == "
      "$f))' report.json\n"
      "jq --arg f branchy '.. | objects | select(.function? == $f or "
      ".source_function? == $f or .function_name? == $f)' report.json\n"
      "jq '.pressure_origin_rows.rows[]? | "
      "{function,register_class,peak_block,peak_operation,origin,"
      "origin_operation,semantic_tag,sample_value,live_units,live_values} | "
      "with_entries(select(.value != null))' report.json\n"
      "jq '.allocation_high_water_rows.rows[]? | "
      "{function,value,register_class,origin,semantic_tag,required_unit_count,"
      "location_base,location_count,high_water_units,"
      "lower_largest_free_run_unit_count,"
      "lower_pressure_releasable_largest_free_run_unit_count,"
      "active_assignment_blocker_units,active_storage_lease_blocker_units} | "
      "with_entries(select(.value != null))' report.json\n"
      "jq '.spill_rows.rows[]? | "
      "{function,kind,value,register_class,origin,semantic_tag,byte_size,"
      "store_count,store_bytes,reload_count,reload_bytes} | "
      "with_entries(select(.value != null))' report.json\n"
      "jq '[.spill_rows.rows[]?] | "
      "group_by(.origin + \":\" + (.semantic_tag // \"\"))[] | "
      "{count:length,function:.[0].function,origin:.[0].origin,"
      "semantic_tag:.[0].semantic_tag,bytes:(map(.byte_size)|add),"
      "stores:(map(.store_count)|add),"
      "reloads:(map(.reload_count)|add)} | "
      "with_entries(select(.value != null))' report.json\n"
      "jq '.target_capability_rows.rows[]? | select(.namespace == "
      "\"amdgpu\" or .namespace == \"target\") | {function, namespace, "
      "key, value_kind, value_u64, value_bool, value_string} | "
      "with_entries(select(.value != null))' report.json\n"
      "jq '.target_capability_rows.rows[]? | "
      "select(.namespace == \"amdgpu\" and "
      "(.key | startswith(\"matrix_\") and endswith(\"_native_kind\"))) | "
      "{function,key,native_kind:.value_string} | "
      "with_entries(select(.value != null))' report.json\n"
      "jq '.source_low.selection_summaries.rows[]? | "
      "{function,source_op,selection,plan_key,descriptor_key,"
      "descriptor_semantic_tag,selected_op_count,emitted_low_op_count} | "
      "with_entries(select(.value != null))' report.json\n"
      "jq '[.source_low.selection_summaries.rows[]? | "
      "{function,source_op,selection,plan_key,descriptor_key,"
      "descriptor_semantic_tag,selected_op_count,emitted_low_op_count} | "
      "with_entries(select(.value != null))] | "
      "sort_by(.emitted_low_op_count // 0) | reverse | .[:20][]' "
      "report.json\n"
      "jq '.source_low.selection_summaries.rows[]? | "
      "select((.descriptor_semantic_tag // \"\") | "
      "startswith(\"matrix.\")) | "
      "{function,source_op,plan_key,descriptor_key,"
      "descriptor_semantic_tag,selected_op_count,emitted_low_op_count} | "
      "with_entries(select(.value != null))' report.json\n"
      "jq '.schedule_band_summary_rows.rows[]? | "
      "select((.semantic_tag // \"\") | startswith(\"matrix.\")) | "
      "{function,block,semantic_tag,band_count,node_count,"
      "matrix:.static_instruction_mix.matrix_count,"
      "wmma:.static_instruction_mix.wmma_count,"
      "mfma:.static_instruction_mix.mfma_count,"
      "swmmac:.static_instruction_mix.swmmac_count,"
      "smfmac:.static_instruction_mix.smfmac_count,"
      "dynamic_matrix:(.dynamic_instruction_mix.matrix_count // null),"
      "dynamic_wmma:(.dynamic_instruction_mix.wmma_count // null),"
      "dynamic_mfma:(.dynamic_instruction_mix.mfma_count // null),"
      "dynamic_swmmac:(.dynamic_instruction_mix.swmmac_count // null),"
      "dynamic_smfmac:(.dynamic_instruction_mix.smfmac_count // null),"
      "result_unit_count} | "
      "with_entries(select(.value != null))' report.json\n"
      "jq '.schedule_band_summary_rows.rows[]? | "
      "select(((.semantic_tag // \"\") | startswith(\"memory.global.\")) "
      "or ((.semantic_tag // \"\") | startswith(\"memory.buffer.\"))) | "
      "{function,block,semantic_tag,band_count,node_count,"
      "max_band_node_count,"
      "global_loads:.static_instruction_mix.global_load_count,"
      "global_stores:.static_instruction_mix.global_store_count,"
      "buffer_loads:.static_instruction_mix.buffer_load_count,"
      "buffer_stores:.static_instruction_mix.buffer_store_count,"
      "dynamic_global_loads:"
      "(.dynamic_instruction_mix.global_load_count // null),"
      "dynamic_global_stores:"
      "(.dynamic_instruction_mix.global_store_count // null),"
      "dynamic_buffer_loads:"
      "(.dynamic_instruction_mix.buffer_load_count // null),"
      "dynamic_buffer_stores:"
      "(.dynamic_instruction_mix.buffer_store_count // null)} | "
      "with_entries(select(.value != null))' report.json\n"
      "jq '.wait_reason_summary_rows.rows[]? | "
      "{function,counter,reason,"
      "action_count:.summary.action_count,"
      "full_drains:.summary.full_drain_count,"
      "partial_waits:.summary.partial_wait_count,"
      "max_outstanding_before:.summary.max_outstanding_before,"
      "max_drained_count:.summary.max_drained_count} | "
      "with_entries(select(.value != null))' report.json\n"
      "jq '[.wait_reason_summary_rows.rows[]?] | "
      "sort_by(-.summary.full_drain_count)[] | "
      "{function,counter,reason,"
      "full_drains:.summary.full_drain_count,"
      "partial_waits:.summary.partial_wait_count,"
      "max_outstanding_before:.summary.max_outstanding_before,"
      "max_drained_count:.summary.max_drained_count} | "
      "with_entries(select(.value != null))' report.json\n"
      "jq '.wait_action_rows.rows[]? | "
      "{function,counter,action,reason,scheduled_ordinal,"
      "producer:.producer_semantic_tag,consumer:.consumer_semantic_tag,"
      "target_count,outstanding_before,outstanding_after,drained_count} | "
      "with_entries(select(.value != null))' report.json\n"
      "jq '.source_low.memory.strategies[]? | "
      "{function,memory_space,operation,packet,strategy,fallback_reason,"
      "storage,"
      "packet_count,dispatch_source,dispatch_issued,scalar_packet_count,"
      "vector_packet_count} | "
      "with_entries(select(.value != null))' report.json\n"
      "jq '[.source_low.memory.argument_packets[]? | "
      "{function,source_root,source_root_argument_index,memory_space,"
      "operation,packet,strategy,fallback_reason,storage,"
      "packet_count,load_packet_count,store_packet_count,"
      "dispatch_source,dispatch_issued,dynamic_source_byte_count,"
      "dynamic_read_byte_count,dynamic_write_byte_count,"
      "dynamic_issued_read_byte_count,dynamic_issued_write_byte_count,"
      "scalar_packet_count,vector_packet_count,"
      "read_unique_bytes:.read_interval_envelope.unique_byte_count,"
      "write_unique_bytes:.write_interval_envelope.unique_byte_count} | "
      "with_entries(select(.value != null))] | "
      "sort_by(-(.dispatch_issued.total_bytes // 0))[]' report.json\n"
      "jq '[.source_low.memory.arguments[]? | "
      "{function,source_root,source_root_argument_index,memory_space,"
      "packet_count,load_packet_count,store_packet_count,"
      "dispatch_source,dispatch_issued,dynamic_source_byte_count,"
      "dynamic_read_byte_count,dynamic_write_byte_count,"
      "dynamic_issued_read_byte_count,dynamic_issued_write_byte_count,"
      "scalar_packet_count,"
      "vector_packet_count,"
      "read_unique_bytes:.read_interval_envelope.unique_byte_count,"
      "write_unique_bytes:.write_interval_envelope.unique_byte_count} | "
      "with_entries(select(.value != null))] | "
      "sort_by(-(.dispatch_issued.total_bytes // 0))[]' report.json\n"
      "jq 'select(.stage == \"prepared-low\") | .pass' trace.jsonl\n"
      "jq -r 'select(.stage == \"prepared-low\") | .ir' trace.jsonl\n"
      "```\n"
      "\n"
      "`--compile-report=summary|details` records compiler-side facts and\n"
      "status. `loom-compile-report show|diff|suggest` provides compact\n"
      "same-checkout views, exact comparison, and target-owned optimization\n"
      "experiments. Version-zero reports are ephemeral diagnostics; "
      "regenerate\n"
      "them with the current compiler instead of carrying compatibility "
      "paths.\n"
      "`--artifact-manifest=summary|details|analysis` records the\n"
      "artifact's functions, globals, targets, ABI metadata, and optional\n"
      "analysis. `--dump-ir-*` captures intermediate IR with the same tracing\n"
      "flags used by `loom-opt`. JSONL trace rows embed whole-module "
      "snapshots\n"
      "in `.ir`, even when an individual pass is anchored on a function.\n"
      "Compile reports are designed for post-run slicing. For a single entry,\n"
      "top-level `function` and `lowered` identify the selected entry. For a\n"
      "library or multi-entry artifact, `.entries.rows[]` is the stable\n"
      "per-entry summary index; those rows use compact flat summary field\n"
      "names for counts such as `allocation_spill_count` while top-level\n"
      "report sections keep their richer nested structure. Detail mode keeps\n"
      "row arrays such as\n"
      "`pressure_rows.rows[]`, `pressure_origin_rows.rows[]`,\n"
      "`allocation_high_water_rows.rows[]`, `spill_rows.rows[]`,\n"
      "`source_low.rows[]`, `source_low.memory_rows[]`,\n"
      "`math_legalization.rows[]`, and `target_legalization.rows[]`.\n"
      "When summary reports show materialized spill storage, materialized\n"
      "reloads, private memory, or a scheduled-pressure/final-register cliff,\n"
      "rerun with `--compile-report=details` to attribute the pressure peak,\n"
      "high-water placement, and private spill traffic. Summary and detail\n"
      "reports aggregate\n"
      "source-low selections under `.source_low.selection_summaries.rows[]`\n"
      "so large generated packet families can be ranked without detail rows.\n"
      "When a branch, tail peel, or recurrence rewrite suddenly produces many\n"
      "spill diagnostics, group `spill_rows.rows[]` by `origin` and\n"
      "`semantic_tag` before reading individual rows; repeated diagnostics at\n"
      "the same source line usually mean one high-level live-range family, "
      "not\n"
      "hundreds of unrelated bugs.\n"
      "Matrix-family schedule evidence is available in summary mode under\n"
      "`.schedule_band_summary_rows.rows[]`; those rows preserve semantic\n"
      "tags, band/node counts, matrix-family instruction counts, and result\n"
      "unit pressure for comparing Loom output with external GEMM families.\n"
      "When exact fixed-trip loop counts are proven, the same rows include\n"
      "`dynamic_instruction_mix` so repeated matrix packets can be compared\n"
      "without joining against lower-level selection rows.\n"
      "For memory-schedule triage, filter the same schedule-band rows to\n"
      "`memory.global.*`/`memory.buffer.*`: `max_band_node_count` is the\n"
      "widest consecutive load/store band, while `dynamic_instruction_mix`\n"
      "shows exact fixed-trip memory packet counts when proven. Detailed\n"
      "wait density is summarized in compact reports under\n"
      "`.wait_reason_summary_rows.rows[]`; use that first to rank counters\n"
      "and reasons. Detailed `wait_action_rows.rows[]` identify target\n"
      "wait/drain placement with producer and consumer semantic tags; request "
      "`--compile-report=details`\n"
      "when those per-wait rows are needed.\n"
      "Source-low memory economics are summarized under\n"
      "`.economics.memory.source_low`, with dispatch-scaled source and issued\n"
      "bytes, direct dynamic source/read/write byte counts, and interval\n"
      "envelopes when dynamic packet counts are proven. Unknown dynamic\n"
      "coverage is explicit, for example `unknown_dynamic_packet_count`.\n"
      "Per-binding source memory economics are under\n"
      "`.source_low.memory.arguments[]`. The cross product of binding/root,\n"
      "operation, target packet, strategy, fallback reason, and storage\n"
      "contract is under `.source_low.memory.argument_packets[]`; use that\n"
      "row first when comparing which binding paid for a packed FP8 decode,\n"
      "DPP-packed BF16 store, scalarized load, or other target packet family.\n"
      "Source-root and target-strategy rollups are under\n"
      "`.source_low.memory.roots[]` and `.source_low.memory.strategies[]`\n"
      "when available.\n"
      "`economics.memory.per_workitem_issued` and\n"
      "`economics.memory.dispatch_issued` are the target-low dynamic\n"
      "instruction issued-memory estimates, including issued byte counts and\n"
      "nonzero instruction-family counts such as `global_load_count` and\n"
      "`buffer_load_count`. They are omitted when the compiler cannot prove\n"
      "that dynamic instruction mix.\n"
      "For FP8/BF8 checkpoint storage, a plain `f8E4M3`, `f8E5M2`, or BF8\n"
      "view describes the byte format but does not prove semantic payload\n"
      "properties. If imported or authored storage is known finite, put that\n"
      "truth on the physical storage schema with `rounding=finite_only`; if\n"
      "the storage contract also flushes subnormal payloads, use\n"
      "`rounding=finite_flush_subnormal`. Source-low memory strategy keys\n"
      "show which route was selected, for example\n"
      "`fp8_packed_bf16_decode_repair_zero_subnormal` versus\n"
      "`fp8_packed_bf16_decode_repair_zero`. Full decode rows carry a\n"
      "`fallback_reason` such as `missing_finite_not_subnormal` when storage\n"
      "facts are too weak for a cheaper packed route. Do not infer those "
      "facts\n"
      "from raw scalar type spelling.\n"
      "`target_resources.{scalar,vector}.final.register_count` records final\n"
      "target metadata used for occupancy. "
      "`target_resources.{scalar,vector}.scheduled_pressure.peak_live_units`\n"
      "records scheduled virtual pressure before final allocation metadata.\n"
      "Target-native reports may also include\n"
      "`target_capability_rows.rows[]` with selected processor, subgroup,\n"
      "index-width, and matrix-profile facts. AMDGPU narrow matrix support\n"
      "rows use `matrix_fp8_native_kind`, `matrix_bf8_native_kind`,\n"
      "`matrix_fp6_native_kind`, `matrix_bf6_native_kind`, and\n"
      "`matrix_fp4_native_kind` with values `none`, `unscaled`, `scaled`,\n"
      "or `unscaled_scaled`, so target-specialized kernels can distinguish\n"
      "software expansion, unscaled native packets, and scaled native\n"
      "packets without disassembly. Detail rows carry `function`,\n"
      "`source_function`, or diagnostic `function_name` fields so `jq` can\n"
      "slice one kernel out of a whole-module compile report without changing\n"
      "what was compiled.\n"
      "\n"
      "Prepared-low reproducers captured from `--dump-ir-after=source-to-low`\n"
      "or similar pass-boundary dumps should usually be replayed with\n"
      "`--pipeline=none`; the default pipeline expects source-level roots and\n"
      "will try to lower them again. HAL artifact emission still requires the\n"
      "prepared-low root to be ABI-complete, including target live-ins such "
      "as\n"
      "the AMDGPU kernarg segment pointer when the selected ABI uses "
      "kernargs.\n");
}

int main(int argc, char** argv) {
  iree_flags_set_usage(
      "loom-compile",
      "Compiles a Loom module to a runtime artifact.\n"
      "\n"
      "Usage:\n"
      "  loom-compile [file.loom] --backend=vm --output=module.vmfb\n"
      "  loom-compile [file.loom] --backend=command "
      "--output=commands.json --emit-command-artifacts=commands/ "
      "[--emit-kernel-requests=kernels/]\n"
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
  loom_run_execution_environment_t environment = {0};
  iree_status_t status = loom_run_execution_environment_initialize(
      &kLoomCompileProviderSet, &environment);
  loom_tooling_config_set_t config_set;
  loom_tooling_config_set_initialize(allocator, &config_set);

  iree_io_file_contents_t* contents = NULL;
  char* input_filename_storage = NULL;
  loom_run_session_t session = {0};
  loom_run_module_t run_module = {0};
  loom_run_compile_report_capture_options_t compile_report_options = {0};
  loom_run_compile_report_capture_t compile_report_capture = {0};
  loom_tooling_pass_trace_t pass_trace = {0};
  const loom_run_hal_artifact_provider_t* hal_artifact_provider = NULL;
  const loom_target_emitter_t* target_emitter = NULL;
  loom_run_candidate_artifact_manifest_options_t artifact_manifest_options = {
      0};
  iree_string_view_t artifact_manifest_output_path = iree_string_view_empty();
  char* artifact_manifest_output_path_storage = NULL;
  bool is_vm_backend = false;
  bool is_command_backend = false;
  loom_run_hal_device_target_t explicit_hal_target = {0};
  bool explicit_hal_target_selected = false;
  bool emitted = false;
  bool report_written = false;
  int exit_code = 0;
  const iree_string_view_t backend_name = iree_make_cstring_view(FLAG_backend);

  if (iree_status_is_ok(status)) {
    status = loom_compile_initialize_session(&environment, allocator, &session);
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
    status = loom_run_compile_report_capture_initialize(
        &compile_report_options, allocator, &compile_report_capture);
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_compile_report_capture_record_materialized_config(
        &compile_report_capture, run_module.module, &config_set);
  }
  if (iree_status_is_ok(status)) {
    loom_compile_backend_t backend = {0};
    status = loom_compile_select_backend(
        backend_name, &kLoomCompileHalArtifactProviderRegistry,
        loom_run_execution_environment_target_environment(&environment),
        allocator, &backend);
    hal_artifact_provider = backend.hal_artifact_provider;
    target_emitter = backend.target_emitter;
    is_vm_backend = backend.is_vm_backend;
    is_command_backend = backend.is_command_backend;
    if (iree_status_is_ok(status)) {
      status = loom_compile_validate_backend_flags(&backend);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_compile_select_explicit_hal_target(
        hal_artifact_provider, allocator, &explicit_hal_target_selected,
        &explicit_hal_target);
  }
  if (iree_status_is_ok(status) && explicit_hal_target_selected) {
    status = loom_compile_specialize_explicit_hal_target(
        &environment, &session, &run_module, &explicit_hal_target,
        &compile_report_capture, allocator);
  }
  if (iree_status_is_ok(status)) {
    status = loom_compile_select_roots(&session, &run_module,
                                       hal_artifact_provider, allocator);
  }
  if (iree_status_is_ok(status)) {
    status = loom_compile_artifact_manifest_options_initialize(
        &artifact_manifest_options, hal_artifact_provider != NULL, allocator,
        &artifact_manifest_output_path, &artifact_manifest_output_path_storage);
  }
  if (iree_status_is_ok(status)) {
    status = loom_compile_require_hal_kernel_targets(
        hal_artifact_provider,
        explicit_hal_target_selected ? &explicit_hal_target : NULL,
        run_module.module);
  }

  loom_run_candidate_compile_options_t compile_options = {0};
  loom_run_candidate_compile_options_initialize(&compile_options);
  loom_compile_pipeline_result_t pipeline_result = {0};
  compile_options.module_name = iree_make_cstring_view(FLAG_module_name);
  compile_options.artifact_manifest = artifact_manifest_options;
  if (hal_artifact_provider != NULL) {
    compile_options.target_pipeline_options =
        hal_artifact_provider->default_pipeline_options;
  }
  if (iree_status_is_ok(status)) {
    status = loom_compile_sanitizer_options_initialize(
        &compile_options.target_pipeline_options.sanitizer);
  }
  if (iree_status_is_ok(status)) {
    compile_options.source_resolver =
        loom_run_module_source_resolver(&run_module);
    loom_run_compile_report_capture_configure_compile_options(
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
            .active = loom_run_compile_report_capture_options_is_enabled(
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
    const bool run_pipeline =
        !is_command_backend || !loom_compile_pipeline_is_default(
                                   iree_make_cstring_view(FLAG_pipeline));
    if (run_pipeline) {
      status = loom_compile_run_pass_pipeline(
          &environment, &session, &run_module,
          LOOM_COMPILE_DEFAULT_PIPELINE_PREPARED_LOW, &compile_options,
          loom_tooling_pass_trace_options(&pass_trace), &pipeline_result);
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
  if (iree_status_is_ok(status) && exit_code == 0 && !is_command_backend) {
    status =
        loom_tooling_config_require_resolved_module(run_module.module, NULL);
  }

  if (iree_status_is_ok(status) && exit_code == 0) {
    if (hal_artifact_provider != NULL) {
      status = loom_compile_emit_hal(
          hal_artifact_provider,
          explicit_hal_target_selected ? &explicit_hal_target : NULL,
          &run_module, &compile_options, &compile_report_capture,
          artifact_manifest_output_path, allocator, &emitted, &report_written);
    } else if (target_emitter != NULL) {
      status = loom_compile_emit_target(&environment, &session, target_emitter,
                                        &run_module, &compile_options,
                                        allocator, &emitted);
    } else if (is_command_backend) {
      status = loom_compile_emit_command(&session, &run_module,
                                         &compile_options, allocator, &emitted);
    } else if (is_vm_backend) {
      status = loom_compile_emit_vm(&run_module, &compile_options, allocator,
                                    &emitted);
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

  if (explicit_hal_target_selected && hal_artifact_provider != NULL &&
      hal_artifact_provider->deinitialize_device_target != NULL) {
    hal_artifact_provider->deinitialize_device_target(
        hal_artifact_provider, &explicit_hal_target, allocator);
  }
  loom_compile_pipeline_result_deinitialize(&pipeline_result);
  loom_run_compile_report_capture_deinitialize(&compile_report_capture);
  loom_run_module_deinitialize(&run_module);
  iree_allocator_free(allocator, input_filename_storage);
  iree_io_file_contents_free(contents);
  loom_tooling_config_set_deinitialize(&config_set);
  loom_run_session_deinitialize(&session);
  loom_run_execution_environment_deinitialize(&environment);
  iree_allocator_free(allocator, artifact_manifest_output_path_storage);

  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
