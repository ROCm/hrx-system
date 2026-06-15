// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// iree-run-loom: compiles a Loom module and executes the selected export.

#include "loom/tools/iree-run-loom/main.h"

#include <stdio.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "loom/error/diagnostic.h"
#include "loom/ir/module.h"
#include "loom/tooling/cli/help.h"
#include "loom/tooling/compile/pipeline.h"
#include "loom/tooling/context/context.h"
#include "loom/tooling/execution/compile_report_capture.h"
#include "loom/tooling/execution/execution_backend.h"
#include "loom/tooling/execution/one_shot.h"
#include "loom/tooling/execution/session.h"
#include "loom/tooling/io/file.h"

IREE_FLAG(string, backend, "vm",
          "Compilation backend to run, such as 'vm' or a linked native "
          "backend.");
IREE_FLAG_NAMED(string, module_name, "module-name", "loom",
                "Module name to store in the compiled VM bytecode archive.");
IREE_FLAG(string, pipeline, "default",
          "Pass pipeline to run before execution. Use 'default' or empty for "
          "the comprehensive prepared-low pipeline, 'none' to disable pass "
          "execution, '@symbol' to run a module-local pass.pipeline, or a "
          "comma-separated pass list such as 'canonicalize,cse'.");
IREE_FLAG(string, function, "",
          "Function/export name to invoke. Empty selects the single VM export "
          "or HAL executable function.");
IREE_FLAG_LIST(string, input,
               "Appends a VM function input in IREE function I/O syntax.");
IREE_FLAG_LIST(string, output,
               "Appends a VM function output handling spec in IREE function "
               "I/O syntax. Empty prints all outputs.");
IREE_FLAG_LIST_NAMED(
    string, expected_output, "expected-output",
    "Appends an expected VM function output in IREE function I/O syntax. "
    "Expected outputs take precedence over --output.");
IREE_FLAG_NAMED(int32_t, output_max_element_count, "output-max-element-count",
                1024, "Maximum number of VM output elements to format.");
IREE_FLAG_NAMED(
    string, workgroup_count, "workgroup-count", "",
    "Optional HAL dispatch workgroup count as `x,y,z`. When omitted, a static "
    "kernel.launch.config workgroup count is used when available, otherwise "
    "one workgroup is dispatched.");
IREE_FLAG_NAMED(
    string, compile_report, "compile-report", "",
    "Optional compile report output. Use 'summary'/'details' for structured "
    "JSON, 'text-summary'/'text-details' for human-readable text, or "
    "empty/'none'.");
IREE_FLAG_NAMED(string, emit_target_artifact, "emit-target-artifact", "",
                "Optional output path for the selected HAL backend's "
                "target-native artifact, such as AMDGPU HSACO.");
IREE_FLAG_NAMED(string, emit_hal_executable, "emit-hal-executable", "",
                "Optional output path for the executable artifact passed to "
                "the HAL runtime loader.");
IREE_FLAG_NAMED(bool, emit_only, "emit-only", false,
                "Stops after HAL executable emission without dispatching.");
IREE_FLAG_NAMED(bool, probe_hal, "probe-hal", false,
                "Runs the selected backend's target probe, prints the result, "
                "and exits. Not all backends support probing.");

typedef struct iree_run_loom_hal_flag_state_t {
  // Dispatch constants in HAL ABI order.
  uint32_t constants[LOOM_RUN_ONE_SHOT_HAL_MAX_CONSTANT_COUNT];
  // Number of populated entries in |constants|.
  iree_host_size_t constant_count;
  // Binding specs in HAL binding ordinal order.
  iree_string_view_t binding_specs[LOOM_RUN_ONE_SHOT_HAL_MAX_BINDING_COUNT];
  // Calling-convention character for each binding spec.
  char binding_cconv[LOOM_RUN_ONE_SHOT_HAL_MAX_BINDING_COUNT];
  // Number of populated entries in |binding_specs|.
  int32_t binding_count;
  // Expected binding specs in HAL binding ordinal order.
  iree_string_view_t
      expected_binding_specs[LOOM_RUN_ONE_SHOT_HAL_MAX_BINDING_COUNT];
  // Calling-convention character for each expected binding spec.
  char expected_binding_cconv[LOOM_RUN_ONE_SHOT_HAL_MAX_BINDING_COUNT];
  // Number of populated entries in |expected_binding_specs|.
  int32_t expected_binding_count;
} iree_run_loom_hal_flag_state_t;

static iree_run_loom_hal_flag_state_t iree_run_loom_hal_flags = {0};

static iree_status_t iree_run_loom_parse_constant_flag(
    iree_string_view_t flag_name, void* storage, iree_string_view_t value) {
  (void)flag_name;
  (void)storage;
  if (iree_run_loom_hal_flags.constant_count >=
      LOOM_RUN_ONE_SHOT_HAL_MAX_CONSTANT_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "too many HAL dispatch constants; maximum is %d",
                            LOOM_RUN_ONE_SHOT_HAL_MAX_CONSTANT_COUNT);
  }
  uint32_t value_uint32 = 0;
  if (!iree_string_view_atoi_uint32(value, &value_uint32)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid HAL dispatch constant `%.*s`; expected "
                            "uint32_t",
                            (int)value.size, value.data);
  }
  iree_run_loom_hal_flags.constants[iree_run_loom_hal_flags.constant_count++] =
      value_uint32;
  return iree_ok_status();
}

static void iree_run_loom_print_constant_flag(iree_string_view_t flag_name,
                                              void* storage, FILE* file) {
  (void)storage;
  if (iree_run_loom_hal_flags.constant_count == 0) {
    fprintf(file, "# --%.*s=[integer value]\n", (int)flag_name.size,
            flag_name.data);
    return;
  }
  for (iree_host_size_t i = 0; i < iree_run_loom_hal_flags.constant_count;
       ++i) {
    fprintf(file, "--%.*s=%u\n", (int)flag_name.size, flag_name.data,
            iree_run_loom_hal_flags.constants[i]);
  }
}
IREE_FLAG_CALLBACK(iree_run_loom_parse_constant_flag,
                   iree_run_loom_print_constant_flag, NULL, constant,
                   "Appends a uint32_t HAL dispatch constant in ABI order.");

static iree_status_t iree_run_loom_parse_binding_flag(
    iree_string_view_t flag_name, void* storage, iree_string_view_t value) {
  (void)flag_name;
  (void)storage;
  if (iree_run_loom_hal_flags.binding_count >=
      LOOM_RUN_ONE_SHOT_HAL_MAX_BINDING_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "too many HAL bindings; maximum is %d",
                            LOOM_RUN_ONE_SHOT_HAL_MAX_BINDING_COUNT);
  }
  const int32_t index = iree_run_loom_hal_flags.binding_count++;
  iree_run_loom_hal_flags.binding_specs[index] = value;
  iree_run_loom_hal_flags.binding_cconv[index] = 'r';
  return iree_ok_status();
}

static void iree_run_loom_print_binding_flag(iree_string_view_t flag_name,
                                             void* storage, FILE* file) {
  (void)storage;
  if (iree_run_loom_hal_flags.binding_count == 0) {
    fprintf(file, "# --%.*s=\"shapextype[=values]\"\n", (int)flag_name.size,
            flag_name.data);
    return;
  }
  for (int32_t i = 0; i < iree_run_loom_hal_flags.binding_count; ++i) {
    iree_string_view_t binding_spec = iree_run_loom_hal_flags.binding_specs[i];
    fprintf(file, "--%.*s=\"%.*s\"\n", (int)flag_name.size, flag_name.data,
            (int)binding_spec.size, binding_spec.data);
  }
}
IREE_FLAG_CALLBACK(iree_run_loom_parse_binding_flag,
                   iree_run_loom_print_binding_flag, NULL, binding,
                   "Appends a HAL dispatch binding. Bindings use the same "
                   "shape/type/data syntax as iree-benchmark-executable.");

static iree_status_t iree_run_loom_parse_expected_binding_flag(
    iree_string_view_t flag_name, void* storage, iree_string_view_t value) {
  (void)flag_name;
  (void)storage;
  if (iree_run_loom_hal_flags.expected_binding_count >=
      LOOM_RUN_ONE_SHOT_HAL_MAX_BINDING_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "too many expected HAL bindings; maximum is %d",
                            LOOM_RUN_ONE_SHOT_HAL_MAX_BINDING_COUNT);
  }
  const int32_t index = iree_run_loom_hal_flags.expected_binding_count++;
  iree_run_loom_hal_flags.expected_binding_specs[index] = value;
  iree_run_loom_hal_flags.expected_binding_cconv[index] = 'r';
  return iree_ok_status();
}

static void iree_run_loom_print_expected_binding_flag(
    iree_string_view_t flag_name, void* storage, FILE* file) {
  (void)storage;
  if (iree_run_loom_hal_flags.expected_binding_count == 0) {
    fprintf(file, "# --%.*s=\"shapextype=values\"\n", (int)flag_name.size,
            flag_name.data);
    return;
  }
  for (int32_t i = 0; i < iree_run_loom_hal_flags.expected_binding_count; ++i) {
    iree_string_view_t binding_spec =
        iree_run_loom_hal_flags.expected_binding_specs[i];
    fprintf(file, "--%.*s=\"%.*s\"\n", (int)flag_name.size, flag_name.data,
            (int)binding_spec.size, binding_spec.data);
  }
}
IREE_FLAG_CALLBACK_NAMED(
    iree_run_loom_parse_expected_binding_flag,
    iree_run_loom_print_expected_binding_flag, NULL, expected_binding,
    "expected-binding",
    "Appends an expected HAL binding after dispatch. When present, one "
    "expected binding must be provided for every binding.");

static iree_status_t iree_run_loom_register_context(void* user_data,
                                                    loom_context_t* context) {
  const iree_run_loom_configuration_t* configuration =
      (const iree_run_loom_configuration_t*)user_data;
  IREE_RETURN_IF_ERROR(loom_tooling_context_register_tool_dialects(context));
  if (configuration->register_context.fn == NULL) {
    return iree_ok_status();
  }
  return configuration->register_context.fn(
      configuration->register_context.user_data, context);
}

static iree_status_t iree_run_loom_parse_workgroup_count(
    iree_string_view_t value, uint32_t* out_workgroup_count) {
  iree_string_view_t remaining = value;
  iree_string_view_t x;
  iree_string_view_split(remaining, ',', &x, &remaining);
  iree_string_view_t y;
  iree_string_view_split(remaining, ',', &y, &remaining);
  iree_string_view_t z = remaining;
  if (!iree_string_view_atoi_uint32(x, &out_workgroup_count[0]) ||
      !iree_string_view_atoi_uint32(y, &out_workgroup_count[1]) ||
      !iree_string_view_atoi_uint32(z, &out_workgroup_count[2])) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "invalid --workgroup-count='%.*s'; expected `x,y,z`", (int)value.size,
        value.data);
  }
  return iree_ok_status();
}

static iree_status_t iree_run_loom_validate_artifact_output_path(
    iree_string_view_t flag_name, iree_string_view_t path) {
  if (iree_string_view_is_empty(path)) {
    return iree_ok_status();
  }
  if (loom_tooling_file_path_is_stdio(path)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--%.*s must name a file path; stdout is reserved for textual run "
        "output",
        (int)flag_name.size, flag_name.data);
  }
  return iree_ok_status();
}

static iree_status_t iree_run_loom_one_shot_options_initialize(
    const loom_run_execution_backend_t* backend,
    loom_run_one_shot_options_t* out_options) {
  loom_run_one_shot_options_initialize(out_options);

  const iree_string_view_t target_artifact_output_path =
      iree_make_cstring_view(FLAG_emit_target_artifact);
  const iree_string_view_t hal_executable_output_path =
      iree_make_cstring_view(FLAG_emit_hal_executable);
  IREE_RETURN_IF_ERROR(iree_run_loom_validate_artifact_output_path(
      IREE_SV("emit-target-artifact"), target_artifact_output_path));
  IREE_RETURN_IF_ERROR(iree_run_loom_validate_artifact_output_path(
      IREE_SV("emit-hal-executable"), hal_executable_output_path));
  if (FLAG_output_max_element_count < 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--output-max-element-count must be non-negative; got %d",
        (int)FLAG_output_max_element_count);
  }

  if (iree_any_bit_set(backend->flags,
                       LOOM_RUN_EXECUTION_BACKEND_FLAG_VM_OPTIONS)) {
    out_options->vm_function_name = iree_make_cstring_view(FLAG_function);
    out_options->vm_inputs = (loom_run_one_shot_value_specs_t){
        .values = FLAG_input_list().values,
        .count = FLAG_input_list().count,
    };
    out_options->vm_outputs = (loom_run_one_shot_value_specs_t){
        .values = FLAG_output_list().values,
        .count = FLAG_output_list().count,
    };
    out_options->vm_expected_outputs = (loom_run_one_shot_value_specs_t){
        .values = FLAG_expected_output_list().values,
        .count = FLAG_expected_output_list().count,
    };
    out_options->vm_max_output_element_count =
        (iree_host_size_t)FLAG_output_max_element_count;
  }

  if (iree_any_bit_set(backend->flags,
                       LOOM_RUN_EXECUTION_BACKEND_FLAG_HAL_OPTIONS)) {
    const iree_string_view_t function_name =
        iree_make_cstring_view(FLAG_function);
    out_options->hal_function_name = !iree_string_view_is_empty(function_name)
                                         ? function_name
                                         : iree_string_view_empty();
    const iree_string_view_t workgroup_count =
        iree_make_cstring_view(FLAG_workgroup_count);
    if (!iree_string_view_is_empty(workgroup_count)) {
      IREE_RETURN_IF_ERROR(iree_run_loom_parse_workgroup_count(
          workgroup_count, out_options->hal_workgroup_count));
    }
    out_options->hal_constant_count = iree_run_loom_hal_flags.constant_count;
    memcpy(out_options->hal_constants, iree_run_loom_hal_flags.constants,
           out_options->hal_constant_count *
               sizeof(out_options->hal_constants[0]));
    out_options->hal_bindings = (loom_run_one_shot_binding_specs_t){
        .values = iree_run_loom_hal_flags.binding_specs,
        .conventions = iree_run_loom_hal_flags.binding_cconv,
        .count = iree_run_loom_hal_flags.binding_count,
    };
    out_options->hal_expected_bindings = (loom_run_one_shot_binding_specs_t){
        .values = iree_run_loom_hal_flags.expected_binding_specs,
        .conventions = iree_run_loom_hal_flags.expected_binding_cconv,
        .count = iree_run_loom_hal_flags.expected_binding_count,
    };
    out_options->hal_target_artifact_output_path = target_artifact_output_path;
    out_options->hal_executable_output_path = hal_executable_output_path;
    out_options->hal_emit_only = FLAG_emit_only;
    out_options->hal_max_output_element_count =
        (iree_host_size_t)FLAG_output_max_element_count;
  } else if (!iree_string_view_is_empty(target_artifact_output_path) ||
             !iree_string_view_is_empty(hal_executable_output_path) ||
             FLAG_emit_only) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--emit-target-artifact, --emit-hal-executable, and --emit-only "
        "require a HAL backend");
  }
  return iree_ok_status();
}

static iree_status_t iree_run_loom_compile_report_options_initialize(
    loom_run_compile_report_capture_options_t* out_options) {
  loom_run_compile_report_capture_options_initialize(out_options);
  IREE_RETURN_IF_ERROR(loom_run_compile_report_capture_options_parse_request(
      iree_make_cstring_view(FLAG_compile_report), out_options));
  return iree_ok_status();
}

static iree_status_t iree_run_loom_run_pass_pipeline(
    const iree_run_loom_configuration_t* configuration,
    loom_run_session_t* session, loom_run_module_t* run_module,
    const loom_run_candidate_compile_options_t* compile_options,
    loom_pass_run_result_t* out_run_result) {
  loom_compile_pipeline_options_t pipeline_options = {0};
  loom_compile_pipeline_options_initialize(&pipeline_options);
  pipeline_options.pipeline = iree_make_cstring_view(FLAG_pipeline);
  pipeline_options.target_environment = configuration->target_environment;
  pipeline_options.low_descriptor_registry =
      loom_run_session_low_descriptor_registry(session);
  pipeline_options.source_resolver =
      loom_run_module_source_resolver(run_module);
  pipeline_options.report = compile_options->report;
  pipeline_options.diagnostic_sink = (loom_diagnostic_sink_t){
      .fn = loom_diagnostic_stderr_sink,
  };
  return loom_compile_run_pipeline(run_module->module, &pipeline_options,
                                   loom_run_session_block_pool(session),
                                   out_run_result);
}

static iree_status_t iree_run_loom_make_unknown_backend_status(
    iree_string_view_t backend_name,
    const loom_run_execution_backend_registry_t* backend_registry,
    iree_allocator_t allocator) {
  iree_string_builder_t backend_names;
  iree_string_builder_initialize(allocator, &backend_names);
  iree_status_t status = loom_run_execution_backend_registry_format_names(
      backend_registry, &backend_names);
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

static void iree_run_loom_print_agents_markdown(FILE* stream) {
  fprintf(
      stream,
      "## iree-run-loom\n"
      "\n"
      "`iree-run-loom` compiles one Loom module and invokes one export. Use "
      "it\n"
      "for quick scalar VM checks, single-kernel HAL smoke tests, and "
      "emit-only\n"
      "artifact probes before moving a scenario into `check.case` or\n"
      "`check.benchmark`.\n"
      "\n"
      "### VM flow\n"
      "\n"
      "```shell\n"
      "iree-run-loom module.loom --backend=vm --function=branchy \\\n"
      "  --input=0 --input=21 --expected-output=42\n"
      "iree-run-loom module.loom --backend=vm --function=branchy \\\n"
      "  --input=50 --input=8 --output=-\n"
      "```\n"
      "\n"
      "VM inputs and outputs use IREE function I/O syntax. Empty `--function`\n"
      "selects the single export when the module has exactly one runnable\n"
      "export.\n"
      "\n"
      "### HAL flow\n"
      "\n"
      "```shell\n"
      "iree-run-loom kernel.loom --backend=amdgpu --function=q8_kernel \\\n"
      "  --binding=64xf32=0 --expected-binding=64xf32=0\n"
      "iree-run-loom kernel.loom --backend=amdgpu --function=q8_kernel \\\n"
      "  --workgroup-count=64,8,1 --constant=512 --binding=4096xf32=0\n"
      "iree-run-loom kernel.loom --backend=amdgpu --emit-only \\\n"
      "  --emit-target-artifact=kernel.hsaco "
      "--emit-hal-executable=kernel.vmfb\n"
      "iree-run-loom --backend=amdgpu --probe-hal\n"
      "```\n"
      "\n"
      "`--binding` and `--expected-binding` use the same shape/type/value "
      "syntax\n"
      "as `iree-benchmark-executable`. `--workgroup-count` overrides a static\n"
      "`kernel.launch.config` dispatch count when the test needs a different\n"
      "grid. `--emit-only` is HAL-only and stops after producing artifacts.\n"
      "\n"
      "### Debugging\n"
      "\n"
      "```shell\n"
      "iree-run-loom module.loom --compile-report=summary\n"
      "iree-run-loom module.loom --pipeline=none\n"
      "iree-run-loom module.loom --pipeline=@my_pipeline\n"
      "```\n"
      "\n"
      "`--compile-report=summary|details` prints the same structured compile\n"
      "report family as `loom-compile`. Use `iree-test-loom` once the "
      "scenario\n"
      "belongs in checked `check.case` coverage, and `iree-benchmark-loom` "
      "once\n"
      "the same case should produce benchmark evidence.\n");
}

int iree_run_loom_main(int argc, char** argv,
                       const iree_run_loom_configuration_t* configuration) {
  iree_flags_set_usage(
      configuration->tool_name,
      "Compiles a Loom module to a runtime artifact and executes the selected "
      "export.\n"
      "\n"
      "Usage:\n"
      "  iree-run-loom [file.loom] --function=name --input=... "
      "--expected-output=...\n"
      "  cat module.loom | iree-run-loom - --function=name --input=...\n"
      "  iree-run-loom --agents_md\n"
      "\n"
      "The 'vm' backend compiles VM-targeted functions into a real IREE VM "
      "bytecode archive and runs them with IREE function I/O syntax for "
      "--input, --output, and --expected-output. Native execution backends "
      "compile target-low kernels into runtime artifacts and dispatch them "
      "through their production runtime path.\n");
  for (int i = 1; i < argc; ++i) {
    if (loom_tooling_cli_is_agents_markdown_arg(argv[i])) {
      iree_run_loom_print_agents_markdown(stdout);
      return 0;
    }
  }
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);

  loom_tooling_cli_set_default_help_filter();
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_allocator_t allocator = iree_allocator_system();
  const loom_run_execution_backend_registry_t* backend_registry =
      &configuration->execution_backend_registry;
  const iree_string_view_t backend_name = iree_make_cstring_view(FLAG_backend);
  const loom_run_execution_backend_t* backend =
      loom_run_execution_backend_registry_lookup(backend_registry,
                                                 backend_name);
  if (FLAG_probe_hal) {
    loom_run_one_shot_result_t probe_result = {0};
    loom_run_one_shot_result_initialize(allocator, &probe_result);
    iree_status_t probe_status = iree_ok_status();
    if (backend == NULL) {
      probe_status = iree_run_loom_make_unknown_backend_status(
          backend_name, backend_registry, allocator);
    } else if (backend->probe == NULL) {
      probe_status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "--probe-hal requires --backend to name a probeable backend");
    } else {
      const loom_run_one_shot_probe_request_t probe_request = {
          .host_allocator = allocator,
          .result = &probe_result,
      };
      probe_status = backend->probe(backend, &probe_request);
    }
    if (iree_status_is_ok(probe_status)) {
      probe_status = loom_tooling_write_stdout(
          iree_string_builder_view(&probe_result.output));
    }
    int probe_exit_code = 0;
    if (!iree_status_is_ok(probe_status)) {
      iree_status_fprint(stderr, probe_status);
      iree_status_free(probe_status);
      probe_exit_code = 1;
    }
    loom_run_one_shot_result_deinitialize(&probe_result);
    IREE_TRACE_ZONE_END(z0);
    IREE_TRACE_APP_EXIT(probe_exit_code);
    return probe_exit_code;
  }

  iree_io_file_contents_t* contents = NULL;
  loom_run_session_t session = {0};
  loom_run_module_t run_module = {0};
  loom_run_compile_report_capture_t compile_report_capture = {0};
  loom_run_one_shot_result_t run_result = {0};
  loom_run_one_shot_result_initialize(allocator, &run_result);
  int exit_code = 0;

  iree_status_t status = iree_ok_status();
  if (argc > 2) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "iree-run-loom accepts at most one input file or '-' for stdin; got %d "
        "inputs",
        argc - 1);
  }
  if (iree_status_is_ok(status) && backend == NULL) {
    status = iree_run_loom_make_unknown_backend_status(
        backend_name, backend_registry, allocator);
  }

  if (iree_status_is_ok(status)) {
    loom_run_session_options_t session_options = {0};
    loom_run_session_options_initialize(&session_options);
    session_options.host_allocator = allocator;
    session_options.register_context = (loom_run_register_context_callback_t){
        .fn = iree_run_loom_register_context,
        .user_data = (void*)configuration,
    };
    session_options.initialize_low_descriptor_registry =
        configuration->initialize_low_descriptor_registry;
    status = loom_run_session_initialize(&session_options, &session);
  }

  const iree_string_view_t input_path =
      argc < 2 ? iree_string_view_empty() : iree_make_cstring_view(argv[1]);
  const iree_string_view_t filename =
      (argc < 2 || iree_string_view_equal(input_path, IREE_SV("-")))
          ? IREE_SV("<stdin>")
          : input_path;
  iree_string_view_t source = iree_string_view_empty();
  if (iree_status_is_ok(status)) {
    status = loom_tooling_read_input_file(input_path, allocator, &contents);
    if (iree_status_is_ok(status)) {
      source = loom_tooling_file_contents_string_view(contents);
    }
  }
  loom_run_candidate_compile_options_t compile_options = {0};
  loom_run_candidate_compile_options_initialize(&compile_options);
  compile_options.module_name = iree_make_cstring_view(FLAG_module_name);
  loom_run_compile_report_capture_options_t compile_report_options = {0};
  if (iree_status_is_ok(status)) {
    status = iree_run_loom_compile_report_options_initialize(
        &compile_report_options);
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_compile_report_capture_initialize(
        &compile_report_options, allocator, &compile_report_capture);
  }
  if (iree_status_is_ok(status)) {
    loom_run_compile_report_capture_configure_compile_options(
        &compile_report_capture, &compile_options);
  }
  loom_run_one_shot_options_t one_shot_options = {0};
  if (iree_status_is_ok(status)) {
    status =
        iree_run_loom_one_shot_options_initialize(backend, &one_shot_options);
  }
  if (iree_status_is_ok(status)) {
    loom_run_module_parse_options_t parse_options = {0};
    loom_run_module_parse_options_initialize(&parse_options);
    parse_options.filename = filename;
    parse_options.source = source;
    status = loom_run_module_parse(&session, &parse_options, &run_module);
  }
  if (iree_status_is_ok(status) &&
      iree_any_bit_set(backend->flags,
                       LOOM_RUN_EXECUTION_BACKEND_FLAG_HAL_OPTIONS) &&
      iree_string_view_is_empty(iree_make_cstring_view(FLAG_workgroup_count))) {
    loom_run_one_shot_options_apply_static_hal_workgroup_count(
        run_module.module, one_shot_options.hal_function_name,
        &one_shot_options);
  }
  if (iree_status_is_ok(status)) {
    compile_options.source_resolver =
        loom_run_module_source_resolver(&run_module);
  }
  if (iree_status_is_ok(status)) {
    loom_pass_run_result_t pass_run_result = {0};
    status =
        iree_run_loom_run_pass_pipeline(configuration, &session, &run_module,
                                        &compile_options, &pass_run_result);
    if (iree_status_is_ok(status) && pass_run_result.error_count != 0) {
      exit_code = 1;
    }
  }
  if (iree_status_is_ok(status) && exit_code == 0) {
    const loom_run_one_shot_request_t run_request = {
        .run_module = &run_module,
        .compile_options = &compile_options,
        .options = &one_shot_options,
        .compile_report_capture = &compile_report_capture,
        .host_allocator = allocator,
        .result = &run_result,
    };
    status = backend->run_one_shot(backend, &run_request);
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_tooling_write_stdout(iree_string_builder_view(&run_result.output));
  }
  if (iree_status_is_ok(status) && exit_code == 0) {
    exit_code = run_result.exit_code;
  }

  const bool had_error = !iree_status_is_ok(status);
  if (had_error) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = 1;
  }

  loom_run_compile_report_capture_deinitialize(&compile_report_capture);
  loom_run_one_shot_result_deinitialize(&run_result);
  loom_run_module_deinitialize(&run_module);
  iree_io_file_contents_free(contents);
  loom_run_session_deinitialize(&session);

  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
