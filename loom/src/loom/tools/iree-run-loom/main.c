// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// iree-run-loom: compiles one Loom kernel and executes it on one HAL device.

#include "loom/tools/iree-run-loom/main.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "iree/base/tooling/flags.h"
#include "iree/tooling/device_util.h"
#include "iree/tooling/value_io.h"
#include "loom/config/text_binding.h"
#include "loom/sanitizer/options.h"
#include "loom/tooling/cli/help.h"
#include "loom/tooling/cli/loomc_options.h"
#include "loom/tooling/cli/loomc_result.h"
#include "loom/tooling/config/config.h"
#include "loom/tooling/execution/hal/invocation.h"
#include "loom/tooling/execution/hal/runtime.h"
#include "loom/tooling/input/flags.h"
#include "loom/tooling/input/loomc.h"
#include "loom/tooling/io/file.h"
#include "loomc/launch_config.h"

IREE_FLAG(string, pipeline, "default",
          "Pass pipeline to run before execution. Use 'default' or empty for "
          "the selected target's default pipeline. 'none' disables all "
          "compiler transformations. Use '@symbol' to run a module-local "
          "pass.pipeline or a comma-separated pass list such as "
          "'canonicalize,cse'.");
IREE_FLAG(string, target, "",
          "Optional compiler target as `family:selector`. The selected HAL "
          "device must be able to load the target. Empty derives the best "
          "compatible target from the live device.");
IREE_FLAG(string, sanitizer, "none",
          "Sanitizer checks to insert in the default target pipeline: none, "
          "all, or a '|'-separated set of access, value, operation, and race.");
IREE_FLAG_NAMED(string, sanitizer_reporting, "sanitizer-reporting", "default",
                "Sanitizer assertion failure reporting mode in the default "
                "target pipeline: default, trap, or report-only.");
IREE_FLAG(string, root, "",
          "Kernel root symbol to compile and invoke, with an optional leading "
          "'@'. Empty requires the module to contain exactly one kernel.");
IREE_FLAG_LIST(
    string, config,
    "Compile-time config binding. Repeat as --config=key=value. Bindings not "
    "referenced by the loaded module are ignored.");
IREE_FLAG_LIST_NAMED(
    string, config_file, "config-file",
    "JSON/JSONC config object file. Repeat for multiple files. Nested object "
    "keys are flattened with '.' separators.");
IREE_FLAG_NAMED(int32_t, output_max_element_count, "output-max-element-count",
                1024, "Maximum number of HAL output elements to format.");
IREE_FLAG_NAMED(
    string, workgroup_count, "workgroup-count", "",
    "Optional HAL dispatch workgroup count as `x,y,z`. When omitted, the "
    "compiled kernel launch configuration is evaluated. Dynamic workload "
    "kernels currently require this override.");
IREE_FLAG_NAMED(
    string, compile_report, "compile-report", "",
    "Optional compile report output. Use 'summary'/'details' for structured "
    "JSON, 'text-summary'/'text-details' for human-readable text, or "
    "empty/'none'.");
IREE_FLAG_NAMED(string, compile_report_output, "compile-report-output",
                "stderr",
                "Output path for --compile-report. Use 'stderr' or a file "
                "path; stdout is reserved for kernel output.");

typedef struct iree_run_loom_hal_flag_state_t {
  // Dispatch constants in HAL ABI order.
  uint32_t constants[LOOM_RUN_HAL_MAX_CONSTANT_BYTE_LENGTH / sizeof(uint32_t)];
  // Number of populated entries in |constants|.
  iree_host_size_t constant_count;
  // Binding specs in HAL binding ordinal order.
  iree_string_view_t binding_specs[LOOM_RUN_HAL_MAX_BINDING_COUNT];
  // Number of populated entries in |binding_specs|.
  iree_host_size_t binding_count;
  // Expected binding specs in HAL binding ordinal order.
  iree_string_view_t expected_binding_specs[LOOM_RUN_HAL_MAX_BINDING_COUNT];
  // Number of populated entries in |expected_binding_specs|.
  iree_host_size_t expected_binding_count;
} iree_run_loom_hal_flag_state_t;

static iree_run_loom_hal_flag_state_t iree_run_loom_hal_flags = {0};

static iree_status_t iree_run_loom_parse_kernel_input_value_flag(
    iree_string_view_t flag_name, void* storage, iree_string_view_t value) {
  (void)flag_name;
  (void)storage;
  iree_tooling_value_t parsed_value = {0};
  IREE_RETURN_IF_ERROR(iree_tooling_value_spec_parse(value, &parsed_value));
  iree_host_size_t word_count = 0;
  const iree_host_size_t capacity =
      IREE_ARRAYSIZE(iree_run_loom_hal_flags.constants);
  IREE_RETURN_IF_ERROR(iree_tooling_value_write_abi_words(
      &parsed_value, capacity - iree_run_loom_hal_flags.constant_count,
      &iree_run_loom_hal_flags
           .constants[iree_run_loom_hal_flags.constant_count],
      &word_count));
  iree_run_loom_hal_flags.constant_count += word_count;
  return iree_ok_status();
}

static void iree_run_loom_print_kernel_input_value_flag(
    iree_string_view_t flag_name, void* storage, FILE* file) {
  (void)storage;
  if (iree_run_loom_hal_flags.constant_count == 0) {
    fprintf(file, "# --%.*s=i32=0\n", (int)flag_name.size, flag_name.data);
    return;
  }
  for (iree_host_size_t i = 0; i < iree_run_loom_hal_flags.constant_count;
       ++i) {
    fprintf(file, "--%.*s=0x%08X\n", (int)flag_name.size, flag_name.data,
            iree_run_loom_hal_flags.constants[i]);
  }
}
IREE_FLAG_CALLBACK_NAMED(
    iree_run_loom_parse_kernel_input_value_flag,
    iree_run_loom_print_kernel_input_value_flag, NULL, kernel_input_value,
    "kernel-input-value",
    "Appends a scalar HAL kernel input in ABI order. Supported forms include "
    "i32=..., u32=..., i64=..., u64=..., f32=..., f64=..., and bare 0x... "
    "raw uint32 ABI words.");

static iree_status_t iree_run_loom_parse_kernel_input_buffer_flag(
    iree_string_view_t flag_name, void* storage, iree_string_view_t value) {
  (void)flag_name;
  (void)storage;
  if (iree_run_loom_hal_flags.binding_count >= LOOM_RUN_HAL_MAX_BINDING_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "too many HAL bindings; maximum is %d",
                            LOOM_RUN_HAL_MAX_BINDING_COUNT);
  }
  iree_run_loom_hal_flags
      .binding_specs[iree_run_loom_hal_flags.binding_count++] = value;
  return iree_ok_status();
}

static void iree_run_loom_print_kernel_input_buffer_flag(
    iree_string_view_t flag_name, void* storage, FILE* file) {
  (void)storage;
  if (iree_run_loom_hal_flags.binding_count == 0) {
    fprintf(file, "# --%.*s=\"shapextype[=values]\"\n", (int)flag_name.size,
            flag_name.data);
    return;
  }
  for (iree_host_size_t i = 0; i < iree_run_loom_hal_flags.binding_count; ++i) {
    const iree_string_view_t binding_spec =
        iree_run_loom_hal_flags.binding_specs[i];
    fprintf(file, "--%.*s=\"%.*s\"\n", (int)flag_name.size, flag_name.data,
            (int)binding_spec.size, binding_spec.data);
  }
}
IREE_FLAG_CALLBACK_NAMED(
    iree_run_loom_parse_kernel_input_buffer_flag,
    iree_run_loom_print_kernel_input_buffer_flag, NULL, kernel_input_buffer,
    "kernel-input-buffer",
    "Appends a HAL kernel buffer binding. Bindings use the HAL tooling "
    "shape/type/data syntax and may use '&' for in-place storage buffers.");

static iree_status_t iree_run_loom_parse_expected_kernel_buffer_flag(
    iree_string_view_t flag_name, void* storage, iree_string_view_t value) {
  (void)flag_name;
  (void)storage;
  if (iree_run_loom_hal_flags.expected_binding_count >=
      LOOM_RUN_HAL_MAX_BINDING_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "too many expected HAL bindings; maximum is %d",
                            LOOM_RUN_HAL_MAX_BINDING_COUNT);
  }
  iree_run_loom_hal_flags.expected_binding_specs
      [iree_run_loom_hal_flags.expected_binding_count++] = value;
  return iree_ok_status();
}

static void iree_run_loom_print_expected_kernel_buffer_flag(
    iree_string_view_t flag_name, void* storage, FILE* file) {
  (void)storage;
  if (iree_run_loom_hal_flags.expected_binding_count == 0) {
    fprintf(file, "# --%.*s=\"shapextype=values\"\n", (int)flag_name.size,
            flag_name.data);
    return;
  }
  for (iree_host_size_t i = 0;
       i < iree_run_loom_hal_flags.expected_binding_count; ++i) {
    const iree_string_view_t binding_spec =
        iree_run_loom_hal_flags.expected_binding_specs[i];
    fprintf(file, "--%.*s=\"%.*s\"\n", (int)flag_name.size, flag_name.data,
            (int)binding_spec.size, binding_spec.data);
  }
}
IREE_FLAG_CALLBACK_NAMED(
    iree_run_loom_parse_expected_kernel_buffer_flag,
    iree_run_loom_print_expected_kernel_buffer_flag, NULL,
    expected_kernel_buffer, "expected-kernel-buffer",
    "Appends an expected HAL binding after dispatch. When present, one "
    "expected binding must be provided for every input binding.");

static iree_status_t iree_run_loom_parse_workgroup_count(
    iree_string_view_t value, uint32_t* out_workgroup_count) {
  iree_string_view_t remaining = value;
  iree_string_view_t x;
  iree_string_view_split(remaining, ',', &x, &remaining);
  iree_string_view_t y;
  iree_string_view_split(remaining, ',', &y, &remaining);
  const iree_string_view_t z = remaining;
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

static iree_status_t iree_run_loom_parse_report_options(
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

static iree_status_t iree_run_loom_parse_sanitizer_options(
    loomc_sanitizer_options_t* out_options, bool* out_enabled) {
  loom_sanitizer_options_t options = {0};
  IREE_RETURN_IF_ERROR(loom_sanitizer_options_parse_checks(
      iree_make_cstring_view(FLAG_sanitizer), IREE_SV("--sanitizer"),
      &options));
  IREE_RETURN_IF_ERROR(loom_sanitizer_reporting_mode_parse(
      iree_make_cstring_view(FLAG_sanitizer_reporting),
      IREE_SV("--sanitizer-reporting"), &options.reporting_mode));
  loom_tooling_cli_make_loomc_sanitizer_options(&options, out_options);
  *out_enabled =
      options.checks != 0 ||
      options.reporting_mode != LOOM_SANITIZER_REPORTING_MODE_DEFAULT;
  return iree_ok_status();
}

static iree_status_t iree_run_loom_append_config_options(
    loom_config_text_binding_set_t* config_set, iree_allocator_t allocator) {
  const iree_flag_string_list_t files = FLAG_config_file_list();
  for (iree_host_size_t i = 0; i < files.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_tooling_config_text_binding_set_append_json_file(
        config_set, files.values[i], allocator));
  }
  const iree_flag_string_list_t assignments = FLAG_config_list();
  for (iree_host_size_t i = 0; i < assignments.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_tooling_config_text_binding_set_append_assignment(
        config_set, assignments.values[i]));
  }
  return iree_ok_status();
}

static iree_status_t iree_run_loom_clone_string(iree_string_view_t value,
                                                iree_allocator_t allocator,
                                                iree_string_view_t* out_value,
                                                char** out_storage) {
  *out_value = iree_string_view_empty();
  *out_storage = NULL;
  iree_host_size_t allocation_size = 0;
  if (!iree_host_size_checked_add(value.size, 1, &allocation_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "kernel symbol length overflow");
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
      allocator, allocation_size, (void**)out_storage));
  if (value.size != 0) {
    memcpy(*out_storage, value.data, value.size);
  }
  (*out_storage)[value.size] = 0;
  *out_value = iree_make_string_view(*out_storage, value.size);
  return iree_ok_status();
}

static iree_status_t iree_run_loom_select_kernel(const loomc_module_t* module,
                                                 iree_allocator_t allocator,
                                                 iree_string_view_t* out_root,
                                                 char** out_root_storage,
                                                 iree_string_view_t* out_export,
                                                 char** out_export_storage) {
  *out_root = iree_string_view_empty();
  *out_root_storage = NULL;
  *out_export = iree_string_view_empty();
  *out_export_storage = NULL;

  loomc_module_function_t selected = {0};
  const iree_string_view_t requested_root =
      iree_string_view_trim(iree_make_cstring_view(FLAG_root));
  if (!iree_string_view_is_empty(requested_root)) {
    IREE_RETURN_IF_ERROR(iree_status_from_loomc(loomc_module_lookup_function(
        module, loomc_string_view_from_iree(requested_root), &selected)));
    if (!loomc_module_function_kind_is_kernel(selected.kind)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--root='%.*s' is not a kernel",
                              (int)requested_root.size, requested_root.data);
    }
  } else {
    const loomc_module_function_query_options_t query_options = {
        .type = LOOMC_STRUCTURE_TYPE_MODULE_FUNCTION_QUERY_OPTIONS,
        .structure_size = sizeof(query_options),
    };
    loomc_host_size_t function_count = 0;
    loomc_result_t* query_result = NULL;
    IREE_RETURN_IF_ERROR(iree_status_from_loomc(loomc_module_query_functions(
        module, &query_options, loomc_allocator_from_iree(allocator), 0, NULL,
        &function_count, &query_result)));
    bool query_succeeded = false;
    iree_status_t status = loom_tooling_cli_print_loomc_result(
        stderr, query_result, &query_succeeded);
    loomc_result_release(query_result);
    if (!iree_status_is_ok(status)) {
      return status;
    }
    if (!query_succeeded) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "module function query was rejected");
    }

    loomc_module_function_t* functions = NULL;
    if (function_count != 0) {
      IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
          allocator, function_count, sizeof(*functions), (void**)&functions));
    }
    query_result = NULL;
    status = iree_status_from_loomc(loomc_module_query_functions(
        module, &query_options, loomc_allocator_from_iree(allocator),
        function_count, functions, &function_count, &query_result));
    query_succeeded = false;
    if (iree_status_is_ok(status)) {
      status = loom_tooling_cli_print_loomc_result(stderr, query_result,
                                                   &query_succeeded);
    }
    loomc_result_release(query_result);
    iree_host_size_t kernel_count = 0;
    for (iree_host_size_t i = 0;
         iree_status_is_ok(status) && query_succeeded && i < function_count;
         ++i) {
      if (loomc_module_function_kind_is_kernel(functions[i].kind)) {
        selected = functions[i];
        ++kernel_count;
      }
    }
    iree_allocator_free(allocator, functions);
    if (!iree_status_is_ok(status)) {
      return status;
    }
    if (!query_succeeded) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "module function query was rejected");
    }
    if (kernel_count != 1) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "iree-run-loom requires exactly one kernel when --root is omitted; "
          "found %" PRIhsz,
          kernel_count);
    }
  }

  const iree_string_view_t root =
      iree_string_view_from_loomc(selected.symbol_name);
  iree_string_view_t export_name = root;
  loomc_module_function_export_info_t export_info = {0};
  if (loomc_module_function_try_get_export_info(module, &selected,
                                                &export_info) &&
      (export_info.flags & LOOMC_MODULE_FUNCTION_EXPORT_FLAG_HAS_SYMBOL) != 0 &&
      !loomc_string_view_is_empty(export_info.export_symbol)) {
    export_name = iree_string_view_from_loomc(export_info.export_symbol);
  }
  IREE_RETURN_IF_ERROR(
      iree_run_loom_clone_string(root, allocator, out_root, out_root_storage));
  iree_status_t status = iree_run_loom_clone_string(
      export_name, allocator, out_export, out_export_storage);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(allocator, *out_root_storage);
    *out_root = iree_string_view_empty();
    *out_root_storage = NULL;
  }
  return status;
}

typedef struct iree_run_loom_artifacts_t {
  // Loadable executable produced for the selected HAL target.
  const loomc_artifact_t* executable;
  // Optional compiled launch-configuration sidecar.
  const loomc_artifact_t* launch_config;
  // Optional requested compile report.
  const loomc_artifact_t* report;
} iree_run_loom_artifacts_t;

static iree_status_t iree_run_loom_select_artifacts(
    const loomc_result_t* result, iree_run_loom_artifacts_t* out_artifacts) {
  *out_artifacts = (iree_run_loom_artifacts_t){0};
  for (loomc_host_size_t i = 0; i < loomc_result_artifact_count(result); ++i) {
    const loomc_artifact_t* artifact = loomc_result_artifact_at(result, i);
    const bool is_report =
        loomc_string_view_equal(
            artifact->format, loomc_make_cstring_view(
                                  LOOMC_ARTIFACT_FORMAT_COMPILE_REPORT_JSON)) ||
        loomc_string_view_equal(
            artifact->format,
            loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_COMPILE_REPORT_TEXT));
    const loomc_artifact_t** slot = NULL;
    if (artifact->kind == LOOMC_ARTIFACT_KIND_EXECUTABLE) {
      slot = &out_artifacts->executable;
    } else if (artifact->kind == LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG) {
      slot = &out_artifacts->launch_config;
    } else if (is_report) {
      slot = &out_artifacts->report;
    }
    if (slot != NULL && *slot != NULL) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "compiler returned duplicate run artifacts");
    }
    if (slot != NULL) {
      *slot = artifact;
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_run_loom_write_artifact(
    const loomc_artifact_t* artifact, iree_string_view_t output_path,
    iree_allocator_t allocator) {
  if (artifact == NULL) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "compiler did not return a requested artifact");
  }
  FILE* file = NULL;
  if (iree_string_view_equal(output_path, IREE_SV("stderr"))) {
    file = stderr;
  } else if (loom_tooling_output_path_is_stdout(output_path)) {
    file = stdout;
  }
  if (file == NULL) {
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

static iree_status_t iree_run_loom_evaluate_launch_config(
    const loomc_artifact_t* artifact, iree_string_view_t export_name,
    iree_allocator_t allocator, uint32_t* out_workgroup_count) {
  if (artifact == NULL) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "compiler did not return the requested launch configuration");
  }
  loomc_launch_config_program_t* program = NULL;
  IREE_RETURN_IF_ERROR(iree_status_from_loomc(loomc_launch_config_program_load(
      artifact, loomc_allocator_from_iree(allocator), &program)));
  loomc_launch_config_function_t function =
      loomc_launch_config_function_invalid();
  iree_status_t status =
      iree_status_from_loomc(loomc_launch_config_program_lookup_function(
          program, loomc_string_view_from_iree(export_name), &function));
  loomc_launch_config_t launch_config = {
      .type = LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG,
      .structure_size = sizeof(launch_config),
  };
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_launch_config_program_invoke(
        program, function, /*workload_argument_bits=*/NULL,
        /*workload_argument_count=*/0, &launch_config));
  }
  if (!iree_status_is_ok(status)) {
    status = iree_status_annotate(
        status,
        IREE_SV("while evaluating the kernel launch configuration; kernels "
                "with dynamic workload arguments currently require "
                "--workgroup-count=x,y,z"));
  } else {
    out_workgroup_count[0] = launch_config.workgroup_count.x;
    out_workgroup_count[1] = launch_config.workgroup_count.y;
    out_workgroup_count[2] = launch_config.workgroup_count.z;
  }
  loomc_launch_config_program_release(program);
  return status;
}

static void iree_run_loom_print_agents_markdown(FILE* stream) {
  static const char* const kLines[] = {
      "## iree-run-loom",
      "",
      "`iree-run-loom` compiles and invokes exactly one Loom kernel on one",
      "HAL device. It is a focused kernel runner, not a command-program or",
      "heterogeneous-device runner.",
      "",
      "```shell",
      "iree-run-loom kernel.loom --device=amdgpu --root=q8_kernel \\",
      "  --kernel-input-buffer=64xf32=0 \\",
      "  --expected-kernel-buffer=64xf32=0",
      "iree-run-loom kernel.loom --device=vulkan --root=q8_kernel \\",
      "  --workgroup-count=64,1,1 --kernel-input-value=i32=512",
      "```",
      "",
      "`--kernel-input-buffer` and `--expected-kernel-buffer` use the HAL",
      "tooling shape/type/value syntax. `--workgroup-count` overrides the",
      "compiled launch configuration.",
      "",
      "```shell",
      "iree-run-loom module.loom --device=amdgpu --compile-report=summary \\",
      "  --compile-report-output=report.json",
      "iree-run-loom prepared-low.loom --device=amdgpu --root=kernel \\",
      "  --pipeline=none",
      "```",
      "",
      "`--pipeline=none` requires input that already satisfies the selected",
      "target emitter. Repeated `--device` flags are rejected because kernels",
      "and command programs need one compilation transaction before they can",
      "be placed across heterogeneous devices.",
  };
  _Static_assert(IREE_ARRAYSIZE(kLines) <= 100,
                 "--agents_md must remain at most 100 lines");
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kLines); ++i) {
    fprintf(stream, "%s\n", kLines[i]);
  }
}

int iree_run_loom_main(int argc, char** argv,
                       const iree_run_loom_configuration_t* configuration) {
  iree_flags_set_usage(
      configuration->tool_name,
      "Compiles and executes exactly one Loom kernel on one HAL device.\n"
      "\n"
      "Usage:\n"
      "  iree-run-loom [file] --device=URI [--root=name] "
      "--kernel-input-buffer=...\n"
      "  cat module.loom | iree-run-loom - --device=URI [--root=name]\n"
      "  iree-run-loom --agents_md\n"
      "\n"
      "The live device selects a compatible public compiler profile and exact "
      "HAL loader target. Multiple --device flags are not accepted.\n");
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

  const iree_allocator_t allocator = iree_allocator_system();
  const loomc_allocator_t loom_allocator = loomc_allocator_from_iree(allocator);
  const loom_input_options_t input_options = loom_input_options_from_flags();
  iree_status_t status = iree_ok_status();
  int exit_code = 0;

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(32 * 1024, allocator, &block_pool);
  loom_config_text_binding_set_t config_set;
  loom_config_text_binding_set_initialize(allocator, &config_set);
  loomc_config_binding_t* config_bindings = NULL;
  loomc_config_options_t config_options = {0};
  char* root_storage = NULL;
  char* export_storage = NULL;
  iree_string_view_t root = iree_string_view_empty();
  iree_string_view_t export_name = iree_string_view_empty();

  iree_io_file_contents_t* input_contents = NULL;
  loomc_target_environment_t* target_environment = NULL;
  loomc_context_t* context = NULL;
  loomc_workspace_t* workspace = NULL;
  loomc_compiler_t* compiler = NULL;
  loomc_module_t* module = NULL;
  loomc_pass_program_t* pass_program = NULL;
  loomc_target_profile_t* requested_target_profile = NULL;
  loomc_target_profile_t* selected_target_profile = NULL;
  loomc_result_t* result = NULL;

  loom_run_hal_runtime_t runtime = {0};
  loom_run_hal_invocation_plan_t invocation_plan = {0};
  loom_run_hal_invocation_plan_initialize(&invocation_plan);
  loom_run_hal_prepared_candidate_t candidate = {0};
  loom_run_hal_prepared_candidate_initialize(&candidate);
  loom_run_hal_dispatch_batch_t dispatch_batch = {0};
  loom_run_hal_dispatch_batch_initialize(&dispatch_batch);
  loom_run_hal_invocation_result_t invocation_result = {0};
  loom_run_hal_invocation_result_initialize(allocator, &invocation_result);
  iree_hal_profiling_from_flags_t* profiling = NULL;

  loomc_compile_report_options_t report_options = {0};
  loomc_sanitizer_options_t sanitizer_options = {0};
  bool report_enabled = false;
  bool sanitizer_enabled = false;
  bool workgroup_override = false;
  uint32_t workgroup_count[3] = {0};

  if (configuration == NULL || configuration->tool_name == NULL ||
      configuration->create_target_environment == NULL) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "iree-run-loom is missing compiler composition");
  } else if (argc > 2) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "iree-run-loom accepts at most one input file or '-' for stdin; got "
        "%d inputs",
        argc - 1);
  }

  const iree_string_view_list_t device_flags = iree_hal_device_flag_list();
  iree_string_view_t device_driver_name = iree_string_view_empty();
  if (iree_status_is_ok(status) && device_flags.count != 1) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "iree-run-loom requires exactly one --device= flag; got %" PRIhsz,
        device_flags.count);
  }
  if (iree_status_is_ok(status)) {
    iree_string_view_t device_path = iree_string_view_empty();
    iree_string_view_split(device_flags.values[0], ':', &device_driver_name,
                           &device_path);
    if (iree_string_view_is_empty(device_driver_name)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "--device must name a HAL driver");
    }
  }
  if (iree_status_is_ok(status) && FLAG_output_max_element_count < 0) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--output-max-element-count must be non-negative; got %d",
        (int)FLAG_output_max_element_count);
  }
  const iree_string_view_t workgroup_count_value =
      iree_string_view_trim(iree_make_cstring_view(FLAG_workgroup_count));
  workgroup_override = !iree_string_view_is_empty(workgroup_count_value);
  if (iree_status_is_ok(status) && workgroup_override) {
    status = iree_run_loom_parse_workgroup_count(workgroup_count_value,
                                                 workgroup_count);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_run_loom_parse_report_options(&report_options, &report_enabled);
  }
  const iree_string_view_t report_output_path =
      iree_make_cstring_view(FLAG_compile_report_output);
  if (iree_status_is_ok(status) && report_enabled &&
      loom_tooling_output_path_is_stdout(report_output_path)) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--compile-report-output must name stderr or a file; stdout is "
        "reserved for kernel output");
  }
  if (iree_status_is_ok(status)) {
    status = iree_run_loom_parse_sanitizer_options(&sanitizer_options,
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
    status = iree_run_loom_append_config_options(&config_set, allocator);
  }
  if (iree_status_is_ok(status)) {
    status = loom_tooling_cli_make_loomc_config_options(
        &config_set, allocator, &config_bindings, &config_options);
  }

  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(configuration->create_target_environment(
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

  const iree_string_view_t input_path =
      argc < 2 ? iree_string_view_empty() : iree_make_cstring_view(argv[1]);
  if (iree_status_is_ok(status)) {
    status =
        loom_tooling_read_input_file(input_path, allocator, &input_contents);
  }
  if (iree_status_is_ok(status)) {
    const loom_tooling_loomc_input_options_t admission_options = {
        .providers = configuration->input_providers,
        .input = input_options,
        .path = input_path,
        .source = loom_tooling_file_contents_string_view(input_contents),
        .import = configuration->import,
        .import_user_data = configuration->import_user_data,
    };
    status = loom_tooling_input_admit_loomc_module(&admission_options, context,
                                                   workspace, &block_pool,
                                                   &module, &result, allocator);
  }
  if (iree_status_is_ok(status)) {
    bool admitted = false;
    status = loom_tooling_cli_print_loomc_result(stderr, result, &admitted);
    if (iree_status_is_ok(status) && !admitted) {
      exit_code = 1;
    }
    loomc_result_release(result);
    result = NULL;
  }
  if (iree_status_is_ok(status) && exit_code == 0) {
    status = iree_run_loom_select_kernel(
        module, allocator, &root, &root_storage, &export_name, &export_storage);
  }
  if (iree_status_is_ok(status) && exit_code == 0) {
    status = loom_tooling_cli_prepare_loomc_pass_program(
        context, module, iree_make_cstring_view(FLAG_pipeline),
        IREE_SV("iree-run-loom pipeline"), &pass_program, &result, allocator);
  }
  if (iree_status_is_ok(status) && result != NULL) {
    bool prepared = false;
    status = loom_tooling_cli_print_loomc_result(stderr, result, &prepared);
    if (iree_status_is_ok(status) && !prepared) {
      exit_code = 1;
    }
    loomc_result_release(result);
    result = NULL;
  }

  const iree_string_view_t requested_target =
      iree_string_view_trim(iree_make_cstring_view(FLAG_target));
  if (iree_status_is_ok(status) && exit_code == 0 &&
      !iree_string_view_is_empty(requested_target)) {
    status = iree_status_from_loomc(loomc_target_profile_select(
        target_environment, loomc_string_view_from_iree(requested_target),
        loom_allocator, &requested_target_profile));
  }
  iree_hal_device_runtime_feature_flags_t runtime_features =
      IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_NONE;
  if (iree_status_is_ok(status) && exit_code == 0) {
    status =
        iree_status_from_loomc(loomc_iree_hal_module_query_runtime_features(
            module, sanitizer_enabled ? &sanitizer_options : NULL,
            &runtime_features));
  }
  if (iree_status_is_ok(status) && exit_code == 0) {
    loom_run_hal_runtime_options_t runtime_options;
    loom_run_hal_runtime_options_initialize(device_driver_name,
                                            &runtime_options);
    runtime_options.runtime_features = runtime_features;
    status =
        loom_run_hal_runtime_initialize(&runtime_options, allocator, &runtime);
  }

  const iree_hal_executable_target_t* executable_target = NULL;
  if (iree_status_is_ok(status) && exit_code == 0) {
    const loomc_iree_hal_target_options_t target_options = {
        .type = LOOMC_STRUCTURE_TYPE_IREE_HAL_TARGET_OPTIONS,
        .structure_size = sizeof(target_options),
        .identifier = loomc_make_cstring_view("iree-run-loom live device"),
        .device = runtime.device,
        .physical_device_affinity =
            loom_run_hal_runtime_dispatch_physical_device_affinity(&runtime),
        .target_profile = requested_target_profile,
        .providers = configuration->hal_target_providers,
        .provider_count = configuration->hal_target_provider_count,
    };
    loomc_iree_hal_target_selection_t selection = {0};
    status = iree_status_from_loomc(
        loomc_target_select_iree_hal(target_environment, &target_options,
                                     loom_allocator, &selection, &result));
    selected_target_profile = selection.target_profile;
    executable_target = selection.executable_target;
  }
  if (iree_status_is_ok(status) && result != NULL) {
    bool selected = false;
    status = loom_tooling_cli_print_loomc_result(stderr, result, &selected);
    if (iree_status_is_ok(status) && !selected) {
      exit_code = 1;
    }
    loomc_result_release(result);
    result = NULL;
  }

  if (report_enabled &&
      !loom_tooling_output_path_is_stdout(report_output_path) &&
      !iree_string_view_equal(report_output_path, IREE_SV("stderr"))) {
    report_options.identifier = loomc_string_view_from_iree(report_output_path);
  }
  const loomc_emit_options_t emit_options = {
      .type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
      .structure_size = sizeof(emit_options),
      .next = report_enabled ? (const void*)&report_options : NULL,
      .artifact_flags = LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY,
  };
  const loomc_string_view_t compile_root = loomc_string_view_from_iree(root);
  const loomc_compile_artifact_options_t compile_options = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_ARTIFACT_OPTIONS,
      .structure_size = sizeof(compile_options),
      .next = sanitizer_enabled ? (const void*)&sanitizer_options : NULL,
      .roots = &compile_root,
      .root_count = 1,
      .target_profile = selected_target_profile,
      .config = &config_options,
      .emit_options = &emit_options,
      .artifact_flags =
          workgroup_override ? 0 : LOOMC_COMPILE_ARTIFACT_FLAG_LAUNCH_CONFIG,
  };
  if (iree_status_is_ok(status) && exit_code == 0) {
    status = iree_status_from_loomc(
        loomc_compile_artifact(compiler, workspace, pass_program, module,
                               &compile_options, loom_allocator, &result));
  }

  iree_run_loom_artifacts_t artifacts = {0};
  if (iree_status_is_ok(status) && exit_code == 0) {
    bool compiled = false;
    status = loom_tooling_cli_print_loomc_result(stderr, result, &compiled);
    if (iree_status_is_ok(status)) {
      status = iree_run_loom_select_artifacts(result, &artifacts);
    }
    if (iree_status_is_ok(status) && report_enabled) {
      status = iree_run_loom_write_artifact(artifacts.report,
                                            report_output_path, allocator);
    }
    if (iree_status_is_ok(status) && !compiled) {
      exit_code = 1;
    }
    if (iree_status_is_ok(status) && compiled && artifacts.executable == NULL) {
      status = iree_make_status(
          IREE_STATUS_INTERNAL,
          "compiler did not return a loadable executable artifact");
    }
  }
  if (iree_status_is_ok(status) && exit_code == 0 && !workgroup_override) {
    status = iree_run_loom_evaluate_launch_config(
        artifacts.launch_config, export_name, allocator, workgroup_count);
  }

  loom_run_hal_invocation_options_t invocation_options;
  loom_run_hal_invocation_options_initialize(&invocation_options);
  invocation_options.function_name = export_name;
  memcpy(invocation_options.workgroup_count, workgroup_count,
         sizeof(workgroup_count));
  invocation_options.constant_byte_length =
      iree_run_loom_hal_flags.constant_count * sizeof(uint32_t);
  memcpy(invocation_options.constants, iree_run_loom_hal_flags.constants,
         invocation_options.constant_byte_length);
  const loom_run_hal_binding_specs_t bindings = {
      .values = iree_run_loom_hal_flags.binding_specs,
      .count = iree_run_loom_hal_flags.binding_count,
  };
  const loom_run_hal_binding_specs_t expected_bindings = {
      .values = iree_run_loom_hal_flags.expected_binding_specs,
      .count = iree_run_loom_hal_flags.expected_binding_count,
  };
  bool retain_command_buffer_metadata = false;
  if (iree_status_is_ok(status) && exit_code == 0) {
    status =
        iree_hal_profiling_from_flags_requires_retained_command_buffer_metadata(
            &retain_command_buffer_metadata);
  }
  if (iree_status_is_ok(status) && exit_code == 0) {
    status = iree_hal_begin_device_group_profiling_from_flags(
        runtime.device_group, allocator, &profiling);
  }
  if (iree_status_is_ok(status) && exit_code == 0) {
    status = loom_run_hal_invocation_plan_prepare_from_specs(
        &runtime, &invocation_options, &bindings, &expected_bindings,
        (iree_host_size_t)FLAG_output_max_element_count, allocator,
        &invocation_plan);
  }
  const loom_device_artifact_t device_artifact = {
      .executable_target = executable_target,
      .contents =
          artifacts.executable != NULL
              ? iree_byte_sequence_from_loomc(artifacts.executable->contents)
              : NULL,
  };
  if (iree_status_is_ok(status) && exit_code == 0) {
    status = loom_run_hal_prepared_candidate_prepare(&runtime, &device_artifact,
                                                     allocator, &candidate);
  }
  loom_run_hal_dispatch_batch_options_t batch_options;
  loom_run_hal_dispatch_batch_options_initialize(&batch_options);
  if (retain_command_buffer_metadata) {
    batch_options.command_buffer_mode &=
        ~IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED;
  }
  if (iree_status_is_ok(status) && exit_code == 0) {
    status = loom_run_hal_dispatch_batch_prepare(
        &runtime, &candidate, &invocation_plan, &batch_options, allocator,
        &dispatch_batch);
  }
  if (iree_status_is_ok(status) && exit_code == 0) {
    status = loom_run_hal_dispatch_batch_execute(&runtime, &dispatch_batch);
  }
  if (iree_status_is_ok(status) && exit_code == 0) {
    status = loom_run_hal_dispatch_batch_collect_results(
        &runtime, &invocation_plan, &dispatch_batch, allocator,
        &invocation_result);
  }
  if (profiling != NULL) {
    status =
        iree_status_join(status, iree_hal_end_profiling_from_flags(profiling));
    profiling = NULL;
  }
  if (iree_status_is_ok(status) && exit_code == 0) {
    status = loom_tooling_write_stdout(
        iree_string_builder_view(&invocation_result.output));
  }
  if (iree_status_is_ok(status) && exit_code == 0) {
    exit_code = invocation_result.exit_code;
  }

  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = 1;
  }

  loom_run_hal_invocation_result_deinitialize(&invocation_result);
  loom_run_hal_dispatch_batch_deinitialize(&dispatch_batch);
  loom_run_hal_prepared_candidate_deinitialize(&candidate);
  loom_run_hal_invocation_plan_deinitialize(&invocation_plan);
  loom_run_hal_runtime_deinitialize(&runtime);
  loomc_result_release(result);
  loomc_target_profile_release(selected_target_profile);
  loomc_target_profile_release(requested_target_profile);
  loomc_pass_program_release(pass_program);
  loomc_module_release(module);
  loomc_compiler_release(compiler);
  loomc_workspace_release(workspace);
  loomc_context_release(context);
  loomc_target_environment_release(target_environment);
  iree_io_file_contents_free(input_contents);
  iree_allocator_free(allocator, export_storage);
  iree_allocator_free(allocator, root_storage);
  iree_allocator_free(allocator, config_bindings);
  loom_config_text_binding_set_deinitialize(&config_set);
  iree_arena_block_pool_deinitialize(&block_pool);

  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
