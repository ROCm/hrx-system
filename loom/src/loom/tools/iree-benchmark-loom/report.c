// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-benchmark-loom/report.h"

#include <inttypes.h>
#include <math.h>
#include <stdio.h>

#include "iree/base/internal/path.h"
#include "iree/hal/api.h"
#include "iree/tooling/device_util.h"
#include "loom/sanitizer/options.h"
#include "loom/tooling/execution/benchmark.h"
#include "loom/tooling/io/file.h"
#include "loom/tooling/testbench/issue_report.h"
#include "loom/tools/iree-benchmark-loom/device_spec_report.h"
#include "loom/tools/iree-benchmark-loom/diagnostics.h"
#include "loom/tools/iree-benchmark-loom/module_query.h"
#include "loom/tools/iree-benchmark-loom/options.h"
#include "loom/tools/iree-benchmark-loom/profile_report.h"
#include "loom/util/json.h"

static iree_string_view_t iree_benchmark_loom_selected_device_uri(
    const iree_benchmark_loom_hal_context_t* context) {
  const iree_string_view_list_t device_uris = iree_hal_device_flag_list();
  if (device_uris.count == 1) {
    return device_uris.values[0];
  }
  if (context->execution.artifact_provider != NULL) {
    return context->execution.artifact_provider->hal_driver_name;
  }
  return iree_string_view_empty();
}

iree_status_t iree_benchmark_loom_write_status_code_json(
    iree_status_code_t code, iree_string_view_t message,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  bool first_field = true;
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u32_field(
      stream, &first_field, "code", (uint32_t)code));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      stream, &first_field, "name",
      iree_make_cstring_view(iree_status_code_string(code))));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_optional_string_field(
      stream, &first_field, "message", message));
  return loom_output_stream_write_cstring(stream, "}");
}

iree_status_t iree_benchmark_loom_write_status_field_json(
    iree_status_code_t code, iree_string_view_t message,
    loom_output_stream_t* stream, bool* first_field) {
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
      stream, first_field, "status"));
  return iree_benchmark_loom_write_status_code_json(code, message, stream);
}

iree_status_t iree_benchmark_loom_write_status_object_json(
    const iree_status_t status, loom_output_stream_t* stream) {
  const iree_status_code_t code = iree_status_code(status);
  char message[512];
  iree_host_size_t required_length = 0;
  if (!iree_status_format(status, sizeof(message), message, &required_length)) {
    const char* status_string = iree_status_code_string(code);
    iree_host_size_t index = 0;
    for (; status_string[index] != '\0' && index + 1 < sizeof(message);
         ++index) {
      message[index] = status_string[index];
    }
    message[index] = '\0';
    required_length = index;
  }
  if (required_length >= sizeof(message)) {
    required_length = sizeof(message) - 1;
  }

  return iree_benchmark_loom_write_status_code_json(
      code, iree_make_string_view(message, required_length), stream);
}

iree_status_t iree_benchmark_loom_write_run_id_field_json(
    const iree_benchmark_loom_run_identity_t* run,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"run_id\":"));
  return loom_json_write_escaped_string(stream, run->run_id);
}

iree_status_t iree_benchmark_loom_write_candidate_identity_json(
    const iree_benchmark_loom_candidate_identity_t* candidate,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"candidate_id\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(stream, candidate->candidate_id));
  return loom_output_stream_write_format(
      stream, ",\"candidate_index\":%" PRIhsz, candidate->candidate_index);
}

iree_status_t iree_benchmark_loom_write_sample_compilation_field_json(
    iree_string_view_t sample_compilation, loom_output_stream_t* stream) {
  if (iree_string_view_is_empty(sample_compilation)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"sample_compilation\":"));
  return loom_json_write_escaped_string(stream, sample_compilation);
}

iree_status_t iree_benchmark_loom_write_sanitizer_options_json(
    const loom_sanitizer_options_t* sanitizer, loom_output_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(sanitizer);
  IREE_RETURN_IF_ERROR(loom_sanitizer_options_validate(sanitizer));
  iree_string_view_t checks = iree_string_view_empty();
  iree_string_view_t reporting_mode = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      loom_sanitizer_checks_format(sanitizer->checks, &checks));
  IREE_RETURN_IF_ERROR(loom_sanitizer_reporting_mode_format(
      sanitizer->reporting_mode, &reporting_mode));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\"checks\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(stream, checks));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"reporting_mode\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(stream, reporting_mode));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t iree_benchmark_loom_write_work_item_index_field_json(
    iree_host_size_t work_item_index, loom_output_stream_t* stream) {
  if (work_item_index == IREE_BENCHMARK_LOOM_INDEX_INVALID) {
    return iree_ok_status();
  }
  return loom_output_stream_write_format(
      stream, ",\"work_item_index\":%" PRIhsz, work_item_index);
}

static const char* iree_benchmark_loom_parameter_kind_name(
    loom_testbench_parameter_kind_t kind) {
  switch (kind) {
    case LOOM_TESTBENCH_PARAMETER_RANGE:
      return "range";
    case LOOM_TESTBENCH_PARAMETER_CHOICE:
      return "choice";
    case LOOM_TESTBENCH_PARAMETER_SEED:
      return "seed";
    default:
      return "unknown";
  }
}

static iree_status_t iree_benchmark_loom_write_parameter_name_json(
    const loom_module_t* module,
    const loom_testbench_parameter_plan_t* parameter,
    iree_host_size_t parameter_index, loom_output_stream_t* stream) {
  if (!iree_string_view_is_empty(parameter->name)) {
    return loom_json_write_escaped_string(stream, parameter->name);
  }
  iree_string_view_t name =
      iree_benchmark_loom_value_name(module, parameter->value_id);
  if (!iree_string_view_is_empty(name)) {
    return loom_json_write_escaped_string(stream, name);
  }
  return loom_output_stream_write_format(stream, "\"param_%" PRIhsz "\"",
                                         parameter_index);
}

static iree_status_t iree_benchmark_loom_write_sample_attr_json(
    loom_attribute_t value, loom_output_stream_t* stream) {
  switch ((loom_attr_kind_t)value.kind) {
    case LOOM_ATTR_I64:
      return loom_output_stream_write_format(stream, "%" PRIi64,
                                             loom_attr_as_i64(value));
    case LOOM_ATTR_F64: {
      const double f64 = loom_attr_as_f64(value);
      if (!isfinite(f64)) {
        return loom_json_write_escaped_cstring(stream, "nonfinite");
      }
      char buffer[64];
      const int length = iree_snprintf(buffer, sizeof(buffer), "%.17g", f64);
      if (length <= 0 || (iree_host_size_t)length >= sizeof(buffer)) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to format f64 sample value");
      }
      return loom_output_stream_write(stream,
                                      iree_make_string_view(buffer, length));
    }
    case LOOM_ATTR_BOOL:
      return loom_output_stream_write_cstring(
          stream, loom_attr_as_bool(value) ? "true" : "false");
    case LOOM_ATTR_STRING:
      return loom_json_write_escaped_cstring(stream, "<string>");
    default:
      return loom_json_write_escaped_cstring(stream, "<unsupported>");
  }
}

static iree_status_t iree_benchmark_loom_write_case_parameter_map_json(
    const loom_module_t* module, const loom_testbench_case_plan_t* case_plan,
    iree_host_size_t sample_ordinal, bool write_ordinals,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  for (iree_host_size_t parameter_index = 0;
       parameter_index < case_plan->parameter_count; ++parameter_index) {
    const loom_testbench_parameter_plan_t* parameter =
        &case_plan->parameters[parameter_index];
    const iree_host_size_t parameter_sample_ordinal =
        loom_testbench_case_sample_parameter_ordinal(case_plan, sample_ordinal,
                                                     parameter_index);
    if (parameter_index != 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
    }
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_parameter_name_json(
        module, parameter, parameter_index, stream));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ":"));
    if (write_ordinals) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
          stream, "%" PRIhsz, parameter_sample_ordinal));
    } else {
      loom_attribute_t sample_value = loom_attr_absent();
      IREE_RETURN_IF_ERROR(loom_testbench_parameter_sample_value(
          parameter, parameter_sample_ordinal, &sample_value));
      IREE_RETURN_IF_ERROR(
          iree_benchmark_loom_write_sample_attr_json(sample_value, stream));
    }
  }
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t iree_benchmark_loom_write_sample_json(
    const loom_module_t* module, const loom_testbench_case_plan_t* case_plan,
    iree_host_size_t sample_ordinal, loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "\"sample_ordinal\":%" PRIhsz, sample_ordinal));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"parameter_count\":%" PRIhsz, case_plan->parameter_count));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"parameters\":"));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_case_parameter_map_json(
      module, case_plan, sample_ordinal, /*write_ordinals=*/false, stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"parameter_ordinals\":"));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_case_parameter_map_json(
      module, case_plan, sample_ordinal, /*write_ordinals=*/true, stream));
  return loom_output_stream_write_cstring(stream, "}");
}

iree_status_t iree_benchmark_loom_write_sample_fields_json(
    const loom_module_t* module, const loom_testbench_case_plan_t* case_plan,
    iree_host_size_t sample_ordinal, loom_output_stream_t* stream) {
  if (case_plan->parameter_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"sample_id\":\"s%" PRIhsz "\",\"sample_index\":%" PRIhsz,
      sample_ordinal, sample_ordinal));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"sample\":"));
  return iree_benchmark_loom_write_sample_json(module, case_plan,
                                               sample_ordinal, stream);
}

iree_status_t iree_benchmark_loom_write_case_sample_plan_fields_json(
    const loom_module_t* module, const loom_testbench_case_plan_t* case_plan,
    loom_output_stream_t* stream) {
  if (case_plan->parameter_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream,
      ",\"case_sample_count\":%" PRIhsz
      ",\"case_cartesian_sample_count\":%" PRIhsz
      ",\"case_sample_count_truncated\":%s",
      case_plan->sample_count, case_plan->cartesian_sample_count,
      case_plan->sample_count_truncated ? "true" : "false"));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"case_parameters\":["));
  for (iree_host_size_t parameter_index = 0;
       parameter_index < case_plan->parameter_count; ++parameter_index) {
    const loom_testbench_parameter_plan_t* parameter =
        &case_plan->parameters[parameter_index];
    if (parameter_index != 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
    }
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, "{\"name\":"));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_parameter_name_json(
        module, parameter, parameter_index, stream));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"kind\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(
        stream, iree_benchmark_loom_parameter_kind_name(parameter->kind)));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, ",\"sample_count\":%" PRIhsz, parameter->sample_count));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}"));
  }
  return loom_output_stream_write_cstring(stream, "]");
}

iree_status_t iree_benchmark_loom_append_sample_row(
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_candidate_identity_t* candidate,
    iree_host_size_t work_item_index, const loom_module_t* module,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    iree_string_view_t sample_compilation,
    iree_host_size_t benchmark_sample_ordinal,
    iree_host_size_t case_sample_ordinal,
    const loom_testbench_case_sample_result_t* sample_result,
    iree_string_builder_t* sample_output) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(sample_output, &stream);
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, "{\"row\":\"sample\""));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_run_id_field_json(run, &stream));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_candidate_identity_json(candidate, &stream));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_work_item_index_field_json(
      work_item_index, &stream));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_sample_compilation_field_json(
      sample_compilation, &stream));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_sample_fields_json(
      module, case_plan, case_sample_ordinal, &stream));
  if (benchmark_sample_ordinal != case_sample_ordinal) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        &stream, ",\"benchmark_sample_ordinal\":%" PRIhsz,
        benchmark_sample_ordinal));
  }
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"sample_result\":"));
  IREE_RETURN_IF_ERROR(
      loom_testbench_case_sample_result_write_json(sample_result, &stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}\n"));
  return iree_ok_status();
}

iree_status_t iree_benchmark_loom_append_run_row(
    const iree_benchmark_loom_run_identity_t* run, bool dry_run,
    iree_benchmark_loom_sample_compilation_mode_t sample_compilation_mode,
    const loom_sanitizer_options_t* sanitizer, iree_string_builder_t* output) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(output, &stream);
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, "{\"row\":\"run\""));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_run_id_field_json(run, &stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
      &stream, ",\"tool\":\"iree-benchmark-loom\""));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"source\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(&stream, run->source));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"results_path\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(&stream, run->results_path));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"file_output_dir\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(&stream, run->file_output_dir));
  if (!iree_string_view_is_empty(run->profile_artifacts_dir)) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
        &stream, ",\"profile_artifacts_dir\":"));
    IREE_RETURN_IF_ERROR(
        loom_json_write_escaped_string(&stream, run->profile_artifacts_dir));
  }
  if (!iree_string_view_is_empty(run->artifact_bundle_dir)) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
        &stream, ",\"artifact_bundle\":{\"dir\":"));
    IREE_RETURN_IF_ERROR(
        loom_json_write_escaped_string(&stream, run->artifact_bundle_dir));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"policy\":"));
    IREE_RETURN_IF_ERROR(
        loom_json_write_escaped_string(&stream, run->artifact_bundle_policy));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}"));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"dry_run\":%s", dry_run ? "true" : "false"));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"sample_compilation\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream, iree_benchmark_loom_sample_compilation_mode_name(
                   sample_compilation_mode)));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"sanitizer\":"));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_sanitizer_options_json(sanitizer, &stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}\n"));
  return iree_ok_status();
}

iree_status_t iree_benchmark_loom_write_hal_context_identity_fields_json(
    const iree_benchmark_loom_hal_context_t* context,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "\"device_uri\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      stream, iree_benchmark_loom_selected_device_uri(context)));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"driver\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      stream, context->execution.artifact_provider->hal_driver_name));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"provider\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      stream, context->execution.artifact_provider->name));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"target_family\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      stream, context->execution.artifact_provider->target_family_name));
  if (context->execution.runtime_initialized &&
      context->execution.runtime.device != NULL) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"state\":\"created\""));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"device_id\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        stream, iree_hal_device_id(context->execution.runtime.device)));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"device_spec\":"));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_device_spec_json(
        iree_hal_device_spec(context->execution.runtime.device), stream));
  } else {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"state\":\"planned\""));
  }
  return iree_ok_status();
}

iree_status_t iree_benchmark_loom_append_device_row(
    const iree_benchmark_loom_run_identity_t* run,
    iree_benchmark_loom_hal_context_t* context,
    iree_benchmark_loom_device_row_state_t* row_state,
    iree_string_builder_t* device_output) {
  if (row_state->appended) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_run_hal_testbench_context_ensure_runtime(&context->execution));

  loom_output_stream_t stream;
  loom_output_stream_for_builder(device_output, &stream);
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, "{\"row\":\"device\""));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_run_id_field_json(run, &stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ","));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_hal_context_identity_fields_json(context,
                                                                 &stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}\n"));
  row_state->appended = true;
  return iree_ok_status();
}

static iree_status_t iree_benchmark_loom_write_actual_invocation_plan_json(
    const loom_module_t* module, const loom_testbench_case_plan_t* case_plan,
    loom_output_stream_t* stream) {
  const loom_testbench_invocation_plan_t* actual_invocation = NULL;
  iree_host_size_t actual_invocation_count = 0;
  for (iree_host_size_t i = 0; i < case_plan->invocation_count; ++i) {
    const loom_testbench_invocation_plan_t* invocation =
        &case_plan->invocations[i];
    if (invocation->kind != LOOM_TESTBENCH_INVOCATION_ACTUAL) {
      continue;
    }
    actual_invocation = invocation;
    ++actual_invocation_count;
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"actual_invocation_count\":%" PRIhsz,
      actual_invocation_count));
  if (actual_invocation_count != 1) {
    return iree_ok_status();
  }

  iree_string_view_t actual_entry = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_module_symbol_name_from_ref(
      module, actual_invocation->callee_ref, &actual_entry));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"actual_entry\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(stream, actual_entry));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"actual_input_count\":%" PRIhsz,
      actual_invocation->input_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"actual_result_count\":%" PRIhsz,
      actual_invocation->result_count));
  return iree_ok_status();
}

iree_status_t iree_benchmark_loom_append_plan_row(
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_candidate_identity_t* candidate,
    const loom_module_t* module,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_benchmark_loom_benchmark_policy_t* policy,
    const iree_benchmark_loom_options_t* options,
    iree_benchmark_loom_sample_compilation_mode_t sample_compilation_mode,
    iree_allocator_t allocator, iree_string_builder_t* plan_output) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(plan_output, &stream);
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, "{\"row\":\"plan\""));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_run_id_field_json(run, &stream));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_candidate_identity_json(candidate, &stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"benchmark\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(&stream, benchmark_plan->name));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ",\"case\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(&stream, case_plan->name));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"measure\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(&stream, policy->measure));
  if (policy->measure_kind == IREE_BENCHMARK_LOOM_MEASURE_DISPATCH_COMPLETE) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"sample_compilation\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        &stream, iree_benchmark_loom_sample_compilation_mode_name(
                     sample_compilation_mode)));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"warmup_iterations\":%" PRIhsz, policy->warmup_iterations));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"iterations\":%" PRIhsz, policy->iterations));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream,
      ",\"benchmark_sample_count\":%" PRIhsz
      ",\"benchmark_cartesian_sample_count\":%" PRIhsz
      ",\"benchmark_sample_count_truncated\":%s",
      benchmark_plan->sample_count, benchmark_plan->cartesian_sample_count,
      benchmark_plan->sample_count_truncated ? "true" : "false"));
  if (options->sample_ordinal >= 0) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        &stream, ",\"selected_sample\":%" PRId32, options->sample_ordinal));
  }
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_actual_invocation_plan_json(
      module, case_plan, &stream));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_case_sample_plan_fields_json(
      module, case_plan, &stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"cli_overrides\":{"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, "\"iterations\":%s,\"warmup_iterations\":%s",
      options->iterations_specified ? "true" : "false",
      options->warmup_iterations_specified ? "true" : "false"));
  if (policy->measure_kind == IREE_BENCHMARK_LOOM_MEASURE_DISPATCH_COMPLETE) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        &stream,
        ",\"batch_size\":%s,\"min_time_ms\":%s,\"warmup_time_ms\":%s,"
        "\"max_batches\":%s,\"stable_p90_to_p50_ppm\":%s,"
        "\"profile_final_batch\":%s,\"profile_data\":%s,"
        "\"profile_counter\":%s,\"profile_artifacts_dir\":%s,"
        "\"input_ring_min_bytes\":%s,\"input_ring_count\":%s",
        options->batch_size_specified ? "true" : "false",
        options->min_time_ms_specified ? "true" : "false",
        options->warmup_time_ms_specified ? "true" : "false",
        options->max_batches_specified ? "true" : "false",
        options->stable_p90_to_p50_ppm_specified ? "true" : "false",
        options->profile_final_batch_specified ? "true" : "false",
        options->profile_data_requested ? "true" : "false",
        options->profile_counters.count != 0 ? "true" : "false",
        !iree_string_view_is_empty(options->profile_artifacts_dir) ? "true"
                                                                   : "false",
        options->input_ring_min_bytes_specified ? "true" : "false",
        options->input_ring_count_specified ? "true" : "false"));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}"));
  if (policy->measure_kind == IREE_BENCHMARK_LOOM_MEASURE_DISPATCH_COMPLETE) {
    const loom_run_benchmark_options_t* timing = &policy->hal_options.timing;
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        &stream, ",\"batch_size\":%" PRIhsz, timing->batch_size));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        &stream, ",\"min_time_ns\":%" PRIi64, timing->min_duration_ns));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_format(&stream, ",\"warmup_time_ns\":%" PRIi64,
                                        timing->warmup_min_duration_ns));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        &stream, ",\"max_batches\":%" PRIhsz, timing->max_batch_count));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        &stream, ",\"stable_p90_to_p50_ppm\":%" PRIu64,
        timing->stable_p90_to_p50_delta_ppm));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        &stream, ",\"profile_final_batch\":%s",
        iree_all_bits_set(policy->hal_options.flags,
                          LOOM_RUN_HAL_BENCHMARK_FLAG_PROFILE_FINAL_BATCH)
            ? "true"
            : "false"));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"data_cache_policy\":{"));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
        &stream, "\"validity\":\"check_ops\""));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
        &stream, ",\"cache_policy\":\"binding_ring\""));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        &stream, ",\"input_ring_min_bytes\":%" PRIi64,
        options->input_ring_min_bytes));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        &stream, ",\"input_ring_count\":%" PRIhsz, options->input_ring_count));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}"));
    if (iree_all_bits_set(policy->hal_options.flags,
                          LOOM_RUN_HAL_BENCHMARK_FLAG_PROFILE_FINAL_BATCH)) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
          &stream, ",\"profile_data_families\":%" PRIu64,
          policy->hal_options.profile_data_families));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
          &stream, ",\"profile_data_family_names\":"));
      IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_profile_family_names_json(
          policy->hal_options.profile_data_families, &stream));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
          &stream, ",\"profile_counter_set_count\":%" PRIhsz,
          policy->hal_options.profile_counter_set_count));
      if (policy->hal_options.profile_counter_set_count != 0) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
            &stream, ",\"profile_counter_request\":"));
        IREE_RETURN_IF_ERROR(
            iree_benchmark_loom_write_profile_counter_request_json(policy,
                                                                   &stream));
      }
      iree_string_builder_t profile_artifacts_dir;
      iree_string_builder_initialize(allocator, &profile_artifacts_dir);
      iree_status_t profile_artifacts_status =
          iree_benchmark_loom_append_effective_profile_artifacts_dir(
              run, policy->hal_options.profile_data_families,
              &profile_artifacts_dir);
      if (iree_status_is_ok(profile_artifacts_status) &&
          iree_string_builder_size(&profile_artifacts_dir) != 0) {
        profile_artifacts_status = loom_output_stream_write_cstring(
            &stream, ",\"profile_artifacts_dir\":");
        if (iree_status_is_ok(profile_artifacts_status)) {
          profile_artifacts_status = loom_json_write_escaped_string(
              &stream, iree_string_builder_view(&profile_artifacts_dir));
        }
      }
      iree_string_builder_deinitialize(&profile_artifacts_dir);
      IREE_RETURN_IF_ERROR(profile_artifacts_status);
    }
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}\n"));
  return iree_ok_status();
}

iree_status_t iree_benchmark_loom_write_timing_stats_json(
    const iree_benchmark_loom_timing_stats_t* stats,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "\"count\":%" PRIhsz, stats->count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"total\":%" PRIi64, stats->total_ns));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"min\":%" PRIi64, stats->minimum_ns));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"max\":%" PRIi64, stats->maximum_ns));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(stream, ",\"mean\":%.3f",
                                                       stats->mean_ns));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"p50\":%" PRIi64, stats->p50_ns));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"p90\":%" PRIi64, stats->p90_ns));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}"));
  return iree_ok_status();
}

iree_status_t iree_benchmark_loom_write_benchmark_timing_stats_json(
    const loom_run_benchmark_timing_stats_t* stats,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "\"count\":%" PRIhsz, stats->count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"total\":%" PRIi64, stats->total_ns));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"min\":%" PRIi64, stats->minimum_ns));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"max\":%" PRIi64, stats->maximum_ns));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(stream, ",\"mean\":%.3f",
                                                       stats->mean_ns));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"p50\":%" PRIi64, stats->p50_ns));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"p90\":%" PRIi64, stats->p90_ns));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"p90_to_p50_delta_ppm\":%" PRIu64,
      stats->p90_to_p50_delta_ppm));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}"));
  return iree_ok_status();
}

static const char* iree_benchmark_loom_profile_statistics_row_type_name(
    iree_hal_profile_statistics_row_type_t row_type) {
  switch (row_type) {
    case IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_DISPATCH_FUNCTION:
      return "dispatch_function";
    case IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_DISPATCH_COMMAND_BUFFER:
      return "dispatch_command_buffer";
    case IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_DISPATCH_COMMAND_OPERATION:
      return "dispatch_command_operation";
    case IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_QUEUE_DEVICE_OPERATION:
      return "queue_device_operation";
    case IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_QUEUE_HOST_OPERATION:
      return "queue_host_operation";
    case IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_HOST_EXECUTION_FUNCTION:
      return "host_execution_function";
    case IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_HOST_EXECUTION_COMMAND_BUFFER:
      return "host_execution_command_buffer";
    case IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_HOST_EXECUTION_COMMAND_OPERATION:
      return "host_execution_command_operation";
    case IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_HOST_EXECUTION_QUEUE_OPERATION:
      return "host_execution_queue_operation";
    case IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_MEMORY_LIFECYCLE:
      return "memory_lifecycle";
    default:
      return "unknown";
  }
}

static const char* iree_benchmark_loom_profile_statistics_time_domain_name(
    iree_hal_profile_statistics_time_domain_t time_domain) {
  switch (time_domain) {
    case IREE_HAL_PROFILE_STATISTICS_TIME_DOMAIN_DEVICE_TICK:
      return "device_tick";
    case IREE_HAL_PROFILE_STATISTICS_TIME_DOMAIN_IREE_HOST_TIME_NS:
      return "iree_host_time_ns";
    default:
      return "none";
  }
}

static iree_status_t iree_benchmark_loom_write_profile_flag_names_json(
    iree_hal_device_profiling_flags_t flags, loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "["));
  if (iree_all_bits_set(
          flags, IREE_HAL_DEVICE_PROFILING_FLAG_LIGHTWEIGHT_STATISTICS)) {
    IREE_RETURN_IF_ERROR(
        loom_json_write_escaped_cstring(stream, "lightweight_statistics"));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "]"));
  return iree_ok_status();
}

static iree_status_t iree_benchmark_loom_write_hal_profile_duration_json(
    const loom_run_hal_profile_row_summary_t* row,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "\"available\":%s",
      iree_all_bits_set(row->flags, IREE_HAL_PROFILE_STATISTICS_ROW_FLAG_TIMING)
          ? "true"
          : "false"));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"time_domain\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(
      stream, iree_benchmark_loom_profile_statistics_time_domain_name(
                  row->time_domain)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream,
      ",\"first_start\":%" PRIu64 ",\"last_end\":%" PRIu64 ",\"total\":%" PRIu64
      ",\"min\":%" PRIu64 ",\"max\":%" PRIu64,
      row->first_start_time, row->last_end_time, row->total_duration,
      row->minimum_duration, row->maximum_duration));
  const uint64_t valid_sample_count =
      row->sample_count >= row->invalid_sample_count
          ? row->sample_count - row->invalid_sample_count
          : 0;
  if (valid_sample_count != 0) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, ",\"mean\":%" PRIu64,
        row->total_duration / valid_sample_count));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"scaled_ns\":%s",
      row->has_scaled_duration_ns ? "true" : "false"));
  if (row->has_scaled_duration_ns) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream,
        ",\"total_ns\":%" PRIu64 ",\"min_ns\":%" PRIu64 ",\"max_ns\":%" PRIu64,
        row->total_duration_ns, row->minimum_duration_ns,
        row->maximum_duration_ns));
    if (valid_sample_count != 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
          stream, ",\"mean_ns\":%" PRIu64,
          row->total_duration_ns / valid_sample_count));
    }
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}"));
  return iree_ok_status();
}

static iree_status_t iree_benchmark_loom_write_hal_profile_row_summary_json(
    const loom_run_hal_profile_row_summary_t* row,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\"type\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(
      stream,
      iree_benchmark_loom_profile_statistics_row_type_name(row->row_type)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"row_type\":%" PRIu32 ",\"flags\":%" PRIu32, row->row_type,
      row->flags));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream,
      ",\"physical_device_ordinal\":%" PRIu32 ",\"queue_ordinal\":%" PRIu32
      ",\"event_type\":%" PRIu32,
      row->physical_device_ordinal, row->queue_ordinal, row->event_type));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream,
      ",\"executable_id\":%" PRIu64 ",\"command_buffer_id\":%" PRIu64
      ",\"function_ordinal\":%" PRIu32 ",\"command_index\":%" PRIu32,
      row->executable_id, row->command_buffer_id, row->function_ordinal,
      row->command_index));
  if (row->function_name_length != 0) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"function_name\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        stream,
        iree_make_string_view(row->function_name, row->function_name_length)));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream,
      ",\"sample_count\":%" PRIu64 ",\"invalid_sample_count\":%" PRIu64
      ",\"operation_count\":%" PRIu64 ",\"payload_bytes\":%" PRIu64
      ",\"tile_count\":%" PRIu64 ",\"tile_duration_sum_ns\":%" PRIu64,
      row->sample_count, row->invalid_sample_count, row->operation_count,
      row->payload_bytes, row->tile_count, row->tile_duration_sum_ns));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"timing\":"));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_hal_profile_duration_json(row, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}"));
  return iree_ok_status();
}

iree_status_t iree_benchmark_loom_write_hal_profile_summary_json(
    const loom_run_hal_profile_summary_t* profile,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "\"requested\":%s", profile->requested ? "true" : "false"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"executed\":%s", profile->executed ? "true" : "false"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"flags\":%" PRIu32, profile->flags));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"flag_names\":"));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_profile_flag_names_json(
      profile->flags, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"data_families\":%" PRIu64, profile->data_families));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"data_family_names\":"));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_profile_family_names_json(
      profile->data_families, stream));
  if (profile->has_artifact_path) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"artifact_path\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        stream, iree_make_string_view(profile->artifact_path,
                                      profile->artifact_path_length)));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"row_count\":%" PRIhsz, profile->row_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"captured_row_count\":%" PRIhsz, profile->captured_row_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"truncated_row_count\":%" PRIhsz,
      profile->truncated_row_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"dropped_record_count\":%" PRIu64,
      profile->dropped_record_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\"rows\":["));
  for (iree_host_size_t i = 0; i < profile->captured_row_count; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
    }
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_hal_profile_row_summary_json(
        &profile->rows[i], stream));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "]"));
  if (profile->has_error) {
    bool first_field = false;
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_status_field_json(
        profile->error_code,
        iree_make_string_view(profile->error_message,
                              profile->error_message_length),
        stream, &first_field));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}"));
  return iree_ok_status();
}

static iree_string_view_t iree_benchmark_loom_benchmark_result_state(
    const iree_benchmark_loom_benchmark_result_t* benchmark_result) {
  if (!iree_string_view_is_empty(benchmark_result->state)) {
    return benchmark_result->state;
  }
  if (benchmark_result->executed) {
    return benchmark_result->passed ? IREE_SV("ok") : IREE_SV("failed");
  }
  return IREE_SV("skipped");
}

iree_status_t iree_benchmark_loom_write_benchmark_failure_json(
    const iree_benchmark_loom_benchmark_result_t* benchmark_result,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  bool first_field = true;
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      stream, &first_field, "stage", benchmark_result->failure_stage));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      stream, &first_field, "kind", benchmark_result->failure_kind));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_optional_string_field(
      stream, &first_field, "message", benchmark_result->failure_message));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
      stream, &first_field, "diagnostic_error_count",
      benchmark_result->diagnostic_error_count));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
      stream, &first_field, "diagnostic_warning_count",
      benchmark_result->diagnostic_warning_count));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
      stream, &first_field, "diagnostic_remark_count",
      benchmark_result->diagnostic_remark_count));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
      stream, &first_field, "diagnostics"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "["));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write(stream, benchmark_result->diagnostic_json));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "]"));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t iree_benchmark_loom_write_compile_rejection_fields_json(
    const iree_benchmark_loom_hal_actual_provider_t* provider,
    loom_output_stream_t* stream) {
  bool first_field = false;
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      stream, &first_field, "stage",
      provider->execution.compile_failure_stage));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      stream, &first_field, "kind", provider->execution.compile_failure_kind));
  return iree_benchmark_loom_write_json_optional_string_field(
      stream, &first_field, "message",
      provider->execution.compile_failure_message);
}

iree_status_t iree_benchmark_loom_write_json_object_field_name(
    loom_output_stream_t* stream, bool* first_field, const char* name) {
  if (!*first_field) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
  }
  *first_field = false;
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(stream, name));
  return loom_output_stream_write_cstring(stream, ":");
}

iree_status_t iree_benchmark_loom_write_json_string_field(
    loom_output_stream_t* stream, bool* first_field, const char* name,
    iree_string_view_t value) {
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
      stream, first_field, name));
  return loom_json_write_escaped_string(stream, value);
}

iree_status_t iree_benchmark_loom_write_json_optional_string_field(
    loom_output_stream_t* stream, bool* first_field, const char* name,
    iree_string_view_t value) {
  if (iree_string_view_is_empty(value)) {
    return iree_ok_status();
  }
  return iree_benchmark_loom_write_json_string_field(stream, first_field, name,
                                                     value);
}

iree_status_t iree_benchmark_loom_write_json_u32_field(
    loom_output_stream_t* stream, bool* first_field, const char* name,
    uint32_t value) {
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
      stream, first_field, name));
  return loom_output_stream_write_format(stream, "%" PRIu32, value);
}

iree_status_t iree_benchmark_loom_write_json_u64_field(
    loom_output_stream_t* stream, bool* first_field, const char* name,
    uint64_t value) {
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
      stream, first_field, name));
  return loom_output_stream_write_format(stream, "%" PRIu64, value);
}

iree_status_t iree_benchmark_loom_write_json_i64_field(
    loom_output_stream_t* stream, bool* first_field, const char* name,
    int64_t value) {
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
      stream, first_field, name));
  return loom_output_stream_write_format(stream, "%" PRIi64, value);
}

iree_status_t iree_benchmark_loom_write_json_bool_field(
    loom_output_stream_t* stream, bool* first_field, const char* name,
    bool value) {
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
      stream, first_field, name));
  return loom_output_stream_write_cstring(stream, value ? "true" : "false");
}

iree_status_t iree_benchmark_loom_write_json_size_field(
    loom_output_stream_t* stream, bool* first_field, const char* name,
    iree_host_size_t value) {
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
      stream, first_field, name));
  return loom_output_stream_write_format(stream, "%" PRIhsz, value);
}

iree_status_t iree_benchmark_loom_write_diagnostic_capture_fields_json(
    const iree_benchmark_loom_diagnostic_capture_t* diagnostics,
    loom_output_stream_t* stream, bool* first_field) {
  if (diagnostics == NULL) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
      stream, first_field, "diagnostic_error_count", diagnostics->error_count));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
      stream, first_field, "diagnostic_warning_count",
      diagnostics->warning_count));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
      stream, first_field, "diagnostic_remark_count",
      diagnostics->remark_count));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
      stream, first_field, "diagnostics"));
  return iree_benchmark_loom_write_diagnostic_array_json(diagnostics, stream);
}

iree_status_t iree_benchmark_loom_write_planning_issue_fields_json(
    const loom_testbench_module_plan_t* testbench_plan,
    const loom_testbench_issue_t* planning_issues,
    iree_host_size_t planning_issue_count, loom_output_stream_t* stream,
    bool* first_field) {
  if (planning_issue_count == 0) {
    return iree_ok_status();
  }
  IREE_ASSERT_ARGUMENT(testbench_plan);
  IREE_ASSERT_ARGUMENT(planning_issues);
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
      stream, first_field, "planning_issue_count", planning_issue_count));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
      stream, first_field, "planning_issues"));
  return loom_testbench_issue_array_write_json(testbench_plan, planning_issues,
                                               planning_issue_count, stream);
}

enum {
  IREE_BENCHMARK_LOOM_SHORT_MEASURED_DURATION_NS = 1000 * 1000,
  IREE_BENCHMARK_LOOM_SUB_MICROSECOND_DURATION_NS = 1000,
  IREE_BENCHMARK_LOOM_SMALL_PHYSICAL_DISPATCH_SAMPLE_COUNT = 100,
};

typedef struct iree_benchmark_loom_profiled_dispatch_timing_t {
  // True when at least one profiled dispatch-function row contributed.
  bool available;
  // True when every contributing row uses the same device/time domain.
  bool comparable;
  // True when every contributing row has scaled nanosecond durations.
  bool has_scaled_duration_ns;
  // True when contributing dispatch durations overlap in the final batch.
  bool overlapped;
  // True when captured profile rows were truncated before reporting.
  bool truncated;
  // Number of dispatch-function rows that contributed to the summary.
  iree_host_size_t row_count;
  // Number of dispatch-function rows ignored as non-comparable.
  iree_host_size_t ignored_row_count;
  // Physical device ordinal for the comparable timing group.
  uint32_t physical_device_ordinal;
  // Time domain used by raw timing fields.
  iree_hal_profile_statistics_time_domain_t time_domain;
  // Source dispatch samples represented by contributing rows.
  uint64_t sample_count;
  // Source dispatch samples rejected by profiling.
  uint64_t invalid_sample_count;
  // Valid profiled dispatch samples represented by contributing rows.
  uint64_t valid_sample_count;
  // Sum of valid dispatch durations in |time_domain| units.
  uint64_t total_duration;
  // Minimum valid dispatch duration in |time_domain| units.
  uint64_t minimum_duration;
  // Maximum valid dispatch duration in |time_domain| units.
  uint64_t maximum_duration;
  // Earliest valid dispatch start in |time_domain| units.
  uint64_t first_start_time;
  // Latest valid dispatch end in |time_domain| units.
  uint64_t last_end_time;
  // Covered final-batch span in |time_domain| units.
  uint64_t span_duration;
  // Ratio of total dispatch duration to covered final-batch span.
  uint64_t overlap_ratio_ppm;
  // Sum of valid dispatch durations in nanoseconds when scaled.
  uint64_t total_duration_ns;
  // Minimum valid dispatch duration in nanoseconds when scaled.
  uint64_t minimum_duration_ns;
  // Maximum valid dispatch duration in nanoseconds when scaled.
  uint64_t maximum_duration_ns;
} iree_benchmark_loom_profiled_dispatch_timing_t;

static uint64_t iree_benchmark_loom_saturating_add_u64(uint64_t lhs,
                                                       uint64_t rhs) {
  return UINT64_MAX - lhs < rhs ? UINT64_MAX : lhs + rhs;
}

static bool iree_benchmark_loom_profile_row_has_timing(
    const loom_run_hal_profile_row_summary_t* row) {
  return iree_all_bits_set(row->flags,
                           IREE_HAL_PROFILE_STATISTICS_ROW_FLAG_TIMING) &&
         row->sample_count > row->invalid_sample_count &&
         row->last_end_time >= row->first_start_time;
}

static void iree_benchmark_loom_accumulate_profiled_dispatch_timing(
    const loom_run_hal_profile_row_summary_t* row,
    iree_benchmark_loom_profiled_dispatch_timing_t* timing) {
  const uint64_t valid_sample_count =
      row->sample_count - row->invalid_sample_count;
  if (!timing->available) {
    timing->available = true;
    timing->comparable = true;
    timing->has_scaled_duration_ns = row->has_scaled_duration_ns;
    timing->physical_device_ordinal = row->physical_device_ordinal;
    timing->time_domain = row->time_domain;
    timing->minimum_duration = row->minimum_duration;
    timing->maximum_duration = row->maximum_duration;
    timing->first_start_time = row->first_start_time;
    timing->last_end_time = row->last_end_time;
    if (row->has_scaled_duration_ns) {
      timing->minimum_duration_ns = row->minimum_duration_ns;
      timing->maximum_duration_ns = row->maximum_duration_ns;
    }
  } else if (timing->physical_device_ordinal != row->physical_device_ordinal ||
             timing->time_domain != row->time_domain) {
    timing->comparable = false;
    ++timing->ignored_row_count;
    return;
  } else {
    timing->has_scaled_duration_ns =
        timing->has_scaled_duration_ns && row->has_scaled_duration_ns;
    if (row->minimum_duration < timing->minimum_duration) {
      timing->minimum_duration = row->minimum_duration;
    }
    if (row->maximum_duration > timing->maximum_duration) {
      timing->maximum_duration = row->maximum_duration;
    }
    if (row->first_start_time < timing->first_start_time) {
      timing->first_start_time = row->first_start_time;
    }
    if (row->last_end_time > timing->last_end_time) {
      timing->last_end_time = row->last_end_time;
    }
    if (row->has_scaled_duration_ns) {
      if (row->minimum_duration_ns < timing->minimum_duration_ns) {
        timing->minimum_duration_ns = row->minimum_duration_ns;
      }
      if (row->maximum_duration_ns > timing->maximum_duration_ns) {
        timing->maximum_duration_ns = row->maximum_duration_ns;
      }
    }
  }

  ++timing->row_count;
  timing->sample_count = iree_benchmark_loom_saturating_add_u64(
      timing->sample_count, row->sample_count);
  timing->invalid_sample_count = iree_benchmark_loom_saturating_add_u64(
      timing->invalid_sample_count, row->invalid_sample_count);
  timing->valid_sample_count = iree_benchmark_loom_saturating_add_u64(
      timing->valid_sample_count, valid_sample_count);
  timing->total_duration = iree_benchmark_loom_saturating_add_u64(
      timing->total_duration, row->total_duration);
  if (row->has_scaled_duration_ns) {
    timing->total_duration_ns = iree_benchmark_loom_saturating_add_u64(
        timing->total_duration_ns, row->total_duration_ns);
  }
}

static iree_benchmark_loom_profiled_dispatch_timing_t
iree_benchmark_loom_profiled_dispatch_timing(
    const loom_run_hal_profile_summary_t* profile) {
  iree_benchmark_loom_profiled_dispatch_timing_t timing = {0};
  timing.truncated = profile->truncated_row_count != 0;
  for (iree_host_size_t i = 0; i < profile->captured_row_count; ++i) {
    const loom_run_hal_profile_row_summary_t* row = &profile->rows[i];
    if (row->row_type !=
            IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_DISPATCH_FUNCTION ||
        !iree_benchmark_loom_profile_row_has_timing(row)) {
      continue;
    }
    iree_benchmark_loom_accumulate_profiled_dispatch_timing(row, &timing);
  }
  if (timing.available && timing.comparable) {
    timing.span_duration = timing.last_end_time - timing.first_start_time;
    timing.overlapped = timing.valid_sample_count > 1 &&
                        timing.span_duration != 0 &&
                        timing.total_duration > timing.span_duration;
    if (timing.span_duration != 0) {
      timing.overlap_ratio_ppm =
          (uint64_t)(((double)timing.total_duration * 1000000.0) /
                     (double)timing.span_duration);
    }
  }
  return timing;
}

static iree_status_t iree_benchmark_loom_validate_hal_benchmark_result(
    const iree_benchmark_loom_benchmark_result_t* benchmark_result) {
  if (!benchmark_result->has_hal_benchmark) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL benchmark timing metadata requires a HAL "
                            "benchmark result");
  }
  const loom_run_benchmark_result_t* timing =
      &benchmark_result->hal_benchmark.timing;
  if (timing->batch_size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL benchmark result has zero batch size");
  }
  iree_host_size_t expected_operation_count = 0;
  if (!iree_host_size_checked_mul(timing->measured_batch_count,
                                  timing->batch_size,
                                  &expected_operation_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "HAL benchmark expected operation count "
                            "overflowed host size limits");
  }
  if (timing->measured_operation_count != expected_operation_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HAL benchmark result measured operation count does not match measured "
        "batch count and batch size");
  }
  return iree_ok_status();
}

iree_status_t iree_benchmark_loom_hal_physical_dispatches_per_batch(
    const iree_benchmark_loom_benchmark_result_t* benchmark_result,
    iree_host_size_t* out_dispatch_count) {
  *out_dispatch_count = 0;
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_validate_hal_benchmark_result(benchmark_result));
  const loom_run_benchmark_result_t* timing =
      &benchmark_result->hal_benchmark.timing;
  *out_dispatch_count = benchmark_result->data_cache.populated
                            ? benchmark_result->data_cache.dispatches_per_batch
                            : timing->batch_size;
  if (*out_dispatch_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL benchmark physical dispatch count per batch "
                            "must be positive");
  }
  return iree_ok_status();
}

iree_status_t iree_benchmark_loom_hal_physical_dispatches_per_logical_operation(
    const iree_benchmark_loom_benchmark_result_t* benchmark_result,
    iree_host_size_t* out_dispatch_count) {
  *out_dispatch_count = 0;
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_validate_hal_benchmark_result(benchmark_result));
  iree_host_size_t dispatches_per_batch = 0;
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_hal_physical_dispatches_per_batch(
      benchmark_result, &dispatches_per_batch));
  const iree_host_size_t logical_operations_per_batch =
      benchmark_result->hal_benchmark.timing.batch_size;
  if (dispatches_per_batch % logical_operations_per_batch != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HAL benchmark physical dispatches per batch %" PRIhsz
        " must be an integer multiple of logical operations per batch %" PRIhsz,
        dispatches_per_batch, logical_operations_per_batch);
  }
  *out_dispatch_count = dispatches_per_batch / logical_operations_per_batch;
  return iree_ok_status();
}

iree_status_t iree_benchmark_loom_hal_measured_physical_dispatch_count(
    const iree_benchmark_loom_benchmark_result_t* benchmark_result,
    iree_host_size_t* out_dispatch_count) {
  *out_dispatch_count = 0;
  iree_host_size_t dispatches_per_batch = 0;
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_hal_physical_dispatches_per_batch(
      benchmark_result, &dispatches_per_batch));
  if (!iree_host_size_checked_mul(
          benchmark_result->hal_benchmark.timing.measured_batch_count,
          dispatches_per_batch, out_dispatch_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "measured physical dispatch count overflowed host "
                            "size limits");
  }
  return iree_ok_status();
}

iree_status_t iree_benchmark_loom_hal_mean_physical_dispatch_duration_ns(
    const iree_benchmark_loom_benchmark_result_t* benchmark_result,
    double* out_duration_ns) {
  *out_duration_ns = 0.0;
  iree_host_size_t measured_physical_dispatch_count = 0;
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_hal_measured_physical_dispatch_count(
      benchmark_result, &measured_physical_dispatch_count));
  if (measured_physical_dispatch_count == 0) {
    return iree_ok_status();
  }
  *out_duration_ns =
      (double)benchmark_result->hal_benchmark.timing.measured_duration_ns /
      (double)measured_physical_dispatch_count;
  return iree_ok_status();
}

static iree_status_t iree_benchmark_loom_write_timing_warning_json(
    loom_output_stream_t* stream, bool* first_warning,
    iree_string_view_t warning) {
  if (!*first_warning) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
  }
  *first_warning = false;
  return loom_json_write_escaped_string(stream, warning);
}

static iree_status_t
iree_benchmark_loom_write_profiled_dispatch_duration_ns_json(
    const iree_benchmark_loom_profiled_dispatch_timing_t* timing,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "\"count\":%" PRIu64, timing->valid_sample_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"total\":%" PRIu64, timing->total_duration_ns));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"min\":%" PRIu64, timing->minimum_duration_ns));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"max\":%" PRIu64, timing->maximum_duration_ns));
  if (timing->valid_sample_count != 0) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, ",\"mean\":%.3f",
        (double)timing->total_duration_ns /
            (double)timing->valid_sample_count));
  }
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t iree_benchmark_loom_write_profiled_dispatch_timing_json(
    const iree_benchmark_loom_profiled_dispatch_timing_t* timing,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "\"available\":%s", timing->available ? "true" : "false"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"comparable\":%s", timing->comparable ? "true" : "false"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"overlapped\":%s", timing->overlapped ? "true" : "false"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"truncated\":%s", timing->truncated ? "true" : "false"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"row_count\":%" PRIhsz, timing->row_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"ignored_row_count\":%" PRIhsz, timing->ignored_row_count));
  if (timing->available) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream,
        ",\"physical_device_ordinal\":%" PRIu32 ",\"sample_count\":%" PRIu64
        ",\"invalid_sample_count\":%" PRIu64 ",\"valid_sample_count\":%" PRIu64
        ",\"span\":%" PRIu64 ",\"total\":%" PRIu64 ",\"min\":%" PRIu64
        ",\"max\":%" PRIu64 ",\"overlap_ratio_ppm\":%" PRIu64,
        timing->physical_device_ordinal, timing->sample_count,
        timing->invalid_sample_count, timing->valid_sample_count,
        timing->span_duration, timing->total_duration, timing->minimum_duration,
        timing->maximum_duration, timing->overlap_ratio_ppm));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"time_domain\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(
        stream, iree_benchmark_loom_profile_statistics_time_domain_name(
                    timing->time_domain)));
  }
  if (timing->available && timing->has_scaled_duration_ns) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"duration_ns\":"));
    IREE_RETURN_IF_ERROR(
        iree_benchmark_loom_write_profiled_dispatch_duration_ns_json(timing,
                                                                     stream));
  }
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t iree_benchmark_loom_write_hal_timing_warnings_json(
    const iree_benchmark_loom_benchmark_policy_t* policy,
    const iree_benchmark_loom_benchmark_result_t* benchmark_result,
    loom_output_stream_t* stream) {
  const loom_run_benchmark_result_t* timing =
      &benchmark_result->hal_benchmark.timing;
  const loom_run_benchmark_timing_stats_t* operation_timing =
      &timing->operation_timing;
  iree_host_size_t measured_physical_dispatch_count = 0;
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_hal_measured_physical_dispatch_count(
      benchmark_result, &measured_physical_dispatch_count));
  bool first_warning = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "["));
  if (timing->measured_duration_ns <
      IREE_BENCHMARK_LOOM_SHORT_MEASURED_DURATION_NS) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_timing_warning_json(
        stream, &first_warning, IREE_SV("short_measured_duration")));
  }
  if (timing->batch_size == 1) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_timing_warning_json(
        stream, &first_warning, IREE_SV("single_logical_operation_batch")));
  }
  if (measured_physical_dispatch_count <
      IREE_BENCHMARK_LOOM_SMALL_PHYSICAL_DISPATCH_SAMPLE_COUNT) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_timing_warning_json(
        stream, &first_warning, IREE_SV("low_physical_dispatch_sample_count")));
  }
  if (operation_timing->p50_ns <
      IREE_BENCHMARK_LOOM_SUB_MICROSECOND_DURATION_NS) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_timing_warning_json(
        stream, &first_warning, IREE_SV("sub_microsecond_logical_operation")));
  }
  const uint64_t accepted_spread =
      policy->hal_options.timing.stable_p90_to_p50_delta_ppm;
  if (accepted_spread != 0 &&
      operation_timing->p90_to_p50_delta_ppm > accepted_spread) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_timing_warning_json(
        stream, &first_warning, IREE_SV("unstable_p90_to_p50")));
  }
  const iree_benchmark_loom_profiled_dispatch_timing_t profiled_dispatch =
      iree_benchmark_loom_profiled_dispatch_timing(
          &benchmark_result->hal_benchmark.profile);
  if (timing->batch_size > 1 && profiled_dispatch.overlapped) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_timing_warning_json(
        stream, &first_warning, IREE_SV("profiled_dispatch_overlap")));
  }
  if (profiled_dispatch.truncated) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_timing_warning_json(
        stream, &first_warning, IREE_SV("profiled_dispatch_rows_truncated")));
  }
  return loom_output_stream_write_cstring(stream, "]");
}

iree_status_t iree_benchmark_loom_write_hal_timing_interpretation_json(
    const iree_benchmark_loom_benchmark_policy_t* policy,
    const iree_benchmark_loom_benchmark_result_t* benchmark_result,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_validate_hal_benchmark_result(benchmark_result));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  bool first_field = true;
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      stream, &first_field, "score", IREE_SV("operation_timing_ns")));
  const iree_benchmark_loom_profiled_dispatch_timing_t profiled_dispatch =
      iree_benchmark_loom_profiled_dispatch_timing(
          &benchmark_result->hal_benchmark.profile);
  const bool overlapped_logical_batch =
      benchmark_result->hal_benchmark.timing.batch_size > 1 &&
      profiled_dispatch.overlapped;
  const iree_string_view_t score_meaning =
      overlapped_logical_batch ? IREE_SV("throughput_normalized_batch_time")
                               : IREE_SV("normalized_logical_operation_time");
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      stream, &first_field, "score_meaning", score_meaning));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      stream, &first_field, "score_unit", IREE_SV("logical_operation")));
  if (profiled_dispatch.available) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_bool_field(
        stream, &first_field, "profiled_dispatch_overlap",
        profiled_dispatch.overlapped));
  }
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
      stream, &first_field, "warnings"));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_hal_timing_warnings_json(
      policy, benchmark_result, stream));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t iree_benchmark_loom_write_compile_report_json(
    const loom_run_compile_report_capture_t* compile_report_capture,
    loom_output_stream_t* stream) {
  if (compile_report_capture == NULL ||
      !loom_run_compile_report_capture_is_enabled(compile_report_capture)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"compile_report\":"));
  IREE_RETURN_IF_ERROR(loom_run_compile_report_capture_append_json(
      compile_report_capture, stream));
  return iree_ok_status();
}

static iree_status_t iree_benchmark_loom_write_compile_report_field_json(
    const loom_run_compile_report_capture_t* compile_report_capture,
    loom_output_stream_t* stream, bool* first_field) {
  if (compile_report_capture == NULL ||
      !loom_run_compile_report_capture_is_enabled(compile_report_capture)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
      stream, first_field, "compile_report"));
  return loom_run_compile_report_capture_append_json(compile_report_capture,
                                                     stream);
}

static iree_status_t iree_benchmark_loom_append_candidate_artifact_stem(
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_candidate_identity_t* candidate,
    const iree_benchmark_loom_hal_actual_provider_t* provider,
    iree_string_builder_t* stem) {
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_append_sanitized_path_component(run->run_id, stem));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(stem, "_"));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_append_sanitized_path_component(
      candidate->candidate_id, stem));
  if (!iree_string_view_is_empty(provider->sample_compilation)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(stem, "_"));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_append_sanitized_path_component(
        provider->sample_compilation, stem));
  }
  if (provider->execution.has_sample_constant_ordinal) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        stem, "_sample%" PRIhsz, provider->execution.sample_constant_ordinal));
  }
  return iree_ok_status();
}

static iree_status_t iree_benchmark_loom_append_compile_report_artifact_leaf(
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_candidate_identity_t* candidate,
    const iree_benchmark_loom_hal_actual_provider_t* provider,
    iree_string_builder_t* leaf) {
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_append_candidate_artifact_stem(
      run, candidate, provider, leaf));
  return iree_string_builder_append_cstring(leaf, "_compile_report.json");
}

static iree_status_t iree_benchmark_loom_append_artifact_manifest_leaf(
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_candidate_identity_t* candidate,
    const iree_benchmark_loom_hal_actual_provider_t* provider,
    iree_string_builder_t* leaf) {
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_append_candidate_artifact_stem(
      run, candidate, provider, leaf));
  return iree_string_builder_append_cstring(leaf, "_artifact_manifest.json");
}

static iree_status_t iree_benchmark_loom_append_artifact_extension(
    iree_string_view_t format, iree_string_view_t fallback_extension,
    iree_string_builder_t* leaf) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(leaf, "."));
  if (iree_string_view_is_empty(format)) {
    return iree_benchmark_loom_append_sanitized_path_component(
        fallback_extension, leaf);
  }
  return iree_benchmark_loom_append_sanitized_path_component(format, leaf);
}

static iree_status_t iree_benchmark_loom_append_target_artifact_extension(
    loom_target_artifact_format_t format, iree_string_view_t fallback_extension,
    iree_string_builder_t* leaf) {
  iree_string_view_t format_name = iree_string_view_empty();
  if (format != LOOM_TARGET_ARTIFACT_FORMAT_UNKNOWN) {
    format_name = loom_target_artifact_format_name(format);
  }
  return iree_benchmark_loom_append_artifact_extension(
      format_name, fallback_extension, leaf);
}

static iree_status_t iree_benchmark_loom_write_candidate_byte_artifact(
    iree_benchmark_loom_artifact_bundle_t* bundle,
    iree_benchmark_loom_bundle_file_kind_t kind, iree_string_view_t directory,
    iree_string_view_t leaf, iree_const_byte_span_t contents,
    iree_allocator_t allocator, char** inout_path_storage,
    iree_string_view_t* inout_path) {
  if (!iree_benchmark_loom_artifact_bundle_wants_debug_artifacts(bundle) ||
      !iree_string_view_is_empty(*inout_path)) {
    return iree_ok_status();
  }
  if (contents.data == NULL || contents.data_length == 0) {
    return iree_ok_status();
  }

  char* path_storage = NULL;
  iree_status_t status =
      iree_benchmark_loom_join_path(directory, leaf, allocator, &path_storage);
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_create_parent_directory(
        iree_make_cstring_view(path_storage), allocator);
  }
  if (iree_status_is_ok(status)) {
    status = loom_tooling_write_output_file(
        iree_make_cstring_view(path_storage),
        iree_make_string_view((const char*)contents.data, contents.data_length),
        allocator);
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_artifact_bundle_record_file(
        bundle, kind, iree_make_cstring_view(path_storage));
  }
  if (iree_status_is_ok(status)) {
    iree_allocator_free(allocator, *inout_path_storage);
    *inout_path_storage = path_storage;
    *inout_path = iree_make_cstring_view(path_storage);
    path_storage = NULL;
  }
  iree_allocator_free(allocator, path_storage);
  return status;
}

static iree_status_t iree_benchmark_loom_find_artifact_manifest_sidecar(
    const loom_run_hal_artifact_t* artifact,
    const loom_target_emit_sidecar_artifact_t** out_sidecar) {
  *out_sidecar = NULL;
  for (iree_host_size_t i = 0; i < artifact->sidecar_count; ++i) {
    const loom_target_emit_sidecar_artifact_t* sidecar = &artifact->sidecars[i];
    if (sidecar->kind !=
        LOOM_TARGET_EMIT_SIDECAR_ARTIFACT_KIND_ARTIFACT_MANIFEST) {
      continue;
    }
    if (*out_sidecar != NULL) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "candidate emitted multiple artifact manifest sidecars");
    }
    *out_sidecar = sidecar;
  }
  return iree_ok_status();
}

iree_status_t iree_benchmark_loom_write_compiled_artifacts(
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_candidate_identity_t* candidate,
    iree_benchmark_loom_hal_actual_provider_t* provider,
    iree_allocator_t allocator) {
  iree_benchmark_loom_artifact_bundle_t* bundle =
      provider->context->artifact_bundle;
  if (!iree_benchmark_loom_artifact_bundle_wants_debug_artifacts(bundle) ||
      !provider->execution.candidate_initialized ||
      !provider->execution.candidate.compiled) {
    return iree_ok_status();
  }

  iree_string_builder_t leaf;
  iree_string_builder_initialize(allocator, &leaf);
  const loom_run_hal_artifact_t* artifact =
      &provider->execution.candidate.artifact;
  iree_status_t status = iree_benchmark_loom_append_candidate_artifact_stem(
      run, candidate, provider, &leaf);
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(&leaf, "_target");
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_append_target_artifact_extension(
        artifact->target_artifact_format, IREE_SV("bin"), &leaf);
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_write_candidate_byte_artifact(
        bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_TARGET_ARTIFACT,
        bundle->target_artifact_dir, iree_string_builder_view(&leaf),
        artifact->target_artifact_data, allocator,
        &provider->target_artifact_path_storage,
        &provider->target_artifact_path);
  }

  iree_string_builder_reset(&leaf);
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_append_candidate_artifact_stem(
        run, candidate, provider, &leaf);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(&leaf, "_target_listing");
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_append_artifact_extension(
        artifact->target_listing_format, IREE_SV("txt"), &leaf);
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_write_candidate_byte_artifact(
        bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_TARGET_LISTING,
        bundle->target_listing_dir, iree_string_builder_view(&leaf),
        artifact->target_listing_data, allocator,
        &provider->target_listing_path_storage, &provider->target_listing_path);
  }

  iree_string_builder_reset(&leaf);
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_append_candidate_artifact_stem(
        run, candidate, provider, &leaf);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(&leaf, "_hal_executable.hal");
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_write_candidate_byte_artifact(
        bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_HAL_EXECUTABLE,
        bundle->hal_executable_dir, iree_string_builder_view(&leaf),
        artifact->executable_data, allocator,
        &provider->hal_executable_path_storage, &provider->hal_executable_path);
  }

  const loom_target_emit_sidecar_artifact_t* artifact_manifest = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_find_artifact_manifest_sidecar(
        artifact, &artifact_manifest);
  }
  iree_string_builder_reset(&leaf);
  if (iree_status_is_ok(status) && artifact_manifest != NULL) {
    status = iree_benchmark_loom_append_artifact_manifest_leaf(run, candidate,
                                                               provider, &leaf);
  }
  if (iree_status_is_ok(status) && artifact_manifest != NULL) {
    status = iree_benchmark_loom_write_candidate_byte_artifact(
        bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_ARTIFACT_MANIFEST,
        bundle->artifact_manifest_dir, iree_string_builder_view(&leaf),
        artifact_manifest->contents, allocator,
        &provider->artifact_manifest_path_storage,
        &provider->artifact_manifest_path);
  }
  iree_string_builder_deinitialize(&leaf);
  return status;
}

static iree_status_t iree_benchmark_loom_append_compile_report_artifact_json(
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_candidate_identity_t* candidate,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_benchmark_loom_hal_actual_provider_t* provider,
    iree_string_builder_t* output) {
  iree_string_view_t entry_symbol = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_module_symbol_name_from_ref(
      provider->execution.test_module,
      provider->execution.actual_invocation->callee_ref, &entry_symbol));

  loom_output_stream_t stream;
  loom_output_stream_for_builder(output, &stream);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
      &stream, "{\"type\":\"compile_report\""));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_run_id_field_json(run, &stream));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_candidate_identity_json(candidate, &stream));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_sample_compilation_field_json(
      provider->sample_compilation, &stream));
  if (provider->execution.has_sample_constant_ordinal) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_sample_fields_json(
        provider->execution.test_module, case_plan,
        provider->execution.sample_constant_ordinal, &stream));
  }
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"benchmark\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(&stream, benchmark_plan->name));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ",\"case\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(&stream, case_plan->name));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"entry\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(&stream, entry_symbol));
  if (!iree_string_view_is_empty(provider->target_artifact_path)) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
        &stream, ",\"target_artifact_path\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        &stream, provider->target_artifact_path));
  }
  if (!iree_string_view_is_empty(provider->artifact_manifest_path)) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
        &stream, ",\"artifact_manifest_path\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        &stream, provider->artifact_manifest_path));
  }
  if (!iree_string_view_is_empty(provider->target_listing_path)) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"target_listing_path\":"));
    IREE_RETURN_IF_ERROR(
        loom_json_write_escaped_string(&stream, provider->target_listing_path));
  }
  if (!iree_string_view_is_empty(provider->hal_executable_path)) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"hal_executable_path\":"));
    IREE_RETURN_IF_ERROR(
        loom_json_write_escaped_string(&stream, provider->hal_executable_path));
  }
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"state\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(
      &stream, provider->execution.compile_rejected ? "failed" : "ok"));
  if (provider->execution.compile_rejected) {
    IREE_RETURN_IF_ERROR(
        iree_benchmark_loom_write_compile_rejection_fields_json(provider,
                                                                &stream));
  }
  bool first_diagnostic_field = false;
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_diagnostic_capture_fields_json(
      &provider->diagnostics, &stream, &first_diagnostic_field));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"sample_constant_argument_count\":%" PRIhsz,
      provider->execution.sample_constant_argument_count));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_compile_report_json(
      &provider->compile_report_capture, &stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}\n"));
  return iree_ok_status();
}

iree_status_t iree_benchmark_loom_write_compile_report_artifact(
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_candidate_identity_t* candidate,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    iree_benchmark_loom_hal_actual_provider_t* provider,
    iree_allocator_t allocator) {
  if (!provider->execution.compile_report_available ||
      !loom_run_compile_report_capture_is_enabled(
          &provider->compile_report_capture) ||
      !iree_string_view_is_empty(provider->compile_report_artifact_path)) {
    return iree_ok_status();
  }
  iree_benchmark_loom_artifact_bundle_t* bundle =
      provider->context->artifact_bundle;
  if (!iree_benchmark_loom_artifact_bundle_wants_compile_reports(bundle)) {
    return iree_ok_status();
  }

  iree_string_builder_t leaf;
  iree_string_builder_initialize(allocator, &leaf);
  char* path_storage = NULL;
  iree_status_t status =
      iree_benchmark_loom_append_compile_report_artifact_leaf(run, candidate,
                                                              provider, &leaf);
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_join_path(bundle->compile_report_dir,
                                           iree_string_builder_view(&leaf),
                                           allocator, &path_storage);
  }

  iree_string_builder_t artifact;
  iree_string_builder_initialize(allocator, &artifact);
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_append_compile_report_artifact_json(
        run, candidate, benchmark_plan, case_plan, provider, &artifact);
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_create_parent_directory(
        iree_make_cstring_view(path_storage), allocator);
  }
  if (iree_status_is_ok(status)) {
    status = loom_tooling_write_output_file(
        iree_make_cstring_view(path_storage),
        iree_string_builder_view(&artifact), allocator);
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_artifact_bundle_record_file(
        bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_COMPILE_REPORT,
        iree_make_cstring_view(path_storage));
  }
  if (iree_status_is_ok(status)) {
    provider->compile_report_artifact_path_storage = path_storage;
    provider->compile_report_artifact_path =
        iree_make_cstring_view(path_storage);
    path_storage = NULL;
  }
  iree_string_builder_deinitialize(&artifact);
  iree_allocator_free(allocator, path_storage);
  iree_string_builder_deinitialize(&leaf);
  return status;
}

static iree_string_view_t iree_benchmark_loom_buffer_materialization_name(
    iree_benchmark_loom_buffer_materialization_t materialization) {
  switch (materialization) {
    case IREE_BENCHMARK_LOOM_BUFFER_MATERIALIZATION_HOST_VISIBLE:
      return IREE_SV("host_visible");
    case IREE_BENCHMARK_LOOM_BUFFER_MATERIALIZATION_DEVICE_LOCAL:
      return IREE_SV("device_local");
    default:
      return iree_string_view_empty();
  }
}

static iree_status_t iree_benchmark_loom_write_data_cache_materialization_json(
    loom_output_stream_t* stream, bool* first_field, const char* field_name,
    iree_benchmark_loom_buffer_materialization_t materialization) {
  iree_string_view_t name =
      iree_benchmark_loom_buffer_materialization_name(materialization);
  if (iree_string_view_is_empty(name)) {
    return iree_ok_status();
  }
  return iree_benchmark_loom_write_json_string_field(stream, first_field,
                                                     field_name, name);
}

static iree_status_t iree_benchmark_loom_write_data_cache_summary_json(
    const iree_benchmark_loom_data_cache_summary_t* summary,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  bool first_field = true;
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      stream, &first_field, "validity", IREE_SV("check_ops")));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      stream, &first_field, "cache_policy", IREE_SV("binding_ring")));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_data_cache_materialization_json(
          stream, &first_field, "correctness_materialization",
          summary->correctness_materialization));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_data_cache_materialization_json(
          stream, &first_field, "measurement_materialization",
          summary->measurement_materialization));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
      stream, &first_field, "binding_count", summary->binding_count));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
      stream, &first_field, "binding_ring_count", summary->binding_ring_count));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
      stream, &first_field, "command_buffer_ring_count",
      summary->command_buffer_ring_count));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
      stream, &first_field, "dispatches_per_batch",
      summary->dispatches_per_batch));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
      stream, &first_field, "requested_min_ring_bytes",
      summary->requested_min_ring_bytes));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
      stream, &first_field, "binding_set_bytes", summary->binding_set_bytes));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
      stream, &first_field, "binding_ring_bytes", summary->binding_ring_bytes));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_bool_field(
      stream, &first_field, "hot_reuse", summary->binding_ring_count <= 1));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t iree_benchmark_loom_write_data_cache_summary_field_json(
    const iree_benchmark_loom_data_cache_summary_t* summary,
    loom_output_stream_t* stream, bool* first_field) {
  if (!summary->populated) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
      stream, first_field, "data_cache"));
  return iree_benchmark_loom_write_data_cache_summary_json(summary, stream);
}

iree_status_t iree_benchmark_loom_write_benchmark_policy_json(
    const iree_benchmark_loom_benchmark_policy_t* policy,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  bool first_field = true;
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      stream, &first_field, "measure", policy->measure));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
      stream, &first_field, "warmup_iterations", policy->warmup_iterations));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
      stream, &first_field, "iterations", policy->iterations));
  return loom_output_stream_write_cstring(stream, "}");
}

iree_status_t iree_benchmark_loom_write_benchmark_correctness_json(
    iree_host_size_t sample_count, iree_host_size_t failed_sample_count,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  bool first_field = true;
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
      stream, &first_field, "sample_count", sample_count));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
      stream, &first_field, "failed_sample_count", failed_sample_count));
  return loom_output_stream_write_cstring(stream, "}");
}

static bool iree_benchmark_loom_benchmark_result_has_measurement(
    const iree_benchmark_loom_benchmark_result_t* benchmark_result) {
  return benchmark_result->executed || benchmark_result->has_hal_benchmark;
}

iree_status_t iree_benchmark_loom_write_benchmark_measurement_json(
    const iree_benchmark_loom_benchmark_policy_t* policy,
    const iree_benchmark_loom_benchmark_result_t* benchmark_result,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  bool first_field = true;
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
      stream, &first_field, "samples_per_iteration",
      benchmark_result->samples_per_iteration));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
      stream, &first_field, "failed_sample_count",
      benchmark_result->failed_sample_count));
  if (benchmark_result->executed && !benchmark_result->has_hal_benchmark) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
        stream, &first_field, "timing_ns"));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_timing_stats_json(
        &benchmark_result->timing, stream));
  }
  if (benchmark_result->has_hal_benchmark) {
    const loom_run_benchmark_result_t* timing =
        &benchmark_result->hal_benchmark.timing;
    iree_host_size_t physical_dispatches_per_batch = 0;
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_hal_physical_dispatches_per_batch(
        benchmark_result, &physical_dispatches_per_batch));
    iree_host_size_t physical_dispatches_per_logical_operation = 0;
    IREE_RETURN_IF_ERROR(
        iree_benchmark_loom_hal_physical_dispatches_per_logical_operation(
            benchmark_result, &physical_dispatches_per_logical_operation));
    iree_host_size_t measured_physical_dispatch_count = 0;
    IREE_RETURN_IF_ERROR(
        iree_benchmark_loom_hal_measured_physical_dispatch_count(
            benchmark_result, &measured_physical_dispatch_count));
    double mean_physical_dispatch_duration_ns = 0.0;
    IREE_RETURN_IF_ERROR(
        iree_benchmark_loom_hal_mean_physical_dispatch_duration_ns(
            benchmark_result, &mean_physical_dispatch_duration_ns));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
        stream, &first_field, "logical_operations_per_batch",
        timing->batch_size));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
        stream, &first_field, "physical_dispatches_per_batch",
        physical_dispatches_per_batch));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
        stream, &first_field, "physical_dispatches_per_logical_operation",
        physical_dispatches_per_logical_operation));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
        stream, &first_field, "warmup_batch_count",
        timing->warmup_batch_count));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_i64_field(
        stream, &first_field, "warmup_duration_ns",
        timing->warmup_duration_ns));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
        stream, &first_field, "measured_batch_count",
        timing->measured_batch_count));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
        stream, &first_field, "measured_logical_operation_count",
        timing->measured_operation_count));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
        stream, &first_field, "measured_physical_dispatch_count",
        measured_physical_dispatch_count));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_i64_field(
        stream, &first_field, "measured_duration_ns",
        timing->measured_duration_ns));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
        stream, &first_field, "mean_physical_dispatch_duration_ns"));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, "%.3f", mean_physical_dispatch_duration_ns));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
        stream, &first_field, "stop_reason",
        loom_run_benchmark_stop_reason_name(timing->stop_reason)));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
        stream, &first_field, "timing_interpretation"));
    IREE_RETURN_IF_ERROR(
        iree_benchmark_loom_write_hal_timing_interpretation_json(
            policy, benchmark_result, stream));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
        stream, &first_field, "batch_timing_ns"));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_benchmark_timing_stats_json(
        &timing->batch_timing, stream));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
        stream, &first_field, "operation_timing_ns"));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_benchmark_timing_stats_json(
        &timing->operation_timing, stream));
  }
  return loom_output_stream_write_cstring(stream, "}");
}

iree_status_t iree_benchmark_loom_write_benchmark_evidence_fields_json(
    const iree_benchmark_loom_benchmark_policy_t* policy,
    const iree_benchmark_loom_benchmark_result_t* benchmark_result,
    iree_host_size_t correctness_sample_count,
    iree_host_size_t correctness_failed_sample_count,
    loom_output_stream_t* stream, bool* first_field) {
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
      stream, first_field, "policy"));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_benchmark_policy_json(policy, stream));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
      stream, first_field, "correctness"));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_benchmark_correctness_json(
      correctness_sample_count, correctness_failed_sample_count, stream));
  if (iree_benchmark_loom_benchmark_result_has_measurement(benchmark_result)) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
        stream, first_field, "measurement"));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_benchmark_measurement_json(
        policy, benchmark_result, stream));
  }
  if (benchmark_result->has_hal_benchmark) {
    const iree_benchmark_loom_profiled_dispatch_timing_t profiled_dispatch =
        iree_benchmark_loom_profiled_dispatch_timing(
            &benchmark_result->hal_benchmark.profile);
    if (profiled_dispatch.available) {
      IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
          stream, first_field, "profiled_dispatch_timing"));
      IREE_RETURN_IF_ERROR(
          iree_benchmark_loom_write_profiled_dispatch_timing_json(
              &profiled_dispatch, stream));
    }
  }
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_data_cache_summary_field_json(
      &benchmark_result->data_cache, stream, first_field));
  if (benchmark_result->hal_benchmark.profile.requested ||
      benchmark_result->hal_benchmark.profile.executed ||
      benchmark_result->hal_benchmark.profile.has_error) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
        stream, first_field, "profile"));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_hal_profile_summary_json(
        &benchmark_result->hal_benchmark.profile, stream));
  }
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_compile_report_field_json(
      benchmark_result->compile_report_capture, stream, first_field));
  if (benchmark_result->has_failure) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
        stream, first_field, "failure"));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_benchmark_failure_json(
        benchmark_result, stream));
  }
  return iree_ok_status();
}

iree_status_t iree_benchmark_loom_write_summary_counts_json(
    const iree_benchmark_loom_summary_counts_t* counts,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream,
      "\"planned\":{\"case_count\":%" PRIhsz ",\"benchmark_count\":%" PRIhsz
      ",\"selected_benchmark_count\":%" PRIhsz
      ",\"logical_sample_count\":%" PRIhsz ",\"work_item_count\":%" PRIhsz "}",
      counts->planned_case_count, counts->planned_benchmark_count,
      counts->selected_benchmark_count, counts->logical_sample_count,
      counts->work_item_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream,
      ",\"failures\":{\"row_count\":%" PRIhsz
      ",\"failed_benchmark_count\":%" PRIhsz "}",
      counts->failure_count, counts->failed_benchmark_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream,
      ",\"correctness\":{\"sample_count\":%" PRIhsz
      ",\"failed_sample_count\":%" PRIhsz "}",
      counts->correctness_sample_count,
      counts->correctness_failed_sample_count));
  if (counts->artifact_bundle_enabled || counts->fixture_read_count != 0 ||
      counts->file_output_count != 0 || counts->profile_count != 0 ||
      counts->compile_report_count != 0 ||
      counts->artifact_manifest_count != 0 ||
      counts->target_artifact_count != 0 || counts->target_listing_count != 0 ||
      counts->hal_executable_count != 0) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream,
        ",\"artifacts\":{\"bundle_enabled\":%s,\"fixture_read_count\":%" PRIhsz
        ",\"file_output_count\":%" PRIhsz ",\"profile_count\":%" PRIhsz
        ",\"compile_report_count\":%" PRIhsz
        ",\"artifact_manifest_count\":%" PRIhsz
        ",\"target_artifact_count\":%" PRIhsz
        ",\"target_listing_count\":%" PRIhsz
        ",\"hal_executable_count\":%" PRIhsz "}",
        counts->artifact_bundle_enabled ? "true" : "false",
        counts->fixture_read_count, counts->file_output_count,
        counts->profile_count, counts->compile_report_count,
        counts->artifact_manifest_count, counts->target_artifact_count,
        counts->target_listing_count, counts->hal_executable_count));
  }
  return loom_output_stream_write_cstring(stream, "}");
}

iree_status_t iree_benchmark_loom_write_benchmark_result_json(
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_benchmark_loom_benchmark_policy_t* policy,
    const iree_benchmark_loom_benchmark_result_t* benchmark_result,
    iree_host_size_t correctness_sample_count,
    iree_host_size_t correctness_failed_sample_count,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "\"benchmark\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(stream, benchmark_plan->name));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\"case\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(stream, case_plan->name));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\"state\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      stream, iree_benchmark_loom_benchmark_result_state(benchmark_result)));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_sample_compilation_field_json(
      benchmark_result->sample_compilation, stream));
  if (benchmark_result->has_sample_ordinal) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_format(stream, ",\"sample_ordinal\":%" PRIhsz,
                                        benchmark_result->sample_ordinal));
  }
  if (!iree_string_view_is_empty(
          benchmark_result->compile_report_artifact_path)) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"compile_report_path\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        stream, benchmark_result->compile_report_artifact_path));
  }
  if (!iree_string_view_is_empty(benchmark_result->artifact_manifest_path)) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
        stream, ",\"artifact_manifest_path\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        stream, benchmark_result->artifact_manifest_path));
  }
  if (!iree_string_view_is_empty(benchmark_result->target_artifact_path)) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"target_artifact_path\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        stream, benchmark_result->target_artifact_path));
  }
  if (!iree_string_view_is_empty(benchmark_result->target_listing_path)) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"target_listing_path\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        stream, benchmark_result->target_listing_path));
  }
  if (!iree_string_view_is_empty(benchmark_result->hal_executable_path)) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"hal_executable_path\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        stream, benchmark_result->hal_executable_path));
  }
  bool first_field = false;
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_benchmark_evidence_fields_json(
      policy, benchmark_result, correctness_sample_count,
      correctness_failed_sample_count, stream, &first_field));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}"));
  return iree_ok_status();
}

iree_status_t iree_benchmark_loom_append_compile_row(
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_candidate_identity_t* candidate,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_benchmark_loom_hal_actual_provider_t* provider,
    iree_string_builder_t* compile_output) {
  iree_string_view_t entry_symbol = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_module_symbol_name_from_ref(
      provider->execution.test_module,
      provider->execution.actual_invocation->callee_ref, &entry_symbol));

  loom_output_stream_t stream;
  loom_output_stream_for_builder(compile_output, &stream);
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, "{\"row\":\"compile\""));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_run_id_field_json(run, &stream));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_candidate_identity_json(candidate, &stream));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_sample_compilation_field_json(
      provider->sample_compilation, &stream));
  if (provider->execution.has_sample_constant_ordinal) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_sample_fields_json(
        provider->execution.test_module, case_plan,
        provider->execution.sample_constant_ordinal, &stream));
  }
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"benchmark\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(&stream, benchmark_plan->name));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ",\"case\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(&stream, case_plan->name));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"entry\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(&stream, entry_symbol));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"state\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(
      &stream, provider->execution.compile_rejected ? "failed" : "ok"));
  if (provider->execution.compile_rejected) {
    IREE_RETURN_IF_ERROR(
        iree_benchmark_loom_write_compile_rejection_fields_json(provider,
                                                                &stream));
  }
  bool first_diagnostic_field = false;
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_diagnostic_capture_fields_json(
      &provider->diagnostics, &stream, &first_diagnostic_field));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"sample_constant_argument_count\":%" PRIhsz,
      provider->execution.sample_constant_argument_count));
  if (!iree_string_view_is_empty(provider->compile_report_artifact_path)) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"compile_report_path\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        &stream, provider->compile_report_artifact_path));
  }
  if (!iree_string_view_is_empty(provider->artifact_manifest_path)) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
        &stream, ",\"artifact_manifest_path\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        &stream, provider->artifact_manifest_path));
  }
  if (!iree_string_view_is_empty(provider->target_artifact_path)) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
        &stream, ",\"target_artifact_path\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        &stream, provider->target_artifact_path));
  }
  if (!iree_string_view_is_empty(provider->target_listing_path)) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"target_listing_path\":"));
    IREE_RETURN_IF_ERROR(
        loom_json_write_escaped_string(&stream, provider->target_listing_path));
  }
  if (!iree_string_view_is_empty(provider->hal_executable_path)) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"hal_executable_path\":"));
    IREE_RETURN_IF_ERROR(
        loom_json_write_escaped_string(&stream, provider->hal_executable_path));
  }
  if (provider->execution.compile_report_available) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_compile_report_json(
        &provider->compile_report_capture, &stream));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}\n"));
  return iree_ok_status();
}

iree_status_t iree_benchmark_loom_append_benchmark_result(
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_candidate_identity_t* candidate,
    iree_host_size_t work_item_index, const loom_module_t* module,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_benchmark_loom_benchmark_policy_t* policy,
    const iree_benchmark_loom_benchmark_result_t* benchmark_result,
    iree_host_size_t correctness_sample_count,
    iree_host_size_t correctness_failed_sample_count,
    iree_string_builder_t* benchmark_output) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(benchmark_output, &stream);
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, "{\"row\":\"benchmark\""));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_run_id_field_json(run, &stream));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_candidate_identity_json(candidate, &stream));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_work_item_index_field_json(
      work_item_index, &stream));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_sample_compilation_field_json(
      benchmark_result->sample_compilation, &stream));
  if (benchmark_result->has_sample_ordinal) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_sample_fields_json(
        module, case_plan, benchmark_result->sample_ordinal, &stream));
  }
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"benchmark_result\":"));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_benchmark_result_json(
      benchmark_plan, case_plan, policy, benchmark_result,
      correctness_sample_count, correctness_failed_sample_count, &stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}\n"));
  return iree_ok_status();
}

iree_status_t iree_benchmark_loom_append_benchmark_repetition_row(
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_dispatch_comparison_candidate_t* candidate,
    const iree_benchmark_loom_candidate_identity_t* baseline,
    iree_string_view_t comparison_group, iree_string_view_t method,
    iree_host_size_t order_index, iree_host_size_t repetition_index,
    char schedule_token, bool profile_suppressed,
    const iree_benchmark_loom_benchmark_result_t* benchmark_result,
    iree_string_builder_t* benchmark_output) {
  const iree_benchmark_loom_selected_benchmark_t* selection =
      candidate->selection;
  loom_output_stream_t stream;
  loom_output_stream_for_builder(benchmark_output, &stream);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
      &stream, "{\"row\":\"benchmark.repetition\""));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_run_id_field_json(run, &stream));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_candidate_identity_json(
      &selection->identity, &stream));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_sample_compilation_field_json(
      candidate->sample_compilation, &stream));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_sample_fields_json(
      candidate->module, selection->case_plan, benchmark_result->sample_ordinal,
      &stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"comparison_group\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(&stream, comparison_group));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"baseline_candidate_id\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(&stream, baseline->candidate_id));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"method\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(&stream, method));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"order_index\":%" PRIhsz, order_index));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"repetition_index\":%" PRIhsz, repetition_index));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"schedule_token\":\"%c\"", schedule_token));
  if (profile_suppressed) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
        &stream, ",\"profile_suppressed_for_interleave\":true"));
  }
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"benchmark_result\":"));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_benchmark_result_json(
      selection->benchmark_plan, selection->case_plan, &selection->policy,
      benchmark_result, candidate->correctness_sample_count,
      candidate->correctness_failed_sample_count, &stream));
  return loom_output_stream_write_cstring(&stream, "}\n");
}

iree_status_t iree_benchmark_loom_append_comparison_row(
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_dispatch_comparison_candidate_t* baseline,
    const iree_benchmark_loom_dispatch_comparison_candidate_t* candidate,
    iree_string_view_t comparison_group, iree_string_view_t method,
    iree_string_builder_t* benchmark_output) {
  if (baseline->sample_count == 0 || candidate->sample_count == 0) {
    return iree_ok_status();
  }

  loom_run_benchmark_timing_stats_t baseline_p50 = {0};
  loom_run_benchmark_timing_stats_t candidate_p50 = {0};
  loom_run_benchmark_timing_stats_t baseline_p90 = {0};
  loom_run_benchmark_timing_stats_t candidate_p90 = {0};
  IREE_RETURN_IF_ERROR(loom_run_benchmark_compute_timing_stats(
      baseline->p50_samples, baseline->sample_count, &baseline_p50));
  IREE_RETURN_IF_ERROR(loom_run_benchmark_compute_timing_stats(
      candidate->p50_samples, candidate->sample_count, &candidate_p50));
  IREE_RETURN_IF_ERROR(loom_run_benchmark_compute_timing_stats(
      baseline->p90_samples, baseline->sample_count, &baseline_p90));
  IREE_RETURN_IF_ERROR(loom_run_benchmark_compute_timing_stats(
      candidate->p90_samples, candidate->sample_count, &candidate_p90));

  const double baseline_p50_ns = (double)baseline_p50.p50_ns;
  const double candidate_p50_ns = (double)candidate_p50.p50_ns;
  const double ratio_p50 =
      baseline_p50_ns == 0.0 ? 0.0 : candidate_p50_ns / baseline_p50_ns;
  const double speedup_p50 =
      candidate_p50_ns == 0.0 ? 0.0 : baseline_p50_ns / candidate_p50_ns;
  const double baseline_p90_ns = (double)baseline_p90.p50_ns;
  const double candidate_p90_ns = (double)candidate_p90.p50_ns;
  const double ratio_p90 =
      baseline_p90_ns == 0.0 ? 0.0 : candidate_p90_ns / baseline_p90_ns;
  const double speedup_p90 =
      candidate_p90_ns == 0.0 ? 0.0 : baseline_p90_ns / candidate_p90_ns;

  loom_output_stream_t stream;
  loom_output_stream_for_builder(benchmark_output, &stream);
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, "{\"row\":\"comparison\""));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_run_id_field_json(run, &stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"comparison_group\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(&stream, comparison_group));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"method\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(&stream, method));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"baseline_candidate_id\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream, baseline->selection->identity.candidate_id));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"candidate_id\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream, candidate->selection->identity.candidate_id));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"baseline_repetition_count\":%" PRIhsz,
      baseline->sample_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"candidate_repetition_count\":%" PRIhsz,
      candidate->sample_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"baseline_p50_ns\":%" PRIi64, baseline_p50.p50_ns));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"candidate_p50_ns\":%" PRIi64, candidate_p50.p50_ns));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"baseline_p90_ns\":%" PRIi64, baseline_p90.p50_ns));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"candidate_p90_ns\":%" PRIi64, candidate_p90.p50_ns));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"baseline_p50_spread_ppm\":%" PRIu64,
      baseline_p50.p90_to_p50_delta_ppm));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"candidate_p50_spread_ppm\":%" PRIu64,
      candidate_p50.p90_to_p50_delta_ppm));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"baseline_p90_spread_ppm\":%" PRIu64,
      baseline_p90.p90_to_p50_delta_ppm));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"candidate_p90_spread_ppm\":%" PRIu64,
      candidate_p90.p90_to_p50_delta_ppm));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"ratio_p50\":%.6f", ratio_p50));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"speedup_p50\":%.6f", speedup_p50));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"ratio_p90\":%.6f", ratio_p90));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"speedup_p90\":%.6f", speedup_p90));
  return loom_output_stream_write_cstring(&stream, "}\n");
}

iree_status_t iree_benchmark_loom_append_failure_row(
    const iree_benchmark_loom_run_identity_t* run, iree_string_view_t stage,
    iree_string_view_t kind, iree_string_view_t message,
    const iree_benchmark_loom_diagnostic_capture_t* diagnostics,
    const loom_testbench_module_plan_t* testbench_plan,
    const loom_testbench_issue_t* planning_issues,
    iree_host_size_t planning_issue_count,
    iree_string_builder_t* failure_output) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(failure_output, &stream);
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, "{\"row\":\"failure\""));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_run_id_field_json(run, &stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"stage\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(&stream, stage));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ",\"kind\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(&stream, kind));
  if (!iree_string_view_is_empty(message)) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"message\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(&stream, message));
  }
  bool first_diagnostic_field = false;
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_diagnostic_capture_fields_json(
      diagnostics, &stream, &first_diagnostic_field));
  bool first_planning_issue_field = false;
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_planning_issue_fields_json(
      testbench_plan, planning_issues, planning_issue_count, &stream,
      &first_planning_issue_field));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}\n"));
  return iree_ok_status();
}

iree_status_t iree_benchmark_loom_append_summary_row(
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_artifact_bundle_t* bundle,
    iree_host_size_t planned_case_count,
    iree_host_size_t planned_benchmark_count,
    iree_host_size_t selected_benchmark_count,
    iree_host_size_t logical_sample_count, iree_host_size_t work_item_count,
    iree_host_size_t failure_count, iree_host_size_t failed_benchmark_count,
    iree_host_size_t correctness_sample_count,
    iree_host_size_t correctness_failed_sample_count, bool dry_run,
    iree_benchmark_loom_sample_compilation_mode_t sample_compilation_mode,
    iree_string_builder_t* output) {
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(output, "{\"row\":\"summary\""));
  loom_output_stream_t stream;
  loom_output_stream_for_builder(output, &stream);
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_run_id_field_json(run, &stream));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, ",\"dry_run\":%s", dry_run ? "true" : "false"));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"sample_compilation\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream, iree_benchmark_loom_sample_compilation_mode_name(
                   sample_compilation_mode)));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(output, ",\"summary\":"));
  const iree_benchmark_loom_summary_counts_t counts = {
      .planned_case_count = planned_case_count,
      .planned_benchmark_count = planned_benchmark_count,
      .selected_benchmark_count = selected_benchmark_count,
      .logical_sample_count = logical_sample_count,
      .work_item_count = work_item_count,
      .failure_count = failure_count,
      .failed_benchmark_count = failed_benchmark_count,
      .correctness_sample_count = correctness_sample_count,
      .correctness_failed_sample_count = correctness_failed_sample_count,
      .artifact_bundle_enabled = bundle != NULL && bundle->enabled,
      .fixture_read_count = iree_benchmark_loom_artifact_bundle_file_count(
          bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_FIXTURE_READ),
      .file_output_count = iree_benchmark_loom_artifact_bundle_file_count(
          bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_OUTPUT),
      .profile_count = iree_benchmark_loom_artifact_bundle_file_count(
          bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_PROFILE),
      .compile_report_count = iree_benchmark_loom_artifact_bundle_file_count(
          bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_COMPILE_REPORT),
      .artifact_manifest_count = iree_benchmark_loom_artifact_bundle_file_count(
          bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_ARTIFACT_MANIFEST),
      .target_artifact_count = iree_benchmark_loom_artifact_bundle_file_count(
          bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_TARGET_ARTIFACT),
      .target_listing_count = iree_benchmark_loom_artifact_bundle_file_count(
          bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_TARGET_LISTING),
      .hal_executable_count = iree_benchmark_loom_artifact_bundle_file_count(
          bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_HAL_EXECUTABLE),
  };
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_summary_counts_json(&counts, &stream));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "}\n"));
  return iree_ok_status();
}
