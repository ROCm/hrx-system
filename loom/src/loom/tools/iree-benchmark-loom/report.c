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
#include "loom/tooling/execution/benchmark.h"
#include "loom/tooling/io/file.h"
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

  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_format(stream, "\"code\":%d", (int)code));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"status\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      stream, iree_make_cstring_view(iree_status_code_string(code))));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"message\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      stream, iree_make_string_view(message, required_length)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}"));
  return iree_ok_status();
}

static iree_status_t iree_benchmark_loom_write_optional_i64_query_json(
    iree_hal_device_t* device, iree_string_view_t category,
    iree_string_view_t key, const char* field_name,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(stream, field_name));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ":"));
  int64_t value = 0;
  iree_status_t status =
      iree_hal_device_query_i64(device, category, key, &value);
  if (iree_status_is_ok(status)) {
    return loom_output_stream_write_format(stream, "%" PRIi64, value);
  }
  iree_status_t write_status =
      iree_benchmark_loom_write_status_object_json(status, stream);
  iree_status_free(status);
  return write_status;
}

static iree_status_t iree_benchmark_loom_write_hex_bytes_json(
    const uint8_t* bytes, iree_host_size_t byte_count,
    loom_output_stream_t* stream) {
  static const char kHexDigits[] = "0123456789abcdef";
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\""));
  for (iree_host_size_t i = 0; i < byte_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_char(stream, kHexDigits[bytes[i] >> 4]));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_char(stream, kHexDigits[bytes[i] & 0x0F]));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\""));
  return iree_ok_status();
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
    iree_string_builder_t* output) {
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
        loom_output_stream_write_cstring(stream, ",\"status\":\"created\""));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"device_id\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        stream, iree_hal_device_id(context->execution.runtime.device)));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"queries\":{"));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, "\"attempted\":true"));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_optional_i64_query_json(
        context->execution.runtime.device, IREE_SV("hal.device"),
        IREE_SV("concurrency"), "hal_device_concurrency", stream));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_optional_i64_query_json(
        context->execution.runtime.device, IREE_SV("hal.dispatch"),
        IREE_SV("concurrency"), "hal_dispatch_concurrency", stream));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}"));

    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"capabilities\":"));
    iree_hal_device_capabilities_t capabilities = {0};
    iree_status_t capabilities_status = iree_hal_device_query_capabilities(
        context->execution.runtime.device, &capabilities);
    if (iree_status_is_ok(capabilities_status)) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
          stream,
          "{\"flags\":%" PRIu64 ",\"numa_node\":%" PRIu8
          ",\"has_physical_device_uuid\":%s,\"device_group_index\":%" PRIu32
          ",\"has_device_group\":%s",
          capabilities.flags, capabilities.numa_node,
          capabilities.has_physical_device_uuid ? "true" : "false",
          capabilities.device_group_index,
          capabilities.has_device_group ? "true" : "false"));
      if (capabilities.has_physical_device_uuid) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
            stream, ",\"physical_device_uuid\":"));
        IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_hex_bytes_json(
            capabilities.physical_device_uuid,
            IREE_ARRAYSIZE(capabilities.physical_device_uuid), stream));
      }
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}"));
    } else {
      iree_status_t write_status = iree_benchmark_loom_write_status_object_json(
          capabilities_status, stream);
      iree_status_free(capabilities_status);
      IREE_RETURN_IF_ERROR(write_status);
    }
  } else {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"status\":\"planned\""));
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

static iree_status_t iree_benchmark_loom_write_timing_stats_json(
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

static iree_status_t iree_benchmark_loom_write_benchmark_timing_stats_json(
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
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"error\":{"));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, "\"code\":%d", (int)profile->error_code));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"status\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        stream,
        iree_make_cstring_view(iree_status_code_string(profile->error_code))));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"message\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        stream, iree_make_string_view(profile->error_message,
                                      profile->error_message_length)));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}"));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}"));
  return iree_ok_status();
}

static iree_string_view_t iree_benchmark_loom_benchmark_result_status(
    const iree_benchmark_loom_benchmark_result_t* benchmark_result) {
  if (!iree_string_view_is_empty(benchmark_result->status)) {
    return benchmark_result->status;
  }
  if (benchmark_result->executed) {
    return benchmark_result->passed ? IREE_SV("ok") : IREE_SV("failed");
  }
  return IREE_SV("skipped");
}

static iree_status_t iree_benchmark_loom_write_benchmark_failure_json(
    const iree_benchmark_loom_benchmark_result_t* benchmark_result,
    loom_output_stream_t* stream) {
  if (!benchmark_result->has_failure) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"failure\":{"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\"stage\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(stream, benchmark_result->failure_stage));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\"kind\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(stream, benchmark_result->failure_kind));
  if (!iree_string_view_is_empty(benchmark_result->failure_message)) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"message\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        stream, benchmark_result->failure_message));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"diagnostic_error_count\":%" PRIhsz,
      benchmark_result->diagnostic_error_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"diagnostic_warning_count\":%" PRIhsz,
      benchmark_result->diagnostic_warning_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"diagnostic_remark_count\":%" PRIhsz,
      benchmark_result->diagnostic_remark_count));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"diagnostics\":["));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write(stream, benchmark_result->diagnostic_json));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "]}"));
  return iree_ok_status();
}

static iree_string_view_t iree_benchmark_loom_compile_artifact_kind_name(
    loom_target_compile_artifact_kind_t kind) {
  switch (kind) {
    case LOOM_TARGET_COMPILE_ARTIFACT_KIND_NONE:
      return IREE_SV("none");
    case LOOM_TARGET_COMPILE_ARTIFACT_KIND_VM_ARCHIVE:
      return IREE_SV("vm-archive");
    case LOOM_TARGET_COMPILE_ARTIFACT_KIND_HAL_EXECUTABLE:
      return IREE_SV("hal-executable");
    case LOOM_TARGET_COMPILE_ARTIFACT_KIND_HAL_KERNEL_LIBRARY:
      return IREE_SV("hal-kernel-library");
    default:
      return IREE_SV("unknown");
  }
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

iree_status_t iree_benchmark_loom_write_json_size_field(
    loom_output_stream_t* stream, bool* first_field, const char* name,
    iree_host_size_t value) {
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
      stream, first_field, name));
  return loom_output_stream_write_format(stream, "%" PRIhsz, value);
}

enum {
  IREE_BENCHMARK_LOOM_SHORT_MEASURED_DURATION_NS = 1000 * 1000,
  IREE_BENCHMARK_LOOM_SUB_MICROSECOND_DURATION_NS = 1000,
  IREE_BENCHMARK_LOOM_SMALL_PHYSICAL_DISPATCH_SAMPLE_COUNT = 100,
};

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
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      stream, &first_field, "score_unit", IREE_SV("logical_operation")));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
      stream, &first_field, "warnings"));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_hal_timing_warnings_json(
      policy, benchmark_result, stream));
  return loom_output_stream_write_cstring(stream, "}");
}

static void iree_benchmark_loom_compile_report_move_cause_totals(
    const loom_target_compile_report_t* report, uint64_t* out_kind_count,
    uint64_t* out_packet_count, uint64_t* out_unit_count) {
  *out_kind_count = 0;
  *out_packet_count = 0;
  *out_unit_count = 0;
  for (iree_host_size_t i = 1; i < LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_COUNT;
       ++i) {
    const loom_target_compile_report_move_cause_counts_t* counts =
        &report->move_causes[i];
    if (counts->packet_count == 0 && counts->unit_count == 0) {
      continue;
    }
    ++*out_kind_count;
    *out_packet_count += counts->packet_count;
    *out_unit_count += counts->unit_count;
  }
}

static iree_status_t iree_benchmark_loom_write_static_summary_json(
    const loom_run_compile_report_capture_t* compile_report_capture,
    loom_output_stream_t* stream) {
  if (compile_report_capture == NULL ||
      !loom_run_compile_report_capture_is_enabled(compile_report_capture)) {
    return iree_ok_status();
  }
  const loom_target_compile_report_t* report = &compile_report_capture->report;
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"static_summary\":{"));
  bool first_field = true;
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      stream, &first_field, "artifact_kind",
      iree_benchmark_loom_compile_artifact_kind_name(report->artifact_kind)));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      stream, &first_field, "status",
      iree_make_cstring_view(iree_status_code_string(report->status_code))));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u32_field(
      stream, &first_field, "detail_flags", report->detail_flags));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_optional_string_field(
      stream, &first_field, "backend", report->backend_name));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_optional_string_field(
      stream, &first_field, "target_family", report->target_family_name));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_optional_string_field(
      stream, &first_field, "target_key", report->target_key));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_optional_string_field(
      stream, &first_field, "function", report->function_name));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_optional_string_field(
      stream, &first_field, "target_bundle", report->target_bundle_name));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_optional_string_field(
      stream, &first_field, "target_export", report->target_export_name));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_optional_string_field(
      stream, &first_field, "target_export_symbol",
      report->target_export_symbol));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_optional_string_field(
      stream, &first_field, "target_config", report->target_config_name));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_optional_string_field(
      stream, &first_field, "lowered", report->lowered_symbol));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_optional_string_field(
      stream, &first_field, "executable_format", report->executable_format));
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_ARTIFACT_SIZE)) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
        stream, &first_field, "artifact_size", report->artifact_size));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_ENTRIES)) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
        stream, &first_field, "entry_count", report->entry_rows.count));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_EMISSION)) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
        stream, &first_field, "instruction_count",
        report->emitted_instruction_count));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
        stream, &first_field, "code_byte_count",
        report->emitted_code_byte_count));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
        stream, &first_field, "code_storage_byte_count",
        report->emitted_code_storage_byte_count));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_MEMORY)) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
        stream, &first_field, "private_memory_bytes",
        report->private_memory_bytes));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
        stream, &first_field, "local_memory_bytes",
        report->local_memory_bytes));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION)) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
        stream, &first_field, "allocation_assignment_count",
        report->allocation_assignment_count));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
        stream, &first_field, "allocation_spill_count",
        report->allocation_spill_count));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
        stream, &first_field, "allocation_spill_plan_count",
        report->allocation_spill_plan_count));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
        stream, &first_field, "allocation_coalesced_copy_count",
        report->allocation_coalesced_copy_count));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
        stream, &first_field, "allocation_materialized_copy_count",
        report->allocation_materialized_copy_count));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE)) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
        stream, &first_field, "schedule_node_count",
        report->schedule_node_count));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
        stream, &first_field, "scheduled_node_count",
        report->scheduled_node_count));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
        stream, &first_field, "register_pressure_summary_count",
        report->register_pressure_summary_count));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
        stream, &first_field, "register_pressure_peak_live_units",
        report->register_pressure_peak_live_units));
  }
  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_STATIC_INSTRUCTION_MIX)) {
    const loom_target_compile_report_static_instruction_mix_t* mix =
        &report->static_instruction_mix;
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
        stream, &first_field, "descriptor_count", mix->descriptor_count));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
        stream, &first_field, "unknown_descriptor_count", mix->unknown_count));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
        stream, &first_field, "scalar_alu_count", mix->scalar_alu_count));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
        stream, &first_field, "vector_alu_count", mix->vector_alu_count));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
        stream, &first_field, "matrix_count", mix->matrix_count));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
        stream, &first_field, "mfma_count", mix->mfma_count));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
        stream, &first_field, "wmma_count", mix->wmma_count));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
        stream, &first_field, "dot_count", mix->dot_count));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
        stream, &first_field, "global_memory_count", mix->global_memory_count));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
        stream, &first_field, "local_memory_count", mix->local_memory_count));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
        stream, &first_field, "scalar_memory_count", mix->scalar_memory_count));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
        stream, &first_field, "generic_memory_count",
        mix->generic_memory_count));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
        stream, &first_field, "atomic_count", mix->atomic_count));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
        stream, &first_field, "branch_count", mix->branch_count));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
        stream, &first_field, "barrier_count", mix->barrier_count));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
        stream, &first_field, "control_count", mix->control_count));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
        stream, &first_field, "conversion_count", mix->conversion_count));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
        stream, &first_field, "cache_count", mix->cache_count));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
        stream, &first_field, "register_move_count", mix->register_move_count));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS)) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
        stream, &first_field, "source_low_selected_op_count",
        report->source_low_selected_op_count));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
        stream, &first_field, "source_low_emitted_op_count",
        report->source_low_emitted_op_count));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
        stream, &first_field, "source_low_row_count",
        report->source_low_rows.count));
  }
  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_LEGALIZATION_ROWS)) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
        stream, &first_field, "target_legalization_legal_op_count",
        report->target_legalization_legal_op_count));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
        stream, &first_field, "target_legalization_rewritten_op_count",
        report->target_legalization_rewritten_op_count));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
        stream, &first_field, "target_legalization_target_rewritten_op_count",
        report->target_legalization_target_rewritten_op_count));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
        stream, &first_field,
        "target_legalization_reference_rewritten_op_count",
        report->target_legalization_reference_rewritten_op_count));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
        stream, &first_field, "target_legalization_deferred_op_count",
        report->target_legalization_deferred_op_count));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
        stream, &first_field, "target_legalization_invalid_ir_op_count",
        report->target_legalization_invalid_ir_op_count));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
        stream, &first_field, "target_legalization_unsupported_op_count",
        report->target_legalization_unsupported_op_count));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
        stream, &first_field, "target_legalization_unhandled_op_count",
        report->target_legalization_unhandled_op_count));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
        stream, &first_field, "target_legalization_row_count",
        report->target_legalization_rows.count));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ROWS)) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
        stream, &first_field, "pressure_row_count",
        report->pressure_rows.count));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_SPILL_ROWS)) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
        stream, &first_field, "spill_row_count", report->spill_rows.count));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_MOVE_CAUSES)) {
    uint64_t kind_count = 0;
    uint64_t packet_count = 0;
    uint64_t unit_count = 0;
    iree_benchmark_loom_compile_report_move_cause_totals(
        report, &kind_count, &packet_count, &unit_count);
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
        stream, &first_field, "move_cause_kind_count", kind_count));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
        stream, &first_field, "move_cause_packet_count", packet_count));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
        stream, &first_field, "move_cause_unit_count", unit_count));
  }
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
      loom_output_stream_write_cstring(&stream, ",\"status\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(
      &stream, provider->execution.compile_rejected ? "failed" : "ok"));
  if (provider->execution.compile_rejected) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"stage\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        &stream, provider->execution.compile_failure_stage));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"kind\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        &stream, provider->execution.compile_failure_kind));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"diagnostic_error_count\":%" PRIhsz,
      provider->diagnostics.error_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"diagnostic_warning_count\":%" PRIhsz,
      provider->diagnostics.warning_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"diagnostic_remark_count\":%" PRIhsz,
      provider->diagnostics.remark_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"sample_constant_argument_count\":%" PRIhsz,
      provider->execution.sample_constant_argument_count));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"diagnostics\":"));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_diagnostic_array_json(
      &provider->diagnostics, &stream));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_static_summary_json(
      &provider->compile_report_capture, &stream));
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

static iree_status_t iree_benchmark_loom_write_data_cache_summary_json(
    const iree_benchmark_loom_data_cache_summary_t* summary,
    loom_output_stream_t* stream) {
  if (!summary->populated) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"data_cache\":{"));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "\"validity\":\"check_ops\""));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
      stream, ",\"cache_policy\":\"binding_ring\""));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"binding_count\":%" PRIhsz, summary->binding_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"binding_ring_count\":%" PRIhsz, summary->binding_ring_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"command_buffer_ring_count\":%" PRIhsz,
      summary->command_buffer_ring_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"dispatches_per_batch\":%" PRIhsz,
      summary->dispatches_per_batch));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"requested_min_ring_bytes\":%" PRIu64,
      summary->requested_min_ring_bytes));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"binding_set_bytes\":%" PRIu64, summary->binding_set_bytes));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"binding_ring_bytes\":%" PRIu64, summary->binding_ring_bytes));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"hot_reuse\":"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
      stream, summary->binding_ring_count <= 1 ? "true" : "false"));
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
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"status\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      stream, iree_benchmark_loom_benchmark_result_status(benchmark_result)));
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
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"measure\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(stream, policy->measure));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"warmup_iterations\":%" PRIhsz, policy->warmup_iterations));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"iterations\":%" PRIhsz, policy->iterations));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"samples_per_iteration\":%" PRIhsz,
      benchmark_result->samples_per_iteration));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"correctness_sample_count\":%" PRIhsz,
      correctness_sample_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"correctness_failed_sample_count\":%" PRIhsz,
      correctness_failed_sample_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"benchmark_failed_sample_count\":%" PRIhsz,
      benchmark_result->failed_sample_count));
  if (benchmark_result->executed && !benchmark_result->has_hal_benchmark) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"timing_ns\":"));
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
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, ",\"logical_operations_per_batch\":%" PRIhsz,
        timing->batch_size));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, ",\"physical_dispatches_per_batch\":%" PRIhsz,
        physical_dispatches_per_batch));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, ",\"physical_dispatches_per_logical_operation\":%" PRIhsz,
        physical_dispatches_per_logical_operation));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, ",\"warmup_batch_count\":%" PRIhsz,
        timing->warmup_batch_count));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, ",\"warmup_duration_ns\":%" PRIi64,
        timing->warmup_duration_ns));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, ",\"measured_batch_count\":%" PRIhsz,
        timing->measured_batch_count));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, ",\"measured_logical_operation_count\":%" PRIhsz,
        timing->measured_operation_count));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, ",\"measured_physical_dispatch_count\":%" PRIhsz,
        measured_physical_dispatch_count));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, ",\"measured_duration_ns\":%" PRIi64,
        timing->measured_duration_ns));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, ",\"mean_physical_dispatch_duration_ns\":%.3f",
        mean_physical_dispatch_duration_ns));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"stop_reason\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        stream, loom_run_benchmark_stop_reason_name(timing->stop_reason)));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
        stream, ",\"timing_interpretation\":"));
    IREE_RETURN_IF_ERROR(
        iree_benchmark_loom_write_hal_timing_interpretation_json(
            policy, benchmark_result, stream));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"batch_timing_ns\":"));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_benchmark_timing_stats_json(
        &timing->batch_timing, stream));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"operation_timing_ns\":"));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_benchmark_timing_stats_json(
        &timing->operation_timing, stream));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_data_cache_summary_json(
        &benchmark_result->data_cache, stream));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"profile\":"));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_hal_profile_summary_json(
        &benchmark_result->hal_benchmark.profile, stream));
  }
  if (benchmark_result->compile_report_capture != NULL &&
      loom_run_compile_report_capture_is_enabled(
          benchmark_result->compile_report_capture)) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_static_summary_json(
        benchmark_result->compile_report_capture, stream));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_compile_report_json(
        benchmark_result->compile_report_capture, stream));
  }
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_benchmark_failure_json(
      benchmark_result, stream));
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
      loom_output_stream_write_cstring(&stream, ",\"status\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(
      &stream, provider->execution.compile_rejected ? "failed" : "ok"));
  if (provider->execution.compile_rejected) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"stage\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        &stream, provider->execution.compile_failure_stage));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"kind\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        &stream, provider->execution.compile_failure_kind));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"diagnostic_error_count\":%" PRIhsz,
      provider->diagnostics.error_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"diagnostic_warning_count\":%" PRIhsz,
      provider->diagnostics.warning_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"diagnostic_remark_count\":%" PRIhsz,
      provider->diagnostics.remark_count));
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
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"diagnostics\":"));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_diagnostic_array_json(
      &provider->diagnostics, &stream));
  if (provider->execution.compile_report_available) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_static_summary_json(
        &provider->compile_report_capture, &stream));
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
  if (diagnostics != NULL) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        &stream, ",\"diagnostic_error_count\":%" PRIhsz,
        diagnostics->error_count));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        &stream, ",\"diagnostic_warning_count\":%" PRIhsz,
        diagnostics->warning_count));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        &stream, ",\"diagnostic_remark_count\":%" PRIhsz,
        diagnostics->remark_count));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"diagnostics\":"));
    IREE_RETURN_IF_ERROR(
        iree_benchmark_loom_write_diagnostic_array_json(diagnostics, &stream));
  }
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
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, ",\"planned_case_count\":%" PRIhsz, planned_case_count));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, ",\"planned_benchmark_count\":%" PRIhsz,
      planned_benchmark_count));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, ",\"benchmark_count\":%" PRIhsz, selected_benchmark_count));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, ",\"logical_sample_count\":%" PRIhsz, logical_sample_count));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, ",\"work_item_count\":%" PRIhsz, work_item_count));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, ",\"failure_count\":%" PRIhsz, failure_count));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, ",\"failed_benchmark_count\":%" PRIhsz, failed_benchmark_count));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, ",\"correctness_sample_count\":%" PRIhsz,
      correctness_sample_count));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, ",\"correctness_failed_sample_count\":%" PRIhsz,
      correctness_failed_sample_count));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(output, ",\"summary\":{"));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output,
      "\"planned\":{\"case_count\":%" PRIhsz ",\"benchmark_count\":%" PRIhsz
      ",\"selected_benchmark_count\":%" PRIhsz
      ",\"logical_sample_count\":%" PRIhsz ",\"work_item_count\":%" PRIhsz "}",
      planned_case_count, planned_benchmark_count, selected_benchmark_count,
      logical_sample_count, work_item_count));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output,
      ",\"failures\":{\"row_count\":%" PRIhsz
      ",\"failed_benchmark_count\":%" PRIhsz "}",
      failure_count, failed_benchmark_count));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output,
      ",\"correctness\":{\"sample_count\":%" PRIhsz
      ",\"failed_sample_count\":%" PRIhsz "}",
      correctness_sample_count, correctness_failed_sample_count));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output,
      ",\"artifacts\":{\"bundle_enabled\":%s,\"fixture_read_count\":%" PRIhsz
      ",\"file_output_count\":%" PRIhsz ",\"profile_count\":%" PRIhsz
      ",\"compile_report_count\":%" PRIhsz
      ",\"artifact_manifest_count\":%" PRIhsz
      ",\"target_artifact_count\":%" PRIhsz ",\"target_listing_count\":%" PRIhsz
      ",\"hal_executable_count\":%" PRIhsz "}}",
      bundle != NULL && bundle->enabled ? "true" : "false",
      iree_benchmark_loom_artifact_bundle_file_count(
          bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_FIXTURE_READ),
      iree_benchmark_loom_artifact_bundle_file_count(
          bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_OUTPUT),
      iree_benchmark_loom_artifact_bundle_file_count(
          bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_PROFILE),
      iree_benchmark_loom_artifact_bundle_file_count(
          bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_COMPILE_REPORT),
      iree_benchmark_loom_artifact_bundle_file_count(
          bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_ARTIFACT_MANIFEST),
      iree_benchmark_loom_artifact_bundle_file_count(
          bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_TARGET_ARTIFACT),
      iree_benchmark_loom_artifact_bundle_file_count(
          bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_TARGET_LISTING),
      iree_benchmark_loom_artifact_bundle_file_count(
          bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_HAL_EXECUTABLE)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "}\n"));
  return iree_ok_status();
}
