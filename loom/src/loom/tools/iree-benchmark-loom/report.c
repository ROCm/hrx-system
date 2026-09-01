// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-benchmark-loom/report.h"

#include <inttypes.h>
#include <math.h>
#include <stdio.h>

#include "iree/base/byte_sequence.h"
#include "iree/base/internal/path.h"
#include "iree/hal/api.h"
#include "iree/tooling/device_util.h"
#include "loom/sanitizer/options.h"
#include "loom/tooling/execution/benchmark.h"
#include "loom/tooling/io/file.h"
#include "loom/tooling/testbench/issue_report.h"
#include "loom/tools/iree-benchmark-loom/device_spec_report.h"
#include "loom/tools/iree-benchmark-loom/diagnostics.h"
#include "loom/tools/iree-benchmark-loom/launch_evidence.h"
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
  if (context->execution.device_provider != NULL) {
    return context->execution.device_provider->driver_name;
  }
  return iree_string_view_empty();
}

iree_status_t iree_benchmark_loom_write_status_field_json(
    iree_status_code_t code, iree_string_view_t message,
    loom_json_object_writer_t* object) {
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(object, IREE_SV("status")));
  return loom_json_write_status_object(object->stream, code, message);
}

iree_status_t iree_benchmark_loom_write_run_id_field_json(
    const iree_benchmark_loom_run_identity_t* run,
    loom_json_object_writer_t* object) {
  return loom_json_object_write_string_field(object, IREE_SV("run_id"),
                                             run->run_id);
}

iree_status_t iree_benchmark_loom_write_candidate_identity_json(
    const iree_benchmark_loom_candidate_identity_t* candidate,
    loom_json_object_writer_t* object) {
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      object, IREE_SV("candidate_id"), candidate->candidate_id));
  return loom_json_object_write_host_size_field(
      object, IREE_SV("candidate_index"), candidate->candidate_index);
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
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_string_field(&object, IREE_SV("checks"), checks));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("reporting_mode"), reporting_mode));
  return loom_json_object_end(&object);
}

static iree_status_t iree_benchmark_loom_write_work_item_index_field_json(
    iree_host_size_t work_item_index, loom_json_object_writer_t* object) {
  if (work_item_index == IREE_BENCHMARK_LOOM_INDEX_INVALID) {
    return iree_ok_status();
  }
  return loom_json_object_write_host_size_field(
      object, IREE_SV("work_item_index"), work_item_index);
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
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  for (iree_host_size_t parameter_index = 0;
       parameter_index < case_plan->parameter_count; ++parameter_index) {
    const loom_testbench_parameter_plan_t* parameter =
        &case_plan->parameters[parameter_index];
    const iree_host_size_t parameter_sample_ordinal =
        loom_testbench_case_sample_parameter_ordinal(case_plan, sample_ordinal,
                                                     parameter_index);
    iree_string_view_t name = parameter->name;
    if (iree_string_view_is_empty(name)) {
      name = iree_benchmark_loom_value_name(module, parameter->value_id);
    }
    char generated_name[32];
    if (iree_string_view_is_empty(name)) {
      const int length = iree_snprintf(generated_name, sizeof(generated_name),
                                       "param_%" PRIhsz, parameter_index);
      if (length <= 0 || (iree_host_size_t)length >= sizeof(generated_name)) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to format parameter name");
      }
      name = iree_make_string_view(generated_name, (iree_host_size_t)length);
    }
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&object, name));
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
  return loom_json_object_end(&object);
}

static iree_status_t iree_benchmark_loom_write_sample_json(
    const loom_module_t* module, const loom_testbench_case_plan_t* case_plan,
    iree_host_size_t sample_ordinal, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("sample_ordinal"), sample_ordinal));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("parameter_count"), case_plan->parameter_count));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("parameters")));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_case_parameter_map_json(
      module, case_plan, sample_ordinal, /*write_ordinals=*/false, stream));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("parameter_ordinals")));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_case_parameter_map_json(
      module, case_plan, sample_ordinal, /*write_ordinals=*/true, stream));
  return loom_json_object_end(&object);
}

iree_status_t iree_benchmark_loom_write_sample_fields_json(
    const loom_module_t* module, const loom_testbench_case_plan_t* case_plan,
    iree_host_size_t sample_ordinal, loom_json_object_writer_t* object) {
  if (case_plan->parameter_count == 0) {
    return iree_ok_status();
  }
  char sample_id[32];
  const int sample_id_length =
      iree_snprintf(sample_id, sizeof(sample_id), "s%" PRIhsz, sample_ordinal);
  if (sample_id_length <= 0 ||
      (iree_host_size_t)sample_id_length >= sizeof(sample_id)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "failed to format sample identifier");
  }
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      object, IREE_SV("sample_id"),
      iree_make_string_view(sample_id, (iree_host_size_t)sample_id_length)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      object, IREE_SV("sample_index"), sample_ordinal));
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(object, IREE_SV("sample")));
  return iree_benchmark_loom_write_sample_json(module, case_plan,
                                               sample_ordinal, object->stream);
}

iree_status_t iree_benchmark_loom_write_case_sample_plan_fields_json(
    const loom_module_t* module, const loom_testbench_case_plan_t* case_plan,
    loom_json_object_writer_t* object) {
  if (case_plan->parameter_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      object, IREE_SV("case_sample_count"), case_plan->sample_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      object, IREE_SV("case_cartesian_sample_count"),
      case_plan->cartesian_sample_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
      object, IREE_SV("case_sample_count_truncated"),
      case_plan->sample_count_truncated));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(object, IREE_SV("case_parameters")));
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(object->stream, &array));
  for (iree_host_size_t parameter_index = 0;
       parameter_index < case_plan->parameter_count; ++parameter_index) {
    const loom_testbench_parameter_plan_t* parameter =
        &case_plan->parameters[parameter_index];
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
    loom_json_object_writer_t parameter_object;
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin(object->stream, &parameter_object));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&parameter_object, IREE_SV("name")));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_parameter_name_json(
        module, parameter, parameter_index, object->stream));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &parameter_object, IREE_SV("kind"),
        iree_make_cstring_view(
            iree_benchmark_loom_parameter_kind_name(parameter->kind))));
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &parameter_object, IREE_SV("sample_count"), parameter->sample_count));
    IREE_RETURN_IF_ERROR(loom_json_object_end(&parameter_object));
  }
  return loom_json_array_end(&array);
}

iree_status_t iree_benchmark_loom_append_sample_row(
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_candidate_identity_t* candidate,
    iree_host_size_t work_item_index, const loom_module_t* module,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    iree_host_size_t benchmark_sample_ordinal,
    iree_host_size_t case_sample_ordinal,
    const loom_testbench_case_sample_result_t* sample_result,
    iree_string_builder_t* sample_output) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(sample_output, &stream);
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("row"), IREE_SV("sample")));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_run_id_field_json(run, &object));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_candidate_identity_json(candidate, &object));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_work_item_index_field_json(
      work_item_index, &object));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_sample_fields_json(
      module, case_plan, case_sample_ordinal, &object));
  if (benchmark_sample_ordinal != case_sample_ordinal) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &object, IREE_SV("benchmark_sample_ordinal"),
        benchmark_sample_ordinal));
  }
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("sample_result")));
  IREE_RETURN_IF_ERROR(
      loom_testbench_case_sample_result_write_json(sample_result, &stream));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&object));
  return loom_output_stream_write_char(&stream, '\n');
}

iree_status_t iree_benchmark_loom_append_run_row(
    const iree_benchmark_loom_run_identity_t* run, bool dry_run,
    const loom_sanitizer_options_t* sanitizer, iree_string_builder_t* output) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(output, &stream);
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("row"), IREE_SV("run")));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_run_id_field_json(run, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("tool"), IREE_SV("iree-benchmark-loom")));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("source"), run->source));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("results_path"), run->results_path));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("file_output_dir"), run->file_output_dir));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field_if_nonempty(
      &object, IREE_SV("profile_artifacts_dir"), run->profile_artifacts_dir));
  if (!iree_string_view_is_empty(run->artifact_bundle_dir)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("artifact_bundle")));
    loom_json_object_writer_t artifact_bundle_object;
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin(&stream, &artifact_bundle_object));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &artifact_bundle_object, IREE_SV("dir"), run->artifact_bundle_dir));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &artifact_bundle_object, IREE_SV("policy"),
        run->artifact_bundle_policy));
    IREE_RETURN_IF_ERROR(loom_json_object_end(&artifact_bundle_object));
  }
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_bool_field(&object, IREE_SV("dry_run"), dry_run));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("sanitizer")));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_sanitizer_options_json(sanitizer, &stream));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&object));
  return loom_output_stream_write_char(&stream, '\n');
}

iree_status_t iree_benchmark_loom_write_hal_context_identity_fields_json(
    const iree_benchmark_loom_hal_context_t* context,
    loom_json_object_writer_t* object) {
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      object, IREE_SV("device_uri"),
      iree_benchmark_loom_selected_device_uri(context)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      object, IREE_SV("driver"),
      context->execution.device_provider->driver_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      object, IREE_SV("provider"),
      context->execution.device_provider->artifact_provider->name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      object, IREE_SV("target_family"),
      context->execution.device_provider->artifact_provider
          ->target_family_name));
  if (context->execution.runtime_initialized &&
      context->execution.runtime.device != NULL) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        object, IREE_SV("state"), IREE_SV("created")));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        object, IREE_SV("device_id"),
        iree_hal_device_id(context->execution.runtime.device)));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(object, IREE_SV("device_spec")));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_device_spec_json(
        iree_hal_device_spec(context->execution.runtime.device),
        object->stream));
  } else {
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        object, IREE_SV("state"), IREE_SV("planned")));
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
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("row"), IREE_SV("device")));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("run_id"), run->run_id));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_hal_context_identity_fields_json(context,
                                                                 &object));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&object));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(&stream, '\n'));
  row_state->appended = true;
  return iree_ok_status();
}

static iree_status_t iree_benchmark_loom_write_kernel_launch_plan_json(
    const loom_module_t* module, const loom_testbench_case_plan_t* case_plan,
    loom_json_object_writer_t* object) {
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      object, IREE_SV("kernel_launch_count"), case_plan->kernel_launch_count));
  if (case_plan->kernel_launch_count != 1) {
    return iree_ok_status();
  }

  iree_string_view_t kernel_entry = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_module_symbol_name_from_ref(
      module, case_plan->first_kernel_launch->callee_ref, &kernel_entry));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      object, IREE_SV("kernel_entry"), kernel_entry));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      object, IREE_SV("kernel_argument_count"),
      case_plan->first_kernel_launch->input_count));
  return iree_ok_status();
}

iree_status_t iree_benchmark_loom_append_plan_row(
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_candidate_identity_t* candidate,
    const loom_module_t* module,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_benchmark_loom_benchmark_policy_t* policy,
    const iree_benchmark_loom_options_t* options, iree_allocator_t allocator,
    iree_string_builder_t* plan_output) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(plan_output, &stream);
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("row"), IREE_SV("plan")));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_run_id_field_json(run, &object));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_candidate_identity_json(candidate, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("benchmark"), benchmark_plan->name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("case"), case_plan->name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("measure"), policy->measure));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("warmup_iterations"), policy->warmup_iterations));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("iterations"), policy->iterations));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("benchmark_sample_count"),
      benchmark_plan->sample_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("benchmark_cartesian_sample_count"),
      benchmark_plan->cartesian_sample_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
      &object, IREE_SV("benchmark_sample_count_truncated"),
      benchmark_plan->sample_count_truncated));
  if (options->sample_ordinal >= 0) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_int32_field(
        &object, IREE_SV("selected_sample"), options->sample_ordinal));
  }
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_kernel_launch_plan_json(
      module, case_plan, &object));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_case_sample_plan_fields_json(
      module, case_plan, &object));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("cli_overrides")));
  loom_json_object_writer_t cli_overrides;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &cli_overrides));
  IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
      &cli_overrides, IREE_SV("iterations"), options->iterations_specified));
  IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
      &cli_overrides, IREE_SV("warmup_iterations"),
      options->warmup_iterations_specified));
  if (policy->measure_kind == IREE_BENCHMARK_LOOM_MEASURE_DISPATCH_COMPLETE) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
        &cli_overrides, IREE_SV("batch_size"), options->batch_size_specified));
    IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
        &cli_overrides, IREE_SV("min_time_ms"),
        options->min_time_ms_specified));
    IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
        &cli_overrides, IREE_SV("warmup_time_ms"),
        options->warmup_time_ms_specified));
    IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
        &cli_overrides, IREE_SV("max_batches"),
        options->max_batches_specified));
    IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
        &cli_overrides, IREE_SV("stable_p90_to_p50_ppm"),
        options->stable_p90_to_p50_ppm_specified));
    IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
        &cli_overrides, IREE_SV("profile_final_batch"),
        options->profile_final_batch_specified));
    IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
        &cli_overrides, IREE_SV("profile_data"),
        options->profile_data_requested));
    IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
        &cli_overrides, IREE_SV("profile_counter"),
        options->profile_counters.count != 0));
    IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
        &cli_overrides, IREE_SV("profile_artifacts_dir"),
        !iree_string_view_is_empty(options->profile_artifacts_dir)));
    IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
        &cli_overrides, IREE_SV("input_ring_min_bytes"),
        options->input_ring_min_bytes_specified));
    IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
        &cli_overrides, IREE_SV("input_ring_count"),
        options->input_ring_count_specified));
  }
  IREE_RETURN_IF_ERROR(loom_json_object_end(&cli_overrides));
  if (policy->measure_kind == IREE_BENCHMARK_LOOM_MEASURE_DISPATCH_COMPLETE) {
    const loom_run_benchmark_options_t* timing = &policy->hal_options.timing;
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &object, IREE_SV("batch_size"), timing->batch_size));
    IREE_RETURN_IF_ERROR(loom_json_object_write_int64_field(
        &object, IREE_SV("min_time_ns"), timing->min_duration_ns));
    IREE_RETURN_IF_ERROR(loom_json_object_write_int64_field(
        &object, IREE_SV("warmup_time_ns"), timing->warmup_min_duration_ns));
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &object, IREE_SV("max_batches"), timing->max_batch_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("stable_p90_to_p50_ppm"),
        timing->stable_p90_to_p50_delta_ppm));
    IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
        &object, IREE_SV("profile_final_batch"),
        iree_all_bits_set(policy->hal_options.flags,
                          LOOM_RUN_HAL_BENCHMARK_FLAG_PROFILE_FINAL_BATCH)));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("data_cache_policy")));
    loom_json_object_writer_t data_cache_policy;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &data_cache_policy));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &data_cache_policy, IREE_SV("validity"), IREE_SV("check_ops")));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &data_cache_policy, IREE_SV("cache_policy"), IREE_SV("binding_ring")));
    IREE_RETURN_IF_ERROR(loom_json_object_write_int64_field(
        &data_cache_policy, IREE_SV("input_ring_min_bytes"),
        options->input_ring_min_bytes));
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &data_cache_policy, IREE_SV("input_ring_count"),
        options->input_ring_count));
    IREE_RETURN_IF_ERROR(loom_json_object_end(&data_cache_policy));
    if (iree_all_bits_set(policy->hal_options.flags,
                          LOOM_RUN_HAL_BENCHMARK_FLAG_PROFILE_FINAL_BATCH)) {
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
          &object, IREE_SV("profile_data_families"),
          policy->hal_options.profile_data_families));
      IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
          &object, IREE_SV("profile_data_family_names")));
      IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_profile_family_names_json(
          policy->hal_options.profile_data_families, &stream));
      IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
          &object, IREE_SV("profile_counter_set_count"),
          policy->hal_options.profile_counter_set_count));
      if (policy->hal_options.profile_counter_set_count != 0) {
        IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
            &object, IREE_SV("profile_counter_request")));
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
        profile_artifacts_status = loom_json_object_write_string_field(
            &object, IREE_SV("profile_artifacts_dir"),
            iree_string_builder_view(&profile_artifacts_dir));
      }
      iree_string_builder_deinitialize(&profile_artifacts_dir);
      IREE_RETURN_IF_ERROR(profile_artifacts_status);
    }
  }
  IREE_RETURN_IF_ERROR(loom_json_object_end(&object));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "\n"));
  return iree_ok_status();
}

iree_status_t iree_benchmark_loom_write_timing_stats_json(
    const iree_benchmark_loom_timing_stats_t* stats,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("count"), stats->count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_int64_field(
      &object, IREE_SV("total"), stats->total_ns));
  IREE_RETURN_IF_ERROR(loom_json_object_write_int64_field(
      &object, IREE_SV("min"), stats->minimum_ns));
  IREE_RETURN_IF_ERROR(loom_json_object_write_int64_field(
      &object, IREE_SV("max"), stats->maximum_ns));
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&object, IREE_SV("mean")));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_format(stream, "%.3f", stats->mean_ns));
  IREE_RETURN_IF_ERROR(loom_json_object_write_int64_field(
      &object, IREE_SV("p50"), stats->p50_ns));
  IREE_RETURN_IF_ERROR(loom_json_object_write_int64_field(
      &object, IREE_SV("p90"), stats->p90_ns));
  return loom_json_object_end(&object);
}

iree_status_t iree_benchmark_loom_write_benchmark_timing_stats_json(
    const loom_run_benchmark_timing_stats_t* stats,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("count"), stats->count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_int64_field(
      &object, IREE_SV("total"), stats->total_ns));
  IREE_RETURN_IF_ERROR(loom_json_object_write_int64_field(
      &object, IREE_SV("min"), stats->minimum_ns));
  IREE_RETURN_IF_ERROR(loom_json_object_write_int64_field(
      &object, IREE_SV("max"), stats->maximum_ns));
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&object, IREE_SV("mean")));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_format(stream, "%.3f", stats->mean_ns));
  IREE_RETURN_IF_ERROR(loom_json_object_write_int64_field(
      &object, IREE_SV("p50"), stats->p50_ns));
  IREE_RETURN_IF_ERROR(loom_json_object_write_int64_field(
      &object, IREE_SV("p90"), stats->p90_ns));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("p90_to_p50_delta_ppm"), stats->p90_to_p50_delta_ppm));
  return loom_json_object_end(&object);
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
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
  if (iree_all_bits_set(
          flags, IREE_HAL_DEVICE_PROFILING_FLAG_LIGHTWEIGHT_STATISTICS)) {
    IREE_RETURN_IF_ERROR(loom_json_array_write_string_element(
        &array, IREE_SV("lightweight_statistics")));
  }
  return loom_json_array_end(&array);
}

static iree_status_t iree_benchmark_loom_write_hal_profile_duration_json(
    const loom_run_hal_profile_row_summary_t* row,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
      &object, IREE_SV("available"),
      iree_all_bits_set(row->flags,
                        IREE_HAL_PROFILE_STATISTICS_ROW_FLAG_TIMING)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("time_domain"),
      iree_make_cstring_view(
          iree_benchmark_loom_profile_statistics_time_domain_name(
              row->time_domain))));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("first_start"), row->first_start_time));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("last_end"), row->last_end_time));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("total"), row->total_duration));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("min"), row->minimum_duration));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("max"), row->maximum_duration));
  const uint64_t valid_sample_count =
      row->sample_count >= row->invalid_sample_count
          ? row->sample_count - row->invalid_sample_count
          : 0;
  if (valid_sample_count != 0) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("mean"), row->total_duration / valid_sample_count));
  }
  IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
      &object, IREE_SV("scaled_ns"), row->has_scaled_duration_ns));
  if (row->has_scaled_duration_ns) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("total_ns"), row->total_duration_ns));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("min_ns"), row->minimum_duration_ns));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("max_ns"), row->maximum_duration_ns));
    if (valid_sample_count != 0) {
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
          &object, IREE_SV("mean_ns"),
          row->total_duration_ns / valid_sample_count));
    }
  }
  return loom_json_object_end(&object);
}

static iree_status_t iree_benchmark_loom_write_hal_profile_row_summary_json(
    const loom_run_hal_profile_row_summary_t* row,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("type"),
      iree_make_cstring_view(
          iree_benchmark_loom_profile_statistics_row_type_name(
              row->row_type))));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("row_type"), row->row_type));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("flags"), row->flags));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("physical_device_ordinal"),
      row->physical_device_ordinal));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("queue_ordinal"), row->queue_ordinal));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("event_type"), row->event_type));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("executable_id"), row->executable_id));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("command_buffer_id"), row->command_buffer_id));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("function_ordinal"), row->function_ordinal));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("command_index"), row->command_index));
  if (row->function_name_length != 0) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("function_name"),
        iree_make_string_view(row->function_name, row->function_name_length)));
  }
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("sample_count"), row->sample_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("invalid_sample_count"), row->invalid_sample_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("operation_count"), row->operation_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("payload_bytes"), row->payload_bytes));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("tile_count"), row->tile_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("tile_duration_sum_ns"), row->tile_duration_sum_ns));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("timing")));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_hal_profile_duration_json(row, stream));
  return loom_json_object_end(&object);
}

static iree_status_t
iree_benchmark_loom_write_hal_profile_dispatch_distribution_json(
    const loom_run_hal_profile_dispatch_distribution_t* distribution,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
      &object, IREE_SV("available"), distribution->available));
  IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
      &object, IREE_SV("complete"), distribution->complete));
  IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
      &object, IREE_SV("comparable"), distribution->comparable));
  IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
      &object, IREE_SV("homogeneous_function"),
      distribution->homogeneous_function));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("source"),
      iree_make_cstring_view(
          iree_benchmark_loom_profile_statistics_row_type_name(
              distribution->source_row_type))));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("source_row_type"), distribution->source_row_type));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("source_sample_count"),
      distribution->source_sample_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("invalid_sample_count"),
      distribution->invalid_sample_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("unrepresented_sample_count"),
      distribution->unrepresented_sample_count));
  if (distribution->available) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("physical_device_ordinal"),
        distribution->physical_device_ordinal));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("source_time_domain"),
        iree_make_cstring_view(
            iree_benchmark_loom_profile_statistics_time_domain_name(
                distribution->time_domain))));
    if (distribution->homogeneous_function) {
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
          &object, IREE_SV("executable_id"), distribution->executable_id));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &object, IREE_SV("function_ordinal"),
          distribution->function_ordinal));
    }
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("duration_ns")));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_benchmark_timing_stats_json(
        &distribution->duration_ns, stream));
  }
  return loom_json_object_end(&object);
}

iree_status_t iree_benchmark_loom_write_hal_profile_summary_json(
    const loom_run_hal_profile_summary_t* profile,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
      &object, IREE_SV("requested"), profile->requested));
  IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
      &object, IREE_SV("executed"), profile->executed));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("flags"), profile->flags));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("flag_names")));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_profile_flag_names_json(
      profile->flags, stream));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("data_families"), profile->data_families));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("data_family_names")));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_profile_family_names_json(
      profile->data_families, stream));
  if (profile->has_artifact_path) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("artifact_path"),
        iree_make_string_view(profile->artifact_path,
                              profile->artifact_path_length)));
  }
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("row_count"), profile->row_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("captured_row_count"), profile->captured_row_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("truncated_row_count"), profile->truncated_row_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("dropped_record_count"), profile->dropped_record_count));
  if (profile->dispatch_distribution.source_row_type !=
      IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_NONE) {
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
        &object, IREE_SV("dispatch_distribution")));
    IREE_RETURN_IF_ERROR(
        iree_benchmark_loom_write_hal_profile_dispatch_distribution_json(
            &profile->dispatch_distribution, stream));
  }
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&object, IREE_SV("rows")));
  loom_json_array_writer_t rows;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &rows));
  for (iree_host_size_t i = 0; i < profile->captured_row_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&rows));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_hal_profile_row_summary_json(
        &profile->rows[i], stream));
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&rows));
  if (profile->has_error) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_status_field_json(
        profile->error_code,
        iree_make_string_view(profile->error_message,
                              profile->error_message_length),
        &object));
  }
  return loom_json_object_end(&object);
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
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field_if_nonempty(
      &object, IREE_SV("entry"), benchmark_result->failure_entry));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("stage"), benchmark_result->failure_stage));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("kind"), benchmark_result->failure_kind));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field_if_nonempty(
      &object, IREE_SV("message"), benchmark_result->failure_message));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("diagnostic_error_count"),
      benchmark_result->diagnostic_error_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("diagnostic_warning_count"),
      benchmark_result->diagnostic_warning_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("diagnostic_remark_count"),
      benchmark_result->diagnostic_remark_count));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("diagnostics")));
  IREE_RETURN_IF_ERROR(loom_json_write_value_list_array(
      benchmark_result->diagnostic_json, stream));
  return loom_json_object_end(&object);
}

static iree_status_t iree_benchmark_loom_write_compile_rejection_fields_json(
    const iree_benchmark_loom_hal_actual_provider_t* provider,
    loom_json_object_writer_t* object) {
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      object, IREE_SV("stage"), provider->execution.compile_failure_stage));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      object, IREE_SV("kind"), provider->execution.compile_failure_kind));
  return loom_json_object_write_string_field_if_nonempty(
      object, IREE_SV("message"), provider->execution.compile_failure_message);
}

iree_status_t iree_benchmark_loom_write_diagnostic_capture_fields_json(
    const iree_benchmark_loom_diagnostic_capture_t* diagnostics,
    loom_json_object_writer_t* object) {
  if (diagnostics == NULL) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      object, IREE_SV("diagnostic_error_count"), diagnostics->error_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      object, IREE_SV("diagnostic_warning_count"), diagnostics->warning_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      object, IREE_SV("diagnostic_remark_count"), diagnostics->remark_count));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(object, IREE_SV("diagnostics")));
  return iree_benchmark_loom_write_diagnostic_array_json(diagnostics,
                                                         object->stream);
}

iree_status_t iree_benchmark_loom_write_planning_issue_fields_json(
    const loom_testbench_module_plan_t* testbench_plan,
    const loom_testbench_issue_t* planning_issues,
    iree_host_size_t planning_issue_count, loom_json_object_writer_t* object) {
  if (planning_issue_count == 0) {
    return iree_ok_status();
  }
  IREE_ASSERT_ARGUMENT(testbench_plan);
  IREE_ASSERT_ARGUMENT(planning_issues);
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      object, IREE_SV("planning_issue_count"), planning_issue_count));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(object, IREE_SV("planning_issues")));
  return loom_testbench_issue_array_write_json(
      testbench_plan, planning_issues, planning_issue_count, object->stream);
}

enum {
  IREE_BENCHMARK_LOOM_SHORT_MEASURED_DURATION_NS = 1000 * 1000,
  IREE_BENCHMARK_LOOM_SUB_MICROSECOND_DURATION_NS = 1000,
  IREE_BENCHMARK_LOOM_SMALL_PHYSICAL_DISPATCH_SAMPLE_COUNT = 100,
  IREE_BENCHMARK_LOOM_MIN_PROFILED_DISPATCH_SAMPLE_COUNT = 16,
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
  // True when bounded diagnostic row copies omit profile detail rows.
  bool detail_rows_truncated;
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
  // Exact per-dispatch duration distribution recovered during profiling.
  loom_run_hal_profile_dispatch_distribution_t dispatch_distribution;
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
  timing.detail_rows_truncated = profile->truncated_row_count != 0;
  timing.dispatch_distribution = profile->dispatch_distribution;
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

static iree_status_t
iree_benchmark_loom_write_profiled_dispatch_duration_ns_json(
    const iree_benchmark_loom_profiled_dispatch_timing_t* timing,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("count"), timing->valid_sample_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("total"), timing->total_duration_ns));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("min"), timing->minimum_duration_ns));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("max"), timing->maximum_duration_ns));
  if (timing->valid_sample_count != 0) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("mean")));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, "%.3f",
        (double)timing->total_duration_ns /
            (double)timing->valid_sample_count));
  }
  return loom_json_object_end(&object);
}

static iree_status_t iree_benchmark_loom_write_profiled_dispatch_timing_json(
    const iree_benchmark_loom_profiled_dispatch_timing_t* timing,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
      &object, IREE_SV("available"), timing->available));
  IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
      &object, IREE_SV("comparable"), timing->comparable));
  IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
      &object, IREE_SV("overlapped"), timing->overlapped));
  IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
      &object, IREE_SV("detail_rows_truncated"),
      timing->detail_rows_truncated));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("row_count"), timing->row_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("ignored_row_count"), timing->ignored_row_count));
  if (timing->available) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("physical_device_ordinal"),
        timing->physical_device_ordinal));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("sample_count"), timing->sample_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("invalid_sample_count"),
        timing->invalid_sample_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("valid_sample_count"), timing->valid_sample_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("span"), timing->span_duration));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("total"), timing->total_duration));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("min"), timing->minimum_duration));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("max"), timing->maximum_duration));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("overlap_ratio_ppm"), timing->overlap_ratio_ppm));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("time_domain"),
        iree_make_cstring_view(
            iree_benchmark_loom_profile_statistics_time_domain_name(
                timing->time_domain))));
  }
  if (timing->available && timing->has_scaled_duration_ns) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("duration_ns")));
    IREE_RETURN_IF_ERROR(
        iree_benchmark_loom_write_profiled_dispatch_duration_ns_json(timing,
                                                                     stream));
  }
  if (timing->dispatch_distribution.source_row_type !=
      IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_NONE) {
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
        &object, IREE_SV("dispatch_distribution")));
    IREE_RETURN_IF_ERROR(
        iree_benchmark_loom_write_hal_profile_dispatch_distribution_json(
            &timing->dispatch_distribution, stream));
  }
  return loom_json_object_end(&object);
}

static iree_status_t iree_benchmark_loom_write_hal_measurement_warnings_json(
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
  loom_json_array_writer_t warnings;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &warnings));
  if (timing->measured_duration_ns <
      IREE_BENCHMARK_LOOM_SHORT_MEASURED_DURATION_NS) {
    IREE_RETURN_IF_ERROR(loom_json_array_write_string_element(
        &warnings, IREE_SV("short_measured_duration")));
  }
  if (timing->batch_size == 1) {
    IREE_RETURN_IF_ERROR(loom_json_array_write_string_element(
        &warnings, IREE_SV("single_logical_operation_batch")));
  }
  if (measured_physical_dispatch_count <
      IREE_BENCHMARK_LOOM_SMALL_PHYSICAL_DISPATCH_SAMPLE_COUNT) {
    IREE_RETURN_IF_ERROR(loom_json_array_write_string_element(
        &warnings, IREE_SV("low_physical_dispatch_sample_count")));
  }
  if (operation_timing->p50_ns <
      IREE_BENCHMARK_LOOM_SUB_MICROSECOND_DURATION_NS) {
    IREE_RETURN_IF_ERROR(loom_json_array_write_string_element(
        &warnings, IREE_SV("sub_microsecond_logical_operation")));
  }
  const uint64_t accepted_spread =
      policy->hal_options.timing.stable_p90_to_p50_delta_ppm;
  if (accepted_spread != 0 &&
      operation_timing->p90_to_p50_delta_ppm > accepted_spread) {
    IREE_RETURN_IF_ERROR(loom_json_array_write_string_element(
        &warnings, IREE_SV("unstable_p90_to_p50")));
  }
  return loom_json_array_end(&warnings);
}

static iree_status_t iree_benchmark_loom_write_hal_profile_replay_warnings_json(
    const iree_benchmark_loom_benchmark_policy_t* policy,
    const iree_benchmark_loom_benchmark_result_t* benchmark_result,
    const iree_benchmark_loom_profiled_dispatch_timing_t* dispatch_timing,
    loom_output_stream_t* stream) {
  loom_json_array_writer_t warnings;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &warnings));
  const loom_run_hal_profile_summary_t* profile_replay =
      &benchmark_result->hal_benchmark.profile_replay;
  const loom_run_hal_profile_dispatch_distribution_t* dispatch_distribution =
      &dispatch_timing->dispatch_distribution;
  if (dispatch_timing->overlapped) {
    IREE_RETURN_IF_ERROR(loom_json_array_write_string_element(
        &warnings, IREE_SV("dispatch_overlap")));
  }
  if (profile_replay->requested && profile_replay->executed) {
    if (!dispatch_distribution->available) {
      IREE_RETURN_IF_ERROR(loom_json_array_write_string_element(
          &warnings, IREE_SV("dispatch_distribution_unavailable")));
    } else {
      if (!dispatch_distribution->complete) {
        IREE_RETURN_IF_ERROR(loom_json_array_write_string_element(
            &warnings, IREE_SV("dispatch_distribution_incomplete")));
      }
      if (!dispatch_distribution->comparable) {
        IREE_RETURN_IF_ERROR(loom_json_array_write_string_element(
            &warnings, IREE_SV("dispatch_distribution_noncomparable")));
      }
      if (!dispatch_distribution->homogeneous_function) {
        IREE_RETURN_IF_ERROR(loom_json_array_write_string_element(
            &warnings, IREE_SV("dispatch_distribution_heterogeneous")));
      }
      if (dispatch_distribution->duration_ns.count <
          IREE_BENCHMARK_LOOM_MIN_PROFILED_DISPATCH_SAMPLE_COUNT) {
        IREE_RETURN_IF_ERROR(loom_json_array_write_string_element(
            &warnings, IREE_SV("low_dispatch_sample_count")));
      }
      const uint64_t accepted_spread =
          policy->hal_options.timing.stable_p90_to_p50_delta_ppm;
      if (accepted_spread != 0 &&
          dispatch_distribution->duration_ns.p90_to_p50_delta_ppm >
              accepted_spread) {
        IREE_RETURN_IF_ERROR(loom_json_array_write_string_element(
            &warnings, IREE_SV("unstable_dispatch_p90_to_p50")));
      }
    }
  }
  return loom_json_array_end(&warnings);
}

iree_status_t iree_benchmark_loom_write_hal_profile_replay_json(
    const iree_benchmark_loom_benchmark_policy_t* policy,
    const iree_benchmark_loom_benchmark_result_t* benchmark_result,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_validate_hal_benchmark_result(benchmark_result));
  const iree_benchmark_loom_profiled_dispatch_timing_t profiled_dispatch =
      iree_benchmark_loom_profiled_dispatch_timing(
          &benchmark_result->hal_benchmark.profile_replay);
  const loom_run_hal_profile_dispatch_distribution_t* dispatch_distribution =
      &profiled_dispatch.dispatch_distribution;
  iree_host_size_t physical_dispatches_per_logical_operation = 0;
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_hal_physical_dispatches_per_logical_operation(
          benchmark_result, &physical_dispatches_per_logical_operation));
  const bool comparison_eligible =
      physical_dispatches_per_logical_operation == 1 &&
      !profiled_dispatch.overlapped && dispatch_distribution->available &&
      dispatch_distribution->complete && dispatch_distribution->comparable &&
      dispatch_distribution->homogeneous_function &&
      dispatch_distribution->duration_ns.count >=
          IREE_BENCHMARK_LOOM_MIN_PROFILED_DISPATCH_SAMPLE_COUNT;

  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("measurement_relationship"),
      IREE_SV("distinct_execution")));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("summary")));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_hal_profile_summary_json(
      &benchmark_result->hal_benchmark.profile_replay, stream));
  if (profiled_dispatch.available) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("dispatch_timing")));
    IREE_RETURN_IF_ERROR(
        iree_benchmark_loom_write_profiled_dispatch_timing_json(
            &profiled_dispatch, stream));
  }
  if (comparison_eligible) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("comparison")));
    loom_json_object_writer_t comparison;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &comparison));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &comparison, IREE_SV("metric"),
        IREE_SV("dispatch_timing.dispatch_distribution.duration_ns.p50")));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &comparison, IREE_SV("time_domain"), IREE_SV("device_tick_scaled_ns")));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &comparison, IREE_SV("meaning"),
        IREE_SV("profile_replay_physical_dispatch_p50")));
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &comparison, IREE_SV("sample_count"),
        dispatch_distribution->duration_ns.count));
    IREE_RETURN_IF_ERROR(loom_json_object_end(&comparison));
  }
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("warnings")));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_hal_profile_replay_warnings_json(
          policy, benchmark_result, &profiled_dispatch, stream));
  return loom_json_object_end(&object);
}

iree_status_t iree_benchmark_loom_write_hal_timing_interpretation_json(
    const iree_benchmark_loom_benchmark_policy_t* policy,
    const iree_benchmark_loom_benchmark_result_t* benchmark_result,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_validate_hal_benchmark_result(benchmark_result));
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("score"), IREE_SV("operation_timing_ns")));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("score_time_domain"), IREE_SV("host_wall")));
  const iree_string_view_t score_meaning =
      benchmark_result->hal_benchmark.timing.batch_size > 1
          ? IREE_SV("host_queue_completion_throughput_normalized_batch_time")
          : IREE_SV("host_queue_completion_normalized_logical_operation_time");
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("score_meaning"), score_meaning));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("score_unit"), IREE_SV("logical_operation")));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("warnings")));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_hal_measurement_warnings_json(
      policy, benchmark_result, stream));
  return loom_json_object_end(&object);
}

static iree_status_t iree_benchmark_loom_write_compile_report_field_json(
    const loom_compile_report_capture_t* compile_report_capture,
    loom_json_object_writer_t* object) {
  if (compile_report_capture == NULL ||
      !loom_compile_report_capture_is_enabled(compile_report_capture)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(object, IREE_SV("compile_report")));
  return loom_compile_report_capture_append_json(compile_report_capture,
                                                 object->stream);
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
  if (!iree_string_view_is_empty(provider->artifact_path_suffix)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(stem, "_"));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_append_sanitized_path_component(
        provider->artifact_path_suffix, stem));
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
    iree_string_view_t leaf, const iree_byte_sequence_t* contents,
    iree_allocator_t allocator, char** inout_path_storage,
    iree_string_view_t* inout_path) {
  if (!iree_benchmark_loom_artifact_bundle_wants_debug_artifacts(bundle) ||
      !iree_string_view_is_empty(*inout_path)) {
    return iree_ok_status();
  }
  if (contents == NULL || iree_byte_sequence_length(contents) == 0) {
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
    status = loom_tooling_write_output_byte_sequence(
        iree_make_cstring_view(path_storage), contents, allocator);
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
    const loom_artifact_t* artifact,
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
      !provider->execution.candidate.artifact_candidate.compiled) {
    return iree_ok_status();
  }

  iree_string_builder_t leaf;
  iree_string_builder_initialize(allocator, &leaf);
  const loom_artifact_t* artifact =
      &provider->execution.candidate.artifact_candidate.artifact;
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
      provider->execution.run_module->module,
      provider->execution.kernel_launch->callee_ref, &entry_symbol));

  loom_output_stream_t stream;
  loom_output_stream_for_builder(output, &stream);
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("type"), IREE_SV("compile_report")));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_run_id_field_json(run, &object));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_candidate_identity_json(candidate, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("benchmark"), benchmark_plan->name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("case"), case_plan->name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("entry"), entry_symbol));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field_if_nonempty(
      &object, IREE_SV("target_artifact_path"),
      provider->target_artifact_path));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field_if_nonempty(
      &object, IREE_SV("artifact_manifest_path"),
      provider->artifact_manifest_path));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field_if_nonempty(
      &object, IREE_SV("target_listing_path"), provider->target_listing_path));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field_if_nonempty(
      &object, IREE_SV("hal_executable_path"), provider->hal_executable_path));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("state"),
      provider->execution.compile_rejected ? IREE_SV("failed")
                                           : IREE_SV("ok")));
  if (provider->execution.compile_rejected) {
    IREE_RETURN_IF_ERROR(
        iree_benchmark_loom_write_compile_rejection_fields_json(provider,
                                                                &object));
  }
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_diagnostic_capture_fields_json(
      &provider->diagnostics, &object));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_compile_report_field_json(
      &provider->compile_report_capture, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&object));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "\n"));
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
      !loom_compile_report_capture_is_enabled(
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
    loom_json_object_writer_t* object, iree_string_view_t field_name,
    iree_benchmark_loom_buffer_materialization_t materialization) {
  iree_string_view_t name =
      iree_benchmark_loom_buffer_materialization_name(materialization);
  if (iree_string_view_is_empty(name)) {
    return iree_ok_status();
  }
  return loom_json_object_write_string_field(object, field_name, name);
}

static iree_status_t iree_benchmark_loom_write_data_cache_summary_json(
    const iree_benchmark_loom_data_cache_summary_t* summary,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("validity"), IREE_SV("check_ops")));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("cache_policy"), IREE_SV("binding_ring")));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_data_cache_materialization_json(
          &object, IREE_SV("correctness_materialization"),
          summary->correctness_materialization));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_data_cache_materialization_json(
          &object, IREE_SV("measurement_materialization"),
          summary->measurement_materialization));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("binding_count"), summary->binding_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("binding_ring_count"), summary->binding_ring_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("command_buffer_ring_count"),
      summary->command_buffer_ring_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("dispatches_per_batch"), summary->dispatches_per_batch));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("requested_min_ring_bytes"),
      summary->requested_min_ring_bytes));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("binding_set_bytes"), summary->binding_set_bytes));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("binding_ring_bytes"), summary->binding_ring_bytes));
  const bool ring_byte_target_met =
      summary->requested_min_ring_bytes == 0 ||
      summary->binding_ring_bytes >= summary->requested_min_ring_bytes;
  IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
      &object, IREE_SV("ring_byte_target_met"), ring_byte_target_met));
  return loom_json_object_end(&object);
}

static iree_status_t iree_benchmark_loom_write_data_cache_summary_field_json(
    const iree_benchmark_loom_data_cache_summary_t* summary,
    loom_json_object_writer_t* object) {
  if (!summary->populated) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(object, IREE_SV("data_cache")));
  return iree_benchmark_loom_write_data_cache_summary_json(summary,
                                                           object->stream);
}

iree_status_t iree_benchmark_loom_write_benchmark_policy_json(
    const iree_benchmark_loom_benchmark_policy_t* policy,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("measure"), policy->measure));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("warmup_iterations"), policy->warmup_iterations));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("iterations"), policy->iterations));
  return loom_json_object_end(&object);
}

iree_status_t iree_benchmark_loom_write_benchmark_correctness_json(
    iree_host_size_t sample_count, iree_host_size_t failed_sample_count,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("sample_count"), sample_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("failed_sample_count"), failed_sample_count));
  return loom_json_object_end(&object);
}

static bool iree_benchmark_loom_benchmark_result_has_measurement(
    const iree_benchmark_loom_benchmark_result_t* benchmark_result) {
  return benchmark_result->executed || benchmark_result->has_hal_benchmark;
}

iree_status_t iree_benchmark_loom_write_benchmark_measurement_json(
    const iree_benchmark_loom_benchmark_policy_t* policy,
    const iree_benchmark_loom_benchmark_result_t* benchmark_result,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("samples_per_iteration"),
      benchmark_result->samples_per_iteration));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("failed_sample_count"),
      benchmark_result->failed_sample_count));
  if (benchmark_result->executed && !benchmark_result->has_hal_benchmark) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("timing_ns")));
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
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &object, IREE_SV("logical_operations_per_batch"), timing->batch_size));
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &object, IREE_SV("physical_dispatches_per_batch"),
        physical_dispatches_per_batch));
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &object, IREE_SV("physical_dispatches_per_logical_operation"),
        physical_dispatches_per_logical_operation));
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &object, IREE_SV("warmup_batch_count"), timing->warmup_batch_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_int64_field(
        &object, IREE_SV("warmup_duration_ns"), timing->warmup_duration_ns));
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &object, IREE_SV("measured_batch_count"),
        timing->measured_batch_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &object, IREE_SV("measured_logical_operation_count"),
        timing->measured_operation_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &object, IREE_SV("measured_physical_dispatch_count"),
        measured_physical_dispatch_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_int64_field(
        &object, IREE_SV("measured_duration_ns"),
        timing->measured_duration_ns));
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
        &object, IREE_SV("mean_physical_dispatch_duration_ns")));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, "%.3f", mean_physical_dispatch_duration_ns));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("stop_reason"),
        loom_run_benchmark_stop_reason_name(timing->stop_reason)));
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
        &object, IREE_SV("timing_interpretation")));
    IREE_RETURN_IF_ERROR(
        iree_benchmark_loom_write_hal_timing_interpretation_json(
            policy, benchmark_result, stream));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("batch_timing_ns")));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_benchmark_timing_stats_json(
        &timing->batch_timing, stream));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("operation_timing_ns")));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_benchmark_timing_stats_json(
        &timing->operation_timing, stream));
  }
  return loom_json_object_end(&object);
}

iree_status_t iree_benchmark_loom_write_benchmark_evidence_fields_json(
    const iree_benchmark_loom_benchmark_policy_t* policy,
    const iree_benchmark_loom_benchmark_result_t* benchmark_result,
    iree_host_size_t correctness_sample_count,
    iree_host_size_t correctness_failed_sample_count,
    loom_json_object_writer_t* object) {
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(object, IREE_SV("policy")));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_benchmark_policy_json(policy, object->stream));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(object, IREE_SV("correctness")));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_benchmark_correctness_json(
      correctness_sample_count, correctness_failed_sample_count,
      object->stream));
  if (benchmark_result->launch_evidence != NULL &&
      benchmark_result->launch_evidence->record_count != 0) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(object, IREE_SV("launches")));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_launch_evidence_json(
        benchmark_result->launch_evidence, object->stream));
  }
  if (iree_benchmark_loom_benchmark_result_has_measurement(benchmark_result)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(object, IREE_SV("measurement")));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_benchmark_measurement_json(
        policy, benchmark_result, object->stream));
  }
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_data_cache_summary_field_json(
      &benchmark_result->data_cache, object));
  if (benchmark_result->hal_benchmark.profile_replay.requested ||
      benchmark_result->hal_benchmark.profile_replay.executed ||
      benchmark_result->hal_benchmark.profile_replay.has_error) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(object, IREE_SV("profile_replay")));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_hal_profile_replay_json(
        policy, benchmark_result, object->stream));
  }
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_compile_report_field_json(
      benchmark_result->compile_report_capture, object));
  if (benchmark_result->has_failure) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(object, IREE_SV("failure")));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_benchmark_failure_json(
        benchmark_result, object->stream));
  }
  return iree_ok_status();
}

iree_status_t iree_benchmark_loom_write_summary_counts_json(
    const iree_benchmark_loom_summary_counts_t* counts,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("planned")));
  loom_json_object_writer_t planned;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &planned));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &planned, IREE_SV("case_count"), counts->planned_case_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &planned, IREE_SV("benchmark_count"), counts->planned_benchmark_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &planned, IREE_SV("selected_benchmark_count"),
      counts->selected_benchmark_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &planned, IREE_SV("logical_sample_count"), counts->logical_sample_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &planned, IREE_SV("work_item_count"), counts->work_item_count));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&planned));

  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("failures")));
  loom_json_object_writer_t failures;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &failures));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &failures, IREE_SV("row_count"), counts->failure_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &failures, IREE_SV("failed_benchmark_count"),
      counts->failed_benchmark_count));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&failures));

  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("correctness")));
  loom_json_object_writer_t correctness;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &correctness));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &correctness, IREE_SV("sample_count"), counts->correctness_sample_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &correctness, IREE_SV("failed_sample_count"),
      counts->correctness_failed_sample_count));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&correctness));

  if (counts->artifact_bundle_enabled || counts->fixture_read_count != 0 ||
      counts->file_output_count != 0 || counts->profile_count != 0 ||
      counts->compile_report_count != 0 ||
      counts->artifact_manifest_count != 0 ||
      counts->target_artifact_count != 0 || counts->target_listing_count != 0 ||
      counts->hal_executable_count != 0) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("artifacts")));
    loom_json_object_writer_t artifacts;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &artifacts));
    IREE_RETURN_IF_ERROR(
        loom_json_object_write_bool_field(&artifacts, IREE_SV("bundle_enabled"),
                                          counts->artifact_bundle_enabled));
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &artifacts, IREE_SV("fixture_read_count"), counts->fixture_read_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &artifacts, IREE_SV("file_output_count"), counts->file_output_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &artifacts, IREE_SV("profile_count"), counts->profile_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &artifacts, IREE_SV("compile_report_count"),
        counts->compile_report_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &artifacts, IREE_SV("artifact_manifest_count"),
        counts->artifact_manifest_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &artifacts, IREE_SV("target_artifact_count"),
        counts->target_artifact_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &artifacts, IREE_SV("target_listing_count"),
        counts->target_listing_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &artifacts, IREE_SV("hal_executable_count"),
        counts->hal_executable_count));
    IREE_RETURN_IF_ERROR(loom_json_object_end(&artifacts));
  }
  return loom_json_object_end(&object);
}

iree_status_t iree_benchmark_loom_write_benchmark_result_json(
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_benchmark_loom_benchmark_policy_t* policy,
    const iree_benchmark_loom_benchmark_result_t* benchmark_result,
    iree_host_size_t correctness_sample_count,
    iree_host_size_t correctness_failed_sample_count,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("benchmark"), benchmark_plan->name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("case"), case_plan->name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("state"),
      iree_benchmark_loom_benchmark_result_state(benchmark_result)));
  if (benchmark_result->has_sample_ordinal) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &object, IREE_SV("sample_ordinal"), benchmark_result->sample_ordinal));
  }
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field_if_nonempty(
      &object, IREE_SV("compile_report_path"),
      benchmark_result->compile_report_artifact_path));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field_if_nonempty(
      &object, IREE_SV("artifact_manifest_path"),
      benchmark_result->artifact_manifest_path));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field_if_nonempty(
      &object, IREE_SV("target_artifact_path"),
      benchmark_result->target_artifact_path));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field_if_nonempty(
      &object, IREE_SV("target_listing_path"),
      benchmark_result->target_listing_path));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field_if_nonempty(
      &object, IREE_SV("hal_executable_path"),
      benchmark_result->hal_executable_path));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_benchmark_evidence_fields_json(
      policy, benchmark_result, correctness_sample_count,
      correctness_failed_sample_count, &object));
  return loom_json_object_end(&object);
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
      provider->execution.run_module->module,
      provider->execution.kernel_launch->callee_ref, &entry_symbol));

  loom_output_stream_t stream;
  loom_output_stream_for_builder(compile_output, &stream);
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("row"), IREE_SV("compile")));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_run_id_field_json(run, &object));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_candidate_identity_json(candidate, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("benchmark"), benchmark_plan->name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("case"), case_plan->name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("entry"), entry_symbol));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("state"),
      provider->execution.compile_rejected ? IREE_SV("failed")
                                           : IREE_SV("ok")));
  if (provider->execution.compile_rejected) {
    IREE_RETURN_IF_ERROR(
        iree_benchmark_loom_write_compile_rejection_fields_json(provider,
                                                                &object));
  }
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_diagnostic_capture_fields_json(
      &provider->diagnostics, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field_if_nonempty(
      &object, IREE_SV("compile_report_path"),
      provider->compile_report_artifact_path));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field_if_nonempty(
      &object, IREE_SV("artifact_manifest_path"),
      provider->artifact_manifest_path));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field_if_nonempty(
      &object, IREE_SV("target_artifact_path"),
      provider->target_artifact_path));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field_if_nonempty(
      &object, IREE_SV("target_listing_path"), provider->target_listing_path));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field_if_nonempty(
      &object, IREE_SV("hal_executable_path"), provider->hal_executable_path));
  if (provider->execution.compile_report_available) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_compile_report_field_json(
        &provider->compile_report_capture, &object));
  }
  IREE_RETURN_IF_ERROR(loom_json_object_end(&object));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "\n"));
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
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("row"), IREE_SV("benchmark")));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_run_id_field_json(run, &object));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_candidate_identity_json(candidate, &object));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_work_item_index_field_json(
      work_item_index, &object));
  if (benchmark_result->has_sample_ordinal) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_sample_fields_json(
        module, case_plan, benchmark_result->sample_ordinal, &object));
  }
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("benchmark_result")));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_benchmark_result_json(
      benchmark_plan, case_plan, policy, benchmark_result,
      correctness_sample_count, correctness_failed_sample_count, &stream));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&object));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "\n"));
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
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("row"), IREE_SV("benchmark.repetition")));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_run_id_field_json(run, &object));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_candidate_identity_json(
      &selection->identity, &object));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_sample_fields_json(
      candidate->module, selection->case_plan, benchmark_result->sample_ordinal,
      &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("comparison_group"), comparison_group));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("baseline_candidate_id"), baseline->candidate_id));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_string_field(&object, IREE_SV("method"), method));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("order_index"), order_index));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("repetition_index"), repetition_index));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("schedule_token"),
      iree_make_string_view(&schedule_token, 1)));
  if (profile_suppressed) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
        &object, IREE_SV("profile_suppressed_for_interleave"), true));
  }
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("benchmark_result")));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_benchmark_result_json(
      selection->benchmark_plan, selection->case_plan, &selection->policy,
      benchmark_result, candidate->correctness_sample_count,
      candidate->correctness_failed_sample_count, &stream));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&object));
  return loom_output_stream_write_cstring(&stream, "\n");
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
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("row"), IREE_SV("comparison")));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_run_id_field_json(run, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("comparison_group"), comparison_group));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_string_field(&object, IREE_SV("method"), method));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("baseline_candidate_id"),
      baseline->selection->identity.candidate_id));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("candidate_id"),
      candidate->selection->identity.candidate_id));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("baseline_repetition_count"), baseline->sample_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("candidate_repetition_count"), candidate->sample_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_int64_field(
      &object, IREE_SV("baseline_p50_ns"), baseline_p50.p50_ns));
  IREE_RETURN_IF_ERROR(loom_json_object_write_int64_field(
      &object, IREE_SV("candidate_p50_ns"), candidate_p50.p50_ns));
  IREE_RETURN_IF_ERROR(loom_json_object_write_int64_field(
      &object, IREE_SV("baseline_p90_ns"), baseline_p90.p50_ns));
  IREE_RETURN_IF_ERROR(loom_json_object_write_int64_field(
      &object, IREE_SV("candidate_p90_ns"), candidate_p90.p50_ns));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("baseline_p50_spread_ppm"),
      baseline_p50.p90_to_p50_delta_ppm));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("candidate_p50_spread_ppm"),
      candidate_p50.p90_to_p50_delta_ppm));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("baseline_p90_spread_ppm"),
      baseline_p90.p90_to_p50_delta_ppm));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("candidate_p90_spread_ppm"),
      candidate_p90.p90_to_p50_delta_ppm));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("ratio_p50")));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_format(&stream, "%.6f", ratio_p50));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("speedup_p50")));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_format(&stream, "%.6f", speedup_p50));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("ratio_p90")));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_format(&stream, "%.6f", ratio_p90));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("speedup_p90")));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_format(&stream, "%.6f", speedup_p90));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&object));
  return loom_output_stream_write_cstring(&stream, "\n");
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
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("row"), IREE_SV("failure")));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_run_id_field_json(run, &object));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_string_field(&object, IREE_SV("stage"), stage));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_string_field(&object, IREE_SV("kind"), kind));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field_if_nonempty(
      &object, IREE_SV("message"), message));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_diagnostic_capture_fields_json(
      diagnostics, &object));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_planning_issue_fields_json(
      testbench_plan, planning_issues, planning_issue_count, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&object));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "\n"));
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
    iree_string_builder_t* output) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(output, &stream);
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("row"), IREE_SV("summary")));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_run_id_field_json(run, &object));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_bool_field(&object, IREE_SV("dry_run"), dry_run));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("summary")));
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
  IREE_RETURN_IF_ERROR(loom_json_object_end(&object));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "\n"));
  return iree_ok_status();
}
