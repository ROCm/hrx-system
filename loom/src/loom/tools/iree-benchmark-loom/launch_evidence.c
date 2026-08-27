// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-benchmark-loom/launch_evidence.h"

#include <string.h>

#include "loom/ir/module.h"
#include "loom/tooling/execution/hal/testbench_actual.h"

iree_status_t iree_benchmark_loom_launch_evidence_initialize(
    iree_host_size_t record_count, iree_host_size_t workload_value_count,
    iree_allocator_t host_allocator,
    iree_benchmark_loom_launch_evidence_t* out_evidence) {
  *out_evidence = (iree_benchmark_loom_launch_evidence_t){
      .host_allocator = host_allocator,
  };
  iree_status_t status = iree_ok_status();
  if (record_count != 0) {
    status = iree_allocator_malloc_array(host_allocator, record_count,
                                         sizeof(*out_evidence->records),
                                         (void**)&out_evidence->records);
    if (iree_status_is_ok(status)) {
      memset(out_evidence->records, 0,
             record_count * sizeof(*out_evidence->records));
      out_evidence->record_count = record_count;
    }
  }
  if (iree_status_is_ok(status) && workload_value_count != 0) {
    status =
        iree_allocator_malloc_array(host_allocator, workload_value_count,
                                    sizeof(*out_evidence->workload_values),
                                    (void**)&out_evidence->workload_values);
    if (iree_status_is_ok(status)) {
      memset(out_evidence->workload_values, 0,
             workload_value_count * sizeof(*out_evidence->workload_values));
      out_evidence->workload_value_count = workload_value_count;
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_benchmark_loom_launch_evidence_deinitialize(out_evidence);
  }
  return status;
}

void iree_benchmark_loom_launch_evidence_deinitialize(
    iree_benchmark_loom_launch_evidence_t* evidence) {
  if (evidence == NULL) {
    return;
  }
  iree_allocator_free(evidence->host_allocator, evidence->workload_values);
  iree_allocator_free(evidence->host_allocator, evidence->records);
  *evidence = (iree_benchmark_loom_launch_evidence_t){0};
}

void iree_benchmark_loom_launch_evidence_capture(
    const loom_run_hal_testbench_actual_provider_t* provider,
    iree_host_size_t case_sample_ordinal,
    iree_host_size_t sequence_step_ordinal, iree_host_size_t record_ordinal,
    iree_host_size_t workload_value_ordinal,
    iree_benchmark_loom_launch_evidence_t* evidence) {
  IREE_ASSERT(record_ordinal < evidence->record_count);
  iree_benchmark_loom_launch_record_t* record =
      &evidence->records[record_ordinal];

  const loom_testbench_invocation_plan_t* invocation = provider->kernel_launch;
  IREE_ASSERT(workload_value_ordinal <= evidence->workload_value_count);
  IREE_ASSERT(invocation->workload_count <=
              evidence->workload_value_count - workload_value_ordinal);
  record->case_sample_ordinal = case_sample_ordinal;
  record->sequence_step_ordinal = sequence_step_ordinal;
  record->entry = provider->invocation_options.function_name;
  iree_benchmark_loom_workload_value_t* workload_values =
      invocation->workload_count == 0
          ? NULL
          : &evidence->workload_values[workload_value_ordinal];
  record->workload_values = workload_values;
  record->workload_value_count = invocation->workload_count;
  record->launch_config = provider->resolved_launch_config;
  IREE_ASSERT(
      iree_any_bit_set(record->launch_config.fields,
                       LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT));
  for (iree_host_size_t i = 0; i < invocation->workload_count; ++i) {
    const loom_value_id_t value_id = invocation->workload_value_ids[i];
    const loom_type_t type =
        loom_module_value_type(invocation->module, value_id);
    IREE_ASSERT(loom_type_is_scalar(type));
    IREE_ASSERT(loom_scalar_type_name(loom_type_element_type(type)) != NULL);
    workload_values[i] = (iree_benchmark_loom_workload_value_t){
        .type = loom_type_element_type(type),
        .value = provider->workload_arguments[i],
    };
  }
}

static iree_status_t iree_benchmark_loom_write_launch_dimension_json(
    uint32_t x, uint32_t y, uint32_t z, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_uint32_field(&object, IREE_SV("x"), x));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_uint32_field(&object, IREE_SV("y"), y));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_uint32_field(&object, IREE_SV("z"), z));
  return loom_json_object_end(&object);
}

static iree_status_t iree_benchmark_loom_write_workload_json(
    const iree_benchmark_loom_launch_record_t* record,
    loom_output_stream_t* stream) {
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
  for (iree_host_size_t i = 0; i < record->workload_value_count; ++i) {
    const iree_benchmark_loom_workload_value_t* workload_value =
        &record->workload_values[i];
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
    loom_json_object_writer_t object;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("type"),
        iree_make_cstring_view(loom_scalar_type_name(workload_value->type))));
    IREE_RETURN_IF_ERROR(loom_json_object_write_int64_field(
        &object, IREE_SV("value"), workload_value->value));
    IREE_RETURN_IF_ERROR(loom_json_object_end(&object));
  }
  return loom_json_array_end(&array);
}

static iree_status_t iree_benchmark_loom_write_launch_config_json(
    const loom_kernel_launch_config_t* config, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  if (config->fields & LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("workgroup_count")));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_launch_dimension_json(
        config->workgroup_count.x, config->workgroup_count.y,
        config->workgroup_count.z, stream));
  }
  if (config->fields & LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_SIZE) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("workgroup_size")));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_launch_dimension_json(
        config->workgroup_size.x, config->workgroup_size.y,
        config->workgroup_size.z, stream));
  }
  if (config->fields & LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_SUBGROUP_SIZE) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("subgroup_size"), config->subgroup_size));
  }
  if (config->fields &
      LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_STORAGE_BYTES) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("workgroup_storage_bytes"),
        config->workgroup_storage_bytes));
  }
  return loom_json_object_end(&object);
}

static iree_status_t iree_benchmark_loom_write_launch_record_json(
    const iree_benchmark_loom_launch_record_t* record,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("case_sample_index"), record->case_sample_ordinal));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("sequence_step_index"), record->sequence_step_ordinal));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("entry"), record->entry));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("workload")));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_workload_json(record, stream));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("launch_config")));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_launch_config_json(
      &record->launch_config, stream));
  return loom_json_object_end(&object);
}

iree_status_t iree_benchmark_loom_write_launch_evidence_json(
    const iree_benchmark_loom_launch_evidence_t* evidence,
    loom_output_stream_t* stream) {
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
  for (iree_host_size_t i = 0; i < evidence->record_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_launch_record_json(
        &evidence->records[i], stream));
  }
  return loom_json_array_end(&array);
}
