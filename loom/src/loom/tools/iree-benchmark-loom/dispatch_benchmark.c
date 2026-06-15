// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-benchmark-loom/dispatch_benchmark.h"

#include <inttypes.h>
#include <stdint.h>
#include <string.h>

#include "iree/vm/api.h"
#include "loom/tooling/execution/hal/benchmark.h"
#include "loom/tooling/execution/hal/invocation.h"
#include "loom/tooling/execution/hal/testbench_actual.h"
#include "loom/tools/iree-benchmark-loom/case_execution.h"
#include "loom/tools/iree-benchmark-loom/output.h"

typedef struct iree_benchmark_loom_hal_input_ring_t {
  // Host allocator used for ring-owned arrays.
  iree_allocator_t host_allocator;
  // Ring-owned invocation plans materialized from check ops.
  loom_run_hal_invocation_plan_t* plans;
  // Borrowed binding-list pointers into |plans| for HAL benchmark setup.
  iree_vm_list_t** binding_lists;
  // Number of entries in |plans| and |binding_lists|.
  iree_host_size_t plan_count;
  // Data/cache summary derived while materializing the ring.
  iree_benchmark_loom_data_cache_summary_t summary;
} iree_benchmark_loom_hal_input_ring_t;

static bool iree_benchmark_loom_hal_invocation_options_equal(
    const loom_run_hal_invocation_options_t* lhs,
    const loom_run_hal_invocation_options_t* rhs) {
  if (!iree_string_view_equal(lhs->function_name, rhs->function_name) ||
      lhs->constant_count != rhs->constant_count) {
    return false;
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(lhs->workgroup_count); ++i) {
    if (lhs->workgroup_count[i] != rhs->workgroup_count[i]) {
      return false;
    }
  }
  for (iree_host_size_t i = 0; i < lhs->constant_count; ++i) {
    if (lhs->constants[i] != rhs->constants[i]) {
      return false;
    }
  }
  return true;
}

static iree_status_t iree_benchmark_loom_hal_input_ring_count_for_sample(
    const iree_benchmark_loom_options_t* options, uint64_t binding_set_bytes,
    iree_host_size_t dispatches_per_batch, iree_host_size_t* out_ring_count) {
  *out_ring_count = 1;
  if (options->input_ring_count > 0) {
    *out_ring_count = options->input_ring_count;
    return iree_ok_status();
  }
  if (options->input_ring_min_bytes == 0 || binding_set_bytes == 0) {
    return iree_ok_status();
  }
  const uint64_t requested_min_bytes = (uint64_t)options->input_ring_min_bytes;
  const uint64_t byte_sized_count =
      requested_min_bytes / binding_set_bytes +
      (requested_min_bytes % binding_set_bytes == 0 ? 0 : 1);
  uint64_t ring_count = byte_sized_count;
  if (ring_count < dispatches_per_batch) {
    ring_count = dispatches_per_batch;
  }
  if (ring_count > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "HAL benchmark input ring count %" PRIu64
                            " exceeds host size limits",
                            ring_count);
  }
  *out_ring_count = (iree_host_size_t)ring_count;
  return iree_ok_status();
}

static iree_status_t iree_benchmark_loom_prepare_hal_invocation_plan_for_sample(
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_benchmark_loom_hal_actual_provider_t* provider,
    const loom_testbench_value_materializer_options_t* materializer_options,
    iree_host_size_t sample_ordinal, iree_allocator_t allocator,
    loom_run_hal_invocation_plan_t* out_plan) {
  loom_run_hal_invocation_options_t invocation_options = {0};
  iree_vm_list_t* bindings = NULL;
  iree_status_t status =
      loom_run_hal_testbench_create_invocation_inputs_for_sample(
          module_plan->module, materializer_options, case_plan,
          provider->execution.actual_invocation, sample_ordinal,
          &provider->execution.invocation_options, allocator,
          &invocation_options, &bindings);
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_invocation_plan_prepare_from_lists(
        &invocation_options, bindings, /*expected_bindings=*/NULL,
        /*max_output_element_count=*/0, out_plan);
  }
  iree_vm_list_release(bindings);
  return status;
}

static iree_status_t iree_benchmark_loom_hal_input_ring_initialize(
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_benchmark_loom_benchmark_policy_t* policy,
    const iree_benchmark_loom_options_t* options,
    const iree_benchmark_loom_hal_actual_provider_t* provider,
    const loom_testbench_value_materializer_options_t* materializer_options,
    iree_host_size_t sample_ordinal, iree_allocator_t allocator,
    iree_benchmark_loom_hal_input_ring_t* out_ring) {
  *out_ring = (iree_benchmark_loom_hal_input_ring_t){
      .host_allocator = allocator,
  };

  loom_run_hal_invocation_plan_t first_plan = {0};
  iree_status_t status =
      iree_benchmark_loom_prepare_hal_invocation_plan_for_sample(
          module_plan, case_plan, provider, materializer_options,
          sample_ordinal, allocator, &first_plan);
  uint64_t first_binding_set_bytes = 0;
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_binding_list_total_byte_length(
        first_plan.bindings, &first_binding_set_bytes);
  }

  iree_host_size_t ring_count = 1;
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_hal_input_ring_count_for_sample(
        options, first_binding_set_bytes, policy->hal_options.timing.batch_size,
        &ring_count);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(allocator, ring_count,
                                         sizeof(*out_ring->plans),
                                         (void**)&out_ring->plans);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(allocator, ring_count,
                                         sizeof(*out_ring->binding_lists),
                                         (void**)&out_ring->binding_lists);
  }
  if (iree_status_is_ok(status)) {
    memset(out_ring->plans, 0, ring_count * sizeof(*out_ring->plans));
    out_ring->plan_count = ring_count;
    out_ring->plans[0] = first_plan;
    first_plan = (loom_run_hal_invocation_plan_t){0};
    out_ring->binding_lists[0] = out_ring->plans[0].bindings;
    out_ring->summary = (iree_benchmark_loom_data_cache_summary_t){
        .populated = true,
        .binding_count = iree_vm_list_size(out_ring->plans[0].bindings),
        .binding_ring_count = ring_count,
        .dispatches_per_batch = policy->hal_options.timing.batch_size,
        .requested_min_ring_bytes = (uint64_t)options->input_ring_min_bytes,
        .binding_set_bytes = first_binding_set_bytes,
        .binding_ring_bytes = first_binding_set_bytes,
    };
  }
  for (iree_host_size_t i = 1; iree_status_is_ok(status) && i < ring_count;
       ++i) {
    status = iree_benchmark_loom_prepare_hal_invocation_plan_for_sample(
        module_plan, case_plan, provider, materializer_options, sample_ordinal,
        allocator, &out_ring->plans[i]);
    if (iree_status_is_ok(status) &&
        !iree_benchmark_loom_hal_invocation_options_equal(
            &out_ring->plans[0].options, &out_ring->plans[i].options)) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "HAL benchmark input ring materialization changed dispatch constants "
          "or geometry for sample %" PRIhsz,
          sample_ordinal);
    }
    uint64_t binding_set_bytes = 0;
    if (iree_status_is_ok(status)) {
      status = loom_run_hal_binding_list_total_byte_length(
          out_ring->plans[i].bindings, &binding_set_bytes);
    }
    if (iree_status_is_ok(status)) {
      if (UINT64_MAX - out_ring->summary.binding_ring_bytes <
          binding_set_bytes) {
        status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "HAL benchmark input ring byte count "
                                  "overflowed uint64");
      } else {
        out_ring->binding_lists[i] = out_ring->plans[i].bindings;
        out_ring->summary.binding_ring_bytes += binding_set_bytes;
      }
    }
  }
  if (!iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < out_ring->plan_count; ++i) {
      loom_run_hal_invocation_plan_deinitialize(&out_ring->plans[i]);
    }
    iree_allocator_free(out_ring->host_allocator, out_ring->binding_lists);
    iree_allocator_free(out_ring->host_allocator, out_ring->plans);
    *out_ring = (iree_benchmark_loom_hal_input_ring_t){0};
  }
  loom_run_hal_invocation_plan_deinitialize(&first_plan);
  return status;
}

static void iree_benchmark_loom_hal_input_ring_deinitialize(
    iree_benchmark_loom_hal_input_ring_t* ring) {
  for (iree_host_size_t i = 0; i < ring->plan_count; ++i) {
    loom_run_hal_invocation_plan_deinitialize(&ring->plans[i]);
  }
  iree_allocator_free(ring->host_allocator, ring->binding_lists);
  iree_allocator_free(ring->host_allocator, ring->plans);
  *ring = (iree_benchmark_loom_hal_input_ring_t){0};
}

typedef struct iree_benchmark_loom_hal_sequence_input_ring_t {
  // Host allocator used for ring-owned arrays.
  iree_allocator_t host_allocator;
  // Ring-owned invocation plans in ring-slot, sequence-step order.
  loom_run_hal_invocation_plan_t* plans;
  // Borrowed plan pointers into |plans| for HAL benchmark setup.
  const loom_run_hal_invocation_plan_t** plan_ptrs;
  // Borrowed prepared candidates in sequence-step order.
  const loom_run_hal_prepared_candidate_t** candidates;
  // Number of logical ring slots in |plans|.
  iree_host_size_t plan_ring_count;
  // Number of actual dispatch invocations per logical ring slot.
  iree_host_size_t sequence_count;
  // Data/cache summary derived while materializing the ring.
  iree_benchmark_loom_data_cache_summary_t summary;
} iree_benchmark_loom_hal_sequence_input_ring_t;

static iree_host_size_t iree_benchmark_loom_hal_sequence_input_ring_plan_count(
    const iree_benchmark_loom_hal_sequence_input_ring_t* ring) {
  return ring->plan_ring_count * ring->sequence_count;
}

static iree_status_t iree_benchmark_loom_hal_sequence_plans_total_byte_length(
    const loom_run_hal_invocation_plan_t* plans, iree_host_size_t plan_count,
    uint64_t* out_byte_length) {
  *out_byte_length = 0;
  for (iree_host_size_t i = 0; i < plan_count; ++i) {
    uint64_t plan_byte_length = 0;
    IREE_RETURN_IF_ERROR(loom_run_hal_binding_list_total_byte_length(
        plans[i].bindings, &plan_byte_length));
    if (UINT64_MAX - *out_byte_length < plan_byte_length) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "HAL benchmark sequence binding byte count "
                              "overflowed uint64");
    }
    *out_byte_length += plan_byte_length;
  }
  return iree_ok_status();
}

static iree_host_size_t iree_benchmark_loom_hal_sequence_binding_count(
    const loom_run_hal_invocation_plan_t* plans, iree_host_size_t plan_count) {
  iree_host_size_t binding_count = 0;
  for (iree_host_size_t i = 0; i < plan_count; ++i) {
    binding_count += iree_vm_list_size(plans[i].bindings);
  }
  return binding_count;
}

static iree_status_t iree_benchmark_loom_prepare_hal_sequence_plans_for_sample(
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_case_plan_t* case_plan,
    const loom_run_hal_testbench_actual_sequence_t* sequence,
    const loom_testbench_value_materializer_options_t* materializer_options,
    iree_host_size_t sample_ordinal, iree_allocator_t allocator,
    loom_run_hal_invocation_plan_t* out_plans) {
  loom_testbench_value_table_t table = {0};
  iree_status_t status = loom_testbench_value_table_initialize(
      module_plan->module, case_plan, allocator, &table);
  if (iree_status_is_ok(status)) {
    status = loom_testbench_materialize_case_sample(
        materializer_options, case_plan, sample_ordinal, &table);
  }
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < sequence->provider_count; ++i) {
    const loom_run_hal_testbench_actual_provider_t* provider =
        &sequence->providers[i];
    loom_run_hal_invocation_options_t invocation_options = {0};
    iree_vm_list_t* bindings = NULL;
    status = loom_run_hal_testbench_create_invocation_inputs_from_table(
        &table, provider->actual_invocation, &provider->invocation_options,
        allocator, &invocation_options, &bindings);
    if (iree_status_is_ok(status)) {
      status = loom_run_hal_invocation_plan_prepare_from_lists(
          &invocation_options, bindings, /*expected_bindings=*/NULL,
          /*max_output_element_count=*/0, &out_plans[i]);
    }
    iree_vm_list_release(bindings);
  }
  loom_testbench_value_table_deinitialize(&table);
  return status;
}

static iree_status_t iree_benchmark_loom_hal_sequence_input_ring_initialize(
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_benchmark_loom_benchmark_policy_t* policy,
    const iree_benchmark_loom_options_t* options,
    const loom_run_hal_testbench_actual_sequence_t* sequence,
    const loom_testbench_value_materializer_options_t* materializer_options,
    iree_host_size_t sample_ordinal, iree_allocator_t allocator,
    iree_benchmark_loom_hal_sequence_input_ring_t* out_ring) {
  *out_ring = (iree_benchmark_loom_hal_sequence_input_ring_t){
      .host_allocator = allocator,
      .sequence_count = sequence->provider_count,
  };

  loom_run_hal_invocation_plan_t* first_plans = NULL;
  iree_host_size_t first_plan_count = 0;
  iree_status_t status =
      iree_allocator_malloc_array(allocator, sequence->provider_count,
                                  sizeof(*first_plans), (void**)&first_plans);
  if (iree_status_is_ok(status)) {
    memset(first_plans, 0, sequence->provider_count * sizeof(*first_plans));
    first_plan_count = sequence->provider_count;
    status = iree_benchmark_loom_prepare_hal_sequence_plans_for_sample(
        module_plan, case_plan, sequence, materializer_options, sample_ordinal,
        allocator, first_plans);
  }
  uint64_t first_binding_set_bytes = 0;
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_hal_sequence_plans_total_byte_length(
        first_plans, sequence->provider_count, &first_binding_set_bytes);
  }

  iree_host_size_t ring_count = 1;
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_hal_input_ring_count_for_sample(
        options, first_binding_set_bytes, policy->hal_options.timing.batch_size,
        &ring_count);
  }
  iree_host_size_t plan_count = 0;
  if (iree_status_is_ok(status) &&
      !iree_host_size_checked_mul(ring_count, sequence->provider_count,
                                  &plan_count)) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "HAL benchmark sequence input ring plan count "
                              "overflowed host size limits");
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(allocator, plan_count,
                                         sizeof(*out_ring->plans),
                                         (void**)&out_ring->plans);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(allocator, plan_count,
                                         sizeof(*out_ring->plan_ptrs),
                                         (void**)&out_ring->plan_ptrs);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(allocator, sequence->provider_count,
                                         sizeof(*out_ring->candidates),
                                         (void**)&out_ring->candidates);
  }
  if (iree_status_is_ok(status)) {
    memset(out_ring->plans, 0, plan_count * sizeof(*out_ring->plans));
    out_ring->plan_ring_count = ring_count;
    for (iree_host_size_t i = 0; i < sequence->provider_count; ++i) {
      out_ring->plans[i] = first_plans[i];
      first_plans[i] = (loom_run_hal_invocation_plan_t){0};
      out_ring->plan_ptrs[i] = &out_ring->plans[i];
      out_ring->candidates[i] = &sequence->providers[i].prepared_candidate;
    }
    iree_host_size_t dispatches_per_batch = 0;
    if (!iree_host_size_checked_mul(policy->hal_options.timing.batch_size,
                                    sequence->provider_count,
                                    &dispatches_per_batch)) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "HAL benchmark sequence dispatch count "
                                "overflowed host size limits");
    } else {
      out_ring->summary = (iree_benchmark_loom_data_cache_summary_t){
          .populated = true,
          .binding_count = iree_benchmark_loom_hal_sequence_binding_count(
              out_ring->plans, sequence->provider_count),
          .binding_ring_count = ring_count,
          .dispatches_per_batch = dispatches_per_batch,
          .requested_min_ring_bytes = (uint64_t)options->input_ring_min_bytes,
          .binding_set_bytes = first_binding_set_bytes,
          .binding_ring_bytes = first_binding_set_bytes,
      };
    }
  }
  for (iree_host_size_t ring_index = 1;
       iree_status_is_ok(status) && ring_index < ring_count; ++ring_index) {
    loom_run_hal_invocation_plan_t* ring_plans =
        &out_ring->plans[ring_index * sequence->provider_count];
    status = iree_benchmark_loom_prepare_hal_sequence_plans_for_sample(
        module_plan, case_plan, sequence, materializer_options, sample_ordinal,
        allocator, ring_plans);
    for (iree_host_size_t step_index = 0;
         iree_status_is_ok(status) && step_index < sequence->provider_count;
         ++step_index) {
      if (!iree_benchmark_loom_hal_invocation_options_equal(
              &out_ring->plans[step_index].options,
              &ring_plans[step_index].options)) {
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "HAL benchmark sequence input ring materialization changed "
            "dispatch constants or geometry for sample %" PRIhsz
            " step %" PRIhsz,
            sample_ordinal, step_index);
      }
    }
    uint64_t binding_set_bytes = 0;
    if (iree_status_is_ok(status)) {
      status = iree_benchmark_loom_hal_sequence_plans_total_byte_length(
          ring_plans, sequence->provider_count, &binding_set_bytes);
    }
    if (iree_status_is_ok(status)) {
      if (UINT64_MAX - out_ring->summary.binding_ring_bytes <
          binding_set_bytes) {
        status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "HAL benchmark sequence input ring byte "
                                  "count overflowed uint64");
      } else {
        out_ring->summary.binding_ring_bytes += binding_set_bytes;
      }
    }
    for (iree_host_size_t step_index = 0;
         iree_status_is_ok(status) && step_index < sequence->provider_count;
         ++step_index) {
      const iree_host_size_t plan_index =
          ring_index * sequence->provider_count + step_index;
      out_ring->plan_ptrs[plan_index] = &out_ring->plans[plan_index];
    }
  }
  if (!iree_status_is_ok(status)) {
    const iree_host_size_t plan_count =
        iree_benchmark_loom_hal_sequence_input_ring_plan_count(out_ring);
    for (iree_host_size_t i = 0; i < plan_count; ++i) {
      loom_run_hal_invocation_plan_deinitialize(&out_ring->plans[i]);
    }
    iree_allocator_free(out_ring->host_allocator, out_ring->candidates);
    iree_allocator_free(out_ring->host_allocator, out_ring->plan_ptrs);
    iree_allocator_free(out_ring->host_allocator, out_ring->plans);
    *out_ring = (iree_benchmark_loom_hal_sequence_input_ring_t){0};
  }
  for (iree_host_size_t i = 0; i < first_plan_count; ++i) {
    loom_run_hal_invocation_plan_deinitialize(&first_plans[i]);
  }
  iree_allocator_free(allocator, first_plans);
  return status;
}

static void iree_benchmark_loom_hal_sequence_input_ring_deinitialize(
    iree_benchmark_loom_hal_sequence_input_ring_t* ring) {
  const iree_host_size_t plan_count =
      iree_benchmark_loom_hal_sequence_input_ring_plan_count(ring);
  for (iree_host_size_t i = 0; i < plan_count; ++i) {
    loom_run_hal_invocation_plan_deinitialize(&ring->plans[i]);
  }
  iree_allocator_free(ring->host_allocator, ring->candidates);
  iree_allocator_free(ring->host_allocator, ring->plan_ptrs);
  iree_allocator_free(ring->host_allocator, ring->plans);
  *ring = (iree_benchmark_loom_hal_sequence_input_ring_t){0};
}

static iree_status_t iree_benchmark_loom_append_profile_artifact_path(
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_candidate_identity_t* candidate,
    iree_hal_device_profiling_data_families_t profile_data_families,
    iree_string_view_t sample_compilation, iree_host_size_t sample_ordinal,
    iree_string_builder_t* artifact_path) {
  const iree_host_size_t initial_size = iree_string_builder_size(artifact_path);
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_append_effective_profile_artifacts_dir(
          run, profile_data_families, artifact_path));
  if (iree_string_builder_size(artifact_path) == initial_size) {
    return iree_ok_status();
  }
  iree_string_view_t artifacts_dir = iree_string_builder_view(artifact_path);
  if (!iree_string_view_ends_with(artifacts_dir, IREE_SV("/"))) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(artifact_path, "/"));
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(artifact_path, run->run_id));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(artifact_path, "_"));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
      artifact_path, candidate->candidate_id));
  if (!iree_string_view_is_empty(sample_compilation)) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(artifact_path, "_"));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_string(artifact_path, sample_compilation));
  }
  return iree_string_builder_append_format(
      artifact_path, "_sample%" PRIhsz ".irpf", sample_ordinal);
}

iree_status_t iree_benchmark_loom_run_hal_benchmark_sample(
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_candidate_identity_t* candidate,
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_benchmark_loom_benchmark_policy_t* policy,
    const iree_benchmark_loom_options_t* options,
    iree_benchmark_loom_hal_actual_provider_t* provider,
    const loom_testbench_value_materializer_options_t* materializer_options,
    iree_host_size_t benchmark_sample_ordinal, iree_allocator_t allocator,
    iree_benchmark_loom_benchmark_result_t* out_result) {
  const iree_host_size_t case_sample_ordinal =
      iree_benchmark_loom_case_sample_from_benchmark_sample(
          benchmark_plan, case_plan, benchmark_sample_ordinal);
  memset(out_result, 0, sizeof(*out_result));
  out_result->sample_compilation = provider->sample_compilation;
  out_result->has_sample_ordinal = true;
  out_result->sample_ordinal = case_sample_ordinal;
  out_result->samples_per_iteration = 1;

  if (provider->execution.compile_report_available) {
    out_result->compile_report_capture = &provider->compile_report_capture;
  }
  out_result->compile_report_artifact_path =
      provider->compile_report_artifact_path;
  out_result->artifact_manifest_path = provider->artifact_manifest_path;
  out_result->target_artifact_path = provider->target_artifact_path;
  out_result->target_listing_path = provider->target_listing_path;
  out_result->hal_executable_path = provider->hal_executable_path;
  if (provider->execution.compile_rejected) {
    iree_benchmark_loom_benchmark_result_set_compile_rejection(provider,
                                                               out_result);
    out_result->has_sample_ordinal = true;
    out_result->sample_ordinal = case_sample_ordinal;
    out_result->samples_per_iteration = 1;
    return iree_ok_status();
  }

  iree_benchmark_loom_hal_input_ring_t input_ring = {0};
  iree_status_t status = iree_benchmark_loom_hal_input_ring_initialize(
      module_plan, case_plan, policy, options, provider, materializer_options,
      case_sample_ordinal, allocator, &input_ring);
  if (iree_status_is_ok(status)) {
    out_result->has_hal_benchmark = true;
    out_result->data_cache = input_ring.summary;
    loom_run_hal_benchmark_options_t hal_options = policy->hal_options;
    iree_string_builder_t profile_artifact_path;
    iree_string_builder_initialize(allocator, &profile_artifact_path);
    status = iree_benchmark_loom_append_profile_artifact_path(
        run, candidate, policy->hal_options.profile_data_families,
        provider->sample_compilation, case_sample_ordinal,
        &profile_artifact_path);
    if (iree_status_is_ok(status) &&
        iree_string_builder_size(&profile_artifact_path) != 0) {
      status = iree_benchmark_loom_create_parent_directory(
          iree_string_builder_view(&profile_artifact_path), allocator);
    }
    if (iree_status_is_ok(status) &&
        iree_string_builder_size(&profile_artifact_path) != 0) {
      status = iree_benchmark_loom_artifact_bundle_record_file(
          provider->context->artifact_bundle,
          IREE_BENCHMARK_LOOM_BUNDLE_FILE_PROFILE,
          iree_string_builder_view(&profile_artifact_path));
    }
    if (iree_status_is_ok(status) &&
        iree_string_builder_size(&profile_artifact_path) != 0) {
      hal_options.profile_artifact_path =
          iree_string_builder_view(&profile_artifact_path);
    }
    if (iree_status_is_ok(status)) {
      status = loom_run_hal_benchmark_dispatch_binding_ring(
          &provider->context->execution.runtime,
          &provider->execution.prepared_candidate, &input_ring.plans[0],
          input_ring.plan_count, input_ring.binding_lists, &hal_options,
          allocator, &out_result->hal_benchmark);
    }
    if (iree_status_is_ok(status)) {
      out_result->data_cache.command_buffer_ring_count =
          out_result->hal_benchmark.command_buffer_ring_count;
    }
    iree_string_builder_deinitialize(&profile_artifact_path);
  }
  if (iree_status_is_ok(status)) {
    out_result->executed = true;
    out_result->passed = true;
  }
  iree_benchmark_loom_hal_input_ring_deinitialize(&input_ring);
  return status;
}

iree_status_t iree_benchmark_loom_run_hal_sequence_benchmark_sample(
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_candidate_identity_t* candidate,
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_benchmark_loom_benchmark_policy_t* policy,
    const iree_benchmark_loom_options_t* options,
    iree_benchmark_loom_hal_context_t* hal_context,
    loom_run_hal_testbench_actual_sequence_t* sequence,
    const loom_testbench_value_materializer_options_t* materializer_options,
    iree_string_view_t sample_compilation,
    iree_host_size_t benchmark_sample_ordinal, iree_allocator_t allocator,
    iree_benchmark_loom_benchmark_result_t* out_result) {
  const iree_host_size_t case_sample_ordinal =
      iree_benchmark_loom_case_sample_from_benchmark_sample(
          benchmark_plan, case_plan, benchmark_sample_ordinal);
  memset(out_result, 0, sizeof(*out_result));
  out_result->sample_compilation = sample_compilation;
  out_result->has_sample_ordinal = true;
  out_result->sample_ordinal = case_sample_ordinal;
  out_result->samples_per_iteration = 1;

  const loom_run_hal_testbench_actual_provider_t* rejected_provider =
      iree_benchmark_loom_hal_actual_sequence_first_rejection(sequence);
  if (rejected_provider != NULL) {
    iree_benchmark_loom_benchmark_result_set_sequence_compile_rejection(
        rejected_provider, sample_compilation, out_result);
    out_result->has_sample_ordinal = true;
    out_result->sample_ordinal = case_sample_ordinal;
    out_result->samples_per_iteration = 1;
    return iree_ok_status();
  }

  iree_benchmark_loom_hal_sequence_input_ring_t input_ring = {0};
  iree_status_t status = iree_benchmark_loom_hal_sequence_input_ring_initialize(
      module_plan, case_plan, policy, options, sequence, materializer_options,
      case_sample_ordinal, allocator, &input_ring);
  if (iree_status_is_ok(status)) {
    out_result->has_hal_benchmark = true;
    out_result->data_cache = input_ring.summary;
    loom_run_hal_benchmark_options_t hal_options = policy->hal_options;
    iree_string_builder_t profile_artifact_path;
    iree_string_builder_initialize(allocator, &profile_artifact_path);
    status = iree_benchmark_loom_append_profile_artifact_path(
        run, candidate, policy->hal_options.profile_data_families,
        sample_compilation, case_sample_ordinal, &profile_artifact_path);
    if (iree_status_is_ok(status) &&
        iree_string_builder_size(&profile_artifact_path) != 0) {
      status = iree_benchmark_loom_create_parent_directory(
          iree_string_builder_view(&profile_artifact_path), allocator);
    }
    if (iree_status_is_ok(status) &&
        iree_string_builder_size(&profile_artifact_path) != 0) {
      status = iree_benchmark_loom_artifact_bundle_record_file(
          hal_context->artifact_bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_PROFILE,
          iree_string_builder_view(&profile_artifact_path));
    }
    if (iree_status_is_ok(status) &&
        iree_string_builder_size(&profile_artifact_path) != 0) {
      hal_options.profile_artifact_path =
          iree_string_builder_view(&profile_artifact_path);
    }
    if (iree_status_is_ok(status)) {
      status = loom_run_hal_benchmark_dispatch_sequence_plan_ring(
          &hal_context->execution.runtime, sequence->provider_count,
          input_ring.candidates, input_ring.plan_ring_count,
          input_ring.plan_ptrs, &hal_options, allocator,
          &out_result->hal_benchmark);
    }
    if (iree_status_is_ok(status)) {
      out_result->data_cache.command_buffer_ring_count =
          out_result->hal_benchmark.command_buffer_ring_count;
    }
    iree_string_builder_deinitialize(&profile_artifact_path);
  }
  if (iree_status_is_ok(status)) {
    out_result->executed = true;
    out_result->passed = true;
  }
  iree_benchmark_loom_hal_sequence_input_ring_deinitialize(&input_ring);
  return status;
}
