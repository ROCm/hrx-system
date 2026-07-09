// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/stage.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

struct id4_pipeline_bundle_t {
  // Reference count for shared bundle ownership.
  iree_atomic_ref_count_t ref_count;
  // Allocator used for bundle storage.
  iree_allocator_t host_allocator;
  // Stage-specific payload storage inside the bundle allocation.
  void* payload;
  // Destroys initialized stage-specific payload storage.
  id4_pipeline_bundle_payload_destroy_fn_t payload_destroy;
  // Retained plan used to prepare this bundle.
  id4_pipeline_plan_t* plan;
  // Optional loaded parameter slabs retained by this bundle.
  id4_pipeline_parameter_slab_set_t* parameter_slabs;
  // Number of retained readiness semaphores.
  iree_host_size_t readiness_count;
  // Retained semaphores that must reach readiness values before use.
  iree_hal_semaphore_t** readiness_semaphores;
  // Payload values paired with retained readiness semaphores.
  uint64_t* readiness_payload_values;
};

static iree_status_t id4_pipeline_validate_options_size(
    iree_host_size_t actual_size, iree_host_size_t expected_size,
    iree_string_view_t options_name) {
  if (actual_size >= expected_size) {
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "%.*s options structure size %" PRIhsz
                          " is smaller than expected %" PRIhsz,
                          (int)options_name.size, options_name.data,
                          actual_size, expected_size);
}

static iree_status_t id4_pipeline_validate_load_options(
    const id4_pipeline_stage_load_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "load options are required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_validate_options_size(
      options->structure_size, sizeof(*options), IREE_SV("load")));
  if (options->next) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "load extension structures are not supported");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_diagnostics_validate_sink(
      options->diagnostics_sink, IREE_SV("load")));
  return iree_ok_status();
}

static iree_status_t id4_pipeline_validate_plan_options(
    const id4_pipeline_stage_plan_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "plan options are required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_validate_options_size(
      options->structure_size, sizeof(*options), IREE_SV("plan")));
  const id4_pipeline_stage_plan_flags_t allowed_flags =
      ID4_PIPELINE_STAGE_PLAN_FLAG_CAPTURE_DIAGNOSTIC_TAPS |
      ID4_PIPELINE_STAGE_PLAN_FLAG_REGION_PER_DISPATCH;
  if (iree_any_bit_set(options->flags, ~allowed_flags)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported plan flags 0x%x", options->flags);
  }
  if (iree_all_bits_set(options->flags,
                        ID4_PIPELINE_STAGE_PLAN_FLAG_CAPTURE_DIAGNOSTIC_TAPS)) {
    if (options->diagnostic_tap_names.count == 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "diagnostic tap capture requires at least one tap name");
    }
    if (!options->diagnostic_tap_names.values) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "diagnostic tap capture requires a tap name list");
    }
    for (iree_host_size_t i = 0; i < options->diagnostic_tap_names.count; ++i) {
      if (iree_string_view_is_empty(options->diagnostic_tap_names.values[i])) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "diagnostic tap capture name %" PRIhsz " is empty", i);
      }
    }
  } else {
    if (options->diagnostic_tap_names.count != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "diagnostic tap names require diagnostic tap capture");
    }
    if (options->diagnostic_tap_names.values) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "diagnostic tap name list requires diagnostic tap capture");
    }
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_diagnostics_validate_sink(
      options->diagnostics_sink, IREE_SV("plan")));
  return iree_ok_status();
}

static iree_status_t id4_pipeline_validate_semaphore_list(
    iree_hal_semaphore_list_t semaphore_list, iree_string_view_t list_name) {
  if (semaphore_list.count == 0) return iree_ok_status();
  if (!semaphore_list.semaphores) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s semaphore array is required",
                            (int)list_name.size, list_name.data);
  }
  if (!semaphore_list.payload_values) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s payload value array is required",
                            (int)list_name.size, list_name.data);
  }
  for (iree_host_size_t i = 0; i < semaphore_list.count; ++i) {
    if (!semaphore_list.semaphores[i]) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "%.*s semaphore %" PRIhsz " is NULL",
                              (int)list_name.size, list_name.data, i);
    }
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_validate_prepare_options(
    const id4_pipeline_stage_prepare_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "prepare options are required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_validate_options_size(
      options->structure_size, sizeof(*options), IREE_SV("prepare")));
  if (options->next) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "prepare extension structures are not supported");
  }
  const id4_pipeline_stage_prepare_flags_t allowed_flags =
      ID4_PIPELINE_STAGE_PREPARE_FLAG_DEFER_PARAMETER_LOADS_TO_ISSUE |
      ID4_PIPELINE_STAGE_PREPARE_FLAG_REUSE_PARAMETER_SLABS;
  if (iree_any_bit_set(options->flags, ~allowed_flags)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported prepare flags 0x%x", options->flags);
  }
  const bool defer_parameter_loads_to_issue = iree_all_bits_set(
      options->flags,
      ID4_PIPELINE_STAGE_PREPARE_FLAG_DEFER_PARAMETER_LOADS_TO_ISSUE);
  const bool reuse_parameter_slabs = iree_all_bits_set(
      options->flags, ID4_PIPELINE_STAGE_PREPARE_FLAG_REUSE_PARAMETER_SLABS);
  if (defer_parameter_loads_to_issue && reuse_parameter_slabs) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "deferred parameter loading cannot reuse parameter slabs");
  }
  if (reuse_parameter_slabs && !options->parameter_slabs) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter slab reuse requires a parameter slab set");
  }
  if (!reuse_parameter_slabs && options->parameter_slabs) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter slabs require explicit parameter slab reuse");
  }
  if (reuse_parameter_slabs && options->parameter_provider) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter slab reuse cannot also load from a parameter provider");
  }
  if (reuse_parameter_slabs && options->wait_semaphore_list.count != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter slab reuse cannot wait on prepare semaphores");
  }
  if (reuse_parameter_slabs && options->signal_semaphore_list.count != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter slab reuse cannot signal prepare readiness");
  }
  if (defer_parameter_loads_to_issue &&
      options->signal_semaphore_list.count != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "deferred parameter loading cannot signal prepare readiness");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_validate_semaphore_list(
      options->wait_semaphore_list, IREE_SV("prepare wait")));
  IREE_RETURN_IF_ERROR(id4_pipeline_validate_semaphore_list(
      options->signal_semaphore_list, IREE_SV("prepare signal")));
  IREE_RETURN_IF_ERROR(id4_pipeline_diagnostics_validate_sink(
      options->diagnostics_sink, IREE_SV("prepare")));
  return iree_ok_status();
}

static iree_status_t id4_pipeline_validate_issue_boundary_binding(
    const id4_pipeline_boundary_tensor_plan_t* boundary_tensor,
    const iree_hal_buffer_binding_t* binding, iree_host_size_t binding_index) {
  if (!binding->buffer) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "issue boundary binding %" PRIhsz " for tensor %.*s has no buffer",
        binding_index, (int)boundary_tensor->layout.name.size,
        boundary_tensor->layout.name.data);
  }
  if (binding->length < boundary_tensor->layout.byte_length) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "issue boundary binding %" PRIhsz " for tensor %.*s has length %" PRIu64
        " but requires %" PRIu64,
        binding_index, (int)boundary_tensor->layout.name.size,
        boundary_tensor->layout.name.data, binding->length,
        boundary_tensor->layout.byte_length);
  }
  const iree_device_size_t alignment =
      boundary_tensor->layout.alignment ? boundary_tensor->layout.alignment : 1;
  if ((binding->offset % alignment) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "issue boundary binding %" PRIhsz " for tensor %.*s has offset %" PRIu64
        " that is not aligned to %" PRIu64,
        binding_index, (int)boundary_tensor->layout.name.size,
        boundary_tensor->layout.name.data, binding->offset, alignment);
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_validate_issue_boundary_bindings(
    const id4_pipeline_plan_t* plan,
    const id4_pipeline_stage_issue_options_t* options) {
  const iree_host_size_t boundary_tensor_count =
      id4_pipeline_plan_boundary_tensor_count(plan);
  if (boundary_tensor_count == 0) {
    if (options->boundary_binding_count != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "issue boundary binding count %" PRIhsz
                              " is present for a plan with no boundary tensors",
                              options->boundary_binding_count);
    }
    if (options->boundary_bindings) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "issue boundary bindings are present for a plan with no boundary "
          "tensors");
    }
    return iree_ok_status();
  }
  if (options->boundary_binding_count != boundary_tensor_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "issue expected %" PRIhsz " boundary bindings but got %" PRIhsz,
        boundary_tensor_count, options->boundary_binding_count);
  }
  if (!options->boundary_bindings) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "issue boundary binding array is required");
  }
  for (iree_host_size_t i = 0; i < boundary_tensor_count; ++i) {
    const id4_pipeline_boundary_tensor_plan_t* boundary_tensor =
        id4_pipeline_plan_boundary_tensor_at(plan, i);
    IREE_RETURN_IF_ERROR(id4_pipeline_validate_issue_boundary_binding(
        boundary_tensor, &options->boundary_bindings[i], i));
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_validate_issue_diagnostic_tap_binding(
    const id4_pipeline_diagnostic_tap_plan_t* diagnostic_tap,
    const iree_hal_buffer_binding_t* binding, iree_host_size_t binding_index) {
  if (!binding->buffer) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "issue diagnostic tap binding %" PRIhsz
                            " for tap %.*s has no buffer",
                            binding_index, (int)diagnostic_tap->name.size,
                            diagnostic_tap->name.data);
  }
  if (binding->length < diagnostic_tap->layout.byte_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "issue diagnostic tap binding %" PRIhsz
                            " for tap %.*s has length "
                            "%" PRIu64 " but requires %" PRIu64,
                            binding_index, (int)diagnostic_tap->name.size,
                            diagnostic_tap->name.data, binding->length,
                            diagnostic_tap->layout.byte_length);
  }
  const iree_device_size_t alignment =
      diagnostic_tap->layout.alignment ? diagnostic_tap->layout.alignment : 1;
  if ((binding->offset % alignment) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "issue diagnostic tap binding %" PRIhsz
        " for tap %.*s has offset %" PRIu64 " that is not aligned to %" PRIu64,
        binding_index, (int)diagnostic_tap->name.size,
        diagnostic_tap->name.data, binding->offset, alignment);
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_validate_issue_diagnostic_tap_bindings(
    const id4_pipeline_plan_t* plan,
    const id4_pipeline_stage_issue_options_t* options) {
  const iree_host_size_t diagnostic_tap_count =
      id4_pipeline_plan_diagnostic_tap_count(plan);
  if (diagnostic_tap_count == 0) {
    if (options->diagnostic_tap_binding_count != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "issue diagnostic tap binding count %" PRIhsz
                              " is present for a plan with no diagnostic taps",
                              options->diagnostic_tap_binding_count);
    }
    if (options->diagnostic_tap_bindings) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "issue diagnostic tap bindings are present for a plan with no "
          "diagnostic taps");
    }
    return iree_ok_status();
  }
  if (options->diagnostic_tap_binding_count != diagnostic_tap_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "issue expected %" PRIhsz " diagnostic tap bindings but got %" PRIhsz,
        diagnostic_tap_count, options->diagnostic_tap_binding_count);
  }
  if (!options->diagnostic_tap_bindings) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "issue diagnostic tap binding array is required");
  }
  for (iree_host_size_t i = 0; i < diagnostic_tap_count; ++i) {
    const id4_pipeline_diagnostic_tap_plan_t* diagnostic_tap =
        id4_pipeline_plan_diagnostic_tap_at(plan, i);
    IREE_RETURN_IF_ERROR(id4_pipeline_validate_issue_diagnostic_tap_binding(
        diagnostic_tap, &options->diagnostic_tap_bindings[i], i));
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_validate_issue_options(
    id4_pipeline_bundle_t* bundle,
    const id4_pipeline_stage_issue_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "issue options are required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_validate_options_size(
      options->structure_size, sizeof(*options), IREE_SV("issue")));
  if (options->next) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "issue extension structures are not supported");
  }
  const id4_pipeline_stage_issue_flags_t allowed_flags =
      ID4_PIPELINE_STAGE_ISSUE_FLAG_WAIT_AFTER_EACH_REGION;
  if (iree_any_bit_set(options->flags, ~allowed_flags)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported issue flags 0x%x", options->flags);
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_validate_semaphore_list(
      options->wait_semaphore_list, IREE_SV("issue wait")));
  IREE_RETURN_IF_ERROR(id4_pipeline_validate_semaphore_list(
      options->signal_semaphore_list, IREE_SV("issue signal")));
  if (options->signal_semaphore_list.count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "issue final signal is required");
  }
  if (options->region_submission_window == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "issue region submission window is required");
  }
  if (options->parameter_load_prefetch_region_distance != 0) {
    id4_pipeline_parameter_slab_set_t* parameter_slabs =
        id4_pipeline_bundle_parameter_slabs(bundle);
    if (!parameter_slabs ||
        !id4_pipeline_parameter_slab_set_has_deferred_load_context(
            parameter_slabs)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "issue parameter load prefetch requires deferred parameter loads");
    }
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_validate_issue_boundary_bindings(
      id4_pipeline_bundle_plan(bundle), options));
  IREE_RETURN_IF_ERROR(id4_pipeline_validate_issue_diagnostic_tap_bindings(
      id4_pipeline_bundle_plan(bundle), options));
  IREE_RETURN_IF_ERROR(id4_pipeline_diagnostics_validate_sink(
      options->diagnostics_sink, IREE_SV("issue")));
  return iree_ok_status();
}

static iree_status_t id4_pipeline_validate_bundle_create_options(
    const id4_pipeline_bundle_create_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "bundle create options are required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_validate_options_size(
      options->structure_size, sizeof(*options), IREE_SV("bundle create")));
  if (options->next) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "bundle create extension structures are not "
                            "supported");
  }
  if (!options->plan) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "bundle plan is required");
  }
  const iree_host_size_t parameter_slab_count =
      id4_pipeline_plan_parameter_slab_count(options->plan);
  if (parameter_slab_count != 0 && !options->parameter_slabs) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "bundle parameter slabs are required by the plan");
  }
  if (parameter_slab_count == 0 && options->parameter_slabs) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "bundle parameter slabs are present for a plan with no slabs");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_validate_semaphore_list(
      options->readiness_semaphore_list, IREE_SV("bundle readiness")));
  if (options->payload_size == 0 && options->payload_destroy) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "bundle payload destructor requires payload storage");
  }
  if (options->payload_alignment != 0 &&
      !iree_host_size_is_power_of_two(options->payload_alignment)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "bundle payload alignment must be a power of two");
  }
  return iree_ok_status();
}

static void id4_pipeline_bundle_release_readiness(
    id4_pipeline_bundle_t* bundle) {
  for (iree_host_size_t i = 0; i < bundle->readiness_count; ++i) {
    iree_hal_semaphore_release(bundle->readiness_semaphores[i]);
  }
  iree_allocator_free(bundle->host_allocator, bundle->readiness_semaphores);
  iree_allocator_free(bundle->host_allocator, bundle->readiness_payload_values);
  bundle->readiness_count = 0;
  bundle->readiness_semaphores = NULL;
  bundle->readiness_payload_values = NULL;
}

static iree_status_t id4_pipeline_bundle_copy_readiness(
    id4_pipeline_bundle_t* bundle,
    iree_hal_semaphore_list_t readiness_semaphore_list) {
  IREE_RETURN_IF_ERROR(id4_pipeline_validate_semaphore_list(
      readiness_semaphore_list, IREE_SV("bundle readiness")));
  if (readiness_semaphore_list.count == 0) return iree_ok_status();

  iree_hal_semaphore_t** semaphores = NULL;
  uint64_t* payload_values = NULL;
  iree_status_t status = iree_allocator_malloc_array(
      bundle->host_allocator, readiness_semaphore_list.count,
      sizeof(semaphores[0]), (void**)&semaphores);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(
        bundle->host_allocator, readiness_semaphore_list.count,
        sizeof(payload_values[0]), (void**)&payload_values);
  }
  iree_host_size_t retained_count = 0;
  for (iree_host_size_t i = 0;
       i < readiness_semaphore_list.count && iree_status_is_ok(status); ++i) {
    semaphores[i] = readiness_semaphore_list.semaphores[i];
    iree_hal_semaphore_retain(semaphores[i]);
    payload_values[i] = readiness_semaphore_list.payload_values[i];
    ++retained_count;
  }
  if (iree_status_is_ok(status)) {
    bundle->readiness_count = readiness_semaphore_list.count;
    bundle->readiness_semaphores = semaphores;
    bundle->readiness_payload_values = payload_values;
  } else {
    for (iree_host_size_t i = 0; i < retained_count; ++i) {
      iree_hal_semaphore_release(semaphores[i]);
    }
    iree_allocator_free(bundle->host_allocator, semaphores);
    iree_allocator_free(bundle->host_allocator, payload_values);
  }
  return status;
}

iree_status_t id4_pipeline_stage_initialize(
    const id4_pipeline_stage_vtable_t* vtable,
    const id4_pipeline_stage_services_t* services,
    id4_pipeline_stage_t* out_stage) {
  IREE_ASSERT_ARGUMENT(vtable);
  IREE_ASSERT_ARGUMENT(services);
  IREE_ASSERT_ARGUMENT(out_stage);
  if (!vtable->destroy || !vtable->load || !vtable->plan || !vtable->prepare ||
      !vtable->issue) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "stage vtable is incomplete");
  }
  if (!services->device_group) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "stage device group is required");
  }
  const iree_host_size_t device_count =
      iree_hal_device_group_device_count(services->device_group);
  if (device_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "stage device group must not be empty");
  }

  memset(out_stage, 0, sizeof(*out_stage));
  iree_atomic_ref_count_init(&out_stage->ref_count);
  out_stage->vtable = vtable;
  out_stage->services = *services;
  iree_hal_device_group_retain(out_stage->services.device_group);
  if (out_stage->services.executable_cache) {
    iree_hal_executable_cache_retain(out_stage->services.executable_cache);
  }
  return iree_ok_status();
}

void id4_pipeline_stage_deinitialize(id4_pipeline_stage_t* stage) {
  if (!stage) return;
  iree_hal_executable_cache_release(stage->services.executable_cache);
  iree_hal_device_group_release(stage->services.device_group);
  memset(&stage->services, 0, sizeof(stage->services));
  stage->vtable = NULL;
}

void id4_pipeline_stage_retain(id4_pipeline_stage_t* stage) {
  if (!stage) return;
  iree_atomic_ref_count_inc(&stage->ref_count);
}

void id4_pipeline_stage_release(id4_pipeline_stage_t* stage) {
  if (stage && iree_atomic_ref_count_dec(&stage->ref_count) == 1) {
    stage->vtable->destroy(stage);
  }
}

const id4_pipeline_stage_services_t* id4_pipeline_stage_services(
    const id4_pipeline_stage_t* stage) {
  return stage ? &stage->services : NULL;
}

iree_status_t id4_pipeline_stage_load(
    id4_pipeline_stage_t* stage,
    const id4_pipeline_stage_load_options_t* options) {
  IREE_ASSERT_ARGUMENT(stage);
  IREE_RETURN_IF_ERROR(id4_pipeline_validate_load_options(options));
  return stage->vtable->load(stage, options);
}

iree_status_t id4_pipeline_stage_plan(
    id4_pipeline_stage_t* stage,
    const id4_pipeline_stage_plan_options_t* options,
    id4_pipeline_plan_t** out_plan) {
  IREE_ASSERT_ARGUMENT(stage);
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = NULL;
  IREE_RETURN_IF_ERROR(id4_pipeline_validate_plan_options(options));
  return stage->vtable->plan(stage, options, out_plan);
}

iree_status_t id4_pipeline_stage_prepare(
    id4_pipeline_stage_t* stage, const id4_pipeline_plan_t* plan,
    const id4_pipeline_stage_prepare_options_t* options,
    id4_pipeline_bundle_t** out_bundle) {
  IREE_ASSERT_ARGUMENT(stage);
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(out_bundle);
  *out_bundle = NULL;
  IREE_RETURN_IF_ERROR(id4_pipeline_validate_prepare_options(options));
  return stage->vtable->prepare(stage, plan, options, out_bundle);
}

iree_status_t id4_pipeline_stage_issue(
    id4_pipeline_stage_t* stage, id4_pipeline_bundle_t* bundle,
    const id4_pipeline_stage_issue_options_t* options) {
  IREE_ASSERT_ARGUMENT(stage);
  IREE_ASSERT_ARGUMENT(bundle);
  IREE_RETURN_IF_ERROR(id4_pipeline_validate_issue_options(bundle, options));
  return stage->vtable->issue(stage, bundle, options);
}

static void id4_pipeline_bundle_destroy(id4_pipeline_bundle_t* bundle) {
  iree_allocator_t host_allocator = bundle->host_allocator;
  if (bundle->payload_destroy) {
    bundle->payload_destroy(bundle, bundle->payload);
  }
  id4_pipeline_bundle_release_readiness(bundle);
  id4_pipeline_parameter_slab_set_release(bundle->parameter_slabs);
  id4_pipeline_plan_release(bundle->plan);
  iree_allocator_free(host_allocator, bundle);
}

iree_status_t id4_pipeline_bundle_create(
    const id4_pipeline_bundle_create_options_t* options,
    iree_allocator_t host_allocator, id4_pipeline_bundle_t** out_bundle) {
  IREE_ASSERT_ARGUMENT(out_bundle);
  *out_bundle = NULL;
  IREE_RETURN_IF_ERROR(id4_pipeline_validate_bundle_create_options(options));

  const iree_host_size_t payload_alignment = options->payload_alignment
                                                 ? options->payload_alignment
                                                 : iree_alignof(void*);
  iree_host_size_t payload_offset = 0;
  iree_host_size_t total_size = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(id4_pipeline_bundle_t), &total_size,
      IREE_STRUCT_FIELD_ALIGNED(options->payload_size, uint8_t,
                                payload_alignment, &payload_offset)));

  id4_pipeline_bundle_t* bundle = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&bundle));
  memset(bundle, 0, total_size);
  iree_atomic_ref_count_init(&bundle->ref_count);
  bundle->host_allocator = host_allocator;
  if (options->payload_size != 0) {
    bundle->payload = (uint8_t*)bundle + payload_offset;
    bundle->payload_destroy = options->payload_destroy;
  }
  bundle->plan = (id4_pipeline_plan_t*)options->plan;
  id4_pipeline_plan_retain(bundle->plan);
  bundle->parameter_slabs = options->parameter_slabs;
  id4_pipeline_parameter_slab_set_retain(bundle->parameter_slabs);
  iree_status_t status = id4_pipeline_bundle_copy_readiness(
      bundle, options->readiness_semaphore_list);
  if (iree_status_is_ok(status)) {
    *out_bundle = bundle;
  } else {
    id4_pipeline_bundle_destroy(bundle);
  }
  return status;
}

void id4_pipeline_bundle_retain(id4_pipeline_bundle_t* bundle) {
  if (!bundle) return;
  iree_atomic_ref_count_inc(&bundle->ref_count);
}

void id4_pipeline_bundle_release(id4_pipeline_bundle_t* bundle) {
  if (bundle && iree_atomic_ref_count_dec(&bundle->ref_count) == 1) {
    id4_pipeline_bundle_destroy(bundle);
  }
}

const id4_pipeline_plan_t* id4_pipeline_bundle_plan(
    const id4_pipeline_bundle_t* bundle) {
  return bundle ? bundle->plan : NULL;
}

id4_pipeline_parameter_slab_set_t* id4_pipeline_bundle_parameter_slabs(
    const id4_pipeline_bundle_t* bundle) {
  return bundle ? bundle->parameter_slabs : NULL;
}

iree_hal_semaphore_list_t id4_pipeline_bundle_readiness_semaphore_list(
    const id4_pipeline_bundle_t* bundle) {
  if (!bundle || bundle->readiness_count == 0) {
    return iree_hal_semaphore_list_empty();
  }
  return (iree_hal_semaphore_list_t){
      // Number of readiness semaphores retained by the bundle.
      .count = bundle->readiness_count,
      // Readiness semaphores retained by the bundle.
      .semaphores = bundle->readiness_semaphores,
      // Payload values paired with the readiness semaphores.
      .payload_values = bundle->readiness_payload_values,
  };
}

static iree_status_t id4_pipeline_bundle_emit_readiness_diagnostic(
    const id4_pipeline_bundle_t* bundle, iree_host_size_t index,
    uint64_t current_payload_value, iree_status_code_t query_status_code,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  char message[256];
  iree_string_view_t stage_name = id4_pipeline_plan_stage_name(bundle->plan);
  const uint64_t target_payload_value = bundle->readiness_payload_values[index];
  iree_string_view_t key = IREE_SV("bundle.readiness.not_ready");
  if (query_status_code == IREE_STATUS_OK) {
    snprintf(message, sizeof(message),
             "bundle readiness semaphore %" PRIhsz
             " has current payload %" PRIu64 " below target %" PRIu64,
             index, current_payload_value, target_payload_value);
  } else {
    key = IREE_SV("bundle.readiness.failure");
    snprintf(message, sizeof(message),
             "bundle readiness semaphore %" PRIhsz
             " failed before target %" PRIu64
             "; current payload query returned %" PRIu64,
             index, target_payload_value, current_payload_value);
  }
  id4_pipeline_diagnostic_event_t event;
  memset(&event, 0, sizeof(event));
  event.kind = ID4_PIPELINE_DIAGNOSTIC_EVENT_KIND_LIFECYCLE;
  event.stage_name = stage_name;
  event.key = key;
  event.message = iree_make_cstring_view(message);
  return id4_pipeline_diagnostics_emit(diagnostics_sink, &event);
}

static iree_status_t id4_pipeline_bundle_check_readiness_semaphores(
    const id4_pipeline_bundle_t* bundle,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < bundle->readiness_count && iree_status_is_ok(status); ++i) {
    iree_hal_semaphore_t* semaphore = bundle->readiness_semaphores[i];
    const uint64_t target_payload_value = bundle->readiness_payload_values[i];
    uint64_t current_payload_value = 0;
    status = iree_hal_semaphore_query(semaphore, &current_payload_value);
    const iree_status_code_t query_status_code = iree_status_code(status);
    if (query_status_code == IREE_STATUS_OK) {
      if (current_payload_value >= target_payload_value) {
        continue;
      }
    }
    iree_status_t emit_status = id4_pipeline_bundle_emit_readiness_diagnostic(
        bundle, i, current_payload_value, query_status_code, diagnostics_sink);
    if (iree_status_is_ok(emit_status)) {
      if (query_status_code == IREE_STATUS_OK) {
        status = iree_status_join(
            status, iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                     "pipeline bundle readiness semaphore "
                                     "%" PRIhsz " is not ready",
                                     i));
      } else {
        status = iree_status_annotate_f(status,
                                        "pipeline bundle readiness semaphore "
                                        "%" PRIhsz " failed",
                                        i);
      }
    } else {
      status = iree_status_join(emit_status, status);
    }
  }
  return status;
}

iree_status_t id4_pipeline_bundle_check_readiness_failures(
    const id4_pipeline_bundle_t* bundle,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  if (!bundle) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pipeline bundle is required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_diagnostics_validate_sink(
      diagnostics_sink, IREE_SV("bundle readiness failure check")));
  iree_status_t status =
      id4_pipeline_bundle_check_readiness_semaphores(bundle, diagnostics_sink);
  if (bundle->parameter_slabs) {
    status = iree_status_join(
        status,
        id4_pipeline_parameter_slab_set_check_load_group_failures(
            bundle->parameter_slabs, id4_pipeline_plan_stage_name(bundle->plan),
            diagnostics_sink));
  }
  return status;
}

void* id4_pipeline_bundle_payload(id4_pipeline_bundle_t* bundle) {
  return bundle ? bundle->payload : NULL;
}

const void* id4_pipeline_bundle_const_payload(
    const id4_pipeline_bundle_t* bundle) {
  return bundle ? bundle->payload : NULL;
}
