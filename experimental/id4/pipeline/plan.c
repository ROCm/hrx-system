// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/plan.h"

#include <stdint.h>
#include <string.h>

struct id4_pipeline_plan_t {
  // Reference count for shared plan ownership.
  iree_atomic_ref_count_t ref_count;
  // Allocator used for plan storage.
  iree_allocator_t host_allocator;
  // Stage name owned by this plan.
  iree_string_view_t stage_name;
  // Retained device group used for placement resolution.
  iree_hal_device_group_t* device_group;
  // Number of planned device placements.
  iree_host_size_t placement_count;
  // Planned device placements owned by this plan.
  id4_pipeline_device_placement_t* placements;
  // Number of planned parameter slabs.
  iree_host_size_t parameter_slab_count;
  // Planned parameter slabs owned by this plan.
  id4_pipeline_parameter_slab_plan_t* parameter_slabs;
  // Number of planned non-parameter memory slabs.
  iree_host_size_t memory_slab_count;
  // Planned non-parameter memory slabs owned by this plan.
  id4_pipeline_memory_slab_plan_t* memory_slabs;
  // Number of planned kernel specializations.
  iree_host_size_t kernel_count;
  // Planned kernel specializations owned by this plan.
  id4_pipeline_kernel_plan_t* kernels;
  // Number of planned reusable regions.
  iree_host_size_t region_count;
  // Planned reusable regions owned by this plan.
  id4_pipeline_region_plan_t* regions;
  // Number of planned diagnostics taps.
  iree_host_size_t diagnostic_tap_count;
  // Planned diagnostics taps owned by this plan.
  id4_pipeline_diagnostic_tap_plan_t* diagnostic_taps;
};

static iree_status_t id4_pipeline_string_clone(iree_string_view_t source,
                                               iree_allocator_t host_allocator,
                                               iree_string_view_t* out_target) {
  IREE_ASSERT_ARGUMENT(out_target);
  *out_target = iree_string_view_empty();
  if (iree_string_view_is_empty(source)) return iree_ok_status();
  if (source.size == IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "string is too large to clone");
  }
  char* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, source.size + 1, sizeof(storage[0]), (void**)&storage));
  memcpy(storage, source.data, source.size);
  storage[source.size] = 0;
  *out_target = iree_make_string_view(storage, source.size);
  return iree_ok_status();
}

static void id4_pipeline_string_release(iree_string_view_t value,
                                        iree_allocator_t host_allocator) {
  if (value.data) {
    iree_allocator_free(host_allocator, (void*)value.data);
  }
}

static uint32_t id4_pipeline_plan_calculate_topology_fingerprint(
    const id4_pipeline_plan_t* plan) {
  uint32_t value = 2166136261u;
  value = (value ^
           (uint32_t)iree_hal_device_group_device_count(plan->device_group)) *
          16777619u;
  for (iree_host_size_t i = 0; i < plan->placement_count; ++i) {
    const id4_pipeline_device_placement_t* placement = &plan->placements[i];
    value = (value ^ (uint32_t)placement->device_index) * 16777619u;
    value = (value ^ (uint32_t)placement->queue_affinity) * 16777619u;
  }
  return value;
}

static void id4_pipeline_plan_destroy(id4_pipeline_plan_t* plan) {
  iree_allocator_t host_allocator = plan->host_allocator;
  for (iree_host_size_t i = 0; i < plan->diagnostic_tap_count; ++i) {
    id4_pipeline_diagnostic_tap_plan_t* tap = &plan->diagnostic_taps[i];
    id4_pipeline_string_release(tap->target_name, host_allocator);
    id4_pipeline_string_release(tap->name, host_allocator);
  }
  iree_allocator_free(host_allocator, plan->diagnostic_taps);
  for (iree_host_size_t i = 0; i < plan->region_count; ++i) {
    id4_pipeline_string_release(plan->regions[i].name, host_allocator);
  }
  iree_allocator_free(host_allocator, plan->regions);
  for (iree_host_size_t i = 0; i < plan->kernel_count; ++i) {
    id4_pipeline_kernel_plan_t* kernel = &plan->kernels[i];
    id4_pipeline_plan_config_binding_t* config_bindings =
        (id4_pipeline_plan_config_binding_t*)kernel->config_bindings;
    for (iree_host_size_t j = 0; j < kernel->config_binding_count; ++j) {
      id4_pipeline_string_release(config_bindings[j].value, host_allocator);
      id4_pipeline_string_release(config_bindings[j].key, host_allocator);
    }
    iree_allocator_free(host_allocator, config_bindings);
    id4_pipeline_string_release(kernel->function_name, host_allocator);
    id4_pipeline_string_release(kernel->module_path, host_allocator);
    id4_pipeline_string_release(kernel->specialization_key, host_allocator);
  }
  iree_allocator_free(host_allocator, plan->kernels);
  for (iree_host_size_t i = 0; i < plan->memory_slab_count; ++i) {
    id4_pipeline_string_release(plan->memory_slabs[i].name, host_allocator);
  }
  iree_allocator_free(host_allocator, plan->memory_slabs);
  for (iree_host_size_t i = 0; i < plan->parameter_slab_count; ++i) {
    id4_pipeline_parameter_slab_plan_t* slab = &plan->parameter_slabs[i];
    id4_pipeline_parameter_request_t* requests =
        (id4_pipeline_parameter_request_t*)slab->requests;
    for (iree_host_size_t j = 0; j < slab->request_count; ++j) {
      id4_pipeline_string_release(requests[j].key, host_allocator);
    }
    iree_allocator_free(host_allocator, requests);
    id4_pipeline_string_release(slab->scope, host_allocator);
  }
  iree_allocator_free(host_allocator, plan->parameter_slabs);
  for (iree_host_size_t i = 0; i < plan->placement_count; ++i) {
    id4_pipeline_string_release(plan->placements[i].role, host_allocator);
  }
  iree_allocator_free(host_allocator, plan->placements);
  iree_hal_device_group_release(plan->device_group);
  id4_pipeline_string_release(plan->stage_name, host_allocator);
  iree_allocator_free(host_allocator, plan);
}

static iree_status_t id4_pipeline_plan_copy_placements(
    id4_pipeline_plan_t* plan,
    const id4_pipeline_plan_create_options_t* options) {
  if (options->placement_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "at least one placement is required");
  }
  if (!options->placements) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "placement array is required");
  }
  plan->placement_count = options->placement_count;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      plan->host_allocator, plan->placement_count, sizeof(plan->placements[0]),
      (void**)&plan->placements));
  memset(plan->placements, 0,
         plan->placement_count * sizeof(plan->placements[0]));
  for (iree_host_size_t i = 0; i < options->placement_count; ++i) {
    const id4_pipeline_device_placement_t* source = &options->placements[i];
    if (source->device_index >=
        iree_hal_device_group_device_count(options->device_group)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "placement %" PRIhsz " device index %" PRIhsz
          " outside device group count %" PRIhsz,
          i, source->device_index,
          iree_hal_device_group_device_count(options->device_group));
    }
    plan->placements[i].device_index = source->device_index;
    plan->placements[i].queue_affinity = source->queue_affinity;
    IREE_RETURN_IF_ERROR(id4_pipeline_string_clone(
        source->role, plan->host_allocator, &plan->placements[i].role));
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_plan_copy_parameter_slabs(
    id4_pipeline_plan_t* plan,
    const id4_pipeline_plan_create_options_t* options) {
  plan->parameter_slab_count = options->parameter_slab_count;
  if (plan->parameter_slab_count == 0) return iree_ok_status();
  if (!options->parameter_slabs) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter slab array is required");
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      plan->host_allocator, plan->parameter_slab_count,
      sizeof(plan->parameter_slabs[0]), (void**)&plan->parameter_slabs));
  memset(plan->parameter_slabs, 0,
         plan->parameter_slab_count * sizeof(plan->parameter_slabs[0]));
  for (iree_host_size_t i = 0; i < plan->parameter_slab_count; ++i) {
    const id4_pipeline_parameter_slab_plan_t* source =
        &options->parameter_slabs[i];
    IREE_RETURN_IF_ERROR(
        id4_pipeline_parameter_slab_validate(source, plan->placement_count));
    id4_pipeline_parameter_slab_plan_t* target = &plan->parameter_slabs[i];
    target->placement_id = source->placement_id;
    target->target_params = source->target_params;
    target->byte_length = source->byte_length;
    target->alignment = source->alignment;
    target->request_count = source->request_count;
    IREE_RETURN_IF_ERROR(id4_pipeline_string_clone(
        source->scope, plan->host_allocator, &target->scope));
    if (target->request_count == 0) continue;
    id4_pipeline_parameter_request_t* requests = NULL;
    IREE_RETURN_IF_ERROR(
        iree_allocator_malloc_array(plan->host_allocator, target->request_count,
                                    sizeof(requests[0]), (void**)&requests));
    memset(requests, 0, target->request_count * sizeof(requests[0]));
    target->requests = requests;
    for (iree_host_size_t j = 0; j < target->request_count; ++j) {
      requests[j].span = source->requests[j].span;
      IREE_RETURN_IF_ERROR(id4_pipeline_string_clone(
          source->requests[j].key, plan->host_allocator, &requests[j].key));
    }
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_plan_validate_placement_id(
    const id4_pipeline_plan_t* plan, uint32_t placement_id,
    iree_string_view_t descriptor_name) {
  if ((iree_host_size_t)placement_id < plan->placement_count) {
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_OUT_OF_RANGE,
      "%.*s placement id %u exceeds placement count %" PRIhsz,
      (int)descriptor_name.size, descriptor_name.data, placement_id,
      plan->placement_count);
}

static iree_status_t id4_pipeline_plan_copy_memory_slabs(
    id4_pipeline_plan_t* plan,
    const id4_pipeline_plan_create_options_t* options) {
  plan->memory_slab_count = options->memory_slab_count;
  if (plan->memory_slab_count == 0) return iree_ok_status();
  if (!options->memory_slabs) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "memory slab array is required");
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      plan->host_allocator, plan->memory_slab_count,
      sizeof(plan->memory_slabs[0]), (void**)&plan->memory_slabs));
  memset(plan->memory_slabs, 0,
         plan->memory_slab_count * sizeof(plan->memory_slabs[0]));
  for (iree_host_size_t i = 0; i < plan->memory_slab_count; ++i) {
    const id4_pipeline_memory_slab_plan_t* source = &options->memory_slabs[i];
    if (iree_string_view_is_empty(source->name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "memory slab %" PRIhsz " name is required", i);
    }
    IREE_RETURN_IF_ERROR(id4_pipeline_plan_validate_placement_id(
        plan, source->placement_id, IREE_SV("memory slab")));
    if (source->byte_length == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "memory slab %.*s byte length is zero",
                              (int)source->name.size, source->name.data);
    }
    if (source->alignment != 0 &&
        !iree_device_size_is_power_of_two(source->alignment)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "memory slab %.*s alignment must be a power of "
                              "two",
                              (int)source->name.size, source->name.data);
    }
    if (source->high_water_mark > source->byte_length) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "memory slab %.*s high water mark exceeds byte "
                              "length",
                              (int)source->name.size, source->name.data);
    }
    id4_pipeline_memory_slab_plan_t* target = &plan->memory_slabs[i];
    target->placement_id = source->placement_id;
    target->binding_slot = source->binding_slot;
    target->params = source->params;
    target->byte_length = source->byte_length;
    target->alignment = source->alignment;
    target->high_water_mark = source->high_water_mark;
    IREE_RETURN_IF_ERROR(id4_pipeline_string_clone(
        source->name, plan->host_allocator, &target->name));
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_plan_copy_kernel_config_bindings(
    id4_pipeline_plan_t* plan, const id4_pipeline_kernel_plan_t* source,
    id4_pipeline_kernel_plan_t* target) {
  target->config_binding_count = source->config_binding_count;
  if (target->config_binding_count == 0) return iree_ok_status();
  if (!source->config_bindings) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel config binding array is required");
  }
  id4_pipeline_plan_config_binding_t* config_bindings = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      plan->host_allocator, target->config_binding_count,
      sizeof(config_bindings[0]), (void**)&config_bindings));
  memset(config_bindings, 0,
         target->config_binding_count * sizeof(config_bindings[0]));
  target->config_bindings = config_bindings;
  for (iree_host_size_t i = 0; i < target->config_binding_count; ++i) {
    const id4_pipeline_plan_config_binding_t* source_binding =
        &source->config_bindings[i];
    if (iree_string_view_is_empty(source_binding->key)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "kernel config binding %" PRIhsz " key is required", i);
    }
    if (iree_string_view_is_empty(source_binding->value)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "kernel config binding %.*s value is required",
                              (int)source_binding->key.size,
                              source_binding->key.data);
    }
    IREE_RETURN_IF_ERROR(id4_pipeline_string_clone(
        source_binding->key, plan->host_allocator, &config_bindings[i].key));
    IREE_RETURN_IF_ERROR(id4_pipeline_string_clone(source_binding->value,
                                                   plan->host_allocator,
                                                   &config_bindings[i].value));
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_plan_copy_kernels(
    id4_pipeline_plan_t* plan,
    const id4_pipeline_plan_create_options_t* options) {
  plan->kernel_count = options->kernel_count;
  if (plan->kernel_count == 0) return iree_ok_status();
  if (!options->kernels) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel array is required");
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      plan->host_allocator, plan->kernel_count, sizeof(plan->kernels[0]),
      (void**)&plan->kernels));
  memset(plan->kernels, 0, plan->kernel_count * sizeof(plan->kernels[0]));
  for (iree_host_size_t i = 0; i < plan->kernel_count; ++i) {
    const id4_pipeline_kernel_plan_t* source = &options->kernels[i];
    if (iree_string_view_is_empty(source->specialization_key)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "kernel %" PRIhsz " specialization key is required", i);
    }
    if (iree_string_view_is_empty(source->module_path) ||
        iree_string_view_is_empty(source->function_name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "kernel %.*s identity fields are required",
                              (int)source->specialization_key.size,
                              source->specialization_key.data);
    }
    IREE_RETURN_IF_ERROR(id4_pipeline_plan_validate_placement_id(
        plan, source->placement_id, IREE_SV("kernel")));
    id4_pipeline_kernel_plan_t* target = &plan->kernels[i];
    target->placement_id = source->placement_id;
    IREE_RETURN_IF_ERROR(id4_pipeline_string_clone(
        source->specialization_key, plan->host_allocator,
        &target->specialization_key));
    IREE_RETURN_IF_ERROR(id4_pipeline_string_clone(
        source->module_path, plan->host_allocator, &target->module_path));
    IREE_RETURN_IF_ERROR(id4_pipeline_string_clone(
        source->function_name, plan->host_allocator, &target->function_name));
    IREE_RETURN_IF_ERROR(
        id4_pipeline_plan_copy_kernel_config_bindings(plan, source, target));
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_plan_copy_regions(
    id4_pipeline_plan_t* plan,
    const id4_pipeline_plan_create_options_t* options) {
  plan->region_count = options->region_count;
  if (plan->region_count == 0) return iree_ok_status();
  if (!options->regions) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "region array is required");
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      plan->host_allocator, plan->region_count, sizeof(plan->regions[0]),
      (void**)&plan->regions));
  memset(plan->regions, 0, plan->region_count * sizeof(plan->regions[0]));
  for (iree_host_size_t i = 0; i < plan->region_count; ++i) {
    const id4_pipeline_region_plan_t* source = &options->regions[i];
    if (iree_string_view_is_empty(source->name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "region %" PRIhsz " name is required", i);
    }
    IREE_RETURN_IF_ERROR(id4_pipeline_plan_validate_placement_id(
        plan, source->placement_id, IREE_SV("region")));
    if (source->binding_capacity == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "region %.*s binding capacity is zero",
                              (int)source->name.size, source->name.data);
    }
    if (source->local_binding_slot >= source->binding_capacity) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "region %.*s local binding slot %u exceeds "
                              "binding capacity %" PRIhsz,
                              (int)source->name.size, source->name.data,
                              source->local_binding_slot,
                              source->binding_capacity);
    }
    id4_pipeline_region_plan_t* target = &plan->regions[i];
    target->placement_id = source->placement_id;
    target->binding_capacity = source->binding_capacity;
    target->local_binding_slot = source->local_binding_slot;
    target->statistics = source->statistics;
    IREE_RETURN_IF_ERROR(id4_pipeline_string_clone(
        source->name, plan->host_allocator, &target->name));
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_plan_copy_diagnostic_taps(
    id4_pipeline_plan_t* plan,
    const id4_pipeline_plan_create_options_t* options) {
  plan->diagnostic_tap_count = options->diagnostic_tap_count;
  if (plan->diagnostic_tap_count == 0) return iree_ok_status();
  if (!options->diagnostic_taps) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "diagnostic tap array is required");
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      plan->host_allocator, plan->diagnostic_tap_count,
      sizeof(plan->diagnostic_taps[0]), (void**)&plan->diagnostic_taps));
  memset(plan->diagnostic_taps, 0,
         plan->diagnostic_tap_count * sizeof(plan->diagnostic_taps[0]));
  for (iree_host_size_t i = 0; i < plan->diagnostic_tap_count; ++i) {
    const id4_pipeline_diagnostic_tap_plan_t* source =
        &options->diagnostic_taps[i];
    if (iree_string_view_is_empty(source->name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "diagnostic tap %" PRIhsz " name is required", i);
    }
    if (iree_string_view_is_empty(source->target_name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "diagnostic tap %.*s target name is required",
                              (int)source->name.size, source->name.data);
    }
    if ((iree_host_size_t)source->region_id >= plan->region_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "diagnostic tap %.*s region id %u exceeds "
                              "region count %" PRIhsz,
                              (int)source->name.size, source->name.data,
                              source->region_id, plan->region_count);
    }
    const id4_pipeline_region_plan_t* region =
        &plan->regions[source->region_id];
    if (source->after_operation_ordinal >= region->statistics.operation_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "diagnostic tap %.*s operation ordinal %" PRIhsz
                              " exceeds region operation count %" PRIhsz,
                              (int)source->name.size, source->name.data,
                              source->after_operation_ordinal,
                              region->statistics.operation_count);
    }
    id4_pipeline_diagnostic_tap_plan_t* target = &plan->diagnostic_taps[i];
    target->region_id = source->region_id;
    target->after_operation_ordinal = source->after_operation_ordinal;
    IREE_RETURN_IF_ERROR(id4_pipeline_string_clone(
        source->name, plan->host_allocator, &target->name));
    IREE_RETURN_IF_ERROR(id4_pipeline_string_clone(
        source->target_name, plan->host_allocator, &target->target_name));
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_plan_emit_parameter_slab_diagnostics(
    const id4_pipeline_plan_t* plan,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  for (iree_host_size_t i = 0; i < plan->parameter_slab_count; ++i) {
    const id4_pipeline_parameter_slab_plan_t* slab = &plan->parameter_slabs[i];
    const id4_pipeline_device_placement_t* placement =
        id4_pipeline_plan_placement_at(plan, slab->placement_id);
    if (!placement) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "parameter slab %" PRIhsz
                              " references missing placement %u",
                              i, slab->placement_id);
    }
    id4_pipeline_parameter_slab_diagnostic_t parameter_slab = {
        // Plan-local slab index.
        .slab_index = i,
        // This event describes the slab instead of one request.
        .request_index = IREE_HOST_SIZE_MAX,
        // Provider scope used by the slab.
        .scope = slab->scope,
        // No request key is associated with this slab-level event.
        .parameter_key = iree_string_view_empty(),
        // No request source offset is associated with this slab-level event.
        .parameter_offset = 0,
        // No request target offset is associated with this slab-level event.
        .buffer_offset = 0,
        // No request byte length is associated with this slab-level event.
        .length = 0,
        // Placement selected for this slab.
        .placement_id = slab->placement_id,
        // Device index selected by the placement.
        .device_index = placement->device_index,
        // Queue affinity selected by the placement.
        .queue_affinity = placement->queue_affinity,
        // Total slab byte length.
        .slab_byte_length = slab->byte_length,
        // Required slab base alignment.
        .slab_alignment = slab->alignment,
        // Number of requests in the slab.
        .request_count = slab->request_count,
    };
    id4_pipeline_diagnostic_event_t event = {
        // Event kind for parameter slab diagnostics.
        .kind = ID4_PIPELINE_DIAGNOSTIC_EVENT_KIND_PARAMETER_SLAB,
        // Stage name copied into the plan.
        .stage_name = plan->stage_name,
        // Stable key for planned slab records.
        .key = IREE_SV("parameter_slab.plan"),
        // Short event summary.
        .message = IREE_SV("planned parameter slab"),
        // Structured slab payload.
        .parameter_slab = &parameter_slab,
    };
    IREE_RETURN_IF_ERROR(
        id4_pipeline_diagnostics_emit(diagnostics_sink, &event));
  }
  return iree_ok_status();
}

iree_status_t id4_pipeline_plan_create(
    const id4_pipeline_plan_create_options_t* options,
    iree_allocator_t host_allocator, id4_pipeline_plan_t** out_plan) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = NULL;

  if (options->structure_size < sizeof(*options)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "plan options structure size %" PRIhsz
                            " is smaller than expected %" PRIhsz,
                            options->structure_size, sizeof(*options));
  }
  if (options->next) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "plan extension structures are not supported");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_diagnostics_validate_sink(
      options->diagnostics_sink, IREE_SV("plan create")));
  if (!options->device_group) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "device group is required");
  }
  const iree_host_size_t device_count =
      iree_hal_device_group_device_count(options->device_group);
  if (device_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "device group must not be empty");
  }
  id4_pipeline_plan_t* plan = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*plan), (void**)&plan);
  if (iree_status_is_ok(status)) {
    memset(plan, 0, sizeof(*plan));
    iree_atomic_ref_count_init(&plan->ref_count);
    plan->host_allocator = host_allocator;
    plan->device_group = options->device_group;
    iree_hal_device_group_retain(plan->device_group);
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_string_clone(options->stage_name, host_allocator,
                                       &plan->stage_name);
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_plan_copy_placements(plan, options);
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_plan_copy_parameter_slabs(plan, options);
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_plan_copy_memory_slabs(plan, options);
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_plan_copy_kernels(plan, options);
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_plan_copy_regions(plan, options);
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_plan_copy_diagnostic_taps(plan, options);
  }
  if (iree_status_is_ok(status)) {
    id4_pipeline_diagnostic_event_t event = {
        // Event kind for plan creation.
        .kind = ID4_PIPELINE_DIAGNOSTIC_EVENT_KIND_PLAN,
        // Stage name copied into the plan.
        .stage_name = plan->stage_name,
        // Stable key for plan creation events.
        .key = IREE_SV("plan.create"),
        // Short creation summary.
        .message = IREE_SV("created pipeline plan"),
    };
    status = id4_pipeline_diagnostics_emit(options->diagnostics_sink, &event);
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_plan_emit_parameter_slab_diagnostics(
        plan, options->diagnostics_sink);
  }
  if (iree_status_is_ok(status)) {
    *out_plan = plan;
  } else if (plan) {
    id4_pipeline_plan_destroy(plan);
  }
  return status;
}

void id4_pipeline_plan_retain(id4_pipeline_plan_t* plan) {
  if (!plan) return;
  iree_atomic_ref_count_inc(&plan->ref_count);
}

void id4_pipeline_plan_release(id4_pipeline_plan_t* plan) {
  if (plan && iree_atomic_ref_count_dec(&plan->ref_count) == 1) {
    id4_pipeline_plan_destroy(plan);
  }
}

iree_string_view_t id4_pipeline_plan_stage_name(
    const id4_pipeline_plan_t* plan) {
  return plan ? plan->stage_name : iree_string_view_empty();
}

iree_hal_device_group_t* id4_pipeline_plan_device_group(
    const id4_pipeline_plan_t* plan) {
  return plan ? plan->device_group : NULL;
}

iree_host_size_t id4_pipeline_plan_placement_count(
    const id4_pipeline_plan_t* plan) {
  return plan ? plan->placement_count : 0;
}

const id4_pipeline_device_placement_t* id4_pipeline_plan_placement_at(
    const id4_pipeline_plan_t* plan, iree_host_size_t index) {
  if (!plan || index >= plan->placement_count) return NULL;
  return &plan->placements[index];
}

iree_host_size_t id4_pipeline_plan_parameter_slab_count(
    const id4_pipeline_plan_t* plan) {
  return plan ? plan->parameter_slab_count : 0;
}

const id4_pipeline_parameter_slab_plan_t* id4_pipeline_plan_parameter_slab_at(
    const id4_pipeline_plan_t* plan, iree_host_size_t index) {
  if (!plan || index >= plan->parameter_slab_count) return NULL;
  return &plan->parameter_slabs[index];
}

iree_host_size_t id4_pipeline_plan_memory_slab_count(
    const id4_pipeline_plan_t* plan) {
  return plan ? plan->memory_slab_count : 0;
}

const id4_pipeline_memory_slab_plan_t* id4_pipeline_plan_memory_slab_at(
    const id4_pipeline_plan_t* plan, iree_host_size_t index) {
  if (!plan || index >= plan->memory_slab_count) return NULL;
  return &plan->memory_slabs[index];
}

iree_host_size_t id4_pipeline_plan_kernel_count(
    const id4_pipeline_plan_t* plan) {
  return plan ? plan->kernel_count : 0;
}

const id4_pipeline_kernel_plan_t* id4_pipeline_plan_kernel_at(
    const id4_pipeline_plan_t* plan, iree_host_size_t index) {
  if (!plan || index >= plan->kernel_count) return NULL;
  return &plan->kernels[index];
}

iree_host_size_t id4_pipeline_plan_region_count(
    const id4_pipeline_plan_t* plan) {
  return plan ? plan->region_count : 0;
}

const id4_pipeline_region_plan_t* id4_pipeline_plan_region_at(
    const id4_pipeline_plan_t* plan, iree_host_size_t index) {
  if (!plan || index >= plan->region_count) return NULL;
  return &plan->regions[index];
}

iree_host_size_t id4_pipeline_plan_diagnostic_tap_count(
    const id4_pipeline_plan_t* plan) {
  return plan ? plan->diagnostic_tap_count : 0;
}

const id4_pipeline_diagnostic_tap_plan_t* id4_pipeline_plan_diagnostic_tap_at(
    const id4_pipeline_plan_t* plan, iree_host_size_t index) {
  if (!plan || index >= plan->diagnostic_tap_count) return NULL;
  return &plan->diagnostic_taps[index];
}

iree_status_t id4_pipeline_plan_load_parameter_slabs(
    const id4_pipeline_plan_t* plan, iree_io_parameter_provider_t* provider,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink,
    iree_allocator_t host_allocator,
    id4_pipeline_parameter_slab_set_t** out_slab_set) {
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(provider);
  IREE_ASSERT_ARGUMENT(out_slab_set);
  *out_slab_set = NULL;
  IREE_RETURN_IF_ERROR(id4_pipeline_diagnostics_validate_sink(
      diagnostics_sink, IREE_SV("parameter slab load")));

  id4_pipeline_parameter_slab_load_t* loads = NULL;
  if (plan->parameter_slab_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_allocator_malloc_array(host_allocator, plan->parameter_slab_count,
                                    sizeof(loads[0]), (void**)&loads));
  }

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < plan->parameter_slab_count && iree_status_is_ok(status); ++i) {
    const id4_pipeline_parameter_slab_plan_t* slab = &plan->parameter_slabs[i];
    const id4_pipeline_device_placement_t* placement =
        id4_pipeline_plan_placement_at(plan, slab->placement_id);
    if (!placement) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "parameter slab %" PRIhsz
                                " references missing placement %u",
                                i, slab->placement_id);
      break;
    }
    loads[i].slab_index = i;
    loads[i].slab = slab;
    loads[i].device_index = placement->device_index;
    loads[i].device = iree_hal_device_group_device_at(plan->device_group,
                                                      placement->device_index);
    loads[i].queue_affinity = placement->queue_affinity;
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_slab_set_load(
        provider, wait_semaphore_list, signal_semaphore_list,
        plan->parameter_slab_count, loads, plan->stage_name, diagnostics_sink,
        host_allocator, out_slab_set);
  }
  iree_allocator_free(host_allocator, loads);
  return status;
}

static iree_status_t id4_pipeline_plan_append_json_string(
    iree_string_builder_t* builder, iree_string_view_t value) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\""));
  for (iree_host_size_t i = 0; i < value.size; ++i) {
    switch (value.data[i]) {
      case '\\': {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_cstring(builder, "\\\\"));
        break;
      }
      case '"': {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_cstring(builder, "\\\""));
        break;
      }
      case '\n': {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_cstring(builder, "\\n"));
        break;
      }
      default: {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
            builder, iree_make_string_view(value.data + i, 1)));
        break;
      }
    }
  }
  return iree_string_builder_append_cstring(builder, "\"");
}

static iree_status_t id4_pipeline_plan_append_buffer_params_json(
    iree_string_builder_t* builder, iree_hal_buffer_params_t params) {
  return iree_string_builder_append_format(
      builder,
      "{\"type\":%" PRIu64 ",\"access\":%" PRIu64 ",\"usage\":%" PRIu64
      ",\"queue_affinity\":%" PRIu64 ",\"min_alignment\":%" PRIu64 "}",
      (uint64_t)params.type, (uint64_t)params.access, (uint64_t)params.usage,
      (uint64_t)params.queue_affinity, (uint64_t)params.min_alignment);
}

static iree_status_t id4_pipeline_plan_append_region_statistics_json(
    iree_string_builder_t* builder,
    id4_pipeline_region_statistics_t statistics) {
  return iree_string_builder_append_format(
      builder,
      "{\"operation_count\":%" PRIhsz ",\"dispatch_count\":%" PRIhsz
      ",\"barrier_count\":%" PRIhsz
      ",\"current_epoch\":%u"
      ",\"local_acquire_count\":%" PRIhsz ",\"local_release_count\":%" PRIhsz
      ",\"local_reuse_count\":%" PRIhsz ",\"bound_import_count\":%" PRIhsz
      ",\"local_slab_byte_length\":%" PRIu64
      ",\"local_slab_high_water_mark\":%" PRIu64 "}",
      statistics.operation_count, statistics.dispatch_count,
      statistics.barrier_count, statistics.current_epoch,
      statistics.local_acquire_count, statistics.local_release_count,
      statistics.local_reuse_count, statistics.bound_import_count,
      (uint64_t)statistics.local_slab_byte_length,
      (uint64_t)statistics.local_slab_high_water_mark);
}

iree_status_t id4_pipeline_plan_format_json(const id4_pipeline_plan_t* plan,
                                            iree_string_builder_t* builder) {
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(builder);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "{"));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "\"stage\":"));
  IREE_RETURN_IF_ERROR(
      id4_pipeline_plan_append_json_string(builder, plan->stage_name));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      ",\"device_group\":{\"device_count\":%" PRIhsz
      ",\"topology_fingerprint\":\"%08x\"},\"placements\":[",
      iree_hal_device_group_device_count(plan->device_group),
      id4_pipeline_plan_calculate_topology_fingerprint(plan)));
  for (iree_host_size_t i = 0; i < plan->placement_count; ++i) {
    const id4_pipeline_device_placement_t* placement = &plan->placements[i];
    if (i != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "{\"id\":%" PRIhsz ",\"role\":", i));
    IREE_RETURN_IF_ERROR(
        id4_pipeline_plan_append_json_string(builder, placement->role));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        ",\"device_index\":%" PRIhsz ",\"queue_affinity\":%" PRIu64 "}",
        placement->device_index, (uint64_t)placement->queue_affinity));
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "],\"parameter_slabs\":["));
  for (iree_host_size_t i = 0; i < plan->parameter_slab_count; ++i) {
    const id4_pipeline_parameter_slab_plan_t* slab = &plan->parameter_slabs[i];
    if (i != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "{\"id\":%" PRIhsz ",\"scope\":", i));
    IREE_RETURN_IF_ERROR(
        id4_pipeline_plan_append_json_string(builder, slab->scope));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        ",\"placement_id\":%u,\"target_params\":", slab->placement_id));
    IREE_RETURN_IF_ERROR(id4_pipeline_plan_append_buffer_params_json(
        builder, slab->target_params));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        ",\"byte_length\":%" PRIu64 ",\"alignment\":%" PRIu64
        ",\"request_count\":%" PRIhsz ",\"requests\":[",
        (uint64_t)slab->byte_length, (uint64_t)slab->alignment,
        slab->request_count));
    for (iree_host_size_t j = 0; j < slab->request_count; ++j) {
      const id4_pipeline_parameter_request_t* request = &slab->requests[j];
      if (j != 0) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
      }
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(builder, "{\"key\":"));
      IREE_RETURN_IF_ERROR(
          id4_pipeline_plan_append_json_string(builder, request->key));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          ",\"parameter_offset\":%" PRIu64 ",\"buffer_offset\":%" PRIu64
          ",\"length\":%" PRIu64 "}",
          (uint64_t)request->span.parameter_offset,
          (uint64_t)request->span.buffer_offset,
          (uint64_t)request->span.length));
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "]}"));
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "],\"memory_slabs\":["));
  for (iree_host_size_t i = 0; i < plan->memory_slab_count; ++i) {
    const id4_pipeline_memory_slab_plan_t* slab = &plan->memory_slabs[i];
    if (i != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "{\"id\":%" PRIhsz ",\"name\":", i));
    IREE_RETURN_IF_ERROR(
        id4_pipeline_plan_append_json_string(builder, slab->name));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, ",\"placement_id\":%u,\"binding_slot\":%u,\"params\":",
        slab->placement_id, slab->binding_slot));
    IREE_RETURN_IF_ERROR(
        id4_pipeline_plan_append_buffer_params_json(builder, slab->params));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        ",\"byte_length\":%" PRIu64 ",\"alignment\":%" PRIu64
        ",\"high_water_mark\":%" PRIu64 "}",
        (uint64_t)slab->byte_length, (uint64_t)slab->alignment,
        (uint64_t)slab->high_water_mark));
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "],\"kernels\":["));
  for (iree_host_size_t i = 0; i < plan->kernel_count; ++i) {
    const id4_pipeline_kernel_plan_t* kernel = &plan->kernels[i];
    if (i != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "{\"id\":%" PRIhsz ",\"specialization_key\":", i));
    IREE_RETURN_IF_ERROR(id4_pipeline_plan_append_json_string(
        builder, kernel->specialization_key));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(builder, ",\"module_path\":"));
    IREE_RETURN_IF_ERROR(
        id4_pipeline_plan_append_json_string(builder, kernel->module_path));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(builder, ",\"function_name\":"));
    IREE_RETURN_IF_ERROR(
        id4_pipeline_plan_append_json_string(builder, kernel->function_name));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, ",\"placement_id\":%u,\"config_bindings\":[",
        kernel->placement_id));
    for (iree_host_size_t j = 0; j < kernel->config_binding_count; ++j) {
      const id4_pipeline_plan_config_binding_t* binding =
          &kernel->config_bindings[j];
      if (j != 0) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
      }
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(builder, "{\"key\":"));
      IREE_RETURN_IF_ERROR(
          id4_pipeline_plan_append_json_string(builder, binding->key));
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(builder, ",\"value\":"));
      IREE_RETURN_IF_ERROR(
          id4_pipeline_plan_append_json_string(builder, binding->value));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "}"));
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "]}"));
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "],\"regions\":["));
  for (iree_host_size_t i = 0; i < plan->region_count; ++i) {
    const id4_pipeline_region_plan_t* region = &plan->regions[i];
    if (i != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "{\"id\":%" PRIhsz ",\"name\":", i));
    IREE_RETURN_IF_ERROR(
        id4_pipeline_plan_append_json_string(builder, region->name));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        ",\"placement_id\":%u,\"binding_capacity\":%" PRIhsz
        ",\"local_binding_slot\":%u,\"statistics\":",
        region->placement_id, region->binding_capacity,
        region->local_binding_slot));
    IREE_RETURN_IF_ERROR(id4_pipeline_plan_append_region_statistics_json(
        builder, region->statistics));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "}"));
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "],\"diagnostic_taps\":["));
  for (iree_host_size_t i = 0; i < plan->diagnostic_tap_count; ++i) {
    const id4_pipeline_diagnostic_tap_plan_t* tap = &plan->diagnostic_taps[i];
    if (i != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "{\"id\":%" PRIhsz ",\"name\":", i));
    IREE_RETURN_IF_ERROR(
        id4_pipeline_plan_append_json_string(builder, tap->name));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        ",\"region_id\":%u,\"after_operation_ordinal\":%" PRIhsz
        ",\"target_name\":",
        tap->region_id, tap->after_operation_ordinal));
    IREE_RETURN_IF_ERROR(
        id4_pipeline_plan_append_json_string(builder, tap->target_name));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "}"));
  }
  return iree_string_builder_append_cstring(builder, "]}");
}
