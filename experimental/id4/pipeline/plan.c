// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/plan.h"

#include <stdint.h>
#include <string.h>

#include "experimental/id4/pipeline/program.h"

struct id4_pipeline_plan_t {
  // Reference count for shared plan ownership.
  iree_atomic_ref_count_t ref_count;
  // Allocator used for plan storage.
  iree_allocator_t host_allocator;
  // Stage name owned by this plan.
  iree_string_view_t stage_name;
  // Optional immutable semantic program retained by program-backed plans.
  id4_pipeline_program_t* source_program;
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
  // Number of prepare-time parameter load steps.
  iree_host_size_t parameter_load_step_count;
  // Prepare-time parameter load steps owned by this plan.
  id4_pipeline_parameter_load_step_t* parameter_load_steps;
  // Number of planned constant slabs.
  iree_host_size_t constant_slab_count;
  // Planned constant slabs owned by this plan.
  id4_pipeline_constant_slab_plan_t* constant_slabs;
  // Number of planned non-parameter memory slabs.
  iree_host_size_t memory_slab_count;
  // Planned non-parameter memory slabs owned by this plan.
  id4_pipeline_memory_slab_plan_t* memory_slabs;
  // Number of planned tensors backed by plan-shared memory slabs.
  iree_host_size_t shared_tensor_count;
  // Planned tensors backed by plan-shared memory slabs owned by this plan.
  id4_pipeline_shared_tensor_plan_t* shared_tensors;
  // Number of planned external boundary tensors.
  iree_host_size_t boundary_tensor_count;
  // Planned external boundary tensors owned by this plan.
  id4_pipeline_boundary_tensor_plan_t* boundary_tensors;
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

static void id4_pipeline_plan_release_parameter_load_sources(
    iree_host_size_t source_count,
    const id4_pipeline_parameter_load_source_t* sources,
    iree_allocator_t host_allocator) {
  if (!sources) return;
  id4_pipeline_parameter_load_source_t* mutable_sources =
      (id4_pipeline_parameter_load_source_t*)sources;
  for (iree_host_size_t i = 0; i < source_count; ++i) {
    id4_pipeline_string_release(mutable_sources[i].source_scope,
                                host_allocator);
    id4_pipeline_string_release(mutable_sources[i].key, host_allocator);
  }
  iree_allocator_free(host_allocator, mutable_sources);
}

static iree_status_t id4_pipeline_plan_copy_parameter_load_sources(
    iree_host_size_t source_count,
    const id4_pipeline_parameter_load_source_t* sources,
    iree_allocator_t host_allocator,
    const id4_pipeline_parameter_load_source_t** out_sources) {
  *out_sources = NULL;
  if (source_count == 0) return iree_ok_status();
  if (!sources) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter load source array is required");
  }
  id4_pipeline_parameter_load_source_t* target_sources = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(host_allocator, source_count,
                                                   sizeof(target_sources[0]),
                                                   (void**)&target_sources));
  memset(target_sources, 0, source_count * sizeof(target_sources[0]));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < source_count && iree_status_is_ok(status);
       ++i) {
    target_sources[i].dtype = sources[i].dtype;
    target_sources[i].shape = sources[i].shape;
    target_sources[i].byte_length = sources[i].byte_length;
    status = id4_pipeline_string_clone(sources[i].source_scope, host_allocator,
                                       &target_sources[i].source_scope);
    if (iree_status_is_ok(status)) {
      status = id4_pipeline_string_clone(sources[i].key, host_allocator,
                                         &target_sources[i].key);
    }
  }
  if (iree_status_is_ok(status)) {
    *out_sources = target_sources;
  } else {
    id4_pipeline_plan_release_parameter_load_sources(
        source_count, target_sources, host_allocator);
  }
  return status;
}

static void id4_pipeline_plan_destroy(id4_pipeline_plan_t* plan) {
  iree_allocator_t host_allocator = plan->host_allocator;
  for (iree_host_size_t i = 0; i < plan->diagnostic_tap_count; ++i) {
    id4_pipeline_diagnostic_tap_plan_t* tap = &plan->diagnostic_taps[i];
    id4_pipeline_string_release(tap->layout.name, host_allocator);
    id4_pipeline_string_release(tap->target_name, host_allocator);
    id4_pipeline_string_release(tap->name, host_allocator);
  }
  iree_allocator_free(host_allocator, plan->diagnostic_taps);
  for (iree_host_size_t i = 0; i < plan->region_count; ++i) {
    id4_pipeline_region_plan_t* region = &plan->regions[i];
    id4_pipeline_region_local_lifetime_t* lifetimes =
        (id4_pipeline_region_local_lifetime_t*)region->local_lifetimes;
    for (iree_host_size_t j = 0; j < region->local_lifetime_count; ++j) {
      id4_pipeline_string_release(lifetimes[j].name, host_allocator);
    }
    iree_allocator_free(host_allocator, lifetimes);
    iree_allocator_free(host_allocator, (void*)region->parameter_load_groups);
    id4_pipeline_string_release(region->name, host_allocator);
  }
  iree_allocator_free(host_allocator, plan->regions);
  for (iree_host_size_t i = 0; i < plan->boundary_tensor_count; ++i) {
    id4_pipeline_string_release(plan->boundary_tensors[i].layout.name,
                                host_allocator);
  }
  iree_allocator_free(host_allocator, plan->boundary_tensors);
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
  for (iree_host_size_t i = 0; i < plan->shared_tensor_count; ++i) {
    id4_pipeline_string_release(plan->shared_tensors[i].layout.name,
                                host_allocator);
  }
  iree_allocator_free(host_allocator, plan->shared_tensors);
  for (iree_host_size_t i = 0; i < plan->constant_slab_count; ++i) {
    id4_pipeline_constant_slab_plan_t* slab = &plan->constant_slabs[i];
    id4_pipeline_constant_request_t* requests =
        (id4_pipeline_constant_request_t*)slab->requests;
    for (iree_host_size_t j = 0; j < slab->request_count; ++j) {
      id4_pipeline_string_release(requests[j].name, host_allocator);
    }
    iree_allocator_free(host_allocator, requests);
    id4_pipeline_string_release(slab->name, host_allocator);
  }
  iree_allocator_free(host_allocator, plan->constant_slabs);
  for (iree_host_size_t i = 0; i < plan->parameter_load_step_count; ++i) {
    id4_pipeline_string_release(plan->parameter_load_steps[i].name,
                                host_allocator);
    id4_pipeline_string_release(plan->parameter_load_steps[i].source_scope,
                                host_allocator);
    id4_pipeline_plan_release_parameter_load_sources(
        plan->parameter_load_steps[i].source_count,
        plan->parameter_load_steps[i].sources, host_allocator);
    iree_allocator_free(host_allocator,
                        (void*)plan->parameter_load_steps[i].request_indices);
  }
  iree_allocator_free(host_allocator, plan->parameter_load_steps);
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
  id4_pipeline_program_release(plan->source_program);
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
    target->binding_slot = source->binding_slot;
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

static iree_status_t id4_pipeline_plan_copy_parameter_load_steps(
    id4_pipeline_plan_t* plan,
    const id4_pipeline_plan_create_options_t* options) {
  if (plan->parameter_slab_count == 0) {
    if (options->parameter_load_step_count != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "parameter load steps require at least one parameter slab");
    }
    return iree_ok_status();
  }
  const iree_host_size_t load_step_count = options->parameter_load_step_count;
  if (load_step_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter slabs require an explicit parameter load step schedule");
  }
  if (!options->parameter_load_steps) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter load step array is required");
  }
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc_array(plan->host_allocator, load_step_count,
                                  sizeof(plan->parameter_load_steps[0]),
                                  (void**)&plan->parameter_load_steps));
  plan->parameter_load_step_count = load_step_count;
  memset(
      plan->parameter_load_steps, 0,
      plan->parameter_load_step_count * sizeof(plan->parameter_load_steps[0]));
  for (iree_host_size_t i = 0; i < plan->parameter_load_step_count; ++i) {
    const id4_pipeline_parameter_load_step_t* source =
        &options->parameter_load_steps[i];
    IREE_RETURN_IF_ERROR(id4_pipeline_parameter_load_step_validate(
        source, plan->parameter_slab_count, plan->parameter_slabs));
    id4_pipeline_parameter_load_step_t* target = &plan->parameter_load_steps[i];
    target->kind = source->kind;
    target->target_slab_index = source->target_slab_index;
    target->request_offset = source->request_offset;
    target->request_count = source->request_count;
    target->source_count = source->source_count;
    target->readiness_group_key = source->readiness_group_key;
    IREE_RETURN_IF_ERROR(id4_pipeline_string_clone(
        source->name, plan->host_allocator, &target->name));
    IREE_RETURN_IF_ERROR(id4_pipeline_string_clone(
        source->source_scope, plan->host_allocator, &target->source_scope));
    IREE_RETURN_IF_ERROR(id4_pipeline_plan_copy_parameter_load_sources(
        source->source_count, source->sources, plan->host_allocator,
        &target->sources));
    if (source->request_indices) {
      iree_host_size_t* request_indices = NULL;
      IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
          plan->host_allocator, target->request_count,
          sizeof(request_indices[0]), (void**)&request_indices));
      memcpy(request_indices, source->request_indices,
             target->request_count * sizeof(request_indices[0]));
      target->request_indices = request_indices;
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

static iree_status_t id4_pipeline_plan_copy_constant_slabs(
    id4_pipeline_plan_t* plan,
    const id4_pipeline_plan_create_options_t* options) {
  plan->constant_slab_count = options->constant_slab_count;
  if (plan->constant_slab_count == 0) return iree_ok_status();
  if (!options->constant_slabs) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "constant slab array is required");
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      plan->host_allocator, plan->constant_slab_count,
      sizeof(plan->constant_slabs[0]), (void**)&plan->constant_slabs));
  memset(plan->constant_slabs, 0,
         plan->constant_slab_count * sizeof(plan->constant_slabs[0]));
  for (iree_host_size_t i = 0; i < plan->constant_slab_count; ++i) {
    const id4_pipeline_constant_slab_plan_t* source =
        &options->constant_slabs[i];
    if (iree_string_view_is_empty(source->name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "constant slab %" PRIhsz " name is required", i);
    }
    IREE_RETURN_IF_ERROR(id4_pipeline_plan_validate_placement_id(
        plan, source->placement_id, IREE_SV("constant slab")));
    if (source->byte_length == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "constant slab %.*s byte length is zero",
                              (int)source->name.size, source->name.data);
    }
    if (source->alignment != 0 &&
        !iree_device_size_is_power_of_two(source->alignment)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "constant slab %.*s alignment must be a power of two",
          (int)source->name.size, source->name.data);
    }
    if (source->request_count != 0 && !source->requests) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "constant slab %.*s request array is required",
                              (int)source->name.size, source->name.data);
    }
    id4_pipeline_constant_slab_plan_t* target = &plan->constant_slabs[i];
    target->placement_id = source->placement_id;
    target->binding_slot = source->binding_slot;
    target->target_params = source->target_params;
    target->byte_length = source->byte_length;
    target->alignment = source->alignment;
    target->request_count = source->request_count;
    IREE_RETURN_IF_ERROR(id4_pipeline_string_clone(
        source->name, plan->host_allocator, &target->name));
    if (target->request_count == 0) continue;
    id4_pipeline_constant_request_t* requests = NULL;
    IREE_RETURN_IF_ERROR(
        iree_allocator_malloc_array(plan->host_allocator, target->request_count,
                                    sizeof(requests[0]), (void**)&requests));
    memset(requests, 0, target->request_count * sizeof(requests[0]));
    target->requests = requests;
    for (iree_host_size_t j = 0; j < target->request_count; ++j) {
      const id4_pipeline_constant_request_t* request = &source->requests[j];
      if (iree_string_view_is_empty(request->name)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "constant slab %.*s request %" PRIhsz
                                " name is required",
                                (int)source->name.size, source->name.data, j);
      }
      if (request->span.length == 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "constant slab %.*s request %.*s has zero byte length",
            (int)source->name.size, source->name.data, (int)request->name.size,
            request->name.data);
      }
      if (request->span.buffer_offset > target->byte_length ||
          request->span.length >
              target->byte_length - request->span.buffer_offset) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "constant slab %.*s request %.*s exceeds slab byte length",
            (int)source->name.size, source->name.data, (int)request->name.size,
            request->name.data);
      }
      requests[j].span = request->span;
      IREE_RETURN_IF_ERROR(id4_pipeline_string_clone(
          request->name, plan->host_allocator, &requests[j].name));
    }
  }
  return iree_ok_status();
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
    switch (source->scope) {
      case ID4_PIPELINE_MEMORY_SLAB_SCOPE_REGION_LOCAL:
        if ((iree_host_size_t)source->region_id >= plan->region_count) {
          return iree_make_status(
              IREE_STATUS_OUT_OF_RANGE,
              "memory slab %.*s region id %u exceeds region count %" PRIhsz,
              (int)source->name.size, source->name.data, source->region_id,
              plan->region_count);
        }
        break;
      case ID4_PIPELINE_MEMORY_SLAB_SCOPE_PLAN_SHARED:
        if (source->region_id != 0) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "plan-shared memory slab %.*s must have region id 0",
              (int)source->name.size, source->name.data);
        }
        break;
      default:
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "memory slab %.*s scope %u is invalid",
                                (int)source->name.size, source->name.data,
                                (uint32_t)source->scope);
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
    target->scope = source->scope;
    target->region_id = source->region_id;
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

  iree_host_size_t parameter_load_group_count = 0;
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_load_group_count(
      plan->parameter_load_step_count, plan->parameter_load_steps,
      &parameter_load_group_count));
  for (iree_host_size_t i = 0; i < plan->region_count; ++i) {
    const id4_pipeline_region_plan_t* source = &options->regions[i];
    if (iree_string_view_is_empty(source->name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "region %" PRIhsz " name is required", i);
    }
    if (plan->source_program) {
      const iree_host_size_t source_operation_total =
          id4_pipeline_program_operation_count(plan->source_program);
      if (source->source_operation_offset > source_operation_total) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "region %.*s source operation offset %" PRIhsz
            " exceeds source program operation count %" PRIhsz,
            (int)source->name.size, source->name.data,
            source->source_operation_offset, source_operation_total);
      }
      if (source->source_operation_count >
          source_operation_total - source->source_operation_offset) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "region %.*s source operation range [%" PRIhsz ", %" PRIhsz
            ") exceeds source program operation count %" PRIhsz,
            (int)source->name.size, source->name.data,
            source->source_operation_offset,
            source->source_operation_offset + source->source_operation_count,
            source_operation_total);
      }
    } else if (source->source_operation_offset != 0 ||
               source->source_operation_count != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "region %.*s source operation range requires a source program",
          (int)source->name.size, source->name.data);
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
    if (source->local_tensor_alignment != 0 &&
        !iree_device_size_is_power_of_two(source->local_tensor_alignment)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "region %.*s local tensor alignment must be a power of two",
          (int)source->name.size, source->name.data);
    }
    if (source->local_lifetime_count != 0 && !source->local_lifetimes) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "region %.*s local lifetime array is required",
                              (int)source->name.size, source->name.data);
    }
    if (source->local_lifetime_count !=
        source->statistics.local_acquire_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "region %.*s local lifetime count %" PRIhsz
                              " does not match local acquire count %" PRIhsz,
                              (int)source->name.size, source->name.data,
                              source->local_lifetime_count,
                              source->statistics.local_acquire_count);
    }
    if (source->parameter_load_group_count != 0 &&
        !source->parameter_load_groups) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "region %.*s parameter load group array is required",
          (int)source->name.size, source->name.data);
    }
    for (iree_host_size_t j = 0; j < source->parameter_load_group_count; ++j) {
      const iree_host_size_t group_index = source->parameter_load_groups[j];
      if (group_index >= parameter_load_group_count) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "region %.*s parameter load group %" PRIhsz
                                " exceeds parameter load group count %" PRIhsz,
                                (int)source->name.size, source->name.data,
                                group_index, parameter_load_group_count);
      }
      if (j != 0 && source->parameter_load_groups[j - 1] >= group_index) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "region %.*s parameter load groups must be strictly increasing",
            (int)source->name.size, source->name.data);
      }
    }
    id4_pipeline_region_plan_t* target = &plan->regions[i];
    target->source_operation_offset = source->source_operation_offset;
    target->source_operation_count = source->source_operation_count;
    target->placement_id = source->placement_id;
    target->binding_capacity = source->binding_capacity;
    target->local_binding_slot = source->local_binding_slot;
    target->local_tensor_alignment = source->local_tensor_alignment;
    target->statistics = source->statistics;
    IREE_RETURN_IF_ERROR(id4_pipeline_string_clone(
        source->name, plan->host_allocator, &target->name));
    target->parameter_load_group_count = source->parameter_load_group_count;
    if (target->parameter_load_group_count != 0) {
      iree_host_size_t* parameter_load_groups = NULL;
      IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
          plan->host_allocator, target->parameter_load_group_count,
          sizeof(parameter_load_groups[0]), (void**)&parameter_load_groups));
      memcpy(parameter_load_groups, source->parameter_load_groups,
             target->parameter_load_group_count *
                 sizeof(parameter_load_groups[0]));
      target->parameter_load_groups = parameter_load_groups;
    }
    target->local_lifetime_count = source->local_lifetime_count;
    if (target->local_lifetime_count == 0) continue;
    id4_pipeline_region_local_lifetime_t* lifetimes = NULL;
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        plan->host_allocator, target->local_lifetime_count,
        sizeof(lifetimes[0]), (void**)&lifetimes));
    memset(lifetimes, 0, target->local_lifetime_count * sizeof(lifetimes[0]));
    target->local_lifetimes = lifetimes;
    for (iree_host_size_t j = 0; j < target->local_lifetime_count; ++j) {
      const id4_pipeline_region_local_lifetime_t* source_lifetime =
          &source->local_lifetimes[j];
      if (iree_string_view_is_empty(source_lifetime->name)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "region %.*s local lifetime %" PRIhsz
                                " name is required",
                                (int)source->name.size, source->name.data, j);
      }
      if (id4_pipeline_tensor_dtype_byte_length(source_lifetime->dtype) == 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "region %.*s local lifetime %.*s dtype is "
                                "invalid",
                                (int)source->name.size, source->name.data,
                                (int)source_lifetime->name.size,
                                source_lifetime->name.data);
      }
      if (source_lifetime->byte_length == 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "region %.*s local lifetime %.*s byte length "
                                "is zero",
                                (int)source->name.size, source->name.data,
                                (int)source_lifetime->name.size,
                                source_lifetime->name.data);
      }
      iree_device_size_t lifetime_end = 0;
      if (!iree_device_size_checked_add(source_lifetime->offset,
                                        source_lifetime->byte_length,
                                        &lifetime_end)) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "region %.*s local lifetime %.*s range "
                                "overflow",
                                (int)source->name.size, source->name.data,
                                (int)source_lifetime->name.size,
                                source_lifetime->name.data);
      }
      if (lifetime_end > source->statistics.local_slab_byte_length) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "region %.*s local lifetime %.*s end offset %" PRIu64
            " exceeds local slab byte length %" PRIu64,
            (int)source->name.size, source->name.data,
            (int)source_lifetime->name.size, source_lifetime->name.data,
            (uint64_t)lifetime_end,
            (uint64_t)source->statistics.local_slab_byte_length);
      }
      id4_pipeline_region_local_lifetime_t* target_lifetime = &lifetimes[j];
      *target_lifetime = *source_lifetime;
      IREE_RETURN_IF_ERROR(id4_pipeline_string_clone(
          source_lifetime->name, plan->host_allocator, &target_lifetime->name));
    }
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_plan_validate_region_binding_slot(
    const id4_pipeline_plan_t* plan, uint32_t region_id, uint32_t binding_slot,
    iree_string_view_t binding_name) {
  if ((iree_host_size_t)region_id >= plan->region_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "%.*s region id %u exceeds region count %" PRIhsz,
                            (int)binding_name.size, binding_name.data,
                            region_id, plan->region_count);
  }
  const id4_pipeline_region_plan_t* region = &plan->regions[region_id];
  if (binding_slot >= region->binding_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "%.*s binding slot %u exceeds region binding "
                            "capacity %" PRIhsz,
                            (int)binding_name.size, binding_name.data,
                            binding_slot, region->binding_capacity);
  }
  if (binding_slot == region->local_binding_slot) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s binding slot must not match the local slab "
                            "binding slot",
                            (int)binding_name.size, binding_name.data);
  }
  return iree_ok_status();
}

static bool id4_pipeline_plan_memory_slab_visible_to_region(
    const id4_pipeline_memory_slab_plan_t* slab, uint32_t region_id) {
  switch (slab->scope) {
    case ID4_PIPELINE_MEMORY_SLAB_SCOPE_REGION_LOCAL:
      return slab->region_id == region_id;
    case ID4_PIPELINE_MEMORY_SLAB_SCOPE_PLAN_SHARED:
      return true;
    default:
      return false;
  }
}

static iree_status_t id4_pipeline_plan_validate_memory_slab_region_binding(
    const id4_pipeline_plan_t* plan,
    const id4_pipeline_memory_slab_plan_t* slab, uint32_t region_id) {
  if ((iree_host_size_t)region_id >= plan->region_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "memory slab %.*s region id %u exceeds region "
                            "count %" PRIhsz,
                            (int)slab->name.size, slab->name.data, region_id,
                            plan->region_count);
  }
  const id4_pipeline_region_plan_t* region = &plan->regions[region_id];
  if (slab->binding_slot >= region->binding_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "memory slab %.*s binding slot %u exceeds region "
                            "binding capacity %" PRIhsz,
                            (int)slab->name.size, slab->name.data,
                            slab->binding_slot, region->binding_capacity);
  }
  switch (slab->scope) {
    case ID4_PIPELINE_MEMORY_SLAB_SCOPE_REGION_LOCAL:
      if (slab->binding_slot != region->local_binding_slot) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "region-local memory slab %.*s binding slot %u must match region "
            "%u local binding slot %u",
            (int)slab->name.size, slab->name.data, slab->binding_slot,
            region_id, region->local_binding_slot);
      }
      if (slab->placement_id != region->placement_id) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "region-local memory slab %.*s placement %u does not match region "
            "%u placement %u",
            (int)slab->name.size, slab->name.data, slab->placement_id,
            region_id, region->placement_id);
      }
      return iree_ok_status();
    case ID4_PIPELINE_MEMORY_SLAB_SCOPE_PLAN_SHARED:
      if (slab->binding_slot == region->local_binding_slot) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "plan-shared memory slab %.*s binding slot must not match region "
            "%u local binding slot",
            (int)slab->name.size, slab->name.data, region_id);
      }
      return iree_ok_status();
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT, "memory slab %.*s scope %u is invalid",
          (int)slab->name.size, slab->name.data, (uint32_t)slab->scope);
  }
}

static iree_status_t id4_pipeline_plan_validate_storage_binding_slots(
    const id4_pipeline_plan_t* plan) {
  if (plan->region_count == 0) return iree_ok_status();
  for (iree_host_size_t region_index = 0; region_index < plan->region_count;
       ++region_index) {
    const uint32_t region_id = (uint32_t)region_index;
    for (iree_host_size_t i = 0; i < plan->parameter_slab_count; ++i) {
      IREE_RETURN_IF_ERROR(id4_pipeline_plan_validate_region_binding_slot(
          plan, region_id, plan->parameter_slabs[i].binding_slot,
          IREE_SV("parameter slab")));
    }
    for (iree_host_size_t i = 0; i < plan->constant_slab_count; ++i) {
      IREE_RETURN_IF_ERROR(id4_pipeline_plan_validate_region_binding_slot(
          plan, region_id, plan->constant_slabs[i].binding_slot,
          IREE_SV("constant slab")));
      for (iree_host_size_t j = i + 1; j < plan->constant_slab_count; ++j) {
        if (plan->constant_slabs[i].binding_slot ==
            plan->constant_slabs[j].binding_slot) {
          return iree_make_status(
              IREE_STATUS_ALREADY_EXISTS,
              "constant slab %.*s binding slot %u is already planned",
              (int)plan->constant_slabs[i].name.size,
              plan->constant_slabs[i].name.data,
              plan->constant_slabs[i].binding_slot);
        }
      }
      for (iree_host_size_t j = 0; j < plan->parameter_slab_count; ++j) {
        if (plan->constant_slabs[i].binding_slot ==
            plan->parameter_slabs[j].binding_slot) {
          return iree_make_status(
              IREE_STATUS_ALREADY_EXISTS,
              "constant slab %.*s binding slot %u conflicts with parameter "
              "slab",
              (int)plan->constant_slabs[i].name.size,
              plan->constant_slabs[i].name.data,
              plan->constant_slabs[i].binding_slot);
        }
      }
    }
    for (iree_host_size_t i = 0; i < plan->memory_slab_count; ++i) {
      const id4_pipeline_memory_slab_plan_t* slab = &plan->memory_slabs[i];
      if (!id4_pipeline_plan_memory_slab_visible_to_region(slab, region_id)) {
        continue;
      }
      IREE_RETURN_IF_ERROR(
          id4_pipeline_plan_validate_memory_slab_region_binding(plan, slab,
                                                                region_id));
      for (iree_host_size_t j = i + 1; j < plan->memory_slab_count; ++j) {
        const id4_pipeline_memory_slab_plan_t* other_slab =
            &plan->memory_slabs[j];
        if (!id4_pipeline_plan_memory_slab_visible_to_region(other_slab,
                                                             region_id)) {
          continue;
        }
        if (slab->binding_slot == other_slab->binding_slot) {
          return iree_make_status(
              IREE_STATUS_ALREADY_EXISTS,
              "memory slab %.*s binding slot %u is already planned in region "
              "%u",
              (int)slab->name.size, slab->name.data, slab->binding_slot,
              region_id);
        }
      }
      for (iree_host_size_t j = 0; j < plan->parameter_slab_count; ++j) {
        if (slab->binding_slot == plan->parameter_slabs[j].binding_slot) {
          return iree_make_status(
              IREE_STATUS_ALREADY_EXISTS,
              "memory slab %.*s binding slot %u conflicts with parameter slab",
              (int)slab->name.size, slab->name.data, slab->binding_slot);
        }
      }
      for (iree_host_size_t j = 0; j < plan->constant_slab_count; ++j) {
        if (slab->binding_slot == plan->constant_slabs[j].binding_slot) {
          return iree_make_status(
              IREE_STATUS_ALREADY_EXISTS,
              "memory slab %.*s binding slot %u conflicts with constant slab",
              (int)slab->name.size, slab->name.data, slab->binding_slot);
        }
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_plan_validate_tensor_shape(
    id4_pipeline_tensor_shape_t shape, iree_string_view_t tensor_name) {
  if (shape.rank > ID4_PIPELINE_TENSOR_MAX_RANK) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "tensor layout %.*s rank %u exceeds max rank %u",
                            (int)tensor_name.size, tensor_name.data, shape.rank,
                            ID4_PIPELINE_TENSOR_MAX_RANK);
  }
  for (uint32_t i = 0; i < shape.rank; ++i) {
    if (shape.dims[i] == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "tensor layout %.*s dimension %u is zero",
                              (int)tensor_name.size, tensor_name.data, i);
    }
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_plan_validate_tensor_layout(
    const id4_pipeline_tensor_layout_t* layout) {
  if (iree_string_view_is_empty(layout->name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tensor layout name is required");
  }
  const iree_device_size_t element_byte_length =
      id4_pipeline_tensor_dtype_byte_length(layout->dtype);
  if (element_byte_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tensor layout %.*s dtype is invalid",
                            (int)layout->name.size, layout->name.data);
  }
  if (layout->byte_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tensor layout %.*s byte length is zero",
                            (int)layout->name.size, layout->name.data);
  }
  if (layout->alignment != 0 &&
      !iree_device_size_is_power_of_two(layout->alignment)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tensor layout %.*s alignment must be a power "
                            "of two",
                            (int)layout->name.size, layout->name.data);
  }
  IREE_RETURN_IF_ERROR(
      id4_pipeline_plan_validate_tensor_shape(layout->shape, layout->name));
  uint64_t element_count = 1;
  for (uint32_t i = 0; i < layout->shape.rank; ++i) {
    if (element_count > UINT64_MAX / layout->shape.dims[i]) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "tensor layout %.*s element count overflow",
                              (int)layout->name.size, layout->name.data);
    }
    element_count *= layout->shape.dims[i];
  }
  if (element_count > UINT64_MAX / element_byte_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "tensor layout %.*s byte length overflow",
                            (int)layout->name.size, layout->name.data);
  }
  const iree_device_size_t expected_byte_length =
      (iree_device_size_t)(element_count * element_byte_length);
  if (layout->byte_length != expected_byte_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tensor layout %.*s byte length %" PRIu64
                            " does not match dtype/shape byte length %" PRIu64,
                            (int)layout->name.size, layout->name.data,
                            (uint64_t)layout->byte_length,
                            (uint64_t)expected_byte_length);
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_plan_validate_boundary_tensor_flags(
    const id4_pipeline_boundary_tensor_plan_t* tensor) {
  const id4_pipeline_boundary_tensor_flags_t allowed_flags =
      ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_IMPORTED |
      ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_EXPORTED |
      ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_INITIALIZED;
  if (iree_any_bit_set(tensor->flags, ~allowed_flags)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "boundary tensor %.*s has unsupported flags 0x%x",
                            (int)tensor->layout.name.size,
                            tensor->layout.name.data, tensor->flags);
  }
  if (!iree_all_bits_set(tensor->flags,
                         ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_IMPORTED)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT, "boundary tensor %.*s must be imported",
        (int)tensor->layout.name.size, tensor->layout.name.data);
  }
  return iree_ok_status();
}

static bool id4_pipeline_plan_shared_tensor_ordinal_exists(
    const id4_pipeline_plan_t* plan, uint32_t program_tensor_ordinal) {
  for (iree_host_size_t i = 0; i < plan->shared_tensor_count; ++i) {
    if (plan->shared_tensors[i].program_tensor_ordinal ==
        program_tensor_ordinal) {
      return true;
    }
  }
  return false;
}

static iree_status_t id4_pipeline_plan_copy_shared_tensors(
    id4_pipeline_plan_t* plan,
    const id4_pipeline_plan_create_options_t* options) {
  const iree_host_size_t shared_tensor_count = options->shared_tensor_count;
  if (shared_tensor_count == 0) return iree_ok_status();
  if (!options->shared_tensors) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "shared tensor array is required");
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      plan->host_allocator, shared_tensor_count,
      sizeof(plan->shared_tensors[0]), (void**)&plan->shared_tensors));
  memset(plan->shared_tensors, 0,
         shared_tensor_count * sizeof(plan->shared_tensors[0]));
  for (iree_host_size_t i = 0; i < shared_tensor_count; ++i) {
    const id4_pipeline_shared_tensor_plan_t* source =
        &options->shared_tensors[i];
    IREE_RETURN_IF_ERROR(
        id4_pipeline_plan_validate_tensor_layout(&source->layout));
    if (id4_pipeline_plan_shared_tensor_ordinal_exists(
            plan, source->program_tensor_ordinal)) {
      return iree_make_status(
          IREE_STATUS_ALREADY_EXISTS,
          "shared tensor %.*s program tensor ordinal %u is already planned",
          (int)source->layout.name.size, source->layout.name.data,
          source->program_tensor_ordinal);
    }
    if (source->memory_slab_index >= plan->memory_slab_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "shared tensor %.*s memory slab index %" PRIhsz
          " exceeds memory slab count %" PRIhsz,
          (int)source->layout.name.size, source->layout.name.data,
          source->memory_slab_index, plan->memory_slab_count);
    }
    const id4_pipeline_memory_slab_plan_t* slab =
        &plan->memory_slabs[source->memory_slab_index];
    if (slab->scope != ID4_PIPELINE_MEMORY_SLAB_SCOPE_PLAN_SHARED) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "shared tensor %.*s must reference a plan-shared memory slab",
          (int)source->layout.name.size, source->layout.name.data);
    }
    if (source->offset > slab->byte_length ||
        source->layout.byte_length > slab->byte_length - source->offset) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "shared tensor %.*s byte range exceeds memory slab %.*s",
          (int)source->layout.name.size, source->layout.name.data,
          (int)slab->name.size, slab->name.data);
    }
    if (source->layout.alignment != 0 &&
        !iree_device_size_has_alignment(source->offset,
                                        source->layout.alignment)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "shared tensor %.*s offset %" PRIu64
          " does not satisfy alignment %" PRIu64,
          (int)source->layout.name.size, source->layout.name.data,
          (uint64_t)source->offset, (uint64_t)source->layout.alignment);
    }
    if ((iree_host_size_t)source->acquire_region_id >= plan->region_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "shared tensor %.*s acquire region %u exceeds region count %" PRIhsz,
          (int)source->layout.name.size, source->layout.name.data,
          source->acquire_region_id, plan->region_count);
    }
    if ((iree_host_size_t)source->last_use_region_id >= plan->region_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "shared tensor %.*s last-use region %u exceeds region count %" PRIhsz,
          (int)source->layout.name.size, source->layout.name.data,
          source->last_use_region_id, plan->region_count);
    }
    if (source->last_use_region_id < source->acquire_region_id) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "shared tensor %.*s last-use region precedes acquire region",
          (int)source->layout.name.size, source->layout.name.data);
    }
    id4_pipeline_shared_tensor_plan_t* target =
        &plan->shared_tensors[plan->shared_tensor_count];
    target->layout = source->layout;
    target->program_tensor_ordinal = source->program_tensor_ordinal;
    target->memory_slab_index = source->memory_slab_index;
    target->offset = source->offset;
    target->acquire_region_id = source->acquire_region_id;
    target->last_use_region_id = source->last_use_region_id;
    IREE_RETURN_IF_ERROR(id4_pipeline_string_clone(
        source->layout.name, plan->host_allocator, &target->layout.name));
    ++plan->shared_tensor_count;
  }
  return iree_ok_status();
}

static bool id4_pipeline_plan_boundary_binding_conflicts(
    const id4_pipeline_plan_t* plan, uint32_t region_id,
    uint32_t binding_slot) {
  for (iree_host_size_t i = 0; i < plan->parameter_slab_count; ++i) {
    if (plan->parameter_slabs[i].binding_slot == binding_slot) return true;
  }
  for (iree_host_size_t i = 0; i < plan->constant_slab_count; ++i) {
    if (plan->constant_slabs[i].binding_slot == binding_slot) return true;
  }
  for (iree_host_size_t i = 0; i < plan->memory_slab_count; ++i) {
    const id4_pipeline_memory_slab_plan_t* slab = &plan->memory_slabs[i];
    if (id4_pipeline_plan_memory_slab_visible_to_region(slab, region_id) &&
        slab->binding_slot == binding_slot) {
      return true;
    }
  }
  for (iree_host_size_t i = 0; i < plan->boundary_tensor_count; ++i) {
    const id4_pipeline_boundary_tensor_plan_t* tensor =
        &plan->boundary_tensors[i];
    if (tensor->binding_slot == binding_slot) return true;
  }
  return false;
}

static iree_status_t id4_pipeline_plan_validate_boundary_binding_slot(
    const id4_pipeline_plan_t* plan,
    const id4_pipeline_boundary_tensor_plan_t* tensor) {
  if (plan->region_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "boundary tensor %.*s requires at least one "
                            "executable region",
                            (int)tensor->layout.name.size,
                            tensor->layout.name.data);
  }
  for (iree_host_size_t i = 0; i < plan->region_count; ++i) {
    if (i > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "region index %" PRIhsz " exceeds uint32_t", i);
    }
    const uint32_t region_id = (uint32_t)i;
    IREE_RETURN_IF_ERROR(id4_pipeline_plan_validate_region_binding_slot(
        plan, region_id, tensor->binding_slot, tensor->layout.name));
    if (id4_pipeline_plan_boundary_binding_conflicts(plan, region_id,
                                                     tensor->binding_slot)) {
      return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                              "boundary tensor %.*s binding slot %u is "
                              "already planned",
                              (int)tensor->layout.name.size,
                              tensor->layout.name.data, tensor->binding_slot);
    }
  }
  return iree_ok_status();
}

static bool id4_pipeline_plan_diagnostic_tap_binding_conflicts(
    const id4_pipeline_plan_t* plan, uint32_t region_id,
    uint32_t binding_slot) {
  if (id4_pipeline_plan_boundary_binding_conflicts(plan, region_id,
                                                   binding_slot)) {
    return true;
  }
  for (iree_host_size_t i = 0; i < plan->diagnostic_tap_count; ++i) {
    const id4_pipeline_diagnostic_tap_plan_t* tap = &plan->diagnostic_taps[i];
    if (tap->region_id == region_id && tap->binding_slot == binding_slot) {
      return true;
    }
  }
  return false;
}

static iree_status_t id4_pipeline_plan_copy_boundary_tensors(
    id4_pipeline_plan_t* plan,
    const id4_pipeline_plan_create_options_t* options) {
  const iree_host_size_t boundary_tensor_count = options->boundary_tensor_count;
  if (boundary_tensor_count == 0) return iree_ok_status();
  if (!options->boundary_tensors) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "boundary tensor array is required");
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      plan->host_allocator, boundary_tensor_count,
      sizeof(plan->boundary_tensors[0]), (void**)&plan->boundary_tensors));
  memset(plan->boundary_tensors, 0,
         boundary_tensor_count * sizeof(plan->boundary_tensors[0]));
  for (iree_host_size_t i = 0; i < boundary_tensor_count; ++i) {
    const id4_pipeline_boundary_tensor_plan_t* source =
        &options->boundary_tensors[i];
    IREE_RETURN_IF_ERROR(
        id4_pipeline_plan_validate_tensor_layout(&source->layout));
    IREE_RETURN_IF_ERROR(
        id4_pipeline_plan_validate_boundary_tensor_flags(source));
    IREE_RETURN_IF_ERROR(id4_pipeline_plan_validate_placement_id(
        plan, source->placement_id, IREE_SV("boundary tensor")));
    IREE_RETURN_IF_ERROR(
        id4_pipeline_plan_validate_boundary_binding_slot(plan, source));
    id4_pipeline_boundary_tensor_plan_t* target = &plan->boundary_tensors[i];
    target->layout = source->layout;
    target->flags = source->flags;
    target->placement_id = source->placement_id;
    target->binding_slot = source->binding_slot;
    IREE_RETURN_IF_ERROR(id4_pipeline_string_clone(
        source->layout.name, plan->host_allocator, &target->layout.name));
    ++plan->boundary_tensor_count;
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_plan_copy_diagnostic_taps(
    id4_pipeline_plan_t* plan,
    const id4_pipeline_plan_create_options_t* options) {
  const iree_host_size_t diagnostic_tap_count = options->diagnostic_tap_count;
  if (diagnostic_tap_count == 0) return iree_ok_status();
  if (!options->diagnostic_taps) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "diagnostic tap array is required");
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      plan->host_allocator, diagnostic_tap_count,
      sizeof(plan->diagnostic_taps[0]), (void**)&plan->diagnostic_taps));
  memset(plan->diagnostic_taps, 0,
         diagnostic_tap_count * sizeof(plan->diagnostic_taps[0]));
  for (iree_host_size_t i = 0; i < diagnostic_tap_count; ++i) {
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
    IREE_RETURN_IF_ERROR(
        id4_pipeline_plan_validate_tensor_layout(&source->layout));
    IREE_RETURN_IF_ERROR(id4_pipeline_plan_validate_placement_id(
        plan, source->placement_id, IREE_SV("diagnostic tap")));
    if ((iree_host_size_t)source->region_id >= plan->region_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "diagnostic tap %.*s region id %u exceeds "
                              "region count %" PRIhsz,
                              (int)source->name.size, source->name.data,
                              source->region_id, plan->region_count);
    }
    IREE_RETURN_IF_ERROR(id4_pipeline_plan_validate_region_binding_slot(
        plan, source->region_id, source->binding_slot, source->name));
    if (id4_pipeline_plan_diagnostic_tap_binding_conflicts(
            plan, source->region_id, source->binding_slot)) {
      return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                              "diagnostic tap %.*s binding slot %u is already "
                              "planned",
                              (int)source->name.size, source->name.data,
                              source->binding_slot);
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
    target->placement_id = source->placement_id;
    target->binding_slot = source->binding_slot;
    target->after_operation_ordinal = source->after_operation_ordinal;
    target->layout = source->layout;
    IREE_RETURN_IF_ERROR(id4_pipeline_string_clone(
        source->name, plan->host_allocator, &target->name));
    IREE_RETURN_IF_ERROR(id4_pipeline_string_clone(
        source->target_name, plan->host_allocator, &target->target_name));
    IREE_RETURN_IF_ERROR(id4_pipeline_string_clone(
        source->layout.name, plan->host_allocator, &target->layout.name));
    ++plan->diagnostic_tap_count;
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
    plan->source_program = (id4_pipeline_program_t*)options->source_program;
    id4_pipeline_program_retain(plan->source_program);
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
    status = id4_pipeline_plan_copy_parameter_load_steps(plan, options);
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_plan_copy_constant_slabs(plan, options);
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_plan_copy_kernels(plan, options);
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_plan_copy_regions(plan, options);
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_plan_copy_memory_slabs(plan, options);
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_plan_copy_shared_tensors(plan, options);
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_plan_validate_storage_binding_slots(plan);
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_plan_copy_boundary_tensors(plan, options);
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

const id4_pipeline_program_t* id4_pipeline_plan_source_program(
    const id4_pipeline_plan_t* plan) {
  return plan ? plan->source_program : NULL;
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

iree_host_size_t id4_pipeline_plan_parameter_load_step_count(
    const id4_pipeline_plan_t* plan) {
  return plan ? plan->parameter_load_step_count : 0;
}

const id4_pipeline_parameter_load_step_t*
id4_pipeline_plan_parameter_load_step_at(const id4_pipeline_plan_t* plan,
                                         iree_host_size_t index) {
  if (!plan || index >= plan->parameter_load_step_count) return NULL;
  return &plan->parameter_load_steps[index];
}

iree_status_t id4_pipeline_plan_parameter_load_group_count(
    const id4_pipeline_plan_t* plan, iree_host_size_t* out_count) {
  IREE_ASSERT_ARGUMENT(out_count);
  *out_count = 0;
  if (!plan) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "plan is required");
  }
  return id4_pipeline_parameter_load_group_count(
      plan->parameter_load_step_count, plan->parameter_load_steps, out_count);
}

iree_status_t id4_pipeline_plan_parameter_load_group_at(
    const id4_pipeline_plan_t* plan, iree_host_size_t index,
    id4_pipeline_parameter_load_group_t* out_group) {
  IREE_ASSERT_ARGUMENT(out_group);
  memset(out_group, 0, sizeof(*out_group));
  if (!plan) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "plan is required");
  }
  return id4_pipeline_parameter_load_group_at(plan->parameter_load_step_count,
                                              plan->parameter_load_steps, index,
                                              out_group);
}

iree_host_size_t id4_pipeline_plan_constant_slab_count(
    const id4_pipeline_plan_t* plan) {
  return plan ? plan->constant_slab_count : 0;
}

const id4_pipeline_constant_slab_plan_t* id4_pipeline_plan_constant_slab_at(
    const id4_pipeline_plan_t* plan, iree_host_size_t index) {
  if (!plan || index >= plan->constant_slab_count) return NULL;
  return &plan->constant_slabs[index];
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

iree_host_size_t id4_pipeline_plan_shared_tensor_count(
    const id4_pipeline_plan_t* plan) {
  return plan ? plan->shared_tensor_count : 0;
}

const id4_pipeline_shared_tensor_plan_t* id4_pipeline_plan_shared_tensor_at(
    const id4_pipeline_plan_t* plan, iree_host_size_t index) {
  if (!plan || index >= plan->shared_tensor_count) return NULL;
  return &plan->shared_tensors[index];
}

iree_host_size_t id4_pipeline_plan_boundary_tensor_count(
    const id4_pipeline_plan_t* plan) {
  return plan ? plan->boundary_tensor_count : 0;
}

const id4_pipeline_boundary_tensor_plan_t* id4_pipeline_plan_boundary_tensor_at(
    const id4_pipeline_plan_t* plan, iree_host_size_t index) {
  if (!plan || index >= plan->boundary_tensor_count) return NULL;
  return &plan->boundary_tensors[index];
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

static iree_device_size_t id4_pipeline_plan_direct_load_step_source_length(
    const id4_pipeline_plan_t* plan,
    const id4_pipeline_parameter_load_step_t* step) {
  const id4_pipeline_parameter_slab_plan_t* slab =
      &plan->parameter_slabs[step->target_slab_index];
  iree_device_size_t byte_length = 0;
  for (iree_host_size_t i = 0; i < step->request_count; ++i) {
    const iree_host_size_t request_index = step->request_indices
                                               ? step->request_indices[i]
                                               : step->request_offset + i;
    byte_length += slab->requests[request_index].span.length;
  }
  return byte_length;
}

static iree_device_size_t id4_pipeline_plan_encoded_load_step_source_length(
    const id4_pipeline_parameter_load_step_t* step) {
  iree_device_size_t byte_length = 0;
  for (iree_host_size_t i = 0; i < step->source_count; ++i) {
    byte_length += step->sources[i].byte_length;
  }
  return byte_length;
}

static bool id4_pipeline_plan_parameter_load_step_is_encode(
    const id4_pipeline_parameter_load_step_t* step) {
  switch (step->kind) {
    case ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_FP8_E4M3_SCALED_TO_BF16:
    case ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_BF16_LINEAR_RHS_TILE:
    case ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_FP8_E4M3_SCALED_TO_BF16_LINEAR_RHS_TILE:
      return true;
    default:
      return false;
  }
}

static bool id4_pipeline_plan_parameter_load_steps_share_readiness_group(
    const id4_pipeline_parameter_load_step_t* lhs,
    const id4_pipeline_parameter_load_step_t* rhs) {
  return lhs->target_slab_index == rhs->target_slab_index &&
         lhs->readiness_group_key == rhs->readiness_group_key;
}

static void id4_pipeline_plan_accumulate_parameter_load_group_statistics(
    const id4_pipeline_plan_t* plan,
    id4_pipeline_plan_statistics_t* statistics) {
  for (iree_host_size_t step_index = 0;
       step_index < plan->parameter_load_step_count;) {
    const id4_pipeline_parameter_load_step_t* step =
        &plan->parameter_load_steps[step_index];
    ++statistics->parameter_load_group_count;
    if (!id4_pipeline_plan_parameter_load_step_is_encode(step)) {
      ++statistics->parameter_gather_load_group_count;
      ++step_index;
      continue;
    }

    ++statistics->parameter_encode_load_group_count;
    do {
      ++step_index;
    } while (step_index < plan->parameter_load_step_count &&
             id4_pipeline_plan_parameter_load_step_is_encode(
                 &plan->parameter_load_steps[step_index]) &&
             id4_pipeline_plan_parameter_load_steps_share_readiness_group(
                 step, &plan->parameter_load_steps[step_index]));
  }
}

id4_pipeline_plan_statistics_t id4_pipeline_plan_statistics(
    const id4_pipeline_plan_t* plan) {
  id4_pipeline_plan_statistics_t statistics;
  memset(&statistics, 0, sizeof(statistics));
  if (!plan) return statistics;

  for (iree_host_size_t i = 0; i < plan->parameter_slab_count; ++i) {
    const id4_pipeline_parameter_slab_plan_t* slab = &plan->parameter_slabs[i];
    statistics.parameter_slab_byte_length += slab->byte_length;
    if (slab->byte_length > statistics.largest_parameter_slab_byte_length) {
      statistics.largest_parameter_slab_byte_length = slab->byte_length;
    }
  }
  for (iree_host_size_t i = 0; i < plan->parameter_load_step_count; ++i) {
    const id4_pipeline_parameter_load_step_t* step =
        &plan->parameter_load_steps[i];
    switch (step->kind) {
      case ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_GATHER: {
        const iree_device_size_t byte_length =
            id4_pipeline_plan_direct_load_step_source_length(plan, step);
        statistics.parameter_direct_source_byte_length += byte_length;
        statistics.parameter_source_byte_length += byte_length;
        ++statistics.parameter_gather_load_step_count;
        break;
      }
      case ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_FP8_E4M3_SCALED_TO_BF16:
      case ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_BF16_LINEAR_RHS_TILE:
      case ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_FP8_E4M3_SCALED_TO_BF16_LINEAR_RHS_TILE: {
        const iree_device_size_t byte_length =
            id4_pipeline_plan_encoded_load_step_source_length(step);
        statistics.parameter_encoded_source_byte_length += byte_length;
        statistics.parameter_source_byte_length += byte_length;
        ++statistics.parameter_encode_load_step_count;
        break;
      }
      default:
        break;
    }
  }
  id4_pipeline_plan_accumulate_parameter_load_group_statistics(plan,
                                                               &statistics);
  for (iree_host_size_t i = 0; i < plan->constant_slab_count; ++i) {
    const id4_pipeline_constant_slab_plan_t* slab = &plan->constant_slabs[i];
    statistics.constant_slab_byte_length += slab->byte_length;
  }
  for (iree_host_size_t i = 0; i < plan->memory_slab_count; ++i) {
    const id4_pipeline_memory_slab_plan_t* slab = &plan->memory_slabs[i];
    statistics.memory_slab_byte_length += slab->byte_length;
    statistics.memory_slab_high_water_mark += slab->high_water_mark;
  }
  for (iree_host_size_t i = 0; i < plan->shared_tensor_count; ++i) {
    const id4_pipeline_shared_tensor_plan_t* shared_tensor =
        &plan->shared_tensors[i];
    statistics.shared_tensor_byte_length += shared_tensor->layout.byte_length;
  }
  for (iree_host_size_t i = 0; i < plan->boundary_tensor_count; ++i) {
    const id4_pipeline_boundary_tensor_plan_t* boundary =
        &plan->boundary_tensors[i];
    statistics.boundary_tensor_byte_length += boundary->layout.byte_length;
  }
  for (iree_host_size_t i = 0; i < plan->diagnostic_tap_count; ++i) {
    const id4_pipeline_diagnostic_tap_plan_t* tap = &plan->diagnostic_taps[i];
    statistics.diagnostic_tap_byte_length += tap->layout.byte_length;
  }
  statistics.kernel_count = plan->kernel_count;
  statistics.region_count = plan->region_count;
  statistics.shared_tensor_count = plan->shared_tensor_count;
  for (iree_host_size_t i = 0; i < plan->region_count; ++i) {
    const id4_pipeline_region_plan_t* region = &plan->regions[i];
    statistics.operation_count += region->statistics.operation_count;
    statistics.dispatch_count += region->statistics.dispatch_count;
  }
  return statistics;
}

static iree_status_t id4_pipeline_plan_make_parameter_slab_loads(
    const id4_pipeline_plan_t* plan, iree_allocator_t host_allocator,
    id4_pipeline_parameter_slab_load_t** out_loads) {
  *out_loads = NULL;
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
    *out_loads = loads;
  } else {
    iree_allocator_free(host_allocator, loads);
  }
  return status;
}

iree_status_t id4_pipeline_plan_load_parameter_slabs(
    const id4_pipeline_plan_t* plan,
    const id4_pipeline_parameter_slab_set_load_options_t* options,
    iree_allocator_t host_allocator,
    id4_pipeline_parameter_slab_set_t** out_slab_set) {
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(out_slab_set);
  *out_slab_set = NULL;

  id4_pipeline_parameter_slab_load_t* loads = NULL;
  IREE_RETURN_IF_ERROR(id4_pipeline_plan_make_parameter_slab_loads(
      plan, host_allocator, &loads));
  iree_status_t status = id4_pipeline_parameter_slab_set_load(
      options, plan->parameter_slab_count, loads,
      plan->parameter_load_step_count, plan->parameter_load_steps,
      plan->stage_name, host_allocator, out_slab_set);
  iree_allocator_free(host_allocator, loads);
  return status;
}

iree_status_t id4_pipeline_plan_prepare_parameter_slabs(
    const id4_pipeline_plan_t* plan,
    const id4_pipeline_parameter_slab_set_load_options_t* options,
    iree_allocator_t host_allocator,
    id4_pipeline_parameter_slab_set_t** out_slab_set) {
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(out_slab_set);
  *out_slab_set = NULL;

  id4_pipeline_parameter_slab_load_t* loads = NULL;
  IREE_RETURN_IF_ERROR(id4_pipeline_plan_make_parameter_slab_loads(
      plan, host_allocator, &loads));
  iree_status_t status = id4_pipeline_parameter_slab_set_prepare(
      options, plan->parameter_slab_count, loads,
      plan->parameter_load_step_count, plan->parameter_load_steps,
      plan->stage_name, host_allocator, out_slab_set);
  iree_allocator_free(host_allocator, loads);
  return status;
}

static iree_host_size_t id4_pipeline_plan_find_load_group_first_region(
    const id4_pipeline_plan_t* plan, iree_host_size_t group_index) {
  for (iree_host_size_t i = 0; i < plan->region_count; ++i) {
    const id4_pipeline_region_plan_t* region = &plan->regions[i];
    for (iree_host_size_t j = 0; j < region->parameter_load_group_count; ++j) {
      if (region->parameter_load_groups[j] == group_index) return i;
    }
  }
  return IREE_HOST_SIZE_MAX;
}

iree_status_t id4_pipeline_plan_create_parameter_slab_issue_context(
    const id4_pipeline_plan_t* plan,
    id4_pipeline_parameter_slab_set_t* slab_set,
    iree_allocator_t host_allocator,
    id4_pipeline_parameter_slab_issue_context_t** out_context) {
  IREE_ASSERT_ARGUMENT(plan);
  return id4_pipeline_parameter_slab_issue_context_create(
      slab_set, plan->parameter_load_step_count, plan->parameter_load_steps,
      host_allocator, out_context);
}

iree_status_t id4_pipeline_plan_submit_parameter_load_group(
    const id4_pipeline_plan_t* plan,
    id4_pipeline_parameter_slab_issue_context_t* context,
    iree_host_size_t group_index, iree_host_size_t submit_region_id,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  IREE_ASSERT_ARGUMENT(plan);
  id4_pipeline_parameter_load_group_context_t group_context = {
      // Plan-local load group ordinal.
      .group_index = group_index,
      // First planned region that consumes the load group.
      .first_consumer_region_id =
          id4_pipeline_plan_find_load_group_first_region(plan, group_index),
      // Region currently submitting the load group.
      .submit_region_id = submit_region_id,
  };
  return id4_pipeline_parameter_slab_issue_context_submit_load_group(
      context, plan->parameter_load_step_count, plan->parameter_load_steps,
      group_context, plan->stage_name, diagnostics_sink);
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

static iree_string_view_t id4_pipeline_memory_slab_scope_format(
    id4_pipeline_memory_slab_scope_t scope) {
  switch (scope) {
    case ID4_PIPELINE_MEMORY_SLAB_SCOPE_REGION_LOCAL:
      return IREE_SV("region_local");
    case ID4_PIPELINE_MEMORY_SLAB_SCOPE_PLAN_SHARED:
      return IREE_SV("plan_shared");
    default:
      return IREE_SV("invalid");
  }
}

static iree_status_t id4_pipeline_plan_append_parameter_load_step_kind_json(
    iree_string_builder_t* builder,
    id4_pipeline_parameter_load_step_kind_t kind) {
  switch (kind) {
    case ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_GATHER:
      return id4_pipeline_plan_append_json_string(builder, IREE_SV("gather"));
    case ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_FP8_E4M3_SCALED_TO_BF16:
      return id4_pipeline_plan_append_json_string(
          builder, IREE_SV("encode_fp8_e4m3_scaled_to_bf16"));
    case ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_BF16_LINEAR_RHS_TILE:
      return id4_pipeline_plan_append_json_string(
          builder, IREE_SV("encode_bf16_linear_rhs_tile"));
    case ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_FP8_E4M3_SCALED_TO_BF16_LINEAR_RHS_TILE:
      return id4_pipeline_plan_append_json_string(
          builder, IREE_SV("encode_fp8_e4m3_scaled_to_bf16_linear_rhs_tile"));
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown parameter load step kind %u",
                              (uint32_t)kind);
  }
}

static iree_status_t id4_pipeline_plan_append_parameter_load_group_kind_json(
    iree_string_builder_t* builder,
    id4_pipeline_parameter_load_group_kind_t kind) {
  switch (kind) {
    case ID4_PIPELINE_PARAMETER_LOAD_GROUP_KIND_GATHER:
      return id4_pipeline_plan_append_json_string(builder, IREE_SV("gather"));
    case ID4_PIPELINE_PARAMETER_LOAD_GROUP_KIND_ENCODE:
      return id4_pipeline_plan_append_json_string(builder, IREE_SV("encode"));
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown parameter load group kind %u",
                              (uint32_t)kind);
  }
}

static iree_status_t id4_pipeline_plan_append_region_statistics_json(
    iree_string_builder_t* builder,
    id4_pipeline_region_statistics_t statistics) {
  return iree_string_builder_append_format(
      builder,
      "{\"operation_count\":%" PRIhsz ",\"dispatch_count\":%" PRIhsz
      ",\"copy_count\":%" PRIhsz ",\"barrier_count\":%" PRIhsz
      ",\"current_epoch\":%u"
      ",\"local_acquire_count\":%" PRIhsz ",\"local_release_count\":%" PRIhsz
      ",\"local_reuse_count\":%" PRIhsz ",\"bound_import_count\":%" PRIhsz
      ",\"local_slab_byte_length\":%" PRIu64
      ",\"local_slab_high_water_mark\":%" PRIu64 "}",
      statistics.operation_count, statistics.dispatch_count,
      statistics.copy_count, statistics.barrier_count, statistics.current_epoch,
      statistics.local_acquire_count, statistics.local_release_count,
      statistics.local_reuse_count, statistics.bound_import_count,
      (uint64_t)statistics.local_slab_byte_length,
      (uint64_t)statistics.local_slab_high_water_mark);
}

static iree_status_t id4_pipeline_plan_append_shape_json(
    iree_string_builder_t* builder, id4_pipeline_tensor_shape_t shape) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "["));
  for (uint32_t i = 0; i < shape.rank; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_format(builder, "%" PRIu64, shape.dims[i]));
  }
  return iree_string_builder_append_cstring(builder, "]");
}

static iree_status_t id4_pipeline_plan_append_parameter_load_sources_json(
    iree_string_builder_t* builder, iree_host_size_t source_count,
    const id4_pipeline_parameter_load_source_t* sources) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "["));
  for (iree_host_size_t i = 0; i < source_count; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    const id4_pipeline_parameter_load_source_t* source = &sources[i];
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(builder, "{\"source_scope\":"));
    IREE_RETURN_IF_ERROR(
        id4_pipeline_plan_append_json_string(builder, source->source_scope));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(builder, ",\"key\":"));
    IREE_RETURN_IF_ERROR(
        id4_pipeline_plan_append_json_string(builder, source->key));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(builder, ",\"dtype\":"));
    IREE_RETURN_IF_ERROR(id4_pipeline_plan_append_json_string(
        builder, id4_pipeline_tensor_dtype_format(source->dtype)));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, ",\"byte_length\":%" PRIu64 ",\"shape\":",
        (uint64_t)source->byte_length));
    IREE_RETURN_IF_ERROR(
        id4_pipeline_plan_append_shape_json(builder, source->shape));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "}"));
  }
  return iree_string_builder_append_cstring(builder, "]");
}

static iree_status_t id4_pipeline_plan_append_program_shape_json(
    iree_string_builder_t* builder, id4_pipeline_program_shape_t shape) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "["));
  for (uint32_t i = 0; i < shape.rank; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_format(builder, "%" PRIu64, shape.dims[i]));
  }
  return iree_string_builder_append_cstring(builder, "]");
}

static iree_status_t id4_pipeline_plan_append_u32_array3_json(
    iree_string_builder_t* builder, const uint32_t values[3]) {
  return iree_string_builder_append_format(builder, "[%u,%u,%u]", values[0],
                                           values[1], values[2]);
}

static iree_status_t id4_pipeline_plan_append_host_size_array_json(
    iree_string_builder_t* builder, iree_host_size_t count,
    const iree_host_size_t* values) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "["));
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_format(builder, "%" PRIhsz, values[i]));
  }
  return iree_string_builder_append_cstring(builder, "]");
}

static iree_string_view_t id4_pipeline_plan_program_dtype_format(
    id4_pipeline_program_dtype_t dtype) {
  switch (dtype) {
    case ID4_PIPELINE_PROGRAM_DTYPE_F32:
      return IREE_SV("f32");
    case ID4_PIPELINE_PROGRAM_DTYPE_F16:
      return IREE_SV("f16");
    case ID4_PIPELINE_PROGRAM_DTYPE_BF16:
      return IREE_SV("bf16");
    case ID4_PIPELINE_PROGRAM_DTYPE_I32:
      return IREE_SV("i32");
    case ID4_PIPELINE_PROGRAM_DTYPE_U32:
      return IREE_SV("u32");
    case ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3:
      return IREE_SV("f8e4m3");
    default:
      return IREE_SV("invalid");
  }
}

static iree_string_view_t id4_pipeline_plan_program_tensor_access_format(
    id4_pipeline_program_tensor_access_flags_t access) {
  const id4_pipeline_program_tensor_access_flags_t read_write =
      ID4_PIPELINE_PROGRAM_TENSOR_ACCESS_READ |
      ID4_PIPELINE_PROGRAM_TENSOR_ACCESS_WRITE;
  if (iree_all_bits_set(access, read_write)) return IREE_SV("read_write");
  if (iree_all_bits_set(access, ID4_PIPELINE_PROGRAM_TENSOR_ACCESS_READ)) {
    return IREE_SV("read");
  }
  if (iree_all_bits_set(access, ID4_PIPELINE_PROGRAM_TENSOR_ACCESS_WRITE)) {
    return IREE_SV("write");
  }
  return IREE_SV("invalid");
}

static iree_status_t id4_pipeline_plan_append_config_bindings_json(
    iree_string_builder_t* builder, iree_host_size_t binding_count,
    const id4_pipeline_plan_config_binding_t* bindings) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "["));
  for (iree_host_size_t i = 0; i < binding_count; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    const id4_pipeline_plan_config_binding_t* binding = &bindings[i];
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
  return iree_string_builder_append_cstring(builder, "]");
}

static bool id4_pipeline_plan_has_captured_tap(const id4_pipeline_plan_t* plan,
                                               iree_string_view_t tap_name) {
  for (iree_host_size_t i = 0; i < plan->diagnostic_tap_count; ++i) {
    if (iree_string_view_equal(plan->diagnostic_taps[i].name, tap_name)) {
      return true;
    }
  }
  return false;
}

static iree_host_size_t id4_pipeline_plan_source_program_dispatch_count(
    const id4_pipeline_program_t* program) {
  iree_host_size_t dispatch_count = 0;
  const iree_host_size_t operation_count =
      id4_pipeline_program_operation_count(program);
  for (iree_host_size_t i = 0; i < operation_count; ++i) {
    const id4_pipeline_program_op_t* op =
        id4_pipeline_program_operation_at(program, i);
    if (op && op->kind == ID4_PIPELINE_PROGRAM_OP_KIND_DISPATCH_LOOM) {
      ++dispatch_count;
    }
  }
  return dispatch_count;
}

static iree_status_t id4_pipeline_plan_append_program_binding_json(
    const id4_pipeline_program_t* program, iree_string_builder_t* builder,
    iree_host_size_t binding_index,
    const id4_pipeline_program_dispatch_binding_t* binding) {
  const id4_pipeline_program_tensor_record_t* tensor =
      id4_pipeline_program_tensor_at(program, binding->tensor.ordinal);
  if (!tensor) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "program dispatch binding tensor %u is missing",
                            binding->tensor.ordinal);
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "{\"index\":%" PRIhsz ",\"tensor_ordinal\":%u,\"name\":",
      binding_index, binding->tensor.ordinal));
  IREE_RETURN_IF_ERROR(
      id4_pipeline_plan_append_json_string(builder, tensor->name));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, ",\"access\":"));
  IREE_RETURN_IF_ERROR(id4_pipeline_plan_append_json_string(
      builder,
      id4_pipeline_plan_program_tensor_access_format(binding->access)));
  if (iree_all_bits_set(
          binding->flags,
          ID4_PIPELINE_PROGRAM_DISPATCH_BINDING_FLAG_WRITE_RANGE)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        ",\"write_range\":{\"offset\":%" PRIu64 ",\"length\":%" PRIu64 "}",
        (uint64_t)binding->write_range.offset,
        (uint64_t)binding->write_range.length));
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, ",\"dtype\":"));
  IREE_RETURN_IF_ERROR(id4_pipeline_plan_append_json_string(
      builder, id4_pipeline_plan_program_dtype_format(tensor->dtype)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, ",\"byte_length\":%" PRIu64 ",\"shape\":",
      (uint64_t)tensor->byte_length));
  IREE_RETURN_IF_ERROR(
      id4_pipeline_plan_append_program_shape_json(builder, tensor->shape));
  return iree_string_builder_append_cstring(builder, "}");
}

static iree_status_t id4_pipeline_plan_append_program_dispatch_json(
    const id4_pipeline_program_t* program, iree_string_builder_t* builder,
    const id4_pipeline_program_op_t* op, iree_host_size_t dispatch_ordinal,
    iree_host_size_t region_id, iree_host_size_t region_operation_ordinal) {
  const id4_pipeline_program_dispatch_loom_op_t* dispatch =
      &op->payload.dispatch_loom;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      "{\"dispatch_ordinal\":%" PRIhsz ",\"operation_ordinal\":%" PRIhsz
      ",\"region_id\":%" PRIhsz ",\"region_operation_ordinal\":%" PRIhsz
      ",\"name\":",
      dispatch_ordinal, op->ordinal, region_id, region_operation_ordinal));
  IREE_RETURN_IF_ERROR(
      id4_pipeline_plan_append_json_string(builder, dispatch->name));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, ",\"module_path\":"));
  IREE_RETURN_IF_ERROR(id4_pipeline_plan_append_json_string(
      builder, dispatch->kernel.module_path));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, ",\"function_name\":"));
  IREE_RETURN_IF_ERROR(id4_pipeline_plan_append_json_string(
      builder, dispatch->kernel.function_name));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, ",\"config_bindings\":"));
  IREE_RETURN_IF_ERROR(id4_pipeline_plan_append_config_bindings_json(
      builder, dispatch->config_binding_count, dispatch->config_bindings));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, ",\"bindings\":["));
  for (iree_host_size_t i = 0; i < dispatch->binding_count; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    IREE_RETURN_IF_ERROR(id4_pipeline_plan_append_program_binding_json(
        program, builder, i, &dispatch->bindings[i]));
  }
  return iree_string_builder_append_cstring(builder, "]}");
}

static iree_status_t id4_pipeline_plan_append_program_json(
    const id4_pipeline_plan_t* plan, iree_string_builder_t* builder) {
  const id4_pipeline_program_t* program = plan->source_program;
  if (!program) return iree_string_builder_append_cstring(builder, "null");

  const iree_host_size_t operation_count =
      id4_pipeline_program_operation_count(program);
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "{\"name\":"));
  IREE_RETURN_IF_ERROR(id4_pipeline_plan_append_json_string(
      builder, id4_pipeline_program_name(program)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      ",\"operation_count\":%" PRIhsz ",\"dispatch_count\":%" PRIhsz
      ",\"dispatches\":[",
      operation_count,
      id4_pipeline_plan_source_program_dispatch_count(program)));

  bool emitted_dispatch = false;
  iree_host_size_t dispatch_ordinal = 0;
  for (iree_host_size_t region_index = 0; region_index < plan->region_count;
       ++region_index) {
    const id4_pipeline_region_plan_t* region = &plan->regions[region_index];
    iree_host_size_t region_operation_ordinal = 0;
    const iree_host_size_t operation_limit =
        region->source_operation_offset + region->source_operation_count;
    for (iree_host_size_t i = region->source_operation_offset;
         i < operation_limit; ++i) {
      const id4_pipeline_program_op_t* op =
          id4_pipeline_program_operation_at(program, i);
      if (!op) continue;
      switch (op->kind) {
        case ID4_PIPELINE_PROGRAM_OP_KIND_DISPATCH_LOOM:
          if (emitted_dispatch) {
            IREE_RETURN_IF_ERROR(
                iree_string_builder_append_cstring(builder, ","));
          }
          IREE_RETURN_IF_ERROR(id4_pipeline_plan_append_program_dispatch_json(
              program, builder, op, dispatch_ordinal, region_index,
              region_operation_ordinal));
          emitted_dispatch = true;
          ++dispatch_ordinal;
          ++region_operation_ordinal;
          break;
        case ID4_PIPELINE_PROGRAM_OP_KIND_BARRIER:
          ++region_operation_ordinal;
          break;
        case ID4_PIPELINE_PROGRAM_OP_KIND_TAP:
          if (id4_pipeline_plan_has_captured_tap(plan, op->payload.tap.name)) {
            region_operation_ordinal += 3;
          }
          break;
        default:
          break;
      }
    }
  }
  return iree_string_builder_append_cstring(builder, "]}");
}

iree_status_t id4_pipeline_plan_format_json(const id4_pipeline_plan_t* plan,
                                            iree_string_builder_t* builder) {
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(builder);
  const id4_pipeline_plan_statistics_t statistics =
      id4_pipeline_plan_statistics(plan);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "{"));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "\"stage\":"));
  IREE_RETURN_IF_ERROR(
      id4_pipeline_plan_append_json_string(builder, plan->stage_name));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      ",\"device_group\":{\"device_count\":%" PRIhsz
      ",\"topology_fingerprint\":\"%08x\"}",
      iree_hal_device_group_device_count(plan->device_group),
      id4_pipeline_plan_calculate_topology_fingerprint(plan)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      ",\"statistics\":{\"parameter_slab_byte_length\":%" PRIu64
      ",\"largest_parameter_slab_byte_length\":%" PRIu64
      ",\"parameter_source_byte_length\":%" PRIu64
      ",\"parameter_direct_source_byte_length\":%" PRIu64
      ",\"parameter_encoded_source_byte_length\":%" PRIu64
      ",\"parameter_gather_load_step_count\":%" PRIhsz
      ",\"parameter_encode_load_step_count\":%" PRIhsz
      ",\"parameter_load_group_count\":%" PRIhsz
      ",\"parameter_gather_load_group_count\":%" PRIhsz
      ",\"parameter_encode_load_group_count\":%" PRIhsz
      ",\"constant_slab_byte_length\":%" PRIu64
      ",\"memory_slab_byte_length\":%" PRIu64
      ",\"memory_slab_high_water_mark\":%" PRIu64
      ",\"shared_tensor_byte_length\":%" PRIu64
      ",\"boundary_tensor_byte_length\":%" PRIu64
      ",\"diagnostic_tap_byte_length\":%" PRIu64 ",\"kernel_count\":%" PRIhsz
      ",\"region_count\":%" PRIhsz ",\"shared_tensor_count\":%" PRIhsz
      ",\"operation_count\":%" PRIhsz ",\"dispatch_count\":%" PRIhsz "}",
      (uint64_t)statistics.parameter_slab_byte_length,
      (uint64_t)statistics.largest_parameter_slab_byte_length,
      (uint64_t)statistics.parameter_source_byte_length,
      (uint64_t)statistics.parameter_direct_source_byte_length,
      (uint64_t)statistics.parameter_encoded_source_byte_length,
      statistics.parameter_gather_load_step_count,
      statistics.parameter_encode_load_step_count,
      statistics.parameter_load_group_count,
      statistics.parameter_gather_load_group_count,
      statistics.parameter_encode_load_group_count,
      (uint64_t)statistics.constant_slab_byte_length,
      (uint64_t)statistics.memory_slab_byte_length,
      (uint64_t)statistics.memory_slab_high_water_mark,
      (uint64_t)statistics.shared_tensor_byte_length,
      (uint64_t)statistics.boundary_tensor_byte_length,
      (uint64_t)statistics.diagnostic_tap_byte_length, statistics.kernel_count,
      statistics.region_count, statistics.shared_tensor_count,
      statistics.operation_count, statistics.dispatch_count));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, ",\"placements\":["));
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
        builder, ",\"placement_id\":%u,\"binding_slot\":%u,\"target_params\":",
        slab->placement_id, slab->binding_slot));
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
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
      builder, "],\"parameter_load_steps\":["));
  for (iree_host_size_t i = 0; i < plan->parameter_load_step_count; ++i) {
    const id4_pipeline_parameter_load_step_t* step =
        &plan->parameter_load_steps[i];
    if (i != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(builder, "{\"name\":"));
    IREE_RETURN_IF_ERROR(
        id4_pipeline_plan_append_json_string(builder, step->name));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(builder, ",\"source_scope\":"));
    IREE_RETURN_IF_ERROR(
        id4_pipeline_plan_append_json_string(builder, step->source_scope));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(builder, ",\"kind\":"));
    IREE_RETURN_IF_ERROR(id4_pipeline_plan_append_parameter_load_step_kind_json(
        builder, step->kind));
    if (step->readiness_group_key !=
        ID4_PIPELINE_PARAMETER_LOAD_READINESS_GROUP_NONE) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, ",\"readiness_group_key\":%" PRIhsz,
          step->readiness_group_key));
    }
    if (step->source_count != 0) {
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(builder, ",\"sources\":"));
      IREE_RETURN_IF_ERROR(id4_pipeline_plan_append_parameter_load_sources_json(
          builder, step->source_count, step->sources));
    }
    if (step->request_indices) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
          builder, ",\"request_indices\":["));
      for (iree_host_size_t j = 0; j < step->request_count; ++j) {
        if (j != 0) {
          IREE_RETURN_IF_ERROR(
              iree_string_builder_append_cstring(builder, ","));
        }
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, "%" PRIhsz, step->request_indices[j]));
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "]"));
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        ",\"target_slab_index\":%" PRIhsz ",\"request_offset\":%" PRIhsz
        ",\"request_count\":%" PRIhsz "}",
        step->target_slab_index, step->request_offset, step->request_count));
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
      builder, "],\"parameter_load_groups\":["));
  iree_host_size_t parameter_load_group_count = 0;
  IREE_RETURN_IF_ERROR(id4_pipeline_plan_parameter_load_group_count(
      plan, &parameter_load_group_count));
  for (iree_host_size_t i = 0; i < parameter_load_group_count; ++i) {
    id4_pipeline_parameter_load_group_t group;
    IREE_RETURN_IF_ERROR(
        id4_pipeline_plan_parameter_load_group_at(plan, i, &group));
    const id4_pipeline_parameter_load_step_t* step =
        &plan->parameter_load_steps[group.step_offset];
    const iree_host_size_t first_consumer_region =
        id4_pipeline_plan_find_load_group_first_region(plan, i);
    if (i != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "{\"id\":%" PRIhsz ",\"kind\":", i));
    IREE_RETURN_IF_ERROR(
        id4_pipeline_plan_append_parameter_load_group_kind_json(builder,
                                                                group.kind));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        ",\"step_offset\":%" PRIhsz ",\"step_count\":%" PRIhsz
        ",\"target_slab_index\":%" PRIhsz,
        group.step_offset, group.step_count, group.target_slab_index));
    if (step->readiness_group_key !=
        ID4_PIPELINE_PARAMETER_LOAD_READINESS_GROUP_NONE) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, ",\"readiness_group_key\":%" PRIhsz,
          step->readiness_group_key));
    }
    if (first_consumer_region == IREE_HOST_SIZE_MAX) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
          builder, ",\"first_consumer_region_id\":null"));
    } else {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, ",\"first_consumer_region_id\":%" PRIhsz,
          first_consumer_region));
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "}"));
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "],\"constant_slabs\":["));
  for (iree_host_size_t i = 0; i < plan->constant_slab_count; ++i) {
    const id4_pipeline_constant_slab_plan_t* slab = &plan->constant_slabs[i];
    if (i != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "{\"id\":%" PRIhsz ",\"name\":", i));
    IREE_RETURN_IF_ERROR(
        id4_pipeline_plan_append_json_string(builder, slab->name));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, ",\"placement_id\":%u,\"binding_slot\":%u,\"target_params\":",
        slab->placement_id, slab->binding_slot));
    IREE_RETURN_IF_ERROR(id4_pipeline_plan_append_buffer_params_json(
        builder, slab->target_params));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        ",\"byte_length\":%" PRIu64 ",\"alignment\":%" PRIu64
        ",\"request_count\":%" PRIhsz ",\"requests\":[",
        (uint64_t)slab->byte_length, (uint64_t)slab->alignment,
        slab->request_count));
    for (iree_host_size_t j = 0; j < slab->request_count; ++j) {
      const id4_pipeline_constant_request_t* request = &slab->requests[j];
      if (j != 0) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
      }
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(builder, "{\"name\":"));
      IREE_RETURN_IF_ERROR(
          id4_pipeline_plan_append_json_string(builder, request->name));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, ",\"buffer_offset\":%" PRIu64 ",\"length\":%" PRIu64 "}",
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
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(builder, ",\"scope\":"));
    IREE_RETURN_IF_ERROR(id4_pipeline_plan_append_json_string(
        builder, id4_pipeline_memory_slab_scope_format(slab->scope)));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        ",\"region_id\":%u,\"placement_id\":%u,\"binding_slot\":%u,"
        "\"params\":",
        slab->region_id, slab->placement_id, slab->binding_slot));
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
      iree_string_builder_append_cstring(builder, "],\"shared_tensors\":["));
  for (iree_host_size_t i = 0; i < plan->shared_tensor_count; ++i) {
    const id4_pipeline_shared_tensor_plan_t* tensor = &plan->shared_tensors[i];
    if (i != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "{\"id\":%" PRIhsz ",\"name\":", i));
    IREE_RETURN_IF_ERROR(
        id4_pipeline_plan_append_json_string(builder, tensor->layout.name));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(builder, ",\"dtype\":"));
    IREE_RETURN_IF_ERROR(id4_pipeline_plan_append_json_string(
        builder, id4_pipeline_tensor_dtype_format(tensor->layout.dtype)));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        ",\"program_tensor_ordinal\":%u,\"memory_slab_index\":%" PRIhsz
        ",\"offset\":%" PRIu64 ",\"byte_length\":%" PRIu64
        ",\"alignment\":%" PRIu64
        ",\"acquire_region_id\":%u"
        ",\"last_use_region_id\":%u,\"shape\":",
        tensor->program_tensor_ordinal, tensor->memory_slab_index,
        (uint64_t)tensor->offset, (uint64_t)tensor->layout.byte_length,
        (uint64_t)tensor->layout.alignment, tensor->acquire_region_id,
        tensor->last_use_region_id));
    IREE_RETURN_IF_ERROR(
        id4_pipeline_plan_append_shape_json(builder, tensor->layout.shape));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "}"));
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "],\"boundary_tensors\":["));
  for (iree_host_size_t i = 0; i < plan->boundary_tensor_count; ++i) {
    const id4_pipeline_boundary_tensor_plan_t* tensor =
        &plan->boundary_tensors[i];
    if (i != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "{\"id\":%" PRIhsz ",\"name\":", i));
    IREE_RETURN_IF_ERROR(
        id4_pipeline_plan_append_json_string(builder, tensor->layout.name));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(builder, ",\"dtype\":"));
    IREE_RETURN_IF_ERROR(id4_pipeline_plan_append_json_string(
        builder, id4_pipeline_tensor_dtype_format(tensor->layout.dtype)));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        ",\"flags\":%u,\"placement_id\":%u,\"binding_slot\":%u"
        ",\"byte_length\":%" PRIu64 ",\"alignment\":%" PRIu64 ",\"shape\":",
        tensor->flags, tensor->placement_id, tensor->binding_slot,
        (uint64_t)tensor->layout.byte_length,
        (uint64_t)tensor->layout.alignment));
    IREE_RETURN_IF_ERROR(
        id4_pipeline_plan_append_shape_json(builder, tensor->layout.shape));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "}"));
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
        builder,
        ",\"placement_id\":%u,\"config_bindings\":", kernel->placement_id));
    IREE_RETURN_IF_ERROR(id4_pipeline_plan_append_config_bindings_json(
        builder, kernel->config_binding_count, kernel->config_bindings));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "}"));
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
        ",\"source_operation_offset\":%" PRIhsz
        ",\"source_operation_count\":%" PRIhsz
        ",\"placement_id\":%u,\"binding_capacity\":%" PRIhsz
        ",\"local_binding_slot\":%u,\"local_tensor_alignment\":%" PRIu64
        ",\"statistics\":",
        region->source_operation_offset, region->source_operation_count,
        region->placement_id, region->binding_capacity,
        region->local_binding_slot, (uint64_t)region->local_tensor_alignment));
    IREE_RETURN_IF_ERROR(id4_pipeline_plan_append_region_statistics_json(
        builder, region->statistics));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
        builder, ",\"parameter_load_groups\":"));
    IREE_RETURN_IF_ERROR(id4_pipeline_plan_append_host_size_array_json(
        builder, region->parameter_load_group_count,
        region->parameter_load_groups));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(builder, ",\"local_lifetimes\":["));
    for (iree_host_size_t j = 0; j < region->local_lifetime_count; ++j) {
      const id4_pipeline_region_local_lifetime_t* lifetime =
          &region->local_lifetimes[j];
      if (j != 0) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, "{\"id\":%" PRIhsz ",\"name\":", j));
      IREE_RETURN_IF_ERROR(
          id4_pipeline_plan_append_json_string(builder, lifetime->name));
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(builder, ",\"dtype\":"));
      IREE_RETURN_IF_ERROR(id4_pipeline_plan_append_json_string(
          builder, id4_pipeline_tensor_dtype_format(lifetime->dtype)));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          ",\"ordinal\":%u,\"flags\":%u,\"offset\":%" PRIu64
          ",\"byte_length\":%" PRIu64 ",\"alignment\":%" PRIu64
          ",\"acquire_operation_ordinal\":%" PRIhsz ",\"acquire_epoch\":%u",
          lifetime->ordinal, lifetime->flags, (uint64_t)lifetime->offset,
          (uint64_t)lifetime->byte_length, (uint64_t)lifetime->alignment,
          lifetime->acquire_operation_ordinal, lifetime->acquire_epoch));
      if (lifetime->release_operation_ordinal == IREE_HOST_SIZE_MAX) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
            builder, ",\"release_operation_ordinal\":null"));
      } else {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, ",\"release_operation_ordinal\":%" PRIhsz,
            lifetime->release_operation_ordinal));
      }
      if (lifetime->release_epoch == UINT32_MAX) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
            builder, ",\"release_epoch\":null,\"shape\":"));
      } else {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder,
            ",\"release_epoch\":%u,\"shape\":", lifetime->release_epoch));
      }
      IREE_RETURN_IF_ERROR(
          id4_pipeline_plan_append_shape_json(builder, lifetime->shape));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "}"));
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "]"));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "}"));
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "],\"program\":"));
  IREE_RETURN_IF_ERROR(id4_pipeline_plan_append_program_json(plan, builder));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, ",\"diagnostic_taps\":["));
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
        ",\"placement_id\":%u,\"binding_slot\":%u,\"target_name\":",
        tap->region_id, tap->after_operation_ordinal, tap->placement_id,
        tap->binding_slot));
    IREE_RETURN_IF_ERROR(
        id4_pipeline_plan_append_json_string(builder, tap->target_name));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(builder, ",\"dtype\":"));
    IREE_RETURN_IF_ERROR(id4_pipeline_plan_append_json_string(
        builder, id4_pipeline_tensor_dtype_format(tap->layout.dtype)));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        ",\"byte_length\":%" PRIu64 ",\"alignment\":%" PRIu64 ",\"shape\":",
        (uint64_t)tap->layout.byte_length, (uint64_t)tap->layout.alignment));
    IREE_RETURN_IF_ERROR(
        id4_pipeline_plan_append_shape_json(builder, tap->layout.shape));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "}"));
  }
  return iree_string_builder_append_cstring(builder, "]}");
}
