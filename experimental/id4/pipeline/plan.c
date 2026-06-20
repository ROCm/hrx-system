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
    target->requests = requests;
    for (iree_host_size_t j = 0; j < target->request_count; ++j) {
      requests[j].span = source->requests[j].span;
      IREE_RETURN_IF_ERROR(id4_pipeline_string_clone(
          source->requests[j].key, plan->host_allocator, &requests[j].key));
    }
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

iree_status_t id4_pipeline_plan_load_parameter_slabs(
    const id4_pipeline_plan_t* plan, iree_io_parameter_provider_t* provider,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_allocator_t host_allocator,
    id4_pipeline_parameter_slab_set_t** out_slab_set) {
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(provider);
  IREE_ASSERT_ARGUMENT(out_slab_set);
  *out_slab_set = NULL;

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
    loads[i].slab = slab;
    loads[i].device = iree_hal_device_group_device_at(plan->device_group,
                                                      placement->device_index);
    loads[i].queue_affinity = placement->queue_affinity;
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_slab_set_load(
        provider, wait_semaphore_list, signal_semaphore_list,
        plan->parameter_slab_count, loads, host_allocator, out_slab_set);
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
        ",\"placement_id\":%u,\"target_params\":{\"type\":%" PRIu64
        ",\"access\":%" PRIu64 ",\"usage\":%" PRIu64
        ",\"queue_affinity\":%" PRIu64 ",\"min_alignment\":%" PRIu64
        "},\"byte_length\":%" PRIu64 ",\"alignment\":%" PRIu64
        ",\"request_count\":%" PRIhsz ",\"requests\":[",
        slab->placement_id, (uint64_t)slab->target_params.type,
        (uint64_t)slab->target_params.access,
        (uint64_t)slab->target_params.usage,
        (uint64_t)slab->target_params.queue_affinity,
        (uint64_t)slab->target_params.min_alignment,
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
  return iree_string_builder_append_cstring(builder, "]}");
}
