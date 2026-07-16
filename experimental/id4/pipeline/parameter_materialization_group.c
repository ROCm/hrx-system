// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 WITH LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/parameter_materialization_group.h"

#include <string.h>

#include "experimental/id4/pipeline/parameter_binding.h"
#include "experimental/id4/pipeline/plan.h"

typedef uint32_t id4_pipeline_parameter_materialization_group_state_t;

enum id4_pipeline_parameter_materialization_group_state_e {
  ID4_PIPELINE_PARAMETER_MATERIALIZATION_GROUP_STATE_BUILDING = 1u,
  ID4_PIPELINE_PARAMETER_MATERIALIZATION_GROUP_STATE_READY = 2u,
  ID4_PIPELINE_PARAMETER_MATERIALIZATION_GROUP_STATE_RETIRING = 3u,
  ID4_PIPELINE_PARAMETER_MATERIALIZATION_GROUP_STATE_RETIRED = 4u,
};

struct id4_pipeline_parameter_materialization_group_t {
  // Host allocator owning the packed group allocation.
  iree_allocator_t host_allocator;
  // Explicit lifecycle state governing valid group operations.
  id4_pipeline_parameter_materialization_group_state_t state;
  // Exact plan shared by every adopted materialization.
  const id4_pipeline_plan_t* plan;
  // Number of plan-local parameter domains in |materializations|.
  iree_host_size_t capacity;
  // Number of adopted domains that have not completed retirement.
  iree_host_size_t active_count;
  // Adopted management references indexed by plan-local slab ordinal.
  id4_pipeline_parameter_materialization_t** materializations;
  // Complete binding retained only while the group is ready.
  id4_pipeline_parameter_binding_t* binding;
};

static iree_status_t
id4_pipeline_parameter_materialization_group_validate_semaphore_list(
    iree_hal_semaphore_list_t list) {
  if (list.count == 0) return iree_ok_status();
  if (!list.semaphores || !list.payload_values) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "materialization group last-use semaphore list is incomplete");
  }
  for (iree_host_size_t i = 0; i < list.count; ++i) {
    if (!list.semaphores[i]) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "materialization group last-use semaphore %" PRIhsz " is NULL", i);
    }
  }
  return iree_ok_status();
}

iree_status_t id4_pipeline_parameter_materialization_group_create(
    const id4_pipeline_plan_t* plan, iree_allocator_t host_allocator,
    id4_pipeline_parameter_materialization_group_t** out_group) {
  IREE_ASSERT_ARGUMENT(out_group);
  *out_group = NULL;
  if (!plan) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "materialization group plan is required");
  }
  const iree_host_size_t capacity =
      id4_pipeline_plan_parameter_slab_count(plan);
  if (capacity == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "materialization group plan has no parameter domains");
  }

  iree_host_size_t materializations_offset = 0;
  iree_host_size_t total_size = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(id4_pipeline_parameter_materialization_group_t), &total_size,
      IREE_STRUCT_FIELD(capacity, id4_pipeline_parameter_materialization_t*,
                        &materializations_offset)));
  id4_pipeline_parameter_materialization_group_t* group = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&group));
  memset(group, 0, total_size);
  group->host_allocator = host_allocator;
  group->state = ID4_PIPELINE_PARAMETER_MATERIALIZATION_GROUP_STATE_BUILDING;
  group->plan = plan;
  id4_pipeline_plan_retain((id4_pipeline_plan_t*)group->plan);
  group->capacity = capacity;
  group->materializations =
      (id4_pipeline_parameter_materialization_t**)((uint8_t*)group +
                                                   materializations_offset);
  *out_group = group;
  return iree_ok_status();
}

iree_status_t id4_pipeline_parameter_materialization_group_adopt(
    id4_pipeline_parameter_materialization_group_t* group,
    id4_pipeline_parameter_materialization_t** inout_materialization) {
  if (!group) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "materialization group is required");
  }
  if (!inout_materialization || !*inout_materialization) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "published materialization is required");
  }
  if (group->state !=
      ID4_PIPELINE_PARAMETER_MATERIALIZATION_GROUP_STATE_BUILDING) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "materialization group is no longer accepting domains");
  }

  id4_pipeline_parameter_materialization_binding_t published;
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_materialization_query_binding(
      *inout_materialization, &published));
  if (published.plan != group->plan) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "materialization was acquired from a different plan");
  }
  if (published.slab_index >= group->capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "materialization slab index %" PRIhsz
                            " is outside group capacity %" PRIhsz,
                            published.slab_index, group->capacity);
  }
  if (group->materializations[published.slab_index]) {
    return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                            "parameter slab %" PRIhsz " was already adopted",
                            published.slab_index);
  }

  group->materializations[published.slab_index] = *inout_materialization;
  ++group->active_count;
  *inout_materialization = NULL;
  return iree_ok_status();
}

iree_status_t id4_pipeline_parameter_materialization_group_finalize(
    id4_pipeline_parameter_materialization_group_t* group) {
  if (!group) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "materialization group is required");
  }
  if (group->state !=
      ID4_PIPELINE_PARAMETER_MATERIALIZATION_GROUP_STATE_BUILDING) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "only a building materialization group can be finalized");
  }
  if (group->active_count != group->capacity) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "materialization group requires %" PRIhsz
                            " domains but contains %" PRIhsz,
                            group->capacity, group->active_count);
  }

  const id4_pipeline_parameter_binding_create_options_t options = {
      .structure_size = sizeof(options),
      .plan = group->plan,
      .materialization_count = group->capacity,
      .materializations = group->materializations,
  };
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_binding_create(
      &options, group->host_allocator, &group->binding));
  group->state = ID4_PIPELINE_PARAMETER_MATERIALIZATION_GROUP_STATE_READY;
  return iree_ok_status();
}

id4_pipeline_parameter_slab_set_t*
id4_pipeline_parameter_materialization_group_parameter_slabs(
    const id4_pipeline_parameter_materialization_group_t* group) {
  return group && group->state ==
                      ID4_PIPELINE_PARAMETER_MATERIALIZATION_GROUP_STATE_READY
             ? id4_pipeline_parameter_binding_slabs(group->binding)
             : NULL;
}

iree_hal_semaphore_list_t
id4_pipeline_parameter_materialization_group_readiness_semaphore_list(
    const id4_pipeline_parameter_materialization_group_t* group) {
  return group && group->state ==
                      ID4_PIPELINE_PARAMETER_MATERIALIZATION_GROUP_STATE_READY
             ? id4_pipeline_parameter_binding_readiness_semaphore_list(
                   group->binding)
             : iree_hal_semaphore_list_empty();
}

iree_status_t id4_pipeline_parameter_materialization_group_wait_ready(
    const id4_pipeline_parameter_materialization_group_t* group) {
  if (!group) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "materialization group is required");
  }
  if (group->state !=
      ID4_PIPELINE_PARAMETER_MATERIALIZATION_GROUP_STATE_READY) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "materialization group is not ready");
  }
  return iree_hal_semaphore_list_wait(
      id4_pipeline_parameter_binding_readiness_semaphore_list(group->binding),
      iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE);
}

iree_status_t id4_pipeline_parameter_materialization_group_retire(
    id4_pipeline_parameter_materialization_group_t* group,
    iree_hal_semaphore_list_t last_use_wait_list,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  if (!group) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "materialization group is required");
  }
  IREE_RETURN_IF_ERROR(
      id4_pipeline_parameter_materialization_group_validate_semaphore_list(
          last_use_wait_list));
  IREE_RETURN_IF_ERROR(id4_pipeline_diagnostics_validate_sink(
      diagnostics_sink, IREE_SV("parameter materialization group retirement")));
  if (group->state ==
      ID4_PIPELINE_PARAMETER_MATERIALIZATION_GROUP_STATE_RETIRED) {
    return iree_ok_status();
  }
  if (group->state !=
          ID4_PIPELINE_PARAMETER_MATERIALIZATION_GROUP_STATE_BUILDING &&
      group->state !=
          ID4_PIPELINE_PARAMETER_MATERIALIZATION_GROUP_STATE_READY &&
      group->state !=
          ID4_PIPELINE_PARAMETER_MATERIALIZATION_GROUP_STATE_RETIRING) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "materialization group cannot be retired");
  }

  id4_pipeline_parameter_binding_release(group->binding);
  group->binding = NULL;
  group->state = ID4_PIPELINE_PARAMETER_MATERIALIZATION_GROUP_STATE_RETIRING;

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < group->capacity && iree_status_is_ok(status);
       ++i) {
    id4_pipeline_parameter_materialization_t* materialization =
        group->materializations[i];
    if (!materialization) continue;

    id4_pipeline_parameter_materialization_binding_t published;
    memset(&published, 0, sizeof(published));
    status = id4_pipeline_parameter_materialization_query_binding(
        materialization, &published);
    if (iree_status_is_ok(status)) {
      const iree_hal_semaphore_list_t retirement_wait_list =
          last_use_wait_list.count ? last_use_wait_list
                                   : published.readiness_semaphore_list;
      status = id4_pipeline_parameter_materialization_retire_and_wait(
          materialization, retirement_wait_list, IREE_HAL_DEALLOCA_FLAG_NONE,
          diagnostics_sink);
    }
    if (iree_status_is_ok(status)) {
      id4_pipeline_parameter_materialization_release(materialization);
      group->materializations[i] = NULL;
      --group->active_count;
    }
  }
  if (iree_status_is_ok(status) && group->active_count == 0) {
    group->state = ID4_PIPELINE_PARAMETER_MATERIALIZATION_GROUP_STATE_RETIRED;
  }
  return status;
}

void id4_pipeline_parameter_materialization_group_release(
    id4_pipeline_parameter_materialization_group_t* group) {
  if (!group) return;
  if (group->state !=
          ID4_PIPELINE_PARAMETER_MATERIALIZATION_GROUP_STATE_RETIRED ||
      group->active_count != 0 || group->binding) {
    iree_abort();
  }
  const iree_allocator_t host_allocator = group->host_allocator;
  id4_pipeline_plan_release((id4_pipeline_plan_t*)group->plan);
  iree_allocator_free(host_allocator, group);
}
