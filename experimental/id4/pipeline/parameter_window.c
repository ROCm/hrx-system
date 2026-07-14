// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/parameter_window.h"

#include <string.h>

#include "experimental/id4/pipeline/parameter_layout.h"
#include "experimental/id4/pipeline/parameter_slab.h"

struct id4_pipeline_parameter_window_t {
  // Host allocator used for window storage.
  iree_allocator_t host_allocator;
  // Selected semantic program parameter tensor ordinals in plan order.
  uint32_t* parameter_tensor_ordinals;
  // Number of selected semantic program parameter tensor ordinals.
  iree_host_size_t parameter_tensor_count;
  // Compact slabs in original parameter slab order.
  id4_pipeline_parameter_window_slab_t* slabs;
  // Number of compact slabs.
  iree_host_size_t slab_count;
  // Compact requests in original slab/request order.
  id4_pipeline_parameter_window_request_t* requests;
  // Number of compact requests.
  iree_host_size_t request_count;
  // Original load group ordinals in ascending order.
  iree_host_size_t* load_groups;
  // Number of original load group ordinals.
  iree_host_size_t load_group_count;
  // Compact request ordinal by global original request ordinal.
  iree_host_size_t* compact_request_indices_by_global_request;
  // Number of entries in compact_request_indices_by_global_request.
  iree_host_size_t global_request_count;
};

struct id4_pipeline_parameter_window_schedule_t {
  // Host allocator used for schedule storage.
  iree_allocator_t host_allocator;
  // Plan retained for borrowed string views, sources, and device group access.
  id4_pipeline_plan_t* plan;
  // Owned provider scope for execution-layout archive gathers.
  iree_string_view_t source_scope;
  // Compact parameter slab plans in schedule load order.
  id4_pipeline_parameter_slab_plan_t* slabs;
  // Compact provider request tables parallel to slabs.
  id4_pipeline_parameter_request_table_t* request_tables;
  // Compact parameter slab loads in schedule load order.
  id4_pipeline_parameter_slab_load_t* loads;
  // Number of compact parameter slab loads.
  iree_host_size_t load_count;
  // Compact parameter requests referenced by slabs.
  id4_pipeline_parameter_request_t* requests;
  // Number of compact parameter requests.
  iree_host_size_t request_count;
  // Compact load steps in compact load group order.
  id4_pipeline_parameter_load_step_t* load_steps;
  // Number of compact load steps.
  iree_host_size_t load_step_count;
  // Rebased request-index storage used by compact indexed gather steps.
  iree_host_size_t* request_indices;
  // Number of entries in request_indices.
  iree_host_size_t request_index_count;
  // Original plan load group ordinal by compact load group ordinal.
  iree_host_size_t* original_load_groups;
  // Number of compact load groups.
  iree_host_size_t load_group_count;
  // Compact load group ordinal by original plan load group ordinal.
  iree_host_size_t* compact_load_groups_by_original;
  // Number of entries in compact_load_groups_by_original.
  iree_host_size_t original_load_group_count;
};

typedef struct id4_pipeline_parameter_window_build_state_t {
  // Original plan used while building the compact window.
  const id4_pipeline_plan_t* plan;
  // Prefix sum of global request ordinals by original slab index.
  iree_host_size_t* global_request_offsets_by_slab;
  // Number of entries in global_request_offsets_by_slab.
  iree_host_size_t global_request_offset_count;
  // True when a load group is used by the window.
  uint8_t* load_group_used_bits;
  // Number of entries in load_group_used_bits.
  iree_host_size_t load_group_count;
  // True when a global parameter request is used by the window.
  uint8_t* request_used_bits;
  // Number of entries in request_used_bits.
  iree_host_size_t global_request_count;
  // Parameter tensor ordinal by global original request ordinal.
  iree_host_size_t* parameter_tensor_indices_by_global_request;
  // True when a plan parameter tensor is selected by the window.
  uint8_t* parameter_tensor_used_bits;
  // Number of plan parameter tensors.
  iree_host_size_t parameter_tensor_count;
} id4_pipeline_parameter_window_build_state_t;

static iree_status_t id4_pipeline_parameter_window_validate_options_size(
    iree_host_size_t actual_size, iree_host_size_t expected_size,
    iree_string_view_t options_name) {
  if (actual_size >= expected_size) return iree_ok_status();
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "%.*s options structure size %" PRIhsz
                          " is smaller than expected %" PRIhsz,
                          (int)options_name.size, options_name.data,
                          actual_size, expected_size);
}

static iree_status_t id4_pipeline_parameter_window_validate_options(
    const id4_pipeline_parameter_window_create_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter window options are required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_window_validate_options_size(
      options->structure_size, sizeof(*options), IREE_SV("parameter window")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "parameter window extension structures are not supported");
  }
  if (!options->plan) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter window plan is required");
  }
  if (options->parameter_tensor_count != 0 &&
      !options->parameter_tensor_ordinals) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter window tensor ordinals are required");
  }
  if (options->parameter_tensor_count == 0 &&
      options->parameter_tensor_ordinals) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter window tensor ordinals require a nonzero count");
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_parameter_window_build_global_offsets(
    const id4_pipeline_plan_t* plan, iree_allocator_t host_allocator,
    iree_host_size_t** out_offsets, iree_host_size_t* out_offset_count,
    iree_host_size_t* out_global_request_count) {
  *out_offsets = NULL;
  *out_offset_count = 0;
  *out_global_request_count = 0;
  const iree_host_size_t slab_count =
      id4_pipeline_plan_parameter_slab_count(plan);
  if (slab_count == 0) return iree_ok_status();

  iree_host_size_t* offsets = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, slab_count + 1, sizeof(offsets[0]), (void**)&offsets));
  iree_host_size_t global_request_count = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < slab_count && iree_status_is_ok(status);
       ++i) {
    offsets[i] = global_request_count;
    const id4_pipeline_parameter_request_table_t* request_table =
        id4_pipeline_plan_parameter_request_table_at(plan, i);
    if (!request_table) {
      status =
          iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                           "parameter request table %" PRIhsz " is missing", i);
      break;
    }
    if (!iree_host_size_checked_add(global_request_count, request_table->count,
                                    &global_request_count)) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "parameter window request count overflow");
      break;
    }
  }
  if (iree_status_is_ok(status)) {
    offsets[slab_count] = global_request_count;
    *out_offsets = offsets;
    *out_offset_count = slab_count + 1;
    *out_global_request_count = global_request_count;
  } else {
    iree_allocator_free(host_allocator, offsets);
  }
  return status;
}

static iree_status_t id4_pipeline_parameter_window_global_request_index(
    const id4_pipeline_parameter_window_build_state_t* state,
    iree_host_size_t slab_index, iree_host_size_t slab_request_index,
    iree_host_size_t* out_global_request_index) {
  *out_global_request_index = IREE_HOST_SIZE_MAX;
  if (slab_index + 1 >= state->global_request_offset_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "parameter load step target slab %" PRIhsz
                            " exceeds slab count %" PRIhsz,
                            slab_index,
                            state->global_request_offset_count == 0
                                ? 0
                                : state->global_request_offset_count - 1);
  }
  const iree_host_size_t slab_request_count =
      state->global_request_offsets_by_slab[slab_index + 1] -
      state->global_request_offsets_by_slab[slab_index];
  if (slab_request_index >= slab_request_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "parameter load step request %" PRIhsz
                            " exceeds slab %" PRIhsz " request count %" PRIhsz,
                            slab_request_index, slab_index, slab_request_count);
  }
  *out_global_request_index =
      state->global_request_offsets_by_slab[slab_index] + slab_request_index;
  return iree_ok_status();
}

static iree_status_t id4_pipeline_parameter_window_step_request_index(
    const id4_pipeline_parameter_load_step_t* step,
    iree_host_size_t request_ordinal, iree_host_size_t* out_request_index) {
  *out_request_index = IREE_HOST_SIZE_MAX;
  if (request_ordinal >= step->request_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "parameter load step request ordinal %" PRIhsz
                            " exceeds request count %" PRIhsz,
                            request_ordinal, step->request_count);
  }
  if (step->request_indices) {
    *out_request_index = step->request_indices[request_ordinal];
    return iree_ok_status();
  }
  if (!iree_host_size_checked_add(step->request_offset, request_ordinal,
                                  out_request_index)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "parameter load step request index overflow");
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_parameter_window_mark_selected_tensors(
    const id4_pipeline_parameter_window_create_options_t* options,
    id4_pipeline_parameter_window_build_state_t* state) {
  for (iree_host_size_t ordinal_index = 0;
       ordinal_index < options->parameter_tensor_count; ++ordinal_index) {
    const uint32_t program_tensor_ordinal =
        options->parameter_tensor_ordinals[ordinal_index];
    iree_host_size_t parameter_tensor_index = IREE_HOST_SIZE_MAX;
    for (iree_host_size_t i = 0; i < state->parameter_tensor_count; ++i) {
      const id4_pipeline_parameter_tensor_plan_t* tensor =
          id4_pipeline_plan_parameter_tensor_at(state->plan, i);
      if (tensor && tensor->program_tensor_ordinal == program_tensor_ordinal) {
        parameter_tensor_index = i;
        break;
      }
    }
    if (parameter_tensor_index == IREE_HOST_SIZE_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "parameter window program tensor %" PRIu32
                              " is not a planned parameter tensor",
                              program_tensor_ordinal);
    }
    if (state->parameter_tensor_used_bits[parameter_tensor_index]) {
      return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                              "parameter window program tensor %" PRIu32
                              " is selected more than once",
                              program_tensor_ordinal);
    }
    state->parameter_tensor_used_bits[parameter_tensor_index] = 1;
    const id4_pipeline_parameter_tensor_plan_t* tensor =
        id4_pipeline_plan_parameter_tensor_at(state->plan,
                                              parameter_tensor_index);
    for (iree_host_size_t i = 0; i < tensor->request_count; ++i) {
      const iree_host_size_t global_request_index =
          tensor->global_request_offset + i;
      if (global_request_index >= state->global_request_count) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "parameter window tensor %.*s request range exceeds the plan",
            (int)tensor->layout.name.size, tensor->layout.name.data);
      }
      state->request_used_bits[global_request_index] = 1;
    }
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_parameter_window_mark_used_groups(
    id4_pipeline_parameter_window_build_state_t* state) {
  for (iree_host_size_t group_index = 0; group_index < state->load_group_count;
       ++group_index) {
    id4_pipeline_parameter_load_group_t group;
    IREE_RETURN_IF_ERROR(id4_pipeline_plan_parameter_load_group_at(
        state->plan, group_index, &group));
    bool group_used = false;
    for (iree_host_size_t step_ordinal = 0;
         step_ordinal < group.step_count && !group_used; ++step_ordinal) {
      const iree_host_size_t step_index = group.step_offset + step_ordinal;
      const id4_pipeline_parameter_load_step_t* step =
          id4_pipeline_plan_parameter_load_step_at(state->plan, step_index);
      if (!step) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "parameter load step %" PRIhsz " is missing",
                                step_index);
      }
      for (iree_host_size_t request_ordinal = 0;
           request_ordinal < step->request_count; ++request_ordinal) {
        iree_host_size_t slab_request_index = IREE_HOST_SIZE_MAX;
        IREE_RETURN_IF_ERROR(id4_pipeline_parameter_window_step_request_index(
            step, request_ordinal, &slab_request_index));
        iree_host_size_t global_request_index = IREE_HOST_SIZE_MAX;
        IREE_RETURN_IF_ERROR(id4_pipeline_parameter_window_global_request_index(
            state, step->target_slab_index, slab_request_index,
            &global_request_index));
        if (state->request_used_bits[global_request_index]) {
          group_used = true;
          break;
        }
      }
    }
    state->load_group_used_bits[group_index] = group_used ? 1 : 0;
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_parameter_window_map_parameter_tensors(
    id4_pipeline_parameter_window_build_state_t* state,
    iree_allocator_t host_allocator) {
  const iree_host_size_t parameter_tensor_count =
      id4_pipeline_plan_parameter_tensor_count(state->plan);
  if (parameter_tensor_count == 0 || state->global_request_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, state->global_request_count,
      sizeof(state->parameter_tensor_indices_by_global_request[0]),
      (void**)&state->parameter_tensor_indices_by_global_request));
  for (iree_host_size_t i = 0; i < state->global_request_count; ++i) {
    state->parameter_tensor_indices_by_global_request[i] = IREE_HOST_SIZE_MAX;
  }
  for (iree_host_size_t tensor_index = 0; tensor_index < parameter_tensor_count;
       ++tensor_index) {
    const id4_pipeline_parameter_tensor_plan_t* tensor =
        id4_pipeline_plan_parameter_tensor_at(state->plan, tensor_index);
    if (!tensor) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "parameter tensor %" PRIhsz " is missing",
                              tensor_index);
    }
    if (tensor->parameter_slab_index >= state->global_request_offset_count ||
        state->global_request_offset_count - tensor->parameter_slab_index <=
            1) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "parameter tensor %.*s slab index %" PRIhsz
                              " exceeds window slab map count %" PRIhsz,
                              (int)tensor->layout.name.size,
                              tensor->layout.name.data,
                              tensor->parameter_slab_index,
                              state->global_request_offset_count == 0
                                  ? 0
                                  : state->global_request_offset_count - 1);
    }
    iree_host_size_t global_request_offset = 0;
    if (!iree_host_size_checked_add(
            state->global_request_offsets_by_slab[tensor->parameter_slab_index],
            tensor->request_offset, &global_request_offset)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "parameter tensor %.*s global request offset overflows",
          (int)tensor->layout.name.size, tensor->layout.name.data);
    }
    if (tensor->global_request_offset != global_request_offset) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "parameter tensor %.*s global request offset %" PRIhsz
          " does not match slab request offset %" PRIhsz,
          (int)tensor->layout.name.size, tensor->layout.name.data,
          tensor->global_request_offset, global_request_offset);
    }
    for (iree_host_size_t i = 0; i < tensor->request_count; ++i) {
      iree_host_size_t global_request_index = 0;
      if (!iree_host_size_checked_add(tensor->global_request_offset, i,
                                      &global_request_index)) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "parameter tensor %.*s request index overflows",
                                (int)tensor->layout.name.size,
                                tensor->layout.name.data);
      }
      if (global_request_index >= state->global_request_count) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "parameter tensor %.*s request range exceeds global request count",
            (int)tensor->layout.name.size, tensor->layout.name.data);
      }
      if (state->parameter_tensor_indices_by_global_request
              [global_request_index] != IREE_HOST_SIZE_MAX) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "parameter tensor %.*s overlaps another parameter tensor request",
            (int)tensor->layout.name.size, tensor->layout.name.data);
      }
      state->parameter_tensor_indices_by_global_request[global_request_index] =
          tensor_index;
    }
  }
  return iree_ok_status();
}

static iree_host_size_t id4_pipeline_parameter_window_count_used_bits(
    iree_host_size_t bit_count, const uint8_t* bits) {
  iree_host_size_t used_count = 0;
  for (iree_host_size_t i = 0; i < bit_count; ++i) {
    if (bits[i]) ++used_count;
  }
  return used_count;
}

static iree_status_t id4_pipeline_parameter_window_create_empty(
    iree_host_size_t parameter_tensor_count, iree_host_size_t slab_count,
    iree_host_size_t request_count, iree_host_size_t load_group_count,
    iree_host_size_t global_request_count, iree_allocator_t host_allocator,
    id4_pipeline_parameter_window_t** out_window) {
  *out_window = NULL;
  id4_pipeline_parameter_window_t* window = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*window), (void**)&window));
  memset(window, 0, sizeof(*window));
  window->host_allocator = host_allocator;
  window->parameter_tensor_count = parameter_tensor_count;
  window->slab_count = slab_count;
  window->request_count = request_count;
  window->load_group_count = load_group_count;
  window->global_request_count = global_request_count;

  iree_status_t status = iree_ok_status();
  if (parameter_tensor_count != 0) {
    status = iree_allocator_malloc_array(
        host_allocator, parameter_tensor_count,
        sizeof(window->parameter_tensor_ordinals[0]),
        (void**)&window->parameter_tensor_ordinals);
  }
  if (iree_status_is_ok(status) && slab_count != 0) {
    status = iree_allocator_malloc_array(host_allocator, slab_count,
                                         sizeof(window->slabs[0]),
                                         (void**)&window->slabs);
  }
  if (iree_status_is_ok(status) && request_count != 0) {
    status = iree_allocator_malloc_array(host_allocator, request_count,
                                         sizeof(window->requests[0]),
                                         (void**)&window->requests);
  }
  if (iree_status_is_ok(status) && load_group_count != 0) {
    status = iree_allocator_malloc_array(host_allocator, load_group_count,
                                         sizeof(window->load_groups[0]),
                                         (void**)&window->load_groups);
  }
  if (iree_status_is_ok(status) && global_request_count != 0) {
    status = iree_allocator_malloc_array(
        host_allocator, global_request_count,
        sizeof(window->compact_request_indices_by_global_request[0]),
        (void**)&window->compact_request_indices_by_global_request);
  }
  if (iree_status_is_ok(status) && window->parameter_tensor_ordinals) {
    memset(
        window->parameter_tensor_ordinals, 0,
        parameter_tensor_count * sizeof(window->parameter_tensor_ordinals[0]));
  }
  if (iree_status_is_ok(status) && window->slabs) {
    memset(window->slabs, 0, slab_count * sizeof(window->slabs[0]));
  }
  if (iree_status_is_ok(status) && window->requests) {
    memset(window->requests, 0, request_count * sizeof(window->requests[0]));
  }
  if (iree_status_is_ok(status) && window->load_groups) {
    memset(window->load_groups, 0,
           load_group_count * sizeof(window->load_groups[0]));
  }
  if (iree_status_is_ok(status) &&
      window->compact_request_indices_by_global_request) {
    for (iree_host_size_t i = 0; i < global_request_count; ++i) {
      window->compact_request_indices_by_global_request[i] = IREE_HOST_SIZE_MAX;
    }
  }
  if (iree_status_is_ok(status)) {
    *out_window = window;
  } else {
    id4_pipeline_parameter_window_release(window);
  }
  return status;
}

static iree_host_size_t id4_pipeline_parameter_window_count_used_slabs(
    const id4_pipeline_parameter_window_build_state_t* state) {
  iree_host_size_t used_slab_count = 0;
  const iree_host_size_t slab_count =
      id4_pipeline_plan_parameter_slab_count(state->plan);
  for (iree_host_size_t slab_index = 0; slab_index < slab_count; ++slab_index) {
    const iree_host_size_t request_offset =
        state->global_request_offsets_by_slab[slab_index];
    const iree_host_size_t request_limit =
        state->global_request_offsets_by_slab[slab_index + 1];
    for (iree_host_size_t request_index = request_offset;
         request_index < request_limit; ++request_index) {
      if (!state->request_used_bits[request_index]) continue;
      ++used_slab_count;
      break;
    }
  }
  return used_slab_count;
}

static iree_status_t id4_pipeline_parameter_window_pack_parameter_tensors(
    const id4_pipeline_parameter_window_build_state_t* state,
    id4_pipeline_parameter_window_t* window) {
  iree_host_size_t compact_tensor_index = 0;
  for (iree_host_size_t tensor_index = 0;
       tensor_index < state->parameter_tensor_count; ++tensor_index) {
    if (!state->parameter_tensor_used_bits[tensor_index]) continue;
    const id4_pipeline_parameter_tensor_plan_t* tensor =
        id4_pipeline_plan_parameter_tensor_at(state->plan, tensor_index);
    if (!tensor) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "parameter tensor %" PRIhsz " is missing",
                              tensor_index);
    }
    window->parameter_tensor_ordinals[compact_tensor_index++] =
        tensor->program_tensor_ordinal;
  }
  if (compact_tensor_index != window->parameter_tensor_count) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "parameter window tensor count mismatch");
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_parameter_window_pack_requests(
    const id4_pipeline_parameter_window_build_state_t* state,
    id4_pipeline_parameter_window_t* window) {
  iree_host_size_t compact_slab_index = 0;
  iree_host_size_t compact_request_index = 0;
  const iree_host_size_t original_slab_count =
      id4_pipeline_plan_parameter_slab_count(state->plan);
  for (iree_host_size_t original_slab_index = 0;
       original_slab_index < original_slab_count; ++original_slab_index) {
    const id4_pipeline_parameter_slab_plan_t* original_slab =
        id4_pipeline_plan_parameter_slab_at(state->plan, original_slab_index);
    const id4_pipeline_parameter_request_table_t* original_request_table =
        id4_pipeline_plan_parameter_request_table_at(state->plan,
                                                     original_slab_index);
    if (!original_slab || !original_request_table) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "parameter window original slab %" PRIhsz
                              " is missing",
                              original_slab_index);
    }

    id4_pipeline_parameter_window_slab_t compact_slab;
    memset(&compact_slab, 0, sizeof(compact_slab));
    compact_slab.original_slab_index = original_slab_index;
    compact_slab.binding_slot = original_slab->binding_slot;
    compact_slab.target_params = original_slab->target_params;
    compact_slab.alignment = original_slab->alignment;
    compact_slab.request_offset = compact_request_index;

    const iree_host_size_t global_request_offset =
        state->global_request_offsets_by_slab[original_slab_index];
    iree_host_size_t active_parameter_tensor_index = IREE_HOST_SIZE_MAX;
    iree_device_size_t active_parameter_tensor_offset = 0;
    for (iree_host_size_t original_request_index = 0;
         original_request_index < original_request_table->count;
         ++original_request_index) {
      const iree_host_size_t global_request_index =
          global_request_offset + original_request_index;
      if (!state->request_used_bits[global_request_index]) continue;
      const id4_pipeline_parameter_request_t* original_request =
          &original_request_table->values[original_request_index];
      iree_io_parameter_span_t compact_span;
      const iree_host_size_t parameter_tensor_index =
          state->parameter_tensor_indices_by_global_request
              ? state->parameter_tensor_indices_by_global_request
                    [global_request_index]
              : IREE_HOST_SIZE_MAX;
      if (parameter_tensor_index == IREE_HOST_SIZE_MAX) {
        IREE_RETURN_IF_ERROR(id4_pipeline_parameter_slab_pack_span(
            original_request->span.length, compact_slab.alignment,
            &compact_slab.byte_length, &compact_span));
      } else {
        const id4_pipeline_parameter_tensor_plan_t* parameter_tensor =
            id4_pipeline_plan_parameter_tensor_at(state->plan,
                                                  parameter_tensor_index);
        if (!parameter_tensor) {
          return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "parameter tensor %" PRIhsz " is missing",
                                  parameter_tensor_index);
        }
        if (parameter_tensor_index != active_parameter_tensor_index) {
          if (original_request_index != parameter_tensor->request_offset) {
            return iree_make_status(
                IREE_STATUS_FAILED_PRECONDITION,
                "parameter window does not materialize leading request for "
                "parameter tensor %.*s",
                (int)parameter_tensor->layout.name.size,
                parameter_tensor->layout.name.data);
          }
          for (iree_host_size_t i = 0; i < parameter_tensor->request_count;
               ++i) {
            iree_host_size_t tensor_global_request_index = 0;
            if (!iree_host_size_checked_add(
                    parameter_tensor->global_request_offset, i,
                    &tensor_global_request_index)) {
              return iree_make_status(
                  IREE_STATUS_OUT_OF_RANGE,
                  "parameter tensor %.*s request index overflows",
                  (int)parameter_tensor->layout.name.size,
                  parameter_tensor->layout.name.data);
            }
            if (tensor_global_request_index >= state->global_request_count ||
                !state->request_used_bits[tensor_global_request_index]) {
              return iree_make_status(
                  IREE_STATUS_FAILED_PRECONDITION,
                  "parameter window does not materialize all requests for "
                  "parameter tensor %.*s",
                  (int)parameter_tensor->layout.name.size,
                  parameter_tensor->layout.name.data);
            }
          }
          IREE_RETURN_IF_ERROR(id4_pipeline_parameter_slab_pack_span(
              parameter_tensor->layout.byte_length, compact_slab.alignment,
              &compact_slab.byte_length, &compact_span));
          active_parameter_tensor_index = parameter_tensor_index;
          active_parameter_tensor_offset = compact_span.buffer_offset;
        } else {
          memset(&compact_span, 0, sizeof(compact_span));
        }
        if (original_request->span.buffer_offset < parameter_tensor->offset) {
          return iree_make_status(
              IREE_STATUS_OUT_OF_RANGE,
              "parameter tensor %.*s request starts before tensor storage",
              (int)parameter_tensor->layout.name.size,
              parameter_tensor->layout.name.data);
        }
        const iree_device_size_t request_tensor_offset =
            original_request->span.buffer_offset - parameter_tensor->offset;
        if (request_tensor_offset > parameter_tensor->layout.byte_length ||
            original_request->span.length >
                parameter_tensor->layout.byte_length - request_tensor_offset) {
          return iree_make_status(
              IREE_STATUS_OUT_OF_RANGE,
              "parameter tensor %.*s request exceeds tensor storage",
              (int)parameter_tensor->layout.name.size,
              parameter_tensor->layout.name.data);
        }
        if (!iree_device_size_checked_add(active_parameter_tensor_offset,
                                          request_tensor_offset,
                                          &compact_span.buffer_offset)) {
          return iree_make_status(
              IREE_STATUS_OUT_OF_RANGE,
              "parameter tensor %.*s compact request offset overflows",
              (int)parameter_tensor->layout.name.size,
              parameter_tensor->layout.name.data);
        }
        compact_span.length = original_request->span.length;
      }
      compact_span.parameter_offset = original_request->span.parameter_offset;
      window->requests[compact_request_index] =
          (id4_pipeline_parameter_window_request_t){
              // Original slab containing this request.
              .original_slab_index = original_slab_index,
              // Original request index within the original slab.
              .original_request_index = original_request_index,
              // Plan parameter tensor owning this request.
              .parameter_tensor_index = parameter_tensor_index,
              // Global request ordinal in the containing plan.
              .global_request_index = global_request_index,
              // Compact target span.
              .span = compact_span,
          };
      window->compact_request_indices_by_global_request[global_request_index] =
          compact_request_index;
      ++compact_request_index;
      ++compact_slab.request_count;
    }

    if (compact_slab.request_count == 0) continue;
    window->slabs[compact_slab_index++] = compact_slab;
  }
  if (compact_slab_index != window->slab_count ||
      compact_request_index != window->request_count) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "parameter window packed count mismatch");
  }
  return iree_ok_status();
}

static void id4_pipeline_parameter_window_pack_load_groups(
    const id4_pipeline_parameter_window_build_state_t* state,
    id4_pipeline_parameter_window_t* window) {
  iree_host_size_t compact_group_index = 0;
  for (iree_host_size_t group_index = 0; group_index < state->load_group_count;
       ++group_index) {
    if (!state->load_group_used_bits[group_index]) continue;
    window->load_groups[compact_group_index++] = group_index;
  }
}

static iree_status_t id4_pipeline_parameter_window_plan_global_request_index(
    const id4_pipeline_plan_t* plan, iree_host_size_t slab_index,
    iree_host_size_t request_index, iree_host_size_t* out_global_index) {
  *out_global_index = IREE_HOST_SIZE_MAX;
  if (slab_index >= id4_pipeline_plan_parameter_slab_count(plan)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "parameter window slab index %" PRIhsz
                            " is outside the plan",
                            slab_index);
  }
  iree_host_size_t global_index = request_index;
  for (iree_host_size_t i = 0; i < slab_index; ++i) {
    const id4_pipeline_parameter_request_table_t* request_table =
        id4_pipeline_plan_parameter_request_table_at(plan, i);
    if (!request_table ||
        !iree_host_size_checked_add(global_index, request_table->count,
                                    &global_index)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "parameter window global request index overflows");
    }
  }
  *out_global_index = global_index;
  return iree_ok_status();
}

iree_status_t id4_pipeline_parameter_window_create(
    const id4_pipeline_parameter_window_create_options_t* options,
    iree_allocator_t host_allocator,
    id4_pipeline_parameter_window_t** out_window) {
  IREE_ASSERT_ARGUMENT(out_window);
  *out_window = NULL;
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_window_validate_options(options));

  id4_pipeline_parameter_window_build_state_t state;
  memset(&state, 0, sizeof(state));
  state.plan = options->plan;
  state.parameter_tensor_count =
      id4_pipeline_plan_parameter_tensor_count(options->plan);

  iree_status_t status = id4_pipeline_plan_parameter_load_group_count(
      options->plan, &state.load_group_count);
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_window_build_global_offsets(
        options->plan, host_allocator, &state.global_request_offsets_by_slab,
        &state.global_request_offset_count, &state.global_request_count);
  }
  if (iree_status_is_ok(status) && state.load_group_count != 0) {
    status = iree_allocator_malloc_array(host_allocator, state.load_group_count,
                                         sizeof(state.load_group_used_bits[0]),
                                         (void**)&state.load_group_used_bits);
  }
  if (iree_status_is_ok(status) && state.global_request_count != 0) {
    status = iree_allocator_malloc_array(
        host_allocator, state.global_request_count,
        sizeof(state.request_used_bits[0]), (void**)&state.request_used_bits);
  }
  if (iree_status_is_ok(status) && state.parameter_tensor_count != 0) {
    status = iree_allocator_malloc_array(
        host_allocator, state.parameter_tensor_count,
        sizeof(state.parameter_tensor_used_bits[0]),
        (void**)&state.parameter_tensor_used_bits);
  }
  if (iree_status_is_ok(status) && state.load_group_used_bits) {
    memset(state.load_group_used_bits, 0,
           state.load_group_count * sizeof(state.load_group_used_bits[0]));
  }
  if (iree_status_is_ok(status) && state.request_used_bits) {
    memset(state.request_used_bits, 0,
           state.global_request_count * sizeof(state.request_used_bits[0]));
  }
  if (iree_status_is_ok(status) && state.parameter_tensor_used_bits) {
    memset(state.parameter_tensor_used_bits, 0,
           state.parameter_tensor_count *
               sizeof(state.parameter_tensor_used_bits[0]));
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_window_map_parameter_tensors(
        &state, host_allocator);
  }
  if (iree_status_is_ok(status)) {
    status =
        id4_pipeline_parameter_window_mark_selected_tensors(options, &state);
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_window_mark_used_groups(&state);
  }

  const iree_host_size_t compact_parameter_tensor_count =
      state.parameter_tensor_used_bits
          ? id4_pipeline_parameter_window_count_used_bits(
                state.parameter_tensor_count, state.parameter_tensor_used_bits)
          : 0;
  const iree_host_size_t compact_slab_count =
      iree_status_is_ok(status)
          ? id4_pipeline_parameter_window_count_used_slabs(&state)
          : 0;
  const iree_host_size_t compact_request_count =
      state.request_used_bits
          ? id4_pipeline_parameter_window_count_used_bits(
                state.global_request_count, state.request_used_bits)
          : 0;
  const iree_host_size_t compact_load_group_count =
      state.load_group_used_bits
          ? id4_pipeline_parameter_window_count_used_bits(
                state.load_group_count, state.load_group_used_bits)
          : 0;

  id4_pipeline_parameter_window_t* window = NULL;
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_window_create_empty(
        compact_parameter_tensor_count, compact_slab_count,
        compact_request_count, compact_load_group_count,
        state.global_request_count, host_allocator, &window);
  }
  if (iree_status_is_ok(status)) {
    status =
        id4_pipeline_parameter_window_pack_parameter_tensors(&state, window);
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_window_pack_requests(&state, window);
  }
  if (iree_status_is_ok(status)) {
    id4_pipeline_parameter_window_pack_load_groups(&state, window);
    *out_window = window;
    window = NULL;
  }

  id4_pipeline_parameter_window_release(window);
  iree_allocator_free(host_allocator,
                      state.parameter_tensor_indices_by_global_request);
  iree_allocator_free(host_allocator, state.parameter_tensor_used_bits);
  iree_allocator_free(host_allocator, state.request_used_bits);
  iree_allocator_free(host_allocator, state.load_group_used_bits);
  iree_allocator_free(host_allocator, state.global_request_offsets_by_slab);
  return status;
}

iree_status_t id4_pipeline_parameter_window_create_for_region(
    const id4_pipeline_plan_t* plan, iree_host_size_t region_index,
    iree_allocator_t host_allocator,
    id4_pipeline_parameter_window_t** out_window) {
  IREE_ASSERT_ARGUMENT(out_window);
  *out_window = NULL;
  if (!plan) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter window plan is required");
  }
  const id4_pipeline_region_plan_t* region =
      id4_pipeline_plan_region_at(plan, region_index);
  if (!region) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "parameter window region %" PRIhsz " is missing",
                            region_index);
  }

  id4_pipeline_parameter_window_build_state_t state;
  memset(&state, 0, sizeof(state));
  state.plan = plan;
  state.parameter_tensor_count = id4_pipeline_plan_parameter_tensor_count(plan);
  iree_status_t status = id4_pipeline_parameter_window_build_global_offsets(
      plan, host_allocator, &state.global_request_offsets_by_slab,
      &state.global_request_offset_count, &state.global_request_count);
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_window_map_parameter_tensors(
        &state, host_allocator);
  }

  uint8_t* parameter_tensor_used_bits = NULL;
  uint32_t* parameter_tensor_ordinals = NULL;
  if (iree_status_is_ok(status) && state.parameter_tensor_count != 0) {
    status = iree_allocator_malloc_array(host_allocator,
                                         state.parameter_tensor_count,
                                         sizeof(parameter_tensor_used_bits[0]),
                                         (void**)&parameter_tensor_used_bits);
  }
  if (iree_status_is_ok(status) && parameter_tensor_used_bits) {
    memset(
        parameter_tensor_used_bits, 0,
        state.parameter_tensor_count * sizeof(parameter_tensor_used_bits[0]));
  }
  for (iree_host_size_t group_ordinal = 0;
       group_ordinal < region->parameter_load_group_count &&
       iree_status_is_ok(status);
       ++group_ordinal) {
    id4_pipeline_parameter_load_group_t group;
    status = id4_pipeline_plan_parameter_load_group_at(
        plan, region->parameter_load_groups[group_ordinal], &group);
    for (iree_host_size_t step_ordinal = 0;
         step_ordinal < group.step_count && iree_status_is_ok(status);
         ++step_ordinal) {
      const id4_pipeline_parameter_load_step_t* step =
          id4_pipeline_plan_parameter_load_step_at(
              plan, group.step_offset + step_ordinal);
      if (!step) {
        status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "parameter window load step %" PRIhsz
                                  " is missing",
                                  group.step_offset + step_ordinal);
        break;
      }
      for (iree_host_size_t request_ordinal = 0;
           request_ordinal < step->request_count && iree_status_is_ok(status);
           ++request_ordinal) {
        iree_host_size_t slab_request_index = IREE_HOST_SIZE_MAX;
        status = id4_pipeline_parameter_window_step_request_index(
            step, request_ordinal, &slab_request_index);
        iree_host_size_t global_request_index = IREE_HOST_SIZE_MAX;
        if (iree_status_is_ok(status)) {
          status = id4_pipeline_parameter_window_global_request_index(
              &state, step->target_slab_index, slab_request_index,
              &global_request_index);
        }
        if (!iree_status_is_ok(status)) break;
        const iree_host_size_t parameter_tensor_index =
            state.parameter_tensor_indices_by_global_request
                ? state.parameter_tensor_indices_by_global_request
                      [global_request_index]
                : IREE_HOST_SIZE_MAX;
        if (parameter_tensor_index == IREE_HOST_SIZE_MAX) {
          status = iree_make_status(
              IREE_STATUS_FAILED_PRECONDITION,
              "parameter window region %" PRIhsz " request %" PRIhsz
              " does not belong to a planned parameter tensor",
              region_index, global_request_index);
          break;
        }
        parameter_tensor_used_bits[parameter_tensor_index] = 1;
      }
    }
  }

  const iree_host_size_t selected_parameter_tensor_count =
      parameter_tensor_used_bits
          ? id4_pipeline_parameter_window_count_used_bits(
                state.parameter_tensor_count, parameter_tensor_used_bits)
          : 0;
  if (iree_status_is_ok(status) && selected_parameter_tensor_count != 0) {
    status = iree_allocator_malloc_array(host_allocator,
                                         selected_parameter_tensor_count,
                                         sizeof(parameter_tensor_ordinals[0]),
                                         (void**)&parameter_tensor_ordinals);
  }
  if (iree_status_is_ok(status)) {
    iree_host_size_t selected_index = 0;
    for (iree_host_size_t i = 0; i < state.parameter_tensor_count; ++i) {
      if (!parameter_tensor_used_bits[i]) continue;
      const id4_pipeline_parameter_tensor_plan_t* tensor =
          id4_pipeline_plan_parameter_tensor_at(plan, i);
      parameter_tensor_ordinals[selected_index++] =
          tensor->program_tensor_ordinal;
    }
    id4_pipeline_parameter_window_create_options_t options;
    memset(&options, 0, sizeof(options));
    options.structure_size = sizeof(options);
    options.plan = plan;
    options.parameter_tensor_count = selected_parameter_tensor_count;
    options.parameter_tensor_ordinals = parameter_tensor_ordinals;
    status = id4_pipeline_parameter_window_create(&options, host_allocator,
                                                  out_window);
  }

  iree_allocator_free(host_allocator, parameter_tensor_ordinals);
  iree_allocator_free(host_allocator, parameter_tensor_used_bits);
  iree_allocator_free(host_allocator,
                      state.parameter_tensor_indices_by_global_request);
  iree_allocator_free(host_allocator, state.global_request_offsets_by_slab);
  return status;
}

void id4_pipeline_parameter_window_release(
    id4_pipeline_parameter_window_t* window) {
  if (!window) return;
  iree_allocator_t host_allocator = window->host_allocator;
  iree_allocator_free(host_allocator,
                      window->compact_request_indices_by_global_request);
  iree_allocator_free(host_allocator, window->load_groups);
  iree_allocator_free(host_allocator, window->requests);
  iree_allocator_free(host_allocator, window->slabs);
  iree_allocator_free(host_allocator, window->parameter_tensor_ordinals);
  iree_allocator_free(host_allocator, window);
}

iree_host_size_t id4_pipeline_parameter_window_parameter_tensor_count(
    const id4_pipeline_parameter_window_t* window) {
  return window ? window->parameter_tensor_count : 0;
}

uint32_t id4_pipeline_parameter_window_parameter_tensor_ordinal_at(
    const id4_pipeline_parameter_window_t* window, iree_host_size_t index) {
  return window && index < window->parameter_tensor_count
             ? window->parameter_tensor_ordinals[index]
             : UINT32_MAX;
}

iree_host_size_t id4_pipeline_parameter_window_slab_count(
    const id4_pipeline_parameter_window_t* window) {
  return window ? window->slab_count : 0;
}

const id4_pipeline_parameter_window_slab_t*
id4_pipeline_parameter_window_slab_at(
    const id4_pipeline_parameter_window_t* window, iree_host_size_t index) {
  if (!window || index >= window->slab_count) return NULL;
  return &window->slabs[index];
}

iree_host_size_t id4_pipeline_parameter_window_request_count(
    const id4_pipeline_parameter_window_t* window) {
  return window ? window->request_count : 0;
}

const id4_pipeline_parameter_window_request_t*
id4_pipeline_parameter_window_request_at(
    const id4_pipeline_parameter_window_t* window, iree_host_size_t index) {
  if (!window || index >= window->request_count) return NULL;
  return &window->requests[index];
}

iree_host_size_t id4_pipeline_parameter_window_load_group_count(
    const id4_pipeline_parameter_window_t* window) {
  return window ? window->load_group_count : 0;
}

iree_host_size_t id4_pipeline_parameter_window_load_group_at(
    const id4_pipeline_parameter_window_t* window, iree_host_size_t index) {
  if (!window || index >= window->load_group_count) return IREE_HOST_SIZE_MAX;
  return window->load_groups[index];
}

const id4_pipeline_parameter_window_request_t*
id4_pipeline_parameter_window_resolve_request(
    const id4_pipeline_parameter_window_t* window,
    iree_host_size_t global_request_index) {
  if (!window || global_request_index >= window->global_request_count) {
    return NULL;
  }
  const iree_host_size_t compact_request_index =
      window->compact_request_indices_by_global_request[global_request_index];
  if (compact_request_index >= window->request_count) return NULL;
  return &window->requests[compact_request_index];
}

static iree_status_t id4_pipeline_parameter_window_schedule_validate_options(
    const id4_pipeline_parameter_window_schedule_create_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter window schedule options are required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_window_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("parameter window schedule")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "parameter window schedule extension structures are not supported");
  }
  if (!options->plan) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter window schedule plan is required");
  }
  if (!options->window) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter window schedule window is required");
  }
  return iree_ok_status();
}

static iree_status_t
id4_pipeline_parameter_window_execution_layout_schedule_validate_options(
    const id4_pipeline_parameter_window_execution_layout_schedule_create_options_t*
        options) {
  if (!options) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "execution-layout parameter window schedule options are required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_window_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("execution-layout parameter window schedule")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "execution-layout parameter window schedule extension structures are "
        "not supported");
  }
  if (!options->plan || !options->window) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "execution-layout parameter window schedule requires a plan and "
        "window");
  }
  if (iree_string_view_is_empty(options->source_scope)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "execution-layout parameter window schedule source scope is required");
  }
  return iree_ok_status();
}

static iree_status_t
id4_pipeline_parameter_window_schedule_step_selected_request_count(
    const id4_pipeline_plan_t* plan,
    const id4_pipeline_parameter_window_t* window,
    const id4_pipeline_parameter_load_step_t* step,
    iree_host_size_t* out_selected_request_count) {
  *out_selected_request_count = 0;
  for (iree_host_size_t request_ordinal = 0;
       request_ordinal < step->request_count; ++request_ordinal) {
    iree_host_size_t original_request_index = IREE_HOST_SIZE_MAX;
    IREE_RETURN_IF_ERROR(id4_pipeline_parameter_window_step_request_index(
        step, request_ordinal, &original_request_index));
    iree_host_size_t global_request_index = IREE_HOST_SIZE_MAX;
    IREE_RETURN_IF_ERROR(
        id4_pipeline_parameter_window_plan_global_request_index(
            plan, step->target_slab_index, original_request_index,
            &global_request_index));
    if (id4_pipeline_parameter_window_resolve_request(window,
                                                      global_request_index)) {
      ++*out_selected_request_count;
    }
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_parameter_window_schedule_count_load_steps(
    const id4_pipeline_plan_t* plan,
    const id4_pipeline_parameter_window_t* window,
    iree_host_size_t* out_load_step_count) {
  *out_load_step_count = 0;
  for (iree_host_size_t i = 0; i < window->load_group_count; ++i) {
    id4_pipeline_parameter_load_group_t group;
    IREE_RETURN_IF_ERROR(id4_pipeline_plan_parameter_load_group_at(
        plan, window->load_groups[i], &group));
    for (iree_host_size_t step_ordinal = 0; step_ordinal < group.step_count;
         ++step_ordinal) {
      const id4_pipeline_parameter_load_step_t* step =
          id4_pipeline_plan_parameter_load_step_at(
              plan, group.step_offset + step_ordinal);
      if (!step) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "parameter window schedule step %" PRIhsz
                                " is missing",
                                group.step_offset + step_ordinal);
      }
      iree_host_size_t selected_request_count = 0;
      IREE_RETURN_IF_ERROR(
          id4_pipeline_parameter_window_schedule_step_selected_request_count(
              plan, window, step, &selected_request_count));
      if (selected_request_count == 0) continue;
      if (!iree_host_size_checked_add(*out_load_step_count, 1,
                                      out_load_step_count)) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "parameter window schedule load step count overflow");
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t
id4_pipeline_parameter_window_schedule_count_request_indices(
    const id4_pipeline_plan_t* plan,
    const id4_pipeline_parameter_window_t* window,
    iree_host_size_t* out_request_index_count) {
  *out_request_index_count = 0;
  for (iree_host_size_t group_ordinal = 0;
       group_ordinal < window->load_group_count; ++group_ordinal) {
    id4_pipeline_parameter_load_group_t group;
    IREE_RETURN_IF_ERROR(id4_pipeline_plan_parameter_load_group_at(
        plan, window->load_groups[group_ordinal], &group));
    for (iree_host_size_t step_ordinal = 0; step_ordinal < group.step_count;
         ++step_ordinal) {
      const id4_pipeline_parameter_load_step_t* step =
          id4_pipeline_plan_parameter_load_step_at(
              plan, group.step_offset + step_ordinal);
      if (!step) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "parameter window schedule step %" PRIhsz
                                " is missing",
                                group.step_offset + step_ordinal);
      }
      if (step->kind != ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_GATHER) continue;
      iree_host_size_t selected_request_count = 0;
      IREE_RETURN_IF_ERROR(
          id4_pipeline_parameter_window_schedule_step_selected_request_count(
              plan, window, step, &selected_request_count));
      if (!iree_host_size_checked_add(*out_request_index_count,
                                      selected_request_count,
                                      out_request_index_count)) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "parameter window schedule request index count overflow");
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_parameter_window_schedule_create_empty(
    id4_pipeline_plan_t* plan, iree_host_size_t load_count,
    iree_host_size_t request_count, iree_host_size_t load_step_count,
    iree_host_size_t request_index_count, iree_host_size_t load_group_count,
    iree_host_size_t original_load_group_count, iree_allocator_t host_allocator,
    id4_pipeline_parameter_window_schedule_t** out_schedule) {
  *out_schedule = NULL;
  id4_pipeline_parameter_window_schedule_t* schedule = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*schedule), (void**)&schedule);
  if (iree_status_is_ok(status)) {
    memset(schedule, 0, sizeof(*schedule));
    schedule->host_allocator = host_allocator;
    schedule->plan = plan;
    id4_pipeline_plan_retain(schedule->plan);
    schedule->load_count = load_count;
    schedule->request_count = request_count;
    schedule->load_step_count = load_step_count;
    schedule->request_index_count = request_index_count;
    schedule->load_group_count = load_group_count;
    schedule->original_load_group_count = original_load_group_count;
  }
  if (iree_status_is_ok(status) && load_count != 0) {
    status = iree_allocator_malloc_array(host_allocator, load_count,
                                         sizeof(schedule->slabs[0]),
                                         (void**)&schedule->slabs);
  }
  if (iree_status_is_ok(status) && load_count != 0) {
    status = iree_allocator_malloc_array(host_allocator, load_count,
                                         sizeof(schedule->loads[0]),
                                         (void**)&schedule->loads);
  }
  if (iree_status_is_ok(status) && load_count != 0) {
    status = iree_allocator_malloc_array(host_allocator, load_count,
                                         sizeof(schedule->request_tables[0]),
                                         (void**)&schedule->request_tables);
  }
  if (iree_status_is_ok(status) && request_count != 0) {
    status = iree_allocator_malloc_array(host_allocator, request_count,
                                         sizeof(schedule->requests[0]),
                                         (void**)&schedule->requests);
  }
  if (iree_status_is_ok(status) && load_step_count != 0) {
    status = iree_allocator_malloc_array(host_allocator, load_step_count,
                                         sizeof(schedule->load_steps[0]),
                                         (void**)&schedule->load_steps);
  }
  if (iree_status_is_ok(status) && request_index_count != 0) {
    status = iree_allocator_malloc_array(host_allocator, request_index_count,
                                         sizeof(schedule->request_indices[0]),
                                         (void**)&schedule->request_indices);
  }
  if (iree_status_is_ok(status) && load_group_count != 0) {
    status =
        iree_allocator_malloc_array(host_allocator, load_group_count,
                                    sizeof(schedule->original_load_groups[0]),
                                    (void**)&schedule->original_load_groups);
  }
  if (iree_status_is_ok(status) && original_load_group_count != 0) {
    status = iree_allocator_malloc_array(
        host_allocator, original_load_group_count,
        sizeof(schedule->compact_load_groups_by_original[0]),
        (void**)&schedule->compact_load_groups_by_original);
  }
  if (iree_status_is_ok(status) && schedule->slabs) {
    memset(schedule->slabs, 0, load_count * sizeof(schedule->slabs[0]));
  }
  if (iree_status_is_ok(status) && schedule->loads) {
    memset(schedule->loads, 0, load_count * sizeof(schedule->loads[0]));
  }
  if (iree_status_is_ok(status) && schedule->request_tables) {
    memset(schedule->request_tables, 0,
           load_count * sizeof(schedule->request_tables[0]));
  }
  if (iree_status_is_ok(status) && schedule->requests) {
    memset(schedule->requests, 0,
           request_count * sizeof(schedule->requests[0]));
  }
  if (iree_status_is_ok(status) && schedule->load_steps) {
    memset(schedule->load_steps, 0,
           load_step_count * sizeof(schedule->load_steps[0]));
  }
  if (iree_status_is_ok(status) && schedule->request_indices) {
    memset(schedule->request_indices, 0,
           request_index_count * sizeof(schedule->request_indices[0]));
  }
  if (iree_status_is_ok(status) && schedule->original_load_groups) {
    memset(schedule->original_load_groups, 0,
           load_group_count * sizeof(schedule->original_load_groups[0]));
  }
  if (iree_status_is_ok(status) && schedule->compact_load_groups_by_original) {
    for (iree_host_size_t i = 0; i < original_load_group_count; ++i) {
      schedule->compact_load_groups_by_original[i] = IREE_HOST_SIZE_MAX;
    }
  }
  if (iree_status_is_ok(status)) {
    *out_schedule = schedule;
  } else {
    id4_pipeline_parameter_window_schedule_release(schedule);
  }
  return status;
}

static iree_status_t id4_pipeline_parameter_window_schedule_make_slab_index_map(
    const id4_pipeline_plan_t* plan,
    const id4_pipeline_parameter_window_t* window,
    iree_allocator_t host_allocator, iree_host_size_t** out_slab_indices,
    iree_host_size_t* out_slab_index_count) {
  *out_slab_indices = NULL;
  *out_slab_index_count = id4_pipeline_plan_parameter_slab_count(plan);
  if (*out_slab_index_count == 0) return iree_ok_status();
  iree_host_size_t* slab_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, *out_slab_index_count, sizeof(slab_indices[0]),
      (void**)&slab_indices));
  for (iree_host_size_t i = 0; i < *out_slab_index_count; ++i) {
    slab_indices[i] = IREE_HOST_SIZE_MAX;
  }
  for (iree_host_size_t i = 0; i < window->slab_count; ++i) {
    const id4_pipeline_parameter_window_slab_t* slab = &window->slabs[i];
    if (slab->original_slab_index >= *out_slab_index_count) {
      iree_allocator_free(host_allocator, slab_indices);
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "parameter window original slab %" PRIhsz
                              " exceeds plan slab count %" PRIhsz,
                              slab->original_slab_index, *out_slab_index_count);
    }
    slab_indices[slab->original_slab_index] = i;
  }
  *out_slab_indices = slab_indices;
  return iree_ok_status();
}

static const id4_pipeline_parameter_window_request_t*
id4_pipeline_parameter_window_find_original_request(
    const id4_pipeline_parameter_window_t* window,
    iree_host_size_t original_slab_index,
    iree_host_size_t original_request_index,
    iree_host_size_t* out_compact_request_index) {
  for (iree_host_size_t i = 0; i < window->request_count; ++i) {
    const id4_pipeline_parameter_window_request_t* request =
        &window->requests[i];
    if (request->original_slab_index != original_slab_index ||
        request->original_request_index != original_request_index) {
      continue;
    }
    *out_compact_request_index = i;
    return request;
  }
  *out_compact_request_index = IREE_HOST_SIZE_MAX;
  return NULL;
}

static iree_status_t id4_pipeline_parameter_window_schedule_rebase_request(
    const id4_pipeline_parameter_window_t* window,
    const iree_host_size_t* compact_slab_indices_by_original,
    iree_host_size_t compact_slab_index_count,
    iree_host_size_t original_slab_index,
    iree_host_size_t original_request_index,
    iree_host_size_t* out_compact_slab_index,
    iree_host_size_t* out_compact_request_index) {
  *out_compact_slab_index = IREE_HOST_SIZE_MAX;
  *out_compact_request_index = IREE_HOST_SIZE_MAX;
  if (original_slab_index >= compact_slab_index_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "parameter window request original slab %" PRIhsz
                            " exceeds compact slab map count %" PRIhsz,
                            original_slab_index, compact_slab_index_count);
  }
  const iree_host_size_t compact_slab_index =
      compact_slab_indices_by_original[original_slab_index];
  if (compact_slab_index == IREE_HOST_SIZE_MAX ||
      compact_slab_index >= window->slab_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "parameter window does not contain original slab %" PRIhsz,
        original_slab_index);
  }
  iree_host_size_t compact_global_request_index = IREE_HOST_SIZE_MAX;
  const id4_pipeline_parameter_window_request_t* request =
      id4_pipeline_parameter_window_find_original_request(
          window, original_slab_index, original_request_index,
          &compact_global_request_index);
  if (!request) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "parameter window does not contain original slab %" PRIhsz
        " request %" PRIhsz,
        original_slab_index, original_request_index);
  }
  const id4_pipeline_parameter_window_slab_t* slab =
      &window->slabs[compact_slab_index];
  if (compact_global_request_index < slab->request_offset ||
      compact_global_request_index >=
          slab->request_offset + slab->request_count) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "parameter window request/slab ordinal mismatch");
  }
  *out_compact_slab_index = compact_slab_index;
  *out_compact_request_index =
      compact_global_request_index - slab->request_offset;
  return iree_ok_status();
}

static iree_status_t id4_pipeline_parameter_window_schedule_step_request_index(
    const id4_pipeline_parameter_load_step_t* step,
    iree_host_size_t request_ordinal, iree_host_size_t* out_request_index) {
  return id4_pipeline_parameter_window_step_request_index(step, request_ordinal,
                                                          out_request_index);
}

static iree_status_t id4_pipeline_parameter_window_schedule_populate_slabs(
    const id4_pipeline_plan_t* plan,
    const id4_pipeline_parameter_window_t* window, bool uses_execution_layout,
    id4_pipeline_parameter_window_schedule_t* schedule) {
  iree_hal_device_group_t* device_group = id4_pipeline_plan_device_group(plan);
  if (!device_group) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter window schedule device group is "
                            "required");
  }
  for (iree_host_size_t compact_slab_index = 0;
       compact_slab_index < window->slab_count; ++compact_slab_index) {
    const id4_pipeline_parameter_window_slab_t* window_slab =
        &window->slabs[compact_slab_index];
    const id4_pipeline_parameter_slab_plan_t* original_slab =
        id4_pipeline_plan_parameter_slab_at(plan,
                                            window_slab->original_slab_index);
    const id4_pipeline_parameter_request_table_t* original_request_table =
        id4_pipeline_plan_parameter_request_table_at(
            plan, window_slab->original_slab_index);
    if (!original_slab || !original_request_table) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "parameter window original slab %" PRIhsz
                              " is missing",
                              window_slab->original_slab_index);
    }
    for (iree_host_size_t request_ordinal = 0;
         request_ordinal < window_slab->request_count; ++request_ordinal) {
      const iree_host_size_t compact_request_index =
          window_slab->request_offset + request_ordinal;
      const id4_pipeline_parameter_window_request_t* window_request =
          &window->requests[compact_request_index];
      const id4_pipeline_parameter_request_t* original_request =
          &original_request_table
               ->values[window_request->original_request_index];
      if (uses_execution_layout) {
        IREE_RETURN_IF_ERROR(id4_pipeline_parameter_layout_make_archive_request(
            plan, window_request->parameter_tensor_index, original_request,
            window_request->span, &schedule->requests[compact_request_index]));
      } else {
        schedule->requests[compact_request_index] =
            id4_pipeline_parameter_request(original_request->key,
                                           window_request->span);
      }
    }
    schedule->slabs[compact_slab_index] = id4_pipeline_make_parameter_slab_plan(
        original_slab->placement_id, window_slab->binding_slot,
        window_slab->target_params, window_slab->byte_length,
        window_slab->alignment);
    schedule->slabs[compact_slab_index].domain = original_slab->domain;
    schedule->request_tables[compact_slab_index] =
        id4_pipeline_make_parameter_request_table(
            window_slab->request_count,
            window_slab->request_count == 0
                ? NULL
                : &schedule->requests[window_slab->request_offset]);

    const id4_pipeline_device_placement_t* placement =
        id4_pipeline_plan_placement_at(plan, original_slab->placement_id);
    if (!placement) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "parameter window schedule slab %" PRIhsz
                              " references missing placement %u",
                              compact_slab_index, original_slab->placement_id);
    }
    iree_hal_device_t* device =
        iree_hal_device_group_device_at(device_group, placement->device_index);
    if (!device) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "parameter window schedule slab %" PRIhsz
                              " device is required",
                              compact_slab_index);
    }
    schedule->loads[compact_slab_index] = (id4_pipeline_parameter_slab_load_t){
        // Schedule-local compact slab index.
        .slab_index = compact_slab_index,
        // Compact slab plan owned by the schedule.
        .slab = &schedule->slabs[compact_slab_index],
        // Compact provider requests owned by the schedule.
        .request_table = &schedule->request_tables[compact_slab_index],
        // Device index inherited from the original slab placement.
        .device_index = placement->device_index,
        // HAL device borrowed from the retained plan device group.
        .device = device,
        // Queue affinity inherited from the original slab placement.
        .queue_affinity = placement->queue_affinity,
    };
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_parameter_window_schedule_populate_step(
    const id4_pipeline_plan_t* plan,
    const id4_pipeline_parameter_window_t* window,
    const iree_host_size_t* compact_slab_indices_by_original,
    iree_host_size_t compact_slab_index_count,
    iree_host_size_t original_load_group_index,
    const id4_pipeline_parameter_load_step_t* original_step,
    iree_host_size_t* request_indices,
    iree_host_size_t* io_request_index_offset,
    id4_pipeline_parameter_load_step_t* out_step) {
  *out_step = *original_step;
  iree_host_size_t* compact_request_indices =
      original_step->kind == ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_GATHER
          ? &request_indices[*io_request_index_offset]
          : NULL;
  out_step->request_indices = compact_request_indices;

  iree_host_size_t compact_slab_index = IREE_HOST_SIZE_MAX;
  iree_host_size_t first_compact_request_index = IREE_HOST_SIZE_MAX;
  iree_host_size_t selected_request_count = 0;
  for (iree_host_size_t request_ordinal = 0;
       request_ordinal < original_step->request_count; ++request_ordinal) {
    iree_host_size_t original_request_index = IREE_HOST_SIZE_MAX;
    IREE_RETURN_IF_ERROR(
        id4_pipeline_parameter_window_schedule_step_request_index(
            original_step, request_ordinal, &original_request_index));
    iree_host_size_t global_request_index = IREE_HOST_SIZE_MAX;
    IREE_RETURN_IF_ERROR(
        id4_pipeline_parameter_window_plan_global_request_index(
            plan, original_step->target_slab_index, original_request_index,
            &global_request_index));
    if (!id4_pipeline_parameter_window_resolve_request(window,
                                                       global_request_index)) {
      continue;
    }
    iree_host_size_t step_compact_slab_index = IREE_HOST_SIZE_MAX;
    iree_host_size_t step_compact_request_index = IREE_HOST_SIZE_MAX;
    IREE_RETURN_IF_ERROR(id4_pipeline_parameter_window_schedule_rebase_request(
        window, compact_slab_indices_by_original, compact_slab_index_count,
        original_step->target_slab_index, original_request_index,
        &step_compact_slab_index, &step_compact_request_index));
    if (selected_request_count == 0) {
      compact_slab_index = step_compact_slab_index;
      first_compact_request_index = step_compact_request_index;
    } else if (compact_slab_index != step_compact_slab_index) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "parameter load step '%.*s' spans compact slabs",
                              (int)original_step->name.size,
                              original_step->name.data);
    }
    if (compact_request_indices) {
      compact_request_indices[selected_request_count] =
          step_compact_request_index;
    }
    ++selected_request_count;
  }
  if (selected_request_count == 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "parameter load step '%.*s' has no selected requests",
        (int)original_step->name.size, original_step->name.data);
  }

  out_step->target_slab_index = compact_slab_index;
  out_step->request_offset = first_compact_request_index;
  out_step->request_count = selected_request_count;
  if (compact_request_indices) {
    out_step->request_offset = 0;
    *io_request_index_offset += selected_request_count;
  } else {
    out_step->request_indices = NULL;
  }
  if (original_step->kind != ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_GATHER) {
    out_step->readiness_group_key = original_load_group_index;
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_parameter_window_schedule_populate_steps(
    const id4_pipeline_plan_t* plan,
    const id4_pipeline_parameter_window_t* window,
    const iree_host_size_t* compact_slab_indices_by_original,
    iree_host_size_t compact_slab_index_count,
    id4_pipeline_parameter_window_schedule_t* schedule) {
  iree_host_size_t load_step_index = 0;
  iree_host_size_t request_index_offset = 0;
  for (iree_host_size_t compact_group_index = 0;
       compact_group_index < window->load_group_count; ++compact_group_index) {
    const iree_host_size_t original_group_index =
        window->load_groups[compact_group_index];
    schedule->original_load_groups[compact_group_index] = original_group_index;
    if (original_group_index >= schedule->original_load_group_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "parameter window schedule original load group %" PRIhsz
          " exceeds group count %" PRIhsz,
          original_group_index, schedule->original_load_group_count);
    }
    schedule->compact_load_groups_by_original[original_group_index] =
        compact_group_index;

    id4_pipeline_parameter_load_group_t group;
    IREE_RETURN_IF_ERROR(id4_pipeline_plan_parameter_load_group_at(
        plan, original_group_index, &group));
    for (iree_host_size_t step_ordinal = 0; step_ordinal < group.step_count;
         ++step_ordinal) {
      const iree_host_size_t original_step_index =
          group.step_offset + step_ordinal;
      const id4_pipeline_parameter_load_step_t* original_step =
          id4_pipeline_plan_parameter_load_step_at(plan, original_step_index);
      if (!original_step) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "parameter window schedule step %" PRIhsz
                                " is missing",
                                original_step_index);
      }
      iree_host_size_t selected_request_count = 0;
      IREE_RETURN_IF_ERROR(
          id4_pipeline_parameter_window_schedule_step_selected_request_count(
              plan, window, original_step, &selected_request_count));
      if (selected_request_count == 0) continue;
      id4_pipeline_parameter_load_step_t* compact_step =
          &schedule->load_steps[load_step_index++];
      IREE_RETURN_IF_ERROR(id4_pipeline_parameter_window_schedule_populate_step(
          plan, window, compact_slab_indices_by_original,
          compact_slab_index_count, original_group_index, original_step,
          schedule->request_indices, &request_index_offset, compact_step));
    }
  }
  if (load_step_index != schedule->load_step_count ||
      request_index_offset != schedule->request_index_count) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "parameter window schedule packed count mismatch");
  }
  return iree_ok_status();
}

iree_status_t id4_pipeline_parameter_window_schedule_create(
    const id4_pipeline_parameter_window_schedule_create_options_t* options,
    iree_allocator_t host_allocator,
    id4_pipeline_parameter_window_schedule_t** out_schedule) {
  IREE_ASSERT_ARGUMENT(out_schedule);
  *out_schedule = NULL;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_parameter_window_schedule_validate_options(options));

  iree_host_size_t load_step_count = 0;
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_window_schedule_count_load_steps(
      options->plan, options->window, &load_step_count));
  iree_host_size_t request_index_count = 0;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_parameter_window_schedule_count_request_indices(
          options->plan, options->window, &request_index_count));
  iree_host_size_t original_load_group_count = 0;
  IREE_RETURN_IF_ERROR(id4_pipeline_plan_parameter_load_group_count(
      options->plan, &original_load_group_count));

  id4_pipeline_parameter_window_schedule_t* schedule = NULL;
  iree_status_t status = id4_pipeline_parameter_window_schedule_create_empty(
      options->plan, options->window->slab_count,
      options->window->request_count, load_step_count, request_index_count,
      options->window->load_group_count, original_load_group_count,
      host_allocator, &schedule);

  iree_host_size_t* compact_slab_indices_by_original = NULL;
  iree_host_size_t compact_slab_index_count = 0;
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_window_schedule_make_slab_index_map(
        options->plan, options->window, host_allocator,
        &compact_slab_indices_by_original, &compact_slab_index_count);
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_window_schedule_populate_slabs(
        options->plan, options->window, /*uses_execution_layout=*/false,
        schedule);
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_window_schedule_populate_steps(
        options->plan, options->window, compact_slab_indices_by_original,
        compact_slab_index_count, schedule);
  }
  iree_allocator_free(host_allocator, compact_slab_indices_by_original);

  if (iree_status_is_ok(status)) {
    *out_schedule = schedule;
  } else {
    id4_pipeline_parameter_window_schedule_release(schedule);
  }
  return status;
}

iree_status_t id4_pipeline_parameter_window_execution_layout_schedule_create(
    const id4_pipeline_parameter_window_execution_layout_schedule_create_options_t*
        options,
    iree_allocator_t host_allocator,
    id4_pipeline_parameter_window_schedule_t** out_schedule) {
  IREE_ASSERT_ARGUMENT(out_schedule);
  *out_schedule = NULL;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_parameter_window_execution_layout_schedule_validate_options(
          options));

  const iree_host_size_t load_count = options->window->slab_count;
  id4_pipeline_parameter_window_schedule_t* schedule = NULL;
  iree_status_t status = id4_pipeline_parameter_window_schedule_create_empty(
      options->plan, load_count, options->window->request_count,
      /*load_step_count=*/load_count, /*request_index_count=*/0,
      /*load_group_count=*/load_count, /*original_load_group_count=*/0,
      host_allocator, &schedule);
  if (iree_status_is_ok(status)) {
    char* source_scope_storage = NULL;
    status = iree_allocator_malloc(host_allocator, options->source_scope.size,
                                   (void**)&source_scope_storage);
    if (iree_status_is_ok(status)) {
      memcpy(source_scope_storage, options->source_scope.data,
             options->source_scope.size);
      schedule->source_scope = iree_make_string_view(
          source_scope_storage, options->source_scope.size);
    }
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_window_schedule_populate_slabs(
        options->plan, options->window, /*uses_execution_layout=*/true,
        schedule);
  }
  for (iree_host_size_t i = 0; i < load_count && iree_status_is_ok(status);
       ++i) {
    const id4_pipeline_parameter_request_table_t* request_table =
        schedule->loads[i].request_table;
    if (!request_table || request_table->count == 0) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "execution-layout parameter window slab %" PRIhsz " has no requests",
          i);
      break;
    }
    schedule->load_steps[i] = id4_pipeline_parameter_gather_load_step(
        IREE_SV("baked execution layout"), schedule->source_scope,
        /*target_slab_index=*/i, /*request_offset=*/0, request_table->count);
    schedule->original_load_groups[i] = IREE_HOST_SIZE_MAX;
  }
  if (iree_status_is_ok(status)) {
    *out_schedule = schedule;
  } else {
    id4_pipeline_parameter_window_schedule_release(schedule);
  }
  return status;
}

void id4_pipeline_parameter_window_schedule_release(
    id4_pipeline_parameter_window_schedule_t* schedule) {
  if (!schedule) return;
  iree_allocator_t host_allocator = schedule->host_allocator;
  id4_pipeline_plan_release(schedule->plan);
  iree_allocator_free(host_allocator, (void*)schedule->source_scope.data);
  iree_allocator_free(host_allocator,
                      schedule->compact_load_groups_by_original);
  iree_allocator_free(host_allocator, schedule->original_load_groups);
  iree_allocator_free(host_allocator, schedule->request_indices);
  iree_allocator_free(host_allocator, schedule->load_steps);
  iree_allocator_free(host_allocator, schedule->requests);
  iree_allocator_free(host_allocator, schedule->request_tables);
  iree_allocator_free(host_allocator, schedule->loads);
  iree_allocator_free(host_allocator, schedule->slabs);
  iree_allocator_free(host_allocator, schedule);
}

iree_host_size_t id4_pipeline_parameter_window_schedule_load_count(
    const id4_pipeline_parameter_window_schedule_t* schedule) {
  return schedule ? schedule->load_count : 0;
}

const id4_pipeline_parameter_slab_load_t*
id4_pipeline_parameter_window_schedule_loads(
    const id4_pipeline_parameter_window_schedule_t* schedule) {
  return schedule ? schedule->loads : NULL;
}

iree_host_size_t id4_pipeline_parameter_window_schedule_load_step_count(
    const id4_pipeline_parameter_window_schedule_t* schedule) {
  return schedule ? schedule->load_step_count : 0;
}

const id4_pipeline_parameter_load_step_t*
id4_pipeline_parameter_window_schedule_load_steps(
    const id4_pipeline_parameter_window_schedule_t* schedule) {
  return schedule ? schedule->load_steps : NULL;
}

iree_host_size_t id4_pipeline_parameter_window_schedule_load_group_count(
    const id4_pipeline_parameter_window_schedule_t* schedule) {
  return schedule ? schedule->load_group_count : 0;
}

iree_host_size_t id4_pipeline_parameter_window_schedule_original_load_group_at(
    const id4_pipeline_parameter_window_schedule_t* schedule,
    iree_host_size_t index) {
  if (!schedule || index >= schedule->load_group_count) {
    return IREE_HOST_SIZE_MAX;
  }
  return schedule->original_load_groups[index];
}

iree_host_size_t id4_pipeline_parameter_window_schedule_compact_load_group(
    const id4_pipeline_parameter_window_schedule_t* schedule,
    iree_host_size_t original_load_group_index) {
  if (!schedule ||
      original_load_group_index >= schedule->original_load_group_count) {
    return IREE_HOST_SIZE_MAX;
  }
  return schedule->compact_load_groups_by_original[original_load_group_index];
}
