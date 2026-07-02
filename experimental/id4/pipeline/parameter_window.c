// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/parameter_window.h"

#include <string.h>

struct id4_pipeline_parameter_window_t {
  // Host allocator used for window storage.
  iree_allocator_t host_allocator;
  // First contiguous region represented by the window.
  iree_host_size_t region_offset;
  // Number of contiguous regions represented by the window.
  iree_host_size_t region_count;
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
  const iree_host_size_t plan_region_count =
      id4_pipeline_plan_region_count(options->plan);
  if (options->region_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter window region count must be non-zero");
  }
  if (options->region_offset >= plan_region_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "parameter window region offset %" PRIhsz
                            " exceeds plan region count %" PRIhsz,
                            options->region_offset, plan_region_count);
  }
  if (options->region_count > plan_region_count - options->region_offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "parameter window region range [%" PRIhsz
                            ", %" PRIhsz ") exceeds plan region count %" PRIhsz,
                            options->region_offset,
                            options->region_offset + options->region_count,
                            plan_region_count);
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
    const id4_pipeline_parameter_slab_plan_t* slab =
        id4_pipeline_plan_parameter_slab_at(plan, i);
    if (!slab) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "parameter slab %" PRIhsz " is missing", i);
      break;
    }
    if (!iree_host_size_checked_add(global_request_count, slab->request_count,
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

static iree_status_t id4_pipeline_parameter_window_mark_load_group_requests(
    id4_pipeline_parameter_window_build_state_t* state,
    iree_host_size_t group_index) {
  id4_pipeline_parameter_load_group_t group;
  IREE_RETURN_IF_ERROR(id4_pipeline_plan_parameter_load_group_at(
      state->plan, group_index, &group));
  for (iree_host_size_t step_ordinal = 0; step_ordinal < group.step_count;
       ++step_ordinal) {
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
      state->request_used_bits[global_request_index] = 1;
    }
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_parameter_window_mark_used_groups(
    const id4_pipeline_parameter_window_create_options_t* options,
    id4_pipeline_parameter_window_build_state_t* state) {
  const iree_host_size_t region_limit =
      options->region_offset + options->region_count;
  for (iree_host_size_t region_index = options->region_offset;
       region_index < region_limit; ++region_index) {
    const id4_pipeline_region_plan_t* region =
        id4_pipeline_plan_region_at(options->plan, region_index);
    if (!region) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "parameter window region %" PRIhsz " is missing",
                              region_index);
    }
    for (iree_host_size_t group_ordinal = 0;
         group_ordinal < region->parameter_load_group_count; ++group_ordinal) {
      const iree_host_size_t group_index =
          region->parameter_load_groups[group_ordinal];
      if (group_index >= state->load_group_count) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "parameter window region %" PRIhsz " load group %" PRIhsz
            " exceeds load group count %" PRIhsz,
            region_index, group_index, state->load_group_count);
      }
      if (state->load_group_used_bits[group_index]) continue;
      state->load_group_used_bits[group_index] = 1;
      IREE_RETURN_IF_ERROR(
          id4_pipeline_parameter_window_mark_load_group_requests(state,
                                                                 group_index));
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
    const id4_pipeline_parameter_window_create_options_t* options,
    iree_host_size_t slab_count, iree_host_size_t request_count,
    iree_host_size_t load_group_count, iree_host_size_t global_request_count,
    iree_allocator_t host_allocator,
    id4_pipeline_parameter_window_t** out_window) {
  *out_window = NULL;
  id4_pipeline_parameter_window_t* window = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*window), (void**)&window));
  memset(window, 0, sizeof(*window));
  window->host_allocator = host_allocator;
  window->region_offset = options->region_offset;
  window->region_count = options->region_count;
  window->slab_count = slab_count;
  window->request_count = request_count;
  window->load_group_count = load_group_count;
  window->global_request_count = global_request_count;

  iree_status_t status = iree_ok_status();
  if (slab_count != 0) {
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
    if (!original_slab) {
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
    for (iree_host_size_t original_request_index = 0;
         original_request_index < original_slab->request_count;
         ++original_request_index) {
      const iree_host_size_t global_request_index =
          global_request_offset + original_request_index;
      if (!state->request_used_bits[global_request_index]) continue;
      const id4_pipeline_parameter_request_t* original_request =
          &original_slab->requests[original_request_index];
      iree_io_parameter_span_t compact_span;
      IREE_RETURN_IF_ERROR(id4_pipeline_parameter_slab_pack_span(
          original_request->span.length, compact_slab.alignment,
          &compact_slab.byte_length, &compact_span));
      compact_span.parameter_offset = original_request->span.parameter_offset;
      window->requests[compact_request_index] =
          (id4_pipeline_parameter_window_request_t){
              // Original slab containing this request.
              .original_slab_index = original_slab_index,
              // Original request index within the original slab.
              .original_request_index = original_request_index,
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
  if (iree_status_is_ok(status) && state.load_group_used_bits) {
    memset(state.load_group_used_bits, 0,
           state.load_group_count * sizeof(state.load_group_used_bits[0]));
  }
  if (iree_status_is_ok(status) && state.request_used_bits) {
    memset(state.request_used_bits, 0,
           state.global_request_count * sizeof(state.request_used_bits[0]));
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_window_mark_used_groups(options, &state);
  }

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
        options, compact_slab_count, compact_request_count,
        compact_load_group_count, state.global_request_count, host_allocator,
        &window);
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
  iree_allocator_free(host_allocator, state.request_used_bits);
  iree_allocator_free(host_allocator, state.load_group_used_bits);
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
  iree_allocator_free(host_allocator, window);
}

iree_host_size_t id4_pipeline_parameter_window_region_offset(
    const id4_pipeline_parameter_window_t* window) {
  return window ? window->region_offset : IREE_HOST_SIZE_MAX;
}

iree_host_size_t id4_pipeline_parameter_window_region_count(
    const id4_pipeline_parameter_window_t* window) {
  return window ? window->region_count : 0;
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
