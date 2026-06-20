// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/parameter_slab.h"

iree_status_t id4_pipeline_parameter_slab_validate(
    const id4_pipeline_parameter_slab_plan_t* slab,
    iree_host_size_t placement_count) {
  IREE_ASSERT_ARGUMENT(slab);
  if (slab->placement_id >= placement_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "parameter slab placement %u outside placement count %" PRIhsz,
        slab->placement_id, placement_count);
  }
  if (slab->request_count != 0 && !slab->requests) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter slab request array is required");
  }
  for (iree_host_size_t i = 0; i < slab->request_count; ++i) {
    const id4_pipeline_parameter_request_t* request = &slab->requests[i];
    if (iree_string_view_is_empty(request->key)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "parameter request %" PRIhsz " has no key", i);
    }
    if (request->span.length == 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "parameter request %" PRIhsz " has zero byte length", i);
    }
    if (request->span.buffer_offset > slab->byte_length ||
        request->span.length >
            slab->byte_length - request->span.buffer_offset) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "parameter request %" PRIhsz " exceeds slab byte length", i);
    }
  }
  return iree_ok_status();
}

iree_status_t id4_pipeline_parameter_slab_enumerate(
    void* user_data, iree_host_size_t i, iree_string_view_t* out_key,
    iree_io_parameter_span_t* out_span) {
  IREE_ASSERT_ARGUMENT(user_data);
  IREE_ASSERT_ARGUMENT(out_key);
  IREE_ASSERT_ARGUMENT(out_span);
  id4_pipeline_parameter_slab_enumerator_state_t* state =
      (id4_pipeline_parameter_slab_enumerator_state_t*)user_data;
  if (!state->slab || i >= state->slab->request_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "parameter request index %" PRIhsz " is outside slab request count", i);
  }
  const id4_pipeline_parameter_request_t* request = &state->slab->requests[i];
  *out_key = request->key;
  *out_span = request->span;
  return iree_ok_status();
}

iree_io_parameter_enumerator_t id4_pipeline_parameter_slab_enumerator(
    id4_pipeline_parameter_slab_enumerator_state_t* state) {
  iree_io_parameter_enumerator_t enumerator = {
      // Callback used by IREE parameter provider gather/load APIs.
      .fn = id4_pipeline_parameter_slab_enumerate,
      // Enumerator state holding the planned slab request array.
      .user_data = state,
  };
  return enumerator;
}
