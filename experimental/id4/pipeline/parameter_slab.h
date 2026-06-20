// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_PIPELINE_PARAMETER_SLAB_H_
#define EXPERIMENTAL_ID4_PIPELINE_PARAMETER_SLAB_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/io/parameter_provider.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Plan-local device placement identifier used by parameter slabs.
typedef uint32_t id4_pipeline_device_placement_id_t;

// Invalid device placement identifier.
#define ID4_PIPELINE_DEVICE_PLACEMENT_ID_INVALID UINT32_MAX

// Source parameter span and planned slab target span.
typedef struct id4_pipeline_parameter_request_t {
  // Parameter key in the provider scope.
  iree_string_view_t key;
  // Source and target byte range for the gather operation.
  iree_io_parameter_span_t span;
} id4_pipeline_parameter_request_t;

// Planned parameter slab populated during prepare.
typedef struct id4_pipeline_parameter_slab_plan_t {
  // Provider scope containing all request keys in this slab.
  iree_string_view_t scope;
  // Placement where the slab is allocated.
  id4_pipeline_device_placement_id_t placement_id;
  // HAL buffer parameters used for slab allocation.
  iree_hal_buffer_params_t target_params;
  // Total slab byte length.
  iree_device_size_t byte_length;
  // Required slab base alignment in bytes.
  iree_device_size_t alignment;
  // Number of parameter requests.
  iree_host_size_t request_count;
  // Parameter requests in gather enumeration order.
  const id4_pipeline_parameter_request_t* requests;
} id4_pipeline_parameter_slab_plan_t;

// State passed to the IREE parameter provider enumerator callback.
typedef struct id4_pipeline_parameter_slab_enumerator_state_t {
  // Slab plan being enumerated.
  const id4_pipeline_parameter_slab_plan_t* slab;
} id4_pipeline_parameter_slab_enumerator_state_t;

// Validates that a parameter slab references valid placements and byte ranges.
iree_status_t id4_pipeline_parameter_slab_validate(
    const id4_pipeline_parameter_slab_plan_t* slab,
    iree_host_size_t placement_count);

// Enumerates one planned parameter request in IREE provider callback form.
iree_status_t id4_pipeline_parameter_slab_enumerate(
    void* user_data, iree_host_size_t i, iree_string_view_t* out_key,
    iree_io_parameter_span_t* out_span);

// Returns an IREE parameter enumerator for |state|.
iree_io_parameter_enumerator_t id4_pipeline_parameter_slab_enumerator(
    id4_pipeline_parameter_slab_enumerator_state_t* state);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_PIPELINE_PARAMETER_SLAB_H_
