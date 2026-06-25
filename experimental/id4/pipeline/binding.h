// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_PIPELINE_BINDING_H_
#define EXPERIMENTAL_ID4_PIPELINE_BINDING_H_

#include "experimental/id4/pipeline/plan.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Owned set of standalone HAL buffers and binding table entries.
typedef struct id4_pipeline_buffer_binding_set_t {
  // Allocator used for arrays in this set.
  iree_allocator_t host_allocator;
  // Number of bindings in plan order.
  iree_host_size_t count;
  // Owned buffers backing each binding.
  iree_hal_buffer_t** buffers;
  // Binding table entries in plan order.
  iree_hal_buffer_binding_t* bindings;
} id4_pipeline_buffer_binding_set_t;

// Releases all buffers and arrays owned by |binding_set|.
void id4_pipeline_buffer_binding_set_deinitialize(
    id4_pipeline_buffer_binding_set_t* binding_set);

// Allocates one device-local buffer for each planned boundary tensor.
iree_status_t id4_pipeline_allocate_boundary_bindings(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const id4_pipeline_plan_t* plan, iree_allocator_t host_allocator,
    id4_pipeline_buffer_binding_set_t* out_binding_set);

// Allocates one device-local buffer for each planned diagnostic tap.
iree_status_t id4_pipeline_allocate_diagnostic_tap_bindings(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const id4_pipeline_plan_t* plan, iree_allocator_t host_allocator,
    id4_pipeline_buffer_binding_set_t* out_binding_set);

// Finds a boundary binding by planned boundary tensor name.
iree_status_t id4_pipeline_find_boundary_binding(
    const id4_pipeline_plan_t* plan,
    const id4_pipeline_buffer_binding_set_t* binding_set,
    iree_string_view_t name, iree_hal_buffer_binding_t* out_binding);

// Finds a diagnostic tap binding by planned tap name.
iree_status_t id4_pipeline_find_diagnostic_tap_binding(
    const id4_pipeline_plan_t* plan,
    const id4_pipeline_buffer_binding_set_t* binding_set,
    iree_string_view_t name, iree_hal_buffer_binding_t* out_binding);

// Queues chunked host-to-device updates into |binding| on |semaphore|.
iree_status_t id4_pipeline_queue_update_binding(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_buffer_binding_t* binding, const void* source_data,
    iree_host_size_t source_length,
    iree_hal_semaphore_list_t initial_wait_semaphore_list,
    iree_hal_semaphore_t* semaphore, uint64_t* inout_payload_value);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_PIPELINE_BINDING_H_
