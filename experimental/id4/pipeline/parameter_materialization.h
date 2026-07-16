// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 WITH LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_PIPELINE_PARAMETER_MATERIALIZATION_H_
#define EXPERIMENTAL_ID4_PIPELINE_PARAMETER_MATERIALIZATION_H_

#include "experimental/id4/pipeline/diagnostics.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Opaque asynchronously constructed parameter domain.
typedef struct id4_pipeline_parameter_materialization_t
    id4_pipeline_parameter_materialization_t;

// Opaque pipeline plan defining the materialized domain.
typedef struct id4_pipeline_plan_t id4_pipeline_plan_t;

// Borrowed target state for populating one acquired parameter domain.
typedef struct id4_pipeline_parameter_materialization_target_t {
  // Plan-local slab index populated by the materialization.
  iree_host_size_t slab_index;
  // Undefined domain buffer owned by the materialization.
  iree_hal_buffer_t* target_buffer;
  // Edge that must be reached before the target buffer is accessed.
  iree_hal_semaphore_list_t readiness_semaphore_list;
} id4_pipeline_parameter_materialization_target_t;

// Borrowed binding state for one published parameter domain.
typedef struct id4_pipeline_parameter_materialization_binding_t {
  // Exact plan used to acquire and populate the domain.
  const id4_pipeline_plan_t* plan;
  // Plan-local slab index populated by the materialization.
  iree_host_size_t slab_index;
  // Published domain buffer owned by the materialization.
  iree_hal_buffer_t* buffer;
  // Edge that must be reached before the domain buffer is used.
  iree_hal_semaphore_list_t readiness_semaphore_list;
} id4_pipeline_parameter_materialization_binding_t;

// Options for acquiring storage for one parameter domain.
typedef struct id4_pipeline_parameter_materialization_acquire_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Plan whose parameter-domain layout the materialization preserves.
  const id4_pipeline_plan_t* plan;
  // Plan-local slab index populated by the materialized domain.
  iree_host_size_t target_slab_index;
  // Optional HAL allocation pool retained through asynchronous retirement.
  iree_hal_pool_t* allocation_pool;
  // HAL queue-allocation flags for the domain storage.
  iree_hal_alloca_flags_t alloca_flags;
  // Semaphores expressing domain-allocation backpressure.
  iree_hal_semaphore_list_t wait_semaphore_list;
  // Semaphores signaled when domain storage is safe for first access.
  iree_hal_semaphore_list_t signal_semaphore_list;
  // Diagnostics sink for materialization lifecycle events.
  id4_pipeline_diagnostics_sink_t* diagnostics_sink;
} id4_pipeline_parameter_materialization_acquire_options_t;

// Acquires undefined queue-ordered storage matching one planned parameter
// domain. The allocation wait list may be empty; the signal list must be
// nonempty. The returned materialization owns the asynchronous allocation
// independently of every other parameter domain. Callers must populate the
// target buffer after the acquire-readiness edge, publish it, and explicitly
// retire it after releasing every complete binding that retains it.
iree_status_t id4_pipeline_parameter_materialization_acquire(
    const id4_pipeline_parameter_materialization_acquire_options_t* options,
    iree_allocator_t host_allocator,
    id4_pipeline_parameter_materialization_t** out_materialization);

// Retains |materialization| for the caller.
void id4_pipeline_parameter_materialization_retain(
    id4_pipeline_parameter_materialization_t* materialization);

// Releases |materialization| from the caller. The final reference may only be
// released after id4_pipeline_parameter_materialization_complete_retirement or
// id4_pipeline_parameter_materialization_abort succeeds.
void id4_pipeline_parameter_materialization_release(
    id4_pipeline_parameter_materialization_t* materialization);

// Queries borrowed target state before publication. The returned buffers and
// semaphore list remain valid until publication or retirement begins.
iree_status_t id4_pipeline_parameter_materialization_query_target(
    const id4_pipeline_parameter_materialization_t* materialization,
    id4_pipeline_parameter_materialization_target_t* out_target);

// Abandons an unpublished domain after failed population.
//
// This is a synchronous error-unwind operation. |wait_semaphore_list| must
// dominate acquisition and every accepted operation that may access either
// materialization buffer. The function waits until those operations terminate,
// reclaims the domain allocation through ordinary buffer release, and
// transitions the materialization to its retired state. No queue operation is
// submitted. The caller must own the only materialization reference and may
// release it after this function returns.
iree_status_t id4_pipeline_parameter_materialization_abort(
    id4_pipeline_parameter_materialization_t* materialization,
    iree_hal_semaphore_list_t wait_semaphore_list,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink);

// Publishes completely initialized domain contents. |wait_semaphore_list| must
// dominate every write to the target buffer. |signal_semaphore_list| is
// retained as the immutable readiness edge for complete parameter bindings.
iree_status_t id4_pipeline_parameter_materialization_publish(
    id4_pipeline_parameter_materialization_t* materialization,
    iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_list_t signal_semaphore_list,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink);

// Queries borrowed binding state after publication. The returned plan, buffer,
// and semaphore list remain valid until retirement begins.
iree_status_t id4_pipeline_parameter_materialization_query_binding(
    const id4_pipeline_parameter_materialization_t* materialization,
    id4_pipeline_parameter_materialization_binding_t* out_binding);

// Schedules domain retirement after every complete binding has been
// released. The caller must own the only materialization reference and provide
// waits that dominate every in-flight use. The signals are published after the
// allocation is available for pool reuse. The materialization retains the
// retirement edge and allocation pool; its final reference must remain live
// until every supplied retirement signal has reached its payload value.
iree_status_t id4_pipeline_parameter_materialization_retire(
    id4_pipeline_parameter_materialization_t* materialization,
    iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_dealloca_flags_t dealloca_flags,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink);

// Schedules domain retirement on the device and queue recorded by the domain
// allocation, waits for its independently signaled completion, and completes
// host-side retirement. The caller must own the only materialization reference
// and provide waits that dominate every in-flight use. A rejected queue
// submission leaves the published materialization available for another
// retirement attempt. An asynchronous queue failure is terminal and returned
// to the caller.
iree_status_t id4_pipeline_parameter_materialization_retire_and_wait(
    id4_pipeline_parameter_materialization_t* materialization,
    iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_dealloca_flags_t dealloca_flags,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink);

// Completes host-side retirement after every retirement signal has reached its
// payload value. This call never waits. The final materialization reference may
// only be released after this succeeds.
iree_status_t id4_pipeline_parameter_materialization_complete_retirement(
    id4_pipeline_parameter_materialization_t* materialization);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_PIPELINE_PARAMETER_MATERIALIZATION_H_
