// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_EXECUTION_DISPATCH_H_
#define LOOM_SERVE_RUNTIME_EXECUTION_DISPATCH_H_

#include "loom/serve/runtime/execution/execution.h"
#include "loom/serve/runtime/execution/group.h"

#ifdef __cplusplus
extern "C" {
#endif

// Immutable native stage. Retains its executable and the source argument to
// HAL binding map, without recording a command buffer or allocating workspace.
typedef struct loom_serve_dispatch_t loom_serve_dispatch_t;

// Prepares a buffer-only entry using its compiled launch configuration. Scalar
// invocation parameters are not part of runner.execute_N; specialization and
// device-read input packets carry those model values instead. Reflection is
// checked here before the stage can accept work.
iree_status_t loom_serve_dispatch_create(
    iree_hal_executable_t* executable, iree_hal_executable_function_t function,
    iree_hal_dispatch_config_t config, iree_allocator_t host_allocator,
    loom_serve_dispatch_t** out_dispatch);

void loom_serve_dispatch_retain(loom_serve_dispatch_t* dispatch);
void loom_serve_dispatch_release(loom_serve_dispatch_t* dispatch);

// Whether another owner retains this descriptor. The caller holds a reference
// and serializes new borrows; concurrent releases may delay collection only.
bool loom_serve_dispatch_is_shared(const loom_serve_dispatch_t* dispatch);

// Number of logical buffer arguments, in authored launch signature order.
uint16_t loom_serve_dispatch_binding_count(
    const loom_serve_dispatch_t* dispatch);

// Submits on the model's ordinary work timeline. bindings has exactly the
// prepared arity; scratch has that many entries and is private to the current
// invocation. HAL captures scratch and retains resources before returning.
iree_status_t loom_serve_dispatch_issue(
    const loom_serve_dispatch_t* dispatch, loom_serve_execution_t* execution,
    iree_hal_buffer_binding_table_t bindings, iree_hal_buffer_ref_t* scratch,
    uint64_t* out_value);

// Describes the same launch as dispatch_issue for an execution group. scratch
// receives the native binding order and is copied by group append.
loom_serve_group_node_t loom_serve_dispatch_group_node(
    const loom_serve_dispatch_t* dispatch,
    iree_hal_buffer_binding_table_t bindings, iree_hal_buffer_ref_t* scratch);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_RUNTIME_EXECUTION_DISPATCH_H_
