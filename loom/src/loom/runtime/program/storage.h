// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_RUNTIME_PROGRAM_STORAGE_H_
#define LOOM_RUNTIME_PROGRAM_STORAGE_H_

#include "iree/hal/vm/types.h"
#include "iree/vm/buffer.h"
#include "iree/vm/module.h"
#include "loom/runtime/execution/execution.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_runtime_storage_module_options_t {
  // Execution owner for ordered device copies and fills.
  loom_runtime_execution_t* execution;
  // Stable model-owned roots; the table and its references outlive the module.
  struct {
    // Number of declared roots.
    iree_host_size_t count;
    // Borrowed roots in cold declaration order.
    iree_hal_buffer_t* const* values;
  } buffers;
  // Host feedback is readable only during successful source collection.
  struct {
    // Number of declared destinations.
    iree_host_size_t count;
    // Borrowed read-only views, outliving the module and all accepted writes.
    iree_vm_buffer_t* const* values;
  } feedback;
} loom_runtime_storage_module_options_t;

// Creates the source "storage" capability:
//   buffer(i32 ordinal) -> hal.buffer
//   copy(hal.buffer source, i64 source_offset, hal.buffer target,
//        i64 target_offset, i64 length) -> i64 work
//   fill(hal.buffer target, i64 offset, i64 length, i32 pattern) -> i64 work
//   completed(i32 slot) -> buffer
// Device operations are available only inside the owner's submission scope;
// completed is available only after successful feedback readiness. Roots remain
// stable between bounded executions. Resource allocation and range ownership
// belong to the hosting service, not to this submission capability.
iree_status_t loom_runtime_storage_module_create(
    const iree_hal_vm_types_t* hal_types, const iree_vm_ref_types_t* vm_types,
    loom_runtime_storage_module_options_t options,
    iree_allocator_t host_allocator, iree_vm_module_t** out_module);

typedef enum loom_runtime_storage_access_e {
  LOOM_RUNTIME_STORAGE_ACCESS_NONE = 0,
  LOOM_RUNTIME_STORAGE_ACCESS_SUBMIT,
  LOOM_RUNTIME_STORAGE_ACCESS_COLLECT,
} loom_runtime_storage_access_t;

// The serialized owner brackets one synchronous source invocation. Collection
// follows successful device and feedback readiness; NONE revokes the scope.
void loom_runtime_storage_module_set_access(
    iree_vm_module_t* module, loom_runtime_storage_access_t access);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_RUNTIME_PROGRAM_STORAGE_H_
