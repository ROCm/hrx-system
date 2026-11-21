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
#include "loom/runtime/execution/retirement.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_runtime_preparation_t loom_runtime_preparation_t;

// Stable source-owned roots, independent of the service's protocol buffers.
// The service retires source consumers and these references before joining the
// retirement scope passed to initialize. The table survives that join.
typedef struct loom_runtime_storage_t {
  // Number of initialized table entries in cold declaration order.
  iree_host_size_t count;
  // Owned tracked roots; release clears entries without freeing the table.
  iree_hal_buffer_t** buffers;
  // Optional native host source, retained through every declared root's use.
  iree_hal_pool_t* host_pool;
} loom_runtime_storage_t;

// Materializes the cold declarations, batching each native allocation source
// on execution. Host declarations use cached host backing accessible by the
// execution's dispatch and transfer families. The group outlives storage.
// No declarations allocate no storage. Partial ownership returned on failure
// follows the same release, retirement join, and deinitialize sequence.
iree_status_t loom_runtime_storage_initialize(
    const loom_runtime_preparation_t* preparation,
    iree_hal_device_group_t* group, loom_runtime_execution_t* execution,
    iree_hal_pool_t* pool, loom_runtime_retirement_t* retirement,
    iree_allocator_t host_allocator, loom_runtime_storage_t* out_storage);

// Drops owned roots before the caller joins final native retirement. Accepts
// partial initialization and repeated release.
void loom_runtime_storage_release(loom_runtime_storage_t* storage);

// Frees the stable table after source consumers and tracked roots retire.
void loom_runtime_storage_deinitialize(loom_runtime_storage_t* storage,
                                       iree_allocator_t host_allocator);

typedef struct loom_runtime_storage_module_options_t {
  // Released service modules declare owning HAL arguments. Both conventions
  // use the same synchronous operations and preserve queue ownership.
  bool owned_hal_arguments;
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
//   buffer(i32 ordinal) -> ref<hal.buffer>
//   copy(ptr<hal.buffer> source, i64 source_offset, ptr<hal.buffer> target,
//        i64 target_offset, i64 length) -> i64 work
//   fill(ptr<hal.buffer> target, i64 offset, i64 length, i32 pattern)
//       -> i64 work
//   completed(i32 slot) -> buffer
// Buffer results transfer an owner; transfer arguments borrow handles while
// accepted queue work retains the buffers independently through completion.
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
