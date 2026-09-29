// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_LOOM_SERVE_MODULE_H_
#define IREE_EXPERIMENTAL_LOOM_SERVE_MODULE_H_

#include "experimental/loom_serve/execution.h"
#include "iree/module/hal/types.h"
#include "iree/vm/module.h"

#ifdef __cplusplus
extern "C" {
#endif

// A prepared stage exposed as one fixed-signature native VM import.
typedef struct loom_serve_stage_t {
  // Export name in the runner module, copied during module creation.
  iree_string_view_t name;
  // Reusable commands retained by the module.
  iree_hal_command_buffer_t* command_buffer;
  // Exact number of rebindable slots, including explicit workspace slots.
  uint16_t binding_count;
} loom_serve_stage_t;

// Creates the runner-local native "runner" module. Each stage exports
// (hal.buffer, ... N slots ...) -> i64. Linking checks these ordinary VM
// signatures; native calls borrow the actual HAL buffers and enqueue without
// waiting for device completion. Null buffers are rejected before submission.
// Returned values identify accepted submissions in |execution| only. The host
// retains that context when interpreting results; they are not global handles.
//
// The module retains |execution| and stage commands. Its declarations are
// immutable. Per-invocation binding scratch lives in the process storage, so
// sessions sharing one model process need no VM state or allocation of their
// own. Registered HAL types must outlive this module and all its references.
iree_status_t loom_serve_module_create(const iree_hal_module_types_t* types,
                                       loom_serve_execution_t* execution,
                                       iree_host_size_t stage_count,
                                       const loom_serve_stage_t* stages,
                                       iree_allocator_t host_allocator,
                                       iree_vm_module_t** out_module);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_EXPERIMENTAL_LOOM_SERVE_MODULE_H_
