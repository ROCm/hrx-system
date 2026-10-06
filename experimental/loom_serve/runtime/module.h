// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_LOOM_SERVE_RUNTIME_MODULE_H_
#define IREE_EXPERIMENTAL_LOOM_SERVE_RUNTIME_MODULE_H_

#include "experimental/loom_serve/runtime/execution.h"
#include "iree/module/hal/types.h"
#include "iree/vm/module.h"

#ifdef __cplusplus
extern "C" {
#endif

// A prepared stage selected by its index in one model residency.
typedef struct loom_serve_stage_t {
  // Reusable commands retained by the module.
  iree_hal_command_buffer_t* command_buffer;
  // Exact number of rebindable slots, including explicit workspace slots.
  uint16_t binding_count;
} loom_serve_stage_t;

// Cold capabilities shared by every invocation of one model program.
typedef struct loom_serve_module_options_t {
  // Largest typed execute_N import exposed, independent of loaded stage count.
  // Every arity from zero through this value is available, including branches
  // whose stages are not loaded. UINT16_MAX is not a valid capacity.
  uint16_t binding_capacity;
  // Immutable command table; indices are local to this module, not handles.
  struct {
    // Number of prepared stages.
    iree_host_size_t count;
    // Stage descriptors copied and commands retained at creation.
    const loom_serve_stage_t* values;
  } stages;
  // Fixed feedback destinations shared by serialized model invocations.
  struct {
    // Number of addressable feedback slots.
    iree_host_size_t count;
    // Descriptors are copied; payload storage remains borrowed through all
    // accepted feedback, including partial submission failure and teardown.
    const iree_byte_span_t* values;
  } feedback;
} loom_serve_module_options_t;

// Creates the runner-local native "runner" module. execute_N exports
// (i32 stage, hal.buffer, ... N slots ...) -> i64. Linking checks reference
// types; calls check the stage's exact arity and reject null buffers before
// enqueuing without waiting. feedback exports (i32 slot, hal.buffer source,
// i64 source_offset, i64 length) -> i64, downloading into a registered host
// span from its beginning. Host bounds are checked here; HAL checks device
// ranges. Feedback advances only the independent feedback timeline.
// Returned values identify accepted submissions in |execution| only. The host
// retains that context when interpreting results; they are not global handles.
//
// The module retains |execution| and stage commands. Its declarations are
// immutable. Per-invocation binding scratch lives in the process storage, so
// sessions sharing one model process need no VM state or allocation of their
// own. Registered HAL types must outlive this module and all its references.
iree_status_t loom_serve_module_create(const iree_hal_module_types_t* types,
                                       loom_serve_execution_t* execution,
                                       loom_serve_module_options_t options,
                                       iree_allocator_t host_allocator,
                                       iree_vm_module_t** out_module);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_EXPERIMENTAL_LOOM_SERVE_RUNTIME_MODULE_H_
