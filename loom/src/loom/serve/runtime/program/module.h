// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_PROGRAM_MODULE_H_
#define LOOM_SERVE_RUNTIME_PROGRAM_MODULE_H_

#include "iree/module/hal/types.h"
#include "iree/vm/module.h"
#include "loom/serve/runtime/execution/dispatch.h"
#include "loom/target/arch/cmd/program.h"

#ifdef __cplusplus
extern "C" {
#endif

// A prepared stage selected by its index in one model residency.
typedef struct loom_serve_stage_t {
  // Reusable commands retained by the module; NULL for a direct stage.
  iree_hal_command_buffer_t* command_buffer;
  // Exact number of model-provided buffers, excluding private workspace.
  uint16_t binding_count;
  // Compiler-owned private binding, inserted by the runner when nonempty.
  loom_cmd_program_transient_requirement_t transient;
  // Retained native stage, mutually exclusive with command_buffer. Direct
  // stages have no compiler-planned global transient workspace.
  loom_serve_dispatch_t* dispatch;
} loom_serve_stage_t;

// Cold capabilities shared by every invocation of one model program.
typedef struct loom_serve_module_options_t {
  // Largest typed execute_N import exposed, independent of loaded stage count.
  // Every arity from zero through this value is available, including branches
  // whose stages are not loaded. UINT16_MAX is not a valid capacity.
  uint16_t binding_capacity;
  // Initial command table; indices are local to this module, not handles.
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
    // Initial descriptors are copied; payload storage remains borrowed through
    // every accepted feedback targeting it.
    const iree_byte_span_t* values;
  } feedback;
} loom_serve_module_options_t;

// Creates the runner-local native "runner" module. execute_N exports
// (i32 stage, hal.buffer, ... N slots ...) -> i64. Linking checks reference
// types; calls check the stage's exact arity and reject null buffers before
// enqueuing without waiting. Private transient bindings are queue-allocated
// around each execute and are not model VM arguments. feedback exports
// (i32 slot, hal.buffer source,
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

// Rebinds one registered feedback slot between serialized model invocations.
// The caller has joined every prior feedback submission targeting the slot;
// their captured host spans remain live independently. The new payload remains
// borrowed through every accepted feedback targeting it. Empty spans preserve
// the registered slot while making nonempty feedback calls fail their bound.
void loom_serve_module_bind_feedback(iree_vm_module_t* module,
                                     iree_host_size_t slot,
                                     iree_byte_span_t destination);

// Rebinds the native stages of a module created above without changing its VM
// declarations, process storage, feedback destinations or execution timeline.
// The nonempty table may grow or shrink; each stage's model-buffer arity must
// fit the original binding capacity. Commands, direct executables and private
// workspace requirements may change. The owner updates source stage indices
// together with the table, never midway through an invocation using them.
// All validation and workspace reservation precede publication. Failure leaves
// the previous bindings usable; success retains the new commands/executables.
// Accepted queue work retains its previous resources independently.
//
// The model owner serializes this call with VM invocations at a semantic model
// boundary. No device wait occurs. Buffer contents and parameter/state layout
// compatibility belong to that owner, not the native submission module.
iree_status_t loom_serve_module_bind_stages(iree_vm_module_t* module,
                                            iree_host_size_t stage_count,
                                            const loom_serve_stage_t* stages);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_RUNTIME_PROGRAM_MODULE_H_
