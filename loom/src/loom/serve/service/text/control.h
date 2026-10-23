// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_CONTROL_H_
#define LOOM_SERVE_SERVICE_TEXT_CONTROL_H_

#include "loom/serve/runtime/program/module.h"
#include "loom/serve/service/text/source.h"
#include "loom/serve/service/text/state.h"

#ifdef __cplusplus
extern "C" {
#endif

// Host-side protocol buffers shared by the single model invocation. Their
// storage outlives final queue payload access, independently of the VM program.
enum {
  TEXT_HOST_PLAN = 0,
  TEXT_HOST_INPUT = 1,
  TEXT_HOST_METADATA = 2,
  TEXT_HOST_TOKENS = 3,
  TEXT_HOST_OUTPUTS = 4,
  TEXT_HOST_RESULTS = 5,
  TEXT_HOST_NEXT_RESULTS = 6,
  TEXT_HOST_PROGRESS = 7,
  TEXT_HOST_STATE_BINDINGS = 8,
  TEXT_HOST_STATE_ORIGINS = 9,
  TEXT_HOST_BUFFER_COUNT = 10,
};

// Linked model-wide VM control and its borrowed native capabilities. The
// serialized model owner excludes concurrent invocation and joins CPU users
// before unlinking; pending device payloads retain the host buffers separately.
typedef struct loom_serve_text_control_t {
  // Native stage module retained independently of its linked program.
  iree_vm_module_t* native_module;
  // Shared VM process; rows hold no private VM instances.
  loom_serve_program_t* program;
  // Source-owned isolated input/decode invocation.
  iree_vm_function_t step;
  // Source-owned packed execution sequence.
  iree_vm_function_t epoch_step;
  // Semantic spans to packets and a source-owned view of exact upload ranges.
  iree_vm_function_t encode_epoch;
  // Retired feedback to semantic row progress.
  iree_vm_function_t publish_epoch;
  // Row-table encoding after recurrent slot ownership changes.
  iree_vm_function_t encode_state;
  // Request options to retained row policy.
  iree_vm_function_t bind_request;
  // Source-owned ranges for independent known-input activation capture.
  iree_vm_function_t capture_epoch;
  // Source-owned per-request options for a known-input model.
  iree_vm_function_t request_options;
  // Owned known-input defaults; generative defaults belong to chat_policy.
  iree_vm_buffer_t* defaults;
  // Chat policy borrowing this program and owning its source-returned defaults.
  loom_serve_text_chat_policy_t chat_policy;
  // Queue-visible host payloads retained until actual device retirement.
  struct {
    // Owned buffers in the protocol slot order above.
    iree_vm_buffer_t* buffers[TEXT_HOST_BUFFER_COUNT];
    // Writable mappings valid for the corresponding buffer lifetime.
    iree_byte_span_t bytes[TEXT_HOST_BUFFER_COUNT];
  } host;
} loom_serve_text_control_t;

// Links prepared source against the existing state and native stages. Source,
// execution, and stage storage outlive the control program. Partial ownership
// is returned on failure and follows the same unlink/retirement cleanup.
iree_status_t loom_serve_text_control_initialize(
    const loom_serve_text_source_t* source, loom_serve_execution_t* execution,
    iree_hal_queue_t* transfer, const loom_serve_text_state_t* state,
    iree_host_size_t shape_count, const loom_serve_packing_shape_t* shapes,
    iree_host_size_t stage_count, const loom_serve_stage_t* stages,
    iree_allocator_t host_allocator, loom_serve_text_control_t* out_control);

// Drops VM references before native retirement, without freeing queue-visible
// host payloads. The owner first joins CPU invocation and stops submissions.
void loom_serve_text_control_unlink(loom_serve_text_control_t* control);

// Releases host payloads after actual queue retirement. Unlink must precede
// this call. Both cleanup operations accept partial initialization.
void loom_serve_text_control_deinitialize(loom_serve_text_control_t* control);

// Resolves source-owned known-input request policy on an exclusive VM loan.
// Source may retain the cloned body; a returned options buffer is owned.
iree_status_t loom_serve_text_control_request_options(
    loom_serve_text_control_t* control, const loom_serve_text_source_t* source,
    iree_string_view_t body, uint64_t request_id,
    iree_allocator_t host_allocator, iree_vm_buffer_t** out_options);

// Evaluates the checked model source against a validated native cohort, then
// copies and validates its output ranges before any retained row is mutated.
iree_status_t loom_serve_text_control_capture(
    loom_serve_text_control_t* control, const loom_serve_text_source_t* source,
    iree_host_size_t shape_index, iree_host_size_t span_count,
    iree_device_size_t residual_length, iree_allocator_t host_allocator,
    loom_serve_text_capture_t** out_capture);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_SERVE_SERVICE_TEXT_CONTROL_H_
