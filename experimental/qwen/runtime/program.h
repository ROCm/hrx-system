// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_QWEN_RUNTIME_PROGRAM_H_
#define EXPERIMENTAL_QWEN_RUNTIME_PROGRAM_H_

#include "experimental/qwen/runtime/model.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Prepared immutable Qwen execution program.
//
// A program owns its compiled executables and one reusable indirect command
// buffer. Program objects are thread-compatible and permit one in-flight issue.
// The caller externally synchronizes issue operations on the same program;
// callers requiring concurrent issues create another program object.
typedef struct qwen_program_t qwen_program_t;

// Forward declaration of request state consumed by a program issue.
typedef struct qwen_request_t qwen_request_t;

// Mathematical scope recorded into a program.
typedef enum qwen_program_kind_e {
  // Invalid program kind.
  QWEN_PROGRAM_KIND_INVALID = 0,
  // Executes one transformer layer over uploaded hidden states.
  QWEN_PROGRAM_KIND_LAYER = 1,
  // Executes embedding, all layers, and greedy selection for a token chunk.
  QWEN_PROGRAM_KIND_PREFILL = 2,
  // Executes embedding, all layers, and greedy selection for one decode token.
  QWEN_PROGRAM_KIND_DECODE = 3,
} qwen_program_kind_t;

// Options controlling program specialization and command recording.
typedef struct qwen_program_options_t {
  // Size of this structure in bytes.
  iree_host_size_t structure_size;
  // Optional extension chain; must be NULL when no extensions are used.
  const void* next;
  // Mathematical scope recorded by the program.
  qwen_program_kind_t kind;
  // Model layer selected by a layer program; zero for full-model programs.
  iree_host_size_t layer_index;
  // Exact active token rows consumed by each issue.
  iree_host_size_t token_count;
  // Exact initialized K/V rows visible to attention.
  iree_host_size_t context_count;
  // Request token-storage capacity compatible with this program.
  iree_host_size_t token_capacity;
  // Request K/V storage capacity compatible with this program.
  iree_host_size_t context_capacity;
  // HAL command-buffer mode used for recording and profiling.
  iree_hal_command_buffer_mode_t command_buffer_mode;
} qwen_program_options_t;

// Initializes |out_options| with conservative layer-program defaults.
IREE_API_EXPORT void qwen_program_options_initialize(
    qwen_program_options_t* out_options);

// Prepares a reusable program against |model|.
//
// Kernel specialization and compilation are synchronous cold-path operations.
// The program retains |model| and may be prepared while the model's
// asynchronous parameter gather is still in flight.
IREE_API_EXPORT iree_status_t qwen_program_prepare(
    qwen_model_t* model, const qwen_program_options_t* options,
    iree_allocator_t host_allocator, qwen_program_t** out_program);

// Retains |program| for the caller.
IREE_API_EXPORT void qwen_program_retain(qwen_program_t* program);

// Releases |program| and its compiled executables and command buffer.
IREE_API_EXPORT void qwen_program_release(qwen_program_t* program);

// Issues |program| against compatible |request| state.
//
// The issue waits for model residency, request readiness, and every caller
// semaphore. It queue-allocates transient storage, executes the recorded
// command buffer, queue-deallocates the transient storage, and only then
// publishes |signal_semaphore_list|. A second issue before the prior issue
// completes fails with FAILED_PRECONDITION.
IREE_API_EXPORT iree_status_t
qwen_program_issue(qwen_program_t* program, qwen_request_t* request,
                   iree_hal_semaphore_list_t wait_semaphore_list,
                   iree_hal_semaphore_list_t signal_semaphore_list);

// Returns the exact active token count specialized into |program|.
IREE_API_EXPORT iree_host_size_t
qwen_program_token_count(const qwen_program_t* program);

// Returns the number of Loom dispatches recorded per issue.
IREE_API_EXPORT iree_host_size_t
qwen_program_dispatch_count(const qwen_program_t* program);

// Returns the peak transient allocation size in bytes.
IREE_API_EXPORT iree_device_size_t
qwen_program_transient_byte_length(const qwen_program_t* program);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_QWEN_RUNTIME_PROGRAM_H_
