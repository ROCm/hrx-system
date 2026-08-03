// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_QWEN_RUNTIME_PROGRAM_H_
#define EXPERIMENTAL_QWEN_RUNTIME_PROGRAM_H_

#include "experimental/qwen/runtime/request.h"
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

// Number of active context positions covered by one reusable decode program.
#define QWEN_PROGRAM_DECODE_CONTEXT_CLASS_SIZE 64

// Maximum context supported by the cooperative decode-attention adapter.
#define QWEN_PROGRAM_DECODE_CONTEXT_LIMIT 2048

// Number of reusable decode program classes spanning the supported context.
#define QWEN_PROGRAM_DECODE_CONTEXT_CLASS_COUNT \
  (QWEN_PROGRAM_DECODE_CONTEXT_LIMIT / QWEN_PROGRAM_DECODE_CONTEXT_CLASS_SIZE)

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
  // Exact visible K/V rows for layer/prefill, or decode class upper bound.
  iree_host_size_t context_count;
  // Request token-storage capacity compatible with this program.
  iree_host_size_t token_capacity;
  // Request K/V storage capacity compatible with this program.
  iree_host_size_t context_capacity;
  // Optional request storage behavior addressed by the recorded program.
  qwen_request_flags_t request_flags;
  // HAL command-buffer mode used for recording and profiling.
  iree_hal_command_buffer_mode_t command_buffer_mode;
} qwen_program_options_t;

// Initializes |out_options| with conservative layer-program defaults.
IREE_API_EXPORT void qwen_program_options_initialize(
    qwen_program_options_t* out_options);

// Returns the decode class upper bound containing |context_base|.
IREE_API_EXPORT iree_host_size_t
qwen_program_decode_context_class(iree_host_size_t context_base);

// Prepares a reusable program against |model|.
//
// Kernel specialization and compilation are synchronous cold-path operations.
// The program retains |model| and may be prepared while the model's
// asynchronous parameter gather is still in flight.
IREE_API_EXPORT iree_status_t qwen_program_prepare(
    qwen_model_t* model, const qwen_program_options_t* options,
    iree_allocator_t host_allocator, qwen_program_t** out_program);

// Prepares multiple reusable programs through one JIT request union.
//
// Every program is described before compilation begins. Exact executable
// requests and compatible code identities are shared across the complete
// union, then each command buffer is recorded after all executables are ready.
// On failure every output remains NULL and all partial programs are released.
IREE_API_EXPORT iree_status_t qwen_program_prepare_batch(
    qwen_model_t* model, iree_host_size_t program_count,
    const qwen_program_options_t* options, iree_allocator_t host_allocator,
    qwen_program_t** out_programs);

// Retains |program| for the caller.
IREE_API_EXPORT void qwen_program_retain(qwen_program_t* program);

// Releases |program| and its compiled executables and command buffer.
IREE_API_EXPORT void qwen_program_release(qwen_program_t* program);

// Issues |program| against compatible |request| state.
//
// Model identity, storage capacities, and optional request flags must exactly
// match the values used to prepare the program.
//
// The issue waits for model residency, request readiness, and every caller
// semaphore. It queue-allocates transient storage, initializes any
// schedule-owned synchronization state, executes the recorded command buffer,
// queue-deallocates the transient storage, and only then publishes
// |signal_semaphore_list|. A second issue before the prior issue completes
// fails with FAILED_PRECONDITION.
//
// A full-model issue publishes its selected token and next position into
// request-local device state before signaling completion. A compatible decode
// program may be reused at consecutive positions and consumes that state
// directly through a semaphore dependency; no host read or token reset is
// required. The request also retains a host-observable token copy for
// qwen_request_read_selected_token.
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
