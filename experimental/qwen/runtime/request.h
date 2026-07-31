// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_QWEN_RUNTIME_REQUEST_H_
#define EXPERIMENTAL_QWEN_RUNTIME_REQUEST_H_

#include "experimental/qwen/runtime/model.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/tokenizer/types.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Mutable request-local Qwen execution state.
//
// A request owns token IDs, hidden states, all layer K/V caches, control data,
// and authoritative host-visible result storage. Request operations form one
// internal timeline; callers may add dependencies and observe completion with
// their own timeline. Requests are thread-compatible: callers externally
// synchronize reset, issue, and read operations on the same request or use
// distinct request objects.
typedef struct qwen_request_t qwen_request_t;

// Options controlling request-local persistent storage.
typedef struct qwen_request_options_t {
  // Size of this structure in bytes.
  iree_host_size_t structure_size;
  // Optional extension chain; must be NULL when no extensions are used.
  const void* next;
  // Maximum physical token rows held in input and metadata spans.
  iree_host_size_t token_capacity;
  // Maximum number of K/V rows retained by each layer.
  iree_host_size_t context_capacity;
} qwen_request_options_t;

// Initializes |out_options| with conservative request defaults.
IREE_API_EXPORT void qwen_request_options_initialize(
    qwen_request_options_t* out_options);

// Asynchronously creates request-local persistent state.
//
// The request retains |model|. Device-local storage uses an
// indeterminate-lifetime queue allocation. The caller signals are published
// only after the allocation is ready for use.
IREE_API_EXPORT iree_status_t qwen_request_create(
    qwen_model_t* model, const qwen_request_options_t* options,
    iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_list_t signal_semaphore_list,
    iree_allocator_t host_allocator, qwen_request_t** out_request);

// Retains |request| for the caller.
IREE_API_EXPORT void qwen_request_retain(qwen_request_t* request);

// Releases |request| after its last externally observed operation completes.
IREE_API_EXPORT void qwen_request_release(qwen_request_t* request);

// Asynchronously publishes dense F32 hidden-state input at |context_base|.
//
// |hidden_state_data| must contain one or more complete 2048-element
// little-endian IEEE F32 rows, up to the request token capacity. The active row
// count plus |context_base| must fit the request context capacity. The reset
// captures the host data before returning and publishes caller signals after
// the device upload completes.
IREE_API_EXPORT iree_status_t qwen_request_reset_hidden_state(
    qwen_request_t* request, iree_host_size_t context_base,
    iree_const_byte_span_t hidden_state_data,
    iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_list_t signal_semaphore_list);

// Asynchronously publishes full-model token IDs at |context_base|.
//
// |token_ids| must contain one or more values, up to the request token
// capacity. The active count plus |context_base| must fit the request context
// capacity. Every value is validated in [0, QWEN_MODEL_VOCABULARY_SIZE) before
// any queue operation is submitted. The reset captures the host data before
// returning and publishes caller signals after the device upload completes.
IREE_API_EXPORT iree_status_t qwen_request_reset_tokens(
    qwen_request_t* request, iree_host_size_t context_base,
    iree_tokenizer_token_id_list_t token_ids,
    iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_list_t signal_semaphore_list);

// Copies the completed layer output into |target_data|.
//
// The caller must first observe the signal from qwen_program_issue.
// |target_data| must have the exact hidden-state byte length. This invalidates
// the persistent host mapping when required by the HAL memory type and then
// performs the host copy.
IREE_API_EXPORT iree_status_t qwen_request_read_hidden_state(
    qwen_request_t* request, iree_byte_span_t target_data);

// Reads the observation copy of a completed full-model selected token.
//
// The caller must first observe the signal from qwen_program_issue. This
// invalidates the authoritative four-byte host-visible span when required by
// the HAL memory type and copies its I32 token into |out_token_id|. Model
// continuation consumes the device-local copy and does not depend on this read.
IREE_API_EXPORT iree_status_t qwen_request_read_selected_token(
    qwen_request_t* request, iree_tokenizer_token_id_t* out_token_id);

// Returns the maximum physical token rows owned by |request|.
IREE_API_EXPORT iree_host_size_t
qwen_request_token_capacity(const qwen_request_t* request);

// Returns the K/V row capacity owned by |request|.
IREE_API_EXPORT iree_host_size_t
qwen_request_context_capacity(const qwen_request_t* request);

// Returns the total persistent request storage in bytes.
IREE_API_EXPORT iree_device_size_t
qwen_request_persistent_byte_length(const qwen_request_t* request);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_QWEN_RUNTIME_REQUEST_H_
