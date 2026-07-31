// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_QWEN_RUNTIME_REQUEST_STATE_H_
#define EXPERIMENTAL_QWEN_RUNTIME_REQUEST_STATE_H_

#include "experimental/qwen/runtime/request.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Byte span within request-local persistent storage.
typedef struct qwen_request_span_t {
  // Byte offset from the start of request storage.
  iree_device_size_t offset;
  // Exact accessible byte length.
  iree_device_size_t length;
} qwen_request_span_t;

// Compact device control record uploaded before each physical issue.
typedef struct qwen_request_control_t {
  // Zero-based logical position and direct no-ring cache row for token zero.
  int32_t context_base;
} qwen_request_control_t;
static_assert(sizeof(qwen_request_control_t) == sizeof(int32_t),
              "Qwen request control must be exactly one I32 word");

// Authoritative input representation established by the latest reset.
typedef enum qwen_request_input_kind_e {
  // No reset has established valid request input.
  QWEN_REQUEST_INPUT_KIND_INVALID = 0,
  // Dense F32 hidden states for a layer program.
  QWEN_REQUEST_INPUT_KIND_HIDDEN_STATE = 1,
  // Validated I32 token IDs for a full-model program.
  QWEN_REQUEST_INPUT_KIND_TOKEN_IDS = 2,
} qwen_request_input_kind_t;

// Host-readable result representation published by the latest program issue.
typedef enum qwen_request_result_kind_e {
  // No program has published a result after the latest reset.
  QWEN_REQUEST_RESULT_KIND_INVALID = 0,
  // Dense F32 hidden states from a layer program.
  QWEN_REQUEST_RESULT_KIND_HIDDEN_STATE = 1,
  // One selected I32 token from a full-model program.
  QWEN_REQUEST_RESULT_KIND_SELECTED_TOKEN = 2,
} qwen_request_result_kind_t;

// Deterministic request-local persistent storage layout.
typedef struct qwen_request_storage_layout_t {
  // Dense F32 residual stream read and updated by a layer program.
  qwen_request_span_t hidden_state;
  // Validated I32 token IDs consumed by the embedding producer.
  qwen_request_span_t token_ids;
  // Compact control record uploaded by request reset.
  qwen_request_span_t control;
  // Device-derived I32 logical position for each physical token.
  qwen_request_span_t positions;
  // Device-derived I64 K cache row selected for each physical token.
  qwen_request_span_t key_cache_indices;
  // Device-derived I64 V cache row selected for each physical token.
  qwen_request_span_t value_cache_indices;
  // Device-derived F16 causal mask over token by context-capacity rows.
  qwen_request_span_t attention_mask;
  // All layer-local F16 key caches.
  qwen_request_span_t key_cache;
  // All layer-local F16 value caches.
  qwen_request_span_t value_cache;
  // Byte length uploaded by either request reset operation.
  iree_device_size_t reset_upload_byte_length;
  // Complete request-state binding length visible to recorded dispatches.
  iree_device_size_t dispatch_state_byte_length;
  // Byte length of one layer-local K or V cache.
  iree_device_size_t layer_cache_byte_length;
  // Complete request-local device allocation size.
  iree_device_size_t persistent_byte_length;
} qwen_request_storage_layout_t;

// Validates one active input shape against fixed request capacities.
iree_status_t qwen_request_active_shape_validate(
    iree_host_size_t token_capacity, iree_host_size_t context_capacity,
    iree_host_size_t active_token_count, iree_host_size_t context_base);

// Calculates the fixed request storage layout.
iree_status_t qwen_request_storage_layout_calculate(
    iree_host_size_t token_capacity, iree_host_size_t context_capacity,
    qwen_request_storage_layout_t* out_layout);

// Returns the model retained by |request|.
qwen_model_t* qwen_request_model(const qwen_request_t* request);

// Returns request-local device storage.
iree_hal_buffer_t* qwen_request_storage_buffer(const qwen_request_t* request);

// Returns the persistently mapped authoritative result storage.
iree_hal_buffer_t* qwen_request_output_staging_buffer(
    const qwen_request_t* request);

// Returns the immutable request storage layout.
const qwen_request_storage_layout_t* qwen_request_storage_layout(
    const qwen_request_t* request);

// Returns the internal request timeline semaphore.
iree_hal_semaphore_t* qwen_request_timeline_semaphore(
    const qwen_request_t* request);

// Returns the latest request timeline value reserved by a submitted operation.
uint64_t qwen_request_timeline_value(const qwen_request_t* request);

// Returns the input representation established by the latest reset.
qwen_request_input_kind_t qwen_request_input_kind(
    const qwen_request_t* request);

// Returns the active token count published by the latest successful reset.
iree_host_size_t qwen_request_active_token_count(const qwen_request_t* request);

// Returns the context base published by the latest successful reset.
iree_host_size_t qwen_request_context_base(const qwen_request_t* request);

// Commits a successfully submitted layer result to the request timeline.
//
// |signal_value| must be exactly one greater than the current timeline value
// and must be published by the terminal queue operation before this call.
void qwen_request_commit_hidden_state_signal(qwen_request_t* request,
                                             uint64_t signal_value);

// Commits a full-model result that already published its continuation token.
//
// The device has written the token into request-local input storage and the
// matching context base into request control before |signal_value| completes.
// If |next_context_base| equals request context capacity, the result remains
// readable but no further model issue can consume it.
void qwen_request_commit_selected_token_signal(
    qwen_request_t* request, uint64_t signal_value,
    iree_host_size_t next_context_base);

// Permanently fails the request timeline after a partial submission failure.
void qwen_request_fail(qwen_request_t* request, iree_status_t status);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_QWEN_RUNTIME_REQUEST_STATE_H_
