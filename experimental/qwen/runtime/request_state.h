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

// Deterministic request-local persistent storage layout.
typedef struct qwen_request_storage_layout_t {
  // Dense F32 residual stream read and updated by a layer program.
  qwen_request_span_t hidden_state;
  // I32 logical position for each physical token.
  qwen_request_span_t positions;
  // I64 K cache row selected for each physical token.
  qwen_request_span_t key_cache_indices;
  // I64 V cache row selected for each physical token.
  qwen_request_span_t value_cache_indices;
  // F16 causal attention mask for the physical token square.
  qwen_request_span_t attention_mask;
  // All layer-local F16 key caches.
  qwen_request_span_t key_cache;
  // All layer-local F16 value caches.
  qwen_request_span_t value_cache;
  // Byte length uploaded by qwen_request_reset_hidden_state.
  iree_device_size_t reset_upload_byte_length;
  // Byte length of one layer-local K or V cache.
  iree_device_size_t layer_cache_byte_length;
  // Complete request-local device allocation size.
  iree_device_size_t persistent_byte_length;
} qwen_request_storage_layout_t;

// Calculates the fixed request storage layout.
iree_status_t qwen_request_storage_layout_calculate(
    iree_host_size_t token_count, iree_host_size_t context_capacity,
    qwen_request_storage_layout_t* out_layout);

// Returns the model retained by |request|.
qwen_model_t* qwen_request_model(const qwen_request_t* request);

// Returns request-local device storage.
iree_hal_buffer_t* qwen_request_storage_buffer(const qwen_request_t* request);

// Returns persistently mapped layer-output staging storage.
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

// Commits a successfully submitted program result to the request timeline.
//
// |signal_value| must be exactly one greater than the current timeline value
// and must be published by the terminal queue operation before this call.
void qwen_request_commit_program_signal(qwen_request_t* request,
                                        uint64_t signal_value);

// Permanently fails the request timeline after a partial submission failure.
void qwen_request_fail(qwen_request_t* request, iree_status_t status);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_QWEN_RUNTIME_REQUEST_STATE_H_
