// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_QWEN_RUNTIME_MODEL_SHAPE_H_
#define EXPERIMENTAL_QWEN_RUNTIME_MODEL_SHAPE_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Number of transformer layers in the fixed model.
#define QWEN_MODEL_LAYER_COUNT 48

// Width of every residual-stream row.
#define QWEN_MODEL_HIDDEN_SIZE 2048

// Number of query heads in each attention layer.
#define QWEN_MODEL_QUERY_HEAD_COUNT 32

// Number of key/value heads in each attention layer.
#define QWEN_MODEL_KEY_VALUE_HEAD_COUNT 4

// Number of channels in each attention head.
#define QWEN_MODEL_HEAD_SIZE 128

// Number of routed feed-forward experts in each layer.
#define QWEN_MODEL_EXPERT_COUNT 128

// Number of experts selected for each token.
#define QWEN_MODEL_ROUTE_COUNT 8

// Width of one expert's SwiGLU output.
#define QWEN_MODEL_EXPERT_INTERMEDIATE_SIZE 768

// Number of output vocabulary rows.
#define QWEN_MODEL_VOCABULARY_SIZE 151936

// Number of vocabulary logits reduced by each endpoint projection workgroup.
#define QWEN_MODEL_VOCABULARY_PARTIAL_WIDTH 8

// Number of endpoint maximum pairs finalized into one selected token.
#define QWEN_MODEL_VOCABULARY_PARTIAL_COUNT \
  (QWEN_MODEL_VOCABULARY_SIZE / QWEN_MODEL_VOCABULARY_PARTIAL_WIDTH)

// Token terminating ordinary generation.
#define QWEN_MODEL_END_OF_SEQUENCE_TOKEN 151645

// Number of F32 values in the NEOX inverse-frequency table.
#define QWEN_MODEL_ROPE_FREQUENCY_COUNT (QWEN_MODEL_HEAD_SIZE / 2)

// Largest physical prefill microbatch accepted by the current kernel corpus.
#define QWEN_MODEL_MAX_PHYSICAL_TOKEN_COUNT 2048

// Largest K/V context accepted by the current attention kernel corpus.
#define QWEN_MODEL_MAX_CONTEXT_CAPACITY 32768

// Returns the byte length of one dense F32 hidden-state tensor.
iree_status_t qwen_model_hidden_state_byte_length(
    iree_host_size_t token_count, iree_device_size_t* out_byte_length);

// Returns the byte length of one layer's F16 K cache or V cache.
iree_status_t qwen_model_layer_cache_byte_length(
    iree_host_size_t context_capacity, iree_device_size_t* out_byte_length);

// Returns the byte length of all layer-local F16 K and V caches.
iree_status_t qwen_model_all_cache_byte_length(
    iree_host_size_t context_capacity, iree_device_size_t* out_byte_length);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_QWEN_RUNTIME_MODEL_SHAPE_H_
