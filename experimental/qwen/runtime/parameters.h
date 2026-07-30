// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_QWEN_RUNTIME_PARAMETERS_H_
#define EXPERIMENTAL_QWEN_RUNTIME_PARAMETERS_H_

#include "experimental/qwen/runtime/model_shape.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/io/parameter_index.h"
#include "iree/io/parameter_provider.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Number of indexed GGUF payloads in the fixed model.
#define QWEN_PARAMETER_COUNT 579

// Maximum formatted fixed-schema parameter key length including a terminator.
#define QWEN_PARAMETER_KEY_CAPACITY 64

// Quantized storage selected by the encoded tensor length.
typedef enum qwen_quantized_storage_e {
  // Fixed GGML Q4_K super-block storage.
  QWEN_QUANTIZED_STORAGE_Q4_K = 0,
  // Fixed GGML Q6_K super-block storage.
  QWEN_QUANTIZED_STORAGE_Q6_K = 1,
} qwen_quantized_storage_t;

// Exact subrange of the resident model allocation.
typedef struct qwen_parameter_span_t {
  // Byte offset from the start of the model allocation.
  iree_device_size_t offset;
  // Encoded payload length in bytes.
  iree_device_size_t length;
} qwen_parameter_span_t;

// Fixed resident parameter allocation statistics.
typedef struct qwen_parameter_statistics_t {
  // Number of original encoded GGUF parameter bytes.
  iree_device_size_t encoded_parameter_bytes;
  // Padding inserted between encoded parameter payloads.
  iree_device_size_t parameter_padding_bytes;
  // Immutable non-GGUF auxiliary bytes appended to the model allocation.
  iree_device_size_t immutable_auxiliary_bytes;
  // Complete resident allocation size including parameters and auxiliaries.
  iree_device_size_t allocation_bytes;
} qwen_parameter_statistics_t;

// Typed parameter spans for one transformer layer.
typedef struct qwen_layer_parameters_t {
  // Attention RMS normalization weights.
  qwen_parameter_span_t attention_norm;
  // Quantized query projection weights.
  qwen_parameter_span_t query;
  // Quantized key projection weights.
  qwen_parameter_span_t key;
  // Quantized value projection weights.
  qwen_parameter_span_t value;
  // Query RMS normalization weights.
  qwen_parameter_span_t query_norm;
  // Key RMS normalization weights.
  qwen_parameter_span_t key_norm;
  // Quantized attention output projection weights.
  qwen_parameter_span_t attention_output;
  // Feed-forward RMS normalization weights.
  qwen_parameter_span_t feed_forward_norm;
  // Dense router projection weights.
  qwen_parameter_span_t router;
  // Quantized expert gate projection weights.
  qwen_parameter_span_t expert_gate;
  // Quantized expert up projection weights.
  qwen_parameter_span_t expert_up;
  // Quantized expert down projection weights.
  qwen_parameter_span_t expert_down;
  // Shared storage kind selected by value and expert-down encoded lengths.
  qwen_quantized_storage_t value_and_down_storage;
} qwen_layer_parameters_t;

// Complete fixed parameter layout constructed from an IREE parameter index.
typedef struct qwen_parameter_layout_t {
  // Quantized token embedding table.
  qwen_parameter_span_t token_embedding;
  // Per-layer parameter spans.
  qwen_layer_parameters_t layers[QWEN_MODEL_LAYER_COUNT];
  // Final RMS normalization weights.
  qwen_parameter_span_t output_norm;
  // Quantized vocabulary projection weights.
  qwen_parameter_span_t output;
  // Immutable RoPE inverse-frequency table.
  qwen_parameter_span_t rope_inverse_frequencies;
  // Allocation statistics derived while packing the spans.
  qwen_parameter_statistics_t statistics;
} qwen_parameter_layout_t;

// Builds the fixed typed layout after validating all index entries.
//
// The index must contain exactly QWEN_PARAMETER_COUNT entries. Every fixed key
// and encoded length is validated, and each layer's value and expert-down
// tensors must select the same Q4_K or Q6_K storage.
iree_status_t qwen_parameter_layout_build(
    iree_io_parameter_index_t* parameter_index,
    qwen_parameter_layout_t* out_layout);

// Enumerates one source key and target span in deterministic packing order.
//
// |key_storage| receives any formatted layer key. The returned |out_key| is
// valid until |key_storage| is modified.
iree_status_t qwen_parameter_layout_enumerate(
    const qwen_parameter_layout_t* layout, iree_host_size_t index,
    char key_storage[QWEN_PARAMETER_KEY_CAPACITY], iree_string_view_t* out_key,
    iree_io_parameter_span_t* out_span);

// Populates the fixed RoPE inverse-frequency table.
void qwen_parameter_calculate_rope_inverse_frequencies(
    float out_values[QWEN_MODEL_ROPE_FREQUENCY_COUNT]);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_QWEN_RUNTIME_PARAMETERS_H_
