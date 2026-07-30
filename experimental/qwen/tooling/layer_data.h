// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_QWEN_TOOLING_LAYER_DATA_H_
#define EXPERIMENTAL_QWEN_TOOLING_LAYER_DATA_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_io_file_contents_t iree_io_file_contents_t;

// Fixed hidden width of the Qwen layer execution boundary.
#define QWEN_TOOLING_LAYER_HIDDEN_SIZE 2048

// Raw dense-F32 input and expected output for one Qwen layer invocation.
//
// Both files remain mapped until deinitialization. The fixture format contains
// no header: each file is exactly token_count * 2048 little-endian IEEE F32
// elements in the runtime's token-major hidden-state layout.
typedef struct qwen_tooling_layer_data_t {
  // Mapped input hidden-state file.
  iree_io_file_contents_t* input_contents;
  // Mapped expected output hidden-state file.
  iree_io_file_contents_t* expected_contents;
  // Exact hidden-state byte length derived from the token count.
  iree_host_size_t byte_length;
} qwen_tooling_layer_data_t;

// Numeric comparison statistics for one completed layer invocation.
typedef struct qwen_tooling_layer_comparison_t {
  // Number of F32 elements compared.
  iree_host_size_t element_count;
  // Number of elements outside the requested tolerance.
  iree_host_size_t mismatch_count;
  // Index of the first mismatched element, or |element_count| on success.
  iree_host_size_t first_mismatch_index;
  // Actual value at |first_mismatch_index|.
  float first_actual_value;
  // Expected value at |first_mismatch_index|.
  float first_expected_value;
  // Largest finite absolute error observed.
  float maximum_absolute_error;
  // Largest finite relative error observed.
  float maximum_relative_error;
} qwen_tooling_layer_comparison_t;

// Returns the exact dense-F32 hidden-state byte length for |token_count|.
iree_status_t qwen_tooling_layer_hidden_state_byte_length(
    iree_host_size_t token_count, iree_host_size_t* out_byte_length);

// Maps and validates raw layer input and expected-output files.
iree_status_t qwen_tooling_layer_data_initialize(
    iree_string_view_t input_path, iree_string_view_t expected_path,
    iree_host_size_t token_count, iree_allocator_t host_allocator,
    qwen_tooling_layer_data_t* out_data);

// Releases the mapped files owned by |data|.
void qwen_tooling_layer_data_deinitialize(qwen_tooling_layer_data_t* data);

// Returns the mapped input hidden-state payload.
iree_const_byte_span_t qwen_tooling_layer_data_input(
    const qwen_tooling_layer_data_t* data);

// Compares an exact-size F32 output with the mapped expected payload.
//
// Matching NaNs compare equal. All other values pass when their absolute error
// is at most absolute_tolerance + relative_tolerance * abs(expected).
iree_status_t qwen_tooling_layer_data_compare(
    const qwen_tooling_layer_data_t* data, iree_const_byte_span_t actual_data,
    float absolute_tolerance, float relative_tolerance,
    qwen_tooling_layer_comparison_t* out_comparison);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_QWEN_TOOLING_LAYER_DATA_H_
