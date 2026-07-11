// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_PIPELINE_PROGRAM_MATRIX_H_
#define EXPERIMENTAL_ID4_PIPELINE_PROGRAM_MATRIX_H_

#include "experimental/id4/pipeline/program.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Physical matrix storage layout visible at a Loom kernel boundary.
typedef enum id4_pipeline_program_matrix_layout_e {
  // Invalid or unspecified matrix layout.
  ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_INVALID = 0,
  // Dense row-major matrix storage.
  ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_ROW_MAJOR = 1,
  // Logical KxN RHS stored as a dense row-major NxK matrix.
  ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_RHS_TRANSPOSED_ROW_MAJOR = 2,
  // RHS storage packed as consecutive 16x16 KxN tiles.
  ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_RHS_TILE_16X16 = 3,
} id4_pipeline_program_matrix_layout_t;

// Scale tensor semantics applied to a stored matrix operand.
typedef enum id4_pipeline_program_matrix_scale_layout_e {
  // Invalid or unspecified scale layout.
  ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_INVALID = 0,
  // No scale tensor is applied.
  ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_NONE = 1,
  // One F32 scale is applied to each logical output row.
  ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_OUTPUT_ROW = 2,
  // One F32 scale is applied to each logical 128x128 output/input block.
  ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_OUTPUT_INPUT_BLOCK_128X128 = 3,
} id4_pipeline_program_matrix_scale_layout_t;

// Elementwise operation fused after a matrix contraction.
typedef enum id4_pipeline_program_matrix_epilogue_e {
  // Invalid or unspecified matrix epilogue.
  ID4_PIPELINE_PROGRAM_MATRIX_EPILOGUE_INVALID = 0,
  // Stores the matrix contraction result without another operation.
  ID4_PIPELINE_PROGRAM_MATRIX_EPILOGUE_NONE = 1,
  // Adds a same-shaped residual tensor after output-type rounding.
  ID4_PIPELINE_PROGRAM_MATRIX_EPILOGUE_ADD = 2,
} id4_pipeline_program_matrix_epilogue_t;

// Semantic tensor contract for C = A * B, independent of weight storage.
typedef struct id4_pipeline_program_matrix_problem_t {
  // Number of semantically valid M rows.
  uint32_t valid_m;
  // Number of allocated and dispatched M rows, including padding.
  uint32_t m_capacity;
  // Number of logical output columns N.
  uint32_t n;
  // Contracted input dimension K.
  uint32_t k;
  // Scalar element type of the MxK activation matrix.
  id4_pipeline_program_dtype_t input_dtype;
  // Physical layout of the MxK activation matrix.
  id4_pipeline_program_matrix_layout_t input_layout;
  // Scalar element type used by matrix accumulators.
  id4_pipeline_program_dtype_t accumulator_dtype;
  // Elementwise operation applied after the contraction.
  id4_pipeline_program_matrix_epilogue_t epilogue;
  // Scalar element type of the MxN output matrix.
  id4_pipeline_program_dtype_t output_dtype;
  // Physical layout of the MxN output matrix.
  id4_pipeline_program_matrix_layout_t output_layout;
} id4_pipeline_program_matrix_problem_t;

// Provider-side storage for one logical KxN parameter matrix.
typedef struct id4_pipeline_program_matrix_parameter_t {
  // Provider tensor containing logical NxK weight rows.
  id4_pipeline_program_parameter_source_t weight;
  // Physical layout of the provider weight tensor.
  id4_pipeline_program_matrix_layout_t weight_layout;
  // Provider scale tensor, or an invalid source when scale_layout is NONE.
  id4_pipeline_program_parameter_source_t scale;
  // Scale semantics applied to the provider weight tensor.
  id4_pipeline_program_matrix_scale_layout_t scale_layout;
} id4_pipeline_program_matrix_parameter_t;

// Selected execution storage for a prepared KxN weight matrix.
typedef struct id4_pipeline_program_matrix_execution_t {
  // Scalar storage type consumed by the Loom kernel.
  id4_pipeline_program_dtype_t weight_dtype;
  // Physical weight layout consumed by the Loom kernel.
  id4_pipeline_program_matrix_layout_t weight_layout;
  // Scalar type of the scale tensor consumed by the Loom kernel.
  id4_pipeline_program_dtype_t scale_dtype;
  // Scale semantics implemented by the Loom kernel.
  id4_pipeline_program_matrix_scale_layout_t scale_layout;
} id4_pipeline_program_matrix_execution_t;

// Program tensor roles consumed by a plain matrix contraction.
typedef struct id4_pipeline_program_matrix_operands_t {
  // Initialized MxK activation tensor.
  id4_pipeline_program_tensor_t input;
  // Provider-side weight and scale parameter storage.
  id4_pipeline_program_matrix_parameter_t parameter;
  // Initialized MxN addend tensor, or invalid when the epilogue is NONE.
  id4_pipeline_program_tensor_t addend;
  // Uninitialized MxN output tensor written by the contraction.
  id4_pipeline_program_tensor_t output;
} id4_pipeline_program_matrix_operands_t;

// Options for authoring one semantic matrix contraction.
typedef struct id4_pipeline_program_matrix_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Stable operation name used for diagnostics.
  iree_string_view_t name;
  // Semantic matrix contract independent of selected execution storage.
  id4_pipeline_program_matrix_problem_t problem;
  // Program tensor roles bound to the selected implementation.
  id4_pipeline_program_matrix_operands_t operands;
} id4_pipeline_program_matrix_options_t;

// Selects parameter preparation and authors a full-coverage implementation.
iree_status_t id4_pipeline_program_matrix(
    id4_pipeline_program_builder_t* builder,
    const id4_pipeline_program_matrix_options_t* options);

// Program tensor roles consumed by a matrix with prepared weight storage.
typedef struct id4_pipeline_program_matrix_prepared_operands_t {
  // Initialized MxK activation tensor.
  id4_pipeline_program_tensor_t input;
  // Initialized KxN weight tensor in the requested execution layout.
  id4_pipeline_program_tensor_t weight;
  // Initialized scale tensor, or invalid when execution scale_layout is NONE.
  id4_pipeline_program_tensor_t scale;
  // Initialized MxN addend tensor, or invalid when the epilogue is NONE.
  id4_pipeline_program_tensor_t addend;
  // Uninitialized MxN output tensor written by the contraction.
  id4_pipeline_program_tensor_t output;
} id4_pipeline_program_matrix_prepared_operands_t;

// Options for authoring a matrix from already-prepared execution storage.
typedef struct id4_pipeline_program_matrix_prepared_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Stable operation name used for diagnostics.
  iree_string_view_t name;
  // Semantic matrix contract.
  id4_pipeline_program_matrix_problem_t problem;
  // Explicit execution storage contract of the prepared weight tensor.
  id4_pipeline_program_matrix_execution_t execution;
  // Prepared program tensor roles bound to the selected implementation.
  id4_pipeline_program_matrix_prepared_operands_t operands;
} id4_pipeline_program_matrix_prepared_options_t;

// Authors a full-coverage implementation using prepared execution storage.
iree_status_t id4_pipeline_program_matrix_prepared(
    id4_pipeline_program_builder_t* builder,
    const id4_pipeline_program_matrix_prepared_options_t* options);

// Program tensor roles consumed by a fused SwiGLU projection pair.
typedef struct id4_pipeline_program_swiglu_operands_t {
  // Initialized MxK activation tensor shared by both projections.
  id4_pipeline_program_tensor_t input;
  // Provider-side gate-projection weight and scale storage.
  id4_pipeline_program_matrix_parameter_t gate_parameter;
  // Provider-side up-projection weight and scale storage.
  id4_pipeline_program_matrix_parameter_t up_parameter;
  // Uninitialized MxN output tensor receiving silu(gate) * up.
  id4_pipeline_program_tensor_t output;
} id4_pipeline_program_swiglu_operands_t;

// Options for authoring one fused SwiGLU projection pair.
typedef struct id4_pipeline_program_swiglu_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Stable operation name used for diagnostics.
  iree_string_view_t name;
  // Shared contract; output_dtype governs projection and product rounding.
  id4_pipeline_program_matrix_problem_t projection;
  // Program tensor roles bound to the selected implementation.
  id4_pipeline_program_swiglu_operands_t operands;
} id4_pipeline_program_swiglu_options_t;

// Selects and authors a full-coverage Loom SwiGLU implementation of |options|.
iree_status_t id4_pipeline_program_swiglu(
    id4_pipeline_program_builder_t* builder,
    const id4_pipeline_program_swiglu_options_t* options);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_PIPELINE_PROGRAM_MATRIX_H_
