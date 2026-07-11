// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/program_matrix.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

enum {
  ID4_PIPELINE_PROGRAM_MATRIX_CONFIG_CAPACITY = 5,
  ID4_PIPELINE_PROGRAM_MATRIX_CONFIG_VALUE_CAPACITY = 16,
};

typedef enum id4_pipeline_program_matrix_operation_e {
  ID4_PIPELINE_PROGRAM_MATRIX_OPERATION_CONTRACTION = 0,
  ID4_PIPELINE_PROGRAM_MATRIX_OPERATION_SWIGLU = 1,
} id4_pipeline_program_matrix_operation_t;

typedef struct id4_pipeline_program_matrix_candidate_config_t {
  // Config key receiving the semantically valid M extent; empty when unused.
  iree_string_view_t valid_m_key;
  // Config key receiving the allocated M extent; empty when unused.
  iree_string_view_t m_capacity_key;
  // Config key receiving the dispatched M extent.
  iree_string_view_t dispatch_m_key;
  // Config key receiving K.
  iree_string_view_t k_key;
  // Config key receiving N.
  iree_string_view_t n_key;
} id4_pipeline_program_matrix_candidate_config_t;

typedef struct id4_pipeline_program_matrix_candidate_t {
  // Semantic matrix-family operation implemented by the candidate.
  id4_pipeline_program_matrix_operation_t operation;
  // Scalar type stored by the provider weight tensor.
  id4_pipeline_program_dtype_t source_weight_dtype;
  // Physical layout stored by the provider weight tensor.
  id4_pipeline_program_matrix_layout_t source_weight_layout;
  // Scalar type stored by the provider scale tensor.
  id4_pipeline_program_dtype_t source_scale_dtype;
  // Scale semantics stored by the provider.
  id4_pipeline_program_matrix_scale_layout_t source_scale_layout;
  // Prepare-time transformation producing execution weight storage.
  id4_pipeline_program_parameter_encoding_t weight_encoding;
  // Input scalar type accepted by the candidate.
  id4_pipeline_program_dtype_t input_dtype;
  // Input physical layout accepted by the candidate.
  id4_pipeline_program_matrix_layout_t input_layout;
  // Weight scalar storage type consumed by the candidate.
  id4_pipeline_program_dtype_t weight_dtype;
  // Weight physical layout consumed by the candidate.
  id4_pipeline_program_matrix_layout_t weight_layout;
  // Scale scalar type consumed by the candidate.
  id4_pipeline_program_dtype_t scale_dtype;
  // Scale semantics implemented by the candidate.
  id4_pipeline_program_matrix_scale_layout_t scale_layout;
  // Accumulator scalar type used by the candidate.
  id4_pipeline_program_dtype_t accumulator_dtype;
  // Fused contraction epilogue implemented by the candidate.
  id4_pipeline_program_matrix_epilogue_t epilogue;
  // Output scalar type produced by the candidate.
  id4_pipeline_program_dtype_t output_dtype;
  // Output physical layout produced by the candidate.
  id4_pipeline_program_matrix_layout_t output_layout;
  // Minimum allocated M extent accepted by the candidate.
  uint32_t minimum_m_capacity;
  // Required allocated M divisibility.
  uint32_t m_multiple;
  // Minimum N extent accepted by the candidate.
  uint32_t minimum_n;
  // Required N divisibility.
  uint32_t n_multiple;
  // Required K divisibility.
  uint32_t k_multiple;
  // Relative preference among fused or composed schedules for the same result.
  uint32_t selection_priority;
  // Loom kernel implementing the candidate.
  id4_pipeline_kernel_ref_t kernel;
  // Config-key mapping used by the candidate module.
  id4_pipeline_program_matrix_candidate_config_t config;
} id4_pipeline_program_matrix_candidate_t;

typedef struct id4_pipeline_program_swiglu_prepared_operands_t {
  // Initialized MxK activation tensor shared by both projections.
  id4_pipeline_program_tensor_t input;
  // Initialized KxN gate-projection weight tensor.
  id4_pipeline_program_tensor_t gate_weight;
  // Initialized gate scale tensor, or invalid when scales are not consumed.
  id4_pipeline_program_tensor_t gate_scale;
  // Initialized KxN up-projection weight tensor.
  id4_pipeline_program_tensor_t up_weight;
  // Initialized up scale tensor, or invalid when scales are not consumed.
  id4_pipeline_program_tensor_t up_scale;
  // Uninitialized MxN output tensor receiving silu(gate) * up.
  id4_pipeline_program_tensor_t output;
} id4_pipeline_program_swiglu_prepared_operands_t;

typedef enum id4_pipeline_program_matrix_composition_operation_e {
  ID4_PIPELINE_PROGRAM_MATRIX_COMPOSITION_OPERATION_ADD = 0,
  ID4_PIPELINE_PROGRAM_MATRIX_COMPOSITION_OPERATION_SWIGLU = 1,
} id4_pipeline_program_matrix_composition_operation_t;

typedef struct id4_pipeline_program_matrix_composition_candidate_t {
  // Semantic operation completed after the plain contraction dispatches.
  id4_pipeline_program_matrix_composition_operation_t operation;
  // Execution weight type required from the selected contraction schedule.
  id4_pipeline_program_dtype_t weight_dtype;
  // Execution weight layout required from the selected contraction schedule.
  id4_pipeline_program_matrix_layout_t weight_layout;
  // Execution scale layout required from the selected contraction schedule.
  id4_pipeline_program_matrix_scale_layout_t scale_layout;
  // Activation scalar type accepted by the pointwise kernel.
  id4_pipeline_program_dtype_t input_dtype;
  // Activation physical layout accepted by the pointwise kernel.
  id4_pipeline_program_matrix_layout_t input_layout;
  // Output scalar type produced by the pointwise kernel.
  id4_pipeline_program_dtype_t output_dtype;
  // Output physical layout produced by the pointwise kernel.
  id4_pipeline_program_matrix_layout_t output_layout;
  // Loom kernel completing the composed operation.
  id4_pipeline_kernel_ref_t kernel;
  // Config key receiving the dispatched M extent.
  iree_string_view_t m_key;
  // Config key receiving the output N extent.
  iree_string_view_t n_key;
} id4_pipeline_program_matrix_composition_candidate_t;

typedef struct id4_pipeline_program_matrix_schedule_t {
  // Candidate implementing the fused operation or its contraction prefix.
  const id4_pipeline_program_matrix_candidate_t* primary_candidate;
  // Pointwise continuation for a composed schedule, or NULL when fused.
  const id4_pipeline_program_matrix_composition_candidate_t*
      composition_candidate;
} id4_pipeline_program_matrix_schedule_t;

static const id4_pipeline_program_matrix_composition_candidate_t
    id4_pipeline_program_matrix_composition_candidates[] = {
        {
            .operation = ID4_PIPELINE_PROGRAM_MATRIX_COMPOSITION_OPERATION_ADD,
            .weight_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
            .weight_layout = ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_RHS_TILE_16X16,
            .scale_layout =
                ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_OUTPUT_INPUT_BLOCK_128X128,
            .input_dtype = ID4_PIPELINE_PROGRAM_DTYPE_BF16,
            .input_layout = ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_ROW_MAJOR,
            .output_dtype = ID4_PIPELINE_PROGRAM_DTYPE_BF16,
            .output_layout = ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_ROW_MAJOR,
            .kernel =
                {
                    IREE_SVL("qwen3_vl/residual_add_f32"),
                    IREE_SVL("id4_qwen3_vl_residual_add_bf16"),
                },
            .m_key = IREE_SVL("id4.qwen3_vl.residual_add.token_count"),
            .n_key = IREE_SVL("id4.qwen3_vl.residual_add.hidden_size"),
        },
        {
            .operation =
                ID4_PIPELINE_PROGRAM_MATRIX_COMPOSITION_OPERATION_SWIGLU,
            .weight_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
            .weight_layout = ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_RHS_TILE_16X16,
            .scale_layout =
                ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_OUTPUT_INPUT_BLOCK_128X128,
            .input_dtype = ID4_PIPELINE_PROGRAM_DTYPE_BF16,
            .input_layout = ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_ROW_MAJOR,
            .output_dtype = ID4_PIPELINE_PROGRAM_DTYPE_BF16,
            .output_layout = ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_ROW_MAJOR,
            .kernel =
                {
                    IREE_SVL("qwen3_vl/silu_gate_f32"),
                    IREE_SVL("id4_qwen3_vl_silu_gate_bf16"),
                },
            .m_key = IREE_SVL("id4.qwen3_vl.silu_gate.token_count"),
            .n_key = IREE_SVL("id4.qwen3_vl.silu_gate.intermediate_size"),
        },
};

static const id4_pipeline_program_matrix_candidate_t id4_pipeline_program_matrix_candidates[] = {
    // Qwen block-scaled plain contractions, fastest schedule first.
    {
        .operation = ID4_PIPELINE_PROGRAM_MATRIX_OPERATION_CONTRACTION,
        .source_weight_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
        .source_weight_layout =
            ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_RHS_TRANSPOSED_ROW_MAJOR,
        .source_scale_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F32,
        .source_scale_layout =
            ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_OUTPUT_INPUT_BLOCK_128X128,
        .weight_encoding =
            ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_FP8_E4M3_LINEAR_RHS_TILE,
        .input_dtype = ID4_PIPELINE_PROGRAM_DTYPE_BF16,
        .input_layout = ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_ROW_MAJOR,
        .weight_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
        .weight_layout = ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_RHS_TILE_16X16,
        .scale_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F32,
        .scale_layout =
            ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_OUTPUT_INPUT_BLOCK_128X128,
        .accumulator_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F32,
        .epilogue = ID4_PIPELINE_PROGRAM_MATRIX_EPILOGUE_NONE,
        .output_dtype = ID4_PIPELINE_PROGRAM_DTYPE_BF16,
        .output_layout = ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_ROW_MAJOR,
        .minimum_m_capacity = 128,
        .m_multiple = 128,
        .minimum_n = 128,
        .n_multiple = 128,
        .k_multiple = 128,
        .selection_priority = 4,
        .kernel =
            {
                IREE_SVL("qwen3_vl/linear_fp8_block_scaled_bf16_wmma_"
                         "m128n128_4wave_workgroup_staged"),
                IREE_SVL("id4_qwen3_vl_linear_fp8_block_scaled_bf16_"
                         "wmma_m128n128_4wave_workgroup_staged"),
            },
        .config =
            {
                .dispatch_m_key = IREE_SVL(
                    "id4.qwen3_vl.linear_fp8_block_scaled_m128n128_wmma."
                    "dispatch_token_count"),
                .k_key = IREE_SVL(
                    "id4.qwen3_vl.linear_fp8_block_scaled_m128n128_wmma."
                    "input_size"),
                .n_key = IREE_SVL(
                    "id4.qwen3_vl.linear_fp8_block_scaled_m128n128_wmma."
                    "output_size"),
            },
    },
    {
        .operation = ID4_PIPELINE_PROGRAM_MATRIX_OPERATION_CONTRACTION,
        .source_weight_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
        .source_weight_layout =
            ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_RHS_TRANSPOSED_ROW_MAJOR,
        .source_scale_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F32,
        .source_scale_layout =
            ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_OUTPUT_INPUT_BLOCK_128X128,
        .weight_encoding =
            ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_FP8_E4M3_LINEAR_RHS_TILE,
        .input_dtype = ID4_PIPELINE_PROGRAM_DTYPE_BF16,
        .input_layout = ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_ROW_MAJOR,
        .weight_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
        .weight_layout = ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_RHS_TILE_16X16,
        .scale_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F32,
        .scale_layout =
            ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_OUTPUT_INPUT_BLOCK_128X128,
        .accumulator_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F32,
        .epilogue = ID4_PIPELINE_PROGRAM_MATRIX_EPILOGUE_NONE,
        .output_dtype = ID4_PIPELINE_PROGRAM_DTYPE_BF16,
        .output_layout = ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_ROW_MAJOR,
        .minimum_m_capacity = 64,
        .m_multiple = 64,
        .minimum_n = 128,
        .n_multiple = 128,
        .k_multiple = 128,
        .selection_priority = 2,
        .kernel =
            {
                IREE_SVL("qwen3_vl/linear_fp8_block_scaled_bf16_wmma_"
                         "m64n128_4wave_workgroup_staged"),
                IREE_SVL("id4_qwen3_vl_linear_fp8_block_scaled_bf16_"
                         "wmma_m64n128_4wave_workgroup_staged"),
            },
        .config =
            {
                .dispatch_m_key = IREE_SVL(
                    "id4.qwen3_vl.linear_fp8_block_scaled_m64n128_wmma."
                    "dispatch_token_count"),
                .k_key = IREE_SVL(
                    "id4.qwen3_vl.linear_fp8_block_scaled_m64n128_wmma."
                    "input_size"),
                .n_key = IREE_SVL(
                    "id4.qwen3_vl.linear_fp8_block_scaled_m64n128_wmma."
                    "output_size"),
            },
    },
    {
        .operation = ID4_PIPELINE_PROGRAM_MATRIX_OPERATION_CONTRACTION,
        .source_weight_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
        .source_weight_layout =
            ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_RHS_TRANSPOSED_ROW_MAJOR,
        .source_scale_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F32,
        .source_scale_layout =
            ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_OUTPUT_INPUT_BLOCK_128X128,
        .weight_encoding =
            ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_FP8_E4M3_LINEAR_RHS_TILE,
        .input_dtype = ID4_PIPELINE_PROGRAM_DTYPE_BF16,
        .input_layout = ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_ROW_MAJOR,
        .weight_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
        .weight_layout = ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_RHS_TILE_16X16,
        .scale_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F32,
        .scale_layout =
            ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_OUTPUT_INPUT_BLOCK_128X128,
        .accumulator_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F32,
        .epilogue = ID4_PIPELINE_PROGRAM_MATRIX_EPILOGUE_NONE,
        .output_dtype = ID4_PIPELINE_PROGRAM_DTYPE_BF16,
        .output_layout = ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_ROW_MAJOR,
        .minimum_m_capacity = 512,
        .m_multiple = 128,
        .minimum_n = 1024,
        .n_multiple = 128,
        .k_multiple = 128,
        .selection_priority = 2,
        .kernel =
            {
                IREE_SVL("qwen3_vl/linear_fp8_block_scaled_bf16_wmma_"
                         "m128n64_2wave_compact_rhs"),
                IREE_SVL("id4_qwen3_vl_linear_fp8_block_scaled_bf16_"
                         "wmma_m128n64_2wave_compact_rhs"),
            },
        .config =
            {
                .dispatch_m_key =
                    IREE_SVL("id4.qwen3_vl.linear_fp8_block_scaled_wmma."
                             "dispatch_token_count"),
                .k_key = IREE_SVL(
                    "id4.qwen3_vl.linear_fp8_block_scaled_wmma.input_size"),
                .n_key = IREE_SVL("id4.qwen3_vl.linear_fp8_block_scaled_wmma."
                                  "output_size"),
            },
    },
    {
        .operation = ID4_PIPELINE_PROGRAM_MATRIX_OPERATION_CONTRACTION,
        .source_weight_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
        .source_weight_layout =
            ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_RHS_TRANSPOSED_ROW_MAJOR,
        .source_scale_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F32,
        .source_scale_layout =
            ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_OUTPUT_INPUT_BLOCK_128X128,
        .weight_encoding =
            ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_FP8_E4M3_LINEAR_RHS_TILE,
        .input_dtype = ID4_PIPELINE_PROGRAM_DTYPE_BF16,
        .input_layout = ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_ROW_MAJOR,
        .weight_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
        .weight_layout = ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_RHS_TILE_16X16,
        .scale_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F32,
        .scale_layout =
            ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_OUTPUT_INPUT_BLOCK_128X128,
        .accumulator_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F32,
        .epilogue = ID4_PIPELINE_PROGRAM_MATRIX_EPILOGUE_NONE,
        .output_dtype = ID4_PIPELINE_PROGRAM_DTYPE_BF16,
        .output_layout = ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_ROW_MAJOR,
        .minimum_m_capacity = 64,
        .m_multiple = 64,
        .minimum_n = 128,
        .n_multiple = 128,
        .k_multiple = 128,
        .selection_priority = 1,
        .kernel =
            {
                IREE_SVL("qwen3_vl/linear_fp8_block_scaled_bf16_wmma_"
                         "m64n64_compact_rhs"),
                IREE_SVL("id4_qwen3_vl_linear_fp8_block_scaled_bf16_"
                         "wmma_m64n64_compact_rhs"),
            },
        .config =
            {
                .dispatch_m_key =
                    IREE_SVL("id4.qwen3_vl.linear_fp8_block_scaled_wmma."
                             "dispatch_token_count"),
                .k_key = IREE_SVL(
                    "id4.qwen3_vl.linear_fp8_block_scaled_wmma.input_size"),
                .n_key = IREE_SVL("id4.qwen3_vl.linear_fp8_block_scaled_wmma."
                                  "output_size"),
            },
    },
    // Qwen block-scaled contractions with rounded residual addition.
    {
        .operation = ID4_PIPELINE_PROGRAM_MATRIX_OPERATION_CONTRACTION,
        .source_weight_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
        .source_weight_layout =
            ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_RHS_TRANSPOSED_ROW_MAJOR,
        .source_scale_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F32,
        .source_scale_layout =
            ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_OUTPUT_INPUT_BLOCK_128X128,
        .weight_encoding =
            ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_FP8_E4M3_LINEAR_RHS_TILE,
        .input_dtype = ID4_PIPELINE_PROGRAM_DTYPE_BF16,
        .input_layout = ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_ROW_MAJOR,
        .weight_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
        .weight_layout = ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_RHS_TILE_16X16,
        .scale_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F32,
        .scale_layout =
            ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_OUTPUT_INPUT_BLOCK_128X128,
        .accumulator_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F32,
        .epilogue = ID4_PIPELINE_PROGRAM_MATRIX_EPILOGUE_ADD,
        .output_dtype = ID4_PIPELINE_PROGRAM_DTYPE_BF16,
        .output_layout = ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_ROW_MAJOR,
        .minimum_m_capacity = 512,
        .m_multiple = 128,
        .minimum_n = 128,
        .n_multiple = 128,
        .k_multiple = 128,
        .selection_priority = 2,
        .kernel =
            {
                IREE_SVL("qwen3_vl/linear_residual_fp8_block_scaled_bf16_"
                         "wmma_m128n64_2wave_compact_rhs"),
                IREE_SVL("id4_qwen3_vl_linear_residual_fp8_block_scaled_"
                         "bf16_wmma_m128n64_2wave_compact_rhs"),
            },
        .config =
            {
                .valid_m_key =
                    IREE_SVL("id4.qwen3_vl.linear_residual_wmma.token_count"),
                .dispatch_m_key = IREE_SVL("id4.qwen3_vl.linear_residual_"
                                           "wmma.dispatch_token_count"),
                .k_key =
                    IREE_SVL("id4.qwen3_vl.linear_residual_wmma.input_size"),
                .n_key =
                    IREE_SVL("id4.qwen3_vl.linear_residual_wmma.output_size"),
            },
    },
    {
        .operation = ID4_PIPELINE_PROGRAM_MATRIX_OPERATION_CONTRACTION,
        .source_weight_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
        .source_weight_layout =
            ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_RHS_TRANSPOSED_ROW_MAJOR,
        .source_scale_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F32,
        .source_scale_layout =
            ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_OUTPUT_INPUT_BLOCK_128X128,
        .weight_encoding =
            ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_FP8_E4M3_LINEAR_RHS_TILE,
        .input_dtype = ID4_PIPELINE_PROGRAM_DTYPE_BF16,
        .input_layout = ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_ROW_MAJOR,
        .weight_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
        .weight_layout = ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_RHS_TILE_16X16,
        .scale_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F32,
        .scale_layout =
            ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_OUTPUT_INPUT_BLOCK_128X128,
        .accumulator_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F32,
        .epilogue = ID4_PIPELINE_PROGRAM_MATRIX_EPILOGUE_ADD,
        .output_dtype = ID4_PIPELINE_PROGRAM_DTYPE_BF16,
        .output_layout = ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_ROW_MAJOR,
        .minimum_m_capacity = 512,
        .m_multiple = 64,
        .minimum_n = 128,
        .n_multiple = 128,
        .k_multiple = 128,
        .selection_priority = 1,
        .kernel =
            {
                IREE_SVL("qwen3_vl/linear_residual_fp8_block_scaled_bf16_"
                         "wmma_compact_rhs"),
                IREE_SVL("id4_qwen3_vl_linear_residual_fp8_block_scaled_"
                         "bf16_wmma_m64n64_compact_rhs"),
            },
        .config =
            {
                .valid_m_key =
                    IREE_SVL("id4.qwen3_vl.linear_residual_wmma.token_count"),
                .dispatch_m_key = IREE_SVL("id4.qwen3_vl.linear_residual_"
                                           "wmma.dispatch_token_count"),
                .k_key =
                    IREE_SVL("id4.qwen3_vl.linear_residual_wmma.input_size"),
                .n_key =
                    IREE_SVL("id4.qwen3_vl.linear_residual_wmma.output_size"),
            },
    },
    // Qwen block-scaled paired SwiGLU projections.
    {
        .operation = ID4_PIPELINE_PROGRAM_MATRIX_OPERATION_SWIGLU,
        .source_weight_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
        .source_weight_layout =
            ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_RHS_TRANSPOSED_ROW_MAJOR,
        .source_scale_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F32,
        .source_scale_layout =
            ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_OUTPUT_INPUT_BLOCK_128X128,
        .weight_encoding =
            ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_FP8_E4M3_LINEAR_RHS_TILE,
        .input_dtype = ID4_PIPELINE_PROGRAM_DTYPE_BF16,
        .input_layout = ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_ROW_MAJOR,
        .weight_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
        .weight_layout = ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_RHS_TILE_16X16,
        .scale_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F32,
        .scale_layout =
            ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_OUTPUT_INPUT_BLOCK_128X128,
        .accumulator_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F32,
        .epilogue = ID4_PIPELINE_PROGRAM_MATRIX_EPILOGUE_NONE,
        .output_dtype = ID4_PIPELINE_PROGRAM_DTYPE_BF16,
        .output_layout = ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_ROW_MAJOR,
        .minimum_m_capacity = 128,
        .m_multiple = 128,
        .minimum_n = 128,
        .n_multiple = 128,
        .k_multiple = 128,
        .selection_priority = 2,
        .kernel =
            {
                IREE_SVL("qwen3_vl/mlp_gate_up_silu_product_fp8_block_"
                         "scaled_bf16_wmma_m128n32_2wave_compact_rhs"),
                IREE_SVL("id4_qwen3_vl_mlp_gate_up_silu_product_fp8_"
                         "block_scaled_bf16_wmma_m128n32_2wave_compact_"
                         "rhs"),
            },
        .config =
            {
                .dispatch_m_key =
                    IREE_SVL("id4.qwen3_vl.mlp_gate_up_silu_product_wmma."
                             "dispatch_token_count"),
                .k_key = IREE_SVL("id4.qwen3_vl.mlp_gate_up_silu_product_"
                                  "wmma.input_size"),
                .n_key = IREE_SVL("id4.qwen3_vl.mlp_gate_up_silu_product_wmma."
                                  "intermediate_size"),
            },
    },
    {
        .operation = ID4_PIPELINE_PROGRAM_MATRIX_OPERATION_SWIGLU,
        .source_weight_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
        .source_weight_layout =
            ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_RHS_TRANSPOSED_ROW_MAJOR,
        .source_scale_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F32,
        .source_scale_layout =
            ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_OUTPUT_INPUT_BLOCK_128X128,
        .weight_encoding =
            ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_FP8_E4M3_LINEAR_RHS_TILE,
        .input_dtype = ID4_PIPELINE_PROGRAM_DTYPE_BF16,
        .input_layout = ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_ROW_MAJOR,
        .weight_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
        .weight_layout = ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_RHS_TILE_16X16,
        .scale_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F32,
        .scale_layout =
            ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_OUTPUT_INPUT_BLOCK_128X128,
        .accumulator_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F32,
        .epilogue = ID4_PIPELINE_PROGRAM_MATRIX_EPILOGUE_NONE,
        .output_dtype = ID4_PIPELINE_PROGRAM_DTYPE_BF16,
        .output_layout = ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_ROW_MAJOR,
        .minimum_m_capacity = 32,
        .m_multiple = 32,
        .minimum_n = 128,
        .n_multiple = 128,
        .k_multiple = 128,
        .selection_priority = 1,
        .kernel =
            {
                IREE_SVL("qwen3_vl/mlp_gate_up_silu_product_fp8_block_"
                         "scaled_bf16_wmma_compact_rhs"),
                IREE_SVL("id4_qwen3_vl_mlp_gate_up_silu_product_fp8_"
                         "block_scaled_bf16_wmma_m32n32_compact_rhs"),
            },
        .config =
            {
                .dispatch_m_key =
                    IREE_SVL("id4.qwen3_vl.mlp_gate_up_silu_product_wmma."
                             "dispatch_token_count"),
                .k_key = IREE_SVL("id4.qwen3_vl.mlp_gate_up_silu_product_"
                                  "wmma.input_size"),
                .n_key = IREE_SVL("id4.qwen3_vl.mlp_gate_up_silu_product_wmma."
                                  "intermediate_size"),
            },
    },
    // DiT row-scaled paired SwiGLU projections by physical RHS layout.
    {
        .operation = ID4_PIPELINE_PROGRAM_MATRIX_OPERATION_SWIGLU,
        .source_weight_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
        .source_weight_layout =
            ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_RHS_TRANSPOSED_ROW_MAJOR,
        .source_scale_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F32,
        .source_scale_layout =
            ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_OUTPUT_ROW,
        .weight_encoding = ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_DIRECT,
        .input_dtype = ID4_PIPELINE_PROGRAM_DTYPE_BF16,
        .input_layout = ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_ROW_MAJOR,
        .weight_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
        .weight_layout =
            ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_RHS_TRANSPOSED_ROW_MAJOR,
        .scale_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F32,
        .scale_layout = ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_OUTPUT_ROW,
        .accumulator_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F32,
        .epilogue = ID4_PIPELINE_PROGRAM_MATRIX_EPILOGUE_NONE,
        .output_dtype = ID4_PIPELINE_PROGRAM_DTYPE_BF16,
        .output_layout = ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_ROW_MAJOR,
        .minimum_m_capacity = 32,
        .m_multiple = 32,
        .minimum_n = 32,
        .n_multiple = 32,
        .k_multiple = 16,
        .selection_priority = 1,
        .kernel =
            {
                IREE_SVL("ideogram4/"
                         "mlp_gate_up_silu_product_fp8_bf16_wmma"),
                IREE_SVL("id4_ideogram4_mlp_gate_up_silu_product_fp8_"
                         "bf16_wmma"),
            },
        .config =
            {
                .m_capacity_key =
                    IREE_SVL("id4.ideogram4.mlp_gate_up_silu_product_fp8_wmma."
                             "token_capacity"),
                .dispatch_m_key =
                    IREE_SVL("id4.ideogram4.mlp_gate_up_silu_product_fp8_wmma."
                             "dispatch_token_count"),
                .k_key =
                    IREE_SVL("id4.ideogram4.mlp_gate_up_silu_product_fp8_wmma."
                             "input_size"),
                .n_key =
                    IREE_SVL("id4.ideogram4.mlp_gate_up_silu_product_fp8_wmma."
                             "intermediate_size"),
            },
    },
    {
        .operation = ID4_PIPELINE_PROGRAM_MATRIX_OPERATION_SWIGLU,
        .source_weight_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
        .source_weight_layout =
            ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_RHS_TRANSPOSED_ROW_MAJOR,
        .source_scale_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F32,
        .source_scale_layout =
            ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_OUTPUT_ROW,
        .weight_encoding =
            ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_FP8_E4M3_LINEAR_RHS_TILE,
        .input_dtype = ID4_PIPELINE_PROGRAM_DTYPE_BF16,
        .input_layout = ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_ROW_MAJOR,
        .weight_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
        .weight_layout = ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_RHS_TILE_16X16,
        .scale_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F32,
        .scale_layout = ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_OUTPUT_ROW,
        .accumulator_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F32,
        .epilogue = ID4_PIPELINE_PROGRAM_MATRIX_EPILOGUE_NONE,
        .output_dtype = ID4_PIPELINE_PROGRAM_DTYPE_BF16,
        .output_layout = ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_ROW_MAJOR,
        .minimum_m_capacity = 64,
        .m_multiple = 64,
        .minimum_n = 64,
        .n_multiple = 64,
        .k_multiple = 16,
        .selection_priority = 2,
        .kernel =
            {
                IREE_SVL("ideogram4/"
                         "mlp_gate_up_silu_product_fp8_bf16_wmma"),
                IREE_SVL("id4_ideogram4_mlp_gate_up_silu_product_fp8_"
                         "bf16_wmma_compact_rhs_tile_m64n64_4wave"),
            },
        .config =
            {
                .m_capacity_key =
                    IREE_SVL("id4.ideogram4.mlp_gate_up_silu_product_fp8_wmma."
                             "token_capacity"),
                .dispatch_m_key =
                    IREE_SVL("id4.ideogram4.mlp_gate_up_silu_product_fp8_wmma."
                             "dispatch_token_count"),
                .k_key =
                    IREE_SVL("id4.ideogram4.mlp_gate_up_silu_product_fp8_wmma."
                             "input_size"),
                .n_key =
                    IREE_SVL("id4.ideogram4.mlp_gate_up_silu_product_fp8_wmma."
                             "intermediate_size"),
            },
    },
    // DiT row-scaled plain contractions.
    {
        .operation = ID4_PIPELINE_PROGRAM_MATRIX_OPERATION_CONTRACTION,
        .source_weight_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
        .source_weight_layout =
            ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_RHS_TRANSPOSED_ROW_MAJOR,
        .source_scale_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F32,
        .source_scale_layout =
            ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_OUTPUT_ROW,
        .weight_encoding = ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_DIRECT,
        .input_dtype = ID4_PIPELINE_PROGRAM_DTYPE_BF16,
        .input_layout = ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_ROW_MAJOR,
        .weight_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
        .weight_layout =
            ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_RHS_TRANSPOSED_ROW_MAJOR,
        .scale_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F32,
        .scale_layout = ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_OUTPUT_ROW,
        .accumulator_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F32,
        .epilogue = ID4_PIPELINE_PROGRAM_MATRIX_EPILOGUE_NONE,
        .output_dtype = ID4_PIPELINE_PROGRAM_DTYPE_BF16,
        .output_layout = ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_ROW_MAJOR,
        .minimum_m_capacity = 128,
        .m_multiple = 128,
        .minimum_n = 64,
        .n_multiple = 64,
        .k_multiple = 16,
        .selection_priority = 2,
        .kernel =
            {
                IREE_SVL("ideogram4/linear_fp8_bf16_wmma_m128n64_2wave"),
                IREE_SVL("id4_ideogram4_linear_fp8_bf16_wmma_m128n64_"
                         "2wave"),
            },
        .config =
            {
                .m_capacity_key =
                    IREE_SVL("id4.ideogram4.linear_fp8_wmma.token_count"),
                .dispatch_m_key = IREE_SVL(
                    "id4.ideogram4.linear_fp8_wmma.dispatch_token_count"),
                .k_key = IREE_SVL("id4.ideogram4.linear_fp8_wmma.input_size"),
                .n_key = IREE_SVL("id4.ideogram4.linear_fp8_wmma.output_size"),
            },
    },
    {
        .operation = ID4_PIPELINE_PROGRAM_MATRIX_OPERATION_CONTRACTION,
        .source_weight_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
        .source_weight_layout =
            ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_RHS_TRANSPOSED_ROW_MAJOR,
        .source_scale_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F32,
        .source_scale_layout =
            ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_OUTPUT_ROW,
        .weight_encoding = ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_DIRECT,
        .input_dtype = ID4_PIPELINE_PROGRAM_DTYPE_BF16,
        .input_layout = ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_ROW_MAJOR,
        .weight_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
        .weight_layout =
            ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_RHS_TRANSPOSED_ROW_MAJOR,
        .scale_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F32,
        .scale_layout = ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_OUTPUT_ROW,
        .accumulator_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F32,
        .epilogue = ID4_PIPELINE_PROGRAM_MATRIX_EPILOGUE_NONE,
        .output_dtype = ID4_PIPELINE_PROGRAM_DTYPE_BF16,
        .output_layout = ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_ROW_MAJOR,
        .minimum_m_capacity = 32,
        .m_multiple = 32,
        .minimum_n = 32,
        .n_multiple = 32,
        .k_multiple = 16,
        .selection_priority = 1,
        .kernel =
            {
                IREE_SVL("ideogram4/linear_fp8_bf16_wmma"),
                IREE_SVL("id4_ideogram4_linear_fp8_bf16_wmma"),
            },
        .config =
            {
                .m_capacity_key =
                    IREE_SVL("id4.ideogram4.linear_fp8_wmma.token_count"),
                .dispatch_m_key = IREE_SVL(
                    "id4.ideogram4.linear_fp8_wmma.dispatch_token_count"),
                .k_key = IREE_SVL("id4.ideogram4.linear_fp8_wmma.input_size"),
                .n_key = IREE_SVL("id4.ideogram4.linear_fp8_wmma.output_size"),
            },
    },
    {
        .operation = ID4_PIPELINE_PROGRAM_MATRIX_OPERATION_CONTRACTION,
        .source_weight_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
        .source_weight_layout =
            ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_RHS_TRANSPOSED_ROW_MAJOR,
        .source_scale_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F32,
        .source_scale_layout =
            ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_OUTPUT_ROW,
        .weight_encoding =
            ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_FP8_E4M3_LINEAR_RHS_TILE,
        .input_dtype = ID4_PIPELINE_PROGRAM_DTYPE_BF16,
        .input_layout = ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_ROW_MAJOR,
        .weight_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
        .weight_layout = ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_RHS_TILE_16X16,
        .scale_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F32,
        .scale_layout = ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_OUTPUT_ROW,
        .accumulator_dtype = ID4_PIPELINE_PROGRAM_DTYPE_F32,
        .epilogue = ID4_PIPELINE_PROGRAM_MATRIX_EPILOGUE_NONE,
        .output_dtype = ID4_PIPELINE_PROGRAM_DTYPE_BF16,
        .output_layout = ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_ROW_MAJOR,
        .minimum_m_capacity = 128,
        .m_multiple = 128,
        .minimum_n = 128,
        .n_multiple = 128,
        .k_multiple = 16,
        .selection_priority = 3,
        .kernel =
            {
                IREE_SVL("ideogram4/linear_fp8_bf16_wmma_compact_rhs_"
                         "tile_m128n128_4wave_workgroup_staged"),
                IREE_SVL("id4_ideogram4_linear_fp8_bf16_wmma_compact_"
                         "rhs_tile_m128n128_4wave_workgroup_staged"),
            },
        .config =
            {
                .m_capacity_key =
                    IREE_SVL("id4.ideogram4.linear_fp8_wmma.token_count"),
                .dispatch_m_key = IREE_SVL(
                    "id4.ideogram4.linear_fp8_wmma.dispatch_token_count"),
                .k_key = IREE_SVL("id4.ideogram4.linear_fp8_wmma.input_size"),
                .n_key = IREE_SVL("id4.ideogram4.linear_fp8_wmma.output_size"),
            },
    },
};

static iree_status_t id4_pipeline_program_matrix_validate_options_size(
    iree_host_size_t actual_size, iree_host_size_t expected_size) {
  if (actual_size >= expected_size) return iree_ok_status();
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "program matrix options structure size %" PRIhsz
                          " is smaller than expected %" PRIhsz,
                          actual_size, expected_size);
}

static bool id4_pipeline_program_matrix_layout_is_valid(
    id4_pipeline_program_matrix_layout_t layout) {
  return layout == ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_ROW_MAJOR ||
         layout ==
             ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_RHS_TRANSPOSED_ROW_MAJOR ||
         layout == ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_RHS_TILE_16X16;
}

static bool id4_pipeline_program_matrix_scale_layout_is_valid(
    id4_pipeline_program_matrix_scale_layout_t layout) {
  return layout == ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_NONE ||
         layout == ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_OUTPUT_ROW ||
         layout ==
             ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_OUTPUT_INPUT_BLOCK_128X128;
}

static bool id4_pipeline_program_matrix_epilogue_is_valid(
    id4_pipeline_program_matrix_epilogue_t epilogue) {
  return epilogue == ID4_PIPELINE_PROGRAM_MATRIX_EPILOGUE_NONE ||
         epilogue == ID4_PIPELINE_PROGRAM_MATRIX_EPILOGUE_ADD;
}

static bool id4_pipeline_program_matrix_shape_equal(
    id4_pipeline_program_shape_t lhs, id4_pipeline_program_shape_t rhs) {
  if (lhs.rank != rhs.rank) return false;
  for (uint32_t i = 0; i < lhs.rank; ++i) {
    if (lhs.dims[i] != rhs.dims[i]) return false;
  }
  return true;
}

static bool id4_pipeline_program_matrix_parameter_source_is_valid(
    const id4_pipeline_program_parameter_source_t* source) {
  return !iree_string_view_is_empty(source->key) &&
         id4_pipeline_program_dtype_byte_length(source->dtype) != 0 &&
         source->shape.rank != 0;
}

static iree_status_t id4_pipeline_program_matrix_validate_problem(
    const id4_pipeline_program_matrix_problem_t* problem) {
  if (problem->valid_m == 0 || problem->m_capacity == 0 || problem->n == 0 ||
      problem->k == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program matrix M/N/K dimensions must be nonzero");
  }
  if (problem->valid_m > problem->m_capacity) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program matrix valid M %" PRIu32
                            " exceeds allocated M capacity %" PRIu32,
                            problem->valid_m, problem->m_capacity);
  }
  if (id4_pipeline_program_dtype_byte_length(problem->input_dtype) == 0 ||
      id4_pipeline_program_dtype_byte_length(problem->accumulator_dtype) == 0 ||
      id4_pipeline_program_dtype_byte_length(problem->output_dtype) == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "program matrix input, accumulator, and output dtypes must be valid");
  }
  if (!id4_pipeline_program_matrix_layout_is_valid(problem->input_layout) ||
      !id4_pipeline_program_matrix_layout_is_valid(problem->output_layout)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "program matrix input and output layouts must be valid");
  }
  if (!id4_pipeline_program_matrix_epilogue_is_valid(problem->epilogue)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program matrix epilogue must be valid");
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_matrix_validate_execution(
    const id4_pipeline_program_matrix_execution_t* execution) {
  if (id4_pipeline_program_dtype_byte_length(execution->weight_dtype) == 0 ||
      !id4_pipeline_program_matrix_layout_is_valid(execution->weight_layout)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "program matrix execution weight dtype and layout must be valid");
  }
  if (!id4_pipeline_program_matrix_scale_layout_is_valid(
          execution->scale_layout)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "program matrix execution scale layout must be valid");
  }
  const bool has_scale =
      execution->scale_layout != ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_NONE;
  if (has_scale !=
      (id4_pipeline_program_dtype_byte_length(execution->scale_dtype) != 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "program matrix execution scale dtype validity must match semantics");
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_matrix_expected_scale_shape(
    const id4_pipeline_program_matrix_problem_t* problem,
    id4_pipeline_program_matrix_scale_layout_t scale_layout,
    id4_pipeline_program_shape_t* out_shape) {
  switch (scale_layout) {
    case ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_NONE:
      *out_shape = (id4_pipeline_program_shape_t){0};
      return iree_ok_status();
    case ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_OUTPUT_ROW:
      *out_shape = id4_pipeline_program_make_shape_rank1(problem->n);
      return iree_ok_status();
    case ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_OUTPUT_INPUT_BLOCK_128X128:
      *out_shape = id4_pipeline_program_make_shape_rank2(
          (problem->n + 127u) / 128u, (problem->k + 127u) / 128u);
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "program matrix scale layout %u is invalid",
                              (uint32_t)scale_layout);
  }
}

static iree_status_t id4_pipeline_program_matrix_validate_parameter(
    const id4_pipeline_program_matrix_problem_t* problem,
    const id4_pipeline_program_matrix_parameter_t* parameter) {
  if (!id4_pipeline_program_matrix_parameter_source_is_valid(
          &parameter->weight)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program matrix provider weight is required");
  }
  if (!id4_pipeline_program_matrix_layout_is_valid(parameter->weight_layout)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program matrix provider weight layout is invalid");
  }
  const id4_pipeline_program_shape_t expected_weight_shape =
      id4_pipeline_program_make_shape_rank2(problem->n, problem->k);
  if (!id4_pipeline_program_matrix_shape_equal(parameter->weight.shape,
                                               expected_weight_shape)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "program matrix provider weight shape must be [N=%" PRIu32
        ", K=%" PRIu32 "]",
        problem->n, problem->k);
  }
  if (!id4_pipeline_program_matrix_scale_layout_is_valid(
          parameter->scale_layout)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program matrix provider scale layout is invalid");
  }
  const bool has_scale =
      parameter->scale_layout != ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_NONE;
  if (has_scale != id4_pipeline_program_matrix_parameter_source_is_valid(
                       &parameter->scale)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "program matrix provider scale validity must match scale semantics");
  }
  if (has_scale) {
    id4_pipeline_program_shape_t expected_scale_shape;
    IREE_RETURN_IF_ERROR(id4_pipeline_program_matrix_expected_scale_shape(
        problem, parameter->scale_layout, &expected_scale_shape));
    if (!id4_pipeline_program_matrix_shape_equal(parameter->scale.shape,
                                                 expected_scale_shape)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "program matrix provider scale shape does not match scale semantics");
    }
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_matrix_validate_addend(
    const id4_pipeline_program_matrix_problem_t* problem,
    id4_pipeline_program_tensor_t addend) {
  const bool has_addend =
      problem->epilogue == ID4_PIPELINE_PROGRAM_MATRIX_EPILOGUE_ADD;
  if (has_addend != id4_pipeline_program_tensor_is_valid(addend)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "program matrix addend tensor validity must match epilogue semantics");
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_matrix_validate_scale_operand(
    const id4_pipeline_program_matrix_execution_t* execution,
    id4_pipeline_program_tensor_t scale) {
  const bool has_scale =
      execution->scale_layout != ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_NONE;
  if (has_scale != id4_pipeline_program_tensor_is_valid(scale)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "program matrix scale tensor validity must match execution semantics");
  }
  return iree_ok_status();
}

static bool id4_pipeline_program_matrix_candidate_matches_problem(
    const id4_pipeline_program_matrix_candidate_t* candidate,
    id4_pipeline_program_matrix_operation_t operation,
    const id4_pipeline_program_matrix_problem_t* problem) {
  return candidate->operation == operation &&
         candidate->input_dtype == problem->input_dtype &&
         candidate->input_layout == problem->input_layout &&
         candidate->accumulator_dtype == problem->accumulator_dtype &&
         candidate->epilogue == problem->epilogue &&
         candidate->output_dtype == problem->output_dtype &&
         candidate->output_layout == problem->output_layout &&
         problem->m_capacity >= candidate->minimum_m_capacity &&
         problem->n >= candidate->minimum_n &&
         (problem->m_capacity % candidate->m_multiple) == 0 &&
         (problem->n % candidate->n_multiple) == 0 &&
         (problem->k % candidate->k_multiple) == 0;
}

static bool id4_pipeline_program_matrix_candidate_matches_parameter(
    const id4_pipeline_program_matrix_candidate_t* candidate,
    id4_pipeline_program_matrix_operation_t operation,
    const id4_pipeline_program_matrix_problem_t* problem,
    const id4_pipeline_program_matrix_parameter_t* parameter) {
  return id4_pipeline_program_matrix_candidate_matches_problem(
             candidate, operation, problem) &&
         candidate->source_weight_dtype == parameter->weight.dtype &&
         candidate->source_weight_layout == parameter->weight_layout &&
         candidate->source_scale_dtype == parameter->scale.dtype &&
         candidate->source_scale_layout == parameter->scale_layout;
}

static bool id4_pipeline_program_matrix_candidate_matches_execution(
    const id4_pipeline_program_matrix_candidate_t* candidate,
    id4_pipeline_program_matrix_operation_t operation,
    const id4_pipeline_program_matrix_problem_t* problem,
    const id4_pipeline_program_matrix_execution_t* execution) {
  return id4_pipeline_program_matrix_candidate_matches_problem(
             candidate, operation, problem) &&
         candidate->weight_dtype == execution->weight_dtype &&
         candidate->weight_layout == execution->weight_layout &&
         candidate->scale_dtype == execution->scale_dtype &&
         candidate->scale_layout == execution->scale_layout;
}

static const id4_pipeline_program_matrix_candidate_t*
id4_pipeline_program_matrix_find_parameter_candidate(
    id4_pipeline_program_matrix_operation_t operation,
    const id4_pipeline_program_matrix_problem_t* problem,
    const id4_pipeline_program_matrix_parameter_t* parameter) {
  const id4_pipeline_program_matrix_candidate_t* best_candidate = NULL;
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(id4_pipeline_program_matrix_candidates); ++i) {
    const id4_pipeline_program_matrix_candidate_t* candidate =
        &id4_pipeline_program_matrix_candidates[i];
    if (id4_pipeline_program_matrix_candidate_matches_parameter(
            candidate, operation, problem, parameter)) {
      if (!best_candidate ||
          candidate->selection_priority > best_candidate->selection_priority) {
        best_candidate = candidate;
      }
    }
  }
  return best_candidate;
}

static const id4_pipeline_program_matrix_candidate_t*
id4_pipeline_program_matrix_find_parameter_pair_candidate(
    id4_pipeline_program_matrix_operation_t operation,
    const id4_pipeline_program_matrix_problem_t* problem,
    const id4_pipeline_program_matrix_parameter_t* lhs_parameter,
    const id4_pipeline_program_matrix_parameter_t* rhs_parameter) {
  const id4_pipeline_program_matrix_candidate_t* best_candidate = NULL;
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(id4_pipeline_program_matrix_candidates); ++i) {
    const id4_pipeline_program_matrix_candidate_t* candidate =
        &id4_pipeline_program_matrix_candidates[i];
    if (id4_pipeline_program_matrix_candidate_matches_parameter(
            candidate, operation, problem, lhs_parameter) &&
        id4_pipeline_program_matrix_candidate_matches_parameter(
            candidate, operation, problem, rhs_parameter) &&
        (!best_candidate ||
         candidate->selection_priority > best_candidate->selection_priority)) {
      best_candidate = candidate;
    }
  }
  return best_candidate;
}

static iree_status_t id4_pipeline_program_matrix_unsupported_parameter(
    id4_pipeline_program_matrix_operation_t operation,
    const id4_pipeline_program_matrix_problem_t* problem,
    const id4_pipeline_program_matrix_parameter_t* parameter) {
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "no full-coverage matrix schedule supports provider operation=%u "
      "M=%" PRIu32 "/%" PRIu32 " N=%" PRIu32 " K=%" PRIu32
      " input=%u:%u weight=%u:%u scale=%u:%u accumulator=%u epilogue=%u "
      "output=%u:%u",
      (uint32_t)operation, problem->valid_m, problem->m_capacity, problem->n,
      problem->k, (uint32_t)problem->input_dtype,
      (uint32_t)problem->input_layout, (uint32_t)parameter->weight.dtype,
      (uint32_t)parameter->weight_layout, (uint32_t)parameter->scale.dtype,
      (uint32_t)parameter->scale_layout, (uint32_t)problem->accumulator_dtype,
      (uint32_t)problem->epilogue, (uint32_t)problem->output_dtype,
      (uint32_t)problem->output_layout);
}

static const id4_pipeline_program_matrix_candidate_t*
id4_pipeline_program_matrix_find_prepared_candidate(
    id4_pipeline_program_matrix_operation_t operation,
    const id4_pipeline_program_matrix_problem_t* problem,
    const id4_pipeline_program_matrix_execution_t* execution) {
  const id4_pipeline_program_matrix_candidate_t* best_candidate = NULL;
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(id4_pipeline_program_matrix_candidates); ++i) {
    const id4_pipeline_program_matrix_candidate_t* candidate =
        &id4_pipeline_program_matrix_candidates[i];
    if (id4_pipeline_program_matrix_candidate_matches_execution(
            candidate, operation, problem, execution) &&
        (!best_candidate ||
         candidate->selection_priority > best_candidate->selection_priority)) {
      best_candidate = candidate;
    }
  }
  return best_candidate;
}

static iree_status_t id4_pipeline_program_matrix_unsupported_prepared(
    id4_pipeline_program_matrix_operation_t operation,
    const id4_pipeline_program_matrix_problem_t* problem,
    const id4_pipeline_program_matrix_execution_t* execution) {
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "no full-coverage matrix schedule supports prepared operation=%u "
      "M=%" PRIu32 "/%" PRIu32 " N=%" PRIu32 " K=%" PRIu32
      " input=%u:%u weight=%u:%u scale=%u:%u accumulator=%u epilogue=%u "
      "output=%u:%u",
      (uint32_t)operation, problem->valid_m, problem->m_capacity, problem->n,
      problem->k, (uint32_t)problem->input_dtype,
      (uint32_t)problem->input_layout, (uint32_t)execution->weight_dtype,
      (uint32_t)execution->weight_layout, (uint32_t)execution->scale_dtype,
      (uint32_t)execution->scale_layout, (uint32_t)problem->accumulator_dtype,
      (uint32_t)problem->epilogue, (uint32_t)problem->output_dtype,
      (uint32_t)problem->output_layout);
}

static const id4_pipeline_program_matrix_composition_candidate_t*
id4_pipeline_program_matrix_find_composition_candidate(
    id4_pipeline_program_matrix_composition_operation_t operation,
    const id4_pipeline_program_matrix_problem_t* problem,
    const id4_pipeline_program_matrix_candidate_t* contraction_candidate) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(id4_pipeline_program_matrix_composition_candidates);
       ++i) {
    const id4_pipeline_program_matrix_composition_candidate_t* candidate =
        &id4_pipeline_program_matrix_composition_candidates[i];
    if (candidate->operation == operation &&
        candidate->weight_dtype == contraction_candidate->weight_dtype &&
        candidate->weight_layout == contraction_candidate->weight_layout &&
        candidate->scale_layout == contraction_candidate->scale_layout &&
        candidate->input_dtype == problem->input_dtype &&
        candidate->input_layout == problem->input_layout &&
        candidate->output_dtype == problem->output_dtype &&
        candidate->output_layout == problem->output_layout) {
      return candidate;
    }
  }
  return NULL;
}

static iree_status_t id4_pipeline_program_matrix_append_config(
    iree_string_view_t key, uint32_t value,
    char value_buffer[ID4_PIPELINE_PROGRAM_MATRIX_CONFIG_VALUE_CAPACITY],
    iree_host_size_t* count,
    id4_pipeline_kernel_config_binding_t
        bindings[ID4_PIPELINE_PROGRAM_MATRIX_CONFIG_CAPACITY]) {
  if (iree_string_view_is_empty(key)) return iree_ok_status();
  int value_length =
      snprintf(value_buffer, ID4_PIPELINE_PROGRAM_MATRIX_CONFIG_VALUE_CAPACITY,
               "%" PRIu32, value);
  if (value_length < 0 ||
      value_length >= ID4_PIPELINE_PROGRAM_MATRIX_CONFIG_VALUE_CAPACITY) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "program matrix config value is too large");
  }
  bindings[*count] = id4_pipeline_make_kernel_config_binding(
      key, iree_make_string_view(value_buffer, (iree_host_size_t)value_length));
  ++*count;
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_matrix_dispatch_candidate(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    const id4_pipeline_program_matrix_problem_t* problem,
    const id4_pipeline_program_matrix_candidate_t* candidate,
    iree_host_size_t binding_count,
    const id4_pipeline_program_dispatch_binding_t* bindings) {
  char value_buffers[ID4_PIPELINE_PROGRAM_MATRIX_CONFIG_CAPACITY]
                    [ID4_PIPELINE_PROGRAM_MATRIX_CONFIG_VALUE_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_PIPELINE_PROGRAM_MATRIX_CONFIG_CAPACITY];
  iree_host_size_t config_binding_count = 0;
  IREE_RETURN_IF_ERROR(id4_pipeline_program_matrix_append_config(
      candidate->config.valid_m_key, problem->valid_m,
      value_buffers[config_binding_count], &config_binding_count,
      config_bindings));
  IREE_RETURN_IF_ERROR(id4_pipeline_program_matrix_append_config(
      candidate->config.m_capacity_key, problem->m_capacity,
      value_buffers[config_binding_count], &config_binding_count,
      config_bindings));
  IREE_RETURN_IF_ERROR(id4_pipeline_program_matrix_append_config(
      candidate->config.dispatch_m_key, problem->m_capacity,
      value_buffers[config_binding_count], &config_binding_count,
      config_bindings));
  IREE_RETURN_IF_ERROR(id4_pipeline_program_matrix_append_config(
      candidate->config.k_key, problem->k, value_buffers[config_binding_count],
      &config_binding_count, config_bindings));
  IREE_RETURN_IF_ERROR(id4_pipeline_program_matrix_append_config(
      candidate->config.n_key, problem->n, value_buffers[config_binding_count],
      &config_binding_count, config_bindings));

  id4_pipeline_program_dispatch_loom_options_t dispatch_options = {
      .structure_size = sizeof(dispatch_options),
      .name = name,
      .kernel = candidate->kernel,
      .config_binding_count = config_binding_count,
      .config_bindings = config_bindings,
      .binding_count = binding_count,
      .bindings = bindings,
  };
  return id4_pipeline_program_dispatch_loom(builder, &dispatch_options);
}

static iree_status_t id4_pipeline_program_matrix_dispatch_composition_candidate(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    const id4_pipeline_program_matrix_problem_t* problem,
    const id4_pipeline_program_matrix_composition_candidate_t* candidate,
    id4_pipeline_program_tensor_t lhs, id4_pipeline_program_tensor_t rhs,
    id4_pipeline_program_tensor_t output) {
  char value_buffers[ID4_PIPELINE_PROGRAM_MATRIX_CONFIG_CAPACITY]
                    [ID4_PIPELINE_PROGRAM_MATRIX_CONFIG_VALUE_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_PIPELINE_PROGRAM_MATRIX_CONFIG_CAPACITY];
  iree_host_size_t config_binding_count = 0;
  IREE_RETURN_IF_ERROR(id4_pipeline_program_matrix_append_config(
      candidate->m_key, problem->m_capacity,
      value_buffers[config_binding_count], &config_binding_count,
      config_bindings));
  IREE_RETURN_IF_ERROR(id4_pipeline_program_matrix_append_config(
      candidate->n_key, problem->n, value_buffers[config_binding_count],
      &config_binding_count, config_bindings));
  const id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(lhs),
      id4_pipeline_program_read(rhs),
      id4_pipeline_program_write(output),
  };
  const id4_pipeline_program_dispatch_loom_options_t dispatch_options = {
      .structure_size = sizeof(dispatch_options),
      .name = name,
      .kernel = candidate->kernel,
      .config_binding_count = config_binding_count,
      .config_bindings = config_bindings,
      .binding_count = IREE_ARRAYSIZE(bindings),
      .bindings = bindings,
  };
  return id4_pipeline_program_dispatch_loom(builder, &dispatch_options);
}

static iree_status_t id4_pipeline_program_matrix_format_child_name(
    iree_string_view_t parent_name, iree_string_view_t child_name, char* buffer,
    iree_host_size_t buffer_capacity, iree_string_view_t* out_name) {
  int name_length =
      snprintf(buffer, buffer_capacity, "%.*s.%.*s", (int)parent_name.size,
               parent_name.data, (int)child_name.size, child_name.data);
  if (name_length < 0 || (iree_host_size_t)name_length >= buffer_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "failed to format program matrix child name");
  }
  *out_name = iree_make_string_view(buffer, (iree_host_size_t)name_length);
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_matrix_acquire_intermediate(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    const id4_pipeline_program_matrix_problem_t* problem,
    id4_pipeline_program_tensor_t* out_tensor) {
  const id4_pipeline_program_acquire_tensor_options_t acquire_options = {
      .structure_size = sizeof(acquire_options),
      .name = name,
      .dtype = problem->output_dtype,
      .shape = id4_pipeline_program_make_shape_rank2(problem->m_capacity,
                                                     problem->n),
  };
  return id4_pipeline_program_acquire_tensor(builder, &acquire_options,
                                             out_tensor);
}

static iree_status_t id4_pipeline_program_matrix_barrier(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name) {
  const id4_pipeline_program_barrier_options_t barrier_options = {
      .structure_size = sizeof(barrier_options),
      .name = name,
  };
  return id4_pipeline_program_barrier(builder, &barrier_options);
}

static bool id4_pipeline_program_matrix_encoding_uses_scale_source(
    id4_pipeline_program_parameter_encoding_t encoding) {
  return encoding ==
             ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_FP8_E4M3_SCALED_TO_BF16 ||
         encoding ==
             ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_FP8_E4M3_SCALED_TO_BF16_LINEAR_RHS_TILE ||
         encoding ==
             ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_FP8_E4M3_BLOCK_SCALED_TO_BF16_LINEAR_RHS_TILE;
}

static iree_status_t id4_pipeline_program_matrix_materialize_parameter(
    id4_pipeline_program_builder_t* builder,
    const id4_pipeline_program_matrix_problem_t* problem,
    const id4_pipeline_program_matrix_parameter_t* parameter,
    const id4_pipeline_program_matrix_candidate_t* candidate,
    id4_pipeline_program_tensor_t* out_weight,
    id4_pipeline_program_tensor_t* out_scale) {
  *out_weight = id4_pipeline_program_tensor_invalid();
  *out_scale = id4_pipeline_program_tensor_invalid();

  id4_pipeline_program_parameter_source_t weight_sources[2] = {
      parameter->weight,
      parameter->scale,
  };
  const bool encoding_uses_scale =
      id4_pipeline_program_matrix_encoding_uses_scale_source(
          candidate->weight_encoding);
  if (encoding_uses_scale &&
      parameter->scale_layout ==
          ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_NONE) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "program matrix candidate weight encoding requires a scale source");
  }
  const id4_pipeline_program_parameter_options_t weight_options = {
      .structure_size = sizeof(weight_options),
      .encoding = candidate->weight_encoding,
      .source_count = encoding_uses_scale ? 2 : 1,
      .sources = weight_sources,
      .key = parameter->weight.key,
      .dtype = candidate->weight_dtype,
      .shape = id4_pipeline_program_make_shape_rank2(problem->n, problem->k),
  };
  IREE_RETURN_IF_ERROR(
      id4_pipeline_program_parameter(builder, &weight_options, out_weight));

  if (candidate->scale_layout !=
      ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_NONE) {
    if (candidate->scale_dtype != parameter->scale.dtype ||
        candidate->scale_layout != parameter->scale_layout) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "program matrix candidate requires an unsupported scale transform");
    }
    const id4_pipeline_program_parameter_options_t scale_options = {
        .structure_size = sizeof(scale_options),
        .encoding = ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_DIRECT,
        .source_count = 1,
        .sources = &parameter->scale,
        .key = parameter->scale.key,
        .dtype = parameter->scale.dtype,
        .shape = parameter->scale.shape,
    };
    IREE_RETURN_IF_ERROR(
        id4_pipeline_program_parameter(builder, &scale_options, out_scale));
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_matrix_dispatch_prepared(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    const id4_pipeline_program_matrix_problem_t* problem,
    const id4_pipeline_program_matrix_candidate_t* candidate,
    const id4_pipeline_program_matrix_prepared_operands_t* operands) {
  id4_pipeline_program_dispatch_binding_t bindings[5];
  iree_host_size_t binding_count = 0;
  bindings[binding_count++] = id4_pipeline_program_read(operands->input);
  bindings[binding_count++] = id4_pipeline_program_read(operands->weight);
  if (candidate->scale_layout !=
      ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_NONE) {
    bindings[binding_count++] = id4_pipeline_program_read(operands->scale);
  }
  if (problem->epilogue == ID4_PIPELINE_PROGRAM_MATRIX_EPILOGUE_ADD) {
    bindings[binding_count++] = id4_pipeline_program_read(operands->addend);
  }
  bindings[binding_count++] = id4_pipeline_program_write(operands->output);
  return id4_pipeline_program_matrix_dispatch_candidate(
      builder, name, problem, candidate, binding_count, bindings);
}

static iree_status_t id4_pipeline_program_swiglu_dispatch_prepared(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    const id4_pipeline_program_matrix_problem_t* problem,
    const id4_pipeline_program_matrix_candidate_t* candidate,
    const id4_pipeline_program_swiglu_prepared_operands_t* operands) {
  id4_pipeline_program_dispatch_binding_t bindings[6];
  iree_host_size_t binding_count = 0;
  bindings[binding_count++] = id4_pipeline_program_read(operands->input);
  bindings[binding_count++] = id4_pipeline_program_read(operands->gate_weight);
  if (candidate->scale_layout !=
      ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_NONE) {
    bindings[binding_count++] = id4_pipeline_program_read(operands->gate_scale);
  }
  bindings[binding_count++] = id4_pipeline_program_read(operands->up_weight);
  if (candidate->scale_layout !=
      ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_NONE) {
    bindings[binding_count++] = id4_pipeline_program_read(operands->up_scale);
  }
  bindings[binding_count++] = id4_pipeline_program_write(operands->output);
  return id4_pipeline_program_matrix_dispatch_candidate(
      builder, name, problem, candidate, binding_count, bindings);
}

static bool id4_pipeline_program_matrix_should_compose(
    const id4_pipeline_program_matrix_candidate_t* fused_candidate,
    const id4_pipeline_program_matrix_candidate_t* contraction_candidate,
    const id4_pipeline_program_matrix_composition_candidate_t*
        composition_candidate) {
  return contraction_candidate && composition_candidate &&
         (!fused_candidate || contraction_candidate->selection_priority >
                                  fused_candidate->selection_priority);
}

static id4_pipeline_program_matrix_schedule_t
id4_pipeline_program_matrix_resolve_parameter_schedule(
    const id4_pipeline_program_matrix_problem_t* problem,
    const id4_pipeline_program_matrix_parameter_t* parameter) {
  id4_pipeline_program_matrix_schedule_t schedule = {
      .primary_candidate = id4_pipeline_program_matrix_find_parameter_candidate(
          ID4_PIPELINE_PROGRAM_MATRIX_OPERATION_CONTRACTION, problem,
          parameter),
  };
  if (problem->epilogue != ID4_PIPELINE_PROGRAM_MATRIX_EPILOGUE_ADD) {
    return schedule;
  }

  id4_pipeline_program_matrix_problem_t contraction_problem = *problem;
  contraction_problem.epilogue = ID4_PIPELINE_PROGRAM_MATRIX_EPILOGUE_NONE;
  const id4_pipeline_program_matrix_candidate_t* contraction_candidate =
      id4_pipeline_program_matrix_find_parameter_candidate(
          ID4_PIPELINE_PROGRAM_MATRIX_OPERATION_CONTRACTION,
          &contraction_problem, parameter);
  const id4_pipeline_program_matrix_composition_candidate_t*
      composition_candidate = NULL;
  if (contraction_candidate) {
    composition_candidate =
        id4_pipeline_program_matrix_find_composition_candidate(
            ID4_PIPELINE_PROGRAM_MATRIX_COMPOSITION_OPERATION_ADD, problem,
            contraction_candidate);
  }
  if (id4_pipeline_program_matrix_should_compose(schedule.primary_candidate,
                                                 contraction_candidate,
                                                 composition_candidate)) {
    schedule.primary_candidate = contraction_candidate;
    schedule.composition_candidate = composition_candidate;
  }
  return schedule;
}

static id4_pipeline_program_matrix_schedule_t
id4_pipeline_program_matrix_resolve_prepared_schedule(
    const id4_pipeline_program_matrix_problem_t* problem,
    const id4_pipeline_program_matrix_execution_t* execution) {
  id4_pipeline_program_matrix_schedule_t schedule = {
      .primary_candidate = id4_pipeline_program_matrix_find_prepared_candidate(
          ID4_PIPELINE_PROGRAM_MATRIX_OPERATION_CONTRACTION, problem,
          execution),
  };
  if (problem->epilogue != ID4_PIPELINE_PROGRAM_MATRIX_EPILOGUE_ADD) {
    return schedule;
  }

  id4_pipeline_program_matrix_problem_t contraction_problem = *problem;
  contraction_problem.epilogue = ID4_PIPELINE_PROGRAM_MATRIX_EPILOGUE_NONE;
  const id4_pipeline_program_matrix_candidate_t* contraction_candidate =
      id4_pipeline_program_matrix_find_prepared_candidate(
          ID4_PIPELINE_PROGRAM_MATRIX_OPERATION_CONTRACTION,
          &contraction_problem, execution);
  const id4_pipeline_program_matrix_composition_candidate_t*
      composition_candidate = NULL;
  if (contraction_candidate) {
    composition_candidate =
        id4_pipeline_program_matrix_find_composition_candidate(
            ID4_PIPELINE_PROGRAM_MATRIX_COMPOSITION_OPERATION_ADD, problem,
            contraction_candidate);
  }
  if (id4_pipeline_program_matrix_should_compose(schedule.primary_candidate,
                                                 contraction_candidate,
                                                 composition_candidate)) {
    schedule.primary_candidate = contraction_candidate;
    schedule.composition_candidate = composition_candidate;
  }
  return schedule;
}

static id4_pipeline_program_matrix_schedule_t
id4_pipeline_program_swiglu_resolve_parameter_schedule(
    const id4_pipeline_program_matrix_problem_t* problem,
    const id4_pipeline_program_matrix_parameter_t* gate_parameter,
    const id4_pipeline_program_matrix_parameter_t* up_parameter) {
  const id4_pipeline_program_matrix_candidate_t* fused_candidate =
      id4_pipeline_program_matrix_find_parameter_pair_candidate(
          ID4_PIPELINE_PROGRAM_MATRIX_OPERATION_SWIGLU, problem, gate_parameter,
          up_parameter);
  const id4_pipeline_program_matrix_candidate_t* contraction_candidate =
      id4_pipeline_program_matrix_find_parameter_pair_candidate(
          ID4_PIPELINE_PROGRAM_MATRIX_OPERATION_CONTRACTION, problem,
          gate_parameter, up_parameter);
  const id4_pipeline_program_matrix_composition_candidate_t*
      composition_candidate = NULL;
  if (contraction_candidate) {
    composition_candidate =
        id4_pipeline_program_matrix_find_composition_candidate(
            ID4_PIPELINE_PROGRAM_MATRIX_COMPOSITION_OPERATION_SWIGLU, problem,
            contraction_candidate);
  }
  if (id4_pipeline_program_matrix_should_compose(
          fused_candidate, contraction_candidate, composition_candidate)) {
    return (id4_pipeline_program_matrix_schedule_t){
        .primary_candidate = contraction_candidate,
        .composition_candidate = composition_candidate,
    };
  }
  return (id4_pipeline_program_matrix_schedule_t){
      .primary_candidate = fused_candidate,
  };
}

static iree_status_t id4_pipeline_program_matrix_dispatch_composed_add(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    const id4_pipeline_program_matrix_problem_t* problem,
    const id4_pipeline_program_matrix_candidate_t* contraction_candidate,
    const id4_pipeline_program_matrix_composition_candidate_t*
        composition_candidate,
    const id4_pipeline_program_matrix_prepared_operands_t* operands) {
  id4_pipeline_program_matrix_problem_t contraction_problem = *problem;
  contraction_problem.epilogue = ID4_PIPELINE_PROGRAM_MATRIX_EPILOGUE_NONE;

  char child_name_buffer[IREE_MAX_PATH];
  iree_string_view_t child_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_pipeline_program_matrix_format_child_name(
      name, IREE_SV("contraction.output"), child_name_buffer,
      IREE_ARRAYSIZE(child_name_buffer), &child_name));
  id4_pipeline_program_tensor_t contraction_output =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_pipeline_program_matrix_acquire_intermediate(
      builder, child_name, &contraction_problem, &contraction_output));

  IREE_RETURN_IF_ERROR(id4_pipeline_program_matrix_format_child_name(
      name, IREE_SV("contraction"), child_name_buffer,
      IREE_ARRAYSIZE(child_name_buffer), &child_name));
  const id4_pipeline_program_matrix_prepared_operands_t contraction_operands = {
      .input = operands->input,
      .weight = operands->weight,
      .scale = operands->scale,
      .addend = id4_pipeline_program_tensor_invalid(),
      .output = contraction_output,
  };
  IREE_RETURN_IF_ERROR(id4_pipeline_program_matrix_dispatch_prepared(
      builder, child_name, &contraction_problem, contraction_candidate,
      &contraction_operands));

  IREE_RETURN_IF_ERROR(id4_pipeline_program_matrix_format_child_name(
      name, IREE_SV("contraction_ready"), child_name_buffer,
      IREE_ARRAYSIZE(child_name_buffer), &child_name));
  IREE_RETURN_IF_ERROR(
      id4_pipeline_program_matrix_barrier(builder, child_name));

  IREE_RETURN_IF_ERROR(id4_pipeline_program_matrix_format_child_name(
      name, IREE_SV("epilogue"), child_name_buffer,
      IREE_ARRAYSIZE(child_name_buffer), &child_name));
  return id4_pipeline_program_matrix_dispatch_composition_candidate(
      builder, child_name, problem, composition_candidate, contraction_output,
      operands->addend, operands->output);
}

static iree_status_t id4_pipeline_program_swiglu_dispatch_composed(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    const id4_pipeline_program_matrix_problem_t* problem,
    const id4_pipeline_program_matrix_candidate_t* contraction_candidate,
    const id4_pipeline_program_matrix_composition_candidate_t*
        composition_candidate,
    const id4_pipeline_program_swiglu_prepared_operands_t* operands) {
  char child_name_buffer[IREE_MAX_PATH];
  iree_string_view_t child_name = iree_string_view_empty();

  IREE_RETURN_IF_ERROR(id4_pipeline_program_matrix_format_child_name(
      name, IREE_SV("gate.output"), child_name_buffer,
      IREE_ARRAYSIZE(child_name_buffer), &child_name));
  id4_pipeline_program_tensor_t gate_output =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_pipeline_program_matrix_acquire_intermediate(
      builder, child_name, problem, &gate_output));
  IREE_RETURN_IF_ERROR(id4_pipeline_program_matrix_format_child_name(
      name, IREE_SV("up.output"), child_name_buffer,
      IREE_ARRAYSIZE(child_name_buffer), &child_name));
  id4_pipeline_program_tensor_t up_output =
      id4_pipeline_program_tensor_invalid();
  IREE_RETURN_IF_ERROR(id4_pipeline_program_matrix_acquire_intermediate(
      builder, child_name, problem, &up_output));

  IREE_RETURN_IF_ERROR(id4_pipeline_program_matrix_format_child_name(
      name, IREE_SV("gate"), child_name_buffer,
      IREE_ARRAYSIZE(child_name_buffer), &child_name));
  const id4_pipeline_program_matrix_prepared_operands_t gate_operands = {
      .input = operands->input,
      .weight = operands->gate_weight,
      .scale = operands->gate_scale,
      .addend = id4_pipeline_program_tensor_invalid(),
      .output = gate_output,
  };
  IREE_RETURN_IF_ERROR(id4_pipeline_program_matrix_dispatch_prepared(
      builder, child_name, problem, contraction_candidate, &gate_operands));

  IREE_RETURN_IF_ERROR(id4_pipeline_program_matrix_format_child_name(
      name, IREE_SV("up"), child_name_buffer, IREE_ARRAYSIZE(child_name_buffer),
      &child_name));
  const id4_pipeline_program_matrix_prepared_operands_t up_operands = {
      .input = operands->input,
      .weight = operands->up_weight,
      .scale = operands->up_scale,
      .addend = id4_pipeline_program_tensor_invalid(),
      .output = up_output,
  };
  IREE_RETURN_IF_ERROR(id4_pipeline_program_matrix_dispatch_prepared(
      builder, child_name, problem, contraction_candidate, &up_operands));

  IREE_RETURN_IF_ERROR(id4_pipeline_program_matrix_format_child_name(
      name, IREE_SV("projections_ready"), child_name_buffer,
      IREE_ARRAYSIZE(child_name_buffer), &child_name));
  IREE_RETURN_IF_ERROR(
      id4_pipeline_program_matrix_barrier(builder, child_name));

  IREE_RETURN_IF_ERROR(id4_pipeline_program_matrix_format_child_name(
      name, IREE_SV("product"), child_name_buffer,
      IREE_ARRAYSIZE(child_name_buffer), &child_name));
  return id4_pipeline_program_matrix_dispatch_composition_candidate(
      builder, child_name, problem, composition_candidate, gate_output,
      up_output, operands->output);
}

iree_status_t id4_pipeline_program_matrix(
    id4_pipeline_program_builder_t* builder,
    const id4_pipeline_program_matrix_options_t* options) {
  if (!builder) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program matrix builder is required");
  }
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program matrix options are required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_program_matrix_validate_options_size(
      options->structure_size, sizeof(*options)));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "program matrix extension structures are not supported");
  }
  if (iree_string_view_is_empty(options->name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program matrix operation name is required");
  }
  IREE_RETURN_IF_ERROR(
      id4_pipeline_program_matrix_validate_problem(&options->problem));
  IREE_RETURN_IF_ERROR(id4_pipeline_program_matrix_validate_parameter(
      &options->problem, &options->operands.parameter));
  if (!id4_pipeline_program_tensor_is_valid(options->operands.input) ||
      !id4_pipeline_program_tensor_is_valid(options->operands.output)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "program matrix input and output tensors are required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_program_matrix_validate_addend(
      &options->problem, options->operands.addend));

  const id4_pipeline_program_matrix_schedule_t schedule =
      id4_pipeline_program_matrix_resolve_parameter_schedule(
          &options->problem, &options->operands.parameter);
  if (!schedule.primary_candidate) {
    return id4_pipeline_program_matrix_unsupported_parameter(
        ID4_PIPELINE_PROGRAM_MATRIX_OPERATION_CONTRACTION, &options->problem,
        &options->operands.parameter);
  }
  id4_pipeline_program_matrix_prepared_operands_t prepared_operands = {
      .input = options->operands.input,
      .addend = options->operands.addend,
      .output = options->operands.output,
  };
  IREE_RETURN_IF_ERROR(id4_pipeline_program_matrix_materialize_parameter(
      builder, &options->problem, &options->operands.parameter,
      schedule.primary_candidate, &prepared_operands.weight,
      &prepared_operands.scale));
  if (schedule.composition_candidate) {
    return id4_pipeline_program_matrix_dispatch_composed_add(
        builder, options->name, &options->problem, schedule.primary_candidate,
        schedule.composition_candidate, &prepared_operands);
  }
  return id4_pipeline_program_matrix_dispatch_prepared(
      builder, options->name, &options->problem, schedule.primary_candidate,
      &prepared_operands);
}

iree_status_t id4_pipeline_program_matrix_prepared(
    id4_pipeline_program_builder_t* builder,
    const id4_pipeline_program_matrix_prepared_options_t* options) {
  if (!builder) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "prepared program matrix builder is required");
  }
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "prepared program matrix options are required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_program_matrix_validate_options_size(
      options->structure_size, sizeof(*options)));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "prepared program matrix extension structures are not supported");
  }
  if (iree_string_view_is_empty(options->name)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "prepared program matrix operation name is required");
  }
  IREE_RETURN_IF_ERROR(
      id4_pipeline_program_matrix_validate_problem(&options->problem));
  IREE_RETURN_IF_ERROR(
      id4_pipeline_program_matrix_validate_execution(&options->execution));
  IREE_RETURN_IF_ERROR(id4_pipeline_program_matrix_validate_scale_operand(
      &options->execution, options->operands.scale));
  if (!id4_pipeline_program_tensor_is_valid(options->operands.input) ||
      !id4_pipeline_program_tensor_is_valid(options->operands.weight) ||
      !id4_pipeline_program_tensor_is_valid(options->operands.output)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "prepared program matrix input, weight, and output are required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_program_matrix_validate_addend(
      &options->problem, options->operands.addend));

  const id4_pipeline_program_matrix_schedule_t schedule =
      id4_pipeline_program_matrix_resolve_prepared_schedule(
          &options->problem, &options->execution);
  if (!schedule.primary_candidate) {
    return id4_pipeline_program_matrix_unsupported_prepared(
        ID4_PIPELINE_PROGRAM_MATRIX_OPERATION_CONTRACTION, &options->problem,
        &options->execution);
  }
  if (schedule.composition_candidate) {
    return id4_pipeline_program_matrix_dispatch_composed_add(
        builder, options->name, &options->problem, schedule.primary_candidate,
        schedule.composition_candidate, &options->operands);
  }
  return id4_pipeline_program_matrix_dispatch_prepared(
      builder, options->name, &options->problem, schedule.primary_candidate,
      &options->operands);
}

iree_status_t id4_pipeline_program_swiglu(
    id4_pipeline_program_builder_t* builder,
    const id4_pipeline_program_swiglu_options_t* options) {
  if (!builder) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program SwiGLU builder is required");
  }
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program SwiGLU options are required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_program_matrix_validate_options_size(
      options->structure_size, sizeof(*options)));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "program SwiGLU extension structures are not supported");
  }
  if (iree_string_view_is_empty(options->name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program SwiGLU operation name is required");
  }
  IREE_RETURN_IF_ERROR(
      id4_pipeline_program_matrix_validate_problem(&options->projection));
  if (options->projection.epilogue !=
      ID4_PIPELINE_PROGRAM_MATRIX_EPILOGUE_NONE) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "program SwiGLU projections cannot carry another epilogue");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_program_matrix_validate_parameter(
      &options->projection, &options->operands.gate_parameter));
  IREE_RETURN_IF_ERROR(id4_pipeline_program_matrix_validate_parameter(
      &options->projection, &options->operands.up_parameter));
  if (!id4_pipeline_program_tensor_is_valid(options->operands.input) ||
      !id4_pipeline_program_tensor_is_valid(options->operands.output)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program SwiGLU input and output are required");
  }

  const id4_pipeline_program_matrix_schedule_t schedule =
      id4_pipeline_program_swiglu_resolve_parameter_schedule(
          &options->projection, &options->operands.gate_parameter,
          &options->operands.up_parameter);
  if (!schedule.primary_candidate) {
    if (id4_pipeline_program_matrix_find_parameter_candidate(
            ID4_PIPELINE_PROGRAM_MATRIX_OPERATION_SWIGLU, &options->projection,
            &options->operands.gate_parameter) ||
        id4_pipeline_program_matrix_find_parameter_candidate(
            ID4_PIPELINE_PROGRAM_MATRIX_OPERATION_SWIGLU, &options->projection,
            &options->operands.up_parameter)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "program SwiGLU gate and up parameters require one common schedule");
    }
    return id4_pipeline_program_matrix_unsupported_parameter(
        ID4_PIPELINE_PROGRAM_MATRIX_OPERATION_SWIGLU, &options->projection,
        &options->operands.gate_parameter);
  }
  id4_pipeline_program_swiglu_prepared_operands_t prepared_operands = {
      .input = options->operands.input,
      .output = options->operands.output,
  };
  IREE_RETURN_IF_ERROR(id4_pipeline_program_matrix_materialize_parameter(
      builder, &options->projection, &options->operands.gate_parameter,
      schedule.primary_candidate, &prepared_operands.gate_weight,
      &prepared_operands.gate_scale));
  IREE_RETURN_IF_ERROR(id4_pipeline_program_matrix_materialize_parameter(
      builder, &options->projection, &options->operands.up_parameter,
      schedule.primary_candidate, &prepared_operands.up_weight,
      &prepared_operands.up_scale));
  if (schedule.composition_candidate) {
    return id4_pipeline_program_swiglu_dispatch_composed(
        builder, options->name, &options->projection,
        schedule.primary_candidate, schedule.composition_candidate,
        &prepared_operands);
  }
  return id4_pipeline_program_swiglu_dispatch_prepared(
      builder, options->name, &options->projection, schedule.primary_candidate,
      &prepared_operands);
}
