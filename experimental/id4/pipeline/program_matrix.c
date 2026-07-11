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
  // Input scalar type accepted by the candidate.
  id4_pipeline_program_dtype_t input_dtype;
  // Input physical layout accepted by the candidate.
  id4_pipeline_program_matrix_layout_t input_layout;
  // Weight scalar storage type accepted by the candidate.
  id4_pipeline_program_dtype_t weight_dtype;
  // Weight physical layout accepted by the candidate.
  id4_pipeline_program_matrix_layout_t weight_layout;
  // Scale scalar type accepted by the candidate.
  id4_pipeline_program_dtype_t scale_dtype;
  // Scale semantics accepted by the candidate.
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
  // Loom kernel implementing the candidate.
  id4_pipeline_kernel_ref_t kernel;
  // Config-key mapping used by the candidate module.
  id4_pipeline_program_matrix_candidate_config_t config;
} id4_pipeline_program_matrix_candidate_t;

static const id4_pipeline_program_matrix_candidate_t
    id4_pipeline_program_matrix_candidates[] = {
        // Qwen block-scaled plain contractions, fastest schedule first.
        {
            .operation = ID4_PIPELINE_PROGRAM_MATRIX_OPERATION_CONTRACTION,
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
                    .n_key =
                        IREE_SVL("id4.qwen3_vl.linear_fp8_block_scaled_wmma."
                                 "output_size"),
                },
        },
        {
            .operation = ID4_PIPELINE_PROGRAM_MATRIX_OPERATION_CONTRACTION,
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
                    .n_key =
                        IREE_SVL("id4.qwen3_vl.linear_fp8_block_scaled_wmma."
                                 "output_size"),
                },
        },
        // Qwen block-scaled contractions with rounded residual addition.
        {
            .operation = ID4_PIPELINE_PROGRAM_MATRIX_OPERATION_CONTRACTION,
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
            .kernel =
                {
                    IREE_SVL("qwen3_vl/linear_residual_fp8_block_scaled_bf16_"
                             "wmma_m128n64_2wave_compact_rhs"),
                    IREE_SVL("id4_qwen3_vl_linear_residual_fp8_block_scaled_"
                             "bf16_wmma_m128n64_2wave_compact_rhs"),
                },
            .config =
                {
                    .valid_m_key = IREE_SVL(
                        "id4.qwen3_vl.linear_residual_wmma.token_count"),
                    .dispatch_m_key = IREE_SVL("id4.qwen3_vl.linear_residual_"
                                               "wmma.dispatch_token_count"),
                    .k_key = IREE_SVL(
                        "id4.qwen3_vl.linear_residual_wmma.input_size"),
                    .n_key = IREE_SVL(
                        "id4.qwen3_vl.linear_residual_wmma.output_size"),
                },
        },
        {
            .operation = ID4_PIPELINE_PROGRAM_MATRIX_OPERATION_CONTRACTION,
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
            .kernel =
                {
                    IREE_SVL("qwen3_vl/linear_residual_fp8_block_scaled_bf16_"
                             "wmma_compact_rhs"),
                    IREE_SVL("id4_qwen3_vl_linear_residual_fp8_block_scaled_"
                             "bf16_wmma_m64n64_compact_rhs"),
                },
            .config =
                {
                    .valid_m_key = IREE_SVL(
                        "id4.qwen3_vl.linear_residual_wmma.token_count"),
                    .dispatch_m_key = IREE_SVL("id4.qwen3_vl.linear_residual_"
                                               "wmma.dispatch_token_count"),
                    .k_key = IREE_SVL(
                        "id4.qwen3_vl.linear_residual_wmma.input_size"),
                    .n_key = IREE_SVL(
                        "id4.qwen3_vl.linear_residual_wmma.output_size"),
                },
        },
        // Qwen block-scaled paired SwiGLU projections.
        {
            .operation = ID4_PIPELINE_PROGRAM_MATRIX_OPERATION_SWIGLU,
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
                    .n_key =
                        IREE_SVL("id4.qwen3_vl.mlp_gate_up_silu_product_wmma."
                                 "intermediate_size"),
                },
        },
        {
            .operation = ID4_PIPELINE_PROGRAM_MATRIX_OPERATION_SWIGLU,
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
                    .n_key =
                        IREE_SVL("id4.qwen3_vl.mlp_gate_up_silu_product_wmma."
                                 "intermediate_size"),
                },
        },
        // DiT row-scaled paired SwiGLU projections by physical RHS layout.
        {
            .operation = ID4_PIPELINE_PROGRAM_MATRIX_OPERATION_SWIGLU,
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
            .kernel =
                {
                    IREE_SVL("ideogram4/"
                             "mlp_gate_up_silu_product_fp8_bf16_wmma"),
                    IREE_SVL("id4_ideogram4_mlp_gate_up_silu_product_fp8_"
                             "bf16_wmma"),
                },
            .config =
                {
                    .m_capacity_key = IREE_SVL(
                        "id4.ideogram4.mlp_gate_up_silu_product_fp8_wmma."
                        "token_capacity"),
                    .dispatch_m_key = IREE_SVL(
                        "id4.ideogram4.mlp_gate_up_silu_product_fp8_wmma."
                        "dispatch_token_count"),
                    .k_key = IREE_SVL(
                        "id4.ideogram4.mlp_gate_up_silu_product_fp8_wmma."
                        "input_size"),
                    .n_key = IREE_SVL(
                        "id4.ideogram4.mlp_gate_up_silu_product_fp8_wmma."
                        "intermediate_size"),
                },
        },
        {
            .operation = ID4_PIPELINE_PROGRAM_MATRIX_OPERATION_SWIGLU,
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
            .kernel =
                {
                    IREE_SVL("ideogram4/"
                             "mlp_gate_up_silu_product_fp8_bf16_wmma"),
                    IREE_SVL("id4_ideogram4_mlp_gate_up_silu_product_fp8_"
                             "bf16_wmma_compact_rhs_tile_m64n64_4wave"),
                },
            .config =
                {
                    .m_capacity_key = IREE_SVL(
                        "id4.ideogram4.mlp_gate_up_silu_product_fp8_wmma."
                        "token_capacity"),
                    .dispatch_m_key = IREE_SVL(
                        "id4.ideogram4.mlp_gate_up_silu_product_fp8_wmma."
                        "dispatch_token_count"),
                    .k_key = IREE_SVL(
                        "id4.ideogram4.mlp_gate_up_silu_product_fp8_wmma."
                        "input_size"),
                    .n_key = IREE_SVL(
                        "id4.ideogram4.mlp_gate_up_silu_product_fp8_wmma."
                        "intermediate_size"),
                },
        },
        // DiT row-scaled plain contractions.
        {
            .operation = ID4_PIPELINE_PROGRAM_MATRIX_OPERATION_CONTRACTION,
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
                    .k_key =
                        IREE_SVL("id4.ideogram4.linear_fp8_wmma.input_size"),
                    .n_key =
                        IREE_SVL("id4.ideogram4.linear_fp8_wmma.output_size"),
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

static iree_status_t id4_pipeline_program_matrix_validate_request(
    const id4_pipeline_program_matrix_request_t* request) {
  if (request->valid_m == 0 || request->m_capacity == 0 || request->n == 0 ||
      request->k == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program matrix M/N/K dimensions must be nonzero");
  }
  if (request->valid_m > request->m_capacity) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program matrix valid M %" PRIu32
                            " exceeds allocated M capacity %" PRIu32,
                            request->valid_m, request->m_capacity);
  }
  if (id4_pipeline_program_dtype_byte_length(request->input_dtype) == 0 ||
      id4_pipeline_program_dtype_byte_length(request->weight_dtype) == 0 ||
      id4_pipeline_program_dtype_byte_length(request->accumulator_dtype) == 0 ||
      id4_pipeline_program_dtype_byte_length(request->output_dtype) == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program matrix input, weight, accumulator, and "
                            "output dtypes must be valid");
  }
  if (!id4_pipeline_program_matrix_layout_is_valid(request->input_layout) ||
      !id4_pipeline_program_matrix_layout_is_valid(request->weight_layout) ||
      !id4_pipeline_program_matrix_layout_is_valid(request->output_layout)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program matrix physical layouts must be valid");
  }
  if (!id4_pipeline_program_matrix_scale_layout_is_valid(
          request->scale_layout)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program matrix scale layout must be valid");
  }
  if (!id4_pipeline_program_matrix_epilogue_is_valid(request->epilogue)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program matrix epilogue must be valid");
  }
  const bool has_scale =
      request->scale_layout != ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_NONE;
  if (has_scale !=
      (id4_pipeline_program_dtype_byte_length(request->scale_dtype) != 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "program matrix scale dtype validity must match scale semantics");
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_matrix_validate_scale_operand(
    const id4_pipeline_program_matrix_request_t* request,
    id4_pipeline_program_tensor_t scale) {
  const bool has_scale =
      request->scale_layout != ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_NONE;
  if (has_scale != id4_pipeline_program_tensor_is_valid(scale)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "program matrix scale tensor validity must match scale semantics");
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_matrix_validate_operands(
    const id4_pipeline_program_matrix_options_t* options) {
  IREE_RETURN_IF_ERROR(id4_pipeline_program_matrix_validate_scale_operand(
      &options->request, options->operands.scale));
  if (!id4_pipeline_program_tensor_is_valid(options->operands.input) ||
      !id4_pipeline_program_tensor_is_valid(options->operands.weight) ||
      !id4_pipeline_program_tensor_is_valid(options->operands.output)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program matrix input, weight, and output tensors "
                            "are required");
  }
  const bool has_addend =
      options->request.epilogue == ID4_PIPELINE_PROGRAM_MATRIX_EPILOGUE_ADD;
  if (has_addend !=
      id4_pipeline_program_tensor_is_valid(options->operands.addend)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "program matrix addend tensor validity must match epilogue semantics");
  }
  return iree_ok_status();
}

static bool id4_pipeline_program_matrix_candidate_matches(
    const id4_pipeline_program_matrix_candidate_t* candidate,
    id4_pipeline_program_matrix_operation_t operation,
    const id4_pipeline_program_matrix_request_t* request) {
  return candidate->operation == operation &&
         candidate->input_dtype == request->input_dtype &&
         candidate->input_layout == request->input_layout &&
         candidate->weight_dtype == request->weight_dtype &&
         candidate->weight_layout == request->weight_layout &&
         candidate->scale_dtype == request->scale_dtype &&
         candidate->scale_layout == request->scale_layout &&
         candidate->accumulator_dtype == request->accumulator_dtype &&
         candidate->epilogue == request->epilogue &&
         candidate->output_dtype == request->output_dtype &&
         candidate->output_layout == request->output_layout &&
         request->m_capacity >= candidate->minimum_m_capacity &&
         request->n >= candidate->minimum_n &&
         (request->m_capacity % candidate->m_multiple) == 0 &&
         (request->n % candidate->n_multiple) == 0 &&
         (request->k % candidate->k_multiple) == 0;
}

static iree_status_t id4_pipeline_program_matrix_select_candidate(
    id4_pipeline_program_matrix_operation_t operation,
    const id4_pipeline_program_matrix_request_t* request,
    const id4_pipeline_program_matrix_candidate_t** out_candidate) {
  *out_candidate = NULL;
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(id4_pipeline_program_matrix_candidates); ++i) {
    const id4_pipeline_program_matrix_candidate_t* candidate =
        &id4_pipeline_program_matrix_candidates[i];
    if (id4_pipeline_program_matrix_candidate_matches(candidate, operation,
                                                      request)) {
      *out_candidate = candidate;
      return iree_ok_status();
    }
  }
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "no full-coverage matrix schedule supports operation=%u M=%" PRIu32
      "/%" PRIu32 " N=%" PRIu32 " K=%" PRIu32
      " input=%u:%u weight=%u:%u scale=%u:%u accumulator=%u epilogue=%u "
      "output=%u:%u",
      (uint32_t)operation, request->valid_m, request->m_capacity, request->n,
      request->k, (uint32_t)request->input_dtype,
      (uint32_t)request->input_layout, (uint32_t)request->weight_dtype,
      (uint32_t)request->weight_layout, (uint32_t)request->scale_dtype,
      (uint32_t)request->scale_layout, (uint32_t)request->accumulator_dtype,
      (uint32_t)request->epilogue, (uint32_t)request->output_dtype,
      (uint32_t)request->output_layout);
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

static iree_status_t id4_pipeline_program_matrix_dispatch(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    id4_pipeline_program_matrix_operation_t operation,
    const id4_pipeline_program_matrix_request_t* request,
    iree_host_size_t binding_count,
    const id4_pipeline_program_dispatch_binding_t* bindings) {
  const id4_pipeline_program_matrix_candidate_t* candidate = NULL;
  IREE_RETURN_IF_ERROR(id4_pipeline_program_matrix_select_candidate(
      operation, request, &candidate));

  char value_buffers[ID4_PIPELINE_PROGRAM_MATRIX_CONFIG_CAPACITY]
                    [ID4_PIPELINE_PROGRAM_MATRIX_CONFIG_VALUE_CAPACITY];
  id4_pipeline_kernel_config_binding_t
      config_bindings[ID4_PIPELINE_PROGRAM_MATRIX_CONFIG_CAPACITY];
  iree_host_size_t config_binding_count = 0;
  IREE_RETURN_IF_ERROR(id4_pipeline_program_matrix_append_config(
      candidate->config.valid_m_key, request->valid_m,
      value_buffers[config_binding_count], &config_binding_count,
      config_bindings));
  IREE_RETURN_IF_ERROR(id4_pipeline_program_matrix_append_config(
      candidate->config.m_capacity_key, request->m_capacity,
      value_buffers[config_binding_count], &config_binding_count,
      config_bindings));
  IREE_RETURN_IF_ERROR(id4_pipeline_program_matrix_append_config(
      candidate->config.dispatch_m_key, request->m_capacity,
      value_buffers[config_binding_count], &config_binding_count,
      config_bindings));
  IREE_RETURN_IF_ERROR(id4_pipeline_program_matrix_append_config(
      candidate->config.k_key, request->k, value_buffers[config_binding_count],
      &config_binding_count, config_bindings));
  IREE_RETURN_IF_ERROR(id4_pipeline_program_matrix_append_config(
      candidate->config.n_key, request->n, value_buffers[config_binding_count],
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
      id4_pipeline_program_matrix_validate_request(&options->request));
  IREE_RETURN_IF_ERROR(id4_pipeline_program_matrix_validate_operands(options));

  id4_pipeline_program_dispatch_binding_t bindings[5];
  iree_host_size_t binding_count = 0;
  bindings[binding_count++] =
      id4_pipeline_program_read(options->operands.input);
  bindings[binding_count++] =
      id4_pipeline_program_read(options->operands.weight);
  if (options->request.scale_layout !=
      ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_NONE) {
    bindings[binding_count++] =
        id4_pipeline_program_read(options->operands.scale);
  }
  if (options->request.epilogue == ID4_PIPELINE_PROGRAM_MATRIX_EPILOGUE_ADD) {
    bindings[binding_count++] =
        id4_pipeline_program_read(options->operands.addend);
  }
  bindings[binding_count++] =
      id4_pipeline_program_write(options->operands.output);
  return id4_pipeline_program_matrix_dispatch(
      builder, options->name, ID4_PIPELINE_PROGRAM_MATRIX_OPERATION_CONTRACTION,
      &options->request, binding_count, bindings);
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
      id4_pipeline_program_matrix_validate_request(&options->projection));
  if (options->projection.epilogue !=
      ID4_PIPELINE_PROGRAM_MATRIX_EPILOGUE_NONE) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "program SwiGLU projections cannot carry another epilogue");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_program_matrix_validate_scale_operand(
      &options->projection, options->operands.gate_scale));
  IREE_RETURN_IF_ERROR(id4_pipeline_program_matrix_validate_scale_operand(
      &options->projection, options->operands.up_scale));
  if (!id4_pipeline_program_tensor_is_valid(options->operands.input) ||
      !id4_pipeline_program_tensor_is_valid(options->operands.gate_weight) ||
      !id4_pipeline_program_tensor_is_valid(options->operands.up_weight) ||
      !id4_pipeline_program_tensor_is_valid(options->operands.output)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program SwiGLU input, gate weight, up weight, and "
                            "output tensors are required");
  }

  id4_pipeline_program_dispatch_binding_t bindings[6];
  iree_host_size_t binding_count = 0;
  bindings[binding_count++] =
      id4_pipeline_program_read(options->operands.input);
  bindings[binding_count++] =
      id4_pipeline_program_read(options->operands.gate_weight);
  if (options->projection.scale_layout !=
      ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_NONE) {
    bindings[binding_count++] =
        id4_pipeline_program_read(options->operands.gate_scale);
  }
  bindings[binding_count++] =
      id4_pipeline_program_read(options->operands.up_weight);
  if (options->projection.scale_layout !=
      ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_NONE) {
    bindings[binding_count++] =
        id4_pipeline_program_read(options->operands.up_scale);
  }
  bindings[binding_count++] =
      id4_pipeline_program_write(options->operands.output);
  return id4_pipeline_program_matrix_dispatch(
      builder, options->name, ID4_PIPELINE_PROGRAM_MATRIX_OPERATION_SWIGLU,
      &options->projection, binding_count, bindings);
}
