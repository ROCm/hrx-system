// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.
// Regenerate: python3 loom/py/loom/gen/run.py c_tables --in-place
// clang-format off

#ifndef LOOM_OPS_ENCODING_OPS_H_
#define LOOM_OPS_ENCODING_OPS_H_

#include "loom/ir/encoding.h"
#include "loom/ops/op_defs.h"

// Rounding and exceptional-value policy for encoded values.
typedef enum loom_encoding_rounding_policy_e {
  LOOM_ENCODING_ROUNDING_POLICY_NONE = 0,
  LOOM_ENCODING_ROUNDING_POLICY_NEAREST_EVEN = 1,
  LOOM_ENCODING_ROUNDING_POLICY_NEAREST_AWAY = 2,
  LOOM_ENCODING_ROUNDING_POLICY_TOWARD_ZERO = 3,
  LOOM_ENCODING_ROUNDING_POLICY_DOWN = 4,
  LOOM_ENCODING_ROUNDING_POLICY_UP = 5,
  LOOM_ENCODING_ROUNDING_POLICY_STOCHASTIC = 6,
  LOOM_ENCODING_ROUNDING_POLICY_SATFINITE = 7,
  LOOM_ENCODING_ROUNDING_POLICY_OVERFLOW_TO_INF = 8,
  LOOM_ENCODING_ROUNDING_POLICY_OVERFLOW_TO_NAN = 9,
  LOOM_ENCODING_ROUNDING_POLICY_FLUSH_SUBNORMAL = 10,
  LOOM_ENCODING_ROUNDING_POLICY_PRESERVE_SUBNORMAL = 11,
  LOOM_ENCODING_ROUNDING_POLICY_RELU_CLAMP = 12,
  LOOM_ENCODING_ROUNDING_POLICY_FINITE_ONLY = 13,
  LOOM_ENCODING_ROUNDING_POLICY_FINITE_FLUSH_SUBNORMAL = 14,
  LOOM_ENCODING_ROUNDING_POLICY_COUNT_ = 15,
} loom_encoding_rounding_policy_t;

// Affine transform applied to encoded values.
typedef enum loom_encoding_affine_policy_e {
  LOOM_ENCODING_AFFINE_POLICY_NONE = 0,
  LOOM_ENCODING_AFFINE_POLICY_SCALE_ONLY = 1,
  LOOM_ENCODING_AFFINE_POLICY_SCALE_PLUS_MIN = 2,
  LOOM_ENCODING_AFFINE_POLICY_SCALE_PLUS_ZERO_POINT = 3,
  LOOM_ENCODING_AFFINE_POLICY_SCALE_PLUS_BIAS = 4,
  LOOM_ENCODING_AFFINE_POLICY_SUPER_SCALE_TIMES_SUBSCALE = 5,
  LOOM_ENCODING_AFFINE_POLICY_SUM_CORRECTION = 6,
  LOOM_ENCODING_AFFINE_POLICY_COUNT_ = 7,
} loom_encoding_affine_policy_t;

// Codebook source for indexed encoded values.
typedef enum loom_encoding_codebook_policy_e {
  LOOM_ENCODING_CODEBOOK_POLICY_NONE = 0,
  LOOM_ENCODING_CODEBOOK_POLICY_STATIC_BUILTIN_TABLE = 1,
  LOOM_ENCODING_CODEBOOK_POLICY_STATIC_SYMBOL_TABLE = 2,
  LOOM_ENCODING_CODEBOOK_POLICY_GLOBAL_DATA_TABLE = 3,
  LOOM_ENCODING_CODEBOOK_POLICY_DYNAMIC_TABLE_OPERAND = 4,
  LOOM_ENCODING_CODEBOOK_POLICY_PER_SUPERBLOCK_TABLE = 5,
  LOOM_ENCODING_CODEBOOK_POLICY_COUNT_ = 6,
} loom_encoding_codebook_policy_t;

// Target-independent encoded numeric format.
typedef enum loom_encoding_numeric_format_e {
  LOOM_ENCODING_NUMERIC_FORMAT_NONE = 0,
  LOOM_ENCODING_NUMERIC_FORMAT_F64 = 1,
  LOOM_ENCODING_NUMERIC_FORMAT_F32 = 2,
  LOOM_ENCODING_NUMERIC_FORMAT_TF32 = 3,
  LOOM_ENCODING_NUMERIC_FORMAT_F16 = 4,
  LOOM_ENCODING_NUMERIC_FORMAT_BF16 = 5,
  LOOM_ENCODING_NUMERIC_FORMAT_I32 = 6,
  LOOM_ENCODING_NUMERIC_FORMAT_U32 = 7,
  LOOM_ENCODING_NUMERIC_FORMAT_I16 = 8,
  LOOM_ENCODING_NUMERIC_FORMAT_U16 = 9,
  LOOM_ENCODING_NUMERIC_FORMAT_I8 = 10,
  LOOM_ENCODING_NUMERIC_FORMAT_U8 = 11,
  LOOM_ENCODING_NUMERIC_FORMAT_I6 = 12,
  LOOM_ENCODING_NUMERIC_FORMAT_U6 = 13,
  LOOM_ENCODING_NUMERIC_FORMAT_I5 = 14,
  LOOM_ENCODING_NUMERIC_FORMAT_U5 = 15,
  LOOM_ENCODING_NUMERIC_FORMAT_I4 = 16,
  LOOM_ENCODING_NUMERIC_FORMAT_U4 = 17,
  LOOM_ENCODING_NUMERIC_FORMAT_I3 = 18,
  LOOM_ENCODING_NUMERIC_FORMAT_U3 = 19,
  LOOM_ENCODING_NUMERIC_FORMAT_I2 = 20,
  LOOM_ENCODING_NUMERIC_FORMAT_U2 = 21,
  LOOM_ENCODING_NUMERIC_FORMAT_I1 = 22,
  LOOM_ENCODING_NUMERIC_FORMAT_U1 = 23,
  LOOM_ENCODING_NUMERIC_FORMAT_F8E4M3 = 24,
  LOOM_ENCODING_NUMERIC_FORMAT_F8E5M2 = 25,
  LOOM_ENCODING_NUMERIC_FORMAT_F8E4M3FN = 26,
  LOOM_ENCODING_NUMERIC_FORMAT_F8E4M3FNUZ = 27,
  LOOM_ENCODING_NUMERIC_FORMAT_F8E5M2FNUZ = 28,
  LOOM_ENCODING_NUMERIC_FORMAT_E8M0 = 29,
  LOOM_ENCODING_NUMERIC_FORMAT_BF8 = 30,
  LOOM_ENCODING_NUMERIC_FORMAT_F6E3M2 = 31,
  LOOM_ENCODING_NUMERIC_FORMAT_F6E2M3 = 32,
  LOOM_ENCODING_NUMERIC_FORMAT_BF6 = 33,
  LOOM_ENCODING_NUMERIC_FORMAT_F4E2M1 = 34,
  LOOM_ENCODING_NUMERIC_FORMAT_TERNARY = 35,
  LOOM_ENCODING_NUMERIC_FORMAT_SIGN_BIT = 36,
  LOOM_ENCODING_NUMERIC_FORMAT_CODEBOOK_INDEX = 37,
  LOOM_ENCODING_NUMERIC_FORMAT_QUANT_I8 = 38,
  LOOM_ENCODING_NUMERIC_FORMAT_QUANT_I6 = 39,
  LOOM_ENCODING_NUMERIC_FORMAT_QUANT_I4 = 40,
  LOOM_ENCODING_NUMERIC_FORMAT_COUNT_ = 41,
} loom_encoding_numeric_format_t;

// Physical payload bit and field layout.
typedef enum loom_encoding_payload_packing_e {
  LOOM_ENCODING_PAYLOAD_PACKING_DENSE_LANES = 1,
  LOOM_ENCODING_PAYLOAD_PACKING_LITTLE_ENDIAN_NIBBLES = 2,
  LOOM_ENCODING_PAYLOAD_PACKING_BIG_ENDIAN_NIBBLES = 3,
  LOOM_ENCODING_PAYLOAD_PACKING_BITFIELD_STREAM = 4,
  LOOM_ENCODING_PAYLOAD_PACKING_BITPLANE_STREAM = 5,
  LOOM_ENCODING_PAYLOAD_PACKING_MULTI_STREAM = 6,
  LOOM_ENCODING_PAYLOAD_PACKING_BASE_N_PACKED = 7,
  LOOM_ENCODING_PAYLOAD_PACKING_CODEBOOK_INDICES = 8,
  LOOM_ENCODING_PAYLOAD_PACKING_TARGET_FRAGMENT = 9,
  LOOM_ENCODING_PAYLOAD_PACKING_INTERLEAVED_SCALE_PAYLOAD = 10,
  LOOM_ENCODING_PAYLOAD_PACKING_SEPARATE_SCALE_PAYLOAD = 11,
  LOOM_ENCODING_PAYLOAD_PACKING_COUNT_ = 12,
} loom_encoding_payload_packing_t;

// Scale sharing topology for encoded values.
typedef enum loom_encoding_scale_topology_e {
  LOOM_ENCODING_SCALE_TOPOLOGY_NONE = 0,
  LOOM_ENCODING_SCALE_TOPOLOGY_TENSOR_GLOBAL = 1,
  LOOM_ENCODING_SCALE_TOPOLOGY_ROW = 2,
  LOOM_ENCODING_SCALE_TOPOLOGY_COLUMN = 3,
  LOOM_ENCODING_SCALE_TOPOLOGY_CHANNEL = 4,
  LOOM_ENCODING_SCALE_TOPOLOGY_GROUP_1D = 5,
  LOOM_ENCODING_SCALE_TOPOLOGY_BLOCK_1D = 6,
  LOOM_ENCODING_SCALE_TOPOLOGY_BLOCK_2D = 7,
  LOOM_ENCODING_SCALE_TOPOLOGY_SUBBLOCK_IN_SUPERBLOCK = 8,
  LOOM_ENCODING_SCALE_TOPOLOGY_HIERARCHICAL = 9,
  LOOM_ENCODING_SCALE_TOPOLOGY_PER_TOKEN = 10,
  LOOM_ENCODING_SCALE_TOPOLOGY_PER_HEAD = 11,
  LOOM_ENCODING_SCALE_TOPOLOGY_PER_PAGE = 12,
  LOOM_ENCODING_SCALE_TOPOLOGY_RUNTIME_AMAX_DERIVED = 13,
  LOOM_ENCODING_SCALE_TOPOLOGY_COUNT_ = 14,
} loom_encoding_scale_topology_t;

// Sparse payload organization for encoded values.
typedef enum loom_encoding_sparsity_policy_e {
  LOOM_ENCODING_SPARSITY_POLICY_NONE = 0,
  LOOM_ENCODING_SPARSITY_POLICY_MASK = 1,
  LOOM_ENCODING_SPARSITY_POLICY_N_M_STRUCTURED = 2,
  LOOM_ENCODING_SPARSITY_POLICY_BLOCK_SPARSE = 3,
  LOOM_ENCODING_SPARSITY_POLICY_BSR = 4,
  LOOM_ENCODING_SPARSITY_POLICY_CSR = 5,
  LOOM_ENCODING_SPARSITY_POLICY_COO = 6,
  LOOM_ENCODING_SPARSITY_POLICY_PAGE_TABLE = 7,
  LOOM_ENCODING_SPARSITY_POLICY_MOE_ROUTING = 8,
  LOOM_ENCODING_SPARSITY_POLICY_OUTLIER_SIDE_STREAM = 9,
  LOOM_ENCODING_SPARSITY_POLICY_COUNT_ = 10,
} loom_encoding_sparsity_policy_t;

enum {
  LOOM_ENCODING_FAMILY_DYNAMIC_PARAMETER_COUNT_MAX_ = 6,
};

// Composes an address layout and storage schema.
typedef enum loom_encoding_physical_storage_parameter_e {
  LOOM_ENCODING_PHYSICAL_STORAGE_PARAMETER_LAYOUT = 0,
  LOOM_ENCODING_PHYSICAL_STORAGE_PARAMETER_SCHEMA = 1,
  LOOM_ENCODING_PHYSICAL_STORAGE_PARAMETER_COUNT_ = 2,
} loom_encoding_physical_storage_parameter_t;

typedef enum loom_encoding_physical_storage_dynamic_parameter_e {
  LOOM_ENCODING_PHYSICAL_STORAGE_DYNAMIC_PARAMETER_LAYOUT = 0,
  LOOM_ENCODING_PHYSICAL_STORAGE_DYNAMIC_PARAMETER_SCHEMA = 1,
  LOOM_ENCODING_PHYSICAL_STORAGE_DYNAMIC_PARAMETER_COUNT_ = 2,
} loom_encoding_physical_storage_dynamic_parameter_t;

// Explicit element-stride address layout.
typedef enum loom_encoding_strided_parameter_e {
  LOOM_ENCODING_STRIDED_PARAMETER_STRIDE = 0,
  LOOM_ENCODING_STRIDED_PARAMETER_STRIDES = 1,
  LOOM_ENCODING_STRIDED_PARAMETER_COUNT_ = 2,
} loom_encoding_strided_parameter_t;

// Blockwise eight-bit quantized storage schema.
typedef enum loom_encoding_q8_0_parameter_e {
  LOOM_ENCODING_Q8_0_PARAMETER_BLOCK = 0,
  LOOM_ENCODING_Q8_0_PARAMETER_COUNT_ = 1,
} loom_encoding_q8_0_parameter_t;

// GGML-compatible block storage schema.
typedef enum loom_encoding_ggml_q4_0_parameter_e {
  LOOM_ENCODING_GGML_Q4_0_PARAMETER_BLOCK_ELEMS = 0,
  LOOM_ENCODING_GGML_Q4_0_PARAMETER_STORAGE_BYTES = 1,
  LOOM_ENCODING_GGML_Q4_0_PARAMETER_COUNT_ = 2,
} loom_encoding_ggml_q4_0_parameter_t;

// GGML-compatible block storage schema.
typedef enum loom_encoding_ggml_q8_0_parameter_e {
  LOOM_ENCODING_GGML_Q8_0_PARAMETER_BLOCK_ELEMS = 0,
  LOOM_ENCODING_GGML_Q8_0_PARAMETER_STORAGE_BYTES = 1,
  LOOM_ENCODING_GGML_Q8_0_PARAMETER_COUNT_ = 2,
} loom_encoding_ggml_q8_0_parameter_t;

// GGML-compatible block storage schema.
typedef enum loom_encoding_ggml_q6_k_parameter_e {
  LOOM_ENCODING_GGML_Q6_K_PARAMETER_BLOCK_ELEMS = 0,
  LOOM_ENCODING_GGML_Q6_K_PARAMETER_STORAGE_BYTES = 1,
  LOOM_ENCODING_GGML_Q6_K_PARAMETER_COUNT_ = 2,
} loom_encoding_ggml_q6_k_parameter_t;

// GGML indexed-grid storage schema.
typedef enum loom_encoding_ggml_iq_grid_parameter_e {
  LOOM_ENCODING_GGML_IQ_GRID_PARAMETER_CODE_BITS = 0,
  LOOM_ENCODING_GGML_IQ_GRID_PARAMETER_GRID_ELEMS = 1,
  LOOM_ENCODING_GGML_IQ_GRID_PARAMETER_COUNT_ = 2,
} loom_encoding_ggml_iq_grid_parameter_t;

// Table-decoded four-bit storage schema.
typedef enum loom_encoding_loom_fp4_table_parameter_e {
  LOOM_ENCODING_LOOM_FP4_TABLE_PARAMETER_CODE_BITS = 0,
  LOOM_ENCODING_LOOM_FP4_TABLE_PARAMETER_TABLE_ELEMS = 1,
  LOOM_ENCODING_LOOM_FP4_TABLE_PARAMETER_COUNT_ = 2,
} loom_encoding_loom_fp4_table_parameter_t;

// Named eight-bit floating-point storage schema.
typedef enum loom_encoding_ieee_fp8_e4m3_parameter_e {
  LOOM_ENCODING_IEEE_FP8_E4M3_PARAMETER_ROUNDING = 0,
  LOOM_ENCODING_IEEE_FP8_E4M3_PARAMETER_STORAGE_BITS = 1,
  LOOM_ENCODING_IEEE_FP8_E4M3_PARAMETER_COUNT_ = 2,
} loom_encoding_ieee_fp8_e4m3_parameter_t;

// Named eight-bit floating-point storage schema.
typedef enum loom_encoding_ieee_fp8_e5m2_parameter_e {
  LOOM_ENCODING_IEEE_FP8_E5M2_PARAMETER_ROUNDING = 0,
  LOOM_ENCODING_IEEE_FP8_E5M2_PARAMETER_STORAGE_BITS = 1,
  LOOM_ENCODING_IEEE_FP8_E5M2_PARAMETER_COUNT_ = 2,
} loom_encoding_ieee_fp8_e5m2_parameter_t;

// Named eight-bit floating-point storage schema.
typedef enum loom_encoding_fp8_e4m3fn_parameter_e {
  LOOM_ENCODING_FP8_E4M3FN_PARAMETER_ROUNDING = 0,
  LOOM_ENCODING_FP8_E4M3FN_PARAMETER_STORAGE_BITS = 1,
  LOOM_ENCODING_FP8_E4M3FN_PARAMETER_COUNT_ = 2,
} loom_encoding_fp8_e4m3fn_parameter_t;

// Named eight-bit floating-point storage schema.
typedef enum loom_encoding_fp8_e4m3fnuz_parameter_e {
  LOOM_ENCODING_FP8_E4M3FNUZ_PARAMETER_ROUNDING = 0,
  LOOM_ENCODING_FP8_E4M3FNUZ_PARAMETER_STORAGE_BITS = 1,
  LOOM_ENCODING_FP8_E4M3FNUZ_PARAMETER_COUNT_ = 2,
} loom_encoding_fp8_e4m3fnuz_parameter_t;

// Named eight-bit floating-point storage schema.
typedef enum loom_encoding_fp8_e5m2fnuz_parameter_e {
  LOOM_ENCODING_FP8_E5M2FNUZ_PARAMETER_ROUNDING = 0,
  LOOM_ENCODING_FP8_E5M2FNUZ_PARAMETER_STORAGE_BITS = 1,
  LOOM_ENCODING_FP8_E5M2FNUZ_PARAMETER_COUNT_ = 2,
} loom_encoding_fp8_e5m2fnuz_parameter_t;

// Target-independent encoded matrix operand schema.
typedef enum loom_encoding_matrix_operand_parameter_e {
  LOOM_ENCODING_MATRIX_OPERAND_PARAMETER_AFFINE = 0,
  LOOM_ENCODING_MATRIX_OPERAND_PARAMETER_CODEBOOK = 1,
  LOOM_ENCODING_MATRIX_OPERAND_PARAMETER_ELEMENT_FORMAT = 2,
  LOOM_ENCODING_MATRIX_OPERAND_PARAMETER_PAYLOAD_ELEMENTS = 3,
  LOOM_ENCODING_MATRIX_OPERAND_PARAMETER_PAYLOAD_PACKING = 4,
  LOOM_ENCODING_MATRIX_OPERAND_PARAMETER_PAYLOAD_REGISTERS = 5,
  LOOM_ENCODING_MATRIX_OPERAND_PARAMETER_ROUNDING = 6,
  LOOM_ENCODING_MATRIX_OPERAND_PARAMETER_SCALE_FORMAT = 7,
  LOOM_ENCODING_MATRIX_OPERAND_PARAMETER_SCALE_GROUP_ELEMENTS = 8,
  LOOM_ENCODING_MATRIX_OPERAND_PARAMETER_SCALE_GROUP_SHAPE = 9,
  LOOM_ENCODING_MATRIX_OPERAND_PARAMETER_SCALE_OPERANDS = 10,
  LOOM_ENCODING_MATRIX_OPERAND_PARAMETER_SCALE_TOPOLOGY = 11,
  LOOM_ENCODING_MATRIX_OPERAND_PARAMETER_SECONDARY_SCALE_FORMAT = 12,
  LOOM_ENCODING_MATRIX_OPERAND_PARAMETER_SPARSITY = 13,
  LOOM_ENCODING_MATRIX_OPERAND_PARAMETER_SPARSITY_GROUP_ELEMENTS = 14,
  LOOM_ENCODING_MATRIX_OPERAND_PARAMETER_SPARSITY_GROUP_NONZERO_ELEMENTS = 15,
  LOOM_ENCODING_MATRIX_OPERAND_PARAMETER_ZERO_SCALE_FALLBACK = 16,
  LOOM_ENCODING_MATRIX_OPERAND_PARAMETER_COUNT_ = 17,
} loom_encoding_matrix_operand_parameter_t;

// Numerical transform with static shape and policy parameters.
typedef enum loom_encoding_numeric_transform_parameter_e {
  LOOM_ENCODING_NUMERIC_TRANSFORM_PARAMETER_FAMILY = 0,
  LOOM_ENCODING_NUMERIC_TRANSFORM_PARAMETER_INPUT_ELEMS = 1,
  LOOM_ENCODING_NUMERIC_TRANSFORM_PARAMETER_NORMALIZATION = 2,
  LOOM_ENCODING_NUMERIC_TRANSFORM_PARAMETER_OUTPUT_ELEMS = 3,
  LOOM_ENCODING_NUMERIC_TRANSFORM_PARAMETER_COUNT_ = 4,
} loom_encoding_numeric_transform_parameter_t;

typedef enum loom_encoding_numeric_transform_dynamic_parameter_e {
  LOOM_ENCODING_NUMERIC_TRANSFORM_DYNAMIC_PARAMETER_INPUT_ELEMS = 0,
  LOOM_ENCODING_NUMERIC_TRANSFORM_DYNAMIC_PARAMETER_MATRIX = 1,
  LOOM_ENCODING_NUMERIC_TRANSFORM_DYNAMIC_PARAMETER_OUTPUT_ELEMS = 2,
  LOOM_ENCODING_NUMERIC_TRANSFORM_DYNAMIC_PARAMETER_PERMUTATION = 3,
  LOOM_ENCODING_NUMERIC_TRANSFORM_DYNAMIC_PARAMETER_SEED = 4,
  LOOM_ENCODING_NUMERIC_TRANSFORM_DYNAMIC_PARAMETER_SIGNS = 5,
  LOOM_ENCODING_NUMERIC_TRANSFORM_DYNAMIC_PARAMETER_COUNT_ = 6,
} loom_encoding_numeric_transform_dynamic_parameter_t;

// TurboQuant key/value storage schema.
typedef enum loom_encoding_turboquant_kv_parameter_e {
  LOOM_ENCODING_TURBOQUANT_KV_PARAMETER_FIRST_STAGE_BITS = 0,
  LOOM_ENCODING_TURBOQUANT_KV_PARAMETER_LOGICAL_ELEMENT = 1,
  LOOM_ENCODING_TURBOQUANT_KV_PARAMETER_LOGICAL_ELEMS = 2,
  LOOM_ENCODING_TURBOQUANT_KV_PARAMETER_PACK_ORDER = 3,
  LOOM_ENCODING_TURBOQUANT_KV_PARAMETER_QJL_ROWS = 4,
  LOOM_ENCODING_TURBOQUANT_KV_PARAMETER_RECORD_BYTES = 5,
  LOOM_ENCODING_TURBOQUANT_KV_PARAMETER_RESIDUAL_BITS = 6,
  LOOM_ENCODING_TURBOQUANT_KV_PARAMETER_SCALAR_QUANTIZER = 7,
  LOOM_ENCODING_TURBOQUANT_KV_PARAMETER_TRANSFORM_FAMILY = 8,
  LOOM_ENCODING_TURBOQUANT_KV_PARAMETER_COUNT_ = 9,
} loom_encoding_turboquant_kv_parameter_t;

typedef enum loom_encoding_turboquant_kv_dynamic_parameter_e {
  LOOM_ENCODING_TURBOQUANT_KV_DYNAMIC_PARAMETER_CENTROIDS = 0,
  LOOM_ENCODING_TURBOQUANT_KV_DYNAMIC_PARAMETER_QJL_TRANSFORM = 1,
  LOOM_ENCODING_TURBOQUANT_KV_DYNAMIC_PARAMETER_THRESHOLDS = 2,
  LOOM_ENCODING_TURBOQUANT_KV_DYNAMIC_PARAMETER_TRANSFORM = 3,
  LOOM_ENCODING_TURBOQUANT_KV_DYNAMIC_PARAMETER_COUNT_ = 4,
} loom_encoding_turboquant_kv_dynamic_parameter_t;

#ifdef __cplusplus
extern "C" {
#endif

extern const loom_encoding_family_descriptor_t loom_encoding_physical_storage_family_descriptor;
extern const loom_encoding_family_descriptor_t loom_encoding_dense_family_descriptor;
extern const loom_encoding_family_descriptor_t loom_encoding_strided_family_descriptor;
extern const loom_encoding_family_descriptor_t loom_encoding_q8_0_family_descriptor;
extern const loom_encoding_family_descriptor_t loom_encoding_ggml_q4_0_family_descriptor;
extern const loom_encoding_family_descriptor_t loom_encoding_ggml_q8_0_family_descriptor;
extern const loom_encoding_family_descriptor_t loom_encoding_q6_k_family_descriptor;
extern const loom_encoding_family_descriptor_t loom_encoding_ggml_q6_k_family_descriptor;
extern const loom_encoding_family_descriptor_t loom_encoding_ggml_iq_grid_family_descriptor;
extern const loom_encoding_family_descriptor_t loom_encoding_loom_fp4_table_family_descriptor;
extern const loom_encoding_family_descriptor_t loom_encoding_ieee_fp8_e4m3_family_descriptor;
extern const loom_encoding_family_descriptor_t loom_encoding_ieee_fp8_e5m2_family_descriptor;
extern const loom_encoding_family_descriptor_t loom_encoding_fp8_e4m3fn_family_descriptor;
extern const loom_encoding_family_descriptor_t loom_encoding_fp8_e4m3fnuz_family_descriptor;
extern const loom_encoding_family_descriptor_t loom_encoding_fp8_e5m2fnuz_family_descriptor;
extern const loom_encoding_family_descriptor_t loom_encoding_matrix_operand_family_descriptor;
extern const loom_encoding_family_descriptor_t loom_encoding_numeric_transform_family_descriptor;
extern const loom_encoding_family_descriptor_t loom_encoding_orthogonal_transform_family_descriptor;
extern const loom_encoding_family_descriptor_t loom_encoding_turboquant_kv_family_descriptor;

enum {
  LOOM_OP_ENCODING_LAYOUT_DENSE = LOOM_OP_KIND(LOOM_DIALECT_ENCODING, 0),
  LOOM_OP_ENCODING_LAYOUT_STRIDED = LOOM_OP_KIND(LOOM_DIALECT_ENCODING, 1),
  LOOM_OP_ENCODING_DEFINE = LOOM_OP_KIND(LOOM_DIALECT_ENCODING, 2),
  LOOM_OP_ENCODING_ISA = LOOM_OP_KIND(LOOM_DIALECT_ENCODING, 3),
  LOOM_OP_ENCODING_LAYOUT_ASSUME_DENSE = LOOM_OP_KIND(LOOM_DIALECT_ENCODING, 4),
  LOOM_OP_ENCODING_LAYOUT_ASSUME_STRIDED = LOOM_OP_KIND(LOOM_DIALECT_ENCODING, 5),
  LOOM_OP_ENCODING_ASSUME_SPEC = LOOM_OP_KIND(LOOM_DIALECT_ENCODING, 6),
  LOOM_OP_ENCODING_COUNT_ = 7,
};

// LOOM_OP_ENCODING_LAYOUT_DENSE: Construct a dense row-major address layout. The consuming view type provides the rank and logical extents.
// %layout = encoding.layout.dense : encoding<layout>
LOOM_DEFINE_ISA(loom_encoding_layout_dense_isa, LOOM_OP_ENCODING_LAYOUT_DENSE)
LOOM_DEFINE_RESULT(loom_encoding_layout_dense_result, 0)
iree_status_t loom_encoding_layout_dense_build(
    loom_builder_t* builder,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_encoding_layout_dense_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_ENCODING_LAYOUT_STRIDED: Construct an address layout from per-dimension element strides. Static and dynamic stride values are interleaved in one bracket list.
// %layout = encoding.layout.strided [%row_stride, 1] : encoding<layout>
LOOM_DEFINE_ISA(loom_encoding_layout_strided_isa, LOOM_OP_ENCODING_LAYOUT_STRIDED)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_encoding_layout_strided_strides, 0)
LOOM_DEFINE_RESULT(loom_encoding_layout_strided_result, 0)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_encoding_layout_strided_static_strides, 0)
iree_status_t loom_encoding_layout_strided_build(
    loom_builder_t* builder,
    const loom_value_id_t* strides,
    iree_host_size_t strides_count,
    const int64_t* static_strides,
    iree_host_size_t static_strides_count,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_encoding_layout_strided_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_encoding_layout_strided_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_ENCODING_DEFINE: Create an encoding value from a static encoding specification.
// %enc = encoding.define #q8_0<block=32> : encoding<schema>
LOOM_DEFINE_ISA(loom_encoding_define_isa, LOOM_OP_ENCODING_DEFINE)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_encoding_define_params, 0)
LOOM_DEFINE_RESULT(loom_encoding_define_result, 0)
LOOM_DEFINE_ATTR_ENCODING(loom_encoding_define_spec, 0)
LOOM_DEFINE_ATTR_DICT(loom_encoding_define_param_names, 1)
iree_status_t loom_encoding_define_build(
    loom_builder_t* builder,
    uint16_t spec,
    const loom_named_value_t* params,
    iree_host_size_t params_count,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_encoding_define_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_encoding_define_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_ENCODING_ISA: Test if an encoding exactly matches a static encoding specification.
// %is_q4 = encoding.isa<#ggml.q4_k> %schema : encoding<schema>
LOOM_DEFINE_ISA(loom_encoding_isa_isa, LOOM_OP_ENCODING_ISA)
LOOM_DEFINE_OPERAND(loom_encoding_isa_enc, 0)
LOOM_DEFINE_RESULT(loom_encoding_isa_result, 0)
LOOM_DEFINE_ATTR_ENCODING(loom_encoding_isa_spec, 0)
iree_status_t loom_encoding_isa_build(
    loom_builder_t* builder,
    uint16_t spec,
    loom_value_id_t enc,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_encoding_isa_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_encoding_isa_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_ENCODING_LAYOUT_ASSUME_DENSE: Refine an existing address-layout encoding value with the fact that it is dense row-major. The result is the same encoding value in SSA form with stronger local facts.
// %dense = encoding.layout.assume.dense %layout : encoding<layout>
LOOM_DEFINE_ISA(loom_encoding_layout_assume_dense_isa, LOOM_OP_ENCODING_LAYOUT_ASSUME_DENSE)
LOOM_DEFINE_OPERAND(loom_encoding_layout_assume_dense_layout, 0)
LOOM_DEFINE_RESULT(loom_encoding_layout_assume_dense_result, 0)
iree_status_t loom_encoding_layout_assume_dense_build(
    loom_builder_t* builder,
    loom_value_id_t layout,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_encoding_layout_assume_dense_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_ENCODING_LAYOUT_ASSUME_STRIDED: Refine an existing address-layout encoding value with the fact that it is strided and has the given rank. Per-axis stride values remain unknown unless a concrete encoding.layout.strided value is available.
// %strided = encoding.layout.assume.strided %layout {rank = 2} : encoding<layout>
LOOM_DEFINE_ISA(loom_encoding_layout_assume_strided_isa, LOOM_OP_ENCODING_LAYOUT_ASSUME_STRIDED)
LOOM_DEFINE_OPERAND(loom_encoding_layout_assume_strided_layout, 0)
LOOM_DEFINE_RESULT(loom_encoding_layout_assume_strided_result, 0)
LOOM_DEFINE_ATTR_I64(loom_encoding_layout_assume_strided_rank, 0)
iree_status_t loom_encoding_layout_assume_strided_build(
    loom_builder_t* builder,
    loom_value_id_t layout,
    int64_t rank,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_encoding_layout_assume_strided_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_encoding_layout_assume_strided_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_ENCODING_ASSUME_SPEC: Refine an existing encoding value with an exact static encoding specification. Dynamic values remain ordinary SSA operands elsewhere; this op only states the selected static family and static parameters.
// %schema2 = encoding.assume.spec %schema, #ggml_q4_0<block_elems=32, storage_bytes=18> : encoding<schema>
LOOM_DEFINE_ISA(loom_encoding_assume_spec_isa, LOOM_OP_ENCODING_ASSUME_SPEC)
LOOM_DEFINE_OPERAND(loom_encoding_assume_spec_enc, 0)
LOOM_DEFINE_RESULT(loom_encoding_assume_spec_result, 0)
LOOM_DEFINE_ATTR_ENCODING(loom_encoding_assume_spec_spec, 0)
iree_status_t loom_encoding_assume_spec_build(
    loom_builder_t* builder,
    loom_value_id_t enc,
    uint16_t spec,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_encoding_assume_spec_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_encoding_assume_spec_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// Returns the vtable array for the encoding dialect.
const loom_op_vtable_t* const* loom_encoding_dialect_vtables(
    iree_host_size_t* out_count);

// Returns the dense semantic metadata array for the encoding dialect.
const loom_op_semantics_t* loom_encoding_dialect_op_semantics(
    iree_host_size_t* out_count);

// Returns semantic metadata for a encoding op kind, or empty metadata.
loom_op_semantics_t loom_encoding_op_semantics(
    loom_op_kind_t kind);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_ENCODING_OPS_H_
