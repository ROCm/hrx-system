// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU source-to-low lowering plans selected before emission.
//
// These structs are immutable emission contracts. The planner computes them
// once from source IR, facts, target record, and descriptor availability; the
// emitter consumes them without re-running legality or descriptor selection.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_PLAN_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_PLAN_H_

#include <stdint.h>

#include "loom/codegen/low/lower/lower.h"
#include "loom/codegen/low/source_memory_plan.h"
#include "loom/ir/ir.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/kernel/ops.h"
#include "loom/target/arch/amdgpu/lower/kinds.h"
#include "loom/target/arch/amdgpu/matrix/contract.h"
#include "loom/target/arch/amdgpu/planning/wait_counters.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/util/numeric_format.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_amdgpu_constant_plan_kind_e {
  LOOM_AMDGPU_CONSTANT_PLAN_KIND_NONE = 0,
  LOOM_AMDGPU_CONSTANT_PLAN_KIND_U32_BITS = 1,
  LOOM_AMDGPU_CONSTANT_PLAN_KIND_I1_SCC = 2,
  LOOM_AMDGPU_CONSTANT_PLAN_KIND_I1_MASK = 3,
} loom_amdgpu_constant_plan_kind_t;

typedef struct loom_amdgpu_constant_plan_t {
  // Representation-specific lowering selected for the constant.
  loom_amdgpu_constant_plan_kind_t kind;
  // Source result value receiving the emitted low constant.
  loom_value_id_t result;
  // Primary descriptor row selected for the constant packet.
  loom_low_lower_resolved_descriptor_t descriptor;
  // Descriptor row selected for an auxiliary zero constant.
  loom_low_lower_resolved_descriptor_t zero_descriptor;
  // Module string ID for the descriptor's imm32 attribute.
  loom_string_id_t imm32_attr_name_id;
  // Number of 32-bit registers receiving immediate bit patterns.
  uint32_t register_count;
  // Immediate bit patterns emitted into selected result registers.
  uint32_t bit_patterns[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  // Boolean payload for i1 constants.
  bool i1_value;
} loom_amdgpu_constant_plan_t;

typedef enum loom_amdgpu_i8_pack_permute_kind_e {
  LOOM_AMDGPU_I8_PACK_PERMUTE_KIND_NONE = 0,
  LOOM_AMDGPU_I8_PACK_PERMUTE_KIND_LITERAL_SELECTOR = 1,
  LOOM_AMDGPU_I8_PACK_PERMUTE_KIND_REGISTER_SELECTOR = 2,
} loom_amdgpu_i8_pack_permute_kind_t;

typedef struct loom_amdgpu_i8_pack_permute_plan_t {
  // Representation selected for V_PERM_B32 byte selector operands.
  loom_amdgpu_i8_pack_permute_kind_t kind;
  // Descriptor selected for each byte permutation packet.
  loom_amdgpu_descriptor_ref_t descriptor_ref;
} loom_amdgpu_i8_pack_permute_plan_t;

typedef enum loom_amdgpu_fp8_encode_kind_e {
  LOOM_AMDGPU_FP8_ENCODE_KIND_NONE = 0,
  LOOM_AMDGPU_FP8_ENCODE_KIND_F32_PAIR = 1,
  LOOM_AMDGPU_FP8_ENCODE_KIND_F32_PAIR_SATURATE_E4M3 = 2,
  LOOM_AMDGPU_FP8_ENCODE_KIND_F16_PAIR = 3,
  LOOM_AMDGPU_FP8_ENCODE_KIND_F32_SOFTWARE_E4M3 = 4,
  LOOM_AMDGPU_FP8_ENCODE_KIND_F32_SOFTWARE_E5M2 = 5,
  LOOM_AMDGPU_FP8_ENCODE_KIND_F32_FNUZ_BRIDGE_E4M3 = 6,
  LOOM_AMDGPU_FP8_ENCODE_KIND_F32_FNUZ_BRIDGE_E5M2 = 7,
  LOOM_AMDGPU_FP8_ENCODE_KIND_F16_SOFTWARE_E5M2 = 8,
} loom_amdgpu_fp8_encode_kind_t;

typedef enum loom_amdgpu_fp8_encode_plan_flag_bits_e {
  LOOM_AMDGPU_FP8_ENCODE_PLAN_FLAG_NONE = 0u,
  // Packed 16-bit packets can encode complete F16 E5M2 lane groups.
  LOOM_AMDGPU_FP8_ENCODE_PLAN_FLAG_PACKED_F16_E5M2 = 1u << 0,
  // Packed 16-bit packets can encode complete F16 E4M3 lane groups.
  LOOM_AMDGPU_FP8_ENCODE_PLAN_FLAG_PACKED_F16_E4M3 = 1u << 1,
  // Native OCP pair packets need their packed NaN bytes canonicalized.
  LOOM_AMDGPU_FP8_ENCODE_PLAN_FLAG_NATIVE_NAN_CANONICALIZATION = 1u << 2,
} loom_amdgpu_fp8_encode_plan_flag_bits_t;
typedef uint32_t loom_amdgpu_fp8_encode_plan_flags_t;

typedef struct loom_amdgpu_fp8_encode_plan_t {
  // Exact encoding strategy selected from the target descriptor set.
  loom_amdgpu_fp8_encode_kind_t kind;
  // Optional packet-sequence capabilities refining the selected strategy.
  loom_amdgpu_fp8_encode_plan_flags_t flags;
  // Exact exponent, mantissa, bias, and special-value semantics.
  loom_scalar_type_fp8_format_t format;
  // Descriptor writing the low encoded byte pair of a result register.
  loom_amdgpu_descriptor_ref_t low_descriptor_ref;
  // Descriptor continuing a result register with its high encoded byte pair.
  loom_amdgpu_descriptor_ref_t high_descriptor_ref;
  // Optional fused descriptor adding the rounding LSB and literal bias.
  loom_amdgpu_descriptor_ref_t round_add_descriptor_ref;
  // Optional descriptor inserting encoded sign bits into a result word.
  loom_amdgpu_descriptor_ref_t sign_insert_descriptor_ref;
  // Byte permutation plan selected for encoded bytes or packed sign bits.
  loom_amdgpu_i8_pack_permute_plan_t packed_i8_permute;
} loom_amdgpu_fp8_encode_plan_t;

typedef enum loom_amdgpu_vector_16bit_float_conversion_kind_e {
  LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_NONE = 0,
  LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_EXTF = 1,
  LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_FPTRUNC = 2,
  LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_DECODE = 3,
  LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_ENCODE = 4,
} loom_amdgpu_vector_16bit_float_conversion_kind_t;

typedef struct loom_amdgpu_fp4_decode_recipe_t loom_amdgpu_fp4_decode_recipe_t;
typedef struct loom_amdgpu_fp4_native_pair_decode_recipe_t
    loom_amdgpu_fp4_native_pair_decode_recipe_t;
typedef struct loom_amdgpu_fp4_native_pk8_decode_recipe_t
    loom_amdgpu_fp4_native_pk8_decode_recipe_t;

typedef enum loom_amdgpu_fp4_decode_kind_e {
  LOOM_AMDGPU_FP4_DECODE_KIND_NONE = 0,
  LOOM_AMDGPU_FP4_DECODE_KIND_PORTABLE_LOOKUP = 1,
  LOOM_AMDGPU_FP4_DECODE_KIND_NATIVE_SCALEF32_PAIR = 2,
  LOOM_AMDGPU_FP4_DECODE_KIND_NATIVE_E8M0_PK8 = 3,
} loom_amdgpu_fp4_decode_kind_t;

typedef struct loom_amdgpu_fp4_decode_plan_t {
  // Packet strategy selected from the schema and target descriptor set.
  loom_amdgpu_fp4_decode_kind_t kind;
  // Function-local portable lookup recipe when kind is PORTABLE_LOOKUP.
  const loom_amdgpu_fp4_decode_recipe_t* portable_recipe;
  // Function-local scaled-pair recipe when kind is NATIVE_SCALEF32_PAIR.
  const loom_amdgpu_fp4_native_pair_decode_recipe_t* native_pair_recipe;
  // Function-local eight-lane recipe when kind is NATIVE_E8M0_PK8.
  const loom_amdgpu_fp4_native_pk8_decode_recipe_t* native_pk8_recipe;
} loom_amdgpu_fp4_decode_plan_t;

typedef struct loom_amdgpu_vector_16bit_float_conversion_plan_t {
  // Source vector value being converted.
  loom_value_id_t source;
  // Result vector value receiving the converted lane payload.
  loom_value_id_t result;
  // Source vector whose storage materializes the logical conversion lanes.
  loom_value_id_t storage_source;
  // Value carrying logical lane content facts for FP8 simplification.
  loom_value_id_t content_fact_source;
  // Optional scale source for scaled vector.decode operations.
  loom_value_id_t scale_source;
  // Numeric format of scale_source, or NONE when there is no scale source.
  loom_value_fact_numeric_format_flags_t scale_format;
  // Number of logical payload lanes covered by each scale value.
  uint32_t scale_group_element_count;
  // Number of encoded scale values covering the conversion.
  uint32_t scale_count;
  // Number of 32-bit registers carrying the encoded scale values.
  uint32_t scale_register_count;
  // Conversion operation selected for the source/result type pair.
  loom_amdgpu_vector_16bit_float_conversion_kind_t kind;
  // Source scalar element type.
  loom_scalar_type_t source_element_type;
  // Exact numeric format represented by the source payload.
  loom_value_fact_numeric_format_flags_t source_format;
  // Semantically equivalent source format accepted by native descriptors.
  loom_value_fact_numeric_format_flags_t descriptor_source_format;
  // Result scalar element type.
  loom_scalar_type_t result_element_type;
  // Static vector lane count.
  uint32_t lane_count;
  // Number of 32-bit source registers occupied by the source vector.
  uint32_t source_register_count;
  // First logical lane read from storage_source for result lane zero.
  uint32_t storage_lane_offset;
  // Logical lane stride through storage_source for adjacent result lanes.
  uint32_t storage_lane_stride;
  // Number of scalar lanes proven available in storage_source.
  uint32_t storage_lane_count;
  // Number of 32-bit registers occupied by storage_source.
  uint32_t storage_register_count;
  // Number of 32-bit result registers occupied by the result vector.
  uint32_t result_register_count;
  // Packed FP4 decode strategy, or NONE for other conversions.
  loom_amdgpu_fp4_decode_plan_t fp4_decode;
  // Native packed FP8 encode strategy for an FP8-result truncation.
  loom_amdgpu_fp8_encode_plan_t fp8_encode;
} loom_amdgpu_vector_16bit_float_conversion_plan_t;

typedef enum loom_amdgpu_index_cast_kind_e {
  LOOM_AMDGPU_INDEX_CAST_KIND_NONE = 0,
  LOOM_AMDGPU_INDEX_CAST_KIND_ALIAS = 1,
  LOOM_AMDGPU_INDEX_CAST_KIND_PRESERVING_LOW_32 = 2,
  LOOM_AMDGPU_INDEX_CAST_KIND_ZERO_EXTENDING_LOW_32 = 3,
  LOOM_AMDGPU_INDEX_CAST_KIND_DIAGNOSTIC_REJECTED = 4,
} loom_amdgpu_index_cast_kind_t;

typedef struct loom_amdgpu_index_cast_plan_t {
  // Lowering strategy selected for the index cast.
  loom_amdgpu_index_cast_kind_t kind;
  // Source value being cast.
  loom_value_id_t source;
  // Result value receiving the cast payload.
  loom_value_id_t result;
  // Descriptor materializing the high zero lane for a widening cast.
  loom_amdgpu_descriptor_ref_t zero_descriptor_ref;
  // Target index bit width used by width-changing casts.
  uint32_t index_bitwidth;
} loom_amdgpu_index_cast_plan_t;

typedef enum loom_amdgpu_address_i64_alu_kind_e {
  LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_NONE = 0,
  LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_SGPR_ADD = 1,
  LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_ADD = 2,
  LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_SUB = 3,
  LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_MUL_LO = 4,
  LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_SHL = 5,
  LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_MADD_LO = 6,
} loom_amdgpu_address_i64_alu_kind_t;

typedef struct loom_amdgpu_address_i64_alu_plan_t {
  // Left-hand address-domain value.
  loom_value_id_t lhs;
  // Right-hand address-domain value.
  loom_value_id_t rhs;
  // Addend address-domain value for multiply-add operations.
  loom_value_id_t addend;
  // Result address-domain value receiving the full-width result.
  loom_value_id_t result;
  // Lowering strategy selected for the full-width address operation.
  loom_amdgpu_address_i64_alu_kind_t kind;
} loom_amdgpu_address_i64_alu_plan_t;

typedef struct loom_amdgpu_i64_compare_plan_t {
  // Left-hand 64-bit source value.
  loom_value_id_t lhs;
  // Right-hand 64-bit source value.
  loom_value_id_t rhs;
  // Result i1 value receiving the native lane mask.
  loom_value_id_t result;
  // Descriptor comparing high 32-bit lanes.
  loom_amdgpu_descriptor_ref_t high_descriptor_ref;
  // Descriptor comparing low 32-bit lanes.
  loom_amdgpu_descriptor_ref_t low_descriptor_ref;
  // Descriptor combining high and low lane masks.
  loom_amdgpu_descriptor_ref_t combine_descriptor_ref;
  // True when low-lane comparison is guarded by high-lane equality.
  bool needs_high_equal;
} loom_amdgpu_i64_compare_plan_t;

typedef enum loom_amdgpu_scalar_i64_alu_kind_e {
  LOOM_AMDGPU_SCALAR_I64_ALU_KIND_NONE = 0,
  LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_ADD = 1,
  LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_SUB = 2,
  LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_MUL_LO = 3,
  LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_SHL = 4,
} loom_amdgpu_scalar_i64_alu_kind_t;

typedef struct loom_amdgpu_scalar_i64_alu_plan_t {
  // Left-hand 64-bit source value.
  loom_value_id_t lhs;
  // Right-hand 64-bit source value.
  loom_value_id_t rhs;
  // Source result value receiving the emitted low result.
  loom_value_id_t result;
  // Lowering strategy selected for the scalar i64 ALU op.
  loom_amdgpu_scalar_i64_alu_kind_t kind;
} loom_amdgpu_scalar_i64_alu_plan_t;

typedef enum loom_amdgpu_scalar_i64_ctpop_kind_e {
  LOOM_AMDGPU_SCALAR_I64_CTPOP_KIND_NONE = 0,
  LOOM_AMDGPU_SCALAR_I64_CTPOP_KIND_SGPR_B32 = 1,
  LOOM_AMDGPU_SCALAR_I64_CTPOP_KIND_SGPR_B64 = 2,
  LOOM_AMDGPU_SCALAR_I64_CTPOP_KIND_VGPR_B32 = 3,
  LOOM_AMDGPU_SCALAR_I64_CTPOP_KIND_VGPR_B64 = 4,
} loom_amdgpu_scalar_i64_ctpop_kind_t;

typedef struct loom_amdgpu_scalar_i64_ctpop_plan_t {
  // Source 64-bit integer whose set bits are counted.
  loom_value_id_t source;
  // Result 64-bit integer receiving the zero-extended population count.
  loom_value_id_t result;
  // Register-bank-specific population-count strategy.
  loom_amdgpu_scalar_i64_ctpop_kind_t kind;
} loom_amdgpu_scalar_i64_ctpop_plan_t;

typedef enum loom_amdgpu_scalar_cttz_kind_e {
  LOOM_AMDGPU_SCALAR_CTTZ_KIND_NONE = 0,
  LOOM_AMDGPU_SCALAR_CTTZ_KIND_SGPR_B32 = 1,
  LOOM_AMDGPU_SCALAR_CTTZ_KIND_SGPR_B64 = 2,
  LOOM_AMDGPU_SCALAR_CTTZ_KIND_VGPR_B32 = 3,
  LOOM_AMDGPU_SCALAR_CTTZ_KIND_VGPR_B64 = 4,
} loom_amdgpu_scalar_cttz_kind_t;

typedef uint8_t loom_amdgpu_scalar_cttz_flags_t;

// The source value is known to be nonzero, so native CTZ needs no zero repair.
#define LOOM_AMDGPU_SCALAR_CTTZ_FLAG_SOURCE_NONZERO ((uint8_t)1u << 0)

typedef struct loom_amdgpu_scalar_cttz_plan_t {
  // Source integer whose trailing zero bits are counted.
  loom_value_id_t source;
  // Result integer receiving the trailing-zero count.
  loom_value_id_t result;
  // Register-bank and physical-width lowering strategy.
  loom_amdgpu_scalar_cttz_kind_t kind;
  // Declared source width governing the zero result.
  uint8_t semantic_bit_width;
  // Fact-derived lowering properties.
  loom_amdgpu_scalar_cttz_flags_t flags;
} loom_amdgpu_scalar_cttz_plan_t;

typedef enum loom_amdgpu_scalar_conversion_kind_e {
  LOOM_AMDGPU_SCALAR_CONVERSION_KIND_NONE = 0,
  LOOM_AMDGPU_SCALAR_CONVERSION_KIND_ALIAS,
  LOOM_AMDGPU_SCALAR_CONVERSION_KIND_TRUNCATE_LOW_32,
  LOOM_AMDGPU_SCALAR_CONVERSION_KIND_SIGN_EXTEND_NARROW,
  LOOM_AMDGPU_SCALAR_CONVERSION_KIND_SIGN_EXTEND_NARROW_LOW_32,
  LOOM_AMDGPU_SCALAR_CONVERSION_KIND_SIGN_EXTEND_I64,
  LOOM_AMDGPU_SCALAR_CONVERSION_KIND_ZERO_EXTEND,
  LOOM_AMDGPU_SCALAR_CONVERSION_KIND_UITOFP_NARROW_TO_F32,
  LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FP8_TO_BF16,
  LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FP8_ENCODE,
  LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FPTOI_F32_TO_I32,
  LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FPTOI_F32_TO_NARROW,
} loom_amdgpu_scalar_conversion_kind_t;

typedef struct loom_amdgpu_scalar_conversion_plan_t {
  // Source value being converted.
  loom_value_id_t source;
  // Result value receiving the converted payload.
  loom_value_id_t result;
  // Lowering strategy selected for the source/result type pair.
  loom_amdgpu_scalar_conversion_kind_t kind;
  // Static source integer payload bit count, or zero for non-integer sources.
  uint32_t source_bit_count;
  // Static result integer payload bit count, or zero for non-integer results.
  uint32_t result_bit_count;
  // Descriptor selected for conversion packets used by the strategy.
  loom_amdgpu_descriptor_ref_t convert_descriptor_ref;
  // Native packed FP8 encode strategy for an FP8-result truncation.
  loom_amdgpu_fp8_encode_plan_t fp8_encode;
} loom_amdgpu_scalar_conversion_plan_t;

typedef enum loom_amdgpu_vector_conversion_kind_e {
  LOOM_AMDGPU_VECTOR_CONVERSION_KIND_NONE = 0,
  LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_32_TO_FULL_32,
  LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_64_TO_FULL_32,
  LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_32_TO_PACKED_INTEGER,
  LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_64_TO_PACKED_INTEGER,
  LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_INTEGER_TO_FULL_32,
  LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_INTEGER_TO_PACKED_INTEGER,
  LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_U8_TO_F32,
  LOOM_AMDGPU_VECTOR_CONVERSION_KIND_COUNT_,
} loom_amdgpu_vector_conversion_kind_t;

typedef struct loom_amdgpu_vector_conversion_plan_t {
  // Source vector value being converted.
  loom_value_id_t source;
  // Result vector value receiving the converted payload.
  loom_value_id_t result;
  // Lowering strategy selected for the source/result storage pair.
  loom_amdgpu_vector_conversion_kind_t kind;
  // Source scalar element type.
  loom_scalar_type_t source_element_type;
  // Result scalar element type.
  loom_scalar_type_t result_element_type;
  // Static source lane payload bit count.
  uint32_t source_bit_count;
  // Static result lane payload bit count.
  uint32_t result_bit_count;
  // Static vector lane count shared by source and result.
  uint32_t lane_count;
  // Number of 32-bit source registers occupied by the source vector.
  uint32_t source_register_count;
  // Number of 32-bit result registers occupied by the result vector.
  uint32_t result_register_count;
  // Number of 32-bit source registers occupied by one source lane.
  uint32_t source_element_register_count;
  // Descriptor selected for conversion packets used by the strategy.
  loom_amdgpu_descriptor_ref_t convert_descriptor_ref;
  // Byte permutation plan selected for full-register i8 result assembly.
  loom_amdgpu_i8_pack_permute_plan_t packed_i8_permute;
  // True when packed integer source lanes require sign extension.
  bool sign_extend_packed_source;
} loom_amdgpu_vector_conversion_plan_t;

typedef struct loom_amdgpu_bitpack_plan_t {
  // Source vector value containing unpacked i32 lanes.
  loom_value_id_t source;
  // Result vector value containing packed i8 lanes.
  loom_value_id_t result;
  // Number of low bits packed from each source lane.
  uint32_t width;
  // Number of unpacked source lanes.
  uint32_t lane_count;
  // Number of packed 32-bit registers in the result.
  uint32_t result_register_count;
  // Byte permutation plan selected for full i8 register packs.
  loom_amdgpu_i8_pack_permute_plan_t i8_permute;
} loom_amdgpu_bitpack_plan_t;

typedef enum loom_amdgpu_bitunpack_result_kind_e {
  LOOM_AMDGPU_BITUNPACK_RESULT_KIND_NONE = 0,
  LOOM_AMDGPU_BITUNPACK_RESULT_KIND_I32_LANES = 1,
  LOOM_AMDGPU_BITUNPACK_RESULT_KIND_PACKED_I8 = 2,
} loom_amdgpu_bitunpack_result_kind_t;

typedef struct loom_amdgpu_bitunpack_plan_t {
  // Source vector value containing packed integer bitstream storage.
  loom_value_id_t source;
  // Result vector value receiving unpacked lanes.
  loom_value_id_t result;
  // Selected result payload representation.
  loom_amdgpu_bitunpack_result_kind_t result_kind;
  // Number of bits unpacked into each result lane.
  uint32_t width;
  // Number of packed 32-bit source registers.
  uint32_t source_register_count;
  // Number of packed 32-bit result registers.
  uint32_t result_register_count;
  // Number of unpacked result lanes.
  uint32_t lane_count;
  // True when unpacked lanes are sign-extended.
  bool is_signed;
} loom_amdgpu_bitunpack_plan_t;

typedef enum loom_amdgpu_dotf_accumulation_kind_e {
  LOOM_AMDGPU_DOTF_ACCUMULATION_STRICT_CHAIN = 0,
  LOOM_AMDGPU_DOTF_ACCUMULATION_RELAXED_FOREST = 1,
} loom_amdgpu_dotf_accumulation_kind_t;

typedef enum loom_amdgpu_dotf_init_kind_e {
  LOOM_AMDGPU_DOTF_INIT_GENERIC = 0,
  LOOM_AMDGPU_DOTF_INIT_ZERO = 1,
} loom_amdgpu_dotf_init_kind_t;

typedef struct loom_amdgpu_dotf_plan_t {
  // Left-hand source vector value.
  loom_value_id_t lhs;
  // Right-hand source vector value.
  loom_value_id_t rhs;
  // Scalar accumulator seed value.
  loom_value_id_t init;
  // Scalar dot-product result value.
  loom_value_id_t result;
  // Static number of f32 vector lanes.
  uint32_t lane_count;
  // Selected accumulation topology.
  loom_amdgpu_dotf_accumulation_kind_t accumulation_kind;
  // Accumulator seed identity proven by the planner.
  loom_amdgpu_dotf_init_kind_t init_kind;
  // Optional tied-accumulator packet used after the accumulator is dot-local.
  loom_amdgpu_descriptor_ref_t tied_accumulate_descriptor_ref;
} loom_amdgpu_dotf_plan_t;

typedef uint32_t loom_amdgpu_fma_mix_plan_flags_t;

enum {
  // Source 2 is encoded as a literal positive zero immediate.
  LOOM_AMDGPU_FMA_MIX_PLAN_SRC2_LITERAL_ZERO = 1u << 0,
  // Source 2 is supplied by a materialized VGPR positive zero.
  LOOM_AMDGPU_FMA_MIX_PLAN_SRC2_MATERIALIZED_ZERO = 1u << 1,
};

typedef struct loom_amdgpu_fma_mix_plan_t {
  // Source values consumed by the selected descriptor in a, b, c order.
  loom_value_id_t sources[LOOM_AMDGPU_FMA_MIX_SOURCE_COUNT];
  // Source register-unit offsets consumed by the selected descriptor.
  uint32_t source_register_offsets[LOOM_AMDGPU_FMA_MIX_SOURCE_COUNT];
  // Source result value produced by the selected mixed-FMA packet.
  loom_value_id_t result;
  // Descriptor row selected for the mixed-source fma/mad packet.
  loom_amdgpu_descriptor_ref_t descriptor_ref;
  // Descriptor source interpretation for each source value.
  loom_amdgpu_fma_mix_source_kind_t
      source_kinds[LOOM_AMDGPU_FMA_MIX_SOURCE_COUNT];
  // Flags describing implicit literal or materialized operands.
  loom_amdgpu_fma_mix_plan_flags_t flags;
} loom_amdgpu_fma_mix_plan_t;

typedef struct loom_amdgpu_packed_ternary_plan_t {
  // Packed vector values consumed in the selected descriptor's operand order.
  loom_value_id_t sources[LOOM_AMDGPU_PACKED_TERNARY_SOURCE_COUNT];
  // Packed vector result value.
  loom_value_id_t result;
  // Descriptor row selected for each packed ternary packet.
  loom_amdgpu_descriptor_ref_t descriptor_ref;
  // Flags describing selected descriptor packet semantics.
  loom_amdgpu_packed_ternary_flags_t flags;
  // Number of 32-bit register units in each source and result vector.
  uint32_t register_count;
  // Number of 32-bit register units consumed and produced by each packet.
  uint32_t packet_unit_count;
  // Number of descriptor packets emitted to cover the full vector payload.
  uint32_t packet_count;
} loom_amdgpu_packed_ternary_plan_t;

typedef struct loom_amdgpu_mulf_mix_plan_t {
  // Source values consumed by the selected descriptor in a, b order.
  loom_value_id_t sources[LOOM_AMDGPU_MULF_MIX_SOURCE_COUNT];
  // Source register-unit offsets consumed by the selected descriptor.
  uint32_t source_register_offsets[LOOM_AMDGPU_MULF_MIX_SOURCE_COUNT];
  // Scalar or vector mulf result value.
  loom_value_id_t result;
  // Descriptor row selected for the mixed-source fma/mad packet.
  loom_amdgpu_descriptor_ref_t descriptor_ref;
  // True when the selected descriptor encodes source 2 as a literal zero.
  bool addend_literal_zero;
  // Descriptor source interpretation for each multiplicand source value.
  loom_amdgpu_fma_mix_source_kind_t
      source_kinds[LOOM_AMDGPU_MULF_MIX_SOURCE_COUNT];
  // Static f32 lane count produced by the multiply.
  uint32_t lane_count;
} loom_amdgpu_mulf_mix_plan_t;

typedef struct loom_amdgpu_vector_bitcast_plan_t {
  // Source vector value being reinterpreted.
  loom_value_id_t source;
  // Result vector value receiving the same register payload.
  loom_value_id_t result;
} loom_amdgpu_vector_bitcast_plan_t;

typedef struct loom_amdgpu_vector_register_map_plan_t {
  // Source vector values read by the register map.
  loom_value_id_t sources[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  // Result vector value receiving mapped source registers.
  loom_value_id_t result;
  // Static number of active source values.
  uint32_t source_count;
  // Static 32-bit backing register count for the result vector.
  uint32_t result_register_count;
  // Static 32-bit backing register count for each source value.
  uint32_t source_register_counts[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  // Source table index selected for each result register.
  uint32_t result_source_indices[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  // Source register index selected for each result register.
  uint32_t source_register_indices[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
} loom_amdgpu_vector_register_map_plan_t;

typedef enum loom_amdgpu_vector_even_odd_kind_e {
  LOOM_AMDGPU_VECTOR_EVEN_ODD_KIND_NONE = 0,
  LOOM_AMDGPU_VECTOR_EVEN_ODD_KIND_32BIT_LANES = 1,
  LOOM_AMDGPU_VECTOR_EVEN_ODD_KIND_PACKED_16BIT_FLOAT = 2,
} loom_amdgpu_vector_even_odd_kind_t;

typedef struct loom_amdgpu_vector_deinterleave_plan_t {
  // Source vector value split into even and odd lane payloads.
  loom_value_id_t source;
  // Even-position result vector followed by odd-position result vector.
  loom_value_id_t results[2];
  // Selected lowering strategy for source/result storage.
  loom_amdgpu_vector_even_odd_kind_t kind;
  // Static logical lane count for each result vector.
  uint32_t result_lane_count;
  // Static 32-bit backing register count for the source vector.
  uint32_t source_register_count;
  // Static 32-bit backing register count for each result vector.
  uint32_t result_register_count;
  // Optional literal-selector byte permute descriptor for packed 16-bit lanes.
  loom_low_lower_resolved_descriptor_t packed_permute_descriptor;
} loom_amdgpu_vector_deinterleave_plan_t;

typedef struct loom_amdgpu_vector_interleave_plan_t {
  // Even-position source vector followed by odd-position source vector.
  loom_value_id_t sources[2];
  // Result vector value receiving the interleaved payload.
  loom_value_id_t result;
  // Selected lowering strategy for source/result storage.
  loom_amdgpu_vector_even_odd_kind_t kind;
  // Static 32-bit backing register count for each source vector.
  uint32_t source_register_count;
  // Static 32-bit backing register count for the result vector.
  uint32_t result_register_count;
  // Optional literal-selector byte permute descriptor for packed 16-bit lanes.
  loom_low_lower_resolved_descriptor_t packed_permute_descriptor;
} loom_amdgpu_vector_interleave_plan_t;

typedef struct loom_amdgpu_vector_extract_plan_t {
  // Source vector value containing the extracted payload.
  loom_value_id_t source;
  // Optional dynamic source lane index, or invalid for static extraction.
  loom_value_id_t dynamic_index;
  // Result scalar or vector value receiving the extracted payload.
  loom_value_id_t result;
  // Static flattened logical source lane offset.
  uint32_t lane_offset;
  // Static source lane count for dynamic scalar extraction.
  uint32_t lane_count;
  // Number of 32-bit register units occupied by the source payload.
  uint32_t register_count;
  // Number of 32-bit register units occupied by the result payload.
  uint32_t result_register_count;
  // Number of 32-bit register units occupied by one logical source lane.
  uint32_t element_register_count;
  // Number of payload bits occupied by each logical source lane.
  uint32_t lane_bit_count;
  // True when packed integer extraction must produce scalar sign-extension.
  bool sign_extend_packed_lane;
  // True when extraction uses |dynamic_index| instead of |lane_offset|.
  bool is_dynamic;
} loom_amdgpu_vector_extract_plan_t;

typedef struct loom_amdgpu_vector_transform_plan_t {
  // Source vector consumed by the register-local transform.
  loom_value_id_t source;
  // Result vector produced by the register-local transform.
  loom_value_id_t result;
  // Total number of 32-bit lanes in the source and result vectors.
  uint32_t lane_count;
  // Number of lanes in each independently transformed final-axis slice.
  uint32_t slice_extent;
  // Optional f32 scale bit pattern applied after the butterfly stages.
  uint32_t normalization_scale_bits;
} loom_amdgpu_vector_transform_plan_t;

typedef struct loom_amdgpu_buffer_alloca_plan_t {
  // Exact allocation byte length proven during planning.
  int64_t byte_length;
  // Power-of-two allocation byte alignment proven during planning.
  int64_t base_alignment;
  // Low storage space reserved for the source allocation.
  loom_storage_space_t storage_space;
} loom_amdgpu_buffer_alloca_plan_t;

typedef enum loom_amdgpu_table_index_kind_e {
  LOOM_AMDGPU_TABLE_INDEX_KIND_NONE = 0,
  LOOM_AMDGPU_TABLE_INDEX_KIND_I32 = 1,
  LOOM_AMDGPU_TABLE_INDEX_KIND_PACKED_I8 = 2,
} loom_amdgpu_table_index_kind_t;

typedef enum loom_amdgpu_table_lookup_strategy_e {
  LOOM_AMDGPU_TABLE_LOOKUP_STRATEGY_NONE = 0,
  LOOM_AMDGPU_TABLE_LOOKUP_STRATEGY_F32_LADDER = 1,
  LOOM_AMDGPU_TABLE_LOOKUP_STRATEGY_PACKED_I8_PERMUTE = 2,
  LOOM_AMDGPU_TABLE_LOOKUP_STRATEGY_PACKED_I8_U4_PERMUTE = 3,
} loom_amdgpu_table_lookup_strategy_t;

typedef struct loom_amdgpu_table_lookup_plan_t {
  // Register table value selected by each index lane.
  loom_value_id_t table;
  // Index vector selecting table lanes.
  loom_value_id_t indices;
  // Result vector receiving selected table lanes.
  loom_value_id_t result;
  // Selected lowering strategy.
  loom_amdgpu_table_lookup_strategy_t strategy;
  // Descriptor row selected for index-lane equality comparisons.
  loom_low_lower_resolved_descriptor_t compare_register_descriptor;
  // Optional descriptor row selected when the compare rhs ordinal is inline.
  loom_low_lower_resolved_descriptor_t compare_src1_inline_descriptor;
  // Descriptor row selected for register-register table lane selects.
  loom_low_lower_resolved_descriptor_t select_register_descriptor;
  // Optional descriptor row selected when the true table lane is a literal.
  loom_low_lower_resolved_descriptor_t select_src1_literal_descriptor;
  // Descriptor row selected for packed byte table permutation.
  loom_low_lower_resolved_descriptor_t permute_descriptor;
  // Selected index payload representation.
  loom_amdgpu_table_index_kind_t index_kind;
  // Static number of table lanes.
  uint32_t table_lane_count;
  // Number of 32-bit registers occupied by the table vector.
  uint32_t table_register_count;
  // Static number of result lanes.
  uint32_t result_lane_count;
  // Number of 32-bit registers occupied by the index vector.
  uint32_t index_register_count;
} loom_amdgpu_table_lookup_plan_t;

typedef struct loom_amdgpu_vector_compare_plan_t {
  // Left-hand payload vector value.
  loom_value_id_t lhs;
  // Right-hand payload vector value.
  loom_value_id_t rhs;
  // Descriptor row selected for the compare predicate.
  loom_low_lower_resolved_descriptor_t descriptor;
  // Optional descriptor row selected when the left-hand lane is inline.
  loom_low_lower_resolved_descriptor_t src0_inline_descriptor;
  // Optional descriptor row selected when the right-hand lane is inline.
  loom_low_lower_resolved_descriptor_t src1_inline_descriptor;
  // Result mask vector value.
  loom_value_id_t result;
  // Static number of payload and mask lanes compared.
  uint32_t lane_count;
} loom_amdgpu_vector_compare_plan_t;

typedef uint32_t loom_amdgpu_cndmask_b32_descriptor_flags_t;

enum {
  LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_REGISTER = 1u << 0,
  LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_SRC0_INLINE = 1u << 1,
  LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_SRC1_INLINE = 1u << 2,
  LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_SRC0_LITERAL = 1u << 3,
  LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_SRC1_LITERAL = 1u << 4,
  LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_SRC0_LITERAL_SRC1_INLINE = 1u << 5,
  LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_SRC1_LITERAL_SRC0_INLINE = 1u << 6,
  LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_ALL =
      LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_REGISTER |
      LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_SRC0_INLINE |
      LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_SRC1_INLINE |
      LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_SRC0_LITERAL |
      LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_SRC1_LITERAL |
      LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_SRC0_LITERAL_SRC1_INLINE |
      LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_SRC1_LITERAL_SRC0_INLINE,
};

typedef struct loom_amdgpu_cndmask_b32_descriptors_t {
  // Descriptor row selected for register-register lane selects.
  loom_low_lower_resolved_descriptor_t register_descriptor;
  // Optional descriptor row selected when the false lane is an inline source.
  loom_low_lower_resolved_descriptor_t src0_inline_descriptor;
  // Optional descriptor row selected when the true lane is an inline source.
  loom_low_lower_resolved_descriptor_t src1_inline_descriptor;
  // Optional descriptor row selected when the false lane is a literal source.
  loom_low_lower_resolved_descriptor_t src0_literal_descriptor;
  // Optional descriptor row selected when the true lane is a literal source.
  loom_low_lower_resolved_descriptor_t src1_literal_descriptor;
  // Optional descriptor row selected when false is literal and true is inline.
  loom_low_lower_resolved_descriptor_t src0_literal_src1_inline_descriptor;
  // Optional descriptor row selected when true is literal and false is inline.
  loom_low_lower_resolved_descriptor_t src1_literal_src0_inline_descriptor;
} loom_amdgpu_cndmask_b32_descriptors_t;

typedef enum loom_amdgpu_select_condition_kind_e {
  LOOM_AMDGPU_SELECT_CONDITION_KIND_NONE = 0,
  LOOM_AMDGPU_SELECT_CONDITION_KIND_SCC = 1,
  LOOM_AMDGPU_SELECT_CONDITION_KIND_SCALAR_MASK = 2,
  LOOM_AMDGPU_SELECT_CONDITION_KIND_VECTOR_MASK = 3,
  LOOM_AMDGPU_SELECT_CONDITION_KIND_SGPR_BOOL = 4,
} loom_amdgpu_select_condition_kind_t;

typedef enum loom_amdgpu_select_payload_kind_e {
  LOOM_AMDGPU_SELECT_PAYLOAD_KIND_NONE = 0,
  LOOM_AMDGPU_SELECT_PAYLOAD_KIND_DATA = 1,
  LOOM_AMDGPU_SELECT_PAYLOAD_KIND_I1_MASK = 2,
} loom_amdgpu_select_payload_kind_t;

typedef struct loom_amdgpu_vector_select_plan_t {
  // Source condition selecting true lanes.
  loom_value_id_t condition;
  // Source vector used when the corresponding condition lane is true.
  loom_value_id_t true_value;
  // Source vector used when the corresponding condition lane is false.
  loom_value_id_t false_value;
  // Selected representation of the true/false/result payload.
  loom_amdgpu_select_payload_kind_t payload_kind;
  // Selected representation of the scalar or vector condition.
  loom_amdgpu_select_condition_kind_t condition_kind;
  // Descriptor row selected for SCC-controlled scalar selects.
  loom_low_lower_resolved_descriptor_t scc_descriptor;
  // Descriptor row rematerializing SCC from an SGPR boolean condition.
  loom_low_lower_resolved_descriptor_t sgpr_bool_compare_descriptor;
  // Descriptor rows selected for scalar-mask v_cndmask_b32 lane selects.
  loom_amdgpu_cndmask_b32_descriptors_t cndmask_descriptors;
  // Descriptor row selected to read EXEC for i1 mask selection.
  loom_low_lower_resolved_descriptor_t mask_exec_read_descriptor;
  // Descriptor row selected to AND i1 mask payloads.
  loom_low_lower_resolved_descriptor_t mask_and_descriptor;
  // Descriptor row selected to OR i1 mask payloads.
  loom_low_lower_resolved_descriptor_t mask_or_descriptor;
  // Descriptor row selected to XOR i1 mask payloads.
  loom_low_lower_resolved_descriptor_t mask_xor_descriptor;
  // Result vector value.
  loom_value_id_t result;
  // Static number of selected 32-bit register units.
  uint32_t lane_count;
  // Number of selected register units controlled by one vector mask lane.
  uint32_t registers_per_condition_lane;
  // True when cndmask literal/inline operand forms can be selected per lane.
  bool allow_lane_immediates;
} loom_amdgpu_vector_select_plan_t;

typedef enum loom_amdgpu_clampf_mode_e {
  LOOM_AMDGPU_CLAMPF_MODE_NONE = 0,
  LOOM_AMDGPU_CLAMPF_MODE_ORDERED = 1,
  LOOM_AMDGPU_CLAMPF_MODE_NUMBER = 2,
} loom_amdgpu_clampf_mode_t;

typedef struct loom_amdgpu_clampf_plan_t {
  // Source payload being clamped.
  loom_value_id_t value;
  // Source lower bound.
  loom_value_id_t lower;
  // Source upper bound.
  loom_value_id_t upper;
  // Selected clamp semantics with native AMDGPU packet support.
  loom_amdgpu_clampf_mode_t mode;
  // Descriptor row selected for the ordered lower-bound comparison.
  loom_low_lower_resolved_descriptor_t lower_compare_descriptor;
  // Descriptor row selected for the ordered upper-bound comparison.
  loom_low_lower_resolved_descriptor_t upper_compare_descriptor;
  // Descriptor rows selected for ordered-mode v_cndmask_b32 lane selects.
  loom_amdgpu_cndmask_b32_descriptors_t select_descriptors;
  // Descriptor row selected for register-register lower-bound maxnum.
  loom_low_lower_resolved_descriptor_t lower_bound_register_descriptor;
  // Optional descriptor row selected for literal lower-bound maxnum.
  loom_low_lower_resolved_descriptor_t lower_bound_literal_descriptor;
  // Descriptor row selected for register-register upper-bound minnum.
  loom_low_lower_resolved_descriptor_t upper_bound_register_descriptor;
  // Optional descriptor row selected for literal upper-bound minnum.
  loom_low_lower_resolved_descriptor_t upper_bound_literal_descriptor;
  // Result value.
  loom_value_id_t result;
  // Static number of f32 lanes lowered.
  uint32_t lane_count;
} loom_amdgpu_clampf_plan_t;

typedef enum loom_amdgpu_subgroup_payload_kind_e {
  LOOM_AMDGPU_SUBGROUP_PAYLOAD_NONE = 0,
  LOOM_AMDGPU_SUBGROUP_PAYLOAD_I32_SCALAR = 1,
  LOOM_AMDGPU_SUBGROUP_PAYLOAD_F32_SCALAR = 2,
  LOOM_AMDGPU_SUBGROUP_PAYLOAD_I32_VECTOR = 3,
  LOOM_AMDGPU_SUBGROUP_PAYLOAD_F32_VECTOR = 4,
} loom_amdgpu_subgroup_payload_kind_t;

typedef enum loom_amdgpu_subgroup_broadcast_strategy_e {
  // Read the named lane directly through the target's native subgroup crossbar.
  LOOM_AMDGPU_SUBGROUP_BROADCAST_STRATEGY_BPERMUTE = 0,
  // Read one statically named lane into an SGPR and publish it to every lane.
  LOOM_AMDGPU_SUBGROUP_BROADCAST_STRATEGY_SCALAR_READLANE = 1,
} loom_amdgpu_subgroup_broadcast_strategy_t;

typedef struct loom_amdgpu_subgroup_broadcast_plan_t {
  // Descriptor row selected for the strategy's native lane read.
  loom_low_lower_resolved_descriptor_t exchange_descriptor;
  // Descriptor row publishing an SGPR read result back to VGPRs.
  loom_low_lower_resolved_descriptor_t scalar_copy_descriptor;
  // Source value broadcast from source_lane.
  loom_value_id_t value;
  // Result value receiving the broadcast payload.
  loom_value_id_t result;
  // Subgroup lane SSA value read by the broadcast.
  loom_value_id_t source_lane;
  // Exact source lane when known during planning, or UINT32_MAX when dynamic.
  uint32_t exact_source_lane;
  // Source/result payload shape selected during planning.
  loom_amdgpu_subgroup_payload_kind_t payload_kind;
  // Number of 32-bit registers in the broadcast payload.
  uint32_t register_count;
  // Native exchange and publication strategy selected during planning.
  loom_amdgpu_subgroup_broadcast_strategy_t strategy;
} loom_amdgpu_subgroup_broadcast_plan_t;

typedef struct loom_amdgpu_subgroup_broadcast_first_plan_t {
  // Source value broadcast from the first active subgroup lane.
  loom_value_id_t value;
  // Descriptor row selected to read the first active lane into an SGPR.
  loom_low_lower_resolved_descriptor_t descriptor;
  // Result value receiving the broadcast payload.
  loom_value_id_t result;
  // Source/result payload shape selected during planning.
  loom_amdgpu_subgroup_payload_kind_t payload_kind;
  // Number of 32-bit registers in the broadcast payload.
  uint32_t register_count;
} loom_amdgpu_subgroup_broadcast_first_plan_t;

typedef enum loom_amdgpu_crosslane_kind_e {
  // Use DS bpermute with a byte-addressed source lane.
  LOOM_AMDGPU_CROSSLANE_BPERMUTE = 0,
  // Use DPP row-lane moves with an immediate control value.
  LOOM_AMDGPU_CROSSLANE_DPP = 1,
  // Use DS swizzle with a bitmask permutation immediate.
  LOOM_AMDGPU_CROSSLANE_SWIZZLE = 2,
} loom_amdgpu_crosslane_kind_t;

typedef struct loom_amdgpu_direct_xor_lane_recipe_t {
  // Descriptor row implementing the lane exchange.
  loom_amdgpu_descriptor_ref_t descriptor_ref;
  // Optional descriptor fusing the exchange into a conditional false operand.
  loom_amdgpu_descriptor_ref_t conditional_ref;
  // Optional tied descriptor updating selected DPP destination banks.
  loom_amdgpu_descriptor_ref_t masked_move_ref;
  // DPP control or DS swizzle offset interpreted by |crosslane_kind|.
  uint32_t immediate;
  // Native packet family selected for the lane exchange.
  loom_amdgpu_crosslane_kind_t crosslane_kind;
} loom_amdgpu_direct_xor_lane_recipe_t;

typedef struct loom_amdgpu_subgroup_shuffle_plan_t {
  // Source value moved across subgroup lanes.
  loom_value_id_t value;
  // Dynamic lane offset interpreted by |mode| when |exact_offset| is absent.
  loom_value_id_t source_offset;
  // Descriptor row selected for the native cross-lane read.
  loom_low_lower_resolved_descriptor_t descriptor;
  // Result value receiving the moved payload.
  loom_value_id_t result;
  // Per-lane mask reporting whether the selected source lane is valid.
  loom_value_id_t valid;
  // Source/result payload shape selected during planning.
  loom_amdgpu_subgroup_payload_kind_t payload_kind;
  // Number of 32-bit registers in the shuffled payload.
  uint32_t register_count;
  // Cross-lane packet family selected for the shuffle.
  loom_amdgpu_crosslane_kind_t crosslane_kind;
  // Full-width lane addressing mode selected by the source op.
  loom_kernel_subgroup_shuffle_mode_t mode;
  // DPP control or DS swizzle offset used by direct cross-lane packets.
  uint32_t crosslane_immediate;
  // Exact lane offset or lane index, or UINT32_MAX when dynamic.
  uint32_t exact_offset;
  // Exact shuffle segment width from the source op.
  uint32_t width;
  // Exact subgroup width selected by the active target bundle.
  uint32_t wavefront_size;
} loom_amdgpu_subgroup_shuffle_plan_t;

typedef enum loom_amdgpu_subgroup_reduce_crosslane_kind_e {
  // Use DS bpermute for every subgroup tree exchange.
  LOOM_AMDGPU_SUBGROUP_REDUCE_CROSSLANE_BPERMUTE = 0,
  // Use DPP row moves within 16-lane rows and DS bpermute between rows.
  LOOM_AMDGPU_SUBGROUP_REDUCE_CROSSLANE_DPP_ROW_BPERMUTE = 1,
  // Use DPP row moves within 16-lane rows and permlanex16 between row pairs.
  LOOM_AMDGPU_SUBGROUP_REDUCE_CROSSLANE_DPP_ROW_PERMLANEX16 = 2,
} loom_amdgpu_subgroup_reduce_crosslane_kind_t;

typedef enum loom_amdgpu_subgroup_reduce_publication_kind_e {
  // Publish a correct reduced VGPR payload to every active subgroup lane.
  LOOM_AMDGPU_SUBGROUP_REDUCE_PUBLICATION_ALL_LANES = 0,
  // Combine 32-lane halves through SGPRs and broadcast the completed payload.
  LOOM_AMDGPU_SUBGROUP_REDUCE_PUBLICATION_SCALAR_BROADCAST = 1,
} loom_amdgpu_subgroup_reduce_publication_kind_t;

typedef struct loom_amdgpu_subgroup_reduce_plan_t {
  // Source value reduced across subgroup lanes.
  loom_value_id_t value;
  // Descriptor row selected for each native cross-lane read.
  loom_low_lower_resolved_descriptor_t bpermute_descriptor;
  // Descriptor row selected for all-lane DPP row moves.
  loom_low_lower_resolved_descriptor_t dpp_descriptor;
  // Descriptor row selected for fused DPP row moves and lane combines.
  loom_low_lower_resolved_descriptor_t dpp_combine_descriptor;
  // Descriptor row selected for paired 16-lane row exchanges.
  loom_low_lower_resolved_descriptor_t permlanex16_descriptor;
  // Descriptor row selected for reading a fixed VGPR lane into an SGPR.
  loom_low_lower_resolved_descriptor_t readlane_descriptor;
  // Descriptor row selected for each native lane combine.
  loom_low_lower_resolved_descriptor_t combine_descriptor;
  // Descriptor row selected to guard inactive source lanes.
  loom_low_lower_resolved_descriptor_t guard_descriptor;
  // Descriptor row selected to replace inactive source lanes with identity.
  loom_low_lower_resolved_descriptor_t select_descriptor;
  // Result value receiving the reduced payload.
  loom_value_id_t result;
  // Source/result payload shape selected during planning.
  loom_amdgpu_subgroup_payload_kind_t payload_kind;
  // Number of 32-bit registers in the reduced payload.
  uint32_t register_count;
  // Exact subgroup width selected by the active target bundle.
  uint32_t wavefront_size;
  // Number of low-numbered lanes participating in the emitted reduce tree.
  uint32_t active_lane_count;
  // 32-bit identity element bit pattern used for inactive source lanes.
  uint32_t identity_bits;
  // Cross-lane exchange strategy selected for the subgroup tree.
  loom_amdgpu_subgroup_reduce_crosslane_kind_t crosslane_kind;
  // Strategy used to publish the final reduced value to result users.
  loom_amdgpu_subgroup_reduce_publication_kind_t publication_kind;
} loom_amdgpu_subgroup_reduce_plan_t;

typedef enum loom_amdgpu_workgroup_reduce_publication_kind_e {
  // One wave reduces LDS-published per-wave partials and publishes the final
  // value back through LDS for all workitems to reload.
  LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_LDS = 0,
  // Every wave reloads the per-wave partials, redundantly reduces them within
  // the wave, and broadcasts the wave-local lane-zero result to its lanes.
  LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_REDUNDANT_SUBGROUP = 1,
  // One wave reduces LDS-published per-wave partials and leaves the result
  // valid only for leader-guarded uses.
  LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_LEADER_WORKITEM = 2,
  // Every wave redundantly reduces LDS-published per-wave partials and leaves
  // the result valid only in subgroup leader lanes.
  LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_REDUNDANT_SUBGROUP_LEADER_LANE = 3,
} loom_amdgpu_workgroup_reduce_publication_kind_t;

#define LOOM_AMDGPU_EXPLICIT_PACKET_IMMEDIATE_CAPACITY 4

typedef struct loom_amdgpu_explicit_packet_immediate_t {
  // Module string ID for the immediate field name.
  loom_string_id_t name_id;
  // Concrete immediate value emitted for the packet.
  uint16_t value;
} loom_amdgpu_explicit_packet_immediate_t;

typedef struct loom_amdgpu_explicit_packet_immediate_template_t {
  // Borrowed immediate field name resolved during packet planning.
  iree_string_view_t name;
  // Concrete immediate value emitted for the packet.
  uint16_t value;
} loom_amdgpu_explicit_packet_immediate_template_t;

typedef struct loom_amdgpu_explicit_packet_plan_t {
  // Descriptor row selected for the explicit packet.
  loom_low_lower_resolved_descriptor_t descriptor;
  // Immediate rows emitted on the descriptor.
  loom_amdgpu_explicit_packet_immediate_t
      immediates[LOOM_AMDGPU_EXPLICIT_PACKET_IMMEDIATE_CAPACITY];
  // Number of populated immediate rows.
  iree_host_size_t immediate_count;
} loom_amdgpu_explicit_packet_plan_t;

typedef enum loom_amdgpu_kernel_barrier_lowering_kind_e {
  // No lowering has been selected.
  LOOM_AMDGPU_KERNEL_BARRIER_LOWERING_KIND_NONE = 0,
  // Emit a full workgroup barrier packet.
  LOOM_AMDGPU_KERNEL_BARRIER_LOWERING_KIND_S_BARRIER = 1,
  // Emit a wait packet that drains LDS effects for a single-wave workgroup.
  LOOM_AMDGPU_KERNEL_BARRIER_LOWERING_KIND_LDS_WAIT = 2,
  // Emit the split signal/wait barrier packet pair used by GFX12+ targets.
  LOOM_AMDGPU_KERNEL_BARRIER_LOWERING_KIND_SPLIT_BARRIER = 3,
} loom_amdgpu_kernel_barrier_lowering_kind_t;

typedef struct loom_amdgpu_kernel_barrier_plan_t {
  // Concrete synchronization packet path selected for kernel.barrier.
  loom_amdgpu_kernel_barrier_lowering_kind_t kind;
  // Explicit wait packet selected when |kind| is LDS_WAIT.
  loom_amdgpu_explicit_packet_plan_t wait;
  // Explicit signal packet selected when |kind| is SPLIT_BARRIER.
  loom_amdgpu_explicit_packet_plan_t split_signal;
  // Explicit wait packet selected when |kind| is SPLIT_BARRIER.
  loom_amdgpu_explicit_packet_plan_t split_wait;
} loom_amdgpu_kernel_barrier_plan_t;

typedef struct loom_amdgpu_workgroup_collective_cross_wave_descriptors_t {
  // Descriptor row selected for LDS reads between waves.
  loom_low_lower_resolved_descriptor_t lds_read_descriptor;
  // Descriptor row selected for LDS writes between waves.
  loom_low_lower_resolved_descriptor_t lds_write_descriptor;
  // Target-selected packet plan used to synchronize LDS publication.
  loom_amdgpu_kernel_barrier_plan_t barrier;
  // Descriptor row selected to restrict publication to producer lanes.
  loom_low_lower_resolved_descriptor_t saveexec_descriptor;
  // Descriptor row selected to restore EXEC after lane-restricted regions.
  loom_low_lower_resolved_descriptor_t restore_exec_descriptor;
} loom_amdgpu_workgroup_collective_cross_wave_descriptors_t;

typedef struct loom_amdgpu_workgroup_reduce_plan_t {
  // Source value reduced across workgroup lanes.
  loom_value_id_t value;
  // Descriptor row selected for each native cross-lane read.
  loom_low_lower_resolved_descriptor_t bpermute_descriptor;
  // Descriptor row selected for all-lane DPP row moves.
  loom_low_lower_resolved_descriptor_t dpp_descriptor;
  // Descriptor row selected for fused DPP row moves and lane combines.
  loom_low_lower_resolved_descriptor_t dpp_combine_descriptor;
  // Descriptor row selected for paired 16-lane row exchanges.
  loom_low_lower_resolved_descriptor_t permlanex16_descriptor;
  // Descriptor row selected for each native lane combine.
  loom_low_lower_resolved_descriptor_t combine_descriptor;
  // Descriptor row selected to guard inactive source lanes.
  loom_low_lower_resolved_descriptor_t guard_descriptor;
  // Descriptor row selected to identify the partial tail wave.
  loom_low_lower_resolved_descriptor_t lane_ge_descriptor;
  // Descriptor row selected to replace inactive source lanes with identity.
  loom_low_lower_resolved_descriptor_t select_descriptor;
  // Descriptor bundle used for cross-wave LDS publication.
  loom_amdgpu_workgroup_collective_cross_wave_descriptors_t cross_wave;
  // Result value receiving the reduced payload.
  loom_value_id_t result;
  // Source/result payload shape selected during planning.
  loom_amdgpu_subgroup_payload_kind_t payload_kind;
  // Number of 32-bit registers in the reduced payload.
  uint32_t register_count;
  // Exact subgroup width selected by the active target bundle.
  uint32_t wavefront_size;
  // Execution wavefront width used to partition cross-wave workgroup staging.
  uint32_t partition_wavefront_size;
  // Exact flattened workgroup size selected by launch configuration.
  uint32_t flat_workgroup_size;
  // 32-bit identity element bit pattern used for inactive source lanes.
  uint32_t identity_bits;
  // Cross-lane exchange strategy selected for full-wave subgroup trees.
  loom_amdgpu_subgroup_reduce_crosslane_kind_t crosslane_kind;
  // Strategy used to publish the final reduced value to all workitems.
  loom_amdgpu_workgroup_reduce_publication_kind_t publication_kind;
} loom_amdgpu_workgroup_reduce_plan_t;

typedef struct loom_amdgpu_subgroup_scan_plan_t {
  // Source value scanned across subgroup lanes.
  loom_value_id_t value;
  // Descriptor row selected for each native cross-lane read.
  loom_low_lower_resolved_descriptor_t bpermute_descriptor;
  // Descriptor row selected for each native lane combine.
  loom_low_lower_resolved_descriptor_t combine_descriptor;
  // Descriptor row selected to guard each prefix step.
  loom_low_lower_resolved_descriptor_t guard_descriptor;
  // Descriptor row selected to merge guarded prefix-step results.
  loom_low_lower_resolved_descriptor_t select_descriptor;
  // Result value receiving the scanned payload.
  loom_value_id_t result;
  // Source/result payload shape selected during planning.
  loom_amdgpu_subgroup_payload_kind_t payload_kind;
  // Number of 32-bit registers in the scanned payload.
  uint32_t register_count;
  // Inclusive or exclusive scan mode selected by the source op.
  loom_kernel_subgroup_scan_mode_t mode;
  // Lane order selected by the source op.
  loom_kernel_subgroup_scan_direction_t direction;
  // 32-bit identity element bit pattern used by exclusive scans.
  uint32_t identity_bits;
  // Exact subgroup width selected by the active target bundle.
  uint32_t wavefront_size;
  // Number of low-numbered lanes participating in the emitted scan tree.
  uint32_t active_lane_count;
} loom_amdgpu_subgroup_scan_plan_t;

typedef struct loom_amdgpu_workgroup_scan_plan_t {
  // Source value scanned across workgroup lanes.
  loom_value_id_t value;
  // Descriptor row selected for each native cross-lane read.
  loom_low_lower_resolved_descriptor_t bpermute_descriptor;
  // Descriptor row selected for each native lane combine.
  loom_low_lower_resolved_descriptor_t combine_descriptor;
  // Descriptor row selected to guard each prefix step.
  loom_low_lower_resolved_descriptor_t guard_descriptor;
  // Descriptor row selected to merge guarded prefix-step results.
  loom_low_lower_resolved_descriptor_t select_descriptor;
  // Descriptor row selected for first-wave predicates.
  loom_low_lower_resolved_descriptor_t lane_lt_descriptor;
  // Descriptor row selected for tail-wave predicates.
  loom_low_lower_resolved_descriptor_t lane_ge_descriptor;
  // Descriptor bundle used for cross-wave LDS publication.
  loom_amdgpu_workgroup_collective_cross_wave_descriptors_t cross_wave;
  // Result value receiving the scanned payload.
  loom_value_id_t result;
  // Source/result payload shape selected during planning.
  loom_amdgpu_subgroup_payload_kind_t payload_kind;
  // Number of 32-bit registers in the scanned payload.
  uint32_t register_count;
  // Inclusive or exclusive scan mode selected by the source op.
  loom_kernel_subgroup_scan_mode_t mode;
  // Lane order selected by the source op.
  loom_kernel_subgroup_scan_direction_t direction;
  // 32-bit identity element bit pattern used by exclusive or cross-wave scans.
  uint32_t identity_bits;
  // Native cross-lane width used to partition workgroup scan staging.
  uint32_t partition_wavefront_size;
  // Exact flattened workgroup size selected by launch configuration.
  uint32_t flat_workgroup_size;
} loom_amdgpu_workgroup_scan_plan_t;

typedef struct loom_amdgpu_subgroup_active_mask_plan_t {
  // Descriptor row selected to read the native EXEC lane mask.
  loom_low_lower_resolved_descriptor_t exec_read_descriptor;
  // Source mask result receiving the active-lane payload.
  loom_value_id_t mask;
  // Static bit width of the source integer mask result.
  uint32_t mask_bit_count;
  // Exact subgroup width selected by the active target bundle.
  uint32_t wavefront_size;
} loom_amdgpu_subgroup_active_mask_plan_t;

typedef struct loom_amdgpu_subgroup_ballot_plan_t {
  // Source predicate already materialized as a native EXEC-width mask.
  loom_value_id_t predicate;
  // Source mask result receiving predicate bits for active lanes.
  loom_value_id_t mask;
  // Static bit width of the source integer mask result.
  uint32_t mask_bit_count;
  // Exact subgroup width selected by the active target bundle.
  uint32_t wavefront_size;
} loom_amdgpu_subgroup_ballot_plan_t;

typedef struct loom_amdgpu_subgroup_vote_any_plan_t {
  // Source predicate already materialized as a native EXEC-width mask.
  loom_value_id_t predicate;
  // Descriptor row selected to compare the predicate mask against zero.
  loom_low_lower_resolved_descriptor_t compare_descriptor;
  // Descriptor row selected to materialize each half of the zero mask.
  loom_low_lower_resolved_descriptor_t zero_descriptor;
  // Subgroup-uniform i1 source result receiving SCC.
  loom_value_id_t result;
  // Exact subgroup width selected by the active target bundle.
  uint32_t wavefront_size;
} loom_amdgpu_subgroup_vote_any_plan_t;

typedef struct loom_amdgpu_subgroup_vote_all_plan_t {
  // Source predicate already materialized as a native EXEC-width mask.
  loom_value_id_t predicate;
  // Descriptor row selected to compare predicate and active EXEC masks.
  loom_low_lower_resolved_descriptor_t compare_descriptor;
  // Descriptor row selected to read the native EXEC lane mask.
  loom_low_lower_resolved_descriptor_t exec_read_descriptor;
  // Subgroup-uniform i1 source result receiving SCC.
  loom_value_id_t result;
  // Exact subgroup width selected by the active target bundle.
  uint32_t wavefront_size;
} loom_amdgpu_subgroup_vote_all_plan_t;

typedef enum loom_amdgpu_vector_slice_kind_e {
  LOOM_AMDGPU_VECTOR_SLICE_KIND_NONE = 0,
  LOOM_AMDGPU_VECTOR_SLICE_KIND_32BIT_LANES = 1,
  LOOM_AMDGPU_VECTOR_SLICE_KIND_PACKED_REGISTER_BITS = 2,
} loom_amdgpu_vector_slice_kind_t;

typedef struct loom_amdgpu_vector_slice_plan_t {
  // Source vector value being sliced.
  loom_value_id_t source;
  // Result vector value produced by the slice.
  loom_value_id_t result;
  // Selected lowering strategy for the source/result storage.
  loom_amdgpu_vector_slice_kind_t kind;
  // Static source lane offset.
  uint32_t lane_offset;
  // Static result lane count.
  uint32_t lane_count;
  // Source 32-bit backing register count.
  uint32_t source_register_count;
  // Result 32-bit backing register count.
  uint32_t result_register_count;
  // Source element bit count for packed register-bit slices.
  uint32_t element_bit_count;
} loom_amdgpu_vector_slice_plan_t;

#define LOOM_AMDGPU_MEMORY_ACCESS_BYTE_SHIFT_NONE \
  LOOM_LOW_SOURCE_MEMORY_ACCESS_BYTE_SHIFT_NONE

typedef enum loom_amdgpu_memory_dynamic_index_kind_e {
  LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_NONE = 0,
  LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR = 1,
  LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET = 2,
} loom_amdgpu_memory_dynamic_index_kind_t;

typedef enum loom_amdgpu_memory_payload_register_class_e {
  LOOM_AMDGPU_MEMORY_PAYLOAD_REGISTER_CLASS_VGPR = 0,
  LOOM_AMDGPU_MEMORY_PAYLOAD_REGISTER_CLASS_SGPR = 1,
  LOOM_AMDGPU_MEMORY_PAYLOAD_REGISTER_CLASS_COUNT_,
} loom_amdgpu_memory_payload_register_class_t;

typedef enum loom_amdgpu_memory_payload_format_e {
  LOOM_AMDGPU_MEMORY_PAYLOAD_FORMAT_GENERIC = 0,
  LOOM_AMDGPU_MEMORY_PAYLOAD_FORMAT_LOW_16BIT_FLOAT = 1,
  LOOM_AMDGPU_MEMORY_PAYLOAD_FORMAT_SIGNED_16BIT_INTEGER = 2,
  LOOM_AMDGPU_MEMORY_PAYLOAD_FORMAT_COUNT_,
} loom_amdgpu_memory_payload_format_t;

typedef enum loom_amdgpu_memory_scalar_offset_placement_e {
  LOOM_AMDGPU_MEMORY_SCALAR_OFFSET_PLACEMENT_SOFFSET = 0,
  LOOM_AMDGPU_MEMORY_SCALAR_OFFSET_PLACEMENT_BASE = 1,
} loom_amdgpu_memory_scalar_offset_placement_t;

typedef struct loom_amdgpu_memory_access_t {
  // Target-independent source memory access plan being wrapped.
  loom_low_source_memory_access_plan_t source;
  // Selected target addressing form for the memory packet.
  loom_amdgpu_memory_address_form_t address_form;
  // Target operand path selected for each source dynamic address term.
  loom_amdgpu_memory_dynamic_index_kind_t
      dynamic_term_kinds[LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_CAPACITY];
  // Static offset value encoded in the descriptor's first offset immediate.
  int64_t immediate_offset;
  // Static offset value encoded in the descriptor's second offset immediate.
  int64_t secondary_immediate_offset;
  // Static byte offset materialized through the VGPR VADDR operand.
  uint64_t vaddr_static_byte_offset;
  // Static byte offset materialized through the scalar SOFFSET operand.
  uint32_t scalar_byte_offset;
  // Static byte offset folded into the scalar base pointer.
  uint64_t scalar_base_byte_offset;
  // Location selected for scalar dynamic and static address terms.
  loom_amdgpu_memory_scalar_offset_placement_t scalar_offset_placement;
  // Register file selected for the memory packet payload.
  loom_amdgpu_memory_payload_register_class_t payload_register_class;
  // Semantic payload family used to choose same-footprint memory descriptors.
  loom_amdgpu_memory_payload_format_t payload_format;
  // Number of 32-bit registers moved by the selected memory packet payload.
  uint32_t payload_register_count;
  // Number of bytes moved by the selected memory packet.
  uint32_t packet_byte_count;
  // Descriptor row selected for the active descriptor set.
  const loom_low_descriptor_t* descriptor;
} loom_amdgpu_memory_access_t;

typedef struct loom_amdgpu_memory_packet_plan_t {
  // Selected access form for this emitted direct memory packet.
  loom_amdgpu_memory_access_t access;
  // First 32-bit source register moved by this packet.
  uint32_t source_register_offset;
} loom_amdgpu_memory_packet_plan_t;

// Immutable function-retained direct-memory packet plan.
typedef struct loom_amdgpu_memory_access_plan_t {
  // Number of populated packet plans.
  uint32_t packet_count;
  // Direct memory packets emitted in increasing source-register order. The
  // function-retained allocation contains exactly |packet_count| entries.
  loom_amdgpu_memory_packet_plan_t packets[];
} loom_amdgpu_memory_access_plan_t;

typedef enum loom_amdgpu_fragment_memory_packet_flag_bits_e {
  // Adjacent-lane f32 result values are packed into one BF16 store packet.
  LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_CROSSLANE_PACKED_B16_STORE = 1u << 0,
  // Adjacent-lane f32 result values are exchanged with a DPP packet.
  LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_CROSSLANE_PACKED_B16_STORE_DPP = 1u
                                                                           << 1,
  // Same-lane f32 result values are packed into one BF16 store packet.
  LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_PACKED_B16_STORE = 1u << 19,
  // Packed FP8-to-BF16 decode uses exact F16 arithmetic for subnormals.
  LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_EXACT_BF16_VIA_F16 = 1u << 20,
  // FP8 load payloads are decoded with native packed FP8-to-F32 conversion.
  LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_NATIVE_F32_PAIR = 1u << 2,
  // FP8 load payloads use native scale-f32 BF16 conversion.
  LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_SCALEF32_BF16_PAIR = 1u << 12,
  // FP8 load payloads use native scale-f32 F16 conversion.
  LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_SCALEF32_F16_PAIR = 1u << 14,
  // FP8 load payloads are decoded with native E8M0 scale-pk8 BF16 conversion
  // using a packed identity scale operand.
  LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_IDENTITY_E8M0_PK8_BF16 = 1u << 17,
  // FP8 load payloads are decoded with native E8M0 scale-pk8 F16 conversion
  // using a packed identity scale operand.
  LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_IDENTITY_E8M0_PK8_F16 = 1u << 18,
  // Native FP8-to-F32 conversion feeds native F32-to-BF16 packing.
  LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_NATIVE_BF16_PACK = 1u << 13,
  // FP8 load payloads are decoded with native packed FP8-to-F16 conversion.
  LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_NATIVE_F16_PAIR = 1u << 15,
  // FP8 load payloads are decoded with the finite packed-BF16 software path.
  LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_PACKED_BF16_DECODE = 1u << 3,
  // FP8 load payloads are decoded with the finite packed-F16 software path.
  LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_PACKED_F16_DECODE = 1u << 16,
  // FP8 load payloads require full per-lane BF16 software decode.
  LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_FULL_BF16_DECODE = 1u << 4,
  // Full FP8 decode was selected because value facts do not prove finiteness.
  LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_MISSING_VALUE_FINITE = 1u << 5,
  // Full FP8 decode was selected because value facts do not prove non-subnormal
  // values.
  LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_MISSING_VALUE_NOT_SUBNORMAL =
      1u << 6,
  // Full FP8 decode was selected because target packets are unavailable.
  LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_MISSING_TARGET_PACKETS = 1u << 7,
  // Packed FP8-to-16-bit decode repairs zero payloads after normal expansion.
  LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_REPAIR_ZERO = 1u << 8,
  // Packed FP8-to-16-bit decode repairs subnormal payloads with table packets.
  LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_REPAIR_SUBNORMAL = 1u << 9,
  // Packed FP8-to-BF16 decode repairs NaN payloads after expansion.
  LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_REPAIR_NAN = 1u << 10,
  // Packed FP8-to-BF16 decode repairs infinity payloads after expansion.
  LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_REPAIR_INF = 1u << 11,
} loom_amdgpu_fragment_memory_packet_flag_bits_t;

// Bitset of loom_amdgpu_fragment_memory_packet_flag_bits_t values.
typedef uint32_t loom_amdgpu_fragment_memory_packet_flags_t;

typedef struct loom_amdgpu_fragment_memory_packet_plan_t {
  // Packet-local lowering strategy bits for non-native memory payloads.
  loom_amdgpu_fragment_memory_packet_flags_t flags;
  // Descriptor row selected for this packet.
  loom_amdgpu_descriptor_ref_t descriptor_ref;
  // First target fragment coordinate register covered by this packet.
  uint16_t register_index;
  // Number of target fragment coordinate registers covered by this packet.
  uint16_t result_register_count;
  // Number of 32-bit memory packet registers moved by the descriptor.
  uint16_t packet_register_count;
} loom_amdgpu_fragment_memory_packet_plan_t;
static_assert(sizeof(loom_amdgpu_fragment_memory_packet_plan_t) == 12,
              "fragment memory packet plans must stay cache dense");

typedef enum loom_amdgpu_fragment_memory_payload_form_e {
  // Payload storage matches the selected fragment role layout.
  LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_NATIVE = 0,
  // A 16-bit float vector is loaded with f32 result-fragment coordinates.
  LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_LOAD_PACKED_16BIT_RESULT = 1,
  // FP8 memory lanes are converted to packed BF16 operand registers on load.
  LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_LOAD_FP8_TO_BF16 = 2,
  // FP8 memory lanes are converted to packed F16 operand registers on load.
  LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_LOAD_FP8_TO_F16 = 5,
  // A f32 result fragment is rounded to BF16 lanes before a 16-bit store.
  LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_NARROW_F32_TO_BF16 = 3,
  // A f16 low-subword result fragment is widened to f32 lanes before store.
  LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_EXTEND_F16_TO_F32 = 4,
} loom_amdgpu_fragment_memory_payload_form_t;

typedef enum loom_amdgpu_fragment_memory_packetization_e {
  // Fragment registers load or store through native 32-bit packet groups.
  LOOM_AMDGPU_FRAGMENT_MEMORY_PACKETIZATION_NATIVE = 0,
  // Each fragment register transfers one meaningful low 16-bit element.
  LOOM_AMDGPU_FRAGMENT_MEMORY_PACKETIZATION_SCALAR_B16 = 1,
  // Two independently addressed 16-bit elements are packed per register.
  LOOM_AMDGPU_FRAGMENT_MEMORY_PACKETIZATION_PACKED_B16 = 2,
} loom_amdgpu_fragment_memory_packetization_t;

typedef enum loom_amdgpu_fragment_memory_epilogue_strategy_e {
  // No special result-fragment store epilogue strategy is selected.
  LOOM_AMDGPU_FRAGMENT_MEMORY_EPILOGUE_STRATEGY_NONE = 0,
  // Each f32 result-fragment register is narrowed and stored separately.
  LOOM_AMDGPU_FRAGMENT_MEMORY_EPILOGUE_STRATEGY_SCALAR_B16_STORE = 1,
  // Same-lane adjacent f32 result-fragment registers are packed into b32
  // stores.
  LOOM_AMDGPU_FRAGMENT_MEMORY_EPILOGUE_STRATEGY_PACKED_B16_STORE = 2,
  // Adjacent-lane f32 result-fragment registers are exchanged with DS bpermute
  // and packed into b32 stores.
  LOOM_AMDGPU_FRAGMENT_MEMORY_EPILOGUE_STRATEGY_DS_PACKED_B16_STORE = 3,
  // Adjacent-lane f32 result-fragment registers are exchanged with DPP and
  // packed into b32 stores.
  LOOM_AMDGPU_FRAGMENT_MEMORY_EPILOGUE_STRATEGY_DPP_PACKED_B16_STORE = 4,
} loom_amdgpu_fragment_memory_epilogue_strategy_t;

typedef struct loom_amdgpu_fragment_memory_lane_term_t {
  // Power-of-two divisor applied to the subgroup lane ID.
  uint16_t divisor;
  // Optional power-of-two modulus applied after division; zero omits it.
  uint16_t modulus;
  // Byte stride multiplied by the resulting lane digit.
  uint32_t byte_stride;
} loom_amdgpu_fragment_memory_lane_term_t;

typedef struct loom_amdgpu_fragment_memory_address_layout_t {
  // Constant byte stride between adjacent lanes, or zero when non-linear.
  uint32_t linear_lane_byte_stride;
  // Byte stride between separately addressed elements in one register.
  uint32_t packed_element_byte_stride;
  // Logical payload elements stored in each 32-bit fragment register.
  uint16_t payload_elements_per_register;
  // 32-bit fragment registers occupied by each logical payload element.
  uint16_t payload_registers_per_element;
  // Preferred lane divisor reused by common result epilogue operations.
  uint16_t primary_lane_divisor;
  // Number of populated lane coordinate terms.
  uint8_t lane_term_count;
  // Lane coordinate terms compiled from semantic layout axes.
  loom_amdgpu_fragment_memory_lane_term_t
      lane_terms[LOOM_MATRIX_FRAGMENT_AXIS_COUNT];
  // Static byte offset of each physical fragment register.
  uint32_t
      register_byte_offsets[LOOM_AMDGPU_MAX_MATRIX_FRAGMENT_32BIT_REGISTERS];
} loom_amdgpu_fragment_memory_address_layout_t;

typedef struct loom_amdgpu_fragment_memory_plan_t {
  // Direction of the fragment memory movement.
  loom_low_source_memory_operation_kind_t operation_kind;
  // Contract operand role selected from source IR.
  loom_contract_operand_role_t role;
  // Target-owned lane/register layout selected for the fragment payload.
  loom_amdgpu_matrix_fragment_layout_kind_t layout_kind;
  // Target-independent source view access plan.
  loom_low_source_memory_access_plan_t source;
  // Source store payload or load result SSA value.
  loom_value_id_t payload;
  // Optional F32 scale applied while decoding an FP8 load payload.
  loom_value_id_t fp8_load_scale_source;
  // Per-axis byte strides selected from the view layout.
  uint32_t axis_byte_strides[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK];
  // Rank of the typed view.
  uint8_t view_rank;
  // Number of target fragment coordinate registers in the selected layout.
  uint16_t register_count;
  // Number of 32-bit registers used by the lowered payload storage value.
  uint16_t payload_register_count;
  // Byte count of one logical fragment element.
  uint16_t element_byte_count;
  // Element type stored in the source or destination view.
  loom_scalar_type_t view_element_type;
  // Exact numeric format stored in the source or destination view payload.
  loom_value_fact_numeric_format_flags_t view_element_format;
  // Semantically equivalent source format accepted by native descriptors.
  loom_value_fact_numeric_format_flags_t descriptor_source_format;
  // Compiled lane, register, and packed-element address coefficients.
  loom_amdgpu_fragment_memory_address_layout_t address_layout;
  // Direct memory packets emitted in increasing fragment-register order.
  loom_amdgpu_fragment_memory_packet_plan_t
      packets[LOOM_AMDGPU_MAX_MATRIX_FRAGMENT_32BIT_REGISTERS];
  // Number of populated packet plans.
  uint16_t packet_count;
  // High-half D16 load selected to complete packed B16 payload registers.
  loom_amdgpu_descriptor_ref_t packed_b16_high_descriptor_ref;
  // Aggregate packet-local lowering strategy bits across all packets.
  loom_amdgpu_fragment_memory_packet_flags_t packet_flags;
  // Payload storage form selected for the fragment movement.
  loom_amdgpu_fragment_memory_payload_form_t payload_form;
  // Memory packetization selected from fragment and physical view layouts.
  loom_amdgpu_fragment_memory_packetization_t packetization;
  // Result-fragment store epilogue strategy selected from layout and packet
  // facts.
  loom_amdgpu_fragment_memory_epilogue_strategy_t epilogue_strategy;
  // Optional f32 fragment source to round directly for narrowed stores.
  loom_value_id_t narrowed_result_round_source;
  // Optional scalar scale applied before narrowed f32-to-bf16 stores.
  loom_value_id_t narrowed_result_scale_source;
  // Optional packed bf16 fragment source copied directly for narrowed stores.
  loom_value_id_t narrowed_result_packed_source;
} loom_amdgpu_fragment_memory_plan_t;

typedef enum loom_amdgpu_fragment_repack_strategy_e {
  // No fragment repack strategy was selected.
  LOOM_AMDGPU_FRAGMENT_REPACK_STRATEGY_NONE = 0,
  // Source and result share the same physical fragment representation.
  LOOM_AMDGPU_FRAGMENT_REPACK_STRATEGY_ALIAS = 1,
  // Adjacent source lanes are packed before bpermute selection.
  LOOM_AMDGPU_FRAGMENT_REPACK_STRATEGY_RESULT_TO_LHS_BF16_PACKED_BPERMUTE = 2,
  // F32 result registers are permuted and packed into BF16 LHS registers.
  LOOM_AMDGPU_FRAGMENT_REPACK_STRATEGY_RESULT_TO_LHS_BF16_BPERMUTE = 3,
  // Packed source registers are partially transposed before reduced gathers.
  LOOM_AMDGPU_FRAGMENT_REPACK_STRATEGY_RESULT_TO_LHS_BF16_TRANSPOSE_BPERMUTE =
      4,
  // Source and result require a target strategy that is not implemented.
  LOOM_AMDGPU_FRAGMENT_REPACK_STRATEGY_DIAGNOSTIC = 5,
  // Packed B16 result rows are exchanged and repacked into RHS registers.
  LOOM_AMDGPU_FRAGMENT_REPACK_STRATEGY_RESULT_TO_RHS_PACKED_B16_XOR_PERMUTE = 6,
} loom_amdgpu_fragment_repack_strategy_t;

typedef enum loom_amdgpu_fragment_repack_reason_e {
  // No rejection reason is associated with the selected strategy.
  LOOM_AMDGPU_FRAGMENT_REPACK_REASON_NONE = 0,
  // Source fragment facts were missing.
  LOOM_AMDGPU_FRAGMENT_REPACK_REASON_SOURCE_FACTS = 1,
  // Source and result fragment shapes differ.
  LOOM_AMDGPU_FRAGMENT_REPACK_REASON_SHAPE = 2,
  // Source and result roles require a target-owned layout transition.
  LOOM_AMDGPU_FRAGMENT_REPACK_REASON_ROLE_TRANSITION = 3,
  // Source and result element storage require a numeric conversion.
  LOOM_AMDGPU_FRAGMENT_REPACK_REASON_TYPE_TRANSITION = 4,
  // Source/result roles and element storage both require target work.
  LOOM_AMDGPU_FRAGMENT_REPACK_REASON_ROLE_TYPE_TRANSITION = 5,
  // No target-owned fragment layout matched the source/result transition.
  LOOM_AMDGPU_FRAGMENT_REPACK_REASON_TARGET_LAYOUT = 6,
  // A target-owned layout matched but no in-register strategy covers it yet.
  LOOM_AMDGPU_FRAGMENT_REPACK_REASON_LAYOUT_STRATEGY = 7,
  // The target-owned layout strategy is missing required packet descriptors.
  LOOM_AMDGPU_FRAGMENT_REPACK_REASON_TARGET_PACKETS = 8,
} loom_amdgpu_fragment_repack_reason_t;

typedef struct loom_amdgpu_fragment_repack_lane_recipe_t {
  // Bit mask applied before shifting; zero produces zero and UINT16_MAX is an
  // identity mask.
  uint16_t and_mask;
  // Logical right shift applied after masking.
  uint16_t right_shift;
} loom_amdgpu_fragment_repack_lane_recipe_t;

typedef enum loom_amdgpu_fragment_repack_packed_pair_kind_e {
  // No adjacent-lane pair construction is selected.
  LOOM_AMDGPU_FRAGMENT_REPACK_PACKED_PAIR_NONE = 0,
  // Exchange the adjacent lane, then convert and pack both values.
  LOOM_AMDGPU_FRAGMENT_REPACK_PACKED_PAIR_EXCHANGE_THEN_PACK = 1,
  // Exchange and pack two already-rounded BF16 bit payloads with DPP.
  LOOM_AMDGPU_FRAGMENT_REPACK_PACKED_PAIR_DPP_PACK_U16 = 2,
  // Exchange and convert two f32 values to a packed BF16 payload with DPP.
  LOOM_AMDGPU_FRAGMENT_REPACK_PACKED_PAIR_DPP_PACK_BF16 = 3,
} loom_amdgpu_fragment_repack_packed_pair_kind_t;

typedef struct loom_amdgpu_fragment_repack_packed_pair_recipe_t {
  // Pair-construction operation selected for the target descriptor set.
  loom_amdgpu_fragment_repack_packed_pair_kind_t kind;
  // Descriptor implementing the selected exchange or fused conversion.
  loom_amdgpu_descriptor_ref_t descriptor_ref;
  // DPP control or DS swizzle offset interpreted by the selected kind.
  uint32_t immediate;
  // Packet family used by an explicit exchange-then-pack recipe.
  loom_amdgpu_crosslane_kind_t crosslane_kind;
} loom_amdgpu_fragment_repack_packed_pair_recipe_t;

// Maximum low register bits exchangeable within one 16-lane DPP row after
// adjacent source lanes have been packed.
#define LOOM_AMDGPU_FRAGMENT_REPACK_TRANSPOSE_STAGE_CAPACITY 3

typedef struct loom_amdgpu_fragment_repack_transpose_stage_t {
  // Cross-lane exchange selected for this butterfly stage.
  loom_amdgpu_direct_xor_lane_recipe_t exchange;
  // Constant wave32 predicate selecting lanes whose exchanged bit is set.
  // Zero indicates that the stage materializes its predicate from lane ids.
  uint32_t lane_bit_set_mask;
  // DPP destination banks corresponding to set lane-id bits.
  // Zero indicates that both register halves use conditional exchanges.
  uint8_t lane_bit_set_bank_mask;
} loom_amdgpu_fragment_repack_transpose_stage_t;

typedef struct loom_amdgpu_fragment_repack_plan_t {
  // Source fragment value being repacked.
  loom_value_id_t source;
  // Result fragment value receiving the repacked payload.
  loom_value_id_t result;
  // Target-owned repack strategy selected for this source/result pair.
  loom_amdgpu_fragment_repack_strategy_t strategy;
  // Reason associated with diagnostic strategies.
  loom_amdgpu_fragment_repack_reason_t reason;
  // Contract role selected for the source fragment layout.
  loom_contract_operand_role_t source_role;
  // Contract role selected for the result fragment layout.
  loom_contract_operand_role_t result_role;
  // Target-owned lane/register layout selected for the fragment transition.
  loom_amdgpu_matrix_fragment_layout_kind_t layout_kind;
  // Number of 32-bit source registers consumed by the selected strategy.
  uint16_t source_register_count;
  // Number of 32-bit result registers produced by the selected strategy.
  uint16_t result_register_count;
  // Number of low register/lane index bits exchanged before gathering.
  uint16_t transpose_bit_count;
  // Number of source-register candidates remaining after the transpose.
  uint16_t transposed_source_register_candidate_count;
  // Number of lanes that share one logical result-fragment register row group.
  uint16_t lane_group_count;
  // Tile row divisor used to derive target-row and target-reduction lane ids.
  uint16_t lane_divisor;
  // Log2 byte spacing between source result-fragment lane groups.
  uint16_t source_lane_group_byte_shift;
  // Log2 byte spacing contributed by the target LHS lane-div reduction group.
  uint16_t result_lane_div_byte_shift;
  // Recipe selecting a source payload register from lane_mod.
  loom_amdgpu_fragment_repack_lane_recipe_t source_register_selector;
  // Recipe selecting the source lane group from lane_mod.
  loom_amdgpu_fragment_repack_lane_recipe_t source_lane_group;
  // Recipe constructing one packed pair from adjacent source columns.
  loom_amdgpu_fragment_repack_packed_pair_recipe_t packed_pair;
  // Strategy-specific repack recipes selected during planning.
  union {
    // Cross-lane exchange recipes in increasing transposed-bit order.
    loom_amdgpu_fragment_repack_transpose_stage_t
        transpose_stages[LOOM_AMDGPU_FRAGMENT_REPACK_TRANSPOSE_STAGE_CAPACITY];
    // Cross-lane exchange pairing low-subword result rows for RHS packing.
    loom_amdgpu_direct_xor_lane_recipe_t result_to_rhs_exchange;
  } strategy_payload;
  // Physical VCC predicates enabling fused conditional transpose stages.
  struct {
    // Descriptor materializing a constant wave32 predicate into VCC_LO.
    loom_amdgpu_descriptor_ref_t constant;
    // Descriptor comparing a lane bit equal to inline zero.
    loom_amdgpu_descriptor_ref_t equal_zero;
    // Descriptor comparing a lane bit not equal to inline zero.
    loom_amdgpu_descriptor_ref_t not_equal_zero;
  } transpose_predicate;
  // Source fragment role fact bitset.
  uint32_t source_role_flags;
  // Result fragment role fact bitset.
  uint32_t result_role_flags;
  // Source vector type.
  loom_type_t source_type;
  // Result vector type.
  loom_type_t result_type;
} loom_amdgpu_fragment_repack_plan_t;

#define LOOM_AMDGPU_ATOMIC_WAIT_CAPACITY 2
#define LOOM_AMDGPU_ATOMIC_CACHE_CONTROL_CAPACITY 2

typedef uint32_t loom_amdgpu_atomic_packet_attr_flags_t;

#define LOOM_AMDGPU_ATOMIC_PACKET_ATTR_SCOPE ((uint32_t)1u << 0)

typedef struct loom_amdgpu_atomic_packet_attrs_t {
  // Attribute bits populated for the selected atomic packet.
  loom_amdgpu_atomic_packet_attr_flags_t flags;
  // Module string ID for the scope attribute when present.
  loom_string_id_t scope_attr_name_id;
  // VGLOBAL SCOPE immediate value encoded on GFX12 atomic packets.
  int64_t scope;
} loom_amdgpu_atomic_packet_attrs_t;

typedef struct loom_amdgpu_atomic_ordering_plan_t {
  // Explicit waits emitted before the atomic packet.
  loom_amdgpu_explicit_packet_plan_t
      pre_atomic_waits[LOOM_AMDGPU_ATOMIC_WAIT_CAPACITY];
  // Number of populated pre-atomic wait packets.
  iree_host_size_t pre_atomic_wait_count;
  // Explicit waits emitted after the atomic packet.
  loom_amdgpu_explicit_packet_plan_t
      post_atomic_waits[LOOM_AMDGPU_ATOMIC_WAIT_CAPACITY];
  // Number of populated post-atomic wait packets.
  iree_host_size_t post_atomic_wait_count;
  // Explicit cache controls emitted after the atomic packet.
  loom_amdgpu_explicit_packet_plan_t
      post_atomic_cache_controls[LOOM_AMDGPU_ATOMIC_CACHE_CONTROL_CAPACITY];
  // Number of populated post-atomic cache-control packets.
  iree_host_size_t post_atomic_cache_control_descriptor_count;
} loom_amdgpu_atomic_ordering_plan_t;

typedef uint32_t loom_amdgpu_atomic_plan_flags_t;

#define LOOM_AMDGPU_ATOMIC_PLAN_REQUIRES_M0 ((uint32_t)1u << 0)

typedef struct loom_amdgpu_atomic_plan_t {
  // Target-independent source memory access plan being wrapped.
  loom_low_source_memory_access_plan_t source;
  // Target-specific lowering flags derived from the selected descriptor.
  loom_amdgpu_atomic_plan_flags_t flags;
  // Source atomic operation form being lowered.
  loom_amdgpu_atomic_operation_kind_t operation_kind;
  // Selected target addressing form for the atomic packet.
  loom_amdgpu_memory_address_form_t address_form;
  // Target operand path selected for each source dynamic address term.
  loom_amdgpu_memory_dynamic_index_kind_t
      dynamic_term_kinds[LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_CAPACITY];
  // Static offset value encoded in the descriptor offset immediate.
  int64_t immediate_offset;
  // Static byte offset materialized through the scalar SOFFSET operand.
  uint32_t scalar_byte_offset;
  // Descriptor row selected for the active descriptor set.
  loom_low_lower_resolved_descriptor_t descriptor;
  // Descriptor attrs emitted directly on the selected atomic packet.
  loom_amdgpu_atomic_packet_attrs_t packet_attrs;
  // Explicit packets required to implement source atomic ordering.
  loom_amdgpu_atomic_ordering_plan_t ordering;
} loom_amdgpu_atomic_plan_t;

typedef struct loom_amdgpu_prefetch_plan_t {
  // Descriptor row selected for the prefetch packet.
  loom_low_lower_resolved_descriptor_t descriptor;
  // Descriptor ordinal selected from the active descriptor set.
  uint32_t descriptor_ordinal;
  // Module string ID for the descriptor's offset attribute.
  loom_string_id_t offset_attr_name_id;
  // Module string ID for the descriptor's count attribute.
  loom_string_id_t count_attr_name_id;
  // Target-independent source memory access plan being wrapped.
  loom_low_source_memory_access_plan_t source;
  // Target operand path selected for the source dynamic address term.
  loom_amdgpu_memory_dynamic_index_kind_t dynamic_term_kind;
  // Static offset value encoded in the descriptor offset immediate.
  int64_t immediate_offset;
  // Static byte offset materialized through the scalar SOFFSET operand.
  uint32_t scalar_byte_offset;
  // Prefetch span count encoded in the descriptor count immediate.
  uint32_t count;
} loom_amdgpu_prefetch_plan_t;

typedef struct loom_amdgpu_async_gather_plan_t {
  // Source global-like view access transferred into LDS.
  loom_low_source_memory_access_plan_t source;
  // Target operand path selected for each source dynamic address term.
  loom_amdgpu_memory_dynamic_index_kind_t
      source_dynamic_term_kinds[LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_CAPACITY];
  // Static LDS byte offset materialized into M0.
  uint32_t dest_byte_offset;
  // Static global byte offset encoded in the packet immediate.
  int64_t source_immediate_offset;
  // Number of bytes moved by the selected async packet.
  uint32_t packet_byte_count;
  // Descriptor row selected for the active descriptor set.
  loom_low_lower_resolved_descriptor_t descriptor;
} loom_amdgpu_async_gather_plan_t;

typedef struct loom_amdgpu_cluster_gather_plan_t {
  // Exact u32 global byte offset materialized into the packet VADDR operand.
  loom_amdgpu_memory_access_t source_address;
  // Exact u32 workgroup-relative byte offset materialized into the LDS address
  // operand, including target-assigned alloca layout.
  loom_amdgpu_memory_access_t dest_address;
  // Exact low 16-bit set of participating flat cluster workgroup ranks.
  uint32_t participant_mask;
  // Number of bytes moved by the selected cluster transfer packet.
  uint32_t packet_byte_count;
  // Descriptor row selected for the active descriptor set.
  loom_low_lower_resolved_descriptor_t descriptor;
} loom_amdgpu_cluster_gather_plan_t;

#define LOOM_AMDGPU_TENSOR_DGROUP_CAPACITY 4

typedef struct loom_amdgpu_tensor_load_plan_t {
  // Descriptor row selected for the d2 or d4 tensor-load packet.
  loom_low_lower_resolved_descriptor_t descriptor;
  // Descriptor row used to move each uniform D-group lane into an SGPR.
  loom_low_lower_resolved_descriptor_t readfirstlane_descriptor;
  // Source values materialized as the packet's D0 through D3 SGPR groups.
  loom_value_id_t dgroups[LOOM_AMDGPU_TENSOR_DGROUP_CAPACITY];
  // Number of populated D-group source values.
  uint8_t dgroup_count;
  // Source cache policy encoded on the tensor-load packet.
  loom_vector_memory_cache_policy_t cache_policy;
} loom_amdgpu_tensor_load_plan_t;

#define LOOM_AMDGPU_ASYNC_WAIT_IMMEDIATE_CAPACITY \
  LOOM_AMDGPU_EXPLICIT_PACKET_IMMEDIATE_CAPACITY

typedef loom_amdgpu_explicit_packet_immediate_template_t
    loom_amdgpu_async_wait_immediate_t;

typedef struct loom_amdgpu_async_wait_plan_t {
  // Explicit wait packets selected independently for each async counter.
  loom_amdgpu_explicit_packet_plan_t waits[LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT];
  // Number of populated wait packets.
  uint8_t wait_count;
} loom_amdgpu_async_wait_plan_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_PLAN_H_
