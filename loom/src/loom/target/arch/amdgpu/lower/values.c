// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/values.h"

#include <stdint.h>
#include <string.h>

#include "loom/ir/context.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/encoding_auxiliary.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/contracts/arithmetic_lower_rules.h"
#include "loom/target/arch/amdgpu/error_catalog.h"
#include "loom/target/arch/amdgpu/lower/arithmetic.h"
#include "loom/target/arch/amdgpu/lower/bitpack.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/legality.h"
#include "loom/target/arch/amdgpu/lower/narrow_float/bf16.h"
#include "loom/target/arch/amdgpu/lower/narrow_float/fp8.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

typedef uint32_t loom_amdgpu_vector_iota_plan_flags_t;

enum {
  LOOM_AMDGPU_VECTOR_IOTA_PLAN_BASE_EXACT = 1u << 0,
  LOOM_AMDGPU_VECTOR_IOTA_PLAN_STEP_EXACT = 1u << 1,
};

typedef struct loom_amdgpu_vector_iota_plan_t {
  // Descriptor row selected for each lane constant packet.
  loom_low_lower_resolved_descriptor_t descriptor;
  // Module string ID for the descriptor's imm32 attribute.
  loom_string_id_t imm32_attr_name_id;
  // Source scalar base value used by dynamic lane materialization.
  loom_value_id_t base;
  // Source scalar step value used by dynamic lane materialization.
  loom_value_id_t step;
  // Result vector receiving the generated i32 lane constants.
  loom_value_id_t result;
  // Exact base value when BASE_EXACT is set.
  int32_t exact_base;
  // Exact step value when STEP_EXACT is set.
  int32_t exact_step;
  // Static operand facts selected by the planner.
  loom_amdgpu_vector_iota_plan_flags_t flags;
  // Static number of generated lanes.
  uint32_t lane_count;
  // Precomputed lane bit patterns emitted as VGPR constants.
  uint32_t lane_bit_patterns[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
} loom_amdgpu_vector_iota_plan_t;

static bool loom_amdgpu_vector_iota_plan_has_exact_base(
    const loom_amdgpu_vector_iota_plan_t* plan) {
  return iree_any_bit_set(plan->flags, LOOM_AMDGPU_VECTOR_IOTA_PLAN_BASE_EXACT);
}

static bool loom_amdgpu_vector_iota_plan_has_exact_step(
    const loom_amdgpu_vector_iota_plan_t* plan) {
  return iree_any_bit_set(plan->flags, LOOM_AMDGPU_VECTOR_IOTA_PLAN_STEP_EXACT);
}

static bool loom_amdgpu_vector_iota_plan_is_dynamic(
    const loom_amdgpu_vector_iota_plan_t* plan) {
  return !iree_all_bits_set(plan->flags,
                            LOOM_AMDGPU_VECTOR_IOTA_PLAN_BASE_EXACT |
                                LOOM_AMDGPU_VECTOR_IOTA_PLAN_STEP_EXACT);
}

typedef enum loom_amdgpu_vector_from_elements_materialization_kind_e {
  LOOM_AMDGPU_VECTOR_FROM_ELEMENTS_MATERIALIZATION_OPERANDS = 0,
  LOOM_AMDGPU_VECTOR_FROM_ELEMENTS_MATERIALIZATION_EXACT_PACKED_INTEGER = 1,
} loom_amdgpu_vector_from_elements_materialization_kind_t;

typedef struct loom_amdgpu_vector_from_elements_plan_t {
  // Result vector assembled from the selected source elements.
  loom_value_id_t result;
  // Physical storage selected for the result vector.
  loom_amdgpu_vector_storage_kind_t storage_kind;
  // Materialization path selected from storage shape and value facts.
  loom_amdgpu_vector_from_elements_materialization_kind_t materialization_kind;
  // Static source element count.
  uint32_t element_count;
  // Static result register count after source elements are packed.
  uint32_t register_count;
  // Static 32-bit register count occupied by one source element.
  uint32_t element_register_count;
  // Static payload bit count occupied by one source element.
  uint32_t element_bit_count;
  // Source and result scalar element type.
  loom_scalar_type_t element_type;
  // Bitmask of lanes selected as half-result mixed FMA packets.
  uint64_t fma_mix_half_result_lane_mask;
  // Optional mixed FMA packet plans indexed by result lane.
  loom_amdgpu_fma_mix_plan_t* fma_mix_half_results;
  // State consumed by the selected materialization path.
  union {
    // Source scalar values in result lane order for operand materialization.
    loom_value_id_t elements[LOOM_AMDGPU_MAX_PACKED_16BIT_FLOAT_LANES];
    // Exact packed-register bit patterns for constant materialization.
    uint32_t
        packed_register_bit_patterns[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS];
  } payload;
} loom_amdgpu_vector_from_elements_plan_t;

typedef enum loom_amdgpu_vector_insert_value_kind_e {
  LOOM_AMDGPU_VECTOR_INSERT_VALUE_KIND_DEFAULT = 0,
  LOOM_AMDGPU_VECTOR_INSERT_VALUE_KIND_FMA_MIX_HALF_RESULT = 1,
} loom_amdgpu_vector_insert_value_kind_t;

typedef enum loom_amdgpu_vector_insert_select_flags_e {
  LOOM_AMDGPU_VECTOR_INSERT_SELECT_FLAG_NONE = 0,
  LOOM_AMDGPU_VECTOR_INSERT_SELECT_FLAG_EMIT_DIAGNOSTICS = 1u << 0,
} loom_amdgpu_vector_insert_select_flags_t;

typedef struct loom_amdgpu_vector_insert_plan_t {
  // Scalar value inserted into the destination vector.
  loom_value_id_t value;
  // Destination vector whose lanes are copied except at the selected index.
  loom_value_id_t dest;
  // Optional dynamic destination lane index, or invalid for static insertion.
  loom_value_id_t dynamic_index;
  // Result vector receiving the updated lane payload.
  loom_value_id_t result;
  // Static destination lane offset.
  uint32_t lane_offset;
  // Static logical destination lane count.
  uint32_t lane_count;
  // Static 32-bit backing register count for the destination vector.
  uint32_t register_count;
  // Number of payload bits occupied by each logical destination lane.
  uint32_t lane_bit_count;
  // Source and result scalar element type.
  loom_scalar_type_t element_type;
  // Lowering path selected for the inserted scalar value.
  loom_amdgpu_vector_insert_value_kind_t value_kind;
  // Mixed-FMA packet emitted for FMA_MIX_HALF_RESULT values.
  loom_amdgpu_fma_mix_plan_t fma_mix;
  // True when insertion uses |dynamic_index| instead of |lane_offset|.
  bool is_dynamic;
} loom_amdgpu_vector_insert_plan_t;

static uint32_t loom_amdgpu_integer_bit_mask(uint32_t bit_count) {
  IREE_ASSERT_GT(bit_count, 0);
  IREE_ASSERT_LE(bit_count, 32);
  return bit_count == 32 ? UINT32_MAX : ((UINT32_C(1) << bit_count) - 1u);
}

static uint32_t loom_amdgpu_integer_low_bits(int64_t value,
                                             uint32_t bit_count) {
  return (uint32_t)((uint64_t)value & loom_amdgpu_integer_bit_mask(bit_count));
}

static bool loom_amdgpu_exact_integer_lane_bits(
    loom_low_lower_context_t* context, loom_value_id_t source_value,
    uint32_t bit_count, uint32_t* out_bits) {
  *out_bits = 0;
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  if (fact_table == NULL) {
    return false;
  }
  int64_t value = 0;
  if (!loom_value_facts_as_exact_i64(
          loom_value_fact_table_lookup(fact_table, source_value), &value)) {
    return false;
  }
  *out_bits = loom_amdgpu_integer_low_bits(value, bit_count);
  return true;
}

static bool loom_amdgpu_pack_exact_integer_elements(
    loom_low_lower_context_t* context,
    const loom_amdgpu_vector_from_elements_plan_t* plan,
    uint32_t* out_bit_patterns) {
  if (plan->element_bit_count != 8 && plan->element_bit_count != 16) {
    return false;
  }
  const uint32_t element_bit_count = plan->element_bit_count;
  const uint32_t elements_per_register = 32u / element_bit_count;
  const uint32_t element_mask = loom_amdgpu_integer_bit_mask(element_bit_count);
  for (uint32_t register_index = 0; register_index < plan->register_count;
       ++register_index) {
    uint32_t bit_pattern = 0;
    const uint32_t lane_base = register_index * elements_per_register;
    for (uint32_t lane_index = 0; lane_index < elements_per_register;
         ++lane_index) {
      const uint32_t element_index = lane_base + lane_index;
      if (element_index >= plan->element_count) {
        break;
      }
      uint32_t lane_bits = 0;
      if (!loom_amdgpu_exact_integer_lane_bits(
              context, plan->payload.elements[element_index], element_bit_count,
              &lane_bits)) {
        return false;
      }
      bit_pattern |= (lane_bits & element_mask)
                     << (lane_index * element_bit_count);
    }
    out_bit_patterns[register_index] = bit_pattern;
  }
  return true;
}

static void loom_amdgpu_select_vector_from_elements_materialization(
    loom_low_lower_context_t* context,
    loom_amdgpu_vector_from_elements_plan_t* plan) {
  plan->materialization_kind =
      LOOM_AMDGPU_VECTOR_FROM_ELEMENTS_MATERIALIZATION_OPERANDS;
  if (plan->storage_kind != LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_INTEGER) {
    return;
  }

  uint32_t bit_patterns[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS];
  if (!loom_amdgpu_pack_exact_integer_elements(context, plan, bit_patterns)) {
    return;
  }
  plan->materialization_kind =
      LOOM_AMDGPU_VECTOR_FROM_ELEMENTS_MATERIALIZATION_EXACT_PACKED_INTEGER;
  for (uint32_t i = 0; i < plan->register_count; ++i) {
    plan->payload.packed_register_bit_patterns[i] = bit_patterns[i];
  }
}

typedef enum loom_amdgpu_scalar_conversion_op_group_e {
  LOOM_AMDGPU_SCALAR_CONVERSION_OP_TRUNCI = 0,
  LOOM_AMDGPU_SCALAR_CONVERSION_OP_EXTF,
  LOOM_AMDGPU_SCALAR_CONVERSION_OP_EXTSI,
  LOOM_AMDGPU_SCALAR_CONVERSION_OP_EXTUI,
  LOOM_AMDGPU_SCALAR_CONVERSION_OP_UITOFP,
  LOOM_AMDGPU_SCALAR_CONVERSION_OP_FPTOSI,
  LOOM_AMDGPU_SCALAR_CONVERSION_OP_FPTOUI,
  LOOM_AMDGPU_SCALAR_CONVERSION_OP_COUNT_,
} loom_amdgpu_scalar_conversion_op_group_t;

typedef enum loom_amdgpu_scalar_conversion_rule_index_e {
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_NONE = 0,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_TRUNCI_I16_TO_I8,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_TRUNCI_I32_TO_I8,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_TRUNCI_I32_TO_I16,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_TRUNCI_I64_TO_I8,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_TRUNCI_I64_TO_I16,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_TRUNCI_I64_TO_I32,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTF_F8E4M3_TO_BF16,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTF_F8E5M2_TO_BF16,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTSI_I8_TO_I16,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTSI_I8_TO_I32,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTSI_I8_TO_I64,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTSI_I16_TO_I32,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTSI_I16_TO_I64,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTSI_I32_TO_I64,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTUI_I8_TO_I16,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTUI_I8_TO_I32,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTUI_I8_TO_I64,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTUI_I16_TO_I32,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTUI_I16_TO_I64,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTUI_I32_TO_I64,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_UITOFP_I8_TO_F32,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_UITOFP_I16_TO_F32,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_FPTOSI_F32_TO_I32,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_FPTOSI_F32_TO_I8,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_FPTOSI_F32_TO_I16,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_FPTOUI_F32_TO_I32,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_FPTOUI_F32_TO_I8,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_FPTOUI_F32_TO_I16,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_COUNT_,
} loom_amdgpu_scalar_conversion_rule_index_t;
static_assert(LOOM_AMDGPU_SCALAR_CONVERSION_RULE_COUNT_ <= UINT8_MAX,
              "conversion rule indexes must fit in the byte selector table");

typedef struct loom_amdgpu_scalar_conversion_rule_t {
  // Lowering strategy selected when the rule matches.
  loom_amdgpu_scalar_conversion_kind_t kind;
  // Descriptor emitted by strategies that perform a conversion packet.
  loom_amdgpu_descriptor_ref_t convert_descriptor_ref;
  // Descriptor refs that must be present before the rule can select.
  loom_amdgpu_descriptor_ref_t required_descriptor_refs[4];
} loom_amdgpu_scalar_conversion_rule_t;

typedef struct loom_amdgpu_descriptor_requirement_t {
  // Constraint key reported when this descriptor ref is missing.
  iree_string_view_t constraint_key;
  // Descriptor ref required by the lowering strategy.
  loom_amdgpu_descriptor_ref_t descriptor_ref;
} loom_amdgpu_descriptor_requirement_t;

typedef struct loom_amdgpu_descriptor_requirement_span_t {
  // Descriptor requirements required by this span.
  const loom_amdgpu_descriptor_requirement_t* requirements;
  // Number of entries in requirements.
  iree_host_size_t requirement_count;
} loom_amdgpu_descriptor_requirement_span_t;

typedef struct loom_amdgpu_i64_alu_descriptor_requirement_row_t {
  // First descriptor requirement span for the selected operation kind.
  loom_amdgpu_descriptor_requirement_span_t first;
  // Optional second descriptor requirement span for fused operation kinds.
  loom_amdgpu_descriptor_requirement_span_t second;
} loom_amdgpu_i64_alu_descriptor_requirement_row_t;

#define LOOM_AMDGPU_DESCRIPTOR_REQUIREMENT_SPAN(requirements_) \
  {                                                            \
      .requirements = requirements_,                           \
      .requirement_count = IREE_ARRAYSIZE(requirements_),      \
  }

static bool loom_amdgpu_descriptor_requirements_present(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_descriptor_requirement_t* requirements,
    iree_host_size_t requirement_count,
    iree_string_view_t* out_constraint_key) {
  for (iree_host_size_t i = 0; i < requirement_count; ++i) {
    *out_constraint_key = requirements[i].constraint_key;
    if (!loom_amdgpu_descriptor_set_has_ref(descriptor_set,
                                            requirements[i].descriptor_ref)) {
      return false;
    }
  }
  return true;
}

static bool loom_amdgpu_descriptor_requirement_span_present(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_requirement_span_t span,
    iree_string_view_t* out_constraint_key) {
  if (span.requirement_count == 0) {
    return true;
  }
  IREE_ASSERT(span.requirements != NULL);
  return loom_amdgpu_descriptor_requirements_present(
      descriptor_set, span.requirements, span.requirement_count,
      out_constraint_key);
}

static bool loom_amdgpu_i64_alu_descriptor_requirements_present(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_i64_alu_descriptor_requirement_row_t* rows,
    iree_host_size_t row_count, iree_host_size_t row_index,
    iree_string_view_t fallback_constraint_key,
    iree_string_view_t* out_constraint_key) {
  if (row_index >= row_count || rows[row_index].first.requirement_count == 0) {
    *out_constraint_key = fallback_constraint_key;
    return false;
  }
  const loom_amdgpu_i64_alu_descriptor_requirement_row_t* row =
      &rows[row_index];
  return loom_amdgpu_descriptor_requirement_span_present(
             descriptor_set, row->first, out_constraint_key) &&
         loom_amdgpu_descriptor_requirement_span_present(
             descriptor_set, row->second, out_constraint_key);
}

static bool loom_amdgpu_descriptor_requirement_present(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_string_view_t constraint_key,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    iree_string_view_t* out_constraint_key) {
  *out_constraint_key = constraint_key;
  return loom_amdgpu_descriptor_set_has_ref(descriptor_set, descriptor_ref);
}

static const loom_amdgpu_descriptor_requirement_t
    kAmdgpuOffsetAddVgprDescriptorRequirements[] = {
        {
            .constraint_key = IREE_SVL("descriptor.v_mov_b32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        },
        {
            .constraint_key = IREE_SVL("descriptor.v_mov_b32_copy"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_COPY,
        },
        {
            .constraint_key = IREE_SVL("descriptor.v_add_co_u32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_CO_U32,
        },
        {
            .constraint_key = IREE_SVL("descriptor.v_add_co_ci_u32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_CO_CI_U32,
        },
};

static const loom_amdgpu_descriptor_requirement_t
    kAmdgpuOffsetAddSgprDescriptorRequirements[] = {
        {
            .constraint_key = IREE_SVL("descriptor.s_mov_b32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
        },
        {
            .constraint_key = IREE_SVL("descriptor.s_add_u32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_U32,
        },
        {
            .constraint_key = IREE_SVL("descriptor.s_addc_u32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_ADDC_U32,
        },
};

static const loom_amdgpu_descriptor_requirement_t
    kAmdgpuScalarI64SubVgprDescriptorRequirements[] = {
        {
            .constraint_key = IREE_SVL("descriptor.v_mov_b32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        },
        {
            .constraint_key = IREE_SVL("descriptor.v_mov_b32_copy"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_COPY,
        },
        {
            .constraint_key = IREE_SVL("descriptor.v_sub_co_u32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_SUB_CO_U32,
        },
        {
            .constraint_key = IREE_SVL("descriptor.v_sub_co_ci_u32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_SUB_CO_CI_U32,
        },
};

static const loom_amdgpu_descriptor_requirement_t
    kAmdgpuScalarI64ShlVgprDescriptorRequirements[] = {
        {
            .constraint_key = IREE_SVL("descriptor.v_mov_b32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        },
        {
            .constraint_key = IREE_SVL("descriptor.v_mov_b32_copy"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_COPY,
        },
        {
            .constraint_key = IREE_SVL("descriptor.v_lshlrev_b64"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B64,
        },
};

static const loom_amdgpu_descriptor_requirement_t
    kAmdgpuScalarI64MulVgprDescriptorRequirements[] = {
        {
            .constraint_key = IREE_SVL("descriptor.v_mul_lo_u32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_LO_U32,
        },
        {
            .constraint_key = IREE_SVL("descriptor.v_mul_hi_u32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_HI_U32,
        },
        {
            .constraint_key = IREE_SVL("descriptor.v_add_u32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32,
        },
};

static const loom_amdgpu_descriptor_requirement_t
    kAmdgpuVgprMoveDescriptorRequirements[] = {
        {
            .constraint_key = IREE_SVL("descriptor.v_mov_b32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        },
        {
            .constraint_key = IREE_SVL("descriptor.v_mov_b32_copy"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_COPY,
        },
};

static const loom_amdgpu_descriptor_requirement_t
    kAmdgpuI64CompareHighEqualDescriptorRequirements[] = {
        {
            .constraint_key = IREE_SVL("descriptor.v_cmp_eq_i32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
        },
        {
            .constraint_key = IREE_SVL("descriptor.s_and_b64"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64,
        },
};

static const loom_amdgpu_i64_alu_descriptor_requirement_row_t
    kAmdgpuScalarI64AluDescriptorRequirementRows
        [LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_SHL + 1] = {
            [LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_ADD] =
                {
                    .first = LOOM_AMDGPU_DESCRIPTOR_REQUIREMENT_SPAN(
                        kAmdgpuOffsetAddVgprDescriptorRequirements),
                },
            [LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_SUB] =
                {
                    .first = LOOM_AMDGPU_DESCRIPTOR_REQUIREMENT_SPAN(
                        kAmdgpuScalarI64SubVgprDescriptorRequirements),
                },
            [LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_MUL_LO] =
                {
                    .first = LOOM_AMDGPU_DESCRIPTOR_REQUIREMENT_SPAN(
                        kAmdgpuScalarI64MulVgprDescriptorRequirements),
                },
            [LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_SHL] =
                {
                    .first = LOOM_AMDGPU_DESCRIPTOR_REQUIREMENT_SPAN(
                        kAmdgpuScalarI64ShlVgprDescriptorRequirements),
                },
};

static const loom_amdgpu_i64_alu_descriptor_requirement_row_t
    kAmdgpuAddressI64AluDescriptorRequirementRows
        [LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_MADD_LO + 1] = {
            [LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_SGPR_ADD] =
                {
                    .first = LOOM_AMDGPU_DESCRIPTOR_REQUIREMENT_SPAN(
                        kAmdgpuOffsetAddSgprDescriptorRequirements),
                },
            [LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_ADD] =
                {
                    .first = LOOM_AMDGPU_DESCRIPTOR_REQUIREMENT_SPAN(
                        kAmdgpuOffsetAddVgprDescriptorRequirements),
                },
            [LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_SUB] =
                {
                    .first = LOOM_AMDGPU_DESCRIPTOR_REQUIREMENT_SPAN(
                        kAmdgpuScalarI64SubVgprDescriptorRequirements),
                },
            [LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_MUL_LO] =
                {
                    .first = LOOM_AMDGPU_DESCRIPTOR_REQUIREMENT_SPAN(
                        kAmdgpuScalarI64MulVgprDescriptorRequirements),
                },
            [LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_SHL] =
                {
                    .first = LOOM_AMDGPU_DESCRIPTOR_REQUIREMENT_SPAN(
                        kAmdgpuScalarI64ShlVgprDescriptorRequirements),
                },
            [LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_MADD_LO] =
                {
                    .first = LOOM_AMDGPU_DESCRIPTOR_REQUIREMENT_SPAN(
                        kAmdgpuScalarI64MulVgprDescriptorRequirements),
                    .second = LOOM_AMDGPU_DESCRIPTOR_REQUIREMENT_SPAN(
                        kAmdgpuOffsetAddVgprDescriptorRequirements),
                },
};

#undef LOOM_AMDGPU_DESCRIPTOR_REQUIREMENT_SPAN

static bool loom_amdgpu_value_use_is_atomic_offset(const loom_module_t* module,
                                                   const loom_op_t* user_op,
                                                   loom_value_id_t value_id) {
  loom_memory_access_t access = loom_memory_access_cast(module, user_op);
  return loom_memory_access_isa(access) &&
         loom_memory_access_operation_kind_is_atomic(
             loom_memory_access_operation_kind(access)) &&
         loom_memory_access_offsets(access) == value_id;
}

static bool loom_amdgpu_value_only_feeds_vector_atomic_offsets(
    const loom_module_t* module, loom_value_id_t value_id) {
  if (value_id >= module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_has_no_uses(value)) {
    return false;
  }
  const loom_use_t* use = NULL;
  loom_value_for_each_use(value, use) {
    if (!loom_amdgpu_value_use_is_atomic_offset(module, loom_use_user_op(*use),
                                                value_id)) {
      return false;
    }
  }
  return true;
}

static bool loom_amdgpu_select_fact_only_vector_atomic_offset_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  if (source_op->result_count != 1) {
    return false;
  }
  const loom_value_id_t result = loom_op_const_results(source_op)[0];
  if (!loom_amdgpu_value_only_feeds_vector_atomic_offsets(module, result)) {
    return false;
  }
  *out_plan = loom_low_lower_plan_make(source_op->kind, NULL);
  return true;
}

static loom_scalar_type_t loom_amdgpu_scalar_type_or_none(loom_type_t type) {
  if (!loom_type_is_scalar(type)) {
    return LOOM_SCALAR_TYPE_COUNT_;
  }
  return loom_type_element_type(type);
}

static bool loom_amdgpu_type_is_index_scalar(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_INDEX;
}

static uint32_t loom_amdgpu_target_index_bitwidth(
    loom_low_lower_context_t* context) {
  const loom_target_bundle_t* bundle = loom_low_lower_context_bundle(context);
  IREE_ASSERT(bundle != NULL && bundle->snapshot != NULL);
  return bundle != NULL && bundle->snapshot != NULL
             ? bundle->snapshot->index_bitwidth
             : 0;
}

static bool loom_amdgpu_address_cmp_needs_64bit(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_op_t* source_op) {
  const loom_value_id_t lhs = loom_index_cmp_lhs(source_op);
  const loom_value_id_t rhs = loom_index_cmp_rhs(source_op);
  if (module == NULL || lhs >= module->values.count ||
      rhs >= module->values.count) {
    return false;
  }
  const loom_type_t lhs_type = loom_module_value_type(module, lhs);
  const loom_type_t rhs_type = loom_module_value_type(module, rhs);
  return loom_amdgpu_source_address_value_needs_64bit(module, fact_table, lhs,
                                                      lhs_type) ||
         loom_amdgpu_source_address_value_needs_64bit(module, fact_table, rhs,
                                                      rhs_type);
}

static bool loom_amdgpu_scalar_cmpi_has_i64_operands(
    const loom_module_t* module, const loom_op_t* source_op) {
  if (!loom_scalar_cmpi_isa(source_op)) {
    return false;
  }
  const loom_value_id_t lhs = loom_scalar_cmpi_lhs(source_op);
  const loom_value_id_t rhs = loom_scalar_cmpi_rhs(source_op);
  return lhs < module->values.count && rhs < module->values.count &&
         loom_amdgpu_type_is_i64(loom_module_value_type(module, lhs)) &&
         loom_amdgpu_type_is_i64(loom_module_value_type(module, rhs));
}

static bool loom_amdgpu_index_cmp_predicate_to_scalar(
    uint8_t index_predicate,
    loom_scalar_cmpi_predicate_t* out_scalar_predicate) {
  switch ((loom_index_cmp_predicate_t)index_predicate) {
    case LOOM_INDEX_CMP_PREDICATE_EQ:
      *out_scalar_predicate = LOOM_SCALAR_CMPI_PREDICATE_EQ;
      return true;
    case LOOM_INDEX_CMP_PREDICATE_NE:
      *out_scalar_predicate = LOOM_SCALAR_CMPI_PREDICATE_NE;
      return true;
    case LOOM_INDEX_CMP_PREDICATE_SLT:
      *out_scalar_predicate = LOOM_SCALAR_CMPI_PREDICATE_SLT;
      return true;
    case LOOM_INDEX_CMP_PREDICATE_SLE:
      *out_scalar_predicate = LOOM_SCALAR_CMPI_PREDICATE_SLE;
      return true;
    case LOOM_INDEX_CMP_PREDICATE_SGT:
      *out_scalar_predicate = LOOM_SCALAR_CMPI_PREDICATE_SGT;
      return true;
    case LOOM_INDEX_CMP_PREDICATE_SGE:
      *out_scalar_predicate = LOOM_SCALAR_CMPI_PREDICATE_SGE;
      return true;
    case LOOM_INDEX_CMP_PREDICATE_ULT:
      *out_scalar_predicate = LOOM_SCALAR_CMPI_PREDICATE_ULT;
      return true;
    case LOOM_INDEX_CMP_PREDICATE_ULE:
      *out_scalar_predicate = LOOM_SCALAR_CMPI_PREDICATE_ULE;
      return true;
    case LOOM_INDEX_CMP_PREDICATE_UGT:
      *out_scalar_predicate = LOOM_SCALAR_CMPI_PREDICATE_UGT;
      return true;
    case LOOM_INDEX_CMP_PREDICATE_UGE:
      *out_scalar_predicate = LOOM_SCALAR_CMPI_PREDICATE_UGE;
      return true;
    default:
      return false;
  }
}

typedef uint8_t loom_amdgpu_i64_compare_predicate_flags_t;

// I64 compare must include a high-word equality test before combining the
// high-word ordered compare with the low-word unsigned compare.
#define LOOM_AMDGPU_I64_COMPARE_PREDICATE_NEEDS_HIGH_EQUAL ((uint8_t)1u << 0)

typedef struct loom_amdgpu_i64_compare_predicate_descriptor_row_t {
  // High-word compare descriptor ref.
  loom_amdgpu_descriptor_ref_t high_descriptor_ref;
  // Low-word compare descriptor ref.
  loom_amdgpu_descriptor_ref_t low_descriptor_ref;
  // Scalar mask combine descriptor ref.
  loom_amdgpu_descriptor_ref_t combine_descriptor_ref;
  // Extra lowering requirements for this predicate row.
  loom_amdgpu_i64_compare_predicate_flags_t flags;
} loom_amdgpu_i64_compare_predicate_descriptor_row_t;

#define LOOM_AMDGPU_I64_COMPARE_DESCRIPTOR_ROW(                               \
    high_descriptor_ref_, low_descriptor_ref_, combine_descriptor_ref_,       \
    flags_)                                                                   \
  {                                                                           \
      .high_descriptor_ref =                                                  \
          LOOM_AMDGPU_DESCRIPTOR_REF_##high_descriptor_ref_,                  \
      .low_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_##low_descriptor_ref_, \
      .combine_descriptor_ref =                                               \
          LOOM_AMDGPU_DESCRIPTOR_REF_##combine_descriptor_ref_,               \
      .flags = flags_,                                                        \
  }

static const loom_amdgpu_i64_compare_predicate_descriptor_row_t
    kAmdgpuI64ComparePredicateDescriptorRows
        [LOOM_SCALAR_CMPI_PREDICATE_COUNT_] = {
            [LOOM_SCALAR_CMPI_PREDICATE_EQ] =
                LOOM_AMDGPU_I64_COMPARE_DESCRIPTOR_ROW(
                    V_CMP_EQ_I32, V_CMP_EQ_I32, S_AND_B64, 0),
            [LOOM_SCALAR_CMPI_PREDICATE_NE] =
                LOOM_AMDGPU_I64_COMPARE_DESCRIPTOR_ROW(
                    V_CMP_NE_I32, V_CMP_NE_I32, S_OR_B64, 0),
            [LOOM_SCALAR_CMPI_PREDICATE_SLT] =
                LOOM_AMDGPU_I64_COMPARE_DESCRIPTOR_ROW(
                    V_CMP_SLT_I32, V_CMP_ULT_U32, S_OR_B64,
                    LOOM_AMDGPU_I64_COMPARE_PREDICATE_NEEDS_HIGH_EQUAL),
            [LOOM_SCALAR_CMPI_PREDICATE_SLE] =
                LOOM_AMDGPU_I64_COMPARE_DESCRIPTOR_ROW(
                    V_CMP_SLT_I32, V_CMP_ULE_U32, S_OR_B64,
                    LOOM_AMDGPU_I64_COMPARE_PREDICATE_NEEDS_HIGH_EQUAL),
            [LOOM_SCALAR_CMPI_PREDICATE_SGT] =
                LOOM_AMDGPU_I64_COMPARE_DESCRIPTOR_ROW(
                    V_CMP_SGT_I32, V_CMP_UGT_U32, S_OR_B64,
                    LOOM_AMDGPU_I64_COMPARE_PREDICATE_NEEDS_HIGH_EQUAL),
            [LOOM_SCALAR_CMPI_PREDICATE_SGE] =
                LOOM_AMDGPU_I64_COMPARE_DESCRIPTOR_ROW(
                    V_CMP_SGT_I32, V_CMP_UGE_U32, S_OR_B64,
                    LOOM_AMDGPU_I64_COMPARE_PREDICATE_NEEDS_HIGH_EQUAL),
            [LOOM_SCALAR_CMPI_PREDICATE_ULT] =
                LOOM_AMDGPU_I64_COMPARE_DESCRIPTOR_ROW(
                    V_CMP_ULT_U32, V_CMP_ULT_U32, S_OR_B64,
                    LOOM_AMDGPU_I64_COMPARE_PREDICATE_NEEDS_HIGH_EQUAL),
            [LOOM_SCALAR_CMPI_PREDICATE_ULE] =
                LOOM_AMDGPU_I64_COMPARE_DESCRIPTOR_ROW(
                    V_CMP_ULT_U32, V_CMP_ULE_U32, S_OR_B64,
                    LOOM_AMDGPU_I64_COMPARE_PREDICATE_NEEDS_HIGH_EQUAL),
            [LOOM_SCALAR_CMPI_PREDICATE_UGT] =
                LOOM_AMDGPU_I64_COMPARE_DESCRIPTOR_ROW(
                    V_CMP_UGT_U32, V_CMP_UGT_U32, S_OR_B64,
                    LOOM_AMDGPU_I64_COMPARE_PREDICATE_NEEDS_HIGH_EQUAL),
            [LOOM_SCALAR_CMPI_PREDICATE_UGE] =
                LOOM_AMDGPU_I64_COMPARE_DESCRIPTOR_ROW(
                    V_CMP_UGT_U32, V_CMP_UGE_U32, S_OR_B64,
                    LOOM_AMDGPU_I64_COMPARE_PREDICATE_NEEDS_HIGH_EQUAL),
};

#undef LOOM_AMDGPU_I64_COMPARE_DESCRIPTOR_ROW

static bool loom_amdgpu_i64_compare_predicate_descriptors(
    loom_scalar_cmpi_predicate_t predicate,
    loom_amdgpu_descriptor_ref_t* out_high_descriptor_ref,
    loom_amdgpu_descriptor_ref_t* out_low_descriptor_ref,
    loom_amdgpu_descriptor_ref_t* out_combine_descriptor_ref,
    bool* out_needs_high_equal) {
  *out_high_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  *out_low_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  *out_combine_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  *out_needs_high_equal = false;
  if (predicate >= LOOM_SCALAR_CMPI_PREDICATE_COUNT_) {
    return false;
  }
  const loom_amdgpu_i64_compare_predicate_descriptor_row_t* row =
      &kAmdgpuI64ComparePredicateDescriptorRows[predicate];
  if (row->high_descriptor_ref == LOOM_AMDGPU_DESCRIPTOR_REF_NONE ||
      row->low_descriptor_ref == LOOM_AMDGPU_DESCRIPTOR_REF_NONE ||
      row->combine_descriptor_ref == LOOM_AMDGPU_DESCRIPTOR_REF_NONE) {
    return false;
  }
  *out_high_descriptor_ref = row->high_descriptor_ref;
  *out_low_descriptor_ref = row->low_descriptor_ref;
  *out_combine_descriptor_ref = row->combine_descriptor_ref;
  *out_needs_high_equal = iree_any_bit_set(
      row->flags, LOOM_AMDGPU_I64_COMPARE_PREDICATE_NEEDS_HIGH_EQUAL);
  return true;
}

static bool loom_amdgpu_i64_compare_descriptors_supported(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_i64_compare_plan_t* plan,
    iree_string_view_t* out_constraint_key) {
  if (!loom_amdgpu_descriptor_requirements_present(
          descriptor_set, kAmdgpuVgprMoveDescriptorRequirements,
          IREE_ARRAYSIZE(kAmdgpuVgprMoveDescriptorRequirements),
          out_constraint_key)) {
    return false;
  }
  const loom_amdgpu_descriptor_requirement_t requirements[] = {
      {
          .constraint_key = IREE_SVL("descriptor.high_compare"),
          .descriptor_ref = plan->high_descriptor_ref,
      },
      {
          .constraint_key = IREE_SVL("descriptor.low_compare"),
          .descriptor_ref = plan->low_descriptor_ref,
      },
      {
          .constraint_key = IREE_SVL("descriptor.combine"),
          .descriptor_ref = plan->combine_descriptor_ref,
      },
  };
  if (!loom_amdgpu_descriptor_requirements_present(descriptor_set, requirements,
                                                   IREE_ARRAYSIZE(requirements),
                                                   out_constraint_key)) {
    return false;
  }
  if (plan->needs_high_equal) {
    return loom_amdgpu_descriptor_requirements_present(
        descriptor_set, kAmdgpuI64CompareHighEqualDescriptorRequirements,
        IREE_ARRAYSIZE(kAmdgpuI64CompareHighEqualDescriptorRequirements),
        out_constraint_key);
  }
  return true;
}

static bool loom_amdgpu_iota_i32_lane_value(int64_t base, int64_t step,
                                            uint32_t lane, int64_t* out_value) {
  *out_value = 0;
  int64_t scaled_step = 0;
  if (!iree_checked_mul_i64((int64_t)lane, step, &scaled_step)) {
    return false;
  }
  int64_t value = 0;
  if (!iree_checked_add_i64(base, scaled_step, &value) || value < INT32_MIN ||
      value > INT32_MAX) {
    return false;
  }
  *out_value = value;
  return true;
}

static bool loom_amdgpu_value_type_can_materialize_as_vgpr_i32(
    const loom_module_t* module, loom_value_id_t value_id) {
  return loom_amdgpu_type_is_i32(loom_module_value_type(module, value_id));
}

static bool loom_amdgpu_vector_iota_has_lane_offsets_in_i32_range(
    uint32_t lane_count, int64_t step) {
  for (uint32_t i = 1; i < lane_count; ++i) {
    int64_t lane_offset = 0;
    if (!iree_checked_mul_i64((int64_t)i, step, &lane_offset) ||
        lane_offset < INT32_MIN || lane_offset > INT32_MAX) {
      return false;
    }
  }
  return true;
}

static bool loom_amdgpu_vector_iota_needs_dynamic_add(uint32_t lane_count,
                                                      int64_t step) {
  return lane_count > 1 && step != 0;
}

static bool loom_amdgpu_vector_iota_needs_dynamic_step_shift(
    uint32_t lane_count) {
  for (uint32_t i = 2; i < lane_count; ++i) {
    if (loom_amdgpu_u32_is_power_of_two(i)) {
      return true;
    }
  }
  return false;
}

static bool loom_amdgpu_vector_iota_needs_dynamic_step_multiply(
    uint32_t lane_count) {
  for (uint32_t i = 2; i < lane_count; ++i) {
    if (!loom_amdgpu_u32_is_power_of_two(i)) {
      return true;
    }
  }
  return false;
}

static bool loom_amdgpu_vector_iota_source_supported(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_low_descriptor_set_t* descriptor_set, const loom_op_t* source_op,
    iree_string_view_t* out_constraint_key) {
  *out_constraint_key = IREE_SV("vector_iota.i32_static_elements");
  const loom_value_id_t result = loom_vector_iota_result(source_op);
  const uint32_t element_count = loom_amdgpu_vector_i32_register_count(
      loom_module_value_type(module, result));
  if (element_count == 0) {
    return false;
  }

  *out_constraint_key = IREE_SV("vector_iota.i32_operands");
  const loom_value_id_t base = loom_vector_iota_base(source_op);
  const loom_value_id_t step = loom_vector_iota_step(source_op);
  if (!loom_amdgpu_value_type_can_materialize_as_vgpr_i32(module, base) ||
      !loom_amdgpu_value_type_can_materialize_as_vgpr_i32(module, step)) {
    return false;
  }

  int64_t base_value = 0;
  int64_t step_value = 0;
  const bool has_static_base =
      fact_table != NULL &&
      loom_amdgpu_value_facts_as_exact_i32(
          loom_value_fact_table_lookup(fact_table, base), &base_value);
  const bool has_static_step =
      fact_table != NULL &&
      loom_amdgpu_value_facts_as_exact_i32(
          loom_value_fact_table_lookup(fact_table, step), &step_value);
  if (has_static_base && has_static_step) {
    *out_constraint_key = IREE_SV("vector_iota.i32_lane_range");
    for (uint32_t i = 0; i < element_count; ++i) {
      int64_t lane_value = 0;
      if (!loom_amdgpu_iota_i32_lane_value(base_value, step_value, i,
                                           &lane_value)) {
        return false;
      }
    }
    return loom_amdgpu_descriptor_requirement_present(
        descriptor_set, IREE_SV("descriptor.v_mov_b32"),
        LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, out_constraint_key);
  }

  if (!loom_amdgpu_descriptor_requirements_present(
          descriptor_set, kAmdgpuVgprMoveDescriptorRequirements,
          IREE_ARRAYSIZE(kAmdgpuVgprMoveDescriptorRequirements),
          out_constraint_key)) {
    return false;
  }

  if (has_static_step) {
    *out_constraint_key = IREE_SV("vector_iota.i32_lane_range");
    if (!loom_amdgpu_vector_iota_has_lane_offsets_in_i32_range(element_count,
                                                               step_value)) {
      return false;
    }
    if (loom_amdgpu_vector_iota_needs_dynamic_add(element_count, step_value)) {
      return loom_amdgpu_descriptor_requirement_present(
          descriptor_set, IREE_SV("descriptor.v_add_u32_lit"),
          LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT, out_constraint_key);
    }
    return true;
  }

  if (!loom_amdgpu_descriptor_requirement_present(
          descriptor_set, IREE_SV("descriptor.v_add_u32"),
          LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32, out_constraint_key)) {
    return false;
  }
  if (loom_amdgpu_vector_iota_needs_dynamic_step_shift(element_count)) {
    if (!loom_amdgpu_descriptor_requirement_present(
            descriptor_set, IREE_SV("descriptor.v_lshlrev_b32_lit"),
            LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, out_constraint_key)) {
      return false;
    }
  }
  if (loom_amdgpu_vector_iota_needs_dynamic_step_multiply(element_count)) {
    return loom_amdgpu_descriptor_requirement_present(
        descriptor_set, IREE_SV("descriptor.v_mul_lo_u32"),
        LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_LO_U32, out_constraint_key);
  }
  return true;
}

static iree_status_t loom_amdgpu_resolve_imm32_descriptor(
    loom_low_lower_context_t* context,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_low_lower_resolved_descriptor_t* out_descriptor,
    loom_string_id_t* out_imm32_attr_name_id, bool* out_present) {
  *out_imm32_attr_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, descriptor_ref, out_descriptor, out_present));
  if (!*out_present) {
    return iree_ok_status();
  }
  return loom_amdgpu_intern(context, IREE_SV("imm32"), out_imm32_attr_name_id);
}

static iree_status_t loom_amdgpu_select_u32_bit_pattern_constant_plan(
    loom_low_lower_context_t* context, uint32_t bit_pattern,
    loom_value_id_t result, loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_amdgpu_constant_plan_t* out_plan, bool* out_selected) {
  *out_selected = false;
  loom_low_lower_resolved_descriptor_t descriptor = {0};
  loom_string_id_t imm32_attr_name_id = LOOM_STRING_ID_INVALID;
  bool descriptor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_imm32_descriptor(
      context, descriptor_ref, &descriptor, &imm32_attr_name_id,
      &descriptor_present));
  if (!descriptor_present) {
    return iree_ok_status();
  }
  *out_plan = (loom_amdgpu_constant_plan_t){
      .kind = LOOM_AMDGPU_CONSTANT_PLAN_KIND_U32_BITS,
      .result = result,
      .descriptor = descriptor,
      .imm32_attr_name_id = imm32_attr_name_id,
      .register_count = 1,
  };
  out_plan->bit_patterns[0] = bit_pattern;
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_select_i32_constant_plan(
    loom_low_lower_context_t* context, loom_attribute_t value,
    loom_value_id_t result, loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_amdgpu_constant_plan_t* out_plan, bool* out_selected) {
  if (!loom_amdgpu_attr_is_i32_immediate(value)) {
    *out_selected = false;
    return iree_ok_status();
  }
  return loom_amdgpu_select_u32_bit_pattern_constant_plan(
      context, (uint32_t)(int32_t)value.i64, result, descriptor_ref, out_plan,
      out_selected);
}

static uint32_t loom_amdgpu_integer_sign_extended_bits(int64_t value,
                                                       uint32_t bit_count) {
  IREE_ASSERT_GT(bit_count, 0);
  IREE_ASSERT_LT(bit_count, 32);
  const uint32_t low_bits = loom_amdgpu_integer_low_bits(value, bit_count);
  const uint32_t sign_bit = UINT32_C(1) << (bit_count - 1u);
  if ((low_bits & sign_bit) == 0) {
    return low_bits;
  }
  return low_bits | ~loom_amdgpu_integer_bit_mask(bit_count);
}

static bool loom_amdgpu_type_narrow_integer_bit_count(loom_type_t type,
                                                      uint32_t* out_bit_count) {
  *out_bit_count = 0;
  if (!loom_type_is_scalar(type)) {
    return false;
  }
  const loom_scalar_type_t scalar_type = loom_type_element_type(type);
  if (scalar_type != LOOM_SCALAR_TYPE_I8 &&
      scalar_type != LOOM_SCALAR_TYPE_I16) {
    return false;
  }
  const int32_t bit_count = loom_scalar_type_bitwidth(scalar_type);
  if (bit_count <= 0 || bit_count >= 32) {
    return false;
  }
  *out_bit_count = (uint32_t)bit_count;
  return true;
}

static iree_status_t loom_amdgpu_select_narrow_integer_constant_plan(
    loom_low_lower_context_t* context, loom_attribute_t value,
    loom_value_id_t result, loom_type_t result_type,
    loom_amdgpu_constant_plan_t* out_plan, bool* out_selected) {
  *out_selected = false;
  uint32_t bit_count = 0;
  if (!loom_amdgpu_type_narrow_integer_bit_count(result_type, &bit_count) ||
      value.kind != LOOM_ATTR_I64) {
    return iree_ok_status();
  }
  return loom_amdgpu_select_u32_bit_pattern_constant_plan(
      context, loom_amdgpu_integer_sign_extended_bits(value.i64, bit_count),
      result, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, out_plan, out_selected);
}

static iree_status_t loom_amdgpu_select_i1_constant_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t result, loom_amdgpu_constant_plan_t* out_plan,
    bool* out_selected) {
  *out_selected = false;

  bool value = false;
  if (!loom_amdgpu_value_as_i1_constant(context, result, &value)) {
    return iree_ok_status();
  }

  loom_type_t result_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op, result,
                                                   &result_low_type));
  const bool is_scc = loom_amdgpu_low_type_is_register_class_count(
      context, result_low_type, LOOM_AMDGPU_REG_CLASS_ID_SCC, 1);
  if (is_scc) {
    loom_low_lower_resolved_descriptor_t descriptor = {0};
    bool descriptor_present = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
        context,
        value ? LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_EQ_I32
              : LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_LG_I32,
        &descriptor, &descriptor_present));
    if (!descriptor_present) {
      return iree_ok_status();
    }
    loom_low_lower_resolved_descriptor_t zero_descriptor = {0};
    loom_string_id_t imm32_attr_name_id = LOOM_STRING_ID_INVALID;
    bool zero_descriptor_present = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_imm32_descriptor(
        context, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, &zero_descriptor,
        &imm32_attr_name_id, &zero_descriptor_present));
    if (!zero_descriptor_present) {
      return iree_ok_status();
    }
    *out_plan = (loom_amdgpu_constant_plan_t){
        .kind = LOOM_AMDGPU_CONSTANT_PLAN_KIND_I1_SCC,
        .result = result,
        .descriptor = descriptor,
        .zero_descriptor = zero_descriptor,
        .imm32_attr_name_id = imm32_attr_name_id,
        .i1_value = value,
    };
    *out_selected = true;
    return iree_ok_status();
  }

  const bool is_native_mask = loom_amdgpu_low_type_is_register_class_count(
      context, result_low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);
  if (!is_native_mask) {
    return iree_ok_status();
  }

  if (value) {
    loom_low_lower_resolved_descriptor_t descriptor = {0};
    bool descriptor_present = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
        context, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC_READ, &descriptor,
        &descriptor_present));
    if (!descriptor_present) {
      return iree_ok_status();
    }
    *out_plan = (loom_amdgpu_constant_plan_t){
        .kind = LOOM_AMDGPU_CONSTANT_PLAN_KIND_I1_MASK,
        .result = result,
        .descriptor = descriptor,
        .i1_value = value,
    };
    *out_selected = true;
    return iree_ok_status();
  }

  loom_low_lower_resolved_descriptor_t zero_descriptor = {0};
  loom_string_id_t imm32_attr_name_id = LOOM_STRING_ID_INVALID;
  bool zero_descriptor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_imm32_descriptor(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, &zero_descriptor,
      &imm32_attr_name_id, &zero_descriptor_present));
  if (!zero_descriptor_present) {
    return iree_ok_status();
  }
  *out_plan = (loom_amdgpu_constant_plan_t){
      .kind = LOOM_AMDGPU_CONSTANT_PLAN_KIND_I1_MASK,
      .result = result,
      .zero_descriptor = zero_descriptor,
      .imm32_attr_name_id = imm32_attr_name_id,
      .i1_value = value,
  };
  *out_selected = true;
  return iree_ok_status();
}

static void loom_amdgpu_repeat_first_constant_bit_pattern(
    loom_amdgpu_constant_plan_t* plan, uint32_t register_count) {
  IREE_ASSERT_GT(register_count, 0);
  IREE_ASSERT_LE(register_count, LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES);
  const uint32_t bit_pattern = plan->bit_patterns[0];
  plan->register_count = register_count;
  for (uint32_t i = 1; i < register_count; ++i) {
    plan->bit_patterns[i] = bit_pattern;
  }
}

static uint32_t loom_amdgpu_repeated_integer_lane_pattern(uint32_t lane_bits,
                                                          uint32_t bit_count) {
  IREE_ASSERT(bit_count == 8 || bit_count == 16);
  const uint32_t masked_lane_bits =
      lane_bits & loom_amdgpu_integer_bit_mask(bit_count);
  uint32_t bit_pattern = 0;
  for (uint32_t bit_offset = 0; bit_offset < 32; bit_offset += bit_count) {
    bit_pattern |= masked_lane_bits << bit_offset;
  }
  return bit_pattern;
}

static iree_status_t loom_amdgpu_select_f32_constant_plan(
    loom_low_lower_context_t* context, loom_attribute_t value,
    loom_value_id_t result, uint32_t register_count,
    loom_amdgpu_constant_plan_t* out_plan, bool* out_selected) {
  *out_selected = false;
  if (!loom_amdgpu_attr_is_f32_immediate(value)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_u32_bit_pattern_constant_plan(
      context, loom_amdgpu_attr_f32_bit_pattern(value), result,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, out_plan, out_selected));
  if (!*out_selected) {
    return iree_ok_status();
  }
  loom_amdgpu_repeat_first_constant_bit_pattern(out_plan, register_count);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_select_packed_integer_constant_plan(
    loom_low_lower_context_t* context, loom_type_t result_type,
    loom_attribute_t value, loom_value_id_t result,
    loom_amdgpu_constant_plan_t* out_plan, bool* out_selected) {
  *out_selected = false;
  if (value.kind != LOOM_ATTR_I64) {
    return iree_ok_status();
  }
  loom_amdgpu_vector_storage_t storage = {0};
  if (!loom_amdgpu_type_vector_storage(result_type, &storage) ||
      storage.kind != LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_INTEGER ||
      (storage.element_bit_count != 8 && storage.element_bit_count != 16)) {
    return iree_ok_status();
  }
  const uint32_t lane_bits =
      loom_amdgpu_integer_low_bits(value.i64, storage.element_bit_count);
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_u32_bit_pattern_constant_plan(
      context,
      loom_amdgpu_repeated_integer_lane_pattern(lane_bits,
                                                storage.element_bit_count),
      result, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, out_plan, out_selected));
  if (!*out_selected) {
    return iree_ok_status();
  }
  loom_amdgpu_repeat_first_constant_bit_pattern(out_plan,
                                                storage.register_count);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_select_packed_16bit_float_constant_plan(
    loom_low_lower_context_t* context, loom_type_t result_type,
    loom_attribute_t value, loom_value_id_t result,
    loom_amdgpu_constant_plan_t* out_plan, bool* out_selected) {
  *out_selected = false;
  uint32_t unused_payload_bit_count = 0;
  uint32_t register_count = 0;
  if (!loom_amdgpu_type_packed_16bit_float_storage(
          result_type, &unused_payload_bit_count, &register_count) ||
      !loom_amdgpu_attr_is_16bit_float_immediate(value)) {
    return iree_ok_status();
  }
  const uint32_t lane_bit_pattern = loom_amdgpu_attr_16bit_float_bit_pattern(
      loom_type_element_type(result_type), value);
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_u32_bit_pattern_constant_plan(
      context, lane_bit_pattern | (lane_bit_pattern << 16), result,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, out_plan, out_selected));
  if (!*out_selected) {
    return iree_ok_status();
  }
  loom_amdgpu_repeat_first_constant_bit_pattern(out_plan, register_count);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_index_constant_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_constant_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_constant_plan_t){0};
  *out_selected = false;
  const loom_value_id_t result = loom_index_constant_result(source_op);
  const loom_attribute_t value = loom_index_constant_value(source_op);
  if (!loom_amdgpu_value_is_address_scalar(context, result) ||
      !loom_amdgpu_attr_is_u32_address_immediate(value)) {
    return iree_ok_status();
  }
  bool result_prefers_vgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_context_value_prefers_vgpr(
      context, result, &result_prefers_vgpr));
  const loom_amdgpu_descriptor_ref_t descriptor_ref =
      result_prefers_vgpr ? LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32
                          : LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32;
  return loom_amdgpu_select_u32_bit_pattern_constant_plan(
      context, (uint32_t)value.i64, result, descriptor_ref, out_plan,
      out_selected);
}

static iree_status_t loom_amdgpu_select_i64_constant_plan(
    loom_low_lower_context_t* context, loom_attribute_t value,
    loom_value_id_t result, loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_amdgpu_constant_plan_t* out_plan, bool* out_selected) {
  const uint64_t bit_pattern = (uint64_t)value.i64;
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_u32_bit_pattern_constant_plan(
      context, (uint32_t)bit_pattern, result, descriptor_ref, out_plan,
      out_selected));
  if (!*out_selected) {
    return iree_ok_status();
  }
  out_plan->register_count = 2;
  out_plan->bit_patterns[1] = (uint32_t)(bit_pattern >> 32);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_scalar_constant_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_constant_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_constant_plan_t){0};
  *out_selected = false;
  const loom_value_id_t result = loom_scalar_constant_result(source_op);
  const loom_attribute_t value = loom_scalar_constant_value(source_op);
  const loom_type_t result_type =
      loom_module_value_type(loom_low_lower_context_module(context), result);
  if (loom_amdgpu_type_is_i1(result_type)) {
    return loom_amdgpu_select_i1_constant_plan(context, source_op, result,
                                               out_plan, out_selected);
  }
  if (loom_amdgpu_value_is_f32(context, result)) {
    return loom_amdgpu_select_f32_constant_plan(
        context, value, result, /*register_count=*/1, out_plan, out_selected);
  }
  if (loom_amdgpu_value_is_f16_or_bf16(context, result)) {
    if (!loom_amdgpu_attr_is_16bit_float_immediate(value)) {
      return iree_ok_status();
    }
    const loom_type_t result_type =
        loom_module_value_type(loom_low_lower_context_module(context), result);
    return loom_amdgpu_select_u32_bit_pattern_constant_plan(
        context,
        loom_amdgpu_attr_16bit_float_bit_pattern(
            loom_type_element_type(result_type), value),
        result, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, out_plan, out_selected);
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_narrow_integer_constant_plan(
      context, value, result, result_type, out_plan, out_selected));
  if (*out_selected) {
    return iree_ok_status();
  }
  if (loom_amdgpu_type_is_i64(result_type)) {
    bool result_prefers_vgpr = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_context_value_prefers_vgpr(
        context, result, &result_prefers_vgpr));
    const loom_amdgpu_descriptor_ref_t descriptor_ref =
        result_prefers_vgpr ? LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32
                            : LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32;
    return loom_amdgpu_select_i64_constant_plan(
        context, value, result, descriptor_ref, out_plan, out_selected);
  }
  if (!loom_amdgpu_type_is_i32(result_type)) {
    return iree_ok_status();
  }
  bool result_prefers_vgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_context_value_prefers_vgpr(
      context, result, &result_prefers_vgpr));
  const loom_amdgpu_descriptor_ref_t descriptor_ref =
      result_prefers_vgpr ? LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32
                          : LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32;
  return loom_amdgpu_select_i32_constant_plan(
      context, value, result, descriptor_ref, out_plan, out_selected);
}

iree_status_t loom_amdgpu_select_vector_constant_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_constant_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_constant_plan_t){0};
  *out_selected = false;
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t result = loom_vector_constant_result(source_op);
  const loom_type_t result_type = loom_module_value_type(module, result);
  const loom_attribute_t value = loom_vector_constant_value(source_op);
  const uint32_t i32_register_count =
      loom_amdgpu_vector_i32_register_count(result_type);
  if (i32_register_count != 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_select_i32_constant_plan(
        context, value, result, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, out_plan,
        out_selected));
    if (!*out_selected) {
      return iree_ok_status();
    }
    loom_amdgpu_repeat_first_constant_bit_pattern(out_plan, i32_register_count);
    return iree_ok_status();
  }
  const uint32_t f32_register_count =
      loom_amdgpu_vector_f32_register_count(result_type);
  if (f32_register_count != 0) {
    return loom_amdgpu_select_f32_constant_plan(
        context, value, result, f32_register_count, out_plan, out_selected);
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_packed_16bit_float_constant_plan(
      context, result_type, value, result, out_plan, out_selected));
  if (*out_selected) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_packed_integer_constant_plan(
      context, result_type, value, result, out_plan, out_selected));
  if (*out_selected) {
    return iree_ok_status();
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_select_vector_iota_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_iota_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_vector_iota_plan_t){0};
  *out_selected = false;
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t result = loom_vector_iota_result(source_op);
  const uint32_t element_count = loom_amdgpu_vector_i32_register_count(
      loom_module_value_type(module, result));
  if (element_count == 0) {
    return iree_ok_status();
  }
  const loom_value_id_t base_id = loom_vector_iota_base(source_op);
  const loom_value_id_t step_id = loom_vector_iota_step(source_op);
  int64_t base = 0;
  int64_t step = 0;
  const bool has_static_base =
      loom_amdgpu_value_as_i32_constant(context, base_id, &base);
  const bool has_static_step =
      loom_amdgpu_value_as_i32_constant(context, step_id, &step);
  out_plan->base = base_id;
  out_plan->step = step_id;
  out_plan->result = result;
  out_plan->lane_count = element_count;
  if (has_static_base) {
    out_plan->flags |= LOOM_AMDGPU_VECTOR_IOTA_PLAN_BASE_EXACT;
    out_plan->exact_base = (int32_t)base;
  }
  if (has_static_step) {
    out_plan->flags |= LOOM_AMDGPU_VECTOR_IOTA_PLAN_STEP_EXACT;
    out_plan->exact_step = (int32_t)step;
  }

  iree_string_view_t constraint_key = iree_string_view_empty();
  if (!loom_amdgpu_vector_iota_source_supported(
          module, loom_low_lower_context_fact_table(context),
          loom_low_lower_context_descriptor_set(context), source_op,
          &constraint_key)) {
    return iree_ok_status();
  }

  if (!has_static_base || !has_static_step) {
    *out_selected = true;
    return iree_ok_status();
  }

  for (uint32_t i = 0; i < element_count; ++i) {
    int64_t lane_value = 0;
    if (!loom_amdgpu_iota_i32_lane_value(base, step, i, &lane_value)) {
      return iree_ok_status();
    }
    out_plan->lane_bit_patterns[i] = (uint32_t)(int32_t)lane_value;
  }
  bool descriptor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_imm32_descriptor(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, &out_plan->descriptor,
      &out_plan->imm32_attr_name_id, &descriptor_present));
  if (!descriptor_present) {
    return iree_ok_status();
  }
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_low_legality_verify_vector_iota(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  (void)provider;
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  const loom_module_t* module = loom_target_low_legality_module(context);
  if (loom_amdgpu_value_only_feeds_vector_atomic_offsets(
          module, loom_vector_iota_result(op))) {
    return iree_ok_status();
  }

  iree_string_view_t constraint_key = iree_string_view_empty();
  if (loom_amdgpu_vector_iota_source_supported(
          module, loom_target_low_legality_fact_table(context),
          loom_target_low_legality_descriptor_set(context), op,
          &constraint_key)) {
    return iree_ok_status();
  }
  return loom_amdgpu_low_legality_reject(context, op, constraint_key);
}

iree_status_t loom_amdgpu_low_legality_verify_vector_from_elements(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  (void)provider;
  *out_handled = false;
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  const loom_module_t* module = loom_target_low_legality_module(context);
  if (!loom_amdgpu_value_only_feeds_vector_atomic_offsets(
          module, loom_vector_from_elements_result(op))) {
    return iree_ok_status();
  }
  *out_handled = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_low_legality_verify_address_compare(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  (void)provider;
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_target_low_legality_module(context);
  if (!loom_amdgpu_address_cmp_needs_64bit(
          module, loom_target_low_legality_fact_table(context), op)) {
    return iree_ok_status();
  }

  bool result_is_native_mask = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_low_legality_value_is_native_i1_mask(
      context, loom_index_cmp_result(op), &result_is_native_mask));
  if (!result_is_native_mask) {
    return iree_ok_status();
  }
  *out_handled = true;

  loom_amdgpu_i64_compare_plan_t plan = {
      .lhs = loom_index_cmp_lhs(op),
      .rhs = loom_index_cmp_rhs(op),
      .result = loom_index_cmp_result(op),
  };
  loom_scalar_cmpi_predicate_t predicate;
  if (!loom_amdgpu_index_cmp_predicate_to_scalar(loom_index_cmp_predicate(op),
                                                 &predicate) ||
      !loom_amdgpu_i64_compare_predicate_descriptors(
          predicate, &plan.high_descriptor_ref, &plan.low_descriptor_ref,
          &plan.combine_descriptor_ref, &plan.needs_high_equal)) {
    return loom_amdgpu_low_legality_reject(context, op, IREE_SV("predicate"));
  }
  iree_string_view_t constraint_key = iree_string_view_empty();
  if (loom_amdgpu_i64_compare_descriptors_supported(
          loom_target_low_legality_descriptor_set(context), &plan,
          &constraint_key)) {
    return iree_ok_status();
  }
  return loom_amdgpu_low_legality_reject(context, op, constraint_key);
}

iree_status_t loom_amdgpu_low_legality_verify_scalar_cmpi_i64(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  (void)provider;
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_target_low_legality_module(context);
  if (!loom_amdgpu_scalar_cmpi_has_i64_operands(module, op)) {
    return iree_ok_status();
  }

  bool result_is_native_mask = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_low_legality_value_is_native_i1_mask(
      context, loom_scalar_cmpi_result(op), &result_is_native_mask));
  if (!result_is_native_mask) {
    return iree_ok_status();
  }
  *out_handled = true;

  loom_amdgpu_i64_compare_plan_t plan = {
      .lhs = loom_scalar_cmpi_lhs(op),
      .rhs = loom_scalar_cmpi_rhs(op),
      .result = loom_scalar_cmpi_result(op),
  };
  if (!loom_amdgpu_i64_compare_predicate_descriptors(
          (loom_scalar_cmpi_predicate_t)loom_scalar_cmpi_predicate(op),
          &plan.high_descriptor_ref, &plan.low_descriptor_ref,
          &plan.combine_descriptor_ref, &plan.needs_high_equal)) {
    return loom_amdgpu_low_legality_reject(context, op, IREE_SV("predicate"));
  }
  iree_string_view_t constraint_key = iree_string_view_empty();
  if (loom_amdgpu_i64_compare_descriptors_supported(
          loom_target_low_legality_descriptor_set(context), &plan,
          &constraint_key)) {
    return iree_ok_status();
  }
  return loom_amdgpu_low_legality_reject(context, op, constraint_key);
}

static iree_status_t loom_amdgpu_select_arithmetic_contract(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    bool* out_selected) {
  *out_selected = false;
  loom_low_lower_rule_selection_t selection = {0};
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_set_select_contract(
      context, &loom_amdgpu_arithmetic_lower_rule_set, source_op, &selection));
  *out_selected = selection.rule != NULL;
  return iree_ok_status();
}

static void loom_amdgpu_vector_extract_plan_from_accepted_op(
    const loom_module_t* module, const loom_op_t* source_op,
    loom_amdgpu_vector_extract_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_vector_extract_plan_t){0};
  IREE_ASSERT(loom_vector_extract_isa(source_op));
  loom_attribute_t static_indices =
      loom_vector_extract_static_indices(source_op);
  IREE_ASSERT_EQ(static_indices.kind, LOOM_ATTR_I64_ARRAY);
  const loom_value_slice_t indices = loom_vector_extract_indices(source_op);

  const loom_value_id_t source = loom_vector_extract_source(source_op);
  const loom_value_id_t result = loom_vector_extract_result(source_op);
  const loom_type_t source_type = loom_module_value_type(module, source);
  const loom_type_t result_type = loom_module_value_type(module, result);

  loom_amdgpu_vector_storage_t source_storage = {0};
  const bool source_storage_matches =
      loom_amdgpu_type_vector_storage(source_type, &source_storage);
  IREE_ASSERT(source_storage_matches);

  uint32_t result_register_count = 0;
  uint32_t result_lane_count = 1;
  bool sign_extend_packed_lane = false;
  if (loom_type_is_scalar(result_type)) {
    IREE_ASSERT_EQ(loom_type_element_type(result_type),
                   source_storage.element_type);
    switch (source_storage.kind) {
      case LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_32BIT:
      case LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_64BIT:
      case LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_16BIT_FLOAT:
      case LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_8BIT_FLOAT:
        result_register_count = source_storage.element_register_count;
        break;
      case LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_INTEGER:
        result_register_count = source_storage.element_register_count;
        sign_extend_packed_lane = true;
        break;
      case LOOM_AMDGPU_VECTOR_STORAGE_KIND_NONE:
      case LOOM_AMDGPU_VECTOR_STORAGE_KIND_I1_MASK:
      default:
        IREE_ASSERT_UNREACHABLE(
            "accepted AMDGPU vector.extract has unsupported source storage");
        IREE_BUILTIN_UNREACHABLE();
    }
  } else {
    loom_amdgpu_vector_storage_t result_storage = {0};
    const bool result_storage_matches =
        loom_amdgpu_type_vector_storage(result_type, &result_storage);
    IREE_ASSERT(result_storage_matches);
    IREE_ASSERT_EQ(result_storage.kind, source_storage.kind);
    IREE_ASSERT_EQ(result_storage.element_type, source_storage.element_type);
    switch (source_storage.kind) {
      case LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_32BIT:
      case LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_64BIT:
        result_register_count = result_storage.register_count;
        result_lane_count = result_storage.element_count;
        break;
      case LOOM_AMDGPU_VECTOR_STORAGE_KIND_NONE:
      case LOOM_AMDGPU_VECTOR_STORAGE_KIND_I1_MASK:
      case LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_16BIT_FLOAT:
      case LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_8BIT_FLOAT:
      case LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_INTEGER:
      default:
        IREE_ASSERT_UNREACHABLE(
            "accepted AMDGPU vector.extract has unsupported result storage");
        IREE_BUILTIN_UNREACHABLE();
    }
  }
  IREE_ASSERT_NE(source_storage.element_count, 0u);
  IREE_ASSERT_NE(source_storage.register_count, 0u);
  IREE_ASSERT_NE(result_register_count, 0u);

  IREE_ASSERT_LE(static_indices.count, loom_type_rank(source_type));
  if (loom_type_is_scalar(result_type)) {
    IREE_ASSERT_EQ(static_indices.count, loom_type_rank(source_type));
  } else {
    IREE_ASSERT_EQ(static_indices.count + loom_type_rank(result_type),
                   loom_type_rank(source_type));
  }

  bool is_dynamic = false;
  uint32_t lane_offset = 0;
  loom_value_id_t dynamic_index = LOOM_VALUE_ID_INVALID;
  if (static_indices.count == 1 && static_indices.i64_array[0] == INT64_MIN) {
    IREE_ASSERT_EQ(indices.count, 1u);
    IREE_ASSERT(loom_type_is_scalar(result_type));
    is_dynamic = true;
    dynamic_index = indices.values[0];
  } else {
    IREE_ASSERT_EQ(indices.count, 0u);
    int64_t source_indices[LOOM_TYPE_MAX_RANK] = {0};
    for (uint16_t i = 0; i < static_indices.count; ++i) {
      const int64_t index = static_indices.i64_array[i];
      IREE_ASSERT_GE(index, 0);
      source_indices[i] = index;
    }
    const bool has_static_lane_offset =
        loom_amdgpu_static_vector_flat_register_from_indices(
            source_type, source_indices, &lane_offset);
    IREE_ASSERT(has_static_lane_offset);
    IREE_ASSERT_LE((uint64_t)lane_offset + result_lane_count,
                   (uint64_t)source_storage.element_count);
  }

  *out_plan = (loom_amdgpu_vector_extract_plan_t){
      .source = source,
      .dynamic_index = dynamic_index,
      .result = result,
      .lane_offset = lane_offset,
      .lane_count = source_storage.element_count,
      .register_count = source_storage.register_count,
      .result_register_count = result_register_count,
      .element_register_count = source_storage.element_register_count,
      .lane_bit_count = source_storage.element_bit_count,
      .sign_extend_packed_lane = sign_extend_packed_lane,
      .is_dynamic = is_dynamic,
  };
}

iree_status_t loom_amdgpu_select_vector_extract_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_extract_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_vector_extract_plan_t){0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_select_arithmetic_contract(context, source_op, out_selected));
  if (*out_selected) {
    loom_amdgpu_vector_extract_plan_from_accepted_op(
        loom_low_lower_context_module(context), source_op, out_plan);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_select_vector_from_elements_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_from_elements_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_vector_from_elements_plan_t){0};
  *out_selected = false;
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_slice_t elements =
      loom_vector_from_elements_elements(source_op);
  if (elements.count == 0) {
    return iree_ok_status();
  }
  const loom_value_id_t result = loom_vector_from_elements_result(source_op);
  const loom_type_t result_type = loom_module_value_type(module, result);
  loom_amdgpu_vector_storage_t storage = {0};
  if (!loom_amdgpu_type_vector_storage(result_type, &storage) ||
      elements.count != storage.element_count ||
      elements.count > IREE_ARRAYSIZE(out_plan->payload.elements)) {
    return iree_ok_status();
  }
  for (uint32_t i = 0; i < elements.count; ++i) {
    const loom_value_id_t element = elements.values[i];
    const loom_type_t source_type = loom_module_value_type(module, element);
    if (!loom_type_is_scalar(source_type) ||
        loom_type_element_type(source_type) != storage.element_type) {
      return iree_ok_status();
    }
    if (storage.element_type == LOOM_SCALAR_TYPE_I32) {
      bool can_materialize = false;
      IREE_RETURN_IF_ERROR(loom_amdgpu_value_can_materialize_as_vgpr_i32(
          context, source_op, element, &can_materialize));
      if (!can_materialize) {
        return iree_ok_status();
      }
    }
    out_plan->payload.elements[i] = element;
  }
  out_plan->result = result;
  out_plan->element_count = elements.count;
  out_plan->storage_kind = storage.kind;
  out_plan->register_count = storage.register_count;
  out_plan->element_register_count = storage.element_register_count;
  out_plan->element_bit_count = storage.element_bit_count;
  out_plan->element_type = storage.element_type;
  loom_amdgpu_select_vector_from_elements_materialization(context, out_plan);
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_select_vector_splat_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_from_elements_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_vector_from_elements_plan_t){0};
  *out_selected = false;
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t result = loom_vector_splat_result(source_op);
  const loom_type_t result_type = loom_module_value_type(module, result);
  loom_amdgpu_vector_storage_t storage = {0};
  if (!loom_amdgpu_type_vector_storage(result_type, &storage) ||
      storage.element_count > IREE_ARRAYSIZE(out_plan->payload.elements)) {
    return iree_ok_status();
  }
  const loom_value_id_t scalar = loom_vector_splat_scalar(source_op);
  const loom_type_t scalar_type = loom_module_value_type(module, scalar);
  if (!loom_type_is_scalar(scalar_type) ||
      loom_type_element_type(scalar_type) != storage.element_type) {
    return iree_ok_status();
  }
  if (storage.element_type == LOOM_SCALAR_TYPE_I32) {
    bool can_materialize = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_value_can_materialize_as_vgpr_i32(
        context, source_op, scalar, &can_materialize));
    if (!can_materialize) {
      return iree_ok_status();
    }
  }
  for (uint32_t i = 0; i < storage.element_count; ++i) {
    out_plan->payload.elements[i] = scalar;
  }
  out_plan->result = result;
  out_plan->storage_kind = storage.kind;
  out_plan->element_count = storage.element_count;
  out_plan->register_count = storage.register_count;
  out_plan->element_register_count = storage.element_register_count;
  out_plan->element_bit_count = storage.element_bit_count;
  out_plan->element_type = storage.element_type;
  loom_amdgpu_select_vector_from_elements_materialization(context, out_plan);
  *out_selected = true;
  return iree_ok_status();
}

static const loom_op_t* loom_amdgpu_value_defining_op(
    const loom_module_t* module, loom_value_id_t value_id) {
  const loom_value_t* value = loom_module_value(module, value_id);
  return loom_value_is_block_arg(value) ? NULL : loom_value_def_op(value);
}

static iree_string_view_t loom_amdgpu_descriptor_set_key(
    const loom_low_descriptor_set_t* descriptor_set) {
  if (descriptor_set == NULL) {
    return IREE_SV("<missing>");
  }
  const iree_string_view_t descriptor_set_key = loom_low_descriptor_set_string(
      descriptor_set, descriptor_set->key_string_offset);
  return iree_string_view_is_empty(descriptor_set_key) ? IREE_SV("<empty>")
                                                       : descriptor_set_key;
}

static iree_string_view_t loom_amdgpu_fma_mix_source_kind_key(
    loom_amdgpu_fma_mix_source_kind_t source_kind) {
  switch (source_kind) {
    case LOOM_AMDGPU_FMA_MIX_SOURCE_F32:
      return IREE_SV("f32");
    case LOOM_AMDGPU_FMA_MIX_SOURCE_F16LO:
      return IREE_SV("f16lo");
    case LOOM_AMDGPU_FMA_MIX_SOURCE_F16HI:
      return IREE_SV("f16hi");
    default:
      return IREE_SV("<invalid>");
  }
}

static iree_string_view_t loom_amdgpu_fma_mix_unsupported_source_reason(
    uint32_t source_index) {
  switch (source_index) {
    case 0:
      return IREE_SV("unsupported_source0");
    case 1:
      return IREE_SV("unsupported_source1");
    case 2:
      return IREE_SV("unsupported_source2");
    default:
      return IREE_SV("unsupported_source");
  }
}

static iree_status_t loom_amdgpu_emit_fma_mix_half_result_diagnostic(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    uint32_t destination_lane_index, iree_string_view_t result_half,
    const iree_string_view_t* source_kind_names,
    iree_string_view_t decision_key, iree_string_view_t reason_key) {
  if (!iree_any_bit_set(loom_low_lower_context_diagnostic_flags(context),
                        LOOM_TARGET_LOW_LEGALITY_DIAGNOSTIC_OPERAND_FORM)) {
    return iree_ok_status();
  }

  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  iree_string_view_t descriptor_name = IREE_SV("<none>");
  if (descriptor_ref != LOOM_AMDGPU_DESCRIPTOR_REF_NONE) {
    loom_low_lower_resolved_descriptor_t descriptor = {0};
    bool descriptor_present = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
        context, descriptor_ref, &descriptor, &descriptor_present));
    descriptor_name =
        descriptor_present
            ? loom_low_descriptor_set_string(
                  descriptor_set, descriptor.descriptor->key_string_offset)
            : IREE_SV("<unavailable>");
  }

  loom_module_t* module = loom_low_lower_context_module(context);
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_lower_context_target_key(context)),
      loom_param_string(loom_low_lower_context_export_name(context)),
      loom_param_string(loom_low_lower_context_config_key(context)),
      loom_param_string(loom_low_lower_context_function_name(context)),
      loom_param_string(loom_op_name(module, source_op)),
      loom_param_string(descriptor_name),
      loom_param_u32(destination_lane_index),
      loom_param_string(result_half),
      loom_param_string(loom_amdgpu_descriptor_set_key(descriptor_set)),
      loom_param_string(source_kind_names[0]),
      loom_param_string(source_kind_names[1]),
      loom_param_string(source_kind_names[2]),
      loom_param_string(IREE_SV("f32_to_f16_fptrunc")),
      loom_param_string(decision_key),
      loom_param_string(reason_key),
  };
  return loom_low_lower_emit_error_ref(context, source_op,
                                       LOOM_ERR_AMDGPU_029_REF, params,
                                       IREE_ARRAYSIZE(params));
}

static iree_status_t loom_amdgpu_select_vector_insert_fma_mix_half_result(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_insert_select_flags_t flags,
    loom_amdgpu_vector_insert_plan_t* inout_plan) {
  if (inout_plan->element_type != LOOM_SCALAR_TYPE_F16 ||
      inout_plan->lane_bit_count != 16) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  const uint32_t destination_lane_index =
      inout_plan->is_dynamic ? UINT32_MAX : inout_plan->lane_offset;
  const iree_string_view_t result_half =
      inout_plan->is_dynamic
          ? IREE_SV("dynamic")
          : ((inout_plan->lane_offset & 1u) != 0 ? IREE_SV("high")
                                                 : IREE_SV("low"));
  iree_string_view_t source_kind_names[LOOM_AMDGPU_FMA_MIX_SOURCE_COUNT] = {
      IREE_SV("unknown"),
      IREE_SV("unknown"),
      IREE_SV("unknown"),
  };
  const bool emit_diagnostics = iree_any_bit_set(
      flags, LOOM_AMDGPU_VECTOR_INSERT_SELECT_FLAG_EMIT_DIAGNOSTICS);

  const loom_op_t* fptrunc_op =
      loom_amdgpu_value_defining_op(module, inout_plan->value);
  if (fptrunc_op == NULL || !loom_scalar_fptrunc_isa(fptrunc_op) ||
      loom_scalar_fptrunc_result(fptrunc_op) != inout_plan->value) {
    return iree_ok_status();
  }
  const loom_value_id_t fmaf_result = loom_scalar_fptrunc_input(fptrunc_op);
  if (!loom_type_equal(loom_module_value_type(module, fmaf_result),
                       loom_type_scalar(LOOM_SCALAR_TYPE_F32))) {
    if (emit_diagnostics) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fma_mix_half_result_diagnostic(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
          destination_lane_index, result_half, source_kind_names,
          IREE_SV("rejected"), IREE_SV("rounding_contract_mismatch")));
    }
    return iree_ok_status();
  }
  if (!loom_type_equal(loom_module_value_type(module, inout_plan->value),
                       loom_type_scalar(LOOM_SCALAR_TYPE_F16))) {
    if (emit_diagnostics) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fma_mix_half_result_diagnostic(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
          destination_lane_index, result_half, source_kind_names,
          IREE_SV("rejected"), IREE_SV("rounding_contract_mismatch")));
    }
    return iree_ok_status();
  }

  const loom_op_t* rounding_source_op =
      loom_amdgpu_value_defining_op(module, fmaf_result);
  loom_value_id_t operands[LOOM_AMDGPU_FMA_MIX_SOURCE_COUNT] = {
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  uint32_t operand_count = LOOM_AMDGPU_FMA_MIX_SOURCE_COUNT;
  bool has_implicit_zero_addend = false;
  if (rounding_source_op != NULL && loom_scalar_fmaf_isa(rounding_source_op) &&
      loom_scalar_fmaf_result(rounding_source_op) == fmaf_result) {
    operands[0] = loom_scalar_fmaf_a(rounding_source_op);
    operands[1] = loom_scalar_fmaf_b(rounding_source_op);
    operands[2] = loom_scalar_fmaf_c(rounding_source_op);
  } else if (rounding_source_op != NULL &&
             loom_scalar_mulf_isa(rounding_source_op) &&
             loom_scalar_mulf_result(rounding_source_op) == fmaf_result) {
    if (!loom_amdgpu_scalar_mulf_fastmath_allows_zero_add(rounding_source_op)) {
      if (emit_diagnostics) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fma_mix_half_result_diagnostic(
            context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
            destination_lane_index, result_half, source_kind_names,
            IREE_SV("rejected"),
            IREE_SV("rounding_source_mulf_requires_nnan_nsz_contract")));
      }
      return iree_ok_status();
    }
    operands[0] = loom_scalar_mulf_lhs(rounding_source_op);
    operands[1] = loom_scalar_mulf_rhs(rounding_source_op);
    operand_count = LOOM_AMDGPU_FMA_MIX_SOURCE_COUNT - 1;
    has_implicit_zero_addend = true;
  } else {
    if (emit_diagnostics) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fma_mix_half_result_diagnostic(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
          destination_lane_index, result_half, source_kind_names,
          IREE_SV("rejected"), IREE_SV("rounding_source_not_fmaf_or_mulf")));
    }
    return iree_ok_status();
  }

  loom_value_id_t sources[LOOM_AMDGPU_FMA_MIX_SOURCE_COUNT] = {
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  uint32_t source_register_offsets[LOOM_AMDGPU_FMA_MIX_SOURCE_COUNT] = {0, 0,
                                                                        0};
  loom_amdgpu_fma_mix_source_kind_t
      source_kinds[LOOM_AMDGPU_FMA_MIX_SOURCE_COUNT] = {
          LOOM_AMDGPU_FMA_MIX_SOURCE_F32,
          LOOM_AMDGPU_FMA_MIX_SOURCE_F32,
          LOOM_AMDGPU_FMA_MIX_SOURCE_F32,
      };
  for (uint32_t i = 0; i < operand_count; ++i) {
    if (!loom_amdgpu_select_fma_mix_source(module, operands[i], &sources[i],
                                           &source_kinds[i],
                                           &source_register_offsets[i])) {
      source_kind_names[i] = IREE_SV("unsupported");
      if (emit_diagnostics) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fma_mix_half_result_diagnostic(
            context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
            destination_lane_index, result_half, source_kind_names,
            IREE_SV("rejected"),
            loom_amdgpu_fma_mix_unsupported_source_reason(i)));
      }
      return iree_ok_status();
    }
  }
  if (has_implicit_zero_addend) {
    loom_amdgpu_canonicalize_mulf_mix_sources(sources, source_register_offsets,
                                              source_kinds);
  }
  for (uint32_t i = 0; i < operand_count; ++i) {
    source_kind_names[i] = loom_amdgpu_fma_mix_source_kind_key(source_kinds[i]);
  }
  if (has_implicit_zero_addend) {
    source_kind_names[2] = IREE_SV("f32_zero");
  }

  if (inout_plan->is_dynamic) {
    if (emit_diagnostics) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fma_mix_half_result_diagnostic(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
          destination_lane_index, result_half, source_kind_names,
          IREE_SV("rejected"), IREE_SV("dynamic_destination_lane")));
    }
    return iree_ok_status();
  }

  loom_amdgpu_descriptor_ref_t descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  loom_amdgpu_fma_mix_plan_flags_t plan_flags = 0;
  const bool high_result = (inout_plan->lane_offset & 1u) != 0;
  const bool selected =
      has_implicit_zero_addend
          ? loom_amdgpu_select_fma_mix_half_result_zero_addend_descriptor(
                context, source_kinds, high_result, &descriptor_ref,
                &plan_flags)
          : loom_amdgpu_select_fma_mix_half_result_descriptor(
                context, source_kinds, high_result, &descriptor_ref);
  if (!selected) {
    if (emit_diagnostics) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fma_mix_half_result_diagnostic(
          context, source_op, descriptor_ref, destination_lane_index,
          result_half, source_kind_names, IREE_SV("rejected"),
          IREE_SV("descriptor_unavailable")));
    }
    return iree_ok_status();
  }
  if (has_implicit_zero_addend) {
    source_kind_names[2] =
        iree_any_bit_set(plan_flags, LOOM_AMDGPU_FMA_MIX_PLAN_SRC2_LITERAL_ZERO)
            ? IREE_SV("f32_zero_literal")
            : IREE_SV("f32_zero_vgpr");
  }

  if (emit_diagnostics) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fma_mix_half_result_diagnostic(
        context, source_op, descriptor_ref, destination_lane_index, result_half,
        source_kind_names, IREE_SV("selected"),
        IREE_SV("static_destination_lane")));
  }

  inout_plan->value_kind =
      LOOM_AMDGPU_VECTOR_INSERT_VALUE_KIND_FMA_MIX_HALF_RESULT;
  inout_plan->fma_mix = (loom_amdgpu_fma_mix_plan_t){
      .sources = {sources[0], sources[1], sources[2]},
      .source_register_offsets = {source_register_offsets[0],
                                  source_register_offsets[1],
                                  source_register_offsets[2]},
      .result = inout_plan->value,
      .descriptor_ref = descriptor_ref,
      .source_kinds = {source_kinds[0], source_kinds[1], source_kinds[2]},
      .flags = plan_flags,
  };
  return iree_ok_status();
}

static bool loom_amdgpu_vector_from_elements_uses_fma_mix_half_result(
    const loom_amdgpu_vector_from_elements_plan_t* plan, uint32_t lane_index) {
  IREE_ASSERT_LT(lane_index, 64u);
  return iree_any_bit_set(plan->fma_mix_half_result_lane_mask,
                          UINT64_C(1) << lane_index);
}

static iree_status_t
loom_amdgpu_select_vector_from_elements_fma_mix_half_results(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_insert_select_flags_t flags,
    loom_amdgpu_vector_from_elements_plan_t* inout_plan) {
  if (inout_plan->storage_kind !=
          LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_16BIT_FLOAT ||
      inout_plan->element_type != LOOM_SCALAR_TYPE_F16 ||
      inout_plan->element_bit_count != 16) {
    return iree_ok_status();
  }

  inout_plan->fma_mix_half_result_lane_mask = 0;
  for (uint32_t i = 0; i < inout_plan->element_count; ++i) {
    loom_amdgpu_vector_insert_plan_t lane_plan = {
        .value = inout_plan->payload.elements[i],
        .dest = LOOM_VALUE_ID_INVALID,
        .dynamic_index = LOOM_VALUE_ID_INVALID,
        .result = inout_plan->result,
        .lane_offset = i,
        .lane_count = inout_plan->element_count,
        .register_count = inout_plan->register_count,
        .lane_bit_count = 16,
        .element_type = LOOM_SCALAR_TYPE_F16,
        .value_kind = LOOM_AMDGPU_VECTOR_INSERT_VALUE_KIND_DEFAULT,
        .fma_mix = {0},
        .is_dynamic = false,
    };
    IREE_RETURN_IF_ERROR(loom_amdgpu_select_vector_insert_fma_mix_half_result(
        context, source_op, flags, &lane_plan));
    if (lane_plan.value_kind !=
        LOOM_AMDGPU_VECTOR_INSERT_VALUE_KIND_FMA_MIX_HALF_RESULT) {
      continue;
    }
    if (inout_plan->fma_mix_half_results == NULL) {
      IREE_RETURN_IF_ERROR(loom_low_lower_allocate_scratch_array(
          context, inout_plan->element_count,
          sizeof(*inout_plan->fma_mix_half_results),
          (void**)&inout_plan->fma_mix_half_results));
    }
    inout_plan->fma_mix_half_result_lane_mask |= UINT64_C(1) << i;
    inout_plan->fma_mix_half_results[i] = lane_plan.fma_mix;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_select_vector_insert_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_insert_select_flags_t flags,
    loom_amdgpu_vector_insert_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_vector_insert_plan_t){0};
  *out_selected = false;
  loom_attribute_t static_indices =
      loom_vector_insert_static_indices(source_op);
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY || static_indices.count != 1) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t value = loom_vector_insert_value(source_op);
  const loom_value_id_t dest = loom_vector_insert_dest(source_op);
  const loom_value_id_t result = loom_vector_insert_result(source_op);
  const loom_type_t value_type = loom_module_value_type(module, value);
  const loom_type_t dest_type = loom_module_value_type(module, dest);
  const loom_type_t result_type = loom_module_value_type(module, result);
  if (!loom_type_equal(dest_type, result_type)) {
    return iree_ok_status();
  }

  loom_amdgpu_vector_storage_t storage = {0};
  if (!loom_amdgpu_type_vector_storage(dest_type, &storage) ||
      storage.element_count == 0 || storage.register_count == 0 ||
      !loom_type_is_scalar(value_type)) {
    return iree_ok_status();
  }
  const loom_scalar_type_t element_type = loom_type_element_type(dest_type);
  if (loom_type_element_type(value_type) != element_type) {
    return iree_ok_status();
  }
  if (element_type != LOOM_SCALAR_TYPE_I32 &&
      element_type != LOOM_SCALAR_TYPE_F32 &&
      element_type != LOOM_SCALAR_TYPE_F16 &&
      element_type != LOOM_SCALAR_TYPE_BF16 &&
      element_type != LOOM_SCALAR_TYPE_I8 &&
      element_type != LOOM_SCALAR_TYPE_I16) {
    return iree_ok_status();
  }
  if (element_type == LOOM_SCALAR_TYPE_I32) {
    bool can_materialize = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_value_can_materialize_as_vgpr_i32(
        context, source_op, value, &can_materialize));
    if (!can_materialize) {
      return iree_ok_status();
    }
  }

  bool is_dynamic = false;
  uint32_t lane_offset = 0;
  loom_value_id_t dynamic_index = LOOM_VALUE_ID_INVALID;
  const loom_value_slice_t indices = loom_vector_insert_indices(source_op);
  if (static_indices.i64_array[0] == INT64_MIN) {
    if (indices.count != 1) {
      return iree_ok_status();
    }
    is_dynamic = true;
    dynamic_index = indices.values[0];
  } else {
    if (indices.count != 0 || static_indices.i64_array[0] < 0 ||
        static_indices.i64_array[0] > UINT32_MAX) {
      return iree_ok_status();
    }
    lane_offset = (uint32_t)static_indices.i64_array[0];
    if (lane_offset >= storage.element_count) {
      return iree_ok_status();
    }
  }

  *out_plan = (loom_amdgpu_vector_insert_plan_t){
      .value = value,
      .dest = dest,
      .dynamic_index = dynamic_index,
      .result = result,
      .lane_offset = lane_offset,
      .lane_count = storage.element_count,
      .register_count = storage.register_count,
      .lane_bit_count = storage.element_bit_count,
      .element_type = element_type,
      .value_kind = LOOM_AMDGPU_VECTOR_INSERT_VALUE_KIND_DEFAULT,
      .fma_mix = {0},
      .is_dynamic = is_dynamic,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_vector_insert_fma_mix_half_result(
      context, source_op, flags, out_plan));
  *out_selected = true;
  return iree_ok_status();
}

static bool loom_amdgpu_type_is_16bit_float_packed_vector(
    loom_type_t type, loom_scalar_type_t element_type, uint32_t* out_lane_count,
    uint32_t* out_register_count) {
  *out_lane_count = 0;
  *out_register_count = 0;
  if (loom_type_element_type(type) != element_type) {
    return false;
  }
  uint32_t payload_bit_count = 0;
  if (!loom_amdgpu_type_packed_16bit_float_storage(type, &payload_bit_count,
                                                   out_register_count)) {
    return false;
  }
  *out_lane_count = payload_bit_count / 16u;
  return true;
}

static bool loom_amdgpu_type_is_extf_packed_float_vector(
    loom_type_t type, loom_scalar_type_t element_type, uint32_t* out_lane_count,
    uint32_t* out_register_count) {
  *out_lane_count = 0;
  *out_register_count = 0;
  if (loom_type_element_type(type) != element_type) {
    return false;
  }
  uint32_t payload_bit_count = 0;
  if (element_type == LOOM_SCALAR_TYPE_F8E4M3 ||
      element_type == LOOM_SCALAR_TYPE_F8E5M2) {
    if (!loom_amdgpu_type_packed_8bit_float_storage(type, &payload_bit_count,
                                                    out_register_count)) {
      return false;
    }
    *out_lane_count = payload_bit_count / 8u;
    return true;
  }
  if (!loom_amdgpu_type_packed_16bit_float_storage(type, &payload_bit_count,
                                                   out_register_count)) {
    return false;
  }
  *out_lane_count = payload_bit_count / 16u;
  return true;
}

static bool loom_amdgpu_type_is_vector1_element(
    loom_type_t type, loom_scalar_type_t element_type) {
  if (!loom_type_is_vector(type) ||
      loom_type_element_type(type) != element_type ||
      loom_type_rank(type) != 1) {
    return false;
  }
  const uint64_t dim = loom_type_dim(type, 0);
  return !loom_dim_is_dynamic(dim) && loom_dim_static_size(dim) == 1;
}

static bool loom_amdgpu_direct_fp8_e8m0_pk8_descriptor_available(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_scalar_type_t source_element_type,
    loom_scalar_type_t result_element_type) {
  loom_amdgpu_descriptor_ref_t descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  return loom_amdgpu_fp8_e8m0_pk8_descriptor_ref(
             source_element_type, result_element_type, &descriptor_ref) &&
         loom_amdgpu_descriptor_set_has_ref(descriptor_set, descriptor_ref);
}

static bool loom_amdgpu_direct_fp8_e8m0_pk8_storage_matches(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t source, loom_scalar_type_t source_element_type,
    uint32_t source_lane_count) {
  if (source_lane_count == 0 || (source_lane_count & 7u) != 0) {
    return false;
  }

  uint32_t storage_lane_offset = 0;
  uint32_t storage_lane_stride = 1;
  uint32_t storage_lane_count = source_lane_count;
  uint32_t storage_register_count = 0;
  if (!loom_amdgpu_type_is_extf_packed_float_vector(
          loom_module_value_type(module, source), source_element_type,
          &storage_lane_count, &storage_register_count)) {
    return false;
  }

  loom_value_fact_static_lane_origin_t lane_origin = {0};
  if (fact_table && loom_value_fact_table_query_static_lane_origin(
                        fact_table, module, source, &lane_origin)) {
    uint32_t origin_lane_count = 0;
    uint32_t origin_register_count = 0;
    const loom_type_t origin_source_type =
        loom_module_value_type(module, lane_origin.source_value_id);
    if (loom_amdgpu_type_is_extf_packed_float_vector(
            origin_source_type, source_element_type, &origin_lane_count,
            &origin_register_count)) {
      storage_lane_offset = lane_origin.source_lane_offset;
      storage_lane_stride = lane_origin.source_lane_stride;
      storage_lane_count = origin_lane_count;
      storage_register_count = origin_register_count;
    }
  }

  if (storage_lane_stride != 1u) {
    return false;
  }
  for (uint32_t lane_index = 0; lane_index < source_lane_count;
       lane_index += 8u) {
    const uint64_t storage_lane =
        (uint64_t)storage_lane_offset + (uint64_t)lane_index;
    if ((storage_lane & 3u) != 0 || storage_lane + 7u >= storage_lane_count) {
      return false;
    }
    const uint64_t source_register_index = storage_lane / 4u;
    if (source_register_index + 1u >= storage_register_count ||
        source_register_index > UINT32_MAX) {
      return false;
    }
  }
  return true;
}

static bool loom_amdgpu_vector_decode_scale_element_type(
    loom_value_fact_numeric_format_flags_t scale_format,
    loom_scalar_type_t* out_element_type) {
  switch (scale_format) {
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F32:
      *out_element_type = LOOM_SCALAR_TYPE_F32;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E8M0:
      *out_element_type = LOOM_SCALAR_TYPE_I32;
      return true;
    default:
      return false;
  }
}

static bool loom_amdgpu_vector_decode_scale_source(
    const loom_module_t* module, const loom_op_t* source_op,
    loom_scalar_type_t scale_element_type, loom_value_id_t* out_scale_source) {
  *out_scale_source = LOOM_VALUE_ID_INVALID;
  loom_vector_encoding_auxiliary_view_t auxiliary_view = {0};
  iree_string_view_t unknown_key = iree_string_view_empty();
  if (!loom_vector_encoding_auxiliary_view_resolve(
          module, loom_vector_decode_auxiliary(source_op),
          loom_vector_decode_auxiliary_names(source_op), &auxiliary_view,
          &unknown_key)) {
    return false;
  }
  if (auxiliary_view.present_keys !=
      LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_SCALE) {
    return false;
  }
  const loom_value_id_t scale_source =
      auxiliary_view.values[LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SCALE];
  if (scale_source == LOOM_VALUE_ID_INVALID ||
      scale_source >= module->values.count ||
      !loom_amdgpu_type_is_vector1_element(
          loom_module_value_type(module, scale_source), scale_element_type)) {
    return false;
  }
  *out_scale_source = scale_source;
  return true;
}

static loom_value_id_t loom_amdgpu_vector_decode_materialized_scale_source(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t scale_source) {
  loom_value_id_t scalar_source = LOOM_VALUE_ID_INVALID;
  if (loom_value_fact_table_query_uniform_element_origin(
          fact_table, module, scale_source, &scalar_source)) {
    return scalar_source;
  }
  return scale_source;
}

bool loom_amdgpu_vector_decode_can_lower_as_fp8_conversion(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_op_t* source_op) {
  if (!loom_vector_decode_isa(source_op) || fact_table == NULL) {
    return false;
  }

  const loom_value_id_t source = loom_vector_decode_payload(source_op);
  const loom_value_id_t result = loom_vector_decode_result(source_op);
  const loom_type_t source_type = loom_module_value_type(module, source);
  const loom_type_t result_type = loom_module_value_type(module, result);
  if (!loom_type_is_vector(source_type) || !loom_type_is_vector(result_type) ||
      !loom_type_shape_equals(source_type, result_type)) {
    return false;
  }

  uint32_t source_lane_count = 0;
  uint32_t source_register_count = 0;
  const loom_scalar_type_t source_element_type =
      loom_type_element_type(source_type);
  if (!loom_amdgpu_type_is_extf_packed_float_vector(
          source_type, source_element_type, &source_lane_count,
          &source_register_count)) {
    return false;
  }
  (void)source_register_count;

  const loom_scalar_type_t result_element_type =
      loom_type_element_type(result_type);
  const bool has_auxiliary =
      loom_vector_decode_auxiliary(source_op).count != 0 ||
      loom_vector_decode_auxiliary_names(source_op).count != 0;
  if (result_element_type == LOOM_SCALAR_TYPE_F32) {
    if (loom_amdgpu_vector_f32_register_count(result_type) !=
        source_lane_count) {
      return false;
    }
  } else if (result_element_type == LOOM_SCALAR_TYPE_BF16 ||
             result_element_type == LOOM_SCALAR_TYPE_F16) {
    uint32_t result_lane_count = 0;
    uint32_t result_register_count = 0;
    if (!loom_amdgpu_type_is_16bit_float_packed_vector(
            result_type, result_element_type, &result_lane_count,
            &result_register_count) ||
        result_lane_count != source_lane_count) {
      return false;
    }
    (void)result_register_count;
  } else {
    return false;
  }

  loom_value_fact_encoding_summary_t summary = {0};
  if (!loom_value_facts_query_encoding_summary(
          &fact_table->context,
          loom_value_fact_table_lookup(fact_table,
                                       loom_vector_decode_schema(source_op)),
          &summary)) {
    return false;
  }
  if (!has_auxiliary) {
    return loom_amdgpu_fp8_encoded_operand_schema_matches(
        summary.storage_schema.encoded_operand, source_element_type,
        source_lane_count,
        LOOM_AMDGPU_FP8_ENCODED_OPERAND_SCHEMA_KIND_UNSCALED);
  }

  loom_value_id_t scale_source = LOOM_VALUE_ID_INVALID;
  switch (summary.storage_schema.encoded_operand.scale_format) {
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F32:
      return loom_amdgpu_vector_decode_scale_source(
                 module, source_op, LOOM_SCALAR_TYPE_F32, &scale_source) &&
             loom_amdgpu_fp8_encoded_operand_schema_matches(
                 summary.storage_schema.encoded_operand, source_element_type,
                 source_lane_count,
                 LOOM_AMDGPU_FP8_ENCODED_OPERAND_SCHEMA_KIND_SCALE_F32);
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E8M0:
      return loom_amdgpu_vector_decode_scale_source(
                 module, source_op, LOOM_SCALAR_TYPE_I32, &scale_source) &&
             loom_amdgpu_fp8_encoded_operand_schema_matches(
                 summary.storage_schema.encoded_operand, source_element_type,
                 source_lane_count,
                 LOOM_AMDGPU_FP8_ENCODED_OPERAND_SCHEMA_KIND_SCALE_E8M0) &&
             loom_amdgpu_direct_fp8_e8m0_pk8_descriptor_available(
                 descriptor_set, source_element_type, result_element_type) &&
             loom_amdgpu_direct_fp8_e8m0_pk8_storage_matches(
                 module, fact_table, source, source_element_type,
                 source_lane_count);
    default:
      return false;
  }
}

iree_status_t loom_amdgpu_low_legality_verify_vector_decode(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  (void)provider;
  *out_handled = false;
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  if (loom_amdgpu_vector_decode_can_lower_as_fp8_conversion(
          loom_target_low_legality_module(context),
          loom_target_low_legality_fact_table(context),
          loom_target_low_legality_descriptor_set(context), op)) {
    *out_handled = true;
  }
  return iree_ok_status();
}

#define LOOM_AMDGPU_VECTOR_OP_INDEX(op_kind) ((uint8_t)((op_kind) & 0xFFu))
#define LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_ROW(op, kind_) \
  [LOOM_AMDGPU_VECTOR_OP_INDEX(LOOM_OP_VECTOR_##op)] =                \
      LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_##kind_

static const loom_amdgpu_vector_16bit_float_conversion_kind_t
    kAmdgpuVector16BitFloatConversionKindByVectorOp[LOOM_OP_VECTOR_COUNT_] = {
        LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_ROW(EXTF, EXTF),
        LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_ROW(FPTRUNC, FPTRUNC),
        LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_ROW(DECODE, DECODE),
};

#undef LOOM_AMDGPU_VECTOR_OP_INDEX
#undef LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_ROW

static loom_amdgpu_vector_16bit_float_conversion_kind_t
loom_amdgpu_vector_16bit_float_conversion_kind(loom_op_kind_t op_kind) {
  if (loom_op_dialect_id(op_kind) != LOOM_DIALECT_VECTOR) {
    return LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_NONE;
  }
  const uint8_t op_index = loom_op_dialect_index(op_kind);
  if (op_index >=
      IREE_ARRAYSIZE(kAmdgpuVector16BitFloatConversionKindByVectorOp)) {
    return LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_NONE;
  }
  return kAmdgpuVector16BitFloatConversionKindByVectorOp[op_index];
}

static iree_status_t
loom_amdgpu_vector_16bit_float_conversion_plan_from_accepted_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_16bit_float_conversion_plan_t* out_plan) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  *out_plan = (loom_amdgpu_vector_16bit_float_conversion_plan_t){0};

  const loom_value_id_t source = loom_op_const_operands(source_op)[0];
  const loom_value_id_t result = loom_op_const_results(source_op)[0];
  const loom_amdgpu_vector_16bit_float_conversion_kind_t kind =
      loom_amdgpu_vector_16bit_float_conversion_kind(source_op->kind);
  IREE_ASSERT_NE(kind, LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_NONE);

  const loom_type_t source_type = loom_module_value_type(module, source);
  const loom_type_t result_type = loom_module_value_type(module, result);
  loom_scalar_type_t source_element_type = loom_type_element_type(source_type);
  const loom_scalar_type_t result_element_type =
      loom_type_element_type(result_type);
  uint32_t source_lane_count = 0;
  uint32_t source_register_count = 0;
  loom_value_id_t storage_source = source;
  loom_value_id_t content_fact_source = source;
  loom_value_id_t scale_source = LOOM_VALUE_ID_INVALID;
  loom_value_fact_numeric_format_flags_t scale_format =
      LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE;
  uint32_t scale_group_element_count = 0;
  uint32_t storage_lane_offset = 0;
  uint32_t storage_lane_stride = 1;
  uint32_t storage_lane_count = 0;
  uint32_t storage_register_count = 0;
  uint32_t result_lane_count = 0;
  uint32_t result_register_count = 0;
  if (kind == LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_EXTF ||
      kind == LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_DECODE) {
    IREE_ASSERT_TRUE(
        kind != LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_DECODE ||
        loom_amdgpu_vector_decode_can_lower_as_fp8_conversion(
            module, fact_table, loom_low_lower_context_descriptor_set(context),
            source_op));
    if (kind == LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_DECODE) {
      loom_value_fact_encoding_summary_t summary = {0};
      if (loom_value_facts_query_encoding_summary(
              &fact_table->context,
              loom_value_fact_table_lookup(
                  fact_table, loom_vector_decode_schema(source_op)),
              &summary)) {
        scale_format = summary.storage_schema.encoded_operand.scale_format;
        scale_group_element_count =
            summary.storage_schema.encoded_operand.scale_group_element_count;
      }
      loom_scalar_type_t scale_element_type = 0;
      if (loom_amdgpu_vector_decode_scale_element_type(scale_format,
                                                       &scale_element_type) &&
          loom_amdgpu_vector_decode_scale_source(
              module, source_op, scale_element_type, &scale_source)) {
        scale_source = loom_amdgpu_vector_decode_materialized_scale_source(
            module, fact_table, scale_source);
      } else {
        scale_format = LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE;
      }
    }
    IREE_ASSERT_TRUE(result_element_type == LOOM_SCALAR_TYPE_F32 ||
                     result_element_type == LOOM_SCALAR_TYPE_BF16 ||
                     result_element_type == LOOM_SCALAR_TYPE_F16);
    const bool source_matches = loom_amdgpu_type_is_extf_packed_float_vector(
        source_type, source_element_type, &source_lane_count,
        &source_register_count);
    IREE_ASSERT(source_matches);
    storage_lane_count = source_lane_count;
    storage_register_count = source_register_count;
    loom_value_fact_static_lane_origin_t lane_origin = {0};
    if (loom_value_fact_table_query_static_lane_origin(fact_table, module,
                                                       source, &lane_origin)) {
      uint32_t origin_lane_count = 0;
      uint32_t origin_register_count = 0;
      const loom_type_t origin_source_type =
          loom_module_value_type(module, lane_origin.source_value_id);
      if (loom_amdgpu_type_is_extf_packed_float_vector(
              origin_source_type, source_element_type, &origin_lane_count,
              &origin_register_count)) {
        storage_source = lane_origin.source_value_id;
        storage_lane_offset = lane_origin.source_lane_offset;
        storage_lane_stride = lane_origin.source_lane_stride;
        storage_lane_count = origin_lane_count;
        storage_register_count = origin_register_count;
      }
    }
    content_fact_source =
        kind == LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_DECODE
            ? result
            : storage_source;
    if (result_element_type == LOOM_SCALAR_TYPE_F32) {
      result_lane_count = loom_amdgpu_vector_f32_register_count(result_type);
      result_register_count = result_lane_count;
    } else {
      const bool result_matches = loom_amdgpu_type_is_16bit_float_packed_vector(
          result_type, result_element_type, &result_lane_count,
          &result_register_count);
      IREE_ASSERT(result_matches);
    }
  } else {
    IREE_ASSERT(source_element_type == LOOM_SCALAR_TYPE_F32);
    source_lane_count = loom_amdgpu_vector_f32_register_count(source_type);
    source_register_count = source_lane_count;
    storage_lane_count = source_lane_count;
    const bool result_matches = loom_amdgpu_type_is_16bit_float_packed_vector(
        result_type, result_element_type, &result_lane_count,
        &result_register_count);
    IREE_ASSERT(result_matches);
    storage_register_count = source_register_count;

    loom_value_fact_static_lane_origin_t lane_origin = {0};
    if (loom_value_fact_table_query_static_lane_origin(fact_table, module,
                                                       source, &lane_origin)) {
      uint32_t origin_lane_count = 0;
      uint32_t origin_register_count = 0;
      const loom_type_t origin_source_type =
          loom_module_value_type(module, lane_origin.source_value_id);
      const loom_scalar_type_t origin_element_type =
          loom_type_element_type(origin_source_type);
      if (lane_origin.source_lane_offset == 0 &&
          lane_origin.source_lane_stride == 1 &&
          origin_element_type == result_element_type &&
          loom_amdgpu_type_is_16bit_float_packed_vector(
              origin_source_type, origin_element_type, &origin_lane_count,
              &origin_register_count) &&
          origin_lane_count == source_lane_count &&
          origin_register_count == result_register_count) {
        source_element_type = origin_element_type;
        storage_source = lane_origin.source_value_id;
        content_fact_source = storage_source;
        storage_lane_offset = lane_origin.source_lane_offset;
        storage_lane_stride = lane_origin.source_lane_stride;
        storage_lane_count = origin_lane_count;
        storage_register_count = origin_register_count;
      }
    }

    loom_value_fact_uniform_scale_origin_t scale_origin = {0};
    if (source_element_type == LOOM_SCALAR_TYPE_F32 &&
        loom_value_fact_table_query_uniform_scale_origin(
            fact_table, module, source, &scale_origin)) {
      lane_origin = (loom_value_fact_static_lane_origin_t){0};
      if (loom_value_fact_table_query_static_lane_origin(
              fact_table, module, scale_origin.source_value_id, &lane_origin)) {
        uint32_t origin_lane_count = 0;
        uint32_t origin_register_count = 0;
        const loom_type_t origin_source_type =
            loom_module_value_type(module, lane_origin.source_value_id);
        const loom_scalar_type_t origin_element_type =
            loom_type_element_type(origin_source_type);
        const uint64_t last_storage_lane =
            source_lane_count == 0
                ? UINT64_MAX
                : (uint64_t)lane_origin.source_lane_offset +
                      (uint64_t)(source_lane_count - 1u) *
                          (uint64_t)lane_origin.source_lane_stride;
        const bool is_origin_fp8 =
            origin_element_type == LOOM_SCALAR_TYPE_F8E4M3 ||
            origin_element_type == LOOM_SCALAR_TYPE_F8E5M2;
        const loom_low_lower_resolved_descriptor_t* scale_descriptor = NULL;
        if (is_origin_fp8) {
          IREE_RETURN_IF_ERROR(loom_amdgpu_get_fp8_scalef32_descriptor(
              context, origin_element_type, result_element_type,
              &scale_descriptor));
        }
        if (scale_descriptor != NULL &&
            loom_amdgpu_type_is_extf_packed_float_vector(
                origin_source_type, origin_element_type, &origin_lane_count,
                &origin_register_count) &&
            last_storage_lane < origin_lane_count) {
          source_element_type = origin_element_type;
          storage_source = lane_origin.source_value_id;
          content_fact_source = storage_source;
          scale_source = scale_origin.scale_value_id;
          scale_format = LOOM_VALUE_FACT_NUMERIC_FORMAT_F32;
          scale_group_element_count = source_lane_count;
          storage_lane_offset = lane_origin.source_lane_offset;
          storage_lane_stride = lane_origin.source_lane_stride;
          storage_lane_count = origin_lane_count;
          storage_register_count = origin_register_count;
        }
      }
    }
  }
  IREE_ASSERT_NE(source_lane_count, 0u);
  IREE_ASSERT_EQ(source_lane_count, result_lane_count);
  IREE_ASSERT_NE(source_register_count, 0u);
  IREE_ASSERT_NE(storage_lane_count, 0u);
  IREE_ASSERT_NE(storage_register_count, 0u);
  IREE_ASSERT_NE(storage_lane_stride, 0u);
  IREE_ASSERT_NE(result_register_count, 0u);

  *out_plan = (loom_amdgpu_vector_16bit_float_conversion_plan_t){
      .kind = kind,
      .source = source,
      .result = result,
      .storage_source = storage_source,
      .content_fact_source = content_fact_source,
      .scale_source = scale_source,
      .scale_format = scale_format,
      .scale_group_element_count = scale_group_element_count,
      .source_element_type = source_element_type,
      .result_element_type = result_element_type,
      .lane_count = source_lane_count,
      .source_register_count = source_register_count,
      .storage_lane_offset = storage_lane_offset,
      .storage_lane_stride = storage_lane_stride,
      .storage_lane_count = storage_lane_count,
      .storage_register_count = storage_register_count,
      .result_register_count = result_register_count,
  };
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_vector_16bit_float_conversion_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_16bit_float_conversion_plan_t* out_plan,
    bool* out_selected) {
  *out_plan = (loom_amdgpu_vector_16bit_float_conversion_plan_t){0};
  const loom_amdgpu_vector_16bit_float_conversion_kind_t kind =
      loom_amdgpu_vector_16bit_float_conversion_kind(source_op->kind);
  if (kind == LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_DECODE) {
    *out_selected = loom_amdgpu_vector_decode_can_lower_as_fp8_conversion(
        loom_low_lower_context_module(context),
        loom_low_lower_context_fact_table(context),
        loom_low_lower_context_descriptor_set(context), source_op);
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_select_arithmetic_contract(
        context, source_op, out_selected));
  }
  if (*out_selected) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_vector_16bit_float_conversion_plan_from_accepted_op(
            context, source_op, out_plan));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_index_cast_range_diagnostic(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_type_t source_type, loom_type_t result_type,
    loom_value_facts_t source_facts, uint32_t index_bitwidth) {
  loom_module_t* module = loom_low_lower_context_module(context);
  static const iree_string_view_t accepted_proof_sources[] = {
      IREE_SVL("scalar.assume on the integer source before index.cast"),
      IREE_SVL("config or kernel boundary facts on the integer source"),
  };
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_lower_context_target_key(context)),
      loom_param_string(loom_low_lower_context_export_name(context)),
      loom_param_string(loom_low_lower_context_config_key(context)),
      loom_param_string(loom_low_lower_context_function_name(context)),
      loom_param_string(loom_op_name(module, source_op)),
      loom_param_type(source_type),
      loom_param_type(result_type),
      loom_param_i64(source_facts.range_lo),
      loom_param_i64(source_facts.range_hi),
      loom_param_u32(index_bitwidth),
      loom_param_i64(INT32_MIN),
      loom_param_i64(INT32_MAX),
      loom_param_string(IREE_SV("index_cast.target_width_range")),
      loom_param_string_list(accepted_proof_sources,
                             IREE_ARRAYSIZE(accepted_proof_sources)),
  };
  return loom_low_lower_emit_error_ref(context, source_op,
                                       LOOM_ERR_AMDGPU_033_REF, params,
                                       IREE_ARRAYSIZE(params));
}

iree_status_t loom_amdgpu_select_index_cast_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_index_cast_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_index_cast_plan_t){0};
  *out_selected = false;
  const loom_value_id_t source = loom_index_cast_input(source_op);
  const loom_value_id_t result = loom_index_cast_result(source_op);

  loom_type_t source_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_low_lower_map_value(context, source_op, source, &source_low_type));
  if (!loom_low_type_is_register(source_low_type)) {
    return iree_ok_status();
  }
  loom_type_t result_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op, result,
                                                   &result_low_type));
  if (!loom_type_equal(source_low_type, result_low_type)) {
    const loom_module_t* module = loom_low_lower_context_module(context);
    const loom_type_t source_type = loom_module_value_type(module, source);
    const loom_type_t result_type = loom_module_value_type(module, result);
    const uint32_t index_bitwidth = loom_amdgpu_target_index_bitwidth(context);
    if (!loom_amdgpu_type_is_i64(source_type) ||
        !loom_amdgpu_type_is_index_scalar(result_type) ||
        index_bitwidth != 32) {
      return iree_ok_status();
    }

    loom_value_facts_t source_facts = loom_value_facts_unknown();
    const loom_value_fact_table_t* fact_table =
        loom_low_lower_context_fact_table(context);
    if (fact_table != NULL && source < module->values.count) {
      source_facts = loom_value_fact_table_lookup(fact_table, source);
    }
    if (!loom_value_facts_fit_signed_bit_count(source_facts,
                                               (uint8_t)index_bitwidth)) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_index_cast_range_diagnostic(
          context, source_op, source_type, result_type, source_facts,
          index_bitwidth));
      *out_plan = (loom_amdgpu_index_cast_plan_t){
          .kind = LOOM_AMDGPU_INDEX_CAST_KIND_DIAGNOSTIC_REJECTED,
          .source = source,
          .result = result,
          .index_bitwidth = index_bitwidth,
      };
      *out_selected = true;
      return iree_ok_status();
    }

    *out_plan = (loom_amdgpu_index_cast_plan_t){
        .kind = LOOM_AMDGPU_INDEX_CAST_KIND_PRESERVING_LOW_32,
        .source = source,
        .result = result,
        .index_bitwidth = index_bitwidth,
    };
    *out_selected = true;
    return iree_ok_status();
  }

  *out_plan = (loom_amdgpu_index_cast_plan_t){
      .kind = LOOM_AMDGPU_INDEX_CAST_KIND_ALIAS,
      .source = source,
      .result = result,
  };
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_index_cmp_i64_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_i64_compare_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_i64_compare_plan_t){0};
  *out_selected = false;
  const loom_module_t* module = loom_low_lower_context_module(context);
  if (!loom_amdgpu_address_cmp_needs_64bit(
          module, loom_low_lower_context_fact_table(context), source_op)) {
    return iree_ok_status();
  }

  const loom_value_id_t result = loom_index_cmp_result(source_op);
  loom_type_t result_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op, result,
                                                   &result_low_type));
  const bool result_is_native_mask =
      loom_amdgpu_low_type_is_register_class_count(
          context, result_low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);
  if (!result_is_native_mask) {
    return iree_ok_status();
  }

  loom_amdgpu_i64_compare_plan_t plan = {
      .lhs = loom_index_cmp_lhs(source_op),
      .rhs = loom_index_cmp_rhs(source_op),
      .result = result,
  };
  loom_scalar_cmpi_predicate_t predicate;
  if (!loom_amdgpu_index_cmp_predicate_to_scalar(
          loom_index_cmp_predicate(source_op), &predicate) ||
      !loom_amdgpu_i64_compare_predicate_descriptors(
          predicate, &plan.high_descriptor_ref, &plan.low_descriptor_ref,
          &plan.combine_descriptor_ref, &plan.needs_high_equal)) {
    return iree_ok_status();
  }

  iree_string_view_t constraint_key = iree_string_view_empty();
  if (!loom_amdgpu_i64_compare_descriptors_supported(
          loom_low_lower_context_descriptor_set(context), &plan,
          &constraint_key)) {
    return iree_ok_status();
  }

  *out_plan = plan;
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_scalar_i64_compare_operand_can_lower(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source, bool* out_can_lower) {
  *out_can_lower = false;
  loom_type_t source_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_low_lower_map_value(context, source_op, source, &source_low_type));
  if (!loom_low_type_is_register(source_low_type) ||
      loom_low_register_type_unit_count(source_low_type) != 2) {
    return iree_ok_status();
  }
  const bool is_vgpr = loom_amdgpu_low_type_is_register_class(
      context, source_low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
  if (is_vgpr) {
    *out_can_lower = true;
    return iree_ok_status();
  }
  *out_can_lower = loom_amdgpu_low_type_is_register_class(
      context, source_low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_scalar_cmpi_i64_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_i64_compare_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_i64_compare_plan_t){0};
  *out_selected = false;
  const loom_module_t* module = loom_low_lower_context_module(context);
  if (!loom_amdgpu_scalar_cmpi_has_i64_operands(module, source_op)) {
    return iree_ok_status();
  }

  const loom_value_id_t result = loom_scalar_cmpi_result(source_op);
  loom_type_t result_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op, result,
                                                   &result_low_type));
  const bool result_is_native_mask =
      loom_amdgpu_low_type_is_register_class_count(
          context, result_low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);
  if (!result_is_native_mask) {
    return iree_ok_status();
  }

  bool lhs_can_lower = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_scalar_i64_compare_operand_can_lower(
      context, source_op, loom_scalar_cmpi_lhs(source_op), &lhs_can_lower));
  bool rhs_can_lower = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_scalar_i64_compare_operand_can_lower(
      context, source_op, loom_scalar_cmpi_rhs(source_op), &rhs_can_lower));
  if (!lhs_can_lower || !rhs_can_lower) {
    return iree_ok_status();
  }

  loom_amdgpu_i64_compare_plan_t plan = {
      .lhs = loom_scalar_cmpi_lhs(source_op),
      .rhs = loom_scalar_cmpi_rhs(source_op),
      .result = result,
  };
  if (!loom_amdgpu_i64_compare_predicate_descriptors(
          (loom_scalar_cmpi_predicate_t)loom_scalar_cmpi_predicate(source_op),
          &plan.high_descriptor_ref, &plan.low_descriptor_ref,
          &plan.combine_descriptor_ref, &plan.needs_high_equal)) {
    return iree_ok_status();
  }

  iree_string_view_t constraint_key = iree_string_view_empty();
  if (!loom_amdgpu_i64_compare_descriptors_supported(
          loom_low_lower_context_descriptor_set(context), &plan,
          &constraint_key)) {
    return iree_ok_status();
  }

  *out_plan = plan;
  *out_selected = true;
  return iree_ok_status();
}

static bool loom_amdgpu_scalar_i64_alu_descriptors_supported(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_scalar_i64_alu_kind_t kind,
    iree_string_view_t* out_constraint_key) {
  return loom_amdgpu_i64_alu_descriptor_requirements_present(
      descriptor_set, kAmdgpuScalarI64AluDescriptorRequirementRows,
      IREE_ARRAYSIZE(kAmdgpuScalarI64AluDescriptorRequirementRows),
      (iree_host_size_t)kind, IREE_SV("operation.scalar_i64_alu"),
      out_constraint_key);
}

static bool loom_amdgpu_address_i64_alu_kind_uses_vgpr(
    loom_amdgpu_address_i64_alu_kind_t kind) {
  switch (kind) {
    case LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_ADD:
    case LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_SUB:
    case LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_MUL_LO:
    case LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_SHL:
    case LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_MADD_LO:
      return true;
    case LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_SGPR_ADD:
    case LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_NONE:
      return false;
  }
  return false;
}

static bool loom_amdgpu_address_i64_alu_descriptors_supported(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_address_i64_alu_kind_t kind,
    iree_string_view_t* out_constraint_key) {
  return loom_amdgpu_i64_alu_descriptor_requirements_present(
      descriptor_set, kAmdgpuAddressI64AluDescriptorRequirementRows,
      IREE_ARRAYSIZE(kAmdgpuAddressI64AluDescriptorRequirementRows),
      (iree_host_size_t)kind, IREE_SV("operation.address_i64_alu"),
      out_constraint_key);
}

#define LOOM_AMDGPU_OP_INDEX(op_kind) ((uint8_t)((op_kind) & 0xFFu))

typedef struct loom_amdgpu_address_i64_alu_source_layout_t {
  // Operation kind selected by this source op.
  loom_amdgpu_address_i64_alu_kind_t kind;
  // Number of fixed operands consumed by the source op.
  uint8_t operand_count;
} loom_amdgpu_address_i64_alu_source_layout_t;

static const loom_amdgpu_address_i64_alu_source_layout_t
    kAmdgpuAddressI64AluSourceLayoutByIndexOp[LOOM_OP_INDEX_COUNT_] = {
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_INDEX_ADD)] =
            {
                .kind = LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_ADD,
                .operand_count = 2,
            },
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_INDEX_SUB)] =
            {
                .kind = LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_SUB,
                .operand_count = 2,
            },
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_INDEX_MUL)] =
            {
                .kind = LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_MUL_LO,
                .operand_count = 2,
            },
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_INDEX_MADD)] =
            {
                .kind = LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_MADD_LO,
                .operand_count = 3,
            },
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_INDEX_SHLI)] =
            {
                .kind = LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_SHL,
                .operand_count = 2,
            },
};

static const loom_amdgpu_scalar_i64_alu_kind_t
    kAmdgpuScalarI64AluKindByScalarOp[LOOM_OP_SCALAR_COUNT_] = {
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_SCALAR_ADDI)] =
            LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_ADD,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_SCALAR_SUBI)] =
            LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_SUB,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_SCALAR_MULI)] =
            LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_MUL_LO,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_SCALAR_SHLI)] =
            LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_SHL,
};

#undef LOOM_AMDGPU_OP_INDEX

static bool loom_amdgpu_address_i64_alu_op(
    const loom_op_t* source_op, loom_amdgpu_address_i64_alu_kind_t* out_kind,
    loom_value_id_t* out_lhs, loom_value_id_t* out_rhs,
    loom_value_id_t* out_addend, loom_value_id_t* out_result) {
  *out_kind = LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_NONE;
  *out_lhs = LOOM_VALUE_ID_INVALID;
  *out_rhs = LOOM_VALUE_ID_INVALID;
  *out_addend = LOOM_VALUE_ID_INVALID;
  *out_result = LOOM_VALUE_ID_INVALID;
  if (loom_op_dialect_id(source_op->kind) != LOOM_DIALECT_INDEX) return false;
  const uint8_t op_index = loom_op_dialect_index(source_op->kind);
  if (op_index >= IREE_ARRAYSIZE(kAmdgpuAddressI64AluSourceLayoutByIndexOp)) {
    return false;
  }
  const loom_amdgpu_address_i64_alu_source_layout_t* layout =
      &kAmdgpuAddressI64AluSourceLayoutByIndexOp[op_index];
  if (layout->kind == LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_NONE) return false;
  const loom_value_id_t* operands = loom_op_const_operands(source_op);
  *out_kind = layout->kind;
  *out_lhs = operands[0];
  *out_rhs = operands[1];
  *out_addend = layout->operand_count > 2 ? operands[2] : LOOM_VALUE_ID_INVALID;
  *out_result = loom_op_const_results(source_op)[0];
  return true;
}

static bool loom_amdgpu_address_i64_alu_result_needs_wide(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t result) {
  if (result >= module->values.count) {
    return false;
  }
  const loom_type_t result_type = loom_module_value_type(module, result);
  return loom_amdgpu_source_address_value_needs_64bit(module, fact_table,
                                                      result, result_type);
}

static iree_status_t loom_amdgpu_select_address_i64_alu_kind(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_address_i64_alu_kind_t operation_kind, loom_value_id_t result,
    loom_amdgpu_address_i64_alu_kind_t* out_kind) {
  *out_kind = LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_NONE;
  loom_type_t result_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op, result,
                                                   &result_low_type));

  const bool result_is_vgpr64 = loom_amdgpu_low_type_is_register_class_count(
      context, result_low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2);
  if (result_is_vgpr64) {
    *out_kind = operation_kind;
    return iree_ok_status();
  }

  if (operation_kind != LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_ADD) {
    return iree_ok_status();
  }
  const bool result_is_sgpr64 = loom_amdgpu_low_type_is_register_class_count(
      context, result_low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);
  if (result_is_sgpr64) {
    *out_kind = LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_SGPR_ADD;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_address_i64_operand_can_materialize(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source, loom_amdgpu_address_i64_alu_kind_t result_kind,
    bool* out_can_lower) {
  *out_can_lower = false;
  loom_type_t source_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_low_lower_map_value(context, source_op, source, &source_low_type));
  if (!loom_low_type_is_register(source_low_type)) {
    return iree_ok_status();
  }
  const uint32_t unit_count =
      loom_low_register_type_unit_count(source_low_type);
  if (unit_count != 1 && unit_count != 2) {
    return iree_ok_status();
  }
  if (loom_amdgpu_address_i64_alu_kind_uses_vgpr(result_kind)) {
    const bool is_vgpr = loom_amdgpu_low_type_is_register_class(
        context, source_low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
    if (is_vgpr) {
      *out_can_lower = true;
      return iree_ok_status();
    }
  }
  const bool is_sgpr = loom_amdgpu_low_type_is_register_class(
      context, source_low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
  *out_can_lower = is_sgpr;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_address_i64_alu_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_address_i64_alu_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_address_i64_alu_plan_t){0};
  *out_selected = false;
  loom_amdgpu_address_i64_alu_kind_t operation_kind =
      LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_NONE;
  loom_value_id_t lhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t rhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t addend = LOOM_VALUE_ID_INVALID;
  loom_value_id_t result = LOOM_VALUE_ID_INVALID;
  if (!loom_amdgpu_address_i64_alu_op(source_op, &operation_kind, &lhs, &rhs,
                                      &addend, &result)) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  if (!loom_amdgpu_address_i64_alu_result_needs_wide(
          module, loom_low_lower_context_fact_table(context), result)) {
    return iree_ok_status();
  }

  loom_amdgpu_address_i64_alu_kind_t kind =
      LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_NONE;
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_address_i64_alu_kind(
      context, source_op, operation_kind, result, &kind));
  if (kind == LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_NONE) {
    return iree_ok_status();
  }

  bool lhs_can_lower = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_address_i64_operand_can_materialize(
      context, source_op, lhs, kind, &lhs_can_lower));
  bool rhs_can_lower = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_address_i64_operand_can_materialize(
      context, source_op, rhs, kind, &rhs_can_lower));
  bool addend_can_lower = true;
  if (addend != LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_address_i64_operand_can_materialize(
        context, source_op, addend, kind, &addend_can_lower));
  }
  if (!lhs_can_lower || !rhs_can_lower || !addend_can_lower) {
    return iree_ok_status();
  }

  iree_string_view_t constraint_key = iree_string_view_empty();
  if (!loom_amdgpu_address_i64_alu_descriptors_supported(
          loom_low_lower_context_descriptor_set(context), kind,
          &constraint_key)) {
    return iree_ok_status();
  }

  *out_plan = (loom_amdgpu_address_i64_alu_plan_t){
      .lhs = lhs,
      .rhs = rhs,
      .addend = addend,
      .result = result,
      .kind = kind,
  };
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_low_legality_verify_address_i64_alu(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  (void)provider;
  if (!loom_amdgpu_low_legality_context_is_amdgpu(context)) {
    return iree_ok_status();
  }
  loom_amdgpu_address_i64_alu_kind_t operation_kind =
      LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_NONE;
  loom_value_id_t lhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t rhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t addend = LOOM_VALUE_ID_INVALID;
  loom_value_id_t result = LOOM_VALUE_ID_INVALID;
  if (!loom_amdgpu_address_i64_alu_op(op, &operation_kind, &lhs, &rhs, &addend,
                                      &result)) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_target_low_legality_module(context);
  if (!loom_amdgpu_address_i64_alu_result_needs_wide(
          module, loom_target_low_legality_fact_table(context), result)) {
    return iree_ok_status();
  }
  *out_handled = true;

  bool result_prefers_vgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_low_legality_value_prefers_vgpr(
      context, result, &result_prefers_vgpr));
  loom_amdgpu_address_i64_alu_kind_t kind = operation_kind;
  if (operation_kind == LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_ADD &&
      !result_prefers_vgpr) {
    kind = LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_SGPR_ADD;
  } else if (!result_prefers_vgpr) {
    return loom_amdgpu_low_legality_reject(context, op,
                                           IREE_SV("result.vgpr64"));
  }

  iree_string_view_t constraint_key = iree_string_view_empty();
  if (loom_amdgpu_address_i64_alu_descriptors_supported(
          loom_target_low_legality_descriptor_set(context), kind,
          &constraint_key)) {
    return iree_ok_status();
  }
  return loom_amdgpu_low_legality_reject(context, op, constraint_key);
}

static bool loom_amdgpu_scalar_i64_alu_op(
    const loom_op_t* source_op, loom_amdgpu_scalar_i64_alu_kind_t* out_kind,
    loom_value_id_t* out_lhs, loom_value_id_t* out_rhs,
    loom_value_id_t* out_result) {
  *out_kind = LOOM_AMDGPU_SCALAR_I64_ALU_KIND_NONE;
  *out_lhs = LOOM_VALUE_ID_INVALID;
  *out_rhs = LOOM_VALUE_ID_INVALID;
  *out_result = LOOM_VALUE_ID_INVALID;
  if (loom_op_dialect_id(source_op->kind) != LOOM_DIALECT_SCALAR) return false;
  const uint8_t op_index = loom_op_dialect_index(source_op->kind);
  if (op_index >= IREE_ARRAYSIZE(kAmdgpuScalarI64AluKindByScalarOp)) {
    return false;
  }
  const loom_amdgpu_scalar_i64_alu_kind_t kind =
      kAmdgpuScalarI64AluKindByScalarOp[op_index];
  if (kind == LOOM_AMDGPU_SCALAR_I64_ALU_KIND_NONE) return false;
  const loom_value_id_t* operands = loom_op_const_operands(source_op);
  *out_kind = kind;
  *out_lhs = operands[0];
  *out_rhs = operands[1];
  *out_result = loom_op_const_results(source_op)[0];
  return true;
}

static iree_status_t loom_amdgpu_scalar_i64_operand_can_materialize_as_vgpr64(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source, bool* out_can_lower) {
  *out_can_lower = false;
  loom_type_t source_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_low_lower_map_value(context, source_op, source, &source_low_type));
  if (!loom_low_type_is_register(source_low_type) ||
      loom_low_register_type_unit_count(source_low_type) != 2) {
    return iree_ok_status();
  }
  const bool is_vgpr = loom_amdgpu_low_type_is_register_class(
      context, source_low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
  if (is_vgpr) {
    *out_can_lower = true;
    return iree_ok_status();
  }
  *out_can_lower = loom_amdgpu_low_type_is_register_class(
      context, source_low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_scalar_i64_alu_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_scalar_i64_alu_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_scalar_i64_alu_plan_t){0};
  *out_selected = false;
  loom_amdgpu_scalar_i64_alu_kind_t kind = LOOM_AMDGPU_SCALAR_I64_ALU_KIND_NONE;
  loom_value_id_t lhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t rhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t result = LOOM_VALUE_ID_INVALID;
  if (!loom_amdgpu_scalar_i64_alu_op(source_op, &kind, &lhs, &rhs, &result)) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  if (!loom_amdgpu_type_is_i64(loom_module_value_type(module, result))) {
    return iree_ok_status();
  }

  loom_type_t result_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op, result,
                                                   &result_low_type));
  const bool result_is_vgpr64 = loom_amdgpu_low_type_is_register_class_count(
      context, result_low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2);
  if (!result_is_vgpr64) {
    return iree_ok_status();
  }

  bool lhs_can_lower = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_scalar_i64_operand_can_materialize_as_vgpr64(
      context, source_op, lhs, &lhs_can_lower));
  bool rhs_can_lower = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_scalar_i64_operand_can_materialize_as_vgpr64(
      context, source_op, rhs, &rhs_can_lower));
  if (!lhs_can_lower || !rhs_can_lower) {
    return iree_ok_status();
  }

  iree_string_view_t constraint_key = iree_string_view_empty();
  if (!loom_amdgpu_scalar_i64_alu_descriptors_supported(
          loom_low_lower_context_descriptor_set(context), kind,
          &constraint_key)) {
    return iree_ok_status();
  }

  *out_plan = (loom_amdgpu_scalar_i64_alu_plan_t){
      .kind = kind,
      .lhs = lhs,
      .rhs = rhs,
      .result = result,
  };
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_low_legality_verify_scalar_i64_alu(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  (void)provider;
  if (!loom_amdgpu_low_legality_context_is_amdgpu(context)) {
    return iree_ok_status();
  }
  loom_amdgpu_scalar_i64_alu_kind_t kind = LOOM_AMDGPU_SCALAR_I64_ALU_KIND_NONE;
  loom_value_id_t lhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t rhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t result = LOOM_VALUE_ID_INVALID;
  if (!loom_amdgpu_scalar_i64_alu_op(op, &kind, &lhs, &rhs, &result)) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_target_low_legality_module(context);
  if (!loom_amdgpu_type_is_i64(loom_module_value_type(module, result))) {
    return iree_ok_status();
  }
  *out_handled = true;

  if (!loom_amdgpu_type_is_i64(loom_module_value_type(module, lhs)) ||
      !loom_amdgpu_type_is_i64(loom_module_value_type(module, rhs))) {
    return loom_amdgpu_low_legality_reject(context, op, IREE_SV("operand.i64"));
  }

  bool result_prefers_vgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_low_legality_value_prefers_vgpr(
      context, result, &result_prefers_vgpr));
  if (!result_prefers_vgpr) {
    return loom_amdgpu_low_legality_reject(context, op,
                                           IREE_SV("result.vgpr64"));
  }

  iree_string_view_t constraint_key = iree_string_view_empty();
  if (loom_amdgpu_scalar_i64_alu_descriptors_supported(
          loom_target_low_legality_descriptor_set(context), kind,
          &constraint_key)) {
    return iree_ok_status();
  }
  return loom_amdgpu_low_legality_reject(context, op, constraint_key);
}

iree_status_t loom_amdgpu_low_legality_verify_scalar_remsi_i64(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  (void)provider;
  if (!loom_amdgpu_low_legality_context_is_amdgpu(context)) {
    return iree_ok_status();
  }
  if (!loom_scalar_remsi_isa(op)) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_target_low_legality_module(context);
  if (!loom_amdgpu_type_is_i64(
          loom_module_value_type(module, loom_scalar_remsi_lhs(op))) ||
      !loom_amdgpu_type_is_i64(
          loom_module_value_type(module, loom_scalar_remsi_rhs(op))) ||
      !loom_amdgpu_type_is_i64(
          loom_module_value_type(module, loom_scalar_remsi_result(op)))) {
    return iree_ok_status();
  }

  *out_handled = true;
  return loom_amdgpu_low_legality_reject(
      context, op, IREE_SV("scalar_remsi.signed_i64_dynamic"));
}

#define LOOM_AMDGPU_SCALAR_CONVERSION_REFS_0() \
  {                                            \
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE,         \
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE,         \
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE,         \
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE,         \
  }
#define LOOM_AMDGPU_SCALAR_CONVERSION_REFS_1(ref0) \
  {                                                \
      (ref0),                                      \
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE,             \
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE,             \
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE,             \
  }
#define LOOM_AMDGPU_SCALAR_CONVERSION_REFS_2(ref0, ref1) \
  {                                                      \
      (ref0),                                            \
      (ref1),                                            \
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE,                   \
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE,                   \
  }
#define LOOM_AMDGPU_SCALAR_CONVERSION_REFS_3(ref0, ref1, ref2) \
  {                                                            \
      (ref0),                                                  \
      (ref1),                                                  \
      (ref2),                                                  \
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE,                         \
  }

static const uint8_t kLoomAmdgpuScalarConversionRuleIndexes
    [LOOM_AMDGPU_SCALAR_CONVERSION_OP_COUNT_][LOOM_SCALAR_TYPE_COUNT_]
    [LOOM_SCALAR_TYPE_COUNT_] = {
        [LOOM_AMDGPU_SCALAR_CONVERSION_OP_TRUNCI] =
            {
                [LOOM_SCALAR_TYPE_I16][LOOM_SCALAR_TYPE_I8] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_TRUNCI_I16_TO_I8,
                [LOOM_SCALAR_TYPE_I32][LOOM_SCALAR_TYPE_I8] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_TRUNCI_I32_TO_I8,
                [LOOM_SCALAR_TYPE_I32][LOOM_SCALAR_TYPE_I16] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_TRUNCI_I32_TO_I16,
                [LOOM_SCALAR_TYPE_I64][LOOM_SCALAR_TYPE_I8] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_TRUNCI_I64_TO_I8,
                [LOOM_SCALAR_TYPE_I64][LOOM_SCALAR_TYPE_I16] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_TRUNCI_I64_TO_I16,
                [LOOM_SCALAR_TYPE_I64][LOOM_SCALAR_TYPE_I32] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_TRUNCI_I64_TO_I32,
            },
        [LOOM_AMDGPU_SCALAR_CONVERSION_OP_EXTF] =
            {
                [LOOM_SCALAR_TYPE_F8E4M3][LOOM_SCALAR_TYPE_BF16] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTF_F8E4M3_TO_BF16,
                [LOOM_SCALAR_TYPE_F8E5M2][LOOM_SCALAR_TYPE_BF16] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTF_F8E5M2_TO_BF16,
            },
        [LOOM_AMDGPU_SCALAR_CONVERSION_OP_EXTSI] =
            {
                [LOOM_SCALAR_TYPE_I8][LOOM_SCALAR_TYPE_I16] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTSI_I8_TO_I16,
                [LOOM_SCALAR_TYPE_I8][LOOM_SCALAR_TYPE_I32] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTSI_I8_TO_I32,
                [LOOM_SCALAR_TYPE_I8][LOOM_SCALAR_TYPE_I64] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTSI_I8_TO_I64,
                [LOOM_SCALAR_TYPE_I16][LOOM_SCALAR_TYPE_I32] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTSI_I16_TO_I32,
                [LOOM_SCALAR_TYPE_I16][LOOM_SCALAR_TYPE_I64] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTSI_I16_TO_I64,
                [LOOM_SCALAR_TYPE_I32][LOOM_SCALAR_TYPE_I64] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTSI_I32_TO_I64,
            },
        [LOOM_AMDGPU_SCALAR_CONVERSION_OP_EXTUI] =
            {
                [LOOM_SCALAR_TYPE_I8][LOOM_SCALAR_TYPE_I16] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTUI_I8_TO_I16,
                [LOOM_SCALAR_TYPE_I8][LOOM_SCALAR_TYPE_I32] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTUI_I8_TO_I32,
                [LOOM_SCALAR_TYPE_I8][LOOM_SCALAR_TYPE_I64] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTUI_I8_TO_I64,
                [LOOM_SCALAR_TYPE_I16][LOOM_SCALAR_TYPE_I32] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTUI_I16_TO_I32,
                [LOOM_SCALAR_TYPE_I16][LOOM_SCALAR_TYPE_I64] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTUI_I16_TO_I64,
                [LOOM_SCALAR_TYPE_I32][LOOM_SCALAR_TYPE_I64] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTUI_I32_TO_I64,
            },
        [LOOM_AMDGPU_SCALAR_CONVERSION_OP_UITOFP] =
            {
                [LOOM_SCALAR_TYPE_I8][LOOM_SCALAR_TYPE_F32] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_UITOFP_I8_TO_F32,
                [LOOM_SCALAR_TYPE_I16][LOOM_SCALAR_TYPE_F32] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_UITOFP_I16_TO_F32,
            },
        [LOOM_AMDGPU_SCALAR_CONVERSION_OP_FPTOSI] =
            {
                [LOOM_SCALAR_TYPE_F32][LOOM_SCALAR_TYPE_I32] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_FPTOSI_F32_TO_I32,
                [LOOM_SCALAR_TYPE_F32][LOOM_SCALAR_TYPE_I8] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_FPTOSI_F32_TO_I8,
                [LOOM_SCALAR_TYPE_F32][LOOM_SCALAR_TYPE_I16] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_FPTOSI_F32_TO_I16,
            },
        [LOOM_AMDGPU_SCALAR_CONVERSION_OP_FPTOUI] =
            {
                [LOOM_SCALAR_TYPE_F32][LOOM_SCALAR_TYPE_I32] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_FPTOUI_F32_TO_I32,
                [LOOM_SCALAR_TYPE_F32][LOOM_SCALAR_TYPE_I8] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_FPTOUI_F32_TO_I8,
                [LOOM_SCALAR_TYPE_F32][LOOM_SCALAR_TYPE_I16] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_FPTOUI_F32_TO_I16,
            },
};

static const loom_amdgpu_scalar_conversion_rule_t
    kLoomAmdgpuScalarConversionRules
        [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_COUNT_] = {
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_TRUNCI_I16_TO_I8] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_SIGN_EXTEND_NARROW,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_2(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_TRUNCI_I32_TO_I8] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_SIGN_EXTEND_NARROW,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_2(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_TRUNCI_I32_TO_I16] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_SIGN_EXTEND_NARROW,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_2(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_TRUNCI_I64_TO_I8] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_SIGN_EXTEND_NARROW_LOW_32,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_2(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_TRUNCI_I64_TO_I16] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_SIGN_EXTEND_NARROW_LOW_32,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_2(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_TRUNCI_I64_TO_I32] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_TRUNCATE_LOW_32,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_0()},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTF_F8E4M3_TO_BF16] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FP8_TO_BF16,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_0()},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTF_F8E5M2_TO_BF16] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FP8_TO_BF16,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_0()},

            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTSI_I8_TO_I16] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_ALIAS,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_0()},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTSI_I8_TO_I32] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_ALIAS,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_0()},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTSI_I8_TO_I64] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_SIGN_EXTEND_I64,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_1(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTSI_I16_TO_I32] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_ALIAS,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_0()},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTSI_I16_TO_I64] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_SIGN_EXTEND_I64,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_1(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTSI_I32_TO_I64] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_SIGN_EXTEND_I64,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_2(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_S_ASHR_I32_RHS_INLINE)},

            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTUI_I8_TO_I16] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_ZERO_EXTEND,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_1(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTUI_I8_TO_I32] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_ZERO_EXTEND,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_1(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTUI_I8_TO_I64] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_ZERO_EXTEND,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_2(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTUI_I16_TO_I32] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_ZERO_EXTEND,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_1(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTUI_I16_TO_I64] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_ZERO_EXTEND,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_2(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTUI_I32_TO_I64] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_ZERO_EXTEND,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_2(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
                     LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32)},

            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_UITOFP_I8_TO_F32] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_UITOFP_NARROW_TO_F32,
                 LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F32_U32,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_2(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F32_U32)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_UITOFP_I16_TO_F32] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_UITOFP_NARROW_TO_F32,
                 LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F32_U32,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_2(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F32_U32)},

            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_FPTOSI_F32_TO_I32] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FPTOI_F32_TO_I32,
                 LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_I32_F32,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_1(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_I32_F32)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_FPTOSI_F32_TO_I8] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FPTOI_F32_TO_NARROW,
                 LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_I32_F32,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_3(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_I32_F32,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_FPTOSI_F32_TO_I16] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FPTOI_F32_TO_NARROW,
                 LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_I32_F32,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_3(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_I32_F32,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_FPTOUI_F32_TO_I32] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FPTOI_F32_TO_I32,
                 LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_U32_F32,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_1(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_U32_F32)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_FPTOUI_F32_TO_I8] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FPTOI_F32_TO_NARROW,
                 LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_U32_F32,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_3(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_U32_F32,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_FPTOUI_F32_TO_I16] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FPTOI_F32_TO_NARROW,
                 LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_U32_F32,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_3(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_U32_F32,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT)},
};

#undef LOOM_AMDGPU_SCALAR_CONVERSION_REFS_0
#undef LOOM_AMDGPU_SCALAR_CONVERSION_REFS_1
#undef LOOM_AMDGPU_SCALAR_CONVERSION_REFS_2
#undef LOOM_AMDGPU_SCALAR_CONVERSION_REFS_3

#define LOOM_AMDGPU_SCALAR_OP_INDEX(op_kind) ((uint8_t)((op_kind) & 0xFFu))
#define LOOM_AMDGPU_SCALAR_CONVERSION_OP_GROUP_CODE(op_group) \
  ((uint8_t)((op_group) + 1u))
#define LOOM_AMDGPU_SCALAR_CONVERSION_OP_GROUP_ROW(op, group) \
  [LOOM_AMDGPU_SCALAR_OP_INDEX(LOOM_OP_SCALAR_##op)] =        \
      LOOM_AMDGPU_SCALAR_CONVERSION_OP_GROUP_CODE(            \
          LOOM_AMDGPU_SCALAR_CONVERSION_OP_##group)

static_assert(LOOM_AMDGPU_SCALAR_CONVERSION_OP_COUNT_ < UINT8_MAX,
              "scalar conversion op-group codes must fit in uint8_t");

static const uint8_t
    kAmdgpuScalarConversionOpGroupCodeByScalarOp[LOOM_OP_SCALAR_COUNT_] = {
        LOOM_AMDGPU_SCALAR_CONVERSION_OP_GROUP_ROW(TRUNCI, TRUNCI),
        LOOM_AMDGPU_SCALAR_CONVERSION_OP_GROUP_ROW(EXTF, EXTF),
        LOOM_AMDGPU_SCALAR_CONVERSION_OP_GROUP_ROW(EXTSI, EXTSI),
        LOOM_AMDGPU_SCALAR_CONVERSION_OP_GROUP_ROW(EXTUI, EXTUI),
        LOOM_AMDGPU_SCALAR_CONVERSION_OP_GROUP_ROW(UITOFP, UITOFP),
        LOOM_AMDGPU_SCALAR_CONVERSION_OP_GROUP_ROW(FPTOSI, FPTOSI),
        LOOM_AMDGPU_SCALAR_CONVERSION_OP_GROUP_ROW(FPTOUI, FPTOUI),
};

#undef LOOM_AMDGPU_SCALAR_OP_INDEX
#undef LOOM_AMDGPU_SCALAR_CONVERSION_OP_GROUP_CODE
#undef LOOM_AMDGPU_SCALAR_CONVERSION_OP_GROUP_ROW

static loom_amdgpu_scalar_conversion_op_group_t
loom_amdgpu_scalar_conversion_op_group(loom_op_kind_t op_kind) {
  if (loom_op_dialect_id(op_kind) != LOOM_DIALECT_SCALAR) {
    return LOOM_AMDGPU_SCALAR_CONVERSION_OP_COUNT_;
  }
  const uint8_t op_index = loom_op_dialect_index(op_kind);
  if (op_index >=
      IREE_ARRAYSIZE(kAmdgpuScalarConversionOpGroupCodeByScalarOp)) {
    return LOOM_AMDGPU_SCALAR_CONVERSION_OP_COUNT_;
  }
  const uint8_t op_group_code =
      kAmdgpuScalarConversionOpGroupCodeByScalarOp[op_index];
  if (op_group_code == 0) {
    return LOOM_AMDGPU_SCALAR_CONVERSION_OP_COUNT_;
  }
  return (loom_amdgpu_scalar_conversion_op_group_t)(op_group_code - 1u);
}

static const loom_amdgpu_scalar_conversion_rule_t*
loom_amdgpu_scalar_conversion_rule_for(
    loom_amdgpu_scalar_conversion_op_group_t op_group,
    loom_scalar_type_t source_type, loom_scalar_type_t result_type) {
  if (op_group >= LOOM_AMDGPU_SCALAR_CONVERSION_OP_COUNT_ ||
      source_type >= LOOM_SCALAR_TYPE_COUNT_ ||
      result_type >= LOOM_SCALAR_TYPE_COUNT_) {
    return NULL;
  }
  const uint8_t rule_index =
      kLoomAmdgpuScalarConversionRuleIndexes[op_group][source_type]
                                            [result_type];
  if (rule_index == LOOM_AMDGPU_SCALAR_CONVERSION_RULE_NONE) {
    return NULL;
  }
  IREE_ASSERT(rule_index < IREE_ARRAYSIZE(kLoomAmdgpuScalarConversionRules));
  return &kLoomAmdgpuScalarConversionRules[rule_index];
}

static bool loom_amdgpu_select_scalar_conversion_plan_from_table(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set, const loom_op_t* source_op,
    loom_amdgpu_scalar_conversion_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_scalar_conversion_plan_t){0};
  if (source_op->operand_count != 1 || source_op->result_count != 1) {
    return false;
  }

  const loom_value_id_t source = loom_op_const_operands(source_op)[0];
  const loom_value_id_t result = loom_op_const_results(source_op)[0];
  const loom_scalar_type_t source_type =
      loom_amdgpu_scalar_type_or_none(loom_module_value_type(module, source));
  const loom_scalar_type_t result_type =
      loom_amdgpu_scalar_type_or_none(loom_module_value_type(module, result));
  if (source_type == LOOM_SCALAR_TYPE_COUNT_ ||
      result_type == LOOM_SCALAR_TYPE_COUNT_) {
    return false;
  }

  const loom_amdgpu_scalar_conversion_op_group_t op_group =
      loom_amdgpu_scalar_conversion_op_group(source_op->kind);
  const loom_amdgpu_scalar_conversion_rule_t* rule =
      loom_amdgpu_scalar_conversion_rule_for(op_group, source_type,
                                             result_type);
  if (rule == NULL || !loom_amdgpu_descriptor_set_has_all_refs(
                          descriptor_set, rule->required_descriptor_refs,
                          IREE_ARRAYSIZE(rule->required_descriptor_refs))) {
    return false;
  }
  *out_plan = (loom_amdgpu_scalar_conversion_plan_t){
      .kind = rule->kind,
      .source = source,
      .result = result,
      .source_bit_count =
          loom_amdgpu_integer_scalar_type_bit_count(source_type),
      .result_bit_count =
          loom_amdgpu_integer_scalar_type_bit_count(result_type),
      .convert_descriptor_ref = rule->convert_descriptor_ref,
  };
  return true;
}

iree_status_t loom_amdgpu_select_scalar_conversion_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_scalar_conversion_plan_t* out_plan, bool* out_selected) {
  *out_selected = loom_amdgpu_select_scalar_conversion_plan_from_table(
      loom_low_lower_context_module(context),
      loom_low_lower_context_descriptor_set(context), source_op, out_plan);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_bind_selected_value_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const void* local_plan, iree_host_size_t local_plan_size,
    loom_low_lower_plan_t* out_plan) {
  void* plan_data = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_allocate_plan_data(context, local_plan_size, &plan_data));
  memcpy(plan_data, local_plan, local_plan_size);
  *out_plan = loom_low_lower_plan_make(source_op->kind, plan_data);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_low_legality_verify_scalar_conversion(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  (void)provider;
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }

  loom_amdgpu_scalar_conversion_plan_t plan = {0};
  if (!loom_amdgpu_select_scalar_conversion_plan_from_table(
          loom_target_low_legality_module(context),
          loom_target_low_legality_descriptor_set(context), op, &plan)) {
    return iree_ok_status();
  }
  *out_handled = true;
  return iree_ok_status();
}

typedef uint16_t loom_amdgpu_scalar_type_set_t;
static_assert(LOOM_SCALAR_TYPE_COUNT_ <= 16,
              "scalar type sets must fit in uint16_t");

enum {
  LOOM_AMDGPU_VECTOR_CONVERSION_LANE_RULE_SIGN_EXTEND_PACKED_SOURCE = 1u << 0,
  LOOM_AMDGPU_VECTOR_CONVERSION_LANE_RULE_SOURCE_WIDER_THAN_RESULT = 1u << 1,
};
typedef uint8_t loom_amdgpu_vector_conversion_lane_rule_flags_t;

typedef struct loom_amdgpu_vector_conversion_lane_rule_t {
  // Descriptor emitted by lanes that perform a numeric conversion packet.
  loom_amdgpu_descriptor_ref_t convert_descriptor_ref;
  // Allowed source scalar element types.
  loom_amdgpu_scalar_type_set_t source_element_types;
  // Allowed result scalar element types.
  loom_amdgpu_scalar_type_set_t result_element_types;
  // Bitfield of LOOM_AMDGPU_VECTOR_CONVERSION_LANE_RULE_* flags.
  loom_amdgpu_vector_conversion_lane_rule_flags_t flags;
} loom_amdgpu_vector_conversion_lane_rule_t;

#define LOOM_AMDGPU_SCALAR_TYPE_SET_ONE(type) \
  ((loom_amdgpu_scalar_type_set_t)(1u << (type)))
#define LOOM_AMDGPU_SCALAR_TYPE_SET_INTEGER                \
  (LOOM_AMDGPU_SCALAR_TYPE_SET_ONE(LOOM_SCALAR_TYPE_I8) |  \
   LOOM_AMDGPU_SCALAR_TYPE_SET_ONE(LOOM_SCALAR_TYPE_I16) | \
   LOOM_AMDGPU_SCALAR_TYPE_SET_ONE(LOOM_SCALAR_TYPE_I32) | \
   LOOM_AMDGPU_SCALAR_TYPE_SET_ONE(LOOM_SCALAR_TYPE_I64))
#define LOOM_AMDGPU_SCALAR_TYPE_SET_INTEGER_LE32           \
  (LOOM_AMDGPU_SCALAR_TYPE_SET_ONE(LOOM_SCALAR_TYPE_I8) |  \
   LOOM_AMDGPU_SCALAR_TYPE_SET_ONE(LOOM_SCALAR_TYPE_I16) | \
   LOOM_AMDGPU_SCALAR_TYPE_SET_ONE(LOOM_SCALAR_TYPE_I32))

#define LOOM_AMDGPU_VECTOR_OP_INDEX(op_kind) ((uint8_t)((op_kind) & 0xFFu))
#define LOOM_AMDGPU_VECTOR_CONVERSION_LANE_RULE_ROW(                           \
    op, descriptor_ref_, source_element_types_, result_element_types_, flags_) \
  [LOOM_AMDGPU_VECTOR_OP_INDEX(LOOM_OP_VECTOR_##op)] = {                       \
      .convert_descriptor_ref = (descriptor_ref_),                             \
      .source_element_types = (source_element_types_),                         \
      .result_element_types = (result_element_types_),                         \
      .flags = (flags_),                                                       \
  }

static const loom_amdgpu_vector_conversion_lane_rule_t
    kAmdgpuVectorConversionLaneRulesByVectorOp[LOOM_OP_VECTOR_COUNT_] = {
        LOOM_AMDGPU_VECTOR_CONVERSION_LANE_RULE_ROW(
            TRUNCI, LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
            LOOM_AMDGPU_SCALAR_TYPE_SET_INTEGER,
            LOOM_AMDGPU_SCALAR_TYPE_SET_INTEGER,
            LOOM_AMDGPU_VECTOR_CONVERSION_LANE_RULE_SOURCE_WIDER_THAN_RESULT),
        LOOM_AMDGPU_VECTOR_CONVERSION_LANE_RULE_ROW(
            SITOFP, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F32_I32,
            LOOM_AMDGPU_SCALAR_TYPE_SET_INTEGER_LE32,
            LOOM_AMDGPU_SCALAR_TYPE_SET_ONE(LOOM_SCALAR_TYPE_F32),
            LOOM_AMDGPU_VECTOR_CONVERSION_LANE_RULE_SIGN_EXTEND_PACKED_SOURCE),
        LOOM_AMDGPU_VECTOR_CONVERSION_LANE_RULE_ROW(
            UITOFP, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F32_U32,
            LOOM_AMDGPU_SCALAR_TYPE_SET_INTEGER_LE32,
            LOOM_AMDGPU_SCALAR_TYPE_SET_ONE(LOOM_SCALAR_TYPE_F32), 0),
        LOOM_AMDGPU_VECTOR_CONVERSION_LANE_RULE_ROW(
            FPTOSI, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_I32_F32,
            LOOM_AMDGPU_SCALAR_TYPE_SET_ONE(LOOM_SCALAR_TYPE_F32),
            LOOM_AMDGPU_SCALAR_TYPE_SET_INTEGER_LE32, 0),
        LOOM_AMDGPU_VECTOR_CONVERSION_LANE_RULE_ROW(
            FPTOUI, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_U32_F32,
            LOOM_AMDGPU_SCALAR_TYPE_SET_ONE(LOOM_SCALAR_TYPE_F32),
            LOOM_AMDGPU_SCALAR_TYPE_SET_INTEGER_LE32, 0),
};

#undef LOOM_AMDGPU_VECTOR_OP_INDEX
#undef LOOM_AMDGPU_VECTOR_CONVERSION_LANE_RULE_ROW
#undef LOOM_AMDGPU_SCALAR_TYPE_SET_ONE
#undef LOOM_AMDGPU_SCALAR_TYPE_SET_INTEGER
#undef LOOM_AMDGPU_SCALAR_TYPE_SET_INTEGER_LE32

static bool loom_amdgpu_scalar_type_set_contains(
    loom_amdgpu_scalar_type_set_t set, loom_scalar_type_t type) {
  if (type >= LOOM_SCALAR_TYPE_COUNT_) {
    return false;
  }
  return iree_all_bits_set(set, (loom_amdgpu_scalar_type_set_t)(1u << type));
}

static const loom_amdgpu_vector_conversion_lane_rule_t*
loom_amdgpu_vector_conversion_lane_rule(
    loom_op_kind_t op_kind, loom_scalar_type_t source_element_type,
    loom_scalar_type_t result_element_type) {
  if (loom_op_dialect_id(op_kind) != LOOM_DIALECT_VECTOR) {
    return NULL;
  }
  const uint8_t op_index = loom_op_dialect_index(op_kind);
  if (op_index >= IREE_ARRAYSIZE(kAmdgpuVectorConversionLaneRulesByVectorOp)) {
    return NULL;
  }
  const loom_amdgpu_vector_conversion_lane_rule_t* rule =
      &kAmdgpuVectorConversionLaneRulesByVectorOp[op_index];
  if (!loom_amdgpu_scalar_type_set_contains(rule->source_element_types,
                                            source_element_type) ||
      !loom_amdgpu_scalar_type_set_contains(rule->result_element_types,
                                            result_element_type)) {
    return NULL;
  }
  if (iree_all_bits_set(
          rule->flags,
          LOOM_AMDGPU_VECTOR_CONVERSION_LANE_RULE_SOURCE_WIDER_THAN_RESULT) &&
      loom_amdgpu_integer_scalar_type_bit_count(source_element_type) <=
          loom_amdgpu_integer_scalar_type_bit_count(result_element_type)) {
    return NULL;
  }
  return rule;
}

typedef uint8_t loom_amdgpu_vector_conversion_kind_flags_t;

enum {
  LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_SOURCE = 1u << 0,
  LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_RESULT = 1u << 1,
  LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_SOURCE_MATERIALIZATION = 1u << 2,
};

static bool loom_amdgpu_vector_conversion_can_use_packed_i8_permute(
    const loom_amdgpu_vector_storage_t* result_storage) {
  return result_storage->kind ==
             LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_INTEGER &&
         result_storage->element_bit_count == 8 &&
         result_storage->element_count == result_storage->register_count * 4u;
}

static const loom_amdgpu_vector_conversion_kind_t kAmdgpuVectorConversionKindByStorage
    [LOOM_AMDGPU_VECTOR_STORAGE_KIND_COUNT_]
    [LOOM_AMDGPU_VECTOR_STORAGE_KIND_COUNT_] = {
        [LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_32BIT]
            [LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_32BIT] =
                LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_32_TO_FULL_32,
        [LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_32BIT]
            [LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_INTEGER] =
                LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_32_TO_PACKED_INTEGER,
        [LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_64BIT]
            [LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_32BIT] =
                LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_64_TO_FULL_32,
        [LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_64BIT]
            [LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_INTEGER] =
                LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_64_TO_PACKED_INTEGER,
        [LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_INTEGER]
            [LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_32BIT] =
                LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_INTEGER_TO_FULL_32,
        [LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_INTEGER]
            [LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_INTEGER] =
                LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_INTEGER_TO_PACKED_INTEGER,
};

static const loom_amdgpu_vector_conversion_kind_flags_t
    kAmdgpuVectorConversionKindFlags[LOOM_AMDGPU_VECTOR_CONVERSION_KIND_COUNT_] = {
        [LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_32_TO_FULL_32] =
            LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_SOURCE_MATERIALIZATION,
        [LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_64_TO_FULL_32] =
            LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_SOURCE_MATERIALIZATION,
        [LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_32_TO_PACKED_INTEGER] =
            LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_RESULT |
            LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_SOURCE_MATERIALIZATION,
        [LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_64_TO_PACKED_INTEGER] =
            LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_RESULT |
            LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_SOURCE_MATERIALIZATION,
        [LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_INTEGER_TO_FULL_32] =
            LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_SOURCE,
        [LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_INTEGER_TO_PACKED_INTEGER] =
            LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_SOURCE |
            LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_RESULT,
};

static loom_amdgpu_vector_conversion_kind_flags_t
loom_amdgpu_vector_conversion_kind_flags(
    loom_amdgpu_vector_conversion_kind_t kind) {
  if (kind >= LOOM_AMDGPU_VECTOR_CONVERSION_KIND_COUNT_) {
    return 0;
  }
  return kAmdgpuVectorConversionKindFlags[kind];
}

static bool loom_amdgpu_vector_conversion_select_storage_kind(
    const loom_amdgpu_vector_storage_t* source_storage,
    const loom_amdgpu_vector_storage_t* result_storage,
    loom_amdgpu_vector_conversion_kind_t* out_kind) {
  *out_kind = LOOM_AMDGPU_VECTOR_CONVERSION_KIND_NONE;
  if (source_storage->kind >= LOOM_AMDGPU_VECTOR_STORAGE_KIND_COUNT_ ||
      result_storage->kind >= LOOM_AMDGPU_VECTOR_STORAGE_KIND_COUNT_) {
    return false;
  }
  *out_kind = kAmdgpuVectorConversionKindByStorage[source_storage->kind]
                                                  [result_storage->kind];
  return *out_kind != LOOM_AMDGPU_VECTOR_CONVERSION_KIND_NONE;
}

static bool loom_amdgpu_vector_conversion_descriptor_refs_present(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_vector_conversion_kind_t kind,
    loom_amdgpu_descriptor_ref_t convert_descriptor_ref,
    bool sign_extend_packed_source) {
  const loom_amdgpu_vector_conversion_kind_flags_t kind_flags =
      loom_amdgpu_vector_conversion_kind_flags(kind);
  if (convert_descriptor_ref != LOOM_AMDGPU_DESCRIPTOR_REF_NONE &&
      !loom_amdgpu_descriptor_set_has_ref(descriptor_set,
                                          convert_descriptor_ref)) {
    return false;
  }

  static const loom_amdgpu_descriptor_ref_t kFullSourceRefs[] = {
      LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_COPY,
  };
  if (convert_descriptor_ref != LOOM_AMDGPU_DESCRIPTOR_REF_NONE &&
      iree_any_bit_set(
          kind_flags,
          LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_SOURCE_MATERIALIZATION) &&
      !loom_amdgpu_descriptor_set_has_all_refs(
          descriptor_set, kFullSourceRefs, IREE_ARRAYSIZE(kFullSourceRefs))) {
    return false;
  }

  static const loom_amdgpu_descriptor_ref_t kPackedResultRefs[] = {
      LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32,
  };
  if (iree_any_bit_set(kind_flags,
                       LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_RESULT) &&
      !loom_amdgpu_descriptor_set_has_all_refs(
          descriptor_set, kPackedResultRefs,
          IREE_ARRAYSIZE(kPackedResultRefs))) {
    return false;
  }

  static const loom_amdgpu_descriptor_ref_t kPackedSourceSignedRefs[] = {
      LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT,
  };
  static const loom_amdgpu_descriptor_ref_t kPackedSourceUnsignedRefs[] = {
      LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
  };
  if (!iree_any_bit_set(kind_flags,
                        LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_SOURCE)) {
    return true;
  }
  if (sign_extend_packed_source) {
    return loom_amdgpu_descriptor_set_has_all_refs(
        descriptor_set, kPackedSourceSignedRefs,
        IREE_ARRAYSIZE(kPackedSourceSignedRefs));
  }
  return loom_amdgpu_descriptor_set_has_all_refs(
      descriptor_set, kPackedSourceUnsignedRefs,
      IREE_ARRAYSIZE(kPackedSourceUnsignedRefs));
}

static bool loom_amdgpu_select_vector_conversion_plan_for_op(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set, const loom_op_t* source_op,
    loom_amdgpu_vector_conversion_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_vector_conversion_plan_t){0};
  if (source_op->operand_count != 1 || source_op->result_count != 1) {
    return false;
  }

  const loom_value_id_t source = loom_op_const_operands(source_op)[0];
  const loom_value_id_t result = loom_op_const_results(source_op)[0];
  loom_amdgpu_vector_storage_t source_storage = {0};
  loom_amdgpu_vector_storage_t result_storage = {0};
  if (!loom_amdgpu_type_vector_storage(loom_module_value_type(module, source),
                                       &source_storage) ||
      !loom_amdgpu_type_vector_storage(loom_module_value_type(module, result),
                                       &result_storage) ||
      source_storage.element_count != result_storage.element_count) {
    return false;
  }

  const loom_amdgpu_vector_conversion_lane_rule_t* lane_rule =
      loom_amdgpu_vector_conversion_lane_rule(source_op->kind,
                                              source_storage.element_type,
                                              result_storage.element_type);
  if (lane_rule == NULL) {
    return false;
  }
  const bool sign_extend_packed_source = iree_all_bits_set(
      lane_rule->flags,
      LOOM_AMDGPU_VECTOR_CONVERSION_LANE_RULE_SIGN_EXTEND_PACKED_SOURCE);

  loom_amdgpu_vector_conversion_kind_t kind =
      LOOM_AMDGPU_VECTOR_CONVERSION_KIND_NONE;
  if (!loom_amdgpu_vector_conversion_select_storage_kind(
          &source_storage, &result_storage, &kind)) {
    return false;
  }
  if (!loom_amdgpu_vector_conversion_descriptor_refs_present(
          descriptor_set, kind, lane_rule->convert_descriptor_ref,
          sign_extend_packed_source)) {
    return false;
  }

  loom_amdgpu_descriptor_ref_t packed_i8_permute_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  if (loom_amdgpu_vector_conversion_can_use_packed_i8_permute(
          &result_storage) &&
      loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_PERM_B32_SRC2_LIT)) {
    packed_i8_permute_descriptor_ref =
        LOOM_AMDGPU_DESCRIPTOR_REF_V_PERM_B32_SRC2_LIT;
  }

  *out_plan = (loom_amdgpu_vector_conversion_plan_t){
      .source = source,
      .result = result,
      .kind = kind,
      .source_element_type = source_storage.element_type,
      .result_element_type = result_storage.element_type,
      .source_bit_count = source_storage.element_bit_count,
      .result_bit_count = result_storage.element_bit_count,
      .lane_count = source_storage.element_count,
      .source_register_count = source_storage.register_count,
      .result_register_count = result_storage.register_count,
      .source_element_register_count = source_storage.element_register_count,
      .convert_descriptor_ref = lane_rule->convert_descriptor_ref,
      .packed_i8_permute_descriptor_ref = packed_i8_permute_descriptor_ref,
      .sign_extend_packed_source = sign_extend_packed_source,
  };
  return true;
}

iree_status_t loom_amdgpu_select_vector_conversion_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_conversion_plan_t* out_plan, bool* out_selected) {
  *out_selected = loom_amdgpu_select_vector_conversion_plan_for_op(
      loom_low_lower_context_module(context),
      loom_low_lower_context_descriptor_set(context), source_op, out_plan);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_low_legality_verify_vector_conversion(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  (void)provider;
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }

  loom_amdgpu_vector_conversion_plan_t plan = {0};
  if (!loom_amdgpu_select_vector_conversion_plan_for_op(
          loom_target_low_legality_module(context),
          loom_target_low_legality_descriptor_set(context), op, &plan)) {
    return iree_ok_status();
  }
  *out_handled = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_structural_value_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan) {
  *out_plan = loom_low_lower_plan_empty();
  switch (source_op->kind) {
    case LOOM_OP_VECTOR_IOTA: {
      if (loom_amdgpu_select_fact_only_vector_atomic_offset_plan(
              context, source_op, out_plan)) {
        return iree_ok_status();
      }
      loom_amdgpu_vector_iota_plan_t local_plan = {0};
      bool selected = false;
      IREE_RETURN_IF_ERROR(loom_amdgpu_select_vector_iota_plan(
          context, source_op, &local_plan, &selected));
      if (!selected) {
        return iree_ok_status();
      }
      return loom_amdgpu_bind_selected_value_plan(
          context, source_op, &local_plan, sizeof(local_plan), out_plan);
    }
    case LOOM_OP_VECTOR_FROM_ELEMENTS: {
      if (loom_amdgpu_select_fact_only_vector_atomic_offset_plan(
              context, source_op, out_plan)) {
        return iree_ok_status();
      }
      loom_amdgpu_vector_from_elements_plan_t local_plan = {0};
      bool selected = false;
      IREE_RETURN_IF_ERROR(loom_amdgpu_select_vector_from_elements_plan(
          context, source_op, &local_plan, &selected));
      if (selected) {
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_select_vector_from_elements_fma_mix_half_results(
                context, source_op, LOOM_AMDGPU_VECTOR_INSERT_SELECT_FLAG_NONE,
                &local_plan));
        return loom_amdgpu_bind_selected_value_plan(
            context, source_op, &local_plan, sizeof(local_plan), out_plan);
      }
      return iree_ok_status();
    }
    case LOOM_OP_VECTOR_SPLAT: {
      loom_amdgpu_vector_from_elements_plan_t local_plan = {0};
      bool selected = false;
      IREE_RETURN_IF_ERROR(loom_amdgpu_select_vector_splat_plan(
          context, source_op, &local_plan, &selected));
      if (selected) {
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_select_vector_from_elements_fma_mix_half_results(
                context, source_op, LOOM_AMDGPU_VECTOR_INSERT_SELECT_FLAG_NONE,
                &local_plan));
        return loom_amdgpu_bind_selected_value_plan(
            context, source_op, &local_plan, sizeof(local_plan), out_plan);
      }
      return iree_ok_status();
    }
    case LOOM_OP_VECTOR_INSERT: {
      loom_amdgpu_vector_insert_plan_t local_plan = {0};
      bool selected = false;
      IREE_RETURN_IF_ERROR(loom_amdgpu_select_vector_insert_plan(
          context, source_op, LOOM_AMDGPU_VECTOR_INSERT_SELECT_FLAG_NONE,
          &local_plan, &selected));
      if (!selected) {
        return iree_ok_status();
      }
      return loom_amdgpu_bind_selected_value_plan(
          context, source_op, &local_plan, sizeof(local_plan), out_plan);
    }
    default:
      return iree_ok_status();
  }
}

iree_status_t loom_amdgpu_preselect_structural_value_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan) {
  *out_plan = loom_low_lower_plan_empty();
  if (loom_vector_from_elements_isa(source_op)) {
    loom_amdgpu_vector_from_elements_plan_t local_plan = {0};
    bool selected = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_select_vector_from_elements_plan(
        context, source_op, &local_plan, &selected));
    if (!selected) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_select_vector_from_elements_fma_mix_half_results(
            context, source_op,
            LOOM_AMDGPU_VECTOR_INSERT_SELECT_FLAG_EMIT_DIAGNOSTICS,
            &local_plan));
    if (local_plan.fma_mix_half_result_lane_mask == 0) {
      return iree_ok_status();
    }
    return loom_amdgpu_bind_selected_value_plan(context, source_op, &local_plan,
                                                sizeof(local_plan), out_plan);
  }
  if (loom_vector_splat_isa(source_op)) {
    loom_amdgpu_vector_from_elements_plan_t local_plan = {0};
    bool selected = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_select_vector_splat_plan(
        context, source_op, &local_plan, &selected));
    if (!selected) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_select_vector_from_elements_fma_mix_half_results(
            context, source_op,
            LOOM_AMDGPU_VECTOR_INSERT_SELECT_FLAG_EMIT_DIAGNOSTICS,
            &local_plan));
    if (local_plan.fma_mix_half_result_lane_mask == 0) {
      return iree_ok_status();
    }
    return loom_amdgpu_bind_selected_value_plan(context, source_op, &local_plan,
                                                sizeof(local_plan), out_plan);
  }
  if (!loom_vector_insert_isa(source_op)) {
    return iree_ok_status();
  }

  loom_amdgpu_vector_insert_plan_t local_plan = {0};
  bool selected = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_vector_insert_plan(
      context, source_op,
      LOOM_AMDGPU_VECTOR_INSERT_SELECT_FLAG_EMIT_DIAGNOSTICS, &local_plan,
      &selected));
  if (!selected ||
      local_plan.value_kind !=
          LOOM_AMDGPU_VECTOR_INSERT_VALUE_KIND_FMA_MIX_HALF_RESULT) {
    return iree_ok_status();
  }

  return loom_amdgpu_bind_selected_value_plan(context, source_op, &local_plan,
                                              sizeof(local_plan), out_plan);
}

static iree_status_t loom_amdgpu_bind_register_u32_lane_constants(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_result,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_string_id_t imm32_attr_name_id, const uint32_t* lane_bit_patterns,
    uint32_t lane_count) {
  IREE_ASSERT_GT(lane_count, 0);
  IREE_ASSERT_LE(lane_count, LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES);

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, source_result, &result_type));
  IREE_ASSERT(loom_low_type_is_register(result_type));
  IREE_ASSERT_EQ(loom_low_register_type_unit_count(result_type), lane_count);
  const loom_type_t lane_type =
      loom_low_register_type_with_unit_count(result_type, 1);

  loom_value_id_t low_lane_values[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < lane_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_const_u32(
        context, source_op, descriptor, imm32_attr_name_id,
        lane_bit_patterns[i], lane_type, &low_lane_values[i]));
  }

  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_low_register_range(
      context, source_op, low_lane_values, lane_count, result_type,
      &low_result));
  return loom_low_lower_bind_value(context, source_result, low_result);
}

static iree_status_t loom_amdgpu_lower_u32_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_string_id_t imm32_attr_name_id, uint32_t bit_pattern,
    loom_value_id_t source_result) {
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, source_result, &result_type));

  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_const_u32(
      context, source_op, descriptor, imm32_attr_name_id, bit_pattern,
      result_type, &low_result));
  return loom_low_lower_bind_value(context, source_result, low_result);
}

static iree_status_t loom_amdgpu_lower_i1_scc_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_constant_plan_t* plan) {
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op,
                                                   plan->result, &result_type));

  loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_const_u32(
      context, source_op, &plan->zero_descriptor, plan->imm32_attr_name_id, 0,
      sgpr_type, &zero));
  const loom_value_id_t operands[] = {zero, zero};
  loom_op_t* compare_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), &result_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &compare_op));
  return loom_low_lower_bind_value(
      context, plan->result,
      loom_value_slice_get(loom_low_op_results(compare_op), 0));
}

static iree_status_t loom_amdgpu_lower_i1_mask_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_constant_plan_t* plan) {
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op,
                                                   plan->result, &result_type));
  if (plan->i1_value) {
    loom_op_t* exec_read_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
        context, &plan->descriptor,
        /*operands=*/NULL, /*operand_count=*/0, loom_named_attr_slice_empty(),
        &result_type, 1, /*tied_results=*/NULL, /*tied_result_count=*/0,
        source_op->location, &exec_read_op));
    return loom_low_lower_bind_value(
        context, plan->result,
        loom_value_slice_get(loom_low_op_results(exec_read_op), 0));
  }

  const uint32_t bit_patterns[] = {0, 0};
  return loom_amdgpu_bind_register_u32_lane_constants(
      context, source_op, plan->result, &plan->zero_descriptor,
      plan->imm32_attr_name_id, bit_patterns, IREE_ARRAYSIZE(bit_patterns));
}

iree_status_t loom_amdgpu_lower_constant_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_constant_plan_t* plan) {
  switch (plan->kind) {
    case LOOM_AMDGPU_CONSTANT_PLAN_KIND_U32_BITS:
      break;
    case LOOM_AMDGPU_CONSTANT_PLAN_KIND_I1_SCC:
      return loom_amdgpu_lower_i1_scc_constant(context, source_op, plan);
    case LOOM_AMDGPU_CONSTANT_PLAN_KIND_I1_MASK:
      return loom_amdgpu_lower_i1_mask_constant(context, source_op, plan);
    case LOOM_AMDGPU_CONSTANT_PLAN_KIND_NONE:
      IREE_ASSERT_UNREACHABLE("invalid AMDGPU constant plan kind");
      return iree_ok_status();
  }
  if (plan->register_count == 1) {
    return loom_amdgpu_lower_u32_constant(context, source_op, &plan->descriptor,
                                          plan->imm32_attr_name_id,
                                          plan->bit_patterns[0], plan->result);
  }
  return loom_amdgpu_bind_register_u32_lane_constants(
      context, source_op, plan->result, &plan->descriptor,
      plan->imm32_attr_name_id, plan->bit_patterns, plan->register_count);
}

static iree_status_t loom_amdgpu_lower_vector_iota(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_iota_plan_t* plan) {
  if (!loom_amdgpu_vector_iota_plan_is_dynamic(plan)) {
    return loom_amdgpu_bind_register_u32_lane_constants(
        context, source_op, plan->result, &plan->descriptor,
        plan->imm32_attr_name_id, plan->lane_bit_patterns, plan->lane_count);
  }

  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  loom_value_id_t low_base = LOOM_VALUE_ID_INVALID;
  if (loom_amdgpu_vector_iota_plan_has_exact_base(plan)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        (uint32_t)plan->exact_base, lane_type, &low_base));
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i32(
        context, source_op, plan->base, &low_base));
  }

  const bool has_exact_step = loom_amdgpu_vector_iota_plan_has_exact_step(plan);
  const int32_t exact_step = plan->exact_step;
  loom_value_id_t low_step = LOOM_VALUE_ID_INVALID;
  if (!has_exact_step) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i32(
        context, source_op, plan->step, &low_step));
  }

  loom_value_id_t lanes[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES] = {0};
  lanes[0] = low_base;
  for (uint32_t i = 1; i < plan->lane_count; ++i) {
    if (has_exact_step) {
      int64_t lane_offset = 0;
      const bool lane_offset_in_range =
          iree_checked_mul_i64((int64_t)i, exact_step, &lane_offset) &&
          lane_offset >= INT32_MIN && lane_offset <= INT32_MAX;
      IREE_ASSERT(lane_offset_in_range);
      if (lane_offset == 0) {
        lanes[i] = low_base;
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT,
          low_base, (uint32_t)(int32_t)lane_offset, lane_type, &lanes[i]));
      continue;
    }

    loom_value_id_t scaled_step = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_scale_u32(
        context, source_op, low_step, i, LOOM_AMDGPU_VGPR_SCALE_U32_FLAG_NONE,
        lane_type, &scaled_step));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32, low_base,
        scaled_step, lane_type, &lanes[i]));
  }

  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             lanes, plan->lane_count);
}

static iree_status_t loom_amdgpu_extract_packed_register_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, const loom_amdgpu_vector_extract_plan_t* plan,
    uint32_t lane_offset, loom_amdgpu_bitfield_extract_mode_t mode,
    loom_type_t lane_type, loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT(plan->lane_bit_count == 8 || plan->lane_bit_count == 16);
  const uint32_t lanes_per_register = 32u / plan->lane_bit_count;
  const uint32_t register_offset = lane_offset / lanes_per_register;
  const uint32_t register_bit_offset =
      (lane_offset % lanes_per_register) * plan->lane_bit_count;
  loom_value_id_t source_register = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
      context, source_op, low_source, plan->register_count, register_offset,
      lane_type, &source_register));
  return loom_amdgpu_extract_vgpr_bitfield(
      context, source_op, source_register, register_bit_offset,
      plan->lane_bit_count, mode, lane_type, out_lane);
}

static iree_status_t loom_amdgpu_extract_vector_conversion_packed_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_conversion_plan_t* plan,
    loom_value_id_t low_source, uint32_t lane_index, loom_type_t lane_type,
    loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_vector_extract_plan_t extract_plan = {
      .source = plan->source,
      .result = plan->result,
      .lane_count = plan->lane_count,
      .register_count = plan->source_register_count,
      .result_register_count = 1,
      .element_register_count = 1,
      .lane_bit_count = plan->source_bit_count,
      .sign_extend_packed_lane = plan->sign_extend_packed_source,
  };
  loom_amdgpu_bitfield_extract_mode_t extract_mode =
      plan->sign_extend_packed_source
          ? LOOM_AMDGPU_BITFIELD_EXTRACT_MODE_SIGN_EXTEND
          : LOOM_AMDGPU_BITFIELD_EXTRACT_MODE_RAW_SHIFTED;
  if (plan->kind ==
          LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_INTEGER_TO_PACKED_INTEGER &&
      plan->convert_descriptor_ref == LOOM_AMDGPU_DESCRIPTOR_REF_NONE) {
    return loom_amdgpu_extract_packed_register_lane(
        context, source_op, low_source, &extract_plan, lane_index, extract_mode,
        lane_type, out_lane);
  }
  if (!plan->sign_extend_packed_source) {
    extract_mode = LOOM_AMDGPU_BITFIELD_EXTRACT_MODE_ZERO_EXTEND;
  }
  return loom_amdgpu_extract_packed_register_lane(
      context, source_op, low_source, &extract_plan, lane_index, extract_mode,
      lane_type, out_lane);
}

static iree_status_t loom_amdgpu_extract_vector_conversion_full_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_conversion_plan_t* plan,
    loom_value_id_t low_source, uint32_t lane_index,
    loom_type_t source_lane_type, loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  const uint32_t register_offset =
      lane_index * plan->source_element_register_count;
  return loom_amdgpu_extract_low_register_unit(
      context, source_op, low_source, plan->source_register_count,
      register_offset, source_lane_type, out_lane);
}

static iree_status_t loom_amdgpu_extract_vector_conversion_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_conversion_plan_t* plan,
    loom_value_id_t low_source, uint32_t lane_index,
    loom_type_t source_lane_type, loom_type_t lane_type,
    loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  const loom_amdgpu_vector_conversion_kind_flags_t kind_flags =
      loom_amdgpu_vector_conversion_kind_flags(plan->kind);
  if (iree_any_bit_set(kind_flags,
                       LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_SOURCE)) {
    return loom_amdgpu_extract_vector_conversion_packed_lane(
        context, source_op, plan, low_source, lane_index, lane_type, out_lane);
  }
  if (plan->kind != LOOM_AMDGPU_VECTOR_CONVERSION_KIND_NONE &&
      plan->kind < LOOM_AMDGPU_VECTOR_CONVERSION_KIND_COUNT_) {
    return loom_amdgpu_extract_vector_conversion_full_lane(
        context, source_op, plan, low_source, lane_index, source_lane_type,
        out_lane);
  }
  IREE_ASSERT_UNREACHABLE("invalid AMDGPU vector conversion source kind");
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_convert_vector_conversion_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_conversion_plan_t* plan, loom_value_id_t lane,
    loom_type_t lane_type, loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  if (plan->convert_descriptor_ref == LOOM_AMDGPU_DESCRIPTOR_REF_NONE) {
    *out_lane = lane;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_materialize_low_vgpr_b32(context, source_op, lane, &lane));
  return loom_amdgpu_emit_vgpr_unary(context, source_op,
                                     plan->convert_descriptor_ref, lane,
                                     lane_type, out_lane);
}

static iree_status_t loom_amdgpu_lower_vector_conversion_full_result(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_conversion_plan_t* plan,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t lane_type) {
  loom_value_id_t lanes[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < plan->lane_count; ++i) {
    loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_vector_conversion_lane(
        context, source_op, plan, low_source, i, source_lane_type, lane_type,
        &source_lane));
    IREE_RETURN_IF_ERROR(loom_amdgpu_convert_vector_conversion_lane(
        context, source_op, plan, source_lane, lane_type, &lanes[i]));
  }
  return loom_amdgpu_bind_low_register_range(
      context, source_op, plan->result, lanes, plan->result_register_count);
}

static iree_status_t loom_amdgpu_lower_vector_conversion_packed_result(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_conversion_plan_t* plan,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t lane_type) {
  if (plan->packed_i8_permute_descriptor_ref !=
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE) {
    loom_value_id_t converted_lanes[LOOM_AMDGPU_MAX_PACKED_I8_LANES];
    for (uint32_t i = 0; i < plan->lane_count; ++i) {
      loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_extract_vector_conversion_lane(
          context, source_op, plan, low_source, i, source_lane_type, lane_type,
          &source_lane));
      IREE_RETURN_IF_ERROR(loom_amdgpu_convert_vector_conversion_lane(
          context, source_op, plan, source_lane, lane_type,
          &converted_lanes[i]));
      IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
          context, source_op, converted_lanes[i], &converted_lanes[i]));
    }

    loom_low_lower_resolved_descriptor_t descriptor = {0};
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
        context, plan->packed_i8_permute_descriptor_ref, &descriptor));
    loom_value_id_t result_registers[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS];
    IREE_RETURN_IF_ERROR(loom_amdgpu_pack_i8_lanes_with_permute(
        context, source_op, &descriptor, converted_lanes, plan->lane_count,
        lane_type, result_registers));
    return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                               result_registers,
                                               plan->result_register_count);
  }

  loom_value_id_t result_registers[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS];
  const uint32_t lanes_per_register = 32u / plan->result_bit_count;
  for (uint32_t register_index = 0;
       register_index < plan->result_register_count; ++register_index) {
    loom_value_id_t packed = LOOM_VALUE_ID_INVALID;
    const uint32_t lane_base = register_index * lanes_per_register;
    for (uint32_t register_lane = 0; register_lane < lanes_per_register;
         ++register_lane) {
      const uint32_t lane_index = lane_base + register_lane;
      if (lane_index >= plan->lane_count) {
        break;
      }
      loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_extract_vector_conversion_lane(
          context, source_op, plan, low_source, lane_index, source_lane_type,
          lane_type, &source_lane));
      loom_value_id_t converted_lane = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_convert_vector_conversion_lane(
          context, source_op, plan, source_lane, lane_type, &converted_lane));
      IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
          context, source_op, converted_lane, &converted_lane));
      IREE_RETURN_IF_ERROR(loom_amdgpu_pack_lane_bits_into_register(
          context, source_op, converted_lane, plan->result_bit_count,
          register_lane * plan->result_bit_count, lane_type, &packed));
    }
    result_registers[register_index] = packed;
  }
  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             result_registers,
                                             plan->result_register_count);
}

iree_status_t loom_amdgpu_lower_vector_conversion(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_conversion_plan_t* plan) {
  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->source, &low_source));

  const loom_module_t* module = loom_low_lower_context_module(context);
  loom_type_t source_lane_type =
      loom_amdgpu_low_register_lane_type(module, low_source);
  if (loom_type_kind(source_lane_type) == LOOM_TYPE_NONE) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_vgpr_type(context, &source_lane_type));
  }
  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));

  const loom_amdgpu_vector_conversion_kind_flags_t kind_flags =
      loom_amdgpu_vector_conversion_kind_flags(plan->kind);
  if (iree_any_bit_set(kind_flags,
                       LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_RESULT)) {
    return loom_amdgpu_lower_vector_conversion_packed_result(
        context, source_op, plan, low_source, source_lane_type, lane_type);
  }
  if (plan->kind != LOOM_AMDGPU_VECTOR_CONVERSION_KIND_NONE &&
      plan->kind < LOOM_AMDGPU_VECTOR_CONVERSION_KIND_COUNT_) {
    return loom_amdgpu_lower_vector_conversion_full_result(
        context, source_op, plan, low_source, source_lane_type, lane_type);
  }
  IREE_ASSERT_UNREACHABLE("invalid AMDGPU vector conversion plan kind");
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_extract_vector_register_unit(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, const loom_amdgpu_vector_extract_plan_t* plan,
    uint32_t lane_offset, uint32_t result_register_index, loom_type_t unit_type,
    loom_value_id_t* out_register_unit) {
  *out_register_unit = LOOM_VALUE_ID_INVALID;
  if (plan->lane_bit_count < 32) {
    IREE_ASSERT_TRUE(result_register_index == 0);
    const loom_amdgpu_bitfield_extract_mode_t extract_mode =
        plan->sign_extend_packed_lane
            ? LOOM_AMDGPU_BITFIELD_EXTRACT_MODE_SIGN_EXTEND
            : LOOM_AMDGPU_BITFIELD_EXTRACT_MODE_RAW_SHIFTED;
    return loom_amdgpu_extract_packed_register_lane(
        context, source_op, low_source, plan, lane_offset, extract_mode,
        unit_type, out_register_unit);
  }

  IREE_ASSERT_TRUE(plan->lane_bit_count == 32 || plan->lane_bit_count == 64);
  const uint32_t register_offset =
      lane_offset * plan->element_register_count + result_register_index;
  IREE_ASSERT(register_offset < plan->register_count);
  return loom_amdgpu_extract_low_register_unit(
      context, source_op, low_source, plan->register_count, register_offset,
      unit_type, out_register_unit);
}

static iree_status_t loom_amdgpu_lower_static_vector_extract(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_extract_plan_t* plan) {
  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->source, &low_source));
  if (plan->lane_offset == 0 &&
      plan->result_register_count == plan->register_count &&
      !plan->sign_extend_packed_lane) {
    return loom_low_lower_bind_value(context, plan->result, low_source);
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  loom_type_t register_type =
      loom_amdgpu_low_register_lane_type(module, low_source);
  if (loom_type_kind(register_type) == LOOM_TYPE_NONE) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &register_type));
  }

  loom_value_id_t registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < plan->result_register_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_vector_register_unit(
        context, source_op, low_source, plan, plan->lane_offset, i,
        register_type, &registers[i]));
  }

  return loom_amdgpu_bind_low_register_range(
      context, source_op, plan->result, registers, plan->result_register_count);
}

static iree_status_t loom_amdgpu_lower_dynamic_vector_extract(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_extract_plan_t* plan) {
  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->source, &low_source));
  if (plan->lane_count == 1 && !plan->sign_extend_packed_lane) {
    return loom_low_lower_bind_value(context, plan->result, low_source);
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  loom_type_t source_lane_type =
      loom_amdgpu_low_register_lane_type(module, low_source);
  if (loom_type_kind(source_lane_type) == LOOM_TYPE_NONE) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_vgpr_type(context, &source_lane_type));
  }
  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  loom_type_t mask_lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &mask_lane_type));

  loom_value_id_t selected_registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t register_index = 0;
       register_index < plan->result_register_count; ++register_index) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_vector_register_unit(
        context, source_op, low_source, plan, 0, register_index,
        source_lane_type, &selected_registers[register_index]));
    if (!loom_type_equal(source_lane_type, lane_type)) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_b32_copy(
          context, source_op, selected_registers[register_index],
          &selected_registers[register_index]));
    }
  }

  loom_value_id_t index_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i32(
      context, source_op, plan->dynamic_index, &index_lane));
  for (uint32_t i = 1; i < plan->lane_count; ++i) {
    loom_value_id_t ordinal = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, i, lane_type,
        &ordinal));

    const loom_value_id_t compare_operands[] = {
        index_lane,
        ordinal,
    };
    loom_op_t* compare_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
        compare_operands, IREE_ARRAYSIZE(compare_operands),
        loom_make_named_attr_slice(NULL, 0), &mask_lane_type, 1, &compare_op));

    const loom_value_id_t condition =
        loom_value_slice_get(loom_low_op_results(compare_op), 0);
    for (uint32_t register_index = 0;
         register_index < plan->result_register_count; ++register_index) {
      loom_value_id_t table_lane = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_extract_vector_register_unit(
          context, source_op, low_source, plan, i, register_index,
          source_lane_type, &table_lane));
      if (!loom_type_equal(source_lane_type, lane_type)) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_b32_copy(
            context, source_op, table_lane, &table_lane));
      }
      const loom_value_id_t select_operands[] = {
          selected_registers[register_index],
          table_lane,
          condition,
      };
      loom_op_t* select_op = NULL;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32,
          select_operands, IREE_ARRAYSIZE(select_operands),
          loom_make_named_attr_slice(NULL, 0), &lane_type, 1, &select_op));
      selected_registers[register_index] =
          loom_value_slice_get(loom_low_op_results(select_op), 0);
    }
  }
  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             selected_registers,
                                             plan->result_register_count);
}

iree_status_t loom_amdgpu_lower_vector_extract(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_extract_plan_t* plan) {
  return plan->is_dynamic ? loom_amdgpu_lower_dynamic_vector_extract(
                                context, source_op, plan)
                          : loom_amdgpu_lower_static_vector_extract(
                                context, source_op, plan);
}

static iree_status_t loom_amdgpu_compose_vgpr_16bit_float_lane_bits(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_value, uint32_t register_bit_offset,
    loom_type_t lane_type, loom_value_id_t* out_low_value) {
  *out_low_value = low_value;
  if (loom_amdgpu_low_value_defines_vgpr_low16(context, low_value)) {
    if (register_bit_offset == 0) {
      return loom_amdgpu_emit_vgpr_unary(
          context, source_op,
          LOOM_AMDGPU_DESCRIPTOR_REF_V_BFE_U32_OFFSET_0_WIDTH_16_LOW16,
          low_value, lane_type, out_low_value);
    }
    if (register_bit_offset == 16) {
      return loom_amdgpu_emit_vgpr_unary(
          context, source_op,
          LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_SRC0_16_LOW16, low_value,
          lane_type, out_low_value);
    }
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
      context, source_op, low_value, out_low_value));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      *out_low_value, UINT32_C(0xFFFF), lane_type, out_low_value));
  return loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
      register_bit_offset, *out_low_value, lane_type, out_low_value);
}

static iree_status_t loom_amdgpu_lookup_or_compose_vgpr_16bit_float(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, uint32_t register_bit_offset,
    loom_type_t lane_type, loom_value_id_t* out_low_value) {
  *out_low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_value, out_low_value));
  return loom_amdgpu_compose_vgpr_16bit_float_lane_bits(
      context, source_op, *out_low_value, register_bit_offset, lane_type,
      out_low_value);
}

static iree_status_t loom_amdgpu_lookup_or_materialize_vgpr_16bit_float(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, loom_type_t lane_type,
    loom_value_id_t* out_low_value) {
  return loom_amdgpu_lookup_or_compose_vgpr_16bit_float(
      context, source_op, source_value, 0, lane_type, out_low_value);
}

static iree_status_t loom_amdgpu_lower_vector_from_16bit_element_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_from_elements_plan_t* plan, uint32_t lane_index,
    uint32_t bit_offset, loom_type_t lane_type, loom_value_id_t* out_lane) {
  return loom_amdgpu_lookup_or_compose_vgpr_16bit_float(
      context, source_op, plan->payload.elements[lane_index], bit_offset,
      lane_type, out_lane);
}

static iree_status_t loom_amdgpu_emit_zero_vgpr_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_type_t lane_type, loom_value_id_t* out_register) {
  return loom_amdgpu_emit_const_u32(context, source_op,
                                    LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0,
                                    lane_type, out_register);
}

static iree_status_t loom_amdgpu_emit_vector_from_elements_fma_mix_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_from_elements_plan_t* plan, uint32_t lane_index,
    loom_value_id_t accumulator, loom_type_t lane_type,
    loom_value_id_t* out_register) {
  IREE_ASSERT_ARGUMENT(plan->fma_mix_half_results);
  return loom_amdgpu_emit_tied_fma_mix_packet(
      context, source_op, &plan->fma_mix_half_results[lane_index], accumulator,
      lane_type, out_register);
}

static iree_status_t loom_amdgpu_lower_vector_from_16bit_elements(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_from_elements_plan_t* plan) {
  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));

  loom_value_id_t registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t register_index = 0; register_index < plan->register_count;
       ++register_index) {
    const uint32_t lane_base = register_index * 2u;
    const bool low_lane_uses_fma_mix =
        loom_amdgpu_vector_from_elements_uses_fma_mix_half_result(plan,
                                                                  lane_base);
    const bool high_lane_exists = lane_base + 1u < plan->element_count;
    const bool high_lane_uses_fma_mix =
        high_lane_exists &&
        loom_amdgpu_vector_from_elements_uses_fma_mix_half_result(
            plan, lane_base + 1u);
    loom_value_id_t packed = LOOM_VALUE_ID_INVALID;

    if (low_lane_uses_fma_mix) {
      if (high_lane_exists && !high_lane_uses_fma_mix) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_lower_vector_from_16bit_element_lane(
            context, source_op, plan, lane_base + 1u, 16, lane_type, &packed));
      } else {
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_zero_vgpr_register(
            context, source_op, lane_type, &packed));
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vector_from_elements_fma_mix_lane(
          context, source_op, plan, lane_base, packed, lane_type, &packed));
    } else {
      IREE_RETURN_IF_ERROR(loom_amdgpu_lower_vector_from_16bit_element_lane(
          context, source_op, plan, lane_base, 0, lane_type, &packed));
    }

    if (high_lane_exists && high_lane_uses_fma_mix) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vector_from_elements_fma_mix_lane(
          context, source_op, plan, lane_base + 1u, packed, lane_type,
          &packed));
    } else if (high_lane_exists && !low_lane_uses_fma_mix) {
      loom_value_id_t high_lane = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lower_vector_from_16bit_element_lane(
          context, source_op, plan, lane_base + 1u, 16, lane_type, &high_lane));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32, packed,
          high_lane, lane_type, &packed));
    }
    registers[register_index] = packed;
  }

  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             registers, plan->register_count);
}

static iree_status_t loom_amdgpu_lower_vector_from_packed_integer_elements(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_from_elements_plan_t* plan) {
  const uint32_t element_bit_count = plan->element_bit_count;
  IREE_ASSERT_TRUE(element_bit_count == 8 || element_bit_count == 16);
  const uint32_t element_mask = (UINT32_C(1) << element_bit_count) - 1u;
  const uint32_t elements_per_register = 32u / element_bit_count;

  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));

  if (plan->materialization_kind ==
      LOOM_AMDGPU_VECTOR_FROM_ELEMENTS_MATERIALIZATION_EXACT_PACKED_INTEGER) {
    loom_low_lower_resolved_descriptor_t descriptor = {0};
    loom_string_id_t imm32_attr_name_id = LOOM_STRING_ID_INVALID;
    bool descriptor_present = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_imm32_descriptor(
        context, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, &descriptor,
        &imm32_attr_name_id, &descriptor_present));
    if (!descriptor_present) {
      IREE_ASSERT_UNREACHABLE(
          "AMDGPU packed integer vector constant lowering requires v_mov_b32");
      IREE_BUILTIN_UNREACHABLE();
    }
    return loom_amdgpu_bind_register_u32_lane_constants(
        context, source_op, plan->result, &descriptor, imm32_attr_name_id,
        plan->payload.packed_register_bit_patterns, plan->register_count);
  }

  loom_value_id_t registers[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS];
  for (uint32_t register_index = 0; register_index < plan->register_count;
       ++register_index) {
    loom_value_id_t packed = LOOM_VALUE_ID_INVALID;
    const uint32_t lane_base = register_index * elements_per_register;
    for (uint32_t lane_index = 0; lane_index < elements_per_register;
         ++lane_index) {
      const uint32_t element_index = lane_base + lane_index;
      if (element_index >= plan->element_count) {
        break;
      }

      loom_value_id_t low_element = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
          context, plan->payload.elements[element_index], &low_element));
      IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
          context, source_op, low_element, &low_element));

      loom_value_id_t low_bits = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
          low_element, element_mask, lane_type, &low_bits));
      loom_value_id_t shifted = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
          lane_index * element_bit_count, low_bits, lane_type, &shifted));
      if (packed == LOOM_VALUE_ID_INVALID) {
        packed = shifted;
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32, packed,
          shifted, lane_type, &packed));
    }
    registers[register_index] = packed;
  }

  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             registers, plan->register_count);
}

static iree_status_t loom_amdgpu_lower_vector_from_register_elements(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_from_elements_plan_t* plan) {
  loom_value_id_t elements[LOOM_AMDGPU_MAX_PACKED_16BIT_FLOAT_LANES];
  for (uint32_t i = 0; i < plan->element_count; ++i) {
    bool reused = false;
    for (uint32_t j = 0; j < i; ++j) {
      if (plan->payload.elements[j] == plan->payload.elements[i]) {
        elements[i] = elements[j];
        reused = true;
        break;
      }
    }
    if (reused) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
        context, plan->payload.elements[i], &elements[i]));
    if (plan->storage_kind == LOOM_AMDGPU_VECTOR_STORAGE_KIND_I1_MASK) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32_registers(
        context, source_op, elements[i], &elements[i]));
  }

  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             elements, plan->element_count);
}

static iree_status_t loom_amdgpu_lower_vector_from_elements(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_from_elements_plan_t* plan) {
  switch (plan->storage_kind) {
    case LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_32BIT:
    case LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_64BIT:
    case LOOM_AMDGPU_VECTOR_STORAGE_KIND_I1_MASK:
      return loom_amdgpu_lower_vector_from_register_elements(context, source_op,
                                                             plan);
    case LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_16BIT_FLOAT:
      return loom_amdgpu_lower_vector_from_16bit_elements(context, source_op,
                                                          plan);
    case LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_8BIT_FLOAT:
    case LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_INTEGER:
      return loom_amdgpu_lower_vector_from_packed_integer_elements(
          context, source_op, plan);
    case LOOM_AMDGPU_VECTOR_STORAGE_KIND_NONE:
    default:
      IREE_ASSERT_UNREACHABLE("unsupported vector element plan");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t loom_amdgpu_lookup_vector_insert_value(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_insert_plan_t* plan, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_type_t lane_type = loom_type_none();
  switch (plan->element_type) {
    case LOOM_SCALAR_TYPE_I32:
      return loom_amdgpu_lookup_or_materialize_vgpr_i32(context, source_op,
                                                        plan->value, out_value);
    case LOOM_SCALAR_TYPE_F32: {
      IREE_RETURN_IF_ERROR(
          loom_low_lower_lookup_value(context, plan->value, out_value));
      return loom_amdgpu_materialize_low_vgpr_b32(context, source_op,
                                                  *out_value, out_value);
    }
    case LOOM_SCALAR_TYPE_F16:
    case LOOM_SCALAR_TYPE_BF16: {
      IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
      return loom_amdgpu_lookup_or_materialize_vgpr_16bit_float(
          context, source_op, plan->value, lane_type, out_value);
    }
    case LOOM_SCALAR_TYPE_I8:
    case LOOM_SCALAR_TYPE_I16: {
      IREE_RETURN_IF_ERROR(
          loom_low_lower_lookup_value(context, plan->value, out_value));
      return loom_amdgpu_materialize_low_vgpr_b32(context, source_op,
                                                  *out_value, out_value);
    }
    default:
      IREE_ASSERT_UNREACHABLE("unsupported vector insert element plan");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t loom_amdgpu_select_dynamic_insert_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t old_lane, loom_value_id_t new_lane,
    loom_value_id_t index_lane, uint32_t lane_ordinal, loom_type_t lane_type,
    loom_type_t mask_lane_type, loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;

  loom_value_id_t ordinal = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, lane_ordinal,
      lane_type, &ordinal));

  const loom_value_id_t compare_operands[] = {
      index_lane,
      ordinal,
  };
  loom_op_t* compare_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
      compare_operands, IREE_ARRAYSIZE(compare_operands),
      loom_make_named_attr_slice(NULL, 0), &mask_lane_type, 1, &compare_op));

  const loom_value_id_t select_operands[] = {
      old_lane,
      new_lane,
      loom_value_slice_get(loom_low_op_results(compare_op), 0),
  };
  loom_op_t* select_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32,
      select_operands, IREE_ARRAYSIZE(select_operands),
      loom_make_named_attr_slice(NULL, 0), &lane_type, 1, &select_op));
  *out_lane = loom_value_slice_get(loom_low_op_results(select_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_replace_packed_vector_register_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t old_register, loom_value_id_t low_value,
    uint32_t lane_ordinal, uint32_t lane_bit_count, loom_type_t register_type,
    loom_value_id_t* out_register) {
  *out_register = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT(lane_bit_count == 8 || lane_bit_count == 16);
  const uint32_t lanes_per_register = 32u / lane_bit_count;
  const uint32_t register_lane = lane_ordinal % lanes_per_register;
  const uint32_t lane_bit_offset = register_lane * lane_bit_count;
  const uint32_t lane_mask = loom_amdgpu_integer_bit_mask(lane_bit_count)
                             << lane_bit_offset;
  const uint32_t preserved_mask = ~lane_mask;

  loom_value_id_t preserved = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      old_register, preserved_mask, register_type, &preserved));

  loom_value_id_t inserted = low_value;
  if (lane_bit_offset != 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
        lane_bit_offset, inserted, register_type, &inserted));
  }
  return loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32, preserved,
      inserted, register_type, out_register);
}

static iree_status_t loom_amdgpu_lower_packed_vector_insert(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_insert_plan_t* plan, loom_value_id_t low_value) {
  IREE_ASSERT(plan->lane_bit_count == 8 || plan->lane_bit_count == 16);
  loom_value_id_t low_dest = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->dest, &low_dest));
  const loom_module_t* module = loom_low_lower_context_module(context);
  loom_type_t register_type =
      loom_amdgpu_low_register_lane_type(module, low_dest);
  if (loom_type_kind(register_type) == LOOM_TYPE_NONE) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &register_type));
  }

  loom_value_id_t index_lane = LOOM_VALUE_ID_INVALID;
  loom_type_t mask_lane_type = loom_type_none();
  if (plan->is_dynamic) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i32(
        context, source_op, plan->dynamic_index, &index_lane));
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_sgpr_range_type(context, 2, &mask_lane_type));
  }

  const bool mask_low_value = plan->element_type == LOOM_SCALAR_TYPE_I8 ||
                              plan->element_type == LOOM_SCALAR_TYPE_I16;
  if (mask_low_value) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, low_value,
        loom_amdgpu_integer_bit_mask(plan->lane_bit_count), register_type,
        &low_value));
  }

  loom_value_id_t registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  const uint32_t lanes_per_register = 32u / plan->lane_bit_count;
  for (uint32_t register_index = 0; register_index < plan->register_count;
       ++register_index) {
    loom_value_id_t old_register = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
        context, source_op, low_dest, plan->register_count, register_index,
        register_type, &old_register));

    loom_value_id_t selected_register = old_register;
    const uint32_t first_lane = register_index * lanes_per_register;
    const uint32_t end_lane =
        iree_min(first_lane + lanes_per_register, plan->lane_count);
    for (uint32_t lane_ordinal = first_lane; lane_ordinal < end_lane;
         ++lane_ordinal) {
      if (!plan->is_dynamic && lane_ordinal != plan->lane_offset) {
        continue;
      }

      loom_value_id_t replacement_register = LOOM_VALUE_ID_INVALID;
      if (plan->value_kind ==
          LOOM_AMDGPU_VECTOR_INSERT_VALUE_KIND_FMA_MIX_HALF_RESULT) {
        IREE_ASSERT(!plan->is_dynamic);
        IREE_ASSERT_EQ(plan->lane_bit_count, 16);
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_tied_fma_mix_packet(
            context, source_op, &plan->fma_mix, old_register, register_type,
            &replacement_register));
      } else {
        IREE_RETURN_IF_ERROR(loom_amdgpu_replace_packed_vector_register_lane(
            context, source_op, old_register, low_value, lane_ordinal,
            plan->lane_bit_count, register_type, &replacement_register));
      }
      if (!plan->is_dynamic) {
        selected_register = replacement_register;
        break;
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_select_dynamic_insert_lane(
          context, source_op, selected_register, replacement_register,
          index_lane, lane_ordinal, register_type, mask_lane_type,
          &selected_register));
    }
    registers[register_index] = selected_register;
  }

  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             registers, plan->register_count);
}

static iree_status_t loom_amdgpu_lower_vector_insert(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_insert_plan_t* plan) {
  if (plan->value_kind ==
      LOOM_AMDGPU_VECTOR_INSERT_VALUE_KIND_FMA_MIX_HALF_RESULT) {
    IREE_ASSERT(plan->lane_bit_count == 16);
    return loom_amdgpu_lower_packed_vector_insert(context, source_op, plan,
                                                  LOOM_VALUE_ID_INVALID);
  }

  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_vector_insert_value(
      context, source_op, plan, &low_value));
  if (plan->lane_count == 1) {
    if (plan->element_type == LOOM_SCALAR_TYPE_I8 ||
        plan->element_type == LOOM_SCALAR_TYPE_I16) {
      loom_type_t lane_type = loom_type_none();
      IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
          low_value, loom_amdgpu_integer_bit_mask(plan->lane_bit_count),
          lane_type, &low_value));
    }
    return loom_low_lower_bind_value(context, plan->result, low_value);
  }
  if (plan->lane_bit_count == 8 || plan->lane_bit_count == 16) {
    return loom_amdgpu_lower_packed_vector_insert(context, source_op, plan,
                                                  low_value);
  }

  loom_value_id_t low_dest = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->dest, &low_dest));
  const loom_module_t* module = loom_low_lower_context_module(context);
  loom_type_t lane_type = loom_amdgpu_low_register_lane_type(module, low_dest);
  if (loom_type_kind(lane_type) == LOOM_TYPE_NONE) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  }

  loom_value_id_t index_lane = LOOM_VALUE_ID_INVALID;
  loom_type_t mask_lane_type = loom_type_none();
  if (plan->is_dynamic) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i32(
        context, source_op, plan->dynamic_index, &index_lane));
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_sgpr_range_type(context, 2, &mask_lane_type));
  }

  loom_value_id_t lanes[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < plan->lane_count; ++i) {
    loom_value_id_t old_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
        context, source_op, low_dest, plan->register_count, i, lane_type,
        &old_lane));
    if (!plan->is_dynamic) {
      lanes[i] = i == plan->lane_offset ? low_value : old_lane;
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_select_dynamic_insert_lane(
        context, source_op, old_lane, low_value, index_lane, i, lane_type,
        mask_lane_type, &lanes[i]));
  }

  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             lanes, plan->lane_count);
}

static iree_status_t loom_amdgpu_extract_bf16_register_lane_as_f32_bits(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_register, uint32_t register_lane,
    loom_type_t result_lane_type, loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  if (register_lane == 0) {
    return loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, 16,
        source_register, result_lane_type, out_lane);
  }
  return loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      source_register, UINT32_C(0xFFFF0000), result_lane_type, out_lane);
}

static iree_status_t loom_amdgpu_extract_bf16_range_lane_as_f32_bits(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint32_t source_register_count,
    uint32_t lane_index, loom_type_t source_lane_type,
    loom_type_t result_lane_type, loom_value_id_t* out_lane) {
  const uint32_t register_index = lane_index / 2u;
  loom_value_id_t source_register = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
      context, source_op, low_source, source_register_count, register_index,
      source_lane_type, &source_register));
  return loom_amdgpu_extract_bf16_register_lane_as_f32_bits(
      context, source_op, source_register, lane_index % 2u, result_lane_type,
      out_lane);
}

static iree_status_t loom_amdgpu_extract_f16_lane_as_low_bits(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint32_t source_register_count,
    uint32_t lane_index, loom_type_t lane_type, loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  const uint32_t register_index = lane_index / 2u;
  const uint32_t register_bit_offset = (lane_index % 2u) * 16u;
  loom_value_id_t source_register = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
      context, source_op, low_source, source_register_count, register_index,
      lane_type, &source_register));
  if (register_bit_offset == 0) {
    *out_lane = source_register;
    return iree_ok_status();
  }
  return loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
      register_bit_offset, source_register, lane_type, out_lane);
}

static void loom_amdgpu_require_fma_mix_plan_sources_storage(
    loom_low_lower_context_t* context, const loom_amdgpu_fma_mix_plan_t* plan) {
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(plan->sources); ++i) {
    if (plan->sources[i] == LOOM_VALUE_ID_INVALID) {
      continue;
    }
    loom_low_lower_require_source_value_storage(context, plan->sources[i]);
  }
}

void loom_amdgpu_mark_structural_value_plan_storage_demands(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t plan) {
  switch (plan.id) {
    case LOOM_OP_INDEX_CAST: {
      const loom_amdgpu_index_cast_plan_t* index_cast_plan =
          (const loom_amdgpu_index_cast_plan_t*)plan.target_data;
      switch (index_cast_plan->kind) {
        case LOOM_AMDGPU_INDEX_CAST_KIND_ALIAS:
        case LOOM_AMDGPU_INDEX_CAST_KIND_PRESERVING_LOW_32:
          loom_low_lower_require_source_value_storage(context,
                                                      index_cast_plan->source);
          return;
        case LOOM_AMDGPU_INDEX_CAST_KIND_DIAGNOSTIC_REJECTED:
          return;
        case LOOM_AMDGPU_INDEX_CAST_KIND_NONE:
          break;
      }
      IREE_ASSERT_UNREACHABLE("unknown AMDGPU index cast plan kind");
      return;
    }
    case LOOM_OP_VECTOR_EXTRACT: {
      const loom_amdgpu_vector_extract_plan_t* extract_plan =
          (const loom_amdgpu_vector_extract_plan_t*)plan.target_data;
      loom_low_lower_require_source_value_storage(context,
                                                  extract_plan->source);
      if (extract_plan->is_dynamic) {
        loom_low_lower_require_source_value_storage(
            context, extract_plan->dynamic_index);
      }
      return;
    }
    case LOOM_OP_VECTOR_EXTF:
    case LOOM_OP_VECTOR_DECODE:
    case LOOM_OP_VECTOR_FPTRUNC: {
      const loom_amdgpu_vector_16bit_float_conversion_plan_t* conversion_plan =
          (const loom_amdgpu_vector_16bit_float_conversion_plan_t*)
              plan.target_data;
      loom_low_lower_require_source_value_storage(
          context, conversion_plan->storage_source);
      if (conversion_plan->scale_source != LOOM_VALUE_ID_INVALID) {
        loom_low_lower_require_source_value_storage(
            context, conversion_plan->scale_source);
      }
      return;
    }
    case LOOM_OP_VECTOR_IOTA: {
      if (plan.target_data == NULL) {
        return;
      }
      const loom_amdgpu_vector_iota_plan_t* iota_plan =
          (const loom_amdgpu_vector_iota_plan_t*)plan.target_data;
      if (!loom_amdgpu_vector_iota_plan_is_dynamic(iota_plan)) {
        return;
      }
      if (!loom_amdgpu_vector_iota_plan_has_exact_base(iota_plan)) {
        loom_low_lower_require_source_value_storage(context, iota_plan->base);
      }
      if (!loom_amdgpu_vector_iota_plan_has_exact_step(iota_plan)) {
        loom_low_lower_require_source_value_storage(context, iota_plan->step);
      }
      return;
    }
    case LOOM_OP_VECTOR_FROM_ELEMENTS:
    case LOOM_OP_VECTOR_SPLAT: {
      if (plan.target_data == NULL) {
        return;
      }
      const loom_amdgpu_vector_from_elements_plan_t* vector_plan =
          (const loom_amdgpu_vector_from_elements_plan_t*)plan.target_data;
      if (vector_plan->materialization_kind ==
          LOOM_AMDGPU_VECTOR_FROM_ELEMENTS_MATERIALIZATION_EXACT_PACKED_INTEGER) {
        return;
      }
      if (vector_plan->fma_mix_half_result_lane_mask != 0) {
        for (uint32_t i = 0; i < vector_plan->element_count; ++i) {
          if (loom_amdgpu_vector_from_elements_uses_fma_mix_half_result(
                  vector_plan, i)) {
            loom_amdgpu_require_fma_mix_plan_sources_storage(
                context, &vector_plan->fma_mix_half_results[i]);
          } else {
            loom_low_lower_require_source_value_storage(
                context, vector_plan->payload.elements[i]);
          }
        }
        return;
      }
      loom_low_lower_require_source_operands_storage(context, source_op);
      return;
    }
    case LOOM_OP_VECTOR_INSERT: {
      const loom_amdgpu_vector_insert_plan_t* insert_plan =
          (const loom_amdgpu_vector_insert_plan_t*)plan.target_data;
      if (insert_plan->value_kind !=
          LOOM_AMDGPU_VECTOR_INSERT_VALUE_KIND_FMA_MIX_HALF_RESULT) {
        loom_low_lower_require_source_operands_storage(context, source_op);
        return;
      }
      loom_low_lower_require_source_value_storage(context, insert_plan->dest);
      loom_amdgpu_require_fma_mix_plan_sources_storage(context,
                                                       &insert_plan->fma_mix);
      return;
    }
    default:
      loom_low_lower_require_source_operands_storage(context, source_op);
      return;
  }
}

static loom_amdgpu_vector_extract_plan_t loom_amdgpu_vector_fp8_extract_plan(
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan) {
  return (loom_amdgpu_vector_extract_plan_t){
      .source = plan->storage_source,
      .result = plan->result,
      .lane_count = plan->lane_count,
      .register_count = plan->storage_register_count,
      .result_register_count = 1,
      .element_register_count = 1,
      .lane_bit_count = 8,
      .sign_extend_packed_lane = false,
  };
}

static iree_status_t loom_amdgpu_extract_vector_fp8_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    const loom_amdgpu_vector_extract_plan_t* extract_plan,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    uint32_t lane_index, loom_value_id_t* out_low_byte) {
  const uint64_t storage_lane =
      (uint64_t)plan->storage_lane_offset +
      (uint64_t)lane_index * (uint64_t)plan->storage_lane_stride;
  IREE_ASSERT_LE(storage_lane, UINT32_MAX);
  return loom_amdgpu_extract_packed_register_lane(
      context, source_op, low_source, extract_plan, (uint32_t)storage_lane,
      LOOM_AMDGPU_BITFIELD_EXTRACT_MODE_RAW_SHIFTED, source_lane_type,
      out_low_byte);
}

typedef struct loom_amdgpu_vector_fp8_decode_value_flag_cache_t {
  // Per-result-lane FP8 decode simplification flags.
  loom_amdgpu_fp8_decode_value_flags_t
      lane_flags[LOOM_AMDGPU_MAX_PACKED_16BIT_FLOAT_LANES];
} loom_amdgpu_vector_fp8_decode_value_flag_cache_t;

static void loom_amdgpu_vector_fp8_decode_value_flag_cache_initialize(
    loom_low_lower_context_t* context,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    loom_amdgpu_vector_fp8_decode_value_flag_cache_t* out_cache) {
  *out_cache = (loom_amdgpu_vector_fp8_decode_value_flag_cache_t){0};
  IREE_ASSERT_LE(plan->lane_count, IREE_ARRAYSIZE(out_cache->lane_flags));
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  if (fact_table == NULL) {
    return;
  }

  const loom_value_facts_t content_facts =
      loom_value_fact_table_lookup(fact_table, plan->content_fact_source);
  loom_value_facts_t all_equal_facts = {0};
  const bool has_all_equal_facts = loom_value_facts_query_all_equal_element(
      &fact_table->context, content_facts, &all_equal_facts);

  loom_value_fact_small_static_lanes_t small_lanes = {0};
  const bool has_small_lanes = loom_value_facts_query_small_static_lanes(
      &fact_table->context, content_facts, &small_lanes);
  for (uint32_t lane_index = 0; lane_index < plan->lane_count; ++lane_index) {
    const uint64_t storage_lane =
        (uint64_t)plan->storage_lane_offset +
        (uint64_t)lane_index * (uint64_t)plan->storage_lane_stride;
    loom_value_facts_t lane_facts = {0};
    if (has_small_lanes && storage_lane < small_lanes.count) {
      lane_facts = small_lanes.lanes[storage_lane];
    } else if (has_all_equal_facts) {
      lane_facts = all_equal_facts;
    } else {
      continue;
    }
    out_cache->lane_flags[lane_index] =
        loom_amdgpu_fp8_decode_value_flags_from_facts(lane_facts);
  }
}

static loom_amdgpu_fp8_decode_value_flags_t
loom_amdgpu_vector_fp8_decode_value_flags(
    const loom_amdgpu_vector_fp8_decode_value_flag_cache_t* cache,
    uint32_t lane_index) {
  IREE_ASSERT_LT(lane_index, IREE_ARRAYSIZE(cache->lane_flags));
  return cache->lane_flags[lane_index];
}

static iree_status_t loom_amdgpu_ensure_fp8_software_decode(
    loom_low_lower_context_t* context, loom_scalar_type_t source_element_type,
    loom_type_t* sgpr_type, loom_type_t* mask_type,
    const loom_amdgpu_fp8_decode_plan_t** decode_plan) {
  if (loom_type_kind(*sgpr_type) == LOOM_TYPE_NONE) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, sgpr_type));
  }
  if (loom_type_kind(*mask_type) != LOOM_TYPE_NONE && *decode_plan != NULL) {
    return iree_ok_status();
  }
  if (loom_type_kind(*mask_type) == LOOM_TYPE_NONE) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_sgpr_range_type(context, 2, mask_type));
  }
  if (*decode_plan == NULL) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_get_fp8_decode_plan(
        context, source_element_type, decode_plan));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_lower_vector_fp8_to_f32_software_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    const loom_amdgpu_vector_extract_plan_t* extract_plan,
    const loom_amdgpu_vector_fp8_decode_value_flag_cache_t* value_flag_cache,
    const loom_amdgpu_fp8_decode_plan_t** decode_plan,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t result_lane_type, loom_type_t* sgpr_type,
    loom_type_t* mask_type, uint32_t lane_index,
    loom_value_id_t* out_low_lane) {
  *out_low_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_ensure_fp8_software_decode(
      context, plan->source_element_type, sgpr_type, mask_type, decode_plan));

  loom_value_id_t low_byte = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_vector_fp8_lane(
      context, source_op, plan, extract_plan, low_source, source_lane_type,
      lane_index, &low_byte));
  const loom_amdgpu_fp8_decode_value_flags_t value_flags =
      loom_amdgpu_vector_fp8_decode_value_flags(value_flag_cache, lane_index);
  IREE_RETURN_IF_ERROR(loom_amdgpu_try_emit_fp8_not_subnormal_to_f32_lane(
      context, source_op, *decode_plan, low_byte, value_flags, result_lane_type,
      *mask_type, out_low_lane));
  if (*out_low_lane != LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }

  loom_value_id_t bf16_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_to_bf16_lane(
      context, source_op, *decode_plan, low_byte, value_flags, result_lane_type,
      *sgpr_type, *mask_type, &bf16_lane));
  return loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, 16,
      bf16_lane, result_lane_type, out_low_lane);
}

static iree_status_t loom_amdgpu_emit_fp8_lane_to_f32_native(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t low_byte, loom_type_t result_lane_type,
    loom_value_id_t* out_low_lane) {
  *out_low_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_full_low_vgpr_b32(
      context, source_op, low_byte, &low_byte));
  return loom_amdgpu_emit_resolved_vgpr_unary(
      context, source_op, descriptor, low_byte, result_lane_type, out_low_lane);
}

static iree_status_t loom_amdgpu_lower_vector_fp8_to_f32_native_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    const loom_amdgpu_vector_extract_plan_t* extract_plan,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t result_lane_type, uint32_t lane_index,
    loom_value_id_t* out_low_lane) {
  loom_value_id_t low_byte = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_vector_fp8_lane(
      context, source_op, plan, extract_plan, low_source, source_lane_type,
      lane_index, &low_byte));
  return loom_amdgpu_emit_fp8_lane_to_f32_native(
      context, source_op, descriptor, low_byte, result_lane_type, out_low_lane);
}

typedef struct loom_amdgpu_vector_fp8_pair_storage_t {
  // Source register containing the selected adjacent FP8 byte pair.
  uint32_t source_register_index;
  // First FP8 byte offset within the source register.
  uint32_t byte_offset;
  // Number of live logical lanes consumed from this pair.
  uint32_t live_lane_count;
} loom_amdgpu_vector_fp8_pair_storage_t;

static bool loom_amdgpu_vector_fp8_query_storage_pair(
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    uint32_t lane_index,
    loom_amdgpu_vector_fp8_pair_storage_t* out_pair_storage) {
  *out_pair_storage = (loom_amdgpu_vector_fp8_pair_storage_t){0};
  if (lane_index >= plan->lane_count) {
    return false;
  }

  const uint64_t storage_lane =
      (uint64_t)plan->storage_lane_offset +
      (uint64_t)lane_index * (uint64_t)plan->storage_lane_stride;
  const uint64_t next_storage_lane =
      (uint64_t)plan->storage_lane_offset +
      (uint64_t)(lane_index + 1u) * (uint64_t)plan->storage_lane_stride;
  if (next_storage_lane != storage_lane + 1u) {
    return false;
  }
  if (next_storage_lane >= plan->storage_lane_count) {
    return false;
  }

  const uint32_t register_index = (uint32_t)(storage_lane / 4u);
  const uint32_t byte_offset = (uint32_t)(storage_lane % 4u);
  if (byte_offset >= 3u || next_storage_lane / 4u != register_index) {
    return false;
  }
  IREE_ASSERT_LT(register_index, plan->storage_register_count);
  const uint32_t remaining_lane_count = plan->lane_count - lane_index;
  const uint32_t live_lane_count =
      remaining_lane_count < 2u ? remaining_lane_count : 2u;
  *out_pair_storage = (loom_amdgpu_vector_fp8_pair_storage_t){
      .source_register_index = register_index,
      .byte_offset = byte_offset,
      .live_lane_count = live_lane_count,
  };
  return true;
}

typedef struct loom_amdgpu_vector_fp8_octet_storage_t {
  // First source register containing the selected adjacent FP8 byte octet.
  uint32_t source_register_index;
} loom_amdgpu_vector_fp8_octet_storage_t;

static bool loom_amdgpu_vector_fp8_query_storage_octet(
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    uint32_t lane_index,
    loom_amdgpu_vector_fp8_octet_storage_t* out_octet_storage) {
  *out_octet_storage = (loom_amdgpu_vector_fp8_octet_storage_t){0};
  if (lane_index + 7u >= plan->lane_count || plan->storage_lane_stride != 1u) {
    return false;
  }

  const uint64_t storage_lane =
      (uint64_t)plan->storage_lane_offset + (uint64_t)lane_index;
  if ((storage_lane & 3u) != 0 ||
      storage_lane + 7u >= plan->storage_lane_count) {
    return false;
  }

  const uint64_t source_register_index = storage_lane / 4u;
  if (source_register_index + 1u >= plan->storage_register_count ||
      source_register_index > UINT32_MAX) {
    return false;
  }
  *out_octet_storage = (loom_amdgpu_vector_fp8_octet_storage_t){
      .source_register_index = (uint32_t)source_register_index,
  };
  return true;
}

static iree_status_t loom_amdgpu_extract_vector_fp8_pair_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    const loom_amdgpu_vector_fp8_pair_storage_t* pair_storage,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_value_id_t* out_source_register) {
  *out_source_register = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
      context, source_op, low_source, plan->storage_register_count,
      pair_storage->source_register_index, source_lane_type,
      out_source_register));
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_full_low_vgpr_b32(
      context, source_op, *out_source_register, out_source_register));
  if (pair_storage->byte_offset == 0) {
    return iree_ok_status();
  }
  return loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
      pair_storage->byte_offset * 8u, *out_source_register, source_lane_type,
      out_source_register);
}

static iree_status_t loom_amdgpu_extract_vector_fp8_octet_registers(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    const loom_amdgpu_vector_fp8_octet_storage_t* octet_storage,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t source_pair_type, loom_value_id_t* out_source_pair) {
  loom_value_id_t source_registers[2] = {LOOM_VALUE_ID_INVALID,
                                         LOOM_VALUE_ID_INVALID};
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(source_registers); ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
        context, source_op, low_source, plan->storage_register_count,
        octet_storage->source_register_index + i, source_lane_type,
        &source_registers[i]));
    IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_full_low_vgpr_b32(
        context, source_op, source_registers[i], &source_registers[i]));
  }

  loom_op_t* source_pair_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), source_registers,
      IREE_ARRAYSIZE(source_registers), source_pair_type, source_op->location,
      &source_pair_op));
  *out_source_pair =
      loom_value_slice_get(loom_low_op_results(source_pair_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_vector_fp8_pair_descriptor(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    const loom_amdgpu_vector_fp8_pair_storage_t* pair_storage,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    const loom_value_id_t* extra_operands, uint32_t extra_operand_count,
    loom_type_t result_type, loom_value_id_t* out_converted_pair) {
  IREE_ASSERT_LE(extra_operand_count, 1u);
  *out_converted_pair = LOOM_VALUE_ID_INVALID;

  loom_value_id_t source_register = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_vector_fp8_pair_register(
      context, source_op, plan, pair_storage, low_source, source_lane_type,
      &source_register));

  loom_value_id_t operands[2] = {source_register, LOOM_VALUE_ID_INVALID};
  for (uint32_t i = 0; i < extra_operand_count; ++i) {
    operands[i + 1u] = extra_operands[i];
  }
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, 1u + extra_operand_count,
      loom_named_attr_slice_empty(), &result_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_converted_pair = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_try_emit_vector_fp8_pair_descriptor(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    const loom_value_id_t* extra_operands, uint32_t extra_operand_count,
    loom_type_t result_type, uint32_t lane_index,
    loom_amdgpu_vector_fp8_pair_storage_t* out_pair_storage,
    loom_value_id_t* out_converted_pair) {
  *out_converted_pair = LOOM_VALUE_ID_INVALID;
  if (!loom_amdgpu_vector_fp8_query_storage_pair(plan, lane_index,
                                                 out_pair_storage)) {
    return iree_ok_status();
  }
  return loom_amdgpu_emit_vector_fp8_pair_descriptor(
      context, source_op, plan, out_pair_storage, descriptor, low_source,
      source_lane_type, extra_operands, extra_operand_count, result_type,
      out_converted_pair);
}

static iree_status_t loom_amdgpu_extract_vector_fp8_native_pair_lanes(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t converted_pair, loom_type_t result_lane_type,
    uint32_t low_lane_count, loom_value_id_t* out_low_lanes) {
  for (uint32_t i = 0; i < low_lane_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
        context, source_op, converted_pair, /*register_count=*/2,
        /*register_offset=*/i, result_lane_type, &out_low_lanes[i]));
  }
  return iree_ok_status();
}

static loom_amdgpu_fp8_decode_value_flags_t
loom_amdgpu_vector_fp8_pair_decode_value_flags(
    const loom_amdgpu_vector_fp8_decode_value_flag_cache_t* value_flag_cache,
    uint32_t lane_index, uint32_t live_lane_count) {
  IREE_ASSERT_GE(live_lane_count, 1u);
  IREE_ASSERT_LE(live_lane_count, 2u);
  loom_amdgpu_fp8_decode_value_flags_t value_flags =
      loom_amdgpu_vector_fp8_decode_value_flags(value_flag_cache, lane_index);
  if (live_lane_count == 2u) {
    value_flags &= loom_amdgpu_vector_fp8_decode_value_flags(value_flag_cache,
                                                             lane_index + 1u);
  }
  return value_flags;
}

static bool loom_amdgpu_vector_fp8_query_storage_pair_set(
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    const loom_amdgpu_vector_fp8_decode_value_flag_cache_t* value_flag_cache,
    uint32_t pair_count, loom_amdgpu_vector_fp8_pair_storage_t* pair_storage,
    loom_amdgpu_fp8_decode_value_flags_t* out_value_flags) {
  if (pair_count == 0 || pair_count > LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS ||
      plan->storage_register_count > LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS ||
      plan->lane_count == 0 || plan->lane_count > pair_count * 2u ||
      plan->lane_count <= (pair_count - 1u) * 2u) {
    return false;
  }

  *out_value_flags = LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NONE;
  bool have_value_flags = false;
  for (uint32_t register_index = 0; register_index < pair_count;
       ++register_index) {
    const uint32_t lane_base = register_index * 2u;
    if (!loom_amdgpu_vector_fp8_query_storage_pair(
            plan, lane_base, &pair_storage[register_index])) {
      return false;
    }
    const loom_amdgpu_fp8_decode_value_flags_t pair_value_flags =
        loom_amdgpu_vector_fp8_pair_decode_value_flags(
            value_flag_cache, lane_base,
            pair_storage[register_index].live_lane_count);
    *out_value_flags = have_value_flags ? *out_value_flags & pair_value_flags
                                        : pair_value_flags;
    have_value_flags = true;
  }
  return true;
}

static iree_status_t loom_amdgpu_materialize_vector_fp8_pair_sources(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    const loom_amdgpu_vector_fp8_pair_storage_t* pair_storage,
    uint32_t pair_count, loom_value_id_t low_source,
    loom_type_t source_lane_type,
    loom_amdgpu_fp8_packed_u16_pair_source_t* out_pair_sources) {
  loom_value_id_t source_registers[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS];
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(source_registers); ++i) {
    source_registers[i] = LOOM_VALUE_ID_INVALID;
  }
  for (uint32_t register_index = 0; register_index < pair_count;
       ++register_index) {
    const uint32_t source_register_index =
        pair_storage[register_index].source_register_index;
    if (source_registers[source_register_index] == LOOM_VALUE_ID_INVALID) {
      loom_value_id_t source_register = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
          context, source_op, low_source, plan->storage_register_count,
          source_register_index, source_lane_type, &source_register));
      IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_full_low_vgpr_b32(
          context, source_op, source_register, &source_register));
      source_registers[source_register_index] = source_register;
    }
  }

  for (uint32_t register_index = 0; register_index < pair_count;
       ++register_index) {
    out_pair_sources[register_index] =
        (loom_amdgpu_fp8_packed_u16_pair_source_t){
            .source_register = source_registers[pair_storage[register_index]
                                                    .source_register_index],
            .byte_offset = pair_storage[register_index].byte_offset,
            .live_lane_count = pair_storage[register_index].live_lane_count,
        };
  }
  return iree_ok_status();
}

static iree_status_t
loom_amdgpu_try_lower_vector_fp8_storage_pair_descriptor_to_f32_native(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t low_source, const loom_value_id_t* extra_operands,
    uint32_t extra_operand_count, loom_type_t source_lane_type,
    loom_type_t result_lane_type, loom_type_t result_pair_type,
    uint32_t lane_index, loom_value_id_t* out_low_lanes,
    uint32_t* out_low_lane_count) {
  out_low_lanes[0] = LOOM_VALUE_ID_INVALID;
  out_low_lanes[1] = LOOM_VALUE_ID_INVALID;
  *out_low_lane_count = 0;

  loom_amdgpu_vector_fp8_pair_storage_t pair_storage;
  loom_value_id_t converted_pair = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_try_emit_vector_fp8_pair_descriptor(
      context, source_op, plan, descriptor, low_source, source_lane_type,
      extra_operands, extra_operand_count, result_pair_type, lane_index,
      &pair_storage, &converted_pair));
  if (converted_pair == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  const uint32_t low_lane_count = pair_storage.live_lane_count;
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_vector_fp8_native_pair_lanes(
      context, source_op, converted_pair, result_lane_type, low_lane_count,
      out_low_lanes));
  *out_low_lane_count = low_lane_count;
  return iree_ok_status();
}

static bool loom_amdgpu_vector_16bit_float_conversion_plan_has_scale(
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan) {
  return plan->scale_source != LOOM_VALUE_ID_INVALID;
}

static bool loom_amdgpu_vector_16bit_float_conversion_plan_has_f32_scale(
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan) {
  return loom_amdgpu_vector_16bit_float_conversion_plan_has_scale(plan) &&
         plan->scale_format == LOOM_VALUE_FACT_NUMERIC_FORMAT_F32;
}

static bool loom_amdgpu_vector_16bit_float_conversion_plan_has_e8m0_scale(
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan) {
  return loom_amdgpu_vector_16bit_float_conversion_plan_has_scale(plan) &&
         plan->scale_format == LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E8M0;
}

static bool loom_amdgpu_vector_fp8_scalef32_is_identity(
    loom_low_lower_context_t* context,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan) {
  if (!loom_amdgpu_vector_16bit_float_conversion_plan_has_f32_scale(plan)) {
    return false;
  }
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  uint32_t scale_bit_pattern = 0;
  const loom_type_t scale_type =
      loom_module_value_type(module, plan->scale_source);
  if (loom_amdgpu_type_is_f32(scale_type)) {
    return loom_amdgpu_value_as_f32_bit_pattern(
               module, fact_table, plan->scale_source, &scale_bit_pattern) &&
           scale_bit_pattern == LOOM_AMDGPU_FP8_F32_IDENTITY_SCALE_BITS;
  }
  return loom_amdgpu_type_is_vector1_element(scale_type,
                                             LOOM_SCALAR_TYPE_F32) &&
         loom_amdgpu_source_lane_as_u32_bits(
             fact_table, module, plan->scale_source, 0, &scale_bit_pattern) &&
         scale_bit_pattern == LOOM_AMDGPU_FP8_F32_IDENTITY_SCALE_BITS;
}

static iree_status_t loom_amdgpu_lookup_vector_scale_source(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    loom_value_id_t* out_low_scale) {
  IREE_ASSERT(loom_amdgpu_vector_16bit_float_conversion_plan_has_scale(plan));
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->scale_source, out_low_scale));
  return loom_amdgpu_materialize_full_low_vgpr_b32(
      context, source_op, *out_low_scale, out_low_scale);
}

static void loom_amdgpu_vector_fp8_unscaled_plan(
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    loom_amdgpu_vector_16bit_float_conversion_plan_t* out_plan) {
  *out_plan = *plan;
  out_plan->scale_source = LOOM_VALUE_ID_INVALID;
  out_plan->scale_format = LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE;
  out_plan->scale_group_element_count = 0;
}

static uint32_t loom_amdgpu_vector_fp8_e8m0_pk8_result_register_count(
    loom_scalar_type_t result_element_type) {
  return result_element_type == LOOM_SCALAR_TYPE_F32 ? 8u : 4u;
}

static uint32_t loom_amdgpu_vector_fp8_e8m0_pk8_scale_sel(
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    uint32_t lane_index) {
  if (!loom_amdgpu_vector_16bit_float_conversion_plan_has_e8m0_scale(plan) ||
      plan->scale_group_element_count == 0 ||
      plan->scale_group_element_count >= plan->lane_count) {
    return 0;
  }
  const uint32_t scale_selector = lane_index / plan->scale_group_element_count;
  IREE_ASSERT_LE(scale_selector, 15u);
  return scale_selector;
}

static iree_status_t loom_amdgpu_try_lower_vector_fp8_e8m0_pk8_native(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    loom_value_id_t low_source, loom_value_id_t low_scale,
    loom_type_t source_lane_type, loom_type_t result_lane_type,
    loom_value_id_t* out_low_registers, uint32_t* out_selected_count) {
  *out_selected_count = 0;
  IREE_ASSERT_TRUE(
      !loom_amdgpu_vector_16bit_float_conversion_plan_has_scale(plan) ||
      loom_amdgpu_vector_16bit_float_conversion_plan_has_e8m0_scale(plan));
  const bool requires_full_selection =
      loom_amdgpu_vector_16bit_float_conversion_plan_has_e8m0_scale(plan);
  const uint32_t result_registers_per_octet =
      loom_amdgpu_vector_fp8_e8m0_pk8_result_register_count(
          plan->result_element_type);
  const uint32_t result_register_count =
      plan->result_element_type == LOOM_SCALAR_TYPE_F32
          ? plan->lane_count
          : plan->result_register_count;
  for (uint32_t i = 0; i < result_register_count; ++i) {
    out_low_registers[i] = LOOM_VALUE_ID_INVALID;
  }

  const loom_low_lower_resolved_descriptor_t* descriptor = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_get_fp8_e8m0_pk8_descriptor(
      context, plan->source_element_type, plan->result_element_type,
      &descriptor));
  if (descriptor == NULL) {
    IREE_ASSERT_FALSE(requires_full_selection);
    return iree_ok_status();
  }
  if (plan->lane_count == 0 || (plan->lane_count & 7u) != 0) {
    IREE_ASSERT_FALSE(requires_full_selection);
    return iree_ok_status();
  }
  if (!requires_full_selection) {
    for (uint32_t lane_index = 0; lane_index < plan->lane_count;
         lane_index += 8u) {
      loom_amdgpu_vector_fp8_octet_storage_t octet_storage;
      if (!loom_amdgpu_vector_fp8_query_storage_octet(plan, lane_index,
                                                      &octet_storage)) {
        return iree_ok_status();
      }
    }
  }

  loom_type_t source_pair_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_vgpr_range_type(context, 2, &source_pair_type));
  loom_type_t result_octet_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_range_type(
      context, result_registers_per_octet, &result_octet_type));

  loom_value_id_t low_e8m0_scale = low_scale;
  loom_string_id_t scale_sel_name_id = LOOM_STRING_ID_INVALID;
  for (uint32_t lane_index = 0; lane_index < plan->lane_count;
       lane_index += 8u) {
    loom_amdgpu_vector_fp8_octet_storage_t octet_storage;
    IREE_ASSERT_TRUE(loom_amdgpu_vector_fp8_query_storage_octet(
        plan, lane_index, &octet_storage));
    if (low_e8m0_scale == LOOM_VALUE_ID_INVALID) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
          LOOM_AMDGPU_FP8_E8M0FNU_PACKED_IDENTITY_SCALE_BITS, result_lane_type,
          &low_e8m0_scale));
    }

    loom_value_id_t source_pair = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_vector_fp8_octet_registers(
        context, source_op, plan, &octet_storage, low_source, source_lane_type,
        source_pair_type, &source_pair));

    const loom_value_id_t operands[] = {source_pair, low_e8m0_scale};
    loom_named_attr_t attrs[1] = {0};
    iree_host_size_t attr_count = 0;
    const uint32_t scale_sel =
        loom_amdgpu_vector_fp8_e8m0_pk8_scale_sel(plan, lane_index);
    if (scale_sel != 0) {
      if (scale_sel_name_id == LOOM_STRING_ID_INVALID) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_intern(context, IREE_SV("scale_sel"),
                                                &scale_sel_name_id));
      }
      attrs[attr_count++] = (loom_named_attr_t){
          .name_id = scale_sel_name_id,
          .value = loom_attr_i64(scale_sel),
      };
    }
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
        context, descriptor, operands, IREE_ARRAYSIZE(operands),
        loom_make_named_attr_slice(attrs, attr_count), &result_octet_type, 1,
        /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
        &low_op));
    const loom_value_id_t converted_octet =
        loom_value_slice_get(loom_low_op_results(low_op), 0);
    const uint32_t result_register_base =
        plan->result_element_type == LOOM_SCALAR_TYPE_F32 ? lane_index
                                                          : lane_index / 2u;
    for (uint32_t i = 0; i < result_registers_per_octet; ++i) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
          context, source_op, converted_octet, result_registers_per_octet, i,
          result_lane_type, &out_low_registers[result_register_base + i]));
      ++(*out_selected_count);
    }
  }
  return iree_ok_status();
}

static iree_status_t
loom_amdgpu_try_lower_vector_fp8_identity_scalef32_to_packed_16bit_native(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t result_lane_type, loom_value_id_t* out_low_packets,
    uint32_t* out_selected_count) {
  *out_selected_count = 0;
  for (uint32_t i = 0; i < plan->result_register_count; ++i) {
    out_low_packets[i] = LOOM_VALUE_ID_INVALID;
  }

  const loom_low_lower_resolved_descriptor_t* descriptor = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_get_fp8_scalef32_descriptor(
      context, plan->source_element_type, plan->result_element_type,
      &descriptor));
  if (descriptor == NULL) {
    return iree_ok_status();
  }

  loom_value_id_t low_scale_one = LOOM_VALUE_ID_INVALID;
  for (uint32_t register_index = 0;
       register_index < plan->result_register_count; ++register_index) {
    const uint32_t lane_index = register_index * 2u;
    loom_amdgpu_vector_fp8_pair_storage_t pair_storage;
    if (!loom_amdgpu_vector_fp8_query_storage_pair(plan, lane_index,
                                                   &pair_storage)) {
      continue;
    }
    if (low_scale_one == LOOM_VALUE_ID_INVALID) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
          LOOM_AMDGPU_FP8_F32_IDENTITY_SCALE_BITS, result_lane_type,
          &low_scale_one));
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vector_fp8_pair_descriptor(
        context, source_op, plan, &pair_storage, descriptor, low_source,
        source_lane_type, &low_scale_one, /*extra_operand_count=*/1,
        result_lane_type, &out_low_packets[register_index]));
    if (out_low_packets[register_index] == LOOM_VALUE_ID_INVALID) {
      IREE_ASSERT_UNREACHABLE("selected pair storage was already queried");
      IREE_BUILTIN_UNREACHABLE();
    }
    ++(*out_selected_count);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_try_lower_vector_fp8_pair_to_packed_bf16(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    const loom_amdgpu_vector_fp8_decode_value_flag_cache_t* value_flag_cache,
    const loom_amdgpu_fp8_decode_plan_t** decode_plan,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t result_lane_type, loom_type_t* sgpr_type,
    loom_type_t* mask_type, uint32_t lane_index,
    loom_value_id_t* out_low_packet) {
  *out_low_packet = LOOM_VALUE_ID_INVALID;

  loom_amdgpu_vector_fp8_pair_storage_t pair_storage;
  if (!loom_amdgpu_vector_fp8_query_storage_pair(plan, lane_index,
                                                 &pair_storage)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_ensure_fp8_software_decode(
      context, plan->source_element_type, sgpr_type, mask_type, decode_plan));

  const loom_amdgpu_fp8_decode_value_flags_t value_flags =
      loom_amdgpu_vector_fp8_pair_decode_value_flags(
          value_flag_cache, lane_index, pair_storage.live_lane_count);
  if (!loom_amdgpu_can_emit_fp8_pair_to_packed_bf16(*decode_plan,
                                                    value_flags)) {
    return iree_ok_status();
  }

  loom_value_id_t source_register = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_vector_fp8_pair_register(
      context, source_op, plan, &pair_storage, low_source, source_lane_type,
      &source_register));

  const loom_amdgpu_fp8_packed_u16_pair_source_t pair_source = {
      .source_register = source_register,
      .byte_offset = pair_storage.byte_offset,
      .live_lane_count = pair_storage.live_lane_count,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_pairs_to_packed_bf16(
      context, source_op, *decode_plan, &pair_source, 1, value_flags,
      result_lane_type, *sgpr_type, *mask_type, out_low_packet));
  return iree_ok_status();
}

typedef enum loom_amdgpu_vector_fp8_packed_bf16_selection_flag_bits_e {
  LOOM_AMDGPU_VECTOR_FP8_PACKED_BF16_SELECTION_FLAG_NONE = 0u,
  LOOM_AMDGPU_VECTOR_FP8_PACKED_BF16_SELECTION_FLAG_REQUIRE_PREFERRED = 1u << 0,
} loom_amdgpu_vector_fp8_packed_bf16_selection_flag_bits_t;
typedef uint32_t loom_amdgpu_vector_fp8_packed_bf16_selection_flags_t;

static iree_status_t loom_amdgpu_try_lower_vector_fp8_pairs_to_packed_bf16(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    const loom_amdgpu_vector_fp8_decode_value_flag_cache_t* value_flag_cache,
    const loom_amdgpu_fp8_decode_plan_t** decode_plan,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t result_lane_type, loom_type_t* sgpr_type,
    loom_type_t* mask_type, uint32_t pair_count,
    loom_amdgpu_vector_fp8_packed_bf16_selection_flags_t selection_flags,
    loom_value_id_t* out_low_packets) {
  for (uint32_t i = 0; i < pair_count; ++i) {
    out_low_packets[i] = LOOM_VALUE_ID_INVALID;
  }
  loom_amdgpu_vector_fp8_pair_storage_t
      pair_storage[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {0};
  loom_amdgpu_fp8_decode_value_flags_t value_flags =
      LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NONE;
  if (!loom_amdgpu_vector_fp8_query_storage_pair_set(
          plan, value_flag_cache, pair_count, pair_storage, &value_flags)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_ensure_fp8_software_decode(
      context, plan->source_element_type, sgpr_type, mask_type, decode_plan));
  if (!loom_amdgpu_can_emit_fp8_pair_to_packed_bf16(*decode_plan,
                                                    value_flags)) {
    return iree_ok_status();
  }
  if (iree_any_bit_set(
          selection_flags,
          LOOM_AMDGPU_VECTOR_FP8_PACKED_BF16_SELECTION_FLAG_REQUIRE_PREFERRED) &&
      !loom_amdgpu_fp8_prefers_packed_bf16_pair_decode(*decode_plan,
                                                       value_flags)) {
    return iree_ok_status();
  }

  loom_amdgpu_fp8_packed_u16_pair_source_t
      pair_sources[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_vector_fp8_pair_sources(
      context, source_op, plan, pair_storage, pair_count, low_source,
      source_lane_type, pair_sources));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_pairs_to_packed_bf16(
      context, source_op, *decode_plan, pair_sources, pair_count, value_flags,
      result_lane_type, *sgpr_type, *mask_type, out_low_packets));
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_try_lower_vector_fp8_pairs_to_packed_f16(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    const loom_amdgpu_vector_fp8_decode_value_flag_cache_t* value_flag_cache,
    const loom_amdgpu_fp8_decode_plan_t** decode_plan,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t result_lane_type, loom_type_t* sgpr_type,
    loom_type_t* mask_type, uint32_t pair_count,
    loom_value_id_t* out_low_packets) {
  for (uint32_t i = 0; i < pair_count; ++i) {
    out_low_packets[i] = LOOM_VALUE_ID_INVALID;
  }
  loom_amdgpu_vector_fp8_pair_storage_t
      pair_storage[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {0};
  loom_amdgpu_fp8_decode_value_flags_t value_flags =
      LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NONE;
  if (!loom_amdgpu_vector_fp8_query_storage_pair_set(
          plan, value_flag_cache, pair_count, pair_storage, &value_flags)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_ensure_fp8_software_decode(
      context, plan->source_element_type, sgpr_type, mask_type, decode_plan));
  if (!loom_amdgpu_can_emit_fp8_pair_to_packed_f16_finite(*decode_plan,
                                                          value_flags)) {
    return iree_ok_status();
  }

  loom_amdgpu_fp8_packed_u16_pair_source_t
      pair_sources[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_vector_fp8_pair_sources(
      context, source_op, plan, pair_storage, pair_count, low_source,
      source_lane_type, pair_sources));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_pairs_to_packed_f16_finite(
      context, source_op, *decode_plan, pair_sources, pair_count, value_flags,
      result_lane_type, *sgpr_type, *mask_type, out_low_packets));
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_try_lower_vector_fp8_pair_to_packed_f16(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    const loom_amdgpu_vector_fp8_decode_value_flag_cache_t* value_flag_cache,
    const loom_amdgpu_fp8_decode_plan_t** decode_plan,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t result_lane_type, loom_type_t* sgpr_type,
    loom_type_t* mask_type, uint32_t lane_index,
    loom_value_id_t* out_low_packet) {
  *out_low_packet = LOOM_VALUE_ID_INVALID;

  loom_amdgpu_vector_fp8_pair_storage_t pair_storage;
  if (!loom_amdgpu_vector_fp8_query_storage_pair(plan, lane_index,
                                                 &pair_storage)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_ensure_fp8_software_decode(
      context, plan->source_element_type, sgpr_type, mask_type, decode_plan));
  const loom_amdgpu_fp8_decode_value_flags_t value_flags =
      loom_amdgpu_vector_fp8_pair_decode_value_flags(
          value_flag_cache, lane_index, pair_storage.live_lane_count);
  if (!loom_amdgpu_can_emit_fp8_pair_to_packed_f16_finite(*decode_plan,
                                                          value_flags)) {
    return iree_ok_status();
  }

  loom_value_id_t source_register = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_vector_fp8_pair_register(
      context, source_op, plan, &pair_storage, low_source, source_lane_type,
      &source_register));
  const loom_amdgpu_fp8_packed_u16_pair_source_t pair_source = {
      .source_register = source_register,
      .byte_offset = pair_storage.byte_offset,
      .live_lane_count = pair_storage.live_lane_count,
  };
  return loom_amdgpu_emit_fp8_pairs_to_packed_f16_finite(
      context, source_op, *decode_plan, &pair_source, 1, value_flags,
      result_lane_type, *sgpr_type, *mask_type, out_low_packet);
}

static iree_status_t loom_amdgpu_try_lower_vector_fp8_pairs_to_f32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    const loom_amdgpu_vector_fp8_decode_value_flag_cache_t* value_flag_cache,
    const loom_amdgpu_fp8_decode_plan_t** decode_plan,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t result_lane_type, loom_type_t* sgpr_type,
    loom_type_t* mask_type, loom_value_id_t* out_low_lanes) {
  out_low_lanes[0] = LOOM_VALUE_ID_INVALID;
  if (plan->lane_count == 0 ||
      plan->lane_count > LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES ||
      (plan->lane_count & 1u) != 0) {
    return iree_ok_status();
  }

  const uint32_t pair_count = plan->lane_count / 2u;
  loom_value_id_t packed_bf16[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS];
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(packed_bf16); ++i) {
    packed_bf16[i] = LOOM_VALUE_ID_INVALID;
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_try_lower_vector_fp8_pairs_to_packed_bf16(
      context, source_op, plan, value_flag_cache, decode_plan, low_source,
      source_lane_type, result_lane_type, sgpr_type, mask_type, pair_count,
      LOOM_AMDGPU_VECTOR_FP8_PACKED_BF16_SELECTION_FLAG_NONE, packed_bf16));
  if (packed_bf16[0] == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }

  for (uint32_t pair_index = 0; pair_index < pair_count; ++pair_index) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_bf16_register_lane_as_f32_bits(
        context, source_op, packed_bf16[pair_index], 0, result_lane_type,
        &out_low_lanes[pair_index * 2u]));
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_bf16_register_lane_as_f32_bits(
        context, source_op, packed_bf16[pair_index], 1, result_lane_type,
        &out_low_lanes[pair_index * 2u + 1u]));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_try_lower_vector_fp8_pair_to_f32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    const loom_amdgpu_vector_fp8_decode_value_flag_cache_t* value_flag_cache,
    const loom_amdgpu_fp8_decode_plan_t** decode_plan,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t result_lane_type, loom_type_t* sgpr_type,
    loom_type_t* mask_type, uint32_t lane_index,
    loom_value_id_t* out_low_lanes) {
  out_low_lanes[0] = LOOM_VALUE_ID_INVALID;
  out_low_lanes[1] = LOOM_VALUE_ID_INVALID;

  if (lane_index + 1u >= plan->lane_count) {
    return iree_ok_status();
  }

  loom_value_id_t packed_bf16 = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_try_lower_vector_fp8_pair_to_packed_bf16(
      context, source_op, plan, value_flag_cache, decode_plan, low_source,
      source_lane_type, result_lane_type, sgpr_type, mask_type, lane_index,
      &packed_bf16));
  if (packed_bf16 == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_bf16_register_lane_as_f32_bits(
      context, source_op, packed_bf16, 0, result_lane_type, &out_low_lanes[0]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_bf16_register_lane_as_f32_bits(
      context, source_op, packed_bf16, 1, result_lane_type, &out_low_lanes[1]));
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_vector_fp8_to_f32_lanes(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t result_lane_type, loom_value_id_t* lanes) {
  IREE_ASSERT_TRUE(
      !loom_amdgpu_vector_16bit_float_conversion_plan_has_scale(plan) ||
      loom_amdgpu_vector_16bit_float_conversion_plan_has_e8m0_scale(plan));

  for (uint32_t lane_index = 0; lane_index < plan->lane_count; ++lane_index) {
    lanes[lane_index] = LOOM_VALUE_ID_INVALID;
  }
  loom_value_id_t low_e8m0_scale = LOOM_VALUE_ID_INVALID;
  if (loom_amdgpu_vector_16bit_float_conversion_plan_has_e8m0_scale(plan)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_vector_scale_source(
        context, source_op, plan, &low_e8m0_scale));
  }
  uint32_t e8m0_pk8_lane_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_try_lower_vector_fp8_e8m0_pk8_native(
      context, source_op, plan, low_source, low_e8m0_scale, source_lane_type,
      result_lane_type, lanes, &e8m0_pk8_lane_count));
  if (e8m0_pk8_lane_count == plan->lane_count) {
    return iree_ok_status();
  }
  if (loom_amdgpu_vector_16bit_float_conversion_plan_has_e8m0_scale(plan)) {
    IREE_ASSERT_EQ(e8m0_pk8_lane_count, plan->lane_count);
    IREE_ASSERT_UNREACHABLE(
        "accepted AMDGPU E8M0 pk8 FP8 vector decode did not cover all lanes");
    IREE_BUILTIN_UNREACHABLE();
  }

  loom_type_t mask_type = loom_type_none();
  loom_type_t sgpr_type = loom_type_none();
  const loom_amdgpu_vector_extract_plan_t extract_plan =
      loom_amdgpu_vector_fp8_extract_plan(plan);
  const loom_amdgpu_fp8_native_descriptors_t* native_descriptors = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_get_fp8_native_descriptors(
      context, plan->source_element_type, LOOM_SCALAR_TYPE_F32,
      &native_descriptors));
  const bool native_pair_present =
      native_descriptors != NULL &&
      iree_any_bit_set(native_descriptors->flags,
                       LOOM_AMDGPU_FP8_NATIVE_DESCRIPTOR_FLAG_HAS_PAIR);
  const bool native_lane_present =
      native_descriptors != NULL &&
      iree_any_bit_set(native_descriptors->flags,
                       LOOM_AMDGPU_FP8_NATIVE_DESCRIPTOR_FLAG_HAS_LANE);

  loom_type_t result_pair_type = loom_type_none();
  if (native_pair_present) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_vgpr_range_type(context, 2, &result_pair_type));
  }

  loom_amdgpu_vector_fp8_decode_value_flag_cache_t value_flag_cache;
  loom_amdgpu_vector_fp8_decode_value_flag_cache_initialize(context, plan,
                                                            &value_flag_cache);

  const loom_amdgpu_fp8_decode_plan_t* decode_plan = NULL;
  if (!native_pair_present && !native_lane_present) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_try_lower_vector_fp8_pairs_to_f32(
        context, source_op, plan, &value_flag_cache, &decode_plan, low_source,
        source_lane_type, result_lane_type, &sgpr_type, &mask_type, lanes));
    if (lanes[0] != LOOM_VALUE_ID_INVALID) {
      return iree_ok_status();
    }
  }

  for (uint32_t i = 0; i < plan->lane_count;) {
    if (lanes[i] != LOOM_VALUE_ID_INVALID) {
      ++i;
      continue;
    }
    if (native_pair_present) {
      loom_value_id_t native_lanes[2] = {LOOM_VALUE_ID_INVALID,
                                         LOOM_VALUE_ID_INVALID};
      uint32_t native_lane_count = 0;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_try_lower_vector_fp8_storage_pair_descriptor_to_f32_native(
              context, source_op, plan, &native_descriptors->pair_descriptor,
              low_source, /*extra_operands=*/NULL, /*extra_operand_count=*/0,
              source_lane_type, result_lane_type, result_pair_type, i,
              native_lanes, &native_lane_count));
      if (native_lanes[0] != LOOM_VALUE_ID_INVALID) {
        for (uint32_t lane_offset = 0; lane_offset < native_lane_count;
             ++lane_offset) {
          lanes[i + lane_offset] = native_lanes[lane_offset];
        }
        i += native_lane_count;
        continue;
      }
    }

    if (native_lane_present) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_lower_vector_fp8_to_f32_native_lane(
          context, source_op, plan, &extract_plan,
          &native_descriptors->lane_descriptor, low_source, source_lane_type,
          result_lane_type, i, &lanes[i]));
      ++i;
      continue;
    }

    loom_value_id_t packed_decode_lanes[2] = {LOOM_VALUE_ID_INVALID,
                                              LOOM_VALUE_ID_INVALID};
    IREE_RETURN_IF_ERROR(loom_amdgpu_try_lower_vector_fp8_pair_to_f32(
        context, source_op, plan, &value_flag_cache, &decode_plan, low_source,
        source_lane_type, result_lane_type, &sgpr_type, &mask_type, i,
        packed_decode_lanes));
    if (packed_decode_lanes[0] != LOOM_VALUE_ID_INVALID) {
      lanes[i] = packed_decode_lanes[0];
      lanes[i + 1u] = packed_decode_lanes[1];
      i += 2u;
      continue;
    }

    IREE_RETURN_IF_ERROR(loom_amdgpu_lower_vector_fp8_to_f32_software_lane(
        context, source_op, plan, &extract_plan, &value_flag_cache,
        &decode_plan, low_source, source_lane_type, result_lane_type,
        &sgpr_type, &mask_type, i, &lanes[i]));
    ++i;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_vector_fp8_scalef32_to_f32_lanes(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    loom_value_id_t low_source, loom_value_id_t low_scale,
    loom_type_t source_lane_type, loom_type_t result_lane_type,
    loom_value_id_t* lanes) {
  loom_amdgpu_vector_16bit_float_conversion_plan_t unscaled_plan;
  loom_amdgpu_vector_fp8_unscaled_plan(plan, &unscaled_plan);
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vector_fp8_to_f32_lanes(
      context, source_op, &unscaled_plan, low_source, source_lane_type,
      result_lane_type, lanes));
  for (uint32_t lane_index = 0; lane_index < plan->lane_count; ++lane_index) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_F32,
        lanes[lane_index], low_scale, result_lane_type, &lanes[lane_index]));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_lower_vector_fp8_scalef32_to_f32_software_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    const loom_amdgpu_vector_extract_plan_t* extract_plan,
    const loom_amdgpu_vector_fp8_decode_value_flag_cache_t* value_flag_cache,
    const loom_amdgpu_fp8_decode_plan_t** decode_plan,
    loom_value_id_t low_source, loom_value_id_t low_scale,
    loom_type_t source_lane_type, loom_type_t result_lane_type,
    loom_type_t* sgpr_type, loom_type_t* mask_type, uint32_t lane_index,
    loom_value_id_t* out_low_lane) {
  loom_value_id_t unscaled_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lower_vector_fp8_to_f32_software_lane(
      context, source_op, plan, extract_plan, value_flag_cache, decode_plan,
      low_source, source_lane_type, result_lane_type, sgpr_type, mask_type,
      lane_index, &unscaled_lane));
  return loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_F32, unscaled_lane,
      low_scale, result_lane_type, out_low_lane);
}

static iree_status_t loom_amdgpu_lower_vector_fp8_scalef32_to_f32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t result_lane_type) {
  loom_value_id_t low_scale = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_TRUE(
      loom_amdgpu_vector_16bit_float_conversion_plan_has_f32_scale(plan));
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_vector_scale_source(
      context, source_op, plan, &low_scale));

  const loom_low_lower_resolved_descriptor_t* descriptor = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_get_fp8_scalef32_descriptor(
      context, plan->source_element_type, plan->result_element_type,
      &descriptor));

  loom_value_id_t lanes[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  uint32_t selected_lane_count = 0;
  for (uint32_t lane_index = 0; lane_index < plan->lane_count; ++lane_index) {
    lanes[lane_index] = LOOM_VALUE_ID_INVALID;
  }

  if (descriptor != NULL) {
    loom_type_t result_pair_type = loom_type_none();
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_vgpr_range_type(context, 2, &result_pair_type));

    if (plan->lane_count == 2u) {
      loom_amdgpu_vector_fp8_pair_storage_t pair_storage;
      loom_value_id_t converted_pair = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_try_emit_vector_fp8_pair_descriptor(
          context, source_op, plan, descriptor, low_source, source_lane_type,
          &low_scale, /*extra_operand_count=*/1, result_pair_type,
          /*lane_index=*/0, &pair_storage, &converted_pair));
      if (converted_pair != LOOM_VALUE_ID_INVALID) {
        return loom_low_lower_bind_value(context, plan->result, converted_pair);
      }
    }

    for (uint32_t lane_index = 0; lane_index < plan->lane_count;
         lane_index += 2u) {
      loom_value_id_t native_lanes[2] = {LOOM_VALUE_ID_INVALID,
                                         LOOM_VALUE_ID_INVALID};
      uint32_t native_lane_count = 0;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_try_lower_vector_fp8_storage_pair_descriptor_to_f32_native(
              context, source_op, plan, descriptor, low_source, &low_scale,
              /*extra_operand_count=*/1, source_lane_type, result_lane_type,
              result_pair_type, lane_index, native_lanes, &native_lane_count));
      if (native_lanes[0] == LOOM_VALUE_ID_INVALID) {
        continue;
      }
      for (uint32_t lane_offset = 0; lane_offset < native_lane_count;
           ++lane_offset) {
        lanes[lane_index + lane_offset] = native_lanes[lane_offset];
      }
      selected_lane_count += native_lane_count;
    }
    if (selected_lane_count == plan->lane_count) {
      return loom_amdgpu_bind_low_register_range(
          context, source_op, plan->result, lanes, plan->lane_count);
    }
  }

  if (selected_lane_count != 0) {
    loom_type_t mask_type = loom_type_none();
    loom_type_t sgpr_type = loom_type_none();
    const loom_amdgpu_vector_extract_plan_t extract_plan =
        loom_amdgpu_vector_fp8_extract_plan(plan);
    loom_amdgpu_vector_fp8_decode_value_flag_cache_t value_flag_cache;
    loom_amdgpu_vector_fp8_decode_value_flag_cache_initialize(
        context, plan, &value_flag_cache);
    const loom_amdgpu_fp8_decode_plan_t* decode_plan = NULL;
    for (uint32_t lane_index = 0; lane_index < plan->lane_count; ++lane_index) {
      if (lanes[lane_index] != LOOM_VALUE_ID_INVALID) {
        continue;
      }
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_lower_vector_fp8_scalef32_to_f32_software_lane(
              context, source_op, plan, &extract_plan, &value_flag_cache,
              &decode_plan, low_source, low_scale, source_lane_type,
              result_lane_type, &sgpr_type, &mask_type, lane_index,
              &lanes[lane_index]));
    }
    return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                               lanes, plan->lane_count);
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vector_fp8_scalef32_to_f32_lanes(
      context, source_op, plan, low_source, low_scale, source_lane_type,
      result_lane_type, lanes));
  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             lanes, plan->lane_count);
}

static iree_status_t loom_amdgpu_pack_f32_lane_to_f16_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_lane, uint32_t register_lane, loom_type_t lane_type,
    loom_value_id_t* inout_packed) {
  loom_value_id_t half_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_unary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F16_F32, source_lane,
      lane_type, &half_lane));
  IREE_RETURN_IF_ERROR(loom_amdgpu_compose_vgpr_16bit_float_lane_bits(
      context, source_op, half_lane, 0, lane_type, &half_lane));
  return loom_amdgpu_pack_bits_into_register(context, source_op, half_lane,
                                             register_lane * 16u, lane_type,
                                             inout_packed);
}

static iree_status_t loom_amdgpu_pack_f32_lanes_to_16bit_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_scalar_type_t result_element_type,
    const loom_amdgpu_bf16_pack_descriptors_t* bf16_pack_descriptors,
    const loom_value_id_t* f32_lanes, uint32_t f32_lane_count,
    loom_type_t lane_type, loom_value_id_t* out_packed) {
  IREE_ASSERT_GE(f32_lane_count, 1u);
  IREE_ASSERT_LE(f32_lane_count, 2u);

  *out_packed = LOOM_VALUE_ID_INVALID;
  if (result_element_type == LOOM_SCALAR_TYPE_BF16) {
    loom_value_id_t high_lane = LOOM_VALUE_ID_INVALID;
    if (f32_lane_count == 2u) {
      high_lane = f32_lanes[1];
    } else {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0,
          lane_type, &high_lane));
    }
    return loom_amdgpu_emit_f32_pair_to_packed_bf16_with_descriptors(
        context, source_op, bf16_pack_descriptors, f32_lanes[0], high_lane,
        lane_type, out_packed);
  }

  IREE_ASSERT_EQ(result_element_type, LOOM_SCALAR_TYPE_F16);
  for (uint32_t register_lane = 0; register_lane < f32_lane_count;
       ++register_lane) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_pack_f32_lane_to_f16_register(
        context, source_op, f32_lanes[register_lane], register_lane, lane_type,
        out_packed));
  }
  return iree_ok_status();
}

static iree_status_t
loom_amdgpu_try_lower_vector_fp8_storage_pair_descriptor_to_packed_16bit_from_f32_native(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t low_source, const loom_value_id_t* extra_operands,
    uint32_t extra_operand_count, loom_type_t source_lane_type,
    loom_type_t result_lane_type, loom_type_t result_pair_type,
    const loom_amdgpu_bf16_pack_descriptors_t* bf16_pack_descriptors,
    uint32_t lane_index, loom_value_id_t* out_low_pair) {
  *out_low_pair = LOOM_VALUE_ID_INVALID;

  loom_value_id_t f32_lanes[2] = {LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID};
  uint32_t f32_lane_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_try_lower_vector_fp8_storage_pair_descriptor_to_f32_native(
          context, source_op, plan, descriptor, low_source, extra_operands,
          extra_operand_count, source_lane_type, result_lane_type,
          result_pair_type, lane_index, f32_lanes, &f32_lane_count));
  if (f32_lanes[0] == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  return loom_amdgpu_pack_f32_lanes_to_16bit_register(
      context, source_op, plan->result_element_type, bf16_pack_descriptors,
      f32_lanes, f32_lane_count, result_lane_type, out_low_pair);
}

static iree_status_t loom_amdgpu_lower_vector_fp8_scalef32_to_packed_16bit(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t result_lane_type) {
  loom_value_id_t low_scale = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_TRUE(
      loom_amdgpu_vector_16bit_float_conversion_plan_has_f32_scale(plan));
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_vector_scale_source(
      context, source_op, plan, &low_scale));

  const loom_low_lower_resolved_descriptor_t* descriptor = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_get_fp8_scalef32_descriptor(
      context, plan->source_element_type, plan->result_element_type,
      &descriptor));

  loom_value_id_t packed_registers[LOOM_AMDGPU_MAX_PACKED_16BIT_FLOAT_LANES];
  uint32_t missing_register_count = plan->result_register_count;
  for (uint32_t register_index = 0;
       register_index < plan->result_register_count; ++register_index) {
    packed_registers[register_index] = LOOM_VALUE_ID_INVALID;
  }

  if (descriptor != NULL) {
    for (uint32_t register_index = 0;
         register_index < plan->result_register_count; ++register_index) {
      const uint32_t lane_index = register_index * 2u;
      loom_amdgpu_vector_fp8_pair_storage_t pair_storage;
      IREE_RETURN_IF_ERROR(loom_amdgpu_try_emit_vector_fp8_pair_descriptor(
          context, source_op, plan, descriptor, low_source, source_lane_type,
          &low_scale, /*extra_operand_count=*/1, result_lane_type, lane_index,
          &pair_storage, &packed_registers[register_index]));
      if (packed_registers[register_index] != LOOM_VALUE_ID_INVALID) {
        --missing_register_count;
      }
    }
  }

  const loom_low_lower_resolved_descriptor_t* f32_descriptor = NULL;
  if (missing_register_count != 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_get_fp8_scalef32_descriptor(
        context, plan->source_element_type, LOOM_SCALAR_TYPE_F32,
        &f32_descriptor));
  }
  if (f32_descriptor != NULL) {
    loom_type_t result_pair_type = loom_type_none();
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_vgpr_range_type(context, 2, &result_pair_type));
    const loom_amdgpu_bf16_pack_descriptors_t* bf16_pack_descriptors = NULL;
    if (plan->result_element_type == LOOM_SCALAR_TYPE_BF16) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_get_bf16_pack_descriptors(
          context, &bf16_pack_descriptors));
    }
    for (uint32_t register_index = 0;
         register_index < plan->result_register_count; ++register_index) {
      if (packed_registers[register_index] != LOOM_VALUE_ID_INVALID) {
        continue;
      }
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_try_lower_vector_fp8_storage_pair_descriptor_to_packed_16bit_from_f32_native(
              context, source_op, plan, f32_descriptor, low_source, &low_scale,
              /*extra_operand_count=*/1, source_lane_type, result_lane_type,
              result_pair_type, bf16_pack_descriptors, register_index * 2u,
              &packed_registers[register_index]));
      if (packed_registers[register_index] != LOOM_VALUE_ID_INVALID) {
        --missing_register_count;
      }
    }
  }

  if (missing_register_count == 0) {
    return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                               packed_registers,
                                               plan->result_register_count);
  }

  if (missing_register_count == plan->result_register_count) {
    loom_value_id_t f32_lanes[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vector_fp8_scalef32_to_f32_lanes(
        context, source_op, plan, low_source, low_scale, source_lane_type,
        result_lane_type, f32_lanes));

    const loom_amdgpu_bf16_pack_descriptors_t* bf16_pack_descriptors = NULL;
    if (plan->result_element_type == LOOM_SCALAR_TYPE_BF16) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_get_bf16_pack_descriptors(
          context, &bf16_pack_descriptors));
    }
    for (uint32_t register_index = 0;
         register_index < plan->result_register_count; ++register_index) {
      const uint32_t lane_base = register_index * 2u;
      const uint32_t remaining_lane_count = plan->lane_count - lane_base;
      const uint32_t register_lane_count =
          remaining_lane_count < 2u ? remaining_lane_count : 2u;
      IREE_RETURN_IF_ERROR(loom_amdgpu_pack_f32_lanes_to_16bit_register(
          context, source_op, plan->result_element_type, bf16_pack_descriptors,
          &f32_lanes[lane_base], register_lane_count, result_lane_type,
          &packed_registers[register_index]));
    }

    return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                               packed_registers,
                                               plan->result_register_count);
  }

  loom_type_t mask_type = loom_type_none();
  loom_type_t sgpr_type = loom_type_none();
  const loom_amdgpu_vector_extract_plan_t extract_plan =
      loom_amdgpu_vector_fp8_extract_plan(plan);
  loom_amdgpu_vector_fp8_decode_value_flag_cache_t value_flag_cache;
  loom_amdgpu_vector_fp8_decode_value_flag_cache_initialize(context, plan,
                                                            &value_flag_cache);
  const loom_amdgpu_fp8_decode_plan_t* decode_plan = NULL;

  const loom_amdgpu_bf16_pack_descriptors_t* bf16_pack_descriptors = NULL;
  if (plan->result_element_type == LOOM_SCALAR_TYPE_BF16) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_get_bf16_pack_descriptors(context, &bf16_pack_descriptors));
  } else {
    IREE_ASSERT_EQ(plan->result_element_type, LOOM_SCALAR_TYPE_F16);
  }
  for (uint32_t register_index = 0;
       register_index < plan->result_register_count; ++register_index) {
    if (packed_registers[register_index] != LOOM_VALUE_ID_INVALID) {
      continue;
    }
    const uint32_t lane_base = register_index * 2u;
    loom_value_id_t f32_lanes[2] = {LOOM_VALUE_ID_INVALID,
                                    LOOM_VALUE_ID_INVALID};
    const uint32_t remaining_lane_count = plan->lane_count - lane_base;
    const uint32_t register_lane_count =
        remaining_lane_count < 2u ? remaining_lane_count : 2u;
    for (uint32_t register_lane = 0; register_lane < register_lane_count;
         ++register_lane) {
      const uint32_t lane_index = lane_base + register_lane;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_lower_vector_fp8_scalef32_to_f32_software_lane(
              context, source_op, plan, &extract_plan, &value_flag_cache,
              &decode_plan, low_source, low_scale, source_lane_type,
              result_lane_type, &sgpr_type, &mask_type, lane_index,
              &f32_lanes[register_lane]));
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_pack_f32_lanes_to_16bit_register(
        context, source_op, plan->result_element_type, bf16_pack_descriptors,
        f32_lanes, register_lane_count, result_lane_type,
        &packed_registers[register_index]));
  }

  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             packed_registers,
                                             plan->result_register_count);
}

static iree_status_t loom_amdgpu_lower_vector_fp8_to_f32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t result_lane_type) {
  if (loom_amdgpu_vector_16bit_float_conversion_plan_has_f32_scale(plan)) {
    return loom_amdgpu_lower_vector_fp8_scalef32_to_f32(
        context, source_op, plan, low_source, source_lane_type,
        result_lane_type);
  }

  loom_value_id_t lanes[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vector_fp8_to_f32_lanes(
      context, source_op, plan, low_source, source_lane_type, result_lane_type,
      lanes));
  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             lanes, plan->lane_count);
}

static iree_status_t loom_amdgpu_lower_vector_fp8_to_packed_bf16(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t result_lane_type) {
  loom_type_t mask_type = loom_type_none();
  loom_type_t sgpr_type = loom_type_none();

  loom_value_id_t packed_registers[LOOM_AMDGPU_MAX_PACKED_16BIT_FLOAT_LANES];
  loom_value_id_t low_e8m0_scale = LOOM_VALUE_ID_INVALID;
  if (loom_amdgpu_vector_16bit_float_conversion_plan_has_e8m0_scale(plan)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_vector_scale_source(
        context, source_op, plan, &low_e8m0_scale));
  }
  uint32_t e8m0_pk8_register_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_try_lower_vector_fp8_e8m0_pk8_native(
      context, source_op, plan, low_source, low_e8m0_scale, source_lane_type,
      result_lane_type, packed_registers, &e8m0_pk8_register_count));
  if (e8m0_pk8_register_count == plan->result_register_count) {
    return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                               packed_registers,
                                               plan->result_register_count);
  }
  if (loom_amdgpu_vector_16bit_float_conversion_plan_has_e8m0_scale(plan)) {
    IREE_ASSERT_EQ(e8m0_pk8_register_count, plan->result_register_count);
    IREE_ASSERT_UNREACHABLE(
        "accepted AMDGPU E8M0 pk8 FP8 vector decode did not cover all lanes");
    IREE_BUILTIN_UNREACHABLE();
  }

  uint32_t identity_scalef32_register_count = 0;
  if (e8m0_pk8_register_count == 0) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_try_lower_vector_fp8_identity_scalef32_to_packed_16bit_native(
            context, source_op, plan, low_source, source_lane_type,
            result_lane_type, packed_registers,
            &identity_scalef32_register_count));
    if (identity_scalef32_register_count == plan->result_register_count) {
      return loom_amdgpu_bind_low_register_range(context, source_op,
                                                 plan->result, packed_registers,
                                                 plan->result_register_count);
    }
  }

  const loom_amdgpu_vector_extract_plan_t extract_plan =
      loom_amdgpu_vector_fp8_extract_plan(plan);
  const loom_amdgpu_fp8_native_descriptors_t* native_descriptors = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_get_fp8_native_descriptors(
      context, plan->source_element_type, LOOM_SCALAR_TYPE_F32,
      &native_descriptors));
  const bool native_pair_present =
      native_descriptors != NULL &&
      iree_any_bit_set(native_descriptors->flags,
                       LOOM_AMDGPU_FP8_NATIVE_DESCRIPTOR_FLAG_HAS_PAIR);
  const bool native_lane_present =
      native_descriptors != NULL &&
      iree_any_bit_set(native_descriptors->flags,
                       LOOM_AMDGPU_FP8_NATIVE_DESCRIPTOR_FLAG_HAS_LANE);

  loom_type_t result_pair_type = loom_type_none();
  if (native_pair_present) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_vgpr_range_type(context, 2, &result_pair_type));
  }

  const loom_amdgpu_bf16_pack_descriptors_t* bf16_pack_descriptors = NULL;
  if (native_pair_present || native_lane_present) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_get_bf16_pack_descriptors(context, &bf16_pack_descriptors));
  }

  loom_amdgpu_vector_fp8_decode_value_flag_cache_t value_flag_cache;
  loom_amdgpu_vector_fp8_decode_value_flag_cache_initialize(context, plan,
                                                            &value_flag_cache);

  const loom_amdgpu_fp8_decode_plan_t* decode_plan = NULL;
  if (!native_pair_present && !native_lane_present &&
      identity_scalef32_register_count == 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_try_lower_vector_fp8_pairs_to_packed_bf16(
        context, source_op, plan, &value_flag_cache, &decode_plan, low_source,
        source_lane_type, result_lane_type, &sgpr_type, &mask_type,
        plan->result_register_count,
        LOOM_AMDGPU_VECTOR_FP8_PACKED_BF16_SELECTION_FLAG_NONE,
        packed_registers));
    if (packed_registers[0] != LOOM_VALUE_ID_INVALID) {
      return loom_amdgpu_bind_low_register_range(context, source_op,
                                                 plan->result, packed_registers,
                                                 plan->result_register_count);
    }
  } else if (identity_scalef32_register_count == 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_try_lower_vector_fp8_pairs_to_packed_bf16(
        context, source_op, plan, &value_flag_cache, &decode_plan, low_source,
        source_lane_type, result_lane_type, &sgpr_type, &mask_type,
        plan->result_register_count,
        LOOM_AMDGPU_VECTOR_FP8_PACKED_BF16_SELECTION_FLAG_REQUIRE_PREFERRED,
        packed_registers));
    if (packed_registers[0] != LOOM_VALUE_ID_INVALID) {
      return loom_amdgpu_bind_low_register_range(context, source_op,
                                                 plan->result, packed_registers,
                                                 plan->result_register_count);
    }
  }

  for (uint32_t register_index = 0;
       register_index < plan->result_register_count; ++register_index) {
    if (packed_registers[register_index] != LOOM_VALUE_ID_INVALID) {
      continue;
    }
    const uint32_t lane_base = register_index * 2u;
    if (native_pair_present) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_try_lower_vector_fp8_storage_pair_descriptor_to_packed_16bit_from_f32_native(
              context, source_op, plan, &native_descriptors->pair_descriptor,
              low_source, /*extra_operands=*/NULL, /*extra_operand_count=*/0,
              source_lane_type, result_lane_type, result_pair_type,
              bf16_pack_descriptors, lane_base,
              &packed_registers[register_index]));
      if (packed_registers[register_index] != LOOM_VALUE_ID_INVALID) {
        continue;
      }
    }

    if (native_lane_present) {
      loom_value_id_t f32_lanes[2] = {LOOM_VALUE_ID_INVALID,
                                      LOOM_VALUE_ID_INVALID};
      for (uint32_t register_lane = 0; register_lane < 2u; ++register_lane) {
        const uint32_t lane_index = lane_base + register_lane;
        if (lane_index >= plan->lane_count) {
          IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
              context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0,
              result_lane_type, &f32_lanes[register_lane]));
          continue;
        }
        IREE_RETURN_IF_ERROR(loom_amdgpu_lower_vector_fp8_to_f32_native_lane(
            context, source_op, plan, &extract_plan,
            &native_descriptors->lane_descriptor, low_source, source_lane_type,
            result_lane_type, lane_index, &f32_lanes[register_lane]));
      }
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_f32_pair_to_packed_bf16_with_descriptors(
              context, source_op, bf16_pack_descriptors, f32_lanes[0],
              f32_lanes[1], result_lane_type,
              &packed_registers[register_index]));
      continue;
    }

    IREE_RETURN_IF_ERROR(loom_amdgpu_try_lower_vector_fp8_pair_to_packed_bf16(
        context, source_op, plan, &value_flag_cache, &decode_plan, low_source,
        source_lane_type, result_lane_type, &sgpr_type, &mask_type, lane_base,
        &packed_registers[register_index]));
    if (packed_registers[register_index] != LOOM_VALUE_ID_INVALID) {
      continue;
    }

    loom_value_id_t lanes[2] = {LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID};
    for (uint32_t register_lane = 0; register_lane < 2u; ++register_lane) {
      const uint32_t lane_index = lane_base + register_lane;
      if (lane_index >= plan->lane_count) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
            context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0,
            result_lane_type, &lanes[register_lane]));
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_ensure_fp8_software_decode(
          context, plan->source_element_type, &sgpr_type, &mask_type,
          &decode_plan));
      loom_value_id_t low_byte = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_extract_vector_fp8_lane(
          context, source_op, plan, &extract_plan, low_source, source_lane_type,
          lane_index, &low_byte));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_to_bf16_lane(
          context, source_op, decode_plan, low_byte,
          loom_amdgpu_vector_fp8_decode_value_flags(&value_flag_cache,
                                                    lane_index),
          result_lane_type, sgpr_type, mask_type, &lanes[register_lane]));
    }
    const loom_low_lower_resolved_descriptor_t* pack_u16_descriptor =
        iree_any_bit_set(decode_plan->flags,
                         LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PACK_U16)
            ? &decode_plan->pack_u16_descriptor
            : NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_packed_bf16_lane_pair(
        context, source_op, pack_u16_descriptor, lanes[0], lanes[1],
        result_lane_type, &packed_registers[register_index]));
  }

  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             packed_registers,
                                             plan->result_register_count);
}

static iree_status_t loom_amdgpu_lower_vector_fp8_to_packed_f16(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t result_lane_type) {
  loom_value_id_t packed_registers[LOOM_AMDGPU_MAX_PACKED_16BIT_FLOAT_LANES];
  uint32_t missing_register_count = plan->result_register_count;
  for (uint32_t register_index = 0;
       register_index < plan->result_register_count; ++register_index) {
    packed_registers[register_index] = LOOM_VALUE_ID_INVALID;
  }

  loom_value_id_t low_e8m0_scale = LOOM_VALUE_ID_INVALID;
  if (loom_amdgpu_vector_16bit_float_conversion_plan_has_e8m0_scale(plan)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_vector_scale_source(
        context, source_op, plan, &low_e8m0_scale));
  }
  uint32_t e8m0_pk8_register_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_try_lower_vector_fp8_e8m0_pk8_native(
      context, source_op, plan, low_source, low_e8m0_scale, source_lane_type,
      result_lane_type, packed_registers, &e8m0_pk8_register_count));
  missing_register_count -= e8m0_pk8_register_count;
  if (missing_register_count == 0) {
    return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                               packed_registers,
                                               plan->result_register_count);
  }
  if (loom_amdgpu_vector_16bit_float_conversion_plan_has_e8m0_scale(plan)) {
    IREE_ASSERT_EQ(e8m0_pk8_register_count, plan->result_register_count);
    IREE_ASSERT_UNREACHABLE(
        "accepted AMDGPU E8M0 pk8 FP8 vector decode did not cover all lanes");
    IREE_BUILTIN_UNREACHABLE();
  }

  const loom_amdgpu_fp8_native_descriptors_t* native_descriptors = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_get_fp8_native_descriptors(
      context, plan->source_element_type, LOOM_SCALAR_TYPE_F16,
      &native_descriptors));
  const bool native_pair_present =
      native_descriptors != NULL &&
      iree_any_bit_set(native_descriptors->flags,
                       LOOM_AMDGPU_FP8_NATIVE_DESCRIPTOR_FLAG_HAS_PAIR);

  if (native_pair_present) {
    for (uint32_t register_index = 0;
         register_index < plan->result_register_count; ++register_index) {
      loom_amdgpu_vector_fp8_pair_storage_t pair_storage;
      IREE_RETURN_IF_ERROR(loom_amdgpu_try_emit_vector_fp8_pair_descriptor(
          context, source_op, plan, &native_descriptors->pair_descriptor,
          low_source, source_lane_type, /*extra_operands=*/NULL,
          /*extra_operand_count=*/0, result_lane_type, register_index * 2u,
          &pair_storage, &packed_registers[register_index]));
      if (packed_registers[register_index] != LOOM_VALUE_ID_INVALID) {
        --missing_register_count;
      }
    }
    if (missing_register_count == 0) {
      return loom_amdgpu_bind_low_register_range(context, source_op,
                                                 plan->result, packed_registers,
                                                 plan->result_register_count);
    }
  }

  if (missing_register_count == plan->result_register_count) {
    uint32_t identity_scalef32_register_count = 0;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_try_lower_vector_fp8_identity_scalef32_to_packed_16bit_native(
            context, source_op, plan, low_source, source_lane_type,
            result_lane_type, packed_registers,
            &identity_scalef32_register_count));
    missing_register_count -= identity_scalef32_register_count;
    if (missing_register_count == 0) {
      return loom_amdgpu_bind_low_register_range(context, source_op,
                                                 plan->result, packed_registers,
                                                 plan->result_register_count);
    }
  }

  const loom_amdgpu_fp8_native_descriptors_t* f32_native_descriptors = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_get_fp8_native_descriptors(
      context, plan->source_element_type, LOOM_SCALAR_TYPE_F32,
      &f32_native_descriptors));
  const bool f32_native_pair_present =
      f32_native_descriptors != NULL &&
      iree_any_bit_set(f32_native_descriptors->flags,
                       LOOM_AMDGPU_FP8_NATIVE_DESCRIPTOR_FLAG_HAS_PAIR);
  if (f32_native_pair_present && missing_register_count != 0) {
    loom_type_t result_pair_type = loom_type_none();
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_vgpr_range_type(context, 2, &result_pair_type));
    for (uint32_t register_index = 0;
         register_index < plan->result_register_count; ++register_index) {
      if (packed_registers[register_index] != LOOM_VALUE_ID_INVALID) {
        continue;
      }
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_try_lower_vector_fp8_storage_pair_descriptor_to_packed_16bit_from_f32_native(
              context, source_op, plan,
              &f32_native_descriptors->pair_descriptor, low_source,
              /*extra_operands=*/NULL, /*extra_operand_count=*/0,
              source_lane_type, result_lane_type, result_pair_type,
              /*bf16_pack_descriptors=*/NULL, register_index * 2u,
              &packed_registers[register_index]));
      if (packed_registers[register_index] != LOOM_VALUE_ID_INVALID) {
        --missing_register_count;
      }
    }
    if (missing_register_count == 0) {
      return loom_amdgpu_bind_low_register_range(context, source_op,
                                                 plan->result, packed_registers,
                                                 plan->result_register_count);
    }
  }

  loom_type_t mask_type = loom_type_none();
  loom_type_t sgpr_type = loom_type_none();
  loom_amdgpu_vector_fp8_decode_value_flag_cache_t value_flag_cache;
  loom_amdgpu_vector_fp8_decode_value_flag_cache_initialize(context, plan,
                                                            &value_flag_cache);
  const loom_amdgpu_fp8_decode_plan_t* decode_plan = NULL;
  if (missing_register_count == plan->result_register_count) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_try_lower_vector_fp8_pairs_to_packed_f16(
        context, source_op, plan, &value_flag_cache, &decode_plan, low_source,
        source_lane_type, result_lane_type, &sgpr_type, &mask_type,
        plan->result_register_count, packed_registers));
    if (packed_registers[0] != LOOM_VALUE_ID_INVALID) {
      return loom_amdgpu_bind_low_register_range(context, source_op,
                                                 plan->result, packed_registers,
                                                 plan->result_register_count);
    }
  }

  if (missing_register_count != 0) {
    for (uint32_t register_index = 0;
         register_index < plan->result_register_count; ++register_index) {
      if (packed_registers[register_index] != LOOM_VALUE_ID_INVALID) {
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_try_lower_vector_fp8_pair_to_packed_f16(
          context, source_op, plan, &value_flag_cache, &decode_plan, low_source,
          source_lane_type, result_lane_type, &sgpr_type, &mask_type,
          register_index * 2u, &packed_registers[register_index]));
      if (packed_registers[register_index] != LOOM_VALUE_ID_INVALID) {
        --missing_register_count;
      }
    }
    if (missing_register_count == 0) {
      return loom_amdgpu_bind_low_register_range(context, source_op,
                                                 plan->result, packed_registers,
                                                 plan->result_register_count);
    }
  }

  if (missing_register_count == plan->result_register_count) {
    loom_value_id_t f32_lanes[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vector_fp8_to_f32_lanes(
        context, source_op, plan, low_source, source_lane_type,
        result_lane_type, f32_lanes));
    for (uint32_t register_index = 0;
         register_index < plan->result_register_count; ++register_index) {
      loom_value_id_t packed = LOOM_VALUE_ID_INVALID;
      const uint32_t lane_base = register_index * 2u;
      for (uint32_t register_lane = 0; register_lane < 2u; ++register_lane) {
        const uint32_t lane_index = lane_base + register_lane;
        if (lane_index >= plan->lane_count) {
          break;
        }
        IREE_RETURN_IF_ERROR(loom_amdgpu_pack_f32_lane_to_f16_register(
            context, source_op, f32_lanes[lane_index], register_lane,
            result_lane_type, &packed));
      }
      packed_registers[register_index] = packed;
    }
    return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                               packed_registers,
                                               plan->result_register_count);
  }

  const loom_amdgpu_vector_extract_plan_t extract_plan =
      loom_amdgpu_vector_fp8_extract_plan(plan);
  for (uint32_t register_index = 0;
       register_index < plan->result_register_count; ++register_index) {
    if (packed_registers[register_index] != LOOM_VALUE_ID_INVALID) {
      continue;
    }
    const uint32_t lane_base = register_index * 2u;
    const uint32_t remaining_lane_count = plan->lane_count - lane_base;
    const uint32_t register_lane_count =
        remaining_lane_count < 2u ? remaining_lane_count : 2u;
    loom_value_id_t f32_lanes[2] = {LOOM_VALUE_ID_INVALID,
                                    LOOM_VALUE_ID_INVALID};
    for (uint32_t register_lane = 0; register_lane < 2u; ++register_lane) {
      const uint32_t lane_index = lane_base + register_lane;
      if (register_lane >= register_lane_count) {
        break;
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_lower_vector_fp8_to_f32_software_lane(
          context, source_op, plan, &extract_plan, &value_flag_cache,
          &decode_plan, low_source, source_lane_type, result_lane_type,
          &sgpr_type, &mask_type, lane_index, &f32_lanes[register_lane]));
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_pack_f32_lanes_to_16bit_register(
        context, source_op, plan->result_element_type,
        /*bf16_pack_descriptors=*/NULL, f32_lanes, register_lane_count,
        result_lane_type, &packed_registers[register_index]));
  }

  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             packed_registers,
                                             plan->result_register_count);
}

static bool loom_amdgpu_scalar_type_is_fp8(loom_scalar_type_t type) {
  return type == LOOM_SCALAR_TYPE_F8E4M3 || type == LOOM_SCALAR_TYPE_F8E5M2;
}

typedef enum loom_amdgpu_vector_fp8_conversion_capability_bits_e {
  LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NONE = 0u,
  LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_E8M0_PK8_RESULT = 1u << 0,
  LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_SCALEF32_RESULT_PAIR = 1u << 1,
  LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_SCALEF32_F32_PAIR = 1u << 2,
  LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NATIVE_F32_PAIR = 1u << 3,
  LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NATIVE_F32_LANE = 1u << 4,
  LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NATIVE_BF16_PAIR = 1u << 5,
  LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NATIVE_F16_PAIR = 1u << 6,
  LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NATIVE_BF16_PACK = 1u << 7,
} loom_amdgpu_vector_fp8_conversion_capability_bits_t;
typedef uint32_t loom_amdgpu_vector_fp8_conversion_capabilities_t;

static bool loom_amdgpu_vector_fp8_plan_has_pair_storage(
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan) {
  for (uint32_t lane_index = 0; lane_index < plan->lane_count;
       lane_index += 2u) {
    loom_amdgpu_vector_fp8_pair_storage_t pair_storage;
    if (!loom_amdgpu_vector_fp8_query_storage_pair(plan, lane_index,
                                                   &pair_storage)) {
      return false;
    }
  }
  return true;
}

static bool loom_amdgpu_vector_fp8_plan_has_octet_storage(
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan) {
  if (plan->lane_count == 0 || (plan->lane_count & 7u) != 0) {
    return false;
  }
  for (uint32_t lane_index = 0; lane_index < plan->lane_count;
       lane_index += 8u) {
    loom_amdgpu_vector_fp8_octet_storage_t octet_storage;
    if (!loom_amdgpu_vector_fp8_query_storage_octet(plan, lane_index,
                                                    &octet_storage)) {
      return false;
    }
  }
  return true;
}

static bool loom_amdgpu_vector_fp8_descriptor_set_has_ref(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref) {
  return descriptor_ref != LOOM_AMDGPU_DESCRIPTOR_REF_NONE &&
         loom_amdgpu_descriptor_set_has_ref(descriptor_set, descriptor_ref);
}

static bool loom_amdgpu_vector_fp8_has_scalef32_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_scalar_type_t source_element_type,
    loom_scalar_type_t result_element_type) {
  loom_amdgpu_descriptor_ref_t descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  return loom_amdgpu_fp8_scalef32_descriptor_ref(
             source_element_type, result_element_type, &descriptor_ref) &&
         loom_amdgpu_vector_fp8_descriptor_set_has_ref(descriptor_set,
                                                       descriptor_ref);
}

static bool loom_amdgpu_vector_fp8_has_e8m0_pk8_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_scalar_type_t source_element_type,
    loom_scalar_type_t result_element_type) {
  loom_amdgpu_descriptor_ref_t descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  return loom_amdgpu_fp8_e8m0_pk8_descriptor_ref(
             source_element_type, result_element_type, &descriptor_ref) &&
         loom_amdgpu_vector_fp8_descriptor_set_has_ref(descriptor_set,
                                                       descriptor_ref);
}

static bool loom_amdgpu_vector_fp8_native_descriptor_set_refs(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_scalar_type_t source_element_type,
    loom_scalar_type_t result_element_type,
    loom_amdgpu_fp8_native_descriptor_refs_t* out_refs) {
  *out_refs = (loom_amdgpu_fp8_native_descriptor_refs_t){0};
  if (!loom_amdgpu_fp8_native_descriptor_refs(source_element_type,
                                              result_element_type, out_refs)) {
    return false;
  }
  out_refs->pair = loom_amdgpu_vector_fp8_descriptor_set_has_ref(descriptor_set,
                                                                 out_refs->pair)
                       ? out_refs->pair
                       : LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  out_refs->lane = loom_amdgpu_vector_fp8_descriptor_set_has_ref(descriptor_set,
                                                                 out_refs->lane)
                       ? out_refs->lane
                       : LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  return out_refs->pair != LOOM_AMDGPU_DESCRIPTOR_REF_NONE ||
         out_refs->lane != LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
}

static loom_amdgpu_vector_fp8_conversion_capabilities_t
loom_amdgpu_vector_fp8_conversion_capabilities(
    loom_low_lower_context_t* context,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  loom_amdgpu_vector_fp8_conversion_capabilities_t capabilities =
      LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NONE;
  const bool has_pair_storage =
      loom_amdgpu_vector_fp8_plan_has_pair_storage(plan);
  if (has_pair_storage && loom_amdgpu_vector_fp8_has_scalef32_descriptor(
                              descriptor_set, plan->source_element_type,
                              plan->result_element_type)) {
    capabilities |=
        LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_SCALEF32_RESULT_PAIR;
  }
  if (has_pair_storage &&
      loom_amdgpu_vector_fp8_has_scalef32_descriptor(
          descriptor_set, plan->source_element_type, LOOM_SCALAR_TYPE_F32)) {
    capabilities |=
        LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_SCALEF32_F32_PAIR;
  }
  if (loom_amdgpu_vector_fp8_plan_has_octet_storage(plan) &&
      loom_amdgpu_vector_fp8_has_e8m0_pk8_descriptor(
          descriptor_set, plan->source_element_type,
          plan->result_element_type)) {
    capabilities |=
        LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_E8M0_PK8_RESULT;
  }
  loom_amdgpu_fp8_native_descriptor_refs_t native_refs = {0};
  if (loom_amdgpu_vector_fp8_native_descriptor_set_refs(
          descriptor_set, plan->source_element_type, LOOM_SCALAR_TYPE_F32,
          &native_refs)) {
    if (has_pair_storage &&
        native_refs.pair != LOOM_AMDGPU_DESCRIPTOR_REF_NONE) {
      capabilities |=
          LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NATIVE_F32_PAIR;
    }
    if (native_refs.lane != LOOM_AMDGPU_DESCRIPTOR_REF_NONE) {
      capabilities |=
          LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NATIVE_F32_LANE;
    }
  }
  if (has_pair_storage &&
      loom_amdgpu_vector_fp8_native_descriptor_set_refs(
          descriptor_set, plan->source_element_type, LOOM_SCALAR_TYPE_BF16,
          &native_refs) &&
      native_refs.pair != LOOM_AMDGPU_DESCRIPTOR_REF_NONE) {
    capabilities |=
        LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NATIVE_BF16_PAIR;
  }
  if (has_pair_storage &&
      loom_amdgpu_vector_fp8_native_descriptor_set_refs(
          descriptor_set, plan->source_element_type, LOOM_SCALAR_TYPE_F16,
          &native_refs) &&
      native_refs.pair != LOOM_AMDGPU_DESCRIPTOR_REF_NONE) {
    capabilities |=
        LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NATIVE_F16_PAIR;
  }
  if (loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_BF16_F32)) {
    capabilities |=
        LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NATIVE_BF16_PACK;
  }
  return capabilities;
}

static bool loom_amdgpu_vector_fp8_plan_packed_u16_decode_plan(
    loom_low_lower_context_t* context,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    loom_amdgpu_fp8_decode_plan_t* out_decode_plan,
    loom_amdgpu_fp8_decode_value_flags_t* out_value_flags) {
  loom_amdgpu_vector_fp8_decode_value_flag_cache_t value_flag_cache;
  loom_amdgpu_vector_fp8_decode_value_flag_cache_initialize(context, plan,
                                                            &value_flag_cache);
  loom_amdgpu_vector_fp8_pair_storage_t
      pair_storage[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {0};
  loom_amdgpu_fp8_decode_value_flags_t value_flags =
      LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NONE;
  if (!loom_amdgpu_vector_fp8_query_storage_pair_set(
          plan, &value_flag_cache, plan->result_register_count, pair_storage,
          &value_flags)) {
    return false;
  }
  loom_amdgpu_initialize_fp8_decode_plan_from_descriptor_set(
      loom_low_lower_context_descriptor_set(context), plan->source_element_type,
      out_decode_plan);
  *out_value_flags = value_flags;
  return true;
}

static bool loom_amdgpu_vector_fp8_try_packed_bf16_plan_key(
    loom_low_lower_context_t* context,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    loom_amdgpu_vector_fp8_packed_bf16_selection_flags_t selection_flags,
    iree_string_view_t* out_plan_key) {
  *out_plan_key = iree_string_view_empty();
  loom_amdgpu_fp8_decode_plan_t decode_plan;
  loom_amdgpu_fp8_decode_value_flags_t value_flags =
      LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NONE;
  if (!loom_amdgpu_vector_fp8_plan_packed_u16_decode_plan(
          context, plan, &decode_plan, &value_flags) ||
      !loom_amdgpu_can_emit_fp8_pair_to_packed_bf16(&decode_plan,
                                                    value_flags)) {
    return false;
  }
  if (iree_any_bit_set(
          selection_flags,
          LOOM_AMDGPU_VECTOR_FP8_PACKED_BF16_SELECTION_FLAG_REQUIRE_PREFERRED) &&
      !loom_amdgpu_fp8_prefers_packed_bf16_pair_decode(&decode_plan,
                                                       value_flags)) {
    return false;
  }
  const loom_amdgpu_fp8_packed_u16_repairs_t repairs =
      loom_amdgpu_fp8_pair_to_packed_bf16_repairs(&decode_plan, value_flags);
  *out_plan_key = loom_amdgpu_fp8_packed_bf16_repair_reason_key(repairs);
  return true;
}

static iree_string_view_t loom_amdgpu_vector_fp8_packed_bf16_plan_key(
    loom_low_lower_context_t* context,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan) {
  iree_string_view_t plan_key = iree_string_view_empty();
  if (!loom_amdgpu_vector_fp8_try_packed_bf16_plan_key(
          context, plan, LOOM_AMDGPU_VECTOR_FP8_PACKED_BF16_SELECTION_FLAG_NONE,
          &plan_key)) {
    return IREE_SV(
        "amdgpu.vector_16bit_float_conversion.strategy."
        "fp8_software_packed_bf16_decode");
  }
  return plan_key;
}

static iree_string_view_t loom_amdgpu_vector_fp8_packed_f16_plan_key(
    loom_low_lower_context_t* context,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan) {
  loom_amdgpu_fp8_decode_plan_t decode_plan;
  loom_amdgpu_fp8_decode_value_flags_t value_flags =
      LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NONE;
  if (!loom_amdgpu_vector_fp8_plan_packed_u16_decode_plan(
          context, plan, &decode_plan, &value_flags) ||
      !loom_amdgpu_can_emit_fp8_pair_to_packed_f16_finite(&decode_plan,
                                                          value_flags)) {
    return IREE_SV(
        "amdgpu.vector_16bit_float_conversion.strategy."
        "fp8_software_packed_f16_decode");
  }
  const loom_amdgpu_fp8_packed_u16_repairs_t repairs =
      loom_amdgpu_fp8_pair_to_packed_f16_repairs(&decode_plan, value_flags);
  return loom_amdgpu_fp8_packed_f16_repair_reason_key(repairs);
}

static iree_string_view_t loom_amdgpu_vector_fp8_e8m0_pk8_conversion_plan_key(
    loom_scalar_type_t result_element_type) {
  switch (result_element_type) {
    case LOOM_SCALAR_TYPE_F32:
      return IREE_SV(
          "amdgpu.vector_16bit_float_conversion.strategy.fp8_e8m0_pk8_f32");
    case LOOM_SCALAR_TYPE_BF16:
      return IREE_SV(
          "amdgpu.vector_16bit_float_conversion.strategy.fp8_e8m0_pk8_bf16");
    case LOOM_SCALAR_TYPE_F16:
      return IREE_SV(
          "amdgpu.vector_16bit_float_conversion.strategy.fp8_e8m0_pk8_f16");
    default:
      return iree_string_view_empty();
  }
}

static iree_string_view_t
loom_amdgpu_vector_fp8_scalef32_native_conversion_plan_key(
    loom_scalar_type_t result_element_type) {
  switch (result_element_type) {
    case LOOM_SCALAR_TYPE_F32:
      return IREE_SV(
          "amdgpu.vector_16bit_float_conversion.strategy."
          "fp8_scalef32_native_f32_pair");
    case LOOM_SCALAR_TYPE_BF16:
      return IREE_SV(
          "amdgpu.vector_16bit_float_conversion.strategy."
          "fp8_scalef32_native_bf16_pair");
    case LOOM_SCALAR_TYPE_F16:
      return IREE_SV(
          "amdgpu.vector_16bit_float_conversion.strategy."
          "fp8_scalef32_native_f16_pair");
    default:
      return iree_string_view_empty();
  }
}

static iree_string_view_t
loom_amdgpu_vector_fp8_scalef32_native_f32_lane_conversion_plan_key(
    loom_scalar_type_t result_element_type) {
  switch (result_element_type) {
    case LOOM_SCALAR_TYPE_F32:
      return IREE_SV(
          "amdgpu.vector_16bit_float_conversion.strategy."
          "fp8_scalef32_native_f32_lane");
    case LOOM_SCALAR_TYPE_BF16:
      return IREE_SV(
          "amdgpu.vector_16bit_float_conversion.strategy."
          "fp8_scalef32_native_f32_lane_manual_bf16_pack");
    case LOOM_SCALAR_TYPE_F16:
      return IREE_SV(
          "amdgpu.vector_16bit_float_conversion.strategy."
          "fp8_scalef32_native_f32_lane_manual_f16_pack");
    default:
      return iree_string_view_empty();
  }
}

static iree_string_view_t
loom_amdgpu_vector_fp8_scalef32_native_f32_pair_to_16bit_conversion_plan_key(
    loom_amdgpu_vector_fp8_conversion_capabilities_t capabilities,
    loom_scalar_type_t result_element_type) {
  switch (result_element_type) {
    case LOOM_SCALAR_TYPE_BF16:
      if (iree_any_bit_set(
              capabilities,
              LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NATIVE_BF16_PACK)) {
        return IREE_SV(
            "amdgpu.vector_16bit_float_conversion.strategy."
            "fp8_scalef32_native_f32_pair_native_bf16_pack");
      }
      return IREE_SV(
          "amdgpu.vector_16bit_float_conversion.strategy."
          "fp8_scalef32_native_f32_pair_manual_bf16_pack");
    case LOOM_SCALAR_TYPE_F16:
      return IREE_SV(
          "amdgpu.vector_16bit_float_conversion.strategy."
          "fp8_scalef32_native_f32_pair_manual_f16_pack");
    default:
      return iree_string_view_empty();
  }
}

static iree_string_view_t
loom_amdgpu_vector_fp8_scalef32_software_conversion_plan_key(
    loom_scalar_type_t result_element_type) {
  switch (result_element_type) {
    case LOOM_SCALAR_TYPE_F32:
      return IREE_SV(
          "amdgpu.vector_16bit_float_conversion.strategy."
          "fp8_scalef32_software_f32_decode");
    case LOOM_SCALAR_TYPE_BF16:
      return IREE_SV(
          "amdgpu.vector_16bit_float_conversion.strategy."
          "fp8_scalef32_software_packed_bf16_decode");
    case LOOM_SCALAR_TYPE_F16:
      return IREE_SV(
          "amdgpu.vector_16bit_float_conversion.strategy."
          "fp8_scalef32_software_packed_f16_decode");
    default:
      return iree_string_view_empty();
  }
}

static iree_string_view_t
loom_amdgpu_vector_fp8_unscaled_f32_conversion_plan_key(
    loom_amdgpu_vector_fp8_conversion_capabilities_t capabilities) {
  if (iree_any_bit_set(
          capabilities,
          LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NATIVE_F32_PAIR)) {
    return IREE_SV(
        "amdgpu.vector_16bit_float_conversion.strategy.fp8_native_f32_pair");
  }
  if (iree_any_bit_set(
          capabilities,
          LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NATIVE_F32_LANE)) {
    return IREE_SV(
        "amdgpu.vector_16bit_float_conversion.strategy.fp8_native_f32_lane");
  }
  return IREE_SV(
      "amdgpu.vector_16bit_float_conversion.strategy.fp8_software_f32_decode");
}

static iree_string_view_t
loom_amdgpu_vector_fp8_unscaled_bf16_conversion_plan_key(
    loom_low_lower_context_t* context,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    loom_amdgpu_vector_fp8_conversion_capabilities_t capabilities) {
  if (iree_any_bit_set(
          capabilities,
          LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NATIVE_BF16_PAIR)) {
    return IREE_SV(
        "amdgpu.vector_16bit_float_conversion.strategy.fp8_native_bf16_pair");
  }
  if (iree_any_bit_set(
          capabilities,
          LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NATIVE_F32_PAIR |
              LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NATIVE_F32_LANE)) {
    iree_string_view_t preferred_plan_key = iree_string_view_empty();
    if (loom_amdgpu_vector_fp8_try_packed_bf16_plan_key(
            context, plan,
            LOOM_AMDGPU_VECTOR_FP8_PACKED_BF16_SELECTION_FLAG_REQUIRE_PREFERRED,
            &preferred_plan_key)) {
      return preferred_plan_key;
    }
  }
  if (iree_any_bit_set(
          capabilities,
          LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NATIVE_F32_PAIR)) {
    if (iree_any_bit_set(
            capabilities,
            LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NATIVE_BF16_PACK)) {
      return IREE_SV(
          "amdgpu.vector_16bit_float_conversion.strategy."
          "fp8_native_f32_pair_native_bf16_pack");
    }
    return IREE_SV(
        "amdgpu.vector_16bit_float_conversion.strategy."
        "fp8_native_f32_pair_manual_bf16_pack");
  }
  if (iree_any_bit_set(
          capabilities,
          LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NATIVE_F32_LANE)) {
    return IREE_SV(
        "amdgpu.vector_16bit_float_conversion.strategy."
        "fp8_native_f32_lane_manual_bf16_pack");
  }
  return loom_amdgpu_vector_fp8_packed_bf16_plan_key(context, plan);
}

static iree_string_view_t
loom_amdgpu_vector_fp8_unscaled_f16_conversion_plan_key(
    loom_low_lower_context_t* context,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    loom_amdgpu_vector_fp8_conversion_capabilities_t capabilities) {
  if (iree_any_bit_set(
          capabilities,
          LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NATIVE_F16_PAIR)) {
    return IREE_SV(
        "amdgpu.vector_16bit_float_conversion.strategy.fp8_native_f16_pair");
  }
  if (iree_any_bit_set(
          capabilities,
          LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NATIVE_F32_PAIR)) {
    return IREE_SV(
        "amdgpu.vector_16bit_float_conversion.strategy."
        "fp8_native_f32_pair_manual_f16_pack");
  }
  return loom_amdgpu_vector_fp8_packed_f16_plan_key(context, plan);
}

static iree_string_view_t loom_amdgpu_vector_fp8_conversion_plan_key(
    loom_low_lower_context_t* context,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan) {
  const loom_amdgpu_vector_16bit_float_conversion_plan_t* fp8_plan = plan;
  loom_amdgpu_vector_16bit_float_conversion_plan_t unscaled_plan;
  if (loom_amdgpu_vector_fp8_scalef32_is_identity(context, plan)) {
    loom_amdgpu_vector_fp8_unscaled_plan(plan, &unscaled_plan);
    fp8_plan = &unscaled_plan;
  }
  const loom_amdgpu_vector_fp8_conversion_capabilities_t capabilities =
      loom_amdgpu_vector_fp8_conversion_capabilities(context, fp8_plan);
  if (loom_amdgpu_vector_16bit_float_conversion_plan_has_e8m0_scale(fp8_plan)) {
    if (iree_any_bit_set(
            capabilities,
            LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_E8M0_PK8_RESULT)) {
      return loom_amdgpu_vector_fp8_e8m0_pk8_conversion_plan_key(
          fp8_plan->result_element_type);
    }
    IREE_ASSERT_UNREACHABLE(
        "accepted AMDGPU E8M0 pk8 FP8 vector decode did not cover all lanes");
    IREE_BUILTIN_UNREACHABLE();
  }
  if (loom_amdgpu_vector_16bit_float_conversion_plan_has_f32_scale(fp8_plan)) {
    if (iree_any_bit_set(
            capabilities,
            LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_SCALEF32_RESULT_PAIR)) {
      return loom_amdgpu_vector_fp8_scalef32_native_conversion_plan_key(
          fp8_plan->result_element_type);
    }
    const bool has_native_f32_pair = iree_any_bit_set(
        capabilities,
        LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NATIVE_F32_PAIR);
    const bool has_native_f32_lane =
        !has_native_f32_pair &&
        iree_any_bit_set(
            capabilities,
            LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NATIVE_F32_LANE);
    if (fp8_plan->result_element_type == LOOM_SCALAR_TYPE_F32) {
      if (has_native_f32_pair) {
        return loom_amdgpu_vector_fp8_scalef32_native_conversion_plan_key(
            fp8_plan->result_element_type);
      }
      if (has_native_f32_lane) {
        return loom_amdgpu_vector_fp8_scalef32_native_f32_lane_conversion_plan_key(
            fp8_plan->result_element_type);
      }
    }
    if ((fp8_plan->result_element_type == LOOM_SCALAR_TYPE_BF16 ||
         fp8_plan->result_element_type == LOOM_SCALAR_TYPE_F16) &&
        (iree_any_bit_set(
             capabilities,
             LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_SCALEF32_F32_PAIR) ||
         has_native_f32_pair)) {
      return loom_amdgpu_vector_fp8_scalef32_native_f32_pair_to_16bit_conversion_plan_key(
          capabilities, fp8_plan->result_element_type);
    }
    if ((fp8_plan->result_element_type == LOOM_SCALAR_TYPE_BF16 ||
         fp8_plan->result_element_type == LOOM_SCALAR_TYPE_F16) &&
        has_native_f32_lane) {
      return loom_amdgpu_vector_fp8_scalef32_native_f32_lane_conversion_plan_key(
          fp8_plan->result_element_type);
    }
    return loom_amdgpu_vector_fp8_scalef32_software_conversion_plan_key(
        fp8_plan->result_element_type);
  }
  switch (fp8_plan->result_element_type) {
    case LOOM_SCALAR_TYPE_F32:
      return loom_amdgpu_vector_fp8_unscaled_f32_conversion_plan_key(
          capabilities);
    case LOOM_SCALAR_TYPE_BF16:
      return loom_amdgpu_vector_fp8_unscaled_bf16_conversion_plan_key(
          context, fp8_plan, capabilities);
    case LOOM_SCALAR_TYPE_F16:
      return loom_amdgpu_vector_fp8_unscaled_f16_conversion_plan_key(
          context, fp8_plan, capabilities);
    default:
      return iree_string_view_empty();
  }
}

iree_string_view_t loom_amdgpu_vector_16bit_float_conversion_plan_key(
    loom_low_lower_context_t* context,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan) {
  if (loom_amdgpu_scalar_type_is_fp8(plan->source_element_type)) {
    return loom_amdgpu_vector_fp8_conversion_plan_key(context, plan);
  }
  if (plan->source_element_type == LOOM_SCALAR_TYPE_F32) {
    switch (plan->result_element_type) {
      case LOOM_SCALAR_TYPE_BF16:
        return loom_amdgpu_descriptor_set_has_ref(
                   loom_low_lower_context_descriptor_set(context),
                   LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_BF16_F32)
                   ? IREE_SV(
                         "amdgpu.vector_16bit_float_conversion.strategy."
                         "f32_to_packed_bf16_native")
                   : IREE_SV(
                         "amdgpu.vector_16bit_float_conversion.strategy."
                         "f32_to_packed_bf16_integer_pack");
      case LOOM_SCALAR_TYPE_F16:
        return IREE_SV(
            "amdgpu.vector_16bit_float_conversion.strategy.f32_to_packed_f16");
      default:
        return iree_string_view_empty();
    }
  }
  if (plan->source_element_type == LOOM_SCALAR_TYPE_F16 &&
      plan->result_element_type == LOOM_SCALAR_TYPE_F32) {
    return IREE_SV("amdgpu.vector_16bit_float_conversion.strategy.f16_to_f32");
  }
  if (plan->source_element_type == LOOM_SCALAR_TYPE_BF16 &&
      plan->result_element_type == LOOM_SCALAR_TYPE_F32) {
    return IREE_SV("amdgpu.vector_16bit_float_conversion.strategy.bf16_to_f32");
  }
  if (plan->kind == LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_FPTRUNC &&
      plan->source_element_type == plan->result_element_type) {
    switch (plan->result_element_type) {
      case LOOM_SCALAR_TYPE_BF16:
        return IREE_SV(
            "amdgpu.vector_16bit_float_conversion.strategy."
            "packed_bf16_origin_copy");
      case LOOM_SCALAR_TYPE_F16:
        return IREE_SV(
            "amdgpu.vector_16bit_float_conversion.strategy."
            "packed_f16_origin_copy");
      default:
        return iree_string_view_empty();
    }
  }
  return iree_string_view_empty();
}

static iree_status_t loom_amdgpu_lower_vector_16bit_float_extf(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan) {
  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->storage_source, &low_source));

  const loom_module_t* module = loom_low_lower_context_module(context);
  loom_type_t source_lane_type =
      loom_amdgpu_low_register_lane_type(module, low_source);
  if (loom_type_kind(source_lane_type) == LOOM_TYPE_NONE) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_vgpr_type(context, &source_lane_type));
  }
  loom_type_t result_lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &result_lane_type));

  if (plan->source_element_type == LOOM_SCALAR_TYPE_F8E4M3 ||
      plan->source_element_type == LOOM_SCALAR_TYPE_F8E5M2) {
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* fp8_plan = plan;
    loom_amdgpu_vector_16bit_float_conversion_plan_t unscaled_plan;
    if (loom_amdgpu_vector_fp8_scalef32_is_identity(context, plan)) {
      loom_amdgpu_vector_fp8_unscaled_plan(plan, &unscaled_plan);
      fp8_plan = &unscaled_plan;
    }
    if (loom_amdgpu_vector_16bit_float_conversion_plan_has_f32_scale(
            fp8_plan)) {
      if (fp8_plan->result_element_type == LOOM_SCALAR_TYPE_F32) {
        return loom_amdgpu_lower_vector_fp8_scalef32_to_f32(
            context, source_op, fp8_plan, low_source, source_lane_type,
            result_lane_type);
      }
      return loom_amdgpu_lower_vector_fp8_scalef32_to_packed_16bit(
          context, source_op, fp8_plan, low_source, source_lane_type,
          result_lane_type);
    }
    if (fp8_plan->result_element_type == LOOM_SCALAR_TYPE_BF16) {
      return loom_amdgpu_lower_vector_fp8_to_packed_bf16(
          context, source_op, fp8_plan, low_source, source_lane_type,
          result_lane_type);
    }
    if (fp8_plan->result_element_type == LOOM_SCALAR_TYPE_F16) {
      return loom_amdgpu_lower_vector_fp8_to_packed_f16(
          context, source_op, fp8_plan, low_source, source_lane_type,
          result_lane_type);
    }
    return loom_amdgpu_lower_vector_fp8_to_f32(context, source_op, fp8_plan,
                                               low_source, source_lane_type,
                                               result_lane_type);
  }

  loom_value_id_t lanes[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < plan->lane_count; ++i) {
    const uint64_t storage_lane =
        (uint64_t)plan->storage_lane_offset +
        (uint64_t)i * (uint64_t)plan->storage_lane_stride;
    IREE_ASSERT_LE(storage_lane, UINT32_MAX);
    if (plan->source_element_type == LOOM_SCALAR_TYPE_BF16) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_extract_bf16_range_lane_as_f32_bits(
          context, source_op, low_source, plan->storage_register_count,
          (uint32_t)storage_lane, source_lane_type, result_lane_type,
          &lanes[i]));
    } else {
      loom_value_id_t half_lane = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_extract_f16_lane_as_low_bits(
          context, source_op, low_source, plan->storage_register_count,
          (uint32_t)storage_lane, source_lane_type, &half_lane));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_unary(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F32_F16,
          half_lane, result_lane_type, &lanes[i]));
    }
  }

  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             lanes, plan->lane_count);
}

static iree_status_t loom_amdgpu_lower_vector_f32_to_packed_f16(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t lane_type) {
  loom_value_id_t packed_registers[LOOM_AMDGPU_MAX_PACKED_16BIT_FLOAT_LANES];
  for (uint32_t register_index = 0;
       register_index < plan->result_register_count; ++register_index) {
    loom_value_id_t packed = LOOM_VALUE_ID_INVALID;
    const uint32_t lane_base = register_index * 2u;
    for (uint32_t register_lane = 0; register_lane < 2u; ++register_lane) {
      const uint32_t lane_index = lane_base + register_lane;
      if (lane_index >= plan->lane_count) {
        break;
      }
      loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
          context, source_op, low_source, plan->source_register_count,
          lane_index, source_lane_type, &source_lane));
      IREE_RETURN_IF_ERROR(loom_amdgpu_pack_f32_lane_to_f16_register(
          context, source_op, source_lane, register_lane, lane_type, &packed));
    }
    packed_registers[register_index] = packed;
  }

  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             packed_registers,
                                             plan->result_register_count);
}

static iree_status_t loom_amdgpu_lower_vector_f32_to_packed_bf16(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t lane_type) {
  const loom_amdgpu_bf16_pack_descriptors_t* descriptors = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_get_bf16_pack_descriptors(context, &descriptors));

  loom_value_id_t packed_registers[LOOM_AMDGPU_MAX_PACKED_16BIT_FLOAT_LANES];
  for (uint32_t register_index = 0;
       register_index < plan->result_register_count; ++register_index) {
    const uint32_t lane_base = register_index * 2u;
    loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
        context, source_op, low_source, plan->source_register_count, lane_base,
        source_lane_type, &source_lane));
    if (lane_base + 1u < plan->lane_count) {
      loom_value_id_t high_source_lane = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
          context, source_op, low_source, plan->source_register_count,
          lane_base + 1u, source_lane_type, &high_source_lane));
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_f32_pair_to_packed_bf16_with_descriptors(
              context, source_op, descriptors, source_lane, high_source_lane,
              lane_type, &packed_registers[register_index]));
    } else if (iree_any_bit_set(
                   descriptors->flags,
                   LOOM_AMDGPU_BF16_PACK_DESCRIPTOR_FLAG_HAS_NATIVE)) {
      loom_value_id_t zero_lane = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0,
          lane_type, &zero_lane));
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_f32_pair_to_packed_bf16_with_descriptors(
              context, source_op, descriptors, source_lane, zero_lane,
              lane_type, &packed_registers[register_index]));
    } else {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_f32_to_bf16_lane_with_descriptors(
          context, source_op, descriptors, source_lane, lane_type,
          &packed_registers[register_index]));
    }
  }

  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             packed_registers,
                                             plan->result_register_count);
}

static bool loom_amdgpu_vector_16bit_float_fptrunc_has_storage_origin(
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan) {
  return plan->kind == LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_FPTRUNC &&
         plan->source_element_type == plan->result_element_type &&
         (plan->result_element_type == LOOM_SCALAR_TYPE_BF16 ||
          plan->result_element_type == LOOM_SCALAR_TYPE_F16);
}

static iree_status_t
loom_amdgpu_lower_vector_16bit_float_fptrunc_from_storage_origin(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan) {
  (void)source_op;
  return loom_low_lower_bind_value_alias(context, plan->storage_source,
                                         plan->result);
}

static iree_status_t loom_amdgpu_lower_vector_16bit_float_fptrunc(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan) {
  if (plan->source_element_type == LOOM_SCALAR_TYPE_F8E4M3 ||
      plan->source_element_type == LOOM_SCALAR_TYPE_F8E5M2) {
    return loom_amdgpu_lower_vector_16bit_float_extf(context, source_op, plan);
  }
  if (loom_amdgpu_vector_16bit_float_fptrunc_has_storage_origin(plan)) {
    return loom_amdgpu_lower_vector_16bit_float_fptrunc_from_storage_origin(
        context, source_op, plan);
  }

  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->source, &low_source));

  const loom_module_t* module = loom_low_lower_context_module(context);
  loom_type_t source_lane_type =
      loom_amdgpu_low_register_lane_type(module, low_source);
  if (loom_type_kind(source_lane_type) == LOOM_TYPE_NONE) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_vgpr_type(context, &source_lane_type));
  }
  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));

  if (plan->result_element_type == LOOM_SCALAR_TYPE_BF16) {
    return loom_amdgpu_lower_vector_f32_to_packed_bf16(
        context, source_op, plan, low_source, source_lane_type, lane_type);
  }
  return loom_amdgpu_lower_vector_f32_to_packed_f16(
      context, source_op, plan, low_source, source_lane_type, lane_type);
}

iree_status_t loom_amdgpu_lower_vector_16bit_float_conversion(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan) {
  switch (plan->kind) {
    case LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_EXTF:
    case LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_DECODE:
      return loom_amdgpu_lower_vector_16bit_float_extf(context, source_op,
                                                       plan);
    case LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_FPTRUNC:
      return loom_amdgpu_lower_vector_16bit_float_fptrunc(context, source_op,
                                                          plan);
    default:
      IREE_ASSERT_UNREACHABLE("unknown 16-bit float conversion plan");
      IREE_BUILTIN_UNREACHABLE();
  }
}

iree_status_t loom_amdgpu_lower_index_cast(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_index_cast_plan_t* plan) {
  switch (plan->kind) {
    case LOOM_AMDGPU_INDEX_CAST_KIND_ALIAS:
      return loom_low_lower_bind_value_alias(context, plan->source,
                                             plan->result);
    case LOOM_AMDGPU_INDEX_CAST_KIND_PRESERVING_LOW_32: {
      IREE_ASSERT_EQ(plan->index_bitwidth, 32u);
      loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_low_lower_lookup_value(context, plan->source, &low_source));
      loom_type_t result_type = loom_type_none();
      IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
          context, source_op, plan->result, &result_type));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
          context, source_op, low_source, /*lane_offset=*/0, result_type,
          &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_INDEX_CAST_KIND_DIAGNOSTIC_REJECTED:
      return iree_ok_status();
    case LOOM_AMDGPU_INDEX_CAST_KIND_NONE:
      break;
  }
  IREE_ASSERT_UNREACHABLE("unknown AMDGPU index cast plan kind");
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_lookup_or_materialize_address_i64_operand(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, loom_amdgpu_address_i64_alu_kind_t kind,
    loom_value_id_t* out_low_value) {
  *out_low_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_value, &low_value));

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t low_type = loom_module_value_type(module, low_value);
  if (!loom_low_type_is_register(low_type)) {
    IREE_ASSERT_UNREACHABLE(
        "AMDGPU address i64 ALU plan selected non-register operand");
    IREE_BUILTIN_UNREACHABLE();
  }
  const uint32_t unit_count = loom_low_register_type_unit_count(low_type);

  if (loom_amdgpu_address_i64_alu_kind_uses_vgpr(kind)) {
    const bool is_vgpr = loom_amdgpu_low_type_is_register_class(
        context, low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
    if (is_vgpr && unit_count == 2) {
      *out_low_value = low_value;
      return iree_ok_status();
    }
    if (unit_count == 2) {
      return loom_amdgpu_materialize_low_vgpr_b32_registers(
          context, source_op, low_value, out_low_value);
    }
    if (unit_count == 1) {
      return loom_amdgpu_emit_vgpr64_from_u32(context, source_op, low_value,
                                              out_low_value);
    }
  } else {
    const bool is_sgpr = loom_amdgpu_low_type_is_register_class(
        context, low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
    if (is_sgpr && unit_count == 2) {
      *out_low_value = low_value;
      return iree_ok_status();
    }
    if (is_sgpr && unit_count == 1) {
      return loom_amdgpu_emit_sgpr64_from_u32(context, source_op, low_value,
                                              out_low_value);
    }
  }

  IREE_ASSERT_UNREACHABLE(
      "AMDGPU address i64 ALU plan selected incompatible register type");
  IREE_BUILTIN_UNREACHABLE();
}

static iree_status_t loom_amdgpu_i64_compare_operand_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, uint32_t lane_index, loom_type_t lane_type,
    loom_value_id_t* out_low_lane) {
  *out_low_lane = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_value, &low_source));

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t low_type = loom_module_value_type(module, low_source);
  if (!loom_low_type_is_register(low_type)) {
    IREE_ASSERT_UNREACHABLE(
        "AMDGPU i64 compare plan selected non-register operand");
    IREE_BUILTIN_UNREACHABLE();
  }
  const uint32_t unit_count = loom_low_register_type_unit_count(low_type);
  if (unit_count == 1 && lane_index == 1) {
    return loom_amdgpu_emit_const_u32(context, source_op,
                                      LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0,
                                      lane_type, out_low_lane);
  }
  if (unit_count != 1 && unit_count != 2) {
    IREE_ASSERT_UNREACHABLE(
        "AMDGPU i64 compare plan selected wrong operand register count");
    IREE_BUILTIN_UNREACHABLE();
  }

  const loom_type_t source_lane_type =
      loom_low_register_type_with_unit_count(low_type, 1);
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
      context, source_op, low_source, unit_count, lane_index, source_lane_type,
      out_low_lane));
  return loom_amdgpu_materialize_low_vgpr_b32(context, source_op, *out_low_lane,
                                              out_low_lane);
}

static iree_status_t loom_amdgpu_emit_i64_compare_mask(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t mask_type, loom_value_id_t* out_mask) {
  *out_mask = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {lhs, rhs};
  loom_op_t* compare_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, descriptor_ref, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), &mask_type, 1, &compare_op));
  *out_mask = loom_value_slice_get(loom_low_op_results(compare_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_i64_compare_mask_combine(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t mask_type, loom_value_id_t* out_mask) {
  *out_mask = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {lhs, rhs};
  loom_op_t* combine_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, descriptor_ref, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), &mask_type, 1, &combine_op));
  *out_mask = loom_value_slice_get(loom_low_op_results(combine_op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lower_i64_compare(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_i64_compare_plan_t* plan) {
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  loom_type_t mask_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &mask_type));

  loom_value_id_t lhs_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_i64_compare_operand_lane(
      context, source_op, plan->lhs, 0, vgpr_type, &lhs_lo));
  loom_value_id_t lhs_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_i64_compare_operand_lane(
      context, source_op, plan->lhs, 1, vgpr_type, &lhs_hi));
  loom_value_id_t rhs_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_i64_compare_operand_lane(
      context, source_op, plan->rhs, 0, vgpr_type, &rhs_lo));
  loom_value_id_t rhs_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_i64_compare_operand_lane(
      context, source_op, plan->rhs, 1, vgpr_type, &rhs_hi));

  loom_value_id_t high_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_i64_compare_mask(
      context, source_op, plan->high_descriptor_ref, lhs_hi, rhs_hi, mask_type,
      &high_mask));
  loom_value_id_t low_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_i64_compare_mask(
      context, source_op, plan->low_descriptor_ref, lhs_lo, rhs_lo, mask_type,
      &low_mask));

  loom_value_id_t combined_low_mask = low_mask;
  if (plan->needs_high_equal) {
    loom_value_id_t high_equal_mask = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_i64_compare_mask(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32, lhs_hi,
        rhs_hi, mask_type, &high_equal_mask));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_i64_compare_mask_combine(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64,
        high_equal_mask, low_mask, mask_type, &combined_low_mask));
  }

  loom_value_id_t result_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_i64_compare_mask_combine(
      context, source_op, plan->combine_descriptor_ref, high_mask,
      combined_low_mask, mask_type, &result_mask));
  return loom_low_lower_bind_value(context, plan->result, result_mask);
}

static iree_status_t loom_amdgpu_extract_low_32_bits_as_vgpr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source, loom_value_id_t* out_low_source) {
  *out_low_source = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_source_pair = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source, &low_source_pair));
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t source_lane_type =
      loom_amdgpu_low_register_lane_type(module, low_source_pair);
  if (loom_type_kind(source_lane_type) == LOOM_TYPE_NONE) {
    IREE_ASSERT_UNREACHABLE(
        "AMDGPU scalar source lowered to a non-register type");
    IREE_BUILTIN_UNREACHABLE();
  }
  loom_value_id_t low_source_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, low_source_pair, /*lane_offset=*/0, source_lane_type,
      &low_source_lane));
  return loom_amdgpu_materialize_low_vgpr_b32_registers(
      context, source_op, low_source_lane, out_low_source);
}

iree_status_t loom_amdgpu_lower_address_i64_alu(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_address_i64_alu_plan_t* plan) {
  switch (plan->kind) {
    case LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_SGPR_ADD: {
      loom_value_id_t low_lhs = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_lookup_or_materialize_address_i64_operand(
              context, source_op, plan->lhs, plan->kind, &low_lhs));
      loom_value_id_t low_rhs = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_lookup_or_materialize_address_i64_operand(
              context, source_op, plan->rhs, plan->kind, &low_rhs));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr64_add(
          context, source_op, low_lhs, low_rhs, &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_ADD: {
      loom_value_id_t low_lhs = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_lookup_or_materialize_address_i64_operand(
              context, source_op, plan->lhs, plan->kind, &low_lhs));
      loom_value_id_t low_rhs = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_lookup_or_materialize_address_i64_operand(
              context, source_op, plan->rhs, plan->kind, &low_rhs));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr64_add(
          context, source_op, low_lhs, low_rhs, &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_SUB: {
      loom_value_id_t low_lhs = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_lookup_or_materialize_address_i64_operand(
              context, source_op, plan->lhs, plan->kind, &low_lhs));
      loom_value_id_t low_rhs = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_lookup_or_materialize_address_i64_operand(
              context, source_op, plan->rhs, plan->kind, &low_rhs));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr64_sub(
          context, source_op, low_lhs, low_rhs, &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_MUL_LO: {
      loom_value_id_t low_lhs = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_lookup_or_materialize_address_i64_operand(
              context, source_op, plan->lhs, plan->kind, &low_lhs));
      loom_value_id_t low_rhs = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_lookup_or_materialize_address_i64_operand(
              context, source_op, plan->rhs, plan->kind, &low_rhs));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr64_mul_lo(
          context, source_op, low_lhs, low_rhs, &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_MADD_LO: {
      loom_value_id_t low_lhs = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_lookup_or_materialize_address_i64_operand(
              context, source_op, plan->lhs, plan->kind, &low_lhs));
      loom_value_id_t low_rhs = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_lookup_or_materialize_address_i64_operand(
              context, source_op, plan->rhs, plan->kind, &low_rhs));
      loom_value_id_t product = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr64_mul_lo(
          context, source_op, low_lhs, low_rhs, &product));
      loom_value_id_t low_addend = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_lookup_or_materialize_address_i64_operand(
              context, source_op, plan->addend, plan->kind, &low_addend));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr64_add(
          context, source_op, product, low_addend, &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_SHL: {
      loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_lookup_or_materialize_address_i64_operand(
              context, source_op, plan->lhs, plan->kind, &low_value));
      loom_value_id_t low_shift = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_32_bits_as_vgpr(
          context, source_op, plan->rhs, &low_shift));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr64_shl(
          context, source_op, low_value, low_shift, &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_NONE:
      break;
  }
  IREE_ASSERT_UNREACHABLE("unknown AMDGPU address i64 ALU plan kind");
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lower_scalar_i64_alu(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_scalar_i64_alu_plan_t* plan) {
  switch (plan->kind) {
    case LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_ADD: {
      loom_value_id_t low_lhs = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i64(
          context, source_op, plan->lhs, &low_lhs));
      loom_value_id_t low_rhs = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i64(
          context, source_op, plan->rhs, &low_rhs));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr64_add(
          context, source_op, low_lhs, low_rhs, &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_SUB: {
      loom_value_id_t low_lhs = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i64(
          context, source_op, plan->lhs, &low_lhs));
      loom_value_id_t low_rhs = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i64(
          context, source_op, plan->rhs, &low_rhs));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr64_sub(
          context, source_op, low_lhs, low_rhs, &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_MUL_LO: {
      loom_value_id_t low_lhs = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i64(
          context, source_op, plan->lhs, &low_lhs));
      loom_value_id_t low_rhs = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i64(
          context, source_op, plan->rhs, &low_rhs));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr64_mul_lo(
          context, source_op, low_lhs, low_rhs, &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_SHL: {
      loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i64(
          context, source_op, plan->lhs, &low_value));
      loom_value_id_t low_shift = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_32_bits_as_vgpr(
          context, source_op, plan->rhs, &low_shift));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr64_shl(
          context, source_op, low_value, low_shift, &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_SCALAR_I64_ALU_KIND_NONE:
      break;
  }
  IREE_ASSERT_UNREACHABLE("unknown AMDGPU scalar i64 ALU plan kind");
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_vgpr_zero_extend(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint32_t source_bit_count,
    loom_value_id_t* out_low_result) {
  *out_low_result = low_source;
  IREE_ASSERT(source_bit_count == 8 || source_bit_count == 16 ||
              source_bit_count == 32);
  if (source_bit_count >= 32) {
    return iree_ok_status();
  }
  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  return loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, low_source,
      loom_amdgpu_integer_bit_mask(source_bit_count), lane_type,
      out_low_result);
}

static iree_status_t loom_amdgpu_lookup_scalar_conversion_source_for_result(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source, loom_value_id_t result,
    loom_value_id_t* out_low_source) {
  *out_low_source = LOOM_VALUE_ID_INVALID;
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_low_result_type(context, source_op, result, &result_type));
  if (!loom_low_type_is_register(result_type) ||
      loom_low_register_type_unit_count(result_type) != 2) {
    IREE_ASSERT_UNREACHABLE(
        "AMDGPU scalar conversion selected a non-register i64 result");
    IREE_BUILTIN_UNREACHABLE();
  }
  const bool result_is_vgpr = loom_amdgpu_low_type_is_register_class(
      context, result_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
  if (result_is_vgpr) {
    return loom_amdgpu_lookup_or_materialize_vgpr_i32(context, source_op,
                                                      source, out_low_source);
  }

  const bool result_is_sgpr = loom_amdgpu_low_type_is_register_class(
      context, result_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
  if (!result_is_sgpr) {
    IREE_ASSERT_UNREACHABLE(
        "AMDGPU scalar conversion selected an unsupported i64 register class");
    IREE_BUILTIN_UNREACHABLE();
  }

  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source, out_low_source));
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t source_type =
      loom_module_value_type(module, *out_low_source);
  const bool source_is_sgpr = loom_amdgpu_low_type_is_register_class(
      context, source_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
  if (!source_is_sgpr || loom_low_register_type_unit_count(source_type) != 1) {
    IREE_ASSERT_UNREACHABLE(
        "AMDGPU scalar conversion selected an SGPR i64 result from a "
        "non-SGPR source");
    IREE_BUILTIN_UNREACHABLE();
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_bind_register64_lanes(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_result, loom_value_id_t low_bits,
    loom_value_id_t high_bits) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t low_type = loom_module_value_type(module, low_bits);
  const loom_type_t high_type = loom_module_value_type(module, high_bits);
  if (!loom_low_type_is_register(low_type) ||
      loom_low_register_type_unit_count(low_type) != 1 ||
      !loom_type_equal(low_type, high_type)) {
    IREE_ASSERT_UNREACHABLE(
        "AMDGPU scalar conversion produced incompatible i64 lanes");
    IREE_BUILTIN_UNREACHABLE();
  }
  const loom_type_t result_type =
      loom_low_register_type_with_unit_count(low_type, 2);
  const loom_value_id_t lanes[] = {
      low_bits,
      high_bits,
  };
  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_low_register_range(
      context, source_op, lanes, IREE_ARRAYSIZE(lanes), result_type,
      &low_result));
  return loom_low_lower_bind_value(context, source_result, low_result);
}

static iree_status_t loom_amdgpu_bind_sign_extended_i64(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_result, loom_value_id_t low_source) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t lane_type = loom_module_value_type(module, low_source);
  const bool lane_is_vgpr = loom_amdgpu_low_type_is_register_class(
      context, lane_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
  loom_value_id_t high_bits = LOOM_VALUE_ID_INVALID;
  if (lane_is_vgpr) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT,
        /*shift=*/31, low_source, lane_type, &high_bits));
  } else {
    const bool lane_is_sgpr = loom_amdgpu_low_type_is_register_class(
        context, lane_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
    if (!lane_is_sgpr) {
      IREE_ASSERT_UNREACHABLE(
          "AMDGPU scalar conversion sign-extended a non-register source");
      IREE_BUILTIN_UNREACHABLE();
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_binary_immediate(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_ASHR_I32, low_source,
        /*immediate=*/31, lane_type, &high_bits));
  }
  return loom_amdgpu_bind_register64_lanes(context, source_op, source_result,
                                           low_source, high_bits);
}

static iree_status_t loom_amdgpu_bind_zero_extended_i64(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_result, loom_value_id_t low_source) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t lane_type = loom_module_value_type(module, low_source);
  const bool lane_is_vgpr = loom_amdgpu_low_type_is_register_class(
      context, lane_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
  loom_amdgpu_descriptor_ref_t zero_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32;
  if (!lane_is_vgpr) {
    const bool lane_is_sgpr = loom_amdgpu_low_type_is_register_class(
        context, lane_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
    if (!lane_is_sgpr) {
      IREE_ASSERT_UNREACHABLE(
          "AMDGPU scalar conversion zero-extended a non-register source");
      IREE_BUILTIN_UNREACHABLE();
    }
    zero_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32;
  }
  loom_value_id_t high_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, zero_descriptor_ref, 0, lane_type, &high_bits));
  return loom_amdgpu_bind_register64_lanes(context, source_op, source_result,
                                           low_source, high_bits);
}

static loom_amdgpu_fp8_decode_value_flags_t
loom_amdgpu_scalar_fp8_decode_value_flags(loom_low_lower_context_t* context,
                                          loom_value_id_t source) {
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  if (fact_table == NULL) {
    return LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NONE;
  }
  return loom_amdgpu_fp8_decode_value_flags_from_facts(
      loom_value_fact_table_lookup(fact_table, source));
}

static iree_status_t loom_amdgpu_emit_scalar_fp8_to_bf16(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_scalar_conversion_plan_t* plan) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_scalar_type_t source_type = loom_amdgpu_scalar_type_or_none(
      loom_module_value_type(module, plan->source));
  IREE_ASSERT(source_type == LOOM_SCALAR_TYPE_F8E4M3 ||
              source_type == LOOM_SCALAR_TYPE_F8E5M2);

  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i32(
      context, source_op, plan->source, &low_source));
  loom_value_id_t low_byte = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_zero_extend(
      context, source_op, low_source, 8, &low_byte));

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  loom_type_t mask_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &mask_type));
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  const loom_amdgpu_fp8_decode_plan_t* decode_plan = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_get_fp8_decode_plan(context, source_type, &decode_plan));

  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_to_bf16_lane(
      context, source_op, decode_plan, low_byte,
      loom_amdgpu_scalar_fp8_decode_value_flags(context, plan->source),
      vgpr_type, sgpr_type, mask_type, &low_result));
  return loom_low_lower_bind_value(context, plan->result, low_result);
}

iree_status_t loom_amdgpu_lower_scalar_conversion(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_scalar_conversion_plan_t* plan) {
  switch (plan->kind) {
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_ALIAS:
      return loom_low_lower_bind_value_alias(context, plan->source,
                                             plan->result);
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_TRUNCATE_LOW_32: {
      loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_low_lower_lookup_value(context, plan->source, &low_source));
      loom_type_t result_type = loom_type_none();
      IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
          context, source_op, plan->result, &result_type));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
          context, source_op, low_source, /*lane_offset=*/0, result_type,
          &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_SIGN_EXTEND_NARROW: {
      loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i32(
          context, source_op, plan->source, &low_source));
      loom_type_t lane_type = loom_type_none();
      IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_extract_vgpr_bitfield(
          context, source_op, low_source, /*bit_offset=*/0,
          plan->result_bit_count, LOOM_AMDGPU_BITFIELD_EXTRACT_MODE_SIGN_EXTEND,
          lane_type, &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_SIGN_EXTEND_NARROW_LOW_32: {
      loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_32_bits_as_vgpr(
          context, source_op, plan->source, &low_source));
      loom_type_t lane_type = loom_type_none();
      IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_extract_vgpr_bitfield(
          context, source_op, low_source, /*bit_offset=*/0,
          plan->result_bit_count, LOOM_AMDGPU_BITFIELD_EXTRACT_MODE_SIGN_EXTEND,
          lane_type, &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_SIGN_EXTEND_I64: {
      loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_lookup_scalar_conversion_source_for_result(
              context, source_op, plan->source, plan->result, &low_source));
      return loom_amdgpu_bind_sign_extended_i64(context, source_op,
                                                plan->result, low_source);
    }
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_ZERO_EXTEND: {
      loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
      if (plan->result_bit_count == 64) {
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_lookup_scalar_conversion_source_for_result(
                context, source_op, plan->source, plan->result, &low_source));
      } else {
        IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i32(
            context, source_op, plan->source, &low_source));
      }
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_zero_extend(
          context, source_op, low_source, plan->source_bit_count, &low_result));
      if (plan->result_bit_count == 64) {
        return loom_amdgpu_bind_zero_extended_i64(context, source_op,
                                                  plan->result, low_result);
      }
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_UITOFP_NARROW_TO_F32: {
      loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i32(
          context, source_op, plan->source, &low_source));
      loom_value_id_t zero_extended_source = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_zero_extend(
          context, source_op, low_source, plan->source_bit_count,
          &zero_extended_source));
      loom_type_t result_type = loom_type_none();
      IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
          context, source_op, plan->result, &result_type));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_unary(
          context, source_op, plan->convert_descriptor_ref,
          zero_extended_source, result_type, &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FP8_TO_BF16:
      return loom_amdgpu_emit_scalar_fp8_to_bf16(context, source_op, plan);
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FPTOI_F32_TO_I32: {
      loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_f32(
          context, source_op, plan->source, &low_source));
      loom_type_t lane_type = loom_type_none();
      IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_unary(
          context, source_op, plan->convert_descriptor_ref, low_source,
          lane_type, &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FPTOI_F32_TO_NARROW: {
      loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_f32(
          context, source_op, plan->source, &low_source));
      loom_type_t lane_type = loom_type_none();
      IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
      loom_value_id_t converted_source = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_unary(
          context, source_op, plan->convert_descriptor_ref, low_source,
          lane_type, &converted_source));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_extract_vgpr_bitfield(
          context, source_op, converted_source, /*bit_offset=*/0,
          plan->result_bit_count, LOOM_AMDGPU_BITFIELD_EXTRACT_MODE_SIGN_EXTEND,
          lane_type, &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_NONE:
      break;
  }
  IREE_ASSERT_UNREACHABLE("invalid AMDGPU scalar conversion plan kind");
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lower_structural_value_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t plan) {
  switch (source_op->kind) {
    case LOOM_OP_VECTOR_IOTA:
      if (plan.target_data == NULL) {
        IREE_ASSERT_UNREACHABLE(
            "AMDGPU fact-only vector atomic offset reached emission");
        IREE_BUILTIN_UNREACHABLE();
      }
      return loom_amdgpu_lower_vector_iota(
          context, source_op,
          (const loom_amdgpu_vector_iota_plan_t*)plan.target_data);
    case LOOM_OP_VECTOR_FROM_ELEMENTS:
    case LOOM_OP_VECTOR_SPLAT:
      if (plan.target_data == NULL) {
        IREE_ASSERT_UNREACHABLE(
            "AMDGPU fact-only vector atomic offset reached emission");
        IREE_BUILTIN_UNREACHABLE();
      }
      return loom_amdgpu_lower_vector_from_elements(
          context, source_op,
          (const loom_amdgpu_vector_from_elements_plan_t*)plan.target_data);
    case LOOM_OP_VECTOR_INSERT:
      return loom_amdgpu_lower_vector_insert(
          context, source_op,
          (const loom_amdgpu_vector_insert_plan_t*)plan.target_data);
    default:
      IREE_ASSERT_UNREACHABLE("AMDGPU value plan selected unknown op kind");
      IREE_BUILTIN_UNREACHABLE();
  }
}
