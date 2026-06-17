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
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/contracts/arithmetic_lower_rules.h"
#include "loom/target/arch/amdgpu/error_catalog.h"
#include "loom/target/arch/amdgpu/lower/arithmetic.h"
#include "loom/target/arch/amdgpu/lower/bitpack.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/legality.h"
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

static bool loom_amdgpu_descriptor_refs_present(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_descriptor_ref_t* descriptor_refs,
    iree_host_size_t descriptor_ref_count) {
  for (iree_host_size_t i = 0; i < descriptor_ref_count; ++i) {
    if (descriptor_refs[i] == LOOM_AMDGPU_DESCRIPTOR_REF_NONE) {
      return true;
    }
    if (!loom_amdgpu_descriptor_set_has_ref(descriptor_set,
                                            descriptor_refs[i])) {
      return false;
    }
  }
  return true;
}

typedef struct loom_amdgpu_descriptor_requirement_t {
  // Constraint key reported when this descriptor ref is missing.
  iree_string_view_t constraint_key;
  // Descriptor ref required by the lowering strategy.
  loom_amdgpu_descriptor_ref_t descriptor_ref;
} loom_amdgpu_descriptor_requirement_t;

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

static bool loom_amdgpu_value_use_is_vector_atomic_offset(
    const loom_op_t* user_op, loom_value_id_t value_id) {
  if (loom_vector_atomic_reduce_isa(user_op)) {
    return loom_vector_atomic_reduce_offsets(user_op) == value_id;
  }
  if (loom_vector_atomic_reduce_mask_isa(user_op)) {
    return loom_vector_atomic_reduce_mask_offsets(user_op) == value_id;
  }
  if (loom_vector_atomic_rmw_isa(user_op)) {
    return loom_vector_atomic_rmw_offsets(user_op) == value_id;
  }
  if (loom_vector_atomic_rmw_mask_isa(user_op)) {
    return loom_vector_atomic_rmw_mask_offsets(user_op) == value_id;
  }
  if (loom_vector_atomic_cmpxchg_isa(user_op)) {
    return loom_vector_atomic_cmpxchg_offsets(user_op) == value_id;
  }
  return false;
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
    if (!loom_amdgpu_value_use_is_vector_atomic_offset(loom_use_user_op(*use),
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
  loom_value_id_t result = LOOM_VALUE_ID_INVALID;
  if (loom_vector_iota_isa(source_op)) {
    result = loom_vector_iota_result(source_op);
  } else if (loom_vector_from_elements_isa(source_op)) {
    result = loom_vector_from_elements_result(source_op);
  } else {
    return false;
  }
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

static uint32_t loom_amdgpu_scalar_type_integer_bit_count(
    loom_scalar_type_t scalar_type) {
  switch (scalar_type) {
    case LOOM_SCALAR_TYPE_I8:
      return 8;
    case LOOM_SCALAR_TYPE_I16:
      return 16;
    case LOOM_SCALAR_TYPE_I32:
      return 32;
    case LOOM_SCALAR_TYPE_I64:
      return 64;
    default:
      return 0;
  }
}

static uint32_t loom_amdgpu_scalar_integer_bit_count(loom_type_t type) {
  const loom_scalar_type_t scalar_type = loom_amdgpu_scalar_type_or_none(type);
  if (scalar_type == LOOM_SCALAR_TYPE_COUNT_) {
    return 0;
  }
  return loom_amdgpu_scalar_type_integer_bit_count(scalar_type);
}

static bool loom_amdgpu_type_is_offset_scalar(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_OFFSET;
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

static bool loom_amdgpu_offset_add_needs_64bit(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_op_t* source_op) {
  const loom_value_id_t result = loom_index_add_result(source_op);
  const loom_type_t result_type = loom_module_value_type(module, result);
  if (!loom_amdgpu_type_is_offset_scalar(result_type)) {
    return false;
  }
  if (fact_table == NULL || result >= module->values.count) {
    return true;
  }
  return !loom_value_facts_fit_unsigned_bit_count(
      loom_value_fact_table_lookup(fact_table, result), 32);
}

static bool loom_amdgpu_offset_add_descriptors_supported(
    const loom_low_descriptor_set_t* descriptor_set, bool result_is_vgpr,
    iree_string_view_t* out_constraint_key) {
  return result_is_vgpr
             ? loom_amdgpu_descriptor_requirements_present(
                   descriptor_set, kAmdgpuOffsetAddVgprDescriptorRequirements,
                   IREE_ARRAYSIZE(kAmdgpuOffsetAddVgprDescriptorRequirements),
                   out_constraint_key)
             : loom_amdgpu_descriptor_requirements_present(
                   descriptor_set, kAmdgpuOffsetAddSgprDescriptorRequirements,
                   IREE_ARRAYSIZE(kAmdgpuOffsetAddSgprDescriptorRequirements),
                   out_constraint_key);
}

static bool loom_amdgpu_offset_cmp_needs_64bit(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_op_t* source_op) {
  const loom_value_id_t lhs = loom_index_cmp_lhs(source_op);
  const loom_value_id_t rhs = loom_index_cmp_rhs(source_op);
  if (!loom_amdgpu_type_is_offset_scalar(loom_module_value_type(module, lhs)) ||
      !loom_amdgpu_type_is_offset_scalar(loom_module_value_type(module, rhs))) {
    return false;
  }
  if (fact_table == NULL || lhs >= module->values.count ||
      rhs >= module->values.count) {
    return true;
  }
  return !loom_value_facts_fit_unsigned_bit_count(
             loom_value_fact_table_lookup(fact_table, lhs), 32) ||
         !loom_value_facts_fit_unsigned_bit_count(
             loom_value_fact_table_lookup(fact_table, rhs), 32);
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

static_assert((int)LOOM_INDEX_CMP_PREDICATE_EQ ==
                  (int)LOOM_SCALAR_CMPI_PREDICATE_EQ,
              "index and scalar integer predicate enums must align");
static_assert((int)LOOM_INDEX_CMP_PREDICATE_NE ==
                  (int)LOOM_SCALAR_CMPI_PREDICATE_NE,
              "index and scalar integer predicate enums must align");
static_assert((int)LOOM_INDEX_CMP_PREDICATE_SLT ==
                  (int)LOOM_SCALAR_CMPI_PREDICATE_SLT,
              "index and scalar integer predicate enums must align");
static_assert((int)LOOM_INDEX_CMP_PREDICATE_SLE ==
                  (int)LOOM_SCALAR_CMPI_PREDICATE_SLE,
              "index and scalar integer predicate enums must align");
static_assert((int)LOOM_INDEX_CMP_PREDICATE_SGT ==
                  (int)LOOM_SCALAR_CMPI_PREDICATE_SGT,
              "index and scalar integer predicate enums must align");
static_assert((int)LOOM_INDEX_CMP_PREDICATE_SGE ==
                  (int)LOOM_SCALAR_CMPI_PREDICATE_SGE,
              "index and scalar integer predicate enums must align");
static_assert((int)LOOM_INDEX_CMP_PREDICATE_ULT ==
                  (int)LOOM_SCALAR_CMPI_PREDICATE_ULT,
              "index and scalar integer predicate enums must align");
static_assert((int)LOOM_INDEX_CMP_PREDICATE_ULE ==
                  (int)LOOM_SCALAR_CMPI_PREDICATE_ULE,
              "index and scalar integer predicate enums must align");
static_assert((int)LOOM_INDEX_CMP_PREDICATE_UGT ==
                  (int)LOOM_SCALAR_CMPI_PREDICATE_UGT,
              "index and scalar integer predicate enums must align");
static_assert((int)LOOM_INDEX_CMP_PREDICATE_UGE ==
                  (int)LOOM_SCALAR_CMPI_PREDICATE_UGE,
              "index and scalar integer predicate enums must align");

static bool loom_amdgpu_i64_compare_predicate_descriptors(
    uint8_t predicate, loom_amdgpu_descriptor_ref_t* out_high_descriptor_ref,
    loom_amdgpu_descriptor_ref_t* out_low_descriptor_ref,
    loom_amdgpu_descriptor_ref_t* out_combine_descriptor_ref,
    bool* out_needs_high_equal) {
  *out_high_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  *out_low_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  *out_combine_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  *out_needs_high_equal = false;
  switch (predicate) {
    case LOOM_SCALAR_CMPI_PREDICATE_EQ:
      *out_high_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32;
      *out_low_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32;
      *out_combine_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64;
      return true;
    case LOOM_SCALAR_CMPI_PREDICATE_NE:
      *out_high_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_NE_I32;
      *out_low_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_NE_I32;
      *out_combine_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_OR_B64;
      return true;
    case LOOM_SCALAR_CMPI_PREDICATE_SLT:
      *out_high_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_SLT_I32;
      *out_low_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_ULT_U32;
      *out_combine_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_OR_B64;
      *out_needs_high_equal = true;
      return true;
    case LOOM_SCALAR_CMPI_PREDICATE_SLE:
      *out_high_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_SLT_I32;
      *out_low_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_ULE_U32;
      *out_combine_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_OR_B64;
      *out_needs_high_equal = true;
      return true;
    case LOOM_SCALAR_CMPI_PREDICATE_SGT:
      *out_high_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_SGT_I32;
      *out_low_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_UGT_U32;
      *out_combine_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_OR_B64;
      *out_needs_high_equal = true;
      return true;
    case LOOM_SCALAR_CMPI_PREDICATE_SGE:
      *out_high_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_SGT_I32;
      *out_low_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_UGE_U32;
      *out_combine_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_OR_B64;
      *out_needs_high_equal = true;
      return true;
    case LOOM_SCALAR_CMPI_PREDICATE_ULT:
      *out_high_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_ULT_U32;
      *out_low_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_ULT_U32;
      *out_combine_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_OR_B64;
      *out_needs_high_equal = true;
      return true;
    case LOOM_SCALAR_CMPI_PREDICATE_ULE:
      *out_high_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_ULT_U32;
      *out_low_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_ULE_U32;
      *out_combine_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_OR_B64;
      *out_needs_high_equal = true;
      return true;
    case LOOM_SCALAR_CMPI_PREDICATE_UGT:
      *out_high_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_UGT_U32;
      *out_low_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_UGT_U32;
      *out_combine_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_OR_B64;
      *out_needs_high_equal = true;
      return true;
    case LOOM_SCALAR_CMPI_PREDICATE_UGE:
      *out_high_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_UGT_U32;
      *out_low_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_UGE_U32;
      *out_combine_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_OR_B64;
      *out_needs_high_equal = true;
      return true;
    default:
      return false;
  }
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
  bool is_scc = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_is_register_class_count(
      context, result_low_type, LOOM_AMDGPU_REG_CLASS_ID_SCC, 1, &is_scc));
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

  bool is_native_mask = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_is_register_class_count(
      context, result_low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2,
      &is_native_mask));
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
  if (loom_amdgpu_value_is_16bit_float(context, result)) {
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

iree_status_t loom_amdgpu_low_legality_verify_offset_add(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  (void)provider;
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_target_low_legality_module(context);
  if (!loom_amdgpu_offset_add_needs_64bit(
          module, loom_target_low_legality_fact_table(context), op)) {
    return iree_ok_status();
  }
  *out_handled = true;

  const loom_view_region_table_t* view_regions = NULL;
  IREE_RETURN_IF_ERROR(
      loom_target_low_legality_view_regions(context, &view_regions));
  const bool result_is_vgpr = loom_amdgpu_source_value_prefers_vgpr(
      module, loom_target_low_legality_fact_table(context), view_regions,
      loom_index_add_result(op));
  iree_string_view_t constraint_key = iree_string_view_empty();
  if (loom_amdgpu_offset_add_descriptors_supported(
          loom_target_low_legality_descriptor_set(context), result_is_vgpr,
          &constraint_key)) {
    return iree_ok_status();
  }
  return loom_amdgpu_low_legality_reject(context, op, constraint_key);
}

iree_status_t loom_amdgpu_low_legality_verify_offset_compare(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  (void)provider;
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_target_low_legality_module(context);
  if (!loom_amdgpu_offset_cmp_needs_64bit(
          module, loom_target_low_legality_fact_table(context), op)) {
    return iree_ok_status();
  }

  const loom_view_region_table_t* view_regions = NULL;
  IREE_RETURN_IF_ERROR(
      loom_target_low_legality_view_regions(context, &view_regions));
  if (!loom_amdgpu_source_value_is_native_i1_mask(
          module, loom_target_low_legality_fact_table(context), view_regions,
          loom_index_cmp_result(op))) {
    return iree_ok_status();
  }
  *out_handled = true;

  loom_amdgpu_i64_compare_plan_t plan = {
      .lhs = loom_index_cmp_lhs(op),
      .rhs = loom_index_cmp_rhs(op),
      .result = loom_index_cmp_result(op),
  };
  if (!loom_amdgpu_i64_compare_predicate_descriptors(
          loom_index_cmp_predicate(op), &plan.high_descriptor_ref,
          &plan.low_descriptor_ref, &plan.combine_descriptor_ref,
          &plan.needs_high_equal)) {
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

  const loom_view_region_table_t* view_regions = NULL;
  IREE_RETURN_IF_ERROR(
      loom_target_low_legality_view_regions(context, &view_regions));
  if (!loom_amdgpu_source_value_is_native_i1_mask(
          module, loom_target_low_legality_fact_table(context), view_regions,
          loom_scalar_cmpi_result(op))) {
    return iree_ok_status();
  }
  *out_handled = true;

  loom_amdgpu_i64_compare_plan_t plan = {
      .lhs = loom_scalar_cmpi_lhs(op),
      .rhs = loom_scalar_cmpi_rhs(op),
      .result = loom_scalar_cmpi_result(op),
  };
  if (!loom_amdgpu_i64_compare_predicate_descriptors(
          loom_scalar_cmpi_predicate(op), &plan.high_descriptor_ref,
          &plan.low_descriptor_ref, &plan.combine_descriptor_ref,
          &plan.needs_high_equal)) {
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

static bool loom_amdgpu_select_vector_from_elements_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_from_elements_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_vector_from_elements_plan_t){0};
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_slice_t elements =
      loom_vector_from_elements_elements(source_op);
  if (elements.count == 0) {
    return false;
  }
  const loom_value_id_t result = loom_vector_from_elements_result(source_op);
  const loom_type_t result_type = loom_module_value_type(module, result);
  loom_amdgpu_vector_storage_t storage = {0};
  if (!loom_amdgpu_type_vector_storage(result_type, &storage) ||
      elements.count != storage.element_count ||
      elements.count > IREE_ARRAYSIZE(out_plan->payload.elements)) {
    return false;
  }
  for (uint32_t i = 0; i < elements.count; ++i) {
    const loom_value_id_t element = elements.values[i];
    const loom_type_t source_type = loom_module_value_type(module, element);
    if (!loom_type_is_scalar(source_type) ||
        loom_type_element_type(source_type) != storage.element_type) {
      return false;
    }
    if (storage.element_type == LOOM_SCALAR_TYPE_I32 &&
        !loom_amdgpu_value_can_materialize_as_vgpr_i32(context, element)) {
      return false;
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
  return true;
}

static bool loom_amdgpu_select_vector_splat_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_from_elements_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_vector_from_elements_plan_t){0};
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t result = loom_vector_splat_result(source_op);
  const loom_type_t result_type = loom_module_value_type(module, result);
  loom_amdgpu_vector_storage_t storage = {0};
  if (!loom_amdgpu_type_vector_storage(result_type, &storage) ||
      storage.element_count > IREE_ARRAYSIZE(out_plan->payload.elements)) {
    return false;
  }
  const loom_value_id_t scalar = loom_vector_splat_scalar(source_op);
  const loom_type_t scalar_type = loom_module_value_type(module, scalar);
  if (!loom_type_is_scalar(scalar_type) ||
      loom_type_element_type(scalar_type) != storage.element_type) {
    return false;
  }
  if (storage.element_type == LOOM_SCALAR_TYPE_I32 &&
      !loom_amdgpu_value_can_materialize_as_vgpr_i32(context, scalar)) {
    return false;
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
  return true;
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

  const loom_op_t* fmaf_op = loom_amdgpu_value_defining_op(module, fmaf_result);
  if (fmaf_op == NULL || !loom_scalar_fmaf_isa(fmaf_op) ||
      loom_scalar_fmaf_result(fmaf_op) != fmaf_result) {
    if (emit_diagnostics) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fma_mix_half_result_diagnostic(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
          destination_lane_index, result_half, source_kind_names,
          IREE_SV("rejected"), IREE_SV("rounding_source_not_fmaf")));
    }
    return iree_ok_status();
  }

  const loom_value_id_t operands[LOOM_AMDGPU_FMA_MIX_SOURCE_COUNT] = {
      loom_scalar_fmaf_a(fmaf_op),
      loom_scalar_fmaf_b(fmaf_op),
      loom_scalar_fmaf_c(fmaf_op),
  };
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
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(operands); ++i) {
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
    source_kind_names[i] = loom_amdgpu_fma_mix_source_kind_key(source_kinds[i]);
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
  const bool high_result = (inout_plan->lane_offset & 1u) != 0;
  if (!loom_amdgpu_select_fma_mix_half_result_descriptor(
          context, source_kinds, high_result, &descriptor_ref)) {
    if (emit_diagnostics) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fma_mix_half_result_diagnostic(
          context, source_op, descriptor_ref, destination_lane_index,
          result_half, source_kind_names, IREE_SV("rejected"),
          IREE_SV("descriptor_unavailable")));
    }
    return iree_ok_status();
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
  };
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
  if (element_type == LOOM_SCALAR_TYPE_I32 &&
      !loom_amdgpu_value_can_materialize_as_vgpr_i32(context, value)) {
    return iree_ok_status();
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

static bool loom_amdgpu_type_is_bf16_packed_vector(
    loom_type_t type, uint32_t* out_lane_count, uint32_t* out_register_count) {
  *out_lane_count = 0;
  *out_register_count = 0;
  if (loom_type_element_type(type) != LOOM_SCALAR_TYPE_BF16) {
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

static void loom_amdgpu_vector_bf16_conversion_plan_from_accepted_op(
    const loom_module_t* module, const loom_op_t* source_op,
    loom_amdgpu_vector_bf16_conversion_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_vector_bf16_conversion_plan_t){0};

  loom_value_id_t source = LOOM_VALUE_ID_INVALID;
  loom_value_id_t result = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_vector_bf16_conversion_kind_t kind =
      LOOM_AMDGPU_VECTOR_BF16_CONVERSION_KIND_NONE;
  switch (source_op->kind) {
    case LOOM_OP_VECTOR_EXTF:
      source = loom_vector_extf_input(source_op);
      result = loom_vector_extf_result(source_op);
      kind = LOOM_AMDGPU_VECTOR_BF16_CONVERSION_KIND_EXTF;
      break;
    case LOOM_OP_VECTOR_FPTRUNC:
      source = loom_vector_fptrunc_input(source_op);
      result = loom_vector_fptrunc_result(source_op);
      kind = LOOM_AMDGPU_VECTOR_BF16_CONVERSION_KIND_FPTRUNC;
      break;
    default:
      IREE_ASSERT_UNREACHABLE("accepted AMDGPU BF16 conversion has wrong op");
      IREE_BUILTIN_UNREACHABLE();
  }

  const loom_type_t source_type = loom_module_value_type(module, source);
  const loom_type_t result_type = loom_module_value_type(module, result);
  uint32_t source_lane_count = 0;
  uint32_t source_register_count = 0;
  uint32_t result_lane_count = 0;
  uint32_t result_register_count = 0;
  if (kind == LOOM_AMDGPU_VECTOR_BF16_CONVERSION_KIND_EXTF) {
    const bool source_matches = loom_amdgpu_type_is_bf16_packed_vector(
        source_type, &source_lane_count, &source_register_count);
    IREE_ASSERT(source_matches);
    result_lane_count = loom_amdgpu_vector_f32_register_count(result_type);
    result_register_count = result_lane_count;
  } else {
    source_lane_count = loom_amdgpu_vector_f32_register_count(source_type);
    source_register_count = source_lane_count;
    const bool result_matches = loom_amdgpu_type_is_bf16_packed_vector(
        result_type, &result_lane_count, &result_register_count);
    IREE_ASSERT(result_matches);
  }
  IREE_ASSERT_NE(source_lane_count, 0u);
  IREE_ASSERT_EQ(source_lane_count, result_lane_count);
  IREE_ASSERT_NE(source_register_count, 0u);
  IREE_ASSERT_NE(result_register_count, 0u);

  *out_plan = (loom_amdgpu_vector_bf16_conversion_plan_t){
      .kind = kind,
      .source = source,
      .result = result,
      .lane_count = source_lane_count,
      .source_register_count = source_register_count,
      .result_register_count = result_register_count,
  };
}

iree_status_t loom_amdgpu_select_vector_bf16_conversion_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_bf16_conversion_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_vector_bf16_conversion_plan_t){0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_select_arithmetic_contract(context, source_op, out_selected));
  if (*out_selected) {
    loom_amdgpu_vector_bf16_conversion_plan_from_accepted_op(
        loom_low_lower_context_module(context), source_op, out_plan);
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

iree_status_t loom_amdgpu_select_offset_add_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_offset_add_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_offset_add_plan_t){0};
  *out_selected = false;
  const loom_module_t* module = loom_low_lower_context_module(context);
  if (!loom_amdgpu_offset_add_needs_64bit(
          module, loom_low_lower_context_fact_table(context), source_op)) {
    return iree_ok_status();
  }

  const loom_value_id_t result = loom_index_add_result(source_op);
  loom_type_t result_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op, result,
                                                   &result_low_type));

  bool result_is_vgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_is_register_class_count(
      context, result_low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2,
      &result_is_vgpr));
  bool result_is_sgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_is_register_class_count(
      context, result_low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2,
      &result_is_sgpr));
  if (!result_is_vgpr && !result_is_sgpr) {
    return iree_ok_status();
  }

  iree_string_view_t constraint_key = iree_string_view_empty();
  if (!loom_amdgpu_offset_add_descriptors_supported(
          loom_low_lower_context_descriptor_set(context), result_is_vgpr,
          &constraint_key)) {
    return iree_ok_status();
  }

  *out_plan = (loom_amdgpu_offset_add_plan_t){
      .lhs = loom_index_add_lhs(source_op),
      .rhs = loom_index_add_rhs(source_op),
      .result = result,
      .result_is_vgpr = result_is_vgpr,
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
  if (!loom_amdgpu_offset_cmp_needs_64bit(
          module, loom_low_lower_context_fact_table(context), source_op)) {
    return iree_ok_status();
  }

  const loom_value_id_t result = loom_index_cmp_result(source_op);
  loom_type_t result_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op, result,
                                                   &result_low_type));
  bool result_is_native_mask = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_is_register_class_count(
      context, result_low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2,
      &result_is_native_mask));
  if (!result_is_native_mask) {
    return iree_ok_status();
  }

  loom_amdgpu_i64_compare_plan_t plan = {
      .lhs = loom_index_cmp_lhs(source_op),
      .rhs = loom_index_cmp_rhs(source_op),
      .result = result,
  };
  if (!loom_amdgpu_i64_compare_predicate_descriptors(
          loom_index_cmp_predicate(source_op), &plan.high_descriptor_ref,
          &plan.low_descriptor_ref, &plan.combine_descriptor_ref,
          &plan.needs_high_equal)) {
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
  bool is_vgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
      context, source_low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR, &is_vgpr));
  if (is_vgpr) {
    *out_can_lower = true;
    return iree_ok_status();
  }
  return loom_amdgpu_low_type_register_class_is(
      context, source_low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, out_can_lower);
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
  bool result_is_native_mask = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_is_register_class_count(
      context, result_low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2,
      &result_is_native_mask));
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
          loom_scalar_cmpi_predicate(source_op), &plan.high_descriptor_ref,
          &plan.low_descriptor_ref, &plan.combine_descriptor_ref,
          &plan.needs_high_equal)) {
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

static bool loom_amdgpu_scalar_i64_mul_descriptors_supported(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_string_view_t* out_constraint_key) {
  *out_constraint_key = IREE_SV("descriptor.v_mul_lo_u32");
  if (!loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_LO_U32)) {
    return false;
  }
  *out_constraint_key = IREE_SV("descriptor.v_mul_hi_u32");
  if (!loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_HI_U32)) {
    return false;
  }
  *out_constraint_key = IREE_SV("descriptor.v_add_u32");
  return loom_amdgpu_descriptor_set_has_ref(
      descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32);
}

static bool loom_amdgpu_scalar_i64_sub_descriptors_supported(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_string_view_t* out_constraint_key) {
  return loom_amdgpu_descriptor_requirements_present(
      descriptor_set, kAmdgpuScalarI64SubVgprDescriptorRequirements,
      IREE_ARRAYSIZE(kAmdgpuScalarI64SubVgprDescriptorRequirements),
      out_constraint_key);
}

static bool loom_amdgpu_scalar_i64_shl_descriptors_supported(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_string_view_t* out_constraint_key) {
  return loom_amdgpu_descriptor_requirements_present(
      descriptor_set, kAmdgpuScalarI64ShlVgprDescriptorRequirements,
      IREE_ARRAYSIZE(kAmdgpuScalarI64ShlVgprDescriptorRequirements),
      out_constraint_key);
}

static bool loom_amdgpu_scalar_i64_alu_descriptors_supported(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_scalar_i64_alu_kind_t kind,
    iree_string_view_t* out_constraint_key) {
  switch (kind) {
    case LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_ADD:
      return loom_amdgpu_offset_add_descriptors_supported(
          descriptor_set, /*result_is_vgpr=*/true, out_constraint_key);
    case LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_SUB:
      return loom_amdgpu_scalar_i64_sub_descriptors_supported(
          descriptor_set, out_constraint_key);
    case LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_MUL_LO:
      return loom_amdgpu_scalar_i64_mul_descriptors_supported(
          descriptor_set, out_constraint_key);
    case LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_SHL:
      return loom_amdgpu_scalar_i64_shl_descriptors_supported(
          descriptor_set, out_constraint_key);
    case LOOM_AMDGPU_SCALAR_I64_ALU_KIND_NONE:
      break;
  }
  *out_constraint_key = IREE_SV("operation.scalar_i64_alu");
  return false;
}

static bool loom_amdgpu_scalar_i64_alu_op(
    const loom_op_t* source_op, loom_amdgpu_scalar_i64_alu_kind_t* out_kind,
    loom_value_id_t* out_lhs, loom_value_id_t* out_rhs,
    loom_value_id_t* out_result) {
  *out_kind = LOOM_AMDGPU_SCALAR_I64_ALU_KIND_NONE;
  *out_lhs = LOOM_VALUE_ID_INVALID;
  *out_rhs = LOOM_VALUE_ID_INVALID;
  *out_result = LOOM_VALUE_ID_INVALID;
  if (loom_scalar_addi_isa(source_op)) {
    *out_kind = LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_ADD;
    *out_lhs = loom_scalar_addi_lhs(source_op);
    *out_rhs = loom_scalar_addi_rhs(source_op);
    *out_result = loom_scalar_addi_result(source_op);
    return true;
  }
  if (loom_scalar_subi_isa(source_op)) {
    *out_kind = LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_SUB;
    *out_lhs = loom_scalar_subi_lhs(source_op);
    *out_rhs = loom_scalar_subi_rhs(source_op);
    *out_result = loom_scalar_subi_result(source_op);
    return true;
  }
  if (loom_scalar_muli_isa(source_op)) {
    *out_kind = LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_MUL_LO;
    *out_lhs = loom_scalar_muli_lhs(source_op);
    *out_rhs = loom_scalar_muli_rhs(source_op);
    *out_result = loom_scalar_muli_result(source_op);
    return true;
  }
  if (loom_scalar_shli_isa(source_op)) {
    *out_kind = LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_SHL;
    *out_lhs = loom_scalar_shli_lhs(source_op);
    *out_rhs = loom_scalar_shli_rhs(source_op);
    *out_result = loom_scalar_shli_result(source_op);
    return true;
  }
  return false;
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
  bool is_vgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
      context, source_low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR, &is_vgpr));
  if (is_vgpr) {
    *out_can_lower = true;
    return iree_ok_status();
  }
  return loom_amdgpu_low_type_register_class_is(
      context, source_low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, out_can_lower);
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
  bool result_is_vgpr64 = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_is_register_class_count(
      context, result_low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2,
      &result_is_vgpr64));
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

  const loom_view_region_table_t* view_regions = NULL;
  IREE_RETURN_IF_ERROR(
      loom_target_low_legality_view_regions(context, &view_regions));
  if (!loom_amdgpu_source_value_prefers_vgpr(
          module, loom_target_low_legality_fact_table(context), view_regions,
          result)) {
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

static loom_amdgpu_scalar_conversion_op_group_t
loom_amdgpu_scalar_conversion_op_group(loom_op_kind_t op_kind) {
  switch (op_kind) {
    case LOOM_OP_SCALAR_TRUNCI:
      return LOOM_AMDGPU_SCALAR_CONVERSION_OP_TRUNCI;
    case LOOM_OP_SCALAR_EXTSI:
      return LOOM_AMDGPU_SCALAR_CONVERSION_OP_EXTSI;
    case LOOM_OP_SCALAR_EXTUI:
      return LOOM_AMDGPU_SCALAR_CONVERSION_OP_EXTUI;
    case LOOM_OP_SCALAR_UITOFP:
      return LOOM_AMDGPU_SCALAR_CONVERSION_OP_UITOFP;
    case LOOM_OP_SCALAR_FPTOSI:
      return LOOM_AMDGPU_SCALAR_CONVERSION_OP_FPTOSI;
    case LOOM_OP_SCALAR_FPTOUI:
      return LOOM_AMDGPU_SCALAR_CONVERSION_OP_FPTOUI;
    default:
      return LOOM_AMDGPU_SCALAR_CONVERSION_OP_COUNT_;
  }
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
  if (rule == NULL || !loom_amdgpu_descriptor_refs_present(
                          descriptor_set, rule->required_descriptor_refs,
                          IREE_ARRAYSIZE(rule->required_descriptor_refs))) {
    return false;
  }
  *out_plan = (loom_amdgpu_scalar_conversion_plan_t){
      .kind = rule->kind,
      .source = source,
      .result = result,
      .source_bit_count =
          loom_amdgpu_scalar_type_integer_bit_count(source_type),
      .result_bit_count =
          loom_amdgpu_scalar_type_integer_bit_count(result_type),
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

static bool loom_amdgpu_vector_conversion_lane_rule(
    loom_op_kind_t op_kind, loom_scalar_type_t source_element_type,
    loom_scalar_type_t result_element_type,
    loom_amdgpu_descriptor_ref_t* out_convert_descriptor_ref,
    bool* out_sign_extend_packed_source) {
  *out_convert_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  *out_sign_extend_packed_source = false;
  const uint32_t source_integer_bit_count =
      loom_amdgpu_scalar_type_integer_bit_count(source_element_type);
  const uint32_t result_integer_bit_count =
      loom_amdgpu_scalar_type_integer_bit_count(result_element_type);
  switch (op_kind) {
    case LOOM_OP_VECTOR_TRUNCI:
      return source_integer_bit_count != 0 && result_integer_bit_count != 0 &&
             source_integer_bit_count > result_integer_bit_count;
    case LOOM_OP_VECTOR_SITOFP:
      if (result_element_type != LOOM_SCALAR_TYPE_F32 ||
          source_integer_bit_count == 0 || source_integer_bit_count > 32) {
        return false;
      }
      *out_convert_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F32_I32;
      *out_sign_extend_packed_source = true;
      return true;
    case LOOM_OP_VECTOR_UITOFP:
      if (result_element_type != LOOM_SCALAR_TYPE_F32 ||
          source_integer_bit_count == 0 || source_integer_bit_count > 32) {
        return false;
      }
      *out_convert_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F32_U32;
      return true;
    case LOOM_OP_VECTOR_FPTOSI:
      if (source_element_type != LOOM_SCALAR_TYPE_F32 ||
          result_integer_bit_count == 0 || result_integer_bit_count > 32) {
        return false;
      }
      *out_convert_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_I32_F32;
      return true;
    case LOOM_OP_VECTOR_FPTOUI:
      if (source_element_type != LOOM_SCALAR_TYPE_F32 ||
          result_integer_bit_count == 0 || result_integer_bit_count > 32) {
        return false;
      }
      *out_convert_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_U32_F32;
      return true;
    default:
      return false;
  }
}

static bool loom_amdgpu_vector_conversion_needs_packed_source(
    loom_amdgpu_vector_conversion_kind_t kind) {
  return kind == LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_INTEGER_TO_FULL_32 ||
         kind ==
             LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_INTEGER_TO_PACKED_INTEGER;
}

static bool loom_amdgpu_vector_conversion_needs_packed_result(
    loom_amdgpu_vector_conversion_kind_t kind) {
  return kind == LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_32_TO_PACKED_INTEGER ||
         kind == LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_64_TO_PACKED_INTEGER ||
         kind ==
             LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_INTEGER_TO_PACKED_INTEGER;
}

static bool loom_amdgpu_vector_conversion_needs_full_source_materialization(
    loom_amdgpu_vector_conversion_kind_t kind) {
  return kind == LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_32_TO_FULL_32 ||
         kind == LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_32_TO_PACKED_INTEGER ||
         kind == LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_64_TO_FULL_32 ||
         kind == LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_64_TO_PACKED_INTEGER;
}

static bool loom_amdgpu_vector_conversion_select_storage_kind(
    const loom_amdgpu_vector_storage_t* source_storage,
    const loom_amdgpu_vector_storage_t* result_storage,
    loom_amdgpu_vector_conversion_kind_t* out_kind) {
  *out_kind = LOOM_AMDGPU_VECTOR_CONVERSION_KIND_NONE;
  switch (source_storage->kind) {
    case LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_32BIT:
      if (result_storage->kind == LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_32BIT) {
        *out_kind = LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_32_TO_FULL_32;
        return true;
      }
      if (result_storage->kind ==
          LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_INTEGER) {
        *out_kind =
            LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_32_TO_PACKED_INTEGER;
        return true;
      }
      return false;
    case LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_64BIT:
      if (result_storage->kind == LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_32BIT) {
        *out_kind = LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_64_TO_FULL_32;
        return true;
      }
      if (result_storage->kind ==
          LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_INTEGER) {
        *out_kind =
            LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_64_TO_PACKED_INTEGER;
        return true;
      }
      return false;
    case LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_INTEGER:
      if (result_storage->kind == LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_32BIT) {
        *out_kind =
            LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_INTEGER_TO_FULL_32;
        return true;
      }
      if (result_storage->kind ==
          LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_INTEGER) {
        *out_kind =
            LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_INTEGER_TO_PACKED_INTEGER;
        return true;
      }
      return false;
    default:
      return false;
  }
}

static bool loom_amdgpu_vector_conversion_descriptor_refs_present(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_vector_conversion_kind_t kind,
    loom_amdgpu_descriptor_ref_t convert_descriptor_ref,
    bool sign_extend_packed_source) {
  if (convert_descriptor_ref != LOOM_AMDGPU_DESCRIPTOR_REF_NONE &&
      !loom_amdgpu_descriptor_set_has_ref(descriptor_set,
                                          convert_descriptor_ref)) {
    return false;
  }

  static const loom_amdgpu_descriptor_ref_t kFullSourceRefs[] = {
      LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_COPY,
  };
  if (convert_descriptor_ref != LOOM_AMDGPU_DESCRIPTOR_REF_NONE &&
      loom_amdgpu_vector_conversion_needs_full_source_materialization(kind) &&
      !loom_amdgpu_descriptor_refs_present(descriptor_set, kFullSourceRefs,
                                           IREE_ARRAYSIZE(kFullSourceRefs))) {
    return false;
  }

  static const loom_amdgpu_descriptor_ref_t kPackedResultRefs[] = {
      LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32,
  };
  if (loom_amdgpu_vector_conversion_needs_packed_result(kind) &&
      !loom_amdgpu_descriptor_refs_present(descriptor_set, kPackedResultRefs,
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
  if (!loom_amdgpu_vector_conversion_needs_packed_source(kind)) {
    return true;
  }
  if (sign_extend_packed_source) {
    return loom_amdgpu_descriptor_refs_present(
        descriptor_set, kPackedSourceSignedRefs,
        IREE_ARRAYSIZE(kPackedSourceSignedRefs));
  }
  return loom_amdgpu_descriptor_refs_present(
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

  loom_amdgpu_descriptor_ref_t convert_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  bool sign_extend_packed_source = false;
  if (!loom_amdgpu_vector_conversion_lane_rule(
          source_op->kind, source_storage.element_type,
          result_storage.element_type, &convert_descriptor_ref,
          &sign_extend_packed_source)) {
    return false;
  }

  loom_amdgpu_vector_conversion_kind_t kind =
      LOOM_AMDGPU_VECTOR_CONVERSION_KIND_NONE;
  if (!loom_amdgpu_vector_conversion_select_storage_kind(
          &source_storage, &result_storage, &kind)) {
    return false;
  }
  if (!loom_amdgpu_vector_conversion_descriptor_refs_present(
          descriptor_set, kind, convert_descriptor_ref,
          sign_extend_packed_source)) {
    return false;
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
      .convert_descriptor_ref = convert_descriptor_ref,
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
      if (loom_amdgpu_select_vector_from_elements_plan(context, source_op,
                                                       &local_plan)) {
        return loom_amdgpu_bind_selected_value_plan(
            context, source_op, &local_plan, sizeof(local_plan), out_plan);
      }
      return iree_ok_status();
    }
    case LOOM_OP_VECTOR_SPLAT: {
      loom_amdgpu_vector_from_elements_plan_t local_plan = {0};
      if (loom_amdgpu_select_vector_splat_plan(context, source_op,
                                               &local_plan)) {
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

static iree_status_t loom_amdgpu_extract_register_unit(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint32_t register_count,
    uint32_t register_offset, loom_type_t unit_type,
    loom_value_id_t* out_register_unit) {
  *out_register_unit = LOOM_VALUE_ID_INVALID;
  if (register_count == 1) {
    *out_register_unit = low_source;
    return iree_ok_status();
  }
  return loom_amdgpu_emit_low_slice(context, source_op, low_source,
                                    register_offset, unit_type,
                                    out_register_unit);
}

static iree_status_t loom_amdgpu_emit_vgpr_conversion_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t low_source,
    loom_type_t result_type, loom_value_id_t* out_low_result) {
  *out_low_result = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {low_source};
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, descriptor_ref, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), &result_type, 1, &low_op));
  *out_low_result = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_vgpr_sign_extend_narrow(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint32_t result_bit_count,
    loom_value_id_t* out_low_result) {
  *out_low_result = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT(result_bit_count == 8 || result_bit_count == 16);
  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  bool selected_sdwa = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_try_emit_vgpr_b32_sdwa_extract(
      context, source_op, low_source, /*bit_offset=*/0, result_bit_count,
      LOOM_AMDGPU_VGPR_SDWA_EXTRACT_FLAG_SIGN_EXTEND, lane_type, out_low_result,
      &selected_sdwa));
  if (!selected_sdwa) {
    const uint32_t shift = 32u - result_bit_count;
    loom_value_id_t shifted_left = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, shift,
        low_source, lane_type, &shifted_left));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT, shift,
        shifted_left, lane_type, out_low_result));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_extract_packed_register_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, const loom_amdgpu_vector_extract_plan_t* plan,
    uint32_t lane_offset, loom_type_t lane_type, loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT(plan->lane_bit_count == 8 || plan->lane_bit_count == 16);
  const uint32_t lanes_per_register = 32u / plan->lane_bit_count;
  const uint32_t register_offset = lane_offset / lanes_per_register;
  const uint32_t register_bit_offset =
      (lane_offset % lanes_per_register) * plan->lane_bit_count;
  loom_value_id_t source_register = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_register_unit(
      context, source_op, low_source, plan->register_count, register_offset,
      lane_type, &source_register));
  if (!plan->sign_extend_packed_lane) {
    return loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
        register_bit_offset, source_register, lane_type, out_lane);
  }

  bool selected_bfe = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_try_emit_vgpr_b32_bfe_extract(
      context, source_op, source_register, register_bit_offset,
      plan->lane_bit_count, LOOM_AMDGPU_VGPR_BFE_EXTRACT_FLAG_SIGN_EXTEND,
      lane_type, out_lane, &selected_bfe));
  if (selected_bfe) {
    return iree_ok_status();
  }

  loom_value_id_t shifted_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
      register_bit_offset, source_register, lane_type, &shifted_lane));
  return loom_amdgpu_emit_vgpr_sign_extend_narrow(
      context, source_op, shifted_lane, plan->lane_bit_count, out_lane);
}

static loom_type_t loom_amdgpu_low_register_lane_type(
    const loom_module_t* module, loom_value_id_t low_value) {
  const loom_type_t low_type = loom_module_value_type(module, low_value);
  if (!loom_low_type_is_register(low_type)) {
    return loom_type_none();
  }
  return loom_low_register_type_with_unit_count(low_type, 1);
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
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_packed_register_lane(
      context, source_op, low_source, &extract_plan, lane_index, lane_type,
      out_lane));
  if (plan->sign_extend_packed_source) {
    return iree_ok_status();
  }
  if (plan->kind ==
          LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_INTEGER_TO_PACKED_INTEGER &&
      plan->convert_descriptor_ref == LOOM_AMDGPU_DESCRIPTOR_REF_NONE) {
    return iree_ok_status();
  }
  return loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, *out_lane,
      loom_amdgpu_integer_bit_mask(plan->source_bit_count), lane_type,
      out_lane);
}

static iree_status_t loom_amdgpu_extract_vector_conversion_full_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_conversion_plan_t* plan,
    loom_value_id_t low_source, uint32_t lane_index,
    loom_type_t source_lane_type, loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  const uint32_t register_offset =
      lane_index * plan->source_element_register_count;
  return loom_amdgpu_extract_register_unit(
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
  switch (plan->kind) {
    case LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_INTEGER_TO_FULL_32:
    case LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_INTEGER_TO_PACKED_INTEGER:
      return loom_amdgpu_extract_vector_conversion_packed_lane(
          context, source_op, plan, low_source, lane_index, lane_type,
          out_lane);
    case LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_32_TO_FULL_32:
    case LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_32_TO_PACKED_INTEGER:
    case LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_64_TO_FULL_32:
    case LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_64_TO_PACKED_INTEGER:
      return loom_amdgpu_extract_vector_conversion_full_lane(
          context, source_op, plan, low_source, lane_index, source_lane_type,
          out_lane);
    case LOOM_AMDGPU_VECTOR_CONVERSION_KIND_NONE:
      break;
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
  return loom_amdgpu_emit_vgpr_conversion_packet(context, source_op,
                                                 plan->convert_descriptor_ref,
                                                 lane, lane_type, out_lane);
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

  switch (plan->kind) {
    case LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_32_TO_FULL_32:
    case LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_64_TO_FULL_32:
    case LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_INTEGER_TO_FULL_32:
      return loom_amdgpu_lower_vector_conversion_full_result(
          context, source_op, plan, low_source, source_lane_type, lane_type);
    case LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_32_TO_PACKED_INTEGER:
    case LOOM_AMDGPU_VECTOR_CONVERSION_KIND_FULL_64_TO_PACKED_INTEGER:
    case LOOM_AMDGPU_VECTOR_CONVERSION_KIND_PACKED_INTEGER_TO_PACKED_INTEGER:
      return loom_amdgpu_lower_vector_conversion_packed_result(
          context, source_op, plan, low_source, source_lane_type, lane_type);
    case LOOM_AMDGPU_VECTOR_CONVERSION_KIND_NONE:
      break;
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
    return loom_amdgpu_extract_packed_register_lane(
        context, source_op, low_source, plan, lane_offset, unit_type,
        out_register_unit);
  }

  IREE_ASSERT_TRUE(plan->lane_bit_count == 32 || plan->lane_bit_count == 64);
  const uint32_t register_offset =
      lane_offset * plan->element_register_count + result_register_index;
  IREE_ASSERT(register_offset < plan->register_count);
  return loom_amdgpu_extract_register_unit(
      context, source_op, low_source, plan->register_count, register_offset,
      unit_type, out_register_unit);
}

static bool loom_amdgpu_low_value_defines_vgpr_low16(
    loom_low_lower_context_t* context, loom_value_id_t low_value_id) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_t* low_value = loom_module_value(module, low_value_id);
  if (low_value == NULL || loom_value_is_block_arg(low_value)) {
    return false;
  }
  const loom_op_t* low_op = loom_value_def_op(low_value);
  if (low_op == NULL || !loom_low_op_isa(low_op)) {
    return false;
  }
  const int64_t descriptor_ordinal = loom_low_op_descriptor_ordinal(low_op);
  if (descriptor_ordinal < 0 || (uint64_t)descriptor_ordinal > UINT32_MAX) {
    return false;
  }
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set,
                                            (uint32_t)descriptor_ordinal);
  const uint16_t result_index = loom_value_def_index(low_value);
  if (descriptor == NULL || result_index >= descriptor->result_count) {
    return false;
  }
  const uint32_t operand_index = descriptor->operand_start + result_index;
  if (operand_index >= descriptor_set->operand_count) {
    return false;
  }
  const loom_low_operand_t* result_operand =
      &descriptor_set->operands[operand_index];
  if (result_operand->register_part_id >= descriptor_set->register_part_count) {
    return false;
  }
  const loom_low_register_part_t* register_part =
      &descriptor_set->register_parts[result_operand->register_part_id];
  const iree_string_view_t register_part_name = loom_low_descriptor_set_string(
      descriptor_set, register_part->name_string_offset);
  return iree_string_view_equal(register_part_name,
                                IREE_SV("amdgpu.vgpr.low16"));
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

static iree_status_t loom_amdgpu_lookup_or_compose_vgpr_16bit_float(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, uint32_t register_bit_offset,
    loom_type_t lane_type, loom_value_id_t* out_low_value) {
  *out_low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_value, out_low_value));

  if (loom_amdgpu_low_value_defines_vgpr_low16(context, *out_low_value)) {
    if (register_bit_offset == 0) {
      const loom_value_id_t operands[] = {*out_low_value};
      loom_op_t* low_op = NULL;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
          context, source_op,
          LOOM_AMDGPU_DESCRIPTOR_REF_V_BFE_U32_OFFSET_0_WIDTH_16_LOW16,
          operands, IREE_ARRAYSIZE(operands),
          loom_make_named_attr_slice(NULL, 0), &lane_type, 1, &low_op));
      *out_low_value = loom_value_slice_get(loom_low_op_results(low_op), 0);
      return iree_ok_status();
    }
    if (register_bit_offset == 16) {
      const loom_value_id_t operands[] = {*out_low_value};
      loom_op_t* low_op = NULL;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
          context, source_op,
          LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_SRC0_16_LOW16, operands,
          IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(NULL, 0),
          &lane_type, 1, &low_op));
      *out_low_value = loom_value_slice_get(loom_low_op_results(low_op), 0);
      return iree_ok_status();
    }
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
      context, source_op, *out_low_value, out_low_value));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      *out_low_value, UINT32_C(0xFFFF), lane_type, out_low_value));
  return loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
      register_bit_offset, *out_low_value, lane_type, out_low_value);
}

static iree_status_t loom_amdgpu_lookup_or_materialize_vgpr_16bit_float(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, loom_type_t lane_type,
    loom_value_id_t* out_low_value) {
  return loom_amdgpu_lookup_or_compose_vgpr_16bit_float(
      context, source_op, source_value, 0, lane_type, out_low_value);
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
    loom_value_id_t packed = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_compose_vgpr_16bit_float(
        context, source_op, plan->payload.elements[lane_base], 0, lane_type,
        &packed));
    if (lane_base + 1u < plan->element_count) {
      loom_value_id_t high_lane = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_compose_vgpr_16bit_float(
          context, source_op, plan->payload.elements[lane_base + 1u], 16,
          lane_type, &high_lane));
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
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "AMDGPU packed integer vector constant lowering "
                              "requires v_mov_b32");
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
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_register_unit(
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
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_register_unit(
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

static iree_status_t loom_amdgpu_extract_16bit_float_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint32_t source_register_count,
    uint32_t lane_index, loom_type_t source_lane_type,
    loom_type_t result_lane_type, loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  const uint32_t register_index = lane_index / 2u;
  const uint32_t register_bit_offset = (lane_index % 2u) * 16u;
  loom_value_id_t source_register = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_register_unit(
      context, source_op, low_source, source_register_count, register_index,
      source_lane_type, &source_register));
  if (register_bit_offset == 0) {
    return loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, 16,
        source_register, result_lane_type, out_lane);
  }
  return loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      source_register, UINT32_C(0xFFFF0000), result_lane_type, out_lane);
}

iree_status_t loom_amdgpu_emit_f32_to_bf16_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_lane, loom_type_t lane_type,
    loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
      context, source_op, source_lane, &source_lane));

  loom_value_id_t upper = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT, 16,
      source_lane, lane_type, &upper));
  loom_value_id_t lsb = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, upper, 1,
      lane_type, &lsb));
  loom_value_id_t bias = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT, lsb,
      UINT32_C(0x7FFF), lane_type, &bias));
  loom_value_id_t rounded = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32, source_lane,
      bias, lane_type, &rounded));
  return loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT, 16,
      rounded, lane_type, out_lane);
}

static iree_status_t loom_amdgpu_emit_bf16_pack_descriptor(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t low_lane, loom_value_id_t high_lane, loom_type_t lane_type,
    loom_value_id_t* out_packed) {
  *out_packed = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {low_lane, high_lane};
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), &lane_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_packed = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_native_f32_to_packed_bf16(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t low_source_lane, loom_value_id_t high_source_lane,
    loom_type_t lane_type, loom_value_id_t* out_packed) {
  *out_packed = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
      context, source_op, low_source_lane, &low_source_lane));
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
      context, source_op, high_source_lane, &high_source_lane));
  return loom_amdgpu_emit_bf16_pack_descriptor(
      context, source_op, descriptor, low_source_lane, high_source_lane,
      lane_type, out_packed);
}

iree_status_t loom_amdgpu_emit_f32_pair_to_packed_bf16(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source_lane, loom_value_id_t high_source_lane,
    loom_type_t lane_type, loom_value_id_t* out_packed) {
  *out_packed = LOOM_VALUE_ID_INVALID;
  loom_low_lower_resolved_descriptor_t native_descriptor = {0};
  bool has_native_descriptor = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_BF16_F32, &native_descriptor,
      &has_native_descriptor));
  if (has_native_descriptor) {
    return loom_amdgpu_emit_native_f32_to_packed_bf16(
        context, source_op, &native_descriptor, low_source_lane,
        high_source_lane, lane_type, out_packed);
  }

  loom_low_lower_resolved_descriptor_t pack_descriptor = {0};
  bool has_pack_descriptor = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_U16_U32, &pack_descriptor,
      &has_pack_descriptor));

  loom_value_id_t low_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_f32_to_bf16_lane(
      context, source_op, low_source_lane, lane_type, &low_lane));
  loom_value_id_t high_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_f32_to_bf16_lane(
      context, source_op, high_source_lane, lane_type, &high_lane));
  if (has_pack_descriptor) {
    return loom_amdgpu_emit_bf16_pack_descriptor(
        context, source_op, &pack_descriptor, low_lane, high_lane, lane_type,
        out_packed);
  }

  loom_value_id_t high_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, 16,
      high_lane, lane_type, &high_bits));
  return loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32, low_lane,
      high_bits, lane_type, out_packed);
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
      for (uint32_t i = 0; i < IREE_ARRAYSIZE(insert_plan->fma_mix.sources);
           ++i) {
        loom_low_lower_require_source_value_storage(
            context, insert_plan->fma_mix.sources[i]);
      }
      return;
    }
    default:
      loom_low_lower_require_source_operands_storage(context, source_op);
      return;
  }
}

static iree_status_t loom_amdgpu_lower_vector_bf16_extf(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_bf16_conversion_plan_t* plan) {
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
  loom_type_t result_lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &result_lane_type));

  loom_value_id_t lanes[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < plan->lane_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_16bit_float_lane(
        context, source_op, low_source, plan->source_register_count, i,
        source_lane_type, result_lane_type, &lanes[i]));
  }

  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             lanes, plan->lane_count);
}

static iree_status_t loom_amdgpu_lower_vector_bf16_fptrunc(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_bf16_conversion_plan_t* plan) {
  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->source, &low_source));

  loom_low_lower_resolved_descriptor_t native_descriptor = {0};
  bool has_native_descriptor = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_BF16_F32, &native_descriptor,
      &has_native_descriptor));
  loom_low_lower_resolved_descriptor_t pack_descriptor = {0};
  bool has_pack_descriptor = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_U16_U32, &pack_descriptor,
      &has_pack_descriptor));

  const loom_module_t* module = loom_low_lower_context_module(context);
  loom_type_t source_lane_type =
      loom_amdgpu_low_register_lane_type(module, low_source);
  if (loom_type_kind(source_lane_type) == LOOM_TYPE_NONE) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_vgpr_type(context, &source_lane_type));
  }
  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));

  loom_value_id_t packed_registers[LOOM_AMDGPU_MAX_PACKED_16BIT_FLOAT_LANES];
  for (uint32_t register_index = 0;
       register_index < plan->result_register_count; ++register_index) {
    const uint32_t lane_base = register_index * 2u;
    loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_register_unit(
        context, source_op, low_source, plan->source_register_count, lane_base,
        source_lane_type, &source_lane));
    loom_value_id_t packed = LOOM_VALUE_ID_INVALID;

    if (lane_base + 1u < plan->lane_count) {
      loom_value_id_t high_source_lane = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_extract_register_unit(
          context, source_op, low_source, plan->source_register_count,
          lane_base + 1u, source_lane_type, &high_source_lane));
      if (has_native_descriptor) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_native_f32_to_packed_bf16(
            context, source_op, &native_descriptor, source_lane,
            high_source_lane, lane_type, &packed));
        packed_registers[register_index] = packed;
        continue;
      }

      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_f32_to_bf16_lane(
          context, source_op, source_lane, lane_type, &packed));
      loom_value_id_t high_lane = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_f32_to_bf16_lane(
          context, source_op, high_source_lane, lane_type, &high_lane));
      if (has_pack_descriptor) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_bf16_pack_descriptor(
            context, source_op, &pack_descriptor, packed, high_lane, lane_type,
            &packed));
      } else {
        loom_value_id_t high_bits = LOOM_VALUE_ID_INVALID;
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
            context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
            16, high_lane, lane_type, &high_bits));
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
            context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32, packed,
            high_bits, lane_type, &packed));
      }
    } else if (has_native_descriptor) {
      loom_value_id_t zero_lane = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0,
          lane_type, &zero_lane));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_native_f32_to_packed_bf16(
          context, source_op, &native_descriptor, source_lane, zero_lane,
          lane_type, &packed));
    } else {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_f32_to_bf16_lane(
          context, source_op, source_lane, lane_type, &packed));
    }
    packed_registers[register_index] = packed;
  }

  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             packed_registers,
                                             plan->result_register_count);
}

iree_status_t loom_amdgpu_lower_vector_bf16_conversion(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_bf16_conversion_plan_t* plan) {
  switch (plan->kind) {
    case LOOM_AMDGPU_VECTOR_BF16_CONVERSION_KIND_EXTF:
      return loom_amdgpu_lower_vector_bf16_extf(context, source_op, plan);
    case LOOM_AMDGPU_VECTOR_BF16_CONVERSION_KIND_FPTRUNC:
      return loom_amdgpu_lower_vector_bf16_fptrunc(context, source_op, plan);
    default:
      IREE_ASSERT_UNREACHABLE("unknown BF16 conversion plan");
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

static iree_status_t loom_amdgpu_lookup_or_materialize_offset_add_operand(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, bool result_is_vgpr,
    loom_value_id_t* out_low_value) {
  *out_low_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_value, &low_value));

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t low_type = loom_module_value_type(module, low_value);
  if (!loom_low_type_is_register(low_type)) {
    IREE_ASSERT_UNREACHABLE(
        "AMDGPU offset-add plan selected non-register operand");
    IREE_BUILTIN_UNREACHABLE();
  }
  const uint32_t unit_count = loom_low_register_type_unit_count(low_type);

  if (result_is_vgpr) {
    bool is_vgpr = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
        context, low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR, &is_vgpr));
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
    bool is_sgpr = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
        context, low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, &is_sgpr));
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
      "AMDGPU offset-add plan selected incompatible operand register type");
  IREE_BUILTIN_UNREACHABLE();
}

iree_status_t loom_amdgpu_lower_offset_add(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_offset_add_plan_t* plan) {
  loom_value_id_t low_lhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_offset_add_operand(
      context, source_op, plan->lhs, plan->result_is_vgpr, &low_lhs));
  loom_value_id_t low_rhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_offset_add_operand(
      context, source_op, plan->rhs, plan->result_is_vgpr, &low_rhs));

  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  if (plan->result_is_vgpr) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr64_add(
        context, source_op, low_lhs, low_rhs, &low_result));
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr64_add(
        context, source_op, low_lhs, low_rhs, &low_result));
  }
  return loom_low_lower_bind_value(context, plan->result, low_result);
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
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_register_unit(
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
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU scalar source lowered to a non-register type");
  }
  loom_value_id_t low_source_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, low_source_pair, /*lane_offset=*/0, source_lane_type,
      &low_source_lane));
  return loom_amdgpu_materialize_low_vgpr_b32_registers(
      context, source_op, low_source_lane, out_low_source);
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
  bool result_is_vgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
      context, result_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR, &result_is_vgpr));
  if (result_is_vgpr) {
    return loom_amdgpu_lookup_or_materialize_vgpr_i32(context, source_op,
                                                      source, out_low_source);
  }

  bool result_is_sgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
      context, result_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, &result_is_sgpr));
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
  bool source_is_sgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
      context, source_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, &source_is_sgpr));
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
  bool lane_is_vgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
      context, lane_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR, &lane_is_vgpr));
  loom_value_id_t high_bits = LOOM_VALUE_ID_INVALID;
  if (lane_is_vgpr) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT,
        /*shift=*/31, low_source, lane_type, &high_bits));
  } else {
    bool lane_is_sgpr = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
        context, lane_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, &lane_is_sgpr));
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
  bool lane_is_vgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
      context, lane_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR, &lane_is_vgpr));
  loom_amdgpu_descriptor_ref_t zero_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32;
  if (!lane_is_vgpr) {
    bool lane_is_sgpr = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
        context, lane_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, &lane_is_sgpr));
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
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_sign_extend_narrow(
          context, source_op, low_source, plan->result_bit_count, &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_SIGN_EXTEND_NARROW_LOW_32: {
      loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_32_bits_as_vgpr(
          context, source_op, plan->source, &low_source));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_sign_extend_narrow(
          context, source_op, low_source, plan->result_bit_count, &low_result));
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
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_conversion_packet(
          context, source_op, plan->convert_descriptor_ref,
          zero_extended_source, result_type, &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FPTOI_F32_TO_I32: {
      loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_f32(
          context, source_op, plan->source, &low_source));
      loom_type_t lane_type = loom_type_none();
      IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_conversion_packet(
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
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_conversion_packet(
          context, source_op, plan->convert_descriptor_ref, low_source,
          lane_type, &converted_source));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_sign_extend_narrow(
          context, source_op, converted_source, plan->result_bit_count,
          &low_result));
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
        return iree_make_status(
            IREE_STATUS_INTERNAL,
            "AMDGPU fact-only vector atomic offset reached emission");
      }
      return loom_amdgpu_lower_vector_iota(
          context, source_op,
          (const loom_amdgpu_vector_iota_plan_t*)plan.target_data);
    case LOOM_OP_VECTOR_FROM_ELEMENTS:
    case LOOM_OP_VECTOR_SPLAT:
      if (plan.target_data == NULL) {
        IREE_ASSERT_UNREACHABLE(
            "AMDGPU fact-only vector atomic offset reached emission");
        return iree_make_status(
            IREE_STATUS_INTERNAL,
            "AMDGPU fact-only vector atomic offset reached emission");
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
