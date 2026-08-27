// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/value/vector_construct.h"

#include <stdint.h>
#include <string.h>

#include "iree/base/internal/math.h"
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
#include "loom/target/arch/amdgpu/lower/descriptor_ref.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/legality.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

typedef enum loom_amdgpu_vector_construct_source_kind_e {
  LOOM_AMDGPU_VECTOR_CONSTRUCT_SOURCE_KIND_NONE = 0,
  LOOM_AMDGPU_VECTOR_CONSTRUCT_SOURCE_KIND_VECTOR_IOTA = 1,
  LOOM_AMDGPU_VECTOR_CONSTRUCT_SOURCE_KIND_VECTOR_FROM_ELEMENTS = 2,
  LOOM_AMDGPU_VECTOR_CONSTRUCT_SOURCE_KIND_VECTOR_SPLAT = 3,
  LOOM_AMDGPU_VECTOR_CONSTRUCT_SOURCE_KIND_VECTOR_INSERT = 4,
} loom_amdgpu_vector_construct_source_kind_t;

typedef enum loom_amdgpu_vector_construct_source_flags_e {
  LOOM_AMDGPU_VECTOR_CONSTRUCT_SOURCE_FLAG_NONE = 0u,
  LOOM_AMDGPU_VECTOR_CONSTRUCT_SOURCE_FLAG_ATOMIC_OFFSET_FACT_ONLY = 1u << 0,
  LOOM_AMDGPU_VECTOR_CONSTRUCT_SOURCE_FLAG_PRESELECT_FMA_MIX = 1u << 1,
} loom_amdgpu_vector_construct_source_flags_t;

typedef struct loom_amdgpu_vector_construct_source_row_t {
  // Source operation family routed by this row.
  loom_amdgpu_vector_construct_source_kind_t kind;
  // Row behavior flags used by selection and preselection.
  loom_amdgpu_vector_construct_source_flags_t flags;
} loom_amdgpu_vector_construct_source_row_t;

#define LOOM_AMDGPU_VECTOR_OP_INDEX(op_kind) ((uint8_t)((op_kind) & 0xFFu))
#define LOOM_AMDGPU_VECTOR_CONSTRUCT_SOURCE_ROW(op, kind_, flags_) \
  [LOOM_AMDGPU_VECTOR_OP_INDEX(LOOM_OP_VECTOR_##op)] = {           \
      .kind = LOOM_AMDGPU_VECTOR_CONSTRUCT_SOURCE_KIND_##kind_,    \
      .flags = flags_,                                             \
  }

static const loom_amdgpu_vector_construct_source_row_t
    kAmdgpuVectorConstructSourceRows[LOOM_OP_VECTOR_COUNT_] = {
        LOOM_AMDGPU_VECTOR_CONSTRUCT_SOURCE_ROW(
            IOTA, VECTOR_IOTA,
            LOOM_AMDGPU_VECTOR_CONSTRUCT_SOURCE_FLAG_ATOMIC_OFFSET_FACT_ONLY),
        LOOM_AMDGPU_VECTOR_CONSTRUCT_SOURCE_ROW(
            FROM_ELEMENTS, VECTOR_FROM_ELEMENTS,
            LOOM_AMDGPU_VECTOR_CONSTRUCT_SOURCE_FLAG_ATOMIC_OFFSET_FACT_ONLY |
                LOOM_AMDGPU_VECTOR_CONSTRUCT_SOURCE_FLAG_PRESELECT_FMA_MIX),
        LOOM_AMDGPU_VECTOR_CONSTRUCT_SOURCE_ROW(
            SPLAT, VECTOR_SPLAT,
            LOOM_AMDGPU_VECTOR_CONSTRUCT_SOURCE_FLAG_PRESELECT_FMA_MIX),
        LOOM_AMDGPU_VECTOR_CONSTRUCT_SOURCE_ROW(
            INSERT, VECTOR_INSERT,
            LOOM_AMDGPU_VECTOR_CONSTRUCT_SOURCE_FLAG_PRESELECT_FMA_MIX),
};

#undef LOOM_AMDGPU_VECTOR_CONSTRUCT_SOURCE_ROW
#undef LOOM_AMDGPU_VECTOR_OP_INDEX

static const loom_amdgpu_vector_construct_source_row_t*
loom_amdgpu_vector_construct_source_row(loom_op_kind_t op_kind) {
  if (loom_op_dialect_id(op_kind) != LOOM_DIALECT_VECTOR) {
    return NULL;
  }
  const uint8_t op_index = loom_op_dialect_index(op_kind);
  if (op_index >= IREE_ARRAYSIZE(kAmdgpuVectorConstructSourceRows)) {
    return NULL;
  }
  const loom_amdgpu_vector_construct_source_row_t* row =
      &kAmdgpuVectorConstructSourceRows[op_index];
  return row->kind == LOOM_AMDGPU_VECTOR_CONSTRUCT_SOURCE_KIND_NONE ? NULL
                                                                    : row;
}

static bool loom_amdgpu_vector_construct_source_row_has_flag(
    const loom_amdgpu_vector_construct_source_row_t* row,
    loom_amdgpu_vector_construct_source_flags_t flag) {
  return row != NULL && iree_any_bit_set(row->flags, flag);
}

typedef struct loom_amdgpu_vector_construct_plan_header_t {
  // Vector-construction family selected by the planner.
  loom_amdgpu_vector_construct_source_kind_t kind;
} loom_amdgpu_vector_construct_plan_header_t;

typedef uint32_t loom_amdgpu_vector_iota_plan_flags_t;

enum {
  LOOM_AMDGPU_VECTOR_IOTA_PLAN_BASE_EXACT = 1u << 0,
  LOOM_AMDGPU_VECTOR_IOTA_PLAN_STEP_EXACT = 1u << 1,
};

typedef struct loom_amdgpu_vector_iota_plan_t {
  // Common vector construct plan header.
  loom_amdgpu_vector_construct_plan_header_t header;
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
  // Common vector construct plan header.
  loom_amdgpu_vector_construct_plan_header_t header;
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
  // Common vector construct plan header.
  loom_amdgpu_vector_construct_plan_header_t header;
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
  *out_bits = iree_math_mask_low_bits_u32((uint32_t)value, (int32_t)bit_count);
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
  const uint32_t element_mask =
      iree_math_mask_low_bits_u32(UINT32_MAX, (int32_t)element_bit_count);
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

static const loom_amdgpu_descriptor_requirement_t
    kAmdgpuVectorIotaDynamicDescriptorRequirements[] = {
        {
            .constraint_key = IREE_SVL("descriptor.v_mov_b32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        },
        {
            .constraint_key = IREE_SVL("descriptor.v_mov_b32_copy"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_COPY,
        },
};

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
          descriptor_set, kAmdgpuVectorIotaDynamicDescriptorRequirements,
          IREE_ARRAYSIZE(kAmdgpuVectorIotaDynamicDescriptorRequirements),
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
  out_plan->header.kind = LOOM_AMDGPU_VECTOR_CONSTRUCT_SOURCE_KIND_VECTOR_IOTA;
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
  out_plan->header.kind =
      LOOM_AMDGPU_VECTOR_CONSTRUCT_SOURCE_KIND_VECTOR_FROM_ELEMENTS;
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
  out_plan->header.kind = LOOM_AMDGPU_VECTOR_CONSTRUCT_SOURCE_KIND_VECTOR_SPLAT;
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
        .header =
            {
                .kind = LOOM_AMDGPU_VECTOR_CONSTRUCT_SOURCE_KIND_VECTOR_INSERT,
            },
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
        .is_dynamic = false,
    };
    IREE_RETURN_IF_ERROR(loom_amdgpu_select_vector_insert_fma_mix_half_result(
        context, source_op, flags, &lane_plan));
    if (lane_plan.value_kind !=
        LOOM_AMDGPU_VECTOR_INSERT_VALUE_KIND_FMA_MIX_HALF_RESULT) {
      continue;
    }
    if (inout_plan->fma_mix_half_results == NULL) {
      IREE_RETURN_IF_ERROR(loom_low_lower_allocate_function_array(
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
      .header =
          {
              .kind = LOOM_AMDGPU_VECTOR_CONSTRUCT_SOURCE_KIND_VECTOR_INSERT,
          },
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
      .is_dynamic = is_dynamic,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_vector_insert_fma_mix_half_result(
      context, source_op, flags, out_plan));
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_bind_selected_vector_construct_plan(
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

static iree_status_t loom_amdgpu_select_vector_construct_plan_from_row(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_construct_source_row_t* row,
    loom_amdgpu_vector_insert_select_flags_t flags, bool require_fma_mix,
    loom_low_lower_plan_t* out_plan) {
  switch (row->kind) {
    case LOOM_AMDGPU_VECTOR_CONSTRUCT_SOURCE_KIND_VECTOR_IOTA: {
      loom_amdgpu_vector_iota_plan_t local_plan = {0};
      bool selected = false;
      IREE_RETURN_IF_ERROR(loom_amdgpu_select_vector_iota_plan(
          context, source_op, &local_plan, &selected));
      if (!selected) {
        return iree_ok_status();
      }
      return loom_amdgpu_bind_selected_vector_construct_plan(
          context, source_op, &local_plan, sizeof(local_plan), out_plan);
    }
    case LOOM_AMDGPU_VECTOR_CONSTRUCT_SOURCE_KIND_VECTOR_FROM_ELEMENTS:
    case LOOM_AMDGPU_VECTOR_CONSTRUCT_SOURCE_KIND_VECTOR_SPLAT: {
      loom_amdgpu_vector_from_elements_plan_t local_plan = {0};
      bool selected = false;
      if (row->kind ==
          LOOM_AMDGPU_VECTOR_CONSTRUCT_SOURCE_KIND_VECTOR_FROM_ELEMENTS) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_select_vector_from_elements_plan(
            context, source_op, &local_plan, &selected));
      } else {
        IREE_RETURN_IF_ERROR(loom_amdgpu_select_vector_splat_plan(
            context, source_op, &local_plan, &selected));
      }
      if (!selected) {
        return iree_ok_status();
      }
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_select_vector_from_elements_fma_mix_half_results(
              context, source_op, flags, &local_plan));
      if (require_fma_mix && local_plan.fma_mix_half_result_lane_mask == 0) {
        return iree_ok_status();
      }
      return loom_amdgpu_bind_selected_vector_construct_plan(
          context, source_op, &local_plan, sizeof(local_plan), out_plan);
    }
    case LOOM_AMDGPU_VECTOR_CONSTRUCT_SOURCE_KIND_VECTOR_INSERT: {
      loom_amdgpu_vector_insert_plan_t local_plan = {0};
      bool selected = false;
      IREE_RETURN_IF_ERROR(loom_amdgpu_select_vector_insert_plan(
          context, source_op, flags, &local_plan, &selected));
      if (!selected ||
          (require_fma_mix &&
           local_plan.value_kind !=
               LOOM_AMDGPU_VECTOR_INSERT_VALUE_KIND_FMA_MIX_HALF_RESULT)) {
        return iree_ok_status();
      }
      return loom_amdgpu_bind_selected_vector_construct_plan(
          context, source_op, &local_plan, sizeof(local_plan), out_plan);
    }
    case LOOM_AMDGPU_VECTOR_CONSTRUCT_SOURCE_KIND_NONE:
    default:
      IREE_ASSERT_UNREACHABLE("unknown AMDGPU vector construct source kind");
      IREE_BUILTIN_UNREACHABLE();
  }
}

iree_status_t loom_amdgpu_select_vector_construct_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan) {
  *out_plan = loom_low_lower_plan_empty();
  const loom_amdgpu_vector_construct_source_row_t* row =
      loom_amdgpu_vector_construct_source_row(source_op->kind);
  if (row == NULL) {
    return iree_ok_status();
  }
  if (loom_amdgpu_vector_construct_source_row_has_flag(
          row,
          LOOM_AMDGPU_VECTOR_CONSTRUCT_SOURCE_FLAG_ATOMIC_OFFSET_FACT_ONLY) &&
      loom_amdgpu_select_fact_only_vector_atomic_offset_plan(context, source_op,
                                                             out_plan)) {
    return iree_ok_status();
  }
  return loom_amdgpu_select_vector_construct_plan_from_row(
      context, source_op, row, LOOM_AMDGPU_VECTOR_INSERT_SELECT_FLAG_NONE,
      /*require_fma_mix=*/false, out_plan);
}

iree_status_t loom_amdgpu_preselect_vector_construct_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan) {
  *out_plan = loom_low_lower_plan_empty();
  const loom_amdgpu_vector_construct_source_row_t* row =
      loom_amdgpu_vector_construct_source_row(source_op->kind);
  if (!loom_amdgpu_vector_construct_source_row_has_flag(
          row, LOOM_AMDGPU_VECTOR_CONSTRUCT_SOURCE_FLAG_PRESELECT_FMA_MIX)) {
    return iree_ok_status();
  }
  return loom_amdgpu_select_vector_construct_plan_from_row(
      context, source_op, row,
      LOOM_AMDGPU_VECTOR_INSERT_SELECT_FLAG_EMIT_DIAGNOSTICS,
      /*require_fma_mix=*/true, out_plan);
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
  const uint32_t lane_mask =
      iree_math_mask_low_bits_u32(UINT32_MAX, (int32_t)lane_bit_count)
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

  const uint32_t integer_bit_count =
      loom_amdgpu_integer_scalar_type_bit_count(plan->element_type);
  const bool mask_low_value = integer_bit_count > 0 && integer_bit_count < 32;
  if (mask_low_value) {
    IREE_ASSERT_EQ(integer_bit_count, plan->lane_bit_count);
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, low_value,
        iree_math_mask_low_bits_u32(UINT32_MAX, (int32_t)plan->lane_bit_count),
        register_type, &low_value));
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
    const uint32_t integer_bit_count =
        loom_amdgpu_integer_scalar_type_bit_count(plan->element_type);
    if (integer_bit_count > 0 && integer_bit_count < 32) {
      IREE_ASSERT_EQ(integer_bit_count, plan->lane_bit_count);
      loom_type_t lane_type = loom_type_none();
      IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
          low_value,
          iree_math_mask_low_bits_u32(UINT32_MAX,
                                      (int32_t)plan->lane_bit_count),
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

static void loom_amdgpu_require_fma_mix_plan_sources_storage(
    loom_low_lower_context_t* context, const loom_amdgpu_fma_mix_plan_t* plan) {
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(plan->sources); ++i) {
    if (plan->sources[i] == LOOM_VALUE_ID_INVALID) {
      continue;
    }
    loom_low_lower_require_source_value_storage(context, plan->sources[i]);
  }
}

void loom_amdgpu_mark_vector_construct_plan_storage_demands(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t plan) {
  if (plan.target_data == NULL) {
    return;
  }
  const loom_amdgpu_vector_construct_plan_header_t* header =
      (const loom_amdgpu_vector_construct_plan_header_t*)plan.target_data;
  switch (header->kind) {
    case LOOM_AMDGPU_VECTOR_CONSTRUCT_SOURCE_KIND_VECTOR_IOTA: {
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
    case LOOM_AMDGPU_VECTOR_CONSTRUCT_SOURCE_KIND_VECTOR_FROM_ELEMENTS:
    case LOOM_AMDGPU_VECTOR_CONSTRUCT_SOURCE_KIND_VECTOR_SPLAT: {
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
    case LOOM_AMDGPU_VECTOR_CONSTRUCT_SOURCE_KIND_VECTOR_INSERT: {
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
    case LOOM_AMDGPU_VECTOR_CONSTRUCT_SOURCE_KIND_NONE:
    default:
      IREE_ASSERT_UNREACHABLE("unknown AMDGPU vector construct plan kind");
      return;
  }
}

iree_status_t loom_amdgpu_lower_vector_construct_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t plan) {
  if (plan.target_data == NULL) {
    IREE_ASSERT_UNREACHABLE(
        "AMDGPU fact-only vector atomic offset reached emission");
    IREE_BUILTIN_UNREACHABLE();
  }
  const loom_amdgpu_vector_construct_plan_header_t* header =
      (const loom_amdgpu_vector_construct_plan_header_t*)plan.target_data;
  switch (header->kind) {
    case LOOM_AMDGPU_VECTOR_CONSTRUCT_SOURCE_KIND_VECTOR_IOTA:
      return loom_amdgpu_lower_vector_iota(
          context, source_op,
          (const loom_amdgpu_vector_iota_plan_t*)plan.target_data);
    case LOOM_AMDGPU_VECTOR_CONSTRUCT_SOURCE_KIND_VECTOR_FROM_ELEMENTS:
    case LOOM_AMDGPU_VECTOR_CONSTRUCT_SOURCE_KIND_VECTOR_SPLAT:
      return loom_amdgpu_lower_vector_from_elements(
          context, source_op,
          (const loom_amdgpu_vector_from_elements_plan_t*)plan.target_data);
    case LOOM_AMDGPU_VECTOR_CONSTRUCT_SOURCE_KIND_VECTOR_INSERT:
      return loom_amdgpu_lower_vector_insert(
          context, source_op,
          (const loom_amdgpu_vector_insert_plan_t*)plan.target_data);
    case LOOM_AMDGPU_VECTOR_CONSTRUCT_SOURCE_KIND_NONE:
    default:
      IREE_ASSERT_UNREACHABLE(
          "AMDGPU value plan selected unknown vector construct kind");
      IREE_BUILTIN_UNREACHABLE();
  }
}
