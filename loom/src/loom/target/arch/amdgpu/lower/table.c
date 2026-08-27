// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/table.h"

#include <stdint.h>

#include "loom/ir/context.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/legality.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/util/fact_table.h"

typedef uint8_t loom_amdgpu_table_lookup_lane_count_mode_t;
typedef uint8_t loom_amdgpu_table_lookup_descriptor_flags_t;

enum loom_amdgpu_table_lookup_lane_count_mode_e {
  LOOM_AMDGPU_TABLE_LOOKUP_LANE_COUNT_EXACT = 0,
  LOOM_AMDGPU_TABLE_LOOKUP_LANE_COUNT_ANY = 1,
  LOOM_AMDGPU_TABLE_LOOKUP_LANE_COUNT_RESULT = 2,
};

enum loom_amdgpu_table_lookup_descriptor_flag_bits_e {
  LOOM_AMDGPU_TABLE_LOOKUP_DESCRIPTOR_FLAG_LADDER = 1u << 0,
  LOOM_AMDGPU_TABLE_LOOKUP_DESCRIPTOR_FLAG_PERMUTE = 1u << 1,
};

typedef struct loom_amdgpu_table_lookup_shape_t {
  // Required scalar element type.
  loom_scalar_type_t element_type;
  // Lane count matching policy.
  loom_amdgpu_table_lookup_lane_count_mode_t lane_count_mode;
  // Exact lane count when lane_count_mode is EXACT.
  uint8_t lane_count;
  // Maximum accepted lane count for non-exact matches.
  uint8_t maximum_lane_count;
  // Required packed payload bit count, or 0 when not checked.
  uint16_t packed_payload_bit_count;
  // Required packed payload bits per lane, or 0 when not checked.
  uint8_t packed_lane_bit_count;
  // Required packed 32-bit register count, or 0 when not checked.
  uint8_t packed_register_count;
} loom_amdgpu_table_lookup_shape_t;

typedef struct loom_amdgpu_table_lookup_strategy_row_t {
  // Lowering strategy selected when all shapes match.
  loom_amdgpu_table_lookup_strategy_t strategy;
  // Index payload representation consumed by the emitter.
  loom_amdgpu_table_index_kind_t index_kind;
  // Register table vector shape.
  loom_amdgpu_table_lookup_shape_t table_shape;
  // Index vector shape.
  loom_amdgpu_table_lookup_shape_t index_shape;
  // Result vector shape.
  loom_amdgpu_table_lookup_shape_t result_shape;
  // Required unsigned fit for each index lane, or 0 when not checked.
  uint8_t index_lane_unsigned_bit_count;
  // Descriptor groups required by this strategy.
  loom_amdgpu_table_lookup_descriptor_flags_t descriptor_flags;
} loom_amdgpu_table_lookup_strategy_row_t;

typedef struct loom_amdgpu_table_lookup_descriptor_requirement_t {
  // Strategy descriptor flag that requires this descriptor ref.
  loom_amdgpu_table_lookup_descriptor_flags_t descriptor_flags;
  // Required descriptor ref for strategy availability.
  loom_amdgpu_descriptor_ref_t descriptor_ref;
} loom_amdgpu_table_lookup_descriptor_requirement_t;

typedef struct loom_amdgpu_table_lookup_type_summary_t {
  // Number of f32 vector lanes accepted by the table lookup lowering surface.
  uint32_t f32_lane_count;
  // Number of i32 vector lanes accepted by the table lookup lowering surface.
  uint32_t i32_lane_count;
  // Number of i8 vector lanes accepted by the table lookup lowering surface.
  uint32_t i8_lane_count;
  // Packed integer payload bit count, or 0 when the type is not packed.
  uint32_t packed_payload_bit_count;
  // Packed integer 32-bit register count, or 0 when the type is not packed.
  uint32_t packed_register_count;
} loom_amdgpu_table_lookup_type_summary_t;

#define LOOM_AMDGPU_TABLE_LOOKUP_SHAPE_ANY(element_type_value, max_lanes) \
  {                                                                       \
      .element_type = (element_type_value),                               \
      .lane_count_mode = LOOM_AMDGPU_TABLE_LOOKUP_LANE_COUNT_ANY,         \
      .maximum_lane_count = (max_lanes),                                  \
  }

#define LOOM_AMDGPU_TABLE_LOOKUP_SHAPE_RESULT(element_type_value, max_lanes) \
  {                                                                          \
      .element_type = (element_type_value),                                  \
      .lane_count_mode = LOOM_AMDGPU_TABLE_LOOKUP_LANE_COUNT_RESULT,         \
      .maximum_lane_count = (max_lanes),                                     \
  }

#define LOOM_AMDGPU_TABLE_LOOKUP_SHAPE_RESULT_PACKED_LANE(           \
    element_type_value, max_lanes, lane_bits)                        \
  {                                                                  \
      .element_type = (element_type_value),                          \
      .lane_count_mode = LOOM_AMDGPU_TABLE_LOOKUP_LANE_COUNT_RESULT, \
      .maximum_lane_count = (max_lanes),                             \
      .packed_lane_bit_count = (lane_bits),                          \
  }

#define LOOM_AMDGPU_TABLE_LOOKUP_SHAPE_EXACT_PACKED(                \
    element_type_value, exact_lanes, payload_bits, register_count)  \
  {                                                                 \
      .element_type = (element_type_value),                         \
      .lane_count_mode = LOOM_AMDGPU_TABLE_LOOKUP_LANE_COUNT_EXACT, \
      .lane_count = (exact_lanes),                                  \
      .packed_payload_bit_count = (payload_bits),                   \
      .packed_register_count = (register_count),                    \
  }

#define LOOM_AMDGPU_TABLE_LOOKUP_SHAPE_RESULT_PACKED(                \
    element_type_value, max_lanes, payload_bits, register_count)     \
  {                                                                  \
      .element_type = (element_type_value),                          \
      .lane_count_mode = LOOM_AMDGPU_TABLE_LOOKUP_LANE_COUNT_RESULT, \
      .maximum_lane_count = (max_lanes),                             \
      .packed_payload_bit_count = (payload_bits),                    \
      .packed_register_count = (register_count),                     \
  }

static const loom_amdgpu_table_lookup_strategy_row_t
    kTableLookupStrategyRows[] = {
        {
            .strategy = LOOM_AMDGPU_TABLE_LOOKUP_STRATEGY_F32_LADDER,
            .index_kind = LOOM_AMDGPU_TABLE_INDEX_KIND_I32,
            .table_shape = LOOM_AMDGPU_TABLE_LOOKUP_SHAPE_ANY(
                LOOM_SCALAR_TYPE_F32, LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES),
            .index_shape = LOOM_AMDGPU_TABLE_LOOKUP_SHAPE_RESULT(
                LOOM_SCALAR_TYPE_I32, LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES),
            .result_shape = LOOM_AMDGPU_TABLE_LOOKUP_SHAPE_ANY(
                LOOM_SCALAR_TYPE_F32, LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES),
            .descriptor_flags = LOOM_AMDGPU_TABLE_LOOKUP_DESCRIPTOR_FLAG_LADDER,
        },
        {
            .strategy = LOOM_AMDGPU_TABLE_LOOKUP_STRATEGY_F32_LADDER,
            .index_kind = LOOM_AMDGPU_TABLE_INDEX_KIND_PACKED_I8,
            .table_shape = LOOM_AMDGPU_TABLE_LOOKUP_SHAPE_ANY(
                LOOM_SCALAR_TYPE_F32, LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES),
            .index_shape = LOOM_AMDGPU_TABLE_LOOKUP_SHAPE_RESULT_PACKED_LANE(
                LOOM_SCALAR_TYPE_I8, LOOM_AMDGPU_MAX_PACKED_I8_LANES, 8),
            .result_shape = LOOM_AMDGPU_TABLE_LOOKUP_SHAPE_ANY(
                LOOM_SCALAR_TYPE_F32, LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES),
            .descriptor_flags = LOOM_AMDGPU_TABLE_LOOKUP_DESCRIPTOR_FLAG_LADDER,
        },
        {
            .strategy = LOOM_AMDGPU_TABLE_LOOKUP_STRATEGY_PACKED_I8_PERMUTE,
            .index_kind = LOOM_AMDGPU_TABLE_INDEX_KIND_PACKED_I8,
            .table_shape = LOOM_AMDGPU_TABLE_LOOKUP_SHAPE_EXACT_PACKED(
                LOOM_SCALAR_TYPE_I8, 4, 32, 1),
            .index_shape = LOOM_AMDGPU_TABLE_LOOKUP_SHAPE_RESULT_PACKED(
                LOOM_SCALAR_TYPE_I8, 4, 32, 1),
            .result_shape = LOOM_AMDGPU_TABLE_LOOKUP_SHAPE_EXACT_PACKED(
                LOOM_SCALAR_TYPE_I8, 4, 32, 1),
            .index_lane_unsigned_bit_count = 2,
            .descriptor_flags =
                LOOM_AMDGPU_TABLE_LOOKUP_DESCRIPTOR_FLAG_PERMUTE,
        },
        {
            .strategy = LOOM_AMDGPU_TABLE_LOOKUP_STRATEGY_PACKED_I8_U4_PERMUTE,
            .index_kind = LOOM_AMDGPU_TABLE_INDEX_KIND_PACKED_I8,
            .table_shape = LOOM_AMDGPU_TABLE_LOOKUP_SHAPE_EXACT_PACKED(
                LOOM_SCALAR_TYPE_I8, 16, 128, 4),
            .index_shape = LOOM_AMDGPU_TABLE_LOOKUP_SHAPE_RESULT_PACKED(
                LOOM_SCALAR_TYPE_I8, 4, 32, 1),
            .result_shape = LOOM_AMDGPU_TABLE_LOOKUP_SHAPE_EXACT_PACKED(
                LOOM_SCALAR_TYPE_I8, 4, 32, 1),
            .index_lane_unsigned_bit_count = 4,
            .descriptor_flags =
                LOOM_AMDGPU_TABLE_LOOKUP_DESCRIPTOR_FLAG_PERMUTE,
        },
};

static const loom_amdgpu_table_lookup_descriptor_requirement_t
    kTableLookupDescriptorRequirements[] = {
        {
            .descriptor_flags = LOOM_AMDGPU_TABLE_LOOKUP_DESCRIPTOR_FLAG_LADDER,
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
        },
        {
            .descriptor_flags = LOOM_AMDGPU_TABLE_LOOKUP_DESCRIPTOR_FLAG_LADDER,
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32,
        },
        {
            .descriptor_flags =
                LOOM_AMDGPU_TABLE_LOOKUP_DESCRIPTOR_FLAG_PERMUTE,
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_PERM_B32,
        },
};

#undef LOOM_AMDGPU_TABLE_LOOKUP_SHAPE_RESULT_PACKED
#undef LOOM_AMDGPU_TABLE_LOOKUP_SHAPE_EXACT_PACKED
#undef LOOM_AMDGPU_TABLE_LOOKUP_SHAPE_RESULT_PACKED_LANE
#undef LOOM_AMDGPU_TABLE_LOOKUP_SHAPE_RESULT
#undef LOOM_AMDGPU_TABLE_LOOKUP_SHAPE_ANY

static loom_amdgpu_table_lookup_type_summary_t
loom_amdgpu_table_lookup_type_summary(loom_type_t type) {
  loom_amdgpu_table_lookup_type_summary_t summary = {
      .f32_lane_count = loom_amdgpu_static_vector_lane_count(
          type, LOOM_SCALAR_TYPE_F32, LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES),
      .i32_lane_count = loom_amdgpu_static_vector_lane_count(
          type, LOOM_SCALAR_TYPE_I32, LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES),
      .i8_lane_count = loom_amdgpu_static_vector_lane_count(
          type, LOOM_SCALAR_TYPE_I8, LOOM_AMDGPU_MAX_PACKED_I8_LANES),
  };
  uint32_t packed_payload_bit_count = 0;
  uint32_t packed_register_count = 0;
  if (loom_amdgpu_type_packed_integer_storage(type, &packed_payload_bit_count,
                                              &packed_register_count)) {
    summary.packed_payload_bit_count = packed_payload_bit_count;
    summary.packed_register_count = packed_register_count;
  }
  return summary;
}

static uint32_t loom_amdgpu_table_lookup_summary_lane_count(
    const loom_amdgpu_table_lookup_type_summary_t* summary,
    loom_scalar_type_t element_type) {
  switch (element_type) {
    case LOOM_SCALAR_TYPE_F32:
      return summary->f32_lane_count;
    case LOOM_SCALAR_TYPE_I32:
      return summary->i32_lane_count;
    case LOOM_SCALAR_TYPE_I8:
      return summary->i8_lane_count;
    default:
      return 0;
  }
}

static bool loom_amdgpu_table_lookup_shape_lane_count_matches(
    const loom_amdgpu_table_lookup_shape_t* shape, uint32_t lane_count,
    uint32_t result_lane_count) {
  if (shape->maximum_lane_count != 0 &&
      lane_count > shape->maximum_lane_count) {
    return false;
  }
  switch (shape->lane_count_mode) {
    case LOOM_AMDGPU_TABLE_LOOKUP_LANE_COUNT_EXACT:
      return lane_count == shape->lane_count;
    case LOOM_AMDGPU_TABLE_LOOKUP_LANE_COUNT_ANY:
      return lane_count != 0;
    case LOOM_AMDGPU_TABLE_LOOKUP_LANE_COUNT_RESULT:
      return lane_count == result_lane_count;
    default:
      return false;
  }
}

static bool loom_amdgpu_table_lookup_shape_matches(
    const loom_amdgpu_table_lookup_shape_t* shape,
    const loom_amdgpu_table_lookup_type_summary_t* summary,
    uint32_t result_lane_count, uint32_t* out_lane_count,
    uint32_t* out_register_count) {
  if (out_lane_count != NULL) {
    *out_lane_count = 0;
  }
  if (out_register_count != NULL) {
    *out_register_count = 0;
  }
  const uint32_t lane_count =
      loom_amdgpu_table_lookup_summary_lane_count(summary, shape->element_type);
  if (!loom_amdgpu_table_lookup_shape_lane_count_matches(shape, lane_count,
                                                         result_lane_count)) {
    return false;
  }

  const bool requires_packed_storage = shape->packed_payload_bit_count != 0 ||
                                       shape->packed_lane_bit_count != 0 ||
                                       shape->packed_register_count != 0;
  if (!requires_packed_storage) {
    if (out_lane_count != NULL) {
      *out_lane_count = lane_count;
    }
    if (out_register_count != NULL) {
      *out_register_count = lane_count;
    }
    return true;
  }

  if (summary->packed_payload_bit_count == 0 ||
      summary->packed_register_count == 0) {
    return false;
  }
  if (shape->packed_payload_bit_count != 0 &&
      summary->packed_payload_bit_count != shape->packed_payload_bit_count) {
    return false;
  }
  if (shape->packed_lane_bit_count != 0 &&
      summary->packed_payload_bit_count !=
          lane_count * shape->packed_lane_bit_count) {
    return false;
  }
  if (shape->packed_register_count != 0 &&
      summary->packed_register_count != shape->packed_register_count) {
    return false;
  }

  if (out_lane_count != NULL) {
    *out_lane_count = lane_count;
  }
  if (out_register_count != NULL) {
    *out_register_count = summary->packed_register_count;
  }
  return true;
}

static bool loom_amdgpu_table_lookup_index_lane_fits_unsigned_bit_count(
    loom_value_facts_t facts, uint32_t bit_count) {
  return loom_value_facts_fit_unsigned_bit_count(facts, bit_count);
}

static bool loom_amdgpu_table_lookup_indices_fit_unsigned_bit_count(
    const loom_value_fact_table_t* fact_table, loom_value_id_t indices,
    uint32_t lane_count, uint32_t bit_count) {
  if (fact_table == NULL) {
    return false;
  }

  const loom_value_facts_t index_facts =
      loom_value_fact_table_lookup(fact_table, indices);
  loom_value_fact_uniform_element_t uniform = {0};
  if (loom_value_facts_query_uniform_element(&fact_table->context, index_facts,
                                             &uniform)) {
    return loom_amdgpu_table_lookup_index_lane_fits_unsigned_bit_count(
        uniform.element, bit_count);
  }

  loom_value_fact_small_static_lanes_t lanes = {0};
  if (loom_value_facts_query_small_static_lanes(&fact_table->context,
                                                index_facts, &lanes)) {
    if (lanes.count < lane_count) {
      return false;
    }
    for (uint32_t i = 0; i < lane_count; ++i) {
      if (!loom_amdgpu_table_lookup_index_lane_fits_unsigned_bit_count(
              lanes.lanes[i], bit_count)) {
        return false;
      }
    }
    return true;
  }

  return loom_amdgpu_table_lookup_index_lane_fits_unsigned_bit_count(
      index_facts, bit_count);
}

static bool loom_amdgpu_table_lookup_plan_from_op(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_op_t* source_op,
    const loom_amdgpu_table_lookup_strategy_row_t** out_row,
    loom_amdgpu_table_lookup_plan_t* out_plan) {
  *out_row = NULL;
  *out_plan = (loom_amdgpu_table_lookup_plan_t){0};
  if (!loom_vector_table_lookup_isa(source_op)) {
    return false;
  }

  out_plan->table = loom_vector_table_lookup_table(source_op);
  out_plan->indices = loom_vector_table_lookup_indices(source_op);
  out_plan->result = loom_vector_table_lookup_result(source_op);

  const loom_type_t table_type =
      loom_module_value_type(module, out_plan->table);
  const loom_type_t indices_type =
      loom_module_value_type(module, out_plan->indices);
  const loom_type_t result_type =
      loom_module_value_type(module, out_plan->result);

  const loom_amdgpu_table_lookup_type_summary_t table_summary =
      loom_amdgpu_table_lookup_type_summary(table_type);
  const loom_amdgpu_table_lookup_type_summary_t indices_summary =
      loom_amdgpu_table_lookup_type_summary(indices_type);
  const loom_amdgpu_table_lookup_type_summary_t result_summary =
      loom_amdgpu_table_lookup_type_summary(result_type);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kTableLookupStrategyRows);
       ++i) {
    const loom_amdgpu_table_lookup_strategy_row_t* row =
        &kTableLookupStrategyRows[i];
    uint32_t result_lane_count = 0;
    if (!loom_amdgpu_table_lookup_shape_matches(
            &row->result_shape, &result_summary, /*result_lane_count=*/0,
            &result_lane_count, /*out_register_count=*/NULL)) {
      continue;
    }
    uint32_t table_lane_count = 0;
    uint32_t table_register_count = 0;
    if (!loom_amdgpu_table_lookup_shape_matches(
            &row->table_shape, &table_summary, result_lane_count,
            &table_lane_count, &table_register_count)) {
      continue;
    }
    uint32_t index_register_count = 0;
    if (!loom_amdgpu_table_lookup_shape_matches(
            &row->index_shape, &indices_summary, result_lane_count,
            /*out_lane_count=*/NULL, &index_register_count)) {
      continue;
    }
    if (row->index_lane_unsigned_bit_count != 0 &&
        !loom_amdgpu_table_lookup_indices_fit_unsigned_bit_count(
            fact_table, out_plan->indices, result_lane_count,
            row->index_lane_unsigned_bit_count)) {
      continue;
    }

    out_plan->strategy = row->strategy;
    out_plan->index_kind = row->index_kind;
    out_plan->table_lane_count = table_lane_count;
    out_plan->table_register_count = table_register_count;
    out_plan->result_lane_count = result_lane_count;
    out_plan->index_register_count = index_register_count;
    *out_row = row;
    return true;
  }
  return false;
}

iree_status_t loom_amdgpu_select_vector_table_lookup_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_table_lookup_plan_t* out_plan, bool* out_selected) {
  *out_selected = false;
  const loom_amdgpu_table_lookup_strategy_row_t* row = NULL;
  if (!loom_amdgpu_table_lookup_plan_from_op(
          loom_low_lower_context_module(context),
          loom_low_lower_context_fact_table(context), source_op, &row,
          out_plan)) {
    return iree_ok_status();
  }

  if (iree_any_bit_set(row->descriptor_flags,
                       LOOM_AMDGPU_TABLE_LOOKUP_DESCRIPTOR_FLAG_PERMUTE)) {
    bool descriptor_present = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
        context, LOOM_AMDGPU_DESCRIPTOR_REF_V_PERM_B32,
        &out_plan->permute_descriptor, &descriptor_present));
    *out_selected = descriptor_present;
    return iree_ok_status();
  }

  if (iree_any_bit_set(row->descriptor_flags,
                       LOOM_AMDGPU_TABLE_LOOKUP_DESCRIPTOR_FLAG_LADDER)) {
    bool compare_descriptor_present = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
        context, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
        &out_plan->compare_register_descriptor, &compare_descriptor_present));
    loom_amdgpu_cndmask_b32_descriptors_t select_descriptors = {0};
    bool select_descriptor_present = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_cndmask_b32_descriptors(
        context, LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_REGISTER,
        LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_SRC1_LITERAL, &select_descriptors,
        &select_descriptor_present));
    if (!compare_descriptor_present || !select_descriptor_present) {
      return iree_ok_status();
    }
    out_plan->select_register_descriptor =
        select_descriptors.register_descriptor;
    out_plan->select_src1_literal_descriptor =
        select_descriptors.src1_literal_descriptor;

    bool optional_descriptor_present = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
        context, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32_SRC1_INLINE,
        &out_plan->compare_src1_inline_descriptor,
        &optional_descriptor_present));

    *out_selected = true;
    return iree_ok_status();
  }
  return iree_ok_status();
}

static bool loom_amdgpu_table_lookup_strategy_descriptors_present(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_table_lookup_strategy_row_t* row) {
  bool matched_requirement = false;
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kTableLookupDescriptorRequirements); ++i) {
    const loom_amdgpu_table_lookup_descriptor_requirement_t* requirement =
        &kTableLookupDescriptorRequirements[i];
    if (!iree_any_bit_set(row->descriptor_flags,
                          requirement->descriptor_flags)) {
      continue;
    }
    matched_requirement = true;
    if (!loom_amdgpu_descriptor_set_has_ref(descriptor_set,
                                            requirement->descriptor_ref)) {
      return false;
    }
  }
  return matched_requirement;
}

static iree_status_t loom_amdgpu_table_lookup_extract_i32_index_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_table_lookup_plan_t* plan, loom_value_id_t low_indices,
    uint32_t result_lane, loom_type_t lane_type,
    loom_value_id_t* out_index_lane) {
  return loom_amdgpu_extract_low_register_unit(
      context, source_op, low_indices, plan->index_register_count, result_lane,
      lane_type, out_index_lane);
}

static bool loom_amdgpu_table_lookup_i8_extract_prefers_bfe(
    uint32_t byte_offset) {
  return byte_offset != 0;
}

static iree_status_t loom_amdgpu_table_lookup_extract_i8_index_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_table_lookup_plan_t* plan, loom_value_id_t low_indices,
    uint32_t result_lane, loom_type_t lane_type,
    loom_value_id_t* out_index_lane) {
  *out_index_lane = LOOM_VALUE_ID_INVALID;
  const uint32_t register_offset = result_lane / 4u;
  const uint32_t byte_offset = result_lane & 3u;
  loom_value_id_t source_register = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
      context, source_op, low_indices, plan->index_register_count,
      register_offset, lane_type, &source_register));

  if (loom_amdgpu_table_lookup_i8_extract_prefers_bfe(byte_offset)) {
    bool selected_bfe = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_try_emit_vgpr_b32_bfe_extract(
        context, source_op, source_register, byte_offset * 8u, 8u,
        LOOM_AMDGPU_VGPR_BFE_EXTRACT_FLAG_NONE, lane_type, out_index_lane,
        &selected_bfe));
    if (selected_bfe) {
      return iree_ok_status();
    }
  }

  loom_value_id_t shifted = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
      byte_offset * 8u, source_register, lane_type, &shifted));

  return loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, shifted,
      0xFFu, lane_type, out_index_lane);
}

static iree_status_t loom_amdgpu_table_lookup_extract_index_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_table_lookup_plan_t* plan, loom_value_id_t low_indices,
    uint32_t result_lane, loom_type_t lane_type,
    loom_value_id_t* out_index_lane) {
  if (plan->index_kind == LOOM_AMDGPU_TABLE_INDEX_KIND_I32) {
    return loom_amdgpu_table_lookup_extract_i32_index_lane(
        context, source_op, plan, low_indices, result_lane, lane_type,
        out_index_lane);
  }
  return loom_amdgpu_table_lookup_extract_i8_index_lane(
      context, source_op, plan, low_indices, result_lane, lane_type,
      out_index_lane);
}

static iree_status_t loom_amdgpu_table_lookup_extract_table_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_table_lookup_plan_t* plan, loom_value_id_t low_table,
    uint32_t table_lane, loom_type_t lane_type,
    loom_value_id_t* out_table_lane) {
  return loom_amdgpu_extract_low_register_unit(
      context, source_op, low_table, plan->table_lane_count, table_lane,
      lane_type, out_table_lane);
}

static iree_status_t loom_amdgpu_table_lookup_emit_index_compare(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_table_lookup_plan_t* plan, loom_value_id_t index_lane,
    loom_value_id_t ordinal_lane, uint32_t ordinal, loom_type_t mask_lane_type,
    loom_value_id_t* out_mask_lane) {
  *out_mask_lane = LOOM_VALUE_ID_INVALID;
  loom_named_attr_t attrs[1] = {0};
  iree_host_size_t attr_count = 0;
  const loom_low_lower_resolved_descriptor_t* descriptor =
      &plan->compare_register_descriptor;
  loom_value_id_t operands[2] = {index_lane, ordinal_lane};
  iree_host_size_t operand_count = 2;
  if (plan->compare_src1_inline_descriptor.descriptor != NULL &&
      ordinal <= LOOM_AMDGPU_SOURCE_INLINE_U32_MAX) {
    descriptor = &plan->compare_src1_inline_descriptor;
    operand_count = 1;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_append_i64_attr(context, IREE_SV("rhs"), ordinal, attrs,
                                    IREE_ARRAYSIZE(attrs), &attr_count));
  }
  loom_op_t* compare_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, operand_count,
      loom_make_named_attr_slice(attrs, attr_count), &mask_lane_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &compare_op));
  *out_mask_lane = loom_value_slice_get(loom_low_op_results(compare_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_table_lookup_emit_table_select(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_table_lookup_plan_t* plan, loom_value_id_t false_lane,
    loom_value_id_t true_lane, loom_value_id_t condition, uint32_t table_lane,
    loom_type_t lane_type, loom_value_id_t* out_selected_lane) {
  *out_selected_lane = LOOM_VALUE_ID_INVALID;
  loom_named_attr_t attrs[1] = {0};
  iree_host_size_t attr_count = 0;
  const loom_low_lower_resolved_descriptor_t* descriptor =
      &plan->select_register_descriptor;
  loom_value_id_t operands[3] = {false_lane, true_lane, condition};
  iree_host_size_t operand_count = 3;
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  uint32_t table_bits = 0;
  if (plan->select_src1_literal_descriptor.descriptor != NULL &&
      loom_amdgpu_source_lane_as_u32_bits(
          fact_table, loom_low_lower_context_module(context), plan->table,
          table_lane, &table_bits)) {
    descriptor = &plan->select_src1_literal_descriptor;
    operands[1] = condition;
    operand_count = 2;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_append_i64_attr(context, IREE_SV("imm32"), table_bits,
                                    attrs, IREE_ARRAYSIZE(attrs), &attr_count));
  }
  loom_op_t* select_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, operand_count,
      loom_make_named_attr_slice(attrs, attr_count), &lane_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &select_op));
  *out_selected_lane = loom_value_slice_get(loom_low_op_results(select_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_table_lookup_select_table_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_table_lookup_plan_t* plan,
    const loom_value_id_t* table_lanes, const loom_value_id_t* ordinals,
    loom_value_id_t index_lane, loom_type_t lane_type,
    loom_type_t mask_lane_type, loom_value_id_t* out_selected_lane) {
  *out_selected_lane = LOOM_VALUE_ID_INVALID;
  loom_value_id_t selected_lane = table_lanes[0];
  for (uint32_t i = 1; i < plan->table_lane_count; ++i) {
    loom_value_id_t condition = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_table_lookup_emit_index_compare(
        context, source_op, plan, index_lane, ordinals[i], i, mask_lane_type,
        &condition));
    IREE_RETURN_IF_ERROR(loom_amdgpu_table_lookup_emit_table_select(
        context, source_op, plan, selected_lane, table_lanes[i], condition, i,
        lane_type, &selected_lane));
  }
  *out_selected_lane = selected_lane;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_lower_vector_table_lookup_packed_i8_permute(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_table_lookup_plan_t* plan, loom_value_id_t low_table,
    loom_value_id_t low_indices) {
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op,
                                                   plan->result, &result_type));
  const loom_value_id_t operands[3] = {low_table, low_table, low_indices};
  loom_op_t* permute_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->permute_descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), &result_type, 1, /*tied_results=*/NULL,
      /*tied_result_count=*/0, source_op->location, &permute_op));
  return loom_low_lower_bind_value(
      context, plan->result,
      loom_value_slice_get(loom_low_op_results(permute_op), 0));
}

static iree_status_t loom_amdgpu_lower_vector_table_lookup_packed_i8_u4_permute(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_table_lookup_plan_t* plan, loom_value_id_t low_table,
    loom_value_id_t low_indices) {
  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  loom_value_id_t table_registers[4] = {0};
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(table_registers); ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
        context, source_op, low_table, plan->table_register_count, i, lane_type,
        &table_registers[i]));
  }

  loom_value_id_t low_selector = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, low_indices,
      UINT32_C(0x07070707), lane_type, &low_selector));

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op,
                                                   plan->result, &result_type));
  const loom_value_id_t low_lookup_operands[3] = {
      table_registers[0],
      table_registers[1],
      low_selector,
  };
  loom_op_t* low_lookup_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->permute_descriptor, low_lookup_operands,
      IREE_ARRAYSIZE(low_lookup_operands), loom_named_attr_slice_empty(),
      &result_type, 1, /*tied_results=*/NULL, /*tied_result_count=*/0,
      source_op->location, &low_lookup_op));

  const loom_value_id_t high_lookup_operands[3] = {
      table_registers[2],
      table_registers[3],
      low_selector,
  };
  loom_op_t* high_lookup_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->permute_descriptor, high_lookup_operands,
      IREE_ARRAYSIZE(high_lookup_operands), loom_named_attr_slice_empty(),
      &result_type, 1, /*tied_results=*/NULL, /*tied_result_count=*/0,
      source_op->location, &high_lookup_op));

  loom_value_id_t high_table_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, low_indices,
      UINT32_C(0x08080808), lane_type, &high_table_bits));
  loom_value_id_t high_table_selector_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT, 1,
      high_table_bits, lane_type, &high_table_selector_offset));
  loom_value_id_t merge_selector = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32_LIT,
      high_table_selector_offset, UINT32_C(0x03020100), lane_type,
      &merge_selector));

  const loom_value_id_t merge_operands[3] = {
      loom_value_slice_get(loom_low_op_results(low_lookup_op), 0),
      loom_value_slice_get(loom_low_op_results(high_lookup_op), 0),
      merge_selector,
  };
  loom_op_t* merge_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->permute_descriptor, merge_operands,
      IREE_ARRAYSIZE(merge_operands), loom_named_attr_slice_empty(),
      &result_type, 1, /*tied_results=*/NULL, /*tied_result_count=*/0,
      source_op->location, &merge_op));
  return loom_low_lower_bind_value(
      context, plan->result,
      loom_value_slice_get(loom_low_op_results(merge_op), 0));
}

iree_status_t loom_amdgpu_lower_vector_table_lookup(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_table_lookup_plan_t* plan) {
  loom_value_id_t low_table = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->table, &low_table));
  loom_value_id_t low_indices = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->indices, &low_indices));

  if (plan->strategy == LOOM_AMDGPU_TABLE_LOOKUP_STRATEGY_PACKED_I8_PERMUTE) {
    return loom_amdgpu_lower_vector_table_lookup_packed_i8_permute(
        context, source_op, plan, low_table, low_indices);
  }
  if (plan->strategy ==
      LOOM_AMDGPU_TABLE_LOOKUP_STRATEGY_PACKED_I8_U4_PERMUTE) {
    return loom_amdgpu_lower_vector_table_lookup_packed_i8_u4_permute(
        context, source_op, plan, low_table, low_indices);
  }

  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  loom_type_t mask_lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &mask_lane_type));
  loom_value_id_t table_lanes[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  loom_value_id_t ordinals[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES] = {0};
  const bool use_ordinal_registers =
      plan->compare_src1_inline_descriptor.descriptor == NULL;
  for (uint32_t i = 0; i < plan->table_lane_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_table_lookup_extract_table_lane(
        context, source_op, plan, low_table, i, lane_type, &table_lanes[i]));
    if (i == 0 || !use_ordinal_registers) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, i, lane_type,
        &ordinals[i]));
  }

  loom_value_id_t result_lanes[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < plan->result_lane_count; ++i) {
    loom_value_id_t index_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_table_lookup_extract_index_lane(
        context, source_op, plan, low_indices, i, lane_type, &index_lane));
    IREE_RETURN_IF_ERROR(loom_amdgpu_table_lookup_select_table_lane(
        context, source_op, plan, table_lanes, ordinals, index_lane, lane_type,
        mask_lane_type, &result_lanes[i]));
  }

  return loom_amdgpu_bind_low_register_range(
      context, source_op, plan->result, result_lanes, plan->result_lane_count);
}

iree_status_t loom_amdgpu_low_legality_verify_vector_table(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  const loom_module_t* module = loom_target_low_legality_module(context);
  loom_amdgpu_table_lookup_plan_t unused_plan = {0};
  const loom_amdgpu_table_lookup_strategy_row_t* row = NULL;
  if (loom_amdgpu_table_lookup_plan_from_op(
          module, loom_target_low_legality_fact_table(context), op, &row,
          &unused_plan) &&
      loom_amdgpu_table_lookup_strategy_descriptors_present(
          loom_target_low_legality_descriptor_set(context), row)) {
    return iree_ok_status();
  }
  return loom_amdgpu_low_legality_reject(context, op,
                                         IREE_SV("table_lookup.shape"));
}
