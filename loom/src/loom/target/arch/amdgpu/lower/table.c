// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/table.h"

#include <stdint.h>
#include <string.h>

#include "loom/ir/context.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/lower/bitpack.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/legality.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/util/fact_table.h"

typedef uint8_t loom_amdgpu_table_lookup_lane_count_mode_t;
typedef uint8_t loom_amdgpu_table_lookup_descriptor_flags_t;

enum loom_amdgpu_table_lookup_lane_count_mode_e {
  LOOM_AMDGPU_TABLE_LOOKUP_LANE_COUNT_ANY = 0,
  LOOM_AMDGPU_TABLE_LOOKUP_LANE_COUNT_RESULT = 1,
};

enum loom_amdgpu_table_lookup_descriptor_flag_bits_e {
  LOOM_AMDGPU_TABLE_LOOKUP_DESCRIPTOR_FLAG_LADDER = 1u << 0,
  LOOM_AMDGPU_TABLE_LOOKUP_DESCRIPTOR_FLAG_PERMUTE = 1u << 1,
};

typedef struct loom_amdgpu_table_lookup_shape_t {
  // Accepted scalar element types.
  loom_scalar_type_set_t element_types;
  // Lane count matching policy.
  loom_amdgpu_table_lookup_lane_count_mode_t lane_count_mode;
  // Maximum accepted lane count for non-exact matches.
  uint8_t maximum_lane_count;
  // Required packed payload bits per lane, or 0 when not checked.
  uint8_t packed_lane_bit_count;
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
  // Whether every index lane must be proven inside the authored table extent.
  bool indices_must_fit_table;
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
  // Scalar element type carried by the vector.
  loom_scalar_type_t element_type;
  // Static rank-one vector lane count, or 0 when unsupported.
  uint32_t lane_count;
  // Packed payload bit count, or 0 when the type is not packed.
  uint32_t packed_payload_bit_count;
  // Packed 32-bit register count, or 0 when the type is not packed.
  uint32_t packed_register_count;
} loom_amdgpu_table_lookup_type_summary_t;

#define LOOM_AMDGPU_TABLE_LOOKUP_SHAPE_ANY(element_type_set, max_lanes) \
  {                                                                     \
      .element_types = (element_type_set),                              \
      .lane_count_mode = LOOM_AMDGPU_TABLE_LOOKUP_LANE_COUNT_ANY,       \
      .maximum_lane_count = (max_lanes),                                \
  }

#define LOOM_AMDGPU_TABLE_LOOKUP_SHAPE_RESULT(element_type_set, max_lanes) \
  {                                                                        \
      .element_types = (element_type_set),                                 \
      .lane_count_mode = LOOM_AMDGPU_TABLE_LOOKUP_LANE_COUNT_RESULT,       \
      .maximum_lane_count = (max_lanes),                                   \
  }

#define LOOM_AMDGPU_TABLE_LOOKUP_SHAPE_RESULT_PACKED_LANE(           \
    element_type_set, max_lanes, lane_bits)                          \
  {                                                                  \
      .element_types = (element_type_set),                           \
      .lane_count_mode = LOOM_AMDGPU_TABLE_LOOKUP_LANE_COUNT_RESULT, \
      .maximum_lane_count = (max_lanes),                             \
      .packed_lane_bit_count = (lane_bits),                          \
  }

#define LOOM_AMDGPU_TABLE_LOOKUP_SHAPE_ANY_PACKED_LANE(element_type_set,     \
                                                       max_lanes, lane_bits) \
  {                                                                          \
      .element_types = (element_type_set),                                   \
      .lane_count_mode = LOOM_AMDGPU_TABLE_LOOKUP_LANE_COUNT_ANY,            \
      .maximum_lane_count = (max_lanes),                                     \
      .packed_lane_bit_count = (lane_bits),                                  \
  }

#define LOOM_AMDGPU_TABLE_LOOKUP_PACKED_BYTE_TYPES         \
  (LOOM_SCALAR_TYPE_SET_I8 | LOOM_SCALAR_TYPE_SET_F8E4M3 | \
   LOOM_SCALAR_TYPE_SET_F8E5M2)

static const loom_amdgpu_table_lookup_strategy_row_t
    kTableLookupStrategyRows[] = {
        {
            .strategy = LOOM_AMDGPU_TABLE_LOOKUP_STRATEGY_F32_LADDER,
            .index_kind = LOOM_AMDGPU_TABLE_INDEX_KIND_I32,
            .table_shape = LOOM_AMDGPU_TABLE_LOOKUP_SHAPE_ANY(
                LOOM_SCALAR_TYPE_SET_F32,
                LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES),
            .index_shape = LOOM_AMDGPU_TABLE_LOOKUP_SHAPE_RESULT(
                LOOM_SCALAR_TYPE_SET_I32,
                LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES),
            .result_shape = LOOM_AMDGPU_TABLE_LOOKUP_SHAPE_ANY(
                LOOM_SCALAR_TYPE_SET_F32,
                LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES),
            .descriptor_flags = LOOM_AMDGPU_TABLE_LOOKUP_DESCRIPTOR_FLAG_LADDER,
        },
        {
            .strategy = LOOM_AMDGPU_TABLE_LOOKUP_STRATEGY_F32_LADDER,
            .index_kind = LOOM_AMDGPU_TABLE_INDEX_KIND_PACKED_I8,
            .table_shape = LOOM_AMDGPU_TABLE_LOOKUP_SHAPE_ANY(
                LOOM_SCALAR_TYPE_SET_F32,
                LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES),
            .index_shape = LOOM_AMDGPU_TABLE_LOOKUP_SHAPE_RESULT_PACKED_LANE(
                LOOM_SCALAR_TYPE_SET_I8, LOOM_AMDGPU_MAX_PACKED_I8_LANES, 8),
            .result_shape = LOOM_AMDGPU_TABLE_LOOKUP_SHAPE_ANY(
                LOOM_SCALAR_TYPE_SET_F32,
                LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES),
            .descriptor_flags = LOOM_AMDGPU_TABLE_LOOKUP_DESCRIPTOR_FLAG_LADDER,
        },
        {
            .strategy =
                LOOM_AMDGPU_TABLE_LOOKUP_STRATEGY_PACKED_BYTE_PERMUTE_TREE,
            .index_kind = LOOM_AMDGPU_TABLE_INDEX_KIND_PACKED_I8,
            .table_shape = LOOM_AMDGPU_TABLE_LOOKUP_SHAPE_ANY_PACKED_LANE(
                LOOM_AMDGPU_TABLE_LOOKUP_PACKED_BYTE_TYPES,
                LOOM_AMDGPU_MAX_PACKED_I8_LANES, 8),
            .index_shape = LOOM_AMDGPU_TABLE_LOOKUP_SHAPE_RESULT_PACKED_LANE(
                LOOM_SCALAR_TYPE_SET_I8, LOOM_AMDGPU_MAX_PACKED_I8_LANES, 8),
            .result_shape = LOOM_AMDGPU_TABLE_LOOKUP_SHAPE_ANY_PACKED_LANE(
                LOOM_AMDGPU_TABLE_LOOKUP_PACKED_BYTE_TYPES,
                LOOM_AMDGPU_MAX_PACKED_I8_LANES, 8),
            .indices_must_fit_table = true,
            .descriptor_flags =
                LOOM_AMDGPU_TABLE_LOOKUP_DESCRIPTOR_FLAG_PERMUTE,
        },
        {
            .strategy =
                LOOM_AMDGPU_TABLE_LOOKUP_STRATEGY_PACKED_BYTE_PERMUTE_TREE,
            .index_kind = LOOM_AMDGPU_TABLE_INDEX_KIND_PACKED_I16,
            .table_shape = LOOM_AMDGPU_TABLE_LOOKUP_SHAPE_ANY_PACKED_LANE(
                LOOM_AMDGPU_TABLE_LOOKUP_PACKED_BYTE_TYPES,
                LOOM_AMDGPU_MAX_PACKED_I8_LANES, 8),
            .index_shape = LOOM_AMDGPU_TABLE_LOOKUP_SHAPE_RESULT_PACKED_LANE(
                LOOM_SCALAR_TYPE_SET_I16, LOOM_AMDGPU_MAX_PACKED_I16_LANES, 16),
            .result_shape = LOOM_AMDGPU_TABLE_LOOKUP_SHAPE_ANY_PACKED_LANE(
                LOOM_AMDGPU_TABLE_LOOKUP_PACKED_BYTE_TYPES,
                LOOM_AMDGPU_MAX_PACKED_I8_LANES, 8),
            .indices_must_fit_table = true,
            .descriptor_flags =
                LOOM_AMDGPU_TABLE_LOOKUP_DESCRIPTOR_FLAG_PERMUTE,
        },
        {
            .strategy =
                LOOM_AMDGPU_TABLE_LOOKUP_STRATEGY_PACKED_BYTE_PERMUTE_TREE,
            .index_kind = LOOM_AMDGPU_TABLE_INDEX_KIND_I32,
            .table_shape = LOOM_AMDGPU_TABLE_LOOKUP_SHAPE_ANY_PACKED_LANE(
                LOOM_AMDGPU_TABLE_LOOKUP_PACKED_BYTE_TYPES,
                LOOM_AMDGPU_MAX_PACKED_I8_LANES, 8),
            .index_shape = LOOM_AMDGPU_TABLE_LOOKUP_SHAPE_RESULT(
                LOOM_SCALAR_TYPE_SET_I32,
                LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES),
            .result_shape = LOOM_AMDGPU_TABLE_LOOKUP_SHAPE_ANY_PACKED_LANE(
                LOOM_AMDGPU_TABLE_LOOKUP_PACKED_BYTE_TYPES,
                LOOM_AMDGPU_MAX_PACKED_I8_LANES, 8),
            .indices_must_fit_table = true,
            .descriptor_flags =
                LOOM_AMDGPU_TABLE_LOOKUP_DESCRIPTOR_FLAG_PERMUTE,
        },
        {
            .strategy =
                LOOM_AMDGPU_TABLE_LOOKUP_STRATEGY_PACKED_BYTE_PERMUTE_TREE,
            .index_kind = LOOM_AMDGPU_TABLE_INDEX_KIND_I64,
            .table_shape = LOOM_AMDGPU_TABLE_LOOKUP_SHAPE_ANY_PACKED_LANE(
                LOOM_AMDGPU_TABLE_LOOKUP_PACKED_BYTE_TYPES,
                LOOM_AMDGPU_MAX_PACKED_I8_LANES, 8),
            .index_shape = LOOM_AMDGPU_TABLE_LOOKUP_SHAPE_RESULT(
                LOOM_SCALAR_TYPE_SET_I64,
                LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES / 2u),
            .result_shape = LOOM_AMDGPU_TABLE_LOOKUP_SHAPE_ANY_PACKED_LANE(
                LOOM_AMDGPU_TABLE_LOOKUP_PACKED_BYTE_TYPES,
                LOOM_AMDGPU_MAX_PACKED_I8_LANES, 8),
            .indices_must_fit_table = true,
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

#undef LOOM_AMDGPU_TABLE_LOOKUP_SHAPE_ANY_PACKED_LANE
#undef LOOM_AMDGPU_TABLE_LOOKUP_SHAPE_RESULT_PACKED_LANE
#undef LOOM_AMDGPU_TABLE_LOOKUP_SHAPE_RESULT
#undef LOOM_AMDGPU_TABLE_LOOKUP_SHAPE_ANY
#undef LOOM_AMDGPU_TABLE_LOOKUP_PACKED_BYTE_TYPES

static loom_amdgpu_table_lookup_type_summary_t
loom_amdgpu_table_lookup_type_summary(loom_type_t type) {
  loom_amdgpu_table_lookup_type_summary_t summary = {0};
  if (!loom_type_is_vector(type)) {
    return summary;
  }
  summary.element_type = loom_type_element_type(type);
  summary.lane_count = loom_amdgpu_static_vector_lane_count(
      type, summary.element_type, LOOM_AMDGPU_MAX_VECTOR_STORAGE_ELEMENTS);

  loom_amdgpu_vector_storage_t storage = {0};
  if (loom_amdgpu_type_vector_storage(type, &storage) &&
      iree_any_bit_set(loom_amdgpu_vector_storage_kind_flags(storage.kind),
                       LOOM_AMDGPU_VECTOR_STORAGE_KIND_FLAG_PACKED_PAYLOAD)) {
    summary.packed_payload_bit_count =
        storage.element_count * storage.element_bit_count;
    summary.packed_register_count = storage.register_count;
  }
  return summary;
}

static bool loom_amdgpu_table_lookup_shape_lane_count_matches(
    const loom_amdgpu_table_lookup_shape_t* shape, uint32_t lane_count,
    uint32_t result_lane_count) {
  if (shape->maximum_lane_count != 0 &&
      lane_count > shape->maximum_lane_count) {
    return false;
  }
  switch (shape->lane_count_mode) {
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
  if (!loom_scalar_type_set_contains(shape->element_types,
                                     summary->element_type)) {
    return false;
  }
  const uint32_t lane_count = summary->lane_count;
  if (!loom_amdgpu_table_lookup_shape_lane_count_matches(shape, lane_count,
                                                         result_lane_count)) {
    return false;
  }

  const bool requires_packed_storage = shape->packed_lane_bit_count != 0;
  if (!requires_packed_storage) {
    if (out_lane_count != NULL) {
      *out_lane_count = lane_count;
    }
    if (out_register_count != NULL) {
      *out_register_count = summary->element_type == LOOM_SCALAR_TYPE_I64
                                ? lane_count * 2u
                                : lane_count;
    }
    return true;
  }

  if (summary->packed_payload_bit_count == 0 ||
      summary->packed_register_count == 0) {
    return false;
  }
  if (summary->packed_payload_bit_count !=
      lane_count * shape->packed_lane_bit_count) {
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

static bool loom_amdgpu_table_lookup_index_lane_fits_table(
    loom_value_facts_t facts, uint32_t table_lane_count) {
  int64_t maximum = 0;
  return loom_value_facts_as_non_negative_i64_maximum(facts, &maximum) &&
         (uint64_t)maximum < table_lane_count;
}

static bool loom_amdgpu_table_lookup_indices_fit_table(
    const loom_value_fact_table_t* fact_table, loom_value_id_t indices,
    uint32_t lane_count, uint32_t table_lane_count) {
  if (fact_table == NULL) {
    return false;
  }

  const loom_value_facts_t index_facts =
      loom_value_fact_table_lookup(fact_table, indices);
  loom_value_fact_uniform_element_t uniform = {0};
  if (loom_value_facts_query_uniform_element(&fact_table->context, index_facts,
                                             &uniform)) {
    return loom_amdgpu_table_lookup_index_lane_fits_table(uniform.element,
                                                          table_lane_count);
  }

  loom_value_fact_small_static_lanes_t lanes = {0};
  if (loom_value_facts_query_small_static_lanes(&fact_table->context,
                                                index_facts, &lanes)) {
    if (lanes.count < lane_count) {
      return false;
    }
    for (uint32_t i = 0; i < lane_count; ++i) {
      if (!loom_amdgpu_table_lookup_index_lane_fits_table(lanes.lanes[i],
                                                          table_lane_count)) {
        return false;
      }
    }
    return true;
  }

  return loom_amdgpu_table_lookup_index_lane_fits_table(index_facts,
                                                        table_lane_count);
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
    uint32_t result_register_count = 0;
    if (!loom_amdgpu_table_lookup_shape_matches(
            &row->result_shape, &result_summary, /*result_lane_count=*/0,
            &result_lane_count, &result_register_count)) {
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
    if (row->indices_must_fit_table &&
        !loom_amdgpu_table_lookup_indices_fit_table(
            fact_table, out_plan->indices, result_lane_count,
            table_lane_count)) {
      continue;
    }

    out_plan->strategy = row->strategy;
    out_plan->index_kind = row->index_kind;
    out_plan->table_lane_count = table_lane_count;
    out_plan->table_register_count = table_register_count;
    out_plan->result_lane_count = result_lane_count;
    out_plan->index_register_count = index_register_count;
    out_plan->result_register_count = result_register_count;
    if (row->strategy == LOOM_AMDGPU_TABLE_LOOKUP_STRATEGY_F32_LADDER) {
      uint32_t dynamic_index_count = 0;
      for (uint32_t lane = 0; lane < result_lane_count; ++lane) {
        uint32_t table_index = 0;
        if (loom_amdgpu_source_lane_as_u32_bits(
                fact_table, module, out_plan->indices, lane, &table_index) &&
            table_index < table_lane_count) {
          out_plan->table_lane_indices[lane] = (uint8_t)table_index;
        } else {
          out_plan->table_lane_indices[lane] = UINT8_MAX;
          ++dynamic_index_count;
        }
      }
      if (dynamic_index_count == 0) {
        out_plan->indices = LOOM_VALUE_ID_INVALID;
      }
    }
    *out_row = row;
    return true;
  }
  return false;
}

static_assert(LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES <= 32,
              "table literal lanes must fit in their selection mask");

static iree_status_t loom_amdgpu_select_table_literals(
    loom_low_lower_context_t* context, loom_amdgpu_table_lookup_plan_t* plan) {
  if (plan->indices == LOOM_VALUE_ID_INVALID ||
      plan->select_src1_literal_descriptor.descriptor == NULL) {
    return iree_ok_status();
  }
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  const loom_module_t* module = loom_low_lower_context_module(context);
  uint32_t bits[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  uint32_t count = 0;
  // Lane zero seeds the accumulator; only subsequent lanes emit selects.
  for (uint32_t lane = 1; lane < plan->table_lane_count; ++lane) {
    if (loom_amdgpu_source_lane_as_u32_bits(fact_table, module, plan->table,
                                            lane, &bits[count])) {
      plan->literal_lane_mask |= UINT32_C(1) << lane;
      ++count;
    }
  }
  if (count == 0) {
    return iree_ok_status();
  }
  uint32_t* retained = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
      context, count * sizeof(*retained), (void**)&retained));
  memcpy(retained, bits, count * sizeof(*retained));
  plan->literal_bits = retained;
  return iree_ok_status();
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
    if (!descriptor_present) {
      return iree_ok_status();
    }
    if (out_plan->index_kind != LOOM_AMDGPU_TABLE_INDEX_KIND_PACKED_I8) {
      loom_amdgpu_select_byte_permute_plan(
          loom_low_lower_context_descriptor_set(context),
          &out_plan->index_pack_permute);
      if (out_plan->index_pack_permute.kind ==
          LOOM_AMDGPU_BYTE_PERMUTE_KIND_NONE) {
        return iree_ok_status();
      }
    }
    *out_selected = true;
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
    IREE_RETURN_IF_ERROR(loom_amdgpu_select_table_literals(context, out_plan));

    *out_selected = true;
    return iree_ok_status();
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_finalize_vector_table_lookup_plan(
    loom_low_lower_context_t* context,
    const loom_amdgpu_table_lookup_plan_t* plan) {
  const bool is_ladder =
      plan->strategy == LOOM_AMDGPU_TABLE_LOOKUP_STRATEGY_F32_LADDER;
  loom_type_t result_type = loom_low_register_type(
      loom_low_lower_context_descriptor_set(context)->stable_id,
      LOOM_AMDGPU_REG_CLASS_ID_VGPR,
      is_ladder ? plan->result_lane_count : plan->result_register_count);
  // A singleton table bypasses the comparison ladder and reuses its carrier.
  if (is_ladder && plan->table_lane_count == 1) {
    result_type = loom_low_lower_value_binding_type(context, plan->table);
    if (plan->result_lane_count != 1) {
      result_type = loom_low_register_carrier_type_with_unit_count(
          result_type, plan->result_lane_count);
    }
  }
  return loom_low_lower_plan_value_type(context, plan->result, result_type);
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
    loom_value_id_t true_lane, loom_value_id_t condition,
    const uint32_t* literal_bits, loom_type_t lane_type,
    loom_value_id_t* out_selected_lane) {
  *out_selected_lane = LOOM_VALUE_ID_INVALID;
  loom_named_attr_t attrs[1] = {0};
  iree_host_size_t attr_count = 0;
  const loom_low_lower_resolved_descriptor_t* descriptor =
      &plan->select_register_descriptor;
  loom_value_id_t operands[3] = {false_lane, true_lane, condition};
  iree_host_size_t operand_count = 3;
  if (literal_bits != NULL) {
    descriptor = &plan->select_src1_literal_descriptor;
    operands[1] = condition;
    operand_count = 2;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_append_i64_attr(context, IREE_SV("imm32"), *literal_bits,
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
  const uint32_t* literal_bits = plan->literal_bits;
  for (uint32_t i = 1; i < plan->table_lane_count; ++i) {
    loom_value_id_t condition = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_table_lookup_emit_index_compare(
        context, source_op, plan, index_lane, ordinals[i], i, mask_lane_type,
        &condition));
    const uint32_t* selected_bits =
        (plan->literal_lane_mask & (UINT32_C(1) << i)) != 0 ? literal_bits++
                                                            : NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_table_lookup_emit_table_select(
        context, source_op, plan, selected_lane, table_lanes[i], condition,
        selected_bits, lane_type, &selected_lane));
  }
  *out_selected_lane = selected_lane;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_table_lookup_emit_permute(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_table_lookup_plan_t* plan, loom_value_id_t source0,
    loom_value_id_t source1, loom_value_id_t selector, loom_type_t lane_type,
    loom_value_id_t* out_value) {
  loom_value_id_t operands[3] = {source0, source1, selector};
  IREE_RETURN_IF_ERROR(loom_amdgpu_legalize_vop3_scalar_sources(
      context, source_op, operands, IREE_ARRAYSIZE(operands)));
  loom_op_t* permute_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->permute_descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), &lane_type, 1, /*tied_results=*/NULL,
      /*tied_result_count=*/0, source_op->location, &permute_op));
  *out_value = loom_value_slice_get(loom_low_op_results(permute_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_table_lookup_emit_permute_tree_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_table_lookup_plan_t* plan,
    const loom_value_id_t* table_registers, loom_value_id_t index_register,
    loom_type_t lane_type, loom_value_id_t* out_result_register) {
  if (plan->table_register_count <= 2u) {
    const loom_value_id_t high_table = plan->table_register_count == 1u
                                           ? table_registers[0]
                                           : table_registers[1];
    return loom_amdgpu_table_lookup_emit_permute(
        context, source_op, plan, high_table, table_registers[0],
        index_register, lane_type, out_result_register);
  }

  loom_value_id_t low_selector = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      index_register, UINT32_C(0x07070707), lane_type, &low_selector));

  loom_value_id_t first_level[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS];
  loom_value_id_t next_level[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS];
  uint32_t level_count = 0;
  for (uint32_t i = 0; i < plan->table_register_count; i += 2u) {
    const uint32_t high_index =
        i + 1u < plan->table_register_count ? i + 1u : i;
    IREE_RETURN_IF_ERROR(loom_amdgpu_table_lookup_emit_permute(
        context, source_op, plan, table_registers[high_index],
        table_registers[i], low_selector, lane_type,
        &first_level[level_count++]));
  }

  loom_value_id_t* current = first_level;
  loom_value_id_t* next = next_level;
  uint32_t selection_bit = 3u;
  while (level_count > 1u) {
    const uint32_t selection_mask = UINT32_C(0x01010101) << selection_bit;
    loom_value_id_t selected_bits = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
        index_register, selection_mask, lane_type, &selected_bits));
    loom_value_id_t selector_offset = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
        selection_bit - 2u, selected_bits, lane_type, &selector_offset));
    loom_value_id_t merge_selector = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32_LIT,
        selector_offset, UINT32_C(0x03020100), lane_type, &merge_selector));

    uint32_t next_count = 0;
    for (uint32_t i = 0; i < level_count; i += 2u) {
      const uint32_t high_index = i + 1u < level_count ? i + 1u : i;
      IREE_RETURN_IF_ERROR(loom_amdgpu_table_lookup_emit_permute(
          context, source_op, plan, current[high_index], current[i],
          merge_selector, lane_type, &next[next_count++]));
    }
    loom_value_id_t* swap = current;
    current = next;
    next = swap;
    level_count = next_count;
    ++selection_bit;
  }

  *out_result_register = current[0];
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_table_lookup_extract_index_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_table_lookup_plan_t* plan, loom_value_id_t low_indices,
    uint32_t register_index, loom_type_t index_lane_type,
    loom_value_id_t* out_register) {
  return loom_amdgpu_extract_low_register_unit(
      context, source_op, low_indices, plan->index_register_count,
      register_index, index_lane_type, out_register);
}

static iree_status_t loom_amdgpu_table_lookup_pack_i16_indices(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_table_lookup_plan_t* plan, loom_value_id_t low_indices,
    loom_type_t index_lane_type, loom_type_t lane_type,
    loom_value_id_t* out_index_registers) {
  loom_amdgpu_byte_permute_emitter_t emitter;
  IREE_RETURN_IF_ERROR(loom_amdgpu_byte_permute_emitter_initialize(
      context, &plan->index_pack_permute, &emitter));
  for (uint32_t i = 0; i < plan->result_register_count; ++i) {
    const uint32_t low_index = i * 2u;
    const uint32_t high_index = low_index + 1u < plan->index_register_count
                                    ? low_index + 1u
                                    : low_index;
    loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_table_lookup_extract_index_register(
        context, source_op, plan, low_indices, low_index, index_lane_type,
        &low_source));
    IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
        context, source_op, low_source, &low_source));
    loom_value_id_t high_source = low_source;
    if (high_index != low_index) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_table_lookup_extract_index_register(
          context, source_op, plan, low_indices, high_index, index_lane_type,
          &high_source));
      IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
          context, source_op, high_source, &high_source));
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_byte_permute_emitter_emit(
        context, source_op, &emitter, high_source, low_source,
        UINT32_C(0x06040200), lane_type, &out_index_registers[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_table_lookup_pack_full_indices(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_table_lookup_plan_t* plan, loom_value_id_t low_indices,
    loom_type_t index_lane_type, loom_type_t lane_type,
    loom_value_id_t* out_index_registers) {
  const uint32_t lane_register_count =
      plan->index_kind == LOOM_AMDGPU_TABLE_INDEX_KIND_I64 ? 2u : 1u;
  loom_value_id_t source_lanes[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t lane = 0; lane < plan->result_lane_count; ++lane) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_table_lookup_extract_index_register(
        context, source_op, plan, low_indices, lane * lane_register_count,
        index_lane_type, &source_lanes[lane]));
    IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
        context, source_op, source_lanes[lane], &source_lanes[lane]));
  }
  return loom_amdgpu_pack_i8_lanes_with_permute(
      context, source_op, &plan->index_pack_permute, source_lanes,
      plan->result_lane_count, lane_type, out_index_registers);
}

static iree_status_t loom_amdgpu_table_lookup_pack_indices(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_table_lookup_plan_t* plan, loom_value_id_t low_indices,
    loom_type_t index_lane_type, loom_type_t lane_type,
    loom_value_id_t* out_index_registers) {
  if (plan->index_kind == LOOM_AMDGPU_TABLE_INDEX_KIND_PACKED_I8) {
    for (uint32_t i = 0; i < plan->result_register_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_table_lookup_extract_index_register(
          context, source_op, plan, low_indices, i, index_lane_type,
          &out_index_registers[i]));
    }
    return iree_ok_status();
  }
  if (plan->index_kind == LOOM_AMDGPU_TABLE_INDEX_KIND_PACKED_I16) {
    return loom_amdgpu_table_lookup_pack_i16_indices(
        context, source_op, plan, low_indices, index_lane_type, lane_type,
        out_index_registers);
  }
  return loom_amdgpu_table_lookup_pack_full_indices(
      context, source_op, plan, low_indices, index_lane_type, lane_type,
      out_index_registers);
}

static iree_status_t loom_amdgpu_lower_vector_table_lookup_packed_byte(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_table_lookup_plan_t* plan, loom_value_id_t low_table,
    loom_value_id_t low_indices) {
  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t table_lane_type =
      loom_amdgpu_low_register_lane_type(module, low_table);
  const loom_type_t index_lane_type =
      loom_amdgpu_low_register_lane_type(module, low_indices);
  loom_value_id_t table_registers[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS];
  for (uint32_t i = 0; i < plan->table_register_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
        context, source_op, low_table, plan->table_register_count, i,
        table_lane_type, &table_registers[i]));
  }

  // Each selector register has one index byte per result byte. Unused high
  // bytes in a partial final register remain unspecified independently.
  loom_value_id_t index_registers[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS];
  IREE_RETURN_IF_ERROR(loom_amdgpu_table_lookup_pack_indices(
      context, source_op, plan, low_indices, index_lane_type, lane_type,
      index_registers));
  loom_value_id_t result_registers[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS];
  for (uint32_t i = 0; i < plan->result_register_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_table_lookup_emit_permute_tree_register(
        context, source_op, plan, table_registers, index_registers[i],
        lane_type, &result_registers[i]));
  }
  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             result_registers,
                                             plan->result_register_count);
}

iree_status_t loom_amdgpu_lower_vector_table_lookup(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_table_lookup_plan_t* plan) {
  loom_value_id_t low_table = loom_low_lower_lookup_value(context, plan->table);
  loom_value_id_t low_indices = LOOM_VALUE_ID_INVALID;
  if (plan->indices != LOOM_VALUE_ID_INVALID) {
    low_indices = loom_low_lower_lookup_value(context, plan->indices);
  }

  if (plan->strategy ==
      LOOM_AMDGPU_TABLE_LOOKUP_STRATEGY_PACKED_BYTE_PERMUTE_TREE) {
    return loom_amdgpu_lower_vector_table_lookup_packed_byte(
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
    if (plan->table_lane_indices[i] != UINT8_MAX) {
      result_lanes[i] = table_lanes[plan->table_lane_indices[i]];
      continue;
    }
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

bool loom_amdgpu_vector_table_lookup_is_supported(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_low_descriptor_set_t* descriptor_set, const loom_op_t* op) {
  loom_amdgpu_table_lookup_plan_t unused_plan = {0};
  const loom_amdgpu_table_lookup_strategy_row_t* row = NULL;
  return loom_amdgpu_table_lookup_plan_from_op(module, fact_table, op, &row,
                                               &unused_plan) &&
         loom_amdgpu_table_lookup_strategy_descriptors_present(descriptor_set,
                                                               row);
}

iree_status_t loom_amdgpu_low_legality_verify_vector_table(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  if (!loom_amdgpu_low_legality_context_is_amdgpu(context)) {
    return iree_ok_status();
  }
  *out_handled = true;

  if (loom_amdgpu_vector_table_lookup_is_supported(
          loom_target_low_legality_module(context),
          loom_target_low_legality_fact_table(context),
          loom_target_low_legality_descriptor_set(context), op)) {
    return iree_ok_status();
  }
  return loom_amdgpu_low_legality_reject(context, op,
                                         IREE_SV("table_lookup.shape"));
}
