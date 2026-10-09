// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/rule_value.h"

#include <stdint.h>

#include "iree/base/internal/math.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"

const loom_low_lower_value_materializer_t*
loom_low_lower_rule_value_materializer(
    const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_value_ref_t* value_ref) {
  const uint16_t materializer_index =
      (uint16_t)(value_ref->materializer_index - 1);
  return &rule_set->materializers[materializer_index];
}

loom_low_lower_unsigned_divisor_magic_info_t
loom_low_lower_unsigned_divisor_magic_info(uint64_t divisor, uint32_t bit_width,
                                           uint64_t numerator_maximum) {
  IREE_ASSERT_GE(bit_width, 2u);
  IREE_ASSERT_LE(bit_width, 64u);
  IREE_ASSERT_GT(divisor, 1u);
  const uint64_t mask = UINT64_MAX >> (64 - bit_width);
  IREE_ASSERT_LE(divisor, mask);
  const uint64_t half = UINT64_C(1) << (bit_width - 1);
  const uint64_t half_minus_one = half - 1;
  if (numerator_maximum < divisor) {
    return (loom_low_lower_unsigned_divisor_magic_info_t){0};
  }

  // The last numerator congruent to divisor-1 bounds reciprocal-rounding
  // error. Using the proven domain here retains native product precision while
  // allowing fewer reciprocal bits. The subtraction also works at width 64
  // without forming an unrepresentable 2^64.
  const uint64_t nc =
      numerator_maximum - ((numerator_maximum - divisor + 1) % divisor);
  uint32_t p = bit_width - 1;
  uint64_t q1 = half / nc;
  uint64_t r1 = half % nc;
  uint64_t q2 = half_minus_one / divisor;
  uint64_t r2 = half_minus_one % divisor;
  bool is_add = false;
  for (;;) {
    ++p;
    const bool carry = r1 >= nc - r1;
    // q1 is used only for the termination comparison against delta < divisor.
    // A small numerator bound can make it exceed the product width; saturation
    // preserves that comparison whereas wrapping would lose the range proof.
    q1 = q1 >= half ? mask : (q1 << 1) + carry;
    if (carry) {
      r1 = ((r1 << 1) - nc) & mask;
    } else {
      r1 = (r1 << 1) & mask;
    }
    if (r2 + 1 >= divisor - r2) {
      if (q2 >= half_minus_one) {
        is_add = true;
      }
      q2 = ((q2 << 1) + 1) & mask;
      r2 = ((r2 << 1) + 1 - divisor) & mask;
    } else {
      if (q2 >= half) {
        is_add = true;
      }
      q2 = (q2 << 1) & mask;
      r2 = ((r2 << 1) + 1) & mask;
    }
    const uint64_t delta = (divisor - 1 - r2) & mask;
    if (!(p < 2 * bit_width && (q1 < delta || (q1 == delta && r1 == 0)))) {
      break;
    }
  }

  loom_low_lower_unsigned_divisor_magic_info_t info = {
      .multiplier = (q2 + 1) & mask,
      .post_shift = (uint8_t)(p - bit_width),
      .is_add = is_add,
  };
  if (info.is_add) {
    IREE_ASSERT_GT(info.post_shift, 0);
    --info.post_shift;
  }
  return info;
}

uint64_t loom_low_lower_u32_divisor_reciprocal(uint32_t divisor) {
  return UINT64_MAX / divisor + 1;
}

const loom_op_t* loom_low_lower_rule_source_op(
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_op_t* const* source_nodes, uint8_t source_node_count,
    uint16_t value_ref_index) {
  const uint8_t source_node_index =
      rule_set->value_refs[value_ref_index].source_node_index;
  if (source_node_index == 0) {
    return source_op;
  }
  IREE_ASSERT(source_nodes != NULL);
  IREE_ASSERT_GT(source_node_count, 1);
  IREE_ASSERT_LT(source_node_index, source_node_count);
  IREE_ASSERT_EQ(source_nodes[0], source_op);
  return source_nodes[source_node_index];
}

loom_value_id_t loom_low_lower_rule_source_value_from_nodes(
    const loom_module_t* module, const loom_low_lower_rule_set_t* rule_set,
    const loom_op_t* source_op, const loom_op_t* const* source_nodes,
    uint8_t source_node_count, uint16_t value_ref_index) {
  source_op = loom_low_lower_rule_source_op(rule_set, source_op, source_nodes,
                                            source_node_count, value_ref_index);
  const loom_low_lower_value_ref_t* value_ref =
      &rule_set->value_refs[value_ref_index];
  switch (value_ref->kind) {
    case LOOM_LOW_LOWER_VALUE_REF_OPERAND: {
      const loom_op_vtable_t* vtable = loom_op_vtable(module, source_op);
      loom_value_slice_t span =
          loom_op_operand_field_span(vtable, source_op, value_ref->index);
      IREE_ASSERT_LT(value_ref->element_index, span.count);
      return span.values[value_ref->element_index];
    }
    case LOOM_LOW_LOWER_VALUE_REF_RESULT: {
      const loom_op_vtable_t* vtable = loom_op_vtable(module, source_op);
      loom_value_slice_t span =
          loom_op_result_field_span(vtable, source_op, value_ref->index);
      IREE_ASSERT_LT(value_ref->element_index, span.count);
      return span.values[value_ref->element_index];
    }
    case LOOM_LOW_LOWER_VALUE_REF_TEMPORARY:
      IREE_ASSERT_UNREACHABLE("temporary value ref has no source value");
      IREE_BUILTIN_UNREACHABLE();
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_DYNAMIC_TERM:
      IREE_ASSERT_UNREACHABLE(
          "source-memory dynamic term value ref needs a selected memory plan");
      IREE_BUILTIN_UNREACHABLE();
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_DYNAMIC_BYTE_OFFSET:
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_BYTE_OFFSET:
      IREE_ASSERT_UNREACHABLE(
          "source-memory byte offset value ref needs a selected memory plan");
      IREE_BUILTIN_UNREACHABLE();
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_ADDRESS:
      IREE_ASSERT_UNREACHABLE(
          "source-memory address value ref needs a selected memory plan");
      IREE_BUILTIN_UNREACHABLE();
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_ROOT:
      IREE_ASSERT_UNREACHABLE(
          "source-memory root value ref needs a selected memory plan");
      IREE_BUILTIN_UNREACHABLE();
    case LOOM_LOW_LOWER_VALUE_REF_EXACT_LANE_ORIGIN_OPERAND:
      IREE_ASSERT_UNREACHABLE(
          "exact lane origin value ref needs a populated fact table");
      IREE_BUILTIN_UNREACHABLE();
    case LOOM_LOW_LOWER_VALUE_REF_EXACT_UNIFORM_ELEMENT_ORIGIN_OPERAND:
    case LOOM_LOW_LOWER_VALUE_REF_UNIFORM_ELEMENT_ORIGIN_OPERAND:
      IREE_ASSERT_UNREACHABLE(
          "uniform origin value ref needs a populated fact table");
      IREE_BUILTIN_UNREACHABLE();
    default:
      IREE_ASSERT_UNREACHABLE("unknown generated value ref kind");
      IREE_BUILTIN_UNREACHABLE();
  }
}

bool loom_low_lower_rule_resolve_source_value_from_nodes(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_target_contract_vector_lane_projection_t vector_lane_projection,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_op_t* const* source_nodes, uint8_t source_node_count,
    uint16_t value_ref_index, loom_value_id_t* out_source_value_id) {
  *out_source_value_id = LOOM_VALUE_ID_INVALID;
  const loom_low_lower_value_ref_t* value_ref =
      &rule_set->value_refs[value_ref_index];
  switch (value_ref->kind) {
    case LOOM_LOW_LOWER_VALUE_REF_OPERAND:
    case LOOM_LOW_LOWER_VALUE_REF_RESULT:
      *out_source_value_id = loom_low_lower_rule_source_value_from_nodes(
          module, rule_set, source_op, source_nodes, source_node_count,
          value_ref_index);
      return true;
    case LOOM_LOW_LOWER_VALUE_REF_EXACT_LANE_ORIGIN_OPERAND: {
      if (fact_table == NULL) {
        return false;
      }
      const loom_op_t* referenced_op =
          loom_low_lower_rule_source_op(rule_set, source_op, source_nodes,
                                        source_node_count, value_ref_index);
      const loom_op_vtable_t* vtable = loom_op_vtable(module, referenced_op);
      const loom_value_slice_t field =
          loom_op_operand_field_span(vtable, referenced_op, value_ref->index);
      IREE_ASSERT_LT(value_ref->element_index, field.count);
      const loom_value_id_t value_id = field.values[value_ref->element_index];
      loom_value_fact_exact_lane_origin_t origin = {0};
      if (!loom_value_fact_table_query_exact_lane_origin(fact_table, module,
                                                         value_id, &origin) ||
          origin.source_lane_offset != 0 || origin.source_lane_stride != 1) {
        return false;
      }
      uint64_t value_lane_count = 0;
      uint64_t source_lane_count = 0;
      if (!loom_type_static_element_count(
              loom_target_contract_query_value_type(vector_lane_projection,
                                                    module, value_id),
              &value_lane_count) ||
          !loom_type_static_element_count(
              loom_target_contract_query_value_type(
                  vector_lane_projection, module, origin.source_value_id),
              &source_lane_count) ||
          value_lane_count != source_lane_count) {
        return false;
      }
      *out_source_value_id = origin.source_value_id;
      return true;
    }
    case LOOM_LOW_LOWER_VALUE_REF_EXACT_UNIFORM_ELEMENT_ORIGIN_OPERAND:
    case LOOM_LOW_LOWER_VALUE_REF_UNIFORM_ELEMENT_ORIGIN_OPERAND: {
      if (fact_table == NULL) {
        return false;
      }
      const loom_op_t* referenced_op =
          loom_low_lower_rule_source_op(rule_set, source_op, source_nodes,
                                        source_node_count, value_ref_index);
      const loom_op_vtable_t* vtable = loom_op_vtable(module, referenced_op);
      const loom_value_slice_t field =
          loom_op_operand_field_span(vtable, referenced_op, value_ref->index);
      IREE_ASSERT_LT(value_ref->element_index, field.count);
      if (value_ref->kind ==
          LOOM_LOW_LOWER_VALUE_REF_UNIFORM_ELEMENT_ORIGIN_OPERAND) {
        return loom_value_fact_table_query_uniform_element_origin(
            fact_table, module, field.values[value_ref->element_index],
            out_source_value_id);
      }
      return loom_value_fact_table_query_exact_uniform_element_origin(
          fact_table, module, field.values[value_ref->element_index],
          out_source_value_id);
    }
    case LOOM_LOW_LOWER_VALUE_REF_TEMPORARY:
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_DYNAMIC_TERM:
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_DYNAMIC_BYTE_OFFSET:
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_BYTE_OFFSET:
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_ADDRESS:
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_ROOT:
      return false;
    default:
      IREE_ASSERT_UNREACHABLE("unknown generated value ref kind");
      IREE_BUILTIN_UNREACHABLE();
  }
}

loom_value_id_t loom_low_lower_rule_source_value(
    const loom_module_t* module, const loom_low_lower_rule_set_t* rule_set,
    const loom_op_t* source_op, uint16_t value_ref_index) {
  return loom_low_lower_rule_source_value_from_nodes(
      module, rule_set, source_op, /*source_nodes=*/NULL,
      /*source_node_count=*/1, value_ref_index);
}

loom_value_slice_t loom_low_lower_rule_value_ref_field_span_from_nodes(
    const loom_module_t* module, const loom_low_lower_rule_set_t* rule_set,
    const loom_op_t* source_op, const loom_op_t* const* source_nodes,
    uint8_t source_node_count, uint16_t value_ref_index) {
  source_op = loom_low_lower_rule_source_op(rule_set, source_op, source_nodes,
                                            source_node_count, value_ref_index);
  const loom_low_lower_value_ref_t* value_ref =
      &rule_set->value_refs[value_ref_index];
  const loom_op_vtable_t* vtable = loom_op_vtable(module, source_op);
  switch (value_ref->kind) {
    case LOOM_LOW_LOWER_VALUE_REF_OPERAND:
      return loom_op_operand_field_span(vtable, source_op, value_ref->index);
    case LOOM_LOW_LOWER_VALUE_REF_RESULT:
      return loom_op_result_field_span(vtable, source_op, value_ref->index);
    case LOOM_LOW_LOWER_VALUE_REF_TEMPORARY:
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_DYNAMIC_TERM:
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_DYNAMIC_BYTE_OFFSET:
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_BYTE_OFFSET:
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_ADDRESS:
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_ROOT:
    case LOOM_LOW_LOWER_VALUE_REF_EXACT_LANE_ORIGIN_OPERAND:
    case LOOM_LOW_LOWER_VALUE_REF_EXACT_UNIFORM_ELEMENT_ORIGIN_OPERAND:
    case LOOM_LOW_LOWER_VALUE_REF_UNIFORM_ELEMENT_ORIGIN_OPERAND:
      return (loom_value_slice_t){0};
    default:
      IREE_ASSERT_UNREACHABLE("unknown generated value ref kind");
      IREE_BUILTIN_UNREACHABLE();
  }
}

loom_value_slice_t loom_low_lower_rule_value_ref_field_span(
    const loom_module_t* module, const loom_low_lower_rule_set_t* rule_set,
    const loom_op_t* source_op, uint16_t value_ref_index) {
  return loom_low_lower_rule_value_ref_field_span_from_nodes(
      module, rule_set, source_op, /*source_nodes=*/NULL,
      /*source_node_count=*/1, value_ref_index);
}

bool loom_low_lower_rule_integer_immediate_facts(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t value_id, loom_value_facts_t* out_facts) {
  *out_facts = loom_value_facts_unknown();
  if (fact_table == NULL) {
    return false;
  }
  const loom_type_t type = loom_module_value_type(module, value_id);
  loom_value_facts_t facts = loom_value_fact_table_lookup(fact_table, value_id);
  if (loom_type_is_vector(type)) {
    if (loom_scalar_type_is_float(loom_type_element_type(type))) {
      return false;
    }
    loom_value_fact_uniform_element_t uniform = {0};
    if (!loom_value_facts_query_uniform_element(&fact_table->context, facts,
                                                &uniform)) {
      return false;
    }
    facts = uniform.element;
  } else if (loom_type_is_scalar(type)) {
    if (loom_scalar_type_is_float(loom_type_element_type(type))) {
      return false;
    }
  } else {
    return false;
  }
  *out_facts = facts;
  return true;
}

bool loom_low_lower_rule_float_immediate_facts(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t value_id, loom_value_facts_t* out_facts) {
  *out_facts = loom_value_facts_unknown();
  if (fact_table == NULL) {
    return false;
  }
  const loom_type_t type = loom_module_value_type(module, value_id);
  loom_value_facts_t facts = loom_value_fact_table_lookup(fact_table, value_id);
  if (loom_type_is_vector(type)) {
    if (!loom_scalar_type_is_float(loom_type_element_type(type))) {
      return false;
    }
    loom_value_fact_uniform_element_t uniform = {0};
    if (!loom_value_facts_query_uniform_element(&fact_table->context, facts,
                                                &uniform)) {
      return false;
    }
    facts = uniform.element;
  } else if (loom_type_is_scalar(type)) {
    if (!loom_scalar_type_is_float(loom_type_element_type(type))) {
      return false;
    }
  } else {
    return false;
  }
  *out_facts = facts;
  return true;
}

uint64_t loom_low_lower_unsigned_numerator_maximum(
    const loom_value_fact_table_t* fact_table, loom_value_id_t numerator,
    uint32_t bit_width) {
  const uint64_t mask = UINT64_MAX >> (64 - bit_width);
  if (fact_table == NULL) {
    return mask;
  }
  const loom_value_facts_t facts =
      loom_value_fact_table_lookup(fact_table, numerator);
  if (facts.range_lo >= 0) {
    return iree_min((uint64_t)facts.range_hi, mask);
  }
  // A wholly negative signed interval is contiguous in the unsigned domain.
  // An interval crossing zero includes -1 and therefore the unsigned maximum.
  return facts.range_hi < 0 ? (uint64_t)facts.range_hi & mask : mask;
}

bool loom_low_lower_rule_value_facts_u32_divisor_magic_info(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t numerator, loom_value_id_t divisor,
    loom_low_lower_unsigned_divisor_magic_info_t* out_info) {
  *out_info = (loom_low_lower_unsigned_divisor_magic_info_t){0};
  loom_value_facts_t facts = loom_value_facts_unknown();
  if (!loom_low_lower_rule_integer_immediate_facts(module, fact_table, divisor,
                                                   &facts)) {
    return false;
  }
  int64_t exact_value = 0;
  if (!loom_value_facts_as_exact_i64(facts, &exact_value) || exact_value < 2 ||
      exact_value > UINT32_MAX) {
    return false;
  }
  *out_info = loom_low_lower_unsigned_divisor_magic_info(
      (uint32_t)exact_value, /*bit_width=*/32,
      loom_low_lower_unsigned_numerator_maximum(fact_table, numerator, 32));
  return true;
}
