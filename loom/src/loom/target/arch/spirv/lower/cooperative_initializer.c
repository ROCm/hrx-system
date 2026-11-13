// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/lower/cooperative_initializer.h"

#include "loom/codegen/low/lower/participation.h"
#include "loom/ops/vector/fragment.h"

static bool loom_spirv_initializer_has_native_payload(
    const loom_value_fact_table_t* facts, const loom_op_t* source_op) {
  loom_vector_fragment_fact_t fragment;
  if (!loom_vector_fragment_fact_query_value_facts(
          &facts->context,
          loom_value_fact_table_lookup(facts,
                                       loom_vector_fragment_result(source_op)),
          &fragment)) {
    return false;
  }
  if (fragment.shape_rank == 3) {
    const loom_value_facts_t block_count = loom_value_fact_table_lookup(
        facts, loom_vector_fragment_fact_block_value(fragment));
    if (!loom_value_facts_is_exact(block_count) || block_count.range_lo != 1) {
      return false;
    }
  }
  if (iree_any_bit_set(fragment.flags,
                       LOOM_VECTOR_FRAGMENT_FACT_FLAG_HAS_SCHEMA)) {
    // Scalar construction preserves the component bits. Encoded scale, sparse,
    // and packing interpretations require a distinct native operation.
    const loom_value_fact_encoded_operand_schema_t schema =
        fragment.encoded_operand;
    const loom_value_fact_encoded_operand_schema_t native_schema = {
        .element_format = schema.element_format,
        .payload_packing = LOOM_VALUE_FACT_PAYLOAD_PACKING_TARGET_FRAGMENT,
        .payload_register_count = schema.payload_register_count,
        .payload_element_count = schema.payload_element_count,
    };
    return loom_value_fact_encoded_operand_schema_equal(schema, native_schema);
  }
  return true;
}

iree_status_t loom_spirv_can_materialize_cooperative_initializer_component(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value_id, bool* out_can_materialize) {
  const loom_value_fact_table_t* facts =
      loom_low_lower_context_fact_table(context);
  *out_can_materialize = false;
  if (!loom_spirv_initializer_has_native_payload(facts, source_op) ||
      !loom_value_facts_is_subgroup_uniform(
          loom_value_fact_table_lookup(facts, source_value_id))) {
    return iree_ok_status();
  }
  loom_low_lower_participation_t participation;
  IREE_RETURN_IF_ERROR(loom_low_lower_source_subgroup_participation(
      context, source_op, &participation));
  *out_can_materialize =
      participation.kind == LOOM_LOW_LOWER_PARTICIPATION_FULL;
  return iree_ok_status();
}

loom_type_t loom_spirv_cooperative_initializer_component_type(
    const loom_low_lower_context_t* context, loom_value_id_t source_value_id,
    const void* plan) {
  (void)plan;
  return loom_low_lower_value_binding_type(context, source_value_id);
}

iree_status_t loom_spirv_materialize_cooperative_initializer_component(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value_id, loom_value_id_t* out_low_value_id) {
  (void)source_op;
  *out_low_value_id = loom_low_lower_lookup_value(context, source_value_id);
  return iree_ok_status();
}
