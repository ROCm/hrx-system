// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/lower_rule_value.h"

#include "loom/codegen/low/lower/context.h"
#include "loom/codegen/low/lower/lower_rule_source_memory.h"
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

loom_value_id_t loom_low_lower_rule_source_value(
    const loom_module_t* module, const loom_low_lower_rule_set_t* rule_set,
    const loom_op_t* source_op, uint16_t value_ref_index) {
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
    case LOOM_LOW_LOWER_VALUE_REF_RESULT:
      IREE_ASSERT_LT(value_ref->index, source_op->result_count);
      return loom_op_const_results(source_op)[value_ref->index];
    case LOOM_LOW_LOWER_VALUE_REF_TEMPORARY:
      IREE_ASSERT_UNREACHABLE("temporary value ref has no source value");
      IREE_BUILTIN_UNREACHABLE();
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_DYNAMIC_TERM:
      IREE_ASSERT_UNREACHABLE(
          "source-memory dynamic term value ref needs a selected memory plan");
      IREE_BUILTIN_UNREACHABLE();
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_DYNAMIC_BYTE_OFFSET:
      IREE_ASSERT_UNREACHABLE(
          "source-memory byte offset value ref needs a selected memory plan");
      IREE_BUILTIN_UNREACHABLE();
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_ADDRESS:
      IREE_ASSERT_UNREACHABLE(
          "source-memory address value ref needs a selected memory plan");
      IREE_BUILTIN_UNREACHABLE();
    default:
      IREE_ASSERT_UNREACHABLE("unknown generated value ref kind");
      IREE_BUILTIN_UNREACHABLE();
  }
}

loom_value_slice_t loom_low_lower_rule_value_ref_field_span(
    const loom_module_t* module, const loom_low_lower_rule_set_t* rule_set,
    const loom_op_t* source_op, uint16_t value_ref_index) {
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
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_ADDRESS:
      return (loom_value_slice_t){0};
    default:
      IREE_ASSERT_UNREACHABLE("unknown generated value ref kind");
      IREE_BUILTIN_UNREACHABLE();
  }
}

iree_status_t loom_low_lower_rule_low_value(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_rule_emit_state_t* state,
    const loom_low_lower_source_memory_t* source_memory,
    const loom_low_source_memory_access_plan_t* source_memory_access,
    uint16_t value_ref_index, loom_value_id_t* out_low_value_id) {
  *out_low_value_id = LOOM_VALUE_ID_INVALID;
  const loom_low_lower_value_ref_t* value_ref =
      &rule_set->value_refs[value_ref_index];
  switch (value_ref->kind) {
    case LOOM_LOW_LOWER_VALUE_REF_OPERAND:
    case LOOM_LOW_LOWER_VALUE_REF_RESULT: {
      loom_value_id_t source_value_id = loom_low_lower_rule_source_value(
          context->module, rule_set, source_op, value_ref_index);
      if (value_ref->materializer_index != 0) {
        const loom_low_lower_value_materializer_t* materializer =
            loom_low_lower_rule_value_materializer(rule_set, value_ref);
        return materializer->materialize(context, source_op, source_value_id,
                                         out_low_value_id);
      }
      return loom_low_lower_lookup_value(context, source_value_id,
                                         out_low_value_id);
    }
    case LOOM_LOW_LOWER_VALUE_REF_TEMPORARY:
      IREE_ASSERT_LT(value_ref->index, state->temporary_count);
      IREE_ASSERT(state->temporaries != NULL);
      IREE_ASSERT(state->temporaries[value_ref->index] !=
                  LOOM_VALUE_ID_INVALID);
      *out_low_value_id = state->temporaries[value_ref->index];
      return iree_ok_status();
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_DYNAMIC_TERM: {
      IREE_ASSERT(source_memory != NULL);
      IREE_ASSERT(source_memory_access != NULL);
      IREE_ASSERT_EQ(value_ref->materializer_index, 0);
      IREE_ASSERT_LT(value_ref->index,
                     source_memory_access->dynamic_term_count);
      const loom_value_id_t source_value_id =
          source_memory_access->dynamic_terms[value_ref->index].index;
      return loom_low_lower_lookup_value(context, source_value_id,
                                         out_low_value_id);
    }
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_DYNAMIC_BYTE_OFFSET:
      IREE_ASSERT(source_memory != NULL);
      IREE_ASSERT_EQ(value_ref->materializer_index, 0);
      return loom_low_lower_rule_materialize_source_memory_byte_offset(
          context, rule_set, source_op, source_memory, source_memory_access,
          out_low_value_id);
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_ADDRESS: {
      IREE_ASSERT(source_memory != NULL);
      IREE_ASSERT(source_memory_access != NULL);
      IREE_ASSERT_EQ(value_ref->materializer_index, 0);
      return loom_low_lower_rule_materialize_source_memory_address(
          context, rule_set, source_op, source_memory, source_memory_access,
          out_low_value_id);
    }
    default:
      IREE_ASSERT_UNREACHABLE("unknown generated value ref kind");
      IREE_BUILTIN_UNREACHABLE();
  }
}
