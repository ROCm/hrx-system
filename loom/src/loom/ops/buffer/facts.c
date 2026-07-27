// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fact implementations for the buffer dialect.

#include "loom/ir/facts.h"

#include <stdint.h>

#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/view/reference.h"

static loom_value_facts_t loom_buffer_nonnegative_unknown_facts(void) {
  return loom_value_facts_make(0, INT64_MAX, 1);
}

static loom_value_facts_t loom_buffer_clamp_nonnegative(
    loom_value_facts_t facts) {
  if (loom_value_facts_is_float(facts) || facts.range_hi < 0) {
    return loom_buffer_nonnegative_unknown_facts();
  }
  int64_t lower_bound = facts.range_lo < 0 ? 0 : facts.range_lo;
  int64_t upper_bound = facts.range_hi < 0 ? 0 : facts.range_hi;
  int64_t divisor = facts.known_divisor > 0 ? facts.known_divisor : 1;
  return loom_value_facts_make(lower_bound, upper_bound, divisor);
}

static loom_value_fact_buffer_reference_t loom_buffer_default_reference(
    loom_value_id_t buffer_value_id) {
  return (loom_value_fact_buffer_reference_t){
      .maximum_byte_extent = loom_buffer_nonnegative_unknown_facts(),
      .minimum_alignment = 1,
      .memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN,
      .root_value_id = buffer_value_id,
      .alias_scope_id = LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE,
      .nullability = LOOM_VALUE_FACT_REFERENCE_NULLABILITY_UNKNOWN,
  };
}

iree_status_t loom_buffer_alloca_facts(loom_fact_context_t* context,
                                       const loom_module_t* module,
                                       const loom_op_t* op,
                                       const loom_value_facts_t* operand_facts,
                                       loom_value_facts_t* result_facts) {
  int64_t base_alignment = loom_buffer_alloca_base_alignment(op);
  loom_value_fact_buffer_reference_t reference = {
      .maximum_byte_extent = loom_buffer_clamp_nonnegative(operand_facts[0]),
      .minimum_alignment = base_alignment > 0 ? (uint64_t)base_alignment : 1,
      .memory_space = loom_buffer_alloca_memory_space(op),
      .root_value_id = loom_buffer_alloca_result(op),
      .alias_scope_id = loom_buffer_alloca_result(op),
      .nullability = LOOM_VALUE_FACT_REFERENCE_NULLABILITY_NON_NULL,
  };
  return loom_value_facts_make_buffer_reference(context, reference,
                                                &result_facts[0]);
}

iree_status_t loom_buffer_assume_memory_space_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_value_fact_buffer_reference_t reference =
      loom_buffer_default_reference(loom_buffer_assume_memory_space_buffer(op));
  (void)loom_value_facts_query_buffer_reference(context, operand_facts[0],
                                                &reference);
  reference.memory_space = loom_buffer_assume_memory_space_memory_space(op);
  return loom_value_facts_make_buffer_reference(context, reference,
                                                &result_facts[0]);
}

iree_status_t loom_buffer_assume_alignment_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  const uint64_t minimum_alignment =
      (uint64_t)loom_buffer_assume_alignment_minimum_alignment(op);
  loom_value_slice_t buffers = loom_buffer_assume_alignment_buffers(op);
  loom_value_slice_t results = loom_buffer_assume_alignment_results(op);
  const uint16_t fact_count =
      buffers.count < results.count ? buffers.count : results.count;
  for (uint16_t i = 0; i < fact_count; ++i) {
    loom_value_fact_buffer_reference_t reference =
        loom_buffer_default_reference(buffers.values[i]);
    (void)loom_value_facts_query_buffer_reference(context, operand_facts[i],
                                                  &reference);
    if (reference.minimum_alignment < minimum_alignment) {
      reference.minimum_alignment = minimum_alignment;
    }
    IREE_RETURN_IF_ERROR(loom_value_facts_make_buffer_reference(
        context, reference, &result_facts[i]));
  }
  for (uint16_t i = fact_count; i < results.count; ++i) {
    result_facts[i] = loom_value_facts_unknown();
  }
  return iree_ok_status();
}

iree_status_t loom_buffer_assume_noalias_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_value_slice_t buffers = loom_buffer_assume_noalias_buffers(op);
  loom_value_slice_t results = loom_buffer_assume_noalias_results(op);
  const uint16_t fact_count =
      buffers.count < results.count ? buffers.count : results.count;
  for (uint16_t i = 0; i < fact_count; ++i) {
    loom_value_fact_buffer_reference_t reference =
        loom_buffer_default_reference(buffers.values[i]);
    (void)loom_value_facts_query_buffer_reference(context, operand_facts[i],
                                                  &reference);
    reference.alias_scope_id = reference.root_value_id;
    IREE_RETURN_IF_ERROR(loom_value_facts_make_buffer_reference(
        context, reference, &result_facts[i]));
  }
  for (uint16_t i = fact_count; i < results.count; ++i) {
    result_facts[i] = loom_value_facts_unknown();
  }
  return iree_ok_status();
}

iree_status_t loom_buffer_assume_same_root_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_value_fact_buffer_reference_t reference =
      loom_buffer_default_reference(loom_buffer_assume_same_root_buffer(op));
  (void)loom_value_facts_query_buffer_reference(context, operand_facts[0],
                                                &reference);

  loom_value_fact_buffer_reference_t root_reference =
      loom_buffer_default_reference(loom_buffer_assume_same_root_root(op));
  (void)loom_value_facts_query_buffer_reference(context, operand_facts[1],
                                                &root_reference);
  reference.root_value_id = root_reference.root_value_id;
  reference.alias_scope_id = root_reference.alias_scope_id;
  if (reference.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN) {
    reference.memory_space = root_reference.memory_space;
  }
  if (reference.minimum_alignment < root_reference.minimum_alignment) {
    reference.minimum_alignment = root_reference.minimum_alignment;
  }
  return loom_value_facts_make_buffer_reference(context, reference,
                                                &result_facts[0]);
}

iree_status_t loom_buffer_view_facts(loom_fact_context_t* context,
                                     const loom_module_t* module,
                                     const loom_op_t* op,
                                     const loom_value_facts_t* operand_facts,
                                     loom_value_facts_t* result_facts) {
  loom_type_t result_type =
      loom_module_value_type(module, loom_buffer_view_result(op));
  return loom_view_reference_make_buffer_view(
      context, module, loom_buffer_view_buffer(op), operand_facts[0],
      operand_facts[1], result_type, &result_facts[0]);
}
