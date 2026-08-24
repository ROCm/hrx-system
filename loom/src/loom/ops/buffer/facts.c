// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fact implementations for the buffer dialect.

#include "loom/ir/facts.h"

#include <stdint.h>

#include "iree/base/internal/math.h"
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

static iree_status_t loom_buffer_meet_reference_extension(
    const loom_value_fact_domain_t* domain, const loom_module_t* module,
    loom_type_t type, loom_value_fact_table_t* target,
    const loom_value_fact_table_t* lhs_table, loom_value_facts_t lhs,
    const loom_value_fact_table_t* rhs_table, loom_value_facts_t rhs,
    loom_value_facts_t* inout_facts) {
  (void)domain;
  (void)module;
  (void)type;
  loom_value_fact_buffer_reference_t lhs_reference = {0};
  loom_value_fact_buffer_reference_t rhs_reference = {0};
  if (!loom_value_facts_query_buffer_reference(&lhs_table->context, lhs,
                                               &lhs_reference) ||
      !loom_value_facts_query_buffer_reference(&rhs_table->context, rhs,
                                               &rhs_reference)) {
    inout_facts->extension_id = LOOM_VALUE_FACT_EXTENSION_ID_NONE;
    return iree_ok_status();
  }

  loom_value_facts_t lhs_extent = lhs_reference.maximum_byte_extent;
  loom_value_facts_t rhs_extent = rhs_reference.maximum_byte_extent;
  lhs_extent.extension_id = LOOM_VALUE_FACT_EXTENSION_ID_NONE;
  rhs_extent.extension_id = LOOM_VALUE_FACT_EXTENSION_ID_NONE;
  loom_value_fact_buffer_reference_t reference = {
      .minimum_alignment = iree_math_gcd_u64(lhs_reference.minimum_alignment,
                                             rhs_reference.minimum_alignment),
      .memory_space = lhs_reference.memory_space == rhs_reference.memory_space
                          ? lhs_reference.memory_space
                          : LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN,
      .root_value_id =
          lhs_reference.root_value_id == rhs_reference.root_value_id
              ? lhs_reference.root_value_id
              : LOOM_VALUE_ID_INVALID,
      .alias_scope_id =
          lhs_reference.alias_scope_id == rhs_reference.alias_scope_id
              ? lhs_reference.alias_scope_id
              : LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE,
      .nullability = lhs_reference.nullability == rhs_reference.nullability
                         ? lhs_reference.nullability
                         : LOOM_VALUE_FACT_REFERENCE_NULLABILITY_UNKNOWN,
  };
  loom_value_facts_meet(&lhs_extent, &rhs_extent,
                        &reference.maximum_byte_extent);
  if (reference.minimum_alignment == 0) reference.minimum_alignment = 1;
  return loom_value_facts_make_buffer_reference(&target->context, reference,
                                                inout_facts);
}

static iree_status_t loom_buffer_widen_reference_extension(
    const loom_value_fact_domain_t* domain, const loom_module_t* module,
    loom_type_t type, loom_value_fact_table_t* target,
    const loom_value_fact_table_t* previous_table, loom_value_facts_t previous,
    const loom_value_fact_table_t* next_table, loom_value_facts_t next,
    uint32_t iteration, loom_value_facts_t* inout_facts) {
  (void)iteration;
  return loom_buffer_meet_reference_extension(domain, module, type, target,
                                              previous_table, previous,
                                              next_table, next, inout_facts);
}

const loom_value_fact_domain_t loom_buffer_fact_domain = {
    .meet_extension = loom_buffer_meet_reference_extension,
    .widen_extension = loom_buffer_widen_reference_extension,
};

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

static bool loom_buffer_pack_align_exact(int64_t value, int64_t alignment,
                                         int64_t* out_aligned_value) {
  const int64_t mask = alignment - 1;
  int64_t biased_value = 0;
  if (value < 0 || !iree_checked_add_i64(value, mask, &biased_value)) {
    return false;
  }
  *out_aligned_value = biased_value & ~mask;
  return true;
}

iree_status_t loom_buffer_pack_facts(loom_fact_context_t* context,
                                     const loom_module_t* module,
                                     const loom_op_t* op,
                                     const loom_value_facts_t* operand_facts,
                                     loom_value_facts_t* result_facts) {
  loom_attribute_t minimum_alignments = loom_buffer_pack_minimum_alignments(op);
  int64_t current_offset = 0;
  int64_t slab_alignment = 1;
  bool exact = true;
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    const int64_t minimum_alignment = minimum_alignments.i64_array[i];
    if (minimum_alignment > slab_alignment) {
      slab_alignment = minimum_alignment;
    }

    int64_t byte_offset = 0;
    int64_t byte_length = 0;
    if (!exact ||
        !loom_buffer_pack_align_exact(current_offset, minimum_alignment,
                                      &byte_offset) ||
        !loom_value_facts_as_exact_i64(operand_facts[i], &byte_length) ||
        byte_length < 0 ||
        !iree_checked_add_i64(byte_offset, byte_length, &current_offset)) {
      exact = false;
      result_facts[i + 1] = loom_buffer_nonnegative_unknown_facts();
      continue;
    }
    result_facts[i + 1] = loom_value_facts_exact_i64(byte_offset);
  }

  int64_t total_byte_length = 0;
  if (exact && loom_buffer_pack_align_exact(current_offset, slab_alignment,
                                            &total_byte_length)) {
    result_facts[0] = loom_value_facts_exact_i64(total_byte_length);
  } else {
    result_facts[0] = loom_buffer_nonnegative_unknown_facts();
  }
  return iree_ok_status();
}

iree_status_t loom_buffer_assume_memory_space_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_value_fact_buffer_reference_t reference =
      loom_buffer_default_reference(loom_buffer_assume_memory_space_buffer(op));
  (void)loom_value_facts_query_buffer_reference(context, operand_facts[0],
                                                &reference);
  reference.root_value_id = loom_value_fact_buffer_reference_resolve_root_value(
      reference, loom_buffer_assume_memory_space_buffer(op));
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
    reference.root_value_id =
        loom_value_fact_buffer_reference_resolve_root_value(reference,
                                                            buffers.values[i]);
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
    reference.root_value_id =
        loom_value_fact_buffer_reference_resolve_root_value(reference,
                                                            buffers.values[i]);
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
  reference.root_value_id = loom_value_fact_buffer_reference_resolve_root_value(
      reference, loom_buffer_assume_same_root_buffer(op));

  loom_value_fact_buffer_reference_t root_reference =
      loom_buffer_default_reference(loom_buffer_assume_same_root_root(op));
  (void)loom_value_facts_query_buffer_reference(context, operand_facts[1],
                                                &root_reference);
  reference.root_value_id = loom_value_fact_buffer_reference_resolve_root_value(
      root_reference, loom_buffer_assume_same_root_root(op));
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

iree_status_t loom_buffer_length_facts(loom_fact_context_t* context,
                                       const loom_module_t* module,
                                       const loom_op_t* op,
                                       const loom_value_facts_t* operand_facts,
                                       loom_value_facts_t* result_facts) {
  loom_value_fact_buffer_reference_t reference = {0};
  if (!loom_value_facts_query_buffer_reference(context, operand_facts[0],
                                               &reference)) {
    result_facts[0] = loom_buffer_nonnegative_unknown_facts();
    return iree_ok_status();
  }

  if (reference.nullability == LOOM_VALUE_FACT_REFERENCE_NULLABILITY_NULL) {
    result_facts[0] = loom_value_facts_exact_i64(0);
    return iree_ok_status();
  }

  loom_value_facts_t extent =
      loom_buffer_clamp_nonnegative(reference.maximum_byte_extent);
  if (reference.nullability == LOOM_VALUE_FACT_REFERENCE_NULLABILITY_NON_NULL) {
    result_facts[0] = extent;
    return iree_ok_status();
  }

  // A buffer with unknown nullability has either its physical extent or the
  // null length of zero. Zero preserves every known positive divisor.
  result_facts[0] =
      loom_value_facts_make(0, extent.range_hi, extent.known_divisor);
  return iree_ok_status();
}

iree_status_t loom_buffer_compare_facts(loom_fact_context_t* context,
                                        const loom_module_t* module,
                                        const loom_op_t* op,
                                        const loom_value_facts_t* operand_facts,
                                        loom_value_facts_t* result_facts) {
  (void)context;
  (void)module;
  (void)op;
  (void)operand_facts;
  result_facts[0] = loom_value_facts_make(-1, 1, 1);
  return iree_ok_status();
}
