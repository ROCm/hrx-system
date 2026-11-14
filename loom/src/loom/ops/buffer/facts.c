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
  loom_value_facts_t result =
      loom_value_facts_make(lower_bound, upper_bound, divisor);
  loom_value_facts_propagate_unary_distribution(facts, &result);
  return result;
}

static loom_value_fact_buffer_reference_t loom_buffer_default_reference(
    loom_value_id_t buffer_value_id) {
  return (loom_value_fact_buffer_reference_t){
      .maximum_byte_extent = loom_buffer_nonnegative_unknown_facts(),
      .minimum_alignment = 1,
      .memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN,
      .root_value_id = buffer_value_id,
      .root_is_byte_origin = true,
      .alias_scope_id = LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE,
      .nullability = LOOM_VALUE_FACT_REFERENCE_NULLABILITY_UNKNOWN,
  };
}

// Resolves a self-root before forwarding its reference facts to another value.
static void loom_buffer_resolve_reference_root(
    loom_value_fact_buffer_reference_t* reference, loom_value_id_t value) {
  if (reference->root_value_id == LOOM_VALUE_ID_INVALID) {
    reference->root_value_id = value;
    reference->root_is_byte_origin = true;
  }
}

iree_status_t loom_buffer_duplicate_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  const loom_value_slice_t sources = loom_buffer_duplicate_sources(op);
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0; i < sources.count && iree_status_is_ok(status); ++i) {
    result_facts[i] = operand_facts[i];
    loom_value_fact_buffer_reference_t reference;
    if (loom_value_facts_query_buffer_reference(context, operand_facts[i],
                                                &reference) &&
        reference.root_value_id == LOOM_VALUE_ID_INVALID) {
      // Joins of different roots use INVALID to mean self. Flatten that
      // coordinate before giving the same buffer another ownership identity.
      loom_buffer_resolve_reference_root(&reference, sources.values[i]);
      status = loom_value_facts_make_buffer_reference(context, reference,
                                                      &result_facts[i]);
      loom_value_facts_propagate_unary_distribution(operand_facts[i],
                                                    &result_facts[i]);
    }
  }
  return status;
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
      .root_symbol = loom_symbol_ref_null(),
      .root_is_byte_origin =
          lhs_reference.root_is_byte_origin &&
          rhs_reference.root_is_byte_origin &&
          lhs_reference.root_value_id == rhs_reference.root_value_id,
      .has_root_symbol = lhs_reference.has_root_symbol &&
                         rhs_reference.has_root_symbol &&
                         lhs_reference.root_symbol.module_id ==
                             rhs_reference.root_symbol.module_id &&
                         lhs_reference.root_symbol.symbol_id ==
                             rhs_reference.root_symbol.symbol_id,
      .alias_scope_id =
          lhs_reference.alias_scope_id == rhs_reference.alias_scope_id
              ? lhs_reference.alias_scope_id
              : LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE,
      .restrict_access = lhs_reference.restrict_access.domain_id ==
                                     rhs_reference.restrict_access.domain_id &&
                                 lhs_reference.restrict_access.scope_id ==
                                     rhs_reference.restrict_access.scope_id
                             ? lhs_reference.restrict_access
                             : (loom_value_fact_restrict_access_t){0},
      .nullability = lhs_reference.nullability == rhs_reference.nullability
                         ? lhs_reference.nullability
                         : LOOM_VALUE_FACT_REFERENCE_NULLABILITY_UNKNOWN,
  };
  if (reference.has_root_symbol) {
    reference.root_symbol = lhs_reference.root_symbol;
  }
  reference.origin = loom_value_fact_reference_origin_meet(
      lhs_reference.origin, rhs_reference.origin);
  // Null contributes no storage domain, but selecting it must not turn the
  // other arm's root identity into an unconditional identity for the result.
  // Retain only the storage class; the joined value remains its own root.
  const loom_value_fact_buffer_reference_t* storage_reference = NULL;
  if (lhs_reference.nullability == LOOM_VALUE_FACT_REFERENCE_NULLABILITY_NULL) {
    storage_reference = &rhs_reference;
  } else if (rhs_reference.nullability ==
             LOOM_VALUE_FACT_REFERENCE_NULLABILITY_NULL) {
    storage_reference = &lhs_reference;
  }
  if (storage_reference) {
    reference.memory_space = storage_reference->memory_space;
    reference.origin = storage_reference->origin;
    reference.origin.entry_value_id = LOOM_VALUE_ID_INVALID;
  }
  loom_value_facts_meet(&lhs_extent, &rhs_extent,
                        &reference.maximum_byte_extent);
  if (reference.minimum_alignment == 0) {
    reference.minimum_alignment = 1;
  }
  return loom_value_facts_make_buffer_reference(&target->context, reference,
                                                inout_facts);
}

static iree_status_t loom_buffer_widen_reference_extension(
    const loom_value_fact_domain_t* domain, const loom_module_t* module,
    loom_type_t type, loom_value_fact_table_t* target,
    const loom_value_fact_table_t* previous_table, loom_value_facts_t previous,
    const loom_value_fact_table_t* next_table, loom_value_facts_t next,
    uint32_t feedback_round, loom_value_facts_t* inout_facts) {
  (void)feedback_round;
  return loom_buffer_meet_reference_extension(domain, module, type, target,
                                              previous_table, previous,
                                              next_table, next, inout_facts);
}

static iree_status_t loom_buffer_narrow_reference_extension(
    const loom_value_fact_domain_t* domain, const loom_module_t* module,
    loom_type_t type, loom_value_fact_table_t* target,
    const loom_value_fact_table_t* previous_table, loom_value_facts_t previous,
    const loom_value_fact_table_t* next_table, loom_value_facts_t next,
    loom_value_facts_t* inout_facts) {
  (void)domain;
  (void)module;
  (void)type;
  loom_value_fact_buffer_reference_t reference = {0};
  loom_value_fact_buffer_reference_t incoming = {0};
  loom_value_facts_query_buffer_reference(&previous_table->context, previous,
                                          &reference);
  loom_value_facts_query_buffer_reference(&next_table->context, next,
                                          &incoming);
  reference.maximum_byte_extent = loom_value_facts_narrow(
      LOOM_SCALAR_TYPE_OFFSET, reference.maximum_byte_extent,
      incoming.maximum_byte_extent);
  if (reference.minimum_alignment == 1) {
    reference.minimum_alignment = incoming.minimum_alignment;
  }
  if (reference.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN) {
    reference.memory_space = incoming.memory_space;
  }
  if (reference.nullability == LOOM_VALUE_FACT_REFERENCE_NULLABILITY_UNKNOWN) {
    reference.nullability = incoming.nullability;
  }
  // Identity is deliberately retained from the existing coordinate. A new
  // numeric/storage proof does not authorize importing a different root.
  loom_value_facts_t narrowed = loom_value_facts_unknown();
  IREE_RETURN_IF_ERROR(loom_value_facts_make_buffer_reference(
      &target->context, reference, &narrowed));
  inout_facts->extension_id = narrowed.extension_id;
  return iree_ok_status();
}

const loom_value_fact_domain_t loom_buffer_fact_domain = {
    .meet_extension = loom_buffer_meet_reference_extension,
    .widen_extension = loom_buffer_widen_reference_extension,
    .narrow_extension = loom_buffer_narrow_reference_extension,
};

iree_status_t loom_buffer_null_facts(loom_fact_context_t* context,
                                     const loom_module_t* module,
                                     const loom_op_t* op,
                                     const loom_value_facts_t* operand_facts,
                                     loom_value_facts_t* result_facts) {
  loom_value_fact_buffer_reference_t reference =
      loom_buffer_default_reference(loom_buffer_null_result(op));
  reference.maximum_byte_extent = loom_value_facts_exact_i64(0);
  reference.nullability = LOOM_VALUE_FACT_REFERENCE_NULLABILITY_NULL;
  return loom_value_facts_make_buffer_reference(context, reference,
                                                &result_facts[0]);
}

iree_status_t loom_buffer_compare_null_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_value_fact_buffer_reference_t reference = {0};
  result_facts[0] = loom_value_facts_make(0, 1, 1);
  if (loom_value_facts_query_buffer_reference(context, operand_facts[0],
                                              &reference) &&
      reference.nullability != LOOM_VALUE_FACT_REFERENCE_NULLABILITY_UNKNOWN) {
    result_facts[0] = loom_value_facts_exact_i64(
        reference.nullability == LOOM_VALUE_FACT_REFERENCE_NULLABILITY_NULL);
  }
  loom_value_facts_propagate_unary_distribution(operand_facts[0],
                                                &result_facts[0]);
  if (loom_value_facts_is_lane_varying(result_facts[0])) {
    loom_value_facts_mark_lane_predicate(&result_facts[0]);
  }
  return iree_ok_status();
}

iree_status_t loom_buffer_compare_address_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  result_facts[0] = loom_value_facts_make(0, 1, 1);
  loom_value_fact_buffer_reference_t lhs = {0};
  loom_value_fact_buffer_reference_t rhs = {0};
  if (loom_value_facts_query_buffer_reference(context, operand_facts[0],
                                              &lhs) &&
      loom_value_facts_query_buffer_reference(context, operand_facts[2],
                                              &rhs) &&
      lhs.nullability != LOOM_VALUE_FACT_REFERENCE_NULLABILITY_UNKNOWN &&
      rhs.nullability != LOOM_VALUE_FACT_REFERENCE_NULLABILITY_UNKNOWN &&
      (lhs.nullability == LOOM_VALUE_FACT_REFERENCE_NULLABILITY_NULL ||
       rhs.nullability == LOOM_VALUE_FACT_REFERENCE_NULLABILITY_NULL)) {
    const bool equal = lhs.nullability == rhs.nullability;
    result_facts[0] = loom_value_facts_exact_i64(
        equal != (loom_buffer_compare_address_predicate(op) ==
                  LOOM_BUFFER_COMPARE_ADDRESS_PREDICATE_NE));
  }
  loom_value_facts_propagate_ternary_distribution(
      operand_facts[0], operand_facts[1], operand_facts[2], &result_facts[0]);
  loom_value_facts_propagate_binary_distribution(
      result_facts[0], operand_facts[3], &result_facts[0]);
  if (loom_value_facts_is_lane_varying(result_facts[0])) {
    loom_value_facts_mark_lane_predicate(&result_facts[0]);
  }
  return iree_ok_status();
}

iree_status_t loom_buffer_constant_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  const loom_value_fact_buffer_reference_t reference = {
      .maximum_byte_extent = loom_value_facts_exact_i64(
          (int64_t)loom_buffer_constant_contents(op).data_length),
      .minimum_alignment = loom_buffer_constant_has_alignment(op)
                               ? (uint64_t)loom_buffer_constant_alignment(op)
                               : 1,
      .memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT,
      .root_value_id = loom_buffer_constant_result(op),
      .root_is_byte_origin = true,
      .alias_scope_id = LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE,
      .nullability = LOOM_VALUE_FACT_REFERENCE_NULLABILITY_NON_NULL,
  };
  return loom_value_facts_make_buffer_reference(context, reference,
                                                &result_facts[0]);
}

static iree_status_t loom_buffer_allocation_facts(
    loom_fact_context_t* context, loom_value_id_t result,
    loom_value_facts_t byte_extent, uint64_t minimum_alignment,
    loom_value_fact_memory_space_t memory_space,
    loom_value_facts_t* out_facts) {
  loom_value_fact_reference_origin_t origin = context->reference_origin;
  if (origin.kind != LOOM_VALUE_FACT_REFERENCE_ORIGIN_UNKNOWN) {
    origin.kind = LOOM_VALUE_FACT_REFERENCE_ORIGIN_ALLOCATION;
  }
  loom_value_fact_buffer_reference_t reference = {
      .maximum_byte_extent = byte_extent,
      .minimum_alignment = minimum_alignment,
      .memory_space = memory_space,
      .root_value_id = result,
      .root_is_byte_origin = true,
      .alias_scope_id = result,
      .nullability = LOOM_VALUE_FACT_REFERENCE_NULLABILITY_NON_NULL,
      .origin = origin,
  };
  IREE_RETURN_IF_ERROR(
      loom_value_facts_make_buffer_reference(context, reference, out_facts));
  // Allocation identity follows its sharing scope, independently of the
  // requested size or backing pool. Equal sizes do not identify equal storage.
  if (reference.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    loom_value_facts_mark_workgroup_uniform(out_facts);
  } else {
    loom_value_facts_mark_lane_varying(out_facts);
  }
  return iree_ok_status();
}

iree_status_t loom_buffer_alloca_facts(loom_fact_context_t* context,
                                       const loom_module_t* module,
                                       const loom_op_t* op,
                                       const loom_value_facts_t* operand_facts,
                                       loom_value_facts_t* result_facts) {
  return loom_buffer_allocation_facts(
      context, loom_buffer_alloca_result(op),
      loom_buffer_clamp_nonnegative(operand_facts[op->operand_count - 1]),
      (uint64_t)loom_buffer_alloca_base_alignment(op),
      loom_buffer_alloca_memory_space(op), &result_facts[0]);
}

iree_status_t loom_buffer_alloca_init_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  return loom_buffer_allocation_facts(
      context, loom_buffer_alloca_init_result(op),
      loom_value_facts_exact_i64(
          (int64_t)loom_buffer_alloca_init_contents(op).data_length),
      (uint64_t)loom_buffer_alloca_init_base_alignment(op),
      loom_buffer_alloca_init_memory_space(op), &result_facts[0]);
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
    if (!exact || !loom_buffer_pack_align_exact(
                      current_offset, minimum_alignment, &byte_offset)) {
      exact = false;
      result_facts[i + 1] = loom_buffer_nonnegative_unknown_facts();
      continue;
    }
    result_facts[i + 1] = loom_value_facts_exact_i64(byte_offset);
    exact = loom_value_facts_as_exact_i64(operand_facts[i], &byte_length) &&
            byte_length >= 0 &&
            iree_checked_add_i64(byte_offset, byte_length, &current_offset);
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
  loom_buffer_resolve_reference_root(
      &reference, loom_buffer_assume_memory_space_buffer(op));
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
    loom_buffer_resolve_reference_root(&reference, buffers.values[i]);
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
    loom_buffer_resolve_reference_root(&reference, buffers.values[i]);
    reference.alias_scope_id = reference.root_value_id;
    IREE_RETURN_IF_ERROR(loom_value_facts_make_buffer_reference(
        context, reference, &result_facts[i]));
  }
  for (uint16_t i = fact_count; i < results.count; ++i) {
    result_facts[i] = loom_value_facts_unknown();
  }
  return iree_ok_status();
}

static iree_status_t loom_buffer_refine_restrict_access(
    loom_fact_context_t* context, const loom_module_t* module,
    loom_value_id_t source, loom_value_facts_t source_facts,
    loom_value_fact_restrict_access_t restrict_access,
    loom_value_facts_t* out_facts) {
  const loom_type_t type = loom_module_value_type(module, source);
  if (loom_type_is_buffer(type)) {
    loom_value_fact_buffer_reference_t reference =
        loom_buffer_default_reference(source);
    (void)loom_value_facts_query_buffer_reference(context, source_facts,
                                                  &reference);
    loom_buffer_resolve_reference_root(&reference, source);
    reference.restrict_access = restrict_access;
    return loom_value_facts_make_buffer_reference(context, reference,
                                                  out_facts);
  }
  IREE_ASSERT(loom_type_is_view(type));
  return loom_view_reference_refine_restrict_access(
      context, source, source_facts, type, restrict_access, out_facts);
}

iree_status_t loom_buffer_restrict_begin_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  const loom_value_slice_t restricted =
      loom_buffer_restrict_begin_restricted(op);
  const loom_value_slice_t results = loom_buffer_restrict_begin_results(op);
  const uint32_t domain_id =
      loom_value_fact_restrict_id_from_value(results.values[0]);
  for (uint16_t i = 0; i < results.count; ++i) {
    const loom_value_id_t operand = loom_op_const_operands(op)[i];
    const loom_value_fact_restrict_access_t restrict_access = {
        .domain_id = domain_id,
        .scope_id =
            i < restricted.count
                ? loom_value_fact_restrict_id_from_value(results.values[i])
                : 0,
    };
    IREE_RETURN_IF_ERROR(loom_buffer_refine_restrict_access(
        context, module, operand, operand_facts[i], restrict_access,
        &result_facts[i]));
  }
  return iree_ok_status();
}

iree_status_t loom_buffer_restrict_end_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  const loom_value_slice_t references = loom_buffer_restrict_end_references(op);
  for (uint16_t i = 0; i < references.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_buffer_refine_restrict_access(
        context, module, references.values[i], operand_facts[i],
        (loom_value_fact_restrict_access_t){0}, &result_facts[i]));
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
  loom_buffer_resolve_reference_root(&reference,
                                     loom_buffer_assume_same_root_buffer(op));

  loom_value_fact_buffer_reference_t root_reference =
      loom_buffer_default_reference(loom_buffer_assume_same_root_root(op));
  (void)loom_value_facts_query_buffer_reference(context, operand_facts[1],
                                                &root_reference);
  reference.root_value_id = loom_value_fact_buffer_reference_resolve_root_value(
      root_reference, loom_buffer_assume_same_root_root(op));
  reference.root_is_byte_origin = false;
  reference.root_symbol = root_reference.root_symbol;
  reference.has_root_symbol = root_reference.has_root_symbol;
  reference.alias_scope_id = root_reference.alias_scope_id;
  reference.origin = root_reference.origin;
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

iree_status_t loom_buffer_load_i8_u_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  (void)context;
  (void)module;
  (void)op;
  (void)operand_facts;
  result_facts[0] = loom_value_facts_make(0, UINT8_MAX, 1);
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
