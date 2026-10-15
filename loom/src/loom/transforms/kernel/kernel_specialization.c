// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/kernel/kernel_specialization.h"

#include <string.h>

#include "loom/analysis/type_refinement.h"
#include "loom/ir/module.h"
#include "loom/ir/structural_hash.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/special_values.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/rewrite/remap.h"
#include "loom/util/predicate_facts.h"

bool loom_kernel_dispatch_grid_can_specialize(
    const loom_target_dispatch_workgroup_count_t* grid) {
  return grid->x != 0 && grid->y != 0 && grid->z != 0;
}

bool loom_kernel_dispatch_grid_equal(
    const loom_target_dispatch_workgroup_count_t* lhs,
    const loom_target_dispatch_workgroup_count_t* rhs) {
  const bool lhs_is_exact = loom_kernel_dispatch_grid_can_specialize(lhs);
  const bool rhs_is_exact = loom_kernel_dispatch_grid_can_specialize(rhs);
  if (lhs_is_exact != rhs_is_exact) {
    return false;
  }
  return !lhs_is_exact ||
         (lhs->x == rhs->x && lhs->y == rhs->y && lhs->z == rhs->z);
}

static iree_status_t loom_kernel_specialization_capture(
    const loom_module_t* module, const loom_kernel_class_site_t* site,
    uint16_t argument_ordinal, iree_arena_allocator_t* arena,
    loom_kernel_specialization_value_t* out_value, uint32_t* inout_hash) {
  const loom_value_id_t value_id = site->argument_values[argument_ordinal];
  loom_type_t type = loom_module_value_type(module, value_id);
  loom_type_refinement_result_t refinement;
  IREE_RETURN_IF_ERROR(loom_type_refine_with_value_facts(
      type, site->facts, arena, &type, &refinement));
  if (refinement == LOOM_TYPE_REFINEMENT_CONFLICT) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pipeline staged argument %u has an impossible type", argument_ordinal);
  }
  const loom_value_facts_t facts =
      loom_value_fact_table_lookup(site->facts, value_id);
  *out_value = (loom_kernel_specialization_value_t){.type = type};
  *inout_hash = loom_structural_hash_mix_u32(*inout_hash, loom_type_hash(type));
  if (loom_value_facts_can_materialize_constant(facts, type)) {
    out_value->facts = facts;
    *inout_hash = loom_structural_hash_mix_u64(*inout_hash, facts.range_lo);
    return iree_ok_status();
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  const loom_op_t* literal =
      loom_value_is_block_arg(value) ? NULL : loom_value_def_op(value);
  if (literal && iree_all_bits_set(literal->traits, LOOM_TRAIT_CONSTANT_LIKE) &&
      !loom_type_may_reference_values(type)) {
    out_value->literal = literal;
    out_value->facts = loom_value_facts_unknown();
    *inout_hash = loom_structural_hash_mix_u16(*inout_hash, literal->kind);
    *inout_hash =
        loom_structural_hash_mix_u8(*inout_hash, literal->instance_flags);
    const loom_attribute_t* attributes = loom_op_attrs((loom_op_t*)literal);
    for (uint8_t i = 0; i < literal->attribute_count; ++i) {
      *inout_hash = loom_structural_hash_mix_u32(
          *inout_hash, loom_attribute_hash(&attributes[i]));
    }
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "pipeline staged argument %u (value %%%u) must resolve to an exact "
      "scalar or a closed constant literal before command publication",
      argument_ordinal, value_id);
}

static bool loom_kernel_specializations_equal(
    const loom_kernel_specialization_value_t* lhs,
    const loom_kernel_specialization_value_t* rhs, uint16_t count) {
  for (uint16_t i = 0; i < count; ++i) {
    if (!loom_type_equal(lhs[i].type, rhs[i].type) ||
        (lhs[i].literal == NULL) != (rhs[i].literal == NULL)) {
      return false;
    }
    if (lhs[i].literal) {
      const loom_op_t* a = lhs[i].literal;
      const loom_op_t* b = rhs[i].literal;
      if (a->kind != b->kind || a->instance_flags != b->instance_flags ||
          a->attribute_count != b->attribute_count) {
        return false;
      }
      const loom_attribute_t* a_attributes = loom_op_attrs((loom_op_t*)a);
      const loom_attribute_t* b_attributes = loom_op_attrs((loom_op_t*)b);
      for (uint8_t j = 0; j < a->attribute_count; ++j) {
        if (!loom_attribute_equal(&a_attributes[j], &b_attributes[j])) {
          return false;
        }
      }
    } else if (lhs[i].facts.range_lo != rhs[i].facts.range_lo) {
      return false;
    }
  }
  return true;
}

iree_status_t loom_kernel_specializations_collect(
    const loom_module_t* site_module, const loom_kernel_class_site_t* sites,
    uint16_t specialization_count, loom_decision_class_partition_t* partition,
    iree_arena_allocator_t* arena,
    const loom_kernel_specialization_value_t*** out_site_specializations) {
  const iree_host_size_t class_capacity =
      iree_min(partition->site_count, partition->class_limit);
  const loom_kernel_specialization_value_t** site_values = NULL;
  loom_kernel_specialization_value_t* values = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, partition->site_count,
                                                 sizeof(*site_values),
                                                 (void**)&site_values));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, (class_capacity + 1) * specialization_count, sizeof(*values),
      (void**)&values));
  loom_kernel_specialization_value_t* candidate =
      values + class_capacity * specialization_count;
  uint32_t bucket_count = 1;
  while (bucket_count < 2 * class_capacity) {
    bucket_count *= 2;
  }
  loom_decision_class_ordinal_t* buckets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, bucket_count, sizeof(*buckets), (void**)&buckets));
  memset(buckets, 0xFF, bucket_count * sizeof(*buckets));
  partition->class_count = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < partition->site_count && iree_status_is_ok(status); ++i) {
    uint32_t hash = loom_structural_hash_initialize();
    for (uint16_t j = 0; j < specialization_count && iree_status_is_ok(status);
         ++j) {
      status = loom_kernel_specialization_capture(site_module, &sites[i], j,
                                                  arena, &candidate[j], &hash);
    }
    if (!iree_status_is_ok(status)) {
      break;
    }
    uint32_t bucket = loom_structural_hash_finalize(hash) & (bucket_count - 1);
    while (buckets[bucket] != LOOM_DECISION_CLASS_ORDINAL_INVALID &&
           !loom_kernel_specializations_equal(
               candidate, values + buckets[bucket] * specialization_count,
               specialization_count)) {
      bucket = (bucket + 1) & (bucket_count - 1);
    }
    if (buckets[bucket] == LOOM_DECISION_CLASS_ORDINAL_INVALID) {
      if (partition->class_count == partition->class_limit) {
        status = iree_make_status(
            IREE_STATUS_RESOURCE_EXHAUSTED,
            "mandatory pipeline specializations exceed the class limit %u",
            partition->class_limit);
        break;
      }
      buckets[bucket] = partition->class_count++;
      memcpy(values + buckets[bucket] * specialization_count, candidate,
             specialization_count * sizeof(*values));
    }
    partition->site_classes[i] = buckets[bucket];
    site_values[i] = values + buckets[bucket] * specialization_count;
  }
  if (iree_status_is_ok(status)) {
    *out_site_specializations = site_values;
  }
  return status;
}

// Signature predicates have integer literal operands. A residual condition
// involving a bound float therefore becomes an ordinary body assumption over
// the bound constant and invocation arguments. It remains a precondition of
// the specialized program, including when the constant has a NaN payload.
static iree_status_t loom_kernel_specialize_float_predicates(
    loom_func_like_t function, const loom_predicate_t* predicates,
    uint16_t predicate_count, loom_rewriter_t* rewriter,
    iree_arena_allocator_t* arena) {
  uint16_t argument_count = 0;
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(function, &argument_count);
  uint8_t* participating = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, argument_count, sizeof(*participating), (void**)&participating));
  memset(participating, 0, argument_count * sizeof(*participating));
  for (uint16_t i = 0; i < predicate_count; ++i) {
    for (uint8_t j = 0; j < predicates[i].arg_count; ++j) {
      if (predicates[i].arg_tags[j] == LOOM_PRED_ARG_VALUE) {
        const loom_value_t* value =
            loom_module_value(rewriter->module, predicates[i].args[j]);
        participating[loom_value_def_index(value)] = 1;
      }
    }
  }
  uint16_t value_count = 0;
  for (uint16_t i = 0; i < argument_count; ++i) {
    value_count += participating[i];
  }
  loom_value_id_t* values = NULL;
  loom_type_t* types = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, value_count, sizeof(*values), (void**)&values));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, value_count, sizeof(*types), (void**)&types));
  uint16_t value_index = 0;
  for (uint16_t i = 0; i < argument_count; ++i) {
    if (participating[i]) {
      values[value_index] = arguments[i];
      types[value_index++] =
          loom_module_value_type(rewriter->module, arguments[i]);
    }
  }

  const uint8_t predicates_index = function.vtable->predicates_attr_index;
  const loom_attribute_t signature_predicates =
      loom_op_attrs(function.op)[predicates_index];
  loom_op_t* assume = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_assume_build(
      &rewriter->builder, values, value_count, predicates, predicate_count,
      types, value_count, function.op->location, &assume));
  const loom_value_id_t* results = loom_op_const_results(assume);
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0; i < value_count && iree_status_is_ok(status); ++i) {
    status = loom_rewriter_replace_all_uses_except(rewriter, values[i],
                                                   results[i], assume);
  }
  // The remaining exported predicates still refer to invocation arguments.
  if (iree_status_is_ok(status)) {
    status = loom_rewriter_set_attr(rewriter, function.op, predicates_index,
                                    signature_predicates);
  }
  return status;
}

static iree_status_t loom_kernel_specialize_predicates(
    loom_func_like_t function, const loom_value_fact_table_t* facts,
    loom_rewriter_t* rewriter, iree_arena_allocator_t* arena) {
  uint16_t predicate_count = 0;
  const loom_predicate_t* predicates =
      loom_func_like_predicates(function, &predicate_count);
  if (predicate_count == 0) {
    return iree_ok_status();
  }
  loom_predicate_t* projected = NULL;
  IREE_RETURN_IF_ERROR(loom_builder_copy_predicate_list_attr_storage(
      &rewriter->builder, predicates, predicate_count,
      IREE_SV("specialized pipeline predicates"), &projected));
  uint16_t projected_count = 0;
  uint16_t body_count = 0;
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0; i < predicate_count && iree_status_is_ok(status); ++i) {
    const loom_decision_truth_t truth =
        loom_value_fact_table_evaluate_predicate(facts, &predicates[i]);
    if (truth == LOOM_DECISION_TRUTH_TRUE) {
      continue;
    }
    if (truth == LOOM_DECISION_TRUTH_FALSE) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "pipeline specialization violates predicate %u", i);
      break;
    }
    loom_predicate_t predicate = predicates[i];
    bool has_bound_float = false;
    for (uint8_t j = 0; j < predicate.arg_count; ++j) {
      if (predicate.arg_tags[j] != LOOM_PRED_ARG_VALUE) {
        continue;
      }
      const loom_value_facts_t argument_facts =
          loom_value_fact_table_lookup(facts, predicate.args[j]);
      has_bound_float |= loom_value_facts_is_float(argument_facts);
      int64_t constant = 0;
      if (loom_value_facts_as_exact_i64(argument_facts, &constant)) {
        predicate.arg_tags[j] = LOOM_PRED_ARG_CONST;
        predicate.args[j] = constant;
      }
    }
    if (has_bound_float) {
      projected[predicate_count - ++body_count] = predicate;
    } else {
      projected[projected_count++] = predicate;
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_rewriter_set_attr(
        rewriter, function.op, function.vtable->predicates_attr_index,
        loom_attr_predicate_list(projected, projected_count));
  }
  if (iree_status_is_ok(status) && body_count) {
    status = loom_kernel_specialize_float_predicates(
        function, projected + predicate_count - body_count, body_count,
        rewriter, arena);
  }
  return status;
}

// ConstantLike owns a single result without operands or regions. Rebuilding
// that payload with the target argument type preserves dependent body types
// without importing caller SSA identities into the independent product.
static iree_status_t loom_kernel_specialize_literal(
    const loom_op_t* source, loom_type_t body_type, loom_ir_remap_t* remap,
    loom_builder_t* builder, loom_value_id_t* out_value) {
  loom_location_id_t location;
  IREE_RETURN_IF_ERROR(
      loom_ir_remap_location_id(remap, source->location, &location));
  loom_op_t* literal = NULL;
  iree_status_t status = iree_ok_status();
  if (loom_op_is_empty(source) &&
      !loom_type_has_static_zero_extent(body_type)) {
    const loom_scalar_type_t element_type = loom_type_element_type(body_type);
    const loom_attribute_t zero =
        loom_scalar_type_is_float(element_type) ? loom_attr_f64(0.0)
        : element_type == LOOM_SCALAR_TYPE_I1   ? loom_attr_bool(false)
                                                : loom_attr_i64(0);
    IREE_RETURN_IF_ERROR(loom_vector_constant_build(builder, zero, body_type,
                                                    location, &literal));
  } else {
    IREE_RETURN_IF_ERROR(loom_builder_allocate_op(builder, source->kind, 0, 1,
                                                  0, 0, source->attribute_count,
                                                  location, &literal));
    literal->instance_flags = source->instance_flags;
    IREE_RETURN_IF_ERROR(loom_builder_define_result(builder, body_type,
                                                    loom_op_results(literal)));
    const loom_attribute_t* attributes = loom_op_attrs((loom_op_t*)source);
    for (uint8_t i = 0;
         i < source->attribute_count && iree_status_is_ok(status); ++i) {
      status = loom_ir_remap_attribute(remap, attributes[i],
                                       &loom_op_attrs(literal)[i]);
    }
    if (iree_status_is_ok(status)) {
      status = loom_builder_finalize_op(builder, literal);
    }
  }
  if (iree_status_is_ok(status)) {
    *out_value = loom_op_results(literal)[0];
  }
  return status;
}

// The invocation ABI has concrete views, while the body retains its authored
// dependent types. An explicit identity view preserves that boundary until
// ordinary type propagation can specialize the complete constraint closure.
static iree_status_t loom_kernel_specialize_binding(
    loom_value_id_t argument, const loom_value_fact_table_t* facts,
    loom_rewriter_t* rewriter, iree_arena_allocator_t* arena) {
  const loom_type_t body_type =
      loom_module_value_type(rewriter->module, argument);
  loom_type_t binding_type = {0};
  loom_type_refinement_result_t result;
  IREE_RETURN_IF_ERROR(loom_type_specialize_with_value_facts(
      rewriter->module, body_type, facts, arena, &binding_type, &result));
  if (result == LOOM_TYPE_REFINEMENT_CONFLICT) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pipeline specialization contradicts a binding type");
  }
  if (result == LOOM_TYPE_REFINEMENT_UNCHANGED) {
    return iree_ok_status();
  }
  if (!loom_type_is_view(body_type)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "a dependent pipeline invocation binding must be a view");
  }
  loom_op_t* refinement = NULL;
  IREE_RETURN_IF_ERROR(loom_view_refine_build(&rewriter->builder, argument,
                                              body_type, LOOM_LOCATION_UNKNOWN,
                                              &refinement));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_except(
      rewriter, argument, loom_view_refine_result(refinement), refinement));
  return loom_rewriter_set_value_type(rewriter, argument, binding_type);
}

static iree_status_t loom_kernel_specialize_argument(
    const loom_kernel_specialization_value_t* value, loom_value_id_t argument,
    loom_ir_remap_t* remap, loom_value_fact_table_t* facts,
    loom_rewriter_t* rewriter, loom_location_id_t location) {
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  const loom_type_t body_type =
      loom_module_value_type(rewriter->module, argument);
  if (value->literal) {
    IREE_RETURN_IF_ERROR(loom_kernel_specialize_literal(
        value->literal, body_type, remap, &rewriter->builder, &replacement));
  } else {
    IREE_RETURN_IF_ERROR(loom_constant_build(
        &rewriter->builder, value->facts, body_type, location, &replacement));
  }
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_define(facts, replacement, value->facts));
  return loom_rewriter_replace_all_uses_with(rewriter, argument, replacement);
}

iree_status_t loom_kernel_specializations_apply(
    const loom_module_t* site_module,
    const loom_kernel_specialization_value_t* specializations,
    uint16_t specialization_count, loom_func_like_t function,
    loom_rewriter_t* rewriter, iree_arena_allocator_t* arena) {
  loom_block_t* block = loom_region_entry_block(loom_func_like_body(function));
  loom_value_fact_table_t facts;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_initialize(&facts, arena, 0));
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0; i < specialization_count && iree_status_is_ok(status);
       ++i) {
    status = loom_value_fact_table_define(&facts, block->arg_ids[i],
                                          specializations[i].facts);
  }
  IREE_RETURN_IF_ERROR(status);
  loom_ir_remap_t remap;
  IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(site_module, rewriter->module,
                                                arena, NULL, &remap));
  if (block->first_op) {
    loom_builder_set_before(&rewriter->builder, block->first_op);
  } else {
    loom_builder_enter_region(&rewriter->builder, function.op,
                              loom_func_like_body(function));
  }
  IREE_RETURN_IF_ERROR(
      loom_kernel_specialize_predicates(function, &facts, rewriter, arena));
  if (block->first_op) {
    loom_builder_set_before(&rewriter->builder, block->first_op);
  }
  // Scalar literals define all possible dimension dependencies before shaped
  // literals are built, including signatures with forward argument references.
  for (uint8_t phase = 0; phase < 2 && iree_status_is_ok(status); ++phase) {
    for (uint16_t i = 0; i < specialization_count && iree_status_is_ok(status);
         ++i) {
      const loom_kernel_specialization_value_t* value = &specializations[i];
      if ((value->literal != NULL) != (phase != 0)) {
        continue;
      }
      status = loom_kernel_specialize_argument(value, block->arg_ids[i], &remap,
                                               &facts, rewriter,
                                               function.op->location);
    }
  }
  for (uint16_t i = specialization_count;
       i < block->arg_count && iree_status_is_ok(status); ++i) {
    status = loom_kernel_specialize_binding(block->arg_ids[i], &facts, rewriter,
                                            arena);
  }
  for (uint16_t i = specialization_count; i > 0 && iree_status_is_ok(status);
       --i) {
    status = loom_block_remove_arg(rewriter->module, block, i - 1);
  }
  if (iree_status_is_ok(status)) {
    status = loom_rewriter_set_attr(
        rewriter, function.op, function.vtable->specialization_count_attr_index,
        loom_attr_i64(0));
  }
  return status;
}
