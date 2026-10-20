// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/scf/selectify_facts.h"

#include <stdint.h>
#include <string.h>

#include "iree/base/internal/arena.h"
#include "loom/analysis/execution_condition.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/value_refs.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/rewrite/remap.h"
#include "loom/util/fact_extensions.h"
#include "loom/util/fact_table.h"
#include "loom/util/predicate_facts.h"

bool loom_scf_if_op_is_strippable_fact_identity(const loom_module_t* module,
                                                const loom_op_t* op,
                                                loom_trait_flags_t traits) {
  if (loom_traits_are_fact_identity(traits) &&
      iree_any_bit_set(traits, LOOM_TRAIT_PURE) &&
      !iree_any_bit_set(traits, LOOM_TRAIT_HINT | LOOM_TRAIT_CONVERGENT |
                                    LOOM_TRAIT_OBSERVABLE_EFFECT) &&
      !loom_traits_may_read(traits) && !loom_traits_may_write(traits) &&
      op->operand_count == 1 && op->result_count == 1) {
    return loom_type_equal(
        loom_module_value_type(module, loom_op_const_operands(op)[0]),
        loom_module_value_type(module, loom_op_const_results(op)[0]));
  }
  return false;
}

typedef struct loom_scf_if_fact_preview_t {
  // Module containing the conditional and candidate operations.
  const loom_module_t* module;
  // Facts computed for the current function before selectification.
  const loom_value_fact_table_t* source_facts;
  // Scratch facts recomputed after removing branch-local identities.
  loom_value_fact_table_t facts;
  // Sparse replacement map from stripped identity results to their sources.
  loom_ir_remap_t stripped_values;
  // Whether branch-local identities require operand remapping.
  bool remaps_values;
  // Invocation-local storage for remaps, cloned operations, and facts.
  iree_arena_allocator_t* arena;
  // False when the assumption-free eager program cannot replace the regions.
  bool can_selectify;
} loom_scf_if_fact_preview_t;

static bool loom_scf_if_type_is_scalar_integer(loom_type_t type) {
  if (!loom_type_is_scalar(type)) {
    return false;
  }
  const loom_scalar_type_t scalar_type = loom_type_element_type(type);
  return loom_scalar_type_is_integer(scalar_type) ||
         scalar_type == LOOM_SCALAR_TYPE_INDEX ||
         scalar_type == LOOM_SCALAR_TYPE_OFFSET;
}

static loom_value_id_t loom_scf_if_preview_remap_value(
    const loom_scf_if_fact_preview_t* preview, loom_value_id_t value_id) {
  if (!preview->remaps_values) {
    return value_id;
  }
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  return loom_ir_remap_try_lookup_value(&preview->stripped_values, value_id,
                                        &replacement)
             ? replacement
             : value_id;
}

static iree_host_size_t loom_scf_if_preview_op_storage_size(
    const loom_module_t* module, const loom_op_t* op) {
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  const uint8_t operand_segment_count =
      loom_op_vtable_operand_segment_count(vtable);
  if (op->attribute_count == 0 && operand_segment_count == 0) {
    return (iree_host_size_t)((const uint8_t*)(loom_op_tied_results(op) +
                                               op->tied_result_count) -
                              (const uint8_t*)op);
  }
  if (operand_segment_count != 0) {
    return (
        iree_host_size_t)((const uint8_t*)(loom_op_const_operand_segment_counts(
                                               op) +
                                           operand_segment_count) -
                          (const uint8_t*)op);
  }
  return (iree_host_size_t)((const uint8_t*)(loom_op_attribute_owners(op) +
                                             op->attribute_count) -
                            (const uint8_t*)op);
}

static iree_status_t loom_scf_if_preview_clone_remapped_op(
    loom_scf_if_fact_preview_t* preview, const loom_op_t* source_op,
    loom_op_t** out_op) {
  const iree_host_size_t storage_size =
      loom_scf_if_preview_op_storage_size(preview->module, source_op);
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(preview->arena, storage_size, (void**)&op));
  memcpy(op, source_op, storage_size);
  loom_value_id_t* operands = loom_op_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
    if (loom_ir_remap_try_lookup_value(&preview->stripped_values, operands[i],
                                       &replacement)) {
      operands[i] = replacement;
    }
  }
  *out_op = op;
  return iree_ok_status();
}

static iree_status_t loom_scf_if_preview_seed_value(loom_value_id_t value_id,
                                                    void* user_data) {
  loom_scf_if_fact_preview_t* preview = user_data;
  value_id = loom_scf_if_preview_remap_value(preview, value_id);
  if (value_id == LOOM_VALUE_ID_INVALID ||
      value_id >= preview->module->values.count ||
      loom_value_fact_table_has_entry(&preview->facts, value_id)) {
    return iree_ok_status();
  }
  const loom_value_fact_table_view_t source = {
      .table = preview->source_facts,
      .value_ids = &value_id,
      .value_count = 1,
  };
  return loom_value_fact_table_clone_values(&preview->facts, source,
                                            preview->module);
}

static iree_status_t loom_scf_if_preview_prepare_block(
    loom_scf_if_fact_preview_t* preview, const loom_block_t* block,
    const loom_op_t* yield) {
  for (const loom_op_t* child_op = block->first_op; child_op != yield;
       child_op = child_op->next_op) {
    const loom_trait_flags_t traits =
        loom_op_effective_traits(preview->module, child_op);
    if (!loom_scf_if_op_is_strippable_fact_identity(preview->module, child_op,
                                                    traits)) {
      continue;
    }
    const loom_value_id_t result = loom_op_const_results(child_op)[0];
    const loom_value_t* result_value =
        loom_module_value(preview->module, result);
    if (loom_value_has_attribute_uses(result_value) ||
        loom_module_value_has_type_uses(preview->module, result)) {
      preview->can_selectify = false;
      return iree_ok_status();
    }
    loom_value_id_t source = loom_op_const_operands(child_op)[0];
    source = loom_scf_if_preview_remap_value(preview, source);
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_map_value(&preview->stripped_values, result, source));
  }
  return iree_ok_status();
}

static iree_status_t loom_scf_if_preview_compute_block(
    loom_scf_if_fact_preview_t* preview, const loom_block_t* block,
    const loom_op_t* yield, loom_value_id_t* out_yield_values,
    loom_value_facts_t* out_yield_facts) {
  for (const loom_op_t* child_op = block->first_op; child_op != yield;
       child_op = child_op->next_op) {
    const loom_trait_flags_t traits =
        loom_op_effective_traits(preview->module, child_op);
    if (loom_scf_if_op_is_strippable_fact_identity(preview->module, child_op,
                                                   traits)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_op_walk_subtree_value_refs(
        preview->module, child_op, loom_scf_if_preview_seed_value, preview));
    const loom_op_t* preview_op = child_op;
    if (preview->remaps_values) {
      loom_op_t* remapped_op = NULL;
      IREE_RETURN_IF_ERROR(loom_scf_if_preview_clone_remapped_op(
          preview, child_op, &remapped_op));
      preview_op = remapped_op;
    }
    if (!loom_traits_are_safe_to_speculate(traits) &&
        !loom_execution_condition_is_satisfied(preview->module, preview_op,
                                               &preview->facts)) {
      preview->can_selectify = false;
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_op(
        &preview->facts, preview->module, preview_op));
  }

  const loom_value_slice_t yield_values = loom_scf_yield_values(yield);
  for (iree_host_size_t i = 0; i < yield_values.count; ++i) {
    const loom_value_id_t value =
        loom_scf_if_preview_remap_value(preview, yield_values.values[i]);
    IREE_RETURN_IF_ERROR(loom_scf_if_preview_seed_value(value, preview));
    out_yield_values[i] = value;
    out_yield_facts[i] = loom_value_fact_table_lookup(&preview->facts, value);
  }
  return iree_ok_status();
}

static iree_status_t loom_scf_if_preview_selectification_impl(
    loom_op_t* op, loom_rewriter_t* rewriter, const loom_block_t* then_block,
    const loom_op_t* then_yield, const loom_block_t* else_block,
    const loom_op_t* else_yield, bool strips_fact_identities,
    iree_arena_allocator_t* arena,
    loom_scf_if_selectification_facts_t* out_result_facts,
    bool* out_can_selectify) {
  *out_can_selectify = false;
  loom_scf_if_fact_preview_t preview = {
      .module = rewriter->module,
      .source_facts = rewriter->fact_table,
      .arena = arena,
      .remaps_values = strips_fact_identities,
      .can_selectify = true,
  };
  if (strips_fact_identities) {
    const loom_ir_remap_options_t remap_options = {
        .allow_unmapped_values = true,
        .value_map_kind = LOOM_IR_REMAP_VALUE_MAP_SPARSE,
    };
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_initialize(rewriter->module, rewriter->module, arena,
                                 &remap_options, &preview.stripped_values));
    IREE_RETURN_IF_ERROR(
        loom_scf_if_preview_prepare_block(&preview, then_block, then_yield));
    IREE_RETURN_IF_ERROR(
        loom_scf_if_preview_prepare_block(&preview, else_block, else_yield));
    if (!preview.can_selectify) {
      return iree_ok_status();
    }
  }

  IREE_ASSERT(preview.source_facts);
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_initialize(&preview.facts, arena, 0));
  preview.facts.context = preview.source_facts->context;
  preview.facts.context.table = &preview.facts;
  preview.facts.context.refine_operands.fn = NULL;
  preview.facts.context.refine_operands.user_data = NULL;
  preview.facts.context.result_facts_changed = NULL;

  const iree_host_size_t arm_result_count = 2 * op->result_count;
  loom_value_id_t* arm_values = NULL;
  loom_value_facts_t* arm_facts = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, arm_result_count, sizeof(*arm_values), (void**)&arm_values));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, arm_result_count, sizeof(*arm_facts), (void**)&arm_facts));
  loom_value_id_t* else_values = arm_values + op->result_count;
  loom_value_facts_t* else_facts = arm_facts + op->result_count;
  IREE_RETURN_IF_ERROR(loom_scf_if_preview_compute_block(
      &preview, then_block, then_yield, arm_values, arm_facts));
  if (!preview.can_selectify) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_scf_if_preview_compute_block(
      &preview, else_block, else_yield, else_values, else_facts));
  if (!preview.can_selectify) {
    return iree_ok_status();
  }

  const loom_value_id_t condition = loom_scf_if_condition(op);
  IREE_RETURN_IF_ERROR(loom_scf_if_preview_seed_value(condition, &preview));
  const loom_value_facts_t condition_facts =
      loom_value_fact_table_lookup(&preview.facts, condition);
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    const loom_type_t result_type =
        loom_module_value_type(rewriter->module, results[i]);
    loom_value_facts_t joined_facts = loom_value_facts_unknown();
    IREE_RETURN_IF_ERROR(loom_value_fact_table_meet_for_type(
        &preview.facts, rewriter->module, result_type, &preview.facts,
        arm_facts[i], &preview.facts, else_facts[i], &joined_facts));
    if (arm_values[i] != else_values[i]) {
      loom_value_facts_propagate_binary_distribution(
          joined_facts, condition_facts, &joined_facts);
      if (loom_value_facts_is_lane_varying(joined_facts)) {
        loom_value_facts_mark_lane_distribution_for_type(result_type,
                                                         &joined_facts);
      }
    }
    const loom_value_facts_t source_facts =
        loom_value_fact_table_lookup(preview.source_facts, results[i]);
    loom_value_facts_t representable_facts = joined_facts;
    if (loom_scf_if_type_is_scalar_integer(result_type)) {
      loom_predicate_t predicates[LOOM_VALUE_FACT_INTEGER_REFINEMENT_CAPACITY];
      const uint16_t predicate_count =
          loom_value_facts_export_integer_refinements(
              results[i], joined_facts, source_facts, predicates,
              IREE_ARRAYSIZE(predicates));
      for (uint16_t j = 0; j < predicate_count; ++j) {
        loom_value_facts_apply_predicate(&representable_facts, &predicates[j]);
      }
    }
    loom_value_facts_t preserved_facts = loom_value_facts_unknown();
    IREE_RETURN_IF_ERROR(loom_value_fact_table_meet_for_type(
        &preview.facts, rewriter->module, result_type, preview.source_facts,
        source_facts, &preview.facts, representable_facts, &preserved_facts));
    if (!loom_value_fact_table_facts_equal_for_type(
            rewriter->module, result_type, preview.source_facts, source_facts,
            &preview.facts, preserved_facts)) {
      return iree_ok_status();
    }
    out_result_facts[i] = (loom_scf_if_selectification_facts_t){
        .eager_facts = joined_facts,
        .structured_facts = source_facts,
    };
    // Extension IDs belong to the preview and source fact tables. The
    // materializer consumes only scalar fields proven representable by the
    // equality check above.
    out_result_facts[i].eager_facts.extension_id =
        LOOM_VALUE_FACT_EXTENSION_ID_NONE;
    out_result_facts[i].structured_facts.extension_id =
        LOOM_VALUE_FACT_EXTENSION_ID_NONE;
  }

  *out_can_selectify = true;
  return iree_ok_status();
}

iree_status_t loom_scf_if_preview_selectification(
    loom_op_t* op, loom_rewriter_t* rewriter, const loom_block_t* then_block,
    const loom_op_t* then_yield, const loom_block_t* else_block,
    const loom_op_t* else_yield, bool strips_fact_identities,
    iree_arena_allocator_t* scratch_arena,
    loom_scf_if_selectification_facts_t* out_result_facts,
    bool* out_can_selectify) {
  return loom_scf_if_preview_selectification_impl(
      op, rewriter, then_block, then_yield, else_block, else_yield,
      strips_fact_identities, scratch_arena, out_result_facts,
      out_can_selectify);
}

typedef struct loom_scf_if_integer_fact_group_t {
  // Select results collected for one variadic assumption.
  loom_value_id_t* values;
  // Result types corresponding to values.
  loom_type_t* types;
  // Positions in the conditional result tuple.
  uint16_t* positions;
  // Flat predicates constraining the collected values.
  loom_predicate_t* predicates;
  // Number of collected values.
  uint16_t value_count;
  // Number of collected predicates.
  uint16_t predicate_count;
} loom_scf_if_integer_fact_group_t;

static bool loom_scf_if_integer_assume_kind(loom_type_t type,
                                            bool* out_uses_index_assume) {
  if (!loom_type_is_scalar(type)) {
    return false;
  }
  const loom_scalar_type_t scalar_type = loom_type_element_type(type);
  if (scalar_type == LOOM_SCALAR_TYPE_INDEX ||
      scalar_type == LOOM_SCALAR_TYPE_OFFSET) {
    *out_uses_index_assume = true;
    return true;
  }
  if (loom_scalar_type_is_integer(scalar_type)) {
    *out_uses_index_assume = false;
    return true;
  }
  return false;
}

static iree_status_t loom_scf_if_build_integer_fact_group(
    loom_op_t* op, loom_rewriter_t* rewriter, bool uses_index_assume,
    loom_scf_if_integer_fact_group_t* group,
    loom_value_id_t* inout_replacements) {
  if (group->value_count == 0) {
    return iree_ok_status();
  }
  loom_op_t* assume_op = NULL;
  IREE_RETURN_IF_ERROR(
      uses_index_assume
          ? loom_index_assume_build(
                &rewriter->builder, group->values, group->value_count,
                group->predicates, group->predicate_count, group->types,
                group->value_count, op->location, &assume_op)
          : loom_scalar_assume_build(
                &rewriter->builder, group->values, group->value_count,
                group->predicates, group->predicate_count, group->types,
                group->value_count, op->location, &assume_op));
  const loom_value_slice_t results =
      uses_index_assume ? loom_index_assume_results(assume_op)
                        : loom_scalar_assume_results(assume_op);
  for (uint16_t i = 0; i < group->value_count; ++i) {
    inout_replacements[group->positions[i]] = results.values[i];
  }
  group->value_count = 0;
  group->predicate_count = 0;
  return iree_ok_status();
}

iree_status_t loom_scf_if_materialize_preserved_result_facts(
    loom_op_t* op, loom_rewriter_t* rewriter,
    const loom_scf_if_selectification_facts_t* result_facts,
    loom_value_id_t* inout_replacements) {
  uint16_t refined_value_count = 0;
  iree_host_size_t refined_predicate_count = 0;
  for (uint16_t i = 0; i < op->result_count; ++i) {
    const loom_type_t type =
        loom_module_value_type(rewriter->module, inout_replacements[i]);
    bool uses_index_assume = false;
    if (!loom_scf_if_integer_assume_kind(type, &uses_index_assume)) {
      continue;
    }
    loom_predicate_t predicates[LOOM_VALUE_FACT_INTEGER_REFINEMENT_CAPACITY];
    const uint16_t predicate_count =
        loom_value_facts_export_integer_refinements(
            inout_replacements[i], result_facts[i].eager_facts,
            result_facts[i].structured_facts, predicates,
            IREE_ARRAYSIZE(predicates));
    if (predicate_count != 0) {
      ++refined_value_count;
      refined_predicate_count += predicate_count;
    }
  }
  if (refined_value_count == 0) {
    return iree_ok_status();
  }

  const iree_host_size_t predicate_capacity =
      iree_min(refined_predicate_count, (iree_host_size_t)UINT16_MAX);
  loom_scf_if_integer_fact_group_t group = {0};
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(rewriter->arena, refined_value_count,
                                sizeof(*group.values), (void**)&group.values));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(rewriter->arena, refined_value_count,
                                sizeof(*group.types), (void**)&group.types));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      rewriter->arena, refined_value_count, sizeof(*group.positions),
      (void**)&group.positions));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      rewriter->arena, predicate_capacity, sizeof(*group.predicates),
      (void**)&group.predicates));

  for (uint8_t dialect_group = 0; dialect_group < 2; ++dialect_group) {
    const bool uses_index_assume = dialect_group == 0;
    for (uint16_t i = 0; i < op->result_count; ++i) {
      const loom_value_id_t value = inout_replacements[i];
      const loom_type_t type = loom_module_value_type(rewriter->module, value);
      bool candidate_uses_index_assume = false;
      if (!loom_scf_if_integer_assume_kind(type,
                                           &candidate_uses_index_assume) ||
          candidate_uses_index_assume != uses_index_assume) {
        continue;
      }
      loom_predicate_t predicates[LOOM_VALUE_FACT_INTEGER_REFINEMENT_CAPACITY];
      const uint16_t predicate_count =
          loom_value_facts_export_integer_refinements(
              value, result_facts[i].eager_facts,
              result_facts[i].structured_facts, predicates,
              IREE_ARRAYSIZE(predicates));
      if (predicate_count == 0) {
        continue;
      }
      if (group.value_count == UINT16_MAX ||
          predicate_count > UINT16_MAX - group.predicate_count) {
        IREE_RETURN_IF_ERROR(loom_scf_if_build_integer_fact_group(
            op, rewriter, uses_index_assume, &group, inout_replacements));
      }
      group.values[group.value_count] = value;
      group.types[group.value_count] = type;
      group.positions[group.value_count] = i;
      ++group.value_count;
      memcpy(group.predicates + group.predicate_count, predicates,
             predicate_count * sizeof(*predicates));
      group.predicate_count += predicate_count;
    }
    IREE_RETURN_IF_ERROR(loom_scf_if_build_integer_fact_group(
        op, rewriter, uses_index_assume, &group, inout_replacements));
  }
  return iree_ok_status();
}
