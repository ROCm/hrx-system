// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/rewrite/type_propagation.h"

#include <string.h>

#include "loom/analysis/type_refinement.h"
#include "loom/ir/context.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ir/module.h"
#include "loom/ir/type_refinement.h"
#include "loom/target/registers.h"
#include "loom/util/cfg_graph.h"

#define LOOM_TYPE_PROPAGATION_INITIAL_ORDINAL_CAPACITY 64
#define LOOM_TYPE_PROPAGATION_INITIAL_LIST_CAPACITY 32

typedef struct loom_type_value_span_t {
  // Value ids in the span. May point at op trailing storage, block args, or
  // this span's inline single_value field.
  const loom_value_id_t* values;

  // Number of value ids in values.
  uint16_t count;

  // Inline storage for fixed single-value field references.
  loom_value_id_t single_value;
} loom_type_value_span_t;

typedef struct loom_type_propagator_queue_marks_t {
  // Transaction generation in which the value was queued.
  uint32_t value_generation;

  // Transaction generation in which the forwarding group was queued.
  uint32_t forwarding_generation;
} loom_type_propagator_queue_marks_t;

struct loom_type_propagator_t {
  // Module whose value table provides canonical value payloads.
  loom_module_t* module;

  // Scratch arena owning transaction arrays and temporary overflow dimensions.
  iree_arena_allocator_t* arena;

  // Region-local domain mapping module value IDs to compact ordinals.
  loom_local_value_domain_t value_domain;

  // Number of local value ordinals addressable by the dense arrays below.
  iree_host_size_t ordinal_capacity;

  // Current transaction generation for candidate and worklist marks.
  uint32_t transaction_generation;

  // Candidate type for each value touched in the active transaction.
  loom_type_t* candidate_types;

  // Generation mark indicating that candidate_types[ordinal] is live.
  uint32_t* candidate_generations;

  // Queue generation marks indexed by value ordinal. A forwarding group's
  // first destination value owns its forwarding mark.
  loom_type_propagator_queue_marks_t* queue_marks;

  // Parent op owning the region that defines each block argument, when known.
  loom_op_t** owner_ops;

  // Local value ordinals whose candidate type differs from the module type.
  loom_value_ordinal_t* touched_ordinals;

  // Number of live entries in touched_ordinals.
  iree_host_size_t touched_count;

  // Allocated entry count for touched_ordinals.
  iree_host_size_t touched_capacity;

  // Local value ordinal worklist used to expand the candidate closure.
  loom_value_ordinal_t* value_worklist;

  // Number of live entries in value_worklist.
  iree_host_size_t value_worklist_count;

  // Allocated entry count for value_worklist.
  iree_host_size_t value_worklist_capacity;

  // Forwarding groups awaiting collective tuple propagation.
  loom_value_ordinal_t* forwarding_worklist;

  // Number of live entries in forwarding_worklist.
  iree_host_size_t forwarding_worklist_count;

  // Allocated entry count for forwarding_worklist.
  iree_host_size_t forwarding_worklist_capacity;

  // True when a candidate contradicts another candidate or existing type.
  bool conflict;
};

struct loom_type_transfer_context_t {
  // Propagator owning the active transaction.
  loom_type_propagator_t* propagator;
};

static bool loom_type_propagator_valid_value_id(
    const loom_type_propagator_t* propagator, loom_value_id_t value_id) {
  return value_id != LOOM_VALUE_ID_INVALID &&
         (iree_host_size_t)value_id < propagator->module->values.count;
}

static iree_status_t loom_type_propagator_ensure_ordinal_capacity(
    loom_type_propagator_t* propagator, iree_host_size_t minimum_capacity) {
  if (minimum_capacity <= propagator->ordinal_capacity) {
    return iree_ok_status();
  }

  const iree_host_size_t old_capacity = propagator->ordinal_capacity;
  iree_host_size_t new_capacity =
      old_capacity ? old_capacity * 2
                   : LOOM_TYPE_PROPAGATION_INITIAL_ORDINAL_CAPACITY;
  if (new_capacity < minimum_capacity) {
    new_capacity = minimum_capacity;
  }

  loom_type_t* candidate_types = NULL;
  uint32_t* candidate_generations = NULL;
  loom_type_propagator_queue_marks_t* queue_marks = NULL;
  loom_op_t** owner_ops = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      propagator->arena, new_capacity, sizeof(*candidate_types),
      (void**)&candidate_types));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      propagator->arena, new_capacity, sizeof(*candidate_generations),
      (void**)&candidate_generations));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(propagator->arena, new_capacity,
                                sizeof(*queue_marks), (void**)&queue_marks));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      propagator->arena, new_capacity, sizeof(*owner_ops), (void**)&owner_ops));
  memset(candidate_types, 0, new_capacity * sizeof(*candidate_types));
  memset(candidate_generations, 0,
         new_capacity * sizeof(*candidate_generations));
  memset(queue_marks, 0, new_capacity * sizeof(*queue_marks));
  memset(owner_ops, 0, new_capacity * sizeof(*owner_ops));

  if (old_capacity > 0) {
    memcpy(candidate_types, propagator->candidate_types,
           old_capacity * sizeof(*candidate_types));
    memcpy(candidate_generations, propagator->candidate_generations,
           old_capacity * sizeof(*candidate_generations));
    memcpy(queue_marks, propagator->queue_marks,
           old_capacity * sizeof(*queue_marks));
    memcpy(owner_ops, propagator->owner_ops, old_capacity * sizeof(*owner_ops));
  }

  propagator->candidate_types = candidate_types;
  propagator->candidate_generations = candidate_generations;
  propagator->queue_marks = queue_marks;
  propagator->owner_ops = owner_ops;
  propagator->ordinal_capacity = new_capacity;
  return iree_ok_status();
}

static iree_status_t loom_type_propagator_grow_ordinal_list(
    loom_type_propagator_t* propagator, loom_value_ordinal_t** list,
    iree_host_size_t count, iree_host_size_t* capacity) {
  iree_host_size_t minimum_capacity = count + 1;
  if (minimum_capacity <= *capacity) {
    return iree_ok_status();
  }
  iree_host_size_t new_capacity =
      *capacity ? *capacity * 2 : LOOM_TYPE_PROPAGATION_INITIAL_LIST_CAPACITY;
  if (new_capacity < minimum_capacity) {
    new_capacity = minimum_capacity;
  }
  return iree_arena_grow_array(propagator->arena, count, new_capacity,
                               sizeof(**list), capacity, (void**)list);
}

static iree_status_t loom_type_propagator_register_value(
    loom_type_propagator_t* propagator, loom_value_id_t value_id,
    loom_value_ordinal_t* out_ordinal) {
  IREE_ASSERT(loom_local_value_domain_is_acquired(&propagator->value_domain));
  if (!loom_type_propagator_valid_value_id(propagator, value_id)) {
    *out_ordinal = LOOM_VALUE_ORDINAL_INVALID;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_local_value_domain_register_value(
      &propagator->value_domain, propagator->arena, value_id, out_ordinal));
  return loom_type_propagator_ensure_ordinal_capacity(
      propagator, (iree_host_size_t)*out_ordinal + 1);
}

iree_status_t loom_type_propagator_allocate(
    loom_module_t* module, iree_arena_allocator_t* arena,
    loom_type_propagator_t** out_propagator) {
  loom_type_propagator_t* propagator = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*propagator), (void**)&propagator));
  memset(propagator, 0, sizeof(*propagator));
  propagator->module = module;
  propagator->arena = arena;
  propagator->transaction_generation = 1;
  *out_propagator = propagator;
  return iree_ok_status();
}

void loom_type_propagator_deinitialize(loom_type_propagator_t* propagator) {
  if (!propagator) {
    return;
  }
  loom_local_value_domain_release(&propagator->value_domain);
}

static void loom_type_propagator_next_transaction(
    loom_type_propagator_t* propagator) {
  ++propagator->transaction_generation;
  if (propagator->transaction_generation == 0) {
    memset(propagator->candidate_generations, 0,
           propagator->ordinal_capacity *
               sizeof(*propagator->candidate_generations));
    memset(propagator->queue_marks, 0,
           propagator->ordinal_capacity * sizeof(*propagator->queue_marks));
    propagator->transaction_generation = 1;
  }
  propagator->touched_count = 0;
  propagator->value_worklist_count = 0;
  propagator->forwarding_worklist_count = 0;
  propagator->conflict = false;
}

static iree_status_t loom_type_propagator_note_owner(
    loom_type_propagator_t* propagator, loom_value_id_t value_id,
    loom_op_t* owner_op) {
  loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  IREE_RETURN_IF_ERROR(loom_type_propagator_register_value(propagator, value_id,
                                                           &value_ordinal));
  if (value_ordinal == LOOM_VALUE_ORDINAL_INVALID) {
    return iree_ok_status();
  }
  propagator->owner_ops[value_ordinal] = owner_op;
  return iree_ok_status();
}

static iree_status_t loom_type_propagator_record_op_region_owners(
    loom_type_propagator_t* propagator, loom_op_t* op) {
  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t region_index = 0; region_index < op->region_count;
       ++region_index) {
    loom_region_t* region = regions[region_index];
    if (!region || region->block_count == 0) {
      continue;
    }
    loom_block_t* entry = loom_region_entry_block(region);
    for (uint16_t arg_index = 0; arg_index < entry->arg_count; ++arg_index) {
      IREE_RETURN_IF_ERROR(loom_type_propagator_note_owner(
          propagator, loom_block_arg_id(entry, arg_index), op));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_type_propagator_record_region_tree_owners(
    loom_type_propagator_t* propagator, loom_region_t* region) {
  if (!region) {
    return iree_ok_status();
  }
  loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      IREE_RETURN_IF_ERROR(
          loom_type_propagator_record_op_region_owners(propagator, op));
      loom_region_t** regions = loom_op_regions(op);
      for (uint8_t region_index = 0; region_index < op->region_count;
           ++region_index) {
        IREE_RETURN_IF_ERROR(loom_type_propagator_record_region_tree_owners(
            propagator, regions[region_index]));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_type_propagator_record_root_entry_owner(
    loom_type_propagator_t* propagator, loom_region_t* region,
    loom_op_t* parent_op) {
  if (!region || !parent_op || region->block_count == 0) {
    return iree_ok_status();
  }
  loom_block_t* entry = loom_region_entry_block(region);
  for (uint16_t arg_index = 0; arg_index < entry->arg_count; ++arg_index) {
    IREE_RETURN_IF_ERROR(loom_type_propagator_note_owner(
        propagator, loom_block_arg_id(entry, arg_index), parent_op));
  }
  return iree_ok_status();
}

iree_status_t loom_type_propagator_prepare_region(
    loom_type_propagator_t* propagator, loom_region_t* region,
    loom_op_t* parent_op) {
  loom_local_value_domain_release(&propagator->value_domain);
  if (!region) {
    return iree_ok_status();
  }
  iree_status_t status = loom_local_value_domain_acquire_for_region(
      propagator->module, region, propagator->arena, &propagator->value_domain);
  if (iree_status_is_ok(status)) {
    status = loom_type_propagator_ensure_ordinal_capacity(
        propagator, propagator->value_domain.value_count);
  }
  if (iree_status_is_ok(status) && propagator->ordinal_capacity > 0) {
    memset(propagator->owner_ops, 0,
           propagator->ordinal_capacity * sizeof(*propagator->owner_ops));
  }
  if (iree_status_is_ok(status)) {
    status = loom_type_propagator_record_root_entry_owner(propagator, region,
                                                          parent_op);
  }
  if (iree_status_is_ok(status)) {
    status = loom_type_propagator_record_region_tree_owners(propagator, region);
  }
  if (!iree_status_is_ok(status)) {
    loom_local_value_domain_release(&propagator->value_domain);
  }
  return status;
}

iree_status_t loom_type_propagator_prepare_function(
    loom_type_propagator_t* propagator, loom_func_like_t function) {
  return loom_type_propagator_prepare_region(
      propagator, loom_func_like_body(function), function.op);
}

loom_local_value_domain_t* loom_type_propagator_value_domain(
    loom_type_propagator_t* propagator) {
  IREE_ASSERT(loom_local_value_domain_is_acquired(&propagator->value_domain));
  return &propagator->value_domain;
}

static bool loom_type_propagator_type_has_refinement_surface(loom_type_t type) {
  if ((loom_type_is_shaped(type) || loom_type_is_pool(type)) &&
      !loom_type_is_all_static(type)) {
    return true;
  }
  if (loom_type_has_ssa_encoding(type)) {
    return true;
  }
  return loom_type_is_encoding(type) &&
         loom_type_encoding_role(type) == LOOM_ENCODING_ROLE_UNKNOWN;
}

static bool loom_type_propagator_value_has_refinement_surface(
    const loom_type_propagator_t* propagator, loom_value_id_t value_id) {
  if (!loom_type_propagator_valid_value_id(propagator, value_id)) {
    return false;
  }
  return loom_type_propagator_type_has_refinement_surface(
      loom_module_value_type(propagator->module, value_id));
}

static bool loom_type_propagator_op_values_have_refinement_surface(
    const loom_type_propagator_t* propagator, const loom_op_t* op) {
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    if (loom_type_propagator_value_has_refinement_surface(propagator,
                                                          operands[i])) {
      return true;
    }
  }
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (loom_type_propagator_value_has_refinement_surface(propagator,
                                                          results[i])) {
      return true;
    }
  }
  return false;
}

bool loom_type_propagator_may_apply_op(const loom_type_propagator_t* propagator,
                                       const loom_rewriter_t* rewriter,
                                       const loom_op_t* op,
                                       const loom_op_vtable_t* vtable) {
  if (!propagator || !op) {
    return false;
  }
  // Facts refine dynamic dimensions and SSA encoding attachments. Both are
  // represented by the module's incrementally maintained type-use table.
  const bool facts_can_refine =
      rewriter && rewriter->fact_table &&
      loom_module_has_active_type_uses(propagator->module);
  const bool vtable_can_refine =
      vtable && iree_any_bit_set(vtable->vtable_flags,
                                 LOOM_OP_VTABLE_TYPE_PROPAGATION_CANDIDATE);

  bool values_have_refinement_surface = false;
  if (facts_can_refine || vtable_can_refine) {
    values_have_refinement_surface =
        loom_type_propagator_op_values_have_refinement_surface(propagator, op);
  }
  if (facts_can_refine && values_have_refinement_surface) {
    return true;
  }
  if (!vtable_can_refine) {
    return false;
  }

  if (vtable->type_transfer || op->region_count > 0) {
    return true;
  }
  return values_have_refinement_surface;
}

static bool loom_type_propagator_has_candidate(
    const loom_type_propagator_t* propagator,
    loom_value_ordinal_t value_ordinal) {
  return value_ordinal != LOOM_VALUE_ORDINAL_INVALID &&
         (iree_host_size_t)value_ordinal < propagator->ordinal_capacity &&
         propagator->candidate_generations[value_ordinal] ==
             propagator->transaction_generation;
}

static loom_type_t loom_type_propagator_value_type(
    const loom_type_propagator_t* propagator, loom_value_id_t value_id) {
  if (!loom_type_propagator_valid_value_id(propagator, value_id)) {
    return loom_type_none();
  }
  const loom_value_ordinal_t value_ordinal =
      loom_local_value_domain_try_ordinal(&propagator->value_domain, value_id);
  if (loom_type_propagator_has_candidate(propagator, value_ordinal)) {
    return propagator->candidate_types[value_ordinal];
  }
  return loom_module_value_type(propagator->module, value_id);
}

static iree_status_t loom_type_propagator_enqueue_value(
    loom_type_propagator_t* propagator, loom_value_ordinal_t value_ordinal) {
  if (propagator->queue_marks[value_ordinal].value_generation ==
      propagator->transaction_generation) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_type_propagator_grow_ordinal_list(
      propagator, &propagator->value_worklist, propagator->value_worklist_count,
      &propagator->value_worklist_capacity));
  propagator->queue_marks[value_ordinal].value_generation =
      propagator->transaction_generation;
  propagator->value_worklist[propagator->value_worklist_count++] =
      value_ordinal;
  return iree_ok_status();
}

static iree_status_t loom_type_propagator_enqueue_forwarding(
    loom_type_propagator_t* propagator, loom_value_id_t key_value_id) {
  loom_value_ordinal_t key_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  IREE_RETURN_IF_ERROR(loom_type_propagator_register_value(
      propagator, key_value_id, &key_ordinal));
  if (key_ordinal == LOOM_VALUE_ORDINAL_INVALID ||
      propagator->queue_marks[key_ordinal].forwarding_generation ==
          propagator->transaction_generation) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_type_propagator_grow_ordinal_list(
      propagator, &propagator->forwarding_worklist,
      propagator->forwarding_worklist_count,
      &propagator->forwarding_worklist_capacity));
  propagator->queue_marks[key_ordinal].forwarding_generation =
      propagator->transaction_generation;
  propagator->forwarding_worklist[propagator->forwarding_worklist_count++] =
      key_ordinal;
  return iree_ok_status();
}

static iree_status_t loom_type_propagator_mark_touched(
    loom_type_propagator_t* propagator, loom_value_ordinal_t value_ordinal) {
  if (!loom_type_propagator_has_candidate(propagator, value_ordinal)) {
    IREE_RETURN_IF_ERROR(loom_type_propagator_grow_ordinal_list(
        propagator, &propagator->touched_ordinals, propagator->touched_count,
        &propagator->touched_capacity));
    propagator->touched_ordinals[propagator->touched_count++] = value_ordinal;
  }
  return iree_ok_status();
}

static iree_status_t loom_type_propagator_refine_shape_with_candidate(
    loom_type_t current_type, loom_type_t candidate_type,
    iree_arena_allocator_t* arena, loom_type_t* out_type,
    loom_type_refinement_result_t* out_result) {
  uint8_t rank = loom_type_rank(candidate_type);
  uint64_t candidate_dimensions[LOOM_TYPE_MAX_RANK] = {0};
  for (uint8_t i = 0; i < rank; ++i) {
    candidate_dimensions[i] = loom_type_dim(candidate_type, i);
  }
  return loom_type_refine_shape_with_dims(current_type, candidate_dimensions,
                                          rank, arena, out_type, out_result);
}

static iree_status_t loom_type_propagator_refine_encoding_with_candidate(
    loom_type_t current_type, loom_type_t candidate_type,
    iree_arena_allocator_t* arena, loom_type_t* out_type,
    loom_type_refinement_result_t* out_result) {
  return loom_type_refine_encoding_with_attachment(
      current_type, candidate_type.encoding_id, candidate_type.encoding_flags,
      arena, out_type, out_result);
}

static iree_status_t loom_type_propagator_refine_element_with_candidate(
    loom_type_t current_type, loom_type_t candidate_type,
    iree_arena_allocator_t* arena, loom_type_t* out_type,
    loom_type_refinement_result_t* out_result) {
  *out_type = current_type;
  *out_result = LOOM_TYPE_REFINEMENT_UNCHANGED;
  if (loom_type_kind(current_type) == loom_type_kind(candidate_type)) {
    return loom_type_refine_element_with_candidate(current_type, candidate_type,
                                                   arena, out_type, out_result);
  }
  if (loom_type_element_type(current_type) !=
      loom_type_element_type(candidate_type)) {
    *out_result = LOOM_TYPE_REFINEMENT_CONFLICT;
  }
  return iree_ok_status();
}

static iree_status_t loom_type_propagator_refine_property_with_candidate(
    loom_type_t current_type, loom_type_t candidate_type,
    loom_constraint_property_t property, iree_arena_allocator_t* arena,
    loom_type_t* out_type, loom_type_refinement_result_t* out_result) {
  switch ((enum loom_constraint_property_e)property) {
    case LOOM_PROPERTY_TYPE:
      return loom_type_refine_with_candidate(current_type, candidate_type,
                                             arena, out_type, out_result);
    case LOOM_PROPERTY_ELEMENT_TYPE:
      return loom_type_propagator_refine_element_with_candidate(
          current_type, candidate_type, arena, out_type, out_result);
    case LOOM_PROPERTY_ENCODING:
      return loom_type_propagator_refine_encoding_with_candidate(
          current_type, candidate_type, arena, out_type, out_result);
    case LOOM_PROPERTY_SHAPE:
      return loom_type_propagator_refine_shape_with_candidate(
          current_type, candidate_type, arena, out_type, out_result);
    case LOOM_PROPERTY_KIND:
      *out_type = current_type;
      *out_result =
          loom_type_kind(current_type) == loom_type_kind(candidate_type)
              ? LOOM_TYPE_REFINEMENT_UNCHANGED
              : LOOM_TYPE_REFINEMENT_CONFLICT;
      return iree_ok_status();
    case LOOM_PROPERTY_RANK:
      *out_type = current_type;
      *out_result =
          loom_type_rank(current_type) == loom_type_rank(candidate_type)
              ? LOOM_TYPE_REFINEMENT_UNCHANGED
              : LOOM_TYPE_REFINEMENT_CONFLICT;
      return iree_ok_status();
    case LOOM_PROPERTY_REGISTER_CLASS:
      *out_type = current_type;
      *out_result =
          loom_low_register_type_same_class(current_type, candidate_type)
              ? LOOM_TYPE_REFINEMENT_UNCHANGED
              : LOOM_TYPE_REFINEMENT_CONFLICT;
      return iree_ok_status();
    case LOOM_PROPERTY_REGISTER_UNIT_COUNT:
      *out_type = current_type;
      *out_result =
          loom_low_register_type_same_unit_count(current_type, candidate_type)
              ? LOOM_TYPE_REFINEMENT_UNCHANGED
              : LOOM_TYPE_REFINEMENT_CONFLICT;
      return iree_ok_status();
    default:
      *out_type = current_type;
      *out_result = LOOM_TYPE_REFINEMENT_UNCHANGED;
      return iree_ok_status();
  }
}

static iree_status_t loom_type_propagator_seed_candidate(
    loom_type_propagator_t* propagator, loom_value_id_t value_id,
    loom_type_t candidate_type, loom_constraint_property_t property) {
  if (propagator->conflict) {
    return iree_ok_status();
  }
  loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  IREE_RETURN_IF_ERROR(loom_type_propagator_register_value(propagator, value_id,
                                                           &value_ordinal));
  if (value_ordinal == LOOM_VALUE_ORDINAL_INVALID) {
    return iree_ok_status();
  }

  loom_type_t current_type =
      loom_type_propagator_value_type(propagator, value_id);
  loom_type_t refined_type = current_type;
  loom_type_refinement_result_t result = LOOM_TYPE_REFINEMENT_UNCHANGED;
  IREE_RETURN_IF_ERROR(loom_type_propagator_refine_property_with_candidate(
      current_type, candidate_type, property, propagator->arena, &refined_type,
      &result));
  if (result == LOOM_TYPE_REFINEMENT_CONFLICT) {
    propagator->conflict = true;
    return iree_ok_status();
  }
  if (result == LOOM_TYPE_REFINEMENT_UNCHANGED ||
      loom_type_equal(refined_type, current_type)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      loom_type_propagator_mark_touched(propagator, value_ordinal));
  propagator->candidate_types[value_ordinal] = refined_type;
  propagator->candidate_generations[value_ordinal] =
      propagator->transaction_generation;
  return loom_type_propagator_enqueue_value(propagator, value_ordinal);
}

loom_type_t loom_type_transfer_value_type(
    const loom_type_transfer_context_t* context, loom_value_id_t value_id) {
  if (!context || !context->propagator) {
    return loom_type_none();
  }
  return loom_type_propagator_value_type(context->propagator, value_id);
}

iree_status_t loom_type_transfer_seed_candidate(
    loom_type_transfer_context_t* context, loom_value_id_t value_id,
    loom_type_t candidate_type, loom_constraint_property_t property) {
  if (!context || !context->propagator) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "type transfer requires an active context");
  }
  return loom_type_propagator_seed_candidate(context->propagator, value_id,
                                             candidate_type, property);
}

iree_status_t loom_type_transfer_seed_shape_dims(
    loom_type_transfer_context_t* context, loom_value_id_t value_id,
    const uint64_t* candidate_dimensions, uint8_t candidate_rank) {
  if (!context || !context->propagator) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "type transfer requires an active context");
  }
  loom_type_propagator_t* propagator = context->propagator;
  if (propagator->conflict ||
      !loom_type_propagator_valid_value_id(propagator, value_id)) {
    return iree_ok_status();
  }
  loom_type_t current_type =
      loom_type_propagator_value_type(propagator, value_id);
  loom_type_t refined_type = current_type;
  loom_type_refinement_result_t result = LOOM_TYPE_REFINEMENT_UNCHANGED;
  IREE_RETURN_IF_ERROR(loom_type_refine_shape_with_dims(
      current_type, candidate_dimensions, candidate_rank, propagator->arena,
      &refined_type, &result));
  if (result == LOOM_TYPE_REFINEMENT_CONFLICT) {
    propagator->conflict = true;
    return iree_ok_status();
  }
  if (result == LOOM_TYPE_REFINEMENT_UNCHANGED) {
    return iree_ok_status();
  }
  return loom_type_propagator_seed_candidate(propagator, value_id, refined_type,
                                             LOOM_PROPERTY_SHAPE);
}

iree_status_t loom_type_transfer_seed_encoding_attachment(
    loom_type_transfer_context_t* context, loom_value_id_t value_id,
    uint16_t candidate_encoding_id,
    loom_encoding_flags_t candidate_encoding_flags) {
  if (!context || !context->propagator) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "type transfer requires an active context");
  }
  loom_type_propagator_t* propagator = context->propagator;
  if (propagator->conflict ||
      !loom_type_propagator_valid_value_id(propagator, value_id)) {
    return iree_ok_status();
  }
  loom_type_t current_type =
      loom_type_propagator_value_type(propagator, value_id);
  loom_type_t refined_type = current_type;
  loom_type_refinement_result_t result = LOOM_TYPE_REFINEMENT_UNCHANGED;
  IREE_RETURN_IF_ERROR(loom_type_refine_encoding_with_attachment(
      current_type, candidate_encoding_id, candidate_encoding_flags,
      propagator->arena, &refined_type, &result));
  if (result == LOOM_TYPE_REFINEMENT_CONFLICT) {
    propagator->conflict = true;
    return iree_ok_status();
  }
  if (result == LOOM_TYPE_REFINEMENT_UNCHANGED) {
    return iree_ok_status();
  }
  return loom_type_propagator_seed_candidate(propagator, value_id, refined_type,
                                             LOOM_PROPERTY_ENCODING);
}

iree_status_t loom_type_transfer_seed_static_structure_from_type(
    loom_type_transfer_context_t* context, loom_value_id_t value_id,
    loom_type_t candidate_type) {
  if (!context || !context->propagator) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "type transfer requires an active context");
  }
  loom_type_propagator_t* propagator = context->propagator;
  if (propagator->conflict ||
      !loom_type_propagator_valid_value_id(propagator, value_id)) {
    return iree_ok_status();
  }
  if (loom_type_kind(candidate_type) == LOOM_TYPE_NONE) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_type_propagator_seed_candidate(
      propagator, value_id, candidate_type, LOOM_PROPERTY_KIND));
  if (propagator->conflict) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_type_propagator_seed_candidate(
      propagator, value_id, candidate_type, LOOM_PROPERTY_ELEMENT_TYPE));
  if (propagator->conflict) {
    return iree_ok_status();
  }

  loom_type_t current_type =
      loom_type_propagator_value_type(propagator, value_id);
  if (loom_type_is_shaped(current_type) &&
      loom_type_is_shaped(candidate_type) &&
      loom_type_rank(current_type) == loom_type_rank(candidate_type)) {
    uint8_t rank = loom_type_rank(candidate_type);
    uint64_t candidate_dimensions[LOOM_TYPE_MAX_RANK] = {0};
    bool has_static_candidate_dimension = false;
    for (uint8_t i = 0; i < rank; ++i) {
      uint64_t candidate_dimension = loom_type_dim(candidate_type, i);
      if (loom_dim_is_dynamic(candidate_dimension)) {
        candidate_dimensions[i] = loom_type_dim(current_type, i);
      } else {
        candidate_dimensions[i] = candidate_dimension;
        has_static_candidate_dimension = true;
      }
    }
    if (has_static_candidate_dimension) {
      IREE_RETURN_IF_ERROR(loom_type_transfer_seed_shape_dims(
          context, value_id, candidate_dimensions, rank));
      if (propagator->conflict) {
        return iree_ok_status();
      }
    }
  }

  if (loom_type_can_have_encoding(candidate_type) &&
      !loom_type_has_ssa_encoding(candidate_type)) {
    IREE_RETURN_IF_ERROR(loom_type_transfer_seed_encoding_attachment(
        context, value_id, candidate_type.encoding_id,
        candidate_type.encoding_flags));
  }
  return iree_ok_status();
}

static bool loom_type_propagator_field_is_variadic(
    const loom_op_vtable_t* vtable, loom_field_ref_t field_ref) {
  uint8_t category = LOOM_FIELD_REF_CATEGORY(field_ref);
  uint8_t index = LOOM_FIELD_REF_INDEX(field_ref);
  switch (category) {
    case LOOM_FIELD_OPERAND:
      return vtable &&
             iree_any_bit_set(vtable->vtable_flags,
                              LOOM_OP_VTABLE_VARIADIC_OPERANDS) &&
             index == vtable->fixed_operand_count;
    case LOOM_FIELD_RESULT:
      return vtable &&
             iree_any_bit_set(vtable->vtable_flags,
                              LOOM_OP_VTABLE_VARIADIC_RESULTS) &&
             index == vtable->fixed_result_count;
    default:
      return false;
  }
}

static bool loom_type_propagator_resolve_value_field(
    const loom_op_t* op, const loom_op_vtable_t* vtable,
    loom_field_ref_t field_ref, loom_type_value_span_t* out_span) {
  *out_span = (loom_type_value_span_t){0};
  uint8_t category = LOOM_FIELD_REF_CATEGORY(field_ref);
  uint8_t index = LOOM_FIELD_REF_INDEX(field_ref);
  if (loom_type_propagator_field_is_variadic(vtable, field_ref)) {
    switch (category) {
      case LOOM_FIELD_OPERAND:
        if (index > op->operand_count) {
          return false;
        }
        out_span->values = loom_op_const_operands(op) + index;
        out_span->count = (uint16_t)(op->operand_count - index);
        return true;
      case LOOM_FIELD_RESULT:
        if (index > op->result_count) {
          return false;
        }
        out_span->values = loom_op_const_results(op) + index;
        out_span->count = (uint16_t)(op->result_count - index);
        return true;
      default:
        return false;
    }
  }

  switch (category) {
    case LOOM_FIELD_OPERAND:
      if (index >= op->operand_count) {
        return false;
      }
      out_span->single_value = loom_op_const_operands(op)[index];
      out_span->values = &out_span->single_value;
      out_span->count = 1;
      return true;
    case LOOM_FIELD_RESULT:
      if (index >= op->result_count) {
        return false;
      }
      out_span->single_value = loom_op_const_results(op)[index];
      out_span->values = &out_span->single_value;
      out_span->count = 1;
      return true;
    default:
      return false;
  }
}

static bool loom_type_propagator_region_entry_args(
    loom_op_t* op, loom_field_ref_t field_ref,
    loom_type_value_span_t* out_span) {
  *out_span = (loom_type_value_span_t){0};
  if (LOOM_FIELD_REF_CATEGORY(field_ref) != LOOM_FIELD_REGION) {
    return false;
  }
  uint8_t region_index = LOOM_FIELD_REF_INDEX(field_ref);
  if (region_index >= op->region_count) {
    return false;
  }
  loom_region_t* region = loom_op_regions(op)[region_index];
  if (!region || region->block_count == 0) {
    return false;
  }
  loom_block_t* entry = loom_region_entry_block(region);
  out_span->values = entry->arg_ids;
  out_span->count = entry->arg_count;
  return true;
}

static bool loom_type_propagator_op_is_terminator(
    const loom_type_propagator_t* propagator, const loom_op_t* op) {
  const loom_op_vtable_t* vtable = loom_op_vtable(propagator->module, op);
  return vtable && iree_any_bit_set(vtable->traits, LOOM_TRAIT_TERMINATOR);
}

static bool loom_type_propagator_terminator_matches(
    const loom_region_descriptor_t* descriptor, const loom_op_t* terminator) {
  if (!terminator || !descriptor) {
    return false;
  }
  if (descriptor->terminator == LOOM_OP_KIND_UNKNOWN) {
    return true;
  }
  return terminator->kind == descriptor->terminator;
}

static bool loom_type_propagator_region_yield_operands(
    const loom_type_propagator_t* propagator, loom_op_t* op,
    const loom_op_vtable_t* vtable, loom_field_ref_t field_ref,
    loom_type_value_span_t* out_span) {
  *out_span = (loom_type_value_span_t){0};
  if (LOOM_FIELD_REF_CATEGORY(field_ref) != LOOM_FIELD_REGION) {
    return false;
  }
  uint8_t region_index = LOOM_FIELD_REF_INDEX(field_ref);
  if (region_index >= op->region_count) {
    return false;
  }
  const loom_region_descriptor_t* descriptor =
      loom_op_vtable_region_descriptor(vtable, region_index);
  if (!descriptor) {
    return false;
  }
  loom_region_t* region = loom_op_regions(op)[region_index];
  if (!region || region->block_count == 0) {
    return false;
  }
  const loom_block_t* entry = loom_region_const_entry_block(region);
  if (entry->op_count == 0) {
    return false;
  }

  const loom_op_t* terminator = loom_block_const_last_op(entry);
  if (!loom_type_propagator_op_is_terminator(propagator, terminator)) {
    return false;
  }
  if (!loom_type_propagator_terminator_matches(descriptor, terminator)) {
    return false;
  }
  out_span->values = loom_op_const_operands(terminator);
  out_span->count = terminator->operand_count;
  return true;
}

static iree_status_t loom_type_propagator_join_value_span(
    loom_type_propagator_t* propagator, loom_type_value_span_t span,
    loom_constraint_property_t property) {
  loom_value_id_t first_value = LOOM_VALUE_ID_INVALID;
  for (uint16_t i = 0; i < span.count; ++i) {
    if (loom_type_propagator_valid_value_id(propagator, span.values[i])) {
      first_value = span.values[i];
      break;
    }
  }
  if (first_value == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }

  loom_type_t joined_type =
      loom_type_propagator_value_type(propagator, first_value);
  for (uint16_t i = 0; i < span.count; ++i) {
    loom_value_id_t value_id = span.values[i];
    if (!loom_type_propagator_valid_value_id(propagator, value_id)) {
      continue;
    }
    loom_type_t value_type =
        loom_type_propagator_value_type(propagator, value_id);
    loom_type_t refined_type = joined_type;
    loom_type_refinement_result_t result = LOOM_TYPE_REFINEMENT_UNCHANGED;
    IREE_RETURN_IF_ERROR(loom_type_propagator_refine_property_with_candidate(
        joined_type, value_type, property, propagator->arena, &refined_type,
        &result));
    if (result == LOOM_TYPE_REFINEMENT_CONFLICT) {
      propagator->conflict = true;
      return iree_ok_status();
    }
    joined_type = refined_type;
  }

  for (uint16_t i = 0; i < span.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_type_propagator_seed_candidate(
        propagator, span.values[i], joined_type, property));
    if (propagator->conflict) {
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_type_propagator_join_pair(
    loom_type_propagator_t* propagator, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_constraint_property_t property) {
  loom_value_id_t values[] = {lhs, rhs};
  return loom_type_propagator_join_value_span(
      propagator,
      (loom_type_value_span_t){.values = values,
                               .count = IREE_ARRAYSIZE(values)},
      property);
}

static iree_status_t loom_type_propagator_relation_pairwise_eq(
    loom_type_propagator_t* propagator, loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->arg_count < 2) {
    return iree_ok_status();
  }
  for (uint8_t i = 0; i < constraint->arg_count; ++i) {
    loom_type_value_span_t span = {0};
    if (!loom_type_propagator_resolve_value_field(op, vtable,
                                                  constraint->args[i], &span)) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_type_propagator_join_value_span(
        propagator, span, constraint->property));
    if (propagator->conflict) {
      return iree_ok_status();
    }
  }

  loom_type_value_span_t first_span = {0};
  if (!loom_type_propagator_resolve_value_field(op, vtable, constraint->args[0],
                                                &first_span)) {
    return iree_ok_status();
  }
  if (first_span.count == 0) {
    return iree_ok_status();
  }
  loom_value_id_t reference = first_span.values[0];
  for (uint8_t i = 1; i < constraint->arg_count; ++i) {
    loom_type_value_span_t span = {0};
    if (!loom_type_propagator_resolve_value_field(op, vtable,
                                                  constraint->args[i], &span)) {
      return iree_ok_status();
    }
    for (uint16_t j = 0; j < span.count; ++j) {
      IREE_RETURN_IF_ERROR(loom_type_propagator_join_pair(
          propagator, reference, span.values[j], constraint->property));
      if (propagator->conflict) {
        return iree_ok_status();
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_type_propagator_relation_all_same(
    loom_type_propagator_t* propagator, loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->arg_count < 1) {
    return iree_ok_status();
  }
  loom_type_value_span_t span = {0};
  if (!loom_type_propagator_resolve_value_field(op, vtable, constraint->args[0],
                                                &span)) {
    return iree_ok_status();
  }
  return loom_type_propagator_join_value_span(propagator, span,
                                              constraint->property);
}

static iree_status_t loom_type_propagator_relation_region_arg_match(
    loom_type_propagator_t* propagator, loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->arg_count < 2) {
    return iree_ok_status();
  }
  loom_type_value_span_t args = {0};
  loom_type_value_span_t inputs = {0};
  if (!loom_type_propagator_region_entry_args(op, constraint->args[0], &args)) {
    return iree_ok_status();
  }
  if (LOOM_FIELD_REF_CATEGORY(constraint->args[1]) == LOOM_FIELD_REGION) {
    if (!loom_type_propagator_region_entry_args(op, constraint->args[1],
                                                &inputs)) {
      return iree_ok_status();
    }
  } else {
    if (!loom_type_propagator_resolve_value_field(
            op, vtable, constraint->args[1], &inputs)) {
      return iree_ok_status();
    }
  }
  uint16_t count = args.count < inputs.count ? args.count : inputs.count;
  for (uint16_t i = 0; i < count; ++i) {
    IREE_RETURN_IF_ERROR(loom_type_propagator_join_pair(
        propagator, args.values[i], inputs.values[i], constraint->property));
    if (propagator->conflict) {
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_type_propagator_relation_yield_match(
    loom_type_propagator_t* propagator, loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->arg_count < 2) {
    return iree_ok_status();
  }
  if (constraint->property == LOOM_PROPERTY_TYPE) {
    return iree_ok_status();
  }
  loom_type_value_span_t yield_operands = {0};
  loom_type_value_span_t results = {0};
  if (!loom_type_propagator_region_yield_operands(
          propagator, op, vtable, constraint->args[0], &yield_operands)) {
    return iree_ok_status();
  }
  if (!loom_type_propagator_resolve_value_field(op, vtable, constraint->args[1],
                                                &results)) {
    return iree_ok_status();
  }
  uint16_t count = yield_operands.count < results.count ? yield_operands.count
                                                        : results.count;
  for (uint16_t i = 0; i < count; ++i) {
    IREE_RETURN_IF_ERROR(loom_type_propagator_join_pair(
        propagator, yield_operands.values[i], results.values[i],
        constraint->property));
    if (propagator->conflict) {
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_type_propagator_relation_variadic_match(
    loom_type_propagator_t* propagator, loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->arg_count < 2) {
    return iree_ok_status();
  }
  if (constraint->property == LOOM_PROPERTY_TYPE) {
    return iree_ok_status();
  }
  loom_type_value_span_t lhs = {0};
  loom_type_value_span_t rhs = {0};
  if (!loom_type_propagator_resolve_value_field(op, vtable, constraint->args[0],
                                                &lhs)) {
    return iree_ok_status();
  }
  if (!loom_type_propagator_resolve_value_field(op, vtable, constraint->args[1],
                                                &rhs)) {
    return iree_ok_status();
  }
  uint16_t count = lhs.count < rhs.count ? lhs.count : rhs.count;
  for (uint16_t i = 0; i < count; ++i) {
    IREE_RETURN_IF_ERROR(loom_type_propagator_join_pair(
        propagator, lhs.values[i], rhs.values[i], constraint->property));
    if (propagator->conflict) {
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_type_propagator_seed_value_facts(
    loom_type_propagator_t* propagator, const loom_rewriter_t* rewriter,
    loom_value_id_t value_id) {
  if (!rewriter->fact_table ||
      !loom_type_propagator_valid_value_id(propagator, value_id)) {
    return iree_ok_status();
  }
  loom_type_t current_type =
      loom_type_propagator_value_type(propagator, value_id);
  loom_type_t refined_type = current_type;
  loom_type_refinement_result_t result = LOOM_TYPE_REFINEMENT_UNCHANGED;
  IREE_RETURN_IF_ERROR(loom_type_specialize_with_value_facts(
      propagator->module, current_type, rewriter->fact_table, propagator->arena,
      &refined_type, &result));
  if (result == LOOM_TYPE_REFINEMENT_CONFLICT) {
    propagator->conflict = true;
    return iree_ok_status();
  }
  return loom_type_propagator_seed_candidate(propagator, value_id, refined_type,
                                             LOOM_PROPERTY_TYPE);
}

static iree_status_t loom_type_propagator_seed_value_span_facts(
    loom_type_propagator_t* propagator, const loom_rewriter_t* rewriter,
    const loom_value_id_t* values, uint16_t count) {
  for (uint16_t i = 0; i < count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_type_propagator_seed_value_facts(propagator, rewriter, values[i]));
    if (propagator->conflict) {
      break;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_type_propagator_seed_op_value_facts(
    loom_type_propagator_t* propagator, const loom_rewriter_t* rewriter,
    loom_op_t* op) {
  IREE_RETURN_IF_ERROR(loom_type_propagator_seed_value_span_facts(
      propagator, rewriter, loom_op_const_operands(op), op->operand_count));
  if (propagator->conflict) {
    return iree_ok_status();
  }
  return loom_type_propagator_seed_value_span_facts(
      propagator, rewriter, loom_op_const_results(op), op->result_count);
}

static iree_status_t loom_type_propagator_forward_source_tuple(
    loom_type_propagator_t* propagator,
    const loom_value_id_t* destination_values,
    const loom_value_id_t* source_values, uint16_t count) {
  loom_type_transfer_context_t context = {.propagator = propagator};
  for (uint16_t i = 0; i < count; ++i) {
    IREE_RETURN_IF_ERROR(loom_type_transfer_seed_static_structure_from_type(
        &context, source_values[i],
        loom_type_propagator_value_type(propagator, destination_values[i])));
    if (propagator->conflict) {
      return iree_ok_status();
    }
  }

  const loom_type_value_remap_t remap = {
      .source_values = destination_values,
      .target_values = source_values,
      .count = count,
      .flags = LOOM_TYPE_VALUE_REMAP_FLAG_SOURCE_DEFINITION_SLICE,
  };
  for (uint16_t i = 0; i < count; ++i) {
    if (!loom_type_equal_after_value_remap(
            propagator->module,
            loom_type_propagator_value_type(propagator, destination_values[i]),
            loom_type_propagator_value_type(propagator, source_values[i]),
            &remap)) {
      propagator->conflict = true;
      break;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_type_propagator_process_region_forwarding(
    loom_type_propagator_t* propagator, const loom_rewriter_t* rewriter,
    loom_op_t* op) {
  const loom_region_branch_t branch =
      loom_region_branch_cast(propagator->module, op);

  const loom_value_id_t* results = loom_op_const_results(op);
  IREE_RETURN_IF_ERROR(loom_type_propagator_seed_value_span_facts(
      propagator, rewriter, results, op->result_count));
  if (propagator->conflict) {
    return iree_ok_status();
  }

  for (uint8_t region_index = 0; region_index < op->region_count;
       ++region_index) {
    const loom_op_t* terminator = loom_region_branch_region_terminator(
        propagator->module, branch, region_index);
    IREE_ASSERT(terminator);
    IREE_ASSERT_EQ(terminator->operand_count, op->result_count);
    const loom_value_id_t* sources = loom_op_const_operands(terminator);
    IREE_RETURN_IF_ERROR(loom_type_propagator_forward_source_tuple(
        propagator, results, sources, op->result_count));
    if (propagator->conflict) {
      break;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_type_propagator_process_cfg_forwarding(
    loom_type_propagator_t* propagator, const loom_rewriter_t* rewriter,
    loom_block_t* target) {
  const loom_cfg_graph_t* graph =
      rewriter->fact_table ? loom_value_fact_table_lookup_cfg_graph(
                                 rewriter->fact_table, target->parent_region)
                           : NULL;
  if (!graph || graph->malformed) {
    propagator->conflict = true;
    return iree_ok_status();
  }
  const iree_host_size_t target_index =
      loom_cfg_graph_block_index(graph, target);
  IREE_ASSERT(target_index != IREE_HOST_SIZE_MAX);
  const loom_cfg_edge_index_span_t predecessor_edges =
      loom_cfg_graph_predecessor_edges(graph, (uint16_t)target_index);
  if (predecessor_edges.count == 0) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_type_propagator_seed_value_span_facts(
      propagator, rewriter, target->arg_ids, target->arg_count));
  if (propagator->conflict) {
    return iree_ok_status();
  }

  for (iree_host_size_t i = 0; i < predecessor_edges.count; ++i) {
    const loom_cfg_edge_info_t* edge =
        &graph->edges[predecessor_edges.values[i]];
    IREE_ASSERT_EQ(edge->terminator->successor_count, 1);
    IREE_ASSERT_EQ(edge->terminator->operand_count, target->arg_count);
    const loom_value_id_t* sources = loom_op_const_operands(edge->terminator);
    IREE_RETURN_IF_ERROR(loom_type_propagator_forward_source_tuple(
        propagator, target->arg_ids, sources, target->arg_count));
    if (propagator->conflict) {
      break;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_type_propagator_schedule_region_forwarding(
    loom_type_propagator_t* propagator, loom_op_t* op) {
  const loom_region_branch_t branch =
      loom_region_branch_cast(propagator->module, op);
  if (!loom_region_branch_isa(branch) || op->result_count == 0) {
    return iree_ok_status();
  }
  return loom_type_propagator_enqueue_forwarding(propagator,
                                                 loom_op_const_results(op)[0]);
}

static iree_status_t loom_type_propagator_schedule_cfg_forwarding(
    loom_type_propagator_t* propagator, loom_block_t* block) {
  if (!block || block->arg_count == 0) {
    return iree_ok_status();
  }
  return loom_type_propagator_enqueue_forwarding(propagator, block->arg_ids[0]);
}

static iree_status_t loom_type_propagator_schedule_forwarding(
    loom_type_propagator_t* propagator, loom_op_t* op) {
  IREE_RETURN_IF_ERROR(
      loom_type_propagator_schedule_region_forwarding(propagator, op));
  if (loom_type_propagator_op_is_terminator(propagator, op) && op->parent_op) {
    IREE_RETURN_IF_ERROR(loom_type_propagator_schedule_region_forwarding(
        propagator, op->parent_op));
  }
  loom_block_t* const* successors = loom_op_successors(op);
  for (uint8_t i = 0; i < op->successor_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_type_propagator_schedule_cfg_forwarding(
        propagator, successors[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_type_propagator_process_op_constraints(
    loom_type_propagator_t* propagator, const loom_rewriter_t* rewriter,
    loom_op_t* op) {
  if (!op || iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_type_propagator_record_op_region_owners(propagator, op));
  IREE_RETURN_IF_ERROR(
      loom_type_propagator_seed_op_value_facts(propagator, rewriter, op));
  if (propagator->conflict) {
    return iree_ok_status();
  }

  const loom_op_vtable_t* vtable = loom_op_vtable(propagator->module, op);
  if (!vtable) {
    return iree_ok_status();
  }

  if (vtable->constraint_count > 0 && vtable->constraints) {
    for (uint8_t i = vtable->operand_dictionary_count;
         i < vtable->constraint_count; ++i) {
      const loom_constraint_t* constraint = &vtable->constraints[i];
      switch ((enum loom_constraint_relation_e)constraint->relation) {
        case LOOM_RELATION_PAIRWISE_EQ: {
          IREE_RETURN_IF_ERROR(loom_type_propagator_relation_pairwise_eq(
              propagator, op, vtable, constraint));
          break;
        }
        case LOOM_RELATION_ALL_SAME: {
          IREE_RETURN_IF_ERROR(loom_type_propagator_relation_all_same(
              propagator, op, vtable, constraint));
          break;
        }
        case LOOM_RELATION_REGION_ARG_MATCH: {
          IREE_RETURN_IF_ERROR(loom_type_propagator_relation_region_arg_match(
              propagator, op, vtable, constraint));
          break;
        }
        case LOOM_RELATION_YIELD_MATCH: {
          IREE_RETURN_IF_ERROR(loom_type_propagator_relation_yield_match(
              propagator, op, vtable, constraint));
          break;
        }
        case LOOM_RELATION_VARIADIC_MATCH: {
          IREE_RETURN_IF_ERROR(loom_type_propagator_relation_variadic_match(
              propagator, op, vtable, constraint));
          break;
        }
        default:
          break;
      }
      if (propagator->conflict) {
        return iree_ok_status();
      }
    }
  }

  if (vtable->type_transfer) {
    loom_type_transfer_context_t context = {.propagator = propagator};
    IREE_RETURN_IF_ERROR(
        vtable->type_transfer(&context, propagator->module, op));
    if (propagator->conflict) {
      return iree_ok_status();
    }
  }
  return loom_type_propagator_schedule_forwarding(propagator, op);
}

static iree_status_t loom_type_propagator_process_use_constraints(
    loom_type_propagator_t* propagator, const loom_rewriter_t* rewriter,
    loom_op_t* op) {
  const loom_op_vtable_t* vtable = loom_op_vtable(propagator->module, op);
  const bool has_type_constraints =
      vtable && vtable->constraint_count > vtable->operand_dictionary_count;
  const bool has_type_transfer = vtable && vtable->type_transfer;
  const bool is_cfg_forwarding = op->successor_count > 0;
  const bool is_region_forwarding =
      loom_type_propagator_op_is_terminator(propagator, op) && op->parent_op &&
      loom_region_branch_isa(
          loom_region_branch_cast(propagator->module, op->parent_op));
  if ((is_cfg_forwarding || is_region_forwarding) && !has_type_constraints &&
      !has_type_transfer && op->region_count == 0) {
    return loom_type_propagator_schedule_forwarding(propagator, op);
  }
  return loom_type_propagator_process_op_constraints(propagator, rewriter, op);
}

static iree_status_t loom_type_propagator_process_def_constraints(
    loom_type_propagator_t* propagator, const loom_rewriter_t* rewriter,
    loom_value_t* value) {
  loom_op_t* def_op = loom_value_def_op(value);
  const loom_region_branch_t branch =
      loom_region_branch_cast(propagator->module, def_op);
  return loom_region_branch_isa(branch)
             ? loom_type_propagator_schedule_region_forwarding(propagator,
                                                               def_op)
             : loom_type_propagator_process_op_constraints(propagator, rewriter,
                                                           def_op);
}

static iree_status_t loom_type_propagator_process_value_adjacency(
    loom_type_propagator_t* propagator, const loom_rewriter_t* rewriter,
    loom_value_id_t value_id) {
  if (!loom_type_propagator_valid_value_id(propagator, value_id)) {
    return iree_ok_status();
  }

  loom_value_t* value = loom_module_value(propagator->module, value_id);
  if (loom_value_is_block_arg(value)) {
    IREE_RETURN_IF_ERROR(loom_type_propagator_schedule_cfg_forwarding(
        propagator, loom_value_def_block(value)));
    const loom_value_ordinal_t value_ordinal =
        loom_local_value_domain_try_ordinal(&propagator->value_domain,
                                            value_id);
    if (value_ordinal != LOOM_VALUE_ORDINAL_INVALID &&
        (iree_host_size_t)value_ordinal < propagator->ordinal_capacity &&
        propagator->owner_ops[value_ordinal]) {
      IREE_RETURN_IF_ERROR(loom_type_propagator_process_op_constraints(
          propagator, rewriter, propagator->owner_ops[value_ordinal]));
      if (propagator->conflict) {
        return iree_ok_status();
      }
    }
  } else {
    IREE_RETURN_IF_ERROR(loom_type_propagator_process_def_constraints(
        propagator, rewriter, value));
    if (propagator->conflict) {
      return iree_ok_status();
    }
  }

  const loom_use_t* uses = loom_value_uses(value);
  for (uint32_t i = 0; i < value->use_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_type_propagator_process_use_constraints(
        propagator, rewriter, loom_use_user_op(uses[i])));
    if (propagator->conflict) {
      return iree_ok_status();
    }
  }

  loom_type_use_iterator_t type_users;
  loom_module_value_type_users(propagator->module, value_id, &type_users);
  for (loom_value_id_t user_value_id = loom_type_users_next(&type_users);
       user_value_id != LOOM_VALUE_ID_INVALID;
       user_value_id = loom_type_users_next(&type_users)) {
    if (loom_type_propagator_valid_value_id(propagator, user_value_id)) {
      loom_value_t* user_value =
          loom_module_value(propagator->module, user_value_id);
      if (loom_value_is_block_arg(user_value)) {
        const loom_value_ordinal_t user_value_ordinal =
            loom_local_value_domain_try_ordinal(&propagator->value_domain,
                                                user_value_id);
        if (user_value_ordinal != LOOM_VALUE_ORDINAL_INVALID &&
            (iree_host_size_t)user_value_ordinal <
                propagator->ordinal_capacity &&
            propagator->owner_ops[user_value_ordinal]) {
          IREE_RETURN_IF_ERROR(loom_type_propagator_process_op_constraints(
              propagator, rewriter, propagator->owner_ops[user_value_ordinal]));
        }
      } else {
        IREE_RETURN_IF_ERROR(loom_type_propagator_process_def_constraints(
            propagator, rewriter, user_value));
      }
      if (propagator->conflict) {
        return iree_ok_status();
      }
      const loom_use_t* user_value_uses = loom_value_uses(user_value);
      for (uint32_t i = 0; i < user_value->use_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_type_propagator_process_use_constraints(
            propagator, rewriter, loom_use_user_op(user_value_uses[i])));
        if (propagator->conflict) {
          return iree_ok_status();
        }
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_type_propagator_commit(
    loom_type_propagator_t* propagator, loom_rewriter_t* rewriter,
    bool* out_changed) {
  *out_changed = false;
  for (iree_host_size_t i = 0; i < propagator->touched_count; ++i) {
    const loom_value_ordinal_t value_ordinal = propagator->touched_ordinals[i];
    loom_value_id_t value_id =
        propagator->value_domain.value_ids[value_ordinal];
    loom_type_t current_type =
        loom_module_value_type(propagator->module, value_id);
    loom_type_t candidate_type = propagator->candidate_types[value_ordinal];
    if (loom_type_equal(current_type, candidate_type)) {
      continue;
    }

    loom_type_t committed_type = current_type;
    loom_type_refinement_result_t result = LOOM_TYPE_REFINEMENT_UNCHANGED;
    IREE_RETURN_IF_ERROR(loom_type_refine_with_candidate(
        current_type, candidate_type, &propagator->module->arena,
        &committed_type, &result));
    if (result == LOOM_TYPE_REFINEMENT_CONFLICT) {
      IREE_ASSERT_UNREACHABLE(
          "accepted type propagation transaction conflicted during commit");
      IREE_BUILTIN_UNREACHABLE();
    }
    if (result == LOOM_TYPE_REFINEMENT_UNCHANGED ||
        loom_type_equal(current_type, committed_type)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_rewriter_set_value_type(rewriter, value_id, committed_type));
    *out_changed = true;
  }
  return iree_ok_status();
}

iree_status_t loom_type_propagator_apply_op(loom_type_propagator_t* propagator,
                                            loom_rewriter_t* rewriter,
                                            loom_op_t* op, bool* out_changed) {
  IREE_ASSERT(loom_local_value_domain_is_acquired(&propagator->value_domain));
  *out_changed = false;
  loom_type_propagator_next_transaction(propagator);
  IREE_RETURN_IF_ERROR(
      loom_type_propagator_process_op_constraints(propagator, rewriter, op));

  while (!propagator->conflict && (propagator->value_worklist_count > 0 ||
                                   propagator->forwarding_worklist_count > 0)) {
    if (propagator->value_worklist_count > 0) {
      const loom_value_ordinal_t value_ordinal =
          propagator->value_worklist[--propagator->value_worklist_count];
      propagator->queue_marks[value_ordinal].value_generation = 0;
      const loom_value_id_t value_id =
          propagator->value_domain.value_ids[value_ordinal];
      IREE_RETURN_IF_ERROR(loom_type_propagator_process_value_adjacency(
          propagator, rewriter, value_id));
      continue;
    }

    const loom_value_ordinal_t key_ordinal =
        propagator
            ->forwarding_worklist[--propagator->forwarding_worklist_count];
    propagator->queue_marks[key_ordinal].forwarding_generation = 0;
    const loom_value_id_t key_value_id =
        propagator->value_domain.value_ids[key_ordinal];
    loom_value_t* key_value =
        loom_module_value(propagator->module, key_value_id);
    if (loom_value_is_block_arg(key_value)) {
      IREE_RETURN_IF_ERROR(loom_type_propagator_process_cfg_forwarding(
          propagator, rewriter, loom_value_def_block(key_value)));
    } else {
      IREE_RETURN_IF_ERROR(loom_type_propagator_process_region_forwarding(
          propagator, rewriter, loom_value_def_op(key_value)));
    }
  }
  if (propagator->conflict) {
    return iree_ok_status();
  }
  return loom_type_propagator_commit(propagator, rewriter, out_changed);
}
