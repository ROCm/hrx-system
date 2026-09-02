// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/motion.h"

#include "loom/analysis/loop_domain.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"

//===----------------------------------------------------------------------===//
// Scratch stacks
//===----------------------------------------------------------------------===//

#define LOOM_MOTION_INITIAL_REGION_STACK_CAPACITY 16

static iree_status_t loom_motion_region_stack_initialize(
    iree_arena_allocator_t* arena, loom_motion_region_stack_t* stack) {
  stack->count = 0;
  stack->capacity = LOOM_MOTION_INITIAL_REGION_STACK_CAPACITY;
  return iree_arena_allocate_array(
      arena, stack->capacity, sizeof(loom_region_t*), (void**)&stack->regions);
}

static iree_status_t loom_motion_region_stack_push(
    iree_arena_allocator_t* arena, loom_motion_region_stack_t* stack,
    loom_region_t* region) {
  if (!region || region->block_count == 0) return iree_ok_status();
  if (stack->count >= stack->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, stack->count, stack->count + 1, sizeof(loom_region_t*),
        &stack->capacity, (void**)&stack->regions));
  }
  stack->regions[stack->count++] = region;
  return iree_ok_status();
}

static loom_region_t* loom_motion_region_stack_pop(
    loom_motion_region_stack_t* stack) {
  return stack->count > 0 ? stack->regions[--stack->count] : NULL;
}

//===----------------------------------------------------------------------===//
// Analysis state
//===----------------------------------------------------------------------===//

iree_status_t loom_motion_analysis_initialize_region(
    const loom_module_t* module, const loom_region_t* region,
    loom_value_fact_table_t* fact_table,
    loom_local_value_domain_t* value_domain, iree_arena_allocator_t* arena,
    loom_motion_analysis_t* out_analysis) {
  *out_analysis = (loom_motion_analysis_t){
      .module = module,
      .arena = arena,
      .fact_table = fact_table,
      .value_domain = value_domain,
  };
  IREE_RETURN_IF_ERROR(loom_availability_analysis_initialize_region(
      module, region, arena, &out_analysis->availability));
  if (fact_table && value_domain) {
    IREE_RETURN_IF_ERROR(loom_movement_analysis_initialize(
        fact_table, value_domain, arena, &out_analysis->movement));
    out_analysis->movement_initialized = true;
  }
  return loom_motion_region_stack_initialize(arena,
                                             &out_analysis->region_stack);
}

//===----------------------------------------------------------------------===//
// Ancestry and local classification
//===----------------------------------------------------------------------===//

static bool loom_motion_op_is_nested_under(const loom_op_t* root,
                                           const loom_op_t* op) {
  if (!root || !op) return false;
  for (const loom_op_t* current = op; current; current = current->parent_op) {
    if (current == root) return true;
  }
  return false;
}

static bool loom_motion_traits_are_effect_free_relocatable(
    loom_trait_flags_t traits, bool is_root_op) {
  if (!iree_any_bit_set(traits, LOOM_TRAIT_PURE)) return false;
  if (iree_any_bit_set(traits, LOOM_TRAIT_HINT | LOOM_TRAIT_CONVERGENT |
                                   LOOM_TRAIT_OBSERVABLE_EFFECT)) {
    return false;
  }
  if (is_root_op && iree_any_bit_set(traits, LOOM_TRAIT_TERMINATOR)) {
    return false;
  }
  return !loom_traits_may_read(traits) && !loom_traits_may_write(traits);
}

static bool loom_motion_traits_are_speculatable(loom_trait_flags_t traits,
                                                bool is_root_op) {
  if (is_root_op && iree_any_bit_set(traits, LOOM_TRAIT_TERMINATOR)) {
    return false;
  }
  if (iree_any_bit_set(traits, LOOM_TRAIT_TERMINATOR)) {
    return loom_motion_traits_are_effect_free_relocatable(traits,
                                                          /*is_root_op=*/false);
  }
  if (!iree_any_bit_set(traits, LOOM_TRAIT_PURE)) return false;
  if (!loom_traits_are_safe_to_speculate(traits)) return false;
  if (iree_any_bit_set(traits, LOOM_TRAIT_HINT | LOOM_TRAIT_CONVERGENT |
                                   LOOM_TRAIT_OBSERVABLE_EFFECT)) {
    return false;
  }
  if (loom_traits_has_unique_identity(traits)) return false;
  return !loom_traits_may_read(traits) && !loom_traits_may_write(traits);
}

static bool loom_motion_op_has_retained_regions(const loom_module_t* module,
                                                const loom_op_t* op) {
  return loom_op_regions_have_read_effects(op) ||
         loom_op_regions_have_write_effects(op) ||
         loom_op_regions_have_convergent_effects(op) ||
         loom_op_regions_have_observable_effects(op) ||
         loom_op_regions_have_hints(module, op);
}

bool loom_motion_op_can_erase(const loom_module_t* module,
                              const loom_op_t* op) {
  if (!module || !op || iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) {
    return false;
  }
  return loom_op_is_trivially_dead(module, op);
}

bool loom_motion_op_can_relocate_effect_free(const loom_module_t* module,
                                             const loom_op_t* op) {
  if (!module || !op || iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) {
    return false;
  }
  if (op->tied_result_count != 0) return false;
  loom_trait_flags_t traits = loom_op_effective_traits(module, op);
  if (!loom_motion_traits_are_effect_free_relocatable(traits,
                                                      /*is_root_op=*/true)) {
    return false;
  }
  return !loom_motion_op_has_retained_regions(module, op);
}

bool loom_motion_op_can_rematerialize_effect_free(const loom_module_t* module,
                                                  const loom_op_t* op) {
  if (!module || !op || iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD) ||
      op->tied_result_count != 0 || op->region_count != 0) {
    return false;
  }
  const loom_trait_flags_t traits = loom_op_effective_traits(module, op);
  if (!iree_any_bit_set(traits, LOOM_TRAIT_PURE) ||
      loom_traits_may_read(traits) || loom_traits_may_write(traits) ||
      iree_any_bit_set(traits, LOOM_TRAIT_TERMINATOR | LOOM_TRAIT_HINT |
                                   LOOM_TRAIT_POISON_BOUNDARY |
                                   LOOM_TRAIT_CONVERGENT |
                                   LOOM_TRAIT_OBSERVABLE_EFFECT |
                                   LOOM_TRAIT_UNIQUE_IDENTITY)) {
    return false;
  }
  return true;
}

bool loom_motion_op_can_speculate(const loom_module_t* module,
                                  const loom_op_t* op) {
  if (!module || !op || iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) {
    return false;
  }
  if (op->tied_result_count != 0) return false;
  loom_trait_flags_t traits = loom_op_effective_traits(module, op);
  if (!loom_motion_traits_are_speculatable(traits, /*is_root_op=*/true)) {
    return false;
  }
  return !loom_motion_op_has_retained_regions(module, op);
}

bool loom_motion_read_can_cross_op(const loom_module_t* module,
                                   const loom_op_t* op) {
  if (!module || !op) return false;
  if (iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) return true;
  if (op->region_count != 0) return false;
  const loom_trait_flags_t traits = loom_op_effective_traits(module, op);
  if (loom_traits_may_write(traits)) return false;
  return !iree_any_bit_set(
      traits, LOOM_TRAIT_NON_DETERMINISTIC | LOOM_TRAIT_HINT |
                  LOOM_TRAIT_POISON_BOUNDARY | LOOM_TRAIT_CONVERGENT |
                  LOOM_TRAIT_OBSERVABLE_EFFECT | LOOM_TRAIT_UNIQUE_IDENTITY);
}

//===----------------------------------------------------------------------===//
// Subtree motion
//===----------------------------------------------------------------------===//

typedef enum loom_motion_policy_e {
  // Relocation preserves the caller-proven dynamic execution predicate.
  LOOM_MOTION_POLICY_EFFECT_FREE_RELOCATION = 0,
  // Speculation may execute the subtree on additional control paths.
  LOOM_MOTION_POLICY_SPECULATION = 1,
} loom_motion_policy_t;

static bool loom_motion_op_satisfies_policy(const loom_module_t* module,
                                            const loom_op_t* op,
                                            loom_motion_policy_t policy,
                                            bool is_root_op) {
  if (!module || !op || iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) {
    return false;
  }
  if (op->tied_result_count != 0) return false;
  loom_trait_flags_t traits = loom_op_effective_traits(module, op);
  switch (policy) {
    case LOOM_MOTION_POLICY_EFFECT_FREE_RELOCATION:
      return loom_motion_traits_are_effect_free_relocatable(traits, is_root_op);
    case LOOM_MOTION_POLICY_SPECULATION:
      return loom_motion_traits_are_speculatable(traits, is_root_op);
  }
  return false;
}

static iree_status_t loom_motion_subtree_can_move_before(
    loom_motion_analysis_t* analysis, const loom_op_t* candidate_op,
    const loom_op_t* before_op, loom_motion_policy_t policy,
    bool* out_can_move) {
  *out_can_move = false;
  if (!analysis || !analysis->module || !candidate_op || !before_op) {
    return iree_ok_status();
  }
  if (iree_any_bit_set(before_op->flags, LOOM_OP_FLAG_DEAD)) {
    return iree_ok_status();
  }
  if (candidate_op == before_op ||
      loom_motion_op_is_nested_under(candidate_op, before_op)) {
    return iree_ok_status();
  }
  if (!loom_motion_op_satisfies_policy(analysis->module, candidate_op, policy,
                                       /*is_root_op=*/true)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_availability_op_captures_are_available_before_op(
      &analysis->availability, candidate_op, before_op, candidate_op,
      out_can_move));
  if (!*out_can_move) return iree_ok_status();

  analysis->region_stack.count = 0;
  loom_region_t** regions = loom_op_regions(candidate_op);
  for (uint8_t i = 0; i < candidate_op->region_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_motion_region_stack_push(
        analysis->arena, &analysis->region_stack, regions[i]));
  }

  while (true) {
    loom_region_t* region =
        loom_motion_region_stack_pop(&analysis->region_stack);
    if (!region) break;

    loom_block_t* block = NULL;
    loom_region_for_each_block(region, block) {
      IREE_RETURN_IF_ERROR(
          loom_availability_block_arg_types_are_available_before_op(
              &analysis->availability, candidate_op, before_op, block,
              out_can_move));
      if (!*out_can_move) return iree_ok_status();

      loom_op_t* child_op = NULL;
      loom_block_for_each_op(block, child_op) {
        if (!loom_motion_op_satisfies_policy(analysis->module, child_op, policy,
                                             /*is_root_op=*/false)) {
          *out_can_move = false;
          return iree_ok_status();
        }
        IREE_RETURN_IF_ERROR(
            loom_availability_op_captures_are_available_before_op(
                &analysis->availability, candidate_op, before_op, child_op,
                out_can_move));
        if (!*out_can_move) return iree_ok_status();

        loom_region_t** child_regions = loom_op_regions(child_op);
        for (uint8_t i = 0; i < child_op->region_count; ++i) {
          IREE_RETURN_IF_ERROR(loom_motion_region_stack_push(
              analysis->arena, &analysis->region_stack, child_regions[i]));
        }
      }
    }
  }

  *out_can_move = true;
  return iree_ok_status();
}

iree_status_t loom_motion_subtree_can_relocate_before(
    loom_motion_analysis_t* analysis, const loom_op_t* candidate_op,
    const loom_op_t* before_op, bool* out_can_relocate) {
  return loom_motion_subtree_can_move_before(
      analysis, candidate_op, before_op,
      LOOM_MOTION_POLICY_EFFECT_FREE_RELOCATION, out_can_relocate);
}

iree_status_t loom_motion_subtree_can_speculate_before(
    loom_motion_analysis_t* analysis, const loom_op_t* candidate_op,
    const loom_op_t* before_op, bool* out_can_speculate) {
  return loom_motion_subtree_can_move_before(analysis, candidate_op, before_op,
                                             LOOM_MOTION_POLICY_SPECULATION,
                                             out_can_speculate);
}

//===----------------------------------------------------------------------===//
// Loop hoist evaluation
//===----------------------------------------------------------------------===//

static void loom_motion_loop_hoist_reject(
    loom_motion_loop_hoist_result_t* result,
    loom_motion_loop_hoist_rejection_flags_t rejection,
    const loom_op_t* blocking_op) {
  result->rejection_bits |= rejection;
  if (!result->blocking_op) result->blocking_op = blocking_op;
}

static iree_status_t loom_motion_analysis_ensure_movement(
    loom_motion_analysis_t* analysis, bool* out_available) {
  *out_available = analysis->movement_initialized;
  if (!*out_available || analysis->movement_analyzed) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_movement_analysis_analyze(&analysis->movement));
  analysis->movement_analyzed = true;
  return iree_ok_status();
}

static iree_status_t loom_motion_evaluate_speculative_subtree(
    loom_motion_analysis_t* analysis, const loom_op_t* candidate_op,
    const loom_op_t* loop_op, loom_motion_loop_hoist_result_t* result) {
  if (!loom_motion_op_can_speculate(analysis->module, candidate_op)) {
    loom_motion_loop_hoist_reject(
        result, LOOM_MOTION_LOOP_HOIST_REJECTION_CANDIDATE_SEMANTICS,
        candidate_op);
    return iree_ok_status();
  }
  bool captures_available = false;
  IREE_RETURN_IF_ERROR(loom_availability_op_captures_are_available_before_op(
      &analysis->availability, candidate_op, loop_op, candidate_op,
      &captures_available));
  if (!captures_available) {
    loom_motion_loop_hoist_reject(
        result, LOOM_MOTION_LOOP_HOIST_REJECTION_CAPTURE_UNAVAILABLE,
        candidate_op);
    return iree_ok_status();
  }
  bool can_speculate = false;
  IREE_RETURN_IF_ERROR(loom_motion_subtree_can_speculate_before(
      analysis, candidate_op, loop_op, &can_speculate));
  if (!can_speculate) {
    loom_motion_loop_hoist_reject(
        result, LOOM_MOTION_LOOP_HOIST_REJECTION_CANDIDATE_SEMANTICS,
        candidate_op);
  }
  return iree_ok_status();
}

bool loom_motion_op_is_ordinary_load(const loom_module_t* module,
                                     const loom_op_t* op) {
  if (!module || !op || iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD) ||
      op->region_count != 0 || op->tied_result_count != 0) {
    return false;
  }
  const loom_trait_flags_t traits = loom_op_effective_traits(module, op);
  if (!loom_traits_may_read(traits) || loom_traits_may_write(traits) ||
      iree_any_bit_set(
          traits,
          LOOM_TRAIT_TERMINATOR | LOOM_TRAIT_HINT | LOOM_TRAIT_CONVERGENT |
              LOOM_TRAIT_UNIQUE_IDENTITY | LOOM_TRAIT_POISON_BOUNDARY |
              LOOM_TRAIT_MEMORY_FENCE | LOOM_TRAIT_NON_DETERMINISTIC |
              LOOM_TRAIT_OBSERVABLE_EFFECT | LOOM_TRAIT_UNKNOWN_EFFECTS)) {
    return false;
  }
  const loom_memory_access_t access = loom_memory_access_cast(module, op);
  return loom_memory_access_isa(access) &&
         loom_memory_access_operation_kind(access) ==
             LOOM_MEMORY_ACCESS_OPERATION_LOAD;
}

static bool loom_motion_request_is_ordinary_load(
    const loom_movement_request_t* request) {
  if (!request || request->source.kind != LOOM_MOVEMENT_ENDPOINT_VIEW) {
    return false;
  }
  if (request->kind != LOOM_MOVEMENT_KIND_VIEW_LOAD &&
      request->kind != LOOM_MOVEMENT_KIND_VECTOR_LOAD) {
    return false;
  }
  return !iree_any_bit_set(request->flags,
                           LOOM_MOVEMENT_REQUEST_MASKED |
                               LOOM_MOVEMENT_REQUEST_ASYNC |
                               LOOM_MOVEMENT_REQUEST_IRREGULAR_OFFSETS);
}

// Queries the conventional memory-space domain carried by a fence op. The
// MemoryFence trait itself remains conservative: a fence with no precise
// memory_space enum is treated as ordering every memory space. This keeps the
// legality layer target-neutral without requiring it to recognize each
// dialect's concrete fence operation.
static bool loom_motion_memory_fence_space(
    const loom_module_t* module, const loom_op_t* op,
    loom_value_fact_memory_space_t* out_memory_space) {
  *out_memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN;
  if (!module || !op ||
      !loom_traits_order_memory(loom_op_effective_traits(module, op))) {
    return false;
  }
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  if (!vtable || !vtable->attr_descriptors) return false;
  for (uint8_t i = 0; i < vtable->attribute_count && i < op->attribute_count;
       ++i) {
    const loom_attr_descriptor_t* descriptor = &vtable->attr_descriptors[i];
    if (descriptor->attr_kind != LOOM_ATTR_ENUM ||
        !iree_string_view_equal(loom_attr_descriptor_name(descriptor),
                                IREE_SV("memory_space"))) {
      continue;
    }
    const loom_attribute_t attr = loom_op_const_attrs(op)[i];
    if (attr.kind != LOOM_ATTR_ENUM) return false;
    const uint8_t value = loom_attr_as_enum(attr);
    if (value > LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC) return false;
    *out_memory_space = (loom_value_fact_memory_space_t)value;
    return true;
  }
  return false;
}

static iree_status_t loom_motion_loop_writes_are_disjoint(
    loom_motion_analysis_t* analysis, loom_loop_like_t loop,
    const loom_op_t* candidate_op,
    const loom_movement_endpoint_t* read_endpoint,
    loom_motion_loop_hoist_result_t* result) {
  loom_view_region_t read_region = {0};
  if (!loom_movement_endpoint_as_view_region(read_endpoint, &read_region)) {
    loom_motion_loop_hoist_reject(
        result, LOOM_MOTION_LOOP_HOIST_REJECTION_MOVEMENT_UNAVAILABLE,
        candidate_op);
    return iree_ok_status();
  }

  analysis->region_stack.count = 0;
  IREE_RETURN_IF_ERROR(loom_motion_region_stack_push(
      analysis->arena, &analysis->region_stack, loom_loop_like_body(loop)));
  while (true) {
    loom_region_t* region =
        loom_motion_region_stack_pop(&analysis->region_stack);
    if (!region) break;

    loom_block_t* block = NULL;
    loom_region_for_each_block(region, block) {
      loom_op_t* op = NULL;
      loom_block_for_each_op(block, op) {
        if (op == candidate_op ||
            iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) {
          continue;
        }
        const loom_trait_flags_t traits =
            loom_op_effective_traits(analysis->module, op);
        if (iree_any_bit_set(traits, LOOM_TRAIT_UNKNOWN_EFFECTS)) {
          loom_motion_loop_hoist_reject(
              result, LOOM_MOTION_LOOP_HOIST_REJECTION_UNKNOWN_INTERFERENCE,
              op);
          return iree_ok_status();
        }
        if (loom_traits_order_memory(traits)) {
          loom_value_fact_memory_space_t fence_memory_space =
              LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN;
          if (!loom_motion_memory_fence_space(analysis->module, op,
                                              &fence_memory_space) ||
              !loom_view_memory_spaces_are_disjoint(read_region.memory_space,
                                                    fence_memory_space)) {
            loom_motion_loop_hoist_reject(
                result, LOOM_MOTION_LOOP_HOIST_REJECTION_ORDERING_INTERFERENCE,
                op);
            return iree_ok_status();
          }
        }
        if (iree_any_bit_set(traits, LOOM_TRAIT_WRITES_MEMORY)) {
          loom_movement_request_t request = {0};
          loom_movement_diagnostic_t diagnostic = {0};
          bool described = false;
          IREE_RETURN_IF_ERROR(loom_movement_request_describe_op(
              &analysis->movement, op, &request, &diagnostic, &described));
          if (!described || request.dest.kind != LOOM_MOVEMENT_ENDPOINT_VIEW ||
              !iree_any_bit_set(request.dest.flags,
                                LOOM_MOVEMENT_ENDPOINT_WRITE)) {
            result->movement_rejection_bits |= diagnostic.rejection_bits;
            loom_motion_loop_hoist_reject(
                result, LOOM_MOTION_LOOP_HOIST_REJECTION_UNKNOWN_INTERFERENCE,
                op);
            return iree_ok_status();
          }
          loom_view_region_t write_region = {0};
          if (!loom_movement_endpoint_as_view_region(&request.dest,
                                                     &write_region)) {
            loom_motion_loop_hoist_reject(
                result, LOOM_MOTION_LOOP_HOIST_REJECTION_UNKNOWN_INTERFERENCE,
                op);
            return iree_ok_status();
          }
          bool no_overlap = false;
          IREE_RETURN_IF_ERROR(loom_view_regions_prove_no_overlap(
              &analysis->movement.view_regions, &read_region, &write_region,
              &no_overlap));
          if (!no_overlap) {
            loom_motion_loop_hoist_reject(
                result, LOOM_MOTION_LOOP_HOIST_REJECTION_OVERLAPPING_WRITE, op);
            return iree_ok_status();
          }
        }

        loom_region_t** child_regions = loom_op_regions(op);
        for (uint8_t i = 0; i < op->region_count; ++i) {
          IREE_RETURN_IF_ERROR(loom_motion_region_stack_push(
              analysis->arena, &analysis->region_stack, child_regions[i]));
        }
      }
    }
  }
  return iree_ok_status();
}

iree_status_t loom_motion_subtree_evaluate_hoist_before_loop(
    loom_motion_analysis_t* analysis, loom_loop_like_t loop,
    const loom_op_t* candidate_op,
    loom_motion_loop_hoist_result_t* out_result) {
  *out_result = (loom_motion_loop_hoist_result_t){0};
  if (!analysis || !analysis->module || !loom_loop_like_isa(loop) ||
      !candidate_op || !loop.op || candidate_op == loop.op ||
      iree_any_bit_set(loop.op->flags, LOOM_OP_FLAG_DEAD)) {
    loom_motion_loop_hoist_reject(
        out_result, LOOM_MOTION_LOOP_HOIST_REJECTION_INVALID_REQUEST,
        candidate_op);
    return iree_ok_status();
  }

  if (!loom_motion_op_is_ordinary_load(analysis->module, candidate_op)) {
    return loom_motion_evaluate_speculative_subtree(analysis, candidate_op,
                                                    loop.op, out_result);
  }

  loom_region_t* loop_body = loom_loop_like_body(loop);
  if (!loop_body || !candidate_op->parent_block ||
      candidate_op->parent_block->parent_region != loop_body) {
    loom_motion_loop_hoist_reject(
        out_result, LOOM_MOTION_LOOP_HOIST_REJECTION_PREDICATE_CROSSING,
        candidate_op);
    return iree_ok_status();
  }
  if (!loom_loop_like_has_counted_range(loop)) {
    loom_motion_loop_hoist_reject(
        out_result, LOOM_MOTION_LOOP_HOIST_REJECTION_LOOP_MAY_NOT_EXECUTE,
        loop.op);
    return iree_ok_status();
  }
  const loom_loop_domain_t domain = {
      .lower_bound = loom_loop_like_lower_bound(loop),
      .upper_bound = loom_loop_like_upper_bound(loop),
      .step = loom_loop_like_step(loop),
  };
  if (!loom_loop_domain_proven_nonempty(analysis->fact_table, domain)) {
    loom_motion_loop_hoist_reject(
        out_result, LOOM_MOTION_LOOP_HOIST_REJECTION_LOOP_MAY_NOT_EXECUTE,
        loop.op);
    return iree_ok_status();
  }

  bool captures_available = false;
  IREE_RETURN_IF_ERROR(loom_availability_op_captures_are_available_before_op(
      &analysis->availability, candidate_op, loop.op, candidate_op,
      &captures_available));
  if (!captures_available) {
    loom_motion_loop_hoist_reject(
        out_result, LOOM_MOTION_LOOP_HOIST_REJECTION_CAPTURE_UNAVAILABLE,
        candidate_op);
    return iree_ok_status();
  }

  bool movement_available = false;
  IREE_RETURN_IF_ERROR(
      loom_motion_analysis_ensure_movement(analysis, &movement_available));
  if (!movement_available) {
    loom_motion_loop_hoist_reject(
        out_result, LOOM_MOTION_LOOP_HOIST_REJECTION_MOVEMENT_UNAVAILABLE,
        candidate_op);
    return iree_ok_status();
  }

  loom_movement_request_t request = {0};
  loom_movement_diagnostic_t diagnostic = {0};
  bool described = false;
  IREE_RETURN_IF_ERROR(loom_movement_request_describe_op(
      &analysis->movement, candidate_op, &request, &diagnostic, &described));
  if (!described || !loom_motion_request_is_ordinary_load(&request)) {
    out_result->movement_rejection_bits = diagnostic.rejection_bits;
    loom_motion_loop_hoist_reject(
        out_result, LOOM_MOTION_LOOP_HOIST_REJECTION_MOVEMENT_UNAVAILABLE,
        candidate_op);
    return iree_ok_status();
  }
  return loom_motion_loop_writes_are_disjoint(analysis, loop, candidate_op,
                                              &request.source, out_result);
}
