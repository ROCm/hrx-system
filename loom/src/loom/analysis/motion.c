// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/motion.h"

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

iree_status_t loom_motion_analysis_initialize(
    const loom_module_t* module, iree_arena_allocator_t* arena,
    loom_motion_analysis_t* out_analysis) {
  out_analysis->module = module;
  out_analysis->arena = arena;
  IREE_RETURN_IF_ERROR(loom_availability_analysis_initialize(
      module, arena, &out_analysis->availability));
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
  if (iree_any_bit_set(traits, LOOM_TRAIT_HINT | LOOM_TRAIT_CONVERGENT)) {
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
  if (iree_any_bit_set(traits, LOOM_TRAIT_HINT | LOOM_TRAIT_CONVERGENT)) {
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
