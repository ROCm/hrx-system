// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/motion_policy.h"

#include "loom/ir/context.h"
#include "loom/ops/op_defs.h"

bool loom_motion_traits_are_effect_free_relocatable(loom_trait_flags_t traits,
                                                    bool is_root_op) {
  if (!iree_any_bit_set(traits, LOOM_TRAIT_PURE)) {
    return false;
  }
  if (iree_any_bit_set(traits, LOOM_TRAIT_HINT | LOOM_TRAIT_CONVERGENT |
                                   LOOM_TRAIT_OBSERVABLE_EFFECT)) {
    return false;
  }
  if (is_root_op && iree_any_bit_set(traits, LOOM_TRAIT_TERMINATOR)) {
    return false;
  }
  return !loom_traits_may_read(traits) && !loom_traits_may_write(traits);
}

bool loom_motion_traits_are_speculation_compatible(loom_trait_flags_t traits,
                                                   bool is_root_op) {
  if (is_root_op && iree_any_bit_set(traits, LOOM_TRAIT_TERMINATOR)) {
    return false;
  }
  if (iree_any_bit_set(traits, LOOM_TRAIT_TERMINATOR)) {
    return loom_motion_traits_are_effect_free_relocatable(traits,
                                                          /*is_root_op=*/false);
  }
  if (!iree_any_bit_set(traits, LOOM_TRAIT_PURE)) {
    return false;
  }
  if (iree_any_bit_set(traits, LOOM_TRAIT_HINT | LOOM_TRAIT_CONVERGENT |
                                   LOOM_TRAIT_OBSERVABLE_EFFECT)) {
    return false;
  }
  if (loom_traits_has_unique_identity(traits)) {
    return false;
  }
  return !loom_traits_may_read(traits) && !loom_traits_may_write(traits);
}

bool loom_motion_op_has_retained_regions(const loom_op_t* op) {
  return loom_op_regions_have_read_effects(op) ||
         loom_op_regions_have_write_effects(op) ||
         loom_op_regions_have_convergent_effects(op) ||
         loom_op_regions_have_observable_effects(op) ||
         loom_op_regions_have_hints(op);
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
  if (op->tied_result_count != 0) {
    return false;
  }
  const loom_trait_flags_t traits = loom_op_effective_traits(module, op);
  if (!loom_motion_traits_are_effect_free_relocatable(traits,
                                                      /*is_root_op=*/true)) {
    return false;
  }
  if (loom_op_requires_source_context(module, op)) {
    return false;
  }
  return !loom_motion_op_has_retained_regions(op);
}

bool loom_motion_op_can_rematerialize_effect_free(const loom_module_t* module,
                                                  const loom_op_t* op) {
  if (!module || !op || iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD) ||
      op->tied_result_count != 0 || op->successor_count != 0 ||
      op->region_count != 0) {
    return false;
  }
  const loom_trait_flags_t traits = loom_op_effective_traits(module, op);
  if (!iree_any_bit_set(traits, LOOM_TRAIT_PURE) ||
      loom_traits_may_read(traits) || loom_traits_may_write(traits) ||
      iree_any_bit_set(
          traits,
          LOOM_TRAIT_TERMINATOR | LOOM_TRAIT_HINT | LOOM_TRAIT_POISON_BOUNDARY |
              LOOM_TRAIT_CONVERGENT | LOOM_TRAIT_OBSERVABLE_EFFECT |
              LOOM_TRAIT_UNIQUE_IDENTITY | LOOM_TRAIT_CALLABLE_BOUNDARY)) {
    return false;
  }
  return !loom_op_requires_source_context(module, op);
}

bool loom_motion_op_is_speculation_compatible_with_traits(
    const loom_module_t* module, const loom_op_t* op,
    loom_trait_flags_t traits) {
  if (!module || !op || iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) {
    return false;
  }
  if (op->tied_result_count != 0) {
    return false;
  }
  if (!loom_motion_traits_are_speculation_compatible(traits,
                                                     /*is_root_op=*/true)) {
    return false;
  }
  if (loom_op_requires_source_context(module, op)) {
    return false;
  }
  return op->region_count == 0 || !loom_motion_op_has_retained_regions(op);
}

bool loom_motion_op_can_speculate_with_traits(const loom_module_t* module,
                                              const loom_op_t* op,
                                              loom_trait_flags_t traits) {
  return loom_motion_op_is_speculation_compatible_with_traits(module, op,
                                                              traits) &&
         loom_traits_are_safe_to_speculate(traits);
}
