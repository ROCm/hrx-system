// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/descriptor_traits.h"

static bool loom_low_descriptor_defines_implicit_state_result(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  for (uint16_t i = 0; i < descriptor->operand_count; ++i) {
    const loom_low_operand_t* operand =
        loom_low_descriptor_operand(descriptor_set, descriptor, i);
    if (operand->role == LOOM_LOW_OPERAND_ROLE_RESULT &&
        iree_all_bits_set(operand->flags,
                          LOOM_LOW_OPERAND_FLAG_IMPLICIT |
                              LOOM_LOW_OPERAND_FLAG_STATE_WRITE)) {
      return true;
    }
  }
  return false;
}

static loom_memory_space_mask_t loom_low_descriptor_memory_space(
    loom_low_memory_space_t space) {
  switch (space) {
    case LOOM_LOW_MEMORY_SPACE_NONE:
      return 0;
    case LOOM_LOW_MEMORY_SPACE_GLOBAL:
      return 1u << LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL;
    case LOOM_LOW_MEMORY_SPACE_WORKGROUP:
      return 1u << LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP;
    case LOOM_LOW_MEMORY_SPACE_STACK:
      return 1u << LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE;
    case LOOM_LOW_MEMORY_SPACE_GENERIC:
    case LOOM_LOW_MEMORY_SPACE_WASM_MEMORY:
      return 1u << LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN;
    default:
      IREE_ASSERT_UNREACHABLE("invalid descriptor memory space");
      IREE_BUILTIN_UNREACHABLE();
  }
}

loom_trait_flags_t loom_low_descriptor_effective_traits(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor,
    loom_memory_space_mask_t* out_intrinsic_read_spaces,
    loom_memory_space_mask_t* out_intrinsic_write_spaces) {
  loom_trait_flags_t traits = 0;
  loom_memory_space_mask_t read_spaces = 0;
  loom_memory_space_mask_t write_spaces = 0;
  const loom_low_descriptor_layout_t* layout =
      loom_low_descriptor_layout(descriptor_set, descriptor);
  for (uint16_t i = 0; i < layout->effect_count; ++i) {
    const uint32_t effect_index = layout->effect_start + i;
    IREE_ASSERT(effect_index < descriptor_set->effect_count);
    const loom_low_effect_t* effect = &descriptor_set->effects[effect_index];
    switch (effect->kind) {
      case LOOM_LOW_EFFECT_KIND_READ:
        read_spaces |= loom_low_descriptor_memory_space(effect->memory_space);
        traits |= effect->memory_space == LOOM_LOW_MEMORY_SPACE_NONE
                      ? LOOM_TRAIT_NON_DETERMINISTIC
                      : LOOM_TRAIT_READS_MEMORY;
        if (iree_any_bit_set(effect->flags, LOOM_LOW_EFFECT_FLAG_ORDERED)) {
          traits |= LOOM_TRAIT_OBSERVABLE_EFFECT;
        }
        break;
      case LOOM_LOW_EFFECT_KIND_WRITE:
        write_spaces |= loom_low_descriptor_memory_space(effect->memory_space);
        traits |= effect->memory_space == LOOM_LOW_MEMORY_SPACE_NONE
                      ? LOOM_TRAIT_OBSERVABLE_EFFECT
                      : LOOM_TRAIT_WRITES_MEMORY;
        if (iree_any_bit_set(effect->flags, LOOM_LOW_EFFECT_FLAG_ORDERED)) {
          traits |= LOOM_TRAIT_OBSERVABLE_EFFECT;
        }
        break;
      case LOOM_LOW_EFFECT_KIND_CONTROL:
        traits |= LOOM_TRAIT_TERMINATOR;
        break;
      case LOOM_LOW_EFFECT_KIND_CALL:
        traits |= LOOM_TRAIT_UNKNOWN_EFFECTS;
        break;
      case LOOM_LOW_EFFECT_KIND_COUNTER:
      case LOOM_LOW_EFFECT_KIND_BARRIER:
        traits |= LOOM_TRAIT_MEMORY_FENCE;
        break;
      case LOOM_LOW_EFFECT_KIND_CONVERGENT:
        traits |= LOOM_TRAIT_CONVERGENT;
        break;
      default:
        traits |= LOOM_TRAIT_UNKNOWN_EFFECTS;
        break;
    }
  }

  if (iree_any_bit_set(descriptor->flags,
                       LOOM_LOW_DESCRIPTOR_FLAG_TERMINATOR)) {
    traits |= LOOM_TRAIT_TERMINATOR;
  }
  if (iree_any_bit_set(descriptor->flags,
                       LOOM_LOW_DESCRIPTOR_FLAG_UNIQUE_IDENTITY)) {
    traits |= LOOM_TRAIT_UNIQUE_IDENTITY;
  }
  if (iree_any_bit_set(descriptor->flags,
                       LOOM_LOW_DESCRIPTOR_FLAG_STATE_ASSIGNMENT)) {
    // The descriptor guarantees that its sole effect is replacing one whole
    // architectural register. Preserve that effect without inventing storage
    // writes or acquisition barriers.
    traits |= LOOM_TRAIT_OBSERVABLE_EFFECT;
  }
  if (loom_low_descriptor_defines_implicit_state_result(descriptor_set,
                                                        descriptor)) {
    // Each definition owns a distinct architectural-register lifetime. Its
    // value is deterministic, but merging it with another definition would
    // extend that lifetime through intervening clobbers. This is local register
    // storage, not an observation of external state or a callable resource.
    traits |= LOOM_TRAIT_UNIQUE_IDENTITY | LOOM_TRAIT_STORAGE_RELATION;
  }
  if (out_intrinsic_read_spaces) {
    *out_intrinsic_read_spaces = read_spaces;
  }
  if (out_intrinsic_write_spaces) {
    *out_intrinsic_write_spaces = write_spaces;
  }
  if (iree_any_bit_set(traits, LOOM_TRAIT_UNKNOWN_EFFECTS)) {
    // Unknown effects subsume generic reads and writes. The scheduler still
    // consumes the precise accesses from the descriptor's effect rows.
    return traits & ~(LOOM_TRAIT_READS_MEMORY | LOOM_TRAIT_WRITES_MEMORY);
  }

  if (!iree_any_bit_set(
          traits, LOOM_TRAIT_READS_MEMORY | LOOM_TRAIT_WRITES_MEMORY |
                      LOOM_TRAIT_NON_DETERMINISTIC | LOOM_TRAIT_MEMORY_FENCE |
                      LOOM_TRAIT_UNIQUE_IDENTITY |
                      LOOM_TRAIT_OBSERVABLE_EFFECT)) {
    if (layout->effect_count == 0 &&
        iree_any_bit_set(descriptor->flags,
                         LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING)) {
      traits |= LOOM_TRAIT_UNKNOWN_EFFECTS;
    } else {
      traits |= LOOM_TRAIT_PURE;
    }
  }
  return traits;
}

bool loom_low_descriptor_result_can_rematerialize(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint16_t result_index) {
  return iree_any_bit_set(
      loom_low_descriptor_operand(descriptor_set, descriptor, result_index)
          ->flags,
      LOOM_LOW_OPERAND_FLAG_REMATERIALIZABLE);
}

uint16_t loom_low_descriptor_result_rematerialization_source_operand(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint16_t result_index) {
  const uint16_t source_plus_one =
      loom_low_descriptor_operand(descriptor_set, descriptor, result_index)
          ->rematerialization_source_operand_plus_one;
  return source_plus_one == 0 ? LOOM_LOW_ID_NONE : source_plus_one - 1;
}
