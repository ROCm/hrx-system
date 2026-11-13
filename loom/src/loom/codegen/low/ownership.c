// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/ownership.h"

#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/representation_binding.h"
#include "loom/ir/module.h"

static uint8_t loom_low_ownership_query(const loom_op_t* op, bool is_result,
                                        uint16_t index, const void* user_data) {
  const loom_low_descriptor_set_t* set = user_data;
  const loom_low_descriptor_packet_kind_t kind =
      loom_low_descriptor_packet_kind(op);
  if (kind == LOOM_LOW_DESCRIPTOR_PACKET_NONE) {
    return 0;
  }
  const uint32_t ordinal = loom_low_descriptor_packet_ordinal(op, kind);
  const uint16_t offset = set->ownership_offsets[ordinal];
  if (!offset) {
    return 0;
  }
  const loom_low_descriptor_t* descriptor = &set->descriptors[ordinal];
  const loom_low_descriptor_layout_t* layout =
      loom_low_descriptor_layout(set, descriptor);
  if (!is_result && index >= layout->minimum_packet_operand_count) {
    return 0;
  }
  const uint8_t effect =
      set->ownership_effects[offset - 1 +
                             (is_result ? index
                                        : descriptor->result_count + index)];
  static const uint8_t effects[] = {
      [LOOM_LOW_OWNERSHIP_NONE] = 0,
      [LOOM_LOW_OWNERSHIP_BORROW] = LOOM_OPERAND_OWNERSHIP_BORROW,
      [LOOM_LOW_OWNERSHIP_RETAIN] = LOOM_OPERAND_OWNERSHIP_RETAIN,
      [LOOM_LOW_OWNERSHIP_OWNED] = LOOM_RESULT_OWNERSHIP_FRESH,
      [LOOM_LOW_OWNERSHIP_BORROWED] = LOOM_RESULT_OWNERSHIP_BORROWED,
  };
  return effects[effect];
}

loom_ownership_effect_provider_t loom_low_ownership_effect_provider(
    const loom_low_descriptor_set_t* set) {
  return set->ownership_offsets
             ? (loom_ownership_effect_provider_t){loom_low_ownership_query, set}
             : (loom_ownership_effect_provider_t){0};
}

static loom_ownership_effect_provider_t loom_low_ownership_bind(
    const loom_module_t* module, loom_func_like_t function,
    const void* user_data) {
  const loom_string_id_t contract = loom_func_like_repr_contract(function);
  if (contract == LOOM_STRING_ID_INVALID) {
    return (loom_ownership_effect_provider_t){0};
  }
  const loom_low_descriptor_set_t* set = loom_low_descriptor_registry_lookup(
      user_data, loom_string_table_get(&module->strings, contract));
  // The public Low verifier established this binding before shared analysis.
  IREE_ASSERT(set);
  return loom_low_ownership_effect_provider(set);
}

loom_ownership_effect_resolver_t loom_low_ownership_effect_resolver(
    const loom_low_descriptor_registry_t* registry) {
  return (loom_ownership_effect_resolver_t){
      .bind = loom_low_ownership_bind,
      .user_data = registry,
  };
}
