// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/ownership.h"

#include <string.h>

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"

static bool loom_ownership_result_descriptor_at(
    const loom_op_vtable_t* vtable, uint16_t result_index,
    const loom_result_descriptor_t** out_descriptor, uint8_t* out_field_index) {
  *out_descriptor = NULL;
  *out_field_index = 0;
  if (!vtable || !vtable->result_descriptors) return false;
  if (result_index < vtable->fixed_result_count) {
    *out_descriptor = &vtable->result_descriptors[result_index];
    *out_field_index = (uint8_t)result_index;
    return true;
  }
  if (iree_any_bit_set(vtable->vtable_flags, LOOM_OP_VTABLE_VARIADIC_RESULTS)) {
    *out_descriptor = &vtable->result_descriptors[vtable->fixed_result_count];
    *out_field_index = vtable->fixed_result_count;
    return true;
  }
  return false;
}

static bool loom_ownership_find_tied_result(const loom_op_t* op,
                                            uint16_t result_index,
                                            uint16_t* out_operand_index) {
  loom_tied_result_t* tied_results = loom_op_tied_results(op);
  for (uint16_t i = 0; i < op->tied_result_count; ++i) {
    if (tied_results[i].result_index == result_index) {
      *out_operand_index = tied_results[i].operand_index;
      return true;
    }
  }
  return false;
}

bool loom_ownership_type_matches(const loom_ownership_resource_family_t* family,
                                 loom_type_t type) {
  return family && family->type_matches &&
         family->type_matches(type, family->user_data);
}

bool loom_ownership_value_matches(
    const loom_module_t* module, const loom_ownership_resource_family_t* family,
    loom_value_id_t value_id) {
  if (!module || value_id == LOOM_VALUE_ID_INVALID ||
      value_id >= module->values.count) {
    return false;
  }
  return loom_ownership_type_matches(family,
                                     module->values.entries[value_id].type);
}

bool loom_ownership_operand_effect_at(
    const loom_module_t* module, const loom_op_t* op, uint16_t operand_index,
    loom_ownership_operand_effect_t* out_effect) {
  if (out_effect) memset(out_effect, 0, sizeof(*out_effect));
  if (!module || !op || !out_effect || operand_index >= op->operand_count) {
    return false;
  }
  const loom_operand_descriptor_t* descriptor = NULL;
  uint8_t field_index = 0;
  if (!loom_op_operand_descriptor_at(loom_op_vtable(module, op), op,
                                     operand_index, &descriptor, &field_index,
                                     NULL)) {
    return false;
  }
  if (descriptor->ownership_effect == LOOM_OPERAND_OWNERSHIP_NONE) {
    return false;
  }
  *out_effect = (loom_ownership_operand_effect_t){
      .operand_index = operand_index,
      .field_index = field_index,
      .value_id = loom_op_const_operands(op)[operand_index],
      .effect = descriptor->ownership_effect,
      .carrier = descriptor->ownership_carrier,
      .descriptor = descriptor,
  };
  return true;
}

bool loom_ownership_result_effect_at(
    const loom_module_t* module, const loom_op_t* op, uint16_t result_index,
    loom_ownership_result_effect_t* out_effect) {
  if (out_effect) memset(out_effect, 0, sizeof(*out_effect));
  if (!module || !op || !out_effect || result_index >= op->result_count) {
    return false;
  }
  const loom_result_descriptor_t* descriptor = NULL;
  uint8_t field_index = 0;
  if (!loom_ownership_result_descriptor_at(loom_op_vtable(module, op),
                                           result_index, &descriptor,
                                           &field_index)) {
    return false;
  }
  loom_result_ownership_effect_t effect = descriptor->ownership_effect;
  uint16_t source_operand_index = LOOM_OWNERSHIP_SOURCE_OPERAND_NONE;
  if (effect != LOOM_RESULT_OWNERSHIP_NONE &&
      descriptor->ownership_source_operand_index !=
          LOOM_RESULT_OWNERSHIP_SOURCE_FIELD_NONE) {
    source_operand_index = descriptor->ownership_source_operand_index;
  }
  if (effect == LOOM_RESULT_OWNERSHIP_NONE &&
      iree_any_bit_set(descriptor->flags, LOOM_RESULT_ALLOCATES)) {
    effect = LOOM_RESULT_OWNERSHIP_FRESH;
  }
  if (effect == LOOM_RESULT_OWNERSHIP_NONE &&
      loom_ownership_find_tied_result(op, result_index,
                                      &source_operand_index)) {
    effect = LOOM_RESULT_OWNERSHIP_TIED;
  }
  if (effect == LOOM_RESULT_OWNERSHIP_NONE) {
    return false;
  }
  *out_effect = (loom_ownership_result_effect_t){
      .result_index = result_index,
      .field_index = field_index,
      .source_operand_index = source_operand_index,
      .value_id = loom_op_const_results(op)[result_index],
      .effect = effect,
      .descriptor = descriptor,
  };
  return true;
}
