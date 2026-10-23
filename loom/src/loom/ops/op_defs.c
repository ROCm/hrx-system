// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/op_defs.h"

#include <stdint.h>
#include <string.h>

#include "loom/ir/context.h"
#include "loom/ir/module.h"

//===----------------------------------------------------------------------===//
// Vtable helpers
//===----------------------------------------------------------------------===//

const loom_op_vtable_t* const* loom_dialect_vtable_array(
    const loom_op_vtable_t* const* vtables, iree_host_size_t vtable_count,
    iree_host_size_t* out_count) {
  if (out_count != NULL) {
    *out_count = vtable_count;
  }
  return vtables;
}

const loom_op_semantics_t* loom_dialect_semantics_array(
    const loom_op_semantics_t* semantics, iree_host_size_t semantic_count,
    iree_host_size_t* out_count) {
  if (out_count != NULL) {
    *out_count = semantic_count;
  }
  return semantics;
}

loom_op_semantics_t loom_dialect_semantics_lookup(
    loom_op_kind_t kind, loom_dialect_id_t dialect_id,
    const loom_op_semantics_t* semantics, iree_host_size_t semantic_count) {
  if (loom_op_dialect_id(kind) != dialect_id) {
    return loom_op_semantics_empty();
  }
  uint8_t op_index = loom_op_dialect_index(kind);
  if (op_index >= semantic_count) {
    return loom_op_semantics_empty();
  }
  return semantics[op_index];
}

const loom_region_descriptor_t* loom_op_vtable_region_descriptor(
    const loom_op_vtable_t* vtable, uint8_t region_index) {
  if (!vtable || !vtable->region_descriptors || vtable->region_count == 0) {
    return NULL;
  }
  if (region_index < vtable->region_count) {
    return &vtable->region_descriptors[region_index];
  }
  if (iree_any_bit_set(vtable->vtable_flags, LOOM_OP_VTABLE_VARIADIC_REGIONS)) {
    return &vtable->region_descriptors[vtable->region_count - 1];
  }
  return NULL;
}

loom_symbol_id_t loom_op_defining_symbol_id(const loom_module_t* module,
                                            const loom_op_t* op,
                                            const loom_op_vtable_t* vtable) {
  if (!vtable || !vtable->symbol_def) {
    return LOOM_SYMBOL_ID_INVALID;
  }
  const uint8_t symbol_attr_index = vtable->symbol_def->name_attr_index;
  const loom_symbol_ref_t ref =
      loom_attr_as_symbol(loom_op_const_attrs(op)[symbol_attr_index]);
  if (!loom_symbol_ref_is_valid(ref) || ref.module_id != 0 ||
      ref.symbol_id >= module->symbols.count) {
    return LOOM_SYMBOL_ID_INVALID;
  }
  return ref.symbol_id;
}

loom_value_slice_t loom_op_operand_field_span(const loom_op_vtable_t* vtable,
                                              const loom_op_t* op,
                                              uint8_t field_index) {
  if (!op) {
    return (loom_value_slice_t){0};
  }
  if (loom_op_vtable_has_segmented_operands(vtable)) {
    uint8_t segment_count = loom_op_vtable_operand_segment_count(vtable);
    if (field_index >= segment_count) {
      return (loom_value_slice_t){0};
    }
    const uint16_t* counts = loom_op_const_operand_segment_counts(op);
    uint32_t start = 0;
    for (uint8_t i = 0; i < field_index; ++i) {
      start += counts[i];
    }
    uint32_t count = counts[field_index];
    if (start > op->operand_count || count > op->operand_count - start) {
      return (loom_value_slice_t){0};
    }
    return (loom_value_slice_t){
        .values = loom_op_operands(op) + start,
        .count = (uint16_t)count,
    };
  }

  if (!vtable) {
    return (loom_value_slice_t){0};
  }
  if (iree_any_bit_set(vtable->vtable_flags,
                       LOOM_OP_VTABLE_VARIADIC_OPERANDS) &&
      field_index == vtable->fixed_operand_count) {
    if (field_index > op->operand_count) {
      return (loom_value_slice_t){0};
    }
    return (loom_value_slice_t){
        .values = loom_op_operands(op) + field_index,
        .count = (uint16_t)(op->operand_count - field_index),
    };
  }
  if (field_index >= op->operand_count) {
    return (loom_value_slice_t){0};
  }
  return (loom_value_slice_t){
      .values = loom_op_operands(op) + field_index,
      .count = 1,
  };
}

bool loom_op_operand_field_present(const loom_op_vtable_t* vtable,
                                   const loom_op_t* op, uint8_t field_index) {
  return loom_op_operand_field_span(vtable, op, field_index).count > 0;
}

loom_value_slice_t loom_op_result_field_span(const loom_op_vtable_t* vtable,
                                             const loom_op_t* op,
                                             uint8_t field_index) {
  if (!vtable || !op) {
    return (loom_value_slice_t){0};
  }
  if (field_index < vtable->fixed_result_count) {
    if (field_index >= op->result_count) {
      return (loom_value_slice_t){0};
    }
    return (loom_value_slice_t){
        .values = loom_op_results(op) + field_index,
        .count = 1,
    };
  }
  if (iree_any_bit_set(vtable->vtable_flags, LOOM_OP_VTABLE_VARIADIC_RESULTS) &&
      field_index == vtable->fixed_result_count) {
    if (field_index > op->result_count) {
      return (loom_value_slice_t){0};
    }
    return (loom_value_slice_t){
        .values = loom_op_results(op) + field_index,
        .count = (uint16_t)(op->result_count - field_index),
    };
  }
  return (loom_value_slice_t){0};
}

bool loom_op_operand_descriptor_at(
    const loom_op_vtable_t* vtable, const loom_op_t* op, uint16_t operand_index,
    const loom_operand_descriptor_t** out_descriptor, uint8_t* out_field_index,
    uint16_t* out_element_index) {
  *out_descriptor = NULL;
  if (out_field_index) {
    *out_field_index = 0;
  }
  if (out_element_index) {
    *out_element_index = 0;
  }
  if (!vtable || !op || !vtable->operand_descriptors ||
      operand_index >= op->operand_count) {
    return false;
  }
  if (loom_op_vtable_has_segmented_operands(vtable)) {
    uint8_t segment_count = loom_op_vtable_operand_segment_count(vtable);
    const uint16_t* counts = loom_op_const_operand_segment_counts(op);
    uint32_t start = 0;
    for (uint8_t i = 0; i < segment_count; ++i) {
      uint32_t count = counts[i];
      if (operand_index >= start && operand_index < start + count) {
        *out_descriptor = &vtable->operand_descriptors[i];
        if (out_field_index) {
          *out_field_index = i;
        }
        if (out_element_index) {
          *out_element_index = (uint16_t)(operand_index - start);
        }
        return true;
      }
      start += count;
    }
    return false;
  }
  if (operand_index < vtable->fixed_operand_count) {
    *out_descriptor = &vtable->operand_descriptors[operand_index];
    if (out_field_index) {
      *out_field_index = (uint8_t)operand_index;
    }
    if (out_element_index) {
      *out_element_index = 0;
    }
    return true;
  }
  if (iree_any_bit_set(vtable->vtable_flags,
                       LOOM_OP_VTABLE_VARIADIC_OPERANDS)) {
    *out_descriptor = &vtable->operand_descriptors[vtable->fixed_operand_count];
    if (out_field_index) {
      *out_field_index = vtable->fixed_operand_count;
    }
    if (out_element_index) {
      *out_element_index =
          (uint16_t)(operand_index - vtable->fixed_operand_count);
    }
    return true;
  }
  return false;
}

loom_operand_role_t loom_op_operand_role_at(const loom_op_vtable_t* vtable,
                                            const loom_op_t* op,
                                            uint16_t operand_index) {
  if (!vtable || vtable->operand_role_mask == 0) {
    return LOOM_OPERAND_ROLE_NONE;
  }
  const loom_operand_descriptor_t* descriptor = NULL;
  if (!loom_op_operand_descriptor_at(vtable, op, operand_index, &descriptor,
                                     NULL, NULL)) {
    return LOOM_OPERAND_ROLE_NONE;
  }
  return descriptor->role;
}

loom_operand_role_t loom_op_operand_role(const loom_module_t* module,
                                         const loom_op_t* op,
                                         uint16_t operand_index) {
  if (module == NULL || op == NULL) {
    return LOOM_OPERAND_ROLE_NONE;
  }
  return loom_op_operand_role_at(loom_op_vtable(module, op), op, operand_index);
}

bool loom_op_operand_has_role(const loom_module_t* module, const loom_op_t* op,
                              uint16_t operand_index,
                              loom_operand_role_t role) {
  if (role == LOOM_OPERAND_ROLE_NONE) {
    return false;
  }
  return loom_op_operand_role(module, op, operand_index) == role;
}

bool loom_op_first_operand_with_role(const loom_module_t* module,
                                     const loom_op_t* op,
                                     loom_operand_role_t role,
                                     loom_value_id_t* out_value_id) {
  *out_value_id = LOOM_VALUE_ID_INVALID;
  if (module == NULL || op == NULL) {
    return false;
  }
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  if (vtable == NULL || !iree_any_bit_set(vtable->operand_role_mask,
                                          loom_operand_role_mask_bit(role))) {
    return false;
  }
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    if (loom_op_operand_role_at(vtable, op, i) == role) {
      *out_value_id = operands[i];
      return true;
    }
  }
  return false;
}

bool loom_op_defines_value(const loom_op_t* op, loom_value_id_t value_id) {
  if (op == NULL || value_id == LOOM_VALUE_ID_INVALID) {
    return false;
  }
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] == value_id) {
      return true;
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Effect query helpers
//===----------------------------------------------------------------------===//

loom_trait_flags_t loom_op_effective_traits(const loom_module_t* module,
                                            const loom_op_t* op) {
  (void)module;
  return op->traits;
}

void loom_op_refresh_effective_traits(const loom_module_t* module,
                                      loom_op_t* op) {
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  if (vtable && vtable->call_like &&
      vtable->call_like->kind == LOOM_CALL_LIKE_KIND_SEMANTIC) {
    op->flags |= LOOM_OP_FLAG_SEMANTIC_CALL;
  } else {
    op->flags &= ~LOOM_OP_FLAG_SEMANTIC_CALL;
  }
  if (!vtable) {
    op->traits = LOOM_TRAIT_UNKNOWN_EFFECTS;
    return;
  }
  if (vtable->effective_traits) {
    op->traits = vtable->effective_traits(module, op);
  }
}

bool loom_op_may_write(const loom_module_t* module, const loom_op_t* op) {
  return loom_traits_may_write(loom_op_effective_traits(module, op));
}

//===----------------------------------------------------------------------===//
// CallLike interface
//===----------------------------------------------------------------------===//

bool loom_op_requires_source_context(const loom_module_t* module,
                                     const loom_op_t* op) {
  if (iree_any_bit_set(op->traits, LOOM_TRAIT_CONTEXTUAL)) {
    return true;
  }
  const loom_call_like_vtable_t* call = loom_op_vtable(module, op)->call_like;
  if (!call || call->kind != LOOM_CALL_LIKE_KIND_SEMANTIC) {
    return false;
  }
  const loom_attribute_t* attrs = loom_op_const_attrs(op);
  if (call->inline_policy_attr_index != LOOM_ATTR_INDEX_NONE &&
      loom_attr_as_enum(attrs[call->inline_policy_attr_index]) ==
          LOOM_INLINE_POLICY_INLINE) {
    return true;
  }
  const loom_symbol_ref_t callee =
      loom_attr_as_symbol(attrs[call->callee_attr_index]);
  if (callee.module_id != 0) {
    return false;
  }
  const loom_func_like_t function = loom_func_like_const_cast(
      module, module->symbols.entries[callee.symbol_id].defining_op);
  return loom_func_like_inline_policy(function) == LOOM_INLINE_POLICY_INLINE;
}

loom_call_like_t loom_call_like_cast(const loom_module_t* module,
                                     loom_op_t* op) {
  if (!op) {
    return (loom_call_like_t){.op = NULL, .vtable = NULL};
  }
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  if (!vtable || !vtable->call_like) {
    return (loom_call_like_t){.op = NULL, .vtable = NULL};
  }
  return (loom_call_like_t){.op = op, .vtable = vtable->call_like};
}

bool loom_call_like_is_direct_semantic(loom_call_like_t call) {
  return loom_call_like_isa(call) &&
         loom_call_like_kind(call) == LOOM_CALL_LIKE_KIND_SEMANTIC &&
         loom_call_like_operand_offset(call) == 0 &&
         loom_call_like_result_offset(call) == 0 &&
         call.op->region_count == 0 && call.op->successor_count == 0;
}

loom_symbol_ref_t loom_call_like_callee(loom_call_like_t call) {
  if (!call.vtable) {
    return loom_symbol_ref_null();
  }
  return loom_attr_as_symbol(
      loom_op_attrs(call.op)[call.vtable->callee_attr_index]);
}

void loom_call_like_set_callee(loom_module_t* module, loom_call_like_t call,
                               loom_symbol_ref_t callee) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(call.op);
  IREE_ASSERT_ARGUMENT(call.vtable);
  IREE_ASSERT_LT(call.vtable->callee_attr_index, call.op->attribute_count);
  IREE_ASSERT(loom_symbol_ref_is_valid(callee));
  IREE_ASSERT_EQ(callee.module_id, 0);
  IREE_ASSERT_LT(callee.symbol_id, module->symbols.count);
  const loom_trait_flags_t old_traits = call.op->traits;
  loom_op_attrs(call.op)[call.vtable->callee_attr_index] =
      loom_attr_symbol(callee);
  loom_op_refresh_effective_traits(module, call.op);
  loom_module_update_op_direct_summaries(module, call.op, old_traits,
                                         call.op->traits);
}

loom_value_slice_t loom_call_like_operands(loom_call_like_t call) {
  if (!call.vtable) {
    return (loom_value_slice_t){0};
  }
  uint16_t offset = loom_call_like_operand_offset(call);
  if (offset > call.op->operand_count) {
    return (loom_value_slice_t){0};
  }
  return (loom_value_slice_t){
      .values = loom_op_operands(call.op) + offset,
      .count = (uint16_t)(call.op->operand_count - offset),
  };
}

loom_value_slice_t loom_call_like_results(loom_call_like_t call) {
  if (!call.vtable || call.vtable->result_offset > call.op->result_count) {
    return (loom_value_slice_t){0};
  }
  uint16_t offset = call.vtable->result_offset;
  return (loom_value_slice_t){
      .values = loom_op_results(call.op) + offset,
      .count = (uint16_t)(call.op->result_count - offset),
  };
}

uint16_t loom_call_like_operand_offset(loom_call_like_t call) {
  if (!call.vtable) {
    return 0;
  }
  uint8_t field_index = call.vtable->operand_field_index;
  if (call.vtable->operand_segment_count == 0) {
    return field_index;
  }
  if (field_index >= call.vtable->operand_segment_count) {
    return UINT16_MAX;
  }
  const uint16_t* segment_counts =
      loom_op_const_operand_segment_counts(call.op);
  uint16_t offset = 0;
  for (uint8_t i = 0; i < field_index; ++i) {
    offset += segment_counts[i];
  }
  return offset;
}

uint16_t loom_call_like_result_offset(loom_call_like_t call) {
  if (!call.vtable) {
    return 0;
  }
  return call.vtable->result_offset;
}

loom_callable_effect_contract_t loom_call_like_effect_contract(
    loom_call_like_t call) {
  if (!call.vtable) {
    return LOOM_CALLABLE_EFFECT_CONTRACT_UNSPECIFIED;
  }
  if (call.vtable->effect_contract_attr_index == LOOM_ATTR_INDEX_NONE) {
    return LOOM_CALLABLE_EFFECT_CONTRACT_UNSPECIFIED;
  }
  return (loom_callable_effect_contract_t)loom_attr_as_enum(
      loom_op_attrs(call.op)[call.vtable->effect_contract_attr_index]);
}

void loom_call_like_set_effect_contract(
    loom_module_t* module, loom_call_like_t call,
    loom_callable_effect_contract_t effect_contract) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(call.op);
  IREE_ASSERT_ARGUMENT(call.vtable);
  IREE_ASSERT_NE(call.vtable->effect_contract_attr_index, LOOM_ATTR_INDEX_NONE);
  IREE_ASSERT_LT(call.vtable->effect_contract_attr_index,
                 call.op->attribute_count);
  IREE_ASSERT_LT(effect_contract, LOOM_CALLABLE_EFFECT_CONTRACT_COUNT_);
  const loom_trait_flags_t old_traits = call.op->traits;
  loom_op_attrs(call.op)[call.vtable->effect_contract_attr_index] =
      effect_contract == LOOM_CALLABLE_EFFECT_CONTRACT_UNSPECIFIED
          ? loom_attr_absent()
          : loom_attr_enum(effect_contract);
  loom_op_refresh_effective_traits(module, call.op);
  loom_module_update_op_direct_summaries(module, call.op, old_traits,
                                         call.op->traits);
}

uint8_t loom_call_like_temperature(loom_call_like_t call) {
  if (!call.vtable) {
    return 0;
  }
  if (call.vtable->temperature_attr_index == LOOM_ATTR_INDEX_NONE) {
    return 0;
  }
  return loom_attr_as_enum(
      loom_op_attrs(call.op)[call.vtable->temperature_attr_index]);
}

loom_inline_policy_t loom_call_like_inline_policy(loom_call_like_t call) {
  if (!call.vtable) {
    return LOOM_INLINE_POLICY_UNSPECIFIED;
  }
  if (call.vtable->inline_policy_attr_index == LOOM_ATTR_INDEX_NONE) {
    return LOOM_INLINE_POLICY_UNSPECIFIED;
  }
  return (loom_inline_policy_t)loom_attr_as_enum(
      loom_op_attrs(call.op)[call.vtable->inline_policy_attr_index]);
}

loom_call_like_kind_t loom_call_like_kind(loom_call_like_t call) {
  if (!call.vtable) {
    return LOOM_CALL_LIKE_KIND_NONE;
  }
  return call.vtable->kind;
}

//===----------------------------------------------------------------------===//
// FuncLike interface
//===----------------------------------------------------------------------===//

loom_func_like_t loom_func_like_cast(const loom_module_t* module,
                                     loom_op_t* op) {
  if (!op) {
    return (loom_func_like_t){.op = NULL, .vtable = NULL};
  }
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  if (!vtable || !vtable->func_like) {
    return (loom_func_like_t){.op = NULL, .vtable = NULL};
  }
  return (loom_func_like_t){.op = op, .vtable = vtable->func_like};
}

loom_region_t* loom_func_like_body(loom_func_like_t func) {
  return loom_func_like_region(func, loom_func_like_body_region_index(func));
}

const loom_region_descriptor_t* loom_func_like_body_region_descriptor(
    const loom_module_t* module, loom_func_like_t func) {
  const uint8_t body_region_index = loom_func_like_body_region_index(func);
  if (!loom_func_like_isa(func) || body_region_index >= func.op->region_count) {
    return NULL;
  }
  return loom_op_vtable_region_descriptor(loom_op_vtable(module, func.op),
                                          body_region_index);
}

bool loom_func_like_op_is_body_exit(const loom_module_t* module,
                                    loom_func_like_t func,
                                    const loom_op_t* op) {
  if (op == NULL || op->parent_block == NULL ||
      op->parent_block->parent_region != loom_func_like_body(func)) {
    return false;
  }
  const loom_region_descriptor_t* descriptor =
      loom_func_like_body_region_descriptor(module, func);
  return descriptor != NULL && descriptor->terminator != LOOM_OP_KIND_UNKNOWN &&
         op->kind == descriptor->terminator;
}

uint8_t loom_func_like_body_region_index(loom_func_like_t func) {
  if (!func.vtable) {
    return LOOM_REGION_INDEX_NONE;
  }
  return func.vtable->body_region_index;
}

uint8_t loom_func_like_region_count(loom_func_like_t func) {
  if (!func.vtable || !func.op) {
    return 0;
  }
  return func.op->region_count;
}

loom_region_t* loom_func_like_region(loom_func_like_t func,
                                     uint8_t region_index) {
  if (!func.vtable || !func.op || region_index >= func.op->region_count) {
    return NULL;
  }
  return loom_op_regions(func.op)[region_index];
}

bool loom_func_like_region_is_body(loom_func_like_t func,
                                   uint8_t region_index) {
  return region_index == loom_func_like_body_region_index(func);
}

bool loom_func_like_region_projects_args(const loom_module_t* module,
                                         loom_func_like_t func,
                                         uint8_t region_index) {
  if (!loom_func_like_isa(func) || region_index >= func.op->region_count) {
    return false;
  }
  const loom_op_vtable_t* vtable = loom_op_vtable(module, func.op);
  const loom_region_descriptor_t* descriptor =
      loom_op_vtable_region_descriptor(vtable, region_index);
  return descriptor &&
         iree_any_bit_set(descriptor->flags, LOOM_REGION_PROJECT_FUNC_ARGS);
}

loom_callable_effect_contract_t loom_func_like_effect_contract(
    loom_func_like_t func) {
  if (!func.vtable) {
    return LOOM_CALLABLE_EFFECT_CONTRACT_UNSPECIFIED;
  }
  if (func.vtable->effect_contract_attr_index == LOOM_ATTR_INDEX_NONE) {
    return LOOM_CALLABLE_EFFECT_CONTRACT_UNSPECIFIED;
  }
  return (loom_callable_effect_contract_t)loom_attr_as_enum(
      loom_op_attrs(func.op)[func.vtable->effect_contract_attr_index]);
}

void loom_func_like_set_effect_contract(
    loom_module_t* module, loom_func_like_t func,
    loom_callable_effect_contract_t effect_contract) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(func.op);
  IREE_ASSERT_ARGUMENT(func.vtable);
  IREE_ASSERT_NE(func.vtable->effect_contract_attr_index, LOOM_ATTR_INDEX_NONE);
  IREE_ASSERT_LT(func.vtable->effect_contract_attr_index,
                 func.op->attribute_count);
  IREE_ASSERT_LT(effect_contract, LOOM_CALLABLE_EFFECT_CONTRACT_COUNT_);
  const loom_trait_flags_t old_traits = func.op->traits;
  loom_op_attrs(func.op)[func.vtable->effect_contract_attr_index] =
      effect_contract == LOOM_CALLABLE_EFFECT_CONTRACT_UNSPECIFIED
          ? loom_attr_absent()
          : loom_attr_enum(effect_contract);
  loom_op_refresh_effective_traits(module, func.op);
  loom_module_update_op_direct_summaries(module, func.op, old_traits,
                                         func.op->traits);
}

uint8_t loom_func_like_temperature(loom_func_like_t func) {
  if (!func.vtable) {
    return 0;
  }
  if (func.vtable->temperature_attr_index == LOOM_ATTR_INDEX_NONE) {
    return 0;
  }
  return loom_attr_as_enum(
      loom_op_attrs(func.op)[func.vtable->temperature_attr_index]);
}

loom_inline_policy_t loom_func_like_inline_policy(loom_func_like_t func) {
  if (!func.vtable) {
    return LOOM_INLINE_POLICY_UNSPECIFIED;
  }
  if (func.vtable->inline_policy_attr_index == LOOM_ATTR_INDEX_NONE) {
    return LOOM_INLINE_POLICY_UNSPECIFIED;
  }
  return (loom_inline_policy_t)loom_attr_as_enum(
      loom_op_attrs(func.op)[func.vtable->inline_policy_attr_index]);
}

uint8_t loom_func_like_visibility(loom_func_like_t func) {
  if (!func.vtable) {
    return 0;
  }
  if (func.vtable->visibility_attr_index == LOOM_ATTR_INDEX_NONE) {
    return 0;
  }
  return loom_attr_as_enum(
      loom_op_attrs(func.op)[func.vtable->visibility_attr_index]);
}

uint8_t loom_func_like_cc(loom_func_like_t func) {
  if (!func.vtable) {
    return 0;
  }
  if (func.vtable->cc_attr_index == LOOM_ATTR_INDEX_NONE) {
    return 0;
  }
  return loom_attr_as_enum(loom_op_attrs(func.op)[func.vtable->cc_attr_index]);
}

loom_symbol_ref_t loom_func_like_callee(loom_func_like_t func) {
  if (!func.vtable) {
    return (loom_symbol_ref_t){0};
  }
  return loom_attr_as_symbol(
      loom_op_attrs(func.op)[func.vtable->callee_attr_index]);
}

loom_string_id_t loom_func_like_import_module(loom_func_like_t func) {
  if (!func.vtable ||
      func.vtable->import_module_attr_index == LOOM_ATTR_INDEX_NONE) {
    return LOOM_STRING_ID_INVALID;
  }
  loom_attribute_t attr =
      loom_op_attrs(func.op)[func.vtable->import_module_attr_index];
  if (loom_attr_is_absent(attr)) {
    return LOOM_STRING_ID_INVALID;
  }
  return loom_attr_as_string_id(attr);
}

loom_string_id_t loom_func_like_import_symbol(loom_func_like_t func) {
  if (!func.vtable ||
      func.vtable->import_symbol_attr_index == LOOM_ATTR_INDEX_NONE) {
    return LOOM_STRING_ID_INVALID;
  }
  loom_attribute_t attr =
      loom_op_attrs(func.op)[func.vtable->import_symbol_attr_index];
  if (loom_attr_is_absent(attr)) {
    return LOOM_STRING_ID_INVALID;
  }
  return loom_attr_as_string_id(attr);
}

loom_symbol_ref_t loom_func_like_target(loom_func_like_t func) {
  if (!func.vtable || func.vtable->target_attr_index == LOOM_ATTR_INDEX_NONE) {
    return loom_symbol_ref_null();
  }
  return loom_attr_as_symbol(
      loom_op_attrs(func.op)[func.vtable->target_attr_index]);
}

void loom_func_like_set_target(loom_module_t* module, loom_func_like_t func,
                               loom_symbol_ref_t target) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(func.op);
  IREE_ASSERT_ARGUMENT(func.vtable);
  IREE_ASSERT_NE(func.vtable->target_attr_index, LOOM_ATTR_INDEX_NONE);
  IREE_ASSERT_LT(func.vtable->target_attr_index, func.op->attribute_count);
  IREE_ASSERT(loom_symbol_ref_is_valid(target));
  IREE_ASSERT_EQ(target.module_id, 0);
  IREE_ASSERT_LT(target.symbol_id, module->symbols.count);
  const loom_trait_flags_t old_traits = func.op->traits;
  loom_op_attrs(func.op)[func.vtable->target_attr_index] =
      loom_attr_symbol(target);
  loom_op_refresh_effective_traits(module, func.op);
  loom_module_update_op_direct_summaries(module, func.op, old_traits,
                                         func.op->traits);
}

void loom_func_like_set_retained(loom_module_t* module, loom_func_like_t func,
                                 bool retained) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(func.op);
  IREE_ASSERT_ARGUMENT(func.vtable);
  const loom_op_vtable_t* op_vtable = loom_op_vtable(module, func.op);
  IREE_ASSERT_ARGUMENT(op_vtable);
  IREE_ASSERT_ARGUMENT(op_vtable->symbol_def);
  IREE_ASSERT(op_vtable->symbol_def->retain_attr_index_plus_one);
  const uint8_t retain_attr_index =
      op_vtable->symbol_def->retain_attr_index_plus_one - 1;
  IREE_ASSERT_LT(retain_attr_index, func.op->attribute_count);
  loom_op_attrs(func.op)[retain_attr_index] =
      retained ? loom_attr_enum(1) : loom_attr_absent();
  loom_module_link_symbol_defining_op(module, func.op, op_vtable);
}

loom_string_id_t loom_func_like_repr_contract(loom_func_like_t func) {
  if (!func.vtable ||
      func.vtable->repr_contract_attr_index == LOOM_ATTR_INDEX_NONE) {
    return LOOM_STRING_ID_INVALID;
  }
  loom_attribute_t attr =
      loom_op_attrs(func.op)[func.vtable->repr_contract_attr_index];
  if (loom_attr_is_absent(attr)) {
    return LOOM_STRING_ID_INVALID;
  }
  return loom_attr_as_string_id(attr);
}

uint8_t loom_func_like_abi(loom_func_like_t func) {
  if (!func.vtable || func.vtable->abi_attr_index == LOOM_ATTR_INDEX_NONE) {
    return 0;
  }
  loom_attribute_t attr = loom_op_attrs(func.op)[func.vtable->abi_attr_index];
  if (loom_attr_is_absent(attr)) {
    return 0;
  }
  return loom_attr_as_enum(attr);
}

loom_attribute_t loom_func_like_source_signature(loom_func_like_t func) {
  if (!func.vtable ||
      func.vtable->source_signature_attr_index == LOOM_ATTR_INDEX_NONE) {
    return loom_attr_absent();
  }
  return loom_op_attrs(func.op)[func.vtable->source_signature_attr_index];
}

loom_named_attr_slice_t loom_func_like_abi_attrs(loom_func_like_t func) {
  if (!func.vtable ||
      func.vtable->abi_attrs_attr_index == LOOM_ATTR_INDEX_NONE) {
    return loom_named_attr_slice_empty();
  }
  return loom_attr_as_dict(
      loom_op_attrs(func.op)[func.vtable->abi_attrs_attr_index]);
}

loom_string_id_t loom_func_like_export_symbol(loom_func_like_t func) {
  if (!func.vtable ||
      func.vtable->export_symbol_attr_index == LOOM_ATTR_INDEX_NONE) {
    return LOOM_STRING_ID_INVALID;
  }
  loom_attribute_t attr =
      loom_op_attrs(func.op)[func.vtable->export_symbol_attr_index];
  if (loom_attr_is_absent(attr)) {
    return LOOM_STRING_ID_INVALID;
  }
  return loom_attr_as_string_id(attr);
}

loom_named_attr_slice_t loom_func_like_export_attrs(loom_func_like_t func) {
  if (!func.vtable ||
      func.vtable->export_attrs_attr_index == LOOM_ATTR_INDEX_NONE) {
    return loom_named_attr_slice_empty();
  }
  return loom_attr_as_dict(
      loom_op_attrs(func.op)[func.vtable->export_attrs_attr_index]);
}

bool loom_func_like_is_kernel_entry(loom_func_like_t func) {
  return func.vtable &&
         iree_any_bit_set(func.vtable->flags, LOOM_FUNC_LIKE_FLAG_KERNEL_ENTRY);
}

bool loom_func_like_is_kernel(loom_func_like_t func) {
  return func.vtable && iree_any_bit_set(func.vtable->flags,
                                         LOOM_FUNC_LIKE_FLAG_KERNEL |
                                             LOOM_FUNC_LIKE_FLAG_KERNEL_ENTRY);
}

bool loom_func_like_is_exported(loom_func_like_t func) {
  return (loom_func_like_is_kernel(func) ||
          loom_func_like_visibility(func) != 0) &&
         loom_func_like_import_module(func) == LOOM_STRING_ID_INVALID &&
         loom_func_like_import_symbol(func) == LOOM_STRING_ID_INVALID;
}

bool loom_func_like_is_module_internal(loom_func_like_t func) {
  if (!loom_func_like_isa(func) || loom_func_like_is_kernel(func) ||
      loom_func_like_visibility(func) != 0 ||
      loom_func_like_import_module(func) != LOOM_STRING_ID_INVALID ||
      loom_func_like_import_symbol(func) != LOOM_STRING_ID_INVALID) {
    return false;
  }
  return true;
}

bool loom_func_like_export_linkage(loom_func_like_t func,
                                   uint8_t* out_linkage) {
  *out_linkage = 0;
  if (!func.vtable ||
      func.vtable->export_linkage_attr_index == LOOM_ATTR_INDEX_NONE) {
    return false;
  }
  loom_attribute_t attr =
      loom_op_attrs(func.op)[func.vtable->export_linkage_attr_index];
  if (loom_attr_is_absent(attr)) {
    return false;
  }
  *out_linkage = loom_attr_as_enum(attr);
  return true;
}

const loom_value_id_t* loom_func_like_arg_ids(loom_func_like_t func,
                                              uint16_t* out_count) {
  if (!func.vtable) {
    *out_count = 0;
    return NULL;
  }
  if (func.vtable->args_operand_field_index == LOOM_OPERAND_INDEX_NONE) {
    loom_region_t* body = loom_func_like_body(func);
    if (body && body->block_count > 0) {
      loom_block_t* entry = loom_region_entry_block(body);
      *out_count = entry->arg_count;
      return entry->arg_ids;
    }
    *out_count = 0;
    return NULL;
  }
  if (func.vtable->args_operand_segment_count > 0) {
    const uint16_t* segment_counts =
        loom_op_const_operand_segment_counts(func.op);
    uint16_t operand_offset = 0;
    for (uint8_t i = 0; i < func.vtable->args_operand_field_index; ++i) {
      operand_offset += segment_counts[i];
    }
    *out_count = segment_counts[func.vtable->args_operand_field_index];
    return loom_op_operands(func.op) + operand_offset;
  }
  uint8_t operand_offset = func.vtable->args_operand_field_index;
  *out_count = (uint16_t)(func.op->operand_count - operand_offset);
  return loom_op_operands(func.op) + operand_offset;
}

loom_value_slice_t loom_kernel_workload_arg_ids(const loom_module_t* module,
                                                const loom_op_t* op) {
  if (!op) {
    return (loom_value_slice_t){0};
  }
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  const loom_symbol_definition_descriptor_t* definition =
      vtable ? vtable->symbol_def : NULL;
  if (!loom_symbol_definition_implements(definition,
                                         LOOM_SYMBOL_INTERFACE_KERNEL)) {
    return (loom_value_slice_t){0};
  }
  if (definition->kernel_workload_region_index_plus_one != 0) {
    uint8_t region_index =
        definition->kernel_workload_region_index_plus_one - 1;
    loom_region_t* region = loom_op_regions(op)[region_index];
    if (!region || region->block_count == 0) {
      return (loom_value_slice_t){0};
    }
    loom_block_t* entry = loom_region_entry_block(region);
    return (loom_value_slice_t){
        .values = entry->arg_ids,
        .count = entry->arg_count,
    };
  }
  uint8_t operand_field_index =
      definition->kernel_workload_operand_field_index_plus_one - 1;
  return loom_op_operand_field_span(vtable, op, operand_field_index);
}

const loom_predicate_t* loom_func_like_predicates(loom_func_like_t func,
                                                  uint16_t* out_count) {
  if (!func.vtable) {
    *out_count = 0;
    return NULL;
  }
  if (func.vtable->predicates_attr_index == LOOM_ATTR_INDEX_NONE) {
    *out_count = 0;
    return NULL;
  }
  loom_attribute_t attr =
      loom_op_attrs(func.op)[func.vtable->predicates_attr_index];
  *out_count = attr.count;
  return attr.predicate_list;
}

loom_parameterized_attr_array_t loom_func_like_requires(loom_func_like_t func) {
  if (!func.vtable ||
      func.vtable->requires_attr_index == LOOM_ATTR_INDEX_NONE) {
    return loom_parameterized_attr_array_empty();
  }
  return loom_attr_as_parameterized_array(
      loom_op_attrs(func.op)[func.vtable->requires_attr_index]);
}

int64_t loom_func_like_specialization_count(loom_func_like_t func) {
  if (!func.vtable ||
      func.vtable->specialization_count_attr_index == LOOM_ATTR_INDEX_NONE) {
    return 0;
  }
  return loom_attr_as_i64(
      loom_op_attrs(func.op)[func.vtable->specialization_count_attr_index]);
}

loom_symbol_ref_t loom_func_like_template_family(loom_func_like_t func) {
  if (!func.vtable ||
      func.vtable->template_family_attr_index == LOOM_ATTR_INDEX_NONE) {
    return loom_symbol_ref_null();
  }
  return loom_attr_as_symbol(
      loom_op_attrs(func.op)[func.vtable->template_family_attr_index]);
}

int64_t loom_func_like_priority(loom_func_like_t func) {
  if (!func.vtable) {
    return 0;
  }
  if (func.vtable->priority_attr_index == LOOM_ATTR_INDEX_NONE) {
    return 0;
  }
  return loom_attr_as_i64(
      loom_op_attrs(func.op)[func.vtable->priority_attr_index]);
}

//===----------------------------------------------------------------------===//
// TargetLike interface
//===----------------------------------------------------------------------===//

loom_target_like_t loom_target_like_cast(const loom_module_t* module,
                                         const loom_op_t* op) {
  if (!op) {
    return (loom_target_like_t){.op = NULL, .vtable = NULL};
  }
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  if (!vtable || !vtable->target_like) {
    return (loom_target_like_t){.op = NULL, .vtable = NULL};
  }
  return (loom_target_like_t){.op = op, .vtable = vtable->target_like};
}

static loom_attribute_t loom_target_like_attr(loom_target_like_t target,
                                              uint8_t attr_index) {
  if (!target.vtable || attr_index == LOOM_ATTR_INDEX_NONE ||
      attr_index >= target.op->attribute_count) {
    return loom_attr_absent();
  }
  return loom_op_attrs(target.op)[attr_index];
}

loom_symbol_ref_t loom_target_like_symbol(loom_target_like_t target) {
  loom_attribute_t attr = loom_target_like_attr(
      target,
      target.vtable ? target.vtable->symbol_attr_index : LOOM_ATTR_INDEX_NONE);
  if (loom_attr_is_absent(attr)) {
    return loom_symbol_ref_null();
  }
  return loom_attr_as_symbol(attr);
}

loom_attribute_t loom_target_like_selector(loom_target_like_t target) {
  return loom_target_like_attr(target, target.vtable
                                           ? target.vtable->selector_attr_index
                                           : LOOM_ATTR_INDEX_NONE);
}

loom_named_attr_slice_t loom_target_like_extension_attrs(
    loom_target_like_t target) {
  loom_attribute_t attr = loom_target_like_attr(
      target, target.vtable ? target.vtable->extension_attrs_attr_index
                            : LOOM_ATTR_INDEX_NONE);
  if (loom_attr_is_absent(attr)) {
    return loom_named_attr_slice_empty();
  }
  return loom_attr_as_dict(attr);
}

const loom_target_like_descriptor_t* loom_target_like_descriptor(
    loom_target_like_t target) {
  return target.vtable ? target.vtable->descriptor : NULL;
}

//===----------------------------------------------------------------------===//
// LoopLike interface
//===----------------------------------------------------------------------===//

loom_loop_like_t loom_loop_like_cast(const loom_module_t* module,
                                     loom_op_t* op) {
  if (!op) {
    return (loom_loop_like_t){.op = NULL, .vtable = NULL};
  }
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  if (!vtable || !vtable->loop_like) {
    return (loom_loop_like_t){.op = NULL, .vtable = NULL};
  }
  return (loom_loop_like_t){.op = op, .vtable = vtable->loop_like};
}

loom_region_t* loom_loop_like_body(loom_loop_like_t loop) {
  if (!loop.vtable) {
    return NULL;
  }
  return loom_op_regions(loop.op)[loop.vtable->body_region_index];
}

loom_region_t* loom_loop_like_condition_region(loom_loop_like_t loop) {
  if (!loop.vtable) {
    return NULL;
  }
  if (loop.vtable->condition_region_index == LOOM_REGION_INDEX_NONE) {
    return NULL;
  }
  return loom_op_regions(loop.op)[loop.vtable->condition_region_index];
}

loom_value_id_t loom_loop_like_condition(loom_loop_like_t loop) {
  const loom_region_t* region = loom_loop_like_condition_region(loop);
  if (!region) {
    return LOOM_VALUE_ID_INVALID;
  }
  const loom_op_t* terminator = loom_region_const_entry_block(region)->last_op;
  return loom_op_const_operands(terminator)[0];
}

loom_value_id_t loom_loop_like_iv(loom_loop_like_t loop) {
  if (!loop.vtable) {
    return LOOM_VALUE_ID_INVALID;
  }
  if (loop.vtable->iv_block_arg_index == LOOM_BLOCK_ARG_INDEX_NONE) {
    return LOOM_VALUE_ID_INVALID;
  }
  loom_region_t* body = loom_loop_like_body(loop);
  return loom_region_entry_arg_id(body, loop.vtable->iv_block_arg_index);
}

static loom_value_slice_t loom_loop_like_segmented_operand_field_span(
    loom_loop_like_t loop, uint8_t field_index) {
  if (!loop.op || !loop.vtable || !loop.vtable->segmented_operands ||
      field_index >= loop.vtable->operand_field_count) {
    return (loom_value_slice_t){0};
  }
  const uint16_t* counts = loom_op_const_operand_segment_counts(loop.op);
  uint32_t start = 0;
  for (uint8_t i = 0; i < field_index; ++i) {
    start += counts[i];
  }
  uint32_t count = counts[field_index];
  if (start > loop.op->operand_count ||
      count > loop.op->operand_count - start) {
    return (loom_value_slice_t){0};
  }
  return (loom_value_slice_t){
      .values = loom_op_operands(loop.op) + start,
      .count = (uint16_t)count,
  };
}

loom_value_slice_t loom_loop_like_iter_args(loom_loop_like_t loop) {
  if (!loop.vtable) {
    return (loom_value_slice_t){.values = NULL, .count = 0};
  }
  uint8_t field_index = loop.vtable->iter_args_operand_field_index;
  if (loop.vtable->segmented_operands) {
    return loom_loop_like_segmented_operand_field_span(loop, field_index);
  }
  if (field_index >= loop.op->operand_count) {
    return (loom_value_slice_t){.values = NULL, .count = 0};
  }
  return (loom_value_slice_t){
      .values = loom_op_operands(loop.op) + field_index,
      .count = (uint16_t)(loop.op->operand_count - field_index),
  };
}

loom_value_id_t loom_loop_like_lower_bound(loom_loop_like_t loop) {
  if (!loop.vtable) {
    return LOOM_VALUE_ID_INVALID;
  }
  uint8_t index = loop.vtable->lower_bound_operand_index;
  if (index == LOOM_OPERAND_INDEX_NONE) {
    return LOOM_VALUE_ID_INVALID;
  }
  if (loop.vtable->segmented_operands) {
    loom_value_slice_t slice =
        loom_loop_like_segmented_operand_field_span(loop, index);
    return slice.count == 1 ? slice.values[0] : LOOM_VALUE_ID_INVALID;
  }
  if (index >= loop.op->operand_count) {
    return LOOM_VALUE_ID_INVALID;
  }
  return loom_op_operands(loop.op)[index];
}

loom_value_id_t loom_loop_like_upper_bound(loom_loop_like_t loop) {
  if (!loop.vtable) {
    return LOOM_VALUE_ID_INVALID;
  }
  uint8_t index = loop.vtable->upper_bound_operand_index;
  if (index == LOOM_OPERAND_INDEX_NONE) {
    return LOOM_VALUE_ID_INVALID;
  }
  if (loop.vtable->segmented_operands) {
    loom_value_slice_t slice =
        loom_loop_like_segmented_operand_field_span(loop, index);
    return slice.count == 1 ? slice.values[0] : LOOM_VALUE_ID_INVALID;
  }
  if (index >= loop.op->operand_count) {
    return LOOM_VALUE_ID_INVALID;
  }
  return loom_op_operands(loop.op)[index];
}

loom_value_id_t loom_loop_like_step(loom_loop_like_t loop) {
  if (!loop.vtable) {
    return LOOM_VALUE_ID_INVALID;
  }
  uint8_t index = loop.vtable->step_operand_index;
  if (index == LOOM_OPERAND_INDEX_NONE) {
    return LOOM_VALUE_ID_INVALID;
  }
  if (loop.vtable->segmented_operands) {
    loom_value_slice_t slice =
        loom_loop_like_segmented_operand_field_span(loop, index);
    return slice.count == 1 ? slice.values[0] : LOOM_VALUE_ID_INVALID;
  }
  if (index >= loop.op->operand_count) {
    return LOOM_VALUE_ID_INVALID;
  }
  return loom_op_operands(loop.op)[index];
}

bool loom_loop_like_has_counted_range(loom_loop_like_t loop) {
  return loom_loop_like_lower_bound(loop) != LOOM_VALUE_ID_INVALID &&
         loom_loop_like_upper_bound(loop) != LOOM_VALUE_ID_INVALID &&
         loom_loop_like_step(loop) != LOOM_VALUE_ID_INVALID;
}

//===----------------------------------------------------------------------===//
// RegionBranch interface
//===----------------------------------------------------------------------===//

loom_region_branch_t loom_region_branch_cast(const loom_module_t* module,
                                             loom_op_t* op) {
  if (!op) {
    return (loom_region_branch_t){.op = NULL, .vtable = NULL};
  }
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  if (!vtable || !vtable->region_branch) {
    return (loom_region_branch_t){.op = NULL, .vtable = NULL};
  }
  return (loom_region_branch_t){.op = op, .vtable = vtable->region_branch};
}

loom_value_id_t loom_region_branch_selector(loom_region_branch_t branch) {
  if (!branch.vtable) {
    return LOOM_VALUE_ID_INVALID;
  }
  uint8_t selector_index = branch.vtable->selector_operand_index;
  IREE_ASSERT(selector_index < branch.op->operand_count);
  return loom_op_operands(branch.op)[selector_index];
}

loom_region_branch_truth_t loom_region_branch_region_truth(
    loom_region_branch_t branch, uint8_t region_index) {
  if (!branch.vtable || region_index >= branch.op->region_count) {
    return LOOM_REGION_BRANCH_TRUTH_UNKNOWN;
  }
  if (region_index == branch.vtable->true_region_index) {
    return LOOM_REGION_BRANCH_TRUTH_TRUE;
  }
  if (region_index == branch.vtable->false_region_index) {
    return LOOM_REGION_BRANCH_TRUTH_FALSE;
  }
  return LOOM_REGION_BRANCH_TRUTH_UNKNOWN;
}

loom_region_t* loom_region_branch_region(const loom_module_t* module,
                                         loom_region_branch_t branch,
                                         uint8_t region_index) {
  if (!module || !loom_region_branch_isa(branch) ||
      region_index >= branch.op->region_count) {
    return NULL;
  }
  const loom_op_vtable_t* vtable = loom_op_vtable(module, branch.op);
  if (!loom_op_vtable_region_descriptor(vtable, region_index)) {
    return NULL;
  }
  return loom_op_regions(branch.op)[region_index];
}

loom_op_t* loom_region_branch_region_terminator(const loom_module_t* module,
                                                loom_region_branch_t branch,
                                                uint8_t region_index) {
  if (!module || !loom_region_branch_isa(branch) ||
      region_index >= branch.op->region_count) {
    return NULL;
  }
  const loom_op_vtable_t* vtable = loom_op_vtable(module, branch.op);
  const loom_region_descriptor_t* region_descriptor =
      loom_op_vtable_region_descriptor(vtable, region_index);
  if (!region_descriptor) {
    return NULL;
  }

  loom_region_t* region = loom_op_regions(branch.op)[region_index];
  if (!region || region->block_count != 1) {
    return NULL;
  }
  loom_block_t* block = loom_region_entry_block(region);
  if (!block || !block->last_op) {
    return NULL;
  }
  return loom_region_descriptor_matches_terminator(region_descriptor,
                                                   block->last_op->kind)
             ? block->last_op
             : NULL;
}

bool loom_region_branch_region_yield_only_operands(
    const loom_module_t* module, loom_region_branch_t branch,
    uint8_t region_index, uint16_t expected_count,
    loom_value_slice_t* out_values) {
  if (out_values) {
    *out_values = (loom_value_slice_t){0};
  }
  loom_op_t* terminator =
      loom_region_branch_region_terminator(module, branch, region_index);
  if (!terminator) {
    return false;
  }

  loom_region_t* region =
      loom_region_branch_region(module, branch, region_index);
  if (!region) {
    return false;
  }
  loom_block_t* block = loom_region_entry_block(region);
  if (!block || block->first_op != terminator) {
    return false;
  }
  if (terminator->operand_count != expected_count) {
    return false;
  }
  if (out_values) {
    *out_values = (loom_value_slice_t){
        .values = loom_op_operands(terminator),
        .count = terminator->operand_count,
    };
  }
  return true;
}

iree_status_t loom_region_branch_build_region_terminator(
    loom_builder_t* builder, const loom_module_t* module,
    loom_region_branch_t branch, uint8_t region_index,
    const loom_value_id_t* values, iree_host_size_t value_count,
    loom_location_id_t location, loom_op_t** out_op) {
  if (!out_op) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "region branch terminator output is NULL");
  }
  *out_op = NULL;
  if (value_count > UINT16_MAX) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "region branch terminator operand count exceeds uint16_t range");
  }
  if (value_count > 0 && !values) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "region branch terminator operands are NULL");
  }
  if (!builder || builder->module != module ||
      !loom_region_branch_isa(branch) ||
      region_index >= branch.op->region_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid region branch terminator build request");
  }

  const loom_op_vtable_t* vtable = loom_op_vtable(module, branch.op);
  const loom_region_descriptor_t* region_descriptor =
      loom_op_vtable_region_descriptor(vtable, region_index);
  if (!region_descriptor) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "region branch has no region descriptor");
  }

  loom_op_kind_t terminator_kind = region_descriptor->terminator;
  if (terminator_kind == LOOM_OP_KIND_UNKNOWN) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "region branch has no yield-style terminator");
  }

  IREE_RETURN_IF_ERROR(loom_builder_allocate_op(builder, terminator_kind,
                                                (uint16_t)value_count, 0, 0, 0,
                                                0, location, out_op));
  if (value_count > 0) {
    memcpy(loom_op_operands(*out_op), values,
           value_count * sizeof(loom_value_id_t));
  }
  return loom_builder_finalize_op(builder, *out_op);
}

//===----------------------------------------------------------------------===//
// MemoryAccess interface
//===----------------------------------------------------------------------===//

loom_trait_flags_t loom_memory_access_effective_traits(
    const loom_module_t* module, const loom_op_t* op) {
  (void)module;
  loom_trait_flags_t traits = op->traits & ~LOOM_TRAIT_OBSERVABLE_EFFECT;
  if (iree_any_bit_set(op->instance_flags, LOOM_MEMORY_ACCESS_FLAG_VOLATILE)) {
    traits |= LOOM_TRAIT_OBSERVABLE_EFFECT;
  }
  return traits;
}

loom_memory_access_t loom_memory_access_cast(const loom_module_t* module,
                                             const loom_op_t* op) {
  if (!op) {
    return (loom_memory_access_t){.op = NULL, .op_vtable = NULL};
  }
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  if (!vtable || !vtable->memory_access) {
    return (loom_memory_access_t){.op = NULL, .op_vtable = NULL};
  }
  return (loom_memory_access_t){.op = op, .op_vtable = vtable};
}

static const loom_memory_access_vtable_t* loom_memory_access_vtable(
    loom_memory_access_t access) {
  return access.op_vtable ? access.op_vtable->memory_access : NULL;
}

loom_memory_access_operation_kind_t loom_memory_access_operation_kind(
    loom_memory_access_t access) {
  const loom_memory_access_vtable_t* vtable = loom_memory_access_vtable(access);
  return vtable ? (loom_memory_access_operation_kind_t)vtable->operation_kind
                : LOOM_MEMORY_ACCESS_OPERATION_COUNT_;
}

loom_memory_access_flags_t loom_memory_access_flags(
    loom_memory_access_t access) {
  return loom_memory_access_isa(access) ? access.op->instance_flags : 0;
}

static bool loom_memory_access_operand_field_contains(
    loom_memory_access_t access, uint8_t operand_field_index,
    uint16_t operand_index) {
  if (!access.op || !access.op_vtable ||
      operand_field_index == LOOM_OPERAND_INDEX_NONE ||
      operand_index >= access.op->operand_count) {
    return false;
  }
  if (!loom_op_vtable_has_segmented_operands(access.op_vtable)) {
    return operand_field_index == operand_index;
  }
  const loom_value_slice_t span = loom_op_operand_field_span(
      access.op_vtable, access.op, operand_field_index);
  if (span.count == 0 || span.values == NULL) {
    return false;
  }
  const uintptr_t operand_base = (uintptr_t)loom_op_const_operands(access.op);
  const uintptr_t span_base = (uintptr_t)span.values;
  if (span_base < operand_base) {
    return false;
  }
  const uintptr_t span_offset_bytes = span_base - operand_base;
  if (span_offset_bytes % sizeof(*span.values) != 0) {
    return false;
  }
  const uintptr_t first_operand_index =
      span_offset_bytes / sizeof(*span.values);
  return operand_index >= first_operand_index &&
         operand_index < first_operand_index + span.count;
}

bool loom_memory_access_operand_index_is_payload(loom_memory_access_t access,
                                                 uint16_t operand_index) {
  const loom_memory_access_vtable_t* vtable = loom_memory_access_vtable(access);
  if (!vtable ||
      !loom_memory_access_operation_kind_has_payload_operands(
          (loom_memory_access_operation_kind_t)vtable->operation_kind)) {
    return false;
  }
  return loom_memory_access_operand_field_contains(
             access, vtable->value_operand_index, operand_index) ||
         loom_memory_access_operand_field_contains(
             access, vtable->expected_operand_index, operand_index) ||
         loom_memory_access_operand_field_contains(
             access, vtable->replacement_operand_index, operand_index);
}

static loom_value_id_t loom_memory_access_operand(loom_memory_access_t access,
                                                  uint8_t operand_field_index) {
  if (!access.op || !access.op_vtable ||
      operand_field_index == LOOM_OPERAND_INDEX_NONE) {
    return LOOM_VALUE_ID_INVALID;
  }
  if (!loom_op_vtable_has_segmented_operands(access.op_vtable)) {
    if (operand_field_index >= access.op->operand_count) {
      return LOOM_VALUE_ID_INVALID;
    }
    return loom_op_operands(access.op)[operand_field_index];
  }
  loom_value_slice_t span = loom_op_operand_field_span(
      access.op_vtable, access.op, operand_field_index);
  return span.count == 1 ? span.values[0] : LOOM_VALUE_ID_INVALID;
}

static loom_attribute_t loom_memory_access_attr(loom_memory_access_t access,
                                                uint8_t attr_index) {
  if (!access.op || !loom_memory_access_vtable(access) ||
      attr_index == LOOM_ATTR_INDEX_NONE ||
      attr_index >= access.op->attribute_count) {
    return loom_attr_absent();
  }
  return loom_op_attrs(access.op)[attr_index];
}

loom_value_id_t loom_memory_access_view(loom_memory_access_t access) {
  const loom_memory_access_vtable_t* vtable = loom_memory_access_vtable(access);
  return loom_memory_access_operand(
      access, vtable ? vtable->view_operand_index : LOOM_OPERAND_INDEX_NONE);
}

loom_value_id_t loom_memory_access_byte_offset(loom_memory_access_t access) {
  const loom_memory_access_vtable_t* vtable = loom_memory_access_vtable(access);
  return loom_memory_access_operand(
      access,
      vtable ? vtable->byte_offset_operand_index : LOOM_OPERAND_INDEX_NONE);
}

loom_value_id_t loom_memory_access_value(loom_memory_access_t access) {
  const loom_memory_access_vtable_t* vtable = loom_memory_access_vtable(access);
  return loom_memory_access_operand(
      access, vtable ? vtable->value_operand_index : LOOM_OPERAND_INDEX_NONE);
}

loom_value_id_t loom_memory_access_expected(loom_memory_access_t access) {
  const loom_memory_access_vtable_t* vtable = loom_memory_access_vtable(access);
  return loom_memory_access_operand(access, vtable
                                                ? vtable->expected_operand_index
                                                : LOOM_OPERAND_INDEX_NONE);
}

loom_value_id_t loom_memory_access_replacement(loom_memory_access_t access) {
  const loom_memory_access_vtable_t* vtable = loom_memory_access_vtable(access);
  return loom_memory_access_operand(
      access,
      vtable ? vtable->replacement_operand_index : LOOM_OPERAND_INDEX_NONE);
}

loom_value_id_t loom_memory_access_mask(loom_memory_access_t access) {
  const loom_memory_access_vtable_t* vtable = loom_memory_access_vtable(access);
  return loom_memory_access_operand(
      access, vtable ? vtable->mask_operand_index : LOOM_OPERAND_INDEX_NONE);
}

loom_value_id_t loom_memory_access_passthrough(loom_memory_access_t access) {
  const loom_memory_access_vtable_t* vtable = loom_memory_access_vtable(access);
  return loom_memory_access_operand(
      access,
      vtable ? vtable->passthrough_operand_index : LOOM_OPERAND_INDEX_NONE);
}

loom_value_id_t loom_memory_access_offsets(loom_memory_access_t access) {
  const loom_memory_access_vtable_t* vtable = loom_memory_access_vtable(access);
  return loom_memory_access_operand(
      access, vtable ? vtable->offsets_operand_index : LOOM_OPERAND_INDEX_NONE);
}

loom_value_slice_t loom_memory_access_dynamic_indices(
    loom_memory_access_t access) {
  const loom_memory_access_vtable_t* vtable = loom_memory_access_vtable(access);
  if (!access.op || !access.op_vtable || !vtable ||
      vtable->indices_operand_field_index == LOOM_OPERAND_INDEX_NONE) {
    return (loom_value_slice_t){.values = NULL, .count = 0};
  }
  if (!loom_op_vtable_has_segmented_operands(access.op_vtable)) {
    uint8_t offset = vtable->indices_operand_field_index;
    if (offset > access.op->operand_count) {
      return (loom_value_slice_t){.values = NULL, .count = 0};
    }
    return (loom_value_slice_t){
        .values = loom_op_operands(access.op) + offset,
        .count = (uint16_t)(access.op->operand_count - offset),
    };
  }
  return loom_op_operand_field_span(access.op_vtable, access.op,
                                    vtable->indices_operand_field_index);
}

loom_attribute_t loom_memory_access_static_indices(
    loom_memory_access_t access) {
  const loom_memory_access_vtable_t* vtable = loom_memory_access_vtable(access);
  return loom_memory_access_attr(access, vtable
                                             ? vtable->static_indices_attr_index
                                             : LOOM_ATTR_INDEX_NONE);
}

loom_attribute_t loom_memory_access_atomic_kind(loom_memory_access_t access) {
  const loom_memory_access_vtable_t* vtable = loom_memory_access_vtable(access);
  return loom_memory_access_attr(
      access, vtable ? vtable->atomic_kind_attr_index : LOOM_ATTR_INDEX_NONE);
}

loom_attribute_t loom_memory_access_atomic_ordering(
    loom_memory_access_t access) {
  const loom_memory_access_vtable_t* vtable = loom_memory_access_vtable(access);
  return loom_memory_access_attr(
      access,
      vtable ? vtable->atomic_ordering_attr_index : LOOM_ATTR_INDEX_NONE);
}

loom_attribute_t loom_memory_access_atomic_success_ordering(
    loom_memory_access_t access) {
  const loom_memory_access_vtable_t* vtable = loom_memory_access_vtable(access);
  return loom_memory_access_attr(
      access, vtable ? vtable->atomic_success_ordering_attr_index
                     : LOOM_ATTR_INDEX_NONE);
}

loom_attribute_t loom_memory_access_atomic_failure_ordering(
    loom_memory_access_t access) {
  const loom_memory_access_vtable_t* vtable = loom_memory_access_vtable(access);
  return loom_memory_access_attr(
      access, vtable ? vtable->atomic_failure_ordering_attr_index
                     : LOOM_ATTR_INDEX_NONE);
}

loom_attribute_t loom_memory_access_atomic_scope(loom_memory_access_t access) {
  const loom_memory_access_vtable_t* vtable = loom_memory_access_vtable(access);
  return loom_memory_access_attr(
      access, vtable ? vtable->atomic_scope_attr_index : LOOM_ATTR_INDEX_NONE);
}

iree_status_t loom_op_set_attr(loom_module_t* module, loom_op_t* op,
                               uint8_t attribute_index,
                               loom_attribute_t attribute) {
  IREE_RETURN_IF_ERROR(
      loom_module_set_op_attribute(module, op, attribute_index, attribute));
  loom_trait_flags_t old_traits = op->traits;
  loom_op_refresh_effective_traits(module, op);
  loom_module_update_op_direct_summaries(module, op, old_traits, op->traits);
  return iree_ok_status();
}
