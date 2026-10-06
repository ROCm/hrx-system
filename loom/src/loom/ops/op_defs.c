// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/op_defs.h"

#include <stdint.h>
#include <string.h>

#include "iree/base/internal/arena.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/util/adaptive_sort.h"

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
  if (!vtable) {
    op->traits = LOOM_TRAIT_UNKNOWN_EFFECTS;
    return;
  }
  if (vtable->effective_traits) {
    op->traits = vtable->effective_traits(op);
  }
}

bool loom_op_may_write(const loom_module_t* module, const loom_op_t* op) {
  return loom_traits_may_write(loom_op_effective_traits(module, op));
}

static bool loom_value_has_type_uses_outside_op(const loom_module_t* module,
                                                loom_value_id_t value_id,
                                                const loom_op_t* op) {
  if (value_id >= module->values.count) {
    return false;
  }
  loom_type_use_iterator_t type_users;
  loom_module_value_type_users(module, value_id, &type_users);
  for (loom_value_id_t user_value_id = loom_type_users_next(&type_users);
       user_value_id != LOOM_VALUE_ID_INVALID;
       user_value_id = loom_type_users_next(&type_users)) {
    const loom_value_t* user_value = loom_module_value(module, user_value_id);
    if (loom_value_is_block_arg(user_value)) {
      return true;
    }
    if (loom_value_def_op(user_value) != op) {
      return true;
    }
  }
  return false;
}

static bool loom_value_has_attribute_uses_outside_op(
    const loom_module_t* module, loom_value_id_t value_id,
    const loom_op_t* op) {
  if (!loom_value_has_attribute_uses(loom_module_value(module, value_id))) {
    return false;
  }
  loom_type_use_iterator_t users;
  loom_attribute_users_begin(&module->type_uses, value_id, &users);
  for (loom_attribute_user_t user = loom_attribute_users_next(&users); user.op;
       user = loom_attribute_users_next(&users)) {
    // Self-predicates, such as config.decl constraints, die with the owner.
    if (user.op != op) {
      return true;
    }
  }
  return false;
}

bool loom_op_results_unused(const loom_module_t* module, const loom_op_t* op) {
  loom_value_id_t* results = loom_op_results((loom_op_t*)op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] == LOOM_VALUE_ID_INVALID) {
      continue;
    }
    const loom_value_t* value = loom_module_value(module, results[i]);
    if (value->use_count > 0) {
      return false;
    }
    if (loom_value_has_attribute_uses_outside_op(module, results[i], op)) {
      return false;
    }
    if (loom_value_has_type_uses_outside_op(module, results[i], op)) {
      return false;
    }
  }
  return true;
}

bool loom_op_is_trivially_dead(const loom_module_t* module,
                               const loom_op_t* op) {
  if (op->result_count == 0) {
    return false;
  }
  loom_trait_flags_t traits = loom_op_effective_traits(module, op);
  if (iree_any_bit_set(traits, LOOM_TRAIT_HINT)) {
    return false;
  }
  if (loom_traits_are_convergent(traits)) {
    return false;
  }
  if (loom_traits_have_observable_effects(traits)) {
    return false;
  }
  if (loom_traits_may_write(traits)) {
    return false;
  }
  if (loom_op_regions_have_write_effects(op)) {
    return false;
  }
  if (loom_op_regions_have_convergent_effects(op)) {
    return false;
  }
  if (loom_op_regions_have_observable_effects(op)) {
    return false;
  }
  if (loom_op_regions_have_hints(op)) {
    return false;
  }
  return loom_op_results_unused(module, op);
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

uint8_t loom_call_like_purity(loom_call_like_t call) {
  if (!call.vtable) {
    return 0;
  }
  if (call.vtable->purity_attr_index == LOOM_ATTR_INDEX_NONE) {
    return 0;
  }
  return loom_attr_as_enum(
      loom_op_attrs(call.op)[call.vtable->purity_attr_index]);
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

uint8_t loom_func_like_purity(loom_func_like_t func) {
  if (!func.vtable) {
    return 0;
  }
  if (func.vtable->purity_attr_index == LOOM_ATTR_INDEX_NONE) {
    return 0;
  }
  return loom_attr_as_enum(
      loom_op_attrs(func.op)[func.vtable->purity_attr_index]);
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

loom_trait_flags_t loom_memory_access_effective_traits(const loom_op_t* op) {
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

//===----------------------------------------------------------------------===//
// Builder
//===----------------------------------------------------------------------===//

void loom_builder_initialize(loom_module_t* module,
                             iree_arena_allocator_t* arena, loom_block_t* block,
                             loom_builder_t* out_builder) {
  out_builder->module = module;
  out_builder->arena = arena;
  out_builder->ip.block = block;
  out_builder->ip.parent_op = NULL;
  out_builder->ip.before_op = NULL;
  out_builder->on_op_finalized.fn = NULL;
  out_builder->on_op_finalized.user_data = NULL;
  out_builder->reserved_value_ids = NULL;
  out_builder->reserved_value_count = 0;
  out_builder->reserved_value_next = 0;
}

void loom_builder_set_block(loom_builder_t* builder, loom_block_t* block) {
  builder->ip.block = block;
  builder->ip.before_op = NULL;
}

loom_builder_ip_t loom_builder_enter_region(loom_builder_t* builder,
                                            loom_op_t* parent_op,
                                            loom_region_t* region) {
  loom_builder_ip_t saved = builder->ip;
  builder->ip.block = loom_region_entry_block(region);
  builder->ip.parent_op = parent_op;
  builder->ip.before_op = NULL;
  return saved;
}

void loom_builder_set_before(loom_builder_t* builder, const loom_op_t* op) {
  builder->ip.block = op->parent_block;
  builder->ip.parent_op = op->parent_op;
  builder->ip.before_op = (loom_op_t*)op;
}

void loom_builder_set_after(loom_builder_t* builder, const loom_op_t* op) {
  builder->ip.block = op->parent_block;
  builder->ip.parent_op = op->parent_op;
  builder->ip.before_op = op->next_op;
}

loom_builder_ip_t loom_builder_save(const loom_builder_t* builder) {
  return builder->ip;
}

void loom_builder_restore(loom_builder_t* builder, loom_builder_ip_t ip) {
  builder->ip = ip;
}

iree_status_t loom_builder_reserve_values(loom_builder_t* builder,
                                          iree_host_size_t count,
                                          loom_value_id_t* out_value_ids) {
  if (builder->reserved_value_count > 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "cannot reserve values: %" PRIhsz
                            " values already reserved",
                            builder->reserved_value_count);
  }
  loom_type_t none_type = {0};
  for (iree_host_size_t i = 0; i < count; ++i) {
    IREE_RETURN_IF_ERROR(loom_module_define_value(builder->module, none_type,
                                                  &out_value_ids[i]));
  }
  builder->reserved_value_ids = out_value_ids;
  builder->reserved_value_count = count;
  builder->reserved_value_next = 0;
  return iree_ok_status();
}

iree_status_t loom_builder_define_value(loom_builder_t* builder,
                                        loom_type_t type,
                                        loom_value_id_t* out_value_id) {
  if (builder->reserved_value_next < builder->reserved_value_count) {
    loom_value_id_t id =
        builder->reserved_value_ids[builder->reserved_value_next++];
    IREE_RETURN_IF_ERROR(loom_module_set_value_type(builder->module, id, type));
    *out_value_id = id;
    return iree_ok_status();
  }
  return loom_module_define_value(builder->module, type, out_value_id);
}

iree_status_t loom_builder_define_block_arg(loom_builder_t* builder,
                                            loom_block_t* block,
                                            loom_type_t type,
                                            loom_value_id_t* out_value_id) {
  IREE_RETURN_IF_ERROR(
      loom_module_define_value(builder->module, type, out_value_id));
  return loom_block_add_arg(builder->module, block, *out_value_id);
}

iree_status_t loom_builder_create_region(loom_builder_t* builder, loom_op_t* op,
                                         uint8_t region_index,
                                         loom_block_t** out_entry_block) {
  if (!builder || !builder->module) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "builder has no module");
  }
  if (!op) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "region owner op is NULL");
  }
  if (region_index >= op->region_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "region index %u is out of range for op with %u region(s)",
        (unsigned)region_index, (unsigned)op->region_count);
  }
  loom_region_t* region = NULL;
  IREE_RETURN_IF_ERROR(
      loom_module_allocate_region(builder->module, 1, &region));
  loom_op_regions(op)[region_index] = region;
  if (out_entry_block) {
    *out_entry_block = loom_region_entry_block(region);
  }
  return iree_ok_status();
}

iree_status_t loom_builder_define_result(loom_builder_t* builder,
                                         loom_type_t result_type,
                                         loom_value_id_t* out_result) {
  return loom_builder_define_value(builder, result_type, out_result);
}

iree_status_t loom_builder_define_results(loom_builder_t* builder,
                                          const loom_type_t* result_types,
                                          iree_host_size_t result_count,
                                          loom_value_id_t* result_storage) {
  if (result_count == 0) {
    return iree_ok_status();
  }
  if (!result_types) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "result type storage is NULL for non-zero result count");
  }
  if (!result_storage) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "result id storage is NULL for non-zero result count");
  }
  for (iree_host_size_t i = 0; i < result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_builder_define_result(builder, result_types[i],
                                                    &result_storage[i]));
  }
  return iree_ok_status();
}

iree_status_t loom_builder_copy_tied_results(
    const loom_tied_result_t* tied_results, iree_host_size_t tied_result_count,
    loom_op_t* op) {
  if (tied_result_count == 0) {
    return iree_ok_status();
  }
  if (!op) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tied result owner op is NULL");
  }
  if (!tied_results) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "tied result storage is NULL for non-zero tied result count");
  }
  if (tied_result_count > op->tied_result_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tied result count %" PRIhsz
                            " exceeds op tied result storage count %u",
                            tied_result_count, (unsigned)op->tied_result_count);
  }
  memcpy(loom_op_tied_results(op), tied_results,
         tied_result_count * sizeof(loom_tied_result_t));
  return iree_ok_status();
}

iree_status_t loom_builder_intern_string(loom_builder_t* builder,
                                         iree_string_view_t string,
                                         loom_string_id_t* out_string_id) {
  return loom_module_intern_string(builder->module, string, out_string_id);
}

iree_status_t loom_builder_check_count_range(iree_host_size_t count,
                                             iree_host_size_t max_count,
                                             iree_string_view_t label) {
  if (count <= max_count) {
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                          "%.*s count %" PRIhsz " exceeds max %" PRIhsz,
                          (int)label.size, label.data, count, max_count);
}

iree_status_t loom_builder_copy_i64_array_attr_storage(loom_builder_t* builder,
                                                       const int64_t* values,
                                                       iree_host_size_t count,
                                                       iree_string_view_t label,
                                                       int64_t** out_storage) {
  if (!out_storage) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "i64-array storage output is NULL");
  }
  *out_storage = NULL;
  if (count == 0) {
    return iree_ok_status();
  }
  if (!builder || !builder->arena) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "builder has no arena");
  }
  if (!values) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s storage is NULL for non-zero count",
                            (int)label.size, label.data);
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      builder->arena, count, sizeof(**out_storage), (void**)out_storage));
  memcpy(*out_storage, values, count * sizeof(**out_storage));
  return iree_ok_status();
}

iree_status_t loom_builder_copy_enum_array_attr_storage(
    loom_builder_t* builder, loom_enum_array_t values, iree_string_view_t label,
    const uint8_t** out_storage) {
  if (!out_storage) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "enum-array storage output is NULL");
  }
  *out_storage = NULL;
  IREE_RETURN_IF_ERROR(
      loom_builder_check_count_range(values.count, UINT16_MAX, label));
  if (values.count == 0) {
    return iree_ok_status();
  }
  if (!builder || !builder->arena) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "builder has no arena");
  }
  if (!values.values) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s storage is NULL for non-zero count",
                            (int)label.size, label.data);
  }
  uint8_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      builder->arena, values.count, sizeof(*storage), (void**)&storage));
  memcpy(storage, values.values, values.count * sizeof(*storage));
  *out_storage = storage;
  return iree_ok_status();
}

iree_status_t loom_builder_copy_signed_enum_set_attr_storage(
    loom_builder_t* builder, loom_signed_enum_set_t set,
    iree_string_view_t label, const uint64_t** out_storage,
    uint16_t* out_word_count) {
  if (!out_storage || !out_word_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "signed enum-set storage outputs must both be non-NULL");
  }
  *out_storage = NULL;
  *out_word_count = 0;

  iree_host_size_t canonical_word_count = 0;
  iree_status_t status =
      loom_signed_enum_set_canonical_word_count(set, &canonical_word_count);
  if (!iree_status_is_ok(status)) {
    return iree_status_annotate_f(status, "%.*s", (int)label.size, label.data);
  }
  if (canonical_word_count == 0) {
    return iree_ok_status();
  }
  if (!builder || !builder->arena) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "builder has no arena");
  }
  uint64_t* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(builder->arena, canonical_word_count * 2,
                                sizeof(*storage), (void**)&storage));
  memcpy(storage, set.words, canonical_word_count * sizeof(*storage));
  memcpy(storage + canonical_word_count, set.words + set.word_count,
         canonical_word_count * sizeof(*storage));
  *out_storage = storage;
  *out_word_count = (uint16_t)canonical_word_count;
  return iree_ok_status();
}

iree_status_t loom_builder_copy_symbol_array_attr_storage(
    loom_builder_t* builder, loom_symbol_ref_array_t values,
    iree_string_view_t label, const loom_symbol_ref_t** out_storage) {
  if (!out_storage) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol-array storage output is NULL");
  }
  *out_storage = NULL;
  IREE_RETURN_IF_ERROR(
      loom_builder_check_count_range(values.count, UINT16_MAX, label));
  if (values.count == 0) {
    return iree_ok_status();
  }
  if (!builder || !builder->arena) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "builder has no arena");
  }
  if (!values.values) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s storage is NULL for non-zero count",
                            (int)label.size, label.data);
  }
  loom_symbol_ref_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      builder->arena, values.count, sizeof(*storage), (void**)&storage));
  memcpy(storage, values.values, values.count * sizeof(*storage));
  *out_storage = storage;
  return iree_ok_status();
}

iree_status_t loom_builder_copy_predicate_list_attr_storage(
    loom_builder_t* builder, const loom_predicate_t* predicates,
    iree_host_size_t count, iree_string_view_t label,
    loom_predicate_t** out_storage) {
  if (!out_storage) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "predicate-list storage output is NULL");
  }
  *out_storage = NULL;
  if (count == 0) {
    return iree_ok_status();
  }
  if (!builder || !builder->arena) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "builder has no arena");
  }
  if (!predicates) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s storage is NULL for non-zero count",
                            (int)label.size, label.data);
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      builder->arena, count, sizeof(**out_storage), (void**)out_storage));
  memcpy(*out_storage, predicates, count * sizeof(**out_storage));
  return iree_ok_status();
}

iree_status_t loom_builder_copy_bytes_attr_storage(
    loom_builder_t* builder, iree_const_byte_span_t bytes,
    iree_string_view_t label, const uint8_t** out_storage) {
  if (!out_storage) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "byte storage output is NULL");
  }
  *out_storage = NULL;
  if (bytes.data_length == 0) {
    return iree_ok_status();
  }
  if (bytes.data_length > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "%.*s byte length %" PRIhsz " exceeds max %u",
                            (int)label.size, label.data, bytes.data_length,
                            UINT32_MAX);
  }
  if (!builder || !builder->arena) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "builder has no arena");
  }
  if (!bytes.data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s storage is NULL for non-zero byte length",
                            (int)label.size, label.data);
  }
  uint8_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      builder->arena, bytes.data_length, sizeof(*storage), (void**)&storage));
  memcpy(storage, bytes.data, bytes.data_length);
  *out_storage = storage;
  return iree_ok_status();
}

iree_status_t loom_builder_set_operand_dict(
    loom_builder_t* builder, loom_named_value_slice_t named_values,
    loom_value_id_t* operand_storage, loom_attribute_t* out_names_attr) {
  if (!out_names_attr) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "operand dictionary names attribute output is NULL");
  }
  *out_names_attr = loom_attr_absent();
  if (!builder || !builder->module || !builder->arena) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "builder has no module or arena");
  }
  if (named_values.count == 0) {
    return iree_ok_status();
  }
  if (!named_values.entries) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "non-empty operand dictionary has a NULL entry pointer");
  }
  if (!operand_storage) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "non-empty operand dictionary has a NULL operand storage pointer");
  }
  if (named_values.count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "operand dictionary has %" PRIhsz
                            " entries, max %u",
                            named_values.count, (unsigned)UINT16_MAX);
  }

  for (iree_host_size_t i = 0; i < named_values.count; ++i) {
    const loom_named_value_t entry = named_values.entries[i];
    if (entry.reserved != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "operand dictionary entry reserved bits must be zero");
    }
    if (entry.name_id == LOOM_STRING_ID_INVALID ||
        entry.name_id >= builder->module->strings.count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "operand dictionary key string id %u is out of range (module has "
          "%" PRIhsz " strings)",
          entry.name_id, builder->module->strings.count);
    }
    if (entry.value_id == LOOM_VALUE_ID_INVALID ||
        entry.value_id >= builder->module->values.count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "operand dictionary value id %u is out of range (module has %" PRIhsz
          " values)",
          entry.value_id, builder->module->values.count);
    }
  }

  loom_named_attr_t* name_entries = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(&builder->module->arena, named_values.count,
                                sizeof(*name_entries), (void**)&name_entries));
  // Until canonical positions are assigned, each entry carries its paired SSA
  // value. Sorting the owned entries keeps names and values together without
  // retaining a separate permutation or caller-owned payload.
  for (iree_host_size_t i = 0; i < named_values.count; ++i) {
    name_entries[i] = (loom_named_attr_t){
        .name_id = named_values.entries[i].name_id,
        .reserved = 0,
        .value = loom_attr_i64(named_values.entries[i].value_id),
    };
  }
  loom_module_sort_attr_dict_entries(builder->module, name_entries,
                                     named_values.count);
  for (iree_host_size_t i = 1; i < named_values.count; ++i) {
    if (name_entries[i - 1].name_id == name_entries[i].name_id) {
      iree_string_view_t name = loom_string_table_get(&builder->module->strings,
                                                      name_entries[i].name_id);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate operand dictionary key '%.*s'",
                              (int)name.size, name.data);
    }
  }
  for (iree_host_size_t i = 0; i < named_values.count; ++i) {
    operand_storage[i] = (loom_value_id_t)name_entries[i].value.i64;
    name_entries[i].value = loom_attr_i64((int64_t)i);
  }
  *out_names_attr =
      loom_make_canonical_attr_dict(name_entries, named_values.count);
  return iree_ok_status();
}

static iree_status_t loom_builder_validate_operand_segments(
    const loom_op_vtable_t* vtable, uint16_t operand_count,
    const uint16_t* operand_segment_counts, uint8_t operand_segment_count) {
  uint8_t expected_segment_count = loom_op_vtable_operand_segment_count(vtable);
  if (expected_segment_count == 0) {
    if (operand_segment_count != 0 || operand_segment_counts) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "operand segment counts provided for non-segmented op");
    }
    return iree_ok_status();
  }
  if (!operand_segment_counts ||
      operand_segment_count != expected_segment_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "segmented op requires %u operand segment count(s), got %u",
        expected_segment_count, operand_segment_count);
  }
  uint32_t total_count = 0;
  for (uint8_t i = 0; i < operand_segment_count; ++i) {
    const loom_operand_descriptor_t* descriptor =
        &vtable->operand_descriptors[i];
    const uint16_t segment_count = operand_segment_counts[i];
    if (!iree_any_bit_set(descriptor->flags, LOOM_OPERAND_VARIADIC)) {
      const bool optional =
          iree_any_bit_set(descriptor->flags, LOOM_OPERAND_OPTIONAL);
      if ((!optional && segment_count != 1) ||
          (optional && segment_count > 1)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "operand segment %u has %u value(s), expected %s", i,
            (unsigned)segment_count, optional ? "0 or 1" : "1");
      }
    }
    total_count += segment_count;
  }
  if (total_count != operand_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "operand segment counts sum to %u but op has %u operand(s)",
        (unsigned)total_count, (unsigned)operand_count);
  }
  return iree_ok_status();
}

static iree_status_t loom_builder_allocate_op_storage(
    loom_builder_t* builder, loom_op_kind_t kind, uint16_t operand_count,
    const uint16_t* operand_segment_counts, uint8_t operand_segment_count,
    uint16_t result_count, uint8_t successor_count, uint8_t region_count,
    uint16_t tied_result_count, uint8_t attribute_count,
    loom_location_id_t location, loom_op_t** out_op) {
  *out_op = NULL;
  if (!builder->ip.block || !builder->module) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "builder has no insertion block or module");
  }
  const loom_op_vtable_t* vtable =
      loom_context_resolve_op(builder->module->context, kind);
  IREE_RETURN_IF_ERROR(loom_builder_validate_operand_segments(
      vtable, operand_count, operand_segment_counts, operand_segment_count));

  iree_host_size_t successors_size =
      (iree_host_size_t)successor_count * sizeof(loom_block_t*);
  iree_host_size_t regions_size =
      (iree_host_size_t)region_count * sizeof(loom_region_t*);
  iree_host_size_t operands_size =
      (iree_host_size_t)operand_count * sizeof(loom_value_id_t);
  iree_host_size_t results_size =
      (iree_host_size_t)result_count * sizeof(loom_value_id_t);
  iree_host_size_t operand_use_indices_size =
      (iree_host_size_t)operand_count * sizeof(loom_use_index_t);
  iree_host_size_t tied_size =
      (iree_host_size_t)tied_result_count * sizeof(loom_tied_result_t);
  iree_host_size_t operand_segment_counts_size =
      (iree_host_size_t)operand_segment_count * sizeof(uint16_t);

  iree_host_size_t before_attrs = sizeof(loom_op_t) + successors_size +
                                  regions_size + operands_size + results_size +
                                  operand_use_indices_size + tied_size;
  iree_host_size_t aligned_before_attrs =
      (attribute_count > 0 || operand_segment_count > 0)
          ? iree_host_align(before_attrs, iree_alignof(loom_attribute_t))
          : before_attrs;
  iree_host_size_t attrs_size =
      (iree_host_size_t)attribute_count * sizeof(loom_attribute_t);
  iree_host_size_t total_size =
      aligned_before_attrs + attrs_size +
      (iree_host_size_t)attribute_count * sizeof(loom_attribute_owner_id_t) +
      operand_segment_counts_size;

  void* allocation = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(builder->arena, total_size, &allocation));
  memset(allocation, 0, total_size);

  loom_op_t* op = (loom_op_t*)allocation;
  op->kind = kind;
  op->operand_count = operand_count;
  op->result_count = result_count;
  op->successor_count = successor_count;
  op->region_count = region_count;
  op->tied_result_count = tied_result_count;
  op->attribute_count = attribute_count;
  op->traits = vtable ? vtable->traits : LOOM_TRAIT_UNKNOWN_EFFECTS;
  op->location = location;
  op->parent_op = builder->ip.parent_op;
  loom_use_index_t* operand_use_indices = loom_op_operand_use_indices(op);
  for (uint16_t i = 0; i < operand_count; ++i) {
    operand_use_indices[i] = LOOM_USE_INDEX_INVALID;
  }
  if (operand_segment_count > 0) {
    memcpy(loom_op_operand_segment_counts(op), operand_segment_counts,
           operand_segment_counts_size);
  }

  if (!builder->ip.before_op) {
    IREE_RETURN_IF_ERROR(
        loom_block_append_op(builder->module, builder->ip.block, op));
  } else {
    IREE_RETURN_IF_ERROR(loom_block_insert_before_op(
        builder->module, builder->ip.block, builder->ip.before_op, op));
  }

  *out_op = op;
  return iree_ok_status();
}

iree_status_t loom_builder_allocate_op(
    loom_builder_t* builder, loom_op_kind_t kind, uint16_t operand_count,
    uint16_t result_count, uint8_t region_count, uint16_t tied_result_count,
    uint8_t attribute_count, loom_location_id_t location, loom_op_t** out_op) {
  return loom_builder_allocate_op_storage(
      builder, kind, operand_count, /*operand_segment_counts=*/NULL,
      /*operand_segment_count=*/0, result_count, /*successor_count=*/0,
      region_count, tied_result_count, attribute_count, location, out_op);
}

iree_status_t loom_builder_allocate_op_with_successors(
    loom_builder_t* builder, loom_op_kind_t kind, uint16_t operand_count,
    uint16_t result_count, uint8_t successor_count, uint8_t region_count,
    uint16_t tied_result_count, uint8_t attribute_count,
    loom_location_id_t location, loom_op_t** out_op) {
  return loom_builder_allocate_op_storage(
      builder, kind, operand_count, /*operand_segment_counts=*/NULL,
      /*operand_segment_count=*/0, result_count, successor_count, region_count,
      tied_result_count, attribute_count, location, out_op);
}

iree_status_t loom_builder_allocate_segmented_op(
    loom_builder_t* builder, loom_op_kind_t kind, uint16_t operand_count,
    const uint16_t* operand_segment_counts, uint8_t operand_segment_count,
    uint16_t result_count, uint8_t region_count, uint16_t tied_result_count,
    uint8_t attribute_count, loom_location_id_t location, loom_op_t** out_op) {
  return loom_builder_allocate_op_storage(
      builder, kind, operand_count, operand_segment_counts,
      operand_segment_count, result_count, /*successor_count=*/0, region_count,
      tied_result_count, attribute_count, location, out_op);
}

iree_status_t loom_builder_allocate_segmented_op_with_successors(
    loom_builder_t* builder, loom_op_kind_t kind, uint16_t operand_count,
    const uint16_t* operand_segment_counts, uint8_t operand_segment_count,
    uint16_t result_count, uint8_t successor_count, uint8_t region_count,
    uint16_t tied_result_count, uint8_t attribute_count,
    loom_location_id_t location, loom_op_t** out_op) {
  return loom_builder_allocate_op_storage(
      builder, kind, operand_count, operand_segment_counts,
      operand_segment_count, result_count, successor_count, region_count,
      tied_result_count, attribute_count, location, out_op);
}

iree_status_t loom_op_remove_results(loom_module_t* module, loom_op_t* op,
                                     const bool* remove_results,
                                     iree_arena_allocator_t* scratch_arena,
                                     uint16_t* out_removed_count) {
  *out_removed_count = 0;
  uint16_t old_result_count = op->result_count;
  if (old_result_count == 0) {
    return iree_ok_status();
  }

  uint16_t* result_map = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(scratch_arena, old_result_count,
                                sizeof(*result_map), (void**)&result_map));

  loom_value_id_t* results = loom_op_results(op);
  uint16_t kept_count = 0;
  for (uint16_t i = 0; i < old_result_count; ++i) {
    if (!remove_results[i]) {
      result_map[i] = kept_count++;
      continue;
    }

    result_map[i] = UINT16_MAX;
    loom_value_id_t result = results[i];
    if (result == LOOM_VALUE_ID_INVALID || result >= module->values.count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "removed result %u value %%%u is invalid",
                              (unsigned)i, (unsigned)result);
    }
    const loom_value_t* value = loom_module_value(module, result);
    if (value->use_count != 0) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "cannot remove result %%%u with %u operand use(s)", (unsigned)result,
          (unsigned)value->use_count);
    }
    if (loom_value_has_attribute_uses(value)) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "cannot remove result %%%u with attribute use(s)",
                              (unsigned)result);
    }
    if (loom_module_value_has_type_uses(module, result)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "cannot remove result %%%u with incoming type use(s)",
          (unsigned)result);
    }
  }
  *out_removed_count = (uint16_t)(old_result_count - kept_count);
  if (*out_removed_count == 0) {
    return iree_ok_status();
  }

  const loom_tied_result_t* old_tied_results = loom_op_tied_results(op);
  loom_tied_result_t* kept_tied_results = NULL;
  if (op->tied_result_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, op->tied_result_count, sizeof(*kept_tied_results),
        (void**)&kept_tied_results));
  }

  uint16_t kept_tied_count = 0;
  for (uint16_t i = 0; i < op->tied_result_count; ++i) {
    loom_tied_result_t tied_result = old_tied_results[i];
    if (tied_result.result_index >= old_result_count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "tied result index %u is out of range for %u result(s)",
          (unsigned)tied_result.result_index, (unsigned)old_result_count);
    }
    uint16_t new_result_index = result_map[tied_result.result_index];
    if (new_result_index == UINT16_MAX) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "cannot remove result %u because it is tied to operand %u",
          (unsigned)tied_result.result_index,
          (unsigned)tied_result.operand_index);
    }
    tied_result.result_index = new_result_index;
    kept_tied_results[kept_tied_count++] = tied_result;
  }

  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  uint8_t operand_segment_count = loom_op_vtable_operand_segment_count(vtable);
  uint16_t* old_operand_segment_counts =
      operand_segment_count > 0 ? loom_op_operand_segment_counts(op) : NULL;
  loom_use_index_t* old_operand_use_indices = loom_op_operand_use_indices(op);
  loom_attribute_t* old_attrs = loom_op_attrs(op);
  loom_attribute_owner_id_t* old_attribute_owners =
      loom_op_attribute_owners(op);
  for (uint16_t i = 0; i < old_result_count; ++i) {
    loom_value_id_t result = results[i];
    if (remove_results[i]) {
      loom_module_drop_value_type_uses(module, result);
      loom_module_value(module, result)->def = loom_value_def_make_none();
      continue;
    }
    uint16_t new_index = result_map[i];
    results[new_index] = result;
    loom_module_value(module, result)->def =
        loom_value_def_make_op(op, new_index);
  }

  op->result_count = kept_count;
  op->tied_result_count = kept_tied_count;

  if (op->operand_count > 0) {
    memmove(
        loom_op_operand_use_indices(op), old_operand_use_indices,
        (iree_host_size_t)op->operand_count * sizeof(*old_operand_use_indices));
  }
  if (kept_tied_count > 0) {
    memmove(loom_op_tied_results(op), kept_tied_results,
            (iree_host_size_t)kept_tied_count * sizeof(*kept_tied_results));
  }
  if (op->attribute_count > 0) {
    memmove(loom_op_attrs(op), old_attrs,
            (iree_host_size_t)op->attribute_count * sizeof(*old_attrs));
    memmove(
        loom_op_attribute_owners(op), old_attribute_owners,
        (iree_host_size_t)op->attribute_count * sizeof(*old_attribute_owners));
  }
  if (operand_segment_count > 0) {
    memmove(loom_op_operand_segment_counts(op), old_operand_segment_counts,
            (iree_host_size_t)operand_segment_count *
                sizeof(*old_operand_segment_counts));
  }
  loom_op_refresh_operand_ownership(module, op);
  return iree_ok_status();
}

static iree_status_t loom_op_verify_erase_preconditions(loom_module_t* module,
                                                        loom_op_t* op) {
  loom_value_id_t* results = loom_op_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] == LOOM_VALUE_ID_INVALID) {
      continue;
    }
    const loom_value_t* value = loom_module_value(module, results[i]);
    if (value->use_count > 0) {
      iree_string_view_t op_name = loom_op_name(module, op);
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "cannot erase %.*s: result %%%u still has %u use(s)",
          (int)op_name.size, op_name.data, (unsigned)results[i],
          (unsigned)value->use_count);
    }
    if (loom_value_has_attribute_uses_outside_op(module, results[i], op)) {
      iree_string_view_t op_name = loom_op_name(module, op);
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "cannot erase %.*s: result %%%u still has attribute use(s)",
          (int)op_name.size, op_name.data, (unsigned)results[i]);
    }
    if (loom_value_has_type_uses_outside_op(module, results[i], op)) {
      iree_string_view_t op_name = loom_op_name(module, op);
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "cannot erase %.*s: result %%%u still has type use(s)",
          (int)op_name.size, op_name.data, (unsigned)results[i]);
    }
  }
  return iree_ok_status();
}

static void loom_block_drop_arg_type_uses(loom_module_t* module,
                                          loom_block_t* block) {
  for (uint16_t i = 0; i < block->arg_count; ++i) {
    loom_value_id_t arg_id = loom_block_arg_id(block, i);
    if (arg_id == LOOM_VALUE_ID_INVALID || arg_id >= module->values.count) {
      continue;
    }
    loom_module_drop_value_type_uses(module, arg_id);
    loom_value_t* value = loom_module_value(module, arg_id);
    // Dead region block arguments stop being type-use carriers. Clearing the
    // block-arg bit keeps bulk type-use recomputation from resurrecting SSA
    // references through an unreachable region.
    value->flags &= ~LOOM_VALUE_FLAG_BLOCK_ARG;
    value->def = loom_value_def_make_none();
  }
  block->arg_count = 0;
}

static void loom_module_unlink_symbol_defining_op(
    loom_module_t* module, loom_op_t* op, const loom_op_vtable_t* vtable) {
  if (!vtable || !iree_any_bit_set(vtable->traits, LOOM_TRAIT_SYMBOL_DEFINE) ||
      !vtable->symbol_def || !vtable->attr_descriptors) {
    return;
  }
  uint8_t symbol_attr_index = vtable->symbol_def->name_attr_index;
  if (symbol_attr_index >= vtable->attribute_count ||
      symbol_attr_index >= op->attribute_count) {
    return;
  }
  const loom_attribute_t* attrs = loom_op_const_attrs(op);
  loom_symbol_ref_t ref = loom_attr_as_symbol(attrs[symbol_attr_index]);
  if (loom_symbol_ref_is_valid(ref) && ref.module_id == 0 &&
      ref.symbol_id < module->symbols.count &&
      module->symbols.entries[ref.symbol_id].defining_op == op) {
    module->symbols.entries[ref.symbol_id].defining_op = NULL;
    module->symbols.entries[ref.symbol_id].definition = NULL;
    module->symbols.entries[ref.symbol_id].kind = LOOM_SYMBOL_NONE;
  }
}

// Signature traversal stays out of the per-operation erasure hot path.
IREE_ATTRIBUTE_NOINLINE static void loom_op_drop_signature_type_uses(
    loom_module_t* module, loom_op_t* op) {
  // Bodyless declarations own the values stored in their signature operand
  // fields. Retire those carriers at the same boundary as ordinary op results
  // and region block arguments, not when an operand link happens to disappear.
  const loom_value_id_t* arguments = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    loom_module_drop_value_type_uses(module, arguments[i]);
  }
}

// Erases |op| and every operation nested in its regions. The root op must have
// unused results; nested ops are removed as part of the dead subtree and may
// still have uses from sibling ops that will be erased by the same walk.
static iree_status_t loom_op_erase_subtree(loom_module_t* module, loom_op_t* op,
                                           bool verify_results_unused) {
  if (verify_results_unused) {
    IREE_RETURN_IF_ERROR(loom_op_verify_erase_preconditions(module, op));
  }

  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    loom_region_t* region = regions[i];
    if (!region) {
      continue;
    }
    loom_block_t* block = NULL;
    loom_region_for_each_block(region, block) {
      while (block->first_op) {
        IREE_RETURN_IF_ERROR(
            loom_op_erase_subtree(module, block->first_op, false));
      }
      loom_block_drop_arg_type_uses(module, block);
    }
  }

  loom_module_drop_op_attribute_uses(module, op);
  // Remove all operand uses from the referenced values.
  loom_value_id_t* operands = loom_op_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    if (operands[i] != LOOM_VALUE_ID_INVALID) {
      IREE_RETURN_IF_ERROR(loom_value_remove_use(module, operands[i], op, i));
    }
  }
  // Drop type references carried by result values. Result definition pointers
  // remain valid for the module lifetime because operations and values share
  // the module arena. Keeping the definition preserves immutable producer
  // semantics for analyses after the operation leaves the live IR.
  loom_value_id_t* results = loom_op_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] != LOOM_VALUE_ID_INVALID) {
      loom_module_drop_value_type_uses(module, results[i]);
    }
  }
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  loom_module_unlink_symbol_defining_op(module, op, vtable);
  if (loom_op_vtable_owns_operands(vtable)) {
    loom_op_drop_signature_type_uses(module, op);
  }
  loom_block_unlink_op(module, op);
  op->flags |= LOOM_OP_FLAG_DEAD;
  return iree_ok_status();
}

iree_status_t loom_op_erase(loom_module_t* module, loom_op_t* op) {
  return loom_op_erase_subtree(module, op, true);
}

//===----------------------------------------------------------------------===//
// Region block removal
//===----------------------------------------------------------------------===//

// Root blocks already have dense ordinals. Only nested removed blocks need an
// index, shared by every closure query instead of searching subtrees per edge.
typedef struct loom_region_removal_scope_t {
  // Region whose root block table remains unchanged until validation succeeds.
  const loom_region_t* region;
  // Borrowed removal mask indexed by root block ordinal.
  const bool* remove_blocks;
  // Scratch index of removed nested blocks, sorted before closure validation.
  struct {
    // Arena-owned addresses, compared as integers rather than unrelated
    // pointers.
    uintptr_t* addresses;
    // Number of populated block addresses.
    iree_host_size_t count;
    // Allocated address capacity, grown geometrically during collection.
    iree_host_size_t capacity;
  } nested_blocks;
} loom_region_removal_scope_t;

static bool loom_region_removal_block_less(const uintptr_t* lhs,
                                           const uintptr_t* rhs) {
  return *lhs < *rhs;
}
LOOM_DEFINE_ADAPTIVE_SORT(loom_region_removal_sort_blocks, uintptr_t,
                          loom_region_removal_block_less)

static iree_status_t loom_region_removal_collect_op(
    const loom_op_t* op, loom_region_removal_scope_t* scope,
    iree_arena_allocator_t* scratch_arena) {
  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t region_index = 0; region_index < op->region_count;
       ++region_index) {
    loom_region_t* nested_region = regions[region_index];
    if (!nested_region) {
      continue;
    }
    loom_block_t* nested_block = NULL;
    loom_region_for_each_block(nested_region, nested_block) {
      if (scope->nested_blocks.count == scope->nested_blocks.capacity) {
        IREE_RETURN_IF_ERROR(
            iree_arena_grow_array(scratch_arena, scope->nested_blocks.count,
                                  iree_max(scope->nested_blocks.count + 1, 8),
                                  sizeof(*scope->nested_blocks.addresses),
                                  &scope->nested_blocks.capacity,
                                  (void**)&scope->nested_blocks.addresses));
      }
      scope->nested_blocks.addresses[scope->nested_blocks.count++] =
          (uintptr_t)nested_block;
      loom_op_t* child_op = NULL;
      loom_block_for_each_op(nested_block, child_op) {
        IREE_RETURN_IF_ERROR(
            loom_region_removal_collect_op(child_op, scope, scratch_arena));
      }
    }
  }
  return iree_ok_status();
}

static bool loom_region_remove_block_is_removed(
    const loom_region_removal_scope_t* scope, const loom_block_t* block) {
  if (!block) {
    return false;
  }
  if (block->parent_region == scope->region) {
    return scope->remove_blocks[loom_block_region_index(block)];
  }
  const uintptr_t address = (uintptr_t)block;
  iree_host_size_t begin = 0;
  iree_host_size_t end = scope->nested_blocks.count;
  while (begin < end) {
    const iree_host_size_t middle = begin + (end - begin) / 2;
    if (scope->nested_blocks.addresses[middle] < address) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  return begin < scope->nested_blocks.count &&
         scope->nested_blocks.addresses[begin] == address;
}

static bool loom_region_remove_op_is_removed(
    const loom_region_removal_scope_t* scope, const loom_op_t* op) {
  return op && loom_region_remove_block_is_removed(scope, op->parent_block);
}

static bool loom_region_remove_value_is_removed(
    const loom_module_t* module, const loom_region_removal_scope_t* scope,
    loom_value_id_t value_id) {
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return loom_region_remove_block_is_removed(scope,
                                               loom_value_def_block(value));
  }
  return loom_region_remove_op_is_removed(scope, loom_value_owner_op(value));
}

static iree_status_t loom_region_remove_verify_value_uses(
    const loom_module_t* module, const loom_region_removal_scope_t* scope,
    loom_value_id_t value_id) {
  if (value_id == LOOM_VALUE_ID_INVALID || value_id >= module->values.count) {
    return iree_ok_status();
  }

  const loom_value_t* value = loom_module_value(module, value_id);
  const loom_use_t* uses = loom_value_uses(value);
  for (uint32_t i = 0; i < value->use_count; ++i) {
    const loom_op_t* user_op = loom_use_user_op(uses[i]);
    if (!loom_region_remove_op_is_removed(scope, user_op)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "cannot remove block set: value %%%u has an operand use outside "
          "the removed blocks",
          (unsigned)value_id);
    }
  }

  loom_type_use_iterator_t type_users;
  loom_module_value_type_users(module, value_id, &type_users);
  for (loom_value_id_t user_value_id = loom_type_users_next(&type_users);
       user_value_id != LOOM_VALUE_ID_INVALID;
       user_value_id = loom_type_users_next(&type_users)) {
    if (!loom_region_remove_value_is_removed(module, scope, user_value_id)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "cannot remove block set: value %%%u has a type use outside the "
          "removed blocks",
          (unsigned)value_id);
    }
  }

  if (loom_value_has_attribute_uses(value)) {
    loom_type_use_iterator_t users;
    loom_attribute_users_begin(&module->type_uses, value_id, &users);
    for (loom_attribute_user_t user = loom_attribute_users_next(&users);
         user.op; user = loom_attribute_users_next(&users)) {
      if (!loom_region_remove_op_is_removed(scope, user.op)) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "cannot remove block set: value %%%u has an attribute use "
            "outside the removed blocks",
            (unsigned)value_id);
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_region_remove_verify_block_arg_values(
    const loom_module_t* module, const loom_region_removal_scope_t* scope,
    const loom_block_t* block) {
  for (uint16_t arg_index = 0; arg_index < block->arg_count; ++arg_index) {
    IREE_RETURN_IF_ERROR(loom_region_remove_verify_value_uses(
        module, scope, loom_block_arg_id(block, arg_index)));
  }
  return iree_ok_status();
}

static iree_status_t loom_region_remove_verify_op_values(
    const loom_module_t* module, const loom_region_removal_scope_t* scope,
    const loom_op_t* op) {
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t result_index = 0; result_index < op->result_count;
       ++result_index) {
    IREE_RETURN_IF_ERROR(loom_region_remove_verify_value_uses(
        module, scope, results[result_index]));
  }
  if (op->operand_count > 0 &&
      iree_any_bit_set(op->traits, LOOM_TRAIT_SYMBOL_DEFINE) &&
      loom_op_vtable_owns_operands(loom_op_vtable(module, op))) {
    const loom_value_id_t* arguments = loom_op_const_operands(op);
    for (uint16_t i = 0; i < op->operand_count; ++i) {
      IREE_RETURN_IF_ERROR(
          loom_region_remove_verify_value_uses(module, scope, arguments[i]));
    }
  }

  loom_region_t** nested_regions = loom_op_regions(op);
  for (uint8_t region_index = 0; region_index < op->region_count;
       ++region_index) {
    loom_region_t* nested_region = nested_regions[region_index];
    if (!nested_region) {
      continue;
    }
    loom_block_t* nested_block = NULL;
    loom_region_for_each_block(nested_region, nested_block) {
      IREE_RETURN_IF_ERROR(loom_region_remove_verify_block_arg_values(
          module, scope, nested_block));
      loom_op_t* child_op = NULL;
      loom_block_for_each_op(nested_block, child_op) {
        IREE_RETURN_IF_ERROR(
            loom_region_remove_verify_op_values(module, scope, child_op));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_region_remove_verify_removed_values(
    const loom_module_t* module, const loom_region_removal_scope_t* scope) {
  const loom_region_t* region = scope->region;
  for (uint16_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    if (!scope->remove_blocks[block_index]) {
      continue;
    }
    const loom_block_t* block = region->blocks[block_index];
    IREE_RETURN_IF_ERROR(
        loom_region_remove_verify_block_arg_values(module, scope, block));
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      IREE_RETURN_IF_ERROR(
          loom_region_remove_verify_op_values(module, scope, op));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_region_remove_verify_kept_op_successors(
    const loom_region_removal_scope_t* scope, const loom_op_t* op) {
  loom_block_t* const* successors = loom_op_const_successors(op);
  for (uint8_t successor_index = 0; successor_index < op->successor_count;
       ++successor_index) {
    if (loom_region_remove_block_is_removed(scope,
                                            successors[successor_index])) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "cannot remove block set: kept op has a successor edge to a "
          "removed block");
    }
  }

  loom_region_t** nested_regions = loom_op_regions(op);
  for (uint8_t region_index = 0; region_index < op->region_count;
       ++region_index) {
    loom_region_t* nested_region = nested_regions[region_index];
    if (!nested_region) {
      continue;
    }
    loom_block_t* nested_block = NULL;
    loom_region_for_each_block(nested_region, nested_block) {
      loom_op_t* child_op = NULL;
      loom_block_for_each_op(nested_block, child_op) {
        IREE_RETURN_IF_ERROR(
            loom_region_remove_verify_kept_op_successors(scope, child_op));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_region_remove_verify_successor_closure(
    const loom_region_removal_scope_t* scope) {
  const loom_region_t* region = scope->region;
  for (uint16_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    if (scope->remove_blocks[block_index]) {
      continue;
    }
    const loom_block_t* block = region->blocks[block_index];
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      IREE_RETURN_IF_ERROR(
          loom_region_remove_verify_kept_op_successors(scope, op));
    }
  }
  return iree_ok_status();
}

iree_status_t loom_region_remove_blocks(loom_module_t* module,
                                        loom_region_t* region,
                                        const bool* remove_blocks,
                                        uint16_t remove_block_count,
                                        iree_arena_allocator_t* scratch_arena,
                                        uint16_t* out_removed_count) {
  *out_removed_count = 0;
  if (remove_block_count != region->block_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "remove block mask has %u entries but region has %u block(s)",
        (unsigned)remove_block_count, (unsigned)region->block_count);
  }
  if (remove_block_count == 0) {
    return iree_ok_status();
  }
  if (remove_blocks[0]) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "cannot remove a region entry block");
  }

  uint16_t removed_count = 0;
  for (uint16_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    loom_block_t* block = region->blocks[block_index];
    if (!block) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "region block %u is NULL", (unsigned)block_index);
    }
    if (remove_blocks[block_index]) {
      ++removed_count;
    }
  }
  if (removed_count == 0) {
    return iree_ok_status();
  }

  loom_region_removal_scope_t scope = {0};
  scope.region = region;
  scope.remove_blocks = remove_blocks;
  for (uint16_t i = 0; i < region->block_count; ++i) {
    if (!remove_blocks[i]) {
      continue;
    }
    const loom_op_t* op = NULL;
    loom_block_for_each_op(region->blocks[i], op) {
      IREE_RETURN_IF_ERROR(
          loom_region_removal_collect_op(op, &scope, scratch_arena));
    }
  }
  loom_region_removal_sort_blocks(scope.nested_blocks.addresses,
                                  scope.nested_blocks.count);
  IREE_RETURN_IF_ERROR(loom_region_remove_verify_successor_closure(&scope));
  IREE_RETURN_IF_ERROR(
      loom_region_remove_verify_removed_values(module, &scope));

  for (uint16_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    if (!remove_blocks[block_index]) {
      continue;
    }
    loom_block_t* block = region->blocks[block_index];
    while (block->first_op) {
      IREE_RETURN_IF_ERROR(
          loom_op_erase_subtree(module, block->first_op, false));
    }
    loom_block_drop_arg_type_uses(module, block);
    block->parent_region = NULL;
    block->region_index = LOOM_BLOCK_REGION_INDEX_INVALID;
  }

  uint16_t write_index = 0;
  for (uint16_t read_index = 0; read_index < region->block_count;
       ++read_index) {
    loom_block_t* block = region->blocks[read_index];
    if (remove_blocks[read_index]) {
      continue;
    }
    block->region_index = write_index;
    region->blocks[write_index++] = block;
  }
  for (uint16_t block_index = write_index; block_index < region->block_count;
       ++block_index) {
    region->blocks[block_index] = NULL;
  }
  region->block_count = write_index;
  *out_removed_count = removed_count;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Use-def list maintenance
//===----------------------------------------------------------------------===//

void loom_module_link_symbol_defining_op(loom_module_t* module, loom_op_t* op,
                                         const loom_op_vtable_t* vtable) {
  if (!vtable || !vtable->symbol_def || !vtable->attr_descriptors) {
    return;
  }
  uint8_t symbol_attr_index = vtable->symbol_def->name_attr_index;
  if (symbol_attr_index >= vtable->attribute_count ||
      symbol_attr_index >= op->attribute_count) {
    return;
  }
  loom_attribute_t* attrs = loom_op_attrs(op);
  loom_symbol_ref_t ref = loom_attr_as_symbol(attrs[symbol_attr_index]);
  if (loom_symbol_ref_is_valid(ref) && ref.module_id == 0 &&
      ref.symbol_id < module->symbols.count) {
    loom_symbol_t* symbol = &module->symbols.entries[ref.symbol_id];
    if (symbol->defining_op && symbol->defining_op != op) {
      return;
    }
    symbol->defining_op = op;
    symbol->definition = vtable->symbol_def;
    symbol->kind = vtable->symbol_def->bytecode_kind;
    const uint8_t visibility_attr_index_plus_one =
        vtable->symbol_def->visibility_attr_index_plus_one;
    if (visibility_attr_index_plus_one) {
      const uint8_t visibility_attr_index = visibility_attr_index_plus_one - 1;
      symbol->flags &= (loom_symbol_flags_t)~LOOM_SYMBOL_FLAG_PUBLIC;
      if (visibility_attr_index < op->attribute_count &&
          loom_attr_as_enum(attrs[visibility_attr_index]) != 0) {
        symbol->flags |= LOOM_SYMBOL_FLAG_PUBLIC;
      }
    }
    symbol->flags &= (loom_symbol_flags_t)~LOOM_SYMBOL_FLAG_RETAIN;
    const uint8_t retain_attr_index_plus_one =
        vtable->symbol_def->retain_attr_index_plus_one;
    if (retain_attr_index_plus_one) {
      const uint8_t retain_attr_index = retain_attr_index_plus_one - 1;
      if (retain_attr_index < op->attribute_count &&
          loom_attr_as_enum(attrs[retain_attr_index]) != 0) {
        symbol->flags |= LOOM_SYMBOL_FLAG_RETAIN;
      }
    }
  }
}

iree_status_t loom_builder_finalize_op(loom_builder_t* builder, loom_op_t* op) {
  // Verify reserved values were fully consumed.
  if (builder->reserved_value_count > 0) {
    if (builder->reserved_value_next != builder->reserved_value_count) {
      iree_host_size_t consumed = builder->reserved_value_next;
      iree_host_size_t reserved = builder->reserved_value_count;
      builder->reserved_value_ids = NULL;
      builder->reserved_value_count = 0;
      builder->reserved_value_next = 0;
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "reserved %" PRIhsz
                              " value(s) but op consumed %" PRIhsz,
                              reserved, consumed);
    }
    builder->reserved_value_ids = NULL;
    builder->reserved_value_count = 0;
    builder->reserved_value_next = 0;
  }

  IREE_RETURN_IF_ERROR(loom_op_record_operand_uses(builder->module, op));
  // Set the def pointer on each result value.
  loom_value_id_t* results = loom_op_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] != LOOM_VALUE_ID_INVALID) {
      loom_module_value(builder->module, results[i])->def =
          loom_value_def_make_op(op, i);
    }
  }
  // Result type uses are installed when values are defined or their types
  // change. Finalization only assigns the operation definition site.
  IREE_RETURN_IF_ERROR(
      loom_module_refresh_op_attribute_uses(builder->module, op));
  // Wire the symbol table entry for symbol-defining ops so that
  // loom_func_like_cast can find the defining op without a scan.
  const loom_op_vtable_t* vtable = loom_op_vtable(builder->module, op);
  loom_op_refresh_effective_traits(builder->module, op);
  if (vtable && iree_any_bit_set(vtable->traits, LOOM_TRAIT_SYMBOL_DEFINE)) {
    loom_module_link_symbol_defining_op(builder->module, op, vtable);
  }
  loom_module_record_op_summaries(builder->module, op);
  // Notify the rewriter (or other listener) that the op's direct fields are
  // fully wired. Callers may populate nested regions after their parent
  // builder returns.
  if (builder->on_op_finalized.fn) {
    IREE_RETURN_IF_ERROR(
        builder->on_op_finalized.fn(builder->on_op_finalized.user_data, op));
  }
  return iree_ok_status();
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
