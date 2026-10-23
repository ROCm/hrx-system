// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/mutation.h"

#include <stdint.h>
#include <string.h>

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/util/adaptive_sort.h"

//===----------------------------------------------------------------------===//
// Deadness and erasure
//===----------------------------------------------------------------------===//

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
  if (iree_any_bit_set(traits, LOOM_TRAIT_HINT | LOOM_TRAIT_SYMBOL_DEFINE)) {
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

// Erases a subtree whose enclosing removal has established result-use closure.
// Sibling operations may still use its results until that removal completes.
static void loom_op_erase_subtree(loom_module_t* module, loom_op_t* op) {
  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    loom_region_t* region = regions[i];
    if (!region) {
      continue;
    }
    loom_block_t* block = NULL;
    loom_region_for_each_block(region, block) {
      while (block->first_op) {
        loom_op_erase_subtree(module, block->first_op);
      }
      loom_block_drop_arg_type_uses(module, block);
    }
  }

  loom_module_drop_op_attribute_uses(module, op);
  // Remove all operand uses from the referenced values.
  loom_value_id_t* operands = loom_op_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    if (operands[i] != LOOM_VALUE_ID_INVALID) {
      loom_value_remove_use(module, operands[i], op, i);
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
}

iree_status_t loom_op_erase(loom_module_t* module, loom_op_t* op) {
  IREE_RETURN_IF_ERROR(loom_op_verify_erase_preconditions(module, op));
  loom_op_erase_subtree(module, op);
  return iree_ok_status();
}

void loom_op_erase_closed_set(loom_module_t* module, loom_op_t* const* ops,
                              iree_host_size_t count) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    loom_op_erase_subtree(module, ops[i]);
  }
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
      loom_op_erase_subtree(module, block->first_op);
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
// Value replacement
//===----------------------------------------------------------------------===//

// Ensures a value's use list has capacity for at least |additional| more
// entries. Used by RAUW to pre-allocate before bulk transfer.
static iree_status_t loom_value_ensure_use_capacity(loom_module_t* module,
                                                    loom_value_t* value,
                                                    uint32_t additional) {
  if (additional > LOOM_VALUE_MAX_USE_COUNT - value->use_count) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "value use count exceeds maximum (%u)",
                            (unsigned)LOOM_VALUE_MAX_USE_COUNT);
  }
  uint32_t needed = value->use_count + additional;
  if (!loom_value_has_overflow_uses(value)) {
    if (needed <= LOOM_VALUE_INLINE_USE_COUNT) {
      return iree_ok_status();
    }
    // Transition to overflow with enough capacity.
    uint32_t capacity = LOOM_USE_INITIAL_OVERFLOW_CAPACITY;
    while (capacity < needed) {
      capacity *= 2;
    }
    loom_use_t* overflow = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &module->arena, capacity, sizeof(loom_use_t), (void**)&overflow));
    for (uint32_t i = 0; i < value->use_count; ++i) {
      overflow[i] = value->inline_uses[i];
    }
    const uint32_t ownership_use_count = loom_value_ownership_use_count(value);
    const uint32_t consuming_use_count = loom_value_consuming_use_count(value);
    value->overflow_uses = overflow;
    value->overflow_capacity = capacity;
    value->overflow_ownership_use_count = ownership_use_count;
    value->overflow_consuming_use_count = consuming_use_count;
    value->flags |= LOOM_VALUE_FLAG_OVERFLOW_USES;
    return iree_ok_status();
  }
  if (needed <= value->overflow_capacity) {
    return iree_ok_status();
  }
  // Grow overflow (floor to initial capacity for safety).
  uint32_t new_capacity =
      iree_max(value->overflow_capacity, LOOM_USE_INITIAL_OVERFLOW_CAPACITY);
  while (new_capacity < needed) {
    new_capacity *= 2;
  }
  loom_use_t* new_overflow = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &module->arena, new_capacity, sizeof(loom_use_t), (void**)&new_overflow));
  memcpy(new_overflow, value->overflow_uses,
         (iree_host_size_t)value->use_count * sizeof(loom_use_t));
  value->overflow_uses = new_overflow;
  value->overflow_capacity = new_capacity;
  return iree_ok_status();
}

iree_status_t loom_value_replace_all_uses_with(loom_module_t* module,
                                               loom_value_id_t old_id,
                                               loom_value_id_t new_id) {
  if (old_id == new_id) {
    return iree_ok_status();
  }
  loom_value_t* old_value = loom_module_value(module, old_id);
  uint32_t old_use_count = old_value->use_count;

  loom_value_t* new_value = loom_module_value(module, new_id);
  IREE_RETURN_IF_ERROR(
      loom_value_ensure_use_capacity(module, new_value, old_use_count));

  if (loom_module_value_has_type_uses(module, old_id) ||
      loom_value_has_attribute_uses(old_value)) {
    loom_value_replacement_t replacement;
    loom_value_replacement_initialize(module, old_id, new_id, &replacement);
    iree_status_t status = loom_value_replacement_apply_types(&replacement);
    while (loom_value_has_attribute_uses(old_value) &&
           iree_status_is_ok(status)) {
      // Mutation invalidates the cursor. Successful replacement removes this
      // owner's old membership, so restarting selects an unprocessed owner.
      loom_type_use_iterator_t users;
      loom_attribute_users_begin(&module->type_uses, old_id, &users);
      const loom_attribute_user_t use = loom_attribute_users_next(&users);
      status = loom_value_replacement_apply_attribute(&replacement, use.op,
                                                      use.attribute_index);
      if (iree_status_is_ok(status)) {
        loom_trait_flags_t old_traits = use.op->traits;
        loom_op_refresh_effective_traits(module, use.op);
        loom_module_update_op_direct_summaries(module, use.op, old_traits,
                                               use.op->traits);
      }
    }
    loom_value_replacement_deinitialize(&replacement);
    IREE_RETURN_IF_ERROR(status);
  }
  if (old_use_count == 0) {
    return iree_ok_status();
  }

  // Patch every user op's operand slot.
  const loom_use_t* old_uses = loom_value_uses(old_value);
  for (uint32_t i = 0; i < old_use_count; ++i) {
    loom_op_t* user_op = loom_use_user_op(old_uses[i]);
    uint16_t operand_index = loom_use_operand_index(old_uses[i]);
    loom_op_operands(user_op)[operand_index] = new_id;
  }

  // Bulk-transfer use entries from old to new.
  // Append old's entries to new's list. The old_uses pointer (captured
  // above) is still valid: ensure_use_capacity only touches new_value,
  // and old_id != new_id is guarded at entry.
  loom_use_t* new_uses = loom_value_uses_mutable(new_value);
  uint32_t new_use_start = new_value->use_count;
  for (uint32_t i = 0; i < old_use_count; ++i) {
    uint32_t new_use_index = new_use_start + i;
    new_uses[new_use_index] = old_uses[i];
    loom_op_t* user_op = loom_use_user_op(old_uses[i]);
    uint16_t operand_index = loom_use_operand_index(old_uses[i]);
    loom_op_operand_use_indices(user_op)[operand_index] = new_use_index;
  }
  if (loom_value_has_overflow_uses(new_value)) {
    new_value->overflow_ownership_use_count +=
        loom_value_ownership_use_count(old_value);
    new_value->overflow_consuming_use_count +=
        loom_value_consuming_use_count(old_value);
  }
  new_value->use_count += old_use_count;

  // Clear old value's use list.
  old_value->use_count = 0;
  old_value->flags &= ~LOOM_VALUE_FLAG_OVERFLOW_USES;
  return iree_ok_status();
}

iree_status_t loom_value_replace_all_uses_except(loom_module_t* module,
                                                 loom_value_id_t old_id,
                                                 loom_value_id_t new_id,
                                                 const loom_op_t* except_op) {
  if (old_id == new_id) {
    return iree_ok_status();
  }
  loom_value_t* old_value = loom_module_value(module, old_id);
  if (old_value->use_count == 0) {
    return iree_ok_status();
  }

  // Count how many uses will be transferred vs kept.
  const loom_use_t* old_uses = loom_value_uses(old_value);
  uint32_t transfer_count = 0;
  for (uint32_t i = 0; i < old_value->use_count; ++i) {
    if (loom_use_user_op(old_uses[i]) != except_op) {
      ++transfer_count;
    }
  }
  if (transfer_count == 0) {
    return iree_ok_status();
  }

  // Ensure new has capacity.
  loom_value_t* new_value = loom_module_value(module, new_id);
  IREE_RETURN_IF_ERROR(
      loom_value_ensure_use_capacity(module, new_value, transfer_count));

  // Patch operand slots and transfer use entries.
  // Walk old's list backwards so swap-removal doesn't skip entries.
  loom_use_t* old_uses_mutable = loom_value_uses_mutable(old_value);
  loom_use_t* new_uses = loom_value_uses_mutable(new_value);
  for (uint32_t i = old_value->use_count; i-- > 0;) {
    if (loom_use_user_op(old_uses_mutable[i]) == except_op) {
      continue;
    }
    // Patch the operand slot.
    loom_op_t* user_op = loom_use_user_op(old_uses_mutable[i]);
    uint16_t operand_index = loom_use_operand_index(old_uses_mutable[i]);
    loom_op_operands(user_op)[operand_index] = new_id;
    // Add to new's list.
    const loom_use_flags_t flags = loom_use_flags(old_uses_mutable[i]);
    loom_value_update_overflow_use_counts(new_value, 0, flags);
    loom_value_update_overflow_use_counts(old_value, flags, 0);
    uint32_t new_use_index = new_value->use_count;
    new_uses[new_use_index] = old_uses_mutable[i];
    loom_op_operand_use_indices(user_op)[operand_index] = new_use_index;
    ++new_value->use_count;
    // Remove from old's list (swap with last).
    uint32_t last_index = old_value->use_count - 1;
    if (i != last_index) {
      loom_use_t moved_use = old_uses_mutable[last_index];
      old_uses_mutable[i] = moved_use;
      loom_op_t* moved_user_op = loom_use_user_op(moved_use);
      uint16_t moved_operand_index = loom_use_operand_index(moved_use);
      loom_op_operand_use_indices(moved_user_op)[moved_operand_index] = i;
    }
    --old_value->use_count;
  }

  // If old is now empty and was overflow, clear the flag.
  if (old_value->use_count == 0) {
    old_value->flags &= ~LOOM_VALUE_FLAG_OVERFLOW_USES;
  }
  return iree_ok_status();
}

iree_status_t loom_value_replace_uses_if(loom_module_t* module,
                                         loom_value_id_t old_id,
                                         loom_value_id_t new_id,
                                         loom_use_predicate_fn predicate,
                                         void* user_data) {
  if (old_id == new_id) {
    return iree_ok_status();
  }
  loom_value_t* old_value = loom_module_value(module, old_id);
  if (old_value->use_count == 0) {
    return iree_ok_status();
  }

  // Count how many uses will be transferred.
  const loom_use_t* old_uses = loom_value_uses(old_value);
  uint32_t transfer_count = 0;
  for (uint32_t i = 0; i < old_value->use_count; ++i) {
    if (predicate(loom_use_user_op(old_uses[i]), user_data)) {
      ++transfer_count;
    }
  }
  if (transfer_count == 0) {
    return iree_ok_status();
  }

  // Ensure new has capacity.
  loom_value_t* new_value = loom_module_value(module, new_id);
  IREE_RETURN_IF_ERROR(
      loom_value_ensure_use_capacity(module, new_value, transfer_count));

  // Patch and transfer (walk backwards for safe swap-removal).
  loom_use_t* old_uses_mutable = loom_value_uses_mutable(old_value);
  loom_use_t* new_uses = loom_value_uses_mutable(new_value);
  for (uint32_t i = old_value->use_count; i-- > 0;) {
    if (!predicate(loom_use_user_op(old_uses_mutable[i]), user_data)) {
      continue;
    }
    loom_op_t* user_op = loom_use_user_op(old_uses_mutable[i]);
    uint16_t operand_index = loom_use_operand_index(old_uses_mutable[i]);
    loom_op_operands(user_op)[operand_index] = new_id;
    const loom_use_flags_t flags = loom_use_flags(old_uses_mutable[i]);
    loom_value_update_overflow_use_counts(new_value, 0, flags);
    loom_value_update_overflow_use_counts(old_value, flags, 0);
    uint32_t new_use_index = new_value->use_count;
    new_uses[new_use_index] = old_uses_mutable[i];
    loom_op_operand_use_indices(user_op)[operand_index] = new_use_index;
    ++new_value->use_count;
    uint32_t last_index = old_value->use_count - 1;
    if (i != last_index) {
      loom_use_t moved_use = old_uses_mutable[last_index];
      old_uses_mutable[i] = moved_use;
      loom_op_t* moved_user_op = loom_use_user_op(moved_use);
      uint16_t moved_operand_index = loom_use_operand_index(moved_use);
      loom_op_operand_use_indices(moved_user_op)[moved_operand_index] = i;
    }
    --old_value->use_count;
  }
  if (old_value->use_count == 0) {
    old_value->flags &= ~LOOM_VALUE_FLAG_OVERFLOW_USES;
  }
  return iree_ok_status();
}
