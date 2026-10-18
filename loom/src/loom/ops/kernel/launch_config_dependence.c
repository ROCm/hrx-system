// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/kernel/launch_config_dependence.h"

#include <string.h>

#include "loom/analysis/value_relation.h"
#include "loom/ir/module.h"

iree_status_t loom_kernel_launch_config_dependence_initialize(
    const loom_module_t* module, iree_arena_allocator_t* arena,
    loom_kernel_launch_config_dependence_t* out_dependence) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_dependence);
  *out_dependence = (loom_kernel_launch_config_dependence_t){
      .module = module,
      .arena = arena,
      .dependent_word_count = (module->values.count + 63) / 64,
  };
  if (out_dependence->dependent_word_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, out_dependence->dependent_word_count,
                                sizeof(*out_dependence->dependent_bits),
                                (void**)&out_dependence->dependent_bits));
  memset(out_dependence->dependent_bits, 0,
         out_dependence->dependent_word_count *
             sizeof(*out_dependence->dependent_bits));
  return iree_ok_status();
}

bool loom_kernel_launch_config_dependence_contains(
    const loom_kernel_launch_config_dependence_t* dependence,
    loom_value_id_t value_id) {
  return value_id < dependence->module->values.count &&
         (dependence->dependent_bits[value_id / 64] &
          (UINT64_C(1) << (value_id % 64))) != 0;
}

iree_status_t loom_kernel_launch_config_dependence_mark(
    loom_kernel_launch_config_dependence_t* dependence,
    loom_value_id_t value_id) {
  if (value_id == LOOM_VALUE_ID_INVALID ||
      value_id >= dependence->module->values.count ||
      loom_kernel_launch_config_dependence_contains(dependence, value_id)) {
    return iree_ok_status();
  }
  dependence->dependent_bits[value_id / 64] |= UINT64_C(1) << (value_id % 64);
  if (dependence->dependent_value_count ==
      dependence->dependent_value_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        dependence->arena, dependence->dependent_value_count,
        dependence->dependent_value_count + 1,
        sizeof(*dependence->dependent_values),
        &dependence->dependent_value_capacity,
        (void**)&dependence->dependent_values));
  }
  dependence->dependent_values[dependence->dependent_value_count++] = value_id;
  return iree_ok_status();
}

iree_status_t loom_kernel_launch_config_dependence_mark_results(
    loom_kernel_launch_config_dependence_t* dependence, const loom_op_t* op) {
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_kernel_launch_config_dependence_mark(dependence, results[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_kernel_launch_config_dependence_propagate_user(
    loom_kernel_launch_config_dependence_t* dependence, loom_value_id_t source,
    const loom_op_t* user, uint16_t operand_index) {
  // Preserve precise data transport across identity/select operations and
  // structured control flow. Tied-result and reference-source relations only
  // describe storage identity, not exhaustive data dependence, so those uses
  // retain the conservative all-results behavior below.
  const loom_value_relation_mask_t relation_mask =
      LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_FACT_IDENTITY) |
      LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_VALUE_ALIAS) |
      LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_SELECT_PAYLOAD) |
      LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_ELEMENTWISE) |
      LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_CFG_ARGUMENT) |
      LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_LOOP_CARRIED) |
      LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_LOOP_BYPASS) |
      LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_REGION_RESULT);
  bool has_operand_relation = false;
  loom_value_relation_iterator_t relations;
  loom_value_relation_iterator_initialize(dependence->module, user,
                                          relation_mask, &relations);
  loom_value_relation_t relation;
  while (loom_value_relation_iterator_next(&relations, &relation)) {
    if (relation.source_value_id != source ||
        relation.source_operand_index != operand_index) {
      continue;
    }
    has_operand_relation = true;
    IREE_RETURN_IF_ERROR(loom_kernel_launch_config_dependence_mark(
        dependence, relation.destination_value_id));
  }

  // An operand use without an exhaustive relation may contribute to every
  // direct result. A resultless control terminator instead controls the values
  // produced by its region owner, such as scf.condition controlling scf.while
  // results or cfg.cond_br controlling an enclosing region branch.
  if (!has_operand_relation) {
    IREE_RETURN_IF_ERROR(
        loom_kernel_launch_config_dependence_mark_results(dependence, user));
    if (user->result_count == 0 && user->parent_op != NULL &&
        loom_op_operand_has_role(dependence->module, user, operand_index,
                                 LOOM_OPERAND_ROLE_CONTROL_CONDITION)) {
      IREE_RETURN_IF_ERROR(loom_kernel_launch_config_dependence_mark_results(
          dependence, user->parent_op));
    }
  }
  return iree_ok_status();
}

iree_status_t loom_kernel_launch_config_dependence_propagate(
    loom_kernel_launch_config_dependence_t* dependence) {
  for (iree_host_size_t cursor = 0; cursor < dependence->dependent_value_count;
       ++cursor) {
    const loom_value_id_t value_id = dependence->dependent_values[cursor];
    const loom_value_t* value = loom_module_value(dependence->module, value_id);
    const loom_use_t* uses = loom_value_uses(value);
    for (uint32_t i = 0; i < value->use_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_kernel_launch_config_dependence_propagate_user(
          dependence, value_id, loom_use_user_op(uses[i]),
          loom_use_operand_index(uses[i])));
    }

    loom_type_use_iterator_t type_users;
    loom_module_value_type_users(dependence->module, value_id, &type_users);
    for (loom_value_id_t carrier = loom_type_users_next(&type_users);
         carrier != LOOM_VALUE_ID_INVALID;
         carrier = loom_type_users_next(&type_users)) {
      IREE_RETURN_IF_ERROR(
          loom_kernel_launch_config_dependence_mark(dependence, carrier));
    }

    loom_type_use_iterator_t attribute_users;
    loom_attribute_users_begin(&dependence->module->type_uses, value_id,
                               &attribute_users);
    for (loom_attribute_user_t user =
             loom_attribute_users_next(&attribute_users);
         user.op != NULL; user = loom_attribute_users_next(&attribute_users)) {
      IREE_RETURN_IF_ERROR(loom_kernel_launch_config_dependence_mark_results(
          dependence, user.op));
    }
  }
  return iree_ok_status();
}
