// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/fact_select_dependencies.h"

#include <string.h>

#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/util/fact_table.h"

#define LOOM_VALUE_FACT_SELECT_DEPENDENCY_PAGE_SHIFT 6u
#define LOOM_VALUE_FACT_SELECT_DEPENDENCY_PAGE_SIZE \
  (1u << LOOM_VALUE_FACT_SELECT_DEPENDENCY_PAGE_SHIFT)

struct loom_value_fact_select_dependency_page_t {
  // Canonical set roots for one aligned range of SSA IDs; zero is empty.
  loom_value_set_id_t roots[LOOM_VALUE_FACT_SELECT_DEPENDENCY_PAGE_SIZE];
};

static loom_value_set_id_t* loom_value_fact_table_lookup_select_dependency(
    const loom_value_fact_table_t* table, loom_value_id_t value_id) {
  const iree_host_size_t page_index =
      value_id >> LOOM_VALUE_FACT_SELECT_DEPENDENCY_PAGE_SHIFT;
  if (!table || page_index >= table->select_dependencies.page_capacity) {
    return NULL;
  }
  loom_value_fact_select_dependency_page_t* page =
      table->select_dependencies.pages[page_index];
  return page ? &page->roots[value_id &
                             (LOOM_VALUE_FACT_SELECT_DEPENDENCY_PAGE_SIZE - 1)]
              : NULL;
}

loom_value_set_id_t loom_value_fact_table_select_dependencies_begin(
    const loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_set_cursor_t* out_cursor) {
  const loom_value_set_id_t root =
      loom_value_fact_table_select_dependency_root(table, value_id);
  loom_value_set_cursor_begin(table ? table->select_dependencies.index : NULL,
                              root, out_cursor);
  return root;
}

loom_value_set_id_t loom_value_fact_table_select_dependency_root(
    const loom_value_fact_table_t* table, loom_value_id_t value_id) {
  loom_value_fact_table_record_read(table, value_id);
  const loom_value_set_id_t* entry =
      loom_value_fact_table_lookup_select_dependency(table, value_id);
  return entry ? *entry : 0;
}

static iree_status_t loom_value_fact_table_ensure_select_dependency_index(
    loom_value_fact_table_t* table) {
  if (table->select_dependencies.index) {
    return iree_ok_status();
  }
  return loom_value_set_index_allocate(table->transient_arena,
                                       &table->select_dependencies.index);
}

static iree_status_t loom_value_fact_table_ensure_select_dependency_page(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_set_id_t** out_root) {
  const iree_host_size_t page_index =
      value_id >> LOOM_VALUE_FACT_SELECT_DEPENDENCY_PAGE_SHIFT;
  const iree_host_size_t old_capacity =
      table->select_dependencies.page_capacity;
  if (page_index >= old_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        table->transient_arena, old_capacity, page_index + 1,
        sizeof(*table->select_dependencies.pages),
        &table->select_dependencies.page_capacity,
        (void**)&table->select_dependencies.pages));
    memset(table->select_dependencies.pages + old_capacity, 0,
           (table->select_dependencies.page_capacity - old_capacity) *
               sizeof(*table->select_dependencies.pages));
  }
  loom_value_fact_select_dependency_page_t* page =
      table->select_dependencies.pages[page_index];
  if (!page) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate(table->transient_arena,
                                             sizeof(*page), (void**)&page));
    memset(page, 0, sizeof(*page));
    table->select_dependencies.pages[page_index] = page;
  }
  *out_root = &page->roots[value_id &
                           (LOOM_VALUE_FACT_SELECT_DEPENDENCY_PAGE_SIZE - 1)];
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_union_select_dependency(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_set_id_t* inout_root) {
  const loom_value_set_id_t dependency =
      loom_value_fact_table_select_dependency_root(table, value_id);
  if (!dependency) {
    return iree_ok_status();
  }
  return loom_value_set_index_union(table->select_dependencies.index,
                                    *inout_root, dependency, inout_root);
}

static iree_status_t loom_value_fact_table_add_select_condition(
    loom_value_fact_table_t* table, loom_value_id_t condition,
    loom_value_set_id_t* inout_root) {
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_ensure_select_dependency_index(table));
  return loom_value_set_index_add(table->select_dependencies.index, *inout_root,
                                  condition, inout_root);
}

static iree_status_t loom_value_fact_table_set_select_dependency_root(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_set_id_t root, bool* inout_changed) {
  loom_value_set_id_t* entry =
      loom_value_fact_table_lookup_select_dependency(table, value_id);
  if ((entry ? *entry : 0) == root) {
    return iree_ok_status();
  }
  if (!entry) {
    IREE_RETURN_IF_ERROR(loom_value_fact_table_ensure_select_dependency_page(
        table, value_id, &entry));
  }
  *entry = root;
  if (inout_changed) {
    *inout_changed = true;
  }
  return iree_ok_status();
}

iree_status_t loom_value_fact_table_propagate_select_dependencies(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_op_t* op, const loom_op_vtable_t* vtable, bool* inout_changed) {
  if (op->result_count == 0) {
    return iree_ok_status();
  }

  const bool may_add_select_condition =
      vtable && iree_any_bit_set(vtable->operand_role_mask,
                                 LOOM_OPERAND_ROLE_MASK_SELECT_CONDITION);
  if (!table->select_dependencies.index && !may_add_select_condition) {
    return iree_ok_status();
  }

  loom_value_set_id_t root = 0;
  const loom_value_id_t* operands = loom_op_const_operands(op);
  if (table->select_dependencies.index) {
    for (uint16_t i = 0; i < op->operand_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_union_select_dependency(
          table, operands[i], &root));
    }
  }

  if (may_add_select_condition) {
    for (uint16_t i = 0; i < op->operand_count; ++i) {
      if (loom_op_operand_role_at(vtable, op, i) !=
          LOOM_OPERAND_ROLE_SELECT_CONDITION) {
        continue;
      }
      const loom_value_id_t condition = operands[i];
      if (condition == LOOM_VALUE_ID_INVALID ||
          condition >= module->values.count) {
        continue;
      }
      const loom_type_t type = loom_module_value_type(module, condition);
      if (loom_type_is_scalar(type) &&
          loom_type_element_type(type) == LOOM_SCALAR_TYPE_I1) {
        IREE_RETURN_IF_ERROR(loom_value_fact_table_add_select_condition(
            table, condition, &root));
      }
    }
  }

  if (op->attribute_count > 0 && loom_traits_are_fact_identity(op->traits)) {
    const loom_attribute_t* attributes = loom_op_const_attrs(op);
    for (uint8_t i = 0; i < op->attribute_count; ++i) {
      if (attributes[i].kind != LOOM_ATTR_PREDICATE_LIST) {
        continue;
      }
      for (uint16_t j = 0; j < attributes[i].count; ++j) {
        const loom_predicate_t* predicate = &attributes[i].predicate_list[j];
        for (uint8_t k = 0; k < predicate->arg_count; ++k) {
          if (predicate->arg_tags[k] == LOOM_PRED_ARG_VALUE) {
            IREE_RETURN_IF_ERROR(loom_value_fact_table_union_select_dependency(
                table, (loom_value_id_t)predicate->args[k], &root));
          }
        }
      }
    }
  }

  if (!root && !table->select_dependencies.pages) {
    return iree_ok_status();
  }
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    const loom_value_id_t result = results[i];
    if (result == LOOM_VALUE_ID_INVALID || result >= module->values.count) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_value_fact_table_set_select_dependency_root(
        table, result, root, inout_changed));
  }
  return iree_ok_status();
}

iree_status_t loom_value_fact_table_clone_select_dependencies(
    loom_value_fact_table_t* target, const loom_value_fact_table_t* source,
    loom_value_id_t value_id) {
  loom_value_set_cursor_t cursor;
  loom_value_fact_table_select_dependencies_begin(source, value_id, &cursor);
  loom_value_set_id_t root = 0;
  for (loom_value_id_t condition = loom_value_set_cursor_next(&cursor);
       condition != LOOM_VALUE_ID_INVALID;
       condition = loom_value_set_cursor_next(&cursor)) {
    IREE_RETURN_IF_ERROR(
        loom_value_fact_table_add_select_condition(target, condition, &root));
  }
  return loom_value_fact_table_set_select_dependency_root(
      target, value_id, root, /*inout_changed=*/NULL);
}
