// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/availability.h"

#include "loom/ir/encoding.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"

iree_status_t loom_availability_analysis_initialize(
    const loom_module_t* module, iree_arena_allocator_t* arena,
    loom_availability_analysis_t* out_analysis) {
  out_analysis->module = module;
  out_analysis->arena = arena;
  IREE_RETURN_IF_ERROR(
      loom_dominance_info_initialize(module, arena, &out_analysis->dominance));
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Subtree ownership
//===----------------------------------------------------------------------===//

static bool loom_availability_op_is_nested_under(const loom_op_t* root,
                                                 const loom_op_t* op) {
  if (!root || !op) return false;
  for (const loom_op_t* current = op; current; current = current->parent_op) {
    if (current == root) return true;
  }
  return false;
}

static bool loom_availability_value_id_is_valid(const loom_module_t* module,
                                                loom_value_id_t value_id) {
  return module && value_id != LOOM_VALUE_ID_INVALID &&
         value_id < module->values.count;
}

static bool loom_availability_query_is_valid(
    const loom_availability_analysis_t* analysis, const loom_op_t* before_op) {
  return analysis && analysis->module && before_op;
}

static const loom_op_t* loom_availability_region_owner_op(
    const loom_region_t* region) {
  if (!region) {
    return NULL;
  }
  const loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    const loom_op_t* first_op = block->first_op;
    if (first_op) {
      return first_op->parent_op;
    }
  }
  return NULL;
}

static bool loom_availability_region_contains_block_slow(
    const loom_region_t* region, const loom_block_t* target) {
  if (!region || !target) {
    return false;
  }
  for (uint16_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(region, block_index);
    if (block == target) {
      return true;
    }
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      loom_region_t** regions = loom_op_regions(op);
      for (uint8_t i = 0; i < op->region_count; ++i) {
        if (loom_availability_region_contains_block_slow(regions[i], target)) {
          return true;
        }
      }
    }
  }
  return false;
}

static bool loom_availability_block_is_nested_under(const loom_op_t* root,
                                                    const loom_block_t* block) {
  if (!root || !block) {
    return false;
  }
  const loom_op_t* owner_op =
      loom_availability_region_owner_op(block->parent_region);
  if (owner_op) {
    return loom_availability_op_is_nested_under(root, owner_op);
  }

  loom_region_t** regions = loom_op_regions(root);
  for (uint8_t i = 0; i < root->region_count; ++i) {
    if (loom_availability_region_contains_block_slow(regions[i], block)) {
      return true;
    }
  }
  return false;
}

static bool loom_availability_value_moves_with_subtree(
    const loom_module_t* module, const loom_op_t* moving_root_op,
    loom_value_id_t value_id) {
  if (!moving_root_op ||
      !loom_availability_value_id_is_valid(module, value_id)) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return loom_availability_block_is_nested_under(moving_root_op,
                                                   loom_value_def_block(value));
  }
  return loom_availability_op_is_nested_under(moving_root_op,
                                              loom_value_def_op(value));
}

bool loom_availability_value_is_available_before_op(
    const loom_availability_analysis_t* analysis,
    const loom_op_t* moving_root_op, const loom_op_t* before_op,
    loom_value_id_t value_id) {
  if (!loom_availability_query_is_valid(analysis, before_op)) {
    return false;
  }
  if (loom_availability_value_moves_with_subtree(analysis->module,
                                                 moving_root_op, value_id)) {
    return true;
  }
  return loom_value_is_available_before_op(&analysis->dominance, value_id,
                                           before_op);
}

//===----------------------------------------------------------------------===//
// Type availability
//===----------------------------------------------------------------------===//

typedef struct loom_availability_type_ref_query_t {
  // Availability analysis answering value visibility.
  const loom_availability_analysis_t* analysis;
  // Moving subtree whose internal values may be referenced.
  const loom_op_t* moving_root_op;
  // Insertion point before which references must be available.
  const loom_op_t* before_op;
  // Cleared when any type-embedded SSA reference is unavailable.
  bool available;
} loom_availability_type_ref_query_t;

static iree_status_t loom_availability_check_type_ref(loom_value_id_t value_id,
                                                      void* user_data) {
  loom_availability_type_ref_query_t* query =
      (loom_availability_type_ref_query_t*)user_data;
  if (!loom_availability_value_is_available_before_op(
          query->analysis, query->moving_root_op, query->before_op, value_id)) {
    query->available = false;
  }
  return iree_ok_status();
}

iree_status_t loom_availability_type_is_available_before_op(
    const loom_availability_analysis_t* analysis,
    const loom_op_t* moving_root_op, const loom_op_t* before_op,
    loom_type_t type, bool* out_available) {
  *out_available = false;
  if (!loom_availability_query_is_valid(analysis, before_op)) {
    return iree_ok_status();
  }
  loom_availability_type_ref_query_t query = {
      .analysis = analysis,
      .moving_root_op = moving_root_op,
      .before_op = before_op,
      .available = true,
  };
  IREE_RETURN_IF_ERROR(loom_type_walk_value_refs(
      type, loom_availability_check_type_ref, &query));
  *out_available = query.available;
  return iree_ok_status();
}

iree_status_t loom_availability_value_type_is_available_before_op(
    const loom_availability_analysis_t* analysis,
    const loom_op_t* moving_root_op, const loom_op_t* before_op,
    loom_value_id_t value_id, bool* out_available) {
  *out_available = false;
  if (!analysis ||
      !loom_availability_value_id_is_valid(analysis->module, value_id)) {
    return iree_ok_status();
  }
  return loom_availability_type_is_available_before_op(
      analysis, moving_root_op, before_op,
      loom_module_value_type(analysis->module, value_id), out_available);
}

//===----------------------------------------------------------------------===//
// Attribute availability
//===----------------------------------------------------------------------===//

static bool loom_availability_predicate_is_available_before_op(
    const loom_availability_analysis_t* analysis,
    const loom_op_t* moving_root_op, const loom_op_t* before_op,
    const loom_predicate_t* predicate) {
  for (uint8_t i = 0; i < IREE_ARRAYSIZE(predicate->arg_tags); ++i) {
    if (predicate->arg_tags[i] != LOOM_PRED_ARG_VALUE) continue;
    if (predicate->args[i] < 0) return false;
    loom_value_id_t value_id = (loom_value_id_t)predicate->args[i];
    if (!loom_availability_value_id_is_valid(analysis->module, value_id)) {
      return false;
    }
    if (!loom_availability_value_is_available_before_op(
            analysis, moving_root_op, before_op, value_id)) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_availability_attr_is_available_before_op_impl(
    const loom_availability_analysis_t* analysis,
    const loom_op_t* moving_root_op, const loom_op_t* before_op,
    const loom_attribute_t* attr, uint8_t depth, bool* out_available) {
  *out_available = false;
  if (!loom_availability_query_is_valid(analysis, before_op) || !attr) {
    return iree_ok_status();
  }
  if (depth > LOOM_ATTR_DICT_MAX_NESTING_DEPTH) return iree_ok_status();
  switch (attr->kind) {
    case LOOM_ATTR_ABSENT:
    case LOOM_ATTR_I64:
    case LOOM_ATTR_F64:
    case LOOM_ATTR_STRING:
    case LOOM_ATTR_BOOL:
    case LOOM_ATTR_ENUM:
    case LOOM_ATTR_I64_ARRAY:
    case LOOM_ATTR_SYMBOL:
      *out_available = true;
      return iree_ok_status();
    case LOOM_ATTR_TYPE:
      if (attr->type_id == LOOM_TYPE_ID_INVALID ||
          attr->type_id >= analysis->module->types.count) {
        return iree_ok_status();
      }
      return loom_availability_type_is_available_before_op(
          analysis, moving_root_op, before_op,
          analysis->module->types.entries[attr->type_id], out_available);
    case LOOM_ATTR_PREDICATE_LIST:
      if (attr->count > 0 && !attr->predicate_list) return iree_ok_status();
      for (uint16_t i = 0; i < attr->count; ++i) {
        if (!loom_availability_predicate_is_available_before_op(
                analysis, moving_root_op, before_op,
                &attr->predicate_list[i])) {
          return iree_ok_status();
        }
      }
      *out_available = true;
      return iree_ok_status();
    case LOOM_ATTR_DICT:
      if (attr->count > 0 && !attr->dict_entries) return iree_ok_status();
      for (uint16_t i = 0; i < attr->count; ++i) {
        IREE_RETURN_IF_ERROR(loom_availability_attr_is_available_before_op_impl(
            analysis, moving_root_op, before_op, &attr->dict_entries[i].value,
            (uint8_t)(depth + 1), out_available));
        if (!*out_available) return iree_ok_status();
      }
      *out_available = true;
      return iree_ok_status();
    case LOOM_ATTR_ENCODING: {
      const loom_encoding_t* encoding =
          loom_module_encoding(analysis->module, attr->encoding_id);
      if (!encoding) return iree_ok_status();
      if (encoding->attribute_count > 0 && !encoding->attributes) {
        return iree_ok_status();
      }
      for (uint8_t i = 0; i < encoding->attribute_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_availability_attr_is_available_before_op_impl(
            analysis, moving_root_op, before_op, &encoding->attributes[i].value,
            (uint8_t)(depth + 1), out_available));
        if (!*out_available) return iree_ok_status();
      }
      *out_available = true;
      return iree_ok_status();
    }
    default:
      return iree_ok_status();
  }
}

iree_status_t loom_availability_attr_is_available_before_op(
    const loom_availability_analysis_t* analysis,
    const loom_op_t* moving_root_op, const loom_op_t* before_op,
    const loom_attribute_t* attr, bool* out_available) {
  return loom_availability_attr_is_available_before_op_impl(
      analysis, moving_root_op, before_op, attr, 0, out_available);
}

iree_status_t loom_availability_op_attrs_are_available_before_op(
    const loom_availability_analysis_t* analysis,
    const loom_op_t* moving_root_op, const loom_op_t* before_op,
    const loom_op_t* op, bool* out_available) {
  *out_available = false;
  if (!loom_availability_query_is_valid(analysis, before_op) || !op) {
    return iree_ok_status();
  }
  const loom_attribute_t* attrs = loom_op_const_attrs(op);
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_availability_attr_is_available_before_op(
        analysis, moving_root_op, before_op, &attrs[i], out_available));
    if (!*out_available) return iree_ok_status();
  }
  *out_available = true;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Op and block availability
//===----------------------------------------------------------------------===//

iree_status_t loom_availability_op_captures_are_available_before_op(
    const loom_availability_analysis_t* analysis,
    const loom_op_t* moving_root_op, const loom_op_t* before_op,
    const loom_op_t* op, bool* out_available) {
  *out_available = false;
  if (!loom_availability_query_is_valid(analysis, before_op) || !op) {
    return iree_ok_status();
  }

  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    if (!loom_availability_value_is_available_before_op(
            analysis, moving_root_op, before_op, operands[i])) {
      return iree_ok_status();
    }
  }

  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_availability_value_type_is_available_before_op(
        analysis, moving_root_op, before_op, results[i], out_available));
    if (!*out_available) return iree_ok_status();
  }

  return loom_availability_op_attrs_are_available_before_op(
      analysis, moving_root_op, before_op, op, out_available);
}

iree_status_t loom_availability_block_arg_types_are_available_before_op(
    const loom_availability_analysis_t* analysis,
    const loom_op_t* moving_root_op, const loom_op_t* before_op,
    const loom_block_t* block, bool* out_available) {
  *out_available = false;
  if (!loom_availability_query_is_valid(analysis, before_op) || !block) {
    return iree_ok_status();
  }
  for (uint16_t i = 0; i < block->arg_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_availability_value_type_is_available_before_op(
        analysis, moving_root_op, before_op, loom_block_arg_id(block, i),
        out_available));
    if (!*out_available) return iree_ok_status();
  }
  *out_available = true;
  return iree_ok_status();
}
