// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/rewrite/callable.h"

#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/rewrite/materialize.h"

static bool loom_callable_get_call_symbol_ref(const loom_module_t* module,
                                              const loom_op_t* call_op,
                                              loom_symbol_ref_t* out_ref) {
  loom_call_like_t call = loom_call_like_cast(module, (loom_op_t*)call_op);
  if (loom_call_like_isa(call)) {
    *out_ref = loom_call_like_callee(call);
    return true;
  }
  *out_ref = loom_symbol_ref_null();
  return false;
}

static iree_status_t loom_callable_get_whole_call(const loom_module_t* module,
                                                  const loom_op_t* call_op,
                                                  loom_call_like_t* out_call) {
  loom_call_like_t call = loom_call_like_cast(module, (loom_op_t*)call_op);
  if (!loom_call_like_isa(call)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "op is not a direct callable op");
  }
  if (loom_call_like_operand_offset(call) != 0 ||
      loom_call_like_result_offset(call) != 0 || call_op->region_count != 0 ||
      call_op->successor_count != 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "callable rewrite requires a call-like op whose operands and results "
        "are exactly the call payload");
  }
  *out_call = call;
  return iree_ok_status();
}

iree_status_t loom_callable_resolve_direct_callee(
    const loom_module_t* module, const loom_op_t* call_op,
    loom_func_like_t* out_callee) {
  *out_callee = (loom_func_like_t){0};
  loom_symbol_ref_t ref = loom_symbol_ref_null();
  if (!loom_callable_get_call_symbol_ref(module, call_op, &ref)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "op is not a direct callable op");
  }
  if (!loom_symbol_ref_is_valid(ref) || ref.module_id != 0 ||
      ref.symbol_id >= module->symbols.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "call target symbol ref {module=%u, symbol=%u} is invalid",
        (unsigned)ref.module_id, (unsigned)ref.symbol_id);
  }
  loom_symbol_t* symbol = &module->symbols.entries[ref.symbol_id];
  if (!symbol->defining_op) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "call target symbol has no defining op");
  }
  loom_func_like_t callee = loom_func_like_cast(module, symbol->defining_op);
  if (!loom_func_like_isa(callee)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "call target symbol does not define a function-like op");
  }
  *out_callee = callee;
  return iree_ok_status();
}

static iree_status_t loom_callable_validate_same_module_callee(
    const loom_module_t* module, loom_func_like_t callee) {
  if (!loom_func_like_isa(callee)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "callee must be a function-like op");
  }
  loom_symbol_ref_t callee_ref = loom_func_like_callee(callee);
  if (!loom_symbol_ref_is_valid(callee_ref) || callee_ref.module_id != 0 ||
      callee_ref.symbol_id >= module->symbols.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "callee symbol ref {module=%u, symbol=%u} is invalid in target module",
        (unsigned)callee_ref.module_id, (unsigned)callee_ref.symbol_id);
  }
  if (module->symbols.entries[callee_ref.symbol_id].defining_op != callee.op) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "callee does not belong to the target module symbol table");
  }
  return iree_ok_status();
}

static iree_status_t loom_callable_validate_call_targets_callee(
    const loom_module_t* module, const loom_op_t* call_op,
    loom_func_like_t callee) {
  loom_symbol_ref_t call_ref = loom_symbol_ref_null();
  if (!loom_callable_get_call_symbol_ref(module, call_op, &call_ref)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "op is not a direct callable op");
  }
  loom_symbol_ref_t callee_ref = loom_func_like_callee(callee);
  if (call_ref.module_id != callee_ref.module_id ||
      call_ref.symbol_id != callee_ref.symbol_id) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "call target does not match the requested callee");
  }
  return iree_ok_status();
}

static bool loom_callable_op_is_inside_region(const loom_op_t* op,
                                              const loom_region_t* region) {
  for (const loom_op_t* current = op; current; current = current->parent_op) {
    const loom_region_t* parent_region =
        current->parent_block ? current->parent_block->parent_region : NULL;
    if (parent_region == region) return true;
  }
  return false;
}

static iree_status_t loom_callable_validate_inline_body(
    const loom_op_t* call_op, loom_func_like_t callee,
    loom_block_t** out_entry_block, loom_op_t** out_return_op) {
  loom_region_t* body = loom_func_like_body(callee);
  if (!body) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "callee has no inlineable body");
  }
  if (body->block_count != 1) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "callable inlining requires a single-block callee body");
  }
  if (loom_callable_op_is_inside_region(call_op, body)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "cannot inline a call from inside its callee body");
  }
  loom_block_t* entry_block = loom_region_entry_block(body);
  if (entry_block->op_count == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "callee body has no terminator");
  }
  loom_op_t* return_op = loom_block_op(entry_block, entry_block->op_count - 1);
  if (!loom_func_return_isa(return_op)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "callee body must end with func.return to inline");
  }
  *out_entry_block = entry_block;
  *out_return_op = return_op;
  return iree_ok_status();
}

static iree_status_t loom_callable_bind_entry_args(loom_ir_remap_t* remap,
                                                   loom_func_like_t callee,
                                                   loom_call_like_t call) {
  uint16_t arg_count = 0;
  const loom_value_id_t* arg_ids = loom_func_like_arg_ids(callee, &arg_count);
  loom_value_slice_t operands = loom_call_like_operands(call);
  if (arg_count != operands.count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "call operand count %u does not match callee argument count %u",
        (unsigned)operands.count, (unsigned)arg_count);
  }
  for (uint16_t i = 0; i < arg_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_map_value(remap, arg_ids[i], operands.values[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_callable_resolve_return_replacements(
    loom_rewriter_t* rewriter, loom_call_like_t call, loom_op_t* return_op,
    loom_ir_remap_t* remap, loom_value_id_t* replacements) {
  loom_value_slice_t return_operands = loom_func_return_operands(return_op);
  loom_value_slice_t call_results_slice = loom_call_like_results(call);
  if (return_operands.count != call_results_slice.count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "func.return operand count %u does not match call result count %u",
        (unsigned)return_operands.count, (unsigned)call_results_slice.count);
  }
  for (uint16_t i = 0; i < call_results_slice.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_ir_remap_resolve_value(
        remap, return_operands.values[i], &replacements[i]));
    if (call_results_slice.values[i] == LOOM_VALUE_ID_INVALID ||
        replacements[i] == LOOM_VALUE_ID_INVALID) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "call result and replacement must be valid");
    }
    loom_type_t result_type =
        loom_module_value_type(rewriter->module, call_results_slice.values[i]);
    loom_type_t replacement_type =
        loom_module_value_type(rewriter->module, replacements[i]);
    if (!loom_type_equal(result_type, replacement_type)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "inline replacement type does not match call result type");
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_callable_preserve_call_result_names(
    loom_rewriter_t* rewriter, loom_call_like_t call,
    const loom_value_id_t* replacements, uint16_t count,
    loom_value_id_t value_checkpoint) {
  loom_value_slice_t call_results = loom_call_like_results(call);
  for (uint16_t i = 0; i < count; ++i) {
    loom_value_id_t old_result = call_results.values[i];
    loom_value_id_t replacement = replacements[i];
    if (old_result == LOOM_VALUE_ID_INVALID ||
        replacement == LOOM_VALUE_ID_INVALID) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "call result and replacement must be valid");
    }
    if (replacement < value_checkpoint) continue;
    if ((iree_host_size_t)replacement >= rewriter->module->values.count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "inline replacement value %%%u is out of range",
                              (unsigned)replacement);
    }
    IREE_RETURN_IF_ERROR(
        loom_rewriter_copy_value_name(rewriter, old_result, replacement));
  }
  return iree_ok_status();
}

static bool loom_callable_symbol_ref_equal(loom_symbol_ref_t a,
                                           loom_symbol_ref_t b) {
  return a.module_id == b.module_id && a.symbol_id == b.symbol_id;
}

typedef enum loom_callable_symbol_scan_mode_e {
  LOOM_CALLABLE_SYMBOL_SCAN_EXACT,
  LOOM_CALLABLE_SYMBOL_SCAN_OTHER,
} loom_callable_symbol_scan_mode_t;

static bool loom_callable_symbol_scan_matches(
    loom_symbol_ref_t ref, loom_symbol_ref_t target_ref,
    loom_callable_symbol_scan_mode_t mode) {
  switch (mode) {
    case LOOM_CALLABLE_SYMBOL_SCAN_EXACT:
      return loom_callable_symbol_ref_equal(ref, target_ref);
    case LOOM_CALLABLE_SYMBOL_SCAN_OTHER:
      return !loom_symbol_ref_is_valid(ref) ||
             !loom_callable_symbol_ref_equal(ref, target_ref);
  }
  return true;
}

static bool loom_callable_attr_may_reference_symbol_in_mode(
    const loom_module_t* module, const loom_attribute_t* attr,
    loom_symbol_ref_t target_ref, loom_callable_symbol_scan_mode_t mode,
    uint8_t depth) {
  if (!attr) return false;
  if (depth > LOOM_ATTR_DICT_MAX_NESTING_DEPTH) return true;
  switch ((loom_attr_kind_t)attr->kind) {
    case LOOM_ATTR_SYMBOL:
      return loom_callable_symbol_scan_matches(loom_attr_as_symbol(*attr),
                                               target_ref, mode);
    case LOOM_ATTR_DICT:
      if (attr->count > 0 && !attr->dict_entries) return true;
      for (uint16_t i = 0; i < attr->count; ++i) {
        if (loom_callable_attr_may_reference_symbol_in_mode(
                module, &attr->dict_entries[i].value, target_ref, mode,
                (uint8_t)(depth + 1))) {
          return true;
        }
      }
      return false;
    case LOOM_ATTR_ENCODING: {
      const loom_encoding_t* encoding =
          loom_module_encoding(module, attr->encoding_id);
      if (!encoding) return true;
      if (encoding->attribute_count > 0 && !encoding->attributes) return true;
      for (uint8_t i = 0; i < encoding->attribute_count; ++i) {
        if (loom_callable_attr_may_reference_symbol_in_mode(
                module, &encoding->attributes[i].value, target_ref, mode,
                (uint8_t)(depth + 1))) {
          return true;
        }
      }
      return false;
    }
    default:
      return false;
  }
}

static bool loom_callable_op_may_reference_symbol_in_mode(
    const loom_module_t* module, const loom_op_t* op,
    loom_symbol_ref_t target_ref, loom_callable_symbol_scan_mode_t mode) {
  const loom_attribute_t* attrs = loom_op_const_attrs(op);
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    if (loom_callable_attr_may_reference_symbol_in_mode(module, &attrs[i],
                                                        target_ref, mode, 0)) {
      return true;
    }
  }
  return false;
}

static bool loom_callable_op_may_reference_symbol(
    const loom_module_t* module, const loom_op_t* op,
    loom_symbol_ref_t symbol_ref) {
  return loom_callable_op_may_reference_symbol_in_mode(
      module, op, symbol_ref, LOOM_CALLABLE_SYMBOL_SCAN_EXACT);
}

static bool loom_callable_op_may_reference_other_symbol(
    const loom_module_t* module, const loom_op_t* op,
    loom_symbol_ref_t ignored_symbol_ref) {
  return loom_callable_op_may_reference_symbol_in_mode(
      module, op, ignored_symbol_ref, LOOM_CALLABLE_SYMBOL_SCAN_OTHER);
}

static bool loom_callable_region_may_reference_other_symbol(
    const loom_module_t* module, const loom_region_t* region,
    loom_symbol_ref_t ignored_symbol_ref) {
  if (!region) return false;
  loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) continue;
      if (loom_callable_op_may_reference_other_symbol(module, op,
                                                      ignored_symbol_ref)) {
        return true;
      }
      loom_region_t** regions = loom_op_regions(op);
      for (uint8_t i = 0; i < op->region_count; ++i) {
        if (loom_callable_region_may_reference_other_symbol(
                module, regions[i], ignored_symbol_ref)) {
          return true;
        }
      }
    }
  }
  return false;
}

static bool loom_callable_region_may_reference_symbol_except(
    const loom_module_t* module, const loom_region_t* region,
    loom_symbol_ref_t symbol_ref, const loom_op_t* ignored_attr_op_a,
    const loom_op_t* ignored_attr_op_b) {
  if (!region) return false;
  loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) continue;
      if (op != ignored_attr_op_a && op != ignored_attr_op_b &&
          loom_callable_op_may_reference_symbol(module, op, symbol_ref)) {
        return true;
      }
      loom_region_t** regions = loom_op_regions(op);
      for (uint8_t i = 0; i < op->region_count; ++i) {
        if (loom_callable_region_may_reference_symbol_except(
                module, regions[i], symbol_ref, ignored_attr_op_a,
                ignored_attr_op_b)) {
          return true;
        }
      }
    }
  }
  return false;
}

static iree_status_t loom_callable_validate_consumable_callee(
    const loom_module_t* module, const loom_op_t* call_op,
    loom_func_like_t callee) {
  loom_symbol_ref_t callee_ref = loom_func_like_callee(callee);
  if (!loom_symbol_ref_is_valid(callee_ref) || callee_ref.module_id != 0 ||
      callee_ref.symbol_id >= module->symbols.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "callee symbol ref {module=%u, symbol=%u} is invalid in target module",
        (unsigned)callee_ref.module_id, (unsigned)callee_ref.symbol_id);
  }
  const loom_symbol_t* symbol = &module->symbols.entries[callee_ref.symbol_id];
  if (iree_any_bit_set(symbol->flags, LOOM_SYMBOL_FLAG_PUBLIC)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "consuming inline requires a private callee symbol");
  }
  if (loom_callable_region_may_reference_symbol_except(
          module, module->body, callee_ref, call_op, callee.op)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "consuming inline requires the selected call to be the only live "
        "callee reference");
  }
  return iree_ok_status();
}

static bool loom_callable_value_is_call_operand(loom_call_like_t call,
                                                loom_value_id_t value_id) {
  loom_value_slice_t operands = loom_call_like_operands(call);
  for (uint16_t i = 0; i < operands.count; ++i) {
    if (operands.values[i] == value_id) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_callable_preserve_consuming_call_result_names(
    loom_rewriter_t* rewriter, loom_call_like_t call,
    const loom_value_id_t* replacements, uint16_t count) {
  loom_value_slice_t call_results = loom_call_like_results(call);
  for (uint16_t i = 0; i < count; ++i) {
    loom_value_id_t old_result = call_results.values[i];
    loom_value_id_t replacement = replacements[i];
    if (old_result == LOOM_VALUE_ID_INVALID ||
        replacement == LOOM_VALUE_ID_INVALID) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "call result and replacement must be valid");
    }
    if (loom_callable_value_is_call_operand(call, replacement)) {
      continue;
    }
    if ((iree_host_size_t)replacement >= rewriter->module->values.count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "inline replacement value %%%u is out of range",
                              (unsigned)replacement);
    }
    IREE_RETURN_IF_ERROR(
        loom_rewriter_copy_value_name(rewriter, old_result, replacement));
  }
  return iree_ok_status();
}

iree_status_t loom_callable_inline_call(loom_rewriter_t* rewriter,
                                        loom_op_t* call_op,
                                        loom_func_like_t callee) {
  if (!call_op->parent_block ||
      iree_any_bit_set(call_op->flags, LOOM_OP_FLAG_DEAD)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "call op must be live and linked");
  }
  IREE_RETURN_IF_ERROR(
      loom_callable_validate_same_module_callee(rewriter->module, callee));
  IREE_RETURN_IF_ERROR(loom_callable_validate_call_targets_callee(
      rewriter->module, call_op, callee));
  loom_call_like_t call = {0};
  IREE_RETURN_IF_ERROR(
      loom_callable_get_whole_call(rewriter->module, call_op, &call));

  loom_block_t* entry_block = NULL;
  loom_op_t* return_op = NULL;
  IREE_RETURN_IF_ERROR(loom_callable_validate_inline_body(
      call_op, callee, &entry_block, &return_op));

  loom_ir_remap_t remap = {0};
  IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(
      rewriter->module, rewriter->module, rewriter->arena, NULL, &remap));
  IREE_RETURN_IF_ERROR(loom_callable_bind_entry_args(&remap, callee, call));

  loom_value_slice_t call_results = loom_call_like_results(call);
  loom_value_id_t* replacements = NULL;
  if (call_results.count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        rewriter->arena, call_results.count, sizeof(loom_value_id_t),
        (void**)&replacements));
  }

  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_builder_ip_t saved_ip = loom_builder_save(&rewriter->builder);
  loom_builder_set_before(&rewriter->builder, call_op);
  loom_ir_clone_block_options_t clone_options = {
      .omit_terminators = true,
  };
  iree_status_t status = loom_ir_clone_block_ops(
      &rewriter->builder, entry_block, &remap, &clone_options);
  if (iree_status_is_ok(status)) {
    status = loom_callable_resolve_return_replacements(
        rewriter, call, return_op, &remap, replacements);
  }
  loom_builder_restore(&rewriter->builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);

  IREE_RETURN_IF_ERROR(loom_callable_preserve_call_result_names(
      rewriter, call, replacements, call_results.count, value_checkpoint));
  return loom_rewriter_replace_all_uses_and_erase(
      rewriter, call_op, replacements, call_results.count);
}

iree_status_t loom_callable_inline_consuming_call(loom_rewriter_t* rewriter,
                                                  loom_op_t* call_op,
                                                  loom_func_like_t callee) {
  if (!call_op->parent_block ||
      iree_any_bit_set(call_op->flags, LOOM_OP_FLAG_DEAD)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "call op must be live and linked");
  }
  IREE_RETURN_IF_ERROR(
      loom_callable_validate_same_module_callee(rewriter->module, callee));
  IREE_RETURN_IF_ERROR(loom_callable_validate_call_targets_callee(
      rewriter->module, call_op, callee));
  loom_call_like_t call = {0};
  IREE_RETURN_IF_ERROR(
      loom_callable_get_whole_call(rewriter->module, call_op, &call));
  IREE_RETURN_IF_ERROR(loom_callable_validate_consumable_callee(
      rewriter->module, call_op, callee));

  loom_block_t* entry_block = NULL;
  loom_op_t* return_op = NULL;
  IREE_RETURN_IF_ERROR(loom_callable_validate_inline_body(
      call_op, callee, &entry_block, &return_op));

  loom_ir_remap_options_t remap_options = {
      .allow_unmapped_values = true,
  };
  loom_ir_remap_t remap = {0};
  IREE_RETURN_IF_ERROR(
      loom_ir_remap_initialize(rewriter->module, rewriter->module,
                               rewriter->arena, &remap_options, &remap));
  IREE_RETURN_IF_ERROR(loom_callable_bind_entry_args(&remap, callee, call));

  loom_value_slice_t call_results = loom_call_like_results(call);
  loom_value_id_t* replacements = NULL;
  if (call_results.count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        rewriter->arena, call_results.count, sizeof(loom_value_id_t),
        (void**)&replacements));
  }
  IREE_RETURN_IF_ERROR(loom_callable_resolve_return_replacements(
      rewriter, call, return_op, &remap, replacements));

  loom_ir_move_block_options_t move_options = {
      .omit_terminators = true,
  };
  IREE_RETURN_IF_ERROR(loom_ir_move_block_ops_before(
      rewriter, entry_block, call_op, &remap, &move_options));
  IREE_RETURN_IF_ERROR(loom_callable_preserve_consuming_call_result_names(
      rewriter, call, replacements, call_results.count));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      rewriter, call_op, replacements, call_results.count));
  return loom_rewriter_erase(rewriter, callee.op);
}

iree_status_t loom_callable_inline_direct_call(loom_rewriter_t* rewriter,
                                               loom_op_t* call_op) {
  loom_func_like_t callee = {0};
  IREE_RETURN_IF_ERROR(
      loom_callable_resolve_direct_callee(rewriter->module, call_op, &callee));
  return loom_callable_inline_call(rewriter, call_op, callee);
}

iree_status_t loom_callable_inline_consuming_direct_call(
    loom_rewriter_t* rewriter, loom_op_t* call_op) {
  loom_func_like_t callee = {0};
  IREE_RETURN_IF_ERROR(
      loom_callable_resolve_direct_callee(rewriter->module, call_op, &callee));
  return loom_callable_inline_consuming_call(rewriter, call_op, callee);
}

typedef struct loom_callable_import_symbol_state_t {
  // Source-module symbol ref of the callable being imported.
  loom_symbol_ref_t source_callee_ref;
  // Target-module symbol ref created for the imported callable.
  loom_symbol_ref_t target_callee_ref;
  // Optional caller policy for non-callee source symbol refs.
  loom_ir_remap_symbol_callback_t external_symbol_remap;
} loom_callable_import_symbol_state_t;

static iree_status_t loom_callable_import_remap_symbol(
    void* user_data, const loom_module_t* source_module,
    loom_module_t* target_module, loom_symbol_ref_t source_ref,
    loom_symbol_ref_t* out_target_ref) {
  loom_callable_import_symbol_state_t* state =
      (loom_callable_import_symbol_state_t*)user_data;
  if (loom_callable_symbol_ref_equal(source_ref, state->source_callee_ref)) {
    *out_target_ref = state->target_callee_ref;
    return iree_ok_status();
  }
  if (!state->external_symbol_remap.fn) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "callable import encountered an unresolved external symbol reference");
  }
  return state->external_symbol_remap.fn(state->external_symbol_remap.user_data,
                                         source_module, target_module,
                                         source_ref, out_target_ref);
}

iree_status_t loom_callable_import_definition(
    loom_builder_t* builder, const loom_module_t* source_module,
    loom_func_like_t source, const loom_callable_import_options_t* options,
    loom_func_like_t* out_imported, iree_arena_allocator_t* scratch_arena) {
  *out_imported = (loom_func_like_t){0};
  if (!loom_func_like_isa(source) || !source.op ||
      iree_any_bit_set(source.op->flags, LOOM_OP_FLAG_DEAD)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source must be a live function-like op");
  }
  loom_symbol_ref_t source_ref = loom_func_like_callee(source);
  if (!loom_symbol_ref_is_valid(source_ref) || source_ref.module_id != 0 ||
      source_ref.symbol_id >= source_module->symbols.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source symbol ref {module=%u, symbol=%u} is invalid",
        (unsigned)source_ref.module_id, (unsigned)source_ref.symbol_id);
  }
  const loom_symbol_t* source_symbol =
      &source_module->symbols.entries[source_ref.symbol_id];
  if (source_symbol->name_id == LOOM_STRING_ID_INVALID ||
      source_symbol->name_id >= source_module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source symbol name id %u is invalid",
                            (unsigned)source_symbol->name_id);
  }
  if (source_symbol->defining_op != source.op) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "source symbol does not point at the requested callable");
  }
  if (!options || !options->external_symbol_remap.fn) {
    if (loom_callable_op_may_reference_other_symbol(source_module, source.op,
                                                    source_ref)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "callable import requires an external symbol policy for non-callee "
          "symbol references");
    }
    loom_region_t** regions = loom_op_regions(source.op);
    for (uint8_t i = 0; i < source.op->region_count; ++i) {
      if (loom_callable_region_may_reference_other_symbol(
              source_module, regions[i], source_ref)) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "callable import requires an external symbol policy for "
            "non-callee symbol references");
      }
    }
  }

  iree_string_view_t source_name =
      source_module->strings.entries[source_symbol->name_id];
  loom_string_id_t target_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(builder->module, source_name, &target_name_id));
  if (loom_module_find_symbol(builder->module, target_name_id) !=
      LOOM_SYMBOL_ID_INVALID) {
    return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                            "target module already has a symbol named %.*s",
                            (int)source_name.size, source_name.data);
  }

  uint16_t target_symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_add_symbol(builder->module, target_name_id,
                                              &target_symbol_id));
  loom_callable_import_symbol_state_t symbol_state = {
      .source_callee_ref = source_ref,
      .target_callee_ref = {.module_id = 0, .symbol_id = target_symbol_id},
      .external_symbol_remap = options ? options->external_symbol_remap
                                       : loom_ir_remap_symbol_callback_empty(),
  };
  loom_ir_remap_options_t remap_options = {
      .remap_symbol = loom_ir_remap_symbol_callback_make(
          loom_callable_import_remap_symbol, &symbol_state),
  };
  loom_ir_remap_t remap = {0};
  IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(
      source_module, builder->module, scratch_arena, &remap_options, &remap));

  loom_op_t* imported_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_ir_clone_op(builder, source.op, &remap, &imported_op));
  loom_func_like_t imported = loom_func_like_cast(builder->module, imported_op);
  if (!loom_func_like_isa(imported)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "imported op is not function-like after cloning");
  }
  *out_imported = imported;
  return iree_ok_status();
}
