// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/rewrite/callable.h"

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/cfg/ops.h"
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

static iree_status_t loom_callable_validate_single_block_body(
    const loom_module_t* module, const loom_op_t* call_op,
    loom_func_like_t callee, loom_block_t** out_entry_block,
    loom_op_t** out_terminator_op) {
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
  loom_op_t* terminator_op =
      loom_block_op(entry_block, entry_block->op_count - 1);
  const uint8_t body_region_index = loom_func_like_body_region_index(callee);
  const loom_op_vtable_t* callee_vtable = loom_op_vtable(module, callee.op);
  const loom_region_descriptor_t* body_descriptor =
      loom_op_vtable_region_descriptor(callee_vtable, body_region_index);
  if (!body_descriptor ||
      !loom_op_has_trait(module, terminator_op, LOOM_TRAIT_TERMINATOR) ||
      (body_descriptor->terminator != LOOM_OP_KIND_UNKNOWN &&
       terminator_op->kind != body_descriptor->terminator)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "callee body must end with its declared "
                            "terminator to inline");
  }
  *out_entry_block = entry_block;
  *out_terminator_op = terminator_op;
  return iree_ok_status();
}

bool loom_callable_body_is_linear(const loom_module_t* module,
                                  loom_func_like_t callee) {
  if (!module || !loom_func_like_isa(callee)) return false;
  loom_region_t* body = loom_func_like_body(callee);
  if (!body || body->block_count != 1) return false;
  const loom_block_t* entry_block = loom_region_const_entry_block(body);
  if (!entry_block || entry_block->op_count == 0) return false;
  const loom_op_t* terminator = loom_block_const_last_op(entry_block);
  const loom_op_vtable_t* callee_vtable = loom_op_vtable(module, callee.op);
  const loom_region_descriptor_t* body_descriptor =
      loom_op_vtable_region_descriptor(
          callee_vtable, loom_func_like_body_region_index(callee));
  return body_descriptor != NULL &&
         body_descriptor->terminator != LOOM_OP_KIND_UNKNOWN &&
         terminator->kind == body_descriptor->terminator;
}

bool loom_callable_call_site_allows_cfg_splice(const loom_module_t* module,
                                               const loom_op_t* call_op) {
  if (!module || !call_op || !call_op->parent_block || !call_op->parent_op ||
      iree_any_bit_set(call_op->flags, LOOM_OP_FLAG_DEAD)) {
    return false;
  }
  const loom_block_t* caller_block = call_op->parent_block;
  const loom_region_t* caller_region = caller_block->parent_region;
  uint16_t caller_block_index = 0;
  if (!caller_region || !loom_region_try_block_index(
                            caller_region, caller_block, &caller_block_index)) {
    return false;
  }

  const loom_op_t* parent_op = call_op->parent_op;
  const loom_op_vtable_t* parent_vtable = loom_op_vtable(module, parent_op);
  loom_region_t* const* parent_regions = loom_op_regions(parent_op);
  for (uint8_t region_index = 0; region_index < parent_op->region_count;
       ++region_index) {
    if (parent_regions[region_index] != caller_region) continue;
    const loom_region_descriptor_t* descriptor =
        loom_op_vtable_region_descriptor(parent_vtable, region_index);
    return descriptor != NULL &&
           !iree_any_bit_set(descriptor->flags, LOOM_REGION_SINGLE_BLOCK);
  }
  return false;
}

typedef struct loom_callable_cfg_body_t {
  // Callee body being spliced into the caller.
  loom_region_t* region;
  // Entry block receiving the call operands.
  loom_block_t* entry_block;
  // Declared function return operation kind.
  loom_op_kind_t return_op_kind;
  // Number of return exits across all body blocks.
  uint16_t return_count;
} loom_callable_cfg_body_t;

static bool loom_callable_value_is_valid(const loom_module_t* module,
                                         loom_value_id_t value_id) {
  return value_id != LOOM_VALUE_ID_INVALID && value_id < module->values.count;
}

static iree_status_t loom_callable_validate_cfg_body(
    const loom_module_t* module, const loom_op_t* call_op,
    loom_func_like_t callee, loom_call_like_t call,
    loom_callable_cfg_body_t* out_body) {
  *out_body = (loom_callable_cfg_body_t){0};
  loom_region_t* body = loom_func_like_body(callee);
  if (!body || body->block_count == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "callee has no inlineable body");
  }
  if (loom_callable_op_is_inside_region(call_op, body)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "cannot inline a call from inside its callee body");
  }

  const uint8_t body_region_index = loom_func_like_body_region_index(callee);
  const loom_op_vtable_t* callee_vtable = loom_op_vtable(module, callee.op);
  const loom_region_descriptor_t* body_descriptor =
      loom_op_vtable_region_descriptor(callee_vtable, body_region_index);
  if (!body_descriptor || body_descriptor->terminator == LOOM_OP_KIND_UNKNOWN) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "callee body has no declared return terminator to inline");
  }

  loom_block_t* entry_block = loom_region_entry_block(body);
  uint16_t argument_count = 0;
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(callee, &argument_count);
  const loom_value_slice_t call_operands = loom_call_like_operands(call);
  if (argument_count != call_operands.count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "call operand count %u does not match callee argument count %u",
        (unsigned)call_operands.count, (unsigned)argument_count);
  }
  if (entry_block->arg_count != argument_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "callee entry block argument count does not match its signature");
  }
  for (uint16_t i = 0; i < argument_count; ++i) {
    if (!loom_callable_value_is_valid(module, arguments[i]) ||
        !loom_callable_value_is_valid(module, call_operands.values[i])) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "call operand and callee argument must be valid");
    }
    if (loom_block_arg_id(entry_block, i) != arguments[i]) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "callee entry block argument does not match its signature");
    }
    if (!loom_type_equal(
            loom_module_value_type(module, arguments[i]),
            loom_module_value_type(module, call_operands.values[i]))) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "call operand type does not match callee argument type");
    }
  }

  const loom_value_slice_t call_results = loom_call_like_results(call);
  if (call_results.count != callee.op->result_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "call result count %u does not match callee result count %u",
        (unsigned)call_results.count, (unsigned)callee.op->result_count);
  }
  const loom_value_id_t* callee_results = loom_op_const_results(callee.op);
  for (uint16_t i = 0; i < call_results.count; ++i) {
    if (!loom_callable_value_is_valid(module, call_results.values[i]) ||
        !loom_callable_value_is_valid(module, callee_results[i])) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "call and callee results must be valid");
    }
    if (!loom_type_equal(loom_module_value_type(module, call_results.values[i]),
                         loom_module_value_type(module, callee_results[i]))) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "call result type does not match callee result type");
    }
  }

  uint16_t return_count = 0;
  for (uint16_t block_index = 0; block_index < body->block_count;
       ++block_index) {
    loom_block_t* block = loom_region_block(body, block_index);
    loom_op_t* terminator =
        block->op_count > 0 ? loom_block_op(block, block->op_count - 1) : NULL;
    if (!terminator) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "callee body block has no terminator");
    }
    if (!loom_op_has_trait(module, terminator, LOOM_TRAIT_TERMINATOR)) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "callee body block does not end in a terminator");
    }
    for (uint8_t successor_index = 0;
         successor_index < terminator->successor_count; ++successor_index) {
      uint16_t ignored_block_index = 0;
      if (!loom_region_try_block_index(
              body, loom_op_successors(terminator)[successor_index],
              &ignored_block_index)) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "callee terminator successor leaves the callable body");
      }
    }
    if (terminator->kind != body_descriptor->terminator) {
      continue;
    }
    if (return_count == UINT16_MAX) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "callee return count exceeds UINT16_MAX");
    }
    ++return_count;
    if (terminator->operand_count != call_results.count) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "callee return operand count %u does not match call result count %u",
          (unsigned)terminator->operand_count, (unsigned)call_results.count);
    }
    const loom_value_id_t* return_values = loom_op_const_operands(terminator);
    for (uint16_t i = 0; i < call_results.count; ++i) {
      if (!loom_callable_value_is_valid(module, return_values[i])) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "callee return operand must be valid");
      }
      if (!loom_type_equal(
              loom_module_value_type(module, return_values[i]),
              loom_module_value_type(module, call_results.values[i]))) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "callee return operand type does not match call result type");
      }
    }
  }

  *out_body = (loom_callable_cfg_body_t){
      .region = body,
      .entry_block = entry_block,
      .return_op_kind = body_descriptor->terminator,
      .return_count = return_count,
  };
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
    loom_rewriter_t* rewriter, loom_call_like_t call, loom_op_t* terminator_op,
    loom_ir_remap_t* remap, loom_value_id_t* replacements) {
  loom_value_slice_t return_operands = {
      .values = loom_op_operands(terminator_op),
      .count = terminator_op->operand_count,
  };
  loom_value_slice_t call_results_slice = loom_call_like_results(call);
  if (return_operands.count != call_results_slice.count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "callee terminator operand count %u does not match call result count "
        "%u",
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

static iree_status_t loom_callable_inline_single_block_call(
    loom_rewriter_t* rewriter, loom_op_t* call_op, loom_func_like_t callee) {
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
  loom_op_t* terminator_op = NULL;
  IREE_RETURN_IF_ERROR(loom_callable_validate_single_block_body(
      rewriter->module, call_op, callee, &entry_block, &terminator_op));

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
        rewriter, call, terminator_op, &remap, replacements);
  }
  loom_builder_restore(&rewriter->builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);

  IREE_RETURN_IF_ERROR(loom_callable_preserve_call_result_names(
      rewriter, call, replacements, call_results.count, value_checkpoint));
  return loom_rewriter_replace_all_uses_and_erase(
      rewriter, call_op, replacements, call_results.count);
}

static iree_status_t loom_callable_collect_return_projections(
    const loom_callable_cfg_body_t* body,
    loom_ir_remap_op_projection_t* projections) {
  uint16_t return_index = 0;
  for (uint16_t block_index = 0; block_index < body->region->block_count;
       ++block_index) {
    const loom_block_t* block =
        loom_region_const_block(body->region, block_index);
    const loom_op_t* terminator = loom_block_const_last_op(block);
    if (terminator->kind != body->return_op_kind) continue;
    if (return_index >= body->return_count) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "callee return projection count changed");
    }
    projections[return_index++].source_op = terminator;
  }
  if (return_index != body->return_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "callee return projection count changed");
  }
  return iree_ok_status();
}

static iree_status_t loom_callable_inline_cfg_call(
    loom_rewriter_t* rewriter, loom_op_t* call_op, loom_func_like_t callee,
    loom_call_like_t call, const loom_callable_cfg_body_t* body,
    loom_callable_build_branch_fn_t build_branch) {
  if (!build_branch) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "multi-block callable inlining requires a branch builder");
  }
  if (!loom_callable_call_site_allows_cfg_splice(rewriter->module, call_op)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "multi-block callable inlining requires a CFG-capable caller region");
  }
  loom_block_t* caller_block = call_op->parent_block;
  loom_region_t* caller_region = caller_block->parent_region;
  uint16_t caller_block_index = 0;
  if (!caller_region || !loom_region_try_block_index(
                            caller_region, caller_block, &caller_block_index)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "call op is not in a live caller region");
  }
  iree_host_size_t final_block_count = 0;
  if (!iree_host_size_checked_add(caller_region->block_count,
                                  body->region->block_count,
                                  &final_block_count) ||
      !iree_host_size_checked_add(final_block_count, 1, &final_block_count) ||
      final_block_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "inlined caller block count exceeds UINT16_MAX");
  }

  loom_ir_remap_op_projection_t* return_projections = NULL;
  if (body->return_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        rewriter->arena, body->return_count, sizeof(*return_projections),
        (void**)&return_projections));
    memset(return_projections, 0,
           (iree_host_size_t)body->return_count * sizeof(*return_projections));
    IREE_RETURN_IF_ERROR(
        loom_callable_collect_return_projections(body, return_projections));
  }
  const loom_ir_remap_options_t remap_options = {
      .op_projection =
          {
              .entries = return_projections,
              .count = body->return_count,
          },
  };
  loom_ir_remap_t remap = {0};
  IREE_RETURN_IF_ERROR(
      loom_ir_remap_initialize(rewriter->module, rewriter->module,
                               rewriter->arena, &remap_options, &remap));

  const loom_value_slice_t call_operands = loom_call_like_operands(call);
  loom_value_id_t* entry_arguments = NULL;
  if (call_operands.count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        rewriter->arena, call_operands.count, sizeof(*entry_arguments),
        (void**)&entry_arguments));
    memcpy(entry_arguments, call_operands.values,
           (iree_host_size_t)call_operands.count * sizeof(*entry_arguments));
  }
  const loom_location_id_t call_location = call_op->location;
  loom_op_t* caller_parent_op = call_op->parent_op;

  loom_builder_ip_t saved_ip = loom_builder_save(&rewriter->builder);
  rewriter->builder.ip.parent_op = caller_parent_op;
  const uint16_t cloned_block_index = caller_block_index + 1;
  iree_status_t status =
      loom_ir_clone_region_blocks(&rewriter->builder, body->region,
                                  caller_region, cloned_block_index, &remap);
  loom_block_t* continuation_block = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_region_insert_block(
        rewriter->module, caller_region,
        cloned_block_index + body->region->block_count, &continuation_block);
  }

  const loom_value_slice_t call_results = loom_call_like_results(call);
  loom_value_id_t* continuation_arguments = NULL;
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  if (iree_status_is_ok(status) && call_results.count > 0) {
    status = iree_arena_allocate_array(rewriter->arena, call_results.count,
                                       sizeof(*continuation_arguments),
                                       (void**)&continuation_arguments);
  }
  for (uint16_t i = 0; i < call_results.count && iree_status_is_ok(status);
       ++i) {
    status = loom_builder_define_block_arg(
        &rewriter->builder, continuation_block,
        loom_module_value_type(rewriter->module, call_results.values[i]),
        &continuation_arguments[i]);
  }
  if (iree_status_is_ok(status)) {
    status = loom_callable_preserve_call_result_names(
        rewriter, call, continuation_arguments, call_results.count,
        value_checkpoint);
  }
  if (!iree_status_is_ok(status)) {
    loom_builder_restore(&rewriter->builder, saved_ip);
    return status;
  }

  loom_op_t* tail_op = call_op->next_op;
  while (tail_op) {
    loom_op_t* next_op = tail_op->next_op;
    status = loom_rewriter_move_to_block_end(
        rewriter, tail_op, continuation_block, caller_parent_op);
    if (!iree_status_is_ok(status)) break;
    tail_op = next_op;
  }
  for (uint16_t i = 0; i < body->return_count && iree_status_is_ok(status);
       ++i) {
    loom_op_t* cloned_return = return_projections[i].target_op;
    if (!cloned_return) {
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "callee return was not cloned");
      break;
    }
    loom_builder_set_before(&rewriter->builder, cloned_return);
    loom_op_t* continuation_branch = NULL;
    status = build_branch(&rewriter->builder, continuation_block,
                          loom_op_const_operands(cloned_return),
                          cloned_return->operand_count, cloned_return->location,
                          &continuation_branch);
    if (iree_status_is_ok(status)) {
      status = loom_rewriter_erase(rewriter, cloned_return);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_rewriter_replace_all_uses_and_erase(
        rewriter, call_op, continuation_arguments, call_results.count);
  }
  if (iree_status_is_ok(status)) {
    loom_builder_set_block(&rewriter->builder, caller_block);
    rewriter->builder.ip.parent_op = caller_parent_op;
    loom_block_t* cloned_entry_block =
        loom_region_block(caller_region, cloned_block_index);
    loom_op_t* entry_branch = NULL;
    status =
        build_branch(&rewriter->builder, cloned_entry_block, entry_arguments,
                     call_operands.count, call_location, &entry_branch);
  }
  loom_builder_restore(&rewriter->builder, saved_ip);
  return status;
}

iree_status_t loom_callable_inline_call_with_branch(
    loom_rewriter_t* rewriter, loom_op_t* call_op, loom_func_like_t callee,
    loom_callable_build_branch_fn_t build_branch) {
  if (!call_op || !call_op->parent_block ||
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

  loom_callable_cfg_body_t body = {0};
  IREE_RETURN_IF_ERROR(loom_callable_validate_cfg_body(
      rewriter->module, call_op, callee, call, &body));
  if (loom_callable_body_is_linear(rewriter->module, callee)) {
    return loom_callable_inline_single_block_call(rewriter, call_op, callee);
  }
  return loom_callable_inline_cfg_call(rewriter, call_op, callee, call, &body,
                                       build_branch);
}

iree_status_t loom_callable_inline_call(loom_rewriter_t* rewriter,
                                        loom_op_t* call_op,
                                        loom_func_like_t callee) {
  return loom_callable_inline_call_with_branch(rewriter, call_op, callee,
                                               loom_cfg_br_build);
}

iree_status_t loom_callable_inline_consuming_call(
    loom_rewriter_t* rewriter, const loom_availability_analysis_t* availability,
    loom_op_t* call_op, loom_func_like_t callee) {
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
  loom_op_t* terminator_op = NULL;
  IREE_RETURN_IF_ERROR(loom_callable_validate_single_block_body(
      rewriter->module, call_op, callee, &entry_block, &terminator_op));

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
      rewriter, call, terminator_op, &remap, replacements));

  loom_ir_move_block_options_t move_options = {
      .omit_terminators = true,
  };
  IREE_RETURN_IF_ERROR(loom_ir_move_block_ops_before(
      rewriter, availability, entry_block, call_op, &remap, &move_options));
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

typedef struct loom_callable_clone_symbol_state_t {
  // Source function symbol replaced during cloning.
  loom_symbol_ref_t source_ref;

  // Target function symbol replacing |source_ref|.
  loom_symbol_ref_t target_ref;
} loom_callable_clone_symbol_state_t;

static iree_status_t loom_callable_clone_remap_symbol(
    void* user_data, const loom_module_t* source_module,
    loom_module_t* target_module, loom_symbol_ref_t source_ref,
    loom_symbol_ref_t* out_target_ref) {
  (void)source_module;
  (void)target_module;
  const loom_callable_clone_symbol_state_t* state =
      (const loom_callable_clone_symbol_state_t*)user_data;
  *out_target_ref =
      loom_callable_symbol_ref_equal(source_ref, state->source_ref)
          ? state->target_ref
          : source_ref;
  return iree_ok_status();
}

iree_status_t loom_callable_clone_definition(
    loom_builder_t* builder, loom_func_like_t source,
    loom_symbol_ref_t target_ref, loom_func_like_t* out_cloned,
    iree_arena_allocator_t* scratch_arena) {
  *out_cloned = (loom_func_like_t){0};
  IREE_RETURN_IF_ERROR(
      loom_callable_validate_same_module_callee(builder->module, source));
  if (!loom_symbol_ref_is_valid(target_ref) || target_ref.module_id != 0 ||
      target_ref.symbol_id >= builder->module->symbols.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target symbol ref {module=%u, symbol=%u} is invalid",
        (unsigned)target_ref.module_id, (unsigned)target_ref.symbol_id);
  }
  loom_symbol_t* target_symbol =
      &builder->module->symbols.entries[target_ref.symbol_id];
  if (target_symbol->defining_op != NULL) {
    return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                            "target symbol already has a defining op");
  }

  loom_callable_clone_symbol_state_t symbol_state = {
      .source_ref = loom_func_like_callee(source),
      .target_ref = target_ref,
  };
  const loom_ir_remap_options_t remap_options = {
      .remap_symbol = loom_ir_remap_symbol_callback_make(
          loom_callable_clone_remap_symbol, &symbol_state),
      .remap_same_module_symbols = true,
  };
  loom_ir_remap_t remap = {0};
  IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(
      builder->module, builder->module, scratch_arena, &remap_options, &remap));

  loom_op_t* cloned_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_ir_clone_op(builder, source.op, &remap, &cloned_op));
  loom_func_like_t cloned = loom_func_like_cast(builder->module, cloned_op);
  if (!loom_func_like_isa(cloned) || target_symbol->defining_op != cloned_op) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "cloned function did not bind the requested target symbol");
  }
  *out_cloned = cloned;
  return iree_ok_status();
}
