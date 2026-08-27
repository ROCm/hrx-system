// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/abi/materialization.h"

#include <inttypes.h>

#include "iree/vm/module.h"
#include "loom/codegen/low/function.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/arch/vm/abi/layout.h"
#include "loom/target/arch/vm/lower/types.h"
#include "loom/target/types.h"
#include "loom/util/walk.h"

static const loom_pass_info_t loom_vm_materialize_call_abi_pass_info_storage = {
    .name = IREE_SVL("vm-materialize-call-abi"),
    .description = IREE_SVL("Materialize VM call ABI register boundaries."),
    .kind = LOOM_PASS_FUNCTION,
};

const loom_pass_info_t* loom_vm_materialize_call_abi_pass_info(void) {
  return &loom_vm_materialize_call_abi_pass_info_storage;
}

static bool loom_vm_call_abi_type_is_direct_value(loom_type_t type) {
  loom_scalar_type_t scalar_type = 0;
  return loom_vm_value_register_scalar_type(type, &scalar_type);
}

static iree_status_t loom_vm_call_abi_validate_values(
    const loom_module_t* module, const loom_value_id_t* values,
    iree_host_size_t value_count, const char* boundary_name) {
  if (value_count > IREE_VM_CALL_DIRECT_REGISTER_COUNT) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "VM %s has %" PRIhsz
                            " values but the direct ABI supports %u",
                            boundary_name, value_count,
                            (unsigned)IREE_VM_CALL_DIRECT_REGISTER_COUNT);
  }
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    const loom_type_t type = loom_module_value_type(module, values[i]);
    if (!loom_vm_call_abi_type_is_direct_value(type)) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "VM %s value %" PRIhsz
          " must be one scalar vm.value in the direct-only ABI",
          boundary_name, i);
    }
  }
  return iree_ok_status();
}

typedef struct loom_vm_call_abi_validate_walk_t {
  // Module owning all walked values.
  const loom_module_t* module;
} loom_vm_call_abi_validate_walk_t;

static iree_status_t loom_vm_call_abi_validate_op(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  *out_result = LOOM_WALK_CONTINUE;
  const loom_vm_call_abi_validate_walk_t* walk =
      (const loom_vm_call_abi_validate_walk_t*)user_data;
  if (loom_low_func_call_isa(op)) {
    const loom_value_slice_t operands = loom_low_func_call_operands(op);
    IREE_RETURN_IF_ERROR(loom_vm_call_abi_validate_values(
        walk->module, operands.values, operands.count, "call argument list"));
    const loom_value_slice_t results = loom_low_func_call_results(op);
    return loom_vm_call_abi_validate_values(walk->module, results.values,
                                            results.count, "call result list");
  }
  if (loom_low_func_call_indirect_isa(op)) {
    const loom_value_slice_t operands =
        loom_low_func_call_indirect_operands(op);
    IREE_RETURN_IF_ERROR(loom_vm_call_abi_validate_values(
        walk->module, operands.values, operands.count,
        "indirect-call argument list"));
    const loom_value_slice_t results = loom_low_func_call_indirect_results(op);
    return loom_vm_call_abi_validate_values(walk->module, results.values,
                                            results.count,
                                            "indirect-call result list");
  }
  if (loom_low_return_isa(op)) {
    const loom_value_slice_t values = loom_low_return_values(op);
    return loom_vm_call_abi_validate_values(walk->module, values.values,
                                            values.count, "return value list");
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_call_abi_validate_function(
    const loom_module_t* module, loom_func_like_t function,
    iree_arena_allocator_t* arena) {
  uint16_t argument_count = 0;
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(function, &argument_count);
  IREE_RETURN_IF_ERROR(loom_vm_call_abi_validate_values(
      module, arguments, argument_count, "function argument list"));
  const loom_value_slice_t results = loom_low_func_def_results(function.op);
  IREE_RETURN_IF_ERROR(loom_vm_call_abi_validate_values(
      module, results.values, results.count, "function result list"));

  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  loom_vm_call_abi_validate_walk_t walk = {
      .module = module,
  };
  return loom_walk_function(
      module, function, LOOM_WALK_PRE_ORDER,
      (loom_walk_callback_t){loom_vm_call_abi_validate_op, &walk}, arena,
      &walk_result);
}

static iree_status_t loom_vm_call_abi_build_copy(loom_rewriter_t* rewriter,
                                                 loom_value_id_t source,
                                                 loom_location_id_t location,
                                                 loom_op_t** out_copy_op) {
  return loom_low_copy_build(&rewriter->builder, source, /*detached=*/false,
                             loom_module_value_type(rewriter->module, source),
                             location, out_copy_op);
}

static iree_status_t loom_vm_call_abi_materialize_entry(
    loom_rewriter_t* rewriter, loom_func_like_t function) {
  uint16_t argument_count = 0;
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(function, &argument_count);
  if (argument_count == 0) return iree_ok_status();

  loom_region_t* body = loom_low_function_body(function.op);
  loom_block_t* entry_block = loom_region_entry_block(body);
  loom_builder_set_before(&rewriter->builder, entry_block->first_op);
  for (uint16_t i = 0; i < argument_count; ++i) {
    loom_op_t* copy_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vm_call_abi_build_copy(
        rewriter, arguments[i], function.op->location, &copy_op));
    IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_except(
        rewriter, arguments[i], loom_low_copy_result(copy_op), copy_op));
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_call_abi_materialize_operands(
    loom_rewriter_t* rewriter, loom_op_t* op, uint16_t operand_base,
    loom_value_slice_t operands) {
  if (operands.count == 0) return iree_ok_status();
  loom_builder_set_before(&rewriter->builder, op);
  for (uint16_t i = 0; i < operands.count; ++i) {
    loom_op_t* copy_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vm_call_abi_build_copy(
        rewriter, operands.values[i], op->location, &copy_op));
    IREE_RETURN_IF_ERROR(
        loom_rewriter_set_operand(rewriter, op, (uint16_t)(operand_base + i),
                                  loom_low_copy_result(copy_op)));
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_call_abi_materialize_results(
    loom_rewriter_t* rewriter, loom_op_t* op, loom_value_slice_t results) {
  if (results.count == 0) return iree_ok_status();
  loom_builder_set_after(&rewriter->builder, op);
  for (uint16_t i = 0; i < results.count; ++i) {
    loom_op_t* copy_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vm_call_abi_build_copy(
        rewriter, results.values[i], op->location, &copy_op));
    IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_except(
        rewriter, results.values[i], loom_low_copy_result(copy_op), copy_op));
  }
  return iree_ok_status();
}

typedef struct loom_vm_call_abi_materialize_walk_t {
  // Rewriter used to insert boundary copies and update uses.
  loom_rewriter_t* rewriter;
} loom_vm_call_abi_materialize_walk_t;

static iree_status_t loom_vm_call_abi_materialize_op(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  *out_result = LOOM_WALK_CONTINUE;
  loom_vm_call_abi_materialize_walk_t* walk =
      (loom_vm_call_abi_materialize_walk_t*)user_data;
  if (loom_low_func_call_isa(op)) {
    IREE_RETURN_IF_ERROR(loom_vm_call_abi_materialize_operands(
        walk->rewriter, op, /*operand_base=*/0,
        loom_low_func_call_operands(op)));
    return loom_vm_call_abi_materialize_results(walk->rewriter, op,
                                                loom_low_func_call_results(op));
  }
  if (loom_low_func_call_indirect_isa(op)) {
    IREE_RETURN_IF_ERROR(loom_vm_call_abi_materialize_operands(
        walk->rewriter, op, /*operand_base=*/1,
        loom_low_func_call_indirect_operands(op)));
    return loom_vm_call_abi_materialize_results(
        walk->rewriter, op, loom_low_func_call_indirect_results(op));
  }
  if (loom_low_return_isa(op)) {
    return loom_vm_call_abi_materialize_operands(
        walk->rewriter, op, /*operand_base=*/0, loom_low_return_values(op));
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_call_abi_materialize_function(
    const loom_module_t* module, loom_func_like_t function,
    loom_vm_call_abi_materialize_walk_t* walk) {
  IREE_RETURN_IF_ERROR(
      loom_vm_call_abi_materialize_entry(walk->rewriter, function));
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  return loom_walk_function(
      module, function, LOOM_WALK_PRE_ORDER,
      (loom_walk_callback_t){loom_vm_call_abi_materialize_op, walk},
      walk->rewriter->arena, &walk_result);
}

static iree_status_t loom_vm_call_abi_preserve_logical_signature(
    loom_rewriter_t* rewriter, loom_func_like_t function) {
  if (loom_low_func_def_abi_layout(function.op).count != 0) {
    return iree_ok_status();
  }

  uint16_t argument_count = 0;
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(function, &argument_count);
  const loom_value_slice_t results = loom_low_func_def_results(function.op);
  iree_host_size_t type_count = 0;
  if (!iree_host_size_checked_add(argument_count, results.count, &type_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VM logical signature type count overflows");
  }
  loom_type_t* types = NULL;
  if (type_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        rewriter->arena, type_count, sizeof(*types), (void**)&types));
  }
  for (uint16_t i = 0; i < argument_count; ++i) {
    types[i] = loom_module_value_type(rewriter->module, arguments[i]);
  }
  for (iree_host_size_t i = 0; i < results.count; ++i) {
    types[argument_count + i] =
        loom_module_value_type(rewriter->module, results.values[i]);
  }

  loom_attribute_t layout_attr = loom_attr_absent();
  const loom_type_t* result_types =
      results.count != 0 ? types + argument_count : NULL;
  IREE_RETURN_IF_ERROR(loom_vm_call_abi_layout_make_attr(
      rewriter->module, types, argument_count, result_types, results.count,
      &layout_attr));
  return loom_rewriter_set_attr(rewriter, function.op,
                                loom_low_func_def_abi_layout_ATTR_INDEX,
                                layout_attr);
}

iree_status_t loom_vm_materialize_call_abi_run(loom_pass_t* pass,
                                               loom_module_t* module,
                                               loom_func_like_t function) {
  if (!loom_low_func_def_isa(function.op) ||
      loom_func_like_abi(function) != LOOM_TARGET_ABI_VM_FUNCTION) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_vm_call_abi_validate_function(module, function, pass->arena));

  loom_rewriter_t rewriter = {0};
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, module, pass->arena));
  loom_vm_call_abi_materialize_walk_t walk = {
      .rewriter = &rewriter,
  };
  iree_status_t status =
      loom_vm_call_abi_preserve_logical_signature(&rewriter, function);
  if (iree_status_is_ok(status)) {
    status = loom_vm_call_abi_materialize_function(module, function, &walk);
  }
  if (iree_status_is_ok(status) &&
      iree_any_bit_set(rewriter.flags, LOOM_REWRITER_FLAG_CHANGED)) {
    loom_pass_mark_changed(pass);
  }
  loom_rewriter_deinitialize(&rewriter);
  return status;
}
