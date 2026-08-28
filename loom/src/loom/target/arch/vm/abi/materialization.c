// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/abi/materialization.h"

#include "iree/vm/module.h"
#include "loom/codegen/low/builder.h"
#include "loom/codegen/low/function.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/arch/vm/abi/layout.h"
#include "loom/target/arch/vm/descriptors.h"
#include "loom/target/types.h"
#include "loom/util/walk.h"

static const loom_pass_info_t loom_vm_materialize_call_abi_pass_info_storage = {
    .name = IREE_SVL("vm-materialize-call-abi"),
    .description = IREE_SVL("Materialize VM call ABI register boundaries."),
    .kind = LOOM_PASS_MODULE,
};

const loom_pass_info_t* loom_vm_materialize_call_abi_pass_info(void) {
  return &loom_vm_materialize_call_abi_pass_info_storage;
}

static iree_status_t loom_vm_call_abi_validate_function(
    loom_module_t* module, loom_func_like_t function,
    const loom_vm_call_abi_layout_t* layout) {
  uint16_t argument_count = 0;
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(function, &argument_count);
  uint16_t result_count = 0;
  const loom_value_id_t* results =
      loom_low_function_result_ids(function.op, &result_count);
  if (argument_count != layout->arguments.field_count ||
      result_count != layout->results.field_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "VM call ABI materialization requires an unmaterialized physical "
        "signature matching the logical ABI layout");
  }

  const loom_type_t* logical_argument_types =
      loom_type_func_arg_types(layout->signature);
  for (uint16_t i = 0; i < argument_count; ++i) {
    if (!loom_type_equal(loom_module_value_type(module, arguments[i]),
                         logical_argument_types[i])) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "VM physical argument %u does not match its logical ABI type",
          (unsigned)i);
    }
    if (layout->arguments.fields[i].bank_ordinal >=
            IREE_VM_CALL_DIRECT_REGISTER_COUNT &&
        loom_module_value_has_predicate_attribute_uses(module, arguments[i])) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "VM argument overflow requires function predicates to be "
          "materialized first");
    }
  }

  const loom_type_t* logical_result_types =
      loom_type_func_result_types(layout->signature);
  for (uint16_t i = 0; i < result_count; ++i) {
    if (!loom_type_equal(loom_module_value_type(module, results[i]),
                         logical_result_types[i])) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "VM physical result %u does not match its logical ABI type",
          (unsigned)i);
    }
    if (layout->results.fields[i].bank_ordinal >=
            IREE_VM_CALL_DIRECT_REGISTER_COUNT &&
        loom_module_value_has_predicate_attribute_uses(module, results[i])) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "VM result overflow requires function predicates to be "
          "materialized first");
    }
  }

  return iree_ok_status();
}

static iree_status_t loom_vm_call_abi_build_logical_layout(
    loom_module_t* module, loom_func_like_t function,
    iree_arena_allocator_t* arena, loom_vm_call_abi_layout_t* out_layout) {
  loom_type_t logical_signature = loom_type_none();
  const loom_named_attr_slice_t abi_layout =
      loom_low_function_abi_layout(function.op);
  if (abi_layout.count != 0) {
    IREE_RETURN_IF_ERROR(loom_vm_call_abi_layout_resolve_signature(
        module, abi_layout, &logical_signature));
  } else {
    uint16_t argument_count = 0;
    const loom_value_id_t* arguments =
        loom_func_like_arg_ids(function, &argument_count);
    uint16_t result_count = 0;
    const loom_value_id_t* results =
        loom_low_function_result_ids(function.op, &result_count);
    const iree_host_size_t type_count =
        (iree_host_size_t)argument_count + result_count;
    loom_type_t* types = NULL;
    if (type_count != 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          arena, type_count, sizeof(*types), (void**)&types));
    }
    for (uint16_t i = 0; i < argument_count; ++i) {
      types[i] = loom_module_value_type(module, arguments[i]);
    }
    for (uint16_t i = 0; i < result_count; ++i) {
      types[argument_count + i] = loom_module_value_type(module, results[i]);
    }
    IREE_RETURN_IF_ERROR(loom_module_intern_function_type(
        module, types, argument_count,
        result_count != 0 ? types + argument_count : NULL, result_count,
        &logical_signature));
  }
  return loom_vm_call_abi_layout_build(module, logical_signature, arena,
                                       out_layout);
}

static const loom_low_descriptor_t* loom_vm_call_abi_descriptor(
    uint16_t descriptor_ordinal) {
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(loom_vm_core_descriptor_set(),
                                            descriptor_ordinal);
  IREE_ASSERT(descriptor != NULL);
  return descriptor;
}

static iree_status_t loom_vm_call_abi_build_overflow_instruction(
    loom_rewriter_t* rewriter, uint16_t descriptor_ordinal,
    const loom_value_id_t* operands, uint16_t operand_count,
    const loom_type_t* result_types, uint16_t result_count, uint16_t slot,
    loom_location_id_t location, loom_op_t** out_op) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_vm_core_descriptor_set();
  const loom_low_descriptor_t* descriptor =
      loom_vm_call_abi_descriptor(descriptor_ordinal);
  IREE_ASSERT_EQ(descriptor->immediate_count, 1u);
  const loom_low_immediate_t* immediate =
      &descriptor_set->immediates[descriptor->immediate_start];
  IREE_ASSERT_EQ(immediate->kind, LOOM_LOW_IMMEDIATE_KIND_ORDINAL);

  loom_named_attr_t slot_attr = {
      .value = loom_attr_i64(slot),
  };
  const iree_string_view_t immediate_name = loom_low_descriptor_set_string(
      descriptor_set, immediate->field_name_string_offset);
  IREE_RETURN_IF_ERROR(loom_builder_intern_string(
      &rewriter->builder, immediate_name, &slot_attr.name_id));
  return loom_low_build_resolved_descriptor_op(
      &rewriter->builder, descriptor_set, descriptor, operands, operand_count,
      loom_make_named_attr_slice(&slot_attr, 1), result_types, result_count,
      /*tied_results=*/NULL, /*tied_result_count=*/0, location, out_op);
}

static iree_status_t loom_vm_call_abi_build_overflow_argument_load(
    loom_rewriter_t* rewriter, const loom_vm_call_abi_field_layout_t* field,
    loom_type_t result_type, loom_location_id_t location,
    loom_value_id_t* out_result) {
  *out_result = LOOM_VALUE_ID_INVALID;
  uint16_t descriptor_ordinal = 0;
  switch (field->bank) {
    case LOOM_VM_CALL_ABI_BANK_VALUE:
      descriptor_ordinal = VM_CORE_DESCRIPTOR_REF_VALUE_ABI_ARGUMENT_LOAD;
      break;
    case LOOM_VM_CALL_ABI_BANK_REF:
      descriptor_ordinal = VM_CORE_DESCRIPTOR_REF_REF_ABI_ARGUMENT_LOAD_MOVE;
      break;
    case LOOM_VM_CALL_ABI_BANK_FUNCTION:
      descriptor_ordinal = VM_CORE_DESCRIPTOR_REF_FUNC_ABI_ARGUMENT_LOAD;
      break;
    default:
      IREE_ASSERT_UNREACHABLE("classified VM argument overflow bank");
      IREE_BUILTIN_UNREACHABLE();
  }

  loom_op_t* load_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vm_call_abi_build_overflow_instruction(
      rewriter, descriptor_ordinal, /*operands=*/NULL, /*operand_count=*/0,
      &result_type, /*result_count=*/1,
      (uint16_t)(field->bank_ordinal - IREE_VM_CALL_DIRECT_REGISTER_COUNT),
      location, &load_op));
  *out_result = loom_op_results(load_op)[0];
  return iree_ok_status();
}

static iree_status_t loom_vm_call_abi_build_overflow_result_store(
    loom_rewriter_t* rewriter, const loom_vm_call_abi_field_layout_t* field,
    loom_value_id_t source, loom_location_id_t location) {
  uint16_t descriptor_ordinal = 0;
  switch (field->bank) {
    case LOOM_VM_CALL_ABI_BANK_VALUE:
      descriptor_ordinal = VM_CORE_DESCRIPTOR_REF_VALUE_ABI_RESULT_STORE;
      break;
    case LOOM_VM_CALL_ABI_BANK_REF:
      descriptor_ordinal = VM_CORE_DESCRIPTOR_REF_REF_ABI_RESULT_STORE_MOVE;
      break;
    case LOOM_VM_CALL_ABI_BANK_FUNCTION:
      descriptor_ordinal = VM_CORE_DESCRIPTOR_REF_FUNC_ABI_RESULT_STORE;
      break;
    default:
      IREE_ASSERT_UNREACHABLE("classified VM result overflow bank");
      IREE_BUILTIN_UNREACHABLE();
  }

  loom_op_t* store_op = NULL;
  return loom_vm_call_abi_build_overflow_instruction(
      rewriter, descriptor_ordinal, &source, /*operand_count=*/1,
      /*result_types=*/NULL, /*result_count=*/0,
      (uint16_t)(field->bank_ordinal - IREE_VM_CALL_DIRECT_REGISTER_COUNT),
      location, &store_op);
}

static iree_status_t loom_vm_call_abi_build_ref_retain(
    loom_rewriter_t* rewriter, loom_value_id_t source, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op) {
  const loom_low_descriptor_t* descriptor =
      loom_vm_call_abi_descriptor(VM_CORE_DESCRIPTOR_REF_REF_RETAIN);
  IREE_ASSERT_EQ(descriptor->operand_count, 2u);
  IREE_ASSERT_EQ(descriptor->result_count, 1u);
  IREE_ASSERT_EQ(descriptor->immediate_count, 0u);
  return loom_low_build_resolved_descriptor_op(
      &rewriter->builder, loom_vm_core_descriptor_set(), descriptor, &source,
      /*operand_count=*/1, loom_named_attr_slice_empty(), &result_type,
      /*result_count=*/1, /*tied_results=*/NULL, /*tied_result_count=*/0,
      location, out_op);
}

static iree_status_t loom_vm_call_abi_build_transfer(
    loom_rewriter_t* rewriter, loom_value_id_t source, bool consume_ref,
    loom_location_id_t location, loom_op_t** out_transfer_op,
    loom_value_id_t* out_result) {
  *out_transfer_op = NULL;
  *out_result = LOOM_VALUE_ID_INVALID;
  const loom_type_t type = loom_module_value_type(rewriter->module, source);
  loom_vm_call_abi_bank_t bank = LOOM_VM_CALL_ABI_BANK_NONE;
  IREE_RETURN_IF_ERROR(
      loom_vm_call_abi_classify_type(rewriter->module, type, &bank));
  if (bank == LOOM_VM_CALL_ABI_BANK_REF && consume_ref) {
    IREE_RETURN_IF_ERROR(loom_low_move_build(&rewriter->builder, source,
                                             /*detached=*/false, type, location,
                                             out_transfer_op));
    *out_result = loom_low_move_result(*out_transfer_op);
  } else if (bank == LOOM_VM_CALL_ABI_BANK_REF) {
    IREE_RETURN_IF_ERROR(loom_vm_call_abi_build_ref_retain(
        rewriter, source, type, location, out_transfer_op));
    *out_result = loom_op_results(*out_transfer_op)[0];
  } else {
    IREE_RETURN_IF_ERROR(loom_low_copy_build(&rewriter->builder, source,
                                             /*detached=*/false, type, location,
                                             out_transfer_op));
    *out_result = loom_low_copy_result(*out_transfer_op);
  }
  return iree_ok_status();
}

static bool loom_vm_call_abi_side_has_overflow(
    const loom_vm_call_abi_side_layout_t* layout) {
  return layout->bank_counts.value > IREE_VM_CALL_DIRECT_REGISTER_COUNT ||
         layout->bank_counts.ref > IREE_VM_CALL_DIRECT_REGISTER_COUNT ||
         layout->bank_counts.function > IREE_VM_CALL_DIRECT_REGISTER_COUNT;
}

static iree_status_t loom_vm_call_abi_materialize_entry(
    loom_rewriter_t* rewriter, loom_func_like_t function,
    const loom_vm_call_abi_side_layout_t* layout) {
  uint16_t argument_count = 0;
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(function, &argument_count);
  if (argument_count == 0) return iree_ok_status();

  loom_region_t* body = loom_low_function_body(function.op);
  loom_block_t* entry_block = loom_region_entry_block(body);
  loom_builder_set_before(&rewriter->builder, entry_block->first_op);
  for (uint16_t i = 0; i < argument_count; ++i) {
    const loom_vm_call_abi_field_layout_t* field = &layout->fields[i];
    if (field->bank_ordinal < IREE_VM_CALL_DIRECT_REGISTER_COUNT) {
      loom_op_t* transfer_op = NULL;
      loom_value_id_t transferred = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_vm_call_abi_build_transfer(
          rewriter, arguments[i], /*consume_ref=*/true, function.op->location,
          &transfer_op, &transferred));
      IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_except(
          rewriter, arguments[i], transferred, transfer_op));
      continue;
    }

    loom_value_id_t loaded = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vm_call_abi_build_overflow_argument_load(
        rewriter, field, loom_module_value_type(rewriter->module, arguments[i]),
        function.op->location, &loaded));
    IREE_RETURN_IF_ERROR(
        loom_rewriter_move_value_name(rewriter, arguments[i], loaded));
    IREE_RETURN_IF_ERROR(
        loom_rewriter_replace_all_uses_with(rewriter, arguments[i], loaded));
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_call_abi_materialize_moved_operands(
    loom_rewriter_t* rewriter, loom_op_t* op, uint16_t operand_base,
    loom_value_slice_t operands) {
  if (operands.count == 0) return iree_ok_status();
  loom_builder_set_before(&rewriter->builder, op);
  for (uint16_t i = 0; i < operands.count; ++i) {
    loom_op_t* transfer_op = NULL;
    loom_value_id_t transferred = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vm_call_abi_build_transfer(
        rewriter, operands.values[i], /*consume_ref=*/true, op->location,
        &transfer_op, &transferred));
    IREE_RETURN_IF_ERROR(loom_rewriter_set_operand(
        rewriter, op, (uint16_t)(operand_base + i), transferred));
  }
  return iree_ok_status();
}

static bool loom_vm_call_abi_ref_requires_retain(const loom_module_t* module,
                                                 loom_value_id_t value_id) {
  const loom_value_t* value = loom_module_value(module, value_id);
  return !loom_value_has_single_use(value) ||
         loom_value_has_attribute_uses(value) ||
         loom_module_value_has_type_uses(module, value_id);
}

static iree_status_t loom_vm_call_abi_materialize_call_operands(
    loom_rewriter_t* rewriter, loom_op_t* op, uint16_t operand_base,
    loom_value_slice_t operands) {
  if (operands.count == 0) return iree_ok_status();
  loom_builder_set_before(&rewriter->builder, op);
  for (uint16_t i = 0; i < operands.count; ++i) {
    const loom_value_id_t source = operands.values[i];
    loom_vm_call_abi_bank_t bank = LOOM_VM_CALL_ABI_BANK_NONE;
    IREE_RETURN_IF_ERROR(loom_vm_call_abi_classify_type(
        rewriter->module, loom_module_value_type(rewriter->module, source),
        &bank));
    const bool consume_ref =
        bank != LOOM_VM_CALL_ABI_BANK_REF ||
        !loom_vm_call_abi_ref_requires_retain(rewriter->module, source);
    loom_op_t* transfer_op = NULL;
    loom_value_id_t transferred = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vm_call_abi_build_transfer(
        rewriter, source, consume_ref, op->location, &transfer_op,
        &transferred));
    IREE_RETURN_IF_ERROR(loom_rewriter_set_operand(
        rewriter, op, (uint16_t)(operand_base + i), transferred));
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_call_abi_materialize_results(
    loom_rewriter_t* rewriter, loom_op_t* op, loom_value_slice_t results) {
  if (results.count == 0) return iree_ok_status();
  loom_builder_set_after(&rewriter->builder, op);
  for (uint16_t i = 0; i < results.count; ++i) {
    loom_op_t* transfer_op = NULL;
    loom_value_id_t transferred = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vm_call_abi_build_transfer(
        rewriter, results.values[i], /*consume_ref=*/true, op->location,
        &transfer_op, &transferred));
    IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_except(
        rewriter, results.values[i], transferred, transfer_op));
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_call_abi_clear_overflow_call_purity(
    loom_rewriter_t* rewriter, loom_op_t* call_op) {
  if (loom_low_func_call_purity(call_op) == 0) return iree_ok_status();
  const loom_value_slice_t arguments = loom_low_func_call_operands(call_op);
  const loom_value_slice_t results = loom_low_func_call_results(call_op);
  loom_vm_call_abi_packet_layout_t layout = {0};
  IREE_RETURN_IF_ERROR(loom_vm_call_abi_packet_layout_build(
      rewriter->module, arguments.values, arguments.count, results.values,
      results.count, &layout));
  if (layout.local_byte_length == 0 && layout.local_ref_count == 0 &&
      layout.local_function_count == 0) {
    return iree_ok_status();
  }
  // Overflow packet stores make the physical Low call observably effectful
  // even when its logical source-level callable contract is pure.
  return loom_rewriter_set_attr(rewriter, call_op,
                                loom_low_func_call_purity_ATTR_INDEX,
                                loom_attr_absent());
}

typedef struct loom_vm_call_abi_materialize_walk_t {
  // Rewriter used to insert boundary copies and update uses.
  loom_rewriter_t* rewriter;
  // Logical callable layout for the function being rewritten.
  const loom_vm_call_abi_layout_t* layout;
} loom_vm_call_abi_materialize_walk_t;

static iree_status_t loom_vm_call_abi_materialize_return(
    loom_vm_call_abi_materialize_walk_t* walk, loom_op_t* return_op) {
  const loom_value_slice_t values = loom_low_return_values(return_op);
  loom_builder_set_before(&walk->rewriter->builder, return_op);
  for (uint16_t i = 0; i < values.count; ++i) {
    const loom_vm_call_abi_field_layout_t* field =
        &walk->layout->results.fields[i];
    if (field->bank_ordinal < IREE_VM_CALL_DIRECT_REGISTER_COUNT) {
      loom_op_t* transfer_op = NULL;
      loom_value_id_t transferred = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_vm_call_abi_build_transfer(
          walk->rewriter, values.values[i], /*consume_ref=*/true,
          return_op->location, &transfer_op, &transferred));
      IREE_RETURN_IF_ERROR(
          loom_rewriter_set_operand(walk->rewriter, return_op, i, transferred));
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_vm_call_abi_build_overflow_result_store(
        walk->rewriter, field, values.values[i], return_op->location));
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_call_abi_materialize_op(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  *out_result = LOOM_WALK_CONTINUE;
  loom_vm_call_abi_materialize_walk_t* walk =
      (loom_vm_call_abi_materialize_walk_t*)user_data;
  if (loom_low_func_call_isa(op)) {
    IREE_RETURN_IF_ERROR(loom_vm_call_abi_materialize_call_operands(
        walk->rewriter, op, /*operand_base=*/0,
        loom_low_func_call_operands(op)));
    IREE_RETURN_IF_ERROR(loom_vm_call_abi_materialize_results(
        walk->rewriter, op, loom_low_func_call_results(op)));
    return loom_vm_call_abi_clear_overflow_call_purity(walk->rewriter, op);
  }
  if (loom_low_func_call_indirect_isa(op)) {
    IREE_RETURN_IF_ERROR(loom_vm_call_abi_materialize_call_operands(
        walk->rewriter, op, /*operand_base=*/1,
        loom_low_func_call_indirect_operands(op)));
    return loom_vm_call_abi_materialize_results(
        walk->rewriter, op, loom_low_func_call_indirect_results(op));
  }
  if (loom_low_return_isa(op)) {
    if (!loom_vm_call_abi_side_has_overflow(&walk->layout->results)) {
      return loom_vm_call_abi_materialize_moved_operands(
          walk->rewriter, op, /*operand_base=*/0, loom_low_return_values(op));
    }
    return loom_vm_call_abi_materialize_return(walk, op);
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_call_abi_materialize_function(
    const loom_module_t* module, loom_func_like_t function,
    const loom_vm_call_abi_layout_t* layout,
    loom_vm_call_abi_materialize_walk_t* walk) {
  walk->layout = layout;
  IREE_RETURN_IF_ERROR(loom_vm_call_abi_materialize_entry(
      walk->rewriter, function, &layout->arguments));
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  IREE_RETURN_IF_ERROR(loom_walk_function(
      module, function, LOOM_WALK_PRE_ORDER,
      (loom_walk_callback_t){loom_vm_call_abi_materialize_op, walk},
      walk->rewriter->arena, &walk_result));
  if (loom_low_func_def_isa(function.op) &&
      (loom_vm_call_abi_side_has_overflow(&layout->arguments) ||
       loom_vm_call_abi_side_has_overflow(&layout->results)) &&
      loom_func_like_purity(function) != 0) {
    // ABI packet loads and stores are physical effects hidden by the logical
    // source contract and therefore invalidate Low-level purity.
    IREE_RETURN_IF_ERROR(loom_rewriter_set_attr(
        walk->rewriter, function.op, loom_low_func_def_purity_ATTR_INDEX,
        loom_attr_absent()));
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_call_abi_preserve_logical_signature(
    loom_rewriter_t* rewriter, loom_func_like_t function) {
  uint16_t argument_count = 0;
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(function, &argument_count);
  uint16_t result_count = 0;
  const loom_value_id_t* results =
      loom_low_function_result_ids(function.op, &result_count);
  const loom_symbol_ref_t function_ref = loom_func_like_callee(function);
  const bool has_presentation =
      loom_symbol_ref_is_valid(function_ref) && function_ref.module_id == 0 &&
      function_ref.symbol_id < rewriter->module->symbols.count &&
      !iree_string_view_is_empty(loom_func_like_export_name(
          rewriter->module,
          &rewriter->module->symbols.entries[function_ref.symbol_id],
          function));

  const loom_named_attr_slice_t abi_layout =
      loom_low_function_abi_layout(function.op);
  const uint16_t abi_layout_attr_index =
      loom_low_function_abi_layout_attr_index(function.op);
  IREE_ASSERT_NE(abi_layout_attr_index, LOOM_ATTR_INDEX_NONE);
  if (abi_layout.count != 0) {
    if (!has_presentation) return iree_ok_status();
    bool layout_changed = false;
    loom_attribute_t layout_attr = loom_attr_absent();
    IREE_RETURN_IF_ERROR(loom_vm_call_abi_layout_preserve_presentation_names(
        rewriter->module, abi_layout,
        (loom_vm_call_abi_source_fields_t){
            .values = arguments,
            .count = argument_count,
        },
        (loom_vm_call_abi_source_fields_t){
            .values = results,
            .count = result_count,
        },
        rewriter->arena, &layout_changed, &layout_attr));
    if (!layout_changed) return iree_ok_status();
    return loom_rewriter_set_attr(rewriter, function.op, abi_layout_attr_index,
                                  layout_attr);
  }

  iree_host_size_t type_count = 0;
  if (!iree_host_size_checked_add(argument_count, result_count, &type_count)) {
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
  for (uint16_t i = 0; i < result_count; ++i) {
    types[argument_count + i] =
        loom_module_value_type(rewriter->module, results[i]);
  }

  loom_attribute_t layout_attr = loom_attr_absent();
  const loom_type_t* result_types =
      result_count != 0 ? types + argument_count : NULL;
  IREE_RETURN_IF_ERROR(loom_vm_call_abi_layout_make_attr(
      rewriter->module,
      (loom_vm_call_abi_source_fields_t){
          .types = types,
          .values = has_presentation ? arguments : NULL,
          .count = argument_count,
      },
      (loom_vm_call_abi_source_fields_t){
          .types = result_types,
          .values = has_presentation ? results : NULL,
          .count = result_count,
      },
      rewriter->arena, &layout_attr));
  return loom_rewriter_set_attr(rewriter, function.op, abi_layout_attr_index,
                                layout_attr);
}

static bool loom_vm_call_abi_is_definition(const loom_module_t* module,
                                           loom_func_like_t function) {
  if (loom_low_func_def_isa(function.op)) {
    return loom_func_like_abi(function) == LOOM_TARGET_ABI_VM_FUNCTION;
  }
  if (!loom_low_kernel_def_isa(function.op)) return false;
  const loom_string_id_t descriptor_set_id =
      loom_low_kernel_def_descriptor_set(function.op);
  if (descriptor_set_id == LOOM_STRING_ID_INVALID ||
      descriptor_set_id >= module->strings.count) {
    return false;
  }
  const loom_low_descriptor_set_t* descriptor_set =
      loom_vm_core_descriptor_set();
  const iree_string_view_t expected_key = loom_low_descriptor_set_string(
      descriptor_set, descriptor_set->key_string_offset);
  return iree_string_view_equal(module->strings.entries[descriptor_set_id],
                                expected_key);
}

iree_status_t loom_vm_materialize_call_abi_run(loom_pass_t* pass,
                                               loom_module_t* module) {
  loom_rewriter_t rewriter = {0};
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, module, pass->arena));
  loom_vm_call_abi_materialize_walk_t walk = {
      .rewriter = &rewriter,
  };
  iree_status_t status = iree_ok_status();
  loom_block_t* module_block = loom_module_block(module);
  loom_op_t* op = NULL;
  loom_block_for_each_op(module_block, op) {
    loom_func_like_t function = loom_func_like_cast(module, op);
    if (!loom_vm_call_abi_is_definition(module, function)) {
      continue;
    }
    loom_vm_call_abi_layout_t layout = {0};
    status = loom_vm_call_abi_build_logical_layout(module, function,
                                                   pass->arena, &layout);
    if (iree_status_is_ok(status)) {
      status = loom_vm_call_abi_validate_function(module, function, &layout);
    }
    if (iree_status_is_ok(status)) {
      status = loom_vm_call_abi_preserve_logical_signature(&rewriter, function);
    }
    if (iree_status_is_ok(status)) {
      status = loom_vm_call_abi_materialize_function(module, function, &layout,
                                                     &walk);
    }
    if (!iree_status_is_ok(status)) break;
  }
  if (iree_status_is_ok(status) &&
      iree_any_bit_set(rewriter.flags, LOOM_REWRITER_FLAG_CHANGED)) {
    loom_pass_mark_changed(pass);
  }
  loom_rewriter_deinitialize(&rewriter);
  return status;
}
