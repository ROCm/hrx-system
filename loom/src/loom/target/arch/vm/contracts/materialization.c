// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/contracts/materialization.h"

#include <inttypes.h>

#include "iree/vm/bytecode/wire/core/conversion.h"
#include "iree/vm/bytecode/wire/core/float.h"
#include "iree/vm/bytecode/wire/core/integer.h"
#include "loom/codegen/low/builder.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/arch/vm/descriptors.h"
#include "loom/target/arch/vm/lower/constants.h"
#include "loom/target/arch/vm/lower/types.h"

static const loom_pass_info_t
    loom_vm_materialize_function_contracts_pass_info_storage = {
        .name = IREE_SVL("vm-materialize-function-contracts"),
        .description =
            IREE_SVL("Materialize VM function predicates as entry checks."),
        .kind = LOOM_PASS_FUNCTION,
};

const loom_pass_info_t* loom_vm_materialize_function_contracts_pass_info(void) {
  return &loom_vm_materialize_function_contracts_pass_info_storage;
}

static bool loom_vm_contract_scalar_is_integer(loom_scalar_type_t scalar_type) {
  return loom_scalar_type_is_integer(scalar_type) ||
         scalar_type == LOOM_SCALAR_TYPE_INDEX ||
         scalar_type == LOOM_SCALAR_TYPE_OFFSET;
}

static bool loom_vm_contract_value_is_argument(const loom_value_id_t* arguments,
                                               uint16_t argument_count,
                                               loom_value_id_t value) {
  for (uint16_t i = 0; i < argument_count; ++i) {
    if (arguments[i] == value) return true;
  }
  return false;
}

static iree_status_t loom_vm_contract_value_scalar_type(
    const loom_module_t* module, loom_value_id_t value,
    loom_scalar_type_t* out_scalar_type) {
  if (value >= module->values.count ||
      !loom_vm_value_register_scalar_type(loom_module_value_type(module, value),
                                          out_scalar_type)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "VM function predicate value %u must be one scalar vm.value register",
        value);
  }
  return iree_ok_status();
}

static bool loom_vm_contract_predicate_is_float(
    loom_predicate_kind_t predicate_kind) {
  return predicate_kind == LOOM_PREDICATE_NOT_NAN ||
         predicate_kind == LOOM_PREDICATE_NOT_INF ||
         predicate_kind == LOOM_PREDICATE_FINITE;
}

static iree_status_t loom_vm_contract_preflight_predicate(
    const loom_module_t* module, const loom_value_id_t* arguments,
    uint16_t argument_count, const loom_predicate_t* predicate) {
  const char* predicate_name = loom_predicate_kind_name(predicate->kind);
  if (predicate->arg_count == 0 ||
      predicate->arg_tags[0] != LOOM_PRED_ARG_VALUE) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "VM function predicate '%s' must constrain a function argument",
        predicate_name ? predicate_name : "<unknown>");
  }

  const bool is_float = loom_vm_contract_predicate_is_float(predicate->kind);
  for (uint8_t i = 0; i < predicate->arg_count; ++i) {
    if (predicate->arg_tags[i] == LOOM_PRED_ARG_CONST) {
      if (!is_float) continue;
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "VM floating-point predicate '%s' cannot use an integer literal",
          predicate_name ? predicate_name : "<unknown>");
    }
    if (predicate->arg_tags[i] != LOOM_PRED_ARG_VALUE ||
        predicate->args[i] < 0 || predicate->args[i] > UINT32_MAX) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "VM function predicate '%s' argument %u is not a supported value",
          predicate_name ? predicate_name : "<unknown>", (unsigned)i);
    }
    const loom_value_id_t value = (loom_value_id_t)predicate->args[i];
    if (!loom_vm_contract_value_is_argument(arguments, argument_count, value)) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "VM function predicate '%s' value %u is not a function argument",
          predicate_name ? predicate_name : "<unknown>", value);
    }
    loom_scalar_type_t scalar_type = 0;
    IREE_RETURN_IF_ERROR(
        loom_vm_contract_value_scalar_type(module, value, &scalar_type));
    const bool type_supported =
        is_float ? loom_scalar_type_is_float(scalar_type)
                 : loom_vm_contract_scalar_is_integer(scalar_type);
    if (!type_supported) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "VM function predicate '%s' argument %u has an unsupported scalar "
          "type",
          predicate_name ? predicate_name : "<unknown>", (unsigned)i);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_contract_preflight(
    const loom_module_t* module, loom_func_like_t function,
    const loom_predicate_t* predicates, uint16_t predicate_count) {
  uint16_t argument_count = 0;
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(function, &argument_count);
  for (uint16_t i = 0; i < predicate_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_vm_contract_preflight_predicate(
        module, arguments, argument_count, &predicates[i]));
  }
  return iree_ok_status();
}

typedef struct loom_vm_contract_materializer_t {
  // Rewriter owning insertion and function-attribute mutation.
  loom_rewriter_t* rewriter;
  // Core VM descriptor set used for every emitted instruction.
  const loom_low_descriptor_set_t* descriptor_set;
  // Typed VM register carrying a canonical boolean.
  loom_type_t i1_type;
  // Typed VM register carrying a 32-bit integer.
  loom_type_t i32_type;
  // Typed VM register carrying a 64-bit integer.
  loom_type_t i64_type;
  // Typed VM register carrying an f32 value.
  loom_type_t f32_type;
  // Typed VM ref register carrying a logical buffer.
  loom_type_t buffer_ref_type;
  // Shared null diagnostic message emitted before the first assertion.
  loom_value_id_t null_message;
  // Source location assigned to all materialized checks.
  loom_location_id_t location;
} loom_vm_contract_materializer_t;

static const loom_low_descriptor_t* loom_vm_contract_descriptor(
    const loom_vm_contract_materializer_t* materializer,
    uint16_t descriptor_ordinal) {
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(materializer->descriptor_set,
                                            descriptor_ordinal);
  IREE_ASSERT(descriptor != NULL);
  return descriptor;
}

typedef struct loom_vm_contract_instruction_t {
  // Dense Core VM descriptor ordinal naming the physical instruction.
  uint16_t descriptor_ordinal;
  // Selector value, or UINT16_MAX when the instruction has no selector.
  uint16_t selector_value;
} loom_vm_contract_instruction_t;

static loom_vm_contract_instruction_t loom_vm_contract_instruction(
    uint16_t descriptor_ordinal) {
  return (loom_vm_contract_instruction_t){
      .descriptor_ordinal = descriptor_ordinal,
      .selector_value = UINT16_MAX,
  };
}

static loom_vm_contract_instruction_t loom_vm_contract_selected_instruction(
    uint16_t descriptor_ordinal, uint8_t selector_value) {
  return (loom_vm_contract_instruction_t){
      .descriptor_ordinal = descriptor_ordinal,
      .selector_value = selector_value,
  };
}

static iree_status_t loom_vm_contract_build_value(
    loom_vm_contract_materializer_t* materializer,
    loom_vm_contract_instruction_t instruction, const loom_value_id_t* operands,
    uint16_t operand_count, loom_type_t result_type,
    loom_value_id_t* out_result) {
  *out_result = LOOM_VALUE_ID_INVALID;
  const loom_low_descriptor_t* descriptor =
      loom_vm_contract_descriptor(materializer, instruction.descriptor_ordinal);
  loom_named_attr_t selector_attr = {0};
  loom_named_attr_slice_t attrs = loom_named_attr_slice_empty();
  if (instruction.selector_value != UINT16_MAX) {
    IREE_ASSERT_EQ(descriptor->immediate_count, 1);
    const loom_low_immediate_t* immediate =
        &materializer->descriptor_set->immediates[descriptor->immediate_start];
    IREE_ASSERT_EQ(immediate->kind, LOOM_LOW_IMMEDIATE_KIND_ENUM);
    const iree_string_view_t immediate_name = loom_low_descriptor_set_string(
        materializer->descriptor_set, immediate->field_name_string_offset);
    IREE_RETURN_IF_ERROR(
        loom_builder_intern_string(&materializer->rewriter->builder,
                                   immediate_name, &selector_attr.name_id));
    selector_attr.value = loom_attr_i64(instruction.selector_value);
    attrs = loom_make_named_attr_slice(&selector_attr, 1);
  }

  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      &materializer->rewriter->builder, materializer->descriptor_set,
      descriptor, operands, operand_count, attrs, &result_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, materializer->location,
      &op));
  *out_result = loom_op_results(op)[0];
  return iree_ok_status();
}

static iree_status_t loom_vm_contract_build_selected_unary(
    loom_vm_contract_materializer_t* materializer, uint16_t descriptor_ordinal,
    uint8_t selector_value, loom_value_id_t operand, loom_type_t result_type,
    loom_value_id_t* out_result) {
  return loom_vm_contract_build_value(
      materializer,
      loom_vm_contract_selected_instruction(descriptor_ordinal, selector_value),
      &operand, 1, result_type, out_result);
}

static iree_status_t loom_vm_contract_build_binary(
    loom_vm_contract_materializer_t* materializer, uint16_t descriptor_ordinal,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_type_t result_type,
    loom_value_id_t* out_result) {
  const loom_value_id_t operands[] = {lhs, rhs};
  return loom_vm_contract_build_value(
      materializer, loom_vm_contract_instruction(descriptor_ordinal), operands,
      IREE_ARRAYSIZE(operands), result_type, out_result);
}

static iree_status_t loom_vm_contract_build_selected_binary(
    loom_vm_contract_materializer_t* materializer, uint16_t descriptor_ordinal,
    uint8_t selector_value, loom_value_id_t lhs, loom_value_id_t rhs,
    loom_type_t result_type, loom_value_id_t* out_result) {
  const loom_value_id_t operands[] = {lhs, rhs};
  return loom_vm_contract_build_value(
      materializer,
      loom_vm_contract_selected_instruction(descriptor_ordinal, selector_value),
      operands, IREE_ARRAYSIZE(operands), result_type, out_result);
}

static iree_status_t loom_vm_contract_build_assert(
    loom_vm_contract_materializer_t* materializer, loom_value_id_t condition) {
  if (materializer->null_message == LOOM_VALUE_ID_INVALID) {
    loom_op_t* null_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
        &materializer->rewriter->builder, materializer->descriptor_set,
        loom_vm_contract_descriptor(materializer,
                                    VM_CORE_DESCRIPTOR_REF_REF_NULL),
        /*operands=*/NULL, /*operand_count=*/0, loom_named_attr_slice_empty(),
        &materializer->buffer_ref_type, /*result_count=*/1,
        /*tied_results=*/NULL, /*tied_result_count=*/0, materializer->location,
        &null_op));
    materializer->null_message = loom_op_results(null_op)[0];
  }
  const loom_value_id_t operands[] = {condition, materializer->null_message};
  loom_op_t* op = NULL;
  return loom_low_build_resolved_descriptor_op(
      &materializer->rewriter->builder, materializer->descriptor_set,
      loom_vm_contract_descriptor(materializer,
                                  VM_CORE_DESCRIPTOR_REF_CONTROL_ASSERT),
      operands, IREE_ARRAYSIZE(operands), loom_named_attr_slice_empty(),
      /*result_types=*/NULL,
      /*result_count=*/0, /*tied_results=*/NULL, /*tied_result_count=*/0,
      materializer->location, &op);
}

static iree_status_t loom_vm_contract_build_integer_constant(
    loom_vm_contract_materializer_t* materializer, int width, int64_t value,
    loom_value_id_t* out_result) {
  const uint64_t bits = width == 32 ? (uint32_t)value : (uint64_t)value;
  const loom_type_t result_type =
      width == 32 ? materializer->i32_type : materializer->i64_type;
  return loom_vm_inline_constant_build(&materializer->rewriter->builder, bits,
                                       result_type, materializer->location,
                                       out_result);
}

static iree_status_t loom_vm_contract_normalize_integer_value(
    loom_vm_contract_materializer_t* materializer, loom_value_id_t value,
    loom_scalar_type_t scalar_type, int width, loom_value_id_t* out_result) {
  *out_result = value;
  uint16_t first_selector = UINT16_MAX;
  loom_type_t first_result_type = loom_type_none();
  switch (scalar_type) {
    case LOOM_SCALAR_TYPE_I1:
      if (width == 64) {
        first_selector = IREE_VM_ISA_INTEGER_CONVERT_U32_TO_I64;
        first_result_type = materializer->i64_type;
      }
      break;
    case LOOM_SCALAR_TYPE_I8:
      first_selector = IREE_VM_ISA_INTEGER_CONVERT_S8_TO_I32;
      first_result_type = materializer->i32_type;
      break;
    case LOOM_SCALAR_TYPE_I16:
      first_selector = IREE_VM_ISA_INTEGER_CONVERT_S16_TO_I32;
      first_result_type = materializer->i32_type;
      break;
    case LOOM_SCALAR_TYPE_I32:
      if (width == 64) {
        first_selector = IREE_VM_ISA_INTEGER_CONVERT_S32_TO_I64;
        first_result_type = materializer->i64_type;
      }
      break;
    case LOOM_SCALAR_TYPE_I64:
    case LOOM_SCALAR_TYPE_INDEX:
    case LOOM_SCALAR_TYPE_OFFSET:
      IREE_ASSERT_EQ(width, 64);
      break;
    default:
      IREE_ASSERT_UNREACHABLE("preflighted VM integer predicate type");
      IREE_BUILTIN_UNREACHABLE();
  }

  if (first_selector != UINT16_MAX) {
    IREE_RETURN_IF_ERROR(loom_vm_contract_build_selected_unary(
        materializer, VM_CORE_DESCRIPTOR_REF_CONVERSION_INTEGER,
        (uint8_t)first_selector, value, first_result_type, out_result));
  }
  if (width == 64 && (scalar_type == LOOM_SCALAR_TYPE_I8 ||
                      scalar_type == LOOM_SCALAR_TYPE_I16)) {
    return loom_vm_contract_build_selected_unary(
        materializer, VM_CORE_DESCRIPTOR_REF_CONVERSION_INTEGER,
        IREE_VM_ISA_INTEGER_CONVERT_S32_TO_I64, *out_result,
        materializer->i64_type, out_result);
  }
  return iree_ok_status();
}

static int loom_vm_contract_integer_width(const loom_module_t* module,
                                          const loom_predicate_t* predicate) {
  for (uint8_t i = 0; i < predicate->arg_count; ++i) {
    if (predicate->arg_tags[i] == LOOM_PRED_ARG_CONST) {
      if (predicate->args[i] < INT32_MIN || predicate->args[i] > INT32_MAX) {
        return 64;
      }
      continue;
    }
    loom_scalar_type_t scalar_type = 0;
    const loom_value_id_t value = (loom_value_id_t)predicate->args[i];
    const bool has_scalar_type = loom_vm_value_register_scalar_type(
        loom_module_value_type(module, value), &scalar_type);
    IREE_ASSERT(has_scalar_type);
    (void)has_scalar_type;
    if (scalar_type == LOOM_SCALAR_TYPE_I64 ||
        scalar_type == LOOM_SCALAR_TYPE_INDEX ||
        scalar_type == LOOM_SCALAR_TYPE_OFFSET) {
      return 64;
    }
  }
  return 32;
}

static iree_status_t loom_vm_contract_build_integer_arg(
    loom_vm_contract_materializer_t* materializer,
    const loom_predicate_t* predicate, uint8_t argument_index, int width,
    loom_value_id_t* out_result) {
  if (predicate->arg_tags[argument_index] == LOOM_PRED_ARG_CONST) {
    return loom_vm_contract_build_integer_constant(
        materializer, width, predicate->args[argument_index], out_result);
  }
  const loom_value_id_t value =
      (loom_value_id_t)predicate->args[argument_index];
  loom_scalar_type_t scalar_type = 0;
  IREE_RETURN_IF_ERROR(loom_vm_contract_value_scalar_type(
      materializer->rewriter->module, value, &scalar_type));
  return loom_vm_contract_normalize_integer_value(
      materializer, value, scalar_type, width, out_result);
}

enum loom_vm_contract_comparison_e {
  LOOM_VM_CONTRACT_COMPARISON_EQ = 0,
  LOOM_VM_CONTRACT_COMPARISON_NE = 1,
  LOOM_VM_CONTRACT_COMPARISON_LT = 2,
  LOOM_VM_CONTRACT_COMPARISON_LE = 3,
  LOOM_VM_CONTRACT_COMPARISON_GT = 4,
  LOOM_VM_CONTRACT_COMPARISON_GE = 5,
  LOOM_VM_CONTRACT_COMPARISON_COUNT = 6,
};

static const uint8_t
    kVmContractComparisonSelectors[LOOM_VM_CONTRACT_COMPARISON_COUNT] = {
        IREE_VM_ISA_INTEGER_COMPARE_EQ,  IREE_VM_ISA_INTEGER_COMPARE_NE,
        IREE_VM_ISA_INTEGER_COMPARE_SLT, IREE_VM_ISA_INTEGER_COMPARE_SLE,
        IREE_VM_ISA_INTEGER_COMPARE_SGT, IREE_VM_ISA_INTEGER_COMPARE_SGE,
};

static iree_status_t loom_vm_contract_build_comparison(
    loom_vm_contract_materializer_t* materializer, int width,
    enum loom_vm_contract_comparison_e comparison, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_value_id_t* out_condition) {
  const uint16_t descriptor_ordinal =
      width == 32 ? VM_CORE_DESCRIPTOR_REF_INTEGER_COMPARE_I32
                  : VM_CORE_DESCRIPTOR_REF_INTEGER_COMPARE_I64;
  return loom_vm_contract_build_selected_binary(
      materializer, descriptor_ordinal,
      kVmContractComparisonSelectors[comparison], lhs, rhs,
      materializer->i1_type, out_condition);
}

static enum loom_vm_contract_comparison_e loom_vm_contract_relation_comparison(
    loom_predicate_kind_t predicate_kind) {
  switch (predicate_kind) {
    case LOOM_PREDICATE_EQ:
      return LOOM_VM_CONTRACT_COMPARISON_EQ;
    case LOOM_PREDICATE_NE:
      return LOOM_VM_CONTRACT_COMPARISON_NE;
    case LOOM_PREDICATE_LT:
      return LOOM_VM_CONTRACT_COMPARISON_LT;
    case LOOM_PREDICATE_LE:
    case LOOM_PREDICATE_MAX:
      return LOOM_VM_CONTRACT_COMPARISON_LE;
    case LOOM_PREDICATE_GT:
      return LOOM_VM_CONTRACT_COMPARISON_GT;
    case LOOM_PREDICATE_GE:
    case LOOM_PREDICATE_MIN:
      return LOOM_VM_CONTRACT_COMPARISON_GE;
    default:
      IREE_ASSERT_UNREACHABLE("relational VM function predicate");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t loom_vm_contract_materialize_relation(
    loom_vm_contract_materializer_t* materializer,
    const loom_predicate_t* predicate) {
  const int width =
      loom_vm_contract_integer_width(materializer->rewriter->module, predicate);
  loom_value_id_t lhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t rhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vm_contract_build_integer_arg(
      materializer, predicate, 0, width, &lhs));
  IREE_RETURN_IF_ERROR(loom_vm_contract_build_integer_arg(
      materializer, predicate, 1, width, &rhs));
  loom_value_id_t condition = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vm_contract_build_comparison(
      materializer, width,
      loom_vm_contract_relation_comparison(predicate->kind), lhs, rhs,
      &condition));
  return loom_vm_contract_build_assert(materializer, condition);
}

static iree_status_t loom_vm_contract_materialize_range(
    loom_vm_contract_materializer_t* materializer,
    const loom_predicate_t* predicate) {
  const int width =
      loom_vm_contract_integer_width(materializer->rewriter->module, predicate);
  loom_value_id_t value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t lower = LOOM_VALUE_ID_INVALID;
  loom_value_id_t upper = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vm_contract_build_integer_arg(
      materializer, predicate, 0, width, &value));
  IREE_RETURN_IF_ERROR(loom_vm_contract_build_integer_arg(
      materializer, predicate, 1, width, &lower));
  IREE_RETURN_IF_ERROR(loom_vm_contract_build_integer_arg(
      materializer, predicate, 2, width, &upper));

  loom_value_id_t condition = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vm_contract_build_comparison(
      materializer, width, LOOM_VM_CONTRACT_COMPARISON_GE, value, lower,
      &condition));
  IREE_RETURN_IF_ERROR(loom_vm_contract_build_assert(materializer, condition));
  IREE_RETURN_IF_ERROR(loom_vm_contract_build_comparison(
      materializer, width, LOOM_VM_CONTRACT_COMPARISON_LE, value, upper,
      &condition));
  return loom_vm_contract_build_assert(materializer, condition);
}

static iree_status_t loom_vm_contract_materialize_multiple(
    loom_vm_contract_materializer_t* materializer,
    const loom_predicate_t* predicate) {
  const int width =
      loom_vm_contract_integer_width(materializer->rewriter->module, predicate);
  loom_value_id_t value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t divisor = LOOM_VALUE_ID_INVALID;
  loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vm_contract_build_integer_arg(
      materializer, predicate, 0, width, &value));
  IREE_RETURN_IF_ERROR(loom_vm_contract_build_integer_arg(
      materializer, predicate, 1, width, &divisor));
  IREE_RETURN_IF_ERROR(
      loom_vm_contract_build_integer_constant(materializer, width, 0, &zero));

  loom_value_id_t condition = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vm_contract_build_comparison(
      materializer, width, LOOM_VM_CONTRACT_COMPARISON_NE, divisor, zero,
      &condition));
  IREE_RETURN_IF_ERROR(loom_vm_contract_build_assert(materializer, condition));

  const uint16_t remainder_descriptor =
      width == 32 ? VM_CORE_DESCRIPTOR_REF_INTEGER_REM_S32
                  : VM_CORE_DESCRIPTOR_REF_INTEGER_REM_S64;
  const loom_type_t integer_type =
      width == 32 ? materializer->i32_type : materializer->i64_type;
  loom_value_id_t remainder = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_vm_contract_build_binary(materializer, remainder_descriptor, value,
                                    divisor, integer_type, &remainder));
  IREE_RETURN_IF_ERROR(loom_vm_contract_build_comparison(
      materializer, width, LOOM_VM_CONTRACT_COMPARISON_EQ, remainder, zero,
      &condition));
  return loom_vm_contract_build_assert(materializer, condition);
}

static iree_status_t loom_vm_contract_materialize_power_of_two(
    loom_vm_contract_materializer_t* materializer,
    const loom_predicate_t* predicate) {
  const int width =
      loom_vm_contract_integer_width(materializer->rewriter->module, predicate);
  loom_value_id_t value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
  loom_value_id_t one = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vm_contract_build_integer_arg(
      materializer, predicate, 0, width, &value));
  IREE_RETURN_IF_ERROR(
      loom_vm_contract_build_integer_constant(materializer, width, 0, &zero));
  IREE_RETURN_IF_ERROR(
      loom_vm_contract_build_integer_constant(materializer, width, 1, &one));

  loom_value_id_t condition = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vm_contract_build_comparison(
      materializer, width, LOOM_VM_CONTRACT_COMPARISON_GT, value, zero,
      &condition));
  IREE_RETURN_IF_ERROR(loom_vm_contract_build_assert(materializer, condition));

  const uint16_t subtract_descriptor =
      width == 32 ? VM_CORE_DESCRIPTOR_REF_INTEGER_SUB_I32
                  : VM_CORE_DESCRIPTOR_REF_INTEGER_SUB_I64;
  const uint16_t and_descriptor = width == 32
                                      ? VM_CORE_DESCRIPTOR_REF_INTEGER_AND_I32
                                      : VM_CORE_DESCRIPTOR_REF_INTEGER_AND_I64;
  const loom_type_t integer_type =
      width == 32 ? materializer->i32_type : materializer->i64_type;
  loom_value_id_t previous = LOOM_VALUE_ID_INVALID;
  loom_value_id_t masked = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vm_contract_build_binary(
      materializer, subtract_descriptor, value, one, integer_type, &previous));
  IREE_RETURN_IF_ERROR(loom_vm_contract_build_binary(
      materializer, and_descriptor, value, previous, integer_type, &masked));
  IREE_RETURN_IF_ERROR(loom_vm_contract_build_comparison(
      materializer, width, LOOM_VM_CONTRACT_COMPARISON_EQ, masked, zero,
      &condition));
  return loom_vm_contract_build_assert(materializer, condition);
}

static iree_status_t loom_vm_contract_normalize_float_value(
    loom_vm_contract_materializer_t* materializer, loom_value_id_t value,
    loom_scalar_type_t scalar_type, loom_value_id_t* out_result,
    int* out_width) {
  *out_result = value;
  *out_width = 32;
  uint16_t conversion_selector = UINT16_MAX;
  switch (scalar_type) {
    case LOOM_SCALAR_TYPE_F8E4M3:
      conversion_selector = IREE_VM_ISA_FLOAT_EXTEND_F8E4M3_TO_F32;
      break;
    case LOOM_SCALAR_TYPE_F8E5M2:
      conversion_selector = IREE_VM_ISA_FLOAT_EXTEND_F8E5M2_TO_F32;
      break;
    case LOOM_SCALAR_TYPE_F16:
      conversion_selector = IREE_VM_ISA_FLOAT_EXTEND_F16_TO_F32;
      break;
    case LOOM_SCALAR_TYPE_BF16:
      conversion_selector = IREE_VM_ISA_FLOAT_EXTEND_BF16_TO_F32;
      break;
    case LOOM_SCALAR_TYPE_F32:
      break;
    case LOOM_SCALAR_TYPE_F64:
      *out_width = 64;
      break;
    default:
      IREE_ASSERT_UNREACHABLE("preflighted VM floating-point predicate type");
      IREE_BUILTIN_UNREACHABLE();
  }
  if (conversion_selector != UINT16_MAX) {
    return loom_vm_contract_build_selected_unary(
        materializer, VM_CORE_DESCRIPTOR_REF_CONVERSION_FLOAT_EXTEND,
        (uint8_t)conversion_selector, value, materializer->f32_type,
        out_result);
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_contract_materialize_float(
    loom_vm_contract_materializer_t* materializer,
    const loom_predicate_t* predicate) {
  const loom_value_id_t source = (loom_value_id_t)predicate->args[0];
  loom_scalar_type_t scalar_type = 0;
  IREE_RETURN_IF_ERROR(loom_vm_contract_value_scalar_type(
      materializer->rewriter->module, source, &scalar_type));
  loom_value_id_t value = LOOM_VALUE_ID_INVALID;
  int width = 0;
  IREE_RETURN_IF_ERROR(loom_vm_contract_normalize_float_value(
      materializer, source, scalar_type, &value, &width));

  uint8_t classify_selector = 0;
  switch (predicate->kind) {
    case LOOM_PREDICATE_NOT_NAN:
      classify_selector = IREE_VM_ISA_FLOAT_CLASSIFY_ISNAN;
      break;
    case LOOM_PREDICATE_NOT_INF:
      classify_selector = IREE_VM_ISA_FLOAT_CLASSIFY_ISINF;
      break;
    case LOOM_PREDICATE_FINITE:
      classify_selector = IREE_VM_ISA_FLOAT_CLASSIFY_ISFINITE;
      break;
    default:
      IREE_ASSERT_UNREACHABLE("floating-point VM function predicate");
      IREE_BUILTIN_UNREACHABLE();
  }

  loom_value_id_t condition = LOOM_VALUE_ID_INVALID;
  const uint16_t classify_descriptor =
      width == 32 ? VM_CORE_DESCRIPTOR_REF_FLOAT_CLASSIFY_F32
                  : VM_CORE_DESCRIPTOR_REF_FLOAT_CLASSIFY_F64;
  IREE_RETURN_IF_ERROR(loom_vm_contract_build_selected_unary(
      materializer, classify_descriptor, classify_selector, value,
      materializer->i1_type, &condition));
  if (predicate->kind != LOOM_PREDICATE_FINITE) {
    loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_vm_contract_build_integer_constant(materializer, 32, 0, &zero));
    IREE_RETURN_IF_ERROR(loom_vm_contract_build_comparison(
        materializer, 32, LOOM_VM_CONTRACT_COMPARISON_EQ, condition, zero,
        &condition));
  }
  return loom_vm_contract_build_assert(materializer, condition);
}

static iree_status_t loom_vm_contract_materialize_predicate(
    loom_vm_contract_materializer_t* materializer,
    const loom_predicate_t* predicate) {
  switch ((loom_predicate_kind_t)predicate->kind) {
    case LOOM_PREDICATE_EQ:
    case LOOM_PREDICATE_NE:
    case LOOM_PREDICATE_LT:
    case LOOM_PREDICATE_LE:
    case LOOM_PREDICATE_GT:
    case LOOM_PREDICATE_GE:
    case LOOM_PREDICATE_MIN:
    case LOOM_PREDICATE_MAX:
      return loom_vm_contract_materialize_relation(materializer, predicate);
    case LOOM_PREDICATE_MUL:
      return loom_vm_contract_materialize_multiple(materializer, predicate);
    case LOOM_PREDICATE_POW2:
      return loom_vm_contract_materialize_power_of_two(materializer, predicate);
    case LOOM_PREDICATE_RANGE:
      return loom_vm_contract_materialize_range(materializer, predicate);
    case LOOM_PREDICATE_NOT_NAN:
    case LOOM_PREDICATE_NOT_INF:
    case LOOM_PREDICATE_FINITE:
      return loom_vm_contract_materialize_float(materializer, predicate);
    case LOOM_PREDICATE_COUNT_:
      break;
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "unsupported VM function predicate kind %u",
                          (unsigned)predicate->kind);
}

static iree_status_t loom_vm_contract_materializer_initialize(
    loom_rewriter_t* rewriter, loom_location_id_t location,
    loom_vm_contract_materializer_t* out_materializer) {
  *out_materializer = (loom_vm_contract_materializer_t){
      .rewriter = rewriter,
      .descriptor_set = loom_vm_core_descriptor_set(),
      .null_message = LOOM_VALUE_ID_INVALID,
      .location = location,
  };
  IREE_RETURN_IF_ERROR(loom_low_build_typed_register_type(
      rewriter->module, out_materializer->descriptor_set,
      VM_CORE_REG_CLASS_ID_VALUE, /*unit_count=*/1,
      loom_type_scalar(LOOM_SCALAR_TYPE_I1), &out_materializer->i1_type));
  IREE_RETURN_IF_ERROR(loom_low_build_typed_register_type(
      rewriter->module, out_materializer->descriptor_set,
      VM_CORE_REG_CLASS_ID_VALUE, /*unit_count=*/1,
      loom_type_scalar(LOOM_SCALAR_TYPE_I32), &out_materializer->i32_type));
  IREE_RETURN_IF_ERROR(loom_low_build_typed_register_type(
      rewriter->module, out_materializer->descriptor_set,
      VM_CORE_REG_CLASS_ID_VALUE, /*unit_count=*/1,
      loom_type_scalar(LOOM_SCALAR_TYPE_I64), &out_materializer->i64_type));
  IREE_RETURN_IF_ERROR(loom_low_build_typed_register_type(
      rewriter->module, out_materializer->descriptor_set,
      VM_CORE_REG_CLASS_ID_VALUE, /*unit_count=*/1,
      loom_type_scalar(LOOM_SCALAR_TYPE_F32), &out_materializer->f32_type));
  return loom_low_build_typed_register_type(
      rewriter->module, out_materializer->descriptor_set,
      VM_CORE_REG_CLASS_ID_REF, /*unit_count=*/1, loom_type_buffer(),
      &out_materializer->buffer_ref_type);
}

iree_status_t loom_vm_materialize_function_contracts_run(
    loom_pass_t* pass, loom_module_t* module, loom_func_like_t function) {
  if (!loom_low_func_def_isa(function.op) ||
      loom_func_like_abi(function) != LOOM_TARGET_ABI_VM_FUNCTION) {
    return iree_ok_status();
  }
  uint16_t predicate_count = 0;
  const loom_predicate_t* predicates =
      loom_func_like_predicates(function, &predicate_count);
  if (predicate_count == 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_vm_contract_preflight(module, function, predicates,
                                                  predicate_count));

  loom_rewriter_t rewriter = {0};
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, module, pass->arena));
  loom_vm_contract_materializer_t materializer;
  iree_status_t status = loom_vm_contract_materializer_initialize(
      &rewriter, function.op->location, &materializer);
  if (iree_status_is_ok(status)) {
    loom_region_t* body = loom_low_func_def_body(function.op);
    loom_block_t* entry_block = loom_region_entry_block(body);
    IREE_ASSERT(entry_block->first_op != NULL);
    loom_builder_set_before(&rewriter.builder, entry_block->first_op);
    for (uint16_t i = 0; i < predicate_count && iree_status_is_ok(status);
         ++i) {
      status =
          loom_vm_contract_materialize_predicate(&materializer, &predicates[i]);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_rewriter_set_attr(&rewriter, function.op,
                                    loom_low_func_def_predicates_ATTR_INDEX,
                                    loom_attr_absent());
  }
  // Entry checks make the lowered function observably effectful even when the
  // source function was pure over inputs satisfying its contract.
  if (iree_status_is_ok(status) && loom_func_like_purity(function) != 0) {
    status = loom_rewriter_set_attr(&rewriter, function.op,
                                    loom_low_func_def_purity_ATTR_INDEX,
                                    loom_attr_absent());
  }
  if (iree_status_is_ok(status)) loom_pass_mark_changed(pass);
  loom_rewriter_deinitialize(&rewriter);
  return status;
}
