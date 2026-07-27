// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/llvmir/verify.h"

#include "loom/ops/atomic.h"
#include "loom/target/emit/llvmir/types.h"

static bool loom_llvmir_is_terminator(const loom_llvmir_instruction_t* inst) {
  return inst->kind == LOOM_LLVMIR_INST_RET ||
         inst->kind == LOOM_LLVMIR_INST_BR ||
         inst->kind == LOOM_LLVMIR_INST_COND_BR ||
         inst->kind == LOOM_LLVMIR_INST_UNREACHABLE;
}

static const loom_llvmir_type_t* loom_llvmir_verify_type(
    const loom_llvmir_module_t* module, loom_llvmir_type_id_t type_id) {
  return type_id < module->type_count ? &module->types[type_id] : NULL;
}

static const loom_llvmir_value_t* loom_llvmir_verify_value(
    const loom_llvmir_module_t* module, loom_llvmir_value_id_t value_id) {
  return value_id < module->value_count ? &module->values[value_id] : NULL;
}

static bool loom_llvmir_verify_value_is_constant(
    const loom_llvmir_value_t* value) {
  return value && (value->kind == LOOM_LLVMIR_VALUE_CONSTANT_INTEGER ||
                   value->kind == LOOM_LLVMIR_VALUE_CONSTANT_FLOAT_BITS ||
                   value->kind == LOOM_LLVMIR_VALUE_CONSTANT_NULL ||
                   value->kind == LOOM_LLVMIR_VALUE_CONSTANT_INTEGER_VECTOR ||
                   value->kind == LOOM_LLVMIR_VALUE_CONSTANT_POISON);
}

static loom_llvmir_type_id_t loom_llvmir_verify_value_type(
    const loom_llvmir_module_t* module, loom_llvmir_value_id_t value_id) {
  const loom_llvmir_value_t* value = loom_llvmir_verify_value(module, value_id);
  return value ? value->type_id : LOOM_LLVMIR_TYPE_ID_INVALID;
}

static loom_llvmir_function_t* loom_llvmir_verify_function_ref(
    const loom_llvmir_module_t* module, loom_llvmir_function_id_t function_id) {
  return function_id < module->function_count ? module->functions[function_id]
                                              : NULL;
}

static iree_status_t loom_llvmir_verify_expected_value_type(
    const loom_llvmir_module_t* module, loom_llvmir_value_id_t value_id,
    loom_llvmir_type_id_t expected_type) {
  loom_llvmir_type_id_t actual_type =
      loom_llvmir_verify_value_type(module, value_id);
  if (actual_type == LOOM_LLVMIR_TYPE_ID_INVALID) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM instruction references unknown value");
  }
  if (actual_type != expected_type) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM instruction operand type mismatch");
  }
  return iree_ok_status();
}

static bool loom_llvmir_verify_type_is_i1(const loom_llvmir_module_t* module,
                                          loom_llvmir_type_id_t type_id) {
  const loom_llvmir_type_t* type = loom_llvmir_verify_type(module, type_id);
  return type && type->kind == LOOM_LLVMIR_TYPE_INTEGER && type->bit_width == 1;
}

static bool loom_llvmir_verify_type_is_scalar_integer(
    const loom_llvmir_module_t* module, loom_llvmir_type_id_t type_id) {
  const loom_llvmir_type_t* type = loom_llvmir_verify_type(module, type_id);
  return type && type->kind == LOOM_LLVMIR_TYPE_INTEGER;
}

static const loom_llvmir_type_t* loom_llvmir_verify_scalar_type(
    const loom_llvmir_module_t* module, const loom_llvmir_type_t* type) {
  if (!type) return NULL;
  if (type->kind != LOOM_LLVMIR_TYPE_VECTOR) return type;
  return loom_llvmir_verify_type(module, type->element_type);
}

static uint32_t loom_llvmir_verify_vector_lanes(
    const loom_llvmir_type_t* type) {
  return type && type->kind == LOOM_LLVMIR_TYPE_VECTOR ? type->element_count
                                                       : 0;
}

static bool loom_llvmir_verify_same_vector_shape(
    const loom_llvmir_type_t* lhs, const loom_llvmir_type_t* rhs) {
  return loom_llvmir_verify_vector_lanes(lhs) ==
         loom_llvmir_verify_vector_lanes(rhs);
}

static bool loom_llvmir_verify_type_is_first_class(
    const loom_llvmir_module_t* module, const loom_llvmir_type_t* type) {
  const loom_llvmir_type_t* scalar_type =
      loom_llvmir_verify_scalar_type(module, type);
  return scalar_type && scalar_type->kind != LOOM_LLVMIR_TYPE_VOID &&
         scalar_type->kind != LOOM_LLVMIR_TYPE_VECTOR;
}

static bool loom_llvmir_verify_type_is_int_like(
    const loom_llvmir_module_t* module, const loom_llvmir_type_t* type) {
  const loom_llvmir_type_t* scalar_type =
      loom_llvmir_verify_scalar_type(module, type);
  return scalar_type && scalar_type->kind == LOOM_LLVMIR_TYPE_INTEGER;
}

static bool loom_llvmir_verify_type_is_pointer_like(
    const loom_llvmir_module_t* module, const loom_llvmir_type_t* type) {
  const loom_llvmir_type_t* scalar_type =
      loom_llvmir_verify_scalar_type(module, type);
  return scalar_type && scalar_type->kind == LOOM_LLVMIR_TYPE_POINTER;
}

static bool loom_llvmir_verify_type_pointer_address_space(
    const loom_llvmir_module_t* module, const loom_llvmir_type_t* type,
    uint32_t* out_address_space) {
  const loom_llvmir_type_t* scalar_type =
      loom_llvmir_verify_scalar_type(module, type);
  if (!scalar_type || scalar_type->kind != LOOM_LLVMIR_TYPE_POINTER) {
    return false;
  }
  *out_address_space = scalar_type->address_space;
  return true;
}

static bool loom_llvmir_verify_type_is_float_like(
    const loom_llvmir_module_t* module, const loom_llvmir_type_t* type) {
  const loom_llvmir_type_t* scalar_type =
      loom_llvmir_verify_scalar_type(module, type);
  return scalar_type && scalar_type->kind == LOOM_LLVMIR_TYPE_FLOAT;
}

static uint32_t loom_llvmir_verify_type_scalar_bit_width(
    const loom_llvmir_module_t* module, const loom_llvmir_type_t* type) {
  const loom_llvmir_type_t* scalar_type =
      loom_llvmir_verify_scalar_type(module, type);
  if (!scalar_type) return 0;
  switch (scalar_type->kind) {
    case LOOM_LLVMIR_TYPE_INTEGER:
      return scalar_type->bit_width;
    case LOOM_LLVMIR_TYPE_FLOAT:
      switch (scalar_type->float_kind) {
        case LOOM_LLVMIR_FLOAT_F16:
        case LOOM_LLVMIR_FLOAT_BF16:
          return 16;
        case LOOM_LLVMIR_FLOAT_F32:
          return 32;
        case LOOM_LLVMIR_FLOAT_F64:
          return 64;
        default:
          return 0;
      }
    default:
      return 0;
  }
}

static uint64_t loom_llvmir_verify_type_primitive_bit_width(
    const loom_llvmir_module_t* module, const loom_llvmir_type_t* type) {
  uint32_t scalar_bit_width =
      loom_llvmir_verify_type_scalar_bit_width(module, type);
  if (scalar_bit_width == 0) return 0;
  uint32_t lanes = loom_llvmir_verify_vector_lanes(type);
  return lanes == 0 ? scalar_bit_width : (uint64_t)scalar_bit_width * lanes;
}

static bool loom_llvmir_verify_is_power_of_two_u32(uint32_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

static iree_status_t loom_llvmir_verify_memory_alignment(uint32_t alignment) {
  if (alignment == 0) return iree_ok_status();
  if (!loom_llvmir_verify_is_power_of_two_u32(alignment)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM memory alignment is not a power of two");
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_verify_memory_flags(uint32_t flags) {
  if ((flags & ~LOOM_LLVMIR_MEMORY_VOLATILE) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown LLVM memory flags");
  }
  return iree_ok_status();
}

static bool loom_llvmir_verify_atomic_rmw_op_accepts_type(
    const loom_llvmir_module_t* module, loom_llvmir_atomic_rmw_op_t op,
    loom_llvmir_type_id_t type_id) {
  const loom_llvmir_type_t* type = loom_llvmir_verify_type(module, type_id);
  if (!type || type->kind == LOOM_LLVMIR_TYPE_VECTOR) return false;
  switch (op) {
    case LOOM_LLVMIR_ATOMIC_RMW_XCHG:
      return type->kind == LOOM_LLVMIR_TYPE_INTEGER ||
             type->kind == LOOM_LLVMIR_TYPE_FLOAT;
    case LOOM_LLVMIR_ATOMIC_RMW_ADD:
    case LOOM_LLVMIR_ATOMIC_RMW_SUB:
    case LOOM_LLVMIR_ATOMIC_RMW_AND:
    case LOOM_LLVMIR_ATOMIC_RMW_OR:
    case LOOM_LLVMIR_ATOMIC_RMW_XOR:
    case LOOM_LLVMIR_ATOMIC_RMW_MAX:
    case LOOM_LLVMIR_ATOMIC_RMW_MIN:
    case LOOM_LLVMIR_ATOMIC_RMW_UMAX:
    case LOOM_LLVMIR_ATOMIC_RMW_UMIN:
      return type->kind == LOOM_LLVMIR_TYPE_INTEGER;
    case LOOM_LLVMIR_ATOMIC_RMW_FADD:
    case LOOM_LLVMIR_ATOMIC_RMW_FMAX:
    case LOOM_LLVMIR_ATOMIC_RMW_FMIN:
    case LOOM_LLVMIR_ATOMIC_RMW_FMAXIMUM:
    case LOOM_LLVMIR_ATOMIC_RMW_FMINIMUM:
      return type->kind == LOOM_LLVMIR_TYPE_FLOAT;
    default:
      return false;
  }
}

static bool loom_llvmir_verify_atomic_ordering(
    loom_llvmir_atomic_ordering_t ordering) {
  switch (ordering) {
    case LOOM_LLVMIR_ATOMIC_ORDERING_MONOTONIC:
    case LOOM_LLVMIR_ATOMIC_ORDERING_ACQUIRE:
    case LOOM_LLVMIR_ATOMIC_ORDERING_RELEASE:
    case LOOM_LLVMIR_ATOMIC_ORDERING_ACQ_REL:
    case LOOM_LLVMIR_ATOMIC_ORDERING_SEQ_CST:
      return true;
    default:
      return false;
  }
}

static iree_status_t loom_llvmir_verify_atomic_rmw(
    const loom_llvmir_module_t* module,
    const loom_llvmir_instruction_t* instruction) {
  if (instruction->result_value_id == LOOM_LLVMIR_VALUE_ID_INVALID) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM atomicrmw has no result value");
  }
  IREE_RETURN_IF_ERROR(loom_llvmir_verify_expected_value_type(
      module, instruction->result_value_id,
      instruction->atomic_rmw.result_type));
  IREE_RETURN_IF_ERROR(loom_llvmir_verify_expected_value_type(
      module, instruction->atomic_rmw.value,
      instruction->atomic_rmw.result_type));
  if (!loom_llvmir_verify_value(module, instruction->atomic_rmw.pointer)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM atomicrmw references unknown pointer");
  }
  const loom_llvmir_type_t* pointer_type = loom_llvmir_verify_type(
      module,
      loom_llvmir_verify_value_type(module, instruction->atomic_rmw.pointer));
  if (!loom_llvmir_verify_type_is_pointer_like(module, pointer_type)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM atomicrmw pointer must be a pointer");
  }
  if (!loom_llvmir_verify_atomic_rmw_op_accepts_type(
          module, instruction->atomic_rmw.op,
          instruction->atomic_rmw.result_type)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM atomicrmw opcode is invalid for result type");
  }
  if (!loom_llvmir_verify_atomic_ordering(instruction->atomic_rmw.ordering)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown LLVM atomic ordering");
  }
  IREE_RETURN_IF_ERROR(
      loom_llvmir_verify_memory_alignment(instruction->atomic_rmw.alignment));
  return loom_llvmir_verify_memory_flags(instruction->atomic_rmw.flags);
}

static iree_status_t loom_llvmir_verify_cmpxchg(
    const loom_llvmir_module_t* module,
    const loom_llvmir_instruction_t* instruction) {
  if (instruction->result_value_id == LOOM_LLVMIR_VALUE_ID_INVALID) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM cmpxchg has no result value");
  }
  IREE_RETURN_IF_ERROR(loom_llvmir_verify_expected_value_type(
      module, instruction->result_value_id, instruction->cmpxchg.result_type));
  IREE_RETURN_IF_ERROR(loom_llvmir_verify_expected_value_type(
      module, instruction->cmpxchg.expected, instruction->cmpxchg.value_type));
  IREE_RETURN_IF_ERROR(loom_llvmir_verify_expected_value_type(
      module, instruction->cmpxchg.replacement,
      instruction->cmpxchg.value_type));
  if (!loom_llvmir_verify_value(module, instruction->cmpxchg.pointer)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM cmpxchg references unknown pointer");
  }
  const loom_llvmir_type_t* pointer_type = loom_llvmir_verify_type(
      module,
      loom_llvmir_verify_value_type(module, instruction->cmpxchg.pointer));
  if (!loom_llvmir_verify_type_is_pointer_like(module, pointer_type)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM cmpxchg pointer must be a pointer");
  }

  const loom_llvmir_type_t* value_type =
      loom_llvmir_verify_type(module, instruction->cmpxchg.value_type);
  if (!value_type || (value_type->kind != LOOM_LLVMIR_TYPE_INTEGER &&
                      value_type->kind != LOOM_LLVMIR_TYPE_POINTER)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "LLVM cmpxchg value type must be an integer or pointer");
  }

  const loom_llvmir_type_t* result_type =
      loom_llvmir_verify_type(module, instruction->cmpxchg.result_type);
  if (!result_type || result_type->kind != LOOM_LLVMIR_TYPE_STRUCT ||
      result_type->element_count != 2 ||
      result_type->element_types[0] != instruction->cmpxchg.value_type ||
      !loom_llvmir_verify_type_is_i1(module, result_type->element_types[1])) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "LLVM cmpxchg result type must be { value_type, i1 }");
  }
  if (!loom_llvmir_verify_atomic_ordering(
          instruction->cmpxchg.success_ordering) ||
      !loom_llvmir_verify_atomic_ordering(
          instruction->cmpxchg.failure_ordering)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown LLVM atomic ordering");
  }
  loom_atomic_cmpxchg_ordering_error_t ordering_error =
      loom_atomic_cmpxchg_ordering_validate(
          (uint8_t)instruction->cmpxchg.success_ordering,
          (uint8_t)instruction->cmpxchg.failure_ordering);
  if (ordering_error != LOOM_ATOMIC_CMPXCHG_ORDERING_ERROR_NONE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid LLVM cmpxchg ordering pair");
  }
  IREE_RETURN_IF_ERROR(
      loom_llvmir_verify_memory_alignment(instruction->cmpxchg.alignment));
  return loom_llvmir_verify_memory_flags(instruction->cmpxchg.flags);
}

static iree_status_t loom_llvmir_verify_metadata_attachments(
    const loom_llvmir_module_t* module,
    const loom_llvmir_metadata_attachment_storage_t* attachments,
    iree_host_size_t attachment_count) {
  if (attachment_count == 0) return iree_ok_status();
  if (attachments == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "non-empty LLVM metadata attachment list has null storage");
  }
  for (iree_host_size_t i = 0; i < attachment_count; ++i) {
    if (iree_string_view_is_empty(attachments[i].name)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "LLVM metadata attachment name must not be empty");
    }
    if (attachments[i].metadata_id >= module->metadata_node_count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "LLVM metadata attachment references unknown node");
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_verify_mask_result_type(
    const loom_llvmir_module_t* module, loom_llvmir_type_id_t operand_type_id,
    loom_llvmir_type_id_t result_type_id) {
  const loom_llvmir_type_t* operand_type =
      loom_llvmir_verify_type(module, operand_type_id);
  const loom_llvmir_type_t* result_type =
      loom_llvmir_verify_type(module, result_type_id);
  if (!operand_type || !result_type) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM compare references unknown type");
  }
  if (operand_type->kind != LOOM_LLVMIR_TYPE_VECTOR) {
    if (!loom_llvmir_verify_type_is_i1(module, result_type_id)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "LLVM scalar compare result must be i1");
    }
    return iree_ok_status();
  }
  if (result_type->kind != LOOM_LLVMIR_TYPE_VECTOR ||
      result_type->element_count != operand_type->element_count ||
      !loom_llvmir_verify_type_is_i1(module, result_type->element_type)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "LLVM vector compare result must be a same-lane i1 vector");
  }
  return iree_ok_status();
}

static bool loom_llvmir_verify_type_is_integer_or_pointer_like(
    const loom_llvmir_module_t* module, const loom_llvmir_type_t* type) {
  const loom_llvmir_type_t* element_type =
      loom_llvmir_verify_scalar_type(module, type);
  return element_type && (element_type->kind == LOOM_LLVMIR_TYPE_INTEGER ||
                          element_type->kind == LOOM_LLVMIR_TYPE_POINTER);
}

static bool loom_llvmir_verify_parse_data_layout_uint32_field(
    iree_string_view_t* inout_text, bool required, uint32_t* out_value,
    bool* out_present) {
  *out_present = false;
  *out_value = 0;
  if (iree_string_view_is_empty(*inout_text)) {
    return !required;
  }

  iree_string_view_t field = iree_string_view_empty();
  iree_string_view_t remaining = iree_string_view_empty();
  if (iree_string_view_split(*inout_text, ':', &field, &remaining) < 0) {
    field = *inout_text;
    remaining = iree_string_view_empty();
  }
  if (iree_string_view_is_empty(field) ||
      !iree_string_view_atoi_uint32(field, out_value)) {
    return false;
  }
  *inout_text = remaining;
  *out_present = true;
  return true;
}

static bool loom_llvmir_verify_parse_data_layout_pointer_entry(
    iree_string_view_t token, uint32_t* out_address_space,
    uint32_t* out_address_bit_width) {
  if (!iree_string_view_consume_prefix_char(&token, 'p')) {
    return false;
  }

  uint32_t address_space = 0;
  if (!iree_string_view_consume_prefix_char(&token, ':')) {
    iree_string_view_t address_space_text = iree_string_view_empty();
    iree_string_view_t fields = iree_string_view_empty();
    if (iree_string_view_split(token, ':', &address_space_text, &fields) < 0 ||
        iree_string_view_is_empty(address_space_text) ||
        !iree_string_view_atoi_uint32(address_space_text, &address_space)) {
      return false;
    }
    token = fields;
  }

  uint32_t pointer_bit_width = 0;
  bool present = false;
  if (!loom_llvmir_verify_parse_data_layout_uint32_field(
          &token, /*required=*/true, &pointer_bit_width, &present)) {
    return false;
  }

  uint32_t abi_alignment = 0;
  if (!loom_llvmir_verify_parse_data_layout_uint32_field(
          &token, /*required=*/true, &abi_alignment, &present)) {
    return false;
  }

  uint32_t preferred_alignment = 0;
  if (!loom_llvmir_verify_parse_data_layout_uint32_field(
          &token, /*required=*/false, &preferred_alignment, &present)) {
    return false;
  }

  uint32_t address_bit_width = pointer_bit_width;
  uint32_t parsed_address_bit_width = 0;
  if (!loom_llvmir_verify_parse_data_layout_uint32_field(
          &token, /*required=*/false, &parsed_address_bit_width, &present)) {
    return false;
  }
  if (present) {
    address_bit_width = parsed_address_bit_width;
  }
  if (!iree_string_view_is_empty(token)) {
    return false;
  }

  *out_address_space = address_space;
  *out_address_bit_width = address_bit_width;
  return true;
}

static bool loom_llvmir_verify_pointer_address_bit_width(
    const loom_llvmir_module_t* module, uint32_t address_space,
    uint32_t* out_address_bit_width) {
  uint32_t default_address_bit_width =
      module->target_config.default_pointer_bitwidth;
  if (default_address_bit_width == 0) {
    default_address_bit_width = 64;
  }
  *out_address_bit_width = default_address_bit_width;

  iree_string_view_t remaining = module->target_config.data_layout;
  while (!iree_string_view_is_empty(remaining)) {
    iree_string_view_t token = iree_string_view_empty();
    iree_string_view_t suffix = iree_string_view_empty();
    if (iree_string_view_split(remaining, '-', &token, &suffix) < 0) {
      token = remaining;
      suffix = iree_string_view_empty();
    }
    remaining = suffix;

    if (!iree_string_view_starts_with_char(token, 'p')) {
      continue;
    }

    uint32_t entry_address_space = 0;
    uint32_t entry_address_bit_width = 0;
    if (!loom_llvmir_verify_parse_data_layout_pointer_entry(
            token, &entry_address_space, &entry_address_bit_width)) {
      return false;
    }
    if (entry_address_space == 0) {
      default_address_bit_width = entry_address_bit_width;
      *out_address_bit_width = default_address_bit_width;
    }
    if (entry_address_space == address_space) {
      *out_address_bit_width = entry_address_bit_width;
      return true;
    }
  }

  *out_address_bit_width = default_address_bit_width;
  return true;
}

static bool loom_llvmir_verify_ptrtoaddr(
    const loom_llvmir_module_t* module, const loom_llvmir_type_t* source_type,
    const loom_llvmir_type_t* result_type) {
  if (!loom_llvmir_verify_type_is_pointer_like(module, source_type) ||
      !loom_llvmir_verify_type_is_int_like(module, result_type) ||
      !loom_llvmir_verify_same_vector_shape(source_type, result_type)) {
    return false;
  }

  uint32_t address_space = 0;
  if (!loom_llvmir_verify_type_pointer_address_space(module, source_type,
                                                     &address_space)) {
    return false;
  }
  uint32_t expected_bit_width = 0;
  if (!loom_llvmir_verify_pointer_address_bit_width(module, address_space,
                                                    &expected_bit_width)) {
    return false;
  }
  return expected_bit_width != 0 &&
         loom_llvmir_verify_type_scalar_bit_width(module, result_type) ==
             expected_bit_width;
}

static bool loom_llvmir_verify_icmp_predicate(
    loom_llvmir_icmp_predicate_t predicate) {
  switch (predicate) {
    case LOOM_LLVMIR_ICMP_EQ:
    case LOOM_LLVMIR_ICMP_NE:
    case LOOM_LLVMIR_ICMP_UGT:
    case LOOM_LLVMIR_ICMP_UGE:
    case LOOM_LLVMIR_ICMP_ULT:
    case LOOM_LLVMIR_ICMP_ULE:
    case LOOM_LLVMIR_ICMP_SGT:
    case LOOM_LLVMIR_ICMP_SGE:
    case LOOM_LLVMIR_ICMP_SLT:
    case LOOM_LLVMIR_ICMP_SLE:
      return true;
    default:
      return false;
  }
}

static bool loom_llvmir_verify_fcmp_predicate(
    loom_llvmir_fcmp_predicate_t predicate) {
  switch (predicate) {
    case LOOM_LLVMIR_FCMP_FALSE:
    case LOOM_LLVMIR_FCMP_OEQ:
    case LOOM_LLVMIR_FCMP_OGT:
    case LOOM_LLVMIR_FCMP_OGE:
    case LOOM_LLVMIR_FCMP_OLT:
    case LOOM_LLVMIR_FCMP_OLE:
    case LOOM_LLVMIR_FCMP_ONE:
    case LOOM_LLVMIR_FCMP_ORD:
    case LOOM_LLVMIR_FCMP_UNO:
    case LOOM_LLVMIR_FCMP_UEQ:
    case LOOM_LLVMIR_FCMP_UGT:
    case LOOM_LLVMIR_FCMP_UGE:
    case LOOM_LLVMIR_FCMP_ULT:
    case LOOM_LLVMIR_FCMP_ULE:
    case LOOM_LLVMIR_FCMP_UNE:
    case LOOM_LLVMIR_FCMP_TRUE:
      return true;
    default:
      return false;
  }
}

static bool loom_llvmir_verify_binop_accepts_type(
    const loom_llvmir_module_t* module, loom_llvmir_binop_t op,
    const loom_llvmir_type_t* type) {
  switch (op) {
    case LOOM_LLVMIR_BINOP_ADD:
    case LOOM_LLVMIR_BINOP_SUB:
    case LOOM_LLVMIR_BINOP_MUL:
    case LOOM_LLVMIR_BINOP_UDIV:
    case LOOM_LLVMIR_BINOP_SDIV:
    case LOOM_LLVMIR_BINOP_UREM:
    case LOOM_LLVMIR_BINOP_SREM:
    case LOOM_LLVMIR_BINOP_AND:
    case LOOM_LLVMIR_BINOP_OR:
    case LOOM_LLVMIR_BINOP_XOR:
    case LOOM_LLVMIR_BINOP_SHL:
    case LOOM_LLVMIR_BINOP_LSHR:
    case LOOM_LLVMIR_BINOP_ASHR:
      return loom_llvmir_verify_type_is_int_like(module, type);
    case LOOM_LLVMIR_BINOP_FADD:
    case LOOM_LLVMIR_BINOP_FSUB:
    case LOOM_LLVMIR_BINOP_FMUL:
    case LOOM_LLVMIR_BINOP_FDIV:
    case LOOM_LLVMIR_BINOP_FREM:
      return loom_llvmir_verify_type_is_float_like(module, type);
    default:
      return false;
  }
}

static bool loom_llvmir_verify_binop_is_float(loom_llvmir_binop_t op) {
  switch (op) {
    case LOOM_LLVMIR_BINOP_FADD:
    case LOOM_LLVMIR_BINOP_FSUB:
    case LOOM_LLVMIR_BINOP_FMUL:
    case LOOM_LLVMIR_BINOP_FDIV:
    case LOOM_LLVMIR_BINOP_FREM:
      return true;
    default:
      return false;
  }
}

static bool loom_llvmir_verify_binop_allows_no_wrap_flags(
    loom_llvmir_binop_t op) {
  switch (op) {
    case LOOM_LLVMIR_BINOP_ADD:
    case LOOM_LLVMIR_BINOP_SUB:
    case LOOM_LLVMIR_BINOP_MUL:
    case LOOM_LLVMIR_BINOP_SHL:
      return true;
    default:
      return false;
  }
}

static bool loom_llvmir_verify_binop_allows_exact_flag(loom_llvmir_binop_t op) {
  switch (op) {
    case LOOM_LLVMIR_BINOP_UDIV:
    case LOOM_LLVMIR_BINOP_SDIV:
    case LOOM_LLVMIR_BINOP_LSHR:
    case LOOM_LLVMIR_BINOP_ASHR:
      return true;
    default:
      return false;
  }
}

static iree_status_t loom_llvmir_verify_binop_flags(
    const loom_llvmir_instruction_t* instruction) {
  const uint32_t known_integer_flags =
      LOOM_LLVMIR_INTEGER_ARITHMETIC_NO_UNSIGNED_WRAP |
      LOOM_LLVMIR_INTEGER_ARITHMETIC_NO_SIGNED_WRAP |
      LOOM_LLVMIR_INTEGER_ARITHMETIC_EXACT;
  uint32_t integer_flags = (uint32_t)instruction->binop.integer_flags;
  if (integer_flags & ~known_integer_flags) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown LLVM integer arithmetic flags");
  }
  const uint32_t known_fast_math_flags = LOOM_LLVMIR_FAST_MATH_FAST;
  uint32_t fast_math_flags = (uint32_t)instruction->binop.fast_math_flags;
  if (fast_math_flags & ~known_fast_math_flags) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown LLVM fast-math flags");
  }
  if (integer_flags != 0 && fast_math_flags != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM binop mixes integer and fast-math flags");
  }
  if (fast_math_flags != 0 &&
      !loom_llvmir_verify_binop_is_float(instruction->binop.op)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM integer binop cannot carry fast-math flags");
  }
  if (integer_flags != 0 &&
      loom_llvmir_verify_binop_is_float(instruction->binop.op)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM floating-point binop cannot carry integer "
                            "arithmetic flags");
  }
  const uint32_t no_wrap_flags =
      LOOM_LLVMIR_INTEGER_ARITHMETIC_NO_UNSIGNED_WRAP |
      LOOM_LLVMIR_INTEGER_ARITHMETIC_NO_SIGNED_WRAP;
  if ((integer_flags & no_wrap_flags) &&
      !loom_llvmir_verify_binop_allows_no_wrap_flags(instruction->binop.op)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM binop cannot carry no-wrap flags");
  }
  if ((integer_flags & LOOM_LLVMIR_INTEGER_ARITHMETIC_EXACT) &&
      !loom_llvmir_verify_binop_allows_exact_flag(instruction->binop.op)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM binop cannot carry exact flag");
  }
  return iree_ok_status();
}

static bool loom_llvmir_verify_unop_accepts_type(
    const loom_llvmir_module_t* module, loom_llvmir_unop_t op,
    const loom_llvmir_type_t* type) {
  switch (op) {
    case LOOM_LLVMIR_UNOP_FNEG:
      return loom_llvmir_verify_type_is_float_like(module, type);
    default:
      return false;
  }
}

static iree_status_t loom_llvmir_verify_unop_flags(
    const loom_llvmir_instruction_t* instruction) {
  const uint32_t known_fast_math_flags = LOOM_LLVMIR_FAST_MATH_FAST;
  uint32_t fast_math_flags = (uint32_t)instruction->unop.fast_math_flags;
  if (fast_math_flags & ~known_fast_math_flags) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown LLVM fast-math flags");
  }
  if (fast_math_flags != 0 && instruction->unop.op != LOOM_LLVMIR_UNOP_FNEG) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM unary op cannot carry fast-math flags");
  }
  return iree_ok_status();
}

static bool loom_llvmir_verify_type_same_pointer_address_space(
    const loom_llvmir_module_t* module, const loom_llvmir_type_t* lhs,
    const loom_llvmir_type_t* rhs) {
  const loom_llvmir_type_t* lhs_scalar =
      loom_llvmir_verify_scalar_type(module, lhs);
  const loom_llvmir_type_t* rhs_scalar =
      loom_llvmir_verify_scalar_type(module, rhs);
  return lhs_scalar && rhs_scalar &&
         lhs_scalar->kind == LOOM_LLVMIR_TYPE_POINTER &&
         rhs_scalar->kind == LOOM_LLVMIR_TYPE_POINTER &&
         lhs_scalar->address_space == rhs_scalar->address_space;
}

static bool loom_llvmir_verify_bitcast(const loom_llvmir_module_t* module,
                                       const loom_llvmir_type_t* source_type,
                                       const loom_llvmir_type_t* result_type) {
  bool source_is_pointer =
      loom_llvmir_verify_type_is_pointer_like(module, source_type);
  bool result_is_pointer =
      loom_llvmir_verify_type_is_pointer_like(module, result_type);
  if (source_is_pointer != result_is_pointer) return false;
  if (!source_is_pointer) {
    uint64_t source_bit_width =
        loom_llvmir_verify_type_primitive_bit_width(module, source_type);
    uint64_t result_bit_width =
        loom_llvmir_verify_type_primitive_bit_width(module, result_type);
    return source_bit_width != 0 && source_bit_width == result_bit_width;
  }
  if (!loom_llvmir_verify_type_same_pointer_address_space(module, source_type,
                                                          result_type)) {
    return false;
  }
  bool source_is_vector = source_type->kind == LOOM_LLVMIR_TYPE_VECTOR;
  bool result_is_vector = result_type->kind == LOOM_LLVMIR_TYPE_VECTOR;
  if (source_is_vector && result_is_vector) {
    return loom_llvmir_verify_same_vector_shape(source_type, result_type);
  }
  if (source_is_vector) return source_type->element_count == 1;
  if (result_is_vector) return result_type->element_count == 1;
  return true;
}

static bool loom_llvmir_verify_cast_is_valid(
    const loom_llvmir_module_t* module, loom_llvmir_cast_op_t op,
    const loom_llvmir_type_t* source_type,
    const loom_llvmir_type_t* result_type) {
  if (!loom_llvmir_verify_type_is_first_class(module, source_type) ||
      !loom_llvmir_verify_type_is_first_class(module, result_type)) {
    return false;
  }
  uint32_t source_bits =
      loom_llvmir_verify_type_scalar_bit_width(module, source_type);
  uint32_t result_bits =
      loom_llvmir_verify_type_scalar_bit_width(module, result_type);
  switch (op) {
    case LOOM_LLVMIR_CAST_TRUNCATE:
      return loom_llvmir_verify_type_is_int_like(module, source_type) &&
             loom_llvmir_verify_type_is_int_like(module, result_type) &&
             loom_llvmir_verify_same_vector_shape(source_type, result_type) &&
             source_bits > result_bits;
    case LOOM_LLVMIR_CAST_ZERO_EXTEND:
    case LOOM_LLVMIR_CAST_SIGN_EXTEND:
      return loom_llvmir_verify_type_is_int_like(module, source_type) &&
             loom_llvmir_verify_type_is_int_like(module, result_type) &&
             loom_llvmir_verify_same_vector_shape(source_type, result_type) &&
             source_bits < result_bits;
    case LOOM_LLVMIR_CAST_FP_TRUNCATE:
      return loom_llvmir_verify_type_is_float_like(module, source_type) &&
             loom_llvmir_verify_type_is_float_like(module, result_type) &&
             loom_llvmir_verify_same_vector_shape(source_type, result_type) &&
             source_bits > result_bits;
    case LOOM_LLVMIR_CAST_FP_EXTEND:
      return loom_llvmir_verify_type_is_float_like(module, source_type) &&
             loom_llvmir_verify_type_is_float_like(module, result_type) &&
             loom_llvmir_verify_same_vector_shape(source_type, result_type) &&
             source_bits < result_bits;
    case LOOM_LLVMIR_CAST_UNSIGNED_INT_TO_FP:
    case LOOM_LLVMIR_CAST_SIGNED_INT_TO_FP:
      return loom_llvmir_verify_type_is_int_like(module, source_type) &&
             loom_llvmir_verify_type_is_float_like(module, result_type) &&
             loom_llvmir_verify_same_vector_shape(source_type, result_type);
    case LOOM_LLVMIR_CAST_FP_TO_UNSIGNED_INT:
    case LOOM_LLVMIR_CAST_FP_TO_SIGNED_INT:
      return loom_llvmir_verify_type_is_float_like(module, source_type) &&
             loom_llvmir_verify_type_is_int_like(module, result_type) &&
             loom_llvmir_verify_same_vector_shape(source_type, result_type);
    case LOOM_LLVMIR_CAST_PTR_TO_INT:
      return loom_llvmir_verify_type_is_pointer_like(module, source_type) &&
             loom_llvmir_verify_type_is_int_like(module, result_type) &&
             loom_llvmir_verify_same_vector_shape(source_type, result_type);
    case LOOM_LLVMIR_CAST_PTR_TO_ADDR:
      return loom_llvmir_verify_ptrtoaddr(module, source_type, result_type);
    case LOOM_LLVMIR_CAST_INT_TO_PTR:
      return loom_llvmir_verify_type_is_int_like(module, source_type) &&
             loom_llvmir_verify_type_is_pointer_like(module, result_type) &&
             loom_llvmir_verify_same_vector_shape(source_type, result_type);
    case LOOM_LLVMIR_CAST_BITCAST:
      return loom_llvmir_verify_bitcast(module, source_type, result_type);
    case LOOM_LLVMIR_CAST_ADDRESS_SPACE_CAST:
      return loom_llvmir_verify_type_is_pointer_like(module, source_type) &&
             loom_llvmir_verify_type_is_pointer_like(module, result_type) &&
             !loom_llvmir_verify_type_same_pointer_address_space(
                 module, source_type, result_type) &&
             loom_llvmir_verify_same_vector_shape(source_type, result_type);
    default:
      return false;
  }
}

static iree_status_t loom_llvmir_verify_cast(
    const loom_llvmir_module_t* module,
    const loom_llvmir_instruction_t* instruction) {
  if (instruction->result_value_id == LOOM_LLVMIR_VALUE_ID_INVALID) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM cast has no result value");
  }
  loom_llvmir_type_id_t source_type_id =
      loom_llvmir_verify_value_type(module, instruction->cast.value);
  loom_llvmir_type_id_t result_type_id =
      loom_llvmir_verify_value_type(module, instruction->result_value_id);
  const loom_llvmir_type_t* source_type =
      loom_llvmir_verify_type(module, source_type_id);
  const loom_llvmir_type_t* result_type =
      loom_llvmir_verify_type(module, result_type_id);
  if (!source_type || !result_type) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM cast references unknown type");
  }
  if (!loom_llvmir_verify_cast_is_valid(module, instruction->cast.op,
                                        source_type, result_type)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM cast has invalid source/result types");
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_verify_compare(
    const loom_llvmir_module_t* module,
    const loom_llvmir_instruction_t* instruction, loom_llvmir_value_id_t lhs,
    loom_llvmir_value_id_t rhs,
    bool (*is_valid_operand_type)(const loom_llvmir_module_t*,
                                  const loom_llvmir_type_t*)) {
  if (instruction->result_value_id == LOOM_LLVMIR_VALUE_ID_INVALID) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM compare has no result value");
  }
  loom_llvmir_type_id_t lhs_type_id =
      loom_llvmir_verify_value_type(module, lhs);
  loom_llvmir_type_id_t rhs_type_id =
      loom_llvmir_verify_value_type(module, rhs);
  if (lhs_type_id == LOOM_LLVMIR_TYPE_ID_INVALID ||
      rhs_type_id == LOOM_LLVMIR_TYPE_ID_INVALID) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM compare references unknown value");
  }
  if (lhs_type_id != rhs_type_id) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM compare operand type mismatch");
  }
  if (!is_valid_operand_type(module,
                             loom_llvmir_verify_type(module, lhs_type_id))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM compare has invalid operand type");
  }
  return loom_llvmir_verify_mask_result_type(
      module, lhs_type_id,
      loom_llvmir_verify_value_type(module, instruction->result_value_id));
}

static iree_status_t loom_llvmir_verify_select(
    const loom_llvmir_module_t* module,
    const loom_llvmir_instruction_t* instruction) {
  if (instruction->result_value_id == LOOM_LLVMIR_VALUE_ID_INVALID) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM select has no result value");
  }
  loom_llvmir_type_id_t result_type_id =
      loom_llvmir_verify_value_type(module, instruction->result_value_id);
  IREE_RETURN_IF_ERROR(loom_llvmir_verify_expected_value_type(
      module, instruction->select.true_value, result_type_id));
  IREE_RETURN_IF_ERROR(loom_llvmir_verify_expected_value_type(
      module, instruction->select.false_value, result_type_id));

  loom_llvmir_type_id_t condition_type_id =
      loom_llvmir_verify_value_type(module, instruction->select.condition);
  if (condition_type_id == LOOM_LLVMIR_TYPE_ID_INVALID) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM select references unknown condition");
  }
  if (loom_llvmir_verify_type_is_i1(module, condition_type_id)) {
    return iree_ok_status();
  }

  const loom_llvmir_type_t* condition_type =
      loom_llvmir_verify_type(module, condition_type_id);
  const loom_llvmir_type_t* result_type =
      loom_llvmir_verify_type(module, result_type_id);
  if (!condition_type || condition_type->kind != LOOM_LLVMIR_TYPE_VECTOR ||
      !result_type || result_type->kind != LOOM_LLVMIR_TYPE_VECTOR ||
      condition_type->element_count != result_type->element_count ||
      !loom_llvmir_verify_type_is_i1(module, condition_type->element_type)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "LLVM vector select condition must be a same-lane i1 vector");
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_verify_extract_element(
    const loom_llvmir_module_t* module,
    const loom_llvmir_instruction_t* instruction) {
  if (instruction->result_value_id == LOOM_LLVMIR_VALUE_ID_INVALID) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM extractelement has no result value");
  }
  loom_llvmir_type_id_t vector_type_id = loom_llvmir_verify_value_type(
      module, instruction->extract_element.vector);
  const loom_llvmir_type_t* vector_type =
      loom_llvmir_verify_type(module, vector_type_id);
  if (!vector_type || vector_type->kind != LOOM_LLVMIR_TYPE_VECTOR) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM extractelement vector must be a vector");
  }
  IREE_RETURN_IF_ERROR(loom_llvmir_verify_expected_value_type(
      module, instruction->result_value_id, vector_type->element_type));
  if (!loom_llvmir_verify_type_is_scalar_integer(
          module, loom_llvmir_verify_value_type(
                      module, instruction->extract_element.index))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM extractelement index must be an integer");
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_verify_extract_value(
    const loom_llvmir_module_t* module,
    const loom_llvmir_instruction_t* instruction) {
  if (instruction->result_value_id == LOOM_LLVMIR_VALUE_ID_INVALID) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM extractvalue has no result value");
  }
  const loom_llvmir_type_id_t result_type_id =
      loom_llvmir_verify_value_type(module, instruction->result_value_id);
  if (result_type_id == LOOM_LLVMIR_TYPE_ID_INVALID) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM extractvalue has unknown result value");
  }
  const loom_llvmir_value_t* aggregate =
      loom_llvmir_verify_value(module, instruction->extract_value.aggregate);
  if (!aggregate) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM extractvalue references unknown aggregate");
  }
  const loom_llvmir_type_t* aggregate_type =
      loom_llvmir_verify_type(module, aggregate->type_id);
  if (!aggregate_type || aggregate_type->kind != LOOM_LLVMIR_TYPE_STRUCT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM extractvalue operand must be a struct");
  }
  if (instruction->extract_value.index >= aggregate_type->element_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM extractvalue index is out of bounds");
  }
  if (aggregate_type->element_types[instruction->extract_value.index] !=
      result_type_id) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM extractvalue result type mismatch");
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_verify_insert_element(
    const loom_llvmir_module_t* module,
    const loom_llvmir_instruction_t* instruction) {
  if (instruction->result_value_id == LOOM_LLVMIR_VALUE_ID_INVALID) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM insertelement has no result value");
  }
  loom_llvmir_type_id_t vector_type_id =
      loom_llvmir_verify_value_type(module, instruction->insert_element.vector);
  const loom_llvmir_type_t* vector_type =
      loom_llvmir_verify_type(module, vector_type_id);
  if (!vector_type || vector_type->kind != LOOM_LLVMIR_TYPE_VECTOR) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM insertelement vector must be a vector");
  }
  IREE_RETURN_IF_ERROR(loom_llvmir_verify_expected_value_type(
      module, instruction->result_value_id, vector_type_id));
  IREE_RETURN_IF_ERROR(loom_llvmir_verify_expected_value_type(
      module, instruction->insert_element.element, vector_type->element_type));
  if (!loom_llvmir_verify_type_is_scalar_integer(
          module, loom_llvmir_verify_value_type(
                      module, instruction->insert_element.index))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM insertelement index must be an integer");
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_verify_shuffle_vector(
    const loom_llvmir_module_t* module,
    const loom_llvmir_instruction_t* instruction) {
  if (instruction->result_value_id == LOOM_LLVMIR_VALUE_ID_INVALID) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM shufflevector has no result value");
  }
  loom_llvmir_type_id_t lhs_type_id =
      loom_llvmir_verify_value_type(module, instruction->shuffle_vector.lhs);
  const loom_llvmir_type_t* lhs_type =
      loom_llvmir_verify_type(module, lhs_type_id);
  if (!lhs_type || lhs_type->kind != LOOM_LLVMIR_TYPE_VECTOR) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM shufflevector inputs must be vectors");
  }
  IREE_RETURN_IF_ERROR(loom_llvmir_verify_expected_value_type(
      module, instruction->shuffle_vector.rhs, lhs_type_id));

  loom_llvmir_type_id_t result_type_id =
      loom_llvmir_verify_value_type(module, instruction->result_value_id);
  const loom_llvmir_type_t* result_type =
      loom_llvmir_verify_type(module, result_type_id);
  if (!result_type || result_type->kind != LOOM_LLVMIR_TYPE_VECTOR ||
      result_type->element_type != lhs_type->element_type) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "LLVM shufflevector result must be an input-element vector");
  }

  const loom_llvmir_value_t* mask_value =
      loom_llvmir_verify_value(module, instruction->shuffle_vector.mask);
  if (!mask_value ||
      mask_value->kind != LOOM_LLVMIR_VALUE_CONSTANT_INTEGER_VECTOR) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "LLVM shufflevector mask must be a constant integer vector");
  }
  const loom_llvmir_type_t* mask_type =
      loom_llvmir_verify_type(module, mask_value->type_id);
  if (!mask_type || mask_type->kind != LOOM_LLVMIR_TYPE_VECTOR ||
      mask_type->element_type >= module->type_count ||
      module->types[mask_type->element_type].kind != LOOM_LLVMIR_TYPE_INTEGER ||
      module->types[mask_type->element_type].bit_width != 32) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "LLVM shufflevector mask type must be a vector of i32");
  }
  if (result_type->element_count != mask_type->element_count ||
      mask_value->integer_vector.value_count != mask_type->element_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "LLVM shufflevector mask lanes must match the result lanes");
  }
  uint64_t max_index = (uint64_t)lhs_type->element_count * 2;
  for (iree_host_size_t i = 0; i < mask_value->integer_vector.value_count;
       ++i) {
    if (mask_value->integer_vector.values[i] >= max_index) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "LLVM shufflevector mask index is outside the input lanes");
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_verify_alloca(
    const loom_llvmir_module_t* module,
    const loom_llvmir_instruction_t* instruction) {
  if (instruction->result_value_id == LOOM_LLVMIR_VALUE_ID_INVALID) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM alloca has no result value");
  }
  const loom_llvmir_type_t* result_type = loom_llvmir_verify_type(
      module,
      loom_llvmir_verify_value_type(module, instruction->result_value_id));
  if (!result_type || result_type->kind != LOOM_LLVMIR_TYPE_POINTER) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM alloca result must be a pointer");
  }
  const loom_llvmir_type_t* element_type =
      loom_llvmir_verify_type(module, instruction->alloca.element_type);
  if (!element_type || element_type->kind == LOOM_LLVMIR_TYPE_VOID) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM alloca element type is invalid");
  }
  const loom_llvmir_type_t* count_type = loom_llvmir_verify_type(
      module, loom_llvmir_verify_value_type(module, instruction->alloca.count));
  if (!count_type || count_type->kind != LOOM_LLVMIR_TYPE_INTEGER) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM alloca count must be an integer");
  }
  return loom_llvmir_verify_memory_alignment(instruction->alloca.alignment);
}

static iree_status_t loom_llvmir_verify_instruction(
    const loom_llvmir_module_t* module, const loom_llvmir_function_t* function,
    const loom_llvmir_block_t* block,
    const loom_llvmir_instruction_t* instruction) {
  (void)block;
  switch (instruction->kind) {
    case LOOM_LLVMIR_INST_PHI:
      if (instruction->result_value_id == LOOM_LLVMIR_VALUE_ID_INVALID) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVM phi has no result value");
      }
      for (iree_host_size_t i = 0; i < instruction->phi.incoming_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_llvmir_verify_expected_value_type(
            module, instruction->phi.incoming[i].value,
            loom_llvmir_verify_value_type(module,
                                          instruction->result_value_id)));
        if (instruction->phi.incoming[i].predecessor >= function->block_count) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "LLVM phi references unknown predecessor");
        }
      }
      return iree_ok_status();
    case LOOM_LLVMIR_INST_BINOP:
      if (instruction->result_value_id == LOOM_LLVMIR_VALUE_ID_INVALID) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVM binop has no result value");
      }
      IREE_RETURN_IF_ERROR(loom_llvmir_verify_expected_value_type(
          module, instruction->binop.lhs,
          loom_llvmir_verify_value_type(module, instruction->result_value_id)));
      IREE_RETURN_IF_ERROR(loom_llvmir_verify_expected_value_type(
          module, instruction->binop.rhs,
          loom_llvmir_verify_value_type(module, instruction->result_value_id)));
      if (!loom_llvmir_verify_binop_accepts_type(
              module, instruction->binop.op,
              loom_llvmir_verify_type(
                  module, loom_llvmir_verify_value_type(
                              module, instruction->result_value_id)))) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVM binop opcode is invalid for result type");
      }
      return loom_llvmir_verify_binop_flags(instruction);
    case LOOM_LLVMIR_INST_UNOP:
      if (instruction->result_value_id == LOOM_LLVMIR_VALUE_ID_INVALID) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVM unop has no result value");
      }
      IREE_RETURN_IF_ERROR(loom_llvmir_verify_expected_value_type(
          module, instruction->unop.value,
          loom_llvmir_verify_value_type(module, instruction->result_value_id)));
      if (!loom_llvmir_verify_unop_accepts_type(
              module, instruction->unop.op,
              loom_llvmir_verify_type(
                  module, loom_llvmir_verify_value_type(
                              module, instruction->result_value_id)))) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVM unop opcode is invalid for result type");
      }
      return loom_llvmir_verify_unop_flags(instruction);
    case LOOM_LLVMIR_INST_ICMP:
      if (!loom_llvmir_verify_icmp_predicate(instruction->icmp.predicate)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unknown LLVM icmp predicate");
      }
      return loom_llvmir_verify_compare(
          module, instruction, instruction->icmp.lhs, instruction->icmp.rhs,
          loom_llvmir_verify_type_is_integer_or_pointer_like);
    case LOOM_LLVMIR_INST_FCMP:
      if (!loom_llvmir_verify_fcmp_predicate(instruction->fcmp.predicate)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unknown LLVM fcmp predicate");
      }
      return loom_llvmir_verify_compare(
          module, instruction, instruction->fcmp.lhs, instruction->fcmp.rhs,
          loom_llvmir_verify_type_is_float_like);
    case LOOM_LLVMIR_INST_SELECT:
      return loom_llvmir_verify_select(module, instruction);
    case LOOM_LLVMIR_INST_CAST:
      return loom_llvmir_verify_cast(module, instruction);
    case LOOM_LLVMIR_INST_GEP:
      if (instruction->result_value_id == LOOM_LLVMIR_VALUE_ID_INVALID) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVM gep has no result value");
      }
      if (!loom_llvmir_verify_value(module, instruction->gep.base)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVM gep references unknown base");
      }
      for (iree_host_size_t i = 0; i < instruction->gep.index_count; ++i) {
        if (!loom_llvmir_verify_value(module, instruction->gep.indices[i])) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "LLVM gep references unknown index");
        }
      }
      return iree_ok_status();
    case LOOM_LLVMIR_INST_ALLOCA:
      return loom_llvmir_verify_alloca(module, instruction);
    case LOOM_LLVMIR_INST_LOAD:
      if (instruction->result_value_id == LOOM_LLVMIR_VALUE_ID_INVALID) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVM load has no result value");
      }
      IREE_RETURN_IF_ERROR(loom_llvmir_verify_expected_value_type(
          module, instruction->result_value_id, instruction->load.result_type));
      if (!loom_llvmir_verify_value(module, instruction->load.pointer)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVM load references unknown pointer");
      }
      if (!loom_llvmir_verify_type_is_pointer_like(
              module, loom_llvmir_verify_type(
                          module, loom_llvmir_verify_value_type(
                                      module, instruction->load.pointer)))) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVM load pointer must be a pointer");
      }
      IREE_RETURN_IF_ERROR(
          loom_llvmir_verify_memory_alignment(instruction->load.alignment));
      IREE_RETURN_IF_ERROR(
          loom_llvmir_verify_memory_flags(instruction->load.flags));
      return loom_llvmir_verify_metadata_attachments(
          module, instruction->load.metadata_attachments,
          instruction->load.metadata_attachment_count);
    case LOOM_LLVMIR_INST_STORE:
      if (!loom_llvmir_verify_value(module, instruction->store.value) ||
          !loom_llvmir_verify_value(module, instruction->store.pointer)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVM store references unknown value");
      }
      if (!loom_llvmir_verify_type_is_pointer_like(
              module, loom_llvmir_verify_type(
                          module, loom_llvmir_verify_value_type(
                                      module, instruction->store.pointer)))) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVM store pointer must be a pointer");
      }
      IREE_RETURN_IF_ERROR(
          loom_llvmir_verify_memory_alignment(instruction->store.alignment));
      IREE_RETURN_IF_ERROR(
          loom_llvmir_verify_memory_flags(instruction->store.flags));
      return loom_llvmir_verify_metadata_attachments(
          module, instruction->store.metadata_attachments,
          instruction->store.metadata_attachment_count);
    case LOOM_LLVMIR_INST_ATOMIC_RMW:
      return loom_llvmir_verify_atomic_rmw(module, instruction);
    case LOOM_LLVMIR_INST_CMPXCHG:
      return loom_llvmir_verify_cmpxchg(module, instruction);
    case LOOM_LLVMIR_INST_EXTRACT_ELEMENT:
      return loom_llvmir_verify_extract_element(module, instruction);
    case LOOM_LLVMIR_INST_EXTRACT_VALUE:
      return loom_llvmir_verify_extract_value(module, instruction);
    case LOOM_LLVMIR_INST_INSERT_ELEMENT:
      return loom_llvmir_verify_insert_element(module, instruction);
    case LOOM_LLVMIR_INST_SHUFFLE_VECTOR:
      return loom_llvmir_verify_shuffle_vector(module, instruction);
    case LOOM_LLVMIR_INST_CALL: {
      loom_llvmir_function_t* callee =
          loom_llvmir_verify_function_ref(module, instruction->call.callee);
      if (!callee) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVM call references unknown callee");
      }
      if (callee->parameter_count != instruction->call.arg_count) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVM call arg count mismatch");
      }
      for (iree_host_size_t i = 0; i < instruction->call.arg_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_llvmir_verify_expected_value_type(
            module, instruction->call.args[i], callee->parameters[i].type_id));
      }
      return iree_ok_status();
    }
    case LOOM_LLVMIR_INST_INLINE_ASM:
      for (iree_host_size_t i = 0; i < instruction->inline_asm.arg_count; ++i) {
        if (!loom_llvmir_verify_value(module,
                                      instruction->inline_asm.args[i])) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "LLVM inline asm references unknown arg");
        }
      }
      return iree_ok_status();
    case LOOM_LLVMIR_INST_RET:
      if (!instruction->ret.has_value) {
        const loom_llvmir_type_t* return_type =
            loom_llvmir_verify_type(module, function->return_type);
        if (!return_type || return_type->kind != LOOM_LLVMIR_TYPE_VOID) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "LLVM ret void in non-void function");
        }
        return iree_ok_status();
      }
      return loom_llvmir_verify_expected_value_type(
          module, instruction->ret.value, function->return_type);
    case LOOM_LLVMIR_INST_BR:
      if (instruction->br.target >= function->block_count) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVM branch references unknown block");
      }
      return iree_ok_status();
    case LOOM_LLVMIR_INST_COND_BR:
      if (!loom_llvmir_verify_value(module, instruction->cond_br.condition)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVM cond_br references unknown condition");
      }
      const loom_llvmir_type_t* condition_type = loom_llvmir_verify_type(
          module, loom_llvmir_verify_value_type(
                      module, instruction->cond_br.condition));
      if (!condition_type || condition_type->kind != LOOM_LLVMIR_TYPE_INTEGER ||
          condition_type->bit_width != 1) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVM cond_br condition must be i1");
      }
      if (instruction->cond_br.true_block >= function->block_count ||
          instruction->cond_br.false_block >= function->block_count) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVM cond_br references unknown block");
      }
      return iree_ok_status();
    case LOOM_LLVMIR_INST_UNREACHABLE:
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown LLVM instruction kind");
  }
}

static iree_status_t loom_llvmir_verify_function(
    const loom_llvmir_module_t* module,
    const loom_llvmir_function_t* function) {
  if (function->kind == LOOM_LLVMIR_FUNCTION_DECLARATION) {
    return iree_ok_status();
  }
  if (function->block_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM function definition has no blocks");
  }
  for (iree_host_size_t i = 0; i < function->block_count; ++i) {
    const loom_llvmir_block_t* block = function->blocks[i];
    if (block->instruction_count == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "LLVM block has no terminator");
    }
    bool saw_non_phi = false;
    for (iree_host_size_t j = 0; j < block->instruction_count; ++j) {
      const loom_llvmir_instruction_t* instruction = &block->instructions[j];
      if (instruction->kind == LOOM_LLVMIR_INST_PHI && saw_non_phi) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVM phi appears after non-phi instruction");
      }
      if (instruction->kind != LOOM_LLVMIR_INST_PHI) saw_non_phi = true;
      if (loom_llvmir_is_terminator(instruction) &&
          j + 1 != block->instruction_count) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVM terminator is not last in block");
      }
      IREE_RETURN_IF_ERROR(
          loom_llvmir_verify_instruction(module, function, block, instruction));
    }
    if (!loom_llvmir_is_terminator(
            &block->instructions[block->instruction_count - 1])) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "LLVM block does not end with terminator");
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_verify_global(
    const loom_llvmir_module_t* module, const loom_llvmir_global_t* global) {
  if (global->value_type >= module->type_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM global references unknown value type");
  }
  if (module->types[global->value_type].kind == LOOM_LLVMIR_TYPE_VOID) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM global value type must not be void");
  }
  const loom_llvmir_value_t* global_value =
      loom_llvmir_verify_value(module, global->value_id);
  if (!global_value || global_value->kind != LOOM_LLVMIR_VALUE_GLOBAL ||
      global_value->global.global_id != global->id) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM global value table entry is invalid");
  }
  const loom_llvmir_type_t* pointer_type =
      loom_llvmir_verify_type(module, global_value->type_id);
  if (!pointer_type || pointer_type->kind != LOOM_LLVMIR_TYPE_POINTER ||
      pointer_type->address_space != global->address_space) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM global value must be a matching pointer");
  }
  const loom_llvmir_value_t* initializer =
      loom_llvmir_verify_value(module, global->initializer);
  if (!loom_llvmir_verify_value_is_constant(initializer)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM global initializer must be a constant");
  }
  if (initializer->type_id != global->value_type) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM global initializer type mismatch");
  }
  return loom_llvmir_verify_memory_alignment(global->alignment);
}

iree_status_t loom_llvmir_verify_module(const loom_llvmir_module_t* module) {
  for (iree_host_size_t i = 0; i < module->global_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_llvmir_verify_global(module, module->globals[i]));
  }
  for (iree_host_size_t i = 0; i < module->function_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_llvmir_verify_function(module, module->functions[i]));
  }
  return iree_ok_status();
}
