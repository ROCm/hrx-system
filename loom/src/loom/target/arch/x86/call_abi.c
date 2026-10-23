// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/call_abi.h"

#include "loom/ir/module.h"
#include "loom/target/arch/x86/register_classes.h"

static const loom_attribute_t* loom_x86_call_abi_find_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* entry = &attrs.entries[i];
    if (iree_string_view_equal(
            loom_string_table_get(&module->strings, entry->name_id), name)) {
      return &entry->value;
    }
  }
  return NULL;
}

iree_status_t loom_x86_call_abi_layout_make(
    loom_module_t* module, const loom_type_t* argument_types,
    uint16_t argument_count, const loom_type_t* result_types,
    uint16_t result_count, int64_t* argument_registers,
    int64_t* argument_offsets, int64_t* result_registers, uint32_t stack_bytes,
    uint8_t stack_alignment, loom_named_attr_slice_t* out_layout) {
  *out_layout = loom_named_attr_slice_empty();
  loom_type_t signature;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_function_type(module, argument_types, argument_count,
                                       result_types, result_count, &signature));

  loom_named_attr_t sysv_entries[5];
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, IREE_SV("argument_offsets"), &sysv_entries[0].name_id));
  sysv_entries[0].value = loom_attr_i64_array(argument_offsets, argument_count);
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, IREE_SV("argument_registers"), &sysv_entries[1].name_id));
  sysv_entries[1].value =
      loom_attr_i64_array(argument_registers, argument_count);
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, IREE_SV("result_registers"), &sysv_entries[2].name_id));
  sysv_entries[2].value = loom_attr_i64_array(result_registers, result_count);
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, IREE_SV("stack_alignment"), &sysv_entries[3].name_id));
  sysv_entries[3].value = loom_attr_i64(stack_alignment);
  IREE_RETURN_IF_ERROR(loom_module_intern_string(module, IREE_SV("stack_bytes"),
                                                 &sysv_entries[4].name_id));
  sysv_entries[4].value = loom_attr_i64(stack_bytes);
  loom_attribute_t sysv;
  IREE_RETURN_IF_ERROR(loom_module_make_canonical_attr_dict(
      module,
      loom_make_named_attr_slice(sysv_entries, IREE_ARRAYSIZE(sysv_entries)),
      &sysv));

  loom_named_attr_t layout_entries[2];
  IREE_RETURN_IF_ERROR(loom_module_intern_string(module, IREE_SV("signature"),
                                                 &layout_entries[0].name_id));
  layout_entries[0].value =
      loom_attr_type(loom_module_lookup_type_id(module, signature));
  IREE_RETURN_IF_ERROR(loom_module_intern_string(module, IREE_SV("sysv"),
                                                 &layout_entries[1].name_id));
  layout_entries[1].value = sysv;
  loom_attribute_t layout;
  IREE_RETURN_IF_ERROR(loom_module_make_canonical_attr_dict(
      module,
      loom_make_named_attr_slice(layout_entries,
                                 IREE_ARRAYSIZE(layout_entries)),
      &layout));
  *out_layout = loom_attr_as_dict(layout);
  return iree_ok_status();
}

bool loom_x86_call_abi_layout_parse(const loom_module_t* module,
                                    loom_named_attr_slice_t layout,
                                    uint16_t argument_count,
                                    uint16_t result_count,
                                    loom_x86_call_abi_layout_t* out_layout) {
  *out_layout = (loom_x86_call_abi_layout_t){0};
  if (layout.count == 0) {
    return true;
  }
  if (layout.count != 1 && layout.count != 2) {
    return false;
  }

  const loom_attribute_t* signature_attr =
      loom_x86_call_abi_find_attr(module, layout, IREE_SV("signature"));
  if (!signature_attr || signature_attr->kind != LOOM_ATTR_TYPE) {
    return false;
  }
  const loom_type_t signature =
      loom_type_table_get(&module->types, signature_attr->type_id);
  if (!loom_type_is_function(signature)) {
    return false;
  }
  const loom_func_type_data_t* signature_data = loom_type_func_data(signature);
  if (signature_data->arg_count != argument_count ||
      signature_data->result_count != result_count) {
    return false;
  }
  out_layout->signature = signature_data;
  if (layout.count == 1) {
    return true;
  }

  const loom_attribute_t* sysv_attr =
      loom_x86_call_abi_find_attr(module, layout, IREE_SV("sysv"));
  if (!sysv_attr || sysv_attr->kind != LOOM_ATTR_DICT ||
      sysv_attr->count != 5) {
    return false;
  }
  const loom_named_attr_slice_t sysv = loom_attr_as_dict(*sysv_attr);
  const loom_attribute_t* argument_offsets_attr =
      loom_x86_call_abi_find_attr(module, sysv, IREE_SV("argument_offsets"));
  const loom_attribute_t* argument_registers_attr =
      loom_x86_call_abi_find_attr(module, sysv, IREE_SV("argument_registers"));
  const loom_attribute_t* result_registers_attr =
      loom_x86_call_abi_find_attr(module, sysv, IREE_SV("result_registers"));
  const loom_attribute_t* stack_alignment_attr =
      loom_x86_call_abi_find_attr(module, sysv, IREE_SV("stack_alignment"));
  const loom_attribute_t* stack_bytes_attr =
      loom_x86_call_abi_find_attr(module, sysv, IREE_SV("stack_bytes"));
  if (!argument_offsets_attr ||
      argument_offsets_attr->kind != LOOM_ATTR_I64_ARRAY ||
      argument_offsets_attr->count != argument_count ||
      !argument_registers_attr ||
      argument_registers_attr->kind != LOOM_ATTR_I64_ARRAY ||
      argument_registers_attr->count != argument_count ||
      !result_registers_attr ||
      result_registers_attr->kind != LOOM_ATTR_I64_ARRAY ||
      result_registers_attr->count != result_count || !stack_alignment_attr ||
      stack_alignment_attr->kind != LOOM_ATTR_I64 || !stack_bytes_attr ||
      stack_bytes_attr->kind != LOOM_ATTR_I64 ||
      stack_alignment_attr->i64 < 0 || stack_alignment_attr->i64 > UINT8_MAX ||
      stack_bytes_attr->i64 < 0 || stack_bytes_attr->i64 > UINT32_MAX) {
    return false;
  }

  const loom_i64_array_t argument_offsets =
      loom_attr_as_i64_array(*argument_offsets_attr);
  const loom_i64_array_t argument_registers =
      loom_attr_as_i64_array(*argument_registers_attr);
  bool has_stack_argument = false;
  for (uint16_t i = 0; i < argument_count; ++i) {
    const int64_t register_index = argument_registers.values[i];
    const int64_t stack_offset = argument_offsets.values[i];
    const bool is_register = register_index >= 0 &&
                             register_index <= UINT32_MAX && stack_offset == -1;
    const bool is_stack =
        register_index == -1 && stack_offset >= 0 && stack_offset <= UINT32_MAX;
    if (!is_register && !is_stack) {
      return false;
    }
    has_stack_argument |= is_stack;
  }
  const loom_i64_array_t result_registers =
      loom_attr_as_i64_array(*result_registers_attr);
  for (uint16_t i = 0; i < result_count; ++i) {
    if (result_registers.values[i] < 0 ||
        result_registers.values[i] > UINT32_MAX) {
      return false;
    }
  }

  const uint32_t stack_bytes = (uint32_t)stack_bytes_attr->i64;
  const uint8_t stack_alignment = (uint8_t)stack_alignment_attr->i64;
  if (has_stack_argument) {
    if (stack_bytes == 0 || stack_alignment < 16 ||
        !iree_host_size_is_power_of_two(stack_alignment)) {
      return false;
    }
    for (uint16_t i = 0; i < argument_count; ++i) {
      const int64_t stack_offset = argument_offsets.values[i];
      if (stack_offset >= 0 && (uint64_t)stack_offset >= stack_bytes) {
        return false;
      }
    }
  } else if (stack_bytes != 0 || stack_alignment != 0) {
    return false;
  }

  *out_layout = (loom_x86_call_abi_layout_t){
      .signature = signature_data,
      .argument_registers = argument_registers,
      .argument_offsets = argument_offsets,
      .result_registers = result_registers,
      .stack_bytes = stack_bytes,
      .stack_alignment = stack_alignment,
      .is_exact_sysv = true,
  };
  return true;
}

uint16_t loom_x86_call_abi_register_byte_length(uint16_t register_class) {
  switch (register_class) {
    case LOOM_X86_REGISTER_CLASS_GPR32:
      return 4;
    case LOOM_X86_REGISTER_CLASS_GPR64:
      return 8;
    case LOOM_X86_REGISTER_CLASS_XMM:
      return 16;
    case LOOM_X86_REGISTER_CLASS_YMM:
      return 32;
    case LOOM_X86_REGISTER_CLASS_ZMM:
      return 64;
    default:
      return 0;
  }
}

static bool loom_x86_call_abi_classify_scalar(
    loom_type_t source_type,
    loom_x86_call_abi_classification_t* out_classification) {
  if (!loom_type_is_scalar(source_type)) {
    return false;
  }
  switch (loom_type_element_type(source_type)) {
    case LOOM_SCALAR_TYPE_I1:
      *out_classification = (loom_x86_call_abi_classification_t){
          .abi_class = LOOM_X86_CALL_ABI_CLASS_INTEGER,
          .carrier_register_class = LOOM_X86_REGISTER_CLASS_GPR32,
          .boundary_register_class = LOOM_X86_REGISTER_CLASS_GPR32,
          .byte_length = 1,
          .byte_alignment = 1,
          .action = LOOM_X86_CALL_ABI_VALUE_ACTION_NORMALIZE_I1,
      };
      return true;
    case LOOM_SCALAR_TYPE_I8:
    case LOOM_SCALAR_TYPE_F8E4M3:
    case LOOM_SCALAR_TYPE_F8E5M2:
      *out_classification = (loom_x86_call_abi_classification_t){
          .abi_class = LOOM_X86_CALL_ABI_CLASS_INTEGER,
          .carrier_register_class = LOOM_X86_REGISTER_CLASS_GPR32,
          .boundary_register_class = LOOM_X86_REGISTER_CLASS_GPR32,
          .byte_length = 1,
          .byte_alignment = 1,
          .action = LOOM_X86_CALL_ABI_VALUE_ACTION_NORMALIZE_I8,
      };
      return true;
    case LOOM_SCALAR_TYPE_I16:
      *out_classification = (loom_x86_call_abi_classification_t){
          .abi_class = LOOM_X86_CALL_ABI_CLASS_INTEGER,
          .carrier_register_class = LOOM_X86_REGISTER_CLASS_GPR32,
          .boundary_register_class = LOOM_X86_REGISTER_CLASS_GPR32,
          .byte_length = 2,
          .byte_alignment = 2,
          .action = LOOM_X86_CALL_ABI_VALUE_ACTION_NORMALIZE_I16,
      };
      return true;
    case LOOM_SCALAR_TYPE_F16:
    case LOOM_SCALAR_TYPE_BF16:
      *out_classification = (loom_x86_call_abi_classification_t){
          .abi_class = LOOM_X86_CALL_ABI_CLASS_SSE,
          .carrier_register_class = LOOM_X86_REGISTER_CLASS_GPR32,
          .boundary_register_class = LOOM_X86_REGISTER_CLASS_XMM,
          .byte_length = 2,
          .byte_alignment = 2,
      };
      return true;
    case LOOM_SCALAR_TYPE_I32:
      *out_classification = (loom_x86_call_abi_classification_t){
          .abi_class = LOOM_X86_CALL_ABI_CLASS_INTEGER,
          .carrier_register_class = LOOM_X86_REGISTER_CLASS_GPR32,
          .boundary_register_class = LOOM_X86_REGISTER_CLASS_GPR32,
          .byte_length = 4,
          .byte_alignment = 4,
      };
      return true;
    case LOOM_SCALAR_TYPE_I64:
    case LOOM_SCALAR_TYPE_INDEX:
    case LOOM_SCALAR_TYPE_OFFSET:
      *out_classification = (loom_x86_call_abi_classification_t){
          .abi_class = LOOM_X86_CALL_ABI_CLASS_INTEGER,
          .carrier_register_class = LOOM_X86_REGISTER_CLASS_GPR64,
          .boundary_register_class = LOOM_X86_REGISTER_CLASS_GPR64,
          .byte_length = 8,
          .byte_alignment = 8,
      };
      return true;
    case LOOM_SCALAR_TYPE_F32:
      *out_classification = (loom_x86_call_abi_classification_t){
          .abi_class = LOOM_X86_CALL_ABI_CLASS_SSE,
          .carrier_register_class = LOOM_X86_REGISTER_CLASS_XMM,
          .boundary_register_class = LOOM_X86_REGISTER_CLASS_XMM,
          .byte_length = 4,
          .byte_alignment = 4,
      };
      return true;
    case LOOM_SCALAR_TYPE_F64:
      *out_classification = (loom_x86_call_abi_classification_t){
          .abi_class = LOOM_X86_CALL_ABI_CLASS_SSE,
          .carrier_register_class = LOOM_X86_REGISTER_CLASS_XMM,
          .boundary_register_class = LOOM_X86_REGISTER_CLASS_XMM,
          .byte_length = 8,
          .byte_alignment = 8,
      };
      return true;
    default:
      return false;
  }
}

static bool loom_x86_call_abi_vector_element_bit_width(
    loom_scalar_type_t scalar_type, uint32_t* out_bit_width) {
  static const loom_scalar_type_set_t kElementTypes =
      LOOM_SCALAR_TYPE_SET_I8 | LOOM_SCALAR_TYPE_SET_I16 |
      LOOM_SCALAR_TYPE_SET_I32 | LOOM_SCALAR_TYPE_SET_I64 |
      LOOM_SCALAR_TYPE_SET_F8E4M3 | LOOM_SCALAR_TYPE_SET_F8E5M2 |
      LOOM_SCALAR_TYPE_SET_16BIT_FLOAT | LOOM_SCALAR_TYPE_SET_F32 |
      LOOM_SCALAR_TYPE_SET_F64;
  if (!loom_scalar_type_set_contains(kElementTypes, scalar_type)) {
    return false;
  }
  *out_bit_width = (uint32_t)loom_scalar_type_bitwidth(scalar_type);
  return true;
}

static bool loom_x86_call_abi_classify_vector(
    loom_type_t source_type,
    loom_x86_call_abi_classification_t* out_classification) {
  if (!loom_type_is_vector(source_type) || loom_type_rank(source_type) != 1 ||
      !loom_type_is_all_static(source_type)) {
    return false;
  }
  const int64_t lane_count = loom_type_dim_static_size_at(source_type, 0);
  if (lane_count <= 0) {
    return false;
  }
  loom_x86_register_class_t register_class = LOOM_X86_REGISTER_CLASS_GPR32;
  if (loom_type_element_type(source_type) == LOOM_SCALAR_TYPE_I1) {
    switch (lane_count) {
      case 2:
      case 4:
      case 8:
      case 16:
        register_class = LOOM_X86_REGISTER_CLASS_XMM;
        break;
      case 32:
        register_class = LOOM_X86_REGISTER_CLASS_YMM;
        break;
      case 64:
        register_class = LOOM_X86_REGISTER_CLASS_ZMM;
        break;
      default:
        return false;
    }
  } else {
    uint32_t element_bit_width = 0;
    if (!loom_x86_call_abi_vector_element_bit_width(
            loom_type_element_type(source_type), &element_bit_width)) {
      return false;
    }
    const uint64_t vector_bit_width = (uint64_t)lane_count * element_bit_width;
    if (vector_bit_width > UINT32_MAX ||
        !loom_x86_register_class_for_vector_bit_width(
            (uint32_t)vector_bit_width, &register_class)) {
      return false;
    }
    const uint16_t byte_length = (uint16_t)(vector_bit_width / 8);
    *out_classification = (loom_x86_call_abi_classification_t){
        .abi_class = LOOM_X86_CALL_ABI_CLASS_SSE,
        .carrier_register_class = register_class,
        .boundary_register_class = register_class,
        .byte_length = byte_length,
        .byte_alignment = (uint8_t)byte_length,
    };
    return true;
  }
  const uint16_t byte_length =
      loom_x86_call_abi_register_byte_length(register_class);
  *out_classification = (loom_x86_call_abi_classification_t){
      .abi_class = LOOM_X86_CALL_ABI_CLASS_SSE,
      .carrier_register_class = register_class,
      .boundary_register_class = register_class,
      .byte_length = byte_length,
      .byte_alignment = (uint8_t)byte_length,
  };
  return true;
}

bool loom_x86_call_abi_classify_source_type(
    loom_type_t source_type,
    loom_x86_call_abi_classification_t* out_classification) {
  if (loom_type_is_buffer(source_type) || loom_type_is_view(source_type)) {
    *out_classification = (loom_x86_call_abi_classification_t){
        .abi_class = LOOM_X86_CALL_ABI_CLASS_INTEGER,
        .carrier_register_class = LOOM_X86_REGISTER_CLASS_GPR64,
        .boundary_register_class = LOOM_X86_REGISTER_CLASS_GPR64,
        .byte_length = 8,
        .byte_alignment = 8,
    };
    return true;
  }
  return loom_x86_call_abi_classify_scalar(source_type, out_classification) ||
         loom_x86_call_abi_classify_vector(source_type, out_classification);
}

bool loom_x86_call_abi_classify_source_carrier(
    loom_type_t source_type, uint16_t carrier_register_class,
    loom_x86_call_abi_classification_t* out_classification) {
  if (!loom_x86_call_abi_classify_source_type(source_type,
                                              out_classification)) {
    return false;
  }
  if (out_classification->carrier_register_class == carrier_register_class) {
    return true;
  }
  if (carrier_register_class != LOOM_X86_REGISTER_CLASS_XMM ||
      !loom_type_is_scalar(source_type)) {
    return false;
  }
  const loom_scalar_type_t scalar_type = loom_type_element_type(source_type);
  if (scalar_type != LOOM_SCALAR_TYPE_F16 &&
      scalar_type != LOOM_SCALAR_TYPE_BF16) {
    return false;
  }
  out_classification->carrier_register_class = carrier_register_class;
  return true;
}
