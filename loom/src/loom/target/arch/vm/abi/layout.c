// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/abi/layout.h"

#include <inttypes.h>

#include "iree/vm/module.h"
#include "loom/ir/module.h"
#include "loom/ops/func/reference.h"
#include "loom/ops/type_registry.h"
#include "loom/target/arch/vm/descriptors.h"
#include "loom/target/registers.h"

iree_vm_scalar_type_t loom_vm_call_abi_scalar_type(
    loom_scalar_type_t scalar_type) {
  switch (scalar_type) {
    case LOOM_SCALAR_TYPE_I1:
    case LOOM_SCALAR_TYPE_I8:
      return IREE_VM_SCALAR_TYPE_I8;
    case LOOM_SCALAR_TYPE_I16:
      return IREE_VM_SCALAR_TYPE_I16;
    case LOOM_SCALAR_TYPE_I32:
      return IREE_VM_SCALAR_TYPE_I32;
    case LOOM_SCALAR_TYPE_INDEX:
    case LOOM_SCALAR_TYPE_OFFSET:
    case LOOM_SCALAR_TYPE_I64:
      return IREE_VM_SCALAR_TYPE_I64;
    case LOOM_SCALAR_TYPE_F8E4M3:
      return IREE_VM_SCALAR_TYPE_F8E4M3FN;
    case LOOM_SCALAR_TYPE_F8E5M2:
      return IREE_VM_SCALAR_TYPE_F8E5M2;
    case LOOM_SCALAR_TYPE_F16:
      return IREE_VM_SCALAR_TYPE_F16;
    case LOOM_SCALAR_TYPE_BF16:
      return IREE_VM_SCALAR_TYPE_BF16;
    case LOOM_SCALAR_TYPE_F32:
      return IREE_VM_SCALAR_TYPE_F32;
    case LOOM_SCALAR_TYPE_F64:
      return IREE_VM_SCALAR_TYPE_F64;
    default:
      return IREE_VM_SCALAR_TYPE_NONE;
  }
}

bool loom_vm_call_abi_try_classify_logical_type(
    const loom_module_t* module, loom_type_t type,
    loom_vm_call_abi_register_layout_t* out_layout) {
  *out_layout = (loom_vm_call_abi_register_layout_t){0};
  if (loom_type_is_scalar(type)) {
    if (loom_vm_call_abi_scalar_type(loom_type_element_type(type)) ==
        IREE_VM_SCALAR_TYPE_NONE) {
      return false;
    }
    *out_layout = (loom_vm_call_abi_register_layout_t){
        .bank = LOOM_VM_CALL_ABI_BANK_VALUE,
        .unit_count = 1,
    };
    return true;
  }
  if (loom_type_is_vector(type)) {
    uint64_t element_count = 0;
    if (!loom_type_static_element_count(type, &element_count) ||
        element_count == 0 ||
        element_count > LOOM_VM_CALL_ABI_MAX_FIELD_UNIT_COUNT ||
        loom_vm_call_abi_scalar_type(loom_type_element_type(type)) ==
            IREE_VM_SCALAR_TYPE_NONE) {
      return false;
    }
    *out_layout = (loom_vm_call_abi_register_layout_t){
        .bank = LOOM_VM_CALL_ABI_BANK_VALUE,
        .unit_count = (uint16_t)element_count,
    };
    return true;
  }
  if (loom_type_is_buffer(type)) {
    *out_layout = (loom_vm_call_abi_register_layout_t){
        .bank = LOOM_VM_CALL_ABI_BANK_REF,
        .unit_count = 1,
    };
    return true;
  }
  if (loom_func_ref_type_isa(type)) {
    if (!loom_type_is_function(loom_func_ref_resolve_signature(module, type))) {
      return false;
    }
    *out_layout = (loom_vm_call_abi_register_layout_t){
        .bank = LOOM_VM_CALL_ABI_BANK_FUNCTION,
        .unit_count = 1,
    };
    return true;
  }
  const loom_type_descriptor_t* descriptor =
      loom_type_registry_resolve(module, type);
  if (descriptor == NULL ||
      descriptor->semantics.semantic != LOOM_TYPE_SEMANTIC_MANAGED_REFERENCE) {
    return false;
  }
  *out_layout = (loom_vm_call_abi_register_layout_t){
      .bank = LOOM_VM_CALL_ABI_BANK_REF,
      .unit_count = 1,
  };
  return true;
}

iree_status_t loom_vm_call_abi_classify_type(
    const loom_module_t* module, loom_type_t type,
    loom_vm_call_abi_register_layout_t* out_layout) {
  *out_layout = (loom_vm_call_abi_register_layout_t){0};
  if (!loom_low_type_is_register(type) ||
      loom_low_register_type_descriptor_set_stable_id(type) !=
          VM_CORE_DESCRIPTOR_SET_ID) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM callable fields must use vm.core registers");
  }

  const loom_type_t* value_type = loom_type_register_value_type(type);
  if (value_type == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "VM callable fields must retain their logical type");
  }
  loom_vm_call_abi_register_layout_t logical_layout = {0};
  if (!loom_vm_call_abi_try_classify_logical_type(module, *value_type,
                                                  &logical_layout)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "VM callable register carries an unsupported logical type");
  }
  switch (loom_low_register_type_class_id(type)) {
    case VM_CORE_REG_CLASS_ID_VALUE:
      out_layout->bank = LOOM_VM_CALL_ABI_BANK_VALUE;
      break;
    case VM_CORE_REG_CLASS_ID_REF:
      out_layout->bank = LOOM_VM_CALL_ABI_BANK_REF;
      break;
    case VM_CORE_REG_CLASS_ID_FUNCTION:
      out_layout->bank = LOOM_VM_CALL_ABI_BANK_FUNCTION;
      break;
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "VM callable field uses unknown register class %" PRIu16,
          loom_low_register_type_class_id(type));
  }
  const uint32_t register_unit_count = loom_low_register_type_unit_count(type);
  if (out_layout->bank != logical_layout.bank ||
      register_unit_count != logical_layout.unit_count) {
    *out_layout = (loom_vm_call_abi_register_layout_t){0};
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "VM callable register layout does not match its logical type");
  }
  out_layout->unit_count = logical_layout.unit_count;
  return iree_ok_status();
}

static iree_status_t loom_vm_call_abi_layout_resolve_signature_entry(
    const loom_module_t* module, loom_named_attr_slice_t abi_layout,
    iree_string_view_t key, bool required, loom_type_t* out_signature) {
  *out_signature = loom_type_none();
  const loom_string_id_t signature_key = loom_module_lookup_string(module, key);
  const loom_attribute_t* signature_attr = NULL;
  for (iree_host_size_t i = 0; i < abi_layout.count; ++i) {
    const loom_named_attr_t* entry = &abi_layout.entries[i];
    if (entry->name_id != signature_key) continue;
    if (signature_attr != NULL) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "VM abi_layout has duplicate %.*s entries",
                              (int)key.size, key.data);
    }
    signature_attr = &entry->value;
  }
  if (signature_attr == NULL && !required) return iree_ok_status();
  if (signature_attr == NULL || signature_attr->kind != LOOM_ATTR_TYPE ||
      signature_attr->type_id >= module->types.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM abi_layout requires a valid %.*s type entry",
                            (int)key.size, key.data);
  }
  const loom_type_t signature = module->types.entries[signature_attr->type_id];
  if (!loom_type_is_function(signature)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM abi_layout %.*s must be a function type",
                            (int)key.size, key.data);
  }
  *out_signature = signature;
  return iree_ok_status();
}

iree_status_t loom_vm_call_abi_layout_resolve_signature(
    const loom_module_t* module, loom_named_attr_slice_t abi_layout,
    loom_type_t* out_signature) {
  return loom_vm_call_abi_layout_resolve_signature_entry(
      module, abi_layout, IREE_SV("signature"), /*required=*/true,
      out_signature);
}

static iree_status_t loom_vm_call_abi_layout_validate_authored_signature(
    loom_type_t abi_signature, loom_type_t authored_signature) {
  if (!loom_type_is_function(abi_signature) ||
      !loom_type_is_function(authored_signature)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "VM authored signature validation requires function types");
  }
  const uint16_t abi_argument_count = loom_type_func_arg_count(abi_signature);
  const uint16_t authored_argument_count =
      loom_type_func_arg_count(authored_signature);
  const uint16_t abi_result_count = loom_type_func_result_count(abi_signature);
  const uint16_t authored_result_count =
      loom_type_func_result_count(authored_signature);
  if (authored_argument_count > abi_argument_count ||
      authored_result_count > abi_result_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "VM authored signature exceeds the complete ABI signature");
  }
  const loom_type_t* abi_argument_types =
      loom_type_func_arg_types(abi_signature);
  const loom_type_t* authored_argument_types =
      loom_type_func_arg_types(authored_signature);
  for (uint16_t i = 0; i < authored_argument_count; ++i) {
    if (!loom_type_equal(authored_argument_types[i], abi_argument_types[i])) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "VM authored argument %u is not an ABI-signature prefix field",
          (unsigned)i);
    }
  }
  const loom_type_t* abi_result_types =
      loom_type_func_result_types(abi_signature);
  const loom_type_t* authored_result_types =
      loom_type_func_result_types(authored_signature);
  for (uint16_t i = 0; i < authored_result_count; ++i) {
    if (!loom_type_equal(authored_result_types[i], abi_result_types[i])) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "VM authored result %u is not an ABI-signature prefix field",
          (unsigned)i);
    }
  }
  return iree_ok_status();
}

iree_status_t loom_vm_call_abi_layout_resolve_authored_signature(
    const loom_module_t* module, loom_named_attr_slice_t abi_layout,
    loom_type_t abi_signature, loom_type_t* out_authored_signature) {
  *out_authored_signature = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_vm_call_abi_layout_resolve_signature_entry(
      module, abi_layout, IREE_SV("authored_signature"), /*required=*/false,
      out_authored_signature));
  if (loom_type_kind(*out_authored_signature) == LOOM_TYPE_NONE) {
    *out_authored_signature = abi_signature;
    return iree_ok_status();
  }
  return loom_vm_call_abi_layout_validate_authored_signature(
      abi_signature, *out_authored_signature);
}

enum {
  // 'f' plus every decimal digit of a u16 and the NUL terminator.
  LOOM_VM_CALL_ABI_FIELD_KEY_CAPACITY = 7,
};

static iree_string_view_t loom_vm_call_abi_format_field_key(
    uint16_t field_ordinal, char* buffer) {
  buffer[0] = 'f';
  for (iree_host_size_t i = 5; i > 0; --i) {
    buffer[i] = (char)('0' + field_ordinal % 10);
    field_ordinal /= 10;
  }
  buffer[6] = '\0';
  return iree_make_string_view(buffer, LOOM_VM_CALL_ABI_FIELD_KEY_CAPACITY - 1);
}

static iree_status_t loom_vm_call_abi_layout_resolve_name_table(
    const loom_module_t* module, loom_named_attr_slice_t abi_layout,
    iree_string_view_t table_key, uint16_t expected_count,
    loom_named_attr_slice_t* out_names) {
  *out_names = loom_named_attr_slice_empty();
  const loom_string_id_t table_key_id =
      loom_module_lookup_string(module, table_key);
  if (table_key_id == LOOM_STRING_ID_INVALID) return iree_ok_status();
  const loom_attribute_t* table_attr = NULL;
  for (iree_host_size_t i = 0; i < abi_layout.count; ++i) {
    const loom_named_attr_t* entry = &abi_layout.entries[i];
    if (entry->name_id != table_key_id) continue;
    table_attr = &entry->value;
    break;
  }
  if (table_attr == NULL) return iree_ok_status();
  if (table_attr->kind != LOOM_ATTR_DICT ||
      table_attr->count != expected_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM abi_layout %.*s must contain %u string entries",
                            (int)table_key.size, table_key.data,
                            (unsigned)expected_count);
  }

  const loom_named_attr_slice_t names = loom_attr_as_dict(*table_attr);
  for (uint16_t i = 0; i < expected_count; ++i) {
    char expected_key_storage[LOOM_VM_CALL_ABI_FIELD_KEY_CAPACITY];
    const iree_string_view_t expected_key =
        loom_vm_call_abi_format_field_key(i, expected_key_storage);
    const loom_named_attr_t* entry = &names.entries[i];
    if (entry->name_id >= module->strings.count ||
        !iree_string_view_equal(module->strings.entries[entry->name_id],
                                expected_key) ||
        entry->value.kind != LOOM_ATTR_STRING ||
        entry->value.string_id >= module->strings.count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "VM abi_layout %.*s field %u is not a canonical string entry",
          (int)table_key.size, table_key.data, (unsigned)i);
    }
  }
  *out_names = names;
  return iree_ok_status();
}

iree_status_t loom_vm_call_abi_layout_resolve_presentation_names(
    const loom_module_t* module, loom_named_attr_slice_t abi_layout,
    uint16_t argument_count, uint16_t result_count,
    loom_named_attr_slice_t* out_argument_names,
    loom_named_attr_slice_t* out_result_names) {
  *out_argument_names = loom_named_attr_slice_empty();
  *out_result_names = loom_named_attr_slice_empty();
  IREE_RETURN_IF_ERROR(loom_vm_call_abi_layout_resolve_name_table(
      module, abi_layout, IREE_SV("argument_names"), argument_count,
      out_argument_names));
  return loom_vm_call_abi_layout_resolve_name_table(
      module, abi_layout, IREE_SV("result_names"), result_count,
      out_result_names);
}

static uint16_t* loom_vm_call_abi_bank_count(
    loom_vm_call_abi_bank_counts_t* counts, loom_vm_call_abi_bank_t bank) {
  switch (bank) {
    case LOOM_VM_CALL_ABI_BANK_VALUE:
      return &counts->value;
    case LOOM_VM_CALL_ABI_BANK_REF:
      return &counts->ref;
    case LOOM_VM_CALL_ABI_BANK_FUNCTION:
      return &counts->function;
    default:
      IREE_ASSERT_UNREACHABLE("valid VM call ABI bank");
      IREE_BUILTIN_UNREACHABLE();
  }
}

uint16_t loom_vm_call_abi_overflow_count(uint16_t count) {
  return count > IREE_VM_CALL_DIRECT_REGISTER_COUNT
             ? (uint16_t)(count - IREE_VM_CALL_DIRECT_REGISTER_COUNT)
             : 0;
}

static iree_status_t loom_vm_call_abi_count_values(
    const loom_module_t* module, const loom_value_id_t* values,
    iree_host_size_t value_count, loom_vm_call_abi_bank_counts_t* out_counts) {
  *out_counts = (loom_vm_call_abi_bank_counts_t){0};
  if (value_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VM call field count exceeds u16");
  }
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    loom_vm_call_abi_register_layout_t register_layout = {0};
    IREE_RETURN_IF_ERROR(loom_vm_call_abi_classify_type(
        module, loom_module_value_type(module, values[i]), &register_layout));
    uint16_t* bank_count =
        loom_vm_call_abi_bank_count(out_counts, register_layout.bank);
    if (register_layout.unit_count > UINT16_MAX - *bank_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "VM call register count exceeds u16");
    }
    *bank_count = (uint16_t)(*bank_count + register_layout.unit_count);
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_call_abi_side_layout_build(
    const loom_module_t* module, const loom_type_t* types, uint16_t type_count,
    loom_vm_call_abi_field_layout_t* fields,
    loom_vm_call_abi_side_layout_t* out_layout) {
  *out_layout = (loom_vm_call_abi_side_layout_t){
      .fields = type_count != 0 ? fields : NULL,
      .field_count = type_count,
  };
  for (uint16_t i = 0; i < type_count; ++i) {
    loom_vm_call_abi_register_layout_t register_layout = {0};
    IREE_RETURN_IF_ERROR(
        loom_vm_call_abi_classify_type(module, types[i], &register_layout));
    uint16_t* bank_count = loom_vm_call_abi_bank_count(&out_layout->bank_counts,
                                                       register_layout.bank);
    if (register_layout.unit_count > UINT16_MAX - *bank_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "VM call register count exceeds u16");
    }
    fields[i] = (loom_vm_call_abi_field_layout_t){
        .bank = register_layout.bank,
        .unit_count = register_layout.unit_count,
        .bank_ordinal = *bank_count,
    };
    *bank_count = (uint16_t)(*bank_count + register_layout.unit_count);
  }
  return iree_ok_status();
}

iree_status_t loom_vm_call_abi_layout_build(
    const loom_module_t* module, loom_type_t signature,
    iree_arena_allocator_t* arena, loom_vm_call_abi_layout_t* out_layout) {
  *out_layout = (loom_vm_call_abi_layout_t){0};
  if (!loom_type_is_function(signature)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM call ABI layout requires a function type");
  }

  const uint16_t argument_count = loom_type_func_arg_count(signature);
  const uint16_t result_count = loom_type_func_result_count(signature);
  const iree_host_size_t field_count =
      (iree_host_size_t)argument_count + result_count;
  loom_vm_call_abi_field_layout_t* fields = NULL;
  if (field_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, field_count, sizeof(*fields), (void**)&fields));
  }

  loom_vm_call_abi_layout_t layout = {
      .signature = signature,
  };
  IREE_RETURN_IF_ERROR(loom_vm_call_abi_side_layout_build(
      module, loom_type_func_arg_types(signature), argument_count, fields,
      &layout.arguments));
  IREE_RETURN_IF_ERROR(loom_vm_call_abi_side_layout_build(
      module, loom_type_func_result_types(signature), result_count,
      fields != NULL ? fields + argument_count : NULL, &layout.results));
  *out_layout = layout;
  return iree_ok_status();
}

iree_status_t loom_vm_call_abi_packet_layout_build(
    const loom_module_t* module, const loom_value_id_t* arguments,
    iree_host_size_t argument_count, const loom_value_id_t* results,
    iree_host_size_t result_count,
    loom_vm_call_abi_packet_layout_t* out_layout) {
  *out_layout = (loom_vm_call_abi_packet_layout_t){0};
  loom_vm_call_abi_packet_layout_t layout = {0};
  IREE_RETURN_IF_ERROR(loom_vm_call_abi_count_values(
      module, arguments, argument_count, &layout.arguments));
  IREE_RETURN_IF_ERROR(loom_vm_call_abi_count_values(
      module, results, result_count, &layout.results));

  const uint16_t direct_ref_argument_count =
      iree_min(IREE_VM_CALL_DIRECT_REGISTER_COUNT, layout.arguments.ref);
  layout.direct_ref_move_mask =
      direct_ref_argument_count == IREE_VM_CALL_DIRECT_REGISTER_COUNT
          ? UINT16_MAX
          : (uint16_t)((1u << direct_ref_argument_count) - 1u);

  const uint32_t value_overflow_count =
      (uint32_t)loom_vm_call_abi_overflow_count(layout.arguments.value) +
      loom_vm_call_abi_overflow_count(layout.results.value);
  layout.local_byte_length = value_overflow_count * sizeof(uint64_t);
  if (layout.local_byte_length > UINT16_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "VM call value overflow packet exceeds local byte storage");
  }

  layout.local_ref_count =
      (uint32_t)loom_vm_call_abi_overflow_count(layout.arguments.ref) +
      loom_vm_call_abi_overflow_count(layout.results.ref);
  if (layout.local_ref_count > (uint32_t)UINT16_MAX + 1u) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "VM call ref overflow packet exceeds local ref storage");
  }

  layout.local_function_count =
      (uint32_t)loom_vm_call_abi_overflow_count(layout.arguments.function) +
      loom_vm_call_abi_overflow_count(layout.results.function);
  if (layout.local_function_count > (uint32_t)UINT16_MAX + 1u) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "VM call function overflow packet exceeds local function storage");
  }
  *out_layout = layout;
  return iree_ok_status();
}

static iree_status_t loom_vm_call_abi_layout_make_name_table(
    loom_module_t* module, loom_vm_call_abi_source_fields_t fields,
    iree_arena_allocator_t* scratch_arena, loom_attribute_t* out_attr) {
  *out_attr = loom_attr_absent();
  if (fields.values != NULL && fields.presentation_names != NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "VM ABI fields have multiple presentation name sources");
  }
  if ((fields.values == NULL && fields.presentation_names == NULL) ||
      fields.count == 0) {
    return iree_ok_status();
  }

  bool has_names = false;
  for (iree_host_size_t i = 0; i < fields.count; ++i) {
    iree_string_view_t name = iree_string_view_empty();
    if (fields.presentation_names != NULL) {
      name = fields.presentation_names[i];
    } else {
      if (fields.values[i] >= module->values.count) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "VM ABI name source value is out of range");
      }
      const loom_string_id_t name_id =
          loom_module_value(module, fields.values[i])->name_id;
      if (name_id != LOOM_STRING_ID_INVALID &&
          name_id >= module->strings.count) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "VM ABI source value name is out of range");
      }
      name = loom_module_value_name(module, fields.values[i]);
    }
    has_names |= !iree_string_view_is_empty(name);
  }
  if (!has_names) return iree_ok_status();

  loom_named_attr_t* entries = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, fields.count, sizeof(*entries), (void**)&entries));
  loom_string_id_t empty_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, iree_string_view_empty(), &empty_name_id));
  for (iree_host_size_t i = 0; i < fields.count; ++i) {
    char field_key_storage[LOOM_VM_CALL_ABI_FIELD_KEY_CAPACITY];
    const iree_string_view_t field_key =
        loom_vm_call_abi_format_field_key((uint16_t)i, field_key_storage);
    loom_string_id_t field_key_id = LOOM_STRING_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_module_intern_string(module, field_key, &field_key_id));
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    if (fields.presentation_names != NULL) {
      IREE_RETURN_IF_ERROR(loom_module_intern_string(
          module, fields.presentation_names[i], &name_id));
    } else {
      name_id = loom_module_value(module, fields.values[i])->name_id;
      if (name_id == LOOM_STRING_ID_INVALID) name_id = empty_name_id;
    }
    entries[i] = (loom_named_attr_t){
        .name_id = field_key_id,
        .value = loom_attr_string(name_id),
    };
  }
  *out_attr = loom_make_canonical_attr_dict(entries, fields.count);
  return iree_ok_status();
}

iree_status_t loom_vm_call_abi_layout_preserve_presentation_names(
    loom_module_t* module, loom_named_attr_slice_t abi_layout,
    loom_vm_call_abi_source_fields_t arguments,
    loom_vm_call_abi_source_fields_t results,
    iree_arena_allocator_t* scratch_arena, bool* out_changed,
    loom_attribute_t* out_attr) {
  *out_changed = false;
  *out_attr =
      loom_make_canonical_attr_dict(abi_layout.entries, abi_layout.count);
  if (arguments.count > UINT16_MAX || results.count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VM logical call signature exceeds u16");
  }

  loom_named_attr_slice_t existing_argument_names =
      loom_named_attr_slice_empty();
  loom_named_attr_slice_t existing_result_names = loom_named_attr_slice_empty();
  IREE_RETURN_IF_ERROR(loom_vm_call_abi_layout_resolve_presentation_names(
      module, abi_layout, (uint16_t)arguments.count, (uint16_t)results.count,
      &existing_argument_names, &existing_result_names));

  loom_attribute_t argument_names = loom_attr_absent();
  if (existing_argument_names.count == 0) {
    IREE_RETURN_IF_ERROR(loom_vm_call_abi_layout_make_name_table(
        module, arguments, scratch_arena, &argument_names));
  }
  loom_attribute_t result_names = loom_attr_absent();
  if (existing_result_names.count == 0) {
    IREE_RETURN_IF_ERROR(loom_vm_call_abi_layout_make_name_table(
        module, results, scratch_arena, &result_names));
  }

  loom_named_attr_update_t updates[2];
  iree_host_size_t update_count = 0;
  if (!loom_attr_is_absent(argument_names)) {
    loom_string_id_t key_id = LOOM_STRING_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_module_intern_string(module, IREE_SV("argument_names"), &key_id));
    updates[update_count++] = (loom_named_attr_update_t){
        .name_id = key_id,
        .value = argument_names,
    };
  }
  if (!loom_attr_is_absent(result_names)) {
    loom_string_id_t key_id = LOOM_STRING_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_module_intern_string(module, IREE_SV("result_names"), &key_id));
    updates[update_count++] = (loom_named_attr_update_t){
        .name_id = key_id,
        .value = result_names,
    };
  }
  if (update_count == 0) return iree_ok_status();

  IREE_RETURN_IF_ERROR(
      loom_module_replace_canonical_attr_dict(module, abi_layout,
                                              (loom_named_attr_update_slice_t){
                                                  .updates = updates,
                                                  .count = update_count,
                                              },
                                              out_attr));
  *out_changed = true;
  return iree_ok_status();
}

iree_status_t loom_vm_call_abi_layout_make_attr(
    loom_module_t* module, loom_vm_call_abi_source_fields_t arguments,
    loom_vm_call_abi_source_fields_t results, loom_type_t authored_signature,
    iree_arena_allocator_t* scratch_arena, loom_attribute_t* out_attr) {
  *out_attr = loom_attr_absent();
  if (arguments.count > UINT16_MAX || results.count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VM logical call signature exceeds u16");
  }
  if ((arguments.count != 0 && arguments.types == NULL) ||
      (results.count != 0 && results.types == NULL)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM logical call signature types are missing");
  }

  loom_type_t signature = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_module_intern_function_type(
      module, arguments.types, (uint16_t)arguments.count, results.types,
      (uint16_t)results.count, &signature));
  if (loom_type_kind(authored_signature) != LOOM_TYPE_NONE) {
    IREE_RETURN_IF_ERROR(loom_vm_call_abi_layout_validate_authored_signature(
        signature, authored_signature));
  }
  loom_type_id_t signature_id = LOOM_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_type_id(module, signature, &signature_id));

  loom_attribute_t argument_names = loom_attr_absent();
  IREE_RETURN_IF_ERROR(loom_vm_call_abi_layout_make_name_table(
      module, arguments, scratch_arena, &argument_names));
  loom_attribute_t result_names = loom_attr_absent();
  IREE_RETURN_IF_ERROR(loom_vm_call_abi_layout_make_name_table(
      module, results, scratch_arena, &result_names));

  loom_named_attr_t entries[4] = {0};
  iree_host_size_t entry_count = 0;
  if (!loom_attr_is_absent(argument_names)) {
    IREE_RETURN_IF_ERROR(loom_module_intern_string(
        module, IREE_SV("argument_names"), &entries[entry_count].name_id));
    entries[entry_count++].value = argument_names;
  }
  if (!loom_attr_is_absent(result_names)) {
    IREE_RETURN_IF_ERROR(loom_module_intern_string(
        module, IREE_SV("result_names"), &entries[entry_count].name_id));
    entries[entry_count++].value = result_names;
  }
  loom_string_id_t signature_key = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(module, IREE_SV("signature"), &signature_key));
  entries[entry_count++] = (loom_named_attr_t){
      .name_id = signature_key,
      .value = loom_attr_type(signature_id),
  };
  if (loom_type_kind(authored_signature) != LOOM_TYPE_NONE &&
      !loom_type_equal(authored_signature, signature)) {
    loom_type_id_t authored_signature_id = LOOM_TYPE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_module_intern_type_id(module, authored_signature,
                                                    &authored_signature_id));
    IREE_RETURN_IF_ERROR(loom_module_intern_string(
        module, IREE_SV("authored_signature"), &entries[entry_count].name_id));
    entries[entry_count++].value = loom_attr_type(authored_signature_id);
  }
  return loom_module_make_canonical_attr_dict(
      module, loom_make_named_attr_slice(entries, entry_count), out_attr);
}
