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

bool loom_vm_call_abi_try_classify_logical_type(
    const loom_module_t* module, loom_type_t type,
    loom_vm_call_abi_bank_t* out_bank) {
  *out_bank = LOOM_VM_CALL_ABI_BANK_NONE;
  if (loom_type_is_scalar(type)) {
    *out_bank = LOOM_VM_CALL_ABI_BANK_VALUE;
    return true;
  }
  if (loom_type_is_buffer(type)) {
    *out_bank = LOOM_VM_CALL_ABI_BANK_REF;
    return true;
  }
  if (loom_func_ref_type_isa(type)) {
    if (!loom_type_is_function(loom_func_ref_resolve_signature(module, type))) {
      return false;
    }
    *out_bank = LOOM_VM_CALL_ABI_BANK_FUNCTION;
    return true;
  }
  const loom_type_descriptor_t* descriptor =
      loom_type_registry_resolve(module, type);
  if (descriptor == NULL ||
      descriptor->semantics.semantic != LOOM_TYPE_SEMANTIC_MANAGED_REFERENCE) {
    return false;
  }
  *out_bank = LOOM_VM_CALL_ABI_BANK_REF;
  return true;
}

iree_status_t loom_vm_call_abi_classify_type(
    const loom_module_t* module, loom_type_t type,
    loom_vm_call_abi_bank_t* out_bank) {
  *out_bank = LOOM_VM_CALL_ABI_BANK_NONE;
  if (!loom_low_type_is_register(type) ||
      loom_low_register_type_descriptor_set_stable_id(type) !=
          VM_CORE_DESCRIPTOR_SET_ID ||
      loom_low_register_type_unit_count(type) != 1) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "VM callable fields must be exactly one vm.core register unit");
  }

  const loom_type_t* value_type = loom_type_register_value_type(type);
  if (value_type == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "VM callable fields must retain their logical type");
  }
  loom_vm_call_abi_bank_t logical_bank = LOOM_VM_CALL_ABI_BANK_NONE;
  if (!loom_vm_call_abi_try_classify_logical_type(module, *value_type,
                                                  &logical_bank)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "VM callable register carries an unsupported logical type");
  }
  switch (loom_low_register_type_class_id(type)) {
    case VM_CORE_REG_CLASS_ID_VALUE:
      *out_bank = LOOM_VM_CALL_ABI_BANK_VALUE;
      break;
    case VM_CORE_REG_CLASS_ID_REF:
      *out_bank = LOOM_VM_CALL_ABI_BANK_REF;
      break;
    case VM_CORE_REG_CLASS_ID_FUNCTION:
      *out_bank = LOOM_VM_CALL_ABI_BANK_FUNCTION;
      break;
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "VM callable field uses unknown register class %" PRIu16,
          loom_low_register_type_class_id(type));
  }
  if (*out_bank != logical_bank) {
    *out_bank = LOOM_VM_CALL_ABI_BANK_NONE;
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "VM callable register bank does not match its logical type");
  }
  return iree_ok_status();
}

iree_status_t loom_vm_call_abi_layout_resolve_signature(
    const loom_module_t* module, loom_named_attr_slice_t abi_layout,
    loom_type_t* out_signature) {
  *out_signature = loom_type_none();
  const loom_string_id_t signature_key =
      loom_module_lookup_string(module, IREE_SV("signature"));
  const loom_attribute_t* signature_attr = NULL;
  for (iree_host_size_t i = 0; i < abi_layout.count; ++i) {
    const loom_named_attr_t* entry = &abi_layout.entries[i];
    if (entry->name_id != signature_key) continue;
    if (signature_attr != NULL) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "VM abi_layout has duplicate signature entries");
    }
    signature_attr = &entry->value;
  }
  if (signature_attr == NULL || signature_attr->kind != LOOM_ATTR_TYPE ||
      signature_attr->type_id >= module->types.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "VM abi_layout requires a valid signature type entry");
  }
  const loom_type_t signature = module->types.entries[signature_attr->type_id];
  if (!loom_type_is_function(signature)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM abi_layout signature must be a function type");
  }
  *out_signature = signature;
  return iree_ok_status();
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
    loom_vm_call_abi_bank_t bank = LOOM_VM_CALL_ABI_BANK_NONE;
    IREE_RETURN_IF_ERROR(loom_vm_call_abi_classify_type(
        module, loom_module_value_type(module, values[i]), &bank));
    ++*loom_vm_call_abi_bank_count(out_counts, bank);
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
    loom_vm_call_abi_bank_t bank = LOOM_VM_CALL_ABI_BANK_NONE;
    IREE_RETURN_IF_ERROR(
        loom_vm_call_abi_classify_type(module, types[i], &bank));
    uint16_t* bank_count =
        loom_vm_call_abi_bank_count(&out_layout->bank_counts, bank);
    fields[i] = (loom_vm_call_abi_field_layout_t){
        .bank = bank,
        .bank_ordinal = *bank_count,
    };
    ++*bank_count;
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

iree_status_t loom_vm_call_abi_layout_make_attr(
    loom_module_t* module, const loom_type_t* argument_types,
    iree_host_size_t argument_count, const loom_type_t* result_types,
    iree_host_size_t result_count, loom_attribute_t* out_attr) {
  *out_attr = loom_attr_absent();
  if (argument_count > UINT16_MAX || result_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VM logical call signature exceeds u16");
  }

  loom_type_t signature = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_module_intern_function_type(
      module, argument_types, (uint16_t)argument_count, result_types,
      (uint16_t)result_count, &signature));
  loom_type_id_t signature_id = LOOM_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_type_id(module, signature, &signature_id));
  loom_string_id_t signature_key = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(module, IREE_SV("signature"), &signature_key));
  const loom_named_attr_t entries[] = {
      {
          .name_id = signature_key,
          .value = loom_attr_type(signature_id),
      },
  };
  return loom_module_make_canonical_attr_dict(
      module, loom_make_named_attr_slice(entries, IREE_ARRAYSIZE(entries)),
      out_attr);
}
