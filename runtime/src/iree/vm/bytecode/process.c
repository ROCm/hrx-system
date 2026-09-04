// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/process.h"

static iree_status_t iree_vm_bytecode_process_check_immutable_bits(
    const uint64_t* bits, uint32_t count, const char* global_kind) {
  for (uint32_t i = 0; i < count; ++i) {
    if (!iree_vm_bytecode_process_bit_test(bits, i)) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "immutable %s global %" PRIu32 " is unset",
                              global_kind, i);
    }
  }
  return iree_ok_status();
}

iree_status_t iree_vm_bytecode_process_seal_state(iree_vm_module_t* base_module,
                                                  iree_byte_span_t storage) {
  iree_vm_bytecode_image_t* image =
      iree_vm_bytecode_image_from_module(base_module);
  const iree_vm_bytecode_v0_globals_header_t* globals =
      image->layout.globals.header;
  if (!globals) return iree_ok_status();

  iree_vm_bytecode_process_header_t* header =
      iree_vm_bytecode_process_header(storage.data);
  if (header->construction_state !=
      IREE_VM_BYTECODE_PROCESS_CONSTRUCTION_STATE_OPEN) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "bytecode process state is already sealed");
  }
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_process_check_immutable_bits(
      iree_vm_bytecode_process_value_set_bits(image, storage.data),
      globals->immutable_value_count_u32, "value"));

  const iree_vm_ref_t* refs =
      iree_vm_bytecode_process_refs(image, storage.data);
  const uint64_t* ref_set_bits =
      iree_vm_bytecode_process_ref_set_bits(image, storage.data);
  for (uint32_t i = 0; i < globals->ref_count_u32; ++i) {
    const bool is_nullable =
        iree_any_bit_set(image->layout.globals.refs[i].flags_u16,
                         IREE_VM_BYTECODE_GLOBAL_REF_FLAG_NULLABLE);
    if (i < globals->immutable_ref_count_u32 &&
        !iree_vm_bytecode_process_bit_test(ref_set_bits, i) &&
        (!is_nullable || !iree_vm_ref_is_null(refs[i]))) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "immutable ref global %" PRIu32 " is unset", i);
    }
    if (!is_nullable && iree_vm_ref_is_null(refs[i])) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "required ref global %" PRIu32 " is null", i);
    }
  }

  const iree_vm_function_ref_t* functions =
      iree_vm_bytecode_process_functions(image, storage.data);
  const uint64_t* function_set_bits =
      iree_vm_bytecode_process_function_set_bits(image, storage.data);
  for (uint32_t i = 0; i < globals->function_count_u32; ++i) {
    const bool is_nullable =
        iree_any_bit_set(image->layout.globals.functions[i].flags_u16,
                         IREE_VM_BYTECODE_GLOBAL_FUNCTION_FLAG_NULLABLE);
    if (i < globals->immutable_function_count_u32 &&
        !iree_vm_bytecode_process_bit_test(function_set_bits, i) &&
        (!is_nullable || !iree_vm_function_ref_is_null(functions[i]))) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "immutable function global %" PRIu32 " is unset",
                              i);
    }
    if (!is_nullable && iree_vm_function_ref_is_null(functions[i])) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "required function global %" PRIu32 " is null",
                              i);
    }
  }

  header->construction_state =
      IREE_VM_BYTECODE_PROCESS_CONSTRUCTION_STATE_SEALED;
  return iree_ok_status();
}

void iree_vm_bytecode_process_detach_state(iree_vm_module_t* base_module,
                                           iree_byte_span_t storage) {
  iree_vm_bytecode_image_t* image =
      iree_vm_bytecode_image_from_module(base_module);
  const iree_vm_bytecode_v0_globals_header_t* globals =
      image->layout.globals.header;
  if (!globals) return;
  iree_vm_ref_t* refs = iree_vm_bytecode_process_refs(image, storage.data);
  for (uint32_t i = 0; i < globals->ref_count_u32; ++i) {
    iree_vm_ref_reset(&refs[i]);
  }
}
