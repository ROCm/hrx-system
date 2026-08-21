// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_MODULE_READER_H_
#define IREE_VM_BYTECODE_MODULE_READER_H_

#include "iree/vm/bytecode/module_storage.h"

// Returns one already verified string ordinal as a stable image view.
static inline iree_string_view_t iree_vm_bytecode_string_at(
    const iree_vm_bytecode_string_table_t* table, uint16_t ordinal) {
  const uint32_t begin = table->offsets[ordinal];
  const uint32_t end = table->offsets[ordinal + 1];
  return iree_make_string_view((const char*)table->data + begin, end - begin);
}

// Returns a nullable string ordinal as an empty or stable image view.
static inline iree_string_view_t iree_vm_bytecode_nullable_string_at(
    const iree_vm_bytecode_string_table_t* table, uint16_t ordinal) {
  return ordinal == UINT16_MAX ? iree_string_view_empty()
                               : iree_vm_bytecode_string_at(table, ordinal);
}

// Returns the source-ordered descriptor block for a verified signature.
static inline const iree_vm_bytecode_v0_signature_descriptor_row_t*
iree_vm_bytecode_signature_descriptors(
    const iree_vm_bytecode_signature_table_t* table, uint16_t ordinal) {
  return table->descriptors + table->rows[ordinal].descriptor_base_u32;
}

// Returns the source-ordered argument count for a verified signature.
static inline uint32_t iree_vm_bytecode_signature_argument_count(
    const iree_vm_bytecode_v0_signature_row_t* row) {
  return (uint32_t)row->argument_value_count_u16 +
         (uint32_t)row->argument_ref_count_u16 +
         (uint32_t)row->argument_function_count_u16;
}

// Returns the source-ordered result count for a verified signature.
static inline uint32_t iree_vm_bytecode_signature_result_count(
    const iree_vm_bytecode_v0_signature_row_t* row) {
  return (uint32_t)row->result_value_count_u16 +
         (uint32_t)row->result_ref_count_u16 +
         (uint32_t)row->result_function_count_u16;
}

#endif  // IREE_VM_BYTECODE_MODULE_READER_H_
