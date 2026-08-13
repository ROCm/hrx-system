// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared SYMBOLS schema checks used by validation and materialization.

#ifndef LOOM_FORMAT_BYTECODE_READER_SYMBOL_SCHEMA_H_
#define LOOM_FORMAT_BYTECODE_READER_SYMBOL_SCHEMA_H_

#include "iree/base/api.h"
#include "loom/format/bytecode/reader/decoder.h"
#include "loom/format/bytecode/reader/module_view.h"
#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Resolves a string-table reference used by a symbol payload.
iree_status_t loom_bytecode_symbol_validate_string_ref(
    loom_bytecode_reader_decoder_t* decoder,
    const loom_bytecode_reader_module_view_t* module_view, uint64_t string_id,
    iree_string_view_t field_name, uint64_t offset,
    iree_string_view_t* out_string);

// Validates a type-table reference used by a symbol payload.
iree_status_t loom_bytecode_symbol_validate_type_ref(
    loom_bytecode_reader_decoder_t* decoder,
    const loom_bytecode_reader_module_view_t* module_view, uint64_t type_id,
    uint64_t offset);

// Resolves a one-based operation-table reference used by a symbol payload.
iree_status_t loom_bytecode_symbol_resolve_op_ref(
    loom_bytecode_reader_decoder_t* decoder,
    const loom_bytecode_reader_module_view_t* module_view,
    uint64_t op_table_index_plus1, uint64_t offset,
    const loom_op_vtable_t** out_vtable);

// Validates one optional function metadata enumeration value.
iree_status_t loom_bytecode_symbol_validate_func_enum(
    loom_bytecode_reader_decoder_t* decoder, uint64_t symbol_index,
    const loom_op_vtable_t* vtable, uint8_t attr_index,
    iree_string_view_t field_name, uint8_t value, uint64_t offset);

// Finds an operation attribute by its textual name.
uint8_t loom_bytecode_symbol_find_op_attr_index_by_name(
    const loom_op_vtable_t* vtable, iree_string_view_t name);

// Returns true when an attribute carries the defining symbol identity.
bool loom_bytecode_symbol_attr_is_identity(const loom_op_vtable_t* vtable,
                                           uint8_t attr_index);

// Returns true when function metadata reconstructs an operation attribute.
bool loom_bytecode_symbol_func_metadata_attr_is_shared(
    const loom_op_vtable_t* vtable, const loom_func_like_vtable_t* func_like,
    uint8_t attr_index);

// Finds the attribute slot carrying a defining symbol identity.
uint8_t loom_bytecode_symbol_find_identity_attr_index(
    const loom_op_vtable_t* vtable);

// Validates a GLOBAL payload operation contract.
iree_status_t loom_bytecode_symbol_validate_global_vtable(
    loom_bytecode_reader_decoder_t* decoder, uint64_t symbol_index,
    const loom_op_vtable_t* vtable, uint64_t op_ref_offset);

// Validates a RECORD payload operation contract.
iree_status_t loom_bytecode_symbol_validate_record_vtable(
    loom_bytecode_reader_decoder_t* decoder, uint64_t symbol_index,
    const loom_op_vtable_t* vtable, uint64_t op_ref_offset);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LOOM_FORMAT_BYTECODE_READER_SYMBOL_SCHEMA_H_
