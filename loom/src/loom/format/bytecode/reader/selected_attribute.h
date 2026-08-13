// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Reached-only attribute materialization for selected bytecode bodies.

#ifndef LOOM_FORMAT_BYTECODE_READER_SELECTED_ATTRIBUTE_H_
#define LOOM_FORMAT_BYTECODE_READER_SELECTED_ATTRIBUTE_H_

#include "loom/format/bytecode/reader/attribute.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_bytecode_selected_table_materializer_t
    loom_bytecode_selected_table_materializer_t;

// Result of one selected attribute materialization attempt.
typedef enum loom_bytecode_selected_attribute_state_e {
  // Every referenced table identity was available and |out_attr| is canonical.
  LOOM_BYTECODE_SELECTED_ATTRIBUTE_READY = 0,
  // Missing table identities were scheduled and |out_attr| is not usable.
  LOOM_BYTECODE_SELECTED_ATTRIBUTE_WAITING = 1,
} loom_bytecode_selected_attribute_state_t;

// Decodes an attribute whose predicate VALUE arguments are STRINGS ordinals.
// Returned aggregate payloads are scratch-owned and contain target-domain IDs.
// The caller canonicalizes the value only after the containing entry is ready.
iree_status_t loom_bytecode_selected_attribute_decode_named(
    loom_bytecode_selected_table_materializer_t* materializer,
    loom_bytecode_reader_cursor_t* cursor,
    const loom_attr_descriptor_t* descriptor, loom_bytecode_attr_kind_t kind,
    loom_attribute_t* out_attr, iree_host_size_t available_type_count,
    loom_bytecode_selected_attribute_state_t* out_state);

// Decodes an attribute whose predicate VALUE arguments are SSA numbers.
iree_status_t loom_bytecode_selected_attribute_decode_ssa(
    loom_bytecode_selected_table_materializer_t* materializer,
    loom_bytecode_reader_cursor_t* cursor,
    const loom_attr_descriptor_t* descriptor, loom_bytecode_attr_kind_t kind,
    loom_attribute_t* out_attr, iree_host_size_t available_type_count,
    const loom_bytecode_attribute_ssa_materialization_scope_t* ssa_scope,
    loom_bytecode_selected_attribute_state_t* out_state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_BYTECODE_READER_SELECTED_ATTRIBUTE_H_
