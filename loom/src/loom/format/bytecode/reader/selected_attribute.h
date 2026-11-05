// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Reached-only attribute materialization for selected bytecode bodies.

#ifndef LOOM_FORMAT_BYTECODE_READER_SELECTED_ATTRIBUTE_H_
#define LOOM_FORMAT_BYTECODE_READER_SELECTED_ATTRIBUTE_H_

#include "loom/format/bytecode/reader/attribute.h"
#include "loom/format/bytecode/reader/type_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_bytecode_selected_table_materializer_t
    loom_bytecode_selected_table_materializer_t;

// Decodes a validated static parameterized type into scratch-owned slots.
// Missing table identities are scheduled directly into the returned slots;
// the caller retains scratch until those dependencies have completed.
iree_status_t loom_bytecode_selected_attribute_decode_static_type(
    loom_bytecode_selected_table_materializer_t* materializer,
    loom_bytecode_reader_cursor_t* cursor, loom_type_id_t source_type_id,
    const loom_parameterized_type_descriptor_t** out_descriptor,
    loom_attribute_t** out_parameters);

// Constructs a complete parameterized type from scratch-owned parameter slots.
iree_status_t loom_bytecode_selected_attribute_decode_complete_type(
    loom_bytecode_selected_table_materializer_t* materializer,
    loom_bytecode_reader_cursor_t* cursor,
    const loom_bytecode_attribute_ssa_materialization_scope_t* values,
    loom_type_id_t* out_type);

// Decodes an attribute whose predicate VALUE arguments are STRINGS ordinals.
// Returned aggregate payloads are scratch-owned. Missing table identities are
// scheduled into stable slots within the payload; the caller retains scratch
// until those dependencies complete and then canonicalizes the value.
iree_status_t loom_bytecode_selected_attribute_decode_named(
    loom_bytecode_selected_table_materializer_t* materializer,
    loom_bytecode_reader_cursor_t* cursor,
    const loom_attr_descriptor_t* descriptor, loom_bytecode_attr_kind_t kind,
    iree_host_size_t available_type_count, loom_attribute_t* out_attr);

// Materializes an attribute whose predicate VALUE arguments are STRINGS
// ordinals. Reached table dependencies are materialized synchronously and the
// returned value owns no scratch-arena storage.
iree_status_t loom_bytecode_selected_attribute_materialize_named(
    loom_bytecode_selected_table_materializer_t* materializer,
    loom_bytecode_reader_cursor_t* cursor,
    const loom_attr_descriptor_t* descriptor, loom_bytecode_attr_kind_t kind,
    iree_host_size_t available_type_count, loom_attribute_t* out_attr);

// Materializes an attribute whose predicate VALUE arguments are SSA numbers.
// Reached table dependencies are materialized synchronously and the returned
// value owns no scratch-arena storage.
iree_status_t loom_bytecode_selected_attribute_materialize_ssa(
    loom_bytecode_selected_table_materializer_t* materializer,
    loom_bytecode_reader_cursor_t* cursor,
    const loom_attr_descriptor_t* descriptor, loom_bytecode_attr_kind_t kind,
    iree_host_size_t available_type_count,
    const loom_bytecode_attribute_ssa_materialization_scope_t* ssa_scope,
    loom_attribute_t* out_attr);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_BYTECODE_READER_SELECTED_ATTRIBUTE_H_
