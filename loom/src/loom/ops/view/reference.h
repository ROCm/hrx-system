// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// View reference fact construction helpers.
//
// These helpers summarize view-producing ops into root identity, byte-base,
// byte-footprint, and alignment facts. They resolve layouts through encoding
// facts so direct `encoding.layout.*` values and composed `#physical_storage`
// values share one interpretation path.

#ifndef LOOM_OPS_VIEW_REFERENCE_H_
#define LOOM_OPS_VIEW_REFERENCE_H_

#include "loom/ir/attribute.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

// Constructs view-reference facts for `buffer.view`.
iree_status_t loom_view_reference_make_buffer_view(
    loom_fact_context_t* context, const loom_module_t* module,
    loom_value_id_t buffer_value_id, loom_value_facts_t buffer_facts,
    loom_value_facts_t byte_offset_facts, loom_type_t result_type,
    loom_value_facts_t* out);

// Constructs view-reference facts for `view.subview`.
iree_status_t loom_view_reference_make_subview(
    loom_fact_context_t* context, const loom_module_t* module,
    loom_value_id_t source_value_id, loom_value_facts_t source_facts,
    loom_attribute_t static_offsets, loom_value_slice_t dynamic_offsets,
    loom_type_t source_type, loom_type_t result_type, loom_value_facts_t* out);

// Constructs view-reference facts for `view.refine`.
iree_status_t loom_view_reference_make_refine(
    loom_fact_context_t* context, const loom_module_t* module,
    loom_value_id_t source_value_id, loom_value_facts_t source_facts,
    loom_type_t source_type, loom_type_t result_type, loom_value_facts_t* out);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_VIEW_REFERENCE_H_
