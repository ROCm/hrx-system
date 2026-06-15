// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Builders for descriptor-backed target-low packets.
//
// Text parsing resolves descriptor names from source text. Compiled lowering
// paths should keep descriptor selection in generated IDs or resolved
// descriptor rows and never hash descriptor spellings in the hot path.

#ifndef LOOM_CODEGEN_LOW_BUILDER_H_
#define LOOM_CODEGEN_LOW_BUILDER_H_

#include "iree/base/api.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/ops/low/ops.h"

#ifdef __cplusplus
extern "C" {
#endif

// Creates a reg<...> type selected by descriptor-set register-class ID.
iree_status_t loom_low_build_register_type(
    const loom_low_descriptor_set_t* descriptor_set, uint16_t reg_class_id,
    uint32_t unit_count, loom_type_t* out_type);

// Emits a descriptor-backed low.op from a descriptor row resolved earlier in
// the pipeline.
//
// |opcode_id| must be the module string ID for |descriptor|'s key. Supplying it
// lets source-to-low selection resolve the descriptor spelling once and lets
// emission construct the packet without another descriptor lookup.
iree_status_t loom_low_build_resolved_descriptor_op(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, loom_string_id_t opcode_id,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    loom_named_attr_slice_t attrs, const loom_type_t* result_types,
    iree_host_size_t result_count, const loom_tied_result_t* tied_results,
    iree_host_size_t tied_result_count, loom_location_id_t location,
    loom_op_t** out_op);

// Emits a descriptor-backed low.const from a descriptor row resolved earlier in
// the pipeline.
iree_status_t loom_low_build_resolved_descriptor_const(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, loom_string_id_t opcode_id,
    loom_named_attr_slice_t attrs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_BUILDER_H_
