// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.
// Regenerate: python3 loom/py/loom/gen/run.py c_tables --in-place
// clang-format off

#ifndef LOOM_OPS_BUFFER_OPS_H_
#define LOOM_OPS_BUFFER_OPS_H_

#include "loom/ops/op_defs.h"
#include "loom/ir/facts.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  LOOM_OP_BUFFER_ALLOCA = LOOM_OP_KIND(LOOM_DIALECT_BUFFER, 0),
  LOOM_OP_BUFFER_ASSUME_ALIGNMENT = LOOM_OP_KIND(LOOM_DIALECT_BUFFER, 1),
  LOOM_OP_BUFFER_ASSUME_MEMORY_SPACE = LOOM_OP_KIND(LOOM_DIALECT_BUFFER, 2),
  LOOM_OP_BUFFER_ASSUME_NOALIAS = LOOM_OP_KIND(LOOM_DIALECT_BUFFER, 3),
  LOOM_OP_BUFFER_ASSUME_SAME_ROOT = LOOM_OP_KIND(LOOM_DIALECT_BUFFER, 4),
  LOOM_OP_BUFFER_VIEW = LOOM_OP_KIND(LOOM_DIALECT_BUFFER, 5),
  LOOM_OP_BUFFER_COUNT_ = 6,
};

// LOOM_OP_BUFFER_ALLOCA: Create a fixed-frame scratch buffer root in workgroup or private memory. Each execution produces a distinct storage identity; identical allocas must not be commoned. The byte length is a physical byte count, and base_alignment is the minimum byte alignment of the root storage base.
// %scratch = buffer.alloca %bytes {base_alignment = 64, memory_space = workgroup} : buffer
LOOM_DEFINE_ISA(loom_buffer_alloca_isa, LOOM_OP_BUFFER_ALLOCA)
LOOM_DEFINE_OPERAND(loom_buffer_alloca_byte_length, 0)
LOOM_DEFINE_RESULT(loom_buffer_alloca_result, 0)
LOOM_DEFINE_ATTR_I64(loom_buffer_alloca_base_alignment, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_buffer_alloca_memory_space, 1, loom_value_fact_memory_space_t)
iree_status_t loom_buffer_alloca_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t byte_length,
    int64_t base_alignment,
    loom_value_fact_memory_space_t memory_space,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_buffer_alloca_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_buffer_alloca_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_BUFFER_ASSUME_ALIGNMENT: Refine existing buffer roots with an explicit minimum byte alignment contract. The result preserves the same storage identity, extent, memory-space, alias, and nullability facts while strengthening the root base alignment fact.
// %aligned = buffer.assume.alignment %buffer {minimum_alignment = 16} : buffer
LOOM_DEFINE_ISA(loom_buffer_assume_alignment_isa, LOOM_OP_BUFFER_ASSUME_ALIGNMENT)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_buffer_assume_alignment_buffers, 0)
LOOM_DEFINE_VARIADIC_RESULTS(loom_buffer_assume_alignment_results, 0)
LOOM_DEFINE_ATTR_I64(loom_buffer_assume_alignment_minimum_alignment, 0)
iree_status_t loom_buffer_assume_alignment_build(
    loom_builder_t* builder,
    const loom_value_id_t* buffers,
    iree_host_size_t buffers_count,
    int64_t minimum_alignment,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_buffer_assume_alignment_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_buffer_assume_alignment_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_BUFFER_ASSUME_MEMORY_SPACE: Refine an existing buffer root with a concrete target-independent memory-space fact while preserving the same storage identity, extent, alignment, and nullability facts.
// %global = buffer.assume.memory_space<global> %buffer : buffer
LOOM_DEFINE_ISA(loom_buffer_assume_memory_space_isa, LOOM_OP_BUFFER_ASSUME_MEMORY_SPACE)
LOOM_DEFINE_OPERAND(loom_buffer_assume_memory_space_buffer, 0)
LOOM_DEFINE_RESULT(loom_buffer_assume_memory_space_result, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_buffer_assume_memory_space_memory_space, 0, loom_value_fact_memory_space_t)
iree_status_t loom_buffer_assume_memory_space_build(
    loom_builder_t* builder,
    loom_value_fact_memory_space_t memory_space,
    loom_value_id_t buffer,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_buffer_assume_memory_space_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_buffer_assume_memory_space_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_BUFFER_ASSUME_NOALIAS: Refine an existing buffer root with an explicit noalias contract. The result preserves the same storage identity, extent, memory-space, alignment, and nullability facts, and marks the root identity as comparable for disjointness proofs. External buffer arguments do not gain this proof by default.
// %unique = buffer.assume.noalias %buffer : buffer
LOOM_DEFINE_ISA(loom_buffer_assume_noalias_isa, LOOM_OP_BUFFER_ASSUME_NOALIAS)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_buffer_assume_noalias_buffers, 0)
LOOM_DEFINE_VARIADIC_RESULTS(loom_buffer_assume_noalias_results, 0)
iree_status_t loom_buffer_assume_noalias_build(
    loom_builder_t* builder,
    const loom_value_id_t* buffers,
    iree_host_size_t buffers_count,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_buffer_assume_noalias_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_BUFFER_ASSUME_SAME_ROOT: Refine an existing buffer root to share another buffer's storage root. This is a dominance-scoped assertion for internally specialized dispatches that know two incoming handles refer to the same allocation. The result keeps the first operand's value while inheriting the second operand's root identity and comparable alias scope.
// %same = buffer.assume.same_root %buffer, %root : buffer
LOOM_DEFINE_ISA(loom_buffer_assume_same_root_isa, LOOM_OP_BUFFER_ASSUME_SAME_ROOT)
LOOM_DEFINE_OPERAND(loom_buffer_assume_same_root_buffer, 0)
LOOM_DEFINE_OPERAND(loom_buffer_assume_same_root_root, 1)
LOOM_DEFINE_RESULT(loom_buffer_assume_same_root_result, 0)
iree_status_t loom_buffer_assume_same_root_build(
    loom_builder_t* builder,
    loom_value_id_t buffer,
    loom_value_id_t root,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_buffer_assume_same_root_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_BUFFER_VIEW: Form a typed non-owning view from an opaque buffer root and base byte offset. The result view type carries the address layout.
// %view = buffer.view %buffer[%offset] : buffer -> view<[%M]xf32, %layout>
LOOM_DEFINE_ISA(loom_buffer_view_isa, LOOM_OP_BUFFER_VIEW)
LOOM_DEFINE_OPERAND(loom_buffer_view_buffer, 0)
LOOM_DEFINE_OPERAND(loom_buffer_view_byte_offset, 1)
LOOM_DEFINE_RESULT(loom_buffer_view_result, 0)
iree_status_t loom_buffer_view_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t buffer,
    loom_may_consume loom_value_id_t byte_offset,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_buffer_view_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_buffer_view_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// Returns the vtable array for the buffer dialect.
const loom_op_vtable_t* const* loom_buffer_dialect_vtables(
    iree_host_size_t* out_count);

// Returns the dense semantic metadata array for the buffer dialect.
const loom_op_semantics_t* loom_buffer_dialect_op_semantics(
    iree_host_size_t* out_count);

// Returns semantic metadata for a buffer op kind, or empty metadata.
loom_op_semantics_t loom_buffer_op_semantics(
    loom_op_kind_t kind);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_BUFFER_OPS_H_
