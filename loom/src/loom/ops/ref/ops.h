// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.
// Regenerate: python3 loom/py/loom/gen/run.py c_tables --in-place
// clang-format off

#ifndef LOOM_OPS_REF_OPS_H_
#define LOOM_OPS_REF_OPS_H_

#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  LOOM_OP_REF_DUPLICATE = LOOM_OP_KIND(LOOM_DIALECT_REF, 0),
  LOOM_OP_REF_BORROW = LOOM_OP_KIND(LOOM_DIALECT_REF, 1),
  LOOM_OP_REF_RELEASE = LOOM_OP_KIND(LOOM_DIALECT_REF, 2),
  LOOM_OP_REF_NULL = LOOM_OP_KIND(LOOM_DIALECT_REF, 3),
  LOOM_OP_REF_COMPARE_NULL = LOOM_OP_KIND(LOOM_DIALECT_REF, 4),
  LOOM_OP_REF_COMPARE_EQ = LOOM_OP_KIND(LOOM_DIALECT_REF, 5),
  LOOM_OP_REF_COUNT_ = 6,
};

// LOOM_OP_REF_DUPLICATE: Create an independent handle to the same managed payload, preserving its type and null state. Duplicating ref adds an owner that must be released or transferred independently. Duplicating ptr preserves its borrowed mode and any protection needed through its final use. The source remains valid; live source and result lifetimes are independent.
// %copy = ref.duplicate %owner : ref<@Object>
LOOM_DEFINE_ISA(loom_ref_duplicate_isa, LOOM_OP_REF_DUPLICATE)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_ref_duplicate_sources, 0)
LOOM_DEFINE_VARIADIC_RESULTS(loom_ref_duplicate_duplicates, 0)
iree_status_t loom_ref_duplicate_build(
    loom_builder_t* builder,
    const loom_value_id_t* sources,
    iree_host_size_t sources_count,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_value_duplication_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_REF_BORROW: Create a pointer to an owner's exact managed payload without consuming the owner. Null produces null. The pointer remains live through its final reachable use, even when the original owner is released first. The compiler acquires protection when it cannot prove a covering owner and retires that protection after the final use. The source owner's explicit lifetime is unchanged.
// %pointer = ref.borrow %owner : ref<@Frame> -> ptr<@Frame>
LOOM_DEFINE_ISA(loom_ref_borrow_isa, LOOM_OP_REF_BORROW)
LOOM_DEFINE_OPERAND(loom_ref_borrow_owner, 0)
LOOM_DEFINE_RESULT(loom_ref_borrow_pointer, 0)
iree_status_t loom_ref_borrow_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t owner,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_ref_borrow_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_REF_RELEASE: End this managed handle's lifetime. The source identity is invalid afterward. An owning ref releases its owner and runs the native finalizer when it holds the last reference. A ptr retires any compiler-acquired protection; an unprotected borrow or null has no payload effect. Independent handles remain valid.
// ref.release %owner : ref<@Frame>
LOOM_DEFINE_ISA(loom_ref_release_isa, LOOM_OP_REF_RELEASE)
LOOM_DEFINE_OPERAND(loom_ref_release_value, 0)
iree_status_t loom_ref_release_build(
    loom_builder_t* builder,
    loom_value_id_t value,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_REF_NULL: Create an independent null handle of the declared ref or ptr type. No payload is allocated or retained. An owning ref still has its own explicit lifetime and must be released or transferred on every normal path; releasing null has no payload effect.
// %empty = ref.null : ref<@Frame>
LOOM_DEFINE_ISA(loom_ref_null_isa, LOOM_OP_REF_NULL)
LOOM_DEFINE_RESULT(loom_ref_null_result, 0)
iree_status_t loom_ref_null_build(
    loom_builder_t* builder,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_REF_COMPARE_NULL: Return true when the handle is null and false otherwise. The handle is borrowed without retaining, releasing, or inspecting its payload.
// %missing = ref.compare.null %frame : ptr<@Frame>
LOOM_DEFINE_ISA(loom_ref_compare_null_isa, LOOM_OP_REF_COMPARE_NULL)
LOOM_DEFINE_OPERAND(loom_ref_compare_null_value, 0)
LOOM_DEFINE_RESULT(loom_ref_compare_null_result, 0)
iree_status_t loom_ref_compare_null_build(
    loom_builder_t* builder,
    loom_value_id_t value,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_REF_COMPARE_EQ: Return true for two null handles or two handles to the same managed payload, and false otherwise. Both operands must name the same external payload type, but may have different ownership modes. Neither operand is retained, released, or dereferenced. Equality compares object identity, not payload contents.
// %same = ref.compare.eq %owner, %frame : ref<@Frame>, ptr<@Frame>
LOOM_DEFINE_ISA(loom_ref_compare_eq_isa, LOOM_OP_REF_COMPARE_EQ)
LOOM_DEFINE_OPERAND(loom_ref_compare_eq_left, 0)
LOOM_DEFINE_OPERAND(loom_ref_compare_eq_right, 1)
LOOM_DEFINE_RESULT(loom_ref_compare_eq_result, 0)
iree_status_t loom_ref_compare_eq_build(
    loom_builder_t* builder,
    loom_value_id_t left,
    loom_value_id_t right,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_ref_compare_eq_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// Returns the vtable array for the ref dialect.
const loom_op_vtable_t* const* loom_ref_dialect_vtables(
    iree_host_size_t* out_count);

// Returns the dense semantic metadata array for the ref dialect.
const loom_op_semantics_t* loom_ref_dialect_op_semantics(
    iree_host_size_t* out_count);

// Returns semantic metadata for a ref op kind, or empty metadata.
loom_op_semantics_t loom_ref_op_semantics(
    loom_op_kind_t kind);

#ifdef __cplusplus
}
#endif

// Additional named attribute helpers are generated with this dialect.
#include "loom/ops/ref/ops.inc"

#endif  // LOOM_OPS_REF_OPS_H_
