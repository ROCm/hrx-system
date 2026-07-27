// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.
// Regenerate: python3 loom/py/loom/gen/run.py c_tables --in-place
// clang-format off

#ifndef LOOM_OPS_SANITIZER_OPS_H_
#define LOOM_OPS_SANITIZER_OPS_H_

#include "loom/ops/op_defs.h"
#include "loom/ir/facts.h"
#include "loom/ops/atomic.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  LOOM_OP_SANITIZER_ASSERT_ACCESS = LOOM_OP_KIND(LOOM_DIALECT_SANITIZER, 0),
  LOOM_OP_SANITIZER_ASSERT_VALUE = LOOM_OP_KIND(LOOM_DIALECT_SANITIZER, 1),
  LOOM_OP_SANITIZER_ASSERT_OP = LOOM_OP_KIND(LOOM_DIALECT_SANITIZER, 2),
  LOOM_OP_SANITIZER_ASSERT_LAYOUT = LOOM_OP_KIND(LOOM_DIALECT_SANITIZER, 3),
  LOOM_OP_SANITIZER_RACE_ACCESS = LOOM_OP_KIND(LOOM_DIALECT_SANITIZER, 4),
  LOOM_OP_SANITIZER_RACE_SYNC = LOOM_OP_KIND(LOOM_DIALECT_SANITIZER, 5),
  LOOM_OP_SANITIZER_ASSERT_ACCESSES = LOOM_OP_KIND(LOOM_DIALECT_SANITIZER, 6),
  LOOM_OP_SANITIZER_COUNT_ = 7,
};

// Logical memory access kind covered by a sanitizer access assertion.
typedef enum loom_sanitizer_assert_access_kind_e {
  LOOM_SANITIZER_ASSERT_ACCESS_KIND_READ = 0,
  LOOM_SANITIZER_ASSERT_ACCESS_KIND_WRITE = 1,
  LOOM_SANITIZER_ASSERT_ACCESS_KIND_READ_WRITE = 2,
  LOOM_SANITIZER_ASSERT_ACCESS_KIND_COUNT_ = 3,
} loom_sanitizer_assert_access_kind_t;

// Logical memory access kind covered by a sanitizer race observation.
typedef enum loom_sanitizer_race_access_kind_e {
  LOOM_SANITIZER_RACE_ACCESS_KIND_READ = 0,
  LOOM_SANITIZER_RACE_ACCESS_KIND_WRITE = 1,
  LOOM_SANITIZER_RACE_ACCESS_KIND_READ_WRITE = 2,
  LOOM_SANITIZER_RACE_ACCESS_KIND_COUNT_ = 3,
} loom_sanitizer_race_access_kind_t;

// Logical memory access kind covered by a repeated sanitizer access assertion.
typedef enum loom_sanitizer_assert_accesses_kind_e {
  LOOM_SANITIZER_ASSERT_ACCESSES_KIND_READ = 0,
  LOOM_SANITIZER_ASSERT_ACCESSES_KIND_WRITE = 1,
  LOOM_SANITIZER_ASSERT_ACCESSES_KIND_READ_WRITE = 2,
  LOOM_SANITIZER_ASSERT_ACCESSES_KIND_COUNT_ = 3,
} loom_sanitizer_assert_accesses_kind_t;

// LOOM_OP_SANITIZER_ASSERT_ACCESS: Assert that a logical indexed view access is valid. The assertion has the same index-list shape as ordinary view memory operations so source-level memory contracts remain typed until target lowering materializes address checks.
// sanitizer.assert.access<read> %view[%row, %col] : view<[%M]x[%N]xf32, %layout>
LOOM_DEFINE_ISA(loom_sanitizer_assert_access_isa, LOOM_OP_SANITIZER_ASSERT_ACCESS)
LOOM_DEFINE_OPERAND(loom_sanitizer_assert_access_view, 0)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_sanitizer_assert_access_indices, 1)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_sanitizer_assert_access_kind, 0, loom_sanitizer_assert_access_kind_t)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_sanitizer_assert_access_static_indices, 1)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_sanitizer_assert_access_static_extents, 2)
iree_status_t loom_sanitizer_assert_access_build(
    loom_builder_t* builder,
    loom_sanitizer_assert_access_kind_t kind,
    loom_value_id_t view,
    const loom_value_id_t* indices,
    iree_host_size_t indices_count,
    const int64_t* static_indices,
    iree_host_size_t static_indices_count,
    loom_optional const int64_t* static_extents,
    iree_host_size_t static_extents_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_sanitizer_assert_access_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_SANITIZER_ASSERT_VALUE: Assert predicate constraints over SSA values and return checked identity aliases. Unlike assume ops, this op is executable: failing the assertion reports the site and aborts execution. Passing the assertion refines facts for the returned aliases.
// %n_checked = sanitizer.assert.value %n [range(%n, 0, 4096), mul(%n, 16)] : index
LOOM_DEFINE_ISA(loom_sanitizer_assert_value_isa, LOOM_OP_SANITIZER_ASSERT_VALUE)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_sanitizer_assert_value_values, 0)
LOOM_DEFINE_VARIADIC_RESULTS(loom_sanitizer_assert_value_results, 0)
LOOM_DEFINE_ATTR_PREDICATE_LIST(loom_sanitizer_assert_value_predicates, 0)
iree_status_t loom_sanitizer_assert_value_build(
    loom_builder_t* builder,
    const loom_value_id_t* values,
    iree_host_size_t values_count,
    const loom_predicate_t* predicates,
    iree_host_size_t predicates_count,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_sanitizer_assert_value_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_sanitizer_assert_value_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_sanitizer_assert_value_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_SANITIZER_ASSERT_OP: Assert operation-level predicate constraints without producing checked aliases. This is the executable form for contracts such as valid divide operands, shift counts, overflow preconditions, and other facts where the checked operation itself remains the semantic anchor.
// sanitizer.assert.op %lhs, %rhs [ne(%rhs, 0)] : i32, i32
LOOM_DEFINE_ISA(loom_sanitizer_assert_op_isa, LOOM_OP_SANITIZER_ASSERT_OP)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_sanitizer_assert_op_values, 0)
LOOM_DEFINE_ATTR_PREDICATE_LIST(loom_sanitizer_assert_op_predicates, 0)
iree_status_t loom_sanitizer_assert_op_build(
    loom_builder_t* builder,
    const loom_value_id_t* values,
    iree_host_size_t values_count,
    const loom_predicate_t* predicates,
    iree_host_size_t predicates_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_sanitizer_assert_op_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_sanitizer_assert_op_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_SANITIZER_ASSERT_LAYOUT: Assert that a view satisfies a refined layout, shape, or encoding contract and return the same view on the continuing path. This is the executable counterpart to view.refine for diagnostics that must abort instead of trusting unchecked layout facts.
// %checked = sanitizer.assert.layout %view : view<[%M]x[%N]xf32, %layout> -> view<16x[%N]xf32, %layout>
LOOM_DEFINE_ISA(loom_sanitizer_assert_layout_isa, LOOM_OP_SANITIZER_ASSERT_LAYOUT)
LOOM_DEFINE_OPERAND(loom_sanitizer_assert_layout_view, 0)
LOOM_DEFINE_RESULT(loom_sanitizer_assert_layout_result, 0)
iree_status_t loom_sanitizer_assert_layout_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t view,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_sanitizer_assert_layout_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_sanitizer_assert_layout_type_transfer(
    loom_type_transfer_context_t* context,
    const loom_module_t* module, loom_op_t* op);
iree_status_t loom_sanitizer_assert_layout_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_SANITIZER_RACE_ACCESS: Observe a logical indexed view access for race detection. Unlike sanitizer.assert.access, this op does not assert that the access is individually valid and does not refine the continuing path. It records a memory event that target materialization can compare against prior unordered events in the selected race detector.
// sanitizer.race.access<read> %view[%lane] : view<64xi32, #dense>
LOOM_DEFINE_ISA(loom_sanitizer_race_access_isa, LOOM_OP_SANITIZER_RACE_ACCESS)
LOOM_DEFINE_OPERAND(loom_sanitizer_race_access_view, 0)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_sanitizer_race_access_indices, 1)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_sanitizer_race_access_kind, 0, loom_sanitizer_race_access_kind_t)
LOOM_DEFINE_ATTR_BOOL(loom_sanitizer_race_access_atomic, 1)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_sanitizer_race_access_ordering, 2, loom_atomic_ordering_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_sanitizer_race_access_scope, 3, loom_atomic_scope_t)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_sanitizer_race_access_static_indices, 4)
enum loom_sanitizer_race_access_build_flag_bits_e {
  LOOM_SANITIZER_RACE_ACCESS_BUILD_FLAG_HAS_ORDERING = 1u << 0,
  LOOM_SANITIZER_RACE_ACCESS_BUILD_FLAG_HAS_SCOPE = 1u << 1,
};
typedef uint32_t loom_sanitizer_race_access_build_flags_t;
iree_status_t loom_sanitizer_race_access_build(
    loom_builder_t* builder,
    loom_sanitizer_race_access_build_flags_t build_flags,
    loom_sanitizer_race_access_kind_t kind,
    loom_value_id_t view,
    const loom_value_id_t* indices,
    iree_host_size_t indices_count,
    const int64_t* static_indices,
    iree_host_size_t static_indices_count,
    bool atomic,
    loom_optional uint8_t ordering,
    loom_optional uint8_t scope,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_sanitizer_race_access_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_SANITIZER_RACE_SYNC: Observe a synchronization boundary for race detection. The original synchronization operation remains the semantic barrier or fence; this op records the boundary needed by race-detector materialization.
// sanitizer.race.sync<workgroup> {ordering = acq_rel, scope = workgroup}
LOOM_DEFINE_ISA(loom_sanitizer_race_sync_isa, LOOM_OP_SANITIZER_RACE_SYNC)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_sanitizer_race_sync_memory_space, 0, loom_value_fact_memory_space_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_sanitizer_race_sync_ordering, 1, loom_atomic_ordering_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_sanitizer_race_sync_scope, 2, loom_atomic_scope_t)
iree_status_t loom_sanitizer_race_sync_build(
    loom_builder_t* builder,
    loom_value_fact_memory_space_t memory_space,
    loom_atomic_ordering_t ordering,
    loom_atomic_scope_t scope,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_sanitizer_race_sync_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_SANITIZER_ASSERT_ACCESSES: Assert that a static sequence of regularly-strided logical indexed view accesses is valid. Each sub-access has the same static extent tuple, and each successive sub-access origin is advanced by the static stride tuple. This keeps structured fragment and vector footprints as one executable assertion site instead of exploding one source memory operation into many independent assertions.
// sanitizer.assert.accesses<write> %view[%row, %col] {static_extents = [1, 16], static_strides = [1, 0], static_count = 16} : view<128x128xf32, #dense>
LOOM_DEFINE_ISA(loom_sanitizer_assert_accesses_isa, LOOM_OP_SANITIZER_ASSERT_ACCESSES)
LOOM_DEFINE_OPERAND(loom_sanitizer_assert_accesses_view, 0)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_sanitizer_assert_accesses_indices, 1)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_sanitizer_assert_accesses_kind, 0, loom_sanitizer_assert_accesses_kind_t)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_sanitizer_assert_accesses_static_indices, 1)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_sanitizer_assert_accesses_static_extents, 2)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_sanitizer_assert_accesses_static_strides, 3)
LOOM_DEFINE_ATTR_I64(loom_sanitizer_assert_accesses_static_count, 4)
iree_status_t loom_sanitizer_assert_accesses_build(
    loom_builder_t* builder,
    loom_sanitizer_assert_accesses_kind_t kind,
    loom_value_id_t view,
    const loom_value_id_t* indices,
    iree_host_size_t indices_count,
    const int64_t* static_indices,
    iree_host_size_t static_indices_count,
    const int64_t* static_extents,
    iree_host_size_t static_extents_count,
    const int64_t* static_strides,
    iree_host_size_t static_strides_count,
    int64_t static_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_sanitizer_assert_accesses_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// Returns the vtable array for the sanitizer dialect.
const loom_op_vtable_t* const* loom_sanitizer_dialect_vtables(
    iree_host_size_t* out_count);

// Returns the dense semantic metadata array for the sanitizer dialect.
const loom_op_semantics_t* loom_sanitizer_dialect_op_semantics(
    iree_host_size_t* out_count);

// Returns semantic metadata for a sanitizer op kind, or empty metadata.
loom_op_semantics_t loom_sanitizer_op_semantics(
    loom_op_kind_t kind);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_SANITIZER_OPS_H_
