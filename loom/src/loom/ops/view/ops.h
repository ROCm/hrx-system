// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.
// Regenerate: python3 loom/py/loom/gen/run.py c_tables --in-place
// clang-format off

#ifndef LOOM_OPS_VIEW_OPS_H_
#define LOOM_OPS_VIEW_OPS_H_

#include "loom/ops/op_defs.h"
#include "loom/ops/atomic.h"
#include "loom/ops/cache.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  LOOM_OP_VIEW_SUBVIEW = LOOM_OP_KIND(LOOM_DIALECT_VIEW, 0),
  LOOM_OP_VIEW_REFINE = LOOM_OP_KIND(LOOM_DIALECT_VIEW, 1),
  LOOM_OP_VIEW_LOAD = LOOM_OP_KIND(LOOM_DIALECT_VIEW, 2),
  LOOM_OP_VIEW_STORE = LOOM_OP_KIND(LOOM_DIALECT_VIEW, 3),
  LOOM_OP_VIEW_ATOMIC_REDUCE = LOOM_OP_KIND(LOOM_DIALECT_VIEW, 4),
  LOOM_OP_VIEW_ATOMIC_RMW = LOOM_OP_KIND(LOOM_DIALECT_VIEW, 5),
  LOOM_OP_VIEW_ATOMIC_CMPXCHG = LOOM_OP_KIND(LOOM_DIALECT_VIEW, 6),
  LOOM_OP_VIEW_PREFETCH = LOOM_OP_KIND(LOOM_DIALECT_VIEW, 7),
  LOOM_OP_VIEW_COUNT_ = 8,
};

// Intended future access kind for a prefetch hint.
typedef enum loom_view_prefetch_intent_e {
  LOOM_VIEW_PREFETCH_INTENT_READ = 0,
  LOOM_VIEW_PREFETCH_INTENT_WRITE = 1,
  LOOM_VIEW_PREFETCH_INTENT_COUNT_ = 2,
} loom_view_prefetch_intent_t;

// Target-independent prefetch locality hint.
typedef enum loom_view_prefetch_locality_e {
  LOOM_VIEW_PREFETCH_LOCALITY_NONE = 0,
  LOOM_VIEW_PREFETCH_LOCALITY_L1 = 1,
  LOOM_VIEW_PREFETCH_LOCALITY_L2 = 2,
  LOOM_VIEW_PREFETCH_LOCALITY_L3 = 3,
  LOOM_VIEW_PREFETCH_LOCALITY_COUNT_ = 4,
} loom_view_prefetch_locality_t;

// LOOM_OP_VIEW_SUBVIEW: Form a logical subview from an existing view. Offsets select the logical origin; result type dimensions provide the subview extents.
// %sub = view.subview %source[%row, 0] : view<[%M]x[%N]xf32, %layout> -> view<16x[%N]xf32, %layout>
LOOM_DEFINE_ISA(loom_view_subview_isa, LOOM_OP_VIEW_SUBVIEW)
LOOM_DEFINE_OPERAND(loom_view_subview_source, 0)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_view_subview_offsets, 1)
LOOM_DEFINE_RESULT(loom_view_subview_result, 0)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_view_subview_static_offsets, 0)
iree_status_t loom_view_subview_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t source,
    const loom_value_id_t* offsets,
    iree_host_size_t offsets_count,
    const int64_t* static_offsets,
    iree_host_size_t static_offsets_count,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_view_subview_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_view_subview_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_VIEW_REFINE: Refine the static type information attached to an existing view while preserving the same storage root and byte base. This is an explicit SSA assertion point for layout, shape, and encoding facts discovered or required by earlier analysis.
// %refined = view.refine %view : view<[%M]xf32, %layout> -> view<16xf32, #dense>
LOOM_DEFINE_ISA(loom_view_refine_isa, LOOM_OP_VIEW_REFINE)
LOOM_DEFINE_OPERAND(loom_view_refine_source, 0)
LOOM_DEFINE_RESULT(loom_view_refine_result, 0)
iree_status_t loom_view_refine_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t source,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_view_refine_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_view_refine_type_transfer(
    loom_type_transfer_context_t* context,
    const loom_module_t* module, loom_op_t* op);
iree_status_t loom_view_refine_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_VIEW_LOAD: Load one scalar element from a typed view at a full-rank logical index. The index list is expressed in view coordinates and must name one position per view axis.
// %x = view.load %view[%row, %col] : view<[%M]x[%N]xf32, %layout> -> f32
LOOM_DEFINE_ISA(loom_view_load_isa, LOOM_OP_VIEW_LOAD)
LOOM_DEFINE_OPERAND(loom_view_load_view, 0)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_view_load_indices, 1)
LOOM_DEFINE_RESULT(loom_view_load_result, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_view_load_cache_scope, 0, loom_cache_scope_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_view_load_cache_temporal, 1, loom_cache_temporal_t)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_view_load_static_indices, 2)
enum loom_view_load_build_flag_bits_e {
  LOOM_VIEW_LOAD_BUILD_FLAG_HAS_CACHE_SCOPE = 1u << 0,
  LOOM_VIEW_LOAD_BUILD_FLAG_HAS_CACHE_TEMPORAL = 1u << 1,
};
typedef uint32_t loom_view_load_build_flags_t;
iree_status_t loom_view_load_build(
    loom_builder_t* builder,
    loom_view_load_build_flags_t build_flags,
    loom_may_consume loom_value_id_t view,
    const loom_value_id_t* indices,
    iree_host_size_t indices_count,
    const int64_t* static_indices,
    iree_host_size_t static_indices_count,
    loom_optional uint8_t cache_scope,
    loom_optional uint8_t cache_temporal,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_view_load_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_VIEW_STORE: Store one scalar element into a typed view at a full-rank logical index. The index list is expressed in view coordinates and must name one position per view axis.
// view.store %x, %view[%row, %col] : f32, view<[%M]x[%N]xf32, %layout>
LOOM_DEFINE_ISA(loom_view_store_isa, LOOM_OP_VIEW_STORE)
LOOM_DEFINE_OPERAND(loom_view_store_value, 0)
LOOM_DEFINE_OPERAND(loom_view_store_view, 1)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_view_store_indices, 2)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_view_store_cache_scope, 0, loom_cache_scope_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_view_store_cache_temporal, 1, loom_cache_temporal_t)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_view_store_static_indices, 2)
enum loom_view_store_build_flag_bits_e {
  LOOM_VIEW_STORE_BUILD_FLAG_HAS_CACHE_SCOPE = 1u << 0,
  LOOM_VIEW_STORE_BUILD_FLAG_HAS_CACHE_TEMPORAL = 1u << 1,
};
typedef uint32_t loom_view_store_build_flags_t;
iree_status_t loom_view_store_build(
    loom_builder_t* builder,
    loom_view_store_build_flags_t build_flags,
    loom_value_id_t value,
    loom_value_id_t view,
    const loom_value_id_t* indices,
    iree_host_size_t indices_count,
    const int64_t* static_indices,
    iree_host_size_t static_indices_count,
    loom_optional uint8_t cache_scope,
    loom_optional uint8_t cache_temporal,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_view_store_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_VIEW_ATOMIC_REDUCE: Atomically combine one scalar value into a typed view element at a full-rank logical index. Atomic ordering and scope are required so the synchronization contract remains explicit after vector-to-scalar lowering.
// view.atomic.reduce<addi> %value, %view[%row, %col] {ordering = relaxed, scope = workgroup} : i32, view<[%M]x[%N]xi32, %layout>
LOOM_DEFINE_ISA(loom_view_atomic_reduce_isa, LOOM_OP_VIEW_ATOMIC_REDUCE)
LOOM_DEFINE_OPERAND(loom_view_atomic_reduce_value, 0)
LOOM_DEFINE_OPERAND(loom_view_atomic_reduce_view, 1)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_view_atomic_reduce_indices, 2)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_view_atomic_reduce_kind, 0, loom_atomic_kind_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_view_atomic_reduce_ordering, 1, loom_atomic_ordering_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_view_atomic_reduce_scope, 2, loom_atomic_scope_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_view_atomic_reduce_cache_scope, 3, loom_cache_scope_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_view_atomic_reduce_cache_temporal, 4, loom_cache_temporal_t)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_view_atomic_reduce_static_indices, 5)
enum loom_view_atomic_reduce_build_flag_bits_e {
  LOOM_VIEW_ATOMIC_REDUCE_BUILD_FLAG_HAS_CACHE_SCOPE = 1u << 0,
  LOOM_VIEW_ATOMIC_REDUCE_BUILD_FLAG_HAS_CACHE_TEMPORAL = 1u << 1,
};
typedef uint32_t loom_view_atomic_reduce_build_flags_t;
iree_status_t loom_view_atomic_reduce_build(
    loom_builder_t* builder,
    loom_view_atomic_reduce_build_flags_t build_flags,
    loom_atomic_kind_t kind,
    loom_value_id_t value,
    loom_value_id_t view,
    const loom_value_id_t* indices,
    iree_host_size_t indices_count,
    const int64_t* static_indices,
    iree_host_size_t static_indices_count,
    loom_atomic_ordering_t ordering,
    loom_atomic_scope_t scope,
    loom_optional uint8_t cache_scope,
    loom_optional uint8_t cache_temporal,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_view_atomic_reduce_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_VIEW_ATOMIC_RMW: Atomically read one scalar view element, combine it with a scalar update value, write the combined value back, and return the old value observed by that atomic operation.
// %old = view.atomic.rmw<addi> %value, %view[%row, %col] {ordering = relaxed, scope = workgroup} : i32, view<[%M]x[%N]xi32, %layout> -> i32
LOOM_DEFINE_ISA(loom_view_atomic_rmw_isa, LOOM_OP_VIEW_ATOMIC_RMW)
LOOM_DEFINE_OPERAND(loom_view_atomic_rmw_value, 0)
LOOM_DEFINE_OPERAND(loom_view_atomic_rmw_view, 1)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_view_atomic_rmw_indices, 2)
LOOM_DEFINE_RESULT(loom_view_atomic_rmw_result, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_view_atomic_rmw_kind, 0, loom_atomic_kind_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_view_atomic_rmw_ordering, 1, loom_atomic_ordering_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_view_atomic_rmw_scope, 2, loom_atomic_scope_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_view_atomic_rmw_cache_scope, 3, loom_cache_scope_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_view_atomic_rmw_cache_temporal, 4, loom_cache_temporal_t)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_view_atomic_rmw_static_indices, 5)
enum loom_view_atomic_rmw_build_flag_bits_e {
  LOOM_VIEW_ATOMIC_RMW_BUILD_FLAG_HAS_CACHE_SCOPE = 1u << 0,
  LOOM_VIEW_ATOMIC_RMW_BUILD_FLAG_HAS_CACHE_TEMPORAL = 1u << 1,
};
typedef uint32_t loom_view_atomic_rmw_build_flags_t;
iree_status_t loom_view_atomic_rmw_build(
    loom_builder_t* builder,
    loom_view_atomic_rmw_build_flags_t build_flags,
    loom_atomic_kind_t kind,
    loom_may_consume loom_value_id_t value,
    loom_may_consume loom_value_id_t view,
    const loom_value_id_t* indices,
    iree_host_size_t indices_count,
    const int64_t* static_indices,
    iree_host_size_t static_indices_count,
    loom_atomic_ordering_t ordering,
    loom_atomic_scope_t scope,
    loom_optional uint8_t cache_scope,
    loom_optional uint8_t cache_temporal,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_view_atomic_rmw_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_VIEW_ATOMIC_CMPXCHG: Atomically compare one scalar view element with an expected value, write a replacement value when they match, and return the old observed value. Success is derived by comparing old == expected.
// %old = view.atomic.cmpxchg %expected, %replacement, %view[%row, %col] {success_ordering = acq_rel, failure_ordering = acquire, scope = workgroup} : i32, view<[%M]x[%N]xi32, %layout> -> i32
LOOM_DEFINE_ISA(loom_view_atomic_cmpxchg_isa, LOOM_OP_VIEW_ATOMIC_CMPXCHG)
LOOM_DEFINE_OPERAND(loom_view_atomic_cmpxchg_expected, 0)
LOOM_DEFINE_OPERAND(loom_view_atomic_cmpxchg_replacement, 1)
LOOM_DEFINE_OPERAND(loom_view_atomic_cmpxchg_view, 2)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_view_atomic_cmpxchg_indices, 3)
LOOM_DEFINE_RESULT(loom_view_atomic_cmpxchg_old, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_view_atomic_cmpxchg_success_ordering, 0, loom_atomic_ordering_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_view_atomic_cmpxchg_failure_ordering, 1, loom_atomic_ordering_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_view_atomic_cmpxchg_scope, 2, loom_atomic_scope_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_view_atomic_cmpxchg_cache_scope, 3, loom_cache_scope_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_view_atomic_cmpxchg_cache_temporal, 4, loom_cache_temporal_t)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_view_atomic_cmpxchg_static_indices, 5)
enum loom_view_atomic_cmpxchg_build_flag_bits_e {
  LOOM_VIEW_ATOMIC_CMPXCHG_BUILD_FLAG_HAS_CACHE_SCOPE = 1u << 0,
  LOOM_VIEW_ATOMIC_CMPXCHG_BUILD_FLAG_HAS_CACHE_TEMPORAL = 1u << 1,
};
typedef uint32_t loom_view_atomic_cmpxchg_build_flags_t;
iree_status_t loom_view_atomic_cmpxchg_build(
    loom_builder_t* builder,
    loom_view_atomic_cmpxchg_build_flags_t build_flags,
    loom_may_consume loom_value_id_t expected,
    loom_may_consume loom_value_id_t replacement,
    loom_may_consume loom_value_id_t view,
    const loom_value_id_t* indices,
    iree_host_size_t indices_count,
    const int64_t* static_indices,
    iree_host_size_t static_indices_count,
    loom_atomic_ordering_t success_ordering,
    loom_atomic_ordering_t failure_ordering,
    loom_atomic_scope_t scope,
    loom_optional uint8_t cache_scope,
    loom_optional uint8_t cache_temporal,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_view_atomic_cmpxchg_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_VIEW_PREFETCH: Compiler hint for a future access to a logical view origin. Prefetch has no semantic memory effects and may not fault semantically, but it is intentionally preserved by ordinary canonicalization/DCE until an explicit hint-stripping pass removes it.
// view.prefetch %view[%row, %col] {intent = read, locality = l2} : view<[%M]x[%N]xf32, %layout>
LOOM_DEFINE_ISA(loom_view_prefetch_isa, LOOM_OP_VIEW_PREFETCH)
LOOM_DEFINE_OPERAND(loom_view_prefetch_view, 0)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_view_prefetch_indices, 1)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_view_prefetch_intent, 0, loom_view_prefetch_intent_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_view_prefetch_locality, 1, loom_view_prefetch_locality_t)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_view_prefetch_static_indices, 2)
iree_status_t loom_view_prefetch_build(
    loom_builder_t* builder,
    loom_value_id_t view,
    const loom_value_id_t* indices,
    iree_host_size_t indices_count,
    const int64_t* static_indices,
    iree_host_size_t static_indices_count,
    loom_view_prefetch_intent_t intent,
    loom_view_prefetch_locality_t locality,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_view_prefetch_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// Returns the vtable array for the view dialect.
const loom_op_vtable_t* const* loom_view_dialect_vtables(
    iree_host_size_t* out_count);

// Returns the dense semantic metadata array for the view dialect.
const loom_op_semantics_t* loom_view_dialect_op_semantics(
    iree_host_size_t* out_count);

// Returns semantic metadata for a view op kind, or empty metadata.
loom_op_semantics_t loom_view_op_semantics(
    loom_op_kind_t kind);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_VIEW_OPS_H_
