// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.
// Regenerate: python3 loom/py/loom/gen/run.py c_tables --in-place
// clang-format off

#ifndef LOOM_OPS_PIPELINE_OPS_H_
#define LOOM_OPS_PIPELINE_OPS_H_

#include "loom/ops/op_defs.h"
#include "loom/ir/memory_space.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  LOOM_OP_PIPELINE_DEF = LOOM_OP_KIND(LOOM_DIALECT_PIPELINE, 0),
  LOOM_OP_PIPELINE_FINISH = LOOM_OP_KIND(LOOM_DIALECT_PIPELINE, 1),
  LOOM_OP_PIPELINE_STRAND = LOOM_OP_KIND(LOOM_DIALECT_PIPELINE, 2),
  LOOM_OP_PIPELINE_END = LOOM_OP_KIND(LOOM_DIALECT_PIPELINE, 3),
  LOOM_OP_PIPELINE_COMPOSE = LOOM_OP_KIND(LOOM_DIALECT_PIPELINE, 4),
  LOOM_OP_PIPELINE_MEMORY = LOOM_OP_KIND(LOOM_DIALECT_PIPELINE, 5),
  LOOM_OP_PIPELINE_DECL = LOOM_OP_KIND(LOOM_DIALECT_PIPELINE, 6),
  LOOM_OP_PIPELINE_SERIAL = LOOM_OP_KIND(LOOM_DIALECT_PIPELINE, 7),
  LOOM_OP_PIPELINE_YIELD = LOOM_OP_KIND(LOOM_DIALECT_PIPELINE, 8),
  LOOM_OP_PIPELINE_CONCURRENT = LOOM_OP_KIND(LOOM_DIALECT_PIPELINE, 9),
  LOOM_OP_PIPELINE_COUNT_ = 10,
};

// Required materialization boundary. An absent scope permits a generic pipeline program that may span targets and runtime operations.
typedef enum loom_pipeline_scope_e {
  LOOM_PIPELINE_SCOPE_KERNEL = 1,
  LOOM_PIPELINE_SCOPE_COUNT_ = 2,
} loom_pipeline_scope_t;

// Symbol visibility. Absent (0) means private (module-internal).
typedef enum loom_pipeline_visibility_e {
  LOOM_PIPELINE_VISIBILITY_PUBLIC = 1,
  LOOM_PIPELINE_VISIBILITY_COUNT_ = 2,
} loom_pipeline_visibility_t;

// Private symbol retention policy. Absent (0) permits ordinary DCE.
typedef enum loom_pipeline_retain_e {
  LOOM_PIPELINE_RETAIN_RETAIN = 1,
  LOOM_PIPELINE_RETAIN_COUNT_ = 2,
} loom_pipeline_retain_t;

// Channel participation within a strand. Absent participation creates an independent instance per selected worker.
typedef enum loom_pipeline_strand_participation_e {
  LOOM_PIPELINE_STRAND_PARTICIPATION_COOPERATIVE = 1,
  LOOM_PIPELINE_STRAND_PARTICIPATION_COUNT_ = 2,
} loom_pipeline_strand_participation_t;

// LOOM_OP_PIPELINE_DEF: Persistent dataflow program. Leading specialization arguments remain ordinary SSA values and typed run arguments are supplied by each invocation. Their source types are independent of the eventual target export ABI. The optional scope fixes the artifact boundary that lowering must satisfy.
// pipeline.def<kernel> target(@array) @resident() run(%input: buffer, %output: buffer) {
// }
LOOM_DEFINE_ISA(loom_pipeline_def_isa, LOOM_OP_PIPELINE_DEF)
LOOM_DEFINE_ATTR_SYMBOL(loom_pipeline_def_callee, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_pipeline_def_scope, 1, loom_pipeline_scope_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_pipeline_def_visibility, 2, loom_pipeline_visibility_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_pipeline_def_retain, 3, loom_pipeline_retain_t)
LOOM_DEFINE_ATTR_SYMBOL(loom_pipeline_def_target, 4)
LOOM_DEFINE_ATTR_PREDICATE_LIST(loom_pipeline_def_predicates, 5)
LOOM_DEFINE_ATTR_I64(loom_pipeline_def_specialization_count, 6)
LOOM_DEFINE_REGION(loom_pipeline_def_body, 0)
enum loom_pipeline_def_build_flag_bits_e {
  LOOM_PIPELINE_DEF_BUILD_FLAG_HAS_SCOPE = 1u << 0,
  LOOM_PIPELINE_DEF_BUILD_FLAG_HAS_VISIBILITY = 1u << 1,
  LOOM_PIPELINE_DEF_BUILD_FLAG_HAS_RETAIN = 1u << 2,
  LOOM_PIPELINE_DEF_BUILD_FLAG_HAS_TARGET = 1u << 3,
  LOOM_PIPELINE_DEF_BUILD_FLAG_HAS_PREDICATES = 1u << 4,
};
typedef uint32_t loom_pipeline_def_build_flags_t;
iree_status_t loom_pipeline_def_build(
    loom_builder_t* builder,
    loom_pipeline_def_build_flags_t build_flags,
    loom_optional uint8_t scope,
    loom_optional uint8_t visibility,
    loom_optional uint8_t retain,
    loom_optional loom_symbol_ref_t target,
    loom_symbol_ref_t callee,
    const loom_type_t* specializations_types,
    iree_host_size_t specializations_types_count,
    const loom_type_t* bindings_types,
    iree_host_size_t bindings_types_count,
    loom_optional const loom_predicate_t* predicates,
    iree_host_size_t predicates_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_pipeline_signature_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_PIPELINE_FINISH: Complete the pipeline body. Invocation completion follows the execution and communication obligations of the materialized program.
// pipeline.finish
LOOM_DEFINE_ISA(loom_pipeline_finish_isa, LOOM_OP_PIPELINE_FINISH)
iree_status_t loom_pipeline_finish_build(
    loom_builder_t* builder,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_PIPELINE_STRAND: Construct strand instances over a target-relative worker domain. By default each selected worker enters an independent instance once. A cooperative strand instead enters one instance with all selected workers: channel actions collectively acquire, publish or release one record, while ordinary payload operations execute per participant. Channel actions require converged participation, including through calls. Ordinary loops in the body express repeated work. The region captures lexical SSA values and declares work rather than executing inline. Multiple strands may share a worker, with channel dependencies controlling their progress. Origins, counts and strides select coordinates as origin + index * stride. An omitted target inherits the enclosing pipeline's target environment.
// pipeline.strand target(@npu) workers([%column, 2], [1, 1], [1, 1]) {
// }
LOOM_DEFINE_ISA(loom_pipeline_strand_isa, LOOM_OP_PIPELINE_STRAND)
LOOM_DEFINE_SEGMENTED_OPERANDS(loom_pipeline_strand_origins, 0)
LOOM_DEFINE_SEGMENTED_OPERANDS(loom_pipeline_strand_counts, 1)
LOOM_DEFINE_SEGMENTED_OPERANDS(loom_pipeline_strand_strides, 2)
LOOM_DEFINE_ATTR_SYMBOL(loom_pipeline_strand_target, 0)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_pipeline_strand_static_origins, 1)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_pipeline_strand_static_counts, 2)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_pipeline_strand_static_strides, 3)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_pipeline_strand_participation, 4, loom_pipeline_strand_participation_t)
LOOM_DEFINE_REGION(loom_pipeline_strand_body, 0)
enum loom_pipeline_strand_build_flag_bits_e {
  LOOM_PIPELINE_STRAND_BUILD_FLAG_HAS_PARTICIPATION = 1u << 0,
  LOOM_PIPELINE_STRAND_BUILD_FLAG_HAS_TARGET = 1u << 1,
};
typedef uint32_t loom_pipeline_strand_build_flags_t;
iree_status_t loom_pipeline_strand_build(
    loom_builder_t* builder,
    loom_pipeline_strand_build_flags_t build_flags,
    loom_optional uint8_t participation,
    loom_optional loom_symbol_ref_t target,
    const loom_value_id_t* origins,
    iree_host_size_t origins_count,
    const int64_t* static_origins,
    iree_host_size_t static_origins_count,
    const loom_value_id_t* counts,
    iree_host_size_t counts_count,
    const int64_t* static_counts,
    iree_host_size_t static_counts_count,
    const loom_value_id_t* strides,
    iree_host_size_t strides_count,
    const int64_t* static_strides,
    iree_host_size_t static_strides_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_pipeline_strand_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_PIPELINE_END: Complete a strand instance. Implicit at the end of its region.
// pipeline.end
LOOM_DEFINE_ISA(loom_pipeline_end_isa, LOOM_OP_PIPELINE_END)
iree_status_t loom_pipeline_end_build(
    loom_builder_t* builder,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_PIPELINE_COMPOSE: Compose a child pipeline into the enclosing invocation. Each composition constructs fresh child storage and independently progressing strands; it never submits another invocation. When the nearest schedule is pipeline.serial the child completes before that schedule advances. In a concurrent graph, composition adds no sibling dependency; the enclosing group owns its completion boundary. Specialization and run operands substitute the child's complete typed signature. Child target and materialization requirements remain binding.
// pipeline.compose @column[%column](%input, %output) : [index](channel<tile<16xi32>>, channel<tile<16xi32>>)
LOOM_DEFINE_ISA(loom_pipeline_compose_isa, LOOM_OP_PIPELINE_COMPOSE)
LOOM_DEFINE_SEGMENTED_OPERANDS(loom_pipeline_compose_specializations, 0)
LOOM_DEFINE_SEGMENTED_OPERANDS(loom_pipeline_compose_bindings, 1)
LOOM_DEFINE_ATTR_SYMBOL(loom_pipeline_compose_callee, 0)
iree_status_t loom_pipeline_compose_build(
    loom_builder_t* builder,
    loom_symbol_ref_t callee,
    const loom_value_id_t* specializations,
    iree_host_size_t specializations_count,
    const loom_value_id_t* bindings,
    iree_host_size_t bindings_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_pipeline_compose_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_PIPELINE_MEMORY: Select a borrowed memory pool at worker coordinates in the enclosing pipeline invocation. Coordinates use the strand worker domain; the invocation supplies the execution instance, not the target symbol. Repeated selections of the same memory identify the same pool. The query neither allocates storage nor synchronizes access: buffer.alloca creates fresh allocation roots and channel operations govern record ownership. Each accessing worker must have a legal mapping to the selected backing. The pool remains valid for the invocation and may be passed to generic construction helpers. Selection belongs to construction outside strand bodies. Composed children borrow the enclosing invocation's coordinate space and backing pools; the materializing target verifies that the selected pool exists. A memory selection does not require the child to own an executable artifact.
// %memory = pipeline.memory<workgroup>[%column, 3] : pool
LOOM_DEFINE_ISA(loom_pipeline_memory_isa, LOOM_OP_PIPELINE_MEMORY)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_pipeline_memory_coordinates, 0)
LOOM_DEFINE_RESULT(loom_pipeline_memory_result, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_pipeline_memory_memory_space, 0, loom_value_fact_memory_space_t)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_pipeline_memory_static_coordinates, 1)
iree_status_t loom_pipeline_memory_build(
    loom_builder_t* builder,
    loom_value_fact_memory_space_t memory_space,
    const loom_value_id_t* coordinates,
    iree_host_size_t coordinates_count,
    const int64_t* static_coordinates,
    iree_host_size_t static_coordinates_count,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_pipeline_memory_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_PIPELINE_DECL: Bodyless persistent pipeline contract. The leading specialization arguments, invocation arguments, target, predicates, and optional artifact scope have the same meaning as on pipeline.def.
// pipeline.decl<kernel> @resident(%width: index) run(%input: view<[%width]xi32>, %output: buffer)
LOOM_DEFINE_ISA(loom_pipeline_decl_isa, LOOM_OP_PIPELINE_DECL)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_pipeline_decl_args, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_pipeline_decl_callee, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_pipeline_decl_scope, 1, loom_pipeline_scope_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_pipeline_decl_visibility, 2, loom_pipeline_visibility_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_pipeline_decl_retain, 3, loom_pipeline_retain_t)
LOOM_DEFINE_ATTR_SYMBOL(loom_pipeline_decl_target, 4)
LOOM_DEFINE_ATTR_PREDICATE_LIST(loom_pipeline_decl_predicates, 5)
LOOM_DEFINE_ATTR_I64(loom_pipeline_decl_specialization_count, 6)
enum loom_pipeline_decl_build_flag_bits_e {
  LOOM_PIPELINE_DECL_BUILD_FLAG_HAS_SCOPE = 1u << 0,
  LOOM_PIPELINE_DECL_BUILD_FLAG_HAS_VISIBILITY = 1u << 1,
  LOOM_PIPELINE_DECL_BUILD_FLAG_HAS_RETAIN = 1u << 2,
  LOOM_PIPELINE_DECL_BUILD_FLAG_HAS_TARGET = 1u << 3,
  LOOM_PIPELINE_DECL_BUILD_FLAG_HAS_PREDICATES = 1u << 4,
};
typedef uint32_t loom_pipeline_decl_build_flags_t;
iree_status_t loom_pipeline_decl_build(
    loom_builder_t* builder,
    loom_pipeline_decl_build_flags_t build_flags,
    loom_optional uint8_t scope,
    loom_optional uint8_t visibility,
    loom_optional uint8_t retain,
    loom_optional loom_symbol_ref_t target,
    loom_symbol_ref_t callee,
    const loom_type_t* specializations_types,
    iree_host_size_t specializations_types_count,
    const loom_type_t* bindings_types,
    iree_host_size_t bindings_types_count,
    loom_optional const loom_predicate_t* predicates,
    iree_host_size_t predicates_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_pipeline_signature_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_PIPELINE_SERIAL: Order complete child pipelines within one resident invocation. Each composition in the region completes its strands and asynchronous memory operations before the schedule advances. Structured control repeats or selects child lifetimes; nested schedule regions complete before their enclosing schedule advances. Completed writes are visible to subsequent children. Outer storage and published channel records survive the boundary; completion neither empties outer channels nor orders unrelated concurrent strands. Child completion cannot retain uncompleted transfers or owned read/write capabilities. Strands belong to composed child definitions or concurrent regions.
// pipeline.serial {
// }
LOOM_DEFINE_ISA(loom_pipeline_serial_isa, LOOM_OP_PIPELINE_SERIAL)
LOOM_DEFINE_REGION(loom_pipeline_serial_body, 0)
iree_status_t loom_pipeline_serial_build(
    loom_builder_t* builder,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_PIPELINE_YIELD: Complete a resident schedule region. Implicit at the end of its region.
// pipeline.yield
LOOM_DEFINE_ISA(loom_pipeline_yield_isa, LOOM_OP_PIPELINE_YIELD)
iree_status_t loom_pipeline_yield_build(
    loom_builder_t* builder,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_pipeline_yield_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_PIPELINE_CONCURRENT: Construct a concurrent child graph within one resident invocation. Sibling strands, compositions, and nested schedules progress without dependency edges between siblings. The complete group joins before its enclosing serial schedule advances. Local storage and channels belong to the group; outer storage and published records survive its completion. The single-block graph has lexical captures and no results. Runtime repetition and selection belong in nested serial branches; nesting serial and concurrent groups preserves each group's completion boundary.
// pipeline.concurrent {
//   pipeline.serial {
//   }
// }
LOOM_DEFINE_ISA(loom_pipeline_concurrent_isa, LOOM_OP_PIPELINE_CONCURRENT)
LOOM_DEFINE_REGION(loom_pipeline_concurrent_body, 0)
iree_status_t loom_pipeline_concurrent_build(
    loom_builder_t* builder,
    loom_location_id_t location,
    loom_op_t** out_op);

// Returns the vtable array for the pipeline dialect.
const loom_op_vtable_t* const* loom_pipeline_dialect_vtables(
    iree_host_size_t* out_count);

// Returns the dense semantic metadata array for the pipeline dialect.
const loom_op_semantics_t* loom_pipeline_dialect_op_semantics(
    iree_host_size_t* out_count);

// Returns semantic metadata for a pipeline op kind, or empty metadata.
loom_op_semantics_t loom_pipeline_op_semantics(
    loom_op_kind_t kind);

#ifdef __cplusplus
}
#endif

// Additional named attribute helpers are generated with this dialect.
#include "loom/ops/pipeline/ops.inc"

#endif  // LOOM_OPS_PIPELINE_OPS_H_
