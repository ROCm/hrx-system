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

#ifdef __cplusplus
extern "C" {
#endif

enum {
  LOOM_OP_PIPELINE_DEF = LOOM_OP_KIND(LOOM_DIALECT_PIPELINE, 0),
  LOOM_OP_PIPELINE_SCATTER = LOOM_OP_KIND(LOOM_DIALECT_PIPELINE, 1),
  LOOM_OP_PIPELINE_READ = LOOM_OP_KIND(LOOM_DIALECT_PIPELINE, 2),
  LOOM_OP_PIPELINE_STAGE = LOOM_OP_KIND(LOOM_DIALECT_PIPELINE, 3),
  LOOM_OP_PIPELINE_BUFFER = LOOM_OP_KIND(LOOM_DIALECT_PIPELINE, 4),
  LOOM_OP_PIPELINE_REDUCE = LOOM_OP_KIND(LOOM_DIALECT_PIPELINE, 5),
  LOOM_OP_PIPELINE_WRITE = LOOM_OP_KIND(LOOM_DIALECT_PIPELINE, 6),
  LOOM_OP_PIPELINE_RETURN = LOOM_OP_KIND(LOOM_DIALECT_PIPELINE, 7),
  LOOM_OP_PIPELINE_COUNT_ = 8,
};

// Required materialization boundary. An absent scope permits a generic pipeline program that may span targets and runtime operations.
typedef enum loom_pipeline_def_scope_e {
  LOOM_PIPELINE_DEF_SCOPE_KERNEL = 1,
  LOOM_PIPELINE_DEF_SCOPE_COMMAND = 2,
  LOOM_PIPELINE_DEF_SCOPE_COUNT_ = 3,
} loom_pipeline_def_scope_t;

// Function visibility. Absent (0) means private (module-internal).
typedef enum loom_pipeline_def_visibility_e {
  LOOM_PIPELINE_DEF_VISIBILITY_PUBLIC = 1,
  LOOM_PIPELINE_DEF_VISIBILITY_COUNT_ = 2,
} loom_pipeline_def_visibility_t;

// Private symbol retention policy. Absent (0) permits ordinary DCE.
typedef enum loom_pipeline_def_retain_e {
  LOOM_PIPELINE_DEF_RETAIN_RETAIN = 1,
  LOOM_PIPELINE_DEF_RETAIN_COUNT_ = 2,
} loom_pipeline_def_retain_t;

// LOOM_OP_PIPELINE_DEF: Persistent dataflow program. Leading specialization arguments remain ordinary SSA values and launch bindings are supplied when the materialized pipeline is issued. The optional scope fixes the artifact boundary that lowering must satisfy.
// pipeline.def<kernel> target(@array) @resident() launch(%input: buffer, %output: buffer) {
//   pipeline.return
// }
LOOM_DEFINE_ISA(loom_pipeline_def_isa, LOOM_OP_PIPELINE_DEF)
LOOM_DEFINE_ATTR_SYMBOL(loom_pipeline_def_callee, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_pipeline_def_scope, 1, loom_pipeline_def_scope_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_pipeline_def_visibility, 2, loom_pipeline_def_visibility_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_pipeline_def_retain, 3, loom_pipeline_def_retain_t)
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
iree_status_t loom_pipeline_def_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_PIPELINE_SCATTER: Partition the leading dimension of a source view across a scheduling group, producing one suffix-shaped tile record per lane.
// %tiles = pipeline.scatter %source across %workers : view<2x8x8xi8>, group -> pipeline.flow<tile<8x8xi8>>
LOOM_DEFINE_ISA(loom_pipeline_scatter_isa, LOOM_OP_PIPELINE_SCATTER)
LOOM_DEFINE_OPERAND(loom_pipeline_scatter_source, 0)
LOOM_DEFINE_OPERAND(loom_pipeline_scatter_group, 1)
LOOM_DEFINE_RESULT(loom_pipeline_scatter_result, 0)
iree_status_t loom_pipeline_scatter_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t source,
    loom_may_consume loom_value_id_t group,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_pipeline_scatter_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_PIPELINE_READ: Read one complete source-view tile record for each destination lane.
// %bias = pipeline.read %source on %reducers : view<8x8xi32>, group -> pipeline.flow<tile<8x8xi32>>
LOOM_DEFINE_ISA(loom_pipeline_read_isa, LOOM_OP_PIPELINE_READ)
LOOM_DEFINE_OPERAND(loom_pipeline_read_source, 0)
LOOM_DEFINE_OPERAND(loom_pipeline_read_group, 1)
LOOM_DEFINE_RESULT(loom_pipeline_read_result, 0)
iree_status_t loom_pipeline_read_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t source,
    loom_may_consume loom_value_id_t group,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_pipeline_read_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_PIPELINE_STAGE: Instantiate one stage invocation per group lane. Inputs and outputs are lane-wise typed flows; the referenced callable supplies one record-firing implementation.
// %partials = pipeline.stage @product on %workers(%lhs, %rhs) : (group, pipeline.flow<tile<8x8xi8>>, pipeline.flow<tile<8x8xi8>>) -> (pipeline.flow<tile<8x8xi32>>)
LOOM_DEFINE_ISA(loom_pipeline_stage_isa, LOOM_OP_PIPELINE_STAGE)
LOOM_DEFINE_OPERAND(loom_pipeline_stage_group, 0)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_pipeline_stage_inputs, 1)
LOOM_DEFINE_VARIADIC_RESULTS(loom_pipeline_stage_outputs, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_pipeline_stage_entry, 0)
iree_status_t loom_pipeline_stage_build(
    loom_builder_t* builder,
    loom_symbol_ref_t entry,
    loom_may_consume loom_value_id_t group,
    loom_may_consume const loom_value_id_t* inputs,
    iree_host_size_t inputs_count,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    const loom_tied_result_t* tied_results,
    iree_host_size_t tied_result_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_pipeline_stage_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_PIPELINE_BUFFER: Require an independently buffered flow with SSA-defined minimum record capacity. Targets may select a greater capacity.
// %buffered = pipeline.buffer %partials capacity %ring_capacity : (pipeline.flow<tile<8x8xi32>>, index) -> pipeline.flow<tile<8x8xi32>>
LOOM_DEFINE_ISA(loom_pipeline_buffer_isa, LOOM_OP_PIPELINE_BUFFER)
LOOM_DEFINE_OPERAND(loom_pipeline_buffer_source, 0)
LOOM_DEFINE_OPERAND(loom_pipeline_buffer_capacity, 1)
LOOM_DEFINE_RESULT(loom_pipeline_buffer_result, 0)
iree_status_t loom_pipeline_buffer_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t source,
    loom_may_consume loom_value_id_t capacity,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_pipeline_buffer_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_PIPELINE_REDUCE: Gather each source-group lane record into one target-group stage firing. Target inputs remain pointwise with the target group.
// %result = pipeline.reduce @sum from %products(%partials) to %reducers(%bias) : (group, pipeline.flow<tile<8x8xi32>>) to (group, pipeline.flow<tile<8x8xi32>>) -> (pipeline.flow<tile<8x8xi32>>)
LOOM_DEFINE_ISA(loom_pipeline_reduce_isa, LOOM_OP_PIPELINE_REDUCE)
LOOM_DEFINE_SEGMENTED_OPERAND(loom_pipeline_reduce_source_group, 0)
LOOM_DEFINE_SEGMENTED_OPERANDS(loom_pipeline_reduce_source_inputs, 1)
LOOM_DEFINE_SEGMENTED_OPERAND(loom_pipeline_reduce_target_group, 2)
LOOM_DEFINE_SEGMENTED_OPERANDS(loom_pipeline_reduce_target_inputs, 3)
LOOM_DEFINE_VARIADIC_RESULTS(loom_pipeline_reduce_outputs, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_pipeline_reduce_entry, 0)
iree_status_t loom_pipeline_reduce_build(
    loom_builder_t* builder,
    loom_symbol_ref_t entry,
    loom_may_consume loom_value_id_t source_group,
    loom_may_consume const loom_value_id_t* source_inputs,
    iree_host_size_t source_inputs_count,
    loom_may_consume loom_value_id_t target_group,
    loom_may_consume const loom_value_id_t* target_inputs,
    iree_host_size_t target_inputs_count,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    const loom_tied_result_t* tied_results,
    iree_host_size_t tied_result_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_pipeline_reduce_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_PIPELINE_WRITE: Write each source-group tile record to a destination view.
// pipeline.write %result to %output : pipeline.flow<tile<8x8xi32>>, view<8x8xi32>
LOOM_DEFINE_ISA(loom_pipeline_write_isa, LOOM_OP_PIPELINE_WRITE)
LOOM_DEFINE_OPERAND(loom_pipeline_write_source, 0)
LOOM_DEFINE_OPERAND(loom_pipeline_write_target, 1)
iree_status_t loom_pipeline_write_build(
    loom_builder_t* builder,
    loom_value_id_t source,
    loom_value_id_t target,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_pipeline_write_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_PIPELINE_RETURN: Terminate a pipeline definition.
// pipeline.return
LOOM_DEFINE_ISA(loom_pipeline_return_isa, LOOM_OP_PIPELINE_RETURN)
iree_status_t loom_pipeline_return_build(
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

#endif  // LOOM_OPS_PIPELINE_OPS_H_
