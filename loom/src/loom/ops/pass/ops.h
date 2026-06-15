// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.
// Regenerate: python3 loom/py/loom/gen/run.py c_tables --in-place
// clang-format off

#ifndef LOOM_OPS_PASS_OPS_H_
#define LOOM_OPS_PASS_OPS_H_

#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  LOOM_OP_PASS_PIPELINE = LOOM_OP_KIND(LOOM_DIALECT_PASS, 0),
  LOOM_OP_PASS_FOR = LOOM_OP_KIND(LOOM_DIALECT_PASS, 1),
  LOOM_OP_PASS_WHERE = LOOM_OP_KIND(LOOM_DIALECT_PASS, 2),
  LOOM_OP_PASS_REPEAT = LOOM_OP_KIND(LOOM_DIALECT_PASS, 3),
  LOOM_OP_PASS_CALL = LOOM_OP_KIND(LOOM_DIALECT_PASS, 4),
  LOOM_OP_PASS_RUN = LOOM_OP_KIND(LOOM_DIALECT_PASS, 5),
  LOOM_OP_PASS_FAIL = LOOM_OP_KIND(LOOM_DIALECT_PASS, 6),
  LOOM_OP_PASS_HALT = LOOM_OP_KIND(LOOM_DIALECT_PASS, 7),
  LOOM_OP_PASS_YIELD = LOOM_OP_KIND(LOOM_DIALECT_PASS, 8),
  LOOM_OP_PASS_COUNT_ = 9,
};

// Pass pipeline execution anchor.
typedef enum loom_pass_anchor_e {
  LOOM_PASS_ANCHOR_MODULE = 0,
  LOOM_PASS_ANCHOR_FUNC = 1,
  LOOM_PASS_ANCHOR_COUNT_ = 2,
} loom_pass_anchor_t;

// Pass pipeline repeat mode.
typedef enum loom_pass_repeat_mode_e {
  LOOM_PASS_REPEAT_MODE_FIXED = 0,
  LOOM_PASS_REPEAT_MODE_UNTIL_CONVERGED = 1,
  LOOM_PASS_REPEAT_MODE_COUNT_ = 2,
} loom_pass_repeat_mode_t;

// LOOM_OP_PASS_PIPELINE: Named pass pipeline entry point.
// pass.pipeline<module> @cleanup pipeline {
//   canonicalize
// }
LOOM_DEFINE_ISA(loom_pass_pipeline_isa, LOOM_OP_PASS_PIPELINE)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_pass_pipeline_anchor, 0, loom_pass_anchor_t)
LOOM_DEFINE_ATTR_SYMBOL(loom_pass_pipeline_symbol, 1)
LOOM_DEFINE_REGION(loom_pass_pipeline_body, 0)
iree_status_t loom_pass_pipeline_build(
    loom_builder_t* builder,
    loom_pass_anchor_t anchor,
    loom_symbol_ref_t symbol,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_PASS_FOR: Deterministically iterate symbols of the selected anchor kind.
// pass.for<func> pipeline {
//   cse
// }
LOOM_DEFINE_ISA(loom_pass_for_isa, LOOM_OP_PASS_FOR)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_pass_for_anchor, 0, loom_pass_anchor_t)
LOOM_DEFINE_REGION(loom_pass_for_body, 0)
iree_status_t loom_pass_for_build(
    loom_builder_t* builder,
    loom_pass_anchor_t anchor,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_PASS_WHERE: Guard a nested pipeline body with a descriptor-backed pass predicate.
// pass.where<name> {value = "matmul"} pipeline {
//   canonicalize
// }
LOOM_DEFINE_ISA(loom_pass_where_isa, LOOM_OP_PASS_WHERE)
LOOM_DEFINE_ATTR_STRING(loom_pass_where_predicate, 0)
LOOM_DEFINE_ATTR_DICT(loom_pass_where_attrs, 1)
LOOM_DEFINE_REGION(loom_pass_where_body, 0)
iree_status_t loom_pass_where_build(
    loom_builder_t* builder,
    loom_string_id_t predicate,
    loom_optional loom_named_attr_slice_t attrs,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_PASS_REPEAT: Repeat a nested pipeline body.
// pass.repeat<fixed> {count = 2} pipeline {
//   canonicalize
// }
LOOM_DEFINE_ISA(loom_pass_repeat_isa, LOOM_OP_PASS_REPEAT)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_pass_repeat_mode, 0, loom_pass_repeat_mode_t)
LOOM_DEFINE_ATTR_I64(loom_pass_repeat_count, 1)
LOOM_DEFINE_ATTR_I64(loom_pass_repeat_max_iterations, 2)
LOOM_DEFINE_REGION(loom_pass_repeat_body, 0)
enum loom_pass_repeat_build_flag_bits_e {
  LOOM_PASS_REPEAT_BUILD_FLAG_HAS_COUNT = 1u << 0,
  LOOM_PASS_REPEAT_BUILD_FLAG_HAS_MAX_ITERATIONS = 1u << 1,
};
typedef uint32_t loom_pass_repeat_build_flags_t;
iree_status_t loom_pass_repeat_build(
    loom_builder_t* builder,
    loom_pass_repeat_build_flags_t build_flags,
    loom_pass_repeat_mode_t mode,
    loom_optional int64_t count,
    loom_optional int64_t max_iterations,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_PASS_CALL: Statically call another named pass pipeline.
// pass.call @cleanup
LOOM_DEFINE_ISA(loom_pass_call_isa, LOOM_OP_PASS_CALL)
LOOM_DEFINE_ATTR_SYMBOL(loom_pass_call_callee, 0)
iree_status_t loom_pass_call_build(
    loom_builder_t* builder,
    loom_symbol_ref_t callee,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_PASS_RUN: Descriptor-backed leaf pass invocation.
// pass.run<canonicalize>
LOOM_DEFINE_ISA(loom_pass_run_isa, LOOM_OP_PASS_RUN)
LOOM_DEFINE_ATTR_STRING(loom_pass_run_key, 0)
LOOM_DEFINE_ATTR_DICT(loom_pass_run_options, 1)
iree_status_t loom_pass_run_build(
    loom_builder_t* builder,
    loom_string_id_t key,
    loom_optional loom_named_attr_slice_t options,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_PASS_FAIL: Emit a structured pipeline assertion failure.
// pass.fail "expected canonical form"
LOOM_DEFINE_ISA(loom_pass_fail_isa, LOOM_OP_PASS_FAIL)
LOOM_DEFINE_ATTR_STRING(loom_pass_fail_message, 0)
iree_status_t loom_pass_fail_build(
    loom_builder_t* builder,
    loom_string_id_t message,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_PASS_HALT: Deliberately stop pipeline execution with a diagnostic message.
// pass.halt "inspect lowered IR"
LOOM_DEFINE_ISA(loom_pass_halt_isa, LOOM_OP_PASS_HALT)
LOOM_DEFINE_ATTR_STRING(loom_pass_halt_message, 0)
iree_status_t loom_pass_halt_build(
    loom_builder_t* builder,
    loom_string_id_t message,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_PASS_YIELD: Terminate a pass pipeline control region.
// pass.yield
LOOM_DEFINE_ISA(loom_pass_yield_isa, LOOM_OP_PASS_YIELD)
iree_status_t loom_pass_yield_build(
    loom_builder_t* builder,
    loom_location_id_t location,
    loom_op_t** out_op);

// Returns the vtable array for the pass dialect.
const loom_op_vtable_t* const* loom_pass_dialect_vtables(
    iree_host_size_t* out_count);

// Returns the dense semantic metadata array for the pass dialect.
const loom_op_semantics_t* loom_pass_dialect_op_semantics(
    iree_host_size_t* out_count);

// Returns semantic metadata for a pass op kind, or empty metadata.
loom_op_semantics_t loom_pass_op_semantics(
    loom_op_kind_t kind);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_PASS_OPS_H_
