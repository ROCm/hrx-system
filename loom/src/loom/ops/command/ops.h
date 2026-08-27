// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.
// Regenerate: python3 loom/py/loom/gen/run.py c_tables --in-place
// clang-format off

#ifndef LOOM_OPS_COMMAND_OPS_H_
#define LOOM_OPS_COMMAND_OPS_H_

#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  LOOM_OP_COMMAND_PROGRAM_DEF = LOOM_OP_KIND(LOOM_DIALECT_COMMAND, 0),
  LOOM_OP_COMMAND_PROGRAM_DECL = LOOM_OP_KIND(LOOM_DIALECT_COMMAND, 1),
  LOOM_OP_COMMAND_PROGRAM_LAUNCH = LOOM_OP_KIND(LOOM_DIALECT_COMMAND, 2),
  LOOM_OP_COMMAND_RETURN = LOOM_OP_KIND(LOOM_DIALECT_COMMAND, 3),
  LOOM_OP_COMMAND_YIELD = LOOM_OP_KIND(LOOM_DIALECT_COMMAND, 4),
  LOOM_OP_COMMAND_SERIAL = LOOM_OP_KIND(LOOM_DIALECT_COMMAND, 5),
  LOOM_OP_COMMAND_CONCURRENT = LOOM_OP_KIND(LOOM_DIALECT_COMMAND, 6),
  LOOM_OP_COMMAND_PARAMETER = LOOM_OP_KIND(LOOM_DIALECT_COMMAND, 7),
  LOOM_OP_COMMAND_COUNT_ = 8,
};

// Function visibility. Absent (0) means private (module-internal).
typedef enum loom_command_visibility_e {
  LOOM_COMMAND_VISIBILITY_PUBLIC = 1,
  LOOM_COMMAND_VISIBILITY_COUNT_ = 2,
} loom_command_visibility_t;

// Private symbol retention policy. Absent (0) permits ordinary DCE.
typedef enum loom_command_retain_e {
  LOOM_COMMAND_RETAIN_RETAIN = 1,
  LOOM_COMMAND_RETAIN_COUNT_ = 2,
} loom_command_retain_t;

// LOOM_OP_COMMAND_PROGRAM_DEF: Reusable command-program definition. Leading specialization arguments participate in staged specialization and launch-count evaluation; buffer bindings are provided when the materialized program is issued. Observable effects in the body must be explicit command operations; ordinary SSA preparation is effect-free.
// command.program.def @decode(%token_count: index) launch(%parameters: buffer, %transient: buffer) {
//   command.return
// }
LOOM_DEFINE_ISA(loom_command_program_def_isa, LOOM_OP_COMMAND_PROGRAM_DEF)
LOOM_DEFINE_ATTR_SYMBOL(loom_command_program_def_callee, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_command_program_def_visibility, 1, loom_command_visibility_t)
LOOM_DEFINE_ATTR_SYMBOL(loom_command_program_def_target, 2)
LOOM_DEFINE_ATTR_PREDICATE_LIST(loom_command_program_def_predicates, 3)
LOOM_DEFINE_ATTR_I64(loom_command_program_def_specialization_count, 4)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_command_program_def_retain, 5, loom_command_retain_t)
LOOM_DEFINE_REGION(loom_command_program_def_body, 0)
enum loom_command_program_def_build_flag_bits_e {
  LOOM_COMMAND_PROGRAM_DEF_BUILD_FLAG_HAS_VISIBILITY = 1u << 0,
  LOOM_COMMAND_PROGRAM_DEF_BUILD_FLAG_HAS_RETAIN = 1u << 1,
  LOOM_COMMAND_PROGRAM_DEF_BUILD_FLAG_HAS_TARGET = 1u << 2,
  LOOM_COMMAND_PROGRAM_DEF_BUILD_FLAG_HAS_PREDICATES = 1u << 3,
};
typedef uint32_t loom_command_program_def_build_flags_t;
iree_status_t loom_command_program_def_build(
    loom_builder_t* builder,
    loom_command_program_def_build_flags_t build_flags,
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
iree_status_t loom_command_program_def_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_COMMAND_PROGRAM_DECL: Bodyless declaration of a reusable command program.
// command.program.decl @decode(%token_count: index) launch(%parameters: buffer, %transient: buffer)
LOOM_DEFINE_ISA(loom_command_program_decl_isa, LOOM_OP_COMMAND_PROGRAM_DECL)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_command_program_decl_args, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_command_program_decl_callee, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_command_program_decl_visibility, 1, loom_command_visibility_t)
LOOM_DEFINE_ATTR_SYMBOL(loom_command_program_decl_target, 2)
LOOM_DEFINE_ATTR_PREDICATE_LIST(loom_command_program_decl_predicates, 3)
LOOM_DEFINE_ATTR_I64(loom_command_program_decl_specialization_count, 4)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_command_program_decl_retain, 5, loom_command_retain_t)
enum loom_command_program_decl_build_flag_bits_e {
  LOOM_COMMAND_PROGRAM_DECL_BUILD_FLAG_HAS_VISIBILITY = 1u << 0,
  LOOM_COMMAND_PROGRAM_DECL_BUILD_FLAG_HAS_RETAIN = 1u << 1,
  LOOM_COMMAND_PROGRAM_DECL_BUILD_FLAG_HAS_TARGET = 1u << 2,
  LOOM_COMMAND_PROGRAM_DECL_BUILD_FLAG_HAS_PREDICATES = 1u << 3,
};
typedef uint32_t loom_command_program_decl_build_flags_t;
iree_status_t loom_command_program_decl_build(
    loom_builder_t* builder,
    loom_command_program_decl_build_flags_t build_flags,
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
iree_status_t loom_command_program_decl_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_COMMAND_PROGRAM_LAUNCH: Invoke a command program with explicit specialization values and issue-time buffer bindings.
// command.program.launch @decode[%token_count](%parameters, %transient) : [index](buffer, buffer)
LOOM_DEFINE_ISA(loom_command_program_launch_isa, LOOM_OP_COMMAND_PROGRAM_LAUNCH)
LOOM_DEFINE_SEGMENTED_OPERANDS(loom_command_program_launch_specializations, 0)
LOOM_DEFINE_SEGMENTED_OPERANDS(loom_command_program_launch_bindings, 1)
LOOM_DEFINE_ATTR_SYMBOL(loom_command_program_launch_callee, 0)
iree_status_t loom_command_program_launch_build(
    loom_builder_t* builder,
    loom_symbol_ref_t callee,
    const loom_value_id_t* specializations,
    iree_host_size_t specializations_count,
    const loom_value_id_t* bindings,
    iree_host_size_t bindings_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_command_program_launch_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_COMMAND_RETURN: Terminate a command-program definition.
// command.return
LOOM_DEFINE_ISA(loom_command_return_isa, LOOM_OP_COMMAND_RETURN)
iree_status_t loom_command_return_build(
    loom_builder_t* builder,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_COMMAND_YIELD: Terminate a structured command schedule region.
// command.yield
LOOM_DEFINE_ISA(loom_command_yield_isa, LOOM_OP_COMMAND_YIELD)
iree_status_t loom_command_yield_build(
    loom_builder_t* builder,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_COMMAND_SERIAL: Order each child command after the preceding child completes.
// command.serial {
//   command.yield
// }
LOOM_DEFINE_ISA(loom_command_serial_isa, LOOM_OP_COMMAND_SERIAL)
LOOM_DEFINE_REGION(loom_command_serial_body, 0)
iree_status_t loom_command_serial_build(
    loom_builder_t* builder,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_COMMAND_CONCURRENT: Permit child commands to execute without dependency edges between siblings and join them on exit.
// command.concurrent {
//   command.yield
// }
LOOM_DEFINE_ISA(loom_command_concurrent_isa, LOOM_OP_COMMAND_CONCURRENT)
LOOM_DEFINE_REGION(loom_command_concurrent_body, 0)
iree_status_t loom_command_concurrent_build(
    loom_builder_t* builder,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_COMMAND_PARAMETER: Associate immutable named parameter content with an explicit command-program buffer root. The pattern contains one canonical decimal placeholder for each index substitution. The result is a logical typed view; this operation performs no allocation, lookup, transfer, or synchronization.
// %embedding = command.parameter %parameters, "token_embd.weight" : view<175030272xi8>
LOOM_DEFINE_ISA(loom_command_parameter_isa, LOOM_OP_COMMAND_PARAMETER)
LOOM_DEFINE_OPERAND(loom_command_parameter_source, 0)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_command_parameter_substitutions, 1)
LOOM_DEFINE_RESULT(loom_command_parameter_result, 0)
LOOM_DEFINE_ATTR_STRING(loom_command_parameter_pattern, 0)
iree_status_t loom_command_parameter_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t source,
    loom_string_id_t pattern,
    loom_may_consume const loom_value_id_t* substitutions,
    iree_host_size_t substitutions_count,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_command_parameter_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_command_parameter_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// Returns the vtable array for the command dialect.
const loom_op_vtable_t* const* loom_command_dialect_vtables(
    iree_host_size_t* out_count);

// Returns the dense semantic metadata array for the command dialect.
const loom_op_semantics_t* loom_command_dialect_op_semantics(
    iree_host_size_t* out_count);

// Returns semantic metadata for a command op kind, or empty metadata.
loom_op_semantics_t loom_command_op_semantics(
    loom_op_kind_t kind);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_COMMAND_OPS_H_
