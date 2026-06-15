// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.
// Regenerate: python3 loom/py/loom/gen/run.py c_tables --in-place
// clang-format off

#ifndef LOOM_OPS_CONFIG_OPS_H_
#define LOOM_OPS_CONFIG_OPS_H_

#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  LOOM_OP_CONFIG_DECL = LOOM_OP_KIND(LOOM_DIALECT_CONFIG, 0),
  LOOM_OP_CONFIG_DEF = LOOM_OP_KIND(LOOM_DIALECT_CONFIG, 1),
  LOOM_OP_CONFIG_GET = LOOM_OP_KIND(LOOM_DIALECT_CONFIG, 2),
  LOOM_OP_CONFIG_COUNT_ = 3,
};

// LOOM_OP_CONFIG_DECL: Declare a required compile/link-time configuration value. The op defines a symbol and a required type but intentionally carries no initializer. Predicates constrain the unresolved value until a config.def supplies an exact value; final executable compilation must resolve reachable config.get users to exactly one config.def.
// config.decl @model36.model.hidden_size : index
LOOM_DEFINE_ISA(loom_config_decl_isa, LOOM_OP_CONFIG_DECL)
LOOM_DEFINE_RESULT(loom_config_decl_type, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_config_decl_symbol, 0)
iree_status_t loom_config_decl_build(
    loom_builder_t* builder,
    loom_symbol_ref_t symbol,
    loom_type_t result_type,
    const loom_tied_result_t* tied_results,
    iree_host_size_t tied_result_count,
    loom_optional const loom_predicate_t* predicates,
    iree_host_size_t predicates_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_config_decl_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_CONFIG_DEF: Define a compile/link-time configuration value. The initializer is required and must match the declared result type. Scalar values seed ordinary value facts so config.get can fold through canonicalization.
// config.def @model36.model.hidden_size = 2048 : index
LOOM_DEFINE_ISA(loom_config_def_isa, LOOM_OP_CONFIG_DEF)
LOOM_DEFINE_RESULT(loom_config_def_type, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_config_def_symbol, 0)
LOOM_DEFINE_ATTR_ANY(loom_config_def_value, 1)
iree_status_t loom_config_def_build(
    loom_builder_t* builder,
    loom_symbol_ref_t symbol,
    loom_attribute_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_config_def_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_config_def_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_CONFIG_GET: Read a compile/link-time configuration value. The symbol reference is a generated symbol-ref attr so dependency analysis and bytecode index metadata can see config sensitivity through the normal symbol path.
// %hidden = config.get @model36.model.hidden_size : index
LOOM_DEFINE_ISA(loom_config_get_isa, LOOM_OP_CONFIG_GET)
LOOM_DEFINE_RESULT(loom_config_get_result, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_config_get_config, 0)
iree_status_t loom_config_get_build(
    loom_builder_t* builder,
    loom_symbol_ref_t config,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_config_get_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_config_get_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// Returns the vtable array for the config dialect.
const loom_op_vtable_t* const* loom_config_dialect_vtables(
    iree_host_size_t* out_count);

// Returns the dense semantic metadata array for the config dialect.
const loom_op_semantics_t* loom_config_dialect_op_semantics(
    iree_host_size_t* out_count);

// Returns semantic metadata for a config op kind, or empty metadata.
loom_op_semantics_t loom_config_op_semantics(
    loom_op_kind_t kind);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_CONFIG_OPS_H_
