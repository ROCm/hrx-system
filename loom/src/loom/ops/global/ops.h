// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.
// Regenerate: python3 loom/py/loom/gen/run.py c_tables --in-place
// clang-format off

#ifndef LOOM_OPS_GLOBAL_OPS_H_
#define LOOM_OPS_GLOBAL_OPS_H_

#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  LOOM_OP_GLOBAL_CONSTANT = LOOM_OP_KIND(LOOM_DIALECT_GLOBAL, 0),
  LOOM_OP_GLOBAL_VARIABLE = LOOM_OP_KIND(LOOM_DIALECT_GLOBAL, 1),
  LOOM_OP_GLOBAL_LOAD = LOOM_OP_KIND(LOOM_DIALECT_GLOBAL, 2),
  LOOM_OP_GLOBAL_STORE = LOOM_OP_KIND(LOOM_DIALECT_GLOBAL, 3),
  LOOM_OP_GLOBAL_COUNT_ = 4,
};

// LOOM_OP_GLOBAL_CONSTANT: Immutable global value with an optional inline scalar initializer. Declaration-local dim/encoding names in the type annotation express structural constraints. Predicates constrain dynamic dimensions and are propagated to every load site as value facts. Non-scalar or computed initialization is modeled by global.store in initializer functions; resource-backed rodata should become a dedicated global-defining op instead of overloading inline attrs.
// global.constant @pi : f32 = 3.14159265358979
LOOM_DEFINE_ISA(loom_global_constant_isa, LOOM_OP_GLOBAL_CONSTANT)
LOOM_DEFINE_RESULT(loom_global_constant_type, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_global_constant_symbol, 0)
LOOM_DEFINE_ATTR_ANY(loom_global_constant_initializer, 2)
iree_status_t loom_global_constant_build(
    loom_builder_t* builder,
    loom_symbol_ref_t symbol,
    loom_type_t result_type,
    loom_optional const loom_predicate_t* predicates,
    iree_host_size_t predicates_count,
    loom_optional loom_attribute_t initializer,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_global_constant_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_GLOBAL_VARIABLE: Mutable global value with an optional inline scalar default initializer. Can be stored from any function at any time. Declaration-local dim/encoding names and predicates work the same as global.constant.
// global.variable @kv_cache : tile<[%s]x[%d]xf32> where [mul(%s, 64)]
LOOM_DEFINE_ISA(loom_global_variable_isa, LOOM_OP_GLOBAL_VARIABLE)
LOOM_DEFINE_RESULT(loom_global_variable_type, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_global_variable_symbol, 0)
LOOM_DEFINE_ATTR_ANY(loom_global_variable_initializer, 2)
iree_status_t loom_global_variable_build(
    loom_builder_t* builder,
    loom_symbol_ref_t symbol,
    loom_type_t result_type,
    loom_optional const loom_predicate_t* predicates,
    iree_host_size_t predicates_count,
    loom_optional loom_attribute_t initializer,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_global_variable_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_GLOBAL_LOAD: Load a value from a global. Dynamic dims and encodings in the type annotation reference co-results by name. Predicates on the global definition are propagated as value facts.
// %tile, %m, %k = global.load @weights : tile<[%m]x[%k]xf32>
LOOM_DEFINE_ISA(loom_global_load_isa, LOOM_OP_GLOBAL_LOAD)
LOOM_DEFINE_VARIADIC_RESULTS(loom_global_load_result, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_global_load_global, 0)
iree_status_t loom_global_load_build(
    loom_builder_t* builder,
    loom_symbol_ref_t global,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_global_load_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_global_load_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_GLOBAL_STORE: Store a value to a global. Dynamic dims and encodings are captured implicitly from the type annotation, which references existing SSA values. The verifier checks that stores to global.constant only happen from initializer-reachable code paths.
// global.store %tile, @kv_cache : tile<[%m]xf32>
LOOM_DEFINE_ISA(loom_global_store_isa, LOOM_OP_GLOBAL_STORE)
LOOM_DEFINE_OPERAND(loom_global_store_value, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_global_store_global, 0)
iree_status_t loom_global_store_build(
    loom_builder_t* builder,
    loom_value_id_t value,
    loom_symbol_ref_t global,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_global_store_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// Returns the vtable array for the global dialect.
const loom_op_vtable_t* const* loom_global_dialect_vtables(
    iree_host_size_t* out_count);

// Returns the dense semantic metadata array for the global dialect.
const loom_op_semantics_t* loom_global_dialect_op_semantics(
    iree_host_size_t* out_count);

// Returns semantic metadata for a global op kind, or empty metadata.
loom_op_semantics_t loom_global_op_semantics(
    loom_op_kind_t kind);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_GLOBAL_OPS_H_
