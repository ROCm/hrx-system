// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.
// Regenerate: python3 loom/py/loom/gen/run.py c_tables --in-place
// clang-format off

#ifndef LOOM_OPS_CFG_OPS_H_
#define LOOM_OPS_CFG_OPS_H_

#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  LOOM_OP_CFG_BR = LOOM_OP_KIND(LOOM_DIALECT_CFG, 0),
  LOOM_OP_CFG_COND_BR = LOOM_OP_KIND(LOOM_DIALECT_CFG, 1),
  LOOM_OP_CFG_COUNT_ = 2,
};

// LOOM_OP_CFG_BR: Unconditional branch to a successor block, forwarding zero or more block argument values.
// cfg.br ^done
LOOM_DEFINE_ISA(loom_cfg_br_isa, LOOM_OP_CFG_BR)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_cfg_br_args, 0)
LOOM_DEFINE_SUCCESSOR(loom_cfg_br_dest, 0)
iree_status_t loom_cfg_br_build(
    loom_builder_t* builder,
    loom_block_t* dest,
    const loom_value_id_t* args,
    iree_host_size_t args_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_cfg_br_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_CFG_COND_BR: Conditional branch to one of two successor blocks based on an i1 condition.
// cfg.cond_br %condition, ^then, ^else
LOOM_DEFINE_ISA(loom_cfg_cond_br_isa, LOOM_OP_CFG_COND_BR)
LOOM_DEFINE_OPERAND(loom_cfg_cond_br_condition, 0)
LOOM_DEFINE_SUCCESSOR(loom_cfg_cond_br_true_dest, 0)
LOOM_DEFINE_SUCCESSOR(loom_cfg_cond_br_false_dest, 1)
iree_status_t loom_cfg_cond_br_build(
    loom_builder_t* builder,
    loom_value_id_t condition,
    loom_block_t* true_dest,
    loom_block_t* false_dest,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_cfg_cond_br_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// Returns the vtable array for the cfg dialect.
const loom_op_vtable_t* const* loom_cfg_dialect_vtables(
    iree_host_size_t* out_count);

// Returns the dense semantic metadata array for the cfg dialect.
const loom_op_semantics_t* loom_cfg_dialect_op_semantics(
    iree_host_size_t* out_count);

// Returns semantic metadata for a cfg op kind, or empty metadata.
loom_op_semantics_t loom_cfg_op_semantics(
    loom_op_kind_t kind);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_CFG_OPS_H_
