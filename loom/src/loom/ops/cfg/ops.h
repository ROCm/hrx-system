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
  LOOM_OP_CFG_SWITCH = LOOM_OP_KIND(LOOM_DIALECT_CFG, 2),
  LOOM_OP_CFG_ASSERT = LOOM_OP_KIND(LOOM_DIALECT_CFG, 3),
  LOOM_OP_CFG_COUNT_ = 4,
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

// LOOM_OP_CFG_SWITCH: Branch to one of a sorted set of case destinations or a default destination.
// cfg.switch %selector cases [0, 2] -> [^case0, ^case2] default ^fallback
LOOM_DEFINE_ISA(loom_cfg_switch_isa, LOOM_OP_CFG_SWITCH)
LOOM_DEFINE_OPERAND(loom_cfg_switch_selector, 0)
LOOM_DEFINE_SUCCESSOR(loom_cfg_switch_default_dest, 0)
LOOM_DEFINE_VARIADIC_SUCCESSORS(loom_cfg_switch_case_dests, 1)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_cfg_switch_case_keys, 0)
iree_status_t loom_cfg_switch_build(
    loom_builder_t* builder,
    loom_value_id_t selector,
    const int64_t* case_keys,
    iree_host_size_t case_keys_count,
    loom_block_t* const* case_dests,
    iree_host_size_t case_dests_count,
    loom_block_t* default_dest,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_cfg_switch_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_CFG_ASSERT: Continue when the condition is true or fail with the diagnostic message when false.
// cfg.assert %condition, %message : buffer
LOOM_DEFINE_ISA(loom_cfg_assert_isa, LOOM_OP_CFG_ASSERT)
LOOM_DEFINE_OPERAND(loom_cfg_assert_condition, 0)
LOOM_DEFINE_OPERAND(loom_cfg_assert_message, 1)
iree_status_t loom_cfg_assert_build(
    loom_builder_t* builder,
    loom_value_id_t condition,
    loom_value_id_t message,
    loom_location_id_t location,
    loom_op_t** out_op);

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
