// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/module.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/successor_verify.h"

iree_status_t loom_cfg_br_verify(const loom_module_t* module,
                                 const loom_op_t* op,
                                 iree_diagnostic_emitter_t emitter) {
  loom_value_slice_t args = loom_cfg_br_args(op);
  return loom_ops_verify_successor_args(module, emitter, op, IREE_SV("cfg.br"),
                                        0, loom_cfg_br_dest(op), args.values,
                                        args.count);
}

iree_status_t loom_cfg_cond_br_verify(const loom_module_t* module,
                                      const loom_op_t* op,
                                      iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_ops_verify_successor_args(
      module, emitter, op, IREE_SV("cfg.cond_br"), 0,
      loom_cfg_cond_br_true_dest(op), NULL, 0));
  return loom_ops_verify_successor_args(
      module, emitter, op, IREE_SV("cfg.cond_br"), 1,
      loom_cfg_cond_br_false_dest(op), NULL, 0);
}
