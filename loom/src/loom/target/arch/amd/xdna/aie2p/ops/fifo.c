// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/ops/fifo.h"

#include "loom/error/emitter.h"
#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
#include "loom/ir/module.h"
#include "loom/target/arch/amd/xdna/aie2p/ops/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/ops/source_types.h"

static iree_status_t loom_aie2p_emit_load_fifo_result_type(
    const loom_op_t* op, iree_diagnostic_emitter_t emitter,
    iree_string_view_t result_name, loom_type_t actual_type) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(result_name),
      loom_param_type(actual_type),
      loom_param_string(IREE_SV("aie2p.load_fifo")),
  };
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_TYPE_004,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

iree_status_t loom_aie2p_load_fifo_state_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  const loom_type_t fifo0_type =
      loom_module_value_type(module, loom_aie2p_load_fifo_state_fifo0(op));
  if (loom_aie2p_source_type_kind(module, fifo0_type) !=
      LOOM_AIE2P_SOURCE_TYPE_LOAD_FIFO) {
    return loom_aie2p_emit_load_fifo_result_type(op, emitter, IREE_SV("fifo0"),
                                                 fifo0_type);
  }
  const loom_type_t fifo1_type =
      loom_module_value_type(module, loom_aie2p_load_fifo_state_fifo1(op));
  if (loom_aie2p_source_type_kind(module, fifo1_type) !=
      LOOM_AIE2P_SOURCE_TYPE_LOAD_FIFO) {
    return loom_aie2p_emit_load_fifo_result_type(op, emitter, IREE_SV("fifo1"),
                                                 fifo1_type);
  }
  return iree_ok_status();
}
