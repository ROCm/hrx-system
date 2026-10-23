// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/program_capacity.h"

#include "loom/codegen/low/diagnostics.h"
#include "loom/error/diagnostic.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/leaf_program.h"
#include "loom/target/arch/amd/xdna/error_catalog.h"

iree_status_t loom_aie2p_program_capacity_admit(
    const loom_module_t* module, const loom_op_t* function_op,
    uint64_t required_byte_length, iree_diagnostic_emitter_t emitter,
    bool* out_admitted) {
  *out_admitted = required_byte_length <= LOOM_AIE2P_CORE_PROGRAM_MEMORY_SIZE;
  if (*out_admitted) {
    return iree_ok_status();
  }
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_function_name(module, function_op)),
      loom_param_u64(required_byte_length),
      loom_param_u64(LOOM_AIE2P_CORE_PROGRAM_MEMORY_SIZE),
  };
  const loom_diagnostic_emission_t emission = {
      .op = function_op,
      .error = LOOM_ERR_XDNA_058,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}
