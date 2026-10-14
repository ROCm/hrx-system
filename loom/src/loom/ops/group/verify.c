// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/emitter.h"
#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
#include "loom/ir/module.h"
#include "loom/ops/group/ops.h"

iree_status_t loom_group_create_verify(const loom_module_t* module,
                                       const loom_op_t* op,
                                       iree_diagnostic_emitter_t emitter) {
  const loom_type_t result_type =
      loom_module_value_type(module, loom_group_create_result(op));
  const loom_value_id_t cardinality = loom_group_create_cardinality(op);
  if (loom_type_is_group(result_type) && loom_type_rank(result_type) == 1 &&
      loom_type_dim_is_dynamic_at(result_type, 0) &&
      loom_type_dim_value_id_at(result_type, 0) == cardinality) {
    return iree_ok_status();
  }
  const loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("result")),
      loom_param_type(result_type),
      loom_param_string(IREE_SV("group shaped by the cardinality operand")),
  };
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_TYPE_004,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}
