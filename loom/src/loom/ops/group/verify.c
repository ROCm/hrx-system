// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/emitter.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/module.h"
#include "loom/ops/group/ops.h"

static bool loom_group_type_isa(const loom_module_t* module, loom_type_t type) {
  if (!loom_type_is_dialect(type) || loom_type_dialect_param_count(type) != 0) {
    return false;
  }
  const loom_string_id_t name_id = loom_type_dialect_name_id(type);
  return name_id != LOOM_STRING_ID_INVALID && name_id < module->strings.count &&
         iree_string_view_equal(module->strings.entries[name_id],
                                IREE_SV("group"));
}

iree_status_t loom_group_create_verify(const loom_module_t* module,
                                       const loom_op_t* op,
                                       iree_diagnostic_emitter_t emitter) {
  const loom_type_t result_type =
      loom_module_value_type(module, loom_group_create_result(op));
  if (loom_group_type_isa(module, result_type)) return iree_ok_status();
  const loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("result")),
      loom_param_type(result_type),
      loom_param_string(IREE_SV("group")),
  };
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_TYPE_004,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}
