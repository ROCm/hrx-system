// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/ref/verify.h"

#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
#include "loom/ir/module.h"
#include "loom/ops/ref/ops.h"

iree_status_t loom_ref_verify_borrow_types(const loom_op_t* op,
                                           loom_type_t owner_type,
                                           loom_type_t pointer_type,
                                           iree_diagnostic_emitter_t emitter) {
  if (!loom_type_is_reference(owner_type) ||
      loom_type_reference_ownership(owner_type) !=
          LOOM_REFERENCE_OWNERSHIP_OWNED) {
    const loom_diagnostic_param_t params[] = {
        loom_param_string(IREE_SV("owner")),
        loom_param_type(owner_type),
        loom_param_string(IREE_SV("owning managed reference")),
    };
    const loom_diagnostic_emission_t emission = {
        .op = op,
        .error = LOOM_ERR_TYPE_003,
        .params = params,
        .param_count = IREE_ARRAYSIZE(params),
    };
    return iree_diagnostic_emit(emitter, &emission);
  }
  const loom_type_t expected =
      loom_type_reference(LOOM_REFERENCE_OWNERSHIP_BORROWED,
                          loom_type_reference_namespace_id(owner_type),
                          loom_type_reference_name_id(owner_type),
                          loom_type_reference_declaration_id(owner_type));
  if (loom_type_equal(pointer_type, expected)) {
    return iree_ok_status();
  }
  const loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("pointer")),
      loom_param_type(pointer_type),
      loom_param_string(IREE_SV("borrowed owner")),
      loom_param_type(expected),
  };
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_TYPE_001,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

iree_status_t loom_ref_borrow_verify(const loom_module_t* module,
                                     const loom_op_t* op,
                                     iree_diagnostic_emitter_t emitter) {
  return loom_ref_verify_borrow_types(
      op, loom_module_value_type(module, loom_ref_borrow_owner(op)),
      loom_module_value_type(module, loom_ref_borrow_pointer(op)), emitter);
}
