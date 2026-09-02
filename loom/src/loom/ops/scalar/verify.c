// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/emitter.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/attribute.h"
#include "loom/ir/module.h"
#include "loom/ops/scalar/ops.h"

static iree_status_t loom_scalar_emit(iree_diagnostic_emitter_t emitter,
                                      const loom_op_t* op,
                                      const loom_error_def_t* error,
                                      const loom_diagnostic_param_t* params,
                                      iree_host_size_t param_count) {
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = error,
      .params = params,
      .param_count = param_count,
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_scalar_emit_attribute_kind_mismatch(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t attr_name, uint16_t attr_index,
    loom_attr_kind_t actual_kind, loom_attr_kind_t expected_kind) {
  loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(
          loom_param_string(attr_name),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    attr_index)),
      loom_param_u32(actual_kind),
      loom_param_u32(expected_kind),
  };
  return loom_scalar_emit(emitter, op, LOOM_ERR_TYPE_005, params,
                          IREE_ARRAYSIZE(params));
}

iree_status_t loom_scalar_geluf_verify(const loom_module_t* module,
                                       const loom_op_t* op,
                                       iree_diagnostic_emitter_t emitter) {
  loom_attribute_t scale_attr = loom_op_attrs(op)[1];
  bool has_scale = !loom_attr_is_absent(scale_attr);
  if (loom_scalar_geluf_variant(op) == LOOM_SCALAR_GELUF_VARIANT_LOGISTIC) {
    if (has_scale) return iree_ok_status();
    return loom_scalar_emit_attribute_kind_mismatch(
        emitter, op, IREE_SV("scale"), /*attr_index=*/1, LOOM_ATTR_ABSENT,
        LOOM_ATTR_F64);
  }
  if (!has_scale) return iree_ok_status();
  return loom_scalar_emit_attribute_kind_mismatch(
      emitter, op, IREE_SV("scale"), /*attr_index=*/1, scale_attr.kind,
      LOOM_ATTR_ABSENT);
}
