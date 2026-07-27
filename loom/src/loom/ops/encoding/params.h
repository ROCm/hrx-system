// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Helpers for viewing encoding.define static and dynamic parameters together.

#ifndef LOOM_OPS_ENCODING_PARAMS_H_
#define LOOM_OPS_ENCODING_PARAMS_H_

#include "loom/ir/module.h"
#include "loom/ops/encoding/ops.h"

#ifdef __cplusplus
extern "C" {
#endif

struct loom_encoding_define_param_view_t {
  // Static encoding table entry referenced by the `spec` attribute.
  const loom_encoding_t* spec;

  // Static family parameters from `spec`, sorted by parameter name.
  loom_named_attr_slice_t static_attrs;

  // Dynamic parameter operand values from encoding.define.
  loom_value_slice_t dynamic_values;

  // Dynamic parameter names, sorted by name and mapped to dynamic_values
  // ordinals with i64 attributes.
  loom_named_attr_slice_t dynamic_names;
};

// Resolves a dynamic parameter name entry to the operand it names. False means
// the entry is absent or the operand dictionary is structurally invalid.
// Semantic verifiers still own missing-entry diagnostics; the structural
// verifier owns malformed dictionary diagnostics.
static inline bool loom_encoding_define_dynamic_param_ordinal(
    const loom_encoding_define_param_view_t* params,
    const loom_named_attr_t* name_entry, uint16_t* out_ordinal) {
  *out_ordinal = 0;
  if (!name_entry || name_entry->value.kind != LOOM_ATTR_I64) return false;
  int64_t ordinal = name_entry->value.i64;
  if (ordinal < 0 || ordinal >= params->dynamic_values.count) return false;
  *out_ordinal = (uint16_t)ordinal;
  return true;
}

static inline bool loom_encoding_define_dynamic_param_value(
    const loom_encoding_define_param_view_t* params,
    const loom_named_attr_t* name_entry, loom_value_id_t* out_value) {
  uint16_t ordinal = 0;
  *out_value = LOOM_VALUE_ID_INVALID;
  if (!loom_encoding_define_dynamic_param_ordinal(params, name_entry,
                                                  &ordinal)) {
    return false;
  }
  *out_value = params->dynamic_values.values[ordinal];
  return true;
}

static inline loom_encoding_define_param_view_t loom_encoding_define_param_view(
    const loom_module_t* module, const loom_op_t* op) {
  const loom_encoding_t* spec =
      loom_module_encoding(module, loom_encoding_define_spec(op));
  loom_encoding_define_param_view_t view = {
      /*.spec=*/spec,
      /*.static_attrs=*/{},
      /*.dynamic_values=*/loom_encoding_define_params(op),
      /*.dynamic_names=*/loom_encoding_define_param_names(op),
  };
  if (spec) {
    view.static_attrs = loom_encoding_attrs(spec);
  }
  return view;
}

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_ENCODING_PARAMS_H_
