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

// Resolves a structurally verified operand dictionary entry to its ordinal.
static inline bool loom_encoding_define_dynamic_param_ordinal(
    const loom_encoding_define_param_view_t* params,
    const loom_named_attr_t* name_entry, uint16_t* out_ordinal) {
  *out_ordinal = 0;
  if (!name_entry) return false;
  IREE_ASSERT(name_entry->value.kind == LOOM_ATTR_I64);
  IREE_ASSERT(name_entry->value.i64 >= 0);
  IREE_ASSERT(name_entry->value.i64 < params->dynamic_values.count);
  *out_ordinal = (uint16_t)name_entry->value.i64;
  return true;
}

// Resolves a structurally verified operand dictionary entry to its value.
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

// Maximum number of dynamic parameters in one family descriptor. Descriptor
// ordinals reserve UINT8_MAX as an invalid sentinel.
#define LOOM_ENCODING_DYNAMIC_PARAMETER_COUNT_MAX UINT8_MAX

// A descriptor-indexed view over one structurally valid encoding.define.
typedef struct loom_encoding_define_resolved_params_t {
  // Static encoding table entry referenced by the `spec` attribute.
  const loom_encoding_t* spec;
  // Registered family descriptor governing both parameter classes.
  const loom_encoding_family_descriptor_t* descriptor;
  // Static family parameters from `spec`, sorted by parameter name.
  loom_named_attr_slice_t static_attrs;
  // Descriptor-indexed dynamic values with invalid IDs for absent parameters.
  loom_value_slice_t dynamic_values;
} loom_encoding_define_resolved_params_t;

typedef enum loom_encoding_define_param_issue_e {
  LOOM_ENCODING_DEFINE_PARAM_ISSUE_NONE = 0,
  LOOM_ENCODING_DEFINE_PARAM_ISSUE_DUPLICATE_STATIC_DYNAMIC,
  LOOM_ENCODING_DEFINE_PARAM_ISSUE_UNKNOWN_DYNAMIC,
  LOOM_ENCODING_DEFINE_PARAM_ISSUE_DYNAMIC_TYPE_MISMATCH,
} loom_encoding_define_param_issue_t;

// Describes the first authored parameter contract violation found while
// resolving a structurally valid encoding.define.
typedef struct loom_encoding_define_param_resolution_t {
  // Issue classification, or NONE when resolution succeeded.
  loom_encoding_define_param_issue_t issue;
  // Authored dynamic parameter name associated with the issue.
  loom_string_id_t name_id;
  // Dynamic operand associated with a type mismatch, or invalid otherwise.
  loom_value_id_t value_id;
  // Expected type associated with a type mismatch, or ANY otherwise.
  loom_type_constraint_t expected_type;
} loom_encoding_define_param_resolution_t;

// Resolves sorted authored parameters against |descriptor| in linear time.
// |dynamic_value_slots| must provide descriptor->dynamic_parameter_count
// entries. The input must have passed generic OperandDict verification;
// authored family-contract violations are returned as ordinary classifications
// for the public verifier to diagnose.
loom_encoding_define_param_resolution_t loom_encoding_define_resolve_params(
    const loom_module_t* module,
    const loom_encoding_family_descriptor_t* descriptor,
    const loom_encoding_define_param_view_t* params,
    loom_value_id_t* dynamic_value_slots,
    loom_encoding_define_resolved_params_t* out_params);

// Returns the descriptor-indexed dynamic parameter value, or an invalid value
// ID when that parameter is absent.
static inline loom_value_id_t loom_encoding_define_dynamic_parameter(
    const loom_encoding_define_resolved_params_t* params,
    uint8_t descriptor_index) {
  IREE_ASSERT(descriptor_index < params->dynamic_values.count);
  return params->dynamic_values.values[descriptor_index];
}

// Returns whether the descriptor-indexed dynamic parameter is present.
static inline bool loom_encoding_define_has_dynamic_parameter(
    const loom_encoding_define_resolved_params_t* params,
    uint8_t descriptor_index) {
  return loom_encoding_define_dynamic_parameter(params, descriptor_index) !=
         LOOM_VALUE_ID_INVALID;
}

// Returns the stable name of a descriptor-indexed dynamic parameter.
static inline iree_string_view_t loom_encoding_define_dynamic_parameter_name(
    const loom_encoding_define_resolved_params_t* params,
    uint8_t descriptor_index) {
  IREE_ASSERT(descriptor_index < params->dynamic_values.count);
  return loom_bstring_view(
      params->descriptor->dynamic_parameter_descriptors[descriptor_index].name);
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
