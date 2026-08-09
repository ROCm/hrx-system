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

// One descriptor-indexed dynamic parameter binding.
typedef struct loom_encoding_define_dynamic_binding_t {
  // SSA value bound to the parameter, or invalid when absent.
  loom_value_id_t value_id;
  // Ordinal in encoding.define's authored operand list, or UINT16_MAX when
  // absent.
  uint16_t operand_ordinal;
} loom_encoding_define_dynamic_binding_t;

// A descriptor-indexed view over one structurally valid encoding.define.
typedef struct loom_encoding_define_resolved_params_t {
  // Static encoding table entry referenced by the `spec` attribute.
  const loom_encoding_t* spec;
  // Registered family descriptor governing both parameter classes.
  const loom_encoding_family_descriptor_t* descriptor;
  // Static family parameters from `spec`, sorted by parameter name.
  loom_named_attr_slice_t static_attrs;
  // Descriptor-indexed dynamic parameter bindings.
  const loom_encoding_define_dynamic_binding_t* dynamic_bindings;
  // Number of entries in |dynamic_bindings|.
  uint8_t dynamic_binding_count;
} loom_encoding_define_resolved_params_t;

// Resolves a static family instance with every dynamic parameter absent.
// |dynamic_binding_slots| must have one entry per dynamic family descriptor.
void loom_encoding_resolve_static_params(
    const loom_encoding_family_descriptor_t* descriptor,
    const loom_encoding_t* spec,
    loom_encoding_define_dynamic_binding_t* dynamic_binding_slots,
    loom_encoding_define_resolved_params_t* out_params);

// Attempts to resolve parameters from an encoding.define that may already have
// failed its own verifier. Returns false without emitting another diagnostic
// when the producer's family or parameter contract is malformed. This is for
// conservative cross-op verifier queries; verified compiler paths should use
// loom_encoding_define_resolve_verified_params instead.
bool loom_encoding_define_try_resolve_unverified_params(
    const loom_module_t* module,
    const loom_encoding_family_descriptor_t* descriptor,
    const loom_encoding_define_param_view_t* params,
    loom_encoding_define_dynamic_binding_t* dynamic_binding_slots,
    loom_encoding_define_resolved_params_t* out_params);

// Resolves parameters from a verified encoding.define. This is the infallible
// internal query for passes that run after module verification.
void loom_encoding_define_resolve_verified_params(
    const loom_module_t* module,
    const loom_encoding_family_descriptor_t* descriptor,
    const loom_encoding_define_param_view_t* params,
    loom_encoding_define_dynamic_binding_t* dynamic_binding_slots,
    loom_encoding_define_resolved_params_t* out_params);

// Returns the descriptor-indexed dynamic parameter value, or an invalid value
// ID when that parameter is absent.
static inline loom_value_id_t loom_encoding_define_dynamic_parameter(
    const loom_encoding_define_resolved_params_t* params,
    uint8_t descriptor_index) {
  IREE_ASSERT(descriptor_index < params->dynamic_binding_count);
  return params->dynamic_bindings[descriptor_index].value_id;
}

// Returns the authored operand ordinal of a descriptor-indexed dynamic
// parameter, or UINT16_MAX when that parameter is absent.
static inline uint16_t loom_encoding_define_dynamic_parameter_operand_ordinal(
    const loom_encoding_define_resolved_params_t* params,
    uint8_t descriptor_index) {
  IREE_ASSERT(descriptor_index < params->dynamic_binding_count);
  return params->dynamic_bindings[descriptor_index].operand_ordinal;
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
  IREE_ASSERT(descriptor_index < params->dynamic_binding_count);
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
