// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Lightweight structured diagnostic emission callback.
//
// This is the narrow interface between producers that know "which op
// failed with which structured error" and the subsystem that owns the
// final diagnostic sink. Unlike loom_diagnostic_sink_t, this callback
// operates before a loom_diagnostic_t has been materialized.

#ifndef LOOM_ERROR_EMITTER_H_
#define LOOM_ERROR_EMITTER_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_op_t loom_op_t;
typedef struct loom_module_t loom_module_t;
typedef struct loom_error_def_t loom_error_def_t;
typedef struct loom_diagnostic_param_t loom_diagnostic_param_t;

// Structured reference to an op field associated with a diagnostic parameter or
// related-op note. This lives in the lightweight emitter layer so IR headers
// can carry emitter callbacks without depending on the generated error catalog.
enum loom_diagnostic_field_kind_e {
  LOOM_DIAGNOSTIC_FIELD_NONE = 0,
  LOOM_DIAGNOSTIC_FIELD_OPERAND = 1,
  LOOM_DIAGNOSTIC_FIELD_RESULT = 2,
  LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE = 3,
  LOOM_DIAGNOSTIC_FIELD_REGION = 4,
  LOOM_DIAGNOSTIC_FIELD_SUCCESSOR = 5,
};
// Raw diagnostic field-kind storage. Diagnostic sinks intentionally validate
// and render unknown field kinds, so storage cannot use a C enum type.
typedef uint8_t loom_diagnostic_field_kind_t;

// A zero-based |occurrence| disambiguates repeated source spellings of the
// same logical field, such as an operand that appears once in the callee
// operand list and again in a tied-result clause. |occurrence| is counted in
// source emission order among spans with the same (kind, index) pair.
typedef struct loom_diagnostic_field_ref_t {
  loom_diagnostic_field_kind_t kind;
  uint16_t index;
  uint16_t occurrence;
} loom_diagnostic_field_ref_t;

static inline loom_diagnostic_field_ref_t loom_diagnostic_field_ref_none(void) {
  return (loom_diagnostic_field_ref_t){
      /*.kind=*/LOOM_DIAGNOSTIC_FIELD_NONE,
      /*.index=*/0,
      /*.occurrence=*/0,
  };
}

static inline loom_diagnostic_field_ref_t
loom_diagnostic_field_ref_with_occurrence(loom_diagnostic_field_kind_t kind,
                                          uint16_t index, uint16_t occurrence) {
  return (loom_diagnostic_field_ref_t){
      /*.kind=*/kind,
      /*.index=*/index,
      /*.occurrence=*/occurrence,
  };
}

static inline loom_diagnostic_field_ref_t loom_diagnostic_field_ref(
    loom_diagnostic_field_kind_t kind, uint16_t index) {
  return loom_diagnostic_field_ref_with_occurrence(kind, index,
                                                   /*occurrence=*/0);
}

static inline bool loom_diagnostic_field_ref_is_set(
    loom_diagnostic_field_ref_t field_ref) {
  return field_ref.kind != LOOM_DIAGNOSTIC_FIELD_NONE;
}

// A secondary op location to attach to a structured diagnostic. The caller
// identifies the related op and note label; the owning subsystem resolves the
// op to a source range when materializing the final loom_diagnostic_t. An
// optional field_ref metadata lets the note highlight a specific field
// occurrence on the related op without formatting that structure into prose.
typedef struct loom_diagnostic_related_op_t {
  // Human-readable relationship rendered with the secondary location.
  iree_string_view_t label;
  // Module containing |op| when it differs from the emitter's active module.
  const loom_module_t* module;
  // Secondary operation associated with the diagnostic.
  const loom_op_t* op;
  // Optional source field to highlight on |op|.
  loom_diagnostic_field_ref_t field_ref;
} loom_diagnostic_related_op_t;

// A single structured diagnostic emission request from a producer to the
// subsystem that owns source resolution and final sink callbacks.
typedef struct loom_diagnostic_emission_t {
  // Module owning |op| and parameter type identities, or NULL to use the
  // emitter's active module. Borrowed for the emission callback.
  const loom_module_t* module;
  // Primary operation associated with the diagnostic.
  const loom_op_t* op;
  // Stable error definition selecting the diagnostic message contract.
  const loom_error_def_t* error;
  // Error parameters in definition order.
  const loom_diagnostic_param_t* params;
  // Number of entries in |params|.
  iree_host_size_t param_count;
  // Stable name of the export whose compilation emitted this diagnostic, or
  // empty when the emission is not scoped to an export. Borrowed for the
  // emission callback.
  iree_string_view_t export_name;
  // Secondary operation locations associated with the diagnostic.
  const loom_diagnostic_related_op_t* related_ops;
  // Number of entries in |related_ops|.
  iree_host_size_t related_op_count;
} loom_diagnostic_emission_t;

// Structured diagnostic emission callback used by verifier hooks and
// other subsystems that want their caller to build the final diagnostic.
typedef iree_status_t (*iree_diagnostic_emitter_fn_t)(
    void* user_data, const loom_diagnostic_emission_t* emission);

// Callback object pairing the emission function with caller-owned state.
typedef struct iree_diagnostic_emitter_t {
  iree_diagnostic_emitter_fn_t fn;
  void* user_data;
} iree_diagnostic_emitter_t;

// Emits a structured diagnostic through |emitter|. A NULL callback is
// treated as a no-op so optional emitters do not require special cases.
static inline iree_status_t iree_diagnostic_emit(
    iree_diagnostic_emitter_t emitter,
    const loom_diagnostic_emission_t* emission) {
  if (emitter.fn) {
    return emitter.fn(emitter.user_data, emission);
  }
  return iree_ok_status();
}

// Stack- or arena-owned context that supplies one export identity to every
// otherwise-unscoped diagnostic forwarded through it. Producer-supplied export
// identity takes precedence. The scope must outlive all calls to the emitter
// returned by loom_diagnostic_export_scope_emitter.
typedef struct loom_diagnostic_export_scope_t {
  // Stable export identity borrowed for the scope lifetime.
  iree_string_view_t export_name;
  // Downstream emitter receiving contextualized emissions.
  iree_diagnostic_emitter_t target;
} loom_diagnostic_export_scope_t;

// Private forwarding callback used by loom_diagnostic_export_scope_emitter.
// Its address is behavior, not identity: callers never compare callbacks.
static inline iree_status_t loom_diagnostic_export_scope_forward(
    void* user_data, const loom_diagnostic_emission_t* emission) {
  const loom_diagnostic_export_scope_t* scope =
      (const loom_diagnostic_export_scope_t*)user_data;
  if (!iree_string_view_is_empty(emission->export_name)) {
    return iree_diagnostic_emit(scope->target, emission);
  }
  loom_diagnostic_emission_t contextual_emission = *emission;
  contextual_emission.export_name = scope->export_name;
  return iree_diagnostic_emit(scope->target, &contextual_emission);
}

// Initializes an export diagnostic scope forwarding to |target|.
static inline void loom_diagnostic_export_scope_initialize(
    iree_string_view_t export_name, iree_diagnostic_emitter_t target,
    loom_diagnostic_export_scope_t* out_scope) {
  *out_scope = (loom_diagnostic_export_scope_t){
      .export_name = export_name,
      .target = target,
  };
}

// Returns an emitter borrowing |scope|.
static inline iree_diagnostic_emitter_t loom_diagnostic_export_scope_emitter(
    loom_diagnostic_export_scope_t* scope) {
  return (iree_diagnostic_emitter_t){
      .fn = loom_diagnostic_export_scope_forward,
      .user_data = scope,
  };
}

#ifdef __cplusplus
}
#endif

#endif  // LOOM_ERROR_EMITTER_H_
