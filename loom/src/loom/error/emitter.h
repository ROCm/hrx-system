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
  iree_string_view_t label;
  const loom_op_t* op;
  loom_diagnostic_field_ref_t field_ref;
} loom_diagnostic_related_op_t;

// A single structured diagnostic emission request from a producer to the
// subsystem that owns source resolution and final sink callbacks.
typedef struct loom_diagnostic_emission_t {
  const loom_op_t* op;
  const loom_error_def_t* error;
  const loom_diagnostic_param_t* params;
  iree_host_size_t param_count;
  const loom_diagnostic_related_op_t* related_ops;
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

#ifdef __cplusplus
}
#endif

#endif  // LOOM_ERROR_EMITTER_H_
