// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Runtime values carried by structured diagnostics. Catalog definitions own
// parameter schemas; producers supply values and optional source-field facts.

#ifndef LOOM_ERROR_PARAMETER_H_
#define LOOM_ERROR_PARAMETER_H_

#include "iree/base/api.h"
#include "loom/error/emitter.h"
#include "loom/error/error_defs.h"
#include "loom/ir/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_diagnostic_string_list_t {
  // Values in stable display and serialization order.
  const iree_string_view_t* values;
  // Number of values in the list.
  iree_host_size_t count;
} loom_diagnostic_string_list_t;

// A runtime parameter value matching an error definition's parameter entry.
// The kind is redundant with the def but avoids chasing the def pointer
// during rendering.
typedef struct loom_diagnostic_param_t {
  loom_param_kind_t kind;  // Active parameter value variant.
  // Optional structured metadata describing the op field this param names.
  loom_diagnostic_field_ref_t field_ref;
  union {
    iree_string_view_t string;                  // LOOM_PARAM_STRING payload.
    int64_t i64;                                // LOOM_PARAM_I64 payload.
    uint32_t u32;                               // LOOM_PARAM_U32 payload.
    bool boolean;                               // LOOM_PARAM_BOOL payload.
    loom_type_t type;                           // LOOM_PARAM_TYPE payload.
    uint64_t u64;                               // LOOM_PARAM_U64 payload.
    loom_diagnostic_string_list_t string_list;  // LOOM_PARAM_STRING_LIST.
  };
} loom_diagnostic_param_t;

// Inline constructors for diagnostic params. Portable across C and C++
// without relying on designated initializers for anonymous union members.

static inline loom_diagnostic_param_t loom_param_string(
    iree_string_view_t value) {
  loom_diagnostic_param_t param;
  memset(&param, 0, sizeof(param));
  param.kind = LOOM_PARAM_STRING;
  param.string = value;
  return param;
}

static inline loom_diagnostic_param_t loom_param_i64(int64_t value) {
  loom_diagnostic_param_t param;
  memset(&param, 0, sizeof(param));
  param.kind = LOOM_PARAM_I64;
  param.i64 = value;
  return param;
}

static inline loom_diagnostic_param_t loom_param_u32(uint32_t value) {
  loom_diagnostic_param_t param;
  memset(&param, 0, sizeof(param));
  param.kind = LOOM_PARAM_U32;
  param.u32 = value;
  return param;
}

static inline loom_diagnostic_param_t loom_param_u64(uint64_t value) {
  loom_diagnostic_param_t param;
  memset(&param, 0, sizeof(param));
  param.kind = LOOM_PARAM_U64;
  param.u64 = value;
  return param;
}

static inline loom_diagnostic_param_t loom_param_string_list(
    const iree_string_view_t* values, iree_host_size_t count) {
  loom_diagnostic_param_t param;
  memset(&param, 0, sizeof(param));
  param.kind = LOOM_PARAM_STRING_LIST;
  param.string_list.values = values;
  param.string_list.count = count;
  return param;
}

static inline loom_diagnostic_param_t loom_param_bool(bool value) {
  loom_diagnostic_param_t param;
  memset(&param, 0, sizeof(param));
  param.kind = LOOM_PARAM_BOOL;
  param.boolean = value;
  return param;
}

static inline loom_diagnostic_param_t loom_param_type(loom_type_t value) {
  loom_diagnostic_param_t param;
  memset(&param, 0, sizeof(param));
  param.kind = LOOM_PARAM_TYPE;
  param.type = value;
  return param;
}

// Attaches structured field-ref metadata to an existing diagnostic parameter.
// The parameter's rendered value remains unchanged; sinks can consume the
// metadata for highlighting and machine-oriented JSON metadata.
static inline loom_diagnostic_param_t loom_param_with_field_ref(
    loom_diagnostic_param_t param, loom_diagnostic_field_ref_t field_ref) {
  param.field_ref = field_ref;
  return param;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ERROR_PARAMETER_H_
