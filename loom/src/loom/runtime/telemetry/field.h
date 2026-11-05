// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_RUNTIME_TELEMETRY_FIELD_H_
#define LOOM_RUNTIME_TELEMETRY_FIELD_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Values exposed by consumer-side interpretation of a copied observation.
// Source-owned decoders retain enum, derived-value and availability semantics;
// the telemetry core does not catalogue or interpret concrete fields.
typedef enum loom_runtime_telemetry_field_kind_e {
  LOOM_RUNTIME_TELEMETRY_FIELD_NULL = 0,
  LOOM_RUNTIME_TELEMETRY_FIELD_UINT64,
  LOOM_RUNTIME_TELEMETRY_FIELD_INT64,
  LOOM_RUNTIME_TELEMETRY_FIELD_BOOL,
  LOOM_RUNTIME_TELEMETRY_FIELD_STRING,
  LOOM_RUNTIME_TELEMETRY_FIELD_OBJECT,
} loom_runtime_telemetry_field_kind_t;

// Borrowed native field view, valid only during the synchronous sink call.
// Strings and child fields may refer to decoder stack storage. Retained
// observations copy their original record bytes, not these expanded views.
typedef struct loom_runtime_telemetry_field_t {
  // Source-owned UTF-8 field name, independent of deployment identifiers.
  const char* name;
  // Selects the meaningful value member; NULL represents an unavailable value.
  loom_runtime_telemetry_field_kind_t kind;
  // Native value, without a serialized representation or ownership transfer.
  union {
    // Unsigned count, size, duration, identity, or source-defined scalar.
    uint64_t uint64_value;
    // Signed source-defined scalar, preserving the full signed range.
    int64_t int64_value;
    // Logical value, not an integer-shaped formatting convention.
    bool bool_value;
    // Explicit byte extent; embedded NUL and non-terminated strings are valid.
    iree_string_view_t string_value;
    // Nested named fields interpreted by the same source decoder.
    struct {
      // Number of borrowed child fields.
      iree_host_size_t count;
      // Child views, live through the enclosing sink call.
      const struct loom_runtime_telemetry_field_t* fields;
    } object_value;
  } value;
} loom_runtime_telemetry_field_t;

// Consumer of one complete source-decoded field block. Interpretation happens
// after capture and never on the publication path. Only consumer output, IO, or
// allocation can fail; trusted source schemas require no per-record validation.
typedef struct loom_runtime_telemetry_field_sink_t {
  // Consumer-owned context, borrowed through emit.
  void* self;
  // Synchronously consumes all views without retaining their borrowed storage.
  iree_status_t (*emit)(void* self, iree_host_size_t count,
                        const loom_runtime_telemetry_field_t* fields);
} loom_runtime_telemetry_field_sink_t;

static inline loom_runtime_telemetry_field_t
loom_runtime_telemetry_field_uint64(const char* name, uint64_t value) {
  return (loom_runtime_telemetry_field_t){
      .name = name,
      .kind = LOOM_RUNTIME_TELEMETRY_FIELD_UINT64,
      .value = {.uint64_value = value}};
}

static inline loom_runtime_telemetry_field_t loom_runtime_telemetry_field_int64(
    const char* name, int64_t value) {
  return (loom_runtime_telemetry_field_t){
      .name = name,
      .kind = LOOM_RUNTIME_TELEMETRY_FIELD_INT64,
      .value = {.int64_value = value}};
}

static inline loom_runtime_telemetry_field_t loom_runtime_telemetry_field_bool(
    const char* name, bool value) {
  return (loom_runtime_telemetry_field_t){
      .name = name,
      .kind = LOOM_RUNTIME_TELEMETRY_FIELD_BOOL,
      .value = {.bool_value = value}};
}

static inline loom_runtime_telemetry_field_t
loom_runtime_telemetry_field_string(const char* name,
                                    iree_string_view_t value) {
  return (loom_runtime_telemetry_field_t){
      .name = name,
      .kind = LOOM_RUNTIME_TELEMETRY_FIELD_STRING,
      .value = {.string_value = value}};
}

static inline loom_runtime_telemetry_field_t
loom_runtime_telemetry_field_object(
    const char* name, iree_host_size_t count,
    const loom_runtime_telemetry_field_t* fields) {
  return (loom_runtime_telemetry_field_t){
      .name = name,
      .kind = LOOM_RUNTIME_TELEMETRY_FIELD_OBJECT,
      .value = {.object_value = {.count = count, .fields = fields}}};
}

static inline loom_runtime_telemetry_field_t loom_runtime_telemetry_field_null(
    const char* name) {
  return (loom_runtime_telemetry_field_t){.name = name};
}

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_RUNTIME_TELEMETRY_FIELD_H_
