// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Pass pipeline execution reports.
//
// Reports are caller-owned snapshots of interpreter execution. The interpreter
// appends one record per pass invocation after the pass callback finishes and
// before its instance arena is torn down, copying descriptor-defined statistic
// fields and optional pass-emitted detail rows into report-owned storage.

#ifndef LOOM_PASS_REPORT_H_
#define LOOM_PASS_REPORT_H_

#include "iree/base/api.h"
#include "loom/pass/types.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_pass_descriptor_t loom_pass_descriptor_t;
typedef struct loom_pass_program_instruction_t loom_pass_program_instruction_t;
typedef struct loom_pass_registry_t loom_pass_registry_t;

typedef struct loom_pass_report_statistic_t {
  // Descriptor-defined statistic key.
  iree_string_view_t name;
  // Statistic value captured after the pass invocation.
  int64_t value;
} loom_pass_report_statistic_t;

typedef enum loom_pass_report_detail_value_kind_e {
  // Invalid or absent payload.
  LOOM_PASS_REPORT_DETAIL_VALUE_NONE = 0,
  // UTF-8 string payload.
  LOOM_PASS_REPORT_DETAIL_VALUE_STRING = 1,
  // Signed integer payload.
  LOOM_PASS_REPORT_DETAIL_VALUE_INT64 = 2,
  // Unsigned integer payload.
  LOOM_PASS_REPORT_DETAIL_VALUE_UINT64 = 3,
  // Boolean payload.
  LOOM_PASS_REPORT_DETAIL_VALUE_BOOL = 4,
} loom_pass_report_detail_value_kind_t;

typedef struct loom_pass_report_detail_field_t {
  // Stable field key.
  iree_string_view_t name;
  // Payload kind stored in the value union.
  loom_pass_report_detail_value_kind_t value_kind;
  union {
    // String payload when value_kind is STRING.
    iree_string_view_t string_value;
    // Signed integer payload when value_kind is INT64.
    int64_t int64_value;
    // Unsigned integer payload when value_kind is UINT64.
    uint64_t uint64_value;
    // Boolean payload when value_kind is BOOL.
    bool bool_value;
  };
} loom_pass_report_detail_field_t;

struct loom_pass_report_detail_t {
  // Stable detail row category.
  iree_string_view_t category;
  // Report-owned field entries in pass-defined order.
  loom_pass_report_detail_field_t* fields;
  // Number of entries in fields.
  uint16_t field_count;
  // Next pending or invocation-owned detail row.
  loom_pass_report_detail_t* next;
};

typedef struct loom_pass_report_invocation_t {
  // Descriptor selected by this pass.run invocation.
  const loom_pass_descriptor_t* descriptor;
  // Canonical pass key used by textual pipelines.
  iree_string_view_t pass_key;
  // Static pass kind declared by the descriptor metadata.
  loom_pass_kind_t pass_kind;
  // Anchor kind active for this invocation.
  loom_pass_kind_t anchor_kind;
  // Symbol name of the source pass.pipeline that produced this invocation.
  iree_string_view_t pipeline_symbol;
  // Current symbol name when the invocation runs at function anchor.
  iree_string_view_t symbol_name;
  // Program instruction index for deterministic correlation with traces.
  iree_host_size_t instruction_index;
  // Monotonic elapsed time for create/run/destroy in nanoseconds.
  iree_duration_t duration_nanoseconds;
  // True when the invocation changed IR or semantic module state.
  bool changed;
  // Terminal status code returned by this invocation.
  iree_status_code_t status_code;
  // Report-owned statistic values indexed by descriptor statistic field order.
  loom_pass_report_statistic_t* statistics;
  // Number of entries in statistics.
  uint16_t statistic_count;
  // Report-owned detail rows emitted by the invocation.
  loom_pass_report_detail_t* details;
  // Number of entries in details.
  iree_host_size_t detail_count;
} loom_pass_report_invocation_t;

struct loom_pass_report_t {
  // Invocation records in execution order.
  loom_pass_report_invocation_t* invocations;
  // Number of populated invocation records.
  iree_host_size_t invocation_count;
  // Allocated invocation record count.
  iree_host_size_t invocation_capacity;
  // Host allocator that owns report arrays.
  iree_allocator_t allocator;
};

typedef struct loom_pass_report_invocation_options_t {
  // Compiled instruction being invoked.
  const loom_pass_program_instruction_t* instruction;
  // Program instruction index for deterministic correlation with traces.
  iree_host_size_t instruction_index;
  // Anchor kind active for this invocation.
  loom_pass_kind_t anchor_kind;
  // Source pass.pipeline symbol name.
  iree_string_view_t pipeline_symbol;
  // Current anchor symbol name, or "<none>" at module anchor.
  iree_string_view_t symbol_name;
  // Monotonic elapsed time for the invocation in nanoseconds.
  iree_duration_t duration_nanoseconds;
  // True when the invocation changed IR or semantic module state.
  bool changed;
  // Terminal status code returned by this invocation.
  iree_status_code_t status_code;
  // Pass statistic storage matching the invoked descriptor layout.
  const void* statistic_storage;
  // First pending detail row emitted by this pass invocation.
  loom_pass_report_detail_t* detail_head;
  // Number of pending detail rows emitted by this pass invocation.
  iree_host_size_t detail_count;
} loom_pass_report_invocation_options_t;

// Returns true when |pass| has a report sink for detail rows.
static inline bool loom_pass_report_is_enabled(const loom_pass_t* pass) {
  return pass && pass->report;
}

// Constructs a string detail field.
static inline loom_pass_report_detail_field_t
loom_pass_report_detail_string_field(iree_string_view_t name,
                                     iree_string_view_t value) {
  loom_pass_report_detail_field_t field;
  field.name = name;
  field.value_kind = LOOM_PASS_REPORT_DETAIL_VALUE_STRING;
  field.string_value = value;
  return field;
}

// Constructs a signed integer detail field.
static inline loom_pass_report_detail_field_t
loom_pass_report_detail_int64_field(iree_string_view_t name, int64_t value) {
  loom_pass_report_detail_field_t field;
  field.name = name;
  field.value_kind = LOOM_PASS_REPORT_DETAIL_VALUE_INT64;
  field.int64_value = value;
  return field;
}

// Constructs an unsigned integer detail field.
static inline loom_pass_report_detail_field_t
loom_pass_report_detail_uint64_field(iree_string_view_t name, uint64_t value) {
  loom_pass_report_detail_field_t field;
  field.name = name;
  field.value_kind = LOOM_PASS_REPORT_DETAIL_VALUE_UINT64;
  field.uint64_value = value;
  return field;
}

// Constructs a boolean detail field.
static inline loom_pass_report_detail_field_t
loom_pass_report_detail_bool_field(iree_string_view_t name, bool value) {
  loom_pass_report_detail_field_t field;
  field.name = name;
  field.value_kind = LOOM_PASS_REPORT_DETAIL_VALUE_BOOL;
  field.bool_value = value;
  return field;
}

// Initializes an empty report.
void loom_pass_report_initialize(iree_allocator_t allocator,
                                 loom_pass_report_t* out_report);

// Releases all report-owned storage.
void loom_pass_report_deinitialize(loom_pass_report_t* report);

// Appends one pass invocation record, copying statistic values from typed
// statistic storage into report-owned storage. Descriptor metadata and symbol
// strings are borrowed from static registries or live modules and must outlive
// the report.
iree_status_t loom_pass_report_append_invocation(
    loom_pass_report_t* report,
    const loom_pass_report_invocation_options_t* options);

// Appends one structured detail row to the current pass invocation. When pass
// reporting is disabled this returns OK without allocating or copying.
iree_status_t loom_pass_report_append_detail(
    loom_pass_t* pass, iree_string_view_t category,
    const loom_pass_report_detail_field_t* fields, uint16_t field_count);

// Releases pending detail rows that were emitted by |pass| but not moved into
// an invocation record.
void loom_pass_report_discard_pass_details(loom_pass_t* pass);

// Writes the report as stable JSON for tools and tests.
iree_status_t loom_pass_report_format_json(const loom_pass_report_t* report,
                                           loom_output_stream_t* stream);

// Writes registry descriptor and option-schema metadata as stable JSON for
// pass reports and failure reproducers.
iree_status_t loom_pass_report_format_registry_json(
    const loom_pass_registry_t* registry, loom_output_stream_t* stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_PASS_REPORT_H_
