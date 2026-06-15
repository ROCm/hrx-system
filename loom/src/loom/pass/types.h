// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Core pass metadata and invocation contracts.
//
// Pass descriptors, pipeline programs, and interpreter execution all share this
// cold ABI. Concrete pass callbacks receive a loom_pass_t for their current
// invocation; ownership of the pass object, arenas, decoded options,
// statistics, diagnostics, and user data remains with the interpreter.

#ifndef LOOM_PASS_TYPES_H_
#define LOOM_PASS_TYPES_H_

#include <stddef.h>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_module_t loom_module_t;
typedef struct loom_pass_environment_t loom_pass_environment_t;
typedef struct loom_pass_decoded_options_t loom_pass_decoded_options_t;
typedef struct loom_pass_report_detail_t loom_pass_report_detail_t;
typedef struct loom_pass_report_t loom_pass_report_t;
typedef struct loom_pass_t loom_pass_t;
typedef struct loom_pass_value_fact_owner_t loom_pass_value_fact_owner_t;

typedef enum loom_pass_kind_e {
  LOOM_PASS_MODULE = 0,
  LOOM_PASS_FUNCTION = 1,
  LOOM_PASS_COUNT_,
} loom_pass_kind_t;

typedef struct loom_pass_run_result_t {
  // Number of error diagnostics emitted while running the pass program.
  uint32_t error_count;
  // Number of warning diagnostics emitted while running the pass program.
  uint32_t warning_count;
  // Number of remark diagnostics emitted while running the pass program.
  uint32_t remark_count;
} loom_pass_run_result_t;

// Describes one named option a pass accepts.
typedef struct loom_pass_option_def_t {
  // Stable option key accepted by the pass.
  iree_string_view_t name;
  // Human-readable option description.
  iree_string_view_t description;
} loom_pass_option_def_t;

enum {
  // Maximum number of statistic fields one pass descriptor may expose.
  LOOM_PASS_STATISTIC_FIELD_COUNT_MAX = 16,
};

// Describes one int64_t statistic field in a pass-owned statistics struct.
typedef struct loom_pass_statistic_field_t {
  // Stable statistic key reported by the pass.
  iree_string_view_t name;
  // Human-readable statistic description.
  iree_string_view_t description;
  // Byte offset of the int64_t field within the pass statistics struct.
  iree_host_size_t offset;
} loom_pass_statistic_field_t;

// Describes the typed statistics storage owned by one pass descriptor.
typedef struct loom_pass_statistic_layout_t {
  // Size in bytes of the pass-owned statistics struct.
  iree_host_size_t storage_size;
  // Descriptor-owned statistic fields in report order.
  const loom_pass_statistic_field_t* fields;
  // Number of entries in fields.
  uint16_t field_count;
} loom_pass_statistic_layout_t;

// Defines one statistic field for a typed pass statistics layout.
#define LOOM_PASS_STATISTIC_FIELD(statistics_type, field_name, statistic_name, \
                                  statistic_description)                       \
  {                                                                            \
      /*.name=*/IREE_SVL(statistic_name),                                      \
      /*.description=*/IREE_SVL(statistic_description),                        \
      /*.offset=*/offsetof(statistics_type, field_name),                       \
  }

// Defines a typed pass statistics layout from a static field table.
#define LOOM_PASS_STATISTIC_LAYOUT(statistics_type, field_array) \
  {                                                              \
      /*.storage_size=*/sizeof(statistics_type),                 \
      /*.fields=*/field_array,                                   \
      /*.field_count=*/IREE_ARRAYSIZE(field_array),              \
  }

// Declares one int64_t member in a pass statistics struct.
#define LOOM_PASS_STATISTIC_STORAGE_FIELD_(                             \
    statistics_type, field_name, statistic_name, statistic_description) \
  int64_t field_name;

// Declares one cold layout metadata row for a typed statistic field.
#define LOOM_PASS_STATISTIC_LAYOUT_FIELD_(                               \
    statistics_type, field_name, statistic_name, statistic_description)  \
  LOOM_PASS_STATISTIC_FIELD(statistics_type, field_name, statistic_name, \
                            statistic_description),

// Declares typed statistics storage and a pass-local accessor. Shared pass
// engines use this form in headers and define their cold layout metadata in the
// implementation file that owns the descriptor.
//
// |statistic_fields| must be a macro of the form:
//
//   #define MY_PASS_STATISTICS(V, statistics_type) \
//     V(statistics_type, field_name, "report-key", "Description.")
//
// Pass callbacks increment fields on the typed storage returned by the
// generated accessor.
#define LOOM_PASS_STATISTICS_DECLARE(accessor_name, statistics_type,      \
                                     statistic_fields)                    \
  typedef struct statistics_type {                                        \
    statistic_fields(LOOM_PASS_STATISTIC_STORAGE_FIELD_, statistics_type) \
  } statistics_type;                                                      \
  static inline statistics_type* accessor_name(loom_pass_t* pass) {       \
    return (statistics_type*)pass->statistic_storage;                     \
  }

// Defines the typed statistics storage, cold layout metadata, and accessor for
// one ordinary pass-local statistic layout. Generic report and registry code
// consume only the generated layout metadata.
#define LOOM_PASS_STATISTICS_DEFINE(accessor_name, statistics_type,          \
                                    statistic_fields)                        \
  LOOM_PASS_STATISTICS_DECLARE(accessor_name, statistics_type,               \
                               statistic_fields)                             \
  static const loom_pass_statistic_field_t accessor_name##_fields[] = {      \
      statistic_fields(LOOM_PASS_STATISTIC_LAYOUT_FIELD_, statistics_type)}; \
  static const loom_pass_statistic_layout_t accessor_name##_layout =         \
      LOOM_PASS_STATISTIC_LAYOUT(statistics_type, accessor_name##_fields);

// Static metadata for one pass kind, shared across all invocations.
typedef struct loom_pass_info_t {
  // Canonical pass name matching the descriptor key.
  iree_string_view_t name;
  // Human-readable pass description.
  iree_string_view_t description;
  // Anchor kind accepted by the pass callback.
  loom_pass_kind_t kind;
  // Descriptor-owned option definitions sorted by name.
  const loom_pass_option_def_t* option_defs;
  // Number of entries in option_defs.
  uint16_t option_count;
  // Descriptor-owned typed statistic layout, or NULL when the pass has none.
  const loom_pass_statistic_layout_t* statistic_layout;
} loom_pass_info_t;

typedef iree_status_t (*loom_module_pass_fn_t)(loom_pass_t* pass,
                                               loom_module_t* module);
typedef iree_status_t (*loom_function_pass_fn_t)(loom_pass_t* pass,
                                                 loom_module_t* module,
                                                 loom_func_like_t function);
typedef iree_status_t (*loom_pass_create_fn_t)(loom_pass_t* pass,
                                               iree_string_view_t options);
typedef void (*loom_pass_destroy_fn_t)(loom_pass_t* pass);

// One pass callback invocation owned by the interpreter.
struct loom_pass_t {
  // Static pass metadata returned by the resolved descriptor.
  const loom_pass_info_t* info;
  // Callback selected for the resolved pass kind.
  union {
    // Module pass callback for LOOM_PASS_MODULE descriptors.
    loom_module_pass_fn_t module_run;
    // Function pass callback for LOOM_PASS_FUNCTION descriptors.
    loom_function_pass_fn_t function_run;
  };
  // Stable per-invocation arena for create()/destroy() state and statistics.
  iree_arena_allocator_t* instance_arena;
  // Scratch arena for the current run callback.
  iree_arena_allocator_t* arena;
  // Per-invocation typed statistic storage matching info->statistic_layout.
  void* statistic_storage;
  // Optional pass report collecting invocation records, statistics, and rows.
  loom_pass_report_t* report;
  // First pending detail row emitted by this pass invocation.
  loom_pass_report_detail_t* report_detail_head;
  // Last pending detail row emitted by this pass invocation.
  loom_pass_report_detail_t* report_detail_tail;
  // Number of pending detail rows emitted by this pass invocation.
  iree_host_size_t report_detail_count;
  // Per-invocation state produced by create() and consumed by destroy().
  void* state;
  // True when the pass callback explicitly records an IR or semantic change.
  bool changed;
  // Immutable descriptor-decoded options for pass.run-backed invocation.
  const loom_pass_decoded_options_t* decoded_options;
  // Optional structured diagnostic emitter for pass-specific failures.
  iree_diagnostic_emitter_t diagnostic_emitter;
  // Error diagnostics emitted during this pass callback invocation.
  uint32_t error_diagnostic_count;
  // Warning diagnostics emitted during this pass callback invocation.
  uint32_t warning_diagnostic_count;
  // Remark diagnostics emitted during this pass callback invocation.
  uint32_t remark_diagnostic_count;
  // Caller-owned execution environment capabilities.
  const loom_pass_environment_t* environment;
  // Interpreter-owned scoped value-fact workspace for this module execution.
  loom_pass_value_fact_owner_t* value_facts;
};

// Records that the current pass invocation changed IR or semantic module state.
static inline void loom_pass_mark_changed(loom_pass_t* pass) {
  pass->changed = true;
}

// Returns true when the current callback has emitted a terminal error
// diagnostic. Passes use this to stop local work without converting the
// diagnostic into an iree_status_t failure.
static inline bool loom_pass_has_error_diagnostics(const loom_pass_t* pass) {
  return pass->error_diagnostic_count != 0;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_PASS_TYPES_H_
