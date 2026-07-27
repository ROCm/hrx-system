// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Generic measurement scopes for Loom execution sessions.

#ifndef LOOM_TOOLING_EXECUTION_MEASUREMENT_H_
#define LOOM_TOOLING_EXECUTION_MEASUREMENT_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t loom_run_measurement_kind_flags_t;
enum {
  // Records no measurement data.
  LOOM_RUN_MEASUREMENT_KIND_NONE = 0u,
  // Records cheap host-side elapsed time in nanoseconds.
  LOOM_RUN_MEASUREMENT_KIND_TIMING = 1u << 0,
  // Reserves the lightweight statistics measurement kind for HAL/profile sinks.
  LOOM_RUN_MEASUREMENT_KIND_STATISTICS = 1u << 1,
  // Reserves the deep profile measurement kind for explanatory trace capture.
  LOOM_RUN_MEASUREMENT_KIND_DEEP_PROFILE = 1u << 2,
};

typedef uint32_t loom_run_measurement_boundary_t;
enum {
  // Invalid boundary sentinel.
  LOOM_RUN_MEASUREMENT_BOUNDARY_NONE = 0u,
  // Candidate compilation or executable/archive materialization.
  LOOM_RUN_MEASUREMENT_BOUNDARY_COMPILE = 1u << 0,
  // Prepared candidate creation such as module load or HAL executable prepare.
  LOOM_RUN_MEASUREMENT_BOUNDARY_PREPARE_CANDIDATE = 1u << 1,
  // Typed invocation plan materialization.
  LOOM_RUN_MEASUREMENT_BOUNDARY_PREPARE_INVOCATION = 1u << 2,
  // Submit-only execution once a backend exposes non-blocking submission.
  LOOM_RUN_MEASUREMENT_BOUNDARY_SUBMIT = 1u << 3,
  // Dispatch or VM function invocation through completion.
  LOOM_RUN_MEASUREMENT_BOUNDARY_INVOKE = 1u << 4,
  // Result transfer, formatting, comparison, or validation.
  LOOM_RUN_MEASUREMENT_BOUNDARY_COLLECT_RESULTS = 1u << 5,
  // Full user-visible one-shot execution.
  LOOM_RUN_MEASUREMENT_BOUNDARY_END_TO_END = 1u << 6,
  // Every currently defined measurement boundary.
  LOOM_RUN_MEASUREMENT_BOUNDARY_ALL =
      LOOM_RUN_MEASUREMENT_BOUNDARY_COMPILE |
      LOOM_RUN_MEASUREMENT_BOUNDARY_PREPARE_CANDIDATE |
      LOOM_RUN_MEASUREMENT_BOUNDARY_PREPARE_INVOCATION |
      LOOM_RUN_MEASUREMENT_BOUNDARY_SUBMIT |
      LOOM_RUN_MEASUREMENT_BOUNDARY_INVOKE |
      LOOM_RUN_MEASUREMENT_BOUNDARY_COLLECT_RESULTS |
      LOOM_RUN_MEASUREMENT_BOUNDARY_END_TO_END,
};

typedef struct loom_run_measurement_options_t {
  // Measurement kinds requested by the caller.
  loom_run_measurement_kind_flags_t kind_flags;
  // Boundaries that should emit samples when requested kinds are active.
  loom_run_measurement_boundary_t boundary_flags;
} loom_run_measurement_options_t;

typedef struct loom_run_measurement_sample_t {
  // Boundary measured by this sample.
  loom_run_measurement_boundary_t boundary;
  // Measurement kinds captured by this sample.
  loom_run_measurement_kind_flags_t kind_flags;
  // Host-side timestamp captured at scope entry in nanoseconds.
  iree_time_t start_time_ns;
  // Host-side elapsed time captured at scope exit in nanoseconds.
  iree_duration_t duration_ns;
  // Status code observed at scope exit.
  iree_status_code_t status_code;
} loom_run_measurement_sample_t;

typedef struct loom_run_measurement_result_t {
  // Caller-owned sample storage appended by active measurement scopes.
  loom_run_measurement_sample_t* samples;
  // Number of entries available in |samples|.
  iree_host_size_t sample_capacity;
  // Number of entries populated in |samples|.
  iree_host_size_t sample_count;
} loom_run_measurement_result_t;

typedef struct loom_run_measurement_scope_t {
  // Result receiving this scope's sample when |is_recording| is true.
  loom_run_measurement_result_t* result;
  // Index reserved in |result->samples| for this scope.
  iree_host_size_t sample_index;
  // Boundary measured by this scope.
  loom_run_measurement_boundary_t boundary;
  // Measurement kinds captured by this scope.
  loom_run_measurement_kind_flags_t kind_flags;
  // Host-side timestamp captured at scope entry in nanoseconds.
  iree_time_t start_time_ns;
  // True when this scope reserved a sample slot.
  bool is_recording;
} loom_run_measurement_scope_t;

typedef iree_status_t (*loom_run_measurement_step_fn_t)(void* user_data);

typedef struct loom_run_measurement_step_callback_t {
  // Callback invoked inside the measured boundary.
  loom_run_measurement_step_fn_t fn;
  // User data passed to |fn|.
  void* user_data;
} loom_run_measurement_step_callback_t;

// Initializes measurement options with all measurement disabled.
void loom_run_measurement_options_initialize(
    loom_run_measurement_options_t* out_options);

// Initializes result storage backed by caller-owned |samples|.
void loom_run_measurement_result_initialize(
    loom_run_measurement_sample_t* samples, iree_host_size_t sample_capacity,
    loom_run_measurement_result_t* out_result);

// Returns a stable human-readable name for |boundary|.
iree_string_view_t loom_run_measurement_boundary_name(
    loom_run_measurement_boundary_t boundary);

// Begins a measurement scope for |boundary| and reserves any required sample.
iree_status_t loom_run_measurement_scope_begin(
    const loom_run_measurement_options_t* options,
    loom_run_measurement_boundary_t boundary,
    loom_run_measurement_result_t* result,
    loom_run_measurement_scope_t* out_scope);

// Ends |scope| and records |operation_status_code| in the sample.
iree_status_t loom_run_measurement_scope_end(
    loom_run_measurement_scope_t* scope,
    iree_status_code_t operation_status_code);

// Invokes |callback| inside a measured boundary and always closes the scope.
iree_status_t loom_run_measurement_run_step(
    const loom_run_measurement_options_t* options,
    loom_run_measurement_boundary_t boundary,
    loom_run_measurement_step_callback_t callback,
    loom_run_measurement_result_t* result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_MEASUREMENT_H_
