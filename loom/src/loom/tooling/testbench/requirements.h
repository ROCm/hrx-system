// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Provider-backed check.requires/check.skip_if evaluation.
//
// This layer is target-free. It interprets check requirement ops and delegates
// predicate answers to providers injected by the caller.

#ifndef LOOM_TOOLING_TESTBENCH_REQUIREMENTS_H_
#define LOOM_TOOLING_TESTBENCH_REQUIREMENTS_H_

#include "iree/base/api.h"
#include "loom/tooling/testbench/testbench.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_testbench_requirement_provider_t
    loom_testbench_requirement_provider_t;

typedef enum loom_testbench_requirement_op_kind_e {
  // No requirement op is associated with the result.
  LOOM_TESTBENCH_REQUIREMENT_OP_KIND_NONE = 0,
  // A check.requires predicate produced the result.
  LOOM_TESTBENCH_REQUIREMENT_OP_KIND_REQUIRES = 1,
  // A check.skip_if predicate produced the result.
  LOOM_TESTBENCH_REQUIREMENT_OP_KIND_SKIP_IF = 2,
} loom_testbench_requirement_op_kind_t;

typedef enum loom_testbench_requirement_skip_code_e {
  // No skipped-case code is associated with the result.
  LOOM_TESTBENCH_REQUIREMENT_SKIP_CODE_NONE = 0,
  // The case referenced a provider that was not registered.
  LOOM_TESTBENCH_REQUIREMENT_SKIP_CODE_PROVIDER_NOT_REGISTERED = 1,
  // The registered provider cannot answer for the current environment.
  LOOM_TESTBENCH_REQUIREMENT_SKIP_CODE_PROVIDER_UNAVAILABLE = 2,
  // A check.requires predicate did not hold.
  LOOM_TESTBENCH_REQUIREMENT_SKIP_CODE_REQUIREMENT_NOT_SATISFIED = 3,
  // A check.skip_if predicate held.
  LOOM_TESTBENCH_REQUIREMENT_SKIP_CODE_SKIP_PREDICATE_MATCHED = 4,
} loom_testbench_requirement_skip_code_t;

typedef enum loom_testbench_requirement_provider_state_e {
  // Provider did not set a predicate state.
  LOOM_TESTBENCH_REQUIREMENT_PROVIDER_STATE_UNSPECIFIED = 0,
  // Provider predicate holds for the current module and attributes.
  LOOM_TESTBENCH_REQUIREMENT_PROVIDER_STATE_SATISFIED = 1,
  // Provider predicate does not hold for the current module and attributes.
  LOOM_TESTBENCH_REQUIREMENT_PROVIDER_STATE_UNSATISFIED = 2,
  // Provider cannot answer for the current runtime or target environment.
  LOOM_TESTBENCH_REQUIREMENT_PROVIDER_STATE_UNAVAILABLE = 3,
  // Referenced provider was not registered in the evaluator.
  LOOM_TESTBENCH_REQUIREMENT_PROVIDER_STATE_NOT_REGISTERED = 4,
} loom_testbench_requirement_provider_state_t;

typedef struct loom_testbench_requirement_provider_result_t {
  // Predicate state returned by the provider.
  loom_testbench_requirement_provider_state_t state;
  // Stable provider-specific predicate code, or empty if generic.
  iree_string_view_t provider_code;
  // Optional human-facing text for terminal display or manual inspection.
  iree_string_view_t display_message;
} loom_testbench_requirement_provider_result_t;

// Evaluates one provider predicate.
//
// |attrs| is the check.requires/check.skip_if attribute dictionary. Providers
// return a stable predicate |out_result|. Returning a non-OK status is reserved
// for malformed attributes or provider infrastructure failures.
typedef iree_status_t (*loom_testbench_requirement_query_fn_t)(
    void* user_data, const loom_module_t* module, loom_named_attr_slice_t attrs,
    loom_testbench_requirement_provider_result_t* out_result);

struct loom_testbench_requirement_provider_t {
  // Provider name referenced by check.requires/check.skip_if.
  iree_string_view_t name;
  // Opaque provider state passed to |query|.
  void* user_data;
  // Predicate evaluator for this provider.
  loom_testbench_requirement_query_fn_t query;
};

typedef struct loom_testbench_requirement_provider_registry_t {
  // Borrowed provider entries visible to requirement evaluation.
  const loom_testbench_requirement_provider_t* providers;
  // Number of entries in |providers|.
  iree_host_size_t provider_count;
} loom_testbench_requirement_provider_registry_t;

typedef struct loom_testbench_requirement_result_t {
  // True when the selected case should be skipped.
  bool skipped;
  // Requirement op that caused the skip, or NULL when not skipped.
  const loom_op_t* op;
  // Requirement op kind that caused the skip.
  loom_testbench_requirement_op_kind_t op_kind;
  // Stable testbench skip code for machine consumers.
  loom_testbench_requirement_skip_code_t code;
  // Provider name that caused the skip.
  iree_string_view_t provider;
  // Stable provider-specific predicate code, or empty if generic.
  iree_string_view_t provider_code;
  // Optional human-facing text for terminal display or manual inspection.
  iree_string_view_t display_message;
} loom_testbench_requirement_result_t;

// Returns the stable report string for a requirement op kind.
iree_string_view_t loom_testbench_requirement_op_kind_name(
    loom_testbench_requirement_op_kind_t op_kind);

// Returns the stable report string for a skipped-case code.
iree_string_view_t loom_testbench_requirement_skip_code_name(
    loom_testbench_requirement_skip_code_t code);

// Initializes a borrowed provider registry.
void loom_testbench_requirement_provider_registry_initialize(
    const loom_testbench_requirement_provider_t* providers,
    iree_host_size_t provider_count,
    loom_testbench_requirement_provider_registry_t* out_registry);

// Finds a named attribute in |attrs| by decoded string name.
const loom_named_attr_t* loom_testbench_requirement_find_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name);

// Reads a required string attribute from |attrs|.
iree_status_t loom_testbench_requirement_read_string_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name, iree_string_view_t* out_value);

// Reads an optional i64 attribute from |attrs|.
iree_status_t loom_testbench_requirement_read_optional_i64_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name, bool* out_present, int64_t* out_value);

// Evaluates requirement ops in source order for |case_plan|.
iree_status_t loom_testbench_evaluate_case_requirements(
    const loom_module_t* module, const loom_testbench_case_plan_t* case_plan,
    const loom_testbench_requirement_provider_registry_t* registry,
    loom_testbench_requirement_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_TESTBENCH_REQUIREMENTS_H_
