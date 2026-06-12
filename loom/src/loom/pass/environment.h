// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Pass execution environment capabilities.
//
// Pass environments expose caller-owned, typed compiler capabilities to pass
// verification, program compilation, and invocation. They are deliberately not
// target selection state: heterogeneous modules still decide target context
// from IR facts on each symbol. Environment capabilities answer whether the
// compiler binary/session has linked the machinery a pass requires, such as
// target-low descriptor registries or source-to-low lowering policies.

#ifndef LOOM_PASS_ENVIRONMENT_H_
#define LOOM_PASS_ENVIRONMENT_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_pass_environment_capability_t
    loom_pass_environment_capability_t;
typedef struct loom_pass_environment_capability_type_t
    loom_pass_environment_capability_type_t;

typedef bool (*loom_pass_environment_capability_satisfies_requirement_fn_t)(
    const loom_pass_environment_capability_t* capability,
    iree_string_view_t requirement);

// Static type descriptor for one pass environment capability kind.
struct loom_pass_environment_capability_type_t {
  // Stable diagnostic name for the capability kind.
  iree_string_view_t name;
  // Optional requirement predicate implemented by this capability kind.
  loom_pass_environment_capability_satisfies_requirement_fn_t
      satisfies_requirement;
};

// Header embedded as the first field of concrete capability structs.
struct loom_pass_environment_capability_t {
  // Static type descriptor for this capability.
  const loom_pass_environment_capability_type_t* type;
};

typedef struct loom_pass_environment_t {
  // Borrowed capability pointers available to pass verification and execution.
  const loom_pass_environment_capability_t* const* capabilities;
  // Number of capability pointers in |capabilities|.
  iree_host_size_t capability_count;
} loom_pass_environment_t;

// Creates a borrowed pass environment view.
static inline loom_pass_environment_t loom_pass_environment_make(
    const loom_pass_environment_capability_t* const* capabilities,
    iree_host_size_t capability_count) {
  return (loom_pass_environment_t){
      /*.capabilities=*/capabilities,
      /*.capability_count=*/capability_count,
  };
}

// Returns an empty pass environment.
static inline loom_pass_environment_t loom_pass_environment_empty(void) {
  return loom_pass_environment_make(NULL, 0);
}

// Returns true when |environment| has no capabilities.
static inline bool loom_pass_environment_is_empty(
    const loom_pass_environment_t* environment) {
  return environment->capability_count == 0;
}

// Verifies the capability list shape and rejects duplicate capability types.
iree_status_t loom_pass_environment_verify(
    const loom_pass_environment_t* environment);

// Looks up a capability by static type pointer. Returns NULL when absent.
const loom_pass_environment_capability_t* loom_pass_environment_lookup(
    const loom_pass_environment_t* environment,
    const loom_pass_environment_capability_type_t* type);

// Returns true when |capability| satisfies |requirement|.
bool loom_pass_environment_capability_satisfies_requirement(
    const loom_pass_environment_capability_t* capability,
    iree_string_view_t requirement);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_PASS_ENVIRONMENT_H_
