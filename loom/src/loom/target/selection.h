// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Allocation-free target specification parsing and profile selection.

#ifndef LOOM_TARGET_SELECTION_H_
#define LOOM_TARGET_SELECTION_H_

#include "iree/base/api.h"
#include "loom/target/provider.h"

#ifdef __cplusplus
extern "C" {
#endif

// Borrowed components of a public `family:selector` target specification.
typedef struct loom_target_specification_t {
  // Target profile family name.
  iree_string_view_t family;
  // Family-owned selector spelling.
  iree_string_view_t selector;
} loom_target_specification_t;

// Parses |value| as `family:selector` without allocating or copying it.
iree_status_t loom_target_specification_parse(
    iree_string_view_t value, loom_target_specification_t* out_specification);

// Selects a borrowed process-lifetime profile from a configured environment.
//
// Exactly one linked provider must own |specification.family| and expose named
// profile selection. The selected profile must have that provider's profile
// type and a complete target bundle.
iree_status_t loom_target_environment_select_profile(
    const loom_target_environment_t* environment,
    const loom_target_specification_t* specification,
    const loom_target_profile_t** out_profile);

// Selects a native profile eligible for |cpu_data| from linked core providers.
// |requirement| constrains automatic selection, or |profile| forces an exact
// identity; both may be NULL, but they are never supplied together. Returned
// profiles may be immutable process-lifetime values or allocator-owned values.
// The caller owns a successful selection and releases it with
// loom_target_profile_selection_release unless ownership is transferred. An
// unavailable native representation is reported before compilation or dispatch.
iree_status_t loom_target_environment_select_cpu_profile(
    const loom_target_environment_t* environment,
    const iree_cpu_data_t* cpu_data, const loom_target_facts_t* requirement,
    const loom_target_profile_t* profile, iree_allocator_t allocator,
    loom_target_profile_selection_t* out_selection);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_SELECTION_H_
