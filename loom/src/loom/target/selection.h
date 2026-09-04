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

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_SELECTION_H_
