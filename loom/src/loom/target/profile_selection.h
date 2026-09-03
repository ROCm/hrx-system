// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Family-qualified target profile selection.

#ifndef LOOM_TARGET_PROFILE_SELECTION_H_
#define LOOM_TARGET_PROFILE_SELECTION_H_

#include "iree/base/api.h"
#include "loom/target/profile.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_target_environment_t loom_target_environment_t;
typedef struct loom_target_provider_t loom_target_provider_t;

// Borrowed family-qualified target specification.
typedef struct loom_target_profile_specification_t {
  // Configured target family name.
  iree_string_view_t family;

  // Complete family-owned selector following the first colon.
  iree_string_view_t selector;
} loom_target_profile_specification_t;

// Owned target profile selected by one target-family provider.
typedef struct loom_target_profile_selection_t {
  // Target provider owning the selected profile representation.
  const loom_target_provider_t* provider;

  // Family-owned immutable target profile.
  const loom_target_profile_t* profile;

  // Canonical family-owned selector retained by |storage|.
  iree_string_view_t selector;

  // Optional family-owned profile and selector storage.
  void* storage;

  // Host allocator used to allocate |storage|.
  iree_allocator_t allocator;
} loom_target_profile_selection_t;

// Parses |value| as a family-qualified target profile specification.
//
// The value is split at its first colon so family-owned selectors may contain
// additional colons. Leading and trailing whitespace around the complete
// value, family, and selector is ignored. Empty components are rejected.
iree_status_t loom_target_profile_specification_parse(
    iree_string_view_t value,
    loom_target_profile_specification_t* out_specification);

// Asks |provider| to select and canonicalize one family-owned selector.
//
// On success |out_selection| owns its profile and canonical selector until
// deinitialized. Providers without public selector syntax return
// IREE_STATUS_UNIMPLEMENTED.
iree_status_t loom_target_provider_select_profile(
    const loom_target_provider_t* provider, iree_string_view_t selector,
    iree_allocator_t allocator, loom_target_profile_selection_t* out_selection);

// Resolves and selects one family-qualified target profile from |environment|.
iree_status_t loom_target_environment_select_profile(
    const loom_target_environment_t* environment, iree_string_view_t value,
    iree_allocator_t allocator, loom_target_profile_selection_t* out_selection);

// Releases family-owned storage and resets |selection|.
void loom_target_profile_selection_deinitialize(
    loom_target_profile_selection_t* selection);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_PROFILE_SELECTION_H_
