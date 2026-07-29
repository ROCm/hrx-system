// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-family profile identity shared by compiler and embedding boundaries.
//
// Profiles are immutable, family-owned structured target facts. The
// target-neutral base identifies the owning family and exposes the bundle
// projection used by common legality and lowering code. It never enumerates
// target families or stores opaque semantic payloads.

#ifndef LOOM_TARGET_PROFILE_H_
#define LOOM_TARGET_PROFILE_H_

#include "iree/base/api.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Stable process-local identity for one target-family profile representation.
//
// Family packages define one static descriptor and use pointer identity for
// checked dispatch and casts.
typedef struct loom_target_profile_type_t {
  // Stable family name used in diagnostics and reports.
  iree_string_view_t name;
} loom_target_profile_type_t;

// Target-neutral base embedded first in every target-family profile.
struct loom_target_profile_t {
  // Target-family representation owning the complete profile.
  const loom_target_profile_type_t* type;

  // Target-neutral bundle projection, or NULL when the structured facts are
  // not concrete enough to select one.
  const loom_target_bundle_t* target_bundle;
};

// Returns whether |profile| has the expected target-family representation.
static inline bool loom_target_profile_has_type(
    const loom_target_profile_t* profile,
    const loom_target_profile_type_t* expected_type) {
  return profile != NULL && profile->type == expected_type;
}

// Returns the target-neutral bundle projected by |profile|, or NULL.
static inline const loom_target_bundle_t* loom_target_profile_bundle(
    const loom_target_profile_t* profile) {
  return profile ? profile->target_bundle : NULL;
}

// Returns the target profile borrowed by |selection|, or NULL.
static inline const loom_target_profile_t* loom_target_selection_profile(
    loom_target_selection_t selection) {
  return selection.profile;
}

// Returns the target-neutral bundle projected by |selection|, or NULL.
static inline const loom_target_bundle_t* loom_target_selection_bundle(
    loom_target_selection_t selection) {
  return loom_target_profile_bundle(selection.profile);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_PROFILE_H_
