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
#include "loom/target/facts.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Projects one complete family profile into compiler-owned typed facts.
//
// |out_facts| has already been zeroed and initialized from the profile's
// target-neutral bundle and explicit field set. Implementations populate the
// family selector and typed extension, allocating any nested immutable storage
// from |arena|. Family views are rebound after this callback returns.
typedef iree_status_t (*loom_target_profile_project_facts_fn_t)(
    const loom_target_profile_t* profile, iree_arena_allocator_t* arena,
    loom_target_facts_t* out_facts);

// Stable process-local identity for one target-family profile representation.
//
// Family packages define one static descriptor and use pointer identity for
// checked dispatch and casts.
typedef struct loom_target_profile_type_t {
  // Stable family name used in diagnostics and reports.
  iree_string_view_t name;

  // Static typed fact representation produced from this profile family.
  const loom_target_fact_type_t* fact_type;

  // Family-owned structured profile projector.
  loom_target_profile_project_facts_fn_t project_facts;
} loom_target_profile_type_t;

// Target-neutral base embedded first in every target-family profile.
struct loom_target_profile_t {
  // Target-family representation owning the complete profile.
  const loom_target_profile_type_t* type;

  // Target-neutral bundle projection, or NULL when the structured facts are
  // not concrete enough to select one.
  const loom_target_bundle_t* target_bundle;

  // Common fields explicitly supplied by the profile producer. Presence is
  // independent of value equality with a preset, including explicit zeroes.
  loom_target_fact_field_set_t explicit_fields;
};

// Destroys one allocator-owned target profile.
typedef void (*loom_target_profile_destroy_fn_t)(loom_target_profile_t* profile,
                                                 iree_allocator_t allocator);

// A selected target profile and its optional ownership callback.
//
// A NULL destroy callback denotes immutable process-lifetime storage. Otherwise
// the selection owns |profile| until it is released or transferred to another
// owner with the same allocator.
typedef struct loom_target_profile_selection_t {
  // Selected immutable target-family profile, or NULL when no profile matched.
  loom_target_profile_t* profile;

  // Optional callback used to release |profile|.
  loom_target_profile_destroy_fn_t destroy;
} loom_target_profile_selection_t;

// Releases an owned profile selection and resets it to zero.
void loom_target_profile_selection_release(
    loom_target_profile_selection_t* selection, iree_allocator_t allocator);

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

// Projects |profile| once into compiler-owned typed target facts.
//
// The returned construction object and all nested dynamic storage are
// allocated from |arena| and remain valid for its lifetime. The caller may
// apply authored requirements and function-local contract facts before
// publishing it as immutable. Profiles are external input and need only remain
// live for this call.
iree_status_t loom_target_profile_project_facts(
    const loom_target_profile_t* profile, iree_arena_allocator_t* arena,
    loom_target_facts_t** out_facts);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_PROFILE_H_
