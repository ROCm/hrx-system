// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_CODEGEN_LOW_DESCRIPTOR_SCOPE_H_
#define LOOM_CODEGEN_LOW_DESCRIPTOR_SCOPE_H_

#include "loom/codegen/low/descriptors.h"

// Explicit applicability of a provider in a composed registry. Exactly one
// identity is populated: a target family, an exact descriptor set, or SHARED.
// An omitted initializer is invalid and never denotes a shared provider.
typedef struct loom_low_descriptor_scope_t {
  // Target-family identity, UINT64_MAX for shared providers, or NONE when an
  // exact descriptor-set identity is used instead.
  uint64_t target_stable_id;
  // Exact descriptor-set identity, or NONE for family/shared providers.
  uint64_t descriptor_set_stable_id;
} loom_low_descriptor_scope_t;

// Initializer for providers that deliberately serve every target, including
// contexts without a selected descriptor set.
#define LOOM_LOW_DESCRIPTOR_SCOPE_SHARED {UINT64_MAX, LOOM_LOW_STABLE_ID_NONE}

// Validates immutable registration data during provider qualification.
static inline bool loom_low_descriptor_scope_is_valid(
    loom_low_descriptor_scope_t scope) {
  return (scope.target_stable_id != LOOM_LOW_STABLE_ID_NONE) !=
         (scope.descriptor_set_stable_id != LOOM_LOW_STABLE_ID_NONE);
}

// Dispatch establishes applicability before invoking provider-owned callbacks.
// Type, feature, and phase predicates remain the provider's responsibility.
static inline bool loom_low_descriptor_scope_matches(
    loom_low_descriptor_scope_t scope,
    const loom_low_descriptor_set_t* descriptor_set) {
  if (scope.target_stable_id == UINT64_MAX) {
    return true;
  }
  if (descriptor_set == NULL) {
    return false;
  }
  return scope.target_stable_id != LOOM_LOW_STABLE_ID_NONE
             ? scope.target_stable_id == descriptor_set->target_stable_id
             : scope.descriptor_set_stable_id == descriptor_set->stable_id;
}

#endif  // LOOM_CODEGEN_LOW_DESCRIPTOR_SCOPE_H_
