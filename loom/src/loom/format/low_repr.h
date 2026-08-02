// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Low representation-contract value codec.
//
// Low function wrappers select one required representation contract. Text and
// bytecode use stable descriptor keys at their input/output boundaries while
// canonical in-memory packets retain only the dense descriptor ordinal in that
// selected contract. This target-independent interface lets format codecs make
// that conversion without depending on generated descriptor tables.

#ifndef LOOM_FORMAT_LOW_REPR_H_
#define LOOM_FORMAT_LOW_REPR_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_repr_environment_state_t
    loom_low_repr_environment_state_t;
typedef struct loom_low_repr_descriptor_set_t loom_low_repr_descriptor_set_t;

// Canonical packet identity resolved at an input boundary.
typedef struct loom_low_repr_descriptor_value_t {
  // Dense descriptor row ordinal in the selected representation contract.
  uint32_t ordinal;
  // Effective operation traits derived from the selected descriptor row.
  loom_trait_flags_t effective_traits;
} loom_low_repr_descriptor_value_t;

// Resolves a representation-contract key to an environment-owned descriptor
// set. Returns NULL when the contract is not available in the environment.
typedef const loom_low_repr_descriptor_set_t* (
    *loom_low_repr_lookup_descriptor_set_fn_t)(
    const loom_low_repr_environment_state_t* state, iree_string_view_t key);

// Resolves a stable descriptor key within |descriptor_set|. Returns false when
// the key is not present. Successful results are complete canonical packet
// values; there is no unresolved ordinal.
typedef bool (*loom_low_repr_resolve_descriptor_fn_t)(
    const loom_low_repr_environment_state_t* state,
    const loom_low_repr_descriptor_set_t* descriptor_set,
    iree_string_view_t key, loom_low_repr_descriptor_value_t* out_value);

// Returns the stable key for |ordinal| within |descriptor_set|, or an empty
// view when the ordinal is outside the selected contract.
typedef iree_string_view_t (*loom_low_repr_descriptor_key_fn_t)(
    const loom_low_repr_environment_state_t* state,
    const loom_low_repr_descriptor_set_t* descriptor_set, uint32_t ordinal);

typedef struct loom_low_repr_environment_vtable_t {
  // All callbacks are required. Generic format readers and writers may omit
  // the entire environment only when they do not materialize scoped Low
  // representation values. A Low text wrapper requires the environment for
  // its descriptor-scoped signature types even when it has no body.
  // Resolves the one representation contract active for a Low wrapper.
  loom_low_repr_lookup_descriptor_set_fn_t lookup_descriptor_set;
  // Resolves stable packet keys while constructing canonical in-memory IR.
  loom_low_repr_resolve_descriptor_fn_t resolve_descriptor;
  // Recovers stable packet keys while printing or serializing canonical IR.
  loom_low_repr_descriptor_key_fn_t descriptor_key;
} loom_low_repr_environment_vtable_t;

// Borrowed codec environment supplied by the embedding compiler/tool.
typedef struct loom_low_repr_environment_t {
  // Function table implementing representation-contract resolution.
  const loom_low_repr_environment_vtable_t* vtable;
  // Environment-owned state passed to every codec callback.
  const loom_low_repr_environment_state_t* state;
} loom_low_repr_environment_t;

static inline const loom_low_repr_descriptor_set_t*
loom_low_repr_lookup_descriptor_set(
    const loom_low_repr_environment_t* environment, iree_string_view_t key) {
  return environment->vtable->lookup_descriptor_set(environment->state, key);
}

static inline bool loom_low_repr_resolve_descriptor(
    const loom_low_repr_environment_t* environment,
    const loom_low_repr_descriptor_set_t* descriptor_set,
    iree_string_view_t key, loom_low_repr_descriptor_value_t* out_value) {
  return environment->vtable->resolve_descriptor(
      environment->state, descriptor_set, key, out_value);
}

static inline iree_string_view_t loom_low_repr_descriptor_key(
    const loom_low_repr_environment_t* environment,
    const loom_low_repr_descriptor_set_t* descriptor_set, uint32_t ordinal) {
  return environment->vtable->descriptor_key(environment->state, descriptor_set,
                                             ordinal);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_LOW_REPR_H_
