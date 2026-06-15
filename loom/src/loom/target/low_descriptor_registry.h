// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Linked low descriptor registry package support.
//
// These target-neutral support functions operate on caller-provided
// descriptor-set and target-bundle tables. Concrete registry packages live with
// the target families that own the linked descriptor providers and target
// bundles. Descriptor provider functions may come from generated table shards,
// but provider arrays, target snapshots, export plans, feature masks, and
// bundles are target policy and stay explicit in the target package.

#ifndef LOOM_TARGET_LOW_DESCRIPTOR_REGISTRY_H_
#define LOOM_TARGET_LOW_DESCRIPTOR_REGISTRY_H_

#include "iree/base/api.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_target_low_descriptor_registry_t {
  // Borrowed descriptor-set provider table linked into this target package.
  const loom_low_descriptor_set_provider_t* descriptor_set_providers;
  // Number of provider functions in |descriptor_set_providers|.
  iree_host_size_t descriptor_set_provider_count;
  // Registry view over the package descriptor-set providers.
  loom_low_descriptor_registry_t registry;
} loom_target_low_descriptor_registry_t;

// Initializes |out_registry| from linked package tables. The provider and
// bundle tables are borrowed and must outlive |out_registry|.
void loom_target_low_descriptor_registry_initialize_from_tables(
    loom_target_low_descriptor_registry_t* out_registry,
    const loom_low_descriptor_set_provider_t* descriptor_set_providers,
    iree_host_size_t descriptor_set_provider_count);

// Appends the borrowed package tables in |source| into caller-owned table
// storage. The inout counts must point at initialized counts already present in
// the destination arrays. This is intended for tool and embedding boundaries
// that link a target-selected subset of registry packages without depending on
// the all-target aggregate package.
iree_status_t loom_target_low_descriptor_registry_append_to_tables(
    const loom_target_low_descriptor_registry_t* source,
    loom_low_descriptor_set_provider_t* descriptor_set_providers,
    iree_host_size_t descriptor_set_provider_capacity,
    iree_host_size_t* descriptor_set_provider_count);

// Selects the descriptor set named by |bundle->config->contract_set_key| from
// |registry|. This is an exact key lookup; target triples and CPUs are
// descriptive facts, not fallback descriptor-selection rules.
iree_status_t loom_target_low_descriptor_set_select_for_bundle(
    const loom_low_descriptor_registry_t* registry,
    const loom_target_bundle_t* bundle,
    const loom_low_descriptor_set_t** out_descriptor_set);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_LOW_DESCRIPTOR_REGISTRY_H_
