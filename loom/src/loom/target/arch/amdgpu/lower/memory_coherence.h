// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Cache visibility shared by source atomics and compiler-generated publication.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_MEMORY_COHERENCE_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_MEMORY_COHERENCE_H_

#include "loom/codegen/low/descriptors.h"
#include "loom/ops/cache.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint8_t loom_amdgpu_memory_coherence_attrs_t;

enum loom_amdgpu_memory_coherence_attr_bits_e {
  LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_GLC = 1u << 0,
  LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_SC0 = 1u << 1,
  LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_SC1 = 1u << 2,
  LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_SCOPE = 1u << 3,
};

typedef struct loom_amdgpu_memory_coherence_rule_t {
  // Load attributes indexed by the selected cache scope.
  loom_amdgpu_memory_coherence_attrs_t load_attrs[LOOM_CACHE_SCOPE_COUNT_];
  // Store attributes indexed by the selected cache scope.
  loom_amdgpu_memory_coherence_attrs_t store_attrs[LOOM_CACHE_SCOPE_COUNT_];
  // Update coherence attributes; the descriptor owns return-value control.
  loom_amdgpu_memory_coherence_attrs_t atomic_attrs[LOOM_CACHE_SCOPE_COUNT_];
  // Cache-control attributes indexed by the selected cache scope.
  loom_amdgpu_memory_coherence_attrs_t cache_attrs[LOOM_CACHE_SCOPE_COUNT_];
  // Explicit completion before source release updates, in addition to the
  // shared memory frontier's ordering of prior global accesses.
  uint32_t release_wait_masks[2];
  // Number of populated release wait masks.
  uint8_t release_wait_count;
  // Local completion needed for cross-address-space ordering and flat atomics.
  uint32_t local_wait_mask;
  // Cache writeback packet, or zero when global completion suffices.
  loom_amdgpu_descriptor_ref_t writeback;
  // Narrowest cache scope that requires writeback.
  uint8_t writeback_scope;
  // Completion required after writeback.
  uint32_t writeback_wait_mask;
  // Cache invalidation packets, in issue order.
  loom_amdgpu_descriptor_ref_t invalidates[2];
  // Narrowest cache scope requiring each invalidation packet. Zero
  // (workgroup) applies the packet at every acquiring scope.
  uint8_t invalidate_scopes[2];
  // Number of populated cache invalidation packets.
  uint8_t invalidate_count;
  // Completion required after invalidation, or zero for ordered invalidates.
  uint32_t invalidate_wait_mask;
  // Ordering of ordinary global and LDS accesses around workgroup atomics.
  struct {
    // Completion before publication, in issue order.
    uint32_t release_wait_masks[3];
    // Number of populated release wait masks.
    uint8_t release_wait_count;
    // Cache domain shared by all waves in the target's workgroup mode.
    uint8_t cache_scope;
    // Cache invalidation after observation completion, or zero for a shared
    // cache.
    loom_amdgpu_descriptor_ref_t invalidate;
  } workgroup;
} loom_amdgpu_memory_coherence_rule_t;

// Returns the target's coherence recipe, or NULL for an unsupported model.
const loom_amdgpu_memory_coherence_rule_t* loom_amdgpu_memory_coherence_rule(
    const loom_low_descriptor_set_t* descriptor_set);

// Returns whether |rule| issues invalidation |index| when acquiring at |scope|.
static inline bool loom_amdgpu_memory_coherence_invalidates_at(
    const loom_amdgpu_memory_coherence_rule_t* rule, uint8_t index,
    loom_cache_scope_t scope) {
  return scope >= rule->invalidate_scopes[index];
}

// Maps the supported source atomic scopes to cache packet scopes.
loom_cache_scope_t loom_amdgpu_memory_coherence_scope(
    const loom_low_descriptor_set_t* descriptor_set, uint8_t atomic_scope);

// At most two cache fields encode any supported coherence policy.
#define LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_CAPACITY 2

typedef struct loom_amdgpu_memory_coherence_attr_t {
  // Descriptor attribute spelling.
  iree_string_view_t name;
  // Immediate value selected for the requested scope.
  uint32_t value;
} loom_amdgpu_memory_coherence_attr_t;

// Expands a trusted policy mask into descriptor attributes without interning.
uint8_t loom_amdgpu_memory_coherence_select_attrs(
    loom_amdgpu_memory_coherence_attrs_t flags, loom_cache_scope_t scope,
    loom_amdgpu_memory_coherence_attr_t
        attrs[LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_CAPACITY]);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_MEMORY_COHERENCE_H_
