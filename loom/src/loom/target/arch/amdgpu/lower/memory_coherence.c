// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/memory_coherence.h"

#include "loom/ops/atomic.h"
#include "loom/target/arch/amdgpu/planning/wait_packets.h"
#include "loom/target/arch/amdgpu/target_info.h"

static const loom_amdgpu_memory_coherence_rule_t kMemoryCoherenceRules[] = {
    [LOOM_AMDGPU_MEMORY_ORDERING_MODEL_GFX11] =
        {
            // GLC bypasses GL0/GL1 on loads. Stores are write-through; atomic
            // descriptors use GLC as return control rather than coherence.
            .load_attrs =
                {
                    [LOOM_CACHE_SCOPE_CLUSTER] =
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_GLC,
                    [LOOM_CACHE_SCOPE_DEVICE] =
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_GLC,
                    [LOOM_CACHE_SCOPE_SYSTEM] =
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_GLC,
                },
            .release_wait_masks = {LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD |
                                       LOOM_AMDGPU_WAIT_COUNTER_MASK_LDS |
                                       LOOM_AMDGPU_WAIT_COUNTER_MASK_SMEM,
                                   LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE},
            .release_wait_count = 2,
            .local_wait_mask = LOOM_AMDGPU_WAIT_COUNTER_MASK_LDS |
                               LOOM_AMDGPU_WAIT_COUNTER_MASK_SMEM,
            .invalidates = {LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_GL1_INV,
                            LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_GL0_INV},
            .invalidate_count = 2,
            // WGP waves can use different GL0 caches. Complete both memory
            // paths before publication and invalidate GL0 after acquisition.
            .workgroup =
                {
                    .cache_scope = LOOM_CACHE_SCOPE_CLUSTER,
                    .release_wait_masks =
                        {LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD |
                             LOOM_AMDGPU_WAIT_COUNTER_MASK_LDS |
                             LOOM_AMDGPU_WAIT_COUNTER_MASK_SMEM,
                         LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE},
                    .release_wait_count = 2,
                    .invalidate = LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_GL0_INV,
                },
        },
    [LOOM_AMDGPU_MEMORY_ORDERING_MODEL_GFX12] =
        {
            .load_attrs =
                {
                    [LOOM_CACHE_SCOPE_WORKGROUP] =
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_SCOPE,
                    [LOOM_CACHE_SCOPE_CLUSTER] =
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_SCOPE,
                    [LOOM_CACHE_SCOPE_DEVICE] =
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_SCOPE,
                    [LOOM_CACHE_SCOPE_SYSTEM] =
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_SCOPE,
                },
            .store_attrs =
                {
                    [LOOM_CACHE_SCOPE_WORKGROUP] =
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_SCOPE,
                    [LOOM_CACHE_SCOPE_CLUSTER] =
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_SCOPE,
                    [LOOM_CACHE_SCOPE_DEVICE] =
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_SCOPE,
                    [LOOM_CACHE_SCOPE_SYSTEM] =
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_SCOPE,
                },
            .atomic_attrs =
                {
                    [LOOM_CACHE_SCOPE_WORKGROUP] =
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_SCOPE,
                    [LOOM_CACHE_SCOPE_CLUSTER] =
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_SCOPE,
                    [LOOM_CACHE_SCOPE_DEVICE] =
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_SCOPE,
                    [LOOM_CACHE_SCOPE_SYSTEM] =
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_SCOPE,
                },
            .cache_attrs =
                {
                    [LOOM_CACHE_SCOPE_WORKGROUP] =
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_SCOPE,
                    [LOOM_CACHE_SCOPE_CLUSTER] =
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_SCOPE,
                    [LOOM_CACHE_SCOPE_DEVICE] =
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_SCOPE,
                    [LOOM_CACHE_SCOPE_SYSTEM] =
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_SCOPE,
                },
            .release_wait_masks = {LOOM_AMDGPU_WAIT_COUNTER_MASK_LDS},
            .release_wait_count = 1,
            .local_wait_mask = LOOM_AMDGPU_WAIT_COUNTER_MASK_LDS,
            .writeback = LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_WB,
            .writeback_scope = LOOM_CACHE_SCOPE_SYSTEM,
            .writeback_wait_mask = LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE,
            .invalidates = {LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_INV},
            .invalidate_count = 1,
            .workgroup =
                {
                    .cache_scope = LOOM_CACHE_SCOPE_CLUSTER,
                    .release_wait_masks =
                        {LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD,
                         LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE,
                         LOOM_AMDGPU_WAIT_COUNTER_MASK_LDS},
                    .release_wait_count = 3,
                    .invalidate = LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_INV,
                },
        },
    [LOOM_AMDGPU_MEMORY_ORDERING_MODEL_GFX125] =
        {
            .load_attrs =
                {
                    [LOOM_CACHE_SCOPE_WORKGROUP] =
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_SCOPE,
                    [LOOM_CACHE_SCOPE_CLUSTER] =
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_SCOPE,
                    [LOOM_CACHE_SCOPE_DEVICE] =
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_SCOPE,
                    [LOOM_CACHE_SCOPE_SYSTEM] =
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_SCOPE,
                },
            .store_attrs =
                {
                    [LOOM_CACHE_SCOPE_WORKGROUP] =
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_SCOPE,
                    [LOOM_CACHE_SCOPE_CLUSTER] =
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_SCOPE,
                    [LOOM_CACHE_SCOPE_DEVICE] =
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_SCOPE,
                    [LOOM_CACHE_SCOPE_SYSTEM] =
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_SCOPE,
                },
            .atomic_attrs =
                {
                    [LOOM_CACHE_SCOPE_WORKGROUP] =
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_SCOPE,
                    [LOOM_CACHE_SCOPE_CLUSTER] =
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_SCOPE,
                    [LOOM_CACHE_SCOPE_DEVICE] =
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_SCOPE,
                    [LOOM_CACHE_SCOPE_SYSTEM] =
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_SCOPE,
                },
            .cache_attrs =
                {
                    [LOOM_CACHE_SCOPE_WORKGROUP] =
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_SCOPE,
                    [LOOM_CACHE_SCOPE_CLUSTER] =
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_SCOPE,
                    [LOOM_CACHE_SCOPE_DEVICE] =
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_SCOPE,
                    [LOOM_CACHE_SCOPE_SYSTEM] =
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_SCOPE,
                },
            .release_wait_masks = {LOOM_AMDGPU_WAIT_COUNTER_MASK_LDS},
            .release_wait_count = 1,
            .local_wait_mask = LOOM_AMDGPU_WAIT_COUNTER_MASK_LDS,
            .writeback = LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_WB,
            .writeback_scope = LOOM_CACHE_SCOPE_DEVICE,
            .writeback_wait_mask = LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE,
            .invalidates = {LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_INV},
            .invalidate_count = 1,
            .invalidate_wait_mask = LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD,
            // The CU cache is shared, but its ports do not order one another.
            .workgroup =
                {
                    .cache_scope = LOOM_CACHE_SCOPE_WORKGROUP,
                    .release_wait_masks =
                        {LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD,
                         LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE,
                         LOOM_AMDGPU_WAIT_COUNTER_MASK_LDS},
                    .release_wait_count = 3,
                },
        },
    [LOOM_AMDGPU_MEMORY_ORDERING_MODEL_CDNA] =
        {
            // SC0/SC1 encode scope for observations and cache controls. Atomic
            // updates instead use SC0 for return control and SC1 for system
            // scope.
            .load_attrs =
                {
                    [LOOM_CACHE_SCOPE_WORKGROUP] =
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_SC0,
                    [LOOM_CACHE_SCOPE_DEVICE] =
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_SC1,
                    [LOOM_CACHE_SCOPE_SYSTEM] =
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_SC0 |
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_SC1,
                },
            .store_attrs =
                {
                    [LOOM_CACHE_SCOPE_WORKGROUP] =
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_SC0,
                    [LOOM_CACHE_SCOPE_DEVICE] =
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_SC1,
                    [LOOM_CACHE_SCOPE_SYSTEM] =
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_SC0 |
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_SC1,
                },
            .atomic_attrs =
                {
                    [LOOM_CACHE_SCOPE_SYSTEM] =
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_SC1,
                },
            .cache_attrs =
                {
                    [LOOM_CACHE_SCOPE_WORKGROUP] =
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_SC0,
                    [LOOM_CACHE_SCOPE_DEVICE] =
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_SC1,
                    [LOOM_CACHE_SCOPE_SYSTEM] =
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_SC0 |
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_SC1,
                },
            .release_wait_masks = {LOOM_AMDGPU_WAIT_COUNTER_MASK_LDS |
                                   LOOM_AMDGPU_WAIT_COUNTER_MASK_SMEM},
            .release_wait_count = 1,
            .local_wait_mask = LOOM_AMDGPU_WAIT_COUNTER_MASK_LDS |
                               LOOM_AMDGPU_WAIT_COUNTER_MASK_SMEM,
            .writeback = LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_WBL2,
            .writeback_scope = LOOM_CACHE_SCOPE_DEVICE,
            .writeback_wait_mask = LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE,
            .invalidates = {LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_INV},
            .invalidate_count = 1,
            // With TgSplit disabled the workgroup shares an ordered L1.
            .workgroup =
                {
                    .cache_scope = LOOM_CACHE_SCOPE_WORKGROUP,
                    .release_wait_masks = {LOOM_AMDGPU_WAIT_COUNTER_MASK_LDS |
                                           LOOM_AMDGPU_WAIT_COUNTER_MASK_SMEM},
                    .release_wait_count = 1,
                },
        },
    [LOOM_AMDGPU_MEMORY_ORDERING_MODEL_GFX90A] =
        {
            // GLC misses the per-CU L1 for device and system observations.
            // Stores are write-through and update atomics bypass L1, using
            // GLC only as return control (LLVM GFX90A memory-model table).
            .load_attrs =
                {
                    [LOOM_CACHE_SCOPE_DEVICE] =
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_GLC,
                    [LOOM_CACHE_SCOPE_SYSTEM] =
                        LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_GLC,
                },
            .release_wait_masks = {LOOM_AMDGPU_WAIT_COUNTER_MASK_LDS |
                                   LOOM_AMDGPU_WAIT_COUNTER_MASK_SMEM},
            .release_wait_count = 1,
            .local_wait_mask = LOOM_AMDGPU_WAIT_COUNTER_MASK_LDS |
                               LOOM_AMDGPU_WAIT_COUNTER_MASK_SMEM,
            // The device L2 is coherent; only system publication writes it
            // back to memory.
            .writeback = LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_WBL2,
            .writeback_scope = LOOM_CACHE_SCOPE_SYSTEM,
            .writeback_wait_mask = LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE,
            // System acquisition drops stale non-coherent L2 lines before
            // the L1 invalidation shared with device acquisition. Both are
            // ordered with later accesses of the same wave.
            .invalidates = {LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_INVL2,
                            LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_WBINVL1_VOL},
            .invalidate_scopes = {LOOM_CACHE_SCOPE_SYSTEM,
                                  LOOM_CACHE_SCOPE_DEVICE},
            .invalidate_count = 2,
            // With TgSplit disabled the workgroup shares an ordered L1.
            .workgroup =
                {
                    .cache_scope = LOOM_CACHE_SCOPE_WORKGROUP,
                    .release_wait_masks = {LOOM_AMDGPU_WAIT_COUNTER_MASK_LDS |
                                           LOOM_AMDGPU_WAIT_COUNTER_MASK_SMEM},
                    .release_wait_count = 1,
                },
        },
};

const loom_amdgpu_memory_coherence_rule_t* loom_amdgpu_memory_coherence_rule(
    const loom_low_descriptor_set_t* descriptor_set) {
  const loom_amdgpu_descriptor_set_info_t* info =
      loom_amdgpu_target_info_descriptor_set_at(
          descriptor_set->descriptor_set_ordinal);
  return info->vector_memory.ordering_model ==
                 LOOM_AMDGPU_MEMORY_ORDERING_MODEL_NONE
             ? NULL
             : &kMemoryCoherenceRules[info->vector_memory.ordering_model];
}

loom_cache_scope_t loom_amdgpu_memory_coherence_scope(
    const loom_low_descriptor_set_t* descriptor_set, uint8_t atomic_scope) {
  if (atomic_scope == LOOM_ATOMIC_SCOPE_WORKGROUP) {
    return (loom_cache_scope_t)loom_amdgpu_memory_coherence_rule(descriptor_set)
        ->workgroup.cache_scope;
  }
  return atomic_scope == LOOM_ATOMIC_SCOPE_SYSTEM ? LOOM_CACHE_SCOPE_SYSTEM
                                                  : LOOM_CACHE_SCOPE_DEVICE;
}

uint8_t loom_amdgpu_memory_coherence_select_attrs(
    loom_amdgpu_memory_coherence_attrs_t flags, loom_cache_scope_t scope,
    loom_amdgpu_memory_coherence_attr_t
        attrs[LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_CAPACITY]) {
  static const iree_string_view_t names[] = {
      IREE_SVL("glc"), IREE_SVL("sc0"), IREE_SVL("sc1"), IREE_SVL("scope")};
  uint8_t count = 0;
  for (uint8_t i = 0; i < IREE_ARRAYSIZE(names); ++i) {
    const uint8_t flag = 1u << i;
    if (iree_any_bit_set(flags, flag)) {
      attrs[count++] = (loom_amdgpu_memory_coherence_attr_t){
          .name = names[i],
          .value = flag == LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_SCOPE ? scope : 1,
      };
    }
  }
  return count;
}
