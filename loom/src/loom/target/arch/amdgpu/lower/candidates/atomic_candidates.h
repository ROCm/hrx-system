// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU source-to-low atomic descriptor candidate rows.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_CANDIDATES_ATOMIC_CANDIDATES_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_CANDIDATES_ATOMIC_CANDIDATES_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "loom/ops/atomic.h"
#include "loom/target/arch/amdgpu/lower/kinds.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_AMDGPU_ATOMIC_KIND_NONE UINT8_MAX
#define LOOM_AMDGPU_ATOMIC_MEMORY_SPACE_INDEX_COUNT 3u
#define LOOM_AMDGPU_ATOMIC_ADDRESS_FORM_INDEX_COUNT 3u
#define LOOM_AMDGPU_ATOMIC_OPERATION_KIND_INDEX_COUNT \
  LOOM_AMDGPU_ATOMIC_OPERATION_COUNT_
#define LOOM_AMDGPU_ATOMIC_KIND_INDEX_NONE LOOM_ATOMIC_KIND_COUNT_
#define LOOM_AMDGPU_ATOMIC_KIND_INDEX_COUNT (LOOM_ATOMIC_KIND_COUNT_ + 1u)
#define LOOM_AMDGPU_ATOMIC_DESCRIPTOR_CANDIDATE_RANGE_COUNT \
  (LOOM_AMDGPU_ATOMIC_MEMORY_SPACE_INDEX_COUNT *            \
   LOOM_AMDGPU_ATOMIC_ADDRESS_FORM_INDEX_COUNT *            \
   LOOM_AMDGPU_ATOMIC_OPERATION_KIND_INDEX_COUNT *          \
   LOOM_AMDGPU_ATOMIC_KIND_INDEX_COUNT)

#define LOOM_AMDGPU_ATOMIC_DESCRIPTOR_CANDIDATE_RANGE_INDEX(                   \
    memory_space_index, address_form_index, operation_kind, atomic_kind_index) \
  (((((uint32_t)(memory_space_index) *                                         \
      LOOM_AMDGPU_ATOMIC_ADDRESS_FORM_INDEX_COUNT) +                           \
     (uint32_t)(address_form_index)) *                                         \
        LOOM_AMDGPU_ATOMIC_OPERATION_KIND_INDEX_COUNT +                        \
    (uint32_t)(operation_kind)) *                                              \
       LOOM_AMDGPU_ATOMIC_KIND_INDEX_COUNT +                                   \
   (uint32_t)(atomic_kind_index))

typedef enum loom_amdgpu_atomic_value_kind_e {
  LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32 = 0,
  LOOM_AMDGPU_ATOMIC_VALUE_KIND_F32 = 1,
  LOOM_AMDGPU_ATOMIC_VALUE_KIND_I64 = 2,
  LOOM_AMDGPU_ATOMIC_VALUE_KIND_PACKED_F16 = 3,
  LOOM_AMDGPU_ATOMIC_VALUE_KIND_PACKED_BF16 = 4,
  // Type-agnostic 32-bit payload compared by bit pattern.
  LOOM_AMDGPU_ATOMIC_VALUE_KIND_B32 = 5,
  // Type-agnostic 64-bit payload compared by bit pattern.
  LOOM_AMDGPU_ATOMIC_VALUE_KIND_B64 = 6,
} loom_amdgpu_atomic_value_kind_t;

typedef struct loom_amdgpu_atomic_descriptor_candidate_t {
  // Source scalar value type required by this row.
  loom_amdgpu_atomic_value_kind_t value_kind;
  // Dense AMDGPU descriptor ref selected when present in the descriptor set.
  loom_amdgpu_descriptor_ref_t descriptor_ref;
} loom_amdgpu_atomic_descriptor_candidate_t;

typedef struct loom_amdgpu_atomic_descriptor_candidate_range_t {
  // First candidate row for the packed atomic selector key.
  uint16_t first_candidate;
  // Number of contiguous candidate rows for the packed atomic selector key.
  uint16_t candidate_count;
} loom_amdgpu_atomic_descriptor_candidate_range_t;

extern const loom_amdgpu_atomic_descriptor_candidate_t
    kLoomAmdgpuAtomicDescriptorCandidates[];
extern const iree_host_size_t kLoomAmdgpuAtomicDescriptorCandidateCount;
extern const loom_amdgpu_atomic_descriptor_candidate_range_t
    kLoomAmdgpuAtomicDescriptorCandidateRanges
        [LOOM_AMDGPU_ATOMIC_DESCRIPTOR_CANDIDATE_RANGE_COUNT];

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_CANDIDATES_ATOMIC_CANDIDATES_H_
