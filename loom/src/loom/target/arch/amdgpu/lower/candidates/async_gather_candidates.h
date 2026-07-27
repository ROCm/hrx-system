// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU source-to-low async gather descriptor candidate rows.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_CANDIDATES_ASYNC_GATHER_CANDIDATES_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_CANDIDATES_ASYNC_GATHER_CANDIDATES_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_async_gather_descriptor_candidate_t {
  // Number of source bytes moved by the async gather packet.
  uint32_t packet_byte_count;
  // Dense AMDGPU descriptor ref selected when present in the descriptor set.
  loom_amdgpu_descriptor_ref_t descriptor_ref;
} loom_amdgpu_async_gather_descriptor_candidate_t;

// Candidate rows sorted by packet byte count.
extern const loom_amdgpu_async_gather_descriptor_candidate_t
    kLoomAmdgpuAsyncGatherDescriptorCandidates[];
extern const iree_host_size_t kLoomAmdgpuAsyncGatherDescriptorCandidateCount;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_CANDIDATES_ASYNC_GATHER_CANDIDATES_H_
