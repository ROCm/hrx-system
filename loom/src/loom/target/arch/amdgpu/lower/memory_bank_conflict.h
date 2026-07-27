// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU LDS bank-conflict classification for selected memory packets.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_MEMORY_BANK_CONFLICT_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_MEMORY_BANK_CONFLICT_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "loom/target/arch/amdgpu/lower/plan.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_amdgpu_memory_bank_conflict_kind_e {
  // The selected access does not expose one static lane-to-bank pattern.
  LOOM_AMDGPU_MEMORY_BANK_CONFLICT_KIND_UNKNOWN = 0,
  // The selected access is proven conflict-free for the target bank geometry.
  LOOM_AMDGPU_MEMORY_BANK_CONFLICT_KIND_FREE = 1,
  // The selected access is proven conflict-free and carries visible padding.
  LOOM_AMDGPU_MEMORY_BANK_CONFLICT_KIND_PADDED_FREE = 2,
  // The selected access has a statically visible bank conflict risk.
  LOOM_AMDGPU_MEMORY_BANK_CONFLICT_KIND_RISK = 3,
  // The selected access strides through sub-32-bit bank lanes.
  LOOM_AMDGPU_MEMORY_BANK_CONFLICT_KIND_SUBWORD_RISK = 4,
} loom_amdgpu_memory_bank_conflict_kind_t;

typedef struct loom_amdgpu_memory_bank_geometry_t {
  // Number of LDS banks in one bank cycle.
  uint32_t bank_count;
  // Number of consecutive bytes mapped to one bank word.
  uint32_t bank_width_bytes;
} loom_amdgpu_memory_bank_geometry_t;

typedef struct loom_amdgpu_memory_bank_conflict_summary_t {
  // Distance between adjacent workitems in target bank words.
  uint32_t bank_stride_words;
  // Estimated conflict degree across one bank cycle, or zero when unknown.
  uint32_t conflict_degree;
  // Stable structural classification for the selected access.
  loom_amdgpu_memory_bank_conflict_kind_t kind;
} loom_amdgpu_memory_bank_conflict_summary_t;

// Returns the default AMDGPU LDS bank geometry used by current targets.
loom_amdgpu_memory_bank_geometry_t loom_amdgpu_memory_bank_default_lds_geometry(
    void);

// Returns the stable diagnostic/report key for |kind|.
iree_string_view_t loom_amdgpu_memory_bank_conflict_kind_key(
    loom_amdgpu_memory_bank_conflict_kind_t kind);

// Classifies the selected LDS packet using static source and packet facts.
//
// The classifier is intentionally a cold-path helper: it consumes a fully
// selected packet and returns compact facts that sinks can render or copy into
// report rows. It does not allocate, intern names, emit diagnostics, or affect
// packet selection.
loom_amdgpu_memory_bank_conflict_summary_t
loom_amdgpu_memory_access_bank_conflict_summary(
    const loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_bank_geometry_t geometry);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_MEMORY_BANK_CONFLICT_H_
