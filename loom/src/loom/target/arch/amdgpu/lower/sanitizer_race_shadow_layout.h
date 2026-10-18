// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU TSAN queue shadow arena layout lowering.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_SANITIZER_RACE_SHADOW_LAYOUT_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_SANITIZER_RACE_SHADOW_LAYOUT_H_

#include "iree/base/api.h"
#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_sanitizer_race_shadow_layout_config_t {
  // Total bytes in the queue-local shadow allocation.
  loom_value_id_t shadow_size;
} loom_amdgpu_sanitizer_race_shadow_layout_config_t;

typedef struct loom_amdgpu_sanitizer_race_shadow_layout_t {
  // Final group-segment byte size carried by the dispatch packet.
  loom_value_id_t workgroup_local_memory_size;
  // Bytes occupied by one workgroup shadow record for this dispatch.
  loom_value_id_t workgroup_shadow_stride;
  // Flattened 64-bit workgroup ordinal in the dispatch.
  loom_value_id_t workgroup_ordinal;
  // Byte offset of this workgroup record from the queue shadow base.
  loom_value_id_t workgroup_shadow_offset;
  // Exclusive byte end required by this workgroup record.
  loom_value_id_t required_shadow_end;
  // SCC true when this workgroup record is fully contained in the arena.
  loom_value_id_t in_bounds_scc;
  // SCC true only for the first workgroup that crosses the arena boundary.
  loom_value_id_t first_invalid_scc;
} loom_amdgpu_sanitizer_race_shadow_layout_t;

// Builds the checked byte layout for the current dispatch workgroup in the
// queue-local TSAN shadow arena. Arithmetic overflow makes the workgroup
// invalid and saturates |required_shadow_end| for capacity diagnostics.
iree_status_t loom_amdgpu_build_sanitizer_race_shadow_layout(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_race_shadow_layout_config_t* config,
    loom_amdgpu_sanitizer_race_shadow_layout_t* out_layout);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_SANITIZER_RACE_SHADOW_LAYOUT_H_
