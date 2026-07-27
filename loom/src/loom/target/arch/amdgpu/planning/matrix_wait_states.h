// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU matrix wait-state tables.
//
// These rows are target-owned scheduling facts for matrix result forwarding and
// reuse hazards. The wait-state planner classifies producer/consumer packets
// into these compact ids and then looks up the required instruction-slot
// distance here.

#ifndef LOOM_TARGET_ARCH_AMDGPU_PLANNING_MATRIX_WAIT_STATES_H_
#define LOOM_TARGET_ARCH_AMDGPU_PLANNING_MATRIX_WAIT_STATES_H_

#include "iree/base/api.h"
#include "loom/target/arch/amdgpu/target_info_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_amdgpu_matrix_wait_profile_e {
  // Unknown or uninitialized matrix wait profile.
  LOOM_AMDGPU_MATRIX_WAIT_PROFILE_UNKNOWN = 0,
  // MFMA/SMFMAC wait rows used by pre-gfx950 MFMA-capable processors.
  LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_PRE_GFX950 = 1,
  // MFMA/SMFMAC wait rows used by gfx950-family processors.
  LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950 = 2,
  // Number of matrix wait profile values.
  LOOM_AMDGPU_MATRIX_WAIT_PROFILE_COUNT_,
} loom_amdgpu_matrix_wait_profile_t;

typedef enum loom_amdgpu_matrix_wait_result_use_e {
  // Unknown or uninitialized matrix result use.
  LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_UNKNOWN = 0,
  // A non-matrix packet reads or overwrites outstanding matrix result storage.
  LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_NON_MATRIX = 1,
  // A matrix packet reads outstanding matrix result storage as exact SrcC.
  LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_EXACT = 2,
  // A matrix packet reads overlapping outstanding matrix result storage as
  // SrcC, but the producer/consumer physical ranges are not an exact match.
  LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_OVERLAP = 3,
  // A matrix packet reads outstanding matrix result storage as SrcA/SrcB/index.
  LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRC_AB = 4,
  // Number of matrix result use values.
  LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_COUNT_,
} loom_amdgpu_matrix_wait_result_use_t;

typedef struct loom_amdgpu_matrix_wait_result_row_t {
  // Processor-family wait profile that selects this row.
  loom_amdgpu_matrix_wait_profile_t profile;
  // How the later packet consumes the outstanding matrix result.
  loom_amdgpu_matrix_wait_result_use_t use;
  // Matrix result pass count covered by this row.
  uint16_t pass_count;
  // Required instruction-slot distance before the consumer may proceed.
  uint16_t cycle_count;
} loom_amdgpu_matrix_wait_result_row_t;

// Returns the stable spelling for a matrix wait profile.
iree_string_view_t loom_amdgpu_matrix_wait_profile_name(
    loom_amdgpu_matrix_wait_profile_t profile);

// Returns the stable spelling for a matrix result use.
iree_string_view_t loom_amdgpu_matrix_wait_result_use_name(
    loom_amdgpu_matrix_wait_result_use_t use);

// Maps a processor matrix feature profile to the matrix wait table profile.
bool loom_amdgpu_matrix_wait_profile_from_feature_profile(
    loom_amdgpu_matrix_feature_profile_t feature_profile,
    loom_amdgpu_matrix_wait_profile_t* out_profile);

// Returns the number of target-owned matrix result wait rows.
iree_host_size_t loom_amdgpu_matrix_wait_result_row_count(void);

// Returns a target-owned matrix result wait row by ordinal, or NULL when
// |index| is out of range.
const loom_amdgpu_matrix_wait_result_row_t*
loom_amdgpu_matrix_wait_result_row_at(iree_host_size_t index);

// Finds the target-owned matrix result wait row for |profile|, |pass_count|,
// and |use|, or NULL when the combination is unsupported.
const loom_amdgpu_matrix_wait_result_row_t* loom_amdgpu_matrix_wait_result_find(
    loom_amdgpu_matrix_wait_profile_t profile, uint16_t pass_count,
    loom_amdgpu_matrix_wait_result_use_t use);

// Looks up the required wait cycles for a target-owned matrix result use.
bool loom_amdgpu_matrix_wait_result_cycle_count(
    loom_amdgpu_matrix_wait_profile_t profile, uint16_t pass_count,
    loom_amdgpu_matrix_wait_result_use_t use, uint16_t* out_cycle_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_PLANNING_MATRIX_WAIT_STATES_H_
