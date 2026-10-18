// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/sanitizer_race_history_policy.h"

#define LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_BIT(kind) (1u << (kind))

enum loom_amdgpu_sanitizer_race_access_kind_mask_e {
  LOOM_AMDGPU_SANITIZER_RACE_ALL_ACCESS_KIND_MASK =
      LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_BIT(
          LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_READ) |
      LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_BIT(
          LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_WRITE) |
      LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_BIT(
          LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_READ_WRITE) |
      LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_BIT(
          LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_ATOMIC),
  LOOM_AMDGPU_SANITIZER_RACE_WRITE_LIKE_KIND_MASK =
      LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_BIT(
          LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_WRITE) |
      LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_BIT(
          LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_READ_WRITE),
  LOOM_AMDGPU_SANITIZER_RACE_NONWRITE_KIND_MASK =
      LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_BIT(
          LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_READ) |
      LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_BIT(
          LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_ATOMIC),
};

const loom_amdgpu_sanitizer_race_history_rule_t loom_amdgpu_sanitizer_race_history_rules
    [LOOM_AMDGPU_SANITIZER_RACE_HISTORY_RULE_COUNT] = {
        {
            .current_access_kind_mask =
                LOOM_AMDGPU_SANITIZER_RACE_ALL_ACCESS_KIND_MASK,
            .condition =
                LOOM_AMDGPU_SANITIZER_RACE_HISTORY_CONDITION_SAME_WORKITEM_CATEGORY,
            .condition_history_index = 0u,
            .selected_history_index = 0u,
        },
        {
            .current_access_kind_mask =
                LOOM_AMDGPU_SANITIZER_RACE_ALL_ACCESS_KIND_MASK,
            .condition =
                LOOM_AMDGPU_SANITIZER_RACE_HISTORY_CONDITION_SAME_WORKITEM_CATEGORY,
            .condition_history_index = 1u,
            .selected_history_index = 1u,
        },
        {
            .current_access_kind_mask =
                LOOM_AMDGPU_SANITIZER_RACE_ALL_ACCESS_KIND_MASK,
            .condition = LOOM_AMDGPU_SANITIZER_RACE_HISTORY_CONDITION_INVALID,
            .condition_history_index = 0u,
            .selected_history_index = 0u,
        },
        {
            .current_access_kind_mask =
                LOOM_AMDGPU_SANITIZER_RACE_ALL_ACCESS_KIND_MASK,
            .condition = LOOM_AMDGPU_SANITIZER_RACE_HISTORY_CONDITION_INVALID,
            .condition_history_index = 1u,
            .selected_history_index = 1u,
        },
        {
            .current_access_kind_mask =
                LOOM_AMDGPU_SANITIZER_RACE_WRITE_LIKE_KIND_MASK,
            .condition =
                LOOM_AMDGPU_SANITIZER_RACE_HISTORY_CONDITION_CURRENT_WRITE_LIKE,
            .condition_history_index = 0u,
            .selected_history_index = 0u,
        },
        {
            .current_access_kind_mask =
                LOOM_AMDGPU_SANITIZER_RACE_NONWRITE_KIND_MASK,
            .condition =
                LOOM_AMDGPU_SANITIZER_RACE_HISTORY_CONDITION_SAME_WORKITEM_WRITE,
            .condition_history_index = 0u,
            .selected_history_index = 1u,
        },
        {
            .current_access_kind_mask =
                LOOM_AMDGPU_SANITIZER_RACE_NONWRITE_KIND_MASK,
            .condition =
                LOOM_AMDGPU_SANITIZER_RACE_HISTORY_CONDITION_SAME_WORKITEM_WRITE,
            .condition_history_index = 1u,
            .selected_history_index = 0u,
        },
};

#undef LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_BIT
