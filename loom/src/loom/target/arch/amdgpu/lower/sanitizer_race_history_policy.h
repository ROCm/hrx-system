// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Publication policy for the AMDGPU two-history race detector.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_SANITIZER_RACE_HISTORY_POLICY_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_SANITIZER_RACE_HISTORY_POLICY_H_

#include <stdint.h>

#include "loom/target/arch/amdgpu/abi/tsan.h"

#ifdef __cplusplus
extern "C" {
#endif

// Facts about a retained history that can select a publication slot.
typedef uint32_t loom_amdgpu_sanitizer_race_history_condition_t;
enum loom_amdgpu_sanitizer_race_history_condition_e {
  // The retained history has the current workitem and compatibility class.
  LOOM_AMDGPU_SANITIZER_RACE_HISTORY_CONDITION_SAME_WORKITEM_CATEGORY = 0u,
  // The retained history is empty or belongs to an obsolete epoch/dispatch.
  LOOM_AMDGPU_SANITIZER_RACE_HISTORY_CONDITION_INVALID = 1u,
  // The current write-like access dominates a race-free history.
  LOOM_AMDGPU_SANITIZER_RACE_HISTORY_CONDITION_CURRENT_WRITE_LIKE = 2u,
  // The retained history is a write-like access by the current workitem.
  LOOM_AMDGPU_SANITIZER_RACE_HISTORY_CONDITION_SAME_WORKITEM_WRITE = 3u,
};

typedef struct loom_amdgpu_sanitizer_race_history_rule_t {
  // Bit set for each current shadow access kind to which this rule applies.
  uint32_t current_access_kind_mask;
  // History fact that makes this rule a publication candidate.
  loom_amdgpu_sanitizer_race_history_condition_t condition;
  // History slot whose facts are inspected; ignored for current-access facts.
  uint8_t condition_history_index;
  // History slot replaced when this rule is the first matching candidate.
  uint8_t selected_history_index;
} loom_amdgpu_sanitizer_race_history_rule_t;

enum loom_amdgpu_sanitizer_race_history_policy_e {
  // Number of priority-ordered publication rules.
  LOOM_AMDGPU_SANITIZER_RACE_HISTORY_RULE_COUNT = 7u,
};

// Priority-ordered publication rules for one race-free byte history.
extern const loom_amdgpu_sanitizer_race_history_rule_t
    loom_amdgpu_sanitizer_race_history_rules
        [LOOM_AMDGPU_SANITIZER_RACE_HISTORY_RULE_COUNT];

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_SANITIZER_RACE_HISTORY_POLICY_H_
