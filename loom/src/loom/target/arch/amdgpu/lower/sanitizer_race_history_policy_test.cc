// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/sanitizer_race_history_policy.h"

#include <array>
#include <cstdint>
#include <optional>
#include <set>
#include <tuple>
#include <utility>

#include "iree/testing/gtest.h"

namespace {

struct Access {
  uint8_t workitem;
  loom_amdgpu_tsan_shadow_access_kind_t kind;

  bool operator==(const Access& other) const {
    return std::tie(workitem, kind) == std::tie(other.workitem, other.kind);
  }

  bool operator<(const Access& other) const {
    return std::tie(workitem, kind) < std::tie(other.workitem, other.kind);
  }
};

using CompleteHistory = std::set<Access>;
using RetainedHistory = std::array<std::optional<Access>, 2>;

struct State {
  CompleteHistory complete;
  RetainedHistory retained;

  bool operator<(const State& other) const {
    return std::tie(complete, retained) <
           std::tie(other.complete, other.retained);
  }
};

bool IsWriteLike(loom_amdgpu_tsan_shadow_access_kind_t kind) {
  return kind == LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_WRITE ||
         kind == LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_READ_WRITE;
}

bool SameCategory(const Access& lhs, const Access& rhs) {
  if (lhs.workitem != rhs.workitem) {
    return false;
  }
  return lhs.kind == rhs.kind ||
         (IsWriteLike(lhs.kind) && IsWriteLike(rhs.kind));
}

bool Conflicts(const Access& lhs, const Access& rhs) {
  if (lhs.workitem == rhs.workitem) {
    return false;
  }
  if (lhs.kind == LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_READ &&
      rhs.kind == LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_READ) {
    return false;
  }
  if (lhs.kind == LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_ATOMIC &&
      rhs.kind == LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_ATOMIC) {
    return false;
  }
  return true;
}

bool CompleteHistoryConflicts(const CompleteHistory& history,
                              const Access& current) {
  for (const Access& prior : history) {
    if (Conflicts(prior, current)) {
      return true;
    }
  }
  return false;
}

bool RetainedHistoryConflicts(const RetainedHistory& history,
                              const Access& current) {
  for (const std::optional<Access>& prior : history) {
    if (prior && Conflicts(*prior, current)) {
      return true;
    }
  }
  return false;
}

bool RuleMatches(const loom_amdgpu_sanitizer_race_history_rule_t& rule,
                 const RetainedHistory& history, const Access& current) {
  const std::optional<Access>& prior = history[rule.condition_history_index];
  switch (rule.condition) {
    case LOOM_AMDGPU_SANITIZER_RACE_HISTORY_CONDITION_SAME_WORKITEM_CATEGORY:
      return prior && SameCategory(*prior, current);
    case LOOM_AMDGPU_SANITIZER_RACE_HISTORY_CONDITION_INVALID:
      return !prior;
    case LOOM_AMDGPU_SANITIZER_RACE_HISTORY_CONDITION_CURRENT_WRITE_LIKE:
      return true;
    case LOOM_AMDGPU_SANITIZER_RACE_HISTORY_CONDITION_SAME_WORKITEM_WRITE:
      return prior && prior->workitem == current.workitem &&
             IsWriteLike(prior->kind);
  }
  return false;
}

std::optional<uint8_t> SelectPublicationSlot(
    const RetainedHistory& history, const Access& current,
    std::array<bool, LOOM_AMDGPU_SANITIZER_RACE_HISTORY_RULE_COUNT>*
        observed_rules) {
  const uint32_t current_kind_bit = 1u << current.kind;
  for (uint32_t i = 0; i < LOOM_AMDGPU_SANITIZER_RACE_HISTORY_RULE_COUNT; ++i) {
    const loom_amdgpu_sanitizer_race_history_rule_t& rule =
        loom_amdgpu_sanitizer_race_history_rules[i];
    if ((rule.current_access_kind_mask & current_kind_bit) == 0 ||
        !RuleMatches(rule, history, current)) {
      continue;
    }
    (*observed_rules)[i] = true;
    return rule.selected_history_index;
  }
  return std::nullopt;
}

TEST(AmdgpuSanitizerRaceHistoryPolicyTest,
     RetainedHistoryMatchesCompleteRaceFreeHistory) {
  constexpr std::array<loom_amdgpu_tsan_shadow_access_kind_t, 4> kKinds = {
      LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_READ,
      LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_WRITE,
      LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_READ_WRITE,
      LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_ATOMIC,
  };
  constexpr uint8_t kWorkitemCount = 4;

  const State empty_state = {};
  std::set<State> all_states = {empty_state};
  std::set<State> frontier = {empty_state};
  std::array<bool, LOOM_AMDGPU_SANITIZER_RACE_HISTORY_RULE_COUNT>
      observed_rules = {};
  bool observed_saturated_summary = false;
  uint32_t transition_count = 0;

  while (!frontier.empty()) {
    std::set<State> next_frontier;
    for (const State& state : frontier) {
      for (uint8_t workitem = 0; workitem < kWorkitemCount; ++workitem) {
        for (loom_amdgpu_tsan_shadow_access_kind_t kind : kKinds) {
          const Access current = {.workitem = workitem, .kind = kind};
          const bool complete_conflict =
              CompleteHistoryConflicts(state.complete, current);
          const bool retained_conflict =
              RetainedHistoryConflicts(state.retained, current);
          SCOPED_TRACE(::testing::Message()
                       << "transition " << transition_count << ", workitem "
                       << static_cast<uint32_t>(workitem) << ", kind " << kind);
          ASSERT_EQ(complete_conflict, retained_conflict);
          ++transition_count;
          if (complete_conflict) {
            continue;
          }

          State next = state;
          next.complete.insert(current);
          const std::optional<uint8_t> selected_slot =
              SelectPublicationSlot(state.retained, current, &observed_rules);
          if (selected_slot) {
            next.retained[*selected_slot] = current;
          } else {
            observed_saturated_summary = true;
          }
          if (all_states.insert(next).second) {
            next_frontier.insert(std::move(next));
          }
        }
      }
    }
    frontier = std::move(next_frontier);
  }

  EXPECT_TRUE(observed_saturated_summary);
  for (uint32_t i = 0; i < LOOM_AMDGPU_SANITIZER_RACE_HISTORY_RULE_COUNT; ++i) {
    EXPECT_TRUE(observed_rules[i]) << "publication rule " << i;
  }
  EXPECT_GT(all_states.size(), 1u);
  EXPECT_GT(transition_count, 0u);
}

}  // namespace
