// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/decision/program.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <random>

#include "iree/testing/gtest.h"

namespace {

loom_decision_truth_t EvaluateFeature(void* user_data,
                                      uint32_t feature_ordinal) {
  const auto* truths = static_cast<const loom_decision_truth_t*>(user_data);
  return truths[feature_ordinal];
}

uint64_t LiveMask(const uint32_t* actions, uint32_t action_count) {
  uint64_t mask = 0;
  for (uint32_t i = 0; i < action_count; ++i) {
    mask |= UINT64_C(1) << actions[i];
  }
  return mask;
}

TEST(DecisionProgramTest, SelectsScalarPredicateChoiceFromSsaFacts) {
  const int64_t constants[] = {128, 32};
  const loom_decision_program_predicate_t predicates[] = {
      {
          /*.kind=*/LOOM_PREDICATE_GE,
          /*.operand_count=*/2,
          /*.reserved=*/{}, /*.operands=*/
          {
              loom_decision_program_argument_ref(0),
              loom_decision_program_constant_ref(0),
          },
      },
      {
          /*.kind=*/LOOM_PREDICATE_MUL,
          /*.operand_count=*/2,
          /*.reserved=*/{}, /*.operands=*/
          {
              loom_decision_program_argument_ref(1),
              loom_decision_program_constant_ref(1),
          },
      },
  };
  const loom_decision_program_choice_t choices[] = {
      {
          /*.conjunction=*/
          {
              /*.first_predicate=*/0,
              /*.first_feature=*/{},
              /*.predicate_count=*/2,
          },
          /*.action_ordinal=*/7,
      },
      {
          /*.conjunction=*/{},
          /*.action_ordinal=*/9,
      },
  };
  const loom_decision_program_priority_group_t groups[] = {
      {/*.choice_count=*/1},
      {/*.choice_count=*/1},
  };
  const loom_decision_program_t program = {
      /*.predicates=*/predicates,
      /*.choices=*/choices,
      /*.priority_groups=*/groups,
      /*.constants=*/constants,
      /*.hard_requirements=*/{},   /*.predicate_count=*/2,
      /*.feature_count=*/{},       /*.constant_count=*/2,
      /*.choice_count=*/2,
      /*.priority_group_count=*/2,
  };

  std::array<loom_value_facts_t, 16> fact_entries = {};
  fact_entries[3] = loom_value_facts_exact_i64(128);
  fact_entries[11] = loom_value_facts_exact_i64(64);
  const loom_value_fact_table_t fact_table = {
      /*.arena=*/{},
      /*.transient_arena=*/{},
      /*.entries=*/fact_entries.data(),
      /*.count=*/{},
      /*.capacity=*/fact_entries.size(),
  };
  const loom_value_id_t arguments[] = {3, 11};
  const loom_decision_program_binding_t binding = {
      /*.facts=*/&fact_table,
      /*.argument_values=*/arguments,
  };

  uint32_t live_actions[2] = {};
  uint32_t live_action_count = 0;
  loom_decision_program_result_t result = {};
  loom_decision_program_evaluate(
      &program, &binding, /*feature_evaluator=*/{},
      loom_decision_program_predicate_refiner_empty(),
      LOOM_DECISION_PROGRAM_DEFER_UNRESOLVED, live_actions, &live_action_count,
      &result);
  EXPECT_EQ(result.kind, LOOM_DECISION_PROGRAM_RESULT_SELECTED);
  EXPECT_EQ(result.action_ordinal, 7u);
  ASSERT_EQ(live_action_count, 1u);
  EXPECT_EQ(live_actions[0], 7u);

  fact_entries[3] = loom_value_facts_exact_i64(64);
  loom_decision_program_evaluate(
      &program, &binding, /*feature_evaluator=*/{},
      loom_decision_program_predicate_refiner_empty(),
      LOOM_DECISION_PROGRAM_DEFER_UNRESOLVED, live_actions, &live_action_count,
      &result);
  EXPECT_EQ(result.kind, LOOM_DECISION_PROGRAM_RESULT_SELECTED);
  EXPECT_EQ(result.action_ordinal, 9u);
  ASSERT_EQ(live_action_count, 1u);
  EXPECT_EQ(live_actions[0], 9u);
}

TEST(DecisionProgramTest, PreservesSameSsaIdentityAcrossSignaturePositions) {
  const loom_decision_program_predicate_t predicate = {
      /*.kind=*/LOOM_PREDICATE_EQ,
      /*.operand_count=*/2,
      /*.reserved=*/{}, /*.operands=*/
      {
          loom_decision_program_argument_ref(0),
          loom_decision_program_argument_ref(1),
      },
  };
  const loom_decision_program_choice_t choice = {
      /*.conjunction=*/
      {
          /*.first_predicate=*/{},
          /*.first_feature=*/{},
          /*.predicate_count=*/1,
      },
      /*.action_ordinal=*/0,
  };
  const loom_decision_program_priority_group_t group = {/*.choice_count=*/1};
  const loom_decision_program_t program = {
      /*.predicates=*/&predicate,
      /*.choices=*/&choice,
      /*.priority_groups=*/&group,
      /*.constants=*/{},           /*.hard_requirements=*/{},
      /*.predicate_count=*/1,
      /*.feature_count=*/{},       /*.constant_count=*/{},
      /*.choice_count=*/1,
      /*.priority_group_count=*/1,
  };
  loom_value_facts_t unknown_entry = loom_value_facts_unknown();
  const loom_value_fact_table_t fact_table = {
      /*.arena=*/{},
      /*.transient_arena=*/{},
      /*.entries=*/&unknown_entry,
      /*.count=*/{},
      /*.capacity=*/1,
  };
  const loom_value_id_t arguments[] = {0, 0};
  const loom_decision_program_binding_t binding = {
      /*.facts=*/&fact_table,
      /*.argument_values=*/arguments,
  };

  uint32_t live_action = UINT32_MAX;
  uint32_t live_action_count = 0;
  loom_decision_program_result_t result = {};
  loom_decision_program_evaluate(
      &program, &binding, /*feature_evaluator=*/{},
      loom_decision_program_predicate_refiner_empty(),
      LOOM_DECISION_PROGRAM_DEFER_UNRESOLVED, &live_action, &live_action_count,
      &result);
  EXPECT_EQ(result.kind, LOOM_DECISION_PROGRAM_RESULT_SELECTED);
  EXPECT_EQ(result.action_ordinal, 0u);
}

TEST(DecisionProgramTest, ResolutionPolicyControlsHigherUnknownChoice) {
  loom_decision_truth_t feature_truths[] = {
      LOOM_DECISION_TRUTH_UNKNOWN,
      LOOM_DECISION_TRUTH_TRUE,
  };
  const loom_decision_program_choice_t choices[] = {
      {
          /*.conjunction=*/{/*.first_predicate=*/{}, /*.first_feature=*/0,
                            /*.predicate_count=*/{}, /*.feature_count=*/1},
          /*.action_ordinal=*/3,
      },
      {
          /*.conjunction=*/{/*.first_predicate=*/{}, /*.first_feature=*/1,
                            /*.predicate_count=*/{}, /*.feature_count=*/1},
          /*.action_ordinal=*/5,
      },
  };
  const loom_decision_program_priority_group_t groups[] = {
      {/*.choice_count=*/1},
      {/*.choice_count=*/1},
  };
  const loom_decision_program_t program = {
      /*.predicates=*/{},          /*.choices=*/choices,
      /*.priority_groups=*/groups,
      /*.constants=*/{},           /*.hard_requirements=*/{},
      /*.predicate_count=*/{},     /*.feature_count=*/2,
      /*.constant_count=*/{},      /*.choice_count=*/2,
      /*.priority_group_count=*/2,
  };
  const loom_decision_program_feature_evaluator_t feature_evaluator = {
      /*.fn=*/EvaluateFeature,
      /*.user_data=*/feature_truths,
  };

  uint32_t live_actions[2] = {};
  uint32_t live_action_count = 0;
  loom_decision_program_result_t result = {};
  loom_decision_program_evaluate(
      &program, /*binding=*/nullptr, feature_evaluator,
      loom_decision_program_predicate_refiner_empty(),
      LOOM_DECISION_PROGRAM_DEFER_UNRESOLVED, live_actions, &live_action_count,
      &result);
  EXPECT_EQ(result.kind, LOOM_DECISION_PROGRAM_RESULT_UNRESOLVED);
  EXPECT_EQ(result.unresolved_action_ordinal, 3u);
  EXPECT_EQ(LiveMask(live_actions, live_action_count),
            (UINT64_C(1) << 3) | (UINT64_C(1) << 5));

  loom_decision_program_evaluate(
      &program, /*binding=*/nullptr, feature_evaluator,
      loom_decision_program_predicate_refiner_empty(),
      LOOM_DECISION_PROGRAM_SELECT_PROVEN, /*live_action_ordinals=*/nullptr,
      &live_action_count, &result);
  EXPECT_EQ(result.kind, LOOM_DECISION_PROGRAM_RESULT_SELECTED);
  EXPECT_EQ(result.action_ordinal, 5u);
  EXPECT_EQ(live_action_count, 1u);
}

TEST(DecisionProgramTest, HardUnknownPreservesEveryPossibleChoice) {
  loom_decision_truth_t feature_truths[] = {
      LOOM_DECISION_TRUTH_UNKNOWN,
      LOOM_DECISION_TRUTH_FALSE,
      LOOM_DECISION_TRUTH_UNKNOWN,
      LOOM_DECISION_TRUTH_TRUE,
  };
  const loom_decision_program_choice_t choices[] = {
      {
          /*.conjunction=*/{/*.first_predicate=*/{}, /*.first_feature=*/1,
                            /*.predicate_count=*/{}, /*.feature_count=*/1},
          /*.action_ordinal=*/0,
      },
      {
          /*.conjunction=*/{/*.first_predicate=*/{}, /*.first_feature=*/2,
                            /*.predicate_count=*/{}, /*.feature_count=*/1},
          /*.action_ordinal=*/1,
      },
      {
          /*.conjunction=*/{/*.first_predicate=*/{}, /*.first_feature=*/3,
                            /*.predicate_count=*/{}, /*.feature_count=*/1},
          /*.action_ordinal=*/2,
      },
  };
  const loom_decision_program_priority_group_t groups[] = {
      {/*.choice_count=*/1},
      {/*.choice_count=*/1},
      {/*.choice_count=*/1},
  };
  const loom_decision_program_t program = {
      /*.predicates=*/{},
      /*.choices=*/choices,
      /*.priority_groups=*/groups,
      /*.constants=*/{},
      /*.hard_requirements=*/
      {/*.first_predicate=*/{}, /*.first_feature=*/0, /*.predicate_count=*/{},
       /*.feature_count=*/1},
      /*.predicate_count=*/{},
      /*.feature_count=*/4,
      /*.constant_count=*/{},
      /*.choice_count=*/3,
      /*.priority_group_count=*/3,
  };
  const loom_decision_program_feature_evaluator_t feature_evaluator = {
      /*.fn=*/EvaluateFeature,
      /*.user_data=*/feature_truths,
  };

  uint32_t live_actions[3] = {};
  uint32_t live_action_count = 0;
  loom_decision_program_result_t result = {};
  loom_decision_program_evaluate(
      &program, /*binding=*/nullptr, feature_evaluator,
      loom_decision_program_predicate_refiner_empty(),
      LOOM_DECISION_PROGRAM_DEFER_UNRESOLVED, live_actions, &live_action_count,
      &result);
  EXPECT_EQ(result.kind, LOOM_DECISION_PROGRAM_RESULT_UNRESOLVED);
  EXPECT_EQ(LiveMask(live_actions, live_action_count),
            (UINT64_C(1) << 1) | (UINT64_C(1) << 2));
}

TEST(DecisionProgramTest, FullEvaluationCapturesLowerChoiceEvidence) {
  loom_decision_truth_t feature_truths[] = {
      LOOM_DECISION_TRUTH_TRUE,
      LOOM_DECISION_TRUTH_FALSE,
      LOOM_DECISION_TRUTH_UNKNOWN,
  };
  const loom_decision_program_choice_t choices[] = {
      {
          /*.conjunction=*/{/*.first_predicate=*/{}, /*.first_feature=*/0,
                            /*.predicate_count=*/{}, /*.feature_count=*/1},
          /*.action_ordinal=*/0,
      },
      {
          /*.conjunction=*/{/*.first_predicate=*/{}, /*.first_feature=*/1,
                            /*.predicate_count=*/{}, /*.feature_count=*/1},
          /*.action_ordinal=*/1,
      },
      {
          /*.conjunction=*/{/*.first_predicate=*/{}, /*.first_feature=*/2,
                            /*.predicate_count=*/{}, /*.feature_count=*/1},
          /*.action_ordinal=*/2,
      },
  };
  const loom_decision_program_priority_group_t groups[] = {
      {/*.choice_count=*/1},
      {/*.choice_count=*/1},
      {/*.choice_count=*/1},
  };
  const loom_decision_program_t program = {
      /*.predicates=*/{},          /*.choices=*/choices,
      /*.priority_groups=*/groups,
      /*.constants=*/{},           /*.hard_requirements=*/{},
      /*.predicate_count=*/{},     /*.feature_count=*/3,
      /*.constant_count=*/{},      /*.choice_count=*/3,
      /*.priority_group_count=*/3,
  };
  const loom_decision_program_feature_evaluator_t feature_evaluator = {
      /*.fn=*/EvaluateFeature,
      /*.user_data=*/feature_truths,
  };

  loom_decision_program_choice_evidence_t evidence[3] = {};
  uint32_t live_actions[3] = {};
  uint32_t live_action_count = 0;
  loom_decision_program_result_t result = {};
  loom_decision_program_evaluate_all(
      &program, /*binding=*/nullptr, feature_evaluator,
      loom_decision_program_predicate_refiner_empty(),
      LOOM_DECISION_PROGRAM_DEFER_UNRESOLVED, evidence, live_actions,
      &live_action_count, &result);
  EXPECT_EQ(result.kind, LOOM_DECISION_PROGRAM_RESULT_SELECTED);
  EXPECT_EQ(result.action_ordinal, 0u);
  EXPECT_EQ(evidence[0].feasibility, LOOM_DECISION_TRUTH_TRUE);
  EXPECT_EQ(evidence[1].feasibility, LOOM_DECISION_TRUTH_FALSE);
  EXPECT_EQ(evidence[2].feasibility, LOOM_DECISION_TRUTH_UNKNOWN);
}

struct ReferenceChoice {
  int64_t priority;
  loom_decision_truth_t truth;
};

struct ReferenceResult {
  loom_decision_program_result_kind_t kind;
  uint32_t action_ordinal;
  uint32_t unresolved_action_ordinal;
  uint64_t live_mask;
};

ReferenceResult EvaluateReference(
    loom_decision_truth_t hard_truth, const ReferenceChoice* choices,
    uint32_t choice_count,
    loom_decision_program_resolution_policy_t resolution_policy) {
  ReferenceResult result = {
      /*.kind=*/LOOM_DECISION_PROGRAM_RESULT_NO_MATCH,
      /*.action_ordinal=*/LOOM_DECISION_PROGRAM_ACTION_INVALID,
      /*.unresolved_action_ordinal=*/LOOM_DECISION_PROGRAM_ACTION_INVALID,
  };
  if (hard_truth == LOOM_DECISION_TRUTH_FALSE) {
    result.kind = LOOM_DECISION_PROGRAM_RESULT_HARD_REJECT;
    return result;
  }
  if (hard_truth == LOOM_DECISION_TRUTH_UNKNOWN) {
    result.kind = LOOM_DECISION_PROGRAM_RESULT_UNRESOLVED;
    if (resolution_policy == LOOM_DECISION_PROGRAM_DEFER_UNRESOLVED) {
      for (uint32_t i = 0; i < choice_count; ++i) {
        if (choices[i].truth != LOOM_DECISION_TRUTH_FALSE) {
          result.live_mask |= UINT64_C(1) << i;
        }
      }
    }
    return result;
  }

  bool has_match = false;
  bool has_maybe = false;
  int64_t best_match_priority = INT64_MIN;
  int64_t highest_maybe_priority = INT64_MIN;
  uint32_t best_match_action = LOOM_DECISION_PROGRAM_ACTION_INVALID;
  uint32_t highest_maybe_action = LOOM_DECISION_PROGRAM_ACTION_INVALID;
  uint32_t best_match_count = 0;
  for (uint32_t i = 0; i < choice_count; ++i) {
    if (choices[i].truth == LOOM_DECISION_TRUTH_UNKNOWN) {
      has_maybe = true;
      if (highest_maybe_action == LOOM_DECISION_PROGRAM_ACTION_INVALID ||
          choices[i].priority > highest_maybe_priority) {
        highest_maybe_priority = choices[i].priority;
        highest_maybe_action = i;
      }
      continue;
    }
    if (choices[i].truth != LOOM_DECISION_TRUTH_TRUE) continue;
    if (!has_match || choices[i].priority > best_match_priority) {
      has_match = true;
      best_match_priority = choices[i].priority;
      best_match_action = i;
      best_match_count = 1;
    } else if (choices[i].priority == best_match_priority) {
      ++best_match_count;
    }
  }

  if (resolution_policy == LOOM_DECISION_PROGRAM_DEFER_UNRESOLVED &&
      has_maybe &&
      (!has_match || highest_maybe_priority >= best_match_priority)) {
    result.kind = LOOM_DECISION_PROGRAM_RESULT_UNRESOLVED;
    result.unresolved_action_ordinal = highest_maybe_action;
    for (uint32_t i = 0; i < choice_count; ++i) {
      if (choices[i].truth == LOOM_DECISION_TRUTH_FALSE) continue;
      if (has_match && choices[i].truth == LOOM_DECISION_TRUTH_TRUE &&
          choices[i].priority != best_match_priority) {
        continue;
      }
      if (has_match && choices[i].truth == LOOM_DECISION_TRUTH_UNKNOWN &&
          choices[i].priority < best_match_priority) {
        continue;
      }
      result.live_mask |= UINT64_C(1) << i;
    }
    return result;
  }
  if (!has_match) {
    if (has_maybe) {
      result.kind = LOOM_DECISION_PROGRAM_RESULT_UNRESOLVED;
      result.unresolved_action_ordinal = highest_maybe_action;
    }
    if (resolution_policy == LOOM_DECISION_PROGRAM_DEFER_UNRESOLVED) {
      for (uint32_t i = 0; i < choice_count; ++i) {
        if (!has_maybe || choices[i].truth != LOOM_DECISION_TRUTH_FALSE) {
          result.live_mask |= UINT64_C(1) << i;
        }
      }
    }
    return result;
  }
  if (best_match_count > 1) {
    result.kind = LOOM_DECISION_PROGRAM_RESULT_AMBIGUOUS;
    for (uint32_t i = 0; i < choice_count; ++i) {
      if (choices[i].truth == LOOM_DECISION_TRUTH_TRUE &&
          choices[i].priority == best_match_priority) {
        result.live_mask |= UINT64_C(1) << i;
      }
    }
    return result;
  }
  result.kind = LOOM_DECISION_PROGRAM_RESULT_SELECTED;
  result.action_ordinal = best_match_action;
  result.live_mask = UINT64_C(1) << best_match_action;
  return result;
}

TEST(DecisionProgramTest, RandomRankedSemanticsAgreeWithReference) {
  std::mt19937_64 random(0xdec1510);
  std::array<ReferenceChoice, 64> reference_choices;
  std::array<loom_decision_truth_t, 65> truths;
  std::array<loom_decision_program_choice_t, 64> choices;
  std::array<loom_decision_program_priority_group_t, 64> groups;
  std::array<uint32_t, 64> live_actions;
  std::array<uint32_t, 64> full_live_actions;
  std::array<loom_decision_program_choice_evidence_t, 64> choice_evidence;

  for (uint32_t iteration = 0; iteration < 100000; ++iteration) {
    const uint32_t choice_count = 1 + random() % 64;
    truths[0] = static_cast<loom_decision_truth_t>(random() % 3);
    for (uint32_t i = 0; i < choice_count; ++i) {
      reference_choices[i] = {
          /*.priority=*/static_cast<int64_t>(random() % 9) - 4,
          /*.truth=*/static_cast<loom_decision_truth_t>(random() % 3),
      };
    }
    std::stable_sort(
        reference_choices.begin(), reference_choices.begin() + choice_count,
        [](const ReferenceChoice& lhs, const ReferenceChoice& rhs) {
          return lhs.priority > rhs.priority;
        });

    uint32_t group_count = 0;
    for (uint32_t i = 0; i < choice_count; ++i) {
      truths[i + 1] = reference_choices[i].truth;
      choices[i] = {
          /*.conjunction=*/
          {
              /*.first_predicate=*/{},
              /*.first_feature=*/i + 1,
              /*.predicate_count=*/{},
              /*.feature_count=*/1,
          },
          /*.action_ordinal=*/i,
      };
      if (i == 0 ||
          reference_choices[i].priority != reference_choices[i - 1].priority) {
        groups[group_count++] = {/*.choice_count=*/1};
      } else {
        ++groups[group_count - 1].choice_count;
      }
    }
    const loom_decision_program_t program = {
        /*.predicates=*/{},
        /*.choices=*/choices.data(),
        /*.priority_groups=*/groups.data(),
        /*.constants=*/{},
        /*.hard_requirements=*/
        {/*.first_predicate=*/{}, /*.first_feature=*/0, /*.predicate_count=*/{},
         /*.feature_count=*/1},
        /*.predicate_count=*/{},
        /*.feature_count=*/choice_count + 1,
        /*.constant_count=*/{},
        /*.choice_count=*/choice_count,
        /*.priority_group_count=*/group_count,
    };
    const loom_decision_program_feature_evaluator_t feature_evaluator = {
        /*.fn=*/EvaluateFeature,
        /*.user_data=*/truths.data(),
    };
    for (const loom_decision_program_resolution_policy_t policy : {
             LOOM_DECISION_PROGRAM_DEFER_UNRESOLVED,
             LOOM_DECISION_PROGRAM_SELECT_PROVEN,
         }) {
      uint32_t live_action_count = 0;
      loom_decision_program_result_t actual = {};
      loom_decision_program_evaluate(
          &program, /*binding=*/nullptr, feature_evaluator,
          loom_decision_program_predicate_refiner_empty(), policy,
          live_actions.data(), &live_action_count, &actual);
      const ReferenceResult expected = EvaluateReference(
          truths[0], reference_choices.data(), choice_count, policy);
      ASSERT_EQ(actual.kind, expected.kind) << "iteration " << iteration;
      ASSERT_EQ(actual.action_ordinal, expected.action_ordinal)
          << "iteration " << iteration;
      ASSERT_EQ(actual.unresolved_action_ordinal,
                expected.unresolved_action_ordinal)
          << "iteration " << iteration;
      ASSERT_EQ(LiveMask(live_actions.data(), live_action_count),
                expected.live_mask)
          << "iteration " << iteration;

      uint32_t full_live_action_count = 0;
      loom_decision_program_result_t full_result = {};
      loom_decision_program_evaluate_all(
          &program, /*binding=*/nullptr, feature_evaluator,
          loom_decision_program_predicate_refiner_empty(), policy,
          choice_evidence.data(), full_live_actions.data(),
          &full_live_action_count, &full_result);
      ASSERT_EQ(full_result.kind, expected.kind) << "iteration " << iteration;
      ASSERT_EQ(full_result.action_ordinal, expected.action_ordinal)
          << "iteration " << iteration;
      ASSERT_EQ(full_result.unresolved_action_ordinal,
                expected.unresolved_action_ordinal)
          << "iteration " << iteration;
      ASSERT_EQ(full_result.unresolved_constraint, actual.unresolved_constraint)
          << "iteration " << iteration;
      ASSERT_EQ(LiveMask(full_live_actions.data(), full_live_action_count),
                expected.live_mask)
          << "iteration " << iteration;
      const bool choices_were_evaluated =
          truths[0] == LOOM_DECISION_TRUTH_TRUE ||
          (truths[0] == LOOM_DECISION_TRUTH_UNKNOWN &&
           policy == LOOM_DECISION_PROGRAM_DEFER_UNRESOLVED);
      if (choices_were_evaluated) {
        for (uint32_t i = 0; i < choice_count; ++i) {
          ASSERT_EQ(choice_evidence[i].feasibility, reference_choices[i].truth)
              << "iteration " << iteration << ", choice " << i;
        }
      }
    }
  }
}

}  // namespace
