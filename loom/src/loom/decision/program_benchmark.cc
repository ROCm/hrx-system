// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Benchmarks the allocation-free ranked decision evaluator. Fixture creation
// is excluded so these rows isolate per-site classification over an immutable
// program prepared once per compiler request.

#include <cstdint>
#include <vector>

#include "benchmark/benchmark.h"
#include "loom/decision/program.h"

namespace {

enum class ProgramShape : uint8_t {
  kDecisivePrefix,
  kAllRejected,
  kAmbiguousGroup,
};

loom_decision_truth_t EvaluateFeature(void* user_data,
                                      uint32_t feature_ordinal) {
  const auto* truths = static_cast<const loom_decision_truth_t*>(user_data);
  return truths[feature_ordinal];
}

struct ProgramFixture {
  explicit ProgramFixture(uint32_t choice_count, ProgramShape shape)
      : truths(choice_count),
        choices(choice_count),
        groups(shape == ProgramShape::kAmbiguousGroup ? 1 : choice_count),
        live_actions(choice_count),
        evidence(choice_count) {
    for (uint32_t i = 0; i < choice_count; ++i) {
      truths[i] = shape == ProgramShape::kAllRejected
                      ? LOOM_DECISION_TRUTH_FALSE
                      : LOOM_DECISION_TRUTH_TRUE;
      choices[i] = {
          /*.conjunction=*/{
              /*.first_predicate=*/0,
              /*.first_feature=*/i,
              /*.predicate_count=*/0,
              /*.feature_count=*/1,
          },
          /*.action_ordinal=*/i,
      };
    }
    if (shape == ProgramShape::kDecisivePrefix) {
      for (uint32_t i = 1; i < choice_count; ++i) {
        truths[i] = LOOM_DECISION_TRUTH_FALSE;
      }
    }
    for (auto& group : groups) group.choice_count = 1;
    if (shape == ProgramShape::kAmbiguousGroup) {
      groups[0].choice_count = choice_count;
    }
    program = {
        /*.predicates=*/nullptr,
        /*.choices=*/choices.data(),
        /*.priority_groups=*/groups.data(),
        /*.constants=*/nullptr,
        /*.hard_requirements=*/{},
        /*.predicate_count=*/0,
        /*.feature_count=*/choice_count,
        /*.constant_count=*/0,
        /*.choice_count=*/choice_count,
        /*.priority_group_count=*/static_cast<uint32_t>(groups.size()),
    };
    feature_evaluator = {
        /*.fn=*/EvaluateFeature,
        /*.user_data=*/truths.data(),
    };
  }

  std::vector<loom_decision_truth_t> truths;
  std::vector<loom_decision_program_choice_t> choices;
  std::vector<loom_decision_program_priority_group_t> groups;
  std::vector<uint32_t> live_actions;
  std::vector<loom_decision_program_choice_evidence_t> evidence;
  loom_decision_program_t program = {};
  loom_decision_program_feature_evaluator_t feature_evaluator = {};
};

void RunMinimalEvaluation(
    benchmark::State& state, ProgramShape shape,
    loom_decision_program_resolution_policy_t resolution_policy) {
  const uint32_t choice_count = static_cast<uint32_t>(state.range(0));
  ProgramFixture fixture(choice_count, shape);
  for (auto _ : state) {
    uint32_t live_action_count = 0;
    loom_decision_program_result_t result = {};
    loom_decision_program_evaluate(
        &fixture.program, /*binding=*/nullptr, fixture.feature_evaluator,
        loom_decision_program_predicate_refiner_empty(), resolution_policy,
        fixture.live_actions.data(), &live_action_count, &result);
    benchmark::DoNotOptimize(result);
    benchmark::DoNotOptimize(fixture.live_actions.data());
    benchmark::DoNotOptimize(live_action_count);
  }
}

void BM_DecisivePrefix(benchmark::State& state) {
  RunMinimalEvaluation(state, ProgramShape::kDecisivePrefix,
                       LOOM_DECISION_PROGRAM_DEFER_UNRESOLVED);
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_DecisivePrefix)->Arg(1)->Arg(128)->Arg(4096);

void BM_AllRejectedEarly(benchmark::State& state) {
  RunMinimalEvaluation(state, ProgramShape::kAllRejected,
                       LOOM_DECISION_PROGRAM_DEFER_UNRESOLVED);
  state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_AllRejectedEarly)->Arg(1)->Arg(128)->Arg(4096);

void BM_AllRejectedFinal(benchmark::State& state) {
  RunMinimalEvaluation(state, ProgramShape::kAllRejected,
                       LOOM_DECISION_PROGRAM_SELECT_PROVEN);
  state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_AllRejectedFinal)->Arg(1)->Arg(128)->Arg(4096);

void BM_AmbiguousGroup(benchmark::State& state) {
  RunMinimalEvaluation(state, ProgramShape::kAmbiguousGroup,
                       LOOM_DECISION_PROGRAM_DEFER_UNRESOLVED);
  state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_AmbiguousGroup)->Arg(2)->Arg(128)->Arg(4096);

void BM_FullEvidence(benchmark::State& state) {
  const uint32_t choice_count = static_cast<uint32_t>(state.range(0));
  ProgramFixture fixture(choice_count, ProgramShape::kAllRejected);
  for (auto _ : state) {
    uint32_t live_action_count = 0;
    loom_decision_program_result_t result = {};
    loom_decision_program_evaluate_all(
        &fixture.program, /*binding=*/nullptr, fixture.feature_evaluator,
        loom_decision_program_predicate_refiner_empty(),
        LOOM_DECISION_PROGRAM_DEFER_UNRESOLVED, fixture.evidence.data(),
        fixture.live_actions.data(), &live_action_count, &result);
    benchmark::DoNotOptimize(result);
    benchmark::DoNotOptimize(fixture.evidence.data());
    benchmark::DoNotOptimize(fixture.live_actions.data());
  }
  state.SetItemsProcessed(state.iterations() * choice_count);
}
BENCHMARK(BM_FullEvidence)->Arg(1)->Arg(128)->Arg(4096);

}  // namespace
