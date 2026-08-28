// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Benchmarks allocation-free predicate proof over already resolved facts. This
// is the per-atom floor used by specialization, coverage, and search clients;
// IR traversal and predicate construction intentionally remain outside it.

#include "benchmark/benchmark.h"
#include "loom/decision/predicate.h"
#include "loom/ir/float_facts.h"

namespace {

static loom_decision_predicate_operand_t Operand(loom_value_facts_t facts) {
  return {
      /*.facts=*/facts,
      /*.identity=*/LOOM_DECISION_OPERAND_IDENTITY_NONE,
  };
}

static void BM_ExactRelation(benchmark::State& state) {
  loom_decision_predicate_operand_t operands[3] = {
      Operand(loom_value_facts_exact_i64(64)),
      Operand(loom_value_facts_exact_i64(64)),
  };
  for (auto _ : state) {
    benchmark::DoNotOptimize(operands);
    benchmark::DoNotOptimize(
        loom_decision_predicate_evaluate(LOOM_PREDICATE_EQ, operands));
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_ExactRelation);

static void BM_RangeRelation(benchmark::State& state) {
  loom_decision_predicate_operand_t operands[3] = {
      Operand(loom_value_facts_make(64, 256, 16)),
      Operand(loom_value_facts_exact_i64(32)),
  };
  for (auto _ : state) {
    benchmark::DoNotOptimize(operands);
    benchmark::DoNotOptimize(
        loom_decision_predicate_evaluate(LOOM_PREDICATE_GE, operands));
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_RangeRelation);

static void BM_Multiple(benchmark::State& state) {
  loom_decision_predicate_operand_t operands[3] = {
      Operand(loom_value_facts_make(64, 1024, 16)),
      Operand(loom_value_facts_exact_i64(16)),
  };
  for (auto _ : state) {
    benchmark::DoNotOptimize(operands);
    benchmark::DoNotOptimize(
        loom_decision_predicate_evaluate(LOOM_PREDICATE_MUL, operands));
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Multiple);

static void BM_Finite(benchmark::State& state) {
  loom_decision_predicate_operand_t operands[3] = {
      Operand(loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F32, 1.0)),
  };
  for (auto _ : state) {
    benchmark::DoNotOptimize(operands);
    benchmark::DoNotOptimize(
        loom_decision_predicate_evaluate(LOOM_PREDICATE_FINITE, operands));
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Finite);

}  // namespace
