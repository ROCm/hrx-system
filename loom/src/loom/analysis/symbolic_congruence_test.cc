// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/symbolic_congruence.h"

#include <algorithm>
#include <cstdint>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

class SymbolicCongruenceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    loom_symbolic_expr_context_initialize(nullptr, nullptr, nullptr, &arena_,
                                          &context_);
  }
  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  loom_symbolic_expr_t Variable(loom_value_id_t identity) {
    loom_symbolic_term_t* term = nullptr;
    IREE_CHECK_OK(iree_arena_allocate(&arena_, sizeof(*term),
                                      reinterpret_cast<void**>(&term)));
    *term = {1, identity, identity};
    loom_symbolic_expr_t expression = {};
    expression.terms = term;
    expression.term_count = 1;
    expression.facts = loom_value_facts_unknown();
    expression.flags = LOOM_SYMBOLIC_EXPR_FLAG_LINEAR;
    return expression;
  }

  loom_symbolic_expr_t Shift(loom_symbolic_expr_t input, int64_t offset) {
    loom_symbolic_expr_t constant;
    loom_symbolic_expr_constant(offset, &constant);
    IREE_CHECK_OK(loom_symbolic_expr_add(&context_, &input, &constant, &input));
    return input;
  }

  loom_symbolic_expr_t Remainder(loom_symbolic_expr_t input, uint64_t modulus) {
    auto output = Variable(next_identity_++);
    IREE_CHECK_OK(
        loom_symbolic_congruence_restrict(&context_, &input, modulus, &output));
    return output;
  }

  loom_symbolic_expr_t Bank(loom_value_id_t phase, int64_t delta,
                            uint64_t period, int64_t stride, bool wrap) {
    auto value = Shift(Variable(phase), delta);
    if (wrap) {
      value = Remainder(value, 256);
    }
    value = Remainder(value, period);
    IREE_CHECK_OK(
        loom_symbolic_expr_mul_i64(&context_, &value, stride, &value));
    return value;
  }

  // Distinct identities for the exact results of nonlinear operations.
  loom_value_id_t next_identity_ = 10;
  // Allocator for producer-owned expression payloads.
  iree_arena_block_pool_t pool_;
  // One analysis lifetime, including returned normalized forms.
  iree_arena_allocator_t arena_;
  // Arithmetic owner; no module or producer traversal is needed by these tests.
  loom_symbolic_expr_context_t context_;
};

TEST_F(SymbolicCongruenceTest, BankProofsAgreeWithConcreteAddressSets) {
  uint32_t proved_count = 0;
  for (uint64_t period : {2, 3, 4, 8}) {
    for (int64_t stride : {4, 16, 32}) {
      for (int64_t delta = 0; delta <= 8; ++delta) {
        for (bool wrap : {false, true}) {
          const auto left = Bank(0, 0, period, stride, wrap);
          const auto right = Bank(0, delta, period, stride, wrap);
          for (int64_t width : {stride - 1, stride, stride + 1}) {
            const bool proved = loom_symbolic_congruence_excludes_difference(
                &right, &left, 1 - width, width - 1);
            if (!proved) {
              continue;
            }
            ++proved_count;
            for (uint64_t phase = 0; phase < (wrap ? 256 : period); ++phase) {
              const int64_t left_begin = (phase % period) * stride;
              const uint64_t advanced =
                  wrap ? (phase + delta) % 256 : phase + delta;
              const int64_t right_begin = (advanced % period) * stride;
              ASSERT_GE(std::max(left_begin, right_begin),
                        std::min(left_begin, right_begin) + width)
                  << period << ":" << stride << ":" << delta << ":" << phase;
            }
          }
        }
      }
    }
  }
  EXPECT_GT(proved_count, 0u);
}

TEST_F(SymbolicCongruenceTest, FormSelectsTheExpressionTraversedByProofs) {
  const auto direct = Variable(0);
  EXPECT_EQ(loom_symbolic_congruence_form(&direct), &direct);

  const auto periodic = Remainder(direct, 4);
  ASSERT_NE(periodic.congruence, nullptr);
  EXPECT_EQ(loom_symbolic_congruence_form(&periodic),
            &periodic.congruence->expression);
}

TEST_F(SymbolicCongruenceTest, CorrelationRequiresTheSameEvaluation) {
  const auto current = Bank(0, 0, 2, 32768, false);
  const auto next = Bank(0, 1, 2, 32768, false);
  const auto independent = Bank(1, 1, 2, 32768, false);
  EXPECT_TRUE(loom_symbolic_congruence_excludes_difference(&next, &current,
                                                           -32239, 32239));
  EXPECT_FALSE(loom_symbolic_congruence_excludes_difference(
      &independent, &current, -32239, 32239));
  EXPECT_FALSE(loom_symbolic_congruence_excludes_difference(&next, &next,
                                                            -32239, 32239));
  EXPECT_FALSE(loom_symbolic_congruence_excludes_difference(&next, &current,
                                                            -32783, 32783));
}

TEST_F(SymbolicCongruenceTest, SignedBoundaryArithmeticIsConservative) {
  auto low = Variable(0), high = Variable(0);
  low.constant = INT64_MIN;
  high.constant = INT64_MAX;
  EXPECT_FALSE(loom_symbolic_congruence_excludes_difference(&high, &low, 0, 0));
  auto periodic = Remainder(Variable(1), UINT64_C(1) << 63);
  periodic = Shift(periodic, -1);
  auto zero = Variable(1);
  EXPECT_TRUE(
      loom_symbolic_congruence_excludes_difference(&periodic, &zero, 0, 0));
  EXPECT_FALSE(loom_symbolic_congruence_excludes_difference(
      &periodic, &zero, INT64_MIN, INT64_MAX));
}

}  // namespace
}  // namespace loom
