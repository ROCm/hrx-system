// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/loop_domain.h"

#include <cstdint>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/facts.h"

namespace loom {
namespace {

class LoopDomainTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
    IREE_ASSERT_OK(loom_value_fact_table_initialize(&fact_table_, &arena_, 16));
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  void DefineFacts(loom_value_id_t value_id, loom_value_facts_t facts) {
    IREE_ASSERT_OK(loom_value_fact_table_define(&fact_table_, value_id, facts));
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
  loom_value_fact_table_t fact_table_;
};

TEST_F(LoopDomainTest, SameSsaValuesAreEqualWithoutFacts) {
  loom_loop_domain_t domain = {
      /*.lower_bound=*/1,
      /*.upper_bound=*/2,
      /*.step=*/3,
  };

  EXPECT_TRUE(loom_loop_domain_equal(nullptr, domain, domain));
}

TEST_F(LoopDomainTest, InvalidValuesAreNotEqual) {
  loom_loop_domain_t domain = {
      /*.lower_bound=*/LOOM_VALUE_ID_INVALID,
      /*.upper_bound=*/2,
      /*.step=*/3,
  };

  EXPECT_FALSE(loom_loop_domain_equal(nullptr, domain, domain));
}

TEST_F(LoopDomainTest, ExactIntegerFactsProveEquivalentValues) {
  DefineFacts(1, loom_value_facts_exact_i64(0));
  DefineFacts(2, loom_value_facts_exact_i64(16));
  DefineFacts(3, loom_value_facts_exact_i64(1));
  DefineFacts(4, loom_value_facts_exact_i64(0));
  DefineFacts(5, loom_value_facts_exact_i64(16));
  DefineFacts(6, loom_value_facts_exact_i64(1));

  loom_loop_domain_t lhs = {
      /*.lower_bound=*/1,
      /*.upper_bound=*/2,
      /*.step=*/3,
  };
  loom_loop_domain_t rhs = {
      /*.lower_bound=*/4,
      /*.upper_bound=*/5,
      /*.step=*/6,
  };

  EXPECT_TRUE(loom_loop_domain_equal(&fact_table_, lhs, rhs));
}

TEST_F(LoopDomainTest, RangeFactsDoNotProveEquality) {
  DefineFacts(1, loom_value_facts_make(0, 16, 1));
  DefineFacts(2, loom_value_facts_exact_i64(16));
  DefineFacts(3, loom_value_facts_exact_i64(1));
  DefineFacts(4, loom_value_facts_make(0, 16, 1));

  loom_loop_domain_t lhs = {
      /*.lower_bound=*/1,
      /*.upper_bound=*/2,
      /*.step=*/3,
  };
  loom_loop_domain_t rhs = {
      /*.lower_bound=*/4,
      /*.upper_bound=*/2,
      /*.step=*/3,
  };

  EXPECT_FALSE(loom_loop_domain_equal(&fact_table_, lhs, rhs));
}

TEST_F(LoopDomainTest, FloatFactsDoNotProveEquality) {
  DefineFacts(1, loom_value_facts_exact_f64(0.0));
  DefineFacts(2, loom_value_facts_exact_i64(16));
  DefineFacts(3, loom_value_facts_exact_i64(1));
  DefineFacts(4, loom_value_facts_exact_f64(0.0));

  loom_loop_domain_t lhs = {
      /*.lower_bound=*/1,
      /*.upper_bound=*/2,
      /*.step=*/3,
  };
  loom_loop_domain_t rhs = {
      /*.lower_bound=*/4,
      /*.upper_bound=*/2,
      /*.step=*/3,
  };

  EXPECT_FALSE(loom_loop_domain_equal(&fact_table_, lhs, rhs));
}

}  // namespace
}  // namespace loom
