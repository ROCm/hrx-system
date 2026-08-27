// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/index/carrier.h"

#include "iree/testing/gtest.h"
#include "loom/target/facts.h"
#include "loom/util/fact_table.h"

namespace loom {
namespace {

TEST(IndexCarrierTest, ReadsTypedTargetWidths) {
  loom_target_facts_t target_facts = {};
  target_facts.storage.snapshot.index_bitwidth = 32;
  target_facts.storage.snapshot.offset_bitwidth = 64;
  loom_fact_context_t context = {};
  context.target_facts = &target_facts;

  EXPECT_EQ(
      loom_index_target_carrier_bitwidth(&context, LOOM_SCALAR_TYPE_INDEX), 32);
  EXPECT_EQ(
      loom_index_target_carrier_bitwidth(&context, LOOM_SCALAR_TYPE_OFFSET),
      64);
}

TEST(IndexCarrierTest, DistinguishesTargetlessAndInvalidCarriers) {
  loom_fact_context_t context = {};
  EXPECT_EQ(
      loom_index_target_carrier_bitwidth(&context, LOOM_SCALAR_TYPE_INDEX), 0);
  EXPECT_EQ(loom_index_target_carrier_bitwidth(&context, LOOM_SCALAR_TYPE_I32),
            -1);

  loom_target_facts_t target_facts = {};
  target_facts.storage.snapshot.index_bitwidth = 128;
  context.target_facts = &target_facts;
  EXPECT_EQ(
      loom_index_target_carrier_bitwidth(&context, LOOM_SCALAR_TYPE_INDEX), -1);
}

}  // namespace
}  // namespace loom
