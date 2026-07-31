// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/capability_facts.h"

#include "iree/testing/gtest.h"
#include "loom/target/facts.h"
#include "loom/target/types.h"
#include "loom/util/fact_table.h"

namespace loom {
namespace {

TEST(TargetCapabilityFactsTest, QueriesGenericTargetSnapshotValues) {
  loom_target_snapshot_t snapshot = {};
  snapshot.default_pointer_bitwidth = 64;
  snapshot.index_bitwidth = 32;
  snapshot.offset_bitwidth = 64;
  snapshot.max_workgroup_size.x = 1024;
  snapshot.max_workgroup_size.y = 8;
  snapshot.max_workgroup_size.z = 1;
  snapshot.max_flat_workgroup_size = 1024;
  snapshot.max_workgroup_storage_bytes = 65536;
  snapshot.subgroup_size = 32;
  snapshot.max_grid_size.x = 65535;
  snapshot.max_grid_size.y = 65535;
  snapshot.max_grid_size.z = 1;
  snapshot.max_flat_grid_size = 4294967295ull;
  snapshot.max_workgroup_count.x = 4096;
  snapshot.max_workgroup_count.y = 1024;
  snapshot.max_workgroup_count.z = 1;

  loom_target_facts_t target_facts = {};
  target_facts.storage.snapshot = snapshot;
  loom_fact_context_t context = {};
  context.target_facts = &target_facts;

  uint64_t value = 0;
  EXPECT_TRUE(loom_target_fact_context_query_u64(
      &context, IREE_SV("target"), IREE_SV("subgroup_size"), &value));
  EXPECT_EQ(value, 32ull);
  EXPECT_TRUE(loom_target_fact_context_query_u64(
      &context, IREE_SV("target"), IREE_SV("max_workgroup_size_x"), &value));
  EXPECT_EQ(value, 1024ull);
  EXPECT_TRUE(loom_target_fact_context_query_u64(
      &context, IREE_SV("target"), IREE_SV("max_workgroup_storage_bytes"),
      &value));
  EXPECT_EQ(value, 65536ull);
}

TEST(TargetCapabilityFactsTest, OmitsUnknownAndTargetFamilyValues) {
  loom_target_snapshot_t snapshot = {};
  snapshot.index_bitwidth = 64;
  loom_target_facts_t target_facts = {};
  target_facts.storage.snapshot = snapshot;
  loom_fact_context_t context = {};
  context.target_facts = &target_facts;

  uint64_t value = 99;
  EXPECT_FALSE(loom_target_fact_context_query_u64(
      &context, IREE_SV("target"), IREE_SV("subgroup_size"), &value));
  EXPECT_EQ(value, 0ull);
  EXPECT_FALSE(loom_target_fact_context_query_u64(
      &context, IREE_SV("amdgpu"), IREE_SV("wavefront_32"), &value));
  EXPECT_EQ(value, 0ull);
}

}  // namespace
}  // namespace loom
