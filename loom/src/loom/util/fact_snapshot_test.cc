// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/fact_snapshot.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

class FactSnapshotTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    IREE_ASSERT_OK(loom_value_fact_table_initialize(&table_, &arena_, 0));
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  loom_value_fact_auxiliary_snapshot_t Snapshot(loom_value_id_t value = 7) {
    loom_value_fact_auxiliary_snapshot_t snapshot;
    loom_value_fact_table_snapshot_auxiliary(&table_, value, &snapshot);
    return snapshot;
  }

  // Pool backing the populated scope and its retained payloads.
  iree_arena_block_pool_t pool_;
  // Scope storage that remains live across withdrawal and reinference.
  iree_arena_allocator_t arena_;
  // Publication owner; snapshots copy mutable descriptors and borrow payloads.
  loom_value_fact_table_t table_;
};

class FactOriginSnapshotTest : public FactSnapshotTest,
                               public ::testing::WithParamInterface<uint8_t> {
 protected:
  iree_status_t Publish(loom_value_id_t source) {
    switch (GetParam()) {
      case LOOM_VALUE_FACT_ORIGIN_UNIFORM_ELEMENT:
        return loom_value_fact_table_define_uniform_element_origin(
            &table_, 7, source, source);
      case LOOM_VALUE_FACT_ORIGIN_STATIC_LANE:
        return loom_value_fact_table_define_static_lane_origin(&table_, 7,
                                                               {source, 0, 2});
      case LOOM_VALUE_FACT_ORIGIN_EXACT_LANE:
        return loom_value_fact_table_define_exact_lane_origin(&table_, 7,
                                                              {source, 1, 2});
      case LOOM_VALUE_FACT_ORIGIN_EXACT_SCALAR:
        return loom_value_fact_table_define_exact_scalar_origin(&table_, 7,
                                                                {source});
      case LOOM_VALUE_FACT_ORIGIN_UNIFORM_SCALE:
        return loom_value_fact_table_define_uniform_scale_origin(&table_, 7,
                                                                 {source, 5});
      case LOOM_VALUE_FACT_ORIGIN_CONTEXTUAL_QUERY:
        return loom_value_fact_table_define_contextual_query_origin(
            &table_, 7,
            {
                .family_kind = LOOM_PARAMETERIZED_ATTR_KIND(3, 7),
                .key = loom_attr_i64(source),
            });
      case LOOM_VALUE_FACT_ORIGIN_LAYOUT: {
        const loom_value_id_t strides[] = {source, LOOM_VALUE_ID_INVALID};
        return loom_value_fact_table_define_layout_explicit_strides(&table_, 7,
                                                                    strides, 2);
      }
      default:
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
    }
  }
};

TEST_P(FactOriginSnapshotTest, WithdrawalAndReconstructionPreserveOldProof) {
  const auto empty = Snapshot();
  IREE_ASSERT_OK(Publish(1));
  const auto first = Snapshot();
  EXPECT_FALSE(loom_value_fact_auxiliary_snapshot_equal(&empty, &first));
  loom_value_fact_table_undefine(&table_, 7);
  auto current = Snapshot();
  EXPECT_TRUE(loom_value_fact_auxiliary_snapshot_equal(&empty, &current));
  EXPECT_FALSE(loom_value_fact_auxiliary_snapshot_equal(&first, &current));

  // Reconstructed payloads may occupy new addresses. Their meaning is equal.
  IREE_ASSERT_OK(Publish(1));
  current = Snapshot();
  EXPECT_TRUE(loom_value_fact_auxiliary_snapshot_equal(&first, &current));
  IREE_ASSERT_OK(Publish(2));
  current = Snapshot();
  EXPECT_FALSE(loom_value_fact_auxiliary_snapshot_equal(&first, &current));
  IREE_ASSERT_OK(Publish(1));
  current = Snapshot();
  EXPECT_TRUE(loom_value_fact_auxiliary_snapshot_equal(&first, &current));
}

INSTANTIATE_TEST_SUITE_P(
    Families, FactOriginSnapshotTest,
    ::testing::Values(LOOM_VALUE_FACT_ORIGIN_UNIFORM_ELEMENT,
                      LOOM_VALUE_FACT_ORIGIN_STATIC_LANE,
                      LOOM_VALUE_FACT_ORIGIN_EXACT_LANE,
                      LOOM_VALUE_FACT_ORIGIN_EXACT_SCALAR,
                      LOOM_VALUE_FACT_ORIGIN_UNIFORM_SCALE,
                      LOOM_VALUE_FACT_ORIGIN_CONTEXTUAL_QUERY,
                      LOOM_VALUE_FACT_ORIGIN_LAYOUT));

TEST_F(FactSnapshotTest, DefinitionScopeChangesInvalidateEqualGlobalFacts) {
  const auto empty = Snapshot();
  IREE_ASSERT_OK(loom_value_fact_table_define_uniform_scope(
      &table_, 7, LOOM_VALUE_FACT_UNIFORM_SCOPE_SUBGROUP));
  const auto subgroup = Snapshot();
  EXPECT_FALSE(loom_value_fact_auxiliary_snapshot_equal(&empty, &subgroup));
  IREE_ASSERT_OK(loom_value_fact_table_define_uniform_scope(
      &table_, 7, LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP));
  auto current = Snapshot();
  EXPECT_FALSE(loom_value_fact_auxiliary_snapshot_equal(&subgroup, &current));
  IREE_ASSERT_OK(loom_value_fact_table_define_uniform_scope(
      &table_, 7, LOOM_VALUE_FACT_UNIFORM_SCOPE_SUBGROUP));
  current = Snapshot();
  EXPECT_TRUE(loom_value_fact_auxiliary_snapshot_equal(&subgroup, &current));
  loom_value_fact_table_undefine(&table_, 7);
  current = Snapshot();
  EXPECT_TRUE(loom_value_fact_auxiliary_snapshot_equal(&empty, &current));
  EXPECT_FALSE(loom_value_fact_auxiliary_snapshot_equal(&subgroup, &current));
}

TEST_F(FactSnapshotTest, IdentitySnapshotDoesNotFollowMutableRepresentatives) {
  IREE_ASSERT_OK(loom_value_fact_table_define_identity(&table_, 8, 7, nullptr));
  const auto first = Snapshot(8);
  EXPECT_EQ(first.identity, 7u);
  // Redirecting the shared representative changes live queries immediately.
  IREE_ASSERT_OK(loom_value_fact_table_define_identity(&table_, 7, 1, nullptr));
  const auto redirected = Snapshot(8);
  EXPECT_EQ(redirected.identity, 1u);
  EXPECT_FALSE(loom_value_fact_auxiliary_snapshot_equal(&first, &redirected));
  loom_value_fact_table_undefine(&table_, 8);
  IREE_ASSERT_OK(loom_value_fact_table_define_identity(&table_, 8, 7, nullptr));
  const auto rebuilt = Snapshot(8);
  EXPECT_TRUE(loom_value_fact_auxiliary_snapshot_equal(&redirected, &rebuilt));
}

TEST_F(FactSnapshotTest, BothPartitionViewsSurviveWithdrawalAndReconstruction) {
  const loom_value_fact_vector_partition_component_t exact[] = {{1, 1}, {2, 1}};
  const loom_value_fact_vector_partition_component_t packets[] = {{3, 2}};
  const loom_value_fact_vector_partition_t partition = {exact, packets, 2, 2,
                                                        0,     1,       2};
  IREE_ASSERT_OK(loom_value_fact_table_define_vector_partition(
      &table_, 7, partition, nullptr));
  const auto first = Snapshot();
  loom_value_fact_table_undefine(&table_, 7);
  auto current = Snapshot();
  EXPECT_FALSE(loom_value_fact_auxiliary_snapshot_equal(&first, &current));
  IREE_ASSERT_OK(loom_value_fact_table_define_vector_partition(
      &table_, 7, partition, nullptr));
  current = Snapshot();
  EXPECT_NE(first.partition.components, current.partition.components);
  EXPECT_TRUE(loom_value_fact_auxiliary_snapshot_equal(&first, &current));

  // Losing only the packet view must still notify packet materialization.
  IREE_ASSERT_OK(loom_value_fact_table_define_vector_partition(
      &table_, 7, {exact, nullptr, 2, 2, 0, 0, 0}, nullptr));
  current = Snapshot();
  EXPECT_FALSE(loom_value_fact_auxiliary_snapshot_equal(&first, &current));
  // Losing only the exact view must still notify scalar materialization.
  IREE_ASSERT_OK(loom_value_fact_table_define_vector_partition(
      &table_, 7, {nullptr, packets, 0, 2, 0, 1, 2}, nullptr));
  current = Snapshot();
  EXPECT_FALSE(loom_value_fact_auxiliary_snapshot_equal(&first, &current));
}

TEST_F(FactSnapshotTest, ComputedPacketsChangeWithoutChangingComponents) {
  const loom_value_fact_vector_partition_component_t components[] = {{1, 4},
                                                                     {2, 4}};
  loom_value_fact_vector_partition_t partition = {components, components, 2, 8,
                                                  4,          2,          4};
  IREE_ASSERT_OK(loom_value_fact_table_define_vector_partition(
      &table_, 7, partition, nullptr));
  const auto structural = Snapshot();

  // The same components now retain packetized computation. Consumers must
  // revisit their materialization choice even though the geometry is unchanged.
  partition.flags = LOOM_VALUE_FACT_VECTOR_PARTITION_FLAG_COMPUTED_PACKETS;
  bool changed = false;
  IREE_ASSERT_OK(loom_value_fact_table_define_vector_partition(
      &table_, 7, partition, &changed));
  const auto computed = Snapshot();
  EXPECT_TRUE(changed);
  EXPECT_FALSE(
      loom_value_fact_auxiliary_snapshot_equal(&structural, &computed));

  loom_value_fact_table_undefine(&table_, 7);
  IREE_ASSERT_OK(loom_value_fact_table_define_vector_partition(
      &table_, 7, partition, nullptr));
  const auto reconstructed = Snapshot();
  EXPECT_TRUE(
      loom_value_fact_auxiliary_snapshot_equal(&computed, &reconstructed));
  EXPECT_FALSE(
      loom_value_fact_auxiliary_snapshot_equal(&structural, &reconstructed));
}

}  // namespace
}  // namespace loom
