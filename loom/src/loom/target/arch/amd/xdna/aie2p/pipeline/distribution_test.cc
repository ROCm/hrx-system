// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/distribution.h"

#include <utility>

#include "iree/testing/gtest.h"

namespace {

class DistributionReuseTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Binding occurrences may have the same source value but still own
    // distinct storage and synchronization. Identity is the retained object.
    for (auto& identity : identities_) {
      identity.value_id = 17;
    }
    destinations_[0][0].channel = &identities_[0];
    destinations_[0][1].channel = &identities_[1];
    destinations_[1][0].channel = &identities_[2];
    destinations_[1][1].channel = &identities_[1];
    for (size_t i = 0; i < 2; ++i) {
      records_[i].destinations = destinations_[i];
      records_[i].destination_count = 2;
      records_[i].admissions = 1;
      records_[i].transfers = 2;
      batches_[i].records = &records_[i];
      batches_[i].record_count = 1;
      batches_[i].replays = 2;
      distributions_[i].source = &batches_[i];
      distributions_[i].transport = &transport_;
      distributions_[i].sender.channel = &source_;
      distributions_[i].receiver_count = 2;
      distributions_[i].repetitions = 2;
      distributions_[i].path = &distributions_[0];
    }
    distributions_[0].next = &distributions_[1];
    owner_.distributions = &distributions_[0];
  }

  auto* Reuse(const loom_pipeline_transport_batch_t& batch,
              uint32_t repetitions = 2) {
    return loom_aie2p_native_reusable_distribution(
        &owner_, &source_, &transport_, &batch, repetitions);
  }

  // Distinct receiver protocols despite equal source value IDs.
  loom_channel_identity_t identities_[3] = {};
  // Two destination sets on one already admitted physical route.
  loom_pipeline_transport_destination_t destinations_[2][2] = {};
  // Canonical cohort facts for each descriptor geometry.
  loom_pipeline_transport_cohort_t records_[2] = {};
  // Retained source lifetimes consumed by geometry selection.
  loom_pipeline_transport_batch_t batches_[2] = {};
  // Shared transport facts; these uniform batches need no loop table.
  loom_pipeline_transport_t transport_ = {};
  // Native source protocol, shared across phases.
  loom_aie2p_native_channel_t source_ = {};
  // Two descriptor sets whose source and transfer count are identical.
  loom_aie2p_native_distribution_t distributions_[2] = {};
  // Ordered transport owner retaining both geometries.
  loom_aie2p_native_worker_t owner_ = {};
};

TEST_F(DistributionReuseTest,
       ReturningToEachDestinationSetSelectsItsDescriptors) {
  EXPECT_EQ(Reuse(batches_[0]), &distributions_[0]);
  EXPECT_EQ(Reuse(batches_[1]), &distributions_[1]);
  EXPECT_EQ(Reuse(batches_[0]), &distributions_[0]);
  // A later occurrence has its own cohort storage but the same protocols.
  auto records = records_[1];
  auto batch = batches_[1];
  batch.records = &records;
  EXPECT_EQ(Reuse(batch), &distributions_[1]);
}

TEST_F(DistributionReuseTest, ChangedTransferCountNeedsDifferentDescriptors) {
  auto records = records_[0];
  records.transfers = 4;
  auto batch = batches_[0];
  batch.records = &records;
  batch.replays = 4;
  EXPECT_EQ(Reuse(batch, 4), nullptr);
  EXPECT_EQ(Reuse(batches_[0]), &distributions_[0]);
}

TEST_F(DistributionReuseTest, SourceStorageIdentityIsPartOfGeometry) {
  auto another_source = source_;
  EXPECT_EQ(loom_aie2p_native_reusable_distribution(
                &owner_, &another_source, &transport_, &batches_[0], 2),
            nullptr);
}

TEST_F(DistributionReuseTest, ReceiverCountAndOrderArePartOfCircuitGeometry) {
  auto records = records_[0];
  auto batch = batches_[0];
  batch.records = &records;
  records.destination_count = 1;
  EXPECT_EQ(Reuse(batch), nullptr);
  records.destination_count = 2;
  loom_pipeline_transport_destination_t reversed[] = {destinations_[0][1],
                                                      destinations_[0][0]};
  records.destinations = reversed;
  EXPECT_EQ(Reuse(batch), nullptr);
}

TEST_F(DistributionReuseTest, NewOwnerHasNoReusableGeometry) {
  owner_.distributions = nullptr;
  EXPECT_EQ(Reuse(batches_[0]), nullptr);
}

TEST_F(DistributionReuseTest, PacketReusePreservesCopyOrderAndRepetition) {
  owner_.distribution_packets = true;
  owner_.distributions->next = nullptr;
  distributions_[0].repetitions = 0;
  loom_pipeline_transport_copy_t copies[2] = {};
  for (size_t i = 0; i < 2; ++i) {
    copies[i].source = &records_[0];
    copies[i].destination = &destinations_[0][i];
    copies[i].executions = 2;
  }
  batches_[0].copies = copies;
  batches_[0].copy_count = 2;
  auto batch = batches_[0];
  loom_pipeline_transport_copy_t reordered[] = {copies[0], copies[1]};
  batch.copies = reordered;
  EXPECT_EQ(Reuse(batch), &distributions_[0]);
  std::swap(reordered[0], reordered[1]);
  EXPECT_EQ(Reuse(batch), nullptr);
  std::swap(reordered[0], reordered[1]);
  auto records = records_[0];
  records.transfers = 4;
  batch.records = &records;
  batch.replays = 4;
  for (auto& copy : reordered) {
    copy.source = &records;
    copy.executions = 4;
  }
  EXPECT_EQ(Reuse(batch, 4), nullptr);
}

}  // namespace
