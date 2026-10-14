// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/finite/component_test_fixture.h"

namespace loom_serve::testing::finite {

TEST_F(FiniteComponentTest, TrimReleasesBackingAndNextDemandExecutesAgain) {
  loom_serve_finite_model_t* model = nullptr;
  IREE_ASSERT_OK(Prepare(component_, &model));
  ASSERT_NO_FATAL_FAILURE(Invoke(model));
  const auto report = *loom_serve_finite_component_report(component_);
  auto* residency = loom_serve_finite_model_residency(model);
  EXPECT_EQ(loom_serve_residency_activity(residency).activations, 1u);
  EXPECT_EQ(loom_serve_residency_activity(residency).read_bytes, 4u);
  const bool elastic = loom_serve_residency_snapshot(residency).elastic;
  IREE_ASSERT_OK(Trim());
  if (elastic) {
    EXPECT_EQ(loom_serve_finite_component_phase(component_),
              LOOM_SERVE_FINITE_COMPONENT_READY);
    EXPECT_FALSE(loom_serve_residency_snapshot(residency).active);
    EXPECT_EQ(loom_serve_residency_statistics(residency).committed_bytes, 0u);
  } else {
    EXPECT_EQ(loom_serve_finite_component_phase(component_),
              LOOM_SERVE_FINITE_COMPONENT_DECLARED);
  }
  IREE_ASSERT_OK(Prepare(component_, &model));
  ASSERT_NO_FATAL_FAILURE(Invoke(model));
  residency = loom_serve_finite_model_residency(model);
  EXPECT_EQ(loom_serve_finite_component_report(component_)->commands,
            report.commands * (elastic ? 1 : 2));
  EXPECT_EQ(loom_serve_residency_activity(residency).activations,
            elastic ? 2u : 1u);
  EXPECT_EQ(loom_serve_residency_activity(residency).read_bytes,
            elastic ? 8u : 4u);
}

TEST_F(FiniteComponentTest, ComponentReleasePreservesResidentPeer) {
  loom_serve_finite_model_t* model = nullptr;
  IREE_ASSERT_OK(Prepare(component_, &model));
  ASSERT_NO_FATAL_FAILURE(Invoke(model));
  loom_serve_finite_component_t* peer = nullptr;
  IREE_ASSERT_OK(loom_serve_finite_component_allocate(
      device_, &options_, owner_.observer(), host_allocator_, &peer));
  loom_serve_finite_model_t* peer_model = nullptr;
  IREE_ASSERT_OK(Prepare(peer, &peer_model));
  ASSERT_NO_FATAL_FAILURE(Invoke(peer_model));
  auto* peer_residency = loom_serve_finite_model_residency(peer_model);
  loom_serve_residency_set_pinned(peer_residency, true);
  const auto activity = loom_serve_residency_activity(peer_residency);
  IREE_ASSERT_OK(Trim());
  EXPECT_TRUE(loom_serve_residency_snapshot(peer_residency).active);
  ASSERT_NO_FATAL_FAILURE(Invoke(peer_model));
  EXPECT_EQ(loom_serve_residency_activity(peer_residency).activations,
            activity.activations);
  EXPECT_EQ(loom_serve_residency_activity(peer_residency).read_bytes,
            activity.read_bytes);
  loom_serve_residency_set_pinned(peer_residency, false);
  IREE_ASSERT_OK(Retire(peer));
  loom_serve_finite_component_free(peer);
}

TEST_F(FiniteComponentTest, TrimPreservesExplicitRetention) {
  loom_serve_finite_model_t* model = nullptr;
  IREE_ASSERT_OK(Prepare(component_, &model));
  ASSERT_NO_FATAL_FAILURE(Invoke(model));
  auto* residency = loom_serve_finite_model_residency(model);
  loom_serve_residency_set_pinned(residency, true);
  EXPECT_TRUE(loom_serve_residency_is_pinned(residency));
  IREE_ASSERT_OK(Trim());
  EXPECT_EQ(loom_serve_finite_component_phase(component_),
            LOOM_SERVE_FINITE_COMPONENT_READY);
  EXPECT_TRUE(loom_serve_residency_snapshot(residency).active);
  loom_serve_residency_set_pinned(residency, false);
  EXPECT_FALSE(loom_serve_residency_is_pinned(residency));
  IREE_ASSERT_OK(Trim());
}

}  // namespace loom_serve::testing::finite
