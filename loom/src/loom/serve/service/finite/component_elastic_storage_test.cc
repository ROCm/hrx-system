// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/finite/component_test_fixture.h"

namespace loom_serve::testing::finite {

TEST_F(FiniteComponentTest,
       SharedPressurePreservesPinsAndReactivatesIdleImage) {
  loom_serve_finite_model_t* model = nullptr;
  IREE_ASSERT_OK(Prepare(component_, &model));
  ASSERT_NO_FATAL_FAILURE(Invoke(model));
  auto* residency = loom_serve_finite_model_residency(model);
  ASSERT_TRUE(loom_serve_residency_snapshot(residency).elastic);
  const auto report = *loom_serve_finite_component_report(component_);

  loom_serve_finite_component_t* peer = nullptr;
  IREE_ASSERT_OK(loom_serve_finite_component_allocate(
      device_, &options_, owner_.observer(), host_allocator_, &peer));
  loom_serve_finite_model_t* peer_model = nullptr;
  IREE_ASSERT_OK(Prepare(peer, &peer_model));
  ASSERT_NO_FATAL_FAILURE(Invoke(peer_model));
  auto* peer_residency = loom_serve_finite_model_residency(peer_model);
  EXPECT_EQ(loom_serve_residency_cache(residency),
            loom_serve_residency_cache(peer_residency));
  loom_serve_residency_set_pinned(peer_residency, true);
  const auto peer_activity = loom_serve_residency_activity(peer_residency);

  // Device-wide reclamation uses the same registered parameter groups as
  // capacity admission. Neither component needs its own cache or trim call.
  IREE_ASSERT_OK(loom_serve_device_trim(device_, 0));
  EXPECT_FALSE(loom_serve_residency_snapshot(residency).active);
  EXPECT_EQ(loom_serve_residency_statistics(residency).committed_bytes, 0u);
  EXPECT_TRUE(loom_serve_residency_snapshot(peer_residency).active);
  EXPECT_EQ(loom_serve_finite_component_phase(component_),
            LOOM_SERVE_FINITE_COMPONENT_READY);
  ASSERT_NO_FATAL_FAILURE(Invoke(model));
  ASSERT_NO_FATAL_FAILURE(Invoke(peer_model));
  EXPECT_EQ(loom_serve_finite_component_report(component_)->commands,
            report.commands);
  EXPECT_EQ(loom_serve_residency_activity(residency).activations, 2u);
  EXPECT_EQ(loom_serve_residency_activity(residency).read_bytes, 8u);
  EXPECT_EQ(loom_serve_residency_activity(peer_residency).activations,
            peer_activity.activations);

  loom_serve_residency_set_pinned(peer_residency, false);
  IREE_ASSERT_OK(loom_serve_device_trim(device_, 0));
  EXPECT_FALSE(loom_serve_residency_snapshot(residency).active);
  EXPECT_FALSE(loom_serve_residency_snapshot(peer_residency).active);
  IREE_ASSERT_OK(Retire(peer));
  loom_serve_finite_component_free(peer);
}

}  // namespace loom_serve::testing::finite
