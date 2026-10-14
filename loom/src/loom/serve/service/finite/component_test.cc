// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <filesystem>

#include "loom/serve/service/finite/component_test_fixture.h"

namespace loom_serve::testing::finite {

TEST_F(FiniteComponentTest,
       DormantHandleDoesNotTouchMissingSourceOrCheckpoint) {
  ASSERT_TRUE(std::filesystem::remove(directory_ + "/prepare.loom"));
  ASSERT_TRUE(std::filesystem::remove(directory_ + "/weights.safetensors"));
  EXPECT_EQ(loom_serve_finite_component_phase(component_),
            LOOM_SERVE_FINITE_COMPONENT_DECLARED);
  EXPECT_FALSE(loom_serve_finite_component_advance(component_, false));
  EXPECT_FALSE(loom_serve_finite_component_is_pending(component_));
  EXPECT_EQ(loom_serve_finite_component_report(component_)->commands, 0u);
  IREE_ASSERT_OK(Trim());
  EXPECT_EQ(loom_serve_finite_component_phase(component_),
            LOOM_SERVE_FINITE_COMPONENT_DECLARED);
}

TEST_F(FiniteComponentTest, RepeatedDemandSharesOneUnactivatedPreparation) {
  loom_serve_finite_model_t* model = nullptr;
  IREE_ASSERT_OK(
      owner_.Run(loom_serve_host_executor(host_), [&](bool* complete) {
        iree_status_t status =
            loom_serve_finite_component_prepare(component_, &model);
        if (iree_status_is_ok(status) && !model) {
          status = loom_serve_finite_component_prepare(component_, &model);
        }
        *complete = model != nullptr;
        return status;
      }));
  ASSERT_NE(model, nullptr);
  const auto report = *loom_serve_finite_component_report(component_);
  EXPECT_EQ(report.commands, 2u);
  EXPECT_EQ(report.workers, 1u);
  EXPECT_FALSE(
      loom_serve_residency_snapshot(loom_serve_finite_model_residency(model))
          .active);
  EXPECT_EQ(
      loom_serve_residency_activity(loom_serve_finite_model_residency(model))
          .read_bytes,
      0u);
  loom_serve_finite_model_t* repeated = nullptr;
  IREE_ASSERT_OK(Prepare(component_, &repeated));
  EXPECT_EQ(repeated, model);
  EXPECT_EQ(loom_serve_finite_component_report(component_)->commands,
            report.commands);
  ASSERT_NO_FATAL_FAILURE(Invoke(model));
}

TEST_F(FiniteComponentTest, CompletedResultSurvivesNextInvocation) {
  loom_serve_finite_model_t* model = nullptr;
  IREE_ASSERT_OK(Prepare(component_, &model));
  loom_serve_finite_result_t first = {};
  loom_serve_finite_result_t second = {};
  ASSERT_NO_FATAL_FAILURE(Project(model, 1.0f, &first));
  ASSERT_NO_FATAL_FAILURE(Project(model, 9.0f, &second));
  ExpectResult(&first, 1.0f);
  ExpectResult(&second, 9.0f);
  EXPECT_NE(first.feedback.data, second.feedback.data);
  loom_serve_finite_result_deinitialize(&second);
  loom_serve_finite_result_deinitialize(&first);
}

TEST_F(FiniteComponentTest, PreparationFailureIsRetainedWithoutRetrying) {
  ASSERT_TRUE(std::filesystem::remove(directory_ + "/weights.safetensors"));
  loom_serve_finite_model_t* model = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, Prepare(component_, &model));
  EXPECT_EQ(model, nullptr);
  EXPECT_FALSE(loom_serve_finite_component_is_pending(component_));
  EXPECT_EQ(loom_serve_finite_component_phase(component_),
            LOOM_SERVE_FINITE_COMPONENT_FAILED);
  const auto report = *loom_serve_finite_component_report(component_);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, Prepare(component_, &model));
  EXPECT_EQ(loom_serve_finite_component_report(component_)->commands,
            report.commands);
  EXPECT_FALSE(loom_serve_finite_component_advance(component_, false));
}

TEST_F(FiniteComponentTest, InvocationFailureRetiresOnlyTheComponent) {
  loom_serve_finite_model_t* model = nullptr;
  IREE_ASSERT_OK(Prepare(component_, &model));
  bool started = false;
  IREE_ASSERT_OK(owner_.Run(
      loom_serve_host_executor(host_), [&](bool* complete) -> iree_status_t {
        if (!started) {
          loom_serve_finite_component_fail(
              component_,
              iree_make_status(IREE_STATUS_ABORTED, "fixture invocation"));
          started = true;
        } else {
          loom_serve_finite_component_advance(component_, false);
        }
        *complete = loom_serve_finite_component_phase(component_) ==
                    LOOM_SERVE_FINITE_COMPONENT_FAILED;
        return iree_ok_status();
      }));
  EXPECT_FALSE(loom_serve_finite_component_is_pending(component_));
  loom_serve_finite_model_t* unavailable = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ABORTED,
      loom_serve_finite_component_prepare(component_, &unavailable));
  EXPECT_EQ(unavailable, nullptr);
}

TEST_F(FiniteComponentTest, RemovalJoinsAcceptedPreparation) {
  bool started = false;
  IREE_ASSERT_OK(owner_.Run(
      loom_serve_host_executor(host_), [&](bool* complete) -> iree_status_t {
        if (!started) {
          loom_serve_finite_model_t* model = nullptr;
          IREE_RETURN_IF_ERROR(
              loom_serve_finite_component_prepare(component_, &model));
          EXPECT_EQ(model, nullptr);
          EXPECT_TRUE(loom_serve_finite_component_is_pending(component_));
          started = true;
        }
        return loom_serve_finite_component_retire(component_, complete);
      }));
  EXPECT_EQ(loom_serve_finite_component_phase(component_),
            LOOM_SERVE_FINITE_COMPONENT_RETIRED);
  EXPECT_FALSE(loom_serve_finite_component_is_pending(component_));
  EXPECT_EQ(loom_serve_finite_component_report(component_)->commands, 0u);
}

}  // namespace loom_serve::testing::finite
