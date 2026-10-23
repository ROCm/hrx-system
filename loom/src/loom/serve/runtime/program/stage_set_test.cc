// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/program/stage_set.h"

#include "iree/testing/status_matchers.h"
#include "loom/serve/testing/jit.h"

namespace {

class StageSetTest : public loom_serve::testing::JitTest {
 protected:
  iree_status_t CompileSet(bool private_catalog, iree_string_view_t root,
                           iree_vm_variant_span_t inputs) {
    const iree_string_view_t providers[] = {IREE_SV("config.loom"),
                                            IREE_SV("stage.loom")};
    const loom_serve_sources_t sources = {
        .base_directory = iree_make_cstring_view(directory_.c_str()),
        .count = IREE_ARRAYSIZE(providers),
        .values = providers,
    };
    IREE_RETURN_IF_ERROR(
        iree_vm_environment_allocate(host_allocator_, &environment_));
    loom_serve_compilation_report_t preparation_report = {};
    IREE_RETURN_IF_ERROR(loom_serve_preparation_allocate(
        loom_serve_host_target_environment(
            loom_serve_device_host(device_owner_)),
        environment_, &sources, IREE_SV("preparation.loom"), root,
        LOOM_SERVE_PREPARATION_FLAG_PARAMETERS, inputs,
        iree_vm_variant_span_empty(), iree_vm_module_span_empty(),
        &preparation_report, host_allocator_, &preparation_));
    auto status =
        private_catalog
            ? loom_serve_stage_set_compile_async(device_owner_, preparation_,
                                                 &sources, Complete, this,
                                                 host_allocator_)
            : loom_serve_stage_set_compile_catalog_async(
                  jit_, preparation_, Layout, Complete, this, host_allocator_);
    IREE_RETURN_IF_ERROR(status);
    // The final set owns native results independently of either source catalog.
    loom_serve_jit_release(jit_);
    jit_ = nullptr;
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return complete_; });
    status = status_;
    status_ = iree_ok_status();
    return status;
  }

  void ReleaseProgram() override {
    loom_serve_stage_set_free(stages_);
    loom_serve_preparation_free(preparation_);
    iree_vm_environment_free(environment_);
    IREE_EXPECT_OK(status_);
  }

  void CheckSet(bool private_catalog, bool fail) {
    iree_vm_variant_t input = iree_vm_variant_from_i32(fail);
    auto status = CompileSet(private_catalog, IREE_SV("prepare"), {&input, 1});
    if (fail) {
      IREE_ASSERT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
    } else {
      IREE_ASSERT_OK(status);
    }
    ASSERT_NE(stages_, nullptr);
    ASSERT_EQ(stages_->count, 3u);
    ASSERT_NE(stages_->values[0], nullptr);
    ASSERT_NE(loom_serve_jit_stage_program(stages_->values[0]), nullptr);
    EXPECT_EQ(stages_->values[1] == nullptr, fail);
    ASSERT_NE(stages_->values[2], nullptr);
    EXPECT_NE(loom_serve_jit_stage_dispatch(stages_->values[2]), nullptr);
    EXPECT_EQ(stages_->compilation.commands, 2u);
    // Direct stages compile the launch-config program as well as the native
    // entry; command children are already-specialized native requests. Only
    // identical products share a key, not roots with merely matching names.
    EXPECT_EQ(stages_->compilation.kernels, 9u);
    EXPECT_EQ(stages_->compilation.reused_kernels, fail ? 1u : 2u);
    EXPECT_EQ(stages_->compilation.entries, fail ? 9u : 11u);
    if (!private_catalog) {
      EXPECT_EQ(layouts_[0], loom_serve_jit_stage_program(stages_->values[0]));
      EXPECT_EQ(
          layouts_[1],
          fail ? nullptr : loom_serve_jit_stage_program(stages_->values[1]));
      EXPECT_EQ(layouts_[2], nullptr);
    }
  }

  // Final set, including successfully compiled stages around a failed parent.
  loom_serve_stage_set_t* stages_ = nullptr;

 private:
  static void Layout(void* user_data, iree_host_size_t index,
                     const loom_cmd_program_t* program) {
    static_cast<StageSetTest*>(user_data)->layouts_[index] = program;
  }

  static void Complete(void* user_data, iree_status_t status,
                       loom_serve_stage_set_t* stages) {
    auto* test = static_cast<StageSetTest*>(user_data);
    std::lock_guard<std::mutex> lock(test->mutex_);
    test->status_ = status;
    test->stages_ = stages;
    test->complete_ = true;
    test->condition_.notify_one();
  }

  // Preparation declaration types live until every parent has joined.
  iree_vm_environment_t* environment_ = nullptr;
  // Immutable authored declarations, borrowed through stage-set completion.
  loom_serve_preparation_t* preparation_ = nullptr;
  // Joins final callback publication with the outer test thread.
  std::mutex mutex_;
  // Outer-thread notification; compiler workers never wait on their children.
  std::condition_variable condition_;
  // Final callback has relinquished all borrowed inputs.
  bool complete_ = false;
  // Owned terminal status, consumed by CompileSet or teardown.
  iree_status_t status_ = iree_ok_status();
  // Disjoint early command views, borrowed through stage-set destruction.
  std::array<const loom_cmd_program_t*, 3> layouts_ = {};
};

TEST_P(StageSetTest, ExistingCatalogJoinsMixedStages) {
  CheckSet(false, false);
}

TEST_P(StageSetTest, PrivateCatalogJoinsMixedStages) { CheckSet(true, false); }

TEST_P(StageSetTest, ExistingCatalogJoinsEveryStageAfterFailure) {
  CheckSet(false, true);
}

TEST_P(StageSetTest, PrivateCatalogJoinsEveryStageAfterFailure) {
  CheckSet(true, true);
}

TEST_P(StageSetTest, WorkspaceRejectionRetainsEveryCompiledStage) {
  IREE_ASSERT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        CompileSet(false, IREE_SV("workspace_constrained"),
                                   iree_vm_variant_span_empty()));
  ASSERT_NE(stages_, nullptr);
  ASSERT_EQ(stages_->count, 2u);
  EXPECT_NE(stages_->values[0], nullptr);
  EXPECT_NE(stages_->values[1], nullptr);
  EXPECT_EQ(stages_->compilation.commands, 2u);
}

INSTANTIATE_TEST_SUITE_P(SharedWorkers, StageSetTest, ::testing::Values(1, 4));

}  // namespace
