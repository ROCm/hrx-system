// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/testing/jit_module.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/sync.h"
#include "loom/serve/runtime/program/module.h"
#if defined(__linux__)
#include "loom/serve/testing/stalled_stderr.h"
#endif

namespace {

using loom_serve::testing::Compilation;
using loom_serve::testing::JitModuleTest;

TEST_P(JitModuleTest, SourceDefaultsAndOverridesShareVmAndGpuState) {
  // The same exact-output witness runs with a sink unable to accept any byte.
#if defined(__linux__)
  loom_serve::testing::StalledStderr blocked;
  ASSERT_NO_FATAL_FAILURE(blocked.Begin());
#endif
  // A failed native task must join its accepted siblings before compiler reuse.
  IREE_ASSERT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Compile("5", 0, IREE_SV("invalid_native")));
  ASSERT_EQ(commands_[0], nullptr);
  EXPECT_EQ(compilation_.kernels, 9u);
  EXPECT_EQ(compilation_.reused_kernels, 0u);
  EXPECT_GT(compilation_.severities[LOOMC_DIAGNOSTIC_SEVERITY_ERROR], 0u);
  ASSERT_GT(compilation_.diagnostic_count, 0u);
  EXPECT_GT(compilation_.diagnostics[0].message_length, 0u);
  // Successful siblings survive a parent failure; the failed input is retried.
  IREE_ASSERT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Compile("5", 0, IREE_SV("invalid_native")));
  EXPECT_EQ(compilation_.kernels, 10u);
  EXPECT_EQ(compilation_.reused_kernels, 8u);
  IREE_ASSERT_OK(Compile(nullptr, 0));
  EXPECT_EQ(compilation_.kernels, 10u);
  EXPECT_EQ(compilation_.reused_kernels, 16u);
  IREE_ASSERT_OK(Compile("7", 1));
  EXPECT_EQ(compilation_.kernels, 18u);
  EXPECT_EQ(compilation_.reused_kernels, 16u);
  IREE_ASSERT_OK(PrepareVm());
  // Prepared commands must outlive all compiler/source storage.
  loom_serve_jit_free(jit_);
  jit_ = nullptr;
  EXPECT_EQ(compilation_.commands, 4u);
  EXPECT_EQ(compilation_.entries, 16u);
  EXPECT_EQ(compilation_.workers, GetParam());
  EXPECT_GT(compilation_.duration_ns, 0u);
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      loom_serve_device_storage_pool(device_owner_),
      iree_hal_make_pool_reservation_request(sizeof(input_), 0),
      iree_infinite_timeout(), &buffer_));
  iree_hal_transfer_operation_t upload = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
  };
  upload.upload.source = input_.data();
  upload.upload.target_buffer = buffer_;
  upload.upload.length = sizeof(input_);
  uint64_t completion = 0;
  IREE_ASSERT_OK(
      loom_serve_execution_transfer(execution_, 1, &upload, &completion));
  iree_vm_variant_t arguments[] = {
      iree_hal_buffer_variant_from_ptr_borrowed(&types_, buffer_)};
  iree_vm_variant_t results[1] = {};
  auto status = iree_vm_invoke(invocation_, step_,
                               iree_vm_variant_span_from_array(arguments),
                               iree_vm_variant_span_from_array(results));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  IREE_ASSERT_OK(status);
  const iree_hal_buffer_binding_t binding = {buffer_, 0, sizeof(input_)};
  IREE_ASSERT_OK(loom_serve_execution_execute(execution_, commands_[1],
                                              {1, &binding}, &completion));
  iree_hal_transfer_operation_t download = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
  };
  download.download.source_buffer = buffer_;
  download.download.target = output_.data();
  download.download.length = sizeof(output_);
  IREE_ASSERT_OK(
      loom_serve_execution_feedback(execution_, 1, &download, &completion));
  IREE_ASSERT_OK(loom_serve_execution_feedback_wait(execution_, completion));
  for (size_t i = 0; i < output_.size(); ++i) {
    EXPECT_EQ(output_[i], 112 + 2 * i) << "specialized element " << i;
  }
}

TEST_P(JitModuleTest, RebindingChangesWorkspaceWithoutRelinkingVm) {
  IREE_ASSERT_OK(Compile("5", 0, IREE_SV("temporary_advance"), "64"));
  IREE_ASSERT_OK(PrepareVm());
  // This candidate did not exist when the original workspace was registered.
  IREE_ASSERT_OK(Compile("7", 1, IREE_SV("temporary_advance"), "4160"));
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      loom_serve_device_storage_pool(device_owner_),
      iree_hal_make_pool_reservation_request(sizeof(input_), 0),
      iree_infinite_timeout(), &buffer_));
  const iree_hal_transfer_operation_t upload = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
      .upload = {input_.data(), buffer_, 0, sizeof(input_)},
  };
  uint64_t completion = 0;
  IREE_ASSERT_OK(
      loom_serve_execution_transfer(execution_, 1, &upload, &completion));
  for (size_t i = 0; i < 2; ++i) {
    const loom_serve_stage_t wrong_arity = {commands_[i], 2, transients_[i]};
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        loom_serve_module_bind_stages(native_, 1, &wrong_arity));
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          loom_serve_module_bind_stages(native_, 0, nullptr));
    if (i) {
      const loom_serve_stage_t replacement = {commands_[i], 1, transients_[i]};
      const loom_serve_stage_t expanded[] = {replacement, replacement,
                                             replacement};
      IREE_ASSERT_OK(loom_serve_module_bind_stages(
          native_, IREE_ARRAYSIZE(expanded), expanded));
      IREE_ASSERT_OK(loom_serve_module_bind_stages(native_, 1, &replacement));
    }
    iree_vm_variant_t arguments[] = {
        iree_hal_buffer_variant_from_ptr_borrowed(&types_, buffer_)};
    iree_vm_variant_t results[1] = {};
    auto status = iree_vm_invoke(invocation_, step_,
                                 iree_vm_variant_span_from_array(arguments),
                                 iree_vm_variant_span_from_array(results));
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
    IREE_ASSERT_OK(status);
    // The module and submitted work retain code independently of this owner.
    iree_hal_command_buffer_release(commands_[i]);
    commands_[i] = nullptr;
  }
  const iree_hal_transfer_operation_t download = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
      .download = {buffer_, 0, output_.data(), sizeof(output_)},
  };
  IREE_ASSERT_OK(
      loom_serve_execution_feedback(execution_, 1, &download, &completion));
  IREE_ASSERT_OK(loom_serve_execution_feedback_wait(execution_, completion));
  for (size_t i = 0; i < output_.size(); ++i) {
    EXPECT_EQ(output_[i], 112);
  }
}

TEST_P(JitModuleTest, DirectStagesReuseExactInputsAndOutliveCatalog) {
  const loomc_config_binding_t binding = {
      loomc_make_cstring_view("increment.delta"), loomc_make_cstring_view("7")};
  for (size_t i = 0; i < 3; ++i) {
    const loomc_config_options_t config = {
        i == 2 ? &binding : nullptr,
        i == 2 ? 1u : 0u,
        {},
        LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED};
    Compilation job;
    job.jit = jit_;
    job.root = IREE_SV("increment_0");
    job.config = &config;
    job.direct_dispatch = true;
    if (i == 2) {
      job.retire_catalog = true;
      jit_ = nullptr;
    }
    job.Start(loom_serve_host_executor(loom_serve_device_host(device_owner_)));
    loom_serve_jit_stage_t* stage = nullptr;
    IREE_ASSERT_OK(job.Await(&stage));
    EXPECT_EQ(loom_serve_jit_stage_program(stage), nullptr);
    auto* dispatch = loom_serve_jit_stage_dispatch(stage);
    if (i == 1) {
      EXPECT_EQ(dispatch, dispatches_[0]);
    } else {
      dispatches_[i / 2] = dispatch;
      loom_serve_dispatch_retain(dispatch);
    }
    loom_serve_jit_stage_free(stage);
  }
  EXPECT_EQ(compilation_.commands, 0u);
  EXPECT_EQ(compilation_.kernels, 2u);
  EXPECT_EQ(compilation_.reused_kernels, 1u);
  EXPECT_EQ(compilation_.entries, 3u);
  IREE_ASSERT_OK(PrepareVm());
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      loom_serve_device_storage_pool(device_owner_),
      iree_hal_make_pool_reservation_request(sizeof(input_), 0),
      iree_infinite_timeout(), &buffer_));
  iree_hal_transfer_operation_t upload = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
      .upload = {input_.data(), buffer_, 0, sizeof(input_)},
  };
  uint64_t completion = 0;
  IREE_ASSERT_OK(
      loom_serve_execution_transfer(execution_, 1, &upload, &completion));
  for (size_t i = 0; i < 2; ++i) {
    if (i) {
      const loom_serve_stage_t replacement[] = {
          {.binding_count = 1, .dispatch = dispatches_[1]},
          {.binding_count = 1, .dispatch = dispatches_[0]},
      };
      IREE_ASSERT_OK(loom_serve_module_bind_stages(
          native_, IREE_ARRAYSIZE(replacement), replacement));
      // Dropping caller ownership cannot retire selected native executables.
      for (auto*& dispatch : dispatches_) {
        loom_serve_dispatch_release(dispatch);
        dispatch = nullptr;
      }
    }
    iree_vm_variant_t arguments[] = {
        iree_hal_buffer_variant_from_ptr_borrowed(&types_, buffer_)};
    iree_vm_variant_t results[1] = {};
    auto status = iree_vm_invoke(invocation_, step_,
                                 iree_vm_variant_span_from_array(arguments),
                                 iree_vm_variant_span_from_array(results));
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
    IREE_ASSERT_OK(status);
  }
  iree_hal_transfer_operation_t download = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
      .download = {buffer_, 0, output_.data(), sizeof(output_)},
  };
  IREE_ASSERT_OK(
      loom_serve_execution_feedback(execution_, 1, &download, &completion));
  IREE_ASSERT_OK(loom_serve_execution_feedback_wait(execution_, completion));
  for (size_t i = 0; i < output_.size(); ++i) {
    EXPECT_EQ(output_[i], i ? 100 : 112);
  }
}

INSTANTIATE_TEST_SUITE_P(SharedWorkers, JitModuleTest, ::testing::Values(1, 4));

}  // namespace
