// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/ideogram4/session.h"

#include <cstring>

#include "experimental/id4/stages/test_util.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class SessionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    device_group_ = id4::test::CreateLocalSyncDeviceGroup();
    iree_hal_device_t* device =
        iree_hal_device_group_device_at(device_group_, /*index=*/0);
    IREE_ASSERT_OK(iree_hal_executable_cache_create(
        device, IREE_SV("id4_session_test"), &executable_cache_));

    id4_pipeline_kernel_cache_create_options_t kernel_cache_options;
    std::memset(&kernel_cache_options, 0, sizeof(kernel_cache_options));
    kernel_cache_options.structure_size = sizeof(kernel_cache_options);
    kernel_cache_options.target_processor =
        id4_pipeline_kernel_cache_default_target_processor();
    IREE_ASSERT_OK(id4_pipeline_kernel_cache_create(
        &kernel_cache_options, iree_allocator_system(), &kernel_cache_));
  }

  void TearDown() override {
    id4_pipeline_kernel_cache_release(kernel_cache_);
    iree_hal_executable_cache_release(executable_cache_);
    iree_hal_device_group_release(device_group_);
  }

  id4_ideogram4_session_create_options_t CreateOptions() {
    id4_pipeline_stage_services_t services;
    std::memset(&services, 0, sizeof(services));
    services.device_group = device_group_;
    services.executable_cache = executable_cache_;
    services.host_allocator = iree_allocator_system();

    id4_ideogram4_session_create_options_t options;
    std::memset(&options, 0, sizeof(options));
    options.structure_size = sizeof(options);
    options.services = services;
    options.kernel_cache = kernel_cache_;
    options.vae_activation_format = ID4_VAE_ACTIVATION_FORMAT_BF16_CONV_INPUT;
    return options;
  }

  iree_hal_device_group_t* device_group_ = nullptr;
  iree_hal_executable_cache_t* executable_cache_ = nullptr;
  id4_pipeline_kernel_cache_t* kernel_cache_ = nullptr;
};

TEST_F(SessionTest, LoadsOnce) {
  id4_ideogram4_session_create_options_t create_options = CreateOptions();
  id4_ideogram4_session_t* session = nullptr;
  IREE_ASSERT_OK(id4_ideogram4_session_create(
      &create_options, iree_allocator_system(), &session));

  id4_ideogram4_session_load_options_t load_options;
  std::memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  id4::test::StageDiagnostics diagnostics = {};
  id4_pipeline_diagnostics_sink_t diagnostics_sink =
      id4::test::DiagnosticsSink(&diagnostics);
  load_options.diagnostics_sink = &diagnostics_sink;
  IREE_ASSERT_OK(id4_ideogram4_session_load(session, &load_options));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        id4_ideogram4_session_load(session, &load_options));

  id4_ideogram4_session_release(session);
}

TEST_F(SessionTest, RequiresVaeActivationFormat) {
  id4_ideogram4_session_create_options_t create_options = CreateOptions();
  create_options.vae_activation_format = ID4_VAE_ACTIVATION_FORMAT_INVALID;

  id4_ideogram4_session_t* session = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_ideogram4_session_create(&create_options, iree_allocator_system(),
                                   &session));
  EXPECT_EQ(session, nullptr);
}

TEST_F(SessionTest, IssueRequiresLoadedSession) {
  id4_ideogram4_session_create_options_t create_options = CreateOptions();
  id4_ideogram4_session_t* session = nullptr;
  IREE_ASSERT_OK(id4_ideogram4_session_create(
      &create_options, iree_allocator_system(), &session));

  id4_ideogram4_qwen_issue_options_t issue_options;
  std::memset(&issue_options, 0, sizeof(issue_options));
  issue_options.structure_size = sizeof(issue_options);
  id4_ideogram4_qwen_execution_t* execution = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      id4_ideogram4_session_issue_qwen(session, &issue_options, &execution));
  EXPECT_EQ(execution, nullptr);

  id4_ideogram4_session_release(session);
}

}  // namespace
