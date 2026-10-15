// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/program/jit.h"

#include <filesystem>

#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/serve/testing/jit.h"
#if defined(__linux__)
#include "loom/serve/testing/stalled_stderr.h"
#endif

IREE_FLAG(string, invalid_sources, "", "Malformed source catalog.");

namespace {

using loom_serve::testing::Compilation;
using loom_serve::testing::JitTest;

TEST_P(JitTest, VmLibrariesPreserveMissingPrivateAndDuplicateDiagnostics) {
  struct Case {
    // Entry source supplying the VM root.
    iree_string_view_t entry;
    // Number of identical library entries to supply.
    iree_host_size_t library_count;
  };
  const Case cases[] = {
      {IREE_SV("control.loom"), 0},
      {IREE_SV("private_vm_helper.loom"), 1},
      {IREE_SV("control.loom"), 2},
  };
  const iree_string_view_t providers[] = {IREE_SV("vm_helpers.loom"),
                                          IREE_SV("vm_helpers.loom")};
  const iree_string_view_t root = IREE_SV("step");
  for (const auto& test : cases) {
    SCOPED_TRACE(test.library_count);
    const loom_serve_sources_t sources = {
        .base_directory = iree_make_cstring_view(directory_.c_str()),
        .count = test.library_count,
        .values = providers,
    };
    loom_serve_compilation_report_t report = {};
    iree_const_byte_span_t image = iree_const_byte_span_empty();
    auto status = loom_serve_jit_compile_vm(
        loom_serve_host_target_environment(
            loom_serve_device_host(device_owner_)),
        &sources, test.entry, 1, &root, &report, host_allocator_, &image);
    IREE_EXPECT_NOT_OK(status);
    EXPECT_EQ(image.data, nullptr);
    EXPECT_GT(report.severities[LOOMC_DIAGNOSTIC_SEVERITY_ERROR], 0u);
    iree_allocator_free(host_allocator_, const_cast<uint8_t*>(image.data));
  }
}

TEST_P(JitTest, SourceFailureOutlivesUnconstructedCatalogAndVm) {
#if defined(__linux__)
  loom_serve::testing::StalledStderr blocked;
  ASSERT_NO_FATAL_FAILURE(blocked.Begin());
#endif
  const std::string directory =
      std::filesystem::path(FLAG_invalid_sources).parent_path().string();
  loom_serve_jit_t* invalid = nullptr;
  loom_serve_compilation_report_t report = {};
  const iree_string_view_t providers[] = {IREE_SV("invalid.loom")};
  const loom_serve_sources_t sources = {
      .base_directory = iree_make_cstring_view(directory.c_str()),
      .count = IREE_ARRAYSIZE(providers),
      .values = providers,
  };
  auto status = loom_serve_jit_allocate(
      device_, dispatch_, loom_serve_device_host(device_owner_), &sources,
      nullptr, &report, host_allocator_, &invalid);
  EXPECT_NE(iree_status_code(status), IREE_STATUS_OK);
  EXPECT_EQ(invalid, nullptr);
  EXPECT_GT(report.severities[LOOMC_DIAGNOSTIC_SEVERITY_ERROR], 0u);
  ASSERT_GT(report.diagnostic_count, 0u);
  EXPECT_GT(report.diagnostics[0].line, 0u);
  EXPECT_GT(report.diagnostics[0].source_length, 0u);
  const auto message = iree_status_message(status);
  EXPECT_NE(std::string(message.data, message.size).find("invalid.loom"),
            std::string::npos);
  iree_status_free(status);
  loom_serve_jit_free(invalid);

  const iree_string_view_t root = IREE_SVL("broken");
  // More failures than retained slots leave complete severity counters.
  for (size_t i = 0; i < 10; ++i) {
    iree_const_byte_span_t image = iree_const_byte_span_empty();
    status =
        loom_serve_jit_compile_vm(loom_serve_host_target_environment(
                                      loom_serve_device_host(device_owner_)),
                                  &sources, IREE_SV("invalid.loom"), 1, &root,
                                  &report, host_allocator_, &image);
    IREE_EXPECT_NOT_OK(status);
    EXPECT_EQ(image.data, nullptr);
  }
  EXPECT_EQ(report.diagnostic_count, IREE_ARRAYSIZE(report.diagnostics));
  EXPECT_GE(report.severities[LOOMC_DIAGNOSTIC_SEVERITY_ERROR], 11u);
  EXPECT_EQ(report.commands, 0u);
}

TEST_P(JitTest, CompletedCatalogsRetireWithoutClosingSharedCompilerQueue) {
  // The first catalog is warm; the independent peer must compile its own copy.
  IREE_ASSERT_OK(Compile(nullptr, 0));
  iree_hal_command_buffer_release(commands_[0]);
  commands_[0] = nullptr;
  loom_serve_compilation_report_t peer_report = {};
  loom_serve_jit_t* peer = nullptr;
  auto* host = loom_serve_device_host(device_owner_);
  const iree_string_view_t providers[] = {IREE_SV("config.loom"),
                                          IREE_SV("stage.loom")};
  const loom_serve_sources_t sources = {
      .base_directory = iree_make_cstring_view(directory_.c_str()),
      .count = IREE_ARRAYSIZE(providers),
      .values = providers,
  };
  IREE_ASSERT_OK(loom_serve_jit_allocate(device_, dispatch_, host, &sources,
                                         nullptr, &peer_report, host_allocator_,
                                         &peer));
  const loomc_config_options_t config = {
      nullptr, 0, {}, LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED};
  Compilation jobs[2];
  jobs[0].jit = jit_;
  jobs[0].root = IREE_SV("selected_pair");
  jobs[1].jit = peer;
  jobs[1].root = IREE_SV("advance");
  // Both catalogs are now owned through their worker completion callbacks.
  jit_ = nullptr;
  for (auto& job : jobs) {
    job.retire_catalog = true;
    job.config = &config;
    job.Start(loom_serve_host_executor(host));
  }
  auto status = iree_ok_status();
  for (size_t i = 0; i < IREE_ARRAYSIZE(jobs); ++i) {
    loom_serve_jit_stage_t* stage = nullptr;
    auto outcome = jobs[i].Await(&stage);
    EXPECT_EQ(jobs[i].jit, nullptr);
    if (iree_status_is_ok(outcome)) {
      outcome =
          loom_serve_jit_stage_record(stage, iree_hal_queue_family(dispatch_),
                                      IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
                                      nullptr, nullptr, &commands_[i]);
    }
    status = iree_status_join(status, outcome);
    loom_serve_jit_stage_free(stage);
  }
  IREE_ASSERT_OK(status);
  EXPECT_EQ(compilation_.entries, 10u);
  EXPECT_EQ(compilation_.kernels, 8u);
  EXPECT_EQ(compilation_.reused_kernels, 2u);
  EXPECT_EQ(peer_report.entries, 8u);
  EXPECT_EQ(peer_report.kernels, 8u);
  EXPECT_EQ(peer_report.reused_kernels, 0u);
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
  const iree_hal_buffer_binding_t binding = {buffer_, 0, sizeof(input_)};
  for (auto* command : commands_) {
    IREE_ASSERT_OK(loom_serve_execution_execute(execution_, command,
                                                {1, &binding}, &completion));
  }
  iree_hal_transfer_operation_t download = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
      .download = {buffer_, 0, output_.data(), sizeof(output_)},
  };
  IREE_ASSERT_OK(
      loom_serve_execution_feedback(execution_, 1, &download, &completion));
  IREE_ASSERT_OK(loom_serve_execution_feedback_wait(execution_, completion));
  for (size_t i = 0; i < output_.size(); ++i) {
    EXPECT_EQ(output_[i], 105 + i + (i == 7 ? 24 : (i == 2 ? 7 : 0)));
  }
}

TEST_P(JitTest, InvalidConfigurationLeavesCompilerReusable) {
  IREE_ASSERT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Compile("99", 0));
  ASSERT_EQ(commands_[0], nullptr);
  IREE_ASSERT_OK(Compile("3", 0));
}

TEST_P(JitTest, InvalidRootLeavesCompilerReusable) {
  IREE_ASSERT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        Compile("3", 0, IREE_SV("absent")));
  IREE_ASSERT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        Compile("3", 0, IREE_SV("missing")));
  IREE_ASSERT_STATUS_IS(IREE_STATUS_ALREADY_EXISTS,
                        Compile("3", 0, IREE_SV("ambiguous")));
  IREE_ASSERT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Compile("3", 0, IREE_SV("not_command")));
  ASSERT_EQ(commands_[0], nullptr);
  IREE_ASSERT_OK(Compile("3", 0));
}

TEST_P(JitTest, CommandWithoutNativeRequests) {
  IREE_ASSERT_OK(Compile("3", 0, IREE_SV("idle")));
}

INSTANTIATE_TEST_SUITE_P(SharedWorkers, JitTest, ::testing::Values(1, 4));

}  // namespace
