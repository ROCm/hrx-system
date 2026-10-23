// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <filesystem>
#include <string>

#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/serve/runtime/program/jit.h"
#include "loomc/iree.h"
#include "loomc/target/vm.h"

IREE_FLAG(string, vm_sources, "", "VM library diagnostic source catalog.");
IREE_FLAG(string, vm_invalid_source, "", "Malformed VM source.");

namespace {

class JitVmTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_status_from_loomc(loomc_target_environment_create_vm(
        loomc_allocator_from_iree(host_allocator_), &target_environment_)));
  }

  void TearDown() override {
    loomc_target_environment_release(target_environment_);
  }

  // Owns compilation outputs independently from the temporary compiler.
  const iree_allocator_t host_allocator_ = iree_allocator_system();
  // VM compilation has no native device, task pool, or execution queue.
  loomc_target_environment_t* target_environment_ = nullptr;
};

TEST_F(JitVmTest, LibrariesPreserveMissingPrivateAndDuplicateDiagnostics) {
  struct SourceCase {
    // Entry source supplying the VM root.
    iree_string_view_t entry;
    // Number of identical library entries to supply.
    iree_host_size_t library_count;
  };
  const SourceCase cases[] = {
      {IREE_SV("control.loom"), 0},
      {IREE_SV("private_vm_helper.loom"), 1},
      {IREE_SV("control.loom"), 2},
  };
  const std::string directory =
      std::filesystem::path(FLAG_vm_sources).parent_path().string();
  const iree_string_view_t providers[] = {IREE_SV("vm_helpers.loom"),
                                          IREE_SV("vm_helpers.loom")};
  const iree_string_view_t root = IREE_SV("step");
  for (const auto& test : cases) {
    SCOPED_TRACE(test.library_count);
    const loom_serve_sources_t sources = {
        .base_directory = iree_make_cstring_view(directory.c_str()),
        .count = test.library_count,
        .values = providers,
    };
    loom_serve_compilation_report_t report = {};
    iree_const_byte_span_t image = iree_const_byte_span_empty();
    auto status =
        loom_serve_jit_compile_vm(target_environment_, &sources, test.entry, 1,
                                  &root, &report, host_allocator_, &image);
    IREE_EXPECT_NOT_OK(status);
    EXPECT_EQ(image.data, nullptr);
    EXPECT_GT(report.severities[LOOMC_DIAGNOSTIC_SEVERITY_ERROR], 0u);
    iree_allocator_free(host_allocator_, const_cast<uint8_t*>(image.data));
  }
}

TEST_F(JitVmTest, SourceFailuresRetainLocationsAndCountBeyondReportCapacity) {
  const loom_serve_sources_t sources = {.base_directory = IREE_SV(".")};
  const iree_string_view_t root = IREE_SV("broken");
  loom_serve_compilation_report_t report = {};
  const size_t attempts = IREE_ARRAYSIZE(report.diagnostics) + 1;
  for (size_t i = 0; i < attempts; ++i) {
    SCOPED_TRACE(i);
    iree_const_byte_span_t image = iree_const_byte_span_empty();
    auto status = loom_serve_jit_compile_vm(
        target_environment_, &sources,
        iree_make_cstring_view(FLAG_vm_invalid_source), 1, &root, &report,
        host_allocator_, &image);
    const auto message = iree_status_message(status);
    EXPECT_NE(std::string(message.data, message.size).find("invalid.loom"),
              std::string::npos);
    IREE_EXPECT_NOT_OK(status);
    EXPECT_EQ(image.data, nullptr);
    iree_allocator_free(host_allocator_, const_cast<uint8_t*>(image.data));
  }
  EXPECT_EQ(report.diagnostic_count, IREE_ARRAYSIZE(report.diagnostics));
  EXPECT_GE(report.severities[LOOMC_DIAGNOSTIC_SEVERITY_ERROR], attempts);
  EXPECT_EQ(report.commands, 0u);
  for (const auto& diagnostic : report.diagnostics) {
    EXPECT_GT(diagnostic.line, 0u);
    EXPECT_GT(diagnostic.source_length, 0u);
  }
}

}  // namespace
