// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/test_util.h"

#include "loom/tools/loom-check/check.h"
#include "loom/tools/loom-check/report.h"

namespace loom::testing {

static std::string StringBuilderToString(const iree_string_builder_t& builder) {
  if (builder.size == 0) {
    return std::string();
  }
  return std::string(builder.buffer, builder.size);
}

LoomCheckHarness::~LoomCheckHarness() { Deinitialize(); }

iree_status_t LoomCheckHarness::Initialize(
    const loom_check_environment_t* environment) {
  if (environment == nullptr) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "loom-check environment is required");
  }
  if (environment_ != nullptr) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "loom-check harness is already initialized");
  }

  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool_);
  block_pool_initialized_ = true;
  loom_context_initialize(iree_allocator_system(), &context_);
  context_initialized_ = true;

  iree_status_t status =
      loom_check_context_register_and_finalize(environment, &context_);
  if (iree_status_is_ok(status)) {
    environment_ = environment;
  } else {
    Deinitialize();
  }
  return status;
}

void LoomCheckHarness::Deinitialize() {
  if (context_initialized_) {
    loom_context_deinitialize(&context_);
    context_initialized_ = false;
  }
  if (block_pool_initialized_) {
    iree_arena_block_pool_deinitialize(&block_pool_);
    block_pool_initialized_ = false;
  }
  environment_ = nullptr;
}

iree_status_t LoomCheckHarness::ExecuteFirst(iree_string_view_t source,
                                             iree_string_view_t filename,
                                             loom_check_result_t* out_result) {
  if (out_result == nullptr) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "loom-check result output is required");
  }
  if (environment_ == nullptr) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "loom-check harness is not initialized");
  }

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_check_file_t file = {};
  iree_status_t status = loom_check_parse(source, &arena, &file);

  loom_check_file_report_t report = {};
  if (iree_status_is_ok(status)) {
    status = loom_check_file_report_initialize(&file, &arena, &report);
  }
  bool result_initialized = false;
  if (iree_status_is_ok(status) && file.case_count == 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "no test cases");
  }
  if (iree_status_is_ok(status)) {
    loom_check_result_initialize(iree_allocator_system(), out_result);
    result_initialized = true;
    status = loom_check_execute_case(&file.cases[0], 0, &report, filename,
                                     environment_, &context_, &block_pool_,
                                     iree_allocator_system(), out_result);
  }

  iree_arena_deinitialize(&arena);
  if (!iree_status_is_ok(status) && result_initialized) {
    loom_check_result_deinitialize(out_result);
  }
  return status;
}

std::string LoomCheckHarness::ActualOutputString(
    const loom_check_result_t& result) const {
  return StringBuilderToString(result.actual_output);
}

std::string LoomCheckHarness::DetailString(
    const loom_check_result_t& result) const {
  return StringBuilderToString(result.detail);
}

std::string LoomCheckHarness::DiagnosticJsonString(
    const loom_check_result_t& result) const {
  return StringBuilderToString(result.diagnostic_json);
}

}  // namespace loom::testing
