// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test helper for executing .loom-test snippets through a caller-selected
// loom-check environment.

#ifndef LOOM_TOOLS_LOOM_CHECK_TEST_UTIL_H_
#define LOOM_TOOLS_LOOM_CHECK_TEST_UTIL_H_

#include <string>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/context.h"
#include "loom/tools/loom-check/execute.h"

namespace loom::testing {

class LoomCheckHarness {
 public:
  LoomCheckHarness() = default;
  LoomCheckHarness(const LoomCheckHarness&) = delete;
  LoomCheckHarness& operator=(const LoomCheckHarness&) = delete;

  ~LoomCheckHarness();

  // Initializes the parser/execution state using |environment|. The
  // environment and its callback state are borrowed and must outlive the
  // harness.
  iree_status_t Initialize(const loom_check_environment_t* environment);

  // Releases context and arena storage. Safe to call repeatedly.
  void Deinitialize();

  // Executes the first case from |source|. The caller owns |out_result| on
  // success and must release it with loom_check_result_deinitialize().
  iree_status_t ExecuteFirst(iree_string_view_t source,
                             iree_string_view_t filename,
                             loom_check_result_t* out_result);

  // Returns the actual comparable output captured in |result|.
  std::string ActualOutputString(const loom_check_result_t& result) const;

  // Returns the human-readable failure detail captured in |result|.
  std::string DetailString(const loom_check_result_t& result) const;

  // Returns the structured diagnostic JSON captured in |result|.
  std::string DiagnosticJsonString(const loom_check_result_t& result) const;

 private:
  // Arena block pool backing parsed check files and modules.
  iree_arena_block_pool_t block_pool_ = {};
  // Parser and verifier context selected by |environment_|.
  loom_context_t context_ = {};
  // Borrowed loom-check execution environment.
  const loom_check_environment_t* environment_ = nullptr;
  // True after |block_pool_| has been initialized.
  bool block_pool_initialized_ = false;
  // True after |context_| has been initialized.
  bool context_initialized_ = false;
};

}  // namespace loom::testing

#endif  // LOOM_TOOLS_LOOM_CHECK_TEST_UTIL_H_
