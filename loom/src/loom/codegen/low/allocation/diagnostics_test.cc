// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/diagnostics.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

iree_status_t UnexpectedEmit(void* user_data,
                             const loom_diagnostic_emission_t* emission) {
  (void)user_data;
  (void)emission;
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "unexpected allocation diagnostic emission");
}

TEST(LowAllocationDiagnosticsTest, EmptyTableEmitsNoDiagnostics) {
  loom_low_allocation_table_t table = {};
  const iree_diagnostic_emitter_t emitter = {
      /*.fn=*/UnexpectedEmit,
      /*.user_data=*/nullptr,
  };

  IREE_EXPECT_OK(loom_low_allocation_diagnostics_emit(
      &table,
      LOOM_LOW_ALLOCATION_DIAGNOSTIC_PREDICTED_SPILLS |
          LOOM_LOW_ALLOCATION_DIAGNOSTIC_COPY_DECISIONS |
          LOOM_LOW_ALLOCATION_DIAGNOSTIC_PLACEMENT_DECISIONS,
      emitter));
}

}  // namespace
}  // namespace loom
