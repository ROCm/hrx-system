// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/pass/pipeline_snapshot.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/module.h"
#include "loom/ops/pass/ops.h"
#include "loom/pass/test/harness.h"

namespace loom {
namespace {

class PassPipelineSnapshotTest : public PassTestHarness {};

TEST_F(PassPipelineSnapshotTest, MaterializesSelectedPipelineClosure) {
  loom_module_t* source_module =
      Parse(IREE_SV("pass.pipeline<module> @selected pipeline {\n"
                    "  call @callee\n"
                    "}\n"
                    "pass.pipeline<module> @callee pipeline {\n"
                    "  test.module-noop\n"
                    "}\n"
                    "pass.pipeline<module> @unrelated pipeline {\n"
                    "  test.module-noop\n"
                    "}\n"));
  ASSERT_NE(source_module, nullptr);

  loom_pass_pipeline_snapshot_t snapshot = {};
  IREE_ASSERT_OK(loom_pass_pipeline_snapshot_initialize(
      source_module, IREE_SV("@@selected"), IREE_SV("pipeline_snapshot"), {},
      block_pool(), iree_allocator_system(), &snapshot));
  ASSERT_NE(snapshot.module, nullptr);
  ASSERT_NE(snapshot.pipeline_op, nullptr);
  EXPECT_TRUE(loom_pass_pipeline_isa(snapshot.pipeline_op));
  const loom_string_id_t selected_name_id =
      loom_module_lookup_string(snapshot.module, IREE_SV("selected"));
  ASSERT_NE(selected_name_id, LOOM_STRING_ID_INVALID);
  const loom_symbol_id_t selected_symbol_id =
      loom_module_find_symbol(snapshot.module, selected_name_id);
  ASSERT_NE(selected_symbol_id, LOOM_SYMBOL_ID_INVALID);
  EXPECT_EQ(snapshot.module->symbols.entries[selected_symbol_id].defining_op,
            snapshot.pipeline_op);
  EXPECT_EQ(snapshot.module->symbols.count, 2u);
  const loom_string_id_t unrelated_name_id =
      loom_module_lookup_string(snapshot.module, IREE_SV("unrelated"));
  EXPECT_TRUE(unrelated_name_id == LOOM_STRING_ID_INVALID ||
              loom_module_find_symbol(snapshot.module, unrelated_name_id) ==
                  LOOM_SYMBOL_ID_INVALID);

  loom_pass_pipeline_snapshot_deinitialize(&snapshot);
}

TEST_F(PassPipelineSnapshotTest, RejectsMissingPipelineSymbol) {
  loom_module_t* source_module =
      Parse(IREE_SV("pass.pipeline<module> @available pipeline {\n"
                    "  test.module-noop\n"
                    "}\n"));
  ASSERT_NE(source_module, nullptr);

  loom_pass_pipeline_snapshot_t snapshot = {};
  IREE_EXPECT_OK(loom_pass_pipeline_snapshot_initialize(
      source_module, IREE_SV("@missing"), IREE_SV("pipeline_snapshot"), {},
      block_pool(), iree_allocator_system(), &snapshot));
  EXPECT_EQ(snapshot.module, nullptr);
  EXPECT_EQ(snapshot.pipeline_op, nullptr);
}

}  // namespace
}  // namespace loom
