// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/test/ops.h"
#include "loom/target/function_version.h"
#include "loom/target/reporting/format.h"

namespace loom {
namespace {

TEST(LoopPipelineReportTest, EntryMergeOwnsScheduleRows) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  loom_context_t context;
  loom_context_initialize(iree_allocator_system(), &context);
  iree_host_size_t count = 0;
  const loom_op_vtable_t* const* tables = loom_test_dialect_vtables(&count);
  IREE_ASSERT_OK(loom_context_register_dialect(&context, LOOM_DIALECT_TEST,
                                               tables, (uint16_t)count));
  IREE_ASSERT_OK(loom_context_finalize(&context));
  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(loom_module_allocate(&context, IREE_SV("report"), &block_pool,
                                      nullptr, iree_allocator_system(),
                                      &module));
  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &builder);
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_builder_intern_string(&builder, IREE_SV("stream"), &name_id));
  loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(loom_module_add_symbol(module, name_id, &symbol_id));
  loom_op_t* function = nullptr;
  IREE_ASSERT_OK(loom_test_func_build(
      &builder, 0, 0, 0, loom_symbol_ref_t{0, symbol_id}, nullptr, 0, nullptr,
      0, nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &function));
  loom_builder_enter_region(&builder, function, loom_test_func_body(function));
  loom_op_t* terminator = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&builder, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &terminator));

  const loom_source_loop_pipeline_operation_t operations[] = {
      {.source_position = 0,
       .op_name = IREE_SV("view.load"),
       .iteration_lookahead = 2,
       .partition = LOOM_SOURCE_LOOP_PIPELINE_PARTITION_NONE},
      {.source_position = 1,
       .op_name = IREE_SV("scf.if"),
       .iteration_lookahead = 2,
       .partition = LOOM_SOURCE_LOOP_PIPELINE_PARTITION_GUARDED},
      {.source_position = 1,
       .op_name = IREE_SV("scf.if"),
       .iteration_lookahead = 0,
       .partition = LOOM_SOURCE_LOOP_PIPELINE_PARTITION_GUARDED},
      {.source_position = 2,
       .op_name = IREE_SV("scalar.addf"),
       .iteration_lookahead = 0,
       .partition = LOOM_SOURCE_LOOP_PIPELINE_PARTITION_NONE},
  };
  loom_source_loop_pipeline_t pipeline = {};
  pipeline.depth = 3;
  pipeline.values_per_record = 2;
  pipeline.read_count = 2;
  pipeline.operations = operations;
  pipeline.operation_count = IREE_ARRAYSIZE(operations);
  loom_target_function_version_t version = {};
  version.base.type = &loom_target_function_version_type;
  version.base.function = loom_func_like_cast(module, function);
  version.loop_pipelines = {&pipeline, &pipeline};
  loom_function_version_t* handles[] = {&version.base};
  const loom_function_version_list_t versions = {handles, 1};

  loom_target_compile_report_t source;
  loom_target_compile_report_initialize(&source, iree_allocator_system());
  source.function_name = IREE_SV("stream");
  source.requested_detail_flags =
      LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS;
  IREE_ASSERT_OK(loom_target_compile_report_record_loop_pipelines(
      &source, module, &versions));

  loom_target_compile_report_t merged;
  loom_target_compile_report_initialize(&merged, iree_allocator_system());
  IREE_ASSERT_OK(
      loom_target_compile_report_record_entry_report(&merged, &source));
  loom_target_compile_report_deinitialize(&source);

  EXPECT_EQ(merged.loop_pipeline_rows.count, 1u);
  EXPECT_EQ(merged.loop_pipeline_stage_rows.count, 4u);
  for (auto mode : {LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_SUMMARY,
                    LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS}) {
    const loom_target_compile_report_format_options_t options = {mode};
    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    IREE_ASSERT_OK(
        loom_target_compile_report_format_text(&merged, &options, &builder));
    const std::string text(iree_string_builder_buffer(&builder),
                           iree_string_builder_size(&builder));
    EXPECT_NE(text.find("loop_pipeline function=stream loop=0"),
              std::string::npos);
    EXPECT_NE(
        text.find("depth=3 queue_records=2 values_per_record=2 read_count=2"),
        std::string::npos);
    if (mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
      EXPECT_NE(text.find("position=1 op=scf.if partition=guarded "
                          "stage=producer iteration_lookahead=2"),
                std::string::npos);
      EXPECT_NE(text.find("position=1 op=scf.if partition=guarded "
                          "stage=consumer iteration_lookahead=0"),
                std::string::npos);
      EXPECT_NE(text.find("position=2 op=scalar.addf stage=consumer "
                          "iteration_lookahead=0"),
                std::string::npos);
    } else {
      EXPECT_EQ(text.find("loop_pipeline_stage"), std::string::npos);
    }
    iree_string_builder_deinitialize(&builder);
  }
  loom_target_compile_report_deinitialize(&merged);
  loom_module_free(module);
  loom_context_deinitialize(&context);
  iree_arena_block_pool_deinitialize(&block_pool);
}

}  // namespace
}  // namespace loom
