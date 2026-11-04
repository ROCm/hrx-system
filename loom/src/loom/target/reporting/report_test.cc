// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/reporting/format.h"

namespace loom {
namespace {

TEST(CompileReportTest, RetainsAndIndexesDirectCallEvidence) {
  loom_target_compile_report_t report;
  loom_target_compile_report_initialize(&report, iree_allocator_system());
  const loom_symbol_ref_t caller = {.module_id = 0, .symbol_id = 3};
  const loom_symbol_ref_t callee = {.module_id = 0, .symbol_id = 7};
  const loom_target_compile_report_call_edge_t first = {
      .flags = LOOM_TARGET_COMPILE_REPORT_CALL_EDGE_DYNAMIC_COUNT,
      .function_name = IREE_SVL("entry"),
      .function = caller,
      .callee_name = IREE_SVL("helper"),
      .callee = callee,
      .callsite_count = 1,
      .dynamic_call_count = 4,
  };
  IREE_ASSERT_OK(loom_target_compile_report_record_call_edge(&report, &first));

  loom_target_compile_report_call_edge_t second = first;
  second.callsite_count = 2;
  second.dynamic_call_count = 6;
  IREE_ASSERT_OK(loom_target_compile_report_record_call_edge(&report, &second));

  const loom_target_compile_report_call_edge_t* edge =
      loom_target_compile_report_find_call_edge(&report, caller, callee);
  ASSERT_NE(edge, nullptr);
  EXPECT_EQ(edge->callsite_count, 3u);
  EXPECT_EQ(edge->dynamic_call_count, 10u);
  EXPECT_TRUE(iree_all_bits_set(
      edge->flags, LOOM_TARGET_COMPILE_REPORT_CALL_EDGE_DYNAMIC_COUNT));

  loom_target_compile_report_call_edge_t unknown = first;
  unknown.flags = LOOM_TARGET_COMPILE_REPORT_CALL_EDGE_NONE;
  unknown.dynamic_call_count = 0;
  IREE_ASSERT_OK(
      loom_target_compile_report_record_call_edge(&report, &unknown));
  edge = loom_target_compile_report_find_call_edge(&report, caller, callee);
  ASSERT_NE(edge, nullptr);
  EXPECT_EQ(edge->callsite_count, 4u);
  EXPECT_FALSE(iree_any_bit_set(
      edge->flags, LOOM_TARGET_COMPILE_REPORT_CALL_EDGE_DYNAMIC_COUNT));
  EXPECT_EQ(edge->dynamic_call_count, 0u);

  loom_target_compile_report_deinitialize(&report);
}

TEST(CompileReportTest, MergesFunctionEvidenceIntoPhysicalProgram) {
  loom_target_compile_report_t program;
  loom_target_compile_report_initialize(&program, iree_allocator_system());
  program.requested_detail_flags =
      LOOM_TARGET_COMPILE_REPORT_DETAIL_CONFIG_BINDING_ROWS;
  loom_target_compile_report_record_schedule(&program, 2, 2, 1, 0, 1, 0, 0, 0,
                                             0);
  const loom_target_compile_report_static_instruction_mix_t program_mix = {
      .descriptor_count = 2,
      .vector_alu_count = 1,
  };
  loom_target_compile_report_record_static_instruction_mix(&program,
                                                           &program_mix);
  loom_target_compile_report_record_dynamic_instruction_mix(&program,
                                                            &program_mix);
  loom_target_compile_report_record_emission(&program, 8, 32, 40);
  loom_target_compile_report_record_memory(&program, 16, 32);
  const loom_target_compile_report_workload_t workload = {
      .flags = LOOM_TARGET_COMPILE_REPORT_WORKLOAD_FLAT_WORKGROUP_SIZE,
      .flat_workgroup_size = 64,
  };
  loom_target_compile_report_record_workload(&program, &workload);

  loom_target_compile_report_t function;
  loom_target_compile_report_initialize(&function, iree_allocator_system());
  function.requested_detail_flags =
      LOOM_TARGET_COMPILE_REPORT_DETAIL_CONFIG_BINDING_ROWS;
  loom_target_compile_report_record_schedule(&function, 3, 3, 2, 0, 2, 0, 0, 1,
                                             9);
  loom_target_compile_report_record_allocation(&function, 5, 0, 0, 1, 2, 0, 0,
                                               0);
  const loom_target_compile_report_static_instruction_mix_t function_mix = {
      .descriptor_count = 3,
      .scalar_alu_count = 2,
      .global_load_count = 1,
  };
  loom_target_compile_report_record_static_instruction_mix(&function,
                                                           &function_mix);
  loom_target_compile_report_record_dynamic_instruction_mix(&function,
                                                            &function_mix);
  loom_target_compile_report_record_emission(&function, 99, 396, 400);
  loom_target_compile_report_record_memory(&function, 128, 256);
  const loom_target_compile_report_pressure_row_t pressure_row = {
      .function_name = IREE_SVL("helper"),
      .register_class = IREE_SVL("test.reg"),
      .peak_live_units = 9,
  };
  IREE_ASSERT_OK(
      loom_target_compile_report_record_pressure_row(&function, &pressure_row));
  const loom_target_compile_report_legalization_row_t legalization_row = {
      .function_name = IREE_SVL("helper"),
      .source_op_name = IREE_SVL("test.op"),
      .action = LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_REWRITTEN,
      .created_op_count = 2,
      .erased_op_count = 1,
  };
  IREE_ASSERT_OK(loom_target_compile_report_record_legalization_row(
      &function, &legalization_row));
  const loom_target_compile_report_config_binding_row_t binding_row = {
      .key = IREE_SVL("test.value"),
      .value = IREE_SVL("4"),
  };
  IREE_ASSERT_OK(loom_target_compile_report_record_config_binding_row(
      &function, &binding_row));
  const loom_target_compile_report_target_capability_row_t capability_row = {
      .function_name = IREE_SVL("helper"),
      .target_family_name = IREE_SVL("test"),
      .namespace_name = IREE_SVL("target"),
      .key = IREE_SVL("feature"),
      .value_kind = LOOM_TARGET_COMPILE_REPORT_CAPABILITY_VALUE_BOOL,
      .value_u64 = 1,
  };
  IREE_ASSERT_OK(loom_target_compile_report_record_target_capability_row(
      &function, &capability_row));
  const loom_symbol_ref_t function_ref = {.module_id = 0, .symbol_id = 2};
  const loom_symbol_ref_t nested_ref = {.module_id = 0, .symbol_id = 3};
  const loom_target_compile_report_call_edge_t function_edge = {
      .flags = LOOM_TARGET_COMPILE_REPORT_CALL_EDGE_DYNAMIC_COUNT,
      .function_name = IREE_SVL("helper"),
      .function = function_ref,
      .callee_name = IREE_SVL("nested"),
      .callee = nested_ref,
      .callsite_count = 1,
      .dynamic_call_count = 2,
  };
  IREE_ASSERT_OK(
      loom_target_compile_report_record_call_edge(&function, &function_edge));

  const loom_target_compile_report_function_merge_options_t options = {
      .invocation_count = 4,
      .has_exact_invocation_count = true,
  };
  IREE_ASSERT_OK(loom_target_compile_report_merge_function_evidence(
      &program, &function, &options));

  EXPECT_EQ(program.schedule_node_count, 5u);
  EXPECT_EQ(program.allocation_assignment_count, 5u);
  EXPECT_EQ(program.register_pressure_peak_live_units, 9u);
  EXPECT_EQ(program.static_instruction_mix.descriptor_count, 5u);
  EXPECT_EQ(program.static_instruction_mix.scalar_alu_count, 2u);
  EXPECT_EQ(program.dynamic_instruction_mix.descriptor_count, 14u);
  EXPECT_EQ(program.dynamic_instruction_mix.scalar_alu_count, 8u);
  EXPECT_EQ(program.emitted_instruction_count, 8u);
  EXPECT_EQ(program.emitted_code_byte_count, 32u);
  EXPECT_EQ(program.emitted_code_storage_byte_count, 40u);
  EXPECT_EQ(program.private_memory_bytes, 16u);
  EXPECT_EQ(program.local_memory_bytes, 32u);
  EXPECT_EQ(program.workload.flat_workgroup_size, 64u);
  EXPECT_EQ(program.pressure_rows.count, 1u);
  EXPECT_EQ(program.pressure_summaries.count, 1u);
  EXPECT_EQ(program.target_legalization_rows.count, 1u);
  EXPECT_EQ(program.target_legalization_rewritten_op_count, 1u);
  EXPECT_EQ(program.config_binding_rows.count, 0u);
  EXPECT_EQ(program.target_capability_rows.count, 0u);
  const loom_target_compile_report_call_edge_t* merged_edge =
      loom_target_compile_report_find_call_edge(&program, function_ref,
                                                nested_ref);
  ASSERT_NE(merged_edge, nullptr);
  EXPECT_EQ(merged_edge->dynamic_call_count, 2u);
  EXPECT_FALSE(
      iree_any_bit_set(program.detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_CONFIG_BINDING_ROWS));
  EXPECT_FALSE(iree_any_bit_set(
      program.detail_flags,
      LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_CAPABILITY_ROWS));

  const loom_target_compile_report_function_merge_options_t unknown_options = {
      .has_exact_invocation_count = false,
  };
  IREE_ASSERT_OK(loom_target_compile_report_merge_function_evidence(
      &program, &function, &unknown_options));
  EXPECT_FALSE(iree_any_bit_set(
      program.detail_flags,
      LOOM_TARGET_COMPILE_REPORT_DETAIL_DYNAMIC_INSTRUCTION_MIX));

  loom_target_compile_report_deinitialize(&function);
  loom_target_compile_report_deinitialize(&program);
}

TEST(CompileReportTest, AccumulatesEvidenceWithoutEntryRows) {
  loom_target_compile_report_t source;
  loom_target_compile_report_initialize(&source, iree_allocator_system());
  loom_target_compile_report_record_schedule(&source, 3, 3, 2, 0, 1, 0, 0, 0,
                                             0);
  loom_target_compile_report_record_legalization_summary(
      &source, LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_REWRITTEN,
      LOOM_TARGET_COMPILE_REPORT_LEGALIZER_STRATEGY_NONE,
      /*scalarized=*/false, /*created_op_count=*/2);

  loom_target_compile_report_t merged;
  loom_target_compile_report_initialize(&merged, iree_allocator_system());
  IREE_ASSERT_OK(loom_target_compile_report_merge_evidence(&merged, &source));
  IREE_ASSERT_OK(loom_target_compile_report_merge_evidence(&merged, &source));

  EXPECT_EQ(merged.merged_evidence_count, 2u);
  EXPECT_EQ(merged.schedule_node_count, 6u);
  EXPECT_EQ(merged.target_legalization_rewritten_op_count, 2u);

  loom_target_compile_report_deinitialize(&merged);
  loom_target_compile_report_deinitialize(&source);
}

TEST(CompileReportTest, FormatsCallEvidenceOnlyInDetailMode) {
  loom_target_compile_report_t report;
  loom_target_compile_report_initialize(&report, iree_allocator_system());
  const loom_target_compile_report_call_edge_t edge = {
      .flags = LOOM_TARGET_COMPILE_REPORT_CALL_EDGE_DYNAMIC_COUNT,
      .function_name = IREE_SVL("entry"),
      .function = {.module_id = 0, .symbol_id = 1},
      .callee_name = IREE_SVL("helper"),
      .callee = {.module_id = 0, .symbol_id = 2},
      .callsite_count = 1,
      .dynamic_call_count = 256,
  };
  IREE_ASSERT_OK(loom_target_compile_report_record_call_edge(&report, &edge));

  for (const auto mode : {LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_SUMMARY,
                          LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS}) {
    const loom_target_compile_report_format_options_t options = {.mode = mode};
    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    IREE_ASSERT_OK(
        loom_target_compile_report_format_text(&report, &options, &builder));
    const std::string text(iree_string_builder_buffer(&builder),
                           iree_string_builder_size(&builder));
    EXPECT_EQ(text.find("call_edge[0] caller=entry callee=helper callsites=1 "
                        "dynamic_calls=256") != std::string::npos,
              mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS)
        << text;
    iree_string_builder_deinitialize(&builder);
  }

  const loom_target_compile_report_format_options_t options = {
      .mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS,
  };
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  IREE_ASSERT_OK(
      loom_target_compile_report_format_json(&report, &options, &stream));
  const std::string json(iree_string_builder_buffer(&builder),
                         iree_string_builder_size(&builder));
  EXPECT_NE(json.find("\"call_edges\":{\"count\":1,\"rows\":[{"),
            std::string::npos)
      << json;
  EXPECT_NE(json.find("\"caller\":\"entry\",\"callee\":\"helper\","),
            std::string::npos)
      << json;
  EXPECT_NE(json.find("\"callsite_count\":1,\"dynamic_call_count\":256"),
            std::string::npos)
      << json;
  iree_string_builder_deinitialize(&builder);

  loom_target_compile_report_deinitialize(&report);
}

}  // namespace
}  // namespace loom
