// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/compile_report_format.h"

#include <stdint.h>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/types.h"

namespace loom {
namespace {

constexpr uint32_t kTestSourceRejectionDetail = 4;

TEST(CompileReportFormatTest, FormatsSummaryAndDetails) {
  loom_target_compile_report_pressure_row_t pressure_rows[] = {
      {
          /*.register_class=*/IREE_SVL("test.i32"),
          /*.type_kind=*/LOOM_TYPE_REGISTER,
          /*.element_type=*/LOOM_SCALAR_TYPE_I32,
          /*.peak_live_units=*/7,
          /*.peak_live_values=*/4,
          /*.peak_point=*/3,
          /*.peak_block_name=*/IREE_SVL("entry"),
          /*.peak_operation_name=*/IREE_SVL("low.op<test.add.i32>"),
      },
  };
  loom_target_compile_report_spill_row_t spill_rows[] = {
      {
          /*.value_name=*/IREE_SVL("rhs"),
          /*.register_class=*/IREE_SVL("test.i32"),
          /*.type_kind=*/LOOM_TYPE_REGISTER,
          /*.element_type=*/LOOM_SCALAR_TYPE_I32,
          /*.assignment_index=*/2,
          /*.slot_index=*/1,
          /*.slot_space=*/IREE_SVL("stack"),
          /*.byte_size=*/4,
          /*.byte_alignment=*/4,
          /*.store_count=*/1,
          /*.reload_count=*/2,
      },
  };
  loom_target_compile_report_source_low_row_t source_low_rows[] = {
      {
          /*.function_name=*/IREE_SVL("branchy"),
          /*.source_op_name=*/IREE_SVL("scalar.addi"),
          /*.source_op_kind=*/42,
          /*.selection_kind=*/
          LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_SELECTION_RULE,
          /*.rule_set_index=*/0,
          /*.rule_index=*/1,
          /*.plan_id=*/UINT64_MAX,
          /*.descriptor_id=*/7,
          /*.emitted_low_op_count=*/1,
      },
  };
  loom_target_compile_report_legalization_row_t target_legalization_rows[] = {
      {
          /*.function_name=*/IREE_SVL("branchy"),
          /*.source_op_name=*/IREE_SVL("vector.reduce.axes"),
          /*.source_op_kind=*/73,
          /*.target_bundle_name=*/IREE_SVL("vm_target"),
          /*.target_config_name=*/IREE_SVL("vm_o0"),
          /*.legalizer_name=*/IREE_SVL("vector"),
          /*.legalizer_strategy=*/
          LOOM_TARGET_COMPILE_REPORT_LEGALIZER_STRATEGY_REFERENCE,
          /*.mode=*/LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_MODE_FINAL,
          /*.policy=*/
          LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_POLICY_REFERENCE_ONLY,
          /*.action=*/LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_REWRITTEN,
          /*.contract_outcome=*/
          LOOM_TARGET_COMPILE_REPORT_CONTRACT_OUTCOME_UNSUPPORTED,
          /*.binding_index=*/0,
          /*.case_index=*/2,
          /*.rule_set_index=*/3,
          /*.rule_index=*/4,
          /*.diagnostic_index=*/UINT16_MAX,
          /*.descriptor_id=*/UINT64_MAX,
          /*.source_rejection_bits=*/0x1,
          /*.source_rejection_detail=*/kTestSourceRejectionDetail,
          /*.target_rejection_bits=*/0x2,
          /*.missing_feature_bits=*/0x4,
          /*.missing_fact_bits=*/0x8,
          /*.created_op_count=*/6,
          /*.erased_op_count=*/1,
      },
  };
  loom_target_compile_report_t report = {};
  loom_target_compile_report_initialize(&report, iree_allocator_system());
  report.requested_detail_flags =
      LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ROWS |
      LOOM_TARGET_COMPILE_REPORT_DETAIL_SPILL_ROWS |
      LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS |
      LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_LEGALIZATION_ROWS;
  report.artifact_kind = LOOM_TARGET_COMPILE_ARTIFACT_KIND_VM_ARCHIVE;
  report.target_bundle_name = IREE_SVL("vm_target");
  report.lowered_symbol = IREE_SVL("branchy");
  loom_target_compile_report_record_artifact_size(&report, 128);
  loom_target_compile_report_record_schedule(&report, 5, 5, 4, 2, 1, 1, 1, 7);
  loom_target_compile_report_record_allocation(&report, 6, 1, 1, 2, 0);
  loom_target_compile_report_record_move_cause(
      &report, LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_CONSTANT_MATERIALIZATION,
      3, 3);
  loom_target_compile_report_record_move_cause(
      &report, LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_LOW_CONCAT, 2, 8);
  loom_target_compile_report_record_move_cause(
      &report,
      LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_OPERAND_BANK_MATERIALIZATION, 1, 1);
  const loom_target_compile_report_static_instruction_mix_t instruction_mix = {
      /*.descriptor_count=*/9,
      /*.unknown_count=*/{},
      /*.scalar_alu_count=*/2,
      /*.vector_alu_count=*/3,
      /*.matrix_count=*/1,
      /*.mfma_count=*/{},
      /*.wmma_count=*/1,
      /*.dot_count=*/{},
      /*.global_memory_count=*/2,
      /*.local_memory_count=*/{},
      /*.scalar_memory_count=*/{},
      /*.generic_memory_count=*/{},
      /*.atomic_count=*/{},
      /*.branch_count=*/{},
      /*.barrier_count=*/1,
      /*.control_count=*/{},
      /*.conversion_count=*/1,
  };
  loom_target_compile_report_record_static_instruction_mix(&report,
                                                           &instruction_mix);
  loom_target_compile_report_record_emission(&report, 8, 64, 80);
  loom_target_compile_report_record_memory(&report, 16, 32);
  IREE_ASSERT_OK(loom_target_compile_report_record_pressure_row(
      &report, &pressure_rows[0]));
  IREE_ASSERT_OK(
      loom_target_compile_report_record_spill_row(&report, &spill_rows[0]));
  report.source_low_selected_op_count = 4;
  report.source_low_emitted_op_count = 5;
  IREE_ASSERT_OK(loom_target_compile_report_record_source_low_row(
      &report, &source_low_rows[0]));
  IREE_ASSERT_OK(loom_target_compile_report_record_legalization_row(
      &report, &target_legalization_rows[0]));

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  const loom_target_compile_report_format_options_t options = {
      /*.mode=*/LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS,
  };
  IREE_ASSERT_OK(
      loom_target_compile_report_format_text(&report, &options, &builder));

  iree_string_view_t output = iree_string_builder_view(&builder);
  EXPECT_NE(iree_string_view_find(output, IREE_SV("artifact=vm-archive"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output, IREE_SV("pressure_classes=1"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output, IREE_SV("pressure_rows count=1"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output,
                                  IREE_SV("move_causes kinds=3 packets=6 "
                                          "units=12"),
                                  0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output,
                                  IREE_SV("move_cause[low_concat] packets=2 "
                                          "units=8"),
                                  0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                output,
                IREE_SV("move_cause[operand_bank_materialization] packets=1 "
                        "units=1"),
                0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output,
                                  IREE_SV("static_instruction_mix "
                                          "descriptors=9"),
                                  0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output, IREE_SV("vector_alu=3"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(output, IREE_SV("pressure[0] class=test.i32"), 0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output, IREE_SV("spill[0] value=rhs"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(output, IREE_SV("source_low selected_ops=4"), 0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output,
                                  IREE_SV("source_low[0] function=branchy"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output,
                                  IREE_SV("target_legalization legal=0 "
                                          "rewritten=1 target_rewritten=0 "
                                          "reference_rewritten=1 deferred=0"),
                                  0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                output,
                IREE_SV("target_legalization[0] function=branchy source_op="
                        "vector.reduce.axes mode=final policy=reference-only "
                        "action=rewritten"),
                0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output, IREE_SV("strategy=reference"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(output, IREE_SV("source_rejection_detail=4"), 0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(output, IREE_SV("created_ops=6 erased_ops=1"), 0),
      IREE_STRING_VIEW_NPOS);

  iree_string_builder_deinitialize(&builder);

  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  IREE_ASSERT_OK(
      loom_target_compile_report_format_json(&report, &options, &stream));

  output = iree_string_builder_view(&builder);
  EXPECT_NE(iree_string_view_find(
                output, IREE_SV("\"artifact_kind\":\"vm-archive\""), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output,
                                  IREE_SV("\"schedule\":{\"node_count\":5"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                output, IREE_SV("\"move_causes\":{\"kind_count\":3"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(
          output, IREE_SV("\"static_instruction_mix\":{\"descriptor_count\":9"),
          0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output, IREE_SV("\"wmma_count\":1"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(output, IREE_SV("\"cause\":\"low_concat\""), 0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output,
                                  IREE_SV("\"pressure_rows\":{\"count\":1,"
                                          "\"rows\":["),
                                  0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                output, IREE_SV("\"register_class\":\"test.i32\""), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output,
                                  IREE_SV("\"spill_rows\":{\"count\":1,"
                                          "\"rows\":["),
                                  0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                output, IREE_SV("\"source_low\":{\"selected_op_count\":4"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output,
                                  IREE_SV("\"rule_set_index\":0,"
                                          "\"rule_index\":1,\"plan_id\":null,"
                                          "\"descriptor_id\":7"),
                                  0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                output,
                IREE_SV("\"target_legalization\":{\"legal_op_count\":0,"
                        "\"rewritten_op_count\":1,"
                        "\"target_rewritten_op_count\":0,"
                        "\"reference_rewritten_op_count\":1"),
                0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(output,
                            IREE_SV("\"legalizer\":\"vector\","
                                    "\"legalizer_strategy\":\"reference\","
                                    "\"mode\":\"final\","
                                    "\"policy\":\"reference-only\","
                                    "\"action\":\"rewritten\","
                                    "\"contract_outcome\":\"unsupported\""),
                            0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output,
                                  IREE_SV("\"source_rejection_detail\":4"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output,
                                  IREE_SV("\"created_op_count\":6,"
                                          "\"erased_op_count\":1"),
                                  0),
            IREE_STRING_VIEW_NPOS);

  iree_string_builder_deinitialize(&builder);
  loom_target_compile_report_deinitialize(&report);
}

TEST(CompileReportFormatTest, FormatsJsonSummaryWithoutDetailRows) {
  loom_target_compile_report_pressure_row_t pressure_rows[] = {
      {
          /*.register_class=*/IREE_SVL("test.i32"),
          /*.type_kind=*/LOOM_TYPE_REGISTER,
          /*.element_type=*/LOOM_SCALAR_TYPE_I32,
          /*.peak_live_units=*/7,
          /*.peak_live_values=*/4,
          /*.peak_point=*/3,
      },
  };
  loom_target_compile_report_t report = {};
  loom_target_compile_report_initialize(&report, iree_allocator_system());
  report.artifact_kind = LOOM_TARGET_COMPILE_ARTIFACT_KIND_HAL_EXECUTABLE;
  report.backend_name = IREE_SVL("hal");
  loom_target_compile_report_record_artifact_size(&report, 256);
  IREE_ASSERT_OK(loom_target_compile_report_record_pressure_row(
      &report, &pressure_rows[0]));

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  const loom_target_compile_report_format_options_t options = {
      /*.mode=*/LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_SUMMARY,
  };
  IREE_ASSERT_OK(
      loom_target_compile_report_format_json(&report, &options, &stream));

  iree_string_view_t output = iree_string_builder_view(&builder);
  EXPECT_NE(iree_string_view_find(
                output, IREE_SV("\"artifact_kind\":\"hal-executable\""), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output, IREE_SV("\"backend\":\"hal\""), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output, IREE_SV("\"artifact_size\":256"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                output, IREE_SV("\"pressure_rows\":{\"count\":1}"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(iree_string_view_find(output, IREE_SV("\"rows\""), 0),
            IREE_STRING_VIEW_NPOS);

  iree_string_builder_deinitialize(&builder);
  loom_target_compile_report_deinitialize(&report);
}

TEST(CompileReportFormatTest, FormatsJsonEscapedStrings) {
  loom_target_compile_report_t report = {};
  loom_target_compile_report_initialize(&report, iree_allocator_system());
  report.backend_name = IREE_SVL("quote\"line\n");

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  const loom_target_compile_report_format_options_t options = {
      /*.mode=*/LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_SUMMARY,
  };
  IREE_ASSERT_OK(
      loom_target_compile_report_format_json(&report, &options, &stream));

  iree_string_view_t output = iree_string_builder_view(&builder);
  EXPECT_NE(iree_string_view_find(
                output, IREE_SV("\"backend\":\"quote\\\"line\\n\""), 0),
            IREE_STRING_VIEW_NPOS);

  iree_string_builder_deinitialize(&builder);
  loom_target_compile_report_deinitialize(&report);
}

TEST(CompileReportFormatTest, JsonModeNoneWritesNothing) {
  loom_target_compile_report_t report = {};
  loom_target_compile_report_initialize(&report, iree_allocator_system());

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  const loom_target_compile_report_format_options_t options = {
      /*.mode=*/LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_NONE,
  };
  IREE_ASSERT_OK(
      loom_target_compile_report_format_json(&report, &options, &stream));
  EXPECT_EQ(iree_string_builder_size(&builder), 0u);

  iree_string_builder_deinitialize(&builder);
  loom_target_compile_report_deinitialize(&report);
}

TEST(CompileReportFormatTest, ParsesModes) {
  loom_target_compile_report_format_mode_t mode =
      LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_NONE;
  IREE_ASSERT_OK(
      loom_target_compile_report_format_mode_parse(IREE_SV("summary"), &mode));
  EXPECT_EQ(mode, LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_SUMMARY);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_target_compile_report_format_mode_parse(IREE_SV("verbose"), &mode));
}

}  // namespace
}  // namespace loom
