// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "iree/base/internal/json.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/types.h"
#include "loom/target/math_policy.h"
#include "loom/target/reporting/format.h"

namespace loom {
namespace {

constexpr uint32_t kTestSourceRejectionDetail = 4;

static iree_string_view_t ParseJsonDocument(iree_string_view_t json) {
  iree_string_view_t cursor = json;
  iree_string_view_t value = iree_string_view_empty();
  IREE_EXPECT_OK(iree_json_consume_value(&cursor, &value));
  IREE_EXPECT_OK(iree_json_consume_insignificant(&cursor));
  EXPECT_TRUE(iree_string_view_is_empty(cursor));
  return value;
}

static iree_string_view_t LookupObject(iree_string_view_t object,
                                       iree_string_view_t key) {
  iree_string_view_t value = iree_string_view_empty();
  IREE_EXPECT_OK(iree_json_lookup_object_value(object, key, &value));
  return value;
}

static void ExpectObjectValueEquals(iree_string_view_t object,
                                    iree_string_view_t key,
                                    iree_string_view_t expected) {
  EXPECT_TRUE(iree_string_view_equal(LookupObject(object, key), expected));
}

static void ExpectObjectUint64Equals(iree_string_view_t object,
                                     iree_string_view_t key,
                                     uint64_t expected) {
  uint64_t actual = 0;
  IREE_EXPECT_OK(iree_json_parse_uint64(LookupObject(object, key), &actual));
  EXPECT_EQ(actual, expected);
}

static iree_string_view_t LookupArrayElement(iree_string_view_t array,
                                             iree_host_size_t index) {
  iree_string_view_t value = iree_string_view_empty();
  IREE_EXPECT_OK(iree_json_array_get(array, index, &value));
  return value;
}

TEST(CompileReportFormatTest, FormatsSourceToLowSelectionAndMemory) {
  loom_target_compile_report_t report = {};
  loom_target_compile_report_initialize(&report, iree_allocator_system());
  report.requested_detail_flags =
      LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS;
  report.source_low_selected_op_count = 4;
  report.source_low_emitted_op_count = 5;

  loom_target_compile_report_source_low_row_t selection = {};
  selection.function_name = IREE_SVL("branchy");
  selection.source_op_name = IREE_SVL("scalar.addi");
  selection.source_op_kind = 42;
  selection.selection_kind =
      LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_SELECTION_RULE;
  selection.rule_set_index = 0;
  selection.rule_index = 1;
  selection.plan_id = UINT64_MAX;
  selection.plan_key = IREE_SVL("test.scalar_addi.strategy.native");
  selection.descriptor_key = IREE_SVL("test.add.i32");
  selection.descriptor_semantic_tag = IREE_SVL("integer.add.i32");
  selection.emitted_low_op_count = 1;
  IREE_ASSERT_OK(
      loom_target_compile_report_record_source_low_row(&report, &selection));

  loom_target_compile_report_source_low_memory_row_t memory = {};
  memory.function_name = IREE_SVL("branchy");
  memory.source_op_name = IREE_SVL("vector.load");
  memory.source_op_kind = 43;
  memory.source_root_name = IREE_SVL("lhs");
  memory.source_root_argument_index = 0;
  memory.memory_space = IREE_SVL("workgroup");
  memory.operation_kind = IREE_SVL("load");
  memory.packet_key = IREE_SVL("amdgpu.ds_read2_b32");
  memory.strategy_key = IREE_SVL("ds_2addr_memory_report");
  memory.address_form = IREE_SVL("ds_2addr");
  memory.dynamic_term_kind = IREE_SVL("vaddr");
  memory.fallback_reason = IREE_SVL("cross_wave_workgroup");
  memory.element_byte_count = 4;
  memory.vector_lane_count = 2;
  memory.issued_read_byte_count = 8;
  memory.dynamic_stride_bytes = 32;
  memory.vector_lane_stride_bytes = 8;
  memory.storage_element_format = IREE_SVL("f8e4m3fn");
  memory.storage_scale_format = IREE_SVL("f32");
  memory.storage_payload_packing = IREE_SVL("dense_lanes");
  memory.storage_scale_topology = IREE_SVL("block_1d");
  memory.storage_affine_policy = IREE_SVL("scale_only");
  memory.storage_rounding_policy = IREE_SVL("finite_only");
  memory.execution_count_plus_one = 2;
  IREE_ASSERT_OK(loom_target_compile_report_record_source_low_memory_row(
      &report, &memory));

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  const loom_target_compile_report_format_options_t options = {
      /*.mode=*/LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS,
  };
  IREE_ASSERT_OK(
      loom_target_compile_report_format_text(&report, &options, &builder));
  const iree_string_view_t text = iree_string_builder_view(&builder);
  EXPECT_NE(
      iree_string_view_find(text, IREE_SV("source_low selected_ops=4"), 0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(text, IREE_SV("source_low[0] function=branchy"), 0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                text, IREE_SV("plan_key=test.scalar_addi.strategy.native"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(text,
                            IREE_SV("source_low_memory[0] function=branchy "
                                    "source_op=vector.load"),
                            0),
      IREE_STRING_VIEW_NPOS);
  iree_string_builder_deinitialize(&builder);

  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  IREE_ASSERT_OK(
      loom_target_compile_report_format_json(&report, &options, &stream));
  const iree_string_view_t root =
      ParseJsonDocument(iree_string_builder_view(&builder));
  const iree_string_view_t source_low =
      LookupObject(root, IREE_SV("source_low"));
  ExpectObjectUint64Equals(source_low, IREE_SV("selected_op_count"), 4);
  const iree_string_view_t selection_row = LookupArrayElement(
      LookupObject(source_low, IREE_SV("rows")), /*index=*/0);
  ExpectObjectValueEquals(selection_row, IREE_SV("plan_key"),
                          IREE_SV("test.scalar_addi.strategy.native"));
  ExpectObjectValueEquals(selection_row, IREE_SV("descriptor_semantic_tag"),
                          IREE_SV("integer.add.i32"));
  const iree_string_view_t memory_row = LookupArrayElement(
      LookupObject(source_low, IREE_SV("memory_rows")), /*index=*/0);
  ExpectObjectValueEquals(memory_row, IREE_SV("packet"),
                          IREE_SV("amdgpu.ds_read2_b32"));
  ExpectObjectValueEquals(memory_row, IREE_SV("strategy"),
                          IREE_SV("ds_2addr_memory_report"));
  const iree_string_view_t storage =
      LookupObject(memory_row, IREE_SV("storage"));
  ExpectObjectValueEquals(storage, IREE_SV("element_format"),
                          IREE_SV("f8e4m3fn"));
  ExpectObjectValueEquals(storage, IREE_SV("rounding_policy"),
                          IREE_SV("finite_only"));

  iree_string_builder_deinitialize(&builder);
  loom_target_compile_report_deinitialize(&report);
}

TEST(CompileReportFormatTest, FormatsMathAndTargetLegalization) {
  loom_target_compile_report_t report = {};
  loom_target_compile_report_initialize(&report, iree_allocator_system());
  report.requested_detail_flags =
      LOOM_TARGET_COMPILE_REPORT_DETAIL_MATH_LEGALIZATION_ROWS |
      LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_LEGALIZATION_ROWS;

  loom_target_compile_report_math_row_t math = {};
  math.function_name = IREE_SVL("branchy");
  math.source_op_name = IREE_SVL("scalar.roundf");
  math.source_op_kind = 44;
  math.target_bundle_name = IREE_SVL("vm_target");
  math.target_config_name = IREE_SVL("vm_o0");
  math.policy_name = IREE_SVL("amdgpu-math");
  math.constraint_key = IREE_SVL("math.recipe.round_away_f32");
  math.math_op = LOOM_TARGET_MATH_OP_ROUNDF;
  math.lane_domain = LOOM_TARGET_MATH_LANE_DOMAIN_SCALAR;
  math.element_type = LOOM_SCALAR_TYPE_F32;
  math.action = LOOM_TARGET_COMPILE_REPORT_MATH_ACTION_REWRITTEN;
  math.recipe = LOOM_TARGET_MATH_RECIPE_ROUND_AWAY_F32;
  math.created_op_count = 10;
  math.erased_op_count = 1;
  IREE_ASSERT_OK(loom_target_compile_report_record_math_row(&report, &math));

  loom_target_compile_report_legalization_row_t legalization = {};
  legalization.function_name = IREE_SVL("branchy");
  legalization.source_op_name = IREE_SVL("vector.reduce.axes");
  legalization.source_op_kind = 73;
  legalization.target_bundle_name = IREE_SVL("vm_target");
  legalization.target_config_name = IREE_SVL("vm_o0");
  legalization.legalizer_name = IREE_SVL("vector");
  legalization.legalizer_strategy =
      LOOM_TARGET_COMPILE_REPORT_LEGALIZER_STRATEGY_REFERENCE;
  legalization.mode = LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_MODE_FINAL;
  legalization.policy =
      LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_POLICY_REFERENCE_ONLY;
  legalization.action =
      LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_REWRITTEN;
  legalization.legalization_outcome =
      LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_OUTCOME_REFERENCE_FALLBACK;
  legalization.contract_outcome =
      LOOM_TARGET_COMPILE_REPORT_CONTRACT_OUTCOME_UNSUPPORTED;
  legalization.binding_index = 0;
  legalization.case_index = 2;
  legalization.rule_set_index = 3;
  legalization.rule_index = 4;
  legalization.diagnostic_index = UINT16_MAX;
  legalization.descriptor_key = IREE_SVL("test.legalized.descriptor");
  legalization.source_rejection_bits = 0x1;
  legalization.source_rejection_detail = kTestSourceRejectionDetail;
  legalization.target_rejection_bits = 0x2;
  legalization.missing_feature_bits = 0x4;
  legalization.missing_fact_bits = 0x8;
  legalization.created_op_count = 6;
  legalization.erased_op_count = 1;
  IREE_ASSERT_OK(loom_target_compile_report_record_legalization_row(
      &report, &legalization));

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  const loom_target_compile_report_format_options_t options = {
      /*.mode=*/LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS,
  };
  IREE_ASSERT_OK(
      loom_target_compile_report_format_text(&report, &options, &builder));
  const iree_string_view_t text = iree_string_builder_view(&builder);
  EXPECT_NE(
      iree_string_view_find(text, IREE_SV("math_legalization rewritten=1"), 0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(text,
                            IREE_SV("math_legalization[0] function=branchy "
                                    "source_op=scalar.roundf"),
                            0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(text,
                            IREE_SV("target_legalization[0] function=branchy "
                                    "source_op=vector.reduce.axes"),
                            0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                text, IREE_SV("descriptor_key=test.legalized.descriptor"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(iree_string_view_find(text, IREE_SV(" descriptor="), 0),
            IREE_STRING_VIEW_NPOS);
  iree_string_builder_deinitialize(&builder);

  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  IREE_ASSERT_OK(
      loom_target_compile_report_format_json(&report, &options, &stream));
  const iree_string_view_t root =
      ParseJsonDocument(iree_string_builder_view(&builder));
  const iree_string_view_t math_json =
      LookupObject(root, IREE_SV("math_legalization"));
  const iree_string_view_t math_row =
      LookupArrayElement(LookupObject(math_json, IREE_SV("rows")), /*index=*/0);
  ExpectObjectValueEquals(math_row, IREE_SV("recipe"),
                          IREE_SV("round-away-f32"));
  ExpectObjectUint64Equals(math_row, IREE_SV("created_op_count"), 10);
  const iree_string_view_t legalization_json =
      LookupObject(root, IREE_SV("target_legalization"));
  const iree_string_view_t legalization_row = LookupArrayElement(
      LookupObject(legalization_json, IREE_SV("rows")), /*index=*/0);
  ExpectObjectValueEquals(legalization_row, IREE_SV("legalizer_strategy"),
                          IREE_SV("reference"));
  ExpectObjectValueEquals(legalization_row, IREE_SV("descriptor_key"),
                          IREE_SV("test.legalized.descriptor"));
  ExpectObjectUint64Equals(legalization_row, IREE_SV("source_rejection_detail"),
                           4);
  ExpectObjectUint64Equals(legalization_row, IREE_SV("created_op_count"), 6);

  iree_string_builder_deinitialize(&builder);
  loom_target_compile_report_deinitialize(&report);
}

}  // namespace
}  // namespace loom
