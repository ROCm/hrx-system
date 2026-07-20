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
#include "loom/target/reporting/format.h"

namespace loom {
namespace {

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

static iree_string_view_t TryLookupObject(iree_string_view_t object,
                                          iree_string_view_t key) {
  iree_string_view_t value = iree_string_view_empty();
  IREE_EXPECT_OK(iree_json_try_lookup_object_value(object, key, &value));
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

static void ExpectArrayLength(iree_string_view_t array,
                              iree_host_size_t expected) {
  iree_host_size_t actual = 0;
  IREE_EXPECT_OK(iree_json_array_length(array, &actual));
  EXPECT_EQ(actual, expected);
}

TEST(CompileReportFormatTest, FormatsJsonSummaryWithoutDetailRows) {
  loom_target_compile_report_pressure_row_t pressure_rows[] = {
      {
          /*.function_name=*/IREE_SVL("summary_only"),
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
  const loom_target_compile_report_schedule_band_summary_row_t
      schedule_band_summary = {
          /*.flags=*/
          LOOM_TARGET_COMPILE_REPORT_SCHEDULE_BAND_DYNAMIC_INSTRUCTION_MIX,
          /*.function_name=*/IREE_SVL("summary_only"),
          /*.block_name=*/IREE_SVL("^entry"),
          /*.block_index=*/0,
          /*.first_packet_index=*/5,
          /*.band_count=*/2,
          /*.node_count=*/3,
          /*.max_band_node_count=*/2,
          /*.origin_kind=*/LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_MATRIX,
          /*.origin_operation_name=*/IREE_SVL("low.op<amdgpu.wmma>"),
          /*.semantic_tag=*/IREE_SVL("matrix.wmma.f32"),
          /*.sample_value_name=*/IREE_SVL("%acc"),
          /*.static_instruction_mix=*/
          {
              /*.descriptor_count=*/2,
              /*.unknown_count=*/{},
              /*.scalar_alu_count=*/{},
              /*.vector_alu_count=*/{},
              /*.matrix_count=*/2,
              /*.mfma_count=*/{},
              /*.smfmac_count=*/{},
              /*.wmma_count=*/2,
              /*.swmmac_count=*/{},
          },
          /*.dynamic_instruction_mix=*/
          {
              /*.descriptor_count=*/4,
              /*.unknown_count=*/{},
              /*.scalar_alu_count=*/{},
              /*.vector_alu_count=*/{},
              /*.matrix_count=*/4,
              /*.mfma_count=*/{},
              /*.smfmac_count=*/{},
              /*.wmma_count=*/4,
              /*.swmmac_count=*/{},
          },
          /*.result_value_count=*/1,
          /*.result_unit_count=*/8,
      };
  IREE_ASSERT_OK(loom_target_compile_report_record_schedule_band_summary_row(
      &report, &schedule_band_summary));
  const loom_target_compile_report_wait_reason_summary_row_t
      wait_reason_summary = {
          /*.function_name=*/IREE_SVL("summary_only"),
          /*.counter_name=*/IREE_SVL("vmem_load"),
          /*.reason_name=*/IREE_SVL("amdgpu.ssa_use"),
          /*.counter_id=*/1,
          /*.reason_id=*/2,
          /*.summary=*/
          {
              /*.action_count=*/3,
              /*.explicit_action_count=*/0,
              /*.planned_action_count=*/3,
              /*.full_drain_count=*/1,
              /*.partial_wait_count=*/2,
              /*.drained_count=*/5,
              /*.max_drained_count=*/4,
              /*.max_outstanding_before=*/7,
              /*.max_full_drain_outstanding_before=*/6,
          },
      };
  IREE_ASSERT_OK(loom_target_compile_report_record_wait_reason_summary_row(
      &report, &wait_reason_summary));

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  const loom_target_compile_report_format_options_t options = {
      /*.mode=*/LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_SUMMARY,
  };
  IREE_ASSERT_OK(
      loom_target_compile_report_format_json(&report, &options, &stream));

  const iree_string_view_t root =
      ParseJsonDocument(iree_string_builder_view(&builder));
  ExpectObjectValueEquals(root, IREE_SV("artifact_kind"),
                          IREE_SV("hal-executable"));
  ExpectObjectValueEquals(root, IREE_SV("backend"), IREE_SV("hal"));
  ExpectObjectUint64Equals(root, IREE_SV("artifact_size"), 256);

  const iree_string_view_t entries = LookupObject(root, IREE_SV("entries"));
  ExpectObjectUint64Equals(entries, IREE_SV("count"), 0);
  ExpectArrayLength(LookupObject(entries, IREE_SV("rows")), 0);

  const iree_string_view_t pressure_rows_json =
      LookupObject(root, IREE_SV("pressure_rows"));
  ExpectObjectUint64Equals(pressure_rows_json, IREE_SV("count"), 1);
  EXPECT_TRUE(iree_string_view_is_empty(
      TryLookupObject(pressure_rows_json, IREE_SV("rows"))));
  EXPECT_TRUE(iree_string_view_is_empty(
      TryLookupObject(root, IREE_SV("schedule_band_rows"))));

  const iree_string_view_t schedule_band_summary_rows =
      LookupObject(root, IREE_SV("schedule_band_summary_rows"));
  ExpectObjectUint64Equals(schedule_band_summary_rows, IREE_SV("count"), 1);
  const iree_string_view_t schedule_band_rows =
      LookupObject(schedule_band_summary_rows, IREE_SV("rows"));
  ExpectArrayLength(schedule_band_rows, 1);
  const iree_string_view_t schedule_band =
      LookupArrayElement(schedule_band_rows, 0);
  ExpectObjectValueEquals(schedule_band, IREE_SV("semantic_tag"),
                          IREE_SV("matrix.wmma.f32"));
  const iree_string_view_t static_mix =
      LookupObject(schedule_band, IREE_SV("static_instruction_mix"));
  ExpectObjectUint64Equals(static_mix, IREE_SV("wmma_count"), 2);
  const iree_string_view_t dynamic_mix =
      LookupObject(schedule_band, IREE_SV("dynamic_instruction_mix"));
  ExpectObjectUint64Equals(dynamic_mix, IREE_SV("descriptor_count"), 4);
  ExpectObjectUint64Equals(dynamic_mix, IREE_SV("wmma_count"), 4);

  const iree_string_view_t wait_reason_summary_rows =
      LookupObject(root, IREE_SV("wait_reason_summary_rows"));
  ExpectObjectUint64Equals(wait_reason_summary_rows, IREE_SV("count"), 1);
  const iree_string_view_t wait_reason = LookupArrayElement(
      LookupObject(wait_reason_summary_rows, IREE_SV("rows")), 0);
  ExpectObjectValueEquals(wait_reason, IREE_SV("counter"),
                          IREE_SV("vmem_load"));
  ExpectObjectUint64Equals(wait_reason, IREE_SV("counter_id"), 1);
  ExpectObjectValueEquals(wait_reason, IREE_SV("reason"),
                          IREE_SV("amdgpu.ssa_use"));
  ExpectObjectUint64Equals(wait_reason, IREE_SV("reason_id"), 2);
  const iree_string_view_t wait_summary =
      LookupObject(wait_reason, IREE_SV("summary"));
  ExpectObjectUint64Equals(wait_summary, IREE_SV("action_count"), 3);
  ExpectObjectUint64Equals(wait_summary, IREE_SV("explicit_action_count"), 0);
  ExpectObjectUint64Equals(wait_summary, IREE_SV("planned_action_count"), 3);
  ExpectObjectUint64Equals(wait_summary, IREE_SV("full_drain_count"), 1);
  ExpectObjectUint64Equals(wait_summary, IREE_SV("partial_wait_count"), 2);
  ExpectObjectUint64Equals(wait_summary, IREE_SV("drained_count"), 5);
  ExpectObjectUint64Equals(wait_summary, IREE_SV("max_drained_count"), 4);
  ExpectObjectUint64Equals(wait_summary, IREE_SV("max_outstanding_before"), 7);
  ExpectObjectUint64Equals(wait_summary,
                           IREE_SV("max_full_drain_outstanding_before"), 6);

  const iree_string_view_t wait_action_rows =
      LookupObject(root, IREE_SV("wait_action_rows"));
  ExpectObjectUint64Equals(wait_action_rows, IREE_SV("count"), 0);
  EXPECT_TRUE(iree_string_view_is_empty(
      TryLookupObject(wait_action_rows, IREE_SV("rows"))));

  iree_string_builder_deinitialize(&builder);
  loom_target_compile_report_deinitialize(&report);
}

TEST(CompileReportFormatTest, FormatsSourceLowTransformRowsJson) {
  loom_target_compile_report_t report = {};
  loom_target_compile_report_initialize(&report, iree_allocator_system());
  loom_target_compile_report_source_low_transform_row_t row = {};
  row.function_name = IREE_SVL("kernel");
  row.source_op_name = IREE_SVL("scf.for");
  row.source_op_kind = 42;
  row.transform_key = IREE_SVL("stage-loop-carried-fragments");
  row.outcome = IREE_SVL("selected");
  row.reason = IREE_SVL("staged_workgroup_memory");
  row.candidate_value_count = 4;
  row.selected_value_count = 4;
  row.removed_loop_carried_value_count = 4;
  row.removed_loop_carried_payload_register_count = 32;
  row.block_count = 4;
  row.row_count = 16;
  row.column_count = 16;
  row.workgroup_memory_byte_count = 4096;
  row.inserted_load_op_count = 8;
  row.inserted_store_op_count = 8;
  row.inserted_barrier_op_count = 2;
  IREE_ASSERT_OK(loom_target_compile_report_record_source_low_transform_row(
      &report, &row));

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  const loom_target_compile_report_format_options_t summary_options = {
      /*.mode=*/LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_SUMMARY,
  };
  IREE_ASSERT_OK(loom_target_compile_report_format_json(
      &report, &summary_options, &stream));
  iree_string_view_t root =
      ParseJsonDocument(iree_string_builder_view(&builder));
  iree_string_view_t source_low = LookupObject(root, IREE_SV("source_low"));
  iree_string_view_t transforms =
      LookupObject(source_low, IREE_SV("transforms"));
  ExpectObjectUint64Equals(transforms, IREE_SV("count"), 1);
  EXPECT_TRUE(
      iree_string_view_is_empty(TryLookupObject(transforms, IREE_SV("rows"))));
  iree_string_builder_deinitialize(&builder);

  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_for_builder(&builder, &stream);
  const loom_target_compile_report_format_options_t details_options = {
      /*.mode=*/LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS,
  };
  IREE_ASSERT_OK(loom_target_compile_report_format_json(
      &report, &details_options, &stream));
  root = ParseJsonDocument(iree_string_builder_view(&builder));
  source_low = LookupObject(root, IREE_SV("source_low"));
  transforms = LookupObject(source_low, IREE_SV("transforms"));
  ExpectObjectUint64Equals(transforms, IREE_SV("count"), 1);
  const iree_string_view_t transform =
      LookupArrayElement(LookupObject(transforms, IREE_SV("rows")), 0);
  ExpectObjectValueEquals(transform, IREE_SV("function"), IREE_SV("kernel"));
  ExpectObjectValueEquals(transform, IREE_SV("source_op"), IREE_SV("scf.for"));
  ExpectObjectUint64Equals(transform, IREE_SV("source_op_kind"), 42);
  ExpectObjectValueEquals(transform, IREE_SV("transform"),
                          IREE_SV("stage-loop-carried-fragments"));
  ExpectObjectValueEquals(transform, IREE_SV("outcome"), IREE_SV("selected"));
  ExpectObjectValueEquals(transform, IREE_SV("reason"),
                          IREE_SV("staged_workgroup_memory"));
  ExpectObjectUint64Equals(transform, IREE_SV("candidate_value_count"), 4);
  ExpectObjectUint64Equals(transform, IREE_SV("selected_value_count"), 4);
  ExpectObjectUint64Equals(transform,
                           IREE_SV("removed_loop_carried_value_count"), 4);
  ExpectObjectUint64Equals(
      transform, IREE_SV("removed_loop_carried_payload_register_count"), 32);
  ExpectObjectUint64Equals(transform, IREE_SV("block_count"), 4);
  ExpectObjectUint64Equals(transform, IREE_SV("row_count"), 16);
  ExpectObjectUint64Equals(transform, IREE_SV("column_count"), 16);
  ExpectObjectUint64Equals(transform, IREE_SV("workgroup_memory_byte_count"),
                           4096);
  ExpectObjectUint64Equals(transform, IREE_SV("inserted_load_op_count"), 8);
  ExpectObjectUint64Equals(transform, IREE_SV("inserted_store_op_count"), 8);
  ExpectObjectUint64Equals(transform, IREE_SV("inserted_barrier_op_count"), 2);
  iree_string_builder_deinitialize(&builder);

  loom_target_compile_report_deinitialize(&report);
}

TEST(CompileReportFormatTest, FormatsSourceLowResidencyRowsJson) {
  loom_target_compile_report_t report = {};
  loom_target_compile_report_initialize(&report, iree_allocator_system());
  report.requested_detail_flags =
      LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS;
  loom_target_compile_report_source_low_residency_row_t row = {};
  row.function_name = IREE_SVL("kernel");
  row.source_op_name = IREE_SVL("scf.for");
  row.source_op_kind = 42;
  row.policy = IREE_SVL("preserve");
  row.outcome = IREE_SVL("selected");
  row.reason = IREE_SVL("residency_cliff");
  row.baseline_tier = 4;
  row.selected_tier = 4;
  row.required_tier = 4;
  row.candidate_count = 3;
  row.selected_count = 2;
  row.rejected_count = 1;
  row.reserve_count = 2;
  row.projection_complete = true;
  IREE_ASSERT_OK(loom_target_compile_report_record_source_low_residency_row(
      &report, &row));
  loom_target_compile_report_source_low_residency_resource_row_t resource = {};
  resource.function_name = IREE_SVL("kernel");
  resource.source_op_name = IREE_SVL("scf.for");
  resource.source_op_kind = 42;
  resource.phase = IREE_SVL("selected");
  resource.program_point = 17;
  resource.resource_name = IREE_SVL("vgpr");
  resource.resource_kind = IREE_SVL("direct");
  resource.resource_id = 1;
  resource.units = 112;
  resource.tier = 4;
  resource.next_better_tier = 0;
  resource.next_worse_tier = 3;
  resource.next_worse_cliff_units = 128;
  resource.additional_units_to_next_worse_tier = 16;
  resource.reduction_units_to_next_better_tier = 0;
  resource.limiting = false;
  IREE_ASSERT_OK(
      loom_target_compile_report_record_source_low_residency_resource_row(
          &report, &resource));
  loom_target_compile_report_source_low_residency_reserve_row_t reserve = {};
  reserve.function_name = IREE_SVL("kernel");
  reserve.reserve_name = IREE_SVL("target_abi");
  reserve.resource_name = IREE_SVL("sgpr");
  reserve.resource_id = 0;
  reserve.units = 8;
  IREE_ASSERT_OK(
      loom_target_compile_report_record_source_low_residency_reserve_row(
          &report, &reserve));
  loom_target_compile_report_residency_candidate_row_t candidate = {};
  candidate.function_name = IREE_SVL("kernel");
  candidate.source_op_name = IREE_SVL("scalar.addf");
  candidate.source_op_kind = 43;
  candidate.candidate_id = 7;
  candidate.stage = IREE_SVL("exact");
  candidate.outcome = IREE_SVL("restored");
  candidate.projected_recompute_cost = 2;
  candidate.exact_use_count = 3;
  candidate.cloned_packet_count = 1;
  candidate.rewritten_operand_count = 3;
  candidate.preserves_baseline = true;
  IREE_ASSERT_OK(loom_target_compile_report_record_residency_candidate_row(
      &report, &candidate));

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  const loom_target_compile_report_format_options_t summary_options = {
      /*.mode=*/LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_SUMMARY,
  };
  IREE_ASSERT_OK(loom_target_compile_report_format_json(
      &report, &summary_options, &stream));
  iree_string_view_t root =
      ParseJsonDocument(iree_string_builder_view(&builder));
  iree_string_view_t source_low = LookupObject(root, IREE_SV("source_low"));
  iree_string_view_t placements =
      LookupObject(source_low, IREE_SV("residency_placements"));
  ExpectObjectUint64Equals(placements, IREE_SV("count"), 1);
  EXPECT_TRUE(
      iree_string_view_is_empty(TryLookupObject(placements, IREE_SV("rows"))));
  const iree_string_view_t candidate_summary =
      LookupObject(source_low, IREE_SV("residency_candidates"));
  ExpectObjectUint64Equals(candidate_summary, IREE_SV("count"), 1);
  EXPECT_TRUE(iree_string_view_is_empty(
      TryLookupObject(candidate_summary, IREE_SV("rows"))));
  const iree_string_view_t resource_summary =
      LookupObject(source_low, IREE_SV("residency_resources"));
  ExpectObjectUint64Equals(resource_summary, IREE_SV("count"), 1);
  const iree_string_view_t reserve_summary =
      LookupObject(source_low, IREE_SV("residency_reserves"));
  ExpectObjectUint64Equals(reserve_summary, IREE_SV("count"), 1);
  iree_string_builder_deinitialize(&builder);

  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_for_builder(&builder, &stream);
  const loom_target_compile_report_format_options_t details_options = {
      /*.mode=*/LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS,
  };
  IREE_ASSERT_OK(loom_target_compile_report_format_json(
      &report, &details_options, &stream));
  root = ParseJsonDocument(iree_string_builder_view(&builder));
  source_low = LookupObject(root, IREE_SV("source_low"));
  placements = LookupObject(source_low, IREE_SV("residency_placements"));
  const iree_string_view_t placement =
      LookupArrayElement(LookupObject(placements, IREE_SV("rows")), 0);
  ExpectObjectValueEquals(placement, IREE_SV("function"), IREE_SV("kernel"));
  ExpectObjectValueEquals(placement, IREE_SV("source_op"), IREE_SV("scf.for"));
  ExpectObjectUint64Equals(placement, IREE_SV("source_op_kind"), 42);
  ExpectObjectValueEquals(placement, IREE_SV("policy"), IREE_SV("preserve"));
  ExpectObjectValueEquals(placement, IREE_SV("outcome"), IREE_SV("selected"));
  ExpectObjectValueEquals(placement, IREE_SV("reason"),
                          IREE_SV("residency_cliff"));
  ExpectObjectUint64Equals(placement, IREE_SV("baseline_tier"), 4);
  ExpectObjectUint64Equals(placement, IREE_SV("selected_tier"), 4);
  ExpectObjectUint64Equals(placement, IREE_SV("required_tier"), 4);
  ExpectObjectUint64Equals(placement, IREE_SV("candidate_count"), 3);
  ExpectObjectUint64Equals(placement, IREE_SV("selected_count"), 2);
  ExpectObjectUint64Equals(placement, IREE_SV("rejected_count"), 1);
  ExpectObjectUint64Equals(placement, IREE_SV("reserve_count"), 2);
  ExpectObjectValueEquals(placement, IREE_SV("projection_complete"),
                          IREE_SV("true"));
  const iree_string_view_t candidates =
      LookupObject(source_low, IREE_SV("residency_candidates"));
  const iree_string_view_t candidate_row =
      LookupArrayElement(LookupObject(candidates, IREE_SV("rows")), 0);
  ExpectObjectValueEquals(candidate_row, IREE_SV("function"),
                          IREE_SV("kernel"));
  ExpectObjectValueEquals(candidate_row, IREE_SV("source_op"),
                          IREE_SV("scalar.addf"));
  ExpectObjectUint64Equals(candidate_row, IREE_SV("candidate_id"), 7);
  ExpectObjectValueEquals(candidate_row, IREE_SV("stage"), IREE_SV("exact"));
  ExpectObjectValueEquals(candidate_row, IREE_SV("outcome"),
                          IREE_SV("restored"));
  ExpectObjectUint64Equals(candidate_row, IREE_SV("projected_recompute_cost"),
                           2);
  ExpectObjectUint64Equals(candidate_row, IREE_SV("exact_use_count"), 3);
  ExpectObjectUint64Equals(candidate_row, IREE_SV("cloned_packet_count"), 1);
  ExpectObjectUint64Equals(candidate_row, IREE_SV("rewritten_operand_count"),
                           3);
  ExpectObjectValueEquals(candidate_row, IREE_SV("preserves_baseline"),
                          IREE_SV("true"));
  const iree_string_view_t resources =
      LookupObject(source_low, IREE_SV("residency_resources"));
  const iree_string_view_t resource_row =
      LookupArrayElement(LookupObject(resources, IREE_SV("rows")), 0);
  ExpectObjectValueEquals(resource_row, IREE_SV("phase"), IREE_SV("selected"));
  ExpectObjectUint64Equals(resource_row, IREE_SV("program_point"), 17);
  ExpectObjectValueEquals(resource_row, IREE_SV("resource"), IREE_SV("vgpr"));
  ExpectObjectValueEquals(resource_row, IREE_SV("resource_kind"),
                          IREE_SV("direct"));
  ExpectObjectUint64Equals(resource_row, IREE_SV("units"), 112);
  ExpectObjectUint64Equals(resource_row, IREE_SV("next_worse_tier"), 3);
  ExpectObjectUint64Equals(resource_row, IREE_SV("next_worse_cliff_units"),
                           128);
  ExpectObjectUint64Equals(resource_row,
                           IREE_SV("additional_units_to_next_worse_tier"), 16);
  const iree_string_view_t reserves =
      LookupObject(source_low, IREE_SV("residency_reserves"));
  const iree_string_view_t reserve_row =
      LookupArrayElement(LookupObject(reserves, IREE_SV("rows")), 0);
  ExpectObjectValueEquals(reserve_row, IREE_SV("reserve"),
                          IREE_SV("target_abi"));
  ExpectObjectValueEquals(reserve_row, IREE_SV("resource"), IREE_SV("sgpr"));
  ExpectObjectUint64Equals(reserve_row, IREE_SV("units"), 8);
  iree_string_builder_deinitialize(&builder);

  loom_target_compile_report_deinitialize(&report);
}

TEST(CompileReportFormatTest, FormatsAndAggregatesLowPlanningStatistics) {
  loom_target_compile_report_t report;
  loom_target_compile_report_initialize(&report, iree_allocator_system());

  for (uint64_t i = 1; i <= 2; ++i) {
    loom_target_compile_report_t entry;
    loom_target_compile_report_initialize(&entry, iree_allocator_system());
    entry.detail_flags |= LOOM_TARGET_COMPILE_REPORT_DETAIL_LOW_PLANNING;
    loom_low_planning_statistics_t* planning = &entry.low_planning;
    planning->flags = LOOM_LOW_PLANNING_STATISTICS_FLAG_SYSTEM_ALLOCATIONS;
    planning->frame_build_count = i;
    planning->allocation_run_count = i + 1;
    planning->repair.iteration_count = i + 2;
    planning->repair.diagnostic_replay_count = i + 3;
    planning->residency.contract_count = i;
    planning->residency.candidate_count = i + 4;
    planning->residency.maximum_projected_required_tier = i + 1;
    planning->residency.validation_count = i + 5;
    planning->residency.minimum_observed_allocated_tier = 5 - i;
    planning->residency.maximum_observed_tier_shortfall = i - 1;
    planning->residency.repair_attempt_count = i + 6;
    planning->residency.repair_count = i + 7;
    planning->residency.failure_count = i - 1;
    planning->memory.frame_arena.used_bytes_high_water = i * 100;
    planning->memory.frame_arena.owned_bytes_high_water = i * 200;
    planning->memory.block_system_allocation_count = i;
    planning->memory.block_system_allocation_bytes = i * 32768;
    planning->memory.oversized_allocation_count = i + 4;
    planning->memory.oversized_allocation_bytes = i * 4096;
    const bool satisfied = i == 1;
    loom_target_compile_report_exact_residency_t exact_residency = {};
    exact_residency.contract_count = 1;
    exact_residency.evaluated_contract_count = 1;
    exact_residency.minimum_contract_count = i == 2 ? 1u : 0u;
    exact_residency.preserve_contract_count = i == 1 ? 1u : 0u;
    exact_residency.satisfied_contract_count = satisfied ? 1u : 0u;
    exact_residency.unsatisfied_contract_count = satisfied ? 0u : 1u;
    exact_residency.repaired_contract_count = satisfied ? 1u : 0u;
    exact_residency.projection_miss_count = i == 1 ? 1u : 0u;
    exact_residency.candidate_count = i + 1;
    exact_residency.attempted_candidate_count = i;
    exact_residency.repair_count = satisfied ? 1u : 0u;
    exact_residency.maximum_required_tier = (uint32_t)(i + 2);
    exact_residency.maximum_minimum_tier = i == 2 ? 4u : 0u;
    exact_residency.maximum_projected_baseline_tier = i == 1 ? 4u : 0u;
    exact_residency.minimum_allocated_tier = satisfied ? 4u : 2u;
    exact_residency.maximum_tier_shortfall = satisfied ? 0u : 2u;
    loom_target_compile_report_record_exact_residency(&entry, &exact_residency);
    IREE_ASSERT_OK(
        loom_target_compile_report_record_entry_report(&report, &entry));
    loom_target_compile_report_deinitialize(&entry);
  }

  EXPECT_EQ(report.low_planning.frame_build_count, 3u);
  EXPECT_EQ(report.low_planning.allocation_run_count, 5u);
  EXPECT_EQ(report.low_planning.repair.iteration_count, 7u);
  EXPECT_EQ(report.low_planning.residency.contract_count, 3u);
  EXPECT_EQ(report.low_planning.residency.candidate_count, 11u);
  EXPECT_EQ(report.low_planning.residency.maximum_projected_required_tier, 3u);
  EXPECT_EQ(report.low_planning.residency.minimum_observed_allocated_tier, 3u);
  EXPECT_EQ(report.low_planning.residency.maximum_observed_tier_shortfall, 1u);
  EXPECT_EQ(report.low_planning.memory.frame_arena.used_bytes_high_water, 200u);
  EXPECT_EQ(report.low_planning.memory.block_system_allocation_count, 3u);
  EXPECT_EQ(report.exact_residency.contract_count, 2u);
  EXPECT_EQ(report.exact_residency.evaluated_contract_count, 2u);
  EXPECT_EQ(report.exact_residency.unevaluated_contract_count, 0u);
  EXPECT_EQ(report.exact_residency.minimum_contract_count, 1u);
  EXPECT_EQ(report.exact_residency.preserve_contract_count, 1u);
  EXPECT_EQ(report.exact_residency.satisfied_contract_count, 1u);
  EXPECT_EQ(report.exact_residency.unsatisfied_contract_count, 1u);
  EXPECT_EQ(report.exact_residency.repaired_contract_count, 1u);
  EXPECT_EQ(report.exact_residency.projection_miss_count, 1u);
  EXPECT_EQ(report.exact_residency.candidate_count, 5u);
  EXPECT_EQ(report.exact_residency.attempted_candidate_count, 3u);
  EXPECT_EQ(report.exact_residency.repair_count, 1u);
  EXPECT_EQ(report.exact_residency.maximum_required_tier, 4u);
  EXPECT_EQ(report.exact_residency.maximum_minimum_tier, 4u);
  EXPECT_EQ(report.exact_residency.maximum_projected_baseline_tier, 4u);
  EXPECT_EQ(report.exact_residency.minimum_allocated_tier, 2u);
  EXPECT_EQ(report.exact_residency.maximum_tier_shortfall, 2u);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  const loom_target_compile_report_format_options_t options = {
      /*.mode=*/LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS,
  };
  IREE_ASSERT_OK(
      loom_target_compile_report_format_json(&report, &options, &stream));
  const iree_string_view_t root =
      ParseJsonDocument(iree_string_builder_view(&builder));
  ExpectObjectUint64Equals(root, IREE_SV("schema_version"),
                           LOOM_TARGET_COMPILE_REPORT_JSON_SCHEMA_VERSION);
  const iree_string_view_t planning = LookupObject(root, IREE_SV("planning"));
  ExpectObjectUint64Equals(planning, IREE_SV("frame_build_count"), 3);
  ExpectObjectUint64Equals(planning, IREE_SV("allocation_run_count"), 5);
  const iree_string_view_t repair = LookupObject(planning, IREE_SV("repair"));
  ExpectObjectUint64Equals(repair, IREE_SV("iteration_count"), 7);
  const iree_string_view_t residency =
      LookupObject(planning, IREE_SV("residency"));
  ExpectObjectUint64Equals(residency, IREE_SV("contract_count"), 3);
  ExpectObjectUint64Equals(residency, IREE_SV("candidate_count"), 11);
  ExpectObjectUint64Equals(residency,
                           IREE_SV("maximum_projected_required_tier"), 3);
  ExpectObjectUint64Equals(residency, IREE_SV("validation_count"), 13);
  ExpectObjectUint64Equals(residency,
                           IREE_SV("minimum_observed_allocated_tier"), 3);
  ExpectObjectUint64Equals(residency,
                           IREE_SV("maximum_observed_tier_shortfall"), 1);
  ExpectObjectUint64Equals(residency, IREE_SV("repair_attempt_count"), 15);
  ExpectObjectUint64Equals(residency, IREE_SV("repair_count"), 17);
  ExpectObjectUint64Equals(residency, IREE_SV("failure_count"), 1);
  const iree_string_view_t memory = LookupObject(planning, IREE_SV("memory"));
  const iree_string_view_t frame_arena =
      LookupObject(memory, IREE_SV("frame_arena"));
  ExpectObjectUint64Equals(frame_arena, IREE_SV("used_bytes_high_water"), 200);
  ExpectObjectUint64Equals(frame_arena, IREE_SV("owned_bytes_high_water"), 400);
  const iree_string_view_t system_allocations =
      LookupObject(memory, IREE_SV("system_allocations"));
  ExpectObjectUint64Equals(system_allocations, IREE_SV("block_count"), 3);
  ExpectObjectUint64Equals(system_allocations, IREE_SV("block_bytes"), 98304);
  ExpectObjectUint64Equals(system_allocations, IREE_SV("oversized_count"), 11);
  ExpectObjectUint64Equals(system_allocations, IREE_SV("oversized_bytes"),
                           12288);
  const iree_string_view_t exact_residency =
      LookupObject(root, IREE_SV("exact_residency"));
  ExpectObjectUint64Equals(exact_residency, IREE_SV("contract_count"), 2);
  ExpectObjectUint64Equals(exact_residency, IREE_SV("evaluated_contract_count"),
                           2);
  ExpectObjectUint64Equals(exact_residency,
                           IREE_SV("unevaluated_contract_count"), 0);
  ExpectObjectUint64Equals(exact_residency, IREE_SV("minimum_contract_count"),
                           1);
  ExpectObjectUint64Equals(exact_residency, IREE_SV("preserve_contract_count"),
                           1);
  ExpectObjectUint64Equals(exact_residency, IREE_SV("satisfied_contract_count"),
                           1);
  ExpectObjectUint64Equals(exact_residency,
                           IREE_SV("unsatisfied_contract_count"), 1);
  ExpectObjectUint64Equals(exact_residency, IREE_SV("maximum_required_tier"),
                           4);
  ExpectObjectUint64Equals(exact_residency, IREE_SV("maximum_minimum_tier"), 4);
  ExpectObjectUint64Equals(exact_residency,
                           IREE_SV("maximum_projected_baseline_tier"), 4);
  ExpectObjectUint64Equals(exact_residency, IREE_SV("minimum_allocated_tier"),
                           2);
  const iree_string_view_t entries = LookupObject(root, IREE_SV("entries"));
  const iree_string_view_t entry_rows = LookupObject(entries, IREE_SV("rows"));
  const iree_string_view_t first_entry = LookupArrayElement(entry_rows, 0);
  const iree_string_view_t first_exact_residency =
      LookupObject(first_entry, IREE_SV("exact_residency"));
  ExpectObjectUint64Equals(first_exact_residency,
                           IREE_SV("minimum_allocated_tier"), 4);
  ExpectObjectUint64Equals(first_exact_residency,
                           IREE_SV("repaired_contract_count"), 1);
  const iree_string_view_t second_entry = LookupArrayElement(entry_rows, 1);
  const iree_string_view_t second_exact_residency =
      LookupObject(second_entry, IREE_SV("exact_residency"));
  ExpectObjectUint64Equals(second_exact_residency,
                           IREE_SV("minimum_allocated_tier"), 2);
  ExpectObjectUint64Equals(second_exact_residency,
                           IREE_SV("unsatisfied_contract_count"), 1);

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

  const iree_string_view_t root =
      ParseJsonDocument(iree_string_builder_view(&builder));
  char backend_storage[32];
  iree_host_size_t backend_length = 0;
  IREE_ASSERT_OK(iree_json_lookup_string(
      root, IREE_SV("backend"),
      iree_make_mutable_string_view(backend_storage, sizeof(backend_storage)),
      &backend_length));
  EXPECT_TRUE(iree_string_view_equal(
      iree_make_string_view(backend_storage, backend_length),
      IREE_SV("quote\"line\n")));

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
