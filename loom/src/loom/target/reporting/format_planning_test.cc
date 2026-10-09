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

TEST(CompileReportFormatTest, FormatsAllocationPlanningRows) {
  loom_target_compile_report_t report = {};
  loom_target_compile_report_initialize(&report, iree_allocator_system());
  report.requested_detail_flags =
      LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ROWS |
      LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ORIGIN_ROWS |
      LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE_BAND_ROWS |
      LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE_BAND_SUMMARY_ROWS |
      LOOM_TARGET_COMPILE_REPORT_DETAIL_SPILL_ROWS |
      LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION_FAILURE_ROWS |
      LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION_HIGH_WATER_ROWS;
  loom_target_compile_report_record_allocation(
      &report, /*assignment_count=*/6, /*spill_count=*/1,
      /*spill_plan_count=*/1, /*coalesced_copy_count=*/2,
      /*materialized_copy_count=*/0, /*storage_lease_count=*/11,
      /*storage_lease_instance_count=*/9,
      /*storage_release_action_count=*/3);
  loom_target_compile_report_record_move_cause(
      &report, LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_LOW_CONCAT,
      /*packet_count=*/2, /*unit_count=*/8);

  loom_target_compile_report_pressure_row_t pressure = {
      .function_name = IREE_SVL("branchy"),
      .register_class = IREE_SVL("amdgpu.vgpr"),
      .type_kind = LOOM_TYPE_REGISTER,
      .element_type = LOOM_SCALAR_TYPE_I32,
      .peak_live_units = 96,
      .peak_live_values = 16,
      .peak_point = 4,
      .peak_block_name = IREE_SVL("entry"),
      .peak_operation_name = IREE_SVL("low.op<amdgpu.v_add_u32>")};
  IREE_ASSERT_OK(
      loom_target_compile_report_record_pressure_row(&report, &pressure));

  loom_target_compile_report_pressure_origin_row_t origin = {
      .function_name = IREE_SVL("branchy"),
      .register_class = pressure.register_class,
      .type_kind = LOOM_TYPE_REGISTER,
      .element_type = LOOM_SCALAR_TYPE_I32,
      .peak_point = 4,
      .peak_block_name = IREE_SVL("entry"),
      .peak_operation_name = pressure.peak_operation_name,
      .origin_kind = LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_DOT,
      .origin_operation_name = IREE_SVL("low.op<amdgpu.v_dot4_i32_i8>"),
      .semantic_tag = IREE_SVL("dot.i32.i8"),
      .sample_value_name = IREE_SVL("acc"),
      .live_units = 64,
      .live_values = 8};
  IREE_ASSERT_OK(
      loom_target_compile_report_record_pressure_origin_row(&report, &origin));

  loom_target_compile_report_schedule_band_row_t band =
      {};  // NOLINT(iree-cpp-designated-initializer) -- Assignment sequencing
           // spans intervening work.
  band.flags = LOOM_TARGET_COMPILE_REPORT_SCHEDULE_BAND_DYNAMIC_INSTRUCTION_MIX;
  band.function_name = IREE_SVL("branchy");
  band.block_name = IREE_SVL("body");
  band.block_index = 2;
  band.first_packet_index = 17;
  band.first_scheduled_ordinal = 5;
  band.node_count = 4;
  band.origin_kind = LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_LOCAL_MEMORY;
  band.origin_operation_name = IREE_SVL("low.op<amdgpu.ds_read2_b32>");
  band.semantic_tag = IREE_SVL("memory.workgroup.load2.u32");
  band.sample_value_name = IREE_SVL("tile");
  band.static_instruction_mix.descriptor_count = 4;
  band.static_instruction_mix.local_memory_count = 4;
  band.dynamic_instruction_mix.descriptor_count = 8;
  band.dynamic_instruction_mix.local_memory_count = 8;
  band.result_value_count = 4;
  band.result_unit_count = 16;
  IREE_ASSERT_OK(
      loom_target_compile_report_record_schedule_band_row(&report, &band));

  loom_target_compile_report_schedule_band_summary_row_t band_summary =
      {};  // NOLINT(iree-cpp-designated-initializer) -- Assignment sequencing
           // spans intervening work.
  band_summary.flags = band.flags;
  band_summary.function_name = band.function_name;
  band_summary.block_name = band.block_name;
  band_summary.block_index = band.block_index;
  band_summary.first_packet_index = band.first_packet_index;
  band_summary.band_count = 3;
  band_summary.node_count = 12;
  band_summary.max_band_node_count = 4;
  band_summary.origin_kind = band.origin_kind;
  band_summary.origin_operation_name = band.origin_operation_name;
  band_summary.semantic_tag = band.semantic_tag;
  band_summary.sample_value_name = band.sample_value_name;
  band_summary.static_instruction_mix.descriptor_count = 12;
  band_summary.static_instruction_mix.local_memory_count = 12;
  band_summary.dynamic_instruction_mix.descriptor_count = 24;
  band_summary.dynamic_instruction_mix.local_memory_count = 24;
  band_summary.result_value_count = 12;
  band_summary.result_unit_count = 48;
  IREE_ASSERT_OK(loom_target_compile_report_record_schedule_band_summary_row(
      &report, &band_summary));

  loom_target_compile_report_spill_row_t spill = {
      .kind = LOOM_TARGET_COMPILE_REPORT_SPILL_ROW_PLANNED,
      .function_name = IREE_SVL("branchy"),
      .value_name = IREE_SVL("rhs"),
      .register_class = IREE_SVL("amdgpu.vgpr"),
      .type_kind = LOOM_TYPE_REGISTER,
      .element_type = LOOM_SCALAR_TYPE_I32,
      .origin_kind = LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_DOT,
      .origin_operation_name = origin.origin_operation_name,
      .semantic_tag = origin.semantic_tag,
      .assignment_index = 2,
      .slot_index = 1,
      .slot_space = IREE_SVL("stack"),
      .byte_size = 4,
      .byte_alignment = 4,
      .store_count = 1,
      .store_bytes = 4,
      .reload_count = 2,
      .reload_bytes = 8};
  IREE_ASSERT_OK(loom_target_compile_report_record_spill_row(&report, &spill));

  loom_target_compile_report_allocation_failure_row_t failure = {
      .function_name = IREE_SVL("branchy"),
      .value_name = IREE_SVL("blocked"),
      .register_class = IREE_SVL("test.scc"),
      .type_kind = LOOM_TYPE_REGISTER,
      .element_type = LOOM_SCALAR_TYPE_INDEX,
      .failure_code = IREE_SVL("unspillable-register-exhausted"),
      .blocking_kind =
          LOOM_TARGET_COMPILE_REPORT_ALLOCATION_FAILURE_BLOCKING_ACTIVE_ASSIGNMENT,
      .origin_operation_name = IREE_SVL("low.return"),
      .origin_block_name = IREE_SVL("entry"),
      .start_point = 2,
      .end_point = 5,
      .required_unit_count = 1,
      .budget_units = 1,
      .peak_live_units = 2,
      .location_kind = IREE_SVL("physical_register"),
      .location_count = 1,
      .conflict_value_name = IREE_SVL("leader"),
      .conflict_start_point = 0,
      .conflict_end_point = 5};
  IREE_ASSERT_OK(loom_target_compile_report_record_allocation_failure_row(
      &report, &failure));

  loom_target_compile_report_allocation_high_water_row_t high_water = {
      .function_name = IREE_SVL("branchy"),
      .value_name = IREE_SVL("rhs_window"),
      .register_class = IREE_SVL("amdgpu.vgpr"),
      .type_kind = LOOM_TYPE_REGISTER,
      .element_type = LOOM_SCALAR_TYPE_I32,
      .assignment_index = 5,
      .origin_operation_name = IREE_SVL("low.op<amdgpu.ds_load_b128>"),
      .origin_kind = LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_LOCAL_MEMORY,
      .semantic_tag = IREE_SVL("memory.workgroup.load.u128"),
      .start_point = 17,
      .end_point = 24,
      .required_unit_count = 4,
      .location_kind = IREE_SVL("physical_register"),
      .location_base = 248,
      .location_count = 4,
      .high_water_units = 252,
      .lower_free_unit_count = 13,
      .lower_largest_free_run_unit_count = 6,
      .active_assignment_blocker_count = 47,
      .active_assignment_blocker_units = 244};
  IREE_ASSERT_OK(loom_target_compile_report_record_allocation_high_water_row(
      &report, &high_water));

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  const loom_target_compile_report_format_options_t options = {
      .mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS,
  };
  IREE_ASSERT_OK(
      loom_target_compile_report_format_text(&report, &options, &builder));
  const iree_string_view_t text = iree_string_builder_view(&builder);
  EXPECT_NE(
      iree_string_view_find(
          text, IREE_SV("pressure[0] function=branchy class=amdgpu.vgpr"), 0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(
          text, IREE_SV("schedule_band[0] function=branchy block=body"), 0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(
          text, IREE_SV("spill[0] kind=planned function=branchy value=rhs"), 0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(
          text, IREE_SV("allocation_failure[0] function=branchy value=blocked"),
          0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(text,
                            IREE_SV("allocation_high_water[0] function=branchy "
                                    "value=rhs_window"),
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
  const iree_string_view_t pressure_rows =
      LookupObject(root, IREE_SV("pressure_rows"));
  const iree_string_view_t pressure_row = LookupArrayElement(
      LookupObject(pressure_rows, IREE_SV("rows")), /*index=*/0);
  ExpectObjectUint64Equals(pressure_row, IREE_SV("peak_live_units"), 96);
  const iree_string_view_t origin_rows =
      LookupObject(root, IREE_SV("pressure_origin_rows"));
  const iree_string_view_t origin_row = LookupArrayElement(
      LookupObject(origin_rows, IREE_SV("rows")), /*index=*/0);
  ExpectObjectValueEquals(origin_row, IREE_SV("semantic_tag"),
                          IREE_SV("dot.i32.i8"));
  const iree_string_view_t band_rows =
      LookupObject(root, IREE_SV("schedule_band_rows"));
  const iree_string_view_t band_row =
      LookupArrayElement(LookupObject(band_rows, IREE_SV("rows")), /*index=*/0);
  ExpectObjectUint64Equals(band_row, IREE_SV("result_unit_count"), 16);
  const iree_string_view_t summary_rows =
      LookupObject(root, IREE_SV("schedule_band_summary_rows"));
  const iree_string_view_t summary_row = LookupArrayElement(
      LookupObject(summary_rows, IREE_SV("rows")), /*index=*/0);
  ExpectObjectUint64Equals(summary_row, IREE_SV("band_count"), 3);
  const iree_string_view_t spill_rows =
      LookupObject(root, IREE_SV("spill_rows"));
  const iree_string_view_t spill_row = LookupArrayElement(
      LookupObject(spill_rows, IREE_SV("rows")), /*index=*/0);
  ExpectObjectUint64Equals(spill_row, IREE_SV("reload_bytes"), 8);
  const iree_string_view_t failure_rows =
      LookupObject(root, IREE_SV("allocation_failure_rows"));
  const iree_string_view_t failure_row = LookupArrayElement(
      LookupObject(failure_rows, IREE_SV("rows")), /*index=*/0);
  ExpectObjectValueEquals(failure_row, IREE_SV("failure_code"),
                          IREE_SV("unspillable-register-exhausted"));
  const iree_string_view_t high_water_rows =
      LookupObject(root, IREE_SV("allocation_high_water_rows"));
  const iree_string_view_t high_water_row = LookupArrayElement(
      LookupObject(high_water_rows, IREE_SV("rows")), /*index=*/0);
  ExpectObjectUint64Equals(high_water_row, IREE_SV("high_water_units"), 252);
  const iree_string_view_t allocation =
      LookupObject(root, IREE_SV("allocation"));
  ExpectObjectUint64Equals(allocation, IREE_SV("spill_count"), 1);
  const iree_string_view_t move_causes =
      LookupObject(root, IREE_SV("move_causes"));
  ExpectObjectUint64Equals(move_causes, IREE_SV("packet_count"), 2);

  iree_string_builder_deinitialize(&builder);
  loom_target_compile_report_deinitialize(&report);
}

TEST(CompileReportFormatTest, FormatsWaitPlanningRows) {
  loom_target_compile_report_t report = {};
  loom_target_compile_report_initialize(&report, iree_allocator_system());
  report.requested_detail_flags = LOOM_TARGET_COMPILE_REPORT_DETAIL_WAIT_PLAN;

  loom_target_compile_report_wait_plan_t wait_plan = {
      .action_count = 5,
      .explicit_action_count = 1,
      .planned_action_count = 4,
      .full_drain_count = 2,
      .partial_wait_count = 3,
      .drained_count = 6,
      .max_drained_count = 4,
      .max_outstanding_before = 6,
      .max_full_drain_outstanding_before = 6};
  loom_target_compile_report_record_wait_plan(&report, &wait_plan);

  loom_target_compile_report_wait_counter_row_t counter = {
      .function_name = IREE_SVL("branchy"),
      .counter_name = IREE_SVL("vmem_load"),
      .counter_id = 1};
  counter.summary.action_count = 3;
  counter.summary.planned_action_count = 3;
  counter.summary.full_drain_count = 1;
  counter.summary.partial_wait_count = 2;
  counter.summary.drained_count = 4;
  counter.summary.max_drained_count = 4;
  counter.summary.max_outstanding_before = 6;
  counter.summary.max_full_drain_outstanding_before = 6;
  IREE_ASSERT_OK(
      loom_target_compile_report_record_wait_counter_row(&report, &counter));

  loom_target_compile_report_wait_reason_summary_row_t reason = {
      .function_name = counter.function_name,
      .counter_name = counter.counter_name,
      .reason_name = IREE_SVL("amdgpu.ssa_use"),
      .counter_id = 1,
      .reason_id = 2};
  reason.summary.action_count = 1;
  reason.summary.planned_action_count = 1;
  reason.summary.partial_wait_count = 1;
  reason.summary.drained_count = 4;
  reason.summary.max_drained_count = 4;
  reason.summary.max_outstanding_before = 6;
  IREE_ASSERT_OK(loom_target_compile_report_record_wait_reason_summary_row(
      &report, &reason));

  loom_target_compile_report_wait_action_row_t action = {
      .function_name = counter.function_name,
      .counter_name = counter.counter_name,
      .action_name = IREE_SVL("planned"),
      .reason_name = reason.reason_name,
      .counter_id = 1,
      .action_id = 2,
      .reason_id = 2,
      .block_index = 1,
      .node_index = 42,
      .scheduled_ordinal = 17,
      .producer_node = 8,
      .producer_scheduled_ordinal = 3,
      .producer_operation_name = IREE_SVL("low.op<amdgpu.global_load_b32>"),
      .producer_descriptor_key = IREE_SVL("amdgpu.global_load_b32"),
      .producer_semantic_tag = IREE_SVL("memory.load.u32"),
      .consumer_node = 42,
      .consumer_scheduled_ordinal = 17,
      .consumer_operation_name = IREE_SVL("low.op<amdgpu.v_add_u32>"),
      .consumer_descriptor_key = IREE_SVL("amdgpu.v_add_u32"),
      .consumer_semantic_tag = IREE_SVL("vector.add.i32"),
      .target_count = 2,
      .outstanding_before = 6,
      .outstanding_after = 2,
      .drained_count = 4};
  IREE_ASSERT_OK(
      loom_target_compile_report_record_wait_action_row(&report, &action));

  loom_target_compile_report_wait_action_row_t explicit_action = action;
  explicit_action.counter_name = IREE_SVL("smem");
  explicit_action.action_name = IREE_SVL("explicit");
  explicit_action.reason_name = IREE_SVL("amdgpu.explicit_packet");
  explicit_action.counter_id = 4;
  explicit_action.action_id = 1;
  explicit_action.reason_id = 1;
  explicit_action.node_index = 44;
  explicit_action.scheduled_ordinal = 19;
  explicit_action.producer_node = UINT32_MAX;
  explicit_action.producer_scheduled_ordinal = UINT32_MAX;
  explicit_action.producer_operation_name = iree_string_view_empty();
  explicit_action.producer_descriptor_key = iree_string_view_empty();
  explicit_action.producer_semantic_tag = iree_string_view_empty();
  explicit_action.consumer_node = UINT32_MAX;
  explicit_action.consumer_scheduled_ordinal = UINT32_MAX;
  explicit_action.consumer_operation_name = iree_string_view_empty();
  explicit_action.consumer_descriptor_key = iree_string_view_empty();
  explicit_action.consumer_semantic_tag = iree_string_view_empty();
  explicit_action.target_count = 0;
  explicit_action.outstanding_before = 2;
  explicit_action.outstanding_after = 0;
  explicit_action.drained_count = 2;
  IREE_ASSERT_OK(loom_target_compile_report_record_wait_action_row(
      &report, &explicit_action));

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  const loom_target_compile_report_format_options_t options = {
      .mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS,
  };
  IREE_ASSERT_OK(
      loom_target_compile_report_format_text(&report, &options, &builder));
  const iree_string_view_t text = iree_string_builder_view(&builder);
  EXPECT_NE(iree_string_view_find(text, IREE_SV("wait_plan actions=5"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(text,
                                  IREE_SV("wait_counter[0] function=branchy "
                                          "counter=vmem_load"),
                                  0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(
          text, IREE_SV("wait_action[1] function=branchy counter=smem"), 0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(text, IREE_SV("producer_node=-"), 0),
            IREE_STRING_VIEW_NPOS);
  iree_string_builder_deinitialize(&builder);

  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  IREE_ASSERT_OK(
      loom_target_compile_report_format_json(&report, &options, &stream));
  const iree_string_view_t root =
      ParseJsonDocument(iree_string_builder_view(&builder));
  const iree_string_view_t wait_plan_json =
      LookupObject(root, IREE_SV("wait_plan"));
  ExpectObjectUint64Equals(wait_plan_json, IREE_SV("action_count"), 5);
  const iree_string_view_t counter_rows =
      LookupObject(root, IREE_SV("wait_counter_rows"));
  const iree_string_view_t counter_row = LookupArrayElement(
      LookupObject(counter_rows, IREE_SV("rows")), /*index=*/0);
  ExpectObjectValueEquals(counter_row, IREE_SV("counter"),
                          IREE_SV("vmem_load"));
  const iree_string_view_t reason_rows =
      LookupObject(root, IREE_SV("wait_reason_summary_rows"));
  const iree_string_view_t reason_row = LookupArrayElement(
      LookupObject(reason_rows, IREE_SV("rows")), /*index=*/0);
  ExpectObjectValueEquals(reason_row, IREE_SV("reason"),
                          IREE_SV("amdgpu.ssa_use"));
  const iree_string_view_t action_rows =
      LookupObject(root, IREE_SV("wait_action_rows"));
  const iree_string_view_t planned_row = LookupArrayElement(
      LookupObject(action_rows, IREE_SV("rows")), /*index=*/0);
  ExpectObjectValueEquals(planned_row, IREE_SV("producer_semantic_tag"),
                          IREE_SV("memory.load.u32"));
  const iree_string_view_t explicit_row = LookupArrayElement(
      LookupObject(action_rows, IREE_SV("rows")), /*index=*/1);
  ExpectObjectValueEquals(explicit_row, IREE_SV("producer_node"),
                          IREE_SV("null"));
  ExpectObjectValueEquals(explicit_row, IREE_SV("consumer_operation"),
                          IREE_SV("null"));

  iree_string_builder_deinitialize(&builder);
  loom_target_compile_report_deinitialize(&report);
}

}  // namespace
}  // namespace loom
