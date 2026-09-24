// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/reporting/format.h"

#include <stdint.h>

#include "iree/base/internal/arena.h"
#include "iree/base/internal/json.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

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

TEST(CompileReportFormatTest, FormatsCoreReport) {
  loom_target_compile_report_t report = {};
  loom_target_compile_report_initialize(&report, iree_allocator_system());
  report.artifact_kind = LOOM_TARGET_COMPILE_ARTIFACT_KIND_TARGET_ARTIFACT;
  report.backend_name = IREE_SVL("amdgpu-hal");
  report.function_name = IREE_SVL("branchy");
  report.target_bundle_name = IREE_SVL("test_target");
  report.target_export_name = IREE_SVL("test_export");
  report.target_export_symbol = IREE_SVL("branchy_export");
  report.target_config_name = IREE_SVL("test_o0");
  report.lowered_symbol = IREE_SVL("branchy");
  loom_target_compile_report_record_artifact_size(&report, 128);
  loom_target_compile_report_record_schedule(
      &report, /*node_count=*/5, /*scheduled_node_count=*/5,
      /*dependency_count=*/4, /*scope_count=*/2, /*resource_use_count=*/2,
      /*hazard_gap_count=*/1, /*model_summary_count=*/1,
      /*pressure_summary_count=*/2, /*peak_live_units=*/96);

  loom_target_compile_report_static_instruction_mix_t static_mix = {};
  static_mix.descriptor_count = 9;
  static_mix.scalar_alu_count = 2;
  static_mix.vector_alu_count = 3;
  static_mix.matrix_count = 2;
  static_mix.wmma_count = 1;
  static_mix.global_memory_count = 2;
  static_mix.global_load_count = 1;
  static_mix.buffer_load_count = 1;
  static_mix.execution_barrier_count = 1;
  loom_target_compile_report_record_static_instruction_mix(&report,
                                                           &static_mix);
  loom_target_compile_report_static_instruction_mix_t dynamic_mix = static_mix;
  dynamic_mix.global_memory_count = 3;
  dynamic_mix.global_store_count = 1;
  dynamic_mix.memory_read_byte_count = 24;
  dynamic_mix.memory_write_byte_count = 8;
  dynamic_mix.global_load_byte_count = 16;
  dynamic_mix.global_store_byte_count = 8;
  dynamic_mix.buffer_load_byte_count = 8;
  loom_target_compile_report_record_dynamic_instruction_mix(&report,
                                                            &dynamic_mix);
  loom_target_compile_report_record_emission(&report, /*instruction_count=*/8,
                                             /*code_byte_count=*/64,
                                             /*code_storage_byte_count=*/80);
  const loom_target_compile_report_emission_breakdown_t emission_breakdown = {
      /*.body_instruction_count=*/6,
      /*.entry_instruction_count=*/2,
      /*.coissued_instruction_count=*/1,
      /*.coissued_component_count=*/2,
  };
  loom_target_compile_report_record_emission_breakdown(&report,
                                                       &emission_breakdown);
  loom_target_compile_report_record_memory(&report, /*private_memory_bytes=*/16,
                                           /*local_memory_bytes=*/32);

  loom_target_compile_report_workload_t workload = {};
  workload.flags =
      LOOM_TARGET_COMPILE_REPORT_WORKLOAD_WORKGROUP_SIZE |
      LOOM_TARGET_COMPILE_REPORT_WORKLOAD_WORKGROUP_COUNT |
      LOOM_TARGET_COMPILE_REPORT_WORKLOAD_FLAT_WORKGROUP_SIZE |
      LOOM_TARGET_COMPILE_REPORT_WORKLOAD_DISPATCH_WORKGROUP_COUNT |
      LOOM_TARGET_COMPILE_REPORT_WORKLOAD_DISPATCH_WORKITEM_COUNT |
      LOOM_TARGET_COMPILE_REPORT_WORKLOAD_WORKGROUP_CLUSTER_SIZE |
      LOOM_TARGET_COMPILE_REPORT_WORKLOAD_FLAT_WORKGROUP_CLUSTER_SIZE;
  workload.workgroup_size.x = 64;
  workload.workgroup_size.y = 2;
  workload.workgroup_size.z = 1;
  workload.workgroup_count.x = 8;
  workload.workgroup_count.y = 4;
  workload.workgroup_count.z = 1;
  workload.workgroup_cluster_size.x = 1;
  workload.workgroup_cluster_size.y = 2;
  workload.workgroup_cluster_size.z = 1;
  workload.flat_workgroup_size = 128;
  workload.dispatch_workgroup_count = 32;
  workload.dispatch_workitem_count = 4096;
  workload.flat_workgroup_cluster_size = 2;
  loom_target_compile_report_record_workload(&report, &workload);

  loom_target_compile_report_target_resources_t resources = {};
  resources.scalar_register_class = IREE_SVL("amdgpu.sgpr");
  resources.scalar_register_count = 38;
  resources.scalar_pressure_peak_live_units = 32;
  resources.scalar_register_overhead_units = 6;
  resources.vector_register_class = IREE_SVL("amdgpu.vgpr");
  resources.vector_register_count = 160;
  resources.vector_pressure_peak_live_units = 136;
  resources.vector_register_overhead_units = 24;
  resources.subgroup_size = 64;
  resources.max_subgroups_per_simd = 16;
  resources.resident_subgroups_per_simd = 3;
  resources.occupancy_percent = 18;
  resources.limiting_resource = IREE_SVL("amdgpu.vgpr");
  resources.residency_summary.flags =
      LOOM_TARGET_RESIDENCY_SUMMARY_FLAG_VALID |
      LOOM_TARGET_RESIDENCY_SUMMARY_FLAG_HAS_NEXT_BETTER_TIER |
      LOOM_TARGET_RESIDENCY_SUMMARY_FLAG_HAS_UNIQUE_LIMITING_RESOURCE |
      LOOM_TARGET_RESIDENCY_SUMMARY_FLAG_HAS_LIMITING_RESOURCE_NEXT_WORSE_TIER;
  resources.residency_summary.best_tier = 16;
  resources.residency_summary.tier = 3;
  resources.residency_summary.next_better_tier = 4;
  resources.residency_summary.limiting_resource_count = 1;
  resources.residency_summary.limiting_resource = IREE_SVL("amdgpu.vgpr");
  resources.residency_summary.limiting_resource_units = 160;
  resources.residency_summary
      .limiting_resource_reduction_units_to_next_better_tier = 32;
  resources.residency_summary.limiting_resource_next_worse_tier = 2;
  resources.residency_summary.limiting_resource_next_worse_cliff_units = 176;
  resources.residency_summary
      .limiting_resource_additional_units_to_next_worse_tier = 16;
  loom_target_compile_report_record_target_resources(&report, &resources);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  const loom_target_compile_report_format_options_t options = {
      /*.mode=*/LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS,
  };
  IREE_ASSERT_OK(
      loom_target_compile_report_format_text(&report, &options, &builder));
  const iree_string_view_t text = iree_string_builder_view(&builder);
  EXPECT_NE(iree_string_view_find(text, IREE_SV("artifact=target-artifact"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(text, IREE_SV("workload workgroup_size=64x2x1"), 0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                text, IREE_SV("cluster_size=1x2x1 flat_cluster_size=2"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(
          text, IREE_SV("target_resources scalar_register_class=amdgpu.sgpr"),
          0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(text,
                                  IREE_SV("residency_current_tier=3 "
                                          "residency_limiting_resource_count=1 "
                                          "residency_next_better_tier=4"),
                                  0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(
          text, IREE_SV("residency_reduction_units_to_next_better_tier=32"), 0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(
          text, IREE_SV("economics memory per_workitem_issued_read_bytes=24"),
          0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                text,
                IREE_SV("emission instructions=8 code_bytes=64 "
                        "storage_bytes=80 body_instructions=6 "
                        "entry_instructions=2 coissued_instructions=1 "
                        "coissued_components=2"),
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
  ExpectObjectValueEquals(root, IREE_SV("kind"),
                          IREE_SV("loom.compile_report"));
  ExpectObjectUint64Equals(root, IREE_SV("schema_version"), 0);
  ExpectObjectValueEquals(root, IREE_SV("mode"), IREE_SV("details"));
  ExpectObjectValueEquals(root, IREE_SV("artifact_kind"),
                          IREE_SV("target-artifact"));
  ExpectObjectValueEquals(root, IREE_SV("backend"), IREE_SV("amdgpu-hal"));
  ExpectObjectUint64Equals(root, IREE_SV("artifact_size"), 128);
  const iree_string_view_t schedule = LookupObject(root, IREE_SV("schedule"));
  ExpectObjectUint64Equals(schedule, IREE_SV("node_count"), 5);
  ExpectObjectUint64Equals(schedule, IREE_SV("scope_count"), 2);
  ExpectObjectUint64Equals(schedule, IREE_SV("resource_use_count"), 2);
  const iree_string_view_t workload_json =
      LookupObject(root, IREE_SV("workload"));
  const iree_string_view_t workgroup_size =
      LookupObject(workload_json, IREE_SV("workgroup_size"));
  ExpectObjectUint64Equals(workgroup_size, IREE_SV("x"), 64);
  ExpectObjectUint64Equals(workgroup_size, IREE_SV("flat"), 128);
  const iree_string_view_t cluster_size =
      LookupObject(workload_json, IREE_SV("cluster_size"));
  ExpectObjectUint64Equals(cluster_size, IREE_SV("x"), 1);
  ExpectObjectUint64Equals(cluster_size, IREE_SV("y"), 2);
  ExpectObjectUint64Equals(cluster_size, IREE_SV("z"), 1);
  ExpectObjectUint64Equals(cluster_size, IREE_SV("flat"), 2);
  ExpectObjectUint64Equals(workload_json, IREE_SV("dispatch_workitem_count"),
                           4096);
  const iree_string_view_t mix =
      LookupObject(root, IREE_SV("dynamic_instruction_mix"));
  ExpectObjectUint64Equals(mix, IREE_SV("vector_alu_count"), 3);
  ExpectObjectUint64Equals(mix, IREE_SV("global_store_count"), 1);
  ExpectObjectUint64Equals(mix, IREE_SV("execution_barrier_count"), 1);
  const iree_string_view_t memory = LookupObject(root, IREE_SV("memory"));
  ExpectObjectUint64Equals(memory, IREE_SV("private_bytes"), 16);
  ExpectObjectUint64Equals(memory, IREE_SV("local_bytes"), 32);
  const iree_string_view_t emission = LookupObject(root, IREE_SV("emission"));
  ExpectObjectUint64Equals(emission, IREE_SV("instruction_count"), 8);
  ExpectObjectUint64Equals(emission, IREE_SV("body_instruction_count"), 6);
  ExpectObjectUint64Equals(emission, IREE_SV("entry_instruction_count"), 2);
  ExpectObjectUint64Equals(emission, IREE_SV("coissued_instruction_count"), 1);
  ExpectObjectUint64Equals(emission, IREE_SV("coissued_component_count"), 2);
  const iree_string_view_t target_resources =
      LookupObject(root, IREE_SV("target_resources"));
  const iree_string_view_t vector_resources =
      LookupObject(target_resources, IREE_SV("vector"));
  const iree_string_view_t final_vector_resources =
      LookupObject(vector_resources, IREE_SV("final"));
  ExpectObjectUint64Equals(final_vector_resources, IREE_SV("register_count"),
                           160);
  ExpectObjectUint64Equals(target_resources, IREE_SV("occupancy_percent"), 18);
  const iree_string_view_t residency =
      LookupObject(target_resources, IREE_SV("residency"));
  ExpectObjectUint64Equals(residency, IREE_SV("best_tier"), 16);
  ExpectObjectUint64Equals(residency, IREE_SV("current_tier"), 3);
  ExpectObjectUint64Equals(residency, IREE_SV("next_better_tier"), 4);
  const iree_string_view_t limiting =
      LookupObject(residency, IREE_SV("unique_limiting_resource"));
  ExpectObjectValueEquals(limiting, IREE_SV("name"), IREE_SV("amdgpu.vgpr"));
  ExpectObjectUint64Equals(limiting,
                           IREE_SV("reduction_units_to_next_better_tier"), 32);
  iree_string_builder_deinitialize(&builder);
  loom_target_compile_report_deinitialize(&report);
}

TEST(CompileReportFormatTest, EmitsOnlyValidResidencyEvidence) {
  loom_target_compile_report_t report = {};
  loom_target_compile_report_initialize(&report, iree_allocator_system());
  loom_target_compile_report_target_resources_t resources = {};
  resources.residency_summary.flags =
      LOOM_TARGET_RESIDENCY_SUMMARY_FLAG_VALID |
      LOOM_TARGET_RESIDENCY_SUMMARY_FLAG_HAS_NEXT_BETTER_TIER;
  resources.residency_summary.best_tier = 16;
  resources.residency_summary.tier = 3;
  resources.residency_summary.next_better_tier = 4;
  resources.residency_summary.limiting_resource_count = 2;
  loom_target_compile_report_record_target_resources(&report, &resources);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  const loom_target_compile_report_format_options_t options = {
      /*.mode=*/LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS,
  };
  IREE_ASSERT_OK(
      loom_target_compile_report_format_json(&report, &options, &stream));
  iree_string_view_t output = iree_string_builder_view(&builder);
  EXPECT_NE(iree_string_view_find(output,
                                  IREE_SV("\"limiting_resource_count\":2"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(
      iree_string_view_find(output, IREE_SV("\"unique_limiting_resource\""), 0),
      IREE_STRING_VIEW_NPOS);
  iree_string_builder_deinitialize(&builder);
  loom_target_compile_report_deinitialize(&report);

  loom_target_compile_report_initialize(&report, iree_allocator_system());
  loom_target_compile_report_record_target_resources(&report, &resources);
  report.target_resources.residency_summary = {};
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_for_builder(&builder, &stream);
  IREE_ASSERT_OK(
      loom_target_compile_report_format_json(&report, &options, &stream));
  output = iree_string_builder_view(&builder);
  EXPECT_EQ(iree_string_view_find(output, IREE_SV("\"residency\""), 0),
            IREE_STRING_VIEW_NPOS);
  iree_string_builder_deinitialize(&builder);
  loom_target_compile_report_deinitialize(&report);
}

TEST(CompileReportFormatTest, OwnsResidencyConstraintsAcrossEntryMerge) {
  loom_target_compile_report_t report = {};
  loom_target_compile_report_initialize(&report, iree_allocator_system());
  for (const auto* function_name : {"attention", "projection"}) {
    iree_arena_block_pool_t pool;
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool);
    iree_arena_allocator_t arena;
    iree_arena_initialize(&pool, &arena);
    loom_target_residency_constraint_t* rows = nullptr;
    IREE_ASSERT_OK(iree_arena_allocate_array(&arena, 2, sizeof(*rows),
                                             reinterpret_cast<void**>(&rows)));
    rows[0] = {};
    rows[0].name = IREE_SVL("amdgpu.lds");
    rows[0].kind = LOOM_TARGET_RESIDENCY_CONSTRAINT_POOLED_RESOURCE;
    rows[0].flags =
        LOOM_TARGET_RESIDENCY_CONSTRAINT_FLAG_HAS_USAGE |
        LOOM_TARGET_RESIDENCY_CONSTRAINT_FLAG_HAS_TIER |
        LOOM_TARGET_RESIDENCY_CONSTRAINT_FLAG_HAS_LIMITING_RELATION |
        LOOM_TARGET_RESIDENCY_CONSTRAINT_FLAG_LIMITING |
        LOOM_TARGET_RESIDENCY_CONSTRAINT_FLAG_HAS_REDUCTION;
    rows[0].unit = IREE_SVL("bytes");
    rows[0].allocation_scope = IREE_SVL("workgroup");
    rows[0].pool_scope = IREE_SVL("occupancy domain");
    rows[0].units = 15616;
    rows[0].rounded_units = 15872;
    rows[0].pool_units = 131072;
    rows[0].allocation_granularity = 512;
    rows[0].tier = 8;
    rows[0].reduction_units_to_next_better_tier = 1280;
    rows[1] = {};
    rows[1].name = IREE_SVL("amdgpu.workgroup_slots");
    rows[1].kind = LOOM_TARGET_RESIDENCY_CONSTRAINT_FIXED_LIMIT;
    rows[1].flags = LOOM_TARGET_RESIDENCY_CONSTRAINT_FLAG_HAS_TIER |
                    LOOM_TARGET_RESIDENCY_CONSTRAINT_FLAG_HAS_LIMITING_RELATION;
    rows[1].tier = 16;
    const loom_target_residency_constraint_list_t constraints = {rows, 2};
    loom_target_compile_report_t entry = {};
    loom_target_compile_report_initialize(&entry, iree_allocator_system());
    entry.function_name = iree_make_cstring_view(function_name);
    IREE_ASSERT_OK(loom_target_compile_report_record_residency_constraints(
        &entry, &constraints));
    iree_arena_deinitialize(&arena);
    iree_arena_block_pool_deinitialize(&pool);
    IREE_ASSERT_OK(
        loom_target_compile_report_record_entry_report(&report, &entry));
    loom_target_compile_report_deinitialize(&entry);
  }
  EXPECT_EQ(report.residency_constraint_rows.count, 4u);
  for (auto mode : {LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_SUMMARY,
                    LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS}) {
    const loom_target_compile_report_format_options_t options = {mode};
    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    loom_output_stream_t stream;
    loom_output_stream_for_builder(&builder, &stream);
    IREE_ASSERT_OK(
        loom_target_compile_report_format_json(&report, &options, &stream));
    const auto root = ParseJsonDocument(iree_string_builder_view(&builder));
    const auto inventory = LookupObject(root, IREE_SV("residency_constraints"));
    ExpectObjectUint64Equals(inventory, IREE_SV("count"), 4);
    const auto json_rows = LookupObject(inventory, IREE_SV("rows"));
    for (iree_host_size_t i = 0; i < 4; ++i) {
      const auto row = LookupArrayElement(json_rows, i);
      ExpectObjectUint64Equals(row, IREE_SV("index"), i);
      ExpectObjectValueEquals(
          row, IREE_SV("function"),
          i < 2 ? IREE_SV("attention") : IREE_SV("projection"));
      if (i % 2 == 0) {
        ExpectObjectValueEquals(row, IREE_SV("name"), IREE_SV("amdgpu.lds"));
        ExpectObjectValueEquals(row, IREE_SV("unit"), IREE_SV("bytes"));
        ExpectObjectValueEquals(row, IREE_SV("allocation_scope"),
                                IREE_SV("workgroup"));
        ExpectObjectValueEquals(row, IREE_SV("limiting"), IREE_SV("true"));
        ExpectObjectUint64Equals(row, IREE_SV("units"), 15616);
        ExpectObjectUint64Equals(row, IREE_SV("rounded_units"), 15872);
        ExpectObjectUint64Equals(row, IREE_SV("pool_units"), 131072);
        ExpectObjectUint64Equals(row, IREE_SV("allocation_granularity"), 512);
        ExpectObjectUint64Equals(row, IREE_SV("independent_tier"), 8);
        ExpectObjectUint64Equals(
            row, IREE_SV("reduction_units_to_next_better_tier"), 1280);
      } else {
        ExpectObjectValueEquals(row, IREE_SV("kind"), IREE_SV("fixed_limit"));
        ExpectObjectValueEquals(row, IREE_SV("limiting"), IREE_SV("false"));
        ExpectObjectUint64Equals(row, IREE_SV("independent_tier"), 16);
        EXPECT_EQ(iree_string_view_find(row, IREE_SV("\"units\""), 0),
                  IREE_STRING_VIEW_NPOS);
        EXPECT_EQ(
            iree_string_view_find(
                row, IREE_SV("\"reduction_units_to_next_better_tier\""), 0),
            IREE_STRING_VIEW_NPOS);
      }
    }
    iree_string_builder_deinitialize(&builder);
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    IREE_ASSERT_OK(
        loom_target_compile_report_format_text(&report, &options, &builder));
    EXPECT_NE(
        iree_string_view_find(
            iree_string_builder_view(&builder),
            IREE_SV("units=15616 rounded_units=15872 independent_tier=8 "
                    "limiting=true reduction_units_to_next_better_tier=1280"),
            0),
        IREE_STRING_VIEW_NPOS);
    iree_string_builder_deinitialize(&builder);
  }
  loom_target_compile_report_deinitialize(&report);
}

TEST(CompileReportFormatTest,
     ExplainsUnavailableResidencyWithoutInventingFacts) {
  loom_target_compile_report_t report = {};
  loom_target_compile_report_initialize(&report, iree_allocator_system());
  report.function_name = IREE_SVL("attention");
  loom_target_compile_report_target_resources_t resources = {};
  resources.residency_summary.flags =
      LOOM_TARGET_RESIDENCY_SUMMARY_FLAG_INCOMPLETE_RESOURCE_COUNTS |
      LOOM_TARGET_RESIDENCY_SUMMARY_FLAG_UNKNOWN_WORKGROUP_SIZE;
  loom_target_compile_report_record_target_resources(&report, &resources);
  loom_target_residency_constraint_t resource = {};
  resource.name = IREE_SVL("amdgpu.vgpr_agpr");
  resource.kind = LOOM_TARGET_RESIDENCY_CONSTRAINT_POOLED_RESOURCE;
  resource.unit = IREE_SVL("registers");
  resource.allocation_scope = IREE_SVL("subgroup");
  resource.pool_scope = IREE_SVL("SIMD");
  resource.pool_units = 512;
  resource.allocation_granularity = 8;
  const loom_target_residency_constraint_list_t constraints = {&resource, 1};
  IREE_ASSERT_OK(loom_target_compile_report_record_residency_constraints(
      &report, &constraints));
  const loom_target_compile_report_format_options_t options = {
      LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_SUMMARY};
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  IREE_ASSERT_OK(
      loom_target_compile_report_format_json(&report, &options, &stream));
  const auto root = ParseJsonDocument(iree_string_builder_view(&builder));
  const auto residency = LookupObject(
      LookupObject(root, IREE_SV("target_resources")), IREE_SV("residency"));
  const auto reasons = LookupObject(residency, IREE_SV("unavailable_reasons"));
  EXPECT_NE(
      iree_string_view_find(reasons, IREE_SV("incomplete_resource_counts"), 0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(reasons, IREE_SV("unknown_workgroup_size"), 0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(iree_string_view_find(residency, IREE_SV("\"current_tier\""), 0),
            IREE_STRING_VIEW_NPOS);
  const auto row = LookupArrayElement(
      LookupObject(LookupObject(root, IREE_SV("residency_constraints")),
                   IREE_SV("rows")),
      0);
  ExpectObjectUint64Equals(row, IREE_SV("pool_units"), 512);
  for (const char* field :
       {"\"units\"", "\"rounded_units\"", "\"independent_tier\"",
        "\"limiting\"", "\"reduction_units_to_next_better_tier\""}) {
    EXPECT_EQ(iree_string_view_find(row, iree_make_cstring_view(field), 0),
              IREE_STRING_VIEW_NPOS);
  }
  iree_string_builder_deinitialize(&builder);
  loom_target_compile_report_deinitialize(&report);
}

TEST(CompileReportFormatTest, KeepsResidencyTransitionsOnTheirOwnEntries) {
  struct EntryCase {
    // Emitted entry identity.
    const char* function_name;
    // Final vector-register footprint per subgroup.
    uint32_t vector_register_count;
    // Current modeled residency tier.
    uint32_t tier;
    // Resource responsible for the current tier.
    const char* limiting_resource;
    // Final footprint of the limiting resource.
    uint64_t units;
    // Reduction required to reach the next tier.
    uint64_t reduction_units;
  };
  const EntryCase cases[] = {
      {"register_limited", 88, 8, "amdgpu.vgpr", 88, 4},
      {"lds_limited", 136, 4, "amdgpu.lds", 14848, 512},
  };
  // Exercise complete reports and either ordering of a missing entry result.
  for (int missing_entry = -1; missing_entry < 2; ++missing_entry) {
    SCOPED_TRACE(missing_entry);
    loom_target_compile_report_t report = {};
    loom_target_compile_report_initialize(&report, iree_allocator_system());
    for (int entry_index = 0; entry_index < 2; ++entry_index) {
      const auto& test_case = cases[entry_index];
      loom_target_compile_report_t entry = {};
      loom_target_compile_report_initialize(&entry, iree_allocator_system());
      entry.function_name = iree_make_cstring_view(test_case.function_name);
      if (entry_index != missing_entry) {
        loom_target_compile_report_target_resources_t resources = {};
        resources.scalar_register_class = IREE_SVL("amdgpu.sgpr");
        resources.scalar_register_count = 36;
        resources.vector_register_class = IREE_SVL("amdgpu.vgpr");
        resources.vector_register_count = test_case.vector_register_count;
        resources.subgroup_size = 64;
        resources.max_subgroups_per_simd = 16;
        resources.resident_subgroups_per_simd = test_case.tier;
        resources.occupancy_percent = test_case.tier * 100 / 16;
        resources.limiting_resource =
            iree_make_cstring_view(test_case.limiting_resource);
        auto& summary = resources.residency_summary;
        summary.flags =
            LOOM_TARGET_RESIDENCY_SUMMARY_FLAG_VALID |
            LOOM_TARGET_RESIDENCY_SUMMARY_FLAG_HAS_NEXT_BETTER_TIER |
            LOOM_TARGET_RESIDENCY_SUMMARY_FLAG_HAS_UNIQUE_LIMITING_RESOURCE;
        summary.best_tier = 16;
        summary.tier = resources.resident_subgroups_per_simd;
        summary.next_better_tier = summary.tier + 1;
        summary.limiting_resource_count = 1;
        summary.limiting_resource = resources.limiting_resource;
        summary.limiting_resource_units = test_case.units;
        summary.limiting_resource_reduction_units_to_next_better_tier =
            test_case.reduction_units;
        loom_target_compile_report_record_target_resources(&entry, &resources);
      }
      IREE_ASSERT_OK(
          loom_target_compile_report_record_entry_report(&report, &entry));
      if (entry_index == 0) {
        EXPECT_EQ(loom_target_residency_summary_is_valid(
                      &report.target_resources.residency_summary),
                  missing_entry != 0);
      }
      loom_target_compile_report_deinitialize(&entry);
    }
    EXPECT_FALSE(loom_target_residency_summary_is_valid(
        &report.target_resources.residency_summary));

    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    loom_output_stream_t stream;
    loom_output_stream_for_builder(&builder, &stream);
    const loom_target_compile_report_format_options_t options = {
        /*.mode=*/LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_SUMMARY,
    };
    IREE_ASSERT_OK(
        loom_target_compile_report_format_json(&report, &options, &stream));
    const auto root = ParseJsonDocument(iree_string_builder_view(&builder));
    const auto aggregate_resources =
        LookupObject(root, IREE_SV("target_resources"));
    EXPECT_EQ(
        iree_string_view_find(aggregate_resources, IREE_SV("\"residency\""), 0),
        IREE_STRING_VIEW_NPOS);
    const auto rows =
        LookupObject(LookupObject(root, IREE_SV("entries")), IREE_SV("rows"));
    for (int entry_index = 0; entry_index < 2; ++entry_index) {
      if (entry_index == missing_entry) {
        continue;
      }
      const auto entry = LookupArrayElement(rows, entry_index);
      const auto residency =
          LookupObject(LookupObject(entry, IREE_SV("target_resources")),
                       IREE_SV("residency"));
      ExpectObjectUint64Equals(residency, IREE_SV("current_tier"),
                               cases[entry_index].tier);
      ExpectObjectUint64Equals(residency, IREE_SV("next_better_tier"),
                               cases[entry_index].tier + 1);
    }
    iree_string_builder_deinitialize(&builder);
    loom_target_compile_report_deinitialize(&report);
  }
}

TEST(CompileReportFormatTest, FormatsEntryReportsAndTargetCapabilities) {
  loom_target_compile_report_t report = {};
  loom_target_compile_report_initialize(&report, iree_allocator_system());
  report.requested_detail_flags =
      LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_CAPABILITY_ROWS;

  loom_target_compile_report_t entry = {};
  loom_target_compile_report_initialize(&entry, iree_allocator_system());
  entry.requested_detail_flags = report.requested_detail_flags;
  entry.function_name = IREE_SVL("branchy_export");
  entry.lowered_symbol = IREE_SVL("branchy");
  entry.target_bundle_name = IREE_SVL("test_target");
  entry.target_export_name = IREE_SVL("test_export");
  entry.target_export_symbol = IREE_SVL("branchy_export");
  entry.target_config_name = IREE_SVL("test_o0");
  loom_target_compile_report_record_schedule(
      &entry, /*node_count=*/5, /*scheduled_node_count=*/5,
      /*dependency_count=*/4, /*scope_count=*/2, /*resource_use_count=*/2,
      /*hazard_gap_count=*/1, /*model_summary_count=*/1,
      /*pressure_summary_count=*/2, /*peak_live_units=*/96);
  loom_target_compile_report_record_allocation_materialization(
      &entry, /*spill_storage_count=*/4, /*spill_storage_bytes=*/40,
      /*spill_store_count=*/5, /*spill_store_bytes=*/50, /*reload_count=*/6,
      /*reload_bytes=*/60);
  loom_target_compile_report_record_emission(&entry, /*instruction_count=*/8,
                                             /*code_byte_count=*/64,
                                             /*code_storage_byte_count=*/80);
  const loom_target_compile_report_emission_breakdown_t emission_breakdown = {
      /*.body_instruction_count=*/6,
      /*.entry_instruction_count=*/2,
      /*.coissued_instruction_count=*/1,
      /*.coissued_component_count=*/2,
  };
  loom_target_compile_report_record_emission_breakdown(&entry,
                                                       &emission_breakdown);

  loom_target_compile_report_target_capability_row_t capability = {};
  capability.function_name = entry.function_name;
  capability.target_family_name = IREE_SVL("amdgpu");
  capability.namespace_name = IREE_SVL("amdgpu");
  capability.key = IREE_SVL("matrix_feature_profile");
  capability.value_kind = LOOM_TARGET_COMPILE_REPORT_CAPABILITY_VALUE_STRING;
  capability.value_string = IREE_SVL("wmma-gfx11");
  IREE_ASSERT_OK(loom_target_compile_report_record_target_capability_row(
      &entry, &capability));
  capability.namespace_name = IREE_SVL("target");
  capability.key = IREE_SVL("subgroup_size");
  capability.value_kind = LOOM_TARGET_COMPILE_REPORT_CAPABILITY_VALUE_U64;
  capability.value_u64 = 32;
  capability.value_string = iree_string_view_empty();
  IREE_ASSERT_OK(loom_target_compile_report_record_target_capability_row(
      &entry, &capability));
  capability.namespace_name = IREE_SVL("amdgpu");
  capability.key = IREE_SVL("wavefront_64");
  capability.value_kind = LOOM_TARGET_COMPILE_REPORT_CAPABILITY_VALUE_BOOL;
  capability.value_u64 = 1;
  IREE_ASSERT_OK(loom_target_compile_report_record_target_capability_row(
      &entry, &capability));
  IREE_ASSERT_OK(
      loom_target_compile_report_record_entry_report(&report, &entry));
  loom_target_compile_report_deinitialize(&entry);

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
  const iree_string_view_t entries = LookupObject(root, IREE_SV("entries"));
  ExpectObjectUint64Equals(entries, IREE_SV("count"), 1);
  const iree_string_view_t entry_row =
      LookupArrayElement(LookupObject(entries, IREE_SV("rows")), /*index=*/0);
  ExpectObjectValueEquals(entry_row, IREE_SV("function"),
                          IREE_SV("branchy_export"));
  ExpectObjectUint64Equals(entry_row, IREE_SV("schedule_scope_count"), 2);
  ExpectObjectUint64Equals(entry_row, IREE_SV("schedule_resource_use_count"),
                           2);
  ExpectObjectUint64Equals(
      entry_row, IREE_SV("allocation_materialized_spill_storage_count"), 4);
  ExpectObjectUint64Equals(entry_row, IREE_SV("instruction_count"), 8);
  ExpectObjectUint64Equals(entry_row, IREE_SV("body_instruction_count"), 6);
  ExpectObjectUint64Equals(entry_row, IREE_SV("entry_instruction_count"), 2);
  ExpectObjectUint64Equals(entry_row, IREE_SV("coissued_instruction_count"), 1);
  ExpectObjectUint64Equals(entry_row, IREE_SV("coissued_component_count"), 2);

  const iree_string_view_t capability_rows =
      LookupObject(root, IREE_SV("target_capability_rows"));
  ExpectObjectUint64Equals(capability_rows, IREE_SV("count"), 3);
  const iree_string_view_t rows =
      LookupObject(capability_rows, IREE_SV("rows"));
  const iree_string_view_t string_row = LookupArrayElement(rows, 0);
  ExpectObjectValueEquals(string_row, IREE_SV("value_string"),
                          IREE_SV("wmma-gfx11"));
  const iree_string_view_t integer_row = LookupArrayElement(rows, 1);
  ExpectObjectUint64Equals(integer_row, IREE_SV("value_u64"), 32);
  const iree_string_view_t bool_row = LookupArrayElement(rows, 2);
  ExpectObjectValueEquals(bool_row, IREE_SV("value_bool"), IREE_SV("true"));

  iree_string_builder_deinitialize(&builder);
  loom_target_compile_report_deinitialize(&report);
}

TEST(CompileReportFormatTest, FormatsTargetInsertedPacketEconomics) {
  loom_target_compile_report_t report = {};
  loom_target_compile_report_initialize(&report, iree_allocator_system());
  report.requested_detail_flags =
      LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_INSERTION_ROWS;

  loom_target_compile_report_t entry = {};
  loom_target_compile_report_initialize(&entry, iree_allocator_system());
  entry.requested_detail_flags = report.requested_detail_flags;
  entry.function_name = IREE_SVL("extended_vgpr_loop");
  loom_target_compile_report_target_insertion_row_t insertion = {
      /*.flags=*/
      LOOM_TARGET_COMPILE_REPORT_TARGET_INSERTION_FLAG_DYNAMIC_PACKET_COUNT,
      /*.function_name=*/entry.function_name,
      /*.insertion_kind=*/
      LOOM_TARGET_COMPILE_REPORT_TARGET_INSERTION_STATE,
      /*.packet_key=*/IREE_SVL("amdgpu.s_set_vgpr_msb"),
      /*.block_name=*/IREE_SVL("^loop_body"),
      /*.block_index=*/1,
      /*.node_index=*/7,
      /*.scheduled_ordinal=*/3,
      /*.boundary_operation_name=*/IREE_SVL("low.op"),
      /*.boundary_descriptor_key=*/IREE_SVL("amdgpu.v_wmma_f32_16x16x32_bf16"),
      /*.static_packet_count=*/1,
      /*.dynamic_packet_count=*/4,
  };
  IREE_ASSERT_OK(loom_target_compile_report_record_target_insertion_row(
      &entry, &insertion));
  IREE_ASSERT_OK(
      loom_target_compile_report_record_entry_report(&report, &entry));
  loom_target_compile_report_deinitialize(&entry);

  loom_target_compile_report_initialize(&entry, iree_allocator_system());
  entry.requested_detail_flags = report.requested_detail_flags;
  entry.function_name = IREE_SVL("uncountable_loop");
  insertion.flags = LOOM_TARGET_COMPILE_REPORT_TARGET_INSERTION_FLAG_NONE;
  insertion.function_name = entry.function_name;
  insertion.block_name = IREE_SVL("^uncountable");
  insertion.block_index = 2;
  insertion.node_index = 11;
  insertion.scheduled_ordinal = 5;
  insertion.boundary_operation_name = IREE_SVL("low.return");
  insertion.boundary_descriptor_key = iree_string_view_empty();
  insertion.dynamic_packet_count = 0;
  IREE_ASSERT_OK(loom_target_compile_report_record_target_insertion_row(
      &entry, &insertion));
  IREE_ASSERT_OK(
      loom_target_compile_report_record_entry_report(&report, &entry));
  loom_target_compile_report_deinitialize(&entry);

  EXPECT_EQ(report.target_insertion_summary.static_packet_count, 2u);
  EXPECT_EQ(report.target_insertion_summary.exact_dynamic_packet_count, 1u);
  EXPECT_EQ(report.target_insertion_summary.unknown_dynamic_packet_count, 1u);
  EXPECT_EQ(report.target_insertion_summary.dynamic_packet_count, 4u);
  EXPECT_EQ(report.target_insertion_rows.count, 2u);

  const loom_target_compile_report_format_options_t options = {
      /*.mode=*/LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS,
  };
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(
      loom_target_compile_report_format_text(&report, &options, &builder));
  const iree_string_view_t text = iree_string_builder_view(&builder);
  EXPECT_NE(iree_string_view_find(
                text,
                IREE_SV("target_insertions static_packets=2 "
                        "exact_dynamic_packets=1 unknown_dynamic_packets=1 "
                        "dynamic_packets=unavailable rows=2"),
                0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(text,
                                  IREE_SV("target_insertion[0] function="
                                          "extended_vgpr_loop kind=state "
                                          "packet=amdgpu.s_set_vgpr_msb"),
                                  0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(text,
                                  IREE_SV("target_insertion[1] function="
                                          "uncountable_loop kind=state"),
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
  const iree_string_view_t target_insertions =
      LookupObject(root, IREE_SV("target_insertions"));
  ExpectObjectUint64Equals(target_insertions, IREE_SV("static_packet_count"),
                           2);
  ExpectObjectUint64Equals(target_insertions,
                           IREE_SV("exact_dynamic_packet_count"), 1);
  ExpectObjectUint64Equals(target_insertions,
                           IREE_SV("unknown_dynamic_packet_count"), 1);
  ExpectObjectValueEquals(target_insertions, IREE_SV("dynamic_packet_count"),
                          IREE_SV("null"));
  ExpectObjectUint64Equals(target_insertions, IREE_SV("row_count"), 2);
  const iree_string_view_t rows =
      LookupObject(target_insertions, IREE_SV("rows"));
  const iree_string_view_t exact_row = LookupArrayElement(rows, 0);
  ExpectObjectValueEquals(exact_row, IREE_SV("kind"), IREE_SV("state"));
  ExpectObjectValueEquals(exact_row, IREE_SV("packet_key"),
                          IREE_SV("amdgpu.s_set_vgpr_msb"));
  ExpectObjectUint64Equals(exact_row, IREE_SV("dynamic_packet_count"), 4);
  const iree_string_view_t unknown_row = LookupArrayElement(rows, 1);
  ExpectObjectValueEquals(unknown_row, IREE_SV("dynamic_packet_count"),
                          IREE_SV("null"));

  const iree_string_view_t entries = LookupObject(root, IREE_SV("entries"));
  const iree_string_view_t entry_row =
      LookupArrayElement(LookupObject(entries, IREE_SV("rows")), 0);
  const iree_string_view_t entry_target_insertions =
      LookupObject(entry_row, IREE_SV("target_insertions"));
  ExpectObjectUint64Equals(entry_target_insertions,
                           IREE_SV("static_packet_count"), 1);
  ExpectObjectUint64Equals(entry_target_insertions, IREE_SV("row_count"), 1);

  iree_string_builder_deinitialize(&builder);
  loom_target_compile_report_deinitialize(&report);
}

TEST(CompileReportFormatTest, SummarizesTargetInsertionsWithoutDetailRows) {
  loom_target_compile_report_t report = {};
  loom_target_compile_report_initialize(&report, iree_allocator_system());
  const loom_target_compile_report_target_insertion_row_t insertion = {
      /*.flags=*/
      LOOM_TARGET_COMPILE_REPORT_TARGET_INSERTION_FLAG_DYNAMIC_PACKET_COUNT,
      /*.function_name=*/IREE_SVL("summary_only"),
      /*.insertion_kind=*/
      LOOM_TARGET_COMPILE_REPORT_TARGET_INSERTION_STATE,
      /*.packet_key=*/IREE_SVL("amdgpu.s_set_vgpr_msb"),
      /*.block_name=*/IREE_SVL("^entry"),
      /*.block_index=*/0,
      /*.node_index=*/2,
      /*.scheduled_ordinal=*/1,
      /*.boundary_operation_name=*/IREE_SVL("low.op"),
      /*.boundary_descriptor_key=*/IREE_SVL("amdgpu.v_wmma"),
      /*.static_packet_count=*/1,
      /*.dynamic_packet_count=*/8,
  };
  IREE_ASSERT_OK(loom_target_compile_report_record_target_insertion_row(
      &report, &insertion));

  EXPECT_TRUE(iree_any_bit_set(
      report.detail_flags,
      LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_INSERTION_ROWS));
  EXPECT_EQ(report.target_insertion_summary.static_packet_count, 1u);
  EXPECT_EQ(report.target_insertion_summary.exact_dynamic_packet_count, 1u);
  EXPECT_EQ(report.target_insertion_summary.unknown_dynamic_packet_count, 0u);
  EXPECT_EQ(report.target_insertion_summary.dynamic_packet_count, 8u);
  EXPECT_EQ(report.target_insertion_rows.count, 0u);

  const loom_target_compile_report_format_options_t options = {
      /*.mode=*/LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_SUMMARY,
  };
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(
      loom_target_compile_report_format_text(&report, &options, &builder));
  EXPECT_NE(iree_string_view_find(
                iree_string_builder_view(&builder),
                IREE_SV("target_insertions static_packets=1 "
                        "exact_dynamic_packets=1 unknown_dynamic_packets=0 "
                        "dynamic_packets=8 rows=0"),
                0),
            IREE_STRING_VIEW_NPOS);
  iree_string_builder_deinitialize(&builder);

  loom_target_compile_report_deinitialize(&report);
}

}  // namespace
}  // namespace loom
