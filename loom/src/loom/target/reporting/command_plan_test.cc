// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>

#include "iree/base/internal/json.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/reporting/format.h"

namespace loom {
namespace {

TEST(CommandReportTest, CopiesAndMergesRowsWithoutRetainingTemporaryArrays) {
  for (bool details : {false, true}) {
    loom_target_compile_report_t source;
    loom_target_compile_report_t merged;
    loom_target_compile_report_initialize(iree_allocator_system(), &source);
    loom_target_compile_report_initialize(iree_allocator_system(), &merged);
    source.requested_detail_flags = merged.requested_detail_flags =
        details ? LOOM_TARGET_COMPILE_REPORT_DETAIL_COMMAND_PLAN_ROWS : 0;
    loom_target_compile_report_command_t commands[] = {
        {.kind = LOOM_TARGET_COMPILE_REPORT_COMMAND_FILL},
        {
            .kind = LOOM_TARGET_COMPILE_REPORT_COMMAND_COPY,
            .flags = LOOM_TARGET_COMPILE_REPORT_COMMAND_BARRIER_BEFORE,
        },
        {.kind = LOOM_TARGET_COMPILE_REPORT_COMMAND_BARRIER},
    };
    commands[0].payload.fill = {
        .target = {.byte_length = 0},
        .pattern = 0xa5,
        .pattern_byte_count = 1,
    };
    commands[1].payload.copy.source = {
        .byte_offset = UINT64_C(4294967296),
        .byte_length = UINT64_MAX,
    };
    commands[1].payload.copy.target = {
        .role = LOOM_TARGET_COMPILE_REPORT_COMMAND_BUFFER_REBINDABLE,
        .root_index = 2,
        .byte_length = UINT64_MAX,
    };
    const loom_target_compile_report_command_binding_t binding = {
        .role = LOOM_TARGET_COMPILE_REPORT_COMMAND_BUFFER_REBINDABLE,
        .root_index = 2,
    };
    const loom_target_compile_report_command_storage_t storage = {
        .required_byte_length = UINT64_C(4294967312),
        .minimum_alignment = 16,
    };
    const loom_target_compile_report_command_parameter_t parameter = {
        .key = IREE_SVL("weights"),
        .byte_offset = UINT64_C(4294967296),
        .byte_length = 16,
        .minimum_alignment = 16,
    };
    const loom_target_compile_report_command_plan_t plan = {
        .root_name = IREE_SVL("transfer"),
        .command_kind_counts = {1, 1, 0, 0, 0, 1, 0},
        .barrier_before_count = 1,
        .fixed_buffer_count = 1,
        .rebindable_binding_count = 3,
        .transient = {.root_index = UINT32_MAX},
        .launch_counts = {.root_index = UINT32_MAX},
        .command_count = 3,
        .commands = commands,
        .binding_count = 1,
        .bindings = &binding,
        .parameter_root_count = 1,
        .parameter_roots = &storage,
        .parameter_count = 1,
        .parameters = &parameter,
    };
    // Cross row-list blocks, preserving duplicate roots and their caller order.
    for (int i = 0; i < 64; ++i) {
      IREE_ASSERT_OK(
          loom_target_compile_report_record_command_plan(&source, &plan));
    }
    IREE_ASSERT_OK(loom_target_compile_report_merge_evidence(&merged, &source));
    loom_target_compile_report_deinitialize(&source);
    commands[0].payload.fill.pattern = 0;
    ASSERT_EQ(merged.command_plans.count, 64u);
    ASSERT_NE(merged.command_plans.head, merged.command_plans.tail);
    for (const auto* block = merged.command_plans.head; block;
         block = block->next) {
      const auto* rows =
          static_cast<const loom_target_compile_report_command_plan_t*>(
              loom_target_compile_report_vec_const_rows(block));
      for (size_t i = 0; i < block->count; ++i) {
        EXPECT_EQ(rows[i].command_count, 3u);
        EXPECT_EQ(rows[i].parameter_count, 1u);
        if (details) {
          ASSERT_NE(rows[i].commands, nullptr);
          EXPECT_NE(rows[i].commands, commands);
          EXPECT_EQ(rows[i].commands[0].payload.fill.pattern, 0xa5u);
          EXPECT_EQ(rows[i].commands[1].payload.copy.source.byte_offset,
                    UINT64_C(4294967296));
          EXPECT_EQ(rows[i].bindings[0].root_index, 2u);
          EXPECT_EQ(rows[i].parameter_roots[0].required_byte_length,
                    UINT64_C(4294967312));
          EXPECT_EQ(rows[i].parameters[0].byte_offset, UINT64_C(4294967296));
        } else {
          EXPECT_EQ(rows[i].commands, nullptr);
          EXPECT_EQ(rows[i].bindings, nullptr);
          EXPECT_EQ(rows[i].parameter_roots, nullptr);
          EXPECT_EQ(rows[i].parameters, nullptr);
        }
      }
    }
    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    loom_output_stream_t stream;
    loom_output_stream_for_builder(&builder, &stream);
    const loom_target_compile_report_format_options_t options = {
        .mode = details ? LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS
                        : LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_SUMMARY,
    };
    IREE_ASSERT_OK(
        loom_target_compile_report_format_json(&merged, &options, &stream));
    iree_string_view_t roots, first, ordered_commands;
    IREE_ASSERT_OK(
        iree_json_lookup_object_value(iree_string_builder_view(&builder),
                                      IREE_SV("command_programs"), &roots));
    IREE_ASSERT_OK(iree_json_array_get(roots, 0, &first));
    if (details) {
      IREE_ASSERT_OK(iree_json_lookup_object_value(first, IREE_SV("commands"),
                                                   &ordered_commands));
      const std::string expected =
          R"([{"kind":"fill","barrier_before":false,"target":{"role":"fixed","root_index":0,"byte_offset":0,"byte_length":0},"pattern":165,"pattern_byte_count":1},)"
          R"({"kind":"copy","barrier_before":true,"source":{"role":"fixed","root_index":0,"byte_offset":4294967296,"byte_length":null},"target":{"role":"rebindable","root_index":2,"byte_offset":0,"byte_length":null}},)"
          R"({"kind":"barrier","barrier_before":false}])";
      EXPECT_EQ(std::string(ordered_commands.data, ordered_commands.size),
                expected);
    } else {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                            iree_json_lookup_object_value(
                                first, IREE_SV("commands"), &ordered_commands));
    }
    iree_string_builder_deinitialize(&builder);
    loom_target_compile_report_deinitialize(&merged);
  }
}

struct LimitedAllocator {
  // Successful allocations still allowed before failure injection.
  size_t remaining;
  // Owned allocations that teardown must release.
  size_t live = 0;

  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* params, void** pointer) {
    auto& state = *static_cast<LimitedAllocator*>(self);
    if (command != IREE_ALLOCATOR_COMMAND_FREE && state.remaining == 0) {
      return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
    }
    auto backing = iree_allocator_system();
    IREE_RETURN_IF_ERROR(backing.ctl(backing.self, command, params, pointer));
    if (command == IREE_ALLOCATOR_COMMAND_FREE) {
      --state.live;
    } else {
      --state.remaining;
      ++state.live;
    }
    return iree_ok_status();
  }
};

TEST(CommandReportTest, FailedCopiesLeaveNoPartialRootAndCanBeRetried) {
  const loom_target_compile_report_command_t command = {
      .kind = LOOM_TARGET_COMPILE_REPORT_COMMAND_BARRIER,
  };
  const loom_target_compile_report_command_binding_t binding = {};
  const loom_target_compile_report_command_storage_t storage = {};
  const loom_target_compile_report_command_parameter_t parameter = {};
  const loom_target_compile_report_command_plan_t plan = {
      .command_count = 1,
      .commands = &command,
      .binding_count = 1,
      .bindings = &binding,
      .parameter_root_count = 1,
      .parameter_roots = &storage,
      .parameter_count = 1,
      .parameters = &parameter,
  };
  for (size_t allowance = 0; allowance < 5; ++allowance) {
    LimitedAllocator allocator{allowance};
    loom_target_compile_report_t report;
    loom_target_compile_report_initialize(
        {&allocator, LimitedAllocator::Control}, &report);
    report.requested_detail_flags =
        LOOM_TARGET_COMPILE_REPORT_DETAIL_COMMAND_PLAN_ROWS;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        loom_target_compile_report_record_command_plan(&report, &plan));
    EXPECT_EQ(report.command_plans.count, 0u);
    EXPECT_EQ(allocator.live, 0u);
    allocator.remaining = 5;
    IREE_ASSERT_OK(
        loom_target_compile_report_record_command_plan(&report, &plan));
    EXPECT_EQ(report.command_plans.count, 1u);
    loom_target_compile_report_deinitialize(&report);
    EXPECT_EQ(allocator.live, 0u);
  }
  loom_target_compile_report_t disabled;
  loom_target_compile_report_initialize(iree_allocator_null(), &disabled);
  IREE_ASSERT_OK(
      loom_target_compile_report_record_command_plan(&disabled, &plan));
  EXPECT_EQ(disabled.command_plans.count, 0u);
  loom_target_compile_report_deinitialize(&disabled);
}

}  // namespace
}  // namespace loom
