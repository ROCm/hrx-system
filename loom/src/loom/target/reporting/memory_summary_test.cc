// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/types.h"
#include "loom/target/math_policy.h"
#include "loom/target/reporting/format.h"

namespace loom {
namespace {

constexpr uint32_t kTestSourceRejectionDetail = 4;

static loom_target_compile_report_memory_interval_t MakeExactSourceInterval(
    int64_t begin_bytes, int64_t end_bytes) {
  return {
      /*.flags=*/
      LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_BEGIN_RANGE |
          LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_END_RANGE |
          LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_EXACT_LENGTH,
      /*.begin_min_bytes=*/begin_bytes,
      /*.begin_max_bytes=*/begin_bytes,
      /*.end_min_bytes=*/end_bytes,
      /*.end_max_bytes=*/end_bytes,
      /*.exact_length_bytes=*/static_cast<uint64_t>(end_bytes - begin_bytes),
  };
}

static loom_target_compile_report_memory_interval_t MakeExactSymbolicInterval(
    int64_t begin_min_bytes, int64_t begin_max_bytes, int64_t end_min_bytes,
    int64_t end_max_bytes, uint64_t exact_length_bytes, uint32_t begin_expr_id,
    uint32_t end_expr_id) {
  return {
      /*.flags=*/
      LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_BEGIN_RANGE |
          LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_END_RANGE |
          LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_EXACT_LENGTH |
          LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_BEGIN_EXPR |
          LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_END_EXPR,
      /*.begin_min_bytes=*/begin_min_bytes,
      /*.begin_max_bytes=*/begin_max_bytes,
      /*.end_min_bytes=*/end_min_bytes,
      /*.end_max_bytes=*/end_max_bytes,
      /*.exact_length_bytes=*/exact_length_bytes,
      /*.begin_expr_id=*/begin_expr_id,
      /*.end_expr_id=*/end_expr_id,
  };
}

static loom_target_compile_report_source_low_memory_row_t MakeMemoryRow(
    iree_string_view_t source_op_name, uint32_t source_op_kind,
    iree_string_view_t operation_kind, iree_string_view_t packet_key,
    int64_t static_offset_bytes, uint32_t vector_lane_count,
    uint32_t issued_read_byte_count, uint32_t issued_write_byte_count,
    uint32_t dynamic_stride_bytes, uint32_t vector_lane_stride_bytes,
    loom_target_compile_report_memory_interval_t source_interval) {
  loom_target_compile_report_source_low_memory_row_t row = {};
  row.function_name = IREE_SVL("kernel");
  row.source_op_name = source_op_name;
  row.source_op_kind = source_op_kind;
  row.source_root_name = IREE_SVL("scratch");
  row.source_root_argument_index = 1;
  row.memory_space = IREE_SVL("workgroup");
  row.operation_kind = operation_kind;
  row.packet_key = packet_key;
  row.address_form = IREE_SVL("global_saddr");
  row.dynamic_term_kind = IREE_SVL("vaddr");
  row.static_offset_bytes = static_offset_bytes;
  row.element_byte_count = 4;
  row.vector_lane_count = vector_lane_count;
  row.issued_read_byte_count = issued_read_byte_count;
  row.issued_write_byte_count = issued_write_byte_count;
  row.dynamic_stride_bytes = dynamic_stride_bytes;
  row.vector_lane_stride_bytes = vector_lane_stride_bytes;
  row.source_interval = source_interval;
  row.execution_count_plus_one = 2;
  return row;
}

static const loom_target_compile_report_source_low_memory_summary_t*
GetOnlySourceLowMemoryRootSummary(const loom_target_compile_report_t* report) {
  if (report->source_low_memory_root_summaries.count != 1u ||
      report->source_low_memory_root_summaries.head == nullptr) {
    return nullptr;
  }
  const loom_target_compile_report_source_low_memory_root_summary_t*
      root_summaries = static_cast<
          const loom_target_compile_report_source_low_memory_root_summary_t*>(
          loom_target_compile_report_vec_const_rows(
              report->source_low_memory_root_summaries.head));
  return &root_summaries[0].summary;
}

TEST(CompileReportFormatTest, MergesSourceLowMemorySummariesFromEntries) {
  loom_target_compile_report_t report;
  loom_target_compile_report_initialize(&report, iree_allocator_system());

  loom_target_compile_report_source_low_memory_row_t rows[] = {
      MakeMemoryRow(IREE_SVL("vector.load"), /*source_op_kind=*/43,
                    IREE_SVL("load"), IREE_SVL("test.load.v2"),
                    /*static_offset_bytes=*/0,
                    /*vector_lane_count=*/2,
                    /*issued_read_byte_count=*/8,
                    /*issued_write_byte_count=*/0,
                    /*dynamic_stride_bytes=*/8,
                    /*vector_lane_stride_bytes=*/4,
                    MakeExactSourceInterval(/*begin_bytes=*/0,
                                            /*end_bytes=*/8)),
      MakeMemoryRow(IREE_SVL("view.store"), /*source_op_kind=*/44,
                    IREE_SVL("store"), IREE_SVL("test.store.v1"),
                    /*static_offset_bytes=*/8,
                    /*vector_lane_count=*/1,
                    /*issued_read_byte_count=*/0,
                    /*issued_write_byte_count=*/4,
                    /*dynamic_stride_bytes=*/4,
                    /*vector_lane_stride_bytes=*/4,
                    MakeExactSourceInterval(/*begin_bytes=*/8,
                                            /*end_bytes=*/12)),
  };

  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(rows); ++i) {
    loom_target_compile_report_t entry_report;
    loom_target_compile_report_initialize(&entry_report,
                                          iree_allocator_system());
    IREE_ASSERT_OK(loom_target_compile_report_record_source_low_memory_row(
        &entry_report, &rows[i]));
    IREE_ASSERT_OK(
        loom_target_compile_report_record_entry_report(&report, &entry_report));
    loom_target_compile_report_deinitialize(&entry_report);
  }

  const loom_target_compile_report_source_low_memory_summary_t* summary =
      &report.source_low_memory_summary;
  EXPECT_EQ(summary->packet_count, 2u);
  EXPECT_EQ(summary->load_packet_count, 1u);
  EXPECT_EQ(summary->store_packet_count, 1u);
  EXPECT_EQ(summary->scalar_packet_count, 1u);
  EXPECT_EQ(summary->vector_packet_count, 1u);
  EXPECT_EQ(summary->source_lane_count, 3u);
  EXPECT_EQ(summary->source_byte_count, 12u);
  EXPECT_EQ(summary->read_byte_count, 8u);
  EXPECT_EQ(summary->write_byte_count, 4u);
  EXPECT_EQ(summary->issued_read_byte_count, 8u);
  EXPECT_EQ(summary->issued_write_byte_count, 4u);
  EXPECT_EQ(summary->issued_read_unknown_width_count, 0u);
  EXPECT_EQ(summary->issued_write_unknown_width_count, 0u);
  EXPECT_EQ(summary->contiguous_vector_packet_count, 1u);
  EXPECT_EQ(summary->strided_vector_packet_count, 0u);
  EXPECT_EQ(summary->unknown_stride_vector_packet_count, 0u);
  EXPECT_EQ(summary->interval_envelope.packet_count, 2u);
  EXPECT_EQ(summary->interval_envelope.envelope_begin_min_bytes, 0);
  EXPECT_EQ(summary->interval_envelope.envelope_end_max_bytes, 12);
  EXPECT_EQ(summary->interval_envelope.envelope_byte_count, 12u);
  EXPECT_EQ(summary->read_interval_envelope.packet_count, 1u);
  EXPECT_EQ(summary->read_interval_envelope.envelope_byte_count, 8u);
  EXPECT_EQ(summary->write_interval_envelope.packet_count, 1u);
  EXPECT_EQ(summary->write_interval_envelope.envelope_byte_count, 4u);

  ASSERT_EQ(report.source_low_memory_root_summaries.count, 1u);
  ASSERT_NE(report.source_low_memory_root_summaries.head, nullptr);
  const loom_target_compile_report_source_low_memory_root_summary_t*
      root_summaries = static_cast<
          const loom_target_compile_report_source_low_memory_root_summary_t*>(
          loom_target_compile_report_vec_const_rows(
              report.source_low_memory_root_summaries.head));
  const loom_target_compile_report_source_low_memory_root_summary_t*
      root_summary = &root_summaries[0];
  const loom_target_compile_report_source_low_memory_summary_t*
      root_memory_summary = &root_summary->summary;
  EXPECT_EQ(root_memory_summary->packet_count, 2u);
  EXPECT_EQ(root_memory_summary->load_packet_count, 1u);
  EXPECT_EQ(root_memory_summary->store_packet_count, 1u);
  EXPECT_EQ(root_memory_summary->source_byte_count, 12u);
  EXPECT_EQ(root_memory_summary->read_byte_count, 8u);
  EXPECT_EQ(root_memory_summary->write_byte_count, 4u);
  EXPECT_EQ(root_memory_summary->issued_read_byte_count, 8u);
  EXPECT_EQ(root_memory_summary->issued_write_byte_count, 4u);
  EXPECT_EQ(root_memory_summary->interval_envelope.packet_count, 2u);
  EXPECT_EQ(root_memory_summary->interval_envelope.envelope_byte_count, 12u);
  EXPECT_EQ(root_memory_summary->interval_envelope.exact_static_packet_count,
            0u);
  EXPECT_EQ(root_memory_summary->interval_envelope.unique_byte_count, 0u);
  EXPECT_EQ(root_memory_summary->read_interval_envelope.packet_count, 1u);
  EXPECT_EQ(root_memory_summary->read_interval_envelope.envelope_byte_count,
            8u);
  EXPECT_EQ(
      root_memory_summary->read_interval_envelope.exact_static_packet_count,
      0u);
  EXPECT_EQ(root_memory_summary->read_interval_envelope.unique_byte_count, 0u);
  EXPECT_EQ(root_memory_summary->write_interval_envelope.packet_count, 1u);
  EXPECT_EQ(root_memory_summary->write_interval_envelope.envelope_byte_count,
            4u);
  EXPECT_EQ(
      root_memory_summary->write_interval_envelope.exact_static_packet_count,
      0u);
  EXPECT_EQ(root_memory_summary->write_interval_envelope.unique_byte_count, 0u);

  ASSERT_EQ(report.source_low_memory_argument_summaries.count, 1u);
  ASSERT_NE(report.source_low_memory_argument_summaries.head, nullptr);
  const loom_target_compile_report_source_low_memory_argument_summary_t*
      argument_summary = static_cast<
          const loom_target_compile_report_source_low_memory_argument_summary_t*>(
          loom_target_compile_report_vec_const_rows(
              report.source_low_memory_argument_summaries.head));
  EXPECT_TRUE(iree_string_view_equal(argument_summary->function_name,
                                     IREE_SVL("kernel")));
  EXPECT_TRUE(iree_string_view_equal(argument_summary->source_root_name,
                                     IREE_SVL("scratch")));
  EXPECT_EQ(argument_summary->source_root_argument_index, 1u);
  EXPECT_TRUE(iree_string_view_equal(argument_summary->memory_space,
                                     IREE_SVL("workgroup")));
  EXPECT_EQ(argument_summary->summary.packet_count, 2u);
  EXPECT_EQ(argument_summary->summary.source_byte_count, 12u);

  ASSERT_EQ(report.source_low_memory_argument_packet_summaries.count, 2u);
  ASSERT_NE(report.source_low_memory_argument_packet_summaries.head, nullptr);
  const loom_target_compile_report_source_low_memory_argument_packet_summary_t*
      argument_packets = static_cast<
          const loom_target_compile_report_source_low_memory_argument_packet_summary_t*>(
          loom_target_compile_report_vec_const_rows(
              report.source_low_memory_argument_packet_summaries.head));
  EXPECT_TRUE(iree_string_view_equal(argument_packets[0].packet_key,
                                     IREE_SVL("test.load.v2")));
  EXPECT_TRUE(iree_string_view_equal(argument_packets[1].packet_key,
                                     IREE_SVL("test.store.v1")));

  loom_target_compile_report_deinitialize(&report);
}

TEST(CompileReportFormatTest, SeparatesSourceLowMemoryStrategiesByPacket) {
  loom_target_compile_report_t report;
  loom_target_compile_report_initialize(&report, iree_allocator_system());

  loom_target_compile_report_source_low_memory_row_t rows[] = {
      MakeMemoryRow(IREE_SVL("vector.load"), /*source_op_kind=*/43,
                    IREE_SVL("load"), IREE_SVL("test.load.v2"),
                    /*static_offset_bytes=*/0,
                    /*vector_lane_count=*/2,
                    /*issued_read_byte_count=*/8,
                    /*issued_write_byte_count=*/0,
                    /*dynamic_stride_bytes=*/8,
                    /*vector_lane_stride_bytes=*/4,
                    MakeExactSourceInterval(/*begin_bytes=*/0,
                                            /*end_bytes=*/8)),
      MakeMemoryRow(IREE_SVL("vector.load"), /*source_op_kind=*/43,
                    IREE_SVL("load"), IREE_SVL("test.load.v4"),
                    /*static_offset_bytes=*/16,
                    /*vector_lane_count=*/4,
                    /*issued_read_byte_count=*/16,
                    /*issued_write_byte_count=*/0,
                    /*dynamic_stride_bytes=*/16,
                    /*vector_lane_stride_bytes=*/4,
                    MakeExactSourceInterval(/*begin_bytes=*/16,
                                            /*end_bytes=*/32)),
  };
  rows[0].strategy_key = IREE_SVL("test.wide-load");
  rows[1].strategy_key = IREE_SVL("test.wide-load");

  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(rows); ++i) {
    IREE_ASSERT_OK(loom_target_compile_report_record_source_low_memory_row(
        &report, &rows[i]));
  }

  ASSERT_EQ(report.source_low_memory_strategy_summaries.count, 2u);
  ASSERT_NE(report.source_low_memory_strategy_summaries.head, nullptr);
  const loom_target_compile_report_source_low_memory_strategy_summary_t*
      summaries = static_cast<
          const loom_target_compile_report_source_low_memory_strategy_summary_t*>(
          loom_target_compile_report_vec_const_rows(
              report.source_low_memory_strategy_summaries.head));
  EXPECT_TRUE(iree_string_view_equal(summaries[0].packet_key,
                                     IREE_SVL("test.load.v2")));
  EXPECT_TRUE(iree_string_view_equal(summaries[1].packet_key,
                                     IREE_SVL("test.load.v4")));
  ASSERT_EQ(report.source_low_memory_argument_packet_summaries.count, 2u);
  ASSERT_NE(report.source_low_memory_argument_packet_summaries.head, nullptr);
  const loom_target_compile_report_source_low_memory_argument_packet_summary_t*
      argument_packets = static_cast<
          const loom_target_compile_report_source_low_memory_argument_packet_summary_t*>(
          loom_target_compile_report_vec_const_rows(
              report.source_low_memory_argument_packet_summaries.head));
  EXPECT_TRUE(iree_string_view_equal(argument_packets[0].packet_key,
                                     IREE_SVL("test.load.v2")));
  EXPECT_TRUE(iree_string_view_equal(argument_packets[0].strategy_key,
                                     IREE_SVL("test.wide-load")));
  EXPECT_TRUE(iree_string_view_equal(argument_packets[1].packet_key,
                                     IREE_SVL("test.load.v4")));
  EXPECT_TRUE(iree_string_view_equal(argument_packets[1].strategy_key,
                                     IREE_SVL("test.wide-load")));

  loom_target_compile_report_deinitialize(&report);
}

TEST(CompileReportFormatTest, MergesOverlappingSourceLowMemoryIntervals) {
  loom_target_compile_report_t report;
  loom_target_compile_report_initialize(&report, iree_allocator_system());

  const loom_target_compile_report_source_low_memory_row_t rows[] = {
      MakeMemoryRow(IREE_SVL("vector.load"), /*source_op_kind=*/43,
                    IREE_SVL("load"), IREE_SVL("test.load.v2"),
                    /*static_offset_bytes=*/0,
                    /*vector_lane_count=*/2,
                    /*issued_read_byte_count=*/8,
                    /*issued_write_byte_count=*/0,
                    /*dynamic_stride_bytes=*/0,
                    /*vector_lane_stride_bytes=*/4,
                    MakeExactSourceInterval(/*begin_bytes=*/0,
                                            /*end_bytes=*/8)),
      MakeMemoryRow(IREE_SVL("vector.load"), /*source_op_kind=*/43,
                    IREE_SVL("load"), IREE_SVL("test.load.v2"),
                    /*static_offset_bytes=*/4,
                    /*vector_lane_count=*/2,
                    /*issued_read_byte_count=*/8,
                    /*issued_write_byte_count=*/0,
                    /*dynamic_stride_bytes=*/0,
                    /*vector_lane_stride_bytes=*/4,
                    MakeExactSourceInterval(/*begin_bytes=*/4,
                                            /*end_bytes=*/12)),
      MakeMemoryRow(IREE_SVL("view.store"), /*source_op_kind=*/44,
                    IREE_SVL("store"), IREE_SVL("test.store.v1"),
                    /*static_offset_bytes=*/0,
                    /*vector_lane_count=*/2,
                    /*issued_read_byte_count=*/0,
                    /*issued_write_byte_count=*/8,
                    /*dynamic_stride_bytes=*/0,
                    /*vector_lane_stride_bytes=*/4,
                    MakeExactSourceInterval(/*begin_bytes=*/0,
                                            /*end_bytes=*/8)),
  };

  for (const auto& row : rows) {
    IREE_ASSERT_OK(
        loom_target_compile_report_record_source_low_memory_row(&report, &row));
  }

  const loom_target_compile_report_source_low_memory_summary_t* summary =
      &report.source_low_memory_summary;
  EXPECT_EQ(summary->interval_envelope.packet_count, 3u);
  EXPECT_EQ(summary->interval_envelope.envelope_byte_count, 12u);
  EXPECT_EQ(summary->read_interval_envelope.packet_count, 2u);
  EXPECT_EQ(summary->read_interval_envelope.envelope_byte_count, 12u);
  EXPECT_EQ(summary->write_interval_envelope.packet_count, 1u);
  EXPECT_EQ(summary->write_interval_envelope.envelope_byte_count, 8u);

  ASSERT_EQ(report.source_low_memory_root_summaries.count, 1u);
  ASSERT_NE(report.source_low_memory_root_summaries.head, nullptr);
  const loom_target_compile_report_source_low_memory_root_summary_t*
      root_summary = static_cast<
          const loom_target_compile_report_source_low_memory_root_summary_t*>(
          loom_target_compile_report_vec_const_rows(
              report.source_low_memory_root_summaries.head));
  const loom_target_compile_report_source_low_memory_summary_t*
      root_memory_summary = &root_summary->summary;
  EXPECT_EQ(root_memory_summary->interval_envelope.packet_count, 3u);
  EXPECT_EQ(root_memory_summary->interval_envelope.exact_static_packet_count,
            3u);
  EXPECT_EQ(root_memory_summary->interval_envelope.envelope_byte_count, 12u);
  EXPECT_EQ(root_memory_summary->interval_envelope.unique_byte_count, 12u);
  EXPECT_EQ(root_memory_summary->read_interval_envelope.packet_count, 2u);
  EXPECT_EQ(
      root_memory_summary->read_interval_envelope.exact_static_packet_count,
      2u);
  EXPECT_EQ(root_memory_summary->read_interval_envelope.envelope_byte_count,
            12u);
  EXPECT_EQ(root_memory_summary->read_interval_envelope.unique_byte_count, 12u);
  EXPECT_EQ(root_memory_summary->write_interval_envelope.packet_count, 1u);
  EXPECT_EQ(
      root_memory_summary->write_interval_envelope.exact_static_packet_count,
      1u);
  EXPECT_EQ(root_memory_summary->write_interval_envelope.envelope_byte_count,
            8u);
  EXPECT_EQ(root_memory_summary->write_interval_envelope.unique_byte_count, 8u);

  ASSERT_EQ(report.source_low_memory_argument_summaries.count, 1u);
  ASSERT_NE(report.source_low_memory_argument_summaries.head, nullptr);
  const loom_target_compile_report_source_low_memory_argument_summary_t*
      argument_summary = static_cast<
          const loom_target_compile_report_source_low_memory_argument_summary_t*>(
          loom_target_compile_report_vec_const_rows(
              report.source_low_memory_argument_summaries.head));
  EXPECT_TRUE(iree_string_view_equal(argument_summary->source_root_name,
                                     IREE_SVL("scratch")));
  EXPECT_EQ(argument_summary->summary.interval_envelope.packet_count, 3u);
  EXPECT_EQ(
      argument_summary->summary.interval_envelope.exact_static_packet_count,
      3u);
  EXPECT_EQ(argument_summary->summary.interval_envelope.unique_byte_count, 12u);

  loom_target_compile_report_deinitialize(&report);
}

TEST(CompileReportFormatTest, MergesDuplicateSymbolicSourceLowMemoryIntervals) {
  loom_target_compile_report_t report;
  loom_target_compile_report_initialize(&report, iree_allocator_system());

  const loom_target_compile_report_source_low_memory_row_t rows[] = {
      MakeMemoryRow(IREE_SVL("vector.load"), /*source_op_kind=*/43,
                    IREE_SVL("load"), IREE_SVL("test.load.v2"),
                    /*static_offset_bytes=*/0,
                    /*vector_lane_count=*/2,
                    /*issued_read_byte_count=*/8,
                    /*issued_write_byte_count=*/0,
                    /*dynamic_stride_bytes=*/4,
                    /*vector_lane_stride_bytes=*/4,
                    MakeExactSymbolicInterval(
                        /*begin_min_bytes=*/0, /*begin_max_bytes=*/64,
                        /*end_min_bytes=*/8, /*end_max_bytes=*/72,
                        /*exact_length_bytes=*/8, /*begin_expr_id=*/1,
                        /*end_expr_id=*/2)),
      MakeMemoryRow(IREE_SVL("vector.load"), /*source_op_kind=*/43,
                    IREE_SVL("load"), IREE_SVL("test.load.v2"),
                    /*static_offset_bytes=*/0,
                    /*vector_lane_count=*/2,
                    /*issued_read_byte_count=*/8,
                    /*issued_write_byte_count=*/0,
                    /*dynamic_stride_bytes=*/4,
                    /*vector_lane_stride_bytes=*/4,
                    MakeExactSymbolicInterval(
                        /*begin_min_bytes=*/0, /*begin_max_bytes=*/64,
                        /*end_min_bytes=*/8, /*end_max_bytes=*/72,
                        /*exact_length_bytes=*/8, /*begin_expr_id=*/1,
                        /*end_expr_id=*/2)),
  };
  for (const auto& row : rows) {
    IREE_ASSERT_OK(
        loom_target_compile_report_record_source_low_memory_row(&report, &row));
  }

  const loom_target_compile_report_source_low_memory_summary_t* root_summary =
      GetOnlySourceLowMemoryRootSummary(&report);
  ASSERT_NE(root_summary, nullptr);
  EXPECT_EQ(root_summary->interval_envelope.packet_count, 2u);
  EXPECT_EQ(root_summary->interval_envelope.exact_static_packet_count, 0u);
  EXPECT_EQ(root_summary->interval_envelope.exact_symbolic_packet_count, 2u);
  EXPECT_EQ(root_summary->interval_envelope.envelope_byte_count, 72u);
  EXPECT_EQ(root_summary->interval_envelope.unique_byte_count, 8u);
  EXPECT_EQ(root_summary->read_interval_envelope.packet_count, 2u);
  EXPECT_EQ(root_summary->read_interval_envelope.exact_symbolic_packet_count,
            2u);
  EXPECT_EQ(root_summary->read_interval_envelope.unique_byte_count, 8u);

  loom_target_compile_report_deinitialize(&report);
}

TEST(CompileReportFormatTest, MergesAdjacentSymbolicSourceLowMemoryIntervals) {
  loom_target_compile_report_t report;
  loom_target_compile_report_initialize(&report, iree_allocator_system());

  const loom_target_compile_report_source_low_memory_row_t rows[] = {
      MakeMemoryRow(IREE_SVL("vector.load"), /*source_op_kind=*/43,
                    IREE_SVL("load"), IREE_SVL("test.load.v2"),
                    /*static_offset_bytes=*/0,
                    /*vector_lane_count=*/2,
                    /*issued_read_byte_count=*/8,
                    /*issued_write_byte_count=*/0,
                    /*dynamic_stride_bytes=*/4,
                    /*vector_lane_stride_bytes=*/4,
                    MakeExactSymbolicInterval(
                        /*begin_min_bytes=*/0, /*begin_max_bytes=*/64,
                        /*end_min_bytes=*/8, /*end_max_bytes=*/72,
                        /*exact_length_bytes=*/8, /*begin_expr_id=*/1,
                        /*end_expr_id=*/2)),
      MakeMemoryRow(IREE_SVL("vector.load"), /*source_op_kind=*/43,
                    IREE_SVL("load"), IREE_SVL("test.load.v2"),
                    /*static_offset_bytes=*/8,
                    /*vector_lane_count=*/2,
                    /*issued_read_byte_count=*/8,
                    /*issued_write_byte_count=*/0,
                    /*dynamic_stride_bytes=*/4,
                    /*vector_lane_stride_bytes=*/4,
                    MakeExactSymbolicInterval(
                        /*begin_min_bytes=*/8, /*begin_max_bytes=*/72,
                        /*end_min_bytes=*/16, /*end_max_bytes=*/80,
                        /*exact_length_bytes=*/8, /*begin_expr_id=*/2,
                        /*end_expr_id=*/3)),
  };
  for (const auto& row : rows) {
    IREE_ASSERT_OK(
        loom_target_compile_report_record_source_low_memory_row(&report, &row));
  }

  const loom_target_compile_report_source_low_memory_summary_t* root_summary =
      GetOnlySourceLowMemoryRootSummary(&report);
  ASSERT_NE(root_summary, nullptr);
  EXPECT_EQ(root_summary->interval_envelope.packet_count, 2u);
  EXPECT_EQ(root_summary->interval_envelope.exact_symbolic_packet_count, 2u);
  EXPECT_EQ(root_summary->interval_envelope.envelope_byte_count, 80u);
  EXPECT_EQ(root_summary->interval_envelope.unique_byte_count, 16u);
  EXPECT_EQ(root_summary->read_interval_envelope.packet_count, 2u);
  EXPECT_EQ(root_summary->read_interval_envelope.exact_symbolic_packet_count,
            2u);
  EXPECT_EQ(root_summary->read_interval_envelope.unique_byte_count, 16u);

  loom_target_compile_report_deinitialize(&report);
}

TEST(CompileReportFormatTest, KeepsAmbiguousMixedSourceLowMemoryIntervals) {
  loom_target_compile_report_t report;
  loom_target_compile_report_initialize(&report, iree_allocator_system());

  const loom_target_compile_report_source_low_memory_row_t rows[] = {
      MakeMemoryRow(IREE_SVL("vector.load"), /*source_op_kind=*/43,
                    IREE_SVL("load"), IREE_SVL("test.load.v2"),
                    /*static_offset_bytes=*/0,
                    /*vector_lane_count=*/2,
                    /*issued_read_byte_count=*/8,
                    /*issued_write_byte_count=*/0,
                    /*dynamic_stride_bytes=*/0,
                    /*vector_lane_stride_bytes=*/4,
                    MakeExactSourceInterval(/*begin_bytes=*/0,
                                            /*end_bytes=*/8)),
      MakeMemoryRow(IREE_SVL("vector.load"), /*source_op_kind=*/43,
                    IREE_SVL("load"), IREE_SVL("test.load.v2"),
                    /*static_offset_bytes=*/4,
                    /*vector_lane_count=*/2,
                    /*issued_read_byte_count=*/8,
                    /*issued_write_byte_count=*/0,
                    /*dynamic_stride_bytes=*/4,
                    /*vector_lane_stride_bytes=*/4,
                    MakeExactSymbolicInterval(
                        /*begin_min_bytes=*/4, /*begin_max_bytes=*/64,
                        /*end_min_bytes=*/12, /*end_max_bytes=*/72,
                        /*exact_length_bytes=*/8, /*begin_expr_id=*/11,
                        /*end_expr_id=*/12)),
  };
  for (const auto& row : rows) {
    IREE_ASSERT_OK(
        loom_target_compile_report_record_source_low_memory_row(&report, &row));
  }

  const loom_target_compile_report_source_low_memory_summary_t* root_summary =
      GetOnlySourceLowMemoryRootSummary(&report);
  ASSERT_NE(root_summary, nullptr);
  EXPECT_EQ(root_summary->interval_envelope.packet_count, 2u);
  EXPECT_EQ(root_summary->interval_envelope.exact_static_packet_count, 1u);
  EXPECT_EQ(root_summary->interval_envelope.exact_symbolic_packet_count, 0u);
  EXPECT_EQ(root_summary->interval_envelope.envelope_begin_min_bytes, 0);
  EXPECT_EQ(root_summary->interval_envelope.envelope_end_max_bytes, 72);
  EXPECT_EQ(root_summary->interval_envelope.envelope_byte_count, 72u);
  EXPECT_EQ(root_summary->interval_envelope.unique_byte_count, 8u);

  const loom_target_compile_report_format_options_t options = {
      /*.mode=*/LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS,
  };

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(
      loom_target_compile_report_format_text(&report, &options, &builder));
  iree_string_view_t output = iree_string_builder_view(&builder);
  EXPECT_NE(iree_string_view_find(
                output,
                IREE_SV("interval_envelope={packets:2,begin_min_bytes:0,"
                        "end_max_bytes:72,byte_count:72,"
                        "exact_static_packet_count:1}"),
                0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(iree_string_view_find(
                output,
                IREE_SV("interval_envelope={packets:2,begin_min_bytes:0,"
                        "end_max_bytes:72,byte_count:72,"
                        "exact_static_packet_count:1,unique_byte_count:8}"),
                0),
            IREE_STRING_VIEW_NPOS);
  iree_string_builder_deinitialize(&builder);

  loom_target_compile_report_deinitialize(&report);
}

TEST(CompileReportFormatTest, KeepsImpreciseSourceLowMemoryIntervalEnvelopes) {
  loom_target_compile_report_t report;
  loom_target_compile_report_initialize(&report, iree_allocator_system());

  loom_target_compile_report_memory_interval_t imprecise_interval = {};
  imprecise_interval.flags =
      LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_BEGIN_RANGE |
      LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_END_RANGE;
  imprecise_interval.begin_min_bytes = 0;
  imprecise_interval.begin_max_bytes = 64;
  imprecise_interval.end_min_bytes = 4;
  imprecise_interval.end_max_bytes = 68;
  const loom_target_compile_report_source_low_memory_row_t row = MakeMemoryRow(
      IREE_SVL("vector.load"), /*source_op_kind=*/43, IREE_SVL("load"),
      IREE_SVL("test.load.v1"), /*static_offset_bytes=*/0,
      /*vector_lane_count=*/1, /*issued_read_byte_count=*/4,
      /*issued_write_byte_count=*/0, /*dynamic_stride_bytes=*/4,
      /*vector_lane_stride_bytes=*/4, imprecise_interval);
  IREE_ASSERT_OK(
      loom_target_compile_report_record_source_low_memory_row(&report, &row));

  const loom_target_compile_report_source_low_memory_summary_t* root_summary =
      GetOnlySourceLowMemoryRootSummary(&report);
  ASSERT_NE(root_summary, nullptr);
  EXPECT_EQ(root_summary->interval_envelope.packet_count, 1u);
  EXPECT_EQ(root_summary->interval_envelope.exact_static_packet_count, 0u);
  EXPECT_EQ(root_summary->interval_envelope.exact_symbolic_packet_count, 0u);
  EXPECT_EQ(root_summary->interval_envelope.envelope_byte_count, 68u);
  EXPECT_EQ(root_summary->interval_envelope.unique_byte_count, 0u);

  loom_target_compile_report_deinitialize(&report);
}

TEST(CompileReportFormatTest, FormatsSourceLowMemoryIntervals) {
  loom_target_compile_report_t report;
  loom_target_compile_report_initialize(&report, iree_allocator_system());

  loom_target_compile_report_source_low_memory_row_t row = {};
  row.function_name = IREE_SVL("kernel");
  row.source_op_name = IREE_SVL("vector.load");
  row.source_op_kind = 43;
  row.source_root_name = IREE_SVL("input");
  row.source_root_argument_index = 0;
  row.memory_space = IREE_SVL("global");
  row.operation_kind = IREE_SVL("load");
  row.packet_key = IREE_SVL("test.load.v2");
  row.address_form = IREE_SVL("global_saddr");
  row.dynamic_term_kind = IREE_SVL("vaddr");
  row.element_byte_count = 4;
  row.vector_lane_count = 2;
  row.issued_read_byte_count = 8;
  row.dynamic_stride_bytes = 16;
  row.vector_lane_stride_bytes = 4;
  row.source_interval = {
      /*.flags=*/
      LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_BEGIN_RANGE |
          LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_END_RANGE |
          LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_EXACT_LENGTH,
      /*.begin_min_bytes=*/4,
      /*.begin_max_bytes=*/12,
      /*.end_min_bytes=*/12,
      /*.end_max_bytes=*/20,
      /*.exact_length_bytes=*/8,
  };
  IREE_ASSERT_OK(
      loom_target_compile_report_record_source_low_memory_row(&report, &row));

  const loom_target_compile_report_format_options_t options = {
      /*.mode=*/LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS,
  };

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(
      loom_target_compile_report_format_text(&report, &options, &builder));
  iree_string_view_t output = iree_string_builder_view(&builder);
  EXPECT_NE(iree_string_view_find(
                output,
                IREE_SV("interval_envelope={packets:1,begin_min_bytes:4,"
                        "end_max_bytes:20,byte_count:16}"),
                0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                output,
                IREE_SV("read_interval_envelope={packets:1,"
                        "begin_min_bytes:4,end_max_bytes:20,byte_count:16}"),
                0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(output,
                            IREE_SV("source_interval={begin_min_bytes:4,"
                                    "begin_max_bytes:12,end_min_bytes:12,"
                                    "end_max_bytes:20,exact_length_bytes:8}"),
                            0),
      IREE_STRING_VIEW_NPOS);
  iree_string_builder_deinitialize(&builder);

  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  IREE_ASSERT_OK(
      loom_target_compile_report_format_json(&report, &options, &stream));
  output = iree_string_builder_view(&builder);
  EXPECT_NE(iree_string_view_find(
                output,
                IREE_SV("\"interval_envelope\":{\"packet_count\":1,"
                        "\"begin_min_bytes\":4,\"end_max_bytes\":20,"
                        "\"byte_count\":16}"),
                0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                output,
                IREE_SV("\"source_interval\":{\"begin_min_bytes\":4,"
                        "\"begin_max_bytes\":12,\"end_min_bytes\":12,"
                        "\"end_max_bytes\":20,\"exact_length_bytes\":8}"),
                0),
            IREE_STRING_VIEW_NPOS);
  iree_string_builder_deinitialize(&builder);

  loom_target_compile_report_deinitialize(&report);
}

TEST(CompileReportFormatTest, FormatsSourceLowMemorySummaryEconomics) {
  loom_target_compile_report_t report;
  loom_target_compile_report_initialize(&report, iree_allocator_system());

  loom_target_compile_report_workload_t workload = {};
  workload.flags = LOOM_TARGET_COMPILE_REPORT_WORKLOAD_DISPATCH_WORKITEM_COUNT;
  workload.dispatch_workitem_count = 16;
  loom_target_compile_report_record_workload(&report, &workload);

  const loom_target_compile_report_source_low_memory_row_t load_row =
      MakeMemoryRow(IREE_SVL("vector.load"), /*source_op_kind=*/43,
                    IREE_SVL("load"), IREE_SVL("test.load.v2"),
                    /*static_offset_bytes=*/0,
                    /*vector_lane_count=*/2,
                    /*issued_read_byte_count=*/8,
                    /*issued_write_byte_count=*/0,
                    /*dynamic_stride_bytes=*/8,
                    /*vector_lane_stride_bytes=*/4,
                    MakeExactSourceInterval(/*begin_bytes=*/0,
                                            /*end_bytes=*/8));
  const loom_target_compile_report_source_low_memory_row_t store_row =
      MakeMemoryRow(IREE_SVL("view.store"), /*source_op_kind=*/44,
                    IREE_SVL("store"), IREE_SVL("test.store.v1"),
                    /*static_offset_bytes=*/8,
                    /*vector_lane_count=*/1,
                    /*issued_read_byte_count=*/0,
                    /*issued_write_byte_count=*/4,
                    /*dynamic_stride_bytes=*/4,
                    /*vector_lane_stride_bytes=*/4,
                    MakeExactSourceInterval(/*begin_bytes=*/8,
                                            /*end_bytes=*/12));
  IREE_ASSERT_OK(loom_target_compile_report_record_source_low_memory_row(
      &report, &load_row));
  IREE_ASSERT_OK(loom_target_compile_report_record_source_low_memory_row(
      &report, &store_row));

  const loom_target_compile_report_format_options_t options = {
      /*.mode=*/LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_SUMMARY,
  };
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  IREE_ASSERT_OK(
      loom_target_compile_report_format_json(&report, &options, &stream));

  iree_string_view_t output = iree_string_builder_view(&builder);
  EXPECT_NE(iree_string_view_find(output, IREE_SV("\"source_low\":{"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output,
                                  IREE_SV("\"memory\":{"
                                          "\"packet_count\":2"),
                                  0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(output,
                            IREE_SV("\"dispatch_source\":{\"read_bytes\":128,"
                                    "\"write_bytes\":64,\"total_bytes\":192}"),
                            0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(output,
                            IREE_SV("\"dispatch_issued\":{\"read_bytes\":128,"
                                    "\"write_bytes\":64,\"total_bytes\":192}"),
                            0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                output,
                IREE_SV("\"economics\":{\"memory\":{\"source_low\":{"
                        "\"packet_count\":2"),
                0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output,
                                  IREE_SV("\"source_low\":{\"packet_count\":2,"
                                          "\"load_packet_count\":1,"
                                          "\"store_packet_count\":1,"
                                          "\"scalar_packet_count\":1,"
                                          "\"vector_packet_count\":1"),
                                  0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output,
                                  IREE_SV("\"dispatch_source\":{\"read_bytes\":"
                                          "128,\"write_bytes\":64,"
                                          "\"total_bytes\":192}"),
                                  0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(iree_string_view_find(output, IREE_SV("\"memory_rows\""), 0),
            IREE_STRING_VIEW_NPOS);

  iree_string_builder_deinitialize(&builder);
  loom_target_compile_report_deinitialize(&report);
}

TEST(CompileReportFormatTest, FormatsExactSymbolicSourceLowMemoryIntervals) {
  loom_target_compile_report_t report;
  loom_target_compile_report_initialize(&report, iree_allocator_system());

  const loom_target_compile_report_source_low_memory_row_t row = MakeMemoryRow(
      IREE_SVL("vector.load"), /*source_op_kind=*/43, IREE_SVL("load"),
      IREE_SVL("test.load.v2"), /*static_offset_bytes=*/4,
      /*vector_lane_count=*/2, /*issued_read_byte_count=*/8,
      /*issued_write_byte_count=*/0, /*dynamic_stride_bytes=*/4,
      /*vector_lane_stride_bytes=*/4,
      MakeExactSymbolicInterval(/*begin_min_bytes=*/4,
                                /*begin_max_bytes=*/64,
                                /*end_min_bytes=*/12,
                                /*end_max_bytes=*/72,
                                /*exact_length_bytes=*/8,
                                /*begin_expr_id=*/1, /*end_expr_id=*/2));
  IREE_ASSERT_OK(
      loom_target_compile_report_record_source_low_memory_row(&report, &row));

  const loom_target_compile_report_format_options_t options = {
      /*.mode=*/LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS,
  };

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(
      loom_target_compile_report_format_text(&report, &options, &builder));
  iree_string_view_t output = iree_string_builder_view(&builder);
  EXPECT_NE(iree_string_view_find(
                output,
                IREE_SV("interval_envelope={packets:1,begin_min_bytes:4,"
                        "end_max_bytes:72,byte_count:68,"
                        "exact_symbolic_packet_count:1,unique_byte_count:8}"),
                0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(iree_string_view_find(output, IREE_SV("expr_id"), 0),
            IREE_STRING_VIEW_NPOS);
  iree_string_builder_deinitialize(&builder);

  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  IREE_ASSERT_OK(
      loom_target_compile_report_format_json(&report, &options, &stream));
  output = iree_string_builder_view(&builder);
  EXPECT_NE(iree_string_view_find(
                output,
                IREE_SV("\"interval_envelope\":{\"packet_count\":1,"
                        "\"begin_min_bytes\":4,\"end_max_bytes\":72,"
                        "\"byte_count\":68,\"exact_symbolic_packet_count\":1,"
                        "\"unique_byte_count\":8}"),
                0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(iree_string_view_find(output, IREE_SV("expr_id"), 0),
            IREE_STRING_VIEW_NPOS);
  iree_string_builder_deinitialize(&builder);

  loom_target_compile_report_deinitialize(&report);
}

TEST(CompileReportFormatTest, FormatsExactStaticSourceLowMemoryIntervals) {
  loom_target_compile_report_t report;
  loom_target_compile_report_initialize(&report, iree_allocator_system());

  const loom_target_compile_report_source_low_memory_row_t row = MakeMemoryRow(
      IREE_SVL("vector.load"), /*source_op_kind=*/43, IREE_SVL("load"),
      IREE_SVL("test.load.v2"), /*static_offset_bytes=*/4,
      /*vector_lane_count=*/2, /*issued_read_byte_count=*/8,
      /*issued_write_byte_count=*/0, /*dynamic_stride_bytes=*/0,
      /*vector_lane_stride_bytes=*/4,
      MakeExactSourceInterval(/*begin_bytes=*/4, /*end_bytes=*/12));
  IREE_ASSERT_OK(
      loom_target_compile_report_record_source_low_memory_row(&report, &row));

  const loom_target_compile_report_format_options_t options = {
      /*.mode=*/LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS,
  };

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(
      loom_target_compile_report_format_text(&report, &options, &builder));
  iree_string_view_t output = iree_string_builder_view(&builder);
  EXPECT_NE(iree_string_view_find(
                output,
                IREE_SV("interval_envelope={packets:1,begin_min_bytes:4,"
                        "end_max_bytes:12,byte_count:8,"
                        "exact_static_packet_count:1,unique_byte_count:8}"),
                0),
            IREE_STRING_VIEW_NPOS);
  iree_string_builder_deinitialize(&builder);

  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  IREE_ASSERT_OK(
      loom_target_compile_report_format_json(&report, &options, &stream));
  output = iree_string_builder_view(&builder);
  EXPECT_NE(iree_string_view_find(
                output,
                IREE_SV("\"interval_envelope\":{\"packet_count\":1,"
                        "\"begin_min_bytes\":4,\"end_max_bytes\":12,"
                        "\"byte_count\":8,\"exact_static_packet_count\":1,"
                        "\"unique_byte_count\":8}"),
                0),
            IREE_STRING_VIEW_NPOS);
  iree_string_builder_deinitialize(&builder);

  loom_target_compile_report_deinitialize(&report);
}

}  // namespace
}  // namespace loom
