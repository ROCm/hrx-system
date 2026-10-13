// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include <algorithm>
#include <bitset>
#include <random>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/types.h"
#include "loom/target/math_policy.h"
#include "loom/target/reporting/format.h"
#include "loom/target/reporting/memory_summary_test_util.h"

namespace loom {
namespace {

constexpr uint32_t kTestSourceRejectionDetail = 4;

static loom_target_compile_report_memory_interval_t MakeExactSourceInterval(
    int64_t begin_bytes, int64_t end_bytes) {
  return {
      .flags = LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_BEGIN_RANGE |
               LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_END_RANGE |
               LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_EXACT_LENGTH,
      .begin_min_bytes = begin_bytes,
      .begin_max_bytes = begin_bytes,
      .end_min_bytes = end_bytes,
      .end_max_bytes = end_bytes,
      .exact_length_bytes = static_cast<uint64_t>(end_bytes - begin_bytes),
  };
}

static loom_target_compile_report_memory_interval_t MakeExactSymbolicInterval(
    int64_t begin_min_bytes, int64_t begin_max_bytes, int64_t end_min_bytes,
    int64_t end_max_bytes, uint64_t exact_length_bytes, uint32_t begin_expr_id,
    uint32_t end_expr_id) {
  return {
      .flags = LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_BEGIN_RANGE |
               LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_END_RANGE |
               LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_EXACT_LENGTH |
               LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_BEGIN_EXPR |
               LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_END_EXPR,
      .begin_min_bytes = begin_min_bytes,
      .begin_max_bytes = begin_max_bytes,
      .end_min_bytes = end_min_bytes,
      .end_max_bytes = end_max_bytes,
      .exact_length_bytes = exact_length_bytes,
      .begin_expr_id = begin_expr_id,
      .end_expr_id = end_expr_id,
  };
}

static loom_target_compile_report_source_low_memory_row_t MakeMemoryRow(
    iree_string_view_t source_op_name, uint32_t source_op_kind,
    iree_string_view_t operation_kind, iree_string_view_t packet_key,
    int64_t static_offset_bytes, uint32_t vector_lane_count,
    uint32_t issued_read_byte_count, uint32_t issued_write_byte_count,
    uint32_t dynamic_stride_bytes, uint32_t vector_lane_stride_bytes,
    loom_target_compile_report_memory_interval_t source_interval) {
  loom_target_compile_report_source_low_memory_row_t row = {
      .function_name = IREE_SVL("kernel"),
      .source_op_name = source_op_name,
      .source_op_kind = source_op_kind,
      .source_root_name = IREE_SVL("scratch"),
      .source_root_argument_index = 1,
      .memory_space = IREE_SVL("workgroup"),
      .operation_kind = operation_kind,
      .packet_key = packet_key,
      .address_form = IREE_SVL("default"),
      .dynamic_term_kind = IREE_SVL("vaddr"),
      .static_offset_bytes = static_offset_bytes,
      .element_byte_count = 4,
      .vector_lane_count = vector_lane_count,
      .issued_read_byte_count = issued_read_byte_count,
      .issued_write_byte_count = issued_write_byte_count,
      .dynamic_stride_bytes = dynamic_stride_bytes,
      .vector_lane_stride_bytes = vector_lane_stride_bytes,
      .source_interval = source_interval,
      .execution_count_plus_one = 2,
  };
  return row;
}

static void SetBankService(
    iree_string_view_t proof, iree_string_view_t classification,
    iree_string_view_t unknown_reason, uint16_t required_rounds,
    uint16_t uncontended_rounds, uint16_t maximum_request_multiplicity,
    loom_target_compile_report_source_low_memory_row_t* row) {
  row->bank_service.proof = proof;
  row->bank_service.classification = classification;
  row->bank_service.model_key = IREE_SVL("test.ds_read_b32");
  row->bank_service.model_revision = IREE_SVL("test-model@1");
  row->bank_service.model_evidence = IREE_SVL("silicon-calibrated");
  row->bank_service.request_policy = IREE_SVL("collapse-identical");
  row->bank_service.unknown_reason = unknown_reason;
  row->bank_service.wave_size = 32;
  row->bank_service.bank_count = 32;
  row->bank_service.bank_word_byte_count = 4;
  row->bank_service.packet_byte_count = 4;
  row->bank_service.required_rounds = required_rounds;
  row->bank_service.uncontended_rounds = uncontended_rounds;
  row->bank_service.extra_rounds = required_rounds - uncontended_rounds;
  row->bank_service.maximum_request_multiplicity = maximum_request_multiplicity;
}

static void SetSubgroupAccess(
    iree_string_view_t proof, iree_string_view_t interval_coverage,
    iree_string_view_t unknown_reason, uint64_t requested_byte_count,
    uint64_t unique_byte_count, uint64_t span_byte_count,
    loom_target_compile_report_source_low_memory_row_t* row) {
  row->subgroup_access.proof = proof;
  row->subgroup_access.lane_address_proof =
      IREE_SVL("compiled-fragment-lane-register-layout");
  row->subgroup_access.active_lane_proof = IREE_SVL("full-wave");
  row->subgroup_access.lane_mapping = IREE_SVL("linear");
  row->subgroup_access.interval_coverage = interval_coverage;
  row->subgroup_access.unknown_reason = unknown_reason;
  row->subgroup_access.subgroup_size = 32;
  row->subgroup_access.active_lane_count = 32;
  row->subgroup_access.lane_term_count = 1;
  row->subgroup_access.lane_terms[0] = {
      .divisor = 1,
      .modulus = 0,
      .byte_stride = 4,
  };
  row->subgroup_access.per_lane_packet_byte_count = 4;
  row->subgroup_access.linear_lane_byte_stride = 4;
  row->subgroup_access.subgroup_requested_byte_count = requested_byte_count;
  row->subgroup_access.subgroup_unique_byte_count = unique_byte_count;
  row->subgroup_access.subgroup_span_byte_count = span_byte_count;
  row->subgroup_access.distinct_lane_address_count = 32;
  row->subgroup_access.contiguous_region_count =
      iree_string_view_equal(interval_coverage, IREE_SV("gapped")) ? 2 : 1;
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

TEST(CompileReportFormatTest, PreservesBankServiceProofAndDynamicCoverage) {
  loom_target_compile_report_t entry_report;
  loom_target_compile_report_initialize(&entry_report, iree_allocator_system());
  testing::MemoryReportBatch entry_report_memory(&entry_report);
  entry_report.function_name = IREE_SVL("kernel_v0");
  entry_report.lowered_symbol = IREE_SVL("kernel");

  loom_target_compile_report_source_low_memory_row_t rows[] = {
      MakeMemoryRow(IREE_SVL("view.load"), /*source_op_kind=*/43,
                    IREE_SVL("load"), IREE_SVL("test.ds_read_b32"),
                    /*static_offset_bytes=*/0,
                    /*vector_lane_count=*/1,
                    /*issued_read_byte_count=*/4,
                    /*issued_write_byte_count=*/0,
                    /*dynamic_stride_bytes=*/4,
                    /*vector_lane_stride_bytes=*/4,
                    MakeExactSourceInterval(/*begin_bytes=*/0,
                                            /*end_bytes=*/4)),
      MakeMemoryRow(IREE_SVL("view.load"), /*source_op_kind=*/43,
                    IREE_SVL("load"), IREE_SVL("test.ds_read_b32"),
                    /*static_offset_bytes=*/4,
                    /*vector_lane_count=*/1,
                    /*issued_read_byte_count=*/4,
                    /*issued_write_byte_count=*/0,
                    /*dynamic_stride_bytes=*/4,
                    /*vector_lane_stride_bytes=*/4,
                    MakeExactSourceInterval(/*begin_bytes=*/4,
                                            /*end_bytes=*/8)),
      MakeMemoryRow(IREE_SVL("view.load"), /*source_op_kind=*/43,
                    IREE_SVL("load"), IREE_SVL("test.ds_read_b32"),
                    /*static_offset_bytes=*/8,
                    /*vector_lane_count=*/1,
                    /*issued_read_byte_count=*/4,
                    /*issued_write_byte_count=*/0,
                    /*dynamic_stride_bytes=*/4,
                    /*vector_lane_stride_bytes=*/4,
                    MakeExactSourceInterval(/*begin_bytes=*/8,
                                            /*end_bytes=*/12)),
  };
  SetBankService(IREE_SVL("exact"), IREE_SVL("conflict-free"),
                 iree_string_view_empty(), /*required_rounds=*/1,
                 /*uncontended_rounds=*/1,
                 /*maximum_request_multiplicity=*/1, &rows[0]);
  rows[0].execution_count_plus_one = 3;
  SetBankService(IREE_SVL("exact"), IREE_SVL("conflicted"),
                 iree_string_view_empty(), /*required_rounds=*/3,
                 /*uncontended_rounds=*/1,
                 /*maximum_request_multiplicity=*/3, &rows[1]);
  rows[1].execution_count_plus_one = 4;
  SetBankService(IREE_SVL("unknown"), iree_string_view_empty(),
                 IREE_SVL("base-residue-unknown"), /*required_rounds=*/0,
                 /*uncontended_rounds=*/0,
                 /*maximum_request_multiplicity=*/0, &rows[2]);
  rows[2].execution_count_plus_one =
      LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_MEMORY_EXECUTION_COUNT_PLUS_ONE_UNKNOWN;
  SetSubgroupAccess(IREE_SVL("exact"), IREE_SVL("dense"),
                    iree_string_view_empty(), /*requested_byte_count=*/128,
                    /*unique_byte_count=*/64, /*span_byte_count=*/64, &rows[0]);
  SetSubgroupAccess(IREE_SVL("exact"), IREE_SVL("gapped"),
                    iree_string_view_empty(), /*requested_byte_count=*/128,
                    /*unique_byte_count=*/128, /*span_byte_count=*/256,
                    &rows[1]);
  SetSubgroupAccess(IREE_SVL("unknown"), iree_string_view_empty(),
                    IREE_SVL("active-lane-control-not-uniform"),
                    /*requested_byte_count=*/0, /*unique_byte_count=*/0,
                    /*span_byte_count=*/0, &rows[2]);
  for (const auto& row : rows) {
    IREE_ASSERT_OK(entry_report_memory.Append(&row));
  }

  const loom_target_compile_report_bank_service_summary_t* summary =
      &entry_report.bank_service_summary;
  EXPECT_EQ(summary->modeled_packet_count, 3u);
  EXPECT_EQ(summary->exact_packet_count, 2u);
  EXPECT_EQ(summary->unknown_packet_count, 1u);
  EXPECT_EQ(summary->conflict_free_packet_count, 1u);
  EXPECT_EQ(summary->conflicted_packet_count, 1u);
  EXPECT_EQ(summary->required_round_count, 4u);
  EXPECT_EQ(summary->uncontended_round_count, 2u);
  EXPECT_EQ(summary->extra_round_count, 2u);
  EXPECT_EQ(summary->maximum_request_multiplicity, 3u);
  EXPECT_EQ(summary->exact_dynamic_packet_count, 2u);
  EXPECT_EQ(summary->unknown_dynamic_packet_count, 1u);
  EXPECT_EQ(summary->dynamic_packet_count, 5u);
  EXPECT_EQ(summary->dynamic_required_round_count, 11u);
  EXPECT_EQ(summary->dynamic_uncontended_round_count, 5u);
  EXPECT_EQ(summary->dynamic_extra_round_count, 6u);

  ASSERT_EQ(entry_report.source_low_bank_service_summaries.count, 1u);
  ASSERT_NE(entry_report.source_low_bank_service_summaries.head, nullptr);
  const auto* group = static_cast<
      const loom_target_compile_report_source_low_bank_service_summary_t*>(
      loom_target_compile_report_vec_const_rows(
          entry_report.source_low_bank_service_summaries.head));
  EXPECT_EQ(group[0].summary.modeled_packet_count, 3u);
  EXPECT_TRUE(iree_string_view_equal(group[0].unknown_reason,
                                     IREE_SVL("base-residue-unknown")));
  EXPECT_FALSE(group[0].has_mixed_unknown_reasons);

  const loom_target_compile_report_subgroup_access_summary_t* access_summary =
      &entry_report.subgroup_access_summary;
  EXPECT_EQ(access_summary->modeled_packet_count, 3u);
  EXPECT_EQ(access_summary->exact_packet_count, 2u);
  EXPECT_EQ(access_summary->unknown_packet_count, 1u);
  EXPECT_EQ(access_summary->dense_packet_count, 1u);
  EXPECT_EQ(access_summary->gapped_packet_count, 1u);
  EXPECT_EQ(access_summary->overlapping_packet_count, 1u);
  EXPECT_EQ(access_summary->exact_dynamic_packet_count, 2u);
  EXPECT_EQ(access_summary->unknown_dynamic_packet_count, 1u);
  EXPECT_EQ(access_summary->dynamic_packet_count, 5u);
  EXPECT_EQ(access_summary->dynamic_dense_packet_count, 2u);
  EXPECT_EQ(access_summary->dynamic_gapped_packet_count, 3u);
  EXPECT_EQ(access_summary->dynamic_overlapping_packet_count, 2u);
  EXPECT_EQ(entry_report.source_low_subgroup_access_summaries.count, 3u);

  auto unmodeled = MakeMemoryRow(
      IREE_SVL("view.load"), /*source_op_kind=*/43, IREE_SVL("load"),
      IREE_SVL("test.ds_read_b16"), /*static_offset_bytes=*/12,
      /*vector_lane_count=*/1, /*issued_read_byte_count=*/2,
      /*issued_write_byte_count=*/0, /*dynamic_stride_bytes=*/2,
      /*vector_lane_stride_bytes=*/2,
      MakeExactSourceInterval(/*begin_bytes=*/12, /*end_bytes=*/14));
  unmodeled.element_byte_count = 2;
  unmodeled.bank_service.proof = IREE_SVL("unmodeled");
  unmodeled.bank_service.unknown_reason =
      IREE_SVL("packet-wave-model-unavailable");
  unmodeled.bank_service.wave_size = 32;
  IREE_ASSERT_OK(entry_report_memory.Append(&unmodeled));
  IREE_ASSERT_OK(entry_report_memory.Append(&unmodeled));
  IREE_ASSERT_OK(entry_report_memory.Finish());
  EXPECT_EQ(entry_report.bank_service_summary.unmodeled_packet_count, 2u);
  EXPECT_EQ(entry_report.bank_service_summary.modeled_packet_count, 3u);
  EXPECT_EQ(entry_report.bank_service_summary.unknown_dynamic_packet_count, 3u);

  loom_target_compile_report_t report;
  loom_target_compile_report_initialize(&report, iree_allocator_system());
  IREE_ASSERT_OK(
      loom_target_compile_report_record_entry_report(&report, &entry_report));
  loom_target_compile_report_deinitialize(&entry_report);
  ASSERT_EQ(report.entry_rows.count, 1u);
  const auto* entry = static_cast<const loom_target_compile_report_entry_t*>(
      loom_target_compile_report_vec_const_rows(report.entry_rows.head));
  EXPECT_EQ(entry[0].bank_service_summary.dynamic_extra_round_count, 6u);
  EXPECT_EQ(entry[0].subgroup_access_summary.dynamic_gapped_packet_count, 3u);
  EXPECT_EQ(report.bank_service_summary.dynamic_extra_round_count, 6u);
  EXPECT_EQ(report.subgroup_access_summary.dynamic_gapped_packet_count, 3u);
  ASSERT_EQ(report.source_low_bank_service_summaries.count, 2u);
  EXPECT_EQ(entry[0].bank_service_summary.unmodeled_packet_count, 2u);
  EXPECT_EQ(report.bank_service_summary.unmodeled_packet_count, 2u);
  ASSERT_EQ(report.source_low_subgroup_access_summaries.count, 3u);

  loom_target_compile_report_deinitialize(&report);
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
    testing::MemoryReportBatch entry_report_memory(&entry_report);
    IREE_ASSERT_OK(entry_report_memory.Append(&rows[i]));
    IREE_ASSERT_OK(entry_report_memory.Finish());
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
  testing::MemoryReportBatch report_memory(&report);

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
    IREE_ASSERT_OK(report_memory.Append(&rows[i]));
  }
  IREE_ASSERT_OK(report_memory.Finish());

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
  testing::MemoryReportBatch report_memory(&report);

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
    IREE_ASSERT_OK(report_memory.Append(&row));
  }
  IREE_ASSERT_OK(report_memory.Finish());

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
  testing::MemoryReportBatch report_memory(&report);

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
    IREE_ASSERT_OK(report_memory.Append(&row));
  }
  IREE_ASSERT_OK(report_memory.Finish());

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
  testing::MemoryReportBatch report_memory(&report);

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
    IREE_ASSERT_OK(report_memory.Append(&row));
  }
  IREE_ASSERT_OK(report_memory.Finish());

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
  testing::MemoryReportBatch report_memory(&report);

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
    IREE_ASSERT_OK(report_memory.Append(&row));
  }
  IREE_ASSERT_OK(report_memory.Finish());

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
      .mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS,
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
  testing::MemoryReportBatch report_memory(&report);

  loom_target_compile_report_memory_interval_t imprecise_interval = {
      .flags = LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_BEGIN_RANGE |
               LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_END_RANGE,
      .begin_min_bytes = 0,
      .begin_max_bytes = 64,
      .end_min_bytes = 4,
      .end_max_bytes = 68,
  };
  const loom_target_compile_report_source_low_memory_row_t row = MakeMemoryRow(
      IREE_SVL("vector.load"), /*source_op_kind=*/43, IREE_SVL("load"),
      IREE_SVL("test.load.v1"), /*static_offset_bytes=*/0,
      /*vector_lane_count=*/1, /*issued_read_byte_count=*/4,
      /*issued_write_byte_count=*/0, /*dynamic_stride_bytes=*/4,
      /*vector_lane_stride_bytes=*/4, imprecise_interval);
  IREE_ASSERT_OK(report_memory.Append(&row));
  IREE_ASSERT_OK(report_memory.Finish());

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
  testing::MemoryReportBatch report_memory(&report);

  loom_target_compile_report_source_low_memory_row_t row = {
      .function_name = IREE_SVL("kernel"),
      .source_op_name = IREE_SVL("vector.load"),
      .source_op_kind = 43,
      .source_root_name = IREE_SVL("input"),
      .source_root_argument_index = 0,
      .memory_space = IREE_SVL("global"),
      .operation_kind = IREE_SVL("load"),
      .packet_key = IREE_SVL("test.load.v2"),
      .address_form = IREE_SVL("global_saddr"),
      .dynamic_term_kind = IREE_SVL("vaddr"),
      .element_byte_count = 4,
      .vector_lane_count = 2,
      .issued_read_byte_count = 8,
      .dynamic_stride_bytes = 16,
      .vector_lane_stride_bytes = 4,
      .source_interval =
          {
              .flags = LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_BEGIN_RANGE |
                       LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_END_RANGE |
                       LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_EXACT_LENGTH,
              .begin_min_bytes = 4,
              .begin_max_bytes = 12,
              .end_min_bytes = 12,
              .end_max_bytes = 20,
              .exact_length_bytes = 8,
          },
  };
  IREE_ASSERT_OK(report_memory.Append(&row));
  IREE_ASSERT_OK(report_memory.Finish());

  const loom_target_compile_report_format_options_t options = {
      .mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS,
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
  testing::MemoryReportBatch report_memory(&report);

  loom_target_compile_report_workload_t workload = {
      .flags = LOOM_TARGET_COMPILE_REPORT_WORKLOAD_DISPATCH_WORKITEM_COUNT,
      .dispatch_workitem_count = 16,
  };
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
  IREE_ASSERT_OK(report_memory.Append(&load_row));
  IREE_ASSERT_OK(report_memory.Append(&store_row));
  IREE_ASSERT_OK(report_memory.Finish());

  const loom_target_compile_report_format_options_t options = {
      .mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_SUMMARY,
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
  testing::MemoryReportBatch report_memory(&report);

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
  IREE_ASSERT_OK(report_memory.Append(&row));
  IREE_ASSERT_OK(report_memory.Finish());

  const loom_target_compile_report_format_options_t options = {
      .mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS,
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
  testing::MemoryReportBatch report_memory(&report);

  const loom_target_compile_report_source_low_memory_row_t row = MakeMemoryRow(
      IREE_SVL("vector.load"), /*source_op_kind=*/43, IREE_SVL("load"),
      IREE_SVL("test.load.v2"), /*static_offset_bytes=*/4,
      /*vector_lane_count=*/2, /*issued_read_byte_count=*/8,
      /*issued_write_byte_count=*/0, /*dynamic_stride_bytes=*/0,
      /*vector_lane_stride_bytes=*/4,
      MakeExactSourceInterval(/*begin_bytes=*/4, /*end_bytes=*/12));
  IREE_ASSERT_OK(report_memory.Append(&row));
  IREE_ASSERT_OK(report_memory.Finish());

  const loom_target_compile_report_format_options_t options = {
      .mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS,
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

static uint32_t IntervalProofKind(
    const loom_target_compile_report_memory_interval_t& interval) {
  if (interval.begin_min_bytes == interval.begin_max_bytes &&
      interval.end_min_bytes == interval.end_max_bytes) {
    return 1;
  }
  const auto symbolic = LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_BEGIN_EXPR |
                        LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_END_EXPR |
                        LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_EXACT_LENGTH;
  return (interval.flags & symbolic) == symbolic ? 2 : 0;
}

// Direct finite-byte oracle for these small fixtures. It deliberately retains
// encounter order: the first duplicate or ambiguous predecessor is decisive.
static loom_target_compile_report_memory_interval_summary_t ReferenceIntervals(
    const std::vector<loom_target_compile_report_source_low_memory_row_t>& rows,
    iree_string_view_t operation) {
  loom_target_compile_report_memory_interval_summary_t result = {};
  for (size_t i = 0; i < rows.size(); ++i) {
    if (!iree_string_view_is_empty(operation) &&
        !iree_string_view_equal(rows[i].operation_kind, operation)) {
      continue;
    }
    const auto& interval = rows[i].source_interval;
    const auto kind = IntervalProofKind(interval);
    if (!kind) {
      continue;
    }
    std::bitset<64> covered;
    bool ambiguous = false;
    bool duplicate = false;
    for (size_t j = 0; j < i; ++j) {
      if (!iree_string_view_is_empty(operation) &&
          !iree_string_view_equal(rows[j].operation_kind, operation)) {
        continue;
      }
      const auto& prior = rows[j].source_interval;
      const auto prior_kind = IntervalProofKind(prior);
      if (!prior_kind) {
        continue;
      }
      const bool disjoint = prior.end_max_bytes <= interval.begin_min_bytes ||
                            interval.end_max_bytes <= prior.begin_min_bytes;
      if (kind == 1 && prior_kind == 1) {
        for (uint64_t byte = 0; byte < interval.exact_length_bytes; ++byte) {
          const int64_t address =
              interval.begin_min_bytes + static_cast<int64_t>(byte);
          if (address >= prior.begin_min_bytes &&
              address < prior.end_max_bytes) {
            covered.set(byte);
          }
        }
      } else if (kind == 1 || prior_kind == 1) {
        ambiguous = !disjoint;
      } else if (prior.begin_expr_id == interval.begin_expr_id &&
                 prior.end_expr_id == interval.end_expr_id) {
        duplicate = true;
      } else if (prior.end_expr_id != interval.begin_expr_id &&
                 prior.begin_expr_id != interval.end_expr_id) {
        ambiguous = !disjoint;
      }
      if (ambiguous || duplicate) {
        break;
      }
    }
    if (ambiguous) {
      continue;
    }
    result.exact_static_packet_count += kind == 1;
    result.exact_symbolic_packet_count += kind == 2;
    if (!duplicate) {
      result.unique_byte_count += interval.exact_length_bytes - covered.count();
    }
  }
  return result;
}

TEST(CompileReportMemoryTest, PreservesMixedIntervalProofsAndDirectionOrder) {
  std::mt19937 random(27);
  for (size_t trial = 0; trial < 32; ++trial) {
    SCOPED_TRACE(trial);
    loom_target_compile_report_t report;
    loom_target_compile_report_initialize(&report, iree_allocator_system());
    testing::MemoryReportBatch batch(&report);
    std::vector<loom_target_compile_report_source_low_memory_row_t> rows;
    for (size_t i = 0; i < 96; ++i) {
      const uint32_t base = random() % 5;
      const int64_t offset = static_cast<int64_t>(random() % 32) * 4 - 64;
      const uint32_t lane_count = 1 + random() % 8;
      const int64_t length = lane_count * 4;
      const int64_t begin =
          offset + (base ? 4 * static_cast<int64_t>(random() % 32) : 0);
      auto interval =
          base == 0
              ? MakeExactSourceInterval(begin, begin + length)
              : MakeExactSymbolicInterval(
                    begin, begin + 16, begin + length, begin + 16 + length,
                    length,
                    base * 256 + static_cast<uint32_t>((offset + 64) / 4),
                    base * 256 +
                        static_cast<uint32_t>((offset + length + 64) / 4));
      if (base == 4) {
        interval.flags &=
            ~(LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_BEGIN_EXPR |
              LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_END_EXPR);
      }
      const bool read = random() % 3 != 0;
      rows.push_back(MakeMemoryRow(
          read ? IREE_SV("vector.load") : IREE_SV("vector.store"),
          read ? 43 : 44, read ? IREE_SV("load") : IREE_SV("store"),
          read ? IREE_SV("test.load") : IREE_SV("test.store"), begin,
          lane_count, read ? static_cast<uint32_t>(length) : 0,
          read ? 0 : static_cast<uint32_t>(length), 4, 4, interval));
      IREE_ASSERT_OK(batch.Append(&rows.back()));
    }
    IREE_ASSERT_OK(batch.Finish());
    const auto* summary = GetOnlySourceLowMemoryRootSummary(&report);
    ASSERT_NE(summary, nullptr);
    const loom_target_compile_report_memory_interval_summary_t* actual[] = {
        &summary->interval_envelope, &summary->read_interval_envelope,
        &summary->write_interval_envelope};
    const iree_string_view_t operations[] = {
        {}, IREE_SV("load"), IREE_SV("store")};
    for (size_t direction = 0; direction < IREE_ARRAYSIZE(actual);
         ++direction) {
      SCOPED_TRACE(direction);
      const auto expected = ReferenceIntervals(rows, operations[direction]);
      EXPECT_EQ(actual[direction]->exact_static_packet_count,
                expected.exact_static_packet_count);
      EXPECT_EQ(actual[direction]->exact_symbolic_packet_count,
                expected.exact_symbolic_packet_count);
      EXPECT_EQ(actual[direction]->unique_byte_count,
                expected.unique_byte_count);
    }
    loom_target_compile_report_deinitialize(&report);
  }
}

TEST(CompileReportMemoryTest, FindsConflictAfterBothAdjacencyExclusions) {
  // [20,24) and [12,16) are each adjacent to [16,20). An intervening
  // [12,24) remains a conflict even after both adjacent witnesses are skipped.
  for (bool reverse : {false, true}) {
    for (bool include_conflict : {false, true}) {
      SCOPED_TRACE(reverse);
      SCOPED_TRACE(include_conflict);
      loom_target_compile_report_t report;
      loom_target_compile_report_initialize(&report, iree_allocator_system());
      testing::MemoryReportBatch batch(&report);
      const int64_t intervals[][2] = {
          {reverse ? 12 : 20, reverse ? 16 : 24},
          {reverse ? 20 : 12, reverse ? 24 : 16},
          {12, 24},
          {16, 20},
      };
      for (size_t i = 0; i < IREE_ARRAYSIZE(intervals); ++i) {
        if (i == 2 && !include_conflict) {
          continue;
        }
        const int64_t begin = intervals[i][0];
        const int64_t end = intervals[i][1];
        auto row = MakeMemoryRow(
            IREE_SV("vector.load"), 43, IREE_SV("load"), IREE_SV("test.load"),
            begin, (end - begin) / 4, end - begin, 0, 4, 4,
            MakeExactSymbolicInterval(begin, begin + 64, end, end + 64,
                                      end - begin, begin / 4, end / 4));
        IREE_ASSERT_OK(batch.Append(&row));
      }
      IREE_ASSERT_OK(batch.Finish());
      const auto* summary = GetOnlySourceLowMemoryRootSummary(&report);
      ASSERT_NE(summary, nullptr);
      EXPECT_EQ(summary->interval_envelope.exact_symbolic_packet_count,
                include_conflict ? 1u : 2u);
      EXPECT_EQ(summary->interval_envelope.unique_byte_count,
                include_conflict ? 4u : 8u);
      loom_target_compile_report_deinitialize(&report);
    }
  }
}

TEST(CompileReportMemoryTest, OrdersDuplicatesBeforeOrAfterConflictingPackets) {
  // The same expression can have different range facts at different accesses.
  // Its first duplicate still wins before a later conflicting packet, even
  // when coordinate overlap alone cannot find that duplicate.
  for (bool conflict_first : {false, true}) {
    SCOPED_TRACE(conflict_first);
    loom_target_compile_report_t report;
    loom_target_compile_report_initialize(&report, iree_allocator_system());
    testing::MemoryReportBatch batch(&report);
    const loom_target_compile_report_memory_interval_t intervals[] = {
        MakeExactSymbolicInterval(0, 12, 4, 16, 4, 10, 11),
        MakeExactSourceInterval(32, 36),
        MakeExactSymbolicInterval(28, 36, 32, 40, 4, 10, 11),
    };
    for (size_t i = 0; i < IREE_ARRAYSIZE(intervals); ++i) {
      const auto& interval = intervals[conflict_first && i < 2 ? 1 - i : i];
      auto row =
          MakeMemoryRow(IREE_SV("vector.load"), 43, IREE_SV("load"),
                        IREE_SV("test.load"), 0, 1, 4, 0, 4, 4, interval);
      IREE_ASSERT_OK(batch.Append(&row));
    }
    IREE_ASSERT_OK(batch.Finish());
    const auto* summary = GetOnlySourceLowMemoryRootSummary(&report);
    ASSERT_NE(summary, nullptr);
    EXPECT_EQ(summary->interval_envelope.exact_static_packet_count, 1u);
    EXPECT_EQ(summary->interval_envelope.exact_symbolic_packet_count,
              conflict_first ? 1u : 2u);
    EXPECT_EQ(summary->interval_envelope.unique_byte_count, 8u);
    loom_target_compile_report_deinitialize(&report);
  }
}

TEST(CompileReportMemoryTest, CountsWithoutRetainingDisabledDetails) {
  loom_target_compile_report_t report;
  loom_target_compile_report_initialize(&report, iree_allocator_null());
  testing::MemoryReportBatch batch(&report);
  auto row = MakeMemoryRow(IREE_SV("vector.load"), 43, IREE_SV("load"),
                           IREE_SV("test.load"), 0, 1, 4, 0, 0, 4,
                           MakeExactSourceInterval(0, 4));
  IREE_ASSERT_OK(batch.Append(&row));
  IREE_ASSERT_OK(batch.Finish());
  EXPECT_EQ(report.source_low_memory_summary.packet_count, 1u);
  EXPECT_EQ(report.source_low_memory_summary.interval_envelope.packet_count,
            1u);
  EXPECT_EQ(report.source_low_memory_rows.count, 0u);
  EXPECT_EQ(report.source_low_memory_root_summaries.count, 0u);
  loom_target_compile_report_deinitialize(&report);
}

struct MemoryReportAllocator {
  // Successful allocation requests allowed before injecting exhaustion.
  size_t remaining = SIZE_MAX;
  // Live allocations requiring teardown.
  size_t live = 0;
  // Total successful backing allocations; pool reuse makes no requests here.
  size_t requests = 0;
  // Total requested bytes, bounding retained and temporary growth together.
  size_t bytes = 0;
  // Largest request, distinguishing pooled blocks from oversized allocation.
  size_t maximum_request = 0;

  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* params, void** pointer) {
    auto& state = *static_cast<MemoryReportAllocator*>(self);
    if (command != IREE_ALLOCATOR_COMMAND_FREE && !state.remaining) {
      return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
    }
    auto backing = iree_allocator_system();
    IREE_RETURN_IF_ERROR(backing.ctl(backing.self, command, params, pointer));
    if (command == IREE_ALLOCATOR_COMMAND_FREE) {
      --state.live;
    } else {
      --state.remaining;
      ++state.live;
      ++state.requests;
      const auto size =
          static_cast<const iree_allocator_alloc_params_t*>(params)
              ->byte_length;
      state.bytes += size;
      state.maximum_request = std::max(state.maximum_request, size);
    }
    return iree_ok_status();
  }
};

TEST(CompileReportMemoryTest, BoundsScratchAndReusesPooledStorage) {
  MemoryReportAllocator storage;
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(
      64 * 1024, {&storage, MemoryReportAllocator::Control}, &pool);
  size_t warm_requests = 0;
  for (size_t iteration = 0; iteration < 2; ++iteration) {
    iree_arena_allocator_t result_arena;
    iree_arena_initialize(&pool, &result_arena);
    iree_arena_allocator_t scratch;
    iree_arena_initialize(&pool, &scratch);
    loom_target_compile_report_t report;
    loom_target_compile_report_initialize(&report,
                                          iree_arena_allocator(&result_arena));
    loom_target_compile_report_memory_batch_t batch;
    loom_target_compile_report_memory_batch_initialize(&report, &scratch,
                                                       &batch);
    constexpr size_t kPacketCount = 8192;
    for (size_t i = 0; i < kPacketCount; ++i) {
      auto row = MakeMemoryRow(IREE_SV("vector.load"), 43, IREE_SV("load"),
                               IREE_SV("test.load"), 8 * i, 1, 4, 0, 0, 4,
                               MakeExactSourceInterval(8 * i, 8 * i + 4));
      IREE_ASSERT_OK(
          loom_target_compile_report_memory_batch_append(&batch, &row));
    }
    IREE_ASSERT_OK(loom_target_compile_report_memory_batch_finish(&batch));
    iree_arena_deinitialize(&scratch);
    const auto* summary = GetOnlySourceLowMemoryRootSummary(&report);
    ASSERT_NE(summary, nullptr);
    EXPECT_EQ(summary->interval_envelope.exact_static_packet_count,
              kPacketCount);
    EXPECT_EQ(summary->interval_envelope.unique_byte_count, kPacketCount * 4);
    EXPECT_LE(storage.maximum_request, 64 * 1024u);
    EXPECT_LT(storage.bytes, 12 * 1024 * 1024u);
    if (iteration == 0) {
      warm_requests = storage.requests;
    } else {
      EXPECT_EQ(storage.requests, warm_requests);
    }
    loom_target_compile_report_deinitialize(&report);
    iree_arena_deinitialize(&result_arena);
  }
  iree_arena_block_pool_deinitialize(&pool);
  EXPECT_EQ(storage.live, 0u);
}

TEST(CompileReportMemoryTest, ReleasesPartialIndexOnAllocationFailure) {
  bool observed_failure = false;
  bool observed_success = false;
  for (size_t allowance = 0; allowance < 6; ++allowance) {
    SCOPED_TRACE(allowance);
    MemoryReportAllocator storage;
    const iree_allocator_t allocator = {&storage,
                                        MemoryReportAllocator::Control};
    iree_arena_block_pool_t pool;
    iree_arena_block_pool_initialize(64 * 1024, allocator, &pool);
    iree_arena_allocator_t scratch;
    iree_arena_initialize(&pool, &scratch);
    loom_target_compile_report_t report;
    loom_target_compile_report_initialize(&report, allocator);
    loom_target_compile_report_memory_batch_t batch;
    loom_target_compile_report_memory_batch_initialize(&report, &scratch,
                                                       &batch);
    for (size_t i = 0; i < 512; ++i) {
      auto row = MakeMemoryRow(IREE_SV("vector.load"), 43, IREE_SV("load"),
                               IREE_SV("test.load"), 8 * i, 1, 4, 0, 0, 4,
                               MakeExactSourceInterval(8 * i, 8 * i + 4));
      IREE_ASSERT_OK(
          loom_target_compile_report_memory_batch_append(&batch, &row));
    }
    storage.remaining = allowance;
    iree_status_t status =
        loom_target_compile_report_memory_batch_finish(&batch);
    if (iree_status_is_ok(status)) {
      observed_success = true;
    } else {
      observed_failure = true;
      IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
    }
    loom_target_compile_report_deinitialize(&report);
    iree_arena_deinitialize(&scratch);
    iree_arena_block_pool_deinitialize(&pool);
    EXPECT_EQ(storage.live, 0u);
  }
  EXPECT_TRUE(observed_failure);
  EXPECT_TRUE(observed_success);
}

}  // namespace
}  // namespace loom
