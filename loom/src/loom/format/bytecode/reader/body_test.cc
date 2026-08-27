// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader/body.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

struct CapturedDiagnostic {
  // Structured error emitted while decoding the region summary.
  const loom_error_def_t* error;
  // Absolute source range associated with the diagnostic.
  loom_source_range_t origin;
};

static iree_status_t CaptureDiagnostic(void* user_data,
                                       const loom_diagnostic_t* diagnostic) {
  CapturedDiagnostic* captured = static_cast<CapturedDiagnostic*>(user_data);
  captured->error = diagnostic->error;
  captured->origin = diagnostic->origin;
  return iree_ok_status();
}

class BytecodeRegionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    loom_bytecode_reader_decoder_initialize(
        loom_diagnostic_sink_t{CaptureDiagnostic, &captured_},
        IREE_SV("body_test.loombc"), &error_count_, &decoder_);
  }

  iree_status_t ReadSummary(const uint8_t* data, iree_host_size_t length,
                            loom_bytecode_region_summary_t* out_summary) {
    return loom_bytecode_region_summary_read(
        &decoder_, IREE_SV("@test"), iree_make_const_byte_span(data, length),
        /*payload_absolute_offset=*/4096, out_summary);
  }

  // Bounded decoder under test.
  loom_bytecode_reader_decoder_t decoder_ = {};
  // Number of malformed-input diagnostics accepted by the sink.
  uint32_t error_count_ = 0;
  // Last structured diagnostic accepted by the sink.
  CapturedDiagnostic captured_ = {};
};

TEST_F(BytecodeRegionTest, ReadsExactAllocationSummaryPrefix) {
  const uint8_t data[] = {
      0x03,  // value_count
      0x02,  // region_count
      0x01,  // block_count
      0x04,  // op_count
      0xAA,  // first payload byte
  };
  loom_bytecode_region_summary_t summary;
  IREE_ASSERT_OK(ReadSummary(data, sizeof(data), &summary));

  EXPECT_EQ(summary.value_count, 3u);
  EXPECT_EQ(summary.region_count, 2u);
  EXPECT_EQ(summary.block_count, 1u);
  EXPECT_EQ(summary.op_count, 4u);
  EXPECT_EQ(summary.payload_offset, 4u);
  EXPECT_EQ(error_count_, 0u);
}

TEST_F(BytecodeRegionTest, RejectsSummaryCountLargerThanBoundedRegion) {
  const uint8_t data[] = {
      0x06,  // value_count
      0x00,  // region_count
      0x00,  // block_count
      0x00,  // op_count
  };
  loom_bytecode_region_summary_t summary = {/*.value_count=*/99};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DEFERRED,
                        ReadSummary(data, sizeof(data), &summary));

  EXPECT_EQ(summary.value_count, 0u);
  EXPECT_EQ(summary.payload_offset, 0u);
  EXPECT_EQ(error_count_, 1u);
  ASSERT_NE(captured_.error, nullptr);
  EXPECT_EQ(captured_.error->domain, LOOM_ERROR_DOMAIN_BYTECODE);
  EXPECT_EQ(captured_.error->code, 16u);
  EXPECT_EQ(captured_.origin.start, 4096u);
}

TEST_F(BytecodeRegionTest, RejectsTruncatedSummaryPrefix) {
  const uint8_t data[] = {
      0x00,  // value_count
      0x00,  // region_count
      0x00,  // block_count
  };
  loom_bytecode_region_summary_t summary;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DEFERRED,
                        ReadSummary(data, sizeof(data), &summary));

  EXPECT_EQ(error_count_, 1u);
  ASSERT_NE(captured_.error, nullptr);
  EXPECT_EQ(captured_.error->domain, LOOM_ERROR_DOMAIN_BYTECODE);
  EXPECT_EQ(captured_.error->code, 8u);
  EXPECT_EQ(captured_.origin.start, 4099u);
}

}  // namespace
}  // namespace loom
