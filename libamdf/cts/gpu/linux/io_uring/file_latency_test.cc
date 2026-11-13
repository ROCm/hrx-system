// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "gtest/gtest-spi.h"
#include "libamdf/cts/gpu/linux/io_uring/file_latency_fixture.h"

namespace {

TEST(FileClockTest, SameShaderIntervalAllowsLowWordWrap) {
  ASSERT_NO_FATAL_FAILURE(CheckTickInterval(0xfffffff0u, 7, 16, 8));
  ASSERT_NO_FATAL_FAILURE(CheckTickInterval(1, 7, 0xffffffffu, 7));
}

TEST(FileClockTest, BackwardsIntervalRemainsAFailure) {
  EXPECT_FATAL_FAILURE(CheckTickInterval(16, 8, 15, 8),
                       "shader reference clock moved backwards");
}

TEST(FileClockTest, AmbiguousCompactIntervalRemainsAFailure) {
  EXPECT_FATAL_FAILURE(
      CheckTickInterval(1, 7, 0, 8),
      "shader interval exceeds compact request timestamp width");
  EXPECT_FATAL_FAILURE(
      CheckTickInterval(1, 7, 2, 9),
      "shader interval exceeds compact request timestamp width");
}

TEST(PollerAccountingTest, KernelWithoutCpuTimePreservesThreadIdentity) {
  std::istringstream stream("SqThread:\t37\nSqThreadCpu:\t5\nUserFiles:\t1\n");
  PollerAccounting accounting;
  ASSERT_NO_FATAL_FAILURE(ReadPollerAccounting(stream, &accounting));
  EXPECT_EQ(accounting.thread, 37);
  EXPECT_EQ(accounting.cpu, 5);
  EXPECT_FALSE(accounting.microseconds.has_value());
  EXPECT_EQ(CounterJson(accounting.microseconds), "null");
}

TEST(PollerAccountingTest, KernelCpuTimeIncludesAValidZero) {
  for (uint64_t microseconds : {0u, 123456u}) {
    std::istringstream stream("SqThread:\t37\nSqThreadCpu:\t5\nSqTotalTime:\t" +
                              std::to_string(microseconds) + "\n");
    PollerAccounting accounting;
    ASSERT_NO_FATAL_FAILURE(ReadPollerAccounting(stream, &accounting));
    ASSERT_TRUE(accounting.microseconds.has_value());
    EXPECT_EQ(*accounting.microseconds, microseconds);
    EXPECT_EQ(CounterJson(accounting.microseconds),
              std::to_string(microseconds));
  }
}

TEST(PollerAccountingTest, UnobservedOwnerDoesNotReportZeroCpuTime) {
  std::istringstream stream(
      "SqThread:\t-1\nSqThreadCpu:\t-1\nSqTotalTime:\t0\n");
  PollerAccounting accounting;
  ASSERT_NO_FATAL_FAILURE(ReadPollerAccounting(stream, &accounting));
  EXPECT_EQ(accounting.thread, -1);
  EXPECT_EQ(accounting.cpu, -1);
  EXPECT_FALSE(accounting.microseconds.has_value());
}

TEST(PollerAccountingTest, MalformedAccountingRemainsAFailure) {
  EXPECT_FATAL_FAILURE(
      {
        std::istringstream stream("SqThread:\t37\nSqTotalTime:\tinvalid\n");
        PollerAccounting accounting;
        ReadPollerAccounting(stream, &accounting);
      },
      "value >> microseconds");
}

TEST(PollerAccountingTest, FailedReadRemainsAFailure) {
  EXPECT_FATAL_FAILURE(
      {
        std::istringstream stream;
        stream.setstate(std::ios::badbit);
        PollerAccounting accounting;
        ReadPollerAccounting(stream, &accounting);
      },
      "stream.bad()");
}

TEST_P(GpuFileLatencyTest, DependentLookup4KiB) { Run({4096, 1, 1, 0, 1024}); }
TEST_P(GpuFileLatencyTest, FourIndependentLookups4KiB) {
  Run({4096, 4, 1, 0, 256});
}
TEST_P(GpuFileLatencyTest, BlockRoundTrip64KiB) { Run({65536, 1, 3, 0, 256}); }
TEST_P(GpuFileLatencyTest, SparseLookup2ms) { Run({4096, 1, 1, 2000, 64}); }
TEST_P(GpuFileLatencyTest, FailedInputDrainsAcceptedReads) {
  Run({4096, 4, 1, 0, 8}, InputFault::kAbsentFile);
}
TEST_P(GpuFileLatencyTest, ShortInputThenEofDrainsWithoutConsumption) {
  // The short completion and its EOF retry must both satisfy direct alignment.
  const uint32_t byte_length =
      GetParam() == FileMode::kDirect
          ? 2 * static_cast<uint32_t>(page_byte_length_)
          : 4096;
  Run({byte_length, 4, 1, 0, 8}, InputFault::kShortFile);
}

INSTANTIATE_TEST_SUITE_P(FileModes, GpuFileLatencyTest,
                         ::testing::Values(FileMode::kBuffered,
                                           FileMode::kDirect),
                         [](const auto& info) {
                           return info.param == FileMode::kDirect ? "Direct"
                                                                  : "Buffered";
                         });

TEST_P(GpuFileDemandTest, LookupBursts32) {
  RunDemand({"lookup32", 4096, 32, 32, 256, 1000, 0, 1, false});
}
TEST_P(GpuFileDemandTest, LookupBacklog64On8Credits) {
  RunDemand({"lookup_backlog", 4096, 64, 8, 64, 1000, 200, 1, false});
}
TEST_P(GpuFileDemandTest, StaggeredLookupWithGpuWork) {
  RunDemand({"lookup_staggered", 4096, 32, 32, 128, 40, 0, 1, true});
}
TEST_P(GpuFileDemandTest, SharedExpertTiles64KiB) {
  RunDemand({"expert_tiles", 65536, 32, 8, 8, 1000, 200, 1, false});
}
TEST_P(GpuFileDemandTest, ExpertTileCreditPressureWithGpuWork) {
  RunDemand({"expert_pressure", 65536, 32, 4, 16, 1000, 200, 1, true});
}
TEST_P(GpuFileDemandTest, ScatteredKvBlocks16KiB) {
  RunDemand({"kv_blocks", 16384, 32, 8, 32, 1000, 200, 3, false});
}
TEST_P(GpuFileDemandTest, InvalidFileDrainsConcurrentDemands) {
  RunDemand({"invalid", 4096, 32, 8, 32, 1000, 0, 1, false},
            InputFault::kAbsentFile);
}
TEST_P(GpuFileDemandTest, PartialInputDrainsHeldDemands) {
  // A complete first page is still an incomplete logical two-page direct input.
  const uint32_t byte_length =
      GetParam() == FileMode::kDirect
          ? 2 * static_cast<uint32_t>(page_byte_length_)
          : 4096;
  RunDemand({"partial", byte_length, 32, 1, 1, 1000, 200, 1, false},
            InputFault::kShortFile);
}

INSTANTIATE_TEST_SUITE_P(FileModes, GpuFileDemandTest,
                         ::testing::Values(FileMode::kBuffered,
                                           FileMode::kDirect),
                         [](const auto& info) {
                           return info.param == FileMode::kDirect ? "Direct"
                                                                  : "Buffered";
                         });

}  // namespace
