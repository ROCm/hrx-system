// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/gpu/linux/io_uring/file_latency_fixture.h"

namespace {

TEST_P(GpuFileLatencyTest, DependentLookup4KiB) { Run({4096, 1, 1, 0, 1024}); }
TEST_P(GpuFileLatencyTest, FourIndependentLookups4KiB) {
  Run({4096, 4, 1, 0, 256});
}
TEST_P(GpuFileLatencyTest, BlockRoundTrip64KiB) { Run({65536, 1, 3, 0, 256}); }
TEST_P(GpuFileLatencyTest, FourBlockRoundTrips4MiB) {
  Run({4194304, 4, 3, 0, 8});
}
TEST_P(GpuFileLatencyTest, SparseLookup200us) { Run({4096, 1, 1, 200, 128}); }
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
TEST_P(GpuFileDemandTest, LookupBursts64) {
  RunDemand({"lookup64", 4096, 64, 64, 256, 1000, 0, 1, false});
}
TEST_P(GpuFileDemandTest, LookupBursts128) {
  RunDemand({"lookup128", 4096, 128, 128, 256, 1000, 0, 1, false});
}
TEST_P(GpuFileDemandTest, LookupBursts256) {
  RunDemand({"lookup256", 4096, 256, 256, 256, 1000, 0, 1, false});
}
TEST_P(GpuFileDemandTest, LookupBacklog256On32Credits) {
  RunDemand({"lookup_backlog", 4096, 256, 32, 256, 1000, 200, 1, false});
}
TEST_P(GpuFileDemandTest, StaggeredLookupWithGpuWork) {
  RunDemand({"lookup_staggered", 4096, 32, 64, 256, 40, 0, 1, true});
}
TEST_P(GpuFileDemandTest, SharedExpertTiles256KiB) {
  RunDemand({"expert_tiles", 262144, 256, 32, 16, 1000, 2000, 1, false});
}
TEST_P(GpuFileDemandTest, ExpertTileCreditPressureWithGpuWork) {
  RunDemand({"expert_pressure", 262144, 256, 8, 64, 1000, 200, 1, true});
}
TEST_P(GpuFileDemandTest, ScatteredKvBlocks64KiB) {
  RunDemand({"kv_blocks", 65536, 256, 64, 256, 1000, 200, 3, false});
}
TEST_P(GpuFileDemandTest, InvalidFileDrainsConcurrentDemands) {
  RunDemand({"invalid", 4096, 256, 64, 256, 1000, 0, 1, false},
            InputFault::kAbsentFile);
}
TEST_P(GpuFileDemandTest, PartialInputDrainsHeldDemands) {
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
