// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/gpu/aql/dispatch_fixture.h"
#include "libamdf/cts/gpu/util/program_atomic_fixture.h"

namespace {

using AqlProgramAtomicTest = GpuProgramAtomicTest<AqlDispatchTest>;

TEST_P(AqlProgramAtomicTest, ContendedShaderUpdatesRespectSelectedScope) {
  GpuMemory* completion = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096, &completion,
      AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_QUEUE_STORAGE));
  auto& signal = *static_cast<aql::Signal*>(completion->host.pointer);
  signal = aql::UserSignal();
  GpuUserQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&queue));
  uint64_t index = 0;
  uint64_t descriptor_address = 0;
  ASSERT_NO_FATAL_FAILURE(PublishKernel(*queue, *kernel_, "program_atomic",
                                        &index, &descriptor_address));
  for (uint32_t epoch = 0; epoch < kEpochCount; ++epoch) {
    SCOPED_TRACE(epoch);
    InitializeEpoch(epoch);
    signal.value = 1;
    const auto packet =
        aql::Dispatch(aql::HeaderBarrier::kDisabled,
                      {1,
                       {kernels::program_atomic::kWorkgroupSize, 1, 1},
                       {kProducerCount, 1, 1}},
                      0, 0, descriptor_address, arguments_->device_address,
                      completion->device_address,
                      {aql::FenceScope::kSystem,
                       UsesSystemScope() ? aql::FenceScope::kNone
                                         : aql::FenceScope::kSystem});
    GpuStoreRelease(queue->host.write_index_address, index + 1);
    ASSERT_TRUE(aql::Publish(*queue, index++, packet));
    if (UsesSystemScope()) {
      const auto observed = WaitForRecords();
      EXPECT_TRUE(observed);
      if (observed) {
        CheckRecords();
      }
    }
    // System scope proves shader publication independently of packet release.
    // Device scope needs the packet's release and join before CPU observation.
    // Both retain backing until native completion and consumption join.
    ASSERT_NO_FATAL_FAILURE(
        WaitCompletionAndConsumption(*queue, signal, index));
    if (!UsesSystemScope()) {
      CheckRecords();
    }
    if (HasFailure()) {
      return;
    }
  }
  RecordProperty("program_atomic_epochs", kEpochCount);
  RecordProperty("program_atomic_producers", kProducerCount);
}

INSTANTIATE_TEST_SUITE_P(
    ScopeAndWidth, AqlProgramAtomicTest,
    ::testing::Values(kernels::program_atomic::Variant{32, false},
                      kernels::program_atomic::Variant{64, false},
                      kernels::program_atomic::Variant{32, true},
                      kernels::program_atomic::Variant{64, true}));

}  // namespace
