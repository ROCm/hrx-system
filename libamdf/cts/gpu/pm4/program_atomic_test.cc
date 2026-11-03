// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/gpu/pm4/dispatch_fixture.h"
#include "libamdf/cts/gpu/util/program_atomic_fixture.h"

namespace {

using Pm4ProgramAtomicTest = GpuProgramAtomicTest<Pm4DispatchTest>;

TEST_P(Pm4ProgramAtomicTest, ContendedShaderUpdatesRespectSelectedScope) {
  Pm4ComputeProgram program = {0,
                               kernel_->program.resource1,
                               kernel_->program.resource2,
                               kernel_->program.resource3,
                               kernel_->group_segment_byte_length,
                               kernel_->wavefront_size,
                               {kernel_->workgroup_size(), 1, 1}};
  ASSERT_NO_FATAL_FAILURE(PrepareProgram(kernel_->executable,
                                         kernel_->entry_byte_offset, &program,
                                         "program_atomic"));
  GpuMemory* completion = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096, &completion,
      AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_QUEUE_STORAGE));
  std::memset(completion->host.pointer, 0, completion->info.byte_length);
  GpuCommandQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&queue, 65536));
  Pm4CommandWriter commands(queue->words().data(), *pm4_profile_);
  for (uint32_t epoch = 0; epoch < kEpochCount; ++epoch) {
    SCOPED_TRACE(epoch);
    InitializeEpoch(epoch);
    commands.SystemBarrier();
    commands.BindCompute(program, arguments_->device_address,
                         family_.format_features);
    commands.Dispatch(program, kProducerCount, 1, 1);
    if (UsesSystemScope()) {
      commands.PadToEightWords();
      ASSERT_NO_FATAL_FAILURE(
          queue->Publish(api_, gpu_api_, commands.word_count()));
      const auto observed = WaitForRecords();
      EXPECT_TRUE(observed);
      if (observed) {
        CheckRecords();
      }
      // The shader publication oracle precedes any native join. Retire this
      // publication before reusing the fixture's single pending KERNEL slot.
      ASSERT_NO_FATAL_FAILURE(queue->WaitRetired(api_));
    }
    // System scope joins only after the shader-owned publication oracle,
    // including on failure. Device scope submits the dispatch and join
    // together.
    commands.SystemBarrier();
    commands.WriteData32(completion->device_address, epoch + 1);
    commands.PadToEightWords();
    ASSERT_NO_FATAL_FAILURE(
        queue->Publish(api_, gpu_api_, commands.word_count()));
    ASSERT_TRUE(GpuWaitEqual<uint32_t>(
        reinterpret_cast<uintptr_t>(completion->host.pointer), epoch + 1));
    ASSERT_NO_FATAL_FAILURE(queue->WaitRetired(api_));
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
    ScopeAndWidth, Pm4ProgramAtomicTest,
    ::testing::Values(kernels::program_atomic::Variant{32, false},
                      kernels::program_atomic::Variant{64, false},
                      kernels::program_atomic::Variant{32, true},
                      kernels::program_atomic::Variant{64, true}));

}  // namespace
