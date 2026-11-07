// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/gpu/linux/io_uring/file_io_fixture.h"

#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <string>
#include <thread>

#include "util/wait.h"

void GpuFileIoFixture::SetUp() {
  Pm4DispatchTest::SetUp();
  if (HasFatalFailure() || IsSkipped()) {
    return;
  }
  InitializeFileIo(api_, system_scope_, device_, features_);
}

void GpuFileIoFixture::TearDown() {
  // A failed queue release cannot authorize unregistering its caller pages.
  ASSERT_NO_FATAL_FAILURE(Pm4DispatchTest::TearDown());
  ReleaseFileIo();
}

void GpuFileIoFixture::Execute(const kernels::Kernel& kernel,
                               GpuMemory* arguments, GpuMemory* completion,
                               const char* property_prefix) {
  Pm4ComputeProgram program = {
      0,
      kernel.program.resource1,
      kernel.program.resource2,
      kernel.program.resource3,
      kernel.group_segment_byte_length,
      kernel.wavefront_size,
      {1, 1, 1},
  };
  ASSERT_NO_FATAL_FAILURE(PrepareProgram(
      kernel.executable, kernel.entry_byte_offset, &program, property_prefix));
  GpuCommandQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&queue));
  Pm4CommandWriter commands(queue->words().data(), *pm4_profile_);
  commands.SystemBarrier();
  commands.BindCompute(program, arguments->device_address,
                       family_.format_features);
  commands.Dispatch(program, 1, 1, 1);
  // Only the host needs completion. Publish the shader's stores at
  // bottom-of-pipe without blocking the command processor on a compute join.
  commands.ReleaseSystem32(completion->device_address, 1);
  commands.PadToEightWords();

  // Observe the actual idle transition instead of assuming a delay sleeps
  // the poller. The shader's first request then needs the ordinary wake path.
  ASSERT_TRUE(amdf::cts::WaitUntil("SQPOLL idle", [&] {
    return !(ring_->parameters.flags & IORING_SETUP_SQPOLL) ||
           (GpuLoadAcquire<uint32_t>(RingWord(ring_->parameters.sq_off.flags)) &
            IORING_SQ_NEED_WAKEUP);
  }));
  ASSERT_NO_FATAL_FAILURE(
      queue->Publish(api_, gpu_api_, commands.word_count()));
  uint64_t wake_count = 0;
  uint64_t enter_count = 0;
  uint64_t wait_count = 0;
  while (GpuLoadAcquire<uint32_t>(
             reinterpret_cast<uintptr_t>(completion->host.pointer)) != 1) {
    // This host services the live GPU owner; returning before it completes
    // would abandon the very I/O that permits it to retire.
    std::this_thread::sleep_for(
        std::chrono::nanoseconds(amdf::cts::kWaitPollIntervalNs));
    RelayFileIo();
    if (!(ring_->parameters.flags & IORING_SETUP_SQPOLL)) {
      if (ring_->path == FileIoPath::kDeviceWait) {
        ASSERT_NO_FATAL_FAILURE(ServiceDeviceIo(&enter_count, &wait_count));
      } else {
        ASSERT_NO_FATAL_FAILURE(ServiceHostIo(&enter_count));
      }
      continue;
    }
    // Waking an idle kernel owner is separate from the optional control-record
    // relay. Neither path reads or modifies application payloads.
    if ((GpuLoadAcquire<uint32_t>(RingWord(ring_->parameters.sq_off.flags)) &
         IORING_SQ_NEED_WAKEUP) &&
        GpuLoadAcquire<uint32_t>(RingWord(ring_->parameters.sq_off.head)) !=
            GpuLoadAcquire<uint32_t>(RingWord(ring_->parameters.sq_off.tail))) {
      const long result = syscall(__NR_io_uring_enter, ring_->file, 0, 0,
                                  IORING_ENTER_SQ_WAKEUP, nullptr, 0);
      if (result < 0 && errno == EINTR) {
        continue;
      }
      ASSERT_EQ(result, 0) << "wake SQPOLL: " << std::strerror(errno);
      ++wake_count;
    }
  }
  ASSERT_NO_FATAL_FAILURE(queue->WaitRetired(api_));
  if (device_ring_memory_ != ring_->memory) {
    const uintptr_t device_control =
        reinterpret_cast<uintptr_t>(device_ring_memory_->host.pointer) +
        ring_->control_offset;
    for (const uint32_t offset :
         {ring_->parameters.sq_off.head, ring_->parameters.sq_off.tail,
          ring_->parameters.cq_off.head, ring_->parameters.cq_off.tail}) {
      EXPECT_EQ(GpuLoadAcquire<uint32_t>(device_control + offset),
                GpuLoadAcquire<uint32_t>(RingWord(offset)));
    }
    EXPECT_EQ(std::memcmp(device_ring_memory_->host.pointer,
                          ring_->memory->host.pointer,
                          ring_->parameters.sq_entries * sizeof(io_uring_sqe)),
              0);
    EXPECT_EQ(std::memcmp(reinterpret_cast<const void*>(
                              device_control + ring_->parameters.cq_off.cqes),
                          reinterpret_cast<const void*>(
                              RingWord(ring_->parameters.cq_off.cqes)),
                          ring_->parameters.cq_entries * sizeof(io_uring_cqe)),
              0);
  }
  RecordProperty("io_idle_wake_calls", std::to_string(wake_count));
  RecordProperty("io_submit_calls", std::to_string(enter_count));
  RecordProperty("io_wait_calls", std::to_string(wait_count));
}
