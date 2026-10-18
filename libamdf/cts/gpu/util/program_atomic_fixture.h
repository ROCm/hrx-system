// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_CTS_GPU_UTIL_PROGRAM_ATOMIC_FIXTURE_H_
#define AMDF_CTS_GPU_UTIL_PROGRAM_ATOMIC_FIXTURE_H_

#include <array>
#include <cstring>

#include "gtest/gtest.h"
#include "libamdf/cts/gpu/kernels/program_atomic.h"
#include "libamdf/cts/gpu/kernels/program_atomic_kernels.h"
#include "libamdf/cts/gpu/util/memory.h"
#include "libamdf/cts/gpu/util/user_queue.h"
#include "libamdf/cts/util/wait.h"

// Both engines use the same contended shader and host oracle. Each engine owns
// its native dispatch and execution join; this fixture never supplies either.
template <typename QueueFixture>
class GpuProgramAtomicTest : public QueueFixture,
                             public ::testing::WithParamInterface<uint32_t> {
 protected:
  using Control = kernels::program_atomic::Control;
  using Record = kernels::program_atomic::Record;
  using Arguments = kernels::program_atomic::Arguments;
  static constexpr uint32_t kProducerCount =
      kernels::program_atomic::kProducerCount;
  static constexpr uint32_t kEpochCount = 8;
  static constexpr uint64_t kGuard = UINT64_C(0xfedcba9876543210);

  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(QueueFixture::SetUp());
    if (this->IsSkipped()) {
      return;
    }
    kernel_ = kernels::program_atomic::kKernels.Find(this->gpu_endpoint_info_);
    ASSERT_NE(kernel_, nullptr) << "missing compiled program-atomic kernel";
    this->RecordProperty("program_atomic_target", kernel_->target);
    this->RecordProperty("program_atomic_width", this->GetParam());
    ASSERT_EQ(kernel_->workgroup_size(),
              kernels::program_atomic::kWorkgroupSize);
    ASSERT_EQ(kernel_->private_segment_byte_length, 0u);
    ASSERT_EQ(kernel_->group_segment_byte_length, 0u);
    // Live control cells need queue-storage policy, not just host coherence.
    // Arguments are immutable during execution and retain payload policy.
    ASSERT_NO_FATAL_FAILURE(this->CreateMemory(
        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096, &control_,
        AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_QUEUE_STORAGE));
    ASSERT_NO_FATAL_FAILURE(this->CreateMemory(
        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
        (sizeof(Record) * (kProducerCount + 2) + 4095) & ~UINT64_C(4095),
        &records_,
        AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_QUEUE_STORAGE));
    ASSERT_NO_FATAL_FAILURE(this->CreateMemory(AMDF_MEMORY_ACCESS_READ, 4096,
                                               &arguments_,
                                               AMDF_MEMORY_FLAG_HOST_COHERENT));
    ASSERT_EQ(control_->info.memory_class, AMDF_MEMORY_CLASS_SYSTEM);
    ASSERT_NE(control_->access_info.flags & AMDF_MEMORY_FLAG_HOST_COHERENT, 0u);
    ASSERT_NE(records_->access_info.flags & AMDF_MEMORY_FLAG_HOST_COHERENT, 0u);
  }

  // Called only before dispatch or after the preceding native execution join.
  void InitializeEpoch(uint32_t epoch) {
    const bool narrow = this->GetParam() == 32;
    initial_ = narrow ? UINT64_C(0x9abcdef0) + (uint64_t{epoch} << 16)
                      : UINT64_C(0x1fffffff0) + (uint64_t{epoch} << 32);
    counter_ = narrow ? UINT64_C(0xfffffff0)
                      : UINT64_C(0x2fffffff0) + (uint64_t{epoch} << 32);
    tail_ = narrow ? UINT64_C(0xa5000000) + epoch
                   : UINT64_C(0x12345678a5000000) + epoch;
    generation_ = narrow ? uint64_t{epoch} + 1 : UINT64_C(0x100000000) + epoch;
    const uint64_t upper_guard =
        narrow ? kGuard & UINT64_C(0xffffffff00000000) : 0;
    expected_control_ = {kGuard, upper_guard | initial_, upper_guard | counter_,
                         upper_guard | tail_, kGuard};
    std::memcpy(control_->host.pointer, &expected_control_, sizeof(Control));
    std::array<Record, kProducerCount + 2> initial_records;
    initial_records.fill({kGuard, kGuard, kGuard, kGuard, kGuard});
    std::memcpy(records_->host.pointer, initial_records.data(),
                sizeof(initial_records));
    // This is the host-owned protocol cell. The CPU never performs an RMW on
    // either shared shader-owned reservation or counter cell.
    const uint64_t tail_address =
        reinterpret_cast<uintptr_t>(control_->host.pointer) +
        offsetof(Control, tail);
    if (narrow) {
      GpuStoreRelease<uint32_t>(tail_address, static_cast<uint32_t>(tail_));
    } else {
      GpuStoreRelease<uint64_t>(tail_address, tail_);
    }
    const Arguments arguments = {
        control_->device_address,
        records_->device_address + sizeof(Record),
        initial_,
        initial_ ^ (narrow ? UINT64_C(1) << 31 : UINT64_C(1) << 40),
        generation_,
        this->GetParam()};
    std::memcpy(arguments_->host.pointer, &arguments, sizeof(arguments));
  }

  // No native completion or command-processor release participates in this
  // observation. All producer records must be acquired before reading payloads.
  ::testing::AssertionResult WaitForRecords() {
    uint32_t observed_count = 0;
    const auto result = amdf::cts::WaitUntil("shader atomic records", [&] {
      observed_count = 0;
      for (uint32_t producer = 0; producer < kProducerCount; ++producer) {
        const uint64_t address =
            reinterpret_cast<uintptr_t>(records_->host.pointer) +
            sizeof(Record) * (producer + 1) + offsetof(Record, ready);
        const uint64_t observed = this->GetParam() == 32
                                      ? GpuLoadAcquire<uint32_t>(address)
                                      : GpuLoadAcquire<uint64_t>(address);
        observed_count += observed == generation_;
      }
      return observed_count == kProducerCount;
    });
    if (!result) {
      return ::testing::AssertionFailure()
             << result.message() << ": " << observed_count << "/"
             << kProducerCount << " records at generation " << generation_;
    }
    return ::testing::AssertionSuccess();
  }

  void CheckRecords() {
    const bool narrow = this->GetParam() == 32;
    const uint64_t mask = narrow ? UINT64_C(0xffffffff) : UINT64_MAX;
    const auto& control = *static_cast<const Control*>(control_->host.pointer);
    const uint64_t winner = control.reservation & mask;
    ASSERT_GT(winner, initial_);
    ASSERT_LE(winner - initial_, kProducerCount);
    expected_control_.reservation =
        (expected_control_.reservation & ~mask) | winner;
    expected_control_.counter = (expected_control_.counter & ~mask) |
                                ((counter_ + kProducerCount) & mask);
    EXPECT_EQ(std::memcmp(&control, &expected_control_, sizeof(Control)), 0);
    const auto* records = static_cast<const Record*>(records_->host.pointer);
    const Record guard = {kGuard, kGuard, kGuard, kGuard, kGuard};
    EXPECT_EQ(std::memcmp(&records[0], &guard, sizeof(Record)), 0);
    EXPECT_EQ(std::memcmp(&records[kProducerCount + 1], &guard, sizeof(Record)),
              0);
    uint32_t successful_count = 0;
    for (uint32_t producer = 0; producer < kProducerCount; ++producer) {
      SCOPED_TRACE(producer);
      const auto& record = records[producer + 1];
      const uint64_t proposal = initial_ + producer + 1;
      const bool succeeded = record.contended_value == initial_;
      successful_count += succeeded;
      EXPECT_TRUE(record.failed_value == initial_ ||
                  record.failed_value == winner);
      EXPECT_EQ(record.contended_value, proposal == winner ? initial_ : winner);
      EXPECT_EQ(record.tail, tail_);
      EXPECT_EQ(record.payload, proposal ^ tail_);
      EXPECT_EQ(record.ready, (kGuard & ~mask) | generation_);
    }
    EXPECT_EQ(successful_count, 1u);
  }

  // Immutable selected compiler product, borrowed from the embedded kernel set.
  const kernels::Kernel* kernel_ = nullptr;
  // Queue-fixture-owned control cells, retained through checked queue teardown.
  GpuMemory* control_ = nullptr;
  // Queue-fixture-owned guarded records, one per shader producer.
  GpuMemory* records_ = nullptr;
  // Queue-fixture-owned kernel arguments, replaced only after execution joins.
  GpuMemory* arguments_ = nullptr;

 private:
  // Expected complete control object, including untouched upper guard words.
  Control expected_control_ = {};
  // Initial reservation expected by every producer in the current epoch.
  uint64_t initial_ = 0;
  // Initial ADD counter for the current epoch.
  uint64_t counter_ = 0;
  // Host publication expected by each shader acquire.
  uint64_t tail_ = 0;
  // Current record generation, acquired independently for every producer.
  uint64_t generation_ = 0;
};

#endif  // AMDF_CTS_GPU_UTIL_PROGRAM_ATOMIC_FIXTURE_H_
