// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

#include "libamdf/cts/gpu/aql/queue/queue_fixture.h"
#include "libamdf/cts/gpu/pm4/encoding/memory_commands.h"

namespace {

class AqlAtomicTest : public AqlQueueTest,
                      public ::testing::WithParamInterface<size_t> {
 protected:
  AqlAtomicTest() : AqlQueueTest(AMDF_QUEUE_ROLE_ATOMIC) {}

  template <typename T>
  void RunHostHandoffs() {
    constexpr uint32_t kByteLength = 4096;
    constexpr uint32_t kEnteredOffset = 64;
    constexpr uint32_t kInputOffset = 128;
    constexpr uint32_t kObservedOffset = 192;
    constexpr uint32_t kAcknowledgedOffset = 256;
    constexpr uint32_t kOutputOffset = kByteLength - sizeof(T);
    constexpr auto kWidth = sizeof(T) == 4 ? pm4::CopyDataWidth::k32Bit
                                           : pm4::CopyDataWidth::k64Bit;
    constexpr auto kOperations =
        AMDF_ATOMIC_OPERATION_WAIT | AMDF_ATOMIC_OPERATION_STORE;
    const auto& atomics = family_.atomic_capabilities;
    const auto operations =
        sizeof(T) == 4 ? atomics.operations_32 : atomics.operations_64;
    const auto native_operations = sizeof(T) == 4
                                       ? atomics.operations_without_dispatch_32
                                       : atomics.operations_without_dispatch_64;
    const auto conditions = sizeof(T) == 4 ? atomics.wait_conditions_32
                                           : atomics.wait_conditions_64;
    if ((operations & native_operations & kOperations) != kOperations ||
        (atomics.operations_without_dispatch_32 & kOperations) != kOperations ||
        (atomics.wait_conditions_32 & AMDF_ATOMIC_WAIT_CONDITION_EQUAL) == 0) {
      GTEST_SKIP() << "AQL carrier lacks native WAIT/STORE for the used widths";
    }
    const amdf_memory_device_access_t attachment = {
        device_,
        {
            .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
            .flags = AMDF_MEMORY_FLAG_HOST_COHERENT |
                     AMDF_MEMORY_FLAG_DEVICE_ADDRESS |
                     AMDF_MEMORY_FLAG_QUEUE_STORAGE,
        }};
    amdf_memory_create_info_t creation = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
        .structure_size = sizeof(creation),
        .memory_profile_ordinal = FindMemoryProfileOrdinal(
            system_scope_,
            AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
            AMDF_MEMORY_FLAG_HOST_VISIBLE, attachment.requirements),
        .access_count = 1,
        .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
        .byte_length = kByteLength,
        .minimum_alignment = kByteLength,
        .accesses = &attachment,
    };
    ASSERT_NE(creation.memory_profile_ordinal,
              AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
    amdf_memory_profile_t profile = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
        .structure_size = sizeof(profile),
    };
    amdf_memory_access_capabilities_t capabilities = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
        .structure_size = sizeof(capabilities),
    };
    ASSERT_EQ(
        QueryMemoryProfile(system_scope_, creation.memory_profile_ordinal,
                           attachment.requirements, &profile, &capabilities),
        AMDF_STATUS_OK);
    const auto memory_operations = sizeof(T) == 4
                                       ? capabilities.atomic_operations_32
                                       : capabilities.atomic_operations_64;
    if ((memory_operations & capabilities.atomic_operations_32 & kOperations) !=
        kOperations) {
      GTEST_SKIP()
          << "control backing lacks CPU/GPU WAIT/STORE for used widths";
    }
    std::array<amdf_memory_pair_info_t, 2> prospective = {};
    for (size_t direction = 0; direction < prospective.size(); ++direction) {
      amdf_memory_profile_site_t host = {.kind = AMDF_MEMORY_SITE_KIND_HOST};
      host.value.host_access =
          AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
      amdf_memory_profile_site_t gpu = {.kind = AMDF_MEMORY_SITE_KIND_DEVICE};
      gpu.value.device.queue_family_ordinal = family_.ordinal;
      amdf_memory_profile_pair_query_t query = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE_PAIR_QUERY,
          .structure_size = sizeof(query),
          .memory_profile_ordinal = creation.memory_profile_ordinal,
          .access_count = creation.access_count,
          .required_flags = creation.required_flags,
          .accesses = creation.accesses,
          .producer = direction == 0 ? host : gpu,
          .consumer = direction == 0 ? gpu : host,
      };
      auto& pair = prospective[direction];
      pair.type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO;
      pair.structure_size = sizeof(pair);
      ASSERT_EQ(
          api_->memory_scope_query_pair_info(system_scope_, &query, &pair),
          AMDF_STATUS_OK);
      ASSERT_EQ(pair.atomic_reach.scope_32, AMDF_ATOMIC_SCOPE_SYSTEM);
      if constexpr (sizeof(T) == 8) {
        ASSERT_EQ(pair.atomic_reach.scope_64, AMDF_ATOMIC_SCOPE_SYSTEM);
      }
    }

    GpuMemory* control = nullptr;
    GpuMemory* commands = nullptr;
    GpuMemory* completion = nullptr;
    ASSERT_NO_FATAL_FAILURE(CreateMemory(system_scope_, creation, &control));
    ASSERT_NO_FATAL_FAILURE(
        CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_EXECUTE,
                     kByteLength, &commands));
    ASSERT_NO_FATAL_FAILURE(
        CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                     kByteLength, &completion));
    ASSERT_EQ(control->access_info.atomic_operations_32,
              capabilities.atomic_operations_32);
    ASSERT_EQ(control->access_info.atomic_operations_64,
              capabilities.atomic_operations_64);
    ASSERT_EQ(control->host.cacheability, AMDF_HOST_CACHEABILITY_WRITE_BACK);
    for (size_t direction = 0; direction < prospective.size(); ++direction) {
      const auto host = control->HostSite();
      const auto gpu = control->DeviceSite(family_.ordinal);
      const auto& producer = direction == 0 ? host : gpu;
      const auto& consumer = direction == 0 ? gpu : host;
      amdf_memory_pair_info_t pair = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
          .structure_size = sizeof(pair),
      };
      ASSERT_EQ(api_->memory_query_pair_info(&producer, &consumer, &pair),
                AMDF_STATUS_OK);
      EXPECT_EQ(pair.atomic_reach.scope_32,
                prospective[direction].atomic_reach.scope_32);
      EXPECT_EQ(pair.atomic_reach.scope_64,
                prospective[direction].atomic_reach.scope_64);
      EXPECT_EQ(pair.release.operation,
                prospective[direction].release.operation);
      EXPECT_EQ(pair.acquire.operation,
                prospective[direction].acquire.operation);
    }
    ASSERT_FALSE(HasFailure());
    GpuUserQueue* queue = nullptr;
    ASSERT_NO_FATAL_FAILURE(CreateQueue(&queue));
    uint64_t index = GpuLoadAcquire<uint64_t>(queue->host.write_index_address);
    const uintptr_t host_address =
        reinterpret_cast<uintptr_t>(control->host.pointer);
    const uint64_t gpu_address = control->device_address;
    ASSERT_EQ((host_address + kOutputOffset) % sizeof(T), 0u);
    ASSERT_EQ((gpu_address + kOutputOffset) % sizeof(T), 0u);
    auto& signal = *static_cast<aql::Signal*>(completion->host.pointer);
    constexpr T kHighBit = T{1} << (sizeof(T) * 8 - 1);
    const struct {
      // Native masked comparison performed by the command processor.
      Pm4MemoryComparison comparison;
      // Public capability required for this row.
      amdf_atomic_wait_conditions_t condition;
      // Value that does not satisfy the comparison, including masked-off noise.
      T initial;
      // Value published by the CPU after observing entry into the IB.
      T released;
      // Reference encoded independently of both CPU values.
      T reference;
      // Mask applied to the loaded value by the command processor.
      T mask;
    } cases[] = {
        {Pm4MemoryComparison::kEqual, AMDF_ATOMIC_WAIT_CONDITION_EQUAL, T{0x55},
         T(kHighBit | 0x55), kHighBit, kHighBit},
        {Pm4MemoryComparison::kNotEqual, AMDF_ATOMIC_WAIT_CONDITION_NOT_EQUAL,
         T(kHighBit | 0x55), T{0x55}, kHighBit, kHighBit},
        {Pm4MemoryComparison::kGreaterOrEqual,
         AMDF_ATOMIC_WAIT_CONDITION_UNSIGNED_GREATER_EQUAL, T(kHighBit - 1),
         T(kHighBit + 1), kHighBit, T(~T{0})},
    };
    RecordProperty("atomic_width_bits", sizeof(T) * 8);
    RecordProperty("atomic_profile", creation.memory_profile_ordinal);
    RecordProperty("atomic_memory_operations",
                   std::to_string(memory_operations));
    RecordProperty("atomic_xcc_count", gpu_endpoint_info_.topology.xcc_count);
    for (const auto& test : cases) {
      SCOPED_TRACE(static_cast<uint32_t>(test.comparison));
      ASSERT_NE(conditions & test.condition, 0u);
      for (uint32_t epoch = 1; epoch <= 2; ++epoch) {
        SCOPED_TRACE(epoch);
        std::array<uint8_t, kByteLength> expected, observed;
        expected.fill(0x69);
        const auto set_expected = [&](uint32_t offset, const auto& value) {
          std::memcpy(expected.data() + offset, &value, sizeof(value));
        };
        const uint32_t zero = 0;
        const T poison = T(0xfedcba9876543210ull);
        const T result = T(0x1234567887654321ull + epoch);
        for (uint32_t offset :
             {kEnteredOffset, kObservedOffset, kAcknowledgedOffset}) {
          set_expected(offset, zero);
        }
        set_expected(kInputOffset, test.initial);
        set_expected(kOutputOffset, poison);
        std::memcpy(control->host.pointer, expected.data(), expected.size());
        signal = aql::UserSignal(1);
        std::array<uint32_t, kByteLength / sizeof(uint32_t)> words;
        words.fill(0x13579bdf);
        constexpr uint32_t kBodyWords =
            6 + (sizeof(T) == 4 ? 7 : 9) + 6 + 6 + 7;
        uint32_t word_count = aql::SingleExecutorPrefix(
            words.data(), gpu_endpoint_info_.topology.xcc_count, kBodyWords);
        word_count += pm4::StoreImmediate(words.data() + word_count,
                                          gpu_address + kEnteredOffset, epoch,
                                          pm4::CopyDataWidth::k32Bit);
        if constexpr (sizeof(T) == 4) {
          word_count += pm4::WaitMemory32(
              words.data() + word_count, gpu_address + kInputOffset,
              test.reference, test.comparison, test.mask);
        } else {
          word_count += pm4::WaitMemory64(
              words.data() + word_count, gpu_address + kInputOffset,
              test.reference, test.comparison, test.mask);
        }
        word_count +=
            pm4::StoreImmediate(words.data() + word_count,
                                gpu_address + kOutputOffset, result, kWidth);
        word_count += pm4::StoreImmediate(words.data() + word_count,
                                          gpu_address + kObservedOffset, epoch,
                                          pm4::CopyDataWidth::k32Bit);
        word_count += pm4::WaitMemory32(
            words.data() + word_count, gpu_address + kAcknowledgedOffset, epoch,
            Pm4MemoryComparison::kEqual, UINT32_MAX);
        std::memcpy(commands->host.pointer, words.data(), sizeof(words));
        // Acquire publishes the host-written IB. Release NONE keeps the control
        // observation independent of any carrier-completion cache operation.
        const auto packet = aql::IndirectBuffer(
            aql::HeaderBarrier::kEnabled, commands->device_address, word_count,
            completion->device_address,
            {aql::FenceScope::kSystem, aql::FenceScope::kNone});
        GpuStoreRelease(queue->host.write_index_address, index + 1);
        ASSERT_TRUE(aql::Publish(*queue, index++, packet));
        EXPECT_TRUE(
            GpuWaitEqual<uint32_t>(host_address + kEnteredOffset, epoch));
        EXPECT_EQ(GpuLoadAcquire<T>(host_address + kOutputOffset), poison);
        GpuStoreRelease(host_address + kInputOffset, test.released);
        EXPECT_TRUE(
            GpuWaitEqual<uint32_t>(host_address + kObservedOffset, epoch));
        // Read the exact control word before permitting the carrier to finish.
        // The host acknowledgement always runs, even after an oracle mismatch.
        const T observed_result =
            GpuLoadAcquire<T>(host_address + kOutputOffset);
        const int64_t pending =
            GpuLoadAcquire<int64_t>(reinterpret_cast<uintptr_t>(&signal.value));
        GpuStoreRelease(host_address + kAcknowledgedOffset, epoch);
        EXPECT_NO_FATAL_FAILURE(
            WaitCompletionAndConsumption(*queue, signal, index));
        EXPECT_EQ(observed_result, result);
        EXPECT_EQ(pending, 1);
        set_expected(kEnteredOffset, epoch);
        set_expected(kObservedOffset, epoch);
        set_expected(kAcknowledgedOffset, epoch);
        set_expected(kInputOffset, test.released);
        set_expected(kOutputOffset, result);
        std::memcpy(observed.data(), control->host.pointer, observed.size());
        EXPECT_EQ(observed, expected);
        EXPECT_EQ(
            std::memcmp(commands->host.pointer, words.data(), sizeof(words)),
            0);
        if (HasFailure()) {
          return;
        }
      }
    }
  }
};

TEST_P(AqlAtomicTest, WaitStoreHandoffsPrecedeCarrierCompletion) {
  if (GetParam() == sizeof(uint32_t)) {
    RunHostHandoffs<uint32_t>();
  } else {
    RunHostHandoffs<uint64_t>();
  }
}

INSTANTIATE_TEST_SUITE_P(Width, AqlAtomicTest,
                         ::testing::Values(sizeof(uint32_t), sizeof(uint64_t)),
                         [](const ::testing::TestParamInfo<size_t>& info) {
                           return info.param == sizeof(uint32_t) ? "Dword"
                                                                 : "Qword";
                         });

}  // namespace
