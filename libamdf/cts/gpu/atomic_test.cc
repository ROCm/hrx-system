// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>

#include "libamdf/cts/gpu/aql/publication.h"
#include "libamdf/cts/gpu/pm4/encoding/memory_commands.h"
#include "libamdf/cts/gpu/util/command_fixture.h"

namespace {

struct QueueAtomicCase {
  // Outer queue format; both formats execute the same MEC memory operations.
  amdf_queue_command_type_t command_type;
  // Naturally aligned counter width in bytes.
  size_t byte_width;
};

class GpuAtomicAddTest : public GpuCommandTest,
                         public ::testing::WithParamInterface<QueueAtomicCase> {
 protected:
  GpuAtomicAddTest()
      : GpuCommandTest({
            .command_type = GetParam().command_type,
            .roles = AMDF_QUEUE_ROLE_ATOMIC,
            .cache_operations = AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM |
                                AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM,
            .cache_transition_kinds = AMDF_CACHE_TRANSITION_KINDS_GLOBAL,
            .publication_modes = AMDF_QUEUE_PUBLICATION_MODE_USER |
                                 AMDF_QUEUE_PUBLICATION_MODE_KERNEL,
        }) {}

  template <typename T>
  void RunContendedAdds() {
    static_assert(std::atomic_ref<T>::is_always_lock_free);
    constexpr uint32_t kByteLength = 4096;
    constexpr uint32_t kEnteredOffset = 64;
    constexpr uint32_t kStartedOffset = 128;
    constexpr uint32_t kObservedOffset = 192;
    constexpr uint32_t kAcknowledgedOffset = 256;
    constexpr uint32_t kCompletedOffset = 320;
    constexpr uint32_t kCounterOffset = kByteLength - sizeof(T);
    constexpr uint32_t kGpuAddCount = 128;
    constexpr auto kOperations = AMDF_ATOMIC_OPERATION_WAIT |
                                 AMDF_ATOMIC_OPERATION_STORE |
                                 AMDF_ATOMIC_OPERATION_ADD;
    const bool is_aql = family_.command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL;
    const auto& atomics = family_.atomic_capabilities;
    const auto operations =
        sizeof(T) == 4 ? atomics.operations_32 : atomics.operations_64;
    const auto native_operations = sizeof(T) == 4
                                       ? atomics.operations_without_dispatch_32
                                       : atomics.operations_without_dispatch_64;
    if ((operations & native_operations & kOperations) != kOperations ||
        (atomics.operations_without_dispatch_32 & kOperations) != kOperations ||
        (atomics.wait_conditions_32 & AMDF_ATOMIC_WAIT_CONDITION_EQUAL) == 0) {
      GTEST_SKIP() << "queue lacks native control-word and ADD commands";
    }

    const amdf_memory_device_access_t attachment = {
        device_,
        {
            .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
            .flags = AMDF_MEMORY_FLAG_HOST_COHERENT |
                     AMDF_MEMORY_FLAG_DEVICE_ADDRESS |
                     AMDF_MEMORY_FLAG_QUEUE_STORAGE,
        },
    };
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
      GTEST_SKIP() << "control backing lacks CPU/GPU ADD for used widths";
    }
    RecordProperty("atomic_profile", creation.memory_profile_ordinal);
    RecordProperty("atomic_xcc_count", gpu_endpoint_info_.topology.xcc_count);
    RecordProperty("gpu_add_count_per_epoch", kGpuAddCount);
    GpuMemory* control = nullptr;
    ASSERT_NO_FATAL_FAILURE(CreateMemory(system_scope_, creation, &control));
    ASSERT_EQ(control->access_info.atomic_operations_32,
              capabilities.atomic_operations_32);
    ASSERT_EQ(control->access_info.atomic_operations_64,
              capabilities.atomic_operations_64);
    ASSERT_EQ(control->host.cacheability, AMDF_HOST_CACHEABILITY_WRITE_BACK);
    for (uint32_t direction = 0; direction < 2; ++direction) {
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
      amdf_memory_pair_info_t prospective = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
          .structure_size = sizeof(prospective),
      };
      ASSERT_EQ(api_->memory_scope_query_pair_info(system_scope_, &query,
                                                   &prospective),
                AMDF_STATUS_OK);
      ASSERT_EQ(prospective.atomic_reach.scope_32, AMDF_ATOMIC_SCOPE_SYSTEM);
      if constexpr (sizeof(T) == 8) {
        ASSERT_EQ(prospective.atomic_reach.scope_64, AMDF_ATOMIC_SCOPE_SYSTEM);
      }
      const auto host_site = control->HostSite();
      const auto gpu_site = control->DeviceSite(family_.ordinal);
      amdf_memory_pair_info_t achieved = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
          .structure_size = sizeof(achieved),
      };
      ASSERT_EQ(api_->memory_query_pair_info(
                    direction == 0 ? &host_site : &gpu_site,
                    direction == 0 ? &gpu_site : &host_site, &achieved),
                AMDF_STATUS_OK);
      ASSERT_EQ(achieved.atomic_reach.scope_32,
                prospective.atomic_reach.scope_32);
      ASSERT_EQ(achieved.atomic_reach.scope_64,
                prospective.atomic_reach.scope_64);
    }

    GpuCommandQueue* pm4_queue = nullptr;
    GpuUserQueue* aql_queue = nullptr;
    GpuMemory* commands = nullptr;
    GpuMemory* completion = nullptr;
    if (is_aql) {
      ASSERT_NO_FATAL_FAILURE(CreateQueue(&aql_queue));
      ASSERT_NO_FATAL_FAILURE(
          CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_EXECUTE,
                       8192, &commands));
      ASSERT_NO_FATAL_FAILURE(
          CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096,
                       &completion));
    } else {
      ASSERT_NO_FATAL_FAILURE(CreateQueue(&pm4_queue, 32768));
    }
    const uintptr_t host_address =
        reinterpret_cast<uintptr_t>(control->host.pointer);
    const uint64_t gpu_address = control->device_address;
    ASSERT_EQ((host_address + kCounterOffset) %
                  std::atomic_ref<T>::required_alignment,
              0u);
    ASSERT_EQ((gpu_address + kCounterOffset) % sizeof(T), 0u);
    uint32_t pm4_word_count = 0;
    uint64_t aql_index =
        is_aql ? GpuLoadAcquire<uint64_t>(aql_queue->host.write_index_address)
               : 0;
    const struct {
      // Initial value exercises full-width carry and wraparound.
      T initial;
      // Unsigned ADD operand, including two's-complement negative values.
      T operand;
    } cases[] = {
        {T{0}, T{1}},
        {T(~T{0} - 64), T{1}},
        {T{64}, T(~T{0})},
        {T(0x1fffffffeull), T(0x100000001ull)},
    };
    for (uint32_t epoch = 1; epoch <= std::size(cases); ++epoch) {
      SCOPED_TRACE(epoch);
      const auto& test = cases[epoch - 1];
      std::array<uint8_t, kByteLength> expected, observed;
      expected.fill(0x69);
      const auto set_expected = [&](uint32_t offset, const auto& value) {
        std::memcpy(expected.data() + offset, &value, sizeof(value));
      };
      const uint32_t zero = 0;
      for (uint32_t offset : {kEnteredOffset, kStartedOffset, kObservedOffset,
                              kAcknowledgedOffset, kCompletedOffset}) {
        set_expected(offset, zero);
      }
      set_expected(kCounterOffset, test.initial);
      std::memcpy(control->host.pointer, expected.data(), expected.size());
      T* counter = std::construct_at(
          reinterpret_cast<T*>(host_address + kCounterOffset), test.initial);
      std::array<uint32_t, 2048> words;
      words.fill(0x13579bdf);
      constexpr uint32_t kBodyWords = 6 + 7 + kGpuAddCount * 9 + 6 + 7 + 6;
      uint32_t word_count =
          is_aql ? aql::SingleExecutorPrefix(
                       words.data(), gpu_endpoint_info_.topology.xcc_count,
                       kBodyWords)
                 : 0;
      word_count += pm4::StoreImmediate(words.data() + word_count,
                                        gpu_address + kEnteredOffset, epoch,
                                        pm4::CopyDataWidth::k32Bit);
      word_count += pm4::WaitMemory32(words.data() + word_count,
                                      gpu_address + kStartedOffset, epoch,
                                      Pm4MemoryComparison::kEqual, UINT32_MAX);
      for (uint32_t i = 0; i < kGpuAddCount; ++i) {
        if constexpr (sizeof(T) == 4) {
          word_count +=
              pm4::AtomicAdd32(words.data() + word_count,
                               gpu_address + kCounterOffset, test.operand);
        } else {
          word_count +=
              pm4::AtomicAdd64(words.data() + word_count,
                               gpu_address + kCounterOffset, test.operand);
        }
      }
      word_count += pm4::StoreImmediate(words.data() + word_count,
                                        gpu_address + kObservedOffset, epoch,
                                        pm4::CopyDataWidth::k32Bit);
      word_count += pm4::WaitMemory32(words.data() + word_count,
                                      gpu_address + kAcknowledgedOffset, epoch,
                                      Pm4MemoryComparison::kEqual, UINT32_MAX);
      word_count += pm4::StoreImmediate(words.data() + word_count,
                                        gpu_address + kCompletedOffset, epoch,
                                        pm4::CopyDataWidth::k32Bit);
      if (is_aql) {
        *static_cast<aql::Signal*>(completion->host.pointer) =
            aql::UserSignal(1);
        std::memcpy(commands->host.pointer, words.data(), sizeof(words));
        const auto packet = aql::IndirectBuffer(
            aql::HeaderBarrier::kEnabled, commands->device_address, word_count,
            completion->device_address,
            {aql::FenceScope::kSystem, aql::FenceScope::kNone});
        GpuStoreRelease(aql_queue->host.write_index_address, aql_index + 1);
        ASSERT_TRUE(aql::Publish(*aql_queue, aql_index++, packet));
      } else {
        ASSERT_LE(pm4_word_count + word_count, pm4_queue->words().size());
        std::memcpy(pm4_queue->words().data() + pm4_word_count, words.data(),
                    word_count * sizeof(uint32_t));
        pm4_word_count += word_count;
        ASSERT_NO_FATAL_FAILURE(
            pm4_queue->Publish(api_, gpu_api_, pm4_word_count));
      }
      EXPECT_TRUE(GpuWaitEqual<uint32_t>(host_address + kEnteredOffset, epoch));
      GpuStoreRelease(host_address + kStartedOffset, epoch);
      uint64_t cpu_add_count = 0;
      T result;
      // End atomic_ref ownership before the bytewise readback below.
      {
        const std::atomic_ref<T> atomic_counter(*counter);
        do {
          atomic_counter.fetch_add(T{3}, std::memory_order_relaxed);
          ++cpu_add_count;
        } while (GpuLoadAcquire<uint32_t>(host_address + kObservedOffset) !=
                 epoch);
        result = atomic_counter.load(std::memory_order_relaxed);
      }
      const uint32_t completed =
          GpuLoadAcquire<uint32_t>(host_address + kCompletedOffset);
      // Acknowledgement always runs before reporting an oracle mismatch.
      GpuStoreRelease(host_address + kAcknowledgedOffset, epoch);
      EXPECT_TRUE(
          GpuWaitEqual<uint32_t>(host_address + kCompletedOffset, epoch));
      if (is_aql) {
        auto& signal = *static_cast<aql::Signal*>(completion->host.pointer);
        EXPECT_TRUE(GpuWaitEqual<int64_t>(
            reinterpret_cast<uintptr_t>(&signal.value), 0));
        ASSERT_NO_FATAL_FAILURE(aql_queue->WaitConsumed(api_, aql_index));
        EXPECT_EQ(
            std::memcmp(commands->host.pointer, words.data(), sizeof(words)),
            0);
      } else {
        ASSERT_NO_FATAL_FAILURE(pm4_queue->WaitRetired(api_));
      }
      RecordProperty("cpu_add_count_epoch_" + std::to_string(epoch),
                     std::to_string(cpu_add_count));
      EXPECT_EQ(completed, 0u);
      const T expected_result =
          T(test.initial + T(kGpuAddCount) * test.operand +
            T(cpu_add_count) * T{3});
      EXPECT_EQ(result, expected_result) << "CPU additions: " << cpu_add_count;
      for (uint32_t offset : {kEnteredOffset, kStartedOffset, kObservedOffset,
                              kAcknowledgedOffset, kCompletedOffset}) {
        set_expected(offset, epoch);
      }
      set_expected(kCounterOffset, expected_result);
      std::memcpy(observed.data(), control->host.pointer, observed.size());
      EXPECT_EQ(observed, expected);
      if (HasFailure()) {
        return;
      }
    }
  }
};

TEST_P(GpuAtomicAddTest, CpuAndQueueUpdatesPrecedeCompletion) {
  if (GetParam().byte_width == sizeof(uint32_t)) {
    RunContendedAdds<uint32_t>();
  } else {
    RunContendedAdds<uint64_t>();
  }
}

INSTANTIATE_TEST_SUITE_P(
    FormatAndWidth, GpuAtomicAddTest,
    ::testing::Values(
        QueueAtomicCase{AMDF_QUEUE_COMMAND_TYPE_GPU_PM4, sizeof(uint32_t)},
        QueueAtomicCase{AMDF_QUEUE_COMMAND_TYPE_GPU_PM4, sizeof(uint64_t)},
        QueueAtomicCase{AMDF_QUEUE_COMMAND_TYPE_GPU_AQL, sizeof(uint32_t)},
        QueueAtomicCase{AMDF_QUEUE_COMMAND_TYPE_GPU_AQL, sizeof(uint64_t)}),
    [](const ::testing::TestParamInfo<QueueAtomicCase>& info) {
      return std::string(info.param.command_type ==
                                 AMDF_QUEUE_COMMAND_TYPE_GPU_AQL
                             ? "Aql"
                             : "Pm4") +
             (info.param.byte_width == sizeof(uint32_t) ? "Dword" : "Qword");
    });

}  // namespace
