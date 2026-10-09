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
#include <vector>

#include "libamdf/cts/gpu/pm4/command_fixture.h"
#include "libamdf/cts/gpu/pm4/encoding/commands.h"

namespace {

void CheckTransition(const amdf_cache_transition_t& transition,
                     amdf_cache_operation_t operation) {
  const bool global = operation != AMDF_CACHE_OPERATION_NONE;
  ASSERT_EQ(transition.kind, global ? AMDF_CACHE_TRANSITION_KIND_GLOBAL
                                    : AMDF_CACHE_TRANSITION_KIND_NONE);
  ASSERT_EQ(transition.executor, global ? AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE
                                        : AMDF_CACHE_TRANSITION_EXECUTOR_NONE);
  ASSERT_EQ(transition.operation, operation);
  ASSERT_EQ(transition.host_operation, AMDF_HOST_CACHE_OPERATION_NONE);
  ASSERT_EQ(transition.host_instruction, AMDF_HOST_CACHE_INSTRUCTION_NONE);
  ASSERT_EQ(transition.host_fence_before, AMDF_HOST_CACHE_FENCE_NONE);
  ASSERT_EQ(transition.host_fence_after, AMDF_HOST_CACHE_FENCE_NONE);
  ASSERT_EQ(transition.range_granularity, 0u);
}

class Pm4AtomicStoreTest : public Pm4CommandTest,
                           public ::testing::WithParamInterface<size_t> {
 protected:
  Pm4AtomicStoreTest() : Pm4CommandTest(AMDF_QUEUE_ROLE_ATOMIC) {}

  template <typename T>
  void RunStores() {
    static_assert(std::atomic_ref<T>::is_always_lock_free);
    constexpr uint32_t kPageByteLength = 4096;
    constexpr uint32_t kCompletionByteOffset = 128;
    constexpr uint32_t kEpochCount = 2;
    constexpr uint32_t kCellCount = 2;
    constexpr uint32_t kWordsPerEpoch = 48;
    constexpr uint32_t kCommandWordCount = kEpochCount * kWordsPerEpoch;
    constexpr amdf_memory_access_t kReadWrite =
        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
    constexpr std::array<uint32_t, kCellCount> kOffsets =
        sizeof(T) == 4 ? std::array<uint32_t, kCellCount>{60, 4088}
                       : std::array<uint32_t, kCellCount>{56, 4080};
    constexpr std::array<std::array<uint64_t, kCellCount>, kEpochCount>
        kDwordValues = {{{0, 0xffffffffu}, {0x13579bdfu, 0xa5c31f27u}}};
    constexpr std::array<std::array<uint64_t, kCellCount>, kEpochCount>
        kQwordValues = {
            {{UINT64_C(0x0123456789abcdef), UINT64_C(0xfedcba9876543210)},
             {UINT64_C(0x89abcdef01234567), UINT64_C(0x76543210fedcba98)}}};
    const auto& values = sizeof(T) == 4 ? kDwordValues : kQwordValues;
    constexpr std::array<std::array<uint64_t, kCellCount>, kEpochCount>
        kHostValues = {
            {{UINT64_C(0x2468ace12468ace1), UINT64_C(0xdb97531edb97531e)},
             {UINT64_C(0x4badc0de4badc0de), UINT64_C(0xb4523f21b4523f21)}}};
    constexpr std::array<uint64_t, kCellCount> kInitialValues = {
        UINT64_C(0x0badf00d0badf00d), UINT64_C(0xc001d00dc001d00d)};

    const auto& atomics = family_.atomic_capabilities;
    const auto queue_operations =
        sizeof(T) == 4 ? atomics.operations_32 : atomics.operations_64;
    const auto queue_native_operations =
        sizeof(T) == 4 ? atomics.operations_without_dispatch_32
                       : atomics.operations_without_dispatch_64;
    if ((queue_operations & AMDF_ATOMIC_OPERATION_STORE) == 0 ||
        (queue_native_operations & AMDF_ATOMIC_OPERATION_STORE) == 0) {
      GTEST_SKIP() << "queue does not encode native STORE for this width";
    }
    const amdf_memory_device_access_t attachment = {
        device_,
        {.access = kReadWrite,
         .flags =
             AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_DEVICE_ADDRESS}};
    amdf_memory_create_info_t creation = {};
    creation.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
    creation.structure_size = sizeof(creation);
    creation.memory_profile_ordinal = FindMemoryProfileOrdinal(
        system_scope_,
        AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
        AMDF_MEMORY_FLAG_HOST_VISIBLE, attachment.requirements);
    ASSERT_NE(creation.memory_profile_ordinal,
              AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
    creation.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
    creation.access_count = 1;
    creation.accesses = &attachment;
    creation.byte_length = kPageByteLength;
    creation.minimum_alignment = kPageByteLength;
    amdf_memory_profile_t profile = {.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
                                     .structure_size = sizeof(profile)};
    amdf_memory_access_capabilities_t capabilities = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
        .structure_size = sizeof(capabilities)};
    ASSERT_EQ(
        QueryMemoryProfile(system_scope_, creation.memory_profile_ordinal,
                           attachment.requirements, &profile, &capabilities),
        AMDF_STATUS_OK);
    ASSERT_EQ(profile.memory_class, AMDF_MEMORY_CLASS_SYSTEM);
    const auto profile_operations = sizeof(T) == 4
                                        ? capabilities.atomic_operations_32
                                        : capabilities.atomic_operations_64;
    if ((profile_operations & AMDF_ATOMIC_OPERATION_STORE) == 0) {
      GTEST_SKIP()
          << "owned SYSTEM profile has no STORE contract for this width";
    }

    std::array<amdf_memory_pair_info_t, 2> prospective = {};
    for (size_t direction = 0; direction < prospective.size(); ++direction) {
      amdf_memory_profile_pair_query_t query = {};
      query.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE_PAIR_QUERY;
      query.structure_size = sizeof(query);
      query.memory_profile_ordinal = creation.memory_profile_ordinal;
      query.required_flags = creation.required_flags;
      query.access_count = creation.access_count;
      query.accesses = creation.accesses;
      query.registered_host_cacheability =
          creation.registered_host_cacheability;
      amdf_memory_profile_site_t host = {.kind = AMDF_MEMORY_SITE_KIND_HOST};
      host.value.host_access = direction == 0 ? AMDF_MEMORY_MAP_FLAG_WRITE
                                              : AMDF_MEMORY_MAP_FLAG_READ;
      amdf_memory_profile_site_t pm4 = {.kind = AMDF_MEMORY_SITE_KIND_DEVICE};
      pm4.value.device.access_ordinal = 0;
      pm4.value.device.queue_family_ordinal = family_.ordinal;
      query.producer = direction == 0 ? host : pm4;
      query.consumer = direction == 0 ? pm4 : host;
      auto& pair = prospective[direction];
      pair.type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO;
      pair.structure_size = sizeof(pair);
      const amdf_status_t status =
          api_->memory_scope_query_pair_info(system_scope_, &query, &pair);
      if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
        GTEST_SKIP() << "owned SYSTEM pair has no PM4 handoff contract";
      }
      ASSERT_EQ(status, AMDF_STATUS_OK);
      const auto reach = sizeof(T) == 4 ? pair.atomic_reach.scope_32
                                        : pair.atomic_reach.scope_64;
      if (reach != AMDF_ATOMIC_SCOPE_SYSTEM) {
        GTEST_SKIP() << "owned SYSTEM pair has no CPU/PM4 atomic reach";
      }
      ASSERT_NE(pair.flags & AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE,
                0u);
      ASSERT_NO_FATAL_FAILURE(CheckTransition(
          pair.release, direction == 0
                            ? AMDF_CACHE_OPERATION_NONE
                            : AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM));
      ASSERT_NO_FATAL_FAILURE(CheckTransition(
          pair.acquire, direction == 0
                            ? AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM
                            : AMDF_CACHE_OPERATION_NONE));
    }

    // The same prospective inputs describe each separate owned allocation.
    // Concrete queries still name two sites within one backing at a time.
    GpuMemory* target = nullptr;
    GpuMemory* control = nullptr;
    ASSERT_NO_FATAL_FAILURE(CreateMemory(system_scope_, creation, &target));
    ASSERT_NO_FATAL_FAILURE(CreateMemory(system_scope_, creation, &control));
    const auto target_operations =
        sizeof(T) == 4 ? target->access_info.atomic_operations_32
                       : target->access_info.atomic_operations_64;
    ASSERT_NE(target_operations & AMDF_ATOMIC_OPERATION_STORE, 0u);
    for (GpuMemory* memory : {target, control}) {
      for (size_t direction = 0; direction < prospective.size(); ++direction) {
        const auto host = memory->HostSite();
        const auto pm4 = memory->DeviceSite(family_.ordinal);
        const auto& producer = direction == 0 ? host : pm4;
        const auto& consumer = direction == 0 ? pm4 : host;
        amdf_memory_pair_info_t pair = {
            .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
            .structure_size = sizeof(pair)};
        ASSERT_EQ(api_->memory_query_pair_info(&producer, &consumer, &pair),
                  AMDF_STATUS_OK);
        ASSERT_NE(pair.flags & AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE,
                  0u);
        ASSERT_NO_FATAL_FAILURE(CheckTransition(
            pair.release, direction == 0
                              ? AMDF_CACHE_OPERATION_NONE
                              : AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM));
        ASSERT_NO_FATAL_FAILURE(CheckTransition(
            pair.acquire, direction == 0
                              ? AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM
                              : AMDF_CACHE_OPERATION_NONE));
        if (memory == target) {
          const auto reach = sizeof(T) == 4 ? pair.atomic_reach.scope_32
                                            : pair.atomic_reach.scope_64;
          ASSERT_EQ(reach, AMDF_ATOMIC_SCOPE_SYSTEM);
        }
      }
    }

    std::array<uint8_t, kPageByteLength> expected_target, observed_target;
    std::array<uint8_t, kPageByteLength> expected_control, observed_control;
    for (uint32_t i = 0; i < kPageByteLength; ++i) {
      expected_target[i] = static_cast<uint8_t>(0x5au ^ (i * 37u + 11u));
      expected_control[i] = static_cast<uint8_t>(0xc3u ^ (i * 17u + 5u));
    }
    std::memcpy(target->host.pointer, expected_target.data(), kPageByteLength);
    std::array<T*, kCellCount> cells;
    for (size_t cell = 0; cell < kCellCount; ++cell) {
      const uintptr_t address =
          reinterpret_cast<uintptr_t>(target->host.pointer) + kOffsets[cell];
      ASSERT_EQ(address % std::atomic_ref<T>::required_alignment, 0u);
      ASSERT_EQ((target->device_address + kOffsets[cell]) % sizeof(T), 0u);
      const T value = static_cast<T>(kInitialValues[cell]);
      cells[cell] = std::construct_at(reinterpret_cast<T*>(address), value);
      std::memcpy(expected_target.data() + kOffsets[cell], &value, sizeof(T));
    }
    const uint32_t initial_completion = 0;
    std::memcpy(expected_control.data() + kCompletionByteOffset,
                &initial_completion, sizeof(initial_completion));
    std::memcpy(control->host.pointer, expected_control.data(),
                kPageByteLength);
    const uintptr_t completion_address =
        reinterpret_cast<uintptr_t>(control->host.pointer) +
        kCompletionByteOffset;

    GpuCommandQueue* queue = nullptr;
    ASSERT_NO_FATAL_FAILURE(CreateQueue(&queue));
    const size_t command_capacity = queue->words().size();
    ASSERT_GT(command_capacity, kCommandWordCount);
    std::vector<uint32_t> expected_commands(command_capacity, 0);
    std::vector<uint32_t> observed_commands(command_capacity);
    Pm4CommandWriter commands(expected_commands.data(), *pm4_profile_);
    for (uint32_t epoch = 0; epoch < kEpochCount; ++epoch) {
      commands.SystemBarrier();
      for (size_t cell = 0; cell < kCellCount; ++cell) {
        const uint64_t address = target->device_address + kOffsets[cell];
        if constexpr (sizeof(T) == 4) {
          commands.AtomicStore32(address,
                                 static_cast<uint32_t>(values[epoch][cell]));
        } else {
          commands.AtomicStore64(address, values[epoch][cell]);
        }
      }
      commands.SystemBarrier();
      commands.WriteData32(control->device_address + kCompletionByteOffset,
                           epoch + 1);
      commands.PadToEightWords();
      ASSERT_EQ(commands.word_count(), (epoch + 1) * kWordsPerEpoch);
    }
    // Initialize even the unsubmitted extent; neither batch is rewritten.
    std::memcpy(queue->words().data(), expected_commands.data(),
                queue->words().size_bytes());
    RecordProperty("pm4_atomic_store_width_bits", sizeof(T) * 8);
    RecordProperty("pm4_atomic_store_profile_ordinal",
                   creation.memory_profile_ordinal);
    RecordProperty("pm4_atomic_store_profile_operations",
                   std::to_string(profile_operations));
    RecordProperty("pm4_atomic_store_target_operations",
                   std::to_string(target_operations));
    RecordProperty("pm4_atomic_store_host_to_pm4_scope",
                   AMDF_ATOMIC_SCOPE_SYSTEM);
    RecordProperty("pm4_atomic_store_pm4_to_host_scope",
                   AMDF_ATOMIC_SCOPE_SYSTEM);
    RecordProperty("pm4_atomic_store_cell_0_byte_offset", kOffsets[0]);
    RecordProperty("pm4_atomic_store_cell_1_byte_offset", kOffsets[1]);
    RecordProperty("pm4_atomic_store_completion_byte_offset",
                   kCompletionByteOffset);
    RecordProperty("pm4_atomic_store_checked_page_bytes_each", kPageByteLength);
    RecordProperty("pm4_atomic_store_command_capacity_dwords",
                   std::to_string(command_capacity));
    RecordProperty("pm4_atomic_store_command_word_count",
                   commands.word_count());
    RecordProperty("pm4_atomic_store_completed_epochs", 0);

    for (uint32_t epoch = 0; epoch < kEpochCount; ++epoch) {
      SCOPED_TRACE(epoch);
      for (size_t cell = 0; cell < kCellCount; ++cell) {
        const T value = static_cast<T>(values[epoch][cell]);
        std::memcpy(expected_target.data() + kOffsets[cell], &value, sizeof(T));
      }
      const uint32_t completion = epoch + 1;
      std::memcpy(expected_control.data() + kCompletionByteOffset, &completion,
                  sizeof(completion));
      const uint64_t published_word_count = completion * kWordsPerEpoch;
      ASSERT_NO_FATAL_FAILURE(
          queue->Publish(api_, gpu_api_, published_word_count));
      GpuWaitEqual<uint32_t>(completion_address, completion);
      // Capture all initialized storage before diagnostics or retirement can
      // supply additional synchronization to the observation under test.
      std::memcpy(observed_target.data(), target->host.pointer,
                  kPageByteLength);
      std::memcpy(observed_control.data(), control->host.pointer,
                  kPageByteLength);
      std::memcpy(observed_commands.data(), queue->words().data(),
                  queue->words().size_bytes());
      for (uint32_t i = 0; i < kPageByteLength; ++i) {
        EXPECT_EQ(observed_target[i], expected_target[i])
            << "target byte=" << i;
        EXPECT_EQ(observed_control[i], expected_control[i])
            << "control byte=" << i;
      }
      for (size_t i = 0; i < command_capacity; ++i) {
        EXPECT_EQ(observed_commands[i], expected_commands[i])
            << "command storage word=" << i;
      }
      // The marker joins STORE visibility; native progress separately retires
      // the command borrow. Oracle mismatches never skip that obligation.
      EXPECT_NO_FATAL_FAILURE(queue->WaitRetired(api_));
      if (HasFailure()) {
        return;
      }

      std::array<T, kCellCount> loaded_values, exchanged_values;
      for (size_t cell = 0; cell < kCellCount; ++cell) {
        const T value = static_cast<T>(kHostValues[epoch][cell]);
        // No atomic_ref survives into an ordinary snapshot or a GPU epoch.
        {
          std::atomic_ref<T> reference(*cells[cell]);
          loaded_values[cell] = reference.load(std::memory_order_acquire);
          exchanged_values[cell] =
              reference.exchange(value, std::memory_order_acq_rel);
        }
        std::memcpy(expected_target.data() + kOffsets[cell], &value, sizeof(T));
      }
      std::memcpy(observed_target.data(), target->host.pointer,
                  kPageByteLength);
      for (size_t cell = 0; cell < kCellCount; ++cell) {
        const T expected = static_cast<T>(values[epoch][cell]);
        EXPECT_EQ(loaded_values[cell], expected) << "cell=" << cell;
        EXPECT_EQ(exchanged_values[cell], expected) << "cell=" << cell;
      }
      for (uint32_t i = 0; i < kPageByteLength; ++i) {
        EXPECT_EQ(observed_target[i], expected_target[i])
            << "CPU target byte=" << i;
      }
      if (HasFailure()) {
        return;
      }
      RecordProperty("pm4_atomic_store_completed_epochs", completion);
    }
    RecordProperty("pm4_atomic_store_store_count", kEpochCount * kCellCount);
    RecordProperty("pm4_atomic_store_cpu_exchange_count",
                   kEpochCount * kCellCount);
    RecordProperty("pm4_atomic_store_final_published_word_count",
                   std::to_string(kCommandWordCount));
  }
};

TEST_P(Pm4AtomicStoreTest, StoresObserveCpuHandoffsAcrossEpochs) {
  if (GetParam() == sizeof(uint32_t)) {
    RunStores<uint32_t>();
  } else {
    RunStores<uint64_t>();
  }
}

INSTANTIATE_TEST_SUITE_P(Width, Pm4AtomicStoreTest,
                         ::testing::Values(sizeof(uint32_t), sizeof(uint64_t)),
                         [](const ::testing::TestParamInfo<size_t>& info) {
                           return info.param == sizeof(uint32_t) ? "Dword"
                                                                 : "Qword";
                         });

}  // namespace
