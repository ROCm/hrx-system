// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/gpu/user_queue_memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <functional>
#include <thread>

namespace {

constexpr uint64_t kMemoryByteLength = 4096;
constexpr amdf_queue_roles_t kRequiredQueueRoles =
    AMDF_QUEUE_ROLE_TRANSFER | AMDF_QUEUE_ROLE_CACHE_CONTROL;
constexpr amdf_cache_operations_t kRequiredCacheOperations =
    AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM |
    AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM;

// The host publication tests require x86-64. Naturally aligned 64-bit device
// state uses single-copy accesses; the fences order CPU accesses around them.
// In particular, a doorbell store must not become a locked read-modify-write
// instruction against a write-only MMIO aperture.
uint64_t LoadAcquire(volatile uint64_t* address) {
  const uint64_t value = *address;
  std::atomic_thread_fence(std::memory_order_acquire);
  return value;
}

void StoreRelease(volatile uint64_t* address, uint64_t value) {
  std::atomic_thread_fence(std::memory_order_release);
  *address = value;
}

class UserQueueMemoryScenario {
 public:
  UserQueueMemoryScenario(const amdf_api_t* api, const amdf_gpu_api_t* gpu_api,
                          const amdf_queue_family_info_t& family,
                          const UserQueueMemoryCommands& commands,
                          amdf_device_t* device,
                          amdf_memory_scope_t* system_scope)
      : api_(api),
        gpu_api_(gpu_api),
        family_(family),
        commands_(commands),
        device_(device),
        system_scope_(system_scope) {}

  void RunCopiesBetweenExactAccessAttachments(
      const std::function<void()>& before_publication = {});

  bool Release() {
    if (queue_mapping_ != nullptr) {
      const amdf_status_t status =
          api_->user_queue_mapping_destroy(queue_mapping_);
      EXPECT_EQ(status, AMDF_STATUS_OK);
      if (amdf_status_is_ok(status)) queue_mapping_ = nullptr;
    }
    if (queue_mapping_ != nullptr) return false;
    if (queue_ != nullptr) {
      const amdf_status_t status = api_->user_queue_destroy(queue_);
      EXPECT_EQ(status, AMDF_STATUS_OK);
      if (amdf_status_is_ok(status)) queue_ = nullptr;
    }
    // Workload memory remains attached while native execution may still reach
    // it. A queue that cannot prove destruction retains the complete fixture.
    if (queue_ != nullptr) return false;

    DestroyHostMapping(source_mapping_);
    DestroyHostMapping(target_mapping_);
    if (source_mapping_ == nullptr) DestroyMemory(source_memory_);
    if (target_mapping_ == nullptr) DestroyMemory(target_memory_);
    return source_memory_ == nullptr && target_memory_ == nullptr;
  }

 private:
  void CreateMappedSystemMemory(amdf_memory_access_t device_access,
                                amdf_memory_t*& memory,
                                amdf_memory_info_t& memory_info,
                                amdf_memory_access_info_t& access_info,
                                amdf_host_mapping_t*& mapping,
                                amdf_host_mapping_info_t& mapping_info) {
    constexpr amdf_memory_flags_t kRequiredFlags =
        AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_SHAREABLE;
    const amdf_memory_device_access_t access = {
        device_,
        {.access = device_access,
         .flags =
             AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_DEVICE_ADDRESS}};
    const uint32_t profile_ordinal = FindGpuMemoryProfileOrdinal(
        api_, system_scope_, device_,
        AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
        kRequiredFlags, access.requirements);
    ASSERT_NE(profile_ordinal, AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);

    amdf_memory_create_info_t create_info = {};
    create_info.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
    create_info.structure_size = sizeof(create_info);
    create_info.memory_profile_ordinal = profile_ordinal;
    create_info.access_count = 1;
    create_info.accesses = &access;
    create_info.required_flags = kRequiredFlags;
    create_info.byte_length = kMemoryByteLength;
    create_info.minimum_alignment = 4096;
    ASSERT_EQ(api_->memory_create(system_scope_, &create_info, &memory),
              AMDF_STATUS_OK);

    memory_info.type = AMDF_STRUCTURE_TYPE_MEMORY_INFO;
    memory_info.structure_size = sizeof(memory_info);
    ASSERT_EQ(api_->memory_query_info(memory, &memory_info), AMDF_STATUS_OK);
    access_info.type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_INFO;
    access_info.structure_size = sizeof(access_info);
    ASSERT_EQ(api_->memory_query_access_info(memory, 0, &access_info),
              AMDF_STATUS_OK);
    EXPECT_EQ(memory_info.memory_profile_ordinal, profile_ordinal);
    EXPECT_EQ(memory_info.memory_class, AMDF_MEMORY_CLASS_SYSTEM);
    EXPECT_EQ(access_info.access, device_access);
    EXPECT_EQ(access_info.flags & access.requirements.flags,
              access.requirements.flags);
    EXPECT_EQ((memory_info.flags | access_info.flags) & kRequiredFlags,
              kRequiredFlags);
    EXPECT_EQ(memory_info.byte_length, kMemoryByteLength);
    EXPECT_GE(memory_info.native_allocation_byte_length, kMemoryByteLength);
    EXPECT_GE(memory_info.alignment, create_info.minimum_alignment);
    uint64_t address = 0;
    ASSERT_EQ(api_->memory_query_address(memory, 0, AMDF_MEMORY_ADDRESS_GPU,
                                         &address),
              AMDF_STATUS_OK);
    EXPECT_NE(address, 0u);
    EXPECT_EQ(address & (sizeof(uint32_t) - 1), 0u);
    EXPECT_NE(memory_info.physical_backing_id.words[0] |
                  memory_info.physical_backing_id.words[1],
              0u);

    amdf_memory_map_info_t map_info = {};
    map_info.type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO;
    map_info.structure_size = sizeof(map_info);
    map_info.byte_length = kMemoryByteLength;
    map_info.flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
    ASSERT_EQ(api_->memory_map(memory, &map_info, &mapping), AMDF_STATUS_OK);
    mapping_info.type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO;
    mapping_info.structure_size = sizeof(mapping_info);
    ASSERT_EQ(api_->host_mapping_query_info(mapping, &mapping_info),
              AMDF_STATUS_OK);
    ASSERT_NE(mapping_info.pointer, nullptr);
    EXPECT_EQ(mapping_info.memory_byte_offset, 0u);
    EXPECT_EQ(mapping_info.byte_length, kMemoryByteLength);
    EXPECT_EQ(mapping_info.flags,
              AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE);
  }

  void DestroyHostMapping(amdf_host_mapping_t*& mapping) {
    if (mapping == nullptr) return;
    const amdf_status_t status = api_->host_mapping_destroy(mapping);
    EXPECT_EQ(status, AMDF_STATUS_OK);
    if (amdf_status_is_ok(status)) mapping = nullptr;
  }

  void DestroyMemory(amdf_memory_t*& memory) {
    if (memory == nullptr) return;
    const amdf_status_t status = api_->memory_destroy(memory);
    EXPECT_EQ(status, AMDF_STATUS_OK);
    if (amdf_status_is_ok(status)) memory = nullptr;
  }

  // Core API table borrowed from the enclosing device fixture.
  const amdf_api_t* api_;
  // GPU extension table borrowed from the enclosing device fixture.
  const amdf_gpu_api_t* gpu_api_;
  // Complete family facts borrowed from the enclosing fixture.
  const amdf_queue_family_info_t& family_;
  // Caller encoding borrowed through completion and cleanup.
  const UserQueueMemoryCommands& commands_;
  // Execution owner borrowed through the release of all children below.
  amdf_device_t* device_;
  // Shared system placement scope borrowed through memory destruction.
  amdf_memory_scope_t* system_scope_;
  // GPU-readable source attachment retained through queue destruction.
  amdf_memory_t* source_memory_ = nullptr;
  // GPU-writable target attachment retained through queue destruction.
  amdf_memory_t* target_memory_ = nullptr;
  // Host view of the source attachment.
  amdf_host_mapping_t* source_mapping_ = nullptr;
  // Host view of the target attachment.
  amdf_host_mapping_t* target_mapping_ = nullptr;
  // Directly published GPU queue.
  amdf_user_queue_t* queue_ = nullptr;
  // Host producer mapping of the GPU queue.
  amdf_user_queue_mapping_t* queue_mapping_ = nullptr;
  // Immutable properties of the source attachment.
  amdf_memory_info_t source_memory_info_ = {};
  // Immutable properties of the target attachment.
  amdf_memory_info_t target_memory_info_ = {};
  // Prepared GPU access to the source backing.
  amdf_memory_access_info_t source_access_info_ = {};
  // Prepared GPU access to the target backing.
  amdf_memory_access_info_t target_access_info_ = {};
  // Host-view properties of the source attachment.
  amdf_host_mapping_info_t source_mapping_info_ = {};
  // Host-view properties of the target attachment.
  amdf_host_mapping_info_t target_mapping_info_ = {};
};

void UserQueueMemoryScenario::RunCopiesBetweenExactAccessAttachments(
    const std::function<void()>& before_publication) {
  const amdf_queue_command_type_t command_type = family_.command_type;
  const uint32_t queue_family_ordinal = family_.ordinal;

  ASSERT_NO_FATAL_FAILURE(CreateMappedSystemMemory(
      AMDF_MEMORY_ACCESS_READ, source_memory_, source_memory_info_,
      source_access_info_, source_mapping_, source_mapping_info_));
  ASSERT_NO_FATAL_FAILURE(CreateMappedSystemMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, target_memory_,
      target_memory_info_, target_access_info_, target_mapping_,
      target_mapping_info_));
  uint64_t source_address = 0;
  uint64_t target_address = 0;
  ASSERT_EQ(api_->memory_query_address(
                source_memory_, 0, AMDF_MEMORY_ADDRESS_GPU, &source_address),
            AMDF_STATUS_OK);
  ASSERT_EQ(api_->memory_query_address(
                target_memory_, 0, AMDF_MEMORY_ADDRESS_GPU, &target_address),
            AMDF_STATUS_OK);
  EXPECT_NE(source_address, target_address);
  EXPECT_FALSE(amdf_physical_memory_id_is_equal(
      &source_memory_info_.physical_backing_id,
      &target_memory_info_.physical_backing_id));

  auto* source = static_cast<uint32_t*>(source_mapping_info_.pointer);
  auto* target = static_cast<uint32_t*>(target_mapping_info_.pointer);
  auto* completion = reinterpret_cast<uint32_t*>(
      static_cast<uint8_t*>(target_mapping_info_.pointer) +
      kUserQueueMemoryCompletionByteOffset);
  std::array<uint32_t, kUserQueueMemoryElementCount> expected = {};
  for (size_t i = 0; i < kUserQueueMemoryElementCount; ++i) {
    expected[i] =
        UINT32_C(0x13570000) + static_cast<uint32_t>(i) * UINT32_C(0x00110101);
    source[i] = expected[i];
    target[i] = UINT32_C(0xdeadbeef);
  }
  *completion = 0;
  ASSERT_EQ(api_->host_mapping_cache_control(source_mapping_,
                                             AMDF_HOST_CACHE_OPERATION_FLUSH, 0,
                                             kMemoryByteLength),
            AMDF_STATUS_OK);
  ASSERT_EQ(api_->host_mapping_cache_control(target_mapping_,
                                             AMDF_HOST_CACHE_OPERATION_FLUSH, 0,
                                             kMemoryByteLength),
            AMDF_STATUS_OK);

  amdf_gpu_user_queue_create_info_t create_info = {};
  create_info.type = AMDF_STRUCTURE_TYPE_GPU_USER_QUEUE_CREATE_INFO;
  create_info.structure_size = sizeof(create_info);
  create_info.queue_family_ordinal = queue_family_ordinal;
  create_info.priority = AMDF_QUEUE_PRIORITY_NORMAL;
  create_info.producer_mode = AMDF_QUEUE_PRODUCER_MODE_SINGLE;
  create_info.required_capabilities = AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER;
  create_info.ring_byte_length = std::max(family_.minimum_ring_byte_length,
                                          kUserQueueMemoryCommandByteCapacity);
  ASSERT_EQ(gpu_api_->user_queue_create(device_, &create_info, &queue_),
            AMDF_STATUS_OK);

  amdf_user_queue_info_t queue_info = {};
  queue_info.type = AMDF_STRUCTURE_TYPE_USER_QUEUE_INFO;
  queue_info.structure_size = sizeof(queue_info);
  ASSERT_EQ(api_->user_queue_query_info(queue_, &queue_info), AMDF_STATUS_OK);
  EXPECT_EQ(queue_info.queue_family_ordinal, queue_family_ordinal);
  EXPECT_EQ(queue_info.command_type, command_type);
  EXPECT_EQ(queue_info.format_version, family_.format_version);
  EXPECT_EQ(queue_info.format_features, family_.format_features);
  EXPECT_EQ(queue_info.producer_mode, AMDF_QUEUE_PRODUCER_MODE_SINGLE);
  EXPECT_EQ(queue_info.priority, AMDF_QUEUE_PRIORITY_NORMAL);
  EXPECT_NE(queue_info.capabilities & AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER,
            0u);
  EXPECT_EQ(queue_info.roles & kRequiredQueueRoles, kRequiredQueueRoles);
  EXPECT_EQ(queue_info.metadata.command_type, AMDF_QUEUE_COMMAND_TYPE_UNKNOWN);
  EXPECT_EQ(queue_info.metadata_ring_byte_length, 0u);
  EXPECT_TRUE(amdf_device_id_is_equal(&queue_info.device_id,
                                      &source_access_info_.device_id));
  EXPECT_TRUE(amdf_device_id_is_equal(&queue_info.device_id,
                                      &target_access_info_.device_id));

  ASSERT_EQ(api_->user_queue_map(queue_, nullptr, &queue_mapping_),
            AMDF_STATUS_OK);
  amdf_user_queue_mapping_info_t mapping_info = {};
  mapping_info.type = AMDF_STRUCTURE_TYPE_USER_QUEUE_MAPPING_INFO;
  mapping_info.structure_size = sizeof(mapping_info);
  ASSERT_EQ(api_->user_queue_mapping_query_info(queue_mapping_, &mapping_info),
            AMDF_STATUS_OK);
  EXPECT_TRUE(
      amdf_queue_id_is_equal(&mapping_info.queue_id, &queue_info.queue_id));
  EXPECT_EQ(mapping_info.producer_device_id.words[0] |
                mapping_info.producer_device_id.words[1],
            0u);
  EXPECT_EQ(mapping_info.queue_reset_epoch, queue_info.reset_epoch);
  EXPECT_EQ(mapping_info.producer_reset_epoch, 0u);
  EXPECT_EQ(mapping_info.command_type, queue_info.command_type);
  EXPECT_EQ(mapping_info.format_version, queue_info.format_version);
  EXPECT_EQ(mapping_info.format_features, queue_info.format_features);
  EXPECT_EQ(mapping_info.ring_byte_length, queue_info.ring_byte_length);
  EXPECT_EQ(mapping_info.index_bits, 64u);
  EXPECT_EQ(mapping_info.doorbell_bits, 64u);
  EXPECT_EQ(mapping_info.metadata.command_type,
            AMDF_QUEUE_COMMAND_TYPE_UNKNOWN);
  EXPECT_EQ(mapping_info.metadata_ring_address, 0u);
  EXPECT_EQ(mapping_info.metadata_ring_byte_length, 0u);
  ASSERT_NE(mapping_info.ring_address, 0u);
  ASSERT_NE(mapping_info.read_index_address, 0u);
  ASSERT_NE(mapping_info.write_index_address, 0u);
  ASSERT_NE(mapping_info.doorbell_address, 0u);
  ASSERT_EQ(mapping_info.ring_address & (sizeof(uint32_t) - 1), 0u);
  ASSERT_EQ(mapping_info.read_index_address & (sizeof(uint64_t) - 1), 0u);
  ASSERT_EQ(mapping_info.write_index_address & (sizeof(uint64_t) - 1), 0u);
  ASSERT_EQ(mapping_info.doorbell_address & (sizeof(uint64_t) - 1), 0u);
  ASSERT_GE(mapping_info.ring_byte_length, kUserQueueMemoryCommandByteCapacity);

  auto* read_index = reinterpret_cast<volatile uint64_t*>(
      static_cast<uintptr_t>(mapping_info.read_index_address));
  auto* write_index = reinterpret_cast<volatile uint64_t*>(
      static_cast<uintptr_t>(mapping_info.write_index_address));
  auto* doorbell = reinterpret_cast<volatile uint64_t*>(
      static_cast<uintptr_t>(mapping_info.doorbell_address));
  EXPECT_EQ(LoadAcquire(read_index), 0u);
  EXPECT_EQ(LoadAcquire(write_index), 0u);

  amdf_user_queue_status_t queue_status = {};
  queue_status.type = AMDF_STRUCTURE_TYPE_USER_QUEUE_STATUS;
  queue_status.structure_size = sizeof(queue_status);
  ASSERT_EQ(api_->user_queue_query_status(queue_, &queue_status),
            AMDF_STATUS_OK);
  EXPECT_EQ(queue_status.state, AMDF_QUEUE_STATE_ACTIVE);
  EXPECT_EQ(queue_status.reset_epoch, queue_info.reset_epoch);
  EXPECT_EQ(queue_status.producer_index, 0u);
  EXPECT_EQ(queue_status.consumed_index, 0u);
  EXPECT_EQ(queue_status.terminal_status, AMDF_STATUS_OK);

  // Lifecycle cases can close peer devices while this device's memory, queue,
  // and CPU views are live, before the GPU proves their continued usability.
  if (before_publication) {
    before_publication();
    if (::testing::Test::HasFatalFailure()) return;
  }

  auto* ring = reinterpret_cast<uint32_t*>(
      static_cast<uintptr_t>(mapping_info.ring_address));
  const EncodedUserQueueStream stream = commands_.encode(
      family_.format_features, ring, source_address, target_address);
  ASSERT_LE(stream.byte_length, mapping_info.ring_byte_length);
  StoreRelease(write_index, stream.published_index);
  StoreRelease(doorbell, stream.published_index);

  ASSERT_EQ(
      api_->user_queue_wait_consumed(queue_, stream.published_index,
                                     AMDF_TIMEOUT_INFINITE, UINT64_C(10000000)),
      AMDF_STATUS_OK);
  ASSERT_EQ(api_->user_queue_query_status(queue_, &queue_status),
            AMDF_STATUS_OK);
  EXPECT_EQ(queue_status.state, AMDF_QUEUE_STATE_ACTIVE);
  EXPECT_EQ(queue_status.reset_epoch, queue_info.reset_epoch);
  EXPECT_EQ(queue_status.producer_index, stream.published_index);
  EXPECT_EQ(queue_status.consumed_index, stream.published_index);
  EXPECT_EQ(queue_status.terminal_status, AMDF_STATUS_OK);

  ASSERT_EQ(api_->host_mapping_cache_control(
                target_mapping_, AMDF_HOST_CACHE_OPERATION_INVALIDATE, 0,
                kMemoryByteLength),
            AMDF_STATUS_OK);
  ASSERT_EQ(api_->host_mapping_cache_control(
                source_mapping_, AMDF_HOST_CACHE_OPERATION_INVALIDATE, 0,
                kMemoryByteLength),
            AMDF_STATUS_OK);
  for (size_t i = 0; i < kUserQueueMemoryElementCount; ++i) {
    EXPECT_EQ(target[i], expected[i]) << "target word " << i;
    EXPECT_EQ(source[i], expected[i]) << "source word " << i;
  }
  EXPECT_EQ(*completion, kUserQueueMemoryCompletionValue);
}

bool RunUserQueueMemoryCopies(const amdf_api_t* api,
                              const amdf_gpu_api_t* gpu_api,
                              const amdf_queue_family_info_t& family,
                              const UserQueueMemoryCommands& commands,
                              amdf_device_t* device,
                              amdf_memory_scope_t* system_scope) {
  UserQueueMemoryScenario scenario(api, gpu_api, family, commands, device,
                                   system_scope);
  scenario.RunCopiesBetweenExactAccessAttachments();
  return scenario.Release();
}

}  // namespace

amdf_status_t UserQueueMemoryTest::MatchGpuEndpoint(amdf_endpoint_t* endpoint,
                                                    bool* out_matches) {
  amdf_endpoint_info_t info = {};
  info.type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO;
  info.structure_size = sizeof(info);
  amdf_status_t status = api_->endpoint_query_info(endpoint, &info);
  if (!amdf_status_is_ok(status)) return status;
  for (uint32_t ordinal = 0; ordinal < info.queue_family_count; ++ordinal) {
    amdf_queue_family_info_t family = {};
    family.type = AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO;
    family.structure_size = sizeof(family);
    status = api_->endpoint_query_queue_family_info(endpoint, ordinal, &family);
    if (!amdf_status_is_ok(status)) return status;
    if (family.command_type == commands_.command_type &&
        family.format_version == commands_.format_version &&
        family.maximum_ring_byte_length >=
            kUserQueueMemoryCommandByteCapacity &&
        (family.format_features & commands_.required_format_features) ==
            commands_.required_format_features &&
        (family.publication_modes & AMDF_QUEUE_PUBLICATION_MODE_USER) != 0 &&
        (family.roles & kRequiredQueueRoles) == kRequiredQueueRoles &&
        (family.cache_operations & kRequiredCacheOperations) ==
            kRequiredCacheOperations &&
        (family.cache_transition_kinds & AMDF_CACHE_TRANSITION_KINDS_GLOBAL) !=
            0 &&
        (family.user_queue_capabilities &
         AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER) != 0 &&
        (family.producer_modes & AMDF_QUEUE_PRODUCER_MODE_BIT_SINGLE) != 0 &&
        (family.priority_capabilities &
         AMDF_QUEUE_PRIORITY_CAPABILITY_NORMAL) != 0) {
      family_ = family;
      *out_matches = true;
      return AMDF_STATUS_OK;
    }
  }
  *out_matches = false;
  return AMDF_STATUS_OK;
}

void UserQueueMemoryTest::RunCopiesBetweenExactAccessAttachments() {
  ASSERT_TRUE(RunUserQueueMemoryCopies(api_, gpu_api_, family_, commands_,
                                       device_, system_scope_));
}

void UserQueueMemoryTest::RunConcurrentDeviceCreationAndRecreation() {
  amdf_gpu_device_create_info_t create_info = {};
  create_info.type = AMDF_STRUCTURE_TYPE_GPU_DEVICE_CREATE_INFO;
  create_info.structure_size = sizeof(create_info);
  // This lifecycle case deliberately creates peers; all ordinary queue and
  // memory tests continue borrowing the one cached device.
  UserQueueMemoryScenario survivor(api_, gpu_api_, family_, commands_, device_,
                                   system_scope_);
  survivor.RunCopiesBetweenExactAccessAttachments([&]() {
    for (size_t generation = 0; generation < 2; ++generation) {
      std::array<amdf_device_t*, 2> peers = {};
      std::array<amdf_status_t, 2> statuses = {};
      std::array<std::thread, 2> threads;
      for (size_t i = 0; i < peers.size(); ++i) {
        threads[i] = std::thread([&, i]() {
          statuses[i] =
              gpu_api_->device_create(endpoint_, &create_info, &peers[i]);
        });
      }
      for (auto& thread : threads) thread.join();
      for (size_t i = 0; i < peers.size(); ++i) {
        EXPECT_EQ(statuses[i], AMDF_STATUS_OK);
        if (peers[i] == nullptr) continue;
        const bool released = RunUserQueueMemoryCopies(
            api_, gpu_api_, family_, commands_, peers[i], system_scope_);
        // An unretired queue retains its entire device chain on failure.
        if (released) {
          EXPECT_EQ(api_->device_destroy(peers[i]), AMDF_STATUS_OK);
        }
      }
      ASSERT_FALSE(HasFailure());
    }
  });
  ASSERT_TRUE(survivor.Release());
}
