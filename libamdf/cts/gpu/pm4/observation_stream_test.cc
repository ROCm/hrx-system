// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "libamdf/cts/gpu/pm4/command_fixture.h"
#include "util/wait.h"

namespace {

constexpr amdf_memory_access_t kReadWrite =
    AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
constexpr uint32_t kRecordCount = 16387;
constexpr uint32_t kGuard = 0x39c571a7u;
constexpr uint32_t kPublishedOffset = 0;
constexpr uint32_t kConsumedOffset = 64;
constexpr uint32_t kDoneOffset = 128;
constexpr uint32_t kReadbackOffset = 192;

struct RecordHeader {
  // Raw CP-clock sample before the confirmed payload transfer.
  uint64_t begin_ticks;
  // Raw CP-clock sample after DMA completion, without a frequency claim.
  uint64_t end_ticks;
  // One-based record identity, independent of its modulo storage offset.
  uint32_t sequence;
  // Complete payload extent following this header.
  uint32_t payload_byte_length;
  // Identity-dependent header guard, including the boundary-straddling case.
  uint64_t tag;
};
static_assert(sizeof(RecordHeader) == 32);

uint32_t SourceWord(uint32_t index) {
  return 0x712ac593u ^ (index * 0x9e3779b9u);
}

uint32_t PayloadWords(uint32_t ordinal, uint64_t capacity) {
  // Scale with native page geometry so even large-granule views wrap often.
  return 16 + 2 * ((ordinal * 37) % 33) + 2 * (capacity / 2048);
}

uint64_t RecordTag(uint32_t sequence) {
  return UINT64_C(0xa39bc675d18ef024) ^
         (uint64_t{sequence} * UINT64_C(0x010307090b0d1113));
}

class Pm4ObservationStreamTest
    : public Pm4CommandTest,
      public ::testing::WithParamInterface<amdf_queue_publication_modes_t> {
 protected:
  Pm4ObservationStreamTest()
      : Pm4CommandTest(AMDF_QUEUE_ROLE_ATOMIC, GetParam()) {}

  void TearDown() override {
    // A validation failure still releases every finite producer wait.
    // Backing stays owned until the queue and every alias are removed.
    if (queue_ != nullptr && control_ != nullptr) {
      GpuStoreRelease<uint32_t>(
          reinterpret_cast<uintptr_t>(control_->host.pointer) + kConsumedOffset,
          UINT32_MAX);
    }
    if (queue_ != nullptr) {
      ASSERT_TRUE(queue_->Release(api_));
    }
    if (view_ != nullptr) {
      ASSERT_EQ(api_->host_mapping_destroy(view_), AMDF_STATUS_OK);
    }
    if (reservation_ != nullptr) {
      ASSERT_EQ(api_->memory_reservation_destroy(reservation_), AMDF_STATUS_OK);
    }
    Pm4CommandTest::TearDown();
  }

  void PrepareControl();
  void PrepareRing();

  void CheckPair(const amdf_memory_site_t& producer,
                 const amdf_memory_site_t& consumer, bool atomic_control) {
    amdf_memory_pair_info_t pair = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
        .structure_size = sizeof(pair),
    };
    ASSERT_EQ(api_->memory_query_pair_info(&producer, &consumer, &pair),
              AMDF_STATUS_OK);
    ASSERT_NE(pair.flags & AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE, 0u);
    ASSERT_EQ(pair.release.operation,
              producer.kind == AMDF_MEMORY_SITE_KIND_HOST
                  ? AMDF_CACHE_OPERATION_NONE
                  : AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM);
    ASSERT_EQ(pair.acquire.operation,
              consumer.kind == AMDF_MEMORY_SITE_KIND_HOST
                  ? AMDF_CACHE_OPERATION_NONE
                  : AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM);
    ASSERT_EQ(pair.release.host_operation, AMDF_HOST_CACHE_OPERATION_NONE);
    ASSERT_EQ(pair.acquire.host_operation, AMDF_HOST_CACHE_OPERATION_NONE);
    for (const auto* transition : {&pair.release, &pair.acquire}) {
      const bool global = transition->operation != AMDF_CACHE_OPERATION_NONE;
      ASSERT_EQ(transition->kind, global ? AMDF_CACHE_TRANSITION_KIND_GLOBAL
                                         : AMDF_CACHE_TRANSITION_KIND_NONE);
      ASSERT_EQ(transition->executor,
                global ? AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE
                       : AMDF_CACHE_TRANSITION_EXECUTOR_NONE);
      ASSERT_EQ(transition->host_instruction, AMDF_HOST_CACHE_INSTRUCTION_NONE);
      ASSERT_EQ(transition->host_fence_before, AMDF_HOST_CACHE_FENCE_NONE);
      ASSERT_EQ(transition->host_fence_after, AMDF_HOST_CACHE_FENCE_NONE);
      ASSERT_EQ(transition->range_granularity, 0u);
    }
    if (atomic_control) {
      ASSERT_EQ(pair.atomic_reach.scope_32, AMDF_ATOMIC_SCOPE_SYSTEM);
    }
  }

  // Case-owned execution removed before aliases and backing are released.
  GpuCommandQueue* queue_ = nullptr;
  // One physical payload allocation, borrowed from the fixture.
  GpuMemory* backing_ = nullptr;
  // Separate coherent publication and credit cells, borrowed from the fixture.
  GpuMemory* control_ = nullptr;
  // Stable CPU/GPU virtual address owners for three repeated source ranges.
  amdf_memory_reservation_t* reservation_ = nullptr;
  // CPU projection borrowing the reservation through final consumption.
  amdf_host_mapping_t* view_ = nullptr;
  // Physical payload capacity, independent of the number of observations.
  uint64_t capacity_ = 0;
  // Middle CPU alias, permitting contiguous accesses across either boundary.
  uint8_t* host_ring_ = nullptr;
  // Middle GPU alias, independent of the CPU virtual address.
  uint64_t gpu_ring_ = 0;
};

void Pm4ObservationStreamTest::PrepareControl() {
  // Packet support alone does not establish the CPU-shared atomic route used
  // by publication and credits. Admit that route before allocating storage.
  const auto& atomics = family_.atomic_capabilities;
  if ((atomics.operations_32 & atomics.operations_without_dispatch_32 &
       AMDF_ATOMIC_OPERATION_STORE) == 0) {
    GTEST_SKIP() << "queue does not encode native 32-bit STORE";
  }
  const amdf_memory_device_access_t attachment = {
      device_,
      {
          .access = kReadWrite,
          .flags =
              AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
      }};
  amdf_memory_scope_info_t scope = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
      .structure_size = sizeof(scope),
  };
  ASSERT_EQ(api_->memory_scope_query_info(system_scope_, &scope),
            AMDF_STATUS_OK);
  for (uint32_t ordinal = 0; ordinal < scope.memory_profile_count; ++ordinal) {
    amdf_memory_profile_t profile = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
        .structure_size = sizeof(profile),
    };
    amdf_memory_access_capabilities_t capabilities = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
        .structure_size = sizeof(capabilities),
    };
    const amdf_status_t status =
        QueryMemoryProfile(system_scope_, ordinal, attachment.requirements,
                           &profile, &capabilities);
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
      continue;
    }
    ASSERT_EQ(status, AMDF_STATUS_OK);
    constexpr auto kRoles =
        AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP;
    if ((profile.roles & kRoles) != kRoles ||
        (profile.supported_flags & AMDF_MEMORY_FLAG_HOST_VISIBLE) == 0 ||
        (capabilities.atomic_operations_32 & AMDF_ATOMIC_OPERATION_STORE) ==
            0) {
      continue;
    }

    amdf_memory_profile_site_t host = {.kind = AMDF_MEMORY_SITE_KIND_HOST};
    host.value.host_access =
        AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
    amdf_memory_profile_site_t gpu = {.kind = AMDF_MEMORY_SITE_KIND_DEVICE};
    gpu.value.device.access_ordinal = 0;
    gpu.value.device.queue_family_ordinal = family_.ordinal;
    bool system_reach = true;
    for (bool host_producer : {true, false}) {
      const amdf_memory_profile_pair_query_t query = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE_PAIR_QUERY,
          .structure_size = sizeof(query),
          .memory_profile_ordinal = ordinal,
          .access_count = 1,
          .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
          .accesses = &attachment,
          .producer = host_producer ? host : gpu,
          .consumer = host_producer ? gpu : host,
      };
      amdf_memory_pair_info_t pair = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
          .structure_size = sizeof(pair),
      };
      const amdf_status_t pair_status =
          api_->memory_scope_query_pair_info(system_scope_, &query, &pair);
      if (pair_status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
        system_reach = false;
        break;
      }
      ASSERT_EQ(pair_status, AMDF_STATUS_OK);
      if (pair.atomic_reach.scope_32 != AMDF_ATOMIC_SCOPE_SYSTEM) {
        system_reach = false;
        break;
      }
    }
    if (!system_reach) {
      continue;
    }
    const amdf_memory_create_info_t create = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
        .structure_size = sizeof(create),
        .memory_profile_ordinal = ordinal,
        .access_count = 1,
        .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
        .byte_length = 4096,
        .minimum_alignment = 4096,
        .accesses = &attachment,
    };
    ASSERT_NO_FATAL_FAILURE(CreateMemory(system_scope_, create, &control_));
    return;
  }
  GTEST_SKIP() << "no coherent SYSTEM profile supplies CPU/PM4 32-bit atomic "
                  "STORE reach";
}

void Pm4ObservationStreamTest::PrepareRing() {
  amdf_memory_reservation_capabilities_t host_capabilities = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CAPABILITIES,
      .structure_size = sizeof(host_capabilities),
  };
  amdf_memory_reservation_capabilities_t gpu_capabilities = host_capabilities;
  ASSERT_EQ(api_->instance_query_host_memory_reservation_capabilities(
                instance_, &host_capabilities),
            AMDF_STATUS_OK);
  ASSERT_EQ(api_->device_query_memory_reservation_capabilities(
                device_, 0, &gpu_capabilities),
            AMDF_STATUS_OK);
  const amdf_memory_device_access_t attachment = {
      device_,
      {
          .access = kReadWrite | gpu_capabilities.required_access,
          .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS |
                   AMDF_MEMORY_FLAG_HOST_COHERENT |
                   AMDF_MEMORY_FLAG_MAPPING_SOURCE,
      }};
  const amdf_memory_flags_t backing_flags =
      AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_HOST_MAPPING_SOURCE |
      host_capabilities.required_source_flags |
      gpu_capabilities.required_source_flags;
  amdf_memory_scope_info_t scope = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
      .structure_size = sizeof(scope),
  };
  ASSERT_EQ(api_->memory_scope_query_info(system_scope_, &scope),
            AMDF_STATUS_OK);
  amdf_memory_profile_t profile = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
      .structure_size = sizeof(profile),
  };
  amdf_memory_access_capabilities_t access_capabilities = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
      .structure_size = sizeof(access_capabilities),
  };
  uint32_t profile_ordinal = AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN;
  for (uint32_t ordinal = 0; ordinal < scope.memory_profile_count; ++ordinal) {
    const auto status = api_->memory_scope_query_device_profile(
        system_scope_, ordinal, 1, &attachment, &profile, &access_capabilities);
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
      continue;
    }
    ASSERT_EQ(status, AMDF_STATUS_OK);
    constexpr auto kRoles =
        AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP;
    if ((profile.roles & kRoles) == kRoles &&
        (backing_flags & ~profile.supported_flags) == 0) {
      profile_ordinal = ordinal;
      break;
    }
  }
  if (profile_ordinal == AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN) {
    GTEST_SKIP() << "SYSTEM backing does not supply coherent CPU/GPU aliases";
  }
  capacity_ = std::max({UINT64_C(4096), host_capabilities.mapping_granularity,
                        gpu_capabilities.mapping_granularity,
                        host_capabilities.reservation_granularity,
                        gpu_capabilities.reservation_granularity,
                        profile.allocation.byte_length_granularity,
                        profile.allocation.native_byte_length_granularity});
  const amdf_memory_create_info_t create = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
      .structure_size = sizeof(create),
      .memory_profile_ordinal = profile_ordinal,
      .access_count = 1,
      .required_flags = backing_flags,
      .byte_length = capacity_,
      .minimum_alignment = profile.allocation.minimum_alignment,
      .accesses = &attachment,
  };
  ASSERT_NO_FATAL_FAILURE(CreateMemory(system_scope_, create, &backing_));
  const amdf_memory_reservation_access_t accesses[] = {
      {nullptr, host_capabilities.address_domain_ordinal,
       kReadWrite | host_capabilities.required_access, capacity_},
      {device_, gpu_capabilities.address_domain_ordinal,
       attachment.requirements.access, capacity_},
  };
  const amdf_memory_reservation_create_info_t reservation_create = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CREATE_INFO,
      .structure_size = sizeof(reservation_create),
      .byte_length = 3 * capacity_,
      .minimum_alignment = capacity_,
      .access_count = 2,
      .accesses = accesses,
  };
  ASSERT_EQ(api_->memory_reservation_create(instance_, &reservation_create,
                                            &reservation_),
            AMDF_STATUS_OK);
  for (uint32_t ordinal = 0; ordinal < 2; ++ordinal) {
    amdf_memory_reservation_access_info_t info = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_ACCESS_INFO,
        .structure_size = sizeof(info),
    };
    ASSERT_EQ(api_->memory_reservation_query_access_info(reservation_, ordinal,
                                                         &info),
              AMDF_STATUS_OK);
    ASSERT_EQ(info.mapping_access_ordinal, ordinal);
    if (ordinal == 1) {
      ASSERT_EQ(info.address_kind, AMDF_MEMORY_ADDRESS_GPU);
      gpu_ring_ = info.address + capacity_;
    }
    std::array<amdf_memory_mapping_update_t, 3> aliases = {};
    for (size_t i = 0; i < aliases.size(); ++i) {
      aliases[i] = {
          .target_byte_offset = i * capacity_,
          .byte_length = capacity_,
          .memory = backing_->memory,
          .access = accesses[ordinal].access,
      };
    }
    ASSERT_EQ(api_->memory_reservation_update(reservation_, ordinal,
                                              aliases.size(), aliases.data()),
              AMDF_STATUS_OK);
  }
  const amdf_memory_map_info_t map = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO,
      .structure_size = sizeof(map),
      .byte_length = 3 * capacity_,
      .flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
  };
  ASSERT_EQ(api_->memory_reservation_map(reservation_, 0, &map, &view_),
            AMDF_STATUS_OK);
  amdf_host_mapping_info_t host = {
      .type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO,
      .structure_size = sizeof(host),
  };
  ASSERT_EQ(api_->host_mapping_query_info(view_, &host), AMDF_STATUS_OK);
  host_ring_ = static_cast<uint8_t*>(host.pointer) + capacity_;
  amdf_memory_site_t host_site = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_SITE,
      .structure_size = sizeof(host_site),
      .kind = AMDF_MEMORY_SITE_KIND_HOST,
  };
  host_site.value.host = {view_, backing_->memory};
  const auto gpu_site = backing_->DeviceSite(family_.ordinal);
  ASSERT_NO_FATAL_FAILURE(CheckPair(gpu_site, host_site, false));
  ASSERT_NO_FATAL_FAILURE(CheckPair(host_site, gpu_site, false));
}

TEST_P(Pm4ObservationStreamTest, DrainsMirroredRecordsBeforeQueueCompletion) {
  ASSERT_NO_FATAL_FAILURE(PrepareControl());
  if (IsSkipped()) {
    return;
  }
  ASSERT_NO_FATAL_FAILURE(PrepareRing());
  if (IsSkipped()) {
    return;
  }
  const uint64_t start_position = capacity_ - 16;
  std::vector<uint32_t> expected_backing(capacity_ / 4, kGuard);
  std::memcpy(backing_->host.pointer, expected_backing.data(), capacity_);
  ASSERT_EQ(
      api_->host_mapping_cache_control(
          backing_->mapping, AMDF_HOST_CACHE_OPERATION_FLUSH, 0, capacity_),
      AMDF_STATUS_OK);
  GpuMemory* source = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ, capacity_, &source));
  const uint32_t source_word_count = capacity_ / 4;
  for (uint32_t i = 0; i < source_word_count; ++i) {
    static_cast<uint32_t*>(source->host.pointer)[i] = SourceWord(i);
  }
  ASSERT_EQ(api_->host_mapping_cache_control(
                source->mapping, AMDF_HOST_CACHE_OPERATION_FLUSH, 0, capacity_),
            AMDF_STATUS_OK);
  std::array<uint32_t, 1024> expected_control;
  expected_control.fill(kGuard);
  expected_control[kPublishedOffset / 4] = 0;
  expected_control[kConsumedOffset / 4] = start_position;
  expected_control[kDoneOffset / 4] = 0;
  std::memcpy(control_->host.pointer, expected_control.data(), 4096);
  const uint64_t host_control_address =
      reinterpret_cast<uintptr_t>(control_->host.pointer);
  for (const uint32_t offset :
       {kPublishedOffset, kConsumedOffset, kDoneOffset}) {
    std::construct_at(
        reinterpret_cast<uint32_t*>(host_control_address + offset),
        expected_control[offset / 4]);
  }

  const auto host_control = control_->HostSite();
  const auto gpu_control = control_->DeviceSite(family_.ordinal);
  ASSERT_NO_FATAL_FAILURE(CheckPair(host_control, gpu_control, true));
  ASSERT_NO_FATAL_FAILURE(CheckPair(gpu_control, host_control, true));
  ASSERT_NE(
      control_->access_info.atomic_operations_32 & AMDF_ATOMIC_OPERATION_STORE,
      0u);

  // Two immutable blocks respect the native IB length field. Their chain keeps
  // one submission active while observation storage repeatedly wraps and
  // drains.
  constexpr uint64_t kBlockBytes = 3 * 1024 * 1024;
  constexpr uint32_t kFirstRecordCount = kRecordCount / 2;
  GpuMemory* command_backing = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_EXECUTE,
                   2 * kBlockBytes, &command_backing));
  auto* command_words = static_cast<uint32_t*>(command_backing->host.pointer);
  Pm4CommandWriter first(command_words, *pm4_profile_);
  Pm4CommandWriter second(command_words + kBlockBytes / 4, *pm4_profile_);
  first.SystemBarrier();
  uint64_t position = start_position;
  uint32_t crossing_count = 0;
  for (uint32_t ordinal = 0; ordinal < kRecordCount; ++ordinal) {
    auto& commands = ordinal < kFirstRecordCount ? first : second;
    const uint32_t payload_words = PayloadWords(ordinal, capacity_);
    const uint32_t record_bytes = sizeof(RecordHeader) + payload_words * 4;
    ASSERT_LT(record_bytes, capacity_);
    const uint32_t source_offset =
        (ordinal * 13) % (source_word_count - payload_words);
    const uint64_t target = gpu_ring_ + position % capacity_;
    crossing_count += position % capacity_ + record_bytes > capacity_;
    position += record_bytes;
    ASSERT_LT(position, UINT32_MAX);
    commands.WaitMemory32(control_->device_address + kConsumedOffset,
                          static_cast<uint32_t>(position - capacity_),
                          Pm4MemoryComparison::kGreaterOrEqual);
    commands.CopyGpuClock64(target);
    const uint64_t tag = RecordTag(ordinal + 1);
    const uint32_t header_words[] = {ordinal + 1, payload_words * 4,
                                     static_cast<uint32_t>(tag),
                                     static_cast<uint32_t>(tag >> 32)};
    commands.WriteData(target + offsetof(RecordHeader, sequence), header_words,
                       4);
    commands.DmaCopy(source->device_address + source_offset * 4,
                     target + sizeof(RecordHeader), payload_words * 4);
    commands.WaitDma();
    commands.CopyGpuClock64(target + offsetof(RecordHeader, end_ticks));
    commands.SystemBarrier();
    // Commit identifies a complete ordered prefix, never a reserved header.
    commands.Store32(control_->device_address + kPublishedOffset, ordinal + 1);
    // The atomic cell can be a GL2 client. Finish its host transition before
    // the next update to this shared committed-prefix cell.
    commands.AcquireFromSystem();
  }
  const uint32_t final_position = position;
  second.WaitMemory32(control_->device_address + kConsumedOffset,
                      final_position, Pm4MemoryComparison::kGreaterOrEqual);
  // Both neighboring aliases participate in actual GPU reads as well.
  second.DmaCopy(gpu_ring_ - 16, control_->device_address + kReadbackOffset,
                 64);
  second.DmaCopy(gpu_ring_ + capacity_ - 16,
                 control_->device_address + kReadbackOffset + 64, 64);
  second.WaitDma();
  second.SystemBarrier();
  second.Store32(control_->device_address + kDoneOffset, 1);
  second.AcquireFromSystem();
  second.PadToEightWords();
  ASSERT_LT(second.word_count() * 4, kBlockBytes);
  first.PadToEightWords(4);
  first.ChainIndirectBuffer({command_backing->device_address + kBlockBytes,
                             static_cast<uint32_t>(second.word_count())});
  ASSERT_LT(first.word_count() * 4, kBlockBytes);
  ASSERT_LE(first.word_count(), 0xfffffu);
  ASSERT_LE(second.word_count(), 0xfffffu);
  ASSERT_GT(crossing_count, 16u);
  ASSERT_EQ(api_->host_mapping_cache_control(command_backing->mapping,
                                             AMDF_HOST_CACHE_OPERATION_FLUSH, 0,
                                             2 * kBlockBytes),
            AMDF_STATUS_OK);
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&queue_));
  ASSERT_EQ(api_->host_mapping_cache_control(
                control_->mapping, AMDF_HOST_CACHE_OPERATION_FLUSH, 0, 4096),
            AMDF_STATUS_OK);
  if (queue_->publication_mode() == AMDF_QUEUE_PUBLICATION_MODE_USER) {
    ASSERT_GE(queue_->words().size(), 8u);
    Pm4CommandWriter root(queue_->words().data(), *pm4_profile_);
    root.CallIndirectBuffer(command_backing->device_address,
                            static_cast<uint32_t>(first.word_count()));
    root.PadToEightWords();
    ASSERT_NO_FATAL_FAILURE(queue_->Publish(api_, gpu_api_, root.word_count()));
  } else {
    const amdf_gpu_kernel_command_t root = {
        .memory = command_backing->memory,
        .byte_length = first.word_count() * 4,
    };
    ASSERT_NO_FATAL_FAILURE(queue_->Submit(gpu_api_, root));
  }

  uint64_t prior_ticks = 0;
  uint64_t payload_digest = 0;
  uint64_t expected_digest = 0;
  uint64_t first_ticks = 0;
  position = start_position;
  const amdf::cts::WaitDeadline deadline;
  for (uint32_t ordinal = 0; ordinal < kRecordCount; ++ordinal) {
    ASSERT_TRUE(amdf::cts::WaitUntil(
        "PM4 observation record",
        [&] {
          return GpuLoadAcquire<uint32_t>(host_control_address +
                                          kPublishedOffset) >= ordinal + 1;
        },
        deadline))
        << "record " << ordinal + 1 << ", published "
        << GpuLoadAcquire<uint32_t>(host_control_address + kPublishedOffset);
    if (ordinal == 0) {
      EXPECT_EQ(GpuLoadAcquire<uint32_t>(host_control_address + kDoneOffset),
                0u);
    }
    const uint32_t payload_words = PayloadWords(ordinal, capacity_);
    const uint32_t record_bytes = sizeof(RecordHeader) + payload_words * 4;
    const uint32_t source_offset =
        (ordinal * 13) % (source_word_count - payload_words);
    const uint8_t* record = host_ring_ + position % capacity_;
    RecordHeader header;
    std::memcpy(&header, record, sizeof(header));
    EXPECT_EQ(header.sequence, ordinal + 1) << ordinal;
    EXPECT_EQ(header.payload_byte_length, payload_words * 4) << ordinal;
    EXPECT_EQ(header.tag, RecordTag(ordinal + 1)) << ordinal;
    EXPECT_LE(prior_ticks, header.begin_ticks) << ordinal;
    EXPECT_LE(header.begin_ticks, header.end_ticks) << ordinal;
    if (ordinal == 0) {
      first_ticks = header.begin_ticks;
    }
    prior_ticks = header.end_ticks;
    const RecordHeader expected_header = {header.begin_ticks, header.end_ticks,
                                          ordinal + 1, payload_words * 4,
                                          RecordTag(ordinal + 1)};
    const auto* header_bytes =
        reinterpret_cast<const uint8_t*>(&expected_header);
    auto* backing_bytes = reinterpret_cast<uint8_t*>(expected_backing.data());
    for (size_t i = 0; i < sizeof(expected_header); ++i) {
      backing_bytes[(position + i) % capacity_] = header_bytes[i];
    }
    const auto* payload =
        reinterpret_cast<const uint32_t*>(record + sizeof(header));
    for (uint32_t i = 0; i < payload_words; ++i) {
      const uint32_t expected = SourceWord(source_offset + i);
      EXPECT_EQ(payload[i], expected) << "record " << ordinal << " word " << i;
      payload_digest = payload_digest * 33 + payload[i];
      expected_digest = expected_digest * 33 + expected;
      expected_backing[((position + sizeof(header)) / 4 + i) %
                       expected_backing.size()] = expected;
    }
    position += record_bytes;
    // The sink has consumed the complete record before returning its bytes.
    GpuStoreRelease<uint32_t>(host_control_address + kConsumedOffset,
                              static_cast<uint32_t>(position));
    if (HasFailure()) {
      break;
    }
  }
  const bool records_valid = !HasFailure();
  if (!records_valid) {
    GpuStoreRelease<uint32_t>(host_control_address + kConsumedOffset,
                              final_position);
  }
  ASSERT_TRUE(GpuWaitEqual<uint32_t>(host_control_address + kDoneOffset, 1));
  ASSERT_NO_FATAL_FAILURE(queue_->WaitRetired(api_));
  if (!records_valid) {
    return;
  }
  EXPECT_GT(prior_ticks, first_ticks);
  EXPECT_EQ(payload_digest, expected_digest);
  EXPECT_EQ(std::memcmp(host_ring_, expected_backing.data(), capacity_), 0);
  EXPECT_EQ(
      std::memcmp(backing_->host.pointer, expected_backing.data(), capacity_),
      0);
  expected_control[kPublishedOffset / 4] = kRecordCount;
  expected_control[kConsumedOffset / 4] = final_position;
  expected_control[kDoneOffset / 4] = 1;
  for (uint32_t i = 0; i < 32; ++i) {
    expected_control[kReadbackOffset / 4 + i] =
        expected_backing[(expected_backing.size() - 4 + i % 16) %
                         expected_backing.size()];
  }
  EXPECT_EQ(std::memcmp(control_->host.pointer, expected_control.data(), 4096),
            0);
  for (uint32_t i = 0; i < source_word_count; ++i) {
    EXPECT_EQ(static_cast<const uint32_t*>(source->host.pointer)[i],
              SourceWord(i))
        << i;
  }
  RecordProperty("stream_record_count", kRecordCount);
  RecordProperty("stream_control_profile",
                 control_->creation.memory_profile_ordinal);
  RecordProperty("stream_backing_profile",
                 backing_->creation.memory_profile_ordinal);
  RecordProperty("stream_payload_capacity_bytes", std::to_string(capacity_));
  RecordProperty("stream_virtual_bytes_per_projection",
                 std::to_string(3 * capacity_));
  RecordProperty("stream_observation_bytes",
                 std::to_string(position - start_position));
  RecordProperty("stream_boundary_crossings", crossing_count);
  RecordProperty(
      "stream_command_bytes",
      std::to_string((first.word_count() + second.word_count()) * 4));
  RecordProperty("stream_begin_ticks", std::to_string(first_ticks));
  RecordProperty("stream_end_ticks", std::to_string(prior_ticks));
  RecordProperty("stream_payload_digest", std::to_string(payload_digest));
}

INSTANTIATE_TEST_SUITE_P(Publication, Pm4ObservationStreamTest,
                         ::testing::Values(AMDF_QUEUE_PUBLICATION_MODE_USER,
                                           AMDF_QUEUE_PUBLICATION_MODE_KERNEL),
                         [](const auto& info) {
                           return info.param == AMDF_QUEUE_PUBLICATION_MODE_USER
                                      ? "User"
                                      : "Kernel";
                         });

}  // namespace
