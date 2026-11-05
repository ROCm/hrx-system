// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/gpu/pm4/mapping_fixture.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "libamdf/cts/gpu/kernels/byte_copy_unaligned.h"
#include "libamdf/cts/gpu/kernels/byte_copy_unaligned_kernels.h"

uint32_t Pm4MappingFixture::TaggedWord(uint32_t owner, uint64_t word) {
  return 0x93b57fd1u ^
         (owner * 0x01010101u + static_cast<uint32_t>(word) * 0x9e3779b9u);
}

void Pm4MappingFixture::TearDown() {
  // Remove execution before addresses, then addresses before backing. Native
  // failure never turns into an implicit destructor retry or backing reuse.
  if (cleanup_failed_) {
    return;
  }
  if (queue_ != nullptr) {
    ASSERT_TRUE(queue_->Release(api_));
  }
  for (auto** owner : {&reservation_, &second_reservation_}) {
    if (*owner != nullptr) {
      ASSERT_NO_FATAL_FAILURE(ReleaseReservation(owner));
    }
  }
  for (auto& backing : backing_) {
    ASSERT_NO_FATAL_FAILURE(ReleaseBacking(&backing));
  }
  for (void* storage : caller_storage_) {
    std::free(storage);
  }
  if (second_device_ != nullptr) {
    const amdf_status_t status = api_->device_destroy(second_device_);
    second_device_ = nullptr;
    ASSERT_EQ(status, AMDF_STATUS_OK);
  }
  Pm4DispatchTest::TearDown();
}

void Pm4MappingFixture::ReleaseReservation(amdf_memory_reservation_t** owner) {
  const amdf_status_t status = api_->memory_reservation_destroy(*owner);
  *owner = nullptr;
  cleanup_failed_ = !amdf_status_is_ok(status);
  ASSERT_EQ(status, AMDF_STATUS_OK);
}

void Pm4MappingFixture::ReleaseBacking(GpuMemory* backing) {
  cleanup_failed_ = !backing->Release(api_);
  ASSERT_FALSE(cleanup_failed_);
}

void Pm4MappingFixture::Prepare(amdf_memory_scope_t* scope,
                                uint64_t command_byte_length,
                                uint64_t secondary_mapping_granularity) {
  ASSERT_NE(scope, nullptr);
  scope_ = scope;
  amdf_memory_reservation_capabilities_t capabilities = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CAPABILITIES,
      .structure_size = sizeof(capabilities),
  };
  const amdf_status_t status =
      api_->device_query_memory_reservation_capabilities(device_, 0,
                                                         &capabilities);
  if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
    GTEST_SKIP() << "host-completed GPU mapping updates are unavailable";
  }
  ASSERT_EQ(status, AMDF_STATUS_OK);
  constexpr uint64_t kReservationByteLength = UINT64_C(8) << 40;
  constexpr uint64_t kAlignment = UINT64_C(128) << 20;
  ASSERT_GE(capabilities.maximum_byte_length, kReservationByteLength);
  ASSERT_GE(capabilities.maximum_alignment, kAlignment);
  ASSERT_EQ(kReservationByteLength % capabilities.reservation_granularity, 0u);
  ASSERT_EQ(kSlabByteLength % capabilities.mapping_granularity, 0u);
  required_access_ = capabilities.required_access;
  access_ = kReadWrite | required_access_;
  amdf_memory_reservation_capabilities_t host_capabilities = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CAPABILITIES,
      .structure_size = sizeof(host_capabilities),
  };
  if (second_device_ != nullptr) {
    ASSERT_EQ(api_->instance_query_host_memory_reservation_capabilities(
                  instance_, &host_capabilities),
              AMDF_STATUS_OK);
    ASSERT_GE(host_capabilities.maximum_byte_length, kReservationByteLength);
    ASSERT_GE(host_capabilities.maximum_alignment, kAlignment);
  }
  // Interleaving a CPU projection keeps logical GPU aliases independent of
  // their caller ordinals while preserving the shared GPU mapping group.
  const amdf_memory_reservation_access_t accesses[] = {
      {device_, 0, access_, 0},
      {nullptr, host_capabilities.address_domain_ordinal,
       kReadWrite | host_capabilities.required_access, 0},
      {second_device_, 0, access_, secondary_mapping_granularity}};
  const amdf_memory_reservation_create_info_t create = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CREATE_INFO,
      .structure_size = sizeof(create),
      .byte_length = kReservationByteLength,
      .minimum_alignment = kAlignment,
      .access_count = second_device_ == nullptr ? 1u : 3u,
      .accesses = accesses,
  };
  ASSERT_EQ(api_->memory_reservation_create(instance_, &create, &reservation_),
            AMDF_STATUS_OK);
  reservation_info_.type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_ACCESS_INFO;
  reservation_info_.structure_size = sizeof(reservation_info_);
  ASSERT_EQ(api_->memory_reservation_query_access_info(reservation_, 0,
                                                       &reservation_info_),
            AMDF_STATUS_OK);
  ASSERT_EQ(reservation_info_.address % kAlignment, 0u);
  ASSERT_EQ(reservation_info_.byte_length, kReservationByteLength);
  RecordProperty("mapping_reserved_byte_length",
                 std::to_string(kReservationByteLength));
  RecordProperty("mapping_base", std::to_string(reservation_info_.address));
  RecordProperty("mapping_granularity",
                 std::to_string(reservation_info_.mapping_granularity));

  const auto* kernel =
      kernels::byte_copy_unaligned::kKernels.Find(gpu_endpoint_info_);
  ASSERT_NE(kernel, nullptr) << "missing compiled byte-copy kernel";
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ, kLargePayloadByteLength, &upload_));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(kReadWrite, kLargePayloadByteLength, &readback_));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ, 4096, &arguments_));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(kReadWrite, 4096, &completion_));
  std::memset(completion_->host.pointer, 0, completion_->info.byte_length);
  program_ = {
      0,
      kernel->program.resource1,
      kernel->program.resource2,
      kernel->program.resource3,
      kernel->group_segment_byte_length,
      kernel->wavefront_size,
      {kernel->workgroup_size(), 1, 1},
  };
  ASSERT_NO_FATAL_FAILURE(PrepareProgram(
      kernel->executable, kernel->entry_byte_offset, &program_, "mapping"));
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&queue_, command_byte_length));
}

void Pm4MappingFixture::CreateBacking(uint64_t byte_length,
                                      GpuMemory** out_memory,
                                      amdf_device_t* consumer) {
  if (consumer == nullptr) {
    consumer = device_;
  }
  const bool system = scope_ == system_scope_;
  const amdf_memory_device_access_t attachment = {
      consumer,
      {
          .access = access_,
          .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS |
                   AMDF_MEMORY_FLAG_MAPPING_SOURCE |
                   (system ? AMDF_MEMORY_FLAG_HOST_COHERENT : UINT64_C(0)),
      },
  };
  amdf_memory_create_info_t create = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
      .structure_size = sizeof(create),
  };
  create.required_flags =
      system ? AMDF_MEMORY_FLAG_HOST_VISIBLE : AMDF_MEMORY_FLAG_DEVICE_LOCAL;
  create.memory_profile_ordinal = FindGpuMemoryProfileOrdinal(
      api_, scope_, consumer, AMDF_MEMORY_PROFILE_ROLE_CREATE,
      create.required_flags, attachment.requirements);
  ASSERT_NE(create.memory_profile_ordinal, AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
  create.byte_length = byte_length;
  create.minimum_alignment = reservation_info_.mapping_granularity;
  create.access_count = 1;
  create.accesses = &attachment;
  auto& memory = backing_.emplace_back();
  ASSERT_NO_FATAL_FAILURE(memory.Initialize(api_, scope_, create));
  *out_memory = &memory;
}

void Pm4MappingFixture::Copy(uint64_t source, uint64_t target,
                             uint64_t byte_length) {
  constexpr uint32_t kGridSize = 1024;
  const kernels::byte_copy_unaligned::Arguments arguments = {
      source, target, byte_length, kGridSize, 1, program_.workgroup_size[0],
  };
  std::memset(arguments_->host.pointer, 0, arguments_->info.byte_length);
  std::memcpy(
      arguments_->host.pointer, &arguments,
      offsetof(kernels::byte_copy_unaligned::Arguments, workgroup_size_x) +
          sizeof(uint32_t));
  // Leave room for the complete fixed compute recipe before writing it. The
  // finite test stream never wraps or overwrites commands already published.
  ASSERT_GE(queue_->words().size() - command_word_count_, 128u);
  Pm4CommandWriter commands(queue_->words().data() + command_word_count_,
                            *pm4_profile_);
  commands.SystemBarrier();
  commands.BindCompute(program_, arguments_->device_address,
                       family_.format_features);
  commands.Dispatch(program_, kGridSize, 1, 1);
  commands.SystemBarrier();
  commands.WriteData32(completion_->device_address, ++completed_);
  commands.PadToEightWords();
  command_word_count_ += commands.word_count();
  ASSERT_LE(command_word_count_, queue_->words().size());
  ASSERT_NO_FATAL_FAILURE(queue_->Publish(api_, gpu_api_, command_word_count_));
  ASSERT_TRUE(GpuWaitEqual<uint32_t>(
      reinterpret_cast<uintptr_t>(completion_->host.pointer), completed_));
  // Completion is the program's final system-visible marker. There is no
  // allocation, mapping, native query or queue reconstruction on this edge.
  ASSERT_NO_FATAL_FAILURE(queue_->WaitRetired(api_));
}

void Pm4MappingFixture::Verify(uint64_t address, uint64_t byte_length,
                               uint32_t owner) {
  auto* words = static_cast<uint32_t*>(readback_->host.pointer);
  std::fill_n(words, byte_length / sizeof(uint32_t), 0u);
  ASSERT_NO_FATAL_FAILURE(
      Copy(address, readback_->device_address, byte_length));
  const uint64_t count = byte_length / sizeof(uint32_t);
  uint64_t incorrect = 0;
  uint64_t first = 0;
  for (uint64_t i = 0; i < count; ++i) {
    if (words[i] != TaggedWord(owner, i)) {
      if (incorrect == 0) {
        first = i;
      }
      ++incorrect;
    }
  }
  ASSERT_EQ(incorrect, 0u) << "owner=" << owner << " first_word=" << first
                           << " observed=" << words[first]
                           << " expected=" << TaggedWord(owner, first);
}

void Pm4MappingFixture::WriteAndVerify(uint64_t offset, uint64_t byte_length,
                                       uint32_t owner,
                                       const GpuMemory& backing) {
  auto* words = static_cast<uint32_t*>(upload_->host.pointer);
  for (uint64_t i = 0; i < byte_length / sizeof(uint32_t); ++i) {
    words[i] = TaggedWord(owner, i);
  }
  const uint64_t address =
      reservation_info_.address + offset + kPayloadByteOffset;
  ASSERT_NO_FATAL_FAILURE(Copy(upload_->device_address, address, byte_length));
  ASSERT_NO_FATAL_FAILURE(Verify(address, byte_length, owner));
  // A read through the same stale alias could mask a stale write. The fixed
  // original source projection must observe this generation independently.
  ASSERT_NO_FATAL_FAILURE(
      Verify(backing.device_address + kPayloadByteOffset, byte_length, owner));
}

void Pm4MappingFixture::WriteSparseGeneration(
    std::span<const amdf::testing::SparseRange> ranges,
    const std::array<uint64_t, 2>& addresses, uint32_t generation) {
  auto* input = static_cast<uint32_t*>(upload_->host.pointer);
  uint64_t payload_position = 0;
  for (const auto& range : ranges) {
    for (uint64_t offset = 0; offset < range.byte_length;) {
      const uint64_t length =
          std::min(kLargePayloadByteLength, range.byte_length - offset);
      const uint64_t first_word = (payload_position + offset) / 4;
      for (uint64_t i = 0; i < length / 4; ++i) {
        input[i] = amdf::testing::SparsePattern(
            generation, static_cast<uint32_t>(first_word + i));
      }
      ASSERT_NO_FATAL_FAILURE(Copy(
          upload_->device_address,
          addresses[range.reservation] + range.byte_offset + offset, length));
      offset += length;
    }
    payload_position += range.byte_length;
  }
}

void Pm4MappingFixture::ReadSparseGeneration(
    std::span<const amdf::testing::SparseRange> ranges,
    std::span<const amdf::testing::SparseMapping> mappings,
    std::span<GpuMemory* const> slabs, const std::array<uint64_t, 2>& addresses,
    uint32_t generation, SparseReadRoute route, std::span<uint32_t> capture) {
  auto* output = static_cast<uint32_t*>(readback_->host.pointer);
  uint64_t payload_position = 0;
  for (const auto& range : ranges) {
    for (uint64_t offset = 0; offset < range.byte_length;) {
      const uint64_t position = range.byte_offset + offset;
      const uint64_t slab_offset = position % kSlabByteLength;
      const uint64_t length =
          std::min(kSlabByteLength - slab_offset, range.byte_length - offset);
      uint64_t source = addresses[range.reservation] + position;
      if (route == SparseReadRoute::kOriginalBacking) {
        const auto mapping = std::find_if(
            mappings.begin(), mappings.end(), [&](const auto& entry) {
              return entry.reservation == range.reservation &&
                     entry.byte_offset == position - slab_offset;
            });
        ASSERT_NE(mapping, mappings.end());
        source = slabs[mapping->backing]->device_address + slab_offset;
      }
      std::fill_n(output, length / 4, 0u);
      ASSERT_NO_FATAL_FAILURE(Copy(source, readback_->device_address, length));
      const uint64_t first_word = (payload_position + offset) / 4;
      uint64_t incorrect = 0;
      uint64_t first_incorrect = 0;
      for (uint64_t i = 0; i < length / 4; ++i) {
        if (output[i] !=
            amdf::testing::SparsePattern(
                generation, static_cast<uint32_t>(first_word + i))) {
          if (incorrect == 0) {
            first_incorrect = i;
          }
          ++incorrect;
        }
      }
      ASSERT_EQ(incorrect, 0u)
          << "generation=" << generation << " reservation=" << range.reservation
          << " offset=" << position
          << " first_word=" << first_word + first_incorrect
          << " observed=" << output[first_incorrect];
      if (!capture.empty()) {
        ASSERT_LE(first_word + length / 4, capture.size());
        std::copy_n(output, length / 4, capture.data() + first_word);
      }
      offset += length;
    }
    payload_position += range.byte_length;
  }
  std::fprintf(
      stderr, "public compute generation=%u route=%s checked_words=%llu\n",
      generation,
      route == SparseReadRoute::kReservation ? "reservation" : "backing",
      static_cast<unsigned long long>(payload_position / 4));
}

void Pm4MappingFixture::RunRetainedSparse(
    amdf_memory_scope_t* scope,
    std::span<const amdf::testing::SparseRange> original_ranges,
    std::span<const amdf::testing::SparseRange> replacement_ranges) {
  // The finite stream does not wrap. All backing, command storage and capture
  // storage are prepared before the first mapping generation is installed.
  ASSERT_NO_FATAL_FAILURE(Prepare(scope, UINT64_C(1) << 20));
  if (IsSkipped()) {
    return;
  }
  auto original = amdf::testing::SparseMappings(original_ranges);
  auto replacement = amdf::testing::SparseMappings(replacement_ranges);
  const uint32_t original_count = original.size();
  const uint32_t slab_count = std::max(original.size(), replacement.size());
  std::vector<GpuMemory*> slabs(slab_count);
  for (auto& slab : slabs) {
    ASSERT_NO_FATAL_FAILURE(CreateBacking(kSlabByteLength, &slab));
  }
  for (uint32_t i = 0; i < original.size(); ++i) {
    original[i].backing = i;
  }
  for (uint32_t i = 0; i < replacement.size(); ++i) {
    replacement[i].backing = i < original_count ? original_count - 1 - i : i;
  }
  auto direct = replacement;
  for (auto& mapping : direct) {
    mapping.backing = (mapping.backing + 1) % slab_count;
  }
  const amdf_memory_reservation_access_t access = {device_, 0, access_};
  const amdf_memory_reservation_create_info_t create = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CREATE_INFO,
      .structure_size = sizeof(create),
      .byte_length = reservation_info_.byte_length,
      .access_count = 1,
      .accesses = &access,
  };
  ASSERT_EQ(
      api_->memory_reservation_create(instance_, &create, &second_reservation_),
      AMDF_STATUS_OK);
  amdf_memory_reservation_access_info_t second = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_ACCESS_INFO,
      .structure_size = sizeof(second),
  };
  ASSERT_EQ(api_->memory_reservation_query_access_info(second_reservation_, 0,
                                                       &second),
            AMDF_STATUS_OK);
  const std::array<uint64_t, 2> addresses = {reservation_info_.address,
                                             second.address};
  const std::array<amdf_memory_reservation_t*, 2> reservations = {
      reservation_, second_reservation_};
  using UpdateBatch = std::array<std::vector<amdf_memory_mapping_update_t>, 2>;
  const auto make_batch =
      [&](std::span<const amdf::testing::SparseMapping> mappings) {
        UpdateBatch batch;
        for (const auto& mapping : mappings) {
          batch[mapping.reservation].push_back({
              .target_byte_offset = mapping.byte_offset,
              .byte_length = kSlabByteLength,
              .memory = slabs[mapping.backing]->memory,
              .access = access_,
          });
        }
        return batch;
      };
  const auto original_batch = make_batch(original);
  const auto replacement_batch = make_batch(replacement);
  auto direct_batch = make_batch(direct);
  for (size_t i = 0; i < direct_batch.size(); ++i) {
    for (size_t j = 0; j < direct_batch[i].size(); ++j) {
      direct_batch[i][j].previous_memory = replacement_batch[i][j].memory;
    }
  }
  auto detach_batch = original_batch;
  for (auto& projection : detach_batch) {
    for (auto& entry : projection) {
      entry.previous_memory = entry.memory;
      entry.memory = nullptr;
      entry.access = 0;
    }
  }
  const auto update = [&](const UpdateBatch& batch) {
    for (uint32_t i = 0; i < batch.size(); ++i) {
      ASSERT_EQ(api_->memory_reservation_update(
                    reservations[i], 0, batch[i].size(), batch[i].data()),
                AMDF_STATUS_OK);
    }
  };
  uint64_t payload_byte_length = 0;
  for (const auto& range : original_ranges) {
    payload_byte_length += range.byte_length;
  }
  std::vector<uint32_t> capture(payload_byte_length / 4);
  RecordProperty("mapping_sparse_range_count", original_ranges.size());
  RecordProperty("mapping_retained_slab_count", slab_count);
  RecordProperty("mapping_sparse_payload_byte_length",
                 std::to_string(payload_byte_length));

  ASSERT_NO_FATAL_FAILURE(update(original_batch));
  ASSERT_NO_FATAL_FAILURE(WriteSparseGeneration(original_ranges, addresses, 1));
  ASSERT_NO_FATAL_FAILURE(
      ReadSparseGeneration(original_ranges, original, slabs, addresses, 1,
                           SparseReadRoute::kReservation, capture));
  ASSERT_NO_FATAL_FAILURE(update(detach_batch));
  ASSERT_NO_FATAL_FAILURE(update(replacement_batch));
  ASSERT_NO_FATAL_FAILURE(
      WriteSparseGeneration(replacement_ranges, addresses, 2));
  ASSERT_NO_FATAL_FAILURE(
      ReadSparseGeneration(replacement_ranges, replacement, slabs, addresses, 2,
                           SparseReadRoute::kReservation, {}));
  ASSERT_NO_FATAL_FAILURE(
      ReadSparseGeneration(replacement_ranges, replacement, slabs, addresses, 2,
                           SparseReadRoute::kOriginalBacking, {}));
  ASSERT_TRUE(amdf::testing::CheckSparseWords(
      "independent capture", capture.data(), capture.size(), 1));
  ASSERT_NO_FATAL_FAILURE(update(direct_batch));
  ASSERT_NO_FATAL_FAILURE(
      WriteSparseGeneration(replacement_ranges, addresses, 3));
  ASSERT_NO_FATAL_FAILURE(
      ReadSparseGeneration(replacement_ranges, direct, slabs, addresses, 3,
                           SparseReadRoute::kReservation, {}));
  ASSERT_NO_FATAL_FAILURE(
      ReadSparseGeneration(replacement_ranges, direct, slabs, addresses, 3,
                           SparseReadRoute::kOriginalBacking, {}));
  ASSERT_TRUE(amdf::testing::CheckSparseWords(
      "preserved capture", capture.data(), capture.size(), 1));
  for (auto** owner : {&reservation_, &second_reservation_}) {
    ASSERT_NO_FATAL_FAILURE(ReleaseReservation(owner));
  }
  for (auto* slab : slabs) {
    ASSERT_NO_FATAL_FAILURE(ReleaseBacking(slab));
  }
}
