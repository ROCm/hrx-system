// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstring>

#include "libamdf/cts/gpu/sdma/encoding/commands.h"
#include "libamdf/cts/gpu/util/command_fixture.h"
#include "util/wait.h"

namespace {

constexpr auto kReadWrite = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
constexpr uint32_t kGenerationCount = 8;
constexpr uint32_t kPayloadLength = 256;

class SdmaQueuedMappingTest : public GpuCommandTest {
 protected:
  SdmaQueuedMappingTest()
      : GpuCommandTest({
            .command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA,
            .roles = AMDF_QUEUE_ROLE_TRANSFER,
            .publication_modes = AMDF_QUEUE_PUBLICATION_MODE_KERNEL,
        }) {}

  void TearDown() override {
    if (control_ != nullptr) {
      GpuStoreRelease<uint32_t>(
          reinterpret_cast<uintptr_t>(control_->host.pointer), 1);
    }
    if (queue_ != nullptr) {
      if (point_ != 0) {
        ASSERT_EQ(
            api_->kernel_queue_wait(queue_, point_, amdf::cts::kWaitTimeoutNs,
                                    amdf::cts::kWaitPollIntervalNs),
            AMDF_STATUS_OK);
      }
      const auto status = api_->kernel_queue_destroy(queue_);
      if (status != amdf_make_api_status(AMDF_STATUS_CODE_BUSY)) {
        queue_ = nullptr;
      }
      ASSERT_EQ(status, AMDF_STATUS_OK);
    }
    if (reservation_ != nullptr) {
      const auto status = api_->memory_reservation_destroy(reservation_);
      reservation_ = nullptr;
      ASSERT_EQ(status, AMDF_STATUS_OK);
    }
    GpuCommandTest::TearDown();
  }

  void Submit(uint32_t segment, uint64_t byte_length) {
    const amdf_gpu_kernel_command_t command = {
        .memory = commands_->memory,
        .byte_offset = segment * UINT64_C(4096),
        .byte_length = byte_length,
    };
    const amdf_gpu_kernel_queue_submission_info_t submit = {
        .type = AMDF_STRUCTURE_TYPE_GPU_KERNEL_QUEUE_SUBMISSION_INFO,
        .structure_size = sizeof(submit),
        .command_count = 1,
        .commands = &command,
    };
    ASSERT_EQ(gpu_api_->kernel_queue_submit(queue_, &submit, &point_),
              AMDF_STATUS_OK);
  }

  // Queue owns both mapping and transfer progress.
  amdf_kernel_queue_t* queue_ = nullptr;
  // Last accepted point, including checked final detach.
  uint64_t point_ = 0;
  // Stable projection, removed before fixture-owned backing.
  amdf_memory_reservation_t* reservation_ = nullptr;
  // Finite predecessor gate released even on assertion failure.
  GpuMemory* control_ = nullptr;
  // Immutable, separately aligned command segments.
  GpuMemory* commands_ = nullptr;
};

TEST_F(SdmaQueuedMappingTest, PendingTransferThenReplacementAndDetach) {
  const amdf_gpu_kernel_queue_create_info_t queue_create = {
      .type = AMDF_STRUCTURE_TYPE_GPU_KERNEL_QUEUE_CREATE_INFO,
      .structure_size = sizeof(queue_create),
      .queue_family_ordinal = family_.ordinal,
      .maximum_pending_submission_count = 32,
  };
  ASSERT_EQ(gpu_api_->kernel_queue_create(device_, &queue_create, &queue_),
            AMDF_STATUS_OK);
  amdf_kernel_queue_info_t queue_info = {
      .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_INFO,
      .structure_size = sizeof(queue_info),
  };
  ASSERT_EQ(api_->kernel_queue_query_info(queue_, &queue_info), AMDF_STATUS_OK);
  if ((queue_info.capabilities & AMDF_KERNEL_QUEUE_CAPABILITY_VIRTUAL_MEMORY) ==
      0) {
    GTEST_SKIP() << "native queue-ordered mapping is not advertised";
  }
  amdf_memory_reservation_capabilities_t capabilities = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CAPABILITIES,
      .structure_size = sizeof(capabilities),
  };
  ASSERT_EQ(api_->device_query_memory_reservation_capabilities(device_, 0,
                                                               &capabilities),
            AMDF_STATUS_OK);
  const uint64_t granularity = capabilities.mapping_granularity;
  ASSERT_GE(granularity, kPayloadLength);
  const auto access = kReadWrite | capabilities.required_access;
  const amdf_memory_reservation_access_t consumer = {device_, 0, access};
  const amdf_memory_reservation_create_info_t reservation_create = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CREATE_INFO,
      .structure_size = sizeof(reservation_create),
      .byte_length = UINT64_C(8) << 30,
      .access_count = 1,
      .accesses = &consumer,
  };
  ASSERT_EQ(api_->memory_reservation_create(instance_, &reservation_create,
                                            &reservation_),
            AMDF_STATUS_OK);
  amdf_memory_reservation_access_info_t projection = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_ACCESS_INFO,
      .structure_size = sizeof(projection),
  };
  ASSERT_EQ(
      api_->memory_reservation_query_access_info(reservation_, 0, &projection),
      AMDF_STATUS_OK);
  const uint64_t offset = ((UINT64_C(5) << 30) / granularity) * granularity;
  const uint64_t address = projection.address + offset;
  const amdf_memory_device_access_t attachment = {
      device_,
      {
          .access = access,
          .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS |
                   AMDF_MEMORY_FLAG_HOST_COHERENT |
                   AMDF_MEMORY_FLAG_MAPPING_SOURCE,
      }};
  const auto ordinal = FindGpuMemoryProfileOrdinal(
      api_, system_scope_, device_, AMDF_MEMORY_PROFILE_ROLE_CREATE,
      AMDF_MEMORY_FLAG_HOST_VISIBLE, attachment.requirements);
  ASSERT_NE(ordinal, AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
  const amdf_memory_create_info_t backing_create = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
      .structure_size = sizeof(backing_create),
      .memory_profile_ordinal = ordinal,
      .access_count = 1,
      .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
      .byte_length = 2 * granularity,
      .minimum_alignment = granularity,
      .accesses = &attachment,
  };
  GpuMemory* backing = nullptr;
  GpuMemory* capture = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(system_scope_, backing_create, &backing));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(kReadWrite, 4096, &control_));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(kReadWrite, 8192, &capture));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_EXECUTE,
                   (kGenerationCount + 1) * 4096, &commands_));
  std::memset(control_->host.pointer, 0, 4096);
  std::memset(capture->host.pointer, 0, 8192);
  std::array<uint64_t, kGenerationCount + 1> lengths = {};
  for (uint32_t generation = 0; generation <= kGenerationCount; ++generation) {
    SdmaCommandWriter writer(
        static_cast<uint32_t*>(commands_->host.pointer) + generation * 1024,
        family_.format_features);
    if (generation == 0) {
      writer.WaitMemory32(control_->device_address, 1);
    }
    const bool user_gcr =
        (family_.format_features & AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR) != 0;
    if (user_gcr) {
      writer.AcquireFromSystem();
    }
    writer.Fill32(address, 0xadc00000u + generation, kPayloadLength);
    writer.CopyLinear(address,
                      capture->device_address + 2 * generation * kPayloadLength,
                      kPayloadLength);
    writer.CopyLinear(
        backing->device_address + (generation % 2) * granularity,
        capture->device_address + (2 * generation + 1) * kPayloadLength,
        kPayloadLength);
    if (user_gcr) {
      writer.ReleaseToSystem();
    }
    while (writer.word_count() % 8 != 0) {
      writer.Noop();
    }
    ASSERT_LE(writer.word_count(), 1024u);
    lengths[generation] = writer.word_count() * 4;
  }
  ASSERT_EQ(api_->host_mapping_cache_control(commands_->mapping,
                                             AMDF_HOST_CACHE_OPERATION_FLUSH, 0,
                                             commands_->info.byte_length),
            AMDF_STATUS_OK);
  amdf_memory_mapping_update_t update = {
      .target_byte_offset = offset,
      .byte_length = granularity,
      .memory = backing->memory,
      .access = access,
  };
  ASSERT_EQ(api_->memory_reservation_update(reservation_, 0, 1, &update),
            AMDF_STATUS_OK);
  ASSERT_NO_FATAL_FAILURE(Submit(0, lengths[0]));
  for (uint32_t generation = 1; generation <= kGenerationCount; ++generation) {
    update.previous_memory = backing->memory;
    update.previous_source_byte_offset = update.source_byte_offset;
    update.source_byte_offset = (generation % 2) * granularity;
    ASSERT_EQ(api_->kernel_queue_update_memory(queue_, reservation_, 0, 1,
                                               &update, &point_),
              AMDF_STATUS_OK);
    ASSERT_NO_FATAL_FAILURE(Submit(generation, lengths[generation]));
    if (generation == 1) {
      amdf_kernel_queue_status_t status = {
          .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS,
          .structure_size = sizeof(status),
      };
      ASSERT_EQ(api_->kernel_queue_refresh_status(queue_, &status),
                AMDF_STATUS_OK);
      EXPECT_EQ(status.retired_submission, 0u);
      GpuStoreRelease<uint32_t>(
          reinterpret_cast<uintptr_t>(control_->host.pointer), 1);
    }
  }
  const amdf_memory_mapping_update_t detach = {
      .target_byte_offset = offset,
      .byte_length = granularity,
      .previous_memory = backing->memory,
      .previous_source_byte_offset = (kGenerationCount % 2) * granularity,
  };
  ASSERT_EQ(api_->kernel_queue_update_memory(queue_, reservation_, 0, 1,
                                             &detach, &point_),
            AMDF_STATUS_OK);
  ASSERT_EQ(api_->kernel_queue_wait(queue_, point_, amdf::cts::kWaitTimeoutNs,
                                    amdf::cts::kWaitPollIntervalNs),
            AMDF_STATUS_OK);
  const auto* words = static_cast<const uint32_t*>(capture->host.pointer);
  for (uint32_t generation = 0; generation <= kGenerationCount; ++generation) {
    for (uint32_t word = 0; word < 2 * kPayloadLength / 4; ++word) {
      ASSERT_EQ(words[generation * 2 * kPayloadLength / 4 + word],
                0xadc00000u + generation)
          << "generation=" << generation << " word=" << word;
    }
  }
  ASSERT_TRUE(backing->Release(api_));
}

}  // namespace
