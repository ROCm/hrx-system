// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/gpu/util/command_queue.h"

#include "gtest/gtest.h"
#include "libamdf/cts/gpu/gpu_device_fixture.h"

amdf_queue_publication_modes_t SelectGpuHostPublication(
    const amdf_queue_family_info_t& family,
    amdf_queue_publication_modes_t permitted_modes) {
  const auto modes = family.publication_modes & permitted_modes;
  if ((modes & AMDF_QUEUE_PUBLICATION_MODE_USER) != 0 &&
      (family.user_queue_capabilities &
       AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER) != 0 &&
      (family.producer_modes & AMDF_QUEUE_PRODUCER_MODE_BIT_SINGLE) != 0 &&
      (family.priority_capabilities & AMDF_QUEUE_PRIORITY_CAPABILITY_NORMAL) !=
          0) {
    return AMDF_QUEUE_PUBLICATION_MODE_USER;
  }
  return modes & AMDF_QUEUE_PUBLICATION_MODE_KERNEL;
}

void GpuCommandQueue::Initialize(
    const amdf_api_t* api, const amdf_gpu_api_t* gpu_api, amdf_device_t* device,
    amdf_memory_scope_t* system_scope, const amdf_queue_family_info_t& family,
    amdf_queue_publication_modes_t publication_mode,
    uint64_t command_byte_length) {
  ASSERT_TRUE(family.command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_PM4 ||
              family.command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA);
  command_type_ = family.command_type;
  ::testing::Test::RecordProperty(
      command_type_ == AMDF_QUEUE_COMMAND_TYPE_GPU_PM4
          ? "pm4_publication_mode"
          : "sdma_publication_mode",
      publication_mode == AMDF_QUEUE_PUBLICATION_MODE_USER ? "user" : "kernel");
  if (publication_mode == AMDF_QUEUE_PUBLICATION_MODE_USER) {
    ASSERT_NO_FATAL_FAILURE(user_queue_.Initialize(
        api, gpu_api, device, family, AMDF_QUEUE_PRODUCER_MODE_SINGLE, {},
        AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER, command_byte_length));
    words_ = {reinterpret_cast<uint32_t*>(user_queue_.host.ring_address),
              user_queue_.host.ring_byte_length / sizeof(uint32_t)};
    device_id_ = user_queue_.info.device_id;
    return;
  }
  ASSERT_EQ(publication_mode, AMDF_QUEUE_PUBLICATION_MODE_KERNEL);
  amdf_gpu_kernel_queue_create_info_t create_queue = {
      .type = AMDF_STRUCTURE_TYPE_GPU_KERNEL_QUEUE_CREATE_INFO,
      .structure_size = sizeof(create_queue),
      .queue_family_ordinal = family.ordinal,
      .maximum_pending_submission_count = 1,
  };
  ASSERT_EQ(gpu_api->kernel_queue_create(device, &create_queue, &kernel_queue_),
            AMDF_STATUS_OK);
  amdf_kernel_queue_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_INFO,
      .structure_size = sizeof(info),
  };
  ASSERT_EQ(api->kernel_queue_query_info(kernel_queue_, &info), AMDF_STATUS_OK);
  ASSERT_EQ(info.queue_family_ordinal, family.ordinal);
  ASSERT_EQ(info.command_type, command_type_);
  ASSERT_GE(info.maximum_pending_submission_count, 1u);
  ASSERT_GE(info.maximum_command_count, 1u);
  device_id_ = info.device_id;

  const amdf_memory_device_access_t access = {
      device,
      {
          .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE |
                    AMDF_MEMORY_ACCESS_EXECUTE,
          .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
          .address_kinds = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU,
      }};
  const auto ordinal = FindGpuMemoryProfileOrdinal(
      api, system_scope, device,
      AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
      AMDF_MEMORY_FLAG_HOST_VISIBLE, access.requirements);
  ASSERT_NE(ordinal, AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
  amdf_memory_profile_t profile = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
      .structure_size = sizeof(profile),
  };
  amdf_memory_access_capabilities_t capabilities = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
      .structure_size = sizeof(capabilities),
  };
  ASSERT_EQ(api->memory_scope_query_device_profile(
                system_scope, ordinal, 1, &access, &profile, &capabilities),
            AMDF_STATUS_OK);
  const uint64_t granularity = profile.allocation.byte_length_granularity;
  ASSERT_GT(granularity, 0u);
  const uint64_t requested_byte_length =
      command_byte_length ? command_byte_length : 4096;
  amdf_memory_create_info_t create_memory = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
      .structure_size = sizeof(create_memory),
      .memory_profile_ordinal = ordinal,
      .access_count = 1,
      .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
      .byte_length =
          (requested_byte_length + granularity - 1) / granularity * granularity,
      .minimum_alignment = profile.allocation.minimum_alignment,
      .accesses = &access,
  };
  ASSERT_NO_FATAL_FAILURE(commands_.Create(api, system_scope, create_memory));
  words_ = {reinterpret_cast<uint32_t*>(commands_.host.pointer),
            commands_.host.byte_length / sizeof(uint32_t)};
}

const void* GpuCommandQueue::native_handle() const {
  if (kernel_queue_) {
    return kernel_queue_;
  }
  return user_queue_.queue;
}

void GpuCommandQueue::Publish(const amdf_api_t* api,
                              const amdf_gpu_api_t* gpu_api,
                              size_t word_count) {
  ASSERT_GT(word_count, published_word_count_);
  // Leave a free word even on KERNEL so the same stream fits a USER ring.
  ASSERT_LT(word_count, words_.size());
  if (kernel_queue_) {
    const uint64_t byte_offset = published_word_count_ * sizeof(uint32_t);
    const uint64_t byte_length =
        (word_count - published_word_count_) * sizeof(uint32_t);
    ASSERT_EQ(api->host_mapping_cache_control(commands_.mapping,
                                              AMDF_HOST_CACHE_OPERATION_FLUSH,
                                              byte_offset, byte_length),
              AMDF_STATUS_OK);
    const amdf_gpu_kernel_command_t command = {
        .memory = commands_.memory,
        .byte_offset = byte_offset,
        .byte_length = byte_length,
    };
    ASSERT_NO_FATAL_FAILURE(Submit(gpu_api, command));
  } else {
    const uint64_t index = command_type_ == AMDF_QUEUE_COMMAND_TYPE_GPU_PM4
                               ? word_count
                               : word_count * sizeof(uint32_t);
    ASSERT_NO_FATAL_FAILURE(user_queue_.PublishStream(index));
  }
  published_word_count_ = word_count;
}

void GpuCommandQueue::Submit(const amdf_gpu_api_t* gpu_api,
                             const amdf_gpu_kernel_command_t& command) {
  ASSERT_NE(kernel_queue_, nullptr);
  amdf_gpu_kernel_queue_submission_info_t submit = {
      .type = AMDF_STRUCTURE_TYPE_GPU_KERNEL_QUEUE_SUBMISSION_INFO,
      .structure_size = sizeof(submit),
      .command_count = 1,
      .commands = &command,
  };
  ASSERT_EQ(gpu_api->kernel_queue_submit(kernel_queue_, &submit, &submission_),
            AMDF_STATUS_OK);
}

void GpuCommandQueue::WaitRetired(const amdf_api_t* api) {
  if (kernel_queue_) {
    ASSERT_EQ(api->kernel_queue_wait(kernel_queue_, submission_,
                                     AMDF_TIMEOUT_INFINITE, 0),
              AMDF_STATUS_OK);
    amdf_kernel_queue_status_t status = {
        .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS,
        .structure_size = sizeof(status),
    };
    ASSERT_EQ(api->kernel_queue_query_status(kernel_queue_, &status),
              AMDF_STATUS_OK);
    EXPECT_EQ(status.state, AMDF_QUEUE_STATE_ACTIVE);
    EXPECT_EQ(status.terminal_status, AMDF_STATUS_OK);
    EXPECT_GE(status.retired_submission, submission_);
  } else {
    const uint64_t index = command_type_ == AMDF_QUEUE_COMMAND_TYPE_GPU_PM4
                               ? published_word_count_
                               : published_word_count_ * sizeof(uint32_t);
    ASSERT_NO_FATAL_FAILURE(user_queue_.WaitConsumed(api, index));
    amdf_user_queue_status_t status = {
        .type = AMDF_STRUCTURE_TYPE_USER_QUEUE_STATUS,
        .structure_size = sizeof(status),
    };
    ASSERT_EQ(api->user_queue_query_status(user_queue_.queue, &status),
              AMDF_STATUS_OK);
    EXPECT_EQ(status.state, AMDF_QUEUE_STATE_ACTIVE);
    EXPECT_EQ(status.terminal_status, AMDF_STATUS_OK);
    EXPECT_EQ(status.producer_index, index);
    EXPECT_EQ(status.consumed_index, index);
  }
}

bool GpuCommandQueue::Release(const amdf_api_t* api) {
  if (!user_queue_.Release(api)) {
    return false;
  }
  if (kernel_queue_) {
    const auto status = api->kernel_queue_destroy(kernel_queue_);
    if (status != amdf_make_api_status(AMDF_STATUS_CODE_BUSY)) {
      kernel_queue_ = nullptr;
    }
    EXPECT_EQ(status, AMDF_STATUS_OK);
    if (!amdf_status_is_ok(status)) {
      return false;
    }
  }
  return commands_.Release(api);
}
