// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstring>

#include "libamdf/cts/gpu/pm4/encoding/commands.h"
#include "libamdf/cts/gpu/sdma/encoding/commands.h"
#include "libamdf/cts/gpu/util/command_fixture.h"

namespace {

constexpr uint32_t kStepCount = 16;
constexpr uint64_t kCommandStride = 8192;
constexpr uint32_t kBlockWordCount = 256;
constexpr uint64_t kBlockByteLength = kBlockWordCount * sizeof(uint32_t);
constexpr uint64_t kDataOffset = kStepCount * kCommandStride;
constexpr uint64_t kByteLength =
    kDataOffset + (kStepCount + 1) * kBlockByteLength;

GpuQueueRequirements Requirements(amdf_queue_command_type_t command_type) {
  return {
      .command_type = command_type,
      .roles = AMDF_QUEUE_ROLE_TRANSFER | AMDF_QUEUE_ROLE_CACHE_CONTROL,
      .format_features =
          command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_PM4
              ? amdf_queue_format_features_t{AMDF_GPU_PM4_FORMAT_FEATURE_ACQUIRE_MEM_GCR}
              : amdf_queue_format_features_t{0},
      .cache_operations = AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM |
                          AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM,
      .cache_transition_kinds = AMDF_CACHE_TRANSITION_KINDS_GLOBAL,
      .publication_modes = AMDF_QUEUE_PUBLICATION_MODE_KERNEL,
  };
}

class GpuKernelQueueDependencyTest
    : public GpuCommandTest,
      public ::testing::WithParamInterface<uint32_t> {
 protected:
  GpuKernelQueueDependencyTest()
      : GpuCommandTest(Requirements(AMDF_QUEUE_COMMAND_TYPE_GPU_PM4)) {}

  amdf_status_t MatchGpuEndpoint(amdf_endpoint_t* endpoint,
                                 bool* out_matches) override {
    *out_matches = false;
    amdf_status_t status = AMDF_STATUS_OK;
    bool matches = true;
    for (uint32_t i = 0; i < 2 && matches && amdf_status_is_ok(status); ++i) {
      const auto command_type = (GetParam() & (1u << i)) != 0
                                    ? AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA
                                    : AMDF_QUEUE_COMMAND_TYPE_GPU_PM4;
      status = FindGpuQueueFamily(api_, endpoint, Requirements(command_type),
                                  &families_[i], &matches);
      if (amdf_status_is_ok(status) && matches) {
        matches = (families_[i].kernel_queue_capabilities &
                   AMDF_KERNEL_QUEUE_CAPABILITY_EXTERNAL_SYNCHRONIZATION) != 0;
      }
      if (amdf_status_is_ok(status) && matches &&
          command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA) {
        matches = (families_[i].format_features &
                   (AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR |
                    AMDF_GPU_SDMA_FORMAT_FEATURE_SUBMISSION_GCR)) != 0;
      }
    }
    if (!amdf_status_is_ok(status) || !matches) {
      return status;
    }
    amdf_gpu_endpoint_info_t info = {
        .type = AMDF_STRUCTURE_TYPE_GPU_ENDPOINT_INFO,
        .structure_size = sizeof(info),
    };
    status = gpu_api_->endpoint_query_info(endpoint, &info);
    if (amdf_status_is_ok(status)) {
      pm4_profile_ = Pm4CommandProfile::Find(info);
      *out_matches = pm4_profile_ != nullptr || GetParam() == 3;
    }
    return status;
  }

  void SetUp() override {
    GpuCommandTest::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    for (uint32_t i = 0; i < 2; ++i) {
      const amdf_gpu_kernel_queue_create_info_t create = {
          .type = AMDF_STRUCTURE_TYPE_GPU_KERNEL_QUEUE_CREATE_INFO,
          .structure_size = sizeof(create),
          .queue_family_ordinal = families_[i].ordinal,
          .maximum_pending_submission_count = kStepCount,
          .required_capabilities =
              AMDF_KERNEL_QUEUE_CAPABILITY_EXTERNAL_SYNCHRONIZATION,
      };
      const auto status =
          gpu_api_->kernel_queue_create(device_, &create, &queues_[i]);
      if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
        GTEST_SKIP() << "native queue dependencies are unavailable";
      }
      ASSERT_EQ(status, AMDF_STATUS_OK);
    }
    const amdf_kernel_queue_dependency_create_info_t create = {
        .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_DEPENDENCY_CREATE_INFO,
        .structure_size = sizeof(create),
    };
    for (uint32_t i = 0; i < 2; ++i) {
      ASSERT_EQ(api_->kernel_queue_dependency_create(
                    queues_[i], queues_[1 - i], &create, &dependencies_[i]),
                AMDF_STATUS_OK);
    }
    ASSERT_NO_FATAL_FAILURE(
        CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE |
                         AMDF_MEMORY_ACCESS_EXECUTE,
                     kByteLength, &backing_, AMDF_MEMORY_FLAG_HOST_COHERENT));
    auto* bytes = static_cast<uint8_t*>(backing_->host.pointer);
    std::memset(bytes, 0xa5, kByteLength);
    for (uint32_t i = 0; i < kBlockWordCount; ++i) {
      expected_[i] = 0x3bf741c9u ^ (i * 0x9e3779b9u);
    }
    std::memcpy(bytes + kDataOffset, expected_.data(), kBlockByteLength);
    for (uint32_t step = 0; step < kStepCount; ++step) {
      auto* words = reinterpret_cast<uint32_t*>(bytes + step * kCommandStride);
      const auto& family = families_[step % 2];
      const uint64_t source =
          backing_->device_address + kDataOffset + step * kBlockByteLength;
      const uint64_t destination = source + kBlockByteLength;
      uint64_t command_byte_length = 0;
      if (family.command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_PM4) {
        Pm4CommandWriter writer(words, *pm4_profile_);
        writer.SystemBarrier();
        for (uint32_t i = 0; i < kBlockWordCount; ++i) {
          writer.CopyData32(source + i * 4, destination + i * 4);
        }
        writer.SystemBarrier();
        writer.PadToEightWords();
        command_byte_length = writer.word_count() * sizeof(uint32_t);
      } else {
        SdmaCommandWriter writer(words, family.format_features);
        const bool user_cache_operations =
            (family.format_features & AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR) !=
            0;
        if (user_cache_operations) {
          writer.AcquireFromSystem();
        }
        writer.CopyLinear(source, destination, kBlockByteLength);
        if (user_cache_operations) {
          writer.ReleaseToSystem();
        }
        writer.PadToEightWords();
        command_byte_length = writer.word_count() * sizeof(uint32_t);
      }
      ASSERT_LE(command_byte_length, kCommandStride);
      commands_[step] = {
          .memory = backing_->memory,
          .byte_offset = step * kCommandStride,
          .byte_length = command_byte_length,
      };
    }
    for (const auto& family : families_) {
      ASSERT_NO_FATAL_FAILURE(backing_->TransferHostCache(
          api_, family.ordinal, AMDF_HOST_CACHE_OPERATION_FLUSH));
    }
    for (uint32_t i = 0; i < 2; ++i) {
      const auto producer = backing_->DeviceSite(families_[i].ordinal);
      const auto consumer = backing_->DeviceSite(families_[1 - i].ordinal);
      amdf_memory_pair_info_t pair = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
          .structure_size = sizeof(pair),
      };
      ASSERT_EQ(api_->memory_query_pair_info(&producer, &consumer, &pair),
                AMDF_STATUS_OK);
      ASSERT_NE(pair.flags & AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE,
                0u);
      ASSERT_TRUE(pair.release.kind == AMDF_CACHE_TRANSITION_KIND_NONE ||
                  pair.release.kind == AMDF_CACHE_TRANSITION_KIND_GLOBAL);
      ASSERT_TRUE(pair.acquire.kind == AMDF_CACHE_TRANSITION_KIND_NONE ||
                  pair.acquire.kind == AMDF_CACHE_TRANSITION_KIND_GLOBAL);
    }
  }

  void TearDown() override {
    for (uint32_t i = 0; i < 2; ++i) {
      if (points_[i] != 0) {
        ASSERT_EQ(api_->kernel_queue_wait(queues_[i], points_[i],
                                          AMDF_TIMEOUT_INFINITE, 0),
                  AMDF_STATUS_OK);
      }
    }
    for (auto*& dependency : dependencies_) {
      if (dependency != nullptr) {
        const auto status = api_->kernel_queue_dependency_destroy(dependency);
        if (status != amdf_make_api_status(AMDF_STATUS_CODE_BUSY)) {
          dependency = nullptr;
        }
        ASSERT_EQ(status, AMDF_STATUS_OK);
      }
    }
    for (auto*& queue : queues_) {
      if (queue != nullptr) {
        const auto status = api_->kernel_queue_destroy(queue);
        if (status != amdf_make_api_status(AMDF_STATUS_CODE_BUSY)) {
          queue = nullptr;
        }
        ASSERT_EQ(status, AMDF_STATUS_OK);
      }
    }
    GpuCommandTest::TearDown();
  }

  // Exact native families independently admitted for the two queue owners.
  std::array<amdf_queue_family_info_t, 2> families_ = {};
  // Target-specific PM4 cache/transfer encoding, absent for SDMA-only cases.
  const Pm4CommandProfile* pm4_profile_ = nullptr;
  // Distinct native queues, even when both use the same engine family.
  std::array<amdf_kernel_queue_t*, 2> queues_ = {};
  // Direction i is queues_[i] -> queues_[1-i].
  std::array<amdf_kernel_queue_dependency_t*, 2> dependencies_ = {};
  // Last accepted point on each queue, retained through every failure path.
  std::array<uint64_t, 2> points_ = {};
  // Fixed command and payload allocation owned by the base fixture.
  GpuMemory* backing_ = nullptr;
  // Immutable command ranges authored before any native publication.
  std::array<amdf_gpu_kernel_command_t, kStepCount> commands_ = {};
  // Independent complete-block output oracle for every link in the chain.
  std::array<uint32_t, kBlockWordCount> expected_ = {};
};

TEST_P(GpuKernelQueueDependencyTest,
       ChainsNativeQueuesWithoutHostCompletionEdges) {
  EXPECT_EQ(api_->kernel_queue_destroy(queues_[0]),
            amdf_make_api_status(AMDF_STATUS_CODE_BUSY));
  amdf_kernel_queue_dependency_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_DEPENDENCY_INFO,
      .structure_size = sizeof(info),
  };
  ASSERT_EQ(api_->kernel_queue_dependency_query_info(dependencies_[0], &info),
            AMDF_STATUS_OK);
  EXPECT_EQ(info.protocol,
            AMDF_KERNEL_QUEUE_DEPENDENCY_PROTOCOL_WINDOWS_MONITORED_FENCE);
  EXPECT_EQ(info.producer_signal_operation_count, 1u);
  EXPECT_EQ(info.consumer_wait_operation_count, 1u);

  amdf_kernel_queue_dependency_wait_t wait = {dependencies_[0], UINT64_MAX};
  amdf_gpu_kernel_queue_submission_info_t submit = {
      .type = AMDF_STRUCTURE_TYPE_GPU_KERNEL_QUEUE_SUBMISSION_INFO,
      .structure_size = sizeof(submit),
      .command_count = 1,
      .wait_count = 1,
      .commands = &commands_[0],
      .waits = &wait,
  };
  uint64_t rejected = UINT64_MAX;
  EXPECT_EQ(gpu_api_->kernel_queue_submit(queues_[0], &submit, &rejected),
            amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT));
  EXPECT_EQ(gpu_api_->kernel_queue_submit(queues_[1], &submit, &rejected),
            amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE));
  EXPECT_EQ(rejected, UINT64_MAX);

  for (uint32_t step = 0; step < kStepCount; ++step) {
    const uint32_t consumer = step % 2;
    wait = {dependencies_[1 - consumer], points_[1 - consumer]};
    submit.commands = &commands_[step];
    submit.wait_count = step == 0 ? 0 : 1;
    ASSERT_EQ(gpu_api_->kernel_queue_submit(queues_[consumer], &submit,
                                            &points_[consumer]),
              AMDF_STATUS_OK);
  }
  for (uint32_t i = 0; i < 2; ++i) {
    ASSERT_EQ(api_->kernel_queue_wait(queues_[i], points_[i],
                                      AMDF_TIMEOUT_INFINITE, 0),
              AMDF_STATUS_OK);
  }
  ASSERT_NO_FATAL_FAILURE(backing_->TransferHostCache(
      api_, families_[1].ordinal, AMDF_HOST_CACHE_OPERATION_INVALIDATE));
  const auto* bytes = static_cast<const uint8_t*>(backing_->host.pointer);
  for (uint32_t step = 0; step <= kStepCount; ++step) {
    EXPECT_EQ(std::memcmp(bytes + kDataOffset + step * kBlockByteLength,
                          expected_.data(), kBlockByteLength),
              0)
        << "step=" << step;
  }
}

INSTANTIATE_TEST_SUITE_P(Engines, GpuKernelQueueDependencyTest,
                         ::testing::Values(0u, 1u, 2u, 3u),
                         ([](const ::testing::TestParamInfo<uint32_t>& info) {
                           constexpr const char* names[] = {
                               "Pm4ToPm4", "SdmaToPm4", "Pm4ToSdma",
                               "SdmaToSdma"};
                           return names[info.param];
                         }));

}  // namespace
