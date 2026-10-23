// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/aql/code.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

#include "iree/base/byte_sequence.h"
#include "iree/base/internal/atomics.h"
#include "iree/hal/drivers/amd/gpu/cts/code_images.h"
#include "iree/hal/drivers/amd/gpu/cts/test_device.h"
#include "iree/hal/drivers/amd/gpu/image/hsaco.h"
#include "iree/hal/drivers/amd/gpu/queue/aql/ring.h"
#include "iree/hal/drivers/amd/gpu/queue/user_queue.h"
#include "iree/hal/drivers/amd/status.h"
#include "iree/testing/status_matchers.h"

namespace {

class AqlCodeTest : public iree::hal::amd::gpu::cts::GpuNativeTest {
 protected:
  amdf_queue_command_type_t command_type() const override {
    return AMDF_QUEUE_COMMAND_TYPE_GPU_AQL;
  }

  void SetUp() override {
    GpuNativeTest::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    IREE_ASSERT_OK(iree_hal_amd_gpu_context_prepare_completion(
        context_, AMDF_QUEUE_PUBLICATION_MODE_USER));
    ASSERT_NO_FATAL_FAILURE(SelectControlMemory(&access_, &control_policy_));
    IREE_ASSERT_OK(iree_hal_amd_gpu_aql_code_allocate(
        context_, options_.queue_family_ordinal, &control_policy_,
        iree_allocator_system(), &publisher_));
    IREE_ASSERT_OK(iree_hal_amd_fabric_memory_allocate(
        context_->api, &control_policy_, 4096, 64, context_->event_sink,
        &control_, nullptr));
    IREE_ASSERT_OK(IREE_HAL_AMD_STATUS_FROM_AMDF(
        context_->api->memory_query_address(
            control_.handle, 0, AMDF_MEMORY_ADDRESS_GPU, &control_address_),
        "query code test control address"));
    std::memset(control_.contents.data, 0, control_.contents.data_length);
    signal_ = reinterpret_cast<iree_hal_amd_gpu_aql_signal_t*>(
        control_.contents.data);
    *signal_ = {
        .kind = 1,
        .event_mailbox = context_->event_info.mailbox_address,
        .event_id = context_->event_info.interrupt_context,
    };
    const amdf_gpu_user_queue_create_info_t queue_info = {
        .type = AMDF_STRUCTURE_TYPE_GPU_USER_QUEUE_CREATE_INFO,
        .structure_size = sizeof(queue_info),
        .queue_family_ordinal = options_.queue_family_ordinal,
        .priority = AMDF_QUEUE_PRIORITY_NORMAL,
        .producer_mode = AMDF_QUEUE_PRODUCER_MODE_SINGLE,
        .required_capabilities = AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER,
        .ring_byte_length = 4096,
    };
    amdf_user_queue_mapping_info_t mapping;
    IREE_ASSERT_OK(iree_hal_amd_gpu_user_queue_initialize(
        context_, &queue_info, &queue_, &mapping, nullptr));
    iree_hal_amd_gpu_aql_ring_initialize(
        reinterpret_cast<iree_hal_amd_gpu_aql_packet_t*>(mapping.ring_address),
        0, static_cast<uint32_t>(mapping.ring_byte_length / 64),
        reinterpret_cast<const iree_atomic_int64_t*>(
            mapping.read_index_address),
        reinterpret_cast<iree_atomic_int64_t*>(mapping.write_index_address),
        reinterpret_cast<volatile uint64_t*>(mapping.doorbell_address),
        queue_.host_publish_executor, &ring_);
  }

  void TearDown() override {
    if (pending_) {
      ASSERT_NO_FATAL_FAILURE(Wait());
    }
    if (queue_.context) {
      IREE_ASSERT_OK(IREE_HAL_AMD_STATUS_FROM_AMDF(
          iree_hal_amd_gpu_user_queue_deinitialize(&queue_, ring_.reserved),
          "join code test queue"));
    }
    IREE_ASSERT_OK(IREE_HAL_AMD_STATUS_FROM_AMDF(
        iree_hal_amd_gpu_aql_code_free(publisher_), "join code publication"));
    if (context_) {
      iree_hal_amd_fabric_memory_deinitialize(context_->api,
                                              context_->event_sink, &memory_);
      iree_hal_amd_fabric_memory_deinitialize(context_->api,
                                              context_->event_sink, &control_);
    }
    for (auto& image : images_) {
      iree_hal_amd_gpu_image_deinitialize(&image);
    }
    GpuNativeTest::TearDown();
  }

  void ReadImages() {
    const auto gfx = context_->target_info.gfx_ip;
    const char* selector = gfx.major == 9
                               ? (gfx.minor == 4 ? "gfx9-4-generic" : "gfx90a")
                           : gfx.major == 11 ? "gfx11-generic"
                                             : "gfx12-generic";
    const auto* files = iree_hal_amd_gpu_code_images_create();
    for (size_t i = 0; i < iree_hal_amd_gpu_code_images_size(); ++i) {
      if (!std::strstr(files[i].name, selector)) {
        continue;
      }
      const size_t ordinal = std::strstr(files[i].name, "alternate") ? 1 : 0;
      iree_byte_span_t bytes = {nullptr, files[i].size};
      IREE_ASSERT_OK(
          iree_allocator_malloc(iree_allocator_system(), bytes.data_length,
                                reinterpret_cast<void**>(&bytes.data)));
      std::memcpy(bytes.data, files[i].data, bytes.data_length);
      iree_byte_sequence_t* sequence = nullptr;
      iree_status_t status = iree_byte_sequence_create_from_span_move(
          &bytes, iree_allocator_system(), &sequence);
      iree_allocator_free(iree_allocator_system(), bytes.data);
      IREE_ASSERT_OK(status);
      status = iree_hal_amd_gpu_hsaco_image_initialize(
          sequence, iree_allocator_system(), &images_[ordinal]);
      iree_byte_sequence_release(sequence);
      IREE_ASSERT_OK(status);
    }
    for (auto& image : images_) {
      ASSERT_EQ(image.metadata.kernel_count, 1u);
    }
  }

  void Wait() {
    while (iree_atomic_load(
               reinterpret_cast<iree_atomic_int64_t*>(&signal_->value),
               iree_memory_order_acquire) != 0) {
      IREE_ASSERT_OK(IREE_HAL_AMD_STATUS_FROM_AMDF(
          context_->gpu->event_wait(context_->event, AMDF_TIMEOUT_INFINITE),
          "wait replaced code"));
    }
    pending_ = false;
  }

  // Immutable source images, both safe under the same argument/access contract.
  std::array<iree_hal_amd_gpu_image_t, 2> images_ = {};
  // A single native code allocation retained across all completed generations.
  iree_hal_amd_fabric_memory_t memory_ = {};
  // Device-owned cold loader machinery under test.
  iree_hal_amd_gpu_aql_code_t* publisher_ = nullptr;
  // Independent consumer queue: publication cannot rely on a same-queue fence.
  iree_hal_amd_gpu_user_queue_t queue_ = {};
  // Consumer ring operands retained until its native consumption joins.
  iree_hal_amd_gpu_aql_ring_t ring_ = {};
  // Borrowed scope requirements for coherent control storage.
  amdf_memory_device_access_t access_ = {};
  // Native control allocation policy.
  iree_hal_amd_fabric_memory_options_t control_policy_ = {};
  // Signal, native arguments, input, and output in one coherent page.
  iree_hal_amd_fabric_memory_t control_ = {};
  // GPU address of control byte zero.
  uint64_t control_address_ = 0;
  // Completion signal borrowed from control_.
  iree_hal_amd_gpu_aql_signal_t* signal_ = nullptr;
  // Accepted work must complete before teardown releases code or arguments.
  bool pending_ = false;
};

TEST_F(AqlCodeTest, ReplacesCompletedInstructionsAtSameAddress) {
  ASSERT_NO_FATAL_FAILURE(ReadImages());
  const amdf_memory_device_access_t access = {
      .device = context_->device,
      .requirements =
          {
              .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_EXECUTE,
              .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
              .address_kinds = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU,
          },
  };
  iree_hal_amd_fabric_memory_options_t staging, resident;
  IREE_ASSERT_OK(iree_hal_amd_gpu_context_query_program_memory(
      context_, options_.queue_family_ordinal, &access, &staging, &resident));
  if (!resident.host_access) {
    GTEST_SKIP()
        << "Retained-address host publication requires a local CPU view";
  }
  const uint64_t length =
      std::max(images_[0].byte_length, images_[1].byte_length) + 256;
  IREE_ASSERT_OK(iree_hal_amd_fabric_memory_allocate(
      context_->api, &resident, length,
      std::max(images_[0].alignment, images_[1].alignment),
      context_->event_sink, &memory_, nullptr));
  EXPECT_NE(memory_.info.flags & AMDF_MEMORY_FLAG_DEVICE_LOCAL, 0u);
  uint64_t address = 0;
  IREE_ASSERT_OK(IREE_HAL_AMD_STATUS_FROM_AMDF(
      context_->api->memory_query_address(memory_.handle, 0,
                                          AMDF_MEMORY_ADDRESS_GPU, &address),
      "query retained code address"));
  ASSERT_EQ(images_[0].kernels[0].descriptor_offset,
            images_[1].kernels[0].descriptor_offset);
  ASSERT_EQ(images_[0].kernels[0].entry_offset,
            images_[1].kernels[0].entry_offset);
  constexpr uint32_t kCount = 259;
  const uint64_t arguments[] = {control_address_ + 512, control_address_ + 2048,
                                uint64_t{7} << 32 | kCount};
  std::memcpy(control_.contents.data + 64, arguments, sizeof(arguments));
  auto* input = reinterpret_cast<uint32_t*>(control_.contents.data + 512);
  auto* output = reinterpret_cast<uint32_t*>(control_.contents.data + 2048);
  for (uint32_t i = 0; i < kCount; ++i) {
    input[i] = 2 * i + 1;
  }
  // Exceed the publication ring's capacity to exercise consumption/reuse too.
  for (uint32_t generation = 0; generation < 65; ++generation) {
    SCOPED_TRACE(generation);
    const auto& image = images_[generation % 2];
    std::memset(memory_.contents.data, 0, memory_.contents.data_length);
    std::vector<iree_byte_span_t> mappings;
    for (size_t i = 0; i < image.segment_count; ++i) {
      mappings.push_back(
          iree_make_byte_span(memory_.contents.data + image.segments[i].offset,
                              image.segments[i].length));
    }
    IREE_ASSERT_OK(iree_hal_amd_gpu_image_load(
        &image, address, 0, nullptr, mappings.size(), mappings.data()));
    iree_hal_amd_gpu_code_upload_t* upload = nullptr;
    IREE_ASSERT_OK(iree_hal_amd_gpu_code_publish(context_, nullptr, 0,
                                                 options_.queue_family_ordinal,
                                                 &memory_, &upload));
    IREE_ASSERT_OK(iree_hal_amd_gpu_aql_code_publish(publisher_, &memory_));
    std::fill_n(output, kCount + 1, 0xDEADBEEFu);
    iree_atomic_store(reinterpret_cast<iree_atomic_int64_t*>(&signal_->value),
                      1, iree_memory_order_release);
    const iree_hal_amd_gpu_aql_dispatch_t dispatch = {
        .header = IREE_HAL_AMD_GPU_AQL_HEADER_DISPATCH | (1u << 16) |
                  IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_SYSTEM |
                  IREE_HAL_AMD_GPU_AQL_HEADER_RELEASE_SYSTEM,
        .workgroup_size = {64, 1, 1},
        .grid_size = {320, 1, 1},
        .kernel_object = address + image.kernels[0].descriptor_offset,
        .kernarg_address = control_address_ + 64,
        .completion_signal = control_address_,
    };
    const auto packet = iree_hal_amd_gpu_aql_packet_dispatch(&dispatch);
    uint64_t position = 0;
    ASSERT_TRUE(iree_hal_amd_gpu_aql_ring_try_reserve(&ring_, 1, &position));
    iree_hal_amd_gpu_aql_ring_write_body(&ring_, position, &packet);
    ASSERT_EQ(iree_hal_amd_gpu_aql_ring_publish_header(
                  &ring_, &queue_, position, packet.words[0]),
              AMDF_STATUS_OK);
    iree_hal_amd_gpu_aql_ring_notify(&ring_);
    pending_ = true;
    ASSERT_NO_FATAL_FAILURE(Wait());
    IREE_ASSERT_OK(IREE_HAL_AMD_STATUS_FROM_AMDF(
        context_->api->user_queue_wait_consumed(queue_.queue, ring_.reserved,
                                                AMDF_TIMEOUT_INFINITE, 0),
        "join replaced code dispatch"));
    const uint32_t multiplier = generation % 2 ? 5 : 3;
    for (uint32_t i = 0; i < kCount; ++i) {
      ASSERT_EQ(output[i], input[i] * multiplier + 7) << "element=" << i;
      ASSERT_EQ(input[i], 2 * i + 1);
    }
    ASSERT_EQ(output[kCount], 0xDEADBEEFu);
  }
}

}  // namespace
