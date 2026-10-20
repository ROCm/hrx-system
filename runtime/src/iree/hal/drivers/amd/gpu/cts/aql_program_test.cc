// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cstring>
#include <vector>

#include "iree/base/internal/atomics.h"
#include "iree/hal/drivers/amd/gpu/cts/test_device.h"
#include "iree/hal/drivers/amd/gpu/queue/aql/program.h"
#include "iree/hal/drivers/amd/gpu/queue/aql/ring.h"
#include "iree/hal/drivers/amd/gpu/queue/user_queue.h"
#include "iree/hal/drivers/amd/status.h"
#include "iree/testing/status_matchers.h"

namespace {

using iree::hal::amd::gpu::cts::GpuNativeTest;

// Exercises resident native bytes against the real command processor. Public
// recording/admission owns separate tests; no host packet replay supplies the
// device-produced range here.
class AqlProgramTest : public GpuNativeTest {
 protected:
  amdf_queue_command_type_t command_type() const override {
    return AMDF_QUEUE_COMMAND_TYPE_GPU_AQL;
  }

  void SetUp() override {
    GpuNativeTest::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    IREE_ASSERT_OK(iree_hal_amd_gpu_context_prepare_memory(context_, 0));
    IREE_ASSERT_OK(iree_hal_amd_gpu_context_prepare_completion(
        context_, AMDF_QUEUE_PUBLICATION_MODE_USER));
    IREE_ASSERT_OK(iree_hal_amd_gpu_aql_program_pool_initialize(
        context_, options_.queue_family_ordinal, 1, iree_allocator_system(),
        &pool_));
    // Exercise the actual first-copy path even on large-BAR machines. This is
    // the same local allocation policy with no host mapping requested.
    pool_.storage.resident.host_access = 0;
    pool_.storage.resident.required_flags &= ~AMDF_MEMORY_FLAG_HOST_VISIBLE;
    ASSERT_NO_FATAL_FAILURE(SelectControlMemory(&access_, &control_policy_));
    IREE_ASSERT_OK(iree_hal_amd_fabric_memory_allocate(
        context_->api, &control_policy_, 8192, 256, context_->event_sink,
        &control_, nullptr));
    IREE_ASSERT_OK(IREE_HAL_AMD_STATUS_FROM_AMDF(
        context_->api->memory_query_address(
            control_.handle, 0, AMDF_MEMORY_ADDRESS_GPU, &control_address_),
        "query AQL invocation storage"));
    std::memset(control_.contents.data, 0, control_.contents.data_length);
    const amdf_gpu_user_queue_create_info_t queue_info = {
        .type = AMDF_STRUCTURE_TYPE_GPU_USER_QUEUE_CREATE_INFO,
        .structure_size = sizeof(queue_info),
        .queue_family_ordinal = options_.queue_family_ordinal,
        .priority = AMDF_QUEUE_PRIORITY_NORMAL,
        .producer_mode = AMDF_QUEUE_PRODUCER_MODE_SINGLE,
        .required_capabilities = AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER |
                                 AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER,
        .ring_byte_length = 4096,
    };
    amdf_user_queue_mapping_info_t host, device;
    IREE_ASSERT_OK(iree_hal_amd_gpu_user_queue_initialize(
        context_, &queue_info, &queue_, &host, &device));
    iree_hal_amd_gpu_aql_ring_initialize(
        reinterpret_cast<iree_hal_amd_gpu_aql_packet_t*>(host.ring_address),
        device.ring_address, static_cast<uint32_t>(host.ring_byte_length / 64),
        reinterpret_cast<const iree_atomic_int64_t*>(host.read_index_address),
        reinterpret_cast<iree_atomic_int64_t*>(host.write_index_address),
        reinterpret_cast<volatile uint64_t*>(host.doorbell_address), &ring_);
    signal_ = reinterpret_cast<iree_hal_amd_gpu_aql_signal_t*>(
        control_.contents.data);
    *signal_ = {
        .kind = 1,
        .event_mailbox = context_->event_info.mailbox_address,
        .event_id = context_->event_info.interrupt_context,
    };
  }

  void TearDown() override {
    if (pending_) {
      ASSERT_NO_FATAL_FAILURE(Wait());
    }
    if (queue_.context) {
      IREE_ASSERT_OK(IREE_HAL_AMD_STATUS_FROM_AMDF(
          iree_hal_amd_gpu_user_queue_deinitialize(&queue_, ring_.reserved),
          "join AQL native packet ownership"));
    }
    if (instance_) {
      iree_hal_amd_gpu_program_cancel(&program_, instance_);
    }
    iree_hal_amd_gpu_program_deinitialize(&program_);
    iree_hal_amd_gpu_aql_program_pool_deinitialize(&pool_);
    if (control_.handle) {
      iree_hal_amd_fabric_memory_deinitialize(context_->api,
                                               context_->event_sink, &control_);
    }
    GpuNativeTest::TearDown();
  }

  void Write(uint64_t position, iree_hal_amd_gpu_aql_packet_t packet) {
    iree_hal_amd_gpu_aql_ring_write_body(&ring_, position, &packet);
    iree_hal_amd_gpu_aql_ring_publish_header(&ring_, position, packet.words[0]);
  }

  iree_hal_amd_gpu_aql_packet_t Dispatch(uint64_t kernel, uint64_t arguments,
                                         uint32_t groups, uint32_t header) {
    const iree_hal_amd_gpu_aql_dispatch_t dispatch = {
        .header = header | IREE_HAL_AMD_GPU_AQL_HEADER_DISPATCH | (1u << 16),
        .workgroup_size = {64, 1, 1},
        .grid_size = {groups * 64, 1, 1},
        .kernel_object = kernel,
        .kernarg_address = arguments,
    };
    return iree_hal_amd_gpu_aql_packet_dispatch(&dispatch);
  }

  void Wait() {
    while (iree_atomic_load(
               reinterpret_cast<iree_atomic_int64_t*>(&signal_->value),
               iree_memory_order_acquire) != 0) {
      IREE_ASSERT_OK(IREE_HAL_AMD_STATUS_FROM_AMDF(
          context_->gpu->event_wait(context_->event, AMDF_TIMEOUT_INFINITE),
          "wait AQL program completion"));
    }
    pending_ = false;
    IREE_ASSERT_OK(IREE_HAL_AMD_STATUS_FROM_AMDF(
        context_->api->user_queue_wait_consumed(queue_.queue, ring_.reserved,
                                                AMDF_TIMEOUT_INFINITE, 0),
        "join consumed AQL materialization slots"));
  }

  // Cold builtin placement and resident instance registry.
  iree_hal_amd_gpu_aql_program_pool_t pool_ = {};
  // Sealed native packet/private-data allocation.
  iree_hal_amd_gpu_program_t program_ = {};
  // Exclusive invocation lease, retained across every published chunk.
  iree_hal_amd_gpu_program_instance_t* instance_ = nullptr;
  // Native ring ownership, independent of the template placement.
  iree_hal_amd_gpu_user_queue_t queue_ = {};
  // Single-publisher mapped packet operands.
  iree_hal_amd_gpu_aql_ring_t ring_ = {};
  // Coherent control-memory access requirements.
  amdf_memory_device_access_t access_ = {};
  // Qualified CPU/GPU control policy.
  iree_hal_amd_fabric_memory_options_t control_policy_ = {};
  // Host uploads, count producer output, completion signal, and checked output.
  iree_hal_amd_fabric_memory_t control_ = {};
  // Native address of control byte zero.
  uint64_t control_address_ = 0;
  // Borrowed control signal through native completion and consumption.
  iree_hal_amd_gpu_aql_signal_t* signal_ = nullptr;
  // A terminal packet has been accepted and must be joined before teardown.
  bool pending_ = false;
};

TEST_F(AqlProgramTest, ResidentFixupsAndIndirectPairsSurviveRingWrap) {
  constexpr uint32_t kOutputs = 97;
  constexpr uint32_t kPacketCount = 1 + kOutputs + kOutputs / 3;
  constexpr uint32_t kDataOffset = kPacketCount * 64;
  std::vector<iree_hal_amd_gpu_aql_packet_t> packets;
  std::vector<uint8_t> data;
  std::vector<iree_hal_amd_gpu_fixup_t> fixups;
  auto append_data = [&](const auto& value) {
    const uint32_t offset = kDataOffset + static_cast<uint32_t>(data.size());
    const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
    data.insert(data.end(), bytes, bytes + sizeof(value));
    return offset;
  };
  auto pointer = [&](uint32_t target, uint32_t source, uint64_t addend) {
    fixups.push_back({target, source, addend});
    fixups.push_back({target + 4, source | (32u << 24), addend});
  };
  auto append_dispatch = [&](iree_hal_amd_gpu_aql_packet_t packet,
                             uint32_t argument_offset) {
    pointer(static_cast<uint32_t>(packets.size()) * 64 + 40, 0,
            argument_offset);
    packets.push_back(packet);
  };
  // The first real kernel produces all three indirect counts. Every count
  // patch is ordered after this producer, including across chunk boundaries.
  const iree_hal_amd_gpu_blit_arguments_t counts = {
      .target = control_address_ + 512,
      .length = 12,
      .fill = 1,
  };
  const uint32_t counts_offset = append_data(counts);
  fixups.push_back({counts_offset + 24, 3, 0});
  append_dispatch(
      Dispatch(pool_.blit, 0, 1, IREE_HAL_AMD_GPU_AQL_HEADER_RELEASE_AGENT),
      counts_offset);
  for (uint32_t i = 0; i < kOutputs; ++i) {
    const iree_hal_amd_gpu_blit_arguments_t fill = {.length = 4, .fill = 1};
    const uint32_t arguments = append_data(fill);
    pointer(arguments + 8, 1, i * 4);
    fixups.push_back({arguments + 24, 2, i});
    auto target =
        Dispatch(pool_.blit, 0, 1, IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_AGENT);
    if (i % 3 == 1) {
      const iree_hal_amd_gpu_aql_indirect_arguments_t patch = {
          .counts = control_address_ + 512,
          .header = target.words[0] | IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED,
          .workgroup_size = {64, 1, 1},
          .count_offsets = {UINT32_MAX, UINT32_MAX, UINT32_MAX},
      };
      const uint32_t patch_offset = append_data(patch);
      append_dispatch(iree_hal_amd_gpu_aql_indirect_packet(pool_.indirect, 0),
                      patch_offset);
      target.words[0] = IREE_HAL_AMD_GPU_AQL_HEADER_INVALID;
    }
    append_dispatch(target, arguments);
  }
  ASSERT_EQ(packets.size(), kPacketCount);
  IREE_ASSERT_OK(iree_hal_amd_gpu_program_initialize(
      &pool_.storage, 64, kDataOffset + static_cast<uint32_t>(data.size()),
      static_cast<uint32_t>(fixups.size()), &program_));
  std::memcpy(program_.contents.data, packets.data(), kDataOffset);
  std::memcpy(program_.contents.data + kDataOffset, data.data(), data.size());
  std::memcpy(program_.contents.data + program_.fixup_offset, fixups.data(),
              fixups.size() * sizeof(fixups[0]));
  IREE_ASSERT_OK(iree_hal_amd_gpu_program_seal(&program_));

  uint64_t epoch = 0;
  for (uint32_t maximum : {2u, 7u, 31u, 60u}) {
    SCOPED_TRACE(maximum);
    // Recorder-owned boundaries are frozen before publication. Neither
    // submission nor the materializer discovers them from the native ring.
    std::vector<uint32_t> blocks;
    for (uint32_t offset = 0; offset < kPacketCount;) {
      uint32_t count = std::min(maximum, kPacketCount - offset);
      if (offset + count < kPacketCount &&
          packets[offset + count].words[0] ==
              IREE_HAL_AMD_GPU_AQL_HEADER_INVALID) {
        --count;
      }
      blocks.push_back(count);
      offset += count;
    }
    const uint64_t frontier[] = {epoch};
    instance_ = iree_hal_amd_gpu_program_try_acquire(&program_, 0, frontier);
    ASSERT_NE(instance_, nullptr);
    const auto source = iree_hal_amd_gpu_program_source(&program_);
    const uint64_t values[] = {instance_->address, control_address_ + 4096,
                               1000 + epoch * 100, epoch & 1};
    std::memcpy(control_.contents.data + 256, values, sizeof(values));
    std::memset(control_.contents.data + 4096, 0xA5, kOutputs * 4);
    const iree_hal_amd_gpu_blit_arguments_t copy = {
        .source = source.address,
        .target = instance_->address,
        .length = program_.fixup_offset +
                  program_.fixup_count * sizeof(iree_hal_amd_gpu_fixup_t),
    };
    const iree_hal_amd_gpu_fixup_arguments_t apply = {
        .entries = source.fixups,
        .values = control_address_ + 256,
        .target = instance_->address,
        .entry_count = program_.fixup_count,
    };
    std::memcpy(control_.contents.data + 64, &copy, sizeof(copy));
    std::memcpy(control_.contents.data + 96, &apply, sizeof(apply));
    uint32_t offset = 0;
    for (uint32_t count : blocks) {
      const uint32_t initialize = offset == 0 && !instance_->initialized;
      const uint32_t patch = offset == 0;
      uint64_t position = 0;
      ASSERT_TRUE(iree_hal_amd_gpu_aql_ring_try_reserve(
          &ring_, initialize + patch + 1 + count + 1, &position));
      const uint64_t first = position + initialize + patch + 1;
      const uint64_t packet_address = instance_->address + offset * 64;
      const iree_hal_amd_gpu_aql_materialize_arguments_t materialize = {
          .ring = ring_.device_address,
          .packets = packet_address,
          .ring_address = ring_.device_address,
          .packet_address = packet_address,
          .ring_mask = ring_.capacity_mask,
          .first = static_cast<uint32_t>(first) & ring_.capacity_mask,
          .packet_count = count,
      };
      std::memcpy(control_.contents.data + 128, &materialize,
                  sizeof(materialize));
      signal_->value = 1;
      if (initialize) {
        Write(position++,
              Dispatch(pool_.blit, control_address_ + 64, 256,
                       IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED |
                           IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_SYSTEM |
                           IREE_HAL_AMD_GPU_AQL_HEADER_RELEASE_AGENT));
      }
      if (patch) {
        Write(position++,
              Dispatch(pool_.fixup, control_address_ + 96,
                       (program_.fixup_count + 63) / 64,
                       IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED |
                           IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_SYSTEM |
                           IREE_HAL_AMD_GPU_AQL_HEADER_RELEASE_AGENT));
      }
      Write(position++, iree_hal_amd_gpu_aql_materialize_packet(
                            pool_.materialize, control_address_ + 128,
                            IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED |
                                IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_SYSTEM));
      // These count slots are written exclusively by the executing kernel.
      position += count;
      Write(position, iree_hal_amd_gpu_aql_packet_barrier(
                          IREE_HAL_AMD_GPU_AQL_HEADER_BARRIER_AND |
                              IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED |
                              IREE_HAL_AMD_GPU_AQL_HEADER_RELEASE_SYSTEM,
                          0, nullptr, control_address_));
      pending_ = true;
      iree_hal_amd_gpu_aql_ring_notify(&ring_);
      ASSERT_NO_FATAL_FAILURE(Wait());
      offset += count;
    }
    iree_hal_amd_gpu_program_publish(&program_, instance_, 0, ++epoch);
    iree_hal_amd_gpu_program_release(&program_, instance_);
    instance_ = nullptr;
    const auto* output =
        reinterpret_cast<const uint32_t*>(control_.contents.data + 4096);
    for (uint32_t i = 0; i < kOutputs; ++i) {
      EXPECT_EQ(output[i], i % 3 == 1 && values[3] == 0
                               ? 0xA5A5A5A5u
                               : static_cast<uint32_t>(values[2] + i))
          << "output " << i;
    }
    iree_hal_amd_gpu_program_pool_trim(&pool_.storage);
  }
  EXPECT_GT(ring_.reserved, ring_.capacity * 8);
}

TEST_F(AqlProgramTest, TaggedCompletionsCaptureAndElideAcrossRingWrap) {
  constexpr uint32_t kHeader = IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED |
                               IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_SYSTEM |
                               IREE_HAL_AMD_GPU_AQL_HEADER_RELEASE_SYSTEM;
  constexpr uint32_t kPacketCount = 5;
  constexpr uint32_t kSampleCount = 3;
  auto* packets = reinterpret_cast<iree_hal_amd_gpu_aql_packet_t*>(
      control_.contents.data + 512);
  auto* samples = reinterpret_cast<iree_hal_amd_gpu_aql_signal_t*>(
      control_.contents.data + 2048);
  for (uint32_t i = 0; i < kSampleCount; ++i) {
    const iree_hal_amd_gpu_blit_arguments_t fill = {
        .target = control_address_ + 4096 + i * 4,
        .length = 4,
        .pattern = 100 + i,
        .fill = 1,
    };
    std::memcpy(control_.contents.data + 256 + i * sizeof(fill), &fill,
                sizeof(fill));
  }
  packets[0] = Dispatch(pool_.blit, control_address_ + 256, 1, kHeader);
  packets[0].words[14] = 1;
  packets[1] = iree_hal_amd_gpu_aql_indirect_packet(pool_.indirect,
                                                    control_address_ + 1024);
  packets[2] = Dispatch(
      pool_.blit,
      control_address_ + 256 + sizeof(iree_hal_amd_gpu_blit_arguments_t), 1,
      kHeader);
  packets[2].words[14] = 2;
  const iree_hal_amd_gpu_aql_indirect_arguments_t patch = {
      .counts = control_address_ + 1088,
      .header = packets[2].words[0],
      .workgroup_size = {64, 1, 1},
      .count_offsets = {UINT32_MAX, UINT32_MAX, UINT32_MAX},
  };
  std::memcpy(control_.contents.data + 1024, &patch, sizeof(patch));
  const uint32_t counts[] = {1, 1, 1};
  std::memcpy(control_.contents.data + 1088, counts, sizeof(counts));
  packets[2].words[0] = IREE_HAL_AMD_GPU_AQL_HEADER_INVALID;
  packets[3] = iree_hal_amd_gpu_aql_packet_barrier(
      IREE_HAL_AMD_GPU_AQL_HEADER_BARRIER_AND | kHeader, 0, nullptr, 0);
  packets[4] = Dispatch(
      pool_.blit,
      control_address_ + 256 + 2 * sizeof(iree_hal_amd_gpu_blit_arguments_t), 1,
      kHeader);
  packets[4].words[14] = 3;

  uint64_t previous_end = 0;
  for (bool enabled : {true, false, true, false}) {
    SCOPED_TRACE(enabled);
    IREE_ASSERT_OK(IREE_HAL_AMD_STATUS_FROM_AMDF(
        context_->gpu->user_queue_set_dispatch_timestamps(queue_.queue,
                                                          enabled),
        "configure native dispatch timestamps"));
    for (uint32_t iteration = 0; iteration < 17; ++iteration) {
      SCOPED_TRACE(iteration);
      std::memset(control_.contents.data + 1984, 0xA5, (kSampleCount + 2) * 64);
      std::memset(control_.contents.data + 4096, 0, kSampleCount * 4);
      uint64_t position = 0;
      ASSERT_TRUE(iree_hal_amd_gpu_aql_ring_try_reserve(
          &ring_, kPacketCount + 2, &position));
      const iree_hal_amd_gpu_aql_materialize_profile_arguments_t arguments = {
          .materialize =
              {
                  .ring = ring_.device_address,
                  .packets = control_address_ + 512,
                  .ring_address = ring_.device_address,
                  .packet_address = control_address_ + 512,
                  .ring_mask = ring_.capacity_mask,
                  .first =
                      static_cast<uint32_t>(position + 1) & ring_.capacity_mask,
                  .packet_count = kPacketCount,
              },
          .signals = enabled ? control_address_ + 2048 : 0,
          .signal_address = enabled ? control_address_ + 2048 : 0,
      };
      std::memcpy(control_.contents.data + 128, &arguments, sizeof(arguments));
      signal_->value = 1;
      Write(position,
            iree_hal_amd_gpu_aql_materialize_packet(
                pool_.materialize_profile, control_address_ + 128, kHeader));
      Write(position + 1 + kPacketCount,
            iree_hal_amd_gpu_aql_packet_barrier(
                IREE_HAL_AMD_GPU_AQL_HEADER_BARRIER_AND | kHeader, 0, nullptr,
                control_address_));
      pending_ = true;
      iree_hal_amd_gpu_aql_ring_notify(&ring_);
      ASSERT_NO_FATAL_FAILURE(Wait());
      const auto* output =
          reinterpret_cast<const uint32_t*>(control_.contents.data + 4096);
      for (uint32_t i = 0; i < kSampleCount; ++i) {
        EXPECT_EQ(output[i], 100 + i);
        if (enabled) {
          EXPECT_EQ(samples[i].kind, 1u);
          EXPECT_EQ(samples[i].value, 0u);
          EXPECT_GT(samples[i].start_timestamp, previous_end);
          EXPECT_GE(samples[i].end_timestamp, samples[i].start_timestamp);
          previous_end = samples[i].end_timestamp;
          EXPECT_EQ(samples[i].event_mailbox, 0u);
          EXPECT_EQ(samples[i].event_id, 0u);
          EXPECT_EQ(samples[i].reserved0, 0u);
          EXPECT_EQ(samples[i].queue_address, 0u);
          EXPECT_EQ(samples[i].reserved1, 0u);
        } else {
          const auto* bytes = reinterpret_cast<const uint8_t*>(&samples[i]);
          EXPECT_TRUE(std::all_of(bytes, bytes + 64,
                                  [](uint8_t value) { return value == 0xA5; }));
        }
      }
      for (uint32_t offset : {1984u, 2048u + kSampleCount * 64}) {
        const auto* bytes = control_.contents.data + offset;
        EXPECT_TRUE(std::all_of(bytes, bytes + 64,
                                [](uint8_t value) { return value == 0xA5; }));
      }
    }
  }
  EXPECT_GT(ring_.reserved, ring_.capacity * 4);
}

}  // namespace
