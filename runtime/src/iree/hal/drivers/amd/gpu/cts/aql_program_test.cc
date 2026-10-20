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
    access_ = pool_.storage.access;
    access_.requirements.flags |=
        AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_QUEUE_STORAGE;
    iree_hal_amd_fabric_memory_options_t unused_resident;
    IREE_ASSERT_OK(iree_hal_amd_gpu_context_query_program_memory(
        context_, options_.queue_family_ordinal, &access_, &control_policy_,
        &unused_resident));
    control_policy_.host_access =
        AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
    const iree_hal_amd_fabric_memory_request_t request = {
        .options = &control_policy_,
        .byte_length = 8192,
        .alignment = 256,
    };
    IREE_ASSERT_OK(iree_hal_amd_gpu_context_acquire_memory_sync(
        context_, 1, &request, &control_));
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
      iree_hal_amd_gpu_context_release_memory_sync(context_, 1, &control_);
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

}  // namespace
