// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/program.h"

#include <array>
#include <cstring>
#include <vector>

#include "iree/base/internal/atomics.h"
#include "iree/hal/drivers/amd/gpu/cts/test_device.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/barrier.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/commands.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/fixup.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/kmq/storage.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/launch.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/queue.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/umq/ring.h"
#include "iree/hal/drivers/amd/status.h"
#include "iree/testing/status_matchers.h"

namespace {
using iree::hal::amd::gpu::cts::GpuNativeTest;

// Tests the resident owner and native entry/return protocol directly. Public
// command recording has separate CTS; this exercises shader-produced command
// bytes, fetch completion, and causal reuse without a host replay intermediary.
class GpuProgramTest : public GpuNativeTest {
 protected:
  void SetUp() override {
    GpuNativeTest::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    uint32_t family = 0;
    amdf_queue_publication_modes_t modes = 0;
    for (; family < context_->endpoint_info.queue_family_count; ++family) {
      modes = iree_hal_amd_gpu_pm4_queue_publication_modes(
          &context_->families[family]);
      if (modes) {
        break;
      }
    }
    if (!modes) {
      GTEST_SKIP() << "No native PM4 publication family";
    }
    kernel_ = !(modes & AMDF_QUEUE_PUBLICATION_MODE_USER);
    IREE_ASSERT_OK(iree_hal_amd_gpu_context_prepare_completion(
        context_, kernel_ ? AMDF_QUEUE_PUBLICATION_MODE_KERNEL
                          : AMDF_QUEUE_PUBLICATION_MODE_USER));
    const iree_hal_amd_gpu_gfxip_version_t gfxip = {
        context_->target_info.gfx_ip.major, context_->target_info.gfx_ip.minor,
        context_->target_info.gfx_ip.stepping};
    IREE_ASSERT_OK(
        iree_hal_amd_gpu_pm4_barrier_profile_initialize(gfxip, &barriers_));
    ASSERT_NO_FATAL_FAILURE(
        SelectMemory(true, &control_access_, &control_policy_));
    ASSERT_NO_FATAL_FAILURE(
        SelectMemory(false, &program_access_, &program_policy_));
    ASSERT_NO_FATAL_FAILURE(
        Allocate(control_policy_, 65536, &control_, &control_address_));
    std::memset(control_.contents.data, 0, control_.contents.data_length);
    iree_const_byte_span_t code;
    iree_hal_amd_gpu_image_kernel_t descriptor;
    IREE_ASSERT_OK(
        iree_hal_amd_gpu_pm4_fixup_program_query(gfxip, &code, &descriptor));
    uint64_t code_address = 0;
    ASSERT_NO_FATAL_FAILURE(Allocate(program_policy_, code.data_length + 256,
                                     &code_, &code_address));
    std::memcpy(code_.contents.data, code.data, code.data_length);
    IREE_ASSERT_OK(IREE_HAL_AMD_STATUS_FROM_AMDF(
        context_->api->host_mapping_cache_control(
            code_.mapping, AMDF_HOST_CACHE_OPERATION_FLUSH, 0,
            code_.contents.data_length),
        "publish fixup instructions"));
    IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_launch_initialize(
        gfxip, &descriptor, code_address, &fixup_launch_));
    for (auto& queue : queues_native_) {
      if (kernel_) {
        IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_kmq_storage_allocate(
            context_, family, 65536, 16, &control_policy_,
            iree_allocator_system(), &queue.kernel, &queue.ring));
        IREE_ASSERT_OK(IREE_HAL_AMD_STATUS_FROM_AMDF(
            context_->api->memory_query_address(queue.kernel->memory.handle, 0,
                                                AMDF_MEMORY_ADDRESS_GPU,
                                                &queue.address),
            "query command address"));
      } else {
        const amdf_gpu_user_queue_create_info_t request = {
            .type = AMDF_STRUCTURE_TYPE_GPU_USER_QUEUE_CREATE_INFO,
            .structure_size = sizeof(request),
            .queue_family_ordinal = family,
            .priority = AMDF_QUEUE_PRIORITY_NORMAL,
            .producer_mode = AMDF_QUEUE_PRODUCER_MODE_SINGLE,
            .required_capabilities = AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER,
            .ring_byte_length = 65536,
        };
        IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_ring_storage_initialize(
            context_, &request, &queue.user, &queue.ring, &queue.mapping));
      }
    }
  }

  void TearDown() override {
    if (context_) {
      if (HasFatalFailure() && control_.contents.data) {
        // A failed setup/submission can leave the overlap gates closed. Open
        // only those test-owned gates before joining the real native work.
        auto* gates =
            reinterpret_cast<iree_atomic_int32_t*>(control_.contents.data);
        iree_atomic_store(&gates[8], 1, iree_memory_order_release);
        iree_atomic_store(&gates[9], 1, iree_memory_order_release);
      }
      for (auto& queue : queues_native_) {
        if (queue.kernel) {
          IREE_ASSERT_OK(IREE_HAL_AMD_STATUS_FROM_AMDF(
              iree_hal_amd_gpu_pm4_kmq_storage_free(queue.kernel),
              "join KERNEL commands"));
        } else if (queue.user.context) {
          IREE_ASSERT_OK(IREE_HAL_AMD_STATUS_FROM_AMDF(
              iree_hal_amd_gpu_pm4_ring_storage_deinitialize(
                  &queue.user, queue.ring.reserved_words),
              "join USER commands"));
        }
      }
      for (auto* instance : uses_) {
        iree_hal_amd_gpu_pm4_program_release(&program_, instance);
      }
      iree_hal_amd_gpu_pm4_program_deinitialize(&program_);
      iree_hal_amd_fabric_memory_deinitialize(context_->api,
                                              context_->event_sink, &code_);
      iree_hal_amd_fabric_memory_deinitialize(context_->api,
                                              context_->event_sink, &control_);
    }
    GpuNativeTest::TearDown();
  }

  void SelectMemory(bool coherent, amdf_memory_device_access_t* access,
                    iree_hal_amd_fabric_memory_options_t* policy) {
    *access = {
        .device = context_->device,
        .requirements =
            {
                .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE |
                          AMDF_MEMORY_ACCESS_EXECUTE,
                .flags =
                    AMDF_MEMORY_FLAG_DEVICE_ADDRESS |
                    (coherent ? AMDF_MEMORY_FLAG_HOST_COHERENT : UINT64_C(0)),
                .address_kinds = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU,
            },
    };
    amdf_memory_scope_info_t info = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
        .structure_size = sizeof(info),
    };
    IREE_ASSERT_OK(IREE_HAL_AMD_STATUS_FROM_AMDF(
        context_->api->memory_scope_query_info(context_->fabric->system_scope,
                                               &info),
        "query system storage"));
    for (uint32_t i = 0; i < info.memory_profile_count; ++i) {
      amdf_memory_profile_t profile = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
          .structure_size = sizeof(profile),
      };
      amdf_memory_access_capabilities_t capabilities = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
          .structure_size = sizeof(capabilities),
      };
      const auto status = context_->api->memory_scope_query_device_profile(
          context_->fabric->system_scope, i, 1, access, &profile,
          &capabilities);
      if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
        continue;
      }
      IREE_ASSERT_OK(
          IREE_HAL_AMD_STATUS_FROM_AMDF(status, "qualify program storage"));
      if (iree_all_bits_set(profile.roles,
                            AMDF_MEMORY_PROFILE_ROLE_CREATE |
                                AMDF_MEMORY_PROFILE_ROLE_HOST_MAP)) {
        *policy = {
            .scope = context_->fabric->system_scope,
            .profile = profile,
            .access_count = 1,
            .accesses = access,
            .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
            .host_access =
                AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
        };
        return;
      }
    }
    FAIL() << "No admitted host upload profile";
  }

  void Allocate(const iree_hal_amd_fabric_memory_options_t& policy,
                uint64_t length, iree_hal_amd_fabric_memory_t* memory,
                uint64_t* address) {
    IREE_ASSERT_OK(iree_hal_amd_fabric_memory_allocate(
        context_->api, &policy, length, 256, context_->event_sink, memory));
    IREE_ASSERT_OK(IREE_HAL_AMD_STATUS_FROM_AMDF(
        context_->api->memory_query_address(memory->handle, 0,
                                            AMDF_MEMORY_ADDRESS_GPU, address),
        "query backing address"));
  }

  void Seal() {
    // Each resident packet writes one independently checked output. Pointer
    // halves and values are all mutable, with immutable opcodes between them.
    std::vector<uint32_t> words(kOutputs * 5 + 96, 0);
    std::vector<iree_hal_amd_gpu_pm4_fixup_t> fixups;
    uint32_t* cursor = words.data();
    const uint32_t ready = 1;
    cursor = iree_hal_amd_gpu_pm4_emit_write(cursor, control_address_ + 9 * 4,
                                             1, &ready);
    cursor = iree_hal_amd_gpu_pm4_emit_wait32(
        cursor, control_address_ + 8 * 4, 1, UINT32_MAX,
        IREE_HAL_AMD_GPU_PM4_COMPARISON_EQUAL);
    // Packed kernargs need byte alignment, while PM4 pointer fields need only
    // DWORD alignment. The shader must support both without reading old bytes.
    const uint32_t packed_offset = (kOutputs * 5 + 80) * 4 + 1;
    const uint32_t copy_offset =
        static_cast<uint32_t>(cursor - words.data()) * 4;
    cursor = iree_hal_amd_gpu_pm4_emit_dma_copy(cursor, 0, 0, 8);
    cursor = iree_hal_amd_gpu_pm4_emit_dma_flush(cursor);
    fixups.push_back({copy_offset + 8, 3, packed_offset});
    fixups.push_back({copy_offset + 12, 3 | (32u << 24), packed_offset});
    fixups.push_back({copy_offset + 16, 0, kOutputs * 4});
    fixups.push_back({copy_offset + 20, 32u << 24, kOutputs * 4});
    fixups.push_back({packed_offset, 1, kPackedAddend});
    fixups.push_back({packed_offset + 4, 1 | (32u << 24), kPackedAddend});
    for (uint32_t i = 0; i < kOutputs; ++i) {
      const uint32_t offset = static_cast<uint32_t>(cursor - words.data()) * 4;
      const uint32_t initial = 0xCCCCCCCC;
      cursor = iree_hal_amd_gpu_pm4_emit_write(cursor, UINT64_MAX, 1, &initial);
      fixups.push_back({offset + 8, 0, i * 4});
      fixups.push_back({offset + 12, 32u << 24, i * 4});
      fixups.push_back({offset + 16, 1, i});
    }
    // Native KERNEL submission enters at IB1, so return is an explicit chain.
    // Padding precedes that terminal packet; it is never appended after it.
    uint32_t padding =
        (0u - static_cast<uint32_t>(cursor - words.data()) - 4) & 7u;
    if (padding == 1) {
      padding += 8;
    }
    if (padding) {
      cursor = iree_hal_amd_gpu_pm4_emit_nop(cursor, padding);
    }
    const uint32_t exit_offset =
        static_cast<uint32_t>(cursor - words.data()) * 4;
    if (kernel_) {
      cursor = iree_hal_amd_gpu_pm4_emit_indirect_buffer(cursor, 0, 32, true);
      fixups.push_back({exit_offset + 4, 2, 0});
      fixups.push_back({exit_offset + 8, 2 | (32u << 24), 0});
    } else {
      cursor = iree_hal_amd_gpu_pm4_emit_nop(cursor, 4);
    }
    block_ = {
        .offset = 0,
        .word_count = static_cast<uint32_t>(cursor - words.data()),
    };
    IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_program_initialize(
        context_, &program_policy_, 3, 256,
        iree_make_const_byte_span(words.data(), words.size() * 4),
        static_cast<uint32_t>(fixups.size()), fixups.data(),
        iree_allocator_system(), &program_));
  }

  void Submit(uint32_t queue_index, uint32_t slot, uint32_t value,
              const uint64_t* frontier,
              iree_hal_amd_gpu_pm4_program_instance_t** out_instance,
              uint32_t wait_slot = UINT32_MAX,
              uint32_t release_slot = UINT32_MAX) {
    auto& queue = queues_native_[queue_index];
    auto* instance = iree_hal_amd_gpu_pm4_program_try_acquire(
        &program_, queue_index, frontier);
    ASSERT_NE(instance, nullptr);
    *out_instance = instance;
    const uint64_t position = queue.ring.reserved_words;
    auto* words = queue.ring.words + position;
    const uint64_t continuation = queue.address + (position + 128) * 4;
    const uint64_t upload_offset = 4096 + slot * 64;
    const uint64_t values[4] = {
        control_address_ + 8192 + slot * (kOutputs + 2) * 4, value,
        continuation, instance->address};
    const iree_hal_amd_gpu_pm4_fixup_arguments_t arguments = {
        program_.fixup_address, control_address_ + upload_offset + 32,
        instance->address, program_.fixup_count, 0};
    std::memcpy(control_.contents.data + upload_offset, &arguments,
                sizeof(arguments));
    std::memcpy(control_.contents.data + upload_offset + 32, values,
                sizeof(values));
    if (wait_slot != UINT32_MAX) {
      words = iree_hal_amd_gpu_pm4_emit_wait32(
          words, control_address_ + wait_slot * 4, 1, UINT32_MAX,
          IREE_HAL_AMD_GPU_PM4_COMPARISON_EQUAL);
    }
    words = iree_hal_amd_gpu_pm4_emit_compute_flush(words);
    words = iree_hal_amd_gpu_pm4_emit_acquire(words, barriers_.acquire | 1u);
    if (!instance->initialized) {
      words = iree_hal_amd_gpu_pm4_emit_dma_copy(
          words, program_.instances[0]->address, instance->address,
          program_.byte_length);
      words = iree_hal_amd_gpu_pm4_emit_dma_flush(words);
    }
    words = iree_hal_amd_gpu_pm4_fixup_emit(words, &fixup_launch_,
                                            control_address_ + upload_offset,
                                            program_.fixup_count);
    words = iree_hal_amd_gpu_pm4_emit_compute_flush(words);
    words = iree_hal_amd_gpu_pm4_emit_acquire(
        words, barriers_.release | barriers_.shader);
    if (release_slot != UINT32_MAX) {
      const uint32_t released = 1;
      words = iree_hal_amd_gpu_pm4_emit_write(
          words, control_address_ + release_slot * 4, 1, &released);
    }
    if (kernel_) {
      words = iree_hal_amd_gpu_pm4_emit_nop(
          words,
          static_cast<uint32_t>(queue.ring.words + position + 124 - words));
      words = iree_hal_amd_gpu_pm4_emit_indirect_buffer(
          words, instance->address + block_.offset, block_.word_count, true);
      words = iree_hal_amd_gpu_pm4_emit_release32(
          words, 0, control_address_ + slot * 4, 1);
      iree_hal_amd_gpu_pm4_emit_nop(
          words,
          static_cast<uint32_t>(queue.ring.words + position + 160 - words));
      queue.kernel->pending = {
          .memory = queue.kernel->memory.handle,
          .byte_offset = position * 4,
          .byte_length = 128 * 4,
      };
      const auto status =
          iree_hal_amd_gpu_pm4_kmq_storage_submit(queue.kernel, slot);
      if (!amdf_status_is_ok(status)) {
        iree_hal_amd_gpu_pm4_program_cancel(&program_, instance);
        IREE_ASSERT_OK(
            IREE_HAL_AMD_STATUS_FROM_AMDF(status, "submit resident program"));
      }
    } else {
      words = iree_hal_amd_gpu_pm4_emit_indirect_buffer(
          words, instance->address + block_.offset, block_.word_count, false);
      words = iree_hal_amd_gpu_pm4_emit_release32(
          words, 0, control_address_ + slot * 4, 1);
      words = iree_hal_amd_gpu_pm4_emit_notify32(
          words, 0, context_->event_info.mailbox_address,
          context_->event_info.trigger_data,
          context_->event_info.interrupt_context);
      iree_hal_amd_gpu_pm4_emit_nop(
          words,
          static_cast<uint32_t>(queue.ring.words + position + 160 - words));
    }
    queue.ring.reserved_words += 160;
    iree_hal_amd_gpu_pm4_program_publish(&program_, instance, queue_index,
                                         ++queue.epoch);
    uses_.push_back(instance);
    if (!kernel_) {
      iree_atomic_store(reinterpret_cast<iree_atomic_int64_t*>(
                            queue.mapping.write_index_address),
                        queue.ring.reserved_words, iree_memory_order_release);
      const iree_hal_memory_transition_recipe_info_t fence = {
          .host = {.fence_after = IREE_HAL_HOST_CACHE_FENCE_X86_SFENCE},
      };
      iree_hal_memory_transition_execute_host_direct(&fence, nullptr, 1);
      *reinterpret_cast<volatile uint64_t*>(queue.mapping.doorbell_address) =
          queue.ring.reserved_words;
    }
  }

  void Wait(uint32_t queue_index, uint32_t slot) {
    auto& queue = queues_native_[queue_index];
    if (kernel_) {
      IREE_ASSERT_OK(IREE_HAL_AMD_STATUS_FROM_AMDF(
          context_->api->kernel_queue_wait(queue.kernel->queue,
                                           queue.kernel->last_submission,
                                           AMDF_TIMEOUT_INFINITE, 0),
          "wait native command retirement"));
    } else {
      auto* completed =
          reinterpret_cast<iree_atomic_int32_t*>(control_.contents.data);
      while (!iree_atomic_load(&completed[slot], iree_memory_order_acquire)) {
        IREE_ASSERT_OK(IREE_HAL_AMD_STATUS_FROM_AMDF(
            context_->gpu->event_wait(context_->event, AMDF_TIMEOUT_INFINITE),
            "wait resident program notification"));
      }
    }
  }

  void Check(uint32_t slot, uint32_t value) {
    const auto* result = reinterpret_cast<const uint32_t*>(
        control_.contents.data + 8192 + slot * (kOutputs + 2) * 4);
    for (uint32_t i = 0; i < kOutputs; ++i) {
      ASSERT_EQ(result[i], value + i) << "slot=" << slot << " word=" << i;
    }
    uint64_t packed = 0;
    std::memcpy(&packed, result + kOutputs, sizeof(packed));
    EXPECT_EQ(packed, kPackedAddend + value);
  }

  struct Queue {
    // Native USER ownership and its mapped publication operands.
    iree_hal_amd_gpu_pm4_ring_storage_t user = {};
    // Native KERNEL ownership, when selected by family capability.
    iree_hal_amd_gpu_pm4_kmq_storage_t* kernel = nullptr;
    // Executable root/continuation storage with a monotonic host cursor.
    iree_hal_amd_gpu_pm4_ring_t ring = {};
    // USER doorbell and index addresses.
    amdf_user_queue_mapping_info_t mapping = {};
    // GPU address of KERNEL command storage.
    uint64_t address = 0;
    // Actual accepted publication coordinate.
    uint64_t epoch = 0;
  };
  // Many fixups span several workgroups, including a partial last wave.
  static constexpr uint32_t kOutputs = 513;
  // Pointer-style addend exercising carry between separately stored halves.
  static constexpr uint64_t kPackedAddend = UINT64_C(0x789ABCDEFFFFFFF0);
  // Family-selected native transport; no platform preprocessor branches.
  bool kernel_ = false;
  // Captured physical cache contract.
  iree_hal_amd_gpu_pm4_barrier_profile_t barriers_ = {};
  // Native access requirements backing the two qualified policies.
  amdf_memory_device_access_t control_access_ = {};
  // Cached instruction/program access without a coherent-control constraint.
  amdf_memory_device_access_t program_access_ = {};
  // Host/GPU control storage policy.
  iree_hal_amd_fabric_memory_options_t control_policy_ = {};
  // One-time upload policy for resident program/code bytes.
  iree_hal_amd_fabric_memory_options_t program_policy_ = {};
  // Upload tables, result slots, and completion cells.
  iree_hal_amd_fabric_memory_t control_ = {};
  // Exact GPU address of control byte zero.
  uint64_t control_address_ = 0;
  // The embedded production fixup kernel, placed once.
  iree_hal_amd_fabric_memory_t code_ = {};
  // Prepared resource registers for that placed kernel.
  iree_hal_amd_gpu_pm4_launch_t fixup_launch_ = {};
  // Independently published native queues in the same domain.
  std::array<Queue, 3> queues_native_;
  // Owning sealed program and its instance pool.
  iree_hal_amd_gpu_pm4_program_t program_ = {};
  // Entry block independent of instance placement.
  iree_hal_amd_gpu_pm4_program_block_t block_ = {};
  // Native leases retained until completion and checked teardown.
  std::vector<iree_hal_amd_gpu_pm4_program_instance_t*> uses_;
};

TEST_F(GpuProgramTest, PipelinedFixupsCloneAndCausalReuse) {
  ASSERT_NO_FATAL_FAILURE(Seal());
  uint64_t frontier[3] = {};
  // Cold allocation precedes publication. Rejected reservations do not claim
  // an initialized clone or overwrite the initial instance's causal state.
  auto* reservation =
      iree_hal_amd_gpu_pm4_program_try_acquire(&program_, 0, frontier);
  ASSERT_NE(reservation, nullptr);
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_program_grow(&program_));
  iree_hal_amd_gpu_pm4_program_cancel(&program_, reservation);
  iree_hal_amd_gpu_pm4_program_instance_t* first = nullptr;
  ASSERT_NO_FATAL_FAILURE(Submit(0, 0, 0x12345678, frontier, &first));
  frontier[0] = 1;
  iree_hal_amd_gpu_pm4_program_instance_t* same = nullptr;
  ASSERT_NO_FATAL_FAILURE(Submit(0, 1, 0xFEDCBA00, frontier, &same));
  EXPECT_EQ(first, same);
  frontier[0] = 0;
  iree_hal_amd_gpu_pm4_program_instance_t* clone = nullptr;
  // Queue 1 waits until queue 0 is executing the initial resident IB, copies
  // it while live, fully fixes up its own instance, and releases queue 0's
  // device-side gate. No host timing assumption establishes the overlap.
  ASSERT_NO_FATAL_FAILURE(Submit(1, 2, 0x76543210, frontier, &clone, 9, 8));
  EXPECT_NE(first, clone);
  EXPECT_NE(first->address, clone->address);
  // A device wait covers the last writer on queue 1, so queue 2 can acquire
  // that same instance while every host lease remains outstanding.
  frontier[1] = 1;
  iree_hal_amd_gpu_pm4_program_instance_t* joined = nullptr;
  ASSERT_NO_FATAL_FAILURE(Submit(2, 3, 0xABCDE000, frontier, &joined, 2));
  EXPECT_EQ(clone, joined);
  iree_hal_amd_gpu_pm4_program_trim(&program_);
  EXPECT_EQ(program_.instances[1], clone);
  ASSERT_NO_FATAL_FAILURE(Wait(2, 3));
  ASSERT_NO_FATAL_FAILURE(Wait(1, 2));
  ASSERT_NO_FATAL_FAILURE(Wait(0, 1));
  Check(0, 0x12345678);
  Check(1, 0xFEDCBA00);
  Check(2, 0x76543210);
  Check(3, 0xABCDE000);
  for (auto* instance : uses_) {
    iree_hal_amd_gpu_pm4_program_release(&program_, instance);
  }
  uses_.clear();
  iree_hal_amd_gpu_pm4_program_trim(&program_);
  EXPECT_EQ(program_.instances[1], nullptr);
  EXPECT_NE(program_.instances[0], nullptr);
}

}  // namespace
