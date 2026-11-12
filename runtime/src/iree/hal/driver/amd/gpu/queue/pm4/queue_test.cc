// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/pm4/queue.h"

#include <cstring>
#include <vector>

#include "iree/async/event.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/driver/amd/fabric/memory_backend_test_base.h"
#include "iree/hal/driver/amd/gpu/queue/semaphore.h"
#include "iree/testing/gtest.h"

namespace {

amdf_queue_family_info_t CompleteFamily() {
  return {
      .command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
      .format_version = AMDF_GPU_PM4_QUEUE_FORMAT_VERSION_1,
      .format_features = AMDF_GPU_PM4_FORMAT_FEATURE_ACQUIRE_MEM_GCR,
      .roles = AMDF_QUEUE_ROLE_COMPUTE | AMDF_QUEUE_ROLE_TRANSFER |
               AMDF_QUEUE_ROLE_ATOMIC | AMDF_QUEUE_ROLE_CACHE_CONTROL,
      .cache_operations = AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM |
                          AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM,
      .cache_transition_kinds = AMDF_CACHE_TRANSITION_KINDS_GLOBAL,
      .atomic_capabilities =
          {
              .wait_conditions_32 = AMDF_ATOMIC_WAIT_CONDITION_EQUAL,
              .operations_without_dispatch_32 =
                  AMDF_ATOMIC_OPERATION_STORE | AMDF_ATOMIC_OPERATION_WAIT,
          },
  };
}

TEST(QueueFamilyTest,
     PublicationComesFromProtocolAndNativeProducerCapabilities) {
  auto family = CompleteFamily();
  family.publication_modes = AMDF_QUEUE_PUBLICATION_MODE_KERNEL;
  EXPECT_EQ(iree_hal_amd_gpu_pm4_queue_publication_modes(&family),
            AMDF_QUEUE_PUBLICATION_MODE_KERNEL);
  family.publication_modes |= AMDF_QUEUE_PUBLICATION_MODE_USER;
  EXPECT_EQ(iree_hal_amd_gpu_pm4_queue_publication_modes(&family),
            AMDF_QUEUE_PUBLICATION_MODE_KERNEL);
  family.user_queue_capabilities = AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER;
  EXPECT_EQ(iree_hal_amd_gpu_pm4_queue_publication_modes(&family),
            family.publication_modes);
  family.publication_modes = AMDF_QUEUE_PUBLICATION_MODE_USER;
  EXPECT_EQ(iree_hal_amd_gpu_pm4_queue_publication_modes(&family),
            AMDF_QUEUE_PUBLICATION_MODE_USER);
  family.user_queue_capabilities = 0;
  EXPECT_EQ(iree_hal_amd_gpu_pm4_queue_publication_modes(&family), 0u);
}

TEST(QueueFamilyTest, AutomaticSelectionRequiresTheCompleteExecutionContract) {
  for (uint32_t missing = 0; missing < 11; ++missing) {
    SCOPED_TRACE(missing);
    auto family = CompleteFamily();
    family.publication_modes = AMDF_QUEUE_PUBLICATION_MODE_KERNEL;
    switch (missing) {
      case 0:
        family.command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_AQL;
        break;
      case 1:
        family.format_version = 0;
        break;
      case 2:
        family.roles &= ~AMDF_QUEUE_ROLE_COMPUTE;
        break;
      case 3:
        family.roles &= ~AMDF_QUEUE_ROLE_TRANSFER;
        break;
      case 4:
        family.roles &= ~AMDF_QUEUE_ROLE_ATOMIC;
        break;
      case 5:
        family.roles &= ~AMDF_QUEUE_ROLE_CACHE_CONTROL;
        break;
      case 6:
        family.format_features = 0;
        break;
      case 7:
        family.cache_operations = AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM;
        break;
      case 8:
        family.cache_transition_kinds = 0;
        break;
      case 9:
        family.atomic_capabilities.operations_without_dispatch_32 =
            AMDF_ATOMIC_OPERATION_STORE;
        break;
      case 10:
        family.atomic_capabilities.wait_conditions_32 = 0;
        break;
    }
    EXPECT_EQ(iree_hal_amd_gpu_pm4_queue_publication_modes(&family), 0u);
  }
}

using iree::hal::amd::fabric::FakeNative;
using iree::hal::amd::fabric::MemoryBackendTest;
using iree::hal::cts::Ref;

// The actual queue owns preparation, metadata, publication, and retirement.
// Only the native dependency controls acceptance and executes barrier packets.
class KernelQueueTest : public MemoryBackendTest {
 protected:
  struct NativeQueue {
    // Descriptors accepted by the native API, retaining their original backing.
    std::vector<amdf_gpu_kernel_command_t> commands;
    // Exact bytes seen by each native submit attempt, including BUSY attempts.
    std::vector<std::vector<uint32_t>> attempts;
    // Native completion point after executing an accepted command.
    uint64_t retired = 0;
    // Terminal execution result independently reported by native refresh.
    amdf_status_t terminal_status = AMDF_STATUS_OK;
    // Independent native capacity, which may be smaller than advertised credit.
    bool busy = false;
  };

  static NativeQueue* Native(amdf_kernel_queue_t* queue) {
    return reinterpret_cast<NativeQueue*>(queue);
  }

  static const uint32_t* Words(const amdf_gpu_kernel_command_t& command) {
    const auto* memory =
        reinterpret_cast<const FakeNative::Memory*>(command.memory);
    return reinterpret_cast<const uint32_t*>(
        static_cast<const uint8_t*>(memory->data) + command.byte_offset);
  }

  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(MemoryBackendTest::SetUp());
    native_.user_data = this;
    native_.required_flags = 0;
    native_.host_coherent = true;
    native_.queue_family_ordinals[0] = 0;
    native_.address_kinds[0] = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU;
    native_.atomic_reach[0].scope_32 = AMDF_ATOMIC_SCOPE_DEVICE;
    native_.api.memory_query_access_info = [](amdf_memory_t*, uint32_t,
                                              amdf_memory_access_info_t* info) {
      info->flags =
          AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
      info->atomic_operations_32 =
          AMDF_ATOMIC_OPERATION_WAIT | AMDF_ATOMIC_OPERATION_STORE;
      return AMDF_STATUS_OK;
    };
    native_.api.memory_query_address = [](amdf_memory_t* memory, uint32_t,
                                          amdf_memory_address_kind_t,
                                          uint64_t* address) {
      *address = reinterpret_cast<uintptr_t>(
          reinterpret_cast<FakeNative::Memory*>(memory)->data);
      return AMDF_STATUS_OK;
    };
    gpu_.kernel_queue_create = [](amdf_device_t* device,
                                  const amdf_gpu_kernel_queue_create_info_t*,
                                  amdf_kernel_queue_t** queue) {
      auto* self = static_cast<KernelQueueTest*>(
          reinterpret_cast<FakeNative::Device*>(device)->owner->user_data);
      *queue = reinterpret_cast<amdf_kernel_queue_t*>(
          &self->native_queues_[self->created_++]);
      return AMDF_STATUS_OK;
    };
    native_.api.kernel_queue_query_info = [](amdf_kernel_queue_t*,
                                             amdf_kernel_queue_info_t* info) {
      info->notification_types = AMDF_NATIVE_EVENT_TYPE_BIT_EVENTFD |
                                 AMDF_NATIVE_EVENT_TYPE_BIT_WIN32_EVENT;
      return AMDF_STATUS_OK;
    };
    gpu_.kernel_queue_submit =
        [](amdf_kernel_queue_t* queue,
           const amdf_gpu_kernel_queue_submission_info_t* info,
           uint64_t* point) {
          auto* native = Native(queue);
          EXPECT_EQ(info->command_count, 1u);
          const auto& command = info->commands[0];
          native->attempts.emplace_back(
              Words(command), Words(command) + command.byte_length / 4);
          if (native->busy) {
            return amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
          }
          native->commands.push_back(command);
          *point = native->commands.size();
          return AMDF_STATUS_OK;
        };
    native_.api.kernel_queue_refresh_status =
        [](amdf_kernel_queue_t* queue, amdf_kernel_queue_status_t* status) {
          status->retired_submission = Native(queue)->retired;
          status->terminal_status = Native(queue)->terminal_status;
          return AMDF_STATUS_OK;
        };
    native_.api.kernel_queue_request_notification =
        [](amdf_kernel_queue_t*, uint64_t, const amdf_native_event_t*) {
          return AMDF_STATUS_OK;
        };
    native_.api.kernel_queue_wait = [](amdf_kernel_queue_t* queue,
                                       uint64_t point, uint64_t, uint64_t) {
      return point <= Native(queue)->retired
                 ? AMDF_STATUS_OK
                 : amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
    };
    native_.api.kernel_queue_destroy = [](amdf_kernel_queue_t*) {
      return AMDF_STATUS_OK;
    };
    gpu_.kernel_queue_inactivate = [](amdf_kernel_queue_t*) {
      return AMDF_STATUS_OK;
    };
    IREE_ASSERT_OK(iree_allocator_malloc(allocator_, sizeof(*fabric_),
                                         reinterpret_cast<void**>(&fabric_)));
    iree_atomic_ref_count_init(&fabric_->ref_count);
    iree_slim_mutex_initialize(&fabric_->allocation_mutex);
    fabric_->host_allocator = allocator_;
    fabric_->api = &native_.api;
    context_.fabric = fabric_;
    context_.api = &native_.api;
    context_.gpu = &gpu_;
    context_.device = native_.devices[0];
    context_.event_sink = {
        [](void* user_data, const iree_hal_device_event_t* event) {
          EXPECT_EQ(event->severity, IREE_HAL_DEVICE_EVENT_SEVERITY_ERROR);
          ++static_cast<KernelQueueTest*>(user_data)->diagnostics_;
        },
        this};
    IREE_ASSERT_OK(iree_hal_amd_gpu_context_prepare_completion(
        &context_, AMDF_QUEUE_PUBLICATION_MODE_KERNEL));
    IREE_ASSERT_OK(
        iree_hal_amd_gpu_context_prepare_memory(&context_, UINT32_MAX));
    IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_barrier_profile_initialize(
        {12, 0, 1}, &barrier_profile_));
    iree_notification_initialize(&publication_.notification);
    iree_slim_mutex_initialize(&completion_.drain_mutex);
    access_ = {
        .device = context_.device,
        .requirements =
            {
                .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
                .address_kinds = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU,
            },
    };
    amdf_memory_access_capabilities_t capabilities = {};
    amdf_memory_profile_t profile = {};
    ASSERT_EQ(native_.api.memory_scope_query_device_profile(
                  native_.scope(), 0, 1, &access_, &profile, &capabilities),
              AMDF_STATUS_OK);
    memory_ = {
        .scope = native_.scope(),
        .profile = profile,
        .access_count = 1,
        .accesses = &access_,
        .host_access = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
    };
    for (uint32_t i = 0; i < queues_.size(); ++i) {
      iree_hal_amd_gpu_signal_cache_initialize(&publication_, i, &signals_[i]);
      const iree_hal_amd_gpu_pm4_queue_options_t options = {
          .queue_ordinal = i,
          .source_ordinal = i,
          .queue_count = 2,
          .signals = &signals_[i],
          .publication_mode = AMDF_QUEUE_PUBLICATION_MODE_KERNEL,
          .command_byte_capacity = 4096,
          .memory = &memory_,
          .publication = &publication_,
          .barriers = barrier_profile_,
          .retirement_capacity = 4,
          .argument_capacity = 4096,
          .capture_block_size = 4096,
          .capture_block_count = 4,
      };
      iree_hal_queue_params_t params;
      iree_hal_queue_params_initialize(&params);
      IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_queue_create(
          families_[0], &context_, &options, &params, allocator_,
          queues_[i].out()));
    }
  }

  void TearDown() override {
    for (auto& queue : queues_) {
      if (queue) {
        EXPECT_TRUE(iree_hal_amd_gpu_pm4_queue_is_idle(queue));
        EXPECT_EQ(iree_hal_amd_gpu_pm4_queue_shutdown(queue), AMDF_STATUS_OK);
        queue.reset();
      }
    }
    for (auto& signals : signals_) {
      iree_hal_amd_gpu_signal_cache_deinitialize(&context_, &signals);
    }
    iree_hal_amd_gpu_context_deinitialize_memory(&context_);
    iree_async_event_native_deinitialize(&context_.kernel_event);
    iree_hal_amd_fabric_release(fabric_);
    iree_slim_mutex_deinitialize(&completion_.drain_mutex);
    iree_notification_deinitialize(&publication_.notification);
    MemoryBackendTest::TearDown();
  }

  Ref<iree_hal_semaphore_t> Semaphore() {
    Ref<iree_hal_semaphore_t> semaphore;
    IREE_CHECK_OK(iree_hal_amd_gpu_semaphore_create(
        iree::hal::amd::fabric::test_proactor(), nullptr, 0, 2, allocator_,
        semaphore.out()));
    return semaphore;
  }

  void ExpandFrontiers() {
    std::array sources = {
        iree_hal_amd_gpu_pm4_queue_publication_source(queues_[0]),
        iree_hal_amd_gpu_pm4_queue_publication_source(queues_[1]),
    };
    iree_notification_deinitialize(&publication_.notification);
    IREE_ASSERT_OK(iree_hal_amd_gpu_publication_initialize(
        sources.size(), sources.data(), {}, allocator_, &publication_));
    IREE_EXPECT_OK(iree_hal_amd_gpu_pm4_queue_reserve_frontiers(
        queues_[0], 4, &publication_.notification));
    iree_hal_amd_gpu_publication_deinitialize(&publication_);
    iree_notification_initialize(&publication_.notification);
  }

  void Publish(uint32_t index) {
    const auto source =
        iree_hal_amd_gpu_pm4_queue_publication_source(queues_[index]);
    while (source.publish(source.user_data)) {
    }
  }

  void Poll(uint32_t index) {
    auto source = iree_hal_amd_gpu_pm4_queue_completion_source(queues_[index]);
    bool ready = true;
    while (ready) {
      iree_slim_mutex_lock(&completion_.drain_mutex);
      iree_hal_amd_gpu_completion_scope_t scope = {
          .completion = &completion_,
          .source = &source,
          .fn = source.fn,
      };
      iree_hal_amd_gpu_completion_wait_t wait = {};
      ready =
          source.fn(source.user_data, AMDF_STATUS_OK, &scope, &wait, nullptr);
      EXPECT_FALSE(iree_hal_amd_gpu_completion_scope_finish(&scope));
      iree_slim_mutex_unlock(&completion_.drain_mutex);
    }
  }

  void Execute(uint32_t index) {
    auto& native = native_queues_[index];
    while (native.retired < native.commands.size()) {
      const auto& command = native.commands[native.retired];
      const auto* words = Words(command);
      for (size_t cursor = 0; cursor < command.byte_length / 4;) {
        const uint32_t length = ((words[cursor] >> 16) & 0x3fff) + 2;
        ASSERT_LE(cursor + length, command.byte_length / 4);
        const auto address = [&](uint32_t offset) {
          return uint64_t(words[cursor + offset]) |
                 (uint64_t(words[cursor + offset + 1]) << 32);
        };
        switch ((words[cursor] >> 8) & 255) {
          case 0x10:
            break;    // NOP padding.
          case 0x37:  // WRITE_DATA to the native retirement record.
            std::memcpy(reinterpret_cast<void*>(address(2)), words + cursor + 4,
                        (length - 4) * 4);
            break;
          case 0x3c:  // WAIT_REG_MEM on a dependency signal.
            ASSERT_EQ(*reinterpret_cast<const uint32_t*>(address(2)) &
                          words[cursor + 5],
                      words[cursor + 4]);
            break;
          case 0x46:  // EVENT_WRITE joins preceding shader work.
            EXPECT_EQ(length, 2u);
            EXPECT_EQ(words[cursor + 1], 0x80000407u);
            break;
          case 0x50:  // Zero-byte DMA_DATA joins preceding DMA work.
            EXPECT_EQ(length, 7u);
            EXPECT_EQ(words[cursor + 6], 0u);
            break;
          case 0x58:  // ACQUIRE_MEM completes synchronous cache work.
            EXPECT_EQ(length, 8u);
            break;
          case 0x49:  // RELEASE_MEM publishes the signal or record generation.
            iree_atomic_store(
                reinterpret_cast<iree_atomic_int32_t*>(address(3)),
                words[cursor + 5], iree_memory_order_release);
            break;
          default:
            FAIL() << "Unexpected PM4 barrier opcode "
                   << ((words[cursor] >> 8) & 255);
        }
        cursor += length;
      }
      ++native.retired;
    }
  }

  // Native dependency state with independently controlled per-queue credit.
  std::array<NativeQueue, 2> native_queues_;
  // Number of native queues constructed in the fixture's stable order.
  uint32_t created_ = 0;
  // Native failures delivered through the required device diagnostic sink.
  uint32_t diagnostics_ = 0;
  // Protocol API borrowed by the real context.
  amdf_gpu_api_t gpu_ = {};
  // Native allocation owner shared by both real queues.
  iree_hal_amd_fabric_t* fabric_ = nullptr;
  // Exact native signal domain and allocation executor.
  iree_hal_amd_gpu_context_t context_ = {};
  // Qualified coherent control-storage access.
  amdf_memory_device_access_t access_ = {};
  // Exact gfx12 cache encoding used by both real queue instances.
  iree_hal_amd_gpu_pm4_barrier_profile_t barrier_profile_ = {};
  // Captured native backing profile, retained through queue cleanup.
  iree_hal_amd_fabric_memory_options_t memory_ = {};
  // Production typed publisher, started for synchronous metadata installation.
  iree_hal_amd_gpu_publication_t publication_ = {};
  // Existing native retirement ownership and scope synchronization.
  iree_hal_amd_gpu_completion_t completion_ = {};
  // Native signals with real publication and causal metadata.
  std::array<iree_hal_amd_gpu_signal_cache_t, 2> signals_ = {};
  // Public queues under test, borrowing the fixture's device families.
  std::array<Ref<iree_hal_queue_t>, 2> queues_;
};

TEST_F(KernelQueueTest, MetadataPreservesBusyCommandAndItsCausalFrontier) {
  auto first = Semaphore(), foreign = Semaphore(), second = Semaphore(),
       later = Semaphore();
  iree_hal_semaphore_t* first_handle = first;
  iree_hal_semaphore_t* foreign_handle = foreign;
  iree_hal_semaphore_t* second_handle = second;
  iree_hal_semaphore_t* later_handle = later;
  uint64_t one = 1;
  IREE_ASSERT_OK(iree_hal_queue_barrier(queues_[0], {},
                                        {1, &first_handle, &one}, nullptr, 0));
  IREE_ASSERT_OK(iree_hal_queue_barrier(
      queues_[1], {}, {1, &foreign_handle, &one}, nullptr, 0));
  native_queues_[0].busy = true;
  IREE_ASSERT_OK(iree_hal_queue_barrier(queues_[0], {1, &foreign_handle, &one},
                                        {1, &second_handle, &one}, nullptr, 0));
  ASSERT_EQ(native_queues_[0].attempts.size(), 2u);
  const auto prepared = native_queues_[0].attempts.back();
  IREE_ASSERT_OK(iree_hal_queue_barrier(queues_[0], {},
                                        {1, &later_handle, &one}, nullptr, 0));
  ASSERT_NO_FATAL_FAILURE(ExpandFrontiers());
  EXPECT_EQ(native_queues_[0].attempts.size(), 2u);
  EXPECT_EQ(native_queues_[0].commands.size(), 1u);
  EXPECT_EQ(native_queues_[0].retired, 0u);
  ASSERT_NO_FATAL_FAILURE(Execute(0));
  Poll(0);
  native_queues_[0].busy = false;
  Publish(0);
  ASSERT_EQ(native_queues_[0].attempts.size(), 4u);
  EXPECT_EQ(native_queues_[0].attempts[2], prepared);
  for (auto* semaphore : {second_handle, later_handle}) {
    iree_hal_amd_gpu_semaphore_wait_t wait = {};
    IREE_ASSERT_OK(iree_hal_amd_gpu_semaphore_acquire_wait(semaphore, one,
                                                           &context_, &wait));
    ASSERT_EQ(wait.state, IREE_HAL_AMD_GPU_SEMAPHORE_WAIT_NATIVE);
    std::array<uint64_t, 4> frontier = {};
    iree_hal_amd_gpu_signal_join_frontier(wait.signal, frontier.size(),
                                          frontier.data());
    EXPECT_EQ(frontier[0], semaphore == second_handle ? 2u : 3u);
    EXPECT_EQ(frontier[1], 1u);
    EXPECT_EQ(frontier[2], 0u);
    EXPECT_EQ(frontier[3], 0u);
    iree_hal_amd_gpu_signal_release(wait.signal);
  }
  ASSERT_NO_FATAL_FAILURE(Execute(1));
  Poll(1);
  ASSERT_NO_FATAL_FAILURE(Execute(0));
  Poll(0);
  for (auto* semaphore :
       {first_handle, foreign_handle, second_handle, later_handle}) {
    uint64_t value = 0;
    IREE_EXPECT_OK(iree_hal_semaphore_query(semaphore, &value));
    EXPECT_EQ(value, one);
  }
}

TEST_F(KernelQueueTest, ReleaseOnlyCompletionPreservesEachObserver) {
  const iree_hal_barrier_list_t empty = {};
  const iree_hal_barrier_t release = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .flags = IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
  };
  const iree_hal_barrier_list_t after = {1, &release};
  const iree_hal_queue_barriers_t barriers = {&empty, &after};

  const auto expect_packets =
      [&](const amdf_gpu_kernel_command_t& command,
          const std::vector<uint32_t>& expected_opcodes,
          const std::vector<uint32_t>& expected_releases, bool expect_wait) {
        const uint32_t* words = Words(command);
        std::vector<uint32_t> opcodes;
        std::vector<uint32_t> release_cache_controls;
        uint64_t signal_address = 0;
        uint64_t wait_address = 0;
        for (size_t cursor = 0; cursor < command.byte_length / 4;) {
          const uint32_t length = ((words[cursor] >> 16) & 0x3fff) + 2;
          ASSERT_LE(cursor + length, command.byte_length / 4);
          const uint32_t opcode = (words[cursor] >> 8) & 255;
          if (opcode != 0x10) {
            opcodes.push_back(opcode);
          }
          if (opcode == 0x49) {
            release_cache_controls.push_back(words[cursor + 1] >> 12);
            if (!signal_address) {
              signal_address = uint64_t(words[cursor + 3]) |
                               (uint64_t(words[cursor + 4]) << 32);
            }
          } else if (opcode == 0x3c) {
            wait_address = uint64_t(words[cursor + 2]) |
                           (uint64_t(words[cursor + 3]) << 32);
            EXPECT_EQ(words[cursor + 4], 1u);
            EXPECT_EQ(words[cursor + 5], UINT32_MAX);
          }
          cursor += length;
        }
        EXPECT_EQ(opcodes, expected_opcodes);
        EXPECT_EQ(release_cache_controls, expected_releases);
        EXPECT_EQ(wait_address, expect_wait ? signal_address : 0u);
      };

  auto completion = Semaphore();
  iree_hal_semaphore_t* completion_handle = completion;
  uint64_t one = 1;
  IREE_ASSERT_OK(iree_hal_queue_barrier(
      queues_[0], {}, {1, &completion_handle, &one}, &barriers, 0));
  Publish(0);

  ASSERT_EQ(native_queues_[0].commands.size(), 1u);
  expect_packets(native_queues_[0].commands.back(), {0x49, 0x3c, 0x37, 0x49},
                 {0x1600, 0}, true);

  ASSERT_NO_FATAL_FAILURE(Execute(0));
  Poll(0);
  uint64_t value = 0;
  IREE_ASSERT_OK(iree_hal_semaphore_query(completion, &value));
  EXPECT_EQ(value, one);

  IREE_ASSERT_OK(iree_hal_queue_barrier(queues_[0], {}, {}, &barriers, 0));
  Publish(0);
  ASSERT_EQ(native_queues_[0].commands.size(), 2u);
  expect_packets(native_queues_[0].commands.back(), {0x37, 0x49}, {0x1600},
                 false);
  ASSERT_NO_FATAL_FAILURE(Execute(0));
  Poll(0);
}

TEST_F(KernelQueueTest, FailureReleasesCapacityWaitersBeforeNativeRetirement) {
  std::array<Ref<iree_hal_semaphore_t>, 7> completions;
  uint64_t one = 1;
  for (auto& completion : completions) {
    completion = Semaphore();
    iree_hal_semaphore_t* handle = completion;
    IREE_ASSERT_OK(
        iree_hal_queue_barrier(queues_[0], {}, {1, &handle, &one}, nullptr, 0));
  }
  Publish(0);
  ASSERT_EQ(native_queues_[0].commands.size(), 4u);
  EXPECT_FALSE(iree_hal_amd_gpu_pm4_queue_is_host_idle(queues_[0]));
  native_queues_[0].terminal_status =
      amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  Poll(0);
  Publish(0);
  EXPECT_TRUE(iree_hal_amd_gpu_pm4_queue_is_host_idle(queues_[0]));
  EXPECT_FALSE(iree_hal_amd_gpu_pm4_queue_is_idle(queues_[0]));
  EXPECT_EQ(native_queues_[0].attempts.size(), 4u);
  EXPECT_EQ(native_queues_[0].retired, 0u);
  // Issued operands remain owned until the native API proves access stopped.
  EXPECT_GT(diagnostics_, 0u);
  amdf_status_t status = AMDF_STATUS_OK;
  ASSERT_TRUE(iree_hal_amd_gpu_pm4_queue_try_inactivate(queues_[0], &status));
  ASSERT_EQ(status, AMDF_STATUS_OK);
  ASSERT_TRUE(iree_hal_amd_gpu_pm4_queue_retire_failed(queues_[0],
                                                       IREE_STATUS_DATA_LOSS));
  for (auto& completion : completions) {
    uint64_t value = 0;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                          iree_hal_semaphore_query(completion, &value));
  }
}

}  // namespace
