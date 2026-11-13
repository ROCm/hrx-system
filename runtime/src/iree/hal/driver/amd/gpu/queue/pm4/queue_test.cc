// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/pm4/queue.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <semaphore>
#include <vector>

#include "iree/async/event.h"
#include "iree/hal/cts/util/profile_test_util.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/driver/amd/fabric/memory_backend_test_base.h"
#include "iree/hal/driver/amd/gpu/queue/byte_storage_pool.h"
#include "iree/hal/driver/amd/gpu/queue/pm4/state.h"
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
    // One rejected submission after capacity becomes available.
    amdf_status_t rejection = AMDF_STATUS_OK;
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
    native_.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
    native_.host_coherent = true;
    native_.queue_family_ordinals[0] = 0;
    native_families_[0].queue_family_ordinal = 0;
    native_.address_kinds[0] = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU;
    native_.atomic_reach[0].scope_32 = AMDF_ATOMIC_SCOPE_DEVICE;
    for (size_t i = 0; i < binding_descriptions_.size(); ++i) {
      binding_descriptions_[i] = {IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
                                  AMDF_MEMORY_ADDRESS_GPU};
      native_.address_kinds[i] = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU;
    }
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
          if (!amdf_status_is_ok(native->rejection)) {
            const amdf_status_t status = native->rejection;
            native->rejection = AMDF_STATUS_OK;
            return status;
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
        .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
        .host_access = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
    };

    transfer_memory_ = memory_;
    programs_.inputs.options = memory_;
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
          .argument_memory = &memory_,
          .transfer_memory = &transfer_memory_,
          .programs = &programs_,
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
        const bool idle = iree_hal_amd_gpu_pm4_queue_is_idle(queue);
        EXPECT_TRUE(idle);
        if (!idle) {
          // A fatal assertion can leave accepted mock commands unexecuted.
          // Native inactivation joins them before failed retirement, just as
          // device teardown does after a real execution failure.
          iree_hal_amd_gpu_pm4_queue_close(queue);
          const auto source =
              iree_hal_amd_gpu_pm4_queue_publication_source(queue);
          while (source.publish(source.user_data)) {
          }
          amdf_status_t status = AMDF_STATUS_OK;
          ASSERT_TRUE(
              iree_hal_amd_gpu_pm4_queue_try_inactivate(queue, &status));
          ASSERT_EQ(status, AMDF_STATUS_OK);
          ASSERT_TRUE(iree_hal_amd_gpu_pm4_queue_retire_failed(
              queue, IREE_STATUS_ABORTED));
        }
        IREE_EXPECT_OK(iree_hal_amd_gpu_pm4_queue_set_profile(queue, nullptr));
        EXPECT_EQ(iree_hal_amd_gpu_pm4_queue_shutdown(queue), AMDF_STATUS_OK);
        queue.reset();
      }
    }
    for (auto& signals : signals_) {
      iree_hal_amd_gpu_signal_cache_deinitialize(&context_, &signals);
    }
    IREE_EXPECT_OK(iree_hal_amd_gpu_profile_end(profile_));
    transfer_pool_.reset();
    iree_hal_amd_gpu_context_deinitialize_memory(&context_);
    iree_async_event_native_deinitialize(&context_.kernel_event);
    iree_hal_amd_fabric_release(fabric_);
    iree_slim_mutex_deinitialize(&completion_.drain_mutex);
    iree_notification_deinitialize(&publication_.notification);
    MemoryBackendTest::TearDown();
  }

  void BeginProfile() {
    iree::hal::cts::TestProfileSinkInitialize(&profile_sink_);
    const iree_hal_device_profiling_options_t options = {
        .data_families = IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS |
                         IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS,
        .sink = iree::hal::cts::TestProfileSinkAsBase(&profile_sink_),
    };
    IREE_ASSERT_OK(iree_hal_amd_gpu_profile_begin(&options, 1, 2, 1000000,
                                                  allocator_, &profile_));
    IREE_ASSERT_OK(
        iree_hal_amd_gpu_pm4_queue_set_profile(queues_[0], profile_));
  }

  Ref<iree_hal_semaphore_t> Semaphore() {
    Ref<iree_hal_semaphore_t> semaphore;
    IREE_CHECK_OK(iree_hal_amd_gpu_semaphore_create(
        iree::hal::amd::fabric::test_proactor(), nullptr, 0, 2, allocator_,
        semaphore.out()));
    return semaphore;
  }

  Ref<iree_hal_buffer_t> Buffer(uint64_t length) {
    if (!transfer_pool_) {
      IREE_CHECK_OK(
          CreatePool(IREE_HAL_BUFFER_USAGE_TRANSFER,
                     UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
                     transfer_pool_.out()));
    }
    Ref<iree_hal_buffer_t> buffer;
    IREE_CHECK_OK(iree_hal_pool_allocate_buffer(
        transfer_pool_, iree_hal_make_pool_reservation_request(length, 64),
        iree_infinite_timeout(), buffer.out()));
    iree_hal_buffer_mapping_t mapping;
    IREE_CHECK_OK(iree_hal_buffer_map_range(
        buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_WRITE,
        IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, length, &mapping));
    std::memset(mapping.contents.data, 0, mapping.contents.data_length);
    IREE_CHECK_OK(iree_hal_buffer_unmap_range(&mapping));
    return buffer;
  }

  std::vector<uint8_t> Read(iree_hal_buffer_t* buffer, uint64_t length) {
    iree_hal_buffer_mapping_t mapping;
    IREE_CHECK_OK(iree_hal_buffer_map_range(
        buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
        IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, length, &mapping));
    std::vector<uint8_t> result(mapping.contents.data,
                                mapping.contents.data + length);
    IREE_CHECK_OK(iree_hal_buffer_unmap_range(&mapping));
    return result;
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

  void Execute(uint32_t index, uint64_t command_count = UINT64_MAX) {
    auto& native = native_queues_[index];
    const uint64_t end =
        native.retired +
        std::min<uint64_t>(command_count,
                           native.commands.size() - native.retired);
    while (native.retired < end) {
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
          case 0x40:  // Confirmed immediate COPY_DATA store.
            ASSERT_EQ(words[cursor + 1] & 0xf, 5u);
            ASSERT_NE(words[cursor + 1] & (1u << 20), 0u);
            std::memcpy(reinterpret_cast<void*>(address(4)), words + cursor + 2,
                        words[cursor + 1] & (1u << 16) ? 8 : 4);
            break;
          case 0x46:  // EVENT_WRITE joins preceding shader work.
            EXPECT_EQ(length, 2u);
            EXPECT_EQ(words[cursor + 1], 0x80000407u);
            break;
          case 0x50: {  // DMA_DATA or its zero-byte completion join.
            EXPECT_EQ(length, 7u);
            const uint32_t byte_length = words[cursor + 6];
            if (byte_length) {
              if (((words[cursor + 1] >> 29) & 3) == 2) {
                ASSERT_EQ(byte_length % 4, 0u);
                auto* target = reinterpret_cast<uint32_t*>(address(4));
                for (uint32_t i = 0; i < byte_length / 4; ++i) {
                  target[i] = words[cursor + 2];
                }
              } else {
                std::memcpy(reinterpret_cast<void*>(address(4)),
                            reinterpret_cast<const void*>(address(2)),
                            byte_length);
              }
            }
            break;
          }
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

  // Real profiling session, retained through queue retirement and shutdown.
  iree_hal_amd_gpu_profile_t* profile_ = nullptr;
  // Host sink receiving the bounded profiling stream.
  iree::hal::cts::TestProfileSink profile_sink_;
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
  // Independent explicit-payload policy, borrowed through native retirement.
  iree_hal_amd_fabric_memory_options_t transfer_memory_ = {};
  // Borrowed input policy; these control-only queues execute no builtin shader.
  iree_hal_amd_gpu_pm4_program_pool_t programs_ = {};
  // Production typed publisher, started for synchronous metadata installation.
  iree_hal_amd_gpu_publication_t publication_ = {};
  // Existing native retirement ownership and scope synchronization.
  iree_hal_amd_gpu_completion_t completion_ = {};
  // Native signals with real publication and causal metadata.
  std::array<iree_hal_amd_gpu_signal_cache_t, 2> signals_ = {};
  // Public queues under test, borrowing the fixture's device families.
  std::array<Ref<iree_hal_queue_t>, 2> queues_;
  // Public transfer allocation owner, retained through every captured buffer.
  Ref<iree_hal_pool_t> transfer_pool_;
};

TEST_F(KernelQueueTest, PayloadGrowthPreservesLaunchArgumentCapacity) {
  auto buffer = Buffer(40000);
  auto* queue =
      reinterpret_cast<iree_hal_amd_gpu_pm4_queue_t*>(queues_[0].get());
  auto* arguments = queue->producer.arguments;
  ASSERT_EQ(queue->producer.transfers.current, nullptr);
  const std::vector<uint8_t> first(11117, 0x39), second(22001, 0xA7);
  IREE_ASSERT_OK(iree_hal_queue_update(queues_[0], {}, {}, first.data(), 0,
                                       buffer, 0, first.size(), nullptr, 0));
  iree_hal_amd_gpu_context_join_memory(&context_);
  Publish(0);
  auto* first_payload = queue->producer.transfers.current;
  ASSERT_NE(first_payload, nullptr);
  EXPECT_EQ(first_payload->capacity, 16384u);
  IREE_ASSERT_OK(iree_hal_queue_update(queues_[0], {}, {}, second.data(), 0,
                                       buffer, first.size(), second.size(),
                                       nullptr, 0));
  iree_hal_amd_gpu_context_join_memory(&context_);
  Publish(0);
  EXPECT_EQ(queue->producer.arguments, arguments);
  EXPECT_EQ(arguments->capacity, 4096u);
  EXPECT_EQ(arguments->end, 0u);
  EXPECT_EQ(queue->producer.transfers.retired, first_payload);
  EXPECT_EQ(queue->producer.transfers.current->capacity, 32768u);
  ASSERT_NO_FATAL_FAILURE(Execute(0));
  Poll(0);
  std::vector<uint8_t> expected(first.size() + second.size());
  std::copy(first.begin(), first.end(), expected.begin());
  std::copy(second.begin(), second.end(), expected.begin() + first.size());
  EXPECT_EQ(Read(buffer, expected.size()), expected);
  iree_hal_amd_gpu_pm4_queue_trim(queues_[0]);
  Publish(0);
  EXPECT_EQ(queue->producer.transfers.retired, nullptr);
}

TEST_F(KernelQueueTest, RejectionRollsBackTheBorrowedPayloadGeneration) {
  auto buffer = Buffer(2048);
  auto rejected = Semaphore(), completed = Semaphore();
  iree_hal_semaphore_t* rejected_handle = rejected;
  iree_hal_semaphore_t* completed_handle = completed;
  uint64_t one = 1;
  auto* queue =
      reinterpret_cast<iree_hal_amd_gpu_pm4_queue_t*>(queues_[0].get());
  const std::vector<uint8_t> warm(73, 0x19), pending(17, 0x5B),
      large(1000, 0xD7);
  IREE_ASSERT_OK(iree_hal_queue_update(queues_[0], {}, {}, warm.data(), 0,
                                       buffer, 0, warm.size(), nullptr, 0));
  iree_hal_amd_gpu_context_join_memory(&context_);
  Publish(0);
  auto* borrowed = queue->producer.transfers.current;
  ASSERT_NE(borrowed, nullptr);
  ASSERT_EQ(borrowed->end, warm.size());
  ASSERT_EQ(native_queues_[0].commands.size(), 1u);

  std::binary_semaphore allocation_entered(0), resume_allocation(0);
  native_.before_create = [&] {
    allocation_entered.release();
    resume_allocation.acquire();
  };
  IREE_EXPECT_OK(iree_hal_queue_update(
      queues_[0], {}, {1, &completed_handle, &one}, large.data(), 0, buffer,
      256, large.size(), nullptr, 0));
  allocation_entered.acquire();
  native_queues_[0].busy = true;
  IREE_EXPECT_OK(iree_hal_queue_update(
      queues_[0], {}, {1, &rejected_handle, &one}, pending.data(), 0, buffer,
      80, pending.size(), nullptr, 0));
  EXPECT_EQ(queue->kernel->rollback.storage, borrowed);
  EXPECT_EQ(queue->kernel->rollback.bytes, warm.size());
  EXPECT_NE(queue->kernel->pending.byte_length, 0u);
  const uint64_t reserved_end = borrowed->end;
  resume_allocation.release();
  iree_hal_amd_gpu_context_join_memory(&context_);
  native_.before_create = {};
  Publish(0);
  ASSERT_NE(queue->producer.transfers.current, borrowed);
  EXPECT_EQ(queue->producer.transfers.retired, borrowed);
  EXPECT_EQ(queue->producer.transfers.current->begin, reserved_end);
  EXPECT_EQ(queue->kernel->rollback.storage, borrowed);

  ASSERT_NO_FATAL_FAILURE(Execute(0));
  Poll(0);
  native_queues_[0].busy = false;
  native_queues_[0].rejection =
      amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  Publish(0);
  EXPECT_EQ(borrowed->end, warm.size());
  EXPECT_EQ(queue->producer.transfers.current->end,
            reserved_end + large.size());
  EXPECT_EQ(queue->producer.arguments->end, 0u);
  uint64_t value = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_hal_semaphore_query(rejected, &value));
  ASSERT_NO_FATAL_FAILURE(Execute(0));
  Poll(0);
  IREE_ASSERT_OK(iree_hal_semaphore_query(completed, &value));
  EXPECT_EQ(value, one);
  std::vector<uint8_t> expected(2048, 0);
  std::copy(warm.begin(), warm.end(), expected.begin());
  std::copy(large.begin(), large.end(), expected.begin() + 256);
  EXPECT_EQ(Read(buffer, expected.size()), expected);
  iree_hal_amd_gpu_pm4_queue_trim(queues_[0]);
  Publish(0);
  EXPECT_EQ(queue->producer.transfers.retired, nullptr);
}

TEST_F(KernelQueueTest, PacketStorageSurvivesBusyAndRejectedDispatches) {
  auto* queue =
      reinterpret_cast<iree_hal_amd_gpu_pm4_queue_t*>(queues_[0].get());
  EXPECT_EQ(queue->packets, nullptr);
  const iree_hal_amd_gpu_metadata_kernel_t metadata = {};
  iree_hal_amd_gpu_kernarg_layout_t layout = {
      .byte_length = 4096,
      .alignment = 8,
      .native_byte_length = 4096,
  };
  const iree_hal_amd_gpu_pm4_dispatch_packet_info_t packet_info = {
      .kernel_object = UINT64_C(0x123456780040),
  };
  iree_hal_amd_gpu_pm4_function_t function = {};
  function.base.metadata = &metadata;
  function.base.kernarg = &layout;
  function.packet = &packet_info;
  iree_hal_amd_gpu_pm4_dispatch_t dispatch = {};
  for (uint32_t axis = 0; axis < 3; ++axis) {
    dispatch.geometry.workgroup_size[axis] = 1;
    dispatch.geometry.workgroup_count[axis] = 1;
  }
  std::array<uint8_t, 4096> arguments;
  auto* original_arguments = queue->producer.arguments;
  iree_hal_amd_gpu_byte_storage_t* packets = nullptr;
  for (uint32_t iteration = 0; iteration < 5; ++iteration) {
    SCOPED_TRACE(iteration);
    // One accepted predecessor keeps BUSY a valid capacity response. The
    // unaccepted shader will be rejected, so this native fixture need not
    // simulate shader execution to exercise its real publication ownership.
    IREE_ASSERT_OK(iree_hal_queue_barrier(queues_[0], {}, {}, nullptr, 0));
    native_queues_[0].busy = true;
    const uint64_t position = (uint64_t)iree_atomic_load(
        &queue->producer.count, iree_memory_order_acquire);
    arguments.fill((uint8_t)(iteration + 1));
    dispatch.geometry.workgroup_count[0] = iteration + 1;
    auto rejected = Semaphore();
    auto* rejected_handle = rejected.get();
    uint64_t one = 1;
    iree_hal_amd_gpu_pm4_submission_t* submission = nullptr;
    IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_submission_create(
        &queue->captures, false, {}, {1, &rejected_handle, &one}, nullptr,
        &function, dispatch, {},
        iree_make_const_byte_span(arguments.data(), arguments.size()), {},
        &queue->options.barriers, nullptr,
        IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS, &submission));
    iree_hal_amd_gpu_publication_enqueue(&publication_, 0, &queue->publisher,
                                         &submission->entry.publication);
    Publish(0);
    iree_hal_amd_gpu_context_join_memory(&context_);
    Publish(0);
    ASSERT_NE(queue->kernel->pending.byte_length, 0u);
    ASSERT_NE(queue->packets, nullptr);
    if (!packets) {
      packets = queue->packets;
    }
    EXPECT_EQ(queue->packets, packets);
    EXPECT_EQ(packets->capacity, 4 * 64u);
    EXPECT_EQ(queue->producer.arguments, original_arguments);
    EXPECT_EQ(original_arguments->capacity, arguments.size());
    EXPECT_EQ(original_arguments->end, arguments.size());
    EXPECT_EQ(std::memcmp(original_arguments->memory.contents.data,
                          arguments.data(), arguments.size()),
              0);
    const auto* packet =
        reinterpret_cast<const iree_hal_amd_gpu_pm4_dispatch_packet_t*>(
            packets->memory.contents.data) +
        (position & 3);
    EXPECT_EQ(packet->kernarg_address, original_arguments->address);
    EXPECT_EQ(packet->kernel_object, packet_info.kernel_object);
    EXPECT_EQ(packet->grid_size[0], iteration + 1);
    const auto preserved = *packet;
    Publish(0);
    EXPECT_EQ(std::memcmp(packet, &preserved, sizeof(preserved)), 0);
    ASSERT_NO_FATAL_FAILURE(Execute(0));
    Poll(0);
    native_queues_[0].busy = false;
    native_queues_[0].rejection =
        amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
    Publish(0);
    EXPECT_EQ(queue->kernel->pending.byte_length, 0u);
    EXPECT_EQ(original_arguments->end, 0u);
    EXPECT_EQ(queue->packets, packets);
    uint64_t value = 0;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                          iree_hal_semaphore_query(rejected, &value));
    EXPECT_TRUE(iree_hal_amd_gpu_pm4_queue_is_idle(queues_[0]));
  }
}

TEST_F(KernelQueueTest, SampleRejectionRetainsTheGenerationAcrossNativeGrowth) {
  ASSERT_NO_FATAL_FAILURE(BeginProfile());
  auto* queue =
      reinterpret_cast<iree_hal_amd_gpu_pm4_queue_t*>(queues_[0].get());
  auto* original = queue->samples->current;
  ASSERT_EQ(original->capacity, 64u);
  IREE_ASSERT_OK(iree_hal_queue_barrier(queues_[0], {}, {}, nullptr, 0));
  native_queues_[0].busy = true;

  // Enter at the real deferred capture boundary with admitted function facts.
  // Native acceptance is deliberately rejected before any shader could execute.
  const iree_hal_amd_gpu_metadata_kernel_t metadata = {};
  iree_hal_amd_gpu_kernarg_layout_t layout = {
      .byte_length = 16,
      .alignment = 8,
  };
  iree_hal_amd_gpu_pm4_function_t function = {};
  function.base.metadata = &metadata;
  function.base.kernarg = &layout;
  iree_hal_amd_gpu_pm4_dispatch_t dispatch = {};
  for (uint32_t i = 0; i < 3; ++i) {
    dispatch.geometry.workgroup_size[i] = 1;
    dispatch.geometry.workgroup_count[i] = 1;
  }
  const std::array<uint64_t, 2> arguments = {11, 17};
  auto rejected = Semaphore();
  auto* rejected_handle = rejected.get();
  uint64_t one = 1;
  iree_hal_amd_gpu_pm4_submission_t* submission = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_submission_create(
      &queue->captures, true, {}, {1, &rejected_handle, &one}, nullptr,
      &function, dispatch, {},
      iree_make_const_byte_span(arguments.data(), sizeof(arguments)), {},
      &queue->options.barriers, nullptr,
      IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS, &submission));
  auto* operation = iree_hal_amd_gpu_pm4_submission_profile(submission);
  iree_hal_amd_gpu_profile_operation_initialize(
      profile_, 0, IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH, 0, 1, 1,
      operation);
  operation->completion.count = 1;
  operation->completion.sample_stride =
      sizeof(iree_hal_amd_gpu_profile_sample_t);
  iree_hal_amd_gpu_publication_enqueue(&publication_, 0, &queue->publisher,
                                       &submission->entry.publication);
  Publish(0);
  ASSERT_NE(queue->kernel->pending.byte_length, 0u);
  EXPECT_EQ(original->end, 16u);
  EXPECT_EQ(queue->producer.arguments->end, 16u);
  EXPECT_EQ(queue->samples->pending, original);

  // A separate native result is allowed to install while that command is BUSY.
  iree_hal_amd_gpu_pm4_submission_t* waiting = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_submission_create(
      &queue->captures, false, {}, {}, nullptr, nullptr, {}, {},
      iree_const_byte_span_empty(), {}, &queue->options.barriers, nullptr, 0,
      &waiting));
  ASSERT_TRUE(iree_hal_amd_gpu_publisher_try_acquire(&queue->publisher));
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_growth_begin(
      &queue->growth, IREE_HAL_AMD_GPU_PM4_ENTRY_GROW_SAMPLES, waiting, 128));
  iree_hal_amd_gpu_publisher_handoff(&queue->publisher, nullptr);
  iree_hal_amd_gpu_context_join_memory(&context_);
  Publish(0);
  ASSERT_NE(queue->samples->current, original);
  EXPECT_EQ(queue->samples->retired, original);
  EXPECT_EQ(queue->samples->current->begin, 16u);
  EXPECT_EQ(queue->samples->pending, original);

  ASSERT_NO_FATAL_FAILURE(Execute(0));
  Poll(0);
  native_queues_[0].busy = false;
  native_queues_[0].rejection =
      amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  Publish(0);
  EXPECT_EQ(original->end, 0u);
  EXPECT_EQ(queue->samples->current->end, 16u);
  EXPECT_EQ(queue->samples->pending, nullptr);
  EXPECT_EQ(queue->producer.arguments->end, 0u);
  uint64_t value = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_hal_semaphore_query(rejected, &value));
  ASSERT_NO_FATAL_FAILURE(Execute(0));
  Poll(0);
  iree_hal_amd_gpu_pm4_queue_trim(queues_[0]);
  Publish(0);
  EXPECT_EQ(queue->samples->retired, nullptr);
}

TEST_F(KernelQueueTest, NarrowFillAcquiresPayloadWithoutUsingArguments) {
  auto buffer = Buffer(16);
  auto* queue =
      reinterpret_cast<iree_hal_amd_gpu_pm4_queue_t*>(queues_[0].get());
  const uint8_t pattern = 0xB5;
  IREE_ASSERT_OK(iree_hal_queue_fill(queues_[0], {}, {}, buffer, 1, 9, &pattern,
                                     sizeof(pattern), nullptr, 0));
  iree_hal_amd_gpu_context_join_memory(&context_);
  Publish(0);
  ASSERT_NE(queue->producer.transfers.current, nullptr);
  EXPECT_EQ(queue->producer.transfers.current->capacity, 8u);
  EXPECT_EQ(queue->producer.arguments->end, 0u);
  ASSERT_NO_FATAL_FAILURE(Execute(0));
  Poll(0);
  const std::vector<uint8_t> expected = {0,    0xB5, 0xB5, 0xB5, 0xB5, 0xB5,
                                         0xB5, 0xB5, 0xB5, 0xB5, 0,    0,
                                         0,    0,    0,    0};
  EXPECT_EQ(Read(buffer, expected.size()), expected);
}

TEST_F(KernelQueueTest, MetadataPreservesBusyCommandAndItsCausalFrontier) {
  ASSERT_NO_FATAL_FAILURE(BeginProfile());
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
  std::array<iree_hal_semaphore_t*, 2> dependencies = {first_handle,
                                                       foreign_handle};
  std::array<uint64_t, 2> values = {one, one};
  IREE_ASSERT_OK(iree_hal_queue_barrier(queues_[0],
                                        {2, dependencies.data(), values.data()},
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
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  const auto event = std::find_if(
      profile_sink_.queue_events.begin(), profile_sink_.queue_events.end(),
      [](const auto& value) { return value.wait_count == 2; });
  ASSERT_NE(event, profile_sink_.queue_events.end());
  EXPECT_EQ(event->barrier_count, 1u);
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
          const std::vector<uint32_t>& expected_releases) {
        const uint32_t* words = Words(command);
        std::vector<uint32_t> opcodes;
        std::vector<uint32_t> release_cache_controls;
        for (size_t cursor = 0; cursor < command.byte_length / 4;) {
          const uint32_t length = ((words[cursor] >> 16) & 0x3fff) + 2;
          ASSERT_LE(cursor + length, command.byte_length / 4);
          const uint32_t opcode = (words[cursor] >> 8) & 255;
          if (opcode != 0x10) {
            opcodes.push_back(opcode);
          }
          if (opcode == 0x49) {
            release_cache_controls.push_back(words[cursor + 1] >> 12);
          }
          cursor += length;
        }
        EXPECT_EQ(opcodes, expected_opcodes);
        EXPECT_EQ(release_cache_controls, expected_releases);
      };

  auto completion = Semaphore();
  iree_hal_semaphore_t* completion_handle = completion;
  uint64_t one = 1;
  IREE_ASSERT_OK(iree_hal_queue_barrier(
      queues_[0], {}, {1, &completion_handle, &one}, &barriers, 0));
  Publish(0);

  ASSERT_EQ(native_queues_[0].commands.size(), 1u);
  expect_packets(native_queues_[0].commands.back(), {0x49}, {0x1600});

  Poll(0);
  EXPECT_FALSE(iree_hal_amd_gpu_pm4_queue_is_idle(queues_[0]));
  ASSERT_NO_FATAL_FAILURE(Execute(0));
  // Native status must join the entire command extent even when its signal
  // is already visible. KERNEL backing cannot be reused at the earlier EOP.
  --native_queues_[0].retired;
  Poll(0);
  uint64_t value = 0;
  IREE_ASSERT_OK(iree_hal_semaphore_query(completion, &value));
  EXPECT_EQ(value, 0u);
  ++native_queues_[0].retired;
  Poll(0);
  IREE_ASSERT_OK(iree_hal_semaphore_query(completion, &value));
  EXPECT_EQ(value, one);

  IREE_ASSERT_OK(iree_hal_queue_barrier(queues_[0], {}, {}, &barriers, 0));
  Publish(0);
  ASSERT_EQ(native_queues_[0].commands.size(), 2u);
  expect_packets(native_queues_[0].commands.back(), {0x37, 0x49}, {0x1600});
  ASSERT_NO_FATAL_FAILURE(Execute(0));
  Poll(0);
}

TEST_F(KernelQueueTest, LocalExecutionAndVisibilityJoinOnlyTheirOwnProducers) {
  ASSERT_NO_FATAL_FAILURE(BeginProfile());
  struct Scenario {
    // Which producer queues contribute dependencies to queue zero.
    uint32_t producer_mask;
    // Execution domain joined before the consumer payload.
    iree_hal_execution_stage_t source_stage;
    // Whether that boundary performs full system acquisition.
    bool acquire;
    // Producer queues whose explicit memory predicates remain in the stream.
    uint32_t packet_producer_mask;
  };
  const Scenario scenarios[] = {
      {1, IREE_HAL_EXECUTION_STAGE_DISPATCH, true, 0},
      {2, IREE_HAL_EXECUTION_STAGE_DISPATCH, true, 2},
      {3, IREE_HAL_EXECUTION_STAGE_DISPATCH, true, 2},
      {1, IREE_HAL_EXECUTION_STAGE_DISPATCH, false, 1},
      {1, IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE, true, 1},
      {1, IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE, false, 1},
  };
  for (const auto& scenario : scenarios) {
    SCOPED_TRACE(scenario.producer_mask);
    SCOPED_TRACE(scenario.source_stage);
    SCOPED_TRACE(scenario.acquire);
    auto local = Semaphore(), foreign = Semaphore(), consumer = Semaphore();
    std::array<iree_hal_semaphore_t*, 2> producers = {local, foreign};
    auto* consumer_handle = consumer.get();
    std::array<uint64_t, 2> values = {1, 1};
    std::vector<uint64_t> expected_addresses;
    for (uint32_t i = 0; i < 2; ++i) {
      IREE_ASSERT_OK(iree_hal_queue_barrier(
          queues_[i], {}, {1, &producers[i], values.data()}, nullptr, 0));
      Publish(i);
      iree_hal_amd_gpu_semaphore_wait_t point = {};
      IREE_ASSERT_OK(iree_hal_amd_gpu_semaphore_acquire_wait(
          producers[i], 1, &context_, &point));
      ASSERT_EQ(point.state, IREE_HAL_AMD_GPU_SEMAPHORE_WAIT_NATIVE);
      if (scenario.packet_producer_mask & (1u << i)) {
        expected_addresses.push_back(point.signal->device_address);
      }
      iree_hal_amd_gpu_signal_release(point.signal);
    }
    std::vector<iree_hal_semaphore_t*> waits;
    for (uint32_t i = 0; i < 2; ++i) {
      if (scenario.producer_mask & (1u << i)) {
        waits.push_back(producers[i]);
      }
    }
    const iree_hal_barrier_t acquire = {
        .source_stage_mask = scenario.source_stage,
        .flags = static_cast<iree_hal_barrier_flags_t>(
            scenario.acquire ? IREE_HAL_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE
                             : IREE_HAL_BARRIER_FLAG_NONE),
    };
    const iree_hal_barrier_list_t before = {1, &acquire}, after = {};
    const iree_hal_queue_barriers_t barriers = {&before, &after};
    IREE_ASSERT_OK(iree_hal_queue_barrier(
        queues_[0], {waits.size(), waits.data(), values.data()},
        {1, &consumer_handle, values.data()}, &barriers, 0));
    Publish(0);
    const auto& command = native_queues_[0].commands.back();
    const auto* words = Words(command);
    std::vector<uint32_t> opcodes;
    std::vector<uint64_t> addresses;
    for (size_t cursor = 0; cursor < command.byte_length / 4;) {
      const uint32_t length = ((words[cursor] >> 16) & 0x3fff) + 2;
      ASSERT_LE(cursor + length, command.byte_length / 4);
      const uint32_t opcode = (words[cursor] >> 8) & 255;
      if (opcode != 0x10) {
        opcodes.push_back(opcode);
      }
      if (opcode == 0x3c) {
        addresses.push_back(uint64_t(words[cursor + 2]) |
                            (uint64_t(words[cursor + 3]) << 32));
      }
      if (opcode == 0x58) {
        const std::vector<uint32_t> expected = {
            0xc0065800, 0, UINT32_MAX, 0xff,
            0,          0, 0x0a,       barrier_profile_.acquire};
        EXPECT_EQ(
            std::vector<uint32_t>(words + cursor, words + cursor + length),
            expected);
      }
      cursor += length;
    }
    EXPECT_EQ(addresses, expected_addresses);
    std::vector<uint32_t> expected(expected_addresses.size(), 0x3c);
    if (scenario.source_stage == IREE_HAL_EXECUTION_STAGE_DISPATCH) {
      expected.push_back(0x46);
    }
    if (scenario.acquire) {
      expected.push_back(0x58);
    }
    expected.push_back(0x49);
    EXPECT_EQ(opcodes, expected);

    // Publication retains every producer even when its memory wait vanished.
    auto* queue =
        reinterpret_cast<iree_hal_amd_gpu_pm4_queue_t*>(queues_[0].get());
    const uint64_t position = native_queues_[0].commands.size() - 1;
    auto* submission = iree_hal_amd_gpu_pm4_retirement_submission(
        &queue->issued[position & queue->retirement.capacity_mask].action);
    EXPECT_EQ(submission->native_wait_count, waits.size());
    IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
    ASSERT_FALSE(profile_sink_.queue_events.empty());
    EXPECT_EQ(profile_sink_.queue_events.back().barrier_count,
              expected_addresses.size());
    ASSERT_NO_FATAL_FAILURE(Execute(1));
    ASSERT_NO_FATAL_FAILURE(Execute(0));
    Poll(1);
    Poll(0);
    uint64_t value = 0;
    IREE_ASSERT_OK(iree_hal_semaphore_query(consumer, &value));
    EXPECT_EQ(value, 1u);
  }
}

TEST_F(KernelQueueTest, TransfersAndControlShareTheLocalVisibilityBoundary) {
  native_.atomics[0][0] = AMDF_ATOMIC_OPERATION_STORE;
  Ref<iree_hal_pool_t> pool;
  Ref<iree_hal_buffer_t> buffer;
  // Queue-native atomic targets require the provider's coherent control view.
  // Queue-owned allocations retain their independently selected access policy.
  native_.access_flags[0] =
      AMDF_MEMORY_FLAG_QUEUE_STORAGE | AMDF_MEMORY_FLAG_HOST_COHERENT;
  iree_status_t status = CreatePool(
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_STORAGE,
      UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
      IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_BUFFER_USAGE_NONE,
      {.device_scope_32 = IREE_HAL_ATOMIC_OPERATION_FLAG_STORE}, pool.out());
  if (iree_status_is_ok(status)) {
    status = iree_hal_pool_allocate_buffer(
        pool, iree_hal_make_pool_reservation_request(4, 64),
        iree_infinite_timeout(), buffer.out());
  }
  native_.access_flags[0] = 0;
  IREE_ASSERT_OK(status);
  for (bool control : {false, true}) {
    SCOPED_TRACE(control);
    auto producer = Semaphore(), consumer = Semaphore();
    auto* producer_handle = producer.get();
    auto* consumer_handle = consumer.get();
    uint64_t one = 1;
    IREE_ASSERT_OK(iree_hal_queue_barrier(
        queues_[0], {}, {1, &producer_handle, &one}, nullptr, 0));
    const uint32_t pattern = control ? 0x17171717 : 0x39393939;
    if (control) {
      IREE_ASSERT_OK(iree_hal_queue_atomic_store(
          queues_[0], {1, &producer_handle, &one}, {1, &consumer_handle, &one},
          buffer, 0, {.value = pattern, .width = IREE_HAL_ATOMIC_WIDTH_32},
          nullptr));
    } else {
      IREE_ASSERT_OK(iree_hal_queue_fill(
          queues_[0], {1, &producer_handle, &one}, {1, &consumer_handle, &one},
          buffer, 0, 4, &pattern, sizeof(pattern), nullptr, 0));
    }
    Publish(0);
    const auto& command = native_queues_[0].commands.back();
    const auto* words = Words(command);
    std::vector<uint32_t> opcodes;
    for (size_t cursor = 0; cursor < command.byte_length / 4;) {
      const uint32_t length = ((words[cursor] >> 16) & 0x3fff) + 2;
      ASSERT_LE(cursor + length, command.byte_length / 4);
      const uint32_t opcode = (words[cursor] >> 8) & 255;
      if (opcode != 0x10) {
        opcodes.push_back(opcode);
      }
      cursor += length;
    }
    const std::vector<uint32_t> expected =
        control
            ? std::vector<uint32_t>{0x46, 0x58, 0x40, 0x46, 0x58, 0x49}
            : std::vector<uint32_t>{0x46, 0x58, 0x50, 0x46, 0x50, 0x58, 0x49};
    EXPECT_EQ(opcodes, expected);
    ASSERT_NO_FATAL_FAILURE(Execute(0));
    Poll(0);
    uint64_t value = 0;
    IREE_ASSERT_OK(iree_hal_semaphore_query(consumer, &value));
    EXPECT_EQ(value, one);
    EXPECT_EQ(Read(buffer, 4), std::vector<uint8_t>(4, pattern & 255));
  }
}

TEST_F(KernelQueueTest, RetiredProducerSignalSurvivesDependentExecution) {
  auto producer = Semaphore(), consumer = Semaphore();
  auto* producer_handle = producer.get();
  auto* consumer_handle = consumer.get();
  uint64_t one = 1;
  IREE_ASSERT_OK(iree_hal_queue_barrier(
      queues_[0], {}, {1, &producer_handle, &one}, nullptr, 0));
  IREE_ASSERT_OK(iree_hal_queue_barrier(queues_[1], {1, &producer_handle, &one},
                                        {1, &consumer_handle, &one}, nullptr,
                                        0));
  Publish(0);
  Publish(1);
  ASSERT_EQ(native_queues_[1].commands.size(), 1u);
  iree_hal_amd_gpu_semaphore_wait_t wait = {};
  IREE_ASSERT_OK(
      iree_hal_amd_gpu_semaphore_acquire_wait(producer, one, &context_, &wait));
  ASSERT_EQ(wait.state, IREE_HAL_AMD_GPU_SEMAPHORE_WAIT_NATIVE);
  auto* signal = wait.signal;
  // The submitted consumer owns this borrow throughout its native execution.
  // The test must not retain an extra reference that hides premature reuse.
  iree_hal_amd_gpu_signal_release(signal);
  ASSERT_NO_FATAL_FAILURE(Execute(0));
  Poll(0);
  producer.reset();
  iree_status_code_t code = IREE_STATUS_UNKNOWN;
  EXPECT_TRUE(iree_hal_amd_gpu_signal_query_retirement(signal, &code));
  EXPECT_EQ(code, IREE_STATUS_OK);
  auto* storage = signals_[0].storage;
  iree_hal_amd_gpu_signal_cache_trim(&context_, 0, &signals_[0]);
  EXPECT_EQ(signals_[0].storage, storage);

  // Repeated publication wraps both the four-slot CPU action ring and the
  // native command ring while the dependent queue has not read its predicate.
  for (uint32_t iteration = 0; iteration < 32; ++iteration) {
    auto completion = Semaphore();
    auto* handle = completion.get();
    IREE_ASSERT_OK(
        iree_hal_queue_barrier(queues_[0], {}, {1, &handle, &one}, nullptr, 0));
    Publish(0);
    ASSERT_NO_FATAL_FAILURE(Execute(0));
    Poll(0);
    EXPECT_EQ(iree_atomic_load(
                  reinterpret_cast<iree_atomic_int32_t*>(signal->host_value),
                  iree_memory_order_acquire),
              1);
  }
  ASSERT_NO_FATAL_FAILURE(Execute(1));
  Poll(1);
  uint64_t value = 0;
  IREE_ASSERT_OK(iree_hal_semaphore_query(consumer, &value));
  EXPECT_EQ(value, one);
  iree_hal_amd_gpu_signal_cache_trim(&context_, 0, &signals_[0]);
  EXPECT_EQ(signals_[0].storage, nullptr);
}

TEST_F(KernelQueueTest, NativeCompletionWaitsForProducerHostRetirement) {
  auto producer = Semaphore(), consumer = Semaphore();
  auto* producer_handle = producer.get();
  auto* consumer_handle = consumer.get();
  uint64_t one = 1;
  IREE_ASSERT_OK(iree_hal_queue_barrier(
      queues_[0], {}, {1, &producer_handle, &one}, nullptr, 0));
  IREE_ASSERT_OK(iree_hal_queue_barrier(queues_[1], {1, &producer_handle, &one},
                                        {1, &consumer_handle, &one}, nullptr,
                                        0));
  Publish(0);
  Publish(1);
  ASSERT_NO_FATAL_FAILURE(Execute(0));
  ASSERT_NO_FATAL_FAILURE(Execute(1));
  Poll(1);
  uint64_t value = 0;
  IREE_ASSERT_OK(iree_hal_semaphore_query(consumer, &value));
  EXPECT_EQ(value, 0u);
  Poll(0);
  Poll(1);
  IREE_ASSERT_OK(iree_hal_semaphore_query(consumer, &value));
  EXPECT_EQ(value, one);
}

TEST_F(KernelQueueTest, TransferPrefixRetiresBeforeItsTerminalSignal) {
  std::array<iree_hal_transfer_operation_t, 80> operations = {};
  auto buffer = Buffer(operations.size() * sizeof(uint32_t));
  const uint32_t pattern = 0x71717171u;
  for (size_t i = 0; i < operations.size(); ++i) {
    operations[i] = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
        .fill =
            {
                .target_buffer = buffer,
                .target_offset = i * sizeof(uint32_t),
                .length = sizeof(uint32_t),
                .pattern = &pattern,
                .pattern_length = sizeof(uint32_t),
            },
    };
  }
  auto completion = Semaphore();
  auto* handle = completion.get();
  uint64_t one = 1;
  IREE_ASSERT_OK(iree_hal_queue_transfer(queues_[0], {}, {1, &handle, &one},
                                         operations.size(), operations.data(),
                                         nullptr));
  Publish(0);
  ASSERT_EQ(native_queues_[0].commands.size(), 3u);
  auto* queue =
      reinterpret_cast<iree_hal_amd_gpu_pm4_queue_t*>(queues_[0].get());
  auto* submission =
      iree_hal_amd_gpu_pm4_retirement_submission(&queue->issued[0].action);
  ASSERT_NE(submission->signal, nullptr);
  ASSERT_FALSE(
      iree_hal_amd_gpu_pm4_retirement_is_terminal(&queue->issued[0].action));
  ASSERT_NO_FATAL_FAILURE(Execute(0, 1));
  Poll(0);
  EXPECT_EQ(
      iree_atomic_load(&queue->completed.count, iree_memory_order_acquire), 1);
  uint64_t value = 0;
  IREE_ASSERT_OK(iree_hal_semaphore_query(completion, &value));
  EXPECT_EQ(value, 0u);
  ASSERT_NO_FATAL_FAILURE(Execute(0));
  Poll(0);
  IREE_ASSERT_OK(iree_hal_semaphore_query(completion, &value));
  EXPECT_EQ(value, one);
  EXPECT_EQ(Read(buffer, operations.size() * sizeof(uint32_t)),
            std::vector<uint8_t>(operations.size() * sizeof(uint32_t), 0x71));
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
