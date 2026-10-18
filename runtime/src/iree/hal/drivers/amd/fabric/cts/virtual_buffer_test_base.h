// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_FABRIC_CTS_VIRTUAL_BUFFER_TEST_BASE_H_
#define IREE_HAL_DRIVERS_AMD_FABRIC_CTS_VIRTUAL_BUFFER_TEST_BASE_H_

#include <emmintrin.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#include "amdf/gpu.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/device_group.h"
#include "iree/hal/drivers/amd/fabric/fabric.h"
#include "iree/hal/drivers/amd/fabric/memory_backend.h"
#include "iree/hal/drivers/task/device.h"
#include "iree/hal/memory/host_memory_backend.h"
#include "iree/hal/memory/tlsf_pool.h"
#include "iree/hal/testing/mock_device.h"
#include "iree/task/executor.h"
#include "libamdf/cts/gpu/sdma/encoding/commands.h"

namespace iree::hal::amd::fabric::cts {

// The metadata-only GPU dependency borrows its native handles from the fixture.
// Observe its actual final release before reclaiming those handles and facts.
struct NativeDeviceLifetime {
  // Retained by the mock device and indirectly by each real fabric provider.
  iree_hal_resource_t resource;
  // Guards final-release publication to the fixture owner.
  std::mutex mutex;
  // Wakes teardown after the last native user releases its owner.
  std::condition_variable condition;
  // True after the resource's final callback, guarded by mutex.
  bool released = false;

  NativeDeviceLifetime() {
    static const iree_hal_resource_vtable_t vtable = {
        .destroy =
            +[](iree_hal_resource_t* resource) {
              auto* lifetime =
                  reinterpret_cast<NativeDeviceLifetime*>(resource);
              std::lock_guard<std::mutex> lock(lifetime->mutex);
              lifetime->released = true;
              lifetime->condition.notify_all();
            },
    };
    iree_hal_resource_initialize(&vtable, &resource);
  }

  void ReleaseAndAwait() {
    iree_hal_resource_release(&resource);
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [&] { return released; });
  }
};

// The memory adapter and Task queue are real HAL implementations. Only GPU
// family metadata uses a mock device until the GPU execution HAL exists; all
// address mutation and GPU execution use the public native API and hardware.
class FabricVirtualBufferTest
    : public ::testing::TestWithParam<amdf_native_lifetime_t> {
 protected:
  static void SetUpTestSuite() {
    const amdf_api_t* api = nullptr;
    ASSERT_EQ(
        amdf_query_api(AMDF_ABI_VERSION_LATEST, AMDF_ABI_VERSION_LATEST, &api),
        AMDF_STATUS_OK);
    for (uint32_t i = 0; i < fabrics_.size(); ++i) {
      IREE_ASSERT_OK(iree_hal_amd_fabric_create(api, i, iree_allocator_system(),
                                                &fabrics_[i]));
    }
  }

  static void TearDownTestSuite() {
    for (auto*& fabric : fabrics_) {
      iree_hal_amd_fabric_release(fabric);
      fabric = nullptr;
    }
  }

  void SetUp() override {
    ASSERT_EQ(
        amdf_query_api(AMDF_ABI_VERSION_LATEST, AMDF_ABI_VERSION_LATEST, &api_),
        AMDF_STATUS_OK);
    const void* extension = nullptr;
    ASSERT_EQ(
        api_->query_extension(AMDF_EXTENSION_GPU, AMDF_GPU_EXTENSION_VERSION_1,
                              AMDF_GPU_EXTENSION_VERSION_LATEST, &extension),
        AMDF_STATUS_OK);
    gpu_ = static_cast<const amdf_gpu_api_t*>(extension);
    fabric_ = fabrics_[GetParam()];
    iree_hal_amd_fabric_retain(fabric_);
    RecordProperty("native_lifetime", GetParam() == AMDF_NATIVE_LIFETIME_PROCESS
                                          ? "process"
                                          : "instance");
    for (uint32_t i = 0; i < fabric_->endpoint_count && !endpoint_; ++i) {
      if (fabric_->endpoints[i].engine_kind != AMDF_ENGINE_KIND_GPU) {
        continue;
      }
      ASSERT_EQ(api_->endpoint_open(fabric_->instance,
                                    &fabric_->endpoints[i].id, &endpoint_),
                AMDF_STATUS_OK);
      amdf_endpoint_info_t info = {
          .type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO,
          .structure_size = sizeof(info),
      };
      ASSERT_EQ(api_->endpoint_query_info(endpoint_, &info), AMDF_STATUS_OK);
      bool selected = false;
      for (uint32_t f = 0; f < info.queue_family_count; ++f) {
        native_family_.type = AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO;
        native_family_.structure_size = sizeof(native_family_);
        ASSERT_EQ(api_->endpoint_query_queue_family_info(endpoint_, f,
                                                         &native_family_),
                  AMDF_STATUS_OK);
        if (native_family_.command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA &&
            native_family_.format_version == 1 &&
            (native_family_.publication_modes &
             AMDF_QUEUE_PUBLICATION_MODE_USER) &&
            (native_family_.user_queue_capabilities &
             AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER)) {
          selected = true;
          break;
        }
      }
      if (!selected) {
        ASSERT_EQ(api_->endpoint_close(endpoint_), AMDF_STATUS_OK);
        endpoint_ = nullptr;
      }
    }
    if (!endpoint_) {
      GTEST_SKIP() << "no host-published SDMA family";
    }
    amdf_gpu_endpoint_info_t target = {
        .type = AMDF_STRUCTURE_TYPE_GPU_ENDPOINT_INFO,
        .structure_size = sizeof(target),
    };
    ASSERT_EQ(gpu_->endpoint_query_info(endpoint_, &target), AMDF_STATUS_OK);
    char target_name[32];
    std::snprintf(target_name, sizeof(target_name), "gfx%u%x%x",
                  target.gfx_ip.major, target.gfx_ip.minor,
                  target.gfx_ip.stepping);
    RecordProperty("gpu_target", target_name);
    const amdf_gpu_device_create_info_t create = {
        .type = AMDF_STRUCTURE_TYPE_GPU_DEVICE_CREATE_INFO,
        .structure_size = sizeof(create),
    };
    const auto device_status =
        gpu_->device_create(endpoint_, &create, &native_device_);
    if (GetParam() == AMDF_NATIVE_LIFETIME_INSTANCE &&
        device_status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
      GTEST_SKIP()
          << "native driver does not support instance-lifetime devices";
    }
    ASSERT_EQ(device_status, AMDF_STATUS_OK);
    amdf_memory_reservation_capabilities_t capabilities = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CAPABILITIES,
        .structure_size = sizeof(capabilities),
    };
    const auto capability_status =
        api_->device_query_memory_reservation_capabilities(native_device_, 0,
                                                           &capabilities);
    if (capability_status ==
        amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
      GTEST_SKIP() << "native device does not expose reservations";
    }
    ASSERT_EQ(capability_status, AMDF_STATUS_OK);
    granularity_ = std::max(capabilities.mapping_granularity,
                            capabilities.reservation_granularity);

    IREE_ASSERT_OK(context_.Initialize(allocator_));
    iree_task_topology_t topology;
    iree_task_topology_initialize_from_group_count(1, &topology);
    iree_task_executor_options_t executor_options;
    iree_task_executor_options_initialize(&executor_options);
    auto status = iree_task_executor_create(executor_options, &topology,
                                            allocator_, &executor_);
    iree_task_topology_deinitialize(&topology);
    IREE_ASSERT_OK(status);
    IREE_ASSERT_OK(iree_hal_allocator_create_heap(allocator_, &cpu_allocator_));
    iree_hal_task_device_params_t params;
    iree_hal_task_device_params_initialize(&params);
    IREE_ASSERT_OK(iree_hal_task_device_create(
        IREE_SV("cpu"), &params, 1, &executor_, 0, nullptr, cpu_allocator_,
        context_.params(), allocator_, &cpu_));
    iree_hal_mock_device_options_t device_options;
    iree_hal_mock_device_options_initialize(&device_options);
    device_options.identifier = IREE_SV("native-gpu");
    device_options.executable_loading_enabled = true;
    device_options.memory_backend = &backend_.base;
    device_options.lifetime_owner = &native_lifetime_.resource;
    IREE_ASSERT_OK(
        iree_hal_mock_device_create(&device_options, allocator_, &device_));
    family_.queue_family_ordinal = native_family_.ordinal;
    family_.supported_usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
    family_.minimum_access = capabilities.required_access;
    family_.binding_count = 1;
    family_.bindings = &binding_;
    const auto* host = reinterpret_cast<const iree_hal_host_memory_backend_t*>(
        iree_hal_device_memory_backend(cpu_));
    storage_.push_back({
        .scope = fabric_->system_scope,
        .memory_type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL,
        .minimum_access = capabilities.required_access,
    });
    uint32_t local_count = 0;
    const auto count_status = api_->device_enumerate_memory_scopes(
        native_device_, 0, nullptr, &local_count);
    ASSERT_TRUE(count_status == AMDF_STATUS_OK ||
                count_status ==
                    amdf_make_api_status(AMDF_STATUS_CODE_BUFFER_TOO_SMALL));
    std::vector<amdf_memory_scope_t*> local_scopes(local_count);
    ASSERT_EQ(
        api_->device_enumerate_memory_scopes(native_device_, local_count,
                                             local_scopes.data(), &local_count),
        AMDF_STATUS_OK);
    for (auto* local_scope : local_scopes) {
      amdf_memory_scope_info_t info = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
          .structure_size = sizeof(info),
      };
      ASSERT_EQ(api_->memory_scope_query_info(local_scope, &info),
                AMDF_STATUS_OK);
      ASSERT_EQ(info.kind, AMDF_MEMORY_SCOPE_KIND_LOCAL);
      storage_.push_back({
          .scope = local_scope,
          .physical_device_affinity = 1,
          .memory_type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
          .minimum_access = capabilities.required_access,
      });
    }
    iree_hal_amd_fabric_memory_backend_options_t memory_options = {
        .api = api_,
        .instance = fabric_->instance,
        .device = native_device_,
        .storage_count = storage_.size(),
        .storage = storage_.data(),
        .families = &family_,
    };
    IREE_ASSERT_OK(iree_hal_amd_fabric_acquire_allocation_queue(
        fabric_, UINT32_MAX, &allocation_queue_));
    iree_hal_amd_fabric_memory_backend_initialize(
        device_, &memory_options, host->notification, allocation_queue_,
        host->epoch_query, &backend_);
    iree_hal_device_group_builder_t builder;
    iree_hal_device_group_builder_initialize(&builder,
                                             context_.frontier_tracker());
    IREE_ASSERT_OK(iree_hal_device_group_builder_add_device(&builder, cpu_));
    IREE_ASSERT_OK(iree_hal_device_group_builder_add_device(&builder, device_));
    AddDevices(&builder);
    if (HasFatalFailure() || IsSkipped()) {
      iree_hal_device_group_builder_deinitialize(&builder);
      return;
    }
    status =
        iree_hal_device_group_builder_finalize(&builder, allocator_, &group_);
    iree_hal_device_group_builder_deinitialize(&builder);
    IREE_ASSERT_OK(status);
    access_.family = iree_hal_device_queue_family(device_, 0);
    access_.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
    iree_hal_virtual_buffer_capabilities_t reservation_capabilities;
    IREE_ASSERT_OK(iree_hal_virtual_buffer_query_capabilities(
        group_, {1, &access_, {}}, allocator_, &reservation_capabilities));
    ASSERT_EQ(reservation_capabilities.mapping_granularity,
              capabilities.mapping_granularity);
    ASSERT_EQ(reservation_capabilities.reservation_granularity, granularity_);
    ASSERT_EQ(reservation_capabilities.max_byte_length,
              capabilities.maximum_byte_length);
    ASSERT_EQ(reservation_capabilities.max_alignment,
              capabilities.maximum_alignment);
    granularity_ = reservation_capabilities.reservation_granularity;
    access_.interfaces = UINT64_C(1)
                         << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE;
    access_.requirements = IREE_HAL_POOL_ACCESS_REQUIRE_COHERENT_WITH_HOST;
    iree_hal_pool_scope_t scope = {1, &access_, {}};
    scope.host.access = IREE_HAL_MEMORY_ACCESS_ALL;
    scope.host.modes = IREE_HAL_MAPPING_MODE_PERSISTENT;
    iree_hal_slab_pool_options_t pool_options;
    iree_hal_slab_pool_options_initialize(&pool_options);
    IREE_ASSERT_OK(iree_hal_slab_pool_create(group_, scope, &pool_options,
                                             allocator_, &pool_));
    const auto source_requirements =
        iree_hal_pool_query_mapping_requirements(pool_);
    ASSERT_EQ(source_requirements.minimum_granularity,
              capabilities.mapping_granularity);
    ASSERT_GE(source_requirements.recommended_granularity,
              source_requirements.minimum_granularity);
    ASSERT_EQ(source_requirements.flags, 0u);
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        pool_,
        iree_hal_make_pool_reservation_request(6 * granularity_, granularity_),
        iree_infinite_timeout(), &backing_));
    IREE_ASSERT_OK(iree_hal_buffer_map_range(
        backing_, IREE_HAL_MAPPING_MODE_PERSISTENT, IREE_HAL_MEMORY_ACCESS_ALL,
        0, 0, IREE_HAL_WHOLE_BUFFER, &host_mapping_));
    mapped_ = true;
    access_.interfaces = UINT64_C(1)
                         << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS;
    access_.requirements = 0;
    scope.host = {};
    const iree_hal_virtual_buffer_options_t reservation_options = {
        .byte_length = granularity_,
        .mapping_granularity = granularity_,
    };
    const auto physical_charge = iree_hal_buffer_memory_view(backing_)
                                     .backing->native_allocation.byte_length;
    for (size_t i = 0; i < reservations_.size(); ++i) {
      IREE_ASSERT_OK(iree_hal_virtual_buffer_create(
          group_, scope, &reservation_options, allocator_, &reservations_[i]));
      iree_hal_tlsf_pool_options_t child_options = {};
      child_options.tlsf_options.alignment = 256;
      IREE_ASSERT_OK(iree_hal_tlsf_pool_create_from_buffer(
          reservations_[i], 256, 256, &child_options, allocator_,
          &child_pools_[i]));
      IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
          child_pools_[i], iree_hal_make_pool_reservation_request(256, 256),
          iree_infinite_timeout(), &children_[i]));
      const auto memory = iree_hal_buffer_memory_view(children_[i]);
      ASSERT_NE(memory.backing, nullptr);
      EXPECT_EQ(memory.backing->native_allocation.byte_length, 0u);
      EXPECT_EQ(memory.offset, 256u);
    }
    iree_hal_pool_stats_t stats;
    iree_hal_pool_query_stats(pool_, &stats);
    EXPECT_EQ(stats.bytes_committed, physical_charge);
    IREE_ASSERT_OK(iree_hal_device_group_resolve_memory_scope(
        group_, {IREE_HAL_MEMORY_SITE_QUEUE, access_.family}, &scope_));
    IREE_ASSERT_OK(iree_hal_pool_resolve_binding(
        pool_, scope_, IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
        &source_slot_));
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        cpu_, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0, 0, &done_));
    const amdf_gpu_user_queue_create_info_t queue_create = {
        .type = AMDF_STRUCTURE_TYPE_GPU_USER_QUEUE_CREATE_INFO,
        .structure_size = sizeof(queue_create),
        .queue_family_ordinal = native_family_.ordinal,
        .priority = AMDF_QUEUE_PRIORITY_NORMAL,
        .producer_mode = AMDF_QUEUE_PRODUCER_MODE_SINGLE,
        .required_capabilities = AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER,
        .ring_byte_length = 65536,
    };
    ASSERT_EQ(gpu_->user_queue_create(native_device_, &queue_create, &queue_),
              AMDF_STATUS_OK);
    ASSERT_EQ(api_->user_queue_map(queue_, nullptr, &queue_mapping_),
              AMDF_STATUS_OK);
    queue_host_.type = AMDF_STRUCTURE_TYPE_USER_QUEUE_MAPPING_INFO;
    queue_host_.structure_size = sizeof(queue_host_);
    ASSERT_EQ(api_->user_queue_mapping_query_info(queue_mapping_, &queue_host_),
              AMDF_STATUS_OK);
    ASSERT_EQ(queue_host_.index_bits, 64u);
    ASSERT_EQ(queue_host_.doorbell_bits, 64u);
  }

  // Adds engine participants before the shared group becomes immutable.
  virtual void AddDevices(iree_hal_device_group_builder_t* builder) {
    (void)builder;
  }

  void TearDown() override {
    for (size_t i = 0; i < children_.size(); ++i) {
      iree_hal_buffer_release(children_[i]);
      iree_hal_pool_release(child_pools_[i]);
    }
    // Checked address removal happens while the native execution queue and
    // original backing still exist. A failed removal never authorizes reuse.
    for (auto*& reservation : reservations_) {
      auto* owner = reservation;
      reservation = nullptr;
      iree_hal_buffer_release(owner);
    }
    if (queue_mapping_) {
      ASSERT_EQ(api_->user_queue_mapping_destroy(queue_mapping_),
                AMDF_STATUS_OK);
    }
    if (queue_) {
      ASSERT_EQ(api_->user_queue_destroy(queue_), AMDF_STATUS_OK);
    }
    if (mapped_) {
      IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&host_mapping_));
    }
    iree_hal_buffer_release(backing_);
    iree_hal_pool_release(pool_);
    iree_hal_semaphore_release(done_);
    iree_hal_device_group_release(group_);
    iree_hal_device_release(device_);
    native_lifetime_.ReleaseAndAwait();
    iree_hal_amd_fabric_allocation_queue_release(allocation_queue_);
    iree_hal_device_release(cpu_);
    iree_hal_allocator_release(cpu_allocator_);
    iree_task_executor_release(executor_);
    context_.Deinitialize();
    if (native_device_) {
      ASSERT_EQ(api_->device_destroy(native_device_), AMDF_STATUS_OK);
    }
    if (endpoint_) {
      ASSERT_EQ(api_->endpoint_close(endpoint_), AMDF_STATUS_OK);
    }
    iree_hal_amd_fabric_release(fabric_);
  }

  template <size_t N>
  iree_status_t Map(const std::array<iree_hal_memory_mapping_t, N>& mappings) {
    uint64_t value = ++sequence_;
    const iree_hal_semaphore_list_t signals = {1, &done_, &value};
    IREE_RETURN_IF_ERROR(iree_hal_queue_map(iree_hal_device_queue(cpu_, 0, 0),
                                            {}, signals, mappings.size(),
                                            mappings.data(), 0));
    return iree_hal_semaphore_wait(done_, value, iree_infinite_timeout(), 0);
  }

  void CpuFillAndReadback(iree_hal_buffer_t* buffer) {
    auto* queue = iree_hal_device_queue(cpu_, 0, 0);
    iree_hal_memory_scope_t cpu_scope;
    IREE_ASSERT_OK(iree_hal_device_group_resolve_memory_scope(
        group_,
        {IREE_HAL_MEMORY_SITE_QUEUE, iree_hal_device_queue_family(cpu_, 0)},
        &cpu_scope));
    const auto table = iree_hal_buffer_transition_table(buffer);
    iree_hal_memory_transition_pair_t pair;
    IREE_ASSERT_OK(iree_hal_memory_transition_prepare_pair(
        table, cpu_scope, cpu_scope, IREE_HAL_MEMORY_TRANSITION_RELEASE,
        &pair));
    const auto transition = iree_hal_memory_transition_query(table, pair);
    const iree_hal_buffer_barrier_t release_buffer = {
        .buffer_ref = iree_hal_make_buffer_ref(buffer, 0, granularity_),
        .recipe = iree_hal_memory_transition_recipe(
            table, pair, IREE_HAL_MEMORY_TRANSITION_RELEASE),
    };
    const iree_hal_buffer_barrier_t acquire_buffer = {
        .buffer_ref = release_buffer.buffer_ref,
        .recipe = iree_hal_memory_transition_recipe(
            table, pair, IREE_HAL_MEMORY_TRANSITION_ACQUIRE),
    };
    const iree_hal_barrier_t release = {
        .effects = transition.release,
        .buffer_barrier_count = release_buffer.recipe ? 1u : 0u,
        .buffer_barriers = &release_buffer,
    };
    const iree_hal_barrier_t acquire = {
        .effects = transition.acquire,
        .buffer_barrier_count = acquire_buffer.recipe ? 1u : 0u,
        .buffer_barriers = &acquire_buffer,
    };
    const iree_hal_barrier_list_t no_barriers = {};
    const iree_hal_barrier_list_t release_barriers = {1, &release};
    const iree_hal_barrier_list_t acquire_barriers = {1, &acquire};
    const iree_hal_queue_barriers_t fill_barriers = {&no_barriers,
                                                     &release_barriers};
    const iree_hal_queue_barriers_t download_barriers = {&acquire_barriers,
                                                         &no_barriers};
    const uint32_t pattern = UINT32_C(0xA5913CE7);
    uint64_t filled = ++sequence_;
    const iree_hal_semaphore_list_t fill_done = {1, &done_, &filled};
    IREE_ASSERT_OK(iree_hal_queue_fill(queue, {}, fill_done, buffer, 0,
                                       granularity_, &pattern, sizeof(pattern),
                                       &fill_barriers, 0));
    std::array<uint32_t, 64> result = {};
    uint64_t downloaded = ++sequence_;
    const iree_hal_semaphore_list_t download_done = {1, &done_, &downloaded};
    IREE_ASSERT_OK(iree_hal_queue_download(queue, fill_done, download_done,
                                           buffer, 0, result.data(),
                                           sizeof(result), &download_barriers));
    IREE_ASSERT_OK(
        iree_hal_semaphore_wait(done_, downloaded, iree_infinite_timeout(), 0));
    for (uint32_t word : result) {
      EXPECT_EQ(word, pattern);
    }
  }

  void CopyAndVerify(uint32_t first_pattern, uint32_t second_pattern) {
    const auto source_address =
        iree_hal_buffer_native_binding(backing_, source_slot_).device_address;
    auto* words = reinterpret_cast<uint32_t*>(host_mapping_.contents.data);
    const size_t stride = granularity_ / sizeof(uint32_t);
    SdmaCommandWriter commands(
        reinterpret_cast<uint32_t*>(queue_host_.ring_address) + ring_words_,
        native_family_.format_features);
    const bool cache_control =
        native_family_.format_features & AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR;
    if (cache_control) {
      commands.AcquireFromSystem();
    }
    for (uint32_t i = 0; i < 2; ++i) {
      auto* buffer = children_[i];
      const iree_hal_buffer_native_binding_slot_t slot = {
          .index = buffer->memory.contract->scopes[scope_.id]
                       .bindings[IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS],
          .type = IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
      };
      commands.CopyLinear(
          iree_hal_buffer_native_binding(buffer, slot).device_address,
          source_address + (2 + i) * granularity_, 256);
    }
    if (cache_control) {
      commands.ReleaseToSystem();
    }
    ASSERT_NO_FATAL_FAILURE(SubmitAndWait(commands));
    for (uint32_t i = 0; i < 64; ++i) {
      ASSERT_EQ(words[2 * stride + i], first_pattern + 64 + i);
      ASSERT_EQ(words[3 * stride + i], second_pattern + 64 + i);
    }
  }

  void SubmitAndWait(SdmaCommandWriter& commands) {
    const auto source_address =
        iree_hal_buffer_native_binding(backing_, source_slot_).device_address;
    auto* completion = reinterpret_cast<uint32_t*>(host_mapping_.contents.data +
                                                   4 * granularity_);
    std::atomic_ref<uint32_t>(*completion).store(0, std::memory_order_relaxed);
    commands.Fence32(source_address + 4 * granularity_, 1);
    ring_words_ += commands.word_count();
    ASSERT_LT(ring_words_ * sizeof(uint32_t), queue_host_.ring_byte_length);
    std::atomic_ref<uint64_t>(
        *reinterpret_cast<uint64_t*>(queue_host_.write_index_address))
        .store(ring_words_ * sizeof(uint32_t), std::memory_order_release);
    // Publish command and index stores before the write-only MMIO notification.
    _mm_sfence();
    std::atomic_ref<uint64_t>(
        *reinterpret_cast<uint64_t*>(queue_host_.doorbell_address))
        .store(ring_words_ * sizeof(uint32_t), std::memory_order_release);
    while (std::atomic_ref<uint32_t>(*completion)
               .load(std::memory_order_acquire) != 1) {
      std::this_thread::yield();
    }
    ASSERT_EQ(
        api_->user_queue_wait_consumed(queue_, ring_words_ * sizeof(uint32_t),
                                       AMDF_TIMEOUT_INFINITE, 0),
        AMDF_STATUS_OK);
  }

  // Host metadata allocator used throughout this application flow.
  iree_allocator_t allocator_ = iree_allocator_system();
  // Process-lifetime KFD VM bindings can survive instance destruction. Keep
  // their explicit native owners through the suite, as an application does.
  inline static std::array<iree_hal_amd_fabric_t*, 2> fabrics_ = {};
  // Native core table and GPU extension, borrowed from the linked provider.
  const amdf_api_t* api_ = nullptr;
  // GPU execution API used only by the independent native consumer.
  const amdf_gpu_api_t* gpu_ = nullptr;
  // Shared instance and passive discovery owner used by the adapter.
  iree_hal_amd_fabric_t* fabric_ = nullptr;
  // Open native endpoint, destroyed after its device.
  amdf_endpoint_t* endpoint_ = nullptr;
  // Initialized native consumer borrowed by the HAL memory backend.
  amdf_device_t* native_device_ = nullptr;
  // Selected real SDMA family and its exact packet features.
  amdf_queue_family_info_t native_family_ = {};
  // Existing Task creation/progress services.
  iree::hal::cts::DeviceCreateContext context_;
  // Task worker executing the captured host mapping batch.
  iree_task_executor_t* executor_ = nullptr;
  // CPU device's ordinary allocator.
  iree_hal_allocator_t* cpu_allocator_ = nullptr;
  // CPU HAL queue owner.
  iree_hal_device_t* cpu_ = nullptr;
  // GPU family metadata owner with no mock execution implementation.
  iree_hal_device_t* device_ = nullptr;
  // Native dependency lifetime retained through asynchronous provider cleanup.
  NativeDeviceLifetime native_lifetime_;
  // Real native allocation queue shared with other engines in this fabric.
  iree_hal_amd_fabric_allocation_queue_t* allocation_queue_ = nullptr;
  // Shared sealed namespace containing the CPU and native GPU consumer.
  iree_hal_device_group_t* group_ = nullptr;
  // Native execution address interface captured by the family.
  iree_hal_amd_fabric_memory_binding_t binding_ = {
      IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS, AMDF_MEMORY_ADDRESS_GPU};
  // Immutable family facts borrowed by the adapter.
  iree_hal_amd_fabric_memory_family_t family_ = {};
  // Borrowed system and native local scopes consumed by pool construction.
  std::vector<iree_hal_amd_fabric_memory_storage_t> storage_;
  // Real fabric memory implementation under test.
  iree_hal_amd_fabric_memory_backend_t backend_ = {};
  // Explicit requested family access reused for pool/reservation creation.
  iree_hal_pool_family_access_t access_ = {};
  // Prepared exact queue scope in the sealed group.
  iree_hal_memory_scope_t scope_ = {};
  // Prepared source GPU address slot.
  iree_hal_buffer_native_binding_slot_t source_slot_ = {};
  // Explicit pool owning the one physical native slab.
  iree_hal_pool_t* pool_ = nullptr;
  // Ordinary source buffer retained until mappings and execution are removed.
  iree_hal_buffer_t* backing_ = nullptr;
  // Persistent CPU alias used for initialization and exact result observation.
  iree_hal_buffer_mapping_t host_mapping_ = {};
  // Whether the persistent alias was acquired.
  bool mapped_ = false;
  // Independent address owners updated by each single HAL batch.
  std::array<iree_hal_buffer_t*, 2> reservations_ = {};
  // Finite suballocators over unmapped virtual ranges, retained across remaps.
  std::array<iree_hal_pool_t*, 2> child_pools_ = {};
  // Stable interior buffer addresses actually read by the native GPU.
  std::array<iree_hal_buffer_t*, 2> children_ = {};
  // Exact native replacement unit used for this witness.
  uint64_t granularity_ = 0;
  // Completed HAL mapping sequence.
  uint64_t sequence_ = 0;
  // Semaphore publishing mapping completion to the native consumer.
  iree_hal_semaphore_t* done_ = nullptr;
  // Live native consumer queue retained across replacement and removal.
  amdf_user_queue_t* queue_ = nullptr;
  // Explicit host producer mapping for the native queue.
  amdf_user_queue_mapping_t* queue_mapping_ = nullptr;
  // Borrowed mapped ring and producer fields.
  amdf_user_queue_mapping_info_t queue_host_ = {};
  // Finite command cursor; the test never wraps or overwrites live commands.
  size_t ring_words_ = 0;
};

}  // namespace iree::hal::amd::fabric::cts

#endif  // IREE_HAL_DRIVERS_AMD_FABRIC_CTS_VIRTUAL_BUFFER_TEST_BASE_H_
