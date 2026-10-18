// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <linux/mempolicy.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>
#include <climits>

#include "iree/base/threading/numa.h"
#include "iree/hal/drivers/amd/fabric/cts/virtual_buffer_test_base.h"

namespace iree::hal::amd::fabric::cts {
namespace {

using iree::hal::cts::Ref;

class FabricHostNumaMemoryTest : public FabricVirtualBufferTest {
 protected:
  void AddDevices(iree_hal_device_group_builder_t* builder) override {
    const long page_size = sysconf(_SC_PAGESIZE);
    ASSERT_GT(page_size, 0);
    constexpr size_t kWordBits = sizeof(unsigned long) * CHAR_BIT;
    std::vector<unsigned long> allowed(page_size / sizeof(unsigned long));
    ASSERT_EQ(
        syscall(SYS_get_mempolicy, nullptr, allowed.data(),
                allowed.size() * kWordBits + 1, nullptr, MPOL_F_MEMS_ALLOWED),
        0)
        << std::strerror(errno);
    const auto online = iree_numa_online_nodes();
    for (uint32_t node = 0; node < online.bit_count; ++node) {
      if (!iree_bitmap_test(online, node) ||
          !(allowed[node / kWordBits] & (1UL << (node % kWordBits)))) {
        continue;
      }
      iree_thread_affinity_t affinity;
      iree_thread_affinity_set_group_any(node, &affinity);
      iree_task_topology_t topology;
      IREE_ASSERT_OK(iree_task_topology_initialize_from_thread_affinities(
          1, &affinity, &topology));
      iree_task_executor_options_t executor_options;
      iree_task_executor_options_initialize(&executor_options);
      iree_task_executor_t* executor = nullptr;
      auto status = iree_task_executor_create(executor_options, &topology,
                                              allocator_, &executor);
      iree_task_topology_deinitialize(&topology);
      IREE_ASSERT_OK(status);
      iree_hal_task_device_params_t params;
      iree_hal_task_device_params_initialize(&params);
      Ref<iree_hal_device_t> cpu;
      status = iree_hal_task_device_create(
          IREE_SV("placed-cpu"), &params, 1, &executor, 0, nullptr,
          cpu_allocator_, context_.params(), allocator_, cpu.out());
      iree_task_executor_release(executor);
      IREE_ASSERT_OK(status);
      IREE_ASSERT_OK(iree_hal_device_group_builder_add_device(builder, cpu));
      numa_cpus_.push_back(std::move(cpu));
    }
    ASSERT_FALSE(numa_cpus_.empty());
  }

  void TearDown() override {
    // The sealed group retains these devices through the base fixture's work
    // and releases them before its shared progress services are destroyed.
    numa_cpus_.clear();
    FabricVirtualBufferTest::TearDown();
  }

  void RunPlacement(bool with_gpu);
  void RoundTrip(iree_hal_pool_t* pool, iree_hal_buffer_t* buffer,
                 uint32_t node_id);

  // Real affined CPU participants supplying immutable host-node identities.
  std::vector<Ref<iree_hal_device_t>> numa_cpus_;
};

void FabricHostNumaMemoryTest::RunPlacement(bool with_gpu) {
  const long page_size = sysconf(_SC_PAGESIZE);
  ASSERT_GT(page_size, 0);
  const iree_hal_pool_family_access_t families[] = {
      {
          .family = iree_hal_device_queue_family(cpu_, 0),
          .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
      },
      {
          .family = access_.family,
          .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
      },
  };
  // Placement nodes belong to the additional affined CPUs; neither the GPU
  // nor this executing CPU has to be local to the selected backing node.
  const iree_hal_pool_scope_t scope = {
      with_gpu ? 2u : 1u,
      families,
      {IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_MAPPING_MODE_PERSISTENT}};
  const amdf_memory_device_access_t native_access = {
      .device = native_device_,
      .requirements =
          {
              .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE |
                        family_.minimum_access,
              .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
              .address_kinds = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU,
          },
  };
  amdf_memory_scope_info_t scope_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
      .structure_size = sizeof(scope_info),
  };
  ASSERT_EQ(api_->memory_scope_query_info(fabric_->system_scope, &scope_info),
            AMDF_STATUS_OK);
  bool placement_supported = false;
  for (uint32_t i = 0; i < scope_info.memory_profile_count; ++i) {
    amdf_memory_profile_t profile = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
        .structure_size = sizeof(profile),
    };
    amdf_memory_access_capabilities_t capabilities = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
        .structure_size = sizeof(capabilities),
    };
    const auto status = api_->memory_scope_query_device_profile(
        fabric_->system_scope, i, with_gpu ? 1u : 0u,
        with_gpu ? &native_access : nullptr, &profile,
        with_gpu ? &capabilities : nullptr);
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
      continue;
    }
    ASSERT_EQ(status, AMDF_STATUS_OK);
    placement_supported |=
        amdf_status_is_ok(profile.availability) &&
        (profile.roles & AMDF_MEMORY_PROFILE_ROLE_CREATE) &&
        (profile.allocation.placement_kinds &
         (UINT32_C(1) << AMDF_MEMORY_PLACEMENT_KIND_HOST_NUMA));
  }
  RecordProperty("host_numa_profile_qualified",
                 placement_supported ? "true" : "false");
  if (!with_gpu) {
    ASSERT_TRUE(placement_supported);
  }
  uint32_t tested_nodes = 0;
  const auto* topology = iree_hal_device_group_topology(group_);
  for (uint32_t i = 0; i < topology->node_count; ++i) {
    const auto* node = iree_hal_topology_node_at(topology, i);
    if (node->kind != IREE_HAL_TOPOLOGY_NODE_KIND_HOST_NUMA) {
      continue;
    }
    SCOPED_TRACE(node->local_ordinal);
    iree_hal_slab_pool_options_t options;
    iree_hal_slab_pool_options_initialize(&options);
    options.placement = {IREE_HAL_POOL_PLACEMENT_REQUIRED, node->ordinal};
    Ref<iree_hal_pool_t> pool;
    if (!placement_supported) {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                            iree_hal_slab_pool_create(group_, scope, &options,
                                                      allocator_, pool.out()));
      EXPECT_FALSE(pool);
      continue;
    }
    IREE_ASSERT_OK(iree_hal_slab_pool_create(group_, scope, &options,
                                             allocator_, pool.out()));
    iree_hal_pool_capabilities_t capabilities;
    iree_hal_pool_query_capabilities(pool, &capabilities);
    EXPECT_EQ(capabilities.placement.mode, IREE_HAL_POOL_PLACEMENT_REQUIRED);
    EXPECT_EQ(capabilities.placement.node, node->ordinal);
    Ref<iree_hal_buffer_t> buffer;
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        pool,
        iree_hal_make_pool_reservation_request(2 * granularity_, granularity_),
        iree_infinite_timeout(), buffer.out()));
    ASSERT_NO_FATAL_FAILURE(CpuFillAndReadback(buffer));
    if (with_gpu) {
      ASSERT_NO_FATAL_FAILURE(RoundTrip(pool, buffer, node->local_ordinal));
    } else {
      iree_hal_buffer_native_binding_slot_t slot;
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_PERMISSION_DENIED,
          iree_hal_pool_resolve_binding(
              pool, scope_, IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS, &slot));
    }
    iree_byte_span_t host_span;
    IREE_ASSERT_OK(iree_hal_buffer_native_host_span(buffer, 0, 2 * granularity_,
                                                    &host_span));
    for (size_t offset = 0; offset < host_span.data_length;
         offset += page_size) {
      int resident_node = -1;
      ASSERT_EQ(syscall(SYS_get_mempolicy, &resident_node, nullptr, 0,
                        host_span.data + offset, MPOL_F_NODE | MPOL_F_ADDR),
                0)
          << std::strerror(errno);
      EXPECT_EQ(resident_node, static_cast<int>(node->local_ordinal));
    }
    buffer.reset();
    iree_hal_pool_trim(pool, IREE_HAL_POOL_TRIM_FLAG_ALL, 0);
    ++tested_nodes;
  }
  EXPECT_EQ(tested_nodes, placement_supported ? numa_cpus_.size() : 0u);
  RecordProperty("numa_nodes", tested_nodes);
}

void FabricHostNumaMemoryTest::RoundTrip(iree_hal_pool_t* pool,
                                         iree_hal_buffer_t* buffer,
                                         uint32_t node_id) {
  iree_hal_memory_scope_t cpu_scope;
  IREE_ASSERT_OK(iree_hal_device_group_resolve_memory_scope(
      group_,
      {IREE_HAL_MEMORY_SITE_QUEUE, iree_hal_device_queue_family(cpu_, 0)},
      &cpu_scope));
  const auto table = iree_hal_pool_transition_table(pool);
  iree_hal_memory_transition_pair_t publication_pair;
  iree_hal_memory_transition_pair_t observation_pair;
  IREE_ASSERT_OK(iree_hal_memory_transition_prepare_pair(
      table, cpu_scope, scope_, IREE_HAL_MEMORY_TRANSITION_RELEASE,
      &publication_pair));
  IREE_ASSERT_OK(iree_hal_memory_transition_prepare_pair(
      table, scope_, cpu_scope, IREE_HAL_MEMORY_TRANSITION_ACQUIRE,
      &observation_pair));
  const auto publication =
      iree_hal_memory_transition_query(table, publication_pair);
  const auto observation =
      iree_hal_memory_transition_query(table, observation_pair);
  ASSERT_EQ(publication.acquire.bits &
                ~IREE_HAL_MEMORY_EFFECT_GLOBAL_ACQUIRE_FROM_SYSTEM,
            0u);
  ASSERT_EQ(observation.release.bits &
                ~IREE_HAL_MEMORY_EFFECT_GLOBAL_RELEASE_TO_SYSTEM,
            0u);
  if (publication.acquire.bits || observation.release.bits) {
    ASSERT_NE(
        native_family_.format_features & AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR,
        0u);
  }
  const iree_hal_buffer_barrier_t release_buffer = {
      .buffer_ref = iree_hal_make_buffer_ref(buffer, 0, 2 * granularity_),
      .recipe = iree_hal_memory_transition_recipe(
          table, publication_pair, IREE_HAL_MEMORY_TRANSITION_RELEASE),
  };
  const iree_hal_buffer_barrier_t acquire_buffer = {
      .buffer_ref = release_buffer.buffer_ref,
      .recipe = iree_hal_memory_transition_recipe(
          table, observation_pair, IREE_HAL_MEMORY_TRANSITION_ACQUIRE),
  };
  const iree_hal_barrier_t release = {
      .effects = publication.release,
      .buffer_barrier_count = release_buffer.recipe ? 1u : 0u,
      .buffer_barriers = &release_buffer,
  };
  const iree_hal_barrier_t acquire = {
      .effects = observation.acquire,
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
  iree_hal_buffer_native_binding_slot_t slot;
  IREE_ASSERT_OK(iree_hal_pool_resolve_binding(
      pool, scope_, IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS, &slot));
  const auto address =
      iree_hal_buffer_native_binding(buffer, slot).device_address;
  auto* queue = iree_hal_device_queue(cpu_, 0, 0);
  std::vector<uint32_t> result(granularity_ / sizeof(uint32_t));
  for (uint32_t epoch = 1; epoch <= 2; ++epoch) {
    SCOPED_TRACE(epoch);
    const uint32_t pattern = UINT32_C(0xA5910000) | (node_id << 8) | epoch;
    uint64_t filled = ++sequence_;
    const iree_hal_semaphore_list_t fill_done = {1, &done_, &filled};
    IREE_ASSERT_OK(iree_hal_queue_fill(queue, {}, fill_done, buffer, 0,
                                       granularity_, &pattern, sizeof(pattern),
                                       &fill_barriers, 0));
    IREE_ASSERT_OK(
        iree_hal_semaphore_wait(done_, filled, iree_infinite_timeout(), 0));
    SdmaCommandWriter commands(
        reinterpret_cast<uint32_t*>(queue_host_.ring_address) + ring_words_,
        native_family_.format_features);
    if (publication.acquire.bits) {
      commands.AcquireFromSystem();
    }
    commands.CopyLinear(address, address + granularity_, granularity_);
    if (observation.release.bits) {
      commands.ReleaseToSystem();
    }
    ASSERT_NO_FATAL_FAILURE(SubmitAndWait(commands));
    uint64_t downloaded = ++sequence_;
    const iree_hal_semaphore_list_t download_done = {1, &done_, &downloaded};
    IREE_ASSERT_OK(iree_hal_queue_download(queue, {}, download_done, buffer,
                                           granularity_, result.data(),
                                           granularity_, &download_barriers));
    IREE_ASSERT_OK(
        iree_hal_semaphore_wait(done_, downloaded, iree_infinite_timeout(), 0));
    for (uint32_t word : result) {
      EXPECT_EQ(word, pattern);
    }
  }
}

TEST_P(FabricHostNumaMemoryTest, CpuQueuesUseEachRequiredHostNode) {
  ASSERT_NO_FATAL_FAILURE(RunPlacement(/*with_gpu=*/false));
}

TEST_P(FabricHostNumaMemoryTest, GpuSharingMatchesNativePlacementSupport) {
  ASSERT_NO_FATAL_FAILURE(RunPlacement(/*with_gpu=*/true));
}

INSTANTIATE_TEST_SUITE_P(
    NativeLifetime, FabricHostNumaMemoryTest,
    ::testing::Values(AMDF_NATIVE_LIFETIME_PROCESS,
                      AMDF_NATIVE_LIFETIME_INSTANCE),
    [](const ::testing::TestParamInfo<amdf_native_lifetime_t>& info) {
      return info.param == AMDF_NATIVE_LIFETIME_PROCESS ? "Process"
                                                        : "Instance";
    });

}  // namespace
}  // namespace iree::hal::amd::fabric::cts
