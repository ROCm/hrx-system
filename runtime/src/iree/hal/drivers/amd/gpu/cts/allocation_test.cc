// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <semaphore>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "iree/async/frontier_tracker.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/amd/fabric/fabric.h"
#include "iree/hal/drivers/amd/fabric/memory_backend.h"
#include "iree/hal/drivers/amd/gpu/cts/test_device.h"
#include "iree/hal/drivers/amd/gpu/device.h"
#include "iree/hal/drivers/amd/gpu/image/hip_images.h"
#include "iree/hal/drivers/amd/gpu/queue/semaphore.h"
#include "iree/hal/drivers/amd/gpu/queue/signal.h"
#include "iree/hal/drivers/amd/status.h"
#include "iree/hal/memory/fixed_block_pool.h"
#include "iree/hal/slab_pool.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

IREE_ASYNC_FIXED_FRONTIER_TYPE(CompletedFrontier, 2);

using iree::hal::amd::gpu::cts::CompletionPause;
using iree::hal::amd::gpu::cts::GpuNativeTest;

class GpuNativeAllocationTest : public GpuNativeTest {
 protected:
  void SetUp() override {
    GpuNativeTest::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    native_domain_ = context_;
    Create();
    if (HasFatalFailure()) {
      return;
    }
    const iree_hal_pool_family_access_t access = {
        .family = iree_hal_queue_family(queues_[0]),
        .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
        .interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
    };
    const iree_hal_pool_scope_t scope = {
        .family_count = 1,
        .families = &access,
    };
    iree_hal_slab_pool_options_t options;
    iree_hal_slab_pool_options_initialize(&options);
    IREE_ASSERT_OK(iree_hal_slab_pool_create(group_, scope, &options,
                                             iree_allocator_system(), &pool_));
    iree_hal_memory_scope_t memory_scope;
    IREE_ASSERT_OK(iree_hal_device_group_resolve_memory_scope(
        group_, {.kind = IREE_HAL_MEMORY_SITE_PROGRAM, .family = access.family},
        &memory_scope));
    IREE_ASSERT_OK(iree_hal_pool_resolve_binding(
        pool_, memory_scope, IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS, &slot_));
    for (auto& semaphore : semaphores_) {
      CreateSemaphore(&semaphore);
    }
  }

  void TearDown() override {
    for (auto* semaphore : semaphores_) {
      iree_hal_semaphore_release(semaphore);
    }
    iree_hal_pool_release(pool_);
    GpuNativeTest::TearDown();
  }

  void Wait(iree_hal_semaphore_t* semaphore, uint64_t value = 1) {
    IREE_ASSERT_OK(iree_hal_semaphore_wait(
        semaphore, value, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  }

  void CreateFinitePool(iree_hal_pool_t** out_pool, uint32_t block_count = 1) {
    iree_hal_fixed_block_pool_options_t options;
    iree_hal_fixed_block_pool_options_initialize(&options);
    options.block_size = 4096;
    options.blocks_per_slab = block_count;
    iree_hal_pool_reservation_request_t request;
    IREE_ASSERT_OK(iree_hal_fixed_block_pool_query_backing_request(
        pool_, &options, &request));
    iree_hal_buffer_t* backing = nullptr;
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        pool_, request, iree_infinite_timeout(), &backing));
    options.blocks_per_slab = 0;
    IREE_ASSERT_OK(iree_hal_fixed_block_pool_create_from_buffer(
        backing, 0, request.allocation_size, &options, iree_allocator_system(),
        out_pool));
    iree_hal_buffer_release(backing);
  }

  void VerifyHostPoolCopy(iree_hal_pool_t* pool) {
    using iree::hal::cts::Ref;
    const auto properties = iree_hal_pool_query_host_access(pool).properties;
    std::array<Ref<iree_hal_buffer_t>, 2> buffers;
    std::array<iree_hal_buffer_mapping_t, 2> mappings = {};
    for (size_t i = 0; i < buffers.size(); ++i) {
      IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
          pool, iree_hal_make_pool_reservation_request(4096, 0),
          iree_infinite_timeout(), buffers[i].out()));
      EXPECT_EQ(
          iree_hal_buffer_memory_view(buffers[i]).backing->host_properties,
          properties);
      IREE_ASSERT_OK(iree_hal_buffer_map_range(
          buffers[i], IREE_HAL_MAPPING_MODE_PERSISTENT,
          IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, 4096,
          &mappings[i]));
    }
    std::array<uint32_t, 1024> expected;
    for (size_t i = 0; i < expected.size(); ++i) {
      expected[i] = 0xFED00000u + i;
    }
    std::memcpy(mappings[0].contents.data, expected.data(), sizeof(expected));
    IREE_ASSERT_OK(iree_hal_buffer_mapping_flush_range(&mappings[0], 0, 4096));
    uint64_t one = 1;
    IREE_ASSERT_OK(iree_hal_queue_copy(
        queues_[0], {}, {1, &semaphores_[0], &one}, buffers[0], 0, buffers[1],
        0, 4096, nullptr, IREE_HAL_COPY_FLAG_NONE));
    Wait(semaphores_[0]);
    IREE_ASSERT_OK(
        iree_hal_buffer_mapping_invalidate_range(&mappings[1], 0, 4096));
    EXPECT_EQ(std::memcmp(mappings[1].contents.data, expected.data(),
                          sizeof(expected)),
              0);
    for (auto& mapping : mappings) {
      IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
    }
  }

  // An aperture permits mapping but does not establish host/device visibility.
  // Query native guarantees independently of HAL pool admission, without
  // allocating backing or assuming a particular architecture or profile
  // ordinal.
  void QueryLocalHostSupport(bool* out_supported) {
    *out_supported = false;
    const auto* backend =
        reinterpret_cast<const iree_hal_amd_fabric_memory_backend_t*>(
            iree_hal_device_memory_backend(device_));
    const auto& family = backend->native.families[iree_hal_queue_family_ordinal(
        iree_hal_queue_family(queues_[0]))];
    const amdf_memory_device_access_t access = {
        .device = native_domain_->device,
        .requirements =
            {
                .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE |
                          family.minimum_access,
                .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
                .address_kinds = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU,
            },
    };
    const amdf_memory_map_flags_t host_access =
        AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
    const amdf_memory_profile_site_t sites[] = {
        {
            .kind = AMDF_MEMORY_SITE_KIND_HOST,
            .value = {.host_access = host_access},
        },
        {
            .kind = AMDF_MEMORY_SITE_KIND_DEVICE,
            .value = {.device = {0, family.queue_family_ordinal}},
        },
    };
    const auto* api = native_domain_->api;
    for (uint32_t i = 0; i < native_domain_->memory_scope_count; ++i) {
      SCOPED_TRACE(i);
      auto* scope = native_domain_->memory_scopes[i];
      amdf_memory_scope_info_t info = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
          .structure_size = sizeof(info),
      };
      ASSERT_EQ(api->memory_scope_query_info(scope, &info), AMDF_STATUS_OK);
      for (uint32_t ordinal = 0; ordinal < info.memory_profile_count;
           ++ordinal) {
        SCOPED_TRACE(ordinal);
        amdf_memory_profile_t profile = {
            .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
            .structure_size = sizeof(profile),
        };
        amdf_memory_access_capabilities_t capabilities = {
            .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
            .structure_size = sizeof(capabilities),
        };
        const amdf_status_t status = api->memory_scope_query_device_profile(
            scope, ordinal, 1, &access, &profile, &capabilities);
        if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
          continue;
        }
        ASSERT_EQ(status, AMDF_STATUS_OK);
        if (!iree_all_bits_set(profile.roles,
                               AMDF_MEMORY_PROFILE_ROLE_CREATE |
                                   AMDF_MEMORY_PROFILE_ROLE_HOST_MAP) ||
            !iree_all_bits_set(profile.supported_flags,
                               AMDF_MEMORY_FLAG_HOST_VISIBLE) ||
            !iree_all_bits_set(profile.host_mapping.supported_access,
                               host_access)) {
          continue;
        }
        amdf_memory_profile_pair_query_t query = {
            .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE_PAIR_QUERY,
            .structure_size = sizeof(query),
            .memory_profile_ordinal = ordinal,
            .access_count = 1,
            .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
            .accesses = &access,
        };
        bool complete = true;
        for (const auto& producer : sites) {
          SCOPED_TRACE(producer.kind);
          for (const auto& consumer : sites) {
            SCOPED_TRACE(consumer.kind);
            query.producer = producer;
            query.consumer = consumer;
            amdf_memory_pair_info_t pair = {
                .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
                .structure_size = sizeof(pair),
            };
            const amdf_status_t pair_status =
                api->memory_scope_query_pair_info(scope, &query, &pair);
            if (pair_status ==
                amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
              complete = false;
              continue;
            }
            ASSERT_EQ(pair_status, AMDF_STATUS_OK);
            complete &= iree_all_bits_set(
                pair.flags, AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE);
          }
        }
        *out_supported |= complete;
      }
    }
  }

  // Explicit native storage owner, kept alive through all borrowed roots.
  iree_hal_pool_t* pool_ = nullptr;
  // Borrowed native dependency domain owned by the constructed device.
  const iree_hal_amd_gpu_context_t* native_domain_ = nullptr;
  // Resolved once from the pool's immutable access contract.
  iree_hal_buffer_native_binding_slot_t slot_ = {};
  // Independent allocation, transfer, release, and host-gate timelines.
  std::array<iree_hal_semaphore_t*, 4> semaphores_ = {};
};

TEST_F(GpuNativeAllocationTest, NativeHostLocalityReachesGroupTopology) {
  const auto* identity =
      iree_hal_device_spec_identity(iree_hal_device_spec(device_));
  ASSERT_EQ(identity->physical_device_count, 1u);
  const auto& physical = identity->physical_devices[0].identity;
  const bool has_locality = native_domain_->host_numa_node != UINT32_MAX;
  EXPECT_EQ(iree_all_bits_set(physical.flags,
                              IREE_HAL_PHYSICAL_DEVICE_IDENTITY_FLAG_NUMA_NODE),
            has_locality);
  if (!has_locality) {
    return;
  }
  EXPECT_EQ(physical.numa.node_id, native_domain_->host_numa_node);
  const auto* topology = iree_hal_device_group_topology(group_);
  uint32_t matching_nodes = 0;
  for (uint32_t i = 0; i < iree_hal_topology_node_count(topology); ++i) {
    const auto* node = iree_hal_topology_node_at(topology, i);
    matching_nodes += node->kind == IREE_HAL_TOPOLOGY_NODE_KIND_HOST_NUMA &&
                      node->local_ordinal == native_domain_->host_numa_node;
  }
  EXPECT_EQ(matching_nodes, 1u);
}

TEST_F(GpuNativeAllocationTest, CachedHostPoolRetainsGuaranteesThroughCopy) {
  using iree::hal::cts::Ref;
  const iree_hal_pool_family_access_t family = {
      .family = iree_hal_queue_family(queues_[0]),
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
  };
  const iree_hal_pool_scope_t scope = {
      .family_count = 1,
      .families = &family,
      .host =
          {
              .access = IREE_HAL_MEMORY_ACCESS_ALL,
              .modes = IREE_HAL_MAPPING_MODE_PERSISTENT,
              .properties = IREE_HAL_HOST_MEMORY_PROPERTY_READ_CACHED,
          },
  };
  iree_hal_slab_pool_options_t options;
  iree_hal_slab_pool_options_initialize(&options);
  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(iree_hal_slab_pool_create(
      group_, scope, &options, iree_allocator_system(), pool.out()));
  const auto properties = iree_hal_pool_query_host_access(pool).properties;
  EXPECT_EQ(properties, IREE_HAL_HOST_MEMORY_PROPERTY_READ_CACHED |
                            IREE_HAL_HOST_MEMORY_PROPERTY_WRITE_BACK);
  VerifyHostPoolCopy(pool);
}

TEST_F(GpuNativeAllocationTest,
       RequiredLocalHostPoolCopiesThroughPublicMapping) {
  bool supported = false;
  ASSERT_NO_FATAL_FAILURE(QueryLocalHostSupport(&supported));
  RecordProperty("local_host_contract",
                 supported ? "qualified" : "unsupported");
  using iree::hal::cts::Ref;
  const iree_hal_pool_family_access_t family = {
      .family = iree_hal_queue_family(queues_[0]),
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
  };
  const iree_hal_pool_scope_t scope = {
      .family_count = 1,
      .families = &family,
      .host =
          {
              .access = IREE_HAL_MEMORY_ACCESS_ALL,
              .modes = IREE_HAL_MAPPING_MODE_PERSISTENT,
          },
  };
  const auto* topology = iree_hal_device_group_topology(group_);
  uint32_t physical_node = IREE_HAL_TOPOLOGY_NODE_ORDINAL_INVALID;
  for (iree_host_size_t i = 0; i < iree_hal_topology_node_count(topology);
       ++i) {
    const auto* node = iree_hal_topology_node_at(topology, i);
    if (node->kind == IREE_HAL_TOPOLOGY_NODE_KIND_PHYSICAL_DEVICE &&
        node->device_ordinal == 0) {
      physical_node = node->ordinal;
      break;
    }
  }
  ASSERT_NE(physical_node, IREE_HAL_TOPOLOGY_NODE_ORDINAL_INVALID);
  const iree_hal_slab_pool_options_t options = {
      .placement = {IREE_HAL_POOL_PLACEMENT_REQUIRED, physical_node},
  };
  Ref<iree_hal_pool_t> pool;
  if (!supported) {
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_UNAVAILABLE,
        iree_hal_slab_pool_create(group_, scope, &options,
                                  iree_allocator_system(), pool.out()));
    EXPECT_EQ(pool.get(), nullptr);
    return;
  }
  IREE_ASSERT_OK(iree_hal_slab_pool_create(
      group_, scope, &options, iree_allocator_system(), pool.out()));
  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(pool, &capabilities);
  EXPECT_EQ(capabilities.placement.mode, IREE_HAL_POOL_PLACEMENT_REQUIRED);
  EXPECT_EQ(capabilities.placement.node, physical_node);
  VerifyHostPoolCopy(pool);
}

class GpuNativeOrderedAllocationTest
    : public GpuNativeAllocationTest,
      public ::testing::WithParamInterface<
          std::tuple<amdf_queue_command_type_t, bool, bool>> {
 protected:
  amdf_queue_command_type_t command_type() const override {
    return std::get<0>(GetParam());
  }
};

TEST_P(GpuNativeOrderedAllocationTest,
       ReusePublishesConsumerBeforeProducerHostRetirement) {
  using iree::hal::cts::Ref;
  Ref<iree_hal_pool_t> finite_pool;
  CreateFinitePool(finite_pool.out());
  if (HasFatalFailure()) {
    return;
  }
  ASSERT_FALSE(iree_hal_pool_requires_asan_advice(finite_pool));
  const auto request = iree_hal_make_pool_reservation_request(4096, 4);
  Ref<iree_hal_buffer_t> first, second;
  uint64_t one = 1, two = 2, three = 3;
  IREE_ASSERT_OK(iree_hal_queue_alloca(queues_[1], {},
                                       {1, &semaphores_[0], &one}, finite_pool,
                                       1, &request, first.out()));
  Wait(semaphores_[0]);
  const uint64_t address =
      iree_hal_buffer_native_binding(first, slot_).device_address;
  ASSERT_NE(address, 0u);

  std::array<std::array<uint32_t, 16>, 2> actual = {};
  std::array<uint32_t, 2> patterns = {0x15A7C0DEu, 0x29B8D1EFu};
  std::array<Ref<iree_hal_buffer_t>, 2> snapshots;
  for (auto& snapshot : snapshots) {
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        pool_, iree_hal_make_pool_reservation_request(sizeof(actual[0]), 4),
        iree_infinite_timeout(), snapshot.out()));
  }
  Ref<iree_hal_semaphore_t> fill_complete;
  CreateSemaphore(fill_complete.out());
  if (HasFatalFailure()) {
    return;
  }
  uint64_t fill_value = 0;
  auto transfer = [&](iree_hal_buffer_t* buffer, size_t index,
                      iree_hal_semaphore_list_t waits,
                      iree_hal_semaphore_list_t signals) {
    // Transfer batch operations are unordered. The snapshot must follow the
    // fill through a native dependency, including while host retirement pauses.
    iree_hal_semaphore_t* filled = fill_complete;
    uint64_t value = ++fill_value;
    IREE_RETURN_IF_ERROR(iree_hal_queue_fill(
        queues_[1], waits, {1, &filled, &value}, buffer, 0,
        sizeof(actual[index]), &patterns[index], sizeof(patterns[index]),
        nullptr, IREE_HAL_FILL_FLAG_NONE));
    return iree_hal_queue_copy(
        queues_[1], {1, &filled, &value}, signals, buffer, 0, snapshots[index],
        0, sizeof(actual[index]), nullptr, IREE_HAL_COPY_FLAG_NONE);
  };
  // Warm the actual native transfer path before holding retirement. Host
  // downloads intentionally follow the native chain: their CPU copy would
  // legitimately prohibit native publication of the producer's completion.
  IREE_ASSERT_OK(transfer(first, 0, {1, &semaphores_[0], &one},
                          {1, &semaphores_[1], &one}));
  Wait(semaphores_[1]);
  patterns[0] = 0xC47A01DBu;

  CompletionPause pause{
      reinterpret_cast<iree_async_semaphore_t*>(semaphores_[3])};
  IREE_ASSERT_OK(pause.Arm(one));
  IREE_ASSERT_OK(iree_hal_queue_barrier(
      queues_[0], {}, {1, &semaphores_[3], &one}, nullptr, 0));
  pause.entered.acquire();

  IREE_ASSERT_OK(transfer(first, 0, {1, &semaphores_[1], &one},
                          {1, &semaphores_[1], &two}));
  iree_hal_amd_gpu_semaphore_wait_t producer = {};
  IREE_ASSERT_OK(iree_hal_amd_gpu_semaphore_acquire_wait(
      semaphores_[1], two, native_domain_, &producer));
  EXPECT_EQ(producer.state, IREE_HAL_AMD_GPU_SEMAPHORE_WAIT_NATIVE);
  iree_hal_amd_gpu_signal_release(producer.signal);
  uint64_t completed_value = 0;
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphores_[1], &completed_value));
  EXPECT_EQ(completed_value, one);
  iree_hal_buffer_t* first_handle = first;
  IREE_ASSERT_OK(iree_hal_queue_dealloca(queues_[1], {1, &semaphores_[1], &two},
                                         {1, &semaphores_[0], &two}, 1,
                                         &first_handle));
  IREE_ASSERT_OK(iree_hal_queue_alloca(
      queues_[std::get<1>(GetParam()) ? 0 : 1], {1, &semaphores_[0], &two},
      {1, &semaphores_[0], &three}, finite_pool, 1, &request, second.out()));
  IREE_ASSERT_OK(transfer(second, 1, {1, &semaphores_[0], &three},
                          {1, &semaphores_[2], &one}));

  // The native chain must already be submitted while its public producer
  // completion remains held. The one-block pool also proves physical reuse.
  iree_hal_amd_gpu_semaphore_wait_t published = {};
  IREE_ASSERT_OK(iree_hal_amd_gpu_semaphore_acquire_wait(
      semaphores_[2], one, native_domain_, &published));
  EXPECT_EQ(published.state, IREE_HAL_AMD_GPU_SEMAPHORE_WAIT_NATIVE);
  iree_hal_amd_gpu_signal_release(published.signal);
  EXPECT_EQ(iree_hal_buffer_native_binding(second, slot_).device_address,
            address);
  EXPECT_EQ(iree_hal_buffer_native_binding(first, slot_).device_address,
            address);
  const bool fail_producer = std::get<2>(GetParam());
  if (fail_producer) {
    iree_hal_semaphore_fail(
        semaphores_[1],
        iree_make_status(IREE_STATUS_ABORTED, "producer failed"));
  }
  pause.Join();
  if (fail_producer) {
    IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                          iree_hal_semaphore_wait(semaphores_[2], one,
                                                  iree_infinite_timeout(), 0));
  } else {
    Wait(semaphores_[2], one);
  }
  if (!fail_producer) {
    std::array<iree_hal_transfer_operation_t, 2> downloads = {};
    for (size_t i = 0; i < downloads.size(); ++i) {
      downloads[i].type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
      downloads[i].download.source_buffer = snapshots[i];
      downloads[i].download.target = actual[i].data();
      downloads[i].download.length = sizeof(actual[i]);
    }
    IREE_ASSERT_OK(iree_hal_queue_transfer(
        queues_[1], {1, &semaphores_[2], &one}, {1, &semaphores_[2], &two},
        downloads.size(), downloads.data(), nullptr));
    Wait(semaphores_[2], two);
    for (size_t i = 0; i < actual.size(); ++i) {
      for (uint32_t word : actual[i]) {
        EXPECT_EQ(word, patterns[i]);
      }
    }
  }
  EXPECT_EQ(iree_hal_buffer_native_binding(first, slot_).device_address, 0u);

  iree_hal_buffer_t* second_handle = second;
  IREE_ASSERT_OK(iree_hal_queue_dealloca(
      queues_[1],
      fail_producer ? iree_hal_semaphore_list_empty()
                    : iree_hal_semaphore_list_t{1, &semaphores_[2], &two},
      {1, &semaphores_[3], &two}, 1, &second_handle));
  // Native FIFO retirement still joins the already accepted consumer after
  // a logical dependency failure. Cleanup uses an independent public point.
  Wait(semaphores_[3], two);
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(finite_pool, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  EXPECT_EQ(stats.reserve_count, 2u);
  EXPECT_EQ(stats.release_count, 2u);
}

INSTANTIATE_TEST_SUITE_P(
    Protocols, GpuNativeOrderedAllocationTest,
    ::testing::Combine(::testing::Values(AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
                                         AMDF_QUEUE_COMMAND_TYPE_GPU_AQL),
                       ::testing::Bool(), ::testing::Bool()),
    [](const auto& info) {
      return std::string(std::get<0>(info.param) ==
                                 AMDF_QUEUE_COMMAND_TYPE_GPU_PM4
                             ? "PM4"
                             : "AQL") +
             (std::get<1>(info.param) ? "CrossQueue" : "SameQueue") +
             (std::get<2>(info.param) ? "ProducerFailure" : "Success");
    });

TEST_F(GpuNativeAllocationTest, PendingSubspansSurviveCrossQueueTransfers) {
  uint64_t one = 1, two = 2;
  std::array<iree_hal_pool_reservation_request_t, 2> requests;
  requests.fill(iree_hal_make_pool_reservation_request(4096, 4));
  std::array<iree_hal_buffer_t*, 2> roots = {}, spans = {};
  IREE_ASSERT_OK(iree_hal_queue_alloca(
      queues_[0], {1, &semaphores_[3], &one}, {1, &semaphores_[0], &one}, pool_,
      requests.size(), requests.data(), roots.data()));
  requests.fill({});
  for (size_t i = 0; i < roots.size(); ++i) {
    EXPECT_EQ(iree_hal_buffer_native_binding(roots[i], slot_).device_address,
              0u);
    IREE_ASSERT_OK(iree_hal_buffer_subspan(roots[i], 17, 257,
                                           iree_allocator_system(), &spans[i]));
  }
  // Failed capture must undo its prefix mark, even before backing exists.
  iree_hal_buffer_t* duplicates[] = {roots[0], roots[0]};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_queue_dealloca(queues_[0], {1, &semaphores_[0], &one}, {},
                              IREE_ARRAYSIZE(duplicates), duplicates));

  std::array<std::array<uint8_t, 257>, 2> actual = {};
  const std::array<uint8_t, 2> patterns = {0xA7, 0x53};
  std::array<iree_hal_transfer_operation_t, 2> fills = {}, downloads = {};
  for (size_t i = 0; i < roots.size(); ++i) {
    fills[i].type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL;
    fills[i].fill.target_buffer = spans[i];
    fills[i].fill.length = actual[i].size();
    fills[i].fill.pattern = &patterns[i];
    fills[i].fill.pattern_length = 1;
    downloads[i].type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
    downloads[i].download.source_buffer = spans[i];
    downloads[i].download.target = actual[i].data();
    downloads[i].download.length = actual[i].size();
  }
  IREE_ASSERT_OK(iree_hal_queue_transfer(
      queues_[1], {1, &semaphores_[1], &one}, {1, &semaphores_[1], &two},
      downloads.size(), downloads.data(), nullptr));
  IREE_ASSERT_OK(iree_hal_queue_transfer(queues_[0], {1, &semaphores_[0], &one},
                                         {1, &semaphores_[1], &one},
                                         fills.size(), fills.data(), nullptr));
  IREE_ASSERT_OK(iree_hal_queue_dealloca(queues_[1], {1, &semaphores_[1], &two},
                                         {1, &semaphores_[2], &one},
                                         roots.size(), roots.data()));
  for (auto* span : spans) {
    iree_hal_buffer_release(span);
  }
  IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphores_[3], one, nullptr));
  Wait(semaphores_[2]);
  for (size_t i = 0; i < roots.size(); ++i) {
    for (uint8_t byte : actual[i]) {
      EXPECT_EQ(byte, i ? 0x53 : 0xA7);
    }
    EXPECT_EQ(iree_hal_buffer_native_binding(roots[i], slot_).device_address,
              0u);
    iree_hal_buffer_release(roots[i]);
  }
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
}

TEST_F(GpuNativeAllocationTest, CapacityWaitDoesNotBlockReleasingItsStorage) {
  iree_hal_pool_t* finite_pool = nullptr;
  CreateFinitePool(&finite_pool);
  if (HasFatalFailure()) {
    return;
  }
  const auto request = iree_hal_make_pool_reservation_request(4096, 4);
  iree_hal_buffer_t* first = nullptr;
  iree_hal_buffer_t* second = nullptr;
  uint64_t one = 1, two = 2;
  IREE_ASSERT_OK(iree_hal_queue_alloca(queues_[0], {},
                                       {1, &semaphores_[0], &one}, finite_pool,
                                       1, &request, &first));
  Wait(semaphores_[0]);
  const uint64_t address =
      iree_hal_buffer_native_binding(first, slot_).device_address;
  ASSERT_NE(address, 0u);
  IREE_ASSERT_OK(iree_hal_queue_alloca(queues_[1], {},
                                       {1, &semaphores_[1], &one}, finite_pool,
                                       1, &request, &second));
  // A finite-pool miss enrolls capacity observation before submission returns.
  // Releasing the first allocation can therefore satisfy the second directly.
  uint64_t observed = 99;
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphores_[1], &observed));
  EXPECT_EQ(observed, 0u);
  IREE_ASSERT_OK(iree_hal_queue_dealloca(queues_[0], {1, &semaphores_[0], &one},
                                         {1, &semaphores_[2], &one}, 1,
                                         &first));
  Wait(semaphores_[1]);
  EXPECT_EQ(iree_hal_buffer_native_binding(second, slot_).device_address,
            address);
  IREE_ASSERT_OK(iree_hal_queue_dealloca(queues_[1], {1, &semaphores_[1], &one},
                                         {1, &semaphores_[2], &two}, 1,
                                         &second));
  iree_hal_buffer_release(first);
  iree_hal_buffer_release(second);
  Wait(semaphores_[2], two);
  iree_hal_pool_release(finite_pool);
}

TEST_F(GpuNativeAllocationTest, RecycledBatchJoinsIndependentOwnerFrontiers) {
  iree_hal_pool_t* finite_pool = nullptr;
  CreateFinitePool(&finite_pool, 2);
  if (HasFatalFailure()) {
    return;
  }
  std::array<iree_hal_pool_reservation_request_t, 2> requests;
  requests.fill(iree_hal_make_pool_reservation_request(4096, 4));
  std::array<iree_hal_pool_reservation_t, 2> reservations;
  std::array<iree_hal_pool_acquire_info_t, 2> infos;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations_sync(
      finite_pool, requests.size(), requests.data(), nullptr,
      reservations.data(), infos.data(), &result,
      IREE_HAL_POOL_RESERVE_FLAG_NONE));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  // Independent host participants return logical ownership before their
  // outstanding accesses complete. The next queue acquisition must join both.
  std::array<iree_async_axis_t, 2> axes;
  for (size_t i = 0; i < axes.size(); ++i) {
    axes[i] = iree_async_axis_make(1, 0, IREE_ASYNC_CAUSAL_DOMAIN_HOST, i);
    IREE_ASSERT_OK(
        iree_async_frontier_tracker_register_axis(tracker_, axes[i], nullptr));
    CompletedFrontier returned = {};
    iree_async_frontier_initialize(
        iree_async_fixed_frontier_as_frontier(&returned), 2);
    returned.entry_count = 1;
    returned.entries[0] = {axes[i], 1};
    iree_hal_pool_release_reservations(
        finite_pool, 1, &reservations[i],
        iree_async_fixed_frontier_as_frontier(&returned));
  }
  std::array<iree_hal_buffer_t*, 2> roots = {};
  uint64_t one = 1;
  IREE_ASSERT_OK(iree_hal_queue_alloca(
      queues_[0], {}, {1, &semaphores_[0], &one}, finite_pool, requests.size(),
      requests.data(), roots.data()));
  uint64_t observed = 99;
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphores_[0], &observed));
  EXPECT_EQ(observed, 0u);
  iree_async_frontier_tracker_advance(tracker_, axes[1], 1);
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphores_[0], &observed));
  EXPECT_EQ(observed, 0u);
  iree_async_frontier_tracker_advance(tracker_, axes[0], 1);
  Wait(semaphores_[0]);
  for (auto* root : roots) {
    EXPECT_NE(iree_hal_buffer_native_binding(root, slot_).device_address, 0u);
  }
  IREE_ASSERT_OK(iree_hal_queue_dealloca(queues_[1], {1, &semaphores_[0], &one},
                                         {1, &semaphores_[1], &one},
                                         roots.size(), roots.data()));
  for (auto* root : roots) {
    iree_hal_buffer_release(root);
  }
  Wait(semaphores_[1]);
  iree_hal_pool_release(finite_pool);
  for (auto axis : axes) {
    iree_async_frontier_tracker_retire_axis(
        tracker_, axis, iree_status_from_code(IREE_STATUS_CANCELLED));
  }
}

TEST_F(GpuNativeAllocationTest, RejectedReleasePreservesIndependentPools) {
  std::array<iree_hal_pool_t*, 2> pools = {};
  for (auto& pool : pools) {
    CreateFinitePool(&pool);
  }
  if (HasFatalFailure()) {
    return;
  }
  const auto request = iree_hal_make_pool_reservation_request(4096, 4);
  std::array<iree_hal_buffer_t*, 2> roots = {};
  std::array<uint64_t, 2> values = {1, 1};
  uint64_t one = 1;
  for (size_t i = 0; i < roots.size(); ++i) {
    IREE_ASSERT_OK(iree_hal_queue_alloca(queues_[i], {1, &semaphores_[3], &one},
                                         {1, &semaphores_[i], &values[i]},
                                         pools[i], 1, &request, &roots[i]));
  }
  const iree_hal_semaphore_list_t rejected_signal = {1, &semaphores_[2], &one};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_queue_dealloca(
                            queues_[0], {2, semaphores_.data(), values.data()},
                            rejected_signal, roots.size(), roots.data()));
  EXPECT_FALSE(iree_hal_semaphore_list_poll(rejected_signal));

  IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphores_[3], one, nullptr));
  uint64_t two = 2;
  for (size_t i = 0; i < roots.size(); ++i) {
    Wait(semaphores_[i]);
    iree_hal_pool_stats_t stats;
    iree_hal_pool_query_stats(pools[i], &stats);
    EXPECT_EQ(stats.reservation_count, 1u);
    EXPECT_NE(iree_hal_buffer_native_binding(roots[i], slot_).device_address,
              0u);
    IREE_ASSERT_OK(
        iree_hal_queue_dealloca(queues_[i], {1, &semaphores_[i], &one},
                                {1, &semaphores_[i], &two}, 1, &roots[i]));
    iree_hal_buffer_release(roots[i]);
    Wait(semaphores_[i], two);
    iree_hal_pool_query_stats(pools[i], &stats);
    EXPECT_EQ(stats.reservation_count, 0u);
    iree_hal_pool_release(pools[i]);
  }
  EXPECT_FALSE(iree_hal_semaphore_list_poll(rejected_signal));
}

TEST_F(GpuNativeAllocationTest, RejectedBatchLeavesOutputsAndPoolUntouched) {
  iree_hal_pool_t* finite_pool = nullptr;
  CreateFinitePool(&finite_pool);
  if (HasFatalFailure()) {
    return;
  }
  const iree_hal_pool_reservation_request_t requests[] = {
      iree_hal_make_pool_reservation_request(4096, 4),
      iree_hal_make_pool_reservation_request(4097, 4),
  };
  // Use existing live objects as sentinels; no output ownership transfers on
  // synchronous rejection of a request the finite pool can never satisfy.
  iree_hal_buffer_t* sentinel = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, requests[0], iree_infinite_timeout(), &sentinel));
  std::array<iree_hal_buffer_t*, 2> outputs = {sentinel, sentinel};
  uint64_t one = 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_queue_alloca(queues_[0], {}, {1, &semaphores_[0], &one},
                            finite_pool, IREE_ARRAYSIZE(requests), requests,
                            outputs.data()));
  for (auto* output : outputs) {
    EXPECT_EQ(output, sentinel);
  }
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(finite_pool, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  uint64_t observed = 99;
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphores_[0], &observed));
  EXPECT_EQ(observed, 0u);
  iree_hal_buffer_release(sentinel);
  iree_hal_pool_release(finite_pool);
}

TEST_F(GpuNativeAllocationTest, FailedReleaseDependencyPreservesReservation) {
  const auto request = iree_hal_make_pool_reservation_request(4096, 4);
  iree_hal_buffer_t* root = nullptr;
  uint64_t one = 1;
  IREE_ASSERT_OK(iree_hal_queue_alloca(
      queues_[0], {}, {1, &semaphores_[0], &one}, pool_, 1, &request, &root));
  Wait(semaphores_[0]);
  IREE_ASSERT_OK(iree_hal_queue_dealloca(queues_[0], {1, &semaphores_[3], &one},
                                         {1, &semaphores_[1], &one}, 1, &root));
  iree_hal_semaphore_fail(
      semaphores_[3], iree_make_status(IREE_STATUS_ABORTED, "producer failed"));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ABORTED,
      iree_hal_semaphore_wait(semaphores_[1], one, iree_infinite_timeout(),
                              IREE_ASYNC_WAIT_FLAG_NONE));
  EXPECT_NE(iree_hal_buffer_native_binding(root, slot_).device_address, 0u);
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 1u);
  IREE_ASSERT_OK(iree_hal_queue_dealloca(queues_[1], {1, &semaphores_[0], &one},
                                         {1, &semaphores_[2], &one}, 1, &root));
  iree_hal_buffer_release(root);
  Wait(semaphores_[2]);
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
}

class GpuNativeMemoryProgressTest : public GpuNativeAllocationTest {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(memory_gate_.CreateFabric(&fabric_));
    GpuNativeAllocationTest::SetUp();
  }

  // Native dependency gate, joined by base TearDown before table destruction.
  iree::hal::amd::gpu::cts::NativeMemoryGate memory_gate_;
};

class GpuNativeRecordingProgressTest
    : public GpuNativeMemoryProgressTest,
      public ::testing::WithParamInterface<
          std::pair<amdf_queue_command_type_t, bool>> {
 protected:
  amdf_queue_command_type_t command_type() const override {
    return GetParam().first;
  }

  void SetUp() override {
    if (GetParam().second) {
      memory_gate_.RequireCodeUpload();
    }
    GpuNativeMemoryProgressTest::SetUp();
  }
};

TEST_P(GpuNativeRecordingProgressTest,
       SealingAndExecutionCompleteWhileAllocationIsHeld) {
  using iree::hal::cts::Ref;
  const auto request = iree_hal_make_pool_reservation_request(4096, 4);
  uint64_t one = 1, two = 2;
  Ref<iree_hal_buffer_t> output, pending;
  IREE_ASSERT_OK(iree_hal_queue_alloca(queues_[0], {},
                                       {1, &semaphores_[0], &one}, pool_, 1,
                                       &request, output.out()));
  Wait(semaphores_[0]);
  {
    iree::hal::amd::gpu::cts::NativeMemoryGate::Hold hold(&memory_gate_);
    IREE_ASSERT_OK(iree_hal_queue_alloca(queues_[0], {},
                                         {1, &semaphores_[3], &one}, pool_, 1,
                                         &request, pending.out()));
    hold.WaitEntered();
    EXPECT_NE(memory_gate_.native_thread(), std::this_thread::get_id());

    Ref<iree_hal_command_buffer_t> recording;
    IREE_ASSERT_OK(iree_hal_command_buffer_create(
        iree_hal_queue_family(queues_[1]), IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
        IREE_HAL_COMMAND_CATEGORY_TRANSFER, 0, recording.out()));
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(recording));
    const uint32_t pattern = 0x1234CAFE;
    IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
        recording, iree_hal_make_buffer_ref(output, 0, 16), &pattern,
        sizeof(pattern), 0));
    const iree_hal_barrier_t release = {
        .source_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
        .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
        .flags = IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
    };
    IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &release));
    // The worker cannot finish its native call until this scope exits. Sealing
    // must allocate both resident and optional upload backing on this caller.
    IREE_ASSERT_OK(iree_hal_command_buffer_end(recording));
    IREE_ASSERT_OK(iree_hal_queue_execute(
        queues_[1], {}, {1, &semaphores_[1], &one}, recording, {}, 0));
    std::array<uint32_t, 4> observed = {};
    const iree_hal_transfer_operation_t download = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .download =
            {
                .source_buffer = output,
                .target = observed.data(),
                .length = sizeof(observed),
            },
    };
    IREE_ASSERT_OK(iree_hal_queue_transfer(
        queues_[1], {1, &semaphores_[1], &one}, {1, &semaphores_[2], &one}, 1,
        &download, nullptr));
    Wait(semaphores_[2]);
    for (uint32_t value : observed) {
      EXPECT_EQ(value, pattern);
    }
  }
  Wait(semaphores_[3]);
  iree_hal_buffer_t* buffers[] = {output, pending};
  IREE_ASSERT_OK(iree_hal_queue_dealloca(queues_[1], {1, &semaphores_[2], &one},
                                         {1, &semaphores_[2], &two},
                                         IREE_ARRAYSIZE(buffers), buffers));
  Wait(semaphores_[2], two);
}

TEST_P(GpuNativeRecordingProgressTest,
       RecordedResourcesRetireWhileUnusedCloneAllocationIsHeld) {
  using iree::hal::cts::Ref;
  const auto request = iree_hal_make_pool_reservation_request(4096, 4);
  uint64_t one = 1;
  Ref<iree_hal_buffer_t> output;
  IREE_ASSERT_OK(iree_hal_queue_alloca(queues_[0], {},
                                       {1, &semaphores_[0], &one}, pool_, 1,
                                       &request, output.out()));
  Wait(semaphores_[0]);
  Ref<iree_hal_command_buffer_t> recording;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      iree_hal_queue_family(queues_[0]), IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, 0, recording.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(recording));
  const uint32_t pattern = 0x1234CAFE;
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      recording, iree_hal_make_buffer_ref(output, 0, 16), &pattern,
      sizeof(pattern), 0));
  const iree_hal_barrier_t release = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      .flags = IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &release));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(recording));

  CompletionPause pause{
      reinterpret_cast<iree_async_semaphore_t*>(semaphores_[3])};
  IREE_ASSERT_OK(pause.Arm(one));
  IREE_ASSERT_OK(iree_hal_queue_barrier(
      queues_[0], {}, {1, &semaphores_[3], &one}, nullptr, 0));
  pause.entered.acquire();
  IREE_ASSERT_OK(iree_hal_queue_execute(
      queues_[0], {}, {1, &semaphores_[1], &one}, recording, {}, 0));
  {
    // The first invocation owns the initial instance until host retirement.
    // An independent queue requests a clone; returned capacity must let it
    // finish while that unnecessary native allocation remains blocked.
    iree::hal::amd::gpu::cts::NativeMemoryGate::Hold hold(&memory_gate_);
    IREE_ASSERT_OK(iree_hal_queue_execute(
        queues_[1], {}, {1, &semaphores_[2], &one}, recording, {}, 0));
    hold.WaitEntered();
    recording.reset();
    pause.Join();
    Wait(semaphores_[1]);
    Wait(semaphores_[2]);
    EXPECT_EQ(iree_atomic_ref_count_load(&output.get()->resource.ref_count), 1);
  }
  std::array<uint32_t, 4> observed = {};
  const iree_hal_transfer_operation_t download = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
      .download =
          {
              .source_buffer = output,
              .target = observed.data(),
              .length = sizeof(observed),
          },
  };
  uint64_t two = 2;
  IREE_ASSERT_OK(iree_hal_queue_transfer(queues_[1], {1, &semaphores_[2], &one},
                                         {1, &semaphores_[2], &two}, 1,
                                         &download, nullptr));
  Wait(semaphores_[2], two);
  for (uint32_t value : observed) {
    EXPECT_EQ(value, pattern);
  }
  // The original recording and both invocations have gone. The unused clone
  // and its final program reference are joined by ordinary device teardown.
}

TEST_P(GpuNativeRecordingProgressTest,
       RejectedInvocationReleasesResourcesBeforeFailureNotification) {
  using iree::hal::cts::Ref;
  const auto request = iree_hal_make_pool_reservation_request(4096, 4);
  uint64_t one = 1;
  Ref<iree_hal_buffer_t> output;
  IREE_ASSERT_OK(iree_hal_queue_alloca(queues_[0], {},
                                       {1, &semaphores_[0], &one}, pool_, 1,
                                       &request, output.out()));
  Wait(semaphores_[0]);
  Ref<iree_hal_command_buffer_t> recording;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      iree_hal_queue_family(queues_[0]), IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, 0, recording.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(recording));
  const uint32_t pattern = 0x1234CAFE;
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      recording, iree_hal_make_buffer_ref(output, 0, 16), &pattern,
      sizeof(pattern), 0));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(recording));
  struct Completion {
    // The caller retains this buffer through the notification observation.
    iree_hal_buffer_t* buffer;
    // Payload ownership observed exactly when public failure becomes visible.
    int32_t references = 0;
    // Terminal result, published with the observation through finished.
    iree_status_code_t code = IREE_STATUS_OK;
    // Joins the callback's last access to this stack state.
    std::binary_semaphore finished{0};
  } completion{output};
  iree_async_semaphore_timepoint_t timepoint = {
      .callback =
          [](void* user_data, iree_async_semaphore_timepoint_t*,
             iree_status_t status) {
            auto* completion = static_cast<Completion*>(user_data);
            completion->references = iree_atomic_ref_count_load(
                &completion->buffer->resource.ref_count);
            completion->code = iree_status_consume_code(status);
            completion->finished.release();
          },
      .user_data = &completion,
  };
  IREE_ASSERT_OK(iree_async_semaphore_acquire_timepoint(
      reinterpret_cast<iree_async_semaphore_t*>(semaphores_[2]), one,
      &timepoint));
  IREE_ASSERT_OK(iree_hal_queue_execute(queues_[0], {1, &semaphores_[1], &one},
                                        {1, &semaphores_[2], &one}, recording,
                                        {}, 0));
  recording.reset();
  iree_hal_semaphore_fail(semaphores_[1],
                          iree_status_from_code(IREE_STATUS_ABORTED));
  completion.finished.acquire();
  EXPECT_EQ(completion.code, IREE_STATUS_ABORTED);
  EXPECT_EQ(completion.references, 1);
}

INSTANTIATE_TEST_SUITE_P(
    LocalPublication, GpuNativeRecordingProgressTest,
    ::testing::Values(std::pair{AMDF_QUEUE_COMMAND_TYPE_GPU_PM4, false},
                      std::pair{AMDF_QUEUE_COMMAND_TYPE_GPU_PM4, true},
                      std::pair{AMDF_QUEUE_COMMAND_TYPE_GPU_AQL, false},
                      std::pair{AMDF_QUEUE_COMMAND_TYPE_GPU_AQL, true}),
    [](const auto& info) {
      return std::string(info.param.first == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL
                             ? "Aql"
                             : "Pm4") +
             (info.param.second ? "Staged" : "Mapped");
    });

TEST_F(GpuNativeMemoryProgressTest,
       HostCallbacksCompleteWhileAllocationIsHeld) {
  using iree::hal::cts::Ref;
  for (auto flags :
       {IREE_HAL_HOST_CALL_FLAG_NONE, IREE_HAL_HOST_CALL_FLAG_RELAXED}) {
    SCOPED_TRACE(flags);
    Ref<iree_hal_semaphore_t> done;
    CreateSemaphore(done.out());
    Ref<iree_hal_semaphore_t> allocated;
    CreateSemaphore(allocated.out());
    Ref<iree_hal_buffer_t> cold;
    {
      iree::hal::amd::gpu::cts::NativeMemoryGate::Hold hold(&memory_gate_);
      const auto request = iree_hal_make_pool_reservation_request(4096, 64);
      uint64_t allocation_value = 1;
      iree_hal_semaphore_t* allocated_handle = allocated;
      IREE_ASSERT_OK(iree_hal_queue_alloca(
          queues_[0], {}, {1, &allocated_handle, &allocation_value}, pool_, 1,
          &request, cold.out()));
      hold.WaitEntered();
      struct Callback {
        // Exact other native queue used for a reentrant asynchronous callback.
        iree_hal_queue_t* next_queue;
        // Actual callback executor identity, published through called.
        std::thread::id thread;
        // Joins callback entry without depending on the return cache boundary.
        std::binary_semaphore called{0};
      } callback = {.next_queue = queues_[1]};
      const auto call = iree_hal_make_host_call(
          +[](void* user_data, const uint64_t arguments[4],
              iree_hal_host_call_context_t* context) {
            auto* callback = static_cast<Callback*>(user_data);
            callback->thread = std::this_thread::get_id();
            iree_status_t status = iree_hal_queue_barrier(
                callback->next_queue, {}, context->signal_semaphore_list,
                nullptr, 0);
            callback->called.release();
            return iree_status_is_ok(status)
                       ? iree_status_from_code(IREE_STATUS_DEFERRED)
                       : status;
          },
          &callback);
      uint64_t value = 1;
      iree_hal_semaphore_t* done_handle = done;
      const uint64_t arguments[4] = {0};
      IREE_ASSERT_OK(iree_hal_queue_host_call(
          queues_[0], {}, {1, &done_handle, &value}, call, arguments, flags));
      callback.called.acquire();
      EXPECT_NE(callback.thread, memory_gate_.native_thread());
      EXPECT_NE(callback.thread, std::this_thread::get_id());
      IREE_ASSERT_OK(
          iree_hal_semaphore_wait(done, 1, iree_infinite_timeout(), 0));
      // Both entry and return GPU boundaries have completed while the native
      // allocation owner is still held. The scoped gate joins on all exits.
    }
    IREE_ASSERT_OK(
        iree_hal_semaphore_wait(allocated, 1, iree_infinite_timeout(), 0));
    iree_hal_buffer_t* cold_handle = cold;
    uint64_t returned = 2;
    iree_hal_semaphore_t* allocated_handle = allocated;
    IREE_ASSERT_OK(iree_hal_queue_dealloca(
        queues_[0], {}, {1, &allocated_handle, &returned}, 1, &cold_handle));
    // The source pool is borrowed until the queued return is terminal.
    Wait(allocated, returned);
  }
}

TEST_F(GpuNativeMemoryProgressTest,
       RecycledStorageAndGpuConsumerCompleteWhileAllocationIsHeld) {
  using iree::hal::cts::Ref;
  Ref<iree_hal_pool_t> finite_pool;
  CreateFinitePool(finite_pool.out());
  if (HasFatalFailure()) {
    return;
  }
  const auto request = iree_hal_make_pool_reservation_request(4096, 4);
  Ref<iree_hal_buffer_t> first;
  uint64_t one = 1, two = 2, three = 3;
  IREE_ASSERT_OK(iree_hal_queue_alloca(queues_[0], {},
                                       {1, &semaphores_[0], &one}, finite_pool,
                                       1, &request, first.out()));
  Wait(semaphores_[0]);
  const uint64_t address =
      iree_hal_buffer_native_binding(first, slot_).device_address;
  ASSERT_NE(address, 0u);

  Ref<iree_hal_buffer_t> cold;
  {
    iree::hal::amd::gpu::cts::NativeMemoryGate::Hold hold(&memory_gate_);
    IREE_ASSERT_OK(iree_hal_queue_alloca(queues_[0], {},
                                         {1, &semaphores_[3], &one}, pool_, 1,
                                         &request, cold.out()));
    hold.WaitEntered();

    // Logical return and reacquisition require no native allocation or map.
    // The other queue must observe the same physical range and its history.
    iree_hal_buffer_t* first_handle = first;
    IREE_ASSERT_OK(
        iree_hal_queue_dealloca(queues_[0], {1, &semaphores_[0], &one},
                                {1, &semaphores_[0], &two}, 1, &first_handle));
    Ref<iree_hal_buffer_t> reused;
    IREE_ASSERT_OK(iree_hal_queue_alloca(
        queues_[1], {1, &semaphores_[0], &two}, {1, &semaphores_[1], &one},
        finite_pool, 1, &request, reused.out()));
    Wait(semaphores_[1]);
    EXPECT_EQ(iree_hal_buffer_native_binding(reused, slot_).device_address,
              address);

    const uint32_t pattern = 0xCAFE1234;
    IREE_ASSERT_OK(iree_hal_queue_fill(
        queues_[1], {1, &semaphores_[1], &one}, {1, &semaphores_[1], &two},
        reused, 0, sizeof(pattern), &pattern, sizeof(pattern), nullptr, 0));
    uint32_t observed = 0;
    const iree_hal_transfer_operation_t download = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .download =
            {
                .source_buffer = reused,
                .target = &observed,
                .length = sizeof(observed),
            },
    };
    IREE_ASSERT_OK(iree_hal_queue_transfer(
        queues_[0], {1, &semaphores_[1], &two}, {1, &semaphores_[2], &one}, 1,
        &download, nullptr));
    Wait(semaphores_[2]);
    EXPECT_EQ(observed, pattern);

    iree_hal_buffer_t* reused_handle = reused;
    IREE_ASSERT_OK(iree_hal_queue_dealloca(
        queues_[1], {1, &semaphores_[2], &one}, {1, &semaphores_[1], &three}, 1,
        &reused_handle));
    Wait(semaphores_[1], three);
  }
  Wait(semaphores_[3]);
  iree_hal_buffer_t* cold_handle = cold;
  IREE_ASSERT_OK(iree_hal_queue_dealloca(queues_[0], {1, &semaphores_[3], &one},
                                         {1, &semaphores_[3], &two}, 1,
                                         &cold_handle));
  Wait(semaphores_[3], two);
}

TEST_F(GpuNativeMemoryProgressTest,
       MappingAndGpuConsumerOvertakeRemainingAllocationBatch) {
  using iree::hal::cts::Ref;
  const iree_hal_pool_family_access_t access = {
      .family = iree_hal_queue_family(queues_[0]),
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
      .interfaces = (UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS) |
                    (UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE),
  };
  iree_hal_slab_pool_options_t options;
  iree_hal_slab_pool_options_initialize(&options);
  Ref<iree_hal_pool_t> mapping_pool;
  IREE_ASSERT_OK(iree_hal_slab_pool_create(group_, {1, &access, {}}, &options,
                                           iree_allocator_system(),
                                           mapping_pool.out()));
  auto target_access = access;
  target_access.interfaces = UINT64_C(1)
                             << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS;
  const iree_hal_pool_scope_t scope = {1, &target_access, {}};
  iree_hal_pool_t* source_pool = mapping_pool.get();
  iree_hal_virtual_buffer_capabilities_t capabilities;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_query_capabilities(
      group_, scope, {1, &source_pool}, iree_allocator_system(),
      &capabilities));
  ASSERT_NE(capabilities.max_byte_length, 0u);
  const auto source = iree_hal_pool_query_mapping_requirements(mapping_pool);
  const uint64_t unit =
      iree_max(source.minimum_granularity,
               iree_max(capabilities.mapping_granularity,
                        capabilities.reservation_granularity));
  const iree_hal_virtual_buffer_options_t reservation_options = {
      .byte_length = unit,
      .mapping_granularity = unit,
  };
  Ref<iree_hal_buffer_t> reservation, backing;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_create(
      group_, scope, {1, &source_pool}, &reservation_options,
      iree_allocator_system(), reservation.out()));
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      mapping_pool, iree_hal_make_pool_reservation_request(unit, 0),
      iree_infinite_timeout(), backing.out()));
  iree_hal_memory_scope_t program_scope;
  IREE_ASSERT_OK(iree_hal_device_group_resolve_memory_scope(
      group_, {IREE_HAL_MEMORY_SITE_PROGRAM, access.family}, &program_scope));
  const iree_hal_memory_mapping_t mapping = {
      .target = reservation,
      .scope = program_scope,
      .target_range = {0, unit},
      .source = backing,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .operation = IREE_HAL_MEMORY_MAPPING_OPERATION_BIND,
  };

  std::array<iree_hal_pool_reservation_request_t, 100> requests;
  std::array<iree_hal_buffer_t*, 100> cold = {};
  for (auto& request : requests) {
    request = iree_hal_make_pool_reservation_request(4096, 64);
  }
  uint64_t one = 1, two = 2;
  {
    iree::hal::amd::gpu::cts::NativeMemoryGate::Hold hold(&memory_gate_);
    IREE_ASSERT_OK(
        iree_hal_queue_alloca(queues_[0], {}, {1, &semaphores_[3], &one}, pool_,
                              requests.size(), requests.data(), cold.data()));
    hold.WaitEntered();
    IREE_ASSERT_OK(iree_hal_queue_map(
        queues_[1], {}, {1, &semaphores_[0], &one}, 1, &mapping, 0));
    hold.ReleaseAndHoldNext();

    // Native mapping must have overtaken slab two. Its public completion and
    // GPU consumers cannot need any further turn on the held allocation owner.
    Wait(semaphores_[0]);
    const uint32_t pattern = 0xC0FFEE42;
    IREE_ASSERT_OK(iree_hal_queue_fill(queues_[1], {1, &semaphores_[0], &one},
                                       {1, &semaphores_[1], &one}, reservation,
                                       0, sizeof(pattern), &pattern,
                                       sizeof(pattern), nullptr, 0));
    uint32_t observed = 0;
    IREE_ASSERT_OK(iree_hal_queue_download(
        queues_[0], {1, &semaphores_[1], &one}, {1, &semaphores_[2], &one},
        reservation, 0, &observed, sizeof(observed), nullptr));
    Wait(semaphores_[2]);
    EXPECT_EQ(observed, pattern);
  }
  Wait(semaphores_[3]);
  IREE_ASSERT_OK(iree_hal_queue_dealloca(queues_[0], {1, &semaphores_[3], &one},
                                         {1, &semaphores_[3], &two},
                                         cold.size(), cold.data()));
  Wait(semaphores_[3], two);
  for (auto* buffer : cold) {
    iree_hal_buffer_release(buffer);
  }
  const iree_hal_memory_mapping_t unmap = {
      .target = reservation,
      .scope = program_scope,
      .target_range = {0, unit},
      .operation = IREE_HAL_MEMORY_MAPPING_OPERATION_UNBIND,
  };
  IREE_ASSERT_OK(iree_hal_queue_map(queues_[0], {}, {1, &semaphores_[0], &two},
                                    1, &unmap, 0));
  Wait(semaphores_[0], two);
}

}  // namespace
