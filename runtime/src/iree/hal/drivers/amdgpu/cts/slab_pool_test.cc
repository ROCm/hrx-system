// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cstring>

#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/memory/fixed_block_pool.h"
#include "iree/hal/memory/maintenance.h"
#include "iree/hal/memory/slab_cache.h"
#include "iree/hal/memory/tlsf_pool.h"

namespace iree::hal::cts {
namespace {

class AmdgpuSlabPoolTest : public CtsTestBase<> {
 protected:
  void SetUp() override {
    CtsTestBase::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    iree_hal_executable_target_selection_result_t selection;
    IREE_ASSERT_OK(SelectExecutableTarget(
        iree_hal_device_queue_family(device_, 0), &selection));
    if (selection.outcome ==
        IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_NO_MATCH) {
      GTEST_SKIP() << "native executable target is unavailable";
    }
    ASSERT_EQ(selection.outcome,
              IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED);
    // Separate native owners and progress services have equal device-local
    // ordinals. Their exact family identities and group sites remain distinct.
    for (size_t i = 0; i < devices_.size(); ++i) {
      IREE_ASSERT_OK(contexts_[i].Initialize(iree_allocator_system()));
      IREE_ASSERT_OK(iree_hal_driver_create_default_device(
          driver_, contexts_[i].params(), iree_allocator_system(),
          &devices_[i]));
    }
    iree_hal_device_group_builder_t builder;
    iree_hal_device_group_builder_initialize(&builder,
                                             contexts_[0].frontier_tracker());
    iree_status_t status = iree_ok_status();
    for (auto* device : devices_) {
      if (iree_status_is_ok(status)) {
        status = iree_hal_device_group_builder_add_device(&builder, device);
      }
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_group_builder_finalize(
          &builder, iree_allocator_system(), &group_);
    }
    iree_hal_device_group_builder_deinitialize(&builder);
    IREE_ASSERT_OK(status);
    for (size_t i = 0; i < devices_.size(); ++i) {
      queues_[i] = iree_hal_device_queue(devices_[i], 0, 0);
      ASSERT_NE(queues_[i], nullptr);
    }
    IREE_ASSERT_OK(SelectBackendExecutableTarget(
        devices_[1], iree_hal_queue_family(queues_[1]), GetParam(),
        &selection));
    ASSERT_EQ(selection.outcome,
              IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED);
    IREE_ASSERT_OK(LoadExecutable(
        iree_hal_queue_family(queues_[1]), selection.target,
        IREE_HAL_EXECUTABLE_LOAD_FLAG_NONE,
        executable_data(
            IREE_SV("command_buffer_dispatch_constants_bindings_test.bin")),
        executable_.out()));
    families_[0].family = iree_hal_queue_family(queues_[0]);
    families_[0].usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
    families_[1].family = iree_hal_queue_family(queues_[1]);
    families_[1].usage =
        IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_STORAGE;
  }

  void TearDown() override {
    executable_.reset();
    for (auto* device : devices_) {
      iree_hal_device_release(device);
    }
    iree_hal_device_group_release(group_);
    for (auto& context : contexts_) {
      context.Deinitialize();
    }
    CtsTestBase::TearDown();
  }

  iree_hal_pool_scope_t Scope() const {
    return {families_.size(), families_.data(), {}};
  }

  iree_status_t CreateSource(iree_hal_pool_scope_t scope,
                             iree_hal_pool_t** out_pool) {
    iree_hal_slab_pool_options_t options;
    iree_hal_slab_pool_options_initialize(&options);
    return iree_hal_slab_pool_create(group_, scope, &options,
                                     iree_allocator_system(), out_pool);
  }

  void Wait(iree_hal_semaphore_list_t semaphores) {
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        semaphores, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  }

  // Joins queued child/cache returns, including work they enqueue. A pending
  // frontier callback can still enqueue native source retirement afterward.
  void JoinMaintenance(iree_hal_pool_t* source) {
    iree_hal_memory_maintenance_call(
        source->maintenance,
        [](void* user_data) {
          auto* owner = static_cast<iree_hal_memory_maintenance_t*>(user_data);
          while (iree_hal_memory_maintenance_run_one(owner)) {
          }
        },
        source->maintenance);
  }

  // Independent native progress owners kept alive until their group retires.
  std::array<DeviceCreateContext, 3> contexts_;
  // Owned logical devices, also retained by the sealed group.
  std::array<iree_hal_device_t*, 3> devices_ = {};
  // Borrowed provisioned queues with equal device-local family ordinals.
  std::array<iree_hal_queue_t*, 3> queues_ = {};
  // Owned immutable group for the three independently created devices.
  iree_hal_device_group_t* group_ = nullptr;
  // Owned native kernel used on the second device.
  Ref<iree_hal_executable_t> executable_;
  // Simultaneous execution scope; the third device is excluded.
  std::array<iree_hal_pool_family_access_t, 2> families_ = {};
};

TEST_P(AmdgpuSlabPoolTest, CapturesExactFamiliesWithoutPayload) {
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(CreateSource(Scope(), source.out()));
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(source, &stats);
  EXPECT_EQ(stats.reserve_count, 0u);
  EXPECT_EQ(stats.bytes_committed, 0u);
  EXPECT_EQ(iree_hal_pool_query_host_access(source).access,
            IREE_HAL_MEMORY_ACCESS_NONE);
  for (size_t i = 0; i < queues_.size(); ++i) {
    for (auto kind :
         {IREE_HAL_MEMORY_SITE_QUEUE, IREE_HAL_MEMORY_SITE_PROGRAM}) {
      const iree_hal_memory_site_t site = {kind,
                                           iree_hal_queue_family(queues_[i])};
      iree_hal_memory_scope_t scope;
      IREE_ASSERT_OK(
          iree_hal_device_group_resolve_memory_scope(group_, site, &scope));
      iree_hal_buffer_native_binding_slot_t slot;
      if (i < families_.size()) {
        IREE_ASSERT_OK(iree_hal_pool_resolve_binding(
            source, scope, IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS, &slot));
      } else {
        IREE_EXPECT_STATUS_IS(
            IREE_STATUS_PERMISSION_DENIED,
            iree_hal_pool_resolve_binding(
                source, scope, IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
                &slot));
      }
    }
  }
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      source, {}, sizeof(uint32_t), iree_infinite_timeout(), buffer.out()));
  uint32_t value = 42;
  SemaphoreList filled(devices_[0], {0}, {1});
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_queue_fill(queues_[2], iree_hal_semaphore_list_empty(), filled,
                          buffer, 0, sizeof(value), &value, sizeof(value),
                          IREE_HAL_FILL_FLAG_NONE));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_buffer_validate_family_usage(buffer, families_[0].family,
                                            IREE_HAL_BUFFER_USAGE_STORAGE));
  IREE_ASSERT_OK(iree_hal_queue_fill(
      queues_[0], iree_hal_semaphore_list_empty(), filled, buffer, 0,
      sizeof(value), &value, sizeof(value), IREE_HAL_FILL_FLAG_NONE));
  uint32_t output = 0;
  SemaphoreList downloaded(devices_[1], {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_download(queues_[1], filled, downloaded, buffer,
                                         0, &output, sizeof(output)));
  Wait(downloaded);
  EXPECT_EQ(output, value);
  buffer.reset();
  JoinMaintenance(source);
}

TEST_P(AmdgpuSlabPoolTest, NativeOnlyQueueAllocationKeepsZeroUsage) {
  for (auto& family : families_) {
    family.usage = IREE_HAL_BUFFER_USAGE_NONE;
  }
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(CreateSource(Scope(), source.out()));
  iree_hal_pool_reservation_request_t request = {};
  request.allocation_size = sizeof(uint32_t);
  Ref<iree_hal_buffer_t> buffer;
  SemaphoreList allocated(devices_[0], {0}, {1});
  IREE_ASSERT_OK(
      iree_hal_queue_alloca(queues_[0], iree_hal_semaphore_list_empty(),
                            allocated, source, 1, &request, buffer.out()));
  EXPECT_EQ(iree_hal_buffer_allowed_usage(buffer), IREE_HAL_BUFFER_USAGE_NONE);
  const uint32_t value = 42;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_queue_fill(
          queues_[1], allocated, iree_hal_semaphore_list_empty(), buffer, 0,
          sizeof(value), &value, sizeof(value), IREE_HAL_FILL_FLAG_NONE));
  SemaphoreList deallocated(devices_[1], {0}, {1});
  auto* raw_buffer = buffer.get();
  IREE_ASSERT_OK(iree_hal_queue_dealloca(queues_[1], allocated, deallocated, 1,
                                         &raw_buffer));
  Wait(deallocated);
  buffer.reset();
  JoinMaintenance(source);
}

TEST_P(AmdgpuSlabPoolTest, PublicHostGrantsAndNativeClassRequirements) {
  for (auto access : {IREE_HAL_MEMORY_ACCESS_READ, IREE_HAL_MEMORY_ACCESS_WRITE,
                      IREE_HAL_MEMORY_ACCESS_ALL}) {
    for (auto mode :
         {IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MAPPING_MODE_PERSISTENT}) {
      auto scope = Scope();
      scope.host.access = access;
      scope.host.modes = mode;
      families_[1].requirements =
          IREE_HAL_POOL_ACCESS_REQUIRE_COHERENT_WITH_HOST;
      Ref<iree_hal_pool_t> source;
      IREE_ASSERT_OK(CreateSource(scope, source.out()));
      const auto host = iree_hal_pool_query_host_access(source);
      EXPECT_EQ(host.access, access);
      EXPECT_EQ(host.modes, mode);
      Ref<iree_hal_buffer_t> buffer;
      IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
          source, {}, sizeof(uint32_t), iree_infinite_timeout(), buffer.out()));
      const uint32_t value = 0xAABBCCDD;
      SemaphoreList filled(devices_[0], {0}, {1});
      IREE_ASSERT_OK(iree_hal_queue_fill(
          queues_[0], iree_hal_semaphore_list_empty(), filled, buffer, 0,
          sizeof(value), &value, sizeof(value), IREE_HAL_FILL_FLAG_NONE));
      Wait(filled);
      iree_hal_buffer_mapping_t mapping = {};
      for (auto denied :
           {IREE_HAL_MEMORY_ACCESS_READ, IREE_HAL_MEMORY_ACCESS_WRITE}) {
        if (!iree_any_bit_set(access, denied)) {
          IREE_EXPECT_STATUS_IS(
              IREE_STATUS_PERMISSION_DENIED,
              iree_hal_buffer_map_range(buffer, mode, denied,
                                        IREE_HAL_BUFFER_MAP_FLAG_NONE, 0,
                                        sizeof(value), &mapping));
        }
      }
      IREE_ASSERT_OK(iree_hal_buffer_map_range(buffer, mode, access,
                                               IREE_HAL_BUFFER_MAP_FLAG_NONE, 0,
                                               sizeof(value), &mapping));
      if (iree_any_bit_set(access, IREE_HAL_MEMORY_ACCESS_READ)) {
        uint32_t actual = 0;
        std::memcpy(&actual, mapping.contents.data, sizeof(actual));
        EXPECT_EQ(actual, value);
      }
      if (iree_any_bit_set(access, IREE_HAL_MEMORY_ACCESS_WRITE)) {
        const uint32_t updated = value + 1;
        std::memcpy(mapping.contents.data, &updated, sizeof(updated));
      }
      IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
      uint32_t actual = 0;
      SemaphoreList downloaded(devices_[1], {0}, {1});
      IREE_ASSERT_OK(iree_hal_queue_download(
          queues_[1], filled, downloaded, buffer, 0, &actual, sizeof(actual)));
      Wait(downloaded);
      EXPECT_EQ(actual,
                value + (iree_any_bit_set(access, IREE_HAL_MEMORY_ACCESS_WRITE)
                             ? 1u
                             : 0u));
      buffer.reset();
      JoinMaintenance(source);
    }
  }
  families_[1].requirements = IREE_HAL_POOL_ACCESS_REQUIRE_UNCACHED;
  Ref<iree_hal_pool_t> rejected;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        CreateSource(Scope(), rejected.out()));
  families_[1].requirements = 0;
  for (auto cacheability : {IREE_HAL_HOST_CACHEABILITY_WRITE_BACK,
                            IREE_HAL_HOST_CACHEABILITY_WRITE_COMBINED,
                            IREE_HAL_HOST_CACHEABILITY_UNCACHED}) {
    auto scope = Scope();
    scope.host = {IREE_HAL_MEMORY_ACCESS_READ, IREE_HAL_MAPPING_MODE_SCOPED,
                  cacheability};
    IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                          CreateSource(scope, rejected.out()));
    iree_hal_slab_pool_options_t options;
    iree_hal_slab_pool_options_initialize(&options);
    options.preferences.host = scope.host;
    Ref<iree_hal_pool_t> preferred;
    IREE_ASSERT_OK(iree_hal_slab_pool_create(
        group_, Scope(), &options, iree_allocator_system(), preferred.out()));
    EXPECT_EQ(iree_hal_pool_query_host_access(preferred).access,
              IREE_HAL_MEMORY_ACCESS_NONE);
  }
}

TEST_P(AmdgpuSlabPoolTest, PrivateHostBindingsSurviveQueuedCommitment) {
  families_[0].interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_HOST;
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(CreateSource(Scope(), source.out()));
  const iree_hal_memory_site_t site = {IREE_HAL_MEMORY_SITE_PROGRAM,
                                       families_[0].family};
  iree_hal_memory_scope_t scope;
  IREE_ASSERT_OK(
      iree_hal_device_group_resolve_memory_scope(group_, site, &scope));
  iree_hal_buffer_native_binding_slot_t slot;
  IREE_ASSERT_OK(iree_hal_pool_resolve_binding(
      source, scope, IREE_HAL_BUFFER_INTERFACE_HOST, &slot));
  EXPECT_EQ(iree_hal_pool_query_host_access(source).access,
            IREE_HAL_MEMORY_ACCESS_NONE);
  iree_hal_pool_reservation_request_t request = {};
  request.allocation_size = 32;
  Ref<iree_hal_buffer_t> buffer;
  SemaphoreList allocated(devices_[1], {0}, {1});
  IREE_ASSERT_OK(
      iree_hal_queue_alloca(queues_[1], iree_hal_semaphore_list_empty(),
                            allocated, source, 1, &request, buffer.out()));
  Ref<iree_hal_buffer_t> subspan;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(
      buffer, 8, sizeof(uint32_t), iree_allocator_system(), subspan.out()));
  const uint32_t value = 0x1234ABCD;
  SemaphoreList filled(devices_[0], {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_fill(queues_[0], allocated, filled, subspan, 0,
                                     sizeof(value), &value, sizeof(value),
                                     IREE_HAL_FILL_FLAG_NONE));
  Wait(filled);
  uint8_t* address = iree_hal_buffer_native_binding(subspan, slot).host_pointer;
  ASSERT_NE(address, nullptr);
  uint32_t actual = 0;
  std::memcpy(&actual, address, sizeof(actual));
  EXPECT_EQ(actual, value);
  const uint32_t updated = value + 1;
  std::memcpy(address, &updated, sizeof(updated));
  SemaphoreList downloaded(devices_[1], {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_download(queues_[1], filled, downloaded,
                                         subspan, 0, &actual, sizeof(actual)));
  Wait(downloaded);
  EXPECT_EQ(actual, updated);
  iree_hal_buffer_mapping_t mapping = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_buffer_map_range(
          subspan, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
          IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, sizeof(value), &mapping));
  subspan.reset();
  SemaphoreList deallocated(devices_[0], {0}, {1});
  auto* raw_buffer = buffer.get();
  IREE_ASSERT_OK(iree_hal_queue_dealloca(queues_[0], downloaded, deallocated, 1,
                                         &raw_buffer));
  Wait(deallocated);
  buffer.reset();
  JoinMaintenance(source);
}

TEST_P(AmdgpuSlabPoolTest, HostOnlySourceDoesNotGrantQueueAccess) {
  iree_hal_pool_scope_t scope = {};
  scope.host.access = IREE_HAL_MEMORY_ACCESS_ALL;
  scope.host.modes = IREE_HAL_MAPPING_MODE_SCOPED;
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(CreateSource(scope, source.out()));
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      source, {}, sizeof(uint32_t), iree_infinite_timeout(), buffer.out()));
  const uint32_t value = 42;
  iree_hal_buffer_mapping_t mapping = {};
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, scope.host.modes, scope.host.access,
      IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, sizeof(value), &mapping));
  std::memcpy(mapping.contents.data, &value, sizeof(value));
  const iree_hal_memory_site_t site = {IREE_HAL_MEMORY_SITE_HOST, nullptr};
  iree_hal_memory_scope_t host_scope;
  IREE_ASSERT_OK(
      iree_hal_device_group_resolve_memory_scope(group_, site, &host_scope));
  iree_hal_buffer_native_binding_slot_t slot;
  IREE_ASSERT_OK(iree_hal_pool_resolve_binding(
      source, host_scope, IREE_HAL_BUFFER_INTERFACE_HOST, &slot));
  uint32_t actual = 0;
  std::memcpy(&actual,
              iree_hal_buffer_native_binding(buffer, slot).host_pointer,
              sizeof(actual));
  EXPECT_EQ(actual, value);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_queue_fill(queues_[0], iree_hal_semaphore_list_empty(),
                          iree_hal_semaphore_list_empty(), buffer, 0,
                          sizeof(value), &value, sizeof(value),
                          IREE_HAL_FILL_FLAG_NONE));
  iree_hal_pool_reservation_request_t request = {};
  request.allocation_size = sizeof(value);
  Ref<iree_hal_buffer_t> queued;
  SemaphoreList allocated(devices_[0], {0}, {1});
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_queue_alloca(queues_[0], iree_hal_semaphore_list_empty(),
                            allocated, source, 1, &request, queued.out()));
  buffer.reset();
  JoinMaintenance(source);
}

TEST_P(AmdgpuSlabPoolTest, PlacementAndFamilyOrderAreIndependent) {
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(CreateSource(Scope(), source.out()));
  std::reverse(families_.begin(), families_.end());
  Ref<iree_hal_pool_t> reversed;
  IREE_ASSERT_OK(CreateSource(Scope(), reversed.out()));
  EXPECT_EQ(iree_hal_pool_notification(source),
            iree_hal_pool_notification(reversed));
  iree_hal_pool_capabilities_t original, capabilities;
  iree_hal_pool_query_capabilities(source, &original);
  iree_hal_pool_query_capabilities(reversed, &capabilities);
  EXPECT_EQ(capabilities.memory_type, original.memory_type);
  EXPECT_EQ(capabilities.placement.mode, IREE_HAL_POOL_PLACEMENT_AUTOMATIC);
  iree_hal_slab_pool_options_t options;
  iree_hal_slab_pool_options_initialize(&options);
  options.placement.mode = IREE_HAL_POOL_PLACEMENT_REQUIRED;
  options.placement.node = 0;
  Ref<iree_hal_pool_t> preferred;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_hal_slab_pool_create(group_, Scope(), &options,
                                iree_allocator_system(), preferred.out()));
  options.placement.mode = IREE_HAL_POOL_PLACEMENT_PREFERRED;
  options.preferences.host.access = IREE_HAL_MEMORY_ACCESS_READ;
  options.preferences.host.modes = IREE_HAL_MAPPING_MODE_SCOPED;
  IREE_ASSERT_OK(iree_hal_slab_pool_create(
      group_, Scope(), &options, iree_allocator_system(), preferred.out()));
  iree_hal_pool_query_capabilities(preferred, &capabilities);
  EXPECT_EQ(capabilities.placement.mode, IREE_HAL_POOL_PLACEMENT_AUTOMATIC);
  EXPECT_EQ(iree_hal_pool_query_host_access(preferred).access,
            IREE_HAL_MEMORY_ACCESS_READ);
}

TEST_P(AmdgpuSlabPoolTest, SharedCacheFeedsBothPoliciesAcrossNativeOwners) {
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(CreateSource(Scope(), source.out()));
  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(source, &capabilities);
  iree_hal_slab_cache_options_t cache_options;
  iree_hal_slab_cache_options_initialize(&cache_options);
  cache_options.slab.allocation_size = 65536;
  cache_options.slab.params.min_alignment =
      capabilities.max_allocation_alignment;
  Ref<iree_hal_pool_t> cache;
  IREE_ASSERT_OK(iree_hal_slab_cache_create(
      source, &cache_options, iree_allocator_system(), cache.out()));
  std::array<Ref<iree_hal_pool_t>, 2> children;
  iree_hal_tlsf_pool_options_t tlsf_options = {};
  tlsf_options.tlsf_options.range_length = 4096;
  tlsf_options.tlsf_options.alignment = IREE_HAL_MEMORY_TLSF_MIN_ALIGNMENT;
  tlsf_options.tlsf_options.frontier_capacity = 4;
  IREE_ASSERT_OK(iree_hal_tlsf_pool_create(
      cache, &tlsf_options, iree_allocator_system(), children[0].out()));
  iree_hal_fixed_block_pool_options_t block_options = {};
  block_options.block_size = 256;
  block_options.blocks_per_slab = 16;
  block_options.frontier_capacity = 4;
  IREE_ASSERT_OK(iree_hal_fixed_block_pool_create(
      cache, &block_options, iree_allocator_system(), children[1].out()));
  const iree_hal_buffer_backing_facts_t* first_backing = nullptr;
  for (auto& child : children) {
    for (uint32_t cycle = 0; cycle < 3; ++cycle) {
      const size_t allocating_device = cycle % 2;
      const size_t deallocating_device = 1 - allocating_device;
      iree_hal_pool_reservation_request_t request = {};
      request.allocation_size = 64;
      Ref<iree_hal_buffer_t> buffer;
      SemaphoreList allocated(devices_[allocating_device], {0}, {1});
      IREE_ASSERT_OK(iree_hal_queue_alloca(
          queues_[allocating_device], iree_hal_semaphore_list_empty(),
          allocated, child, 1, &request, buffer.out()));
      const std::array<uint32_t, 16> input = {11, 22, 1,  2,  30, 400, 77, 88,
                                              33, 44, 99, 99, 99, 99,  55, 66};
      SemaphoreList uploaded(devices_[0], {0}, {1});
      IREE_ASSERT_OK(iree_hal_queue_upload(queues_[0], allocated, uploaded,
                                           input.data(), buffer, 0,
                                           sizeof(input)));
      const iree_hal_buffer_ref_t refs[] = {
          iree_hal_make_buffer_ref(buffer, 8, 16),
          iree_hal_make_buffer_ref(buffer, 40, 16),
      };
      const uint32_t constants[] = {3, cycle + 10};
      SemaphoreList executed(devices_[1], {0}, {1});
      IREE_ASSERT_OK(iree_hal_queue_dispatch(
          queues_[1], uploaded, executed, executable_,
          iree_hal_executable_function_from_index(0),
          iree_hal_make_static_dispatch_config(1, 1, 1),
          iree_make_const_byte_span(constants, sizeof(constants)),
          {IREE_ARRAYSIZE(refs), refs}, IREE_HAL_DISPATCH_FLAG_NONE));
      std::array<uint32_t, 16> output = {};
      SemaphoreList downloaded(devices_[0], {0}, {1});
      IREE_ASSERT_OK(iree_hal_queue_download(queues_[0], executed, downloaded,
                                             buffer, 0, output.data(),
                                             sizeof(output)));
      Wait(downloaded);
      auto expected = input;
      for (size_t i = 0; i < 4; ++i) {
        expected[10 + i] = input[2 + i] * 3 + cycle + 10;
      }
      EXPECT_EQ(output, expected);
      const auto memory = iree_hal_buffer_memory_view(buffer);
      ASSERT_NE(memory.backing, nullptr);
      if (!first_backing) {
        first_backing = memory.backing;
      }
      EXPECT_EQ(memory.backing, first_backing);
      iree_hal_buffer_mapping_t mapping = {};
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_PERMISSION_DENIED,
          iree_hal_buffer_map_range(
              buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
              IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, sizeof(output), &mapping));
      SemaphoreList deallocated(devices_[deallocating_device], {0}, {1});
      auto* raw_buffer = buffer.get();
      IREE_ASSERT_OK(iree_hal_queue_dealloca(queues_[deallocating_device],
                                             downloaded, deallocated, 1,
                                             &raw_buffer));
      Wait(deallocated);
      buffer.reset();
      JoinMaintenance(source);
      iree_hal_pool_stats_t stats;
      iree_hal_pool_query_stats(child, &stats);
      EXPECT_EQ(stats.bytes_committed, 0u);
    }
  }
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(source, &stats);
  EXPECT_EQ(stats.reserve_count, 1u);
  iree_hal_slab_cache_stats_t cache_stats;
  IREE_ASSERT_OK(iree_hal_slab_cache_query_stats(cache, &cache_stats));
  EXPECT_EQ(cache_stats.hit_count, 5u);
  // The returned entry can still carry the completion thread's unpublished
  // frontier even though the deallocation semaphore has reached its value.
  EXPECT_EQ(cache_stats.ready_count + cache_stats.pending_count, 1u);
  iree_hal_pool_trim(cache, IREE_HAL_POOL_TRIM_FLAG_ALL, 0);
  JoinMaintenance(source);
  iree_hal_pool_query_stats(cache, &stats);
  EXPECT_EQ(stats.bytes_committed, 0u);
  iree_hal_pool_query_stats(source, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  EXPECT_EQ(stats.bytes_reserved, 0u);
  IREE_ASSERT_OK(iree_hal_slab_cache_query_stats(cache, &cache_stats));
  EXPECT_EQ(cache_stats.ready_count, 0u);
  EXPECT_EQ(cache_stats.pending_count, 0u);
}

CTS_REGISTER_EXECUTABLE_TEST_SUITE(AmdgpuSlabPoolTest);

}  // namespace
}  // namespace iree::hal::cts
