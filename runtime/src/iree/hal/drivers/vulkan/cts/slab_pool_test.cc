// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cstring>
#include <future>
#include <vector>

#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/memory/fixed_block_pool.h"
#include "iree/hal/memory/maintenance.h"
#include "iree/hal/memory/slab_cache.h"
#include "iree/hal/memory/tlsf_pool.h"

namespace iree::hal::cts {
namespace {

struct ImportedHostAllocation {
  // Pointer released after the Vulkan attachment is retired.
  void* pointer = nullptr;
  // True when the transport callback observed the expected storage class.
  bool storage_class_matches = false;
  // True when the transport callback observed the original host pointer.
  bool payload_matches = false;
  // True after the external transport release callback runs.
  bool released = false;
};

struct ResourceRefCountObservation {
  // Borrowed resource kept live by the test until observation completes.
  iree_hal_resource_t* resource = nullptr;
  // Result published by the terminal semaphore callback.
  std::promise<uint32_t> result;
};

static void ObserveResourceRefCount(void* user_data,
                                    iree_async_semaphore_timepoint_t* timepoint,
                                    iree_status_t status) {
  (void)timepoint;
  auto* observation = static_cast<ResourceRefCountObservation*>(user_data);
  const uint32_t ref_count =
      iree_atomic_ref_count_load(&observation->resource->ref_count);
  iree_status_free(status);
  observation->result.set_value(ref_count);
}

static void ReleaseImportedHostAllocation(
    void* user_data, iree_hal_external_storage_class_t storage_class,
    iree_hal_external_storage_payload_t payload) {
  auto* allocation = static_cast<ImportedHostAllocation*>(user_data);
  allocation->storage_class_matches = iree_hal_external_storage_class_equal(
      storage_class, iree_hal_make_external_storage_class(
                         IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER));
  allocation->payload_matches = payload.host_pointer == allocation->pointer;
  iree_allocator_free_aligned(iree_allocator_system(), allocation->pointer);
  allocation->pointer = nullptr;
  allocation->released = true;
}

class VulkanSlabPoolTest : public CtsTestBase<> {
 protected:
  void SetUp() override {
    CtsTestBase::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    const auto* queues =
        iree_hal_device_spec_queues(iree_hal_device_spec(device_));
    for (iree_host_size_t i = 0; i < queues->family_count; ++i) {
      if (!queues->families[i].provisioned_queue_count) {
        continue;
      }
      auto* queue = iree_hal_device_queue(
          device_, static_cast<iree_hal_queue_family_ordinal_t>(i), 0);
      if (!queue ||
          !iree_any_bit_set(queues->families[i].role_flags,
                            IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER)) {
        continue;
      }
      queues_.push_back(queue);
      iree_hal_pool_family_access_t access = {};
      access.family = iree_hal_queue_family(queue);
      access.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
      families_.push_back(access);
    }
    ASSERT_FALSE(families_.empty());
  }

  iree_hal_pool_scope_t Scope() const {
    iree_hal_pool_scope_t scope = {};
    scope.family_count = families_.size();
    scope.families = families_.data();
    return scope;
  }

  iree_status_t CreateSource(iree_hal_pool_scope_t scope,
                             iree_hal_pool_t** out_pool) {
    iree_hal_slab_pool_options_t options;
    iree_hal_slab_pool_options_initialize(&options);
    return iree_hal_slab_pool_create(device_group_, scope, &options,
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

  // Borrowed provisioned transfer queues with distinct canonical families.
  std::vector<iree_hal_queue_t*> queues_;
  // Construction-time grants for each queue's exact family.
  std::vector<iree_hal_pool_family_access_t> families_;
};

TEST_P(VulkanSlabPoolTest, CapturesExactFamiliesWithoutAllocatingBacking) {
  const auto all_families = families_;
  families_.resize(1);
  families_[0].usage = IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET;
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(CreateSource(Scope(), source.out()));
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(source, &stats);
  EXPECT_EQ(stats.reserve_count, 0u);
  EXPECT_EQ(stats.bytes_committed, 0u);
  EXPECT_EQ(iree_hal_pool_query_host_access(source).access,
            IREE_HAL_MEMORY_ACCESS_NONE);
  for (size_t i = 0; i < all_families.size(); ++i) {
    for (auto kind :
         {IREE_HAL_MEMORY_SITE_QUEUE, IREE_HAL_MEMORY_SITE_PROGRAM}) {
      const iree_hal_memory_site_t site = {kind, all_families[i].family};
      iree_hal_memory_scope_t scope;
      IREE_ASSERT_OK(iree_hal_device_group_resolve_memory_scope(device_group_,
                                                                site, &scope));
      for (auto interface : {IREE_HAL_BUFFER_INTERFACE_VULKAN_BUFFER,
                             IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS}) {
        iree_hal_buffer_native_binding_slot_t slot;
        if (i == 0) {
          IREE_ASSERT_OK(
              iree_hal_pool_resolve_binding(source, scope, interface, &slot));
          EXPECT_NE(slot.index, IREE_HAL_BUFFER_NATIVE_BINDING_INDEX_NONE);
        } else {
          IREE_EXPECT_STATUS_IS(
              IREE_STATUS_PERMISSION_DENIED,
              iree_hal_pool_resolve_binding(source, scope, interface, &slot));
        }
      }
    }
  }

  Ref<iree_hal_buffer_t> buffer;
  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(source, &capabilities);
  const iree_device_size_t min_alignment =
      iree_min((iree_device_size_t)256, capabilities.max_allocation_alignment);
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      source,
      iree_hal_make_pool_reservation_request(sizeof(uint32_t), min_alignment),
      iree_infinite_timeout(), buffer.out()));
  EXPECT_EQ(iree_hal_buffer_allowed_usage(buffer),
            IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET);
  const uint32_t value = 0xAABBCCDD;
  SemaphoreList filled(device_, {0}, {1});
  for (size_t i = 1; i < queues_.size(); ++i) {
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_PERMISSION_DENIED,
        iree_hal_queue_fill(queues_[i], iree_hal_semaphore_list_empty(), filled,
                            buffer, 0, sizeof(value), &value, sizeof(value),
                            /*barriers=*/NULL, IREE_HAL_FILL_FLAG_NONE));
  }
  IREE_ASSERT_OK(
      iree_hal_queue_fill(queues_[0], iree_hal_semaphore_list_empty(), filled,
                          buffer, 0, sizeof(value), &value, sizeof(value),
                          /*barriers=*/NULL, IREE_HAL_FILL_FLAG_NONE));
  Wait(filled);
  uint32_t result = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_queue_download(queues_[0], filled,
                              iree_hal_semaphore_list_empty(), buffer, 0,
                              &result, sizeof(result), /*barriers=*/NULL));
  buffer.reset();
  JoinMaintenance(source);
}

TEST_P(VulkanSlabPoolTest, QueueReleasesBuffersBeforeTerminalSignal) {
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(CreateSource(Scope(), source.out()));
  for (bool fail_dependency : {false, true}) {
    SCOPED_TRACE(fail_dependency);
    Ref<iree_hal_buffer_t> buffer;
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        source, iree_hal_make_pool_reservation_request(sizeof(uint32_t), 0),
        iree_infinite_timeout(), buffer.out()));

    SemaphoreList ready(device_, {0}, {1});
    SemaphoreList completed(device_, {0}, {1});
    const uint32_t pattern = 0xA5A5A5A5u;
    IREE_ASSERT_OK(iree_hal_queue_fill(
        queues_[0], ready, completed, buffer, 0, sizeof(pattern), &pattern,
        sizeof(pattern), /*barriers=*/NULL, IREE_HAL_FILL_FLAG_NONE));

    ResourceRefCountObservation observation = {&buffer.get()->resource};
    std::future<uint32_t> observed_ref_count = observation.result.get_future();
    iree_async_semaphore_timepoint_t timepoint = {};
    timepoint.callback = ObserveResourceRefCount;
    timepoint.user_data = &observation;
    IREE_ASSERT_OK(iree_async_semaphore_acquire_timepoint(
        reinterpret_cast<iree_async_semaphore_t*>(completed.semaphores[0]), 1,
        &timepoint));
    if (fail_dependency) {
      iree_hal_semaphore_fail(ready.semaphores[0],
                              iree_status_from_code(IREE_STATUS_CANCELLED));
    } else {
      IREE_ASSERT_OK(iree_hal_semaphore_signal(ready.semaphores[0], 1, NULL));
    }

    EXPECT_EQ(observed_ref_count.get(), 1u);
    if (fail_dependency) {
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_CANCELLED,
          iree_hal_semaphore_list_wait(completed, iree_infinite_timeout(),
                                       IREE_ASYNC_WAIT_FLAG_NONE));
    } else {
      Wait(completed);
    }
    EXPECT_EQ(iree_atomic_ref_count_load(&buffer.get()->resource.ref_count),
              1u);
    buffer.reset();
    JoinMaintenance(source);
  }
}

TEST_P(VulkanSlabPoolTest,
       NativeReplayDoesNotRetainStaticBufferCommandBuffers) {
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(CreateSource(Scope(), source.out()));
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      source, iree_hal_make_pool_reservation_request(sizeof(uint32_t), 0),
      iree_infinite_timeout(), buffer.out()));

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      iree_hal_queue_family(queues_[0]), IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, /*binding_capacity=*/0,
      command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  const uint32_t pattern = 0x5A5A5A5Au;
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      command_buffer,
      iree_hal_make_buffer_ref(buffer, /*offset=*/0, sizeof(pattern)), &pattern,
      sizeof(pattern), IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  SemaphoreList ready(device_, {0}, {1});
  SemaphoreList completed(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_execute(
      queues_[0], ready, completed, command_buffer,
      iree_hal_buffer_binding_table_empty(), IREE_HAL_QUEUE_EXECUTE_FLAG_NONE));

  ResourceRefCountObservation observation = {&command_buffer.get()->resource};
  std::future<uint32_t> observed_ref_count = observation.result.get_future();
  iree_async_semaphore_timepoint_t timepoint = {};
  timepoint.callback = ObserveResourceRefCount;
  timepoint.user_data = &observation;
  IREE_ASSERT_OK(iree_async_semaphore_acquire_timepoint(
      reinterpret_cast<iree_async_semaphore_t*>(completed.semaphores[0]), 1,
      &timepoint));
  IREE_ASSERT_OK(iree_hal_semaphore_signal(ready.semaphores[0], 1, NULL));

  EXPECT_EQ(observed_ref_count.get(), 1u);
  Wait(completed);
  command_buffer.reset();
  EXPECT_EQ(iree_atomic_ref_count_load(&buffer.get()->resource.ref_count), 1u);
  buffer.reset();
  JoinMaintenance(source);
}

TEST_P(VulkanSlabPoolTest, RequiredAndPreferredHostGrants) {
  for (auto access : {IREE_HAL_MEMORY_ACCESS_READ, IREE_HAL_MEMORY_ACCESS_WRITE,
                      IREE_HAL_MEMORY_ACCESS_ALL}) {
    for (auto modes :
         {IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MAPPING_MODE_PERSISTENT}) {
      SCOPED_TRACE(::testing::Message()
                   << "access=" << access << " modes=" << modes);
      auto scope = Scope();
      scope.host.access = access;
      scope.host.modes = modes;
      Ref<iree_hal_pool_t> source;
      IREE_ASSERT_OK(CreateSource(scope, source.out()));
      const auto host = iree_hal_pool_query_host_access(source);
      EXPECT_EQ(host.access, access);
      EXPECT_EQ(host.modes, modes);
      iree_hal_pool_capabilities_t capabilities;
      iree_hal_pool_query_capabilities(source, &capabilities);
      ASSERT_GE(capabilities.max_allocation_alignment, 256u);
      Ref<iree_hal_buffer_t> buffer;
      IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
          source, iree_hal_make_pool_reservation_request(sizeof(uint32_t), 256),
          iree_infinite_timeout(), buffer.out()));
      const auto memory = iree_hal_buffer_memory_view(buffer);
      ASSERT_NE(memory.backing, nullptr);
      EXPECT_GE(memory.backing->allocation_alignment, 256u);
      const uint32_t value = 0x1234ABCD;
      SemaphoreList filled(device_, {0}, {1});
      IREE_ASSERT_OK(iree_hal_queue_fill(
          queues_[0], iree_hal_semaphore_list_empty(), filled, buffer, 0,
          sizeof(value), &value, sizeof(value), /*barriers=*/NULL,
          IREE_HAL_FILL_FLAG_NONE));
      Wait(filled);
      iree_hal_buffer_mapping_t mapping = {};
      for (auto denied :
           {IREE_HAL_MEMORY_ACCESS_READ, IREE_HAL_MEMORY_ACCESS_WRITE}) {
        if (iree_any_bit_set(access, denied)) {
          continue;
        }
        IREE_EXPECT_STATUS_IS(
            IREE_STATUS_PERMISSION_DENIED,
            iree_hal_buffer_map_range(buffer, modes, denied,
                                      IREE_HAL_BUFFER_MAP_FLAG_NONE, 0,
                                      sizeof(value), &mapping));
      }
      IREE_ASSERT_OK(iree_hal_buffer_map_range(buffer, modes, access,
                                               IREE_HAL_BUFFER_MAP_FLAG_NONE, 0,
                                               sizeof(value), &mapping));
      if (iree_any_bit_set(access, IREE_HAL_MEMORY_ACCESS_READ)) {
        IREE_ASSERT_OK(iree_hal_buffer_mapping_invalidate_range(&mapping, 0,
                                                                sizeof(value)));
        uint32_t result = 0;
        std::memcpy(&result, mapping.contents.data, sizeof(result));
        EXPECT_EQ(result, value);
      }
      if (iree_any_bit_set(access, IREE_HAL_MEMORY_ACCESS_WRITE)) {
        std::memcpy(mapping.contents.data, &value, sizeof(value));
        IREE_ASSERT_OK(
            iree_hal_buffer_mapping_flush_range(&mapping, 0, sizeof(value)));
      }
      IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
      buffer.reset();
      JoinMaintenance(source);
    }
  }

  iree_hal_slab_pool_options_t options;
  iree_hal_slab_pool_options_initialize(&options);
  options.preferences.host.access = IREE_HAL_MEMORY_ACCESS_READ;
  options.preferences.host.modes = IREE_HAL_MAPPING_MODE_SCOPED;
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(iree_hal_slab_pool_create(
      device_group_, Scope(), &options, iree_allocator_system(), source.out()));
  EXPECT_EQ(iree_hal_pool_query_host_access(source).access,
            IREE_HAL_MEMORY_ACCESS_READ);
}

TEST_P(VulkanSlabPoolTest, CachedHostAccessIsAnExactNativeClass) {
  iree_hal_slab_pool_options_t options;
  iree_hal_slab_pool_options_initialize(&options);
  options.preferences.host.access = IREE_HAL_MEMORY_ACCESS_READ;
  options.preferences.host.modes = IREE_HAL_MAPPING_MODE_SCOPED;
  options.preferences.host.cacheability = IREE_HAL_HOST_CACHEABILITY_WRITE_BACK;
  Ref<iree_hal_pool_t> preferred;
  IREE_ASSERT_OK(iree_hal_slab_pool_create(device_group_, Scope(), &options,
                                           iree_allocator_system(),
                                           preferred.out()));
  const auto host = iree_hal_pool_query_host_access(preferred);
  auto scope = Scope();
  scope.host = options.preferences.host;
  Ref<iree_hal_pool_t> required;
  if (!host.access) {
    // A native owner without cached host memory cannot promise that class.
    IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                          CreateSource(scope, required.out()));
    EXPECT_FALSE(required);
    return;
  }
  EXPECT_EQ(host.cacheability, IREE_HAL_HOST_CACHEABILITY_WRITE_BACK);
  IREE_ASSERT_OK(CreateSource(scope, required.out()));
  EXPECT_EQ(iree_hal_pool_query_host_access(required).cacheability,
            IREE_HAL_HOST_CACHEABILITY_WRITE_BACK);
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      required, iree_hal_make_pool_reservation_request(sizeof(uint32_t), 0),
      iree_infinite_timeout(), buffer.out()));
  const uint32_t value = 0xABCD1234;
  SemaphoreList filled(device_, {0}, {1});
  IREE_ASSERT_OK(
      iree_hal_queue_fill(queues_[0], iree_hal_semaphore_list_empty(), filled,
                          buffer, 0, sizeof(value), &value, sizeof(value),
                          /*barriers=*/NULL, IREE_HAL_FILL_FLAG_NONE));
  Wait(filled);
  iree_hal_buffer_mapping_t mapping = {};
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
      IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, sizeof(value), &mapping));
  IREE_ASSERT_OK(
      iree_hal_buffer_mapping_invalidate_range(&mapping, 0, sizeof(value)));
  uint32_t result = 0;
  std::memcpy(&result, mapping.contents.data, sizeof(result));
  EXPECT_EQ(result, value);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
  buffer.reset();
  JoinMaintenance(required);
}

TEST_P(VulkanSlabPoolTest, ImportsHostStorageThroughSelectedNativeRoute) {
  const iree_hal_pool_import_class_t import_class = {
      /*.storage_class=*/iree_hal_make_external_storage_class(
          IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER),
  };
  iree_hal_slab_pool_options_t options;
  iree_hal_slab_pool_options_initialize(&options);
  options.owned.mode = IREE_HAL_SLAB_POOL_IMPORT_ONLY;
  options.imports.class_count = 1;
  options.imports.classes = &import_class;
  auto scope = Scope();
  scope.host.access = IREE_HAL_MEMORY_ACCESS_ALL;
  scope.host.modes = IREE_HAL_MAPPING_MODE_SCOPED;
  Ref<iree_hal_pool_t> source;
  iree_status_t create_status = iree_hal_slab_pool_create(
      device_group_, scope, &options, iree_allocator_system(), source.out());
  if (iree_status_is_unavailable(create_status)) {
    iree::Status unavailable_status(std::move(create_status));
    GTEST_SKIP() << unavailable_status.ToString();
  }
  IREE_ASSERT_OK(create_status);

  iree_hal_pool_import_requirements_t requirements;
  ASSERT_TRUE(iree_hal_pool_query_import_requirements(
      source, import_class.storage_class, &requirements));
  const iree_device_size_t allocation_size = iree_device_align(
      std::max<iree_device_size_t>(requirements.min_byte_length, 4096),
      requirements.byte_length_alignment);
  const iree_device_size_t allocation_alignment =
      std::max<iree_device_size_t>(requirements.base_alignment, 4096);
  ASSERT_LE(allocation_size, IREE_HOST_SIZE_MAX);
  ASSERT_LE(allocation_alignment, IREE_HOST_SIZE_MAX);
  ImportedHostAllocation allocation;
  IREE_ASSERT_OK(iree_allocator_malloc_aligned(
      iree_allocator_system(), static_cast<iree_host_size_t>(allocation_size),
      static_cast<iree_host_size_t>(allocation_alignment), /*offset=*/0,
      &allocation.pointer));
  std::memset(allocation.pointer, 0, static_cast<size_t>(allocation_size));

  iree_hal_external_storage_t storage = {};
  storage.storage_class = import_class.storage_class;
  storage.byte_length = allocation_size;
  storage.payload.host_pointer = allocation.pointer;
  storage.host_cacheability = IREE_HAL_HOST_CACHEABILITY_WRITE_BACK;
  storage.maintenance_range = iree_hal_make_buffer_range(0, allocation_size);
  storage.release = ReleaseImportedHostAllocation;
  storage.release_user_data = &allocation;
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(iree_hal_pool_import_buffer(
      source, &storage, iree_hal_make_buffer_range(0, allocation_size),
      buffer.out()));
  EXPECT_EQ(storage.storage_class.type, IREE_HAL_EXTERNAL_STORAGE_TYPE_NONE);

  const uint32_t value = 0xABCD1234;
  SemaphoreList filled(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_fill(
      queues_.front(), iree_hal_semaphore_list_empty(), filled, buffer, 0,
      sizeof(value), &value, sizeof(value), /*barriers=*/NULL,
      IREE_HAL_FILL_FLAG_NONE));
  Wait(filled);
  iree_hal_buffer_mapping_t mapping = {};
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
      IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, sizeof(value), &mapping));
  IREE_ASSERT_OK(
      iree_hal_buffer_mapping_invalidate_range(&mapping, 0, sizeof(value)));
  uint32_t result = 0;
  std::memcpy(&result, mapping.contents.data, sizeof(result));
  EXPECT_EQ(result, value);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  buffer.reset();
  EXPECT_FALSE(allocation.released);
  JoinMaintenance(source);
  EXPECT_TRUE(allocation.storage_class_matches);
  EXPECT_TRUE(allocation.payload_matches);
  EXPECT_TRUE(allocation.released);
}

TEST_P(VulkanSlabPoolTest, NativeOnlyAndHostOnlyDoNotGrantQueueOperations) {
  for (auto& access : families_) {
    access.usage = IREE_HAL_BUFFER_USAGE_NONE;
  }
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(CreateSource(Scope(), source.out()));
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      source, iree_hal_make_pool_reservation_request(64, 0),
      iree_infinite_timeout(), buffer.out()));
  EXPECT_EQ(iree_hal_buffer_allowed_usage(buffer), IREE_HAL_BUFFER_USAGE_NONE);
  iree_hal_memory_scope_t scope;
  const iree_hal_memory_site_t site = {IREE_HAL_MEMORY_SITE_PROGRAM,
                                       families_[0].family};
  IREE_ASSERT_OK(
      iree_hal_device_group_resolve_memory_scope(device_group_, site, &scope));
  iree_hal_buffer_native_binding_slot_t slot;
  IREE_ASSERT_OK(iree_hal_pool_resolve_binding(
      source, scope, IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS, &slot));
  const auto address =
      iree_hal_buffer_native_binding(buffer, slot).device_address;
  EXPECT_NE(address, 0u);
  const auto memory = iree_hal_buffer_memory_view(buffer);
  ASSERT_NE(memory.backing, nullptr);
  EXPECT_EQ(address % memory.backing->allocation_alignment, 0u);
  const uint32_t value = 42;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_queue_update(queues_[0], iree_hal_semaphore_list_empty(),
                            iree_hal_semaphore_list_empty(), &value, 0, buffer,
                            0, sizeof(value), /*barriers=*/NULL,
                            IREE_HAL_UPDATE_FLAG_NONE));
  buffer.reset();
  JoinMaintenance(source);

  iree_hal_pool_scope_t host_scope = {};
  host_scope.host.access = IREE_HAL_MEMORY_ACCESS_ALL;
  host_scope.host.modes = IREE_HAL_MAPPING_MODE_SCOPED;
  Ref<iree_hal_pool_t> host_source;
  IREE_ASSERT_OK(CreateSource(host_scope, host_source.out()));
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      host_source, iree_hal_make_pool_reservation_request(64, 0),
      iree_infinite_timeout(), buffer.out()));
  iree_hal_buffer_mapping_t mapping = {};
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_ALL,
      IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, sizeof(value), &mapping));
  std::memcpy(mapping.contents.data, &value, sizeof(value));
  uint32_t result = 0;
  std::memcpy(&result, mapping.contents.data, sizeof(result));
  EXPECT_EQ(result, value);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
  SemaphoreList allocated(device_, {0}, {1});
  iree_hal_pool_reservation_request_t request = {};
  request.allocation_size = 64;
  Ref<iree_hal_buffer_t> queued;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_queue_alloca(queues_[0], iree_hal_semaphore_list_empty(),
                            allocated, host_source, 1, &request, queued.out()));
  buffer.reset();
  JoinMaintenance(host_source);
}

TEST_P(VulkanSlabPoolTest, RejectsUnpreparedInterfacesAndNativeProperties) {
  for (auto interface :
       {IREE_HAL_BUFFER_INTERFACE_HOST, IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA,
        IREE_HAL_BUFFER_INTERFACE_XDNA_FIRMWARE, IREE_HAL_BUFFER_INTERFACE_RDMA,
        IREE_HAL_BUFFER_INTERFACE_REGISTERED_IO,
        IREE_HAL_BUFFER_INTERFACE_REMOTE}) {
    families_[0].interfaces = UINT64_C(1) << interface;
    Ref<iree_hal_pool_t> source;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                          CreateSource(Scope(), source.out()));
    EXPECT_FALSE(source);
  }
  families_[0].interfaces = 0;
  for (auto requirement : {IREE_HAL_POOL_ACCESS_REQUIRE_COHERENT_WITH_HOST,
                           IREE_HAL_POOL_ACCESS_REQUIRE_UNCACHED}) {
    families_[0].requirements = requirement;
    auto scope = Scope();
    scope.host.access = IREE_HAL_MEMORY_ACCESS_ALL;
    scope.host.modes = IREE_HAL_MAPPING_MODE_SCOPED;
    Ref<iree_hal_pool_t> source;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                          CreateSource(scope, source.out()));
    EXPECT_FALSE(source);
  }
  families_[0].requirements = 0;
  for (auto cacheability : {IREE_HAL_HOST_CACHEABILITY_WRITE_COMBINED,
                            IREE_HAL_HOST_CACHEABILITY_UNCACHED}) {
    auto scope = Scope();
    scope.host.access = IREE_HAL_MEMORY_ACCESS_WRITE;
    scope.host.modes = IREE_HAL_MAPPING_MODE_SCOPED;
    scope.host.cacheability = cacheability;
    Ref<iree_hal_pool_t> source;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                          CreateSource(scope, source.out()));
    EXPECT_FALSE(source);
    iree_hal_slab_pool_options_t options;
    iree_hal_slab_pool_options_initialize(&options);
    options.preferences.host = scope.host;
    IREE_ASSERT_OK(iree_hal_slab_pool_create(device_group_, Scope(), &options,
                                             iree_allocator_system(),
                                             source.out()));
    EXPECT_EQ(iree_hal_pool_query_host_access(source).access,
              IREE_HAL_MEMORY_ACCESS_NONE);
  }
}

TEST_P(VulkanSlabPoolTest, PlacementAndFamilyOrderAreIndependent) {
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(CreateSource(Scope(), source.out()));
  iree_hal_pool_capabilities_t original;
  iree_hal_pool_query_capabilities(source, &original);
  std::reverse(families_.begin(), families_.end());
  Ref<iree_hal_pool_t> reversed;
  IREE_ASSERT_OK(CreateSource(Scope(), reversed.out()));
  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(reversed, &capabilities);
  EXPECT_EQ(original.placement.mode, capabilities.placement.mode);
  EXPECT_EQ(original.placement.node, capabilities.placement.node);
  EXPECT_EQ(original.memory_type, capabilities.memory_type);
  EXPECT_EQ(iree_hal_pool_notification(source),
            iree_hal_pool_notification(reversed));
  if (original.placement.mode == IREE_HAL_POOL_PLACEMENT_REQUIRED) {
    iree_hal_slab_pool_options_t options;
    iree_hal_slab_pool_options_initialize(&options);
    options.placement = original.placement;
    Ref<iree_hal_pool_t> required;
    IREE_ASSERT_OK(iree_hal_slab_pool_create(device_group_, Scope(), &options,
                                             iree_allocator_system(),
                                             required.out()));
    iree_hal_pool_query_capabilities(required, &capabilities);
    EXPECT_EQ(capabilities.placement.mode, IREE_HAL_POOL_PLACEMENT_REQUIRED);
    EXPECT_EQ(capabilities.placement.node, options.placement.node);
  }
}

TEST_P(VulkanSlabPoolTest, SharedBackingAcrossFamiliesAndAllocationPolicies) {
  if (families_.size() > 1) {
    families_.front().usage = IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET;
    families_.back().usage = IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE;
  }
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(CreateSource(Scope(), source.out()));
  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(source, &capabilities);
  iree_hal_slab_cache_options_t cache_options;
  iree_hal_slab_cache_options_initialize(&cache_options);
  cache_options.slab.allocation_size = 65536;
  cache_options.slab.min_alignment =
      iree_min((iree_device_size_t)4096, capabilities.max_allocation_alignment);
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
      iree_hal_pool_reservation_request_t request = {};
      request.allocation_size = 32;
      Ref<iree_hal_buffer_t> buffer;
      SemaphoreList allocated(device_, {0}, {1});
      IREE_ASSERT_OK(iree_hal_queue_alloca(
          queues_.front(), iree_hal_semaphore_list_empty(), allocated, child, 1,
          &request, buffer.out()));
      const std::array<uint32_t, 8> input = {11, 22, 1, 2, 3, 4, 77, 88};
      SemaphoreList uploaded(device_, {0}, {1});
      IREE_ASSERT_OK(iree_hal_queue_upload(queues_.front(), allocated, uploaded,
                                           input.data(), buffer, 0,
                                           sizeof(input), /*barriers=*/NULL));
      const uint32_t pattern = cycle + 42;
      SemaphoreList filled(device_, {0}, {1});
      IREE_ASSERT_OK(iree_hal_queue_fill(
          queues_.front(), uploaded, filled, buffer, 2 * sizeof(uint32_t),
          4 * sizeof(uint32_t), &pattern, sizeof(pattern), /*barriers=*/NULL,
          IREE_HAL_FILL_FLAG_NONE));
      std::array<uint32_t, 8> output = {};
      SemaphoreList downloaded(device_, {0}, {1});
      IREE_ASSERT_OK(iree_hal_queue_download(
          queues_.back(), filled, downloaded, buffer, 0, output.data(),
          sizeof(output), /*barriers=*/NULL));
      Wait(downloaded);
      EXPECT_THAT(output, ::testing::ElementsAre(11, 22, pattern, pattern,
                                                 pattern, pattern, 77, 88));
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
      SemaphoreList deallocated(device_, {0}, {1});
      auto* raw_buffer = buffer.get();
      IREE_ASSERT_OK(iree_hal_queue_dealloca(queues_.back(), downloaded,
                                             deallocated, 1, &raw_buffer));
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
  EXPECT_EQ(cache_stats.ready_count, 1u);
  iree_hal_pool_trim(cache, IREE_HAL_POOL_TRIM_FLAG_ALL, 0);
  JoinMaintenance(source);
  iree_hal_pool_query_stats(cache, &stats);
  EXPECT_EQ(stats.bytes_committed, 0u);
  iree_hal_pool_query_stats(source, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  EXPECT_EQ(stats.bytes_reserved, 0u);
  IREE_ASSERT_OK(iree_hal_slab_cache_query_stats(cache, &cache_stats));
  EXPECT_EQ(cache_stats.ready_count, 0u);
}

TEST_P(VulkanSlabPoolTest, RejectsScopesCrossingNativeOwners) {
  std::array<DeviceCreateContext, 2> contexts;
  std::array<Ref<iree_hal_device_t>, 2> devices;
  for (size_t i = 0; i < devices.size(); ++i) {
    IREE_ASSERT_OK(contexts[i].Initialize(iree_allocator_system()));
    IREE_ASSERT_OK(iree_hal_driver_create_default_device(
        driver_, contexts[i].params(), iree_allocator_system(),
        devices[i].out()));
  }
  iree_hal_device_group_builder_t builder;
  iree_hal_device_group_builder_initialize(&builder,
                                           contexts[0].frontier_tracker());
  iree_status_t status = iree_ok_status();
  for (auto& device : devices) {
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_group_builder_add_device(&builder, device);
    }
  }
  Ref<iree_hal_device_group_t> group;
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_group_builder_finalize(
        &builder, iree_allocator_system(), group.out());
  }
  iree_hal_device_group_builder_deinitialize(&builder);
  IREE_ASSERT_OK(status);
  std::array<iree_hal_pool_family_access_t, 2> families = {};
  for (size_t i = 0; i < families.size(); ++i) {
    families[i].family = iree_hal_device_queue_family(devices[i], 0);
    families[i].usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  }
  const iree_hal_pool_scope_t scope = {families.size(), families.data(), {}};
  iree_hal_slab_pool_options_t options;
  iree_hal_slab_pool_options_initialize(&options);
  Ref<iree_hal_pool_t> source;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_hal_slab_pool_create(group, scope, &options, iree_allocator_system(),
                                source.out()));
  EXPECT_FALSE(source);
}

CTS_REGISTER_TEST_SUITE(VulkanSlabPoolTest);

}  // namespace
}  // namespace iree::hal::cts
