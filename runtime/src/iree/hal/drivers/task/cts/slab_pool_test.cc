// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

#include "iree/async/notification.h"
#include "iree/hal/cts/util/pool_test_util.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/memory/fixed_block_pool.h"
#include "iree/hal/memory/slab_cache.h"
#include "iree/hal/memory/tlsf_pool.h"

namespace iree::hal::cts {
namespace {

struct ExternalStorageRelease {
  // Joins the release callback's final access before its storage disappears.
  std::mutex mutex;
  // Wakes the external owner after the HAL releases its attachment.
  std::condition_variable condition;
  // Number of completed ownership returns.
  unsigned count = 0;
  // Execution context that returned this borrowed host allocation.
  std::thread::id thread;

  static void Complete(void* user_data,
                       iree_hal_external_storage_class_t storage_class,
                       iree_hal_external_storage_payload_t payload) {
    auto* release = static_cast<ExternalStorageRelease*>(user_data);
    std::lock_guard<std::mutex> lock(release->mutex);
    ++release->count;
    release->thread = std::this_thread::get_id();
    release->condition.notify_all();
  }

  unsigned Query() {
    std::lock_guard<std::mutex> lock(mutex);
    return count;
  }

  unsigned Wait() {
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [&] { return count != 0; });
    return count;
  }
};

class TaskSlabPoolTest : public CtsTestBase<> {
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

    // Distinct devices share a driver/executor but own separate progress
    // services. Their family ordinals are equal; their group sites are not.
    for (size_t i = 0; i < contexts_.size(); ++i) {
      IREE_ASSERT_OK(contexts_[i].Initialize(iree_allocator_system()));
      IREE_ASSERT_OK(iree_hal_driver_create_default_device(
          driver_, contexts_[i].params(), iree_allocator_system(),
          &devices_[i]));
      queues_[i] = iree_hal_device_queue(devices_[i], 0, 0);
      ASSERT_NE(queues_[i], nullptr);
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
    for (size_t i = 0; i < executables_.size(); ++i) {
      IREE_ASSERT_OK(SelectBackendExecutableTarget(
          devices_[i], iree_hal_queue_family(queues_[i]), GetParam(),
          &selection));
      ASSERT_EQ(selection.outcome,
                IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED);
      IREE_ASSERT_OK(LoadExecutable(
          iree_hal_queue_family(queues_[i]), selection.target,
          IREE_HAL_EXECUTABLE_LOAD_FLAG_NONE,
          executable_data(iree_make_cstring_view("elementwise_mul.bin")),
          executables_[i].out()));
    }
    families_[0].family = iree_hal_queue_family(queues_[0]);
    families_[0].usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
    families_[1].family = iree_hal_queue_family(queues_[1]);
    families_[1].usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                         IREE_HAL_BUFFER_USAGE_STORAGE |
                         IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS;
  }

  void TearDown() override {
    for (auto& executable : executables_) {
      executable.reset();
    }
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
    iree_hal_pool_scope_t scope = {
        .family_count = families_.size(),
        .families = families_.data(),
    };
    return scope;
  }

  iree_status_t CreateSource(iree_hal_pool_t** out_pool) {
    iree_hal_slab_pool_options_t options;
    iree_hal_slab_pool_options_initialize(&options);
    return iree_hal_slab_pool_create(group_, Scope(), &options,
                                     iree_allocator_system(), out_pool);
  }

  iree_status_t Wait(iree_hal_semaphore_list_t semaphores) {
    return iree_hal_semaphore_list_wait(semaphores, iree_infinite_timeout(),
                                        IREE_ASYNC_WAIT_FLAG_NONE);
  }

  // Separate device progress services kept alive until their group retires.
  std::array<DeviceCreateContext, 3> contexts_;
  // Owned device references, also retained by group_.
  std::array<iree_hal_device_t*, 3> devices_ = {};
  // Borrowed provisioned queues, all with device-local family ordinal zero.
  std::array<iree_hal_queue_t*, 3> queues_ = {};
  // Owned immutable group, independent of the CTS fixture's cached group.
  iree_hal_device_group_t* group_ = nullptr;
  // Native functions for the transfer-only and dispatch participants.
  std::array<Ref<iree_hal_executable_t>, 2> executables_;
  // Required simultaneous scope; device 2 is deliberately excluded.
  std::array<iree_hal_pool_family_access_t, 2> families_ = {};
};

TEST_P(TaskSlabPoolTest, CapturesExactFamiliesWithoutAllocatingBacking) {
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(CreateSource(source.out()));
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(source, &stats);
  EXPECT_EQ(stats.reserve_count, 0u);
  EXPECT_EQ(stats.bytes_committed, 0u);
  EXPECT_EQ(iree_hal_pool_query_host_access(source).access,
            IREE_HAL_MEMORY_ACCESS_NONE);

  std::array<iree_hal_memory_scope_t, 3> scopes;
  for (size_t i = 0; i < queues_.size(); ++i) {
    const iree_hal_memory_site_t site = {IREE_HAL_MEMORY_SITE_QUEUE,
                                         iree_hal_queue_family(queues_[i])};
    IREE_ASSERT_OK(
        iree_hal_device_group_resolve_memory_scope(group_, site, &scopes[i]));
    EXPECT_EQ(iree_hal_queue_family_ordinal(site.family), 0u);
    EXPECT_EQ(scopes[i].domain, scopes[0].domain);
    if (i) {
      EXPECT_NE(scopes[i].id, scopes[0].id);
    }
    iree_hal_buffer_native_binding_slot_t slot;
    if (i < 2) {
      IREE_ASSERT_OK(iree_hal_pool_resolve_binding(
          source, scopes[i], IREE_HAL_BUFFER_INTERFACE_HOST, &slot));
      EXPECT_EQ(slot.index, 0u);
    } else {
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_PERMISSION_DENIED,
          iree_hal_pool_resolve_binding(source, scopes[i],
                                        IREE_HAL_BUFFER_INTERFACE_HOST, &slot));
    }
  }
  const iree_hal_memory_site_t host_site = {IREE_HAL_MEMORY_SITE_HOST, nullptr};
  iree_hal_memory_scope_t host_scope;
  IREE_ASSERT_OK(iree_hal_device_group_resolve_memory_scope(group_, host_site,
                                                            &host_scope));
  iree_hal_buffer_native_binding_slot_t slot;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_pool_resolve_binding(source, host_scope,
                                    IREE_HAL_BUFFER_INTERFACE_HOST, &slot));
  const iree_hal_memory_site_t foreign_site = {
      IREE_HAL_MEMORY_SITE_QUEUE, iree_hal_device_queue_family(device_, 0)};
  iree_hal_memory_scope_t foreign_scope;
  IREE_ASSERT_OK(iree_hal_device_group_resolve_memory_scope(
      device_group_, foreign_site, &foreign_scope));
  EXPECT_EQ(foreign_scope.id, scopes[0].id);
  EXPECT_NE(foreign_scope.domain, scopes[0].domain);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_pool_resolve_binding(source, foreign_scope,
                                    IREE_HAL_BUFFER_INTERFACE_HOST, &slot));
}

TEST_P(TaskSlabPoolTest, QualifiesRequiredAtomicCapabilities) {
  const iree_hal_atomic_operation_capabilities_t required =
      iree_hal_atomic_operation_capabilities_for_host(
          IREE_HAL_ATOMIC_OPERATION_FLAG_STORE |
          IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_ADD);
  families_[1].atomic_operations = required;
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(CreateSource(source.out()));
  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(source, &capabilities);
  EXPECT_TRUE(iree_hal_atomic_operation_capabilities_support(
      capabilities.atomic_operations, required));
}

TEST_P(TaskSlabPoolTest, RejectsInvalidAtomicRequirements) {
  families_[0].atomic_operations.device_scope_32 = UINT32_C(1) << 31;
  Ref<iree_hal_pool_t> source;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        CreateSource(source.out()));
}

TEST_P(TaskSlabPoolTest, DerivesPublicAccessFromDeclaredUses) {
  families_[0].usage = IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE;
  families_[1].usage = IREE_HAL_BUFFER_USAGE_STORAGE_READ;
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(CreateSource(source.out()));

  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(source, &capabilities);
  EXPECT_EQ(capabilities.allowed_access, IREE_HAL_MEMORY_ACCESS_READ);
  EXPECT_EQ(capabilities.supported_usage,
            IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE |
                IREE_HAL_BUFFER_USAGE_STORAGE_READ);

  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      source,
      iree_hal_make_pool_reservation_request(
          /*allocation_size=*/64, /*min_alignment=*/1),
      iree_infinite_timeout(), buffer.out()));
  EXPECT_EQ(iree_hal_buffer_allowed_access(buffer),
            IREE_HAL_MEMORY_ACCESS_READ);
  EXPECT_EQ(iree_hal_buffer_allowed_usage(buffer),
            capabilities.supported_usage);
}

TEST_P(TaskSlabPoolTest, RejectsIncompleteConstructionRoutes) {
  iree_hal_slab_pool_options_t options;
  iree_hal_slab_pool_options_initialize(&options);
  Ref<iree_hal_pool_t> source;
  families_[1].family = families_[0].family;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_slab_pool_create(group_, Scope(), &options,
                                iree_allocator_system(), source.out()));
  EXPECT_FALSE(source);
  families_[1].family = iree_hal_device_queue_family(device_, 0);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_slab_pool_create(group_, Scope(), &options,
                                iree_allocator_system(), source.out()));
  families_[1].family = iree_hal_queue_family(queues_[1]);
  families_[1].interfaces = UINT64_C(1)
                            << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_hal_slab_pool_create(group_, Scope(), &options,
                                iree_allocator_system(), source.out()));
  families_[1].interfaces = 0;
  options.placement.mode = IREE_HAL_POOL_PLACEMENT_REQUIRED;
  options.placement.node = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_hal_slab_pool_create(group_, Scope(), &options,
                                iree_allocator_system(), source.out()));
  EXPECT_FALSE(source);
}

TEST_P(TaskSlabPoolTest, ReportsAchievedPlacementIndependentlyOfFamilyOrder) {
  Ref<iree_hal_pool_t> original;
  IREE_ASSERT_OK(CreateSource(original.out()));
  std::swap(families_[0], families_[1]);
  iree_hal_slab_pool_options_t options;
  iree_hal_slab_pool_options_initialize(&options);
  options.placement.mode = IREE_HAL_POOL_PLACEMENT_PREFERRED;
  options.placement.node = 0;
  options.preferences.host.access = IREE_HAL_MEMORY_ACCESS_READ;
  options.preferences.host.modes = IREE_HAL_MAPPING_MODE_SCOPED;
  Ref<iree_hal_pool_t> reversed;
  IREE_ASSERT_OK(iree_hal_slab_pool_create(
      group_, Scope(), &options, iree_allocator_system(), reversed.out()));
  EXPECT_EQ(iree_hal_pool_notification(original),
            iree_hal_pool_notification(reversed));
  EXPECT_EQ(iree_hal_pool_query_host_access(reversed).access,
            IREE_HAL_MEMORY_ACCESS_READ);
  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(reversed, &capabilities);
  // Affinity of the malloc caller does not bind recycled physical pages.
  EXPECT_EQ(capabilities.placement.mode, IREE_HAL_POOL_PLACEMENT_AUTOMATIC);
  Ref<iree_hal_pool_t> child;
  iree_hal_tlsf_pool_options_t child_options = {};
  child_options.tlsf_options.range_length = 4096;
  child_options.tlsf_options.alignment = IREE_HAL_MEMORY_TLSF_MIN_ALIGNMENT;
  child_options.tlsf_options.frontier_capacity = 4;
  IREE_ASSERT_OK(iree_hal_tlsf_pool_create(
      reversed, &child_options, iree_allocator_system(), child.out()));
  iree_hal_pool_query_capabilities(child, &capabilities);
  EXPECT_EQ(capabilities.placement.mode, IREE_HAL_POOL_PLACEMENT_AUTOMATIC);
  EXPECT_EQ(iree_hal_pool_query_host_access(child).access,
            IREE_HAL_MEMORY_ACCESS_READ);
}

TEST_P(TaskSlabPoolTest, ImportsFiniteExternalRangeWithExactOwnership) {
  const iree_hal_external_storage_class_t host_pointer_class =
      iree_hal_make_external_storage_class(
          IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER);
  const iree_hal_pool_import_class_t import_class = {
      .storage_class = host_pointer_class,
      .exports =
          {
              .class_count = 1,
              .classes = &host_pointer_class,
          },
  };
  iree_hal_slab_pool_options_t options;
  iree_hal_slab_pool_options_initialize(&options);
  options.imports.class_count = 1;
  options.imports.classes = &import_class;
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(iree_hal_slab_pool_create(
      group_, Scope(), &options, iree_allocator_system(), source.out()));
  iree_hal_pool_import_requirements_t import_requirements;
  EXPECT_TRUE(iree_hal_pool_query_import_requirements(
      source, host_pointer_class, &import_requirements));
  EXPECT_EQ(import_requirements.base_alignment, 1u);
  EXPECT_EQ(import_requirements.byte_length_granularity, 1u);
  EXPECT_FALSE(iree_hal_pool_query_import_requirements(
      source,
      iree_hal_make_external_storage_class(
          IREE_HAL_EXTERNAL_STORAGE_TYPE_DEVICE_ADDRESS),
      &import_requirements));

  alignas(64) std::array<uint8_t, 64> storage_bytes = {};
  ExternalStorageRelease release;
  iree_hal_external_storage_t storage = {
      .storage_class = host_pointer_class,
      .payload = {.host_pointer = storage_bytes.data()},
      .byte_length = storage_bytes.size(),
      .host_cacheability = IREE_HAL_HOST_CACHEABILITY_WRITE_BACK,
      .maintenance_range = iree_hal_make_buffer_range(0, storage_bytes.size()),
      .physical_backing_id = {{0x1234, 0x5678}},
      .release = ExternalStorageRelease::Complete,
      .release_user_data = &release,
  };
  Ref<iree_hal_buffer_t> imported;
  IREE_ASSERT_OK(iree_hal_pool_import_buffer(
      source, &storage, iree_hal_make_buffer_range(8, 24), imported.out()));
  EXPECT_EQ(storage.storage_class.type, IREE_HAL_EXTERNAL_STORAGE_TYPE_NONE);
  EXPECT_EQ(iree_hal_buffer_byte_length(imported), 24u);
  const iree_hal_buffer_memory_view_t memory =
      iree_hal_buffer_memory_view(imported);
  ASSERT_NE(memory.backing, nullptr);
  EXPECT_EQ(memory.offset, 8u);
  EXPECT_GE(memory.backing->allocation_alignment, 64u);
  EXPECT_EQ(memory.backing->maintenance_range.byte_offset, 0u);
  EXPECT_EQ(memory.backing->maintenance_range.byte_length,
            storage_bytes.size());
  EXPECT_EQ(memory.backing->host_cacheability,
            IREE_HAL_HOST_CACHEABILITY_WRITE_BACK);
  EXPECT_EQ(memory.backing->physical_backing_id.words[0], 0x1234u);
  EXPECT_EQ(memory.backing->physical_backing_id.words[1], 0x5678u);

  Ref<iree_hal_buffer_t> nested;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(
      imported, 4, 8, iree_allocator_system(), nested.out()));
  imported.reset();
  EXPECT_EQ(release.Query(), 0u);
  iree_hal_external_storage_t exported = {};
  iree_hal_buffer_range_t exported_range = {};
  IREE_ASSERT_OK(iree_hal_buffer_export(
      nested, iree_hal_make_buffer_range(0, IREE_HAL_WHOLE_BUFFER),
      host_pointer_class, &exported, &exported_range));
  EXPECT_EQ(exported.payload.host_pointer, storage_bytes.data());
  EXPECT_EQ(exported.byte_length, storage_bytes.size());
  EXPECT_EQ(exported.host_cacheability, IREE_HAL_HOST_CACHEABILITY_WRITE_BACK);
  EXPECT_EQ(exported_range.byte_offset, 12u);
  EXPECT_EQ(exported_range.byte_length, 8u);
  EXPECT_EQ(exported.physical_backing_id.words[0], 0x1234u);
  EXPECT_EQ(exported.physical_backing_id.words[1], 0x5678u);
  iree_hal_external_storage_release(&exported);
  nested.reset();
  EXPECT_EQ(release.Wait(), 1u);
  EXPECT_EQ(release.thread, std::this_thread::get_id());
}

TEST_P(TaskSlabPoolTest, ImportAdmissionIsExplicitAndNativeOnly) {
  alignas(64) std::array<uint8_t, 64> storage_bytes = {};
  const iree_hal_external_storage_class_t host_pointer_class =
      iree_hal_make_external_storage_class(
          IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER);
  auto make_storage = [&]() {
    return iree_hal_external_storage_t{
        .storage_class = host_pointer_class,
        .payload = {.host_pointer = storage_bytes.data()},
        .byte_length = storage_bytes.size(),
        .host_cacheability = IREE_HAL_HOST_CACHEABILITY_WRITE_BACK,
        .maintenance_range =
            iree_hal_make_buffer_range(0, storage_bytes.size()),
    };
  };

  Ref<iree_hal_pool_t> ordinary_source;
  IREE_ASSERT_OK(CreateSource(ordinary_source.out()));
  iree_hal_pool_import_requirements_t import_requirements;
  EXPECT_FALSE(iree_hal_pool_query_import_requirements(
      ordinary_source, host_pointer_class, &import_requirements));
  iree_hal_external_storage_t storage = make_storage();
  Ref<iree_hal_buffer_t> imported;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_hal_pool_import_buffer(
          ordinary_source, &storage,
          iree_hal_make_buffer_range(0, IREE_HAL_WHOLE_BUFFER),
          imported.out()));
  EXPECT_EQ(storage.storage_class.type,
            IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER);
  EXPECT_FALSE(imported);

  const iree_hal_pool_import_class_t import_class = {
      .storage_class = host_pointer_class,
  };
  iree_hal_slab_pool_options_t options;
  iree_hal_slab_pool_options_initialize(&options);
  options.owned.mode = IREE_HAL_SLAB_POOL_IMPORT_ONLY;
  options.imports.class_count = 1;
  options.imports.classes = &import_class;
  Ref<iree_hal_pool_t> import_source;
  IREE_ASSERT_OK(iree_hal_slab_pool_create(
      group_, Scope(), &options, iree_allocator_system(), import_source.out()));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_pool_allocate_buffer(
          import_source, iree_hal_make_pool_reservation_request(16, 0),
          iree_immediate_timeout(), imported.out()));

  storage = make_storage();
  storage.host_cacheability = IREE_HAL_HOST_CACHEABILITY_UNKNOWN;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_pool_import_buffer(
          import_source, &storage,
          iree_hal_make_buffer_range(0, IREE_HAL_WHOLE_BUFFER),
          imported.out()));
  EXPECT_EQ(storage.storage_class.type,
            IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER);
  EXPECT_FALSE(imported);

  iree_hal_slab_cache_options_t cache_options;
  iree_hal_slab_cache_options_initialize(&cache_options);
  cache_options.slab.allocation_size = 4096;
  Ref<iree_hal_pool_t> cache;
  IREE_ASSERT_OK(iree_hal_slab_cache_create(
      import_source, &cache_options, iree_allocator_system(), cache.out()));
  storage = make_storage();
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_hal_pool_import_buffer(
          cache, &storage, iree_hal_make_buffer_range(0, IREE_HAL_WHOLE_BUFFER),
          imported.out()));
  EXPECT_EQ(storage.storage_class.type,
            IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER);
  EXPECT_FALSE(imported);
}

TEST_P(TaskSlabPoolTest, PublicHostGrantsRemainIndependentOfExecution) {
  auto scope = Scope();
  scope.host.access = IREE_HAL_MEMORY_ACCESS_READ;
  scope.host.modes = IREE_HAL_MAPPING_MODE_SCOPED;
  scope.host.cacheability = IREE_HAL_HOST_CACHEABILITY_WRITE_BACK;
  iree_hal_slab_pool_options_t options;
  iree_hal_slab_pool_options_initialize(&options);
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(iree_hal_slab_pool_create(
      group_, scope, &options, iree_allocator_system(), source.out()));
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      source, iree_hal_make_pool_reservation_request(sizeof(uint32_t), 0),
      iree_infinite_timeout(), buffer.out()));
  SemaphoreList filled(devices_[0], {0}, {1});
  const uint32_t value = 0xAABBCCDD;
  IREE_ASSERT_OK(
      iree_hal_queue_fill(queues_[0], iree_hal_semaphore_list_empty(), filled,
                          buffer, 0, sizeof(value), &value, sizeof(value),
                          /*barriers=*/NULL, IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(Wait(filled));
  iree_hal_buffer_mapping_t mapping = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_buffer_map_range(
          buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_WRITE,
          IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, sizeof(value), &mapping));
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
      IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, sizeof(value), &mapping));
  EXPECT_EQ(*reinterpret_cast<const uint32_t*>(mapping.contents.data), value);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
  buffer.reset();
}

TEST_P(TaskSlabPoolTest, NativeOnlyScopeDoesNotInventSemanticPermissions) {
  families_[0].usage = IREE_HAL_BUFFER_USAGE_NONE;
  families_[1].usage = IREE_HAL_BUFFER_USAGE_NONE;
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(CreateSource(source.out()));
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      source, iree_hal_make_pool_reservation_request(sizeof(uint32_t), 0),
      iree_infinite_timeout(), buffer.out()));
  EXPECT_EQ(iree_hal_buffer_allowed_usage(buffer), IREE_HAL_BUFFER_USAGE_NONE);
  uint32_t value = 42;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_queue_update(queues_[0], iree_hal_semaphore_list_empty(),
                            iree_hal_semaphore_list_empty(), &value, 0, buffer,
                            0, sizeof(value), /*barriers=*/NULL,
                            IREE_HAL_UPDATE_FLAG_NONE));
  buffer.reset();
}

TEST_P(TaskSlabPoolTest, InteriorArenasInheritTheCompleteScope) {
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(CreateSource(source.out()));
  Ref<iree_hal_buffer_t> backing;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      source, iree_hal_make_pool_reservation_request(8192, 0),
      iree_infinite_timeout(), backing.out()));
  Ref<iree_hal_pool_t> blocks;
  iree_hal_fixed_block_pool_options_t block_options = {
      .block_size = 1024,
      .frontier_capacity = 4,
  };
  IREE_ASSERT_OK(iree_hal_fixed_block_pool_create_from_buffer(
      backing, 128, 4096, &block_options, iree_allocator_system(),
      blocks.out()));
  Ref<iree_hal_buffer_t> region;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      blocks, iree_hal_make_pool_reservation_request(1024, 0),
      iree_infinite_timeout(), region.out()));
  Ref<iree_hal_pool_t> arena;
  iree_hal_tlsf_pool_options_t options = {};
  options.tlsf_options.alignment = IREE_HAL_MEMORY_TLSF_MIN_ALIGNMENT;
  options.tlsf_options.frontier_capacity = 4;
  IREE_ASSERT_OK(iree_hal_tlsf_pool_create_from_buffer(
      region, 64, 768, &options, iree_allocator_system(), arena.out()));
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      arena, iree_hal_make_pool_reservation_request(sizeof(uint32_t), 0),
      iree_infinite_timeout(), buffer.out()));
  const auto memory = iree_hal_buffer_memory_view(buffer);
  EXPECT_EQ(memory.contract, source.get()->memory_contract);
  EXPECT_EQ(memory.backing, iree_hal_buffer_memory_view(backing).backing);
  EXPECT_GE(memory.offset, 192u);
  EXPECT_EQ(memory.offset, memory.binding_offset);

  // A prepared key works through every policy layer and interior view without
  // a parent traversal. Coherence elides cache work, not the semaphore edges
  // below: the second device still waits for the first device's write.
  const iree_hal_memory_scope_t producer = {
      memory.contract->domain, families_[0].family->memory.queue_scope_id};
  const iree_hal_memory_scope_t consumer = {
      memory.contract->domain, families_[1].family->memory.queue_scope_id};
  iree_hal_memory_transition_pair_t pair;
  IREE_ASSERT_OK(iree_hal_memory_transition_prepare_pair(
      iree_hal_pool_transition_table(source), producer, consumer,
      IREE_HAL_MEMORY_TRANSITION_ACQUIRE, &pair));
  for (auto* pool : {source.get(), blocks.get(), arena.get()}) {
    const auto transition = iree_hal_memory_transition_query(
        iree_hal_pool_transition_table(pool), pair);
    EXPECT_TRUE(iree_hal_memory_effects_is_empty(transition.release));
    EXPECT_TRUE(iree_hal_memory_effects_is_empty(transition.acquire));
  }
  const auto table = iree_hal_buffer_transition_table(buffer);
  const auto transition = iree_hal_memory_transition_query(table, pair);
  EXPECT_TRUE(iree_hal_memory_effects_is_empty(transition.release));
  EXPECT_TRUE(iree_hal_memory_effects_is_empty(transition.acquire));
  const auto info = iree_hal_memory_transition_query_info(table, pair);
  EXPECT_TRUE(iree_all_bits_set(info.flags,
                                IREE_HAL_MEMORY_PAIR_SHARED_BACKING_REACHABLE |
                                    IREE_HAL_MEMORY_PAIR_FIXED_COST_KNOWN));
  EXPECT_EQ(info.estimated_fixed_cost_nanoseconds, 0u);
  const iree_hal_memory_scope_t any = {memory.contract->domain, 0};
  EXPECT_TRUE(iree_hal_memory_effects_is_empty(
      iree_hal_buffer_query_transition(buffer, any, consumer).acquire));
  const iree_hal_memory_scope_t excluded = {
      memory.contract->domain,
      iree_hal_queue_family(queues_[2])->memory.queue_scope_id};
  EXPECT_FALSE(iree_hal_memory_effects_is_supported(
      iree_hal_buffer_query_transition(buffer, excluded, consumer).acquire));

  SemaphoreList filled(devices_[0], {0}, {1});
  const uint32_t value = 0x01234567;
  IREE_ASSERT_OK(iree_hal_queue_update(
      queues_[0], iree_hal_semaphore_list_empty(), filled, &value, 0, buffer, 0,
      sizeof(value), /*barriers=*/NULL, IREE_HAL_UPDATE_FLAG_NONE));
  uint32_t result = 0;
  SemaphoreList downloaded(devices_[1], {0}, {1});
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_queue_download(queues_[2], filled, downloaded, buffer, 0,
                              &result, sizeof(result), /*barriers=*/NULL));
  IREE_ASSERT_OK(iree_hal_queue_download(queues_[1], filled, downloaded, buffer,
                                         0, &result, sizeof(result),
                                         /*barriers=*/NULL));
  IREE_ASSERT_OK(Wait(downloaded));
  EXPECT_EQ(result, value);
  EXPECT_EQ(iree_atomic_ref_count_load(&buffer.get()->resource.ref_count), 1);
  buffer.reset();
  arena.reset();
  region.reset();
  blocks.reset();
  backing.reset();
}

#if IREE_HAL_COMMAND_BUFFER_VALIDATION_ENABLE
TEST_P(TaskSlabPoolTest, RecordedBindingsValidateTheAdmittedFamily) {
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(CreateSource(source.out()));
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      source, iree_hal_make_pool_reservation_request(16, 0),
      iree_infinite_timeout(), buffer.out()));
  for (bool indirect : {false, true}) {
    Ref<iree_hal_command_buffer_t> commands;
    IREE_ASSERT_OK(iree_hal_command_buffer_create(
        families_[0].family, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
        IREE_HAL_COMMAND_CATEGORY_DISPATCH, indirect ? 3 : 0, commands.out()));
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(commands));
    iree_hal_buffer_ref_t refs[3];
    iree_hal_buffer_binding_t entries[3];
    for (uint32_t i = 0; i < IREE_ARRAYSIZE(refs); ++i) {
      refs[i] = indirect ? iree_hal_make_indirect_buffer_ref(i, 0, 16)
                         : iree_hal_make_buffer_ref(buffer, 0, 16);
      entries[i] = {buffer, 0, 16};
    }
    const iree_hal_buffer_ref_list_t bindings = {IREE_ARRAYSIZE(refs), refs};
    iree_status_t status = iree_hal_command_buffer_dispatch(
        commands, executables_[0], iree_hal_executable_function_from_index(0),
        iree_hal_make_static_dispatch_config(1, 1, 1),
        iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE);
    if (indirect) {
      IREE_ASSERT_OK(status);
      IREE_ASSERT_OK(iree_hal_command_buffer_end(commands));
      const iree_hal_buffer_binding_table_t table = {IREE_ARRAYSIZE(entries),
                                                     entries};
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_PERMISSION_DENIED,
          iree_hal_queue_execute(queues_[0], iree_hal_semaphore_list_empty(),
                                 iree_hal_semaphore_list_empty(), commands,
                                 table, IREE_HAL_QUEUE_EXECUTE_FLAG_NONE));
    } else {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED, status);
    }
  }
  buffer.reset();
}
#endif  // IREE_HAL_COMMAND_BUFFER_VALIDATION_ENABLE

enum class DispatchMode {
  kQueue,
  kInlineQueue,
  kDirectRecording,
  kIndirectRecording,
  kIndirectParameters,
};

TEST_P(TaskSlabPoolTest, SharedBackingAcrossDevicesAndAllocationPolicies) {
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(CreateSource(source.out()));
  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(source, &capabilities);
  iree_hal_slab_cache_options_t cache_options;
  iree_hal_slab_cache_options_initialize(&cache_options);
  cache_options.slab.allocation_size = 65536;
  cache_options.slab.min_alignment =
      iree_min(4096, capabilities.max_allocation_alignment);
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
  iree_hal_fixed_block_pool_options_t block_options = {
      .block_size = 256,
      .blocks_per_slab = 16,
      .frontier_capacity = 4,
  };
  IREE_ASSERT_OK(iree_hal_fixed_block_pool_create(
      cache, &block_options, iree_allocator_system(), children[1].out()));

  const iree_hal_buffer_backing_facts_t* first_backing = nullptr;
  for (auto& child : children) {
    for (DispatchMode mode :
         {DispatchMode::kQueue, DispatchMode::kInlineQueue,
          DispatchMode::kDirectRecording, DispatchMode::kIndirectRecording,
          DispatchMode::kIndirectParameters}) {
      SCOPED_TRACE(static_cast<int>(mode));
      const std::array<float, 8> inputs[3] = {
          {11, 22, 1, 2, 3, 4, 77, 88},
          {11, 22, 100, 200, 300, 400, 77, 88},
          {11, 22, -1, -1, -1, -1, 77, 88},
      };
      std::array<Ref<iree_hal_buffer_t>, 4> roots;
      std::array<Ref<iree_hal_buffer_t>, 3> views;
      std::array<iree_hal_pool_reservation_request_t, 4> requests = {};
      for (auto& request : requests) {
        request.allocation_size = sizeof(inputs[0]);
      }
      SemaphoreList allocated(devices_[0], {0}, {1});
      iree_hal_buffer_t* allocated_buffers[4] = {};
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_PERMISSION_DENIED,
          iree_hal_queue_alloca(queues_[2], iree_hal_semaphore_list_empty(),
                                allocated, child, requests.size(),
                                requests.data(), allocated_buffers));
      IREE_ASSERT_OK(iree_hal_queue_alloca(
          queues_[0], iree_hal_semaphore_list_empty(), allocated, child,
          requests.size(), requests.data(), allocated_buffers));
      for (size_t i = 0; i < roots.size(); ++i) {
        roots[i].reset(allocated_buffers[i]);
      }
      // Prepare interior views before waiting for allocation commitment.
      for (size_t i = 0; i < views.size(); ++i) {
        Ref<iree_hal_buffer_t> outer;
        IREE_ASSERT_OK(
            iree_hal_buffer_subspan(roots[i], sizeof(float), 6 * sizeof(float),
                                    iree_allocator_system(), outer.out()));
        IREE_ASSERT_OK(
            iree_hal_buffer_subspan(outer, sizeof(float), 4 * sizeof(float),
                                    iree_allocator_system(), views[i].out()));
        iree_hal_buffer_mapping_t mapping = {};
        IREE_EXPECT_STATUS_IS(
            IREE_STATUS_PERMISSION_DENIED,
            iree_hal_buffer_map_range(views[i], IREE_HAL_MAPPING_MODE_SCOPED,
                                      IREE_HAL_MEMORY_ACCESS_READ,
                                      IREE_HAL_BUFFER_MAP_FLAG_NONE, 0,
                                      IREE_HAL_WHOLE_BUFFER, &mapping));
      }
      SemaphoreList uploaded(devices_[0], {0, 0, 0, 0}, {1, 1, 1, 1});
      for (size_t i = 0; i < views.size(); ++i) {
        const iree_hal_semaphore_list_t done = {1, &uploaded.semaphores[i],
                                                &uploaded.payload_values[i]};
        IREE_ASSERT_OK(iree_hal_queue_upload(
            queues_[0], allocated, done, inputs[i].data(), roots[i], 0,
            sizeof(inputs[i]), /*barriers=*/NULL));
      }
      const std::array<uint32_t, 3> workgroups = {1, 1, 1};
      const iree_hal_semaphore_list_t parameters_uploaded = {
          1, &uploaded.semaphores[3], &uploaded.payload_values[3]};
      IREE_ASSERT_OK(iree_hal_queue_upload(
          queues_[0], allocated, parameters_uploaded, workgroups.data(),
          roots[3], 0, sizeof(workgroups), /*barriers=*/NULL));
      const bool indirect = mode == DispatchMode::kIndirectRecording;
      iree_hal_buffer_ref_t refs[3] = {};
      iree_hal_buffer_binding_t entries[3] = {};
      for (uint32_t i = 0; i < views.size(); ++i) {
        refs[i] =
            indirect
                ? iree_hal_make_indirect_buffer_ref(i, 0, 4 * sizeof(float))
                : iree_hal_make_buffer_ref(views[i], 0, 4 * sizeof(float));
        entries[i] = {views[i], 0, 4 * sizeof(float)};
      }
      const iree_hal_buffer_ref_list_t bindings = {IREE_ARRAYSIZE(refs), refs};
      SemaphoreList executed(devices_[1], {0}, {1});
      if (mode == DispatchMode::kDirectRecording || indirect) {
        Ref<iree_hal_command_buffer_t> commands;
        IREE_ASSERT_OK(iree_hal_command_buffer_create(
            families_[1].family, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
            IREE_HAL_COMMAND_CATEGORY_DISPATCH, indirect ? 3 : 0,
            commands.out()));
        IREE_ASSERT_OK(iree_hal_command_buffer_begin(commands));
        IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
            commands, executables_[1],
            iree_hal_executable_function_from_index(0),
            iree_hal_make_static_dispatch_config(1, 1, 1),
            iree_const_byte_span_empty(), bindings,
            IREE_HAL_DISPATCH_FLAG_NONE));
        IREE_ASSERT_OK(iree_hal_command_buffer_end(commands));
        const iree_hal_buffer_binding_table_t table = {
            indirect ? IREE_ARRAYSIZE(entries) : 0,
            indirect ? entries : nullptr};
        IREE_ASSERT_OK(
            iree_hal_queue_execute(queues_[1], uploaded, executed, commands,
                                   table, IREE_HAL_QUEUE_EXECUTE_FLAG_NONE));
      } else {
        auto config = iree_hal_make_static_dispatch_config(1, 1, 1);
        iree_hal_dispatch_flags_t flags = IREE_HAL_DISPATCH_FLAG_NONE;
        if (mode == DispatchMode::kInlineQueue) {
          flags |= IREE_HAL_DISPATCH_FLAG_ALLOW_INLINE_EXECUTION;
        } else if (mode == DispatchMode::kIndirectParameters) {
          config.workgroup_count_ref =
              iree_hal_make_buffer_ref(roots[3], 0, sizeof(workgroups));
          flags |= IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS;
        }
        IREE_EXPECT_STATUS_IS(
            IREE_STATUS_PERMISSION_DENIED,
            iree_hal_queue_dispatch(queues_[0], uploaded, executed,
                                    executables_[0],
                                    iree_hal_executable_function_from_index(0),
                                    config, iree_const_byte_span_empty(),
                                    bindings, /*barriers=*/NULL, flags));
        IREE_ASSERT_OK(iree_hal_queue_dispatch(
            queues_[1], uploaded, executed, executables_[1],
            iree_hal_executable_function_from_index(0), config,
            iree_const_byte_span_empty(), bindings, /*barriers=*/NULL, flags));
      }
      std::array<float, 8> result = {};
      SemaphoreList downloaded(devices_[0], {0}, {1});
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_PERMISSION_DENIED,
          iree_hal_queue_download(queues_[2], executed, downloaded, roots[2], 0,
                                  result.data(), sizeof(result),
                                  /*barriers=*/NULL));
      IREE_ASSERT_OK(iree_hal_queue_download(
          queues_[0], executed, downloaded, roots[2], 0, result.data(),
          sizeof(result), /*barriers=*/NULL));
      IREE_ASSERT_OK(Wait(downloaded));
      EXPECT_THAT(result,
                  ::testing::ElementsAre(11, 22, 100, 400, 900, 1600, 77, 88));
      for (auto& root : roots) {
        const auto memory = iree_hal_buffer_memory_view(root);
        ASSERT_NE(memory.backing, nullptr);
        if (!first_backing) {
          first_backing = memory.backing;
        }
        EXPECT_EQ(memory.backing, first_backing);
      }
      SemaphoreList deallocated(devices_[1], {0}, {1});
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_PERMISSION_DENIED,
          iree_hal_queue_dealloca(queues_[2], downloaded, deallocated,
                                  roots.size(), allocated_buffers));
      IREE_ASSERT_OK(iree_hal_queue_dealloca(queues_[1], downloaded,
                                             deallocated, roots.size(),
                                             allocated_buffers));
      IREE_ASSERT_OK(Wait(deallocated));
      for (auto& view : views) {
        view.reset();
      }
      for (auto& root : roots) {
        root.reset();
      }
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
  EXPECT_EQ(cache_stats.hit_count, 9u);
  EXPECT_EQ(cache_stats.ready_count, 1u);
}

struct VirtualBufferLifetime {
  // Allocation returned as the native virtual buffer's ordinary resource.
  void* buffer_allocation = nullptr;
  // Final frees observed through the real buffer's supplied host allocator.
  std::atomic<uint32_t> release_count = 0;
};

static iree_status_t ObserveVirtualBufferLifetime(
    void* self, iree_allocator_command_t command, const void* params,
    void** inout_pointer) {
  auto* lifetime = static_cast<VirtualBufferLifetime*>(self);
  if (command == IREE_ALLOCATOR_COMMAND_FREE &&
      *inout_pointer == lifetime->buffer_allocation) {
    ++lifetime->release_count;
  }
  auto allocator = iree_allocator_system();
  return allocator.ctl(allocator.self, command, params, inout_pointer);
}

TEST_P(TaskSlabPoolTest, CapturedViewsKeepAddressesAliveThroughRemoval) {
  constexpr iree_device_size_t kUnit = 64 * 1024;
  for (auto& family : families_) {
    family.interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE;
  }
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(CreateSource(source.out()));
  Ref<iree_hal_buffer_t> backing;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      source, iree_hal_make_pool_reservation_request(kUnit, kUnit),
      iree_infinite_timeout(), backing.out()));
  for (auto& family : families_) {
    family.interfaces = 0;
  }
  VirtualBufferLifetime lifetime;
  const iree_allocator_t allocator = {&lifetime, ObserveVirtualBufferLifetime};
  Ref<iree_hal_buffer_t> arena;
  const iree_hal_virtual_buffer_options_t options = {2 * kUnit, kUnit, kUnit};
  IREE_ASSERT_OK(iree_hal_virtual_buffer_create(group_, Scope(), &options,
                                                allocator, arena.out()));
  lifetime.buffer_allocation = arena.get();
  Ref<iree_hal_buffer_t> view;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(arena, kUnit, kUnit,
                                         iree_allocator_system(), view.out()));
  const iree_hal_memory_scope_t all = {
      iree_hal_device_group_memory_domain(group_), IREE_HAL_MEMORY_SCOPE_ANY};
  iree_hal_memory_mapping_t mapping = {
      view, all, {0, kUnit}, backing, 0, IREE_HAL_MEMORY_ACCESS_ALL};
  SemaphoreList gate(devices_[0], {0}, {1});
  SemaphoreList mapped(devices_[0], {0}, {1});
  SemaphoreList filled(devices_[1], {0}, {1});
  SemaphoreList downloaded(devices_[0], {0}, {1});
  SemaphoreList detached(devices_[1], {0}, {1});
  const uint32_t value = 0x11223344;
  uint32_t observed = 0;
  IREE_EXPECT_OK(iree_hal_queue_map(queues_[0], gate, mapped, 1, &mapping,
                                    IREE_HAL_QUEUE_MAP_FLAG_NONE));
  IREE_EXPECT_OK(iree_hal_queue_fill(queues_[1], mapped, filled, view, 0, kUnit,
                                     &value, sizeof(value), nullptr,
                                     IREE_HAL_FILL_FLAG_NONE));
  IREE_EXPECT_OK(iree_hal_queue_download(queues_[0], filled, downloaded,
                                         backing, 0, &observed,
                                         sizeof(observed), nullptr));
  mapping.source = nullptr;
  mapping.access = IREE_HAL_MEMORY_ACCESS_NONE;
  IREE_EXPECT_OK(iree_hal_queue_map(queues_[1], downloaded, detached, 1,
                                    &mapping, IREE_HAL_QUEUE_MAP_FLAG_NONE));
  view.reset();
  arena.reset();
  EXPECT_EQ(lifetime.release_count.load(), 0u);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate.semaphores[0], 1, nullptr));
  IREE_ASSERT_OK(Wait(detached));
  EXPECT_EQ(observed, value);
  EXPECT_EQ(lifetime.release_count.load(), 1u);
  backing.reset();
}

TEST_P(TaskSlabPoolTest, CapturedPoolViewsKeepTheirRangeUntilMappingCompletes) {
  constexpr iree_device_size_t kUnit = 64 * 1024;
  for (auto& family : families_) {
    family.interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE;
  }
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(CreateSource(source.out()));
  Ref<iree_hal_buffer_t> backing;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      source, iree_hal_make_pool_reservation_request(2 * kUnit, kUnit),
      iree_infinite_timeout(), backing.out()));
  for (auto& family : families_) {
    family.interfaces = 0;
  }
  Ref<iree_hal_buffer_t> arena;
  const iree_hal_virtual_buffer_options_t options = {3 * kUnit, kUnit, kUnit};
  IREE_ASSERT_OK(iree_hal_virtual_buffer_create(
      group_, Scope(), &options, iree_allocator_system(), arena.out()));
  Ref<iree_hal_pool_t> child_pool;
  iree_hal_fixed_block_pool_options_t block_options;
  iree_hal_fixed_block_pool_options_initialize(&block_options);
  block_options.block_size = 2 * kUnit;
  block_options.alignment = kUnit;
  IREE_ASSERT_OK(iree_hal_fixed_block_pool_create_from_buffer(
      arena, kUnit, 2 * kUnit, &block_options, iree_allocator_system(),
      child_pool.out()));
  arena.reset();
  const iree_hal_memory_scope_t all = {
      iree_hal_device_group_memory_domain(group_), IREE_HAL_MEMORY_SCOPE_ANY};
  void* stable_address = nullptr;
  for (uint32_t generation = 0; generation < 2; ++generation) {
    Ref<iree_hal_buffer_t> child, view;
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        child_pool, iree_hal_make_pool_reservation_request(2 * kUnit, kUnit),
        iree_infinite_timeout(), child.out()));
    IREE_ASSERT_OK(iree_hal_buffer_subspan(
        child, kUnit, kUnit, iree_allocator_system(), view.out()));
    iree_hal_virtual_buffer_info_t info;
    IREE_ASSERT_OK(iree_hal_virtual_buffer_query_info(view, &info));
    EXPECT_EQ(info.mapping_byte_offset, 2 * kUnit);
    iree_byte_span_t address;
    IREE_ASSERT_OK(iree_hal_buffer_native_host_span(view, 0, kUnit, &address));
    if (generation == 0) {
      stable_address = address.data;
    }
    EXPECT_EQ(address.data, stable_address);
    iree_hal_memory_mapping_t mapping = {view,
                                         all,
                                         {0, kUnit},
                                         backing,
                                         generation * kUnit,
                                         IREE_HAL_MEMORY_ACCESS_ALL};
    SemaphoreList gate(devices_[0], {0}, {1});
    SemaphoreList mapped(devices_[0], {0}, {1});
    SemaphoreList filled(devices_[1], {0}, {1});
    SemaphoreList downloaded(devices_[0], {0}, {1});
    SemaphoreList detached(devices_[1], {0}, {1});
    const uint32_t value = 0x10203040 + generation;
    uint32_t observed = 0;
    const iree_hal_queue_map_flags_t mapping_flags =
        generation == 0 ? IREE_HAL_QUEUE_MAP_FLAG_NONE
                        : IREE_HAL_QUEUE_MAP_FLAG_BORROW_BUFFERS;
    IREE_EXPECT_OK(iree_hal_queue_map(queues_[0], gate, mapped, 1, &mapping,
                                      mapping_flags));
    IREE_EXPECT_OK(iree_hal_queue_fill(queues_[1], mapped, filled, view, 0,
                                       kUnit, &value, sizeof(value), nullptr,
                                       IREE_HAL_FILL_FLAG_NONE));
    IREE_EXPECT_OK(iree_hal_queue_download(
        queues_[0], filled, downloaded, backing, generation * kUnit, &observed,
        sizeof(observed), nullptr));
    mapping.source = nullptr;
    mapping.source_offset = 0;
    mapping.access = IREE_HAL_MEMORY_ACCESS_NONE;
    IREE_EXPECT_OK(iree_hal_queue_map(queues_[1], downloaded, detached, 1,
                                      &mapping, mapping_flags));
    child.reset();
    if (generation == 0) {
      view.reset();
    }
    iree_hal_pool_stats_t stats;
    iree_hal_pool_query_stats(child_pool, &stats);
    EXPECT_EQ(stats.reservation_count, 1u);
    IREE_ASSERT_OK(iree_hal_semaphore_signal(gate.semaphores[0], 1, nullptr));
    IREE_ASSERT_OK(Wait(detached));
    EXPECT_EQ(observed, value);
    if (generation != 0) {
      iree_hal_pool_query_stats(child_pool, &stats);
      EXPECT_EQ(stats.reservation_count, 1u);
      view.reset();
    }
    iree_hal_pool_query_stats(child_pool, &stats);
    EXPECT_EQ(stats.reservation_count, 0u);
  }
  child_pool.reset();
  backing.reset();
}

TEST_P(TaskSlabPoolTest, MappingTargetsResolveAfterQueuedAllocation) {
  constexpr iree_device_size_t kUnit = 64 * 1024;
  for (auto& family : families_) {
    family.interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE;
  }
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(CreateSource(source.out()));
  Ref<iree_hal_buffer_t> backing;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      source, iree_hal_make_pool_reservation_request(kUnit, kUnit),
      iree_infinite_timeout(), backing.out()));
  for (auto& family : families_) {
    family.interfaces = 0;
  }
  Ref<iree_hal_buffer_t> arena;
  const iree_hal_virtual_buffer_options_t options = {2 * kUnit, kUnit, kUnit};
  IREE_ASSERT_OK(iree_hal_virtual_buffer_create(
      group_, Scope(), &options, iree_allocator_system(), arena.out()));
  Ref<iree_hal_pool_t> child_pool;
  iree_hal_fixed_block_pool_options_t block_options;
  iree_hal_fixed_block_pool_options_initialize(&block_options);
  block_options.block_size = kUnit;
  block_options.alignment = kUnit;
  IREE_ASSERT_OK(iree_hal_fixed_block_pool_create_from_buffer(
      arena, kUnit, kUnit, &block_options, iree_allocator_system(),
      child_pool.out()));
  arena.reset();
  SemaphoreList gate(devices_[0], {0}, {1});
  SemaphoreList allocated(devices_[0], {0}, {1});
  SemaphoreList mapped(devices_[1], {0}, {1});
  SemaphoreList filled(devices_[1], {0}, {1});
  SemaphoreList downloaded(devices_[0], {0}, {1});
  SemaphoreList detached(devices_[1], {0}, {1});
  SemaphoreList deallocated(devices_[0], {0}, {1});
  const auto request = iree_hal_make_pool_reservation_request(kUnit, kUnit);
  Ref<iree_hal_buffer_t> target;
  IREE_EXPECT_OK(iree_hal_queue_alloca(queues_[0], gate, allocated, child_pool,
                                       1, &request, target.out()));
  iree_hal_virtual_buffer_info_t info;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        iree_hal_virtual_buffer_query_info(target, &info));
  const iree_hal_memory_scope_t all = {
      iree_hal_device_group_memory_domain(group_), IREE_HAL_MEMORY_SCOPE_ANY};
  iree_hal_memory_mapping_t mapping = {
      target, all, {0, kUnit}, backing, 0, IREE_HAL_MEMORY_ACCESS_ALL};
  const uint32_t value = 0x87654321;
  uint32_t observed = 0;
  IREE_EXPECT_OK(iree_hal_queue_map(queues_[1], allocated, mapped, 1, &mapping,
                                    IREE_HAL_QUEUE_MAP_FLAG_NONE));
  IREE_EXPECT_OK(iree_hal_queue_fill(queues_[1], mapped, filled, target, 0,
                                     kUnit, &value, sizeof(value), nullptr,
                                     IREE_HAL_FILL_FLAG_NONE));
  IREE_EXPECT_OK(iree_hal_queue_download(queues_[0], filled, downloaded,
                                         backing, 0, &observed,
                                         sizeof(observed), nullptr));
  mapping.source = nullptr;
  mapping.access = IREE_HAL_MEMORY_ACCESS_NONE;
  IREE_EXPECT_OK(iree_hal_queue_map(queues_[1], downloaded, detached, 1,
                                    &mapping, IREE_HAL_QUEUE_MAP_FLAG_NONE));
  iree_hal_buffer_t* target_buffer = target.get();
  IREE_EXPECT_OK(iree_hal_queue_dealloca(queues_[0], detached, deallocated, 1,
                                         &target_buffer));
  target.reset();
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate.semaphores[0], 1, nullptr));
  IREE_ASSERT_OK(Wait(deallocated));
  EXPECT_EQ(observed, value);
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(child_pool, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  child_pool.reset();
  backing.reset();
}

TEST_P(TaskSlabPoolTest, CancelledMappingReleasesCapturedAddressReferences) {
  constexpr iree_device_size_t kUnit = 64 * 1024;
  VirtualBufferLifetime lifetime;
  const iree_allocator_t allocator = {&lifetime, ObserveVirtualBufferLifetime};
  Ref<iree_hal_buffer_t> arena;
  const iree_hal_virtual_buffer_options_t options = {kUnit, kUnit, kUnit};
  IREE_ASSERT_OK(iree_hal_virtual_buffer_create(group_, Scope(), &options,
                                                allocator, arena.out()));
  lifetime.buffer_allocation = arena.get();
  const iree_hal_memory_scope_t all = {
      iree_hal_device_group_memory_domain(group_), IREE_HAL_MEMORY_SCOPE_ANY};
  const iree_hal_memory_mapping_t mapping = {arena, all, {0, kUnit}};
  SemaphoreList gate(devices_[0], {0}, {1});
  SemaphoreList detached(devices_[0], {0}, {1});
  IREE_EXPECT_OK(iree_hal_queue_map(queues_[0], gate, detached, 1, &mapping,
                                    IREE_HAL_QUEUE_MAP_FLAG_NONE));
  arena.reset();
  EXPECT_EQ(lifetime.release_count.load(), 0u);
  iree_hal_semaphore_fail(gate.semaphores[0],
                          iree_status_from_code(IREE_STATUS_ABORTED));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED, Wait(detached));
  EXPECT_EQ(lifetime.release_count.load(), 1u);
}

TEST_P(TaskSlabPoolTest, FailedVirtualBufferCreationPublishesNoOwner) {
  const iree_hal_virtual_buffer_options_t options = {64 * 1024, 0, 0};
  Ref<iree_hal_buffer_t> arena;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_virtual_buffer_create(group_, Scope(), &options,
                                     iree_allocator_null(), arena.out()));
  EXPECT_EQ(arena.get(), nullptr);
}

TEST_P(TaskSlabPoolTest, BatchedMappingOrdersOldAndNewUsersAcrossReservations) {
  iree_hal_virtual_buffer_capabilities_t capabilities;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_query_capabilities(
      group_, Scope(), iree_allocator_system(), &capabilities));
  ASSERT_GT(capabilities.mapping_granularity, 0u);
  const iree_device_size_t kUnit =
      iree_max(64 * 1024, capabilities.reservation_granularity);
  ASSERT_GE(capabilities.max_byte_length, 2 * kUnit);
  ASSERT_GE(capabilities.max_alignment, kUnit);
  EXPECT_EQ(capabilities.operations, IREE_HAL_MEMORY_MAPPING_OPERATIONS_ALL);
  for (auto& family : families_) {
    family.interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE;
  }
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(CreateSource(source.out()));
  Ref<iree_hal_buffer_t> backing;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      source, iree_hal_make_pool_reservation_request(4 * kUnit, kUnit),
      iree_infinite_timeout(), backing.out()));
  const auto original = iree_hal_buffer_memory_view(backing);
  ASSERT_NE(original.backing, nullptr);
  const auto physical_charge = original.backing->native_allocation.byte_length;

  // Nested views preserve native source coordinates, including any native
  // prefix needed to align the slab's original CPU mapping.
  Ref<iree_hal_buffer_t> tail;
  Ref<iree_hal_buffer_t> nested;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(backing, kUnit, 3 * kUnit,
                                         iree_allocator_system(), tail.out()));
  IREE_ASSERT_OK(iree_hal_buffer_subspan(
      tail, kUnit, 2 * kUnit, iree_allocator_system(), nested.out()));
  for (auto& family : families_) {
    family.interfaces = 0;
  }
  const iree_hal_virtual_buffer_options_t options = {2 * kUnit, kUnit, kUnit};
  Ref<iree_hal_buffer_t> first, second;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_create(
      group_, Scope(), &options, iree_allocator_system(), first.out()));
  IREE_ASSERT_OK(iree_hal_virtual_buffer_create(
      group_, Scope(), &options, iree_allocator_system(), second.out()));
  iree_hal_virtual_buffer_info_t info;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_query_info(first, &info));
  EXPECT_EQ(info.mapping_granularity, kUnit);

  // Both pool strategies manage virtual subranges before pages are mapped.
  // They retain metadata and address capacity without allocating physical
  // storage or assuming the mapping replacement unit is a suballocation unit.
  constexpr iree_device_size_t kChildOffset = 256;
  Ref<iree_hal_pool_t> first_pool, second_pool;
  iree_hal_tlsf_pool_options_t tlsf_options = {};
  tlsf_options.tlsf_options.alignment = kChildOffset;
  IREE_ASSERT_OK(iree_hal_tlsf_pool_create_from_buffer(
      first.get(), kChildOffset, 1024, &tlsf_options, iree_allocator_system(),
      first_pool.out()));
  iree_hal_fixed_block_pool_options_t block_options;
  iree_hal_fixed_block_pool_options_initialize(&block_options);
  block_options.block_size = kChildOffset;
  block_options.alignment = kChildOffset;
  IREE_ASSERT_OK(iree_hal_fixed_block_pool_create_from_buffer(
      second.get(), kChildOffset, 1024, &block_options, iree_allocator_system(),
      second_pool.out()));
  Ref<iree_hal_buffer_t> first_child, second_child;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      first_pool, iree_hal_make_pool_reservation_request(16, kChildOffset),
      iree_infinite_timeout(), first_child.out()));
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      second_pool, iree_hal_make_pool_reservation_request(16, kChildOffset),
      iree_infinite_timeout(), second_child.out()));
  for (iree_hal_buffer_t* child : {first_child.get(), second_child.get()}) {
    const auto memory = iree_hal_buffer_memory_view(child);
    EXPECT_EQ(memory.offset, kChildOffset);
    ASSERT_NE(memory.backing, nullptr);
    EXPECT_EQ(memory.backing->native_allocation.byte_length, 0u);
    EXPECT_EQ(memory.backing->allocation_alignment, kUnit);
    EXPECT_EQ(memory.backing->advice, nullptr);
  }
  iree_hal_pool_stats_t source_stats;
  iree_hal_pool_query_stats(source, &source_stats);
  EXPECT_EQ(source_stats.bytes_committed, physical_charge);
  const iree_hal_memory_scope_t all = {
      iree_hal_device_group_memory_domain(group_), IREE_HAL_MEMORY_SCOPE_ANY};

  std::array<float, 4> initial = {2, 3, 5, 7};
  SemaphoreList uploaded(devices_[0], {0, 0, 0, 0}, {1, 1, 1, 1});
  for (size_t i = 0; i < initial.size(); ++i) {
    const iree_hal_semaphore_list_t done = {1, &uploaded.semaphores[i],
                                            &uploaded.payload_values[i]};
    IREE_ASSERT_OK(iree_hal_queue_fill(
        queues_[0], iree_hal_semaphore_list_empty(), done, backing, i * kUnit,
        kUnit, &initial[i], sizeof(float), nullptr, IREE_HAL_FILL_FLAG_NONE));
  }
  std::array<iree_hal_memory_mapping_t, 2> mappings = {{
      {first,
       all,
       {0, 2 * kUnit},
       backing,
       0,
       IREE_HAL_MEMORY_ACCESS_NONE,
       IREE_HAL_MEMORY_MAPPING_OPERATION_BIND},
      {second,
       all,
       {0, 2 * kUnit},
       nested,
       0,
       IREE_HAL_MEMORY_ACCESS_NONE,
       IREE_HAL_MEMORY_MAPPING_OPERATION_BIND},
  }};
  // HIP-style installation and access grants are separate calls. The recorder
  // carries the source into each protection change; HAL owns no mapping ledger.
  SemaphoreList bound(devices_[0], {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_map(queues_[0], uploaded, bound,
                                    mappings.size(), mappings.data(),
                                    IREE_HAL_QUEUE_MAP_FLAG_NONE));
  for (auto& mapping : mappings) {
    mapping.operation = IREE_HAL_MEMORY_MAPPING_OPERATION_PROTECT;
    mapping.access = IREE_HAL_MEMORY_ACCESS_ALL;
  }
  SemaphoreList mapped(devices_[0], {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_map(queues_[0], bound, mapped, mappings.size(),
                                    mappings.data(),
                                    IREE_HAL_QUEUE_MAP_FLAG_NONE));
  IREE_ASSERT_OK(Wait(mapped));

  Ref<iree_hal_buffer_t> output;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      source, iree_hal_make_pool_reservation_request(4 * sizeof(float), 0),
      iree_infinite_timeout(), output.out()));
  SemaphoreList gate(devices_[0], {0}, {1});
  SemaphoreList old_done(devices_[0], {0}, {1});
  SemaphoreList replaced(devices_[1], {0}, {1});
  SemaphoreList dispatched(devices_[1], {0}, {1});
  SemaphoreList downloaded(devices_[0], {0}, {1});
  std::array<float, 4> old_values = {};
  std::array<float, 4> result = {};
  // Hold old use while both replacement and its consumer are submitted. Queue
  // submission must return without waiting for this caller-controlled gate.
  IREE_EXPECT_OK(iree_hal_queue_download(queues_[0], gate, old_done,
                                         first_child, 0, old_values.data(),
                                         sizeof(old_values), nullptr));
  mappings[0].source = nested;
  mappings[1].source = backing;
  for (auto& mapping : mappings) {
    mapping.operation = IREE_HAL_MEMORY_MAPPING_OPERATION_REPLACE;
  }
  IREE_EXPECT_OK(iree_hal_queue_map(queues_[1], old_done, replaced,
                                    mappings.size(), mappings.data(),
                                    IREE_HAL_QUEUE_MAP_FLAG_NONE));
  const iree_hal_buffer_ref_t refs[] = {
      iree_hal_make_buffer_ref(first_child, 0, 4 * sizeof(float)),
      iree_hal_make_buffer_ref(second_child, 0, 4 * sizeof(float)),
      iree_hal_make_buffer_ref(output, 0, 4 * sizeof(float)),
  };
  IREE_EXPECT_OK(iree_hal_queue_dispatch(
      queues_[1], replaced, dispatched, executables_[1],
      iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1),
      iree_const_byte_span_empty(), {IREE_ARRAYSIZE(refs), refs}, nullptr,
      IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_EXPECT_OK(iree_hal_queue_download(queues_[0], dispatched, downloaded,
                                         output, 0, result.data(),
                                         sizeof(result), nullptr));
  uint64_t observed = UINT64_MAX;
  IREE_EXPECT_OK(iree_hal_semaphore_query(replaced.semaphores[0], &observed));
  EXPECT_EQ(observed, 0u);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate.semaphores[0], 1, nullptr));
  IREE_ASSERT_OK(Wait(downloaded));
  EXPECT_THAT(old_values, ::testing::ElementsAre(2, 2, 2, 2));
  EXPECT_THAT(result, ::testing::ElementsAre(10, 10, 10, 10));

  // Writes through the virtual buffer update the original physical storage.
  SemaphoreList written(devices_[1], {0}, {1});
  const float value = 11;
  IREE_ASSERT_OK(iree_hal_queue_fill(
      queues_[1], downloaded, written, first_child, 0, 4 * sizeof(float),
      &value, sizeof(value), nullptr, IREE_HAL_FILL_FLAG_NONE));
  SemaphoreList alias_read(devices_[0], {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_download(
      queues_[0], written, alias_read, backing, 2 * kUnit + kChildOffset,
      result.data(), sizeof(result), nullptr));
  IREE_ASSERT_OK(Wait(alias_read));
  EXPECT_THAT(result, ::testing::ElementsAre(11, 11, 11, 11));

  for (auto& mapping : mappings) {
    mapping.operation = IREE_HAL_MEMORY_MAPPING_OPERATION_UNBIND;
    mapping.source = nullptr;
    mapping.source_offset = 0;
    mapping.access = IREE_HAL_MEMORY_ACCESS_NONE;
  }
  SemaphoreList detached(devices_[0], {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_map(queues_[0], alias_read, detached,
                                    mappings.size(), mappings.data(),
                                    IREE_HAL_QUEUE_MAP_FLAG_BORROW_BUFFERS));
  IREE_ASSERT_OK(Wait(detached));
  // Detached storage can immediately be mapped at another virtual range.
  mappings[1].operation = IREE_HAL_MEMORY_MAPPING_OPERATION_BIND;
  mappings[1].source = nested;
  mappings[1].access = IREE_HAL_MEMORY_ACCESS_READ;
  SemaphoreList reused(devices_[0], {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_map(queues_[0], detached, reused, 1,
                                    &mappings[1],
                                    IREE_HAL_QUEUE_MAP_FLAG_NONE));
  SemaphoreList reuse_read(devices_[0], {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_download(queues_[0], reused, reuse_read,
                                         second_child, 0, result.data(),
                                         sizeof(result), nullptr));
  IREE_ASSERT_OK(Wait(reuse_read));
  EXPECT_THAT(result, ::testing::ElementsAre(11, 11, 11, 11));
  mappings[1].operation = IREE_HAL_MEMORY_MAPPING_OPERATION_PROTECT;
  mappings[1].access = IREE_HAL_MEMORY_ACCESS_NONE;
  SemaphoreList revoked(devices_[0], {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_map(queues_[0], reuse_read, revoked, 1,
                                    &mappings[1],
                                    IREE_HAL_QUEUE_MAP_FLAG_NONE));
  mappings[1].access = IREE_HAL_MEMORY_ACCESS_READ;
  SemaphoreList restored(devices_[1], {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_map(queues_[1], revoked, restored, 1,
                                    &mappings[1],
                                    IREE_HAL_QUEUE_MAP_FLAG_NONE));
  SemaphoreList restored_read(devices_[0], {0}, {1});
  result.fill(0);
  IREE_ASSERT_OK(iree_hal_queue_download(queues_[0], restored, restored_read,
                                         second_child, 0, result.data(),
                                         sizeof(result), nullptr));
  IREE_ASSERT_OK(Wait(restored_read));
  EXPECT_THAT(result, ::testing::ElementsAre(11, 11, 11, 11));
  EXPECT_EQ(iree_hal_buffer_memory_view(backing)
                .backing->native_allocation.byte_length,
            physical_charge);
  first_child.reset();
  second_child.reset();
  first_pool.reset();
  second_pool.reset();
  IREE_ASSERT_OK(iree_hal_virtual_buffer_detach(first, all));
  first.reset();
  // The retained ordinary buffer keeps its address interval live.
  Ref<iree_hal_buffer_t> retained(second.get());
  iree_hal_buffer_retain(retained);
  IREE_ASSERT_OK(iree_hal_virtual_buffer_detach(second, all));
  second.reset();
  IREE_ASSERT_OK(iree_hal_virtual_buffer_query_info(retained, &info));
  EXPECT_EQ(info.mapping_granularity, kUnit);
  retained.reset();
  nested.reset();
  tail.reset();
  backing.reset();
  output.reset();
  iree_hal_pool_stats_t stats;
  ASSERT_TRUE(WaitPoolCommittedBytes(source, 0));
  iree_hal_pool_query_stats(source, &stats);
  EXPECT_EQ(stats.bytes_committed, 0u);
}

TEST_P(TaskSlabPoolTest, RejectsMappingConflictsAndUnpreparedSources) {
  constexpr iree_device_size_t kUnit = 64 * 1024;
  Ref<iree_hal_pool_t> ordinary;
  IREE_ASSERT_OK(CreateSource(ordinary.out()));
  EXPECT_EQ(
      iree_hal_pool_query_mapping_requirements(ordinary).minimum_granularity,
      0u);
  Ref<iree_hal_buffer_t> unprepared;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      ordinary, iree_hal_make_pool_reservation_request(2 * kUnit, kUnit),
      iree_infinite_timeout(), unprepared.out()));
  const iree_hal_virtual_buffer_options_t options = {2 * kUnit, kUnit, kUnit};
  Ref<iree_hal_buffer_t> target;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_create(
      group_, Scope(), &options, iree_allocator_system(), target.out()));
  const iree_hal_memory_scope_t all = {
      iree_hal_device_group_memory_domain(group_), IREE_HAL_MEMORY_SCOPE_ANY};
  iree_hal_memory_mapping_t mapping = {
      target, all, {0, kUnit}, unprepared, 0, IREE_HAL_MEMORY_ACCESS_ALL};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_queue_map(queues_[0], iree_hal_semaphore_list_empty(),
                         iree_hal_semaphore_list_empty(), 1, &mapping,
                         IREE_HAL_QUEUE_MAP_FLAG_NONE));
  for (auto& family : families_) {
    family.interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE;
  }
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(CreateSource(source.out()));
  Ref<iree_hal_buffer_t> backing;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      source, iree_hal_make_pool_reservation_request(2 * kUnit, kUnit),
      iree_infinite_timeout(), backing.out()));
  mapping.source = backing;
  mapping.operation = IREE_HAL_MEMORY_MAPPING_OPERATION_UNBIND;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_queue_map(queues_[0], iree_hal_semaphore_list_empty(),
                         iree_hal_semaphore_list_empty(), 1, &mapping,
                         IREE_HAL_QUEUE_MAP_FLAG_NONE));
  mapping.source = nullptr;
  for (auto operation : {IREE_HAL_MEMORY_MAPPING_OPERATION_BIND,
                         IREE_HAL_MEMORY_MAPPING_OPERATION_PROTECT}) {
    mapping.operation = operation;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        iree_hal_queue_map(queues_[0], iree_hal_semaphore_list_empty(),
                           iree_hal_semaphore_list_empty(), 1, &mapping,
                           IREE_HAL_QUEUE_MAP_FLAG_NONE));
  }
  mapping.source = backing;
  mapping.operation = IREE_HAL_MEMORY_MAPPING_OPERATION_REPLACE;
  std::array<iree_hal_memory_mapping_t, 2> overlapping = {mapping, mapping};
  overlapping[1].source_offset = kUnit;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_queue_map(queues_[0], iree_hal_semaphore_list_empty(),
                         iree_hal_semaphore_list_empty(), overlapping.size(),
                         overlapping.data(), IREE_HAL_QUEUE_MAP_FLAG_NONE));
  Ref<iree_hal_buffer_t> alias;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(target, kUnit, kUnit,
                                         iree_allocator_system(), alias.out()));
  overlapping[0].target_range.byte_offset = kUnit;
  overlapping[1].target = alias;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_queue_map(queues_[0], iree_hal_semaphore_list_empty(),
                         iree_hal_semaphore_list_empty(), overlapping.size(),
                         overlapping.data(), IREE_HAL_QUEUE_MAP_FLAG_NONE));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_virtual_buffer_detach(alias, all));
  alias.reset();
  mapping.target_range.byte_length = 3 * kUnit;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_queue_map(queues_[0], iree_hal_semaphore_list_empty(),
                         iree_hal_semaphore_list_empty(), 1, &mapping,
                         IREE_HAL_QUEUE_MAP_FLAG_NONE));
  mapping.target_range.byte_length = kUnit;
  mapping.scope.id = families_[1].family->memory.queue_scope_id;
  uint32_t first_projection = UINT32_MAX, second_projection = UINT32_MAX;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_resolve_projection(
      target, mapping.scope, &first_projection));
  mapping.scope.id = families_[0].family->memory.queue_scope_id;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_resolve_projection(
      target, mapping.scope, &second_projection));
  EXPECT_EQ(first_projection, second_projection);
  mapping.scope.id = iree_hal_queue_family(queues_[2])->memory.queue_scope_id;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_queue_map(queues_[0], iree_hal_semaphore_list_empty(),
                         iree_hal_semaphore_list_empty(), 1, &mapping,
                         IREE_HAL_QUEUE_MAP_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_virtual_buffer_detach(target, all));
  target.reset();
  backing.reset();
  unprepared.reset();
}

TEST_P(TaskSlabPoolTest, QueueMappingSourcesRecycleThroughSharedSlabCache) {
  constexpr iree_device_size_t kUnit = 64 * 1024;
  for (auto& family : families_) {
    family.interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE;
  }
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(CreateSource(source.out()));
  const auto requirements = iree_hal_pool_query_mapping_requirements(source);
  ASSERT_GT(requirements.minimum_granularity, 0u);
  EXPECT_EQ(requirements.recommended_granularity,
            requirements.minimum_granularity);
  EXPECT_EQ(kUnit % requirements.minimum_granularity, 0u);
  EXPECT_EQ(requirements.flags, 0u);
  iree_hal_slab_cache_options_t cache_options;
  iree_hal_slab_cache_options_initialize(&cache_options);
  cache_options.slab = iree_hal_make_pool_reservation_request(4 * kUnit, kUnit);
  Ref<iree_hal_pool_t> cache;
  IREE_ASSERT_OK(iree_hal_slab_cache_create(
      source, &cache_options, iree_allocator_system(), cache.out()));
  std::array<Ref<iree_hal_pool_t>, 2> children;
  iree_hal_tlsf_pool_options_t tlsf_options = {};
  tlsf_options.tlsf_options.range_length = 4 * kUnit;
  tlsf_options.tlsf_options.alignment = kUnit;
  tlsf_options.tlsf_options.frontier_capacity = 4;
  IREE_ASSERT_OK(iree_hal_tlsf_pool_create(
      cache, &tlsf_options, iree_allocator_system(), children[0].out()));
  iree_hal_fixed_block_pool_options_t block_options = {
      .block_size = kUnit,
      .blocks_per_slab = 4,
      .frontier_capacity = 4,
      .alignment = kUnit,
  };
  IREE_ASSERT_OK(iree_hal_fixed_block_pool_create(
      cache, &block_options, iree_allocator_system(), children[1].out()));
  for (auto& family : families_) {
    family.interfaces = 0;
  }
  Ref<iree_hal_buffer_t> target;
  const iree_hal_virtual_buffer_options_t options = {2 * kUnit, kUnit, kUnit};
  IREE_ASSERT_OK(iree_hal_virtual_buffer_create(
      group_, Scope(), &options, iree_allocator_system(), target.out()));
  const iree_hal_memory_scope_t all = {
      iree_hal_device_group_memory_domain(group_), IREE_HAL_MEMORY_SCOPE_ANY};
  const iree_hal_buffer_backing_facts_t* first_backing = nullptr;
  for (auto& child : children) {
    const auto inherited = iree_hal_pool_query_mapping_requirements(child);
    EXPECT_EQ(inherited.minimum_granularity, requirements.minimum_granularity);
    EXPECT_EQ(inherited.recommended_granularity,
              requirements.recommended_granularity);
    EXPECT_EQ(inherited.flags, requirements.flags);
    // The source has no committed native binding until this gate is signaled.
    SemaphoreList gate(devices_[0], {0}, {1});
    SemaphoreList allocated(devices_[0], {0}, {1});
    SemaphoreList mapped(devices_[1], {0}, {1});
    SemaphoreList filled(devices_[1], {0}, {1});
    SemaphoreList downloaded(devices_[0], {0}, {1});
    std::array<iree_hal_pool_reservation_request_t, 2> requests = {
        iree_hal_make_pool_reservation_request(kUnit, kUnit),
        iree_hal_make_pool_reservation_request(kUnit, kUnit)};
    std::array<iree_hal_buffer_t*, 2> buffers = {};
    IREE_EXPECT_OK(iree_hal_queue_alloca(queues_[0], gate, allocated, child,
                                         requests.size(), requests.data(),
                                         buffers.data()));
    std::array<Ref<iree_hal_buffer_t>, 2> roots;
    for (size_t i = 0; i < roots.size(); ++i) {
      roots[i].reset(buffers[i]);
    }
    std::array<iree_hal_memory_mapping_t, 2> mappings = {{
        {target, all, {0, kUnit}, roots[0], 0, IREE_HAL_MEMORY_ACCESS_ALL},
        {target, all, {kUnit, kUnit}, roots[1], 0, IREE_HAL_MEMORY_ACCESS_ALL},
    }};
    IREE_EXPECT_OK(iree_hal_queue_map(queues_[1], allocated, mapped,
                                      mappings.size(), mappings.data(),
                                      IREE_HAL_QUEUE_MAP_FLAG_NONE));
    const uint32_t value = 0x12345678;
    IREE_EXPECT_OK(iree_hal_queue_fill(queues_[1], mapped, filled, target.get(),
                                       0, 2 * kUnit, &value, sizeof(value),
                                       nullptr, IREE_HAL_FILL_FLAG_NONE));
    uint32_t observed = 0;
    IREE_EXPECT_OK(iree_hal_queue_download(queues_[0], filled, downloaded,
                                           roots[1], 0, &observed,
                                           sizeof(observed), nullptr));
    IREE_ASSERT_OK(iree_hal_semaphore_signal(gate.semaphores[0], 1, nullptr));
    IREE_ASSERT_OK(Wait(downloaded));
    EXPECT_EQ(observed, value);
    for (auto& root : roots) {
      const auto memory = iree_hal_buffer_memory_view(root);
      ASSERT_NE(memory.backing, nullptr);
      if (!first_backing) {
        first_backing = memory.backing;
      }
      EXPECT_EQ(memory.backing, first_backing);
    }
    for (auto& mapping : mappings) {
      mapping.source = nullptr;
      mapping.access = IREE_HAL_MEMORY_ACCESS_NONE;
    }
    SemaphoreList detached(devices_[1], {0}, {1});
    IREE_ASSERT_OK(iree_hal_queue_map(queues_[1], downloaded, detached,
                                      mappings.size(), mappings.data(),
                                      IREE_HAL_QUEUE_MAP_FLAG_NONE));
    SemaphoreList deallocated(devices_[0], {0}, {1});
    IREE_ASSERT_OK(iree_hal_queue_dealloca(queues_[0], detached, deallocated,
                                           buffers.size(), buffers.data()));
    IREE_ASSERT_OK(Wait(deallocated));
    for (auto& root : roots) {
      root.reset();
    }
    iree_hal_pool_stats_t stats;
    iree_hal_pool_query_stats(child, &stats);
    EXPECT_EQ(stats.bytes_committed, 0u);
  }
  IREE_ASSERT_OK(iree_hal_virtual_buffer_detach(target, all));
  target.reset();
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(source, &stats);
  EXPECT_EQ(stats.reserve_count, 1u);
  iree_hal_slab_cache_stats_t cache_stats;
  IREE_ASSERT_OK(iree_hal_slab_cache_query_stats(cache, &cache_stats));
  EXPECT_EQ(cache_stats.hit_count, 1u);
  EXPECT_EQ(cache_stats.ready_count, 1u);
  const auto physical_charge = stats.bytes_committed;
  EXPECT_GE(physical_charge, 4 * kUnit);

  // Child pools return idle ranges to the shared cache. Only the cache's
  // explicit trim releases that retained physical backing to the source.
  for (auto& child : children) {
    iree_hal_pool_trim(child, IREE_HAL_POOL_TRIM_FLAG_EXCESS, 0);
  }
  iree_hal_pool_trim(cache, IREE_HAL_POOL_TRIM_FLAG_EXCESS, 4 * kUnit);
  iree_hal_pool_query_stats(source, &stats);
  EXPECT_EQ(stats.bytes_committed, physical_charge);
  IREE_ASSERT_OK(iree_hal_slab_cache_query_stats(cache, &cache_stats));
  EXPECT_EQ(cache_stats.ready_count, 1u);

  iree_hal_pool_trim(cache, IREE_HAL_POOL_TRIM_FLAG_EXCESS, 0);
  ASSERT_TRUE(WaitPoolCommittedBytes(source, 0));
  iree_hal_pool_query_stats(source, &stats);
  EXPECT_EQ(stats.bytes_committed, 0u);
  EXPECT_EQ(stats.release_count, 1u);
}

CTS_REGISTER_EXECUTABLE_TEST_SUITE(TaskSlabPoolTest);

}  // namespace
}  // namespace iree::hal::cts
