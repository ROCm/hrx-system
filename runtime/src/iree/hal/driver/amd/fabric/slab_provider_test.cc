// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/fabric/slab_provider.h"

#include <cstdint>
#include <future>
#include <initializer_list>
#include <thread>
#include <utility>
#include <vector>

#include "iree/async/frontier_tracker.h"
#include "iree/async/notification.h"
#include "iree/async/proactor_platform.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/async/util/proactor_thread.h"
#include "iree/base/status_cc.h"
#include "iree/hal/memory/passthrough_pool.h"
#include "iree/hal/pool_wait.h"
#include "iree/hal/testing/mock_device.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct FakeNative;

struct FakeMemory {
  // Dependency owner observing construction and release.
  FakeNative* owner;
  // Owned bytes modeling one native backing.
  std::vector<uint8_t> bytes;
  // Borrowed application storage for a REGISTER construction.
  void* registered_pointer = nullptr;
  // Complete attached extent, independent of ownership.
  uint64_t byte_length = 0;
  // Achieved alignment returned by the native attachment.
  uint64_t alignment = 64;
  // True only for CREATE backing, independent of imported source extents.
  bool allocated_backing = false;

  void* data() {
    return registered_pointer ? registered_pointer : bytes.data();
  }
};

struct FakeMapping {
  // Borrowed backing retained by the test storage owner.
  FakeMemory* memory;
};

struct CacheOperation {
  // Requested native cache operation.
  amdf_host_cache_operation_t operation;
  // Operation offset from native allocation byte zero.
  uint64_t offset;
  // Operation extent in bytes.
  uint64_t length;
};

struct FakeNative {
  FakeNative() {
    api.memory_create = MemoryCreate;
    api.memory_import = MemoryImport;
    api.memory_export = MemoryExport;
    api.memory_query_info = MemoryQueryInfo;
    api.memory_map = MemoryMap;
    api.host_mapping_query_info = MappingQueryInfo;
    api.host_mapping_cache_control = MappingCacheControl;
    api.host_mapping_destroy = MappingDestroy;
    api.memory_destroy = MemoryDestroy;
    api.memory_query_address = MemoryQueryAddress;
  }

  amdf_memory_scope_t* scope() {
    return reinterpret_cast<amdf_memory_scope_t*>(this);
  }

  static amdf_status_t AMDF_CALL MemoryCreate(
      amdf_memory_scope_t* scope, const amdf_memory_create_info_t* info,
      amdf_memory_t** out_memory,
      amdf_memory_release_info_t* out_release_info) {
    if (out_release_info) {
      *out_release_info = {};
    }
    auto* self = reinterpret_cast<FakeNative*>(scope);
    *out_memory = nullptr;
    const bool registered = info->registered_host_pointer != nullptr;
    EXPECT_EQ(info->memory_profile_ordinal, registered ? 8u : 7u);
    EXPECT_EQ(info->access_count, 2u);
    EXPECT_EQ(info->required_flags, self->required_flags);
    EXPECT_NE(info->byte_length, 0u);
    EXPECT_LE(info->byte_length, 4096u);
    if (!registered) {
      EXPECT_EQ(info->byte_length % 64, 0u);
      EXPECT_GE(info->minimum_alignment, 64u);
    } else {
      EXPECT_EQ(info->registered_host_properties,
                (AMDF_HOST_MEMORY_PROPERTY_READ_CACHED |
                 AMDF_HOST_MEMORY_PROPERTY_WRITE_BACK));
    }
    EXPECT_LE(info->minimum_alignment, 4096u);
    EXPECT_TRUE(iree_device_size_is_power_of_two(info->minimum_alignment));
    if (self->create_count == self->fail_after_create_count) {
      return amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
    }
    self->created_lengths.push_back(info->byte_length);
    self->created_alignments.push_back(info->minimum_alignment);
    auto* memory = new FakeMemory{
        self,
        std::vector<uint8_t>(
            registered ? 0 : iree_host_align(info->byte_length, 4096)),
        info->registered_host_pointer,
        info->byte_length,
        info->minimum_alignment,
        !registered};
    if (++self->create_attempt_count == self->fail_create_at) {
      MemoryDestroy(reinterpret_cast<amdf_memory_t*>(memory), out_release_info);
      return amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
    }
    ++self->create_count;
    *out_memory = reinterpret_cast<amdf_memory_t*>(memory);
    return AMDF_STATUS_OK;
  }

  static amdf_status_t AMDF_CALL MemoryImport(
      amdf_memory_scope_t* scope, const amdf_memory_import_info_t* info,
      amdf_external_memory_t* storage, amdf_memory_t** out_memory,
      amdf_memory_release_info_t* out_release_info) {
    if (out_release_info) {
      *out_release_info = {};
    }
    auto* self = reinterpret_cast<FakeNative*>(scope);
    EXPECT_EQ(info->memory_profile_ordinal, 8u);
    EXPECT_EQ(info->access_count, 2u);
    EXPECT_EQ(info->required_flags, self->required_flags);
    EXPECT_EQ(storage->type, self->external_type);
    EXPECT_EQ(storage->provenance.words[0], self->storage_class.provenance[0]);
    EXPECT_EQ(storage->provenance.words[1], self->storage_class.provenance[1]);
    EXPECT_EQ(storage->source_byte_offset, 0u);
    EXPECT_EQ(storage->release, nullptr);
    EXPECT_EQ(storage->release_user_data, nullptr);
    EXPECT_EQ(storage->payload.device_address, 1234u);
    EXPECT_EQ(storage->physical_backing_id.words[0], 123u);
    auto* memory =
        new FakeMemory{self, std::vector<uint8_t>(storage->byte_length),
                       nullptr, storage->byte_length, info->minimum_alignment};
    ++self->import_count;
    *storage = {};
    *out_memory = reinterpret_cast<amdf_memory_t*>(memory);
    return AMDF_STATUS_OK;
  }

  static amdf_status_t AMDF_CALL
  MemoryExport(amdf_memory_t* handle, const amdf_memory_export_info_t* info,
               amdf_external_memory_t* out_storage) {
    auto* memory = reinterpret_cast<FakeMemory*>(handle);
    auto* self = memory->owner;
    EXPECT_EQ(info->external_memory_type, self->external_type);
    EXPECT_EQ(info->byte_offset, 0u);
    EXPECT_EQ(info->byte_length, memory->byte_length);
    if (self->fail_export) {
      return amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
    }
    *out_storage = {};
    out_storage->type = info->external_memory_type;
    out_storage->provenance = {
        {self->storage_class.provenance[0], self->storage_class.provenance[1]}};
    out_storage->payload.device_address = 5678;
    out_storage->byte_length = memory->byte_length;
    out_storage->physical_backing_id = {{123, 456}};
    out_storage->release = ReleaseExport;
    out_storage->release_user_data = self;
    ++self->export_count;
    return AMDF_STATUS_OK;
  }

  static void AMDF_CALL ReleaseExport(void* user_data,
                                      amdf_external_memory_type_t type,
                                      amdf_external_memory_payload_t payload) {
    auto* self = static_cast<FakeNative*>(user_data);
    EXPECT_EQ(type, self->external_type);
    EXPECT_EQ(payload.device_address, 5678u);
    ++self->release_export_count;
  }

  static amdf_status_t AMDF_CALL MemoryQueryInfo(amdf_memory_t* handle,
                                                 amdf_memory_info_t* out_info) {
    auto* memory = reinterpret_cast<FakeMemory*>(handle);
    const uintptr_t pointer = reinterpret_cast<uintptr_t>(memory->data());
    out_info->byte_length = memory->byte_length;
    out_info->alignment = memory->registered_pointer ? pointer & (~pointer + 1)
                                                     : memory->alignment;
    out_info->native_allocation_byte_length =
        iree_host_align(memory->byte_length, 4096);
    out_info->native_allocation_granularity = 64;
    out_info->physical_backing_id = {{123, 456}};
    return AMDF_STATUS_OK;
  }

  static amdf_status_t AMDF_CALL MemoryMap(amdf_memory_t* base_memory,
                                           const amdf_memory_map_info_t* info,
                                           amdf_host_mapping_t** out_mapping) {
    auto* memory = reinterpret_cast<FakeMemory*>(base_memory);
    EXPECT_EQ(info->byte_length, memory->byte_length);
    EXPECT_EQ(info->flags,
              AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE);
    *out_mapping =
        reinterpret_cast<amdf_host_mapping_t*>(new FakeMapping{memory});
    ++memory->owner->map_count;
    return AMDF_STATUS_OK;
  }

  static amdf_status_t AMDF_CALL MappingQueryInfo(
      amdf_host_mapping_t* base_mapping, amdf_host_mapping_info_t* out_info) {
    auto* mapping = reinterpret_cast<FakeMapping*>(base_mapping);
    out_info->flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
    out_info->properties = (AMDF_HOST_MEMORY_PROPERTY_READ_CACHED |
                            AMDF_HOST_MEMORY_PROPERTY_WRITE_BACK);
    out_info->pointer = mapping->memory->data();
    out_info->byte_length = mapping->memory->byte_length;
    out_info->cache_line_size = 64;
    out_info->flush.kind = AMDF_CACHE_TRANSITION_KIND_RANGE;
    out_info->flush.range_granularity = 64;
    out_info->invalidate.kind = AMDF_CACHE_TRANSITION_KIND_RANGE;
    out_info->invalidate.range_granularity = 64;
    return AMDF_STATUS_OK;
  }

  static amdf_status_t AMDF_CALL MappingCacheControl(
      amdf_host_mapping_t* base_mapping, amdf_host_cache_operation_t operation,
      uint64_t byte_offset, uint64_t byte_length) {
    auto* mapping = reinterpret_cast<FakeMapping*>(base_mapping);
    mapping->memory->owner->cache_operations.push_back(
        {operation, byte_offset, byte_length});
    return AMDF_STATUS_OK;
  }

  static amdf_status_t AMDF_CALL
  MappingDestroy(amdf_host_mapping_t* base_mapping) {
    auto* mapping = reinterpret_cast<FakeMapping*>(base_mapping);
    ++mapping->memory->owner->unmap_count;
    delete mapping;
    return AMDF_STATUS_OK;
  }

  static amdf_status_t AMDF_CALL
  MemoryDestroy(amdf_memory_t* base_memory,
                amdf_memory_release_info_t* out_release_info) {
    auto* memory = reinterpret_cast<FakeMemory*>(base_memory);
    FakeNative* self = memory->owner;
    ++self->destroy_count;
    self->destroy_threads.emplace_back(memory->registered_pointer != nullptr,
                                       std::this_thread::get_id());
    if (self->hold_destroy) {
      self->hold_destroy = false;
      self->destroy_entered.set_value();
      self->destroy_resume.wait();
    }
    const uint64_t abandoned_byte_length = self->fail_memory_destroy &&
                                                   self->abandon_backing &&
                                                   memory->allocated_backing
                                               ? memory->bytes.size()
                                               : 0;
    if (abandoned_byte_length) {
      self->abandoned_backing.push_back(std::move(memory->bytes));
    }
    delete memory;
    const amdf_status_t status =
        self->fail_memory_destroy
            ? amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL)
            : AMDF_STATUS_OK;
    if (out_release_info) {
      *out_release_info = {status, abandoned_byte_length};
    }
    return status;
  }

  static amdf_status_t AMDF_CALL
  MemoryQueryAddress(amdf_memory_t* base_memory, uint32_t access_ordinal,
                     amdf_memory_address_kind_t kind, uint64_t* out_address) {
    auto* memory = reinterpret_cast<FakeMemory*>(base_memory);
    EXPECT_EQ(kind, access_ordinal == 0 ? AMDF_MEMORY_ADDRESS_GPU
                                        : AMDF_MEMORY_ADDRESS_XDNA_DMA);
    EXPECT_LT(access_ordinal, 2u);
    if (access_ordinal == 1 && memory->owner->fail_second_address) {
      return amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
    }
    *out_address = UINT64_C(0x100000) + UINT64_C(0x10000) * access_ordinal +
                   memory->byte_length;
    return AMDF_STATUS_OK;
  }

  // Dependency API consumed by the production provider.
  amdf_api_t api = {};
  // Requested native allocation extents.
  std::vector<uint64_t> created_lengths;
  // Requested native allocation alignments.
  std::vector<uint64_t> created_alignments;
  // Ordered cache-control requests.
  std::vector<CacheOperation> cache_operations;
  // Successful native allocation count.
  int create_count = 0;
  // Number of native construction attempts, including unpublished failures.
  int create_attempt_count = 0;
  // One-based construction attempt to fail after acquiring backing, or zero.
  int fail_create_at = 0;
  // Successful native external attachment count.
  int import_count = 0;
  // Successful independent transport export count.
  int export_count = 0;
  // Released exported payload count, independent of memory destruction.
  int release_export_count = 0;
  // Exact protocol class exposed by the selected native profile.
  iree_hal_external_storage_class_t storage_class = {};
  // Native representation corresponding to the selected HAL class.
  amdf_external_memory_type_t external_type = AMDF_EXTERNAL_MEMORY_TYPE_NONE;
  // Transport allocation failure injected without publishing an output.
  bool fail_export = false;
  // Successful host-view creation count.
  int map_count = 0;
  // Consumed host-view count.
  int unmap_count = 0;
  // Consumed native-memory count.
  int destroy_count = 0;
  // Imported/owned classification and execution thread for each destruction.
  std::vector<std::pair<bool, std::thread::id>> destroy_threads;
  // Whether consuming cleanup reports a native failure.
  bool fail_memory_destroy = false;
  // Whether failed native release retains allocation-owned payload.
  bool abandon_backing = false;
  // Payload retained independently after the native handle was consumed.
  std::vector<std::vector<uint8_t>> abandoned_backing;
  // Holds the next native release at its physical ownership boundary.
  bool hold_destroy = false;
  // Publishes that the native worker has entered the held destruction.
  std::promise<void> destroy_entered;
  // Caller-controlled continuation for that destruction, with no deadline.
  std::shared_future<void> destroy_resume;
  // Successful creation count at which the next acquisition fails, or -1.
  int fail_after_create_count = -1;
  // Expected backing requirements for the active provider.
  amdf_memory_flags_t required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
  // Native address failure injected after memory and mapping acquisition.
  bool fail_second_address = false;
};

class SlabProviderTest : public ::testing::Test {
 protected:
  SlabProviderTest() {
    static const iree_hal_resource_vtable_t vtable = {
        .destroy =
            [](iree_hal_resource_t* resource) {
              reinterpret_cast<NativeLifetime*>(resource)->released.set_value();
            },
    };
    iree_hal_resource_initialize(&vtable, &native_lifetime_.resource);
  }

  void SetUp() override {
    iree_hal_mock_device_options_t options;
    iree_hal_mock_device_options_initialize(&options);
    options.lifetime_owner = &native_lifetime_.resource;
    for (auto** device : {&devices_[0], &devices_[1]}) {
      IREE_ASSERT_OK(iree_hal_mock_device_create(
          &options, iree_allocator_system(), device));
    }
    IREE_ASSERT_OK(iree_hal_amd_fabric_allocation_queue_create(
        UINT32_MAX, iree_allocator_system(), &allocation_queue_));
    CreateProvider(AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE);
  }

  void CreateProvider(
      amdf_memory_map_flags_t host_access,
      const iree_hal_pool_import_class_t* import_policy = nullptr,
      uint64_t maximum_byte_length = 4096,
      iree_async_proactor_t* const* io_proactors = nullptr) {
    iree_hal_slab_provider_release(provider_);
    provider_ = nullptr;
    accesses_[0].device = reinterpret_cast<amdf_device_t*>(uintptr_t{1});
    accesses_[1].device = reinterpret_cast<amdf_device_t*>(uintptr_t{2});
    for (auto& access : accesses_) {
      access.requirements.access =
          AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
      access.requirements.flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
      access.requirements.address_kinds =
          UINT64_C(1) << (&access == accesses_ ? AMDF_MEMORY_ADDRESS_GPU
                                               : AMDF_MEMORY_ADDRESS_XDNA_DMA);
    }
    iree_hal_amd_fabric_slab_provider_options_t options = {};
    options.memory.scope = native_.scope();
    options.memory.profile.ordinal = 7;
    options.memory.profile.roles =
        AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP;
    options.memory.profile.supported_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
    options.memory.profile.allocation.maximum_byte_length = maximum_byte_length;
    options.memory.profile.allocation.byte_length_granularity = 64;
    options.memory.profile.allocation.native_byte_length_granularity = 4096;
    options.memory.profile.allocation.minimum_alignment = 64;
    options.memory.profile.allocation.maximum_alignment = 4096;
    options.memory.access_count = 2;
    options.memory.accesses = accesses_;
    native_.required_flags = host_access ? AMDF_MEMORY_FLAG_HOST_VISIBLE : 0;
    options.memory.required_flags = native_.required_flags;
    options.memory.host_access = host_access;
    iree_hal_amd_fabric_import_route_t import_route = {};
    if (import_policy) {
      import_route.policy = *import_policy;
      import_route.profile = options.memory.profile;
      import_route.profile.ordinal = 8;
      import_route.profile.roles =
          AMDF_MEMORY_PROFILE_ROLE_REGISTER | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP;
      import_route.profile.registration.maximum_byte_length = 4096;
      import_route.profile.registration.byte_length_granularity = 1;
      import_route.profile.registration.registered_host_pointer_alignment = 1;
      import_route.profile.registration.registered_host_properties =
          (AMDF_HOST_MEMORY_PROPERTY_READ_CACHED |
           AMDF_HOST_MEMORY_PROPERTY_WRITE_BACK);
      import_route.profile.registration.minimum_alignment = 1;
      import_route.profile.registration.maximum_alignment = 4096;
      import_route.required_flags = options.memory.required_flags;
      import_route.host_access = host_access;
      if (import_policy->storage_class.type !=
          IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER) {
        import_route.profile.roles = AMDF_MEMORY_PROFILE_ROLE_IMPORT |
                                     AMDF_MEMORY_PROFILE_ROLE_EXPORT |
                                     AMDF_MEMORY_PROFILE_ROLE_HOST_MAP;
        import_route.profile.import = import_route.profile.allocation;
        import_route.profile.allocation = {};
        import_route.profile.import.byte_length_granularity = 192;
        import_route.profile.registration = {};
        import_route.profile.external_memory_support_count = 1;
        auto& support = import_route.profile.external_memory_support[0];
        support.type = native_.external_type;
        support.flags = AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_IMPORT |
                        AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_EXPORT;
        support.provenance = {{import_policy->storage_class.provenance[0],
                               import_policy->storage_class.provenance[1]}};
        support.byte_length_alignment = 192;
        if (import_policy->exports.class_count) {
          import_route.profile.supported_flags |= AMDF_MEMORY_FLAG_SHAREABLE;
          import_route.required_flags |= AMDF_MEMORY_FLAG_SHAREABLE;
          native_.required_flags = import_route.required_flags;
        }
      }
      options.import_count = 1;
      options.imports = &import_route;
    }
    iree_hal_amd_fabric_binding_request_t bindings[5] = {};
    bindings[1] = {IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS, 0,
                   AMDF_MEMORY_ADDRESS_GPU};
    bindings[2] = {IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA, 1,
                   AMDF_MEMORY_ADDRESS_XDNA_DMA};
    options.binding_count = host_access ? 3 : 2;
    options.host_binding_index =
        host_access ? 0 : IREE_HAL_BUFFER_NATIVE_BINDING_INDEX_NONE;
    options.bindings = host_access ? bindings : bindings + 1;
    if (io_proactors) {
      for (uint32_t i = 0; i < 2; ++i) {
        bindings[3 + i].type = IREE_HAL_BUFFER_INTERFACE_REGISTERED_IO;
        bindings[3 + i].access_ordinal = i;
        bindings[3 + i].io_proactor = io_proactors[i];
      }
      options.binding_count += 2;
    }
    options.memory_type =
        IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
        IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
    options.supported_usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                              IREE_HAL_BUFFER_USAGE_DISPATCH |
                              IREE_HAL_BUFFER_USAGE_MAPPING;
    if (!host_access) {
      options.memory_type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
                            IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
      options.supported_usage &= ~IREE_HAL_BUFFER_USAGE_MAPPING;
    }
    options.maintenance_alignment = 64;
    IREE_ASSERT_OK(iree_hal_amd_fabric_slab_provider_create(
        devices_[0], &native_.api, iree_hal_device_event_sink_discard(),
        allocation_queue_, &options, iree_allocator_system(), &provider_));
  }

  void TearDown() override {
    iree_hal_slab_provider_release(provider_);
    for (auto* device : devices_) {
      iree_hal_device_release(device);
    }
    // Native callbacks retain the provider and its devices beyond public
    // pool release. Join those owners before derived fixtures destroy their
    // proactors or the native dependency's storage leaves scope.
    iree_hal_resource_release(&native_lifetime_.resource);
    native_lifetime_.released.get_future().get();
    iree_hal_amd_fabric_allocation_queue_release(allocation_queue_);
  }

  struct NativeLifetime {
    // Shared native owner retained by both mock devices and the fixture.
    iree_hal_resource_t resource;
    // Final device release proves every provider callback has returned.
    std::promise<void> released;
  };

  // Mirrors native device ownership without making statistics a teardown fence.
  NativeLifetime native_lifetime_;
  // Mocked native dependency.
  FakeNative native_;
  // Requested accesses copied by the production provider.
  amdf_memory_device_access_t accesses_[2] = {};
  // Owned production provider under test.
  iree_hal_slab_provider_t* provider_ = nullptr;
  // HAL owners retained by the provider, with native dependencies mocked above.
  iree_hal_device_t* devices_[2] = {};
  // Native worker used by the real provider acquisition and release paths.
  iree_hal_amd_fabric_allocation_queue_t* allocation_queue_ = nullptr;
  // Completion storage remains alive until TearDown joins the native owner.
  std::promise<void> release_completed_;
};

class RegisteredSlabProviderTest : public SlabProviderTest,
                                   public ::testing::WithParamInterface<bool> {
 protected:
  void SetUp() override {
    SlabProviderTest::SetUp();
    IREE_ASSERT_OK(iree_async_proactor_pool_create(
        2, nullptr, iree_async_proactor_pool_options_default(),
        iree_allocator_system(), &proactors_));
    IREE_ASSERT_OK(
        iree_async_proactor_pool_get(proactors_, 0, &io_proactors_[0]));
    IREE_ASSERT_OK(iree_async_proactor_pool_get(proactors_, GetParam() ? 1 : 0,
                                                &io_proactors_[1]));
    for (auto* proactor : io_proactors_) {
      if (!iree_all_bits_set(
              iree_async_proactor_query_capabilities(proactor),
              IREE_ASYNC_PROACTOR_CAPABILITY_CONCURRENT_BUFFER_REGISTRATION)) {
        GTEST_SKIP() << "Prepared buffer attachment is unavailable";
      }
    }
    CreateProvider(AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
                   nullptr, 4096, io_proactors_);
  }

  void TearDown() override {
    SlabProviderTest::TearDown();
    iree_async_proactor_pool_release(proactors_);
  }

  iree_status_t Acquire(
      iree_host_size_t count, iree_hal_slab_t* slabs,
      iree_hal_slab_release_info_t* out_release_info = nullptr) {
    struct Completion {
      // Published after both native and registration callbacks finish.
      std::promise<iree::Status> status;
      // Separate cleanup outcome, owned after the status is published.
      iree_hal_slab_release_info_t release_info = {};
    } completion;
    std::vector<iree_hal_slab_request_t> geometries(count, {256, 64});
    iree_hal_slab_acquire_request_t request = {
        .count = count,
        .requests = geometries.data(),
        .slabs = slabs,
        .completion = {[](void* user_data, iree_status_t status,
                          iree_hal_slab_release_info_t release_info) {
                         auto* completion = static_cast<Completion*>(user_data);
                         completion->release_info = release_info;
                         completion->status.set_value(
                             iree::Status(std::move(status)));
                       },
                       &completion},
    };
    auto result = completion.status.get_future();
    iree_hal_slab_provider_acquire_slabs(provider_, &request);
    auto status = result.get();
    if (out_release_info) {
      *out_release_info = completion.release_info;
    } else {
      IREE_EXPECT_OK(iree::Status(std::move(completion.release_info.status)));
      EXPECT_EQ(completion.release_info.abandoned_byte_length, 0u);
    }
    return status.release();
  }

  // Owns real polling owners until native cleanup and their callbacks finish.
  iree_async_proactor_pool_t* proactors_ = nullptr;
  // The two device bindings may share one owner or use independent owners.
  iree_async_proactor_t* io_proactors_[2] = {};
};

TEST_P(RegisteredSlabProviderTest, PublishesAndReleasesAllBindingsInBatch) {
  iree_hal_slab_t slabs[3] = {};
  IREE_ASSERT_OK(Acquire(IREE_ARRAYSIZE(slabs), slabs));
  for (auto& slab : slabs) {
    iree_hal_buffer_t* buffer = nullptr;
    IREE_ASSERT_OK(iree_hal_slab_provider_wrap_buffer(
        provider_, &slab, 0, slab.length, IREE_HAL_MEMORY_ACCESS_ALL,
        IREE_HAL_BUFFER_USAGE_TRANSFER, {}, &buffer));
    iree_async_region_t* regions[2] = {};
    for (uint16_t i = 0; i < 2; ++i) {
      const auto binding = iree_hal_buffer_native_binding(
          buffer, {static_cast<uint16_t>(3 + i),
                   IREE_HAL_BUFFER_INTERFACE_REGISTERED_IO});
      auto* table = static_cast<const iree_hal_amd_fabric_io_binding_table_t*>(
          binding.registered_io.table);
      EXPECT_EQ(binding.registered_io.index, i);
      regions[i] = table->regions[binding.registered_io.index];
      EXPECT_EQ(regions[i]->proactor, io_proactors_[i]);
      EXPECT_EQ(regions[i]->base_ptr, slab.base_ptr);
      EXPECT_EQ(regions[i]->length, slab.length);
    }
    EXPECT_EQ(regions[0] == regions[1], !GetParam());
    iree_hal_buffer_release(buffer);
  }
  auto completion = release_completed_.get_future();
  iree_hal_slab_release_request_t release = {
      .count = IREE_ARRAYSIZE(slabs),
      .slabs = slabs,
      .completion = {[](void* user_data,
                        iree_hal_slab_release_info_t release_info) {
                       IREE_EXPECT_OK(
                           iree::Status(std::move(release_info.status)));
                       EXPECT_EQ(release_info.abandoned_byte_length, 0u);
                       static_cast<std::promise<void>*>(user_data)->set_value();
                     },
                     &release_completed_},
  };
  iree_hal_slab_provider_release_slabs(provider_, &release);
  completion.get();
  EXPECT_EQ(native_.create_count, 3);
  EXPECT_EQ(native_.unmap_count, 3);
  EXPECT_EQ(native_.destroy_count, 3);
  for (const auto& slab : slabs) {
    EXPECT_EQ(slab.provider_handle, 0u);
  }
}

TEST_P(RegisteredSlabProviderTest,
       PartialPreparationRollsBackBeforeCompletion) {
  native_.fail_after_create_count = 1;
  iree_hal_slab_t slabs[3] = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        Acquire(IREE_ARRAYSIZE(slabs), slabs));
  EXPECT_EQ(native_.create_count, 1);
  EXPECT_EQ(native_.unmap_count, 1);
  EXPECT_EQ(native_.destroy_count, 1);
  for (const auto& slab : slabs) {
    EXPECT_EQ(slab.provider_handle, 0u);
  }

  native_.fail_after_create_count = -1;
  iree_hal_slab_t slab = {};
  IREE_ASSERT_OK(iree_hal_slab_provider_acquire_slab_sync(provider_, 256, 64,
                                                          &slab, nullptr));
  iree_hal_slab_provider_release_slab_sync(provider_, &slab, nullptr);
  EXPECT_EQ(native_.create_count, 2);
  EXPECT_EQ(native_.destroy_count, 2);
}

TEST_P(RegisteredSlabProviderTest,
       ConstructorAndPrefixCleanupLossSurvivesPreparationRollback) {
  native_.fail_create_at = 2;
  native_.fail_memory_destroy = true;
  native_.abandon_backing = true;
  iree_hal_slab_t slabs[3] = {};
  iree_hal_slab_release_info_t release_info = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        Acquire(IREE_ARRAYSIZE(slabs), slabs, &release_info));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL, release_info.status);
  EXPECT_EQ(release_info.abandoned_byte_length, 8192u);
  EXPECT_EQ(native_.create_count, 1);
  EXPECT_EQ(native_.destroy_count, 2);
  EXPECT_EQ(native_.abandoned_backing.size(), 2u);
  for (const auto& slab : slabs) {
    EXPECT_EQ(slab.provider_handle, 0u);
  }
}

TEST_P(RegisteredSlabProviderTest, ReleaseReportsBackingLossAfterIoDetachment) {
  iree_hal_slab_t slab = {};
  IREE_ASSERT_OK(Acquire(1, &slab));
  native_.fail_memory_destroy = true;
  native_.abandon_backing = true;
  iree_hal_slab_release_info_t release_info = {};
  iree_hal_slab_provider_release_slab_sync(provider_, &slab, &release_info);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL, release_info.status);
  EXPECT_EQ(release_info.abandoned_byte_length, 4096u);
  EXPECT_EQ(native_.destroy_count, 1);
  EXPECT_EQ(native_.unmap_count, 1);
  EXPECT_EQ(native_.abandoned_backing.size(), 1u);
}

INSTANTIATE_TEST_SUITE_P(SharedAndDistinctProactors, RegisteredSlabProviderTest,
                         ::testing::Bool());

TEST_F(SlabProviderTest, MixedReleaseDetachesImportsAndQueuesOwnedFrees) {
  const iree_hal_external_storage_class_t storage_class = {
      IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER, {0, 0}};
  iree_hal_pool_import_class_t policy = {storage_class, {1, &storage_class}};
  CreateProvider(AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
                 &policy);
  alignas(64) uint8_t bytes[2][256] = {};
  iree_hal_slab_t slabs[4];
  for (int i = 0; i < 2; ++i) {
    iree_hal_external_storage_t storage = {.storage_class = storage_class};
    storage.payload.host_pointer = bytes[i];
    storage.byte_length = sizeof(bytes[i]);
    storage.host_properties = (IREE_HAL_HOST_MEMORY_PROPERTY_READ_CACHED |
                               IREE_HAL_HOST_MEMORY_PROPERTY_WRITE_BACK);
    storage.maintenance_range = iree_hal_make_buffer_range(0, sizeof(bytes[i]));
    iree_hal_imported_slab_t imported = {};
    IREE_ASSERT_OK(
        iree_hal_slab_provider_import_slab(provider_, &storage, &imported));
    slabs[2 * i] = imported.slab;
    IREE_ASSERT_OK(iree_hal_slab_provider_acquire_slab_sync(
        provider_, 256, 64, &slabs[2 * i + 1], nullptr));
  }
  auto completion = release_completed_.get_future();
  native_.fail_memory_destroy = true;
  native_.abandon_backing = true;
  iree_hal_slab_release_request_t release = {
      .count = IREE_ARRAYSIZE(slabs),
      .slabs = slabs,
      .completion = {[](void* user_data,
                        iree_hal_slab_release_info_t release_info) {
                       IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL,
                                             release_info.status);
                       EXPECT_EQ(release_info.abandoned_byte_length, 8192u);
                       static_cast<std::promise<void>*>(user_data)->set_value();
                     },
                     &release_completed_},
  };
  const auto caller = std::this_thread::get_id();
  iree_hal_slab_provider_release_slabs(provider_, &release);
  completion.get();
  EXPECT_EQ(release.count, 2u);
  ASSERT_EQ(native_.destroy_threads.size(), 4u);
  for (size_t i = 0; i < 4; ++i) {
    const auto& destroyed = native_.destroy_threads[i];
    EXPECT_EQ(destroyed.first, i < 2);
    EXPECT_EQ(destroyed.second == caller, i < 2);
  }
  EXPECT_EQ(native_.destroy_threads[2].second,
            native_.destroy_threads[3].second);
  EXPECT_EQ(native_.abandoned_backing.size(), 2u);
}

TEST_F(SlabProviderTest, PublishesRangedBindingsAndCacheControl) {
  iree_hal_slab_provider_properties_t properties = {};
  iree_hal_slab_provider_query_properties(provider_, &properties);
  EXPECT_EQ(properties.max_allocation_size, 4096u);
  EXPECT_EQ(properties.allocation_alignment, 64u);
  EXPECT_EQ(properties.max_allocation_alignment, 4096u);
  EXPECT_EQ(properties.maintenance_alignment, 64u);

  iree_hal_slab_t slab = {};
  IREE_ASSERT_OK(iree_hal_slab_provider_acquire_slab_sync(
      provider_, 100, /*min_alignment=*/256, &slab, nullptr));
  EXPECT_EQ(slab.length, 100u);
  ASSERT_NE(slab.base_ptr, nullptr);
  EXPECT_EQ(native_.create_count, 1);
  EXPECT_EQ(native_.map_count, 1);
  ASSERT_EQ(native_.created_lengths.size(), 1u);
  EXPECT_EQ(native_.created_lengths[0], 128u);
  ASSERT_EQ(native_.created_alignments.size(), 1u);
  EXPECT_EQ(native_.created_alignments[0], 256u);

  int release_count = 0;
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_slab_provider_wrap_buffer(
      provider_, &slab, 16, 32, IREE_HAL_MEMORY_ACCESS_ALL,
      IREE_HAL_BUFFER_USAGE_DISPATCH | IREE_HAL_BUFFER_USAGE_MAPPING,
      (iree_hal_buffer_release_callback_t){
          +[](void* user_data, iree_hal_buffer_t* buffer) {
            (void)buffer;
            ++*static_cast<int*>(user_data);
          },
          &release_count,
      },
      &buffer));

  const auto host_binding = iree_hal_buffer_native_binding(
      buffer, {/*index=*/0, /*type=*/IREE_HAL_BUFFER_INTERFACE_HOST});
  EXPECT_EQ(host_binding.host_pointer, slab.base_ptr + 16);
  const auto first_device_binding = iree_hal_buffer_native_binding(
      buffer, {/*index=*/1, /*type=*/IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS});
  EXPECT_EQ(first_device_binding.device_address, UINT64_C(0x100000) + 128 + 16);
  const auto second_device_binding = iree_hal_buffer_native_binding(
      buffer, {/*index=*/2, /*type=*/IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA});
  EXPECT_EQ(second_device_binding.device_address,
            UINT64_C(0x110000) + 128 + 16);

  const uint8_t source[] = {1, 2, 3, 4};
  IREE_ASSERT_OK(iree_hal_buffer_map_write(buffer, 4, source, sizeof(source)));
  uint8_t target[sizeof(source)] = {};
  iree_hal_buffer_mapping_t mapping = {};
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
      IREE_HAL_BUFFER_MAP_FLAG_NONE, 8, sizeof(target), &mapping));
  IREE_ASSERT_OK(
      iree_hal_buffer_mapping_invalidate_range(&mapping, 0, sizeof(target)));
  memcpy(target, mapping.contents.data, sizeof(target));
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
  ASSERT_EQ(native_.cache_operations.size(), 2u);
  EXPECT_EQ(native_.cache_operations[0].operation,
            AMDF_HOST_CACHE_OPERATION_FLUSH);
  EXPECT_EQ(native_.cache_operations[0].offset, 20u);
  EXPECT_EQ(native_.cache_operations[0].length, sizeof(source));
  EXPECT_EQ(native_.cache_operations[1].operation,
            AMDF_HOST_CACHE_OPERATION_INVALIDATE);
  EXPECT_EQ(native_.cache_operations[1].offset, 24u);
  EXPECT_EQ(native_.cache_operations[1].length, sizeof(target));

  iree_hal_buffer_release(buffer);
  EXPECT_EQ(release_count, 1);
  EXPECT_EQ(native_.destroy_count, 0);
  iree_hal_slab_provider_release_slab_sync(provider_, &slab, nullptr);
  EXPECT_EQ(native_.unmap_count, 1);
  EXPECT_EQ(native_.destroy_count, 1);

  iree_hal_slab_provider_visited_set_t visited = {};
  iree_hal_slab_provider_stats_t stats = {};
  iree_hal_slab_provider_query_stats(provider_, &visited, &stats);
  EXPECT_EQ(stats.total_acquired, 1u);
  EXPECT_EQ(stats.total_released, 1u);
}

TEST_F(SlabProviderTest, NativeExtentLimitAccountsForAllocationGranularity) {
  CreateProvider(AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
                 nullptr, 4097);
  iree_hal_slab_provider_properties_t properties = {};
  iree_hal_slab_provider_query_properties(provider_, &properties);
  EXPECT_EQ(properties.max_allocation_size, 4096u);
  iree_hal_slab_t slab = {};
  IREE_ASSERT_OK(iree_hal_slab_provider_acquire_slab_sync(
      provider_, properties.max_allocation_size, 64, &slab, nullptr));
  EXPECT_EQ(slab.length, 4096u);
  iree_hal_slab_provider_release_slab_sync(provider_, &slab, nullptr);
  const int create_count = native_.create_count;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_hal_slab_provider_acquire_slab_sync(
                            provider_, 4097, 64, &slab, nullptr));
  EXPECT_EQ(native_.create_count, create_count);
}

TEST_F(SlabProviderTest, CleanupErrorDistinguishesReturnedAndAbandonedBacking) {
  native_.fail_memory_destroy = true;
  for (bool abandon : {false, true}) {
    SCOPED_TRACE(abandon);
    native_.abandon_backing = abandon;
    iree_hal_slab_t slab = {};
    IREE_ASSERT_OK(iree_hal_slab_provider_acquire_slab_sync(
        provider_, 64, /*min_alignment=*/1, &slab, nullptr));
    EXPECT_EQ(slab.native_allocation.byte_length, 4096u);
    iree_hal_slab_release_info_t released = {};
    iree_hal_slab_provider_release_slab_sync(provider_, &slab, &released);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL, released.status);
    EXPECT_EQ(released.abandoned_byte_length, abandon ? 4096u : 0u);
  }
  EXPECT_EQ(native_.unmap_count, 2);
  EXPECT_EQ(native_.destroy_count, 2);
  ASSERT_EQ(native_.abandoned_backing.size(), 1u);
  EXPECT_EQ(native_.abandoned_backing[0].size(), 4096u);
  iree_hal_slab_provider_visited_set_t visited = {};
  iree_hal_slab_provider_stats_t stats = {};
  iree_hal_slab_provider_query_stats(provider_, &visited, &stats);
  EXPECT_EQ(stats.total_acquired, 2u);
  EXPECT_EQ(stats.total_released, 1u);
}

TEST_F(SlabProviderTest, DeviceOnlyBackingHasNoHostMappingOrSlot) {
  CreateProvider(0);
  iree_hal_slab_t slab = {};
  IREE_ASSERT_OK(iree_hal_slab_provider_acquire_slab_sync(provider_, 64, 64,
                                                          &slab, nullptr));
  EXPECT_EQ(slab.base_ptr, nullptr);
  EXPECT_EQ(native_.map_count, 0);
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_slab_provider_wrap_buffer(
      provider_, &slab, 8, 32, IREE_HAL_MEMORY_ACCESS_ALL,
      IREE_HAL_BUFFER_USAGE_STORAGE, iree_hal_buffer_release_callback_null(),
      &buffer));
  EXPECT_EQ(buffer->host_binding_index,
            IREE_HAL_BUFFER_NATIVE_BINDING_INDEX_NONE);
  EXPECT_EQ(iree_hal_buffer_native_binding(
                buffer, {0, IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS})
                .device_address,
            UINT64_C(0x100000) + 64 + 8);
  EXPECT_EQ(iree_hal_buffer_native_binding(
                buffer, {1, IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA})
                .device_address,
            UINT64_C(0x110000) + 64 + 8);
  iree_hal_buffer_release(buffer);
  iree_hal_slab_provider_release_slab_sync(provider_, &slab, nullptr);
  EXPECT_EQ(native_.unmap_count, 0);
  EXPECT_EQ(native_.destroy_count, 1);
}

TEST_F(SlabProviderTest, FailedBindingReleasesMappingAndBacking) {
  native_.fail_second_address = true;
  iree_hal_slab_t slab = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_hal_slab_provider_acquire_slab_sync(
                            provider_, 64, 64, &slab, nullptr));
  EXPECT_EQ(slab.provider_handle, 0u);
  EXPECT_EQ(native_.map_count, 1);
  EXPECT_EQ(native_.unmap_count, 1);
  EXPECT_EQ(native_.destroy_count, 1);
}

TEST_F(SlabProviderTest, FailedBindingReportsUnpublishedBackingLoss) {
  native_.fail_second_address = true;
  native_.fail_memory_destroy = true;
  native_.abandon_backing = true;
  iree_hal_slab_t slab = {};
  iree_hal_slab_release_info_t released = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_hal_slab_provider_acquire_slab_sync(
                            provider_, 64, 64, &slab, &released));
  EXPECT_EQ(slab.provider_handle, 0u);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL, released.status);
  EXPECT_EQ(released.abandoned_byte_length, 4096u);
  EXPECT_EQ(native_.abandoned_backing.size(), 1u);
}

// Real provider, allocation worker, source pools, notifications, and waiters.
// Only the native dependency controls whether consumed handles return backing.
class SlabPoolTest : public SlabProviderTest {
 protected:
  void SetUp() override {
    SlabProviderTest::SetUp();
    auto options = iree_async_proactor_options_default();
    options.threading_mode = IREE_ASYNC_PROACTOR_THREADING_CROSS_THREAD;
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        options, iree_allocator_system(), &proactor_));
    IREE_ASSERT_OK(iree_async_proactor_thread_create(
        proactor_, iree_async_proactor_thread_options_default(),
        iree_allocator_system(), &thread_));
    IREE_ASSERT_OK(iree_async_notification_create(
        proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification_));
    IREE_ASSERT_OK(iree_async_frontier_tracker_create(
        iree_async_frontier_tracker_options_default(), iree_allocator_system(),
        &tracker_));
    for (auto& pool : pools_) {
      IREE_ASSERT_OK(iree_hal_passthrough_pool_create(
          {}, provider_, notification_, tracker_, iree_allocator_system(),
          &pool));
    }
  }

  void TearDown() override {
    for (auto* pool : pools_) {
      iree_hal_pool_release(pool);
    }
    SlabProviderTest::TearDown();
    iree_async_frontier_tracker_release(tracker_);
    iree_async_notification_release(notification_);
    iree_async_proactor_thread_request_stop(thread_);
    IREE_ASSERT_OK(
        iree_async_proactor_thread_join(thread_, IREE_DURATION_INFINITE));
    IREE_EXPECT_OK(iree_async_proactor_thread_consume_status(thread_));
    iree_async_proactor_thread_release(thread_);
    iree_async_proactor_release(proactor_);
  }

  static void CompleteWait(void* user_data, iree_status_t status) {
    static_cast<std::promise<iree_status_t>*>(user_data)->set_value(
        std::move(status));
  }

  void AwaitNativeReturns(iree_hal_pool_t* pool) {
    iree_hal_pool_wait_t* wait = nullptr;
    IREE_ASSERT_OK(
        iree_hal_pool_wait_allocate(pool, iree_allocator_system(), &wait));
    for (;;) {
      iree_hal_pool_wait_prepare(wait);
      const auto stats = Stats(pool);
      EXPECT_EQ(stats.reservation_count, 0u);
      if (!stats.slab_count) {
        iree_hal_pool_wait_abort(wait);
        break;
      }
      std::promise<iree_status_t> completed;
      auto completion = completed.get_future();
      iree_hal_pool_wait_commit(wait, iree_infinite_timeout(),
                                {CompleteWait, &completed});
      IREE_ASSERT_OK(completion.get());
    }
    iree_hal_pool_wait_free(wait);
  }

  static iree_hal_pool_stats_t Stats(iree_hal_pool_t* pool) {
    iree_hal_pool_stats_t stats = {};
    iree_hal_pool_query_stats(pool, &stats);
    return stats;
  }

  static void Acquire(iree_hal_pool_t* pool,
                      iree_hal_pool_reservation_t* reservation) {
    const auto request = iree_hal_make_pool_reservation_request(64, 64);
    iree_hal_pool_acquire_info_t info = {};
    iree_hal_pool_acquire_result_t result;
    IREE_ASSERT_OK(iree_hal_pool_acquire_reservations_sync(
        pool, 1, &request, nullptr, IREE_HAL_POOL_RESERVE_FLAG_NONE,
        reservation, &info, &result));
    EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  }

  static void Release(iree_hal_pool_t* pool,
                      const iree_hal_pool_reservation_t* reservation) {
    iree_hal_pool_release_reservations(pool, 1, reservation, nullptr);
  }

  // Independent capacity owners sharing the same native allocation provider.
  iree_hal_pool_t* pools_[2] = {};
  // Shared advisory wake source does not imply a shared budget or failure.
  iree_async_notification_t* notification_ = nullptr;
  // Execution retirement owner for both source pools.
  iree_async_frontier_tracker_t* tracker_ = nullptr;
  // Platform notification progress owner, retained until waits are joined.
  iree_async_proactor_t* proactor_ = nullptr;
  // Polling thread servicing accepted capacity waits.
  iree_async_proactor_thread_t* thread_ = nullptr;
};

TEST_F(SlabPoolTest, PublicReleaseJoinsNativeCleanup) {
  iree_hal_pool_reservation_t reservation = {};
  Acquire(pools_[0], &reservation);
  native_.hold_destroy = true;
  std::promise<void> resume;
  native_.destroy_resume = resume.get_future().share();
  auto entered = native_.destroy_entered.get_future();
  Release(pools_[0], &reservation);
  entered.get();

  std::thread teardown([&] { iree_hal_pool_release(pools_[0]); });
  resume.set_value();
  teardown.join();
  pools_[0] = nullptr;
  EXPECT_EQ(native_.destroy_count, 1);
}

TEST_F(SlabPoolTest, ConstructorAndPrefixLossesStayWithOriginatingPool) {
  native_.fail_create_at = 2;
  native_.fail_memory_destroy = true;
  native_.abandon_backing = true;
  const iree_hal_pool_reservation_request_t requests[] = {
      iree_hal_make_pool_reservation_request(64, 64),
      iree_hal_make_pool_reservation_request(64, 64),
      iree_hal_make_pool_reservation_request(64, 64),
  };
  iree_hal_pool_reservation_t reservations[3] = {};
  iree_hal_pool_acquire_info_t infos[3] = {};
  iree_hal_pool_acquire_result_t result;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_pool_acquire_reservations_sync(pools_[0], 3, requests, nullptr,
                                              IREE_HAL_POOL_RESERVE_FLAG_NONE,
                                              reservations, infos, &result));
  for (const auto& reservation : reservations) {
    EXPECT_EQ(reservation.block_handle, 0u);
  }
  EXPECT_EQ(native_.create_count, 1);
  EXPECT_EQ(native_.destroy_count, 2);
  EXPECT_EQ(native_.abandoned_backing.size(), 2u);
  auto stats = Stats(pools_[0]);
  EXPECT_EQ(stats.bytes_committed, 8192u);
  EXPECT_EQ(stats.bytes_abandoned, 8192u);
  EXPECT_EQ(stats.bytes_reserved, 0u);
  EXPECT_EQ(stats.slab_count, 0u);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL,
                        iree_hal_pool_query_abandonment_status(pools_[0]));
  EXPECT_EQ(Stats(pools_[1]).bytes_committed, 0u);
  EXPECT_EQ(Stats(pools_[1]).bytes_abandoned, 0u);
  IREE_EXPECT_OK(iree_hal_pool_query_abandonment_status(pools_[1]));

  native_.fail_create_at = 0;
  native_.fail_memory_destroy = false;
  for (auto* pool : pools_) {
    iree_hal_pool_reservation_t reservation = {};
    Acquire(pool, &reservation);
    Release(pool, &reservation);
    AwaitNativeReturns(pool);
  }
  EXPECT_EQ(Stats(pools_[0]).bytes_committed, 8192u);
  EXPECT_EQ(Stats(pools_[1]).bytes_committed, 0u);
}

TEST_F(SlabPoolTest,
       NativeCreditsSettleUnpublishedLossAndFundIndependentSource) {
  iree_hal_memory_budget_t* budget = nullptr;
  IREE_ASSERT_OK(iree_hal_memory_budget_create(
      12288, notification_, iree_allocator_system(), &budget));
  native_.fail_create_at = 2;
  native_.fail_memory_destroy = true;
  native_.abandon_backing = true;
  const iree_hal_pool_reservation_request_t requests[] = {
      iree_hal_make_pool_reservation_request(64, 64),
      iree_hal_make_pool_reservation_request(64, 256),
      iree_hal_make_pool_reservation_request(65, 4096),
  };
  uint64_t cost = 0;
  IREE_ASSERT_OK(
      iree_hal_slab_pool_query_allocation_cost(pools_[0], 3, requests, &cost));
  EXPECT_EQ(cost, 12288u);
  EXPECT_EQ(native_.create_count, 0);
  iree_hal_memory_credit_t credit = {};
  ASSERT_TRUE(iree_hal_memory_budget_try_reserve(budget, cost, &credit));
  iree_hal_pool_reservation_t reservations[3] = {};
  iree_hal_pool_acquire_info_t infos[3] = {};
  std::promise<iree_status_t> completed;
  auto completion = completed.get_future();
  IREE_ASSERT_OK(iree_hal_slab_pool_acquire_with_credit(
      pools_[0], 3, requests, &credit,
      {[](void* user_data, iree_status_t status,
          iree_hal_pool_acquire_result_t result) {
         EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_NONE);
         static_cast<std::promise<iree_status_t>*>(user_data)->set_value(
             std::move(status));
       },
       &completed},
      reservations, infos));
  EXPECT_EQ(credit.budget, nullptr);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, completion.get());
  EXPECT_EQ(iree_hal_memory_budget_charged(budget), 8192u);
  EXPECT_EQ(iree_hal_memory_budget_abandoned(budget), 8192u);
  EXPECT_EQ(native_.abandoned_backing.size(), 2u);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL,
                        iree_hal_memory_budget_query_loss(budget));
  for (const auto& reservation : reservations) {
    EXPECT_EQ(reservation.block_handle, 0u);
  }

  // The failed constructor did not consume its third allocation's credit.
  // A distinct source can use that capacity even though both losses remain.
  native_.fail_create_at = 0;
  native_.fail_memory_destroy = false;
  ASSERT_TRUE(iree_hal_memory_budget_try_reserve(budget, 4096, &credit));
  std::promise<iree_status_t> peer_completed;
  auto peer_completion = peer_completed.get_future();
  IREE_ASSERT_OK(iree_hal_slab_pool_acquire_with_credit(
      pools_[1], 1, requests, &credit,
      {[](void* user_data, iree_status_t status,
          iree_hal_pool_acquire_result_t result) {
         EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
         static_cast<std::promise<iree_status_t>*>(user_data)->set_value(
             std::move(status));
       },
       &peer_completed},
      reservations, infos));
  IREE_ASSERT_OK(peer_completion.get());
  EXPECT_EQ(iree_hal_memory_budget_charged(budget), 12288u);
  Release(pools_[1], reservations);
  AwaitNativeReturns(pools_[1]);
  EXPECT_EQ(iree_hal_memory_budget_charged(budget), 8192u);
  iree_hal_memory_budget_release(budget);
}

TEST_F(SlabPoolTest, CleanRollbackPreservesRecoverableAllocationFailure) {
  native_.fail_create_at = 2;
  const iree_hal_pool_reservation_request_t requests[] = {
      iree_hal_make_pool_reservation_request(64, 64),
      iree_hal_make_pool_reservation_request(64, 64),
  };
  iree_hal_pool_reservation_t reservations[2] = {};
  iree_hal_pool_acquire_info_t infos[2] = {};
  iree_hal_pool_acquire_result_t result;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_pool_acquire_reservations_sync(pools_[0], 2, requests, nullptr,
                                              IREE_HAL_POOL_RESERVE_FLAG_NONE,
                                              reservations, infos, &result));
  EXPECT_EQ(Stats(pools_[0]).bytes_committed, 0u);
  EXPECT_EQ(Stats(pools_[0]).bytes_abandoned, 0u);
  IREE_EXPECT_OK(iree_hal_pool_query_abandonment_status(pools_[0]));
  iree_hal_pool_reservation_t reservation = {};
  Acquire(pools_[0], &reservation);
  Release(pools_[0], &reservation);
}

TEST_F(SlabPoolTest, NativeLossWakesAcceptedWaitAndPreservesFullCause) {
  iree_hal_pool_reservation_t reservation = {};
  Acquire(pools_[0], &reservation);
  native_.fail_memory_destroy = true;
  native_.abandon_backing = true;
  native_.hold_destroy = true;
  std::promise<void> resume;
  native_.destroy_resume = resume.get_future().share();
  auto entered = native_.destroy_entered.get_future();
  Release(pools_[0], &reservation);
  entered.get();

  iree_hal_pool_wait_t* wait = nullptr;
  IREE_ASSERT_OK(
      iree_hal_pool_wait_allocate(pools_[0], iree_allocator_system(), &wait));
  iree_hal_pool_wait_prepare(wait);
  EXPECT_EQ(Stats(pools_[0]).bytes_reserved, 0u);
  EXPECT_EQ(Stats(pools_[0]).bytes_committed, 4096u);
  EXPECT_EQ(Stats(pools_[0]).bytes_abandoned, 0u);
  std::promise<iree_status_t> completed;
  auto completion = completed.get_future();
  iree_hal_pool_wait_commit(wait, iree_infinite_timeout(),
                            {CompleteWait, &completed});
  resume.set_value();
  IREE_EXPECT_OK(completion.get());
  iree_hal_pool_wait_free(wait);

  EXPECT_EQ(Stats(pools_[0]).bytes_committed, 4096u);
  EXPECT_EQ(Stats(pools_[0]).bytes_abandoned, 4096u);
  EXPECT_EQ(Stats(pools_[0]).slab_count, 0u);
  iree_status_t cause = iree_hal_pool_query_abandonment_status(pools_[0]);
  const std::string diagnostic = iree::Status::ToString(cause);
  EXPECT_NE(diagnostic.find("native slab release"), std::string::npos);
  EXPECT_NE(diagnostic.find("libamdf API"), std::string::npos);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL, cause);
  EXPECT_EQ(Stats(pools_[1]).bytes_abandoned, 0u);
  IREE_EXPECT_OK(iree_hal_pool_query_abandonment_status(pools_[1]));
}

TEST_F(SlabProviderTest, RegisteredStorageUsesOnlyAuthorizedEnvelope) {
  const iree_hal_external_storage_class_t storage_class = {
      IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER, {0, 0}};
  iree_hal_pool_import_class_t policy = {storage_class, {1, &storage_class}};
  CreateProvider(AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
                 &policy);
  alignas(64) uint8_t bytes[256] = {};
  iree_hal_external_storage_t storage = {
      .storage_class = storage_class,
  };
  storage.payload.host_pointer = bytes;
  storage.byte_length = sizeof(bytes);
  storage.host_properties = (IREE_HAL_HOST_MEMORY_PROPERTY_READ_CACHED |
                             IREE_HAL_HOST_MEMORY_PROPERTY_WRITE_BACK);
  storage.maintenance_range = iree_hal_make_buffer_range(64, 128);
  iree_hal_imported_slab_t imported = {};
  IREE_ASSERT_OK(
      iree_hal_slab_provider_import_slab(provider_, &storage, &imported));
  EXPECT_EQ(storage.payload.host_pointer, bytes);
  EXPECT_EQ(storage.byte_length, sizeof(bytes));
  EXPECT_EQ(imported.storage_offset, 64u);
  EXPECT_EQ(imported.slab.base_ptr, bytes + 64);
  EXPECT_EQ(imported.slab.length, 128u);
  EXPECT_EQ(imported.slab.maintenance_range.byte_offset, 0u);
  EXPECT_EQ(imported.slab.maintenance_range.byte_length, 128u);
  ASSERT_EQ(native_.created_lengths.size(), 1u);
  EXPECT_EQ(native_.created_lengths[0], 128u);
  EXPECT_GE(imported.properties.allocation_alignment, 64u);
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_slab_provider_wrap_buffer(
      provider_, &imported.slab, 0, 128, IREE_HAL_MEMORY_ACCESS_ALL,
      IREE_HAL_BUFFER_USAGE_STORAGE, iree_hal_buffer_release_callback_null(),
      &buffer));
  iree_hal_buffer_t* subspan = nullptr;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(buffer, 16, 48,
                                         iree_allocator_system(), &subspan));
  iree_hal_external_storage_t exported = {};
  iree_hal_buffer_range_t logical_range = {};
  IREE_ASSERT_OK(
      iree_hal_buffer_export(subspan, iree_hal_make_buffer_range(8, 16),
                             storage_class, &exported, &logical_range));
  EXPECT_EQ(exported.payload.host_pointer, bytes + 64);
  EXPECT_EQ(exported.byte_length, 128u);
  EXPECT_EQ(exported.host_properties,
            (IREE_HAL_HOST_MEMORY_PROPERTY_READ_CACHED |
             IREE_HAL_HOST_MEMORY_PROPERTY_WRITE_BACK));
  EXPECT_EQ(exported.release, nullptr);
  EXPECT_EQ(logical_range.byte_offset, 24u);
  EXPECT_EQ(logical_range.byte_length, 16u);
  iree_hal_external_storage_release(&exported);
  EXPECT_EQ(native_.destroy_count, 0);
  iree_hal_buffer_release(subspan);
  iree_hal_buffer_release(buffer);
  iree_hal_slab_provider_release_slab_sync(provider_, &imported.slab, nullptr);
  EXPECT_EQ(native_.unmap_count, 1);
  EXPECT_EQ(native_.destroy_count, 1);
  bytes[88] = 123;
  EXPECT_EQ(bytes[88], 123);
}

TEST_F(SlabProviderTest, FailedRegistrationPublicationPreservesCallerStorage) {
  const iree_hal_external_storage_class_t storage_class = {
      IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER, {0, 0}};
  iree_hal_pool_import_class_t policy = {storage_class, {}};
  CreateProvider(AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
                 &policy);
  alignas(64) uint8_t bytes[128] = {};
  int release_count = 0;
  iree_hal_external_storage_t storage = {
      .storage_class = storage_class,
  };
  storage.payload.host_pointer = bytes;
  storage.byte_length = sizeof(bytes);
  storage.host_properties = (IREE_HAL_HOST_MEMORY_PROPERTY_READ_CACHED |
                             IREE_HAL_HOST_MEMORY_PROPERTY_WRITE_BACK);
  storage.release = +[](void* user_data, iree_hal_external_storage_class_t,
                        iree_hal_external_storage_payload_t) {
    ++*static_cast<int*>(user_data);
  };
  storage.release_user_data = &release_count;
  iree_hal_imported_slab_t imported = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_slab_provider_import_slab(provider_, &storage, &imported));
  EXPECT_EQ(native_.create_count, 0);
  EXPECT_EQ(imported.slab.provider_handle, 0u);
  EXPECT_EQ(release_count, 0);
  storage.maintenance_range =
      iree_hal_make_buffer_range(0, IREE_HAL_WHOLE_BUFFER);
  for (const auto properties : std::initializer_list<uint32_t>{
           IREE_HAL_HOST_MEMORY_PROPERTY_NONE,
           (IREE_HAL_HOST_MEMORY_PROPERTY_READ_UNCACHED |
            IREE_HAL_HOST_MEMORY_PROPERTY_WRITE_COMBINED),
           IREE_HAL_HOST_MEMORY_PROPERTY_READ_UNCACHED}) {
    storage.host_properties = properties;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_FAILED_PRECONDITION,
        iree_hal_slab_provider_import_slab(provider_, &storage, &imported));
    EXPECT_EQ(imported.slab.provider_handle, 0u);
    EXPECT_EQ(storage.host_properties, properties);
    EXPECT_EQ(native_.create_count, 0);
    EXPECT_EQ(release_count, 0);
  }
  storage.host_properties = (IREE_HAL_HOST_MEMORY_PROPERTY_READ_CACHED |
                             IREE_HAL_HOST_MEMORY_PROPERTY_WRITE_BACK);
  native_.fail_second_address = true;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_slab_provider_import_slab(provider_, &storage, &imported));
  EXPECT_EQ(imported.slab.provider_handle, 0u);
  EXPECT_EQ(storage.payload.host_pointer, bytes);
  EXPECT_EQ(storage.byte_length, sizeof(bytes));
  EXPECT_EQ(release_count, 0);
  EXPECT_EQ(native_.unmap_count, 1);
  EXPECT_EQ(native_.destroy_count, 1);
  native_.fail_second_address = false;
  IREE_ASSERT_OK(
      iree_hal_slab_provider_import_slab(provider_, &storage, &imported));
  iree_hal_slab_provider_release_slab_sync(provider_, &imported.slab, nullptr);
  EXPECT_EQ(release_count, 0);
  iree_hal_external_storage_release(&storage);
  EXPECT_EQ(release_count, 1);
}

struct NativeTransportCase {
  // HAL transport identity admitted by the provider.
  iree_hal_external_storage_class_t storage_class;
  // Corresponding native representation.
  amdf_external_memory_type_t native_type;
};

class NativeTransportTest
    : public SlabProviderTest,
      public ::testing::WithParamInterface<NativeTransportCase> {};

TEST_P(NativeTransportTest, TransactionalImportAndIndependentExport) {
  const auto& transport = GetParam();
  native_.storage_class = transport.storage_class;
  native_.external_type = transport.native_type;
  const iree_hal_pool_import_class_t policy = {transport.storage_class,
                                               {1, &transport.storage_class}};
  CreateProvider(0, &policy);

  iree_hal_pool_import_requirements_t requirements;
  ASSERT_TRUE(iree_hal_slab_provider_query_import_requirements(
      provider_, transport.storage_class, &requirements));
  EXPECT_EQ(requirements.base_alignment, 1u);
  EXPECT_EQ(requirements.byte_length_granularity, 192u);

  int source_release_count = 0;
  iree_hal_external_storage_t storage = {
      .storage_class = transport.storage_class,
  };
  storage.payload.device_address = 1234;
  storage.byte_length = 768;
  storage.physical_backing_id = {{123, 456}};
  storage.release = +[](void* user_data, iree_hal_external_storage_class_t,
                        iree_hal_external_storage_payload_t) {
    ++*static_cast<int*>(user_data);
  };
  storage.release_user_data = &source_release_count;
  const auto original_storage = storage;
  iree_hal_imported_slab_t imported = {};
  native_.fail_second_address = true;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_slab_provider_import_slab(provider_, &storage, &imported));
  EXPECT_EQ(imported.slab.provider_handle, 0u);
  EXPECT_EQ(memcmp(&storage, &original_storage, sizeof(storage)), 0);
  EXPECT_EQ(native_.destroy_count, 1);
  EXPECT_EQ(source_release_count, 0);
  native_.fail_second_address = false;
  IREE_ASSERT_OK(
      iree_hal_slab_provider_import_slab(provider_, &storage, &imported));
  EXPECT_EQ(memcmp(&storage, &original_storage, sizeof(storage)), 0);
  EXPECT_EQ(imported.slab.length, 768u);
  EXPECT_EQ(imported.slab.base_ptr, nullptr);
  EXPECT_EQ(imported.slab.maintenance_range.byte_length, 0u);
  EXPECT_EQ(native_.import_count, 2);
  EXPECT_EQ(native_.map_count, 0);

  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_slab_provider_wrap_buffer(
      provider_, &imported.slab, 64, 256, IREE_HAL_MEMORY_ACCESS_ALL,
      IREE_HAL_BUFFER_USAGE_STORAGE, iree_hal_buffer_release_callback_null(),
      &buffer));
  iree_hal_buffer_t* subspan = nullptr;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(buffer, 32, 128,
                                         iree_allocator_system(), &subspan));
  iree_hal_external_storage_t exported = {};
  iree_hal_buffer_range_t logical_range = {};
  native_.fail_export = true;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_buffer_export(subspan, iree_hal_make_buffer_range(16, 48),
                             transport.storage_class, &exported,
                             &logical_range));
  EXPECT_EQ(exported.release, nullptr);
  native_.fail_export = false;
  IREE_ASSERT_OK(iree_hal_buffer_export(
      subspan, iree_hal_make_buffer_range(16, 48), transport.storage_class,
      &exported, &logical_range));
  EXPECT_TRUE(iree_hal_external_storage_class_equal(exported.storage_class,
                                                    transport.storage_class));
  EXPECT_EQ(exported.payload.device_address, 5678u);
  EXPECT_EQ(exported.byte_length, 768u);
  EXPECT_EQ(exported.maintenance_range.byte_length, 0u);
  EXPECT_EQ(exported.physical_backing_id.words[0], 123u);
  EXPECT_EQ(exported.physical_backing_id.words[1], 456u);
  EXPECT_EQ(logical_range.byte_offset, 112u);
  EXPECT_EQ(logical_range.byte_length, 48u);
  ASSERT_NE(exported.release, nullptr);
  iree_hal_buffer_release(subspan);
  iree_hal_buffer_release(buffer);
  iree_hal_slab_provider_release_slab_sync(provider_, &imported.slab, nullptr);
  EXPECT_EQ(native_.destroy_count, 2);
  EXPECT_EQ(native_.export_count, 1);
  EXPECT_EQ(native_.release_export_count, 0);
  EXPECT_EQ(source_release_count, 0);
  iree_hal_slab_provider_release(provider_);
  provider_ = nullptr;
  iree_hal_external_storage_release(&exported);
  EXPECT_EQ(native_.release_export_count, 1);
  iree_hal_external_storage_release(&storage);
  EXPECT_EQ(source_release_count, 1);
}

INSTANTIATE_TEST_SUITE_P(
    NativeTransports, NativeTransportTest,
    ::testing::Values(
        NativeTransportCase{{IREE_HAL_EXTERNAL_STORAGE_TYPE_DMA_BUF_FD, {0, 0}},
                            AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD},
        NativeTransportCase{{IREE_HAL_EXTERNAL_STORAGE_TYPE_OPAQUE_FD, {1, 2}},
                            AMDF_EXTERNAL_MEMORY_TYPE_OPAQUE_FD},
        NativeTransportCase{
            {IREE_HAL_EXTERNAL_STORAGE_TYPE_D3D12_RESOURCE, {0, 0}},
            AMDF_EXTERNAL_MEMORY_TYPE_D3D12_RESOURCE},
        NativeTransportCase{
            {IREE_HAL_EXTERNAL_STORAGE_TYPE_DEVICE_ADDRESS, {3, 4}},
            AMDF_EXTERNAL_MEMORY_TYPE_DEVICE_ADDRESS}));

TEST_F(SlabProviderTest, QualifiedTransportPreservesHostGuarantees) {
  const iree_hal_external_storage_class_t storage_class = {
      IREE_HAL_EXTERNAL_STORAGE_TYPE_OPAQUE_FD, {1, 2}};
  amdf_memory_profile_t profile = {};
  profile.import.byte_length_granularity = 1;
  profile.import.maximum_byte_length = 4096;
  profile.external_memory_support_count = 1;
  auto& support = profile.external_memory_support[0];
  support.type = AMDF_EXTERNAL_MEMORY_TYPE_OPAQUE_FD;
  support.provenance = {{1, 2}};
  support.flags = AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_IMPORT;
  support.byte_length_alignment = 1;
  support.host_properties = AMDF_HOST_MEMORY_PROPERTY_READ_CACHED |
                            AMDF_HOST_MEMORY_PROPERTY_WRITE_BACK;
  ASSERT_TRUE(iree_hal_amd_fabric_memory_constrain_profile(storage_class, {},
                                                           &profile));
  EXPECT_EQ(profile.host_mapping.properties, support.host_properties);
}

TEST_F(SlabProviderTest, NativeTransportGeometryCombinesAllRequiredMultiples) {
  const iree_hal_external_storage_class_t source_class = {
      IREE_HAL_EXTERNAL_STORAGE_TYPE_DMA_BUF_FD, {0, 0}};
  const iree_hal_external_storage_class_t export_class = {
      IREE_HAL_EXTERNAL_STORAGE_TYPE_OPAQUE_FD, {1, 2}};
  amdf_memory_profile_t profile = {};
  profile.import.byte_length_granularity = 64;
  profile.import.maximum_byte_length = 8192;
  profile.external_memory_support_count = 2;
  auto& source = profile.external_memory_support[0];
  source.type = AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD;
  source.flags = AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_IMPORT;
  source.byte_length_alignment = 96;
  auto& target = profile.external_memory_support[1];
  target.type = AMDF_EXTERNAL_MEMORY_TYPE_OPAQUE_FD;
  target.flags = AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_EXPORT;
  target.provenance = {{1, 2}};
  target.byte_length_alignment = 160;
  target.maximum_byte_length = 2048;
  ASSERT_TRUE(iree_hal_amd_fabric_memory_constrain_profile(
      source_class, {1, &export_class}, &profile));
  EXPECT_EQ(profile.import.byte_length_granularity, 960u);
  EXPECT_EQ(profile.import.maximum_byte_length, 2048u);
  EXPECT_EQ(iree_hal_amd_fabric_memory_external_support(
                &profile, {IREE_HAL_EXTERNAL_STORAGE_TYPE_OPAQUE_FD, {1, 3}},
                AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_EXPORT),
            nullptr);
  target.maximum_byte_length = 512;
  EXPECT_FALSE(iree_hal_amd_fabric_memory_constrain_profile(
      source_class, {1, &export_class}, &profile));
  target.maximum_byte_length = 2048;
  profile.allocation = profile.import;
  profile.allocation.maximum_byte_length = 8192;
  profile.allocation.native_byte_length_prefix = 64;
  EXPECT_FALSE(iree_hal_amd_fabric_memory_constrain_profile(
      {}, {1, &export_class}, &profile));
  target.flags |= AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_SOURCE_OFFSET;
  target.source_offset_alignment = 128;
  EXPECT_FALSE(iree_hal_amd_fabric_memory_constrain_profile(
      {}, {1, &export_class}, &profile));
  target.source_offset_alignment = 64;
  EXPECT_TRUE(iree_hal_amd_fabric_memory_constrain_profile(
      {}, {1, &export_class}, &profile));
  EXPECT_EQ(iree_hal_amd_fabric_memory_external_type(
                IREE_HAL_EXTERNAL_STORAGE_TYPE_VULKAN_OPAQUE_WIN32),
            AMDF_EXTERNAL_MEMORY_TYPE_NONE);
}

}  // namespace
