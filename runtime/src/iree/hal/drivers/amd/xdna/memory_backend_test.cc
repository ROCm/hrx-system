// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/memory_backend.h"

#include <array>
#include <cstdint>

#include "iree/async/frontier_tracker.h"
#include "iree/async/notification.h"
#include "iree/async/proactor_platform.h"
#include "iree/hal/device_group.h"
#include "iree/hal/memory/maintenance_thread.h"
#include "iree/hal/slab_pool.h"
#include "iree/hal/testing/mock_device.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using ::iree::testing::status::StatusIs;

static iree_async_proactor_t* test_proactor() {
  static iree_async_proactor_t* proactor = nullptr;
  if (!proactor) {
    IREE_CHECK_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &proactor));
    atexit([] {
      iree_async_proactor_release(proactor);
      proactor = nullptr;
    });
  }
  return proactor;
}

struct FakeNative {
  FakeNative() {
    api.memory_scope_query_info = MemoryScopeQueryInfo;
    api.memory_scope_query_device_profile = MemoryScopeQueryDeviceProfile;
    api.memory_scope_query_pair_info = MemoryScopeQueryPairInfo;
    devices[0] = reinterpret_cast<amdf_device_t*>(uintptr_t{0xA});
    devices[1] = reinterpret_cast<amdf_device_t*>(uintptr_t{0xB});
  }

  amdf_memory_scope_t* scope() {
    return reinterpret_cast<amdf_memory_scope_t*>(this);
  }

  static amdf_status_t AMDF_CALL MemoryScopeQueryInfo(
      amdf_memory_scope_t* base_scope, amdf_memory_scope_info_t* out_info) {
    auto* self = reinterpret_cast<FakeNative*>(base_scope);
    ++self->scope_query_count;
    out_info->kind = AMDF_MEMORY_SCOPE_KIND_SYSTEM;
    out_info->memory_profile_count = 1;
    return AMDF_STATUS_OK;
  }

  static amdf_status_t AMDF_CALL MemoryScopeQueryDeviceProfile(
      amdf_memory_scope_t* base_scope, uint32_t profile_ordinal,
      uint32_t access_count, const amdf_memory_device_access_t* accesses,
      amdf_memory_profile_t* out_profile,
      amdf_memory_access_capabilities_t* out_capabilities) {
    auto* self = reinterpret_cast<FakeNative*>(base_scope);
    EXPECT_EQ(profile_ordinal, 0u);
    EXPECT_EQ(access_count, 2u);
    EXPECT_EQ(accesses[0].device, self->devices[0]);
    EXPECT_EQ(accesses[1].device, self->devices[1]);
    for (uint32_t i = 0; i < access_count; ++i) {
      EXPECT_EQ(accesses[i].requirements.access,
                AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE);
      EXPECT_EQ(accesses[i].requirements.flags,
                AMDF_MEMORY_FLAG_DEVICE_ADDRESS);
      EXPECT_EQ(accesses[i].requirements.address_kinds,
                UINT64_C(1) << AMDF_MEMORY_ADDRESS_XDNA_DMA);
      out_capabilities[i].guaranteed_access =
          AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
      out_capabilities[i].supported_access =
          AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
      out_capabilities[i].guaranteed_flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
      out_capabilities[i].supported_flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
      out_capabilities[i].address_kinds = UINT64_C(1)
                                          << AMDF_MEMORY_ADDRESS_XDNA_DMA;
    }
    ++self->profile_query_count;
    out_profile->ordinal = profile_ordinal;
    out_profile->memory_class = AMDF_MEMORY_CLASS_SYSTEM;
    out_profile->roles =
        AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP;
    out_profile->guaranteed_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
    out_profile->supported_flags =
        AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
    out_profile->allocation.maximum_byte_length = UINT64_C(1) << 30;
    out_profile->allocation.byte_length_granularity = 4096;
    out_profile->allocation.minimum_alignment = 4096;
    out_profile->allocation.maximum_alignment = UINT64_C(1) << 20;
    out_profile->allocation.native_byte_length_granularity = 4096;
    out_profile->host_mapping.maximum_byte_length = UINT64_C(1) << 30;
    out_profile->host_mapping.byte_offset_granularity = 1;
    out_profile->host_mapping.byte_length_granularity = 1;
    out_profile->host_mapping.supported_access =
        self->supports_private_mapping
            ? AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE
            : AMDF_MEMORY_MAP_FLAG_READ;
    return AMDF_STATUS_OK;
  }

  static void CheckDeviceSite(const FakeNative* self,
                              const amdf_memory_profile_site_t& site) {
    ASSERT_EQ(site.kind, AMDF_MEMORY_SITE_KIND_DEVICE);
    ASSERT_LT(site.value.device.access_ordinal, self->devices.size());
    EXPECT_EQ(site.value.device.queue_family_ordinal,
              self->queue_family_ordinals[site.value.device.access_ordinal]);
  }

  static amdf_status_t AMDF_CALL
  MemoryScopeQueryPairInfo(amdf_memory_scope_t* base_scope,
                           const amdf_memory_profile_pair_query_t* query,
                           amdf_memory_pair_info_t* out_info) {
    auto* self = reinterpret_cast<FakeNative*>(base_scope);
    EXPECT_EQ(query->memory_profile_ordinal, 0u);
    EXPECT_EQ(query->access_count, 2u);
    EXPECT_EQ(query->required_flags, AMDF_MEMORY_FLAG_HOST_VISIBLE);
    EXPECT_EQ(query->accesses[0].device, self->devices[0]);
    EXPECT_EQ(query->accesses[1].device, self->devices[1]);
    ++self->pair_query_count;

    out_info->flags = AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE;
    out_info->release.kind = AMDF_CACHE_TRANSITION_KIND_NONE;
    out_info->acquire.kind = AMDF_CACHE_TRANSITION_KIND_NONE;
    if (query->producer.kind == AMDF_MEMORY_SITE_KIND_HOST) {
      CheckDeviceSite(self, query->consumer);
      EXPECT_EQ(query->producer.value.host_access,
                AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE);
      out_info->release.kind = AMDF_CACHE_TRANSITION_KIND_RANGE;
      out_info->release.executor = AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT;
      out_info->release.host_operation = AMDF_HOST_CACHE_OPERATION_FLUSH;
      out_info->release.host_instruction =
          AMDF_HOST_CACHE_INSTRUCTION_X86_CLFLUSH;
      out_info->release.host_fence_after = AMDF_HOST_CACHE_FENCE_X86_MFENCE;
      out_info->release.range_granularity =
          UINT64_C(64) << query->consumer.value.device.access_ordinal;
    } else if (query->consumer.kind == AMDF_MEMORY_SITE_KIND_HOST) {
      CheckDeviceSite(self, query->producer);
      EXPECT_EQ(query->consumer.value.host_access,
                AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE);
      out_info->acquire.kind = AMDF_CACHE_TRANSITION_KIND_RANGE;
      out_info->acquire.executor = AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT;
      out_info->acquire.host_operation = AMDF_HOST_CACHE_OPERATION_INVALIDATE;
      out_info->acquire.host_instruction =
          AMDF_HOST_CACHE_INSTRUCTION_X86_CLFLUSH;
      out_info->acquire.host_fence_after = AMDF_HOST_CACHE_FENCE_X86_MFENCE;
      out_info->acquire.range_granularity =
          UINT64_C(64) << query->producer.value.device.access_ordinal;
    } else {
      CheckDeviceSite(self, query->producer);
      CheckDeviceSite(self, query->consumer);
      if (self->require_device_transition) {
        out_info->release.kind = AMDF_CACHE_TRANSITION_KIND_GLOBAL;
        out_info->release.executor = AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE;
        out_info->release.operation = AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM;
      }
    }
    return AMDF_STATUS_OK;
  }

  // Fake common API table consumed by the backend.
  amdf_api_t api = {};
  // Canonical native device order expected in every joint query.
  std::array<amdf_device_t*, 2> devices = {};
  // Exact native family ordinal for each canonical device.
  std::array<uint32_t, 2> queue_family_ordinals = {11, 22};
  // Number of scope metadata queries.
  int scope_query_count = 0;
  // Number of complete live-device profile queries.
  int profile_query_count = 0;
  // Number of prospective visibility-pair queries.
  int pair_query_count = 0;
  // Whether the profile supports the provider's private read/write map.
  bool supports_private_mapping = true;
  // Whether device visibility requires an unrepresentable queue transition.
  bool require_device_transition = false;
};

class MemoryBackendTest : public ::testing::Test {
 protected:
  void SetUp() override {
    allocator_ = iree_allocator_system();
    IREE_ASSERT_OK(iree_async_notification_create(
        test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification_));
    IREE_ASSERT_OK(iree_hal_memory_maintenance_thread_create({}, allocator_,
                                                             &maintenance_));
    IREE_ASSERT_OK(iree_async_frontier_tracker_create(
        iree_async_frontier_tracker_options_default(), allocator_, &tracker_));

    for (iree_host_size_t i = 0; i < devices_.size(); ++i) {
      iree_hal_mock_device_options_t options;
      iree_hal_mock_device_options_initialize(&options);
      options.identifier = i == 0 ? IREE_SV("xdna-a") : IREE_SV("xdna-b");
      options.executable_loading_enabled = true;
      options.memory_backend = &backends_[i].base;
      IREE_ASSERT_OK(
          iree_hal_mock_device_create(&options, allocator_, &devices_[i]));
      contexts_[i].api = &native_.api;
      contexts_[i].device = native_.devices[i];
      contexts_[i].queue_family_ordinal = native_.queue_family_ordinals[i];
      contexts_[i].data_source.scope = native_.scope();
      iree_hal_amd_xdna_memory_backend_initialize(
          devices_[i], &contexts_[i], notification_, maintenance_,
          iree_hal_pool_epoch_query_null(), &backends_[i]);
    }

    iree_hal_device_group_builder_t builder;
    iree_hal_device_group_builder_initialize(&builder, tracker_);
    for (iree_hal_device_t* device : devices_) {
      IREE_ASSERT_OK(
          iree_hal_device_group_builder_add_device(&builder, device));
    }
    IREE_ASSERT_OK(
        iree_hal_device_group_builder_finalize(&builder, allocator_, &group_));
    for (iree_host_size_t i = 0; i < devices_.size(); ++i) {
      families_[i] = iree_hal_device_queue_family(devices_[i], 0);
      ASSERT_NE(families_[i], nullptr);
    }
  }

  void TearDown() override {
    iree_hal_device_group_release(group_);
    for (iree_hal_device_t* device : devices_) {
      iree_hal_device_release(device);
    }
    iree_async_frontier_tracker_release(tracker_);
    iree_hal_memory_maintenance_release(maintenance_);
    iree_async_notification_release(notification_);
  }

  iree_status_t CreatePool(iree_hal_buffer_usage_t usage,
                           iree_hal_pool_t** out_pool) {
    iree_hal_pool_family_access_t accesses[2] = {};
    accesses[0].family = families_[1];
    accesses[0].usage = usage;
    accesses[0].interfaces = UINT64_C(1)
                             << IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA;
    accesses[1].family = families_[0];
    accesses[1].usage = usage;
    accesses[1].interfaces = UINT64_C(1)
                             << IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA;
    iree_hal_pool_scope_t scope = {};
    scope.family_count = IREE_ARRAYSIZE(accesses);
    scope.families = accesses;
    scope.host.access = IREE_HAL_MEMORY_ACCESS_ALL;
    scope.host.modes =
        IREE_HAL_MAPPING_MODE_SCOPED | IREE_HAL_MAPPING_MODE_PERSISTENT;
    scope.host.cacheability = IREE_HAL_HOST_CACHEABILITY_UNKNOWN;
    iree_hal_slab_pool_options_t options;
    iree_hal_slab_pool_options_initialize(&options);
    return iree_hal_slab_pool_create(group_, scope, &options, allocator_,
                                     out_pool);
  }

  iree_hal_memory_scope_t ResolveScope(
      iree_hal_memory_site_kind_t kind,
      const iree_hal_queue_family_t* family = nullptr) {
    iree_hal_memory_site_t site = {};
    site.kind = kind;
    site.family = family;
    iree_hal_memory_scope_t scope;
    IREE_CHECK_OK(
        iree_hal_device_group_resolve_memory_scope(group_, site, &scope));
    return scope;
  }

  // Host allocator shared by all test owners.
  iree_allocator_t allocator_;
  // Fake libamdf scope and its query observations.
  FakeNative native_;
  // Native contexts borrowing the fake API and device handles.
  std::array<iree_hal_amd_xdna_context_t, 2> contexts_ = {};
  // Backends borrowed by their corresponding mock devices.
  std::array<iree_hal_amd_xdna_memory_backend_t, 2> backends_ = {};
  // Mock HAL devices in canonical group order.
  std::array<iree_hal_device_t*, 2> devices_ = {};
  // Canonical family identity for each mock device.
  std::array<const iree_hal_queue_family_t*, 2> families_ = {};
  // Capacity notification retained by each created pool.
  iree_async_notification_t* notification_ = nullptr;
  // Cold native allocation and retirement executor.
  iree_hal_memory_maintenance_t* maintenance_ = nullptr;
  // Group completion coordinate owner.
  iree_async_frontier_tracker_t* tracker_ = nullptr;
  // Sealed device group assigning memory scope identities.
  iree_hal_device_group_t* group_ = nullptr;
};

TEST_F(MemoryBackendTest, PublishesCanonicalPerDeviceBindings) {
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(CreatePool(
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_STORAGE, &pool));

  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(pool, &capabilities);
  EXPECT_EQ(capabilities.maintenance_alignment, 128u);
  EXPECT_TRUE(iree_all_bits_set(capabilities.memory_type,
                                IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                                    IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
                                    IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
                                    IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE));
  EXPECT_FALSE(iree_any_bit_set(capabilities.memory_type,
                                IREE_HAL_MEMORY_TYPE_HOST_COHERENT));
  EXPECT_EQ(iree_hal_pool_query_host_access(pool).access,
            IREE_HAL_MEMORY_ACCESS_ALL);

  for (uint16_t device_index = 0; device_index < families_.size();
       ++device_index) {
    iree_hal_memory_scope_t program_scope;
    iree_hal_memory_site_t site = {};
    site.kind = IREE_HAL_MEMORY_SITE_PROGRAM;
    site.family = families_[device_index];
    IREE_ASSERT_OK(iree_hal_device_group_resolve_memory_scope(group_, site,
                                                              &program_scope));
    iree_hal_buffer_native_binding_slot_t slot;
    IREE_ASSERT_OK(iree_hal_pool_resolve_binding(
        pool, program_scope, IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA, &slot));
    EXPECT_EQ(slot.index, device_index + 1);
    EXPECT_EQ(slot.type, IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA);
  }

  const iree_hal_memory_transition_table_t table =
      iree_hal_pool_transition_table(pool);
  const iree_hal_memory_scope_t host = ResolveScope(IREE_HAL_MEMORY_SITE_HOST);
  const iree_hal_memory_scope_t queue_a =
      ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, families_[0]);
  const iree_hal_memory_scope_t queue_b =
      ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, families_[1]);
  const iree_hal_memory_scope_t program_a =
      ResolveScope(IREE_HAL_MEMORY_SITE_PROGRAM, families_[0]);
  iree_hal_memory_transition_pair_t host_to_a;
  IREE_ASSERT_OK(iree_hal_memory_transition_prepare_pair(
      table, host, queue_a, IREE_HAL_MEMORY_TRANSITION_RELEASE, &host_to_a));
  const iree_hal_memory_transition_t publication =
      iree_hal_memory_transition_query(table, host_to_a);
  EXPECT_EQ(publication.release.bits, IREE_HAL_MEMORY_EFFECT_HOST_FLUSH);
  EXPECT_EQ(publication.acquire.bits, 0u);
  const iree_hal_memory_transition_recipe_t* publication_recipe =
      iree_hal_memory_transition_recipe(table, host_to_a,
                                        IREE_HAL_MEMORY_TRANSITION_RELEASE);
  ASSERT_NE(publication_recipe, nullptr);
  ASSERT_EQ(publication_recipe->operation_count, 1u);
  EXPECT_EQ(publication_recipe->operations[0].range_granularity, 64u);
  EXPECT_EQ(publication_recipe->operations[0].host.instruction,
            IREE_HAL_HOST_CACHE_INSTRUCTION_X86_CLFLUSH);
  EXPECT_EQ(publication_recipe->operations[0].host.fence_after,
            IREE_HAL_HOST_CACHE_FENCE_X86_MFENCE);

  iree_hal_memory_transition_pair_t b_to_host;
  IREE_ASSERT_OK(iree_hal_memory_transition_prepare_pair(
      table, queue_b, host, IREE_HAL_MEMORY_TRANSITION_ACQUIRE, &b_to_host));
  const iree_hal_memory_transition_t observation =
      iree_hal_memory_transition_query(table, b_to_host);
  EXPECT_EQ(observation.release.bits, 0u);
  EXPECT_EQ(observation.acquire.bits, IREE_HAL_MEMORY_EFFECT_HOST_INVALIDATE);
  const iree_hal_memory_transition_t unqualified_program =
      iree_hal_pool_query_transition(pool, host, program_a);
  EXPECT_FALSE(
      iree_hal_memory_effects_is_supported(unqualified_program.release));
  EXPECT_FALSE(
      iree_hal_memory_effects_is_supported(unqualified_program.acquire));

  EXPECT_EQ(native_.scope_query_count, 1);
  EXPECT_EQ(native_.profile_query_count, 1);
  EXPECT_EQ(native_.pair_query_count, 8);
  iree_hal_pool_release(pool);
}

TEST_F(MemoryBackendTest, PublishesQueueGlobalDeviceTransition) {
  native_.require_device_transition = true;
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool));
  const iree_hal_memory_transition_table_t table =
      iree_hal_pool_transition_table(pool);
  const iree_hal_memory_scope_t queue_a =
      ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, families_[0]);
  const iree_hal_memory_scope_t queue_b =
      ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, families_[1]);
  iree_hal_memory_transition_pair_t pair;
  IREE_ASSERT_OK(iree_hal_memory_transition_prepare_pair(
      table, queue_a, queue_b, IREE_HAL_MEMORY_TRANSITION_RELEASE, &pair));
  const iree_hal_memory_transition_t transition =
      iree_hal_memory_transition_query(table, pair);
  EXPECT_EQ(transition.release.bits,
            IREE_HAL_MEMORY_EFFECT_GLOBAL_RELEASE_TO_SYSTEM);
  EXPECT_EQ(transition.acquire.bits, 0u);
  EXPECT_EQ(iree_hal_memory_transition_recipe(
                table, pair, IREE_HAL_MEMORY_TRANSITION_RELEASE),
            nullptr);
  iree_hal_pool_release(pool);
}

TEST_F(MemoryBackendTest, RejectsProfileWithoutPrivateReadWriteMapping) {
  native_.supports_private_mapping = false;
  iree_hal_pool_t* pool = nullptr;
  iree_status_t status = CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool);
  EXPECT_THAT(status, StatusIs(iree::StatusCode::kUnavailable));
  iree_status_free(status);
  EXPECT_EQ(pool, nullptr);
  EXPECT_EQ(native_.pair_query_count, 0);
}

TEST_F(MemoryBackendTest, RejectsUnsupportedUsage) {
  iree_hal_pool_t* pool = nullptr;
  iree_status_t status =
      CreatePool(IREE_HAL_BUFFER_USAGE_DISPATCH_UNIFORM_READ, &pool);
  EXPECT_THAT(status, StatusIs(iree::StatusCode::kUnavailable));
  iree_status_free(status);
  EXPECT_EQ(pool, nullptr);
  EXPECT_EQ(native_.profile_query_count, 0);
}

}  // namespace
