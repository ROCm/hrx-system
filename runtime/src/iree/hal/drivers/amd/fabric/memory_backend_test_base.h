// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_FABRIC_MEMORY_BACKEND_TEST_BASE_H_
#define IREE_HAL_DRIVERS_AMD_FABRIC_MEMORY_BACKEND_TEST_BASE_H_

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "iree/async/frontier_tracker.h"
#include "iree/async/notification.h"
#include "iree/async/proactor_platform.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/threading/numa.h"
#include "iree/hal/device_group.h"
#include "iree/hal/drivers/amd/fabric/memory_backend.h"
#include "iree/hal/drivers/task/device.h"
#include "iree/hal/memory/fixed_block_pool.h"
#include "iree/hal/memory/slab_cache.h"
#include "iree/hal/slab_pool.h"
#include "iree/hal/testing/mock_device.h"
#include "iree/task/executor.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amd::fabric {

using ::iree::testing::status::StatusIs;

inline iree_async_proactor_t* test_proactor() {
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
  struct Device {
    // Instance owning this native dependency.
    FakeNative* owner;
    // Stable device identity independent of its position in an access list.
    uint32_t ordinal;
  };

  struct Scope {
    // Native dependency receiving queries for this storage domain.
    FakeNative* owner;
    // Physical storage class exposed by the scope.
    amdf_memory_class_t memory_class;
    // Local storage endpoint, or UINT32_MAX for system memory.
    uint32_t device_ordinal;
    // Native profile result used to exercise absent routes and query failures.
    amdf_status_t profile_status = AMDF_STATUS_OK;
  };

  FakeNative() {
    api.memory_scope_query_info = MemoryScopeQueryInfo;
    api.memory_scope_query_device_profile = MemoryScopeQueryDeviceProfile;
    api.memory_scope_query_pair_info = MemoryScopeQueryPairInfo;
    api.memory_create = MemoryCreate;
    api.memory_query_info = MemoryQueryInfo;
    api.memory_destroy = MemoryDestroy;
    api.memory_query_address = MemoryQueryAddress;
    api.memory_map = MemoryMap;
    api.host_mapping_query_info = MappingQueryInfo;
    api.host_mapping_destroy = MappingDestroy;
    api.device_query_memory_reservation_capabilities = ReservationCapabilities;
    api.instance_query_host_memory_reservation_capabilities =
        HostReservationCapabilities;
    api.memory_reservation_create = ReservationCreate;
    api.memory_reservation_query_access_info = ReservationInfo;
    api.memory_reservation_update = ReservationUpdate;
    api.memory_reservation_detach = ReservationDetach;
    api.memory_reservation_destroy = ReservationDestroy;
    for (uint32_t i = 0; i < 2; ++i) {
      local_scopes[i] = {this, AMDF_MEMORY_CLASS_LOCAL, i};
      native_devices[i] = {this, i};
      devices[i] = reinterpret_cast<amdf_device_t*>(&native_devices[i]);
    }
  }

  amdf_memory_scope_t* scope() {
    return reinterpret_cast<amdf_memory_scope_t*>(&system_scope);
  }

  static void CheckAccesses(const Scope* storage, uint32_t access_count,
                            const amdf_memory_device_access_t* accesses) {
    const auto* self = storage->owner;
    ASSERT_LE(access_count, self->devices.size());
    bool owner_present = storage->memory_class == AMDF_MEMORY_CLASS_SYSTEM;
    for (uint32_t i = 0; i < access_count; ++i) {
      const auto* device = reinterpret_cast<const Device*>(accesses[i].device);
      ASSERT_EQ(device->owner, self);
      ASSERT_LT(device->ordinal, self->devices.size());
      owner_present |= device->ordinal == storage->device_ordinal;
      EXPECT_EQ(accesses[i].requirements.access, self->device_access);
      EXPECT_EQ(accesses[i].requirements.access & self->minimum_access,
                self->minimum_access);
      if (accesses[i].requirements.address_kinds) {
        EXPECT_EQ(accesses[i].requirements.address_kinds,
                  self->address_kinds[device->ordinal]);
        EXPECT_EQ(
            accesses[i].requirements.flags,
            AMDF_MEMORY_FLAG_DEVICE_ADDRESS |
                (self->mapping_enabled ? AMDF_MEMORY_FLAG_MAPPING_SOURCE : 0));
      } else {
        // A private allocation owner does not request an execution binding.
        EXPECT_EQ(accesses[i].requirements.flags, 0u);
      }
    }
    EXPECT_TRUE(owner_present);
  }

  static amdf_status_t AMDF_CALL MemoryScopeQueryInfo(
      amdf_memory_scope_t* base_scope, amdf_memory_scope_info_t* out_info) {
    auto* storage = reinterpret_cast<Scope*>(base_scope);
    auto* self = storage->owner;
    ++self->scope_query_count;
    out_info->kind = storage->memory_class == AMDF_MEMORY_CLASS_LOCAL
                         ? AMDF_MEMORY_SCOPE_KIND_LOCAL
                         : AMDF_MEMORY_SCOPE_KIND_SYSTEM;
    out_info->memory_profile_count = self->ProfileCount(storage);
    return AMDF_STATUS_OK;
  }

  uint32_t ProfileCount(const Scope* storage) const {
    return (registration_enabled || native_transport_enabled ? 2u : 1u) +
           (host_placement_enabled &&
                    storage->memory_class == AMDF_MEMORY_CLASS_SYSTEM
                ? 1u
                : 0u);
  }

  static amdf_status_t AMDF_CALL MemoryScopeQueryDeviceProfile(
      amdf_memory_scope_t* base_scope, uint32_t profile_ordinal,
      uint32_t access_count, const amdf_memory_device_access_t* accesses,
      amdf_memory_profile_t* out_profile,
      amdf_memory_access_capabilities_t* out_capabilities) {
    auto* storage = reinterpret_cast<Scope*>(base_scope);
    auto* self = storage->owner;
    if (storage->profile_status != AMDF_STATUS_OK) {
      return storage->profile_status;
    }
    EXPECT_LT(profile_ordinal, self->ProfileCount(storage));
    const bool placed = self->host_placement_enabled &&
                        storage->memory_class == AMDF_MEMORY_CLASS_SYSTEM &&
                        profile_ordinal == self->ProfileCount(storage) - 1;
    if (!access_count && !placed) {
      return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
    }
    CheckAccesses(storage, access_count, accesses);
    for (uint32_t i = 0; i < access_count; ++i) {
      const auto* device = reinterpret_cast<const Device*>(accesses[i].device);
      out_capabilities[i].guaranteed_access = self->minimum_access;
      out_capabilities[i].supported_access =
          AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
      out_capabilities[i].guaranteed_flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
      out_capabilities[i].supported_flags =
          AMDF_MEMORY_FLAG_DEVICE_ADDRESS |
          (self->mapping_enabled ? AMDF_MEMORY_FLAG_MAPPING_SOURCE : 0);
      out_capabilities[i].address_kinds = self->address_kinds[device->ordinal];
    }
    ++self->profile_query_count;
    out_profile->ordinal = profile_ordinal;
    out_profile->memory_class = storage->memory_class;
    out_profile->roles =
        AMDF_MEMORY_PROFILE_ROLE_CREATE |
        (self->host_mappable ? AMDF_MEMORY_PROFILE_ROLE_HOST_MAP : 0);
    out_profile->guaranteed_flags =
        (self->host_mappable ? AMDF_MEMORY_FLAG_HOST_VISIBLE : 0) |
        (storage->memory_class == AMDF_MEMORY_CLASS_LOCAL
             ? AMDF_MEMORY_FLAG_DEVICE_LOCAL
             : 0);
    out_profile->supported_flags =
        out_profile->guaranteed_flags | AMDF_MEMORY_FLAG_DEVICE_ADDRESS |
        (self->host_source_enabled ? AMDF_MEMORY_FLAG_HOST_MAPPING_SOURCE : 0);
    out_profile->allocation.maximum_byte_length = UINT64_C(1) << 30;
    out_profile->allocation.byte_length_granularity = 4096;
    out_profile->allocation.minimum_alignment = 4096;
    out_profile->allocation.maximum_alignment = UINT64_C(1) << 20;
    out_profile->allocation.native_byte_length_granularity = 4096;
    out_profile->allocation.native_byte_length_prefix = self->native_prefix;
    if (placed) {
      out_profile->allocation.placement_kinds =
          UINT32_C(1) << AMDF_MEMORY_PLACEMENT_KIND_HOST_NUMA;
    }
    out_profile->host_mapping.maximum_byte_length = UINT64_C(1) << 30;
    out_profile->host_mapping.byte_offset_granularity = 1;
    out_profile->host_mapping.byte_length_granularity = 1;
    out_profile->host_mapping.supported_access =
        !self->host_mappable ? 0
        : self->supports_private_mapping
            ? AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE
            : AMDF_MEMORY_MAP_FLAG_READ;
    if (!placed && profile_ordinal == 1 && self->registration_enabled) {
      out_profile->roles =
          AMDF_MEMORY_PROFILE_ROLE_REGISTER | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP;
      out_profile->allocation = {};
      out_profile->registration.maximum_byte_length = UINT64_C(1) << 30;
      out_profile->registration.byte_length_granularity = 1;
      out_profile->registration.registered_host_pointer_alignment = 1;
      out_profile->registration.registered_host_cacheability =
          AMDF_HOST_CACHEABILITY_WRITE_BACK;
      out_profile->registration.minimum_alignment = 1;
      out_profile->registration.maximum_alignment = 4096;
      out_profile->registration.native_byte_length_granularity = 4096;
    }
    if (!placed && self->native_transport_enabled) {
      out_profile->roles |= AMDF_MEMORY_PROFILE_ROLE_EXPORT;
      out_profile->supported_flags |= AMDF_MEMORY_FLAG_SHAREABLE;
      out_profile->external_memory_support_count = 1;
      auto& support = out_profile->external_memory_support[0];
      support.type = AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD;
      support.flags = AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_EXPORT;
      support.byte_length_alignment = 192;
      if (profile_ordinal == 1) {
        out_profile->roles &= ~AMDF_MEMORY_PROFILE_ROLE_CREATE;
        out_profile->roles |= AMDF_MEMORY_PROFILE_ROLE_IMPORT;
        out_profile->import = out_profile->allocation;
        out_profile->allocation = {};
        support.flags |= AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_IMPORT;
      }
    }
    return AMDF_STATUS_OK;
  }

  static void CheckDeviceSite(const amdf_memory_profile_pair_query_t* query,
                              const amdf_memory_profile_site_t& site) {
    ASSERT_EQ(site.kind, AMDF_MEMORY_SITE_KIND_DEVICE);
    ASSERT_LT(site.value.device.access_ordinal, query->access_count);
    const auto* device = reinterpret_cast<const Device*>(
        query->accesses[site.value.device.access_ordinal].device);
    EXPECT_EQ(site.value.device.queue_family_ordinal,
              device->owner->queue_family_ordinals[device->ordinal]);
  }

  static amdf_status_t AMDF_CALL
  MemoryScopeQueryPairInfo(amdf_memory_scope_t* base_scope,
                           const amdf_memory_profile_pair_query_t* query,
                           amdf_memory_pair_info_t* out_info) {
    auto* storage = reinterpret_cast<Scope*>(base_scope);
    auto* self = storage->owner;
    const bool registered =
        query->memory_profile_ordinal == 1 && self->registration_enabled;
    const bool imported =
        query->memory_profile_ordinal == 1 && self->native_transport_enabled;
    EXPECT_LT(query->memory_profile_ordinal, self->ProfileCount(storage));
    EXPECT_EQ(query->registered_host_cacheability,
              registered ? AMDF_HOST_CACHEABILITY_WRITE_BACK
                         : AMDF_HOST_CACHEABILITY_UNKNOWN);
    CheckAccesses(storage, query->access_count, query->accesses);
    EXPECT_EQ(query->required_flags, self->required_flags);
    EXPECT_EQ(query->external_memory_type,
              imported ? AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD
                       : AMDF_EXTERNAL_MEMORY_TYPE_NONE);
    EXPECT_EQ(query->external_memory_provenance.words[0], 0u);
    EXPECT_EQ(query->external_memory_provenance.words[1], 0u);
    ++self->pair_query_count;
    if (imported && (query->producer.kind == AMDF_MEMORY_SITE_KIND_HOST ||
                     query->consumer.kind == AMDF_MEMORY_SITE_KIND_HOST)) {
      return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
    }
    if (registered && self->registration_transition_unknown) {
      return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
    }

    out_info->flags = AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE;
    out_info->release.kind = AMDF_CACHE_TRANSITION_KIND_NONE;
    out_info->acquire.kind = AMDF_CACHE_TRANSITION_KIND_NONE;
    if (self->host_coherent && !registered &&
        (query->producer.kind == AMDF_MEMORY_SITE_KIND_HOST ||
         query->consumer.kind == AMDF_MEMORY_SITE_KIND_HOST)) {
      return AMDF_STATUS_OK;
    }
    if (self->global_host_transitions && !registered) {
      if (query->producer.kind == AMDF_MEMORY_SITE_KIND_HOST) {
        out_info->release.kind = AMDF_CACHE_TRANSITION_KIND_GLOBAL;
        out_info->release.executor = AMDF_CACHE_TRANSITION_EXECUTOR_HOST_API;
        out_info->release.host_operation = AMDF_HOST_CACHE_OPERATION_FLUSH;
      }
      if (query->consumer.kind == AMDF_MEMORY_SITE_KIND_HOST) {
        out_info->acquire.kind = AMDF_CACHE_TRANSITION_KIND_GLOBAL;
        out_info->acquire.executor = AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT;
        out_info->acquire.host_operation = AMDF_HOST_CACHE_OPERATION_INVALIDATE;
        out_info->acquire.host_fence_after = AMDF_HOST_CACHE_FENCE_X86_MFENCE;
      }
      return AMDF_STATUS_OK;
    }
    if (query->producer.kind == AMDF_MEMORY_SITE_KIND_HOST &&
        query->consumer.kind == AMDF_MEMORY_SITE_KIND_HOST) {
      out_info->flags |= AMDF_MEMORY_PAIR_FLAG_FIXED_COST_KNOWN;
    } else if (query->producer.kind == AMDF_MEMORY_SITE_KIND_HOST) {
      CheckDeviceSite(query, query->consumer);
      EXPECT_TRUE(iree_any_bit_set(query->producer.value.host_access,
                                   AMDF_MEMORY_MAP_FLAG_WRITE));
      out_info->release.kind = AMDF_CACHE_TRANSITION_KIND_RANGE;
      out_info->release.executor = AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT;
      out_info->release.host_operation = AMDF_HOST_CACHE_OPERATION_FLUSH;
      out_info->release.host_instruction =
          registered ? AMDF_HOST_CACHE_INSTRUCTION_X86_CLWB
                     : AMDF_HOST_CACHE_INSTRUCTION_X86_CLFLUSH;
      out_info->release.host_fence_after = AMDF_HOST_CACHE_FENCE_X86_MFENCE;
      out_info->release.range_granularity =
          UINT64_C(64)
          << reinterpret_cast<const Device*>(
                 query->accesses[query->consumer.value.device.access_ordinal]
                     .device)
                 ->ordinal;
    } else if (query->consumer.kind == AMDF_MEMORY_SITE_KIND_HOST) {
      CheckDeviceSite(query, query->producer);
      EXPECT_TRUE(iree_any_bit_set(query->consumer.value.host_access,
                                   AMDF_MEMORY_MAP_FLAG_READ));
      out_info->acquire.kind = AMDF_CACHE_TRANSITION_KIND_RANGE;
      out_info->acquire.executor = AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT;
      out_info->acquire.host_operation = AMDF_HOST_CACHE_OPERATION_INVALIDATE;
      out_info->acquire.host_instruction =
          AMDF_HOST_CACHE_INSTRUCTION_X86_CLFLUSH;
      out_info->acquire.host_fence_after = AMDF_HOST_CACHE_FENCE_X86_MFENCE;
      out_info->acquire.range_granularity =
          UINT64_C(64)
          << reinterpret_cast<const Device*>(
                 query->accesses[query->producer.value.device.access_ordinal]
                     .device)
                 ->ordinal;
    } else {
      CheckDeviceSite(query, query->producer);
      CheckDeviceSite(query, query->consumer);
      if (self->require_device_transition) {
        out_info->release.kind = AMDF_CACHE_TRANSITION_KIND_GLOBAL;
        out_info->release.executor = AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE;
        out_info->release.operation = AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM;
      }
    }
    return AMDF_STATUS_OK;
  }

  struct Memory {
    // Owner observing native allocation and release.
    FakeNative* owner;
    // Aligned host bytes modeling the native allocation.
    void* data;
    // Usable extent after native construction rounding.
    uint64_t byte_length;
    // Identity assigned to the native allocation.
    uint64_t identity;
    // Native placement captured for owned construction only.
    amdf_memory_placement_t placement;
    // Whether the native object borrows application storage.
    bool registered;
    // Native attachments captured in the construction request's order.
    std::vector<amdf_memory_device_access_t> accesses;
  };

  static amdf_status_t AMDF_CALL MemoryCreate(
      amdf_memory_scope_t* scope, const amdf_memory_create_info_t* info,
      amdf_memory_t** out_memory) {
    auto* storage = reinterpret_cast<Scope*>(scope);
    auto* self = storage->owner;
    self->created_scope = scope;
    self->created_profile_ordinal = info->memory_profile_ordinal;
    self->created_placement = info->placement;
    CheckAccesses(storage, info->access_count, info->accesses);
    self->created_accesses.clear();
    if (info->access_count) {
      self->created_accesses.assign(info->accesses,
                                    info->accesses + info->access_count);
    }
    EXPECT_EQ(info->required_flags, self->required_flags);
    if (self->native_transport_enabled) {
      EXPECT_EQ(info->byte_length % 12288, 0u);
    }
    const uint64_t identity = ++self->create_count;
    if (self->create_failure != AMDF_STATUS_OK) {
      return self->create_failure;
    }
    auto* memory = new Memory{self,
                              info->registered_host_pointer,
                              info->byte_length,
                              identity,
                              info->placement,
                              info->registered_host_pointer != nullptr,
                              self->created_accesses};
    if (!memory->registered) {
      IREE_CHECK_OK(iree_allocator_malloc_aligned(
          iree_allocator_system(), info->byte_length, info->minimum_alignment,
          0, &memory->data));
    }
    {
      std::lock_guard<std::mutex> lock(self->memory_mutex);
      ++self->live_memories;
    }
    *out_memory = reinterpret_cast<amdf_memory_t*>(memory);
    return AMDF_STATUS_OK;
  }

  static amdf_status_t AMDF_CALL MemoryDestroy(amdf_memory_t* handle) {
    auto* memory = reinterpret_cast<Memory*>(handle);
    auto* self = memory->owner;
    if (!memory->registered) {
      iree_allocator_free_aligned(iree_allocator_system(), memory->data);
    }
    delete memory;
    std::lock_guard<std::mutex> lock(self->memory_mutex);
    ++self->destroy_count;
    --self->live_memories;
    self->memory_condition.notify_all();
    return AMDF_STATUS_OK;
  }

  void AwaitEmpty() {
    std::unique_lock<std::mutex> lock(memory_mutex);
    memory_condition.wait(lock, [&] { return live_memories == 0; });
  }

  static amdf_status_t AMDF_CALL MemoryQueryInfo(amdf_memory_t* handle,
                                                 amdf_memory_info_t* out_info) {
    auto* memory = reinterpret_cast<Memory*>(handle);
    out_info->byte_length = memory->byte_length;
    out_info->placement = memory->placement;
    const uintptr_t pointer = reinterpret_cast<uintptr_t>(memory->data);
    out_info->alignment = memory->registered ? pointer & (~pointer + 1) : 4096;
    out_info->source_byte_offset =
        memory->registered ? pointer % 4096 : memory->owner->native_prefix;
    out_info->native_allocation_byte_length = iree_device_align(
        out_info->source_byte_offset + memory->byte_length, 4096);
    out_info->native_allocation_granularity = 4096;
    if (!memory->registered) {
      out_info->physical_backing_id = {{0xA, memory->identity}};
    }
    return AMDF_STATUS_OK;
  }

  static amdf_status_t AMDF_CALL
  MemoryQueryAddress(amdf_memory_t* handle, uint32_t access_ordinal,
                     amdf_memory_address_kind_t kind, uint64_t* out_address) {
    auto* memory = reinterpret_cast<Memory*>(handle);
    EXPECT_TRUE(iree_any_bit_set(
        memory->accesses[access_ordinal].requirements.address_kinds,
        UINT64_C(1) << kind));
    const auto* device = reinterpret_cast<const Device*>(
        memory->accesses[access_ordinal].device);
    *out_address =
        UINT64_C(0x100000) + device->ordinal * UINT64_C(0x10000) +
        (kind == AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE ? UINT64_C(0x8000) : 0);
    if (access_ordinal == 1 && memory->owner->fail_after_bindings) {
      memory->owner->fail_metadata_allocation.store(true);
    }
    return AMDF_STATUS_OK;
  }

  static amdf_status_t AMDF_CALL MemoryMap(amdf_memory_t* handle,
                                           const amdf_memory_map_info_t* info,
                                           amdf_host_mapping_t** out_mapping) {
    auto* memory = reinterpret_cast<Memory*>(handle);
    EXPECT_EQ(info->byte_length, memory->byte_length);
    ++memory->owner->map_count;
    *out_mapping = reinterpret_cast<amdf_host_mapping_t*>(memory);
    return AMDF_STATUS_OK;
  }

  static amdf_status_t AMDF_CALL MappingQueryInfo(
      amdf_host_mapping_t* mapping, amdf_host_mapping_info_t* info) {
    auto* memory = reinterpret_cast<Memory*>(mapping);
    info->pointer = memory->data;
    info->byte_length = memory->byte_length;
    info->cacheability = AMDF_HOST_CACHEABILITY_WRITE_BACK;
    return AMDF_STATUS_OK;
  }

  static amdf_status_t AMDF_CALL MappingDestroy(amdf_host_mapping_t* mapping) {
    ++reinterpret_cast<Memory*>(mapping)->owner->unmap_count;
    return AMDF_STATUS_OK;
  }

  struct Reservation {
    // Native instance borrowing its access consumers.
    FakeNative* owner;
    // Exact caller access order, deliberately independent from pool order.
    std::vector<amdf_memory_reservation_access_t> accesses;
    // Complete reserved extent.
    uint64_t byte_length;
  };

  static amdf_status_t AMDF_CALL
  ReservationCapabilities(amdf_device_t* device, uint32_t domain,
                          amdf_memory_reservation_capabilities_t* output) {
    auto* native = reinterpret_cast<Device*>(device);
    if (!native->owner->mapping_enabled) {
      return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
    }
    EXPECT_EQ(domain, 0u);
    output->supported_access =
        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
    output->required_access = AMDF_MEMORY_ACCESS_READ;
    output->maximum_byte_length = UINT64_C(1) << 40;
    output->reservation_granularity =
        native->owner->reservation_granularities[native->ordinal];
    output->maximum_alignment = UINT64_C(1) << 30;
    output->mapping_granularity =
        native->owner->reservation_granularities[native->ordinal];
    output->required_source_flags =
        native->owner->reservation_source_flags[native->ordinal];
    return AMDF_STATUS_OK;
  }

  static amdf_status_t AMDF_CALL
  HostReservationCapabilities(amdf_instance_t* instance,
                              amdf_memory_reservation_capabilities_t* output) {
    auto* self = reinterpret_cast<FakeNative*>(instance);
    if (!self->host_reservation_enabled) {
      return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
    }
    output->address_domain_ordinal = AMDF_ADDRESS_DOMAIN_ORDINAL_NONE;
    output->supported_access =
        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
    output->maximum_byte_length = UINT64_C(1) << 40;
    output->reservation_granularity = 4096;
    output->maximum_alignment = UINT64_C(1) << 30;
    output->mapping_granularity = 4096;
    output->required_source_flags = AMDF_MEMORY_FLAG_HOST_MAPPING_SOURCE;
    return AMDF_STATUS_OK;
  }

  static amdf_status_t AMDF_CALL
  ReservationCreate(amdf_instance_t* instance,
                    const amdf_memory_reservation_create_info_t* info,
                    amdf_memory_reservation_t** output) {
    EXPECT_EQ(reinterpret_cast<uintptr_t>(info->accesses) %
                  alignof(amdf_memory_reservation_access_t),
              0u);
    auto* self = reinterpret_cast<FakeNative*>(instance);
    auto* reservation =
        new Reservation{self,
                        {info->accesses, info->accesses + info->access_count},
                        info->byte_length};
    ++self->reservation_count;
    *output = reinterpret_cast<amdf_memory_reservation_t*>(reservation);
    return AMDF_STATUS_OK;
  }

  static amdf_status_t AMDF_CALL
  ReservationInfo(amdf_memory_reservation_t* handle, uint32_t ordinal,
                  amdf_memory_reservation_access_info_t* output) {
    auto* reservation = reinterpret_cast<Reservation*>(handle);
    output->ordinal = ordinal;
    output->mapping_access_ordinal =
        reservation->owner->alias_projections ? 0 : ordinal;
    output->access = reservation->accesses[ordinal].access;
    output->address = UINT64_C(0x40000000);
    output->byte_length = reservation->byte_length;
    output->mapping_granularity = 4096;
    return AMDF_STATUS_OK;
  }

  static amdf_status_t AMDF_CALL ReservationUpdate(
      amdf_memory_reservation_t* handle, uint32_t ordinal, uint32_t count,
      const amdf_memory_mapping_update_t* updates) {
    auto* reservation = reinterpret_cast<Reservation*>(handle);
    auto* device =
        reinterpret_cast<Device*>(reservation->accesses[ordinal].device);
    auto* self = reservation->owner;
    if (self->before_update) {
      self->before_update();
    }
    ++self->update_count;
    self->last_updates.assign(updates, updates + count);
    for (uint32_t i = 0; i < count; ++i) {
      if (updates[i].memory) {
        EXPECT_EQ(updates[i].memory_access_ordinal, device->ordinal);
        EXPECT_EQ(updates[i].access,
                  AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE);
      }
    }
    return self->update_failure;
  }

  static amdf_status_t AMDF_CALL
  ReservationDetach(amdf_memory_reservation_t* handle, uint32_t ordinal) {
    auto* reservation = reinterpret_cast<Reservation*>(handle);
    reservation->owner->detached_accesses.push_back(ordinal);
    return reservation->owner->detach_failure;
  }

  static amdf_status_t AMDF_CALL
  ReservationDestroy(amdf_memory_reservation_t* handle) {
    auto* reservation = reinterpret_cast<Reservation*>(handle);
    auto* self = reservation->owner;
    --self->reservation_count;
    delete reservation;
    return self->destroy_failure;
  }

  // Native consumers carry exact identity independently from aliased page
  // tables.
  std::array<Device, 2> native_devices = {};
  // Native capability admission for these dependency devices.
  bool mapping_enabled = false;
  // Native host address-space support, independent of source eligibility.
  bool host_reservation_enabled = false;
  // Whether owned backing supports independently replaceable CPU aliases.
  bool host_source_enabled = false;
  // Native address geometry per live consumer, before addresses are reserved.
  std::array<uint64_t, 2> reservation_granularities = {4096, 4096};
  // Whole-backing preparation required by each native address domain.
  std::array<amdf_memory_flags_t, 2> reservation_source_flags = {};
  // Distinct logical consumers may share a single native mapping projection.
  bool alias_projections = true;
  // Number of live address owners, independent from physical allocations.
  int reservation_count = 0;
  // Number of native vectors called by completed batches.
  int update_count = 0;
  // Copied normalized vector supplied to the native dependency.
  std::vector<amdf_memory_mapping_update_t> last_updates;
  // Native partial failure returned after accepting an update.
  amdf_status_t update_failure = AMDF_STATUS_OK;
  // Optional native work gate, used to prove independent CPU queue progress.
  std::function<void()> before_update;
  // Exact access ordinals passed to checked full removal.
  std::vector<uint32_t> detached_accesses;
  // Independent native removal result, preserving the original mapping error.
  amdf_status_t detach_failure = AMDF_STATUS_OK;
  // Consuming terminal native destruction result.
  amdf_status_t destroy_failure = AMDF_STATUS_OK;

  // Shared system scope offered by both engine descriptors.
  Scope system_scope = {this, AMDF_MEMORY_CLASS_SYSTEM, UINT32_MAX};
  // Distinct physical storage scopes for the two participating devices.
  std::array<Scope, 2> local_scopes = {};
  // Scope actually used to acquire the last backing allocation.
  amdf_memory_scope_t* created_scope = nullptr;
  // Exact allocation profile selected before allocating owned storage.
  uint32_t created_profile_ordinal = UINT32_MAX;
  // Exact native placement passed to the latest backing construction.
  amdf_memory_placement_t created_placement = {};
  // Exact native attachment list passed to the latest backing construction.
  std::vector<amdf_memory_device_access_t> created_accesses;
  // Whether this profile has any host mapping route.
  bool host_mappable = true;
  // Native payload preceding usable byte zero for owned allocations.
  uint64_t native_prefix = 0;
  // Native permissions that cannot be narrowed by the caller.
  amdf_memory_access_t minimum_access =
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
  // Expected backing requirements from the complete requested access set.
  amdf_memory_flags_t required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
  // Exact permissions the device profile and allocation must receive.
  amdf_memory_access_t device_access =
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
  // Native allocation attempts observed across the maintenance handoff.
  std::atomic<int> create_count{0};
  // Native allocation failure after the pool has selected its fixed contract.
  amdf_status_t create_failure = AMDF_STATUS_OK;
  // Native destruction count observed across allocation-worker completion.
  std::atomic<int> destroy_count{0};
  // Guards native allocation lifetime observations in the dependency.
  std::mutex memory_mutex;
  // Published when an actual native free finishes.
  std::condition_variable memory_condition;
  // Native backings remaining after successful acquisition.
  size_t live_memories = 0;
  // Explicit host-view creation count.
  std::atomic<int> map_count{0};
  // Explicit host-view destruction count.
  std::atomic<int> unmap_count{0};
  // Fake common API table consumed by the backend.
  amdf_api_t api = {};
  // Canonical native device order expected in every joint query.
  std::array<amdf_device_t*, 2> devices = {};
  // Exact native family ordinal for each canonical device.
  std::array<uint32_t, 2> queue_family_ordinals = {11, 22};
  // Exact interface required by each engine for the shared backing.
  std::array<amdf_memory_address_kinds_t, 2> address_kinds = {
      UINT64_C(1) << AMDF_MEMORY_ADDRESS_XDNA_DMA,
      UINT64_C(1) << AMDF_MEMORY_ADDRESS_XDNA_DMA};
  // Number of scope metadata queries.
  int scope_query_count = 0;
  // Number of complete live-device profile queries.
  int profile_query_count = 0;
  // Number of prospective visibility-pair queries.
  int pair_query_count = 0;
  // Whether the profile supports the provider's private read/write map.
  bool supports_private_mapping = true;
  // Whether host/device visibility is coherent in both directions.
  bool host_coherent = false;
  // Owned backing uses global API publication and fence-only observation.
  bool global_host_transitions = false;
  // Whether device visibility requires an explicit queue transition.
  bool require_device_transition = false;
  // Whether a separate host registration profile is offered.
  bool registration_enabled = false;
  // Separate host-owned allocation profile with enforceable NUMA placement.
  bool host_placement_enabled = false;
  // Native handle profile with device reach and unqualified CPU cache access.
  bool native_transport_enabled = false;
  // Missing native visibility qualification for the registration route.
  bool registration_transition_unknown = false;
  // Arms host allocation failure once native attachment/bindings are complete.
  bool fail_after_bindings = false;
  // Single metadata allocation failure consumed by the allocator dependency.
  std::atomic<bool> fail_metadata_allocation{false};
};

class MemoryBackendTest : public ::testing::Test {
 protected:
  void SetUp() override {
    allocator_ = {
        &native_,
        +[](void* self, iree_allocator_command_t command, const void* params,
            void** inout_pointer) -> iree_status_t {
          auto* native = static_cast<FakeNative*>(self);
          if (command != IREE_ALLOCATOR_COMMAND_FREE &&
              native->fail_metadata_allocation.exchange(false)) {
            return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                    "buffer metadata allocation failed");
          }
          const auto system = iree_allocator_system();
          return system.ctl(system.self, command, params, inout_pointer);
        }};
    iree_hal_slab_pool_options_initialize(&pool_options_);
    IREE_ASSERT_OK(iree_async_notification_create(
        test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification_));
    IREE_ASSERT_OK(iree_hal_amd_fabric_allocation_queue_create(
        UINT32_MAX, allocator_, &allocation_queue_));
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
      binding_descriptions_[i] = {IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA,
                                  AMDF_MEMORY_ADDRESS_XDNA_DMA};
      native_families_[i].queue_family_ordinal =
          native_.queue_family_ordinals[i];
      native_families_[i].supported_usage =
          IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH;
      native_families_[i].minimum_access =
          AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
      native_families_[i].host_usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
      native_families_[i].binding_count = 1;
      native_families_[i].bindings = &binding_descriptions_[i];
      storage_[i][0] = {
          .scope = native_.scope(),
          .memory_type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                         IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
          .minimum_access = native_.minimum_access,
      };
      storage_[i][1] = {
          .scope =
              reinterpret_cast<amdf_memory_scope_t*>(&native_.local_scopes[i]),
          .physical_device_affinity = 1,
          .memory_type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
          .minimum_access = native_.minimum_access,
      };
      iree_hal_amd_fabric_memory_backend_options_t memory_options = {
          .api = &native_.api,
          .instance = reinterpret_cast<amdf_instance_t*>(&native_),
          .device = native_.devices[i],
          .storage_count = 1,
          .storage = storage_[i].data(),
          .event_sink =
              {
                  .fn =
                      +[](void* user_data,
                          const iree_hal_device_event_t* event) {
                        auto* self = static_cast<MemoryBackendTest*>(user_data);
                        const auto* payload = reinterpret_cast<
                            const iree_hal_device_driver_failure_event_t*>(
                            event->payload.data);
                        self->native_failures_.push_back(
                            payload->backend_result_code);
                      },
                  .user_data = this,
              },
          .families = &native_families_[i],
      };
      iree_hal_amd_fabric_memory_backend_initialize(
          devices_[i], &memory_options, notification_, allocation_queue_,
          iree_hal_pool_epoch_query_null(), &backends_[i]);
    }

    host_node_id_ = iree_numa_node_for_current_thread();
    ASSERT_NE(host_node_id_, IREE_NUMA_NODE_ANY);
    iree_thread_affinity_t affinity;
    iree_thread_affinity_set_group_any(host_node_id_, &affinity);
    std::vector<iree_thread_affinity_t> affinities(cpu_worker_count_, affinity);
    iree_task_topology_t topology;
    IREE_ASSERT_OK(iree_task_topology_initialize_from_thread_affinities(
        affinities.size(), affinities.data(), &topology));
    iree_task_executor_options_t executor_options;
    iree_task_executor_options_initialize(&executor_options);
    iree_status_t status = iree_task_executor_create(
        executor_options, &topology, allocator_, &executor_);
    iree_task_topology_deinitialize(&topology);
    IREE_ASSERT_OK(status);
    IREE_ASSERT_OK(iree_hal_allocator_create_heap(allocator_, &cpu_allocator_));
    IREE_ASSERT_OK(iree_async_proactor_pool_create(
        1, nullptr, iree_async_proactor_pool_options_default(), allocator_,
        &proactor_pool_));
    auto create_params = iree_hal_device_create_params_default();
    create_params.proactor_pool = proactor_pool_;
    iree_hal_task_device_params_t task_params;
    iree_hal_task_device_params_initialize(&task_params);
    IREE_ASSERT_OK(iree_hal_task_device_create(
        IREE_SV("cpu"), &task_params, 1, &executor_, 0, nullptr, cpu_allocator_,
        &create_params, allocator_, &cpu_device_));

    iree_hal_device_group_builder_t builder;
    iree_hal_device_group_builder_initialize(&builder, tracker_);
    IREE_ASSERT_OK(
        iree_hal_device_group_builder_add_device(&builder, cpu_device_));
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
    native_.AwaitEmpty();
    iree_hal_device_group_release(group_);
    for (iree_hal_device_t* device : devices_) {
      iree_hal_device_release(device);
    }
    iree_hal_device_release(cpu_device_);
    iree_async_proactor_pool_release(proactor_pool_);
    iree_hal_allocator_release(cpu_allocator_);
    iree_task_executor_release(executor_);
    iree_async_frontier_tracker_release(tracker_);
    iree_hal_amd_fabric_allocation_queue_release(allocation_queue_);
    iree_async_notification_release(notification_);
  }

  iree_status_t CreatePool(
      iree_hal_buffer_usage_t usage, iree_hal_pool_t** out_pool,
      uint64_t interfaces = UINT64_C(1)
                            << IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA,
      iree_hal_memory_access_t host_access = IREE_HAL_MEMORY_ACCESS_ALL,
      iree_hal_buffer_usage_t cpu_usage = IREE_HAL_BUFFER_USAGE_NONE,
      iree_hal_atomic_operation_capabilities_t atomics = {},
      uint64_t cpu_interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_HOST) {
    iree_hal_pool_family_access_t accesses[3] = {};
    accesses[0].family = families_[1];
    accesses[0].usage = usage;
    accesses[0].interfaces = interfaces;
    accesses[1].family = families_[0];
    accesses[1].usage = usage;
    accesses[1].interfaces = interfaces;
    accesses[1].atomic_operations = atomics;
    accesses[2].family = iree_hal_device_queue_family(cpu_device_, 0);
    accesses[2].usage = cpu_usage;
    accesses[2].interfaces = cpu_interfaces;
    iree_hal_pool_scope_t scope = {};
    scope.family_count = cpu_usage ? 3 : 2;
    scope.families = accesses;
    scope.host.access = host_access;
    scope.host.modes = host_access ? IREE_HAL_MAPPING_MODE_SCOPED |
                                         IREE_HAL_MAPPING_MODE_PERSISTENT
                                   : 0;
    scope.host.cacheability = IREE_HAL_HOST_CACHEABILITY_UNKNOWN;
    return iree_hal_slab_pool_create(group_, scope, &pool_options_, allocator_,
                                     out_pool);
  }

  uint32_t PhysicalNode(uint32_t device_ordinal) {
    const auto* topology = iree_hal_device_group_topology(group_);
    const uint32_t group_ordinal =
        iree_hal_device_topology_info(devices_[device_ordinal])->topology_index;
    for (iree_host_size_t i = 0; i < iree_hal_topology_node_count(topology);
         ++i) {
      const auto* node = iree_hal_topology_node_at(topology, i);
      if (node->kind == IREE_HAL_TOPOLOGY_NODE_KIND_PHYSICAL_DEVICE &&
          node->device_ordinal == group_ordinal) {
        return node->ordinal;
      }
    }
    ADD_FAILURE() << "mock device has no physical topology node";
    return IREE_HAL_TOPOLOGY_NODE_ORDINAL_INVALID;
  }

  uint32_t HostNode() {
    const auto* topology = iree_hal_device_group_topology(group_);
    for (iree_host_size_t i = 0; i < iree_hal_topology_node_count(topology);
         ++i) {
      const auto* node = iree_hal_topology_node_at(topology, i);
      if (node->kind == IREE_HAL_TOPOLOGY_NODE_KIND_HOST_NUMA &&
          node->local_ordinal == host_node_id_) {
        return node->ordinal;
      }
    }
    ADD_FAILURE() << "Task device has no host NUMA topology node";
    return IREE_HAL_TOPOLOGY_NODE_ORDINAL_INVALID;
  }

  iree_hal_memory_scope_t ResolveScope(
      iree_hal_memory_site_kind_t kind,
      const iree_hal_queue_family_t* family = nullptr) {
    iree_hal_memory_site_t site = {
        .kind = kind,
        .family = family,
    };
    iree_hal_memory_scope_t scope;
    IREE_CHECK_OK(
        iree_hal_device_group_resolve_memory_scope(group_, site, &scope));
    return scope;
  }

  iree_status_t CreateReservation(iree_hal_buffer_t** out_reservation) {
    native_.mapping_enabled = true;
    binding_descriptions_[0] = {IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
                                AMDF_MEMORY_ADDRESS_GPU};
    native_families_[0].host_usage = 0;
    const iree_hal_pool_family_access_t family = {
        .family = families_[0],
        .usage = IREE_HAL_BUFFER_USAGE_STORAGE,
    };
    const iree_hal_pool_scope_t scope = {1, &family, {}};
    const iree_hal_virtual_buffer_options_t options = {
        .byte_length = 16384,
    };
    return iree_hal_virtual_buffer_create(group_, scope, &options, allocator_,
                                          out_reservation);
  }

  // Native failures delivered through the production diagnostic sink.
  std::vector<uint64_t> native_failures_;
  // Host allocator shared by all test owners.
  iree_allocator_t allocator_;
  // Public construction policy used by each production pool creation.
  iree_hal_slab_pool_options_t pool_options_;
  // Fake libamdf scope and its query observations.
  FakeNative native_;
  // Engine-supplied native address interface for each device.
  std::array<iree_hal_amd_fabric_memory_binding_t, 2> binding_descriptions_ =
      {};
  // Family facts borrowed by their corresponding backends.
  std::array<iree_hal_amd_fabric_memory_family_t, 2> native_families_ = {};
  // Stable system/local storage facts borrowed by each engine backend.
  std::array<std::array<iree_hal_amd_fabric_memory_storage_t, 2>, 2> storage_ =
      {};
  // Backends borrowed by their corresponding mock devices.
  std::array<iree_hal_amd_fabric_memory_backend_t, 2> backends_ = {};
  // Mock HAL devices in canonical group order.
  std::array<iree_hal_device_t*, 2> devices_ = {};
  // Canonical family identity for each mock device.
  std::array<const iree_hal_queue_family_t*, 2> families_ = {};
  // Real CPU participant, ordered before native devices in the group.
  iree_hal_device_t* cpu_device_ = nullptr;
  // Existing CPU workers available for native mapping and independent
  // execution.
  iree_host_size_t cpu_worker_count_ = 1;
  // Real physical node shared by the CPU participant's queue workers.
  uint32_t host_node_id_ = IREE_NUMA_NODE_ANY;
  // Existing CPU worker, allocator and proactor services.
  iree_task_executor_t* executor_ = nullptr;
  // CPU device's ordinary allocation source.
  iree_hal_allocator_t* cpu_allocator_ = nullptr;
  // CPU asynchronous progress owner.
  iree_async_proactor_pool_t* proactor_pool_ = nullptr;
  // Capacity notification retained by each created pool.
  iree_async_notification_t* notification_ = nullptr;
  // Cold native allocation and retirement executor.
  iree_hal_amd_fabric_allocation_queue_t* allocation_queue_ = nullptr;
  // Group completion coordinate owner.
  iree_async_frontier_tracker_t* tracker_ = nullptr;
  // Sealed device group assigning memory scope identities.
  iree_hal_device_group_t* group_ = nullptr;
};

}  // namespace iree::hal::amd::fabric

#endif  // IREE_HAL_DRIVERS_AMD_FABRIC_MEMORY_BACKEND_TEST_BASE_H_
