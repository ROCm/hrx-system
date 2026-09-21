// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_XDNA_CTS_EXECUTION_FIXTURE_H_
#define IREE_EXPERIMENTAL_XDNA_CTS_EXECUTION_FIXTURE_H_

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "experimental/xdna/executable.h"
#include "iree/base/internal/shm.h"
#include "iree/hal/drivers/amd/xdna/image/aie2p/npu2.h"
#include "iree/hal/drivers/amd/xdna/image/testdata/mul_i32.h"
#include "iree/hal/drivers/amd/xdna/image/testdata/mul_i32_npu4.h"
#include "iree/hal/drivers/amd/xdna/image/testing/image_fixture.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "libamdf/cts/xdna/xdna_device_fixture.h"

namespace iree::experimental::xdna::testing {

// Both canonical compiler fixtures multiply sixteen low-32-bit integer pairs.
constexpr size_t kElementCount = 16;
constexpr size_t kBindingByteLength = kElementCount * sizeof(uint32_t);
constexpr size_t kBindingByteOffset = kBindingByteLength;
constexpr size_t kBindingStorageByteLength = 3 * kBindingByteLength;
constexpr uint8_t kGuardValue = 0xA5;
using BindingValues = std::array<uint32_t, kElementCount>;
using ResolvedBindings = std::array<iree_hal_amd_xdna_executable_binding_t, 3>;

// Unsigned arithmetic gives exact low bits for signed and overflowing products.
constexpr BindingValues kValues = {
    0,          1,          2,          3,          7,          31,
    65535,      65536,      0x7FFFFFFF, 0x80000000, 0x80000001, 0xFFFFFFFD,
    0xFFFFFFFE, 0xFFFFFFFF, 0x12345678, 0x87654321};

class XdnaExecutionFixture : public XdnaDeviceFixture {
 protected:
  struct MappedMemory {
    // Owned allocation, or null for a view borrowing separately owned backing.
    amdf_memory_t* memory = nullptr;
    // Explicit host mapping, destroyed before the allocation.
    amdf_host_mapping_t* mapping = nullptr;
    // Borrowed host view of the mapped range.
    uint8_t* pointer = nullptr;
  };

  struct Execution {
    // Case-owned context borrowing the shared ordinary device.
    amdf_xdna_context_t* context = nullptr;
    // Private instruction allocation, released before its exact context.
    MappedMemory instructions;
    // Mapped extent covering the entry's command storage.
    iree_host_size_t byte_length = 0;
    // Complete setup and execution command over immutable instruction backing.
    amdf_xdna_kernel_command_t command = {};
    // Published bytes retained to verify command immutability after execution.
    std::vector<uint8_t> original_instructions;
    // Native transport lease borrowing this execution's context.
    amdf_kernel_queue_t* queue = nullptr;
  };

  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(XdnaDeviceFixture::SetUp());
    amdf_xdna_endpoint_info_t info = {};
    info.type = AMDF_STRUCTURE_TYPE_XDNA_ENDPOINT_INFO;
    info.structure_size = sizeof(info);
    ASSERT_EQ(xdna_api_->endpoint_query_info(endpoint_, &info), AMDF_STATUS_OK);
    amdf_xdna_device_info_t device_info = {};
    device_info.type = AMDF_STRUCTURE_TYPE_XDNA_DEVICE_INFO;
    device_info.structure_size = sizeof(device_info);
    ASSERT_EQ(xdna_api_->device_query_info(device_, &device_info),
              AMDF_STATUS_OK);
    ASSERT_NE(device_info.context.scheduling_modes &
                  AMDF_XDNA_SCHEDULING_MODE_TIME_SLICED,
              0u)
        << "required time-sliced XDNA contexts are unavailable";
    instruction_alignment_ = device_info.instruction.address_alignment;

    iree_hal_amd_xdna_aie2p_target_t target;
    IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_npu2_target_initialize(
        iree_make_cstring_view(info.target_id), 1, &target));
    const iree_file_toc_t* image = nullptr;
    if (target.identity.device_profile_id == UINT64_C(0x5354524958000001)) {
      image = iree_hal_amd_xdna_test_mul_i32_npu4_create();
    } else if (target.identity.device_profile_id ==
               UINT64_C(0x535848414C4F0001)) {
      image = iree_hal_amd_xdna_test_mul_i32_create();
    } else {
      FAIL() << "no canonical multiplication fixture for " << info.target_id;
    }
    amdf_endpoint_info_t endpoint_info = {};
    endpoint_info.type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO;
    endpoint_info.structure_size = sizeof(endpoint_info);
    ASSERT_EQ(api_->endpoint_query_info(endpoint_, &endpoint_info),
              AMDF_STATUS_OK);
    uint32_t family_ordinal = UINT32_MAX;
    for (uint32_t i = 0; i < endpoint_info.queue_family_count; ++i) {
      amdf_queue_family_info_t family = {};
      family.type = AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO;
      family.structure_size = sizeof(family);
      ASSERT_EQ(api_->endpoint_query_queue_family_info(endpoint_, i, &family),
                AMDF_STATUS_OK);
      if (family.command_type == AMDF_QUEUE_COMMAND_TYPE_XDNA &&
          (family.publication_modes & AMDF_QUEUE_PUBLICATION_MODE_KERNEL) !=
              0) {
        family_ordinal = i;
        break;
      }
    }
    ASSERT_NE(family_ordinal, UINT32_MAX);
    queue_family_ordinal_ = family_ordinal;
    const auto* image_bytes = reinterpret_cast<const uint8_t*>(image->data);
    auto sequence = iree::hal::amd::xdna::testing::MakeOwnedByteSequence(
        std::vector<uint8_t>(image_bytes, image_bytes + image->size));
    IREE_ASSERT_OK(iree_hal_amd_xdna_image_create(
        sequence.get(), &target, iree_allocator_system(), &executable_));
    IREE_ASSERT_OK(iree_hal_amd_xdna_image_find_entry(
        executable_, IREE_SV("mul_i32"), &entry_ordinal_));
  }

  void DestroyMemory(MappedMemory* memory) {
    if (memory->mapping) {
      ASSERT_EQ(api_->host_mapping_destroy(memory->mapping), AMDF_STATUS_OK);
      memory->mapping = nullptr;
      memory->pointer = nullptr;
    }
    if (memory->memory) {
      ASSERT_EQ(api_->memory_destroy(std::exchange(memory->memory, nullptr)),
                AMDF_STATUS_OK);
    }
  }

  void DestroyExecution(Execution* execution) {
    if (execution->queue) {
      const auto status = api_->kernel_queue_destroy(execution->queue);
      if (status != amdf_make_api_status(AMDF_STATUS_CODE_BUSY)) {
        execution->queue = nullptr;
      }
      ASSERT_EQ(status, AMDF_STATUS_OK);
    }
    ASSERT_NO_FATAL_FAILURE(DestroyMemory(&execution->instructions));
    if (execution->context) {
      ASSERT_EQ(xdna_api_->context_destroy(execution->context), AMDF_STATUS_OK);
      execution->context = nullptr;
    }
  }

  void TearDown() override {
    ASSERT_NO_FATAL_FAILURE(DestroyExecution(&first_));
    ASSERT_NO_FATAL_FAILURE(DestroyExecution(&second_));
    for (auto& binding : bindings_) {
      iree_hal_buffer_release(binding.buffer);
      binding.buffer = nullptr;
      ASSERT_NO_FATAL_FAILURE(DestroyMemory(&binding.storage));
      if (binding.caller_storage.pointer) {
        std::memset(binding.caller_storage.pointer, 0x3C,
                    kBindingStorageByteLength);
      }
      ASSERT_NO_FATAL_FAILURE(DestroyMemory(&binding.caller_storage));
    }
    if (external_memory_.type != AMDF_EXTERNAL_MEMORY_TYPE_NONE) {
      api_->external_memory_release(&external_memory_);
    }
    ASSERT_NO_FATAL_FAILURE(DestroyMemory(&export_source_));
    iree_hal_amd_xdna_image_destroy(executable_);
    executable_ = nullptr;
    XdnaDeviceFixture::TearDown();
  }

  void MapMemory(uint64_t byte_length, MappedMemory* memory) {
    amdf_memory_map_info_t map = {};
    map.type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO;
    map.structure_size = sizeof(map);
    map.flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
    map.byte_length = byte_length;
    ASSERT_EQ(api_->memory_map(memory->memory, &map, &memory->mapping),
              AMDF_STATUS_OK);
    amdf_host_mapping_info_t info = {};
    info.type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO;
    info.structure_size = sizeof(info);
    ASSERT_EQ(api_->host_mapping_query_info(memory->mapping, &info),
              AMDF_STATUS_OK);
    memory->pointer = static_cast<uint8_t*>(info.pointer);
  }

  void PrepareRegistration(const amdf_memory_profile_t& profile,
                           MappedMemory* caller_storage,
                           amdf_memory_create_info_t* create) {
    const auto& registration = profile.registration;
    const uint64_t granularity = registration.byte_length_granularity;
    create->byte_length =
        ((create->byte_length + granularity - 1) / granularity) * granularity;
    create->minimum_alignment = registration.minimum_alignment;
    amdf_memory_create_info_t host = {};
    host.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
    host.structure_size = sizeof(host);
    host.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
    host.byte_length = create->byte_length;
    host.minimum_alignment = registration.registered_host_pointer_alignment;
    ASSERT_EQ(
        api_->memory_create(system_scope_, &host, &caller_storage->memory),
        AMDF_STATUS_OK);
    ASSERT_NO_FATAL_FAILURE(MapMemory(host.byte_length, caller_storage));
    amdf_host_mapping_info_t mapping = {};
    mapping.type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO;
    mapping.structure_size = sizeof(mapping);
    ASSERT_EQ(api_->host_mapping_query_info(caller_storage->mapping, &mapping),
              AMDF_STATUS_OK);
    ASSERT_EQ(mapping.cacheability, registration.registered_host_cacheability);
    create->registered_host_pointer = mapping.pointer;
    create->registered_host_cacheability = mapping.cacheability;
  }

  void CreateBindings(amdf_memory_profile_roles_t role) {
    memory_access_.requirements.address_kinds = uint64_t{1}
                                                << AMDF_MEMORY_ADDRESS_XDNA_DMA;
    if (role != AMDF_MEMORY_PROFILE_ROLE_CREATE) {
      amdf_memory_scope_info_t scope_info = {};
      scope_info.type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO;
      scope_info.structure_size = sizeof(scope_info);
      ASSERT_EQ(api_->memory_scope_query_info(system_scope_, &scope_info),
                AMDF_STATUS_OK);
      amdf_memory_profile_roles_t available_roles = 0;
      for (uint32_t ordinal = 0; ordinal < scope_info.memory_profile_count;
           ++ordinal) {
        amdf_memory_profile_t profile = {};
        profile.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE;
        profile.structure_size = sizeof(profile);
        amdf_memory_access_capabilities_t capabilities = {};
        capabilities.type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES;
        capabilities.structure_size = sizeof(capabilities);
        const amdf_status_t status =
            QueryMemoryProfile(ordinal, &profile, &capabilities);
        if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
          continue;
        }
        ASSERT_EQ(status, AMDF_STATUS_OK);
        available_roles |= profile.roles;
      }
      if ((available_roles & role) == 0) {
        GTEST_SKIP() << "XDNA memory role " << role << " is not advertised";
      }
    }
    const uint32_t profile_ordinal =
        FindMemoryProfileOrdinal(role | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
                                 AMDF_MEMORY_FLAG_HOST_VISIBLE);
    ASSERT_NE(profile_ordinal, AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
    amdf_memory_profile_t profile = {};
    profile.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE;
    profile.structure_size = sizeof(profile);
    amdf_memory_access_capabilities_t capabilities = {};
    capabilities.type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES;
    capabilities.structure_size = sizeof(capabilities);
    ASSERT_EQ(QueryMemoryProfile(profile_ordinal, &profile, &capabilities),
              AMDF_STATUS_OK);
    for (size_t i = 0; i < bindings_.size(); ++i) {
      auto& binding = bindings_[i];
      if (role == AMDF_MEMORY_PROFILE_ROLE_IMPORT) {
        ASSERT_NO_FATAL_FAILURE(
            ImportMemory(profile_ordinal, &binding.storage));
        if (IsSkipped()) {
          return;
        }
      } else {
        amdf_memory_create_info_t create = {};
        create.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
        create.structure_size = sizeof(create);
        create.memory_profile_ordinal = profile_ordinal;
        create.access_count = 1;
        create.accesses = &memory_access_;
        create.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
        create.byte_length = kBindingStorageByteLength;
        if (role == AMDF_MEMORY_PROFILE_ROLE_REGISTER) {
          ASSERT_NO_FATAL_FAILURE(
              PrepareRegistration(profile, &binding.caller_storage, &create));
        }
        ASSERT_EQ(api_->memory_create(system_scope_, &create,
                                      &binding.storage.memory),
                  AMDF_STATUS_OK);
        ASSERT_NO_FATAL_FAILURE(
            MapMemory(kBindingStorageByteLength, &binding.storage));
        if (role == AMDF_MEMORY_PROFILE_ROLE_REGISTER) {
          ASSERT_EQ(binding.storage.pointer, create.registered_host_pointer);
        }
      }
      std::memset(binding.storage.pointer, kGuardValue,
                  kBindingStorageByteLength);
      ASSERT_NO_FATAL_FAILURE(WrapBinding(i, binding.storage.memory, 0));
      ASSERT_NO_FATAL_FAILURE(
          QueryHostCacheOperations(i, binding.storage.memory));
    }
  }

  void WrapBinding(size_t ordinal, amdf_memory_t* memory,
                   uint64_t memory_byte_offset) {
    auto& binding = bindings_[ordinal];
    IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
        iree_hal_buffer_placement_undefined(),
        IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
            IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
        IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE |
            IREE_HAL_MEMORY_ACCESS_UNALIGNED,
        IREE_HAL_BUFFER_USAGE_STORAGE, kBindingByteLength,
        iree_make_byte_span(binding.storage.pointer + kBindingByteOffset,
                            kBindingByteLength),
        iree_hal_buffer_release_callback_null(), iree_allocator_system(),
        &binding.buffer));
    resolved_bindings_[ordinal].buffer_ref =
        iree_hal_make_buffer_ref(binding.buffer, 0, kBindingByteLength);
    resolved_bindings_[ordinal].memory = memory;
    resolved_bindings_[ordinal].memory_byte_offset =
        memory_byte_offset + kBindingByteOffset;
    ASSERT_EQ(
        api_->memory_query_address(memory, 0, AMDF_MEMORY_ADDRESS_XDNA_DMA,
                                   &resolved_bindings_[ordinal].device_address),
        AMDF_STATUS_OK);
    resolved_bindings_[ordinal].device_address +=
        memory_byte_offset + kBindingByteOffset;
  }

  void QueryHostCacheOperations(size_t ordinal, amdf_memory_t* memory) {
    auto& binding = bindings_[ordinal];
    amdf_memory_site_t host = {};
    host.type = AMDF_STRUCTURE_TYPE_MEMORY_SITE;
    host.structure_size = sizeof(host);
    host.kind = AMDF_MEMORY_SITE_KIND_HOST;
    host.value.host_mapping = binding.storage.mapping;
    amdf_memory_site_t device = {};
    device.type = AMDF_STRUCTURE_TYPE_MEMORY_SITE;
    device.structure_size = sizeof(device);
    device.kind = AMDF_MEMORY_SITE_KIND_DEVICE;
    device.value.device.memory = memory;
    device.value.device.queue_family_ordinal = queue_family_ordinal_;
    amdf_memory_pair_info_t pair = {};
    pair.type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO;
    pair.structure_size = sizeof(pair);
    ASSERT_EQ(api_->memory_query_pair_info(&host, &device, &pair),
              AMDF_STATUS_OK);
    ASSERT_NE(pair.flags & AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE, 0u);
    ASSERT_EQ(pair.release.kind, AMDF_CACHE_TRANSITION_KIND_RANGE);
    ASSERT_EQ(pair.release.executor,
              AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT);
    ASSERT_EQ(pair.release.host_operation, AMDF_HOST_CACHE_OPERATION_FLUSH);
    ASSERT_EQ(pair.acquire.kind, AMDF_CACHE_TRANSITION_KIND_NONE);
    ASSERT_EQ(pair.atomic_reach.scope_32, AMDF_ATOMIC_SCOPE_NONE);
    ASSERT_EQ(pair.atomic_reach.scope_64, AMDF_ATOMIC_SCOPE_NONE);
    binding.host_cache.publish = pair.release.host_operation;

    ASSERT_EQ(api_->memory_query_pair_info(&device, &host, &pair),
              AMDF_STATUS_OK);
    ASSERT_NE(pair.flags & AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE, 0u);
    ASSERT_EQ(pair.release.kind, AMDF_CACHE_TRANSITION_KIND_NONE);
    ASSERT_EQ(pair.acquire.kind, AMDF_CACHE_TRANSITION_KIND_RANGE);
    ASSERT_EQ(pair.acquire.executor,
              AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT);
    ASSERT_EQ(pair.acquire.host_operation,
              AMDF_HOST_CACHE_OPERATION_INVALIDATE);
    ASSERT_EQ(pair.atomic_reach.scope_32, AMDF_ATOMIC_SCOPE_NONE);
    ASSERT_EQ(pair.atomic_reach.scope_64, AMDF_ATOMIC_SCOPE_NONE);
    binding.host_cache.acquire = pair.acquire.host_operation;
  }

  void RequireDirectHostTransport(
      const amdf_memory_profile_t& profile,
      amdf_external_memory_support_flags_t required_flags) {
    for (uint32_t i = 0; i < profile.external_memory_support_count; ++i) {
      const auto& support = profile.external_memory_support[i];
      if (support.type == AMDF_EXTERNAL_MEMORY_TYPE_OPAQUE_FD &&
          (support.flags & required_flags) == required_flags) {
        return;
      }
    }
    GTEST_SKIP() << "XDNA direct-host external ranges are not advertised";
  }

  void ImportMemory(uint32_t profile_ordinal, MappedMemory* memory) {
    amdf_memory_profile_t profile = {};
    profile.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE;
    profile.structure_size = sizeof(profile);
    amdf_memory_access_capabilities_t capabilities = {};
    capabilities.type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES;
    capabilities.structure_size = sizeof(capabilities);
    ASSERT_EQ(QueryMemoryProfile(profile_ordinal, &profile, &capabilities),
              AMDF_STATUS_OK);
    RequireDirectHostTransport(
        profile, AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_IMPORT |
                     AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_SOURCE_OFFSET);
    if (IsSkipped()) {
      return;
    }
    const auto source_flags =
        AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_SHAREABLE;
    const uint32_t source_ordinal = FindMemoryProfileOrdinal(
        AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_EXPORT |
            AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
        source_flags);
    ASSERT_NE(source_ordinal, AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
    ASSERT_EQ(QueryMemoryProfile(source_ordinal, &profile, &capabilities),
              AMDF_STATUS_OK);
    RequireDirectHostTransport(
        profile, AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_EXPORT |
                     AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_SOURCE_OFFSET);
    if (IsSkipped()) {
      return;
    }
    const uint64_t source_offset =
        profile.allocation.native_byte_length_granularity + kBindingByteLength;
    amdf_memory_create_info_t create = {};
    create.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
    create.structure_size = sizeof(create);
    create.memory_profile_ordinal = source_ordinal;
    create.access_count = 1;
    create.accesses = &memory_access_;
    create.required_flags = source_flags;
    create.byte_length = source_offset + kBindingStorageByteLength;
    ASSERT_EQ(
        api_->memory_create(system_scope_, &create, &export_source_.memory),
        AMDF_STATUS_OK);
    ASSERT_NO_FATAL_FAILURE(MapMemory(create.byte_length, &export_source_));
    std::memset(export_source_.pointer, kGuardValue, create.byte_length);
    std::memset(export_source_.pointer + source_offset, 0x3C,
                kBindingStorageByteLength);
    ASSERT_EQ(api_->host_mapping_cache_control(export_source_.mapping,
                                               AMDF_HOST_CACHE_OPERATION_FLUSH,
                                               0, create.byte_length),
              AMDF_STATUS_OK);
    amdf_memory_export_info_t export_info = {};
    export_info.type = AMDF_STRUCTURE_TYPE_MEMORY_EXPORT_INFO;
    export_info.structure_size = sizeof(export_info);
    export_info.external_memory_type = AMDF_EXTERNAL_MEMORY_TYPE_OPAQUE_FD;
    export_info.byte_offset = source_offset;
    export_info.byte_length = kBindingStorageByteLength;
    ASSERT_EQ(api_->memory_export(export_source_.memory, &export_info,
                                  &external_memory_),
              AMDF_STATUS_OK);
    const auto identity = external_memory_.physical_backing_id;
    ASSERT_TRUE(amdf_physical_memory_id_is_valid(&identity));
    ASSERT_EQ(external_memory_.source_byte_offset, source_offset);
    amdf_memory_import_info_t import_info = {};
    import_info.type = AMDF_STRUCTURE_TYPE_MEMORY_IMPORT_INFO;
    import_info.structure_size = sizeof(import_info);
    import_info.memory_profile_ordinal = profile_ordinal;
    import_info.access_count = 1;
    import_info.accesses = &memory_access_;
    import_info.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
    import_info.minimum_alignment = kBindingByteLength;
    ASSERT_EQ(api_->memory_import(system_scope_, &import_info,
                                  &external_memory_, &memory->memory),
              AMDF_STATUS_OK);
    const amdf_external_memory_t empty = {};
    ASSERT_EQ(std::memcmp(&external_memory_, &empty, sizeof(empty)), 0);
    ASSERT_NO_FATAL_FAILURE(DestroyMemory(&export_source_));
    amdf_memory_info_t info = {};
    info.type = AMDF_STRUCTURE_TYPE_MEMORY_INFO;
    info.structure_size = sizeof(info);
    ASSERT_EQ(api_->memory_query_info(memory->memory, &info), AMDF_STATUS_OK);
    ASSERT_TRUE(
        amdf_physical_memory_id_is_equal(&info.physical_backing_id, &identity));
    ASSERT_EQ(info.source_byte_offset, source_offset);
    ASSERT_EQ(info.byte_length, kBindingStorageByteLength);
    ASSERT_NO_FATAL_FAILURE(MapMemory(kBindingStorageByteLength, memory));
    for (size_t i = 0; i < kBindingStorageByteLength; ++i) {
      ASSERT_EQ(memory->pointer[i], 0x3C) << "imported byte " << i;
    }
  }

  void PrepareExecution(const ResolvedBindings& bindings, Execution* execution,
                        uint32_t logical_column_count = 1,
                        uint32_t command_capacity = 1) {
    amdf_xdna_context_create_info_t context_create = {};
    context_create.type = AMDF_STRUCTURE_TYPE_XDNA_CONTEXT_CREATE_INFO;
    context_create.structure_size = sizeof(context_create);
    context_create.logical_column_count = logical_column_count;
    context_create.physical_column_origin =
        AMDF_XDNA_PHYSICAL_COLUMN_ORIGIN_ANY;
    context_create.acceptable_scheduling_modes =
        AMDF_XDNA_SCHEDULING_MODE_TIME_SLICED;
    ASSERT_EQ(xdna_api_->context_create(device_, &context_create,
                                        &execution->context),
              AMDF_STATUS_OK);
    const auto* tables = iree_hal_amd_xdna_image_tables(executable_);
    const auto entry =
        iree_hal_amd_xdna_image_tables_entry(tables, entry_ordinal_);
    ASSERT_EQ(entry.allocation_use_count, 1u);
    const auto requirement = iree_hal_amd_xdna_image_tables_allocation(
        tables, iree_hal_amd_xdna_image_tables_allocation_use(
                    tables, entry.first_allocation_use));
    ASSERT_EQ(requirement.domain, IREE_XDNA_ELF_ALLOCATION_DOMAIN_COMMAND);
    // Reserve an aligned immutable instruction slice per pending command.
    const uint64_t alignment =
        std::max<uint64_t>(requirement.alignment, instruction_alignment_);
    execution->byte_length =
        ((requirement.byte_length + alignment - 1) / alignment) * alignment *
        command_capacity;
    amdf_memory_scope_t* scope = nullptr;
    uint32_t count = 0;
    ASSERT_EQ(xdna_api_->context_enumerate_memory_scopes(execution->context, 1,
                                                         &scope, &count),
              AMDF_STATUS_OK);
    amdf_memory_device_access_t access = {};
    access.device = device_;
    access.requirements.access = AMDF_MEMORY_ACCESS_READ |
                                 AMDF_MEMORY_ACCESS_WRITE |
                                 AMDF_MEMORY_ACCESS_EXECUTE;
    access.requirements.flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
    access.requirements.address_kinds = uint64_t{1}
                                        << AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE;
    amdf_memory_profile_t profile = {};
    profile.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE;
    profile.structure_size = sizeof(profile);
    amdf_memory_access_capabilities_t capabilities = {};
    capabilities.type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES;
    capabilities.structure_size = sizeof(capabilities);
    ASSERT_EQ(api_->memory_scope_query_device_profile(scope, 0, 1, &access,
                                                      &profile, &capabilities),
              AMDF_STATUS_OK);
    const uint64_t granularity = profile.allocation.byte_length_granularity;
    ASSERT_GT(granularity, 0u);
    amdf_memory_create_info_t create = {};
    create.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
    create.structure_size = sizeof(create);
    create.memory_profile_ordinal = profile.ordinal;
    create.access_count = 1;
    create.accesses = &access;
    create.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
    create.byte_length =
        ((execution->byte_length + granularity - 1) / granularity) *
        granularity;
    create.minimum_alignment = profile.allocation.minimum_alignment;
    auto& instructions = execution->instructions;
    ASSERT_EQ(api_->memory_create(scope, &create, &instructions.memory),
              AMDF_STATUS_OK);
    uint64_t firmware_address = 0;
    ASSERT_EQ(api_->memory_query_address(instructions.memory, 0,
                                         AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE,
                                         &firmware_address),
              AMDF_STATUS_OK);
    ASSERT_EQ(firmware_address % instruction_alignment_, 0u);
    ASSERT_NO_FATAL_FAILURE(MapMemory(execution->byte_length, &instructions));
    std::memset(instructions.pointer, kGuardValue, execution->byte_length);
    iree_hal_amd_xdna_executable_storage_t storage = {};
    storage.memory = instructions.memory;
    storage.mapping =
        iree_make_byte_span(instructions.pointer, execution->byte_length);
    storage.device_address = firmware_address;
    IREE_ASSERT_OK(iree_hal_amd_xdna_executable_load(
        executable_, entry_ordinal_, 1, &storage));
    IREE_ASSERT_OK(iree_hal_amd_xdna_executable_bind(
        executable_, entry_ordinal_, 1, &storage, bindings.size(),
        bindings.data()));
    IREE_ASSERT_OK(iree_hal_amd_xdna_executable_query_invocation(
        executable_, entry_ordinal_, 1, &storage, &execution->command));
    execution->original_instructions.assign(
        instructions.pointer, instructions.pointer + execution->byte_length);
    ASSERT_EQ(api_->host_mapping_cache_control(instructions.mapping,
                                               AMDF_HOST_CACHE_OPERATION_FLUSH,
                                               0, execution->byte_length),
              AMDF_STATUS_OK);
    amdf_xdna_kernel_queue_create_info_t queue_create = {};
    queue_create.type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_CREATE_INFO;
    queue_create.structure_size = sizeof(queue_create);
    queue_create.queue_family_ordinal = queue_family_ordinal_;
    // Serial callers need one slot; pipelined callers reserve their live
    // window.
    queue_create.maximum_pending_submission_count = command_capacity;
    ASSERT_EQ(xdna_api_->kernel_queue_create(execution->context, &queue_create,
                                             &execution->queue),
              AMDF_STATUS_OK);
    amdf_kernel_queue_info_t queue_info = {};
    queue_info.type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_INFO;
    queue_info.structure_size = sizeof(queue_info);
    ASSERT_EQ(api_->kernel_queue_query_info(execution->queue, &queue_info),
              AMDF_STATUS_OK);
    ASSERT_EQ(queue_info.queue_family_ordinal, queue_family_ordinal_);
    ASSERT_EQ(queue_info.command_type, AMDF_QUEUE_COMMAND_TYPE_XDNA);
    ASSERT_GE(queue_info.maximum_pending_submission_count, command_capacity);
    ASSERT_GE(queue_info.maximum_command_count, 1u);
  }

  void RunExecution(const Execution& execution) {
    amdf_xdna_kernel_queue_submission_info_t submit = {};
    submit.type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_SUBMISSION_INFO;
    submit.structure_size = sizeof(submit);
    submit.command_count = 1;
    submit.commands = &execution.command;
    uint64_t submission = 0;
    ASSERT_EQ(
        xdna_api_->kernel_queue_submit(execution.queue, &submit, &submission),
        AMDF_STATUS_OK);
    ASSERT_EQ(api_->kernel_queue_wait(execution.queue, submission,
                                      AMDF_TIMEOUT_INFINITE, 0),
              AMDF_STATUS_OK);
    amdf_kernel_queue_status_t status = {};
    status.type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS;
    status.structure_size = sizeof(status);
    ASSERT_EQ(api_->kernel_queue_query_status(execution.queue, &status),
              AMDF_STATUS_OK);
    ASSERT_EQ(status.retired_submission, submission);
    ASSERT_EQ(status.terminal_status, AMDF_STATUS_OK);
    ASSERT_NO_FATAL_FAILURE(VerifyInstructions(execution));
  }

  void WriteBinding(size_t ordinal, const BindingValues& values) {
    auto& storage = bindings_[ordinal].storage;
    for (size_t i = 0; i < kElementCount; ++i) {
      iree_unaligned_store_le_u32(storage.pointer + kBindingByteOffset + i * 4,
                                  values[i]);
    }
    ASSERT_EQ(api_->host_mapping_cache_control(
                  storage.mapping, bindings_[ordinal].host_cache.publish, 0,
                  kBindingStorageByteLength),
              AMDF_STATUS_OK);
  }

  void VerifyBindings(const std::array<BindingValues, 3>& expected) {
    for (size_t ordinal = 0; ordinal < bindings_.size(); ++ordinal) {
      SCOPED_TRACE(ordinal);
      const auto& storage = bindings_[ordinal].storage;
      ASSERT_EQ(api_->host_mapping_cache_control(
                    storage.mapping, bindings_[ordinal].host_cache.acquire, 0,
                    kBindingStorageByteLength),
                AMDF_STATUS_OK);
      for (size_t i = 0; i < kBindingByteLength; ++i) {
        ASSERT_EQ(storage.pointer[i], kGuardValue) << "prefix " << i;
        ASSERT_EQ(storage.pointer[2 * kBindingByteLength + i], kGuardValue)
            << "suffix " << i;
      }
      for (size_t i = 0; i < kElementCount; ++i) {
        ASSERT_EQ(iree_unaligned_load_le_u32(storage.pointer +
                                             kBindingByteOffset + i * 4),
                  expected[ordinal][i])
            << "element " << i;
      }
    }
  }

  void VerifyInstructions(const Execution& execution) {
    ASSERT_EQ(
        api_->host_mapping_cache_control(execution.instructions.mapping,
                                         AMDF_HOST_CACHE_OPERATION_INVALIDATE,
                                         0, execution.byte_length),
        AMDF_STATUS_OK);
    ASSERT_EQ(
        std::memcmp(execution.original_instructions.data(),
                    execution.instructions.pointer, execution.byte_length),
        0);
  }

  // Native kernel queue family selected from the libamdf endpoint.
  uint32_t queue_family_ordinal_ = UINT32_MAX;
  // Case-owned immutable decoded compiler image.
  iree_hal_amd_xdna_image_t* executable_ = nullptr;
  // Indexed multiplication entry in the immutable image.
  uint32_t entry_ordinal_ = 0;
  // Storage requirements derived from the endpoint and executable.
  iree_host_size_t instruction_alignment_ = 0;
  // Primary execution owner used by every case.
  Execution first_;
  // Independently prepared execution sharing only ordinary data and device.
  Execution second_;
  // Temporary export source, released before imported bindings are used.
  MappedMemory export_source_;
  // Move-owned DMA-BUF value, consumed by import or released at teardown.
  amdf_external_memory_t external_memory_ = {};
  // Native and HAL owners for the three canonical bindings.
  struct Binding {
    // Independent CPU-only page owner, released after native registration.
    MappedMemory caller_storage;
    // Native memory and its explicit host view.
    MappedMemory storage;
    // HAL wrapper borrowing storage through all command completion.
    iree_hal_buffer_t* buffer = nullptr;
    // Directional operations selected once from the concrete host/device pair.
    struct {
      // Releases CPU writes before the program reads through DMA.
      amdf_host_cache_operation_t publish = 0;
      // Makes completed DMA writes visible before CPU verification.
      amdf_host_cache_operation_t acquire = 0;
    } host_cache;
  };
  // Lhs, rhs and output backing, independent of instruction storage.
  std::array<Binding, 3> bindings_;
  // DMA addresses resolved through the public memory handles.
  ResolvedBindings resolved_bindings_ = {};
};

// Registers one independently owned shared mapping, with all bindings as
// offsets into that allocation. The creator's mapping is not a device owner.
class XdnaSharedMappingFixture : public XdnaExecutionFixture {
 protected:
  void QuerySharedRegistration(amdf_memory_profile_t* out_profile,
                               uint64_t* out_byte_length) {
    memory_access_.requirements.address_kinds = UINT64_C(1)
                                                << AMDF_MEMORY_ADDRESS_XDNA_DMA;
    const uint32_t ordinal = FindMemoryProfileOrdinal(
        AMDF_MEMORY_PROFILE_ROLE_REGISTER | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
        AMDF_MEMORY_FLAG_HOST_VISIBLE);
    ASSERT_NE(ordinal, AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
    amdf_memory_profile_t profile = {};
    profile.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE;
    profile.structure_size = sizeof(profile);
    amdf_memory_access_capabilities_t capabilities = {};
    capabilities.type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES;
    capabilities.structure_size = sizeof(capabilities);
    ASSERT_EQ(QueryMemoryProfile(ordinal, &profile, &capabilities),
              AMDF_STATUS_OK);
    const auto& registration = profile.registration;
    const uint64_t granularity = registration.byte_length_granularity;
    ASSERT_GT(granularity, 0u);
    ASSERT_GT(registration.registered_host_pointer_alignment, 0u);
    // Ordinary shared mappings are write-back host pages on both platforms.
    ASSERT_EQ(registration.registered_host_cacheability,
              AMDF_HOST_CACHEABILITY_WRITE_BACK);
    const uint64_t byte_length =
        ((bindings_.size() * kBindingStorageByteLength + granularity - 1) /
         granularity) *
        granularity;
    ASSERT_LE(byte_length, registration.maximum_byte_length);
    *out_profile = profile;
    *out_byte_length = byte_length;
  }

  void CreateSharedBindings() {
    amdf_memory_profile_t profile = {};
    uint64_t byte_length = 0;
    ASSERT_NO_FATAL_FAILURE(QuerySharedRegistration(&profile, &byte_length));
    IREE_ASSERT_OK(iree_shm_create(nullptr, byte_length, &originator_));
    std::memset(originator_.base, kGuardValue, originator_.size);
    IREE_ASSERT_OK(
        iree_shm_open_handle(originator_.handle, originator_.size, &shared_));
    ASSERT_NE(shared_.base, originator_.base);
    ASSERT_EQ(std::memcmp(shared_.base, originator_.base, byte_length), 0);
    iree_shm_close(&originator_);
    ASSERT_NO_FATAL_FAILURE(RegisterSharedBindings(profile, byte_length));
  }

  void RegisterSharedBindings(const amdf_memory_profile_t& profile,
                              uint64_t byte_length) {
    ASSERT_GE(shared_.size, byte_length);
    ASSERT_EQ(reinterpret_cast<uintptr_t>(shared_.base) %
                  profile.registration.registered_host_pointer_alignment,
              0u);
    amdf_memory_create_info_t create = {};
    create.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
    create.structure_size = sizeof(create);
    create.memory_profile_ordinal = profile.ordinal;
    create.access_count = 1;
    create.accesses = &memory_access_;
    create.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
    create.byte_length = byte_length;
    create.minimum_alignment = profile.registration.minimum_alignment;
    create.registered_host_pointer = shared_.base;
    create.registered_host_cacheability = AMDF_HOST_CACHEABILITY_WRITE_BACK;
    ASSERT_EQ(api_->memory_create(system_scope_, &create, &registration_),
              AMDF_STATUS_OK);

    for (size_t i = 0; i < bindings_.size(); ++i) {
      auto& storage = bindings_[i].storage;
      amdf_memory_map_info_t map = {};
      map.type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO;
      map.structure_size = sizeof(map);
      map.flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
      map.byte_offset = i * kBindingStorageByteLength;
      map.byte_length = kBindingStorageByteLength;
      ASSERT_EQ(api_->memory_map(registration_, &map, &storage.mapping),
                AMDF_STATUS_OK);
      amdf_host_mapping_info_t info = {};
      info.type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO;
      info.structure_size = sizeof(info);
      ASSERT_EQ(api_->host_mapping_query_info(storage.mapping, &info),
                AMDF_STATUS_OK);
      storage.pointer = static_cast<uint8_t*>(info.pointer);
      ASSERT_EQ(storage.pointer,
                static_cast<uint8_t*>(shared_.base) + map.byte_offset);
      // The binding owns its view, not a second copy of the registration.
      ASSERT_NO_FATAL_FAILURE(WrapBinding(i, registration_, map.byte_offset));
      ASSERT_NO_FATAL_FAILURE(QueryHostCacheOperations(i, registration_));
    }
  }

  void TearDown() override {
    // Checked execution retirement and every offset view precede the single
    // registration release. A failed native detach must not unmap its source.
    ASSERT_NO_FATAL_FAILURE(XdnaExecutionFixture::TearDown());
    if (registration_) {
      ASSERT_EQ(api_->memory_destroy(std::exchange(registration_, nullptr)),
                AMDF_STATUS_OK);
    }
    iree_shm_close(&shared_);
    iree_shm_close(&originator_);
  }

  // Temporary creator mapping, closed before native registration.
  iree_shm_mapping_t originator_ = {};
  // Independently opened mapping retained through native registration release.
  iree_shm_mapping_t shared_ = {};
  // One registration borrowed by all three offset bindings.
  amdf_memory_t* registration_ = nullptr;
};

}  // namespace iree::experimental::xdna::testing

#endif  // IREE_EXPERIMENTAL_XDNA_CTS_EXECUTION_FIXTURE_H_
