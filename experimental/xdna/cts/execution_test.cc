// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#include "experimental/xdna/executable.h"
#include "experimental/xdna/prepared_command.h"
#include "iree/hal/drivers/amd/xdna/image/aie2p/npu2.h"
#include "iree/hal/drivers/amd/xdna/image/testdata/mul_i32.h"
#include "iree/hal/drivers/amd/xdna/image/testdata/mul_i32_npu4.h"
#include "iree/hal/drivers/amd/xdna/image/testing/aie2p_image_fixture.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "libamdf/cts/xdna/xdna_device_fixture.h"

namespace {

// Both canonical compiler fixtures multiply sixteen low-32-bit integer pairs.
constexpr size_t kElementCount = 16;
constexpr size_t kBindingByteLength = kElementCount * sizeof(uint32_t);
constexpr size_t kBindingByteOffset = kBindingByteLength;
constexpr size_t kBindingStorageByteLength = 3 * kBindingByteLength;
constexpr uint8_t kGuardValue = 0xA5;
using BindingValues = std::array<uint32_t, kElementCount>;
using PreparedBindings =
    std::array<iree_hal_amd_xdna_prepared_command_binding_t, 3>;

// Unsigned arithmetic gives exact low bits for signed and overflowing products.
constexpr BindingValues kValues = {
    0,          1,          2,          3,          7,          31,
    65535,      65536,      0x7FFFFFFF, 0x80000000, 0x80000001, 0xFFFFFFFD,
    0xFFFFFFFE, 0xFFFFFFFF, 0x12345678, 0x87654321};

class XdnaExecutionTest
    : public XdnaDeviceFixture,
      public ::testing::WithParamInterface<amdf_memory_profile_roles_t> {
 protected:
  struct MappedMemory {
    // Case-owned allocation; its context and device outlive it.
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
    // Mapped extent covering initialization and execution ranges.
    iree_host_size_t byte_length = 0;
    // Immutable command instance retaining the executable and HAL buffers.
    iree_hal_amd_xdna_prepared_command_t* prepared = nullptr;
    // Native transport lease borrowing this execution's context.
    amdf_kernel_queue_t* queue = nullptr;
    // Test-only byte oracle captured after cold preparation.
    std::vector<uint8_t> original_instructions;
  };

  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(XdnaDeviceFixture::SetUp());
    if (IsSkipped()) return;
    amdf_xdna_endpoint_info_t info = {};
    info.type = AMDF_STRUCTURE_TYPE_XDNA_ENDPOINT_INFO;
    info.structure_size = sizeof(info);
    ASSERT_EQ(xdna_api_->endpoint_query_info(endpoint_, &info), AMDF_STATUS_OK);
    if ((info.context.scheduling_modes &
         AMDF_XDNA_SCHEDULING_MODE_TIME_SLICED) == 0) {
      GTEST_SKIP() << "time-sliced XDNA contexts are unavailable";
    }
    instruction_alignment_ = info.instruction.address_alignment;

    const iree_file_toc_t* image = nullptr;
    if (std::strcmp(info.target_id, "amd.xdna.strix.17f0_10") == 0) {
      image = iree_hal_amd_xdna_test_mul_i32_npu4_create();
    } else if (std::strcmp(info.target_id, "amd.xdna.strix_halo.17f0_11") ==
               0) {
      image = iree_hal_amd_xdna_test_mul_i32_create();
    } else {
      GTEST_SKIP() << "no canonical multiplication fixture for "
                   << info.target_id;
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
    queue_family_spec_.name = IREE_SV("xdna");
    queue_family_spec_.physical_device_affinity = 1;
    queue_family_spec_.role_flags = IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH;
    iree_hal_queue_family_initialize(family_ordinal, &queue_family_spec_,
                                     &queue_family_);
    iree_hal_amd_xdna_aie2p_target_t target;
    IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_npu2_target_initialize(
        iree_make_cstring_view(info.target_id), 1, &target));
    const auto* image_bytes = reinterpret_cast<const uint8_t*>(image->data);
    auto sequence = iree::hal::amd::xdna::testing::MakeOwnedByteSequence(
        std::vector<uint8_t>(image_bytes, image_bytes + image->size));
    IREE_ASSERT_OK(iree_hal_amd_xdna_executable_create(
        &queue_family_, sequence.get(), &target, iree_allocator_system(),
        &executable_));
    IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
        executable_, IREE_SV("mul_i32"), &function_));
  }

  void DestroyMemory(MappedMemory* memory) {
    if (memory->mapping) {
      ASSERT_EQ(api_->host_mapping_destroy(memory->mapping), AMDF_STATUS_OK);
      memory->mapping = nullptr;
      memory->pointer = nullptr;
    }
    if (memory->memory) {
      ASSERT_EQ(api_->memory_destroy(memory->memory), AMDF_STATUS_OK);
      memory->memory = nullptr;
    }
  }

  void DestroyExecution(Execution* execution) {
    if (execution->queue) {
      ASSERT_EQ(api_->kernel_queue_destroy(execution->queue), AMDF_STATUS_OK);
      execution->queue = nullptr;
    }
    iree_hal_amd_xdna_prepared_command_destroy(execution->prepared);
    execution->prepared = nullptr;
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
    }
    if (external_memory_.type != AMDF_EXTERNAL_MEMORY_TYPE_NONE) {
      api_->external_memory_release(&external_memory_);
    }
    ASSERT_NO_FATAL_FAILURE(DestroyMemory(&export_source_));
    iree_hal_executable_release(executable_);
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

  void CreateBindings() {
    memory_access_.requirements.address_kinds = uint64_t{1}
                                                << AMDF_MEMORY_ADDRESS_XDNA_DMA;
    if (GetParam() != AMDF_MEMORY_PROFILE_ROLE_CREATE) {
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
        if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED))
          continue;
        ASSERT_EQ(status, AMDF_STATUS_OK);
        available_roles |= profile.roles;
      }
      if ((available_roles & GetParam()) == 0) {
        GTEST_SKIP() << "XDNA memory role " << GetParam()
                     << " is not advertised";
      }
    }
    const uint32_t profile_ordinal =
        FindMemoryProfileOrdinal(GetParam() | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
                                 AMDF_MEMORY_FLAG_HOST_VISIBLE);
    ASSERT_NE(profile_ordinal, AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
    for (size_t i = 0; i < bindings_.size(); ++i) {
      auto& binding = bindings_[i];
      if (GetParam() == AMDF_MEMORY_PROFILE_ROLE_IMPORT) {
        ASSERT_NO_FATAL_FAILURE(
            ImportMemory(profile_ordinal, &binding.storage));
        if (IsSkipped()) return;
      } else {
        amdf_memory_create_info_t create = {};
        create.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
        create.structure_size = sizeof(create);
        create.memory_profile_ordinal = profile_ordinal;
        create.access_count = 1;
        create.accesses = &memory_access_;
        create.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
        create.byte_length = kBindingStorageByteLength;
        if (GetParam() == AMDF_MEMORY_PROFILE_ROLE_REGISTER) {
          create.registered_host_pointer =
              binding.caller_storage.data() + kBindingByteLength;
          create.minimum_alignment = kBindingByteLength;
        }
        ASSERT_EQ(api_->memory_create(system_scope_, &create,
                                      &binding.storage.memory),
                  AMDF_STATUS_OK);
        ASSERT_NO_FATAL_FAILURE(
            MapMemory(kBindingStorageByteLength, &binding.storage));
        if (GetParam() == AMDF_MEMORY_PROFILE_ROLE_REGISTER) {
          ASSERT_EQ(binding.storage.pointer, create.registered_host_pointer);
        }
      }
      std::memset(binding.storage.pointer, kGuardValue,
                  kBindingStorageByteLength);
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
      prepared_bindings_[i].buffer_ref =
          iree_hal_make_buffer_ref(binding.buffer, 0, kBindingByteLength);
      prepared_bindings_[i].memory = binding.storage.memory;
      prepared_bindings_[i].memory_byte_offset = kBindingByteOffset;
      ASSERT_EQ(api_->memory_query_address(
                    binding.storage.memory, 0, AMDF_MEMORY_ADDRESS_XDNA_DMA,
                    &prepared_bindings_[i].device_address),
                AMDF_STATUS_OK);
      prepared_bindings_[i].device_address += kBindingByteOffset;
      ASSERT_NO_FATAL_FAILURE(QueryHostCacheOperations(i));
    }
  }

  void QueryHostCacheOperations(size_t ordinal) {
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
    device.value.device.memory = binding.storage.memory;
    device.value.device.queue_family_ordinal = queue_family_.ordinal;
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

  void RequireDmaBuf(const amdf_memory_profile_t& profile,
                     amdf_external_memory_support_flags_t required_flags) {
    for (uint32_t i = 0; i < profile.external_memory_support_count; ++i) {
      const auto& support = profile.external_memory_support[i];
      if (support.type == AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD &&
          (support.flags & required_flags) == required_flags) {
        return;
      }
    }
    GTEST_SKIP() << "XDNA DMA-BUF source ranges are not advertised";
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
    RequireDmaBuf(profile, AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_IMPORT |
                               AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_SOURCE_OFFSET);
    if (IsSkipped()) return;
    const auto source_flags =
        AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_SHAREABLE;
    const uint32_t source_ordinal = FindMemoryProfileOrdinal(
        AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_EXPORT |
            AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
        source_flags);
    ASSERT_NE(source_ordinal, AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
    ASSERT_EQ(QueryMemoryProfile(source_ordinal, &profile, &capabilities),
              AMDF_STATUS_OK);
    RequireDmaBuf(profile, AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_EXPORT |
                               AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_SOURCE_OFFSET);
    if (IsSkipped()) return;
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
    export_info.external_memory_type = AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD;
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

  void PrepareExecution(const PreparedBindings& bindings,
                        Execution* execution) {
    amdf_xdna_context_create_info_t context_create = {};
    context_create.type = AMDF_STRUCTURE_TYPE_XDNA_CONTEXT_CREATE_INFO;
    context_create.structure_size = sizeof(context_create);
    context_create.logical_column_count = 1;
    context_create.physical_column_origin =
        AMDF_XDNA_PHYSICAL_COLUMN_ORIGIN_ANY;
    context_create.acceptable_scheduling_modes =
        AMDF_XDNA_SCHEDULING_MODE_TIME_SLICED;
    ASSERT_EQ(xdna_api_->context_create(device_, &context_create,
                                        &execution->context),
              AMDF_STATUS_OK);
    IREE_ASSERT_OK(iree_hal_amd_xdna_prepared_command_query_storage_size(
        executable_, function_, instruction_alignment_,
        &execution->byte_length));
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
    amdf_xdna_kernel_command_t storage = {};
    storage.memory = instructions.memory;
    storage.byte_length = execution->byte_length;
    IREE_ASSERT_OK(iree_hal_amd_xdna_prepared_command_create(
        executable_, function_, instruction_alignment_, &storage,
        iree_make_byte_span(instructions.pointer, execution->byte_length),
        bindings.size(), bindings.data(), iree_allocator_system(),
        &execution->prepared));
    execution->original_instructions.assign(
        instructions.pointer, instructions.pointer + execution->byte_length);
    ASSERT_EQ(api_->host_mapping_cache_control(instructions.mapping,
                                               AMDF_HOST_CACHE_OPERATION_FLUSH,
                                               0, execution->byte_length),
              AMDF_STATUS_OK);
    amdf_xdna_kernel_queue_create_info_t queue_create = {};
    queue_create.type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_CREATE_INFO;
    queue_create.structure_size = sizeof(queue_create);
    queue_create.queue_family_ordinal = queue_family_.ordinal;
    ASSERT_EQ(xdna_api_->kernel_queue_create(execution->context, &queue_create,
                                             &execution->queue),
              AMDF_STATUS_OK);
    amdf_kernel_queue_info_t queue_info = {};
    queue_info.type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_INFO;
    queue_info.structure_size = sizeof(queue_info);
    ASSERT_EQ(api_->kernel_queue_query_info(execution->queue, &queue_info),
              AMDF_STATUS_OK);
    ASSERT_EQ(queue_info.queue_family_ordinal, queue_family_.ordinal);
    ASSERT_EQ(queue_info.command_type, AMDF_QUEUE_COMMAND_TYPE_XDNA);
    ASSERT_GE(queue_info.maximum_pending_submission_count, 1u);
    ASSERT_GE(queue_info.maximum_command_count, 1u);
  }

  void RunExecution(const Execution& execution,
                    const amdf_xdna_kernel_command_t* command) {
    amdf_xdna_kernel_queue_submission_info_t submit = {};
    submit.type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_SUBMISSION_INFO;
    submit.structure_size = sizeof(submit);
    submit.command_count = 1;
    submit.commands = command;
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

  // Executable-only family metadata; native queues are managed through libamdf.
  iree_hal_queue_family_spec_t queue_family_spec_ = {};
  // HAL queue-family descriptor borrowed by the executable.
  iree_hal_queue_family_t queue_family_ = {};
  // Case-owned immutable decoded compiler image.
  iree_hal_executable_t* executable_ = nullptr;
  // Reflected multiplication entry in executable_.
  iree_hal_executable_function_t function_ =
      iree_hal_executable_function_invalid();
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
    // Optional caller-owned backing. Registration begins inside this array
    // and remains live until the native memory handle is destroyed.
    alignas(kBindingByteLength)
        std::array<uint8_t, kBindingByteLength +
                                kBindingStorageByteLength> caller_storage = {};
    // Native memory and its explicit host view.
    MappedMemory storage;
    // HAL wrapper borrowing storage until preparation has been destroyed.
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
  // Cold DMA addresses resolved through the public memory handles.
  PreparedBindings prepared_bindings_ = {};
};

TEST_P(XdnaExecutionTest, ReusesImmutableInstructionsWithChangingInputs) {
  ASSERT_NO_FATAL_FAILURE(CreateBindings());
  if (IsSkipped()) return;
  ASSERT_NO_FATAL_FAILURE(PrepareExecution(prepared_bindings_, &first_));
  for (uint32_t iteration = 0; iteration < 3; ++iteration) {
    SCOPED_TRACE(iteration);
    std::array<BindingValues, 3> expected;
    BindingValues poisoned;
    for (size_t i = 0; i < kElementCount; ++i) {
      expected[0][i] = kValues[(i + iteration) % kElementCount];
      expected[1][i] = kValues[(i * 3 + iteration + 5) % kElementCount];
      expected[2][i] = expected[0][i] * expected[1][i];
      // Every output must change; neither zero-fill nor stale output can pass.
      poisoned[i] = ~expected[2][i];
    }
    ASSERT_NO_FATAL_FAILURE(WriteBinding(0, expected[0]));
    ASSERT_NO_FATAL_FAILURE(WriteBinding(1, expected[1]));
    ASSERT_NO_FATAL_FAILURE(WriteBinding(2, poisoned));
    const auto* command =
        iteration == 0
            ? iree_hal_amd_xdna_prepared_command_initialization(first_.prepared)
            : iree_hal_amd_xdna_prepared_command_execution(first_.prepared);
    ASSERT_NO_FATAL_FAILURE(RunExecution(first_, command));
    ASSERT_NO_FATAL_FAILURE(VerifyBindings(expected));
    ASSERT_NO_FATAL_FAILURE(VerifyInstructions(first_));
  }
}

TEST_P(XdnaExecutionTest, SharesDataAcrossIndependentContextLifetimes) {
  ASSERT_NO_FATAL_FAILURE(CreateBindings());
  if (IsSkipped()) return;
  ASSERT_NO_FATAL_FAILURE(PrepareExecution(prepared_bindings_, &first_));
  // A: (lhs, rhs) -> intermediate; B: (intermediate, rhs) -> lhs.
  const PreparedBindings consumer_bindings = {
      prepared_bindings_[2], prepared_bindings_[1], prepared_bindings_[0]};
  ASSERT_NO_FATAL_FAILURE(PrepareExecution(consumer_bindings, &second_));
  ASSERT_NE(first_.context, second_.context);
  ASSERT_NE(first_.instructions.memory, second_.instructions.memory);

  std::array<amdf_memory_info_t, 3> original_info = {};
  for (size_t ordinal = 0; ordinal < bindings_.size(); ++ordinal) {
    auto& info = original_info[ordinal];
    info.type = AMDF_STRUCTURE_TYPE_MEMORY_INFO;
    info.structure_size = sizeof(info);
    ASSERT_EQ(api_->memory_query_info(bindings_[ordinal].storage.memory, &info),
              AMDF_STATUS_OK);
  }

  std::array<BindingValues, 3> expected;
  BindingValues poisoned;
  for (size_t i = 0; i < kElementCount; ++i) {
    expected[0][i] = kValues[i];
    // Odd nonunit factors keep the second computation distinct modulo 2^32.
    expected[1][i] = static_cast<uint32_t>(i * 2 + 3);
    expected[2][i] = expected[0][i] * expected[1][i];
    poisoned[i] = ~expected[2][i];
  }
  ASSERT_NO_FATAL_FAILURE(WriteBinding(0, expected[0]));
  ASSERT_NO_FATAL_FAILURE(WriteBinding(1, expected[1]));
  ASSERT_NO_FATAL_FAILURE(WriteBinding(2, poisoned));
  ASSERT_NO_FATAL_FAILURE(RunExecution(
      first_,
      iree_hal_amd_xdna_prepared_command_initialization(first_.prepared)));
  // The consumer sees the producer's bytes directly. There is no CPU payload
  // access or cache transition between these fully retired finite commands.
  ASSERT_NO_FATAL_FAILURE(RunExecution(
      second_,
      iree_hal_amd_xdna_prepared_command_initialization(second_.prepared)));
  for (size_t i = 0; i < kElementCount; ++i) {
    expected[0][i] = expected[2][i] * expected[1][i];
  }
  ASSERT_NO_FATAL_FAILURE(VerifyBindings(expected));
  ASSERT_NO_FATAL_FAILURE(VerifyInstructions(first_));
  ASSERT_NO_FATAL_FAILURE(VerifyInstructions(second_));
  ASSERT_NO_FATAL_FAILURE(DestroyExecution(&first_));

  // Shared backing borrows the ordinary device, not the departed producer.
  for (size_t ordinal = 0; ordinal < bindings_.size(); ++ordinal) {
    amdf_memory_info_t info = {};
    info.type = AMDF_STRUCTURE_TYPE_MEMORY_INFO;
    info.structure_size = sizeof(info);
    ASSERT_EQ(api_->memory_query_info(bindings_[ordinal].storage.memory, &info),
              AMDF_STATUS_OK);
    ASSERT_EQ(info.byte_length, original_info[ordinal].byte_length);
    ASSERT_EQ(info.source_byte_offset,
              original_info[ordinal].source_byte_offset);
    ASSERT_TRUE(amdf_physical_memory_id_is_equal(
        &info.physical_backing_id,
        &original_info[ordinal].physical_backing_id));
    uint64_t address = 0;
    ASSERT_EQ(
        api_->memory_query_address(bindings_[ordinal].storage.memory, 0,
                                   AMDF_MEMORY_ADDRESS_XDNA_DMA, &address),
        AMDF_STATUS_OK);
    ASSERT_EQ(address + kBindingByteOffset,
              prepared_bindings_[ordinal].device_address);
  }
  for (uint32_t iteration = 0; iteration < 2; ++iteration) {
    SCOPED_TRACE(iteration);
    for (size_t i = 0; i < kElementCount; ++i) {
      expected[1][i] += 2;
      expected[0][i] = expected[2][i] * expected[1][i];
      poisoned[i] = ~expected[0][i];
    }
    ASSERT_NO_FATAL_FAILURE(WriteBinding(0, poisoned));
    ASSERT_NO_FATAL_FAILURE(WriteBinding(1, expected[1]));
    ASSERT_NO_FATAL_FAILURE(RunExecution(
        second_,
        iree_hal_amd_xdna_prepared_command_execution(second_.prepared)));
    ASSERT_NO_FATAL_FAILURE(VerifyBindings(expected));
    ASSERT_NO_FATAL_FAILURE(VerifyInstructions(second_));
  }
}

INSTANTIATE_TEST_SUITE_P(
    MemoryBacking, XdnaExecutionTest,
    ::testing::Values(AMDF_MEMORY_PROFILE_ROLE_CREATE,
                      AMDF_MEMORY_PROFILE_ROLE_REGISTER,
                      AMDF_MEMORY_PROFILE_ROLE_IMPORT),
    [](const ::testing::TestParamInfo<amdf_memory_profile_roles_t>& info) {
      switch (info.param) {
        case AMDF_MEMORY_PROFILE_ROLE_CREATE:
          return "Allocated";
        case AMDF_MEMORY_PROFILE_ROLE_REGISTER:
          return "Registered";
        default:
          return "Imported";
      }
    });

}  // namespace
