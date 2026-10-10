// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/slab_provider.h"

#include <cstdint>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct FakeNative;

struct FakeMemory {
  FakeNative* owner;
  std::vector<uint8_t> bytes;
};

struct FakeMapping {
  FakeMemory* memory;
};

struct CacheOperation {
  amdf_host_cache_operation_t operation;
  uint64_t offset;
  uint64_t length;
};

struct FakeNative {
  FakeNative() {
    api.memory_create = MemoryCreate;
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
      amdf_memory_t** out_memory) {
    auto* self = reinterpret_cast<FakeNative*>(scope);
    EXPECT_EQ(info->memory_profile_ordinal, 7u);
    EXPECT_EQ(info->access_count, 2u);
    EXPECT_EQ(info->required_flags, AMDF_MEMORY_FLAG_HOST_VISIBLE);
    EXPECT_NE(info->byte_length, 0u);
    EXPECT_EQ(info->byte_length % 64, 0u);
    EXPECT_LE(info->byte_length, 4096u);
    EXPECT_GE(info->minimum_alignment, 64u);
    EXPECT_LE(info->minimum_alignment, 4096u);
    EXPECT_TRUE(iree_device_size_is_power_of_two(info->minimum_alignment));
    self->created_lengths.push_back(info->byte_length);
    self->created_alignments.push_back(info->minimum_alignment);
    auto* memory =
        new FakeMemory{self, std::vector<uint8_t>(info->byte_length)};
    ++self->create_count;
    *out_memory = reinterpret_cast<amdf_memory_t*>(memory);
    return AMDF_STATUS_OK;
  }

  static amdf_status_t AMDF_CALL MemoryMap(amdf_memory_t* base_memory,
                                           const amdf_memory_map_info_t* info,
                                           amdf_host_mapping_t** out_mapping) {
    auto* memory = reinterpret_cast<FakeMemory*>(base_memory);
    EXPECT_EQ(info->byte_length, memory->bytes.size());
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
    out_info->cacheability = AMDF_HOST_CACHEABILITY_WRITE_BACK;
    out_info->pointer = mapping->memory->bytes.data();
    out_info->byte_length = mapping->memory->bytes.size();
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

  static amdf_status_t AMDF_CALL MemoryDestroy(amdf_memory_t* base_memory) {
    auto* memory = reinterpret_cast<FakeMemory*>(base_memory);
    FakeNative* self = memory->owner;
    ++self->destroy_count;
    delete memory;
    return self->fail_memory_destroy
               ? amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL)
               : AMDF_STATUS_OK;
  }

  static amdf_status_t AMDF_CALL
  MemoryQueryAddress(amdf_memory_t* base_memory, uint32_t access_ordinal,
                     amdf_memory_address_kind_t kind, uint64_t* out_address) {
    auto* memory = reinterpret_cast<FakeMemory*>(base_memory);
    EXPECT_EQ(kind, AMDF_MEMORY_ADDRESS_XDNA_DMA);
    EXPECT_LT(access_ordinal, 2u);
    *out_address = UINT64_C(0x100000) + UINT64_C(0x10000) * access_ordinal +
                   memory->bytes.size();
    return AMDF_STATUS_OK;
  }

  amdf_api_t api = {};
  std::vector<uint64_t> created_lengths;
  std::vector<uint64_t> created_alignments;
  std::vector<CacheOperation> cache_operations;
  int create_count = 0;
  int map_count = 0;
  int unmap_count = 0;
  int destroy_count = 0;
  bool fail_memory_destroy = false;
};

class SlabProviderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    context_.api = &native_.api;
    context_.event_sink = iree_hal_device_event_sink_discard();
    accesses_[0].device = reinterpret_cast<amdf_device_t*>(uintptr_t{1});
    accesses_[1].device = reinterpret_cast<amdf_device_t*>(uintptr_t{2});
    for (auto& access : accesses_) {
      access.requirements.access =
          AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
      access.requirements.flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
      access.requirements.address_kinds = UINT64_C(1)
                                          << AMDF_MEMORY_ADDRESS_XDNA_DMA;
    }
    iree_hal_amd_xdna_slab_provider_options_t options = {};
    options.scope = native_.scope();
    options.profile.ordinal = 7;
    options.profile.roles =
        AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP;
    options.profile.supported_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
    options.profile.allocation.maximum_byte_length = 4096;
    options.profile.allocation.byte_length_granularity = 64;
    options.profile.allocation.minimum_alignment = 64;
    options.profile.allocation.maximum_alignment = 4096;
    options.access_count = 2;
    options.accesses = accesses_;
    options.memory_type =
        IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
        IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
    options.supported_usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                              IREE_HAL_BUFFER_USAGE_DISPATCH |
                              IREE_HAL_BUFFER_USAGE_MAPPING;
    options.maintenance_alignment = 64;
    IREE_ASSERT_OK(iree_hal_amd_xdna_slab_provider_create(
        reinterpret_cast<iree_hal_device_t*>(uintptr_t{1}), &context_, &options,
        iree_allocator_system(), &provider_));
  }

  void TearDown() override { iree_hal_slab_provider_release(provider_); }

  FakeNative native_;
  iree_hal_amd_xdna_context_t context_ = {};
  amdf_memory_device_access_t accesses_[2] = {};
  iree_hal_slab_provider_t* provider_ = nullptr;
};

TEST_F(SlabProviderTest, PublishesRangedBindingsAndCacheControl) {
  iree_hal_slab_provider_properties_t properties = {};
  iree_hal_slab_provider_query_properties(provider_, &properties);
  EXPECT_EQ(properties.allocation_alignment, 64u);
  EXPECT_EQ(properties.max_allocation_alignment, 4096u);
  EXPECT_EQ(properties.maintenance_alignment, 64u);

  iree_hal_slab_t slab = {};
  IREE_ASSERT_OK(iree_hal_slab_provider_acquire_slab(
      provider_, 100, /*min_alignment=*/256, &slab));
  EXPECT_EQ(slab.length, 100u);
  ASSERT_NE(slab.base_ptr, nullptr);
  EXPECT_EQ(native_.create_count, 1);
  EXPECT_EQ(native_.map_count, 1);
  ASSERT_EQ(native_.created_lengths.size(), 1u);
  EXPECT_EQ(native_.created_lengths[0], 128u);
  ASSERT_EQ(native_.created_alignments.size(), 1u);
  EXPECT_EQ(native_.created_alignments[0], 256u);

  int release_count = 0;
  iree_hal_buffer_params_t params = {};
  params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH | IREE_HAL_BUFFER_USAGE_MAPPING;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.type = properties.memory_type;
  params.queue_family_affinity = IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY;
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_slab_provider_wrap_buffer(
      provider_, &slab, 16, 32, params,
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
      buffer, {/*index=*/1, /*type=*/IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA});
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
  iree_hal_slab_provider_release_slab(provider_, &slab);
  EXPECT_EQ(native_.unmap_count, 1);
  EXPECT_EQ(native_.destroy_count, 1);

  iree_hal_slab_provider_visited_set_t visited = {};
  iree_hal_slab_provider_stats_t stats = {};
  iree_hal_slab_provider_query_stats(provider_, &visited, &stats);
  EXPECT_EQ(stats.total_acquired, 1u);
  EXPECT_EQ(stats.total_released, 1u);
}

TEST_F(SlabProviderTest, NativeDestroyFailureLeaksWithoutAborting) {
  native_.fail_memory_destroy = true;
  iree_hal_slab_t slab = {};
  IREE_ASSERT_OK(iree_hal_slab_provider_acquire_slab(
      provider_, 64, /*min_alignment=*/1, &slab));
  ASSERT_EQ(native_.created_lengths.size(), 1u);
  EXPECT_EQ(native_.created_lengths[0], 64u);
  ASSERT_EQ(native_.created_alignments.size(), 1u);
  EXPECT_EQ(native_.created_alignments[0], 64u);
  iree_hal_slab_provider_release_slab(provider_, &slab);
  EXPECT_EQ(native_.unmap_count, 1);
  EXPECT_EQ(native_.destroy_count, 1);
}

}  // namespace
