// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/fabric/memory_backend_test_base.h"

namespace iree::hal::amd::fabric {

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
    iree_hal_memory_site_t site = {
        .kind = IREE_HAL_MEMORY_SITE_PROGRAM,
        .family = families_[device_index],
    };
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
  EXPECT_EQ(native_.pair_query_count, 9);
  iree_hal_pool_release(pool);
}

TEST_F(MemoryBackendTest,
       NestedPoolsAndCacheShareNativeFactsAndPhysicalCharge) {
  native_.native_prefix = 128;
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool));
  iree_hal_slab_cache_options_t cache_options;
  iree_hal_slab_cache_options_initialize(&cache_options);
  cache_options.slab = iree_hal_make_pool_reservation_request(4096, 128);
  cache_options.max_count = 1;
  iree_hal_pool_t* cache = nullptr;
  IREE_ASSERT_OK(
      iree_hal_slab_cache_create(pool, &cache_options, allocator_, &cache));

  const iree_hal_buffer_backing_facts_t* original_backing = nullptr;
  for (int iteration = 0; iteration < 2; ++iteration) {
    iree_hal_buffer_t* slab = nullptr;
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        cache, cache_options.slab, iree_infinite_timeout(), &slab));
    const auto memory = iree_hal_buffer_memory_view(slab);
    ASSERT_NE(memory.backing, nullptr);
    if (!iteration) {
      original_backing = memory.backing;
    }
    EXPECT_EQ(memory.backing, original_backing);
    EXPECT_EQ(iree_hal_buffer_byte_length(slab), 4096u);
    EXPECT_EQ(memory.backing->native_allocation.byte_length, 8192u);
    EXPECT_EQ(memory.backing->native_allocation.byte_offset, 128u);
    EXPECT_EQ(memory.backing->native_allocation.granularity, 4096u);
    EXPECT_EQ(memory.backing->allocation_alignment, 4096u);
    EXPECT_EQ(memory.backing->maintenance_range.byte_offset, 0u);
    EXPECT_EQ(memory.backing->maintenance_range.byte_length, 4096u);
    EXPECT_EQ(memory.backing->host_cacheability,
              IREE_HAL_HOST_CACHEABILITY_WRITE_BACK);
    EXPECT_EQ(memory.backing->physical_backing_id.words[0], 0xAu);
    EXPECT_EQ(memory.backing->physical_backing_id.words[1], 1u);

    iree_hal_fixed_block_pool_options_t child_options = {
        .block_size = 256,
    };
    iree_hal_pool_t* child = nullptr;
    IREE_ASSERT_OK(iree_hal_fixed_block_pool_create_from_buffer(
        slab, 128, 2048, &child_options, allocator_, &child));
    iree_hal_buffer_t* buffer = nullptr;
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        child, iree_hal_make_pool_reservation_request(256, 128),
        iree_infinite_timeout(), &buffer));
    const auto child_memory = iree_hal_buffer_memory_view(buffer);
    EXPECT_EQ(child_memory.backing, memory.backing);
    EXPECT_EQ(child_memory.bindings, memory.bindings);
    EXPECT_EQ(child_memory.offset, memory.offset + 128);
    const uint8_t pattern = 0x45 + iteration;
    iree_hal_buffer_mapping_t mapping;
    IREE_ASSERT_OK(iree_hal_buffer_map_range(
        buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_WRITE,
        IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, 256, &mapping));
    memset(mapping.contents.data, pattern, mapping.contents.data_length);
    IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
    IREE_ASSERT_OK(iree_hal_buffer_map_range(
        slab, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
        IREE_HAL_BUFFER_MAP_FLAG_NONE, 128, 256, &mapping));
    for (iree_host_size_t i = 0; i < mapping.contents.data_length; ++i) {
      EXPECT_EQ(mapping.contents.data[i], pattern) << i;
    }
    IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
    iree_hal_buffer_release(buffer);
    iree_hal_pool_release(child);
    iree_hal_buffer_release(slab);

    iree_hal_pool_stats_t stats;
    iree_hal_pool_query_stats(pool, &stats);
    EXPECT_EQ(stats.bytes_reserved, 8192u);
    EXPECT_EQ(stats.bytes_committed, 8192u);
    EXPECT_EQ(native_.create_count, 1);
    EXPECT_EQ(native_.destroy_count, 0);
  }
  iree_hal_slab_cache_stats_t cache_stats;
  IREE_ASSERT_OK(iree_hal_slab_cache_query_stats(cache, &cache_stats));
  EXPECT_EQ(cache_stats.ready_count, 1u);
  EXPECT_EQ(cache_stats.hit_count, 1u);

  iree_hal_pool_trim(cache, IREE_HAL_POOL_TRIM_FLAG_ALL, 0);
  iree_hal_pool_stats_t stats;
  while (true) {
    const uint32_t token = iree_async_notification_begin_observe(notification_);
    iree_hal_pool_query_stats(pool, &stats);
    if (!stats.bytes_committed) {
      iree_async_notification_end_observe(notification_);
      break;
    }
    iree_async_notification_wait_for_token(notification_, token,
                                           iree_infinite_timeout());
    iree_async_notification_end_observe(notification_);
  }
  EXPECT_EQ(stats.bytes_reserved, 0u);
  EXPECT_EQ(stats.bytes_committed, 0u);
  EXPECT_EQ(native_.destroy_count, 1);
  iree_hal_pool_release(cache);
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
  iree_status_t status =
      CreatePool(IREE_HAL_BUFFER_USAGE_TRANSFER, &pool, /*interfaces=*/0,
                 IREE_HAL_MEMORY_ACCESS_NONE);
  EXPECT_THAT(status, StatusIs(iree::StatusCode::kUnavailable));
  iree_status_free(status);
  EXPECT_EQ(pool, nullptr);
  EXPECT_EQ(native_.pair_query_count, 0);
}

TEST_F(MemoryBackendTest, JoinsOwnedAndRegisteredVisibilityAtConstruction) {
  native_.registration_enabled = true;
  const iree_hal_external_storage_class_t storage_class = {
      IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER, {0, 0}};
  const iree_hal_pool_import_class_t import_class = {storage_class,
                                                     {1, &storage_class}};
  pool_options_.imports.class_count = 1;
  pool_options_.imports.classes = &import_class;
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool));
  const int query_count = native_.pair_query_count;
  EXPECT_EQ(query_count, 18);
  const auto table = iree_hal_pool_transition_table(pool);
  iree_hal_memory_transition_pair_t pair;
  IREE_ASSERT_OK(iree_hal_memory_transition_prepare_pair(
      table, ResolveScope(IREE_HAL_MEMORY_SITE_HOST),
      ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, families_[0]),
      IREE_HAL_MEMORY_TRANSITION_RELEASE, &pair));
  const auto* recipe = iree_hal_memory_transition_recipe(
      table, pair, IREE_HAL_MEMORY_TRANSITION_RELEASE);
  ASSERT_NE(recipe, nullptr);
  ASSERT_EQ(recipe->operation_count, 1u);
  EXPECT_EQ(recipe->operations[0].executor,
            IREE_HAL_MEMORY_TRANSITION_EXECUTOR_HOST_API);
  EXPECT_EQ(recipe->operations[0].operation,
            IREE_HAL_MEMORY_TRANSITION_OPERATION_HOST_FLUSH);
  EXPECT_EQ(recipe->operations[0].host.instruction,
            IREE_HAL_HOST_CACHE_INSTRUCTION_NONE);

  iree_hal_buffer_t* owned = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(128, 64),
      iree_infinite_timeout(), &owned));
  iree_hal_external_storage_t exported = {};
  iree_hal_buffer_range_t range = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_hal_buffer_export(owned, iree_hal_make_buffer_range(0, 64),
                             storage_class, &exported, &range));
  alignas(4096) uint8_t allocation[4096] = {};
  uint8_t* bytes = allocation + 128;
  iree_hal_external_storage_t storage = {
      .storage_class = storage_class,
  };
  storage.payload.host_pointer = bytes;
  storage.byte_length = 512;
  storage.host_cacheability = IREE_HAL_HOST_CACHEABILITY_WRITE_BACK;
  storage.maintenance_range = iree_hal_make_buffer_range(128, 256);
  storage.physical_backing_id = {{0xC, 0x1234}};
  iree_hal_buffer_t* imported = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_import_buffer(
      pool, &storage, iree_hal_make_buffer_range(128, 256), &imported));
  EXPECT_EQ(storage.byte_length, 0u);
  const auto memory = iree_hal_buffer_memory_view(imported);
  ASSERT_NE(memory.backing, nullptr);
  EXPECT_EQ(memory.offset, 0u);
  EXPECT_EQ(memory.backing->native_allocation.byte_length, 4096u);
  EXPECT_EQ(memory.backing->native_allocation.byte_offset, 256u);
  EXPECT_EQ(memory.backing->native_allocation.granularity, 4096u);
  EXPECT_EQ(memory.backing->allocation_alignment, 256u);
  EXPECT_EQ(memory.backing->host_cacheability,
            IREE_HAL_HOST_CACHEABILITY_WRITE_BACK);
  EXPECT_EQ(memory.backing->maintenance_range.byte_offset, 0u);
  EXPECT_EQ(memory.backing->maintenance_range.byte_length, 256u);
  EXPECT_EQ(memory.backing->physical_backing_id.words[0], 0xCu);
  EXPECT_EQ(memory.backing->physical_backing_id.words[1], 0x1234u);
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_EQ(stats.bytes_reserved, 4096u);
  EXPECT_EQ(stats.bytes_committed, 4096u);
  IREE_ASSERT_OK(iree_hal_buffer_export(imported,
                                        iree_hal_make_buffer_range(32, 64),
                                        storage_class, &exported, &range));
  EXPECT_EQ(exported.payload.host_pointer, bytes + 128);
  EXPECT_EQ(exported.byte_length, 256u);
  EXPECT_EQ(range.byte_offset, 32u);
  EXPECT_EQ(range.byte_length, 64u);
  EXPECT_EQ(exported.physical_backing_id.words[0], 0xCu);
  EXPECT_EQ(exported.physical_backing_id.words[1], 0x1234u);
  EXPECT_EQ(exported.maintenance_range.byte_offset, 0u);
  EXPECT_EQ(exported.maintenance_range.byte_length, 256u);
  EXPECT_EQ(native_.pair_query_count, query_count);
  iree_hal_external_storage_release(&exported);
  iree_hal_buffer_release(imported);
  iree_hal_buffer_release(owned);
  iree_hal_pool_release(pool);
  native_.AwaitEmpty();
  EXPECT_EQ(native_.create_count, 2);
  EXPECT_EQ(native_.destroy_count, 2);
}

TEST_F(MemoryBackendTest, RegistrationRequiresItsOwnVisibilityQualification) {
  native_.registration_enabled = true;
  native_.registration_transition_unknown = true;
  const iree_hal_pool_import_class_t import_class = {
      {IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER, {0, 0}}, {}};
  pool_options_.imports.class_count = 1;
  pool_options_.imports.classes = &import_class;
  iree_hal_pool_t* pool = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool));
  EXPECT_EQ(pool, nullptr);
  EXPECT_EQ(native_.create_count, 0);
}

TEST_F(MemoryBackendTest, NativeImportsQualifyExactTransportAndPrivateCpuUses) {
  native_.native_transport_enabled = true;
  native_.required_flags = 0;
  iree_hal_pool_import_class_t import_class = {
      {IREE_HAL_EXTERNAL_STORAGE_TYPE_DMA_BUF_FD, {0, 0}}, {}};
  pool_options_.owned.mode = IREE_HAL_SLAB_POOL_IMPORT_ONLY;
  pool_options_.imports.class_count = 1;
  pool_options_.imports.classes = &import_class;
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool,
                            /*interfaces=*/0, IREE_HAL_MEMORY_ACCESS_NONE));
  iree_hal_pool_import_requirements_t requirements;
  ASSERT_TRUE(iree_hal_pool_query_import_requirements(
      pool, import_class.storage_class, &requirements));
  EXPECT_EQ(requirements.byte_length_granularity, 12288u);
  iree_hal_pool_release(pool);
  pool = nullptr;

  native_.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool));
  EXPECT_EQ(pool, nullptr);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      CreatePool(IREE_HAL_BUFFER_USAGE_TRANSFER, &pool, /*interfaces=*/0,
                 IREE_HAL_MEMORY_ACCESS_NONE));
  EXPECT_EQ(pool, nullptr);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool, /*interfaces=*/0,
                 IREE_HAL_MEMORY_ACCESS_NONE, IREE_HAL_BUFFER_USAGE_STORAGE));
  EXPECT_EQ(pool, nullptr);
  EXPECT_EQ(native_.create_count, 0);

  // Matching the transport type does not admit an unqualified protocol.
  import_class.storage_class.provenance[0] = 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool, /*interfaces=*/0,
                 IREE_HAL_MEMORY_ACCESS_NONE));
  EXPECT_EQ(pool, nullptr);
}

TEST_F(MemoryBackendTest, NativeExportsCaptureShareableConstructionAndExtent) {
  native_.native_transport_enabled = true;
  native_.required_flags = AMDF_MEMORY_FLAG_SHAREABLE;
  const iree_hal_external_storage_class_t storage_class = {
      IREE_HAL_EXTERNAL_STORAGE_TYPE_DMA_BUF_FD, {0, 0}};
  pool_options_.owned.exports = {1, &storage_class};
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool,
                            /*interfaces=*/0, IREE_HAL_MEMORY_ACCESS_NONE));
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(64, 1),
      iree_infinite_timeout(), &buffer));
  EXPECT_EQ(iree_hal_buffer_byte_length(buffer), 64u);
  EXPECT_EQ(native_.map_count, 0);
  iree_hal_buffer_release(buffer);
  iree_hal_pool_release(pool);
}

TEST_F(MemoryBackendTest, CoherentOwnedAndCachedImportedUseMappingRecipes) {
  native_.registration_enabled = true;
  native_.host_coherent = true;
  const iree_hal_pool_import_class_t import_class = {
      {IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER, {0, 0}}, {}};
  pool_options_.imports.class_count = 1;
  pool_options_.imports.classes = &import_class;
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool));
  const auto table = iree_hal_pool_transition_table(pool);
  iree_hal_memory_transition_pair_t pair;
  IREE_ASSERT_OK(iree_hal_memory_transition_prepare_pair(
      table, ResolveScope(IREE_HAL_MEMORY_SITE_HOST),
      ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, families_[0]),
      IREE_HAL_MEMORY_TRANSITION_RELEASE, &pair));
  const auto* recipe = iree_hal_memory_transition_recipe(
      table, pair, IREE_HAL_MEMORY_TRANSITION_RELEASE);
  ASSERT_NE(recipe, nullptr);
  ASSERT_EQ(recipe->operation_count, 1u);
  EXPECT_EQ(recipe->operations[0].executor,
            IREE_HAL_MEMORY_TRANSITION_EXECUTOR_HOST_API);
  iree_hal_pool_release(pool);
}

TEST_F(MemoryBackendTest, HostExportDoesNotGrantPublicMapping) {
  const iree_hal_external_storage_class_t storage_class = {
      IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER, {0, 0}};
  pool_options_.owned.exports = {1, &storage_class};
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(
      CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool,
                 UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA,
                 IREE_HAL_MEMORY_ACCESS_NONE));
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(128, 64),
      iree_infinite_timeout(), &buffer));
  iree_hal_buffer_mapping_t mapping;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_buffer_map_range(
          buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
          IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, 64, &mapping));
  iree_hal_external_storage_t exported = {};
  iree_hal_buffer_range_t range = {};
  IREE_ASSERT_OK(iree_hal_buffer_export(buffer,
                                        iree_hal_make_buffer_range(32, 64),
                                        storage_class, &exported, &range));
  EXPECT_NE(exported.payload.host_pointer, nullptr);
  EXPECT_EQ(exported.byte_length, 4096u);
  EXPECT_EQ(exported.host_cacheability, IREE_HAL_HOST_CACHEABILITY_WRITE_BACK);
  EXPECT_EQ(exported.release, nullptr);
  EXPECT_EQ(range.byte_offset, 32u);
  EXPECT_EQ(range.byte_length, 64u);
  iree_hal_external_storage_release(&exported);
  EXPECT_EQ(native_.destroy_count, 0);
  iree_hal_buffer_release(buffer);
  iree_hal_pool_release(pool);
  native_.AwaitEmpty();
  EXPECT_EQ(native_.destroy_count, 1);
}

TEST_F(MemoryBackendTest, WrapperFailureRollsBackBeforeConsumingStorage) {
  native_.registration_enabled = true;
  const iree_hal_pool_import_class_t import_class = {
      {IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER, {0, 0}}, {}};
  pool_options_.owned.mode = IREE_HAL_SLAB_POOL_IMPORT_ONLY;
  pool_options_.imports.class_count = 1;
  pool_options_.imports.classes = &import_class;
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool));
  alignas(128) uint8_t bytes[512] = {};
  struct ReleaseState {
    // Native dependency observing detach before the caller callback.
    FakeNative* native;
    // Original transport pointer, independent of the registered envelope.
    void* pointer;
    // Guards the caller-owned external storage lifetime.
    std::mutex mutex;
    // Published by the final external storage release callback.
    std::condition_variable condition;
    // Callback executions, guarded by mutex after ownership transfer.
    int count = 0;
  } release_state = {&native_, bytes};
  iree_hal_external_storage_t storage = {
      .storage_class = import_class.storage_class,
  };
  storage.payload.host_pointer = bytes;
  storage.byte_length = sizeof(bytes);
  storage.host_cacheability = IREE_HAL_HOST_CACHEABILITY_WRITE_BACK;
  storage.maintenance_range = iree_hal_make_buffer_range(64, 384);
  storage.release = +[](void* user_data, iree_hal_external_storage_class_t,
                        iree_hal_external_storage_payload_t payload) {
    auto* state = static_cast<ReleaseState*>(user_data);
    EXPECT_EQ(payload.host_pointer, state->pointer);
    EXPECT_EQ(state->native->destroy_count, 2);
    EXPECT_EQ(state->native->unmap_count, 2);
    std::lock_guard<std::mutex> lock(state->mutex);
    ++state->count;
    state->condition.notify_all();
  };
  storage.release_user_data = &release_state;
  const auto original = storage;
  native_.fail_after_bindings = true;
  iree_hal_buffer_t* buffer = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_pool_import_buffer(
          pool, &storage, iree_hal_make_buffer_range(128, 256), &buffer));
  EXPECT_EQ(buffer, nullptr);
  EXPECT_EQ(memcmp(&storage, &original, sizeof(storage)), 0);
  EXPECT_EQ(release_state.count, 0);
  EXPECT_EQ(native_.destroy_count, 1);
  EXPECT_EQ(native_.unmap_count, 1);
  native_.fail_after_bindings = false;
  IREE_ASSERT_OK(iree_hal_pool_import_buffer(
      pool, &storage, iree_hal_make_buffer_range(128, 256), &buffer));
  EXPECT_EQ(storage.release, nullptr);
  EXPECT_EQ(release_state.count, 0);
  EXPECT_EQ(iree_hal_buffer_memory_view(buffer).offset, 64u);
  iree_hal_buffer_release(buffer);
  iree_hal_pool_release(pool);
  std::unique_lock<std::mutex> lock(release_state.mutex);
  release_state.condition.wait(lock, [&] { return release_state.count == 1; });
  EXPECT_EQ(release_state.count, 1);
}

TEST_F(MemoryBackendTest, RejectsLogicalRangeOutsideRegistrationEnvelope) {
  native_.registration_enabled = true;
  const iree_hal_pool_import_class_t import_class = {
      .storage_class = {IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER, {0, 0}},
  };
  pool_options_.owned.mode = IREE_HAL_SLAB_POOL_IMPORT_ONLY;
  pool_options_.imports = {1, &import_class};
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool));
  alignas(128) uint8_t bytes[512] = {};
  const iree_hal_external_storage_t original = {
      .storage_class = import_class.storage_class,
      .payload = {.host_pointer = bytes},
      .byte_length = sizeof(bytes),
      .host_cacheability = IREE_HAL_HOST_CACHEABILITY_WRITE_BACK,
      .maintenance_range = {128, 256},
  };
  for (const auto logical_range : {iree_hal_make_buffer_range(64, 256),
                                   iree_hal_make_buffer_range(256, 256)}) {
    auto storage = original;
    iree_hal_buffer_t* buffer = nullptr;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_OUT_OF_RANGE,
        iree_hal_pool_import_buffer(pool, &storage, logical_range, &buffer));
    EXPECT_EQ(buffer, nullptr);
    EXPECT_EQ(storage.payload.host_pointer, original.payload.host_pointer);
    EXPECT_EQ(storage.byte_length, original.byte_length);
    EXPECT_EQ(storage.maintenance_range.byte_offset,
              original.maintenance_range.byte_offset);
    EXPECT_EQ(storage.maintenance_range.byte_length,
              original.maintenance_range.byte_length);
  }
  EXPECT_EQ(native_.create_count, 2);
  EXPECT_EQ(native_.destroy_count, 2);
  EXPECT_EQ(native_.unmap_count, 2);
  iree_hal_pool_release(pool);
}

TEST_F(MemoryBackendTest, RejectsUnsupportedDeviceAddressInterface) {
  iree_hal_pool_t* pool = nullptr;
  iree_status_t status =
      CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool,
                 UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS);
  EXPECT_THAT(status, StatusIs(iree::StatusCode::kUnavailable));
  iree_status_free(status);
  EXPECT_EQ(pool, nullptr);
  EXPECT_EQ(native_.profile_query_count, 0);
}

TEST_F(MemoryBackendTest, MixedEnginesAllocateOneBackingWithExactBindings) {
  binding_descriptions_[0] = {IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
                              AMDF_MEMORY_ADDRESS_GPU};
  native_.address_kinds[0] = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU;
  storage_[0][0].memory_type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(
      CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool, /*interfaces=*/0));
  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(pool, &capabilities);
  EXPECT_FALSE(iree_all_bits_set(capabilities.memory_type,
                                 IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL));
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(64, 64),
      iree_infinite_timeout(), &buffer));
  EXPECT_EQ(native_.create_count, 1);
  EXPECT_EQ(native_.map_count, 1);
  for (uint32_t i = 0; i < 2; ++i) {
    const auto program_scope =
        ResolveScope(IREE_HAL_MEMORY_SITE_PROGRAM, families_[i]);
    iree_hal_buffer_native_binding_slot_t slot;
    IREE_ASSERT_OK(iree_hal_pool_resolve_binding(
        pool, program_scope, binding_descriptions_[i].type, &slot));
    EXPECT_EQ(slot.index, i + 1);
    EXPECT_EQ(iree_hal_buffer_native_binding(buffer, slot).device_address,
              UINT64_C(0x100000) + i * UINT64_C(0x10000));
  }
  iree_hal_buffer_release(buffer);
  iree_hal_pool_release(pool);
  native_.AwaitEmpty();
  EXPECT_EQ(native_.unmap_count, 1);
  EXPECT_EQ(native_.destroy_count, 1);
}

TEST_F(MemoryBackendTest, DeviceOnlyContractOmitsHostBindingAndHostQueries) {
  for (uint32_t i = 0; i < 2; ++i) {
    native_families_[i].host_usage = 0;
    native_families_[i].minimum_access = AMDF_MEMORY_ACCESS_READ;
    binding_descriptions_[i] = {IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
                                AMDF_MEMORY_ADDRESS_GPU};
    native_.address_kinds[i] = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU;
    storage_[i][0] = storage_[i][1];
  }
  native_.required_flags = 0;
  native_.host_mappable = false;
  native_.minimum_access = AMDF_MEMORY_ACCESS_READ;
  native_.supports_private_mapping = false;
  native_.device_access = AMDF_MEMORY_ACCESS_READ;
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE_READ, &pool,
                            /*interfaces=*/0, IREE_HAL_MEMORY_ACCESS_NONE));
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(64, 64),
      iree_infinite_timeout(), &buffer));
  EXPECT_EQ(buffer->host_binding_index,
            IREE_HAL_BUFFER_NATIVE_BINDING_INDEX_NONE);
  EXPECT_EQ(native_.map_count, 0);
  // Neither device is a writer, so there are no visibility edges to query.
  EXPECT_EQ(native_.pair_query_count, 0);
  for (uint32_t i = 0; i < 2; ++i) {
    iree_hal_buffer_native_binding_slot_t slot;
    IREE_ASSERT_OK(iree_hal_pool_resolve_binding(
        pool, ResolveScope(IREE_HAL_MEMORY_SITE_PROGRAM, families_[i]),
        IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS, &slot));
    EXPECT_EQ(slot.index, i);
    EXPECT_EQ(iree_hal_buffer_native_binding(buffer, slot).device_address,
              UINT64_C(0x100000) + i * UINT64_C(0x10000));
  }
  iree_hal_buffer_release(buffer);
  iree_hal_pool_release(pool);
  native_.AwaitEmpty();
  EXPECT_EQ(native_.unmap_count, 0);
  EXPECT_EQ(native_.destroy_count, 1);
}

TEST_F(MemoryBackendTest, IndependentNativeInstancesCannotShareBacking) {
  FakeNative independent;
  backends_[1].native.instance =
      reinterpret_cast<amdf_instance_t*>(&independent);
  iree_hal_pool_t* pool = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool));
  EXPECT_EQ(pool, nullptr);
  EXPECT_EQ(native_.profile_query_count, 0);
  EXPECT_EQ(independent.profile_query_count, 0);
}

TEST_F(MemoryBackendTest, RequiredPlacementSelectsBackingOwner) {
  for (auto& backend : backends_) {
    backend.native.storage_count = 2;
  }
  // CreatePool supplies families in reverse order. Both placements must retain
  // the same canonical consumer bindings and choose the requested native owner.
  for (uint32_t i = 0; i < devices_.size(); ++i) {
    SCOPED_TRACE(i);
    pool_options_.placement = {IREE_HAL_POOL_PLACEMENT_REQUIRED,
                               PhysicalNode(i)};
    iree_hal_pool_t* pool = nullptr;
    IREE_ASSERT_OK(CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool));
    iree_hal_pool_capabilities_t capabilities;
    iree_hal_pool_query_capabilities(pool, &capabilities);
    EXPECT_EQ(capabilities.placement.mode, IREE_HAL_POOL_PLACEMENT_REQUIRED);
    EXPECT_EQ(capabilities.placement.node, PhysicalNode(i));
    EXPECT_TRUE(iree_all_bits_set(capabilities.memory_type,
                                  IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL));
    EXPECT_FALSE(iree_all_bits_set(capabilities.memory_type,
                                   IREE_HAL_MEMORY_TYPE_HOST_LOCAL));
    iree_hal_buffer_t* buffer = nullptr;
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        pool, iree_hal_make_pool_reservation_request(64, 64),
        iree_infinite_timeout(), &buffer));
    EXPECT_EQ(native_.created_scope, storage_[i][1].scope);
    EXPECT_EQ(iree_hal_buffer_allocation_placement(buffer).device, devices_[i]);
    for (uint32_t d = 0; d < devices_.size(); ++d) {
      iree_hal_buffer_native_binding_slot_t slot;
      IREE_ASSERT_OK(iree_hal_pool_resolve_binding(
          pool, ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, families_[d]),
          IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA, &slot));
      EXPECT_EQ(iree_hal_buffer_native_binding(buffer, slot).device_address,
                UINT64_C(0x100000) + d * UINT64_C(0x10000));
    }
    iree_hal_buffer_release(buffer);
    iree_hal_pool_release(pool);
  }
  native_.AwaitEmpty();
  EXPECT_EQ(native_.create_count, 2);
  EXPECT_EQ(native_.destroy_count, 2);
}

TEST_F(MemoryBackendTest, RequiredPlacementRejectsUnavailablePeerRoute) {
  for (auto& backend : backends_) {
    backend.native.storage_count = 2;
  }
  native_.local_scopes[1].profile_status =
      amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  pool_options_.placement = {IREE_HAL_POOL_PLACEMENT_REQUIRED, PhysicalNode(1)};
  iree_hal_pool_t* pool = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool));
  EXPECT_EQ(pool, nullptr);
  EXPECT_EQ(native_.create_count, 0);
}

TEST_F(MemoryBackendTest, HostNumaPlacementPreservesAllConsumerBindings) {
  native_.host_placement_enabled = true;
  const auto* cpu_family = iree_hal_device_queue_family(cpu_device_, 0);
  for (const auto mode :
       {IREE_HAL_POOL_PLACEMENT_REQUIRED, IREE_HAL_POOL_PLACEMENT_PREFERRED}) {
    SCOPED_TRACE(mode);
    pool_options_.placement = {mode, HostNode()};
    iree_hal_pool_t* pool = nullptr;
    IREE_ASSERT_OK(CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool,
                              /*interfaces=*/0, IREE_HAL_MEMORY_ACCESS_ALL,
                              IREE_HAL_BUFFER_USAGE_TRANSFER));
    iree_hal_pool_capabilities_t capabilities;
    iree_hal_pool_query_capabilities(pool, &capabilities);
    EXPECT_EQ(capabilities.placement.mode, IREE_HAL_POOL_PLACEMENT_REQUIRED);
    EXPECT_EQ(capabilities.placement.node, HostNode());
    EXPECT_TRUE(iree_all_bits_set(
        capabilities.memory_type,
        IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE));
    EXPECT_FALSE(iree_all_bits_set(capabilities.memory_type,
                                   IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL));
    const int profile_queries = native_.profile_query_count;
    const int pair_queries = native_.pair_query_count;
    iree_hal_buffer_t* buffer = nullptr;
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        pool, iree_hal_make_pool_reservation_request(64, 64),
        iree_infinite_timeout(), &buffer));
    EXPECT_EQ(native_.created_scope, native_.scope());
    EXPECT_EQ(native_.created_profile_ordinal, 1u);
    EXPECT_EQ(native_.created_placement.kind,
              AMDF_MEMORY_PLACEMENT_KIND_HOST_NUMA);
    EXPECT_EQ(native_.created_placement.node, host_node_id_);
    ASSERT_EQ(native_.created_accesses.size(), 2u);
    for (uint32_t i = 0; i < families_.size(); ++i) {
      iree_hal_buffer_native_binding_slot_t slot;
      IREE_ASSERT_OK(iree_hal_pool_resolve_binding(
          pool, ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, families_[i]),
          IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA, &slot));
      EXPECT_EQ(iree_hal_buffer_native_binding(buffer, slot).device_address,
                UINT64_C(0x100000) + i * UINT64_C(0x10000));
      EXPECT_EQ(native_.created_accesses[i].device, native_.devices[i]);
    }
    iree_hal_buffer_native_binding_slot_t cpu_slot;
    IREE_ASSERT_OK(iree_hal_pool_resolve_binding(
        pool, ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, cpu_family),
        IREE_HAL_BUFFER_INTERFACE_HOST, &cpu_slot));
    EXPECT_NE(iree_hal_buffer_native_binding(buffer, cpu_slot).host_pointer,
              nullptr);
    EXPECT_EQ(native_.profile_query_count, profile_queries);
    EXPECT_EQ(native_.pair_query_count, pair_queries);
    iree_hal_buffer_release(buffer);
    iree_hal_pool_release(pool);
  }
  native_.AwaitEmpty();
  EXPECT_EQ(native_.create_count, 2);
  EXPECT_EQ(native_.destroy_count, 2);
}

TEST_F(MemoryBackendTest, HostNumaPreferenceFallsBackOnlyAtPoolConstruction) {
  pool_options_.placement = {IREE_HAL_POOL_PLACEMENT_REQUIRED, HostNode()};
  iree_hal_pool_t* pool = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool));
  EXPECT_EQ(pool, nullptr);
  pool_options_.placement.mode = IREE_HAL_POOL_PLACEMENT_PREFERRED;
  IREE_ASSERT_OK(CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool));
  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(pool, &capabilities);
  EXPECT_EQ(capabilities.placement.mode, IREE_HAL_POOL_PLACEMENT_AUTOMATIC);
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(64, 64),
      iree_infinite_timeout(), &buffer));
  EXPECT_EQ(native_.created_profile_ordinal, 0u);
  EXPECT_EQ(native_.created_placement.kind, AMDF_MEMORY_PLACEMENT_KIND_NONE);
  iree_hal_buffer_release(buffer);
  iree_hal_pool_release(pool);
  native_.AwaitEmpty();
}

TEST_F(MemoryBackendTest, PlacedAllocationFailureDoesNotRelocate) {
  native_.host_placement_enabled = true;
  pool_options_.placement = {IREE_HAL_POOL_PLACEMENT_PREFERRED, HostNode()};
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool));
  native_.create_failure =
      amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  iree_hal_buffer_t* buffer = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_pool_allocate_buffer(
          pool, iree_hal_make_pool_reservation_request(64, 64),
          iree_infinite_timeout(), &buffer));
  EXPECT_EQ(buffer, nullptr);
  EXPECT_EQ(native_.create_count, 1);
  EXPECT_EQ(native_.created_placement.kind,
            AMDF_MEMORY_PLACEMENT_KIND_HOST_NUMA);
  EXPECT_EQ(native_.created_placement.node, host_node_id_);
  iree_hal_pool_release(pool);
  native_.AwaitEmpty();
  EXPECT_EQ(native_.destroy_count, 0);
}

TEST_F(MemoryBackendTest, HostPlacementDoesNotChangeAutomaticSelection) {
  native_.host_placement_enabled = true;
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool));
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(64, 64),
      iree_infinite_timeout(), &buffer));
  EXPECT_EQ(native_.created_profile_ordinal, 0u);
  EXPECT_EQ(native_.created_placement.kind, AMDF_MEMORY_PLACEMENT_KIND_NONE);
  iree_hal_buffer_release(buffer);
  iree_hal_pool_release(pool);
  native_.AwaitEmpty();
}

TEST_F(MemoryBackendTest, CpuOnlyHostPlacementNeedsNoNativeAttachment) {
  native_.host_placement_enabled = true;
  pool_options_.placement = {IREE_HAL_POOL_PLACEMENT_REQUIRED, HostNode()};
  const auto* cpu_family = iree_hal_device_queue_family(cpu_device_, 0);
  const iree_hal_pool_family_access_t family = {
      .family = cpu_family,
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
  };
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(iree_hal_slab_pool_create(group_, {1, &family, {}},
                                           &pool_options_, allocator_, &pool));
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(64, 64),
      iree_infinite_timeout(), &buffer));
  EXPECT_EQ(native_.created_scope, native_.scope());
  EXPECT_EQ(native_.created_placement.node, host_node_id_);
  EXPECT_TRUE(native_.created_accesses.empty());
  EXPECT_EQ(iree_hal_pool_query_host_access(pool).access,
            IREE_HAL_MEMORY_ACCESS_NONE);
  iree_hal_buffer_native_binding_slot_t slot;
  IREE_ASSERT_OK(iree_hal_pool_resolve_binding(
      pool, ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, cpu_family),
      IREE_HAL_BUFFER_INTERFACE_HOST, &slot));
  EXPECT_NE(iree_hal_buffer_native_binding(buffer, slot).host_pointer, nullptr);
  for (auto* native_family : families_) {
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_PERMISSION_DENIED,
        iree_hal_pool_resolve_binding(
            pool, ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, native_family),
            IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA, &slot));
  }
  iree_hal_buffer_release(buffer);
  iree_hal_pool_release(pool);
  native_.AwaitEmpty();
  EXPECT_EQ(native_.create_count, 1);
  EXPECT_EQ(native_.destroy_count, 1);
}

TEST_F(MemoryBackendTest, HostRegistrationKeepsSourcePlacement) {
  native_.host_placement_enabled = true;
  native_.registration_enabled = true;
  const iree_hal_pool_import_class_t import_class = {
      {IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER, {0, 0}}, {}};
  pool_options_.imports = {1, &import_class};
  pool_options_.placement = {IREE_HAL_POOL_PLACEMENT_REQUIRED, HostNode()};
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool));
  iree_hal_buffer_t* owned = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(64, 64),
      iree_infinite_timeout(), &owned));
  EXPECT_EQ(native_.created_profile_ordinal, 2u);
  EXPECT_EQ(native_.created_placement.kind,
            AMDF_MEMORY_PLACEMENT_KIND_HOST_NUMA);

  alignas(4096) uint8_t bytes[4096] = {};
  iree_hal_external_storage_t storage = {
      .storage_class = import_class.storage_class,
  };
  storage.payload.host_pointer = bytes;
  storage.byte_length = sizeof(bytes);
  storage.host_cacheability = IREE_HAL_HOST_CACHEABILITY_WRITE_BACK;
  storage.maintenance_range = iree_hal_make_buffer_range(0, sizeof(bytes));
  iree_hal_buffer_t* imported = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_import_buffer(
      pool, &storage, iree_hal_make_buffer_range(0, sizeof(bytes)), &imported));
  EXPECT_EQ(native_.created_profile_ordinal, 1u);
  EXPECT_EQ(native_.created_placement.kind, AMDF_MEMORY_PLACEMENT_KIND_NONE);
  iree_hal_buffer_release(imported);
  iree_hal_buffer_release(owned);
  iree_hal_pool_release(pool);
  native_.AwaitEmpty();
  EXPECT_EQ(native_.create_count, 2);
  EXPECT_EQ(native_.destroy_count, 2);
}

TEST_F(MemoryBackendTest, PlacementOwnerNeedNotExposeAnAccessFamily) {
  native_.minimum_access = AMDF_MEMORY_ACCESS_READ;
  native_.device_access = AMDF_MEMORY_ACCESS_READ;
  for (uint32_t i = 0; i < devices_.size(); ++i) {
    backends_[i].native.storage_count = 2;
    native_families_[i].minimum_access = AMDF_MEMORY_ACCESS_READ;
    storage_[i][1].minimum_access = AMDF_MEMORY_ACCESS_READ;
  }
  for (auto mode :
       {IREE_HAL_POOL_PLACEMENT_REQUIRED, IREE_HAL_POOL_PLACEMENT_PREFERRED}) {
    for (uint32_t owner = 0; owner < devices_.size(); ++owner) {
      SCOPED_TRACE(::testing::Message() << mode << ":" << owner);
      const uint32_t consumer = 1 - owner;
      pool_options_.placement = {mode, PhysicalNode(owner)};
      const iree_hal_pool_family_access_t family = {
          .family = families_[consumer],
          .usage = IREE_HAL_BUFFER_USAGE_STORAGE_READ,
      };
      // Host writes do not require broader permissions on a private GPU owner.
      const iree_hal_pool_scope_t scope = {
          1,
          &family,
          {IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_MAPPING_MODE_SCOPED}};
      iree_hal_pool_t* pool = nullptr;
      IREE_ASSERT_OK(iree_hal_slab_pool_create(group_, scope, &pool_options_,
                                               allocator_, &pool));
      iree_hal_pool_capabilities_t capabilities;
      iree_hal_pool_query_capabilities(pool, &capabilities);
      EXPECT_EQ(capabilities.placement.mode, IREE_HAL_POOL_PLACEMENT_REQUIRED);
      EXPECT_EQ(capabilities.placement.node, PhysicalNode(owner));
      iree_hal_buffer_t* buffer = nullptr;
      IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
          pool, iree_hal_make_pool_reservation_request(64, 64),
          iree_infinite_timeout(), &buffer));
      EXPECT_EQ(native_.created_scope, storage_[owner][1].scope);
      ASSERT_EQ(native_.created_accesses.size(), 2u);
      EXPECT_EQ(native_.created_accesses[0].device, native_.devices[consumer]);
      EXPECT_EQ(native_.created_accesses[1].device, native_.devices[owner]);
      EXPECT_EQ(native_.created_accesses[1].requirements.address_kinds, 0u);
      EXPECT_EQ(native_.created_accesses[1].requirements.flags, 0u);
      EXPECT_EQ(native_.created_accesses[1].requirements.access,
                AMDF_MEMORY_ACCESS_READ);
      EXPECT_EQ(iree_hal_buffer_allocation_placement(buffer).device,
                devices_[owner]);
      const auto consumer_scope =
          ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, families_[consumer]);
      const auto owner_scope =
          ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, families_[owner]);
      iree_hal_buffer_native_binding_slot_t slot;
      IREE_ASSERT_OK(iree_hal_pool_resolve_binding(
          pool, consumer_scope, IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA,
          &slot));
      EXPECT_EQ(iree_hal_buffer_native_binding(buffer, slot).device_address,
                UINT64_C(0x100000) + consumer * UINT64_C(0x10000));
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_PERMISSION_DENIED,
          iree_hal_pool_resolve_binding(pool, owner_scope,
                                        IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA,
                                        &slot));
      const auto transition =
          iree_hal_pool_query_transition(pool, owner_scope, consumer_scope);
      EXPECT_FALSE(iree_hal_memory_effects_is_supported(transition.release));
      EXPECT_FALSE(iree_hal_memory_effects_is_supported(transition.acquire));
      iree_hal_buffer_release(buffer);
      iree_hal_pool_release(pool);
      native_.AwaitEmpty();
    }
  }
  EXPECT_EQ(native_.create_count, 4);
  EXPECT_EQ(native_.destroy_count, 4);
}

TEST_F(MemoryBackendTest, CombinesGlobalAndRangedHostConstructionRoutes) {
  native_.global_host_transitions = true;
  const iree_hal_pool_import_class_t import_class = {
      .storage_class = {IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER, {0, 0}},
  };
  for (bool registration : {false, true}) {
    SCOPED_TRACE(registration);
    native_.registration_enabled = registration;
    pool_options_.imports = {registration ? 1u : 0u, &import_class};
    iree_hal_pool_t* pool = nullptr;
    IREE_ASSERT_OK(CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool));
    const auto table = iree_hal_pool_transition_table(pool);
    const auto host = ResolveScope(IREE_HAL_MEMORY_SITE_HOST);
    const auto queue = ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, families_[0]);
    for (auto action : {IREE_HAL_MEMORY_TRANSITION_RELEASE,
                        IREE_HAL_MEMORY_TRANSITION_ACQUIRE}) {
      iree_hal_memory_transition_pair_t pair;
      IREE_ASSERT_OK(iree_hal_memory_transition_prepare_pair(
          table, action == IREE_HAL_MEMORY_TRANSITION_RELEASE ? host : queue,
          action == IREE_HAL_MEMORY_TRANSITION_RELEASE ? queue : host, action,
          &pair));
      const auto* recipe =
          iree_hal_memory_transition_recipe(table, pair, action);
      ASSERT_NE(recipe, nullptr);
      ASSERT_EQ(recipe->operation_count, 1u);
      const auto& operation = recipe->operations[0];
      EXPECT_EQ(operation.kind, registration
                                    ? IREE_HAL_MEMORY_TRANSITION_KIND_RANGE
                                    : IREE_HAL_MEMORY_TRANSITION_KIND_GLOBAL);
      EXPECT_EQ(operation.range_granularity, registration ? 64u : 0u);
      EXPECT_EQ(operation.executor,
                registration || action == IREE_HAL_MEMORY_TRANSITION_RELEASE
                    ? IREE_HAL_MEMORY_TRANSITION_EXECUTOR_HOST_API
                    : IREE_HAL_MEMORY_TRANSITION_EXECUTOR_HOST_DIRECT);
      IREE_EXPECT_OK(iree_hal_memory_transition_recipe_validate(recipe));
    }
    iree_hal_pool_release(pool);
  }
}

TEST_F(MemoryBackendTest, CpuAccessKeepsNativeOwnerPrivate) {
  for (auto& backend : backends_) {
    backend.native.storage_count = 2;
  }
  const auto* cpu_family = iree_hal_device_queue_family(cpu_device_, 0);
  const iree_hal_pool_family_access_t family = {
      .family = cpu_family,
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
  };
  for (bool public_host : {false, true}) {
    SCOPED_TRACE(public_host);
    iree_hal_pool_scope_t scope = {1, &family, {}};
    if (public_host) {
      scope = {0,
               nullptr,
               {IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_MAPPING_MODE_SCOPED}};
    }
    for (uint32_t owner = 0; owner < devices_.size(); ++owner) {
      SCOPED_TRACE(owner);
      pool_options_.placement = {IREE_HAL_POOL_PLACEMENT_REQUIRED,
                                 PhysicalNode(owner)};
      iree_hal_pool_t* pool = nullptr;
      IREE_ASSERT_OK(iree_hal_slab_pool_create(group_, scope, &pool_options_,
                                               allocator_, &pool));
      EXPECT_EQ(iree_hal_pool_query_host_access(pool).access,
                public_host ? IREE_HAL_MEMORY_ACCESS_ALL
                            : IREE_HAL_MEMORY_ACCESS_NONE);
      iree_hal_buffer_t* buffer = nullptr;
      IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
          pool, iree_hal_make_pool_reservation_request(64, 64),
          iree_infinite_timeout(), &buffer));
      EXPECT_EQ(native_.created_scope, storage_[owner][1].scope);
      ASSERT_EQ(native_.created_accesses.size(), 1u);
      EXPECT_EQ(native_.created_accesses[0].device, native_.devices[owner]);
      EXPECT_EQ(native_.created_accesses[0].requirements.address_kinds, 0u);
      EXPECT_EQ(native_.created_accesses[0].requirements.flags, 0u);
      iree_hal_buffer_native_binding_slot_t slot;
      IREE_ASSERT_OK(iree_hal_pool_resolve_binding(
          pool,
          public_host ? ResolveScope(IREE_HAL_MEMORY_SITE_HOST)
                      : ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, cpu_family),
          IREE_HAL_BUFFER_INTERFACE_HOST, &slot));
      EXPECT_NE(iree_hal_buffer_native_binding(buffer, slot).host_pointer,
                nullptr);
      for (auto* native_family : families_) {
        IREE_EXPECT_STATUS_IS(
            IREE_STATUS_PERMISSION_DENIED,
            iree_hal_pool_resolve_binding(
                pool, ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, native_family),
                IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA, &slot));
      }
      iree_hal_buffer_release(buffer);
      iree_hal_pool_release(pool);
      native_.AwaitEmpty();
    }
    native_.host_mappable = false;
    iree_hal_pool_t* pool = nullptr;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_UNAVAILABLE,
        iree_hal_slab_pool_create(group_, scope, &pool_options_, allocator_,
                                  &pool));
    EXPECT_EQ(pool, nullptr);
    native_.host_mappable = true;
  }
}

TEST_F(MemoryBackendTest, NonparticipatingPlacementPreservesDomainAndFallback) {
  for (auto& backend : backends_) {
    backend.native.storage_count = 2;
  }
  const iree_hal_pool_family_access_t family = {
      .family = families_[1],
      .usage = IREE_HAL_BUFFER_USAGE_STORAGE,
  };
  const iree_hal_pool_scope_t scope = {
      1, &family, {IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_MAPPING_MODE_SCOPED}};
  for (bool foreign_instance : {false, true}) {
    SCOPED_TRACE(foreign_instance);
    FakeNative foreign;
    if (foreign_instance) {
      backends_[0].native.instance =
          reinterpret_cast<amdf_instance_t*>(&foreign);
    } else {
      native_.local_scopes[0].profile_status =
          amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
    }
    pool_options_.placement = {IREE_HAL_POOL_PLACEMENT_REQUIRED,
                               PhysicalNode(0)};
    iree_hal_pool_t* pool = nullptr;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_UNAVAILABLE,
        iree_hal_slab_pool_create(group_, scope, &pool_options_, allocator_,
                                  &pool));
    EXPECT_EQ(pool, nullptr);
    pool_options_.placement.mode = IREE_HAL_POOL_PLACEMENT_PREFERRED;
    IREE_ASSERT_OK(iree_hal_slab_pool_create(group_, scope, &pool_options_,
                                             allocator_, &pool));
    iree_hal_pool_capabilities_t capabilities;
    iree_hal_pool_query_capabilities(pool, &capabilities);
    // Equal-scoring candidates retain the consumer's shared SYSTEM route.
    EXPECT_EQ(capabilities.placement.mode, IREE_HAL_POOL_PLACEMENT_AUTOMATIC);
    iree_hal_pool_release(pool);
    native_.local_scopes[0].profile_status = AMDF_STATUS_OK;
    backends_[0].native.instance = reinterpret_cast<amdf_instance_t*>(&native_);
  }
}

TEST_F(MemoryBackendTest,
       AutomaticPlacementPrefersQualifiedDeviceLocalStorage) {
  for (uint32_t i = 0; i < devices_.size(); ++i) {
    storage_[i][0].memory_type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
    backends_[i].native.storage_count = 2;
  }
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool));
  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(pool, &capabilities);
  EXPECT_EQ(capabilities.placement.mode, IREE_HAL_POOL_PLACEMENT_REQUIRED);
  EXPECT_EQ(capabilities.placement.node, PhysicalNode(0));
  iree_hal_pool_release(pool);
}

TEST_F(MemoryBackendTest, SystemScopeCannotPromisePhysicalPlacement) {
  pool_options_.placement = {IREE_HAL_POOL_PLACEMENT_REQUIRED, PhysicalNode(0)};
  iree_hal_pool_t* pool = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool));
  EXPECT_EQ(pool, nullptr);
  EXPECT_EQ(native_.profile_query_count, 0);
  EXPECT_EQ(native_.create_count, 0);
}

TEST_F(MemoryBackendTest, SharedSystemLocalityRequiresEveryNativeConsumer) {
  // The second consumer can access shared system storage but only publishes
  // locality facts for its own local scope. Native access qualification must
  // not imply that system memory is local to that device.
  storage_[1][0] = storage_[1][1];
  native_.local_scopes[1].profile_status =
      amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool));
  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(pool, &capabilities);
  EXPECT_TRUE(iree_all_bits_set(capabilities.memory_type,
                                IREE_HAL_MEMORY_TYPE_HOST_LOCAL));
  EXPECT_TRUE(iree_all_bits_set(capabilities.memory_type,
                                IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE));
  EXPECT_FALSE(iree_all_bits_set(capabilities.memory_type,
                                 IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL));
  iree_hal_pool_release(pool);
}

TEST_F(MemoryBackendTest, PreferredPlacementReportsAchievedBacking) {
  for (auto& backend : backends_) {
    backend.native.storage_count = 2;
  }
  pool_options_.placement = {IREE_HAL_POOL_PLACEMENT_PREFERRED,
                             PhysicalNode(1)};
  for (const auto native_status :
       {AMDF_STATUS_OK, amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)}) {
    SCOPED_TRACE(native_status);
    native_.local_scopes[1].profile_status = native_status;
    iree_hal_pool_t* pool = nullptr;
    IREE_ASSERT_OK(CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool));
    iree_hal_pool_capabilities_t capabilities;
    iree_hal_pool_query_capabilities(pool, &capabilities);
    EXPECT_EQ(capabilities.placement.mode,
              native_status == AMDF_STATUS_OK
                  ? IREE_HAL_POOL_PLACEMENT_REQUIRED
                  : IREE_HAL_POOL_PLACEMENT_AUTOMATIC);
    if (native_status == AMDF_STATUS_OK) {
      EXPECT_EQ(capabilities.placement.node, PhysicalNode(1));
    }
    iree_hal_buffer_t* buffer = nullptr;
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        pool, iree_hal_make_pool_reservation_request(64, 64),
        iree_infinite_timeout(), &buffer));
    EXPECT_EQ(native_.created_scope, native_status == AMDF_STATUS_OK
                                         ? storage_[1][1].scope
                                         : native_.scope());
    iree_hal_buffer_release(buffer);
    iree_hal_pool_release(pool);
  }
  native_.AwaitEmpty();
}

TEST_F(MemoryBackendTest, OneDeviceCanPublishSeveralAddressInterfaces) {
  const iree_hal_amd_fabric_memory_binding_t bindings[] = {
      {IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA, AMDF_MEMORY_ADDRESS_XDNA_DMA},
      {IREE_HAL_BUFFER_INTERFACE_XDNA_FIRMWARE,
       AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE},
  };
  native_families_[0].binding_count = IREE_ARRAYSIZE(bindings);
  native_families_[0].bindings = bindings;
  native_.address_kinds[0] |= UINT64_C(1) << AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE;
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(
      CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool, /*interfaces=*/0));
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(64, 64),
      iree_infinite_timeout(), &buffer));
  const auto first = ResolveScope(IREE_HAL_MEMORY_SITE_PROGRAM, families_[0]);
  const auto second = ResolveScope(IREE_HAL_MEMORY_SITE_PROGRAM, families_[1]);
  iree_hal_buffer_native_binding_slot_t slot;
  IREE_ASSERT_OK(iree_hal_pool_resolve_binding(
      pool, first, IREE_HAL_BUFFER_INTERFACE_XDNA_FIRMWARE, &slot));
  EXPECT_EQ(slot.index, 2);
  EXPECT_EQ(iree_hal_buffer_native_binding(buffer, slot).device_address,
            UINT64_C(0x108000));
  IREE_ASSERT_OK(iree_hal_pool_resolve_binding(
      pool, second, IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA, &slot));
  EXPECT_EQ(slot.index, 3);
  EXPECT_EQ(iree_hal_buffer_native_binding(buffer, slot).device_address,
            UINT64_C(0x110000));
  EXPECT_EQ(native_.create_count, 1);
  iree_hal_buffer_release(buffer);
  iree_hal_pool_release(pool);
  native_.AwaitEmpty();
  EXPECT_EQ(native_.destroy_count, 1);
}

TEST_F(MemoryBackendTest, CpuExecutionUsesNativeHostViewWithoutPublicMapping) {
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool,
                            /*interfaces=*/0, IREE_HAL_MEMORY_ACCESS_NONE,
                            IREE_HAL_BUFFER_USAGE_STORAGE));
  EXPECT_EQ(iree_hal_pool_query_host_access(pool).access,
            IREE_HAL_MEMORY_ACCESS_NONE);
  const auto* cpu_family = iree_hal_device_queue_family(cpu_device_, 0);
  const auto cpu_queue = ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, cpu_family);
  const auto cpu_program =
      ResolveScope(IREE_HAL_MEMORY_SITE_PROGRAM, cpu_family);
  const auto npu_queue = ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, families_[0]);
  for (const auto cpu : {cpu_queue, cpu_program}) {
    const auto publication =
        iree_hal_pool_query_transition(pool, cpu, npu_queue);
    EXPECT_EQ(publication.release.bits, IREE_HAL_MEMORY_EFFECT_HOST_FLUSH);
    EXPECT_EQ(publication.acquire.bits, 0u);
    const auto observation =
        iree_hal_pool_query_transition(pool, npu_queue, cpu);
    EXPECT_EQ(observation.release.bits, 0u);
    EXPECT_EQ(observation.acquire.bits, IREE_HAL_MEMORY_EFFECT_HOST_INVALIDATE);
    iree_hal_buffer_native_binding_slot_t slot;
    IREE_ASSERT_OK(iree_hal_pool_resolve_binding(
        pool, cpu, IREE_HAL_BUFFER_INTERFACE_HOST, &slot));
    EXPECT_EQ(slot.index, 0);
  }
  const auto local =
      iree_hal_pool_query_transition(pool, cpu_queue, cpu_program);
  EXPECT_EQ(local.release.bits, 0u);
  EXPECT_EQ(local.acquire.bits, 0u);
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(256, 64),
      iree_infinite_timeout(), &buffer));
  EXPECT_EQ(native_.create_count, 1);
  EXPECT_EQ(native_.map_count, 1);
  iree_byte_span_t span;
  IREE_ASSERT_OK(iree_hal_buffer_native_host_span(buffer, 64, 128, &span));
  EXPECT_EQ(span.data_length, 128u);
  iree_hal_buffer_mapping_t mapping;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_buffer_map_range(
          buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
          IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, 64, &mapping));
  iree_hal_buffer_release(buffer);
  iree_hal_pool_release(pool);
  native_.AwaitEmpty();
  EXPECT_EQ(native_.destroy_count, 1);
  EXPECT_EQ(native_.unmap_count, 1);
}

TEST_F(MemoryBackendTest, CpuReadOnlyAccessDoesNotRequirePrivateWrites) {
  for (auto& family : native_families_) {
    family.host_usage = 0;
  }
  native_.supports_private_mapping = false;
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool,
                            /*interfaces=*/0, IREE_HAL_MEMORY_ACCESS_NONE,
                            IREE_HAL_BUFFER_USAGE_STORAGE_READ));
  const auto cpu = ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE,
                                iree_hal_device_queue_family(cpu_device_, 0));
  const auto npu = ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, families_[0]);
  const auto observation = iree_hal_pool_query_transition(pool, npu, cpu);
  EXPECT_EQ(observation.acquire.bits, IREE_HAL_MEMORY_EFFECT_HOST_INVALIDATE);
  iree_hal_pool_release(pool);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool, /*interfaces=*/0,
                 IREE_HAL_MEMORY_ACCESS_NONE, IREE_HAL_BUFFER_USAGE_STORAGE));
  EXPECT_EQ(pool, nullptr);
}

TEST_F(MemoryBackendTest, CoherentCpuTransitionsHaveNoCacheRecipe) {
  native_.host_coherent = true;
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool,
                            /*interfaces=*/0, IREE_HAL_MEMORY_ACCESS_NONE,
                            IREE_HAL_BUFFER_USAGE_STORAGE));
  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(pool, &capabilities);
  EXPECT_TRUE(iree_any_bit_set(capabilities.memory_type,
                               IREE_HAL_MEMORY_TYPE_HOST_COHERENT));
  const auto cpu = ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE,
                                iree_hal_device_queue_family(cpu_device_, 0));
  const auto device = ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, families_[0]);
  const auto table = iree_hal_pool_transition_table(pool);
  for (const auto producer : {cpu, device}) {
    iree_hal_memory_transition_pair_t pair;
    IREE_ASSERT_OK(iree_hal_memory_transition_prepare_pair(
        table, producer, producer.id == cpu.id ? device : cpu,
        IREE_HAL_MEMORY_TRANSITION_RELEASE, &pair));
    const auto transition = iree_hal_memory_transition_query(table, pair);
    EXPECT_EQ(transition.release.bits, 0u);
    EXPECT_EQ(transition.acquire.bits, 0u);
    EXPECT_EQ(iree_hal_memory_transition_recipe(
                  table, pair, IREE_HAL_MEMORY_TRANSITION_RELEASE),
              nullptr);
    EXPECT_EQ(iree_hal_memory_transition_recipe(
                  table, pair, IREE_HAL_MEMORY_TRANSITION_ACQUIRE),
              nullptr);
  }
  iree_hal_pool_release(pool);
}

TEST_F(MemoryBackendTest, RejectsAtomicPromisesAbsentFromNativeBuffers) {
  for (auto cell :
       {&iree_hal_atomic_operation_capabilities_t::device_scope_32,
        &iree_hal_atomic_operation_capabilities_t::device_scope_64,
        &iree_hal_atomic_operation_capabilities_t::system_scope_32,
        &iree_hal_atomic_operation_capabilities_t::system_scope_64}) {
    for (uint32_t operation = 1;
         operation <= IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL; operation <<= 1) {
      iree_hal_atomic_operation_capabilities_t atomics = {};
      atomics.*cell = operation;
      iree_hal_pool_t* pool = nullptr;
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_UNAVAILABLE,
          CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool, /*interfaces=*/0,
                     IREE_HAL_MEMORY_ACCESS_NONE, IREE_HAL_BUFFER_USAGE_STORAGE,
                     atomics));
      EXPECT_EQ(pool, nullptr);
    }
  }
  EXPECT_EQ(native_.profile_query_count, 0);
}

TEST_F(MemoryBackendTest, PreparesOneHostSourceForCpuAndPublicHostScopes) {
  native_.mapping_enabled = true;
  native_.host_reservation_enabled = true;
  native_.host_source_enabled = true;
  native_.required_flags |= AMDF_MEMORY_FLAG_HOST_MAPPING_SOURCE;
  native_.native_prefix = 4096;
  const uint64_t source_interface = UINT64_C(1)
                                    << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE;
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool,
                            source_interface, IREE_HAL_MEMORY_ACCESS_ALL,
                            IREE_HAL_BUFFER_USAGE_STORAGE, {},
                            source_interface));
  const auto requirements = iree_hal_pool_query_mapping_requirements(pool);
  EXPECT_EQ(requirements.minimum_granularity, 4096u);
  iree_hal_buffer_t* backing = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(16384, 4096),
      iree_infinite_timeout(), &backing));
  iree_hal_buffer_t* view = nullptr;
  IREE_ASSERT_OK(
      iree_hal_buffer_subspan(backing, 4096, 8192, allocator_, &view));
  const auto* cpu_family = iree_hal_device_queue_family(cpu_device_, 0);
  const iree_hal_memory_scope_t scopes[] = {
      ResolveScope(IREE_HAL_MEMORY_SITE_HOST),
      ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, cpu_family),
      ResolveScope(IREE_HAL_MEMORY_SITE_PROGRAM, cpu_family),
  };
  const void* descriptor = nullptr;
  for (auto scope : scopes) {
    iree_hal_buffer_native_binding_slot_t slot;
    IREE_ASSERT_OK(iree_hal_pool_resolve_binding(
        pool, scope, IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE, &slot));
    const auto source =
        iree_hal_buffer_native_binding(view, slot).mapping_source;
    ASSERT_NE(source.descriptor, nullptr);
    EXPECT_EQ(source.offset, 8192u);
    EXPECT_EQ(source.granularity, 4096u);
    if (descriptor) {
      EXPECT_EQ(source.descriptor, descriptor);
    }
    descriptor = source.descriptor;
  }
  for (const auto* family : families_) {
    iree_hal_buffer_native_binding_slot_t slot;
    IREE_ASSERT_OK(iree_hal_pool_resolve_binding(
        pool, ResolveScope(IREE_HAL_MEMORY_SITE_QUEUE, family),
        IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE, &slot));
    EXPECT_EQ(
        iree_hal_buffer_native_binding(view, slot).mapping_source.descriptor,
        descriptor);
  }
  iree_hal_buffer_release(view);
  iree_hal_buffer_release(backing);
  iree_hal_pool_release(pool);
  native_.AwaitEmpty();
  EXPECT_EQ(native_.create_count, 1);
  EXPECT_EQ(native_.destroy_count, 1);
}

TEST_F(MemoryBackendTest, HostVisibilityDoesNotImplyHostMappingSource) {
  native_.host_reservation_enabled = true;
  iree_hal_pool_t* pool = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool, 0,
                 IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_BUFFER_USAGE_STORAGE, {},
                 UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE));
  EXPECT_EQ(pool, nullptr);
  EXPECT_EQ(native_.create_count, 0);
  EXPECT_GT(native_.profile_query_count, 0);
}

TEST_F(MemoryBackendTest, MappingSourcesHonorRequiredBackingPreparation) {
  native_.mapping_enabled = true;
  native_.host_source_enabled = true;
  native_.required_flags = AMDF_MEMORY_FLAG_HOST_MAPPING_SOURCE;
  // Only one domain needs host-remappable backing; both consume the same
  // allocation. The whole-backing requirement must survive the joint query.
  native_.reservation_source_flags[1] = AMDF_MEMORY_FLAG_HOST_MAPPING_SOURCE;
  iree_hal_pool_t* pool = nullptr;
  // No CPU family or public mapping requests the host backing property.
  IREE_ASSERT_OK(
      CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool,
                 UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE, 0));
  EXPECT_EQ(iree_hal_pool_query_host_access(pool).access, 0u);
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(4096, 4096),
      iree_infinite_timeout(), &buffer));
  EXPECT_EQ(native_.create_count, 1);
  EXPECT_EQ(native_.map_count, 0);
  iree_hal_buffer_release(buffer);
  iree_hal_pool_release(pool);

  native_.host_source_enabled = false;
  pool = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool,
                 UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE, 0));
  EXPECT_EQ(pool, nullptr);
  EXPECT_EQ(native_.create_count, 1);
}

TEST_F(MemoryBackendTest, IndependentNpuMappingsNeedNoRemappableHostBacking) {
  native_.mapping_enabled = true;
  native_.required_flags = 0;
  iree_hal_pool_t* pool = nullptr;
  // The same XDNA address kind may use independent device translations. Its
  // source profile has no remappable host route and needs no host preparation.
  IREE_ASSERT_OK(
      CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, &pool,
                 UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE, 0));
  EXPECT_EQ(iree_hal_pool_query_host_access(pool).access, 0u);
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(4096, 4096),
      iree_infinite_timeout(), &buffer));
  EXPECT_EQ(native_.create_count, 1);
  EXPECT_EQ(native_.map_count, 0);
  iree_hal_buffer_release(buffer);
  iree_hal_pool_release(pool);
}

}  // namespace iree::hal::amd::fabric
