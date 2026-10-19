// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/fabric/cts/virtual_buffer_test_base.h"

namespace iree::hal::amd::fabric::cts {

TEST_P(FabricVirtualBufferTest, BatchedReplacementChangesRealGpuAddresses) {
  // A shared group does not make native backing formats interchangeable.
  // Rejection leaves this live group usable for the valid GPU mappings below.
  {
    const iree_hal_pool_family_access_t family = {
        .family = iree_hal_device_queue_family(cpu_, 0),
        .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
    };
    const iree_hal_virtual_buffer_options_t options = {
        .byte_length = granularity_,
        .mapping_granularity = granularity_,
    };
    iree_hal_buffer_t* target = nullptr;
    IREE_ASSERT_OK(iree_hal_virtual_buffer_create(
        group_, {1, &family, {}}, &options, allocator_, &target));
    const iree_hal_memory_mapping_t mapping = {
        .target = target,
        .scope = {iree_hal_device_group_memory_domain(group_),
                  IREE_HAL_MEMORY_SCOPE_ANY},
        .target_range = {0, granularity_},
        .source = backing_,
        .access = IREE_HAL_MEMORY_ACCESS_ALL,
    };
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_PERMISSION_DENIED,
        iree_hal_queue_map(iree_hal_device_queue(cpu_, 0, 0),
                           iree_hal_semaphore_list_empty(),
                           iree_hal_semaphore_list_empty(), 1, &mapping,
                           IREE_HAL_QUEUE_MAP_FLAG_NONE));
    iree_hal_buffer_release(target);
  }
  auto* words = reinterpret_cast<uint32_t*>(host_mapping_.contents.data);
  const size_t stride = granularity_ / sizeof(uint32_t);
  std::memset(words, 0, host_mapping_.contents.data_length);
  for (uint32_t i = 0; i < 128; ++i) {
    words[i] = 0x1000 + i;
    words[stride + i] = 0x2000 + i;
  }
  std::array<iree_hal_memory_mapping_t, 2> mappings = {};
  for (uint32_t i = 0; i < 2; ++i) {
    mappings[i] = {
        .target = reservations_[i],
        .scope = scope_,
        .target_range = {0, granularity_},
        .source = backing_,
        .source_offset = i * granularity_,
        .access = IREE_HAL_MEMORY_ACCESS_NONE,
        .operation = IREE_HAL_MEMORY_MAPPING_OPERATION_BIND,
    };
  }
  IREE_ASSERT_OK(Map(mappings));
  for (auto& mapping : mappings) {
    mapping.operation = IREE_HAL_MEMORY_MAPPING_OPERATION_PROTECT;
    mapping.access = IREE_HAL_MEMORY_ACCESS_ALL;
  }
  IREE_ASSERT_OK(Map(mappings));
  ASSERT_NO_FATAL_FAILURE(CopyAndVerify(0x1000, 0x2000));
  for (auto& mapping : mappings) {
    mapping.access = IREE_HAL_MEMORY_ACCESS_NONE;
  }
  IREE_ASSERT_OK(Map(mappings));
  for (auto& mapping : mappings) {
    mapping.access = IREE_HAL_MEMORY_ACCESS_ALL;
  }
  IREE_ASSERT_OK(Map(mappings));
  ASSERT_NO_FATAL_FAILURE(CopyAndVerify(0x1000, 0x2000));
  for (auto& mapping : mappings) {
    mapping.operation = IREE_HAL_MEMORY_MAPPING_OPERATION_REPLACE;
  }
  std::swap(mappings[0].source_offset, mappings[1].source_offset);
  IREE_ASSERT_OK(Map(mappings));
  ASSERT_NO_FATAL_FAILURE(CopyAndVerify(0x2000, 0x1000));
  for (auto& mapping : mappings) {
    mapping.operation = IREE_HAL_MEMORY_MAPPING_OPERATION_UNBIND;
    mapping.source = nullptr;
    mapping.source_offset = 0;
    mapping.access = 0;
  }
  IREE_ASSERT_OK(Map(mappings));
}

TEST_P(FabricVirtualBufferTest, CpuAndGpuShareStableViewsAcrossReplacement) {
  using iree::hal::cts::Ref;
  const uint64_t source_interface = UINT64_C(1)
                                    << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE;
  std::array<iree_hal_pool_family_access_t, 2> families = {{
      {
          .family = access_.family,
          .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
          .interfaces = source_interface,
      },
      {
          .family = iree_hal_device_queue_family(cpu_, 0),
          .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
          .interfaces = source_interface,
      },
  }};
  iree_hal_pool_scope_t scope = {
      families.size(),
      families.data(),
      {IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_MAPPING_MODE_PERSISTENT,
       IREE_HAL_HOST_CACHEABILITY_UNKNOWN}};
  iree_hal_slab_pool_options_t pool_options;
  iree_hal_slab_pool_options_initialize(&pool_options);
  Ref<iree_hal_pool_t> pool;
  auto pool_status = iree_hal_slab_pool_create(group_, scope, &pool_options,
                                               allocator_, pool.out());
  if (iree_status_is_unavailable(pool_status)) {
    GTEST_SKIP() << "no jointly remappable host/device backing: "
                 << iree::Status(std::move(pool_status)).ToString();
  }
  IREE_ASSERT_OK(pool_status);
  families[0].interfaces = UINT64_C(1)
                           << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS;
  families[1].interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_HOST;
  iree_hal_virtual_buffer_capabilities_t capabilities;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_query_capabilities(
      group_, scope, allocator_, &capabilities));
  ASSERT_NE(capabilities.max_byte_length, 0u);
  const auto unit = std::max(capabilities.reservation_granularity,
                             capabilities.mapping_granularity);
  const size_t stride = unit / sizeof(uint32_t);
  std::array<Ref<iree_hal_buffer_t>, 2> backings;
  std::array<iree_byte_span_t, 2> original_spans;
  std::array<std::vector<uint32_t>, 2> expected;
  for (uint32_t i = 0; i < 2; ++i) {
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        pool, iree_hal_make_pool_reservation_request(6 * unit, unit),
        iree_infinite_timeout(), backings[i].out()));
    IREE_ASSERT_OK(iree_hal_buffer_native_host_span(backings[i], 0, 6 * unit,
                                                    &original_spans[i]));
    expected[i].assign(6 * stride, 0xabab0000u + i);
    std::memcpy(original_spans[i].data, expected[i].data(), 6 * unit);
    IREE_ASSERT_OK(iree_hal_buffer_flush_host_range(backings[i], 0, 6 * unit));
  }
  const iree_hal_virtual_buffer_options_t options = {
      .byte_length = 2 * unit,
      .mapping_granularity = unit,
  };
  Ref<iree_hal_buffer_t> root;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_create(group_, scope, &options,
                                                allocator_, root.out()));
  std::array<Ref<iree_hal_buffer_t>, 2> views;
  std::array<iree_byte_span_t, 2> host_views;
  std::array<uint64_t, 2> addresses;
  const iree_hal_buffer_native_binding_slot_t gpu_slot = {
      .index = root.get()
                   ->memory.contract->scopes[scope_.id]
                   .bindings[IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS],
      .type = IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
  };
  for (uint32_t i = 0; i < 2; ++i) {
    IREE_ASSERT_OK(iree_hal_buffer_subspan(root, i * unit, unit, allocator_,
                                           views[i].out()));
    IREE_ASSERT_OK(
        iree_hal_buffer_native_host_span(views[i], 256, 256, &host_views[i]));
    addresses[i] =
        iree_hal_buffer_native_binding(views[i], gpu_slot).device_address;
  }
  iree_hal_memory_scope_t cpu_scope;
  IREE_ASSERT_OK(iree_hal_device_group_resolve_memory_scope(
      group_, {IREE_HAL_MEMORY_SITE_QUEUE, families[1].family}, &cpu_scope));
  iree_hal_memory_scope_t host_scope;
  IREE_ASSERT_OK(iree_hal_device_group_resolve_memory_scope(
      group_, {IREE_HAL_MEMORY_SITE_HOST, nullptr}, &host_scope));
  uint32_t cpu_projection, host_projection, gpu_projection;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_resolve_projection(root, cpu_scope,
                                                            &cpu_projection));
  IREE_ASSERT_OK(iree_hal_virtual_buffer_resolve_projection(root, host_scope,
                                                            &host_projection));
  IREE_ASSERT_OK(iree_hal_virtual_buffer_resolve_projection(root, scope_,
                                                            &gpu_projection));
  EXPECT_EQ(cpu_projection, host_projection);
  EXPECT_NE(cpu_projection, gpu_projection);
  // Original subspans retain the address owner after the creator releases it.
  root.reset();
  const auto staging_address =
      iree_hal_buffer_native_binding(backing_, source_slot_).device_address;
  auto* staging = reinterpret_cast<uint32_t*>(host_mapping_.contents.data);
  std::array<iree_hal_memory_mapping_t, 2> mappings = {};
  for (uint32_t epoch = 0; epoch < 3; ++epoch) {
    for (uint32_t i = 0; i < 2; ++i) {
      const uint32_t backing_index = (epoch + i) % 2;
      const uint32_t source_unit = 1 + epoch + i;
      mappings[i] = {
          .target = views[i],
          .scope = {iree_hal_device_group_memory_domain(group_),
                    IREE_HAL_MEMORY_SCOPE_ANY},
          .target_range = {0, unit},
          .source = backings[backing_index],
          .source_offset = source_unit * unit,
          .access = IREE_HAL_MEMORY_ACCESS_ALL,
          .operation = epoch ? IREE_HAL_MEMORY_MAPPING_OPERATION_REPLACE
                             : IREE_HAL_MEMORY_MAPPING_OPERATION_BIND,
      };
    }
    IREE_ASSERT_OK(Map(mappings));
    SdmaCommandWriter commands(
        reinterpret_cast<uint32_t*>(queue_host_.ring_address) + ring_words_,
        native_family_.format_features);
    const bool cache_control =
        native_family_.format_features & AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR;
    if (cache_control) {
      commands.AcquireFromSystem();
    }
    for (uint32_t i = 0; i < 2; ++i) {
      EXPECT_EQ(
          iree_hal_buffer_native_binding(views[i], gpu_slot).device_address,
          addresses[i]);
      iree_byte_span_t current;
      IREE_ASSERT_OK(
          iree_hal_buffer_native_host_span(views[i], 256, 256, &current));
      EXPECT_EQ(current.data, host_views[i].data);
      auto* words = reinterpret_cast<uint32_t*>(host_views[i].data);
      for (uint32_t j = 0; j < 64; ++j) {
        words[j] = 0x100000u + epoch * 0x10000u + i * 0x100u + j;
        staging[i * granularity_ / 4 + j] =
            0x200000u + epoch * 0x10000u + i * 0x100u + j;
        expected[(epoch + i) % 2][(1 + epoch + i) * stride + 64 + j] =
            staging[i * granularity_ / 4 + j];
      }
      IREE_ASSERT_OK(iree_hal_buffer_flush_host_range(views[i], 256, 256));
      commands.CopyLinear(addresses[i] + 256,
                          staging_address + (2 + i) * granularity_, 256);
      commands.CopyLinear(staging_address + i * granularity_,
                          addresses[i] + 256, 256);
    }
    if (cache_control) {
      commands.ReleaseToSystem();
    }
    ASSERT_NO_FATAL_FAILURE(SubmitAndWait(commands));
    // Observe the stable alias first: touching the original mapping must not
    // accidentally supply the cache maintenance this path promises.
    for (uint32_t i = 0; i < 2; ++i) {
      IREE_ASSERT_OK(iree_hal_buffer_invalidate_host_range(views[i], 256, 256));
      const auto* words = reinterpret_cast<const uint32_t*>(host_views[i].data);
      for (uint32_t j = 0; j < 64; ++j) {
        ASSERT_EQ(words[j], 0x200000u + epoch * 0x10000u + i * 0x100u + j);
        ASSERT_EQ(staging[(2 + i) * granularity_ / 4 + j],
                  0x100000u + epoch * 0x10000u + i * 0x100u + j);
      }
    }
    for (uint32_t i = 0; i < 2; ++i) {
      IREE_ASSERT_OK(
          iree_hal_buffer_invalidate_host_range(backings[i], 0, 6 * unit));
      const auto* words =
          reinterpret_cast<const uint32_t*>(original_spans[i].data);
      for (size_t j = 0; j < expected[i].size(); ++j) {
        ASSERT_EQ(words[j], expected[i][j]) << "backing=" << i << " word=" << j;
      }
    }
  }
  // Detach requires the full address interval, recovered from a retained view.
  auto* owner = &iree_hal_buffer_virtual_view(views[0]).root->buffer;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_detach(
      owner, {iree_hal_device_group_memory_domain(group_),
              IREE_HAL_MEMORY_SCOPE_ANY}));
  for (auto& view : views) {
    view.reset();
  }
  for (auto& backing : backings) {
    backing.reset();
  }
}

TEST_P(FabricVirtualBufferTest,
       RequiredLocalBackingSurvivesVirtualReplacement) {
  using iree::hal::cts::Ref;
  if (storage_.size() == 1) {
    GTEST_SKIP() << "native device has no independently placed local storage";
  }
  uint32_t physical_node = IREE_HAL_TOPOLOGY_NODE_ORDINAL_INVALID;
  const auto* topology = iree_hal_device_group_topology(group_);
  for (iree_host_size_t i = 0; i < iree_hal_topology_node_count(topology);
       ++i) {
    const auto* node = iree_hal_topology_node_at(topology, i);
    if (node->kind == IREE_HAL_TOPOLOGY_NODE_KIND_PHYSICAL_DEVICE &&
        node->device_ordinal ==
            iree_hal_device_topology_info(device_)->topology_index) {
      physical_node = node->ordinal;
      break;
    }
  }
  ASSERT_NE(physical_node, IREE_HAL_TOPOLOGY_NODE_ORDINAL_INVALID);
  iree_hal_pool_family_access_t family = {
      .family = access_.family,
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
      .interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE,
  };
  iree_hal_slab_pool_options_t pool_options;
  iree_hal_slab_pool_options_initialize(&pool_options);
  pool_options.placement = {IREE_HAL_POOL_PLACEMENT_REQUIRED, physical_node};
  Ref<iree_hal_pool_t> local_pool;
  IREE_ASSERT_OK(iree_hal_slab_pool_create(
      group_, {1, &family, {}}, &pool_options, allocator_, local_pool.out()));
  iree_hal_pool_capabilities_t pool_capabilities;
  iree_hal_pool_query_capabilities(local_pool, &pool_capabilities);
  ASSERT_EQ(pool_capabilities.placement.mode, IREE_HAL_POOL_PLACEMENT_REQUIRED);
  ASSERT_EQ(pool_capabilities.placement.node, physical_node);
  ASSERT_TRUE(iree_all_bits_set(pool_capabilities.memory_type,
                                IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL));
  ASSERT_FALSE(iree_all_bits_set(pool_capabilities.memory_type,
                                 IREE_HAL_MEMORY_TYPE_HOST_LOCAL));
  iree_hal_buffer_native_binding_slot_t local_slot;
  IREE_ASSERT_OK(iree_hal_pool_resolve_binding(
      local_pool, scope_, IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
      &local_slot));
  const uint64_t unit = std::max(
      granularity_,
      iree_hal_pool_query_mapping_requirements(local_pool).minimum_granularity);
  // Two device-local allocations followed by system backing prove that stable
  // virtual addresses carry no physical-placement promise of their own.
  std::array<Ref<iree_hal_buffer_t>, 3> backings;
  const auto staging_address =
      iree_hal_buffer_native_binding(backing_, source_slot_).device_address;
  auto* staging = reinterpret_cast<uint32_t*>(host_mapping_.contents.data);
  const bool cache_control =
      native_family_.format_features & AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR;
  for (uint32_t i = 0; i < backings.size(); ++i) {
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        i < 2 ? local_pool.get() : pool_,
        iree_hal_make_pool_reservation_request(unit, unit),
        iree_infinite_timeout(), backings[i].out()));
    for (uint32_t j = 0; j < 128; ++j) {
      staging[j] = 0x1000u + i * 0x1000u + j;
    }
    IREE_ASSERT_OK(iree_hal_buffer_flush_host_range(backing_, 0, 512));
    SdmaCommandWriter commands(
        reinterpret_cast<uint32_t*>(queue_host_.ring_address) + ring_words_,
        native_family_.format_features);
    if (cache_control) {
      commands.AcquireFromSystem();
    }
    commands.CopyLinear(staging_address,
                        iree_hal_buffer_native_binding(
                            backings[i], i < 2 ? local_slot : source_slot_)
                            .device_address,
                        512);
    if (cache_control) {
      commands.ReleaseToSystem();
    }
    ASSERT_NO_FATAL_FAILURE(SubmitAndWait(commands));
  }

  family.interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS;
  const iree_hal_virtual_buffer_options_t options = {
      .byte_length = unit,
      .mapping_granularity = 2 * unit,
  };
  Ref<iree_hal_buffer_t> root;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_create(
      group_, {1, &family, {}}, &options, allocator_, root.out()));
  Ref<iree_hal_buffer_t> view;
  IREE_ASSERT_OK(
      iree_hal_buffer_subspan(root, 256, 256, allocator_, view.out()));
  const iree_hal_buffer_native_binding_slot_t virtual_slot = {
      .index = root.get()
                   ->memory.contract->scopes[scope_.id]
                   .bindings[IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS],
      .type = IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
  };
  const uint64_t address =
      iree_hal_buffer_native_binding(view, virtual_slot).device_address;
  for (uint32_t epoch = 0; epoch < backings.size(); ++epoch) {
    const std::array<iree_hal_memory_mapping_t, 1> mappings = {{
        {
            .target = root,
            .scope = scope_,
            .target_range = {0, unit},
            .source = backings[epoch],
            .access = IREE_HAL_MEMORY_ACCESS_ALL,
            .operation = epoch ? IREE_HAL_MEMORY_MAPPING_OPERATION_REPLACE
                               : IREE_HAL_MEMORY_MAPPING_OPERATION_BIND,
        },
    }};
    IREE_ASSERT_OK(Map(mappings));
    EXPECT_EQ(iree_hal_buffer_native_binding(view, virtual_slot).device_address,
              address);
    SdmaCommandWriter commands(
        reinterpret_cast<uint32_t*>(queue_host_.ring_address) + ring_words_,
        native_family_.format_features);
    if (cache_control) {
      commands.AcquireFromSystem();
    }
    commands.CopyLinear(address, staging_address + 2 * granularity_, 256);
    if (cache_control) {
      commands.ReleaseToSystem();
    }
    ASSERT_NO_FATAL_FAILURE(SubmitAndWait(commands));
    IREE_ASSERT_OK(
        iree_hal_buffer_invalidate_host_range(backing_, 2 * granularity_, 256));
    for (uint32_t j = 0; j < 64; ++j) {
      EXPECT_EQ(staging[2 * granularity_ / sizeof(uint32_t) + j],
                0x1000u + epoch * 0x1000u + 64 + j);
    }
  }
  IREE_ASSERT_OK(iree_hal_virtual_buffer_detach(root, scope_));
}

TEST_P(FabricVirtualBufferTest, CpuQueueUsesPrivatelyOwnedLocalStorage) {
  using iree::hal::cts::Ref;
  if (storage_.size() == 1) {
    GTEST_SKIP() << "native device has no independently placed local storage";
  }
  uint32_t physical_node = IREE_HAL_TOPOLOGY_NODE_ORDINAL_INVALID;
  const auto* topology = iree_hal_device_group_topology(group_);
  for (iree_host_size_t i = 0; i < iree_hal_topology_node_count(topology);
       ++i) {
    const auto* node = iree_hal_topology_node_at(topology, i);
    if (node->kind == IREE_HAL_TOPOLOGY_NODE_KIND_PHYSICAL_DEVICE &&
        node->device_ordinal ==
            iree_hal_device_topology_info(device_)->topology_index) {
      physical_node = node->ordinal;
      break;
    }
  }
  ASSERT_NE(physical_node, IREE_HAL_TOPOLOGY_NODE_ORDINAL_INVALID);
  const amdf_memory_device_access_t native_owner = {
      .device = native_device_,
      .requirements =
          {
              .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE |
                        storage_[1].minimum_access,
          },
  };
  amdf_memory_profile_t profile = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
      .structure_size = sizeof(profile),
  };
  amdf_memory_access_capabilities_t access_capabilities = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
      .structure_size = sizeof(access_capabilities),
  };
  ASSERT_EQ(api_->memory_scope_query_device_profile(storage_[1].scope, 0, 1,
                                                    &native_owner, &profile,
                                                    &access_capabilities),
            AMDF_STATUS_OK);
  const bool host_visible =
      (profile.supported_flags & AMDF_MEMORY_FLAG_HOST_VISIBLE) != 0;
  RecordProperty("host_visible_local_memory", host_visible ? "true" : "false");
  const amdf_memory_profile_pair_query_t pair_query = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE_PAIR_QUERY,
      .structure_size = sizeof(pair_query),
      .memory_profile_ordinal = profile.ordinal,
      .access_count = 1,
      .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
      .accesses = &native_owner,
      .producer =
          {
              .kind = AMDF_MEMORY_SITE_KIND_HOST,
              .value = {.host_access = AMDF_MEMORY_MAP_FLAG_WRITE},
          },
      .consumer =
          {
              .kind = AMDF_MEMORY_SITE_KIND_HOST,
              .value = {.host_access = AMDF_MEMORY_MAP_FLAG_READ},
          },
  };
  amdf_memory_pair_info_t pair = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
      .structure_size = sizeof(pair),
  };
  const amdf_status_t pair_status =
      api_->memory_scope_query_pair_info(storage_[1].scope, &pair_query, &pair);
  ASSERT_TRUE(pair_status == AMDF_STATUS_OK ||
              pair_status ==
                  amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED));
  RecordProperty("host_transition_qualified",
                 pair_status == AMDF_STATUS_OK ? "true" : "false");

  const iree_hal_pool_family_access_t family = {
      .family = iree_hal_device_queue_family(cpu_, 0),
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
  };
  iree_hal_slab_pool_options_t options;
  iree_hal_slab_pool_options_initialize(&options);
  options.placement = {IREE_HAL_POOL_PLACEMENT_REQUIRED, physical_node};
  Ref<iree_hal_pool_t> pool;
  if (!host_visible || pair_status != AMDF_STATUS_OK) {
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_UNAVAILABLE,
        iree_hal_slab_pool_create(group_, {1, &family, {}}, &options,
                                  allocator_, pool.out()));
    return;
  }
  IREE_ASSERT_OK(iree_hal_slab_pool_create(group_, {1, &family, {}}, &options,
                                           allocator_, pool.out()));
  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(pool, &capabilities);
  EXPECT_EQ(capabilities.placement.mode, IREE_HAL_POOL_PLACEMENT_REQUIRED);
  EXPECT_EQ(capabilities.placement.node, physical_node);
  EXPECT_EQ(iree_hal_pool_query_host_access(pool).access,
            IREE_HAL_MEMORY_ACCESS_NONE);
  iree_hal_buffer_native_binding_slot_t slot;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_pool_resolve_binding(
          pool, scope_, IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS, &slot));
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(granularity_, granularity_),
      iree_infinite_timeout(), buffer.out()));
  EXPECT_EQ(iree_hal_buffer_allocation_placement(buffer).device, device_);
  ASSERT_NO_FATAL_FAILURE(CpuFillAndReadback(buffer));
}

TEST_P(FabricVirtualBufferTest, CpuOnlyExportPoolUsesPrivateNativeOwner) {
  using iree::hal::cts::Ref;
  const amdf_memory_device_access_t native_owner = {
      .device = native_device_,
      .requirements =
          {
              .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE |
                        storage_[0].minimum_access,
          },
  };
  amdf_memory_profile_t profile = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
      .structure_size = sizeof(profile),
  };
  amdf_memory_access_capabilities_t access_capabilities = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
      .structure_size = sizeof(access_capabilities),
  };
  ASSERT_EQ(api_->memory_scope_query_device_profile(fabric_->system_scope, 0, 1,
                                                    &native_owner, &profile,
                                                    &access_capabilities),
            AMDF_STATUS_OK);
  iree_hal_external_storage_class_t storage_class = {};
  for (uint32_t i = 0; i < profile.external_memory_support_count; ++i) {
    const auto& support = profile.external_memory_support[i];
    if (!(support.flags & AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_EXPORT)) {
      continue;
    }
    if (support.type == AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD) {
      storage_class.type = IREE_HAL_EXTERNAL_STORAGE_TYPE_DMA_BUF_FD;
      break;
    } else if (support.type == AMDF_EXTERNAL_MEMORY_TYPE_D3D12_RESOURCE) {
      storage_class.type = IREE_HAL_EXTERNAL_STORAGE_TYPE_D3D12_RESOURCE;
      break;
    }
  }
  if (storage_class.type == IREE_HAL_EXTERNAL_STORAGE_TYPE_NONE) {
    GTEST_SKIP() << "native system storage has no device-specific export route";
  }
  const iree_hal_pool_family_access_t family = {
      .family = iree_hal_device_queue_family(cpu_, 0),
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
  };
  iree_hal_slab_pool_options_t options;
  iree_hal_slab_pool_options_initialize(&options);
  options.owned.exports = {1, &storage_class};
  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(iree_hal_slab_pool_create(group_, {1, &family, {}}, &options,
                                           allocator_, pool.out()));
  EXPECT_EQ(iree_hal_pool_query_host_access(pool).access,
            IREE_HAL_MEMORY_ACCESS_NONE);
  iree_hal_buffer_native_binding_slot_t slot;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_pool_resolve_binding(
          pool, scope_, IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS, &slot));
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(granularity_, granularity_),
      iree_infinite_timeout(), buffer.out()));
  EXPECT_EQ(iree_hal_buffer_allocation_placement(buffer).device, device_);
  ASSERT_NO_FATAL_FAILURE(CpuFillAndReadback(buffer));
  iree_hal_external_storage_t exported = {};
  iree_hal_buffer_range_t range = {};
  IREE_ASSERT_OK(iree_hal_buffer_export(buffer, {0, granularity_},
                                        storage_class, &exported, &range));
  if (storage_class.type == IREE_HAL_EXTERNAL_STORAGE_TYPE_DMA_BUF_FD) {
    EXPECT_GE(exported.payload.file_descriptor, 0);
  } else {
    EXPECT_NE(exported.payload.win32_handle, nullptr);
  }
  EXPECT_EQ(range.byte_length, granularity_);
  iree_hal_external_storage_release(&exported);
}

INSTANTIATE_TEST_SUITE_P(
    NativeLifetime, FabricVirtualBufferTest,
    ::testing::Values(AMDF_NATIVE_LIFETIME_PROCESS,
                      AMDF_NATIVE_LIFETIME_INSTANCE),
    [](const ::testing::TestParamInfo<amdf_native_lifetime_t>& info) {
      return info.param == AMDF_NATIVE_LIFETIME_PROCESS ? "Process"
                                                        : "Instance";
    });

}  // namespace iree::hal::amd::fabric::cts
