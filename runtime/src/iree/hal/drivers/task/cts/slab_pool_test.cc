// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>

#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/memory/fixed_block_pool.h"
#include "iree/hal/memory/maintenance.h"
#include "iree/hal/memory/slab_cache.h"
#include "iree/hal/memory/tlsf_pool.h"

namespace iree::hal::cts {
namespace {

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
    iree_hal_pool_scope_t scope = {.family_count = families_.size(),
                                   .families = families_.data()};
    return scope;
  }

  iree_status_t CreateSource(iree_hal_pool_t** out_pool) {
    iree_hal_slab_pool_options_t options;
    iree_hal_slab_pool_options_initialize(&options);
    return iree_hal_slab_pool_create(group_, Scope(), &options,
                                     iree_allocator_system(), out_pool);
  }

  void Wait(iree_hal_semaphore_list_t semaphores) {
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        semaphores, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  }

  // Joins queued child/cache returns, including work they enqueue. A pending
  // frontier callback can still enqueue native source retirement afterward.
  void JoinMaintenance(iree_hal_pool_t* pool) {
    iree_hal_memory_maintenance_call(
        pool->maintenance,
        [](void* user_data) {
          auto* owner = static_cast<iree_hal_memory_maintenance_t*>(user_data);
          while (iree_hal_memory_maintenance_run_one(owner)) {
          }
        },
        pool->maintenance);
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
      source, {}, sizeof(uint32_t), iree_infinite_timeout(), buffer.out()));
  SemaphoreList filled(devices_[0], {0}, {1});
  const uint32_t value = 0xAABBCCDD;
  IREE_ASSERT_OK(
      iree_hal_queue_fill(queues_[0], iree_hal_semaphore_list_empty(), filled,
                          buffer, 0, sizeof(value), &value, sizeof(value),
                          /*barriers=*/NULL, IREE_HAL_FILL_FLAG_NONE));
  Wait(filled);
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
  JoinMaintenance(source);
}

TEST_P(TaskSlabPoolTest, NativeOnlyScopeDoesNotInventSemanticPermissions) {
  families_[0].usage = IREE_HAL_BUFFER_USAGE_NONE;
  families_[1].usage = IREE_HAL_BUFFER_USAGE_NONE;
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(CreateSource(source.out()));
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      source, {}, sizeof(uint32_t), iree_infinite_timeout(), buffer.out()));
  EXPECT_EQ(iree_hal_buffer_allowed_usage(buffer), IREE_HAL_BUFFER_USAGE_NONE);
  uint32_t value = 42;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_queue_update(queues_[0], iree_hal_semaphore_list_empty(),
                            iree_hal_semaphore_list_empty(), &value, 0, buffer,
                            0, sizeof(value), /*barriers=*/NULL,
                            IREE_HAL_UPDATE_FLAG_NONE));
  buffer.reset();
  JoinMaintenance(source);
}

TEST_P(TaskSlabPoolTest, InteriorArenasInheritTheCompleteScope) {
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(CreateSource(source.out()));
  Ref<iree_hal_buffer_t> backing;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      source, {}, 8192, iree_infinite_timeout(), backing.out()));
  Ref<iree_hal_pool_t> blocks;
  iree_hal_fixed_block_pool_options_t block_options = {.block_size = 1024,
                                                       .frontier_capacity = 4};
  IREE_ASSERT_OK(iree_hal_fixed_block_pool_create_from_buffer(
      backing, 128, 4096, &block_options, iree_allocator_system(),
      blocks.out()));
  Ref<iree_hal_buffer_t> region;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      blocks, {}, 1024, iree_infinite_timeout(), region.out()));
  Ref<iree_hal_pool_t> arena;
  iree_hal_tlsf_pool_options_t options = {};
  options.tlsf_options.alignment = IREE_HAL_MEMORY_TLSF_MIN_ALIGNMENT;
  options.tlsf_options.frontier_capacity = 4;
  IREE_ASSERT_OK(iree_hal_tlsf_pool_create_from_buffer(
      region, 64, 768, &options, iree_allocator_system(), arena.out()));
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      arena, {}, sizeof(uint32_t), iree_infinite_timeout(), buffer.out()));
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
  Wait(downloaded);
  EXPECT_EQ(result, value);
  EXPECT_EQ(iree_atomic_ref_count_load(&buffer.get()->resource.ref_count), 1);
  buffer.reset();
  arena.reset();
  region.reset();
  blocks.reset();
  backing.reset();
  JoinMaintenance(source);
}

#if IREE_HAL_COMMAND_BUFFER_VALIDATION_ENABLE
TEST_P(TaskSlabPoolTest, RecordedBindingsValidateTheAdmittedFamily) {
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(CreateSource(source.out()));
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      source, {}, 16, iree_infinite_timeout(), buffer.out()));
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
  JoinMaintenance(source);
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
  cache_options.slab.params.min_alignment =
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
      .block_size = 256, .blocks_per_slab = 16, .frontier_capacity = 4};
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
      Wait(downloaded);
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
      Wait(deallocated);
      for (auto& view : views) {
        view.reset();
      }
      for (auto& root : roots) {
        root.reset();
      }
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
  EXPECT_EQ(cache_stats.hit_count, 9u);
  EXPECT_EQ(cache_stats.ready_count, 1u);
}

CTS_REGISTER_EXECUTABLE_TEST_SUITE(TaskSlabPoolTest);

}  // namespace
}  // namespace iree::hal::cts
