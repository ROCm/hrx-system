// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory_scope.h"

#include "iree/hal/command_buffer.h"
#include "iree/hal/device.h"
#include "iree/hal/pool.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(HostMemoryPropertiesTest, ValidatesIndependentRequests) {
  const uint32_t valid_requests[] = {
      IREE_HAL_HOST_MEMORY_PROPERTY_NONE,
      IREE_HAL_HOST_MEMORY_PROPERTY_READ_CACHED,
      IREE_HAL_HOST_MEMORY_PROPERTY_READ_UNCACHED,
      IREE_HAL_HOST_MEMORY_PROPERTY_WRITE_BACK,
      IREE_HAL_HOST_MEMORY_PROPERTY_WRITE_COMBINED,
      IREE_HAL_HOST_MEMORY_PROPERTY_READ_CACHED |
          IREE_HAL_HOST_MEMORY_PROPERTY_WRITE_BACK,
      IREE_HAL_HOST_MEMORY_PROPERTY_READ_UNCACHED |
          IREE_HAL_HOST_MEMORY_PROPERTY_WRITE_COMBINED,
  };
  // Include every combination of defined properties and an undefined bit.
  for (uint32_t properties = 0; properties < 32; ++properties) {
    bool expected = false;
    for (uint32_t valid : valid_requests) {
      expected |= properties == valid;
    }
    const iree_hal_pool_host_access_t host = {
        .access = IREE_HAL_MEMORY_ACCESS_ALL,
        .modes = IREE_HAL_MAPPING_MODE_SCOPED,
        .properties = properties,
    };
    SCOPED_TRACE(properties);
    iree_status_t status = iree_hal_pool_host_access_validate(host);
    EXPECT_EQ(iree_status_is_ok(status), expected);
    iree_status_ignore(status);
  }
}

class MemoryScopeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const iree_hal_buffer_binding_layout_t layout = {};
    IREE_ASSERT_OK(iree_hal_memory_contract_create(
        this, 8, &layout, iree_allocator_system(), &contract_));
    contract_->host.access = IREE_HAL_MEMORY_ACCESS_ALL;
    // Sites 2/3 and 4/5 represent two families. Sites 6/7 are excluded.
    for (uint32_t id = 2; id < 6; ++id) {
      contract_->scopes[id].interfaces = 1u << IREE_HAL_BUFFER_INTERFACE_HOST;
      contract_->scopes[id].usage = IREE_HAL_BUFFER_USAGE_STORAGE;
    }
  }

  void TearDown() override { iree_hal_memory_contract_release(contract_); }

  iree_hal_memory_scope_t scope(uint32_t id) { return {this, id}; }

  iree_hal_memory_transition_table_t table() { return {contract_}; }

  iree_hal_memory_transition_t Query(
      uint32_t producer, uint32_t consumer,
      iree_hal_memory_transition_action_t action) {
    iree_hal_memory_transition_pair_t pair;
    IREE_CHECK_OK(iree_hal_memory_transition_prepare_pair(
        table(), scope(producer), scope(consumer), action, &pair));
    return iree_hal_memory_transition_query(table(), pair);
  }

  static iree_status_t CoherentPair(void* user_data, uint32_t producer,
                                    uint32_t consumer,
                                    iree_hal_memory_pair_info_t* out_info) {
    out_info->flags = IREE_HAL_MEMORY_PAIR_SHARED_BACKING_REACHABLE |
                      IREE_HAL_MEMORY_PAIR_FIXED_COST_KNOWN;
    out_info->release.kind = IREE_HAL_MEMORY_TRANSITION_KIND_NONE;
    out_info->acquire.kind = IREE_HAL_MEMORY_TRANSITION_KIND_NONE;
    out_info->atomic_reach.scope_32 = IREE_HAL_ATOMIC_REACH_SYSTEM;
    out_info->atomic_reach.scope_64 = IREE_HAL_ATOMIC_REACH_SYSTEM;
    return iree_ok_status();
  }

  // Owns the actual contract allocation, including its matrix and cold facts.
  iree_hal_memory_contract_t* contract_ = nullptr;
};

TEST_F(MemoryScopeTest, CapturesIndependentAtomicSitesAndWildcardIntersection) {
  const iree_hal_atomic_operation_capabilities_t host = {
      .device_scope_32 = IREE_HAL_ATOMIC_OPERATION_FLAG_STORE,
      .device_scope_64 = IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_ADD,
      .system_scope_32 = IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_XOR,
      .system_scope_64 = IREE_HAL_ATOMIC_OPERATION_FLAG_STORE,
  };
  for (uint32_t id = 1; id < 6; ++id) {
    iree_hal_memory_scope_access_set_atomic_operations(&contract_->scopes[id],
                                                       host);
  }
  const iree_hal_atomic_operation_capabilities_t program = {
      .system_scope_64 = IREE_HAL_ATOMIC_OPERATION_FLAG_STORE,
  };
  iree_hal_memory_scope_access_set_atomic_operations(&contract_->scopes[3],
                                                     program);
  iree_hal_memory_contract_finalize_atomic_operations(contract_);
  iree_hal_pool_t pool = {.memory_contract = contract_};
  iree_hal_buffer_t buffer = {};
  buffer.memory.contract = contract_;
  EXPECT_EQ(reinterpret_cast<uintptr_t>(contract_->scopes) % 32, 0u);
  EXPECT_EQ(
      iree_hal_pool_query_atomic_operations(&pool, scope(1)).device_scope_32,
      host.device_scope_32);
  EXPECT_EQ(iree_hal_buffer_query_atomic_operations(&buffer, scope(2))
                .device_scope_64,
            host.device_scope_64);
  EXPECT_EQ(iree_hal_buffer_query_atomic_operations(&buffer, scope(2))
                .system_scope_32,
            host.system_scope_32);
  EXPECT_EQ(
      iree_hal_pool_query_atomic_operations(&pool, scope(3)).device_scope_32,
      0u);
  auto common = iree_hal_pool_query_atomic_operations(&pool, scope(0));
  EXPECT_EQ(common.device_scope_32, 0u);
  EXPECT_EQ(common.device_scope_64, 0u);
  EXPECT_EQ(common.system_scope_32, 0u);
  EXPECT_EQ(common.system_scope_64, program.system_scope_64);
  EXPECT_EQ(iree_hal_pool_query_host_access(&pool).atomics.system_scope_32,
            host.system_scope_32);

  // An ordinary NPU consumer contributes zero without erasing CPU/GPU facts.
  iree_hal_memory_scope_access_set_atomic_operations(&contract_->scopes[4], {});
  iree_hal_memory_contract_finalize_atomic_operations(contract_);
  EXPECT_TRUE(iree_hal_atomic_operation_capabilities_is_empty(
      iree_hal_pool_query_atomic_operations(&pool, scope(0))));
  EXPECT_TRUE(iree_hal_atomic_operation_capabilities_is_empty(
      iree_hal_buffer_query_atomic_operations(&buffer, scope(4))));
  EXPECT_EQ(iree_hal_buffer_query_atomic_operations(&buffer, scope(2))
                .system_scope_64,
            host.system_scope_64);
  for (auto invalid :
       {scope(6), scope(7), scope(8), scope(IREE_HAL_MEMORY_SCOPE_INVALID),
        iree_hal_memory_scope_t{&pool, 2}}) {
    EXPECT_TRUE(iree_hal_atomic_operation_capabilities_is_empty(
        iree_hal_buffer_query_atomic_operations(&buffer, invalid)));
  }
  buffer.memory.contract = nullptr;
  EXPECT_TRUE(iree_hal_atomic_operation_capabilities_is_empty(
      iree_hal_buffer_query_atomic_operations(&buffer, scope(2))));
}

TEST_F(MemoryScopeTest, PreservesEveryOperationInEveryWidthAndDomain) {
  iree_hal_buffer_t buffer = {};
  buffer.memory.contract = contract_;
  for (auto cell :
       {&iree_hal_atomic_operation_capabilities_t::device_scope_32,
        &iree_hal_atomic_operation_capabilities_t::device_scope_64,
        &iree_hal_atomic_operation_capabilities_t::system_scope_32,
        &iree_hal_atomic_operation_capabilities_t::system_scope_64}) {
    for (uint32_t operations = 0;
         operations <= IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL; ++operations) {
      iree_hal_atomic_operation_capabilities_t expected = {};
      expected.*cell = operations;
      iree_hal_memory_scope_access_set_atomic_operations(&contract_->scopes[2],
                                                         expected);
      iree_hal_memory_contract_set_zero_compute_atomic_operations(contract_, 2,
                                                                  expected);
      const auto actual =
          iree_hal_buffer_query_atomic_operations(&buffer, scope(2));
      EXPECT_EQ(actual.device_scope_32, expected.device_scope_32);
      EXPECT_EQ(actual.device_scope_64, expected.device_scope_64);
      EXPECT_EQ(actual.system_scope_32, expected.system_scope_32);
      EXPECT_EQ(actual.system_scope_64, expected.system_scope_64);
      const auto zero_compute =
          iree_hal_buffer_query_zero_compute_atomic_operations(&buffer,
                                                               scope(2));
      EXPECT_EQ(zero_compute.device_scope_32, expected.device_scope_32);
      EXPECT_EQ(zero_compute.device_scope_64, expected.device_scope_64);
      EXPECT_EQ(zero_compute.system_scope_32, expected.system_scope_32);
      EXPECT_EQ(zero_compute.system_scope_64, expected.system_scope_64);
    }
  }
}

TEST_F(MemoryScopeTest, ZeroComputeIsAnIndependentSubsetOfTheAdmittedSite) {
  iree_hal_pool_t pool = {.memory_contract = contract_};
  iree_hal_buffer_t buffer = {};
  buffer.memory.contract = contract_;
  const iree_hal_atomic_operation_capabilities_t queue = {
      .device_scope_32 = IREE_HAL_ATOMIC_OPERATION_FLAG_STORE,
      .system_scope_64 = IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT |
                         IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_ADD,
  };
  iree_hal_memory_scope_access_set_atomic_operations(&contract_->scopes[2],
                                                     queue);
  iree_hal_memory_contract_set_zero_compute_atomic_operations(
      contract_, 2, {.system_scope_64 = IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT});
  iree_hal_memory_contract_finalize_atomic_operations(contract_);
  const auto subset =
      iree_hal_pool_query_zero_compute_atomic_operations(&pool, scope(2));
  EXPECT_EQ(subset.device_scope_32, 0u);
  EXPECT_EQ(subset.device_scope_64, 0u);
  EXPECT_EQ(subset.system_scope_32, 0u);
  EXPECT_EQ(subset.system_scope_64, IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT);
  for (auto site : {scope(0), scope(1), scope(3), scope(4), scope(5), scope(6),
                    scope(7), scope(8), scope(IREE_HAL_MEMORY_SCOPE_INVALID),
                    iree_hal_memory_scope_t{&pool, 2}}) {
    EXPECT_TRUE(iree_hal_atomic_operation_capabilities_is_empty(
        iree_hal_buffer_query_zero_compute_atomic_operations(&buffer, site)));
  }

  // A construction-time permission restriction narrows both guarantees.
  iree_hal_memory_scope_access_set_atomic_operations(
      &contract_->scopes[2],
      {.system_scope_64 = IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_ADD});
  iree_hal_memory_contract_finalize_atomic_operations(contract_);
  EXPECT_TRUE(iree_hal_atomic_operation_capabilities_is_empty(
      iree_hal_buffer_query_zero_compute_atomic_operations(&buffer, scope(2))));
  buffer.memory.contract = nullptr;
  EXPECT_TRUE(iree_hal_atomic_operation_capabilities_is_empty(
      iree_hal_buffer_query_zero_compute_atomic_operations(&buffer, scope(2))));
  EXPECT_EQ(sizeof(iree_hal_memory_scope_access_t), 32u);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(contract_->scopes) % 32, 0u);
}

TEST_F(MemoryScopeTest, PairOperationIdentityIncludesBothSidesAndEveryCell) {
  const iree_hal_atomic_operation_capabilities_t all = {
      IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL, IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL,
      IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL, IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL};
  for (uint32_t id = 1; id < 6; ++id) {
    iree_hal_memory_scope_access_set_atomic_operations(&contract_->scopes[id],
                                                       all);
  }
  for (auto cell :
       {&iree_hal_atomic_operation_capabilities_t::device_scope_32,
        &iree_hal_atomic_operation_capabilities_t::device_scope_64,
        &iree_hal_atomic_operation_capabilities_t::system_scope_32,
        &iree_hal_atomic_operation_capabilities_t::system_scope_64}) {
    for (bool producer : {false, true}) {
      struct Variant {
        // The single cell that distinguishes this pair's cold record.
        decltype(cell) field;
        // Whether the producer or consumer owns that cell.
        bool producer;
      } variant = {cell, producer};
      IREE_ASSERT_OK(iree_hal_memory_contract_initialize_transitions(
          contract_,
          +[](void* user_data, uint32_t producer, uint32_t consumer,
              iree_hal_memory_pair_info_t* info) -> iree_status_t {
            IREE_RETURN_IF_ERROR(
                CoherentPair(nullptr, producer, consumer, info));
            constexpr uint32_t baseline = IREE_HAL_ATOMIC_OPERATION_FLAG_STORE |
                                          IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT;
            info->producer_atomics = {baseline, baseline, baseline, baseline};
            info->consumer_atomics = info->producer_atomics;
            if (producer == 2 && consumer == 3) {
              const auto* variant = static_cast<Variant*>(user_data);
              auto& side = variant->producer ? info->producer_atomics
                                             : info->consumer_atomics;
              side.*variant->field |= IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_ADD;
            }
            return iree_ok_status();
          },
          &variant));
      for (uint32_t consumer : {2u, 3u}) {
        iree_hal_memory_transition_pair_t pair;
        IREE_ASSERT_OK(iree_hal_memory_transition_prepare_pair(
            table(), scope(2), scope(consumer),
            IREE_HAL_MEMORY_TRANSITION_ACQUIRE, &pair));
        const auto info = iree_hal_memory_transition_query_info(table(), pair);
        const auto& selected =
            producer ? info.producer_atomics : info.consumer_atomics;
        const auto& other =
            producer ? info.consumer_atomics : info.producer_atomics;
        const uint32_t baseline = IREE_HAL_ATOMIC_OPERATION_FLAG_STORE |
                                  IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT;
        EXPECT_EQ(
            selected.*cell,
            baseline |
                (consumer == 3 ? IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_ADD : 0));
        EXPECT_EQ(other.*cell, baseline);
      }
    }
  }
  iree_hal_memory_transition_pair_t wildcard;
  IREE_ASSERT_OK(iree_hal_memory_transition_prepare_pair(
      table(), scope(0), scope(2), IREE_HAL_MEMORY_TRANSITION_ACQUIRE,
      &wildcard));
  const auto info = iree_hal_memory_transition_query_info(table(), wildcard);
  EXPECT_TRUE(
      iree_hal_atomic_operation_capabilities_is_empty(info.producer_atomics));
  EXPECT_TRUE(
      iree_hal_atomic_operation_capabilities_is_empty(info.consumer_atomics));
}

TEST_F(MemoryScopeTest, PairOperationsRequireSiteSupportAndAccess) {
  contract_->scopes[2].usage = IREE_HAL_BUFFER_USAGE_STORAGE_WRITE;
  contract_->scopes[3].usage = IREE_HAL_BUFFER_USAGE_STORAGE_READ;
  const iree_hal_atomic_operation_capabilities_t admitted = {
      .device_scope_32 = IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL,
      .system_scope_64 = IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL,
  };
  iree_hal_memory_scope_access_set_atomic_operations(&contract_->scopes[2],
                                                     admitted);
  iree_hal_memory_scope_access_set_atomic_operations(&contract_->scopes[3],
                                                     admitted);
  IREE_ASSERT_OK(iree_hal_memory_contract_initialize_transitions(
      contract_,
      +[](void*, uint32_t producer, uint32_t consumer,
          iree_hal_memory_pair_info_t* info) -> iree_status_t {
        IREE_RETURN_IF_ERROR(CoherentPair(nullptr, producer, consumer, info));
        info->producer_atomics = {IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL,
                                  IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL,
                                  IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL,
                                  IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL};
        info->consumer_atomics = info->producer_atomics;
        return iree_ok_status();
      },
      nullptr));
  iree_hal_memory_transition_pair_t pair;
  IREE_ASSERT_OK(iree_hal_memory_transition_prepare_pair(
      table(), scope(2), scope(3), IREE_HAL_MEMORY_TRANSITION_ACQUIRE, &pair));
  const auto info = iree_hal_memory_transition_query_info(table(), pair);
  EXPECT_EQ(info.producer_atomics.device_scope_32,
            IREE_HAL_ATOMIC_OPERATION_FLAG_STORE);
  EXPECT_EQ(info.producer_atomics.system_scope_64,
            IREE_HAL_ATOMIC_OPERATION_FLAG_STORE);
  EXPECT_EQ(info.consumer_atomics.device_scope_32,
            IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT);
  EXPECT_EQ(info.consumer_atomics.system_scope_64,
            IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT);
  EXPECT_EQ(info.producer_atomics.device_scope_64, 0u);
  EXPECT_EQ(info.producer_atomics.system_scope_32, 0u);
  EXPECT_EQ(info.consumer_atomics.device_scope_64, 0u);
  EXPECT_EQ(info.consumer_atomics.system_scope_32, 0u);

  // Site 4 has no atomic executor; reach cannot supply the missing guarantee
  // or leave the other side looking like a standalone atomic promise.
  IREE_ASSERT_OK(iree_hal_memory_transition_prepare_pair(
      table(), scope(2), scope(4), IREE_HAL_MEMORY_TRANSITION_ACQUIRE, &pair));
  const auto absent = iree_hal_memory_transition_query_info(table(), pair);
  EXPECT_EQ(absent.atomic_reach.scope_64, IREE_HAL_ATOMIC_REACH_SYSTEM);
  EXPECT_TRUE(
      iree_hal_atomic_operation_capabilities_is_empty(absent.producer_atomics));
  EXPECT_TRUE(
      iree_hal_atomic_operation_capabilities_is_empty(absent.consumer_atomics));
}

TEST(MemoryScopeHostAccessTest, RejectsAtomicRequirementsWithoutHostAccess) {
  iree_hal_pool_host_access_t host = {};
  host.atomics.system_scope_32 = IREE_HAL_ATOMIC_OPERATION_FLAG_STORE;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_pool_host_access_validate(host));
  host.access = IREE_HAL_MEMORY_ACCESS_WRITE;
  host.modes = IREE_HAL_MAPPING_MODE_SCOPED;
  IREE_EXPECT_OK(iree_hal_pool_host_access_validate(host));
  host.atomics.device_scope_64 = UINT32_C(1) << 31;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_pool_host_access_validate(host));
}

TEST(MemoryScopeAtomicTest, FiltersExecutionSitesIndependently) {
  iree_hal_queue_family_spec_t spec = {
      .role_flags = IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH,
  };
  spec.atomic_capabilities.operations.device_scope_32 =
      IREE_HAL_ATOMIC_OPERATION_FLAG_STORE;
  const iree_hal_queue_family_t family = {.spec = &spec};
  const iree_hal_atomic_operation_capabilities_t backing = {
      .device_scope_32 = IREE_HAL_ATOMIC_OPERATION_FLAG_STORE |
                         IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_ADD,
      .system_scope_64 = IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT,
  };
  auto queue = iree_hal_memory_site_filter_atomic_operations(
      {IREE_HAL_MEMORY_SITE_QUEUE, &family}, backing);
  EXPECT_EQ(queue.device_scope_32, IREE_HAL_ATOMIC_OPERATION_FLAG_STORE);
  EXPECT_EQ(queue.system_scope_64, 0u);
  auto program = iree_hal_memory_site_filter_atomic_operations(
      {IREE_HAL_MEMORY_SITE_PROGRAM, &family}, backing);
  EXPECT_EQ(program.device_scope_32, backing.device_scope_32);
  EXPECT_EQ(program.system_scope_64, backing.system_scope_64);
  spec.role_flags = IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER;
  EXPECT_TRUE(iree_hal_atomic_operation_capabilities_is_empty(
      iree_hal_memory_site_filter_atomic_operations(
          {IREE_HAL_MEMORY_SITE_PROGRAM, &family}, backing)));
}

TEST(MemoryTransitionRecipeTest, JoinsDistinctHostProtocolsThroughMappingApi) {
  const iree_hal_memory_transition_recipe_info_t first = {
      .kind = IREE_HAL_MEMORY_TRANSITION_KIND_RANGE,
      .executor = IREE_HAL_MEMORY_TRANSITION_EXECUTOR_HOST_DIRECT,
      .operation = IREE_HAL_MEMORY_TRANSITION_OPERATION_HOST_FLUSH,
      .range_granularity = 64,
      .host = {.instruction = IREE_HAL_HOST_CACHE_INSTRUCTION_X86_CLFLUSH},
  };
  auto combined = first;
  ASSERT_TRUE(iree_hal_memory_transition_recipe_join(&combined, &first));
  EXPECT_EQ(combined.executor, IREE_HAL_MEMORY_TRANSITION_EXECUTOR_HOST_DIRECT);
  EXPECT_EQ(combined.range_granularity, 64u);

  // Direct execution strides by the recipe's cache-line size. A different
  // size must use the mapping's own protocol instead of skipping cache lines.
  auto second = first;
  second.range_granularity = 128;
  ASSERT_TRUE(iree_hal_memory_transition_recipe_join(&combined, &second));
  EXPECT_EQ(combined.executor, IREE_HAL_MEMORY_TRANSITION_EXECUTOR_HOST_API);
  EXPECT_EQ(combined.host.instruction, IREE_HAL_HOST_CACHE_INSTRUCTION_NONE);
  EXPECT_EQ(combined.range_granularity, 128u);

  const iree_hal_memory_transition_recipe_info_t coherent = {
      .kind = IREE_HAL_MEMORY_TRANSITION_KIND_NONE,
  };
  for (bool coherent_first : {false, true}) {
    combined = coherent_first ? coherent : first;
    const auto value = coherent_first ? first : coherent;
    ASSERT_TRUE(iree_hal_memory_transition_recipe_join(&combined, &value));
    EXPECT_EQ(combined.kind, IREE_HAL_MEMORY_TRANSITION_KIND_RANGE);
    EXPECT_EQ(combined.executor, IREE_HAL_MEMORY_TRANSITION_EXECUTOR_HOST_API);
    EXPECT_EQ(combined.operation,
              IREE_HAL_MEMORY_TRANSITION_OPERATION_HOST_FLUSH);
  }
}

TEST(MemoryTransitionRecipeTest, ValidatesEffectsAndQueueOperationShape) {
  iree_hal_memory_transition_recipe_info_t operation = {
      .kind = IREE_HAL_MEMORY_TRANSITION_KIND_RANGE,
      .executor = IREE_HAL_MEMORY_TRANSITION_EXECUTOR_QUEUE,
      .operation = IREE_HAL_MEMORY_TRANSITION_OPERATION_RELEASE_TO_SYSTEM,
      .range_granularity = 64,
  };
  iree_hal_memory_transition_recipe_t recipe = {
      .effects = {IREE_HAL_MEMORY_EFFECT_RANGE_RELEASE_TO_SYSTEM},
      .operation_count = 1,
      .operations = &operation,
  };
  IREE_EXPECT_OK(iree_hal_memory_transition_recipe_validate(&recipe));

  recipe.effects.bits = IREE_HAL_MEMORY_EFFECT_RANGE_ACQUIRE_FROM_SYSTEM;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_memory_transition_recipe_validate(&recipe));
  recipe.effects.bits = IREE_HAL_MEMORY_EFFECT_RANGE_RELEASE_TO_SYSTEM;
  operation.kind = IREE_HAL_MEMORY_TRANSITION_KIND_GLOBAL;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_memory_transition_recipe_validate(&recipe));
  operation.kind = IREE_HAL_MEMORY_TRANSITION_KIND_RANGE;
  operation.host.instruction = IREE_HAL_HOST_CACHE_INSTRUCTION_X86_CLFLUSH;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_memory_transition_recipe_validate(&recipe));
  operation.executor = IREE_HAL_MEMORY_TRANSITION_EXECUTOR_HOST_DIRECT;
  operation.operation = IREE_HAL_MEMORY_TRANSITION_OPERATION_HOST_FLUSH;
  recipe.effects.bits = IREE_HAL_MEMORY_EFFECT_HOST_FLUSH;
  IREE_EXPECT_OK(iree_hal_memory_transition_recipe_validate(&recipe));
  operation.host.instruction = (iree_hal_host_cache_instruction_t)0xFFFFFFFFu;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_memory_transition_recipe_validate(&recipe));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_memory_transition_recipe_validate(nullptr));
}

TEST(MemoryTransitionRecipeTest, QueueFamilyAdmitsItsHostExecutors) {
  iree_hal_queue_family_spec_t family_spec = {};
  iree_hal_queue_family_t family = {
      .spec = &family_spec,
  };
  iree_hal_memory_transition_recipe_info_t operation = {
      .kind = IREE_HAL_MEMORY_TRANSITION_KIND_RANGE,
      .executor = IREE_HAL_MEMORY_TRANSITION_EXECUTOR_HOST_API,
      .operation = IREE_HAL_MEMORY_TRANSITION_OPERATION_HOST_FLUSH,
      .range_granularity = 64,
  };
  iree_hal_memory_transition_recipe_t recipe = {
      .effects = {IREE_HAL_MEMORY_EFFECT_HOST_FLUSH},
      .operation_count = 1,
      .operations = &operation,
  };
  iree_hal_buffer_barrier_t range = {
      .recipe = &recipe,
  };
  iree_hal_barrier_t barrier = {
      .effects = recipe.effects,
      .buffer_barrier_count = 1,
      .buffer_barriers = &range,
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        iree_hal_barrier_validate(&barrier, &family));
  family_spec.flags = IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_HOST_MEMORY_BARRIERS;
  IREE_EXPECT_OK(iree_hal_barrier_validate(&barrier, &family));
  operation.executor = IREE_HAL_MEMORY_TRANSITION_EXECUTOR_HOST_DIRECT;
  operation.host.instruction = IREE_HAL_HOST_CACHE_INSTRUCTION_X86_CLFLUSH;
  // Structural admission is independent of the host running this test.
  // Native qualification belongs to the backend that publishes the recipe.
  IREE_EXPECT_OK(iree_hal_barrier_validate(&barrier, &family));
  // Host effects still require an actual range operand and matching recipe.
  barrier.buffer_barrier_count = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_barrier_validate(&barrier, &family));
}

TEST(MemoryTransitionRecipeTest, GlobalHostActionsRetainResourceOperands) {
  iree_hal_queue_family_spec_t family_spec = {
      .flags = IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_HOST_MEMORY_BARRIERS,
  };
  const iree_hal_queue_family_t family = {.spec = &family_spec};
  for (auto executor : {IREE_HAL_MEMORY_TRANSITION_EXECUTOR_HOST_API,
                        IREE_HAL_MEMORY_TRANSITION_EXECUTOR_HOST_DIRECT}) {
    for (auto operation :
         {IREE_HAL_MEMORY_TRANSITION_OPERATION_HOST_FLUSH,
          IREE_HAL_MEMORY_TRANSITION_OPERATION_HOST_INVALIDATE}) {
      iree_hal_memory_transition_recipe_info_t info = {
          .kind = IREE_HAL_MEMORY_TRANSITION_KIND_GLOBAL,
          .executor = executor,
          .operation = operation,
      };
      const iree_hal_memory_transition_recipe_t recipe = {
          .effects = {static_cast<uint32_t>(
              operation == IREE_HAL_MEMORY_TRANSITION_OPERATION_HOST_FLUSH
                  ? IREE_HAL_MEMORY_EFFECT_HOST_FLUSH
                  : IREE_HAL_MEMORY_EFFECT_HOST_INVALIDATE)},
          .operation_count = 1,
          .operations = &info,
      };
      const iree_hal_buffer_barrier_t buffer_barrier = {.recipe = &recipe};
      iree_hal_barrier_t barrier = {
          .effects = recipe.effects,
          .buffer_barrier_count = 1,
          .buffer_barriers = &buffer_barrier,
      };
      IREE_EXPECT_OK(iree_hal_memory_transition_recipe_validate(&recipe));
      IREE_EXPECT_OK(iree_hal_barrier_validate(&barrier, &family));
      barrier.buffer_barrier_count = 0;
      IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                            iree_hal_barrier_validate(&barrier, &family));
      info.range_granularity = 64;
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_INVALID_ARGUMENT,
          iree_hal_memory_transition_recipe_validate(&recipe));
      info.range_granularity = 0;
      info.host.instruction = IREE_HAL_HOST_CACHE_INSTRUCTION_X86_CLFLUSH;
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_INVALID_ARGUMENT,
          iree_hal_memory_transition_recipe_validate(&recipe));
    }
  }
}

TEST(MemoryScopeFamilyUsageTest, UnscopedBufferHonorsAllocationPlacement) {
  uint8_t device_a_token = 0;
  uint8_t device_b_token = 0;
  iree_hal_queue_family_spec_t family_spec = {};
  iree_hal_queue_family_t family_a0;
  iree_hal_queue_family_t family_a1;
  iree_hal_queue_family_t family_b1;
  iree_hal_queue_family_initialize(
      reinterpret_cast<iree_hal_device_t*>(&device_a_token), 0, &family_spec,
      &family_a0);
  iree_hal_queue_family_initialize(
      reinterpret_cast<iree_hal_device_t*>(&device_a_token), 1, &family_spec,
      &family_a1);
  iree_hal_queue_family_initialize(
      reinterpret_cast<iree_hal_device_t*>(&device_b_token), 1, &family_spec,
      &family_b1);

  iree_hal_buffer_t buffer = {};
  buffer.allocated_buffer = &buffer;
  buffer.allowed_usage = IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE;
  buffer.placement.device =
      reinterpret_cast<iree_hal_device_t*>(&device_a_token);
  buffer.placement.queue_family_affinity =
      iree_hal_make_queue_family_affinity(1);

  IREE_EXPECT_OK(iree_hal_buffer_validate_family_usage(
      &buffer, &family_a1, IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_buffer_validate_family_usage(
          &buffer, &family_a0, IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_buffer_validate_family_usage(
          &buffer, &family_b1, IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE));
  EXPECT_EQ(iree_hal_buffer_family_usage(&buffer, &family_a0),
            IREE_HAL_BUFFER_USAGE_NONE);

  buffer.placement.queue_family_affinity = IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY;
  IREE_EXPECT_OK(iree_hal_buffer_validate_family_usage(
      &buffer, &family_a0, IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE));

  buffer.placement = iree_hal_buffer_placement_undefined();
  IREE_EXPECT_OK(iree_hal_buffer_validate_family_usage(
      &buffer, &family_b1, IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE));
}

TEST_F(MemoryScopeTest, UnknownIsNotCoherent) {
  auto transition = Query(2, 4, IREE_HAL_MEMORY_TRANSITION_RELEASE);
  EXPECT_FALSE(iree_hal_memory_effects_is_supported(transition.release));
  EXPECT_FALSE(iree_hal_memory_effects_is_supported(transition.acquire));
  const uint32_t unsupported = IREE_HAL_MEMORY_EFFECT_UNSUPPORTED;
  EXPECT_EQ(transition.release.bits, unsupported);
  EXPECT_EQ(transition.acquire.bits, unsupported);
  EXPECT_EQ(sizeof(iree_hal_memory_transition_t), 8u);
  EXPECT_EQ(sizeof(iree_hal_memory_transition_pair_t), 4u);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(contract_->transitions) % 8, 0u);
}

TEST_F(MemoryScopeTest, CapturesMappingAliasesWithoutChangingSourceViews) {
  IREE_ASSERT_OK(iree_hal_memory_contract_initialize_transitions(
      contract_, CoherentPair, nullptr));
  for (uint32_t id : {1u, 2u, 3u}) {
    contract_->scopes[id].interfaces |=
        1u << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE;
    iree_hal_memory_scope_access_set_atomic_operations(
        &contract_->scopes[id],
        {.system_scope_32 = IREE_HAL_ATOMIC_OPERATION_FLAG_STORE});
  }
  contract_->host.properties = IREE_HAL_HOST_MEMORY_PROPERTY_WRITE_COMBINED;
  iree_hal_memory_contract_set_zero_compute_atomic_operations(
      contract_, 2, {.system_scope_32 = IREE_HAL_ATOMIC_OPERATION_FLAG_STORE});
  IREE_ASSERT_OK(iree_hal_memory_contract_initialize_mapping_transitions(
      contract_, IREE_HAL_HOST_MEMORY_PROPERTY_WRITE_BACK,
      +[](void* user_data, uint32_t producer, uint32_t consumer,
          iree_hal_memory_pair_info_t* info) {
        IREE_RETURN_IF_ERROR(CoherentPair(user_data, producer, consumer, info));
        if (producer == 1) {
          info->release = {
              .kind = IREE_HAL_MEMORY_TRANSITION_KIND_RANGE,
              .executor = IREE_HAL_MEMORY_TRANSITION_EXECUTOR_HOST_API,
              .operation = IREE_HAL_MEMORY_TRANSITION_OPERATION_HOST_FLUSH,
              .range_granularity = 64,
          };
        }
        return iree_ok_status();
      },
      nullptr));
  const auto* alias = iree_hal_memory_contract_mapping_alias(contract_);
  ASSERT_NE(alias, contract_);
  EXPECT_EQ(alias->host.properties, IREE_HAL_HOST_MEMORY_PROPERTY_WRITE_BACK);
  EXPECT_EQ(contract_->host.properties,
            IREE_HAL_HOST_MEMORY_PROPERTY_WRITE_COMBINED);
  EXPECT_EQ(alias->scopes[2].atomic_operations,
            contract_->scopes[2].atomic_operations);
  EXPECT_EQ(alias->zero_compute_atomic_operations[2],
            contract_->zero_compute_atomic_operations[2]);
  EXPECT_EQ(alias->zero_compute_atomic_operations[4], 0u);
  EXPECT_EQ(alias->scopes[4].interfaces, 0u);
  EXPECT_EQ(alias->scopes[5].interfaces, 0u);
  iree_hal_memory_transition_pair_t pair;
  IREE_ASSERT_OK(iree_hal_memory_transition_prepare_pair(
      {alias}, scope(1), scope(2), IREE_HAL_MEMORY_TRANSITION_RELEASE, &pair));
  EXPECT_EQ(iree_hal_memory_transition_query({alias}, pair).release.bits,
            IREE_HAL_MEMORY_EFFECT_HOST_FLUSH);
  EXPECT_EQ(Query(1, 2, IREE_HAL_MEMORY_TRANSITION_RELEASE).release.bits, 0u);
}

TEST_F(MemoryScopeTest, FailedAliasCapturePublishesNoPartialContract) {
  contract_->scopes[2].interfaces |=
      1u << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_memory_contract_initialize_mapping_transitions(
          contract_, IREE_HAL_HOST_MEMORY_PROPERTY_NONE,
          +[](void*, uint32_t, uint32_t, iree_hal_memory_pair_info_t*) {
            return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED);
          },
          nullptr));
  EXPECT_EQ(contract_->mapping_source.alias_contract, nullptr);
  EXPECT_EQ(iree_hal_memory_contract_mapping_alias(contract_), contract_);
}

TEST_F(MemoryScopeTest, PublicPreparationChecksDomainRoleAndLocalSide) {
  contract_->host.access = IREE_HAL_MEMORY_ACCESS_READ;
  contract_->scopes[2].usage = IREE_HAL_BUFFER_USAGE_STORAGE_READ;
  contract_->scopes[4].usage = IREE_HAL_BUFFER_USAGE_STORAGE_WRITE;
  iree_hal_memory_transition_pair_t pair;
  for (auto producer : {scope(1), scope(2), scope(6), scope(8),
                        iree_hal_memory_scope_t{&pair, 3}}) {
    IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED,
                          iree_hal_memory_transition_prepare_pair(
                              table(), producer, scope(3),
                              IREE_HAL_MEMORY_TRANSITION_RELEASE, &pair));
  }
  IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED,
                        iree_hal_memory_transition_prepare_pair(
                            table(), scope(3), scope(4),
                            IREE_HAL_MEMORY_TRANSITION_ACQUIRE, &pair));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_memory_transition_prepare_pair(
                            table(), scope(0), scope(3),
                            IREE_HAL_MEMORY_TRANSITION_RELEASE, &pair));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_memory_transition_prepare_pair(
                            table(), scope(3), scope(0),
                            IREE_HAL_MEMORY_TRANSITION_ACQUIRE, &pair));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_memory_transition_prepare_pair(
                            table(), scope(0), scope(0),
                            IREE_HAL_MEMORY_TRANSITION_ACQUIRE, &pair));
  IREE_ASSERT_OK(iree_hal_memory_transition_prepare_pair(
      table(), scope(4), scope(1), IREE_HAL_MEMORY_TRANSITION_RELEASE, &pair));
}

TEST_F(MemoryScopeTest, WildcardsJoinOnlyTheApplicableDirection) {
  // Unknown site 5 is read-only, so it must not poison ANY->2.acquire. It
  // remains a possible consumer and must poison 2->ANY.release.
  contract_->scopes[5].usage = IREE_HAL_BUFFER_USAGE_STORAGE_READ;
  IREE_ASSERT_OK(iree_hal_memory_contract_initialize_transitions(
      contract_,
      [](void* user_data, uint32_t producer, uint32_t consumer,
         iree_hal_memory_pair_info_t* out_info) -> iree_status_t {
        if (producer == 5 || consumer == 5) {
          return iree_ok_status();
        }
        IREE_RETURN_IF_ERROR(
            CoherentPair(nullptr, producer, consumer, out_info));
        if (producer == 4 && consumer == 2) {
          out_info->release.kind = IREE_HAL_MEMORY_TRANSITION_KIND_GLOBAL;
          out_info->release.executor =
              IREE_HAL_MEMORY_TRANSITION_EXECUTOR_QUEUE;
          out_info->release.operation =
              IREE_HAL_MEMORY_TRANSITION_OPERATION_RELEASE_TO_SYSTEM;
        }
        return iree_ok_status();
      },
      nullptr));
  auto incoming = Query(0, 2, IREE_HAL_MEMORY_TRANSITION_ACQUIRE);
  EXPECT_FALSE(iree_hal_memory_effects_is_supported(incoming.release));
  EXPECT_TRUE(iree_hal_memory_effects_is_empty(incoming.acquire));
  auto outgoing = Query(2, 0, IREE_HAL_MEMORY_TRANSITION_RELEASE);
  EXPECT_FALSE(iree_hal_memory_effects_is_supported(outgoing.release));
  EXPECT_FALSE(iree_hal_memory_effects_is_supported(outgoing.acquire));
  auto exact = Query(4, 2, IREE_HAL_MEMORY_TRANSITION_RELEASE);
  EXPECT_EQ(exact.release.bits,
            IREE_HAL_MEMORY_EFFECT_GLOBAL_RELEASE_TO_SYSTEM);
  EXPECT_TRUE(iree_hal_memory_effects_is_empty(exact.acquire));
}

TEST_F(MemoryScopeTest, UnknownProgramPoisonsWildcardButNotExactQueuePair) {
  IREE_ASSERT_OK(iree_hal_memory_contract_initialize_transitions(
      contract_,
      [](void* user_data, uint32_t producer, uint32_t consumer,
         iree_hal_memory_pair_info_t* out_info) -> iree_status_t {
        if (producer == 3 || producer == 5 || consumer == 3 || consumer == 5) {
          return iree_ok_status();
        }
        return CoherentPair(nullptr, producer, consumer, out_info);
      },
      nullptr));
  auto exact = Query(2, 4, IREE_HAL_MEMORY_TRANSITION_ACQUIRE);
  EXPECT_TRUE(iree_hal_memory_effects_is_empty(exact.release));
  EXPECT_TRUE(iree_hal_memory_effects_is_empty(exact.acquire));
  EXPECT_FALSE(iree_hal_memory_effects_is_supported(
      Query(0, 4, IREE_HAL_MEMORY_TRANSITION_ACQUIRE).acquire));
  EXPECT_FALSE(iree_hal_memory_effects_is_supported(
      Query(2, 0, IREE_HAL_MEMORY_TRANSITION_RELEASE).release));
}

TEST_F(MemoryScopeTest, CapturesNativeHostActionsAndAtomicFabricReach) {
  IREE_ASSERT_OK(iree_hal_memory_contract_initialize_transitions(
      contract_,
      [](void* user_data, uint32_t producer, uint32_t consumer,
         iree_hal_memory_pair_info_t* out_info) -> iree_status_t {
        IREE_RETURN_IF_ERROR(
            CoherentPair(nullptr, producer, consumer, out_info));
        if (producer == 1) {
          out_info->flags |= IREE_HAL_MEMORY_PAIR_MAPPING_SOURCE;
          out_info->release.kind = IREE_HAL_MEMORY_TRANSITION_KIND_RANGE;
          out_info->release.executor =
              IREE_HAL_MEMORY_TRANSITION_EXECUTOR_HOST_API;
          out_info->release.operation =
              IREE_HAL_MEMORY_TRANSITION_OPERATION_HOST_FLUSH;
          out_info->release.range_granularity = 64;
        }
        if (consumer == 1) {
          out_info->acquire.kind = IREE_HAL_MEMORY_TRANSITION_KIND_RANGE;
          out_info->acquire.executor =
              IREE_HAL_MEMORY_TRANSITION_EXECUTOR_HOST_DIRECT;
          out_info->acquire.operation =
              IREE_HAL_MEMORY_TRANSITION_OPERATION_HOST_INVALIDATE;
          out_info->acquire.range_granularity = 64;
          out_info->acquire.host.instruction =
              IREE_HAL_HOST_CACHE_INSTRUCTION_X86_CLFLUSHOPT;
          out_info->acquire.host.fence_before =
              IREE_HAL_HOST_CACHE_FENCE_X86_MFENCE;
          out_info->acquire.host.fence_after =
              IREE_HAL_HOST_CACHE_FENCE_X86_SFENCE;
        }
        out_info->atomic_reach.scope_32 = IREE_HAL_ATOMIC_REACH_FABRIC;
        out_info->atomic_reach.scope_64 = IREE_HAL_ATOMIC_REACH_DEVICE;
        out_info->estimated_fixed_cost_nanoseconds = 731;
        return iree_ok_status();
      },
      nullptr));
  iree_hal_memory_transition_pair_t pair;
  IREE_ASSERT_OK(iree_hal_memory_transition_prepare_pair(
      table(), scope(1), scope(2), IREE_HAL_MEMORY_TRANSITION_RELEASE, &pair));
  auto effects = iree_hal_memory_transition_query(table(), pair).release;
  EXPECT_TRUE(iree_hal_memory_effects_requires_resources(effects));
  EXPECT_TRUE(iree_hal_memory_effects_is_supported(effects));
  auto info = iree_hal_memory_transition_query_info(table(), pair);
  EXPECT_EQ(info.release.executor,
            IREE_HAL_MEMORY_TRANSITION_EXECUTOR_HOST_API);
  EXPECT_EQ(info.release.range_granularity, 64u);
  const auto* release_recipe = iree_hal_memory_transition_recipe(
      table(), pair, IREE_HAL_MEMORY_TRANSITION_RELEASE);
  ASSERT_NE(release_recipe, nullptr);
  ASSERT_EQ(release_recipe->operation_count, 1u);
  EXPECT_EQ(release_recipe->effects.bits, IREE_HAL_MEMORY_EFFECT_HOST_FLUSH);
  EXPECT_EQ(release_recipe->operations[0].executor,
            IREE_HAL_MEMORY_TRANSITION_EXECUTOR_HOST_API);
  EXPECT_EQ(info.atomic_reach.scope_32, IREE_HAL_ATOMIC_REACH_FABRIC);
  EXPECT_EQ(info.atomic_reach.scope_64, IREE_HAL_ATOMIC_REACH_DEVICE);
  EXPECT_EQ(info.estimated_fixed_cost_nanoseconds, 731u);
  IREE_ASSERT_OK(iree_hal_memory_transition_prepare_pair(
      table(), scope(2), scope(1), IREE_HAL_MEMORY_TRANSITION_ACQUIRE, &pair));
  info = iree_hal_memory_transition_query_info(table(), pair);
  EXPECT_EQ(info.acquire.host.instruction,
            IREE_HAL_HOST_CACHE_INSTRUCTION_X86_CLFLUSHOPT);
  EXPECT_EQ(info.acquire.host.fence_before,
            IREE_HAL_HOST_CACHE_FENCE_X86_MFENCE);
  EXPECT_EQ(info.acquire.host.fence_after,
            IREE_HAL_HOST_CACHE_FENCE_X86_SFENCE);
  IREE_ASSERT_OK(iree_hal_memory_transition_prepare_pair(
      table(), scope(0), scope(1), IREE_HAL_MEMORY_TRANSITION_ACQUIRE, &pair));
  EXPECT_EQ(iree_hal_memory_transition_query_info(table(), pair).acquire.kind,
            IREE_HAL_MEMORY_TRANSITION_KIND_UNKNOWN);
  const auto* acquire_recipe = iree_hal_memory_transition_recipe(
      table(), pair, IREE_HAL_MEMORY_TRANSITION_ACQUIRE);
  ASSERT_NE(acquire_recipe, nullptr);
  // All applicable producers require the same action, captured just once.
  EXPECT_EQ(acquire_recipe->operation_count, 1u);
  EXPECT_EQ(acquire_recipe->effects.bits,
            IREE_HAL_MEMORY_EFFECT_HOST_INVALIDATE);
  EXPECT_EQ(acquire_recipe->operations[0].range_granularity, 64u);
  EXPECT_EQ(acquire_recipe->operations[0].host.instruction,
            IREE_HAL_HOST_CACHE_INSTRUCTION_X86_CLFLUSHOPT);
}

TEST_F(MemoryScopeTest, WildcardComposesResourcesWithoutLosingNativeActions) {
  IREE_ASSERT_OK(iree_hal_memory_contract_initialize_transitions(
      contract_,
      [](void* user_data, uint32_t producer, uint32_t consumer,
         iree_hal_memory_pair_info_t* out_info) -> iree_status_t {
        IREE_RETURN_IF_ERROR(
            CoherentPair(nullptr, producer, consumer, out_info));
        if (producer == 1 && consumer != 1) {
          out_info->release.kind = IREE_HAL_MEMORY_TRANSITION_KIND_RANGE;
          out_info->release.operation =
              IREE_HAL_MEMORY_TRANSITION_OPERATION_HOST_FLUSH;
          out_info->release.executor =
              consumer < 4 ? IREE_HAL_MEMORY_TRANSITION_EXECUTOR_HOST_DIRECT
                           : IREE_HAL_MEMORY_TRANSITION_EXECUTOR_HOST_API;
          out_info->release.range_granularity = consumer < 4 ? 64 : 4096;
          if (consumer < 4) {
            out_info->release.host.instruction =
                IREE_HAL_HOST_CACHE_INSTRUCTION_X86_CLFLUSH;
            out_info->release.host.fence_after =
                IREE_HAL_HOST_CACHE_FENCE_X86_MFENCE;
          }
        }
        return iree_ok_status();
      },
      nullptr));
  iree_hal_memory_transition_pair_t pair;
  IREE_ASSERT_OK(iree_hal_memory_transition_prepare_pair(
      table(), scope(1), scope(0), IREE_HAL_MEMORY_TRANSITION_RELEASE, &pair));
  const auto* recipe = iree_hal_memory_transition_recipe(
      table(), pair, IREE_HAL_MEMORY_TRANSITION_RELEASE);
  ASSERT_NE(recipe, nullptr);
  ASSERT_EQ(recipe->operation_count, 2u);
  EXPECT_EQ(recipe->effects.bits, IREE_HAL_MEMORY_EFFECT_HOST_FLUSH);
  EXPECT_EQ(recipe->operations[0].executor,
            IREE_HAL_MEMORY_TRANSITION_EXECUTOR_HOST_DIRECT);
  EXPECT_EQ(recipe->operations[0].range_granularity, 64u);
  EXPECT_EQ(recipe->operations[1].executor,
            IREE_HAL_MEMORY_TRANSITION_EXECUTOR_HOST_API);
  EXPECT_EQ(recipe->operations[1].range_granularity, 4096u);
}

TEST_F(MemoryScopeTest, WildcardKeepsGlobalAndRangeQueueActionsDistinct) {
  IREE_ASSERT_OK(iree_hal_memory_contract_initialize_transitions(
      contract_,
      [](void* user_data, uint32_t producer, uint32_t consumer,
         iree_hal_memory_pair_info_t* out_info) -> iree_status_t {
        IREE_RETURN_IF_ERROR(
            CoherentPair(nullptr, producer, consumer, out_info));
        if (producer == 2) {
          out_info->release.kind = consumer == 4
                                       ? IREE_HAL_MEMORY_TRANSITION_KIND_GLOBAL
                                       : IREE_HAL_MEMORY_TRANSITION_KIND_RANGE;
          out_info->release.executor =
              IREE_HAL_MEMORY_TRANSITION_EXECUTOR_QUEUE;
          out_info->release.operation =
              IREE_HAL_MEMORY_TRANSITION_OPERATION_RELEASE_TO_SYSTEM;
          out_info->release.range_granularity = consumer == 4 ? 0 : 64;
        }
        return iree_ok_status();
      },
      nullptr));

  iree_hal_memory_transition_pair_t pair;
  IREE_ASSERT_OK(iree_hal_memory_transition_prepare_pair(
      table(), scope(2), scope(0), IREE_HAL_MEMORY_TRANSITION_RELEASE, &pair));
  const iree_hal_memory_effects_t effects =
      iree_hal_memory_transition_query(table(), pair).release;
  EXPECT_NE(effects.bits & IREE_HAL_MEMORY_EFFECT_GLOBAL_RELEASE_TO_SYSTEM, 0u);
  EXPECT_NE(effects.bits & IREE_HAL_MEMORY_EFFECT_RANGE_RELEASE_TO_SYSTEM, 0u);
  const auto* recipe = iree_hal_memory_transition_recipe(
      table(), pair, IREE_HAL_MEMORY_TRANSITION_RELEASE);
  ASSERT_NE(recipe, nullptr);
  EXPECT_EQ(recipe->effects.bits,
            IREE_HAL_MEMORY_EFFECT_RANGE_RELEASE_TO_SYSTEM);
  ASSERT_EQ(recipe->operation_count, 1u);
  EXPECT_EQ(recipe->operations[0].kind, IREE_HAL_MEMORY_TRANSITION_KIND_RANGE);
  EXPECT_EQ(recipe->operations[0].executor,
            IREE_HAL_MEMORY_TRANSITION_EXECUTOR_QUEUE);
}

TEST_F(MemoryScopeTest, NativeQueryFailureDoesNotPublishPartialTable) {
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_hal_memory_contract_initialize_transitions(
          contract_,
          [](void* user_data, uint32_t producer, uint32_t consumer,
             iree_hal_memory_pair_info_t* out_info) -> iree_status_t {
            if (producer == 4) {
              return iree_make_status(IREE_STATUS_UNAVAILABLE,
                                      "native query failed");
            }
            return CoherentPair(nullptr, producer, consumer, out_info);
          },
          nullptr));
  EXPECT_EQ(contract_->transition_details, nullptr);
  EXPECT_FALSE(iree_hal_memory_effects_is_supported(
      Query(2, 4, IREE_HAL_MEMORY_TRANSITION_ACQUIRE).acquire));
  IREE_ASSERT_OK(iree_hal_memory_contract_initialize_transitions(
      contract_, CoherentPair, nullptr));
  EXPECT_TRUE(iree_hal_memory_effects_is_empty(
      Query(2, 4, IREE_HAL_MEMORY_TRANSITION_ACQUIRE).acquire));
}

TEST(MemoryEffectsTest, CombinationPreservesAllRequiredExecutors) {
  iree_hal_memory_effects_t effects = {
      IREE_HAL_MEMORY_EFFECT_GLOBAL_RELEASE_TO_SYSTEM};
  for (uint32_t bits : {IREE_HAL_MEMORY_EFFECT_RANGE_ACQUIRE_FROM_SYSTEM,
                        IREE_HAL_MEMORY_EFFECT_PROGRAM_EXECUTOR,
                        IREE_HAL_MEMORY_EFFECT_UNSUPPORTED}) {
    effects = iree_hal_memory_effects_combine(effects, {bits});
    EXPECT_EQ(effects.bits & bits, bits);
  }
  EXPECT_FALSE(iree_hal_memory_effects_is_supported(effects));
  EXPECT_TRUE(iree_hal_memory_effects_requires_resources(effects));
  EXPECT_NE(effects.bits & IREE_HAL_MEMORY_EFFECT_PROGRAM_EXECUTOR, 0u);
}

TEST(MemoryEffectsTest, CombinationPreservesGlobalAndRangeGranularity) {
  const iree_hal_memory_effects_t global = {
      IREE_HAL_MEMORY_EFFECT_GLOBAL_RELEASE_TO_SYSTEM};
  const iree_hal_memory_effects_t range = {
      IREE_HAL_MEMORY_EFFECT_RANGE_RELEASE_TO_SYSTEM};
  const iree_hal_memory_effects_t combined =
      iree_hal_memory_effects_combine(global, range);
  EXPECT_NE(combined.bits & IREE_HAL_MEMORY_EFFECT_GLOBAL_RELEASE_TO_SYSTEM,
            0u);
  EXPECT_NE(combined.bits & IREE_HAL_MEMORY_EFFECT_RANGE_RELEASE_TO_SYSTEM, 0u);
  EXPECT_TRUE(iree_hal_memory_effects_requires_resources(combined));
}

}  // namespace
