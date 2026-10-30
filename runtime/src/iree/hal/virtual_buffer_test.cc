// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/virtual_buffer.h"

#include <array>

#include "iree/hal/pool.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Qualified source contracts are the dependency of the generic mapping
// admission code. No native address or physical allocation is needed to ask
// whether a replacement preserves the declared contract.
class VirtualBufferContractTest : public ::testing::Test {
 protected:
  void SetUp() override {
    for (auto& source : sources_) {
      IREE_ASSERT_OK(InitializeContract(&source.memory_contract));
      source.memory_contract->mapping_source.type =
          IREE_HAL_MEMORY_BACKEND_HOST;
      source.memory_contract->mapping_source.requirements.minimum_granularity =
          4096;
      for (uint32_t id : {2u, 3u}) {
        source.memory_contract->scopes[id].interfaces |=
            1u << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE;
      }
    }
  }

  void TearDown() override {
    for (auto& source : sources_) {
      iree_hal_memory_contract_release(source.memory_contract);
    }
  }

  iree_status_t InitializeContract(iree_hal_memory_contract_t** out_contract) {
    const iree_hal_buffer_binding_layout_t layout = {};
    IREE_RETURN_IF_ERROR(iree_hal_memory_contract_create(
        this, 4, &layout, iree_allocator_system(), out_contract));
    const iree_hal_atomic_operation_capabilities_t all = {
        IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL,
        IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL,
        IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL,
        IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL};
    for (uint32_t id : {2u, 3u}) {
      (*out_contract)->scopes[id].interfaces =
          1u << IREE_HAL_BUFFER_INTERFACE_HOST;
      (*out_contract)->scopes[id].usage = IREE_HAL_BUFFER_USAGE_STORAGE;
      iree_hal_memory_scope_access_set_atomic_operations(
          &(*out_contract)->scopes[id], all);
    }
    return iree_ok_status();
  }

  iree_status_t CapturePair(size_t index,
                            iree_hal_atomic_operation_capabilities_t producer,
                            iree_hal_atomic_operation_capabilities_t consumer) {
    iree_hal_memory_pair_info_t info = {
        .flags = IREE_HAL_MEMORY_PAIR_SHARED_BACKING_REACHABLE,
        .release = {.kind = IREE_HAL_MEMORY_TRANSITION_KIND_NONE},
        .acquire = {.kind = IREE_HAL_MEMORY_TRANSITION_KIND_NONE},
        .atomic_reach = {IREE_HAL_ATOMIC_REACH_SYSTEM,
                         IREE_HAL_ATOMIC_REACH_SYSTEM},
        .producer_atomics = producer,
        .consumer_atomics = consumer,
    };
    return iree_hal_memory_contract_initialize_transitions(
        sources_[index].memory_contract,
        +[](void* user_data, uint32_t, uint32_t,
            iree_hal_memory_pair_info_t* out_info) -> iree_status_t {
          *out_info = *static_cast<iree_hal_memory_pair_info_t*>(user_data);
          return iree_ok_status();
        },
        &info);
  }

  // Pool identities borrow their contracts during construction and
  // compatibility queries; this fixture owns the contracts independently of
  // those identities.
  std::array<iree_hal_pool_t, 3> sources_ = {};
};

TEST_F(VirtualBufferContractTest, ReplacementPreservesExactPairOperations) {
  for (auto cell :
       {&iree_hal_atomic_operation_capabilities_t::device_scope_32,
        &iree_hal_atomic_operation_capabilities_t::device_scope_64,
        &iree_hal_atomic_operation_capabilities_t::system_scope_32,
        &iree_hal_atomic_operation_capabilities_t::system_scope_64}) {
    iree_hal_atomic_operation_capabilities_t producer = {}, consumer = {};
    producer.*cell = IREE_HAL_ATOMIC_OPERATION_FLAG_STORE |
                     IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_ADD;
    consumer.*cell = IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT |
                     IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_XOR;
    auto broad_producer = producer;
    auto broad_consumer = consumer;
    broad_producer.*cell |= IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_OR;
    broad_consumer.*cell |= IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_AND;
    IREE_ASSERT_OK(CapturePair(0, broad_producer, broad_consumer));
    IREE_ASSERT_OK(CapturePair(1, producer, consumer));

    iree_hal_memory_contract_t* contract = nullptr;
    IREE_ASSERT_OK(InitializeContract(&contract));
    iree_hal_pool_t* declared_sources[] = {&sources_[0], &sources_[1]};
    iree_hal_virtual_buffer_capabilities_t capabilities = {
        .max_byte_length = 4096,
        .max_alignment = 4096,
    };
    IREE_ASSERT_OK(iree_hal_virtual_buffer_initialize_contract(
        {}, {IREE_ARRAYSIZE(declared_sources), declared_sources},
        IREE_HAL_MEMORY_BACKEND_HOST, 0, contract, &capabilities));
    EXPECT_EQ(capabilities.max_byte_length, 4096u);

    static const iree_hal_buffer_vtable_t vtable = {
        .destroy =
            +[](iree_hal_buffer_t* buffer) {
              iree_hal_memory_contract_release(buffer->memory.contract);
            },
        .query_virtual = iree_hal_virtual_buffer_query_root,
    };
    iree_hal_virtual_buffer_t target = {};
    iree_hal_virtual_buffer_initialize(
        contract, nullptr, nullptr, IREE_HAL_MEMORY_TYPE_HOST_LOCAL, 4096, 4096,
        0, IREE_HAL_MEMORY_BACKEND_HOST, &vtable, iree_allocator_system(),
        &target);
    iree_hal_memory_contract_release(contract);
    const auto table = iree_hal_buffer_transition_table(&target.buffer);
    iree_hal_memory_transition_pair_t pair;
    IREE_ASSERT_OK(iree_hal_memory_transition_prepare_pair(
        table, {this, 2}, {this, 3}, IREE_HAL_MEMORY_TRANSITION_ACQUIRE,
        &pair));
    const auto info = iree_hal_memory_transition_query_info(table, pair);
    EXPECT_EQ(info.producer_atomics.*cell, producer.*cell);
    EXPECT_EQ(info.consumer_atomics.*cell, consumer.*cell);
    for (auto* source : declared_sources) {
      EXPECT_TRUE(iree_hal_virtual_buffer_query_pool_compatibility(
          &target.buffer, source));
    }

    // One-site operations, reach, and cache recipes are identical. Losing an
    // exact guarantee on either side alone still makes the replacement unsafe.
    auto weaker_producer = producer;
    weaker_producer.*cell = IREE_HAL_ATOMIC_OPERATION_FLAG_STORE;
    IREE_ASSERT_OK(CapturePair(2, weaker_producer, consumer));
    EXPECT_FALSE(iree_hal_virtual_buffer_query_pool_compatibility(
        &target.buffer, &sources_[2]));
    auto weaker_consumer = consumer;
    weaker_consumer.*cell = IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT;
    IREE_ASSERT_OK(CapturePair(2, producer, weaker_consumer));
    EXPECT_FALSE(iree_hal_virtual_buffer_query_pool_compatibility(
        &target.buffer, &sources_[2]));
    IREE_ASSERT_OK(CapturePair(2, producer, consumer));
    EXPECT_TRUE(iree_hal_virtual_buffer_query_pool_compatibility(&target.buffer,
                                                                 &sources_[2]));
    iree_hal_buffer_release(&target.buffer);
  }
}

}  // namespace
