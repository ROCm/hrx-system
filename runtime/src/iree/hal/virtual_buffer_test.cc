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

  void InitializeTarget(iree_hal_memory_contract_t* contract,
                        const uint32_t* projections,
                        iree_hal_virtual_buffer_t* target) {
    static const iree_hal_buffer_vtable_t vtable = {
        .destroy =
            +[](iree_hal_buffer_t* buffer) {
              iree_hal_memory_contract_release(buffer->memory.contract);
            },
        .query_virtual = iree_hal_virtual_buffer_query_root,
    };
    iree_hal_virtual_buffer_initialize(
        contract, nullptr, nullptr, IREE_HAL_MEMORY_TYPE_HOST_LOCAL, 4096, 4096,
        0, IREE_HAL_MEMORY_BACKEND_HOST, &vtable, iree_allocator_system(),
        target);
    target->projections = projections;
  }

  // Pool identities borrow their contracts during construction and
  // compatibility queries; this fixture owns the contracts independently of
  // those identities.
  std::array<iree_hal_pool_t, 3> sources_ = {};
};

TEST_F(VirtualBufferContractTest, PlacementPreservesTheSharedSourceGuarantee) {
  for (size_t i = 0; i < sources_.size(); ++i) {
    IREE_ASSERT_OK(CapturePair(i, {}, {}));
  }
  const iree_hal_pool_placement_t placements[] = {
      {},
      {IREE_HAL_POOL_PLACEMENT_REQUIRED, 1},
      {IREE_HAL_POOL_PLACEMENT_REQUIRED, 2},
  };
  for (const auto first : placements) {
    for (const auto second : placements) {
      sources_[0].memory_contract->placement = first;
      sources_[1].memory_contract->placement = second;
      iree_hal_memory_contract_t* contract = nullptr;
      IREE_ASSERT_OK(InitializeContract(&contract));
      iree_hal_pool_t* declared[] = {&sources_[0], &sources_[1]};
      iree_hal_virtual_buffer_capabilities_t capabilities = {
          .max_byte_length = 4096,
          .max_alignment = 4096,
      };
      IREE_ASSERT_OK(iree_hal_virtual_buffer_initialize_contract(
          {}, {IREE_ARRAYSIZE(declared), declared},
          IREE_HAL_MEMORY_BACKEND_HOST, 0, contract, &capabilities));
      ASSERT_EQ(capabilities.max_byte_length, 4096u);
      const bool enforced = first.mode == IREE_HAL_POOL_PLACEMENT_REQUIRED &&
                            second.mode == first.mode &&
                            second.node == first.node;
      EXPECT_EQ(contract->placement.mode,
                enforced ? IREE_HAL_POOL_PLACEMENT_REQUIRED
                         : IREE_HAL_POOL_PLACEMENT_AUTOMATIC);
      EXPECT_EQ(contract->placement.node, enforced ? first.node : 0u);
      iree_hal_virtual_buffer_t target = {};
      InitializeTarget(contract, nullptr, &target);
      iree_hal_memory_contract_release(contract);
      for (const auto replacement : placements) {
        sources_[2].memory_contract->placement = replacement;
        EXPECT_EQ(iree_hal_virtual_buffer_query_pool_compatibility(
                      &target.buffer, {this, IREE_HAL_MEMORY_SCOPE_ANY},
                      &sources_[2]),
                  !enforced || (replacement.mode == first.mode &&
                                replacement.node == first.node));
      }
      iree_hal_buffer_release(&target.buffer);
    }
  }
}

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
          &target.buffer, {this, IREE_HAL_MEMORY_SCOPE_ANY}, source));
    }

    // One-site operations, reach, and cache recipes are identical. Losing an
    // exact guarantee on either side alone still makes the replacement unsafe.
    auto weaker_producer = producer;
    weaker_producer.*cell = IREE_HAL_ATOMIC_OPERATION_FLAG_STORE;
    IREE_ASSERT_OK(CapturePair(2, weaker_producer, consumer));
    EXPECT_FALSE(iree_hal_virtual_buffer_query_pool_compatibility(
        &target.buffer, {this, IREE_HAL_MEMORY_SCOPE_ANY}, &sources_[2]));
    auto weaker_consumer = consumer;
    weaker_consumer.*cell = IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT;
    IREE_ASSERT_OK(CapturePair(2, producer, weaker_consumer));
    EXPECT_FALSE(iree_hal_virtual_buffer_query_pool_compatibility(
        &target.buffer, {this, IREE_HAL_MEMORY_SCOPE_ANY}, &sources_[2]));
    IREE_ASSERT_OK(CapturePair(2, producer, consumer));
    EXPECT_TRUE(iree_hal_virtual_buffer_query_pool_compatibility(
        &target.buffer, {this, IREE_HAL_MEMORY_SCOPE_ANY}, &sources_[2]));
    iree_hal_buffer_release(&target.buffer);
  }
}

TEST_F(VirtualBufferContractTest, ReplacementPreservesZeroComputeExecution) {
  const iree_hal_atomic_operation_capabilities_t required = {
      .device_scope_32 = IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT,
      .device_scope_64 = IREE_HAL_ATOMIC_OPERATION_FLAG_STORE,
      .system_scope_32 = IREE_HAL_ATOMIC_OPERATION_FLAG_STORE,
      .system_scope_64 = IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT,
  };
  for (size_t i = 0; i < sources_.size(); ++i) {
    IREE_ASSERT_OK(CapturePair(i, required, required));
    iree_hal_memory_contract_set_zero_compute_atomic_operations(
        sources_[i].memory_contract, 2, required);
  }
  auto broad = required;
  broad.system_scope_64 |= IREE_HAL_ATOMIC_OPERATION_FLAG_STORE;
  iree_hal_memory_contract_set_zero_compute_atomic_operations(
      sources_[0].memory_contract, 2, broad);
  iree_hal_memory_contract_t* contract = nullptr;
  IREE_ASSERT_OK(InitializeContract(&contract));
  iree_hal_memory_contract_set_zero_compute_atomic_operations(contract, 2,
                                                              broad);
  iree_hal_pool_t* declared_sources[] = {&sources_[0], &sources_[1]};
  iree_hal_virtual_buffer_capabilities_t capabilities = {
      .max_byte_length = 4096,
      .max_alignment = 4096,
  };
  IREE_ASSERT_OK(iree_hal_virtual_buffer_initialize_contract(
      {}, {IREE_ARRAYSIZE(declared_sources), declared_sources},
      IREE_HAL_MEMORY_BACKEND_HOST, 0, contract, &capabilities));
  ASSERT_EQ(capabilities.max_byte_length, 4096u);
  static const iree_hal_buffer_vtable_t vtable = {
      .destroy =
          +[](iree_hal_buffer_t* buffer) {
            iree_hal_memory_contract_release(buffer->memory.contract);
          },
      .query_virtual = iree_hal_virtual_buffer_query_root,
  };
  iree_hal_virtual_buffer_t target = {};
  iree_hal_virtual_buffer_initialize(contract, nullptr, nullptr,
                                     IREE_HAL_MEMORY_TYPE_HOST_LOCAL, 4096,
                                     4096, 0, IREE_HAL_MEMORY_BACKEND_HOST,
                                     &vtable, iree_allocator_system(), &target);
  iree_hal_memory_contract_release(contract);
  EXPECT_EQ(iree_hal_buffer_query_zero_compute_atomic_operations(&target.buffer,
                                                                 {this, 2})
                .system_scope_64,
            required.system_scope_64);
  EXPECT_TRUE(iree_hal_virtual_buffer_query_pool_compatibility(
      &target.buffer, {this, IREE_HAL_MEMORY_SCOPE_ANY}, &sources_[2]));
  // Ordinary atomics and exact pair relationships remain unchanged. Each
  // width/scope cell must independently preserve the execution guarantee.
  for (auto cell :
       {&iree_hal_atomic_operation_capabilities_t::device_scope_32,
        &iree_hal_atomic_operation_capabilities_t::device_scope_64,
        &iree_hal_atomic_operation_capabilities_t::system_scope_32,
        &iree_hal_atomic_operation_capabilities_t::system_scope_64}) {
    auto weaker = required;
    weaker.*cell = 0;
    iree_hal_memory_contract_set_zero_compute_atomic_operations(
        sources_[2].memory_contract, 2, weaker);
    EXPECT_FALSE(iree_hal_virtual_buffer_query_pool_compatibility(
        &target.buffer, {this, IREE_HAL_MEMORY_SCOPE_ANY}, &sources_[2]));
  }
  iree_hal_buffer_release(&target.buffer);
}

TEST_F(VirtualBufferContractTest, PartialSourcesPreserveApplicableGuarantees) {
  const iree_hal_atomic_operation_capabilities_t broad = {
      .system_scope_32 = IREE_HAL_ATOMIC_OPERATION_FLAG_STORE |
                         IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT |
                         IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_ADD,
  };
  auto narrow = broad;
  narrow.system_scope_32 &= ~IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_ADD;
  // The first class serves only site 3. The second serves both sites, so only
  // the second contributes to site 2 and to the 2->3 relationship.
  sources_[0].memory_contract->scopes[2].interfaces &=
      ~(1u << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE);
  for (size_t i = 0; i < sources_.size(); ++i) {
    const auto operations = i == 0 ? narrow : broad;
    for (uint32_t id : {2u, 3u}) {
      iree_hal_memory_scope_access_set_atomic_operations(
          &sources_[i].memory_contract->scopes[id], operations);
    }
    IREE_ASSERT_OK(CapturePair(i, operations, operations));
  }
  for (bool reverse : {false, true}) {
    iree_hal_pool_t* declared[] = {&sources_[reverse], &sources_[!reverse]};
    iree_hal_memory_contract_t* contract = nullptr;
    IREE_ASSERT_OK(InitializeContract(&contract));
    iree_hal_virtual_buffer_capabilities_t capabilities = {
        .max_byte_length = 4096,
        .max_alignment = 4096,
    };
    IREE_ASSERT_OK(iree_hal_virtual_buffer_initialize_contract(
        {}, {2, declared}, IREE_HAL_MEMORY_BACKEND_HOST, 0, contract,
        &capabilities));
    ASSERT_EQ(capabilities.max_byte_length, 4096u);
    const uint32_t projections[] = {UINT32_MAX, UINT32_MAX, 0, 1};
    iree_hal_virtual_buffer_t target = {};
    InitializeTarget(contract, projections, &target);
    iree_hal_memory_contract_release(contract);
    EXPECT_EQ(iree_hal_buffer_query_atomic_operations(&target.buffer, {this, 2})
                  .system_scope_32,
              broad.system_scope_32);
    EXPECT_EQ(iree_hal_buffer_query_atomic_operations(&target.buffer, {this, 3})
                  .system_scope_32,
              narrow.system_scope_32);
    const auto table = iree_hal_buffer_transition_table(&target.buffer);
    iree_hal_memory_transition_pair_t pair;
    IREE_ASSERT_OK(iree_hal_memory_transition_prepare_pair(
        table, {this, 2}, {this, 3}, IREE_HAL_MEMORY_TRANSITION_ACQUIRE,
        &pair));
    const auto info = iree_hal_memory_transition_query_info(table, pair);
    EXPECT_EQ(info.producer_atomics.system_scope_32, broad.system_scope_32);
    // Exact pair operations also respect the destination's fixed local limit.
    EXPECT_EQ(info.consumer_atomics.system_scope_32, narrow.system_scope_32);
    EXPECT_TRUE(iree_hal_virtual_buffer_query_pool_compatibility(
        &target.buffer, {this, 3}, &sources_[0]));
    EXPECT_FALSE(iree_hal_virtual_buffer_query_pool_compatibility(
        &target.buffer, {this, 2}, &sources_[0]));
    EXPECT_FALSE(iree_hal_virtual_buffer_query_pool_compatibility(
        &target.buffer, {this, IREE_HAL_MEMORY_SCOPE_ANY}, &sources_[0]));
    EXPECT_TRUE(iree_hal_virtual_buffer_query_pool_compatibility(
        &target.buffer, {this, IREE_HAL_MEMORY_SCOPE_ANY}, &sources_[1]));
    // A replacement cannot weaken a pair merely because only one projection
    // is being mapped now: its other alias may be installed later.
    IREE_ASSERT_OK(CapturePair(2, narrow, broad));
    EXPECT_FALSE(iree_hal_virtual_buffer_query_pool_compatibility(
        &target.buffer, {this, 3}, &sources_[2]));
    IREE_ASSERT_OK(CapturePair(2, broad, broad));
    EXPECT_TRUE(iree_hal_virtual_buffer_query_pool_compatibility(
        &target.buffer, {this, 3}, &sources_[2]));
    // Two logical sites sharing page tables are inseparable even when the
    // caller names just one. Existing native projection IDs enforce this.
    const uint32_t shared_projections[] = {UINT32_MAX, UINT32_MAX, 0, 0};
    target.projections = shared_projections;
    EXPECT_FALSE(iree_hal_virtual_buffer_query_pool_compatibility(
        &target.buffer, {this, 3}, &sources_[0]));
    EXPECT_TRUE(iree_hal_virtual_buffer_query_pool_compatibility(
        &target.buffer, {this, 3}, &sources_[1]));
    iree_hal_buffer_release(&target.buffer);
  }
}

TEST_F(VirtualBufferContractTest, DisjointSourcesHaveNoSharedTransition) {
  sources_[0].memory_contract->scopes[3].interfaces &=
      ~(1u << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE);
  sources_[1].memory_contract->scopes[2].interfaces &=
      ~(1u << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE);
  IREE_ASSERT_OK(CapturePair(0, {}, {}));
  IREE_ASSERT_OK(CapturePair(1, {}, {}));
  iree_hal_pool_t* declared[] = {&sources_[0], &sources_[1]};
  iree_hal_memory_contract_t* contract = nullptr;
  IREE_ASSERT_OK(InitializeContract(&contract));
  iree_hal_virtual_buffer_capabilities_t capabilities = {
      .max_byte_length = 4096,
      .max_alignment = 4096,
  };
  IREE_ASSERT_OK(iree_hal_virtual_buffer_initialize_contract(
      {}, {2, declared}, IREE_HAL_MEMORY_BACKEND_HOST, 0, contract,
      &capabilities));
  ASSERT_EQ(capabilities.max_byte_length, 4096u);
  iree_hal_virtual_buffer_t target = {};
  InitializeTarget(contract, nullptr, &target);
  iree_hal_memory_contract_release(contract);
  const auto table = iree_hal_buffer_transition_table(&target.buffer);
  for (uint32_t producer : {0u, 2u, 3u}) {
    for (uint32_t consumer : {0u, 2u, 3u}) {
      if (!producer && !consumer) {
        continue;
      }
      iree_hal_memory_transition_pair_t pair;
      IREE_ASSERT_OK(iree_hal_memory_transition_prepare_pair(
          table, {this, producer}, {this, consumer},
          consumer ? IREE_HAL_MEMORY_TRANSITION_ACQUIRE
                   : IREE_HAL_MEMORY_TRANSITION_RELEASE,
          &pair));
      const auto transition = iree_hal_memory_transition_query(table, pair);
      const bool same_site = producer && producer == consumer;
      EXPECT_EQ(iree_hal_memory_effects_is_supported(transition.release),
                same_site);
      EXPECT_EQ(iree_hal_memory_effects_is_supported(transition.acquire),
                same_site);
    }
  }
  iree_hal_buffer_release(&target.buffer);
}

TEST_F(VirtualBufferContractTest, SourceCoverageCannotDiscardRequestedUsage) {
  for (uint32_t variant = 0; variant < 3; ++variant) {
    // Missing coverage, insufficient usage at an otherwise participating
    // site, and a wholly unused declaration all reject qualification.
    for (uint32_t id : {2u, 3u}) {
      sources_[0].memory_contract->scopes[id].interfaces =
          1u << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE;
      sources_[0].memory_contract->scopes[id].usage =
          IREE_HAL_BUFFER_USAGE_STORAGE;
    }
    if (variant == 1) {
      sources_[0].memory_contract->scopes[2].usage =
          IREE_HAL_BUFFER_USAGE_TRANSFER;
    } else {
      sources_[0].memory_contract->scopes[2].interfaces = 0;
      if (variant == 2) {
        sources_[0].memory_contract->scopes[3].interfaces = 0;
      }
    }
    iree_hal_memory_contract_t* contract = nullptr;
    IREE_ASSERT_OK(InitializeContract(&contract));
    iree_hal_pool_t* declared[] = {&sources_[0]};
    iree_hal_virtual_buffer_capabilities_t capabilities = {
        .max_byte_length = 4096,
        .max_alignment = 4096,
    };
    IREE_ASSERT_OK(iree_hal_virtual_buffer_initialize_contract(
        {}, {1, declared}, IREE_HAL_MEMORY_BACKEND_HOST, 0, contract,
        &capabilities));
    EXPECT_EQ(capabilities.max_byte_length, 0u) << variant;
    iree_hal_memory_contract_release(contract);
  }
}

}  // namespace
