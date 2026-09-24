// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/graph.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <thread>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using ::iree::Status;
using ::iree::StatusCode;
using ::iree::testing::status::StatusIs;

// Owns a dependency-free graph node using the same variable-sized allocation
// shape as production graph construction.
class GraphNodeStorage {
 public:
  GraphNodeStorage() {
    IREE_CHECK_OK(iree_allocator_malloc(iree_allocator_system(), sizeof(*node_),
                                        (void**)&node_));
    memset(node_, 0, sizeof(*node_));
  }

  ~GraphNodeStorage() { iree_allocator_free(iree_allocator_system(), node_); }

  GraphNodeStorage(const GraphNodeStorage&) = delete;
  GraphNodeStorage& operator=(const GraphNodeStorage&) = delete;

  iree_hal_streaming_graph_node_t* get() const { return node_; }

 private:
  // Allocated graph node header with no trailing dependency pointers.
  iree_hal_streaming_graph_node_t* node_ = nullptr;
};

template <size_t NodeCount>
class GraphNodeBlockStorage {
 public:
  explicit GraphNodeBlockStorage(
      const std::array<GraphNodeStorage, NodeCount>& nodes) {
    const size_t allocation_size =
        sizeof(*block_) + NodeCount * sizeof(block_->nodes[0]);
    IREE_CHECK_OK(iree_allocator_malloc(iree_allocator_system(),
                                        allocation_size, (void**)&block_));
    memset(block_, 0, allocation_size);
    block_->capacity = NodeCount;
    block_->count = NodeCount;
    for (size_t i = 0; i < NodeCount; ++i) {
      block_->nodes[i] = nodes[i].get();
    }
  }

  ~GraphNodeBlockStorage() {
    iree_allocator_free(iree_allocator_system(), block_);
  }

  GraphNodeBlockStorage(const GraphNodeBlockStorage&) = delete;
  GraphNodeBlockStorage& operator=(const GraphNodeBlockStorage&) = delete;

  iree_hal_streaming_node_block_t* get() const { return block_; }

 private:
  // Variable-sized node block populated with the owned test nodes.
  iree_hal_streaming_node_block_t* block_ = nullptr;
};

TEST(GraphTest, AddedDependenciesConstrainDetectedWorkstreams) {
  constexpr size_t kNodeCount = 64;
  std::array<GraphNodeStorage, kNodeCount> node_storage;
  GraphNodeBlockStorage<kNodeCount> node_block(node_storage);
  for (uint32_t i = 0; i < kNodeCount; ++i) {
    iree_hal_streaming_graph_node_t* node = node_storage[i].get();
    node->node_index = i;
    node->type = IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL;
  }

  std::array<iree_hal_streaming_graph_edge_t, kNodeCount - 1> edges = {};
  for (size_t i = 0; i < edges.size(); ++i) {
    edges[i].from = node_storage[i].get();
    edges[i].to = node_storage[i + 1].get();
    edges[i].next = i + 1 < edges.size() ? &edges[i + 1] : nullptr;
  }

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);
  iree_hal_streaming_graph_schedule_t schedule = {};
  IREE_ASSERT_OK(iree_hal_streaming_graph_schedule_nodes(
      node_block.get(), kNodeCount, /*disabled_nodes=*/nullptr,
      /*disabled_node_count=*/0, edges.data(), &arena, &schedule));

  ASSERT_EQ(schedule.partition_count, 1u);
  ASSERT_EQ(schedule.block_count, 1u);
  EXPECT_EQ(schedule.partitions[0].type,
            IREE_HAL_STREAMING_GRAPH_PARTITION_TYPE_RECORDABLE);
  EXPECT_EQ(schedule.partitions[0].stream_count, 1u);
  for (uint32_t i = 0; i < kNodeCount; ++i) {
    EXPECT_EQ(schedule.sorted_nodes[i].node->node_index, i);
    EXPECT_EQ(schedule.sorted_nodes[i].stream_id, 0u);
  }

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(GraphTest, ReorderedAddedDependencyUsesFinalWorkstreamIndex) {
  constexpr size_t kNodeCount = 64;
  std::array<GraphNodeStorage, kNodeCount> node_storage;
  GraphNodeBlockStorage<kNodeCount> node_block(node_storage);
  for (uint32_t i = 0; i < kNodeCount; ++i) {
    iree_hal_streaming_graph_node_t* node = node_storage[i].get();
    node->node_index = i;
    node->type = IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL;
  }

  iree_hal_streaming_graph_edge_t edge = {
      /*.next=*/nullptr,
      /*.from=*/node_storage[17].get(),
      /*.to=*/node_storage[0].get(),
  };

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);
  iree_hal_streaming_graph_schedule_t schedule = {};
  IREE_ASSERT_OK(iree_hal_streaming_graph_schedule_nodes(
      node_block.get(), kNodeCount, /*disabled_nodes=*/nullptr,
      /*disabled_node_count=*/0, &edge, &arena, &schedule));

  const uint32_t source_index = schedule.node_index_map[17];
  const uint32_t target_index = schedule.node_index_map[0];
  ASSERT_LT(source_index, target_index);
  EXPECT_EQ(schedule.sorted_nodes[source_index].node, node_storage[17].get());
  EXPECT_EQ(schedule.sorted_nodes[target_index].node, node_storage[0].get());
  ASSERT_NE(schedule.sorted_nodes[source_index].stream_id, 0u);
  EXPECT_EQ(schedule.sorted_nodes[source_index].stream_id,
            schedule.sorted_nodes[target_index].stream_id);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}
TEST(GraphTest, KernelParameterUpdateIsFailureAtomic) {
  constexpr size_t kArgumentCount = 3;
  std::array<iree_hal_streaming_parameter_op_t, kArgumentCount> operations = {};
  for (uint16_t i = 0; i < kArgumentCount; ++i) {
    operations[i].copy = {
        /*.size=*/sizeof(uint32_t),
        /*.native_abi_destination_offset=*/
        static_cast<uint16_t>(i * sizeof(uint32_t)),
        /*.source_offset=*/static_cast<uint16_t>(i * sizeof(uint32_t)),
        /*.source_ordinal=*/static_cast<uint16_t>(i),
        /*.constant_destination_offset=*/
        static_cast<uint16_t>(i * sizeof(uint32_t)),
    };
  }

  iree_hal_streaming_symbol_t symbol = {};
  symbol.type = IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION;
  symbol.parameters.buffer_size = kArgumentCount * sizeof(uint32_t);
  symbol.parameters.constant_bytes = kArgumentCount * sizeof(uint32_t);
  symbol.parameters.direct_arg_bytes = kArgumentCount * sizeof(uint32_t);
  symbol.parameters.copy_count = kArgumentCount;
  symbol.parameters.ops = operations.data();

  for (size_t missing_ordinal = 0; missing_ordinal < kArgumentCount;
       ++missing_ordinal) {
    iree_hal_streaming_graph_t graph = {};
    graph.host_allocator = iree_allocator_system();

    std::array<uint8_t, kArgumentCount * sizeof(uint32_t)> constants = {};
    memset(constants.data(), 0xA5, constants.size());
    const std::array<uint8_t, kArgumentCount * sizeof(uint32_t)>
        original_constants = constants;
    iree_hal_streaming_symbol_t previous_symbol = {};
    GraphNodeStorage node_storage;
    iree_hal_streaming_graph_node_t& node = *node_storage.get();
    node.graph = &graph;
    node.type = IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL;
    node.attrs.kernel.symbol = &previous_symbol;
    node.attrs.kernel.grid_dim[0] = 7;
    node.attrs.kernel.grid_dim[1] = 5;
    node.attrs.kernel.grid_dim[2] = 3;
    node.attrs.kernel.block_dim[0] = 11;
    node.attrs.kernel.block_dim[1] = 13;
    node.attrs.kernel.block_dim[2] = 17;
    node.attrs.kernel.shared_memory_bytes = 19;
    node.attrs.kernel.constants =
        iree_make_const_byte_span(constants.data(), constants.size());
    node.attrs.kernel.constants_capacity = constants.size();
    std::array<iree_hal_buffer_ref_t, 1> binding_storage = {
        iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/23, /*offset=*/29,
                                          /*length=*/31),
    };
    node.attrs.kernel.bindings = {
        /*.count=*/binding_storage.size(),
        /*.values=*/binding_storage.data(),
    };
    node.attrs.kernel.binding_capacity = binding_storage.size();

    std::array<uint32_t, kArgumentCount> values = {1, 2, 3};
    std::array<void*, kArgumentCount> arguments = {
        &values[0],
        &values[1],
        &values[2],
    };
    arguments[missing_ordinal] = nullptr;
    const iree_hal_streaming_dispatch_params_t params = {
        /*.grid_dim=*/{23, 29, 31},
        /*.block_dim=*/{37, 41, 43},
        /*.shared_memory_bytes=*/47,
        /*.buffer=*/arguments.data(),
        /*.buffer_size=*/0,
        /*.flags=*/IREE_HAL_STREAMING_DISPATCH_FLAG_ARGS_ARRAY,
    };

    EXPECT_THAT(Status(iree_hal_streaming_graph_set_kernel_node_params(
                    &node, &symbol, &params)),
                StatusIs(StatusCode::kInvalidArgument));
    EXPECT_EQ(original_constants, constants);
    EXPECT_EQ(&previous_symbol, node.attrs.kernel.symbol);
    EXPECT_EQ(7u, node.attrs.kernel.grid_dim[0]);
    EXPECT_EQ(5u, node.attrs.kernel.grid_dim[1]);
    EXPECT_EQ(3u, node.attrs.kernel.grid_dim[2]);
    EXPECT_EQ(11u, node.attrs.kernel.block_dim[0]);
    EXPECT_EQ(13u, node.attrs.kernel.block_dim[1]);
    EXPECT_EQ(17u, node.attrs.kernel.block_dim[2]);
    EXPECT_EQ(19u, node.attrs.kernel.shared_memory_bytes);
    EXPECT_EQ(constants.size(), node.attrs.kernel.constants.data_length);
    EXPECT_EQ(binding_storage.size(), node.attrs.kernel.bindings.count);
    EXPECT_EQ(binding_storage.data(), node.attrs.kernel.bindings.values);
    EXPECT_EQ(23u, binding_storage[0].buffer_slot);
    EXPECT_EQ(29u, binding_storage[0].offset);
    EXPECT_EQ(31u, binding_storage[0].length);
  }
}

TEST(GraphTest, KernelParameterUpdateRejectsShortPrepackedSpan) {
  iree_hal_streaming_graph_t graph = {};
  graph.host_allocator = iree_allocator_system();

  std::array<uint8_t, 16> constants = {};
  constants.fill(0x5A);
  const std::array<uint8_t, 16> original_constants = constants;
  GraphNodeStorage node_storage;
  iree_hal_streaming_graph_node_t& node = *node_storage.get();
  node.graph = &graph;
  node.type = IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL;
  iree_hal_streaming_symbol_t previous_symbol = {};
  node.attrs.kernel.symbol = &previous_symbol;
  node.attrs.kernel.grid_dim[0] = 7;
  node.attrs.kernel.block_dim[0] = 11;
  node.attrs.kernel.shared_memory_bytes = 19;
  node.attrs.kernel.constants =
      iree_make_const_byte_span(constants.data(), constants.size());
  node.attrs.kernel.constants_capacity = constants.size();
  std::array<iree_hal_buffer_ref_t, 1> binding_storage = {};
  node.attrs.kernel.bindings = {
      /*.count=*/binding_storage.size(),
      /*.values=*/binding_storage.data(),
  };
  node.attrs.kernel.binding_capacity = binding_storage.size();

  iree_hal_streaming_symbol_t symbol = {};
  symbol.type = IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION;
  symbol.parameters.constant_bytes = constants.size();
  symbol.parameters.direct_arg_bytes = constants.size();

  const iree_hal_streaming_dispatch_params_t params = {
      /*.grid_dim=*/{},
      /*.block_dim=*/{},
      /*.shared_memory_bytes=*/0,
      /*.buffer=*/reinterpret_cast<void*>(uintptr_t{1}),
      /*.buffer_size=*/0,
      /*.flags=*/IREE_HAL_STREAMING_DISPATCH_FLAG_PRE_PACKED,
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_streaming_graph_set_kernel_node_params(&node, &symbol, &params));
  EXPECT_EQ(original_constants, constants);
  EXPECT_EQ(&previous_symbol, node.attrs.kernel.symbol);
  EXPECT_EQ(7u, node.attrs.kernel.grid_dim[0]);
  EXPECT_EQ(11u, node.attrs.kernel.block_dim[0]);
  EXPECT_EQ(19u, node.attrs.kernel.shared_memory_bytes);
  EXPECT_EQ(constants.size(), node.attrs.kernel.constants.data_length);
  EXPECT_EQ(binding_storage.size(), node.attrs.kernel.bindings.count);
  EXPECT_EQ(binding_storage.data(), node.attrs.kernel.bindings.values);
}

TEST(GraphTest, KernelParameterUpdateCapturesPrepackedArgumentSpans) {
  iree_hal_streaming_graph_t graph = {};
  graph.host_allocator = iree_allocator_system();

  std::array<uint8_t, 24> constants = {};
  constants.fill(0xA5);
  GraphNodeStorage node_storage;
  iree_hal_streaming_graph_node_t& node = *node_storage.get();
  node.graph = &graph;
  node.type = IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL;
  node.attrs.kernel.constants =
      iree_make_const_byte_span(constants.data(), constants.size());
  node.attrs.kernel.constants_capacity = constants.size();

  iree_hal_streaming_symbol_t symbol = {};
  symbol.type = IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION;
  symbol.parameters.constant_bytes = 16;
  symbol.parameters.direct_arg_bytes = 16;

  std::array<uint8_t, 16> exact_arguments = {};
  for (uint8_t i = 0; i < exact_arguments.size(); ++i) {
    exact_arguments[i] = i;
  }
  const iree_hal_streaming_dispatch_params_t exact_params = {
      /*.grid_dim=*/{},
      /*.block_dim=*/{},
      /*.shared_memory_bytes=*/0,
      /*.buffer=*/exact_arguments.data(),
      /*.buffer_size=*/exact_arguments.size(),
      /*.flags=*/IREE_HAL_STREAMING_DISPATCH_FLAG_PRE_PACKED,
      /*.workitem_count=*/{},
      /*.binding_function=*/reinterpret_cast<void*>(uintptr_t{0x1234}),
  };
  IREE_EXPECT_OK(iree_hal_streaming_graph_set_kernel_node_params(
      &node, &symbol, &exact_params));
  EXPECT_EQ(exact_arguments.size(), node.attrs.kernel.constants.data_length);
  EXPECT_EQ(exact_params.binding_function, node.attrs.kernel.hip_function);
  EXPECT_EQ(0, memcmp(exact_arguments.data(), constants.data(),
                      exact_arguments.size()));
  for (size_t i = exact_arguments.size(); i < constants.size(); ++i) {
    EXPECT_EQ(0u, constants[i]);
  }
  exact_arguments.fill(0xFF);
  EXPECT_NE(0, memcmp(exact_arguments.data(), constants.data(),
                      exact_arguments.size()));

  std::array<uint8_t, 24> padded_arguments = {};
  for (uint8_t i = 0; i < padded_arguments.size(); ++i) {
    padded_arguments[i] = static_cast<uint8_t>(0x80u + i);
  }
  const iree_hal_streaming_dispatch_params_t padded_params = {
      /*.grid_dim=*/{},
      /*.block_dim=*/{},
      /*.shared_memory_bytes=*/0,
      /*.buffer=*/padded_arguments.data(),
      /*.buffer_size=*/padded_arguments.size(),
      /*.flags=*/IREE_HAL_STREAMING_DISPATCH_FLAG_PRE_PACKED,
  };
  IREE_EXPECT_OK(iree_hal_streaming_graph_set_kernel_node_params(
      &node, &symbol, &padded_params));
  EXPECT_EQ(padded_arguments.size(), node.attrs.kernel.constants.data_length);
  EXPECT_EQ(0, memcmp(padded_arguments.data(), constants.data(),
                      padded_arguments.size()));
  padded_arguments.fill(0xFF);
  EXPECT_NE(0, memcmp(padded_arguments.data(), constants.data(),
                      padded_arguments.size()));

  iree_hal_streaming_symbol_t empty_symbol = {};
  empty_symbol.type = IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION;
  const iree_hal_streaming_dispatch_params_t empty_params = {
      /*.grid_dim=*/{},
      /*.block_dim=*/{},
      /*.shared_memory_bytes=*/0,
      /*.buffer=*/nullptr,
      /*.buffer_size=*/0,
      /*.flags=*/IREE_HAL_STREAMING_DISPATCH_FLAG_PRE_PACKED,
  };
  IREE_EXPECT_OK(iree_hal_streaming_graph_set_kernel_node_params(
      &node, &empty_symbol, &empty_params));
  EXPECT_EQ(0u, node.attrs.kernel.constants.data_length);
}

TEST(GraphTest, ArgsArrayPackingProducesCompleteNativeAbiImage) {
  iree_hal_streaming_graph_t graph = {};
  graph.host_allocator = iree_allocator_system();

  constexpr iree_host_size_t kNativeArgumentSize = 52;
  std::array<uint8_t, kNativeArgumentSize> constants;
  constants.fill(0xA5);
  std::array<iree_hal_buffer_ref_t, 2> binding_storage = {};
  GraphNodeStorage node_storage;
  iree_hal_streaming_graph_node_t& node = *node_storage.get();
  node.graph = &graph;
  node.type = IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL;
  node.attrs.kernel.constants =
      iree_make_const_byte_span(constants.data(), constants.size());
  node.attrs.kernel.constants_capacity = constants.size();
  node.attrs.kernel.bindings = {
      /*.count=*/binding_storage.size(),
      /*.values=*/binding_storage.data(),
  };
  node.attrs.kernel.binding_capacity = binding_storage.size();

  std::array<iree_hal_streaming_parameter_op_t, 4> operations = {};
  operations[0].copy = {
      /*.size=*/sizeof(uint32_t),
      /*.native_abi_destination_offset=*/4,
      /*.source_offset=*/0,
      /*.source_ordinal=*/0,
      /*.constant_destination_offset=*/0,
  };
  operations[1].copy = {
      /*.size=*/sizeof(uint16_t),
      /*.native_abi_destination_offset=*/28,
      /*.source_offset=*/12,
      /*.source_ordinal=*/2,
      /*.constant_destination_offset=*/4,
  };
  operations[2].resolve = {
      /*.native_abi_destination_offset=*/16,
      /*.reserved=*/0,
      /*.source_offset=*/4,
      /*.source_ordinal=*/1,
      /*.destination_ordinal=*/1,
  };
  operations[3].resolve = {
      /*.native_abi_destination_offset=*/40,
      /*.reserved=*/0,
      /*.source_offset=*/14,
      /*.source_ordinal=*/3,
      /*.destination_ordinal=*/0,
  };
  iree_hal_streaming_symbol_t symbol = {};
  symbol.type = IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION;
  symbol.parameters.buffer_size = 22;
  symbol.parameters.constant_bytes = 6;
  symbol.parameters.direct_arg_bytes = kNativeArgumentSize;
  symbol.parameters.binding_count = 2;
  symbol.parameters.copy_count = 2;
  symbol.parameters.ops = operations.data();

  uint32_t scalar0 = 0x11223344u;
  void* pointer1 = reinterpret_cast<void*>(uintptr_t{0x0102030405060708ull});
  uint16_t scalar2 = 0x5566u;
  void* pointer3 = reinterpret_cast<void*>(uintptr_t{0x1112131415161718ull});
  std::array<void*, 4> arguments = {
      &scalar0,
      &pointer1,
      &scalar2,
      &pointer3,
  };
  const iree_hal_streaming_dispatch_params_t params = {
      /*.grid_dim=*/{1, 1, 1},
      /*.block_dim=*/{1, 1, 1},
      /*.shared_memory_bytes=*/0,
      /*.buffer=*/arguments.data(),
      /*.buffer_size=*/0,
      /*.flags=*/IREE_HAL_STREAMING_DISPATCH_FLAG_ARGS_ARRAY,
  };

  IREE_ASSERT_OK(
      iree_hal_streaming_graph_set_kernel_node_params(&node, &symbol, &params));
  std::array<uint8_t, kNativeArgumentSize> expected = {};
  memcpy(expected.data() + 4, &scalar0, sizeof(scalar0));
  const iree_hal_streaming_deviceptr_t device_pointer1 =
      static_cast<iree_hal_streaming_deviceptr_t>(
          reinterpret_cast<uintptr_t>(pointer1));
  memcpy(expected.data() + 16, &device_pointer1, sizeof(device_pointer1));
  memcpy(expected.data() + 28, &scalar2, sizeof(scalar2));
  const iree_hal_streaming_deviceptr_t device_pointer3 =
      static_cast<iree_hal_streaming_deviceptr_t>(
          reinterpret_cast<uintptr_t>(pointer3));
  memcpy(expected.data() + 40, &device_pointer3, sizeof(device_pointer3));

  EXPECT_EQ(expected.size(), node.attrs.kernel.constants.data_length);
  EXPECT_EQ(expected, constants);
  EXPECT_EQ(0u, node.attrs.kernel.bindings.count);
}

TEST(GraphTest, ArgsArrayPackingRejectsDuplicateSourceWithoutMutation) {
  iree_hal_streaming_graph_t graph = {};
  graph.host_allocator = iree_allocator_system();

  std::array<uint8_t, 16> constants;
  constants.fill(0xA5);
  const std::array<uint8_t, 16> original_constants = constants;
  std::array<iree_hal_buffer_ref_t, 1> binding_storage = {
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/23, /*offset=*/29,
                                        /*length=*/31),
  };
  GraphNodeStorage node_storage;
  iree_hal_streaming_graph_node_t& node = *node_storage.get();
  node.graph = &graph;
  node.type = IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL;
  iree_hal_streaming_symbol_t previous_symbol = {};
  node.attrs.kernel.symbol = &previous_symbol;
  node.attrs.kernel.constants =
      iree_make_const_byte_span(constants.data(), constants.size());
  node.attrs.kernel.constants_capacity = constants.size();
  node.attrs.kernel.bindings = {
      /*.count=*/binding_storage.size(),
      /*.values=*/binding_storage.data(),
  };
  node.attrs.kernel.binding_capacity = binding_storage.size();

  std::array<iree_hal_streaming_parameter_op_t, 2> operations = {};
  operations[0].copy = {
      /*.size=*/sizeof(uint32_t),
      /*.native_abi_destination_offset=*/0,
      /*.source_offset=*/0,
      /*.source_ordinal=*/0,
      /*.constant_destination_offset=*/0,
  };
  operations[1].resolve = {
      /*.native_abi_destination_offset=*/8,
      /*.reserved=*/0,
      /*.source_offset=*/4,
      /*.source_ordinal=*/0,
      /*.destination_ordinal=*/0,
  };
  iree_hal_streaming_symbol_t symbol = {};
  symbol.type = IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION;
  symbol.parameters.buffer_size = 12;
  symbol.parameters.constant_bytes = 4;
  symbol.parameters.direct_arg_bytes = constants.size();
  symbol.parameters.binding_count = 1;
  symbol.parameters.copy_count = 1;
  symbol.parameters.ops = operations.data();

  uint32_t value = 7;
  std::array<void*, 1> arguments = {&value};
  const iree_hal_streaming_dispatch_params_t params = {
      /*.grid_dim=*/{},
      /*.block_dim=*/{},
      /*.shared_memory_bytes=*/0,
      /*.buffer=*/arguments.data(),
      /*.buffer_size=*/0,
      /*.flags=*/IREE_HAL_STREAMING_DISPATCH_FLAG_ARGS_ARRAY,
  };

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_streaming_graph_set_kernel_node_params(&node, &symbol, &params));
  EXPECT_EQ(original_constants, constants);
  EXPECT_EQ(&previous_symbol, node.attrs.kernel.symbol);
  EXPECT_EQ(1u, node.attrs.kernel.bindings.count);
  EXPECT_EQ(23u, binding_storage[0].buffer_slot);
  EXPECT_EQ(29u, binding_storage[0].offset);
  EXPECT_EQ(31u, binding_storage[0].length);
}

struct ProbedHostAllocator {
  iree_allocator_t delegate = iree_allocator_system();
  bool fail_allocations = false;
  int allocation_attempt_count = 0;
  int successful_allocation_count = 0;
  int free_count = 0;

  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* params, void** inout_ptr) {
    auto* allocator = static_cast<ProbedHostAllocator*>(self);
    if (command == IREE_ALLOCATOR_COMMAND_MALLOC ||
        command == IREE_ALLOCATOR_COMMAND_CALLOC ||
        command == IREE_ALLOCATOR_COMMAND_REALLOC) {
      ++allocator->allocation_attempt_count;
      if (allocator->fail_allocations) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "injected allocation failure");
      }
      ++allocator->successful_allocation_count;
    } else if (command == IREE_ALLOCATOR_COMMAND_FREE) {
      ++allocator->free_count;
    }
    return allocator->delegate.ctl(allocator->delegate.self, command, params,
                                   inout_ptr);
  }

  iree_allocator_t AsAllocator() {
    return iree_allocator_t{this, &ProbedHostAllocator::Control};
  }
};

struct FailOnAttemptAllocator {
  ~FailOnAttemptAllocator() {
    for (size_t i = 0; i < allocation_count; ++i) {
      iree_allocator_free(delegate, allocations[i]);
    }
  }

  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* params, void** inout_ptr) {
    auto* allocator = static_cast<FailOnAttemptAllocator*>(self);
    if (command != IREE_ALLOCATOR_COMMAND_MALLOC &&
        command != IREE_ALLOCATOR_COMMAND_CALLOC) {
      return allocator->delegate.ctl(allocator->delegate.self, command, params,
                                     inout_ptr);
    }

    ++allocator->allocation_attempt_count;
    if (allocator->allocation_attempt_count ==
        allocator->fail_on_allocation_attempt) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "injected allocation failure");
    }
    IREE_RETURN_IF_ERROR(allocator->delegate.ctl(allocator->delegate.self,
                                                 command, params, inout_ptr));
    if (allocator->allocation_count >= allocator->allocations.size()) {
      iree_allocator_free(allocator->delegate, *inout_ptr);
      *inout_ptr = nullptr;
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "test allocation tracking capacity exceeded");
    }
    allocator->allocations[allocator->allocation_count++] = *inout_ptr;
    return iree_ok_status();
  }

  iree_allocator_t AsAllocator() {
    return iree_allocator_t{this, &FailOnAttemptAllocator::Control};
  }

  iree_allocator_t delegate = iree_allocator_system();
  int fail_on_allocation_attempt = 0;
  int allocation_attempt_count = 0;
  std::array<void*, 8> allocations = {};
  size_t allocation_count = 0;
};

TEST(GraphTest, NodePublicationIsFailureAtomic) {
  FailOnAttemptAllocator allocator;
  allocator.fail_on_allocation_attempt = 3;
  iree_hal_streaming_graph_t graph = {};
  graph.arena_allocator = allocator.AsAllocator();

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_streaming_graph_add_empty_node(&graph, /*dependencies=*/nullptr,
                                              /*dependency_count=*/0,
                                              /*out_node=*/nullptr));
  EXPECT_EQ(0u, graph.node_count);
  EXPECT_EQ(0u, graph.root_count);
  EXPECT_EQ(0u, graph.next_clone_source_node_index);
  EXPECT_EQ(nullptr, graph.node_blocks);
  EXPECT_EQ(nullptr, graph.current_node_block);
  EXPECT_EQ(nullptr, graph.root_blocks);
  EXPECT_EQ(nullptr, graph.current_root_block);

  allocator.fail_on_allocation_attempt = 0;
  iree_hal_streaming_graph_node_t* node = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_empty_node(
      &graph, /*dependencies=*/nullptr, /*dependency_count=*/0, &node));
  ASSERT_NE(nullptr, node);
  EXPECT_EQ(0u, node->node_index);
  EXPECT_EQ(0u, node->clone_source_node_index);
  EXPECT_EQ(1u, graph.node_count);
  EXPECT_EQ(1u, graph.root_count);
}

struct BlockingHostAllocator {
  iree_allocator_t delegate = iree_allocator_system();
  std::atomic<bool> block_next_allocation = false;
  std::atomic<bool> allocation_entered = false;
  std::atomic<bool> release_allocation = false;

  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* params, void** inout_ptr) {
    auto* allocator = static_cast<BlockingHostAllocator*>(self);
    const bool is_allocation = command == IREE_ALLOCATOR_COMMAND_MALLOC ||
                               command == IREE_ALLOCATOR_COMMAND_CALLOC ||
                               command == IREE_ALLOCATOR_COMMAND_REALLOC;
    if (is_allocation && allocator->block_next_allocation.exchange(
                             false, std::memory_order_acq_rel)) {
      allocator->allocation_entered.store(true, std::memory_order_release);
      while (!allocator->release_allocation.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
    }
    return allocator->delegate.ctl(allocator->delegate.self, command, params,
                                   inout_ptr);
  }

  iree_allocator_t AsAllocator() {
    return iree_allocator_t{this, &BlockingHostAllocator::Control};
  }
};

struct CaptureRecordGate {
  // Set once the recorder is executing under the graph/session transaction.
  std::atomic<bool> entered = false;
  // Set by the test after a concurrent operation has attempted the transaction.
  std::atomic<bool> release = false;
};

iree_status_t RecordGatedCaptureNode(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count, void* user_data,
    iree_hal_streaming_graph_node_t** out_terminal_node) {
  auto* gate = static_cast<CaptureRecordGate*>(user_data);
  gate->entered.store(true, std::memory_order_release);
  while (!gate->release.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  return iree_hal_streaming_graph_add_empty_node(
      graph, dependencies, dependency_count, out_terminal_node);
}

iree_status_t FailCaptureNodeRecording(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count, void* user_data,
    iree_hal_streaming_graph_node_t** out_terminal_node) {
  (void)graph;
  (void)dependencies;
  (void)dependency_count;
  (void)user_data;
  (void)out_terminal_node;
  return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                          "injected capture construction failure");
}

iree_status_t RecordLinkedNodeThenFail(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count, void* user_data,
    iree_hal_streaming_graph_node_t** out_terminal_node) {
  (void)user_data;
  iree_hal_streaming_graph_node_t* linked_node = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_add_empty_node(
      graph, dependencies, dependency_count, &linked_node));
  *out_terminal_node = linked_node;
  return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                          "injected failure after graph mutation");
}

// Initializes production graphs and the minimum stream/context state needed to
// exercise shared capture transactions without creating a device.
class CaptureTransactionTestState {
 public:
  CaptureTransactionTestState() {
    iree_atomic_ref_count_init(&context_.ref_count);
    iree_slim_mutex_initialize(&context_.stream_list_mutex);
    iree_atomic_store(&context_.capture_stream_count, 0,
                      iree_memory_order_release);
    context_.next_capture_id = 2;
    context_.host_allocator = iree_allocator_system();
    context_.device_entry = &device_entry_;
    iree_arena_block_pool_initialize(/*block_size=*/64 * 1024,
                                     iree_allocator_system(),
                                     &device_entry_.block_pool);

    InitializeStream(&origin_, /*stream_id=*/1);
    InitializeStream(&participant_, /*stream_id=*/2);
    InitializeStream(&waiter_, /*stream_id=*/3);
    streams_[0] = &origin_;
    streams_[1] = &participant_;
    streams_[2] = &waiter_;
    context_.streams = streams_;
    context_.stream_count = std::size(streams_);
    context_.stream_capacity = std::size(streams_);

    IREE_CHECK_OK(iree_hal_streaming_graph_create(
        &context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
        &graph_));
    IREE_CHECK_OK(
        iree_hal_streaming_graph_add_empty_node(graph_, nullptr, 0, &node_));
    IREE_CHECK_OK(iree_hal_streaming_graph_create(
        &context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
        &second_graph_));

    iree_slim_mutex_lock(&graph_->capture_mutex);
    graph_->capture_id = 1;
    graph_->capture_mode = IREE_HAL_STREAMING_CAPTURE_MODE_RELAXED;
    iree_atomic_store(&graph_->capture_state,
                      IREE_HAL_STREAMING_GRAPH_CAPTURE_STATE_ACTIVE,
                      iree_memory_order_release);
    iree_slim_mutex_lock(&origin_.mutex);
    origin_.capture_mode = IREE_HAL_STREAMING_CAPTURE_MODE_RELAXED;
    origin_.capture_graph = graph_;
    origin_.capture_graph_owned = false;
    origin_.capture_origin = true;
    origin_.capture_id = 1;
    iree_hal_streaming_stream_set_capture_status(
        &origin_, IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE);
    iree_slim_mutex_unlock(&origin_.mutex);
    iree_slim_mutex_unlock(&graph_->capture_mutex);
  }

  ~CaptureTransactionTestState() {
    iree_hal_streaming_context_unregister_stream(&context_, &waiter_);
    iree_hal_streaming_context_unregister_stream(&context_, &participant_);
    iree_hal_streaming_context_unregister_stream(&context_, &origin_);
    iree_allocator_free(origin_.host_allocator, origin_.capture_dependencies);
    iree_allocator_free(participant_.host_allocator,
                        participant_.capture_dependencies);
    iree_allocator_free(waiter_.host_allocator, waiter_.capture_dependencies);
    iree_hal_streaming_graph_release(second_graph_);
    iree_hal_streaming_graph_release(graph_);
    iree_slim_mutex_deinitialize(&waiter_.mutex);
    iree_slim_mutex_deinitialize(&participant_.mutex);
    iree_slim_mutex_deinitialize(&origin_.mutex);
    iree_slim_mutex_deinitialize(&context_.stream_list_mutex);
    iree_arena_block_pool_deinitialize(&device_entry_.block_pool);
  }

  iree_status_t SetOriginFrontierToPrimaryNode() {
    iree_hal_streaming_graph_node_t* dependencies[] = {node_};
    return iree_hal_streaming_update_capture_dependencies(
        &origin_, dependencies, std::size(dependencies),
        IREE_HAL_STREAMING_CAPTURE_DEPENDENCIES_SET);
  }

  iree_status_t JoinParticipantAtPrimaryNode() {
    iree_hal_streaming_graph_node_t* dependencies[] = {node_};
    return iree_hal_streaming_capture_join_graph(&participant_, graph_,
                                                 /*capture_id=*/1, dependencies,
                                                 std::size(dependencies));
  }

  iree_status_t BeginSecondGraphCapture() {
    return iree_hal_streaming_begin_capture_to_graph(
        &waiter_, second_graph_, nullptr, 0,
        IREE_HAL_STREAMING_CAPTURE_MODE_RELAXED);
  }

  iree_hal_streaming_context_t* context() { return &context_; }
  iree_hal_streaming_stream_t* origin() { return &origin_; }
  iree_hal_streaming_stream_t* participant() { return &participant_; }
  iree_hal_streaming_stream_t* waiter() { return &waiter_; }
  iree_hal_streaming_graph_t* graph() { return graph_; }
  iree_hal_streaming_graph_t* second_graph() { return second_graph_; }
  iree_hal_streaming_graph_node_t* node() { return node_; }

 private:
  void InitializeStream(iree_hal_streaming_stream_t* stream,
                        unsigned long long stream_id) {
    iree_atomic_ref_count_init_value(&stream->ref_count, 2);
    iree_slim_mutex_initialize(&stream->mutex);
    stream->context = &context_;
    stream->registration_state =
        IREE_HAL_STREAMING_STREAM_REGISTRATION_STATE_REGISTERED;
    stream->stream_id = stream_id;
    stream->host_allocator = iree_allocator_system();
    stream->capture_status = IREE_HAL_STREAMING_CAPTURE_STATUS_NONE;
  }

  iree_hal_streaming_context_t context_ = {};
  iree_hal_streaming_device_t device_entry_ = {};
  iree_hal_streaming_stream_t origin_ = {};
  iree_hal_streaming_stream_t participant_ = {};
  iree_hal_streaming_stream_t waiter_ = {};
  iree_hal_streaming_stream_t* streams_[3] = {};
  iree_hal_streaming_graph_t* graph_ = nullptr;
  iree_hal_streaming_graph_t* second_graph_ = nullptr;
  iree_hal_streaming_graph_node_t* node_ = nullptr;
};

TEST(GraphTest, ParticipantMutationSerializesOriginTermination) {
  CaptureTransactionTestState state;
  IREE_ASSERT_OK(state.SetOriginFrontierToPrimaryNode());
  IREE_ASSERT_OK(state.JoinParticipantAtPrimaryNode());
  const iree_host_size_t initial_node_count = state.graph()->node_count;

  CaptureRecordGate gate;
  bool was_capturing = false;
  std::atomic<iree_status_code_t> record_status_code = IREE_STATUS_UNKNOWN;
  std::thread record_thread([&] {
    iree_status_t record_status = iree_hal_streaming_capture_try_record_node(
        state.participant(), RecordGatedCaptureNode, &gate, &was_capturing);
    record_status_code.store(iree_status_code(record_status),
                             std::memory_order_release);
    iree_status_ignore(record_status);
  });
  while (!gate.entered.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }

  const int32_t graph_refs_before_end =
      iree_atomic_ref_count_load(&state.graph()->ref_count);
  std::atomic<bool> end_completed = false;
  iree_hal_streaming_graph_t* captured_graph = nullptr;
  std::atomic<iree_status_code_t> end_status_code = IREE_STATUS_UNKNOWN;
  std::thread end_thread([&] {
    iree_status_t end_status =
        iree_hal_streaming_end_capture(state.origin(), &captured_graph);
    end_status_code.store(iree_status_code(end_status),
                          std::memory_order_release);
    iree_status_ignore(end_status);
    end_completed.store(true, std::memory_order_release);
  });
  bool observed_end_snapshot = false;
  for (int i = 0; i < 1000000; ++i) {
    if (iree_atomic_ref_count_load(&state.graph()->ref_count) >
        graph_refs_before_end) {
      observed_end_snapshot = true;
      break;
    }
    if (end_completed.load(std::memory_order_acquire)) {
      break;
    }
    std::this_thread::yield();
  }
  EXPECT_TRUE(observed_end_snapshot);
  // Keep the recorder gated long enough for an incorrectly unlocked end to
  // finish; the graph-wide transaction must keep termination blocked.
  if (observed_end_snapshot) {
    for (int i = 0;
         i < 100000 && !end_completed.load(std::memory_order_acquire); ++i) {
      std::this_thread::yield();
    }
  }
  EXPECT_FALSE(end_completed.load(std::memory_order_acquire));

  gate.release.store(true, std::memory_order_release);
  record_thread.join();
  end_thread.join();

  EXPECT_EQ(IREE_STATUS_OK, record_status_code.load(std::memory_order_acquire));
  EXPECT_EQ(IREE_STATUS_ABORTED,
            end_status_code.load(std::memory_order_acquire));
  EXPECT_TRUE(was_capturing);
  EXPECT_EQ(initial_node_count + 1, state.graph()->node_count);
  EXPECT_EQ(nullptr, captured_graph);
}
TEST(GraphTest, ParticipantPartialPrefixInvalidatesSharedCapture) {
  CaptureTransactionTestState state;
  IREE_ASSERT_OK(state.JoinParticipantAtPrimaryNode());
  const iree_host_size_t initial_node_count = state.graph()->node_count;

  bool was_capturing = false;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_hal_streaming_capture_try_record_node(
                            state.participant(), RecordLinkedNodeThenFail,
                            nullptr, &was_capturing));
  EXPECT_TRUE(was_capturing);
  EXPECT_EQ(initial_node_count + 1, state.graph()->node_count);

  iree_hal_streaming_capture_status_t origin_status =
      IREE_HAL_STREAMING_CAPTURE_STATUS_NONE;
  IREE_EXPECT_OK(iree_hal_streaming_capture_status(state.origin(),
                                                   &origin_status, nullptr));
  EXPECT_EQ(IREE_HAL_STREAMING_CAPTURE_STATUS_INVALIDATED, origin_status);
  iree_hal_streaming_graph_t* captured_graph = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      iree_hal_streaming_end_capture(state.origin(), &captured_graph));
  EXPECT_EQ(nullptr, captured_graph);
}

TEST(GraphTest, ParticipantUnregisterSerializesAndRejectsReadoption) {
  CaptureTransactionTestState state;
  IREE_ASSERT_OK(state.SetOriginFrontierToPrimaryNode());
  IREE_ASSERT_OK(state.JoinParticipantAtPrimaryNode());

  iree_slim_mutex_lock(&state.graph()->capture_mutex);
  const int32_t graph_refs_before_unregister =
      iree_atomic_ref_count_load(&state.graph()->ref_count);
  std::atomic<bool> unregister_completed = false;
  std::thread unregister_thread([&] {
    iree_hal_streaming_context_unregister_stream(state.context(),
                                                 state.participant());
    unregister_completed.store(true, std::memory_order_release);
  });
  bool observed_unregister_snapshot = false;
  for (int i = 0; i < 1000000; ++i) {
    if (iree_atomic_ref_count_load(&state.graph()->ref_count) >
        graph_refs_before_unregister) {
      observed_unregister_snapshot = true;
      break;
    }
    if (unregister_completed.load(std::memory_order_acquire)) {
      break;
    }
    std::this_thread::yield();
  }
  // Keep the graph lock held long enough for an incorrectly unlocked
  // unregister to finish; the real lifecycle transaction must remain blocked.
  if (observed_unregister_snapshot) {
    for (int i = 0;
         i < 100000 && !unregister_completed.load(std::memory_order_acquire);
         ++i) {
      std::this_thread::yield();
    }
  }
  EXPECT_TRUE(observed_unregister_snapshot);
  EXPECT_FALSE(unregister_completed.load(std::memory_order_acquire));

  iree_slim_mutex_unlock(&state.graph()->capture_mutex);
  unregister_thread.join();
  EXPECT_TRUE(unregister_completed.load(std::memory_order_acquire));
  EXPECT_EQ(2u, state.context()->stream_count);
  EXPECT_EQ(IREE_HAL_STREAMING_STREAM_REGISTRATION_STATE_UNREGISTERED,
            state.participant()->registration_state);
  EXPECT_EQ(IREE_HAL_STREAMING_CAPTURE_STATUS_NONE,
            state.participant()->capture_status);
  EXPECT_EQ(nullptr, state.participant()->capture_graph);

  iree_hal_streaming_graph_t* captured_graph = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      iree_hal_streaming_end_capture(state.origin(), &captured_graph));
  EXPECT_EQ(nullptr, captured_graph);

  IREE_ASSERT_OK(state.BeginSecondGraphCapture());
  unsigned long long second_capture_id = 0;
  iree_hal_streaming_capture_status_t second_capture_status =
      IREE_HAL_STREAMING_CAPTURE_STATUS_NONE;
  IREE_ASSERT_OK(iree_hal_streaming_capture_status(
      state.waiter(), &second_capture_status, &second_capture_id));
  ASSERT_EQ(IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE, second_capture_status);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_hal_streaming_capture_join_graph(
                            state.participant(), state.second_graph(),
                            second_capture_id, nullptr, 0));
  EXPECT_EQ(IREE_HAL_STREAMING_CAPTURE_STATUS_NONE,
            state.participant()->capture_status);

  IREE_EXPECT_OK(
      iree_hal_streaming_end_capture(state.waiter(), &captured_graph));
  EXPECT_EQ(state.second_graph(), captured_graph);
}

TEST(GraphTest, OriginUnregisterDetachesJoinedSessionAndParticipantCanReuse) {
  CaptureTransactionTestState state;
  IREE_ASSERT_OK(state.SetOriginFrontierToPrimaryNode());
  IREE_ASSERT_OK(state.JoinParticipantAtPrimaryNode());

  // Force unregister to take its retained snapshot and then stop at the graph
  // transaction. This proves origin destruction uses the same lifecycle lock
  // as adoption, recording, and end-capture.
  iree_slim_mutex_lock(&state.graph()->capture_mutex);
  const int32_t graph_refs_before_unregister =
      iree_atomic_ref_count_load(&state.graph()->ref_count);
  std::atomic<bool> unregister_completed = false;
  std::thread unregister_thread([&] {
    iree_hal_streaming_context_unregister_stream(state.context(),
                                                 state.origin());
    unregister_completed.store(true, std::memory_order_release);
  });
  bool observed_unregister_snapshot = false;
  for (int i = 0; i < 1000000; ++i) {
    if (iree_atomic_ref_count_load(&state.graph()->ref_count) >
        graph_refs_before_unregister) {
      observed_unregister_snapshot = true;
      break;
    }
    if (unregister_completed.load(std::memory_order_acquire)) {
      break;
    }
    std::this_thread::yield();
  }
  EXPECT_TRUE(observed_unregister_snapshot);
  EXPECT_FALSE(unregister_completed.load(std::memory_order_acquire));

  iree_slim_mutex_unlock(&state.graph()->capture_mutex);
  unregister_thread.join();
  EXPECT_TRUE(unregister_completed.load(std::memory_order_acquire));
  EXPECT_EQ(IREE_HAL_STREAMING_STREAM_REGISTRATION_STATE_UNREGISTERED,
            state.origin()->registration_state);
  EXPECT_EQ(IREE_HAL_STREAMING_CAPTURE_STATUS_NONE,
            state.participant()->capture_status);
  EXPECT_EQ(nullptr, state.participant()->capture_graph);
  EXPECT_EQ(0u, state.participant()->capture_id);
  EXPECT_EQ(0u, state.graph()->capture_id);
  EXPECT_EQ(IREE_HAL_STREAMING_GRAPH_CAPTURE_STATE_INACTIVE,
            iree_atomic_load(&state.graph()->capture_state,
                             iree_memory_order_acquire));

  // The surviving participant must not remain a non-origin member of the dead
  // session. Reusing both it and the same graph is the observable guarantee.
  IREE_ASSERT_OK(iree_hal_streaming_begin_capture_to_graph(
      state.participant(), state.graph(), nullptr, 0,
      IREE_HAL_STREAMING_CAPTURE_MODE_RELAXED));
  iree_hal_streaming_capture_status_t capture_status =
      IREE_HAL_STREAMING_CAPTURE_STATUS_NONE;
  unsigned long long capture_id = 0;
  IREE_ASSERT_OK(iree_hal_streaming_capture_status(
      state.participant(), &capture_status, &capture_id));
  EXPECT_EQ(IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE, capture_status);
  EXPECT_NE(0u, capture_id);
  EXPECT_NE(1u, capture_id);

  iree_hal_streaming_graph_t* captured_graph = nullptr;
  IREE_EXPECT_OK(
      iree_hal_streaming_end_capture(state.participant(), &captured_graph));
  EXPECT_EQ(state.graph(), captured_graph);
}

TEST(GraphTest, CapturedEventWaitUsesAtomicSessionFrontierSnapshot) {
  CaptureTransactionTestState state;
  IREE_ASSERT_OK(state.SetOriginFrontierToPrimaryNode());
  IREE_ASSERT_OK(state.BeginSecondGraphCapture());

  BlockingHostAllocator allocator;
  iree_hal_streaming_event_t* event = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_event_create(
      state.context(), IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING,
      allocator.AsAllocator(), &event));
  IREE_ASSERT_OK(iree_hal_streaming_event_record(event, state.origin()));
  allocator.block_next_allocation.store(true, std::memory_order_release);

  std::atomic<bool> wait_completed = false;
  std::atomic<iree_status_code_t> wait_status_code = IREE_STATUS_UNKNOWN;
  std::thread wait_thread([&] {
    iree_status_t status = iree_hal_streaming_stream_wait_event(
        state.participant(), event, /*capture_external_wait=*/false);
    wait_status_code.store(iree_status_code(status), std::memory_order_release);
    iree_status_ignore(status);
    wait_completed.store(true, std::memory_order_release);
  });
  bool observed_blocked_snapshot = false;
  for (int i = 0; i < 1000000; ++i) {
    if (allocator.allocation_entered.load(std::memory_order_acquire)) {
      observed_blocked_snapshot = true;
      break;
    }
    if (wait_completed.load(std::memory_order_acquire)) {
      break;
    }
    std::this_thread::yield();
  }
  if (!observed_blocked_snapshot) {
    allocator.release_allocation.store(true, std::memory_order_release);
    wait_thread.join();
    iree_hal_streaming_event_release(event);
    FAIL() << "captured-event snapshot allocation was not reached";
    return;
  }
  EXPECT_FALSE(wait_completed.load(std::memory_order_acquire));

  const int32_t second_graph_refs_before_record =
      iree_atomic_ref_count_load(&state.second_graph()->ref_count);
  std::atomic<bool> record_completed = false;
  std::atomic<iree_status_code_t> record_status_code = IREE_STATUS_UNKNOWN;
  std::thread record_thread([&] {
    iree_status_t status =
        iree_hal_streaming_event_record(event, state.waiter());
    record_status_code.store(iree_status_code(status),
                             std::memory_order_release);
    iree_status_ignore(status);
    record_completed.store(true, std::memory_order_release);
  });
  bool observed_blocked_record = false;
  for (int i = 0; i < 1000000; ++i) {
    if (iree_atomic_ref_count_load(&state.second_graph()->ref_count) >
        second_graph_refs_before_record) {
      observed_blocked_record = true;
      break;
    }
    if (record_completed.load(std::memory_order_acquire)) {
      break;
    }
    std::this_thread::yield();
  }
  EXPECT_TRUE(observed_blocked_record);
  EXPECT_FALSE(record_completed.load(std::memory_order_acquire));

  allocator.release_allocation.store(true, std::memory_order_release);
  wait_thread.join();
  record_thread.join();
  EXPECT_EQ(IREE_STATUS_OK, wait_status_code.load(std::memory_order_acquire));
  EXPECT_EQ(IREE_STATUS_OK, record_status_code.load(std::memory_order_acquire));
  EXPECT_EQ(state.graph(), state.participant()->capture_graph);
  EXPECT_EQ(1u, state.participant()->capture_id);

  unsigned long long second_capture_id = 0;
  iree_hal_streaming_graph_t* event_graph =
      iree_hal_streaming_event_acquire_capture_graph(event, &second_capture_id);
  EXPECT_EQ(state.second_graph(), event_graph);
  EXPECT_NE(0u, second_capture_id);
  iree_hal_streaming_graph_release(event_graph);
  iree_hal_streaming_event_release(event);
}

TEST(GraphTest, StaleCapturedEventCannotAffectReusedGraphSession) {
  CaptureTransactionTestState state;
  iree_hal_streaming_event_t* event = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_event_create(
      state.context(), IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING,
      iree_allocator_system(), &event));
  IREE_ASSERT_OK(iree_hal_streaming_event_record(event, state.origin()));
  unsigned long long stale_capture_id = 0;
  iree_hal_streaming_graph_t* event_graph =
      iree_hal_streaming_event_acquire_capture_graph(event, &stale_capture_id);
  ASSERT_EQ(state.graph(), event_graph);
  iree_hal_streaming_graph_release(event_graph);

  iree_hal_streaming_graph_t* first_capture_graph = nullptr;
  IREE_ASSERT_OK(
      iree_hal_streaming_end_capture(state.origin(), &first_capture_graph));
  ASSERT_EQ(state.graph(), first_capture_graph);
  IREE_ASSERT_OK(iree_hal_streaming_begin_capture_to_graph(
      state.origin(), state.graph(), nullptr, 0,
      IREE_HAL_STREAMING_CAPTURE_MODE_RELAXED));

  unsigned long long current_capture_id = 0;
  iree_hal_streaming_capture_status_t current_status =
      IREE_HAL_STREAMING_CAPTURE_STATUS_NONE;
  IREE_ASSERT_OK(iree_hal_streaming_capture_status(
      state.origin(), &current_status, &current_capture_id));
  ASSERT_EQ(IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE, current_status);
  ASSERT_NE(stale_capture_id, current_capture_id);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      iree_hal_streaming_stream_wait_event(state.participant(), event,
                                           /*capture_external_wait=*/false));
  EXPECT_FALSE(iree_hal_streaming_capture_graph_invalidate(state.graph(),
                                                           stale_capture_id));
  EXPECT_FALSE(iree_hal_streaming_capture_graph_invalidate(state.graph(), 0));
  IREE_EXPECT_OK(iree_hal_streaming_capture_status(state.origin(),
                                                   &current_status, nullptr));
  EXPECT_EQ(IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE, current_status);

  iree_hal_streaming_graph_t* second_capture_graph = nullptr;
  IREE_EXPECT_OK(
      iree_hal_streaming_end_capture(state.origin(), &second_capture_graph));
  EXPECT_EQ(state.graph(), second_capture_graph);
  iree_hal_streaming_event_release(event);
}

TEST(GraphTest, CaptureRecordingFailureInvalidatesCapture) {
  CaptureTransactionTestState state;
  bool was_capturing = false;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_streaming_capture_try_record_node(
          state.origin(), FailCaptureNodeRecording, nullptr, &was_capturing));
  EXPECT_TRUE(was_capturing);
  EXPECT_EQ(IREE_HAL_STREAMING_CAPTURE_STATUS_INVALIDATED,
            state.origin()->capture_status);
  EXPECT_EQ(IREE_HAL_STREAMING_GRAPH_CAPTURE_STATE_INVALIDATED,
            iree_atomic_load(&state.graph()->capture_state,
                             iree_memory_order_acquire));
}

TEST(GraphTest, CaptureInvalidationUpdatesGraphTransaction) {
  CaptureTransactionTestState state;

  EXPECT_EQ(IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE,
            iree_hal_streaming_capture_invalidate(state.origin()));
  EXPECT_EQ(IREE_HAL_STREAMING_CAPTURE_STATUS_INVALIDATED,
            state.origin()->capture_status);
  EXPECT_EQ(IREE_HAL_STREAMING_GRAPH_CAPTURE_STATE_INVALIDATED,
            iree_atomic_load(&state.graph()->capture_state,
                             iree_memory_order_acquire));
  EXPECT_EQ(IREE_HAL_STREAMING_CAPTURE_STATUS_INVALIDATED,
            iree_hal_streaming_capture_invalidate(state.origin()));
}

TEST(GraphTest, CaptureNoOpPreservesFrontier) {
  CaptureTransactionTestState state;
  bool was_capturing = false;
  IREE_EXPECT_OK(iree_hal_streaming_capture_try_record_noop(state.origin(),
                                                            &was_capturing));
  EXPECT_TRUE(was_capturing);
  EXPECT_EQ(IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE,
            state.origin()->capture_status);
  EXPECT_EQ(0u, state.origin()->capture_dependency_count);
  EXPECT_EQ(0u, state.origin()->capture_dependency_capacity);
  EXPECT_EQ(1u, state.graph()->node_count);
}

void InitializeSingleCopySymbol(uint16_t direct_arg_bytes,
                                uint16_t destination_offset,
                                iree_hal_streaming_parameter_op_t* operation,
                                iree_hal_streaming_symbol_t* out_symbol) {
  operation->copy = {
      /*.size=*/sizeof(uint32_t),
      /*.native_abi_destination_offset=*/destination_offset,
      /*.source_offset=*/0,
      /*.source_ordinal=*/0,
      /*.constant_destination_offset=*/0,
  };
  out_symbol->type = IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION;
  out_symbol->parameters.buffer_size = sizeof(uint32_t);
  out_symbol->parameters.constant_bytes = sizeof(uint32_t);
  out_symbol->parameters.direct_arg_bytes = direct_arg_bytes;
  out_symbol->parameters.copy_count = 1;
  out_symbol->parameters.ops = operation;
}

TEST(GraphTest, LaunchUsesInlineArgumentStorageForSmallMetadata) {
  ProbedHostAllocator allocator;
  iree_hal_streaming_stream_t stream = {};
  stream.host_allocator = allocator.AsAllocator();
  iree_hal_streaming_parameter_op_t operation = {};
  iree_hal_streaming_symbol_t symbol = {};
  InitializeSingleCopySymbol(/*direct_arg_bytes=*/128,
                             /*destination_offset=*/64, &operation, &symbol);
  std::array<void*, 1> arguments = {nullptr};
  const iree_hal_streaming_dispatch_params_t params = {
      /*.grid_dim=*/{},
      /*.block_dim=*/{},
      /*.shared_memory_bytes=*/0,
      /*.buffer=*/arguments.data(),
      /*.buffer_size=*/0,
      /*.flags=*/IREE_HAL_STREAMING_DISPATCH_FLAG_ARGS_ARRAY,
  };

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_streaming_launch_kernel(&symbol, &params, &stream));
  EXPECT_EQ(0, allocator.allocation_attempt_count);
  EXPECT_EQ(0, allocator.free_count);
}

TEST(GraphTest, LaunchFreesHeapArgumentStorageAfterPackingFailure) {
  ProbedHostAllocator allocator;
  iree_hal_streaming_stream_t stream = {};
  stream.host_allocator = allocator.AsAllocator();
  iree_hal_streaming_parameter_op_t operation = {};
  iree_hal_streaming_symbol_t symbol = {};
  InitializeSingleCopySymbol(/*direct_arg_bytes=*/512,
                             /*destination_offset=*/256, &operation, &symbol);
  std::array<void*, 1> arguments = {nullptr};
  const iree_hal_streaming_dispatch_params_t params = {
      /*.grid_dim=*/{},
      /*.block_dim=*/{},
      /*.shared_memory_bytes=*/0,
      /*.buffer=*/arguments.data(),
      /*.buffer_size=*/0,
      /*.flags=*/IREE_HAL_STREAMING_DISPATCH_FLAG_ARGS_ARRAY,
  };

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_streaming_launch_kernel(&symbol, &params, &stream));
  EXPECT_EQ(1, allocator.allocation_attempt_count);
  EXPECT_EQ(1, allocator.successful_allocation_count);
  EXPECT_EQ(1, allocator.free_count);
}

TEST(GraphTest, LaunchReportsHeapArgumentStorageAllocationFailure) {
  ProbedHostAllocator allocator;
  allocator.fail_allocations = true;
  iree_hal_streaming_stream_t stream = {};
  stream.host_allocator = allocator.AsAllocator();
  iree_hal_streaming_parameter_op_t operation = {};
  iree_hal_streaming_symbol_t symbol = {};
  InitializeSingleCopySymbol(/*direct_arg_bytes=*/512,
                             /*destination_offset=*/256, &operation, &symbol);
  uint32_t value = 7;
  std::array<void*, 1> arguments = {&value};
  const iree_hal_streaming_dispatch_params_t params = {
      /*.grid_dim=*/{},
      /*.block_dim=*/{},
      /*.shared_memory_bytes=*/0,
      /*.buffer=*/arguments.data(),
      /*.buffer_size=*/0,
      /*.flags=*/IREE_HAL_STREAMING_DISPATCH_FLAG_ARGS_ARRAY,
  };

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_streaming_launch_kernel(&symbol, &params, &stream));
  EXPECT_EQ(1, allocator.allocation_attempt_count);
  EXPECT_EQ(0, allocator.successful_allocation_count);
  EXPECT_EQ(0, allocator.free_count);
}

}  // namespace
