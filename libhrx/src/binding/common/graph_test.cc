// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/graph.h"

#include <array>
#include <cstdint>
#include <cstring>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using ::iree::Status;
using ::iree::StatusCode;
using ::iree::testing::status::StatusIs;

TEST(GraphTest, KernelParameterUpdateIsFailureAtomic) {
  constexpr size_t kArgumentCount = 3;
  std::array<iree_hal_streaming_parameter_op_t, kArgumentCount> operations = {};
  for (uint16_t i = 0; i < kArgumentCount; ++i) {
    operations[i].copy = {
        .size = sizeof(uint32_t),
        .native_abi_destination_offset =
            static_cast<uint16_t>(i * sizeof(uint32_t)),
        .source_offset = static_cast<uint16_t>(i * sizeof(uint32_t)),
        .source_ordinal = static_cast<uint16_t>(i),
        .constant_destination_offset =
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
    iree_hal_streaming_graph_node_t node = {};
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
        .count = binding_storage.size(),
        .values = binding_storage.data(),
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
        .grid_dim = {23, 29, 31},
        .block_dim = {37, 41, 43},
        .shared_memory_bytes = 47,
        .buffer = arguments.data(),
        .flags = IREE_HAL_STREAMING_DISPATCH_FLAG_ARGS_ARRAY,
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

TEST(GraphTest, KernelParameterUpdatePreservesEmptyPrepackedSpan) {
  iree_hal_streaming_graph_t graph = {};
  graph.host_allocator = iree_allocator_system();

  std::array<uint8_t, 16> constants = {};
  iree_hal_streaming_graph_node_t node = {};
  node.graph = &graph;
  node.type = IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL;
  node.attrs.kernel.constants =
      iree_make_const_byte_span(constants.data(), constants.size());
  node.attrs.kernel.constants_capacity = constants.size();
  std::array<iree_hal_buffer_ref_t, 1> binding_storage = {};
  node.attrs.kernel.bindings = {
      .count = binding_storage.size(),
      .values = binding_storage.data(),
  };
  node.attrs.kernel.binding_capacity = binding_storage.size();

  iree_hal_streaming_symbol_t symbol = {};
  symbol.type = IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION;
  symbol.parameters.constant_bytes = constants.size();
  symbol.parameters.direct_arg_bytes = constants.size();

  const iree_hal_streaming_dispatch_params_t params = {
      .buffer = reinterpret_cast<void*>(uintptr_t{1}),
      .flags = IREE_HAL_STREAMING_DISPATCH_FLAG_PRE_PACKED,
  };
  IREE_EXPECT_OK(
      iree_hal_streaming_graph_set_kernel_node_params(&node, &symbol, &params));
  EXPECT_EQ(0u, node.attrs.kernel.constants.data_length);
  EXPECT_EQ(0u, node.attrs.kernel.bindings.count);
  EXPECT_EQ(binding_storage.data(), node.attrs.kernel.bindings.values);
}

}  // namespace
