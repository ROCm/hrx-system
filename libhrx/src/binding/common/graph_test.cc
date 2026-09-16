// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/graph.h"

#include <array>
#include <cstdint>
#include <cstring>

#include "common/init_test_util.h"
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
  };
  IREE_EXPECT_OK(iree_hal_streaming_graph_set_kernel_node_params(
      &node, &symbol, &exact_params));
  EXPECT_EQ(exact_arguments.size(), node.attrs.kernel.constants.data_length);
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

class CrossContextCaptureTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(HRX_CALL(hrx_cpu_initialize(/*flags=*/0)));
    cpu_initialized_ = true;
    IREE_ASSERT_OK(HRX_CALL(hrx_cpu_device_get(/*index=*/0, &hrx_device_)));

    device_entry_.hrx_device = hrx_device_;
    device_entry_.hal_device = hrx_device_hal(hrx_device_);
    iree_slim_mutex_initialize(&device_entry_.primary_context_mutex);
    iree_slim_mutex_initialize(&device_entry_.graph_memory_mutex);
    iree_arena_block_pool_initialize(/*block_size=*/64 * 1024,
                                     iree_allocator_system(),
                                     &device_entry_.block_pool);
    device_entry_initialized_ = true;

    device_registry_.host_allocator = iree_allocator_system();
    iree_slim_mutex_initialize(&device_registry_.context_list.mutex);
    iree_notification_initialize(&device_registry_.context_list.changed);
    iree_hal_streaming_set_device_registry_for_testing(&device_registry_);
    device_registry_installed_ = true;

    iree_hal_streaming_context_flags_t context_flags = {};
    context_flags.scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO;
    IREE_ASSERT_OK(iree_hal_streaming_context_create(
        &device_entry_, context_flags, allocators_[0].AsAllocator(),
        &contexts_[0]));
    IREE_ASSERT_OK(iree_hal_streaming_context_create(
        &device_entry_, context_flags, allocators_[1].AsAllocator(),
        &contexts_[1]));
  }

  void TearDown() override {
    allocators_[0].fail_allocations = false;
    allocators_[1].fail_allocations = false;
    if (origin_stream_ && origin_stream_->capture_status !=
                              IREE_HAL_STREAMING_CAPTURE_STATUS_NONE) {
      iree_status_ignore(iree_hal_streaming_end_capture(origin_stream_,
                                                        /*out_graph=*/nullptr));
    }
    if (participant_stream_ && participant_stream_->capture_status !=
                                   IREE_HAL_STREAMING_CAPTURE_STATUS_NONE) {
      iree_status_ignore(iree_hal_streaming_end_capture(participant_stream_,
                                                        /*out_graph=*/nullptr));
    }
    iree_hal_streaming_event_release(second_event_);
    iree_hal_streaming_event_release(event_);
    iree_hal_streaming_stream_release(participant_stream_);
    iree_hal_streaming_stream_release(origin_stream_);
    iree_hal_streaming_graph_release(shared_graph_);
    iree_hal_streaming_context_release(contexts_[1]);
    iree_hal_streaming_context_release(contexts_[0]);
    if (device_registry_installed_) {
      EXPECT_EQ(nullptr, device_registry_.context_list.head);
      EXPECT_EQ(nullptr, device_registry_.context_list.tail);
      iree_hal_streaming_set_device_registry_for_testing(nullptr);
      iree_notification_deinitialize(&device_registry_.context_list.changed);
      iree_slim_mutex_deinitialize(&device_registry_.context_list.mutex);
    }
    if (device_entry_initialized_) {
      iree_arena_block_pool_deinitialize(&device_entry_.block_pool);
      iree_slim_mutex_deinitialize(&device_entry_.graph_memory_mutex);
      iree_slim_mutex_deinitialize(&device_entry_.primary_context_mutex);
    }
    if (cpu_initialized_) {
      IREE_EXPECT_OK(HRX_CALL(hrx_cpu_shutdown()));
    }
  }

  iree_status_t CreateNonBlockingStream(
      iree_hal_streaming_context_t* context, iree_allocator_t host_allocator,
      iree_hal_streaming_stream_t** out_stream) {
    IREE_ASSERT_ARGUMENT(context);
    IREE_ASSERT_ARGUMENT(out_stream);
    *out_stream = nullptr;
    const iree_hal_queue_family_t* queue_family =
        iree_hal_device_queue_family(context->device, /*family_ordinal=*/0);
    if (!queue_family) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "task device has no queue family");
    }
    iree_hal_queue_params_t queue_params;
    iree_hal_queue_params_initialize(&queue_params);
    iree_hal_queue_t* queue = nullptr;
    IREE_RETURN_IF_ERROR(
        iree_hal_queue_acquire(queue_family, &queue_params, &queue));
    iree_status_t status = iree_hal_streaming_stream_create(
        context, queue, IREE_HAL_STREAMING_STREAM_FLAG_NON_BLOCKING,
        /*priority=*/0, host_allocator, out_stream);
    iree_hal_queue_release(queue);
    return status;
  }

  void BeginJoinedCapture() {
    IREE_ASSERT_OK(CreateNonBlockingStream(
        contexts_[0], allocators_[0].AsAllocator(), &origin_stream_));
    IREE_ASSERT_OK(CreateNonBlockingStream(
        contexts_[1], allocators_[1].AsAllocator(), &participant_stream_));
    IREE_ASSERT_OK(iree_hal_streaming_event_create(
        contexts_[0], IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING,
        allocators_[0].AsAllocator(), &event_));
    IREE_ASSERT_OK(iree_hal_streaming_begin_capture(
        origin_stream_, IREE_HAL_STREAMING_CAPTURE_MODE_GLOBAL));
    IREE_ASSERT_OK(iree_hal_streaming_event_record(event_, origin_stream_));
    IREE_ASSERT_OK(iree_hal_streaming_stream_wait_event(
        participant_stream_, event_, /*capture_external_wait=*/false));
  }

  void ExpectExactActiveSession(iree_hal_streaming_graph_t* graph,
                                unsigned long long capture_id) {
    EXPECT_EQ(IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE,
              origin_stream_->capture_status);
    EXPECT_EQ(IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE,
              participant_stream_->capture_status);
    EXPECT_EQ(graph, origin_stream_->capture_graph);
    EXPECT_EQ(graph, participant_stream_->capture_graph);
    EXPECT_EQ(capture_id, origin_stream_->capture_id);
    EXPECT_EQ(capture_id, participant_stream_->capture_id);
  }

  void ExpectCaptureCleared() {
    EXPECT_EQ(IREE_HAL_STREAMING_CAPTURE_STATUS_NONE,
              origin_stream_->capture_status);
    EXPECT_EQ(IREE_HAL_STREAMING_CAPTURE_STATUS_NONE,
              participant_stream_->capture_status);
    EXPECT_EQ(nullptr, origin_stream_->capture_graph);
    EXPECT_EQ(nullptr, participant_stream_->capture_graph);
    EXPECT_EQ(0u, origin_stream_->capture_id);
    EXPECT_EQ(0u, participant_stream_->capture_id);
  }

  std::array<ProbedHostAllocator, 2> allocators_;
  bool cpu_initialized_ = false;
  bool device_entry_initialized_ = false;
  bool device_registry_installed_ = false;
  hrx_device_t hrx_device_ = nullptr;
  iree_hal_streaming_device_t device_entry_ = {};
  iree_hal_streaming_device_registry_t device_registry_ = {};
  std::array<iree_hal_streaming_context_t*, 2> contexts_ = {};
  iree_hal_streaming_stream_t* origin_stream_ = nullptr;
  iree_hal_streaming_stream_t* participant_stream_ = nullptr;
  iree_hal_streaming_event_t* event_ = nullptr;
  iree_hal_streaming_event_t* second_event_ = nullptr;
  iree_hal_streaming_graph_t* shared_graph_ = nullptr;
};

TEST_F(CrossContextCaptureTest, EndsJoinedSessionAndClearsEveryParticipant) {
  BeginJoinedCapture();
  ASSERT_FALSE(HasFatalFailure());
  iree_hal_streaming_graph_t* const graph = origin_stream_->capture_graph;
  ASSERT_NE(nullptr, graph);
  ASSERT_EQ(graph, participant_stream_->capture_graph);

  iree_hal_streaming_graph_t* captured_graph = nullptr;
  IREE_ASSERT_OK(
      iree_hal_streaming_end_capture(origin_stream_, &captured_graph));
  EXPECT_EQ(graph, captured_graph);
  ExpectCaptureCleared();
  iree_hal_streaming_graph_release(captured_graph);
}

TEST_F(CrossContextCaptureTest, RejectsUnjoinedParticipantAndClearsSession) {
  BeginJoinedCapture();
  ASSERT_FALSE(HasFatalFailure());
  iree_hal_streaming_graph_t* const graph = origin_stream_->capture_graph;
  ASSERT_NE(nullptr, graph);

  iree_hal_streaming_graph_node_t* node = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_empty_node(
      graph, participant_stream_->capture_dependencies,
      participant_stream_->capture_dependency_count, &node));
  IREE_ASSERT_OK(
      iree_hal_streaming_capture_set_last_node(participant_stream_, node));

  iree_hal_streaming_graph_t* captured_graph = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ABORTED,
      iree_hal_streaming_end_capture(origin_stream_, &captured_graph));
  EXPECT_EQ(nullptr, captured_graph);
  iree_hal_streaming_graph_release(captured_graph);
  ExpectCaptureCleared();
}

TEST_F(CrossContextCaptureTest, InvalidatesAndClearsEveryParticipant) {
  BeginJoinedCapture();
  ASSERT_FALSE(HasFatalFailure());
  iree_hal_streaming_graph_t* const graph = origin_stream_->capture_graph;
  ASSERT_NE(nullptr, graph);
  const unsigned long long capture_id = origin_stream_->capture_id;
  ASSERT_NE(0u, capture_id);

  bool invalidated = false;
  IREE_ASSERT_OK(iree_hal_streaming_invalidate_capture_graph(graph, capture_id,
                                                             &invalidated));
  EXPECT_TRUE(invalidated);
  EXPECT_EQ(IREE_HAL_STREAMING_CAPTURE_STATUS_INVALIDATED,
            origin_stream_->capture_status);
  EXPECT_EQ(IREE_HAL_STREAMING_CAPTURE_STATUS_INVALIDATED,
            participant_stream_->capture_status);

  iree_hal_streaming_graph_t* captured_graph = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      iree_hal_streaming_end_capture(origin_stream_, &captured_graph));
  EXPECT_EQ(nullptr, captured_graph);
  iree_hal_streaming_graph_release(captured_graph);
  ExpectCaptureCleared();
}

TEST_F(CrossContextCaptureTest,
       InvalidationDoesNotCrossIndependentSameGraphSessions) {
  IREE_ASSERT_OK(CreateNonBlockingStream(
      contexts_[0], allocators_[0].AsAllocator(), &origin_stream_));
  IREE_ASSERT_OK(CreateNonBlockingStream(
      contexts_[1], allocators_[1].AsAllocator(), &participant_stream_));
  IREE_ASSERT_OK(iree_hal_streaming_graph_create(
      contexts_[0], IREE_HAL_STREAMING_GRAPH_FLAG_NONE,
      allocators_[0].AsAllocator(), &shared_graph_));
  IREE_ASSERT_OK(iree_hal_streaming_event_create(
      contexts_[0], IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING,
      allocators_[0].AsAllocator(), &event_));

  IREE_ASSERT_OK(iree_hal_streaming_begin_capture_to_graph(
      origin_stream_, shared_graph_, /*dependencies=*/nullptr,
      /*dependency_count=*/0, IREE_HAL_STREAMING_CAPTURE_MODE_GLOBAL));
  IREE_ASSERT_OK(iree_hal_streaming_begin_capture_to_graph(
      participant_stream_, shared_graph_, /*dependencies=*/nullptr,
      /*dependency_count=*/0, IREE_HAL_STREAMING_CAPTURE_MODE_GLOBAL));
  const unsigned long long origin_capture_id = origin_stream_->capture_id;
  const unsigned long long participant_capture_id =
      participant_stream_->capture_id;
  EXPECT_NE(origin_capture_id, participant_capture_id);

  IREE_ASSERT_OK(iree_hal_streaming_event_record(event_, origin_stream_));
  iree_hal_streaming_event_capture_association_t association;
  iree_hal_streaming_event_acquire_capture_association(event_, &association);
  bool invalidated = false;
  IREE_EXPECT_OK(iree_hal_streaming_invalidate_event_captures(&association, 1,
                                                              &invalidated));
  iree_hal_streaming_event_release_capture_association(&association);
  EXPECT_TRUE(invalidated);
  EXPECT_EQ(IREE_HAL_STREAMING_CAPTURE_STATUS_INVALIDATED,
            origin_stream_->capture_status);
  EXPECT_EQ(IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE,
            participant_stream_->capture_status);
  EXPECT_EQ(shared_graph_, participant_stream_->capture_graph);
  EXPECT_EQ(participant_capture_id, participant_stream_->capture_id);

  iree_hal_streaming_graph_t* captured_graph = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      iree_hal_streaming_end_capture(origin_stream_, &captured_graph));
  EXPECT_EQ(nullptr, captured_graph);
  iree_hal_streaming_graph_release(captured_graph);
  EXPECT_EQ(IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE,
            participant_stream_->capture_status);
  EXPECT_EQ(shared_graph_, participant_stream_->capture_graph);
  EXPECT_EQ(participant_capture_id, participant_stream_->capture_id);
  IREE_EXPECT_OK(iree_hal_streaming_end_capture(participant_stream_,
                                                /*out_graph=*/nullptr));
}

TEST_F(CrossContextCaptureTest, EndDoesNotCrossIndependentSameGraphSessions) {
  IREE_ASSERT_OK(CreateNonBlockingStream(
      contexts_[0], allocators_[0].AsAllocator(), &origin_stream_));
  IREE_ASSERT_OK(CreateNonBlockingStream(
      contexts_[1], allocators_[1].AsAllocator(), &participant_stream_));
  IREE_ASSERT_OK(iree_hal_streaming_graph_create(
      contexts_[0], IREE_HAL_STREAMING_GRAPH_FLAG_NONE,
      allocators_[0].AsAllocator(), &shared_graph_));

  IREE_ASSERT_OK(iree_hal_streaming_begin_capture_to_graph(
      origin_stream_, shared_graph_, /*dependencies=*/nullptr,
      /*dependency_count=*/0, IREE_HAL_STREAMING_CAPTURE_MODE_GLOBAL));
  IREE_ASSERT_OK(iree_hal_streaming_begin_capture_to_graph(
      participant_stream_, shared_graph_, /*dependencies=*/nullptr,
      /*dependency_count=*/0, IREE_HAL_STREAMING_CAPTURE_MODE_GLOBAL));
  const unsigned long long origin_capture_id = origin_stream_->capture_id;
  const unsigned long long participant_capture_id =
      participant_stream_->capture_id;
  EXPECT_NE(origin_capture_id, participant_capture_id);

  // Give the second independent session a frontier that would be unjoined from
  // the first. Session matching must exclude it before reachability or cleanup.
  iree_hal_streaming_graph_node_t* participant_node = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_empty_node(
      shared_graph_, /*dependencies=*/nullptr, /*dependency_count=*/0,
      &participant_node));
  IREE_ASSERT_OK(iree_hal_streaming_capture_set_last_node(participant_stream_,
                                                          participant_node));

  IREE_EXPECT_OK(
      iree_hal_streaming_end_capture(origin_stream_, /*out_graph=*/nullptr));
  EXPECT_EQ(IREE_HAL_STREAMING_CAPTURE_STATUS_NONE,
            origin_stream_->capture_status);
  EXPECT_EQ(IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE,
            participant_stream_->capture_status);
  EXPECT_EQ(shared_graph_, participant_stream_->capture_graph);
  EXPECT_EQ(participant_capture_id, participant_stream_->capture_id);
  IREE_EXPECT_OK(iree_hal_streaming_end_capture(participant_stream_,
                                                /*out_graph=*/nullptr));
}

TEST_F(CrossContextCaptureTest, EventWaitRejectsIndependentSameGraphSession) {
  IREE_ASSERT_OK(CreateNonBlockingStream(
      contexts_[0], allocators_[0].AsAllocator(), &origin_stream_));
  IREE_ASSERT_OK(CreateNonBlockingStream(
      contexts_[1], allocators_[1].AsAllocator(), &participant_stream_));
  IREE_ASSERT_OK(iree_hal_streaming_graph_create(
      contexts_[0], IREE_HAL_STREAMING_GRAPH_FLAG_NONE,
      allocators_[0].AsAllocator(), &shared_graph_));
  IREE_ASSERT_OK(iree_hal_streaming_event_create(
      contexts_[0], IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING,
      allocators_[0].AsAllocator(), &event_));

  IREE_ASSERT_OK(iree_hal_streaming_begin_capture_to_graph(
      origin_stream_, shared_graph_, /*dependencies=*/nullptr,
      /*dependency_count=*/0, IREE_HAL_STREAMING_CAPTURE_MODE_GLOBAL));
  IREE_ASSERT_OK(iree_hal_streaming_begin_capture_to_graph(
      participant_stream_, shared_graph_, /*dependencies=*/nullptr,
      /*dependency_count=*/0, IREE_HAL_STREAMING_CAPTURE_MODE_GLOBAL));
  const unsigned long long origin_capture_id = origin_stream_->capture_id;
  const unsigned long long participant_capture_id =
      participant_stream_->capture_id;
  ASSERT_NE(origin_capture_id, participant_capture_id);

  iree_hal_streaming_graph_node_t* origin_node = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_empty_node(
      shared_graph_, /*dependencies=*/nullptr, /*dependency_count=*/0,
      &origin_node));
  IREE_ASSERT_OK(
      iree_hal_streaming_capture_set_last_node(origin_stream_, origin_node));
  iree_hal_streaming_graph_node_t* participant_node = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_empty_node(
      shared_graph_, /*dependencies=*/nullptr, /*dependency_count=*/0,
      &participant_node));
  IREE_ASSERT_OK(iree_hal_streaming_capture_set_last_node(participant_stream_,
                                                          participant_node));

  IREE_ASSERT_OK(iree_hal_streaming_event_record(event_, origin_stream_));
  iree_status_t status = iree_hal_streaming_stream_wait_event(
      participant_stream_, event_, /*capture_external_wait=*/false);
  EXPECT_EQ(IREE_STATUS_INVALID_ARGUMENT, iree_status_code(status));
  iree_status_ignore(status);

  EXPECT_EQ(IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE,
            origin_stream_->capture_status);
  EXPECT_EQ(shared_graph_, origin_stream_->capture_graph);
  EXPECT_EQ(origin_capture_id, origin_stream_->capture_id);
  EXPECT_EQ(IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE,
            participant_stream_->capture_status);
  EXPECT_EQ(shared_graph_, participant_stream_->capture_graph);
  EXPECT_EQ(participant_capture_id, participant_stream_->capture_id);
  ASSERT_EQ(1u, participant_stream_->capture_dependency_count);
  EXPECT_EQ(participant_node, participant_stream_->capture_dependencies[0]);

  IREE_EXPECT_OK(
      iree_hal_streaming_end_capture(origin_stream_, /*out_graph=*/nullptr));
  IREE_EXPECT_OK(iree_hal_streaming_end_capture(participant_stream_,
                                                /*out_graph=*/nullptr));
}

TEST_F(CrossContextCaptureTest,
       InvalidationSnapshotFailureLeavesExactSessionRetryable) {
  BeginJoinedCapture();
  ASSERT_FALSE(HasFatalFailure());
  iree_hal_streaming_graph_t* const graph = origin_stream_->capture_graph;
  ASSERT_NE(nullptr, graph);
  const unsigned long long capture_id = origin_stream_->capture_id;
  ASSERT_NE(0u, capture_id);

  const int origin_success_count = allocators_[0].successful_allocation_count;
  const int origin_free_count = allocators_[0].free_count;
  const int participant_attempt_count = allocators_[1].allocation_attempt_count;
  allocators_[1].fail_allocations = true;
  bool invalidated = true;
  iree_status_t status = iree_hal_streaming_invalidate_capture_graph(
      graph, capture_id, &invalidated);
  allocators_[1].fail_allocations = false;

  EXPECT_EQ(IREE_STATUS_RESOURCE_EXHAUSTED, iree_status_code(status));
  iree_status_ignore(status);
  EXPECT_FALSE(invalidated);
  EXPECT_EQ(origin_success_count + 1,
            allocators_[0].successful_allocation_count);
  EXPECT_EQ(origin_free_count + 1, allocators_[0].free_count);
  EXPECT_EQ(participant_attempt_count + 1,
            allocators_[1].allocation_attempt_count);
  ExpectExactActiveSession(graph, capture_id);

  IREE_ASSERT_OK(iree_hal_streaming_invalidate_capture_graph(graph, capture_id,
                                                             &invalidated));
  EXPECT_TRUE(invalidated);
  iree_hal_streaming_graph_t* captured_graph = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      iree_hal_streaming_end_capture(origin_stream_, &captured_graph));
  EXPECT_EQ(nullptr, captured_graph);
  iree_hal_streaming_graph_release(captured_graph);
  ExpectCaptureCleared();
}

TEST_F(CrossContextCaptureTest, EndSnapshotFailureLeavesExactSessionRetryable) {
  BeginJoinedCapture();
  ASSERT_FALSE(HasFatalFailure());
  iree_hal_streaming_graph_t* const graph = origin_stream_->capture_graph;
  ASSERT_NE(nullptr, graph);
  const unsigned long long capture_id = origin_stream_->capture_id;
  ASSERT_NE(0u, capture_id);

  const int origin_success_count = allocators_[0].successful_allocation_count;
  const int origin_free_count = allocators_[0].free_count;
  const int participant_attempt_count = allocators_[1].allocation_attempt_count;
  allocators_[1].fail_allocations = true;
  iree_hal_streaming_graph_t* failure_graph = nullptr;
  iree_status_t status =
      iree_hal_streaming_end_capture(origin_stream_, &failure_graph);
  allocators_[1].fail_allocations = false;

  EXPECT_EQ(IREE_STATUS_RESOURCE_EXHAUSTED, iree_status_code(status));
  iree_status_ignore(status);
  EXPECT_EQ(nullptr, failure_graph);
  iree_hal_streaming_graph_release(failure_graph);
  EXPECT_EQ(origin_success_count + 1,
            allocators_[0].successful_allocation_count);
  EXPECT_EQ(origin_free_count + 1, allocators_[0].free_count);
  EXPECT_EQ(participant_attempt_count + 1,
            allocators_[1].allocation_attempt_count);
  ExpectExactActiveSession(graph, capture_id);

  iree_hal_streaming_graph_t* captured_graph = nullptr;
  IREE_ASSERT_OK(
      iree_hal_streaming_end_capture(origin_stream_, &captured_graph));
  EXPECT_EQ(graph, captured_graph);
  ExpectCaptureCleared();
  iree_hal_streaming_graph_release(captured_graph);
}

TEST_F(CrossContextCaptureTest,
       StaleEventDoesNotTouchSameStreamGraphSuccessor) {
  IREE_ASSERT_OK(CreateNonBlockingStream(
      contexts_[0], allocators_[0].AsAllocator(), &origin_stream_));
  IREE_ASSERT_OK(CreateNonBlockingStream(
      contexts_[1], allocators_[1].AsAllocator(), &participant_stream_));
  IREE_ASSERT_OK(iree_hal_streaming_event_create(
      contexts_[0], IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING,
      allocators_[0].AsAllocator(), &event_));

  IREE_ASSERT_OK(iree_hal_streaming_begin_capture(
      origin_stream_, IREE_HAL_STREAMING_CAPTURE_MODE_GLOBAL));
  const unsigned long long first_capture_id = origin_stream_->capture_id;
  IREE_ASSERT_OK(iree_hal_streaming_event_record(event_, origin_stream_));
  IREE_ASSERT_OK(
      iree_hal_streaming_end_capture(origin_stream_, &shared_graph_));
  ASSERT_NE(nullptr, shared_graph_);

  IREE_ASSERT_OK(iree_hal_streaming_begin_capture_to_graph(
      origin_stream_, shared_graph_, /*dependencies=*/nullptr,
      /*dependency_count=*/0, IREE_HAL_STREAMING_CAPTURE_MODE_GLOBAL));
  const unsigned long long successor_capture_id = origin_stream_->capture_id;
  ASSERT_NE(first_capture_id, successor_capture_id);

  iree_hal_streaming_event_capture_association_t association;
  iree_hal_streaming_event_acquire_capture_association(event_, &association);
  EXPECT_EQ(shared_graph_, association.graph);
  EXPECT_EQ(origin_stream_, association.recording_stream);
  EXPECT_EQ(first_capture_id, association.capture_id);
  bool invalidated = true;
  IREE_EXPECT_OK(iree_hal_streaming_invalidate_event_captures(&association, 1,
                                                              &invalidated));
  EXPECT_FALSE(invalidated);
  iree_hal_streaming_event_release_capture_association(&association);

  iree_status_t wait_status = iree_hal_streaming_stream_wait_event(
      participant_stream_, event_, /*capture_external_wait=*/false);
  EXPECT_EQ(IREE_STATUS_INVALID_ARGUMENT, iree_status_code(wait_status));
  iree_status_ignore(wait_status);
  EXPECT_EQ(IREE_HAL_STREAMING_CAPTURE_STATUS_NONE,
            participant_stream_->capture_status);
  EXPECT_EQ(IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE,
            origin_stream_->capture_status);
  EXPECT_EQ(shared_graph_, origin_stream_->capture_graph);
  EXPECT_EQ(successor_capture_id, origin_stream_->capture_id);

  IREE_EXPECT_OK(
      iree_hal_streaming_end_capture(origin_stream_, /*out_graph=*/nullptr));
}

TEST_F(CrossContextCaptureTest,
       StaleEventDoesNotTouchDifferentContextSuccessor) {
  IREE_ASSERT_OK(CreateNonBlockingStream(
      contexts_[0], allocators_[0].AsAllocator(), &origin_stream_));
  IREE_ASSERT_OK(CreateNonBlockingStream(
      contexts_[1], allocators_[1].AsAllocator(), &participant_stream_));
  IREE_ASSERT_OK(iree_hal_streaming_event_create(
      contexts_[0], IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING,
      allocators_[0].AsAllocator(), &event_));

  IREE_ASSERT_OK(iree_hal_streaming_begin_capture(
      origin_stream_, IREE_HAL_STREAMING_CAPTURE_MODE_GLOBAL));
  const unsigned long long first_capture_id = origin_stream_->capture_id;
  IREE_ASSERT_OK(iree_hal_streaming_event_record(event_, origin_stream_));
  IREE_ASSERT_OK(
      iree_hal_streaming_end_capture(origin_stream_, &shared_graph_));
  ASSERT_NE(nullptr, shared_graph_);

  IREE_ASSERT_OK(iree_hal_streaming_begin_capture_to_graph(
      participant_stream_, shared_graph_, /*dependencies=*/nullptr,
      /*dependency_count=*/0, IREE_HAL_STREAMING_CAPTURE_MODE_GLOBAL));
  ASSERT_NE(first_capture_id, participant_stream_->capture_id);

  iree_hal_streaming_event_capture_association_t association;
  iree_hal_streaming_event_acquire_capture_association(event_, &association);
  bool invalidated = true;
  IREE_EXPECT_OK(iree_hal_streaming_invalidate_event_captures(&association, 1,
                                                              &invalidated));
  EXPECT_FALSE(invalidated);
  iree_hal_streaming_event_release_capture_association(&association);

  iree_status_t wait_status = iree_hal_streaming_stream_wait_event(
      origin_stream_, event_, /*capture_external_wait=*/false);
  EXPECT_EQ(IREE_STATUS_INVALID_ARGUMENT, iree_status_code(wait_status));
  iree_status_ignore(wait_status);
  EXPECT_EQ(IREE_HAL_STREAMING_CAPTURE_STATUS_NONE,
            origin_stream_->capture_status);
  EXPECT_EQ(IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE,
            participant_stream_->capture_status);
  EXPECT_EQ(shared_graph_, participant_stream_->capture_graph);
  EXPECT_NE(first_capture_id, participant_stream_->capture_id);

  IREE_EXPECT_OK(iree_hal_streaming_end_capture(participant_stream_,
                                                /*out_graph=*/nullptr));
}

TEST_F(CrossContextCaptureTest,
       MultiEventInvalidationAllocationFailureChangesNeitherSession) {
  IREE_ASSERT_OK(CreateNonBlockingStream(
      contexts_[0], allocators_[0].AsAllocator(), &origin_stream_));
  IREE_ASSERT_OK(CreateNonBlockingStream(
      contexts_[1], allocators_[1].AsAllocator(), &participant_stream_));
  IREE_ASSERT_OK(iree_hal_streaming_event_create(
      contexts_[0], IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING,
      allocators_[0].AsAllocator(), &event_));
  IREE_ASSERT_OK(iree_hal_streaming_event_create(
      contexts_[1], IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING,
      allocators_[1].AsAllocator(), &second_event_));

  IREE_ASSERT_OK(iree_hal_streaming_begin_capture(
      origin_stream_, IREE_HAL_STREAMING_CAPTURE_MODE_GLOBAL));
  IREE_ASSERT_OK(iree_hal_streaming_begin_capture(
      participant_stream_, IREE_HAL_STREAMING_CAPTURE_MODE_GLOBAL));
  IREE_ASSERT_OK(iree_hal_streaming_event_record(event_, origin_stream_));
  IREE_ASSERT_OK(
      iree_hal_streaming_event_record(second_event_, participant_stream_));

  ASSERT_NE(origin_stream_->capture_id, participant_stream_->capture_id);

  iree_hal_streaming_event_capture_association_t associations[2];
  iree_hal_streaming_event_acquire_capture_association(event_,
                                                       &associations[0]);
  iree_hal_streaming_event_acquire_capture_association(second_event_,
                                                       &associations[1]);
  EXPECT_NE(associations[0].capture_id, associations[1].capture_id);

  allocators_[1].fail_allocations = true;
  bool invalidated = true;
  iree_status_t status = iree_hal_streaming_invalidate_event_captures(
      associations, IREE_ARRAYSIZE(associations), &invalidated);
  allocators_[1].fail_allocations = false;
  EXPECT_EQ(IREE_STATUS_RESOURCE_EXHAUSTED, iree_status_code(status));
  iree_status_ignore(status);
  EXPECT_FALSE(invalidated);
  EXPECT_EQ(IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE,
            origin_stream_->capture_status);
  EXPECT_EQ(IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE,
            participant_stream_->capture_status);

  IREE_EXPECT_OK(iree_hal_streaming_invalidate_event_captures(
      associations, IREE_ARRAYSIZE(associations), &invalidated));
  EXPECT_TRUE(invalidated);
  EXPECT_EQ(IREE_HAL_STREAMING_CAPTURE_STATUS_INVALIDATED,
            origin_stream_->capture_status);
  EXPECT_EQ(IREE_HAL_STREAMING_CAPTURE_STATUS_INVALIDATED,
            participant_stream_->capture_status);
  iree_hal_streaming_event_release_capture_association(&associations[1]);
  iree_hal_streaming_event_release_capture_association(&associations[0]);

  iree_hal_streaming_graph_t* origin_graph = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      iree_hal_streaming_end_capture(origin_stream_, &origin_graph));
  EXPECT_EQ(nullptr, origin_graph);
  iree_hal_streaming_graph_release(origin_graph);
  iree_hal_streaming_graph_t* participant_graph = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      iree_hal_streaming_end_capture(participant_stream_, &participant_graph));
  EXPECT_EQ(nullptr, participant_graph);
  iree_hal_streaming_graph_release(participant_graph);
}

}  // namespace
