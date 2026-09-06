// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <vector>

#include "hrx_internal.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class TestNode {
 public:
  TestNode(uint32_t node_index, std::vector<hrx_graph_node_s*> dependencies)
      : storage_(new uint8_t[sizeof(hrx_graph_node_s) +
                             dependencies.size() * sizeof(hrx_graph_node_s*)]) {
    memset(storage_.get(), 0,
           sizeof(hrx_graph_node_s) +
               dependencies.size() * sizeof(hrx_graph_node_s*));
    node_ = reinterpret_cast<hrx_graph_node_s*>(storage_.get());
    node_->type = HRX_GRAPH_NODE_TYPE_INTERNAL_KERNEL;
    node_->node_index = node_index;
    node_->dependency_count = dependencies.size();
    for (size_t i = 0; i < dependencies.size(); ++i) {
      node_->dependencies[i] = dependencies[i];
    }
  }

  hrx_graph_node_s* get() const { return node_; }

 private:
  std::unique_ptr<uint8_t[]> storage_;
  hrx_graph_node_s* node_ = nullptr;
};

typedef struct BarrierSpyCommandBuffer {
  iree_hal_command_buffer_t base;
  iree_status_code_t failure_code;
  uint32_t barrier_count;
  iree_hal_execution_stage_t source_stage;
  iree_hal_execution_stage_t target_stage;
  iree_hal_execution_barrier_flags_t flags;
  iree_host_size_t memory_barrier_count;
  iree_hal_memory_barrier_t memory_barrier;
  iree_host_size_t buffer_barrier_count;
} BarrierSpyCommandBuffer;

static void BarrierSpyDestroy(iree_hal_command_buffer_t* base_command_buffer) {}

static iree_status_t BarrierSpyExecutionBarrier(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage,
    iree_hal_execution_stage_t target_stage,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  auto* command_buffer =
      reinterpret_cast<BarrierSpyCommandBuffer*>(base_command_buffer);
  if (command_buffer->failure_code != IREE_STATUS_OK) {
    return iree_make_status(command_buffer->failure_code);
  }
  ++command_buffer->barrier_count;
  command_buffer->source_stage = source_stage;
  command_buffer->target_stage = target_stage;
  command_buffer->flags = flags;
  command_buffer->memory_barrier_count = memory_barrier_count;
  if (memory_barrier_count > 0) {
    command_buffer->memory_barrier = memory_barriers[0];
  }
  command_buffer->buffer_barrier_count = buffer_barrier_count;
  return iree_ok_status();
}

static const iree_hal_command_buffer_vtable_t kBarrierSpyVtable = {
    /*.destroy=*/BarrierSpyDestroy,
    /*.begin=*/nullptr,
    /*.end=*/nullptr,
    /*.begin_debug_group=*/nullptr,
    /*.end_debug_group=*/nullptr,
    /*.execution_barrier=*/BarrierSpyExecutionBarrier,
};

class GraphBarrierTest : public ::testing::Test {
 protected:
  void SetUp() override {
    memset(&command_buffer_, 0, sizeof(command_buffer_));
    iree_hal_command_buffer_initialize(
        /*device_allocator=*/nullptr, IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED,
        IREE_HAL_COMMAND_CATEGORY_ANY, IREE_HAL_QUEUE_AFFINITY_ANY,
        /*binding_capacity=*/0, /*validation_state=*/nullptr,
        &kBarrierSpyVtable, &command_buffer_.base);
    hrx_graph_barrier_state_reset(&state_);
  }

  void TearDown() override {
    iree_hal_command_buffer_release(&command_buffer_.base);
  }

  bool Record(TestNode& node,
              const hrx_graph_edge_t* additional_edges = nullptr) {
    bool did_barrier = false;
    IREE_EXPECT_OK(hrx_graph_record_node_barrier(
        &state_, node.get(), additional_edges, node_index_map_,
        node.get()->node_index, &command_buffer_.base, &did_barrier));
    return did_barrier;
  }

  uint32_t node_index_map_[32] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                  11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                                  22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
  hrx_graph_barrier_state_t state_;
  BarrierSpyCommandBuffer command_buffer_;
};

TEST_F(GraphBarrierTest, SingleAndIndependentNodesDoNotBarrier) {
  TestNode a(0, {});
  TestNode b(1, {});
  EXPECT_FALSE(Record(a));
  EXPECT_FALSE(Record(b));
  EXPECT_EQ(command_buffer_.barrier_count, 0u);
}

TEST_F(GraphBarrierTest, LinearChainBarriersBeforeEachConsumer) {
  TestNode a(0, {});
  TestNode b(1, {a.get()});
  TestNode c(2, {b.get()});
  EXPECT_FALSE(Record(a));
  EXPECT_TRUE(Record(b));
  EXPECT_TRUE(Record(c));
  EXPECT_EQ(command_buffer_.barrier_count, 2u);
}

TEST_F(GraphBarrierTest, FanInEmitsExactlyOneBarrier) {
  TestNode a(0, {});
  TestNode b(1, {});
  TestNode c(2, {a.get(), b.get()});
  EXPECT_FALSE(Record(a));
  EXPECT_FALSE(Record(b));
  EXPECT_TRUE(Record(c));
  EXPECT_EQ(command_buffer_.barrier_count, 1u);
}

TEST_F(GraphBarrierTest, EarlierBarrierCoversOlderProducer) {
  TestNode a(0, {});
  TestNode b(1, {a.get()});
  TestNode c(2, {a.get()});
  EXPECT_FALSE(Record(a));
  EXPECT_TRUE(Record(b));
  EXPECT_FALSE(Record(c));
  EXPECT_EQ(command_buffer_.barrier_count, 1u);
}

TEST_F(GraphBarrierTest, OverflowForcesConservativeDependencyBarrier) {
  std::vector<std::unique_ptr<TestNode>> nodes;
  for (uint32_t i = 0; i < 9; ++i) {
    nodes.push_back(
        std::make_unique<TestNode>(i, std::vector<hrx_graph_node_s*>{}));
    EXPECT_FALSE(Record(*nodes.back()));
  }
  TestNode consumer(9, {nodes[0]->get()});
  EXPECT_TRUE(Record(consumer));
  EXPECT_EQ(command_buffer_.barrier_count, 1u);
}

TEST_F(GraphBarrierTest, AdditionalDependencyEdgeBarriersLikeInlineEdge) {
  TestNode a(0, {});
  TestNode b(1, {});
  hrx_graph_edge_t edge = {/*.next=*/nullptr, /*.from=*/a.get(),
                           /*.to=*/b.get()};
  EXPECT_FALSE(Record(a, &edge));
  EXPECT_TRUE(Record(b, &edge));
  EXPECT_EQ(command_buffer_.barrier_count, 1u);
}

TEST_F(GraphBarrierTest, BarrierUsesRetireIssueAndAllMemoryScopes) {
  TestNode a(0, {});
  TestNode b(1, {a.get()});
  EXPECT_FALSE(Record(a));
  EXPECT_TRUE(Record(b));
  EXPECT_EQ(command_buffer_.source_stage,
            IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE);
  EXPECT_EQ(command_buffer_.target_stage,
            IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE);
  EXPECT_EQ(command_buffer_.flags, IREE_HAL_EXECUTION_BARRIER_FLAG_NONE);
  EXPECT_EQ(command_buffer_.memory_barrier_count, 1u);
  EXPECT_EQ(command_buffer_.memory_barrier.source_scope,
            IREE_HAL_MEMORY_ACCESS_ALL);
  EXPECT_EQ(command_buffer_.memory_barrier.target_scope,
            IREE_HAL_MEMORY_ACCESS_ALL);
  EXPECT_EQ(command_buffer_.buffer_barrier_count, 0u);
}

TEST_F(GraphBarrierTest, BarrierFailureIsPropagatedAndStateIsPreserved) {
  TestNode a(0, {});
  TestNode b(1, {a.get()});
  EXPECT_FALSE(Record(a));
  command_buffer_.failure_code = IREE_STATUS_INTERNAL;
  bool did_barrier = false;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INTERNAL,
      hrx_graph_record_node_barrier(&state_, b.get(), nullptr, node_index_map_,
                                    1, &command_buffer_.base, &did_barrier));
  EXPECT_FALSE(did_barrier);
  command_buffer_.failure_code = IREE_STATUS_OK;
  EXPECT_TRUE(Record(b));
}

TEST(GraphScheduleTest, AdditionalDependencyStaysOnProducerWorkstream) {
  constexpr uint32_t kNodeCount = 17;
  std::vector<std::unique_ptr<TestNode>> nodes;
  nodes.reserve(kNodeCount);
  for (uint32_t i = 0; i < kNodeCount; ++i) {
    nodes.push_back(
        std::make_unique<TestNode>(i, std::vector<hrx_graph_node_s*>{}));
  }

  std::unique_ptr<uint8_t[]> block_storage(
      new uint8_t[sizeof(hrx_graph_node_block_t) +
                  kNodeCount * sizeof(hrx_graph_node_s*)]);
  memset(
      block_storage.get(), 0,
      sizeof(hrx_graph_node_block_t) + kNodeCount * sizeof(hrx_graph_node_s*));
  auto* block = reinterpret_cast<hrx_graph_node_block_t*>(block_storage.get());
  block->capacity = kNodeCount;
  block->count = kNodeCount;
  for (uint32_t i = 0; i < kNodeCount; ++i) {
    block->nodes[i] = nodes[i]->get();
  }

  hrx_graph_edge_t edge = {/*.next=*/nullptr, /*.from=*/nodes[0]->get(),
                           /*.to=*/nodes[16]->get()};
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);

  hrx_graph_schedule_t schedule;
  IREE_ASSERT_OK(
      hrx_graph_schedule_nodes(block, kNodeCount, &edge, &arena, &schedule));
  const hrx_graph_sort_node_t& producer =
      schedule.sorted_nodes[schedule.node_index_map[0]];
  const hrx_graph_sort_node_t& consumer =
      schedule.sorted_nodes[schedule.node_index_map[16]];
  EXPECT_EQ(consumer.partition_id, producer.partition_id);
  EXPECT_EQ(consumer.stream_id, producer.stream_id);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

class GraphExecSemaphoreTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_cpu_initialize(/*flags=*/0)));
    IREE_ASSERT_OK(
        hrx_status_to_iree(hrx_cpu_device_get(/*index=*/0, &device_)));
  }

  void TearDown() override {
    IREE_EXPECT_OK(hrx_status_to_iree(hrx_cpu_shutdown()));
  }

  hrx_device_t device_ = nullptr;
};

// Instantiation creates the semaphores that join consecutive partitions and
// hands ownership of them to the executable resource set. Destroying the
// executable must therefore drop every reference it took on them. Needing more
// than one partition to have any internal semaphore at all makes this the only
// coverage of a block whose payload carries trailing arrays, so it is also what
// exercises the block payload region layout.
TEST_F(GraphExecSemaphoreTest,
       InstantiationLeavesSemaphoresOwnedOnlyByExecutable) {
  hrx_graph_t graph = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_create(device_, 0, &graph)));

  // Host calls are not recordable so each one forms its own partition; a chain
  // of three yields two internal semaphores.
  const hrx_graph_host_call_node_attrs_t attrs = {
      /*.fn=*/[](void* user_data) -> hrx_status_t { return hrx_ok_status(); },
      /*.user_data=*/nullptr,
  };
  hrx_graph_node_t previous_node = nullptr;
  for (int i = 0; i < 3; ++i) {
    hrx_graph_node_t node = nullptr;
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_add_host_call_node(
        graph, previous_node ? &previous_node : nullptr, previous_node ? 1 : 0,
        &attrs, &node)));
    previous_node = node;
  }

  hrx_graph_exec_t exec = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_instantiate(graph, 0, &exec)));
  ASSERT_EQ(exec->semaphore_count, 2u);

  // Take an independent reference on each semaphore so the reference counts
  // stay observable once the executable is gone.
  std::vector<iree_hal_semaphore_t*> semaphores;
  for (uint32_t i = 0; i < exec->semaphore_count; ++i) {
    ASSERT_NE(exec->semaphores[i], nullptr);
    EXPECT_EQ(exec->semaphore_base_values[i], 0u);
    iree_hal_semaphore_retain(exec->semaphores[i]);
    semaphores.push_back(exec->semaphores[i]);
  }

  hrx_graph_exec_release(exec);
  hrx_graph_release(graph);

  for (iree_hal_semaphore_t* semaphore : semaphores) {
    // Every HAL object begins with an iree_hal_resource_t, which is what makes
    // its reference count reachable from an opaque handle.
    EXPECT_EQ(
        iree_atomic_ref_count_load(
            &reinterpret_cast<iree_hal_resource_t*>(semaphore)->ref_count),
        1);
    iree_hal_semaphore_release(semaphore);
  }
}

}  // namespace
