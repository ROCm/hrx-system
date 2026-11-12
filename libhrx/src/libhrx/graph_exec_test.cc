// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <memory>
#include <semaphore>
#include <thread>
#include <vector>

#include "hrx_internal.h"
#include "hrx_runtime_cxx.h"
#include "iree/hal/driver/task/executable/executable.h"
#include "iree/hal/driver/task/executable/library/abi.h"
#include "iree/hal/testing/mock_device.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static constexpr iree_hal_queue_priority_t kQueuePriority =
    IREE_HAL_QUEUE_PRIORITY_NORMAL;
static const iree_hal_queue_family_spec_t kQueueFamilySpec = [] {
  iree_hal_queue_family_spec_t spec = {
      .name = IREE_SV("test"),
      .priority_count = 1,
      .priorities = &kQueuePriority,
      .physical_device_affinity = 1,
      .role_flags = IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH |
                    IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER |
                    IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_ATOMIC,
  };
  return spec;
}();

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
  iree_hal_barrier_flags_t flags;
  iree_host_size_t memory_barrier_count;
  iree_hal_memory_barrier_t memory_barrier;
  iree_host_size_t buffer_barrier_count;
} BarrierSpyCommandBuffer;

static void BarrierSpyDestroy(iree_hal_command_buffer_t* base_command_buffer) {}

static iree_status_t BarrierSpyExecutionBarrier(
    iree_hal_command_buffer_t* base_command_buffer,
    const iree_hal_barrier_t* barrier) {
  auto* command_buffer =
      reinterpret_cast<BarrierSpyCommandBuffer*>(base_command_buffer);
  if (command_buffer->failure_code != IREE_STATUS_OK) {
    return iree_make_status(command_buffer->failure_code);
  }
  ++command_buffer->barrier_count;
  command_buffer->source_stage = barrier->source_stage_mask;
  command_buffer->target_stage = barrier->target_stage_mask;
  command_buffer->flags = barrier->flags;
  command_buffer->memory_barrier_count = barrier->memory_barrier_count;
  if (barrier->memory_barrier_count > 0) {
    command_buffer->memory_barrier = barrier->memory_barriers[0];
  }
  command_buffer->buffer_barrier_count = barrier->buffer_barrier_count;
  return iree_ok_status();
}

static const iree_hal_command_buffer_vtable_t kBarrierSpyVtable = {
    .destroy = BarrierSpyDestroy,
    .begin = nullptr,
    .end = nullptr,
    .begin_debug_group = nullptr,
    .end_debug_group = nullptr,
    .barrier = BarrierSpyExecutionBarrier,
};

class GraphBarrierTest : public ::testing::Test {
 protected:
  void SetUp() override {
    memset(&command_buffer_, 0, sizeof(command_buffer_));
    iree_hal_device_queue_spec_t queues = {
        .family_count = 1,
        .families = &kQueueFamilySpec,
    };
    iree_hal_device_spec_params_t spec_params = {.queues = &queues};
    iree_hal_device_spec_t* device_spec = nullptr;
    IREE_ASSERT_OK(iree_hal_device_spec_create(
        &spec_params, iree_allocator_system(), &device_spec));
    iree_hal_mock_device_options_t device_options;
    iree_hal_mock_device_options_initialize(&device_options);
    device_options.device_spec = device_spec;
    iree_status_t status = iree_hal_mock_device_create(
        &device_options, iree_allocator_system(), &device_);
    iree_hal_device_spec_release(device_spec);
    IREE_ASSERT_OK(status);
    queue_family_ = iree_hal_device_queue_family(device_, 0);
    iree_hal_command_buffer_initialize(
        queue_family_, IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED,
        IREE_HAL_COMMAND_CATEGORY_ANY, /*binding_capacity=*/0,
        /*validation_state=*/nullptr, &kBarrierSpyVtable,
        &command_buffer_.base);
    hrx_graph_barrier_state_reset(&state_);
  }

  void TearDown() override {
    iree_hal_command_buffer_release(&command_buffer_.base);
    iree_hal_device_release(device_);
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
  // Canonical family borrowed from device_.
  const iree_hal_queue_family_t* queue_family_ = nullptr;
  // Device owning the canonical family used by this fixture.
  iree_hal_device_t* device_ = nullptr;
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
  hrx_graph_edge_t edge = {.next = nullptr, .from = a.get(), .to = b.get()};
  EXPECT_FALSE(Record(a, &edge));
  EXPECT_TRUE(Record(b, &edge));
  EXPECT_EQ(command_buffer_.barrier_count, 1u);
}

TEST_F(GraphBarrierTest, InternalBarrierUsesDeviceMemoryScopes) {
  TestNode a(0, {});
  TestNode b(1, {a.get()});
  EXPECT_FALSE(Record(a));
  EXPECT_TRUE(Record(b));
  EXPECT_EQ(command_buffer_.source_stage,
            IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE);
  EXPECT_EQ(command_buffer_.target_stage,
            IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE);
  EXPECT_EQ(command_buffer_.flags, IREE_HAL_BARRIER_FLAG_NONE);
  EXPECT_EQ(command_buffer_.memory_barrier_count, 1u);
  const iree_hal_access_scope_t device_access =
      IREE_HAL_ACCESS_SCOPE_CONSTANT_READ |
      IREE_HAL_ACCESS_SCOPE_DISPATCH_READ |
      IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE |
      IREE_HAL_ACCESS_SCOPE_TRANSFER_READ |
      IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE;
  EXPECT_EQ(command_buffer_.memory_barrier.source_scope, device_access);
  EXPECT_EQ(command_buffer_.memory_barrier.target_scope, device_access);
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

  hrx_graph_edge_t edge = {
      .next = nullptr,
      .from = nodes[0]->get(),
      .to = nodes[16]->get(),
  };
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

// Pauses a real allocator dependency without putting a hook in the executor.
struct AllocationGate {
  // Notifies the test that replacement preparation has reached allocation.
  std::binary_semaphore entered{0};
  // Releases that allocation after an old-recording launch has completed.
  std::binary_semaphore resume{0};
};

// Controls only this thread's allocations through the runtime host allocator.
// Task workers and concurrent launches continue using the underlying allocator.
class AllocationScope {
 public:
  explicit AllocationScope(size_t fail_index = SIZE_MAX,
                           AllocationGate* gate = nullptr)
      : fail_index_(fail_index), gate_(gate) {
    current_ = this;
  }
  ~AllocationScope() { current_ = nullptr; }
  AllocationScope(const AllocationScope&) = delete;
  AllocationScope& operator=(const AllocationScope&) = delete;

  static hrx_host_allocator_t HostAllocator() {
    const iree_allocator_t allocator = {nullptr, Control};
    hrx_host_allocator_t result;
    memcpy(&result, &allocator, sizeof(result));
    return result;
  }

 private:
  static iree_status_t Control(void*, iree_allocator_command_t command,
                               const void* parameters, void** pointer) {
    if (current_ && command != IREE_ALLOCATOR_COMMAND_FREE) {
      const size_t index = current_->allocation_count_++;
      if (index == 0 && current_->gate_) {
        current_->gate_->entered.release();
        current_->gate_->resume.acquire();
      }
      if (index == current_->fail_index_) {
        return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
      }
    }
    const auto system = iree_allocator_system();
    return system.ctl(system.self, command, parameters, pointer);
  }

  // Active test scope on this thread; null for ordinary runtime allocations.
  static thread_local AllocationScope* current_;
  // Allocation ordinal within this scope.
  size_t allocation_count_ = 0;
  // Ordinal to reject, or SIZE_MAX to keep normal behavior.
  size_t fail_index_;
  // Borrowed gate for the first allocation, or null for no pause.
  AllocationGate* gate_;
};
thread_local AllocationScope* AllocationScope::current_ = nullptr;

using Graph =
    hrx::runtime::hrx_ptr<hrx_graph_t, hrx_graph_retain, hrx_graph_release>;
using Semaphore = hrx::runtime::hrx_ptr<hrx_semaphore_t, hrx_semaphore_retain,
                                        hrx_semaphore_release>;

class GraphExecSemaphoreTest : public ::testing::Test {
 protected:
  // Owns the task device and its runtime through fixture cleanup.
  hrx_device_group_t group_ = nullptr;

  void SetUp() override {
    hrx_runtime_t runtime = nullptr;
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_runtime_create(
        nullptr, AllocationScope::HostAllocator(), &runtime)));
    const hrx_device_create_params_t params = {.uri = "task://"};
    hrx_status_t status = hrx_device_group_create(runtime, 1, &params, &group_);
    hrx_runtime_release(runtime);
    IREE_ASSERT_OK(hrx_status_to_iree(status));
    device_ = hrx_device_group_get_device(group_, 0);
  }

  void TearDown() override {
    hrx_device_group_release(group_);
    group_ = nullptr;
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
      .fn = [](void* user_data) -> hrx_status_t { return hrx_ok_status(); },
      .user_data = nullptr,
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
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_instantiate(
      graph, hrx_device_get_queue(device_, 0, 0), 0, &exec)));
  ASSERT_EQ(exec->recording.semaphore_count, 2u);

  // Take an independent reference on each semaphore so the reference counts
  // stay observable once the executable is gone.
  std::vector<iree_hal_semaphore_t*> semaphores;
  for (uint32_t i = 0; i < exec->recording.semaphore_count; ++i) {
    ASSERT_NE(exec->recording.semaphores[i], nullptr);
    iree_hal_semaphore_retain(exec->recording.semaphores[i]);
    semaphores.push_back(exec->recording.semaphores[i]);
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

// A normal task executable may receive pointers in its constant bytes without
// changing to a custom argument ABI or declaring any buffer bindings.
struct KernelArguments {
  // Fixture-owned result storage, valid until stream completion.
  uint32_t* result;
  // Value the dispatch writes to |result|.
  uint32_t value;
};

static int WriteResult(
    const iree_hal_executable_environment_v0_t* environment,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state) {
  KernelArguments arguments;
  memcpy(&arguments, dispatch_state->constants.data, sizeof(arguments));
  *arguments.result = arguments.value;
  return 0;
}

struct TestTaskExecutable {
  // Ordinary Task executable consumed by the real command buffer.
  iree_hal_task_executable_t base = {};
  // Observes destruction even when no HRX template or executable remains.
  std::atomic<bool> destroyed{false};
  // Optional fixture callback exercising resource-release reentrancy.
  void (*on_destroy)(void*) = nullptr;
  // Borrowed state passed to |on_destroy|.
  void* destroy_data = nullptr;
};

class GraphKernelTest : public GraphExecSemaphoreTest,
                        public ::testing::WithParamInterface<uint32_t> {
 protected:
  void SetUp() override {
    GraphExecSemaphoreTest::SetUp();
    ASSERT_FALSE(HasFatalFailure());
    const auto queue = hrx_device_get_queue(device_, 0, 0);
    static const iree_hal_task_executable_vtable_t vtable = {
        .base =
            {
                .destroy =
                    [](iree_hal_executable_t* executable) {
                      iree_hal_task_executable_deinitialize(
                          iree_hal_task_executable_cast(executable));
                      auto* resource =
                          reinterpret_cast<TestTaskExecutable*>(executable);
                      resource->destroyed.store(true);
                      if (resource->on_destroy) {
                        resource->on_destroy(resource->destroy_data);
                      }
                    },
            },
        .issue_call =
            [](iree_hal_task_executable_t* executable, iree_host_size_t ordinal,
               const iree_hal_executable_dispatch_state_v0_t* state,
               const iree_hal_executable_workgroup_state_v0_t* group,
               uint32_t worker_id) {
              WriteResult(&executable->environment, state, group);
              return iree_ok_status();
            },
    };
    iree_hal_task_executable_initialize(iree_hal_queue_family(queue->hal_queue),
                                        &vtable, iree_allocator_system(),
                                        &task_executable_.base);
    task_executable_.base.dispatch_attrs = &dispatch_attrs_;
    task_executable_.base.dispatch_ptrs = &dispatch_;
    task_executable_.base.export_count = 1;
    executable_.hal_executable = &task_executable_.base.base;
    executable_.device = device_;
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_create(queue, 0, &stream_)));
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_create(device_, 0, &graph_)));
  }

  void TearDown() override {
    if (stream_) {
      IREE_EXPECT_OK(hrx_status_to_iree(hrx_stream_synchronize(stream_)));
    }
    hrx_graph_exec_release(exec_);
    hrx_graph_release(graph_);
    hrx_stream_release(stream_);
    iree_hal_executable_release(executable_.hal_executable);
    GraphExecSemaphoreTest::TearDown();
  }

  hrx_graph_kernel_node_attrs_t KernelAttrs(uint32_t flags) {
    return {
        .executable = &executable_,
        .config = {{1, 1, 1}, {1, 1, 1}, 0},
        .constants = &arguments_,
        .constants_size = sizeof(arguments_),
        .flags = flags,
    };
  }

  Graph MakeKernelGraph(uint32_t* result, uint32_t value) {
    Graph graph;
    IREE_EXPECT_OK(
        hrx_status_to_iree(hrx_graph_create(device_, 0, graph.for_output())));
    KernelArguments arguments = {result, value};
    auto attrs = KernelAttrs(GetParam());
    attrs.constants = &arguments;
    IREE_EXPECT_OK(hrx_status_to_iree(
        hrx_graph_add_kernel_node(graph, nullptr, 0, &attrs, nullptr)));
    return graph;
  }

  void InstantiateOriginal() {
    auto attrs = KernelAttrs(GetParam());
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_graph_add_kernel_node(graph_, nullptr, 0, &attrs, nullptr)));
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_instantiate(
        graph_, hrx_stream_get_queue(stream_), 0, &exec_)));
  }

  // Fixture-owned native executable; graph capture retains its HAL resource.
  TestTaskExecutable task_executable_;
  // Borrowed HRX view used to exercise public recording without a file loader.
  hrx_executable_s executable_ = {};
  // Normal task ABI: constants only, with no buffer bindings.
  const iree_hal_executable_dispatch_attrs_v0_t dispatch_attrs_ = {
      .constant_byte_length = sizeof(KernelArguments),
  };
  // Entry point baked into the task command buffer at recording time.
  const iree_hal_executable_dispatch_v0_t dispatch_ = WriteResult;
  // Host result observed only after the stream completes.
  uint32_t result_ = 0;
  // Arguments copied by graph insertion; their pointee remains fixture-owned.
  KernelArguments arguments_ = {&result_, 0x12345678u};
  // Owned stream used for both immediate work and graph replay.
  hrx_stream_t stream_ = nullptr;
  // Owned graph template.
  hrx_graph_t graph_ = nullptr;
  // Owned instantiated graph, released after stream completion.
  hrx_graph_exec_t exec_ = nullptr;
};

TEST_P(GraphKernelTest, ConstantsWithoutBindingsUseNormalArguments) {
  auto attrs = KernelAttrs(GetParam());
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_dispatch(
      stream_, attrs.executable, attrs.export_ordinal, &attrs.config,
      attrs.constants, attrs.constants_size, nullptr, 0, attrs.flags)));
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_synchronize(stream_)));
  EXPECT_EQ(result_, arguments_.value);

  result_ = 0;
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_graph_add_kernel_node(graph_, nullptr, 0, &attrs, nullptr)));
  // The graph snapshots constants, just as immediate recording does.
  const uint32_t expected = arguments_.value;
  arguments_.value = 0;
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_graph_instantiate(graph_, hrx_stream_get_queue(stream_), 0, &exec_)));
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_exec_launch(exec_, stream_)));
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_synchronize(stream_)));
  EXPECT_EQ(result_, expected);
}

TEST_P(GraphKernelTest, CustomArgumentsRemainAnExplicitBackendRequirement) {
  auto attrs =
      KernelAttrs(GetParam() | HRX_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS);
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_graph_add_kernel_node(graph_, nullptr, 0, &attrs, nullptr)));
  // Task dispatch does not support custom arguments; preserve that contract
  // rather than silently treating custom argument bytes as task ABI constants.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        hrx_status_to_iree(hrx_graph_instantiate(
                            graph_, hrx_stream_get_queue(stream_), 0, &exec_)));
  EXPECT_EQ(exec_, nullptr);
}

TEST_P(GraphKernelTest, UnsupportedFlagsLeaveTheGraphUnchanged) {
  auto attrs = KernelAttrs(GetParam() | (1u << 31));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        hrx_status_to_iree(hrx_graph_add_kernel_node(
                            graph_, nullptr, 0, &attrs, nullptr)));
  size_t node_count = 0;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_size(graph_, &node_count)));
  EXPECT_EQ(node_count, 0u);
}

TEST_P(GraphKernelTest, ReplacementPreservesPendingArgumentsAndResources) {
  InstantiateOriginal();
  Semaphore gate;
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_semaphore_create(device_, 0, gate.for_output())));
  IREE_EXPECT_OK(hrx_status_to_iree(hrx_stream_wait_on(stream_, {gate, 1})));
  IREE_EXPECT_OK(hrx_status_to_iree(hrx_graph_exec_launch(exec_, stream_)));

  uint32_t replacement_result = 0;
  auto replacement = MakeKernelGraph(&replacement_result, 31);
  IREE_EXPECT_OK(hrx_status_to_iree(hrx_graph_exec_update(exec_, replacement)));
  hrx_graph_release(graph_);
  graph_ = nullptr;
  replacement.reset();
  IREE_EXPECT_OK(hrx_status_to_iree(hrx_graph_exec_launch(exec_, stream_)));
  hrx_graph_exec_release(exec_);
  exec_ = nullptr;
  iree_hal_executable_release(executable_.hal_executable);
  executable_.hal_executable = nullptr;
  EXPECT_FALSE(task_executable_.destroyed.load());
  EXPECT_EQ(result_, 0u);
  EXPECT_EQ(replacement_result, 0u);

  IREE_ASSERT_OK(hrx_status_to_iree(hrx_semaphore_signal(gate, 1)));
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_synchronize(stream_)));
  EXPECT_EQ(result_, arguments_.value);
  EXPECT_EQ(replacement_result, 31u);
}

TEST_P(GraphKernelTest, LaunchCompletesWhileReplacementAllocationIsPaused) {
  InstantiateOriginal();
  auto replacement = MakeKernelGraph(&result_, 31);
  // Retire cached scratch so the candidate must call the actual allocator.
  iree_arena_block_pool_trim(&device_->block_pool);
  AllocationGate gate;
  hrx_status_t update_status = hrx_ok_status();
  std::thread editor([&] {
    AllocationScope scope(SIZE_MAX, &gate);
    update_status = hrx_graph_exec_update(exec_, replacement);
  });
  gate.entered.acquire();
  IREE_EXPECT_OK(hrx_status_to_iree(hrx_graph_exec_launch(exec_, stream_)));
  IREE_EXPECT_OK(hrx_status_to_iree(hrx_stream_synchronize(stream_)));
  EXPECT_EQ(result_, arguments_.value);
  gate.resume.release();
  editor.join();
  IREE_ASSERT_OK(hrx_status_to_iree(update_status));

  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_exec_launch(exec_, stream_)));
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_synchronize(stream_)));
  EXPECT_EQ(result_, 31u);
}

TEST_P(GraphKernelTest, LaterEditCannotBeOverwrittenByPausedPreparation) {
  InstantiateOriginal();
  auto first = MakeKernelGraph(&result_, 31);
  auto second = MakeKernelGraph(&result_, 47);
  iree_arena_block_pool_trim(&device_->block_pool);
  AllocationGate gate;
  std::binary_semaphore second_entered{0};
  hrx_status_t first_status = hrx_ok_status();
  hrx_status_t second_status = hrx_ok_status();
  std::thread first_editor([&] {
    AllocationScope scope(SIZE_MAX, &gate);
    first_status = hrx_graph_exec_update(exec_, first);
  });
  gate.entered.acquire();
  std::thread second_editor([&] {
    second_entered.release();
    second_status = hrx_graph_exec_update(exec_, second);
  });
  second_entered.acquire();
  IREE_EXPECT_OK(hrx_status_to_iree(hrx_graph_exec_launch(exec_, stream_)));
  IREE_EXPECT_OK(hrx_status_to_iree(hrx_stream_synchronize(stream_)));
  EXPECT_EQ(result_, arguments_.value);
  gate.resume.release();
  first_editor.join();
  second_editor.join();
  IREE_EXPECT_OK(hrx_status_to_iree(first_status));
  IREE_ASSERT_OK(hrx_status_to_iree(second_status));

  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_exec_launch(exec_, stream_)));
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_synchronize(stream_)));
  EXPECT_EQ(result_, 47u);
}

TEST_P(GraphKernelTest, AllocationFailureKeepsTheInstalledRecording) {
  InstantiateOriginal();
  auto replacement = MakeKernelGraph(&result_, 31);
  // Include queue operations and internal semaphores in partial construction.
  hrx_graph_node_t dependency = nullptr;
  const hrx_graph_host_call_node_attrs_t host = {
      .fn = [](void*) { return hrx_ok_status(); },
  };
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_add_host_call_node(
      replacement, nullptr, 0, &host, &dependency)));
  auto attrs = KernelAttrs(GetParam());
  KernelArguments arguments = {&result_, 31};
  attrs.constants = &arguments;
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_graph_add_kernel_node(replacement, &dependency, 1, &attrs, nullptr)));

  size_t failure_index = 0;
  for (;; ++failure_index) {
    SCOPED_TRACE(failure_index);
    iree_arena_block_pool_trim(&device_->block_pool);
    hrx_status_t status;
    {
      AllocationScope scope(failure_index);
      status = hrx_graph_exec_update(exec_, replacement);
    }
    if (hrx_status_is_ok(status)) {
      break;
    }
    IREE_ASSERT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                          hrx_status_to_iree(status));
    result_ = 0;
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_exec_launch(exec_, stream_)));
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_synchronize(stream_)));
    EXPECT_EQ(result_, arguments_.value);
  }
  EXPECT_GT(failure_index, 0u);
  RecordProperty("rejected_allocation_count", static_cast<int>(failure_index));
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_exec_launch(exec_, stream_)));
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_synchronize(stream_)));
  EXPECT_EQ(result_, 31u);
}

TEST_P(GraphKernelTest, ResourceDestructionCanReenterLaunchAndUpdate) {
  InstantiateOriginal();
  Graph empty;
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_graph_create(device_, 0, empty.for_output())));
  struct Reentry {
    // Executable under replacement, live through the callback.
    hrx_graph_exec_t executable;
    // Application-owned stream.
    hrx_stream_t stream;
    // Empty template kept live by the test.
    hrx_graph_t graph;
  } reentry = {exec_, stream_, empty};
  task_executable_.destroy_data = &reentry;
  task_executable_.on_destroy = [](void* data) {
    auto& reentry = *static_cast<Reentry*>(data);
    IREE_EXPECT_OK(hrx_status_to_iree(
        hrx_graph_exec_launch(reentry.executable, reentry.stream)));
    IREE_EXPECT_OK(hrx_status_to_iree(
        hrx_graph_exec_update(reentry.executable, reentry.graph)));
  };
  hrx_graph_release(graph_);
  graph_ = nullptr;
  iree_hal_executable_release(executable_.hal_executable);
  executable_.hal_executable = nullptr;
  EXPECT_FALSE(task_executable_.destroyed.load());
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_exec_update(exec_, empty)));
  EXPECT_TRUE(task_executable_.destroyed.load());
}

TEST_P(GraphKernelTest, InvalidTopologyKeepsTheInstalledRecording) {
  InstantiateOriginal();
  Graph cycle;
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_graph_create(device_, 0, cycle.for_output())));
  hrx_graph_node_t first = nullptr;
  hrx_graph_node_t second = nullptr;
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_graph_add_empty_node(cycle, nullptr, 0, &first)));
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_graph_add_empty_node(cycle, &first, 1, &second)));
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_graph_add_dependencies(cycle, &second, &first, 1)));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      hrx_status_to_iree(hrx_graph_exec_update(exec_, cycle)));
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_exec_launch(exec_, stream_)));
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_synchronize(stream_)));
  EXPECT_EQ(result_, arguments_.value);
}

TEST_P(GraphKernelTest, UnsupportedRecordingKeepsTheInstalledRecording) {
  InstantiateOriginal();
  Graph replacement;
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_graph_create(device_, 0, replacement.for_output())));
  auto attrs =
      KernelAttrs(GetParam() | HRX_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS);
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_graph_add_kernel_node(replacement, nullptr, 0, &attrs, nullptr)));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      hrx_status_to_iree(hrx_graph_exec_update(exec_, replacement)));
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_exec_launch(exec_, stream_)));
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_synchronize(stream_)));
  EXPECT_EQ(result_, arguments_.value);
}

INSTANTIATE_TEST_SUITE_P(
    DispatchFlags, GraphKernelTest,
    ::testing::Values(HRX_DISPATCH_FLAG_NONE,
                      HRX_DISPATCH_FLAG_ALLOW_INLINE_EXECUTION));

}  // namespace
