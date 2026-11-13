// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <atomic>
#include <memory>
#include <semaphore>
#include <thread>
#include <vector>

#include "hrx_internal.h"
#include "hrx_runtime_cxx.h"
#include "iree/hal/driver/task/executable/executable.h"
#include "iree/hal/driver/task/executable/library/abi.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

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

TEST_F(GraphExecSemaphoreTest, UnrepresentableDependenciesLeaveGraphUnchanged) {
  Graph graph;
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_graph_create(device_, 0, graph.for_output())));
  hrx_graph_node_t dependency = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_graph_add_empty_node(graph, nullptr, 0, &dependency)));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        hrx_status_to_iree(hrx_graph_add_empty_node(
                            graph, &dependency, SIZE_MAX, nullptr)));
  size_t count = 0;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_size(graph, &count)));
  EXPECT_EQ(count, 1u);
}

// A normal task executable may receive pointers in its constant bytes without
// changing to a custom argument ABI or declaring any buffer bindings.
struct KernelArguments {
  // Fixture-owned result storage, valid until stream completion.
  uint32_t* result;
  // Value the dispatch writes to |result|.
  uint32_t value;
  // Optional earlier producer read by this dispatch.
  const uint32_t* input = nullptr;
};

static int WriteResult(
    const iree_hal_executable_environment_v0_t* environment,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state) {
  KernelArguments arguments;
  memcpy(&arguments, dispatch_state->constants.data, sizeof(arguments));
  *arguments.result =
      (arguments.input ? *arguments.input : 0) + arguments.value;
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
  // Results carried through a device fork/join, host call, and device consumer.
  std::array<uint32_t, 5> pipeline_results_ = {};
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

TEST_P(GraphKernelTest, ReorderedBranchesJoinAcrossHostAndDeviceOperations) {
  // Create consumers before their producers; all edges are attached afterward.
  hrx_graph_node_t tail = nullptr;
  KernelArguments tail_arguments = {&pipeline_results_[4], 1,
                                    &pipeline_results_[3]};
  auto attrs = KernelAttrs(GetParam());
  attrs.constants = &tail_arguments;
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_graph_add_kernel_node(graph_, nullptr, 0, &attrs, &tail)));
  const hrx_graph_host_call_node_attrs_t host_attrs = {
      .fn = [](void* user_data) -> hrx_status_t {
        auto* values = static_cast<uint32_t*>(user_data);
        values[3] = values[1] + values[2];
        return hrx_ok_status();
      },
      .user_data = pipeline_results_.data(),
  };
  hrx_graph_node_t host = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_graph_add_host_call_node(graph_, nullptr, 0, &host_attrs, &host)));
  hrx_graph_node_t branches[2] = {};
  for (size_t i = 0; i < 2; ++i) {
    KernelArguments arguments = {&pipeline_results_[i + 1],
                                 static_cast<uint32_t>(17 + i * 12),
                                 &pipeline_results_[0]};
    attrs.constants = &arguments;
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_graph_add_kernel_node(graph_, nullptr, 0, &attrs, &branches[i])));
  }
  KernelArguments root_arguments = {&pipeline_results_[0], 3};
  attrs.constants = &root_arguments;
  hrx_graph_node_t root = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_graph_add_kernel_node(graph_, nullptr, 0, &attrs, &root)));
  hrx_graph_node_t join = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_add_empty_node(
      graph_, branches, IREE_ARRAYSIZE(branches), &join)));
  const hrx_graph_node_t from[] = {root, root, join, host};
  const hrx_graph_node_t to[] = {branches[0], branches[1], host, tail};
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_graph_add_dependencies(graph_, from, to, IREE_ARRAYSIZE(from))));
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_graph_instantiate(graph_, hrx_stream_get_queue(stream_), 0, &exec_)));
  for (int replay = 0; replay < 3; ++replay) {
    pipeline_results_.fill(0xDEADBEEFu);
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_exec_launch(exec_, stream_)));
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_synchronize(stream_)));
    EXPECT_EQ(pipeline_results_, (std::array<uint32_t, 5>{3, 20, 32, 52, 53}));
  }
}

TEST_P(GraphKernelTest, ForeignDependencyRejectsReplacement) {
  InstantiateOriginal();
  Graph foreign;
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_graph_create(device_, 0, foreign.for_output())));
  hrx_graph_node_t foreign_node = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_graph_add_empty_node(foreign, nullptr, 0, &foreign_node)));
  for (bool additional : {false, true}) {
    Graph invalid;
    IREE_ASSERT_OK(
        hrx_status_to_iree(hrx_graph_create(device_, 0, invalid.for_output())));
    hrx_graph_node_t node = nullptr;
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_graph_add_empty_node(invalid, additional ? nullptr : &foreign_node,
                                 additional ? 0 : 1, &node)));
    if (additional) {
      IREE_ASSERT_OK(hrx_status_to_iree(
          hrx_graph_add_dependencies(invalid, &foreign_node, &node, 1)));
    }
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        hrx_status_to_iree(hrx_graph_exec_update(exec_, invalid)));
    result_ = 0;
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_exec_launch(exec_, stream_)));
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_synchronize(stream_)));
    EXPECT_EQ(result_, arguments_.value);
  }
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
