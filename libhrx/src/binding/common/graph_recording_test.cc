// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include <semaphore>
#include <thread>
#include <vector>

#include "common/context.h"
#include "common/device.h"
#include "common/graph.h"
#include "common/host_callback.h"
#include "common/hrx_bridge.h"
#include "common/stream.h"
#include "iree/base/api.h"
#include "iree/base/status_cc.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "libhrx/src/libhrx/device.h"

namespace {

struct AllocationGate {
  // Announces that the editor reached its first real allocation.
  std::binary_semaphore entered{0};
  // Releases allocation after the installed recording has finished a launch.
  std::binary_semaphore resume{0};
};

// Only the editor's real allocator is controlled. Launches and task workers
// continue allocating normally, without a hook in the graph implementation.
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

  static iree_allocator_t Allocator() { return {nullptr, Control}; }

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

  // Active controls for this thread, or null during ordinary allocation.
  static thread_local AllocationScope* current_;
  // Number of allocations requested within this scope.
  size_t allocation_count_ = 0;
  // Ordinal to reject, or SIZE_MAX to preserve normal allocation.
  size_t fail_index_;
  // Optional pause on the first allocation.
  AllocationGate* gate_;
};
thread_local AllocationScope* AllocationScope::current_ = nullptr;

enum class EditKind { kGraph, kHostParameters, kChildGraph, kEnabled };

class GraphRecordingTest : public ::testing::TestWithParam<EditKind> {
 protected:
  void SetUp() override {
    const auto allocator = AllocationScope::Allocator();
    iree_hal_streaming_host_callback_queue_initialize(allocator,
                                                      &host_callbacks_);
    hrx_runtime_t runtime = nullptr;
    IREE_ASSERT_OK(HRX_CALL(
        hrx_runtime_create(nullptr, hrx_host_allocator_system(), &runtime)));
    const hrx_device_create_params_t params = {.uri = "task://"};
    const auto status = hrx_device_group_create(runtime, 1, &params, &group_);
    hrx_runtime_release(runtime);
    IREE_ASSERT_OK(HRX_CALL(status));
    device_.hrx_device = hrx_device_group_get_device(group_, 0);
    device_.hal_device = hrx_device_hal(device_.hrx_device);
    device_.host_callbacks = &host_callbacks_;
    iree_slim_mutex_initialize(&device_.primary_context_mutex);
    iree_slim_mutex_initialize(&device_.graph_memory_mutex);
    iree_arena_block_pool_initialize(64 * 1024, allocator, &device_.block_pool);
    const iree_hal_streaming_context_flags_t flags = {
        .scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO,
    };
    IREE_ASSERT_OK(iree_hal_streaming_context_create(&device_, flags, allocator,
                                                     &context_));
    IREE_ASSERT_OK(iree_hal_streaming_stream_create(
        context_, context_->queue, IREE_HAL_STREAMING_STREAM_FLAG_NONE, 0,
        allocator, &stream_));
  }

  void TearDown() override {
    if (context_) {
      IREE_EXPECT_OK(iree_hal_streaming_context_synchronize(context_));
    }
    iree_hal_streaming_graph_exec_release(executable_);
    for (auto* graph : graphs_) {
      iree_hal_streaming_graph_release(graph);
    }
    iree_hal_streaming_host_callback_queue_stop(&host_callbacks_);
    iree_hal_streaming_stream_release(stream_);
    iree_hal_streaming_context_release(context_);
    iree_arena_block_pool_deinitialize(&device_.block_pool);
    iree_slim_mutex_deinitialize(&device_.graph_memory_mutex);
    iree_slim_mutex_deinitialize(&device_.primary_context_mutex);
    hrx_device_group_release(group_);
    iree_hal_streaming_host_callback_queue_deinitialize(&host_callbacks_);
  }

  struct Write {
    // Application output observed after stream completion.
    uint32_t* target;
    // Value captured into the executable's callback parameters.
    uint32_t value;
  };

  void MakeGraph(uint32_t value, iree_hal_streaming_graph_node_t** out_node,
                 iree_hal_streaming_graph_t** out_graph) {
    IREE_ASSERT_OK(iree_hal_streaming_graph_create(
        context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE,
        AllocationScope::Allocator(), out_graph));
    graphs_.push_back(*out_graph);
    Write* write = nullptr;
    IREE_ASSERT_OK(iree_allocator_malloc((*out_graph)->arena_allocator,
                                         sizeof(*write), (void**)&write));
    *write = {&result_, value};
    IREE_ASSERT_OK(iree_hal_streaming_graph_add_host_call_node(
        *out_graph, nullptr, 0,
        +[](void* data) {
          const auto& write = *static_cast<Write*>(data);
          *write.target = write.value;
        },
        write, out_node));
    iree_hal_streaming_graph_node_host(*out_node)->user_data_size =
        sizeof(*write);
  }

  void Prepare() {
    iree_hal_streaming_graph_t* graph = nullptr;
    MakeGraph(17, &node_, &graph);
    ASSERT_FALSE(HasFatalFailure());
    if (GetParam() == EditKind::kChildGraph) {
      iree_hal_streaming_graph_t* parent = nullptr;
      IREE_ASSERT_OK(iree_hal_streaming_graph_create(
          context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE,
          AllocationScope::Allocator(), &parent));
      graphs_.push_back(parent);
      IREE_ASSERT_OK(iree_hal_streaming_graph_add_child_graph_node(
          parent, nullptr, 0, graph, &node_));
      graph = parent;
    }
    IREE_ASSERT_OK(iree_hal_streaming_graph_instantiate(
        graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &executable_));
    iree_hal_streaming_graph_node_t* replacement_node = nullptr;
    MakeGraph(31, &replacement_node, &replacement_);
    ASSERT_FALSE(HasFatalFailure());
  }

  iree_status_t Edit() {
    switch (GetParam()) {
      case EditKind::kGraph: {
        iree_hal_streaming_graph_node_t* error_node = nullptr;
        iree_hal_streaming_graph_exec_update_result_t result;
        return iree_hal_streaming_graph_exec_update(executable_, replacement_,
                                                    &error_node, &result);
      }
      case EditKind::kChildGraph:
        return iree_hal_streaming_graph_exec_set_child_graph(executable_, node_,
                                                             replacement_);
      case EditKind::kEnabled:
        return iree_hal_streaming_graph_exec_set_node_enabled(executable_,
                                                              node_, false);
      case EditKind::kHostParameters: {
        // The same private-description transaction used by HIP node setters.
        iree_hal_streaming_graph_exec_edit_t edit;
        auto* node =
            iree_hal_streaming_graph_exec_lock_node(executable_, node_, &edit);
        auto* write = static_cast<Write*>(
            iree_hal_streaming_graph_node_host(node)->user_data);
        const uint32_t old_value = write->value;
        write->value = 31;
        const auto status =
            iree_hal_streaming_graph_exec_rebuild_locked(executable_, &edit);
        if (!iree_status_is_ok(status)) {
          write->value = old_value;
        }
        iree_hal_streaming_graph_exec_end_edit(executable_, &edit);
        return status;
      }
    }
    return iree_status_from_code(IREE_STATUS_INTERNAL);
  }

  void Launch(uint32_t expected) {
    result_ = 0;
    iree_hal_streaming_graph_exec_launch_result_t result;
    IREE_ASSERT_OK(
        iree_hal_streaming_graph_exec_launch(executable_, stream_, &result));
    IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize(stream_));
    EXPECT_EQ(result_, expected);
  }

  uint32_t UpdatedValue() const {
    return GetParam() == EditKind::kEnabled ? 0 : 31;
  }

  // Task device and its explicit frontend registry entry.
  hrx_device_group_t group_ = nullptr;
  // Resources normally populated by frontend registry initialization.
  iree_hal_streaming_device_t device_ = {};
  // Frontend callout service, distinct from task queue workers.
  iree_hal_streaming_host_callback_queue_t host_callbacks_ = {};
  // Graph compilation and stream context.
  iree_hal_streaming_context_t* context_ = nullptr;
  // Submission stream retained until all work completes.
  iree_hal_streaming_stream_t* stream_ = nullptr;
  // Template ownership through fixture teardown.
  std::vector<iree_hal_streaming_graph_t*> graphs_;
  // Independently compiled executable under concurrent editing.
  iree_hal_streaming_graph_exec_t* executable_ = nullptr;
  // Original public node handle used by node-specific edits.
  iree_hal_streaming_graph_node_t* node_ = nullptr;
  // Topology-compatible replacement with a distinct output value.
  iree_hal_streaming_graph_t* replacement_ = nullptr;
  // Callback output, read only after stream completion.
  uint32_t result_ = 0;
  // Outlives executable cleanup even when a setup assertion fails.
  bool resource_destroyed_ = false;
};

TEST_P(GraphRecordingTest, LaunchCompletesWhilePreparationIsPaused) {
  Prepare();
  ASSERT_FALSE(HasFatalFailure());
  iree_arena_block_pool_trim(&device_.block_pool);
  AllocationGate gate;
  iree::Status update_status;
  std::thread editor([&] {
    AllocationScope scope(SIZE_MAX, &gate);
    update_status = iree::Status(Edit());
  });
  gate.entered.acquire();
  Launch(17);
  gate.resume.release();
  editor.join();
  IREE_ASSERT_OK(update_status);
  Launch(UpdatedValue());
}

TEST_P(GraphRecordingTest, AllocationFailurePreservesInstalledParameters) {
  Prepare();
  ASSERT_FALSE(HasFatalFailure());
  size_t failure_index = 0;
  for (;; ++failure_index) {
    SCOPED_TRACE(failure_index);
    iree_arena_block_pool_trim(&device_.block_pool);
    iree_status_t status;
    {
      AllocationScope scope(failure_index);
      status = Edit();
    }
    if (iree_status_is_ok(status)) {
      break;
    }
    IREE_ASSERT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
    Launch(17);
    ASSERT_FALSE(HasFatalFailure());
  }
  EXPECT_GT(failure_index, 0u);
  RecordProperty("rejected_allocation_count", static_cast<int>(failure_index));
  Launch(UpdatedValue());
}

TEST_P(GraphRecordingTest, LaterEditCannotBeOverwrittenByPausedPreparation) {
  Prepare();
  ASSERT_FALSE(HasFatalFailure());
  iree_arena_block_pool_trim(&device_.block_pool);
  AllocationGate gate;
  std::binary_semaphore second_entered{0};
  iree::Status first_status;
  iree::Status second_status;
  std::thread first_editor([&] {
    AllocationScope scope(SIZE_MAX, &gate);
    first_status = iree::Status(Edit());
  });
  gate.entered.acquire();
  const bool enabled = GetParam() == EditKind::kEnabled;
  std::thread second_editor([&] {
    second_entered.release();
    second_status = iree::Status(iree_hal_streaming_graph_exec_set_node_enabled(
        executable_, node_, enabled));
  });
  second_entered.acquire();
  Launch(17);
  gate.resume.release();
  first_editor.join();
  second_editor.join();
  IREE_EXPECT_OK(first_status);
  IREE_ASSERT_OK(second_status);
  Launch(enabled ? 17 : 0);
}

TEST_F(GraphRecordingTest, DetachedResourceCanReenterLaunchAndEditing) {
  iree_hal_streaming_graph_t* graph = nullptr;
  MakeGraph(17, &node_, &graph);
  ASSERT_FALSE(HasFatalFailure());
  IREE_ASSERT_OK(iree_hal_streaming_graph_instantiate(
      graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &executable_));
  struct Resource {
    // Resource retained by a private node and its compiled recording.
    iree_hal_resource_t base;
    // Executable whose replacement is releasing this resource.
    iree_hal_streaming_graph_exec_t* executable;
    // Original handle used to reenter editing.
    iree_hal_streaming_graph_node_t* node;
    // Stream used to reenter launch.
    iree_hal_streaming_stream_t* stream;
    // Records that destruction completed during the update.
    bool* destroyed;
    // Reentry is armed only when the detaching update succeeds.
    bool reenter;
  };
  static const iree_hal_resource_vtable_t vtable = {
      .destroy =
          +[](iree_hal_resource_t* base) {
            auto* resource = reinterpret_cast<Resource*>(base);
            if (resource->reenter) {
              iree_hal_streaming_graph_exec_launch_result_t result;
              IREE_EXPECT_OK(iree_hal_streaming_graph_exec_launch(
                  resource->executable, resource->stream, &result));
              IREE_EXPECT_OK(iree_hal_streaming_graph_exec_set_node_enabled(
                  resource->executable, resource->node, false));
            }
            *resource->destroyed = true;
            delete resource;
          },
  };
  auto* resource = new Resource{{},      executable_,          node_,
                                stream_, &resource_destroyed_, false};
  iree_hal_resource_initialize(&vtable, &resource->base);
  iree_hal_streaming_graph_exec_edit_t edit;
  auto* node =
      iree_hal_streaming_graph_exec_lock_node(executable_, node_, &edit);
  iree_hal_streaming_graph_node_host(node)->resource = &resource->base;
  auto status =
      iree_hal_streaming_graph_exec_rebuild_locked(executable_, &edit);
  iree_hal_streaming_graph_exec_end_edit(executable_, &edit);
  IREE_ASSERT_OK(status);
  EXPECT_FALSE(resource_destroyed_);

  node = iree_hal_streaming_graph_exec_lock_node(executable_, node_, &edit);
  iree_hal_streaming_graph_node_host(node)->resource = nullptr;
  // The installed recording still owns the resource. Its last release must
  // happen only after publishing the replacement and dropping both locks.
  iree_hal_resource_release(&resource->base);
  status = iree_hal_streaming_graph_exec_rebuild_locked(executable_, &edit);
  resource->reenter = iree_status_is_ok(status);
  iree_hal_streaming_graph_exec_end_edit(executable_, &edit);
  IREE_ASSERT_OK(status);
  EXPECT_TRUE(resource_destroyed_);
  IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize(stream_));
  EXPECT_EQ(result_, 17u);
  Launch(0);
}

TEST_F(GraphRecordingTest, LargeReplayUsesNoExecutableHeapStorage) {
  // Keep the executable allocator distinct from task work and host-callback
  // captures. Replaying the compiled schedule itself needs no heap storage.
  size_t allocations = 0;
  const iree_allocator_t allocator = {
      .self = &allocations,
      .ctl =
          +[](void* self, iree_allocator_command_t command,
              const void* parameters, void** pointer) {
            if (command != IREE_ALLOCATOR_COMMAND_FREE) {
              ++*static_cast<size_t*>(self);
            }
            const auto system = iree_allocator_system();
            return system.ctl(system.self, command, parameters, pointer);
          },
  };
  iree_hal_streaming_graph_t* graph = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_create(
      context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, allocator, &graph));
  std::unique_ptr<iree_hal_streaming_graph_t,
                  decltype(&iree_hal_streaming_graph_release)>
      graph_owner(graph, iree_hal_streaming_graph_release);
  iree_hal_streaming_graph_node_t* tail = nullptr;
  for (int i = 0; i < 64; ++i) {
    IREE_ASSERT_OK(iree_hal_streaming_graph_add_host_call_node(
        graph, tail ? &tail : nullptr, tail ? 1 : 0,
        +[](void* data) { ++*static_cast<uint32_t*>(data); }, &result_, &tail));
  }
  iree_hal_streaming_graph_exec_t* executable = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_instantiate(
      graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &executable));
  std::unique_ptr<iree_hal_streaming_graph_exec_t,
                  decltype(&iree_hal_streaming_graph_exec_release)>
      executable_owner(executable, iree_hal_streaming_graph_exec_release);
  allocations = 0;
  for (int replay = 0; replay < 3; ++replay) {
    iree_hal_streaming_graph_exec_launch_result_t result;
    IREE_EXPECT_OK(
        iree_hal_streaming_graph_exec_launch(executable, stream_, &result));
    IREE_EXPECT_OK(iree_hal_streaming_stream_synchronize(stream_));
  }
  EXPECT_EQ(result_, 3u * 64);
  EXPECT_EQ(allocations, 0u);
}

INSTANTIATE_TEST_SUITE_P(Edits, GraphRecordingTest,
                         ::testing::Values(EditKind::kGraph,
                                           EditKind::kHostParameters,
                                           EditKind::kChildGraph,
                                           EditKind::kEnabled),
                         [](const ::testing::TestParamInfo<EditKind>& info) {
                           switch (info.param) {
                             case EditKind::kGraph:
                               return "Graph";
                             case EditKind::kHostParameters:
                               return "HostParameters";
                             case EditKind::kChildGraph:
                               return "ChildGraph";
                             case EditKind::kEnabled:
                               return "Enabled";
                           }
                           return "";
                         });

}  // namespace
