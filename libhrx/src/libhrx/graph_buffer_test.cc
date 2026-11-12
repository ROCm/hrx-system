// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "hrx_internal.h"
#include "hrx_runtime_cxx.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class GraphBufferTest : public ::testing::Test {
 protected:
  // Owns the task device and its runtime through fixture cleanup.
  hrx_device_group_t group_ = nullptr;

  void SetUp() override {
    hrx_runtime_t runtime = nullptr;
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_runtime_create(nullptr, hrx_host_allocator_system(), &runtime)));
    const hrx_device_create_params_t params = {.uri = "task://"};
    hrx_status_t status = hrx_device_group_create(runtime, 1, &params, &group_);
    hrx_runtime_release(runtime);
    IREE_ASSERT_OK(hrx_status_to_iree(status));
    device_ = hrx_device_group_get_device(group_, 0);
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_create(
        hrx_device_get_queue(device_, 0, 0), /*flags=*/0, &stream_)));
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_buffer_allocate(stream_, 64, HRX_MEMORY_TYPE_DEVICE_LOCAL,
                            HRX_BUFFER_USAGE_DEFAULT, &source_)));
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_buffer_allocate(stream_, 64, HRX_MEMORY_TYPE_DEVICE_LOCAL,
                            HRX_BUFFER_USAGE_DEFAULT, &destination_)));
  }

  void TearDown() override {
    hrx_buffer_release(destination_);
    hrx_buffer_release(source_);
    hrx_stream_release(stream_);
    hrx_device_group_release(group_);
    group_ = nullptr;
  }

  hrx_device_t device_ = nullptr;
  hrx_stream_t stream_ = nullptr;
  hrx_buffer_t source_ = nullptr;
  hrx_buffer_t destination_ = nullptr;
};

TEST_F(GraphBufferTest, RecordsAndInstantiatesHandleBasedCopyAndFill) {
  hrx_graph_t graph = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_create(device_, 0, &graph)));

  const hrx_graph_copy_buffer_node_attrs_t copy_attrs = {
      .src = {source_, 4, 16},
      .dst = {destination_, 8, 16},
  };
  hrx_graph_node_t copy_node = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_add_copy_buffer_node(
      graph, nullptr, 0, &copy_attrs, &copy_node)));

  const hrx_graph_fill_buffer_node_attrs_t fill_attrs = {
      .dst = {destination_, 24, 8},
      .pattern = 0xA5A5A5A5u,
      .pattern_size = 4,
  };
  hrx_graph_node_t fill_node = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_add_fill_buffer_node(
      graph, &copy_node, 1, &fill_attrs, &fill_node)));
  ASSERT_NE(fill_node, nullptr);

  size_t node_count = 0;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_size(graph, &node_count)));
  EXPECT_EQ(node_count, 2u);
  hrx_graph_exec_t executable = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_instantiate(
      graph, hrx_stream_get_queue(stream_), 0, &executable)));

  // The executable retains the graph and its explicit buffer handles. The
  // device owns the convenience allocation sources through fixture teardown.
  hrx_buffer_release(destination_);
  destination_ = nullptr;
  hrx_buffer_release(source_);
  source_ = nullptr;
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_graph_exec_launch(executable, stream_)));
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_synchronize(stream_)));
  hrx_graph_exec_release(executable);
  hrx_graph_release(graph);
}

TEST_F(GraphBufferTest, RetainedGraphDoesNotExtendExplicitAllocationEpoch) {
  hrx::runtime::mem_pool_ptr pool;
  const hrx_fixed_block_pool_options_t options = {.block_size = 64};
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_fixed_block_pool_create_from_buffer(
      source_, 0, 64, &options, pool.for_output())));
  hrx::runtime::semaphore_ptr timeline;
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_semaphore_create(device_, 0, timeline.for_output())));
  hrx_semaphore_t semaphore = timeline;
  uint64_t value = 1;
  const hrx_semaphore_list_t point = {&semaphore, &value, 1};
  const hrx_pool_reservation_request_t request = {64, 16};
  hrx::runtime::buffer_ptr buffer;
  hrx_queue_t queue = hrx_stream_get_queue(stream_);
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_queue_alloca(
      queue, nullptr, &point, pool, 1, &request, buffer.for_output())));
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_semaphore_wait(timeline, value, UINT64_MAX)));

  hrx::runtime::hrx_ptr<hrx_graph_t, hrx_graph_retain, hrx_graph_release> graph;
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_graph_create(device_, 0, graph.for_output())));
  const hrx_graph_fill_buffer_node_attrs_t fill = {
      .dst = {buffer, 0, 64},
      .pattern = 0xA5u,
      .pattern_size = 1,
  };
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_graph_add_fill_buffer_node(graph, nullptr, 0, &fill, nullptr)));
  hrx::runtime::hrx_ptr<hrx_graph_exec_t, hrx_graph_exec_retain,
                        hrx_graph_exec_release>
      executable;
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_graph_instantiate(graph, queue, 0, executable.for_output())));
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_graph_exec_launch(executable, stream_)));
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_synchronize(stream_)));

  uint8_t contents[64] = {};
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_synchronous_d2h(queue, buffer, 0, contents, sizeof(contents))));
  for (uint8_t byte : contents) {
    EXPECT_EQ(byte, 0xA5u);
  }
  hrx_buffer_t root = buffer;
  value = 2;
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_queue_dealloca(queue, nullptr, &point, 1, &root)));
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_semaphore_wait(timeline, value, UINT64_MAX)));
  buffer.reset();

  // The graph still holds the old native handle, but its allocation epoch has
  // ended. The only block is reusable without destroying that graph.
  hrx_pool_stats_t stats;
  hrx_mem_pool_query_stats(pool, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  value = 3;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_queue_alloca(
      queue, nullptr, &point, pool, 1, &request, buffer.for_output())));
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_semaphore_wait(timeline, value, UINT64_MAX)));
  executable.reset();
  graph.reset();
  hrx_mem_pool_query_stats(pool, &stats);
  EXPECT_EQ(stats.reservation_count, 1u);
  buffer.reset();
  hrx_mem_pool_query_stats(pool, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
}

TEST_F(GraphBufferTest, DependentFillThenCopyProducesExpectedContents) {
  hrx_graph_t graph = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_create(device_, 0, &graph)));

  const hrx_graph_fill_buffer_node_attrs_t fill_attrs = {
      .dst = {source_, 0, 64},
      .pattern = 0xA5u,
      .pattern_size = 1,
  };
  hrx_graph_node_t fill_node = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_add_fill_buffer_node(
      graph, nullptr, 0, &fill_attrs, &fill_node)));

  const hrx_graph_copy_buffer_node_attrs_t copy_attrs = {
      .src = {source_, 0, 64},
      .dst = {destination_, 0, 64},
  };
  hrx_graph_node_t copy_node = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_add_copy_buffer_node(
      graph, &fill_node, 1, &copy_attrs, &copy_node)));

  hrx_graph_exec_t executable = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_instantiate(
      graph, hrx_stream_get_queue(stream_), 0, &executable)));
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_graph_exec_launch(executable, stream_)));
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_synchronize(stream_)));

  uint8_t contents[64] = {};
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_synchronous_d2h(hrx_device_get_queue(device_, 0, 0), destination_, 0,
                          contents, sizeof(contents))));
  for (uint8_t value : contents) {
    EXPECT_EQ(value, 0xA5u);
  }

  hrx_graph_exec_release(executable);
  hrx_graph_release(graph);
}

TEST_F(GraphBufferTest, LaunchFlushesPendingStreamWorkBeforeGraphCommands) {
  const uint8_t pattern = 0x5Au;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_fill_buffer(
      stream_, source_, 0, 64, &pattern, sizeof(pattern))));

  hrx_graph_t graph = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_create(device_, 0, &graph)));
  const hrx_graph_copy_buffer_node_attrs_t copy_attrs = {
      .src = {source_, 0, 64},
      .dst = {destination_, 0, 64},
  };
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_graph_add_copy_buffer_node(graph, nullptr, 0, &copy_attrs, nullptr)));

  hrx_graph_exec_t executable = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_instantiate(
      graph, hrx_stream_get_queue(stream_), 0, &executable)));
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_graph_exec_launch(executable, stream_)));
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_synchronize(stream_)));

  uint8_t contents[64] = {};
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_synchronous_d2h(hrx_device_get_queue(device_, 0, 0), destination_, 0,
                          contents, sizeof(contents))));
  for (uint8_t value : contents) {
    EXPECT_EQ(value, pattern);
  }

  hrx_graph_exec_release(executable);
  hrx_graph_release(graph);
}

TEST_F(GraphBufferTest, AddedDependencyOrdersFillBeforeCopy) {
  hrx_graph_t graph = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_create(device_, 0, &graph)));

  const hrx_graph_fill_buffer_node_attrs_t fill_attrs = {
      .dst = {source_, 0, 64},
      .pattern = 0x3Cu,
      .pattern_size = 1,
  };
  hrx_graph_node_t fill_node = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_add_fill_buffer_node(
      graph, nullptr, 0, &fill_attrs, &fill_node)));

  const hrx_graph_copy_buffer_node_attrs_t copy_attrs = {
      .src = {source_, 0, 64},
      .dst = {destination_, 0, 64},
  };
  hrx_graph_node_t copy_node = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_add_copy_buffer_node(
      graph, nullptr, 0, &copy_attrs, &copy_node)));
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_graph_add_dependencies(graph, &fill_node, &copy_node, /*count=*/1)));

  hrx_graph_exec_t executable = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_instantiate(
      graph, hrx_stream_get_queue(stream_), 0, &executable)));
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_graph_exec_launch(executable, stream_)));
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_synchronize(stream_)));

  uint8_t contents[64] = {};
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_synchronous_d2h(hrx_device_get_queue(device_, 0, 0), destination_, 0,
                          contents, sizeof(contents))));
  for (uint8_t value : contents) {
    EXPECT_EQ(value, 0x3Cu);
  }

  hrx_graph_exec_release(executable);
  hrx_graph_release(graph);
}

TEST_F(GraphBufferTest, RejectsInvalidHandleRangesAndPatterns) {
  hrx_graph_t graph = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_create(device_, 0, &graph)));

  hrx_graph_copy_buffer_node_attrs_t copy_attrs = {
      .src = {source_, 60, 8},
      .dst = {destination_, 0, 8},
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        hrx_status_to_iree(hrx_graph_add_copy_buffer_node(
                            graph, nullptr, 0, &copy_attrs, nullptr)));
  copy_attrs.src = {source_, 0, 8};
  copy_attrs.dst = {destination_, 0, 4};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        hrx_status_to_iree(hrx_graph_add_copy_buffer_node(
                            graph, nullptr, 0, &copy_attrs, nullptr)));

  hrx_graph_fill_buffer_node_attrs_t fill_attrs = {
      .dst = {destination_, 0, 8},
      .pattern = 0,
      .pattern_size = 3,
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        hrx_status_to_iree(hrx_graph_add_fill_buffer_node(
                            graph, nullptr, 0, &fill_attrs, nullptr)));
  fill_attrs.pattern_size = 4;
  fill_attrs.dst = {destination_, 0, 6};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        hrx_status_to_iree(hrx_graph_add_fill_buffer_node(
                            graph, nullptr, 0, &fill_attrs, nullptr)));
  fill_attrs.dst = {destination_, 2, 8};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        hrx_status_to_iree(hrx_graph_add_fill_buffer_node(
                            graph, nullptr, 0, &fill_attrs, nullptr)));
  hrx_graph_release(graph);
}

TEST_F(GraphBufferTest, HostCallbackFailureReachesStreamCompletion) {
  hrx_graph_t graph = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_create(device_, 0, &graph)));
  const hrx_graph_host_call_node_attrs_t attrs = {
      .fn = [](void*) -> hrx_status_t {
        return hrx_make_status(HRX_STATUS_ABORTED,
                               "application callback failed");
      },
      .user_data = nullptr,
  };
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_graph_add_host_call_node(graph, nullptr, 0, &attrs, nullptr)));
  hrx_graph_exec_t executable = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_instantiate(
      graph, hrx_stream_get_queue(stream_), 0, &executable)));
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_graph_exec_launch(executable, stream_)));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                        hrx_status_to_iree(hrx_stream_synchronize(stream_)));
  hrx_graph_exec_release(executable);
  hrx_graph_release(graph);
}

}  // namespace
