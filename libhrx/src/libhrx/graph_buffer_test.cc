// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <array>

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
      .region =
          {
              .row_length = 16,
              .row_count = 1,
              .slice_count = 1,
          },
  };
  hrx_graph_node_t copy_node = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_add_copy_buffer_node(
      graph, nullptr, 0, &copy_attrs, &copy_node)));

  const hrx_graph_fill_buffer_node_attrs_t fill_attrs = {
      .dst = {destination_, 24, 8},
      .pattern = 0xA5A5A5A5u,
      .pattern_size = 4,
      .region =
          {
              .row_length = 8,
              .row_count = 1,
              .slice_count = 1,
          },
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
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_stream_synchronize(stream_, HRX_WAIT_FLAG_NONE)));
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
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_semaphore_wait(timeline, value, UINT64_MAX, HRX_WAIT_FLAG_NONE)));

  hrx::runtime::hrx_ptr<hrx_graph_t, hrx_graph_retain, hrx_graph_release> graph;
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_graph_create(device_, 0, graph.for_output())));
  const hrx_graph_fill_buffer_node_attrs_t fill = {
      .dst = {buffer, 0, 64},
      .pattern = 0xA5u,
      .pattern_size = 1,
      .region =
          {
              .row_length = 64,
              .row_count = 1,
              .slice_count = 1,
          },
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
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_stream_synchronize(stream_, HRX_WAIT_FLAG_NONE)));

  uint8_t contents[64] = {};
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_synchronous_d2h_linear(
      queue, buffer, 0, contents, sizeof(contents))));
  for (uint8_t byte : contents) {
    EXPECT_EQ(byte, 0xA5u);
  }
  hrx_buffer_t root = buffer;
  value = 2;
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_queue_dealloca(queue, nullptr, &point, 1, &root)));
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_semaphore_wait(timeline, value, UINT64_MAX, HRX_WAIT_FLAG_NONE)));
  buffer.reset();

  // The graph still holds the old native handle, but its allocation epoch has
  // ended. The only block is reusable without destroying that graph.
  hrx_pool_stats_t stats;
  hrx_mem_pool_query_stats(pool, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  value = 3;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_queue_alloca(
      queue, nullptr, &point, pool, 1, &request, buffer.for_output())));
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_semaphore_wait(timeline, value, UINT64_MAX, HRX_WAIT_FLAG_NONE)));
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
      .region =
          {
              .row_length = 64,
              .row_count = 1,
              .slice_count = 1,
          },
  };
  hrx_graph_node_t fill_node = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_add_fill_buffer_node(
      graph, nullptr, 0, &fill_attrs, &fill_node)));

  const hrx_graph_copy_buffer_node_attrs_t copy_attrs = {
      .src = {source_, 0, 64},
      .dst = {destination_, 0, 64},
      .region =
          {
              .row_length = 64,
              .row_count = 1,
              .slice_count = 1,
          },
  };
  hrx_graph_node_t copy_node = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_add_copy_buffer_node(
      graph, &fill_node, 1, &copy_attrs, &copy_node)));

  hrx_graph_exec_t executable = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_instantiate(
      graph, hrx_stream_get_queue(stream_), 0, &executable)));
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_graph_exec_launch(executable, stream_)));
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_stream_synchronize(stream_, HRX_WAIT_FLAG_NONE)));

  uint8_t contents[64] = {};
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_synchronous_d2h_linear(hrx_device_get_queue(device_, 0, 0),
                                 destination_, 0, contents, sizeof(contents))));
  for (uint8_t value : contents) {
    EXPECT_EQ(value, 0xA5u);
  }

  hrx_graph_exec_release(executable);
  hrx_graph_release(graph);
}

TEST_F(GraphBufferTest, LaunchFlushesPendingStreamWorkBeforeGraphCommands) {
  const uint8_t pattern = 0x5Au;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_fill_buffer_linear(
      stream_, source_, 0, 64, &pattern, sizeof(pattern))));

  hrx_graph_t graph = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_create(device_, 0, &graph)));
  const hrx_graph_copy_buffer_node_attrs_t copy_attrs = {
      .src = {source_, 0, 64},
      .dst = {destination_, 0, 64},
      .region =
          {
              .row_length = 64,
              .row_count = 1,
              .slice_count = 1,
          },
  };
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_graph_add_copy_buffer_node(graph, nullptr, 0, &copy_attrs, nullptr)));

  hrx_graph_exec_t executable = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_instantiate(
      graph, hrx_stream_get_queue(stream_), 0, &executable)));
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_graph_exec_launch(executable, stream_)));
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_stream_synchronize(stream_, HRX_WAIT_FLAG_NONE)));

  uint8_t contents[64] = {};
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_synchronous_d2h_linear(hrx_device_get_queue(device_, 0, 0),
                                 destination_, 0, contents, sizeof(contents))));
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
      .region =
          {
              .row_length = 64,
              .row_count = 1,
              .slice_count = 1,
          },
  };
  hrx_graph_node_t fill_node = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_add_fill_buffer_node(
      graph, nullptr, 0, &fill_attrs, &fill_node)));

  const hrx_graph_copy_buffer_node_attrs_t copy_attrs = {
      .src = {source_, 0, 64},
      .dst = {destination_, 0, 64},
      .region =
          {
              .row_length = 64,
              .row_count = 1,
              .slice_count = 1,
          },
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
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_stream_synchronize(stream_, HRX_WAIT_FLAG_NONE)));

  uint8_t contents[64] = {};
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_synchronous_d2h_linear(hrx_device_get_queue(device_, 0, 0),
                                 destination_, 0, contents, sizeof(contents))));
  for (uint8_t value : contents) {
    EXPECT_EQ(value, 0x3Cu);
  }

  hrx_graph_exec_release(executable);
  hrx_graph_release(graph);
}

TEST_F(GraphBufferTest, RectangularGraphCapturesGeometryAndPreservesPadding) {
  for (size_t pattern_size : {1u, 2u, 4u}) {
    SCOPED_TRACE(pattern_size);
    hrx::runtime::hrx_ptr<hrx_graph_t, hrx_graph_retain, hrx_graph_release>
        graph;
    IREE_ASSERT_OK(
        hrx_status_to_iree(hrx_graph_create(device_, 0, graph.for_output())));
    hrx_graph_fill_buffer_node_attrs_t fill = {
        .dst = {source_, 4, 48},
        .pattern = 0x7635AB12u,
        .pattern_size = pattern_size,
        .region =
            {
                .row_length = 4,
                .row_count = 2,
                .slice_count = 2,
                .target_strides = {12, 28},
            },
    };
    hrx_graph_node_t previous = nullptr;
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_graph_add_fill_buffer_node(graph, nullptr, 0, &fill, &previous)));
    hrx_graph_copy_buffer_node_attrs_t copy = {
        .src = {source_, 4, 52},
        .dst = {destination_, 0, 64},
        .region =
            {
                .row_length = 8,
                .row_count = 2,
                .slice_count = 2,
                .source_strides = {12, 28},
                .target_strides = {16, 32},
            },
    };
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_graph_add_copy_buffer_node(graph, &previous, 1, &copy, nullptr)));
    fill.region = {};
    copy.region = {};
    hrx::runtime::hrx_ptr<hrx_graph_exec_t, hrx_graph_exec_retain,
                          hrx_graph_exec_release>
        executable;
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_instantiate(
        graph, hrx_stream_get_queue(stream_), 0, executable.for_output())));
    std::array<uint8_t, 64> source;
    std::array<uint8_t, 64> observed;
    std::array<uint8_t, 64> expected;
    const auto* pattern = reinterpret_cast<const uint8_t*>(&fill.pattern);
    for (uint32_t iteration = 0; iteration < 2; ++iteration) {
      for (size_t i = 0; i < source.size(); ++i) {
        source[i] = static_cast<uint8_t>(i * 7 + iteration * 31);
      }
      expected.fill(0xCD);
      IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_update_buffer_linear(
          stream_, source.data(), source.size(), source_, 0)));
      IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_update_buffer_linear(
          stream_, expected.data(), expected.size(), destination_, 0)));
      IREE_ASSERT_OK(
          hrx_status_to_iree(hrx_graph_exec_launch(executable, stream_)));
      IREE_ASSERT_OK(hrx_status_to_iree(
          hrx_stream_synchronize(stream_, HRX_WAIT_FLAG_NONE)));
      IREE_ASSERT_OK(hrx_status_to_iree(hrx_synchronous_d2h_linear(
          hrx_stream_get_queue(stream_), destination_, 0, observed.data(),
          observed.size())));
      for (size_t z = 0; z < 2; ++z) {
        for (size_t y = 0; y < 2; ++y) {
          for (size_t x = 0; x < 8; ++x) {
            expected[z * 32 + y * 16 + x] =
                x < 4 ? pattern[x % pattern_size]
                      : source[4 + z * 28 + y * 12 + x];
          }
        }
      }
      EXPECT_EQ(observed, expected);
    }
  }
}

TEST_F(GraphBufferTest, EmptyGeometryPreservesDependencyEdges) {
  hrx::runtime::hrx_ptr<hrx_graph_t, hrx_graph_retain, hrx_graph_release> graph;
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_graph_create(device_, 0, graph.for_output())));
  const hrx_graph_fill_buffer_node_attrs_t fill = {
      .dst = {source_, 0, 64},
      .pattern = 0x4Bu,
      .pattern_size = 1,
      .region =
          {
              .row_length = 64,
              .row_count = 1,
              .slice_count = 1,
          },
  };
  hrx_graph_node_t previous = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_graph_add_fill_buffer_node(graph, nullptr, 0, &fill, &previous)));
  hrx_graph_copy_buffer_node_attrs_t copy = {
      .src = {source_, 64, 0},
      .dst = {destination_, 64, 0},
      .region =
          {
              .row_length = UINT64_MAX,
              .row_count = UINT64_MAX,
              .slice_count = 0,
          },
  };
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_graph_add_copy_buffer_node(graph, &previous, 1, &copy, &previous)));
  copy = {
      .src = {source_, 0, 64},
      .dst = {destination_, 0, 64},
      .region =
          {
              .row_length = 64,
              .row_count = 1,
              .slice_count = 1,
          },
  };
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_graph_add_copy_buffer_node(graph, &previous, 1, &copy, nullptr)));
  hrx::runtime::hrx_ptr<hrx_graph_exec_t, hrx_graph_exec_retain,
                        hrx_graph_exec_release>
      executable;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_instantiate(
      graph, hrx_stream_get_queue(stream_), 0, executable.for_output())));
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_graph_exec_launch(executable, stream_)));
  std::array<uint8_t, 64> observed;
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_stream_synchronize(stream_, HRX_WAIT_FLAG_NONE)));
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_synchronous_d2h_linear(hrx_stream_get_queue(stream_), destination_, 0,
                                 observed.data(), observed.size())));
  for (uint8_t value : observed) {
    EXPECT_EQ(value, 0x4B);
  }
}

TEST_F(GraphBufferTest, RejectsInvalidHandleRangesAndPatterns) {
  hrx_graph_t graph = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_create(device_, 0, &graph)));

  hrx_graph_copy_buffer_node_attrs_t copy_attrs = {
      .src = {source_, 60, 8},
      .dst = {destination_, 0, 8},
      .region =
          {
              .row_length = 8,
              .row_count = 1,
              .slice_count = 1,
          },
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        hrx_status_to_iree(hrx_graph_add_copy_buffer_node(
                            graph, nullptr, 0, &copy_attrs, nullptr)));
  copy_attrs.src = {source_, 0, 8};
  copy_attrs.dst = {destination_, 0, 4};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        hrx_status_to_iree(hrx_graph_add_copy_buffer_node(
                            graph, nullptr, 0, &copy_attrs, nullptr)));

  hrx_graph_fill_buffer_node_attrs_t fill_attrs = {
      .dst = {destination_, 0, 8},
      .pattern = 0,
      .pattern_size = 3,
      .region =
          {
              .row_length = 8,
              .row_count = 1,
              .slice_count = 1,
          },
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        hrx_status_to_iree(hrx_graph_add_fill_buffer_node(
                            graph, nullptr, 0, &fill_attrs, nullptr)));
  fill_attrs.pattern_size = 4;
  fill_attrs.dst = {destination_, 0, 6};
  fill_attrs.region.row_length = 6;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        hrx_status_to_iree(hrx_graph_add_fill_buffer_node(
                            graph, nullptr, 0, &fill_attrs, nullptr)));
  fill_attrs.dst = {destination_, 2, 8};
  fill_attrs.region.row_length = 8;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        hrx_status_to_iree(hrx_graph_add_fill_buffer_node(
                            graph, nullptr, 0, &fill_attrs, nullptr)));
  copy_attrs.src = {source_, 0, 64};
  copy_attrs.dst = {destination_, 0, 64};
  copy_attrs.region.row_count = 2;
  copy_attrs.region.source_strides.row_stride = UINT64_MAX;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        hrx_status_to_iree(hrx_graph_add_copy_buffer_node(
                            graph, nullptr, 0, &copy_attrs, nullptr)));
  copy_attrs.region.row_count = UINT64_MAX;
  copy_attrs.region.source_strides.row_stride = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        hrx_status_to_iree(hrx_graph_add_copy_buffer_node(
                            graph, nullptr, 0, &copy_attrs, nullptr)));
  fill_attrs.dst = {destination_, 0, 64};
  fill_attrs.region.row_count = 2;
  fill_attrs.region.target_strides.row_stride = 10;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        hrx_status_to_iree(hrx_graph_add_fill_buffer_node(
                            graph, nullptr, 0, &fill_attrs, nullptr)));
  fill_attrs.region.target_strides.row_stride = 4;
  fill_attrs.region.slice_count = 2;
  fill_attrs.region.target_strides.slice_stride = 18;
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
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ABORTED,
      hrx_status_to_iree(hrx_stream_synchronize(stream_, HRX_WAIT_FLAG_NONE)));
  hrx_graph_exec_release(executable);
  hrx_graph_release(graph);
}

}  // namespace
