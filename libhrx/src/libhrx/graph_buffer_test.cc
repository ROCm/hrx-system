// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "hrx_internal.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class GraphBufferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_cpu_initialize(/*flags=*/0)));
    IREE_ASSERT_OK(
        hrx_status_to_iree(hrx_cpu_device_get(/*index=*/0, &device_)));
    IREE_ASSERT_OK(
        hrx_status_to_iree(hrx_stream_create(device_, /*flags=*/0, &stream_)));
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
    IREE_EXPECT_OK(hrx_status_to_iree(hrx_cpu_shutdown()));
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
      /*.src=*/{source_, 4, 16},
      /*.dst=*/{destination_, 8, 16},
  };
  hrx_graph_node_t copy_node = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_add_copy_buffer_node(
      graph, nullptr, 0, &copy_attrs, &copy_node)));

  const hrx_graph_fill_buffer_node_attrs_t fill_attrs = {
      /*.dst=*/{destination_, 24, 8},
      /*.pattern=*/0xA5A5A5A5u,
      /*.pattern_size=*/4,
  };
  hrx_graph_node_t fill_node = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_add_fill_buffer_node(
      graph, &copy_node, 1, &fill_attrs, &fill_node)));
  ASSERT_NE(fill_node, nullptr);

  size_t node_count = 0;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_size(graph, &node_count)));
  EXPECT_EQ(node_count, 2u);
  hrx_graph_exec_t executable = nullptr;
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_graph_instantiate(graph, 0, &executable)));

  // The executable retains the graph and its captured buffer allocations.
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

TEST_F(GraphBufferTest, DependentFillThenCopyProducesExpectedContents) {
  hrx_graph_t graph = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_create(device_, 0, &graph)));

  const hrx_graph_fill_buffer_node_attrs_t fill_attrs = {
      /*.dst=*/{source_, 0, 64},
      /*.pattern=*/0xA5u,
      /*.pattern_size=*/1,
  };
  hrx_graph_node_t fill_node = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_add_fill_buffer_node(
      graph, nullptr, 0, &fill_attrs, &fill_node)));

  const hrx_graph_copy_buffer_node_attrs_t copy_attrs = {
      /*.src=*/{source_, 0, 64},
      /*.dst=*/{destination_, 0, 64},
  };
  hrx_graph_node_t copy_node = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_add_copy_buffer_node(
      graph, &fill_node, 1, &copy_attrs, &copy_node)));

  hrx_graph_exec_t executable = nullptr;
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_graph_instantiate(graph, 0, &executable)));
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_graph_exec_launch(executable, stream_)));
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_synchronize(stream_)));

  uint8_t contents[64] = {};
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_synchronous_d2h(
      device_, destination_, 0, contents, sizeof(contents))));
  for (uint8_t value : contents) EXPECT_EQ(value, 0xA5u);

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
      /*.src=*/{source_, 0, 64},
      /*.dst=*/{destination_, 0, 64},
  };
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_graph_add_copy_buffer_node(graph, nullptr, 0, &copy_attrs, nullptr)));

  hrx_graph_exec_t executable = nullptr;
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_graph_instantiate(graph, 0, &executable)));
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_graph_exec_launch(executable, stream_)));
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_synchronize(stream_)));

  uint8_t contents[64] = {};
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_synchronous_d2h(
      device_, destination_, 0, contents, sizeof(contents))));
  for (uint8_t value : contents) EXPECT_EQ(value, pattern);

  hrx_graph_exec_release(executable);
  hrx_graph_release(graph);
}

TEST_F(GraphBufferTest, AddedDependencyOrdersFillBeforeCopy) {
  hrx_graph_t graph = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_create(device_, 0, &graph)));

  const hrx_graph_fill_buffer_node_attrs_t fill_attrs = {
      /*.dst=*/{source_, 0, 64},
      /*.pattern=*/0x3Cu,
      /*.pattern_size=*/1,
  };
  hrx_graph_node_t fill_node = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_add_fill_buffer_node(
      graph, nullptr, 0, &fill_attrs, &fill_node)));

  const hrx_graph_copy_buffer_node_attrs_t copy_attrs = {
      /*.src=*/{source_, 0, 64},
      /*.dst=*/{destination_, 0, 64},
  };
  hrx_graph_node_t copy_node = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_add_copy_buffer_node(
      graph, nullptr, 0, &copy_attrs, &copy_node)));
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_graph_add_dependencies(graph, &fill_node, &copy_node, /*count=*/1)));

  hrx_graph_exec_t executable = nullptr;
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_graph_instantiate(graph, 0, &executable)));
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_graph_exec_launch(executable, stream_)));
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_synchronize(stream_)));

  uint8_t contents[64] = {};
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_synchronous_d2h(
      device_, destination_, 0, contents, sizeof(contents))));
  for (uint8_t value : contents) EXPECT_EQ(value, 0x3Cu);

  hrx_graph_exec_release(executable);
  hrx_graph_release(graph);
}

TEST_F(GraphBufferTest, RejectsInvalidHandleRangesAndPatterns) {
  hrx_graph_t graph = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_create(device_, 0, &graph)));

  hrx_graph_copy_buffer_node_attrs_t copy_attrs = {
      /*.src=*/{source_, 60, 8},
      /*.dst=*/{destination_, 0, 8},
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
      /*.dst=*/{destination_, 0, 8},
      /*.pattern=*/0,
      /*.pattern_size=*/3,
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

}  // namespace
