// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>

#include "hrx_test_fixture.hpp"

namespace {

struct StreamGraphResources {
  void CreateStream(hrx_device_t device, uint32_t family_ordinal,
                    const hrx_queue_family_info_t& family) {
    hrx_queue_t queue = nullptr;
    if (family.supports_acquisition) {
      REQUIRE_OK(hrx().queue_acquire(device, family_ordinal, nullptr, &queue));
    } else {
      queue = hrx().device_get_queue(device, family_ordinal, 0);
    }
    hrx_status_t status = hrx().stream_create(queue, 0, &stream);
    if (family.supports_acquisition) {
      hrx().queue_release(queue);
    }
    REQUIRE_OK(status);
    REQUIRE(hrx().stream_get_queue(stream) == queue);
  }

  ~StreamGraphResources() {
    // Joining the stream keeps caller-owned upload/readback storage valid
    // through completion, including assertion failures inside the test.
    if (stream) {
      hrx().stream_release(stream);
    }
    if (executable) {
      hrx().graph_exec_release(executable);
    }
    if (graph) {
      hrx().graph_release(graph);
    }
    if (target) {
      hrx().buffer_release(target);
    }
    if (source) {
      hrx().buffer_release(source);
    }
  }

  // Stream owning the exact queue used for every submission in the test.
  hrx_stream_t stream = nullptr;
  // Source buffer captured by the graph.
  hrx_buffer_t source = nullptr;
  // Destination buffer captured by the graph.
  hrx_buffer_t target = nullptr;
  // Mutable graph used to construct the executable.
  hrx_graph_t graph = nullptr;
  // Family-specific recording replayed on the stream.
  hrx_graph_exec_t executable = nullptr;
};

}  // namespace

TEST_CASE_METHOD(HrxTestFixture, "Stream create and release", "[stream]") {
  hrx_stream_t stream = nullptr;
  REQUIRE_OK(hrx().stream_create(transfer_queue_, 0, &stream));
  REQUIRE(stream != nullptr);
  REQUIRE(hrx().stream_get_queue(stream) == transfer_queue_);
  hrx().stream_release(stream);
}

TEST_CASE_METHOD(HrxTestFixture,
                 "Stream graphs use the selected transfer family",
                 "[stream][graph][transfer]") {
  for (uint32_t family_ordinal = 0;
       family_ordinal < hrx().device_queue_family_count(device_);
       ++family_ordinal) {
    hrx_queue_family_info_t family;
    REQUIRE_OK(
        hrx().device_query_queue_family(device_, family_ordinal, &family));
    if (!(family.roles & HRX_QUEUE_ROLE_TRANSFER) ||
        (!family.provisioned_queue_count && !family.supports_acquisition)) {
      continue;
    }
    CAPTURE(family_ordinal, family.roles);
    std::array<uint32_t, 64> input = {};
    std::array<uint32_t, 64> output = {};
    StreamGraphResources resources;
    resources.CreateStream(device_, family_ordinal, family);
    REQUIRE_OK(hrx().buffer_allocate(
        resources.stream, sizeof(input), HRX_MEMORY_TYPE_DEVICE_LOCAL,
        HRX_BUFFER_USAGE_DEFAULT, &resources.source));
    REQUIRE_OK(hrx().buffer_allocate(
        resources.stream, sizeof(output), HRX_MEMORY_TYPE_DEVICE_LOCAL,
        HRX_BUFFER_USAGE_DEFAULT, &resources.target));
    REQUIRE_OK(hrx().graph_create(device_, 0, &resources.graph));
    const hrx_graph_copy_buffer_node_attrs_t copy = {
        .src = {resources.source, 0, sizeof(input)},
        .dst = {resources.target, 0, sizeof(output)},
    };
    hrx_graph_node_t copy_node = nullptr;
    REQUIRE_OK(hrx().graph_add_copy_buffer_node(resources.graph, nullptr, 0,
                                                &copy, &copy_node));
    const hrx_graph_fill_buffer_node_attrs_t fill = {
        .dst = {resources.target, 16 * sizeof(uint32_t), 8 * sizeof(uint32_t)},
        .pattern = 0xAABBCCDD,
        .pattern_size = sizeof(uint32_t),
    };
    REQUIRE_OK(hrx().graph_add_fill_buffer_node(resources.graph, &copy_node, 1,
                                                &fill, nullptr));
    REQUIRE_OK(hrx().graph_instantiate(resources.graph,
                                       hrx().stream_get_queue(resources.stream),
                                       0, &resources.executable));
    for (uint32_t iteration = 0; iteration < 3; ++iteration) {
      for (size_t i = 0; i < input.size(); ++i) {
        input[i] = iteration * 101 + i * 13;
      }
      REQUIRE_OK(hrx().stream_copy_h2d(resources.stream, input.data(),
                                       resources.source, 0, sizeof(input)));
      REQUIRE_OK(
          hrx().graph_exec_launch(resources.executable, resources.stream));
      REQUIRE_OK(hrx().stream_copy_d2h(resources.stream, resources.target, 0,
                                       output.data(), sizeof(output)));
      REQUIRE_OK(hrx().stream_synchronize(resources.stream));
      for (size_t i = 0; i < output.size(); ++i) {
        REQUIRE(output[i] == (i >= 16 && i < 24 ? fill.pattern : input[i]));
      }
    }
  }
}

TEST_CASE_METHOD(HrxTestFixture,
                 "Incompatible graph family leaves pending stream work intact",
                 "[stream][graph][transfer]") {
  const uint32_t recording_family =
      hrx().queue_get_family_ordinal(transfer_queue_);
  uint32_t launch_family_ordinal = UINT32_MAX;
  hrx_queue_family_info_t launch_family = {};
  for (uint32_t i = 0; i < hrx().device_queue_family_count(device_); ++i) {
    hrx_queue_family_info_t family;
    REQUIRE_OK(hrx().device_query_queue_family(device_, i, &family));
    if (i != recording_family && (family.roles & HRX_QUEUE_ROLE_TRANSFER) &&
        (family.provisioned_queue_count || family.supports_acquisition)) {
      launch_family_ordinal = i;
      launch_family = family;
      break;
    }
  }
  if (launch_family_ordinal == UINT32_MAX) {
    SKIP("device exposes only one transfer queue family");
  }

  std::array<uint32_t, 16> output = {};
  StreamGraphResources resources;
  resources.CreateStream(device_, launch_family_ordinal, launch_family);
  REQUIRE_OK(hrx().buffer_allocate(
      resources.stream, sizeof(output), HRX_MEMORY_TYPE_DEVICE_LOCAL,
      HRX_BUFFER_USAGE_DEFAULT, &resources.target));
  REQUIRE_OK(hrx().graph_create(device_, 0, &resources.graph));
  const hrx_graph_fill_buffer_node_attrs_t fill = {
      .dst = {resources.target, 0, sizeof(output)},
      .pattern = 0xAABBCCDD,
      .pattern_size = sizeof(uint32_t),
  };
  REQUIRE_OK(hrx().graph_add_fill_buffer_node(resources.graph, nullptr, 0,
                                              &fill, nullptr));
  REQUIRE_OK(hrx().graph_instantiate(resources.graph, transfer_queue_, 0,
                                     &resources.executable));
  const uint32_t pattern = 0x11223344;
  REQUIRE_OK(hrx().stream_fill_buffer(resources.stream, resources.target, 0,
                                      sizeof(output), &pattern,
                                      sizeof(pattern)));
  hrx_timeline_point_t before;
  REQUIRE_OK(hrx().stream_get_timeline_position(resources.stream, &before));
  hrx_status_t status =
      hrx().graph_exec_launch(resources.executable, resources.stream);
  CHECK(hrx().status_code(status) == HRX_STATUS_INVALID_ARGUMENT);
  hrx().status_ignore(status);
  hrx_timeline_point_t after;
  REQUIRE_OK(hrx().stream_get_timeline_position(resources.stream, &after));
  REQUIRE(after.semaphore == before.semaphore);
  REQUIRE(after.value == before.value);
  REQUIRE_OK(hrx().stream_copy_d2h(resources.stream, resources.target, 0,
                                   output.data(), sizeof(output)));
  REQUIRE_OK(hrx().stream_synchronize(resources.stream));
  for (uint32_t value : output) {
    REQUIRE(value == pattern);
  }
}

TEST_CASE_METHOD(HrxTestFixture, "Stream reports owning device", "[stream]") {
  hrx_stream_t stream = nullptr;
  REQUIRE_OK(hrx().stream_create(transfer_queue_, 0, &stream));

  hrx_device_t stream_device = nullptr;
  REQUIRE_OK(hrx().stream_get_device(stream, &stream_device));
  REQUIRE(stream_device == device_);

  hrx().stream_release(stream);
}

TEST_CASE_METHOD(HrxTestFixture, "Stream flush with no work", "[stream]") {
  hrx_stream_t stream = nullptr;
  REQUIRE_OK(hrx().stream_create(transfer_queue_, 0, &stream));
  REQUIRE_OK(hrx().stream_flush(stream));
  hrx().stream_release(stream);
}

TEST_CASE_METHOD(HrxTestFixture, "Stream sync with no work", "[stream]") {
  hrx_stream_t stream = nullptr;
  REQUIRE_OK(hrx().stream_create(transfer_queue_, 0, &stream));
  REQUIRE_OK(hrx().stream_synchronize(stream));
  hrx().stream_release(stream);
}

TEST_CASE_METHOD(HrxTestFixture, "Stream wait with no work", "[stream]") {
  hrx_stream_t stream = nullptr;
  REQUIRE_OK(hrx().stream_create(transfer_queue_, 0, &stream));
  REQUIRE_OK(hrx().stream_wait(stream));
  hrx().stream_release(stream);
}

TEST_CASE_METHOD(HrxTestFixture, "Stream query empty is complete", "[stream]") {
  hrx_stream_t stream = nullptr;
  REQUIRE_OK(hrx().stream_create(transfer_queue_, 0, &stream));
  bool complete = false;
  REQUIRE_OK(hrx().stream_query(stream, &complete));
  REQUIRE(complete);
  hrx().stream_release(stream);
}

TEST_CASE_METHOD(HrxTestFixture, "Stream get_semaphore", "[stream][sync]") {
  hrx_stream_t stream = nullptr;
  REQUIRE_OK(hrx().stream_create(transfer_queue_, 0, &stream));

  hrx_semaphore_t sem = nullptr;
  REQUIRE_OK(hrx().stream_get_semaphore(stream, &sem));
  REQUIRE(sem != nullptr);

  hrx().stream_release(stream);
}

TEST_CASE_METHOD(HrxTestFixture, "Stream get_timeline_position",
                 "[stream][sync]") {
  hrx_stream_t stream = nullptr;
  REQUIRE_OK(hrx().stream_create(transfer_queue_, 0, &stream));

  hrx_timeline_point_t pos;
  REQUIRE_OK(hrx().stream_get_timeline_position(stream, &pos));
  REQUIRE(pos.semaphore != nullptr);
  REQUIRE(pos.value == 0);

  hrx().stream_release(stream);
}
