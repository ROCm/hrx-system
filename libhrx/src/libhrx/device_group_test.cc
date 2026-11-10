// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>

#include "hrx_runtime_cxx.h"
#include "iree/testing/gtest.h"

namespace {

testing::AssertionResult Check(hrx_status_t status) {
  auto result = hrx_status_is_ok(status) ? testing::AssertionSuccess()
                                         : testing::AssertionFailure();
  if (!hrx_status_is_ok(status)) {
    result << hrx::runtime::format_status(status);
  }
  hrx_status_ignore(status);
  return result;
}

TEST(DeviceGroupTest, RetainedDeviceKeepsGroupAndServicesUsable) {
  hrx::runtime::runtime_ptr runtime;
  ASSERT_TRUE(Check(hrx_runtime_create(nullptr, hrx_host_allocator_system(),
                                       runtime.for_output())));
  ASSERT_TRUE(runtime);
  const hrx_device_create_params_t params[] = {{.uri = "task://"},
                                               {.uri = "task://"}};
  hrx::runtime::device_group_ptr group;
  ASSERT_TRUE(
      Check(hrx_device_group_create(runtime, 2, params, group.for_output())));
  ASSERT_TRUE(group);
  EXPECT_EQ(hrx_device_group_device_count(group), 2u);
  EXPECT_EQ(hrx_device_group_get_device(group, 2), nullptr);
  auto first =
      hrx::runtime::device_ptr::borrow(hrx_device_group_get_device(group, 0));
  auto second =
      hrx::runtime::device_ptr::borrow(hrx_device_group_get_device(group, 1));
  ASSERT_NE(first.get(), second.get());
  EXPECT_EQ(hrx_device_get_device_group(first), group.get());
  EXPECT_EQ(hrx_device_get_device_group(second), group.get());
  runtime.reset();
  group.reset();

  hrx::runtime::semaphore_ptr semaphore;
  ASSERT_TRUE(Check(hrx_semaphore_create(first, 0, semaphore.for_output())));
  ASSERT_TRUE(semaphore);
  hrx::runtime::buffer_ptr buffer;
  const hrx_buffer_params_t buffer_params = {
      .type = HRX_MEMORY_TYPE_HOST_LOCAL,
      .access = HRX_MEMORY_ACCESS_ALL,
      .usage = HRX_BUFFER_USAGE_DEFAULT,
  };
  ASSERT_TRUE(Check(hrx_allocator_allocate_buffer(
      hrx_device_allocator(second), buffer_params, 256, buffer.for_output())));
  ASSERT_TRUE(buffer);
  uint64_t value = 1;
  hrx_semaphore_t signal = semaphore;
  const hrx_semaphore_list_t signals = {&signal, &value, 1};
  const uint32_t pattern = 0x12345678;
  ASSERT_TRUE(Check(hrx_queue_fill(hrx_device_get_queue(second, 0, 0), nullptr,
                                   &signals, buffer, 0, 256, &pattern,
                                   sizeof(pattern))));
  ASSERT_TRUE(Check(hrx_semaphore_wait(semaphore, value, UINT64_MAX)));
  second.reset();
  first.reset();

  void* data = nullptr;
  ASSERT_TRUE(Check(hrx_buffer_map(buffer, HRX_MAP_READ, 0, 256, &data)));
  ASSERT_NE(data, nullptr);
  for (size_t i = 0; i < 64; ++i) {
    EXPECT_EQ(static_cast<uint32_t*>(data)[i], pattern);
  }
  ASSERT_TRUE(Check(hrx_buffer_unmap(buffer)));
}

TEST(DeviceGroupTest, FailedConstructionPreservesExistingGroup) {
  hrx::runtime::runtime_ptr runtime;
  ASSERT_TRUE(Check(hrx_runtime_create(nullptr, hrx_host_allocator_system(),
                                       runtime.for_output())));
  ASSERT_TRUE(runtime);
  hrx::runtime::device_group_ptr kept;
  const hrx_device_create_params_t task = {.uri = "task://"};
  ASSERT_TRUE(
      Check(hrx_device_group_create(runtime, 1, &task, kept.for_output())));
  ASSERT_TRUE(kept);

  const hrx_device_create_params_t invalid[] = {task,
                                                {.uri = "missing-driver://"}};
  hrx_device_group_t rejected = nullptr;
  hrx_status_t status = hrx_device_group_create(runtime, 2, invalid, &rejected);
  EXPECT_FALSE(hrx_status_is_ok(status));
  hrx_status_ignore(status);
  EXPECT_EQ(rejected, nullptr);

  hrx::runtime::semaphore_ptr semaphore;
  ASSERT_TRUE(Check(hrx_semaphore_create(hrx_device_group_get_device(kept, 0),
                                         0, semaphore.for_output())));
  ASSERT_TRUE(semaphore);
  ASSERT_TRUE(Check(hrx_semaphore_signal(semaphore, 17)));
  ASSERT_TRUE(Check(hrx_semaphore_wait(semaphore, 17, UINT64_MAX)));
}

TEST(DeviceGroupTest, RejectsInvalidInputsBeforeActivation) {
  hrx::runtime::runtime_ptr runtime;
  ASSERT_TRUE(Check(hrx_runtime_create(nullptr, hrx_host_allocator_system(),
                                       runtime.for_output())));
  ASSERT_TRUE(runtime);
  hrx_device_group_t group = nullptr;
  const hrx_device_create_params_t invalid = {
      .uri = "task://",
      .completion_wait_flags = UINT32_MAX,
  };
  hrx_status_t status = hrx_device_group_create(runtime, 1, &invalid, &group);
  EXPECT_EQ(hrx_status_code(status), HRX_STATUS_INVALID_ARGUMENT);
  hrx_status_ignore(status);
  EXPECT_EQ(group, nullptr);
  status = hrx_device_group_create(runtime, 0, nullptr, &group);
  EXPECT_EQ(hrx_status_code(status), HRX_STATUS_INVALID_ARGUMENT);
  hrx_status_ignore(status);
  EXPECT_EQ(group, nullptr);
}

// Dependent resources keep the explicit group alive after construction owners
// disappear. These cases use public borrowed handles only while such an owner
// remains, then exercise native work and final cleanup under that ownership.
class DeviceGroupLifetimeTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_TRUE(Check(hrx_runtime_create(nullptr, hrx_host_allocator_system(),
                                         runtime_.for_output())));
    ASSERT_TRUE(runtime_);
    const hrx_device_create_params_t task = {.uri = "task://"};
    ASSERT_TRUE(Check(
        hrx_device_group_create(runtime_, 1, &task, group_.for_output())));
    ASSERT_TRUE(group_);
    device_ = hrx_device_group_get_device(group_, 0);
  }

  void ReleaseConstructionOwners() {
    group_.reset();
    runtime_.reset();
  }

  // Construction reference to the services shared by the test's groups.
  hrx::runtime::runtime_ptr runtime_;
  // Construction reference, released before testing dependent ownership.
  hrx::runtime::device_group_ptr group_;
  // Borrowed view, valid while any test resource keeps its group alive.
  hrx_device_t device_ = nullptr;
};

constexpr hrx_buffer_params_t kHostBufferParams = {
    .type = HRX_MEMORY_TYPE_HOST_LOCAL,
    .access = HRX_MEMORY_ACCESS_ALL,
    .usage = HRX_BUFFER_USAGE_DEFAULT,
};

TEST(DeviceGroupTest, SharedReservationHasPerDeviceAddressesAndGraphAccess) {
  hrx::runtime::runtime_ptr runtime;
  ASSERT_TRUE(Check(hrx_runtime_create(nullptr, hrx_host_allocator_system(),
                                       runtime.for_output())));
  const hrx_device_create_params_t params[] = {{.uri = "task://"},
                                               {.uri = "task://"}};
  hrx::runtime::device_group_ptr group;
  ASSERT_TRUE(
      Check(hrx_device_group_create(runtime, 2, params, group.for_output())));
  hrx_device_t devices[] = {hrx_device_group_get_device(group, 0),
                            hrx_device_group_get_device(group, 1)};
  const hrx_pool_family_access_t families[] = {
      {
          .device = devices[0],
          .family_ordinal = 0,
          .usage = HRX_BUFFER_USAGE_TRANSFER | HRX_BUFFER_USAGE_DISPATCH,
          .interfaces = HRX_BUFFER_INTERFACE_BIT(HRX_BUFFER_INTERFACE_HOST),
      },
      {
          .device = devices[1],
          .family_ordinal = 0,
          .usage = HRX_BUFFER_USAGE_TRANSFER | HRX_BUFFER_USAGE_DISPATCH,
          .interfaces = HRX_BUFFER_INTERFACE_BIT(HRX_BUFFER_INTERFACE_HOST),
      },
  };
  const hrx_pool_scope_t scope = {
      .family_count = 2,
      .families = families,
      .host =
          {
              .access = HRX_MEMORY_ACCESS_ALL,
              .modes = HRX_MAPPING_MODE_SCOPED | HRX_MAPPING_MODE_PERSISTENT,
          },
  };
  auto source_families = std::array{families[0], families[1]};
  for (auto& family : source_families) {
    family.interfaces |=
        HRX_BUFFER_INTERFACE_BIT(HRX_BUFFER_INTERFACE_MAPPING_SOURCE);
  }
  auto source_scope = scope;
  source_scope.families = source_families.data();
  hrx::runtime::mem_pool_ptr pool;
  ASSERT_TRUE(Check(
      hrx_slab_pool_create(group, source_scope, nullptr, pool.for_output())));
  const hrx_mem_pool_t source_pool = pool;
  hrx_virtual_buffer_capabilities_t capabilities;
  ASSERT_TRUE(Check(hrx_virtual_buffer_query_capabilities(
      group, scope, {1, &source_pool}, &capabilities)));
  const auto requirements = hrx_mem_pool_query_mapping_requirements(pool);
  const size_t size = std::max({capabilities.reservation_granularity,
                                capabilities.mapping_granularity,
                                requirements.recommended_granularity});
  const hrx_virtual_buffer_options_t options = {
      .byte_length = size,
      .mapping_granularity = size,
  };
  hrx::runtime::buffer_ptr backing;
  hrx::runtime::buffer_ptr buffer;
  ASSERT_TRUE(Check(hrx_virtual_buffer_create(group, scope, {1, &source_pool},
                                              &options, buffer.for_output())));
  hrx::runtime::semaphore_ptr ready;
  ASSERT_TRUE(Check(hrx_semaphore_create(devices[0], 0, ready.for_output())));
  uint64_t value = 1;
  hrx_semaphore_t semaphore = ready;
  const hrx_semaphore_list_t signal = {&semaphore, &value, 1};
  const hrx_pool_reservation_request_t request = {size, 0};
  ASSERT_TRUE(Check(hrx_queue_alloca(hrx_device_get_queue(devices[0], 0, 0),
                                     nullptr, &signal, pool, 1, &request,
                                     backing.for_output())));
  uint64_t allocation_value = value++;
  const hrx_semaphore_list_t allocated = {&semaphore, &allocation_value, 1};
  const hrx_memory_mapping_t mapping = {
      .target = buffer,
      .byte_length = size,
      .source = backing,
      .access = HRX_MEMORY_ACCESS_ALL,
      .operation = HRX_MEMORY_MAPPING_OPERATION_BIND,
  };
  ASSERT_TRUE(
      Check(hrx_queue_map(hrx_device_get_queue(devices[0], 0, 0), &allocated,
                          &signal, 1, &mapping, HRX_QUEUE_MAP_FLAG_NONE)));
  ASSERT_TRUE(Check(hrx_semaphore_wait(ready, value, UINT64_MAX)));
  ready.reset();
  hrx_memory_scope_t program_scopes[2];
  for (size_t i = 0; i < 2; ++i) {
    ASSERT_TRUE(Check(hrx_device_query_queue_family_memory_scope(
        devices[i], 0, HRX_MEMORY_SITE_PROGRAM, &program_scopes[i])));
  }

  pool.reset();

  // A buffer keeps every participant and its address contract alive. Neither
  // the first participant nor a caller-held group reference owns its identity.
  group.reset();
  runtime.reset();
  uint64_t addresses[2];
  for (size_t i = 0; i < 2; ++i) {
    ASSERT_TRUE(Check(hrx_buffer_query_address(
        buffer, program_scopes[i], HRX_BUFFER_INTERFACE_HOST, &addresses[i])));
    ASSERT_NE(addresses[i], 0u);
  }
  EXPECT_EQ(addresses[0], addresses[1]);  // These two task devices share a VM.

  hrx::runtime::hrx_ptr<hrx_graph_t, hrx_graph_retain, hrx_graph_release> graph;
  ASSERT_TRUE(Check(hrx_graph_create(devices[1], 0, graph.for_output())));
  const hrx_graph_fill_buffer_node_attrs_t fill = {
      .dst = {buffer, 0, 256},
      .pattern = 0xA5A5A5A5u,
      .pattern_size = 4,
  };
  hrx_graph_node_t node = nullptr;
  ASSERT_TRUE(
      Check(hrx_graph_add_fill_buffer_node(graph, nullptr, 0, &fill, &node)));
  hrx::runtime::hrx_ptr<hrx_graph_exec_t, hrx_graph_exec_retain,
                        hrx_graph_exec_release>
      executable;
  hrx_queue_t queue = hrx_device_get_queue(devices[1], 0, 0);
  ASSERT_TRUE(
      Check(hrx_graph_instantiate(graph, queue, 0, executable.for_output())));
  hrx::runtime::stream_ptr stream;
  ASSERT_TRUE(Check(hrx_stream_create(queue, 0, stream.for_output())));
  ASSERT_TRUE(Check(hrx_graph_exec_launch(executable, stream)));
  ASSERT_TRUE(Check(hrx_stream_synchronize(stream)));
  uint32_t output[64] = {};
  ASSERT_TRUE(Check(hrx_synchronous_d2h(hrx_device_get_queue(devices[0], 0, 0),
                                        buffer, 0, output, sizeof(output))));
  for (uint32_t element : output) {
    EXPECT_EQ(0xA5A5A5A5u, element);
  }
  stream.reset();
  executable.reset();
  graph.reset();
  ASSERT_TRUE(Check(hrx_virtual_buffer_detach(buffer, {})));
  buffer.reset();
  backing.reset();
}

TEST_F(DeviceGroupLifetimeTest, ViewKeepsAllocationAndServicesAlive) {
  hrx::runtime::buffer_ptr buffer;
  ASSERT_TRUE(Check(hrx_allocator_allocate_buffer(hrx_device_allocator(device_),
                                                  kHostBufferParams, 256,
                                                  buffer.for_output())));
  ASSERT_TRUE(buffer);
  hrx_buffer_t borrowed_buffer = buffer;
  const int64_t shape[] = {8, 8};
  hrx::runtime::buffer_view_ptr view;
  ASSERT_TRUE(Check(hrx_buffer_view_create(
      buffer, 2, shape, HRX_ELEMENT_TYPE_INT_32,
      HRX_ENCODING_TYPE_DENSE_ROW_MAJOR, view.for_output())));
  ASSERT_TRUE(view);
  auto retained_view = view;
  view.reset();
  buffer.reset();
  ReleaseConstructionOwners();

  hrx::runtime::semaphore_ptr done;
  ASSERT_TRUE(Check(hrx_semaphore_create(device_, 0, done.for_output())));
  ASSERT_TRUE(done);
  uint64_t value = 1;
  hrx_semaphore_t signal = done;
  const hrx_semaphore_list_t signals = {&signal, &value, 1};
  const uint32_t pattern = 0x87654321u;
  ASSERT_TRUE(Check(hrx_queue_fill(hrx_device_get_queue(device_, 0, 0), nullptr,
                                   &signals, borrowed_buffer, 0, 256, &pattern,
                                   sizeof(pattern))));
  ASSERT_TRUE(Check(hrx_semaphore_wait(done, value, UINT64_MAX)));
  done.reset();

  void* data = nullptr;
  ASSERT_TRUE(
      Check(hrx_buffer_map(borrowed_buffer, HRX_MAP_READ, 0, 256, &data)));
  for (size_t i = 0; i < 64; ++i) {
    EXPECT_EQ(static_cast<const uint32_t*>(data)[i], pattern);
  }
  ASSERT_TRUE(Check(hrx_buffer_unmap(borrowed_buffer)));
  size_t rank = 0;
  ASSERT_TRUE(Check(hrx_buffer_view_rank(retained_view, &rank)));
  EXPECT_EQ(rank, 2u);
  int64_t dimension = 0;
  ASSERT_TRUE(Check(hrx_buffer_view_dim(retained_view, 1, &dimension)));
  EXPECT_EQ(dimension, 8);
}

TEST_F(DeviceGroupLifetimeTest, ExtendedFenceRetainsDistinctGroups) {
  hrx::runtime::device_group_ptr second_group;
  const hrx_device_create_params_t task = {.uri = "task://"};
  ASSERT_TRUE(Check(
      hrx_device_group_create(runtime_, 1, &task, second_group.for_output())));
  ASSERT_TRUE(second_group);
  hrx::runtime::semaphore_ptr first;
  hrx::runtime::semaphore_ptr second;
  ASSERT_TRUE(Check(hrx_semaphore_create(device_, 0, first.for_output())));
  ASSERT_TRUE(Check(hrx_semaphore_create(
      hrx_device_group_get_device(second_group, 0), 0, second.for_output())));
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  hrx_semaphore_t borrowed_first = first;
  hrx_semaphore_t borrowed_second = second;
  hrx::runtime::fence_ptr source;
  ASSERT_TRUE(Check(hrx_fence_create(2, source.for_output())));
  ASSERT_TRUE(source);
  ASSERT_TRUE(Check(hrx_fence_insert(source, first, 3)));
  ASSERT_TRUE(Check(hrx_fence_insert(source, second, 5)));
  hrx::runtime::fence_ptr target;
  ASSERT_TRUE(Check(hrx_fence_create(2, target.for_output())));
  ASSERT_TRUE(target);
  ASSERT_TRUE(Check(hrx_fence_insert(target, first, 4)));
  ASSERT_TRUE(Check(hrx_fence_extend(target, source)));
  ASSERT_TRUE(Check(hrx_fence_extend(target, target)));
  auto retained = target;
  target.reset();
  source.reset();
  first.reset();
  second.reset();
  second_group.reset();
  ReleaseConstructionOwners();

  ASSERT_TRUE(Check(hrx_semaphore_signal(borrowed_first, 3)));
  ASSERT_TRUE(Check(hrx_semaphore_signal(borrowed_second, 5)));
  hrx_status_t status = hrx_fence_wait(retained, 0);
  EXPECT_EQ(hrx_status_code(status), HRX_STATUS_DEADLINE_EXCEEDED);
  hrx_status_ignore(status);
  ASSERT_TRUE(Check(hrx_semaphore_signal(borrowed_first, 4)));
  ASSERT_TRUE(Check(hrx_fence_wait(retained, 0)));
}

TEST_F(DeviceGroupLifetimeTest, PartialFenceExtensionKeepsInsertedOwners) {
  hrx::runtime::semaphore_ptr timelines[3];
  for (auto& timeline : timelines) {
    ASSERT_TRUE(Check(hrx_semaphore_create(device_, 0, timeline.for_output())));
    ASSERT_TRUE(timeline);
  }
  hrx::runtime::fence_ptr source;
  ASSERT_TRUE(Check(hrx_fence_create(2, source.for_output())));
  ASSERT_TRUE(source);
  ASSERT_TRUE(Check(hrx_fence_insert(source, timelines[1], 7)));
  ASSERT_TRUE(Check(hrx_fence_insert(source, timelines[2], 9)));
  hrx::runtime::fence_ptr target;
  ASSERT_TRUE(Check(hrx_fence_create(2, target.for_output())));
  ASSERT_TRUE(target);
  ASSERT_TRUE(Check(hrx_fence_insert(target, timelines[0], 3)));
  hrx_status_t status = hrx_fence_extend(target, source);
  EXPECT_EQ(hrx_status_code(status), HRX_STATUS_OUT_OF_MEMORY);
  hrx_status_ignore(status);
  // A repeated insertion still succeeds at capacity and takes the maximum.
  ASSERT_TRUE(Check(hrx_fence_insert(target, timelines[1], 8)));
  hrx_semaphore_t borrowed = timelines[1];
  source.reset();
  for (auto& timeline : timelines) {
    timeline.reset();
  }
  ReleaseConstructionOwners();

  ASSERT_TRUE(Check(hrx_fence_signal(target)));
  uint64_t value = 0;
  ASSERT_TRUE(Check(hrx_semaphore_query(borrowed, &value)));
  EXPECT_EQ(value, 8u);
  ASSERT_TRUE(Check(hrx_fence_wait(target, 0)));
}

TEST_F(DeviceGroupLifetimeTest, SingleTimepointFenceRetainsTimeline) {
  hrx::runtime::semaphore_ptr timeline;
  ASSERT_TRUE(Check(hrx_semaphore_create(device_, 0, timeline.for_output())));
  ASSERT_TRUE(timeline);
  hrx::runtime::fence_ptr fence;
  ASSERT_TRUE(Check(hrx_fence_create_at(timeline, 19, fence.for_output())));
  ASSERT_TRUE(fence);
  timeline.reset();
  ReleaseConstructionOwners();
  ASSERT_TRUE(Check(hrx_fence_signal(fence)));
  ASSERT_TRUE(Check(hrx_fence_wait(fence, 0)));
}

TEST_F(DeviceGroupLifetimeTest, GraphKeepsRecordingAndExecutionOwnersAlive) {
  alignas(64) uint32_t values[16] = {};
  hrx::runtime::buffer_ptr buffer;
  ASSERT_TRUE(Check(hrx_allocator_import_buffer(
      hrx_device_allocator(device_), kHostBufferParams, values, sizeof(values),
      buffer.for_output())));
  ASSERT_TRUE(buffer);
  hrx::runtime::hrx_ptr<hrx_graph_t, hrx_graph_retain, hrx_graph_release> graph;
  ASSERT_TRUE(Check(hrx_graph_create(device_, 0, graph.for_output())));
  ASSERT_TRUE(graph);
  const hrx_graph_fill_buffer_node_attrs_t fill = {
      .dst = {buffer, 0, sizeof(values)},
      .pattern = 0xA5A5A5A5u,
      .pattern_size = 4,
  };
  hrx_graph_node_t node = nullptr;
  ASSERT_TRUE(
      Check(hrx_graph_add_fill_buffer_node(graph, nullptr, 0, &fill, &node)));
  buffer.reset();
  ReleaseConstructionOwners();

  // The template is now the sole owner of the device arena. Instantiation
  // transfers that ownership through the executable's retained template.
  hrx::runtime::hrx_ptr<hrx_graph_exec_t, hrx_graph_exec_retain,
                        hrx_graph_exec_release>
      executable;
  ASSERT_TRUE(Check(hrx_graph_instantiate(
      graph, hrx_device_get_queue(device_, 0, 0), 0, executable.for_output())));
  ASSERT_TRUE(executable);
  graph.reset();
  hrx::runtime::stream_ptr stream;
  ASSERT_TRUE(Check(hrx_stream_create(hrx_device_get_queue(device_, 0, 0), 0,
                                      stream.for_output())));
  ASSERT_TRUE(stream);
  ASSERT_TRUE(Check(hrx_graph_exec_launch(executable, stream)));
  ASSERT_TRUE(Check(hrx_stream_synchronize(stream)));
  stream.reset();
  for (uint32_t value : values) {
    EXPECT_EQ(value, 0xA5A5A5A5u);
  }
}

}  // namespace
