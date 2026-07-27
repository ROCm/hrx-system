// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/topology_builder.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal {
namespace {

using ::iree::testing::status::IsOk;
using ::iree::testing::status::StatusIs;
using ::testing::Eq;
using ::testing::Ne;

static iree_hal_topology_t* CreateSingleDeviceTopology() {
  iree_hal_topology_builder_t builder;
  iree_hal_topology_builder_initialize(&builder, 1);
  iree_hal_topology_t* topology = NULL;
  IREE_CHECK_OK(iree_hal_topology_builder_finalize(
      &builder, iree_allocator_system(), &topology));
  return topology;
}

typedef uint32_t TestDeviceSpecFlags;
enum TestDeviceSpecFlagBits {
  TEST_DEVICE_SPEC_FLAG_NONE = 0u,
  TEST_DEVICE_SPEC_FLAG_NUMA_NODE = 1u << 0,
  TEST_DEVICE_SPEC_FLAG_UUID = 1u << 1,
  TEST_DEVICE_SPEC_FLAG_EXTERNAL_BUFFER_HANDLES = 1u << 2,
  TEST_DEVICE_SPEC_FLAG_EXTERNAL_TIMEPOINT_HANDLES = 1u << 3,
};

static iree_hal_uuid_t MakeTestUuid(uint8_t value) {
  iree_hal_uuid_t uuid = {{0}};
  uuid.bytes[15] = value;
  return uuid;
}

static iree_hal_device_spec_t* CreateTestDeviceSpec(
    const char* driver_id, const char* backend_id, uint32_t logical_ordinal,
    TestDeviceSpecFlags flags, uint32_t numa_node, iree_hal_uuid_t uuid) {
  iree_hal_physical_device_identity_flags_t physical_identity_flags =
      IREE_HAL_PHYSICAL_DEVICE_IDENTITY_FLAG_NONE;
  if (flags & TEST_DEVICE_SPEC_FLAG_NUMA_NODE) {
    physical_identity_flags |= IREE_HAL_PHYSICAL_DEVICE_IDENTITY_FLAG_NUMA_NODE;
  }
  if (flags & TEST_DEVICE_SPEC_FLAG_UUID) {
    physical_identity_flags |= IREE_HAL_PHYSICAL_DEVICE_IDENTITY_FLAG_UUID;
  }

  iree_hal_physical_device_spec_t physical_device = {
      /*.identity=*/
      {
          /*.display_name=*/iree_make_cstring_view("test physical device"),
          /*.backend_path=*/iree_make_cstring_view("test://physical"),
          /*.vendor_id=*/0,
          /*.device_id=*/0,
          /*.revision_id=*/0,
          /*.uuid=*/uuid,
          /*.pci=*/{0, 0, 0, 0},
          /*.numa=*/{numa_node},
          /*.flags=*/physical_identity_flags,
      },
      /*.physical_ordinal=*/logical_ordinal,
      /*.partition_ordinal=*/0,
      /*.partition_count=*/1,
      /*.physical_device_affinity=*/1ull << logical_ordinal,
  };
  iree_hal_device_identity_spec_t identity = {
      /*.logical_device_id=*/iree_make_cstring_view(driver_id),
      /*.display_name=*/iree_make_cstring_view("test logical device"),
      /*.driver_id=*/iree_make_cstring_view(driver_id),
      /*.driver_version=*/iree_make_cstring_view("test"),
      /*.backend_id=*/iree_make_cstring_view(backend_id),
      /*.device_path=*/iree_make_cstring_view("test://logical"),
      /*.vendor_name=*/iree_make_cstring_view("test"),
      /*.vendor_id=*/0,
      /*.device_id=*/0,
      /*.revision_id=*/0,
      /*.logical_ordinal=*/logical_ordinal,
      /*.physical_device_count=*/1,
      /*.physical_devices=*/&physical_device,
      /*.flags=*/IREE_HAL_DEVICE_IDENTITY_FLAG_NONE,
  };

  iree_hal_external_buffer_handle_spec_t external_buffer_handles[1] = {
      {
          /*.handle_type_mask=*/IREE_HAL_TOPOLOGY_HANDLE_TYPE_DMA_BUF,
          /*.direction_flags=*/IREE_HAL_EXTERNAL_HANDLE_DIRECTION_FLAG_IMPORT |
              IREE_HAL_EXTERNAL_HANDLE_DIRECTION_FLAG_EXPORT,
          /*.allowed_buffer_usage=*/IREE_HAL_BUFFER_USAGE_DEFAULT,
          /*.allowed_memory_access=*/IREE_HAL_MEMORY_ACCESS_ALL,
          /*.compatible_memory_type_mask=*/UINT32_MAX,
          /*.flags=*/IREE_HAL_EXTERNAL_HANDLE_CAPABILITY_FLAG_CROSS_PROCESS,
      },
  };
  iree_hal_device_memory_spec_t memory = {
      /*.heap_count=*/0,
      /*.heaps=*/NULL,
      /*.memory_type_count=*/0,
      /*.memory_types=*/NULL,
      /*.external_buffer_handle_count=*/
      (flags & TEST_DEVICE_SPEC_FLAG_EXTERNAL_BUFFER_HANDLES) ? 1u : 0u,
      /*.external_buffer_handles=*/external_buffer_handles,
      /*.flags=*/IREE_HAL_DEVICE_MEMORY_SPEC_FLAG_NONE,
  };
  iree_hal_external_timepoint_handle_spec_t external_timepoint_handles[1] = {
      {
          /*.handle_type=*/IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_HIP_EVENT,
          /*.direction_flags=*/IREE_HAL_EXTERNAL_HANDLE_DIRECTION_FLAG_IMPORT |
              IREE_HAL_EXTERNAL_HANDLE_DIRECTION_FLAG_EXPORT,
          /*.compatibility=*/IREE_HAL_SEMAPHORE_COMPATIBILITY_DEVICE_WAIT,
          /*.flags=*/IREE_HAL_EXTERNAL_HANDLE_CAPABILITY_FLAG_NONE,
      },
  };
  iree_hal_device_queue_spec_t queues = {
      /*.family_count=*/0,
      /*.families=*/NULL,
      /*.external_timepoint_handle_count=*/
      (flags & TEST_DEVICE_SPEC_FLAG_EXTERNAL_TIMEPOINT_HANDLES) ? 1u : 0u,
      /*.external_timepoint_handles=*/external_timepoint_handles,
      /*.flags=*/IREE_HAL_DEVICE_QUEUE_SPEC_FLAG_NONE,
  };
  iree_hal_device_spec_params_t params = {
      /*.identity=*/&identity,
      /*.memory=*/&memory,
      /*.virtual_memory=*/NULL,
      /*.queues=*/&queues,
  };
  iree_hal_device_spec_t* device_spec = NULL;
  IREE_CHECK_OK(iree_hal_device_spec_create(&params, iree_allocator_system(),
                                            &device_spec));
  return device_spec;
}

//===----------------------------------------------------------------------===//
// Scheduling word bitfield overlap tests
//===----------------------------------------------------------------------===//

// Verifies that scheduling word bitfields don't overlap and that all fields
// can be independently set without corrupting other fields.
TEST(TopologyEdge, SchedulingWordBitfieldOverlap) {
  iree_hal_topology_edge_scheduling_word_t lo = 0;

  // Set each field to its maximum value.
  lo = iree_hal_topology_edge_set_wait_mode(
      lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_NONE);
  lo = iree_hal_topology_edge_set_signal_mode(
      lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_NONE);
  lo = iree_hal_topology_edge_set_buffer_read_mode_noncoherent(
      lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_NONE);
  lo = iree_hal_topology_edge_set_buffer_write_mode_noncoherent(
      lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_NONE);
  lo = iree_hal_topology_edge_set_buffer_read_mode_coherent(
      lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_NONE);
  lo = iree_hal_topology_edge_set_buffer_write_mode_coherent(
      lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_NONE);
  lo = iree_hal_topology_edge_set_capability_flags(lo, 0xFFFF);  // 16 bits
  lo = iree_hal_topology_edge_set_wait_cost(lo, 15);
  lo = iree_hal_topology_edge_set_signal_cost(lo, 15);
  lo = iree_hal_topology_edge_set_copy_cost(lo, 15);
  lo = iree_hal_topology_edge_set_latency_class(lo, 15);
  lo = iree_hal_topology_edge_set_numa_distance(lo, 15);
  lo = iree_hal_topology_edge_set_link_class(lo, 7);  // 3 bits

  // Verify all fields retained their values.
  EXPECT_EQ(iree_hal_topology_edge_wait_mode(lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_NONE);
  EXPECT_EQ(iree_hal_topology_edge_signal_mode(lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_NONE);
  EXPECT_EQ(iree_hal_topology_edge_buffer_read_mode_noncoherent(lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_NONE);
  EXPECT_EQ(iree_hal_topology_edge_buffer_write_mode_noncoherent(lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_NONE);
  EXPECT_EQ(iree_hal_topology_edge_buffer_read_mode_coherent(lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_NONE);
  EXPECT_EQ(iree_hal_topology_edge_buffer_write_mode_coherent(lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_NONE);
  EXPECT_EQ(iree_hal_topology_edge_capability_flags(lo), 0xFFFF);
  EXPECT_EQ(iree_hal_topology_edge_wait_cost(lo), 15);
  EXPECT_EQ(iree_hal_topology_edge_signal_cost(lo), 15);
  EXPECT_EQ(iree_hal_topology_edge_copy_cost(lo), 15);
  EXPECT_EQ(iree_hal_topology_edge_latency_class(lo), 15);
  EXPECT_EQ(iree_hal_topology_edge_numa_distance(lo), 15);
  EXPECT_EQ(iree_hal_topology_edge_link_class(lo), 7);
}

// Verifies that interop word bitfields don't overlap.
TEST(TopologyEdge, InteropWordBitfieldOverlap) {
  iree_hal_topology_edge_interop_word_t hi = 0;

  // Set each field to its maximum value.
  hi = iree_hal_topology_edge_set_semaphore_import_timepoint_types(hi, 0xFFFF);
  hi = iree_hal_topology_edge_set_semaphore_export_timepoint_types(hi, 0xFFFF);
  hi = iree_hal_topology_edge_set_buffer_import_types(hi, 0xFF);
  hi = iree_hal_topology_edge_set_buffer_export_types(hi, 0xFF);

  // Verify all fields retained their values.
  EXPECT_EQ(iree_hal_topology_edge_semaphore_import_timepoint_types(hi),
            0xFFFF);
  EXPECT_EQ(iree_hal_topology_edge_semaphore_export_timepoint_types(hi),
            0xFFFF);
  EXPECT_EQ(iree_hal_topology_edge_buffer_import_types(hi), 0xFF);
  EXPECT_EQ(iree_hal_topology_edge_buffer_export_types(hi), 0xFF);
}

// Verifies that setting each scheduling field independently doesn't affect
// others.
TEST(TopologyEdge, SchedulingWordBitfieldIndependence) {
  iree_hal_topology_edge_scheduling_word_t lo = 0;

  // Set wait mode and verify only it changes.
  lo = iree_hal_topology_edge_set_wait_mode(
      lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  EXPECT_EQ(iree_hal_topology_edge_wait_mode(lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  EXPECT_EQ(iree_hal_topology_edge_signal_mode(lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);

  // Set signal cost and verify wait mode unchanged.
  lo = iree_hal_topology_edge_set_signal_cost(lo, 13);
  EXPECT_EQ(iree_hal_topology_edge_signal_cost(lo), 13);
  EXPECT_EQ(iree_hal_topology_edge_wait_mode(lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
}

// Verifies that setting interop fields independently doesn't affect others.
TEST(TopologyEdge, InteropWordBitfieldIndependence) {
  iree_hal_topology_edge_interop_word_t hi = 0;

  // Set semaphore import timepoint types.
  hi = iree_hal_topology_edge_set_semaphore_import_timepoint_types(
      hi, IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_MASK_CUDA_EVENT |
              IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_MASK_HIP_EVENT);
  EXPECT_EQ(iree_hal_topology_edge_semaphore_import_timepoint_types(hi),
            IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_MASK_CUDA_EVENT |
                IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_MASK_HIP_EVENT);
  EXPECT_EQ(iree_hal_topology_edge_semaphore_export_timepoint_types(hi),
            IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_MASK_NONE);
  EXPECT_EQ(iree_hal_topology_edge_buffer_import_types(hi), 0);

  // Set buffer export types and verify semaphore import unchanged.
  hi = iree_hal_topology_edge_set_buffer_export_types(
      hi, IREE_HAL_TOPOLOGY_HANDLE_TYPE_DMA_BUF);
  EXPECT_EQ(iree_hal_topology_edge_buffer_export_types(hi),
            IREE_HAL_TOPOLOGY_HANDLE_TYPE_DMA_BUF);
  EXPECT_EQ(iree_hal_topology_edge_semaphore_import_timepoint_types(hi),
            IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_MASK_CUDA_EVENT |
                IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_MASK_HIP_EVENT);
}

//===----------------------------------------------------------------------===//
// Edge construction tests
//===----------------------------------------------------------------------===//

// Tests creation of a self-edge.
TEST(TopologyEdge, CreateSelf) {
  iree_hal_topology_edge_t edge = iree_hal_topology_edge_make_self();

  // Self-edges should have NATIVE mode for all operations.
  EXPECT_EQ(iree_hal_topology_edge_wait_mode(edge.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
  EXPECT_EQ(iree_hal_topology_edge_signal_mode(edge.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
  EXPECT_EQ(iree_hal_topology_edge_buffer_read_mode_noncoherent(edge.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
  EXPECT_EQ(iree_hal_topology_edge_buffer_write_mode_noncoherent(edge.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
  EXPECT_EQ(iree_hal_topology_edge_buffer_read_mode_coherent(edge.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
  EXPECT_EQ(iree_hal_topology_edge_buffer_write_mode_coherent(edge.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);

  // Self-edges should have zero cost.
  EXPECT_EQ(iree_hal_topology_edge_wait_cost(edge.lo), 0);
  EXPECT_EQ(iree_hal_topology_edge_signal_cost(edge.lo), 0);
  EXPECT_EQ(iree_hal_topology_edge_copy_cost(edge.lo), 0);
  EXPECT_EQ(iree_hal_topology_edge_latency_class(edge.lo), 0);
  EXPECT_EQ(iree_hal_topology_edge_numa_distance(edge.lo), 0);

  // Self-edges have all capability flags set.
  iree_hal_topology_capability_t expected_caps =
      IREE_HAL_TOPOLOGY_CAPABILITY_SAME_RUNTIME_DOMAIN |
      IREE_HAL_TOPOLOGY_CAPABILITY_UNIFIED_MEMORY |
      IREE_HAL_TOPOLOGY_CAPABILITY_PEER_COHERENT |
      IREE_HAL_TOPOLOGY_CAPABILITY_HOST_COHERENT |
      IREE_HAL_TOPOLOGY_CAPABILITY_P2P_COPY |
      IREE_HAL_TOPOLOGY_CAPABILITY_CONCURRENT_SAFE |
      IREE_HAL_TOPOLOGY_CAPABILITY_ATOMIC_DEVICE |
      IREE_HAL_TOPOLOGY_CAPABILITY_ATOMIC_SYSTEM |
      IREE_HAL_TOPOLOGY_CAPABILITY_TIMELINE_SEMAPHORE |
      IREE_HAL_TOPOLOGY_CAPABILITY_SHARED_VIRTUAL_ADDRESS;
  EXPECT_EQ(iree_hal_topology_edge_capability_flags(edge.lo), expected_caps);

  // Self-edges use SAME_DIE link class.
  EXPECT_EQ(iree_hal_topology_edge_link_class(edge.lo),
            IREE_HAL_TOPOLOGY_LINK_CLASS_SAME_DIE);

  // Self-edges require no external semaphore handles. Native synchronization is
  // represented by the scheduling word.
  EXPECT_EQ(iree_hal_topology_edge_semaphore_import_timepoint_types(edge.hi),
            IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_MASK_NONE);
  EXPECT_EQ(iree_hal_topology_edge_semaphore_export_timepoint_types(edge.hi),
            IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_MASK_NONE);
  EXPECT_EQ(iree_hal_topology_edge_buffer_import_types(edge.hi),
            IREE_HAL_TOPOLOGY_HANDLE_TYPE_NATIVE);
  EXPECT_EQ(iree_hal_topology_edge_buffer_export_types(edge.hi),
            IREE_HAL_TOPOLOGY_HANDLE_TYPE_NATIVE);
}

// Tests creation of a conservative host-staged edge.
TEST(TopologyEdge, CreateHostStaged) {
  iree_hal_topology_edge_t edge = iree_hal_topology_edge_make_host_staged();

  // Host-staged edges use host-mediated COPY for semaphores and buffers.
  EXPECT_EQ(iree_hal_topology_edge_wait_mode(edge.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  EXPECT_EQ(iree_hal_topology_edge_signal_mode(edge.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  EXPECT_EQ(iree_hal_topology_edge_buffer_read_mode_noncoherent(edge.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  EXPECT_EQ(iree_hal_topology_edge_buffer_write_mode_noncoherent(edge.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  EXPECT_EQ(iree_hal_topology_edge_buffer_read_mode_coherent(edge.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  EXPECT_EQ(iree_hal_topology_edge_buffer_write_mode_coherent(edge.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);

  // Host-staged paths have conservative costs.
  EXPECT_EQ(iree_hal_topology_edge_wait_cost(edge.lo), 10);
  EXPECT_EQ(iree_hal_topology_edge_signal_cost(edge.lo), 10);
  EXPECT_EQ(iree_hal_topology_edge_copy_cost(edge.lo), 13);
  EXPECT_EQ(iree_hal_topology_edge_latency_class(edge.lo), 11);
  EXPECT_EQ(iree_hal_topology_edge_numa_distance(edge.lo), 0);

  EXPECT_EQ(iree_hal_topology_edge_capability_flags(edge.lo),
            IREE_HAL_TOPOLOGY_CAPABILITY_NONE);
  EXPECT_EQ(iree_hal_topology_edge_link_class(edge.lo),
            IREE_HAL_TOPOLOGY_LINK_CLASS_HOST_STAGED);

  EXPECT_EQ(iree_hal_topology_edge_semaphore_import_timepoint_types(edge.hi),
            IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_MASK_NONE);
  EXPECT_EQ(iree_hal_topology_edge_buffer_import_types(edge.hi), 0);
}

//===----------------------------------------------------------------------===//
// Device spec projection tests
//===----------------------------------------------------------------------===//

TEST(TopologyEdge, SameBackendSpecDoesNotImplyNativeSynchronization) {
  iree_hal_device_spec_t* spec_a = CreateTestDeviceSpec(
      "hip", "hsa", 0, TEST_DEVICE_SPEC_FLAG_NONE, 0, MakeTestUuid(0));
  iree_hal_device_spec_t* spec_b = CreateTestDeviceSpec(
      "hip", "hsa", 1, TEST_DEVICE_SPEC_FLAG_NONE, 0, MakeTestUuid(0));

  iree_hal_topology_edge_t edge =
      iree_hal_topology_edge_from_device_specs(spec_a, spec_b);
  EXPECT_EQ(iree_hal_topology_edge_wait_mode(edge.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  EXPECT_EQ(iree_hal_topology_edge_signal_mode(edge.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  EXPECT_EQ(iree_hal_topology_edge_buffer_read_mode_noncoherent(edge.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  EXPECT_EQ(iree_hal_topology_edge_buffer_write_mode_coherent(edge.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  iree_hal_topology_capability_t capabilities =
      iree_hal_topology_edge_capability_flags(edge.lo);
  EXPECT_FALSE(capabilities & IREE_HAL_TOPOLOGY_CAPABILITY_SAME_RUNTIME_DOMAIN);
  EXPECT_FALSE(capabilities & IREE_HAL_TOPOLOGY_CAPABILITY_TIMELINE_SEMAPHORE);
  EXPECT_FALSE(capabilities & IREE_HAL_TOPOLOGY_CAPABILITY_P2P_COPY);
  EXPECT_FALSE(capabilities & IREE_HAL_TOPOLOGY_CAPABILITY_PEER_COHERENT);
  EXPECT_FALSE(capabilities & IREE_HAL_TOPOLOGY_CAPABILITY_ATOMIC_SYSTEM);
  EXPECT_EQ(iree_hal_topology_edge_link_class(edge.lo),
            IREE_HAL_TOPOLOGY_LINK_CLASS_HOST_STAGED);
  EXPECT_EQ(iree_hal_topology_edge_semaphore_import_timepoint_types(edge.hi),
            IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_MASK_NONE);
  EXPECT_EQ(iree_hal_topology_edge_buffer_import_types(edge.hi),
            IREE_HAL_TOPOLOGY_HANDLE_TYPE_NONE);

  iree_hal_device_spec_release(spec_b);
  iree_hal_device_spec_release(spec_a);
}

TEST(TopologyEdge, RefineSameRuntimeDomainUsesNativeSynchronizationOnly) {
  iree_hal_topology_edge_t edge = iree_hal_topology_edge_make_host_staged();

  iree_hal_topology_edge_refine_same_runtime_domain(&edge);
  EXPECT_EQ(iree_hal_topology_edge_wait_mode(edge.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
  EXPECT_EQ(iree_hal_topology_edge_signal_mode(edge.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
  EXPECT_EQ(iree_hal_topology_edge_buffer_read_mode_noncoherent(edge.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  EXPECT_EQ(iree_hal_topology_edge_buffer_write_mode_coherent(edge.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  iree_hal_topology_capability_t capabilities =
      iree_hal_topology_edge_capability_flags(edge.lo);
  EXPECT_TRUE(capabilities & IREE_HAL_TOPOLOGY_CAPABILITY_SAME_RUNTIME_DOMAIN);
  EXPECT_TRUE(capabilities & IREE_HAL_TOPOLOGY_CAPABILITY_TIMELINE_SEMAPHORE);
  EXPECT_FALSE(capabilities & IREE_HAL_TOPOLOGY_CAPABILITY_P2P_COPY);
  EXPECT_FALSE(capabilities & IREE_HAL_TOPOLOGY_CAPABILITY_PEER_COHERENT);
  EXPECT_EQ(iree_hal_topology_edge_semaphore_import_timepoint_types(edge.hi),
            IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_MASK_NONE);
  EXPECT_EQ(iree_hal_topology_edge_buffer_import_types(edge.hi),
            IREE_HAL_TOPOLOGY_HANDLE_TYPE_NONE);
}

TEST(TopologyEdge, UnknownNumaDoesNotProjectDistanceFromNodeZero) {
  iree_hal_device_spec_t* spec_a = CreateTestDeviceSpec(
      "hip", "hsa", 0, TEST_DEVICE_SPEC_FLAG_NUMA_NODE, 1, MakeTestUuid(0));
  iree_hal_device_spec_t* spec_b = CreateTestDeviceSpec(
      "vulkan", "vulkan", 1, TEST_DEVICE_SPEC_FLAG_NONE, 0, MakeTestUuid(0));

  iree_hal_topology_edge_t edge =
      iree_hal_topology_edge_from_device_specs(spec_a, spec_b);
  EXPECT_EQ(iree_hal_topology_edge_numa_distance(edge.lo), 0);

  iree_hal_device_spec_release(spec_b);
  iree_hal_device_spec_release(spec_a);
}

TEST(TopologyEdge, ExternalBufferHandlesUseImportModes) {
  TestDeviceSpecFlags flags = TEST_DEVICE_SPEC_FLAG_EXTERNAL_BUFFER_HANDLES;
  iree_hal_device_spec_t* spec_a =
      CreateTestDeviceSpec("vulkan", "vulkan", 0, flags, 0, MakeTestUuid(0));
  iree_hal_device_spec_t* spec_b =
      CreateTestDeviceSpec("hip", "hsa", 1, flags, 0, MakeTestUuid(0));

  iree_hal_topology_edge_t edge =
      iree_hal_topology_edge_from_device_specs(spec_a, spec_b);
  EXPECT_EQ(iree_hal_topology_edge_wait_mode(edge.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  EXPECT_EQ(iree_hal_topology_edge_buffer_read_mode_noncoherent(edge.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_IMPORT);
  EXPECT_EQ(iree_hal_topology_edge_buffer_write_mode_coherent(edge.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_IMPORT);
  EXPECT_EQ(iree_hal_topology_edge_buffer_import_types(edge.hi),
            IREE_HAL_TOPOLOGY_HANDLE_TYPE_DMA_BUF);
  EXPECT_EQ(iree_hal_topology_edge_buffer_export_types(edge.hi),
            IREE_HAL_TOPOLOGY_HANDLE_TYPE_DMA_BUF);
  EXPECT_EQ(iree_hal_topology_edge_link_class(edge.lo),
            IREE_HAL_TOPOLOGY_LINK_CLASS_OTHER);
  EXPECT_FALSE(iree_hal_topology_edge_capability_flags(edge.lo) &
               IREE_HAL_TOPOLOGY_CAPABILITY_P2P_COPY);

  iree_hal_device_spec_release(spec_b);
  iree_hal_device_spec_release(spec_a);
}

TEST(TopologyEdge, ExternalTimepointsUseImportModes) {
  TestDeviceSpecFlags flags = TEST_DEVICE_SPEC_FLAG_EXTERNAL_TIMEPOINT_HANDLES;
  iree_hal_device_spec_t* spec_a = CreateTestDeviceSpec(
      "producer", "producer", 0, flags, 0, MakeTestUuid(0));
  iree_hal_device_spec_t* spec_b = CreateTestDeviceSpec(
      "consumer", "consumer", 1, flags, 0, MakeTestUuid(0));

  iree_hal_topology_edge_t edge =
      iree_hal_topology_edge_from_device_specs(spec_a, spec_b);
  EXPECT_EQ(iree_hal_topology_edge_wait_mode(edge.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_IMPORT);
  EXPECT_EQ(iree_hal_topology_edge_signal_mode(edge.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_IMPORT);
  EXPECT_EQ(iree_hal_topology_edge_semaphore_import_timepoint_types(edge.hi),
            IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_MASK_HIP_EVENT);
  EXPECT_EQ(iree_hal_topology_edge_semaphore_export_timepoint_types(edge.hi),
            IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_MASK_HIP_EVENT);
  EXPECT_TRUE(iree_hal_topology_edge_capability_flags(edge.lo) &
              IREE_HAL_TOPOLOGY_CAPABILITY_TIMELINE_SEMAPHORE);
  EXPECT_LT(iree_hal_topology_edge_wait_cost(edge.lo), 10);

  iree_hal_device_spec_release(spec_b);
  iree_hal_device_spec_release(spec_a);
}

TEST(TopologyEdge, ExternalTimepointExportWithoutImportStaysHostStaged) {
  iree_hal_device_spec_t* spec_a = CreateTestDeviceSpec(
      "hip", "hsa", 0, TEST_DEVICE_SPEC_FLAG_EXTERNAL_TIMEPOINT_HANDLES, 0,
      MakeTestUuid(0));
  iree_hal_device_spec_t* spec_b = CreateTestDeviceSpec(
      "vulkan", "vulkan", 1, TEST_DEVICE_SPEC_FLAG_NONE, 0, MakeTestUuid(0));

  iree_hal_topology_edge_t edge =
      iree_hal_topology_edge_from_device_specs(spec_a, spec_b);
  EXPECT_EQ(iree_hal_topology_edge_wait_mode(edge.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  EXPECT_EQ(iree_hal_topology_edge_signal_mode(edge.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  EXPECT_EQ(iree_hal_topology_edge_semaphore_import_timepoint_types(edge.hi),
            IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_MASK_NONE);
  EXPECT_EQ(iree_hal_topology_edge_semaphore_export_timepoint_types(edge.hi),
            IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_MASK_NONE);
  EXPECT_FALSE(iree_hal_topology_edge_capability_flags(edge.lo) &
               IREE_HAL_TOPOLOGY_CAPABILITY_TIMELINE_SEMAPHORE);

  iree_hal_device_spec_release(spec_b);
  iree_hal_device_spec_release(spec_a);
}

TEST(TopologyEdge, PhysicalUuidMatchingIsPlacementNotAliasing) {
  TestDeviceSpecFlags flags = TEST_DEVICE_SPEC_FLAG_UUID;
  iree_hal_uuid_t uuid = MakeTestUuid(42);
  iree_hal_device_spec_t* spec_a =
      CreateTestDeviceSpec("vulkan", "vulkan", 0, flags, 0, uuid);
  iree_hal_device_spec_t* spec_b =
      CreateTestDeviceSpec("hip", "hsa", 1, flags, 0, uuid);

  iree_hal_topology_edge_t edge =
      iree_hal_topology_edge_from_device_specs(spec_a, spec_b);
  EXPECT_EQ(iree_hal_topology_edge_link_class(edge.lo),
            IREE_HAL_TOPOLOGY_LINK_CLASS_SAME_DIE);
  EXPECT_EQ(iree_hal_topology_edge_wait_mode(edge.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  EXPECT_EQ(iree_hal_topology_edge_buffer_read_mode_noncoherent(edge.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  EXPECT_NE(iree_hal_topology_edge_copy_cost(edge.lo), 0);
  EXPECT_EQ(iree_hal_topology_edge_capability_flags(edge.lo),
            IREE_HAL_TOPOLOGY_CAPABILITY_NONE);

  iree_hal_device_spec_release(spec_b);
  iree_hal_device_spec_release(spec_a);
}

//===----------------------------------------------------------------------===//
// Resource origin tests
//===----------------------------------------------------------------------===//

// Tests resource origin initialization.
TEST(ResourceOrigin, Initialize) {
  iree_hal_topology_edge_t edge = iree_hal_topology_edge_make_self();

  iree_hal_resource_origin_t origin = {
      /*.self_edge=*/edge.lo,
      /*.topology_index=*/3,
  };

  EXPECT_EQ(origin.self_edge, edge.lo);
  EXPECT_EQ(origin.topology_index, 3);

  // Check size is as expected (16 bytes with padding).
  EXPECT_EQ(sizeof(iree_hal_resource_origin_t), 16);
}

// Tests compatibility checking between resources.
TEST(ResourceOrigin, CompatibilityCheck) {
  iree_hal_topology_edge_t edge1 = iree_hal_topology_edge_make_self();

  iree_hal_topology_edge_scheduling_word_t lo2 = 0;
  lo2 = iree_hal_topology_edge_set_wait_mode(
      lo2, IREE_HAL_TOPOLOGY_INTEROP_MODE_IMPORT);
  lo2 = iree_hal_topology_edge_set_capability_flags(lo2, 0x42);

  iree_hal_resource_origin_t origin1 = {
      /*.self_edge=*/edge1.lo,
      /*.topology_index=*/0,
  };
  iree_hal_resource_origin_t origin2 = {
      /*.self_edge=*/lo2,
      /*.topology_index=*/1,
  };

  // Self-edges should be different.
  EXPECT_NE(origin1.self_edge, origin2.self_edge);

  // Can check compatibility by comparing capabilities.
  EXPECT_NE(iree_hal_topology_edge_capability_flags(origin1.self_edge),
            iree_hal_topology_edge_capability_flags(origin2.self_edge));
}

//===----------------------------------------------------------------------===//
// Edge formatting tests
//===----------------------------------------------------------------------===//

// Tests edge formatting for debugging.
TEST(TopologyEdge, Formatting) {
  iree_hal_topology_edge_t edge = iree_hal_topology_edge_make_self();

  iree_string_builder_t sb;
  iree_string_builder_initialize(iree_allocator_system(), &sb);
  IREE_ASSERT_OK(iree_hal_topology_edge_format(edge, &sb));
  const char* buffer = iree_string_builder_buffer(&sb);

  // Should contain mode information.
  EXPECT_NE(std::strstr(buffer, "NATIVE"), nullptr);

  // Test host-staged edge formatting.
  edge = iree_hal_topology_edge_make_host_staged();
  edge.lo = iree_hal_topology_edge_set_wait_mode(
      edge.lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  edge.lo = iree_hal_topology_edge_set_copy_cost(edge.lo, 13);

  iree_string_builder_reset(&sb);
  IREE_ASSERT_OK(iree_hal_topology_edge_format(edge, &sb));
  buffer = iree_string_builder_buffer(&sb);

  // Should contain copy mode and cost.
  EXPECT_NE(std::strstr(buffer, "COPY"), nullptr);
  EXPECT_NE(std::strstr(buffer, "copy_cost=13"), nullptr);

  iree_string_builder_deinitialize(&sb);
}

// Tests topology matrix formatting.
TEST(Topology, MatrixFormatting) {
  iree_hal_topology_builder_t builder;
  iree_hal_topology_builder_initialize(&builder, 3);

  // Set cross-device edges (self-edges already initialized).
  for (uint32_t i = 0; i < 3; ++i) {
    for (uint32_t j = 0; j < 3; ++j) {
      if (i != j) {
        iree_hal_topology_edge_t edge =
            iree_hal_topology_edge_make_host_staged();
        edge.lo = iree_hal_topology_edge_set_link_class(
            edge.lo, IREE_HAL_TOPOLOGY_LINK_CLASS_NVLINK_IF);
        IREE_ASSERT_OK(
            iree_hal_topology_builder_set_edge(&builder, i, j, edge));
      }
    }
  }

  iree_hal_topology_t* topology = NULL;
  IREE_ASSERT_OK(iree_hal_topology_builder_finalize(
      &builder, iree_allocator_system(), &topology));

  // Dump the matrix for debugging.
  iree_string_builder_t sb;
  iree_string_builder_initialize(iree_allocator_system(), &sb);
  IREE_ASSERT_OK(iree_hal_topology_dump_matrix(topology, &sb));
  printf("%.*s\n", (int)iree_string_builder_size(&sb),
         iree_string_builder_buffer(&sb));
  iree_string_builder_deinitialize(&sb);
  iree_hal_topology_destroy(topology, iree_allocator_system());
}

//===----------------------------------------------------------------------===//
// New handle type tests
//===----------------------------------------------------------------------===//

// Tests that new handle types (RDMA_MR, SHM, etc.) can be set and retrieved.
TEST(TopologyEdge, NewHandleTypes) {
  iree_hal_topology_edge_interop_word_t hi = 0;

  // Set all 8 handle type bits.
  iree_hal_topology_handle_type_t all_types =
      IREE_HAL_TOPOLOGY_HANDLE_TYPE_NATIVE |
      IREE_HAL_TOPOLOGY_HANDLE_TYPE_OPAQUE_FD |
      IREE_HAL_TOPOLOGY_HANDLE_TYPE_OPAQUE_WIN32 |
      IREE_HAL_TOPOLOGY_HANDLE_TYPE_DMA_BUF |
      IREE_HAL_TOPOLOGY_HANDLE_TYPE_RDMA_MR |
      IREE_HAL_TOPOLOGY_HANDLE_TYPE_SHM |
      IREE_HAL_TOPOLOGY_HANDLE_TYPE_METAL_IOSURFACE |
      IREE_HAL_TOPOLOGY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER;

  hi = iree_hal_topology_edge_set_buffer_import_types(hi, all_types);
  EXPECT_EQ(iree_hal_topology_edge_buffer_import_types(hi), all_types);
  EXPECT_EQ(iree_hal_topology_edge_buffer_import_types(hi), 0xFF);
}

// Tests that buffer handle types and semaphore timepoint types are separate
// fields even when they are used by the same edge.
TEST(TopologyEdge, BufferHandlesAndTimepointTypesAreIndependent) {
  iree_hal_topology_edge_interop_word_t hi = 0;

  // An RDMA-capable edge may support MR for buffers while semaphore interop
  // uses API-specific timepoint objects.
  hi = iree_hal_topology_edge_set_buffer_import_types(
      hi, IREE_HAL_TOPOLOGY_HANDLE_TYPE_RDMA_MR |
              IREE_HAL_TOPOLOGY_HANDLE_TYPE_SHM);
  hi = iree_hal_topology_edge_set_buffer_export_types(
      hi, IREE_HAL_TOPOLOGY_HANDLE_TYPE_RDMA_MR |
              IREE_HAL_TOPOLOGY_HANDLE_TYPE_SHM);
  hi = iree_hal_topology_edge_set_semaphore_import_timepoint_types(
      hi, IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_MASK_HIP_EVENT);
  hi = iree_hal_topology_edge_set_semaphore_export_timepoint_types(
      hi, IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_MASK_HIP_EVENT);

  // Verify RDMA_MR is set for buffers while semaphores retain only the
  // requested timepoint type.
  EXPECT_TRUE(iree_hal_topology_edge_buffer_import_types(hi) &
              IREE_HAL_TOPOLOGY_HANDLE_TYPE_RDMA_MR);
  EXPECT_EQ(iree_hal_topology_edge_semaphore_import_timepoint_types(hi),
            IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_MASK_HIP_EVENT);

  // Verify SHM is retained as a buffer handle.
  EXPECT_TRUE(iree_hal_topology_edge_buffer_import_types(hi) &
              IREE_HAL_TOPOLOGY_HANDLE_TYPE_SHM);
}

//===----------------------------------------------------------------------===//
// Topology info cost query tests
//===----------------------------------------------------------------------===//

// Tests iree_hal_device_topology_query_edge returns the correct edge
// when both devices share the same topology.
TEST(TopologyInfo, QueryEdgeSameTopology) {
  // Build a 2-device topology.
  iree_hal_topology_builder_t builder;
  iree_hal_topology_builder_initialize(&builder, 2);

  iree_hal_topology_edge_t cross = iree_hal_topology_edge_make_host_staged();
  cross.lo = iree_hal_topology_edge_set_copy_cost(cross.lo, 7);
  cross.lo = iree_hal_topology_edge_set_link_class(
      cross.lo, IREE_HAL_TOPOLOGY_LINK_CLASS_NVLINK_IF);
  cross.hi = iree_hal_topology_edge_set_buffer_import_types(
      cross.hi, IREE_HAL_TOPOLOGY_HANDLE_TYPE_DMA_BUF);

  IREE_ASSERT_OK(iree_hal_topology_builder_set_edge(&builder, 0, 1, cross));
  IREE_ASSERT_OK(iree_hal_topology_builder_set_edge(&builder, 1, 0, cross));

  iree_hal_topology_t* topology = NULL;
  IREE_ASSERT_OK(iree_hal_topology_builder_finalize(
      &builder, iree_allocator_system(), &topology));

  // Simulate two devices pointing at the same topology.
  iree_hal_device_topology_info_t info0 = {0};
  info0.self_edge = iree_hal_topology_query_edge(topology, 0, 0).lo;
  info0.topology_index = 0;
  info0.topology = topology;

  iree_hal_device_topology_info_t info1 = {0};
  info1.self_edge = iree_hal_topology_query_edge(topology, 1, 1).lo;
  info1.topology_index = 1;
  info1.topology = topology;

  // Query the edge from device 0 to device 1.
  iree_hal_topology_edge_t queried =
      iree_hal_device_topology_query_edge(&info0, &info1);
  EXPECT_EQ(iree_hal_topology_edge_copy_cost(queried.lo), 7);
  EXPECT_EQ(iree_hal_topology_edge_link_class(queried.lo),
            IREE_HAL_TOPOLOGY_LINK_CLASS_NVLINK_IF);
  EXPECT_EQ(iree_hal_topology_edge_buffer_import_types(queried.hi),
            IREE_HAL_TOPOLOGY_HANDLE_TYPE_DMA_BUF);

  // Self-query returns the self-edge.
  iree_hal_topology_edge_t self =
      iree_hal_device_topology_query_edge(&info0, &info0);
  EXPECT_EQ(iree_hal_topology_edge_wait_mode(self.lo),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
  EXPECT_EQ(iree_hal_topology_edge_copy_cost(self.lo), 0);
  iree_hal_topology_destroy(topology, iree_allocator_system());
}

// Tests that iree_hal_device_topology_query_edge returns empty when devices
// are in different topologies or not in any topology.
TEST(TopologyInfo, QueryEdgeDifferentTopologies) {
  iree_hal_topology_t* topology_a = CreateSingleDeviceTopology();
  iree_hal_topology_t* topology_b = CreateSingleDeviceTopology();

  iree_hal_device_topology_info_t info_a = {0};
  info_a.topology_index = 0;
  info_a.topology = topology_a;

  iree_hal_device_topology_info_t info_b = {0};
  info_b.topology_index = 0;
  info_b.topology = topology_b;

  // Different topologies: should return empty edge.
  iree_hal_topology_edge_t edge =
      iree_hal_device_topology_query_edge(&info_a, &info_b);
  EXPECT_TRUE(iree_hal_topology_edge_is_empty(edge));
  iree_hal_topology_destroy(topology_b, iree_allocator_system());
  iree_hal_topology_destroy(topology_a, iree_allocator_system());
}

// Tests that iree_hal_device_topology_query_edge returns empty when the
// topology pointer is NULL (standalone device).
TEST(TopologyInfo, QueryEdgeStandaloneDevice) {
  iree_hal_device_topology_info_t info_standalone = {0};
  info_standalone.topology = NULL;

  iree_hal_topology_t* topology = CreateSingleDeviceTopology();
  iree_hal_device_topology_info_t info_grouped = {0};
  info_grouped.topology = topology;

  // NULL topology: should return empty edge.
  iree_hal_topology_edge_t edge =
      iree_hal_device_topology_query_edge(&info_standalone, &info_grouped);
  EXPECT_TRUE(iree_hal_topology_edge_is_empty(edge));

  // Both NULL: should return empty edge.
  iree_hal_device_topology_info_t info_standalone2 = {0};
  edge =
      iree_hal_device_topology_query_edge(&info_standalone, &info_standalone2);
  EXPECT_TRUE(iree_hal_topology_edge_is_empty(edge));
  iree_hal_topology_destroy(topology, iree_allocator_system());
}

}  // namespace
}  // namespace iree::hal
