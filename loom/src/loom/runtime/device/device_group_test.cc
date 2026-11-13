// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/runtime/device/device.h"
#include "loom/runtime/testing/device.h"

namespace {

class DeviceGroupTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    IREE_ASSERT_OK(
        loom_runtime_host_create(1, nullptr, iree_allocator_system(), &host_));
    IREE_ASSERT_OK(loom_runtime_test_driver_acquire(host_, &driver_));
  }

  static void TearDownTestSuite() {
    iree_hal_driver_release(driver_);
    loom_runtime_host_release(host_);
    driver_ = nullptr;
    host_ = nullptr;
  }

  void TearDown() override {
    for (auto* execution : executions_) {
      if (execution) {
        IREE_EXPECT_OK(loom_runtime_execution_drain(execution));
      }
      loom_runtime_execution_release(execution);
    }
    for (auto* buffer : fixed_buffers_) {
      iree_hal_buffer_release(buffer);
    }
    for (auto* reservation : reservations_) {
      IREE_EXPECT_OK(loom_runtime_virtual_buffer_free(reservation));
    }
    IREE_EXPECT_OK(loom_runtime_device_group_free(group_));
  }

  // Physical granularity, independent of each device's admission budget.
  static constexpr uint64_t kSlabBytes = 2 * 1024 * 1024;
  // Shared workers and asynchronous I/O outliving all device resources.
  inline static loom_runtime_host_t* host_ = nullptr;
  // Exact physical driver supplied by the target-specific test leaf.
  inline static iree_hal_driver_t* driver_ = nullptr;
  // One application group, including nullable partial construction.
  loom_runtime_device_group_t* group_ = nullptr;
  // Independent model-like timelines, one per device record.
  std::array<loom_runtime_execution_t*, 2> executions_ = {};
  // Fixed storage retained through accepted transfers, including assertion
  // exits.
  std::array<iree_hal_buffer_t*, 2> fixed_buffers_ = {};
  // Independent sparse roots that prevent premature group destruction.
  std::array<loom_runtime_virtual_buffer_t*, 2> reservations_ = {};
  // Accounting borrowed through each reservation's final destruction.
  std::array<loom_runtime_storage_statistics_t, 2> statistics_ = {};
  // Download destinations outliving accepted transfers on either device.
  std::array<uint32_t, 2> output_ = {};
};

TEST_F(DeviceGroupTest, LaterDeviceFailureReleasesEarlierHandle) {
  std::string invalid_uri(loom_runtime_test_device()->device_uri.data,
                          loom_runtime_test_device()->device_uri.size);
  invalid_uri.append(":?");
  for (size_t i = 0; i <= 128; ++i) {
    invalid_uri.append("x&");
  }
  const loom_runtime_device_options_t options[] = {
      {
          .driver = driver_,
          .uri = loom_runtime_test_device()->device_uri,
          .name = IREE_SV("first"),
      },
      {
          .driver = driver_,
          .uri = iree_make_string_view(invalid_uri.data(), invalid_uri.size()),
          .name = IREE_SV("second"),
      }};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_runtime_device_group_allocate(
                            host_, IREE_ARRAYSIZE(options), options,
                            iree_allocator_system(), &group_));
  EXPECT_EQ(group_, nullptr);
}

TEST_F(DeviceGroupTest, RejectedPoolPreservesEarlierPoolAndGroup) {
  const loom_runtime_device_options_t device_options = {
      .driver = driver_,
      .uri = loom_runtime_test_device()->device_uri,
  };
  IREE_ASSERT_OK(loom_runtime_device_group_allocate(
      host_, 1, &device_options, iree_allocator_system(), &group_));
  loom_runtime_pool_options_t options = {
      .elastic = true,
      .weight_slab_size = kSlabBytes,
      .state_slab_size = kSlabBytes,
  };
  loom_runtime_pool_t* pool = nullptr;
  IREE_ASSERT_OK(loom_runtime_device_group_create_pool(
      group_, {IREE_SV("first"), 1}, &options, &pool));
  for (const uint64_t invalid_size : {UINT64_MAX, 3 * kSlabBytes}) {
    options.weight_slab_size = invalid_size;
    options.state_slab_size = invalid_size;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        loom_runtime_device_group_create_pool(group_, {IREE_SV("invalid"), 1},
                                              &options, &pool));
    EXPECT_EQ(pool, nullptr);
    EXPECT_EQ(loom_runtime_device_group_pool_count(group_), 1u);
  }
}

TEST_F(DeviceGroupTest, IndependentSlabSizesPreserveShortTails) {
  constexpr uint64_t slab_size = 128 * 1024 * 1024;
  const loom_runtime_device_options_t options = {
      .driver = driver_,
      .uri = loom_runtime_test_device()->device_uri,
  };
  IREE_ASSERT_OK(loom_runtime_device_group_allocate(
      host_, 1, &options, iree_allocator_system(), &group_));
  const loom_runtime_pool_options_t pool_options = {
      .elastic = true,
      .weight_slab_size = slab_size,
      .state_slab_size = kSlabBytes,
  };
  loom_runtime_pool_t* storage_owner = nullptr;
  IREE_ASSERT_OK(loom_runtime_device_group_create_pool(
      group_, {IREE_SV("pool"), 1}, &pool_options, &storage_owner));
  const auto weights = loom_runtime_pool_weight_geometry(storage_owner);
  const auto state = loom_runtime_pool_state_geometry(storage_owner);
  EXPECT_EQ(weights.mapping_size, slab_size);
  EXPECT_EQ(state.mapping_size, state.minimum_size);
  EXPECT_EQ(state.backing_size, kSlabBytes);
  EXPECT_EQ(weights.granularity, state.granularity);
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      loom_runtime_pool_memory(storage_owner), weights, slab_size + 1, 256,
      &statistics_[0], &reservations_[0]));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      loom_runtime_pool_memory(storage_owner), state, 1, 256, &statistics_[1],
      &reservations_[1]));
  EXPECT_EQ(loom_runtime_virtual_buffer_slab_count(reservations_[0]), 2u);
  const auto demand =
      loom_runtime_virtual_buffer_missing_demand(reservations_[0]);
  const uint64_t missing_bytes =
      demand.packed_bytes + demand.weight_bytes + demand.dedicated_bytes;
  EXPECT_EQ(missing_bytes, slab_size + weights.minimum_size);
  EXPECT_EQ(statistics_[0].reserved_bytes, slab_size + weights.minimum_size);
  EXPECT_EQ(statistics_[1].reserved_bytes, state.minimum_size);
  EXPECT_EQ(statistics_[0].committed_bytes, 0u);
}

TEST_F(DeviceGroupTest, ExplicitPlacementPreservesNativeDomainThroughReuse) {
  const loom_runtime_device_options_t device_options = {
      .driver = driver_,
      .uri = loom_runtime_test_device()->device_uri,
  };
  IREE_ASSERT_OK(loom_runtime_device_group_allocate(
      host_, 1, &device_options, iree_allocator_system(), &group_));
  auto* device = loom_runtime_device_group_at(group_, 0);
  const auto* topology =
      iree_hal_device_group_topology(loom_runtime_device_group(device));
  uint32_t node = IREE_HAL_TOPOLOGY_NODE_ORDINAL_INVALID;
  for (uint32_t i = 0; i < iree_hal_topology_node_count(topology); ++i) {
    const auto* candidate = iree_hal_topology_node_at(topology, i);
    if (candidate->kind == IREE_HAL_TOPOLOGY_NODE_KIND_PHYSICAL_DEVICE &&
        candidate->device_ordinal == 0) {
      ASSERT_EQ(node, IREE_HAL_TOPOLOGY_NODE_ORDINAL_INVALID);
      node = candidate->ordinal;
    }
  }
  if (node == IREE_HAL_TOPOLOGY_NODE_ORDINAL_INVALID) {
    GTEST_SKIP() << "device has no physical allocation domain";
  }
  // Preferred placement qualifies the domain without allocating payload.
  // Automatic placement can favor shared memory even when LOCAL is available.
  const iree_hal_pool_family_access_t family = {
      .family =
          iree_hal_queue_family(loom_runtime_device_transfer_queue(device)),
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
  };
  const iree_hal_slab_pool_options_t qualification_options = {
      .placement = {IREE_HAL_POOL_PLACEMENT_PREFERRED, node},
  };
  iree_hal_pool_t* qualified = nullptr;
  IREE_ASSERT_OK(iree_hal_slab_pool_create(
      loom_runtime_device_group(device), {1, &family}, &qualification_options,
      iree_allocator_system(), &qualified));
  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(qualified, &capabilities);
  iree_hal_pool_release(qualified);
  if (capabilities.placement.mode != IREE_HAL_POOL_PLACEMENT_REQUIRED ||
      capabilities.placement.node != node) {
    GTEST_SKIP() << "native allocator cannot enforce physical-device placement";
  }
  loom_runtime_pool_options_t options = {
      .weight_slab_size = kSlabBytes,
      .state_slab_size = kSlabBytes,
      .placement = {LOOM_RUNTIME_POOL_PLACEMENT_DEVICE, 0},
  };
  for (uint32_t i = 0; i < 2; ++i) {
    options.elastic = i != 0;
    loom_runtime_pool_t* pool = nullptr;
    IREE_ASSERT_OK(loom_runtime_device_group_create_pool(
        group_, {i ? IREE_SV("elastic") : IREE_SV("fixed"), 1}, &options,
        &pool));
    iree_hal_pool_query_capabilities(loom_runtime_pool_storage(pool),
                                     &capabilities);
    EXPECT_EQ(capabilities.placement.mode, IREE_HAL_POOL_PLACEMENT_REQUIRED);
    EXPECT_EQ(capabilities.placement.node, node);
    IREE_ASSERT_OK(
        loom_runtime_device_create_execution(device, pool, &executions_[i]));
    if (options.elastic) {
      IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
          loom_runtime_pool_memory(pool),
          loom_runtime_pool_state_geometry(pool), sizeof(uint32_t), 256,
          &statistics_[i], &reservations_[i]));
    }
    for (uint32_t cycle = 0; cycle < 2; ++cycle) {
      iree_hal_buffer_t* buffer = nullptr;
      if (options.elastic) {
        IREE_ASSERT_OK(loom_runtime_virtual_buffer_commit(reservations_[i], 0,
                                                          sizeof(uint32_t)));
        buffer = loom_runtime_virtual_buffer_handle(reservations_[i]);
      } else {
        IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
            loom_runtime_pool_storage(pool),
            iree_hal_make_pool_reservation_request(4096, 0),
            iree_infinite_timeout(), &fixed_buffers_[i]));
        buffer = fixed_buffers_[i];
      }
      const uint32_t pattern = 0x12345678u + cycle;
      const iree_hal_transfer_operation_t operations[] = {
          {
              .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
              .fill =
                  {
                      .target_buffer = buffer,
                      .length = sizeof(pattern),
                      .pattern = &pattern,
                      .pattern_length = sizeof(pattern),
                  },
          },
          {
              .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
              .download =
                  {
                      .source_buffer = buffer,
                      .target = &output_[i],
                      .length = sizeof(output_[i]),
                  },
          },
      };
      uint64_t value = 0;
      // Transfer siblings are unordered. Chain the readback after the fill
      // through the execution timeline without an intermediate host wait.
      for (const auto& operation : operations) {
        IREE_ASSERT_OK(loom_runtime_execution_transfer(executions_[i], 1,
                                                       &operation, &value));
      }
      IREE_ASSERT_OK(loom_runtime_execution_drain(executions_[i]));
      EXPECT_EQ(output_[i], pattern);
      if (options.elastic) {
        loom_runtime_virtual_buffer_begin_trim(reservations_[i]);
        IREE_ASSERT_OK(loom_runtime_virtual_buffer_trim(reservations_[i]));
      } else {
        iree_hal_buffer_release(fixed_buffers_[i]);
        fixed_buffers_[i] = nullptr;
      }
      IREE_ASSERT_OK(loom_runtime_test_pool_trim_and_drain(pool, 0));
    }
  }
}

TEST_F(DeviceGroupTest, SeparateBudgetsAndWholeGroupLifetime) {
  const loom_runtime_device_options_t options[] = {
      {
          .driver = driver_,
          .uri = loom_runtime_test_device()->device_uri,
          .name = IREE_SV("first"),
      },
      {
          .driver = driver_,
          .uri = loom_runtime_test_device()->device_uri,
          .name = IREE_SV("second"),
      }};
  IREE_ASSERT_OK(loom_runtime_device_group_allocate(
      host_, IREE_ARRAYSIZE(options), options, iree_allocator_system(),
      &group_));
  auto* first = loom_runtime_device_group_at(group_, 0);
  auto* second = loom_runtime_device_group_at(group_, 1);
  loom_runtime_pool_t* pools[2] = {};
  for (iree_host_size_t i = 0; i < 2; ++i) {
    const loom_runtime_pool_options_t pool_options = {
        .elastic = true,
        .weight_slab_size = kSlabBytes,
        .state_slab_size = kSlabBytes,
        .memory_limit = (i + 1) * kSlabBytes,
    };
    IREE_ASSERT_OK(loom_runtime_device_group_create_pool(
        group_, {options[i].name, UINT64_C(1) << i}, &pool_options, &pools[i]));
  }
  EXPECT_EQ(loom_runtime_device_group(first),
            loom_runtime_device_group(second));
  EXPECT_NE(loom_runtime_pool_memory(pools[0]),
            loom_runtime_pool_memory(pools[1]));
  EXPECT_NE(loom_runtime_pool_residency_cache(pools[0]),
            loom_runtime_pool_residency_cache(pools[1]));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      loom_runtime_pool_memory(pools[1]),
      loom_runtime_pool_state_geometry(pools[1]), 2 * kSlabBytes, 256,
      &statistics_[1], &reservations_[1]));
  // Only the last record has a reservation. Rejection must preserve the first
  // record's resources too; it can still admit work after the failed teardown.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_runtime_device_group_free(group_));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      loom_runtime_pool_memory(pools[0]),
      loom_runtime_pool_state_geometry(pools[0]), 2 * kSlabBytes, 256,
      &statistics_[0], &reservations_[0]));
  // Fill the first physical state parent before testing the independent pool's
  // limit. Smaller mappings within that parent consume no additional native
  // capacity.
  IREE_ASSERT_OK(
      loom_runtime_virtual_buffer_commit(reservations_[0], 0, kSlabBytes));
  IREE_ASSERT_OK(
      loom_runtime_virtual_buffer_commit(reservations_[1], 0, 2 * kSlabBytes));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_runtime_virtual_buffer_commit(reservations_[0], kSlabBytes, 4));
  for (iree_host_size_t i = 0; i < 2; ++i) {
    auto* device = loom_runtime_device_group_at(group_, i);
    IREE_ASSERT_OK(loom_runtime_device_create_execution(device, pools[i],
                                                        &executions_[i]));
    auto* buffer = loom_runtime_virtual_buffer_handle(reservations_[i]);
    const uint32_t pattern = 0x12345678u + i;
    const iree_hal_transfer_operation_t fill = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
        .fill =
            {
                .target_buffer = buffer,
                .length = sizeof(pattern),
                .pattern = &pattern,
                .pattern_length = sizeof(pattern),
            },
    };
    const iree_hal_transfer_operation_t download = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .download =
            {
                .source_buffer = buffer,
                .target = &output_[i],
                .length = sizeof(output_[i]),
            },
    };
    uint64_t value = 0;
    IREE_ASSERT_OK(
        loom_runtime_execution_transfer(executions_[i], 1, &fill, &value));
    IREE_ASSERT_OK(
        loom_runtime_execution_transfer(executions_[i], 1, &download, &value));
    IREE_ASSERT_OK(loom_runtime_execution_drain(executions_[i]));
    EXPECT_EQ(output_[i], pattern);
  }
  loom_runtime_virtual_buffer_begin_trim(reservations_[1]);
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_trim(reservations_[1]));
  EXPECT_EQ(loom_runtime_pool_statistics(pools[0]).retained.committed_bytes,
            kSlabBytes);
  IREE_ASSERT_OK(
      loom_runtime_memory_pool_await(loom_runtime_pool_memory(pools[1])));
  const auto detached = loom_runtime_pool_statistics(pools[1]).retained;
  EXPECT_EQ(detached.committed_bytes, 2 * kSlabBytes);
  EXPECT_EQ(detached.cached_bytes, 2 * kSlabBytes);
}

}  // namespace
